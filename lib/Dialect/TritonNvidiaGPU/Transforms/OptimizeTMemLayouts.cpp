#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include <numeric>

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZETMEMLAYOUTSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// clang-format off
// Converts:
//  %l  = ttng.tmem_load  %o : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
//                               -> tensor<128x256xf32, #blocked>
//  %r  = tt.reshape %l  : tensor<128x256xf32, #blocked>
//                               -> tensor<128x2x128xf32, #blocked4>
//  %t  = tt.trans   %r  {order = array<i32: 0, 2, 1>}
//                               -> tensor<128x128x2xf32, #blocked5>
//  %lhs, %rhs = tt.split %t
//
// becomes
//  %o0   = ttng.tmem_subslice %o { N = 0   }
//  %lhs  = ttng.tmem_load     %o0
//  %o1   = ttng.tmem_subslice %o { N = 128 }
//  %rhs  = ttng.tmem_load     %o1
//
// and if %lhs / %rhs are split again through the same reshape->trans->split
// pattern, the transformation is can match again so that each further
// split is materialised as an independent `ttng.tmem_subslice` / `ttng.tmem_load`
// pair.  Consequently, a chain such as
//
//   acc0, acc1  = split(permute(reshape(acc , ...)))
//   acc00, acc01 = split(permute(reshape(acc0, ...)))
//   acc10, acc11 = split(permute(reshape(acc1, ...)))
//
// is lowered to four independent TMEM loads operating on four disjoint
// subslices.
//
// clang-format on
// Strip away all intermediate ttg.convert_layout ops to reach the true
// producer.
static Value stripConvertLayout(Value v) {
  while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

static ttg::MemDescType getSplitLoadRootMemDescType(Value memDesc) {
  auto memTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!memTy)
    return {};

  auto reinterpretOp = memDesc.getDefiningOp<ttg::MemDescReinterpretOp>();
  if (!reinterpretOp)
    return memTy;

  auto srcTy = dyn_cast<ttg::MemDescType>(reinterpretOp.getSrc().getType());
  if (!srcTy || srcTy.getRank() != memTy.getRank())
    return memTy;
  auto numElements = [](ArrayRef<int64_t> shape) {
    return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                           std::multiplies<int64_t>());
  };
  if (numElements(srcTy.getShape()) != numElements(memTy.getShape()))
    return memTy;
  if (!isTensorMemoryEncoding(srcTy.getEncoding()) ||
      !isTensorMemoryEncoding(memTy.getEncoding()))
    return memTy;
  return srcTy;
}

static bool shouldPreserveDirectLeadingSliceView(Value memDesc) {
  if (getTMemLdStQueryTypes(memDesc).size() <= 1)
    return false;

  auto memTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!memTy)
    return true;

  std::string error;
  auto maybeLayout = getTMemViewAnalysisLinearLayout(memTy.getShape(),
                                                     memTy.getEncoding(),
                                                     &error);
  if (!maybeLayout)
    return true;

  auto *ctx = memTy.getContext();
  auto kCol = StringAttr::get(ctx, "col");
  if (maybeLayout->hasInDim(kCol)) {
    for (unsigned i = 0, e = maybeLayout->getInDimSizeLog2(kCol); i < e; ++i) {
      if (llvm::all_of(maybeLayout->getBasis(kCol, i),
                       [](int32_t value) { return value == 0; })) {
        // Gapped-column leading-slice views still need the replay rewrite.
        // Direct ld/st collapses them to a contiguous tile and aliases the two
        // logical halves.
        return false;
      }
    }
  }

  return true;
}

enum class TMemTensorViewTransformKind { Reshape, Trans };

struct TMemTensorViewTransform {
  TMemTensorViewTransformKind kind;
  SmallVector<int64_t> srcShape;
  SmallVector<int64_t> dstShape;
  SmallVector<int32_t> order;
};

struct TMemLeadingSliceViewMatch {
  Value base;
  SmallVector<TMemTensorViewTransform> transforms;
  bool selectRHS;
};

static SmallVector<int32_t> invertPermutation(ArrayRef<int32_t> order) {
  SmallVector<int32_t> inverse(order.size());
  for (auto [idx, value] : llvm::enumerate(order))
    inverse[value] = idx;
  return inverse;
}

static SmallVector<int32_t> moveLeadingDimToBackOrder(unsigned rank) {
  SmallVector<int32_t> order;
  order.reserve(rank);
  for (unsigned i = 1; i < rank; ++i)
    order.push_back(i);
  order.push_back(0);
  return order;
}

static std::optional<TMemLeadingSliceViewMatch>
matchLeadingSliceView(Value memDesc) {
  auto indexOp = memDesc.getDefiningOp<ttg::MemDescIndexOp>();
  if (!indexOp)
    return std::nullopt;
  auto indexConst = indexOp.getIndex().getDefiningOp<arith::ConstantIntOp>();
  if (!indexConst || indexConst.value() != 0)
    return std::nullopt;

  auto subsliceOp = indexOp.getSrc().getDefiningOp<ttg::MemDescSubsliceOp>();
  if (!subsliceOp)
    return std::nullopt;

  auto subsliceTy = dyn_cast<ttg::MemDescType>(subsliceOp.getType());
  auto resultTy = dyn_cast<ttg::MemDescType>(indexOp.getType());
  auto subsliceSrcTy = dyn_cast<ttg::MemDescType>(subsliceOp.getSrc().getType());
  if (!subsliceTy || !resultTy || !subsliceSrcTy)
    return std::nullopt;
  if (subsliceSrcTy.getRank() < 1 || subsliceTy.getRank() != subsliceSrcTy.getRank() ||
      resultTy.getRank() + 1 != subsliceTy.getRank())
    return std::nullopt;
  if (subsliceSrcTy.getShape().front() != 2 || subsliceTy.getShape().front() != 1)
    return std::nullopt;
  if (subsliceOp.getOffsets().size() != static_cast<size_t>(subsliceSrcTy.getRank()))
    return std::nullopt;
  if (!llvm::equal(subsliceTy.getShape().drop_front(), resultTy.getShape()))
    return std::nullopt;
  auto offsets = subsliceOp.getOffsets();
  if ((offsets[0] != 0 && offsets[0] != 1) ||
      llvm::any_of(ArrayRef<int32_t>(offsets).drop_front(),
                   [](int32_t value) { return value != 0; }))
    return std::nullopt;

  SmallVector<TMemTensorViewTransform> reverseTransforms;
  Value cur = subsliceOp.getSrc();
  while (true) {
    if (auto reshapeOp = cur.getDefiningOp<ttg::MemDescReshapeOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(reshapeOp.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(reshapeOp.getType());
      if (!srcTy || !dstTy)
        return std::nullopt;
      reverseTransforms.push_back(TMemTensorViewTransform{
          TMemTensorViewTransformKind::Reshape,
          llvm::to_vector(srcTy.getShape()),
          llvm::to_vector(dstTy.getShape()),
          {}});
      cur = reshapeOp.getSrc();
      continue;
    }
    if (auto transOp = cur.getDefiningOp<ttg::MemDescTransOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(transOp.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(transOp.getType());
      if (!srcTy || !dstTy)
        return std::nullopt;
      reverseTransforms.push_back(TMemTensorViewTransform{
          TMemTensorViewTransformKind::Trans,
          llvm::to_vector(srcTy.getShape()),
          llvm::to_vector(dstTy.getShape()),
          llvm::to_vector(transOp.getOrder())});
      cur = transOp.getSrc();
      continue;
    }
    break;
  }

  auto baseTy = dyn_cast<ttg::MemDescType>(cur.getType());
  if (!baseTy)
    return std::nullopt;
  if (!isa<TensorMemorySpaceAttr>(baseTy.getMemorySpace()) ||
      !isTensorMemoryEncoding(baseTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(baseTy.getEncoding()))
    return std::nullopt;
  if (baseTy.getRank() != 2)
    return std::nullopt;
  if (reverseTransforms.empty())
    return std::nullopt;

  SmallVector<TMemTensorViewTransform> transforms(reverseTransforms.rbegin(),
                                                  reverseTransforms.rend());
  return TMemLeadingSliceViewMatch{
      cur,
      std::move(transforms),
      /*selectRHS=*/offsets[0] == 1,
  };
}

static std::optional<RankedTensorType>
getDirectSupportTMemTensorType(Value memDesc, int numWarps) {
  auto memTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!memTy)
    return std::nullopt;
  auto backingRowPlan = getBackingTMemLdStRowPlan(memDesc);
  auto queryTypes = getTMemLdStQueryTypes(memDesc);
  for (ttg::MemDescType queryTy : queryTypes) {
    auto layouts = nvidia_gpu::getTmemCompatibleLayouts(queryTy, numWarps);
    for (auto candidateLayout : layouts) {
      auto regTy = RankedTensorType::get(memTy.getShape(), memTy.getElementType(),
                                         candidateLayout);
      if (succeeded(computeTMemLdStEncodingInfo(
              regTy, queryTy, /*maxnreg=*/256, /*emitError=*/{},
              backingRowPlan))) {
        return regTy;
      }
    }
  }
  return std::nullopt;
}

static bool isRootedAtTMemPhysicalBitcast(Value memDesc) {
  SmallPtrSet<Value, 4> seen;
  Value cur = memDesc;
  while (cur && seen.insert(cur).second) {
    if (auto reinterpret = cur.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      if (reinterpret->hasAttr("tmem_physical_bitcast"))
        return true;
      cur = reinterpret.getSrc();
      continue;
    }
    if (auto op = cur.getDefiningOp<TMEMSubSliceOp>()) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = cur.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = cur.getDefiningOp<ttg::MemDescIndexOp>()) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = cur.getDefiningOp<ttg::MemDescReshapeOp>()) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = cur.getDefiningOp<ttg::MemDescTransOp>()) {
      cur = op.getSrc();
      continue;
    }
    if (auto forwarded = getTMemForwardingSource(cur)) {
      cur = forwarded;
      continue;
    }
    break;
  }
  return false;
}

static gpu::MemDescType getReplaySliceRegLayoutQueryType(Value memDesc) {
  gpu::MemDescType memTy = cast<gpu::MemDescType>(memDesc.getType());
  // Split replay materializes ordinary TMEM subview loads/stores.  Those
  // should use the shape-local subview type for register-layout selection; an
  // exact preserved physical query is only part of the explicit physical
  // bitcast contract.
  if (!isRootedAtTMemPhysicalBitcast(memDesc))
    return memTy;

  std::string layoutQueryError;
  if (auto maybeQueryTy =
          nvidia_gpu::inferStandaloneTMemRegLayoutQueryType(
              memDesc, &layoutQueryError);
      succeeded(maybeQueryTy)) {
    return *maybeQueryTy;
  }
  return memTy;
}

static RankedTensorType getLeadingSliceSplitFriendlyType(MLIRContext *ctx,
                                                         Type elementType,
                                                         ArrayRef<int64_t> shape,
                                                         int numWarps,
                                                         int threadsPerWarp,
                                                         int numCTAs) {
  SmallVector<unsigned> sizePerThread(shape.size(), 1);
  sizePerThread.back() = 2;
  SmallVector<unsigned> order(shape.size());
  std::iota(order.begin(), order.end(), 0);
  std::reverse(order.begin(), order.end());
  auto encoding = ttg::BlockedEncodingAttr::get(
      ctx, shape, sizePerThread, order, numWarps, threadsPerWarp, numCTAs);
  return RankedTensorType::get(shape, elementType, encoding);
}

static Value applyTensorViewTransforms(PatternRewriter &rewriter, Location loc,
                                       Value tensor,
                                       ArrayRef<TMemTensorViewTransform> transforms) {
  Value current = tensor;
  for (const TMemTensorViewTransform &transform : transforms) {
    if (transform.kind == TMemTensorViewTransformKind::Reshape) {
      current = ReshapeOp::create(rewriter, loc, transform.dstShape, current);
      continue;
    }
    current = TransOp::create(rewriter, loc, current, transform.order);
  }
  return current;
}

static Value applyInverseTensorViewTransforms(
    PatternRewriter &rewriter, Location loc, Value tensor,
    ArrayRef<TMemTensorViewTransform> transforms) {
  Value current = tensor;
  for (const TMemTensorViewTransform &transform : llvm::reverse(transforms)) {
    if (transform.kind == TMemTensorViewTransformKind::Reshape) {
      current = ReshapeOp::create(rewriter, loc, transform.srcShape, current);
      continue;
    }
    current =
        TransOp::create(rewriter, loc, current, invertPermutation(transform.order));
  }
  return current;
}

static Value lowerLeadingSliceViewLoad(PatternRewriter &rewriter, Location loc,
                                       const TMemLeadingSliceViewMatch &match,
                                       Type resultTy, int numWarps) {
  auto maybeSupportTy = getDirectSupportTMemTensorType(match.base, numWarps);
  assert(maybeSupportTy && "expected direct TMEM support type for leading slice");
  RankedTensorType supportTy = *maybeSupportTy;
  Value support = TMEMLoadOp::create(rewriter, loc, supportTy, match.base);
  Value transformed =
      applyTensorViewTransforms(rewriter, loc, support, match.transforms);
  auto transformedTy = cast<RankedTensorType>(transformed.getType());
  auto splitOrder = moveLeadingDimToBackOrder(transformedTy.getRank());
  Value transposed = TransOp::create(rewriter, loc, transformed, splitOrder);
  auto splitFriendlyTy = getLeadingSliceSplitFriendlyType(
      rewriter.getContext(), supportTy.getElementType(),
      cast<RankedTensorType>(transposed.getType()).getShape(), numWarps,
      ttg::lookupThreadsPerWarp(rewriter), ttg::lookupNumCTAs(rewriter));
  if (transposed.getType() != splitFriendlyTy) {
    transposed = ttg::ConvertLayoutOp::create(rewriter, loc, splitFriendlyTy,
                                              transposed);
  }
  auto split = SplitOp::create(rewriter, loc, transposed);
  Value selected = split.getResult(match.selectRHS ? 1 : 0);
  if (selected.getType() != resultTy)
    selected = ttg::ConvertLayoutOp::create(rewriter, loc, resultTy, selected);
  return selected;
}

static Value reshapeAndConvertToType(PatternRewriter &rewriter, Location loc,
                                     Value value, RankedTensorType targetTy) {
  Value current = value;
  auto currentTy = cast<RankedTensorType>(current.getType());
  if (!llvm::equal(currentTy.getShape(), targetTy.getShape())) {
    current = ReshapeOp::create(rewriter, loc, targetTy.getShape(), current);
    currentTy = cast<RankedTensorType>(current.getType());
  }
  if (currentTy != targetTy)
    current = ttg::ConvertLayoutOp::create(rewriter, loc, targetTy, current);
  return current;
}

static FailureOr<Value>
lowerTMemPhysicalSupportLoad(PatternRewriter &rewriter, TMEMLoadOp loadOp) {
  std::string error;
  auto standaloneMemTy = inferStandaloneTMemViewType(loadOp.getSrc(), &error);
  if (failed(standaloneMemTy))
    return failure();
  auto maybePlan = getTMemLdStPhysicalSupportPlan(
      *standaloneMemTy, ttg::lookupNumWarps(loadOp), getContextualMaxNReg(loadOp));
  if (!maybePlan)
    return failure();

  auto supportView = ttg::MemDescReinterpretOp::create(
      rewriter, loadOp.getLoc(), maybePlan->memTy, loadOp.getSrc());
  Value support = TMEMLoadOp::create(rewriter, loadOp.getLoc(),
                                     maybePlan->regTy, supportView);
  return reshapeAndConvertToType(
      rewriter, loadOp.getLoc(), support,
      cast<RankedTensorType>(loadOp.getType()));
}

static LogicalResult lowerTMemPhysicalSupportStore(PatternRewriter &rewriter,
                                                   TMEMStoreOp storeOp) {
  std::string error;
  auto standaloneMemTy = inferStandaloneTMemViewType(storeOp.getDst(), &error);
  if (failed(standaloneMemTy))
    return failure();
  auto maybePlan = getTMemLdStPhysicalSupportPlan(
      *standaloneMemTy, ttg::lookupNumWarps(storeOp),
      getContextualMaxNReg(storeOp));
  if (!maybePlan)
    return failure();

  Value supportValue = reshapeAndConvertToType(
      rewriter, storeOp.getLoc(), storeOp.getSrc(), maybePlan->regTy);
  auto supportView = ttg::MemDescReinterpretOp::create(
      rewriter, storeOp.getLoc(), maybePlan->memTy, storeOp.getDst());
  TMEMStoreOp::create(rewriter, storeOp.getLoc(), supportView, supportValue,
                      storeOp.getPred());
  rewriter.eraseOp(storeOp);
  return success();
}

class TMemSplitLoadPattern : public OpRewritePattern<SplitOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SplitOp splitOp,
                                PatternRewriter &rewriter) const override {
    // -----------------------------------------------------------------------
    // Match the pattern:
    //      splitOp
    //        ^  |
    //        |  +-- transOp(order = [0, 2, 1])
    //        |       ^  |
    //        |       |  +-- reshapeOp
    //        |       |        ^  |
    //        |       |        |  +-- (maybe convert_layout)
    //        |       |        +-- tmemLoad
    // -----------------------------------------------------------------------

    // Starting from the split source, peel off convert_layouts if any.
    Value src = stripConvertLayout(splitOp.getSrc());
    auto transOp = src.getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto reshapeOp = transOp.getSrc().getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();

    // Peel off convert_layouts *below* the reshape as well.  This is required
    // for the recursive case where the producer of the reshape is the result
    // of an earlier optimisation pass (i.e. a convert_layout of a previous
    // tmem_load).
    Value reshapeSrc = stripConvertLayout(reshapeOp.getSrc());
    auto tmemLoad = reshapeSrc.getDefiningOp<TMEMLoadOp>();
    if (!tmemLoad)
      return failure();
    if (matchLeadingSliceView(tmemLoad.getSrc()))
      return failure();

    auto shape = reshapeOp.getResult().getType().getShape();
    auto rootMemTy = getSplitLoadRootMemDescType(tmemLoad.getSrc());
    if (!rootMemTy)
      return failure();
    // The split-load replay is currently only semantics-preserving on 32-bit
    // TMEM tiles. Sub-32-bit Blackwell epilogues must keep the original
    // full-load path until the replayed pack::16b decomposition is fixed.
    if (rootMemTy.getElementTypeBitWidth() < 32)
      return failure();
    // Ensure M dimension is preserved by the logical TMEM tile, even if the
    // current source is a reinterpret of the backing TMEM allocation.
    if (shape[0] != rootMemTy.getShape()[rootMemTy.getRank() - 2])
      return failure();
    int mDim = getShapePerCTA(rootMemTy)[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    // Create the two TMEM subslices and their corresponding loads.
    Value tmem = tmemLoad.getSrc(); // Could itself be a subslice.
    int numWarps = ttg::lookupNumWarps(tmemLoad);
    rewriter.setInsertionPoint(tmemLoad);

    auto createSliceLoad =
        [&](int64_t nOffset) -> std::pair<TMEMLoadOp, ttg::ConvertLayoutOp> {
      // Generate the subslice op.
      Value subSlice = TMEMSubSliceOp::create(rewriter, tmemLoad.getLoc(), tmem,
                                              nOffset, splitNSize);

      // Choose a layout compatible with the slice size.
      gpu::MemDescType layoutQueryTy =
          getReplaySliceRegLayoutQueryType(subSlice);
      auto distLayout =
          nvidia_gpu::getDefaultLayoutForTmemLdSt(layoutQueryTy, numWarps);

      RankedTensorType newLoadType =
          splitOp.getOutLHS().getType().cloneWithEncoding(distLayout);

      // Generate the load and convert_layout back to the original layout.
      auto load = TMEMLoadOp::create(rewriter, tmemLoad.getLoc(), newLoadType,
                                     subSlice);
      auto cvt = ttg::ConvertLayoutOp::create(
          rewriter, tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load);

      return {load, cvt};
    };

    auto [load0, cvt0] = createSliceLoad(/*nOffset=*/0);
    auto [load1, cvt1] = createSliceLoad(/*nOffset=*/splitNSize);
    rewriter.replaceOp(splitOp, {cvt0, cvt1});
    return success();
  }
};

class TMemLeadingSliceLoadPattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemLeadingSliceLoadPattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(TMEMLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (shouldPreserveDirectLeadingSliceView(loadOp.getSrc()))
      return failure();

    if (FailureOr<Value> support =
            lowerTMemPhysicalSupportLoad(rewriter, loadOp);
        succeeded(support)) {
      rewriter.replaceOp(loadOp, *support);
      return success();
    }

    auto match = matchLeadingSliceView(loadOp.getSrc());
    if (!match)
      return failure();

    rewriter.setInsertionPoint(loadOp);
    Value replacement = lowerLeadingSliceViewLoad(
        rewriter, loadOp.getLoc(), *match,
        loadOp.getType(), ttg::lookupNumWarps(loadOp));
    rewriter.replaceOp(loadOp, replacement);
    return success();
  }
};

class TMemStoreJoinPattern : public OpRewritePattern<TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMStoreOp storeOp,
                                PatternRewriter &b) const override {
    // Look through layout conversions.
    Value src = storeOp.getSrc();
    while (auto cvt = src.getDefiningOp<ttg::ConvertLayoutOp>()) {
      src = cvt.getSrc();
    }

    // Only support joinin N dimension on the outer most.
    auto reshapeOp = src.getDefiningOp<ReshapeOp>();
    if (!reshapeOp)
      return failure();
    auto shape = reshapeOp.getSrc().getType().getShape();
    if (reshapeOp.getType().getShape().front() != shape[0])
      return failure();
    auto transOp = reshapeOp.getSrc().getDefiningOp<TransOp>();
    if (!transOp || transOp.getOrder() != ArrayRef<int>({0, 2, 1}))
      return failure();
    auto joinOp = transOp.getSrc().getDefiningOp<JoinOp>();
    if (!joinOp)
      return failure();

    // We found a tmem_store that is joined on the N dimension. We can split it
    // into multiple tmem_stores.
    int mDim = getShapePerCTA(storeOp.getDst().getType())[0];
    // TODO: enable other M cases. (the layout is a bit more complex).
    if (mDim != 128)
      return failure();
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    Location loc = storeOp.getLoc();
    Value tmem = storeOp.getDst();
    if (matchLeadingSliceView(tmem))
      return failure();
    int numWarps = ttg::lookupNumWarps(storeOp);
    Value truePred = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));

    auto *ctx = joinOp.getContext();

    auto createSlice = [&](TypedValue<RankedTensorType> input, int offset) {
      auto subSlice = TMEMSubSliceOp::create(b, loc, tmem, offset, splitNSize);
      gpu::MemDescType layoutQueryTy =
          getReplaySliceRegLayoutQueryType(subSlice);
      auto distLayout =
          nvidia_gpu::getDefaultLayoutForTmemLdSt(layoutQueryTy, numWarps);
      auto newType = input.getType().cloneWithEncoding(distLayout);
      auto cvt = ttg::ConvertLayoutOp::create(b, loc, newType, input);
      auto store =
          TMEMStoreOp::create(b, loc, subSlice, cvt.getResult(), truePred);
      return store;
    };

    auto store0 = createSlice(joinOp.getLhs(), 0);
    auto store1 = createSlice(joinOp.getRhs(), splitNSize);
    b.eraseOp(storeOp);
    return success();
  }
};

class TMemLeadingSliceStorePattern : public OpRewritePattern<TMEMStoreOp> {
public:
  TMemLeadingSliceStorePattern(MLIRContext *context)
      : OpRewritePattern<TMEMStoreOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(TMEMStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    if (shouldPreserveDirectLeadingSliceView(storeOp.getDst()))
      return failure();

    if (succeeded(lowerTMemPhysicalSupportStore(rewriter, storeOp)))
      return success();

    if (!matchPattern(storeOp.getPred(), m_One()))
      return failure();

    auto match = matchLeadingSliceView(storeOp.getDst());
    if (!match)
      return failure();

    int numWarps = ttg::lookupNumWarps(storeOp);
    auto maybeSupportTy = getDirectSupportTMemTensorType(match->base, numWarps);
    if (!maybeSupportTy)
      return failure();
    RankedTensorType supportTy = *maybeSupportTy;

    rewriter.setInsertionPoint(storeOp);
    Value support = TMEMLoadOp::create(rewriter, storeOp.getLoc(), supportTy,
                                       match->base);
    Value transformed = applyTensorViewTransforms(rewriter, storeOp.getLoc(),
                                                  support, match->transforms);
    auto transformedTy = cast<RankedTensorType>(transformed.getType());
    auto splitOrder = moveLeadingDimToBackOrder(transformedTy.getRank());
    Value transposed =
        TransOp::create(rewriter, storeOp.getLoc(), transformed, splitOrder);
    auto splitFriendlyTy = getLeadingSliceSplitFriendlyType(
        rewriter.getContext(), supportTy.getElementType(),
        cast<RankedTensorType>(transposed.getType()).getShape(), numWarps,
        ttg::lookupThreadsPerWarp(rewriter), ttg::lookupNumCTAs(rewriter));
    if (transposed.getType() != splitFriendlyTy) {
      transposed = ttg::ConvertLayoutOp::create(rewriter, storeOp.getLoc(),
                                                splitFriendlyTy, transposed);
    }
    auto split = SplitOp::create(rewriter, storeOp.getLoc(), transposed);
    Value lhs = split.getResult(0);
    Value rhs = split.getResult(1);

    Value replacement = storeOp.getSrc();
    Value selected = match->selectRHS ? rhs : lhs;
    if (replacement.getType() != selected.getType())
      replacement = ttg::ConvertLayoutOp::create(
          rewriter, storeOp.getLoc(), selected.getType(), replacement);

    Value newLhs = match->selectRHS ? lhs : replacement;
    Value newRhs = match->selectRHS ? replacement : rhs;
    Value joined =
        JoinOp::create(rewriter, storeOp.getLoc(), newLhs, newRhs);
    Value transposedBack = TransOp::create(
        rewriter, storeOp.getLoc(), joined, invertPermutation(splitOrder));
    Value supportReplacement = applyInverseTensorViewTransforms(
        rewriter, storeOp.getLoc(), transposedBack, match->transforms);
    if (supportReplacement.getType() != supportTy)
      supportReplacement = ttg::ConvertLayoutOp::create(
          rewriter, storeOp.getLoc(), supportTy, supportReplacement);

    TMEMStoreOp::create(rewriter, storeOp.getLoc(), match->base,
                        supportReplacement, storeOp.getPred());
    rewriter.eraseOp(storeOp);
    return success();
  }
};

// Pick an optimized tmem load layout based on its users. When there are
// multiple warpgroups tmem_load results can be distirbuted along M or N across
// the warpgroups. By default distribute along N but when there is a reduction
// along N dimension we want to distribute along M instead to avoid having to
// reduce across warps.
class TMemLoadReducePattern : public OpRewritePattern<TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMLoadOp tmemLoadOp,
                                PatternRewriter &rewriter) const override {
    int numWarps = ttg::lookupNumWarps(tmemLoadOp);
    // If there is only 1 warpgroup there is nothing to optimize as the layout
    // is already reduction friendly.
    if (numWarps != 8)
      return failure();
    bool foundReductionAlongN = false;
    bool storesBackToTMem = false;
    auto filter = [&](Operation *op) {
      if (isa<TMEMStoreOp>(op)) {
        storesBackToTMem = true;
        return false;
      }
      if (isa<ttg::ConvertLayoutOp>(op) || op->hasTrait<OpTrait::Elementwise>())
        return true;
      if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
        foundReductionAlongN = reduce.getAxis() == 1;
      }
      return false;
    };
    ForwardSliceOptions fwdOpt;
    fwdOpt.filter = filter;
    SetVector<mlir::Operation *> fwdSlices;
    getForwardSlice(tmemLoadOp.getResult(), &fwdSlices, fwdOpt);
    if (!foundReductionAlongN || storesBackToTMem)
      return failure();
    // Try to split along M dimension but follow the restrictions of TMEM:
    // warp0 get M = 0, warp 1 gets M = 32, warp 2 gets M = 64, warp 3 gets
    // M = 96 warp 4 gets M = 16, warp 5 gets M = 48, warp 6 gets M = 80,
    // warp 7 gets M = 112
    RankedTensorType oldType = tmemLoadOp.getType();
    std::optional<gpu::DistributedEncodingTrait> newLayout =
        getTmemLoadLayoutSplitLongM(oldType, tmemLoadOp.getSrc().getType(),
                                    numWarps);
    if (!newLayout)
      return failure();
    if (newLayout.value() == oldType.getEncoding())
      return failure();

    auto newType = oldType.cloneWithEncoding(newLayout.value());
    tmemLoadOp.getResult().setType(newType);
    OpBuilder builder(tmemLoadOp);
    builder.setInsertionPointAfter(tmemLoadOp);
    auto cvt = ttg::ConvertLayoutOp::create(builder, tmemLoadOp.getLoc(),
                                            oldType, tmemLoadOp.getResult());
    tmemLoadOp.getResult().replaceAllUsesExcept(cvt.getResult(), cvt);
    return success();
  }
};

// Optimize local_load -> tmem_store when the layout 16x256b allows better
// code generation for local_load lowering.
class TMemFromSharedMemPattern : public OpRewritePattern<TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMStoreOp tmemStoreOp,
                                PatternRewriter &rewriter) const override {
    auto tmemEnc = tmemStoreOp.getDst().getType().getEncoding();
    if (!triton::nvidia_gpu::isTensorMemoryEncoding(tmemEnc) ||
        isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(tmemEnc))
      return failure();
    auto isLocalLoadLike = [&](Value value) {
      while (Operation *def = value.getDefiningOp()) {
        if (isa<gpu::LocalLoadOp>(def))
          return true;
        if (isa<ttg::ConvertLayoutOp, triton::ReshapeOp, triton::TransOp,
                triton::ExpandDimsOp>(def)) {
          value = def->getOperand(0);
          continue;
        }
        return false;
      }
      return false;
    };
    if (!isLocalLoadLike(tmemStoreOp.getSrc()))
      return failure();
    int numWarps = ttg::lookupNumWarps(tmemStoreOp);
    // Compute the alternative layout.
    std::optional<LinearLayout> ll =
        nvidia_gpu::getDistributedLayoutForTmemLdSt(
            tmemStoreOp.getDst().getType(), TMemAccessAtom::I16x256b, numWarps);
    if (!ll)
      return failure();
    Attribute newEncoding =
        gpu::LinearEncodingAttr::get(tmemStoreOp.getContext(), std::move(*ll));
    auto oldType = tmemStoreOp.getSrc().getType();
    if (isa_and_nonnull<ttg::ConvertLayoutOp>(
            tmemStoreOp.getSrc().getDefiningOp()))
      return failure();
    auto memType = cast<gpu::MemDescType>(tmemStoreOp.getDst().getType());
    if (!nvidia_gpu::isDistributedLayoutTMemCompatible(tmemStoreOp, oldType,
                                                       memType))
      return failure();
    auto newType = oldType.cloneWithEncoding(newEncoding);
    if (newType == oldType)
      return failure();

    SetVector<Value> slice;
    DenseMap<Value, Attribute> layoutMap;
    // Check how it may propagate up the SSA chain.
    LogicalResult result = getConvertBackwardSlice(
        tmemStoreOp.getSrcMutable(), slice, newEncoding, layoutMap);
    if (result.failed())
      return failure();
    bool foundImprovedLoad = false;
    for (Value v : slice) {
      auto localLoad = v.getDefiningOp<gpu::LocalLoadOp>();
      if (!localLoad)
        continue;
      // 16x256b is optimized for 16bits load.
      if (localLoad.getType().getElementType().getIntOrFloatBitWidth() != 16)
        return failure();
      LinearLayout regLayout = gpu::toLinearLayout(localLoad.getType());
      LinearLayout smemLayout =
          gpu::toLinearLayout(localLoad.getSrc().getType());
      int vecDim =
          regLayout.invertAndCompose(smemLayout).getNumConsecutiveInOut();
      // If we find a 16bits load that cannot be vectorized use the alternative
      // layout.
      if (vecDim != 1)
        return failure();
      foundImprovedLoad = true;
    }
    if (!foundImprovedLoad)
      return failure();
    // Use the new layout and rely on RemoveLayoutConversions pass to propagate
    // the convert_layout.
    auto cvt = ttg::ConvertLayoutOp::create(rewriter, tmemStoreOp.getLoc(),
                                            newType, tmemStoreOp.getSrc());
    rewriter.modifyOpInPlace(tmemStoreOp, [&]() {
      tmemStoreOp.getSrcMutable().assign(cvt.getResult());
    });
    return success();
  }
};

// Optimize tmem_load -> local_store when the layout 16x256b allows better
// code generation for local_store lowering.
class TMemToSharedMemPattern : public OpRewritePattern<TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMLoadOp tmemLoadOp,
                                PatternRewriter &rewriter) const override {
    auto tmemEnc = tmemLoadOp.getSrc().getType().getEncoding();
    if (!triton::nvidia_gpu::isTensorMemoryEncoding(tmemEnc) ||
        isa<triton::nvidia_gpu::TensorMemoryScalesEncodingAttr>(tmemEnc))
      return failure();
    int numWarps = ttg::lookupNumWarps(tmemLoadOp);
    auto oldType = tmemLoadOp.getType();
    auto memType = cast<gpu::MemDescType>(tmemLoadOp.getSrc().getType());
    if (!nvidia_gpu::isDistributedLayoutTMemCompatible(tmemLoadOp, oldType,
                                                       memType))
      return failure();
    // Compute the alternative layout.
    auto ll = nvidia_gpu::getDistributedLayoutForTmemLdSt(
        memType, TMemAccessAtom::I16x256b, numWarps);
    if (!ll)
      return failure();
    Attribute newEncoding =
        gpu::LinearEncodingAttr::get(tmemLoadOp.getContext(), std::move(*ll));
    auto newType = oldType.cloneWithEncoding(newEncoding);
    if (newType == oldType)
      return failure();

    SetVector<Value> slice;
    DenseMap<Value, Attribute> layoutMap;
    SmallVector<std::pair<Value, Attribute>> uses;
    uses.push_back({tmemLoadOp.getResult(), newEncoding});
    bool foundImprovedStore = false;
    llvm::DenseSet<std::pair<Value, Attribute>> visited;
    while (!uses.empty()) {
      auto [v, encoding] = uses.pop_back_val();
      if (!visited.insert({v, encoding}).second)
        continue;
      for (auto user : v.getUsers()) {
        if (auto localStore = dyn_cast<gpu::LocalStoreOp>(user)) {
          // Check if the store benefits from the new layout.
          // 16x256b is optimized for 16bits load.
          auto srcType = localStore.getSrc().getType();
          if (srcType.getElementType().getIntOrFloatBitWidth() >= 32)
            continue;
          LinearLayout regLayout = gpu::toLinearLayout(srcType);
          LinearLayout smemLayout =
              gpu::toLinearLayout(localStore.getDst().getType());
          int vecDim =
              regLayout.invertAndCompose(smemLayout).getNumConsecutiveInOut();
          // If we find a 8 or 16bits store that cannot be vectorized use the
          // alternative layout.
          // TODO: we could refine the logic to make sure the new layout would
          // help by allowing stmatrix if we can isolate good helpers.
          if (vecDim != 1)
            continue;
          foundImprovedStore = true;
          break;
        }
        // Don't iterate though control flow ops.
        if (isa<RegionBranchOpInterface, scf::YieldOp, BranchOpInterface>(user))
          continue;
        Attribute userEncoding = inferDstEncoding(user, encoding);
        if (!userEncoding) {
          if (isa<ttg::ConvertLayoutOp>(user)) {
            userEncoding = encoding;
          } else {
            continue;
          }
        }
        for (auto result : user->getResults()) {
          uses.push_back({result, userEncoding});
        }
      }
    }
    if (!foundImprovedStore)
      return failure();
    // Use the new layout and rely on RemoveLayoutConversions pass to propagate
    // the convert_layout.
    rewriter.modifyOpInPlace(
        tmemLoadOp, [&]() { tmemLoadOp.getResult().setType(newType); });
    rewriter.setInsertionPointAfter(tmemLoadOp);
    auto cvt = ttg::ConvertLayoutOp::create(rewriter, tmemLoadOp.getLoc(),
                                            oldType, tmemLoadOp.getResult());
    rewriter.replaceAllUsesExcept(tmemLoadOp.getResult(), cvt, cvt);
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUOptimizeTMemLayoutsPass
    : public impl::TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
          TritonNvidiaGPUOptimizeTMemLayoutsPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeTMemLayoutsPassBase<
      TritonNvidiaGPUOptimizeTMemLayoutsPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns
        .add<TMemSplitLoadPattern, TMemLeadingSliceLoadPattern,
             TMemStoreJoinPattern, TMemLeadingSliceStorePattern,
             TMemLoadReducePattern, TMemFromSharedMemPattern,
             TMemToSharedMemPattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

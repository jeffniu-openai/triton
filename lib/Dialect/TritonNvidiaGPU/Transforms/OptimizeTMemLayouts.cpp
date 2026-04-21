#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

enum class TMemReplayHalfSliceStepKind { Reshape, Trans, HalfSlice };

struct TMemReplayHalfSliceStep {
  TMemReplayHalfSliceStepKind kind;
  SmallVector<int64_t> srcShape;
  SmallVector<int64_t> dstShape;
  SmallVector<int32_t> order;
  unsigned dim = 0;
  bool selectRHS = false;
};

struct TMemReplayHalfSliceViewMatch {
  Value base;
  SmallVector<TMemReplayHalfSliceStep> steps;
};

struct TMemReplayFullViewMatch {
  Value base;
  SmallVector<TMemTensorViewTransform> transforms;
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

static std::optional<std::pair<unsigned, bool>>
matchReplayableHalfSlice(ttg::MemDescSubsliceOp subslice) {
  auto dstTy = dyn_cast<ttg::MemDescType>(subslice.getType());
  auto dim = getTMemLdStReplayableHalfSliceDim(subslice);
  if (!dstTy || !dim)
    return std::nullopt;
  return std::make_pair(*dim, subslice.getOffsets()[*dim] ==
                                  dstTy.getShape()[*dim]);
}

static std::optional<TMemReplayHalfSliceViewMatch>
matchReplayableHalfSliceView(Value memDesc) {
  if (!isTMemLdStReplayableHalfSliceView(memDesc))
    return std::nullopt;
  if (!isUnsupportedDirectTMemLdStDescriptorView(memDesc, /*error=*/nullptr))
    return std::nullopt;

  if (auto leading = matchLeadingSliceView(memDesc)) {
    SmallVector<TMemReplayHalfSliceStep> steps;
    steps.reserve(leading->transforms.size() + 2);
    for (const TMemTensorViewTransform &transform : leading->transforms) {
      steps.push_back(TMemReplayHalfSliceStep{
          transform.kind == TMemTensorViewTransformKind::Reshape
              ? TMemReplayHalfSliceStepKind::Reshape
              : TMemReplayHalfSliceStepKind::Trans,
          transform.srcShape,
          transform.dstShape,
          transform.order,
          0,
          false});
    }
    if (steps.empty())
      return std::nullopt;
    SmallVector<int64_t> sliceShape = steps.back().dstShape;
    sliceShape.front() = 1;
    auto resultShape =
        llvm::to_vector(cast<ttg::MemDescType>(memDesc.getType()).getShape());
    steps.push_back(TMemReplayHalfSliceStep{
        TMemReplayHalfSliceStepKind::HalfSlice,
        steps.back().dstShape,
        sliceShape,
        {},
        0,
        leading->selectRHS});
    steps.push_back(TMemReplayHalfSliceStep{
        TMemReplayHalfSliceStepKind::Reshape,
        sliceShape,
        resultShape,
        {},
        0,
        false});
    return TMemReplayHalfSliceViewMatch{leading->base, std::move(steps)};
  }

  SmallVector<TMemReplayHalfSliceStep> reverseSteps;
  Value cur = memDesc;
  while (true) {
    if (auto reshapeOp = cur.getDefiningOp<ttg::MemDescReshapeOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(reshapeOp.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(reshapeOp.getType());
      if (!srcTy || !dstTy)
        return std::nullopt;
      reverseSteps.push_back(TMemReplayHalfSliceStep{
          TMemReplayHalfSliceStepKind::Reshape,
          llvm::to_vector(srcTy.getShape()),
          llvm::to_vector(dstTy.getShape()),
          {},
          0,
          false});
      cur = reshapeOp.getSrc();
      continue;
    }
    if (auto transOp = cur.getDefiningOp<ttg::MemDescTransOp>()) {
      auto srcTy = dyn_cast<ttg::MemDescType>(transOp.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(transOp.getType());
      if (!srcTy || !dstTy)
        return std::nullopt;
      reverseSteps.push_back(TMemReplayHalfSliceStep{
          TMemReplayHalfSliceStepKind::Trans,
          llvm::to_vector(srcTy.getShape()),
          llvm::to_vector(dstTy.getShape()),
          llvm::to_vector(transOp.getOrder()),
          0,
          false});
      cur = transOp.getSrc();
      continue;
    }
    if (auto subsliceOp = cur.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      auto halfSlice = matchReplayableHalfSlice(subsliceOp);
      if (!halfSlice)
        return std::nullopt;
      auto srcTy = dyn_cast<ttg::MemDescType>(subsliceOp.getSrc().getType());
      auto dstTy = dyn_cast<ttg::MemDescType>(subsliceOp.getType());
      reverseSteps.push_back(TMemReplayHalfSliceStep{
          TMemReplayHalfSliceStepKind::HalfSlice,
          llvm::to_vector(srcTy.getShape()),
          llvm::to_vector(dstTy.getShape()),
          {},
          halfSlice->first,
          halfSlice->second});
      cur = subsliceOp.getSrc();
      continue;
    }
    break;
  }

  auto baseTy = dyn_cast<ttg::MemDescType>(cur.getType());
  if (!baseTy || baseTy.getRank() != 2 ||
      !isa<TensorMemorySpaceAttr>(baseTy.getMemorySpace()) ||
      !isTensorMemoryEncoding(baseTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(baseTy.getEncoding()) ||
      reverseSteps.empty())
    return std::nullopt;

  SmallVector<TMemReplayHalfSliceStep> steps(reverseSteps.rbegin(),
                                             reverseSteps.rend());
  if (steps.size() == 1 &&
      steps.front().kind == TMemReplayHalfSliceStepKind::HalfSlice &&
      steps.front().dim == 0 && gpu::getNumCTAs(baseTy.getEncoding()) != 1 &&
      steps.front().srcShape.size() == 2) {
    const TMemReplayHalfSliceStep halfSlice = steps.front();
    int64_t rows = halfSlice.srcShape[0];
    int64_t cols = halfSlice.srcShape[1];
    if (rows <= 1 || rows % 2 != 0)
      return std::nullopt;
    SmallVector<int64_t> leadingShape{2, rows / 2, cols};
    SmallVector<int64_t> unitLeadingShape{1, rows / 2, cols};
    steps.clear();
    steps.push_back(TMemReplayHalfSliceStep{
        TMemReplayHalfSliceStepKind::Reshape,
        halfSlice.srcShape,
        leadingShape,
        {},
        0,
        false});
    steps.push_back(TMemReplayHalfSliceStep{
        TMemReplayHalfSliceStepKind::HalfSlice,
        leadingShape,
        unitLeadingShape,
        {},
        0,
        halfSlice.selectRHS});
    steps.push_back(TMemReplayHalfSliceStep{
        TMemReplayHalfSliceStepKind::Reshape,
        unitLeadingShape,
        halfSlice.dstShape,
        {},
        0,
        false});
  }
  return TMemReplayHalfSliceViewMatch{cur, std::move(steps)};
}

static std::optional<TMemReplayFullViewMatch>
matchReplayableFullView(Value memDesc) {
  if (!isTMemLdStReplayableFullView(memDesc))
    return std::nullopt;

  SmallVector<TMemTensorViewTransform> reverseTransforms;
  Value cur = memDesc;
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
  if (reverseTransforms.empty())
    return std::nullopt;

  SmallVector<TMemTensorViewTransform> transforms(reverseTransforms.rbegin(),
                                                  reverseTransforms.rend());
  return TMemReplayFullViewMatch{cur, std::move(transforms)};
}

static gpu::MemDescType getTMemSubSliceType(Value alloc, int offset, int size) {
  auto allocTy = cast<gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocTy.getShape());
  shape.back() = size;
  SmallVector<int32_t> offsets(shape.size(), 0);
  offsets.back() = offset;
  auto maybeEncoding = inferTMemSubsliceEncoding(
      allocTy.getShape(), allocTy.getEncoding(), shape, offsets);
  Attribute encoding = succeeded(maybeEncoding) ? Attribute(*maybeEncoding)
                                                : allocTy.getEncoding();
  return gpu::MemDescType::get(shape, allocTy.getElementType(), encoding,
                               allocTy.getMemorySpace(),
                               allocTy.getMutableMemory(),
                               allocTy.getAllocShape());
}

static std::optional<gpu::DistributedEncodingTrait>
getReplaySliceRegLayout(gpu::MemDescType layoutQueryTy, int numWarps) {
  auto layouts = getTmemCompatibleLayouts(layoutQueryTy, numWarps);
  if (layouts.empty())
    return std::nullopt;
  return layouts.front();
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

static bool hasReductionAlongNUse(Value value) {
  bool foundReductionAlongN = false;
  auto filter = [&](Operation *op) {
    if (isa<ttg::ConvertLayoutOp>(op) || op->hasTrait<OpTrait::Elementwise>())
      return true;
    if (auto reduce = dyn_cast<triton::ReduceOp>(op))
      foundReductionAlongN |= reduce.getAxis() == 1;
    return false;
  };
  ForwardSliceOptions fwdOpt;
  fwdOpt.filter = filter;
  SetVector<Operation *> fwdSlices;
  getForwardSlice(value, &fwdSlices, fwdOpt);
  return foundReductionAlongN;
}

static bool areEquivalentReplayBaseValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  auto getConstantIndexKey =
      [](Value value) -> std::optional<std::pair<Value, int64_t>> {
    auto indexOp = value.getDefiningOp<ttg::MemDescIndexOp>();
    if (!indexOp)
      return std::nullopt;

    APInt indexValue;
    if (!matchPattern(indexOp.getIndex(), m_ConstantInt(&indexValue)))
      return std::nullopt;

    Value src = indexOp.getSrc();
    int64_t linearIndex = indexValue.getSExtValue();
    if (auto subslice = src.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      auto offsets = subslice.getOffsets();
      if (offsets.empty() ||
          llvm::any_of(offsets.drop_front(), [](int32_t value) {
            return value != 0;
          }))
        return std::nullopt;
      src = subslice.getSrc();
      linearIndex += offsets.front();
    }
    return std::make_pair(src, linearIndex);
  };

  auto lhsKey = getConstantIndexKey(lhs);
  auto rhsKey = getConstantIndexKey(rhs);
  return lhsKey && rhsKey && lhsKey->second == rhsKey->second &&
         areEquivalentReplayBaseValues(lhsKey->first, rhsKey->first);
}

static bool isDerivedFromTMemLoadOfBase(Value value, Value base) {
  SmallVector<Value> worklist{value};
  SmallPtrSet<Value, 8> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!seen.insert(cur).second)
      continue;

    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;

    if (auto load = dyn_cast<TMEMLoadOp>(def)) {
      if (areEquivalentReplayBaseValues(load.getSrc(), base))
        return true;
      auto match = matchReplayableFullView(load.getSrc());
      if (match && areEquivalentReplayBaseValues(match->base, base))
        return true;
      continue;
    }

    if (isa<ttg::ConvertLayoutOp>(def) || def->hasTrait<OpTrait::Elementwise>()) {
      llvm::append_range(worklist, def->getOperands());
      continue;
    }
  }
  return false;
}

static bool shouldApplyReplayFullViewTransforms(
    TMEMLoadOp loadOp, const TMemReplayFullViewMatch &match) {
  if (hasReductionAlongNUse(loadOp.getResult()))
    return true;

  SmallVector<Value> worklist{loadOp.getResult()};
  SmallPtrSet<Value, 8> seen;
  bool storesBackToSameView = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;

    for (Operation *user : value.getUsers()) {
      if (auto store = dyn_cast<TMEMStoreOp>(user)) {
        auto storeMatch = matchReplayableFullView(store.getDst());
        Value storeBase = storeMatch ? storeMatch->base : store.getDst();
        if (!areEquivalentReplayBaseValues(storeBase, match.base))
          return true;
        storesBackToSameView = true;
        continue;
      }

      if (isa<ttg::ConvertLayoutOp, ReshapeOp, TransOp>(user) ||
          user->hasTrait<OpTrait::Elementwise>()) {
        llvm::append_range(worklist, user->getResults());
        continue;
      }

      return true;
    }
  }

  return !storesBackToSameView;
}

static Value lowerLeadingSliceViewLoad(PatternRewriter &rewriter, Location loc,
                                       const TMemLeadingSliceViewMatch &match,
                                       Type resultTy, int numWarps) {
  auto maybeSupportTy = getTMemLdStDirectSupportTensorType(match.base, numWarps);
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

static bool haveSameOutDimsIgnoringOrder(const LinearLayout &lhs,
                                         const LinearLayout &rhs) {
  auto lhsDims = llvm::to_vector(lhs.getOutDimNames());
  auto rhsDims = llvm::to_vector(rhs.getOutDimNames());
  if (lhsDims.size() != rhsDims.size())
    return false;
  for (StringAttr lhsDim : lhsDims) {
    if (!rhs.hasOutDim(lhsDim) ||
        lhs.getOutDimSize(lhsDim) != rhs.getOutDimSize(lhsDim))
      return false;
  }
  return true;
}

static std::optional<LinearLayout> getRegisterLinearLayout(RankedTensorType ty) {
  auto linear = dyn_cast<gpu::LinearEncodingAttr>(ty.getEncoding());
  if (!linear)
    return std::nullopt;
  return linear.getLinearLayout();
}

static bool canComputeTMemLdStEncodingInfo(RankedTensorType regTy,
                                           ttg::MemDescType memTy) {
  auto regLayout = getRegisterLinearLayout(regTy);
  return regLayout && haveSameOutDimsIgnoringOrder(*regLayout,
                                                   gpu::toLinearLayout(memTy));
}

static bool canComputeTMemLdStEncodingInfo(RankedTensorType regTy,
                                           const TMemLdStQueryLayout &query) {
  auto regLayout = getRegisterLinearLayout(regTy);
  return regLayout &&
         haveSameOutDimsIgnoringOrder(*regLayout, query.layout);
}

static bool queryLayoutOutDimsMatchMemDesc(ttg::MemDescType memTy,
                                           const TMemLdStQueryLayout &query) {
  return haveSameOutDimsIgnoringOrder(gpu::toLinearLayout(memTy), query.layout);
}

static std::optional<TMemAccessAtom>
getRequestedTMemLdStAtom(Value memDesc, RankedTensorType regTy, int maxnreg) {
  auto memTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!memTy)
    return std::nullopt;

  auto tryInfo = [](FailureOr<TMemLdStEncodingInfo> maybeInfo)
      -> std::optional<TMemAccessAtom> {
    if (failed(maybeInfo))
      return std::nullopt;
    return maybeInfo->atom;
  };

  if (auto rowPlan = getTMemLdStRowPlanForQuery(memDesc, memTy)) {
    if (canComputeTMemLdStEncodingInfo(regTy, memTy))
      if (auto atom = tryInfo(computeTMemLdStEncodingInfo(
              regTy, memTy, maxnreg, /*emitError=*/{}, rowPlan)))
        return atom;
  }

  auto tryQueryInfo = [&](const TMemLdStQueryLayout &queryLayout,
                          std::optional<TMemLdStRowPlan> rowPlan)
      -> std::optional<TMemAccessAtom> {
    if (!canComputeTMemLdStEncodingInfo(regTy, queryLayout))
      return std::nullopt;
    return tryInfo(computeTMemLdStEncodingInfo(
        regTy, memTy, queryLayout, maxnreg, /*emitError=*/{}, rowPlan));
  };

  std::string rawError;
  if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
          memDesc, /*preserveNonCanonicalView=*/true, &rawError);
      succeeded(rawQuery)) {
    auto rowPlan = getTMemLdStRowPlanForRawQuery(memDesc, memTy, *rawQuery);
    if (auto atom = tryQueryInfo(*rawQuery, rowPlan))
      return atom;
  }

  std::string supportError;
  if (auto supportPlan = getTMemLdStSupportQueryPlan(memDesc, &supportError)) {
    auto rowPlan = getTMemLdStRowPlanForSupportQuery(
        memDesc, memTy, supportPlan->query, supportPlan->rowPlan);
    if (auto atom = tryQueryInfo(supportPlan->query, rowPlan))
      return atom;
  }

  for (ttg::MemDescType queryTy : getTMemLdStQueryTypes(memDesc)) {
    if (!canComputeTMemLdStEncodingInfo(regTy, queryTy))
      continue;
    auto rowPlan = getTMemLdStRowPlanForQuery(memDesc, queryTy);
    if (auto atom = tryInfo(computeTMemLdStEncodingInfo(
            regTy, queryTy, maxnreg, /*emitError=*/{}, rowPlan)))
      return atom;
  }
  return std::nullopt;
}

static std::optional<RankedTensorType>
getReplayFullViewSupportTensorType(Value base, Value view,
                                   RankedTensorType requestedTy,
                                   unsigned numWarps, int maxnreg,
                                   bool allowRequestedTy = true) {
  auto baseTy = dyn_cast<ttg::MemDescType>(base.getType());
  auto desiredAtom = getRequestedTMemLdStAtom(view, requestedTy, maxnreg);
  if (!desiredAtom && view != base)
    desiredAtom = getRequestedTMemLdStAtom(base, requestedTy, maxnreg);
  auto isSupportedByBase = [&](RankedTensorType regTy,
                               TMemAccessAtom atom) -> bool {
    if (!baseTy)
      return false;
    auto hasAtom = [&](FailureOr<TMemLdStEncodingInfo> maybeInfo) {
      return succeeded(maybeInfo) &&
             isTMemAccessAtomCompatibleWithRequest(baseTy, atom,
                                                   maybeInfo->atom);
    };

    if (canComputeTMemLdStEncodingInfo(regTy, baseTy)) {
      auto rowPlan = getTMemLdStRowPlanForQuery(base, baseTy);
      if (hasAtom(computeTMemLdStEncodingInfo(regTy, baseTy, maxnreg,
                                              /*emitError=*/{}, rowPlan)))
        return true;
    }

    std::string rawError;
    if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
            base, /*preserveNonCanonicalView=*/true, &rawError);
        succeeded(rawQuery) && canComputeTMemLdStEncodingInfo(regTy, *rawQuery)) {
      auto rowPlan = getTMemLdStRowPlanForRawQuery(base, baseTy, *rawQuery);
      if (hasAtom(computeTMemLdStEncodingInfo(
              regTy, baseTy, *rawQuery, maxnreg, /*emitError=*/{}, rowPlan)))
        return true;
    }

    std::string supportError;
    if (auto supportPlan = getTMemLdStSupportQueryPlan(base, &supportError)) {
      if (canComputeTMemLdStEncodingInfo(regTy, supportPlan->query)) {
        auto rowPlan = getTMemLdStRowPlanForSupportQuery(
            base, baseTy, supportPlan->query, supportPlan->rowPlan);
        if (hasAtom(computeTMemLdStEncodingInfo(
                regTy, baseTy, supportPlan->query, maxnreg,
                /*emitError=*/{}, rowPlan)))
          return true;
      }
    }

    for (ttg::MemDescType queryTy : getTMemLdStQueryTypes(base)) {
      if (!canComputeTMemLdStEncodingInfo(regTy, queryTy))
        continue;
      auto rowPlan = getTMemLdStRowPlanForQuery(base, queryTy);
      if (hasAtom(computeTMemLdStEncodingInfo(
              regTy, queryTy, maxnreg, /*emitError=*/{}, rowPlan)))
        return true;
    }
    return false;
  };
  auto replayAtomOrder = [&]() {
    SmallVector<TMemAccessAtom> atoms;
    auto push = [&](TMemAccessAtom atom) {
      if (!llvm::is_contained(atoms, atom))
        atoms.push_back(atom);
    };
    if (desiredAtom) {
      if (*desiredAtom == TMemAccessAtom::I32x32b)
        push(TMemAccessAtom::I16x32bx2);
      push(*desiredAtom);
    }
    for (TMemAccessAtom atom :
         {TMemAccessAtom::I16x64b, TMemAccessAtom::I32x32b,
          TMemAccessAtom::I16x128b, TMemAccessAtom::I16x256b,
          TMemAccessAtom::I16x32bx2}) {
      push(atom);
    }
    return atoms;
  }();
  if (allowRequestedTy && baseTy && desiredAtom &&
      llvm::equal(baseTy.getShape(), requestedTy.getShape()) &&
      isSupportedByBase(requestedTy, *desiredAtom))
    return requestedTy;

  auto tryQueryTypes = [&](Value querySource)
      -> std::optional<RankedTensorType> {
    if (!baseTy)
      return std::nullopt;
    for (ttg::MemDescType queryTy : getTMemLdStQueryTypes(querySource)) {
      if (!canComputeTMemLdStEncodingInfo(requestedTy, queryTy))
        continue;
      auto rowPlan = getTMemLdStRowPlanForQuery(querySource, queryTy);
      if (!rowPlan)
        rowPlan = getBackingTMemLdStRowPlan(querySource);
      for (TMemAccessAtom atom : replayAtomOrder) {
        auto layout =
            getDistributedLayoutForTmemLdSt(queryTy, atom, numWarps, rowPlan);
        if (!layout)
          continue;
        Attribute encoding =
            gpu::LinearEncodingAttr::get(baseTy.getContext(), std::move(*layout));
        auto regTy = RankedTensorType::get(
            baseTy.getShape(), baseTy.getElementType(), encoding);
        auto info = computeTMemLdStEncodingInfo(regTy, queryTy, maxnreg,
                                                /*emitError=*/{}, rowPlan);
        if (succeeded(info) &&
            isTMemAccessAtomCompatibleWithRequest(queryTy, atom, info->atom) &&
            isSupportedByBase(regTy, atom))
          return regTy;
      }
    }
    return std::nullopt;
  };
  if (baseTy) {
    if (auto queryTypeSupport = tryQueryTypes(view))
      return queryTypeSupport;
    if (view != base)
      if (auto queryTypeSupport = tryQueryTypes(base))
        return queryTypeSupport;
  }

  if (!baseTy || !desiredAtom)
    return getTMemLdStDirectSupportTensorType(base, numWarps);

  auto makeType = [&](const LinearLayout &layout) -> RankedTensorType {
    Attribute encoding =
        gpu::LinearEncodingAttr::get(baseTy.getContext(), layout);
    return RankedTensorType::get(baseTy.getShape(), baseTy.getElementType(),
                                 encoding);
  };
  auto tryQuery = [&](const TMemLdStQueryLayout *queryLayout,
                      std::optional<TMemLdStRowPlan> rowPlan)
      -> std::optional<RankedTensorType> {
    if (queryLayout && !queryLayoutOutDimsMatchMemDesc(baseTy, *queryLayout))
      return std::nullopt;
    std::optional<LinearLayout> queryOverride;
    if (queryLayout)
      queryOverride = queryLayout->layout;
    auto layout = getDistributedLayoutForTmemLdSt(
        baseTy, *desiredAtom, numWarps, rowPlan, queryOverride);
    if (!layout)
      return std::nullopt;
    RankedTensorType candidateTy = makeType(*layout);
    if (!isSupportedByBase(candidateTy, *desiredAtom))
      return std::nullopt;
    if (queryLayout && !canComputeTMemLdStEncodingInfo(candidateTy, *queryLayout))
      return std::nullopt;
    if (!queryLayout && !canComputeTMemLdStEncodingInfo(candidateTy, baseTy))
      return std::nullopt;
    FailureOr<TMemLdStEncodingInfo> info =
        queryLayout ? computeTMemLdStEncodingInfo(
                          candidateTy, baseTy, *queryLayout, maxnreg,
                          /*emitError=*/{}, rowPlan)
                    : computeTMemLdStEncodingInfo(candidateTy, baseTy, maxnreg,
                                                  /*emitError=*/{}, rowPlan);
    if (failed(info) || info->atom != *desiredAtom)
      return std::nullopt;
    return candidateTy;
  };

  std::string rawError;
  if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
          base, /*preserveNonCanonicalView=*/true, &rawError);
      succeeded(rawQuery)) {
    auto rowPlan = getTMemLdStRowPlanForRawQuery(base, baseTy, *rawQuery);
    if (auto candidate = tryQuery(&*rawQuery, rowPlan))
      return candidate;
  }

  std::string supportError;
  if (auto supportPlan = getTMemLdStSupportQueryPlan(base, &supportError)) {
    auto rowPlan = getTMemLdStRowPlanForSupportQuery(
        base, baseTy, supportPlan->query, supportPlan->rowPlan);
    if (auto candidate = tryQuery(&supportPlan->query, rowPlan))
      return candidate;
  }

  auto rowPlan = getTMemLdStRowPlanForQuery(base, baseTy);
  if (auto candidate = tryQuery(/*queryLayout=*/nullptr, rowPlan))
    return candidate;

  return getTMemLdStDirectSupportTensorType(base, numWarps);
}

static std::optional<RankedTensorType>
getReplayFullViewBaseSupportTensorType(Value base, Value view,
                                       RankedTensorType requestedTy,
                                       unsigned numWarps, int maxnreg) {
  auto baseTy = dyn_cast<ttg::MemDescType>(base.getType());
  if (!baseTy)
    return std::nullopt;

  auto desiredAtom = getRequestedTMemLdStAtom(view, requestedTy, maxnreg);
  if (!desiredAtom && view != base)
    desiredAtom = getRequestedTMemLdStAtom(base, requestedTy, maxnreg);

  std::optional<TMemLdStRowPlan> queryRowPlan =
      getTMemLdStRowPlanForQuery(base, baseTy);
  for (std::optional<TMemLdStRowPlan> rowPlan :
       {std::optional<TMemLdStRowPlan>{}, queryRowPlan}) {
    for (TMemAccessAtom atom : getTMemLdStAtomSearchOrder(desiredAtom)) {
      auto layout =
          getDistributedLayoutForTmemLdSt(baseTy, atom, numWarps, rowPlan);
      if (!layout)
        continue;
      Attribute encoding =
          gpu::LinearEncodingAttr::get(baseTy.getContext(), std::move(*layout));
      auto regTy =
          RankedTensorType::get(baseTy.getShape(), baseTy.getElementType(),
                                encoding);
      FailureOr<TMemLdStEncodingInfo> info = computeTMemLdStEncodingInfo(
          regTy, baseTy, maxnreg, /*emitError=*/{}, rowPlan);
      if (succeeded(info) &&
          isTMemAccessAtomCompatibleWithRequest(baseTy, desiredAtom, info->atom))
        return regTy;
    }
  }
  return std::nullopt;
}

static SmallVector<int64_t>
getHalfSliceReplaySplitShape(ArrayRef<int64_t> srcShape, unsigned dim) {
  SmallVector<int64_t> splitShape;
  splitShape.reserve(srcShape.size() + 1);
  for (auto [idx, size] : llvm::enumerate(srcShape)) {
    if (idx == dim) {
      splitShape.push_back(2);
      splitShape.push_back(size / 2);
      continue;
    }
    splitShape.push_back(size);
  }
  return splitShape;
}

static SmallVector<int32_t> moveDimToBackOrder(unsigned rank, unsigned dim) {
  SmallVector<int32_t> order;
  order.reserve(rank);
  for (unsigned i = 0; i < rank; ++i) {
    if (i != dim)
      order.push_back(i);
  }
  order.push_back(dim);
  return order;
}

static std::pair<Value, Value>
splitTensorHalfAlongDim(PatternRewriter &rewriter, Location loc, Value tensor,
                        const TMemReplayHalfSliceStep &step, int numWarps) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  Value current = tensor;
  if (!llvm::equal(tensorTy.getShape(), step.srcShape))
    current = ReshapeOp::create(rewriter, loc, step.srcShape, current);

  auto splitShape = getHalfSliceReplaySplitShape(step.srcShape, step.dim);
  current = ReshapeOp::create(rewriter, loc, splitShape, current);
  SmallVector<int32_t> splitOrder =
      moveDimToBackOrder(splitShape.size(), step.dim);
  current = TransOp::create(rewriter, loc, current, splitOrder);

  auto currentTy = cast<RankedTensorType>(current.getType());
  auto splitFriendlyTy = getLeadingSliceSplitFriendlyType(
      rewriter.getContext(), currentTy.getElementType(), currentTy.getShape(),
      numWarps, ttg::lookupThreadsPerWarp(rewriter),
      ttg::lookupNumCTAs(rewriter));
  if (current.getType() != splitFriendlyTy) {
    current =
        ttg::ConvertLayoutOp::create(rewriter, loc, splitFriendlyTy, current);
  }

  auto split = SplitOp::create(rewriter, loc, current);
  auto normalizeResultShape = [&](Value value) -> Value {
    auto valueTy = cast<RankedTensorType>(value.getType());
    if (!llvm::equal(valueTy.getShape(), step.dstShape))
      value = ReshapeOp::create(rewriter, loc, step.dstShape, value);
    return value;
  };
  return {normalizeResultShape(split.getResult(0)),
          normalizeResultShape(split.getResult(1))};
}

static Value joinTensorHalfAlongDim(PatternRewriter &rewriter, Location loc,
                                    Value lhs, Value rhs,
                                    const TMemReplayHalfSliceStep &step) {
  Value joined = JoinOp::create(rewriter, loc, lhs, rhs);
  SmallVector<int64_t> splitShape =
      getHalfSliceReplaySplitShape(step.srcShape, step.dim);
  SmallVector<int32_t> splitOrder =
      moveDimToBackOrder(splitShape.size(), step.dim);
  Value untransposed =
      TransOp::create(rewriter, loc, joined, invertPermutation(splitOrder));
  return ReshapeOp::create(rewriter, loc, step.srcShape, untransposed);
}

static Value applyReplayHalfSliceViewSteps(
    PatternRewriter &rewriter, Location loc, Value tensor,
    ArrayRef<TMemReplayHalfSliceStep> steps, int numWarps) {
  Value current = tensor;
  for (const TMemReplayHalfSliceStep &step : steps) {
    switch (step.kind) {
    case TMemReplayHalfSliceStepKind::Reshape:
      current = ReshapeOp::create(rewriter, loc, step.dstShape, current);
      break;
    case TMemReplayHalfSliceStepKind::Trans:
      current = TransOp::create(rewriter, loc, current, step.order);
      break;
    case TMemReplayHalfSliceStepKind::HalfSlice: {
      auto [lhs, rhs] =
          splitTensorHalfAlongDim(rewriter, loc, current, step, numWarps);
      current = step.selectRHS ? rhs : lhs;
      break;
    }
    }
  }
  return current;
}

static Value replaceReplayHalfSliceViewValue(
    PatternRewriter &rewriter, Location loc, Value full,
    ArrayRef<TMemReplayHalfSliceStep> steps, Value replacement,
    int numWarps) {
  if (steps.empty()) {
    return reshapeAndConvertToType(
        rewriter, loc, replacement, cast<RankedTensorType>(full.getType()));
  }

  const TMemReplayHalfSliceStep &step = steps.front();
  auto fullTy = cast<RankedTensorType>(full.getType());
  Value restored;
  switch (step.kind) {
  case TMemReplayHalfSliceStepKind::Reshape: {
    Value transformed = ReshapeOp::create(rewriter, loc, step.dstShape, full);
    Value replaced = replaceReplayHalfSliceViewValue(
        rewriter, loc, transformed, steps.drop_front(), replacement, numWarps);
    restored = ReshapeOp::create(rewriter, loc, step.srcShape, replaced);
    break;
  }
  case TMemReplayHalfSliceStepKind::Trans: {
    Value transformed = TransOp::create(rewriter, loc, full, step.order);
    Value replaced = replaceReplayHalfSliceViewValue(
        rewriter, loc, transformed, steps.drop_front(), replacement, numWarps);
    restored =
        TransOp::create(rewriter, loc, replaced, invertPermutation(step.order));
    break;
  }
  case TMemReplayHalfSliceStepKind::HalfSlice: {
    auto [lhs, rhs] =
        splitTensorHalfAlongDim(rewriter, loc, full, step, numWarps);
    Value selected = step.selectRHS ? rhs : lhs;
    Value replaced = replaceReplayHalfSliceViewValue(
        rewriter, loc, selected, steps.drop_front(), replacement, numWarps);
    replaced = reshapeAndConvertToType(
        rewriter, loc, replaced, cast<RankedTensorType>(selected.getType()));
    restored = step.selectRHS
                   ? joinTensorHalfAlongDim(rewriter, loc, lhs, replaced, step)
                   : joinTensorHalfAlongDim(rewriter, loc, replaced, rhs, step);
    break;
  }
  }
  return reshapeAndConvertToType(rewriter, loc, restored, fullTy);
}

static FailureOr<Value>
lowerReplayHalfSliceViewLoad(PatternRewriter &rewriter, TMEMLoadOp loadOp,
                             const TMemReplayHalfSliceViewMatch &match) {
  int numWarps = ttg::lookupNumWarps(loadOp);
  auto maybeSupportTy = getTMemLdStDirectSupportTensorType(match.base, numWarps);
  if (!maybeSupportTy)
    return failure();

  RankedTensorType supportTy = *maybeSupportTy;
  Value support =
      TMEMLoadOp::create(rewriter, loadOp.getLoc(), supportTy, match.base);
  Value projected = applyReplayHalfSliceViewSteps(
      rewriter, loadOp.getLoc(), support, match.steps, numWarps);
  return reshapeAndConvertToType(
      rewriter, loadOp.getLoc(), projected,
      cast<RankedTensorType>(loadOp.getType()));
}

static LogicalResult
lowerReplayHalfSliceViewStore(PatternRewriter &rewriter, TMEMStoreOp storeOp,
                              const TMemReplayHalfSliceViewMatch &match) {
  if (!matchPattern(storeOp.getPred(), m_One()))
    return failure();

  int numWarps = ttg::lookupNumWarps(storeOp);
  auto maybeSupportTy = getTMemLdStDirectSupportTensorType(match.base, numWarps);
  if (!maybeSupportTy)
    return failure();

  RankedTensorType supportTy = *maybeSupportTy;
  Value support =
      TMEMLoadOp::create(rewriter, storeOp.getLoc(), supportTy, match.base);
  Value supportReplacement = replaceReplayHalfSliceViewValue(
      rewriter, storeOp.getLoc(), support, match.steps, storeOp.getSrc(),
      numWarps);
  supportReplacement = reshapeAndConvertToType(
      rewriter, storeOp.getLoc(), supportReplacement, supportTy);
  TMEMStoreOp::create(rewriter, storeOp.getLoc(), match.base,
                      supportReplacement, storeOp.getPred());
  rewriter.eraseOp(storeOp);
  return success();
}

static Value materializeDirectLeadingSubsliceIndexBase(PatternRewriter &rewriter,
                                                       Location loc,
                                                       Value base) {
  auto indexOp = base.getDefiningOp<ttg::MemDescIndexOp>();
  if (!indexOp)
    return base;

  APInt indexValue;
  if (!matchPattern(indexOp.getIndex(), m_ConstantInt(&indexValue)))
    return base;

  auto subsliceOp = indexOp.getSrc().getDefiningOp<ttg::MemDescSubsliceOp>();
  if (!subsliceOp)
    return base;

  auto srcTy = dyn_cast<ttg::MemDescType>(subsliceOp.getSrc().getType());
  auto sliceTy = dyn_cast<ttg::MemDescType>(subsliceOp.getType());
  auto baseTy = dyn_cast<ttg::MemDescType>(base.getType());
  if (!srcTy || !sliceTy || !baseTy || srcTy.getRank() == 0 ||
      sliceTy.getRank() != srcTy.getRank() ||
      baseTy.getRank() + 1 != sliceTy.getRank())
    return base;
  if (!llvm::equal(sliceTy.getShape().drop_front(), baseTy.getShape()) ||
      !llvm::equal(srcTy.getShape().drop_front(), baseTy.getShape()))
    return base;

  ArrayRef<int32_t> offsets = subsliceOp.getOffsets();
  if (offsets.size() != static_cast<size_t>(srcTy.getRank()) ||
      llvm::any_of(offsets.drop_front(), [](int32_t value) {
        return value != 0;
      }))
    return base;

  int64_t localIndex = indexValue.getSExtValue();
  int64_t globalIndex = static_cast<int64_t>(offsets.front()) + localIndex;
  if (localIndex < 0 || localIndex >= sliceTy.getShape().front() ||
      globalIndex < 0 || globalIndex >= srcTy.getShape().front())
    return base;

  Value globalIndexValue =
      arith::ConstantIntOp::create(rewriter, loc, globalIndex, 32);
  auto directIndex = ttg::MemDescIndexOp::createChecked(
      rewriter, loc, subsliceOp.getSrc(), globalIndexValue);
  if (failed(directIndex) || directIndex->getType() != base.getType())
    return base;
  return directIndex->getResult();
}

static bool isDirectLeadingSubsliceIndexBase(Value base) {
  auto indexOp = base.getDefiningOp<ttg::MemDescIndexOp>();
  if (!indexOp)
    return false;

  APInt indexValue;
  if (!matchPattern(indexOp.getIndex(), m_ConstantInt(&indexValue)))
    return false;

  auto subsliceOp = indexOp.getSrc().getDefiningOp<ttg::MemDescSubsliceOp>();
  if (!subsliceOp)
    return false;

  auto srcTy = dyn_cast<ttg::MemDescType>(subsliceOp.getSrc().getType());
  auto sliceTy = dyn_cast<ttg::MemDescType>(subsliceOp.getType());
  auto baseTy = dyn_cast<ttg::MemDescType>(base.getType());
  if (!srcTy || !sliceTy || !baseTy || srcTy.getRank() == 0 ||
      sliceTy.getRank() != srcTy.getRank() ||
      baseTy.getRank() + 1 != sliceTy.getRank())
    return false;
  if (!llvm::equal(sliceTy.getShape().drop_front(), baseTy.getShape()) ||
      !llvm::equal(srcTy.getShape().drop_front(), baseTy.getShape()))
    return false;

  ArrayRef<int32_t> offsets = subsliceOp.getOffsets();
  if (offsets.size() != static_cast<size_t>(srcTy.getRank()) ||
      llvm::any_of(offsets.drop_front(), [](int32_t value) {
        return value != 0;
      }))
    return false;

  int64_t localIndex = indexValue.getSExtValue();
  int64_t globalIndex = static_cast<int64_t>(offsets.front()) + localIndex;
  return localIndex >= 0 && localIndex < sliceTy.getShape().front() &&
         globalIndex >= 0 && globalIndex < srcTy.getShape().front();
}

static FailureOr<Value>
lowerReplayFullViewValueLoad(PatternRewriter &rewriter, Location loc,
                             Value view, RankedTensorType resultTy,
                             const TMemReplayFullViewMatch &match,
                             unsigned numWarps, int maxnreg,
                             bool applyViewTransforms) {
  Value replayBase =
      materializeDirectLeadingSubsliceIndexBase(rewriter, loc, match.base);
  bool canonicalizedBase = replayBase != match.base;
  std::optional<RankedTensorType> maybeSupportTy;
  if (canonicalizedBase)
    maybeSupportTy = getReplayFullViewBaseSupportTensorType(
        replayBase, view, resultTy, numWarps, maxnreg);
  if (canonicalizedBase && !maybeSupportTy && !applyViewTransforms)
    maybeSupportTy = getTMemLdStDirectSupportTensorType(replayBase, numWarps);
  if (!maybeSupportTy)
    maybeSupportTy = getReplayFullViewSupportTensorType(
        replayBase, view, resultTy, numWarps, maxnreg,
        /*allowRequestedTy=*/!canonicalizedBase);
  if (!maybeSupportTy)
    return failure();

  RankedTensorType supportTy = *maybeSupportTy;
  Value support = TMEMLoadOp::create(rewriter, loc, supportTy, replayBase);
  Value replacement = support;
  if (applyViewTransforms && supportTy != resultTy)
    replacement =
        applyTensorViewTransforms(rewriter, loc, replacement, match.transforms);
  return reshapeAndConvertToType(rewriter, loc, replacement, resultTy);
}

static FailureOr<Value>
lowerReplayFullViewLoad(PatternRewriter &rewriter, TMEMLoadOp loadOp,
                        const TMemReplayFullViewMatch &match) {
  bool applyViewTransforms =
      isDirectLeadingSubsliceIndexBase(match.base)
          ? shouldApplyReplayFullViewTransforms(loadOp, match)
          : hasReductionAlongNUse(loadOp.getResult());
  return lowerReplayFullViewValueLoad(
      rewriter, loadOp.getLoc(), loadOp.getSrc(),
      cast<RankedTensorType>(loadOp.getType()), match,
      ttg::lookupNumWarps(loadOp), getContextualMaxNReg(loadOp),
      applyViewTransforms);
}

static bool
canMaterializeReplayFullViewValue(Value value, RankedTensorType resultTy,
                                  const SmallPtrSetImpl<Value> &carriedValues,
                                  unsigned numWarps, int maxnreg) {
  if (value.getType() == resultTy || carriedValues.contains(value))
    return true;

  if (auto match = matchReplayableFullView(value)) {
    return static_cast<bool>(getReplayFullViewSupportTensorType(
        match->base, value, resultTy, numWarps, maxnreg));
  }

  auto opResult = dyn_cast<OpResult>(value);
  if (!opResult || !opResult.hasOneUse())
    return false;

  auto ifOp = dyn_cast<scf::IfOp>(opResult.getOwner());
  if (!ifOp)
    return false;

  unsigned resultIndex = opResult.getResultNumber();
  if (resultIndex >= ifOp.thenYield().getNumOperands() ||
      resultIndex >= ifOp.elseYield().getNumOperands())
    return false;
  return canMaterializeReplayFullViewValue(
             ifOp.thenYield().getOperand(resultIndex), resultTy, carriedValues,
             numWarps, maxnreg) &&
         canMaterializeReplayFullViewValue(
             ifOp.elseYield().getOperand(resultIndex), resultTy, carriedValues,
             numWarps, maxnreg);
}

static FailureOr<Value> materializeReplayFullViewValue(
    PatternRewriter &rewriter, Location loc, Value value,
    RankedTensorType resultTy, const DenseMap<Value, Value> &carriedValues,
    unsigned numWarps, int maxnreg, bool applyViewTransforms) {
  if (value.getType() == resultTy)
    return value;
  if (auto it = carriedValues.find(value); it != carriedValues.end())
    return it->second;

  if (auto match = matchReplayableFullView(value)) {
    return lowerReplayFullViewValueLoad(rewriter, loc, value, resultTy, *match,
                                        numWarps, maxnreg,
                                        applyViewTransforms);
  }

  auto opResult = dyn_cast<OpResult>(value);
  if (!opResult || !opResult.hasOneUse())
    return failure();

  auto ifOp = dyn_cast<scf::IfOp>(opResult.getOwner());
  if (!ifOp)
    return failure();

  unsigned resultIndex = opResult.getResultNumber();
  if (resultIndex >= ifOp.thenYield().getNumOperands() ||
      resultIndex >= ifOp.elseYield().getNumOperands())
    return failure();
  auto rewriteYield = [&](scf::YieldOp yieldOp) -> LogicalResult {
    rewriter.setInsertionPoint(yieldOp);
    FailureOr<Value> replacement = materializeReplayFullViewValue(
        rewriter, loc, yieldOp.getOperand(resultIndex), resultTy,
        carriedValues, numWarps, maxnreg, applyViewTransforms);
    if (failed(replacement))
      return failure();
    yieldOp.setOperand(resultIndex, *replacement);
    return success();
  };
  if (failed(rewriteYield(ifOp.thenYield())) ||
      failed(rewriteYield(ifOp.elseYield())))
    return failure();
  ifOp.getResult(resultIndex).setType(resultTy);
  return ifOp.getResult(resultIndex);
}

static LogicalResult
lowerReplayFullViewStore(PatternRewriter &rewriter, TMEMStoreOp storeOp,
                         const TMemReplayFullViewMatch &match) {
  if (!matchPattern(storeOp.getPred(), m_One()))
    return failure();

  int numWarps = ttg::lookupNumWarps(storeOp);
  Value replayBase = materializeDirectLeadingSubsliceIndexBase(
      rewriter, storeOp.getLoc(), match.base);
  bool canonicalizedBase = replayBase != match.base;
  std::optional<RankedTensorType> maybeSupportTy;
  if (canonicalizedBase)
    maybeSupportTy = getReplayFullViewBaseSupportTensorType(
        replayBase, storeOp.getDst(),
        cast<RankedTensorType>(storeOp.getSrc().getType()), numWarps,
        getContextualMaxNReg(storeOp));
  if (canonicalizedBase && !maybeSupportTy)
    maybeSupportTy = getTMemLdStDirectSupportTensorType(replayBase, numWarps);
  if (!maybeSupportTy)
    maybeSupportTy = getReplayFullViewSupportTensorType(
        replayBase, storeOp.getDst(),
        cast<RankedTensorType>(storeOp.getSrc().getType()), numWarps,
        getContextualMaxNReg(storeOp),
        /*allowRequestedTy=*/!canonicalizedBase);
  if (!maybeSupportTy)
    return failure();

  RankedTensorType supportTy = *maybeSupportTy;
  Value supportReplacement = storeOp.getSrc();
  bool sourceAlreadyInReplayOrder =
      isDerivedFromTMemLoadOfBase(supportReplacement, replayBase);
  if (!sourceAlreadyInReplayOrder && supportReplacement.getType() != supportTy)
    supportReplacement = applyInverseTensorViewTransforms(
        rewriter, storeOp.getLoc(), supportReplacement, match.transforms);
  supportReplacement = reshapeAndConvertToType(
      rewriter, storeOp.getLoc(), supportReplacement, supportTy);
  TMEMStoreOp::create(rewriter, storeOp.getLoc(), replayBase,
                      supportReplacement, storeOp.getPred());
  rewriter.eraseOp(storeOp);
  return success();
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
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    // Create the two TMEM subslices and their corresponding loads.
    Value tmem = tmemLoad.getSrc(); // Could itself be a subslice.
    int numWarps = ttg::lookupNumWarps(tmemLoad);
    auto slice0Layout =
        getReplaySliceRegLayout(getTMemSubSliceType(tmem, 0, splitNSize),
                                numWarps);
    auto slice1Layout = getReplaySliceRegLayout(
        getTMemSubSliceType(tmem, splitNSize, splitNSize), numWarps);
    if (!slice0Layout || !slice1Layout)
      return failure();
    rewriter.setInsertionPoint(tmemLoad);

    auto createSliceLoad =
        [&](int64_t nOffset, gpu::DistributedEncodingTrait distLayout)
        -> std::pair<TMEMLoadOp, ttg::ConvertLayoutOp> {
      // Generate the subslice op.
      Value subSlice = TMEMSubSliceOp::create(rewriter, tmemLoad.getLoc(), tmem,
                                              nOffset, splitNSize);

      RankedTensorType newLoadType =
          splitOp.getOutLHS().getType().cloneWithEncoding(distLayout);

      // Generate the load and convert_layout back to the original layout.
      auto load = TMEMLoadOp::create(rewriter, tmemLoad.getLoc(), newLoadType,
                                     subSlice);
      auto cvt = ttg::ConvertLayoutOp::create(
          rewriter, tmemLoad.getLoc(), splitOp.getOutLHS().getType(), load);

      return std::pair<TMEMLoadOp, ttg::ConvertLayoutOp>{load, cvt};
    };

    auto [load0, cvt0] = createSliceLoad(/*nOffset=*/0, *slice0Layout);
    auto [load1, cvt1] =
        createSliceLoad(/*nOffset=*/splitNSize, *slice1Layout);
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
    if (shouldPreserveDirectTMemLdStLeadingSliceView(loadOp.getSrc()))
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

class TMemReplayHalfSliceLoadPattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemReplayHalfSliceLoadPattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(TMEMLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchReplayableHalfSliceView(loadOp.getSrc());
    if (!match)
      return failure();

    rewriter.setInsertionPoint(loadOp);
    FailureOr<Value> replacement =
        lowerReplayHalfSliceViewLoad(rewriter, loadOp, *match);
    if (failed(replacement))
      return failure();
    rewriter.replaceOp(loadOp, *replacement);
    return success();
  }
};

class TMemReplayFullViewLoadPattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemReplayFullViewLoadPattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(TMEMLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchReplayableFullView(loadOp.getSrc());
    if (!match)
      return failure();

    rewriter.setInsertionPoint(loadOp);
    FailureOr<Value> replacement =
        lowerReplayFullViewLoad(rewriter, loadOp, *match);
    if (failed(replacement))
      return failure();
    rewriter.replaceOp(loadOp, *replacement);
    return success();
  }
};

class TMemReplayFullViewIfLoadPattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemReplayFullViewIfLoadPattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(TMEMLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!resultTy)
      return failure();

    auto srcResult = dyn_cast<OpResult>(loadOp.getSrc());
    if (!srcResult || !srcResult.hasOneUse())
      return failure();

    auto ifOp = dyn_cast<scf::IfOp>(srcResult.getOwner());
    if (!ifOp)
      return failure();

    unsigned resultIndex = srcResult.getResultNumber();
    auto getReplayableYieldMatch =
        [&](scf::YieldOp yieldOp) -> std::optional<TMemReplayFullViewMatch> {
      if (resultIndex >= yieldOp.getNumOperands())
        return std::nullopt;
      Value view = yieldOp.getOperand(resultIndex);
      auto match = matchReplayableFullView(view);
      if (!match)
        return std::nullopt;
      if (!getReplayFullViewSupportTensorType(
              match->base, view, resultTy, ttg::lookupNumWarps(loadOp),
              getContextualMaxNReg(loadOp)))
        return std::nullopt;
      return match;
    };
    auto thenMatch = getReplayableYieldMatch(ifOp.thenYield());
    auto elseMatch = getReplayableYieldMatch(ifOp.elseYield());
    if (!thenMatch || !elseMatch)
      return failure();

    SmallVector<Type> newResultTypes(ifOp.getResultTypes());
    newResultTypes[resultIndex] = resultTy;

    rewriter.setInsertionPoint(ifOp);
    auto newIf =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*ifOp.getOperation()));
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    for (auto [result, type] : llvm::zip(newIf.getResults(), newResultTypes))
      result.setType(type);

    auto rewriteYield = [&](scf::YieldOp yieldOp,
                            const TMemReplayFullViewMatch &match) {
      Value view = yieldOp.getOperand(resultIndex);
      rewriter.setInsertionPoint(yieldOp);
      bool applyViewTransforms =
          isDirectLeadingSubsliceIndexBase(match.base)
              ? shouldApplyReplayFullViewTransforms(loadOp, match)
              : hasReductionAlongNUse(loadOp.getResult());
      FailureOr<Value> replacement = lowerReplayFullViewValueLoad(
          rewriter, loadOp.getLoc(), view, resultTy, match,
          ttg::lookupNumWarps(loadOp), getContextualMaxNReg(loadOp),
          applyViewTransforms);
      assert(succeeded(replacement) &&
             "prevalidated replayable full-view if yield failed to lower");
      yieldOp.setOperand(resultIndex, *replacement);
    };
    rewriteYield(newIf.thenYield(), *thenMatch);
    rewriteYield(newIf.elseYield(), *elseMatch);

    rewriter.replaceOp(loadOp, newIf.getResult(resultIndex));
    for (unsigned i = 0, e = ifOp.getNumResults(); i < e; ++i) {
      if (i == resultIndex)
        continue;
      ifOp.getResult(i).replaceAllUsesWith(newIf.getResult(i));
    }
    rewriter.eraseOp(ifOp);
    return success();
  }
};

class TMemReplayFullViewForLoadPattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemReplayFullViewForLoadPattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(TMEMLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!resultTy)
      return failure();

    auto srcResult = dyn_cast<OpResult>(loadOp.getSrc());
    if (!srcResult || !srcResult.hasOneUse())
      return failure();

    auto forOp = dyn_cast<scf::ForOp>(srcResult.getOwner());
    if (!forOp)
      return failure();

    unsigned resultIndex = srcResult.getResultNumber();
    if (resultIndex >= forOp.getNumResults() ||
        resultIndex >= forOp.getInitArgs().size() ||
        resultIndex >= forOp.getRegionIterArgs().size())
      return failure();

    BlockArgument regionArg = forOp.getRegionIterArgs()[resultIndex];
    if (llvm::any_of(regionArg.getUsers(), [](Operation *user) {
          return !isa<scf::YieldOp>(user);
        }))
      return failure();

    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    unsigned numWarps = ttg::lookupNumWarps(loadOp);
    int maxnreg = getContextualMaxNReg(loadOp);

    SmallPtrSet<Value, 4> carriedValues;
    carriedValues.insert(regionArg);
    if (!canMaterializeReplayFullViewValue(forOp.getInitArgs()[resultIndex],
                                           resultTy, carriedValues, numWarps,
                                           maxnreg) ||
        !canMaterializeReplayFullViewValue(yieldOp.getOperand(resultIndex),
                                           resultTy, carriedValues, numWarps,
                                           maxnreg))
      return failure();

    bool applyViewTransforms = hasReductionAlongNUse(loadOp.getResult());
    DenseMap<Value, Value> noCarriedValues;
    rewriter.setInsertionPoint(forOp);
    FailureOr<Value> initReplacement = materializeReplayFullViewValue(
        rewriter, loadOp.getLoc(), forOp.getInitArgs()[resultIndex], resultTy,
        noCarriedValues, numWarps, maxnreg, applyViewTransforms);
    if (failed(initReplacement))
      return failure();

    OpOperand &initOperand =
        forOp->getOpOperand(resultIndex + forOp.getNumControlOperands());
    initOperand.set(*initReplacement);
    regionArg.setType(resultTy);
    forOp.getResult(resultIndex).setType(resultTy);

    DenseMap<Value, Value> carriedReplacements;
    carriedReplacements[regionArg] = regionArg;
    rewriter.setInsertionPoint(yieldOp);
    FailureOr<Value> yieldReplacement = materializeReplayFullViewValue(
        rewriter, loadOp.getLoc(), yieldOp.getOperand(resultIndex), resultTy,
        carriedReplacements, numWarps, maxnreg, applyViewTransforms);
    if (failed(yieldReplacement))
      return failure();
    yieldOp.setOperand(resultIndex, *yieldReplacement);

    rewriter.replaceOp(loadOp, forOp.getResult(resultIndex));
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
    int splitNSize = shape[2];
    if (splitNSize < 8)
      return failure();

    Location loc = storeOp.getLoc();
    Value tmem = storeOp.getDst();
    if (matchLeadingSliceView(tmem))
      return failure();
    int numWarps = ttg::lookupNumWarps(storeOp);
    auto slice0Layout =
        getReplaySliceRegLayout(getTMemSubSliceType(tmem, 0, splitNSize),
                                numWarps);
    auto slice1Layout = getReplaySliceRegLayout(
        getTMemSubSliceType(tmem, splitNSize, splitNSize), numWarps);
    if (!slice0Layout || !slice1Layout)
      return failure();
    Value truePred = arith::ConstantOp::create(b, loc, b.getBoolAttr(true));

    auto createSlice =
        [&](TypedValue<RankedTensorType> input,
            int offset, gpu::DistributedEncodingTrait distLayout) {
      auto subSlice = TMEMSubSliceOp::create(b, loc, tmem, offset, splitNSize);
      auto newType = input.getType().cloneWithEncoding(distLayout);
      auto cvt = ttg::ConvertLayoutOp::create(b, loc, newType, input);
      auto store =
          TMEMStoreOp::create(b, loc, subSlice, cvt.getResult(), truePred);
      return store;
    };

    auto store0 = createSlice(joinOp.getLhs(), 0, *slice0Layout);
    auto store1 = createSlice(joinOp.getRhs(), splitNSize, *slice1Layout);
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
    if (shouldPreserveDirectTMemLdStLeadingSliceView(storeOp.getDst()))
      return failure();

    if (succeeded(lowerTMemPhysicalSupportStore(rewriter, storeOp)))
      return success();

    if (!matchPattern(storeOp.getPred(), m_One()))
      return failure();

    auto match = matchLeadingSliceView(storeOp.getDst());
    if (!match)
      return failure();

    int numWarps = ttg::lookupNumWarps(storeOp);
    auto maybeSupportTy =
        getTMemLdStDirectSupportTensorType(match->base, numWarps);
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

class TMemReplayHalfSliceStorePattern : public OpRewritePattern<TMEMStoreOp> {
public:
  TMemReplayHalfSliceStorePattern(MLIRContext *context)
      : OpRewritePattern<TMEMStoreOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(TMEMStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchReplayableHalfSliceView(storeOp.getDst());
    if (!match)
      return failure();

    rewriter.setInsertionPoint(storeOp);
    return lowerReplayHalfSliceViewStore(rewriter, storeOp, *match);
  }
};

class TMemReplayFullViewStorePattern : public OpRewritePattern<TMEMStoreOp> {
public:
  TMemReplayFullViewStorePattern(MLIRContext *context)
      : OpRewritePattern<TMEMStoreOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(TMEMStoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto match = matchReplayableFullView(storeOp.getDst());
    if (!match)
      return failure();

    rewriter.setInsertionPoint(storeOp);
    return lowerReplayFullViewStore(rewriter, storeOp, *match);
  }
};

// Pick an optimized tmem load layout based on its users. When there are
// multiple warpgroups tmem_load results can be distirbuted along M or N across
// the warpgroups. By default distribute along N but when there is a reduction
// along N dimension we want to distribute along M instead to avoid having to
// reduce across warps.
class TMemLoadReducePattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemLoadReducePattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/2) {}

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
        getTMemLoadReductionLayoutForMemDesc(tmemLoadOp.getSrc(), numWarps);
    if (!newLayout) {
      newLayout = getTmemLoadLayoutSplitLongM(
          oldType, tmemLoadOp.getSrc().getType(), numWarps);
    }
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

class TMemFuseLoadReducePattern : public OpRewritePattern<TMEMLoadOp> {
public:
  TMemFuseLoadReducePattern(MLIRContext *context)
      : OpRewritePattern<TMEMLoadOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(TMEMLoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    if (!loadOp || loadOp.getRedOp() || loadOp.getToken())
      return failure();

    auto loadTy = dyn_cast<RankedTensorType>(loadOp.getType());
    auto srcTy = dyn_cast<ttg::MemDescType>(loadOp.getSrc().getType());
    if (!loadTy || !srcTy || loadTy.getElementType().isF32() == false ||
        srcTy.getElementType().isF32() == false ||
        isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding()))
      return failure();

    struct ReduceMatch {
      triton::ReduceOp reduceOp;
      math::AbsFOp absOp;
      TMEMLoadReduceModifier modifier;
    };
    auto matchReduce = [&](Value value) -> std::optional<ReduceMatch> {
      bool useAbs = false;
      math::AbsFOp absOp;
      Value reduceInput = value;
      if (absOp = value.getDefiningOp<math::AbsFOp>()) {
        if (absOp.getOperand() != loadOp.getResult())
          return std::nullopt;
        useAbs = true;
        reduceInput = absOp.getResult();
      }

      for (Operation *user : reduceInput.getUsers()) {
        auto reduceOp = dyn_cast<triton::ReduceOp>(user);
        if (!reduceOp || reduceOp.getAxis() != 1 ||
            reduceOp.getNumOperands() != 1 || reduceOp.getNumResults() != 1 ||
            reduceOp.getOperand(0) != reduceInput) {
          continue;
        }

        Operation *combiner = reduceOp.getSingleCombiner();
        if (!combiner)
          continue;

        std::optional<TMEMLoadReduceModifier> modifier;
        if (isa<arith::MinNumFOp>(combiner))
          modifier = TMEMLoadReduceModifier::MIN;
        else if (isa<arith::MaxNumFOp>(combiner))
          modifier = TMEMLoadReduceModifier::MAX;
        else
          continue;
        return ReduceMatch{reduceOp, useAbs ? absOp : math::AbsFOp(),
                           *modifier};
      }
      return std::nullopt;
    };

    std::optional<ReduceMatch> match = matchReduce(loadOp.getResult());
    if (!match) {
      for (Operation *user : loadOp.getResult().getUsers()) {
        auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user);
        if (!cvt)
          continue;
        match = matchReduce(cvt.getResult());
        if (match)
          break;
      }
    }
    if (!match)
      return failure();

    auto isReductionCompatible =
        [](FailureOr<TMemLdStEncodingInfo> info) -> bool {
      return succeeded(info) && isTMemLdStReductionCompatible(*info);
    };
    int maxnreg = getContextualMaxNReg(loadOp);
    auto canLowerType = [&](RankedTensorType candidateTy) {
      auto support = getTmemLoadReductionLayoutSupport(
          candidateTy, ttg::toLinearLayout(candidateTy));
      if (!support)
        return false;

      if (isReductionFriendlyTmemSourceLayout(srcTy)) {
        auto rowPlan = getTMemLdStRowPlanForQuery(loadOp.getSrc(), srcTy);
        if (isReductionCompatible(computeTMemLdStEncodingInfo(
                candidateTy, srcTy, maxnreg, /*emitError=*/{}, rowPlan))) {
          return true;
        }
      }

      std::string supportError;
      if (auto supportPlan =
              getTMemLdStSupportQueryPlan(loadOp.getSrc(), &supportError)) {
        auto rowPlan = supportPlan->rowPlan;
        if (!rowPlan)
          rowPlan = getTMemLdStRowPlanForQueryLayout(
              loadOp.getSrc(), srcTy, supportPlan->query);
        if (!rowPlan)
          rowPlan = getBackingTMemLdStRowPlan(loadOp.getSrc());
        if (isReductionCompatible(computeTMemLdStEncodingInfo(
                candidateTy, srcTy, supportPlan->query, maxnreg,
                /*emitError=*/{}, rowPlan))) {
          return true;
        }
      }

      std::string rawError;
      if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
              loadOp.getSrc(), /*preserveNonCanonicalView=*/true, &rawError);
          succeeded(rawQuery)) {
        auto rowPlan =
            getTMemLdStRowPlanForQueryLayout(loadOp.getSrc(), srcTy, *rawQuery);
        if (!rowPlan)
          rowPlan = getBackingTMemLdStRowPlan(loadOp.getSrc());
        if (isReductionCompatible(computeTMemLdStEncodingInfo(
                candidateTy, srcTy, *rawQuery, maxnreg, /*emitError=*/{},
                rowPlan))) {
          return true;
        }
      }

      for (ttg::MemDescType queryTy : getTMemLdStQueryTypes(loadOp.getSrc())) {
        auto rowPlan = getTMemLdStRowPlanForQuery(loadOp.getSrc(), queryTy);
        if (isReductionCompatible(computeTMemLdStEncodingInfo(
                candidateTy, queryTy, maxnreg, /*emitError=*/{}, rowPlan))) {
          return true;
        }
      }
      return false;
    };

    RankedTensorType fusedTy = loadTy;
    if (auto plannedLayout = getTMemLoadReductionLayoutForMemDesc(
            loadOp.getSrc(), ttg::lookupNumWarps(loadOp))) {
      auto plannedTy = loadTy.cloneWithEncoding(*plannedLayout);
      if (canLowerType(plannedTy))
        fusedTy = plannedTy;
    }
    if (!canLowerType(fusedTy))
      return failure();

    auto redOpAttr = TMEMLoadReduceModifierAttr::get(rewriter.getContext(),
                                                     match->modifier);
    BoolAttr absAttr =
        match->absOp ? rewriter.getBoolAttr(true) : BoolAttr(nullptr);
    auto fusedLoad = TMEMLoadOp::create(
        rewriter, loadOp.getLoc(), fusedTy, /*token=*/Type(), loadOp.getSrc(),
        /*dep=*/Value(), redOpAttr, absAttr, /*NaN=*/nullptr);

    Value red = fusedLoad.getRed();
    auto reduceTy =
        cast<RankedTensorType>(match->reduceOp.getResult().front().getType());
    if (red.getType() != reduceTy)
      red = reshapeAndConvertToType(rewriter, match->reduceOp.getLoc(), red,
                                    reduceTy);

    Value loaded = fusedLoad.getResult();
    if (loaded.getType() != loadTy)
      loaded = reshapeAndConvertToType(rewriter, loadOp.getLoc(), loaded,
                                       loadTy);

    rewriter.replaceOp(match->reduceOp, red);
    loadOp.getResult().replaceAllUsesWith(loaded);
    if (match->absOp && match->absOp->use_empty())
      rewriter.eraseOp(match->absOp);
    rewriter.eraseOp(loadOp);
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
             TMemReplayHalfSliceLoadPattern, TMemReplayFullViewLoadPattern,
             TMemReplayFullViewIfLoadPattern, TMemReplayFullViewForLoadPattern,
             TMemStoreJoinPattern, TMemLeadingSliceStorePattern,
             TMemReplayHalfSliceStorePattern, TMemReplayFullViewStorePattern,
             TMemLoadReducePattern,
             TMemFuseLoadReducePattern, TMemFromSharedMemPattern,
             TMemToSharedMemPattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

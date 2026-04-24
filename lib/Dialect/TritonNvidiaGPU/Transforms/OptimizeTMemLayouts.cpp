#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

static bool isPlainSingleResultTMemLoad(TMEMLoadOp loadOp) {
  return loadOp && loadOp->getNumResults() == 1 && !loadOp.getRedOp() &&
         !loadOp.getToken();
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
    if (!isPlainSingleResultTMemLoad(tmemLoad))
      return failure();
    auto shape = reshapeOp.getResult().getType().getShape();
    auto rootMemTy = getSplitLoadRootMemDescType(tmemLoad.getSrc());
    if (!rootMemTy)
      return failure();
    // The split-load rewrite is currently only semantics-preserving on 32-bit
    // TMEM tiles. Sub-32-bit Blackwell epilogues must keep the original
    // full-load path until the rewritten pack::16b decomposition is fixed.
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
    if (!isPlainSingleResultTMemLoad(tmemLoadOp))
      return failure();

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
        getTMemLoadReductionLayoutForMemDesc(tmemLoadOp.getSrc().getType(),
                                            numWarps);
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
    if (!isPlainSingleResultTMemLoad(tmemLoadOp))
      return failure();

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
        .add<TMemSplitLoadPattern, TMemStoreJoinPattern, TMemLoadReducePattern,
             TMemFromSharedMemPattern, TMemToSharedMemPattern>(context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

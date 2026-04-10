#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUPROMOTELHSTOTMEMPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {
template <class MMAOpTy>
Attribute getLHSTMemLayout(MMAOpTy tcGen5MMAOp, gpu::MemDescType lhsTMEMType) {
  int numWarps = ttg::lookupNumWarps(tcGen5MMAOp);
  return nvidia_gpu::getDefaultLayoutForTmemLdSt(lhsTMEMType, numWarps);
}

template <class MMAOpTy> class LHSToTMem : public OpRewritePattern<MMAOpTy> {
public:
  using OpRewritePattern<MMAOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MMAOpTy tcGen5MMAOp,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = tcGen5MMAOp->getContext();
    Location loc = tcGen5MMAOp.getLoc();
    auto lhs = tcGen5MMAOp.getA();
    auto localAllocOp = lhs.template getDefiningOp<ttg::LocalAllocOp>();
    if (!localAllocOp)
      return failure();
    // Limit the liverange of the TMem allocations to single block.
    if (localAllocOp->getParentRegion() != tcGen5MMAOp->getParentRegion())
      return failure();
    Value src = localAllocOp.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcLayout = srcType.getEncoding();
    auto accInfo = getMMAv5AccumulatorLayoutInfo(tcGen5MMAOp.getD().getType());
    if (!accInfo)
      return failure();
    auto cgaLayout = triton::gpu::getCGALayout(srcLayout);
    // TMem encoding for A operand is the same as for D (Acc), but packed for
    // bitwidth=16
    unsigned elemBitWidth =
        lhs.getType().getElementType().getIntOrFloatBitWidth();
    // We don't currently support fp8 (not sure if we can)
    if (elemBitWidth != 16 && elemBitWidth != 32) {
      return failure();
    }
    const unsigned colStride = 1;
    auto canonicalATMemEncoding = nvidia_gpu::getCanonicalTMemLinearEncoding(
        lhs.getType().getShape(), accInfo->mmaSizeM,
        lhs.getType().getShape()[1], colStride, cgaLayout, accInfo->twoCTAs);
    if (!canonicalATMemEncoding)
      return failure();
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    ttg::MemDescType lhsMemDescType = ttg::MemDescType::get(
        lhs.getType().getShape(), lhs.getType().getElementType(),
        *canonicalATMemEncoding, tensorMemorySpace,
        /*mutableMemory=*/false);
    auto hasZeroBasisAlong = [](const LinearLayout &layout, StringAttr dim) {
      if (!layout.hasInDim(dim))
        return false;
      unsigned dimBits = layout.getInDimSizeLog2(dim);
      for (unsigned idx = 0; idx < dimBits; ++idx) {
        if (llvm::all_of(layout.getBasis(dim, idx),
                         [](int32_t value) { return value == 0; }))
          return true;
      }
      return false;
    };
    auto lhsMemLayout = toLinearLayout(lhsMemDescType);
    auto kRow = StringAttr::get(context, "row");
    auto kCol = StringAttr::get(context, "col");
    bool isPackedM64Promotion =
        elemBitWidth == 16 && lhsMemDescType.getRank() == 2 &&
        lhsMemDescType.getShape()[0] == 64 &&
        hasZeroBasisAlong(lhsMemLayout, kRow) &&
        !hasZeroBasisAlong(lhsMemLayout, kCol);
    if (isPackedM64Promotion)
      return failure();
    bool layoutTmemCompatible =
        isDistributedLayoutTMemCompatible(tcGen5MMAOp, srcType, lhsMemDescType);
    Attribute newLayout = srcLayout;
    if (!layoutTmemCompatible) {
      if (!comesFromLoadOrBlockArg(src) ||
          triton::tools::getBoolEnv("ALLOW_LHS_TMEM_LAYOUT_CONVERSION")) {
        newLayout = getLHSTMemLayout(tcGen5MMAOp, lhsMemDescType);
      } else {
        return failure();
      }
    }
    rewriter.setInsertionPointAfter(localAllocOp);
    if (newLayout != srcLayout) {
      auto ty = cast<RankedTensorType>(src.getType());
      auto newTy = ty.cloneWithEncoding(newLayout);
      src = ttg::ConvertLayoutOp::create(rewriter, loc, newTy, src);
    }
    auto tMemAlloc = TMEMAllocOp::create(rewriter, loc, lhsMemDescType, src);
    nvidia_gpu::setExplicitMMAv5RootRowPlanIfNeeded(tMemAlloc);
    tcGen5MMAOp.getAMutable().assign(tMemAlloc);
    return success();
  }
};
} // namespace

class TritonNvidiaGPUPromoteLHSToTMemPass
    : public impl::TritonNvidiaGPUPromoteLHSToTMemPassBase<
          TritonNvidiaGPUPromoteLHSToTMemPass> {
public:
  using TritonNvidiaGPUPromoteLHSToTMemPassBase<
      TritonNvidiaGPUPromoteLHSToTMemPass>::
      TritonNvidiaGPUPromoteLHSToTMemPassBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<LHSToTMem<TCGen5MMAOp>>(context);
    patterns.add<LHSToTMem<TCGen5MMAScaledOp>>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

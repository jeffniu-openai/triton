#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_RELAYOUTTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

// Given a tensor and its representation in tensor memory, determine its
// distributed layout.
FailureOr<RankedTensorType>
getTMEMTensorLayout(const TypeConverter *tc, Operation *op,
                    RankedTensorType type, MemDescType memdesc,
                    unsigned numWarps) {
  (void)numWarps;
  type = cast<RankedTensorType>(tc->convertType(type));
  if (ttng::isDistributedLayoutTMemCompatible(op, type, memdesc))
    return type;
  auto layouts = ttng::getTmemCompatibleLayouts(op, type, memdesc);
  if (layouts.empty()) {
    InFlightDiagnostic diag =
        op->emitError("TMEM layout has no supported register layout");
    diag.attachNote() << "tensor type: " << type;
    diag.attachNote() << "tensor memory descriptor type: " << memdesc;
    return failure();
  }
  return type.cloneWithEncoding(layouts.front());
}

struct TMEMLoadOpPattern : public OpConversionPattern<ttng::TMEMLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    auto maybeType =
        getTMEMTensorLayout(typeConverter, op, op.getType(), op.getSrc().getType(),
                            lookupNumWarps(op));
    if (failed(maybeType))
      return failure();
    RankedTensorType type = *maybeType;
    rewriter.modifyOpInPlace(op, [&] { op.getResult().setType(type); });
    if (type == resultType)
      return success();

    rewriter.setInsertionPointAfter(op);
    auto cvt = ConvertLayoutOp::create(rewriter, op.getLoc(), resultType,
                                       op.getResult());
    // Bypass the rewriter to avoid issues with the conversion framework's
    // tracking of conditional replacements.
    // See https://github.com/llvm/llvm-project/commit/504b50789602
    op.getResult().replaceAllUsesExcept(cvt, cvt);
    return success();
  }
};

struct TMEMStoreOpPattern : public OpConversionPattern<ttng::TMEMStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto maybeType =
        getTMEMTensorLayout(typeConverter, op, op.getSrc().getType(),
                            op.getDst().getType(), lookupNumWarps(op));
    if (failed(maybeType))
      return failure();
    RankedTensorType type = *maybeType;
    Value src = adaptor.getSrc();
    if (cast<RankedTensorType>(src.getType()) != type)
      src = ConvertLayoutOp::create(rewriter, op.getLoc(), type, src);
    rewriter.modifyOpInPlace(op, [&] { op.getSrcMutable().assign(src); });
    return success();
  }
};

struct TMEMAllocOpPattern : public OpConversionPattern<ttng::TMEMAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return success();
    auto maybeType = getTMEMTensorLayout(typeConverter, op, op.getSrc().getType(),
                                         op.getType(), lookupNumWarps(op));
    if (failed(maybeType))
      return failure();
    RankedTensorType type = *maybeType;
    Value src = adaptor.getSrc();
    if (cast<RankedTensorType>(src.getType()) != type)
      src = ConvertLayoutOp::create(rewriter, op.getLoc(), type, src);
    rewriter.modifyOpInPlace(op, [&] { op.getSrcMutable().assign(src); });
    return success();
  }
};

class RelayoutTritonGPU
    : public triton::impl::RelayoutTritonGPUBase<RelayoutTritonGPU> {
public:
  using RelayoutTritonGPUBase::RelayoutTritonGPUBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    int numWarps = lookupNumWarps(mod);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);

    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, /*enableSourceRemat=*/true);
    TritonGPUConversionTarget target(*context, typeConverter);
    target.addDynamicallyLegalDialect<ttng::TritonNvidiaGPUDialect>(
        [&](Operation *op) {
          return TritonGPUConversionTarget::isDynamicallyLegal(op,
                                                               typeConverter);
        });

    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    patterns.insert<
        // clang-format off
        GatherScatterOpPattern<ttng::AsyncTMAGatherOp>,
        GatherScatterOpPattern<ttng::AsyncTMAScatterOp>,
        TMEMLoadOpPattern,
        TMEMStoreOpPattern,
        TMEMAllocOpPattern
        // clang-format on
        >(typeConverter, context);

    ConversionConfig config;
    config.allowPatternRollback = false;
    if (failed(
            applyPartialConversion(mod, target, std::move(patterns), config)))
      return signalPassFailure();
  }
};

} // namespace

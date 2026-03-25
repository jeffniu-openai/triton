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
getTMEMTensorLayout(const TypeConverter &typeConverter, Operation *op,
                    RankedTensorType type, MemDescType memdesc,
                    unsigned numWarps) {
  type = cast<RankedTensorType>(typeConverter.convertType(type));
  if (ttng::isDistributedLayoutTMemCompatible(op, type, memdesc))
    return type;
  auto layouts = ttng::getTmemCompatibleLayouts(op, type, memdesc);
  if (layouts.empty()) {
    InFlightDiagnostic diag =
        op->emitError("TMEM layout has no supported register layout");
    diag.attachNote() << "tensor type: " << type;
    diag.attachNote() << "tensor memory descriptor type: " << memdesc;
    diag.attachNote()
        << "reshape or permute so TMEM columns stay contiguous, or use a "
           "supported TMEM register layout and insert convert_layout "
           "explicitly";
    return failure();
  }
  return type.cloneWithEncoding(layouts.front());
}

LogicalResult relayoutTMEMLoadOp(const TypeConverter &typeConverter,
                                 ttng::TMEMLoadOp op) {
  RankedTensorType resultType = op.getType();
  auto maybeType = getTMEMTensorLayout(typeConverter, op, op.getType(),
                                       op.getSrc().getType(),
                                       lookupNumWarps(op));
  if (failed(maybeType))
    return failure();
  RankedTensorType type = *maybeType;
  if (type == resultType)
    return success();

  OpBuilder builder(op);
  op.getResult().setType(type);
  builder.setInsertionPointAfter(op);
  auto cvt =
      ConvertLayoutOp::create(builder, op.getLoc(), resultType, op.getResult());
  op.getResult().replaceAllUsesExcept(cvt, cvt);
  return success();
}

LogicalResult relayoutTMEMStoreOp(const TypeConverter &typeConverter,
                                  ttng::TMEMStoreOp op) {
  auto maybeType = getTMEMTensorLayout(typeConverter, op, op.getSrc().getType(),
                                       op.getDst().getType(),
                                       lookupNumWarps(op));
  if (failed(maybeType))
    return failure();
  RankedTensorType type = *maybeType;
  Value src = op.getSrc();
  if (cast<RankedTensorType>(src.getType()) == type)
    return success();

  OpBuilder builder(op);
  src = ConvertLayoutOp::create(builder, op.getLoc(), type, src);
  op.getSrcMutable().assign(src);
  return success();
}

LogicalResult relayoutTMEMAllocOp(const TypeConverter &typeConverter,
                                  ttng::TMEMAllocOp op) {
  if (!op.getSrc())
    return success();

  auto maybeType = getTMEMTensorLayout(typeConverter, op, op.getSrc().getType(),
                                       op.getType(),
                                       lookupNumWarps(op));
  if (failed(maybeType))
    return failure();
  RankedTensorType type = *maybeType;
  Value src = op.getSrc();
  if (cast<RankedTensorType>(src.getType()) == type)
    return success();

  OpBuilder builder(op);
  src = ConvertLayoutOp::create(builder, op.getLoc(), type, src);
  op.getSrcMutable().assign(src);
  return success();
}

LogicalResult relayoutTMEMOps(ModuleOp mod, const TypeConverter &typeConverter) {
  SmallVector<Operation *> tmemOps;
  mod.walk([&](Operation *op) {
    if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TMEMAllocOp>(op))
      tmemOps.push_back(op);
  });

  for (Operation *op : tmemOps) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(op)) {
      if (failed(relayoutTMEMLoadOp(typeConverter, load)))
        return failure();
      continue;
    }
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(op)) {
      if (failed(relayoutTMEMStoreOp(typeConverter, store)))
        return failure();
      continue;
    }
    if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
      if (failed(relayoutTMEMAllocOp(typeConverter, alloc)))
        return failure();
    }
  }

  return success();
}

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
    if (failed(relayoutTMEMOps(mod, typeConverter)))
      return signalPassFailure();

    TritonGPUConversionTarget target(*context, typeConverter);
    target.addDynamicallyLegalDialect<ttng::TritonNvidiaGPUDialect>(
        [&](Operation *op) {
          return TritonGPUConversionTarget::isDynamicallyLegal(op,
                                                               typeConverter);
        });
    target.addLegalOp<ttng::TMEMLoadOp, ttng::TMEMStoreOp,
                      ttng::TMEMAllocOp>();

    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    patterns.insert<
        // clang-format off
        GatherScatterOpPattern<ttng::AsyncTMAGatherOp>,
        GatherScatterOpPattern<ttng::AsyncTMAScatterOp>
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

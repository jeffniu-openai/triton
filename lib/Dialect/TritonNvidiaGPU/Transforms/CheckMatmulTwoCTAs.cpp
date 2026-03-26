#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"

namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUCHECKMATMULTWOCTAPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

class TritonNvidiaGPUCheckMatmulTwoCTAPass
    : public impl::TritonNvidiaGPUCheckMatmulTwoCTAPassBase<
          TritonNvidiaGPUCheckMatmulTwoCTAPass> {
public:
  using impl::TritonNvidiaGPUCheckMatmulTwoCTAPassBase<
      TritonNvidiaGPUCheckMatmulTwoCTAPass>::
      TritonNvidiaGPUCheckMatmulTwoCTAPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    Operation *firstUser = nullptr;
    bool firstTwoCTA = false;

    auto checkAndRecord = [&](Operation *op, bool currentTwoCTA,
                              StringRef source) -> WalkResult {
      if (!firstUser) {
        firstUser = op;
        firstTwoCTA = currentTwoCTA;
        return WalkResult::advance();
      }
      if (currentTwoCTA != firstTwoCTA) {
        auto diag = op->emitError()
                    << "inconsistent two_ctas setting across tensor memory "
                       "operations; expected all explicit two_ctas users to "
                    << (firstTwoCTA ? "enable" : "disable") << " two_ctas.";
        diag.attachNote(firstUser->getLoc())
            << "first two_ctas user here has two_ctas="
            << (firstTwoCTA ? "true" : "false") << ".";
        diag.attachNote() << "current two_ctas source: " << source << ".";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };

    WalkResult result = mod.walk([&](Operation *op) -> WalkResult {
      if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op))
        return checkAndRecord(op, mma.getTwoCtas(), "MMAv5 op attribute");

      auto checkTypes = [&](TypeRange types, StringRef source) -> WalkResult {
        for (Type type : types) {
          if (auto twoCTAs = getTensorMemoryTwoCTAs(type))
            if (auto res = checkAndRecord(op, *twoCTAs, source);
                res.wasInterrupted())
              return res;
        }
        return WalkResult::advance();
      };

      if (auto res = checkTypes(op->getOperandTypes(), "TMEM operand type");
          res.wasInterrupted())
        return res;
      return checkTypes(op->getResultTypes(), "TMEM result type");
    });

    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    bool twoCTAValue = firstUser ? firstTwoCTA : false;
    mod->setAttr(AttrTwoCTAsName, BoolAttr::get(mod.getContext(), twoCTAValue));
  }
};

} // namespace

} // namespace mlir::triton::nvidia_gpu

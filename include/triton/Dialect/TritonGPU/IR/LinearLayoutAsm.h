#ifndef TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTASM_H_
#define TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTASM_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "triton/Tools/LinearLayout.h"

#include <optional>

namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir

namespace mlir::triton::gpu {

std::optional<LinearLayout> parseLinearLayout(
    const DictionaryAttr &dict, AsmParser &parser,
    ArrayRef<StringRef> inDimNames, int serializedRank = 0);

void printLinearLayout(AsmPrinter &printer, const LinearLayout &ll,
                       bool skipEmptyBases = false);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTASM_H_

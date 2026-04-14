#include "triton/Dialect/TritonGPU/IR/LinearLayoutAsm.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"

#include <limits>
#include <vector>

namespace mlir::triton::gpu {

std::optional<LinearLayout> parseLinearLayout(
    const DictionaryAttr &dict, AsmParser &parser,
    ArrayRef<StringRef> inDimNames, int serializedRank) {
  LinearLayout::BasesT bases;

  // Parse the basis names in order (the order is relevant)
  for (const auto &inDimNameStr : inDimNames) {
    auto inDimName = StringAttr::get(parser.getContext(), inDimNameStr);
    Attribute value = dict.get(inDimName);
    if (!value) {
      parser.emitError(parser.getCurrentLocation(), "Expected basis of '")
          << inDimName.getValue() << "' not found";
      return {};
    }
    // Expecting an array of arrays
    auto arrayOfArraysAttr = mlir::dyn_cast<ArrayAttr>(value);
    if (!arrayOfArraysAttr) {
      parser.emitError(parser.getCurrentLocation(),
                       "Expected array of arrays for basis of '")
          << inDimName.getValue() << "'";
      return {};
    }

    std::vector<std::vector<int32_t>> inDimBases;
    for (Attribute arrayAttr : arrayOfArraysAttr) {
      auto intArrayAttr = mlir::dyn_cast<ArrayAttr>(arrayAttr);
      if (!intArrayAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected array of integers in basis for '")
            << inDimName.getValue() << "'";
        return {};
      }
      std::vector<int32_t> basis;
      for (Attribute intAttr : intArrayAttr) {
        auto intValueAttr = mlir::dyn_cast<IntegerAttr>(intAttr);
        if (!intValueAttr) {
          parser.emitError(parser.getCurrentLocation(),
                           "Expected integer in basis for '")
              << inDimName.getValue() << "'";
          return {};
        }
        basis.push_back(intValueAttr.getInt());
      }
      inDimBases.push_back(std::move(basis));
    }
    bases[inDimName] = std::move(inDimBases);
  }
  size_t rank = 0;
  for (const auto &basesDim : llvm::make_second_range(bases)) {
    if (!basesDim.empty()) {
      rank = basesDim[0].size();
      break;
    }
  }

  auto parseExplicitOutDims = [&]()
      -> std::optional<SmallVector<std::pair<StringAttr, int32_t>>> {
    auto outAttr = dict.get(StringAttr::get(parser.getContext(), "out"));
    if (!outAttr)
      return std::nullopt;
    auto outArray = dyn_cast<ArrayAttr>(outAttr);
    if (!outArray) {
      parser.emitError(parser.getCurrentLocation(),
                       "Expected array of integers for 'out'");
      return SmallVector<std::pair<StringAttr, int32_t>>{};
    }
    if (rank != 0 && outArray.size() != rank) {
      parser.emitError(parser.getCurrentLocation(),
                       "Explicit out rank and rank deduced from LL need to match");
      return SmallVector<std::pair<StringAttr, int32_t>>{};
    }
    SmallVector<int64_t> sizes;
    sizes.reserve(outArray.size());
    for (Attribute sizeAttr : outArray) {
      auto intAttr = dyn_cast<IntegerAttr>(sizeAttr);
      if (!intAttr) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected integer in 'out'");
        return SmallVector<std::pair<StringAttr, int32_t>>{};
      }
      int64_t value = intAttr.getInt();
      if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
        parser.emitError(parser.getCurrentLocation(),
                         "Expected positive 32-bit integer in 'out'");
        return SmallVector<std::pair<StringAttr, int32_t>>{};
      }
      sizes.push_back(value);
    }
    return standardOutDimPairs(parser.getContext(), sizes);
  };

  auto explicitOutDims = parseExplicitOutDims();
  if (explicitOutDims && explicitOutDims->empty())
    return {};

  if (rank == 0 && serializedRank == 0 && !explicitOutDims) {
    parser.emitError(parser.getCurrentLocation(), "Empty Layout not supported");
    return {};
  }

  if (rank == 0) {
    rank = explicitOutDims ? explicitOutDims->size() : serializedRank;
  } else if (serializedRank != 0 && serializedRank != rank) {
    parser.emitError(parser.getCurrentLocation(),
                     "Serialized rank and rank deduced from LL need to match");
    return {};
  }

  std::string error;
  std::optional<LinearLayout> layout;
  if (explicitOutDims) {
    layout = LinearLayout::tryCreate(std::move(bases), *explicitOutDims,
                                     /*requireSurjective=*/false, &error);
  } else {
    layout = LinearLayout::tryCreate(
        std::move(bases), standardOutDimNames(parser.getContext(), rank),
        /*requireSurjective=*/true, &error);
  }
  if (!layout) {
    parser.emitError(parser.getCurrentLocation()) << error;
    return {};
  }
  return layout;
}

void printLinearLayout(AsmPrinter &printer, const LinearLayout &ll,
                       bool skipEmptyBases) {
  auto bases = ll.getBases();
  if (skipEmptyBases) {
    decltype(bases) filtered;
    for (auto &kv : bases)
      if (!kv.second.empty())
        filtered.insert(kv);
    bases = std::move(filtered);
  }

  auto parts = llvm::to_vector(llvm::map_range(bases, [](const auto &base) {
    return base.first.str() + " = " + "[" +
           join(base.second, ", ",
                [](const std::vector<int32_t> &vec) {
                  return "[" + join(vec, ", ") + "]";
                }) +
           "]";
  }));
  if (!ll.isSurjective()) {
    parts.push_back("out = [" +
                    join(llvm::to_vector(ll.getOutDimSizes()), ", ") + "]");
  }
  printer << join(parts, ", ");
}

} // namespace mlir::triton::gpu

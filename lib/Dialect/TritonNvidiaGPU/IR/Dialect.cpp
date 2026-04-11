/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutAsm.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutAsm.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.cpp.inc"

using namespace mlir;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {

namespace gpu {
std::optional<LinearLayout>
tensorMemoryToLinearLayout(ArrayRef<int64_t> shape,
                           nvidia_gpu::TensorMemoryEncodingAttr,
                           std::string *error = nullptr);
}

namespace nvidia_gpu {

static constexpr int numTmemRows = 128;

static std::optional<LinearLayout>
tryLinearToCGAEncodingLayout(const LinearLayout &ll, ArrayRef<unsigned> cgaShape,
                             std::string *error = nullptr) {
  auto inDims = to_vector(ll.getInDimNames());
  if (inDims.empty()) {
    if (error != nullptr)
      *error = "layout must have at least one input dimension";
    return std::nullopt;
  }
  auto *ctx = inDims[0].getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  if (!llvm::is_contained(inDims, kBlock)) {
    if (error != nullptr)
      *error = "layout must contain a 'block' input dimension";
    return std::nullopt;
  }
  auto outDims = to_vector(ll.getOutDimNames());
  if (cgaShape.size() != outDims.size()) {
    if (error != nullptr)
      *error = "layout rank and CGA rank must match";
    return std::nullopt;
  }
  auto cgaLayout = ll.sublayout({kBlock}, outDims);
  for (auto [idx, outDim] : llvm::enumerate(outDims))
    cgaLayout = cgaLayout.resizeOutDim(outDim, cgaShape[idx]);
  return cgaLayout;
}

static LinearLayout linearToCGAEncodingLayout(const LinearLayout &ll,
                                              ArrayRef<unsigned> cgaShape) {
  std::string error;
  auto cgaLayout = tryLinearToCGAEncodingLayout(ll, cgaShape, &error);
  assert(cgaLayout && "layout must have a valid CGA factorization");
  return *cgaLayout;
}

static std::string stringifyAttribute(Attribute attr) {
  std::string str;
  llvm::raw_string_ostream os(str);
  os << attr;
  return str;
}

static void printDiagStr(llvm::raw_ostream &os, const Diagnostic &diag) {
  for (const DiagnosticArgument &arg : diag.getArguments())
    arg.print(os);
  os << "\n";
  for (const Diagnostic &note : diag.getNotes())
    printDiagStr(os, note);
}

static std::string stringifyShape(ArrayRef<int64_t> shape) {
  return "[" + triton::join(shape, ", ") + "]";
}

static int64_t getShapeProduct(ArrayRef<int64_t> shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

static std::optional<LinearLayout>
applyCGALayoutToLegacyLikeTMemTile(LinearLayout tile, ArrayRef<int64_t> shape,
                                   ArrayRef<int64_t> shapePerCTA,
                                   gpu::CGAEncodingAttr cgaLayout,
                                   std::string *error = nullptr) {
  auto setError = [&](const Twine &msg) {
    if (error)
      *error = msg.str();
    return std::nullopt;
  };
  auto *ctx = cgaLayout.getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto bases = tile.getBases();
  auto &blockBases = bases[kBlock];
  for (ArrayRef<int32_t> basis :
       cgaLayout.getLinearLayout().getBases().lookup(kBlock)) {
    std::vector<int32_t> scaledBasis(basis.begin(), basis.end());
    for (size_t i = 0; i < scaledBasis.size(); ++i)
      scaledBasis[i] *= static_cast<int32_t>(shapePerCTA[i]);
    blockBases.emplace_back(std::move(scaledBasis));
  }
  std::string layoutError;
  auto maybeLayout = LinearLayout::tryCreate(
      std::move(bases), standardOutDimPairs(ctx, shape),
      /*requireSurjective=*/true, &layoutError);
  if (!maybeLayout)
    return setError(layoutError);
  return *maybeLayout;
}

static std::optional<LinearLayout>
buildCanonicalLegacyLikeTMemLinearLayout(ArrayRef<int64_t> shape,
                                         unsigned blockM, unsigned blockN,
                                         unsigned colStride,
                                         gpu::CGAEncodingAttr cgaLayout,
                                         bool twoCTAs,
                                         std::string *error = nullptr) {
  auto setError = [&](const Twine &msg) {
    if (error)
      *error = msg.str();
    return std::nullopt;
  };
  if (shape.size() != 2) {
    return setError("expected a rank-2 tensor memory shape but got " +
                    Twine(shape.size()) + " dimensions");
  }
  if (blockM != 64 && blockM != 128)
    return setError("unsupported legacy-like TMEM blockM=" + Twine(blockM) +
                    "; expected 64 or 128");
  if (!llvm::isPowerOf2_32(blockN) || blockN > 512)
    return setError("unsupported legacy-like TMEM blockN=" + Twine(blockN));
  if (!(colStride == 1 || colStride == 2 || colStride == 4))
    return setError("unsupported legacy-like TMEM colStride=" +
                    Twine(colStride));

  auto *ctx = cgaLayout.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto dims = standardOutDimNames(ctx, 2);
  bool isM64TwoCTA = blockM == 64 && twoCTAs;

  auto shapePerCTA = getShapePerCTA(cgaLayout.getCTASplitNum(), shape);
  if (shapePerCTA.size() != 2) {
    return setError("expected a rank-2 per-CTA tensor memory shape but got " +
                    Twine(shapePerCTA.size()) + " dimensions");
  }

  unsigned effectiveBlockN = std::min<int64_t>(blockN, shapePerCTA[1]);
  LinearLayout tile = LinearLayout::zeros1D(colStride, kCol, dims[1]);
  if (blockM == 64 && !twoCTAs) {
    tile *= LinearLayout::identity1D(16, kRow, dims[0]) *
            LinearLayout::identity1D(effectiveBlockN, kCol, dims[1]);
    auto bases = tile.getBases();
    if (shapePerCTA[0] > blockM) {
      bases[kRow].push_back({64, 0});
    } else if (shapePerCTA[1] > effectiveBlockN) {
      bases[kRow].push_back({0, static_cast<int32_t>(effectiveBlockN)});
    } else {
      bases[kRow].push_back({0, 0});
    }
    bases[kRow].push_back({16, 0});
    bases[kRow].push_back({32, 0});
    std::string layoutError;
    auto maybeTile = LinearLayout::tryCreate(std::move(bases), dims,
                                             /*requireSurjective=*/true,
                                             &layoutError);
    if (!maybeTile)
      return setError(layoutError);
    tile = *maybeTile;
  } else {
    tile *= LinearLayout::identity1D(blockM, kRow, dims[0]) *
            LinearLayout::identity1D(effectiveBlockN, kCol, dims[1]);
    if (isM64TwoCTA) {
      auto bases = tile.getBases();
      auto colIt = bases.find(kCol);
      if (colIt == bases.end() || colIt->second.empty()) {
        return setError("legacy-like twoCTA blockM=64 TMEM layout requires "
                        "at least one column basis");
      }
      bases[kRow].push_back(colIt->second.back());
      colIt->second.pop_back();
      std::string layoutError;
      auto maybeTile = LinearLayout::tryCreate(
          std::move(bases), tile.getOutDims(),
          /*requireSurjective=*/tile.isSurjective(), &layoutError);
      if (!maybeTile)
        return setError(layoutError);
      tile = *maybeTile;
    }
  }

  auto tileM = tile.getOutDimSize(dims[0]);
  auto tileN = tile.getOutDimSize(dims[1]);
  if (shapePerCTA[0] < tileM || shapePerCTA[1] < tileN) {
    return setError("shape per CTA " + Twine(shapePerCTA[0]) + "x" +
                    Twine(shapePerCTA[1]) +
                    " is smaller than the TMEM tile " + Twine(tileM) + "x" +
                    Twine(tileN));
  }
  if (shapePerCTA[0] % tileM != 0 || shapePerCTA[1] % tileN != 0) {
    return setError("shape per CTA " + Twine(shapePerCTA[0]) + "x" +
                    Twine(shapePerCTA[1]) +
                    " is not an integer multiple of the TMEM tile " +
                    Twine(tileM) + "x" + Twine(tileN));
  }
  auto repsM = shapePerCTA[0] / tileM;
  auto repsN = shapePerCTA[1] / tileN;
  if (!llvm::isPowerOf2_32(repsM) || !llvm::isPowerOf2_32(repsN)) {
    return setError("shape per CTA " + Twine(shapePerCTA[0]) + "x" +
                    Twine(shapePerCTA[1]) +
                    " expands the TMEM tile by non-power-of-two factors " +
                    Twine(repsM) + "x" + Twine(repsN));
  }

  // Broadcast the remaining dimensions in order [0, 1].
  tile = tile * LinearLayout::identity1D(repsM, kCol, dims[0]) *
         LinearLayout::identity1D(repsN, kCol, dims[1]);
  auto maybeFullTile =
      applyCGALayoutToLegacyLikeTMemTile(tile, shape, shapePerCTA, cgaLayout,
                                         error);
  if (!maybeFullTile)
    return std::nullopt;
  tile = *maybeFullTile;
  auto expectedElems = getShapeProduct(shape);
  auto actualElems = static_cast<int64_t>(tile.getTotalOutDimSize());
  if (actualElems != expectedElems) {
    return setError("TMEM tile expands to " + Twine(actualElems) +
                    " logical elements for requested shape " +
                    stringifyShape(shape) + " (" + Twine(expectedElems) +
                    " elements)");
  }
  return tile;
}

static std::optional<LinearLayout>
canonicalizeLegacyTensorMemoryLayout(ArrayRef<int64_t> shape, Attribute encoding,
                                     std::string *error = nullptr) {
  auto legacy = cast<TensorMemoryEncodingAttr>(encoding);
  if (shape.size() < 2) {
    if (error != nullptr) {
      *error = "tensor memory layout sugar " +
               stringifyAttribute(encoding) +
               " requires at least 2 trailing dimensions, but got shape " +
               stringifyShape(shape);
    }
    return std::nullopt;
  }
  auto trailingShape = shape.take_back(2);
  std::string baseError;
  auto base = buildCanonicalLegacyLikeTMemLinearLayout(
      trailingShape, legacy.getBlockM(), legacy.getBlockN(),
      legacy.getColStride(), legacy.getCGALayout(), legacy.getTwoCTAs(),
      &baseError);
  if (!base) {
    if (error != nullptr) {
      *error = "tensor memory layout sugar " + stringifyAttribute(encoding) +
               " cannot be canonicalized for shape " +
               stringifyShape(trailingShape) + ": " + baseError +
               ". Use #ttng.tensor_memory_linear for arbitrary TMEM views, or "
               "choose a legacy tensor_memory_encoding whose tile matches the "
               "allocation shape.";
    }
    return std::nullopt;
  }
  if (static_cast<int64_t>(base->getTotalOutDimSize()) !=
      getShapeProduct(trailingShape)) {
    if (error != nullptr) {
      SmallVector<int64_t> baseShape;
      for (const auto &[outDim, size] : base->getOutDims())
        baseShape.push_back(size);
      *error = "tensor memory layout sugar " +
               stringifyAttribute(encoding) +
               " produced logical shape " + stringifyShape(baseShape) +
               " for requested shape " + stringifyShape(trailingShape) +
               "; the total number of elements does not match";
    }
    return std::nullopt;
  }
  return base->reshapeOuts(standardOutDimPairs(
      legacy.getContext(), trailingShape,
      static_cast<unsigned>(shape.size() - 2)));
}

FailureOr<gpu::CGAEncodingAttr> parseCGALayoutRankTwo(AsmParser &parser) {
  Attribute attr;
  if (parser.parseAttribute(attr).failed())
    return failure();
  if (auto cgaAttr = gpu::parseCGAAttr(parser, attr, /*rank=*/2))
    return *cgaAttr;
  return failure();
}

void printCGALayoutRankTwo(AsmPrinter &printer, gpu::CGAEncodingAttr cgaAttr) {
  gpu::printCGAAttr(printer, cgaAttr);
}

bool isTensorMemoryEncoding(Attribute layout) {
  return isa<TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr,
             TensorMemoryScalesEncodingAttr>(layout);
}

std::optional<bool> getTensorMemoryTwoCTAs(Attribute layout) {
  if (auto linear = dyn_cast<TensorMemoryLinearEncodingAttr>(layout))
    return linear.getTwoCTAs();
  if (auto legacy = dyn_cast<TensorMemoryEncodingAttr>(layout))
    return legacy.getTwoCTAs();
  return std::nullopt;
}

std::optional<bool> getTensorMemoryTwoCTAs(Type type) {
  auto memDesc = dyn_cast<gpu::MemDescType>(type);
  if (!memDesc)
    return std::nullopt;
  return getTensorMemoryTwoCTAs(memDesc.getEncoding());
}

std::optional<Attribute>
tryGetCanonicalTensorMemoryEncoding(ArrayRef<int64_t> shape, Attribute layout,
                                    std::string *error) {
  if (auto linear = dyn_cast<TensorMemoryLinearEncodingAttr>(layout))
    return linear;
  if (auto legacy = dyn_cast<TensorMemoryEncodingAttr>(layout)) {
    auto canonicalLayout =
        canonicalizeLegacyTensorMemoryLayout(shape, legacy, error);
    if (!canonicalLayout)
      return std::nullopt;
    return tryMakeTensorMemoryLinearEncoding(layout.getContext(),
                                             std::move(*canonicalLayout),
                                             legacy.getTwoCTAs(), error);
  }
  return layout;
}

std::optional<Attribute>
tryGetCanonicalTensorMemoryEncoding(MemDescType memDescType,
                                    std::string *error) {
  auto layout = memDescType.getEncoding();
  if (!isTensorMemoryEncoding(layout))
    return layout;
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = isa<TensorMemoryEncodingAttr>(layout)
                   ? memDescType.getAllocShape().take_back(rank)
                   : memDescType.getShape().take_back(rank);
  return tryGetCanonicalTensorMemoryEncoding(shape, layout, error);
}

Attribute getCanonicalTensorMemoryEncoding(ArrayRef<int64_t> shape,
                                           Attribute layout) {
  std::string error;
  auto canonical = tryGetCanonicalTensorMemoryEncoding(shape, layout, &error);
  assert(canonical && "expected canonical tensor memory encoding to exist");
  return *canonical;
}

Attribute getCanonicalTensorMemoryEncoding(MemDescType memDescType) {
  std::string error;
  auto canonical = tryGetCanonicalTensorMemoryEncoding(memDescType, &error);
  assert(canonical && "expected canonical tensor memory encoding to exist");
  return *canonical;
}

std::optional<LinearLayout>
tryGetCanonicalTensorMemoryLinearLayout(ArrayRef<int64_t> shape,
                                        Attribute layout,
                                        std::string *error) {
  if (auto linear = dyn_cast<TensorMemoryLinearEncodingAttr>(layout))
    return linear.getLinearLayout();
  if (isa<TensorMemoryEncodingAttr>(layout))
    return canonicalizeLegacyTensorMemoryLayout(shape, layout, error);
  return triton::gpu::toLinearLayout(shape, layout);
}

std::optional<LinearLayout>
tryGetCanonicalTensorMemoryLinearLayout(MemDescType memDescType,
                                        std::string *error) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(shape, layout, error);
  if (maybeCanonical &&
      tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, shape)) {
    return maybeCanonical;
  }
  auto allocShape = memDescType.getAllocShape().take_back(rank);
  return tryGetCanonicalTensorMemoryLinearLayout(allocShape, layout, error);
}

bool tensorMemoryLinearLayoutMatchesShape(const LinearLayout &layout,
                                          ArrayRef<int64_t> shape) {
  if (shape.size() != static_cast<size_t>(layout.getNumOutDims()))
    return false;
  auto *ctx = (*layout.getOutDimNames().begin()).getContext();
  auto dims = standardOutDimNames(ctx, shape.size());
  for (auto [dim, size] : llvm::zip_equal(dims, shape)) {
    if (layout.getOutDimSize(dim) != size)
      return false;
  }
  return true;
}

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(MemDescType type, std::string *error) {
  Attribute enc = type.getEncoding();
  if (!isTensorMemoryEncoding(enc) || isa<TensorMemoryScalesEncodingAttr>(enc))
    return std::nullopt;
  auto rank = cast<LayoutEncodingTrait>(enc).getRank();
  auto shape = type.getShape().take_back(rank);
  return getCanonicalTMemLinearEncoding(shape, enc, error);
}

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(ArrayRef<int64_t> shape, Attribute encoding,
                               std::string *error) {
  if (!isTensorMemoryEncoding(encoding) ||
      isa<TensorMemoryScalesEncodingAttr>(encoding))
    return std::nullopt;
  auto canonical = tryGetCanonicalTensorMemoryEncoding(shape, encoding, error);
  if (!canonical)
    return std::nullopt;
  auto linear = cast<TensorMemoryLinearEncodingAttr>(*canonical);
  if (!tensorMemoryLinearLayoutMatchesShape(linear.getLinearLayout(), shape)) {
    if (error)
      *error =
          "tensor memory view is not representable as a standalone TMEM "
          "linear layout";
    return std::nullopt;
  }
  return linear;
}

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(ArrayRef<int64_t> shape, unsigned blockM,
                               unsigned blockN, unsigned colStride,
                               gpu::CGAEncodingAttr cgaLayout, bool twoCTAs,
                               std::string *error) {
  auto maybeLayout = buildCanonicalLegacyLikeTMemLinearLayout(
      shape, blockM, blockN, colStride, cgaLayout, twoCTAs, error);
  if (!maybeLayout)
    return std::nullopt;
  return tryMakeTensorMemoryLinearEncoding(cgaLayout.getContext(),
                                           std::move(*maybeLayout), twoCTAs,
                                           error);
}

LinearLayout getCanonicalTensorMemoryLinearLayout(ArrayRef<int64_t> shape,
                                                  Attribute layout) {
  std::string error;
  auto canonical = tryGetCanonicalTensorMemoryLinearLayout(shape, layout, &error);
  assert(canonical && "expected canonical tensor memory linear layout to exist");
  return *canonical;
}

LinearLayout getCanonicalTensorMemoryLinearLayout(MemDescType memDescType) {
  std::string error;
  auto canonical = tryGetCanonicalTensorMemoryLinearLayout(memDescType, &error);
  assert(canonical && "expected canonical tensor memory linear layout to exist");
  return *canonical;
}

std::optional<TensorMemoryLinearEncodingAttr>
tryMakeTensorMemoryLinearEncoding(MLIRContext *ctx, LinearLayout linearLayout,
                                  bool twoCTAs, std::string *error) {
  std::string diagStr;
  llvm::raw_string_ostream diagOs(diagStr);
  ScopedDiagnosticHandler handler(
      ctx, [&](Diagnostic &diag) { printDiagStr(diagOs, diag); });
  if (failed(TensorMemoryLinearEncodingAttr::verifyInvariants(
          [&] { return mlir::emitError(UnknownLoc::get(ctx)); }, linearLayout,
          twoCTAs))) {
    if (error)
      *error = diagOs.str();
    return std::nullopt;
  }
  return TensorMemoryLinearEncodingAttr::get(ctx, std::move(linearLayout),
                                             twoCTAs);
}

LinearLayout normalizeTensorMemoryLinearLayoutForAnalysis(LinearLayout layout) {
  if (layout.getNumInDims() == 0)
    return layout;
  auto *ctx = (*layout.getInDimNames().begin()).getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  for (StringAttr dim : {kRow, kCol, kBlock}) {
    if (layout.hasInDim(dim))
      layout = layout.removeZeroBasesAlongDim(dim);
  }
  if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1) {
    layout = layout.squeezeIns(kBlock);
  }

  SmallVector<StringAttr> canonicalInDims;
  for (StringAttr dim : {kRow, kCol, kBlock}) {
    if (layout.hasInDim(dim))
      canonicalInDims.push_back(dim);
  }
  if (!canonicalInDims.empty())
    layout = layout.transposeIns(canonicalInDims);
  return layout.transposeOuts(standardOutDimNames(ctx, layout.getNumOutDims()));
}

static LinearLayout
normalizeTensorMemoryLinearLayoutForMMAv5Family(LinearLayout layout) {
  if (layout.getNumInDims() == 0)
    return layout;
  auto *ctx = (*layout.getInDimNames().begin()).getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1)
    layout = layout.squeezeIns(kBlock);

  SmallVector<StringAttr> canonicalInDims;
  for (StringAttr dim : {kRow, kCol, kBlock}) {
    if (layout.hasInDim(dim))
      canonicalInDims.push_back(dim);
  }
  if (!canonicalInDims.empty())
    layout = layout.transposeIns(canonicalInDims);
  return layout.transposeOuts(standardOutDimNames(ctx, layout.getNumOutDims()));
}

struct MMAv5TMemLayoutPlan {
  unsigned instrShapeM;
  unsigned instrShapeN;
  unsigned colStride;
  bool twoCTAs;
};

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5Family(ArrayRef<int64_t> shape, const LinearLayout &canonicalLayout,
                gpu::CGAEncodingAttr cga, bool twoCTAs,
                ArrayRef<unsigned> blockNs,
                std::optional<unsigned> preferredColStride = std::nullopt) {
  if (shape.size() != 2)
    return std::nullopt;

  auto normalizedLinear =
      normalizeTensorMemoryLinearLayoutForMMAv5Family(canonicalLayout);
  if (normalizedLinear.getNumOutDims() != 2)
    return std::nullopt;

  auto *ctx = (*normalizedLinear.getOutDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  auto basisIsWholeTileMultiple = [](ArrayRef<int32_t> basis, unsigned tileM,
                                     unsigned tileN) {
    return basis.size() == 2 && basis[0] % static_cast<int32_t>(tileM) == 0 &&
           basis[1] % static_cast<int32_t>(tileN) == 0;
  };

  auto gatherTileBases =
      [&](const LinearLayout &layout, StringAttr dim, unsigned bitCount) {
        SmallVector<std::array<int32_t, 2>> bases;
        bases.reserve(bitCount);
        for (unsigned i = 0; i < bitCount; ++i) {
          auto basis = layout.getBasis(dim, i);
          std::array<int32_t, 2> entry;
          entry[0] = basis[0];
          entry[1] = basis[1];
          bases.push_back(entry);
        }
        return bases;
      };

  auto matchesTilePreservingFamily =
      [&](const LinearLayout &candidateLayout, unsigned tileM,
          unsigned tileN) -> bool {
        if (candidateLayout.getNumOutDims() != 2 ||
            candidateLayout.getOutDims() != normalizedLinear.getOutDims())
          return false;
        if (!normalizedLinear.hasInDim(kRow) || !normalizedLinear.hasInDim(kCol) ||
            !candidateLayout.hasInDim(kRow) || !candidateLayout.hasInDim(kCol))
          return false;

        auto checkDim = [&](StringAttr dim, unsigned tileSize) {
          unsigned tileBits = llvm::Log2_64(tileSize);
          unsigned candidateBits = candidateLayout.getInDimSizeLog2(dim);
          unsigned dimBits = normalizedLinear.getInDimSizeLog2(dim);
          if (tileBits > dimBits || tileBits > candidateBits)
            return false;
          auto canonicalBases = gatherTileBases(normalizedLinear, dim, tileBits);
          auto candidateBases = gatherTileBases(candidateLayout, dim, tileBits);
          if (canonicalBases != candidateBases)
            return false;
          for (unsigned i = tileBits; i < dimBits; ++i) {
            if (!basisIsWholeTileMultiple(normalizedLinear.getBasis(dim, i), tileM,
                                          tileN))
              return false;
          }
          return true;
        };

        if (!checkDim(kRow, tileM) || !checkDim(kCol, tileN))
          return false;

        for (StringAttr dim : normalizedLinear.getInDimNames()) {
          if (dim == kRow || dim == kCol)
            continue;
          unsigned dimBits = normalizedLinear.getInDimSizeLog2(dim);
          for (unsigned i = 0; i < dimBits; ++i) {
            if (!basisIsWholeTileMultiple(normalizedLinear.getBasis(dim, i), tileM,
                                          tileN))
              return false;
          }
        }
        return true;
      };

  std::optional<MMAv5TMemLayoutPlan> bestPlan;
  auto isBetterMatch = [&](const MMAv5TMemLayoutPlan &candidate) {
    if (!bestPlan)
      return true;
    auto candidateArea =
        static_cast<uint64_t>(candidate.instrShapeM) * candidate.instrShapeN;
    auto bestArea =
        static_cast<uint64_t>(bestPlan->instrShapeM) * bestPlan->instrShapeN;
    if (candidateArea != bestArea)
      return candidateArea > bestArea;
    if (candidate.instrShapeM != bestPlan->instrShapeM)
      return candidate.instrShapeM > bestPlan->instrShapeM;
    if (candidate.instrShapeN != bestPlan->instrShapeN)
      return candidate.instrShapeN > bestPlan->instrShapeN;
    if (preferredColStride) {
      bool candidateMatches = candidate.colStride == *preferredColStride;
      bool bestMatches = bestPlan->colStride == *preferredColStride;
      if (candidateMatches != bestMatches)
        return candidateMatches;
    }
    return candidate.colStride < bestPlan->colStride;
  };

  for (unsigned blockM : {64u, 128u}) {
    for (unsigned blockN : blockNs) {
      for (unsigned colStride : {1u, 2u, 4u}) {
        auto maybeCandidate = buildCanonicalLegacyLikeTMemLinearLayout(
            shape, blockM, blockN, colStride, cga, twoCTAs,
            /*error=*/nullptr);
        if (!maybeCandidate)
          continue;
        auto normalizedCandidate =
            normalizeTensorMemoryLinearLayoutForMMAv5Family(*maybeCandidate);
        if (!matchesTilePreservingFamily(normalizedCandidate, blockM, blockN))
          continue;
        MMAv5TMemLayoutPlan candidatePlan{
            /*instrShapeM=*/blockM,
            /*instrShapeN=*/blockN,
            /*colStride=*/colStride,
            /*twoCTAs=*/twoCTAs,
        };
        if (isBetterMatch(candidatePlan))
          bestPlan = candidatePlan;
      }
    }
  }

  return bestPlan;
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5Family(ArrayRef<int64_t> shape, Attribute layout,
                ArrayRef<unsigned> blockNs,
                std::optional<unsigned> preferredColStride = std::nullopt) {
  if (shape.size() != 2)
    return std::nullopt;

  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(shape, layout, /*error=*/nullptr);
  if (!maybeCanonical)
    return std::nullopt;
  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;
  return planMMAv5Family(shape, *maybeCanonical, gpu::getCGALayout(layout),
                         *twoCTAs, blockNs, preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5ExactFamily(ArrayRef<int64_t> shape, const LinearLayout &canonicalLayout,
                     gpu::CGAEncodingAttr cga, bool twoCTAs,
                     ArrayRef<unsigned> blockNs,
                     std::optional<unsigned> preferredColStride = std::nullopt) {
  if (shape.size() != 2)
    return std::nullopt;

  auto normalizedLinear =
      normalizeTensorMemoryLinearLayoutForMMAv5Family(canonicalLayout);

  std::optional<MMAv5TMemLayoutPlan> bestPlan;
  auto isBetterMatch = [&](const MMAv5TMemLayoutPlan &candidate) {
    if (!bestPlan)
      return true;
    auto candidateArea =
        static_cast<uint64_t>(candidate.instrShapeM) * candidate.instrShapeN;
    auto bestArea =
        static_cast<uint64_t>(bestPlan->instrShapeM) * bestPlan->instrShapeN;
    if (candidateArea != bestArea)
      return candidateArea > bestArea;
    if (candidate.instrShapeM != bestPlan->instrShapeM)
      return candidate.instrShapeM > bestPlan->instrShapeM;
    if (candidate.instrShapeN != bestPlan->instrShapeN)
      return candidate.instrShapeN > bestPlan->instrShapeN;
    if (preferredColStride) {
      bool candidateMatches = candidate.colStride == *preferredColStride;
      bool bestMatches = bestPlan->colStride == *preferredColStride;
      if (candidateMatches != bestMatches)
        return candidateMatches;
    }
    return candidate.colStride < bestPlan->colStride;
  };

  for (unsigned blockM : {64u, 128u}) {
    for (unsigned blockN : blockNs) {
      for (unsigned colStride : {1u, 2u, 4u}) {
        auto maybeCandidate = buildCanonicalLegacyLikeTMemLinearLayout(
            shape, blockM, blockN, colStride, cga, twoCTAs,
            /*error=*/nullptr);
        if (!maybeCandidate)
          continue;
        auto normalizedCandidate =
            normalizeTensorMemoryLinearLayoutForMMAv5Family(*maybeCandidate);
        if (normalizedCandidate != normalizedLinear)
          continue;
        MMAv5TMemLayoutPlan candidatePlan{
            /*instrShapeM=*/blockM,
            /*instrShapeN=*/blockN,
            /*colStride=*/colStride,
            /*twoCTAs=*/twoCTAs,
        };
        if (isBetterMatch(candidatePlan))
          bestPlan = candidatePlan;
      }
    }
  }
  return bestPlan;
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5ExactFamily(ArrayRef<int64_t> shape, Attribute layout,
                     ArrayRef<unsigned> blockNs,
                     std::optional<unsigned> preferredColStride = std::nullopt) {
  if (shape.size() != 2)
    return std::nullopt;

  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(shape, layout, /*error=*/nullptr);
  if (!maybeCanonical)
    return std::nullopt;
  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;
  return planMMAv5ExactFamily(shape, *maybeCanonical, gpu::getCGALayout(layout),
                              *twoCTAs, blockNs, preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5AccumulatorFamily(ArrayRef<int64_t> shape, Attribute layout,
                          std::optional<unsigned> preferredColStride =
                              std::nullopt) {
  static constexpr unsigned kAccumulatorBlockNs[] = {32u, 64u, 128u, 256u};
  if (auto exact = planMMAv5ExactFamily(shape, layout, kAccumulatorBlockNs,
                                        preferredColStride)) {
    return exact;
  }
  return planMMAv5Family(shape, layout, kAccumulatorBlockNs,
                         preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5AccumulatorFamily(ArrayRef<int64_t> shape,
                           const LinearLayout &canonicalLayout,
                           gpu::CGAEncodingAttr cga, bool twoCTAs,
                           std::optional<unsigned> preferredColStride =
                               std::nullopt) {
  static constexpr unsigned kAccumulatorBlockNs[] = {32u, 64u, 128u, 256u};
  if (auto exact = planMMAv5ExactFamily(shape, canonicalLayout, cga, twoCTAs,
                                        kAccumulatorBlockNs,
                                        preferredColStride)) {
    return exact;
  }
  return planMMAv5Family(shape, canonicalLayout, cga, twoCTAs,
                         kAccumulatorBlockNs, preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5LhsFamily(ArrayRef<int64_t> shape, Attribute layout,
                   std::optional<unsigned> preferredColStride = std::nullopt) {
  static constexpr unsigned kLhsBlockNs[] = {32u, 64u, 128u, 256u};
  if (auto exact = planMMAv5ExactFamily(shape, layout, kLhsBlockNs,
                                        preferredColStride)) {
    return exact;
  }
  return planMMAv5Family(shape, layout, kLhsBlockNs, preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5LhsFamily(ArrayRef<int64_t> shape, const LinearLayout &canonicalLayout,
                   gpu::CGAEncodingAttr cga, bool twoCTAs,
                   std::optional<unsigned> preferredColStride = std::nullopt) {
  static constexpr unsigned kLhsBlockNs[] = {32u, 64u, 128u, 256u};
  if (auto exact = planMMAv5ExactFamily(shape, canonicalLayout, cga, twoCTAs,
                                        kLhsBlockNs, preferredColStride)) {
    return exact;
  }
  return planMMAv5Family(shape, canonicalLayout, cga, twoCTAs, kLhsBlockNs,
                         preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5ScaledAccumulatorFamily(ArrayRef<int64_t> shape, Attribute layout,
                                 std::optional<unsigned> preferredColStride =
                                     std::nullopt) {
  static constexpr unsigned kAccumulatorBlockNs[] = {32u, 64u, 128u, 256u};
  if (auto exact = planMMAv5ExactFamily(shape, layout, kAccumulatorBlockNs,
                                        preferredColStride)) {
    return exact;
  }
  return planMMAv5Family(shape, layout, kAccumulatorBlockNs,
                         preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
planMMAv5ScaledAccumulatorFamily(ArrayRef<int64_t> shape,
                                 const LinearLayout &canonicalLayout,
                                 gpu::CGAEncodingAttr cga, bool twoCTAs,
                                 std::optional<unsigned> preferredColStride =
                                     std::nullopt) {
  static constexpr unsigned kAccumulatorBlockNs[] = {32u, 64u, 128u, 256u};
  if (auto exact = planMMAv5ExactFamily(shape, canonicalLayout, cga, twoCTAs,
                                        kAccumulatorBlockNs,
                                        preferredColStride)) {
    return exact;
  }
  return planMMAv5Family(shape, canonicalLayout, cga, twoCTAs,
                         kAccumulatorBlockNs, preferredColStride);
}

using MMAv5FamilyPlanner = std::optional<MMAv5TMemLayoutPlan> (*)(
    ArrayRef<int64_t>, Attribute, std::optional<unsigned>);
using MMAv5LinearFamilyPlanner = std::optional<MMAv5TMemLayoutPlan> (*)(
    ArrayRef<int64_t>, const LinearLayout &, gpu::CGAEncodingAttr, bool,
    std::optional<unsigned>);

static std::optional<LinearLayout>
buildMMAv5FamilyLayout(ArrayRef<int64_t> shape, gpu::CGAEncodingAttr cga,
                       const MMAv5TMemLayoutPlan &plan) {
  return buildCanonicalLegacyLikeTMemLinearLayout(
      shape, plan.instrShapeM, plan.instrShapeN, plan.colStride, cga,
      plan.twoCTAs, /*error=*/nullptr);
}

static std::optional<MMAv5TMemLayoutPlan>
getMMAv5AccumulatorLikeLayoutPlan(MemDescType memDescType,
                                  MMAv5FamilyPlanner planner,
                                  MMAv5LinearFamilyPlanner linearPlanner) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  unsigned preferredColStride = 32 / memDescType.getElementTypeBitWidth();
  if (auto plan = planner(shape, layout, preferredColStride))
    return plan;

  auto allocShape = memDescType.getAllocShape().take_back(rank);
  if (shape.size() != 2 || allocShape.size() != shape.size())
    return std::nullopt;

  // Accumulator subviews remain compatible when they preserve the full M
  // extent of the underlying allocation and only narrow N.
  if (shape == allocShape || shape[0] != allocShape[0])
    return std::nullopt;

  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;
  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(allocShape, layout,
                                              /*error=*/nullptr);
  if (!maybeCanonical ||
      !tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, allocShape)) {
    return std::nullopt;
  }

  return linearPlanner(allocShape, *maybeCanonical, gpu::getCGALayout(layout),
                       *twoCTAs, preferredColStride);
}

static std::optional<MMAv5TMemLayoutPlan>
getMMAv5LhsLikeLayoutPlan(MemDescType memDescType) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  constexpr unsigned kPreferredColStride = 1u;
  if (auto plan =
          planMMAv5LhsFamily(shape, layout, /*preferredColStride=*/kPreferredColStride))
    return plan;

  auto allocShape = memDescType.getAllocShape().take_back(rank);
  if (shape.size() != 2 || allocShape.size() != shape.size())
    return std::nullopt;

  // LHS subviews remain MMAv5-compatible when they preserve the full M extent
  // of the allocation and only narrow K. Lowering already uses the memdesc
  // view to compute tile addresses, so verifier planning only needs the
  // underlying MMAv5 family properties.
  if (shape == allocShape || shape[0] != allocShape[0])
    return std::nullopt;

  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;
  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(allocShape, layout,
                                              /*error=*/nullptr);
  if (!maybeCanonical ||
      !tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, allocShape)) {
    return std::nullopt;
  }

  return planMMAv5LhsFamily(allocShape, *maybeCanonical,
                            gpu::getCGALayout(layout), *twoCTAs,
                            kPreferredColStride);
}

static std::optional<MMAv5LhsLayoutInfo>
getMMAv5LhsLikeLayoutInfo(MemDescType memDescType) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  auto allocShape = memDescType.getAllocShape().take_back(rank);
  constexpr unsigned kPreferredColStride = 1u;
  auto makeInfo = [&](ArrayRef<int64_t> familyShape,
                      const MMAv5TMemLayoutPlan &plan)
      -> std::optional<MMAv5LhsLayoutInfo> {
    auto maybeFamilyLayout =
        buildMMAv5FamilyLayout(familyShape, gpu::getCGALayout(layout), plan);
    if (!maybeFamilyLayout)
      return std::nullopt;
    return MMAv5LhsLayoutInfo{
        /*familyLayout=*/std::move(*maybeFamilyLayout),
        /*mmaSizeM=*/plan.instrShapeM,
        /*mmaSizeN=*/plan.instrShapeN,
        /*colStride=*/plan.colStride,
        /*twoCTAs=*/plan.twoCTAs,
    };
  };

  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(shape, layout, /*error=*/nullptr);
  if (maybeCanonical &&
      tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, shape)) {
    if (auto plan = planMMAv5LhsFamily(shape, layout, kPreferredColStride))
      return makeInfo(shape, *plan);
  }

  if (shape.size() != 2 || allocShape.size() != shape.size())
    return std::nullopt;

  if (shape == allocShape || shape[0] != allocShape[0])
    return std::nullopt;

  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;
  maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(allocShape, layout,
                                              /*error=*/nullptr);
  if (!maybeCanonical ||
      !tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, allocShape)) {
    return std::nullopt;
  }

  if (auto plan = planMMAv5LhsFamily(allocShape, *maybeCanonical,
                                     gpu::getCGALayout(layout), *twoCTAs,
                                     kPreferredColStride)) {
    return makeInfo(allocShape, *plan);
  }
  return std::nullopt;
}

static std::optional<MMAv5AccumulatorLayoutInfo>
getMMAv5AccumulatorLikeLayoutInfo(MemDescType memDescType,
                                  MMAv5FamilyPlanner planner,
                                  MMAv5LinearFamilyPlanner linearPlanner) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  auto allocShape = memDescType.getAllocShape().take_back(rank);
  unsigned preferredColStride = 32 / memDescType.getElementTypeBitWidth();
  auto makeInfo = [&](ArrayRef<int64_t> familyShape,
                      const MMAv5TMemLayoutPlan &plan)
      -> std::optional<MMAv5AccumulatorLayoutInfo> {
    auto maybeFamilyLayout =
        buildMMAv5FamilyLayout(familyShape, gpu::getCGALayout(layout), plan);
    if (!maybeFamilyLayout)
      return std::nullopt;
    return MMAv5AccumulatorLayoutInfo{
        /*familyLayout=*/std::move(*maybeFamilyLayout),
        /*mmaSizeM=*/plan.instrShapeM,
        /*mmaSizeN=*/plan.instrShapeN,
        /*colStride=*/plan.colStride,
        /*twoCTAs=*/plan.twoCTAs,
        /*interleavedM64=*/getTmemAllocSizes(memDescType).numRows != 64 &&
            plan.instrShapeM == 64,
    };
  };

  auto maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(shape, layout, /*error=*/nullptr);
  if (maybeCanonical && tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, shape)) {
    if (auto plan = planner(shape, layout, preferredColStride))
      return makeInfo(shape, *plan);
  }

  if (shape.size() != 2 || allocShape.size() != shape.size())
    return std::nullopt;

  if (shape == allocShape || shape[0] != allocShape[0])
    return std::nullopt;

  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;
  maybeCanonical =
      tryGetCanonicalTensorMemoryLinearLayout(allocShape, layout,
                                              /*error=*/nullptr);
  if (!maybeCanonical ||
      !tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, allocShape)) {
    return std::nullopt;
  }

  if (auto plan = linearPlanner(allocShape, *maybeCanonical,
                                gpu::getCGALayout(layout), *twoCTAs,
                                preferredColStride)) {
    return makeInfo(allocShape, *plan);
  }
  return std::nullopt;
}

std::optional<MMAv5LhsLayoutInfo>
getMMAv5LhsLayoutInfo(MemDescType memDescType) {
  return getMMAv5LhsLikeLayoutInfo(memDescType);
}

std::optional<MMAv5AccumulatorLayoutInfo>
getMMAv5AccumulatorLayoutInfo(MemDescType memDescType) {
  return getMMAv5AccumulatorLikeLayoutInfo(memDescType,
                                           planMMAv5AccumulatorFamily,
                                           planMMAv5AccumulatorFamily);
}

std::optional<MMAv5AccumulatorLayoutInfo>
getMMAv5ScaledAccumulatorLayoutInfo(MemDescType memDescType) {
  return getMMAv5AccumulatorLikeLayoutInfo(memDescType,
                                           planMMAv5ScaledAccumulatorFamily,
                                           planMMAv5ScaledAccumulatorFamily);
}

static int64_t linearizePrefixOffsets(ArrayRef<int64_t> shape,
                                      ArrayRef<int32_t> offsets) {
  assert(shape.size() == offsets.size());
  int64_t linearized = 0;
  int64_t stride = 1;
  for (auto [size, offset] :
       llvm::reverse(llvm::zip_equal(shape, offsets))) {
    linearized += static_cast<int64_t>(offset) * stride;
    stride *= size;
  }
  return linearized;
}

TMemAllocation getTmemAllocSizes(MemDescType memDescType) {
  auto *ctx = memDescType.getContext();
  auto S = [&](StringRef str) { return StringAttr::get(ctx, str); };
  auto kRow = S("row");
  auto kCol = S("col");
  auto leafShape = memDescType.getShape().take_back(2);
  auto encoding = memDescType.getEncoding();
  bool isLegacyLike = isa<TensorMemoryEncodingAttr, TensorMemoryScalesEncodingAttr>(encoding);
  auto ll = [&]() {
    if (isLegacyLike)
      return triton::gpu::toLinearLayout(leafShape, encoding);
    if (auto canonical =
            getCanonicalTMemLinearEncoding(leafShape, encoding,
                                           /*error=*/nullptr))
      return canonical->getLinearLayout();
    return triton::gpu::toLinearLayout(leafShape, encoding);
  }();
  auto extraRank = memDescType.getRank() - 2;
  auto bitwidth = memDescType.getElementTypeBitWidth();
  unsigned preferredColStride = 32 / bitwidth;
  int nRow = ll.getInDimSize(kRow);
  int nCol = ll.getInDimSize(kCol) / preferredColStride;
  // If we have just one 16xcol block per warp, we don't allocate 128 rows
  // we use 64 rows instead.
  // We could generalise this to when we have more zeros in the layout, but
  // the allocator does not support this yet
  if (ll.getInDimSize(kRow) > 16 &&
      llvm::all_of(ll.getBasis(kRow, llvm::Log2_32(16)),
                   [](int32_t value) { return value == 0; })) {
    nRow /= 2;
  }
  // If multibuffering is present, we need to allocate more cols
  if (extraRank > 0) {
    nCol *= product<int64_t>(
        memDescType.getAllocShape().take_front(extraRank));
  }
  return {nRow, nCol};
}

uint32_t getTMemSubSliceOffset(MemDescType memDescType, int32_t nOffset) {
  SmallVector<int32_t> offsets(memDescType.getRank(), 0);
  offsets.back() = nOffset;
  return getTMemViewOffset(memDescType, offsets);
}

static uint32_t getTMemViewOffsetImpl(const LinearLayout &ll, unsigned memRank,
                                      ArrayRef<int32_t> offsets,
                                      uint32_t bitwidth,
                                      ArrayRef<int64_t> prefixShape) {
  assert(offsets.size() == memRank);
  auto *ctx = (*ll.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  unsigned layoutRank = ll.getNumOutDims();
  auto outDimNames = llvm::to_vector(ll.getOutDimNames());
  if (layoutRank > memRank) {
    outDimNames.erase(outDimNames.begin(),
                      outDimNames.begin() + (layoutRank - memRank));
    layoutRank = memRank;
  }
  unsigned extraRank = memRank - layoutRank;

  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  logicalOffsets.reserve(layoutRank);
  for (auto [dim, offset] :
       llvm::zip_equal(outDimNames, offsets.drop_front(extraRank))) {
    logicalOffsets.push_back({dim, offset});
  }

  auto rowColBlock = ll.pseudoinvert().apply(logicalOffsets);
  uint32_t offsetRow = 0;
  uint32_t offsetCol = 0;
  for (auto [dim, value] : rowColBlock) {
    if (dim == kRow) {
      offsetRow = value;
    } else if (dim == kCol) {
      offsetCol = value * bitwidth / 32;
    }
  }
  if (extraRank > 0) {
    assert(prefixShape.size() == extraRank &&
           "prefix shape is required when the logical rank exceeds the layout rank");
    auto singleBufferCols = ll.getInDimSize(kCol) / (32 / bitwidth);
    offsetCol += linearizePrefixOffsets(prefixShape, offsets.take_front(extraRank)) *
                 singleBufferCols;
  }
  return offsetCol | offsetRow << 16;
}

uint32_t getTMemViewOffset(const LinearLayout &layout,
                           ArrayRef<int32_t> offsets, uint32_t bitwidth,
                           ArrayRef<int64_t> prefixShape) {
  return getTMemViewOffsetImpl(layout, layout.getNumOutDims(), offsets,
                               bitwidth, prefixShape);
}

uint32_t getTMemViewOffset(MemDescType memDescType, ArrayRef<int32_t> offsets) {
  LinearLayout ll = [&]() {
    if (isTensorMemoryEncoding(memDescType.getEncoding()) &&
        !isa<TensorMemoryScalesEncodingAttr>(memDescType.getEncoding())) {
      std::string error;
      if (auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
              memDescType.getShape(), memDescType.getEncoding(), &error)) {
        return normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis);
      }
    }
    return normalizeTensorMemoryLinearLayoutForAnalysis(
        triton::gpu::toLinearLayout(memDescType));
  }();
  return getTMemViewOffsetImpl(ll, memDescType.getRank(), offsets,
                               memDescType.getElementTypeBitWidth(),
                               memDescType.getShape().take_front(
                                   memDescType.getRank() > ll.getNumOutDims()
                                       ? memDescType.getRank() -
                                             ll.getNumOutDims()
                                       : 0));
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp, ArrayRef<int32_t> warpBasis0,
                           ArrayRef<int32_t> warpBasis1, int32_t rowSpan) {
  auto str_attr = [&](StringRef str) { return StringAttr::get(ctx, str); };
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");
  // Set the output order to be kRow, kCol and the input order to be kReg first
  LinearLayout tile = LinearLayout({{kReg, {}}, {kLane, {}}}, {kRow, kCol});
  // Each register moves 32/bitwidth (= 2) columns when unpacked
  if (unpacked) {
    tile *= LinearLayout::zeros1D(1, kReg, kCol, 2);
  }
  if (atom == TMemAccessAtom::I32x32b) {
    tile *= LinearLayout::identity1D(32, kLane, kRow);
  } else if (atom == TMemAccessAtom::I16x32bx2) {
    tile *= LinearLayout::identity1D(16, kLane, kRow);
  } else if (atom == TMemAccessAtom::I16x64b) {
    LinearLayout::BasesT bases;
    bases[kLane] = std::vector<std::vector<int32_t>>{
        {8, 0}, {0, 1}, {1, 0}, {2, 0}, {4, 0}};
    tile *= LinearLayout(std::move(bases), {kRow, kCol});
  } else if (atom == TMemAccessAtom::I16x128b) {
    tile *= LinearLayout::identity1D(4, kLane, kCol) *
            LinearLayout::identity1D(8, kLane, kRow) *
            LinearLayout::identity1D(2, kReg, kRow);
  } else if (atom == TMemAccessAtom::I16x256b) {
    tile *= LinearLayout::identity1D(2, kReg, kCol) *
            LinearLayout::identity1D(4, kLane, kCol) *
            LinearLayout::identity1D(8, kLane, kRow) *
            LinearLayout::identity1D(2, kReg, kRow);
  } else {
    llvm_unreachable("Unsupported TMEM access atom");
  }
  if (withWarp) {
    auto nRow = tile.getOutDimSize(kRow);
    auto nCol = tile.getOutDimSize(kCol);
    int32_t rowExtent = rowSpan;
    int32_t colSpan = nCol;
    auto updateRowExtent = [&](ArrayRef<int32_t> basis) {
      assert(basis.size() == 2 && "TMEM warp bases must be 2D row/col vectors");
      rowExtent = std::max<int32_t>(rowExtent, basis[0] + nRow);
    };
    auto updateColSpan = [&](ArrayRef<int32_t> basis) {
      assert(basis.size() == 2 && "TMEM warp bases must be 2D row/col vectors");
      colSpan = std::max<int32_t>(colSpan, basis[1] + nCol);
    };
    updateRowExtent(warpBasis0);
    updateRowExtent(warpBasis1);
    updateColSpan(warpBasis0);
    updateColSpan(warpBasis1);
    if (!llvm::isPowerOf2_32(static_cast<uint32_t>(rowExtent)))
      rowExtent = llvm::PowerOf2Ceil(static_cast<uint32_t>(rowExtent));
    if (!llvm::isPowerOf2_32(static_cast<uint32_t>(colSpan)))
      colSpan = llvm::PowerOf2Ceil(static_cast<uint32_t>(colSpan));
    auto bases = tile.getBases();
    bases[kWarp].push_back({warpBasis0.begin(), warpBasis0.end()});
    bases[kWarp].push_back({warpBasis1.begin(), warpBasis1.end()});
    tile = LinearLayout(std::move(bases), {{kRow, rowExtent}, {kCol, colSpan}},
                        false);
  }
  return tile;
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp, int32_t warpRow0,
                           int32_t warpRow1, int32_t rowSpan) {
  SmallVector<int32_t> warpBasis0 = {warpRow0, 0};
  SmallVector<int32_t> warpBasis1 = {warpRow1, 0};
  return getTileLayout(ctx, atom, unpacked, withWarp, warpBasis0, warpBasis1,
                       rowSpan);
}

static bool canComposeLinearLayouts(const LinearLayout &inner,
                                    const LinearLayout &outer) {
  for (StringAttr outDim : inner.getOutDimNames()) {
    if (inner.getOutDimSize(outDim) > outer.getInDimSize(outDim))
      return false;
  }
  return true;
}

static LinearLayout
stripZeroBasesForTmemLdStSelection(LinearLayout ll);

std::optional<TMemLdStRowPlan> getTMemLdStRowPlan(const LinearLayout &ll) {
  if (ll.getNumInDims() == 0)
    return std::nullopt;
  auto *ctx = (*ll.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  if (!ll.hasInDim(kRow))
    return std::nullopt;
  auto stripped = ll.removeZeroBasesAlongDim(kRow);
  if (stripped.hasInDim(kRow) &&
      stripped.getInDimSizeLog2(kRow) < ll.getInDimSizeLog2(kRow) &&
      stripped.getInDimSizeLog2(kRow) >= 6)
    return getTMemLdStRowPlan(stripped);
  unsigned rowBits = ll.getInDimSizeLog2(kRow);
  auto isZeroRowBasis = [&](unsigned idx) {
    return idx < rowBits && llvm::all_of(ll.getBasis(kRow, idx), [](int32_t v) {
      return v == 0;
    });
  };
  if (rowBits >= 7) {
    if (isZeroRowBasis(rowBits - 2) && isZeroRowBasis(rowBits - 1)) {
      return TMemLdStRowPlan{/*warpRow0=*/0, /*warpRow1=*/0,
                             /*rowSpan=*/128};
    }
    return TMemLdStRowPlan{/*warpRow0=*/32, /*warpRow1=*/64,
                           /*rowSpan=*/128};
  }
  if (rowBits == 6) {
    if (isZeroRowBasis(rowBits - 2) && isZeroRowBasis(rowBits - 1)) {
      return TMemLdStRowPlan{/*warpRow0=*/0, /*warpRow1=*/0,
                             /*rowSpan=*/64};
    }
    return TMemLdStRowPlan{/*warpRow0=*/16, /*warpRow1=*/32,
                           /*rowSpan=*/64};
  }
  return std::nullopt;
}

std::optional<LinearLayout>
getCanonicalM64SplitNLayout(MLIRContext *ctx, int64_t n, unsigned numWarps) {
  if (n < 2 || !llvm::isPowerOf2_64(n) || (numWarps != 4 && numWarps != 8))
    return std::nullopt;
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kDim0 = StringAttr::get(ctx, "dim0");
  auto kDim1 = StringAttr::get(ctx, "dim1");

  SmallVector<std::vector<int32_t>> regBases;
  SmallVector<std::vector<int32_t>> laneBases = {
      {1, 0}, {2, 0}, {4, 0}, {8, 0}};
  SmallVector<std::vector<int32_t>> warpBases;
  if (numWarps == 4) {
    int64_t laneSplitCol = n >= 4 ? n / 4 : 0;
    laneBases.push_back({0, static_cast<int32_t>(laneSplitCol)});
    for (int64_t col = 1; col < n; col <<= 1) {
      if (col == laneSplitCol)
        continue;
      regBases.push_back({0, static_cast<int32_t>(col)});
    }
    warpBases = {{16, 0}, {32, 0}};
  } else {
    int64_t warpSplitCol = n >= 4 ? n / 4 : 0;
    int64_t laneSplitCol = n >= 2 ? n / 2 : 0;
    laneBases.push_back({0, static_cast<int32_t>(laneSplitCol)});
    for (int64_t col = 1; col < warpSplitCol; col <<= 1)
      regBases.push_back({0, static_cast<int32_t>(col)});
    warpBases = {{16, 0}, {32, 0}, {0, static_cast<int32_t>(warpSplitCol)}};
  }

  SmallVector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases = {
      {kReg, std::vector<std::vector<int32_t>>(regBases.begin(), regBases.end())},
      {kLane,
       std::vector<std::vector<int32_t>>(laneBases.begin(), laneBases.end())},
      {kWarp,
       std::vector<std::vector<int32_t>>(warpBases.begin(), warpBases.end())},
  };
  LinearLayout layout(bases, {{kDim0, 64}, {kDim1, static_cast<int32_t>(n)}},
                      /*requireSurjective=*/false);
  return layout;
}

std::optional<LinearLayout>
getCanonicalM64SplitNLayout(MemDescType memType, unsigned numWarps) {
  if (memType.getRank() != 2 || memType.getShape()[0] != 64)
    return std::nullopt;
  return getCanonicalM64SplitNLayout(memType.getContext(), memType.getShape()[1],
                                     numWarps);
}

static bool matchesCanonicalContiguousM64LinearView(const LinearLayout &ll) {
  if (ll.getNumOutDims() != 2 || ll.getNumInDims() != 2)
    return false;
  auto *ctx = (*ll.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) || ll.getInDimSize(kRow) != 64)
    return false;
  int64_t n = ll.getInDimSize(kCol);
  if (n < 2 || !llvm::isPowerOf2_64(n))
    return false;
  for (int bit = 0; bit < 6; ++bit) {
    if (ll.getBasis(kRow, bit) != ArrayRef<int32_t>{1 << bit, 0})
      return false;
  }
  for (int bit = 0; (1ll << bit) < n; ++bit) {
    if (ll.getBasis(kCol, bit) != ArrayRef<int32_t>{0, 1 << bit})
      return false;
  }
  return true;
}

static bool matchesSimplePermutedM64SplitNLinearView(const LinearLayout &ll) {
  if (ll.getNumOutDims() != 2 || ll.getNumInDims() != 2)
    return false;
  auto *ctx = (*ll.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) ||
      ll.getInDimSize(kRow) != 128)
    return false;

  auto outDims = to_vector(ll.getOutDimNames());
  int64_t n = ll.getInDimSize(kCol);
  if (n < 2 || !llvm::isPowerOf2_64(n) ||
      ll.getOutDimSize(outDims[0]) != 64 ||
      ll.getOutDimSize(outDims[1]) != n)
    return false;

  std::array<bool, 6> seenRows = {};
  unsigned zeroRows = 0;
  for (unsigned bit = 0; bit < ll.getInDimSizeLog2(kRow); ++bit) {
    auto basis = ll.getBasis(kRow, bit);
    if (basis.size() != 2 || basis[1] != 0)
      return false;
    if (basis[0] == 0) {
      ++zeroRows;
      continue;
    }
    if (basis[0] < 0 || basis[0] > 32 ||
        !llvm::isPowerOf2_32(static_cast<uint32_t>(basis[0])))
      return false;
    unsigned rowBit = llvm::Log2_32(static_cast<uint32_t>(basis[0]));
    if (rowBit >= seenRows.size() || seenRows[rowBit])
      return false;
    seenRows[rowBit] = true;
  }
  if (zeroRows != 1 || !llvm::all_of(seenRows, [](bool seen) { return seen; }))
    return false;

  SmallVector<bool> seenCols(ll.getInDimSizeLog2(kCol), false);
  for (unsigned bit = 0; bit < ll.getInDimSizeLog2(kCol); ++bit) {
    auto basis = ll.getBasis(kCol, bit);
    if (basis.size() != 2 || basis[0] != 0 || basis[1] <= 0 ||
        basis[1] >= n ||
        !llvm::isPowerOf2_32(static_cast<uint32_t>(basis[1])))
      return false;
    unsigned colBit = llvm::Log2_32(static_cast<uint32_t>(basis[1]));
    if (colBit >= seenCols.size() || seenCols[colBit])
      return false;
    seenCols[colBit] = true;
  }
  return llvm::all_of(seenCols, [](bool seen) { return seen; });
}

static std::optional<LinearLayout>
getCanonicalContiguousM64Layout(MLIRContext *ctx, TMemAccessAtom atom, int64_t n,
                                unsigned numWarps) {
  if (n < 2 || !llvm::isPowerOf2_64(n) || (numWarps != 4 && numWarps != 8))
    return std::nullopt;
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kDim0 = StringAttr::get(ctx, "dim0");
  auto kDim1 = StringAttr::get(ctx, "dim1");

  SmallVector<std::vector<int32_t>> regBases;
  SmallVector<std::vector<int32_t>> laneBases;
  SmallVector<std::vector<int32_t>> warpBases;

  switch (atom) {
  case TMemAccessAtom::I32x32b:
    laneBases = {{1, 0}, {2, 0}, {4, 0}, {8, 0}};
    if (numWarps == 4) {
      // Keep the canonical 4-warp M64 layout on the full x32 register path.
      // Using n/4 here collapses ordinary 64x64 MMAv5 accumulators to the
      // narrower split-N x16 ld/st path and regresses numerical results.
      int64_t laneSplitCol = n >= 2 ? n / 2 : 0;
      laneBases.push_back({0, static_cast<int32_t>(laneSplitCol)});
      for (int64_t col = 1; col < n; col <<= 1) {
        if (col == laneSplitCol)
          continue;
        regBases.push_back({0, static_cast<int32_t>(col)});
      }
      warpBases = {{16, 0}, {32, 0}};
    } else {
      int64_t warpSplitCol = n >= 4 ? n / 4 : 0;
      int64_t laneSplitCol = n >= 2 ? n / 2 : 0;
      laneBases.push_back({0, static_cast<int32_t>(laneSplitCol)});
      for (int64_t col = 1; col < warpSplitCol; col <<= 1)
        regBases.push_back({0, static_cast<int32_t>(col)});
      warpBases = {{16, 0}, {32, 0}, {0, static_cast<int32_t>(warpSplitCol)}};
    }
    break;
  case TMemAccessAtom::I16x64b:
    laneBases = {{8, 0}, {0, 1}, {1, 0}, {2, 0}, {4, 0}};
    for (int64_t col = 2; col <= (numWarps == 4 ? n / 2 : n / 4); col <<= 1)
      regBases.push_back({0, static_cast<int32_t>(col)});
    warpBases = {{16, 0}, {32, 0}};
    if (numWarps == 8)
      warpBases.push_back({0, static_cast<int32_t>(n / 2)});
    break;
  case TMemAccessAtom::I16x128b:
    if (n < 4)
      return std::nullopt;
    laneBases = {{0, 1}, {0, 2}, {1, 0}, {2, 0}, {4, 0}};
    regBases.push_back({8, 0});
    for (int64_t col = 4; col <= (numWarps == 4 ? n / 2 : n / 4); col <<= 1)
      regBases.push_back({0, static_cast<int32_t>(col)});
    warpBases = {{16, 0}, {32, 0}};
    if (numWarps == 8)
      warpBases.push_back({0, static_cast<int32_t>(n / 2)});
    break;
  case TMemAccessAtom::I16x256b:
    if (n < 8)
      return std::nullopt;
    // Keep the legacy M64 family for x256 so the direct ld/st matcher reaches
    // the native 16x256b path instead of degrading to x128.
    laneBases = {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}};
    regBases = {{0, 1}, {8, 0}};
    for (int64_t col = 8; col <= (numWarps == 4 ? n / 2 : n / 4); col <<= 1)
      regBases.push_back({0, static_cast<int32_t>(col)});
    warpBases = {{16, 0}, {32, 0}};
    if (numWarps == 8) {
      // The legacy 8-warp x256 family only shards N across the extra warp bit
      // once there is at least one full 8-column packet to place there. For
      // narrow N, keep the third warp basis zero instead of synthesizing an
      // invalid partial split.
      warpBases.push_back({0, static_cast<int32_t>(n >= 16 ? n / 2 : 0)});
    }
    break;
  case TMemAccessAtom::I16x32bx2:
    return std::nullopt;
  }

  SmallVector<std::pair<StringAttr, std::vector<std::vector<int32_t>>>> bases = {
      {kReg, std::vector<std::vector<int32_t>>(regBases.begin(), regBases.end())},
      {kLane,
       std::vector<std::vector<int32_t>>(laneBases.begin(), laneBases.end())},
      {kWarp,
       std::vector<std::vector<int32_t>>(warpBases.begin(), warpBases.end())},
  };
  return LinearLayout(bases, {{kDim0, 64}, {kDim1, static_cast<int32_t>(n)}},
                      /*requireSurjective=*/false);
}

static std::optional<LinearLayout>
getTMemLdStSplitNLayout(const LinearLayout &ll, unsigned numWarps,
                        const TMemLdStRowPlan &rowPlan) {
  if (numWarps != 4)
    return std::nullopt;
  auto dims = to_vector(ll.getOutDimNames());
  if (dims.size() != 2)
    return std::nullopt;
  auto rowColDims = to_vector(ll.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");

  auto hasZeroBasisAlong = [&](StringAttr dim) {
    if (!ll.hasInDim(dim))
      return false;
    for (unsigned idx = 0; idx < ll.getInDimSizeLog2(dim); ++idx) {
      if (llvm::all_of(ll.getBasis(dim, idx),
                       [](int32_t value) { return value == 0; })) {
        return true;
      }
    }
    return false;
  };
  bool isSourceBackedM64SplitNSupport =
      rowPlan.rowSpan == 128 && ll.hasOutDim(dims[0]) &&
      ll.getOutDimSize(dims[0]) == 64 && hasZeroBasisAlong(kRow);
  if (rowPlan.rowSpan != 64 && !isSourceBackedM64SplitNSupport)
    return std::nullopt;

  bool hasBlockDim = llvm::is_contained(rowColDims, kBlock);
  if (hasBlockDim && ll.getInDimSize(kBlock) > 1) {
    auto ctasPerCGA =
        ::mlir::triton::basesPerDimImpl(ll.getBases(), kBlock, dims.size(),
                                        /*skipBroadcast=*/false);
    auto ctaSplitNum =
        ::mlir::triton::basesPerDimImpl(ll.getBases(), kBlock, dims.size(),
                                        /*skipBroadcast=*/true);
    SmallVector<unsigned> defaultOrder(dims.size());
    std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
    auto ctaOrder = ::mlir::triton::orderPerDimImpl(ll, kBlock, defaultOrder);
    auto blockOnly =
        gpu::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA, ctaSplitNum,
                                              ctaOrder)
            .getLinearLayout();
    if (auto maybePerCTA = divideRight(ll, blockOnly)) {
      if (auto perCTA =
              getTMemLdStSplitNLayout(*maybePerCTA, numWarps, rowPlan)) {
        return *perCTA * blockOnly;
      }
    }
    return std::nullopt;
  }

  int64_t nCols = ll.getInDimSize(rowColDims[1]);
  if (nCols < 2 || !llvm::isPowerOf2_64(nCols))
    return std::nullopt;

  LinearLayout::BasesT bases;
  auto &laneBases = bases[kLane];
  laneBases = {{1, 0}, {2, 0}, {4, 0}, {8, 0}};
  int64_t laneSplitCol = nCols >= 4 ? nCols / 4 : 0;
  laneBases.push_back({0, static_cast<int32_t>(laneSplitCol)});

  auto &regBases = bases[kReg];
  for (int64_t col = 1; col < nCols; col <<= 1) {
    if (col == laneSplitCol)
      continue;
    regBases.push_back({0, static_cast<int32_t>(col)});
  }

  bases[kWarp] = {
      {rowPlan.warpRow0, 0},
      {rowPlan.warpRow1, 0},
  };

  int32_t rowExtent = std::max<int32_t>(rowPlan.rowSpan, 16);
  rowExtent = std::max<int32_t>(rowExtent, rowPlan.warpRow0 + 16);
  rowExtent = std::max<int32_t>(rowExtent, rowPlan.warpRow1 + 16);
  if (!llvm::isPowerOf2_32(static_cast<uint32_t>(rowExtent)))
    rowExtent = llvm::PowerOf2Ceil(static_cast<uint32_t>(rowExtent));
  LinearLayout tile(std::move(bases),
                    {{rowColDims[0], rowExtent}, {rowColDims[1], nCols}},
                    /*requireSurjective=*/false);
  if (hasBlockDim) {
    int64_t nCTAs = ll.getInDimSize(kBlock);
    tile *= LinearLayout::identity1D(nCTAs, kBlock, kBlock);
  }
  if (!canComposeLinearLayouts(tile, ll))
    return std::nullopt;

  auto ret = tile.compose(ll);
  SmallVector<StringAttr> canonicalInDims;
  for (StringAttr dim : {kReg, kLane, kWarp, kBlock}) {
    if (ret.hasInDim(dim))
      canonicalInDims.push_back(dim);
  }
  if (!canonicalInDims.empty())
    ret = ret.transposeIns(canonicalInDims);
  auto withoutBroadcast = ret;
  for (auto inDim : ret.getInDimNames())
    withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
  if (!withoutBroadcast.isInvertible())
    return std::nullopt;
  return ret;
}

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(const LinearLayout &ll, TMemAccessAtom atom,
                                unsigned numWarps, int bitwidth,
                                const TMemLdStRowPlan &rowPlan,
                                bool allowSplitNFastPath) {
  bool debugSupportLayoutGen =
      std::getenv("TRITON_DEBUG_TMEM_SUPPORT_LAYOUT_GEN") != nullptr;
  auto tryLayout = [&](const LinearLayout &candidateLL) -> std::optional<LinearLayout> {
    auto squeezed = candidateLL;
    auto *ctx = (*squeezed.getOutDimNames().begin()).getContext();
    auto kBlock = StringAttr::get(ctx, "block");
    if (squeezed.hasInDim(kBlock) && squeezed.getInDimSize(kBlock) == 1)
      squeezed = squeezed.squeezeIns(kBlock);
    if (squeezed.hasOutDim(kBlock) && squeezed.getOutDimSize(kBlock) == 1)
      squeezed = squeezed.squeezeOuts(kBlock);

    auto dims = to_vector(squeezed.getOutDimNames());
    assert(dims.size() == 2);
    auto rowColDims = to_vector(squeezed.getInDimNames());
    bool hasBlockDim = llvm::is_contained(rowColDims, kBlock);
    if (hasBlockDim && squeezed.getInDimSize(kBlock) > 1) {
      auto ctasPerCGA =
          ::mlir::triton::basesPerDimImpl(squeezed.getBases(), kBlock,
                                          dims.size(),
                                          /*skipBroadcast=*/false);
      auto ctaSplitNum =
          ::mlir::triton::basesPerDimImpl(squeezed.getBases(), kBlock,
                                          dims.size(),
                                          /*skipBroadcast=*/true);
      SmallVector<unsigned> defaultOrder(dims.size());
      std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
      auto ctaOrder =
          ::mlir::triton::orderPerDimImpl(squeezed, kBlock, defaultOrder);
      auto blockOnly =
          gpu::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA, ctaSplitNum,
                                                ctaOrder)
              .getLinearLayout();
      if (auto maybePerCTA = divideRight(squeezed, blockOnly)) {
        if (auto perCTA = getDistributedLayoutForTmemLdSt(
                *maybePerCTA, atom, numWarps, bitwidth, rowPlan,
                allowSplitNFastPath)) {
          return *perCTA * blockOnly;
        }
      }
    }
    if (allowSplitNFastPath && bitwidth == 32 &&
        atom == TMemAccessAtom::I32x32b) {
      if (auto splitN = getTMemLdStSplitNLayout(squeezed, numWarps, rowPlan))
        return splitN;
    }
    // Canonical contiguous M64 TMEM views should keep the dedicated M64
    // register families even when row-plan selection uses the active 64-row
    // footprint. Restricting this fast-path to 128-row plans regresses the
    // preferred 16x256b Blackwell accumulator layout to narrower x128/x64
    // shapes.
    if (bitwidth == 16 && !hasBlockDim && atom != TMemAccessAtom::I16x32bx2 &&
        matchesCanonicalContiguousM64LinearView(squeezed)) {
      if (auto canonical = getCanonicalContiguousM64Layout(
              ctx, atom, squeezed.getInDimSize(rowColDims[1]), numWarps)) {
        auto withoutBroadcast = *canonical;
        for (auto inDim : canonical->getInDimNames())
          withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
        if (withoutBroadcast.isInvertible())
          return canonical;
      }
    }
    // This code is dual to the one in lowerTMemLdSt
    if (bitwidth != 32) {
      auto kReg = StringAttr::get(ctx, "register");
      if (auto maybeQuot = divideLeft(
              squeezed,
              LinearLayout::zeros1D(32 / bitwidth, rowColDims[1], dims[1]) *
                  LinearLayout::identity1D(2, rowColDims[1], dims[1]));
          bitwidth == 16 && atom == TMemAccessAtom::I32x32b &&
          !allowSplitNFastPath && maybeQuot) {
        // Keep the unpacked 16-bit reinterpret rescue on the generic 32-bit
        // I32x32b builder, but do not let it preempt ordinary direct loads.
        // Direct MMAv5 accumulator loads rely on the standard bitwidth-packing
        // path below to preserve the legacy-equivalent packed row/warp layout.
        auto ret = getDistributedLayoutForTmemLdSt(
            *maybeQuot, TMemAccessAtom::I32x32b, numWarps, 32, rowPlan,
            /*allowSplitNFastPath=*/false);
        if (!ret)
          return ret;
        auto castbbitwidth =
            LinearLayout::zeros1D(1, kReg, dims[1], 32 / bitwidth) *
            LinearLayout::identity1D(2, kReg, dims[1]);
        return castbbitwidth * ret.value();
      }
      LinearLayout quot;
      int bestContig = 1;
      for (int contig = 1; bitwidth * contig <= 32; contig *= 2) {
        auto maybeQuot = divideLeft(
            squeezed,
            LinearLayout::identity1D(contig, rowColDims[1], dims[1]));
        if (!maybeQuot)
          break;
        quot = *maybeQuot;
        bestContig = contig;
      }

      if (bestContig > 1) {
        auto ret = getDistributedLayoutForTmemLdSt(
            quot, atom, numWarps, bitwidth * bestContig, rowPlan,
            allowSplitNFastPath && !(bitwidth == 16 &&
                                     atom == TMemAccessAtom::I32x32b));
        if (!ret)
          return ret;
        auto castbbitwidth =
            LinearLayout::identity1D(bestContig, kReg, dims[1]);
        return castbbitwidth * ret.value();
      }
      if (auto maybeQuot = divideLeft(
                     squeezed, LinearLayout::zeros1D(
                                      32 / bitwidth, rowColDims[1], dims[1]))) {
        return getDistributedLayoutForTmemLdSt(
            *maybeQuot, atom, numWarps, 32, rowPlan,
            allowSplitNFastPath && !(bitwidth == 16 &&
                                     atom == TMemAccessAtom::I32x32b));
      } else if (squeezed.getInDimSize(rowColDims[1]) == 1) {
        return getDistributedLayoutForTmemLdSt(squeezed, atom, numWarps, 32,
                                               rowPlan,
                                               allowSplitNFastPath);
      } else {
        return std::nullopt;
      }
    }
    assert(bitwidth == 32);
    if (allowSplitNFastPath && atom == TMemAccessAtom::I16x32bx2 &&
        rowPlan.rowSpan == 64 &&
        !hasBlockDim && squeezed.getInDimSize(rowColDims[0]) == 64) {
      int64_t n = squeezed.getInDimSize(rowColDims[1]);
      if (auto canonical = getCanonicalM64SplitNLayout(ctx, n, numWarps))
        return canonical;
    }
    auto tile = getTileLayout(ctx, atom, false, /*withWarp=*/false,
                              rowPlan.warpRow0, rowPlan.warpRow1,
                              rowPlan.rowSpan);

    auto nColsTile = tile.getOutDimSize(rowColDims[1]);
    auto nColsLL = squeezed.getInDimSize(rowColDims[1]);
    auto nColsMissing = nColsLL / nColsTile;
    if (nColsMissing == 0)
      return std::nullopt;
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    bool instr32Rows = atom == TMemAccessAtom::I32x32b;
    bool layout16Rows =
        squeezed.getInDimSize(rowColDims[0]) <= 16 ||
        squeezed.getBasis(rowColDims[0], llvm::Log2_32(16)) ==
            ArrayRef{0, 0};

    auto compInput = tile;
    if (hasBlockDim)
      compInput *= LinearLayout::identity1D(1, kBlock, kBlock);
    if (!canComposeLinearLayouts(compInput, squeezed)) {
      if (debugSupportLayoutGen)
        llvm::errs() << "[tmem-layout-gen] reject: compInput cannot compose\n";
      return std::nullopt;
    }
    auto comp = compInput.compose(squeezed)
                    .sublayout({kReg, kLane},
                               to_vector(squeezed.getOutDimNames()));
    if (instr32Rows) {
      comp = comp.resizeInDim(kLane, comp.getInDimSize(kLane) / 2);
    }
    auto compWithoutBroadcast = comp;
    for (auto inDim : comp.getInDimNames())
      compWithoutBroadcast =
          compWithoutBroadcast.removeZeroBasesAlongDim(inDim);
    if (!compWithoutBroadcast.isInjective() &&
        rowPlan.warpRow0 != 0 && rowPlan.warpRow1 != 0) {
      if (debugSupportLayoutGen) {
        llvm::errs() << "[tmem-layout-gen] reject: pre-warp injective failed\n"
                     << candidateLL.toString() << "\n";
      }
      return std::nullopt;
    }

    StringAttr row16;
    if (!instr32Rows && !layout16Rows) {
      if (numWarps > 4) {
        row16 = kWarp;
      } else {
        row16 = kReg;
      }
    }

    int warpsToTile = numWarps / ((row16 == kWarp) ? 8 : 4);
    int warpBroadcast = warpsToTile / std::min(nColsMissing, warpsToTile);
    warpsToTile /= warpBroadcast;
    nColsMissing /= warpsToTile;

    if (nColsMissing > 1) {
      if (instr32Rows && layout16Rows) {
        tile =
            divideLeft(tile, LinearLayout::identity1D(2, kLane, rowColDims[0]))
                .value();
        tile *=
            LinearLayout::identity1D(nColsMissing / 2, kReg, rowColDims[1]) *
            LinearLayout::identity1D(2, kLane, rowColDims[1]);

      } else {
        tile *= LinearLayout::identity1D(nColsMissing, kReg, rowColDims[1]);
      }
    }

    auto bases = tile.getBases();
    auto &warpBases = bases[kWarp];
    warpBases.push_back({rowPlan.warpRow0, 0});
    warpBases.push_back({rowPlan.warpRow1, 0});

    if (row16) {
      bases[row16].push_back({16, 0});
    }
    std::string error;
    auto maybeTile = LinearLayout::tryCreate(
        std::move(bases),
        {{rowColDims[0], rowPlan.rowSpan},
         {rowColDims[1], tile.getOutDimSize(rowColDims[1])}},
        /*requireSurjective=*/false, &error);
    if (!maybeTile)
      return std::nullopt;
    tile = *maybeTile;
    tile *= LinearLayout::identity1D(warpsToTile, kWarp, rowColDims[1]);
    tile *= LinearLayout::zeros1D(warpBroadcast, kWarp, rowColDims[1]);
    if (hasBlockDim) {
      auto nCTAs = squeezed.getInDimSize(kBlock);
      tile *= LinearLayout::identity1D(nCTAs, kBlock, kBlock);
    }
    assert(tile.getOutDimSize(rowColDims[1]) ==
           squeezed.getInDimSize(rowColDims[1]));
    if (!canComposeLinearLayouts(tile, squeezed)) {
      if (debugSupportLayoutGen) {
        llvm::errs() << "[tmem-layout-gen] reject: tile cannot compose\n"
                     << tile.toString() << "\nwith\n"
                     << squeezed.toString() << "\n";
      }
      return std::nullopt;
    }

    auto ret = tile.compose(squeezed);
    auto withoutBroadcast = ret;
    for (auto inDim : ret.getInDimNames()) {
      withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
    }
    if (!withoutBroadcast.isInvertible()) {
      if (debugSupportLayoutGen) {
        llvm::errs() << "[tmem-layout-gen] reject: final invertible failed\n"
                     << ret.toString() << "\n";
      }
      return std::nullopt;
    }
    return ret;
  };

  if (auto ret = tryLayout(ll))
    return ret;
  auto stripped = stripZeroBasesForTmemLdStSelection(ll);
  if (stripped == ll)
    return std::nullopt;
  return tryLayout(stripped);
}

static std::optional<LinearLayout>
getDistributedLayoutForTmemLdStLegacyAnchored(const LinearLayout &ll,
                                              TMemAccessAtom atom,
                                              unsigned numWarps,
                                              int bitwidth) {
  auto dims = to_vector(ll.getOutDimNames());
  assert(dims.size() == 2);
  auto rowColDims = to_vector(ll.getInDimNames());
  auto *ctx = dims[0].getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  bool hasBlockDim = llvm::is_contained(rowColDims, kBlock);
  if (hasBlockDim && ll.getInDimSize(kBlock) > 1) {
    auto ctasPerCGA =
        ::mlir::triton::basesPerDimImpl(ll.getBases(), kBlock, dims.size(),
                                        /*skipBroadcast=*/false);
    auto ctaSplitNum =
        ::mlir::triton::basesPerDimImpl(ll.getBases(), kBlock, dims.size(),
                                        /*skipBroadcast=*/true);
    SmallVector<unsigned> defaultOrder(dims.size());
    std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
    auto ctaOrder = ::mlir::triton::orderPerDimImpl(ll, kBlock, defaultOrder);
    auto blockOnly =
        gpu::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA, ctaSplitNum,
                                              ctaOrder)
            .getLinearLayout();
    if (auto maybePerCTA = divideRight(ll, blockOnly)) {
      if (auto perCTA = getDistributedLayoutForTmemLdStLegacyAnchored(
              *maybePerCTA, atom, numWarps, bitwidth)) {
        return *perCTA * blockOnly;
      }
    }
  }
  if (bitwidth != 32) {
    auto kReg = StringAttr::get(ctx, "register");
    LinearLayout quot;
    int bestContig = 1;
    for (int contig = 1; bitwidth * contig <= 32; contig *= 2) {
      auto maybeQuot = divideLeft(
          ll, LinearLayout::identity1D(contig, rowColDims[1], dims[1]));
      if (!maybeQuot)
        break;
      quot = *maybeQuot;
      bestContig = contig;
    }

    if (auto maybeQuot = divideLeft(
            ll, LinearLayout::zeros1D(32 / bitwidth, rowColDims[1], dims[1]) *
                    LinearLayout::identity1D(2, rowColDims[1], dims[1]));
        bitwidth == 16 && maybeQuot) {
      auto ret = getDistributedLayoutForTmemLdStLegacyAnchored(
          *maybeQuot, atom, numWarps, 32);
      if (!ret)
        return ret;
      auto castbbitwidth =
          LinearLayout::zeros1D(1, kReg, dims[1], 32 / bitwidth) *
          LinearLayout::identity1D(2, kReg, dims[1]);
      return castbbitwidth * ret.value();
    }
    if (bestContig > 1) {
      auto ret = getDistributedLayoutForTmemLdStLegacyAnchored(
          quot, atom, numWarps, bitwidth * bestContig);
      if (!ret)
        return ret;
      auto castbbitwidth = LinearLayout::identity1D(bestContig, kReg, dims[1]);
      return castbbitwidth * ret.value();
    }
    if (auto maybeQuot =
                   divideLeft(ll, LinearLayout::zeros1D(
                                      32 / bitwidth, rowColDims[1], dims[1]))) {
      return getDistributedLayoutForTmemLdStLegacyAnchored(*maybeQuot, atom,
                                                           numWarps, 32);
    } else if (ll.getInDimSize(rowColDims[1]) == 1) {
      return getDistributedLayoutForTmemLdStLegacyAnchored(ll, atom, numWarps,
                                                           32);
    } else {
      return std::nullopt;
    }
  }

  assert(bitwidth == 32);
  auto tile = getTileLayout(ctx, atom, false, /*withWarp=*/false);

  auto nColsTile = tile.getOutDimSize(rowColDims[1]);
  auto nColsLL = ll.getInDimSize(rowColDims[1]);
  auto nColsMissing = nColsLL / nColsTile;
  if (nColsMissing == 0) {
    return std::nullopt;
  }
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  bool instr32Rows = atom == TMemAccessAtom::I32x32b;
  bool layout16Rows =
      ll.getInDimSize(rowColDims[0]) <= 16 ||
      ll.getBasis(rowColDims[0], llvm::Log2_32(16)) == ArrayRef{0, 0};

  auto compInput = tile;
  if (hasBlockDim)
    compInput *= LinearLayout::identity1D(1, kBlock, kBlock);
  if (!canComposeLinearLayouts(compInput, ll))
    return std::nullopt;
  auto comp =
      compInput.compose(ll).sublayout({kReg, kLane}, to_vector(ll.getOutDimNames()));
  if (instr32Rows) {
    comp = comp.resizeInDim(kLane, comp.getInDimSize(kLane) / 2);
  }
  auto compWithoutBroadcast = comp;
  for (auto inDim : comp.getInDimNames())
    compWithoutBroadcast = compWithoutBroadcast.removeZeroBasesAlongDim(inDim);
  if (!compWithoutBroadcast.isInjective())
    return std::nullopt;

  StringAttr row16;
  if (!instr32Rows && !layout16Rows) {
    if (numWarps > 4) {
      row16 = kWarp;
    } else {
      row16 = kReg;
    }
  }

  int warpsToTile = numWarps / ((row16 == kWarp) ? 8 : 4);
  int warpBroadcast = warpsToTile / std::min(nColsMissing, warpsToTile);
  warpsToTile /= warpBroadcast;
  nColsMissing /= warpsToTile;

  if (nColsMissing > 1) {
    if (instr32Rows && layout16Rows) {
      tile = divideLeft(tile, LinearLayout::identity1D(2, kLane, rowColDims[0]))
                 .value();
      tile *= LinearLayout::identity1D(nColsMissing / 2, kReg, rowColDims[1]) *
              LinearLayout::identity1D(2, kLane, rowColDims[1]);
    } else if (atom == TMemAccessAtom::I16x32bx2 && layout16Rows &&
               nColsMissing >= 2) {
      // Keep the last half-tile column split on lane=16 so direct lowering can
      // use the native second-half offset instead of materializing two x1
      // packets at different base addresses.
      tile *= LinearLayout::identity1D(nColsMissing / 2, kReg, rowColDims[1]) *
              LinearLayout::identity1D(2, kLane, rowColDims[1]);
    } else {
      tile *= LinearLayout::identity1D(nColsMissing, kReg, rowColDims[1]);
    }
  }

  auto bases = tile.getBases();
  auto &warpBases = bases[kWarp];
  warpBases.push_back({32, 0});
  warpBases.push_back({64, 0});

  if (row16) {
    bases[row16].push_back({16, 0});
  }
  tile = LinearLayout(std::move(bases),
                      {{rowColDims[0], 128},
                       {rowColDims[1], tile.getOutDimSize(rowColDims[1])}},
                      false);
  tile *= LinearLayout::identity1D(warpsToTile, kWarp, rowColDims[1]);
  tile *= LinearLayout::zeros1D(warpBroadcast, kWarp, rowColDims[1]);
  if (hasBlockDim) {
    auto nCTAs = ll.getInDimSize(kBlock);
    tile *= LinearLayout::identity1D(nCTAs, kBlock, kBlock);
  }
  assert(tile.getOutDimSize(rowColDims[1]) == ll.getInDimSize(rowColDims[1]));
  if (!canComposeLinearLayouts(tile, ll))
    return std::nullopt;

  auto ret = tile.compose(ll);
  auto withoutBroadcast = ret;
  for (auto inDim : ret.getInDimNames())
    withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
  if (!withoutBroadcast.isInvertible())
    return std::nullopt;
  return ret;
}

static LinearLayout
stripZeroBasesForTmemLdStSelection(LinearLayout ll) {
  if (ll.getNumInDims() == 0)
    return ll;
  auto *ctx = (*ll.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  for (StringAttr dim : {kRow, kCol, kBlock}) {
    if (ll.hasInDim(dim))
      ll = ll.removeZeroBasesAlongDim(dim);
  }
  if (ll.hasInDim(kBlock) && ll.getInDimSize(kBlock) == 1)
    ll = ll.squeezeIns(kBlock);
  return ll;
}

static std::optional<LinearEncodingAttr>
tryGetLinearEncodingAttr(MLIRContext *ctx, LinearLayout layout) {
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  SmallVector<StringAttr> expectedDims = {kRegister, kLane, kWarp, kBlock};
  auto existingBases = layout.getBases();
  bool needsNormalization = llvm::any_of(
      expectedDims, [&](StringAttr dim) { return !layout.hasInDim(dim); });
  if (needsNormalization) {
    LinearLayout::BasesT normalizedBases;
    for (StringAttr dim : expectedDims) {
      auto it = existingBases.find(dim);
      if (it != existingBases.end()) {
        normalizedBases[dim] = it->second;
      } else {
        normalizedBases[dim] = {};
      }
    }
    for (const auto &[dim, bases] : existingBases) {
      if (!llvm::is_contained(expectedDims, dim))
        normalizedBases[dim] = bases;
    }
    layout = LinearLayout(std::move(normalizedBases), layout.getOutDims(),
                          layout.isSurjective());
  }
  static const auto expectedInDims =
      SmallVector<std::string>({"register", "lane", "warp", "block"});
  auto inDims = to_vector(layout.getInDimNames());
  if (inDims.size() < expectedInDims.size())
    return std::nullopt;
  for (auto [dim, expected] :
       llvm::zip_equal(ArrayRef(inDims).take_front(expectedInDims.size()),
                       expectedInDims)) {
    if (dim.str() != expected)
      return std::nullopt;
  }
  for (auto [i, dim] : llvm::enumerate(layout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str())
      return std::nullopt;
  }
  const auto &bases = layout.getBases();
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(bases)) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return std::count_if(basis.begin(), basis.end(), nonZero) <= 1;
        })) {
      return std::nullopt;
    }
  }
  LinearLayout withoutBroadcast = layout;
  for (auto inDim : layout.getInDimNames())
    withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
  if (!withoutBroadcast.isInvertible())
    return std::nullopt;
  return LinearEncodingAttr::get(ctx, std::move(layout));
}

static bool
isTMemLdStSelectionLayoutValid(gpu::MemDescType memType,
                               const LinearLayout &layout) {
  auto attr = tryGetLinearEncodingAttr(memType.getContext(), layout);
  if (!attr)
    return false;
  auto regTy = RankedTensorType::get(memType.getShape(), memType.getElementType(),
                                     *attr);
  return succeeded(
      computeTMemLdStEncodingInfo(regTy, memType, /*maxnreg=*/256));
}

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps,
                                std::optional<TMemLdStRowPlan> rowPlanOverride,
                                std::optional<LinearLayout> queryLayoutOverride) {
  assert(memType.getMemorySpace() ==
         TensorMemorySpaceAttr::get(memType.getContext()));
  if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
    return std::nullopt;
  auto isValidLayout = [&](const LinearLayout &layout) {
    auto attr = tryGetLinearEncodingAttr(memType.getContext(), layout);
    if (!attr)
      return false;
    auto regTy = RankedTensorType::get(memType.getShape(), memType.getElementType(),
                                       *attr);
    auto info = computeTMemLdStEncodingInfo(regTy, memType, /*maxnreg=*/256,
                                            /*emitError=*/{}, rowPlanOverride);
    return succeeded(info) && info->atom == atom;
  };
  auto isValidLayoutForQuery = [&](const LinearLayout &layout,
                                   const LinearLayout &queryLayout,
                                   std::optional<TMemLdStRowPlan> queryRowPlan) {
    auto attr = tryGetLinearEncodingAttr(memType.getContext(), layout);
    if (!attr)
      return false;
    auto regTy = RankedTensorType::get(memType.getShape(),
                                       memType.getElementType(), *attr);
    auto info = computeTMemLdStEncodingInfo(regTy, memType, queryLayout,
                                            /*maxnreg=*/256,
                                            /*emitError=*/{}, queryRowPlan);
    return succeeded(info) && info->atom == atom;
  };
  auto ll = [&]() -> LinearLayout {
    if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
      return toLinearLayout(memType);
    if (!rowPlanOverride && memType.getShape() == memType.getAllocShape()) {
      auto raw = toLinearLayout(memType.getShape(), memType.getEncoding());
      std::string rawError;
      if (auto maybeLayout = getTMemViewAnalysisLinearLayout(
              memType.getShape(), memType.getEncoding(), &rawError)) {
        auto normalized =
            normalizeTensorMemoryLinearLayoutForAnalysis(*maybeLayout);
        if (matchesCanonicalContiguousM64LinearView(normalized))
          return normalized;
      }
      return raw;
    }
    std::string error;
    SmallVector<int64_t> layoutShape(memType.getShape().begin(),
                                     memType.getShape().end());
    auto layoutRank =
        static_cast<size_t>(cast<LayoutEncodingTrait>(memType.getEncoding())
                                .getRank());
    if (rowPlanOverride && layoutShape.size() >= layoutRank &&
        memType.getAllocShape().size() >= layoutRank &&
        ArrayRef<int64_t>(layoutShape).take_back(layoutRank) !=
            memType.getAllocShape().take_back(layoutRank)) {
      layoutShape.assign(memType.getAllocShape().begin(),
                         memType.getAllocShape().end());
    }
    auto maybe = getTMemViewAnalysisLinearLayout(layoutShape,
                                                 memType.getEncoding(), &error);
    if (!maybe)
      return LinearLayout();
    return normalizeTensorMemoryLinearLayoutForAnalysis(*maybe);
  }();
  if (ll.getNumOutDims() == 0)
    return std::nullopt;
  if (queryLayoutOverride)
    ll = *queryLayoutOverride;
  auto bitwidth = memType.getElementTypeBitWidth();
  auto stripped = stripZeroBasesForTmemLdStSelection(ll);
  if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      !rowPlanOverride) {
    if (auto layout = getDistributedLayoutForTmemLdStLegacyAnchored(
            ll, atom, numWarps, bitwidth);
        layout && isTMemLdStSelectionLayoutValid(memType, *layout)) {
      return layout;
    }

    auto stripped = stripZeroBasesForTmemLdStSelection(ll);
    if (stripped != ll) {
      auto layout = getDistributedLayoutForTmemLdStLegacyAnchored(
          stripped, atom, numWarps, bitwidth);
      if (layout && isValidLayout(*layout))
        return layout;
    }
    return std::nullopt;
  }
  if (!rowPlanOverride && memType.getShape() == memType.getAllocShape()) {
    if (planMMAv5ExactFamily(memType.getShape(), memType.getEncoding(),
                             {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u,
                              512u})) {
      if (auto legacyLike = getDistributedLayoutForTmemLdStLegacyAnchored(
              ll, atom, numWarps, bitwidth)) {
        if (isValidLayout(*legacyLike))
          return legacyLike;
      }
    }
  }
  auto tryCanonicalContiguousM64 =
      [&](const LinearLayout &candidate,
          std::optional<TMemLdStRowPlan> candidateRowPlan,
          std::optional<LinearLayout> queryLayoutOverride)
      -> std::optional<LinearLayout> {
    if (!matchesCanonicalContiguousM64LinearView(candidate))
      return std::nullopt;
    auto *ctx = memType.getContext();
    auto kCol = StringAttr::get(ctx, "col");
    auto canonical = getCanonicalContiguousM64Layout(
        ctx, atom, candidate.getInDimSize(kCol), numWarps);
    if (!canonical)
      return std::nullopt;
    auto attr = tryGetLinearEncodingAttr(memType.getContext(), *canonical);
    if (!attr)
      return std::nullopt;
    auto regTy = RankedTensorType::get(memType.getShape(),
                                       memType.getElementType(), *attr);
    if (queryLayoutOverride) {
      if (candidateRowPlan &&
          succeeded(computeTMemLdStEncodingInfo(
              regTy, memType, *queryLayoutOverride, /*maxnreg=*/256,
              /*emitError=*/{}, candidateRowPlan))) {
        return canonical;
      }
      return std::nullopt;
    }
    if (!candidateRowPlan)
      return isTMemLdStSelectionLayoutValid(memType, *canonical)
                 ? canonical
                 : std::nullopt;
    if (succeeded(computeTMemLdStEncodingInfo(
            regTy, memType, candidate, /*maxnreg=*/256, /*emitError=*/{},
            candidateRowPlan))) {
      return canonical;
    }
    return std::nullopt;
  };
  if (!rowPlanOverride) {
    auto tryCanonicalM64SplitN =
        [&]() -> std::optional<LinearLayout> {
      if (atom != TMemAccessAtom::I16x32bx2 ||
          isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) ||
          memType.getShape() != memType.getAllocShape()) {
        return std::nullopt;
      }
      if (auto canonical = getCanonicalM64SplitNLayout(memType, numWarps);
          canonical && isValidLayout(*canonical)) {
        return canonical;
      }
      return std::nullopt;
    };
    if (auto canonical = tryCanonicalM64SplitN())
      return canonical;
    if (auto canonical =
            tryCanonicalContiguousM64(ll, /*candidateRowPlan=*/std::nullopt,
                                      /*queryLayoutOverride=*/std::nullopt))
      return canonical;
    if (stripped != ll) {
      auto strippedRowPlan =
          isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding())
              ? getTMemLdStRowPlanForType(memType)
              : getTMemLdStRowPlan(stripped);
      // Legacy M64 leaves encode the unused half-tile as a zero row basis.
      // Validate the stripped canonical M64 layout against the stripped row
      // anchors before falling back to the raw generic planner; otherwise the
      // raw 128-row form wins and 64x32 MMAv5 accumulators degrade to the
      // scalar 32x32b.x1 readback path.
      if (auto canonical = tryCanonicalContiguousM64(
              stripped, strippedRowPlan, /*queryLayoutOverride=*/stripped))
        return canonical;
      if (auto canonical = tryCanonicalContiguousM64(
              stripped, /*candidateRowPlan=*/std::nullopt,
              /*queryLayoutOverride=*/std::nullopt))
        return canonical;
    }
  }
  auto rowPlan = rowPlanOverride;
  if (!rowPlan) {
    if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
      rowPlan = getTMemLdStRowPlanForType(memType);
    else
      rowPlan = getTMemLdStRowPlan(ll);
  }
  if (!rowPlan)
    return std::nullopt;
  bool disableSplitNFastPathForProjectedContiguousM64 =
      rowPlanOverride &&
      !isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      matchesCanonicalContiguousM64LinearView(ll);
  if (auto canonical = tryCanonicalContiguousM64(ll, rowPlanOverride,
                                                 /*queryLayoutOverride=*/ll))
    return canonical;
  if (auto layout = getDistributedLayoutForTmemLdSt(ll, atom, numWarps,
                                                    bitwidth, *rowPlan,
                                                    !disableSplitNFastPathForProjectedContiguousM64);
      layout &&
      (rowPlanOverride ? isValidLayoutForQuery(*layout, ll, rowPlan)
                       : isValidLayout(*layout))) {
    return layout;
  }

  if (stripped == ll)
    return std::nullopt;
  auto strippedRowPlan = rowPlanOverride;
  if (!strippedRowPlan) {
    if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
      strippedRowPlan = getTMemLdStRowPlanForType(memType);
    else
      strippedRowPlan = getTMemLdStRowPlan(stripped);
  }
  if (!strippedRowPlan)
    return std::nullopt;
  if (auto canonical = tryCanonicalContiguousM64(
          stripped, strippedRowPlan, /*queryLayoutOverride=*/stripped))
    return canonical;
  bool disableSplitNFastPathForProjectedStrippedM64 =
      rowPlanOverride &&
      !isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      matchesCanonicalContiguousM64LinearView(stripped);
  auto layout = getDistributedLayoutForTmemLdSt(stripped, atom, numWarps,
                                                bitwidth, *strippedRowPlan,
                                                !disableSplitNFastPathForProjectedStrippedM64);
  if (!layout)
    return std::nullopt;
  auto attr = LinearEncodingAttr::get(memType.getContext(), *layout);
  auto regTy =
      RankedTensorType::get(memType.getShape(), memType.getElementType(), attr);
  if (failed(rowPlanOverride ? computeTMemLdStEncodingInfo(
                                  regTy, memType, stripped,
                                  /*maxnreg=*/256, /*emitError=*/{},
                                  strippedRowPlan)
                             : computeTMemLdStEncodingInfo(
                                   regTy, memType, /*maxnreg=*/256,
                                   /*emitError=*/{}, strippedRowPlan))) {
    return std::nullopt;
  }
  return layout;
}

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps) {
  return getDistributedLayoutForTmemLdSt(memType, atom, numWarps,
                                         std::nullopt);
}

static bool isTMemCompatibleCandidate(Operation *op, RankedTensorType tensorType,
                                      gpu::MemDescType memType,
                                      const LinearLayout &layout) {
  auto candidateEncoding = tryGetLinearEncodingAttr(tensorType.getContext(), layout);
  if (!candidateEncoding)
    return false;
  auto candidateType = tensorType.cloneWithEncoding(*candidateEncoding);
  auto maxnreg = getContextualMaxNReg(op);
  return succeeded(
      computeTMemLdStEncodingInfo(candidateType, memType, maxnreg));
}

DistributedEncodingTrait getDefaultLayoutForTmemLdSt(gpu::MemDescType memType,
                                                     unsigned numWarps) {
  auto *ctx = memType.getContext();
  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  if (!isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      memType.getShape() == memType.getAllocShape() &&
      planMMAv5ExactFamily(memType.getShape(), memType.getEncoding(),
                           {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u,
                            512u})) {
    auto raw = toLinearLayout(memType.getShape(), memType.getEncoding());
    SmallVector<TMemAccessAtom> atoms =
        prefer16x256
            ? SmallVector<TMemAccessAtom>{TMemAccessAtom::I16x256b,
                                          TMemAccessAtom::I32x32b,
                                          TMemAccessAtom::I16x128b,
                                          TMemAccessAtom::I16x64b,
                                          TMemAccessAtom::I16x32bx2}
            : SmallVector<TMemAccessAtom>{TMemAccessAtom::I32x32b,
                                          TMemAccessAtom::I16x256b,
                                          TMemAccessAtom::I16x128b,
                                          TMemAccessAtom::I16x64b,
                                          TMemAccessAtom::I16x32bx2};
    for (auto atom : atoms) {
      if (auto preferred = getDistributedLayoutForTmemLdStLegacyAnchored(
              raw, atom, numWarps, memType.getElementTypeBitWidth());
          preferred && isTMemLdStSelectionLayoutValid(memType, *preferred)) {
        return LinearEncodingAttr::get(ctx, std::move(*preferred));
      }
    }
  }
  if (prefer16x256) {
    auto tryLegacyPreferred =
        [&](const LinearLayout &layout)
        -> std::optional<DistributedEncodingTrait> {
      if (auto preferred = getDistributedLayoutForTmemLdStLegacyAnchored(
              layout, TMemAccessAtom::I16x256b, numWarps,
              memType.getElementTypeBitWidth());
          preferred && isTMemLdStSelectionLayoutValid(memType, *preferred)) {
        return LinearEncodingAttr::get(ctx, std::move(*preferred));
      }
      return std::nullopt;
    };
    auto tryCanonicalPreferredM64 =
        [&](const LinearLayout &layout) -> std::optional<DistributedEncodingTrait> {
      auto stripped = stripZeroBasesForTmemLdStSelection(layout);
      if (!matchesCanonicalContiguousM64LinearView(stripped))
        return std::nullopt;
      auto kCol = StringAttr::get(ctx, "col");
      if (auto canonical = getCanonicalContiguousM64Layout(
              ctx, TMemAccessAtom::I16x256b, stripped.getInDimSize(kCol),
              numWarps);
          canonical && isTMemLdStSelectionLayoutValid(memType, *canonical)) {
        return LinearEncodingAttr::get(ctx, std::move(*canonical));
      }
      TMemLdStRowPlan legacyM64Plan{/*warpRow0=*/32, /*warpRow1=*/64,
                                    /*rowSpan=*/128};
      if (auto preferred = getDistributedLayoutForTmemLdSt(
              memType, TMemAccessAtom::I16x256b, numWarps, legacyM64Plan)) {
        return LinearEncodingAttr::get(ctx, std::move(*preferred));
      }
      return std::nullopt;
    };
    if (!isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding())) {
      auto raw = toLinearLayout(memType.getShape(), memType.getEncoding());
      if (memType.getShape() == memType.getAllocShape()) {
        if (auto preferred = tryLegacyPreferred(raw))
          return *preferred;
      }
      std::string error;
      if (auto maybeLayout = getTMemViewAnalysisLinearLayout(
              memType.getShape(), memType.getEncoding(), &error)) {
        auto normalized =
            normalizeTensorMemoryLinearLayoutForAnalysis(*maybeLayout);
        if (auto preferred = tryCanonicalPreferredM64(normalized))
          return *preferred;
      }
      if (auto preferred = tryCanonicalPreferredM64(raw))
        return *preferred;
    }
    auto layout = getDistributedLayoutForTmemLdSt(
        memType, TMemAccessAtom::I16x256b, numWarps);
    if (layout) {
      return LinearEncodingAttr::get(ctx, std::move(*layout));
    }
  }
  if (auto layout = getDistributedLayoutForTmemLdSt(
          memType, TMemAccessAtom::I32x32b, numWarps)) {
    return LinearEncodingAttr::get(ctx, std::move(*layout));
  }
  auto layouts = getTmemCompatibleLayouts(memType, numWarps);
  if (layouts.empty() &&
      !isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      memType.getRank() == 2 && memType.getElementTypeBitWidth() == 32 &&
      memType.getShape() == memType.getAllocShape() &&
      memType.getShape()[0] == 64) {
    auto raw = toLinearLayout(memType.getShape(), memType.getEncoding());
    if (matchesSimplePermutedM64SplitNLinearView(raw)) {
      if (auto canonical = getCanonicalM64SplitNLayout(memType, numWarps))
        return LinearEncodingAttr::get(ctx, std::move(*canonical));
    }
  }
  assert(!layouts.empty() &&
         "expected at least one TMEM-compatible register layout");
  return layouts.front();
}

std::optional<DistributedEncodingTrait>
getTmemLoadLayoutSplitLongM(RankedTensorType tensorType, MemDescType memType,
                            int numWarps) {
  if (numWarps != 8)
    return std::nullopt;
  if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
    return std::nullopt;

  std::optional<LinearLayout> layout = getDistributedLayoutForTmemLdSt(
      memType, TMemAccessAtom::I32x32b, numWarps);
  if (!layout)
    return std::nullopt;
  auto ret = std::move(*layout);

  // Optimisation for reductions:
  // We can map lane=16 to any dimension, and it will be lowered to 32x16bx2.
  // As such, if we have 8 warps and the basis warp=4 is mapped to a different
  // dimension than warp=1, warp=2, and lane=16 is mapped to the same dimension
  // as the first two warp bases, we can swap warp=4 and lane=16.
  // Generally, we don't want warp=4 to have data on a different dimension to
  // dim=1 and dim=2
  auto *ctx = tensorType.getContext();
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto dims = to_vector(ret.getOutDimNames());

  // In most cases this is going to be dim=0, but the optimization
  // also applies for scales where we may be able to have the layout
  // replicated across warps
  for (int dim : {0, 1}) {
    auto w1dim = ret.getBasis(kWarp, 0, dims[dim]) == 0;
    auto w2dim = ret.getBasis(kWarp, 1, dims[dim]) == 0;
    auto w4dim = ret.getBasis(kWarp, 2, dims[dim]) == 0;
    auto l16dim = ret.getBasis(kLane, 4, dims[dim]) == 0;
    if (l16dim != w4dim && w1dim == w2dim && w1dim == l16dim) {
      auto bases = ret.getBases();
      std::swap(bases[kWarp][2], bases[kLane][4]);
      return LinearEncodingAttr::get(
          tensorType.getContext(),
          LinearLayout(std::move(bases), ret.getOutDims(), ret.isSurjective()));
    }
  }
  return std::nullopt;
}

bool isReductionFriendlyTmemLoadLayout(RankedTensorType tensorType,
                                       const LinearLayout &layout) {
  if (layout.getNumOutDims() != 2)
    return false;
  auto attr = LinearEncodingAttr::get(tensorType.getContext(), layout);
  auto regTy = tensorType.cloneWithEncoding(attr);
  auto kReg = StringAttr::get(tensorType.getContext(), "register");
  auto regLayout = toLinearLayout(regTy);
  auto regDims = toLinearEncoding(regTy).basesPerDim(kReg);
  auto outDims = llvm::to_vector(regLayout.getOutDimSizes());
  if (outDims.size() < 2)
    return false;
  return regDims[1] == outDims[1] && regDims[0] == 1;
}

bool isReductionFriendlyTmemSourceLayout(MemDescType memType) {
  std::string error;
  auto maybeCanonical = getCanonicalTMemLinearEncoding(memType, &error);
  if (!maybeCanonical)
    return false;

  auto layout = normalizeTensorMemoryLinearLayoutForAnalysis(
      maybeCanonical->getLinearLayout());
  if (layout.getNumOutDims() != 2)
    return false;

  auto *ctx = memType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto dims = standardOutDimNames(ctx, 2);
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol))
    return false;

  auto rowBits = layout.getInDimSizeLog2(kRow);
  auto blockM = int64_t{1} << rowBits;
  if (blockM != 128)
    return false;

  SmallVector<int32_t> pureRowBases;
  for (unsigned idx = 0; idx < rowBits; ++idx) {
    auto basis = layout.getBasis(kRow, idx);
    if (basis.size() != 2 || basis[0] == 0 || basis[1] != 0) {
      return false;
    }
    pureRowBases.push_back(basis[0]);
  }
  llvm::sort(pureRowBases);
  SmallVector<int32_t> expectedPureRowBases;
  for (unsigned idx = 0; idx < rowBits; ++idx)
    expectedPureRowBases.push_back(static_cast<int32_t>(1u << idx));
  if (!llvm::equal(pureRowBases, expectedPureRowBases))
    return false;

  SmallVector<int32_t> pureRowCarryBases;
  SmallVector<int32_t> pureColBases;
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(kCol); ++idx) {
    auto basis = layout.getBasis(kCol, idx);
    if (basis.size() != 2)
      return false;
    if (basis[0] != 0 && basis[1] == 0) {
      pureRowCarryBases.push_back(basis[0]);
      continue;
    }
    if (basis[0] == 0 && basis[1] != 0) {
      pureColBases.push_back(basis[1]);
      continue;
    }
    return false;
  }

  llvm::sort(pureRowCarryBases);
  llvm::sort(pureColBases);

  SmallVector<int32_t> expectedPureRowCarryBases;
  int64_t repsM = layout.getOutDimSize(dims[0]) / blockM;
  if (repsM < 1 || !llvm::isPowerOf2_64(repsM))
    return false;
  for (int64_t carry = blockM; carry < layout.getOutDimSize(dims[0]);
       carry <<= 1)
    expectedPureRowCarryBases.push_back(static_cast<int32_t>(carry));
  if (!llvm::equal(pureRowCarryBases, expectedPureRowCarryBases))
    return false;

  SmallVector<int32_t> expectedPureColBases;
  int64_t n = layout.getOutDimSize(dims[1]);
  if (n < 1 || !llvm::isPowerOf2_64(n))
    return false;
  for (int64_t col = 1; col < n; col <<= 1)
    expectedPureColBases.push_back(static_cast<int32_t>(col));
  return llvm::equal(pureColBases, expectedPureColBases);
}

std::optional<DistributedEncodingTrait>
getTmemLoadReductionLayout(RankedTensorType tensorType, MemDescType memType,
                           int numWarps) {
  // Reduction layout inference should follow the same direct I32x32b query
  // path as ordinary TMEM load/store. Larger warp counts are legal when the
  // resulting register layout keeps N fully in registers and leaves M
  // unsharded, e.g. 256x128 with 8 warps.
  if (memType.getRank() != 2 ||
      isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) ||
      memType.getElementTypeBitWidth() != 32) {
    return std::nullopt;
  }
  if (!isReductionFriendlyTmemSourceLayout(memType))
    return std::nullopt;

  std::optional<LinearLayout> layout = getDistributedLayoutForTmemLdSt(
      memType, TMemAccessAtom::I32x32b, numWarps);
  if (!layout)
    return std::nullopt;
  auto ret = std::move(*layout);
  if (isReductionFriendlyTmemLoadLayout(tensorType, ret))
    return LinearEncodingAttr::get(tensorType.getContext(), ret);

  auto *ctx = tensorType.getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kWarp = StringAttr::get(ctx, "warp");
  if (!ret.hasInDim(kReg) || !ret.hasInDim(kWarp))
    return std::nullopt;

  auto basisTouchesOnlyDim = [&](StringAttr inDim, unsigned idx,
                                 unsigned dim) -> bool {
    auto dims = to_vector(ret.getOutDimNames());
    if (dim >= dims.size())
      return false;
    bool touchesTarget = ret.getBasis(inDim, idx, dims[dim]) != 0;
    bool touchesOther = false;
    for (auto [otherIdx, otherDim] : llvm::enumerate(dims)) {
      if (otherIdx == dim)
        continue;
      touchesOther |= ret.getBasis(inDim, idx, otherDim) != 0;
    }
    return touchesTarget && !touchesOther;
  };

  SmallVector<unsigned> regMIndices;
  for (unsigned idx = 0; idx < ret.getInDimSizeLog2(kReg); ++idx) {
    if (basisTouchesOnlyDim(kReg, idx, /*dim=*/0))
      regMIndices.push_back(idx);
  }
  SmallVector<unsigned> warpNIndices;
  for (unsigned idx = 0; idx < ret.getInDimSizeLog2(kWarp); ++idx) {
    if (basisTouchesOnlyDim(kWarp, idx, /*dim=*/1))
      warpNIndices.push_back(idx);
  }
  if (regMIndices.empty() || warpNIndices.empty() ||
      regMIndices.size() != warpNIndices.size()) {
    return std::nullopt;
  }

  auto bases = ret.getBases();
  for (auto [regIdx, warpIdx] : llvm::zip_equal(regMIndices, warpNIndices))
    std::swap(bases[kReg][regIdx], bases[kWarp][warpIdx]);
  ret = LinearLayout(std::move(bases), ret.getOutDims(), ret.isSurjective());

  if (!isReductionFriendlyTmemLoadLayout(tensorType, ret))
    return std::nullopt;
  auto attr = LinearEncodingAttr::get(ctx, ret);
  if (failed(computeTMemLdStEncodingInfo(tensorType.cloneWithEncoding(attr),
                                         memType,
                                         /*maxnreg=*/256))) {
    return std::nullopt;
  }
  return attr;
}

SmallVector<DistributedEncodingTrait>
getTmemCompatibleLayouts(MemDescType memType, unsigned numWarps,
                         ArrayRef<int64_t> ctaSplit) {
  (void)ctaSplit;
  SmallVector<DistributedEncodingTrait> layouts;
  if (numWarps % 4 != 0)
    return layouts;

  bool isScales = isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding());
  LinearLayout memLL;
  if (isScales)
    memLL = toLinearLayout(memType);

  auto tryAddScalesNarrowTileLayout = [&]() {
    if (!isScales || numWarps != 4 || memType.getElementTypeBitWidth() != 8)
      return;
    auto shape = memType.getShape();
    if (shape.size() < 2 || shape[shape.size() - 2] != 16 ||
        shape[shape.size() - 1] != 8)
      return;
    auto *ctx = memType.getContext();
    auto dims = standardOutDimNames(ctx, 2);
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    LinearLayout::BasesT bases;
    bases[kReg] = {{0, 1}, {0, 2}, {0, 0}};
    bases[kLane] = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}};
    bases[kWarp] = {{0, 0}, {0, 0}};
    auto narrow =
        LinearLayout(std::move(bases),
                     {{dims[0], static_cast<int32_t>(shape[shape.size() - 2])},
                      {dims[1],
                       static_cast<int32_t>(shape[shape.size() - 1])}},
                     /*requireSurjective=*/false);
    auto candidateEncoding =
        LinearEncodingAttr::get(memType.getContext(), narrow);
    auto tensorTy = RankedTensorType::get(memType.getShape(),
                                          memType.getElementType(),
                                          candidateEncoding);
    if (succeeded(computeTMemLdStEncodingInfo(
            tensorTy, memType, /*maxnreg=*/256))) {
      layouts.push_back(candidateEncoding);
    }
  };
  tryAddScalesNarrowTileLayout();

  auto tensorTy =
      RankedTensorType::get(memType.getShape(), memType.getElementType());
  auto tryPushUniqueLayout = [&](const LinearLayout &layout) {
    auto candidateEncoding =
        tryGetLinearEncodingAttr(memType.getContext(), layout);
    if (!candidateEncoding)
      return;
    if (llvm::is_contained(layouts, *candidateEncoding))
      return;
    auto candidateType = tensorTy.cloneWithEncoding(*candidateEncoding);
    if (succeeded(
            computeTMemLdStEncodingInfo(candidateType, memType,
                                        /*maxnreg=*/256))) {
      layouts.push_back(*candidateEncoding);
    }
  };
  if (!isScales && memType.getElementTypeBitWidth() == 32 &&
      memType.getRank() == 2 && memType.getShape() == memType.getAllocShape() &&
      memType.getShape()[0] == 64) {
    if (auto canonicalSplitN = getCanonicalM64SplitNLayout(memType, numWarps)) {
      auto before = layouts.size();
      tryPushUniqueLayout(*canonicalSplitN);
      if (layouts.size() == before) {
        auto raw = toLinearLayout(memType.getShape(), memType.getEncoding());
        if (matchesSimplePermutedM64SplitNLinearView(raw)) {
          if (auto candidateEncoding =
                  tryGetLinearEncodingAttr(memType.getContext(),
                                           *canonicalSplitN);
              candidateEncoding &&
              !llvm::is_contained(layouts, *candidateEncoding)) {
            layouts.push_back(*candidateEncoding);
          }
        }
      }
    }
  }

  auto isCompatible = [&](const LinearLayout &layout) {
    auto candidateEncoding = tryGetLinearEncodingAttr(memType.getContext(), layout);
    if (!candidateEncoding)
      return false;
    auto candidateType = tensorTy.cloneWithEncoding(*candidateEncoding);
    return succeeded(
        computeTMemLdStEncodingInfo(candidateType, memType, /*maxnreg=*/256));
  };
  int bitwidth = memType.getElementTypeBitWidth();
  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  SmallVector<TMemAccessAtom> atoms =
      prefer16x256
          ? SmallVector<TMemAccessAtom>{TMemAccessAtom::I16x256b,
                                        TMemAccessAtom::I32x32b,
                                        TMemAccessAtom::I16x128b,
                                        TMemAccessAtom::I16x64b,
                                        TMemAccessAtom::I16x32bx2}
          : SmallVector<TMemAccessAtom>{TMemAccessAtom::I32x32b,
                                        TMemAccessAtom::I16x256b,
                                        TMemAccessAtom::I16x128b,
                                        TMemAccessAtom::I16x64b,
                                        TMemAccessAtom::I16x32bx2};
  bool debugCompatLayouts =
      std::getenv("TRITON_DEBUG_TMEM_COMPAT_LAYOUTS") != nullptr;
  for (auto atom : atoms) {
    std::optional<LinearLayout> ll;
    if (isScales) {
      ll = getDistributedLayoutForTmemLdStLegacyAnchored(memLL, atom, numWarps,
                                                         bitwidth);
    } else {
      ll = getDistributedLayoutForTmemLdSt(memType, atom, numWarps);
    }
    if (debugCompatLayouts) {
      llvm::errs() << "[tmem-compat] atom=" << getOpShape(atom)
                   << " shape=" << stringifyShape(memType.getShape())
                   << " bitwidth=" << bitwidth << "\n";
      if (ll)
        llvm::errs() << ll->toString() << "\n";
      else
        llvm::errs() << "<no layout>\n";
    }
    if (ll) {
      auto candidateEncoding =
          tryGetLinearEncodingAttr(memType.getContext(), std::move(*ll));
      if (debugCompatLayouts && !candidateEncoding)
        llvm::errs() << "[tmem-compat] reject: invalid linear attr\n";
      if (candidateEncoding) {
        auto candidateType = tensorTy.cloneWithEncoding(*candidateEncoding);
        auto ok = succeeded(computeTMemLdStEncodingInfo(candidateType, memType,
                                                        /*maxnreg=*/256));
        if (debugCompatLayouts)
          llvm::errs() << "[tmem-compat] ldst=" << (ok ? "ok" : "fail")
                       << "\n";
        if (ok) {
          layouts.push_back(*candidateEncoding);
        }
      }
    }
  }

  if (auto splitLongM = getTmemLoadLayoutSplitLongM(tensorTy, memType,
                                                    numWarps);
      splitLongM &&
      succeeded(computeTMemLdStEncodingInfo(
          tensorTy.cloneWithEncoding(splitLongM.value()), memType,
          /*maxnreg=*/256))) {
    layouts.push_back(splitLongM.value());
  }
  return layouts;
}

SmallVector<DistributedEncodingTrait>
getTmemCompatibleLayouts(Operation *op, RankedTensorType tensorType,
                         MemDescType memType) {
  int numWarps = lookupNumWarps(op);
  SmallVector<DistributedEncodingTrait> layouts;
  if (numWarps % 4 != 0)
    return layouts;
  bool isScales = isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding());
  auto tryAddScalesNarrowTileLayout = [&]() {
    if (!isScales || numWarps != 4 || memType.getElementTypeBitWidth() != 8)
      return;
    auto shape = memType.getShape();
    if (shape.size() < 2 || shape[shape.size() - 2] != 16 ||
        shape[shape.size() - 1] != 8)
      return;
    auto *ctx = tensorType.getContext();
    auto dims = standardOutDimNames(ctx, 2);
    auto kReg = StringAttr::get(ctx, "register");
    auto kLane = StringAttr::get(ctx, "lane");
    auto kWarp = StringAttr::get(ctx, "warp");
    LinearLayout::BasesT bases;
    bases[kReg] = {{0, 1}, {0, 2}, {0, 0}};
    bases[kLane] = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}};
    bases[kWarp] = {{0, 0}, {0, 0}};
    auto narrow =
        LinearLayout(std::move(bases),
                     {{dims[0], static_cast<int32_t>(shape[shape.size() - 2])},
                      {dims[1],
                       static_cast<int32_t>(shape[shape.size() - 1])}},
                     /*requireSurjective=*/false);
    if (isTMemCompatibleCandidate(op, tensorType, memType, narrow)) {
      layouts.push_back(
          LinearEncodingAttr::get(tensorType.getContext(), std::move(narrow)));
    }
  };
  tryAddScalesNarrowTileLayout();
  LinearLayout memLL = [&]() -> LinearLayout {
    if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
      return toLinearLayout(memType);
    std::string error;
    auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
        memType.getShape(), memType.getEncoding(), &error);
    if (!maybeAnalysis)
      return LinearLayout();
    return normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis);
  }();
  if (memLL.getNumOutDims() == 0)
    return layouts;
  int bitwidth = memType.getElementTypeBitWidth();
  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  SmallVector<TMemAccessAtom> atoms =
      prefer16x256
          ? SmallVector<TMemAccessAtom>{TMemAccessAtom::I16x256b,
                                        TMemAccessAtom::I32x32b,
                                        TMemAccessAtom::I16x128b,
                                        TMemAccessAtom::I16x64b,
                                        TMemAccessAtom::I16x32bx2}
          : SmallVector<TMemAccessAtom>{TMemAccessAtom::I32x32b,
                                        TMemAccessAtom::I16x256b,
                                        TMemAccessAtom::I16x128b,
                                        TMemAccessAtom::I16x64b,
                                        TMemAccessAtom::I16x32bx2};
  for (auto atom : atoms) {
    std::optional<LinearLayout> ll;
    if (isScales) {
      ll = getDistributedLayoutForTmemLdStLegacyAnchored(memLL, atom, numWarps,
                                                         bitwidth);
    } else {
      ll = getDistributedLayoutForTmemLdSt(memType, atom, numWarps);
    }
    if (ll) {
      auto candidateEncoding =
          tryGetLinearEncodingAttr(tensorType.getContext(), std::move(*ll));
      if (candidateEncoding) {
        auto candidateType = tensorType.cloneWithEncoding(*candidateEncoding);
        if (succeeded(computeTMemLdStEncodingInfo(
                candidateType, memType, getContextualMaxNReg(op)))) {
          layouts.push_back(*candidateEncoding);
        }
      }
    }
  }
  // Small hack until we generalise isDistributedLayoutTMemCompatible
  auto ll = getTmemLoadLayoutSplitLongM(tensorType, memType, numWarps);
  if (ll && succeeded(computeTMemLdStEncodingInfo(
                tensorType.cloneWithEncoding(ll.value()), memType,
                getContextualMaxNReg(op)))) {
    layouts.push_back(ll.value());
  }
  return layouts;
}

// Verify if the distributed layout can be mapped onto tensor memory.
bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       gpu::MemDescType memType) {
  auto maxnreg = getContextualMaxNReg(op);
  return succeeded(computeTMemLdStEncodingInfo(tensorType, memType, maxnreg));
}

LogicalResult
TensorMemoryEncodingAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 unsigned blockM, unsigned blockN,
                                 unsigned colStride,
                                 gpu::CGAEncodingAttr cgaLayout, bool twoCTAs) {
  if (cgaLayout.getRank() != 2) {
    return emitError() << "CGALayout must have rank 2";
  }
  if (twoCTAs) {
    auto kBlock = StringAttr::get(cgaLayout.getContext(), "block");
    auto cgaLL = cgaLayout.getLinearLayout();
    if (cgaLL.getBasis(kBlock, 0) != ArrayRef{1, 0}) {
      return emitError()
             << "twoCTAs layout requires the first CGALayout block basis to "
                "be [1, 0]";
    }
  }
  if (blockM != 64 && blockM != 128) {
    return emitError() << "blockM must be 64 or 128 but got " << blockM;
  }
  if (!llvm::isPowerOf2_32(blockN)) {
    return emitError() << "blockN must be a power of 2 but got " << blockN;
  }
  if (blockN > 512) {
    return emitError() << "blockN must be less than or equal to 512 but got "
                       << blockN;
  }
  if (!(colStride == 1 || colStride == 2 || colStride == 4)) {
    return emitError() << "colStride must be 1, 2, or 4 but got "
                       << "but got " << colStride;
  }
  return success();
}

LogicalResult TensorMemoryLinearEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, LinearLayout linearLayout,
    bool twoCTAs) {
  static const auto expectedInDims =
      SmallVector<std::string>({"row", "col", "block"});
  SmallVector<StringAttr> inDims = llvm::to_vector(linearLayout.getInDimNames());
  if (inDims.size() < 2 || inDims.size() > 3) {
    return emitError() << "Expected input dimensions [row, col] with optional "
                          "'block'. Got "
                       << inDims.size() << " inputs.";
  }
  for (auto [idx, dim] : llvm::enumerate(inDims)) {
    if (dim.str() != expectedInDims[idx]) {
      return emitError() << "Expected input dimension " << idx << " to be '"
                         << expectedInDims[idx] << "'. Got " << dim;
    }
  }
  if (inDims.size() == 2 && twoCTAs) {
    return emitError()
           << "twoCTAs requires a linear layout with a 'block' input";
  }
  for (auto [i, dim] : llvm::enumerate(linearLayout.getOutDimNames())) {
    if (dim.str() != ("dim" + llvm::Twine(i)).str()) {
      return emitError()
             << "Expected output dimensions to be ['dim0', 'dim1', ...]. Got "
             << dim << " at position " << i;
    }
  }
  if (linearLayout.getNumOutDims() == 0)
    return emitError() << "Expected at least one output dimension";

  auto *ctx = linearLayout.getOutDimNames().begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  auto stripped = linearLayout.removeZeroBasesAlongDim(kRow)
                      .removeZeroBasesAlongDim(kCol);
  if (llvm::is_contained(linearLayout.getInDimNames(), kBlock))
    stripped = stripped.removeZeroBasesAlongDim(kBlock);
  // Allow sparse/non-surjective TMEM-linear layouts when the only aliasing
  // comes from explicit zero bases. After stripping those zero bases, the
  // remaining active TMEM coordinates must still map injectively into the
  // logical tensor space. This keeps sparse/copy-only reachability paths such
  // as tcgen05.copy.warpx2 expressible, while still rejecting arbitrary
  // aliasing layouts.
  if (!stripped.isInjective()) {
    return emitError()
           << "After removing zero bases the layout must be injective";
  }
  if (twoCTAs) {
    auto bases = linearLayout.getBases().lookup(kBlock);
    if (bases.empty()) {
      return emitError()
             << "twoCTAs requires a non-empty 'block' basis sequence";
    }
    if (llvm::all_of(bases, [](ArrayRef<int32_t> basis) {
          return llvm::all_of(basis, [](int32_t value) { return value == 0; });
        })) {
      return emitError()
             << "twoCTAs requires at least one non-zero 'block' basis";
    }
  }
  return success();
}

void TensorMemoryLinearEncodingAttr::print(AsmPrinter &printer) const {
  printer << "<{";
  auto layout = getLinearLayout();
  auto kBlock = StringAttr::get(getContext(), "block");
  if (layout.getBases().lookup(kBlock).empty())
    layout = layout.sublayout({StringAttr::get(getContext(), "row"),
                               StringAttr::get(getContext(), "col")},
                              llvm::to_vector(layout.getOutDimNames()));
  triton::gpu::printLinearLayout(printer, layout);
  if (getTwoCTAs())
    printer << "}, twoCTAs = true>";
  else
    printer << "}>";
}

Attribute TensorMemoryLinearEncodingAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess().failed())
    return {};

  DictionaryAttr layoutDictRaw;
  if (parser.parseAttribute(layoutDictRaw).failed())
    return {};

  NamedAttrList layoutAttrList(layoutDictRaw.getValue());
  auto *ctx = parser.getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  if (!layoutAttrList.get(kBlock))
    layoutAttrList.push_back({kBlock, ArrayAttr::get(ctx, {})});
  DictionaryAttr layoutDict = layoutAttrList.getDictionary(ctx);

  bool twoCTAs = false;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseKeyword("twoCTAs").failed() || parser.parseEqual().failed())
      return {};
    Attribute twoCTAsAttr;
    if (parser.parseAttribute(twoCTAsAttr).failed())
      return {};
    auto boolAttr = dyn_cast<BoolAttr>(twoCTAsAttr);
    if (!boolAttr) {
      parser.emitError(parser.getCurrentLocation(),
                       "expected a boolean value for twoCTAs");
      return {};
    }
    twoCTAs = boolAttr.getValue();
  }

  if (parser.parseGreater().failed())
    return {};

  auto maybeLL = triton::gpu::parseLinearLayout(
      layoutDict, parser, {"row", "col", "block"});
  if (!maybeLL.has_value())
    return {};
  return parser.getChecked<TensorMemoryLinearEncodingAttr>(
      ctx, std::move(*maybeLL), twoCTAs);
}

gpu::CGAEncodingAttr TensorMemoryLinearEncodingAttr::getCGALayout() const {
  auto ctx = getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto ll = getLinearLayout();
  if (!llvm::is_contained(ll.getInDimNames(), kBlock))
    return CGAEncodingAttr::get1CTALayout(ctx, getRank());

  auto ctasPerCGA =
      ::mlir::triton::basesPerDimImpl(ll.getBases(), kBlock, getRank(),
                                      /*skipBroadcast=*/false);
  auto ctaSplitNum =
      ::mlir::triton::basesPerDimImpl(ll.getBases(), kBlock, getRank(),
                                      /*skipBroadcast=*/true);
  SmallVector<unsigned> defaultOrder(getRank());
  std::iota(defaultOrder.begin(), defaultOrder.end(), 0);
  auto ctaOrder =
      ::mlir::triton::orderPerDimImpl(ll, kBlock, defaultOrder);
  return CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA, ctaSplitNum,
                                          ctaOrder);
}

LogicalResult TensorMemoryScalesEncodingAttr::verify(
    function_ref<InFlightDiagnostic()> emitError,
    gpu::CGAEncodingAttr cgaLayout) {
  if (cgaLayout.getRank() != 2) {
    return emitError() << "CGALayout must have rank 2";
  }
  return success();
}

LogicalResult impl::verifyMMAv5Op(Operation *op) {
  auto isInterleaved = [](const std::optional<MMAv5AccumulatorLayoutInfo> &info) {
    return info && info->interleavedM64;
  };

  auto itf = cast<MMAv5OpInterface>(op);
  auto lhsInfo = getMMAv5LhsLayoutInfo(itf.getA().getType());
  auto accPlan = isa<TCGen5MMAScaledOp>(op)
                     ? getMMAv5ScaledAccumulatorLayoutInfo(
                           itf.getAccumulator().getType())
                     : getMMAv5AccumulatorLayoutInfo(
                           itf.getAccumulator().getType());
  if (lhsInfo && getTmemAllocSizes(itf.getA().getType()).numRows != 64 &&
      lhsInfo->mmaSizeM == 64 && isInterleaved(accPlan)) {
    return op->emitOpError(
        "does not support blockM=64 with interleaved blocks in TMEM layout");
  }
  return success();
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/OpsEnums.cpp.inc"
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Type methods
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// TensorDescIm2ColType Verifier
//===----------------------------------------------------------------------===//
LogicalResult
TensorDescIm2ColType::verify(function_ref<InFlightDiagnostic()> emitError,
                             RankedTensorType blockType) {
  // blockType must be rank 2 for im2col mode
  if (blockType.getRank() != 2) {
    return emitError()
           << "TensorDescIm2ColType requires rank-2 blockType, got rank "
           << blockType.getRank();
  }
  return success();
}

namespace {
class TritonNvidiaGPUInferLayoutInterface
    : public triton::DialectInferLayoutInterface {
public:
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return getDelegate()->inferReduceOpEncoding(operandEncoding, axis,
                                                resultEncoding, loc);
  }

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    if (isTensorMemoryEncoding(operandEncoding) &&
        !isa<TensorMemoryScalesEncodingAttr>(operandEncoding)) {
      if (triton::isIota(order)) {
        resultEncoding = operandEncoding;
        return success();
      }
      std::string error;
      auto canonicalAttr =
          tryGetCanonicalTensorMemoryEncoding(shape, operandEncoding, &error);
      if (!canonicalAttr) {
        return emitOptionalError(loc, error);
      }
      auto canonical =
          cast<TensorMemoryLinearEncodingAttr>(*canonicalAttr);
      if (canonical.getRank() != order.size()) {
        return emitOptionalError(
            loc, "TMEM transpose rank does not match the TMEM layout rank");
      }
      std::string transposeError;
      auto result =
          tryMakeTensorMemoryLinearEncoding(getDialect()->getContext(),
                                            transposeLinearLayout(
                                                canonical.getLinearLayout(),
                                                order),
                                            canonical.getTwoCTAs(),
                                            &transposeError);
      if (!result) {
        return emitOptionalError(
            loc, "TMEM transpose produced an invalid tensor memory layout: ",
            transposeError);
      }
      resultEncoding = *result;
      return success();
    }
    return getDelegate()->inferTransOpEncoding(operandEncoding, shape, order,
                                               resultEncoding, loc);
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const override {
    return getDelegate()->inferExpandDimsOpEncoding(operandEncoding, axis,
                                                    resultEncoding, loc);
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> loc) const override {
    return getDelegate()->inferDotOpEncoding(operandEncoding, opIdx,
                                             retEncoding, loc);
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const override {
    if (isTensorMemoryEncoding(srcEnc) &&
        !isa<TensorMemoryScalesEncodingAttr>(srcEnc)) {
      return inferTMemReshapeOpEncoding(srcShape, srcEnc, dstShape, dstEnc,
                                        loc);
    }
    return getDelegate()->inferReshapeOpEncoding(srcShape, srcEnc, dstShape,
                                                 dstEnc, loc);
  }

  LogicalResult
  inferMemDescIndexOpEncoding(ArrayRef<int64_t> srcShape,
                              ArrayRef<int64_t> srcAllocShape,
                              Attribute srcEncoding,
                              ArrayRef<int64_t> dstShape,
                              ArrayRef<int64_t> dstAllocShape,
                              Attribute &dstEncoding,
                              std::optional<Location> loc) const override {
    if (isTensorMemoryEncoding(srcEncoding) &&
        !isa<TensorMemoryScalesEncodingAttr>(srcEncoding)) {
      return inferTMemIndexOpEncoding(srcShape, dstShape, dstAllocShape,
                                      srcEncoding, dstEncoding, loc);
    }
    return getDelegate()->inferMemDescIndexOpEncoding(
        srcShape, srcAllocShape, srcEncoding, dstShape, dstAllocShape,
        dstEncoding, loc);
  }

  LogicalResult
  inferMemDescSubsliceOpEncoding(ArrayRef<int64_t> srcShape,
                                 ArrayRef<int64_t> srcAllocShape,
                                 Attribute srcEncoding,
                                 ArrayRef<int64_t> dstShape,
                                 ArrayRef<int32_t> offsets,
                                 Attribute &dstEncoding,
                                 std::optional<Location> loc) const override {
    if (isTensorMemoryEncoding(srcEncoding) &&
        !isa<TensorMemoryScalesEncodingAttr>(srcEncoding)) {
      return inferTMemSubsliceOpEncoding(srcShape, srcAllocShape, srcEncoding,
                                         dstShape, offsets, dstEncoding, loc);
    }
    return getDelegate()->inferMemDescSubsliceOpEncoding(
        srcShape, srcAllocShape, srcEncoding, dstShape, offsets, dstEncoding,
        loc);
  }

  LogicalResult
  inferMemDescReinterpretOpEncoding(ArrayRef<int64_t> srcShape,
                                    ArrayRef<int64_t> srcAllocShape,
                                    Type srcElementType,
                                    Attribute srcEncoding,
                                    ArrayRef<int64_t> dstShape,
                                    ArrayRef<int64_t> dstAllocShape,
                                    Type dstElementType,
                                    Attribute requestedDstEncoding,
                                    Attribute &dstEncoding,
                                    std::optional<Location> loc) const override {
    (void)srcShape;
    (void)srcAllocShape;
    (void)srcElementType;
    (void)dstAllocShape;
    (void)dstElementType;

    bool srcTMem = srcEncoding && isTensorMemoryEncoding(srcEncoding);
    bool dstTMem =
        requestedDstEncoding && isTensorMemoryEncoding(requestedDstEncoding);
    if (srcTMem || dstTMem) {
      if (!(srcTMem && dstTMem))
        return emitOptionalError(
            loc, "memdesc_reinterpret must stay in tensor memory when either "
                 "side uses a tensor memory encoding");
      if (!requestedDstEncoding)
        return emitOptionalError(
            loc, "TMEM memdesc_reinterpret requires an explicit result layout");
      std::string error;
      if (!tryGetCanonicalTensorMemoryEncoding(dstShape, requestedDstEncoding,
                                               &error)) {
        return emitOptionalError(loc, error);
      }
      dstEncoding = requestedDstEncoding;
      return success();
    }
    return getDelegate()->inferMemDescReinterpretOpEncoding(
        srcShape, srcAllocShape, srcElementType, srcEncoding, dstShape,
        dstAllocShape, dstElementType, requestedDstEncoding, dstEncoding, loc);
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    if (expected == got)
      return success();
    if (!expected || !got)
      return failure();

    bool expectedTMem = isTensorMemoryEncoding(expected);
    bool gotTMem = isTensorMemoryEncoding(got);
    if (expectedTMem || gotTMem) {
      if (!(expectedTMem && gotTMem)) {
        return emitOptionalError(loc, "Expected result encoding ", expected,
                                 " but was ", got);
      }
      if (isa<TensorMemoryScalesEncodingAttr>(expected) ||
          isa<TensorMemoryScalesEncodingAttr>(got)) {
        return emitOptionalError(loc, "Expected result encoding ", expected,
                                 " but was ", got);
      }
      std::string expectedError;
      auto expectedCanonical =
          tryGetCanonicalTensorMemoryEncoding(shape, expected, &expectedError);
      if (!expectedCanonical)
        return emitOptionalError(loc, expectedError);
      std::string gotError;
      auto gotCanonical =
          tryGetCanonicalTensorMemoryEncoding(shape, got, &gotError);
      if (!gotCanonical)
        return emitOptionalError(loc, gotError);
      if (auto expectedLinear =
              dyn_cast<TensorMemoryLinearEncodingAttr>(*expectedCanonical)) {
        if (auto gotLinear =
                dyn_cast<TensorMemoryLinearEncodingAttr>(*gotCanonical)) {
          auto expectedLayout =
              normalizeTensorMemoryLinearLayoutForAnalysis(
                  expectedLinear.getLinearLayout());
          auto gotLayout = normalizeTensorMemoryLinearLayoutForAnalysis(
              gotLinear.getLinearLayout());
          if (expectedLayout == gotLayout)
            return success();
        }
      }
      if (*expectedCanonical == *gotCanonical)
        return success();
      return emitOptionalError(loc, "Expected result encoding ", expected,
                               " but was ", got);
    }

    return getDelegate()->verifyLayoutsAreEqual(shape, expected, got, loc);
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return getDelegate()->inferDefaultJoinOpEncoding(srcEnc, dstEnc, shape,
                                                     loc);
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return getDelegate()->inferSplitOpEncoding(srcEnc, dstEnc, shape, loc);
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    return getDelegate()->verifyDotOpEncodingCompatibility(op,
                                                           operandEncodingA,
                                                           operandEncodingB);
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return getDelegate()->inferFp4ToFpOpEncoding(shape, axis, inEnc, outEnc,
                                                 fwdInference, loc);
  }

private:
  const triton::DialectInferLayoutInterface *getDelegate() const {
    Dialect *dialect =
        getDialect()->getContext()->getOrLoadDialect<triton::gpu::TritonGPUDialect>();
    auto *inferLayoutInterface =
        dyn_cast<triton::DialectInferLayoutInterface>(dialect);
    assert(inferLayoutInterface &&
           "Could not access TritonGPU layout inference interface.");
    return inferLayoutInterface;
  }
};

//===----------------------------------------------------------------------===//
// Verify Tensor/MemDesc Layout Interface
//===----------------------------------------------------------------------===//
class TritonNvidiaGPUVerifyTensorLayoutInterface
    : public triton::DialectVerifyTensorLayoutInterface {
public:
  using DialectVerifyTensorLayoutInterface::DialectVerifyTensorLayoutInterface;

  LogicalResult verifyTensorLayout(
      Attribute layout, RankedTensorType rankedTy, Operation *op,
      function_ref<InFlightDiagnostic()> makeErr) const override {
    Dialect *dialect =
        op->getContext()->getOrLoadDialect<triton::gpu::TritonGPUDialect>();
    auto *verifyLayoutInterface =
        dyn_cast<triton::DialectVerifyTensorLayoutInterface>(dialect);
    if (!verifyLayoutInterface)
      return makeErr() << "Could not access TritonGPU layout verifier.";
    return verifyLayoutInterface->verifyTensorLayout(layout, rankedTy, op,
                                                     makeErr);
  }

  LogicalResult verifyMemDescLayout(
      Attribute layout, Type type, Operation *op,
      function_ref<InFlightDiagnostic()> makeErr) const override {
    Dialect *dialect =
        op->getContext()->getOrLoadDialect<triton::gpu::TritonGPUDialect>();
    auto *verifyLayoutInterface =
        dyn_cast<triton::DialectVerifyTensorLayoutInterface>(dialect);
    if (!verifyLayoutInterface)
      return makeErr() << "Could not access TritonGPU layout verifier.";
    return verifyLayoutInterface->verifyMemDescLayout(layout, type, op,
                                                      makeErr);
  }
};

//===----------------------------------------------------------------------===//
// ASM Interface (i.e.: alias)
//===----------------------------------------------------------------------===//
class TritonGPUOpAsmInterface : public OpAsmDialectInterface {
public:
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (auto sharedAttr = mlir::dyn_cast<TensorMemoryEncodingAttr>(attr)) {
      os << "tmem";
      return AliasResult::FinalAlias;
    }
    if (mlir::isa<TensorMemoryLinearEncodingAttr>(attr)) {
      os << "tmem_linear";
      return AliasResult::FinalAlias;
    }
    if (mlir::isa<TensorMemoryScalesEncodingAttr>(attr)) {
      os << "tmem_scales";
      return AliasResult::FinalAlias;
    }
    return OpAsmDialectInterface::getAlias(attr, os);
  }
};
} // namespace

//===----------------------------------------------------------------------===//

void TritonNvidiaGPUDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/TritonNvidiaGPU/IR/Types.cpp.inc"
      >();
  addInterfaces<TritonNvidiaGPUInferLayoutInterface,
                TritonNvidiaGPUVerifyTensorLayoutInterface>();
  addInterfaces<TritonGPUOpAsmInterface>();
  addInterfaces<TritonInlinerInterface>();
}

// verify TritonNvidiaGPU ops
LogicalResult
TritonNvidiaGPUDialect::verifyOperationAttribute(Operation *op,
                                                 NamedAttribute attr) {
  // TODO: fill this.
  return success();
}

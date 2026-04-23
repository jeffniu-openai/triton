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
#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "third_party/f2reduce/f2reduce.h"

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
applyCGALayoutToTensorMemoryEncodingTile(LinearLayout tile,
                                         ArrayRef<int64_t> shape,
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
buildTensorMemoryEncodingLinearLayout(ArrayRef<int64_t> shape, unsigned blockM,
                                      unsigned blockN, unsigned colStride,
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
    return setError("unsupported tensor_memory_encoding blockM=" +
                    Twine(blockM) + "; expected 64 or 128");
  if (!llvm::isPowerOf2_32(blockN) || blockN > 512)
    return setError("unsupported tensor_memory_encoding blockN=" +
                    Twine(blockN));
  if (!(colStride == 1 || colStride == 2 || colStride == 4))
    return setError("unsupported tensor_memory_encoding colStride=" +
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
        return setError("tensor_memory_encoding twoCTA blockM=64 requires "
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
  auto maybeFullTile = applyCGALayoutToTensorMemoryEncodingTile(
      tile, shape, shapePerCTA, cgaLayout, error);
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
canonicalizeTensorMemoryEncodingLayout(ArrayRef<int64_t> shape,
                                       Attribute encoding,
                                       std::string *error = nullptr) {
  auto tmem = cast<TensorMemoryEncodingAttr>(encoding);
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
  auto base = buildTensorMemoryEncodingLinearLayout(
      trailingShape, tmem.getBlockM(), tmem.getBlockN(), tmem.getColStride(),
      tmem.getCGALayout(), tmem.getTwoCTAs(), &baseError);
  if (!base) {
    if (error != nullptr) {
      *error = "tensor memory layout sugar " + stringifyAttribute(encoding) +
               " cannot be canonicalized for shape " +
               stringifyShape(trailingShape) + ": " + baseError +
               ". Use #ttng.tensor_memory_linear for arbitrary TMEM views, or "
               "choose a tensor_memory_encoding whose tile matches the "
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
      tmem.getContext(), trailingShape,
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
  if (auto tmem = dyn_cast<TensorMemoryEncodingAttr>(layout))
    return tmem.getTwoCTAs();
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
  if (auto tmem = dyn_cast<TensorMemoryEncodingAttr>(layout)) {
    auto canonicalLayout =
        canonicalizeTensorMemoryEncodingLayout(shape, tmem, error);
    if (!canonicalLayout)
      return std::nullopt;
    return tryMakeTensorMemoryLinearEncoding(layout.getContext(),
                                             std::move(*canonicalLayout),
                                             tmem.getTwoCTAs(), error);
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
    return canonicalizeTensorMemoryEncodingLayout(shape, layout, error);
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
  auto maybeLayout = buildTensorMemoryEncodingLinearLayout(
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

LinearLayout foldCanonicalSingleCTABlockRowsForAnalysis(LinearLayout layout,
                                                        bool twoCTAs) {
  if (twoCTAs || layout.getNumInDims() == 0 || layout.getNumOutDims() != 2)
    return layout;

  auto *ctx = (*layout.getInDimNames().begin()).getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto kRow = StringAttr::get(ctx, "row");
  if (!layout.hasInDim(kBlock) || !layout.hasInDim(kRow) ||
      layout.getInDimSize(kBlock) == 1)
    return layout;

  auto outDims = llvm::to_vector(layout.getOutDimNames());
  auto rowOutDim = outDims.front();
  unsigned rowOutIdx = layout.getOutDimIndex(rowOutDim);
  if (layout.getInDimSize(kBlock) * layout.getInDimSize(kRow) !=
      layout.getOutDimSize(rowOutDim))
    return layout;

  auto isPureExpectedRowBasis = [&](ArrayRef<int32_t> basis,
                                    int32_t expected) {
    if (basis.size() != static_cast<size_t>(layout.getNumOutDims()))
      return false;
    for (auto [idx, value] : llvm::enumerate(basis)) {
      int32_t expectedValue = idx == rowOutIdx ? expected : 0;
      if (value != expectedValue)
        return false;
    }
    return true;
  };

  LinearLayout::BasesT bases = layout.getBases();
  auto blockBases = bases.lookup(kBlock);
  auto rowBases = bases.lookup(kRow);
  SmallVector<std::vector<int32_t>> foldedRowBases;
  foldedRowBases.reserve(blockBases.size() + rowBases.size());
  for (ArrayRef<int32_t> basis : blockBases)
    foldedRowBases.emplace_back(basis.begin(), basis.end());
  for (ArrayRef<int32_t> basis : rowBases)
    foldedRowBases.emplace_back(basis.begin(), basis.end());

  for (auto [idx, basis] : llvm::enumerate(foldedRowBases)) {
    if (!isPureExpectedRowBasis(basis, 1 << idx))
      return layout;
  }

  bases[kRow] = std::vector<std::vector<int32_t>>(foldedRowBases.begin(),
                                                  foldedRowBases.end());
  bases.erase(kBlock);
  return LinearLayout(std::move(bases), llvm::to_vector(layout.getOutDims()),
                      layout.isSurjective());
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
        auto maybeCandidate = buildTensorMemoryEncodingLinearLayout(
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
        auto maybeCandidate = buildTensorMemoryEncodingLinearLayout(
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
  // Plain MMAv5 accumulator layouts can be planned down to the public narrow
  // instruction N shapes. Scaled MMAv5 keeps a separate planner because its
  // matrix-B scale fragments currently impose wider alignment constraints.
  static constexpr unsigned kAccumulatorBlockNs[] = {8u,  16u,  32u,
                                                     64u, 128u, 256u};
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
  // See the Attribute overload: this is the plain accumulator planner only.
  static constexpr unsigned kAccumulatorBlockNs[] = {8u,  16u,  32u,
                                                     64u, 128u, 256u};
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
  // TMEM LHS layouts are planned in physical storage columns. Scaled fp4
  // operands pack two logical K values per byte, so a logical K=128 operand
  // has a 64-column storage image and can expose valid narrow tile-preserving
  // storage families.
  static constexpr unsigned kLhsBlockNs[] = {8u,  16u,  32u,
                                             64u, 128u, 256u};
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
  // See the Attribute overload: LHS planning uses packed storage columns.
  static constexpr unsigned kLhsBlockNs[] = {8u,  16u,  32u,
                                             64u, 128u, 256u};
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
  static constexpr unsigned kAccumulatorBlockNs[] = {8u,  16u,  32u,
                                                     64u, 128u, 256u};
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
  static constexpr unsigned kAccumulatorBlockNs[] = {8u,  16u,  32u,
                                                     64u, 128u, 256u};
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
  return buildTensorMemoryEncodingLinearLayout(
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

static std::optional<LinearLayout>
tryRestrictMMAv5ViewLayoutToShape(const LinearLayout &layout,
                                  ArrayRef<int64_t> shape) {
  if (shape.size() != static_cast<size_t>(layout.getNumOutDims()))
    return std::nullopt;

  auto restricted = layout;
  for (auto [idx, dim] : llvm::enumerate(layout.getOutDimNames())) {
    if (shape[idx] > restricted.getOutDimSize(dim))
      return std::nullopt;
    restricted = restricted.resizeOutDim(dim, shape[idx]);
  }

  // A column subview of a wider allocation carries input bits that only select
  // columns outside the active view. Drop those inactive column bases before
  // comparing against MMAv5 tile families; keep zero bases that were already
  // zero in the allocation layout, since those encode real col-stride/broadcast
  // structure rather than the narrowed view boundary.
  if (shape.size() != 2)
    return restricted;
  auto *ctx = (*layout.getOutDimNames().begin()).getContext();
  auto kCol = StringAttr::get(ctx, "col");
  if (!layout.hasInDim(kCol) || !restricted.hasInDim(kCol))
    return restricted;

  auto outDimNames = llvm::to_vector(layout.getOutDimNames());
  unsigned colOutIdx = layout.getOutDimIndex(outDimNames[1]);
  auto originalIt = layout.getBases().find(kCol);
  auto restrictedIt = restricted.getBases().find(kCol);
  assert(originalIt != layout.getBases().end() &&
         restrictedIt != restricted.getBases().end());
  const auto &originalBases = originalIt->second;
  const auto &restrictedBases = restrictedIt->second;
  if (originalBases.size() != restrictedBases.size())
    return restricted;

  bool changed = false;
  LinearLayout::BasesT bases = restricted.getBases();
  auto &columnBases = bases[kCol];
  columnBases.clear();
  for (auto [originalBasis, restrictedBasis] :
       llvm::zip(originalBases, restrictedBases)) {
    if (static_cast<int64_t>(originalBasis[colOutIdx]) >= shape[1]) {
      changed = true;
      continue;
    }
    columnBases.push_back(std::vector<int32_t>(restrictedBasis.begin(),
                                               restrictedBasis.end()));
  }
  if (!changed)
    return restricted;
  return LinearLayout(std::move(bases), restricted.getOutDims(),
                      /*requireSurjective=*/false);
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

  // Some descriptor views are MMAv5-compatible even when their backing
  // allocation is intentionally wider than any single MMAv5 instruction family.
  // Lowering uses the memdesc view op to compute the base offset, so after the
  // full-allocation family check fails we can plan against the restricted view
  // layout as long as that restricted layout is itself a supported family.
  if (auto restricted = tryRestrictMMAv5ViewLayoutToShape(*maybeCanonical, shape)) {
    if (auto plan = linearPlanner(shape, *restricted, gpu::getCGALayout(layout),
                                  *twoCTAs, preferredColStride))
      return makeInfo(shape, *plan);
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

std::optional<LinearLayout>
getMMAv5TMemFamilyAddressLayout(MemDescType memDescType) {
  auto layoutTrait = dyn_cast<LayoutEncodingTrait>(memDescType.getEncoding());
  if (!layoutTrait)
    return std::nullopt;

  if (auto info = getMMAv5AccumulatorLayoutInfo(memDescType))
    return normalizeTensorMemoryLinearLayoutForAnalysis(info->familyLayout);
  if (auto info = getMMAv5ScaledAccumulatorLayoutInfo(memDescType))
    return normalizeTensorMemoryLinearLayoutForAnalysis(info->familyLayout);
  if (auto info = getMMAv5LhsLayoutInfo(memDescType))
    return normalizeTensorMemoryLinearLayoutForAnalysis(info->familyLayout);
  return std::nullopt;
}

static SmallVector<unsigned, 6>
getMMAv5InstructionTileRequirementBlockNs(MMAv5TMemOperandKind operandKind) {
  return {8u, 16u, 32u, 64u, 128u, 256u};
}

static unsigned
getMMAv5InstructionTileRequirementPreferredColStride(MemDescType memDescType,
                                                     MMAv5TMemOperandKind kind) {
  if (kind == MMAv5TMemOperandKind::LHS)
    return 1u;
  return 32 / memDescType.getElementTypeBitWidth();
}

static std::optional<MMAv5TMemInstructionTileOrderMismatch>
getMMAv5InstructionTileOrderMismatch(MemDescType memDescType,
                                     MMAv5TMemOperandKind operandKind) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  auto allocShape = memDescType.getAllocShape().take_back(rank);
  auto twoCTAs = getTensorMemoryTwoCTAs(layout);
  if (!twoCTAs)
    return std::nullopt;

  auto blockNs = getMMAv5InstructionTileRequirementBlockNs(operandKind);
  auto preferredColStride =
      getMMAv5InstructionTileRequirementPreferredColStride(memDescType,
                                                           operandKind);
  SmallVector<unsigned, 4> colStrides;
  if (preferredColStride == 1u || preferredColStride == 2u ||
      preferredColStride == 4u)
    colStrides.push_back(preferredColStride);
  for (unsigned colStride : {1u, 2u, 4u}) {
    if (!llvm::is_contained(colStrides, colStride))
      colStrides.push_back(colStride);
  }

  auto findMismatchForShape =
      [&](ArrayRef<int64_t> familyShape)
          -> std::optional<MMAv5TMemInstructionTileOrderMismatch> {
    if (familyShape.size() != 2)
      return std::nullopt;
    auto maybeCanonical =
        tryGetCanonicalTensorMemoryLinearLayout(familyShape, layout,
                                                /*error=*/nullptr);
    if (!maybeCanonical ||
        !tensorMemoryLinearLayoutMatchesShape(*maybeCanonical, familyShape))
      return std::nullopt;

    auto normalizedLinear =
        normalizeTensorMemoryLinearLayoutForMMAv5Family(*maybeCanonical);
    if (normalizedLinear.getNumOutDims() != 2)
      return std::nullopt;

    auto *ctx = (*normalizedLinear.getOutDimNames().begin()).getContext();
    auto kRow = StringAttr::get(ctx, "row");
    auto kCol = StringAttr::get(ctx, "col");
    if (!normalizedLinear.hasInDim(kRow) || !normalizedLinear.hasInDim(kCol))
      return std::nullopt;

    auto checkDimension =
        [&](const LinearLayout &normalizedCandidate, StringAttr dim,
            StringRef dimName, unsigned tileM, unsigned tileN)
            -> std::optional<MMAv5TMemInstructionTileOrderMismatch> {
      if (!normalizedCandidate.hasInDim(dim))
        return std::nullopt;
      unsigned tileSize = dim == kRow ? tileM : tileN;
      unsigned tileBits = llvm::Log2_64(tileSize);
      unsigned actualBits = normalizedLinear.getInDimSizeLog2(dim);
      unsigned canonicalBits = normalizedCandidate.getInDimSizeLog2(dim);
      if (tileBits > actualBits || tileBits > canonicalBits)
        return std::nullopt;
      for (unsigned bit = 0; bit < tileBits; ++bit) {
        auto actualBasis = normalizedLinear.getBasis(dim, bit);
        auto canonicalBasis = normalizedCandidate.getBasis(dim, bit);
        if (llvm::equal(actualBasis, canonicalBasis))
          continue;
        return MMAv5TMemInstructionTileOrderMismatch{
            /*dimension=*/dimName.str(),
            /*inputBit=*/bit,
            /*actualBasis=*/SmallVector<int32_t, 2>(actualBasis.begin(),
                                                    actualBasis.end()),
            /*canonicalBasis=*/
            SmallVector<int32_t, 2>(canonicalBasis.begin(),
                                    canonicalBasis.end()),
            /*instrShapeM=*/tileM,
            /*instrShapeN=*/tileN};
      }
      return std::nullopt;
    };

    for (unsigned blockM : {64u, 128u}) {
      for (unsigned blockN : blockNs) {
        for (unsigned colStride : colStrides) {
          auto maybeCandidate = buildTensorMemoryEncodingLinearLayout(
              familyShape, blockM, blockN, colStride, gpu::getCGALayout(layout),
              *twoCTAs, /*error=*/nullptr);
          if (!maybeCandidate)
            continue;
          auto normalizedCandidate =
              normalizeTensorMemoryLinearLayoutForMMAv5Family(*maybeCandidate);
          if (auto mismatch = checkDimension(normalizedCandidate, kRow, "row",
                                             blockM, blockN))
            return mismatch;
          if (auto mismatch = checkDimension(normalizedCandidate, kCol, "column",
                                             blockM, blockN))
            return mismatch;
        }
      }
    }
    return std::nullopt;
  };

  if (auto mismatch = findMismatchForShape(shape))
    return mismatch;
  if (shape != allocShape)
    return findMismatchForShape(allocShape);
  return std::nullopt;
}

std::optional<MMAv5TMemInstructionTileRequirement>
getMMAv5TMemInstructionTileRequirement(MemDescType memDescType,
                                       MMAv5TMemOperandKind operandKind) {
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr>(
          memDescType.getEncoding()))
    return std::nullopt;

  switch (operandKind) {
  case MMAv5TMemOperandKind::LHS:
    if (getMMAv5LhsLayoutInfo(memDescType))
      return std::nullopt;
    break;
  case MMAv5TMemOperandKind::Accumulator:
    if (getMMAv5AccumulatorLayoutInfo(memDescType))
      return std::nullopt;
    break;
  case MMAv5TMemOperandKind::ScaledAccumulator:
    if (getMMAv5ScaledAccumulatorLayoutInfo(memDescType))
      return std::nullopt;
    break;
  }

  auto rank = cast<LayoutEncodingTrait>(memDescType.getEncoding()).getRank();
  auto shape = memDescType.getShape().take_back(rank);
  auto ctaShape =
      getShapePerCTA(getCGALayout(memDescType.getEncoding()).getCTASplitNum(),
                     shape);
  return MMAv5TMemInstructionTileRequirement{
      /*operandEncoding=*/memDescType.getEncoding(),
      /*operandKind=*/operandKind,
      /*logicalShape=*/SmallVector<int64_t, 4>(shape.begin(), shape.end()),
      /*ctaShape=*/SmallVector<int64_t, 4>(ctaShape.begin(), ctaShape.end()),
      /*elementBitWidth=*/memDescType.getElementTypeBitWidth(),
      /*minimumInstructionRows=*/64u,
      /*minimumInstructionColumns=*/
      getMMAv5InstructionTileRequirementBlockNs(operandKind).front(),
      /*tileOrderMismatch=*/
      getMMAv5InstructionTileOrderMismatch(memDescType, operandKind)};
}

static StringRef stringifyMMAv5TMemOperandKind(
    MMAv5TMemOperandKind operandKind) {
  switch (operandKind) {
  case MMAv5TMemOperandKind::LHS:
    return "LHS operand";
  case MMAv5TMemOperandKind::Accumulator:
    return "return operand";
  case MMAv5TMemOperandKind::ScaledAccumulator:
    return "block-scaled accumulator operand";
  }
  llvm_unreachable("unknown MMAv5 tensor-memory operand kind");
}

std::string getMMAv5TMemInstructionTileRequirementError(
    const MMAv5TMemInstructionTileRequirement &requirement) {
  std::string message;
  llvm::raw_string_ostream os(message);
  if (requirement.operandKind == MMAv5TMemOperandKind::ScaledAccumulator) {
    os << "expected accumulator layout to be directly supported MMAv5 "
          "block-scaled tensor memory, but got "
       << requirement.operandEncoding
       << ". Block-scaled tcgen05.mma currently requires a directly supported "
          "MMAv5 tensor-memory linear layout; tile-permuted accumulator "
          "layouts are not directly representable.";
    return os.str();
  }
  os << stringifyMMAv5TMemOperandKind(requirement.operandKind)
     << " must have a MMAv5-compatible tensor memory layout, but got "
     << requirement.operandEncoding
     << ". Use a directly supported #ttng.tensor_memory_linear layout, or "
        "reshape/permute the descriptor to a supported MMAv5 tile.";
  return os.str();
}

static void printMMAv5RequirementShape(llvm::raw_ostream &os,
                                       ArrayRef<int64_t> shape) {
  for (auto [index, extent] : llvm::enumerate(shape)) {
    if (index)
      os << "x";
    os << extent;
  }
}

static void printMMAv5RequirementBasis(llvm::raw_ostream &os,
                                       ArrayRef<int32_t> basis) {
  os << "[";
  for (auto [index, value] : llvm::enumerate(basis)) {
    if (index)
      os << ", ";
    os << value;
  }
  os << "]";
}

std::string getMMAv5TMemInstructionTileRequirementNote(
    const MMAv5TMemInstructionTileRequirement &requirement) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "MMAv5 instruction-tile order requirement: "
     << stringifyMMAv5TMemOperandKind(requirement.operandKind)
     << " has logical shape ";
  printMMAv5RequirementShape(os, requirement.logicalShape);
  os << ", CTA shape ";
  printMMAv5RequirementShape(os, requirement.ctaShape);
  os << ", element bitwidth " << requirement.elementBitWidth
     << ". Public tcgen05.mma atoms require each "
     << requirement.minimumInstructionRows << "x"
     << requirement.minimumInstructionColumns
     << " or larger instruction tile to preserve the canonical row/column "
        "basis order";
  if (requirement.tileOrderMismatch) {
    const auto &mismatch = *requirement.tileOrderMismatch;
    os << "; first noncanonical in-tile basis is " << mismatch.dimension
       << " input bit " << mismatch.inputBit << " for candidate tile "
       << mismatch.instrShapeM << "x" << mismatch.instrShapeN
       << ", got physical delta ";
    printMMAv5RequirementBasis(os, mismatch.actualBasis);
    os << " but the canonical delta is ";
    printMMAv5RequirementBasis(os, mismatch.canonicalBasis);
  }
  os << ". Arbitrary row or column permutations inside a tile need an "
        "unsupported permutation or masked writeback schedule.";
  return os.str();
}

static std::optional<MMAv5ScaledRepeatedN32ScaleFragmentRequirement>
getMMAv5ScaledRepeatedN32ScaleFragmentRequirement(
    MemDescType memDescType, const MMAv5AccumulatorLayoutInfo &info) {
  auto ctaShape =
      getShapePerCTA(getCGALayout(memDescType.getEncoding()).getCTASplitNum(),
                     memDescType.getShape());
  if (ctaShape.size() < 2)
    return std::nullopt;

  auto instrSizeN = std::min<unsigned>(info.mmaSizeN, ctaShape[1]);
  if (instrSizeN != 32 ||
      (ctaShape[1] + instrSizeN - 1) / instrSizeN <= 1) {
    return std::nullopt;
  }

  return MMAv5ScaledRepeatedN32ScaleFragmentRequirement{
      /*accumulatorEncoding=*/memDescType.getEncoding(),
      /*instrSizeN=*/instrSizeN,
      /*ctaColumns=*/static_cast<unsigned>(ctaShape[1]),
      /*nInstructionCount=*/static_cast<unsigned>(
          (ctaShape[1] + instrSizeN - 1) / instrSizeN),
      /*minimumAddressableBScaleFragmentN=*/64};
}

static std::optional<MMAv5ScaledNarrowNScaleFragmentRequirement>
getMMAv5ScaledNarrowNScaleFragmentRequirement(
    MemDescType memDescType, const MMAv5AccumulatorLayoutInfo &info);

MMAv5ScaledAccumulatorSupport
getMMAv5ScaledAccumulatorSupport(MemDescType memDescType) {
  MMAv5ScaledAccumulatorSupport support;
  support.layoutInfo = getMMAv5ScaledAccumulatorLayoutInfo(memDescType);
  if (support.layoutInfo) {
    support.repeatedN32ScaleFragmentRequirement =
        getMMAv5ScaledRepeatedN32ScaleFragmentRequirement(
            memDescType, *support.layoutInfo);
    support.narrowNScaleFragmentRequirement =
        getMMAv5ScaledNarrowNScaleFragmentRequirement(memDescType,
                                                      *support.layoutInfo);
  } else {
    support.narrowNScaleFragmentRequirement =
        getMMAv5ScaledNarrowNScaleFragmentRequirement(memDescType);
  }
  return support;
}

static unsigned ceilDivPositive(unsigned numerator, unsigned denominator) {
  assert(denominator > 0);
  return (numerator + denominator - 1) / denominator;
}

MMAv5ScaledMxfpKind
getMMAv5ScaledMxfpKind(ScaleDotElemType typeA, ScaleDotElemType typeB,
                       Type scaleAType, Type scaleBType,
                       bool hasTransposedOperand) {
  if (typeA == ScaleDotElemType::E2M1 && typeB == ScaleDotElemType::E2M1) {
    if (llvm::isa<Float8E4M3FNType>(scaleAType) &&
        llvm::isa<Float8E4M3FNType>(scaleBType)) {
      assert(!hasTransposedOperand &&
             "MMAv5 with kind=mxf4nvf4 does not support transpose");
      return MMAv5ScaledMxfpKind::Mxf4NvF4;
    }
    if (!hasTransposedOperand)
      return MMAv5ScaledMxfpKind::Mxf4;
  }
  return MMAv5ScaledMxfpKind::Mxf8f6f4;
}

bool isMMAv5ScaledMxfp4(MMAv5ScaledMxfpKind kind) {
  return kind != MMAv5ScaledMxfpKind::Mxf8f6f4;
}

unsigned getMMAv5ScaledFormatBitSize(ScaleDotElemType type) {
  switch (type) {
  case ScaleDotElemType::E4M3:
  case ScaleDotElemType::E5M2:
    return 8;
  case ScaleDotElemType::E2M3:
  case ScaleDotElemType::E3M2:
    return 6;
  case ScaleDotElemType::E2M1:
    return 4;
  default:
    llvm_unreachable("Unsupported type.");
  }
}

unsigned getMMAv5ScaleFactorColsPerSet(MMAv5ScaledMxfpKind kind) {
  switch (kind) {
  case MMAv5ScaledMxfpKind::Mxf8f6f4:
    return 1;
  case MMAv5ScaledMxfpKind::Mxf4:
    return 2;
  case MMAv5ScaledMxfpKind::Mxf4NvF4:
    return 4;
  default:
    llvm_unreachable("Unsupported mxfp kind.");
  }
}

MMAv5ScaledInstructionInfo
getMMAv5ScaledInstructionInfo(ScaleDotElemType typeA, ScaleDotElemType typeB,
                              Type scaleAType, Type scaleBType,
                              bool hasTransposedOperand) {
  auto kind = getMMAv5ScaledMxfpKind(typeA, typeB, scaleAType, scaleBType,
                                     hasTransposedOperand);
  bool isMxfp4 = isMMAv5ScaledMxfp4(kind);
  return MMAv5ScaledInstructionInfo{
      /*kind=*/kind,
      /*isMxfp4=*/isMxfp4,
      /*mmaSizeK=*/isMxfp4 ? 64u : 32u,
      /*numBitsPerElementA=*/getMMAv5ScaledFormatBitSize(typeA),
      /*numBitsPerElementB=*/getMMAv5ScaledFormatBitSize(typeB),
      /*scaleFactorColsPerSet=*/getMMAv5ScaleFactorColsPerSet(kind)};
}

MMAv5ScaleFactorFragment getMMAv5ScaleFactorFragment(
    unsigned nonKRep, unsigned kRep, unsigned numRepNonK, unsigned numRepK,
    unsigned numTMemScaleCols, unsigned scaleFactorColsPerSet,
    unsigned minColsPerScaleBlock) {
  assert(numRepNonK > 0 && numRepK > 0);
  assert(scaleFactorColsPerSet > 0 && scaleFactorColsPerSet <= 4);
  assert(4 % scaleFactorColsPerSet == 0);

  unsigned colsPerWord = 4 / scaleFactorColsPerSet;
  unsigned columnsPerScaleBlock =
      ceilDivPositive(numTMemScaleCols,
                      numRepNonK * ceilDivPositive(numRepK, colsPerWord));
  columnsPerScaleBlock =
      std::max(columnsPerScaleBlock, minColsPerScaleBlock);

  unsigned subWordIdx = kRep % colsPerWord;
  unsigned wordIdx = kRep / colsPerWord;
  return MMAv5ScaleFactorFragment{
      /*tmemColumnOffset=*/(nonKRep + wordIdx * numRepNonK) *
          columnsPerScaleBlock,
      /*subColumnId=*/subWordIdx,
      /*columnsPerScaleBlock=*/columnsPerScaleBlock,
      /*wordIndex=*/wordIdx};
}

std::optional<MMAv5ScaledRepeatedN32ScaleFragmentRequirement>
getMMAv5ScaledRepeatedN32ScaleFragmentRequirement(MemDescType memDescType) {
  return getMMAv5ScaledAccumulatorSupport(memDescType)
      .repeatedN32ScaleFragmentRequirement;
}

std::optional<MemDescType>
getMMAv5ScaleTMemTypeForSharedScale(MemDescType sharedScaleType,
                                    int64_t rows) {
  if (rows <= 0 ||
      !isa<SharedMemorySpaceAttr>(sharedScaleType.getMemorySpace()))
    return std::nullopt;

  auto shape = sharedScaleType.getShape();
  if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
      shape[0] < rows || shape[0] % rows != 0)
    return std::nullopt;

  MLIRContext *ctx = sharedScaleType.getContext();
  auto cgaLayout = getCGALayout(sharedScaleType.getEncoding());
  auto scaleEncoding = TensorMemoryScalesEncodingAttr::get(ctx, cgaLayout);
  return MemDescType::get(shape, sharedScaleType.getElementType(),
                          scaleEncoding, TensorMemorySpaceAttr::get(ctx),
                          /*mutableMemory=*/true);
}

static bool hasExact2DBasisSequence(
    const LinearLayout &layout, StringAttr dim,
    ArrayRef<std::array<int32_t, 2>> expected) {
  if (!layout.hasInDim(dim) ||
      layout.getInDimSizeLog2(dim) != expected.size())
    return false;
  for (auto [idx, expectedBasis] : llvm::enumerate(expected)) {
    auto basis = layout.getBasis(dim, idx);
    if (basis.size() != 2 || basis[0] != expectedBasis[0] ||
        basis[1] != expectedBasis[1])
      return false;
  }
  return true;
}

static void padZero2DBases(SmallVectorImpl<std::array<int32_t, 2>> &bases,
                           unsigned count) {
  while (bases.size() < count)
    bases.push_back({0, 0});
}

static std::optional<LinearLayout>
getNormalizedMMAv5Rank2I8LinearScaleStorageLayout(MemDescType scaleType) {
  if (!scaleType || scaleType.getRank() != 2 ||
      scaleType.getElementTypeBitWidth() != 8 ||
      !isTensorMemoryEncoding(scaleType.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(scaleType.getEncoding()))
    return std::nullopt;

  auto linear =
      dyn_cast<TensorMemoryLinearEncodingAttr>(scaleType.getEncoding());
  if (!linear)
    return std::nullopt;

  auto shape = scaleType.getShape();
  if (shape[0] < 16 || shape[1] < 4 || !llvm::isPowerOf2_64(shape[0]) ||
      !llvm::isPowerOf2_64(shape[1]))
    return std::nullopt;

  std::string layoutError;
  auto maybeLayout = getTMemViewAnalysisLinearLayout(
      scaleType.getShape(), scaleType.getEncoding(), &layoutError);
  if (!maybeLayout)
    return std::nullopt;
  LinearLayout layout =
      normalizeTensorMemoryLinearLayoutForAnalysis(*maybeLayout);

  auto outDims = llvm::to_vector(layout.getOutDimSizes());
  if (outDims.size() != 2 || outDims[0] != shape[0] ||
      outDims[1] != shape[1])
    return std::nullopt;

  auto *ctx = scaleType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol))
    return std::nullopt;
  return layout;
}

static bool isMMAv5UnpaddedInterleavedScaleDescriptorViewStorage(
    MemDescType scaleType, const LinearLayout &layout) {
  auto shape = scaleType.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  if (rows < 32)
    return false;

  auto *ctx = scaleType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  unsigned rowBasisCount = layout.getInDimSizeLog2(kRow);
  unsigned colBasisCount = layout.getInDimSizeLog2(kCol);

  SmallVector<std::array<int32_t, 2>> rowBases = {
      {static_cast<int32_t>(rows / 2), 0}};
  for (int32_t row = 1; row <= 8 && row < rows / 2; row <<= 1)
    rowBases.push_back({row, 0});
  padZero2DBases(rowBases, rowBasisCount);

  SmallVector<std::array<int32_t, 2>> colBases = {{0, 1}, {0, 2}};
  for (int32_t row = 16; row <= rows / 4; row <<= 1)
    colBases.push_back({row, 0});
  for (int32_t col = 4; col < cols; col <<= 1)
    colBases.push_back({0, col});

  return rowBases.size() == rowBasisCount &&
         colBases.size() == colBasisCount &&
         hasExact2DBasisSequence(layout, kRow, rowBases) &&
         hasExact2DBasisSequence(layout, kCol, colBases);
}

std::optional<MemDescType> getMMAv5ScaleStorageType(MemDescType scaleType) {
  if (!scaleType)
    return std::nullopt;
  if (isa<TensorMemoryScalesEncodingAttr>(scaleType.getEncoding()))
    return scaleType;
  auto layout = getNormalizedMMAv5Rank2I8LinearScaleStorageLayout(scaleType);
  if (!layout ||
      !isMMAv5UnpaddedInterleavedScaleDescriptorViewStorage(scaleType,
                                                            *layout))
    return std::nullopt;

  MLIRContext *ctx = scaleType.getContext();
  auto scaleEncoding = TensorMemoryScalesEncodingAttr::get(
      ctx, getCGALayout(scaleType.getEncoding()));
  return MemDescType::get(scaleType.getShape(), scaleType.getElementType(),
                          scaleEncoding, scaleType.getMemorySpace(),
                          scaleType.getMutableMemory());
}

static bool isMMAv5ScaledBScaleDescriptorViewStorage(MemDescType bScaleType) {
  if (bScaleType) {
    if (auto linear = dyn_cast_if_present<TensorMemoryLinearEncodingAttr>(
            bScaleType.getEncoding())) {
      if (linear.getTwoCTAs())
        return false;
    }
  }
  auto maybeLayout = getNormalizedMMAv5Rank2I8LinearScaleStorageLayout(
      bScaleType);
  if (!maybeLayout)
    return false;
  auto shape = bScaleType.getShape();
  int64_t rows = shape[0];
  int64_t cols = shape[1];
  LinearLayout layout = *maybeLayout;

  auto *ctx = bScaleType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  unsigned rowBasisCount = layout.getInDimSizeLog2(kRow);
  unsigned colBasisCount = layout.getInDimSizeLog2(kCol);

  auto matchesPaddedStorageView = [&]() {
    SmallVector<std::array<int32_t, 2>> rowBases;
    for (int32_t row = 1; row <= 16 && row < rows; row <<= 1)
      rowBases.push_back({row, 0});
    padZero2DBases(rowBases, rowBasisCount);

    SmallVector<std::array<int32_t, 2>> colBases = {{0, 1}, {0, 2}};
    for (int32_t row = 32; row < rows; row <<= 1)
      colBases.push_back({row, 0});
    for (int32_t col = 4; col < cols; col <<= 1)
      colBases.push_back({0, col});

    return rowBases.size() == rowBasisCount &&
           colBases.size() == colBasisCount &&
           hasExact2DBasisSequence(layout, kRow, rowBases) &&
           hasExact2DBasisSequence(layout, kCol, colBases);
  };

  return matchesPaddedStorageView() ||
         isMMAv5UnpaddedInterleavedScaleDescriptorViewStorage(bScaleType,
                                                              layout);
}

std::optional<MemDescType>
getMMAv5ScaledBScaleStorageType(MemDescType bScaleType) {
  if (!bScaleType)
    return std::nullopt;
  if (isa<TensorMemoryScalesEncodingAttr>(bScaleType.getEncoding()))
    return bScaleType;
  if (!isMMAv5ScaledBScaleDescriptorViewStorage(bScaleType))
    return std::nullopt;

  MLIRContext *ctx = bScaleType.getContext();
  auto scaleEncoding = TensorMemoryScalesEncodingAttr::get(
      ctx, getCGALayout(bScaleType.getEncoding()));
  return MemDescType::get(bScaleType.getShape(), bScaleType.getElementType(),
                          scaleEncoding, bScaleType.getMemorySpace(),
                          bScaleType.getMutableMemory());
}

std::optional<MemDescType>
getMMAv5ScaledBScaleStorageTypeThroughViews(Value bScale) {
  auto bScaleType = dyn_cast<MemDescType>(bScale.getType());
  if (!bScaleType)
    return std::nullopt;
  if (auto typeLocal = getMMAv5ScaledBScaleStorageType(bScaleType))
    return typeLocal;

  Value current = bScale;
  while (Operation *defOp = current.getDefiningOp()) {
    if (!defOp->hasTrait<OpTrait::MemDescViewTrait>() ||
        defOp->getNumOperands() == 0)
      return std::nullopt;

    current = defOp->getOperand(0);
    auto currentType = dyn_cast<MemDescType>(current.getType());
    if (!currentType)
      return std::nullopt;
    if (!isa<TensorMemoryScalesEncodingAttr>(currentType.getEncoding()))
      continue;
    if (currentType.getElementType() != bScaleType.getElementType() ||
        currentType.getMemorySpace() != bScaleType.getMemorySpace())
      return std::nullopt;
    return MemDescType::get(bScaleType.getShape(), bScaleType.getElementType(),
                            currentType.getEncoding(),
                            bScaleType.getMemorySpace(),
                            bScaleType.getMutableMemory());
  }
  return std::nullopt;
}

static std::optional<unsigned> getMMAv5ScaledRepeatedN32PaddingFactor(
    const MMAv5ScaledRepeatedN32ScaleFragmentRequirement &requirement) {
  if (requirement.instrSizeN == 0 ||
      requirement.minimumAddressableBScaleFragmentN % requirement.instrSizeN !=
          0)
    return std::nullopt;
  unsigned factor =
      requirement.minimumAddressableBScaleFragmentN / requirement.instrSizeN;
  if (factor <= 1)
    return std::nullopt;
  return factor;
}

static std::optional<unsigned> getMMAv5ScaledNarrowNPaddingFactor(
    const MMAv5ScaledNarrowNScaleFragmentRequirement &requirement) {
  if (requirement.instrSizeN == 0 ||
      requirement.minimumAddressableBScaleFragmentN % requirement.instrSizeN !=
          0)
    return std::nullopt;
  unsigned factor =
      requirement.minimumAddressableBScaleFragmentN / requirement.instrSizeN;
  if (factor <= 1)
    return std::nullopt;
  return factor;
}

static bool isMMAv5ScaledBScaleStoragePadded(MemDescType bScaleType,
                                             unsigned ctaColumns,
                                             unsigned paddingFactor) {
  if (!isa<TensorMemoryScalesEncodingAttr>(bScaleType.getEncoding()))
    return false;
  auto shape = bScaleType.getShape();
  if (shape.size() != 2)
    return false;
  return shape[0] >= static_cast<int64_t>(ctaColumns) * paddingFactor;
}

static std::optional<SmallVector<int64_t>>
getMMAv5ScaledBScaleRematerializedShape(MemDescType bScaleType,
                                        unsigned ctaColumns,
                                        unsigned paddingFactor) {
  if (!isa<TensorMemoryScalesEncodingAttr>(bScaleType.getEncoding()))
    return std::nullopt;
  SmallVector<int64_t> shape(bScaleType.getShape().begin(),
                             bScaleType.getShape().end());
  if (shape.size() != 2)
    return std::nullopt;
  int64_t &rows = shape[0];
  if (rows != static_cast<int64_t>(ctaColumns))
    return std::nullopt;
  rows *= paddingFactor;
  return shape;
}

bool isMMAv5ScaledRepeatedN32BScaleStorageSupported(
    MemDescType bScaleType,
    const MMAv5ScaledRepeatedN32ScaleFragmentRequirement &requirement) {
  auto paddingFactor = getMMAv5ScaledRepeatedN32PaddingFactor(requirement);
  if (!paddingFactor)
    return false;
  return isMMAv5ScaledBScaleStoragePadded(
      bScaleType, requirement.ctaColumns, *paddingFactor);
}

std::optional<SmallVector<int64_t>>
getMMAv5ScaledRepeatedN32BScaleRematerializedShape(
    MemDescType bScaleType,
    const MMAv5ScaledRepeatedN32ScaleFragmentRequirement &requirement) {
  auto paddingFactor = getMMAv5ScaledRepeatedN32PaddingFactor(requirement);
  if (!paddingFactor)
    return std::nullopt;
  return getMMAv5ScaledBScaleRematerializedShape(
      bScaleType, requirement.ctaColumns, *paddingFactor);
}

bool isMMAv5ScaledNarrowNBScaleStorageSupported(
    MemDescType bScaleType,
    const MMAv5ScaledNarrowNScaleFragmentRequirement &requirement) {
  if (requirement.nInstructionCount <= 1)
    return isMMAv5ScaledBScaleStoragePadded(
        bScaleType, requirement.ctaColumns, /*paddingFactor=*/1);
  auto paddingFactor = getMMAv5ScaledNarrowNPaddingFactor(requirement);
  if (!paddingFactor)
    return false;
  return isMMAv5ScaledBScaleStoragePadded(
      bScaleType, requirement.ctaColumns, *paddingFactor);
}

std::optional<SmallVector<int64_t>>
getMMAv5ScaledNarrowNBScaleRematerializedShape(
    MemDescType bScaleType,
    const MMAv5ScaledNarrowNScaleFragmentRequirement &requirement) {
  auto paddingFactor = getMMAv5ScaledNarrowNPaddingFactor(requirement);
  if (!paddingFactor)
    return std::nullopt;
  return getMMAv5ScaledBScaleRematerializedShape(
      bScaleType, requirement.ctaColumns, *paddingFactor);
}

static std::optional<MMAv5ScaledNarrowNScaleFragmentRequirement>
getMMAv5ScaledNarrowNScaleFragmentRequirement(
    MemDescType memDescType, const MMAv5AccumulatorLayoutInfo &info) {
  constexpr unsigned kMinimumScaledInstrSizeN = 32;

  auto ctaShape =
      getShapePerCTA(getCGALayout(memDescType.getEncoding()).getCTASplitNum(),
                     memDescType.getShape());
  if (ctaShape.size() < 2)
    return std::nullopt;

  auto instrSizeN = std::min<unsigned>(info.mmaSizeN, ctaShape[1]);
  if (instrSizeN >= kMinimumScaledInstrSizeN)
    return std::nullopt;

  unsigned minimumAddressableBScaleFragmentN = 64;
  return MMAv5ScaledNarrowNScaleFragmentRequirement{
      /*accumulatorEncoding=*/memDescType.getEncoding(),
      /*logicalShape=*/
      SmallVector<int64_t, 4>(memDescType.getShape().begin(),
                              memDescType.getShape().end()),
      /*ctaShape=*/SmallVector<int64_t, 4>(ctaShape.begin(), ctaShape.end()),
      /*plainInstrSizeM=*/info.mmaSizeM,
      /*instrSizeN=*/instrSizeN,
      /*minimumScaledInstrSizeN=*/kMinimumScaledInstrSizeN,
      /*minimumAddressableBScaleFragmentN=*/minimumAddressableBScaleFragmentN,
      /*ctaColumns=*/static_cast<unsigned>(ctaShape[1]),
      /*nInstructionCount=*/
      static_cast<unsigned>((ctaShape[1] + instrSizeN - 1) / instrSizeN),
      /*bScalePaddingFactor=*/minimumAddressableBScaleFragmentN / instrSizeN,
      /*tileOrderMismatch=*/getMMAv5InstructionTileOrderMismatch(
          memDescType, MMAv5TMemOperandKind::ScaledAccumulator)};
}

std::optional<MMAv5ScaledNarrowNScaleFragmentRequirement>
getMMAv5ScaledNarrowNScaleFragmentRequirement(MemDescType memDescType) {
  if (getMMAv5ScaledAccumulatorLayoutInfo(memDescType))
    return std::nullopt;

  auto plainInfo = getMMAv5AccumulatorLayoutInfo(memDescType);
  if (!plainInfo)
    return std::nullopt;
  return getMMAv5ScaledNarrowNScaleFragmentRequirement(memDescType,
                                                       *plainInfo);
}

std::optional<MMAv5ScaledMixedFp4ATMemRequirement>
getMMAv5ScaledMixedFp4ATMemRequirement(MemDescType lhsType,
                                       ScaleDotElemType typeA,
                                       ScaleDotElemType typeB) {
  if (!isa<TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr>(
          lhsType.getEncoding())) {
    return std::nullopt;
  }
  if (typeA != ScaleDotElemType::E2M1 || typeB == ScaleDotElemType::E2M1)
    return std::nullopt;
  if (!getMMAv5LhsLayoutInfo(lhsType))
    return std::nullopt;
  auto rank = cast<LayoutEncodingTrait>(lhsType.getEncoding()).getRank();
  auto shape = lhsType.getShape().take_back(rank);
  auto ctaShape =
      getShapePerCTA(getCGALayout(lhsType.getEncoding()).getCTASplitNum(),
                     shape);
  unsigned lhsLogicalBitWidth = getMMAv5ScaledFormatBitSize(typeA);
  unsigned rhsLogicalBitWidth = getMMAv5ScaledFormatBitSize(typeB);
  unsigned lhsStorageColumns = shape.size() >= 2 ? shape[1] : 0;
  unsigned lhsLogicalK =
      lhsLogicalBitWidth == 0 ? 0 : lhsStorageColumns * (8 / lhsLogicalBitWidth);
  return MMAv5ScaledMixedFp4ATMemRequirement{
      /*lhsEncoding=*/lhsType.getEncoding(),
      /*lhsType=*/typeA,
      /*rhsType=*/typeB,
      /*lhsShape=*/SmallVector<int64_t, 4>(shape.begin(), shape.end()),
      /*lhsCTAShape=*/SmallVector<int64_t, 4>(ctaShape.begin(), ctaShape.end()),
      /*lhsLogicalBitWidth=*/lhsLogicalBitWidth,
      /*rhsLogicalBitWidth=*/rhsLogicalBitWidth,
      /*lhsStorageColumns=*/lhsStorageColumns,
      /*lhsLogicalK=*/lhsLogicalK,
      /*fp4PaddedGroupOffsets=*/16u,
      /*fp4PaddedRealOffsets=*/8u,
      /*requiredFp4PaddedSwizzleBytes=*/128u};
}

std::string getMMAv5ScaledRepeatedN32ScaleFragmentError(
    const MMAv5ScaledRepeatedN32ScaleFragmentRequirement &requirement) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "direct block-scaled MMAv5 does not support repeated N=32 "
        "instructions along N for "
     << requirement.accumulatorEncoding
     << ". The public tensor-memory scales layout only exposes matrix-B scale "
        "fragments at "
     << requirement.minimumAddressableBScaleFragmentN
     << "-column alignment, so layouts that would need "
     << requirement.nInstructionCount
     << " separate N=" << requirement.instrSizeN
     << " scaled instructions must be reshaped to a larger directly supported "
        "MMAv5 tile or rematerialize the matrix-B scale fragment storage.";
  return os.str();
}

std::string getMMAv5ScaledNarrowNScaleFragmentError(
    const MMAv5ScaledNarrowNScaleFragmentRequirement &requirement) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "direct block-scaled MMAv5 cannot use unpadded B-scale storage for "
        "accumulator layouts that require N="
     << requirement.instrSizeN << " instructions along N for "
     << requirement.accumulatorEncoding
     << ". The accumulator logical shape is ";
  printMMAv5RequirementShape(os, requirement.logicalShape);
  os << " with CTA shape ";
  printMMAv5RequirementShape(os, requirement.ctaShape);
  os << "; the plain MMAv5-compatible plan would use "
     << requirement.plainInstrSizeM << "x" << requirement.instrSizeN
     << " accumulator instructions and "
     << requirement.nInstructionCount << " instruction fragments along N. The "
        "public tensor-memory scales layout exposes matrix-B scale fragments "
        "at "
     << requirement.minimumAddressableBScaleFragmentN
     << "-column alignment, so this schedule would require a B-scale storage "
        "padding/rematerialization factor of "
     << requirement.bScalePaddingFactor << ".";
  if (requirement.tileOrderMismatch) {
    const auto &mismatch = *requirement.tileOrderMismatch;
    os << " The first scaled-MMAv5 in-tile order mismatch is "
       << mismatch.dimension << " input bit " << mismatch.inputBit
       << " for candidate tile " << mismatch.instrShapeM << "x"
       << mismatch.instrShapeN << ", got physical delta ";
    printMMAv5RequirementBasis(os, mismatch.actualBasis);
    os << " but the canonical delta is ";
    printMMAv5RequirementBasis(os, mismatch.canonicalBasis);
    os << ".";
  }
  os << " This " << requirement.ctaColumns
     << "-column CTA tile must use padded B-scale storage or be reshaped to a "
        "larger directly supported scale-fragment tile before it can use "
        "block-scaled tcgen05.mma.";
  return os.str();
}

std::string getMMAv5ScaledMixedFp4ATMemError(
    const MMAv5ScaledMixedFp4ATMemRequirement &requirement) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "direct block-scaled MMAv5 does not support mixed-precision fp4 LHS "
        "operands in tensor memory for "
     << requirement.lhsEncoding << " (A="
     << mlir::triton::stringifyScaleDotElemType(requirement.lhsType)
     << ", B="
     << mlir::triton::stringifyScaleDotElemType(requirement.rhsType)
     << ", A bits=" << requirement.lhsLogicalBitWidth
     << ", B bits=" << requirement.rhsLogicalBitWidth << "). The LHS tensor "
        "memory storage shape is ";
  printMMAv5RequirementShape(os, requirement.lhsShape);
  os << " with CTA shape ";
  printMMAv5RequirementShape(os, requirement.lhsCTAShape);
  os << ", raw storage K columns " << requirement.lhsStorageColumns
     << ", and logical K " << requirement.lhsLogicalK
     << ". Mixed mxf8f6f4 fp4 LHS operands require the padded operand-A "
        "storage model represented by fp4_padded shared memory: each "
     << requirement.fp4PaddedGroupOffsets
     << "-offset group contains only " << requirement.fp4PaddedRealOffsets
     << " real packed fp4 values and the remaining offsets are padding "
        "aliases under the required "
     << requirement.requiredFp4PaddedSwizzleBytes
     << "-byte swizzle. Linear tensor-memory LHS storage currently exposes raw "
        "packed columns directly, so it cannot model that fp4_padded operand-A "
        "contract. Use shared memory for operand A, or use a homogeneous fp4 "
        "scaled-MMA kind whose TMEM LHS storage is directly modeled.";
  return os.str();
}

std::optional<std::string>
getMMAv5ScaledRepeatedN32ScaleFragmentError(MemDescType memDescType) {
  auto requirement =
      getMMAv5ScaledRepeatedN32ScaleFragmentRequirement(memDescType);
  if (!requirement)
    return std::nullopt;
  return getMMAv5ScaledRepeatedN32ScaleFragmentError(*requirement);
}

std::optional<std::string>
getMMAv5ScaledNarrowNScaleFragmentError(MemDescType memDescType) {
  auto requirement = getMMAv5ScaledNarrowNScaleFragmentRequirement(memDescType);
  if (!requirement)
    return std::nullopt;
  return getMMAv5ScaledNarrowNScaleFragmentError(*requirement);
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

static std::optional<SmallVector<int32_t>>
getPurePowerOfTwoBasisOrder(const LinearLayout &layout, StringAttr inDim,
                            unsigned outDimIdx, int64_t extent) {
  if (!layout.hasInDim(inDim) || extent < 1 || !llvm::isPowerOf2_64(extent) ||
      layout.getInDimSize(inDim) != extent)
    return std::nullopt;

  SmallVector<int32_t> bases;
  bases.reserve(layout.getInDimSizeLog2(inDim));
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(inDim); ++idx) {
    auto basis = layout.getBasis(inDim, idx);
    if (basis.size() != static_cast<size_t>(layout.getNumOutDims()) ||
        basis[outDimIdx] <= 0)
      return std::nullopt;
    for (auto [coordIdx, coord] : llvm::enumerate(basis)) {
      if (coordIdx == outDimIdx)
        continue;
      if (coord != 0)
        return std::nullopt;
    }
    bases.push_back(basis[outDimIdx]);
  }

  auto sorted = bases;
  llvm::sort(sorted);
  SmallVector<int32_t> expected;
  for (int64_t bit = 1; bit < extent; bit <<= 1)
    expected.push_back(static_cast<int32_t>(bit));
  if (!llvm::equal(sorted, expected))
    return std::nullopt;
  return bases;
}

static bool hasPurePowerOfTwoBasisSet(const LinearLayout &layout,
                                      StringAttr inDim, unsigned outDimIdx,
                                      int64_t extent,
                                      bool requireAscendingOrder = false) {
  auto order = getPurePowerOfTwoBasisOrder(layout, inDim, outDimIdx, extent);
  if (!order)
    return false;
  if (!requireAscendingOrder)
    return true;
  SmallVector<int32_t> expected;
  for (int64_t bit = 1; bit < extent; bit <<= 1)
    expected.push_back(static_cast<int32_t>(bit));
  return llvm::equal(*order, expected);
}

static bool hasPackedNonRowBasisSet(const LinearLayout &layout,
                                    StringAttr inDim, unsigned rowOutIdx) {
  if (!layout.hasInDim(inDim))
    return false;

  DenseMap<unsigned, SmallVector<int32_t>> basesByOutDim;
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(inDim); ++idx) {
    auto basis = layout.getBasis(inDim, idx);
    if (basis.size() != static_cast<size_t>(layout.getNumOutDims()) ||
        basis[rowOutIdx] != 0)
      return false;

    std::optional<unsigned> activeOutIdx;
    int32_t activeValue = 0;
    for (auto [coordIdx, coord] : llvm::enumerate(basis)) {
      if (coordIdx == rowOutIdx)
        continue;
      if (coord == 0)
        continue;
      if (activeOutIdx || coord <= 0 || !llvm::isPowerOf2_32(coord))
        return false;
      activeOutIdx = coordIdx;
      activeValue = coord;
    }
    if (!activeOutIdx)
      return false;
    basesByOutDim[*activeOutIdx].push_back(activeValue);
  }

  auto outDims = llvm::to_vector(layout.getOutDimNames());
  for (auto [outIdx, outDim] : llvm::enumerate(outDims)) {
    if (outIdx == rowOutIdx)
      continue;
    SmallVector<int32_t> expected;
    for (int64_t bit = 1; bit < layout.getOutDimSize(outDim); bit <<= 1)
      expected.push_back(static_cast<int32_t>(bit));
    auto actual = basesByOutDim.lookup(outIdx);
    llvm::sort(actual);
    if (!llvm::equal(actual, expected))
      return false;
  }
  return true;
}

LinearLayout completeTensorMemorySubviewRowBasesForAnalysis(
    ArrayRef<int64_t> shape, LinearLayout layout) {
  // A row-preserving column subview can keep the parent logical row extent in
  // its output shape while the printed TMEM-linear row bases only cover the
  // directly-addressable 128-row half. Complete those pure row bases before
  // doing exact layout arithmetic so register-layout selection sees the full
  // logical row space.
  if (shape.size() != 2 || layout.getNumOutDims() != 2 ||
      layout.getNumInDims() == 0) {
    return layout;
  }

  auto *ctx = (*layout.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol))
    return layout;

  auto outDims = llvm::to_vector(layout.getOutDimNames());
  StringAttr rowDim = outDims.front();
  StringAttr colDim = outDims.back();
  unsigned rowDimIdx = layout.getOutDimIndex(rowDim);
  unsigned colDimIdx = layout.getOutDimIndex(colDim);
  int64_t logicalRows = shape[0];
  int64_t logicalCols = shape[1];
  int64_t rowBasisSpan = layout.getInDimSize(kRow);
  int64_t colBasisSpan = layout.getInDimSize(kCol);
  if (logicalRows < 1 || logicalCols < 1 || rowBasisSpan >= logicalRows ||
      layout.getOutDimSize(rowDim) != logicalRows ||
      layout.getOutDimSize(colDim) < logicalCols ||
      colBasisSpan < logicalCols || !llvm::isPowerOf2_64(logicalRows) ||
      !llvm::isPowerOf2_64(rowBasisSpan) ||
      !llvm::isPowerOf2_64(logicalCols)) {
    return layout;
  }

  if (!getPurePowerOfTwoBasisOrder(layout, kRow, rowDimIdx, rowBasisSpan) ||
      !getPurePowerOfTwoBasisOrder(layout, kCol, colDimIdx, logicalCols)) {
    return layout;
  }

  auto bases = layout.getBases();
  auto &rowBases = bases[kRow];
  for (int64_t row = rowBasisSpan; row < logicalRows; row <<= 1) {
    std::vector<int32_t> basis(layout.getNumOutDims(), 0);
    basis[rowDimIdx] = static_cast<int32_t>(row);
    rowBases.push_back(std::move(basis));
  }
  return LinearLayout(std::move(bases), layout.getOutDims(),
                      layout.isSurjective());
}

static std::optional<TMemAllocation>
getExpandedSeparableLinearTMemAllocSizes(const LinearLayout &layout,
                                         unsigned preferredColStride,
                                         bool *coversExtraRank = nullptr) {
  if (coversExtraRank)
    *coversExtraRank = false;
  if (preferredColStride != 1 || layout.getNumOutDims() < 2)
    return std::nullopt;

  auto *ctx = (*layout.getOutDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto dims = standardOutDimNames(ctx, layout.getNumOutDims());
  StringAttr rowOutDim = dims[dims.size() - 2];
  StringAttr colOutDim = dims.back();
  if (!layout.hasOutDim(rowOutDim) || !layout.hasOutDim(colOutDim))
    return std::nullopt;
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol))
    return std::nullopt;

  int64_t logicalRows = layout.getInDimSize(kRow);
  if (logicalRows <= 128 || logicalRows % 128 != 0 ||
      !llvm::isPowerOf2_64(logicalRows))
    return std::nullopt;

  int64_t physicalColumnSelectors = layout.getInDimSize(kCol);
  if (physicalColumnSelectors < 1 ||
      physicalColumnSelectors % preferredColStride != 0 ||
      !llvm::isPowerOf2_64(physicalColumnSelectors))
    return std::nullopt;

  if (layout.getOutDimSize(rowOutDim) != logicalRows)
    return std::nullopt;

  unsigned rowOutIdx = layout.getOutDimIndex(rowOutDim);
  if (!hasPurePowerOfTwoBasisSet(layout, kRow, rowOutIdx, logicalRows) ||
      !hasPackedNonRowBasisSet(layout, kCol, rowOutIdx))
    return std::nullopt;

  if (coversExtraRank && layout.getNumOutDims() > 2)
    *coversExtraRank = true;

  // TMEM has 128 physical rows. A separable linear layout with more logical
  // row bits is still a compact physical image: selectors above row 127 choose
  // another column tile, independent of the order of the row basis bits.
  return TMemAllocation{/*numRows=*/128,
                        /*numCols=*/static_cast<int>(
                            (physicalColumnSelectors / preferredColStride) *
                            (logicalRows / 128))};
}

TMemAllocation getTmemAllocSizes(MemDescType memDescType) {
  auto *ctx = memDescType.getContext();
  auto S = [&](StringRef str) { return StringAttr::get(ctx, str); };
  auto kRow = S("row");
  auto kCol = S("col");
  auto leafShape = memDescType.getShape().take_back(2);
  auto encoding = memDescType.getEncoding();
  bool usesScaleFactorEncoding = isa<TensorMemoryScalesEncodingAttr>(encoding);
  auto ll = [&]() {
    if (!usesScaleFactorEncoding) {
      std::string error;
      if (auto canonical =
              tryGetCanonicalTensorMemoryLinearLayout(memDescType, &error))
        return *canonical;
    }
    return triton::gpu::toLinearLayout(leafShape, encoding);
  }();
  auto extraRank = memDescType.getRank() - 2;
  auto bitwidth = memDescType.getElementTypeBitWidth();
  unsigned preferredColStride = 32 / bitwidth;
  int nRow = ll.getInDimSize(kRow);
  int nCol = ll.getInDimSize(kCol) / preferredColStride;
  bool allocationCoversExtraRank = false;
  if (!usesScaleFactorEncoding) {
    if (auto compactAlloc = getExpandedSeparableLinearTMemAllocSizes(
            ll, preferredColStride, &allocationCoversExtraRank)) {
      nRow = compactAlloc->numRows;
      nCol = compactAlloc->numCols;
    }
  }
  // Some exact linear layouts are logically taller than the 128-row TMEM
  // allocation image but are still an MMAv5 family tile with the high row
  // selector carried in columns. Allocate the proven physical family image.
  if (!usesScaleFactorEncoding && nRow > 128) {
    if (auto lhsInfo = getMMAv5LhsLayoutInfo(memDescType)) {
      int familyRows = lhsInfo->familyLayout.getInDimSize(kRow);
      if (familyRows <= 128) {
        ll = lhsInfo->familyLayout;
        nRow = familyRows;
        nCol = ll.getInDimSize(kCol) / preferredColStride;
      }
    }
  }
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
  if (extraRank > 0 && !allocationCoversExtraRank) {
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

uint32_t getTMemSubSliceElementOffset(MemDescType memDescType,
                                      int32_t nOffset) {
  SmallVector<int32_t> offsets(memDescType.getRank(), 0);
  offsets.back() = nOffset;
  return getTMemViewElementOffset(memDescType, offsets);
}

static std::optional<SmallVector<std::pair<StringAttr, uint32_t>>>
solveLinearLayoutPointPreimage(
    const LinearLayout &ll,
    ArrayRef<std::pair<StringAttr, int32_t>> logicalOffsets) {
  int numRows = ll.getTotalOutDimSizeLog2();
  int numCols = ll.getTotalInDimSizeLog2();
  // Unlike LinearLayout::pseudoinvert(), this solves only one requested logical
  // offset. Non-surjective layouts can still have a valid preimage for that
  // point, which is enough to update the current TMEM descriptor address.
  if (numCols >= 64)
    return std::nullopt;

  auto lookupOffset = [&](StringAttr dim) -> std::optional<int32_t> {
    for (auto [offsetDim, offset] : logicalOffsets) {
      if (offsetDim == dim)
        return offset;
    }
    return 0;
  };

  auto matrix = getMatrix(ll);
  std::unique_ptr<uint64_t[]> augmented(new uint64_t[numRows]());
  int row = 0;
  for (StringAttr outDim : ll.getOutDimNames()) {
    auto maybeOffset = lookupOffset(outDim);
    if (!maybeOffset || *maybeOffset < 0 ||
        *maybeOffset >= ll.getOutDimSize(outDim))
      return std::nullopt;
    auto offset = static_cast<uint32_t>(*maybeOffset);
    for (int bit = 0; bit < ll.getOutDimSizeLog2(outDim); ++bit, ++row) {
      augmented[row] = matrix[row];
      if ((offset >> bit) & 1u)
        augmented[row] |= 1ull << numCols;
    }
  }

  f2reduce::inplace_rref_strided(augmented.get(), numRows, numCols + 1,
                                 /*stride=*/1);

  uint64_t coeffMask = (1ull << numCols) - 1;
  uint64_t rhsMask = 1ull << numCols;
  uint64_t solution = 0;
  for (int r = 0; r < numRows; ++r) {
    uint64_t coeffs = augmented[r] & coeffMask;
    bool rhs = (augmented[r] & rhsMask) != 0;
    if (coeffs == 0) {
      if (rhs)
        return std::nullopt;
      continue;
    }
    int pivot = __builtin_ctzll(coeffs);
    if (rhs)
      solution |= 1ull << pivot;
  }

  SmallVector<std::pair<StringAttr, uint32_t>> physicalOffsets;
  int col = 0;
  for (StringAttr inDim : ll.getInDimNames()) {
    uint32_t value = 0;
    for (int bit = 0; bit < ll.getInDimSizeLog2(inDim); ++bit, ++col) {
      if ((solution >> col) & 1ull)
        value |= 1u << bit;
    }
    physicalOffsets.push_back({inDim, value});
  }
  return physicalOffsets;
}

static std::optional<std::pair<uint32_t, uint32_t>>
tryGetTMemViewPhysicalRowElementColImpl(const LinearLayout &ll,
                                        unsigned memRank,
                                        ArrayRef<int32_t> offsets,
                                        ArrayRef<int64_t> prefixShape) {
  if (offsets.size() != memRank)
    return std::nullopt;
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

  uint32_t offsetRow = 0;
  uint32_t offsetCol = 0;
  if (llvm::any_of(logicalOffsets,
                   [](const std::pair<StringAttr, int32_t> &offset) {
                     return offset.second != 0;
                   })) {
    auto rowColBlock = solveLinearLayoutPointPreimage(ll, logicalOffsets);
    if (!rowColBlock)
      return std::nullopt;
    for (auto [dim, value] : *rowColBlock) {
      if (dim == kRow) {
        offsetRow = value;
      } else if (dim == kCol) {
        offsetCol = value;
      }
    }
  }
  if (extraRank > 0) {
    assert(prefixShape.size() == extraRank &&
           "prefix shape is required when the logical rank exceeds the layout rank");
    auto singleBufferCols = ll.getInDimSize(kCol);
    offsetCol += linearizePrefixOffsets(prefixShape, offsets.take_front(extraRank)) *
                 singleBufferCols;
  }
  return std::make_pair(offsetRow, offsetCol);
}

static std::pair<uint32_t, uint32_t>
getTMemViewPhysicalRowElementColImpl(const LinearLayout &ll, unsigned memRank,
                                     ArrayRef<int32_t> offsets,
                                     ArrayRef<int64_t> prefixShape) {
  auto result =
      tryGetTMemViewPhysicalRowElementColImpl(ll, memRank, offsets, prefixShape);
  if (!result)
    llvm_unreachable(
        "TMEM view offset is not representable by descriptor layout");
  return *result;
}

static uint32_t getTMemViewOffsetImpl(const LinearLayout &ll, unsigned memRank,
                                      ArrayRef<int32_t> offsets,
                                      uint32_t bitwidth,
                                      ArrayRef<int64_t> prefixShape) {
  auto [offsetRow, elementCol] =
      getTMemViewPhysicalRowElementColImpl(ll, memRank, offsets, prefixShape);
  uint32_t offsetCol = getTMemWordColumn(elementCol, bitwidth);
  return packTMemRowColOffset(offsetRow, offsetCol);
}

uint32_t getTMemViewOffset(const LinearLayout &layout,
                           ArrayRef<int32_t> offsets, uint32_t bitwidth,
                           ArrayRef<int64_t> prefixShape) {
  return getTMemViewOffsetImpl(layout, layout.getNumOutDims(), offsets,
                               bitwidth, prefixShape);
}

static LinearLayout getTMemViewOffsetAnalysisLayout(MemDescType memDescType) {
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
  return ll;
}

std::pair<uint32_t, uint32_t>
getTMemViewPhysicalRowElementCol(MemDescType memDescType,
                                 ArrayRef<int32_t> offsets) {
  auto result = tryGetTMemViewPhysicalRowElementCol(memDescType, offsets);
  if (!result)
    llvm_unreachable(
        "TMEM view offset is not representable by descriptor layout");
  return *result;
}

std::optional<std::pair<uint32_t, uint32_t>>
tryGetTMemViewPhysicalRowElementCol(MemDescType memDescType,
                                    ArrayRef<int32_t> offsets) {
  if (!memDescType ||
      offsets.size() != static_cast<size_t>(memDescType.getRank()))
    return std::nullopt;
  LinearLayout ll = getTMemViewOffsetAnalysisLayout(memDescType);
  return tryGetTMemViewPhysicalRowElementColImpl(
      ll, memDescType.getRank(), offsets,
      memDescType.getShape().take_front(
          memDescType.getRank() > ll.getNumOutDims()
              ? memDescType.getRank() - ll.getNumOutDims()
              : 0));
}

uint32_t getTMemViewElementOffset(MemDescType memDescType,
                                  ArrayRef<int32_t> offsets) {
  auto [offsetRow, elementCol] =
      getTMemViewPhysicalRowElementCol(memDescType, offsets);
  return packTMemRowColOffset(offsetRow, elementCol);
}

uint32_t getTMemViewOffset(MemDescType memDescType, ArrayRef<int32_t> offsets) {
  LinearLayout ll = getTMemViewOffsetAnalysisLayout(memDescType);
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
  auto innerOutDims = llvm::to_vector(inner.getOutDimNames());
  auto outerInDims = llvm::to_vector(outer.getInDimNames());
  if (innerOutDims.size() != outerInDims.size() ||
      llvm::any_of(innerOutDims, [&](StringAttr dim) {
        return !llvm::is_contained(outerInDims, dim);
      }))
    return false;
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
    // Keep the canonical M64 family on x256 so the direct ld/st matcher reaches
    // the native 16x256b path instead of degrading to x128.
    laneBases = {{0, 2}, {0, 4}, {1, 0}, {2, 0}, {4, 0}};
    regBases = {{0, 1}, {8, 0}};
    for (int64_t col = 8; col <= (numWarps == 4 ? n / 2 : n / 4); col <<= 1)
      regBases.push_back({0, static_cast<int32_t>(col)});
    warpBases = {{16, 0}, {32, 0}};
    if (numWarps == 8) {
      // The canonical 8-warp x256 family only shards N across the extra warp
      // bit once there is at least one full 8-column packet to place there. For
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
    if (dims.size() != 2)
      return std::nullopt;
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
        // path below to preserve the canonical packed row/warp layout.
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
getDistributedLayoutForTmemLdStAnchored(const LinearLayout &ll,
                                        TMemAccessAtom atom,
                                        unsigned numWarps, int bitwidth) {
  auto dims = to_vector(ll.getOutDimNames());
  if (dims.size() != 2)
    return std::nullopt;
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
      if (auto perCTA = getDistributedLayoutForTmemLdStAnchored(
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
      auto ret = getDistributedLayoutForTmemLdStAnchored(
          *maybeQuot, atom, numWarps, 32);
      if (!ret)
        return ret;
      auto castbbitwidth =
          LinearLayout::zeros1D(1, kReg, dims[1], 32 / bitwidth) *
          LinearLayout::identity1D(2, kReg, dims[1]);
      return castbbitwidth * ret.value();
    }
    if (bestContig > 1) {
      auto ret = getDistributedLayoutForTmemLdStAnchored(
          quot, atom, numWarps, bitwidth * bestContig);
      if (!ret)
        return ret;
      auto castbbitwidth = LinearLayout::identity1D(bestContig, kReg, dims[1]);
      return castbbitwidth * ret.value();
    }
    if (auto maybeQuot =
                   divideLeft(ll, LinearLayout::zeros1D(
                                      32 / bitwidth, rowColDims[1], dims[1]))) {
      return getDistributedLayoutForTmemLdStAnchored(*maybeQuot, atom,
                                                           numWarps, 32);
    } else if (ll.getInDimSize(rowColDims[1]) == 1) {
      return getDistributedLayoutForTmemLdStAnchored(ll, atom, numWarps,
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

static bool isFullShapeOrRowPreservingColumnSubview(gpu::MemDescType memType);

static std::optional<LinearLayout>
getExpandedRowDirectI32x32bLayout(gpu::MemDescType memType,
                                  const LinearLayout &memLayout,
                                  unsigned numWarps);

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps,
                                std::optional<TMemLdStRowPlan> rowPlanOverride,
                                std::optional<LinearLayout> queryLayoutOverride) {
  assert(memType.getMemorySpace() ==
         TensorMemorySpaceAttr::get(memType.getContext()));
  if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
    return std::nullopt;
  auto restoreQueryCTAOwnership = [&](LinearLayout layout) {
    auto *ctx = memType.getContext();
    auto kBlock = StringAttr::get(ctx, "block");
    auto kRow = StringAttr::get(ctx, "row");
    auto kCol = StringAttr::get(ctx, "col");
    if (!layout.hasInDim(kBlock))
      return layout;
    auto bases = layout.getBases();
    auto blockIt = bases.find(kBlock);
    if (blockIt == bases.end() || !blockIt->second.empty())
      return layout;
    for (StringAttr dim : {kRow, kCol}) {
      auto dimIt = bases.find(dim);
      if (dimIt == bases.end() || dimIt->second.empty())
        continue;
      ArrayRef<int32_t> candidate = dimIt->second.back();
      int nonZeroDim = -1;
      bool valid = true;
      for (auto [idx, value] : llvm::enumerate(candidate)) {
        if (value == 0)
          continue;
        if (value < 0 || nonZeroDim >= 0) {
          valid = false;
          break;
        }
        nonZeroDim = static_cast<int>(idx);
      }
      if (!valid || nonZeroDim < 0)
        continue;
      auto outDims = llvm::to_vector(layout.getOutDimNames());
      if (static_cast<size_t>(nonZeroDim) >= outDims.size())
        continue;
      int32_t dimSize = layout.getOutDimSize(outDims[nonZeroDim]);
      if (candidate[nonZeroDim] * 2 != dimSize)
        continue;
      blockIt->second.push_back(candidate.vec());
      dimIt->second.pop_back();
      return LinearLayout(std::move(bases), layout.getOutDims(),
                          layout.isSurjective());
    }
    return layout;
  };
  auto hasExpectedCTAOwnership = [&](Attribute attr) {
    unsigned expectedCTAs = getNumCTAs(memType.getEncoding());
    return product<unsigned>(getCTAsPerCGA(attr)) == expectedCTAs;
  };
  auto isValidLayout = [&](const LinearLayout &layout) {
    auto attr = tryGetLinearEncodingAttr(memType.getContext(), layout);
    if (!attr || !hasExpectedCTAOwnership(*attr))
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
    if (!attr || !hasExpectedCTAOwnership(*attr))
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
    bool twoCTAs = getTensorMemoryTwoCTAs(memType.getEncoding()).value_or(false);
    if (!rowPlanOverride && memType.getShape() == memType.getAllocShape()) {
      auto raw = foldCanonicalSingleCTABlockRowsForAnalysis(
          toLinearLayout(memType.getShape(), memType.getEncoding()), twoCTAs);
      std::string rawError;
      if (auto maybeLayout = getTMemViewAnalysisLinearLayout(
              memType.getShape(), memType.getEncoding(), &rawError)) {
        auto normalized = foldCanonicalSingleCTABlockRowsForAnalysis(
            completeTensorMemorySubviewRowBasesForAnalysis(
                memType.getShape(),
                normalizeTensorMemoryLinearLayoutForAnalysis(*maybeLayout)),
            twoCTAs);
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
    return foldCanonicalSingleCTABlockRowsForAnalysis(
        completeTensorMemorySubviewRowBasesForAnalysis(
            memType.getShape(),
            normalizeTensorMemoryLinearLayoutForAnalysis(*maybe)),
        twoCTAs);
  }();
  if (ll.getNumOutDims() == 0)
    return std::nullopt;
  if (queryLayoutOverride)
    ll = restoreQueryCTAOwnership(*queryLayoutOverride);
  auto bitwidth = memType.getElementTypeBitWidth();
  if (queryLayoutOverride) {
    if (auto layout = getTwoCTAScalesDescriptorViewTMemLdStLayout(
            memType, atom, numWarps, *queryLayoutOverride);
        layout && isValidLayoutForQuery(*layout, *queryLayoutOverride,
                                        rowPlanOverride)) {
      return layout;
    }
  }
  auto stripped = stripZeroBasesForTmemLdStSelection(ll);
  if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      !rowPlanOverride) {
    if (auto layout = getDistributedLayoutForTmemLdStAnchored(
            ll, atom, numWarps, bitwidth);
        layout && isTMemLdStSelectionLayoutValid(memType, *layout)) {
      return layout;
    }

    auto stripped = stripZeroBasesForTmemLdStSelection(ll);
    if (stripped != ll) {
      auto layout = getDistributedLayoutForTmemLdStAnchored(
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
      if (auto canonicalLike = getDistributedLayoutForTmemLdStAnchored(
              ll, atom, numWarps, bitwidth)) {
        if (isValidLayout(*canonicalLike))
          return canonicalLike;
      }
    }
  }
  if (!rowPlanOverride && atom == TMemAccessAtom::I32x32b &&
      !isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) &&
      isFullShapeOrRowPreservingColumnSubview(memType)) {
    if (auto expanded =
            getExpandedRowDirectI32x32bLayout(memType, ll, numWarps);
        expanded && isValidLayout(*expanded)) {
      return expanded;
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
      // Sparse M64 leaves encode the unused half-tile as a zero row basis.
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
  auto candidateEncoding =
      tryGetLinearEncodingAttr(tensorType.getContext(), layout);
  if (!candidateEncoding)
    return false;
  auto candidateType = tensorType.cloneWithEncoding(*candidateEncoding);
  auto maxnreg = getContextualMaxNReg(op);
  return succeeded(
      computeTMemLdStEncodingInfo(candidateType, memType, maxnreg));
}

static bool appendTMemCompatibleEncodingCandidate(
    SmallVectorImpl<DistributedEncodingTrait> &layouts,
    RankedTensorType tensorType, gpu::MemDescType memType,
    DistributedEncodingTrait candidateEncoding, unsigned maxnreg,
    bool requireUnique = false) {
  if (requireUnique && llvm::is_contained(layouts, candidateEncoding))
    return false;
  auto candidateType = tensorType.cloneWithEncoding(candidateEncoding);
  if (failed(computeTMemLdStEncodingInfo(candidateType, memType, maxnreg))) {
    return false;
  }
  layouts.push_back(candidateEncoding);
  return true;
}

static bool appendTMemCompatibleEncodingCandidateForQuery(
    SmallVectorImpl<DistributedEncodingTrait> &layouts,
    RankedTensorType tensorType, gpu::MemDescType memType,
    DistributedEncodingTrait candidateEncoding, const LinearLayout &queryLayout,
    unsigned maxnreg, std::optional<TMemLdStRowPlan> rowPlan,
    bool requireUnique = false) {
  if (requireUnique && llvm::is_contained(layouts, candidateEncoding))
    return false;
  auto candidateType = tensorType.cloneWithEncoding(candidateEncoding);
  if (failed(computeTMemLdStEncodingInfo(candidateType, memType, queryLayout,
                                         maxnreg, /*emitError=*/{}, rowPlan)))
    return false;
  layouts.push_back(candidateEncoding);
  return true;
}

static bool
appendTMemCompatibleCandidate(SmallVectorImpl<DistributedEncodingTrait> &layouts,
                              RankedTensorType tensorType,
                              gpu::MemDescType memType,
                              const LinearLayout &layout, unsigned maxnreg,
                              bool requireUnique = false) {
  auto candidateEncoding =
      tryGetLinearEncodingAttr(tensorType.getContext(), layout);
  if (!candidateEncoding)
    return false;
  return appendTMemCompatibleEncodingCandidate(
      layouts, tensorType, memType, *candidateEncoding, maxnreg,
      requireUnique);
}

static bool
appendTMemCompatibleCandidate(SmallVectorImpl<DistributedEncodingTrait> &layouts,
                              RankedTensorType tensorType,
                              gpu::MemDescType memType,
                              DistributedEncodingTrait candidateEncoding,
                              unsigned maxnreg,
                              bool requireUnique = false) {
  return appendTMemCompatibleEncodingCandidate(
      layouts, tensorType, memType, candidateEncoding, maxnreg, requireUnique);
}

static bool appendTMemCompatibleCandidateForQuery(
    SmallVectorImpl<DistributedEncodingTrait> &layouts,
    RankedTensorType tensorType, gpu::MemDescType memType,
    const LinearLayout &layout, const LinearLayout &queryLayout,
    unsigned maxnreg, std::optional<TMemLdStRowPlan> rowPlan,
    bool requireUnique = false) {
  auto candidateEncoding =
      tryGetLinearEncodingAttr(tensorType.getContext(), layout);
  if (!candidateEncoding)
    return false;
  return appendTMemCompatibleEncodingCandidateForQuery(
      layouts, tensorType, memType, *candidateEncoding, queryLayout, maxnreg,
      rowPlan, requireUnique);
}

static bool
appendTMemCompatibleCandidate(SmallVectorImpl<DistributedEncodingTrait> &layouts,
                              Operation *op, RankedTensorType tensorType,
                              gpu::MemDescType memType,
                              const LinearLayout &layout,
                              bool requireUnique = false) {
  return appendTMemCompatibleCandidate(layouts, tensorType, memType, layout,
                                       getContextualMaxNReg(op),
                                       requireUnique);
}

static bool
appendTMemCompatibleCandidate(SmallVectorImpl<DistributedEncodingTrait> &layouts,
                              Operation *op, RankedTensorType tensorType,
                              gpu::MemDescType memType,
                              DistributedEncodingTrait candidateEncoding,
                              bool requireUnique = false) {
  return appendTMemCompatibleCandidate(layouts, tensorType, memType,
                                       candidateEncoding,
                                       getContextualMaxNReg(op),
                                       requireUnique);
}

static std::optional<LinearLayout>
getTMemScalesNarrowTileLayout(MemDescType memType, unsigned numWarps) {
  if (!isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) ||
      numWarps != 4 || memType.getElementTypeBitWidth() != 8)
    return std::nullopt;
  auto shape = memType.getShape();
  if (shape.size() < 2 || shape[shape.size() - 2] != 16 ||
      shape[shape.size() - 1] != 8)
    return std::nullopt;

  auto *ctx = memType.getContext();
  auto dims = standardOutDimNames(ctx, 2);
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");

  LinearLayout::BasesT bases;
  bases[kReg] = {{0, 1}, {0, 2}, {0, 0}};
  bases[kLane] = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {0, 4}};
  bases[kWarp] = {{0, 0}, {0, 0}};
  return LinearLayout(
      std::move(bases),
      {{dims[0], static_cast<int32_t>(shape[shape.size() - 2])},
       {dims[1], static_cast<int32_t>(shape[shape.size() - 1])}},
      /*requireSurjective=*/false);
}

static bool appendTMemScalesNarrowTileCompatibleLayout(
    SmallVectorImpl<DistributedEncodingTrait> &layouts,
    RankedTensorType tensorType, MemDescType memType, unsigned numWarps,
    unsigned maxnreg, bool requireUnique = false) {
  auto narrow = getTMemScalesNarrowTileLayout(memType, numWarps);
  if (!narrow)
    return false;
  return appendTMemCompatibleCandidate(layouts, tensorType, memType, *narrow,
                                       maxnreg, requireUnique);
}

static std::optional<LinearLayout>
getExpandedRowDirectI32x32bLayout(MemDescType memType,
                                  const LinearLayout &memLayout,
                                  unsigned numWarps) {
  if (numWarps < 8 || memType.getElementTypeBitWidth() != 32 ||
      memType.getRank() != 2 ||
      !isTensorMemoryEncoding(memType.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) ||
      !isFullShapeOrRowPreservingColumnSubview(memType) ||
      memLayout.getNumOutDims() != 2) {
    return std::nullopt;
  }

  auto *ctx = memType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kReg = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto dims = standardOutDimNames(ctx, 2);
  auto outDims = llvm::to_vector(memLayout.getOutDimNames());
  if (!llvm::is_contained(outDims, dims[0]) ||
      !llvm::is_contained(outDims, dims[1]) || !memLayout.hasInDim(kRow) ||
      !memLayout.hasInDim(kCol)) {
    return std::nullopt;
  }

  int64_t logicalRows = memType.getShape()[0];
  int64_t logicalCols = memType.getShape()[1];
  if (logicalRows != 256 || logicalCols < 1 ||
      !llvm::isPowerOf2_64(logicalCols)) {
    return std::nullopt;
  }

  LinearLayout queryLayout =
      completeTensorMemorySubviewRowBasesForAnalysis(memType.getShape(),
                                                     memLayout);
  unsigned rowDim = queryLayout.getOutDimIndex(dims[0]);
  unsigned colDim = queryLayout.getOutDimIndex(dims[1]);
  if (queryLayout.getInDimSize(kRow) != logicalRows ||
      queryLayout.getInDimSize(kCol) < logicalCols ||
      !getPurePowerOfTwoBasisOrder(queryLayout, kRow, rowDim, logicalRows) ||
      !getPurePowerOfTwoBasisOrder(queryLayout, kCol, colDim, logicalCols)) {
    return std::nullopt;
  }

  LinearLayout::BasesT bases;
  for (int64_t col = 1; col < logicalCols; col <<= 1)
    bases[kReg].push_back({0, static_cast<int32_t>(col)});
  bases[kLane] = {{1, 0}, {2, 0}, {4, 0}, {8, 0}, {16, 0}};
  bases[kWarp] = {{32, 0}, {64, 0}, {128, 0}};
  LinearLayout physicalTile(
      std::move(bases),
      {{kRow, static_cast<int32_t>(logicalRows)},
       {kCol, static_cast<int32_t>(logicalCols)}},
      /*requireSurjective=*/false);
  if (!canComposeLinearLayouts(physicalTile, queryLayout)) {
    return std::nullopt;
  }
  auto ret = physicalTile.compose(queryLayout);
  SmallVector<StringAttr> canonicalInDims = {kReg, kLane, kWarp};
  ret = ret.transposeIns(canonicalInDims);
  auto withoutBroadcast = ret;
  for (auto inDim : ret.getInDimNames())
    withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
  if (!withoutBroadcast.isInvertible()) {
    return std::nullopt;
  }
  return ret;
}

static bool isFullShapeOrRowPreservingColumnSubview(gpu::MemDescType memType) {
  if (memType.getShape() == memType.getAllocShape())
    return true;
  if (memType.getRank() != 2 || memType.getAllocShape().size() != 2)
    return false;
  return memType.getShape()[0] == memType.getAllocShape()[0] &&
         memType.getShape()[1] <= memType.getAllocShape()[1];
}

static std::optional<LinearLayout>
getRowPreservingColumnSubviewPhysicalQueryLayout(gpu::MemDescType memType) {
  // 32x32b packets cannot carry row >= 128 directly in the packet schedule.
  // For row-preserving 256-row column subviews, validate candidates against
  // the folded physical query used by actual `ttng.tmem_subslice` lowering:
  // low row bases stay in `row`, and the high row selector is represented as
  // an extra physical `col` basis.
  if (memType.getRank() != 2 || memType.getAllocShape().size() != 2 ||
      memType.getShape() == memType.getAllocShape() ||
      memType.getShape()[0] != memType.getAllocShape()[0] ||
      memType.getShape()[1] > memType.getAllocShape()[1] ||
      !isTensorMemoryEncoding(memType.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding())) {
    return std::nullopt;
  }

  std::string error;
  auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
      memType.getShape(), memType.getEncoding(), &error);
  if (!maybeAnalysis)
    return std::nullopt;
  auto layout = completeTensorMemorySubviewRowBasesForAnalysis(
      memType.getShape(),
      normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis));
  auto *ctx = memType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto dims = standardOutDimNames(ctx, 2);
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol))
    return std::nullopt;
  if (layout.getNumOutDims() != 2 || !llvm::equal(layout.getOutDimNames(), dims))
    return std::nullopt;

  int64_t logicalRows = memType.getShape()[0];
  int64_t logicalCols = memType.getShape()[1];
  if (logicalRows != 256 || logicalCols < 1 ||
      !llvm::isPowerOf2_64(logicalCols))
    return std::nullopt;

  unsigned rowOutIdx = layout.getOutDimIndex(dims[0]);
  unsigned colOutIdx = layout.getOutDimIndex(dims[1]);
  if (!getPurePowerOfTwoBasisOrder(layout, kRow, rowOutIdx, logicalRows) ||
      !getPurePowerOfTwoBasisOrder(layout, kCol, colOutIdx, logicalCols)) {
    return std::nullopt;
  }

  LinearLayout::BasesT bases;
  bases[kRow] = {};
  bases[kCol] = {};
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(kRow); ++idx) {
    ArrayRef<int32_t> basis = layout.getBasis(kRow, idx);
    if (basis[rowOutIdx] < 128)
      bases[kRow].push_back(std::vector<int32_t>(basis.begin(), basis.end()));
  }
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(kCol); ++idx) {
    ArrayRef<int32_t> basis = layout.getBasis(kCol, idx);
    bases[kCol].push_back(std::vector<int32_t>(basis.begin(), basis.end()));
  }
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(kRow); ++idx) {
    ArrayRef<int32_t> basis = layout.getBasis(kRow, idx);
    if (basis[rowOutIdx] >= 128)
      bases[kCol].push_back(std::vector<int32_t>(basis.begin(), basis.end()));
  }
  if (bases[kRow].size() != 7 ||
      bases[kCol].size() != layout.getInDimSizeLog2(kCol) + 1) {
    return std::nullopt;
  }

  std::string layoutError;
  return LinearLayout::tryCreate(std::move(bases),
                                 llvm::to_vector(layout.getOutDims()),
                                 /*requireSurjective=*/true, &layoutError);
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
      if (auto preferred = getDistributedLayoutForTmemLdStAnchored(
              raw, atom, numWarps, memType.getElementTypeBitWidth());
          preferred && isTMemLdStSelectionLayoutValid(memType, *preferred)) {
        return LinearEncodingAttr::get(ctx, std::move(*preferred));
      }
    }
  }
  if (prefer16x256) {
    auto tryCanonicalPreferred =
        [&](const LinearLayout &layout)
        -> std::optional<DistributedEncodingTrait> {
      if (auto preferred = getDistributedLayoutForTmemLdStAnchored(
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
      TMemLdStRowPlan canonicalM64Plan{/*warpRow0=*/32, /*warpRow1=*/64,
                                        /*rowSpan=*/128};
      if (auto preferred = getDistributedLayoutForTmemLdSt(
              memType, TMemAccessAtom::I16x256b, numWarps, canonicalM64Plan)) {
        return LinearEncodingAttr::get(ctx, std::move(*preferred));
      }
      return std::nullopt;
    };
    if (!isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding())) {
      auto raw = toLinearLayout(memType.getShape(), memType.getEncoding());
      if (memType.getShape() == memType.getAllocShape()) {
        if (auto preferred = tryCanonicalPreferred(raw))
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

TMemLoadReductionLayoutSupport
getTmemLoadReductionLayoutSupport(RankedTensorType tensorType,
                                  const LinearLayout &layout) {
  auto unsupported = [](TMemLoadReductionUnsupportedReason kind,
                        Twine reason) {
    return TMemLoadReductionLayoutSupport{std::nullopt, reason.str(), kind};
  };

  if (layout.getNumOutDims() != 2)
    return unsupported(
        TMemLoadReductionUnsupportedReason::LayoutRank,
        "Reduction load layout support requires exactly two logical output "
        "dimensions.");
  auto attr = LinearEncodingAttr::get(tensorType.getContext(), layout);
  auto regTy = tensorType.cloneWithEncoding(attr);
  auto kReg = StringAttr::get(tensorType.getContext(), "register");
  auto kLane = StringAttr::get(tensorType.getContext(), "lane");
  auto regLayout = toLinearLayout(regTy);
  auto regDims = toLinearEncoding(regTy).basesPerDim(kReg);
  auto outDims = llvm::to_vector(regLayout.getOutDimSizes());
  if (outDims.size() < 2)
    return unsupported(
        TMemLoadReductionUnsupportedReason::MissingOutputDims,
        "Reduction load layout support requires materialized M and N output "
        "dimensions.");

  constexpr int dimM = 0;
  constexpr int dimN = 1;
  auto dims = llvm::to_vector(regLayout.getOutDimNames());
  auto describeBases = [&](StringAttr inDim, unsigned dim) {
    std::string text;
    llvm::raw_string_ostream os(text);
    bool first = true;
    if (!regLayout.hasInDim(inDim))
      return std::string();
    for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(inDim); ++idx) {
      int32_t basis = regLayout.getBasis(inDim, idx, dims[dim]);
      if (basis == 0)
        continue;
      if (!first)
        os << ", ";
      first = false;
      os << inDim.getValue() << " bit " << idx << " -> " << basis;
    }
    return text;
  };

  if (regDims[dimM] != 1) {
    std::string mBases = describeBases(kReg, dimM);
    return unsupported(
        TMemLoadReductionUnsupportedReason::MShardedAcrossRegisters,
        Twine("Reduction load layout shards the M dimension across register "
              "values") +
        (mBases.empty() ? Twine(".") : Twine(" (") + mBases + ").") +
        " tcgen05.ld.red returns one reduction value per emitted message and "
        "thread, so lowering cannot assign distinct reduced results to "
        "multiple M rows carried by one thread without a separate software "
        "reduction/writeback schedule.");
  }
  if (regDims[dimN] == outDims[dimN])
    return TMemLoadReductionLayoutSupport{0u, ""};

  auto describeNonRegisterNBases = [&]() {
    std::string text;
    llvm::raw_string_ostream os(text);
    bool first = true;
    for (StringAttr inDim : regLayout.getInDimNames()) {
      if (inDim == kReg)
        continue;
      for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(inDim); ++idx) {
        int32_t nBasis = regLayout.getBasis(inDim, idx, dims[dimN]);
        if (nBasis == 0)
          continue;
        if (!first)
          os << ", ";
        first = false;
        int32_t mBasis = regLayout.getBasis(inDim, idx, dims[dimM]);
        os << inDim.getValue() << " bit " << idx << " -> N " << nBasis;
        if (mBasis != 0)
          os << " and M " << mBasis;
      }
    }
    return text;
  };

  if (outDims[dimN] < 2 || outDims[dimN] != regDims[dimN] * 2) {
    std::string nBases = describeNonRegisterNBases();
    return unsupported(
        TMemLoadReductionUnsupportedReason::PartialNInRegisters,
        Twine("Reduction load layout keeps only ") + Twine(regDims[dimN]) +
        " of " + Twine(outDims[dimN]) +
        " N elements in the register dimension. Current lowering supports "
        "full-register N coverage or exactly one lane-local split of N; this "
        "layout would need a broader cross-thread reduction" +
        (nBases.empty() ? Twine(".") : Twine(" (") + nBases + ")."));
  }

  SmallVector<int32_t> nBases;
  for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(kReg); ++idx) {
    if (regLayout.getBasis(kReg, idx, dims[dimM]) != 0)
      return unsupported(
          TMemLoadReductionUnsupportedReason::RegisterBasisTouchesM,
          "Reduction load layout has a register basis that contributes to "
          "both the per-thread value stream and M, so lowering cannot treat "
          "the register dimension as a pure N-reduction dimension.");
    int32_t nBasis = regLayout.getBasis(kReg, idx, dims[dimN]);
    if (nBasis != 0)
      nBases.push_back(nBasis);
  }

  std::optional<unsigned> laneSplitMask;
  for (StringAttr inDim : regLayout.getInDimNames()) {
    if (inDim == kReg)
      continue;
    for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(inDim); ++idx) {
      int32_t nBasis = regLayout.getBasis(inDim, idx, dims[dimN]);
      if (nBasis == 0)
        continue;
      if (inDim != kLane || idx != 4 ||
          regLayout.getBasis(inDim, idx, dims[dimM]) != 0 || laneSplitMask) {
        std::string nBasesText = describeNonRegisterNBases();
        return unsupported(
            TMemLoadReductionUnsupportedReason::UnsupportedNThreadBasis,
            Twine("Reduction load layout splits N through an unsupported "
                  "thread basis. Current lowering can combine only one pure "
                  "lane bit 4 split with shuffle-xor 16") +
            (nBasesText.empty() ? Twine(".")
                                : Twine("; observed ") + nBasesText + "."));
      }
      laneSplitMask = 1u << idx;
      nBases.push_back(nBasis);
    }
  }
  if (!laneSplitMask)
    return unsupported(
        TMemLoadReductionUnsupportedReason::MissingLaneSplit,
        "Reduction load layout does not keep full N in registers and has no "
        "supported lane bit 4 N split to combine after tcgen05.ld.red.");

  llvm::sort(nBases);
  SmallVector<int32_t> expectedNBases;
  for (int64_t n = 1; n < outDims[dimN]; n <<= 1)
    expectedNBases.push_back(static_cast<int32_t>(n));
  if (!llvm::equal(nBases, expectedNBases))
    return unsupported(
        TMemLoadReductionUnsupportedReason::NonContiguousNBases,
        "Reduction load layout register/lane N bases do not cover the full "
        "contiguous power-of-two reduction dimension.");
  return TMemLoadReductionLayoutSupport{*laneSplitMask, ""};
}

std::optional<unsigned>
getTmemLoadReductionLaneSplitMask(RankedTensorType tensorType,
                                  const LinearLayout &layout) {
  return getTmemLoadReductionLayoutSupport(tensorType, layout).laneSplitMask;
}

std::optional<unsigned>
getTmemLoadReductionLaneSplitMask(RankedTensorType tensorType) {
  return getTmemLoadReductionLaneSplitMask(tensorType,
                                           toLinearLayout(tensorType));
}

bool isReductionFriendlyTmemLoadLayout(RankedTensorType tensorType,
                                       const LinearLayout &layout) {
  return static_cast<bool>(
      getTmemLoadReductionLayoutSupport(tensorType, layout));
}

bool isReductionFriendlyTmemSourceLayout(MemDescType memType) {
  std::string error;
  auto maybeCanonical = getCanonicalTMemLinearEncoding(memType, &error);
  if (!maybeCanonical)
    return false;

  auto layout = normalizeTensorMemoryLinearLayoutForAnalysis(
      maybeCanonical->getLinearLayout());
  bool twoCTAs = getTensorMemoryTwoCTAs(memType.getEncoding()).value_or(false);
  layout =
      foldCanonicalSingleCTABlockRowsForAnalysis(std::move(layout), twoCTAs);
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
  // 256-row identity layouts are still directly reducible: the row packet
  // anchors remain materializable, and the extra row selector is represented
  // by the physical TMEM layout rather than by a column carry basis.
  if (blockM != 64 && blockM != 128 && blockM != 256)
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
  // Reduction layout inference follows the direct TMEM load/store planner but
  // validates the resulting message shape against tcgen05.ld.red. Larger warp
  // counts are legal when the resulting register layout keeps N fully in
  // registers and leaves M unsharded, e.g. 256x128 with 8 warps. M64 split-N
  // layouts may lower through the 16x32bx2 family when that is the first
  // reduction-compatible message schedule.
  if (memType.getRank() != 2 ||
      isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()) ||
      memType.getElementTypeBitWidth() != 32) {
    return std::nullopt;
  }
  if (!isReductionFriendlyTmemSourceLayout(memType))
    return std::nullopt;

  auto *ctx = tensorType.getContext();
  auto validateReductionLayout =
      [&](const LinearLayout &layout) -> std::optional<DistributedEncodingTrait> {
    if (!isReductionFriendlyTmemLoadLayout(tensorType, layout))
      return std::nullopt;
    auto attr = LinearEncodingAttr::get(ctx, layout);
    auto regTy = tensorType.cloneWithEncoding(attr);
    auto info = computeTMemLdStEncodingInfo(regTy, memType, /*maxnreg=*/256);
    if (failed(info) || !isTMemLdStReductionCompatible(*info))
      return std::nullopt;
    return attr;
  };

  auto tryReductionLayout =
      [&](TMemAccessAtom atom) -> std::optional<DistributedEncodingTrait> {
    std::optional<LinearLayout> layout =
        getDistributedLayoutForTmemLdSt(memType, atom, numWarps);
    if (!layout)
      return std::nullopt;
    auto ret = std::move(*layout);
    if (auto attr = validateReductionLayout(ret))
      return attr;

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

    return validateReductionLayout(ret);
  };

  // Message legality is not a sufficient proof that a backend-selected
  // reduction layout preserves the exact logical row/column order or packet
  // order. If the direct 32x32b layout is already reduction-compatible, keep
  // the caller's existing direct choice. This lets the helper rescue
  // scalarized M64 direct layouts while avoiding the non-M64 row/column
  // permutation false-support paths discovered by broad default-routing
  // probes.
  if (std::optional<LinearLayout> directLayout =
          getDistributedLayoutForTmemLdSt(memType, TMemAccessAtom::I32x32b,
                                          numWarps)) {
    if (validateReductionLayout(*directLayout))
      return std::nullopt;
  }

  if (auto attr = tryReductionLayout(TMemAccessAtom::I32x32b))
    return attr;
  if (auto attr = tryReductionLayout(TMemAccessAtom::I16x32bx2))
    return attr;
  return std::nullopt;
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
  if (isScales) {
    memLL = toLinearLayout(memType);
  } else {
    std::string error;
    if (auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
            memType.getShape(), memType.getEncoding(), &error)) {
      memLL = completeTensorMemorySubviewRowBasesForAnalysis(
          memType.getShape(),
          normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis));
    }
  }

  auto tensorTy =
      RankedTensorType::get(memType.getShape(), memType.getElementType());
  appendTMemScalesNarrowTileCompatibleLayout(
      layouts, tensorTy, memType, numWarps, /*maxnreg=*/256);
  auto tryPushUniqueLayout = [&](const LinearLayout &layout) {
    appendTMemCompatibleCandidate(layouts, tensorTy, memType, layout,
                                  /*maxnreg=*/256, /*requireUnique=*/true);
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
  if (!isScales && memType.getElementTypeBitWidth() == 32 &&
      memType.getRank() == 2 &&
      isFullShapeOrRowPreservingColumnSubview(memType) &&
      memLL.getNumOutDims() != 0) {
    if (auto expanded =
            getExpandedRowDirectI32x32bLayout(memType, memLL, numWarps)) {
      auto before = layouts.size();
      tryPushUniqueLayout(*expanded);
      if (layouts.size() == before) {
        if (auto queryLayout =
                getRowPreservingColumnSubviewPhysicalQueryLayout(memType)) {
          appendTMemCompatibleCandidateForQuery(
              layouts, tensorTy, memType, *expanded, *queryLayout,
              /*maxnreg=*/256, getTMemLdStRowPlan(*queryLayout),
              /*requireUnique=*/true);
        }
      }
    }
  }

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
      ll = getDistributedLayoutForTmemLdStAnchored(memLL, atom, numWarps,
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
          tryGetLinearEncodingAttr(memType.getContext(), *ll);
      if (debugCompatLayouts && !candidateEncoding)
        llvm::errs() << "[tmem-compat] reject: invalid linear attr\n";
      bool inserted =
          candidateEncoding &&
          appendTMemCompatibleCandidate(layouts, tensorTy, memType,
                                        *candidateEncoding, /*maxnreg=*/256);
      if (debugCompatLayouts)
        llvm::errs() << "[tmem-compat] ldst=" << (inserted ? "ok" : "fail")
                     << "\n";
    }
  }

  if (auto splitLongM = getTmemLoadLayoutSplitLongM(tensorTy, memType,
                                                    numWarps))
    appendTMemCompatibleCandidate(layouts, tensorTy, memType, splitLongM.value(),
                                  /*maxnreg=*/256);
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
  appendTMemScalesNarrowTileCompatibleLayout(
      layouts, tensorType, memType, numWarps, getContextualMaxNReg(op));
  LinearLayout memLL = [&]() -> LinearLayout {
    if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
      return toLinearLayout(memType);
    std::string error;
    auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
        memType.getShape(), memType.getEncoding(), &error);
    if (!maybeAnalysis)
      return LinearLayout();
    return completeTensorMemorySubviewRowBasesForAnalysis(
        memType.getShape(),
        normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis));
  }();
  if (memLL.getNumOutDims() == 0)
    return layouts;
  if (!isScales && memType.getElementTypeBitWidth() == 32 &&
      memType.getRank() == 2 &&
      isFullShapeOrRowPreservingColumnSubview(memType)) {
    if (auto expanded =
            getExpandedRowDirectI32x32bLayout(memType, memLL, numWarps)) {
      bool supported =
          isTMemCompatibleCandidate(op, tensorType, memType, *expanded);
      if (!supported) {
        if (auto queryLayout =
                getRowPreservingColumnSubviewPhysicalQueryLayout(memType)) {
          supported = appendTMemCompatibleCandidateForQuery(
              layouts, tensorType, memType, *expanded, *queryLayout,
              getContextualMaxNReg(op), getTMemLdStRowPlan(*queryLayout),
              /*requireUnique=*/true);
        }
      }
      if (supported) {
        auto attr = LinearEncodingAttr::get(tensorType.getContext(),
                                            std::move(*expanded));
        if (!llvm::is_contained(layouts, attr))
          layouts.push_back(attr);
      }
    }
  }
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
      ll = getDistributedLayoutForTmemLdStAnchored(memLL, atom, numWarps,
                                                         bitwidth);
    } else {
      ll = getDistributedLayoutForTmemLdSt(memType, atom, numWarps);
    }
    if (ll)
      appendTMemCompatibleCandidate(layouts, op, tensorType, memType, *ll);
  }
  if (auto splitLongM = getTmemLoadLayoutSplitLongM(tensorType, memType,
                                                    numWarps))
    appendTMemCompatibleCandidate(layouts, op, tensorType, memType,
                                  splitLongM.value());
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
// TensorDescIm2ColType Printer/Parser
//===----------------------------------------------------------------------===//
// Format: !ttng.tensordesc_im2col<64x128xf16>
//         !ttng.tensordesc_im2col<64x128xf16, #shared>
Type TensorDescIm2ColType::parse(AsmParser &parser) {
  if (failed(parser.parseLess()))
    return Type();

  SmallVector<int64_t> shape;
  if (failed(parser.parseDimensionList(shape, /*allowDynamic=*/false)))
    return Type();

  Type elementType;
  if (failed(parser.parseType(elementType)))
    return Type();

  Attribute sharedLayout;
  if (succeeded(parser.parseOptionalComma())) {
    if (failed(parser.parseAttribute(sharedLayout)))
      return Type();
  }

  if (failed(parser.parseGreater()))
    return Type();

  Location loc = parser.getEncodedSourceLoc(parser.getCurrentLocation());
  return TensorDescIm2ColType::getChecked(loc, parser.getContext(), shape,
                                          elementType, sharedLayout);
}

void TensorDescIm2ColType::print(AsmPrinter &printer) const {
  printer << "<";
  for (auto dim : getShape())
    printer << dim << "x";
  printer << getElementType();
  if (getSharedLayout())
    printer << ", " << getSharedLayout();
  printer << ">";
}

//===----------------------------------------------------------------------===//
// TensorDescIm2ColType Verifier
//===----------------------------------------------------------------------===//
LogicalResult
TensorDescIm2ColType::verify(function_ref<InFlightDiagnostic()> emitError,
                             ArrayRef<int64_t> shape, Type elementType,
                             Attribute sharedLayout) {
  if (shape.size() != 2) {
    return emitError()
           << "TensorDescIm2ColType requires rank-2 shape, got rank "
           << shape.size();
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
                         bool allowReorder,
                         std::optional<Location> loc) const override {
    if (isTensorMemoryEncoding(srcEnc)) {
      return inferTMemReshapeOpEncoding(srcShape, srcEnc, dstShape, dstEnc,
                                        loc);
    }
    return getDelegate()->inferReshapeOpEncoding(srcShape, srcEnc, dstShape,
                                                 dstEnc, allowReorder, loc);
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

  LogicalResult verifyCatOpEncodingCompatibility(Operation *op) const override {
    return getDelegate()->verifyCatOpEncodingCompatibility(op);
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

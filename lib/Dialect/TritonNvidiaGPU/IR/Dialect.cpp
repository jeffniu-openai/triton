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
  auto base =
      triton::gpu::tensorMemoryToLinearLayout(trailingShape, legacy, &baseError);
    if (!base) {
      if (error != nullptr) {
      *error = "tensor memory layout sugar " +
               stringifyAttribute(encoding) +
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

std::optional<TensorMemoryEncodingAttr>
matchTensorMemoryLegacyEncoding(ArrayRef<int64_t> shape, Attribute layout) {
  if (auto legacy = dyn_cast<TensorMemoryEncodingAttr>(layout))
    return legacy;
  auto linear = dyn_cast<TensorMemoryLinearEncodingAttr>(layout);
  if (!linear || shape.size() != 2)
    return std::nullopt;
  auto normalizedLinear =
      normalizeTensorMemoryLinearLayoutForAnalysis(linear.getLinearLayout());
  auto cga = linear.getCGALayout();
  if (linear.getTwoCTAs()) {
    auto kBlock = StringAttr::get(layout.getContext(), "block");
    if (cga.getLinearLayout().getBasis(kBlock, 0) != ArrayRef<int32_t>{1, 0})
      return std::nullopt;
  }
  std::optional<TensorMemoryEncodingAttr> bestMatch;
  bool isM64TwoCTA = linear.getTwoCTAs() &&
                     llvm::any_of(cga.getCTAsPerCGA(),
                                  [](unsigned count) { return count > 1; });
  auto isBetterMatch = [&](TensorMemoryEncodingAttr candidate) {
    if (!bestMatch)
      return true;
    auto candidateArea =
        static_cast<uint64_t>(candidate.getBlockM()) * candidate.getBlockN();
    auto bestArea =
        static_cast<uint64_t>(bestMatch->getBlockM()) * bestMatch->getBlockN();
    if (candidateArea != bestArea)
      return candidateArea > bestArea;
    if (candidate.getBlockM() != bestMatch->getBlockM())
      return candidate.getBlockM() > bestMatch->getBlockM();
    if (candidate.getBlockN() != bestMatch->getBlockN())
      return candidate.getBlockN() > bestMatch->getBlockN();
    // Prefer the densest legacy layout when multiple legacy encodings
    // normalize to the same canonical TMEM-linear layout.
    return candidate.getColStride() < bestMatch->getColStride();
  };
  for (unsigned blockM : {64u, 128u}) {
    for (unsigned blockN = 1; blockN <= 512; blockN <<= 1) {
      if (isM64TwoCTA && blockM == 64 && blockN == 1)
        continue;
      for (unsigned colStride : {1u, 2u, 4u}) {
        auto candidate = TensorMemoryEncodingAttr::get(
            layout.getContext(), blockM, blockN, colStride, cga,
            linear.getTwoCTAs());
        std::string candidateError;
        auto maybeCandidate = tryGetCanonicalTensorMemoryLinearLayout(
            shape, candidate, &candidateError);
        if (!maybeCandidate)
          continue;
        auto normalizedCandidate = normalizeTensorMemoryLinearLayoutForAnalysis(
            *maybeCandidate);
        if (normalizedCandidate == normalizedLinear && isBetterMatch(candidate))
          bestMatch = candidate;
      }
    }
  }
  return bestMatch;
}

std::optional<TensorMemoryEncodingAttr>
matchTensorMemoryLegacyEncoding(MemDescType memDescType) {
  auto layout = memDescType.getEncoding();
  auto rank = cast<LayoutEncodingTrait>(layout).getRank();
  return matchTensorMemoryLegacyEncoding(
      memDescType.getShape().take_back(rank), layout);
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
  auto ll = triton::gpu::toLinearLayout(memDescType);
  auto layoutRank = ll.getNumOutDims();
  auto extraRank = memDescType.getRank() - layoutRank;
  auto bitwidth = memDescType.getElementTypeBitWidth();
  int nRow = ll.getInDimSize(kRow);
  int nCol = ll.getInDimSize(kCol) / (32 / bitwidth);
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

uint32_t getTMemViewOffset(MemDescType memDescType, ArrayRef<int32_t> offsets) {
  assert(offsets.size() == memDescType.getRank());
  auto *ctx = memDescType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto ll = normalizeTensorMemoryLinearLayoutForAnalysis(
      triton::gpu::toLinearLayout(memDescType));
  auto layoutRank = ll.getNumOutDims();
  auto extraRank = memDescType.getRank() - layoutRank;

  SmallVector<std::pair<StringAttr, int32_t>> logicalOffsets;
  logicalOffsets.reserve(layoutRank);
  for (auto [dim, offset] :
       llvm::zip_equal(ll.getOutDimNames(), offsets.drop_front(extraRank))) {
    logicalOffsets.push_back({dim, offset});
  }

  auto rowColBlock = ll.pseudoinvert().apply(logicalOffsets);
  uint32_t bitwidth = memDescType.getElementTypeBitWidth();
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
    auto singleBufferCols = ll.getInDimSize(kCol) / (32 / bitwidth);
    offsetCol += linearizePrefixOffsets(
                     memDescType.getShape().take_front(extraRank),
                     offsets.take_front(extraRank)) *
                 singleBufferCols;
  }
  return offsetCol | offsetRow << 16;
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp) {
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
    auto nCol = tile.getOutDimSize(kCol);
    auto bases = tile.getBases();
    bases[kWarp].push_back({32, 0});
    bases[kWarp].push_back({64, 0});
    tile = LinearLayout(std::move(bases), {{kRow, 128}, {kCol, nCol}}, false);
  }
  return tile;
}

static std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(const LinearLayout &ll, TMemAccessAtom atom,
                                unsigned numWarps, int bitwidth) {
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
      if (auto perCTA = getDistributedLayoutForTmemLdSt(*maybePerCTA, atom,
                                                        numWarps, bitwidth)) {
        return *perCTA * blockOnly;
      }
    }
  }
  auto canCompose = [](const LinearLayout &inner,
                       const LinearLayout &outer) -> bool {
    for (StringAttr outDim : inner.getOutDimNames()) {
      if (inner.getOutDimSize(outDim) > outer.getInDimSize(outDim))
        return false;
    }
    return true;
  };
  // This code is dual to the one in lowerTMemLdSt
  if (bitwidth != 32) {
    // TODO move this to a helper function
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

    // Pack contiguous elements
    // This works to pack b8 or b16 into b32 but also b8 into b16 and recurse
    if (bestContig > 1) {
      auto ret = getDistributedLayoutForTmemLdSt(quot, atom, numWarps,
                                                 bitwidth * bestContig);
      if (!ret)
        return ret;
      auto castbbitwidth = LinearLayout::identity1D(bestContig, kReg, dims[1]);
      return castbbitwidth * ret.value();
    }
    if (auto maybeQuot = divideLeft(
            ll, LinearLayout::zeros1D(32 / bitwidth, rowColDims[1], dims[1]) *
                    LinearLayout::identity1D(2, rowColDims[1], dims[1]));
        bitwidth == 16 && maybeQuot) {
      // Unpacked case
      auto ret =
          getDistributedLayoutForTmemLdSt(*maybeQuot, atom, numWarps, 32);
      if (!ret)
        return ret;
      auto castbbitwidth = LinearLayout::identity1D(2, kReg, dims[1]);
      return castbbitwidth * ret.value();
    } else if (auto maybeQuot =
                   divideLeft(ll, LinearLayout::zeros1D(
                                      32 / bitwidth, rowColDims[1], dims[1]))) {
      // Software padding
      assert(maybeQuot);
      return getDistributedLayoutForTmemLdSt(*maybeQuot, atom, numWarps, 32);
    } else if (ll.getInDimSize(rowColDims[1]) == 1) {
      // Software padding with just one column
      return getDistributedLayoutForTmemLdSt(ll, atom, numWarps, 32);
    } else {
      return std::nullopt;
    }
  }
  // getTileLayout returns the layout for a bitwidth of 32
  assert(bitwidth == 32);
  auto tile = getTileLayout(ctx, atom, false, /*withWarp=*/false);
  // Plan:
  // tile: register, lane -> row, cols
  // ll: row, cols -> dim0, dim1
  // We extend the tile to have the right vectorisation + warps and
  // the result is given by
  // ll o tile : register, lane, warp -> dim0, dim1

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
      ll.getBasis(rowColDims[0], llvm::Log2_32(16)) == ArrayRef{0, 0};

  // We are choosing the distributed layout (ll o tile). In the lowering
  // we will do ll^{-1} o (ll o tile) and we expect to get tile back.
  // For this to be possible, ll should accept a left-inverse, that is, it
  // should be injective
  // In less fancy words, we look for the `comp` layout not to have any zero
  // basis as that would disallow the resulting layout to be left-divisible by
  // the tile
  auto compInput = tile;
  if (hasBlockDim)
    compInput *= LinearLayout::identity1D(1, kBlock, kBlock);
  if (!canCompose(compInput, ll))
    return std::nullopt;
  auto comp =
      compInput.compose(ll).sublayout({kReg, kLane}, to_vector(ll.getOutDimNames()));
  if (instr32Rows) {
    // We will use 16x32bx2 instruction for lane=16 so we remove the last lane
    // basis
    comp = comp.resizeInDim(kLane, comp.getInDimSize(kLane) / 2);
  }
  if (!comp.isInjective())
    return std::nullopt;

  // Fit the warp bases either tiling on the RHS or in row=16
  StringAttr row16;
  // If we need to fit something (the instruction does not cover it
  // and the layout has 32 rows) we first try to fit a warp, and if we
  // can't we fit a register
  if (!instr32Rows && !layout16Rows) {
    if (numWarps > 4) {
      row16 = kWarp;
    } else {
      row16 = kReg;
    }
  }

  // We reserve enough columns to fit in the warps
  int warpsToTile = numWarps / ((row16 == kWarp) ? 8 : 4);
  // Cap warps to tile above by nColsMissing. The rest go to broadcasting
  int warpBroadcast = warpsToTile / std::min(nColsMissing, warpsToTile);
  warpsToTile /= warpBroadcast;
  nColsMissing /= warpsToTile;

  if (nColsMissing > 1) {
    if (instr32Rows && layout16Rows) {
      // If the lane 16 would load repeated data, instead we make it load half
      // of the data via the 16x32bx2 instruction
      tile = divideLeft(tile, LinearLayout::identity1D(2, kLane, rowColDims[0]))
                 .value();
      tile *= LinearLayout::identity1D(nColsMissing / 2, kReg, rowColDims[1]) *
              LinearLayout::identity1D(2, kLane, rowColDims[1]);

    } else {
      tile *= LinearLayout::identity1D(nColsMissing, kReg, rowColDims[1]);
    }
  }

  // add the warp bases. The M=64 + 2CTA case has already been handled
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
  // Add CTAs as a trivial map
  if (hasBlockDim) {
    auto nCTAs = ll.getInDimSize(kBlock);
    tile *= LinearLayout::identity1D(nCTAs, kBlock, kBlock);
  }
  assert(tile.getOutDimSize(rowColDims[1]) == ll.getInDimSize(rowColDims[1]));
  if (!canCompose(tile, ll))
    return std::nullopt;

  auto ret = tile.compose(ll);
  auto nonZero = [](auto val) { return val != 0; };
  for (const auto &dimBases : llvm::make_second_range(ret.getBases())) {
    if (!llvm::all_of(dimBases, [&](const auto &basis) {
          return std::count_if(basis.begin(), basis.end(), nonZero) <= 1;
        })) {
      return std::nullopt;
    }
  }
  auto withoutBroadcast = ret;
  for (auto inDim : ret.getInDimNames())
    withoutBroadcast = withoutBroadcast.removeZeroBasesAlongDim(inDim);
  if (!withoutBroadcast.isInvertible())
    return std::nullopt;
  return ret;
}

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps) {
  assert(memType.getMemorySpace() ==
         TensorMemorySpaceAttr::get(memType.getContext()));
  assert(numWarps >= 4 && llvm::isPowerOf2_32(numWarps) &&
         "numWarps must be a power of 2 and >= 4");
  assert(atom != TMemAccessAtom::I16x32bx2 &&
         "This layout is inferred sometimes for the 32x32b atom");
  auto ll = toLinearLayout(memType);
  auto bitwidth = memType.getElementTypeBitWidth();
  return getDistributedLayoutForTmemLdSt(ll, atom, numWarps, bitwidth);
}

static bool isTMemCompatibleCandidate(Operation *op, RankedTensorType tensorType,
                                      gpu::MemDescType memType,
                                      const LinearLayout &layout) {
  auto candidateEncoding =
      LinearEncodingAttr::get(tensorType.getContext(), layout);
  auto candidateType = tensorType.cloneWithEncoding(candidateEncoding);
  auto maxnreg = getContextualMaxNReg(op);
  return succeeded(
      computeTMemLdStEncodingInfo(candidateType, memType, maxnreg));
}

DistributedEncodingTrait getDefaultLayoutForTmemLdSt(gpu::MemDescType memType,
                                                     unsigned numWarps) {
  auto *ctx = memType.getContext();
  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  if (prefer16x256) {
    auto layout = getDistributedLayoutForTmemLdSt(
        memType, TMemAccessAtom::I16x256b, numWarps);
    if (layout) {
      return LinearEncodingAttr::get(ctx, std::move(*layout));
    }
  }
  auto layout = getDistributedLayoutForTmemLdSt(
      memType, TMemAccessAtom::I32x32b, numWarps);
  assert(layout);
  return LinearEncodingAttr::get(ctx, std::move(*layout));
}

std::optional<DistributedEncodingTrait>
getTmemLoadLayoutSplitLongM(RankedTensorType tensorType, MemDescType memType,
                            int numWarps) {
  if (numWarps != 8)
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

SmallVector<DistributedEncodingTrait>
getTmemCompatibleLayouts(Operation *op, RankedTensorType tensorType,
                         MemDescType memType) {
  int numWarps = lookupNumWarps(op);
  SmallVector<DistributedEncodingTrait> layouts;
  if (numWarps % 4 != 0)
    return layouts;
  LinearLayout memLL = [&]() -> LinearLayout {
    if (isa<TensorMemoryScalesEncodingAttr>(memType.getEncoding()))
      return toLinearLayout(memType.getShape(), memType.getEncoding());
    std::string error;
    auto maybeCanonical = getCanonicalTMemLinearEncoding(memType, &error);
    if (!maybeCanonical)
      return LinearLayout();
    return maybeCanonical->getLinearLayout();
  }();
  if (memLL.getNumOutDims() == 0)
    return layouts;
  int bitwidth = memType.getElementTypeBitWidth();
  for (auto atom : {TMemAccessAtom::I32x32b, TMemAccessAtom::I16x256b,
                    TMemAccessAtom::I16x128b, TMemAccessAtom::I16x64b,
                    TMemAccessAtom::I16x32bx2}) {
    auto ll = getDistributedLayoutForTmemLdSt(memLL, atom, numWarps, bitwidth);
    if (ll && isTMemCompatibleCandidate(op, tensorType, memType, *ll)) {
      layouts.push_back(LinearEncodingAttr::get(tensorType.getContext(),
                                                std::move(ll.value())));
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
  if (!linearLayout.isSurjective())
    return emitError() << "The layout must be surjective";

  auto *ctx = linearLayout.getOutDimNames().begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  auto stripped = linearLayout.removeZeroBasesAlongDim(kRow)
                      .removeZeroBasesAlongDim(kCol);
  if (llvm::is_contained(linearLayout.getInDimNames(), kBlock))
    stripped = stripped.removeZeroBasesAlongDim(kBlock);
  if (!stripped.isInvertible()) {
    return emitError()
           << "After removing zero bases the layout must be bijective";
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
  auto isInterleaved = [](MemDescType memdesc) {
    auto enc = dyn_cast<TensorMemoryEncodingAttr>(memdesc.getEncoding());
    return enc && getTmemAllocSizes(memdesc).numRows != 64 &&
           enc.getBlockM() == 64;
  };

  auto itf = cast<MMAv5OpInterface>(op);
  if (isInterleaved(itf.getA().getType()) &&
      isInterleaved(itf.getAccumulator().getType())) {
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
      return inferTMemSubsliceOpEncoding(srcShape, srcEncoding, dstShape,
                                         offsets, dstEncoding, loc);
    }
    return getDelegate()->inferMemDescSubsliceOpEncoding(
        srcShape, srcAllocShape, srcEncoding, dstShape, offsets, dstEncoding,
        loc);
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
      if (*expectedCanonical == *gotCanonical) {
        return success();
      }
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

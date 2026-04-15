#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "triton/Tools/LayoutUtils.h"
#include "third_party/f2reduce/f2reduce.h"
#include <algorithm>
#include <tuple>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

namespace {

constexpr int maxRegisters = 256;
constexpr int largestTmemLoadStore = 128;
static Value getUniqueFunctionArgForwardingSource(BlockArgument blockArg) {
  auto func = dyn_cast_if_present<FuncOp>(blockArg.getOwner()->getParentOp());
  if (!func)
    return {};

  auto module = func->getParentOfType<ModuleOp>();
  if (!module)
    return {};

  unsigned argNumber = blockArg.getArgNumber();
  Value uniqueOperand;
  bool found = false;
  bool ambiguous = false;
  module.walk([&](CallOp call) {
    if (call.getCallee() != func.getName())
      return;
    if (argNumber >= call.getNumOperands()) {
      ambiguous = true;
      return;
    }
    Value operand = call.getOperand(argNumber);
    if (!found) {
      uniqueOperand = operand;
      found = true;
      return;
    }
    if (operand != uniqueOperand)
      ambiguous = true;
  });
  if (!found || ambiguous)
    return {};
  return uniqueOperand;
}

} // namespace

Value getTMemForwardingSource(Value memDesc) {
  auto blockArg = dyn_cast<BlockArgument>(memDesc);
  if (!blockArg)
    return {};

  auto *parentOp = blockArg.getOwner()->getParentOp();
  if (auto partitions =
          dyn_cast_if_present<gpu::WarpSpecializePartitionsOp>(parentOp)) {
    auto captures = partitions.getExplicitCaptures();
    unsigned argNumber = blockArg.getArgNumber();
    if (argNumber >= captures.size())
      return {};
    return captures[argNumber];
  }

  return getUniqueFunctionArgForwardingSource(blockArg);
}

namespace {

static int getMatrixRankForLayout(std::unique_ptr<uint64_t[]> matrix, int numRows,
                                  int numCols) {
  f2reduce::inplace_rref_strided(matrix.get(), numRows, numCols,
                                 /*stride=*/1);
  int rank = 0;
  for (int r = 0; r < numRows; ++r)
    rank += matrix[r] != 0;
  return rank;
}

static std::unique_ptr<uint64_t[]>
concatMatricesForImageCheck(const LinearLayout &A, const LinearLayout &B) {
  assert(A.getTotalOutDimSizeLog2() >= B.getTotalOutDimSizeLog2() &&
         "A must have at least as many output bits as B");
  int numColsA = A.getTotalInDimSizeLog2();
  auto concat = getMatrix(A);
  auto bMatrix = getMatrix(B);
  int rowA = 0;
  int rowB = 0;
  for (auto [outDim, outDimSize] : A.getOutDims()) {
    for (int r = 0; r < llvm::Log2_32(outDimSize); ++r) {
      if (r < llvm::Log2_32(B.getOutDimSize(outDim))) {
        concat[rowA] |= bMatrix[rowB] << numColsA;
        rowB++;
      }
      rowA++;
    }
  }
  return concat;
}

static bool imageContainedIn(const LinearLayout &A, const LinearLayout &B) {
  if (A.getTotalOutDimSizeLog2() < B.getTotalOutDimSizeLog2())
    return false;
  auto rankA = getMatrixRankForLayout(
      getMatrix(A), A.getTotalOutDimSizeLog2(), A.getTotalInDimSizeLog2());
  auto rankConcat = getMatrixRankForLayout(
      concatMatricesForImageCheck(A, B), A.getTotalOutDimSizeLog2(),
      A.getTotalInDimSizeLog2() + B.getTotalInDimSizeLog2());
  return rankA == rankConcat;
}

static bool canInvertAndComposeSafely(const LinearLayout &inner,
                                      const LinearLayout &outer) {
  auto outDims = llvm::to_vector(inner.getOutDimNames());
  auto outerOutDims = llvm::to_vector(outer.getOutDimNames());
  if (outDims.size() != outerOutDims.size() ||
      llvm::any_of(outDims, [&](StringAttr dim) {
        return !llvm::is_contained(outerOutDims, dim);
      }))
    return false;

  auto A = outer.transposeOuts(outDims);
  const auto &B = inner;
  for (auto dim : outDims) {
    if (A.getOutDimSize(dim) < B.getOutDimSize(dim))
      return false;
  }

  SmallVector<StringAttr> identityDims;
  for (auto dim : A.getInDimNames()) {
    if (B.hasInDim(dim) &&
        A.sublayout(dim, outDims) == B.sublayout(dim, outDims)) {
      identityDims.push_back(dim);
    }
  }
  SmallVector<StringAttr> aNonIdentityInDims;
  SmallVector<StringAttr> bNonIdentityInDims;
  for (auto dim : A.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim))
      aNonIdentityInDims.push_back(dim);
  }
  for (auto dim : B.getInDimNames()) {
    if (!llvm::is_contained(identityDims, dim))
      bNonIdentityInDims.push_back(dim);
  }
  if (aNonIdentityInDims.empty() != bNonIdentityInDims.empty())
    return false;
  if (aNonIdentityInDims.empty())
    return true;

  auto aReduced = A.sublayout(aNonIdentityInDims, outDims);
  auto bReduced = B.sublayout(bNonIdentityInDims, outDims);
  return imageContainedIn(aReduced, bReduced);
}

// Similar to largestVectorisation in TritonGPUToLLVM/Utility.cpp
std::optional<std::tuple<LinearLayout, ColumnAction, int>>
getVec(const LinearLayout &cvt, const LinearLayout &tile, int maxnreg) {
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto kReg = StringAttr::get(ctx, "register");
  auto kCol = StringAttr::get(ctx, "col");
  LinearLayout reps, vec;
  ColumnAction perm;
  // Heuristic:
  // Do not use more than half the registers as otherwise it's prone to spilling
  assert(maxnreg / 2 <= largestTmemLoadStore);
  auto maxReg = maxnreg / 2;
  // Heuristic:
  // If maxnreg is 256 and we need more than one message, we don't use max
  // vectorisation as ptxas' scheduler breaks...
  if (maxnreg == 256 && cvt.getInDimSize(kReg) > maxReg) {
    maxReg /= 2;
  }
  auto maxVec = maxReg / tile.getInDimSize(kReg);
  int i = 1;
  for (; i <= maxVec; i *= 2) {
    vec = LinearLayout::identity1D(i, kReg, kCol);
    auto vecTile = tile * vec;
    auto maybePerm = regPermForDivide(cvt, vecTile, /*left=*/true);
    if (!maybePerm) {
      break;
    }
    // nb. We could remove this part once we are confident the algo works
    perm = *maybePerm;
    auto newCvt = maybePerm->apply(cvt);
    auto maybeReps = getReps(newCvt, vecTile);
    if (!maybeReps.has_value()) {
      break;
    }
    reps = *maybeReps;
  }
  if (i == 1) {
    // Couldn't lower the tile
    return std::nullopt;
  }
  // i is the smallest power of 2 that *cannot* be used to lower the tile
  // so we return i / 2.
  assert(i > 1);
  return std::make_tuple(std::move(reps), std::move(perm),
                         (i / 2) * tile.getInDimSize(kReg));
}

FailureOr<unsigned> getExpectedTMemLoadValueCount(const TMemLdStEncodingInfo &info,
                                                  unsigned bitwidth) {
  auto kReg = *info.reps.getInDimNames().begin();

  if (bitwidth < 32) {
    // The current sanity check is only used to avoid false-positive direct
    // lowering claims on 32-bit TMEM load/store paths. Packed/unpacked
    // subword cases are validated by the existing lowering logic.
    return failure();
  }

  if (info.broadcast) {
    TMemLdStEncodingInfo nested = info;
    nested.broadcast = std::nullopt;
    auto nestedCount = getExpectedTMemLoadValueCount(nested, bitwidth);
    if (failed(nestedCount))
      return failure();

    uint32_t broadcastMask = info.reps.getFreeVariableMasks().lookup(kReg);
    unsigned expectedInputCount =
        info.reps.getInDimSize(kReg) / (1u << llvm::popcount(broadcastMask));
    if (*nestedCount != expectedInputCount)
      return failure();
  }

  return info.reps.getInDimSize(kReg);
}

static int32_t lookupLinearLayoutCoord(
    ArrayRef<std::pair<StringAttr, int32_t>> coords, StringAttr dim) {
  for (auto [name, value] : coords) {
    if (name == dim)
      return value;
  }
  return 0;
}

static SmallVector<std::pair<StringAttr, int32_t>>
makeFullLinearLayoutCoords(ArrayRef<StringAttr> dims,
                           ArrayRef<std::pair<StringAttr, int32_t>> sparse) {
  SmallVector<std::pair<StringAttr, int32_t>> result;
  result.reserve(dims.size());
  for (auto dim : dims)
    result.push_back({dim, lookupLinearLayoutCoord(sparse, dim)});
  return result;
}

static std::optional<SmallVector<int32_t>>
getLogicalRowAnchorBasis(const LinearLayout &layout, int32_t logicalRow) {
  auto outDims = llvm::to_vector(layout.getOutDimNames());
  if (outDims.empty())
    return std::nullopt;

  auto *ctx = outDims.front().getContext();
  auto kRow = StringAttr::get(ctx, "row");
  if (!layout.hasInDim(kRow) || logicalRow < 0 ||
      logicalRow >= layout.getInDimSize(kRow))
    return std::nullopt;

  auto inDims = llvm::to_vector(layout.getInDimNames());
  auto realizedCoords =
      layout.apply(makeFullLinearLayoutCoords(inDims, {{kRow, logicalRow}}));

  // TMEM row plans are expressed in the analyzed layout's logical row space.
  // The physical warp anchor is the image of that logical row coordinate under
  // the full linear layout, even when row/col/block families are permuted.
  SmallVector<int32_t> anchor;
  anchor.reserve(outDims.size());
  for (StringAttr outDim : outDims)
    anchor.push_back(lookupLinearLayoutCoord(realizedCoords, outDim));
  return anchor;
}

static FailureOr<LinearLayout>
computeLeftInverseLayout(const LinearLayout &layout, std::string *error) {
  if (!layout.isInjective()) {
    if (error)
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
  }

  int numPhysBits = layout.getTotalOutDimSizeLog2();
  int numLogicalBits = layout.getTotalInDimSizeLog2();
  int numCols = numPhysBits + numLogicalBits;
  auto matrix = getMatrix(layout);
  std::unique_ptr<uint64_t[]> combined(new uint64_t[numLogicalBits]());

  for (int logicalBit = 0; logicalBit < numLogicalBits; ++logicalBit) {
    uint64_t row = 0;
    for (int physBit = 0; physBit < numPhysBits; ++physBit) {
      row |= ((matrix[physBit] >> logicalBit) & 1ull) << physBit;
    }
    row |= 1ull << (numPhysBits + logicalBit);
    combined[logicalBit] = row;
  }

  f2reduce::inplace_rref_strided(combined.get(), numLogicalBits, numCols,
                                 /*stride=*/1);

  SmallVector<int32_t> pivotRowOfCol(numPhysBits, -1);
  for (int row = 0; row < numLogicalBits; ++row) {
    uint64_t rrefRow = combined[row];
    if (rrefRow == 0)
      continue;
    int pivot = __builtin_ctzll(rrefRow);
    if (pivot >= numPhysBits) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
    pivotRowOfCol[pivot] = row;
  }

  LinearLayout::BasesT bases;
  auto inDimName = *layout.getOutDimNames().begin();
  auto outDimName = *layout.getInDimNames().begin();
  auto &flatBases = bases[inDimName];
  flatBases.reserve(numPhysBits);
  for (int physBit = 0; physBit < numPhysBits; ++physBit) {
    int row = pivotRowOfCol[physBit];
    uint64_t basis = row == -1 ? 0 : (combined[row] >> numPhysBits);
    flatBases.push_back({static_cast<int32_t>(basis)});
  }

  LinearLayout flat(std::move(bases),
                    {{outDimName, layout.getTotalInDimSize()}},
                    /*requireSurjective=*/false);

  SmallVector<std::pair<StringAttr, int32_t>> inDims;
  inDims.reserve(layout.getNumOutDims());
  for (auto dim : layout.getOutDimNames())
    inDims.push_back({dim, layout.getOutDimSize(dim)});

  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.reserve(layout.getNumInDims());
  for (auto dim : layout.getInDimNames())
    outDims.push_back({dim, layout.getInDimSize(dim)});

  return flat.reshapeIns(inDims).reshapeOuts(outDims);
}

static FailureOr<LinearLayout>
restrictTMemAnalysisLayoutToShape(const LinearLayout &layout,
                                  ArrayRef<int64_t> shape,
                                  std::string *error) {
  if (shape.size() != static_cast<size_t>(layout.getNumOutDims())) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return failure();
  }

  auto restrictedLayout = layout;
  for (auto [dim, dimName] : llvm::enumerate(layout.getOutDimNames())) {
    int64_t dimSize = shape[dim];
    if (dimSize > restrictedLayout.getOutDimSize(dimName)) {
      if (error)
        *error = "TMEM reshape rank does not match the canonical TMEM layout";
      return failure();
    }
    restrictedLayout = restrictedLayout.resizeOutDim(dimName, dimSize);
  }

  return restrictedLayout;
}

static LogicalResult verifyTMemSubsliceProjection(
    const LinearLayout &srcInv, ArrayRef<StringAttr> srcLogicalDims,
    ArrayRef<std::pair<StringAttr, int32_t>> encodedOffsets,
    ArrayRef<std::pair<StringAttr, int32_t>> baseCoords,
    const LinearLayout &dstLayout, ArrayRef<int64_t> dstShape,
    std::string *error) {
  auto verificationLayout = dstLayout;
  auto maybeDstInv = computeLeftInverseLayout(verificationLayout, error);
  if (failed(maybeDstInv)) {
    auto normalized = normalizeTensorMemoryLinearLayoutForAnalysis(dstLayout);
    std::string normalizedError;
    auto normalizedInv = computeLeftInverseLayout(normalized, &normalizedError);
    if (succeeded(normalizedInv) &&
        normalized.getNumOutDims() == dstLayout.getNumOutDims() &&
        llvm::equal(normalized.getOutDimSizes(), dstShape) &&
        static_cast<int64_t>(normalized.getTotalInDimSize()) ==
            product<int64_t>(dstShape)) {
      verificationLayout = normalized;
      maybeDstInv = std::move(normalizedInv);
      if (error)
        error->clear();
    } else {
      if (error && error->empty())
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
  }

  auto dstLogicalDims = llvm::to_vector(verificationLayout.getOutDimNames());
  auto dstBaseCoords =
      maybeDstInv->apply(makeFullLinearLayoutCoords(dstLogicalDims, {}));

  SmallVector<StringAttr> physDims = llvm::to_vector(srcInv.getOutDimNames());
  for (auto physDim : maybeDstInv->getOutDimNames()) {
    if (!llvm::is_contained(physDims, physDim))
      physDims.push_back(physDim);
  }

  for (auto [dim, dstDimSize] : llvm::enumerate(dstShape)) {
    auto dstDimName = dstLogicalDims[dim];
    for (int64_t step = 1; step < dstDimSize; step <<= 1) {
      auto srcPoint = llvm::to_vector(encodedOffsets);
      srcPoint[dim].second += static_cast<int32_t>(step);
      auto srcCoords =
          srcInv.apply(makeFullLinearLayoutCoords(srcLogicalDims, srcPoint));

      SmallVector<std::pair<StringAttr, int32_t>> dstPoint;
      dstPoint.reserve(dstLogicalDims.size());
      for (auto logicalDim : dstLogicalDims)
        dstPoint.push_back({logicalDim, logicalDim == dstDimName ? step : 0});
      auto dstCoords = maybeDstInv->apply(
          makeFullLinearLayoutCoords(dstLogicalDims, dstPoint));

      for (auto physDim : physDims) {
        int32_t expected = lookupLinearLayoutCoord(srcCoords, physDim) -
                           lookupLinearLayoutCoord(baseCoords, physDim);
        int32_t actual = lookupLinearLayoutCoord(dstCoords, physDim) -
                         lookupLinearLayoutCoord(dstBaseCoords, physDim);
        if (expected != actual) {
          if (error)
            *error = "unsupported tensor memory memdesc_subslice view";
          return failure();
        }
      }
    }
  }

  return success();
}

static LogicalResult verifyTMemIndexProjection(
    const LinearLayout &srcInv, ArrayRef<StringAttr> srcLogicalDims,
    const LinearLayout &dstLayout, ArrayRef<int64_t> dstShape,
    std::string *error) {
  auto maybeDstInv = computeLeftInverseLayout(dstLayout, error);
  if (failed(maybeDstInv)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }

  auto baseCoords = srcInv.apply(makeFullLinearLayoutCoords(srcLogicalDims, {}));
  auto dstLogicalDims = llvm::to_vector(dstLayout.getOutDimNames());
  auto dstBaseCoords =
      maybeDstInv->apply(makeFullLinearLayoutCoords(dstLogicalDims, {}));

  SmallVector<StringAttr> physDims = llvm::to_vector(srcInv.getOutDimNames());
  for (auto physDim : maybeDstInv->getOutDimNames()) {
    if (!llvm::is_contained(physDims, physDim))
      physDims.push_back(physDim);
  }

  for (auto [dim, dstDimSize] : llvm::enumerate(dstShape)) {
    auto srcDimName = srcLogicalDims[dim + 1];
    auto dstDimName = dstLogicalDims[dim];
    for (int64_t step = 1; step < dstDimSize; step <<= 1) {
      SmallVector<std::pair<StringAttr, int32_t>> srcPoint;
      srcPoint.reserve(srcLogicalDims.size());
      for (auto logicalDim : srcLogicalDims) {
        srcPoint.push_back(
            {logicalDim, logicalDim == srcDimName ? static_cast<int32_t>(step)
                                                  : 0});
      }
      auto srcCoords =
          srcInv.apply(makeFullLinearLayoutCoords(srcLogicalDims, srcPoint));

      SmallVector<std::pair<StringAttr, int32_t>> dstPoint;
      dstPoint.reserve(dstLogicalDims.size());
      for (auto logicalDim : dstLogicalDims) {
        dstPoint.push_back(
            {logicalDim, logicalDim == dstDimName ? static_cast<int32_t>(step)
                                                  : 0});
      }
      auto dstCoords = maybeDstInv->apply(
          makeFullLinearLayoutCoords(dstLogicalDims, dstPoint));

      for (auto physDim : physDims) {
        int32_t expected = lookupLinearLayoutCoord(srcCoords, physDim) -
                           lookupLinearLayoutCoord(baseCoords, physDim);
        int32_t actual = lookupLinearLayoutCoord(dstCoords, physDim) -
                         lookupLinearLayoutCoord(dstBaseCoords, physDim);
        if (expected != actual) {
          if (error)
            *error = "unsupported tensor memory memdesc_index view";
          return failure();
        }
      }
    }
  }

  return success();
}

static std::optional<gpu::MemDescType>
tryCreateMemDescType(MLIRContext *ctx, ArrayRef<int64_t> shape, Type elementType,
                     Attribute encoding, Attribute memorySpace,
                     bool mutableMemory, ArrayRef<int64_t> allocShape,
                     std::string *error) {
  std::string diagStr;
  llvm::raw_string_ostream diagOs(diagStr);
  ScopedDiagnosticHandler handler(
      ctx, [&](Diagnostic &diag) { diag.print(diagOs); });
  auto ty = gpu::MemDescType::getChecked(UnknownLoc::get(ctx), ctx, shape,
                                         elementType, encoding, memorySpace,
                                         mutableMemory, allocShape);
  if (!ty) {
    if (error)
      *error = diagStr.empty() ? "failed to create tensor memory memdesc type"
                               : diagStr;
    return std::nullopt;
  }
  return ty;
}

static bool canMaterializeTMemViewEncoding(MLIRContext *ctx,
                                           ArrayRef<int64_t> shape,
                                           ArrayRef<int64_t> allocShape,
                                           Attribute encoding) {
  if (isTensorMemoryEncoding(encoding) &&
      !isa<TensorMemoryScalesEncodingAttr>(encoding)) {
    auto layoutTrait = dyn_cast<LayoutEncodingTrait>(encoding);
    if (!layoutTrait)
      return false;
    auto layoutRank = static_cast<size_t>(layoutTrait.getRank());
    if (shape.size() < layoutRank)
      return false;
    if (!tryGetCanonicalTensorMemoryLinearLayout(shape.take_back(layoutRank),
                                                 encoding,
                                                 /*error=*/nullptr)) {
      return false;
    }
  }
  return tryCreateMemDescType(ctx, shape, IntegerType::get(ctx, 8), encoding,
                              TensorMemorySpaceAttr::get(ctx),
                              /*mutableMemory=*/false, allocShape,
                              /*error=*/nullptr)
      .has_value();
}

static std::optional<Attribute>
tryPreserveTMemViewEncodingForShape(MLIRContext *ctx, ArrayRef<int64_t> srcShape,
                                    ArrayRef<int64_t> dstShape,
                                    ArrayRef<int64_t> dstAllocShape,
                                    Attribute srcEncoding,
                                    std::string *error) {
  if (!isTensorMemoryEncoding(srcEncoding) ||
      isa<TensorMemoryScalesEncodingAttr>(srcEncoding))
    return std::nullopt;

  std::string localError;
  auto maybeLayout =
      getTMemViewAnalysisLinearLayout(srcShape, srcEncoding, &localError);
  if (!maybeLayout) {
    if (error && error->empty())
      *error = localError;
    return std::nullopt;
  }

  auto ll = *maybeLayout;
  auto restrictToShape = [&](ArrayRef<int64_t> shape) -> LogicalResult {
    if (shape.size() != static_cast<size_t>(ll.getNumOutDims())) {
      localError = "invalid tensor memory rank/layout combination";
      return failure();
    }
    if (!llvm::equal(ll.getOutDimSizes(), shape)) {
      auto maybeRestricted = restrictTMemAnalysisLayoutToShape(ll, shape, &localError);
      if (failed(maybeRestricted))
        return failure();
      ll = *maybeRestricted;
    }
    return success();
  };

  if (failed(restrictToShape(srcShape.take_back(ll.getNumOutDims())))) {
    if (error && error->empty())
      *error = localError;
    return std::nullopt;
  }

  while (static_cast<size_t>(ll.getNumOutDims()) > dstShape.size()) {
    auto firstDim = *ll.getOutDimNames().begin();
    if (ll.getOutDimSize(firstDim) != 1)
      break;
    ll = ll.squeezeOuts(firstDim);
  }
  if (static_cast<size_t>(ll.getNumOutDims()) > dstShape.size()) {
    if (error && error->empty())
      *error = "rank must be less than or equal to the memdesc rank for tensor "
               "memory";
    return std::nullopt;
  }

  if (failed(restrictToShape(dstShape.take_back(ll.getNumOutDims())))) {
    if (error && error->empty())
      *error = localError;
    return std::nullopt;
  }

  auto maybeEncoding =
      tryMakeTMemViewEncoding(ctx, ll,
                              getTensorMemoryTwoCTAs(srcEncoding).value_or(false),
                              &localError);
  if (!maybeEncoding) {
    if (error && error->empty())
      *error = localError;
    return std::nullopt;
  }
  if (tryCreateMemDescType(ctx, dstShape, IntegerType::get(ctx, 8), *maybeEncoding,
                           TensorMemorySpaceAttr::get(ctx),
                           /*mutableMemory=*/false, dstAllocShape,
                           &localError)) {
    return *maybeEncoding;
  }
  if (error && error->empty())
    *error = localError;
  return std::nullopt;
}

static LogicalResult preserveTMemViewEncodingIfValid(
    MLIRContext *ctx, ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
    ArrayRef<int64_t> dstAllocShape, Attribute srcEncoding,
    Attribute &dstEncoding, std::optional<Location> loc,
    std::string *error) {
  std::string preservedError;
  if (auto preserved = tryPreserveTMemViewEncodingForShape(
          ctx, srcShape, dstShape, dstAllocShape, srcEncoding,
          &preservedError)) {
    dstEncoding = *preserved;
    return success();
  }

  if (isTensorMemoryEncoding(srcEncoding) &&
      !isa<TensorMemoryScalesEncodingAttr>(srcEncoding)) {
    if (error && error->empty())
      *error = preservedError;
    if (error && !error->empty() && !preservedError.empty() &&
        *error != preservedError)
      return emitOptionalError(
          loc, *error,
          "; preserved tensor memory view encoding also failed: ",
          preservedError);
    return emitOptionalError(loc,
                             error && !error->empty() ? *error : preservedError);
  }

  if (tryCreateMemDescType(ctx, dstShape, IntegerType::get(ctx, 8),
                           srcEncoding, TensorMemorySpaceAttr::get(ctx),
                           /*mutableMemory=*/false, dstAllocShape,
                           &preservedError)) {
    dstEncoding = srcEncoding;
    return success();
  }
  if (error && error->empty())
    *error = preservedError;
  if (error && !error->empty() && !preservedError.empty() &&
      *error != preservedError)
    return emitOptionalError(
        loc, *error,
        "; preserved tensor memory view encoding also failed: ",
        preservedError);
  return emitOptionalError(loc,
                           error && !error->empty() ? *error : preservedError);
}

static std::optional<Attribute>
tryPreserveOuterIndexedTMemEncoding(MLIRContext *ctx,
                                    ArrayRef<int64_t> srcShape,
                                    ArrayRef<int64_t> dstShape,
                                    ArrayRef<int64_t> dstAllocShape,
                                    Attribute srcEncoding,
                                    std::string *error) {
  if (!isTensorMemoryEncoding(srcEncoding) ||
      isa<TensorMemoryScalesEncodingAttr>(srcEncoding))
    return std::nullopt;

  auto layoutTrait = dyn_cast<LayoutEncodingTrait>(srcEncoding);
  if (!layoutTrait)
    return std::nullopt;
  auto layoutRank = static_cast<size_t>(layoutTrait.getRank());
  if (srcShape.size() <= layoutRank || dstShape.size() < layoutRank)
    return std::nullopt;
  if (srcShape.take_back(layoutRank) != dstShape.take_back(layoutRank))
    return std::nullopt;

  std::string localError;
  if (!tryCreateMemDescType(ctx, dstShape, IntegerType::get(ctx, 8),
                            srcEncoding, TensorMemorySpaceAttr::get(ctx),
                            /*mutableMemory=*/false, dstAllocShape,
                            &localError)) {
    if (error && error->empty())
      *error = localError;
    return std::nullopt;
  }
  return srcEncoding;
}

static std::optional<Attribute>
tryPreserveExactTMemViewEncoding(MLIRContext *ctx, ArrayRef<int64_t> dstShape,
                                 ArrayRef<int64_t> dstAllocShape,
                                 Attribute srcEncoding, std::string *error) {
  if (!isTensorMemoryEncoding(srcEncoding) ||
      isa<TensorMemoryScalesEncodingAttr>(srcEncoding))
    return std::nullopt;

  std::string localError;
  if (!tryCreateMemDescType(ctx, dstShape, IntegerType::get(ctx, 8), srcEncoding,
                            TensorMemorySpaceAttr::get(ctx),
                            /*mutableMemory=*/false, dstAllocShape,
                            &localError)) {
    if (error && error->empty())
      *error = localError;
    return std::nullopt;
  }
  return srcEncoding;
}

static std::optional<TMemLdStQueryLayout>
getTMemViewAnalysisLayout(ArrayRef<int64_t> shape, Attribute encoding,
                          std::string *error) {
  auto layoutTrait = dyn_cast<LayoutEncodingTrait>(encoding);
  if (!layoutTrait) {
    if (error)
      *error = "expected tensor memory layout encoding";
    return std::nullopt;
  }
  auto layoutRank = static_cast<size_t>(layoutTrait.getRank());
  if (shape.size() < layoutRank) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return std::nullopt;
  }
  auto maybeLayout = tryGetCanonicalTensorMemoryLinearLayout(
      shape.take_back(layoutRank), encoding, error);
  if (!maybeLayout)
    return std::nullopt;
  bool twoCTAs = false;
  if (auto scales = dyn_cast<TensorMemoryScalesEncodingAttr>(encoding)) {
    twoCTAs = product<unsigned>(scales.getCGALayout().getCTAsPerCGA()) > 1;
  } else {
    auto maybeTwoCTAs = getTensorMemoryTwoCTAs(encoding);
    if (!maybeTwoCTAs) {
      if (error)
        *error = "expected tensor memory layout encoding";
      return std::nullopt;
    }
    twoCTAs = *maybeTwoCTAs;
  }
  return TMemLdStQueryLayout{*maybeLayout, twoCTAs,
                             SmallVector<int32_t>(maybeLayout->getNumInDims(),
                                                  0)};
}

static SmallVector<int32_t>
remapTMemLdStQueryOrigin(const TMemLdStQueryLayout &srcQuery,
                         const LinearLayout &dstLayout,
                         ArrayRef<std::pair<StringAttr, int32_t>> deltaCoords) {
  SmallVector<int32_t> origin;
  origin.reserve(dstLayout.getNumInDims());
  auto srcInDims = llvm::to_vector(srcQuery.layout.getInDimNames());
  for (StringAttr dim : dstLayout.getInDimNames()) {
    int32_t value = lookupLinearLayoutCoord(deltaCoords, dim);
    auto it = llvm::find(srcInDims, dim);
    if (it != srcInDims.end())
      value += srcQuery.origin[std::distance(srcInDims.begin(), it)];
    origin.push_back(value);
  }
  return origin;
}

static FailureOr<SmallVector<int32_t>>
remapTMemLdStQueryOriginThroughPhysicalCoords(
    const TMemLdStQueryLayout &srcQuery, const LinearLayout &dstLayout,
    std::string *error) {
  auto maybeDstInv = computeLeftInverseLayout(dstLayout, error);
  if (failed(maybeDstInv))
    return failure();

  auto srcLogicalDims = llvm::to_vector(srcQuery.layout.getInDimNames());
  SmallVector<std::pair<StringAttr, int32_t>> srcOriginCoords;
  srcOriginCoords.reserve(srcLogicalDims.size());
  for (auto [dim, value] : llvm::zip_equal(srcLogicalDims, srcQuery.origin))
    srcOriginCoords.push_back({dim, value});

  auto physCoords = srcQuery.layout.apply(
      makeFullLinearLayoutCoords(srcLogicalDims, srcOriginCoords));
  auto dstCoords = maybeDstInv->apply(physCoords);

  SmallVector<int32_t> dstOrigin;
  dstOrigin.reserve(dstLayout.getNumInDims());
  for (StringAttr dim : dstLayout.getInDimNames())
    dstOrigin.push_back(lookupLinearLayoutCoord(dstCoords, dim));
  return dstOrigin;
}

static bool
isTMemLdStQueryOriginRepresentable(const TMemLdStQueryLayout &query,
                                   const LinearLayout &layout) {
  auto queryInDims = llvm::to_vector(query.layout.getInDimNames());
  for (StringAttr dim : layout.getInDimNames()) {
    auto it = llvm::find(queryInDims, dim);
    if (it == queryInDims.end())
      continue;
    int32_t value = query.origin[std::distance(queryInDims.begin(), it)];
    if (value < 0 || value >= layout.getInDimSize(dim))
      return false;
  }
  return true;
}

static FailureOr<SmallVector<int32_t>>
remapTMemPhysicalOriginForBitcast(const TMemLdStQueryLayout &srcQuery,
                                  const LinearLayout &dstLayout,
                                  int srcBitwidth, int dstBitwidth,
                                  std::string *error) {
  SmallVector<int32_t> dstOrigin =
      remapTMemLdStQueryOrigin(srcQuery, dstLayout, /*deltaCoords=*/{});
  if (srcBitwidth == dstBitwidth)
    return dstOrigin;

  auto srcInDims = llvm::to_vector(srcQuery.layout.getInDimNames());
  auto dstInDims = llvm::to_vector(dstLayout.getInDimNames());
  if (dstInDims.empty())
    return dstOrigin;
  auto *ctx = dstInDims.front().getContext();
  auto kCol = StringAttr::get(ctx, "col");
  auto srcColIt = llvm::find(srcInDims, kCol);
  auto dstColIt = llvm::find(dstInDims, kCol);
  if (srcColIt == srcInDims.end() || dstColIt == dstInDims.end())
    return dstOrigin;

  int32_t srcCol = srcQuery.origin[std::distance(srcInDims.begin(), srcColIt)];
  int64_t dstColBits = static_cast<int64_t>(srcCol) * srcBitwidth;
  if (dstColBits % dstBitwidth != 0) {
    if (error)
      *error = "unsupported tensor memory memdesc_reinterpret view";
    return failure();
  }
  dstOrigin[std::distance(dstInDims.begin(), dstColIt)] =
      static_cast<int32_t>(dstColBits / dstBitwidth);
  return dstOrigin;
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

static SmallVector<int32_t>
addPrefixOffsetsToQueryOrigin(const LinearLayout &layout,
                              ArrayRef<int32_t> origin,
                              ArrayRef<int64_t> prefixShape,
                              ArrayRef<int32_t> prefixOffsets, int bitwidth) {
  SmallVector<int32_t> updated(origin.begin(), origin.end());
  if (prefixShape.empty())
    return updated;
  auto *ctx = layout.getInDimNames().begin()->getContext();
  auto kCol = StringAttr::get(ctx, "col");
  auto inDims = llvm::to_vector(layout.getInDimNames());
  auto it = llvm::find(inDims, kCol);
  if (it == inDims.end())
    return updated;
  int64_t singleBufferCols = layout.getInDimSize(kCol) / (32 / bitwidth);
  int64_t delta =
      linearizePrefixOffsets(prefixShape, prefixOffsets) * singleBufferCols;
  updated[std::distance(inDims.begin(), it)] += static_cast<int32_t>(delta);
  return updated;
}

struct LeadingUnitSubviewLayout {
  LinearLayout layout;
  SmallVector<int32_t> origin;
};

static std::optional<LeadingUnitSubviewLayout>
tryMakeLeadingUnitSubviewLayout(const LinearLayout &srcLayout,
                                ArrayRef<int64_t> srcShape,
                                ArrayRef<int64_t> dstShape,
                                ArrayRef<int32_t> offsets,
                                ArrayRef<int32_t> srcOrigin,
                                std::string *error) {
  if (srcShape.size() != dstShape.size() || srcShape.size() != offsets.size() ||
      srcShape.size() != static_cast<size_t>(srcLayout.getNumOutDims()))
    return std::nullopt;

  auto logicalDims = llvm::to_vector(srcLayout.getOutDimNames());
  unsigned prefixRank = 0;
  while (prefixRank < srcShape.size() &&
         (srcShape[prefixRank] != dstShape[prefixRank] ||
          offsets[prefixRank] != 0)) {
    if (dstShape[prefixRank] != 1 || offsets[prefixRank] < 0 ||
        offsets[prefixRank] >= srcShape[prefixRank]) {
      if (error)
        *error = "unsupported tensor memory leading-unit memdesc_subslice view";
      return std::nullopt;
    }
    ++prefixRank;
  }
  if (prefixRank == 0)
    return std::nullopt;
  for (unsigned dim = prefixRank; dim < srcShape.size(); ++dim) {
    if (srcShape[dim] != dstShape[dim] || offsets[dim] != 0)
      return std::nullopt;
  }

  auto inDims = llvm::to_vector(srcLayout.getInDimNames());
  SmallVector<int32_t> origin;
  if (srcOrigin.empty()) {
    origin.assign(srcLayout.getNumInDims(), 0);
  } else if (srcOrigin.size() == static_cast<size_t>(srcLayout.getNumInDims())) {
    origin.assign(srcOrigin.begin(), srcOrigin.end());
  } else {
    if (error)
      *error = "unsupported tensor memory leading-unit memdesc_subslice view";
    return std::nullopt;
  }

  SmallVector<std::pair<StringAttr, SmallVector<unsigned>>> eraseByDim;
  auto recordErase = [&](StringAttr inDim, unsigned basisIdx) {
    auto it = llvm::find_if(eraseByDim, [&](auto &entry) {
      return entry.first == inDim;
    });
    if (it == eraseByDim.end()) {
      eraseByDim.push_back({inDim, SmallVector<unsigned>{basisIdx}});
      return;
    }
    if (!llvm::is_contained(it->second, basisIdx))
      it->second.push_back(basisIdx);
  };

  auto findLogicalBasis = [&](unsigned logicalIdx, int32_t bit)
      -> std::optional<std::pair<unsigned, unsigned>> {
    std::optional<std::pair<unsigned, unsigned>> match;
    auto allBases = srcLayout.getBases();
    for (auto [inIdx, inDim] : llvm::enumerate(inDims)) {
      auto basesIt = allBases.find(inDim);
      if (basesIt == allBases.end())
        continue;
      for (auto [basisIdx, basis] : llvm::enumerate(basesIt->second)) {
        bool exact = true;
        for (auto [outIdx, value] : llvm::enumerate(basis)) {
          int32_t expected = outIdx == logicalIdx ? bit : 0;
          if (value != expected) {
            exact = false;
            break;
          }
        }
        if (!exact)
          continue;
        if (match)
          return std::nullopt;
        match = {std::pair<unsigned, unsigned>{inIdx, basisIdx}};
      }
    }
    return match;
  };

  for (unsigned dim = 0; dim < prefixRank; ++dim) {
    int32_t remainingOffset = offsets[dim];
    for (int64_t bit64 = 1; bit64 < srcShape[dim]; bit64 <<= 1) {
      auto match = findLogicalBasis(dim, static_cast<int32_t>(bit64));
      if (!match) {
        if (error)
          *error = "unsupported tensor memory leading-unit memdesc_subslice view";
        return std::nullopt;
      }
      auto [inIdx, basisIdx] = *match;
      recordErase(inDims[inIdx], basisIdx);
      if (remainingOffset & static_cast<int32_t>(bit64)) {
        origin[inIdx] += static_cast<int32_t>(1u << basisIdx);
        remainingOffset &= ~static_cast<int32_t>(bit64);
      }
    }
    if (remainingOffset != 0) {
      if (error)
        *error = "unsupported tensor memory leading-unit memdesc_subslice view";
      return std::nullopt;
    }
  }

  auto viewLayout = srcLayout;
  for (unsigned dim = 0; dim < prefixRank; ++dim)
    viewLayout = viewLayout.resizeOutDim(logicalDims[dim], 1);
  for (unsigned dim = 0; dim < prefixRank; ++dim)
    viewLayout = viewLayout.squeezeOuts(logicalDims[dim]);

  auto bases = viewLayout.getBases();
  for (auto &[inDim, basisIndices] : eraseByDim) {
    auto basesIt = bases.find(inDim);
    if (basesIt == bases.end()) {
      if (error)
        *error = "unsupported tensor memory leading-unit memdesc_subslice view";
      return std::nullopt;
    }
    llvm::sort(basisIndices);
    basisIndices.erase(std::unique(basisIndices.begin(), basisIndices.end()),
                       basisIndices.end());
    for (unsigned basisIdx : basisIndices) {
      if (basisIdx >= basesIt->second.size() ||
          !llvm::all_of(basesIt->second[basisIdx],
                        [](int32_t value) { return value == 0; })) {
        if (error)
          *error = "unsupported tensor memory leading-unit memdesc_subslice view";
        return std::nullopt;
      }
    }
    for (unsigned idx = basisIndices.size(); idx > 0; --idx) {
      unsigned basisIdx = basisIndices[idx - 1];
      if (basisIdx + 1 != basesIt->second.size()) {
        if (error)
          *error = "unsupported tensor memory leading-unit memdesc_subslice view";
        return std::nullopt;
      }
      basesIt->second.erase(basesIt->second.begin() + basisIdx);
    }
  }
  viewLayout = LinearLayout(std::move(bases), viewLayout.getOutDims(),
                            viewLayout.isSurjective());
  return LeadingUnitSubviewLayout{std::move(viewLayout), std::move(origin)};
}

static FailureOr<TMemLdStQueryLayout>
inferTMemSubsliceQueryLayout(ArrayRef<int64_t> srcShape,
                             const TMemLdStQueryLayout &srcQuery,
                             ArrayRef<int64_t> dstShape,
                             ArrayRef<int32_t> offsets, int bitwidth,
                             MLIRContext *ctx,
                             std::string *error) {
  auto ll = srcQuery.layout;
  auto layoutRank = ll.getNumOutDims();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank < 0 || dstShape.size() != srcShape.size() ||
      offsets.size() != srcShape.size()) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return failure();
  }

  if (srcShape.drop_front(extraRank) == dstShape.drop_front(extraRank) &&
      llvm::all_of(offsets.drop_front(extraRank),
                   [](int32_t offset) { return offset == 0; })) {
    return TMemLdStQueryLayout{
        srcQuery.layout, srcQuery.twoCTAs,
        addPrefixOffsetsToQueryOrigin(srcQuery.layout, srcQuery.origin,
                                      srcShape.take_front(extraRank),
                                      offsets.take_front(extraRank), bitwidth)};
  }

  if (auto leadingUnit = tryMakeLeadingUnitSubviewLayout(
          ll, srcShape.drop_front(extraRank), dstShape.drop_front(extraRank),
          offsets.drop_front(extraRank), srcQuery.origin, error)) {
    return TMemLdStQueryLayout{
        leadingUnit->layout, srcQuery.twoCTAs,
        addPrefixOffsetsToQueryOrigin(leadingUnit->layout, leadingUnit->origin,
                                      srcShape.take_front(extraRank),
                                      offsets.take_front(extraRank), bitwidth)};
  }

  // Pure 2D column subviews should keep the same local row support and simply
  // narrow the materialized logical column span. Rebuilding them through the
  // generic inverse/projection path preserves raw zero-row support bits from
  // the source image and loses the exact direct-load layout for simple TMEM
  // column slices (for example 64x128 -> 64x32 accumulator epilogues).
  if (extraRank == 0 && layoutRank == 2 && dstShape[0] == srcShape[0] &&
      offsets[0] == 0 && dstShape[1] <= srcShape[1] && offsets[1] >= 0 &&
      offsets[1] + dstShape[1] <= srcShape[1]) {
    auto *layoutCtx = ll.getOutDimNames().begin()->getContext();
    auto kCol = StringAttr::get(layoutCtx, "col");
    auto logicalDims = llvm::to_vector(ll.getOutDimNames());
    if (logicalDims.size() != 2 || !ll.hasInDim(kCol) ||
        ll.getInDimSize(kCol) < dstShape[1] ||
        ll.getOutDimSize(logicalDims[1]) < dstShape[1]) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
    auto narrowedLayout = ll;
    if (narrowedLayout.getOutDimSize(logicalDims[1]) != dstShape[1])
      narrowedLayout =
          narrowedLayout.resizeOutDim(logicalDims[1], dstShape[1]);
    if (narrowedLayout.getInDimSize(kCol) != dstShape[1])
      narrowedLayout = narrowedLayout.resizeInDim(kCol, dstShape[1]);
    return TMemLdStQueryLayout{
        narrowedLayout, srcQuery.twoCTAs,
        remapTMemLdStQueryOrigin(
            srcQuery, narrowedLayout,
            {{kCol, static_cast<int32_t>(offsets[1])}})};
  }

  auto logicalDims = llvm::to_vector(ll.getOutDimNames());
  SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
  encodedOffsets.reserve(layoutRank);
  for (auto [dim, offset] : llvm::enumerate(offsets.drop_front(extraRank))) {
    if (offset < 0) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
    encodedOffsets.push_back(
        {logicalDims[dim], static_cast<int32_t>(offset)});
  }

  auto llInv = computeLeftInverseLayout(ll, error);
  TMemLdStQueryLayout workingQuery = srcQuery;
  if (failed(llInv)) {
    auto normalized = normalizeTensorMemoryLinearLayoutForAnalysis(ll);
    std::string normalizedError;
    auto normalizedInv = computeLeftInverseLayout(normalized, &normalizedError);
    if (succeeded(normalizedInv)) {
      workingQuery = TMemLdStQueryLayout{
          normalized, srcQuery.twoCTAs,
          remapTMemLdStQueryOrigin(srcQuery, normalized, /*deltaCoords=*/{})};
      ll = normalized;
      logicalDims = llvm::to_vector(ll.getOutDimNames());
      encodedOffsets.clear();
      encodedOffsets.reserve(layoutRank);
      for (auto [dim, offset] : llvm::enumerate(offsets.drop_front(extraRank)))
        encodedOffsets.push_back(
            {logicalDims[dim], static_cast<int32_t>(offset)});
      llInv = std::move(normalizedInv);
    } else {
      if (error && error->empty())
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
  }
  auto baseCoords =
      llInv->apply(makeFullLinearLayoutCoords(logicalDims, encodedOffsets));
  auto physOutDimNames = llvm::to_vector(llInv->getOutDimNames());

  LinearLayout::BasesT dstInvBases;
  for (int64_t dim = extraRank; dim < static_cast<int64_t>(srcShape.size());
       ++dim) {
    int64_t dstDimSize = dstShape[dim];
    int64_t srcDimSize = srcShape[dim];
    if (dstDimSize > srcDimSize || offsets[dim] + dstDimSize > srcDimSize) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }

    auto dstDimName =
        StringAttr::get(ctx, "dim" + llvm::Twine(dim - extraRank));
    auto &bases = dstInvBases[dstDimName];
    for (int64_t step = 1; step < dstDimSize; step <<= 1) {
      auto point = encodedOffsets;
      point[dim - extraRank].second += static_cast<int32_t>(step);
      auto pointCoords =
          llInv->apply(makeFullLinearLayoutCoords(logicalDims, point));
      std::vector<int32_t> basis;
      basis.reserve(physOutDimNames.size());
      for (auto physDim : physOutDimNames) {
        int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                        lookupLinearLayoutCoord(baseCoords, physDim);
        if (delta < 0) {
          if (error)
            *error = "unsupported tensor memory memdesc_subslice view";
          return failure();
        }
        basis.push_back(delta);
      }
      bases.push_back(std::move(basis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> activePhysOutDims;
  activePhysOutDims.reserve(physOutDimNames.size());
  for (auto physDim : physOutDimNames) {
    activePhysOutDims.push_back({physDim, llInv->getOutDimSize(physDim)});
  }

  auto dstInv = LinearLayout::tryCreate(std::move(dstInvBases), activePhysOutDims,
                                        /*requireSurjective=*/false, error);
  if (!dstInv) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view: failed to "
               "construct support inverse";
    return failure();
  }
  auto dstLayout = computeLeftInverseLayout(*dstInv, error);
  if (failed(dstLayout)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view: failed to "
               "compute support layout";
    return failure();
  }
  if (failed(verifyTMemSubsliceProjection(
          *llInv, logicalDims, encodedOffsets, baseCoords, *dstLayout,
          dstShape.drop_front(extraRank), error))) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view: projection "
               "mismatch";
    return failure();
  }
  auto dstOrigin =
      remapTMemLdStQueryOrigin(workingQuery, *dstLayout, baseCoords);
  dstOrigin = addPrefixOffsetsToQueryOrigin(
      *dstLayout, dstOrigin, srcShape.take_front(extraRank),
      offsets.take_front(extraRank), bitwidth);
  return TMemLdStQueryLayout{*dstLayout, workingQuery.twoCTAs, dstOrigin};
}

static FailureOr<TMemLdStQueryLayout>
inferTMemIndexQueryLayout(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                          const TMemLdStQueryLayout &srcQuery,
                          int bitwidth, std::optional<int32_t> leadingIndex,
                          MLIRContext *ctx, std::string *error) {
  auto ll = srcQuery.layout;
  while (ll.getNumOutDims() > 0) {
    auto firstDim = *ll.getOutDimNames().begin();
    if (ll.getOutDimSize(firstDim) != 1)
      break;
    ll = ll.squeezeOuts(firstDim);
  }
  auto layoutRank = ll.getNumOutDims();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank < 0) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return failure();
  }

  if (extraRank > 0) {
    // Leading multibuffer indices are lowered by MemDescIndexOpConversion into
    // the TMEM base value itself. The reg-layout/support query must therefore
    // preserve only the row/col translation already tracked in srcQuery.origin
    // and must not try to re-encode the selected buffer as an additional query
    // origin offset. Otherwise dynamic multibuffer views lose their raw query
    // path and later tmem_subslice/tmem_load chains collapse distinct column
    // windows onto the same tcgen05.ld base address.
    (void)leadingIndex;
    (void)bitwidth;
    return TMemLdStQueryLayout{
        ll, srcQuery.twoCTAs,
        remapTMemLdStQueryOrigin(srcQuery, ll, /*deltaCoords=*/{})};
  }

  if (layoutRank == 0) {
    if (error)
      *error = "tensor memory layout rank must be greater than zero";
    return failure();
  }
  auto maybeSrcInv = computeLeftInverseLayout(ll, error);
  if (failed(maybeSrcInv)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }

  auto srcLogicalDims = llvm::to_vector(ll.getOutDimNames());
  auto baseCoords =
      maybeSrcInv->apply(makeFullLinearLayoutCoords(srcLogicalDims, {}));
  auto physOutDimNames = llvm::to_vector(maybeSrcInv->getOutDimNames());

  LinearLayout::BasesT dstInvBases;
  for (int64_t dim = 1; dim < static_cast<int64_t>(srcShape.size()); ++dim) {
    int64_t dstDimSize = dstShape[dim - 1];
    int64_t srcDimSize = srcShape[dim];
    if (dstDimSize != srcDimSize) {
      if (error)
        *error = "unsupported tensor memory memdesc_index view";
      return failure();
    }

    auto dstDimName = StringAttr::get(ctx, "dim" + llvm::Twine(dim - 1));
    auto &bases = dstInvBases[dstDimName];
    for (int64_t step = 1; step < dstDimSize; step <<= 1) {
      SmallVector<std::pair<StringAttr, int32_t>> srcPoint;
      srcPoint.reserve(srcLogicalDims.size());
      for (auto logicalDim : srcLogicalDims) {
        srcPoint.push_back(
            {logicalDim,
             logicalDim == srcLogicalDims[dim] ? static_cast<int32_t>(step)
                                               : 0});
      }
      auto pointCoords = maybeSrcInv->apply(
          makeFullLinearLayoutCoords(srcLogicalDims, srcPoint));
      std::vector<int32_t> basis;
      basis.reserve(physOutDimNames.size());
      for (auto physDim : physOutDimNames) {
        int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                        lookupLinearLayoutCoord(baseCoords, physDim);
        if (delta < 0) {
          if (error)
            *error = "unsupported tensor memory memdesc_index view";
          return failure();
        }
        basis.push_back(delta);
      }
      bases.push_back(std::move(basis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> activePhysOutDims;
  activePhysOutDims.reserve(physOutDimNames.size());
  for (auto physDim : physOutDimNames) {
    activePhysOutDims.push_back(
        {physDim, maybeSrcInv->getOutDimSize(physDim)});
  }

  auto dstInv = LinearLayout::tryCreate(std::move(dstInvBases), activePhysOutDims,
                                        /*requireSurjective=*/false, error);
  if (!dstInv) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view: failed to "
               "construct support inverse";
    return failure();
  }
  auto dstLayout = computeLeftInverseLayout(*dstInv, error);
  if (failed(dstLayout)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view: failed to "
               "compute support layout";
    return failure();
  }

  if (failed(verifyTMemIndexProjection(*maybeSrcInv, srcLogicalDims,
                                       *dstLayout, dstShape, error))) {
    if (error && error->empty())
      *error =
          "unsupported tensor memory memdesc_index view: projection mismatch";
    return failure();
  }
  return TMemLdStQueryLayout{*dstLayout, srcQuery.twoCTAs,
                             remapTMemLdStQueryOrigin(srcQuery, *dstLayout,
                                                      baseCoords)};
}

static FailureOr<TMemLdStQueryLayout>
inferTMemReshapeQueryLayout(ArrayRef<int64_t> srcShape,
                            const TMemLdStQueryLayout &srcQuery,
                            ArrayRef<int64_t> dstShape, MLIRContext *ctx,
                            std::string *error) {
  auto ll = srcQuery.layout;
  auto layoutSrcShape = srcShape;
  auto layoutDstShape = dstShape;
  int64_t layoutElems = static_cast<int64_t>(ll.getTotalOutDimSize());

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape) {
    while (!shape.empty() && shape.front() == 1 &&
           product<int64_t>(shape.drop_front()) >= layoutElems)
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape);
  layoutDstShape = stripLeadingUnitDims(layoutDstShape);
  while (static_cast<size_t>(ll.getNumOutDims()) > layoutSrcShape.size()) {
    auto firstDim = *ll.getOutDimNames().begin();
    if (ll.getOutDimSize(firstDim) != 1)
      break;
    ll = ll.squeezeOuts(firstDim);
  }
  layoutElems = static_cast<int64_t>(ll.getTotalOutDimSize());

  if (product<int64_t>(layoutSrcShape) > layoutElems) {
    if (layoutSrcShape.size() != static_cast<size_t>(ll.getNumOutDims()) + 1) {
      if (error)
        *error = "TMEM reshape requires the descriptor rank to match the TMEM "
                 "layout rank or have one leading multibuffer dimension";
      return failure();
    }
    if (layoutDstShape.empty() || layoutDstShape.front() != layoutSrcShape.front()) {
      if (error)
        *error = "TMEM reshape must preserve the leading multibuffer dimension";
      return failure();
    }
    layoutSrcShape = layoutSrcShape.drop_front();
    layoutDstShape = layoutDstShape.drop_front();
  }

  if (product(layoutSrcShape) != product(layoutDstShape)) {
    if (error)
      *error = "dst shape has different number of elements than src";
    return failure();
  }
  if (!llvm::equal(ll.getOutDimSizes(), layoutSrcShape)) {
    auto maybeRestrictedLayout =
        restrictTMemAnalysisLayoutToShape(ll, layoutSrcShape, error);
    if (failed(maybeRestrictedLayout)) {
      if (error && error->empty())
        *error = "TMEM reshape rank does not match the canonical TMEM layout";
      return failure();
    }
    ll = *maybeRestrictedLayout;
    layoutElems = static_cast<int64_t>(ll.getTotalOutDimSize());
  }
  if (layoutElems != product<int64_t>(layoutSrcShape)) {
    if (error)
      *error = "TMEM reshape rank does not match the canonical TMEM layout";
    return failure();
  }

  return TMemLdStQueryLayout{reshapeLayout(ctx, ll, layoutDstShape),
                             srcQuery.twoCTAs, srcQuery.origin};
}

static FailureOr<TMemLdStQueryLayout>
inferTMemReinterpretQueryLayout(ArrayRef<int64_t> srcShape, int srcBitwidth,
                                const TMemLdStQueryLayout &srcQuery,
                                ArrayRef<int64_t> dstShape, int dstBitwidth,
                                MLIRContext *ctx, std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  auto ll = srcQuery.layout;
  auto layoutSrcShape = srcShape;
  auto layoutDstShape = dstShape;
  int64_t layoutElems = static_cast<int64_t>(ll.getTotalOutDimSize());

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape) {
    while (!shape.empty() && shape.front() == 1 &&
           product<int64_t>(shape.drop_front()) >= layoutElems)
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape);
  layoutDstShape = stripLeadingUnitDims(layoutDstShape);
  while (static_cast<size_t>(ll.getNumOutDims()) > layoutSrcShape.size()) {
    auto firstDim = *ll.getOutDimNames().begin();
    if (ll.getOutDimSize(firstDim) != 1)
      break;
    ll = ll.squeezeOuts(firstDim);
  }
  layoutElems = static_cast<int64_t>(ll.getTotalOutDimSize());

  if (product<int64_t>(layoutSrcShape) > layoutElems) {
    if (layoutSrcShape.size() != static_cast<size_t>(ll.getNumOutDims()) + 1) {
      if (error)
        *error = "TMEM reinterpret requires the descriptor rank to match the "
                 "TMEM layout rank or have one leading multibuffer dimension";
      return failure();
    }
    if (layoutDstShape.empty() ||
        layoutDstShape.front() != layoutSrcShape.front()) {
      if (error)
        *error =
            "TMEM reinterpret must preserve the leading multibuffer dimension";
      return failure();
    }
    layoutSrcShape = layoutSrcShape.drop_front();
    layoutDstShape = layoutDstShape.drop_front();
  }

  if (product<int64_t>(layoutSrcShape) * srcBitwidth !=
      product<int64_t>(layoutDstShape) * dstBitwidth) {
    if (error)
      *error = "TMEM reinterpret must preserve the total number of bits";
    return failure();
  }
  if (!llvm::equal(ll.getOutDimSizes(), layoutSrcShape)) {
    auto maybeRestrictedLayout =
        restrictTMemAnalysisLayoutToShape(ll, layoutSrcShape, error);
    if (failed(maybeRestrictedLayout)) {
      if (error && error->empty())
        *error =
            "TMEM reinterpret rank does not match the canonical TMEM layout";
      return failure();
    }
    ll = *maybeRestrictedLayout;
    layoutElems = static_cast<int64_t>(ll.getTotalOutDimSize());
  }
  if (layoutElems != product<int64_t>(layoutSrcShape)) {
    if (error)
      *error = "TMEM reinterpret rank does not match the canonical TMEM "
               "layout";
    return failure();
  }

  TMemLdStQueryLayout workingQuery = srcQuery;
  auto maybeSrcInv = computeLeftInverseLayout(ll, error);
  if (failed(maybeSrcInv)) {
    // Descriptor subviews can retain inactive zero support bases. A physical
    // bitcast still preserves the same physical image, so normalize only when
    // the normalized view is injective and still exactly covers srcShape.
    auto normalized = normalizeTensorMemoryLinearLayoutForAnalysis(ll);
    std::string normalizedError;
    auto normalizedInv = computeLeftInverseLayout(normalized, &normalizedError);
    if (succeeded(normalizedInv) &&
        normalized.getNumOutDims() == ll.getNumOutDims() &&
        llvm::equal(normalized.getOutDimSizes(), layoutSrcShape) &&
        static_cast<int64_t>(normalized.getTotalInDimSize()) ==
            product<int64_t>(layoutSrcShape)) {
      workingQuery = TMemLdStQueryLayout{
          normalized, srcQuery.twoCTAs,
          remapTMemLdStQueryOrigin(srcQuery, normalized,
                                   /*deltaCoords=*/{})};
      ll = normalized;
      maybeSrcInv = std::move(normalizedInv);
      if (error)
        error->clear();
    } else {
      if (error && error->empty())
        *error = "unsupported tensor memory memdesc_reinterpret view";
      if (debug) {
        llvm::errs() << "[tmem-ldst] reinterpret normalized src inverse fail "
                        "layout:\n"
                     << normalized.toString() << "\nerr=" << normalizedError
                     << "\n";
      }
      return failure();
    }
  }

  auto linearizeRowMajorCoordsLocal = [&](ArrayRef<int64_t> shape,
                                          ArrayRef<int32_t> coords)
      -> FailureOr<int64_t> {
    if (shape.size() != coords.size()) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return failure();
    }
    int64_t linearOffset = 0;
    int64_t stride = 1;
    for (auto [size, coord] : llvm::reverse(llvm::zip_equal(shape, coords))) {
      if (coord < 0 || coord >= size) {
        if (error)
          *error = "unsupported tensor memory memdesc_reinterpret view";
        return failure();
      }
      linearOffset += static_cast<int64_t>(coord) * stride;
      stride *= size;
    }
    return linearOffset;
  };
  auto unravelRowMajorCoordsLocal = [&](ArrayRef<int64_t> shape,
                                        int64_t linearOffset)
      -> FailureOr<SmallVector<int32_t>> {
    if (linearOffset < 0 || linearOffset >= product<int64_t>(shape)) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return failure();
    }
    SmallVector<int32_t> coords(shape.size(), 0);
    for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0;
         --dim) {
      int64_t size = shape[dim];
      if (size <= 0) {
        if (error)
          *error = "unsupported tensor memory memdesc_reinterpret view";
        return failure();
      }
      coords[dim] = static_cast<int32_t>(linearOffset % size);
      linearOffset /= size;
    }
    return coords;
  };
  bool usesSubElementDst = false;
  auto mapPoint = [&](ArrayRef<int32_t> dstPoint)
      -> FailureOr<SmallVector<int32_t>> {
    auto linearDst = linearizeRowMajorCoordsLocal(layoutDstShape, dstPoint);
    if (failed(linearDst))
      return failure();
    int64_t srcBitOffset = *linearDst * static_cast<int64_t>(dstBitwidth);
    if (srcBitOffset % srcBitwidth != 0) {
      if (dstBitwidth >= srcBitwidth || srcBitwidth % dstBitwidth != 0) {
        if (error)
          *error = "unsupported tensor memory memdesc_reinterpret view";
        return failure();
      }
      usesSubElementDst = true;
    }
    int64_t linearSrc = srcBitOffset / srcBitwidth;
    return unravelRowMajorCoordsLocal(layoutSrcShape, linearSrc);
  };
  auto remapOriginPoint = [&](ArrayRef<int32_t> srcPoint)
      -> FailureOr<SmallVector<int32_t>> {
    auto linearSrc = linearizeRowMajorCoordsLocal(layoutSrcShape, srcPoint);
    if (failed(linearSrc))
      return failure();
    int64_t dstBitOffset = *linearSrc * static_cast<int64_t>(srcBitwidth);
    if (dstBitOffset % dstBitwidth != 0) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return failure();
    }
    int64_t linearDst = dstBitOffset / dstBitwidth;
    return unravelRowMajorCoordsLocal(layoutDstShape, linearDst);
  };

  auto srcLogicalDims = llvm::to_vector(ll.getOutDimNames());
  auto makeLogicalCoords = [&](ArrayRef<int32_t> point) {
    SmallVector<std::pair<StringAttr, int32_t>> sparse;
    sparse.reserve(srcLogicalDims.size());
    for (auto [dim, value] : llvm::zip_equal(srcLogicalDims, point))
      sparse.push_back({dim, value});
    return makeFullLinearLayoutCoords(srcLogicalDims, sparse);
  };
  SmallVector<int32_t> zeroPoint(layoutDstShape.size(), 0);
  auto basePoint = mapPoint(zeroPoint);
  if (failed(basePoint))
    return failure();
  auto baseCoords = maybeSrcInv->apply(makeLogicalCoords(*basePoint));
  auto physOutDimNames = llvm::to_vector(maybeSrcInv->getOutDimNames());

  LinearLayout::BasesT dstInvBases;
  for (int64_t dim = 0; dim < static_cast<int64_t>(layoutDstShape.size());
       ++dim) {
    auto dstDimName = StringAttr::get(ctx, "dim" + llvm::Twine(dim));
    auto &bases = dstInvBases[dstDimName];
    for (int64_t step = 1; step < layoutDstShape[dim]; step <<= 1) {
      SmallVector<int32_t> dstPoint(layoutDstShape.size(), 0);
      dstPoint[dim] = static_cast<int32_t>(step);
      auto srcPoint = mapPoint(dstPoint);
      if (failed(srcPoint))
        return failure();
      auto pointCoords = maybeSrcInv->apply(makeLogicalCoords(*srcPoint));
      std::vector<int32_t> basis;
      basis.reserve(physOutDimNames.size());
      for (auto physDim : physOutDimNames) {
        int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                        lookupLinearLayoutCoord(baseCoords, physDim);
        if (delta < 0) {
          if (error)
            *error = "unsupported tensor memory memdesc_reinterpret view";
          return failure();
        }
        basis.push_back(delta);
      }
      bases.push_back(std::move(basis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> activePhysOutDims;
  activePhysOutDims.reserve(physOutDimNames.size());
  for (auto physDim : physOutDimNames)
    activePhysOutDims.push_back({physDim, maybeSrcInv->getOutDimSize(physDim)});

  auto dstInv = LinearLayout::tryCreate(std::move(dstInvBases),
                                        activePhysOutDims,
                                        /*requireSurjective=*/false, error);
  if (!dstInv) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_reinterpret view: failed to "
               "construct support inverse";
    return failure();
  }
  auto dstLayout = computeLeftInverseLayout(*dstInv, error);
  if (failed(dstLayout)) {
    if (usesSubElementDst) {
      if (error)
        error->clear();
      dstLayout = dstInv->pseudoinvert();
    }
  }
  if (failed(dstLayout)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_reinterpret view: failed to "
               "compute support layout";
    return failure();
  }
  FailureOr<SmallVector<int32_t>> maybeDstOrigin = failure();
  if (!isTMemLdStQueryOriginRepresentable(workingQuery, ll)) {
    maybeDstOrigin = remapTMemPhysicalOriginForBitcast(
        workingQuery, *dstLayout, srcBitwidth, dstBitwidth, error);
  } else {
    SmallVector<std::pair<StringAttr, int32_t>> srcOriginSparse;
    srcOriginSparse.reserve(ll.getNumInDims());
    auto srcInDims = llvm::to_vector(ll.getInDimNames());
    for (auto [dim, value] : llvm::zip_equal(srcInDims, workingQuery.origin))
      srcOriginSparse.push_back({dim, value});
    auto srcOriginCoords =
        ll.apply(makeFullLinearLayoutCoords(srcInDims, srcOriginSparse));
    SmallVector<int32_t> srcOriginPoint;
    srcOriginPoint.reserve(ll.getNumOutDims());
    for (auto dim : ll.getOutDimNames())
      srcOriginPoint.push_back(lookupLinearLayoutCoord(srcOriginCoords, dim));
    auto dstOriginPoint = remapOriginPoint(srcOriginPoint);
    if (failed(dstOriginPoint))
      return failure();
    auto dstLogicalDims = llvm::to_vector(dstLayout->getOutDimNames());
    SmallVector<std::pair<StringAttr, int32_t>> dstOriginSparse;
    dstOriginSparse.reserve(dstLogicalDims.size());
    for (auto [dim, value] : llvm::zip_equal(dstLogicalDims, *dstOriginPoint))
      dstOriginSparse.push_back({dim, value});
    auto dstOriginCoords = (*dstInv).apply(
        makeFullLinearLayoutCoords(dstLogicalDims, dstOriginSparse));
    SmallVector<int32_t> dstOrigin;
    dstOrigin.reserve(dstLayout->getNumInDims());
    for (auto dim : dstLayout->getInDimNames())
      dstOrigin.push_back(lookupLinearLayoutCoord(dstOriginCoords, dim));
    maybeDstOrigin = std::move(dstOrigin);
  }
  if (failed(maybeDstOrigin))
    return failure();
  auto result = TMemLdStQueryLayout{*dstLayout, workingQuery.twoCTAs,
                                    std::move(*maybeDstOrigin)};
  if (debug) {
    llvm::errs() << "[tmem-ldst] reinterpret srcShape=";
    for (int64_t size : srcShape)
      llvm::errs() << " " << size;
    llvm::errs() << " srcBitwidth=" << srcBitwidth << " dstShape=";
    for (int64_t size : dstShape)
      llvm::errs() << " " << size;
    llvm::errs() << " dstBitwidth=" << dstBitwidth << "\n";
    llvm::errs() << "[tmem-ldst] reinterpret src layout:\n"
                 << srcQuery.layout.toString() << "\n";
    llvm::errs() << "[tmem-ldst] reinterpret dst layout:\n"
                 << result.layout.toString() << "\n";
    llvm::errs() << "[tmem-ldst] reinterpret origin:";
    for (int32_t value : result.origin)
      llvm::errs() << " " << value;
    llvm::errs() << "\n";
  }
  return result;
}

static FailureOr<LinearLayout>
reinterpretLinearLayout(ArrayRef<int64_t> srcShape, int srcBitwidth,
                        const LinearLayout &srcLayout,
                        ArrayRef<int64_t> dstShape, int dstBitwidth,
                        MLIRContext *ctx, std::string *error) {
  SmallVector<int32_t> zeroOrigin(srcLayout.getNumInDims(), 0);
  TMemLdStQueryLayout srcQuery{srcLayout, /*twoCTAs=*/false,
                               std::move(zeroOrigin)};
  auto result = inferTMemReinterpretQueryLayout(srcShape, srcBitwidth, srcQuery,
                                                dstShape, dstBitwidth, ctx,
                                                error);
  if (failed(result))
    return failure();
  return result->layout;
}

static bool hasFullShapeTMemTile(MemDescType memTy) {
  auto encoding = memTy.getEncoding();
  if (!isTensorMemoryEncoding(encoding) ||
      isa<TensorMemoryScalesEncodingAttr>(encoding))
    return false;
  auto rank = static_cast<size_t>(cast<LayoutEncodingTrait>(encoding).getRank());
  return memTy.getShape().size() >= rank &&
         memTy.getAllocShape().size() >= rank &&
         memTy.getShape().take_back(rank) ==
             memTy.getAllocShape().take_back(rank);
}

static std::optional<TMemLdStRowPlan>
getFullShapeM64TMemRowPlan(MemDescType memTy) {
  if (!hasFullShapeTMemTile(memTy) || memTy.getRank() < 2 ||
      memTy.getShape()[memTy.getRank() - 2] != 64)
    return std::nullopt;
  // Full-shape M64 tensor-memory descriptors use a 128-row backing tile
  // contract even when the logical descriptor type is a 64-row tile. This is a
  // property of the descriptor type/layout family, so expose it through the
  // normal row-plan path instead of producer provenance attrs.
  auto kBlock = StringAttr::get(memTy.getContext(), "block");
  auto memLayout = toLinearLayout(memTy);
  if (memLayout.hasInDim(kBlock) && memLayout.getInDimSize(kBlock) > 1) {
    return TMemLdStRowPlan{/*warpRow0=*/16, /*warpRow1=*/32,
                           /*rowSpan=*/128};
  }
  return TMemLdStRowPlan{/*warpRow0=*/32, /*warpRow1=*/64,
                         /*rowSpan=*/128};
}

static std::optional<TMemLdStQueryLayout>
getFullShapeMMAv5FamilyQueryLayout(MemDescType memTy) {
  if (!hasFullShapeTMemTile(memTy))
    return std::nullopt;

  auto makeQuery = [](const LinearLayout &layout, bool twoCTAs) {
    return TMemLdStQueryLayout{
        layout, twoCTAs, SmallVector<int32_t>(layout.getNumInDims(), 0)};
  };

  auto encoding = memTy.getEncoding();
  if (auto rootPlan = getFullShapeM64TMemRowPlan(memTy)) {
    auto maybeTwoCTAs = getTensorMemoryTwoCTAs(encoding);
    if (!maybeTwoCTAs)
      return std::nullopt;
    auto layoutRank = static_cast<size_t>(
        cast<LayoutEncodingTrait>(encoding).getRank());
    SmallVector<int64_t> allocShape(memTy.getAllocShape().begin(),
                                    memTy.getAllocShape().end());
    if (allocShape.size() < layoutRank)
      allocShape.assign(memTy.getShape().begin(), memTy.getShape().end());
    allocShape[allocShape.size() - layoutRank] =
        std::max<int64_t>(allocShape[allocShape.size() - layoutRank],
                          rootPlan->rowSpan);
    auto widenedTy = MemDescType::get(memTy.getShape(), memTy.getElementType(),
                                      encoding, memTy.getMemorySpace(),
                                      memTy.getMutableMemory(), allocShape);
    return makeQuery(toLinearLayout(widenedTy), *maybeTwoCTAs);
  }

  if (auto info = getMMAv5AccumulatorLayoutInfo(memTy))
    return makeQuery(info->familyLayout, info->twoCTAs);
  if (auto info = getMMAv5ScaledAccumulatorLayoutInfo(memTy))
    return makeQuery(info->familyLayout, info->twoCTAs);
  if (auto info = getMMAv5LhsLayoutInfo(memTy))
    return makeQuery(info->familyLayout, info->twoCTAs);
  return std::nullopt;
}

static FailureOr<TMemLdStQueryLayout>
inferStandaloneTMemLdStQueryLayoutImpl(Value memDesc,
                                       bool preserveNonCanonicalView,
                                       std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  auto memDescTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!memDescTy || memDescTy.getMemorySpace() !=
                        TensorMemorySpaceAttr::get(memDesc.getContext())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }

  auto encoding = memDescTy.getEncoding();
  if (!isTensorMemoryEncoding(encoding)) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  if (isa<TensorMemoryScalesEncodingAttr>(encoding)) {
    auto ll = toLinearLayout(memDescTy);
    auto scales = cast<TensorMemoryScalesEncodingAttr>(encoding);
    bool twoCTAs = product<unsigned>(scales.getCGALayout().getCTAsPerCGA()) > 1;
    return TMemLdStQueryLayout{
        ll, twoCTAs,
        SmallVector<int32_t>(ll.getNumInDims(), 0)};
  }

  Operation *defOp = memDesc.getDefiningOp();
  bool hasDescriptorViewProducer =
      isa_and_nonnull<gpu::MemDescSubsliceOp, TMEMSubSliceOp, gpu::MemDescIndexOp,
                      gpu::MemDescReshapeOp>(defOp);
  if (!hasDescriptorViewProducer) {
    if (auto familyQuery = getFullShapeMMAv5FamilyQueryLayout(memDescTy))
      return *familyQuery;
  }
  auto maybeAnalysis = hasDescriptorViewProducer
                           ? std::optional<TMemLdStQueryLayout>{}
                           : getTMemViewAnalysisLayout(memDescTy.getShape(),
                                                      encoding, error);
  if (!hasDescriptorViewProducer && !maybeAnalysis)
    return failure();

  auto layoutRank =
      static_cast<size_t>(cast<LayoutEncodingTrait>(encoding).getRank());
  if (preserveNonCanonicalView && memDescTy.getShape().size() >= layoutRank &&
      memDescTy.getAllocShape().size() >= layoutRank &&
      memDescTy.getShape().take_back(layoutRank) !=
          memDescTy.getAllocShape().take_back(layoutRank) &&
      !getCanonicalTMemLinearEncoding(memDescTy, /*error=*/nullptr) &&
      !hasDescriptorViewProducer) {
    return *maybeAnalysis;
  }

  if (preserveNonCanonicalView && !hasDescriptorViewProducer &&
      isa<TensorMemoryLinearEncodingAttr>(encoding)) {
    auto rawLayout = toLinearLayout(memDescTy);
    if (rawLayout != maybeAnalysis->layout) {
      auto maybeTwoCTAs = getTensorMemoryTwoCTAs(encoding);
      if (!maybeTwoCTAs) {
        if (error)
          *error = "expected tensor memory layout encoding";
        return failure();
      }
      return TMemLdStQueryLayout{
          rawLayout, *maybeTwoCTAs,
          SmallVector<int32_t>(rawLayout.getNumInDims(), 0)};
    }
  }

  auto *ctx = memDesc.getContext();
  if (auto subslice = memDesc.getDefiningOp<gpu::MemDescSubsliceOp>()) {
    auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        subslice.getSrc(), preserveNonCanonicalView, error);
    if (failed(srcQuery)) {
      if (debug && error && !error->empty()) {
        llvm::errs() << "[tmem-ldst] memdesc_subslice srcQuery fail src="
                     << cast<MemDescType>(subslice.getSrc().getType())
                     << " dst=" << memDescTy << " offsets=";
        for (auto off : subslice.getOffsets())
          llvm::errs() << " " << off;
        llvm::errs() << " err=" << *error << "\n";
      }
      return failure();
    }
    auto srcTy = cast<MemDescType>(subslice.getSrc().getType());
    auto tmemSpace = TensorMemorySpaceAttr::get(memDesc.getContext());
    bool isPureTMem2DColumnSubview =
        srcTy.getMemorySpace() == tmemSpace &&
        memDescTy.getMemorySpace() == tmemSpace && srcTy.getRank() == 2 &&
        memDescTy.getRank() == 2 && subslice.getOffsets().size() == 2 &&
        subslice.getOffsets()[0] == 0 &&
        srcTy.getShape()[0] == memDescTy.getShape()[0] &&
        srcTy.getShape()[1] >= memDescTy.getShape()[1];
    if (srcTy.getRank() == memDescTy.getRank() &&
        llvm::equal(srcTy.getShape(), memDescTy.getShape()) &&
        llvm::all_of(subslice.getOffsets(),
                     [](int64_t offset) { return offset == 0; })) {
      if (debug) {
        llvm::errs() << "[tmem-ldst] memdesc_subslice identity src=" << srcTy
                     << " dst=" << memDescTy << "\n";
      }
      return *srcQuery;
    }
    bool isSameRankTMemSubview =
        srcTy.getMemorySpace() == tmemSpace &&
        memDescTy.getMemorySpace() == tmemSpace &&
        srcTy.getRank() == memDescTy.getRank() &&
        subslice.getOffsets().size() == srcTy.getRank() &&
        static_cast<size_t>(srcQuery->layout.getNumOutDims()) ==
            subslice.getOffsets().size();
    if (!isPureTMem2DColumnSubview && isSameRankTMemSubview) {
      auto maybeDstTy =
          inferStandaloneTMemViewType(memDesc, error);
      if (debug && failed(maybeDstTy) && error && !error->empty()) {
        llvm::errs() << "[tmem-ldst] memdesc_subslice standalone dst type fail src="
                     << srcTy << " dst=" << memDescTy << " offsets=";
        for (auto off : subslice.getOffsets())
          llvm::errs() << " " << off;
        llvm::errs() << " err=" << *error << "\n";
      }
      if (succeeded(maybeDstTy)) {
        auto maybeDstAnalysis = getTMemViewAnalysisLayout(
            maybeDstTy->getShape(), maybeDstTy->getEncoding(), error);
        if (debug && !maybeDstAnalysis && error && !error->empty()) {
          llvm::errs() << "[tmem-ldst] memdesc_subslice dst analysis fail dstTy="
                       << *maybeDstTy << " err=" << *error << "\n";
        }
        if (!maybeDstAnalysis)
          return failure();
        auto projectedSrcQuery = *srcQuery;
        auto llInv = computeLeftInverseLayout(projectedSrcQuery.layout, error);
        if (failed(llInv)) {
          auto normalizedSrcLayout =
              normalizeTensorMemoryLinearLayoutForAnalysis(
                  projectedSrcQuery.layout);
          std::string normalizedError;
          auto normalizedInv =
              computeLeftInverseLayout(normalizedSrcLayout, &normalizedError);
          if (succeeded(normalizedInv)) {
            projectedSrcQuery = TMemLdStQueryLayout{
                normalizedSrcLayout, projectedSrcQuery.twoCTAs,
                remapTMemLdStQueryOrigin(*srcQuery, normalizedSrcLayout,
                                         /*deltaCoords=*/{})};
            llInv = std::move(normalizedInv);
          } else if (debug) {
            llvm::errs()
                << "[tmem-ldst] memdesc_subslice normalized src inverse fail layout:\n"
                << normalizedSrcLayout.toString() << "\nerr="
                << normalizedError << "\n";
          }
        }
        if (debug && failed(llInv) && error && !error->empty()) {
          llvm::errs() << "[tmem-ldst] memdesc_subslice src inverse fail layout:\n"
                       << projectedSrcQuery.layout.toString() << "\nerr="
                       << *error << "\n";
        }
        if (succeeded(llInv)) {
          auto logicalDims =
              llvm::to_vector(projectedSrcQuery.layout.getOutDimNames());
          SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
          encodedOffsets.reserve(logicalDims.size());
          for (auto [dim, logicalDim] : llvm::enumerate(logicalDims))
            encodedOffsets.push_back({logicalDim, subslice.getOffsets()[dim]});
          auto baseCoords = llInv->apply(
              makeFullLinearLayoutCoords(logicalDims, encodedOffsets));
          auto result = TMemLdStQueryLayout{
              maybeDstAnalysis->layout, maybeDstAnalysis->twoCTAs,
              remapTMemLdStQueryOrigin(projectedSrcQuery,
                                       maybeDstAnalysis->layout,
                                       baseCoords)};
          if (debug) {
            llvm::errs() << "[tmem-ldst] memdesc_subslice origin:";
            for (int32_t value : result.origin)
              llvm::errs() << " " << value;
            llvm::errs() << "\n";
          }
          return result;
        }
      }
    }
    return inferTMemSubsliceQueryLayout(srcTy.getShape(), *srcQuery,
                                        memDescTy.getShape(),
                                        subslice.getOffsets(),
                                        srcTy.getElementTypeBitWidth(), ctx,
                                        error);
  }
  if (auto subslice = memDesc.getDefiningOp<TMEMSubSliceOp>()) {
    auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        subslice.getSrc(), preserveNonCanonicalView, error);
    if (failed(srcQuery)) {
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] ttng.tmem_subslice srcQuery fail: "
                     << *error << "\n";
      return failure();
    }
    std::string layoutError;
    if (auto maybeDstAnalysis = getTMemViewAnalysisLayout(
            memDescTy.getShape(), memDescTy.getEncoding(), &layoutError)) {
      auto llInv = computeLeftInverseLayout(srcQuery->layout, &layoutError);
      if (succeeded(llInv)) {
        auto logicalDims = llvm::to_vector(srcQuery->layout.getOutDimNames());
        SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
        encodedOffsets.reserve(logicalDims.size());
        for (auto [dim, logicalDim] : llvm::enumerate(logicalDims))
          encodedOffsets.push_back({logicalDim,
                                    dim + 1 == logicalDims.size()
                                        ? subslice.getN()
                                        : 0});
        auto baseCoords = llInv->apply(
            makeFullLinearLayoutCoords(logicalDims, encodedOffsets));
        auto result = TMemLdStQueryLayout{
            maybeDstAnalysis->layout, maybeDstAnalysis->twoCTAs,
            remapTMemLdStQueryOrigin(*srcQuery, maybeDstAnalysis->layout,
                                     baseCoords)};
        if (debug) {
          llvm::errs() << "[tmem-ldst] ttng.tmem_subslice origin:";
          for (int32_t value : result.origin)
            llvm::errs() << " " << value;
          llvm::errs() << "\n";
        }
        return result;
      }
    }
    auto *layoutCtx = memDesc.getContext();
    auto kCol = StringAttr::get(layoutCtx, "col");
    if (!srcQuery->layout.hasInDim(kCol) || memDescTy.getShape().size() < 2 ||
        memDescTy.getShape().back() > srcQuery->layout.getInDimSize(kCol)) {
      if (error)
        *error = layoutError.empty()
                     ? "unsupported tensor memory memdesc_subslice view"
                     : layoutError;
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] ttng.tmem_subslice resize fail: "
                     << *error << "\n";
      return failure();
    }
    auto resizedLayout = [&]() {
      auto bases = srcQuery->layout.getBases();
      auto colIt = bases.find(kCol);
      if (colIt == bases.end())
        return srcQuery->layout;

      unsigned zeroBases = llvm::count_if(
          colIt->second, [](ArrayRef<int32_t> basis) {
            return llvm::all_of(basis,
                                [](int32_t value) { return value == 0; });
          });
      unsigned targetBits = llvm::Log2_64_Ceil(
          static_cast<uint64_t>(memDescTy.getShape().back()));
      unsigned desiredBases = zeroBases + targetBits;
      if (desiredBases == 0 || desiredBases >= colIt->second.size())
        return srcQuery->layout.resizeInDim(kCol, memDescTy.getShape().back());

      colIt->second.resize(desiredBases);
      auto outDims = srcQuery->layout.getOutDims();
      auto outDimNames = llvm::to_vector(srcQuery->layout.getOutDimNames());
      SmallVector<StringAttr> colOutDims;
      for (auto basis : colIt->second) {
        for (auto [idx, value] : llvm::enumerate(basis)) {
          if (value != 0)
            colOutDims.push_back(outDimNames[idx]);
        }
      }
      llvm::sort(colOutDims, [](StringAttr lhs, StringAttr rhs) {
        return lhs.getValue() < rhs.getValue();
      });
      colOutDims.erase(std::unique(colOutDims.begin(), colOutDims.end()),
                       colOutDims.end());
      if (colOutDims.size() == 1) {
        for (auto &outDim : outDims) {
          if (outDim.first == colOutDims.front()) {
            outDim.second = memDescTy.getShape().back();
            break;
          }
        }
      }
      return LinearLayout(std::move(bases), std::move(outDims),
                          /*isSurjective=*/true);
    }();
    SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
    encodedOffsets.reserve(resizedLayout.getNumInDims());
    for (auto dim : resizedLayout.getInDimNames())
      encodedOffsets.push_back({dim, dim == kCol ? subslice.getN() : 0});
    auto result = TMemLdStQueryLayout{
        resizedLayout, srcQuery->twoCTAs,
        remapTMemLdStQueryOrigin(*srcQuery, resizedLayout, encodedOffsets)};
    if (debug) {
      llvm::errs() << "[tmem-ldst] ttng.tmem_subslice origin:";
      for (int32_t value : result.origin)
        llvm::errs() << " " << value;
      llvm::errs() << "\n";
    }
    return result;
  }
  if (auto index = memDesc.getDefiningOp<gpu::MemDescIndexOp>()) {
    auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        index.getSrc(), preserveNonCanonicalView, error);
    if (failed(srcQuery)) {
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] memdesc_index srcQuery fail src="
                     << cast<MemDescType>(index.getSrc().getType())
                     << " dst=" << memDescTy << " err=" << *error << "\n";
      return failure();
    }
    auto srcTy = cast<MemDescType>(index.getSrc().getType());
    APInt indexValue;
    std::optional<int32_t> leadingIndex;
    if (matchPattern(index.getIndex(), m_ConstantInt(&indexValue)))
      leadingIndex = static_cast<int32_t>(indexValue.getSExtValue());
    return inferTMemIndexQueryLayout(srcTy.getShape(), memDescTy.getShape(),
                                     *srcQuery, srcTy.getElementTypeBitWidth(),
                                     leadingIndex, ctx, error);
  }
  if (auto reshape = memDesc.getDefiningOp<gpu::MemDescReshapeOp>()) {
    auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        reshape.getSrc(), preserveNonCanonicalView, error);
    if (failed(srcQuery)) {
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] memdesc_reshape srcQuery fail src="
                     << cast<MemDescType>(reshape.getSrc().getType())
                     << " dst=" << memDescTy << " err=" << *error << "\n";
      return failure();
    }
    auto srcTy = cast<MemDescType>(reshape.getSrc().getType());
    return inferTMemReshapeQueryLayout(srcTy.getShape(), *srcQuery,
                                       memDescTy.getShape(), ctx, error);
  }
  auto preserveViewOrigin = [&](Value src) -> FailureOr<TMemLdStQueryLayout> {
    auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        src, preserveNonCanonicalView, error);
    if (failed(srcQuery)) {
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] preserveViewOrigin srcQuery fail src="
                     << cast<MemDescType>(src.getType()) << " dst=" << memDescTy
                     << " err=" << *error << "\n";
      return failure();
    }
    auto maybeAnalysis =
        getTMemViewAnalysisLayout(memDescTy.getShape(), memDescTy.getEncoding(),
                                  error);
    if (!maybeAnalysis) {
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] preserveViewOrigin analysis fail dst="
                     << memDescTy << " err=" << *error << "\n";
      return failure();
    }
    if (!isTMemLdStQueryOriginRepresentable(*srcQuery, srcQuery->layout)) {
      return TMemLdStQueryLayout{
          maybeAnalysis->layout, maybeAnalysis->twoCTAs,
          remapTMemLdStQueryOrigin(*srcQuery, maybeAnalysis->layout,
                                   /*deltaCoords=*/{})};
    }
    auto remappedOrigin = remapTMemLdStQueryOriginThroughPhysicalCoords(
        *srcQuery, maybeAnalysis->layout, error);
    if (failed(remappedOrigin)) {
      auto normalizedSrcLayout =
          normalizeTensorMemoryLinearLayoutForAnalysis(srcQuery->layout);
      auto normalizedDstLayout =
          normalizeTensorMemoryLinearLayoutForAnalysis(maybeAnalysis->layout);
      auto normalizedSrcQuery = TMemLdStQueryLayout{
          normalizedSrcLayout, srcQuery->twoCTAs,
          remapTMemLdStQueryOrigin(*srcQuery, normalizedSrcLayout,
                                   /*deltaCoords=*/{})};
      auto normalizedOrigin = remapTMemLdStQueryOriginThroughPhysicalCoords(
          normalizedSrcQuery, normalizedDstLayout, error);
      if (succeeded(normalizedOrigin)) {
        if (debug) {
          llvm::errs() << "[tmem-ldst] preserveViewOrigin normalized remap src="
                       << cast<MemDescType>(src.getType()) << " dst=" << memDescTy
                       << "\n";
        }
        return TMemLdStQueryLayout{normalizedDstLayout, maybeAnalysis->twoCTAs,
                                   *normalizedOrigin};
      }
      if (debug && error && !error->empty())
        llvm::errs() << "[tmem-ldst] preserveViewOrigin remap fail src="
                     << cast<MemDescType>(src.getType()) << " dst=" << memDescTy
                     << " err=" << *error << "\n";
      return failure();
    }
    return TMemLdStQueryLayout{maybeAnalysis->layout, maybeAnalysis->twoCTAs,
                               *remappedOrigin};
  };
  if (auto trans = memDesc.getDefiningOp<gpu::MemDescTransOp>())
    return preserveViewOrigin(trans.getSrc());
  if (auto reinterpret = memDesc.getDefiningOp<gpu::MemDescReinterpretOp>()) {
    if (debug)
      llvm::errs() << "[tmem-ldst] reinterpret branch memTy=" << memDescTy
                   << "\n";
    auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        reinterpret.getSrc(), preserveNonCanonicalView, error);
    if (failed(srcQuery))
      return failure();
    auto srcTy = cast<MemDescType>(reinterpret.getSrc().getType());
    if (srcTy.getElementTypeBitWidth() != memDescTy.getElementTypeBitWidth())
      return inferTMemReinterpretQueryLayout(
          srcTy.getShape(), srcTy.getElementTypeBitWidth(), *srcQuery,
          memDescTy.getShape(), memDescTy.getElementTypeBitWidth(),
          memDesc.getContext(), error);
    return preserveViewOrigin(reinterpret.getSrc());
  }

  if (memDescTy.getShape().take_back(layoutRank) ==
      memDescTy.getAllocShape().take_back(layoutRank)) {
    if (!maybeAnalysis)
      maybeAnalysis =
          getTMemViewAnalysisLayout(memDescTy.getShape(), encoding, error);
    if (!maybeAnalysis)
      return failure();
    return *maybeAnalysis;
  }

  if (error)
    *error = "unsupported tensor memory descriptor view for reg-layout query";
  return failure();
}
} // namespace

std::optional<TMemLdStRowPlan> getTMemLdStRowPlanForType(MemDescType memTy) {
  if (isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
    // TMEM scales use logical broadcast row bases in their linear layout, but
    // the direct ld/st path still programs the hardware using the standard
    // 128-row anchor pair at rows 32 and 64.
    return TMemLdStRowPlan{/*warpRow0=*/32, /*warpRow1=*/64,
                           /*rowSpan=*/128};
  }
  if (auto rootPlan = getFullShapeM64TMemRowPlan(memTy))
    return rootPlan;
  if (auto familyQuery = getFullShapeMMAv5FamilyQueryLayout(memTy))
    if (auto familyPlan = getTMemLdStRowPlan(familyQuery->layout))
      return familyPlan;

  std::string error;
  auto maybeLayout =
      getTMemViewAnalysisLinearLayout(memTy.getShape(), memTy.getEncoding(),
                                      &error);
  if (!maybeLayout)
    return std::nullopt;

  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!maybeLayout->hasInDim(kRow))
    return std::nullopt;
  int64_t logicalRows = memTy.getShape()[memTy.getRank() - 2];
  int64_t logicalCols = memTy.getShape()[memTy.getRank() - 1];
  unsigned rowBits = maybeLayout->getInDimSizeLog2(kRow);
  auto isZeroRowBasis = [&](unsigned idx) {
    return idx < rowBits &&
           llvm::all_of(maybeLayout->getBasis(kRow, idx),
                        [](int32_t v) { return v == 0; });
  };
  auto normalizedLayout =
      normalizeTensorMemoryLinearLayoutForAnalysis(*maybeLayout);
  auto activeLayout = normalizedLayout.removeZeroBasesAlongDim(kRow);
  unsigned activeRowBits =
      activeLayout.hasInDim(kRow) ? activeLayout.getInDimSizeLog2(kRow) : 0;
  auto isZeroActiveRowBasis = [&](unsigned idx) {
    return idx < activeRowBits &&
           llvm::all_of(activeLayout.getBasis(kRow, idx),
                        [](int32_t v) { return v == 0; });
  };
  auto planFromRowBits = [&](unsigned bits, auto &&isZeroBasis)
      -> std::optional<TMemLdStRowPlan> {
    if (bits >= 7) {
      if (isZeroBasis(bits - 2) && isZeroBasis(bits - 1)) {
        return TMemLdStRowPlan{/*warpRow0=*/0, /*warpRow1=*/0,
                               /*rowSpan=*/128};
      }
      if (!isZeroBasis(bits - 2) && isZeroBasis(bits - 1)) {
        return TMemLdStRowPlan{/*warpRow0=*/32, /*warpRow1=*/64,
                               /*rowSpan=*/128, /*baseOffset=*/4};
      }
      return TMemLdStRowPlan{/*warpRow0=*/32, /*warpRow1=*/64,
                             /*rowSpan=*/128};
    }
    if (bits == 6) {
      if (isZeroBasis(bits - 2) && isZeroBasis(bits - 1)) {
        return TMemLdStRowPlan{/*warpRow0=*/0, /*warpRow1=*/0,
                               /*rowSpan=*/64};
      }
      return TMemLdStRowPlan{/*warpRow0=*/16, /*warpRow1=*/32,
                             /*rowSpan=*/64};
    }
    return std::nullopt;
  };
  // When the logical M dimension is widened via CGA block bases, direct
  // ld/st anchors must be derived from the local active row footprint, not
  // from the widened global logical shape. Otherwise blockM=64 accumulator
  // layouts incorrectly widen to the 128-row family and reject valid direct
  // register layouts.
  if (normalizedLayout.hasInDim(kBlock) &&
      normalizedLayout.getInDimSize(kBlock) > 1 &&
      activeLayout.hasInDim(kRow) &&
      logicalRows > activeLayout.getInDimSize(kRow)) {
    return planFromRowBits(activeRowBits, isZeroActiveRowBasis);
  }
  // Legacy and sparse/non-surjective logical M64 TMEM encodings can carry
  // an explicit zero row basis in the raw linear form. Classifying those
  // layouts from the raw row-basis count alone widens them to the 128-row
  // family and breaks direct ld/st layout selection for plain 64xN MMA
  // accumulators as well as split-N variants. For logical M64 tiles, derive
  // the row plan from the active row bases instead.
  bool shapeMatchesAllocTail =
      memTy.getAllocShape().size() >= static_cast<size_t>(memTy.getRank()) &&
      llvm::equal(memTy.getShape().take_back(2),
                  memTy.getAllocShape().take_back(2));
  if (logicalRows == 64 && activeRowBits == 6 && shapeMatchesAllocTail) {
    return planFromRowBits(activeRowBits, isZeroActiveRowBasis);
  }
  return planFromRowBits(rowBits, isZeroRowBasis);
}

std::optional<TMemLdStRowPlan> getBackingTMemLdStRowPlan(Value memDesc) {
  std::optional<TMemLdStRowPlan> best;
  auto consider = [&](Value value) {
    auto memTy = dyn_cast<MemDescType>(value.getType());
    if (!memTy)
      return;
    auto maybePlan = getTMemLdStRowPlanForType(memTy);
    if (!maybePlan)
      return;
    if (!best || maybePlan->rowSpan > best->rowSpan)
      best = *maybePlan;
  };

  Value cur = memDesc;
  while (cur) {
    consider(cur);
    if (auto forwarded = getTMemForwardingSource(cur)) {
      cur = forwarded;
      continue;
    }
    Operation *def = cur.getDefiningOp();
    if (!def)
      break;
    if (auto op = dyn_cast<gpu::MemDescIndexOp>(def)) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = dyn_cast<gpu::MemDescSubsliceOp>(def)) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = dyn_cast<TMEMSubSliceOp>(def)) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = dyn_cast<gpu::MemDescReshapeOp>(def)) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = dyn_cast<gpu::MemDescReinterpretOp>(def)) {
      cur = op.getSrc();
      continue;
    }
    if (auto op = dyn_cast<gpu::MemDescTransOp>(def)) {
      cur = op.getSrc();
      continue;
    }
    break;
  }
  return best;
}

bool preferBackingTMemLdStQueryTypes(Value memDesc) {
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy)
    return false;
  if (!isa_and_nonnull<gpu::MemDescSubsliceOp, TMEMSubSliceOp, gpu::MemDescIndexOp,
                       gpu::MemDescReshapeOp>(memDesc.getDefiningOp())) {
    return false;
  }
  auto queryPlan = getTMemLdStRowPlanForType(memTy);
  auto backingPlan = getBackingTMemLdStRowPlan(memDesc);
  return queryPlan && backingPlan && queryPlan->rowSpan < backingPlan->rowSpan;
}

static bool isPureOuterTMemIndexView(Value memDesc) {
  // Pure outer memdesc_index chains only peel non-layout prefix dimensions.
  // Their trailing 2D TMEM tile is unchanged, so direct ld/st planning should
  // reason from the query row plan rather than the larger backing view.
  auto queryTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!queryTy || queryTy.getRank() != 2 ||
      !isTensorMemoryEncoding(queryTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(queryTy.getEncoding())) {
    return false;
  }

  auto layoutRank =
      static_cast<size_t>(cast<LayoutEncodingTrait>(queryTy.getEncoding()).getRank());
  if (layoutRank != 2)
    return false;

  bool sawIndex = false;
  Value cur = memDesc;
  while (auto index = cur.getDefiningOp<gpu::MemDescIndexOp>()) {
    sawIndex = true;
    auto curTy = dyn_cast<MemDescType>(cur.getType());
    auto srcTy = dyn_cast<MemDescType>(index.getSrc().getType());
    if (!curTy || !srcTy || curTy.getRank() + 1 != srcTy.getRank() ||
        !isTensorMemoryEncoding(srcTy.getEncoding()) ||
        isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
      return false;
    }

    auto curLayoutRank = static_cast<size_t>(
        cast<LayoutEncodingTrait>(curTy.getEncoding()).getRank());
    auto srcLayoutRank = static_cast<size_t>(
        cast<LayoutEncodingTrait>(srcTy.getEncoding()).getRank());
    if (curLayoutRank != layoutRank || srcLayoutRank != layoutRank ||
        srcTy.getRank() <= static_cast<int64_t>(layoutRank) ||
        !llvm::equal(srcTy.getShape().take_back(layoutRank),
                     curTy.getShape().take_back(layoutRank))) {
      return false;
    }

    if (srcTy.getAllocShape().size() >= layoutRank &&
        curTy.getAllocShape().size() >= layoutRank &&
        !llvm::equal(srcTy.getAllocShape().take_back(layoutRank),
                     curTy.getAllocShape().take_back(layoutRank))) {
      return false;
    }

    cur = index.getSrc();
  }

  if (!sawIndex)
    return false;

  Operation *baseDef = cur.getDefiningOp();
  return !baseDef || isa<TMEMAllocOp>(baseDef) ||
         baseDef->getName().getStringRef() == "nvws.aref.buffer";
}

static bool isHigherRankHalfRowsSubview(Value memDesc) {
  auto queryTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  auto index = memDesc.getDefiningOp<gpu::MemDescIndexOp>();
  if (!queryTy || !index || queryTy.getRank() != 2)
    return false;

  APInt indexValue;
  if (!matchPattern(index.getIndex(), m_ConstantInt(&indexValue)) ||
      indexValue.getSExtValue() != 0) {
    return false;
  }

  auto subslice = index.getSrc().getDefiningOp<gpu::MemDescSubsliceOp>();
  if (!subslice || subslice.getOffsets().size() != 3 ||
      subslice.getOffsets()[0] != 1 || subslice.getOffsets()[1] != 0 ||
      subslice.getOffsets()[2] != 0) {
    return false;
  }

  auto reshape = subslice.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
  if (!reshape)
    return false;

  auto reshapeTy = dyn_cast<MemDescType>(reshape.getType());
  auto srcTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
  if (!reshapeTy || !srcTy || reshapeTy.getRank() != 3 || srcTy.getRank() != 2)
    return false;

  int64_t rows = queryTy.getShape()[0];
  int64_t cols = queryTy.getShape()[1];
  return reshapeTy.getShape()[0] == 2 && reshapeTy.getShape()[1] == rows &&
         reshapeTy.getShape()[2] == cols && srcTy.getShape()[0] == rows * 2 &&
         srcTy.getShape()[1] == cols;
}

static bool isDirectHalfRowsSubview(Value memDesc) {
  auto queryTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  auto subslice = memDesc.getDefiningOp<gpu::MemDescSubsliceOp>();
  if (!queryTy || !subslice || queryTy.getRank() != 2)
    return false;

  auto srcTy = dyn_cast<MemDescType>(subslice.getSrc().getType());
  if (!srcTy || srcTy.getRank() != 2)
    return false;

  int64_t rows = queryTy.getShape()[0];
  int64_t cols = queryTy.getShape()[1];
  auto offsets = subslice.getOffsets();
  return offsets.size() == 2 && offsets[0] == rows && offsets[1] == 0 &&
         srcTy.getShape()[0] == rows * 2 && srcTy.getShape()[1] == cols;
}

static bool shouldPreferDirectHalfRowsSubviewRowPlan(
    Value memDesc, MemDescType queryTy,
    std::optional<TMemLdStRowPlan> queryPlan,
    std::optional<TMemLdStRowPlan> backingPlan) {
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy || queryTy != memTy || !queryPlan || !backingPlan ||
      queryTy.getRank() != 2) {
    return false;
  }
  return queryPlan->rowSpan == queryTy.getShape()[0] &&
         backingPlan->rowSpan == queryTy.getShape()[0] * 2 &&
         (isDirectHalfRowsSubview(memDesc) ||
          isHigherRankHalfRowsSubview(memDesc));
}

static bool shouldPreferBackingRowPlanForPureOuterIndexView(
    Value memDesc, MemDescType queryTy,
    std::optional<TMemLdStRowPlan> queryPlan,
    std::optional<TMemLdStRowPlan> backingPlan) {
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy || queryTy != memTy || !queryPlan || !backingPlan ||
      !isPureOuterTMemIndexView(memDesc) ||
      backingPlan->rowSpan <= queryPlan->rowSpan || queryTy.getRank() != 2 ||
      queryTy.getShape()[1] != 32) {
    return false;
  }

  // Pure outer memdesc_index chains preserve the same trailing 2D TMEM tile.
  // When that tile is rooted in a producer-owned larger row-plan contract
  // (for example MMAv5 accumulator roots), keep the backing plan as long as
  // the unchanged raw TMEM view can still materialize those anchors.
  auto rawLayout = [&]() {
    std::string queryError;
    if (auto maybeQueryLayout = inferStandaloneTMemLdStQueryLayoutImpl(
            memDesc, /*preserveNonCanonicalView=*/true, &queryError);
        succeeded(maybeQueryLayout)) {
      return maybeQueryLayout->layout;
    }
    return toLinearLayout(queryTy);
  }();
  return getLogicalRowAnchorBasis(rawLayout, backingPlan->warpRow0) &&
         getLogicalRowAnchorBasis(rawLayout, backingPlan->warpRow1);
}

std::optional<TMemLdStRowPlan> getTMemLdStRowPlanForQuery(Value memDesc,
                                                          MemDescType queryTy) {
  auto queryPlan = getTMemLdStRowPlanForType(queryTy);
  auto backingPlan = memDesc ? getBackingTMemLdStRowPlan(memDesc)
                             : std::optional<TMemLdStRowPlan>{};
  if (!memDesc)
    return queryPlan;
  if (!queryPlan)
    return backingPlan;

  auto memTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!memTy || queryTy != memTy)
    return backingPlan;

  if (shouldPreferDirectHalfRowsSubviewRowPlan(memDesc, queryTy, queryPlan,
                                               backingPlan))
    return queryPlan;
  if (shouldPreferBackingRowPlanForPureOuterIndexView(memDesc, queryTy,
                                                      queryPlan, backingPlan))
    return backingPlan;
  if (isPureOuterTMemIndexView(memDesc)) {
    std::string queryError;
    if (auto maybeQueryLayout = inferStandaloneTMemLdStQueryLayoutImpl(
            memDesc, /*preserveNonCanonicalView=*/true, &queryError);
        succeeded(maybeQueryLayout)) {
      if (auto layoutPlan = getTMemLdStRowPlan(maybeQueryLayout->layout)) {
        // Pure outer indexes use the concrete query layout's row plan. Keep the
        // wider type/family plan only when the query layout itself still carries
        // the MMAv5 family block dimension.
        auto kBlock = StringAttr::get(memDesc.getContext(), "block");
        bool hasFamilyBlockDim = maybeQueryLayout->layout.hasInDim(kBlock);
        if (layoutPlan->rowSpan >= queryPlan->rowSpan || !hasFamilyBlockDim)
          return layoutPlan;
      }
    }
    return queryPlan;
  }

  if (backingPlan && backingPlan->rowSpan > queryPlan->rowSpan &&
      isa_and_nonnull<gpu::MemDescReinterpretOp>(memDesc.getDefiningOp()) &&
      queryTy.getRank() == 2 && queryTy.getShape()[0] == 64) {
    std::string queryError;
    auto rawLayout = [&]() {
      if (auto maybeQueryLayout = inferStandaloneTMemLdStQueryLayoutImpl(
              memDesc, /*preserveNonCanonicalView=*/true, &queryError);
          succeeded(maybeQueryLayout)) {
        return maybeQueryLayout->layout;
      }
      return toLinearLayout(queryTy);
    }();
    if (getLogicalRowAnchorBasis(rawLayout, backingPlan->warpRow0) &&
        getLogicalRowAnchorBasis(rawLayout, backingPlan->warpRow1))
      return backingPlan;
  }

  auto encoding = queryTy.getEncoding();
  if (isa<TensorMemoryScalesEncodingAttr>(encoding))
    return backingPlan;

  auto preferQueryPlanForM64SplitNSubview = [&]() {
    return queryPlan && queryPlan->rowSpan == 64 && queryTy.getRank() == 2 &&
           queryTy.getElementTypeBitWidth() == 32 && queryTy.getShape()[0] == 64 &&
           !queryTy.getAllocShape().empty() &&
           queryTy.getAllocShape().back() > queryTy.getShape()[1];
  };
  if (preferBackingTMemLdStQueryTypes(memDesc) &&
      !preferQueryPlanForM64SplitNSubview())
    return backingPlan;

  auto layoutRank =
      static_cast<size_t>(cast<LayoutEncodingTrait>(encoding).getRank());
  bool shapeMatchesAllocTail =
      queryTy.getShape().take_back(layoutRank) ==
      queryTy.getAllocShape().take_back(layoutRank);
  if (shapeMatchesAllocTail &&
      getCanonicalTMemLinearEncoding(queryTy, /*error=*/nullptr)) {
    return queryPlan;
  }
  if (preferQueryPlanForM64SplitNSubview())
    return queryPlan;
  return backingPlan;
}

std::optional<TMemLdStRowPlan>
getTMemLdStRowPlanForQueryLayout(Value memDesc, MemDescType queryTy,
                                 const TMemLdStQueryLayout &queryLayout) {
  auto queryPlan = getTMemLdStRowPlanForQuery(memDesc, queryTy);
  auto layoutPlan = getTMemLdStRowPlan(queryLayout.layout);
  if (!layoutPlan)
    return queryPlan;
  if (!queryPlan)
    return layoutPlan;
  if (!memDesc)
    return queryPlan;

  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy || memTy != queryTy)
    return queryPlan;
  if (auto backingPlan = getBackingTMemLdStRowPlan(memDesc)) {
    if (queryPlan && backingPlan->rowSpan > queryPlan->rowSpan &&
        isa_and_nonnull<gpu::MemDescReinterpretOp>(memDesc.getDefiningOp()) &&
        queryTy.getRank() == 2 && queryTy.getShape()[0] == 64 &&
        getLogicalRowAnchorBasis(queryLayout.layout, backingPlan->warpRow0) &&
        getLogicalRowAnchorBasis(queryLayout.layout, backingPlan->warpRow1)) {
      return backingPlan;
    }
  }
  auto getProjectedM64LayoutPlan = [&]()
      -> std::optional<TMemLdStRowPlan> {
    if (layoutPlan->rowSpan == queryTy.getShape()[0])
      return layoutPlan;

    auto *ctx = queryTy.getContext();
    auto kRow = StringAttr::get(ctx, "row");
    auto normalizedLayout =
        normalizeTensorMemoryLinearLayoutForAnalysis(queryLayout.layout);
    if (!normalizedLayout.hasInDim(kRow))
      return std::nullopt;
    auto activeLayout = normalizedLayout.removeZeroBasesAlongDim(kRow);
    if (!activeLayout.hasInDim(kRow) ||
        activeLayout.getInDimSize(kRow) != queryTy.getShape()[0] ||
        activeLayout.getInDimSizeLog2(kRow) != 6) {
      return std::nullopt;
    }
    auto isZeroActiveRowBasis = [&](unsigned idx) {
      return idx < activeLayout.getInDimSizeLog2(kRow) &&
             llvm::all_of(activeLayout.getBasis(kRow, idx),
                          [](int32_t v) { return v == 0; });
    };
    if (isZeroActiveRowBasis(4) && isZeroActiveRowBasis(5)) {
      return TMemLdStRowPlan{/*warpRow0=*/0, /*warpRow1=*/0,
                             /*rowSpan=*/64};
    }
    return TMemLdStRowPlan{/*warpRow0=*/16, /*warpRow1=*/32,
                           /*rowSpan=*/64};
  };
  auto projectedM64Plan = getProjectedM64LayoutPlan();
  bool isProjectedM64Subview =
      queryTy.getRank() == 2 && queryTy.getElementTypeBitWidth() == 32 &&
      queryTy.getShape()[0] == 64 && projectedM64Plan &&
      projectedM64Plan->rowSpan == queryTy.getShape()[0] &&
      projectedM64Plan->rowSpan < queryPlan->rowSpan;
  if (isProjectedM64Subview && isPureOuterTMemIndexView(memDesc))
    return queryPlan;
  if (isProjectedM64Subview)
    return projectedM64Plan;
  return queryPlan;
}

static std::optional<gpu::MemDescType>
getCanonicalTMemLdStSurrogateType(gpu::MemDescType queryTy,
                                  std::optional<TMemLdStRowPlan> backingPlan,
                                  std::string *error) {
  if (!backingPlan || queryTy.getRank() != 2)
    return std::nullopt;

  auto *ctx = queryTy.getContext();
  gpu::CGAEncodingAttr cga =
      gpu::CGAEncodingAttr::get1CTALayout(ctx, queryTy.getRank());
  bool twoCTAs = false;
  if (auto linear =
          dyn_cast<TensorMemoryLinearEncodingAttr>(queryTy.getEncoding())) {
    cga = linear.getCGALayout();
    twoCTAs = linear.getTwoCTAs();
  } else if (auto legacy =
                 dyn_cast<TensorMemoryEncodingAttr>(queryTy.getEncoding())) {
    cga = legacy.getCGALayout();
    twoCTAs = legacy.getTwoCTAs();
  } else {
    return std::nullopt;
  }

  SmallVector<int64_t> surrogateAllocShape(queryTy.getAllocShape().begin(),
                                           queryTy.getAllocShape().end());
  if (surrogateAllocShape.size() != static_cast<size_t>(queryTy.getRank()))
    surrogateAllocShape.assign(queryTy.getShape().begin(), queryTy.getShape().end());
  surrogateAllocShape[surrogateAllocShape.size() - 2] =
      std::max<int64_t>(surrogateAllocShape[surrogateAllocShape.size() - 2],
                        backingPlan->rowSpan);
  surrogateAllocShape[surrogateAllocShape.size() - 1] =
      std::max<int64_t>(surrogateAllocShape[surrogateAllocShape.size() - 1],
                        queryTy.getShape()[1]);

  auto maybeCanonical = getCanonicalTMemLinearEncoding(
      ArrayRef<int64_t>(surrogateAllocShape).take_back(2), backingPlan->rowSpan,
      queryTy.getShape()[1],
      /*colStride=*/1, cga, twoCTAs, error);
  if (!maybeCanonical)
    return std::nullopt;

  return tryCreateMemDescType(ctx, queryTy.getShape(), queryTy.getElementType(),
                              *maybeCanonical, queryTy.getMemorySpace(),
                              queryTy.getMutableMemory(), surrogateAllocShape,
                              error);
}

static bool isHigherRankHalfRowsSubview(Value memDesc);
static bool isDirectHalfRowsSubview(Value memDesc);
static bool shouldPreferDirectHalfRowsSubviewRowPlan(
    Value memDesc, MemDescType queryTy,
    std::optional<TMemLdStRowPlan> queryPlan,
    std::optional<TMemLdStRowPlan> backingPlan);

llvm::SmallVector<gpu::MemDescType> getTMemLdStQueryTypes(Value memDesc) {
  llvm::SmallVector<gpu::MemDescType> queryTypes;
  auto memTy = dyn_cast<gpu::MemDescType>(memDesc.getType());
  if (!memTy)
    return queryTypes;
  bool explicitViewProducer =
      isa_and_nonnull<gpu::MemDescSubsliceOp, TMEMSubSliceOp, gpu::MemDescIndexOp,
                      gpu::MemDescReshapeOp, gpu::MemDescTransOp,
                      gpu::MemDescReinterpretOp>(memDesc.getDefiningOp());

  auto add = [&](gpu::MemDescType ty) {
    if (llvm::none_of(queryTypes, [&](gpu::MemDescType existing) {
          return existing == ty;
        })) {
      queryTypes.push_back(ty);
    }
  };

  bool memTyCanonical =
      getCanonicalTMemLinearEncoding(memTy, /*error=*/nullptr).has_value();

  bool preferStandaloneBeforeRawType =
      isa_and_nonnull<TMEMSubSliceOp>(memDesc.getDefiningOp()) &&
      memTy.getElementTypeBitWidth() < 32;

  std::string error;
  std::optional<gpu::MemDescType> standaloneTy;
  if (auto maybeStandalone = inferStandaloneTMemRegLayoutQueryType(
          memDesc, &error);
      succeeded(maybeStandalone)) {
    standaloneTy = *maybeStandalone;
    if (preferStandaloneBeforeRawType && *maybeStandalone != memTy)
      add(*maybeStandalone);
  }
  if (explicitViewProducer)
    add(memTy);
  else if (!memTyCanonical)
    add(memTy);
  if (standaloneTy && !preferStandaloneBeforeRawType && *standaloneTy != memTy)
    add(*standaloneTy);

  auto backingPlan = getBackingTMemLdStRowPlan(memDesc);
  if (standaloneTy) {
    if (auto surrogate =
            getCanonicalTMemLdStSurrogateType(*standaloneTy, backingPlan,
                                              /*error=*/nullptr)) {
      add(*surrogate);
    }
  }
  if (auto surrogate =
          getCanonicalTMemLdStSurrogateType(memTy, backingPlan,
                                            /*error=*/nullptr)) {
    add(*surrogate);
  }

  // For explicit descriptor view chains, try the preserved sparse view first
  // and fall back to canonical/surrogate types only when that direct path is
  // unavailable. Leaf layouts can still prefer the canonical type.
  if (!explicitViewProducer && memTyCanonical)
    add(memTy);
  else if (queryTypes.empty())
    add(memTy);

  return queryTypes;
}

static bool isTensorMemoryColumnHalfDim0Slice(gpu::MemDescSubsliceOp op) {
  auto srcTy = cast<MemDescType>(op.getSrc().getType());
  auto dstTy = cast<MemDescType>(op.getType());
  if (!isTensorMemoryEncoding(srcTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
    return false;
  }
  if (srcTy.getRank() != 3 || dstTy.getRank() != 3)
    return false;
  auto reshape = op.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
  if (!reshape)
    return false;
  auto reshapeSrcTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
  if (!reshapeSrcTy || reshapeSrcTy.getRank() != 2)
    return false;
  auto offsets = op.getOffsets();
  if (offsets.size() != 3 || offsets[0] != 1 || offsets[1] != 0 ||
      offsets[2] != 0) {
    return false;
  }
  if (srcTy.getShape()[0] != 2 || dstTy.getShape()[0] != 1 ||
      srcTy.getShape()[1] != dstTy.getShape()[1] ||
      srcTy.getShape()[2] != dstTy.getShape()[2]) {
    return false;
  }
  int64_t rows = dstTy.getShape()[1];
  int64_t cols = dstTy.getShape()[2];
  return reshapeSrcTy.getShape()[0] == rows &&
         reshapeSrcTy.getShape()[1] == cols * 2;
}

static bool isTensorMemoryRowHalfDim0Slice(gpu::MemDescSubsliceOp op) {
  auto srcTy = cast<MemDescType>(op.getSrc().getType());
  auto dstTy = cast<MemDescType>(op.getType());
  if (!isTensorMemoryEncoding(srcTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
    return false;
  }
  if (srcTy.getRank() != 3 || dstTy.getRank() != 3)
    return false;
  auto reshape = op.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
  if (!reshape)
    return false;
  auto reshapeSrcTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
  if (!reshapeSrcTy || reshapeSrcTy.getRank() != 2)
    return false;
  auto offsets = op.getOffsets();
  if (offsets.size() != 3 || offsets[0] != 1 || offsets[1] != 0 ||
      offsets[2] != 0) {
    return false;
  }
  if (srcTy.getShape()[0] != 2 || dstTy.getShape()[0] != 1 ||
      srcTy.getShape()[1] != dstTy.getShape()[1] ||
      srcTy.getShape()[2] != dstTy.getShape()[2]) {
    return false;
  }
  int64_t rows = dstTy.getShape()[1];
  int64_t cols = dstTy.getShape()[2];
  return reshapeSrcTy.getShape()[0] == rows * 2 &&
         reshapeSrcTy.getShape()[1] == cols;
}

static std::optional<uint32_t>
getCanonicalContiguous32x32SubviewOffset(gpu::MemDescSubsliceOp op) {
  auto srcTy = cast<MemDescType>(op.getSrc().getType());
  auto dstTy = cast<MemDescType>(op.getType());
  if (!isTensorMemoryEncoding(srcTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding()) ||
      srcTy.getRank() != 4 || dstTy.getRank() != 4) {
    return std::nullopt;
  }
  auto reshape = op.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
  if (!reshape)
    return std::nullopt;
  auto rootTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
  if (!rootTy || rootTy.getRank() != 2 || rootTy.getShape()[0] != 128 ||
      rootTy.getShape()[1] != 128) {
    return std::nullopt;
  }
  std::string error;
  if (!getCanonicalTMemLinearEncoding(rootTy, &error))
    return std::nullopt;

  auto offsets = op.getOffsets();
  if (offsets.size() != 4 || offsets[0] != 1 || offsets[1] != 0 ||
      offsets[2] != 1 || offsets[3] != 0) {
    return std::nullopt;
  }
  if (srcTy.getShape()[0] != 2 || srcTy.getShape()[1] != 64 ||
      srcTy.getShape()[2] != 2 || srcTy.getShape()[3] != 64) {
    return std::nullopt;
  }
  if (dstTy.getShape()[0] != 1 || dstTy.getShape()[1] != 32 ||
      dstTy.getShape()[2] != 1 || dstTy.getShape()[3] != 32) {
    return std::nullopt;
  }

  // Direct 32x32 ld/st uses the scalarized x1 packet family, which expects
  // this static contiguous slice to be addressed in the same folded support
  // frame as the working mixed-layout path instead of the raw canonical
  // row<<16|col view offset.
  return static_cast<uint32_t>(srcTy.getShape()[3] +
                               srcTy.getShape()[1] / dstTy.getShape()[1]);
}

uint32_t getTMemViewOffsetForLowering(Value memDesc, ArrayRef<int32_t> offsets) {
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy)
    return 0;
  std::string queryError;
  auto expectedOutDims = standardOutDimNames(memDesc.getContext(), memTy.getRank());
  if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
          memDesc, /*preserveNonCanonicalView=*/true, &queryError);
      succeeded(rawQuery) &&
      rawQuery->layout.getNumOutDims() == static_cast<unsigned>(memTy.getRank()) &&
      llvm::equal(rawQuery->layout.getOutDimNames(), expectedOutDims) &&
      rawQuery->layout.isSurjective()) {
    // Query layouts already describe the physical TMEM view, but the final
    // base offset still needs the descriptor element width so logical columns
    // land on the correct 32-bit TMEM words for packed element types.
    // Projected query layouts can be non-surjective: they are sufficient to
    // describe an ld/st support image, but cannot invert arbitrary logical
    // subview offsets. Fall back to the descriptor type's full layout below.
    return getTMemViewOffset(rawQuery->layout, offsets,
                             memTy.getElementTypeBitWidth());
  }
  return getTMemViewOffset(memTy, offsets);
}

static std::optional<uint32_t> getTMemLdStQueryOriginDeltaBaseOffset(
    const TMemLdStQueryLayout &srcQuery, const TMemLdStQueryLayout &dstQuery,
    int bitwidth) {
  auto *ctx = dstQuery.layout.getInDimNames().begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto srcInDims = llvm::to_vector(srcQuery.layout.getInDimNames());
  auto dstInDims = llvm::to_vector(dstQuery.layout.getInDimNames());

  auto lookupOrigin = [](ArrayRef<StringAttr> dims, ArrayRef<int32_t> origin,
                         StringAttr dim) -> int32_t {
    auto it = llvm::find(dims, dim);
    if (it == dims.end())
      return 0;
    return origin[std::distance(dims.begin(), it)];
  };

  int32_t srcRow = lookupOrigin(srcInDims, srcQuery.origin, kRow);
  int32_t srcCol = lookupOrigin(srcInDims, srcQuery.origin, kCol);
  int32_t dstRow = lookupOrigin(dstInDims, dstQuery.origin, kRow);
  int32_t dstCol = lookupOrigin(dstInDims, dstQuery.origin, kCol);
  if (dstRow < srcRow || dstCol < srcCol)
    return std::nullopt;
  return (static_cast<uint32_t>(dstCol - srcCol) * bitwidth / 32) |
         (static_cast<uint32_t>(dstRow - srcRow) << 16);
}

uint32_t getTMemSubviewOffsetForLowering(gpu::MemDescSubsliceOp op) {
  auto srcTy = cast<MemDescType>(op.getSrc().getType());
  if (auto specialOffset = getCanonicalContiguous32x32SubviewOffset(op))
    return *specialOffset;
  if (isTensorMemoryColumnHalfDim0Slice(op))
    return 0;
  if (isTensorMemoryRowHalfDim0Slice(op)) {
    auto reshape = op.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
    auto rootTy = cast<MemDescType>(reshape.getSrc().getType());
    auto dstTy = cast<MemDescType>(op.getType());
    if (rootTy.getRank() == 2 && dstTy.getRank() == 3 &&
        rootTy.getShape()[0] == dstTy.getShape()[1] * 2 &&
        rootTy.getShape()[1] == dstTy.getShape()[2]) {
      uint32_t packetRowOffset = static_cast<uint32_t>(
          dstTy.getShape()[1] * rootTy.getElementTypeBitWidth() / 128);
      return packetRowOffset << 16;
    }
    SmallVector<int32_t> rootOffsets(rootTy.getRank(), 0);
    rootOffsets[0] = dstTy.getShape()[1];
    return getTMemViewOffsetForLowering(reshape.getSrc(), rootOffsets);
  }
  auto dstTy = cast<MemDescType>(op.getType());
  auto tmemSpace = TensorMemorySpaceAttr::get(op.getContext());
  bool isPureTMem2DColumnSubview =
      srcTy.getMemorySpace() == tmemSpace &&
      dstTy.getMemorySpace() == tmemSpace &&
      isTensorMemoryEncoding(srcTy.getEncoding()) &&
      !isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding()) &&
      srcTy.getRank() == 2 && dstTy.getRank() == 2 &&
      op.getOffsets().size() == 2 && op.getOffsets()[0] == 0 &&
      srcTy.getShape()[0] == dstTy.getShape()[0] &&
      srcTy.getShape()[1] >= dstTy.getShape()[1] &&
      op.getOffsets()[1] >= 0 &&
      op.getOffsets()[1] + dstTy.getShape()[1] <= srcTy.getShape()[1];
  if (isPureTMem2DColumnSubview) {
    std::string error;
    auto srcQuery = inferStandaloneTMemLdStQueryLayout(
        op.getSrc(), /*preserveNonCanonicalView=*/true, &error);
    auto dstQuery = inferStandaloneTMemLdStQueryLayout(
        op.getResult(), /*preserveNonCanonicalView=*/true, &error);
    if (succeeded(srcQuery) && succeeded(dstQuery)) {
      if (auto offset = getTMemLdStQueryOriginDeltaBaseOffset(
              *srcQuery, *dstQuery, srcTy.getElementTypeBitWidth()))
        return *offset;
    }
  }
  SmallVector<int32_t> offsets(op.getOffsets().begin(), op.getOffsets().end());
  if (srcTy.getMemorySpace() == tmemSpace &&
      dstTy.getMemorySpace() == tmemSpace &&
      isTensorMemoryEncoding(srcTy.getEncoding()) &&
      !isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
    std::string error;
    auto srcQuery = inferStandaloneTMemLdStQueryLayout(
        op.getSrc(), /*preserveNonCanonicalView=*/true, &error);
    if (succeeded(srcQuery)) {
      SmallVector<int64_t> srcShape(srcTy.getShape().begin(),
                                    srcTy.getShape().end());
      SmallVector<int64_t> dstShape(dstTy.getShape().begin(),
                                    dstTy.getShape().end());
      auto layoutRank = srcQuery->layout.getNumOutDims();
      auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
      if (extraRank >= 0) {
        if (auto leadingUnit = tryMakeLeadingUnitSubviewLayout(
                srcQuery->layout, ArrayRef<int64_t>(srcShape).drop_front(extraRank),
                ArrayRef<int64_t>(dstShape).drop_front(extraRank),
                ArrayRef<int32_t>(offsets).drop_front(extraRank),
                srcQuery->origin, &error)) {
          auto dstQuery = inferStandaloneTMemLdStQueryLayout(
              op.getResult(), /*preserveNonCanonicalView=*/true, &error);
          if (succeeded(dstQuery)) {
            if (auto offset = getTMemLdStQueryOriginDeltaBaseOffset(
                    *srcQuery, *dstQuery, srcTy.getElementTypeBitWidth()))
              return *offset;
          }
          auto leadingOrigin = addPrefixOffsetsToQueryOrigin(
              leadingUnit->layout, leadingUnit->origin,
              ArrayRef<int64_t>(srcShape).take_front(extraRank),
              ArrayRef<int32_t>(offsets).take_front(extraRank),
              srcTy.getElementTypeBitWidth());
          TMemLdStQueryLayout leadingQuery{leadingUnit->layout,
                                           srcQuery->twoCTAs, leadingOrigin};
          if (auto offset = getTMemLdStQueryOriginDeltaBaseOffset(
                  *srcQuery, leadingQuery, srcTy.getElementTypeBitWidth()))
            return *offset;
        }
      }
    }
  }
  return getTMemViewOffsetForLowering(op.getSrc(), offsets);
}

static FailureOr<MemDescType>
inferStandaloneTMemViewTypeImpl(Value memDesc, bool preserveNonCanonicalView,
                                std::string *error) {
  auto memDescTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!memDescTy || memDescTy.getMemorySpace() !=
                        TensorMemorySpaceAttr::get(memDesc.getContext())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }

  auto encoding = memDescTy.getEncoding();
  if (!isTensorMemoryEncoding(encoding))
    return memDescTy;
  if (isa<TensorMemoryScalesEncodingAttr>(encoding)) {
    if (preserveNonCanonicalView)
      return memDescTy;
    return MemDescType::get(memDescTy.getShape(), memDescTy.getElementType(),
                            encoding, memDescTy.getMemorySpace(),
                            memDescTy.getMutableMemory(),
                            memDescTy.getShape());
  }

  Operation *defOp = memDesc.getDefiningOp();
  bool hasDescriptorViewProducer =
      isa_and_nonnull<gpu::MemDescSubsliceOp, TMEMSubSliceOp,
                      gpu::MemDescIndexOp, gpu::MemDescReshapeOp,
                      gpu::MemDescTransOp, gpu::MemDescReinterpretOp>(defOp);
  auto layoutRank = static_cast<size_t>(cast<LayoutEncodingTrait>(encoding).getRank());
  if (preserveNonCanonicalView && memDescTy.getShape().size() >= layoutRank &&
      memDescTy.getAllocShape().size() >= layoutRank &&
      memDescTy.getShape().take_back(layoutRank) !=
          memDescTy.getAllocShape().take_back(layoutRank) &&
      !getCanonicalTMemLinearEncoding(memDescTy, /*error=*/nullptr) &&
      !hasDescriptorViewProducer) {
    return memDescTy;
  }

  auto makeStandaloneTy =
      [&](Attribute inferredEncoding) -> FailureOr<MemDescType> {
    auto layoutRank =
        static_cast<size_t>(cast<LayoutEncodingTrait>(inferredEncoding).getRank());
    auto logicalShape = memDescTy.getShape().take_back(layoutRank);
    auto logicalShapeMatchesLayout = [&](const LinearLayout &layout) {
      if (layout.getNumInDims() == 0)
        return true;
      auto *layoutCtx = (*layout.getInDimNames().begin()).getContext();
      auto kRow = StringAttr::get(layoutCtx, "row");
      auto kCol = StringAttr::get(layoutCtx, "col");
      auto kBlock = StringAttr::get(layoutCtx, "block");
      if (layoutRank >= 1 && layout.hasInDim(kRow) &&
          layout.getInDimSize(kRow) != logicalShape[0])
        return false;
      if (layoutRank >= 2 && layout.hasInDim(kCol) &&
          layout.getInDimSize(kCol) != logicalShape[1])
        return false;
      if (layoutRank >= 3 && layout.hasInDim(kBlock) &&
          layout.getInDimSize(kBlock) != logicalShape[2])
        return false;
      return true;
    };
    auto restrictLayoutInputsToShape = [&](LinearLayout layout) {
      if (layout.getNumInDims() == 0)
        return layout;
      auto *layoutCtx = (*layout.getInDimNames().begin()).getContext();
      auto kRow = StringAttr::get(layoutCtx, "row");
      auto kCol = StringAttr::get(layoutCtx, "col");
      auto kBlock = StringAttr::get(layoutCtx, "block");
      if (layoutRank >= 1 && layout.hasInDim(kRow) &&
          logicalShape[0] <= layout.getInDimSize(kRow))
        layout = layout.resizeInDim(kRow, logicalShape[0]);
      if (layoutRank >= 2 && layout.hasInDim(kCol) &&
          logicalShape[1] <= layout.getInDimSize(kCol))
        layout = layout.resizeInDim(kCol, logicalShape[1]);
      if (layoutRank >= 3 && layout.hasInDim(kBlock) &&
          logicalShape[2] <= layout.getInDimSize(kBlock))
        layout = layout.resizeInDim(kBlock, logicalShape[2]);
      return layout;
    };

    std::string localError;
    if (auto maybeCanonical = getCanonicalTMemLinearEncoding(
            memDescTy.getShape(), inferredEncoding, &localError)) {
      if (logicalShapeMatchesLayout(maybeCanonical->getLinearLayout())) {
        return MemDescType::get(memDescTy.getShape(), memDescTy.getElementType(),
                                *maybeCanonical, memDescTy.getMemorySpace(),
                                memDescTy.getMutableMemory(),
                                memDescTy.getShape());
      }
    }
    auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
        memDescTy.getShape(), inferredEncoding, &localError);
    if (!maybeAnalysis) {
      if (error && error->empty())
        *error = localError;
      return failure();
    }
    auto standaloneLayout =
        preserveNonCanonicalView && !logicalShapeMatchesLayout(*maybeAnalysis)
            ? *maybeAnalysis
            : restrictLayoutInputsToShape(*maybeAnalysis);
    auto twoCTAs = getTensorMemoryTwoCTAs(inferredEncoding).value_or(false);
    auto maybeStandaloneEncoding = tryMakeTensorMemoryLinearEncoding(
        memDesc.getContext(), std::move(standaloneLayout), twoCTAs, &localError);
    if (!maybeStandaloneEncoding) {
      if (error && error->empty())
        *error = localError;
      return failure();
    }
    return MemDescType::get(memDescTy.getShape(), memDescTy.getElementType(),
                            *maybeStandaloneEncoding,
                            memDescTy.getMemorySpace(),
                            memDescTy.getMutableMemory(),
                            memDescTy.getShape());
  };

  auto rank = static_cast<size_t>(cast<LayoutEncodingTrait>(encoding).getRank());

  if (auto subslice = memDesc.getDefiningOp<gpu::MemDescSubsliceOp>()) {
    auto srcTy = inferStandaloneTMemViewTypeImpl(subslice.getSrc(),
                                                 preserveNonCanonicalView,
                                                 error);
    if (failed(srcTy))
      return failure();
    auto srcLayoutRank = static_cast<size_t>(
        cast<LayoutEncodingTrait>(srcTy->getEncoding()).getRank());
    bool preservesTrailingLayoutDims =
        srcTy->getShape().size() >= srcLayoutRank &&
        memDescTy.getShape().size() >= srcLayoutRank &&
        srcTy->getShape().take_back(srcLayoutRank) ==
            memDescTy.getShape().take_back(srcLayoutRank) &&
        subslice.getOffsets().size() >= srcLayoutRank &&
        llvm::all_of(subslice.getOffsets().take_back(srcLayoutRank),
                     [](int64_t offset) { return offset == 0; });
    if (preservesTrailingLayoutDims) {
      if (auto preserved = tryPreserveExactTMemViewEncoding(
              srcTy->getContext(), memDescTy.getShape(),
              memDescTy.getAllocShape(), srcTy->getEncoding(), error)) {
        auto preservedTy = tryCreateMemDescType(
            srcTy->getContext(), memDescTy.getShape(),
            memDescTy.getElementType(), *preserved, memDescTy.getMemorySpace(),
            memDescTy.getMutableMemory(), memDescTy.getAllocShape(), error);
        if (!preservedTy)
          return failure();
        return *preservedTy;
      }
    }
    auto maybeEncoding = inferTMemSubsliceEncoding(
        srcTy->getShape(), srcTy->getEncoding(), memDescTy.getShape(),
        subslice.getOffsets(), error);
    if (failed(maybeEncoding))
      return failure();
    return makeStandaloneTy(*maybeEncoding);
  }
  if (auto subslice = memDesc.getDefiningOp<TMEMSubSliceOp>()) {
    auto srcTy = inferStandaloneTMemViewTypeImpl(subslice.getSrc(),
                                                 preserveNonCanonicalView,
                                                 error);
    if (failed(srcTy))
      return failure();
    if (auto preserved = tryPreserveExactTMemViewEncoding(
            srcTy->getContext(), memDescTy.getShape(), memDescTy.getAllocShape(),
            srcTy->getEncoding(), error)) {
      auto preservedTy = tryCreateMemDescType(
          srcTy->getContext(), memDescTy.getShape(), memDescTy.getElementType(),
          *preserved, memDescTy.getMemorySpace(), memDescTy.getMutableMemory(),
          memDescTy.getAllocShape(), error);
      if (!preservedTy)
        return failure();
      return *preservedTy;
    }
    SmallVector<int32_t> offsets(srcTy->getRank(), 0);
    if (!offsets.empty())
      offsets.back() = subslice.getN();
    auto maybeEncoding = inferTMemSubsliceEncoding(
        srcTy->getShape(), srcTy->getEncoding(), memDescTy.getShape(),
        offsets, error);
    if (failed(maybeEncoding))
      return failure();
    return makeStandaloneTy(*maybeEncoding);
  }
  if (auto index = memDesc.getDefiningOp<gpu::MemDescIndexOp>()) {
    auto srcTy = inferStandaloneTMemViewTypeImpl(index.getSrc(),
                                                 preserveNonCanonicalView,
                                                 error);
    if (failed(srcTy))
      return failure();
    if (auto preserved = tryPreserveOuterIndexedTMemEncoding(
            srcTy->getContext(), srcTy->getShape(), memDescTy.getShape(),
            memDescTy.getShape(), srcTy->getEncoding(), error)) {
      return makeStandaloneTy(*preserved);
    }
    auto maybeEncoding = inferTMemIndexEncoding(
        srcTy->getShape(), memDescTy.getShape(), memDescTy.getShape(),
        srcTy->getEncoding(), error);
    if (failed(maybeEncoding))
      return failure();
    return makeStandaloneTy(*maybeEncoding);
  }
  if (auto reshape = memDesc.getDefiningOp<gpu::MemDescReshapeOp>()) {
    auto srcTy = inferStandaloneTMemViewTypeImpl(reshape.getSrc(),
                                                 preserveNonCanonicalView,
                                                 error);
    if (failed(srcTy))
      return failure();
    auto reshapedTy =
        inferTMemReshapeOpType(*srcTy, memDescTy.getShape(), error);
    if (failed(reshapedTy))
      return failure();
    return makeStandaloneTy(reshapedTy->getEncoding());
  }
  if (memDesc.getDefiningOp<gpu::MemDescTransOp>() ||
      memDesc.getDefiningOp<gpu::MemDescReinterpretOp>())
    return makeStandaloneTy(memDescTy.getEncoding());

  if (memDescTy.getShape().take_back(rank) ==
      memDescTy.getAllocShape().take_back(rank)) {
    return memDescTy;
  }

  if (error)
    *error = "unsupported tensor memory descriptor view for reg-layout query";
  return failure();
}

FailureOr<MemDescType>
inferStandaloneTMemRegLayoutQueryType(Value memDesc, std::string *error) {
  return inferStandaloneTMemViewTypeImpl(
      memDesc, /*preserveNonCanonicalView=*/true, error);
}

FailureOr<TMemLdStQueryLayout>
inferStandaloneTMemLdStQueryLayout(Value memDesc,
                                   bool preserveNonCanonicalView,
                                   std::string *error) {
  auto maybeQuery = inferStandaloneTMemLdStQueryLayoutImpl(
      memDesc, preserveNonCanonicalView, error);
  if (failed(maybeQuery))
    return failure();
  auto &query = *maybeQuery;
  auto *ctx = memDesc.getContext();
  auto outDimNames = standardOutDimNames(ctx, query.layout.getNumOutDims());
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.reserve(query.layout.getNumOutDims());
  for (auto [idx, size] : llvm::enumerate(query.layout.getOutDimSizes()))
    outDims.emplace_back(outDimNames[idx], static_cast<int32_t>(size));
  query.layout = LinearLayout(query.layout.getBases(), std::move(outDims),
                              query.layout.isSurjective());
  return query;
}

static std::optional<TMemLdStQueryLayout>
getGenericTMemLdStReshapedSupportQueryLayout(Value memDesc,
                                             std::string *error) {
  auto queryTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!queryTy || queryTy.getRank() != 2)
    return std::nullopt;
  if (!isTensorMemoryEncoding(queryTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(queryTy.getEncoding()))
    return std::nullopt;
  Operation *defOp = memDesc.getDefiningOp();
  if (!isa_and_nonnull<gpu::MemDescSubsliceOp, TMEMSubSliceOp,
                       gpu::MemDescIndexOp, gpu::MemDescReshapeOp,
                       gpu::MemDescTransOp, gpu::MemDescReinterpretOp>(defOp))
    return std::nullopt;

  std::string rawError;
  auto maybeSrcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
      memDesc, /*preserveNonCanonicalView=*/true, &rawError);
  if (failed(maybeSrcQuery))
    return std::nullopt;

  auto *ctx = queryTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  auto normalizedSrcLayout =
      normalizeTensorMemoryLinearLayoutForAnalysis(maybeSrcQuery->layout);
  if (!normalizedSrcLayout.hasInDim(kRow) || !normalizedSrcLayout.hasInDim(kCol))
    return std::nullopt;
  auto normalizedSrcQuery = TMemLdStQueryLayout{
      normalizedSrcLayout, maybeSrcQuery->twoCTAs,
      remapTMemLdStQueryOrigin(*maybeSrcQuery, normalizedSrcLayout,
                               /*deltaCoords=*/{})};

  int64_t logicalRows = queryTy.getShape()[0];
  int64_t logicalCols = queryTy.getShape()[1];
  int bitwidth = queryTy.getElementTypeBitWidth();
  int64_t supportRows = normalizedSrcLayout.getInDimSize(kRow);
  int64_t supportCols =
      normalizedSrcLayout.getInDimSize(kCol) / (32 / bitwidth);
  if (supportRows <= logicalRows || supportCols <= 0)
    return std::nullopt;

  if (logicalCols <= supportCols)
    return std::nullopt;
  if (logicalRows * logicalCols != supportRows * supportCols)
    return std::nullopt;

  gpu::CGAEncodingAttr cga =
      gpu::CGAEncodingAttr::get1CTALayout(ctx, queryTy.getRank());
  bool twoCTAs = false;
  if (auto linear =
          dyn_cast<TensorMemoryLinearEncodingAttr>(queryTy.getEncoding())) {
    cga = linear.getCGALayout();
    twoCTAs = linear.getTwoCTAs();
  } else if (auto legacy =
                 dyn_cast<TensorMemoryEncodingAttr>(queryTy.getEncoding())) {
    cga = legacy.getCGALayout();
    twoCTAs = legacy.getTwoCTAs();
  } else {
    return std::nullopt;
  }

  auto reshaped = reshapeLayout(ctx, normalizedSrcLayout, queryTy.getShape());
  return TMemLdStQueryLayout{
      reshaped, twoCTAs,
      remapTMemLdStQueryOrigin(normalizedSrcQuery, reshaped,
                               /*deltaCoords=*/{})};
}

static std::optional<TMemLdStQueryLayout>
getHalfRowsTMemLdStSupportQueryLayout(Value memDesc, std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_HALFROWS") != nullptr;
  auto queryTy = dyn_cast<MemDescType>(memDesc.getType());
  auto index = memDesc.getDefiningOp<gpu::MemDescIndexOp>();
  if (!queryTy || !index) {
    if (debug)
      llvm::errs() << "[halfrows-support] reject: not a memdesc_index view\n";
    return std::nullopt;
  }

  APInt indexValue;
  if (!matchPattern(index.getIndex(), m_ConstantInt(&indexValue)) ||
      indexValue.getSExtValue() != 0) {
    if (debug)
      llvm::errs() << "[halfrows-support] reject: index is not constant zero\n";
    return std::nullopt;
  }

  auto subslice = index.getSrc().getDefiningOp<gpu::MemDescSubsliceOp>();
  if (!subslice) {
    if (debug)
      llvm::errs() << "[halfrows-support] reject: source is not memdesc_subslice\n";
    return std::nullopt;
  }
  if (subslice.getOffsets().size() != 3 || subslice.getOffsets()[0] != 1 ||
      subslice.getOffsets()[1] != 0 || subslice.getOffsets()[2] != 0) {
    if (debug)
      llvm::errs() << "[halfrows-support] reject: subslice offsets are not [1,0,0]\n";
    return std::nullopt;
  }

  auto reshape = subslice.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
  if (!reshape) {
    if (debug)
      llvm::errs() << "[halfrows-support] reject: subslice source is not memdesc_reshape\n";
    return std::nullopt;
  }

  auto reshapeTy = dyn_cast<MemDescType>(reshape.getType());
  auto srcTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
  if (!reshapeTy || !srcTy || queryTy.getRank() != 2 || reshapeTy.getRank() != 3 ||
      srcTy.getRank() != 2) {
    if (debug) {
      llvm::errs() << "[halfrows-support] reject: rank mismatch query="
                   << (queryTy ? queryTy.getRank() : -1) << " reshape="
                   << (reshapeTy ? reshapeTy.getRank() : -1) << " src="
                   << (srcTy ? srcTy.getRank() : -1) << "\n";
    }
    return std::nullopt;
  }

  int64_t rows = queryTy.getShape()[0];
  int64_t cols = queryTy.getShape()[1];
  if (reshapeTy.getShape()[0] != 2 || reshapeTy.getShape()[1] != rows ||
      reshapeTy.getShape()[2] != cols || (cols % 2) != 0) {
    if (debug) {
      llvm::errs() << "[halfrows-support] reject: shape mismatch query="
                   << queryTy << " reshape=" << reshapeTy << " src=" << srcTy
                   << "\n";
    }
    return std::nullopt;
  }
  if (srcTy.getShape()[0] != rows * 2 || srcTy.getShape()[1] != cols) {
    if (debug) {
      llvm::errs() << "[halfrows-support] reject: source tile is not a row-half split\n";
      llvm::errs() << "[halfrows-support] src shape=" << srcTy.getShape()[0]
                   << "x" << srcTy.getShape()[1] << " query=" << rows << "x"
                   << cols << "\n";
    }
    return std::nullopt;
  }

  auto *ctx = memDesc.getContext();
  auto maybeSupportLayout = getTMemViewAnalysisLinearLayout(
      srcTy.getShape(), srcTy.getEncoding(), error);
  if (!maybeSupportLayout) {
    if (debug)
      llvm::errs() << "[halfrows-support] reject: backing layout failed\n";
    return std::nullopt;
  }

  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto supportLayout = normalizeTensorMemoryLinearLayoutForAnalysis(
      *maybeSupportLayout);
  if (!supportLayout.hasInDim(kRow) || !supportLayout.hasInDim(kCol) ||
      supportLayout.getInDimSize(kRow) != rows * 2 ||
      supportLayout.getInDimSize(kCol) != cols) {
    if (debug) {
      llvm::errs() << "[halfrows-support] reject: backing query layout mismatch\n"
                   << supportLayout.toString() << "\n";
    }
    return std::nullopt;
  }

  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding());
  auto result = TMemLdStQueryLayout{
      supportLayout, maybeTwoCTAs.value_or(false),
      remapTMemLdStQueryOrigin(
          TMemLdStQueryLayout{
              supportLayout, maybeTwoCTAs.value_or(false),
              SmallVector<int32_t>(supportLayout.getNumInDims(), 0)},
          supportLayout, {{kRow, static_cast<int32_t>(rows)}})};
  if (debug) {
    llvm::errs() << "[halfrows-support] support layout:\n"
                 << result.layout.toString() << "\n";
    llvm::errs() << "[halfrows-support] origin:";
    for (int32_t value : result.origin)
      llvm::errs() << " " << value;
    llvm::errs() << "\n";
  }
  return result;
}

static bool isExactCanonicalContiguous32x32TMemViewType(MemDescType memTy) {
  if (!memTy || memTy.getRank() != 2 || memTy.getShape()[0] != 32 ||
      memTy.getShape()[1] != 32 ||
      !isTensorMemoryEncoding(memTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
    return false;
  }
  auto linear =
      dyn_cast<TensorMemoryLinearEncodingAttr>(memTy.getEncoding());
  if (!linear)
    return false;
  auto ll = linear.getLinearLayout();
  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) || ll.getNumInDims() != 2 ||
      ll.getNumOutDims() != 2 || ll.getInDimSize(kRow) != 32 ||
      ll.getInDimSize(kCol) != 32) {
    return false;
  }
  SmallVector<int64_t> expectedOutShape{32, 32};
  if (!llvm::equal(ll.getOutDimSizes(), expectedOutShape))
    return false;
  for (unsigned idx = 0; idx < ll.getInDimSizeLog2(kRow); ++idx) {
    auto basis = ll.getBasis(kRow, idx);
    if (basis.size() != 2 || basis[0] != (1 << idx) || basis[1] != 0)
      return false;
  }
  for (unsigned idx = 0; idx < ll.getInDimSizeLog2(kCol); ++idx) {
    auto basis = ll.getBasis(kCol, idx);
    if (basis.size() != 2 || basis[0] != 0 || basis[1] != (1 << idx))
      return false;
  }
  return true;
}

static std::optional<std::string>
getUnsupportedTMemLdStDescriptorViewRowAnchorReason(
    Value memDesc, MemDescType memTy, std::optional<TMemLdStRowPlan> rowPlan) {
  if (!memTy || !rowPlan)
    return std::nullopt;

  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  auto squeezeTrivialBlock = [&](LinearLayout layout) {
    if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1)
      layout = layout.squeezeIns(kBlock);
    if (layout.hasOutDim(kBlock) && layout.getOutDimSize(kBlock) == 1)
      layout = layout.squeezeOuts(kBlock);
    return layout;
  };

  auto maybeMemLayout = [&]() -> std::optional<LinearLayout> {
    std::string queryError;
    if (auto maybeQuery = inferStandaloneTMemLdStQueryLayoutImpl(
            memDesc, /*preserveNonCanonicalView=*/true, &queryError);
        succeeded(maybeQuery)) {
      return squeezeTrivialBlock(maybeQuery->layout);
    }

    std::string analysisError;
    if (auto maybeAnalysisLayout = getTMemViewAnalysisLinearLayout(
            memTy.getShape(), memTy.getEncoding(), &analysisError)) {
      return squeezeTrivialBlock(
          normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysisLayout));
    }

    std::string canonicalError;
    if (auto maybeCanonical =
            getCanonicalTMemLinearEncoding(memTy, &canonicalError)) {
      return squeezeTrivialBlock(normalizeTensorMemoryLinearLayoutForAnalysis(
          maybeCanonical->getLinearLayout()));
    }
    return std::nullopt;
  }();
  if (!maybeMemLayout || !maybeMemLayout->hasInDim(kRow) ||
      !maybeMemLayout->hasInDim(kCol)) {
    return std::nullopt;
  }

  if (getLogicalRowAnchorBasis(*maybeMemLayout, rowPlan->warpRow0) &&
      getLogicalRowAnchorBasis(*maybeMemLayout, rowPlan->warpRow1))
    return std::nullopt;

  return std::string(
             "unsupported tensor memory descriptor view for direct tcgen05.ld/st: "
             "required row anchors ") +
         std::to_string(rowPlan->warpRow0) + "," +
         std::to_string(rowPlan->warpRow1) +
         " are not directly representable in the descriptor view. Access the "
         "full backing tile or reshape/copy so the TMEM row anchors stay "
         "materializable.";
}

bool isUnsupportedDirectTMemLdStDescriptorView(Value memDesc,
                                               std::string *error) {
  auto unsupported = [&](StringRef reason) {
    if (error)
      *error = reason.str();
    return true;
  };

  auto queryTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!queryTy || queryTy.getRank() != 2 ||
      !isTensorMemoryEncoding(queryTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(queryTy.getEncoding())) {
    return false;
  }
  bool explicitViewProducer =
      isa_and_nonnull<gpu::MemDescSubsliceOp, TMEMSubSliceOp, gpu::MemDescIndexOp,
                      gpu::MemDescReshapeOp, gpu::MemDescTransOp,
                      gpu::MemDescReinterpretOp>(memDesc.getDefiningOp());
  if (explicitViewProducer &&
      isExactCanonicalContiguous32x32TMemViewType(queryTy)) {
    return unsupported("unsupported tensor memory descriptor view for direct "
                       "TMEM load/store: exact canonical 32x32 subviews from "
                       "larger TMEM tiles are not directly representable");
  }
  if (isDirectHalfRowsSubview(memDesc) ||
      isHigherRankHalfRowsSubview(memDesc)) {
    return unsupported("unsupported tensor memory descriptor view for direct "
                       "tcgen05.ld/st: lifted row-half TMEM views translate "
                       "the TMEM row origin and are not directly realizable by "
                       "tcgen05.ld/st packets. Access the full backing tile or "
                       "reshape/copy so the TMEM rows stay materializable.");
  }

  std::string supportError;
  if (getTMemLdStSupportQueryPlan(memDesc, &supportError)) {
    return false;
  }

  auto hasZeroBasisAlong = [](const LinearLayout &layout, StringAttr dim) {
    if (!layout.hasInDim(dim))
      return false;
    for (unsigned idx = 0; idx < layout.getInDimSizeLog2(dim); ++idx) {
      if (llvm::all_of(layout.getBasis(dim, idx),
                       [](int32_t value) { return value == 0; }))
        return true;
    }
    return false;
  };
  auto hasTwoCTATensorMemoryScalesRoot = [&]() {
    Value cur = memDesc;
    while (cur) {
      if (auto curTy = dyn_cast<MemDescType>(cur.getType())) {
        if (auto scales =
                dyn_cast<TensorMemoryScalesEncodingAttr>(curTy.getEncoding())) {
          return product<unsigned>(scales.getCGALayout().getCTAsPerCGA()) > 1;
        }
      }
      if (auto forwarded = getTMemForwardingSource(cur)) {
        cur = forwarded;
        continue;
      }
      Operation *def = cur.getDefiningOp();
      if (!def)
        break;
      if (auto op = dyn_cast<gpu::MemDescIndexOp>(def)) {
        cur = op.getSrc();
        continue;
      }
      if (auto op = dyn_cast<gpu::MemDescSubsliceOp>(def)) {
        cur = op.getSrc();
        continue;
      }
      if (auto op = dyn_cast<TMEMSubSliceOp>(def)) {
        cur = op.getSrc();
        continue;
      }
      if (auto op = dyn_cast<gpu::MemDescReshapeOp>(def)) {
        cur = op.getSrc();
        continue;
      }
      if (auto op = dyn_cast<gpu::MemDescTransOp>(def)) {
        cur = op.getSrc();
        continue;
      }
      if (auto op = dyn_cast<gpu::MemDescReinterpretOp>(def)) {
        cur = op.getSrc();
        continue;
      }
      break;
    }
    return false;
  };
  auto linearQuery =
      dyn_cast<TensorMemoryLinearEncodingAttr>(queryTy.getEncoding());
  bool isTwoCTAInt8LinearDescriptorView =
      linearQuery && linearQuery.getTwoCTAs() &&
      queryTy.getElementTypeBitWidth() == 8;
  if (hasTwoCTATensorMemoryScalesRoot() || isTwoCTAInt8LinearDescriptorView) {
    std::string rawQueryError;
    auto rawQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        memDesc, /*preserveNonCanonicalView=*/true, &rawQueryError);
    auto *ctx = queryTy.getContext();
    auto kRow = StringAttr::get(ctx, "row");
    auto kCol = StringAttr::get(ctx, "col");
    auto kBlock = StringAttr::get(ctx, "block");
    auto typeLayout = toLinearLayout(queryTy);
    bool hasNonTrivialBlock = typeLayout.hasInDim(kBlock) &&
                              typeLayout.getInDimSize(kBlock) > 1;
    if (!hasNonTrivialBlock && succeeded(rawQuery)) {
      hasNonTrivialBlock = rawQuery->layout.hasInDim(kBlock) &&
                           rawQuery->layout.getInDimSize(kBlock) > 1;
    }
    if (failed(rawQuery) ||
        (hasNonTrivialBlock && (hasZeroBasisAlong(typeLayout, kRow) ||
                                hasZeroBasisAlong(typeLayout, kCol)))) {
      return unsupported(
          "unsupported tensor memory descriptor view for direct tcgen05.ld/st: "
          "two-CTA 8-bit descriptor views with broadcast/support bases must "
          "keep the exact physical TMEM projection. Access the canonical "
          "descriptor or copy/reshape through a directly supported layout "
          "instead of using a standalone query-type fallback.");
    }
  }

  auto queryPlan = getTMemLdStRowPlanForType(queryTy);
  auto backingPlan = getBackingTMemLdStRowPlan(memDesc);
  if (shouldPreferDirectHalfRowsSubviewRowPlan(memDesc, queryTy, queryPlan,
                                               backingPlan)) {
    return false;
  }

  auto rowPlan = getTMemLdStRowPlanForQuery(memDesc, queryTy);
  if (!rowPlan)
    rowPlan = backingPlan;
  if (auto anchorReason =
          getUnsupportedTMemLdStDescriptorViewRowAnchorReason(memDesc, queryTy,
                                                              rowPlan)) {
    return unsupported(*anchorReason);
  }

  auto rejectHalfRowsView = [&](MemDescType srcTy) {
    return srcTy && srcTy.getRank() == 2 &&
           srcTy.getShape()[0] == queryTy.getShape()[0] * 2 &&
           srcTy.getShape()[1] == queryTy.getShape()[1] &&
           isTensorMemoryEncoding(srcTy.getEncoding()) &&
           !isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding());
  };

  if (auto subslice = memDesc.getDefiningOp<gpu::MemDescSubsliceOp>()) {
    auto srcTy = dyn_cast<MemDescType>(subslice.getSrc().getType());
    auto offsets = subslice.getOffsets();
    if (rejectHalfRowsView(srcTy) && offsets.size() == 2 &&
        offsets[0] == queryTy.getShape()[0] && offsets[1] == 0) {
      return unsupported("unsupported tensor memory descriptor view for "
                         "direct tcgen05.ld/st: lifted row-half TMEM views "
                         "translate the TMEM row origin and are not directly "
                         "realizable by tcgen05.ld/st packets. Access the "
                         "full backing tile or reshape/copy so the TMEM rows "
                         "stay materializable.");
    }
    return false;
  }

  auto index = memDesc.getDefiningOp<gpu::MemDescIndexOp>();
  if (!index)
    return false;

  APInt indexValue;
  if (!matchPattern(index.getIndex(), m_ConstantInt(&indexValue)) ||
      indexValue.getSExtValue() != 0) {
    return false;
  }

  auto subslice = index.getSrc().getDefiningOp<gpu::MemDescSubsliceOp>();
  if (!subslice)
    return false;
  auto offsets = subslice.getOffsets();
  if (offsets.size() != 3 || offsets[0] != 1 || offsets[1] != 0 ||
      offsets[2] != 0) {
    return false;
  }

  auto reshape = subslice.getSrc().getDefiningOp<gpu::MemDescReshapeOp>();
  if (!reshape)
    return false;
  auto reshapeTy = dyn_cast<MemDescType>(reshape.getType());
  auto srcTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
  if (!reshapeTy || !rejectHalfRowsView(srcTy) || reshapeTy.getRank() != 3)
    return false;
  if (reshapeTy.getShape()[0] != 2 || reshapeTy.getShape()[1] != queryTy.getShape()[0] ||
      reshapeTy.getShape()[2] != queryTy.getShape()[1]) {
    return false;
  }

  return unsupported("unsupported tensor memory descriptor view for direct "
                     "tcgen05.ld/st: lifted row-half TMEM views translate "
                     "the TMEM row origin and are not directly realizable by "
                     "tcgen05.ld/st packets. Access the full backing tile or "
                     "reshape/copy so the TMEM rows stay materializable.");
}

static std::optional<TMemLdStQueryLayout>
getDirectHalfRowsTMemLdStSupportQueryLayout(Value memDesc, std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_HALFROWS") != nullptr;
  auto queryTy = dyn_cast<MemDescType>(memDesc.getType());
  auto subslice = memDesc.getDefiningOp<gpu::MemDescSubsliceOp>();
  if (!queryTy || !subslice || queryTy.getRank() != 2)
    return std::nullopt;

  auto srcTy = dyn_cast<MemDescType>(subslice.getSrc().getType());
  if (!srcTy || srcTy.getRank() != 2)
    return std::nullopt;

  int64_t rows = queryTy.getShape()[0];
  int64_t cols = queryTy.getShape()[1];
  auto offsets = subslice.getOffsets();
  if (offsets.size() != 2 || offsets[0] != rows || offsets[1] != 0)
    return std::nullopt;
  if (srcTy.getShape()[0] != rows * 2 || srcTy.getShape()[1] != cols)
    return std::nullopt;

  auto *ctx = memDesc.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto maybeSupportLayout = getTMemViewAnalysisLinearLayout(
      srcTy.getShape(), srcTy.getEncoding(), error);
  if (!maybeSupportLayout)
    return std::nullopt;
  auto projectedQueryLayout = normalizeTensorMemoryLinearLayoutForAnalysis(
      *maybeSupportLayout);
  if (!projectedQueryLayout.hasInDim(kRow) ||
      !projectedQueryLayout.hasInDim(kCol) ||
      projectedQueryLayout.getInDimSize(kRow) != rows * 2 ||
      projectedQueryLayout.getInDimSize(kCol) != cols)
    return std::nullopt;

  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding());
  auto result = TMemLdStQueryLayout{
      projectedQueryLayout,
      maybeTwoCTAs.value_or(false),
      remapTMemLdStQueryOrigin(
          TMemLdStQueryLayout{
              projectedQueryLayout, maybeTwoCTAs.value_or(false),
              SmallVector<int32_t>(projectedQueryLayout.getNumInDims(), 0)},
          projectedQueryLayout,
          {{kRow, static_cast<int32_t>(rows)}})};
  if (debug) {
    llvm::errs() << "[halfrows-direct-support] projected query layout:\n"
                 << result.layout.toString() << "\n";
    llvm::errs() << "[halfrows-direct-support] origin:";
    for (int32_t value : result.origin)
      llvm::errs() << " " << value;
    llvm::errs() << "\n";
  }
  return result;
}

static std::optional<TMemLdStQueryLayout>
reinterpretColumnSubviewSupportQueryLayout(
    ArrayRef<int64_t> srcShape, int srcBitwidth,
    const TMemLdStQueryLayout &srcSupport, ArrayRef<int64_t> dstShape,
    int dstBitwidth, std::string *error) {
  if (srcShape.size() != 2 || dstShape.size() != 2 || srcShape[0] != dstShape[0] ||
      product<int64_t>(srcShape) * srcBitwidth !=
          product<int64_t>(dstShape) * dstBitwidth) {
    return std::nullopt;
  }
  if (srcBitwidth == dstBitwidth)
    return srcSupport;
  SmallVector<int64_t> supportSrcShape;
  supportSrcShape.reserve(srcSupport.layout.getNumOutDims());
  for (int32_t size : srcSupport.layout.getOutDimSizes())
    supportSrcShape.push_back(static_cast<int64_t>(size));
  if (supportSrcShape.size() != 2 || supportSrcShape[0] != srcShape[0] ||
      supportSrcShape[1] < srcShape[1]) {
    return std::nullopt;
  }
  SmallVector<int64_t> supportDstShape(supportSrcShape.begin(),
                                       supportSrcShape.end());
  if (srcBitwidth > dstBitwidth) {
    int64_t ratio = srcBitwidth / dstBitwidth;
    if (srcBitwidth % dstBitwidth != 0 || !llvm::isPowerOf2_64(ratio)) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return std::nullopt;
    }
    supportDstShape.back() *= ratio;
  } else {
    int64_t ratio = dstBitwidth / srcBitwidth;
    if (dstBitwidth % srcBitwidth != 0 || !llvm::isPowerOf2_64(ratio) ||
        supportDstShape.back() % ratio != 0) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return std::nullopt;
    }
    supportDstShape.back() /= ratio;
  }
  if (supportDstShape[0] != dstShape[0] || supportDstShape[1] < dstShape[1]) {
    if (error)
      *error = "unsupported tensor memory memdesc_reinterpret view";
    return std::nullopt;
  }
  auto *ctx = srcSupport.layout.getInDimNames().begin()->getContext();
  auto kCol = StringAttr::get(ctx, "col");
  auto bases = srcSupport.layout.getBases();
  auto colIt = bases.find(kCol);
  if (colIt == bases.end())
    return std::nullopt;

  auto inDims = llvm::to_vector(srcSupport.layout.getInDimNames());
  auto origin = srcSupport.origin;
  auto originIt = llvm::find(inDims, kCol);
  if (originIt == inDims.end())
    return std::nullopt;
  size_t originIdx = std::distance(inDims.begin(), originIt);

  auto zeroBasis = [&]() {
    return std::vector<int32_t>(srcSupport.layout.getNumOutDims(), 0);
  };
  if (srcBitwidth > dstBitwidth) {
    int64_t ratio = srcBitwidth / dstBitwidth;
    if (srcBitwidth % dstBitwidth != 0 || !llvm::isPowerOf2_64(ratio)) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return std::nullopt;
    }
    unsigned extraBits = llvm::Log2_64(ratio);
    colIt->second.insert(colIt->second.begin(), extraBits, zeroBasis());
    origin[originIdx] *= static_cast<int32_t>(ratio);
  } else {
    int64_t ratio = dstBitwidth / srcBitwidth;
    if (dstBitwidth % srcBitwidth != 0 || !llvm::isPowerOf2_64(ratio)) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return std::nullopt;
    }
    unsigned removeBits = llvm::Log2_64(ratio);
    if (removeBits > colIt->second.size()) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return std::nullopt;
    }
    for (unsigned i = 0; i < removeBits; ++i) {
      if (!llvm::all_of(colIt->second[i],
                        [](int32_t value) { return value == 0; })) {
        if (error)
          *error = "unsupported tensor memory memdesc_reinterpret view";
        return std::nullopt;
      }
    }
    colIt->second.erase(colIt->second.begin(), colIt->second.begin() +
                                                removeBits);
    if (origin[originIdx] % static_cast<int32_t>(ratio) != 0) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return std::nullopt;
    }
    origin[originIdx] /= static_cast<int32_t>(ratio);
  }
  if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
    llvm::errs() << "[tmem-ldst] reinterpret support srcShape="
                 << supportSrcShape[0] << "x" << supportSrcShape[1]
                 << " dstShape=" << supportDstShape[0] << "x"
                 << supportDstShape[1] << " viewSrc=" << srcShape[0] << "x"
                 << srcShape[1] << " viewDst=" << dstShape[0] << "x"
                 << dstShape[1] << "\n";
    llvm::errs() << "[tmem-ldst] reinterpret support src layout:\n"
                 << srcSupport.layout.toString() << "\n";
    llvm::errs() << "[tmem-ldst] reinterpret support src origin:";
    for (int32_t value : srcSupport.origin)
      llvm::errs() << " " << value;
    llvm::errs() << "\n";
  }
  auto outDimNames = standardOutDimNames(ctx, supportDstShape.size());
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.reserve(supportDstShape.size());
  for (auto [idx, size] : llvm::enumerate(supportDstShape))
    outDims.emplace_back(outDimNames[idx], static_cast<int32_t>(size));
  auto result = TMemLdStQueryLayout{
      LinearLayout(std::move(bases), std::move(outDims),
                   /*requireSurjective=*/false),
      srcSupport.twoCTAs, std::move(origin)};
  if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
    llvm::errs() << "[tmem-ldst] reinterpret support dst layout:\n"
                 << result.layout.toString() << "\n";
    llvm::errs() << "[tmem-ldst] reinterpret support dst origin:";
    for (int32_t value : result.origin)
      llvm::errs() << " " << value;
    llvm::errs() << "\n";
  }
  return result;
}

static std::optional<TMemLdStSupportQueryPlan>
getGenericTMemLdStReshapedSupportQueryPlan(Value memDesc,
                                           std::string *error) {
  if (auto query = getGenericTMemLdStReshapedSupportQueryLayout(memDesc, error))
    return TMemLdStSupportQueryPlan{*query, std::nullopt};
  return std::nullopt;
}

static std::optional<TMemLdStSupportQueryPlan>
getOuterIndexTMemLdStSupportQueryPlan(Value memDesc, std::string *error) {
  auto queryTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!queryTy || !isPureOuterTMemIndexView(memDesc))
    return std::nullopt;

  auto query = inferStandaloneTMemLdStQueryLayoutImpl(
      memDesc, /*preserveNonCanonicalView=*/true, error);
  if (failed(query))
    return std::nullopt;

  auto rowPlan = getTMemLdStRowPlanForQuery(memDesc, queryTy);
  if (!rowPlan)
    rowPlan = getTMemLdStRowPlan(query->layout);
  if (!rowPlan)
    return std::nullopt;
  return TMemLdStSupportQueryPlan{*query, rowPlan};
}

static std::optional<TMemLdStSupportQueryPlan>
getHalfRowsTMemLdStSupportQueryPlan(Value memDesc, std::string *error) {
  auto queryTy = dyn_cast<MemDescType>(memDesc.getType());
  auto queryPlan = queryTy ? getTMemLdStRowPlanForType(queryTy)
                           : std::optional<TMemLdStRowPlan>{};
  auto backingPlan = getBackingTMemLdStRowPlan(memDesc);
  if (queryTy && shouldPreferDirectHalfRowsSubviewRowPlan(
                     memDesc, queryTy, queryPlan, backingPlan)) {
    return std::nullopt;
  }
  auto query = getHalfRowsTMemLdStSupportQueryLayout(memDesc, error);
  if (!query)
    return std::nullopt;
  auto rowPlan = queryTy ? getTMemLdStRowPlanForType(queryTy)
                         : std::optional<TMemLdStRowPlan>{};
  if (!rowPlan)
    rowPlan = getTMemLdStRowPlan(query->layout);
  if (!rowPlan)
    return std::nullopt;
  return TMemLdStSupportQueryPlan{*query, rowPlan};
}

static std::optional<TMemLdStSupportQueryPlan>
getDirectHalfRowsTMemLdStSupportQueryPlan(Value memDesc, std::string *error) {
  auto queryTy = dyn_cast<MemDescType>(memDesc.getType());
  auto queryPlan = queryTy ? getTMemLdStRowPlanForType(queryTy)
                           : std::optional<TMemLdStRowPlan>{};
  auto backingPlan = getBackingTMemLdStRowPlan(memDesc);
  if (queryTy && shouldPreferDirectHalfRowsSubviewRowPlan(
                     memDesc, queryTy, queryPlan, backingPlan)) {
    return std::nullopt;
  }
  auto query = getDirectHalfRowsTMemLdStSupportQueryLayout(memDesc, error);
  if (!query)
    return std::nullopt;
  auto rowPlan = queryTy ? getTMemLdStRowPlanForType(queryTy)
                         : std::optional<TMemLdStRowPlan>{};
  if (!rowPlan)
    rowPlan = getTMemLdStRowPlan(query->layout);
  if (!rowPlan)
    return std::nullopt;
  return TMemLdStSupportQueryPlan{*query, rowPlan};
}

static std::optional<TMemLdStSupportQueryPlan>
getColumnSubviewTMemLdStSupportQueryPlan(Value memDesc, std::string *error) {
  auto queryTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!queryTy || queryTy.getRank() != 2 ||
      !isTensorMemoryEncoding(queryTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(queryTy.getEncoding())) {
    return std::nullopt;
  }

  auto refreshSupportRowPlan = [&](TMemLdStSupportQueryPlan &support) {
    if (auto rowPlan = getTMemLdStRowPlan(support.query.layout))
      support.rowPlan = rowPlan;
  };
  auto isPhysicalBitcast = [](Value value) {
    auto reinterpret = value.getDefiningOp<gpu::MemDescReinterpretOp>();
    return reinterpret && reinterpret->hasAttr("tmem_physical_bitcast");
  };
  auto getStandalonePhysicalBitcastSupport =
      [&](Value value) -> std::optional<TMemLdStSupportQueryPlan> {
    auto query = inferStandaloneTMemLdStQueryLayoutImpl(
        value, /*preserveNonCanonicalView=*/true, error);
    if (failed(query))
      return std::nullopt;
    auto valueTy = dyn_cast<MemDescType>(value.getType());
    auto rowPlan =
        valueTy ? getTMemLdStRowPlanForQueryLayout(value, valueTy, *query)
                : std::optional<TMemLdStRowPlan>{};
    if (!rowPlan && valueTy)
      rowPlan = getTMemLdStRowPlanForQuery(value, valueTy);
    if (!rowPlan)
      rowPlan = getBackingTMemLdStRowPlan(value);
    if (!rowPlan)
      rowPlan = getTMemLdStRowPlan(query->layout);
    return TMemLdStSupportQueryPlan{*query, rowPlan};
  };
  auto queryCanMaterializeRowPlan = [](const TMemLdStQueryLayout &query,
                                       const TMemLdStRowPlan &rowPlan) {
    return getLogicalRowAnchorBasis(query.layout, rowPlan.warpRow0) &&
           getLogicalRowAnchorBasis(query.layout, rowPlan.warpRow1);
  };

  auto getSourceSupport = [&](Value src)
      -> std::optional<TMemLdStSupportQueryPlan> {
    std::string srcError;
    auto srcTy = dyn_cast<MemDescType>(src.getType());
    if (!srcTy)
      return std::nullopt;
    auto preferredSourceRowPlan = [&](const TMemLdStQueryLayout &query,
                                      std::optional<TMemLdStRowPlan> rowPlan) {
      if (auto backingPlan = getBackingTMemLdStRowPlan(src);
          backingPlan && (!rowPlan || backingPlan->rowSpan > rowPlan->rowSpan) &&
          queryCanMaterializeRowPlan(query, *backingPlan)) {
        return backingPlan;
      }
      return rowPlan;
    };
    auto maybeSrcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
        src, /*preserveNonCanonicalView=*/true, &srcError);
    std::optional<TMemLdStSupportQueryPlan> maybeSrcSupport;
    FailureOr<TMemLdStQueryLayout> srcQuery = maybeSrcQuery;
    if (failed(srcQuery)) {
      maybeSrcSupport = getTMemLdStSupportQueryPlan(src, &srcError);
      if (!maybeSrcSupport)
        return std::nullopt;
      srcQuery = maybeSrcSupport->query;
    }
    auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding());
    auto support = TMemLdStSupportQueryPlan{
        TMemLdStQueryLayout{srcQuery->layout,
                            maybeTwoCTAs.value_or(srcQuery->twoCTAs),
                            srcQuery->origin},
        std::nullopt};
    support.rowPlan =
        preferredSourceRowPlan(support.query,
                               maybeSrcSupport ? maybeSrcSupport->rowPlan
                                               : getTMemLdStRowPlanForQuery(
                                                     src, srcTy));
    if (!support.rowPlan)
      refreshSupportRowPlan(support);
    return support;
  };

  if (auto reinterpret = memDesc.getDefiningOp<gpu::MemDescReinterpretOp>()) {
    if (isPhysicalBitcast(memDesc) &&
        (!queryTy || queryTy.getRank() != 2 || queryTy.getShape()[0] != 64))
      return getStandalonePhysicalBitcastSupport(memDesc);

    auto srcTy = dyn_cast<MemDescType>(reinterpret.getSrc().getType());
    if (!srcTy || srcTy.getRank() != 2)
      return std::nullopt;
    int64_t srcBits =
        product<int64_t>(srcTy.getShape()) * srcTy.getElementTypeBitWidth();
    int64_t dstBits =
        product<int64_t>(queryTy.getShape()) * queryTy.getElementTypeBitWidth();
    if (srcBits != dstBits)
      return std::nullopt;

    if (!isPhysicalBitcast(memDesc) &&
        srcTy.getElementTypeBitWidth() == queryTy.getElementTypeBitWidth()) {
      if (auto support = getStandalonePhysicalBitcastSupport(memDesc))
        return support;
    }

    auto reinterpretSourceSupport = [&]()
        -> std::optional<TMemLdStSupportQueryPlan> {
      auto support = getSourceSupport(reinterpret.getSrc());
      if (!support)
        return std::nullopt;

      auto reinterpretedSupport = reinterpretColumnSubviewSupportQueryLayout(
          srcTy.getShape(), srcTy.getElementTypeBitWidth(), support->query,
          queryTy.getShape(), queryTy.getElementTypeBitWidth(), error);
      if (!reinterpretedSupport)
        return std::nullopt;
      if (auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding()))
        reinterpretedSupport->twoCTAs = *maybeTwoCTAs;
      auto result =
          TMemLdStSupportQueryPlan{*reinterpretedSupport, support->rowPlan};
      if (auto layoutRowPlan = getTMemLdStRowPlan(result.query.layout)) {
        if (isPhysicalBitcast(memDesc) || !result.rowPlan ||
            layoutRowPlan->rowSpan > result.rowPlan->rowSpan)
          result.rowPlan = layoutRowPlan;
      }
      return result;
    };

    if (auto support = reinterpretSourceSupport())
      return support;
    if (isPhysicalBitcast(memDesc))
      return getStandalonePhysicalBitcastSupport(memDesc);
    return std::nullopt;
  }

  if (auto index = memDesc.getDefiningOp<gpu::MemDescIndexOp>()) {
    APInt indexValue;
    if (matchPattern(index.getIndex(), m_ConstantInt(&indexValue)) &&
        indexValue.getSExtValue() == 0) {
      if (auto subslice = index.getSrc().getDefiningOp<gpu::MemDescSubsliceOp>();
          subslice && isTensorMemoryColumnHalfDim0Slice(subslice)) {
        if (auto reshape =
                subslice.getSrc().getDefiningOp<gpu::MemDescReshapeOp>()) {
          auto srcTy = dyn_cast<MemDescType>(reshape.getSrc().getType());
          if (!srcTy || srcTy.getRank() != 2)
            return std::nullopt;
          std::string srcError;
          auto maybeSrcLayout = getTMemViewAnalysisLinearLayout(
              srcTy.getShape(), srcTy.getEncoding(), &srcError);
          if (!maybeSrcLayout) {
            if (auto maybeCanonical =
                    getCanonicalTMemLinearEncoding(srcTy, &srcError)) {
              maybeSrcLayout = normalizeTensorMemoryLinearLayoutForAnalysis(
                  maybeCanonical->getLinearLayout());
            }
          }
          if (!maybeSrcLayout)
            return std::nullopt;
          auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding());
          auto rowPlan = getTMemLdStRowPlanForQuery(reshape.getSrc(), srcTy);
          if (!rowPlan)
            rowPlan = getBackingTMemLdStRowPlan(reshape.getSrc());
          return TMemLdStSupportQueryPlan{
              TMemLdStQueryLayout{*maybeSrcLayout,
                                  maybeTwoCTAs.value_or(false),
                                  SmallVector<int32_t>(
                                      maybeSrcLayout->getNumInDims(), 0)},
              rowPlan};
        }
      }
    }
  }

  auto subslice = memDesc.getDefiningOp<gpu::MemDescSubsliceOp>();
  if (!subslice)
    return std::nullopt;
  if (isPhysicalBitcast(subslice.getSrc()))
    return getStandalonePhysicalBitcastSupport(memDesc);
  auto srcTy = dyn_cast<MemDescType>(subslice.getSrc().getType());
  if (!srcTy || srcTy.getRank() != 2 || subslice.getOffsets().size() != 2 ||
      subslice.getOffsets()[0] != 0 ||
      srcTy.getShape()[0] != queryTy.getShape()[0] ||
      srcTy.getShape()[1] <= queryTy.getShape()[1] ||
      !isTensorMemoryEncoding(srcTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
    return std::nullopt;
  }

  auto maybeSrcSupport = getSourceSupport(subslice.getSrc());
  if (!maybeSrcSupport)
    return std::nullopt;
  auto support = *maybeSrcSupport;

  auto *ctx = memDesc.getContext();
  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding());
  auto kCol = StringAttr::get(ctx, "col");
  support.query = TMemLdStQueryLayout{
      support.query.layout, maybeTwoCTAs.value_or(support.query.twoCTAs),
      remapTMemLdStQueryOrigin(
          support.query, support.query.layout,
          {{kCol, static_cast<int32_t>(subslice.getOffsets()[1])}})};
  // Column subviews borrow the source support image. If the borrowed image
  // exposes a wider row plan than the projected slice, keep it only when the
  // support layout itself can materialize both row anchors.
  if (auto layoutRowPlan = getTMemLdStRowPlan(support.query.layout)) {
    bool keepSourcePlan =
        support.rowPlan && support.rowPlan->rowSpan > layoutRowPlan->rowSpan &&
        queryCanMaterializeRowPlan(support.query, *support.rowPlan);
    if (!keepSourcePlan) {
      support.rowPlan = layoutRowPlan;
    }
  }
  // If neither the source nor borrowed support image can provide a row plan,
  // fall back to the sliced standalone query.
  if (!support.rowPlan)
    if (auto sliceQuery = inferStandaloneTMemLdStQueryLayoutImpl(
            memDesc, /*preserveNonCanonicalView=*/true,
            /*error=*/nullptr);
        succeeded(sliceQuery)) {
      support.rowPlan =
          getTMemLdStRowPlanForQueryLayout(memDesc, queryTy, *sliceQuery);
    }
  if (!support.rowPlan)
    support.rowPlan = getTMemLdStRowPlanForQuery(memDesc, queryTy);
  if (!support.rowPlan)
    refreshSupportRowPlan(support);
  return support;
}

std::optional<TMemLdStSupportQueryPlan>
getTMemLdStSupportQueryPlan(Value memDesc, std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  auto queryTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!queryTy)
    return std::nullopt;
  if (auto support = getDirectHalfRowsTMemLdStSupportQueryPlan(memDesc, error)) {
    if (debug)
      llvm::errs() << "[tmem-ldst-support] direct half-rows support layout:\n"
                   << support->query.layout.toString() << "\n";
    return support;
  }
  if (auto support = getHalfRowsTMemLdStSupportQueryPlan(memDesc, error)) {
    if (debug)
      llvm::errs()
          << "[tmem-ldst-support] higher-rank half-rows support layout:\n"
          << support->query.layout.toString() << "\n";
    return support;
  }
  if (auto support = getOuterIndexTMemLdStSupportQueryPlan(memDesc, error)) {
    if (debug)
      llvm::errs() << "[tmem-ldst-support] outer-index support layout:\n"
                   << support->query.layout.toString() << "\n";
    return support;
  }
  if (auto support = getColumnSubviewTMemLdStSupportQueryPlan(memDesc, error)) {
    if (debug)
      llvm::errs() << "[tmem-ldst-support] column-slice support layout:\n"
                   << support->query.layout.toString() << "\n";
    return support;
  }
  if (auto support = getGenericTMemLdStReshapedSupportQueryPlan(memDesc, error)) {
    if (debug)
      llvm::errs() << "[tmem-ldst-support] generic support layout:\n"
                   << support->query.layout.toString() << "\n";
    return support;
  }
  return std::nullopt;
}

std::optional<TMemLdStQueryLayout>
getTMemLdStSupportQueryLayout(Value memDesc, std::string *error) {
  if (auto support = getTMemLdStSupportQueryPlan(memDesc, error))
    return support->query;
  return std::nullopt;
}

FailureOr<MemDescType> inferStandaloneTMemViewType(Value memDesc,
                                                   std::string *error) {
  return inferStandaloneTMemViewTypeImpl(
      memDesc, /*preserveNonCanonicalView=*/false, error);
}

FailureOr<TMemPhysicalQuery>
inferStandaloneTMemPhysicalQuery(Value memDesc, bool preserveNonCanonicalView,
                                 std::string *error) {
  auto maybeTy = inferStandaloneTMemViewTypeImpl(
      memDesc, preserveNonCanonicalView, error);
  if (failed(maybeTy))
    return failure();
  auto encoding = maybeTy->getEncoding();
  if (!isTensorMemoryEncoding(encoding)) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }

  bool isScales = isa<TensorMemoryScalesEncodingAttr>(encoding);
  bool twoCTAs = false;
  if (auto scales = dyn_cast<TensorMemoryScalesEncodingAttr>(encoding)) {
    twoCTAs = product<unsigned>(scales.getCGALayout().getCTAsPerCGA()) > 1;
  } else {
    auto maybeTwoCTAs = getTensorMemoryTwoCTAs(encoding);
    if (!maybeTwoCTAs) {
      if (error)
        *error = "expected tensor memory layout encoding";
      return failure();
    }
    twoCTAs = *maybeTwoCTAs;
  }

  auto layout = toLinearLayout(*maybeTy);
  return TMemPhysicalQuery{
      *maybeTy,
      llvm::to_vector(maybeTy->getShape()),
      llvm::to_vector(maybeTy->getAllocShape()),
      static_cast<unsigned>(maybeTy->getElementTypeBitWidth()),
      layout,
      twoCTAs,
      SmallVector<int32_t>(layout.getNumInDims(), 0), isScales};
}

FailureOr<TMemPhysicalQuery>
inferStandaloneTMemPhysicalQuery(Value memDesc, std::string *error) {
  return inferStandaloneTMemPhysicalQuery(
      memDesc, /*preserveNonCanonicalView=*/false, error);
}

FailureOr<TMemPhysicalQuery>
inferExactTMemPhysicalQuery(Value memDesc, bool preserveNonCanonicalView,
                            std::string *error) {
  auto memDescTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!memDescTy || memDescTy.getMemorySpace() !=
                        TensorMemorySpaceAttr::get(memDesc.getContext())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  auto encoding = memDescTy.getEncoding();
  if (!isTensorMemoryEncoding(encoding)) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }

  auto maybeQuery = inferStandaloneTMemLdStQueryLayout(
      memDesc, preserveNonCanonicalView, error);
  if (failed(maybeQuery))
    return failure();
  return TMemPhysicalQuery{
      memDescTy,
      llvm::to_vector(memDescTy.getShape()),
      llvm::to_vector(memDescTy.getAllocShape()),
      static_cast<unsigned>(memDescTy.getElementTypeBitWidth()),
      maybeQuery->layout,
      maybeQuery->twoCTAs,
      maybeQuery->origin,
      isa<TensorMemoryScalesEncodingAttr>(encoding)};
}

FailureOr<TMemPhysicalQuery>
inferExactTMemPhysicalQuery(Value memDesc, std::string *error) {
  return inferExactTMemPhysicalQuery(
      memDesc, /*preserveNonCanonicalView=*/true, error);
}

std::optional<TMemPhysicalQueryDifference>
getFirstTMemPhysicalQueryDifference(const TMemPhysicalQuery &lhs,
                                    const TMemPhysicalQuery &rhs) {
  if (lhs.shape != rhs.shape)
    return TMemPhysicalQueryDifference::Shape;
  if (lhs.allocShape != rhs.allocShape)
    return TMemPhysicalQueryDifference::AllocShape;
  if (lhs.elementBitWidth != rhs.elementBitWidth)
    return TMemPhysicalQueryDifference::ElementBitWidth;
  if (lhs.layout != rhs.layout)
    return TMemPhysicalQueryDifference::Layout;
  if (lhs.twoCTAs != rhs.twoCTAs)
    return TMemPhysicalQueryDifference::TwoCTAs;
  if (lhs.origin != rhs.origin)
    return TMemPhysicalQueryDifference::Origin;
  if (lhs.isScales != rhs.isScales)
    return TMemPhysicalQueryDifference::Scales;
  return std::nullopt;
}

bool haveSameTMemPhysicalQueryProjection(const TMemPhysicalQuery &lhs,
                                         const TMemPhysicalQuery &rhs) {
  return !getFirstTMemPhysicalQueryDifference(lhs, rhs).has_value();
}

bool haveSameTMemCopyPhysicalProjection(const TMemPhysicalQuery &lhs,
                                        const TMemPhysicalQuery &rhs) {
  return lhs.shape == rhs.shape && lhs.elementBitWidth == rhs.elementBitWidth &&
         lhs.layout == rhs.layout && lhs.twoCTAs == rhs.twoCTAs &&
         lhs.isScales == rhs.isScales;
}

StringRef stringifyTMemPhysicalQueryDifference(
    TMemPhysicalQueryDifference difference) {
  switch (difference) {
  case TMemPhysicalQueryDifference::Shape:
    return "active shape";
  case TMemPhysicalQueryDifference::AllocShape:
    return "allocation shape";
  case TMemPhysicalQueryDifference::ElementBitWidth:
    return "element bitwidth";
  case TMemPhysicalQueryDifference::Layout:
    return "physical layout";
  case TMemPhysicalQueryDifference::TwoCTAs:
    return "CTA ownership";
  case TMemPhysicalQueryDifference::Origin:
    return "physical origin";
  case TMemPhysicalQueryDifference::Scales:
    return "scales classification";
  }
  llvm_unreachable("unknown TMEM physical query difference");
}

FailureOr<MemDescType> inferTMemBitcastType(Value memDesc,
                                            ArrayRef<int64_t> dstShape,
                                            Type dstElementType,
                                            std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  auto srcTy = dyn_cast<MemDescType>(memDesc.getType());
  if (!srcTy ||
      srcTy.getMemorySpace() != TensorMemorySpaceAttr::get(memDesc.getContext())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  if (!isTensorMemoryEncoding(srcTy.getEncoding())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  if (isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
    if (error)
      *error = "tensor memory scales descriptors do not support bitcast";
    return failure();
  }

  auto srcQuery = inferStandaloneTMemLdStQueryLayoutImpl(
      memDesc, /*preserveNonCanonicalView=*/true, error);
  if (failed(srcQuery))
    return failure();
  if (debug) {
    llvm::errs() << "[tmem-bitcast] src type=" << srcTy << "\n"
                 << "[tmem-bitcast] src query:\n"
                 << srcQuery->layout.toString() << "\n";
  }
  if (error)
    error->clear();

  int64_t dstBitwidth =
      getElementTypeOrSelf(dstElementType).getIntOrFloatBitWidth();
  auto dstQuery = inferTMemReinterpretQueryLayout(
      srcTy.getShape(), srcTy.getElementTypeBitWidth(), *srcQuery, dstShape,
      dstBitwidth, memDesc.getContext(), error);
  if (failed(dstQuery)) {
    if (debug && error)
      llvm::errs() << "[tmem-bitcast] reinterpret query failed: " << *error
                   << "\n";
    return failure();
  }
  if (debug) {
    llvm::errs() << "[tmem-bitcast] dst query:\n"
                 << dstQuery->layout.toString() << "\n";
  }
  if (error)
    error->clear();

  auto dstEncoding = tryMakeTMemViewEncoding(
      memDesc.getContext(), dstQuery->layout, dstQuery->twoCTAs, error);
  if (!dstEncoding)
    return failure();

  auto resultTy = tryCreateMemDescType(
      memDesc.getContext(), dstShape, dstElementType, *dstEncoding,
      srcTy.getMemorySpace(), srcTy.getMutableMemory(), dstShape, error);
  if (!resultTy)
    return failure();
  return *resultTy;
}

std::optional<LinearLayout>
getTMemViewAnalysisLinearLayout(ArrayRef<int64_t> shape, Attribute encoding,
                                std::string *error) {
  auto maybe = getTMemViewAnalysisLayout(shape, encoding, error);
  if (!maybe)
    return std::nullopt;
  return maybe->layout;
}

std::optional<TensorMemoryLinearEncodingAttr>
tryMakeTMemViewEncoding(MLIRContext *ctx, LinearLayout ll, bool twoCTAs,
                        std::string *error) {
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  bool hadBlock = ll.hasInDim(kBlock);
  decltype(ll.getBases().lookup(kBlock)) originalBlockBases;
  if (hadBlock)
    originalBlockBases = ll.getBases().lookup(kBlock);
  auto trimTrailingZeroBases = [&](StringAttr dim) {
    auto bases = ll.getBases();
    auto it = bases.find(dim);
    if (it == bases.end())
      return;
    auto &dimBases = it->second;
    while (!dimBases.empty() &&
           llvm::all_of(dimBases.back(), [](int32_t value) {
             return value == 0;
           })) {
      dimBases.pop_back();
    }
    ll = LinearLayout(std::move(bases), ll.getOutDims(), ll.isSurjective());
  };
  // Row and column zero bases are semantic: they describe physical TMEM
  // broadcast/repetition axes used by families such as tcgen05.copy.warpx2.
  // Only trim inactive block bases when retrying non-two-CTA encodings.
  trimTrailingZeroBases(kBlock);
  SmallVector<std::pair<StringAttr, int32_t>> canonicalOutDims;
  canonicalOutDims.reserve(ll.getNumOutDims());
  for (auto [idx, dim] : llvm::enumerate(ll.getOutDimNames())) {
    canonicalOutDims.push_back(
        {StringAttr::get(ctx, "dim" + llvm::Twine(idx)),
         ll.getOutDimSize(dim)});
  }
  ll = LinearLayout(ll.getBases(), canonicalOutDims, ll.isSurjective());
  if (auto enc = tryMakeTensorMemoryLinearEncoding(ctx, ll, twoCTAs, error))
    return enc;
  if (!hadBlock)
    return std::nullopt;

  SmallVector<StringAttr> noBlockInDims;
  noBlockInDims.reserve(ll.getNumInDims());
  for (auto dim : ll.getInDimNames()) {
    if (dim != kBlock)
      noBlockInDims.push_back(dim);
  }
  auto noBlockOutDims = llvm::to_vector(ll.getOutDimNames());
  auto collapsedBlock = ll.sublayout(noBlockInDims, noBlockOutDims);
  if (auto enc = tryMakeTensorMemoryLinearEncoding(ctx, collapsedBlock,
                                                   /*twoCTAs=*/false, error))
    return enc;

  bool blockInactive = !originalBlockBases.empty() &&
                       llvm::all_of(originalBlockBases,
                                    [](ArrayRef<int32_t> basis) {
                         return llvm::all_of(
                             basis, [](int32_t value) { return value == 0; });
                       });
  if (!blockInactive)
    return std::nullopt;

  if (ll.hasInDim(kBlock))
    ll = ll.removeZeroBasesAlongDim(kBlock);
  if (ll.hasInDim(kBlock) && ll.getInDimSize(kBlock) == 1)
    ll = ll.squeezeIns(kBlock);
  return tryMakeTensorMemoryLinearEncoding(ctx, ll, /*twoCTAs=*/false, error);
}

std::optional<SmallVector<int64_t>>
getTMemAllocShapeForEncoding(ArrayRef<int64_t> shape, Attribute encoding,
                             std::string *error) {
  SmallVector<int64_t> allocShape(shape.begin(), shape.end());
  if (!isTensorMemoryEncoding(encoding) ||
      isa<TensorMemoryScalesEncodingAttr>(encoding)) {
    return allocShape;
  }

  auto layoutTrait = dyn_cast<LayoutEncodingTrait>(encoding);
  if (!layoutTrait) {
    if (error)
      *error = "expected tensor memory layout encoding";
    return std::nullopt;
  }
  auto layoutRank = static_cast<size_t>(layoutTrait.getRank());
  if (shape.size() < layoutRank) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return std::nullopt;
  }

  auto maybeLayout = tryGetCanonicalTensorMemoryLinearLayout(
      shape.take_back(layoutRank), encoding, error);
  if (!maybeLayout)
    return std::nullopt;

  auto layoutShape = maybeLayout->getOutDimSizes();
  for (auto [idx, dimSize] : llvm::enumerate(layoutShape)) {
    auto allocIdx = allocShape.size() - layoutRank + idx;
    allocShape[allocIdx] = std::max<int64_t>(allocShape[allocIdx], dimSize);
  }
  return allocShape;
}

LogicalResult inferTMemReshapeOpEncoding(ArrayRef<int64_t> srcShape,
                                         Attribute srcEncoding,
                                         ArrayRef<int64_t> dstShape,
                                         Attribute &dstEncoding,
                                         std::optional<Location> loc) {
  if (product(srcShape) != product(dstShape)) {
    return emitOptionalError(loc, "numel of dst shape does not match numel of "
                                  "src shape");
  }

  auto *ctx = srcEncoding.getContext();
  auto elemTy = IntegerType::get(ctx, 8);

  std::string error;
  auto srcAllocShape = getTMemAllocShapeForEncoding(srcShape, srcEncoding,
                                                    &error);
  auto memTy = srcAllocShape
                   ? tryCreateMemDescType(ctx, srcShape, elemTy, srcEncoding,
                                          TensorMemorySpaceAttr::get(ctx),
                                          /*mutableMemory=*/false,
                                          *srcAllocShape, &error)
                   : std::nullopt;
  FailureOr<gpu::MemDescType> resultTy = failure();
  if (memTy)
    resultTy = inferTMemReshapeOpType(*memTy, dstShape, &error);
  if (succeeded(resultTy)) {
    dstEncoding = resultTy->getEncoding();
  } else {
    auto layoutRank = cast<LayoutEncodingTrait>(srcEncoding).getRank();
    if (srcShape.size() == static_cast<size_t>(layoutRank)) {
      std::string preservedError;
      auto preservedTy = tryCreateMemDescType(
          ctx, dstShape, elemTy, srcEncoding, TensorMemorySpaceAttr::get(ctx),
          /*mutableMemory=*/false, dstShape, &preservedError);
      if (!preservedTy)
        return emitOptionalError(loc, error.empty() ? preservedError : error);
    }

    // Some TMEM descriptor views are valid pointer transformations but are not
    // representable as standalone TMEM-linear layouts. Preserve the source
    // encoding sugar and let later TMEM consumers decide whether direct
    // codegen is possible.
    dstEncoding = srcEncoding;
  }
  return success();
}

LogicalResult inferTMemIndexOpEncoding(ArrayRef<int64_t> srcShape,
                                       ArrayRef<int64_t> dstShape,
                                       ArrayRef<int64_t> dstAllocShape,
                                       Attribute srcEncoding,
                                       Attribute &dstEncoding,
                                       std::optional<Location> loc) {
  auto *ctx = srcEncoding.getContext();
  std::string error;
  if (auto preserved = tryPreserveOuterIndexedTMemEncoding(
          ctx, srcShape, dstShape, dstAllocShape, srcEncoding, &error)) {
    dstEncoding = *preserved;
    return success();
  }
  auto result =
      inferTMemIndexEncoding(srcShape, dstShape, dstAllocShape, srcEncoding,
                             &error);
  if (succeeded(result) &&
      canMaterializeTMemViewEncoding(ctx, dstShape, dstAllocShape, *result)) {
    dstEncoding = *result;
    return success();
  }
  return preserveTMemViewEncodingIfValid(ctx, srcShape, dstShape, dstAllocShape,
                                         srcEncoding, dstEncoding, loc,
                                         &error);
}

LogicalResult inferTMemSubsliceOpEncoding(ArrayRef<int64_t> srcShape,
                                          ArrayRef<int64_t> srcAllocShape,
                                          Attribute srcEncoding,
                                          ArrayRef<int64_t> dstShape,
                                          ArrayRef<int32_t> offsets,
                                          Attribute &dstEncoding,
                                          std::optional<Location> loc) {
  auto *ctx = srcEncoding.getContext();
  if (isTensorMemoryEncoding(srcEncoding) &&
      !isa<TensorMemoryScalesEncodingAttr>(srcEncoding)) {
    std::string leadingUnitError;
    if (auto maybeSrcLayout =
            getTMemViewAnalysisLayout(srcShape, srcEncoding, &leadingUnitError)) {
      auto layoutRank = maybeSrcLayout->layout.getNumOutDims();
      auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
      if (extraRank >= 0) {
        if (auto leadingUnit = tryMakeLeadingUnitSubviewLayout(
                maybeSrcLayout->layout, srcShape.drop_front(extraRank),
                dstShape.drop_front(extraRank), offsets.drop_front(extraRank),
                /*srcOrigin=*/{}, &leadingUnitError)) {
          if (auto result = tryMakeTMemViewEncoding(
                  ctx, leadingUnit->layout, maybeSrcLayout->twoCTAs,
                  &leadingUnitError)) {
            if (canMaterializeTMemViewEncoding(ctx, dstShape, srcAllocShape,
                                               *result)) {
              dstEncoding = *result;
              return success();
            }
          }
        }
      }
    }
  }
  if (auto preserved = tryPreserveExactTMemViewEncoding(
          ctx, dstShape, srcAllocShape, srcEncoding, /*error=*/nullptr)) {
    dstEncoding = *preserved;
    return success();
  }
  if (isTensorMemoryEncoding(srcEncoding) &&
      !isa<TensorMemoryScalesEncodingAttr>(srcEncoding)) {
    auto layoutRank = cast<LayoutEncodingTrait>(srcEncoding).getRank();
    if (srcShape.take_back(layoutRank) != dstShape.take_back(layoutRank)) {
      std::string preservedError;
      if (succeeded(preserveTMemViewEncodingIfValid(
              ctx, srcShape, dstShape, srcAllocShape, srcEncoding, dstEncoding,
              loc,
              &preservedError))) {
        return success();
      }
    }
  }
  std::string error;
  auto result =
      inferTMemSubsliceEncoding(srcShape, srcEncoding, dstShape, offsets,
                                &error);
  if (succeeded(result) &&
      canMaterializeTMemViewEncoding(ctx, dstShape, srcAllocShape, *result)) {
    dstEncoding = *result;
    return success();
  }
  return preserveTMemViewEncodingIfValid(ctx, srcShape, dstShape, srcAllocShape,
                                         srcEncoding, dstEncoding, loc,
                                         &error);
}

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemSubsliceEncoding(gpu::MemDescType srcTy, gpu::MemDescType dstTy,
                          ArrayRef<int64_t> offsets) {
  SmallVector<int64_t> srcShape(srcTy.getShape().begin(), srcTy.getShape().end());
  SmallVector<int64_t> dstShape(dstTy.getShape().begin(), dstTy.getShape().end());
  SmallVector<int32_t> offsets32(offsets.begin(), offsets.end());
  return inferTMemSubsliceEncoding(srcShape, srcTy.getEncoding(), dstShape,
                                   offsets32);
}

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemSubsliceEncoding(ArrayRef<int64_t> srcShape, Attribute srcEncoding,
                          ArrayRef<int64_t> dstShape,
                          ArrayRef<int32_t> offsets, std::string *error) {
  auto maybeSrcLayout = getTMemViewAnalysisLayout(srcShape, srcEncoding, error);
  if (!maybeSrcLayout) {
    if (error && error->empty())
      *error = "expected canonical tensor memory linear encoding";
    return failure();
  }

  auto ll = maybeSrcLayout->layout;
  auto layoutRank = ll.getNumOutDims();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank < 0 || dstShape.size() != srcShape.size() ||
      offsets.size() != srcShape.size()) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return failure();
  }

  auto *ctx = srcEncoding.getContext();
  if (srcShape.drop_front(extraRank) == dstShape.drop_front(extraRank) &&
      llvm::all_of(offsets.drop_front(extraRank),
                   [](int32_t offset) { return offset == 0; })) {
    auto result =
        tryMakeTMemViewEncoding(ctx, ll, maybeSrcLayout->twoCTAs, error);
    if (!result)
      return failure();
    return *result;
  }

  if (auto leadingUnit = tryMakeLeadingUnitSubviewLayout(
          ll, srcShape.drop_front(extraRank), dstShape.drop_front(extraRank),
          offsets.drop_front(extraRank), /*srcOrigin=*/{}, error)) {
    auto result = tryMakeTMemViewEncoding(ctx, leadingUnit->layout,
                                          maybeSrcLayout->twoCTAs, error);
    if (!result)
      return failure();
    return *result;
  }

  // Pure 2D column subviews preserve the same row mapping and only narrow the
  // materialized logical column span. This also keeps non-injective row support
  // bases, such as the zero-row basis used by the public warpx2 copy layouts,
  // instead of sending them through the generic inverse path.
  if (extraRank == 0 && layoutRank == 2 && dstShape[0] == srcShape[0] &&
      offsets[0] == 0 && dstShape[1] <= srcShape[1] && offsets[1] >= 0 &&
      offsets[1] + dstShape[1] <= srcShape[1]) {
    auto kCol = StringAttr::get(ctx, "col");
    auto logicalDims = llvm::to_vector(ll.getOutDimNames());
    if (logicalDims.size() != 2 || !ll.hasInDim(kCol) ||
        ll.getInDimSize(kCol) < dstShape[1] ||
        ll.getOutDimSize(logicalDims[1]) < dstShape[1]) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
    auto narrowedLayout = ll;
    if (narrowedLayout.getOutDimSize(logicalDims[1]) != dstShape[1])
      narrowedLayout =
          narrowedLayout.resizeOutDim(logicalDims[1], dstShape[1]);
    if (narrowedLayout.getInDimSize(kCol) != dstShape[1])
      narrowedLayout = narrowedLayout.resizeInDim(kCol, dstShape[1]);
    auto result = tryMakeTMemViewEncoding(ctx, narrowedLayout,
                                          maybeSrcLayout->twoCTAs, error);
    if (!result)
      return failure();
    return *result;
  }

  auto logicalDims = llvm::to_vector(ll.getOutDimNames());
  SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
  encodedOffsets.reserve(layoutRank);
  for (auto [dim, offset] : llvm::enumerate(offsets.drop_front(extraRank))) {
    if (offset < 0) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
    encodedOffsets.push_back(
        {logicalDims[dim], static_cast<int32_t>(offset)});
  }

  auto llInv = computeLeftInverseLayout(ll, error);
  if (failed(llInv)) {
    auto normalized = normalizeTensorMemoryLinearLayoutForAnalysis(ll);
    std::string normalizedError;
    auto normalizedInv = computeLeftInverseLayout(normalized, &normalizedError);
    if (succeeded(normalizedInv)) {
      ll = normalized;
      logicalDims = llvm::to_vector(ll.getOutDimNames());
      encodedOffsets.clear();
      encodedOffsets.reserve(layoutRank);
      for (auto [dim, offset] : llvm::enumerate(offsets.drop_front(extraRank))) {
        encodedOffsets.push_back(
            {logicalDims[dim], static_cast<int32_t>(offset)});
      }
      llInv = std::move(normalizedInv);
    } else {
      if (error && error->empty())
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }
  }
  auto baseCoords =
      llInv->apply(makeFullLinearLayoutCoords(logicalDims, encodedOffsets));
  auto physOutDimNames = llvm::to_vector(llInv->getOutDimNames());
  SmallVector<uint32_t> activePhysMasks(physOutDimNames.size(), 0);

  LinearLayout::BasesT dstInvBases;
  for (int64_t dim = extraRank; dim < static_cast<int64_t>(srcShape.size());
       ++dim) {
    int64_t dstDimSize = dstShape[dim];
    int64_t srcDimSize = srcShape[dim];
    if (dstDimSize > srcDimSize || offsets[dim] + dstDimSize > srcDimSize) {
      if (error)
        *error = "unsupported tensor memory memdesc_subslice view";
      return failure();
    }

    auto dstDimName =
        StringAttr::get(ctx, "dim" + llvm::Twine(dim - extraRank));
    auto &bases = dstInvBases[dstDimName];
    for (int64_t step = 1; step < dstDimSize; step <<= 1) {
      auto point = encodedOffsets;
      point[dim - extraRank].second += static_cast<int32_t>(step);
      auto pointCoords =
          llInv->apply(makeFullLinearLayoutCoords(logicalDims, point));
      std::vector<int32_t> basis;
      basis.reserve(physOutDimNames.size());
      for (auto [physIdx, physDim] : llvm::enumerate(physOutDimNames)) {
        int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                        lookupLinearLayoutCoord(baseCoords, physDim);
        if (delta < 0) {
          if (error)
            *error = "unsupported tensor memory memdesc_subslice view";
          return failure();
        }
        activePhysMasks[physIdx] |= static_cast<uint32_t>(delta);
        basis.push_back(delta);
      }
      bases.push_back(std::move(basis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> activePhysOutDims;
  activePhysOutDims.reserve(physOutDimNames.size());
  for (auto [physIdx, physDim] : llvm::enumerate(physOutDimNames)) {
    int32_t activePhysSize = 1;
    while (activePhysSize <= static_cast<int32_t>(activePhysMasks[physIdx]))
      activePhysSize <<= 1;
    activePhysOutDims.push_back({physDim, activePhysSize});
  }

  auto dstInv = LinearLayout::tryCreate(std::move(dstInvBases), activePhysOutDims,
                                        /*requireSurjective=*/false, error);
  if (!dstInv) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
  }
  auto dstLayout = computeLeftInverseLayout(*dstInv, error);
  if (failed(dstLayout)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
  }
  auto result = tryMakeTMemViewEncoding(ctx, *dstLayout,
                                        maybeSrcLayout->twoCTAs, error);
  if (!result)
    return failure();
  if (failed(verifyTMemSubsliceProjection(
          *llInv, logicalDims, encodedOffsets, baseCoords,
          result->getLinearLayout(), dstShape.drop_front(extraRank), error))) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
  }
  return *result;
}

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemIndexEncoding(gpu::MemDescType srcTy, gpu::MemDescType dstTy) {
  SmallVector<int64_t> srcShape(srcTy.getShape().begin(), srcTy.getShape().end());
  SmallVector<int64_t> dstShape(dstTy.getShape().begin(), dstTy.getShape().end());
  SmallVector<int64_t> dstAllocShape(dstTy.getAllocShape().begin(),
                                     dstTy.getAllocShape().end());
  return inferTMemIndexEncoding(srcShape, dstShape, dstAllocShape,
                                srcTy.getEncoding());
}

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemIndexEncoding(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                       ArrayRef<int64_t> dstAllocShape, Attribute srcEncoding,
                       std::string *error) {
  auto maybeSrcLayout = getTMemViewAnalysisLayout(srcShape, srcEncoding, error);
  if (!maybeSrcLayout) {
    if (error && error->empty())
      *error = "expected canonical tensor memory linear encoding";
    return failure();
  }

  auto ll = maybeSrcLayout->layout;
  while (ll.getNumOutDims() > 0) {
    auto firstDim = *ll.getOutDimNames().begin();
    if (ll.getOutDimSize(firstDim) != 1)
      break;
    ll = ll.squeezeOuts(firstDim);
  }
  auto layoutRank = ll.getNumOutDims();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank < 0) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return failure();
  }

  if (extraRank > 0) {
    auto *ctx = srcEncoding.getContext();
    auto result =
        tryMakeTMemViewEncoding(ctx, ll, maybeSrcLayout->twoCTAs, error);
    if (!result)
      return failure();
    return *result;
  }

  auto *ctx = srcEncoding.getContext();
  if (layoutRank == 0) {
    if (error)
      *error = "tensor memory layout rank must be greater than zero";
    return failure();
  }
  auto maybeSrcInv = computeLeftInverseLayout(ll, error);
  if (failed(maybeSrcInv)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }

  auto srcLogicalDims = llvm::to_vector(ll.getOutDimNames());
  auto baseCoords =
      maybeSrcInv->apply(makeFullLinearLayoutCoords(srcLogicalDims, {}));
  auto physOutDimNames = llvm::to_vector(maybeSrcInv->getOutDimNames());
  SmallVector<uint32_t> activePhysMasks(physOutDimNames.size(), 0);

  LinearLayout::BasesT dstInvBases;
  for (int64_t dim = 1; dim < static_cast<int64_t>(srcShape.size()); ++dim) {
    int64_t dstDimSize = dstShape[dim - 1];
    int64_t srcDimSize = srcShape[dim];
    if (dstDimSize != srcDimSize) {
      if (error)
        *error = "unsupported tensor memory memdesc_index view";
      return failure();
    }

    auto dstDimName = StringAttr::get(ctx, "dim" + llvm::Twine(dim - 1));
    auto &bases = dstInvBases[dstDimName];
    for (int64_t step = 1; step < dstDimSize; step <<= 1) {
      SmallVector<std::pair<StringAttr, int32_t>> srcPoint;
      srcPoint.reserve(srcLogicalDims.size());
      for (auto logicalDim : srcLogicalDims) {
        srcPoint.push_back(
            {logicalDim,
             logicalDim == srcLogicalDims[dim] ? static_cast<int32_t>(step)
                                               : 0});
      }
      auto pointCoords =
          maybeSrcInv->apply(makeFullLinearLayoutCoords(srcLogicalDims, srcPoint));
      std::vector<int32_t> basis;
      basis.reserve(physOutDimNames.size());
      for (auto [physIdx, physDim] : llvm::enumerate(physOutDimNames)) {
        int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                        lookupLinearLayoutCoord(baseCoords, physDim);
        if (delta < 0) {
          if (error)
            *error = "unsupported tensor memory memdesc_index view";
          return failure();
        }
        activePhysMasks[physIdx] |= static_cast<uint32_t>(delta);
        basis.push_back(delta);
      }
      bases.push_back(std::move(basis));
    }
  }

  SmallVector<std::pair<StringAttr, int32_t>> activePhysOutDims;
  activePhysOutDims.reserve(physOutDimNames.size());
  for (auto [physIdx, physDim] : llvm::enumerate(physOutDimNames)) {
    int32_t activePhysSize = 1;
    while (activePhysSize <= static_cast<int32_t>(activePhysMasks[physIdx]))
      activePhysSize <<= 1;
    activePhysOutDims.push_back({physDim, activePhysSize});
  }

  auto dstInv = LinearLayout::tryCreate(std::move(dstInvBases), activePhysOutDims,
                                        /*requireSurjective=*/false, error);
  if (!dstInv) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }
  auto dstLayout = computeLeftInverseLayout(*dstInv, error);
  if (failed(dstLayout)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }

  auto result = tryMakeTMemViewEncoding(ctx, std::move(*dstLayout),
                                        maybeSrcLayout->twoCTAs, error);
  if (!result)
    return failure();
  if (failed(verifyTMemIndexProjection(*maybeSrcInv, srcLogicalDims,
                                       result->getLinearLayout(), dstShape,
                                       error))) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }
  return *result;
}

FailureOr<gpu::MemDescType>
inferTMemIndexOpType(gpu::MemDescType srcTy, std::string *error) {
  if (srcTy.getRank() == 0) {
    if (error)
      *error = "result rank must be input rank - 1";
    return failure();
  }
  SmallVector<int64_t> dstShape = llvm::to_vector(srcTy.getShape().drop_front());
  SmallVector<int64_t> dstAllocShape =
      llvm::to_vector(srcTy.getAllocShape().drop_front());
  auto dstEncoding = srcTy.getEncoding();
  auto dstTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                    srcTy.getElementType(), dstEncoding,
                                    srcTy.getMemorySpace(),
                                    srcTy.getMutableMemory(), dstAllocShape,
                                    error);
  if (!dstTy)
    return failure();
  SmallVector<int64_t> srcShape(srcTy.getShape().begin(), srcTy.getShape().end());
  SmallVector<int64_t> dstShapeCopy(dstTy->getShape().begin(),
                                    dstTy->getShape().end());
  SmallVector<int64_t> dstAllocShapeCopy(dstTy->getAllocShape().begin(),
                                         dstTy->getAllocShape().end());
  if (auto preserved = tryPreserveOuterIndexedTMemEncoding(
          srcTy.getContext(), srcShape, dstShapeCopy, dstAllocShapeCopy,
          srcTy.getEncoding(), error)) {
    auto resultTy = tryCreateMemDescType(
        srcTy.getContext(), dstShape, srcTy.getElementType(), *preserved,
        srcTy.getMemorySpace(), srcTy.getMutableMemory(), dstAllocShape,
        error);
    if (!resultTy)
      return failure();
    return *resultTy;
  }
  auto maybeDstEnc =
      inferTMemIndexEncoding(srcShape, dstShapeCopy, dstAllocShapeCopy,
                             srcTy.getEncoding(), error);
  if (failed(maybeDstEnc)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }
  auto resultTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                       srcTy.getElementType(), *maybeDstEnc,
                                       srcTy.getMemorySpace(),
                                       srcTy.getMutableMemory(), dstAllocShape,
                                       error);
  if (!resultTy)
    return failure();
  return *resultTy;
}

FailureOr<gpu::MemDescType>
inferTMemSubsliceOpType(gpu::MemDescType srcTy, ArrayRef<int64_t> dstShape,
                        ArrayRef<int32_t> offsets, std::string *error) {
  auto dstEncoding = srcTy.getEncoding();
  auto dstAllocShape = llvm::to_vector(srcTy.getAllocShape());
  auto dstTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                    srcTy.getElementType(), dstEncoding,
                                    srcTy.getMemorySpace(),
                                    srcTy.getMutableMemory(), dstAllocShape,
                                    error);
  if (!dstTy)
    return failure();
  SmallVector<int64_t> srcShape(srcTy.getShape().begin(), srcTy.getShape().end());
  SmallVector<int64_t> dstShapeCopy(dstTy->getShape().begin(),
                                    dstTy->getShape().end());
  auto maybeDstEnc = inferTMemSubsliceEncoding(srcShape, srcTy.getEncoding(),
                                               dstShapeCopy, offsets, error);
  if (failed(maybeDstEnc)) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
  }
  auto resultTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                       srcTy.getElementType(), *maybeDstEnc,
                                       srcTy.getMemorySpace(),
                                       srcTy.getMutableMemory(), dstAllocShape,
                                       error);
  if (!resultTy)
    return failure();
  return *resultTy;
}

FailureOr<gpu::MemDescType>
inferTMemReshapeOpType(gpu::MemDescType srcTy, ArrayRef<int64_t> dstShape,
                       std::string *error) {
  auto maybeSrcLayout =
      getTMemViewAnalysisLayout(srcTy.getShape(), srcTy.getEncoding(), error);
  if (!maybeSrcLayout)
    return failure();

  auto *ctx = srcTy.getContext();
  auto srcShape = srcTy.getShape();
  auto layoutSrcShape = srcShape;
  auto layoutDstShape = dstShape;
  int64_t layoutElems =
      static_cast<int64_t>(maybeSrcLayout->layout.getTotalOutDimSize());

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape) {
    while (!shape.empty() && shape.front() == 1 &&
           product<int64_t>(shape.drop_front()) >= layoutElems)
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape);
  layoutDstShape = stripLeadingUnitDims(layoutDstShape);
  while (static_cast<size_t>(maybeSrcLayout->layout.getNumOutDims()) >
         layoutSrcShape.size()) {
    auto firstDim = *maybeSrcLayout->layout.getOutDimNames().begin();
    if (maybeSrcLayout->layout.getOutDimSize(firstDim) != 1)
      break;
    maybeSrcLayout->layout = maybeSrcLayout->layout.squeezeOuts(firstDim);
  }
  layoutElems =
      static_cast<int64_t>(maybeSrcLayout->layout.getTotalOutDimSize());

  if (product<int64_t>(layoutSrcShape) > layoutElems) {
    if (layoutSrcShape.size() !=
        static_cast<size_t>(maybeSrcLayout->layout.getNumOutDims()) + 1) {
      if (error)
        *error = "TMEM reshape requires the descriptor rank to match the TMEM "
                 "layout rank or have one leading multibuffer dimension";
      return failure();
    }
    if (layoutDstShape.empty() || layoutDstShape.front() != layoutSrcShape.front()) {
      if (error)
        *error = "TMEM reshape must preserve the leading multibuffer dimension";
      return failure();
    }
    layoutSrcShape = layoutSrcShape.drop_front();
    layoutDstShape = layoutDstShape.drop_front();
  }

  if (product(layoutSrcShape) != product(layoutDstShape)) {
    if (error)
      *error = "dst shape has different number of elements than src";
    return failure();
  }
  if (!llvm::equal(maybeSrcLayout->layout.getOutDimSizes(), layoutSrcShape)) {
    auto maybeRestrictedLayout = restrictTMemAnalysisLayoutToShape(
        maybeSrcLayout->layout, layoutSrcShape, error);
    if (failed(maybeRestrictedLayout)) {
      if (error && error->empty())
        *error = "TMEM reshape rank does not match the canonical TMEM layout";
      return failure();
    }
    maybeSrcLayout->layout = *maybeRestrictedLayout;
    layoutElems =
        static_cast<int64_t>(maybeSrcLayout->layout.getTotalOutDimSize());
  }
  if (layoutElems != product<int64_t>(layoutSrcShape)) {
    if (error)
      *error = "TMEM reshape rank does not match the canonical TMEM layout";
    return failure();
  }

  auto dstLL = reshapeLayout(ctx, maybeSrcLayout->layout, layoutDstShape);
  auto result =
      tryMakeTMemViewEncoding(ctx, std::move(dstLL), maybeSrcLayout->twoCTAs,
                              error);
  if (!result)
    return failure();

  SmallVector<int64_t> dstAllocShape = llvm::to_vector(
      srcTy.getAllocShape().take_front(srcTy.getAllocShape().size() -
                                       srcTy.getShape().size()));
  auto maybeDstAllocTail = getTMemAllocShapeForEncoding(dstShape, *result, error);
  if (!maybeDstAllocTail)
    return failure();
  dstAllocShape.append(maybeDstAllocTail->begin(), maybeDstAllocTail->end());
  auto resultTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                       srcTy.getElementType(), *result,
                                       srcTy.getMemorySpace(),
                                       srcTy.getMutableMemory(), dstAllocShape,
                                       error);
  if (!resultTy)
    return failure();
  return *resultTy;
}

// Get the maximum number of registers per thread based on the context. This is
// by default 256, but it can be overridden by `ttg.maxnreg` set on the module
// or a contextual register limit set by the compiler on partitions.
int getContextualMaxNReg(Operation *op) {
  // Check the immediate parent op to see if it places a register constraint.
  auto getFromParent = [](Operation *op) -> std::optional<int> {
    Operation *parent = op->getParentOp();
    if (auto mod = dyn_cast<ModuleOp>(parent)) {
      if (auto attr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
        return attr.getInt();
      return {};
    }

    if (auto partitions = dyn_cast<WarpSpecializePartitionsOp>(parent)) {
      // Check if the partition has reduced registers.
      unsigned idx = op->getParentRegion()->getRegionNumber();
      if (auto actRegisters = partitions.getParentOp().getActualRegisters())
        return (*actRegisters)[1 + idx];
      return {};
    }

    if (auto wsOp = dyn_cast<WarpSpecializeOp>(op->getParentOp())) {
      // Check the register usage of the default warpgroup.
      if (auto actRegisters = wsOp.getActualRegisters())
        return actRegisters->front();
      return {};
    }

    return {};
  };

  // PTXAS validates the register usage of `tcgen05.ld` and `tcgen05.st`
  // instructions based on the static number of registers set on the module, not
  // the dynamic allocation. This just means the register limit used for the
  // purpose of subtiling TMEM messages cannot be higher than the module's.
  auto mod = op->getParentOfType<ModuleOp>();
  int maxnreg = maxRegisters;

  for (; op != mod; op = op->getParentOp()) {
    if (std::optional<int> limit = getFromParent(op)) {
      maxnreg = std::min(maxnreg, *limit);
      break;
    }
  }

  if (auto maxnregAttr = mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName))
    maxnreg = std::min<int>(maxnreg, maxnregAttr.getInt());

  return maxnreg;
}

FailureOr<TMemLdStEncodingInfo>
lowerTMemLdSt(const LinearLayout &cvt, int maxnreg, int bitwidth,
              std::function<InFlightDiagnostic()> emitError,
              bool unpacked, ArrayRef<int32_t> warpBasis0,
              ArrayRef<int32_t> warpBasis1, int32_t rowSpan = 128,
              bool preferI16x32bx2 = false) {
  // We will fill in the returned value recursively (if it exists)

  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto info =
        lowerTMemLdSt(prmtCvt, maxnreg, bitwidth, emitError, unpacked,
                      warpBasis0, warpBasis1, rowSpan, preferI16x32bx2);
    if (failed(info))
      return failure();
    info->broadcast = std::move(removeBroadcastSrc);
    return info;
  }
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kRow = S("row");
  auto kCol = S("col");
  auto basisIsWholeTileMultiple = [](ArrayRef<int32_t> basis, unsigned tileM,
                                     unsigned tileN) {
    return basis.size() == 2 && basis[0] % static_cast<int32_t>(tileM) == 0 &&
           basis[1] % static_cast<int32_t>(tileN) == 0;
  };
  auto warpBasesAlignToWholeTile = [&](TMemAccessAtom atom) {
    if (atom != TMemAccessAtom::I32x32b &&
        atom != TMemAccessAtom::I16x32bx2)
      return true;
    auto tile = getTileLayout(ctx, atom, unpacked, /*withWarp=*/false);
    unsigned tileRows = tile.getOutDimSize(kRow);
    unsigned tileCols = tile.getOutDimSize(kCol);
    return basisIsWholeTileMultiple(warpBasis0, tileRows, tileCols) &&
           basisIsWholeTileMultiple(warpBasis1, tileRows, tileCols);
  };
  auto getPhysicalOutDims =
      [&](const LinearLayout &layout) -> std::optional<std::pair<StringAttr, StringAttr>> {
    if (layout.hasOutDim(kRow) && layout.hasOutDim(kCol))
      return std::make_pair(kRow, kCol);
    if (layout.getNumOutDims() != 2)
      return std::nullopt;
    auto outDims = layout.getOutDimNames();
    auto it = outDims.begin();
    StringAttr first = *it++;
    StringAttr second = *it;
    return std::make_pair(first, second);
  };
  if (bitwidth < 32) {
    LinearLayout quot;
    int bestContig = 1;
    for (int contig = 1; bitwidth * contig <= 32; contig *= 2) {
      auto maybeQuot =
          divideLeft(cvt, LinearLayout::identity1D(contig, kReg, kCol));
      if (!maybeQuot)
        break;
      quot = *maybeQuot;
      bestContig = contig;
    }
    bool padding = false;
    int newBitwidth = bitwidth;
    if (bestContig > 1) {
      // There are contiguous elements along kCol, so we can pack them into a
      // larger dtype
      unpacked = false;
      newBitwidth = bitwidth * bestContig;
    } else if (auto maybeQuot = divideLeft(
                   cvt, LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) *
                            LinearLayout::identity1D(2, kReg, kCol));
               bitwidth == 16 && maybeQuot) {
      // Unpacked just supported for bitwidth 16
      unpacked = true;
      quot = *maybeQuot;
      newBitwidth = 32;
    } else if (auto maybeQuot = divideLeft(
                   cvt, LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth))) {
      // We software-pad the elements when we either do not have enough elements
      // to fill a full 32b register, e.g., colN = 1 and colStride != 1 or when
      // bitwidth == 8 (this happens with scales with K=1).
      // These two cases are mostly supported for testing purposes.
      unpacked = bitwidth == 16;
      quot = *maybeQuot;
      padding = true;
      newBitwidth = 32;
    } else {
      if (emitError) {
        emitError() << "Failed to lower TMEM load/store: TMEM layout is not "
                       "packed or unpacked";
      }
      return failure();
    }
    // True unpacked f16 views need an extra zero register basis when we widen
    // to 32-bit dword columns. Sparse support views that reached this branch
    // through the padding path are already expressed in dword columns; adding
    // the extra zero basis again can steer them onto the wrong 16x32bx2 family.
    if (unpacked && !(padding && bitwidth == 16)) {
      quot = LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) * quot;
    }
    auto info =
        lowerTMemLdSt(quot, maxnreg, newBitwidth, emitError, unpacked,
                      warpBasis0, warpBasis1, rowSpan, preferI16x32bx2);
    if (failed(info))
      return failure();
    if (bestContig > 1) {
      info->vec = bestContig;
    }
    if (unpacked) {
      info->unpacked = true;
    }
    if (padding) {
      info->padding = true;
    }
    return info;
  }

  assert(bitwidth == 32);

  auto tryCanonicalM64I32x32b = [&]() -> std::optional<TMemLdStEncodingInfo> {
    // 32x32b packets only admit warp anchors on whole 32-row message tiles.
    // Projected 64-row views can carry 16-row anchors; forcing those views
    // through I32x32b collapses a warp anchor into the lane tile and produces
    // the wrong direct decomposition instead of cleanly falling back to a
    // smaller realizable atom family.
    if (!warpBasesAlignToWholeTile(TMemAccessAtom::I32x32b))
      return std::nullopt;
    auto physOutDims = getPhysicalOutDims(cvt);
    if (!physOutDims)
      return std::nullopt;
    auto [rowDim, colDim] = *physOutDims;
    if (cvt.getOutDimSize(rowDim) != 64)
      return std::nullopt;
    int64_t n = cvt.getOutDimSize(colDim);
    if (n < 2 || !llvm::isPowerOf2_64(n))
      return std::nullopt;

    LinearLayout::BasesT bases;
    bases[kReg] = {};
    bases[kLane] = {{1, 0},
                    {2, 0},
                    {4, 0},
                    {8, 0},
                    {0, static_cast<int32_t>(n / 2)}};

    int32_t rowExtent = rowSpan;
    int32_t colExtent = static_cast<int32_t>(n);
    auto updateExtent = [&](ArrayRef<int32_t> basis) {
      assert(basis.size() == 2 && "TMEM warp bases must be 2D row/col vectors");
      rowExtent = std::max<int32_t>(rowExtent, basis[0] + 16);
      colExtent = std::max<int32_t>(colExtent, basis[1] + 2);
    };
    updateExtent(warpBasis0);
    updateExtent(warpBasis1);
    if (!llvm::isPowerOf2_32(static_cast<uint32_t>(rowExtent)))
      rowExtent = llvm::PowerOf2Ceil(static_cast<uint32_t>(rowExtent));
    if (!llvm::isPowerOf2_32(static_cast<uint32_t>(colExtent)))
      colExtent = llvm::PowerOf2Ceil(static_cast<uint32_t>(colExtent));

    bases[kWarp] = {{warpBasis0.begin(), warpBasis0.end()},
                    {warpBasis1.begin(), warpBasis1.end()}};
    LinearLayout tile(std::move(bases),
                      {{rowDim, rowExtent}, {colDim, colExtent}},
                      /*isSurjective=*/false);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      auto &reps = std::get<0>(*maybeReps);
      if (!getPhysicalOutDims(reps))
        maybeReps.reset();
    }
    if (!maybeReps)
      return std::nullopt;
    return TMemLdStEncodingInfo{TMemAccessAtom::I32x32b, std::get<0>(*maybeReps),
                                std::get<1>(*maybeReps),
                                std::get<2>(*maybeReps)};
  };

  if (!preferI16x32bx2) {
    if (auto info = tryCanonicalM64I32x32b())
      return *info;
  }

  auto tryI16x32bx2 = [&]() -> std::optional<TMemLdStEncodingInfo> {
    if (!warpBasesAlignToWholeTile(TMemAccessAtom::I16x32bx2))
      return std::nullopt;
    // Quotient by the smaller tile and then, if possible, we set the
    // secondHalfOffset to the last kLane basis
    auto tile = getTileLayout(ctx, TMemAccessAtom::I16x32bx2, unpacked,
                              /*withWarp=*/true, warpBasis0, warpBasis1,
                              rowSpan);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      auto &reps = std::get<0>(*maybeReps);
      // 16x32bx2 needs a distinct lane=16 basis in the repetition layout to
      // encode the second half offset. Some speculative candidate layouts
      // match the tile quotient but only have 16 active lanes; reject them
      // cleanly instead of indexing a non-existent lane basis. The
      // repetition layout must also already be expressed in physical TMEM
      // row/col coordinates; descriptor queries can still carry logical
      // dim0/dim1 out-dims at this point, and those must not trip a hard
      // crash while probing candidate atoms.
      auto laneIt = reps.getBases().find(kLane);
      if (laneIt == reps.getBases().end() || laneIt->second.size() <= 4 ||
          !getPhysicalOutDims(reps))
        maybeReps.reset();
    }
    if (maybeReps) {
      auto [reps, perm, numRegsPerMessage] = std::move(*maybeReps);
      auto physOutDims = getPhysicalOutDims(reps);
      if (!physOutDims)
        return std::nullopt;
      auto [rowDim, colDim] = *physOutDims;
      // Find the last kLane basis and use it as secondHalfOffset
      auto row = reps.getBasis(kLane, 4, rowDim);
      auto col = reps.getBasis(kLane, 4, colDim);
      std::optional<uint32_t> secondHalfOffset = (row << 16) | col;
      // We "quotient it out", meaning we remove the last basis from reps
      auto basis = reps.getBases();
      basis[kLane][4] = {0, 0};
      reps = LinearLayout(std::move(basis), reps.getOutDims(),
                          /*isSurjective=*/false);
      TMemLdStEncodingInfo info = {TMemAccessAtom::I16x32bx2, reps, perm,
                                   numRegsPerMessage};
      info.secondHalfOffset = secondHalfOffset;
      return info;
    }
    return std::nullopt;
  };

  if (preferI16x32bx2) {
    if (auto info = tryI16x32bx2())
      return *info;
  }

  // The algorithm goes as:
  // - Try to match the tile with one of the standard messages
  // - If it doesn't match, we use the 16x32bx2 message
  // Note that it can match one and only one of the layouts, even after register
  // reordering, as the layouts yield predetermined positions for the lanes
  // We store the instruction, the resulting reps layout, the permutation and
  // the number of registers per message
  std::optional<TMemLdStEncodingInfo> msgInfo;
  for (auto atom : {TMemAccessAtom::I32x32b, TMemAccessAtom::I16x256b,
                    TMemAccessAtom::I16x64b, TMemAccessAtom::I16x128b}) {
    if (!warpBasesAlignToWholeTile(atom))
      continue;
    auto tile =
        getTileLayout(ctx, atom, unpacked, /*withWarp=*/true, warpBasis0,
                      warpBasis1, rowSpan);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      auto &reps = std::get<0>(*maybeReps);
      // Direct TMEM messages require a 2D physical TMEM repetition space.
      // Canonical layouts use row/col names, but descriptor view-analysis can
      // preserve equivalent 2D physical dimensions as dim0/dim1.
      if (!getPhysicalOutDims(reps))
        maybeReps.reset();
    }
    if (maybeReps) {
      // Cannot match more than one
      msgInfo = {atom, std::get<0>(*maybeReps), std::get<1>(*maybeReps),
                 std::get<2>(*maybeReps)};
      break;
    }
  }
  if (!msgInfo)
    msgInfo = tryI16x32bx2();

  if (!msgInfo) {
    if (emitError) {
      emitError()
          << "Failed to lower TMEM load/store: unsupported dst layout\n" +
                 cvt.toString();
    }
    return failure();
  }
  return std::move(*msgInfo);
}

FailureOr<TMemLdStEncodingInfo>
lowerTMemLdSt(const LinearLayout &cvt, int maxnreg, int bitwidth,
              std::function<InFlightDiagnostic()> emitError,
              bool unpacked = false, int32_t warpRow0 = 32,
              int32_t warpRow1 = 64, int32_t rowSpan = 128,
              bool preferI16x32bx2 = false) {
  SmallVector<int32_t> warpBasis0 = {warpRow0, 0};
  SmallVector<int32_t> warpBasis1 = {warpRow1, 0};
  return lowerTMemLdSt(cvt, maxnreg, bitwidth, emitError, unpacked,
                       warpBasis0, warpBasis1, rowSpan, preferI16x32bx2);
}

static FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfoImpl(
    RankedTensorType regTy, MemDescType memTy, LinearLayout memLayout,
    int maxnreg, std::function<InFlightDiagnostic()> emitError,
    std::optional<TMemLdStRowPlan> rowPlanOverride) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_HALFROWS") != nullptr;
  bool debugScales = std::getenv("TRITON_DEBUG_TMEM_SCALES") != nullptr;
  auto *ctx = regTy.getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kBlock = S("block");
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kWarp = S("warp");
  auto kRow = S("row");
  auto kCol = S("col");
  auto squeezeTrivialBlock = [&](LinearLayout layout) {
    if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1)
      layout = layout.squeezeIns(kBlock);
    if (layout.hasOutDim(kBlock) && layout.getOutDimSize(kBlock) == 1)
      layout = layout.squeezeOuts(kBlock);
    return layout;
  };
  LinearLayout originalMemLayout = squeezeTrivialBlock(toLinearLayout(memTy));
  memLayout = squeezeTrivialBlock(std::move(memLayout));
  if (memLayout.getNumOutDims() == 0)
    return failure();
  auto hasZeroBasisAlong = [](const LinearLayout &layout, StringAttr dim) {
    if (!layout.hasInDim(dim))
      return false;
    unsigned dimBits = layout.getInDimSizeLog2(dim);
    for (unsigned idx = 0; idx < dimBits; ++idx) {
      if (llvm::all_of(layout.getBasis(dim, idx),
                       [](int32_t value) { return value == 0; })) {
        return true;
      }
    }
    return false;
  };
  int bitwidth = memTy.getElementTypeBitWidth();
  int64_t logicalRows = memTy.getShape()[memTy.getRank() - 2];
  int64_t logicalCols = memTy.getShape()[memTy.getRank() - 1];
  auto memOutDims = llvm::to_vector(memLayout.getOutDims());
  int64_t supportLogicalRows =
      memOutDims.size() >= 2 ? memOutDims[memOutDims.size() - 2].second
                             : logicalRows;
  int64_t supportLogicalCols =
      memOutDims.size() >= 1 ? memOutDims.back().second : logicalCols;
  bool hasWidenedSupportCols =
      supportLogicalRows == logicalRows &&
      supportLogicalCols == logicalCols * 2;
  int64_t physicalRows = memLayout.hasInDim(kRow) ? memLayout.getInDimSize(kRow)
                                                  : logicalRows;
  int64_t physicalCols =
      memLayout.hasInDim(kCol) ? memLayout.getInDimSize(kCol) / (32 / bitwidth)
                               : logicalCols;
  auto originalActiveMemLayout = originalMemLayout;
  if (originalActiveMemLayout.hasInDim(kRow))
    originalActiveMemLayout =
        originalActiveMemLayout.removeZeroBasesAlongDim(kRow);
  int64_t originalActivePhysicalRows =
      originalActiveMemLayout.hasInDim(kRow)
          ? originalActiveMemLayout.getInDimSize(kRow)
          : logicalRows;
  int64_t originalPhysicalCols =
      originalMemLayout.hasInDim(kCol)
          ? originalMemLayout.getInDimSize(kCol) / (32 / bitwidth)
          : logicalCols;
  bool isOriginalRowZeroLiftedReinterpret =
      bitwidth == 16 && hasZeroBasisAlong(originalMemLayout, kRow) &&
      !hasZeroBasisAlong(originalMemLayout, kCol) &&
      logicalRows == originalActivePhysicalRows &&
      logicalCols == originalPhysicalCols * 2;
  bool isRowZeroM64ReinterpretView =
      bitwidth == 16 && logicalRows == 64 && isOriginalRowZeroLiftedReinterpret;
  bool isScales = isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding());
  auto activeMemLayout = memLayout;
  if (activeMemLayout.hasInDim(kRow))
    activeMemLayout = activeMemLayout.removeZeroBasesAlongDim(kRow);
  int64_t activePhysicalRows =
      activeMemLayout.hasInDim(kRow) ? activeMemLayout.getInDimSize(kRow)
                                     : physicalRows;
  bool hasZeroRowBasis = hasZeroBasisAlong(memLayout, kRow);
  bool hasZeroColBasis = hasZeroBasisAlong(memLayout, kCol);
  bool isColScaledPackedBitcast =
      bitwidth == 16 && !hasZeroRowBasis && !hasZeroColBasis &&
      memLayout.hasInDim(kCol) && logicalRows == activePhysicalRows &&
      logicalCols == memLayout.getInDimSize(kCol) * 2;
  // Zero row/col bases are part of the descriptor layout contract: they
  // describe broadcast/support bits in the logical view, not disposable
  // physical storage. Direct planning must keep those bases so loads, stores,
  // and MMA users agree on the same logical-to-physical TMEM projection.
  LinearLayout directPlanningMemLayout = memLayout;
  bool hasPackedHalfRowAndColSupport =
      hasZeroRowBasis && hasZeroColBasis &&
      logicalRows == activePhysicalRows * 2 &&
      logicalCols == physicalCols * 2;
  LinearLayout regLayout =
      squeezeTrivialBlock(toLinearEncoding(regTy).getLinearLayout());
  auto tryBuildPackedI16SparseSupportLayout =
      [&](bool stripZeroRowSupportBit)
      -> std::optional<LinearLayout> {
    bool hasNonTrivialBlock =
        memLayout.hasInDim(kBlock) && memLayout.getInDimSize(kBlock) > 1;
    if (hasNonTrivialBlock)
      return std::nullopt;
    bool isRowZeroLiftedReinterpret =
        hasZeroRowBasis && !hasZeroColBasis &&
        logicalRows == activePhysicalRows && logicalCols == physicalCols * 2;
    if (bitwidth != 16 || isScales || !memLayout.hasInDim(kCol) ||
        (!(logicalRows == activePhysicalRows) &&
         !hasPackedHalfRowAndColSupport) ||
        (!(hasZeroColBasis &&
           (hasWidenedSupportCols || hasPackedHalfRowAndColSupport)) &&
         logicalCols <= physicalCols) ||
        (!hasZeroColBasis && !isRowZeroLiftedReinterpret &&
         !isColScaledPackedBitcast)) {
      if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
        llvm::errs() << "[tmem-ldst] packed16 support precondition fail: bitwidth="
                     << bitwidth << " isScales=" << isScales
                     << " zeroRow=" << hasZeroRowBasis
                     << " zeroCol=" << hasZeroColBasis
                     << " rowZeroLifted=" << isRowZeroLiftedReinterpret
                     << " packedHalfRowAndColSupport="
                     << hasPackedHalfRowAndColSupport
                     << " widenedSupportCols=" << hasWidenedSupportCols
                     << " hasCol=" << memLayout.hasInDim(kCol)
                     << " logicalRows=" << logicalRows
                     << " activePhysicalRows=" << activePhysicalRows
                     << " logicalCols=" << logicalCols
                     << " supportLogicalCols=" << supportLogicalCols
                     << " physicalCols=" << physicalCols << '\n';
      }
      return std::nullopt;
    }

    auto bases = memLayout.getBases();
    auto colIt = bases.find(kCol);
    if (colIt == bases.end())
      return std::nullopt;
    auto isZeroBasis = [](ArrayRef<int32_t> basis) {
      return llvm::all_of(basis, [](int32_t value) { return value == 0; });
    };
    auto getBasisOutIdx = [&](ArrayRef<int32_t> basis)
        -> std::optional<unsigned> {
      std::optional<unsigned> basisOutIdx;
      for (auto [idx, value] : llvm::enumerate(basis)) {
        if (value == 0)
          continue;
        if (basisOutIdx)
          return std::nullopt;
        basisOutIdx = idx;
      }
      return basisOutIdx;
    };

    std::optional<unsigned> colOutIdx;
    for (ArrayRef<int32_t> basis : colIt->second) {
      auto basisOutIdx = getBasisOutIdx(basis);
      if (!basisOutIdx)
        continue;
      if (colOutIdx && *colOutIdx != *basisOutIdx)
        return std::nullopt;
      colOutIdx = basisOutIdx;
    }
    if (!colOutIdx) {
      if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr)
        llvm::errs() << "[tmem-ldst] packed16 support missing colOutIdx" << '\n';
      return std::nullopt;
    }

    auto eraseIt = colIt->second.end();
    bool shiftPackedColDim = false;
    if (hasZeroColBasis) {
      auto zeroColBasisCount = llvm::count_if(colIt->second, isZeroBasis);
      if (zeroColBasisCount != 1) {
        if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
          llvm::errs() << "[tmem-ldst] packed16 support zero-col-basis-count="
                       << zeroColBasisCount << '\n';
        }
        return std::nullopt;
      }
      eraseIt = llvm::find_if(colIt->second, isZeroBasis);
    } else if (isColScaledPackedBitcast) {
      shiftPackedColDim = true;
    } else {
      eraseIt = llvm::find_if(colIt->second, [&](ArrayRef<int32_t> basis) {
        auto basisOutIdx = getBasisOutIdx(basis);
        return basisOutIdx && *basisOutIdx == *colOutIdx &&
               basis[*colOutIdx] == 1;
      });
      if (eraseIt == colIt->second.end()) {
        if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
          llvm::errs() << "[tmem-ldst] packed16 support missing row-zero pack bit\n";
        }
        return std::nullopt;
      }
      shiftPackedColDim = true;
    }
    if (eraseIt != colIt->second.end())
      colIt->second.erase(eraseIt);

    auto outDims = llvm::to_vector(memLayout.getOutDims());
    if (outDims[*colOutIdx].second % 2 != 0)
      return std::nullopt;
    if (shiftPackedColDim) {
      for (auto &dimBases : llvm::make_second_range(bases)) {
        for (auto &basis : dimBases) {
          if ((basis[*colOutIdx] & 1) != 0)
            return std::nullopt;
          basis[*colOutIdx] >>= 1;
        }
      }
    }
    outDims[*colOutIdx].second /= 2;
    if (hasWidenedSupportCols) {
      int32_t packedLogicalCols = static_cast<int32_t>(logicalCols / 2);
      if (packedLogicalCols <= 0 ||
          outDims[*colOutIdx].second < packedLogicalCols)
        return std::nullopt;
      auto supportBandIt = llvm::find_if(
          colIt->second, [&](ArrayRef<int32_t> basis) {
            if (basis[*colOutIdx] != packedLogicalCols)
              return false;
            for (auto [idx, value] : llvm::enumerate(basis)) {
              if (idx == *colOutIdx)
                continue;
              if (value != 0)
                return false;
            }
            return true;
          });
      if (supportBandIt == colIt->second.end())
        return std::nullopt;
      colIt->second.erase(supportBandIt);
      outDims[*colOutIdx].second = packedLogicalCols;
    }

    if (stripZeroRowSupportBit && hasPackedHalfRowAndColSupport) {
      auto rowIt = bases.find(kRow);
      if (rowIt == bases.end())
        return std::nullopt;
      auto zeroRowIt = llvm::find_if(rowIt->second, isZeroBasis);
      if (zeroRowIt == rowIt->second.end())
        return std::nullopt;
      rowIt->second.erase(zeroRowIt);
    }

    return LinearLayout(std::move(bases), std::move(outDims),
                        /*requireSurjective=*/false);
  };
  auto tryPackedI16SparseSupportInfo = [&]()
      -> std::optional<TMemLdStEncodingInfo> {
    bool debugQuery = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
    auto maybePackedMemLayout =
        tryBuildPackedI16SparseSupportLayout(/*stripZeroRowSupportBit=*/true);
    if (!maybePackedMemLayout) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: no packed mem layout\n";
      }
      return std::nullopt;
    }

    SmallVector<int64_t> regShape(regTy.getShape().begin(), regTy.getShape().end());
    if (regShape.empty() || regShape.back() % 2 != 0)
      return std::nullopt;
    SmallVector<int64_t> packedRegShape(regShape.begin(), regShape.end());
    packedRegShape.back() /= 2;
    std::string packedRegError;
    auto maybePackedRegLayout = reinterpretLinearLayout(
        regShape, /*srcBitwidth=*/bitwidth, regLayout, packedRegShape,
        /*dstBitwidth=*/32, ctx, &packedRegError);
    if (failed(maybePackedRegLayout)) {
      if (debugQuery) {
        llvm::errs()
            << "[tmem-ldst] packed16 support skip: reg layout reinterpret "
               "failed: "
            << packedRegError << "\n"
            << regLayout.toString() << "\n";
      }
      return std::nullopt;
    }
    auto packedRegLayout = squeezeTrivialBlock(std::move(*maybePackedRegLayout));
    auto stripLeadingZeroRegisterBasis = [&](LinearLayout layout)
        -> std::optional<LinearLayout> {
      if (!layout.hasInDim(kReg))
        return std::nullopt;
      auto bases = layout.getBases();
      auto regIt = bases.find(kReg);
      if (regIt == bases.end() || regIt->second.empty())
        return std::nullopt;
      auto isZeroBasis = [](ArrayRef<int32_t> basis) {
        return llvm::all_of(basis, [](int32_t value) { return value == 0; });
      };
      if (!isZeroBasis(regIt->second.front()))
        return std::nullopt;
      regIt->second.erase(regIt->second.begin());
      auto outDims = llvm::to_vector(layout.getOutDims());
      return LinearLayout(std::move(bases), std::move(outDims),
                          /*requireSurjective=*/layout.isSurjective());
    };
    auto maybePackedRegisterSpaceLayout =
        stripLeadingZeroRegisterBasis(packedRegLayout);
    if (!maybePackedRegisterSpaceLayout) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: packed register "
                        "reinterpret did not expose a leading zero register "
                        "basis\n"
                     << packedRegLayout.toString() << "\n";
      }
      return std::nullopt;
    }
    packedRegLayout = squeezeTrivialBlock(std::move(*maybePackedRegisterSpaceLayout));
    auto packedMemLayout =
        squeezeTrivialBlock(std::move(*maybePackedMemLayout));
    auto anchorMemLayout = packedMemLayout;
    if (hasPackedHalfRowAndColSupport) {
      auto maybePackedAnchorMemLayout =
          tryBuildPackedI16SparseSupportLayout(
              /*stripZeroRowSupportBit=*/false);
      if (!maybePackedAnchorMemLayout) {
        if (debugQuery) {
          llvm::errs()
              << "[tmem-ldst] packed16 support skip: no anchor mem layout\n";
        }
        return std::nullopt;
      }
      anchorMemLayout =
          squeezeTrivialBlock(std::move(*maybePackedAnchorMemLayout));
    }
    if (debugQuery) {
      llvm::errs() << "[tmem-ldst] packed16 support reg layout:\n"
                   << packedRegLayout.toString() << "\n";
      llvm::errs() << "[tmem-ldst] packed16 support mem layout:\n"
                   << packedMemLayout.toString() << "\n";
      if (anchorMemLayout != packedMemLayout) {
        llvm::errs() << "[tmem-ldst] packed16 support anchor layout:\n"
                     << anchorMemLayout.toString() << "\n";
      }
    }

    auto rowPlan = getTMemLdStRowPlanForType(memTy);
    auto memLayoutSupportsOverride = [&](const LinearLayout &layout) {
      if (!rowPlanOverride)
        return false;
      return layout.hasInDim(kRow) &&
             layout.getInDimSize(kRow) == rowPlanOverride->rowSpan;
    };
    if (rowPlanOverride &&
        (!rowPlan || rowPlanOverride->rowSpan == rowPlan->rowSpan ||
         memLayoutSupportsOverride(packedMemLayout) ||
         memLayoutSupportsOverride(anchorMemLayout))) {
      rowPlan = rowPlanOverride;
    }
    if (debugQuery && rowPlan) {
      llvm::errs() << "[tmem-ldst] packed16 support rowPlan="
                   << rowPlan->warpRow0 << "," << rowPlan->warpRow1
                   << " span=" << rowPlan->rowSpan
                   << " base=" << rowPlan->baseOffset << "\n";
    }
    if (!rowPlan || !packedRegLayout.hasInDim(kWarp) ||
        packedRegLayout.getInDimSizeLog2(kWarp) < 2)
      return std::nullopt;

    auto getRowAnchorBasis = [&](int32_t logicalRow)
        -> std::optional<SmallVector<int32_t>> {
      return getLogicalRowAnchorBasis(anchorMemLayout, logicalRow);
    };
    auto expectedWarp0Basis = getRowAnchorBasis(rowPlan->warpRow0);
    auto expectedWarp1Basis = getRowAnchorBasis(rowPlan->warpRow1);
    auto basisAllZero = [](const auto &basis) {
      return llvm::all_of(basis, [](int32_t value) { return value == 0; });
    };
    if (expectedWarp0Basis && expectedWarp1Basis &&
        basisAllZero(*expectedWarp0Basis) && !basisAllZero(*expectedWarp1Basis)) {
      if (auto liftedWarp1Basis = getRowAnchorBasis(rowPlan->warpRow1 * 2))
        expectedWarp1Basis = std::move(liftedWarp1Basis);
    }
    if (!expectedWarp0Basis || !expectedWarp1Basis) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: missing row anchors\n";
      }
      return std::nullopt;
    }

    if (!canInvertAndComposeSafely(packedRegLayout, packedMemLayout)) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: packed image containment failed\n";
      }
      return std::nullopt;
    }
    auto packedCvt = packedRegLayout.invertAndCompose(packedMemLayout);
    packedCvt = squeezeTrivialBlock(std::move(packedCvt));
    if (debugQuery) {
      llvm::errs() << "[tmem-ldst] packed16 support cvt\n"
                   << packedCvt.toString() << "\n";
    }
    bool hasBlockIn = packedCvt.hasInDim(kBlock);
    bool hasBlockOut = packedCvt.hasOutDim(kBlock);
    if (hasBlockIn != hasBlockOut)
      return std::nullopt;
    if (hasBlockIn) {
      auto maybeSublayout = packedCvt.quotient({kBlock});
      if (!maybeSublayout)
        return std::nullopt;
      packedCvt = *maybeSublayout;
    }

    if (!packedCvt.hasOutDim(kRow) || !packedCvt.hasInDim(kWarp) ||
        packedCvt.getInDimSizeLog2(kWarp) < 2) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: packed cvt missing row/warp structure\n";
      }
      return std::nullopt;
    }
    int32_t packedRowSpan = packedCvt.getOutDimSize(kRow);
    SmallVector<int32_t> packedWarpBasis0(packedCvt.getBasis(kWarp, 0).begin(),
                                          packedCvt.getBasis(kWarp, 0).end());
    SmallVector<int32_t> packedWarpBasis1(packedCvt.getBasis(kWarp, 1).begin(),
                                          packedCvt.getBasis(kWarp, 1).end());
    bool preferPackedI16x32bx2 = isRowZeroM64ReinterpretView;
    // Row-zero-lifted M64 reinterpret views use the bounded x2 search so the
    // packed 16x32bx2 path keeps the same row-anchor semantics. Canonical
    // packed split-N layouts without this row-zero signature keep the normal
    // direct vectorization and can lower to larger x16/x32 message shapes.
    int packedDirectMaxNreg =
        preferPackedI16x32bx2 ? std::min(maxnreg, 4) : maxnreg;
    auto info = lowerTMemLdSt(packedCvt, packedDirectMaxNreg, /*bitwidth=*/32,
                              /*emitError=*/{}, /*unpacked=*/false,
                              packedWarpBasis0, packedWarpBasis1,
                              packedRowSpan,
                              /*preferI16x32bx2=*/preferPackedI16x32bx2);
    if (failed(info)) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: rowSpan="
                     << packedRowSpan << " warp0=" << packedWarpBasis0.front()
                     << " warp1=" << packedWarpBasis1.front() << "\n";
        auto diagInfo = lowerTMemLdSt(
            packedCvt, packedDirectMaxNreg, /*bitwidth=*/32,
            [&]() -> InFlightDiagnostic {
              llvm::errs() << "[tmem-ldst] packed16 support lower failure: ";
              return mlir::emitRemark(mlir::UnknownLoc::get(ctx));
            },
            /*unpacked=*/false, packedWarpBasis0, packedWarpBasis1,
            packedRowSpan);
        (void)diagInfo;
        llvm::errs() << "[tmem-ldst] packed16 support skip: lower atom=";
        if (succeeded(info))
          llvm::errs() << static_cast<int>(info->atom);
        else
          llvm::errs() << "fail";
        llvm::errs() << "\n";
      }
      return std::nullopt;
    }
    if (debugQuery) {
      llvm::errs() << "[tmem-ldst] packed16 support hit atom="
                   << static_cast<int>(info->atom)
                   << " regsPerMsg=" << info->numRegsPerMessage << "\n";
    }

    auto packTMemBasisOffset = [](ArrayRef<int32_t> basis) -> uint32_t {
      assert(basis.size() == 2 && "TMEM warp bases must be 2D row/col vectors");
      return (static_cast<uint32_t>(basis[0]) << 16) |
             static_cast<uint32_t>(basis[1]);
    };
    auto getStaticOffsetFromLayout = [&](const LinearLayout &layout,
                                         int32_t regIdx,
                                         int32_t colScale) -> int32_t {
      auto rowCol = layout.apply({{kReg, regIdx}, {kLane, 0}, {kWarp, 0}});
      int32_t row = 0;
      int32_t col = 0;
      for (auto [dim, value] : rowCol) {
        if (dim == kRow)
          row = value;
        else if (dim == kCol)
          col = value;
      }
      return (row << 16) | (col * colScale);
    };
    auto getPackedStaticOffset = [&](int32_t regIdx, int32_t colScale) -> int32_t {
      return getStaticOffsetFromLayout(packedCvt, regIdx, colScale);
    };
    bool isPackedRowZeroLiftedView =
        hasZeroBasisAlong(packedMemLayout, kRow) &&
        !hasZeroBasisAlong(packedMemLayout, kCol);
    bool allowPackedI16RowZeroM64 =
        isRowZeroM64ReinterpretView &&
        info->atom == TMemAccessAtom::I16x32bx2;
    if (isPackedRowZeroLiftedView && info->atom != TMemAccessAtom::I32x32b &&
        !allowPackedI16RowZeroM64) {
      if (debugQuery) {
        llvm::errs() << "[tmem-ldst] packed16 support skip: row-zero lifted "
                        "views require packed 32x32b.unpack direct lowering "
                        "or the validated M64 16x32bx2 path\n";
      }
      return std::nullopt;
    }
    if (info->atom == TMemAccessAtom::I32x32b) {
      // Most sparse f16 support views need the native 32x32b unpack path: each
      // logical 16-bit element occupies a distinct selected TMEM dword column.
      //
      // A descriptor bitcast from a 32-bit TMEM region to a 16-bit logical
      // view is different. Its support image is already the exact selected
      // dword columns, and the logical f16 values must be packed into those
      // dwords rather than expanded across twice as many physical columns.
      bool packF16PairsIntoSelectedDwords = isColScaledPackedBitcast;
      info->unpacked = !packF16PairsIntoSelectedDwords;
      if (packF16PairsIntoSelectedDwords)
        info->vec = 2;
      info->packetOffsets.clear();
      info->packetOffsets.reserve(
          packedCvt.hasInDim(kReg)
              ? packedCvt.getInDimSize(kReg) / info->numRegsPerMessage
              : 0);
      bool isRowZeroLiftedPackedSupport =
          hasZeroBasisAlong(packedMemLayout, kRow) &&
          !hasZeroBasisAlong(packedMemLayout, kCol);
      int32_t numMessages =
          packedCvt.getInDimSize(kReg) / info->numRegsPerMessage;
      int32_t packedBandMessages =
          isRowZeroLiftedPackedSupport ? std::max<int32_t>(1, numMessages / 2) : 0;
      for (int32_t messageIdx = 0; messageIdx < numMessages; ++messageIdx) {
        int32_t staticOffset =
            getPackedStaticOffset(messageIdx * info->numRegsPerMessage,
                                  packF16PairsIntoSelectedDwords
                                      ? /*colScale=*/1
                                      : /*colScale=*/2);
        if (isRowZeroLiftedPackedSupport)
          staticOffset +=
              (messageIdx / packedBandMessages) * packedBandMessages * 2;
        info->packetOffsets.push_back(staticOffset);
      }
    } else if (info->atom == TMemAccessAtom::I16x32bx2 &&
               allowPackedI16RowZeroM64) {
      // The row-zero M64 reinterpret bucket is identified in the original
      // logical view space. After exact packing/reinterpret arithmetic the
      // packed sparse-support image is no longer necessarily "row-zero
      // lifted", but it still needs the validated quotient-derived packet
      // decomposition and subword-unpacked lowering.
      info->unpacked = true;
      auto packetOriginLayout = info->reps;
      if (info->secondHalfOffset && packetOriginLayout.hasInDim(kReg) &&
          packetOriginLayout.hasOutDim(kCol)) {
        auto packetBases = packetOriginLayout.getBases();
        auto &regBases = packetBases[kReg];
        for (auto &basis : regBases) {
          bool pureColBasis = packetOriginLayout.hasOutDim(kRow)
                                  ? packetOriginLayout.getBasis(kReg,
                                                                &basis -
                                                                    regBases.data(),
                                                                kRow) == 0
                                  : true;
          int32_t colBasis = packetOriginLayout.getBasis(
              kReg, &basis - regBases.data(), kCol);
          if (pureColBasis && colBasis != 0 &&
              colBasis * 2 == static_cast<int32_t>(*info->secondHalfOffset)) {
            std::fill(basis.begin(), basis.end(), 0);
            break;
          }
        }
        packetOriginLayout =
            LinearLayout(std::move(packetBases), packetOriginLayout.getOutDims(),
                         /*requireSurjective=*/packetOriginLayout.isSurjective());
      }
      info->packetOffsets.clear();
      info->packetOffsets.reserve(
          info->reps.hasInDim(kReg)
              ? info->reps.getInDimSize(kReg) / info->numRegsPerMessage
              : 0);
      if (info->reps.hasInDim(kReg)) {
        for (int32_t regIdx = 0; regIdx < info->reps.getInDimSize(kReg);
             regIdx += info->numRegsPerMessage) {
          info->packetOffsets.push_back(
              getStaticOffsetFromLayout(packetOriginLayout, regIdx,
                                        /*colScale=*/2));
        }
      }
    }
    if (allowPackedI16RowZeroM64) {
      // The sparse support image is already represented by packetOffsets.
      // Keep the validated split-N warp tiling contract here; replacing it
      // with the lifted support-image anchors collapses the 4-warp M=64
      // message family onto only two 16-row bands.
      if (info->packetOffsets.empty())
        info->secondHalfOffset = *info->secondHalfOffset * 2;
      info->warpBaseOffset0 =
          packTMemBasisOffset({rowPlan->warpRow0, /*col=*/0});
      info->warpBaseOffset1 =
          packTMemBasisOffset({rowPlan->warpRow1, /*col=*/0});
      info->warpRow0 = rowPlan->warpRow0;
      info->warpRow1 = rowPlan->warpRow1;
    } else {
      info->warpBaseOffset0 = packTMemBasisOffset(*expectedWarp0Basis);
      info->warpBaseOffset1 = packTMemBasisOffset(*expectedWarp1Basis);
      info->warpRow0 =
          expectedWarp0Basis->empty() ? 0 : expectedWarp0Basis->front();
      info->warpRow1 =
          expectedWarp1Basis->empty() ? 0 : expectedWarp1Basis->front();
    }
    int32_t packedBaseOffset = 0;
    if (rowPlanOverride && rowPlanOverride->rowSpan == packedRowSpan)
      packedBaseOffset = rowPlanOverride->baseOffset;
    else if (rowPlan && rowPlan->rowSpan == packedRowSpan)
      packedBaseOffset = rowPlan->baseOffset;
    info->baseOffset = packedBaseOffset;
    if (debugQuery) {
      llvm::errs() << "[tmem-ldst] packed16 support final atom="
                   << static_cast<int>(info->atom)
                   << " regsPerMsg=" << info->numRegsPerMessage
                   << " baseOffset=" << info->baseOffset
                   << " warpBase0=" << info->warpBaseOffset0
                   << " warpBase1=" << info->warpBaseOffset1;
      if (info->secondHalfOffset)
        llvm::errs() << " secondHalf=" << *info->secondHalfOffset;
      llvm::errs() << "\n[tmem-ldst] packed16 support final reps:\n"
                   << info->reps.toString() << "\n";
      if (!info->packetOffsets.empty()) {
        llvm::errs() << "[tmem-ldst] packed16 support packetOffsets:";
        for (int32_t offset : info->packetOffsets)
          llvm::errs() << " " << offset;
        llvm::errs() << "\n";
      }
    }
    return *info;
  };
  bool isCanonicalDirectMemLayout =
      memTy.getShape() == memTy.getAllocShape() && memLayout == originalMemLayout;
  // The packed-I16 support path exists to lower projected support/query layouts
  // that cannot be described cleanly as a single direct f16 TMEM family.
  if (!isCanonicalDirectMemLayout) {
    if (auto info = tryPackedI16SparseSupportInfo())
      return *info;
  }
  if (isScales) {
    auto outDims = to_vector(regLayout.getOutDims());
    auto rowBases = memLayout.getBases().lookup(kRow);
    std::optional<unsigned> rowOutIdx;
    for (ArrayRef<int32_t> basis : rowBases) {
      for (auto [idx, value] : llvm::enumerate(basis)) {
        if (value != 0) {
          rowOutIdx = idx;
          break;
        }
      }
      if (rowOutIdx)
        break;
    }
    if (rowOutIdx) {
      auto bases = regLayout.getBases();
      for (auto &dimBases : llvm::make_second_range(bases)) {
        for (auto &basis : dimBases) {
          bool touchesOnlyRow = basis[*rowOutIdx] != 0;
          for (auto [idx, value] : llvm::enumerate(basis)) {
            if (idx == *rowOutIdx)
              continue;
            touchesOnlyRow &= value == 0;
          }
          if (touchesOnlyRow &&
              std::abs(basis[*rowOutIdx]) >= logicalRows) {
            std::fill(basis.begin(), basis.end(), 0);
          }
        }
      }
      outDims[*rowOutIdx].second = static_cast<int32_t>(logicalRows);
      regLayout = squeezeTrivialBlock(LinearLayout(std::move(bases),
                                                   std::move(outDims),
                                                   /*requireSurjective=*/false));
    }
  }
  if (!canInvertAndComposeSafely(regLayout, directPlanningMemLayout)) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: register layout image "
                     "is not contained in the descriptor view image.\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  auto cvt = regLayout.invertAndCompose(directPlanningMemLayout);
  cvt = squeezeTrivialBlock(std::move(cvt));
  bool hasBlockIn = cvt.hasInDim(kBlock);
  bool hasBlockOut = cvt.hasOutDim(kBlock);
  if (hasBlockIn != hasBlockOut) {
    if (emitError) {
      emitError() << "The cga_layout of the register and memory layout must be "
                     "the same. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  if (hasBlockIn) {
    auto maybeSublayout = cvt.quotient({kBlock});
    if (!maybeSublayout) {
      if (emitError) {
        emitError() << "The cga_layout of the register and memory layout must "
                       "be the same. Got:\n"
                    << regLayout.toString() << "\n"
                    << memLayout.toString();
      }
      return failure();
    }
    cvt = maybeSublayout.value();
  }
  if (isScales) {
    auto prefersI16x32bx2 = [&]() {
      if (!regLayout.hasInDim(kReg) || regLayout.getNumOutDims() != 2)
        return false;
      int64_t logicalCols = memTy.getShape().back();
      if (logicalCols < 2 || !llvm::isPowerOf2_64(logicalCols))
        return false;
      SmallVector<int32_t> halfNBasis(regLayout.getNumOutDims(), 0);
      halfNBasis.back() = static_cast<int32_t>(logicalCols / 2);
      for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(kReg); ++idx) {
        if (regLayout.getBasis(kReg, idx) == ArrayRef<int32_t>(halfNBasis))
          return true;
      }
      return false;
    }();
    if (debugScales) {
      llvm::errs() << "[tmem-scales] regTy=" << regTy << "\n";
      llvm::errs() << "[tmem-scales] memTy=" << memTy << "\n";
      llvm::errs() << "[tmem-scales] regLayout:\n"
                   << regLayout.toString() << "\n";
      llvm::errs() << "[tmem-scales] memLayout:\n"
                   << memLayout.toString() << "\n";
      llvm::errs() << "[tmem-scales] cvt:\n" << cvt.toString() << "\n";
      llvm::errs() << "[tmem-scales] preferI16x32bx2="
                   << prefersI16x32bx2 << "\n";
    }
    if (regLayout.getInDimSizeLog2(kWarp) < 2 ||
        memLayout.getInDimSizeLog2(kRow) < 7) {
      if (emitError) {
        emitError() << "TMEM load/store requires row anchors at 32 and 64 for "
                       "the selected register and memory layouts. Got:\n"
                    << regLayout.toString() << "\n"
                    << memLayout.toString();
      }
      return failure();
    }
    if (!(regLayout.getBasis(kWarp, 0) == memLayout.getBasis(kRow, 5) &&
          regLayout.getBasis(kWarp, 1) == memLayout.getBasis(kRow, 6))) {
      if (emitError) {
        emitError() << "warps=1,2 must map to rows=32,64. Got:\n"
                    << regLayout.toString() << "\n"
                    << memLayout.toString();
      }
      return failure();
    }
    auto bases = cvt.getBases();
    bases[kWarp][0] = {32, 0};
    bases[kWarp][1] = {64, 0};
    cvt = LinearLayout(std::move(bases), cvt.getOutDims(),
                       /*isSurjective=*/cvt.isSurjective());

    auto info =
        lowerTMemLdSt(cvt, maxnreg, bitwidth, emitError,
                      /*unpacked=*/false, /*warpRow0=*/32, /*warpRow1=*/64,
                      /*rowSpan=*/128, prefersI16x32bx2);
    if (failed(info))
      return failure();
    if (info->atom == TMemAccessAtom::I16x32bx2 && logicalRows <= 16 &&
        logicalCols > 4 && info->numRegsPerMessage > 1 &&
        (!info->secondHalfOffset || *info->secondHalfOffset == 0)) {
      if (emitError) {
        emitError() << "Failed to lower TMEM scales load/store: the selected "
                       "register layout carries the 16x32bx2 half-tile split "
                       "through register repetition instead of the native "
                       "second-half offset. Use a register layout with the "
                       "half-tile split on lane=16.";
      }
      return failure();
    }
    if (debugScales) {
      llvm::errs() << "[tmem-scales] atom="
                   << static_cast<int>(info->atom)
                   << " numRegsPerMessage=" << info->numRegsPerMessage
                   << " vec=" << info->vec;
      if (info->secondHalfOffset)
        llvm::errs() << " secondHalfOffset=" << *info->secondHalfOffset;
      llvm::errs() << "\n[tmem-scales] reps:\n"
                   << info->reps.toString() << "\n";
    }

    auto kReg = *regLayout.getInDimNames().begin();
    if (bitwidth == 32) {
      auto expectedValueCount = getExpectedTMemLoadValueCount(*info, bitwidth);
      if (failed(expectedValueCount) ||
          *expectedValueCount != regLayout.getInDimSize(kReg)) {
        if (emitError) {
          emitError() << "Failed to lower TMEM load/store: unsupported "
                         "register broadcast pattern for direct lowering.\n"
                      << regLayout.toString() << "\n"
                      << memLayout.toString();
        }
        return failure();
      }
    }

    return info;
  }

  auto tryLegacyAnchoredExactFamily = [&]()
      -> std::optional<TMemLdStEncodingInfo> {
    bool hasCompatibleRowPlan =
        !rowPlanOverride ||
        (rowPlanOverride->warpRow0 == 32 && rowPlanOverride->warpRow1 == 64 &&
         rowPlanOverride->rowSpan == 128 && rowPlanOverride->baseOffset == 0);
    if (!hasCompatibleRowPlan || memTy.getShape() != memTy.getAllocShape() ||
        logicalRows < 64 || logicalCols < 1 ||
        !llvm::isPowerOf2_64(logicalCols) || !regLayout.hasInDim(kWarp) ||
        regLayout.getInDimSizeLog2(kWarp) < 2) {
      return std::nullopt;
    }

    auto legacyLayout =
        squeezeTrivialBlock(toLinearLayout(memTy.getShape(), memTy.getEncoding()));
    auto legacyFamilyLayout = legacyLayout;
    if (bitwidth < 32 && legacyFamilyLayout.hasInDim(kCol))
      legacyFamilyLayout = legacyFamilyLayout.removeZeroBasesAlongDim(kCol);
    if (!legacyLayout.hasInDim(kRow) || !legacyLayout.hasInDim(kCol))
      return std::nullopt;
    if (legacyLayout.hasInDim(kBlock) && legacyLayout.getInDimSize(kBlock) > 1)
      return std::nullopt;
    if (legacyLayout.getInDimSizeLog2(kRow) != 7)
      return std::nullopt;

    auto twoCTAs = getTensorMemoryTwoCTAs(memTy);
    if (!twoCTAs)
      return std::nullopt;
    std::optional<LinearLayout> matchedCanonicalLayout;
    std::optional<unsigned> matchedColStride;
    bool matchedExactCanonicalLayout = false;
    for (unsigned blockM : {64u, 128u}) {
      for (unsigned blockN : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u,
                              512u}) {
        for (unsigned colStride : {1u, 2u, 4u}) {
          auto maybeCanonical = getCanonicalTMemLinearEncoding(
              memTy.getShape(), blockM, blockN, colStride,
              gpu::getCGALayout(memTy.getEncoding()), *twoCTAs,
              /*error=*/nullptr);
          if (!maybeCanonical)
            continue;
          auto canonicalLayout =
              squeezeTrivialBlock(maybeCanonical->getLinearLayout());
          if (canonicalLayout == legacyLayout) {
            matchedCanonicalLayout = canonicalLayout;
            matchedColStride = colStride;
            matchedExactCanonicalLayout = true;
            break;
          }
          auto compareLayout = canonicalLayout;
          if (bitwidth < 32 && compareLayout.hasInDim(kCol))
            compareLayout = compareLayout.removeZeroBasesAlongDim(kCol);
          if (!matchedCanonicalLayout && compareLayout == legacyFamilyLayout) {
            matchedCanonicalLayout = canonicalLayout;
            matchedColStride = colStride;
          }
        }
        if (matchedExactCanonicalLayout)
          break;
      }
      if (matchedExactCanonicalLayout)
        break;
    }
    if (!matchedCanonicalLayout)
      return std::nullopt;
    if (!(regLayout.getBasis(kWarp, 0) ==
              matchedCanonicalLayout->getBasis(kRow, 5) &&
          regLayout.getBasis(kWarp, 1) ==
              matchedCanonicalLayout->getBasis(kRow, 6))) {
      return std::nullopt;
    }
    auto legacyCvt = regLayout.invertAndCompose(*matchedCanonicalLayout);
    legacyCvt = squeezeTrivialBlock(std::move(legacyCvt));
    bool legacyHasBlockIn = legacyCvt.hasInDim(kBlock);
    bool legacyHasBlockOut = legacyCvt.hasOutDim(kBlock);
    if (legacyHasBlockIn != legacyHasBlockOut)
      return std::nullopt;
    if (legacyHasBlockIn) {
      auto maybeSublayout = legacyCvt.quotient({kBlock});
      if (!maybeSublayout)
        return std::nullopt;
      legacyCvt = *maybeSublayout;
    }
    auto legacyCvtBases = legacyCvt.getBases();
    legacyCvtBases[kWarp][0] = {32, 0};
    legacyCvtBases[kWarp][1] = {64, 0};
    legacyCvt = LinearLayout(std::move(legacyCvtBases), legacyCvt.getOutDims(),
                             /*isSurjective=*/legacyCvt.isSurjective());
    auto info = lowerTMemLdSt(legacyCvt, maxnreg, bitwidth,
                              /*emitError=*/{}, /*unpacked=*/false,
                              /*warpRow0=*/32, /*warpRow1=*/64,
                              /*rowSpan=*/128,
                              /*preferI16x32bx2=*/false);
    if (failed(info)) {
      return std::nullopt;
    }
    if (bitwidth == 32) {
      auto expectedValueCount = getExpectedTMemLoadValueCount(*info, bitwidth);
      if (failed(expectedValueCount) ||
          *expectedValueCount != regLayout.getInDimSize(kReg)) {
        return std::nullopt;
      }
    }
    if (matchedColStride && bitwidth == 16 && *matchedColStride > 1 &&
        (info->atom == TMemAccessAtom::I32x32b ||
         info->atom == TMemAccessAtom::I16x256b)) {
      info->unpacked = true;
    }
    return *info;
  };
  if (auto info = tryLegacyAnchoredExactFamily())
    return *info;

  auto rowPlan = getTMemLdStRowPlanForType(memTy);
  auto memLayoutSupportsOverride = [&]() {
    if (!rowPlanOverride)
      return false;
    auto kRow = StringAttr::get(ctx, "row");
    return (memLayout.hasInDim(kRow) &&
            memLayout.getInDimSize(kRow) == rowPlanOverride->rowSpan) ||
           (directPlanningMemLayout.hasInDim(kRow) &&
            directPlanningMemLayout.getInDimSize(kRow) ==
                rowPlanOverride->rowSpan);
  };
  bool allowLiftedM64AccumulatorOverride =
      rowPlanOverride && rowPlanOverride->rowSpan == 128 &&
      memTy.getShape() == memTy.getAllocShape() && logicalRows == 64 &&
      ((rowPlanOverride->warpRow0 == 16 && rowPlanOverride->warpRow1 == 32) ||
       (rowPlanOverride->warpRow0 == 32 && rowPlanOverride->warpRow1 == 64)) &&
      getLogicalRowAnchorBasis(memLayout, rowPlanOverride->warpRow0) &&
      getLogicalRowAnchorBasis(memLayout, rowPlanOverride->warpRow1);
  if (rowPlanOverride &&
      (!rowPlan || rowPlanOverride->rowSpan == rowPlan->rowSpan ||
       memLayoutSupportsOverride() || allowLiftedM64AccumulatorOverride))
    rowPlan = rowPlanOverride;
  if (debug) {
    llvm::errs() << "[halfrows-info] regLayout:\n"
                 << regLayout.toString() << "\n";
    llvm::errs() << "[halfrows-info] memLayout:\n"
                 << memLayout.toString() << "\n";
    if (rowPlan) {
      llvm::errs() << "[halfrows-info] rowPlan warpRow0=" << rowPlan->warpRow0
                   << " warpRow1=" << rowPlan->warpRow1
                   << " rowSpan=" << rowPlan->rowSpan
                   << " baseOffset=" << rowPlan->baseOffset << "\n";
    } else {
      llvm::errs() << "[halfrows-info] rowPlan none\n";
    }
  }
  if (!regLayout.hasInDim(kWarp) || !memLayout.hasInDim(kRow) ||
      !memLayout.hasInDim(kCol) || !rowPlan) {
    if (emitError) {
      emitError() << "TMEM load/store requires register layout dim 'warp', "
                     "memory layout dims 'row'/'col', and a supported TMEM "
                     "row footprint (64 or 128 rows). Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  if (regLayout.getInDimSizeLog2(kWarp) < 2) {
    if (emitError) {
      emitError() << "TMEM load/store requires at least two warp-id anchor "
                     "bases. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  auto packTMemBasisOffset = [](ArrayRef<int32_t> basis) -> uint32_t {
    assert(basis.size() == 2 && "TMEM warp bases must be 2D row/col vectors");
    return (static_cast<uint32_t>(basis[0]) << 16) |
           static_cast<uint32_t>(basis[1]);
  };
  auto getRowAnchorBasis =
      [&](int32_t logicalRow) -> std::optional<SmallVector<int32_t>> {
    return getLogicalRowAnchorBasis(directPlanningMemLayout, logicalRow);
  };
  auto expectedWarp0Basis = getRowAnchorBasis(rowPlan->warpRow0);
  auto expectedWarp1Basis = getRowAnchorBasis(rowPlan->warpRow1);
  if (!expectedWarp0Basis || !expectedWarp1Basis) {
    if (emitError) {
      emitError() << "TMEM load/store requires row anchors "
                  << rowPlan->warpRow0 << "," << rowPlan->warpRow1
                  << " to be representable in the descriptor view. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  auto isZeroBasis = [](ArrayRef<int32_t> basis) {
    return llvm::all_of(basis, [](int32_t value) { return value == 0; });
  };
  bool isHalfRowLiftedSupportView =
      logicalRows > 0 && physicalRows == logicalRows * 2 &&
      logicalCols == physicalCols;
  SmallVector<int32_t> warpBasis0(cvt.getBasis(kWarp, 0).begin(),
                                  cvt.getBasis(kWarp, 0).end());
  SmallVector<int32_t> warpBasis1(cvt.getBasis(kWarp, 1).begin(),
                                  cvt.getBasis(kWarp, 1).end());
  auto prefersCanonicalM64SplitN = [&]() {
    if (bitwidth != 32 || isScales || logicalRows != 64 || logicalCols < 2 ||
        !llvm::isPowerOf2_64(logicalCols) || rowPlan->rowSpan != 64 ||
        !regLayout.hasInDim(kWarp))
      return false;
    unsigned numWarps = regLayout.getInDimSize(kWarp);
    auto maybeCanonical = getCanonicalM64SplitNLayout(ctx, logicalCols, numWarps);
    if (!maybeCanonical)
      return false;
    return regLayout == squeezeTrivialBlock(std::move(*maybeCanonical));
  }();

  auto info = lowerTMemLdSt(cvt, maxnreg, bitwidth, emitError,
                            /*unpacked=*/false, warpBasis0, warpBasis1,
                            rowPlan->rowSpan, prefersCanonicalM64SplitN);
  if (failed(info))
    return failure();
  // Sparse higher-rank TMEM views can carry logical selection bits as zero
  // column bases. Collapsing those views into a single wide 32x32b.xN message
  // over-updates the backing tile; retry without vectorization so the lowering
  // emits the required per-tile x1 packets instead.
  if (bitwidth == 32 && info->atom == TMemAccessAtom::I32x32b &&
      info->numRegsPerMessage > 1 &&
      (hasZeroBasisAlong(memLayout, kCol) || isHalfRowLiftedSupportView)) {
    if (auto scalarInfo = lowerTMemLdSt(cvt, /*maxnreg=*/2, bitwidth,
                                        /*emitError=*/{},
                                        /*unpacked=*/false, warpBasis0,
                                        warpBasis1, rowPlan->rowSpan);
        succeeded(scalarInfo) && scalarInfo->atom == info->atom) {
      info = *scalarInfo;
    }
  }
  if (bitwidth == 16 && info->atom == TMemAccessAtom::I16x32bx2 &&
      info->numRegsPerMessage > 1 && hasZeroBasisAlong(memLayout, kCol)) {
    if (auto scalarInfo = lowerTMemLdSt(cvt, /*maxnreg=*/2, bitwidth,
                                        /*emitError=*/{},
                                        /*unpacked=*/false, warpBasis0,
                                        warpBasis1, rowPlan->rowSpan);
        succeeded(scalarInfo) && scalarInfo->atom == info->atom) {
      info = *scalarInfo;
    } else {
      info->numRegsPerMessage = 1;
    }
  }
  auto getLegacyLikeColStride = [&]() -> std::optional<unsigned> {
    if (auto legacyEncoding = dyn_cast<TensorMemoryEncodingAttr>(memTy.getEncoding()))
      return legacyEncoding.getColStride();
    if (bitwidth != 16 || memTy.getShape() != memTy.getAllocShape())
      return std::nullopt;

    auto twoCTAs = getTensorMemoryTwoCTAs(memTy);
    if (!twoCTAs)
      return std::nullopt;

    auto legacyLayout =
        squeezeTrivialBlock(toLinearLayout(memTy.getShape(), memTy.getEncoding()));
    auto legacyFamilyLayout = legacyLayout;
    if (legacyFamilyLayout.hasInDim(kCol))
      legacyFamilyLayout = legacyFamilyLayout.removeZeroBasesAlongDim(kCol);

    std::optional<unsigned> fallbackColStride;
    for (unsigned blockM : {64u, 128u}) {
      for (unsigned blockN : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u,
                              512u}) {
        for (unsigned colStride : {1u, 2u, 4u}) {
          auto maybeCanonical = getCanonicalTMemLinearEncoding(
              memTy.getShape(), blockM, blockN, colStride,
              gpu::getCGALayout(memTy.getEncoding()), *twoCTAs,
              /*error=*/nullptr);
          if (!maybeCanonical)
            continue;
          auto canonicalLayout =
              squeezeTrivialBlock(maybeCanonical->getLinearLayout());
          if (canonicalLayout == legacyLayout)
            return colStride;
          auto compareLayout = canonicalLayout;
          if (compareLayout.hasInDim(kCol))
            compareLayout = compareLayout.removeZeroBasesAlongDim(kCol);
          if (!fallbackColStride && compareLayout == legacyFamilyLayout)
            fallbackColStride = colStride;
        }
      }
    }
    return fallbackColStride;
  };
  auto legacyLikeColStride = getLegacyLikeColStride();
  bool isLegacyUnpackedFullShape =
      legacyLikeColStride && *legacyLikeColStride > 1 && bitwidth == 16 &&
      memTy.getShape() == memTy.getAllocShape();
  if (isLegacyUnpackedFullShape)
    info->unpacked = true;
  if (debug) {
    llvm::errs() << "[halfrows-info] atom=" << static_cast<int>(info->atom)
                 << " secondHalfOffset=";
    if (info->secondHalfOffset)
      llvm::errs() << *info->secondHalfOffset;
    else
      llvm::errs() << "none";
    llvm::errs() << "\n[halfrows-info] reps:\n"
                 << info->reps.toString() << "\n";
  }
  // Lifted views that widen the logical TMEM columns by folding extra
  // physical TMEM rows into the column dimension cannot be vectorized across
  // multiple 32x32b messages without changing the logical-to-physical TMEM
  // projection. Keep those views on a narrower direct path instead of
  // collapsing them into a single xN message.
  if (info->atom == TMemAccessAtom::I32x32b && info->vec > 1 &&
      logicalRows < physicalRows && logicalCols > physicalCols) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: lifted TMEM views that "
                     "fold extra TMEM rows into logical columns are not "
                     "directly supportable through vectorized 32x32b "
                     "messages.";
    }
    return failure();
  }
  if (info->atom == TMemAccessAtom::I32x32b &&
      logicalRows < physicalRows && logicalCols > physicalCols &&
      logicalRows * logicalCols == physicalRows * physicalCols) {
    info->numRegsPerMessage = 1;
  }
  // The 16x32bx2 family cannot directly cover lifted views that encode an
  // extra logical row bit inside the physical TMEM columns. PTX probes on the
  // higher-rank reshape->subslice->index bucket show the upper-half row bit is
  // not addressable through this path, so keep it as a clean negative until a
  // different direct decomposition is implemented.
  if (info->atom == TMemAccessAtom::I16x32bx2 && logicalRows > physicalRows &&
      logicalCols < physicalCols) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: lifted TMEM views that "
                     "encode extra logical row bits in TMEM columns are not "
                     "directly supported by tcgen05.ld/st. Use a layout whose "
                     "TMEM rows/columns stay materializable.";
    }
    return failure();
  }
  bool isUnsupportedRowZeroLiftedReinterpretPlan =
      isOriginalRowZeroLiftedReinterpret &&
      !(info->atom == TMemAccessAtom::I32x32b && info->unpacked) &&
      !(isRowZeroM64ReinterpretView &&
        info->atom == TMemAccessAtom::I16x32bx2);
  if (isUnsupportedRowZeroLiftedReinterpretPlan) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: row-zero lifted TMEM reinterpret views "
                     "require the packed 32x32b.unpack::16b direct path. Use the descriptor's "
                     "own get_reg_layout() result rather than a parent TMEM register layout.";
    }
    return failure();
  }
  if (isHalfRowLiftedSupportView) {
    warpBasis0.assign(expectedWarp0Basis->begin(), expectedWarp0Basis->end());
    warpBasis1.assign(expectedWarp1Basis->begin(), expectedWarp1Basis->end());
  }
  bool isI16RowZeroM64ReinterpretView =
      isRowZeroM64ReinterpretView &&
      info->atom == TMemAccessAtom::I16x32bx2;
  if (isI16RowZeroM64ReinterpretView && info->secondHalfOffset &&
      regLayout.hasInDim(kWarp) && regLayout.getInDimSize(kWarp) <= 4)
    info->secondHalfOffset = *info->secondHalfOffset * 2;
  bool isI16RowZeroM64DirectView =
      bitwidth == 32 && logicalRows == 64 && logicalCols == 32 &&
      physicalRows == 128 && physicalCols == 32 &&
      hasZeroBasisAlong(originalMemLayout, kRow) &&
      !hasZeroBasisAlong(originalMemLayout, kCol) &&
      info->atom == TMemAccessAtom::I16x32bx2 && expectedWarp0Basis &&
      expectedWarp1Basis;
  if (isI16RowZeroM64DirectView) {
    if (debug)
      llvm::errs() << "[halfrows-info] apply row-zero M64 direct-view fix\n";
    // The lifted 64x32 raw-query quotient collapses one warp basis to zero.
    // Preserve that structure, but retarget the active warp basis to the
    // second logical row-anchor image and use the first anchor image as the
    // half-row packet offset.
    if (isZeroBasis(warpBasis0) && !isZeroBasis(warpBasis1)) {
      warpBasis1.assign(expectedWarp1Basis->begin(), expectedWarp1Basis->end());
      info->secondHalfOffset = packTMemBasisOffset(*expectedWarp0Basis);
    } else if (!isZeroBasis(warpBasis0) && isZeroBasis(warpBasis1)) {
      warpBasis0.assign(expectedWarp1Basis->begin(), expectedWarp1Basis->end());
      info->secondHalfOffset = packTMemBasisOffset(*expectedWarp0Basis);
    }
  }
  info->warpBaseOffset0 = packTMemBasisOffset(warpBasis0);
  info->warpBaseOffset1 = packTMemBasisOffset(warpBasis1);
  info->warpRow0 = warpBasis0.empty() ? 0 : warpBasis0.front();
  info->warpRow1 = warpBasis1.empty() ? 0 : warpBasis1.front();
  info->baseOffset = rowPlan->baseOffset;

  auto halvePackedTMemRowOffset = [](uint32_t packedOffset) {
    uint32_t row = packedOffset >> 16;
    uint32_t col = packedOffset & 0xffffu;
    return ((row / 2) << 16) | col;
  };
  bool isI32RowZeroM64DirectView =
      bitwidth == 32 && logicalRows == 64 && logicalCols == physicalCols &&
      physicalRows == 128 && hasZeroBasisAlong(originalMemLayout, kRow) &&
      !hasZeroBasisAlong(originalMemLayout, kCol);
  if (isI32RowZeroM64DirectView && info->atom == TMemAccessAtom::I32x32b &&
      info->warpBaseOffset0 == (32u << 16) &&
      info->warpBaseOffset1 == (64u << 16)) {
    info->warpBaseOffset0 = halvePackedTMemRowOffset(info->warpBaseOffset0);
    info->warpBaseOffset1 = halvePackedTMemRowOffset(info->warpBaseOffset1);
    info->warpRow0 /= 2;
    info->warpRow1 /= 2;
  }
  bool isI16PackedRowZeroM64DirectView =
      bitwidth == 16 && memTy.getShape() == memTy.getAllocShape() &&
      logicalRows == 64 && logicalCols == physicalCols &&
      physicalRows == 128 && hasZeroBasisAlong(originalMemLayout, kRow) &&
      hasZeroBasisAlong(originalMemLayout, kCol);
  if (isI16PackedRowZeroM64DirectView &&
      info->atom == TMemAccessAtom::I16x32bx2 &&
      info->warpBaseOffset0 == (32u << 16) &&
      info->warpBaseOffset1 == (64u << 16)) {
    info->warpBaseOffset0 = halvePackedTMemRowOffset(info->warpBaseOffset0);
    info->warpBaseOffset1 = halvePackedTMemRowOffset(info->warpBaseOffset1);
    info->warpRow0 /= 2;
    info->warpRow1 /= 2;
  }

  auto tmemWarpBasisAlignsToInstruction =
      [](TMemAccessAtom atom, ArrayRef<int32_t> basis) {
        if (atom != TMemAccessAtom::I16x32bx2)
          return true;
        return basis.size() == 2 && basis[0] % 16 == 0;
      };
  if (!tmemWarpBasisAlignsToInstruction(info->atom, warpBasis0) ||
      !tmemWarpBasisAlignsToInstruction(info->atom, warpBasis1)) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: selected "
                     "16x32bx2 warp anchors are not aligned to 16-row "
                     "message tiles.";
    }
    return failure();
  }

  auto regInputDim = *regLayout.getInDimNames().begin();
  if (bitwidth == 32) {
    auto expectedValueCount = getExpectedTMemLoadValueCount(*info, bitwidth);
    if (failed(expectedValueCount) ||
        *expectedValueCount != regLayout.getInDimSize(regInputDim)) {
      if (emitError) {
        emitError() << "Failed to lower TMEM load/store: unsupported register "
                       "broadcast pattern for direct lowering.\n"
                    << regLayout.toString() << "\n"
                    << memLayout.toString();
      }
      return failure();
    }
  }

  return info;
}

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError,
                            std::optional<TMemLdStRowPlan> rowPlanOverride) {
  auto *ctx = regTy.getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto squeezeTrivialBlock = [&](LinearLayout layout) {
    if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1)
      layout = layout.squeezeIns(kBlock);
    if (layout.hasOutDim(kBlock) && layout.getOutDimSize(kBlock) == 1)
      layout = layout.squeezeOuts(kBlock);
    return layout;
  };
  LinearLayout memLayout = [&]() -> LinearLayout {
    if (isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
      return squeezeTrivialBlock(toLinearLayout(memTy));
    // Full-shape TMEM descriptors already carry the exact physical contract in
    // their encoding. Direct ld/st planning must use that exact image instead
    // of the normalized view-analysis layout; otherwise row/col zero bases from
    // split-N and other exact tensor_memory_linear roots disappear and root
    // stores no longer agree with later slice/view consumers of the same TMEM
    // value.
    if (auto familyQuery = getFullShapeMMAv5FamilyQueryLayout(memTy))
      return squeezeTrivialBlock(familyQuery->layout);
    if (isTensorMemoryEncoding(memTy.getEncoding()) &&
        !isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()) &&
        memTy.getShape() == memTy.getAllocShape()) {
      return squeezeTrivialBlock(
          toLinearLayout(memTy.getShape(), memTy.getEncoding()));
    }

    std::string analysisError;
    auto maybeAnalysisLayout = getTMemViewAnalysisLinearLayout(
        memTy.getShape(), memTy.getEncoding(), &analysisError);
    if (maybeAnalysisLayout) {
      return squeezeTrivialBlock(
          normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysisLayout));
    }

    std::string canonicalError;
    auto maybeCanonical = getCanonicalTMemLinearEncoding(memTy, &canonicalError);
    if (maybeCanonical) {
      return squeezeTrivialBlock(normalizeTensorMemoryLinearLayoutForAnalysis(
          maybeCanonical->getLinearLayout()));
    }

    if (emitError) {
      emitError() << (analysisError.empty()
                          ? (canonicalError.empty()
                                 ? "TMEM descriptor view is not representable "
                                   "for direct TMEM load/store lowering"
                                 : canonicalError)
                          : analysisError);
    }
    return LinearLayout();
  }();
  return computeTMemLdStEncodingInfoImpl(regTy, memTy, std::move(memLayout),
                                         maxnreg, emitError, rowPlanOverride);
}

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, MemDescType memTy,
                            const LinearLayout &queryLayout, int maxnreg,
                            std::function<InFlightDiagnostic()> emitError,
                            std::optional<TMemLdStRowPlan> rowPlanOverride) {
  return computeTMemLdStEncodingInfoImpl(regTy, memTy, queryLayout, maxnreg,
                                         emitError, rowPlanOverride);
}

static uint32_t getTMemLdStQueryOriginBaseOffset(const TMemLdStQueryLayout &query,
                                                 int bitwidth) {
  auto *ctx = query.layout.getInDimNames().begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  uint32_t offset = 0;
  auto inDims = llvm::to_vector(query.layout.getInDimNames());
  auto accumulate = [&](StringAttr dim, unsigned shift) {
    auto it = llvm::find(inDims, dim);
    if (it == inDims.end())
      return;
    int32_t value = query.origin[std::distance(inDims.begin(), it)];
    if (value <= 0)
      return;
    if (shift == 16) {
      offset += static_cast<uint32_t>(value) << shift;
    } else {
      offset += static_cast<uint32_t>(value) * bitwidth / 32;
    }
  };
  accumulate(kRow, 16);
  accumulate(kCol, 0);
  return offset;
}

static bool needsScalarized32x32QueryInfo(MemDescType memTy,
                                          const TMemLdStQueryLayout &query) {
  if (memTy.getRank() != 2 || memTy.getShape()[0] != 32 ||
      memTy.getShape()[1] != 32)
    return false;
  if (!llvm::all_of(query.origin, [](int32_t value) { return value == 0; }))
    return true;
  auto normalizedQuery = normalizeTensorMemoryLinearLayoutForAnalysis(query.layout);
  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  int bitwidth = memTy.getElementTypeBitWidth();
  if (!normalizedQuery.hasInDim(kRow) || !normalizedQuery.hasInDim(kCol))
    return false;
  int64_t physicalRows = normalizedQuery.getInDimSize(kRow);
  int64_t physicalCols =
      normalizedQuery.getInDimSize(kCol) / (32 / bitwidth);
  return memTy.getShape()[0] < physicalRows &&
         memTy.getShape()[1] > physicalCols &&
         memTy.getShape()[0] * memTy.getShape()[1] ==
             physicalRows * physicalCols;
}

static bool needsScalarizedLiftedRowQueryInfo(
    MemDescType memTy, const TMemLdStQueryLayout &query) {
  if (memTy.getRank() != 2 || memTy.getElementTypeBitWidth() != 32)
    return false;
  auto normalizedQuery =
      normalizeTensorMemoryLinearLayoutForAnalysis(query.layout);
  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!normalizedQuery.hasInDim(kRow) || !normalizedQuery.hasInDim(kCol))
    return false;
  int64_t physicalRows = normalizedQuery.getInDimSize(kRow);
  int64_t physicalCols = normalizedQuery.getInDimSize(kCol);
  if (physicalRows != memTy.getShape()[0] * 2 ||
      physicalCols != memTy.getShape()[1])
    return false;
  auto inDims = llvm::to_vector(query.layout.getInDimNames());
  auto getOrigin = [&](StringAttr dim) -> int32_t {
    auto it = llvm::find(inDims, dim);
    if (it == inDims.end())
      return 0;
    return query.origin[std::distance(inDims.begin(), it)];
  };
  return getOrigin(kRow) == memTy.getShape()[0] && getOrigin(kCol) == 0;
}

static void adjustTMemLdStInfoForQueryLayout(
    TMemLdStEncodingInfo &info, RankedTensorType regTy, MemDescType memTy,
    const TMemLdStQueryLayout &query, int maxnreg,
    std::optional<TMemLdStRowPlan> rowPlanOverride) {
  // Query-origin translated 32x32 TMEM views can require per-register
  // immediate decomposition even when the standalone tile shape itself is
  // canonical. A single vectorized 32x32b.xN message over-covers these lifted
  // tiles after the TMEM base has been advanced to the subview origin.
  if (info.atom != TMemAccessAtom::I32x32b ||
      info.numRegsPerMessage <= 1)
    return;
  if (!needsScalarized32x32QueryInfo(memTy, query) &&
      !needsScalarizedLiftedRowQueryInfo(memTy, query))
    return;

  auto scalarInfo = computeTMemLdStEncodingInfoImpl(
      regTy, memTy, query.layout,
      /*maxnreg=*/std::min(maxnreg, 2), /*emitError=*/{}, rowPlanOverride);
  if (failed(scalarInfo) || scalarInfo->atom != info.atom ||
      scalarInfo->numRegsPerMessage > 1) {
    info.numRegsPerMessage = 1;
    return;
  }

  scalarInfo->baseOffset += info.baseOffset;
  scalarInfo->warpBaseOffset0 = info.warpBaseOffset0;
  scalarInfo->warpBaseOffset1 = info.warpBaseOffset1;
  scalarInfo->warpRow0 = info.warpRow0;
  scalarInfo->warpRow1 = info.warpRow1;
  info = *scalarInfo;
}

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, MemDescType memTy,
                            const TMemLdStQueryLayout &queryLayout,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError,
                            std::optional<TMemLdStRowPlan> rowPlanOverride) {
  auto info = computeTMemLdStEncodingInfoImpl(regTy, memTy, queryLayout.layout,
                                              maxnreg, emitError,
                                              rowPlanOverride);
  if (failed(info))
    return failure();
  auto *ctx = regTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto inDims = llvm::to_vector(queryLayout.layout.getInDimNames());
  auto getOrigin = [&](StringAttr dim) -> int32_t {
    auto it = llvm::find(inDims, dim);
    if (it == inDims.end())
      return 0;
    return queryLayout.origin[std::distance(inDims.begin(), it)];
  };
  info->baseOffset += getTMemLdStQueryOriginBaseOffset(
      queryLayout, memTy.getElementTypeBitWidth());
  // Projected row-half support queries on 32-bit 64x64 tiles can pick the
  // 16x32bx2 family. That direct path should inherit only the lifted row
  // origin; any packed low-bit base offset would skew the selected 64-row
  // window before the per-message column immediates are applied.
  if (info->atom == TMemAccessAtom::I16x32bx2 && memTy.getRank() == 2 &&
      memTy.getElementTypeBitWidth() == 32 && memTy.getShape()[0] == 64 &&
      getOrigin(kRow) > 0 && getOrigin(kCol) == 0) {
    info->baseOffset &= 0xffff0000u;
  }
  adjustTMemLdStInfoForQueryLayout(*info, regTy, memTy, queryLayout, maxnreg,
                                   rowPlanOverride);
  return info;
}

std::optional<TMemLdStPhysicalSupportPlan>
getTMemLdStPhysicalSupportPlan(MemDescType memTy, unsigned numWarps,
                               int maxnreg) {
  if (memTy.getRank() != 2)
    return std::nullopt;
  if (!isTensorMemoryEncoding(memTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
    return std::nullopt;

  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  std::string error;
  auto maybeMemLayout =
      getTMemViewAnalysisLinearLayout(memTy.getShape(), memTy.getEncoding(),
                                      &error);
  if (!maybeMemLayout || !maybeMemLayout->hasInDim(kRow) ||
      !maybeMemLayout->hasInDim(kCol))
    return std::nullopt;

  auto supportsDirectAtom = [&](TMemAccessAtom atom) {
    auto maybeLayout = getDistributedLayoutForTmemLdSt(memTy, atom, numWarps);
    if (!maybeLayout)
      return false;
    auto regTy = RankedTensorType::get(
        memTy.getShape(), memTy.getElementType(),
        LinearEncodingAttr::get(ctx, std::move(*maybeLayout)));
    auto info = computeTMemLdStEncodingInfo(regTy, memTy, maxnreg);
    return succeeded(info) && info->atom == atom;
  };
  if (supportsDirectAtom(TMemAccessAtom::I16x256b))
    return std::nullopt;

  int bitwidth = memTy.getElementTypeBitWidth();
  int64_t logicalRows = memTy.getShape()[0];
  int64_t logicalCols = memTy.getShape()[1];
  int64_t physicalRows = maybeMemLayout->getInDimSize(kRow);
  int64_t physicalCols = maybeMemLayout->getInDimSize(kCol) / (32 / bitwidth);
  if (logicalRows <= physicalRows || logicalCols >= physicalCols ||
      logicalRows * logicalCols != physicalRows * physicalCols)
    return std::nullopt;

  SmallVector<int64_t> supportShape = {physicalRows, physicalCols};
  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(memTy.getEncoding());
  if (!maybeTwoCTAs)
    return std::nullopt;
  auto supportLayout = reshapeLayout(ctx, *maybeMemLayout, supportShape);
  auto maybeSupportEncoding =
      tryMakeTMemViewEncoding(ctx, std::move(supportLayout), *maybeTwoCTAs,
                              &error);
  if (!maybeSupportEncoding)
    return std::nullopt;
  auto supportMemTy =
      MemDescType::get(supportShape, memTy.getElementType(),
                       *maybeSupportEncoding, memTy.getMemorySpace(),
                       memTy.getMutableMemory(), supportShape);

  bool prefer16x256 =
      triton::tools::getBoolEnv("TRITON_PREFER_TMEM_16x256_LAYOUT");
  SmallVector<TMemAccessAtom> atoms =
      prefer16x256
          ? SmallVector<TMemAccessAtom>{TMemAccessAtom::I16x256b,
                                        TMemAccessAtom::I32x32b,
                                        TMemAccessAtom::I16x128b,
                                        TMemAccessAtom::I16x64b}
          : SmallVector<TMemAccessAtom>{TMemAccessAtom::I32x32b,
                                        TMemAccessAtom::I16x256b,
                                        TMemAccessAtom::I16x128b,
                                        TMemAccessAtom::I16x64b};
  for (auto atom : atoms) {
    auto maybeLayout =
        getDistributedLayoutForTmemLdSt(supportMemTy, atom, numWarps);
    if (!maybeLayout)
      continue;
    auto supportRegTy = RankedTensorType::get(
        supportShape, memTy.getElementType(),
        LinearEncodingAttr::get(ctx, std::move(*maybeLayout)));
    auto supportInfo =
        computeTMemLdStEncodingInfo(supportRegTy, supportMemTy, maxnreg);
    if (succeeded(supportInfo) && supportInfo->atom == atom) {
      return TMemLdStPhysicalSupportPlan{supportMemTy, supportRegTy, atom};
    }
  }
  return std::nullopt;
}

std::optional<TMemCopyAtom> getTMemCopyAtom(const LinearLayout &cvt,
                                            int bitwidth) {
  auto inDims = cvt.getInDimNames();
  if (inDims.empty())
    return std::nullopt;
  auto *ctx = inDims.begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kRow = S("row");
  auto kCol = S("col");
  auto kOffset = S("offset");
  if (!cvt.hasInDim(kRow) || !cvt.hasInDim(kCol) || !cvt.hasOutDim(kOffset))
    return std::nullopt;
  int totalBits = cvt.getInDimSize(kCol) * bitwidth;
  if (cvt.getInDimSize(kRow) == 4) {
    if (totalBits >= 256)
      return TMemCopyAtom{4, 256, 0};
    return std::nullopt;
  }
  if (cvt.getInDimSize(kRow) != 128)
    return std::nullopt;

  auto multicastBit = [&](int i) {
    assert(i == 0 || i == 1);
    return cvt.getBasis(kRow, llvm::Log2_32(32) + i, kOffset) == 0;
  };
  auto multicast = multicastBit(0) | multicastBit(1) << 1;
  if (multicast == 0) {
    if (totalBits == 128)
      return TMemCopyAtom{128, 128, 0};
    if (totalBits >= 256)
      return TMemCopyAtom{128, 256, 0};
    return std::nullopt;
  }
  if (multicast == 1)
    return TMemCopyAtom{64, 128, 1};
  if (multicast == 2)
    return TMemCopyAtom{64, 128, 2};
  if (multicast == 3)
    return TMemCopyAtom{32, 128, 3};
  return std::nullopt;
}

TMemCopyFamily getTMemCopyFamily(const TMemCopyAtom &atom) {
  if (atom.multicast == 1)
    return TMemCopyFamily::Warpx2_01_23_64x128b;
  if (atom.multicast == 2)
    return TMemCopyFamily::Warpx2_02_13_64x128b;
  if (atom.multicast == 3)
    return TMemCopyFamily::Warpx4_32x128b;
  if (atom.nRow == 4 && atom.bCol == 256)
    return TMemCopyFamily::Dense4x256b;
  if (atom.bCol == 128)
    return TMemCopyFamily::Dense128x128b;
  return TMemCopyFamily::Dense128x256b;
}

StringRef stringifyTMemCopyFamily(TMemCopyFamily family) {
  switch (family) {
  case TMemCopyFamily::Dense4x256b:
    return "4x256b";
  case TMemCopyFamily::Dense128x128b:
    return "128x128b";
  case TMemCopyFamily::Dense128x256b:
    return "128x256b";
  case TMemCopyFamily::Warpx2_01_23_64x128b:
    return "warpx2::01_23.64x128b";
  case TMemCopyFamily::Warpx2_02_13_64x128b:
    return "warpx2::02_13.64x128b";
  case TMemCopyFamily::Warpx4_32x128b:
    return "warpx4.32x128b";
  }
  llvm_unreachable("unknown tcgen05.copy family");
}

StringRef
stringifyTMemCopySupportFailureLayer(TMemCopySupportFailureLayer layer) {
  switch (layer) {
  case TMemCopySupportFailureLayer::None:
    return "none";
  case TMemCopySupportFailureLayer::PhysicalQuery:
    return "physical query";
  case TMemCopySupportFailureLayer::IsaAtom:
    return "ISA atom";
  case TMemCopySupportFailureLayer::DescriptorSynthesis:
    return "descriptor synthesis";
  case TMemCopySupportFailureLayer::CtaOwnership:
    return "CTA ownership";
  case TMemCopySupportFailureLayer::SharedLayout:
    return "shared layout";
  case TMemCopySupportFailureLayer::ResourceBoundary:
    return "resource boundary";
  }
  llvm_unreachable("unknown tcgen05.copy support failure layer");
}

static TMemCopySupportResult getSupportedTMemCopyResult() {
  return {true, TMemCopySupportFailureLayer::None, ""};
}

static TMemCopySupportResult
getUnsupportedTMemCopyResult(TMemCopySupportFailureLayer layer, Twine message) {
  return {false, layer, message.str()};
}

TMemCopySupportResult
getTMemCopySharedLayoutRuntimeSupport(MemDescType srcTy,
                                      TMemCopyFamily family) {
  if (family != TMemCopyFamily::Warpx2_01_23_64x128b &&
      family != TMemCopyFamily::Warpx2_02_13_64x128b)
    return getSupportedTMemCopyResult();

  auto setError = [&](Twine msg) {
    return getUnsupportedTMemCopyResult(TMemCopySupportFailureLayer::SharedLayout,
                                        msg);
  };

  if (srcTy.getRank() != 2 || srcTy.getShape()[1] != 4 ||
      (srcTy.getShape()[0] != 128 && srcTy.getShape()[0] != 256)) {
    return setError("warpx2 tcgen05.copy currently requires a 128x4 shared "
                    "tile, or a 256x4 two-CTA shared tile with the canonical "
                    "CTA block basis.");
  }
  if (srcTy.getElementType().getIntOrFloatBitWidth() != 32)
    return setError("warpx2 tcgen05.copy currently requires 32-bit shared "
                    "elements.");
  if (!isa<triton::gpu::SharedLinearEncodingAttr>(srcTy.getEncoding())) {
    return setError("warpx2 tcgen05.copy currently requires the canonical "
                    "shared-linear source layout.");
  }

  auto shmemLl = toLinearLayout(srcTy);
  auto *ctx = srcTy.getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!shmemLl.hasInDim(kOffset))
    return setError("warpx2 tcgen05.copy shared layout has no offset dimension.");
  for (auto dim : shmemLl.getInDimNames()) {
    if (dim != kOffset && dim != kBlock) {
      return setError("warpx2 tcgen05.copy shared layout may only use offset "
                      "and block dimensions.");
    }
  }

  constexpr int32_t expectedOffsetBases[][2] = {
      {32, 0}, {0, 1}, {0, 2}, {1, 0}, {2, 0},
      {4, 0},  {8, 0}, {16, 0}, {64, 0},
  };
  auto actualOffsetBases = shmemLl.getBases().lookup(kOffset);
  if (actualOffsetBases.size() != std::size(expectedOffsetBases)) {
    return setError("warpx2 tcgen05.copy currently supports only the "
                    "canonical 128x4 shared-linear offset basis order.");
  }
  for (auto [actual, expected] :
       llvm::zip(actualOffsetBases, llvm::ArrayRef(expectedOffsetBases))) {
    if (!llvm::equal(actual, llvm::ArrayRef(expected))) {
      return setError("warpx2 tcgen05.copy currently supports only the "
                      "canonical 128x4 shared-linear offset basis order.");
    }
  }

  auto blockBases = shmemLl.getBases().lookup(kBlock);
  if (srcTy.getShape()[0] == 128) {
    if (!llvm::all_of(blockBases, [](ArrayRef<int32_t> basis) {
          return llvm::all_of(basis, [](int32_t v) { return v == 0; });
        })) {
      return setError("single-CTA warpx2 tcgen05.copy does not support a "
                      "non-zero shared block basis.");
    }
    return getSupportedTMemCopyResult();
  }

  if (blockBases.size() != 1 ||
      !llvm::equal(blockBases.front(), ArrayRef<int32_t>{128, 0})) {
    return setError("two-CTA warpx2 tcgen05.copy requires the canonical "
                    "shared block basis [[128, 0]].");
  }
  return getSupportedTMemCopyResult();
}

bool isTMemCopySharedLayoutRuntimeSupported(MemDescType srcTy,
                                            TMemCopyFamily family,
                                            std::string *error) {
  auto result = getTMemCopySharedLayoutRuntimeSupport(srcTy, family);
  if (!result && error)
    *error = result.message;
  return result.supported;
}

static TMemCopySupportResult
getDirectTMemCopyLayoutSupportForLayout(const LinearLayout &layout,
                                        MLIRContext *ctx,
                                        TMemCopyFamily family) {
  if (family != TMemCopyFamily::Dense4x256b &&
      family != TMemCopyFamily::Dense128x128b &&
      family != TMemCopyFamily::Dense128x256b)
    return getSupportedTMemCopyResult();

  auto ll = normalizeTensorMemoryLinearLayoutForAnalysis(layout);
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) || ll.getNumOutDims() != 2) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery,
        "direct tcgen05.copy currently requires a rank-2 TMEM view "
        "with explicit row/col bases.");
  }

  auto isStrictlyIncreasing = [](ArrayRef<int32_t> values) {
    if (values.size() < 2)
      return true;
    for (auto [lhs, rhs] : llvm::zip(values, values.drop_front())) {
      if (rhs <= lhs)
        return false;
    }
    return true;
  };

  SmallVector<int32_t> rowBasisValues;
  for (ArrayRef<int32_t> basis : ll.getBases().lookup(kRow)) {
    if (basis[0] == 0 || basis[1] != 0) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          "direct tcgen05.copy does not support TMEM row bases that mix row "
          "and column contributions.");
    }
    rowBasisValues.push_back(std::abs(basis[0]));
  }
  if (!isStrictlyIncreasing(rowBasisValues)) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery,
        "direct tcgen05.copy requires TMEM row bases to stay in "
        "ascending physical row order.");
  }

  SmallVector<int32_t> pureColBases;
  SmallVector<int32_t> pureRowBases;
  for (ArrayRef<int32_t> basis : ll.getBases().lookup(kCol)) {
    bool touchesRow = basis[0] != 0;
    bool touchesCol = basis[1] != 0;
    if (touchesRow && touchesCol) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          "direct tcgen05.copy does not support TMEM column bases that "
          "mix row and column contributions.");
    }
    if (touchesCol) {
      pureColBases.push_back(std::abs(basis[1]));
    } else if (touchesRow) {
      pureRowBases.push_back(std::abs(basis[0]));
    }
  }
  if (!isStrictlyIncreasing(pureColBases)) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery,
        "direct tcgen05.copy requires TMEM column bases to remain in "
        "ascending physical column order.");
  }
  if (!isStrictlyIncreasing(pureRowBases)) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery,
        "direct tcgen05.copy requires TMEM row-repetition bases stored "
        "in the column address space to remain in ascending row order.");
  }
  return getSupportedTMemCopyResult();
}

TMemCopySupportResult getDirectTMemCopyLayoutSupport(MemDescType memTy,
                                                     TMemCopyFamily family) {
  if (family != TMemCopyFamily::Dense4x256b &&
      family != TMemCopyFamily::Dense128x128b &&
      family != TMemCopyFamily::Dense128x256b)
    return getSupportedTMemCopyResult();

  std::string layoutError;
  auto maybeLayout = getTMemViewAnalysisLinearLayout(memTy.getShape(),
                                                     memTy.getEncoding(),
                                                     &layoutError);
  if (!maybeLayout) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery, layoutError);
  }
  return getDirectTMemCopyLayoutSupportForLayout(*maybeLayout,
                                                memTy.getContext(), family);
}

TMemCopySupportResult
getDirectTMemCopyLayoutSupport(const TMemPhysicalQuery &query,
                               TMemCopyFamily family) {
  return getDirectTMemCopyLayoutSupportForLayout(
      query.layout, query.memTy.getContext(), family);
}

bool isDirectTMemCopyLayoutSupported(MemDescType memTy, TMemCopyFamily family,
                                     std::string *error) {
  auto result = getDirectTMemCopyLayoutSupport(memTy, family);
  if (!result && error)
    *error = result.message;
  return result.supported;
}

bool isDirectTMemCopyLayoutSupported(const TMemPhysicalQuery &query,
                                     TMemCopyFamily family,
                                     std::string *error) {
  auto result = getDirectTMemCopyLayoutSupport(query, family);
  if (!result && error)
    *error = result.message;
  return result.supported;
}

TMemCopySupportResult
getTMemCopyPlanSupport(MemDescType srcTy, const TMemPhysicalQuery &dstQuery,
                       const LinearLayout &shmemLl, const LinearLayout &cvt,
                       const TMemCopyPlan &plan, int bitwidth) {
  auto layoutSupport = getDirectTMemCopyLayoutSupport(dstQuery, plan.family);
  if (!layoutSupport)
    return layoutSupport;

  auto sharedLayoutSupport =
      getTMemCopySharedLayoutRuntimeSupport(srcTy, plan.family);
  if (!sharedLayoutSupport)
    return sharedLayoutSupport;

  if (!canSynthesizeTMemCopySharedDescriptorPlan(srcTy, shmemLl, cvt, plan,
                                                 bitwidth)) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::DescriptorSynthesis, "");
  }
  return getSupportedTMemCopyResult();
}

std::optional<uint64_t>
getDirectTMemCopySeedDescriptorImm(MemDescType srcTy, TMemCopyFamily family) {
  if (family != TMemCopyFamily::Warpx2_02_13_64x128b)
    return std::nullopt;
  if (srcTy.getRank() != 2 || srcTy.getShape()[0] != 128 ||
      srcTy.getShape()[1] != 4)
    return std::nullopt;
  if (srcTy.getElementType().getIntOrFloatBitWidth() != 32)
    return std::nullopt;
  if (!isa<triton::gpu::SharedLinearEncodingAttr>(srcTy.getEncoding()))
    return std::nullopt;

  auto shmemLl = toLinearLayout(srcTy);
  auto *ctx = srcTy.getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!shmemLl.hasInDim(kOffset))
    return std::nullopt;
  for (auto dim : shmemLl.getInDimNames()) {
    if (dim != kOffset && dim != kBlock)
      return std::nullopt;
  }
  if (shmemLl.hasInDim(kBlock) &&
      !llvm::all_of(shmemLl.getBases().lookup(kBlock), [](ArrayRef<int32_t> b) {
        return llvm::all_of(b, [](int32_t v) { return v == 0; });
      })) {
    return std::nullopt;
  }

  constexpr int32_t expectedOffsetBases[][2] = {
      {32, 0}, {0, 1}, {0, 2}, {1, 0}, {2, 0},
      {4, 0},  {8, 0}, {16, 0}, {64, 0},
  };
  auto actualOffsetBases = shmemLl.getBases().lookup(kOffset);
  if (actualOffsetBases.size() != std::size(expectedOffsetBases))
    return std::nullopt;
  for (auto [actual, expected] :
       llvm::zip(actualOffsetBases, llvm::ArrayRef(expectedOffsetBases))) {
    if (!llvm::equal(actual, llvm::ArrayRef(expected)))
      return std::nullopt;
  }

  uint64_t seedImm = 0;
  seedImm |= 1ULL << 46;
  seedImm |= 8ULL << 32;
  return seedImm;
}

llvm::SmallVector<TMemCopyPlan> getTMemCopyPlans(const LinearLayout &cvt,
                                                 int bitwidth) {
  auto atom = getTMemCopyAtom(cvt, bitwidth);
  if (!atom)
    return {};

  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto kBlock = StringAttr::get(ctx, "block");

  auto makePlan = [&](ArrayRef<std::tuple<unsigned, unsigned, unsigned, int>>
                          messageSpecs) {
    TMemCopyPlan plan;
    plan.family = getTMemCopyFamily(*atom);
    for (auto [descriptorRows, sourceWarpGroups, instrRows, smemRow] :
         messageSpecs) {
      TMemCopyMessagePlan message;
      message.atom = *atom;
      message.descriptorRows = descriptorRows;
      message.sourceWarpGroups = sourceWarpGroups;
      message.descriptorShape = {descriptorRows,
                                 static_cast<unsigned>(atom->bCol / bitwidth)};
      message.instrShape = {instrRows,
                            static_cast<unsigned>(atom->bCol / bitwidth)};
      message.smemRow = smemRow;
      plan.messages.push_back(std::move(message));
    }
    return plan;
  };

  llvm::SmallVector<TMemCopyPlan> plans;
  auto appendWarpx2Plan = [&](unsigned descriptorRows,
                              unsigned sourceWarpGroups, bool directSeed,
                              int tmemDwordDelta,
                              int directSourceOffsetB128) {
    auto plan = makePlan({std::tuple{descriptorRows, sourceWarpGroups, 64u, 0}});
    auto &message = plan.messages.front();
    message.useDirectSeedDescriptor = directSeed;
    message.tmemDwordDelta = tmemDwordDelta;
    message.directSourceOffsetB128 = directSourceOffsetB128;
    plans.push_back(std::move(plan));
  };

  // CUTLASS models the two warpx2 families differently:
  //   - 01_23 is a 32x128b core-descriptor family with one broadcast axis and
  //     one repeat axis.
  //   - 02_13 is a true 64x128b core-descriptor family with one broadcast
  //     axis.
  if (atom->multicast == 1) {
    appendWarpx2Plan(/*descriptorRows=*/32u, /*sourceWarpGroups=*/4u,
                     /*directSeed=*/false, /*tmemDwordDelta=*/0,
                     /*directSourceOffsetB128=*/0);
    return plans;
  }
  if (atom->multicast == 2) {
    bool isSingleCTA = !cvt.hasInDim(kBlock) || cvt.getInDimSize(kBlock) == 1;
    if (isSingleCTA) {
      appendWarpx2Plan(/*descriptorRows=*/64u, /*sourceWarpGroups=*/2u,
                       /*directSeed=*/true, /*tmemDwordDelta=*/4,
                       /*directSourceOffsetB128=*/32);
    }
    appendWarpx2Plan(/*descriptorRows=*/64u, /*sourceWarpGroups=*/2u,
                     /*directSeed=*/false, /*tmemDwordDelta=*/0,
                     /*directSourceOffsetB128=*/0);
    // Keep a bounded 32x4 fallback in the descriptor search, but do not treat
    // descriptor representability alone as proof of correctness. Two-CTA
    // 02_13 probes showed the 01_23-style descriptor shape still duplicates the
    // source-column pair unless a real descriptor/address schedule preserves the
    // missing 4-byte source-column bit.
    appendWarpx2Plan(/*descriptorRows=*/32u, /*sourceWarpGroups=*/4u,
                     /*directSeed=*/false, /*tmemDwordDelta=*/0,
                     /*directSourceOffsetB128=*/0);
    return plans;
  }

  if (atom->multicast == 0 && atom->nRow == 4) {
    plans.push_back(makePlan({std::tuple{4u, 1u, 4u, 0}}));
    return plans;
  }

  plans.push_back(makePlan({std::tuple{32u, 4u, 32u, 0}}));
  // Dense families sometimes admit a 64x2 descriptor factorization in addition
  // to the canonical 32x4 split. Keep the canonical plan first and only fall
  // back to the 64x2 variant when descriptor synthesis rejects the canonical
  // one.
  if (atom->multicast == 0 && atom->nRow == 128)
    plans.push_back(makePlan({std::tuple{64u, 2u, 64u, 0}}));
  return plans;
}

llvm::SmallVector<LinearLayout>
getTMemCopyDescriptorLayouts(MemDescType srcTy,
                             const LinearLayout &shmemLl,
                             const LinearLayout &cvt,
                             const TMemCopyMessagePlan &message) {
  auto inDims = cvt.getInDimNames();
  assert(!inDims.empty());
  auto *ctx = inDims.begin()->getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto kCol = StringAttr::get(ctx, "col");
  auto kRow = StringAttr::get(ctx, "row");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto makeLayout = [&](unsigned descriptorRows, unsigned sourceWarpGroups,
                        unsigned descriptorCols) {
    return cvt
        .reshapeIns({{kRow, static_cast<int32_t>(descriptorRows)},
                     {kWarp, static_cast<int32_t>(sourceWarpGroups)},
                     {kCol, static_cast<int32_t>(descriptorCols)},
                     {kBlock, cvt.getInDimSize(kBlock)}})
        .sublayout({kRow, kCol}, to_vector(cvt.getOutDimNames()));
  };
  auto makeSharedSeedLayout = [&](unsigned descriptorRows,
                                  unsigned sourceWarpGroups,
                                  unsigned descriptorCols)
      -> std::optional<LinearLayout> {
    auto dataLayout = shmemLl.pseudoinvert();
    auto dataDims = to_vector(dataLayout.getInDimNames());
    assert(dataDims.size() == 2);
    int64_t reshapeVolume = static_cast<int64_t>(descriptorRows) *
                            sourceWarpGroups * descriptorCols;
    if (dataLayout.getTotalInDimSize() != reshapeVolume)
      return std::nullopt;
    return dataLayout
        .reshapeIns({{dataDims[0], static_cast<int32_t>(descriptorRows)},
                     {kWarp, static_cast<int32_t>(sourceWarpGroups)},
                     {dataDims[1], static_cast<int32_t>(descriptorCols)}})
        .sublayout({dataDims[0], dataDims[1]},
                   to_vector(dataLayout.getOutDimNames()));
  };

  llvm::SmallVector<LinearLayout> layouts;
  auto pushUnique = [&](const LinearLayout &layout) {
    if (!llvm::is_contained(layouts, layout))
      layouts.push_back(layout);
  };
  auto pushReassignedVariant = [&](const LinearLayout &layout,
                                   unsigned promotedColIndex,
                                   std::optional<unsigned> demotedRowIndex) {
    auto bases = layout.getBases();
    const auto &rowBases = bases.lookup(kRow);
    const auto &colBases = bases.lookup(kCol);
    if (promotedColIndex >= colBases.size())
      return;
    if (demotedRowIndex && *demotedRowIndex >= rowBases.size())
      return;
    if (llvm::all_of(colBases[promotedColIndex],
                     [](int32_t value) { return value == 0; }))
      return;

    std::vector<std::vector<int32_t>> newRowBases;
    newRowBases.reserve(rowBases.size() + 1);
    newRowBases.emplace_back(colBases[promotedColIndex].begin(),
                             colBases[promotedColIndex].end());
    for (auto [idx, basis] : llvm::enumerate(rowBases)) {
      if (demotedRowIndex && idx == *demotedRowIndex)
        continue;
      newRowBases.emplace_back(basis.begin(), basis.end());
    }
    if (demotedRowIndex) {
      auto demoted = rowBases[*demotedRowIndex];
      newRowBases.emplace_back(demoted.begin(), demoted.end());
    }

    std::vector<std::vector<int32_t>> newColBases;
    newColBases.reserve(colBases.size() - 1);
    for (auto [idx, basis] : llvm::enumerate(colBases)) {
      if (idx == promotedColIndex)
        continue;
      newColBases.emplace_back(basis.begin(), basis.end());
    }

    bases[kRow] = std::move(newRowBases);
    bases[kCol] = std::move(newColBases);
    pushUnique(LinearLayout(std::move(bases), to_vector(layout.getOutDims()),
                            /*requireSurjective=*/false));
  };
  auto pushRotatedRowVariant = [&](const LinearLayout &layout,
                                   unsigned demotedRowIndex) {
    auto bases = layout.getBases();
    const auto &rowBases = bases.lookup(kRow);
    if (demotedRowIndex >= rowBases.size())
      return;

    std::vector<std::vector<int32_t>> newRowBases;
    newRowBases.reserve(rowBases.size());
    for (auto [idx, basis] : llvm::enumerate(rowBases)) {
      if (idx == demotedRowIndex)
        continue;
      newRowBases.emplace_back(basis.begin(), basis.end());
    }
    auto demoted = rowBases[demotedRowIndex];
    newRowBases.emplace_back(demoted.begin(), demoted.end());
    bases[kRow] = std::move(newRowBases);
    pushUnique(LinearLayout(std::move(bases), to_vector(layout.getOutDims()),
                            /*requireSurjective=*/false));
  };
  auto pushMovedRowVariant = [&](const LinearLayout &layout,
                                 unsigned sourceRowIndex,
                                 unsigned destRowIndex) {
    auto bases = layout.getBases();
    const auto &rowBases = bases.lookup(kRow);
    if (sourceRowIndex >= rowBases.size() || destRowIndex >= rowBases.size() ||
        sourceRowIndex == destRowIndex)
      return;

    std::vector<std::vector<int32_t>> newRowBases;
    newRowBases.reserve(rowBases.size());
    for (auto [idx, basis] : llvm::enumerate(rowBases)) {
      if (idx == sourceRowIndex)
        continue;
      newRowBases.emplace_back(basis.begin(), basis.end());
    }
    auto moved = rowBases[sourceRowIndex];
    newRowBases.insert(newRowBases.begin() + destRowIndex,
                       std::vector<int32_t>(moved.begin(), moved.end()));
    bases[kRow] = std::move(newRowBases);
    pushUnique(LinearLayout(std::move(bases), to_vector(layout.getOutDims()),
                            /*requireSurjective=*/false));
  };
  auto pushSortedInputBasesVariant = [&](const LinearLayout &layout) {
    auto bases = layout.getBases();
    auto sortKey = [](ArrayRef<int32_t> basis) {
      for (int32_t value : basis) {
        if (value != 0)
          return std::abs(value);
      }
      return 0;
    };
    for (auto &dimBases : llvm::make_second_range(bases)) {
      llvm::sort(dimBases, [&](const auto &lhs, const auto &rhs) {
        return sortKey(lhs) < sortKey(rhs);
      });
    }
    pushUnique(LinearLayout(std::move(bases), to_vector(layout.getOutDims()),
                            /*requireSurjective=*/false));
  };
  auto addWarpx2DescriptorVariants = [&]() {
    if (message.sourceWarpGroups > 1) {
      unsigned warpBits = llvm::Log2_32(message.sourceWarpGroups);
      for (unsigned rowFoldBits = 0; rowFoldBits <= warpBits; ++rowFoldBits) {
        for (unsigned colFoldBits = 0;
             rowFoldBits + colFoldBits <= warpBits; ++colFoldBits) {
          unsigned foldedBits = rowFoldBits + colFoldBits;
          unsigned residualWarpGroups = message.sourceWarpGroups >> foldedBits;
          pushUnique(makeLayout(message.descriptorRows << rowFoldBits,
                                residualWarpGroups,
                                cvt.getInDimSize(kCol) << colFoldBits));
        }
      }
    }

    llvm::SmallVector<LinearLayout> seedLayouts = layouts;
    unsigned descriptorColBits = llvm::Log2_32(message.descriptorShape[1]);
    for (const auto &layout : seedLayouts) {
      const auto &rowBases = layout.getBases().lookup(kRow);
      const auto &colBases = layout.getBases().lookup(kCol);
      for (unsigned demotedRowIndex = 0; demotedRowIndex < rowBases.size();
           ++demotedRowIndex)
        pushRotatedRowVariant(layout, demotedRowIndex);
      for (unsigned sourceRowIndex = 0; sourceRowIndex < rowBases.size();
           ++sourceRowIndex)
        for (unsigned destRowIndex = 0; destRowIndex < rowBases.size();
             ++destRowIndex)
          pushMovedRowVariant(layout, sourceRowIndex, destRowIndex);
      if (colBases.size() <= descriptorColBits)
        continue;
      for (unsigned promotedColIndex = descriptorColBits;
           promotedColIndex < colBases.size(); ++promotedColIndex) {
        pushReassignedVariant(layout, promotedColIndex, std::nullopt);
        for (unsigned demotedRowIndex = 0; demotedRowIndex < rowBases.size();
             ++demotedRowIndex)
          pushReassignedVariant(layout, promotedColIndex, demotedRowIndex);
      }
    }
  };
  pushUnique(makeLayout(message.descriptorRows, message.sourceWarpGroups,
                        cvt.getInDimSize(kCol)));
  if (auto directSharedLayout =
          makeSharedSeedLayout(message.descriptorRows,
                               message.sourceWarpGroups,
                               cvt.getInDimSize(kCol))) {
    pushUnique(*directSharedLayout);
    pushSortedInputBasesVariant(*directSharedLayout);
  }
  // Once getTMemCopyPlans(...) has chosen the core warpx2 descriptor family
  // (`32x4` for `01_23`, `64x2` for `02_13`), both public warpx2 paths need
  // the same bounded linear-layout search: repartition any extra source warp
  // bits across descriptor rows / cols and then try the small family of row
  // reorders and promoted low-order column bits that MMAShared descriptors can
  // still materialize.
  if (message.atom.multicast == 1 || message.atom.multicast == 2)
    addWarpx2DescriptorVariants();
  return layouts;
}

bool canRepresentAsMMASmemDescriptor(const LinearLayout &ll,
                                     llvm::ArrayRef<unsigned> instrShape,
                                     int bitwidth, unsigned MNdim,
                                     int mmaVersion, bool allowTransposed) {
  if (ll.getNumOutDims() != 2)
    return false;
  auto dims = to_vector(ll.getInDimNames());
  if (dims.size() != 2)
    return false;
  auto ctx = dims[0].getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  if (!ll.hasOutDim(kOffset))
    return false;
  for (auto [dim, instrSize] : llvm::zip(ll.getInDimNames(), instrShape)) {
    if (instrSize > ll.getInDimSize(dim))
      return false;
  }
  auto CGALayout = triton::gpu::CGAEncodingAttr::get1CTALayout(ctx, 2);
  for (bool fp4Padded :
       (bitwidth == 4 ? SmallVector<bool>({false, true})
                      : SmallVector<bool>({false}))) {
    for (auto transposed : {false, true}) {
      for (int swizzling : {0, 32, 64, 128}) {
        if (transposed && !allowTransposed)
          continue;
        auto shmemEnc = triton::gpu::NVMMASharedEncodingAttr::get(
            ctx, swizzling, transposed, std::max(8, bitwidth), fp4Padded,
            CGALayout);
        auto shmemTile =
            getCoreMatrixLinearLayout(shmemEnc, /*disableSwizzle=*/false);
        auto outDims = to_vector(shmemTile.getOutDims());
        outDims[0].first = dims[0];
        outDims[1].first = dims[1];
        shmemTile = LinearLayout(shmemTile.getBases(), outDims,
                                 /*requireSurjective=*/false);
        if (bitwidth == 4) {
          shmemTile =
              LinearLayout::identity1D(2, kOffset, dims[1]) * shmemTile;
        }
        if (transposed) {
          shmemTile = transposeLinearLayout(shmemTile, {1, 0});
        }
        auto shmemTileInv = shmemTile.pseudoinvert();

        int leadingDim = transposed ? 0 : 1;
        int stridedDim = transposed ? 1 : 0;
        bool MNContig = (MNdim == 0) == transposed;
        if (swizzling == 0 && MNContig) {
          std::swap(leadingDim, stridedDim);
        }

        auto log2RowsTile = shmemTileInv.getInDimSizeLog2(dims[leadingDim]);
        if (llvm::Log2_32(instrShape[leadingDim]) > log2RowsTile)
          (void)ll.getBasis(dims[leadingDim], log2RowsTile, kOffset);
        auto log2ColsTile = shmemTileInv.getInDimSizeLog2(dims[stridedDim]);
        if (llvm::Log2_32(instrShape[stridedDim]) > log2ColsTile)
          (void)ll.getBasis(dims[stridedDim], log2ColsTile, kOffset);

        auto bases = shmemTileInv.getBases();
        for (int d : {0, 1}) {
          for (int i = 1; i < instrShape[d] / shmemTileInv.getInDimSize(dims[d]);
               i *= 2) {
            auto stride = ll.getBasis(
                dims[d], shmemTileInv.getInDimSizeLog2(dims[d]), kOffset);
            bases[dims[d]].push_back({stride * i});
          }
        }
        auto maxBasis = 0;
        for (auto dimBases : llvm::make_second_range(bases)) {
          for (auto basis : dimBases) {
            maxBasis = std::max(maxBasis, basis[0]);
          }
        }
        shmemTileInv = LinearLayout(std::move(bases),
                                    {{kOffset, llvm::NextPowerOf2(maxBasis)}},
                                    /*requireSurjective=*/false);
        shmemTileInv *=
            LinearLayout::identity1D(1, dims[0], StringAttr::get(ctx, "block"));
        auto reps = getReps(ll, shmemTileInv);
        if (reps.has_value())
          return true;
      }
    }
  }
  return false;
}

bool canSynthesizeTMemCopySharedDescriptorPlan(gpu::MemDescType srcTy,
                                               const LinearLayout &shmemLl,
                                               const LinearLayout &cvt,
                                               const TMemCopyPlan &plan,
                                               int bitwidth) {
  return llvm::all_of(plan.messages, [&](const auto &message) {
    if (message.useDirectSeedDescriptor &&
        getDirectTMemCopySeedDescriptorImm(srcTy, plan.family))
      return true;
    bool allowTransposed = plan.family == TMemCopyFamily::Dense4x256b;
    auto srcDescLayouts =
        getTMemCopyDescriptorLayouts(srcTy, shmemLl, cvt, message);
    return llvm::any_of(srcDescLayouts, [&](const LinearLayout &srcDescLayout) {
      static constexpr unsigned kDescriptorOrientations[] = {0u, 1u};
      return llvm::any_of(ArrayRef(kDescriptorOrientations),
                          [&](unsigned mnDim) {
                            return canRepresentAsMMASmemDescriptor(
                                srcDescLayout, message.descriptorShape,
                                bitwidth, mnDim, 5, allowTransposed);
                          });
    });
  });
}

} // namespace mlir::triton::nvidia_gpu

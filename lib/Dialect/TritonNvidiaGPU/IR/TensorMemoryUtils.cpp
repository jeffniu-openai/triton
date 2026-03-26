#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
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
} // namespace

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(gpu::MemDescType type, std::string *error) {
  Attribute enc = type.getEncoding();
  if (!isTensorMemoryEncoding(enc) || isa<TensorMemoryScalesEncodingAttr>(enc))
    return std::nullopt;
  auto rank = cast<gpu::LayoutEncodingTrait>(enc).getRank();
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
tryMakeTMemViewEncoding(MLIRContext *ctx, LinearLayout ll, bool twoCTAs,
                        std::string *error) {
  auto kBlock = StringAttr::get(ctx, "block");
  bool hadBlock = ll.hasInDim(kBlock);
  decltype(ll.getBases().lookup(kBlock)) originalBlockBases;
  if (hadBlock)
    originalBlockBases = ll.getBases().lookup(kBlock);
  if (ll.hasInDim(kBlock))
    ll = ll.removeZeroBasesAlongDim(kBlock);
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
  if (!twoCTAs)
    return std::nullopt;

  if (!hadBlock)
    return std::nullopt;
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
  auto memTy = gpu::MemDescType::get(srcShape, elemTy, srcEncoding,
                                     TensorMemorySpaceAttr::get(ctx),
                                     /*mutableMemory=*/false, srcShape);

  std::string error;
  auto resultTy = inferTMemReshapeOpType(memTy, dstShape, &error);
  if (succeeded(resultTy)) {
    dstEncoding = resultTy->getEncoding();
  } else {
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
  auto layoutRank = cast<LayoutEncodingTrait>(srcEncoding).getRank();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank > 0) {
    dstEncoding = srcEncoding;
    return success();
  }

  std::string error;
  auto result =
      inferTMemIndexEncoding(srcShape, dstShape, dstAllocShape, srcEncoding,
                             &error);
  if (succeeded(result)) {
    dstEncoding = *result;
    return success();
  }
  dstEncoding = srcEncoding;
  return success();
}

LogicalResult inferTMemSubsliceOpEncoding(ArrayRef<int64_t> srcShape,
                                          Attribute srcEncoding,
                                          ArrayRef<int64_t> dstShape,
                                          ArrayRef<int32_t> offsets,
                                          Attribute &dstEncoding,
                                          std::optional<Location> loc) {
  auto layoutRank = cast<LayoutEncodingTrait>(srcEncoding).getRank();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank > 0 &&
      srcShape.drop_front(extraRank) == dstShape.drop_front(extraRank) &&
      llvm::all_of(offsets.drop_front(extraRank),
                   [](int32_t offset) { return offset == 0; })) {
    dstEncoding = srcEncoding;
    return success();
  }

  std::string error;
  auto result =
      inferTMemSubsliceEncoding(srcShape, srcEncoding, dstShape, offsets,
                                &error);
  if (succeeded(result)) {
    dstEncoding = *result;
    return success();
  }
  dstEncoding = srcEncoding;
  return success();
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
  auto srcEnc = getCanonicalTMemLinearEncoding(srcShape, srcEncoding, error);
  if (!srcEnc) {
    if (error && error->empty())
      *error = "expected canonical tensor memory linear encoding";
    return failure();
  }

  auto ll = normalizeTensorMemoryLinearLayoutForAnalysis(
      srcEnc->getLinearLayout());
  auto layoutRank = srcEnc->getRank();
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
    return *srcEnc;
  }

  auto *ctx = srcEncoding.getContext();
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
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
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
  auto result = tryMakeTMemViewEncoding(ctx, *dstLayout, srcEnc->getTwoCTAs(),
                                        error);
  if (!result)
    return failure();
  if (!tensorMemoryLinearLayoutMatchesShape(result->getLinearLayout(),
                                            dstShape.drop_front(extraRank))) {
    if (error)
      *error = "tensor memory view is not representable as a standalone TMEM "
               "linear layout";
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
  auto srcEnc = getCanonicalTMemLinearEncoding(srcShape, srcEncoding, error);
  if (!srcEnc) {
    if (error && error->empty())
      *error = "expected canonical tensor memory linear encoding";
    return failure();
  }

  auto ll = srcEnc->getLinearLayout();
  auto layoutRank = srcEnc->getRank();
  auto extraRank = static_cast<int64_t>(srcShape.size()) - layoutRank;
  if (extraRank < 0) {
    if (error)
      *error = "invalid tensor memory rank/layout combination";
    return failure();
  }

  if (extraRank > 0)
    return *srcEnc;

  auto *ctx = srcEncoding.getContext();
  if (layoutRank == 0) {
    if (error)
      *error = "tensor memory layout rank must be greater than zero";
    return failure();
  }
  SmallVector<StringAttr> outDims = llvm::to_vector(ll.getOutDimNames());
  outDims.erase(outDims.begin());
  ll = ll.sublayout(llvm::to_vector(ll.getInDimNames()), outDims);
  auto resultLayoutShape = dstAllocShape.take_back(layoutRank - 1);
  if (static_cast<int64_t>(ll.getTotalOutDimSize()) !=
      product<int64_t>(resultLayoutShape)) {
    if (error)
      *error = "failed to infer tensor memory encoding for memdesc_index";
    return failure();
  }
  ll = ll.reshapeOuts(standardOutDimPairs(ctx, resultLayoutShape));

  auto result =
      tryMakeTMemViewEncoding(ctx, std::move(ll), srcEnc->getTwoCTAs(), error);
  if (!result)
    return failure();
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
  auto maybeDstEnc = inferTMemIndexEncoding(srcTy, *dstTy);
  if (failed(maybeDstEnc)) {
    if (error)
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
  SmallVector<int64_t> offsets64(offsets.begin(), offsets.end());
  auto maybeDstEnc = inferTMemSubsliceEncoding(srcTy, *dstTy, offsets64);
  if (failed(maybeDstEnc)) {
    if (error)
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
  auto srcEnc = getCanonicalTMemLinearEncoding(srcTy, error);
  if (!srcEnc)
    return failure();

  auto *ctx = srcTy.getContext();
  auto srcShape = srcTy.getShape();
  auto layoutSrcShape = srcShape;
  auto layoutDstShape = dstShape;
  int64_t layoutElems =
      static_cast<int64_t>(srcEnc->getLinearLayout().getTotalOutDimSize());

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape) {
    while (!shape.empty() && shape.front() == 1 &&
           product<int64_t>(shape.drop_front()) >= layoutElems)
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape);
  layoutDstShape = stripLeadingUnitDims(layoutDstShape);

  if (product<int64_t>(layoutSrcShape) > layoutElems) {
    if (layoutSrcShape.size() != static_cast<size_t>(srcEnc->getRank()) + 1) {
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
  if (layoutElems != product<int64_t>(layoutSrcShape)) {
    if (error)
      *error = "TMEM reshape rank does not match the canonical TMEM layout";
    return failure();
  }

  auto dstLL = reshapeLayout(ctx, srcEnc->getLinearLayout(), layoutDstShape);
  auto result =
      tryMakeTMemViewEncoding(ctx, std::move(dstLL), srcEnc->getTwoCTAs(),
                              error);
  if (!result)
    return failure();

  SmallVector<int64_t> dstAllocShape = llvm::to_vector(
      srcTy.getAllocShape().take_front(srcTy.getAllocShape().size() -
                                       srcTy.getShape().size()));
  dstAllocShape.append(dstShape.begin(), dstShape.end());
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
              bool unpacked = false) {
  // We will fill in the returned value recursively (if it exists)

  // Remove broadcasting in the registers
  auto removeBroadcastSrc = actionRemoveBroadcastedRegs(cvt);
  if (!removeBroadcastSrc.isIdentity()) {
    auto prmtCvt = removeBroadcastSrc.apply(cvt);
    auto info =
        lowerTMemLdSt(prmtCvt, maxnreg, bitwidth, emitError, unpacked);
    if (failed(info))
      return failure();
    info->broadcast = std::move(removeBroadcastSrc);
    return info;
  }
  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kReg = S("register");
  auto kLane = S("lane");
  auto kRow = S("row");
  auto kCol = S("col");
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
    // When unpacked each register moves 32/bitwidth (= 2) columns
    if (unpacked) {
      quot = LinearLayout::zeros1D(1, kReg, kCol, 32 / bitwidth) * quot;
    }
    auto info = lowerTMemLdSt(quot, maxnreg, newBitwidth, emitError, unpacked);
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
    auto tile = getTileLayout(ctx, atom, unpacked, /*withWarp=*/true);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      // Cannot match more than one
      msgInfo = {atom, std::get<0>(*maybeReps), std::get<1>(*maybeReps),
                 std::get<2>(*maybeReps)};
      break;
    }
  }
  std::optional<uint32_t> secondHalfOffset = std::nullopt;
  if (!msgInfo) {
    // Quotient by the smaller tile and then, if possible, we set the
    // secondHalfOffset to the last kLane basis
    auto tile = getTileLayout(ctx, TMemAccessAtom::I16x32bx2, unpacked,
                              /*withWarp=*/true);
    auto maybeReps = getVec(cvt, tile, maxnreg);
    if (maybeReps) {
      auto &reps = std::get<0>(*maybeReps);
      // 16x32bx2 needs a distinct lane=16 basis in the repetition layout to
      // encode the second half offset. Some speculative candidate layouts
      // match the tile quotient but only have 16 active lanes; reject them
      // cleanly instead of indexing a non-existent lane basis.
      auto laneIt = reps.getBases().find(kLane);
      if (laneIt == reps.getBases().end() || laneIt->second.size() <= 4)
        maybeReps.reset();
    }
    if (maybeReps) {
      auto [reps, perm, numRegsPerMessage] = std::move(*maybeReps);
      // Find the last kLane basis and use it as secondHalfOffset
      auto row = reps.getBasis(kLane, 4, kRow);
      auto col = reps.getBasis(kLane, 4, kCol);
      secondHalfOffset = (row << 16) | col;
      // We "quotient it out", meaning we remove the last basis from reps
      auto basis = reps.getBases();
      basis[kLane][4] = {0, 0};
      reps = LinearLayout(std::move(basis), reps.getOutDims(),
                          /*isSurjective=*/false);
      msgInfo = {TMemAccessAtom::I16x32bx2, reps, perm, numRegsPerMessage};
    }
  }

  if (!msgInfo) {
    if (emitError) {
      emitError()
          << "Failed to lower TMEM load/store: unsupported dst layout\n" +
                 cvt.toString();
    }
    return failure();
  }
  auto info = std::move(*msgInfo);
  info.secondHalfOffset = secondHalfOffset;
  return info;
}

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError) {
  auto *ctx = regTy.getContext();
  auto S = [ctx](StringRef str) { return StringAttr::get(ctx, str); };
  auto kBlock = S("block");
  auto kWarp = S("warp");
  auto kRow = S("row");
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

    std::string error;
    auto maybeCanonical = getCanonicalTMemLinearEncoding(memTy, &error);
    if (!maybeCanonical) {
      if (emitError) {
        emitError() << (error.empty()
                            ? "TMEM descriptor view is not representable as a "
                              "standalone TMEM layout"
                            : error);
      }
      return LinearLayout();
    }
    return squeezeTrivialBlock(maybeCanonical->getLinearLayout());
  }();
  if (memLayout.getNumOutDims() == 0)
    return failure();
  auto regLayout = squeezeTrivialBlock(toLinearLayout(regTy));
  auto cvt = regLayout.invertAndCompose(memLayout);
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
  if (regLayout.getInDimSizeLog2(kWarp) < 2 || memLayout.getInDimSizeLog2(kRow) < 7) {
    if (emitError) {
      emitError() << "TMEM load/store requires row anchors at 32 and 64 for the "
                     "selected register and memory layouts. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  // Warps 0-3 must map to row=32 and row=64 whether with broadcasting or not.
  if (!(regLayout.getBasis(kWarp, 0) == memLayout.getBasis(kRow, 5) &&
        regLayout.getBasis(kWarp, 1) == memLayout.getBasis(kRow, 6))) {
    if (emitError) {
      emitError() << "warps=1,2 must map to rows=32,64. Got:\n"
                  << regLayout.toString() << "\n"
                  << memLayout.toString();
    }
    return failure();
  }
  // Map warp bases to row=32 and row=64 in the cvt. This would be done
  // automatically in `invertAndCompose` if we had a different dimension name
  // for these rows. We can do this in the future if needed.
  auto bases = cvt.getBases();
  bases[kWarp][0] = {32, 0};
  bases[kWarp][1] = {64, 0};
  cvt = LinearLayout(std::move(bases), cvt.getOutDims(),
                     /*isSurjective=*/cvt.isSurjective());

  int bitwidth = memTy.getElementTypeBitWidth();
  auto info = lowerTMemLdSt(cvt, maxnreg, bitwidth, emitError);
  if (failed(info))
    return failure();

  auto kReg = *regLayout.getInDimNames().begin();
  if (bitwidth == 32) {
    auto expectedValueCount = getExpectedTMemLoadValueCount(*info, bitwidth);
    if (failed(expectedValueCount) ||
        *expectedValueCount != regLayout.getInDimSize(kReg)) {
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
  if (cvt.getInDimSize(kRow) != 128)
    return std::nullopt;

  auto multicastBit = [&](int i) {
    assert(i == 0 || i == 1);
    return cvt.getBasis(kRow, llvm::Log2_32(32) + i, kOffset) == 0;
  };
  auto multicast = multicastBit(0) | multicastBit(1) << 1;
  int totalBits = cvt.getInDimSize(kCol) * bitwidth;
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

bool canRepresentAsMMASmemDescriptor(const LinearLayout &ll,
                                     llvm::ArrayRef<unsigned> instrShape,
                                     int bitwidth, unsigned MNdim,
                                     int mmaVersion) {
  if (ll.getNumOutDims() != 2)
    return false;
  auto dims = to_vector(ll.getInDimNames());
  if (dims.size() != 2)
    return false;
  auto ctx = dims[0].getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto CGALayout = triton::gpu::CGAEncodingAttr::get1CTALayout(ctx, 2);

  for (bool fp4Padded :
       (bitwidth == 4 ? SmallVector<bool>({false, true})
                      : SmallVector<bool>({false}))) {
    for (auto transposed : {false, true}) {
      for (int swizzling : {0, 32, 64, 128}) {
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
        if (llvm::Log2_32(instrShape[leadingDim]) > log2RowsTile) {
          if (log2RowsTile >= ll.getInDimSizeLog2(dims[leadingDim]))
            continue;
          (void)ll.getBasis(dims[leadingDim], log2RowsTile, kOffset);
        }
        auto log2ColsTile = shmemTileInv.getInDimSizeLog2(dims[stridedDim]);
        if (llvm::Log2_32(instrShape[stridedDim]) > log2ColsTile) {
          if (log2ColsTile >= ll.getInDimSizeLog2(dims[stridedDim]))
            continue;
          (void)ll.getBasis(dims[stridedDim], log2ColsTile, kOffset);
        }

        auto bases = shmemTileInv.getBases();
        bool invalidCandidate = false;
        for (int d : {0, 1}) {
          auto log2Tile = shmemTileInv.getInDimSizeLog2(dims[d]);
          if (log2Tile >= ll.getInDimSizeLog2(dims[d]) &&
              instrShape[d] > shmemTileInv.getInDimSize(dims[d])) {
            invalidCandidate = true;
            break;
          }
          for (int i = 1; i < instrShape[d] / shmemTileInv.getInDimSize(dims[d]);
               i *= 2) {
            auto stride = ll.getBasis(dims[d], log2Tile, kOffset);
            bases[dims[d]].push_back({stride * i});
          }
        }
        if (invalidCandidate)
          continue;
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
        if (getReps(ll, shmemTileInv).has_value())
          return true;
      }
    }
  }
  return false;
}

} // namespace mlir::triton::nvidia_gpu

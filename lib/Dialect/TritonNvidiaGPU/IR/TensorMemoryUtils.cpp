#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "triton/Tools/LayoutUtils.h"
#include "third_party/f2reduce/f2reduce.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSwitch.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <limits>
#include <tuple>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

namespace {

constexpr int maxRegisters = 256;
constexpr int largestTmemLoadStore = 128;

static std::optional<TMemLdStQueryLayout>
getTMemViewAnalysisLayout(ArrayRef<int64_t> shape, Attribute encoding,
                          std::string *error = nullptr);

} // namespace

static TMemSubwordPhaseStatus
getTMemElementOffsetModuloStatus(MemDescType memTy, uint32_t modulus) {
  if (modulus <= 1)
    return TMemSubwordPhaseStatus::KnownZero;
  if (!memTy || !isTensorMemoryEncoding(memTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
    return TMemSubwordPhaseStatus::KnownZero;
  }

  // A descriptor whose active shape covers the full allocation image has no
  // encoded subview displacement. Leading allocation dimensions of extent one
  // cannot hide a runtime origin either; they only preserve container rank.
  // Nontrivial hidden prefixes stay conservative because a value selected from
  // one of those slices can carry an origin that is not visible in the current
  // active shape alone.
  auto allocShape = memTy.getAllocShape();
  auto shape = memTy.getShape();
  if (shape == allocShape)
    return TMemSubwordPhaseStatus::KnownZero;
  if (allocShape.size() >= shape.size() &&
      allocShape.take_back(shape.size()) == shape &&
      llvm::all_of(allocShape.drop_back(shape.size()),
                   [](int64_t dim) { return dim == 1; }))
    return TMemSubwordPhaseStatus::KnownZero;

  std::string error;
  auto maybeQuery = getTMemViewAnalysisLayout(memTy.getShape(),
                                              memTy.getEncoding(), &error);
  if (!maybeQuery)
    return TMemSubwordPhaseStatus::Unknown;

  auto *ctx = memTy.getContext();
  auto kCol = StringAttr::get(ctx, "col");
  const LinearLayout &layout = maybeQuery->layout;
  if (!layout.hasInDim(kCol))
    return TMemSubwordPhaseStatus::KnownZero;

  for (unsigned idx = 0, e = layout.getInDimSizeLog2(kCol); idx < e; ++idx) {
    for (int32_t value : layout.getBasis(kCol, idx)) {
      if (value % static_cast<int32_t>(modulus) != 0)
        return TMemSubwordPhaseStatus::Unknown;
    }
  }
  return TMemSubwordPhaseStatus::KnownZero;
}

static TMemSubwordPhaseStatus
getTMemIndexElementOffsetModuloStatus(MemDescIndexOp indexOp,
                                      uint32_t modulus) {
  auto srcTy = dyn_cast_if_present<MemDescType>(indexOp.getSrc().getType());
  if (!srcTy || srcTy.getRank() == 0)
    return TMemSubwordPhaseStatus::Unknown;

  // The indexed result inherits any origin uncertainty already represented by
  // the source type.  Do not inspect the source producer chain; the source type
  // and the immediate index op are the local facts available here.
  if (getTMemElementOffsetModuloStatus(srcTy, modulus) !=
      TMemSubwordPhaseStatus::KnownZero)
    return TMemSubwordPhaseStatus::Unknown;

  auto offsetIsAligned = [&](int64_t index) {
    SmallVector<int32_t> offsets(srcTy.getRank(), 0);
    offsets.front() = static_cast<int32_t>(index);
    auto rowCol = tryGetTMemViewPhysicalRowElementCol(srcTy, offsets);
    return rowCol &&
           rowCol->second % static_cast<uint32_t>(modulus) == 0;
  };

  APInt staticIndex;
  if (matchPattern(indexOp.getIndex(), m_ConstantInt(&staticIndex))) {
    return offsetIsAligned(staticIndex.getSExtValue())
               ? TMemSubwordPhaseStatus::KnownZero
               : TMemSubwordPhaseStatus::Unknown;
  }

  int64_t indexedExtent = srcTy.getShape().front();
  for (int64_t bit = 1; bit < indexedExtent; bit <<= 1) {
    if (!offsetIsAligned(bit))
      return TMemSubwordPhaseStatus::Unknown;
  }
  return TMemSubwordPhaseStatus::KnownZero;
}

TMemSubwordPhaseStatus getTMemSubwordPhaseStatus(Value memDesc) {
  if (!memDesc)
    return TMemSubwordPhaseStatus::Unknown;
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy || memTy.getElementTypeBitWidth() >= 32)
    return TMemSubwordPhaseStatus::KnownZero;
  uint32_t modulus = getTMemElementsPerWord(memTy.getElementTypeBitWidth());
  if (auto indexOp = memDesc.getDefiningOp<MemDescIndexOp>())
    return getTMemIndexElementOffsetModuloStatus(indexOp, modulus);
  return getTMemElementOffsetModuloStatus(memTy, modulus);
}

TMemSubwordPhaseStatus getTMemElementOffsetModuloStatus(Value memDesc,
                                                        uint32_t modulus) {
  if (!memDesc)
    return TMemSubwordPhaseStatus::Unknown;
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  return getTMemElementOffsetModuloStatus(memTy, modulus);
}

bool isTMemLoadReductionAddressAligned(Value memDesc) {
  if (!memDesc)
    return false;
  auto memTy = dyn_cast_if_present<MemDescType>(memDesc.getType());
  if (!memTy)
    return false;
  uint32_t elementBitwidth = memTy.getElementTypeBitWidth();
  if (elementBitwidth == 0)
    return false;
  uint32_t b128ElementAlignment =
      elementBitwidth >= 128 ? 1 : 128 / elementBitwidth;
  return getTMemElementOffsetModuloStatus(memTy, b128ElementAlignment) ==
         TMemSubwordPhaseStatus::KnownZero;
}

bool mayHaveNonZeroTMemSubwordPhase(Value memDesc) {
  return getTMemSubwordPhaseStatus(memDesc) !=
         TMemSubwordPhaseStatus::KnownZero;
}

static std::optional<TensorMemoryScalesEncodingAttr>
getTypeLocalTMemScalesEncoding(MemDescType memTy) {
  if (!memTy)
    return std::nullopt;
  if (auto scales =
          dyn_cast<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
    return scales;
  auto storageType = getMMAv5ScaleStorageType(memTy);
  if (!storageType)
    return std::nullopt;
  return dyn_cast<TensorMemoryScalesEncodingAttr>(
      storageType->getEncoding());
}

static bool isTypeLocalTMemScalesDescriptorView(MemDescType memTy) {
  return memTy && !isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()) &&
         getTypeLocalTMemScalesEncoding(memTy).has_value();
}

FailureOr<std::optional<TMemAccessAtom>>
parseTMemAccessAtomName(StringRef atomName, bool allowAuto,
                        bool splitNAsPacked) {
  if (allowAuto && atomName == "auto")
    return std::optional<TMemAccessAtom>{};

  std::optional<TMemAccessAtom> splitNAtom =
      splitNAsPacked ? TMemAccessAtom::I16x32bx2 : TMemAccessAtom::I32x32b;
  auto atom =
      llvm::StringSwitch<std::optional<TMemAccessAtom>>(atomName)
          .Case("32x32b", TMemAccessAtom::I32x32b)
          .Case("16x64b", TMemAccessAtom::I16x64b)
          .Case("16x128b", TMemAccessAtom::I16x128b)
          .Case("16x256b", TMemAccessAtom::I16x256b)
          .Case("16x32bx2", TMemAccessAtom::I16x32bx2)
          .Case("32x32b_splitn", splitNAtom)
          .Default(std::nullopt);
  if (!atom)
    return failure();
  return atom;
}

llvm::SmallVector<int64_t>
getTMemMemDescIndexResultAllocShape(MemDescType srcTy) {
  if (!srcTy || srcTy.getRank() == 0)
    return {};
  llvm::SmallVector<int64_t> result =
      llvm::to_vector(srcTy.getAllocShape().drop_front());
  if (srcTy.getRank() <= 1 || result.empty() ||
      srcTy.getAllocShape().size() < static_cast<size_t>(srcTy.getRank()))
    return result;

  // Indexing a leading dimension that was already narrowed by a subview must
  // preserve the omitted physical extent in the result type.  The omitted
  // dimension is not always a row selector after reshape/permute chains: it can
  // select a column packet half.  Use the current source type/layout to decide
  // which result dimension owns the hidden allocation extent.
  if (srcTy.getShape()[0] < srcTy.getAllocShape()[0]) {
    int64_t factor = srcTy.getAllocShape()[0];
    auto layoutTrait = dyn_cast<LayoutEncodingTrait>(srcTy.getEncoding());
    int64_t layoutRank = layoutTrait ? layoutTrait.getRank() : srcTy.getRank();
    int64_t extraRank = srcTy.getRank() - layoutRank;

    // If the indexed-away narrowed dimension lives outside the tensor-memory
    // encoding rank and the remaining row dimension is smaller than a full
    // direct ld/st row footprint, the current layout no longer carries the row
    // origin bit. Preserve it as a row allocation extent so direct ld/st rejects
    // the non-self-contained row slice instead of treating the runtime taddr row
    // phase as a compact origin-zero tile. When the remaining row span already
    // covers the full footprint, this pattern is a column split and the normal
    // physical-offset query can safely decide whether the hidden extent belongs
    // to columns.
    bool twoCTAs = getTensorMemoryTwoCTAs(srcTy.getEncoding()).value_or(false);
    int64_t fullRowFootprint = twoCTAs ? 256 : 128;
    bool narrowedExtraRankRowSlice = extraRank > 0 && !result.empty() &&
                                     result[0] < fullRowFootprint;
    if (narrowedExtraRankRowSlice) {
      result[0] *= factor;
    } else {
      SmallVector<int32_t> offsets(srcTy.getRank(), 0);
      offsets.front() = 1;
      auto rowCol = tryGetTMemViewPhysicalRowElementCol(srcTy, offsets);
      if (rowCol && rowCol->second != 0 && rowCol->first == 0) {
        result.back() *= factor;
      } else {
        result[0] *= factor;
      }
    }
  }
  return result;
}

bool isUnsupportedOriginChangingTMemRowSubview(MemDescType memTy) {
  if (!memTy || memTy.getRank() < 2 ||
      memTy.getAllocShape().size() < static_cast<size_t>(memTy.getRank()))
    return false;

  int64_t logicalRows = memTy.getShape()[memTy.getRank() - 2];
  int64_t allocRows = memTy.getAllocShape()[memTy.getAllocShape().size() - 2];
  if (logicalRows >= allocRows)
    return false;

  auto maybeQuery = inferTypeLocalTMemLdStQueryLayout(memTy, /*error=*/nullptr);
  if (failed(maybeQuery))
    return false;

  auto kRow = StringAttr::get(memTy.getContext(), "row");
  if (!maybeQuery->layout.hasInDim(kRow))
    return true;
  for (unsigned idx = 0, e = maybeQuery->layout.getInDimSizeLog2(kRow);
       idx < e; ++idx) {
    auto basis = maybeQuery->layout.getBasis(kRow, idx);
    if (llvm::all_of(basis, [](int32_t value) { return value == 0; }))
      return false;
  }
  return true;
}

bool isM64SplitNDescriptorType(MemDescType memTy, unsigned numWarps,
                               bool allow16Bit) {
  if (!memTy || numWarps != 4 || memTy.getRank() != 2 ||
      memTy.getShape()[0] != 64 ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
    return false;
  unsigned bitwidth = memTy.getElementTypeBitWidth();
  if (bitwidth != 32 && !(allow16Bit && bitwidth == 16))
    return false;

  std::string layoutError;
  auto maybeQuery =
      getTMemViewAnalysisLayout(memTy.getShape(), memTy.getEncoding(),
                                &layoutError);
  if (!maybeQuery)
    return false;
  auto kRow = StringAttr::get(memTy.getContext(), "row");
  if (!maybeQuery->layout.hasInDim(kRow) ||
      maybeQuery->layout.getInDimSize(kRow) != 128)
    return false;
  for (unsigned idx = 0, e = maybeQuery->layout.getInDimSizeLog2(kRow);
       idx < e; ++idx) {
    auto basis = maybeQuery->layout.getBasis(kRow, idx);
    if (llvm::all_of(basis, [](int32_t value) { return value == 0; }))
      return true;
  }
  return false;
}

FailureOr<std::optional<TMemAccessAtom>>
getTMemLdStRequestedAtomForMemDesc(MemDescType memTy, StringRef atomName,
                                   unsigned numWarps) {
  auto maybeAtom = parseTMemAccessAtomName(atomName, /*allowAuto=*/true,
                                           /*splitNAsPacked=*/true);
  if (failed(maybeAtom))
    return failure();
  if (atomName == "32x32b_splitn" &&
      !isM64SplitNDescriptorType(memTy, numWarps, /*allow16Bit=*/true))
    return std::optional<TMemAccessAtom>{TMemAccessAtom::I32x32b};
  return *maybeAtom;
}

bool isTMemAccessAtomCompatibleWithRequest(
    MemDescType queryTy, std::optional<TMemAccessAtom> desiredAtom,
    TMemAccessAtom actualAtom) {
  if (!desiredAtom || actualAtom == *desiredAtom)
    return true;
  if (*desiredAtom != TMemAccessAtom::I32x32b ||
      actualAtom != TMemAccessAtom::I16x32bx2)
    return false;

  // On rank-2 M64 f32 TMEM descriptor views, the user-facing `32x32b` request
  // names the logical direct family, not a promise that the final direct
  // realization must stay on the scalar I32x32b atom. The packed 16x32bx2
  // family is the direct realizable form for these layouts and should satisfy
  // the same request when the planner finds it.
  if (isM64SplitNDescriptorType(queryTy, /*numWarps=*/4))
    return true;

  // Tensor-memory-scales uses i8 logical elements packed into 32-bit TMEM
  // words. For scales roots and type-local scales descriptor views, explicit
  // `32x32b` names the 32-bit hardware packet family; the realizable direct
  // atom is the packed 16x32bx2 form when the layout carries the scale packing.
  return isa<TensorMemoryScalesEncodingAttr>(queryTy.getEncoding()) ||
         isTypeLocalTMemScalesDescriptorView(queryTy);
}

SmallVector<TMemAccessAtom>
getTMemLdStAtomSearchOrder(std::optional<TMemAccessAtom> desiredAtom) {
  SmallVector<TMemAccessAtom> atomOrder;
  if (desiredAtom)
    atomOrder.push_back(*desiredAtom);
  for (TMemAccessAtom atom : {TMemAccessAtom::I32x32b,
                              TMemAccessAtom::I16x256b,
                              TMemAccessAtom::I16x128b,
                              TMemAccessAtom::I16x64b,
                              TMemAccessAtom::I16x32bx2}) {
    if (!desiredAtom || atom != *desiredAtom)
      atomOrder.push_back(atom);
  }
  return atomOrder;
}

std::optional<LinearLayout>
reshapeTMemLdStRegisterLayoutToShape(const LinearLayout &layout,
                                     ArrayRef<int64_t> queryShape) {
  SmallVector<int64_t> layoutShape(layout.getOutDimSizes().begin(),
                                   layout.getOutDimSizes().end());
  if (llvm::equal(layoutShape, queryShape))
    return layout;

  if (layoutShape.size() == queryShape.size() &&
      llvm::all_of(queryShape, llvm::isPowerOf2_64)) {
    auto resized = layout;
    auto outDims = llvm::to_vector(resized.getOutDimNames());
    bool canResize = true;
    for (auto [idx, outDim] : llvm::enumerate(outDims)) {
      if (queryShape[idx] > resized.getOutDimSize(outDim)) {
        canResize = false;
        break;
      }
    }
    if (canResize) {
      for (auto [idx, outDim] : llvm::enumerate(outDims)) {
        if (queryShape[idx] < resized.getOutDimSize(outDim))
          resized = resized.resizeOutDim(outDim, queryShape[idx]);
      }
      return resized;
    }
  }

  if (product<int64_t>(layoutShape) != product<int64_t>(queryShape))
    return std::nullopt;
  auto *ctx = layout.getOutDimNames().begin()->getContext();
  return reshapeLayout(ctx, layout, queryShape);
}

std::optional<RankedTensorType> getTMemLdStFirstLegalRegisterType(
    ArrayRef<int64_t> resultShape, Type elementType, MemDescType queryTy,
    ArrayRef<DistributedEncodingTrait> layouts,
    std::optional<TMemAccessAtom> desiredAtom, int maxnreg,
    const TMemLdStQueryLayout *queryLayout,
    std::optional<TMemLdStRowPlan> rowPlanOverride) {
  for (DistributedEncodingTrait candidateLayout : layouts) {
    auto regTy = RankedTensorType::get(resultShape, elementType, candidateLayout);
    FailureOr<TMemLdStEncodingInfo> maybeInfo =
        queryLayout ? computeTMemLdStEncodingInfo(
                          regTy, queryTy, *queryLayout, maxnreg,
                          /*emitError=*/{}, rowPlanOverride)
                    : computeTMemLdStEncodingInfo(
                          regTy, queryTy, maxnreg, /*emitError=*/{},
                          rowPlanOverride);
    if (succeeded(maybeInfo) &&
        isTMemAccessAtomCompatibleWithRequest(queryTy, desiredAtom,
                                              maybeInfo->atom)) {
      return regTy;
    }
  }
  return std::nullopt;
}

SmallVector<TMemLdStCandidateLayout>
getTMemLdStCandidateLayoutsForQuery(MemDescType queryTy, unsigned numWarps,
                                    StringRef) {
  SmallVector<TMemLdStCandidateLayout> candidates;
  auto rowPlan = getTMemLdStRowPlanForType(queryTy);
  if (!rowPlan)
    return candidates;

  for (TMemAccessAtom atom : getTMemLdStAtomSearchOrder(std::nullopt)) {
    auto layout = getDistributedLayoutForTmemLdSt(queryTy, atom, numWarps,
                                                 rowPlan);
    if (layout)
      candidates.push_back(TMemLdStCandidateLayout{atom, std::move(*layout)});
  }
  return candidates;
}

SmallVector<DistributedEncodingTrait>
getTMemLdStGenericCompatibleLayouts(MemDescType queryTy, unsigned numWarps,
                                    StringRef) {
  SmallVector<DistributedEncodingTrait> layouts;
  for (DistributedEncodingTrait layout :
       getTmemCompatibleLayouts(queryTy, numWarps)) {
    layouts.push_back(layout);
  }
  return layouts;
}

SmallVector<DistributedEncodingTrait>
getTMemLdStBlockedFallbackLayouts(MemDescType queryTy,
                                  ArrayRef<int64_t> tensorShape,
                                  unsigned numWarps) {
  SmallVector<DistributedEncodingTrait> layouts;
  auto rank = tensorShape.size();
  auto cga = getCGALayout(queryTy.getEncoding());
  auto numCTAs = getNumCTAs(queryTy.getEncoding());
  if (rank == 2 && tensorShape[1] >= 32) {
    layouts.push_back(BlockedEncodingAttr::get(
        queryTy.getContext(), /*sizePerThread=*/SmallVector<unsigned>{1, 1},
        /*threadsPerWarp=*/SmallVector<unsigned>{1, 32},
        /*warpsPerCTA=*/SmallVector<unsigned>{numWarps, 1},
        /*order=*/SmallVector<unsigned>{1, 0}, cga));
  }
  layouts.push_back(getDefaultBlockedEncoding(
      queryTy.getContext(), tensorShape, /*numWarps=*/numWarps,
      /*threadsPerWarp=*/32, /*numCTAs=*/numCTAs));
  return layouts;
}

bool shouldTryCanonicalTMemLdStLayoutForM64DirectAtom(MemDescType memTy,
                                                      unsigned numWarps,
                                                      TMemAccessAtom atom) {
  if (atom != TMemAccessAtom::I16x32bx2 &&
      atom != TMemAccessAtom::I32x32b)
    return false;
  return isM64SplitNDescriptorType(memTy, numWarps);
}

bool shouldPreferCanonicalTMemLdStI32x32bForAuto(MemDescType memTy,
                                                 StringRef atomName) {
  if (atomName != "auto" || !memTy ||
      !isTensorMemoryEncoding(memTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
    return false;
  }
  return !(memTy.getRank() == 2 && memTy.getShape()[0] == 64);
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

static void setTMemSubsliceNonAffineWindowError(std::string *error,
                                                int64_t dim, int64_t offset,
                                                int64_t size,
                                                int64_t step) {
  if (!error)
    return;
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "unsupported tensor memory memdesc_subslice view: subview dimension "
     << dim << " with offset " << offset << " and size " << size
     << " is not affine in the selected tensor-memory linear layout; basis "
        "step "
     << step
     << " crosses a physical layout boundary and would require a "
        "carry-dependent descriptor view";
  *error = os.str();
}

static LogicalResult verifyTMemIndexProjection(
    const LinearLayout &srcInv, ArrayRef<StringAttr> srcLogicalDims,
    const LinearLayout &dstLayout, ArrayRef<int64_t> dstShape,
    std::string *error) {
  auto maybeDstInv = computeLeftInverseLayout(dstLayout, error);
  if (failed(maybeDstInv)) {
    // Descriptor index views can retain zero physical bases in the result layout
    // to encode that the surviving logical dimension starts at a larger physical
    // step. For example, indexing away an fp16 subword selector leaves logical
    // columns striding by two element columns. Prove the active image is
    // injective after removing those semantic zero bases, then use the original
    // pseudoinverse so the verification still compares in the uncompressed TMEM
    // row/column coordinate system.
    auto normalized = normalizeTensorMemoryLinearLayoutForAnalysis(dstLayout);
    std::string normalizedError;
    auto normalizedInv = computeLeftInverseLayout(normalized, &normalizedError);
    if (succeeded(normalizedInv) &&
        normalized.getNumOutDims() == dstLayout.getNumOutDims() &&
        llvm::equal(normalized.getOutDimSizes(), dstShape) &&
        static_cast<int64_t>(normalized.getTotalInDimSize()) ==
            product<int64_t>(dstShape)) {
      maybeDstInv = dstLayout.pseudoinvert();
      if (error)
        error->clear();
    } else {
      if (error && error->empty())
        *error = "unsupported tensor memory memdesc_index view";
      return failure();
    }
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

static int32_t lookupTMemOrigin(ArrayRef<StringAttr> dims,
                                ArrayRef<int32_t> origin, StringAttr dim) {
  auto it = llvm::find(dims, dim);
  if (it == dims.end())
    return 0;
  return origin[std::distance(dims.begin(), it)];
}

static int32_t lookupTMemLdStQueryOrigin(const TMemLdStQueryLayout &query,
                                         StringAttr dim) {
  return lookupTMemOrigin(llvm::to_vector(query.layout.getInDimNames()),
                          query.origin, dim);
}

static uint32_t getTMemOriginBaseOffset(const LinearLayout &layout,
                                        ArrayRef<int32_t> origin,
                                        int bitwidth) {
  auto inDims = llvm::to_vector(layout.getInDimNames());
  if (inDims.empty())
    return 0;
  auto *ctx = inDims.front().getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  uint32_t offset = 0;
  int32_t row = lookupTMemOrigin(inDims, origin, kRow);
  if (row > 0)
    offset += getTMemPackedOffsetRowBase(static_cast<uint32_t>(row));
  int32_t col = lookupTMemOrigin(inDims, origin, kCol);
  if (col > 0)
    offset += getTMemWordColumn(static_cast<uint32_t>(col), bitwidth);
  return offset;
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
    value += lookupTMemOrigin(srcInDims, srcQuery.origin, dim);
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
    int32_t value = lookupTMemOrigin(queryInDims, query.origin, dim);
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

  int32_t srcCol = lookupTMemOrigin(srcInDims, srcQuery.origin, kCol);
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
                                bool preserveInteriorZeroBases,
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
    unsigned trailingStart = basesIt->second.size();
    SmallVector<unsigned> eraseIndices;
    for (unsigned idx = basisIndices.size(); idx > 0; --idx) {
      unsigned basisIdx = basisIndices[idx - 1];
      if (basisIdx + 1 == trailingStart) {
        eraseIndices.push_back(basisIdx);
        trailingStart = basisIdx;
        continue;
      }
      if (!preserveInteriorZeroBases) {
        if (error)
          *error = "unsupported tensor memory leading-unit memdesc_subslice view";
        return std::nullopt;
      }
    }
    for (unsigned basisIdx : eraseIndices) {
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
          offsets.drop_front(extraRank), srcQuery.origin,
          /*preserveInteriorZeroBases=*/false, error)) {
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
    auto fastPathLayout = ll;
    if (fastPathLayout.getInDimSize(kCol) != dstShape[1])
      fastPathLayout = fastPathLayout.resizeInDim(kCol, dstShape[1]);

    bool retainedBasesFitNarrowPhysicalSpan = true;
    if (fastPathLayout.getOutDimSize(logicalDims[1]) != dstShape[1]) {
      unsigned colOutIdx = fastPathLayout.getOutDimIndex(logicalDims[1]);
      for (StringAttr inDim : fastPathLayout.getInDimNames()) {
        for (unsigned basisIdx = 0,
                      e = fastPathLayout.getInDimSizeLog2(inDim);
             basisIdx < e; ++basisIdx) {
          ArrayRef<int32_t> basis = fastPathLayout.getBasis(inDim, basisIdx);
          if (basis[colOutIdx] >= dstShape[1])
            retainedBasesFitNarrowPhysicalSpan = false;
        }
      }
    }

    if (retainedBasesFitNarrowPhysicalSpan) {
      auto narrowedLayout = fastPathLayout;
      if (narrowedLayout.getOutDimSize(logicalDims[1]) != dstShape[1])
        narrowedLayout =
            narrowedLayout.resizeOutDim(logicalDims[1], dstShape[1]);
      return TMemLdStQueryLayout{
          narrowedLayout, srcQuery.twoCTAs,
          remapTMemLdStQueryOrigin(
              srcQuery, narrowedLayout,
              {{kCol, static_cast<int32_t>(offsets[1])}})};
    }
  }

  // Some column subviews, such as a half-tile slice of a tile-permuted MMAv5
  // accumulator, keep an active logical column span that maps to a wider
  // physical TMEM column image. Let the exact inverse/projection path build the
  // self-contained relative layout instead of forcing the physical out-dim to
  // the logical view size.

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
          setTMemSubsliceNonAffineWindowError(error, dim, offsets[dim],
                                              dstDimSize, step);
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

  if (static_cast<size_t>(layoutRank) == srcShape.size() &&
      dstShape.size() + 1 == srcShape.size() &&
      srcShape.drop_front() == dstShape) {
    // A leading index over a dimension that is explicitly present in the TMEM
    // linear layout is also materialized by MemDescIndexOpConversion as a base
    // advance.  The query layout should therefore project away that indexed
    // logical dimension and erase the physical bases that only selected among
    // the outer buffers, without adding the constant index to the query origin.
    SmallVector<int64_t> unitDstShape(srcShape.begin(), srcShape.end());
    unitDstShape.front() = 1;
    SmallVector<int32_t> zeroOffsets(srcShape.size(), 0);
    std::string localError;
    if (auto leadingUnit = tryMakeLeadingUnitSubviewLayout(
            ll, srcShape, unitDstShape, zeroOffsets, srcQuery.origin,
            /*preserveInteriorZeroBases=*/true, &localError)) {
      auto outDimNames =
          standardOutDimNames(ctx, leadingUnit->layout.getNumOutDims());
      SmallVector<std::pair<StringAttr, int32_t>> outDims;
      outDims.reserve(leadingUnit->layout.getNumOutDims());
      for (auto [idx, size] :
           llvm::enumerate(leadingUnit->layout.getOutDimSizes())) {
        outDims.emplace_back(outDimNames[idx], static_cast<int32_t>(size));
      }
      return TMemLdStQueryLayout{
          LinearLayout(leadingUnit->layout.getBases(), std::move(outDims),
                       leadingUnit->layout.isSurjective()),
          srcQuery.twoCTAs, std::move(leadingUnit->origin)};
    }
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
        srcPoint.push_back(std::make_pair(
            logicalDim, logicalDim == srcLogicalDims[dim]
                            ? static_cast<int32_t>(step)
                            : int32_t{0}));
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

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape,
                                  int64_t layoutRank) {
    while (!shape.empty() && shape.front() == 1 &&
           (static_cast<int64_t>(shape.size()) > layoutRank ||
            product<int64_t>(shape.drop_front()) >= layoutElems))
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape, ll.getNumOutDims());
  layoutDstShape = stripLeadingUnitDims(layoutDstShape, ll.getNumOutDims());
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
                             srcQuery.twoCTAs,
                             srcQuery.origin};
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

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape,
                                  int64_t layoutRank) {
    while (!shape.empty() && shape.front() == 1 &&
           (static_cast<int64_t>(shape.size()) > layoutRank ||
            product<int64_t>(shape.drop_front()) >= layoutElems))
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape, ll.getNumOutDims());
  layoutDstShape = stripLeadingUnitDims(layoutDstShape, ll.getNumOutDims());
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
      // Keep the original physical TMEM coordinate system for physical
      // bitcasts. The normalized layout proves that the logical image is
      // injective after inactive support bases are removed, but compacting the
      // source layout here would also compact hardware row anchors such as
      // rows 32/64 into rows 1/2. Use the original pseudoinverse so the
      // selected preimage keeps zero support bits at zero while preserving the
      // public TMEM row/column coordinate units.
      maybeSrcInv = ll.pseudoinvert();
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
  struct ReinterpretPoint {
    SmallVector<int32_t> srcPoint;
    int64_t subElementBitOffset = 0;
  };
  bool usesSubElementDst = false;
  auto mapPoint = [&](ArrayRef<int32_t> dstPoint)
      -> FailureOr<ReinterpretPoint> {
    auto linearDst = linearizeRowMajorCoordsLocal(layoutDstShape, dstPoint);
    if (failed(linearDst))
      return failure();
    int64_t srcBitOffset = *linearDst * static_cast<int64_t>(dstBitwidth);
    int64_t subElementBitOffset = srcBitOffset % srcBitwidth;
    if (srcBitOffset % srcBitwidth != 0) {
      if (dstBitwidth >= srcBitwidth || srcBitwidth % dstBitwidth != 0) {
        if (error)
          *error = "unsupported tensor memory memdesc_reinterpret view";
        return failure();
      }
      usesSubElementDst = true;
    }
    int64_t linearSrc = srcBitOffset / srcBitwidth;
    auto srcPoint = unravelRowMajorCoordsLocal(layoutSrcShape, linearSrc);
    if (failed(srcPoint))
      return failure();
    return ReinterpretPoint{std::move(*srcPoint), subElementBitOffset};
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
  auto physOutDimNames = llvm::to_vector(maybeSrcInv->getOutDimNames());
  auto convertPhysicalCoordsForDstElement =
      [&](SmallVector<std::pair<StringAttr, int32_t>> coords,
          int64_t subElementBitOffset)
      -> FailureOr<SmallVector<std::pair<StringAttr, int32_t>>> {
    if (srcBitwidth == dstBitwidth)
      return coords;
    auto kCol = StringAttr::get(ctx, "col");
    auto colIt = llvm::find_if(coords, [&](const auto &coord) {
      return coord.first == kCol;
    });
    if (colIt == coords.end())
      return coords;
    int64_t colBits =
        static_cast<int64_t>(colIt->second) * srcBitwidth +
        subElementBitOffset;
    if (colBits % dstBitwidth != 0) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return failure();
    }
    int64_t dstCol = colBits / dstBitwidth;
    if (dstCol < 0 || dstCol > std::numeric_limits<int32_t>::max()) {
      if (error)
        *error = "unsupported tensor memory memdesc_reinterpret view";
      return failure();
    }
    colIt->second = static_cast<int32_t>(dstCol);
    return coords;
  };
  auto makePhysicalCoordsForDstPoint =
      [&](const ReinterpretPoint &point)
      -> FailureOr<SmallVector<std::pair<StringAttr, int32_t>>> {
    auto srcCoords = maybeSrcInv->apply(makeLogicalCoords(point.srcPoint));
    return convertPhysicalCoordsForDstElement(
        std::move(srcCoords), point.subElementBitOffset);
  };
  auto baseCoords = makePhysicalCoordsForDstPoint(*basePoint);
  if (failed(baseCoords))
    return failure();

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
      auto pointCoords = makePhysicalCoordsForDstPoint(*srcPoint);
      if (failed(pointCoords))
        return failure();
      std::vector<int32_t> basis;
      basis.reserve(physOutDimNames.size());
      for (auto physDim : physOutDimNames) {
        int32_t delta = lookupLinearLayoutCoord(*pointCoords, physDim) -
                        lookupLinearLayoutCoord(*baseCoords, physDim);
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
  for (auto physDim : physOutDimNames) {
    int64_t dimSize = maybeSrcInv->getOutDimSize(physDim);
    if (physDim == StringAttr::get(ctx, "col") &&
        srcBitwidth != dstBitwidth) {
      int64_t dimBits = dimSize * static_cast<int64_t>(srcBitwidth);
      if (dimBits % dstBitwidth != 0 ||
          dimBits / dstBitwidth > std::numeric_limits<int32_t>::max()) {
        if (error)
          *error = "unsupported tensor memory memdesc_reinterpret view";
        return failure();
      }
      dimSize = dimBits / dstBitwidth;
    }
    activePhysOutDims.push_back({physDim, static_cast<int32_t>(dimSize)});
  }

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

static bool isFullShapeM64TensorMemoryDescriptor(MemDescType memTy) {
  return memTy && memTy.getRank() == 2 && memTy.getShape()[0] == 64 &&
         memTy.getElementTypeBitWidth() == 32 &&
         isTensorMemoryEncoding(memTy.getEncoding()) &&
         !isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()) &&
         hasFullShapeTMemTile(memTy);
}

static std::optional<SmallVector<std::vector<int32_t>>>
getPurePowerOfTwoLinearBases(const LinearLayout &layout, StringAttr inDim,
                             unsigned outDimIdx, int64_t extent) {
  if (!layout.hasInDim(inDim) || extent < 1 || !llvm::isPowerOf2_64(extent) ||
      layout.getInDimSize(inDim) != extent)
    return std::nullopt;

  SmallVector<std::vector<int32_t>> bases;
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
    bases.push_back(std::vector<int32_t>(basis.begin(), basis.end()));
  }

  SmallVector<int32_t> sortedValues;
  sortedValues.reserve(bases.size());
  for (ArrayRef<int32_t> basis : bases)
    sortedValues.push_back(basis[outDimIdx]);
  llvm::sort(sortedValues);

  SmallVector<int32_t> expected;
  for (int64_t bit = 1; bit < extent; bit <<= 1)
    expected.push_back(static_cast<int32_t>(bit));
  if (!llvm::equal(sortedValues, expected))
    return std::nullopt;
  return bases;
}

static std::optional<TMemLdStQueryLayout>
getExpandedRowFoldedTMemQueryLayout(MemDescType memTy) {
  if (!hasFullShapeTMemTile(memTy) || memTy.getElementTypeBitWidth() != 32 ||
      memTy.getRank() != 2 ||
      !isa<TensorMemoryLinearEncodingAttr>(memTy.getEncoding()))
    return std::nullopt;

  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(memTy.getEncoding());
  if (!maybeTwoCTAs)
    return std::nullopt;

  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto dims = standardOutDimNames(ctx, 2);
  LinearLayout rawLayout =
      toLinearLayout(memTy.getShape(), memTy.getEncoding());
  if (rawLayout.getNumOutDims() != 2 || !rawLayout.hasInDim(kRow) ||
      !rawLayout.hasInDim(kCol))
    return std::nullopt;
  if (!llvm::equal(rawLayout.getOutDimNames(), dims))
    return std::nullopt;

  int64_t logicalRows = rawLayout.getInDimSize(kRow);
  int64_t logicalCols = rawLayout.getInDimSize(kCol);
  if (logicalRows != 256 || logicalCols < 1 ||
      !llvm::isPowerOf2_64(logicalCols))
    return std::nullopt;

  unsigned rowOutIdx = rawLayout.getOutDimIndex(dims[0]);
  unsigned colOutIdx = rawLayout.getOutDimIndex(dims[1]);
  auto rowBases =
      getPurePowerOfTwoLinearBases(rawLayout, kRow, rowOutIdx, logicalRows);
  auto colBases =
      getPurePowerOfTwoLinearBases(rawLayout, kCol, colOutIdx, logicalCols);
  if (!rowBases || !colBases)
    return std::nullopt;

  LinearLayout::BasesT bases;
  bases[kRow] = {};
  bases[kCol] = {};
  for (const std::vector<int32_t> &basis : *rowBases) {
    if (basis[rowOutIdx] < 128)
      bases[kRow].push_back(basis);
  }
  for (const std::vector<int32_t> &basis : *colBases)
    bases[kCol].push_back(basis);
  for (const std::vector<int32_t> &basis : *rowBases) {
    if (basis[rowOutIdx] >= 128)
      bases[kCol].push_back(basis);
  }

  if (bases[kRow].size() != 7 || bases[kCol].size() != colBases->size() + 1)
    return std::nullopt;

  std::string layoutError;
  auto maybeFolded = LinearLayout::tryCreate(
      std::move(bases), llvm::to_vector(rawLayout.getOutDims()),
      /*requireSurjective=*/true, &layoutError);
  if (!maybeFolded)
    return std::nullopt;

  return TMemLdStQueryLayout{
      *maybeFolded, *maybeTwoCTAs,
      SmallVector<int32_t>(maybeFolded->getNumInDims(), 0)};
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
  auto kRow = StringAttr::get(memTy.getContext(), "row");
  if (memLayout.hasInDim(kRow) && memLayout.getInDimSize(kRow) == 128) {
    constexpr std::array<int32_t, 7> canonicalRows = {1,  2,  4, 8,
                                                      0, 16, 32};
    bool canonicalM64Rows = memLayout.getInDimSizeLog2(kRow) ==
                                canonicalRows.size() &&
                            llvm::all_of(llvm::enumerate(canonicalRows),
                                         [&](auto indexed) {
                                           auto [idx, expected] = indexed;
                                           auto basis =
                                               memLayout.getBasis(kRow, idx);
                                           return basis.size() >= 2 &&
                                                  basis[0] == expected &&
                                                  basis[1] == 0;
                                         });
    auto activeRows = memLayout.removeZeroBasesAlongDim(kRow);
    if (!canonicalM64Rows && activeRows.hasInDim(kRow) &&
        activeRows.getInDimSize(kRow) == 64) {
      return TMemLdStRowPlan{/*warpRow0=*/16, /*warpRow1=*/32,
                             /*rowSpan=*/64};
    }
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
    if (isFullShapeM64TensorMemoryDescriptor(memTy))
      return makeQuery(toLinearLayout(memTy), *maybeTwoCTAs);
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

  if (auto foldedQuery = getExpandedRowFoldedTMemQueryLayout(memTy))
    return foldedQuery;

  if (auto info = getMMAv5AccumulatorLayoutInfo(memTy))
    return makeQuery(info->familyLayout, info->twoCTAs);
  if (auto info = getMMAv5ScaledAccumulatorLayoutInfo(memTy))
    return makeQuery(info->familyLayout, info->twoCTAs);
  if (auto info = getMMAv5LhsLayoutInfo(memTy))
    return makeQuery(info->familyLayout, info->twoCTAs);
  return std::nullopt;
}

static bool hasNonTrivialTMemBlockDim(MemDescType memTy) {
  auto layout = toLinearLayout(memTy);
  auto kBlock = StringAttr::get(memTy.getContext(), "block");
  return layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) > 1;
}

} // namespace

unsigned getTMemLdStReductionRepeats(const TMemLdStEncodingInfo &info) {
  unsigned elementsPerThread = getElementsPerThread(info.atom);
  return info.numRegsPerMessage / elementsPerThread;
}

static std::optional<LinearLayout>
getExactTypeTMemAddressLayout(MemDescType memTy, std::string *layoutError) {
  if (auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
          memTy.getShape(), memTy.getEncoding(), layoutError)) {
    return normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis);
  }
  if (auto maybeCanonical =
          getCanonicalTMemLinearEncoding(memTy, layoutError)) {
    return normalizeTensorMemoryLinearLayoutForAnalysis(
        maybeCanonical->getLinearLayout());
  }
  return std::nullopt;
}

static std::optional<LinearLayout>
getTypeLocalMMAv5TMemAddressLayout(MemDescType memTy) {
  std::string layoutError;
  // The MMAv5 family layout is only an instruction-shape planning layout. The
  // actual TMEM address for a tile must come from the current descriptor type so
  // tile-selector permutations and descriptor-view origins are preserved.
  if (auto maybeLayout = getExactTypeTMemAddressLayout(memTy, &layoutError))
    return *maybeLayout;

  // Keep the family layout as a conservative fallback for descriptor types that
  // still verify as MMAv5-compatible but are not yet representable as an exact
  // type-local view layout.
  if (auto maybeLayout = getMMAv5TMemFamilyAddressLayout(memTy))
    return *maybeLayout;

  return std::nullopt;
}

LinearLayout getMMAv5TMemAddressLayout(MemDescType memTy) {
  std::string layoutError;
  if (auto maybeLayout = getTypeLocalMMAv5TMemAddressLayout(memTy))
    return *maybeLayout;
  if (auto maybeLayout = getExactTypeTMemAddressLayout(memTy, &layoutError))
    return *maybeLayout;
  return toLinearLayout(memTy);
}

static std::optional<uint32_t>
getTypeLocalMMAv5TMemViewOffsetForLowering(MemDescType memTy,
                                           ArrayRef<int32_t> offsets) {
  assert(offsets.size() == memTy.getRank());

  std::string layoutError;
  if (auto maybeLayout = getExactTypeTMemAddressLayout(memTy, &layoutError)) {
    auto layoutRank = static_cast<size_t>(maybeLayout->getNumOutDims());
    auto prefixRank =
        memTy.getRank() > layoutRank ? memTy.getRank() - layoutRank : 0;
    return getTMemViewOffset(
        *maybeLayout, offsets.take_back(layoutRank),
        memTy.getElementTypeBitWidth(), memTy.getShape().take_front(prefixRank));
  }

  if (auto maybeLayout = getMMAv5TMemFamilyAddressLayout(memTy)) {
    auto layoutRank = static_cast<size_t>(maybeLayout->getNumOutDims());
    auto prefixRank =
        memTy.getRank() > layoutRank ? memTy.getRank() - layoutRank : 0;
    return getTMemViewOffset(
        *maybeLayout, offsets.take_back(layoutRank),
        memTy.getElementTypeBitWidth(), memTy.getShape().take_front(prefixRank));
  }
  return std::nullopt;
}

uint32_t getMMAv5TMemViewOffsetForLowering(MemDescType memTy,
                                           ArrayRef<int32_t> offsets) {
  assert(offsets.size() == memTy.getRank());

  if (auto maybeOffset =
          getTypeLocalMMAv5TMemViewOffsetForLowering(memTy, offsets))
    return *maybeOffset;
  return getTMemViewOffset(memTy, offsets);
}

bool isTMemLdStReductionCompatible(const TMemLdStEncodingInfo &info) {
  return !info.unpacked && getTMemLdStReductionRepeats(info) >= 2;
}

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
  if (hasNonTrivialTMemBlockDim(memTy))
    if (auto familyQuery = getFullShapeMMAv5FamilyQueryLayout(memTy))
      if (auto familyPlan = getTMemLdStRowPlan(familyQuery->layout))
        return familyPlan;

  std::string error;
  auto maybeLayout =
      getTMemViewAnalysisLinearLayout(memTy.getShape(), memTy.getEncoding(),
                                      &error);
  if (!maybeLayout)
    return std::nullopt;
  *maybeLayout = foldCanonicalSingleCTABlockRowsForAnalysis(
      std::move(*maybeLayout),
      getTensorMemoryTwoCTAs(memTy.getEncoding()).value_or(false));

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
  // Sparse/non-surjective logical M64 TMEM encodings can carry an explicit zero
  // row basis in the raw linear form. Classifying those layouts from the raw
  // row-basis count alone widens them to the 128-row family and breaks direct
  // ld/st layout selection for plain 64xN MMA accumulators as well as split-N
  // variants. For logical M64 tiles, derive the row plan from the active row
  // bases instead.
  bool shapeMatchesAllocTail =
      memTy.getAllocShape().size() >= static_cast<size_t>(memTy.getRank()) &&
      llvm::equal(memTy.getShape().take_back(2),
                  memTy.getAllocShape().take_back(2));
  if (logicalRows == 64 && activeRowBits == 6 && shapeMatchesAllocTail) {
    return planFromRowBits(activeRowBits, isZeroActiveRowBasis);
  }
  return planFromRowBits(rowBits, isZeroRowBasis);
}

bool hasCanonicalM64SplitNRows(const LinearLayout &layout) {
  if (layout.getNumInDims() == 0)
    return false;
  auto *ctx = (*layout.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  if (!layout.hasInDim(kRow) || layout.getInDimSize(kRow) != 128)
    return false;
  constexpr std::array<int32_t, 7> expectedRows = {1,  2,  4, 8,
                                                   0, 16, 32};
  if (layout.getInDimSizeLog2(kRow) != expectedRows.size())
    return false;
  for (auto [idx, expected] : llvm::enumerate(expectedRows)) {
    auto basis = layout.getBasis(kRow, idx);
    if (basis.size() != 2 || basis[0] != expected || basis[1] != 0)
      return false;
  }
  return true;
}

static bool isSimpleM64SplitNRawQueryLayout(const LinearLayout &layout,
                                            int64_t m, int64_t n) {
  if (m != 64 || n < 2 || !llvm::isPowerOf2_64(n) ||
      layout.getNumOutDims() != 2 || layout.getNumInDims() != 2)
    return false;
  auto *ctx = (*layout.getInDimNames().begin()).getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol) ||
      layout.getInDimSize(kCol) != n)
    return false;
  int64_t physicalRows = layout.getInDimSize(kRow);
  if (physicalRows != 128)
    return false;
  auto outDims = llvm::to_vector(layout.getOutDimNames());
  if (layout.getOutDimSize(outDims[0]) != m ||
      layout.getOutDimSize(outDims[1]) != n)
    return false;

  std::array<bool, 6> seenRows = {};
  unsigned zeroRows = 0;
  for (unsigned bit = 0; bit < layout.getInDimSizeLog2(kRow); ++bit) {
    auto basis = layout.getBasis(kRow, bit);
    if (basis.size() != 2 || basis[1] != 0)
      return false;
    if (basis[0] == 0) {
      ++zeroRows;
      continue;
    }
    if (basis[0] < 0 ||
        !llvm::isPowerOf2_32(static_cast<uint32_t>(basis[0])) ||
        basis[0] > 32)
      return false;
    unsigned rowBit = llvm::Log2_32(static_cast<uint32_t>(basis[0]));
    if (rowBit >= seenRows.size() || seenRows[rowBit])
      return false;
    seenRows[rowBit] = true;
  }
  if (zeroRows != 1 ||
      !llvm::all_of(seenRows, [](bool seen) { return seen; }))
    return false;

  SmallVector<bool> seenCols(layout.getInDimSizeLog2(kCol), false);
  for (unsigned bit = 0; bit < layout.getInDimSizeLog2(kCol); ++bit) {
    auto basis = layout.getBasis(kCol, bit);
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

std::optional<LinearLayout> getCanonicalM64SplitNLayoutForRawQuery(
    MemDescType memTy, const TMemLdStQueryLayout &rawQueryLayout,
    unsigned numWarps, bool allow16Bit) {
  if (!memTy || memTy.getRank() != 2 || memTy.getShape()[0] != 64 ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
    return std::nullopt;
  int bitwidth = memTy.getElementTypeBitWidth();
  if (bitwidth != 32 && !(allow16Bit && bitwidth == 16))
    return std::nullopt;
  int64_t n = memTy.getShape()[1];
  auto layout = rawQueryLayout.layout;
  auto kBlock = StringAttr::get(memTy.getContext(), "block");
  if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1)
    layout = layout.squeezeIns(kBlock);
  if (layout.hasOutDim(kBlock) && layout.getOutDimSize(kBlock) == 1)
    layout = layout.squeezeOuts(kBlock);
  if (!isSimpleM64SplitNRawQueryLayout(layout, memTy.getShape()[0], n))
    return std::nullopt;
  return getCanonicalM64SplitNLayout(memTy.getContext(), n, numWarps);
}

std::optional<LinearLayout> getCanonicalM64SplitNLayoutForRawQueryRequest(
    MemDescType memTy, const TMemLdStQueryLayout &rawQueryLayout,
    unsigned numWarps, StringRef atomName,
    std::optional<TMemAccessAtom> desiredAtom, bool allow16Bit) {
  bool requestedM64SplitN = atomName == "auto" ||
                            atomName == "32x32b_splitn" ||
                            atomName == "16x32bx2";
  if (!requestedM64SplitN ||
      (desiredAtom && *desiredAtom != TMemAccessAtom::I16x32bx2) ||
      !isM64SplitNDescriptorType(memTy, numWarps, allow16Bit))
    return std::nullopt;
  return getCanonicalM64SplitNLayoutForRawQuery(memTy, rawQueryLayout,
                                                numWarps, allow16Bit);
}

static bool shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQueryImpl(
    MemDescType memTy, std::optional<TMemLdStQueryLayout> rawQuery,
    unsigned numWarps, std::optional<TMemAccessAtom> desiredAtom) {
  if (!isM64SplitNDescriptorType(memTy, numWarps))
    return false;
  if (desiredAtom && *desiredAtom != TMemAccessAtom::I32x32b &&
      *desiredAtom != TMemAccessAtom::I16x32bx2) {
    return false;
  }
  if (!rawQuery)
    return true;

  auto kRow = StringAttr::get(memTy.getContext(), "row");
  if (!rawQuery->layout.hasInDim(kRow))
    return true;
  auto activeLayout = rawQuery->layout.removeZeroBasesAlongDim(kRow);
  return !(rawQuery->layout.getInDimSize(kRow) > memTy.getShape()[0] &&
           activeLayout.hasInDim(kRow) &&
           activeLayout.getInDimSize(kRow) == memTy.getShape()[0]);
}

bool shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery(
    MemDescType memTy, unsigned numWarps,
    std::optional<TMemAccessAtom> desiredAtom) {
  std::optional<TMemLdStQueryLayout> rawQuery;
  if (auto maybeRawQuery =
          inferTypeLocalTMemLdStQueryLayout(memTy, /*error=*/nullptr);
      succeeded(maybeRawQuery)) {
    rawQuery = *maybeRawQuery;
  }
  return shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQueryImpl(
      memTy, rawQuery, numWarps, desiredAtom);
}

static llvm::SmallVector<MemDescType>
getCurrentTMemLdStQueryTypes(MemDescType memTy, Value) {
  return getTypeLocalTMemLdStQueryTypes(memTy);
}

static FailureOr<TMemLdStQueryLayout>
inferCurrentTMemLdStQueryLayout(MemDescType memTy, Value,
                                std::string *error) {
  return inferTypeLocalTMemLdStQueryLayout(memTy, error);
}

static std::optional<TMemLdStSupportQueryPlan>
getCurrentTMemLdStSupportQueryPlan(MemDescType memTy, Value,
                                   std::string *error) {
  return getTypeLocalTMemLdStSupportQueryPlan(memTy, error);
}

static std::optional<TMemLdStRowPlan>
getCurrentTMemLdStRowPlanForQuery(MemDescType, Value, MemDescType queryTy) {
  return getTMemLdStRowPlanForType(queryTy);
}

static std::optional<TMemLdStRowPlan>
getCurrentTMemLdStRowPlanForRawQuery(MemDescType currentTy, Value,
                                     const TMemLdStQueryLayout &query) {
  if (auto rowPlan = getTMemLdStRowPlanForType(currentTy))
    return rowPlan;
  return getTMemLdStRowPlan(query.layout);
}

static std::optional<TMemLdStRowPlan>
getCurrentTMemLdStRowPlanForSupportQuery(
    MemDescType currentTy, Value, const TMemLdStSupportQueryPlan &supportPlan) {
  return getTMemLdStRowPlanForSupportQuery(currentTy, supportPlan);
}

FailureOr<TMemLdStEncodingInfo> computeTMemLoadReductionEncodingInfo(
    RankedTensorType regTy, MemDescType memTy, int maxnreg,
    std::function<InFlightDiagnostic()> emitError) {
  if (!memTy)
    return failure();

  auto isReductionCompatible = [](FailureOr<TMemLdStEncodingInfo> info) {
    return succeeded(info) && isTMemLdStReductionCompatible(*info);
  };
  auto tryEncoding = [&](MemDescType queryTy,
                         std::optional<TMemLdStRowPlan> rowPlan)
      -> FailureOr<TMemLdStEncodingInfo> {
    auto maybeInfo = computeTMemLdStEncodingInfo(regTy, queryTy, maxnreg,
                                                 emitError, rowPlan);
    if (isReductionCompatible(maybeInfo))
      return maybeInfo;
    return failure();
  };
  auto tryQueryEncoding = [&](const TMemLdStQueryLayout &query,
                              std::optional<TMemLdStRowPlan> rowPlan)
      -> FailureOr<TMemLdStEncodingInfo> {
    auto maybeInfo = computeTMemLdStEncodingInfo(regTy, memTy, query, maxnreg,
                                                 emitError, rowPlan);
    if (isReductionCompatible(maybeInfo))
      return maybeInfo;
    return failure();
  };

  if (isReductionFriendlyTmemSourceLayout(memTy)) {
    auto rowPlan = getCurrentTMemLdStRowPlanForQuery(memTy, Value{}, memTy);
    if (auto maybeInfo = tryEncoding(memTy, rowPlan); succeeded(maybeInfo))
      return maybeInfo;
  }

  auto queryTypes = getCurrentTMemLdStQueryTypes(memTy, Value{});
  for (MemDescType queryTy : queryTypes) {
    if (!isReductionFriendlyTmemSourceLayout(queryTy))
      continue;
    auto rowPlan = getCurrentTMemLdStRowPlanForQuery(memTy, Value{}, queryTy);
    if (auto maybeInfo = tryEncoding(queryTy, rowPlan); succeeded(maybeInfo))
      return maybeInfo;
  }

  std::string supportError;
  if (auto supportPlan = getCurrentTMemLdStSupportQueryPlan(memTy, Value{},
                                                            &supportError)) {
    auto rowPlan = getCurrentTMemLdStRowPlanForSupportQuery(memTy, Value{},
                                                            *supportPlan);
    if (auto maybeInfo = tryQueryEncoding(supportPlan->query, rowPlan);
        succeeded(maybeInfo)) {
      return maybeInfo;
    }
  }

  std::string rawError;
  if (auto rawQuery = inferCurrentTMemLdStQueryLayout(memTy, Value{},
                                                      &rawError);
      succeeded(rawQuery)) {
    auto rowPlan = getCurrentTMemLdStRowPlanForRawQuery(memTy, Value{},
                                                        *rawQuery);
    if (auto maybeInfo = tryQueryEncoding(*rawQuery, rowPlan);
        succeeded(maybeInfo)) {
      return maybeInfo;
    }
  }

  for (MemDescType queryTy : queryTypes) {
    if (!isReductionFriendlyTmemSourceLayout(queryTy))
      continue;
    auto rowPlan = getCurrentTMemLdStRowPlanForQuery(memTy, Value{}, queryTy);
    if (auto maybeInfo = tryEncoding(queryTy, rowPlan); succeeded(maybeInfo))
      return maybeInfo;
  }
  return failure();
}

static std::optional<gpu::DistributedEncodingTrait>
getTMemLoadReductionLayoutForMemDescImpl(MemDescType memDescTy,
                                         unsigned numWarps) {
  if (!memDescTy || numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
    return std::nullopt;
  if (!isReductionFriendlyTmemSourceLayout(memDescTy) &&
      !hasTypeLocalTMemLdStLayout(memDescTy))
    return std::nullopt;

  auto shape = llvm::to_vector(memDescTy.getShape());
  auto elementType = memDescTy.getElementType();
  auto tensorTy = RankedTensorType::get(shape, elementType);
  auto *ctx = memDescTy.getContext();

  std::optional<TMemLdStQueryLayout> rawQueryLayout;
  std::string rawError;
  if (auto maybeRawQuery = inferTypeLocalTMemLdStQueryLayout(memDescTy,
                                                             &rawError);
      succeeded(maybeRawQuery)) {
    rawQueryLayout = *maybeRawQuery;
  }

  auto tryRawQueryCompatibleM64Layout =
      [&]() -> std::optional<gpu::DistributedEncodingTrait> {
    if (!rawQueryLayout || memDescTy.getRank() != 2 ||
        memDescTy.getShape()[0] != 64 ||
        memDescTy.getElementTypeBitWidth() != 32 ||
        isa<TensorMemoryScalesEncodingAttr>(memDescTy.getEncoding()))
      return std::nullopt;
    if (hasCanonicalM64SplitNRows(rawQueryLayout->layout))
      return std::nullopt;

    auto canonical = getCanonicalM64SplitNLayoutForRawQuery(
        memDescTy, *rawQueryLayout, numWarps);
    if (!canonical)
      return std::nullopt;
    auto attr = LinearEncodingAttr::get(ctx, std::move(*canonical));
    auto regTy = RankedTensorType::get(shape, elementType, attr);
    if (!isReductionFriendlyTmemLoadLayout(tensorTy, toLinearLayout(regTy)))
      return std::nullopt;
    if (succeeded(computeTMemLoadReductionEncodingInfo(
            regTy, memDescTy, /*maxnreg=*/256)))
      return attr;
    return std::nullopt;
  };

  auto tryLayoutWithQuery =
      [&](gpu::DistributedEncodingTrait layout,
          const TMemLdStQueryLayout &queryLayout,
          std::optional<TMemLdStRowPlan> rowPlan)
      -> std::optional<gpu::DistributedEncodingTrait> {
    auto regTy = RankedTensorType::get(shape, elementType, layout);
    if (!isReductionFriendlyTmemLoadLayout(tensorTy, toLinearLayout(regTy)))
      return std::nullopt;
    if (succeeded(computeTMemLdStEncodingInfo(
            regTy, memDescTy, queryLayout, /*maxnreg=*/256,
            /*emitError=*/{}, rowPlan)) &&
        succeeded(computeTMemLoadReductionEncodingInfo(
            regTy, memDescTy, /*maxnreg=*/256))) {
      return layout;
    }
    return std::nullopt;
  };

  auto trySupportReductionLayout =
      [&]() -> std::optional<gpu::DistributedEncodingTrait> {
    std::string supportError;
    auto supportPlan = getTypeLocalTMemLdStSupportQueryPlan(memDescTy,
                                                            &supportError);
    if (!supportPlan)
      return std::nullopt;

    auto supportRowPlan =
        getTMemLdStRowPlanForSupportQuery(memDescTy, *supportPlan);

    SmallVector<gpu::DistributedEncodingTrait> layouts;
    auto addLinearLayout = [&](std::optional<LinearLayout> layout) {
      if (!layout)
        return;
      auto attr = LinearEncodingAttr::get(ctx, std::move(*layout));
      if (llvm::none_of(layouts, [&](gpu::DistributedEncodingTrait existing) {
            return cast<Attribute>(existing) == cast<Attribute>(attr);
          })) {
        layouts.push_back(attr);
      }
    };

    for (TMemAccessAtom atom : getTMemLdStAtomSearchOrder(std::nullopt)) {
      addLinearLayout(getDistributedLayoutForTmemLdSt(
          memDescTy, atom, numWarps, supportRowPlan, supportPlan->query.layout));
    }
    if (auto splitLongM =
            getTmemLoadLayoutSplitLongM(tensorTy, memDescTy, numWarps)) {
      if (llvm::none_of(layouts, [&](gpu::DistributedEncodingTrait existing) {
            return cast<Attribute>(existing) == cast<Attribute>(*splitLongM);
          })) {
        layouts.push_back(*splitLongM);
      }
    }

    for (gpu::DistributedEncodingTrait layout : layouts) {
      if (auto valid =
              tryLayoutWithQuery(layout, supportPlan->query, supportRowPlan)) {
        return valid;
      }
    }
    return std::nullopt;
  };

  auto tryReductionLayout =
      [&](MemDescType queryTy) -> std::optional<gpu::DistributedEncodingTrait> {
    auto maybeLayout = getTmemLoadReductionLayout(tensorTy, queryTy, numWarps);
    if (!maybeLayout) {
      if (auto supportLayout = trySupportReductionLayout())
        return supportLayout;
      return tryRawQueryCompatibleM64Layout();
    }

    auto regTy = RankedTensorType::get(shape, elementType, *maybeLayout);
    if (!isReductionFriendlyTmemLoadLayout(tensorTy, toLinearLayout(regTy)))
      return std::nullopt;

    if (succeeded(computeTMemLoadReductionEncodingInfo(
            regTy, memDescTy, /*maxnreg=*/256))) {
      return *maybeLayout;
    }
    if (auto supportLayout = trySupportReductionLayout())
      return supportLayout;
    return tryRawQueryCompatibleM64Layout();
  };

  for (MemDescType queryTy : getTypeLocalTMemLdStQueryTypes(memDescTy)) {
    if (auto layout = tryReductionLayout(queryTy))
      return layout;
  }
  return std::nullopt;
}

std::optional<gpu::DistributedEncodingTrait>
getTMemLoadReductionLayoutForMemDesc(MemDescType memTy, unsigned numWarps) {
  return getTMemLoadReductionLayoutForMemDescImpl(memTy, numWarps);
}

static std::optional<gpu::MemDescType>
getCanonicalTMemLdStSurrogateType(gpu::MemDescType queryTy,
                                  std::optional<TMemLdStRowPlan> backingPlan,
                                  std::string *error) {
  if (!backingPlan || queryTy.getRank() != 2)
    return std::nullopt;

  auto *ctx = queryTy.getContext();
  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(queryTy.getEncoding());
  if (!maybeTwoCTAs)
    return std::nullopt;
  gpu::CGAEncodingAttr cga = gpu::getCGALayout(queryTy.getEncoding());
  bool twoCTAs = *maybeTwoCTAs;

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


static bool tmemLinearLayoutHasActiveLogicalShape(gpu::MemDescType memTy,
                                                  const LinearLayout &layout) {
  auto layoutRank =
      static_cast<size_t>(cast<LayoutEncodingTrait>(memTy.getEncoding()).getRank());
  if (layoutRank != 2 || memTy.getShape().size() < layoutRank)
    return false;
  auto activeShape = memTy.getShape().take_back(layoutRank);
  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  return layout.hasInDim(kRow) && layout.hasInDim(kCol) &&
         layout.getInDimSize(kRow) == activeShape[0] &&
         layout.getInDimSize(kCol) == activeShape[1];
}

static bool tmemLinearLayoutFitsAllocShape(gpu::MemDescType memTy,
                                           const LinearLayout &layout) {
  auto layoutRank =
      static_cast<size_t>(cast<LayoutEncodingTrait>(memTy.getEncoding()).getRank());
  if (layout.getNumOutDims() != static_cast<int>(layoutRank) ||
      memTy.getAllocShape().size() < layoutRank)
    return false;
  auto allocShape = memTy.getAllocShape().take_back(layoutRank);
  auto outDims = llvm::to_vector(layout.getOutDimNames());
  for (auto [idx, allocDim] : llvm::enumerate(allocShape)) {
    if (layout.getOutDimSize(outDims[idx]) > allocDim)
      return false;
  }
  return true;
}

static bool hasZeroBasisAlongDim(const LinearLayout &layout, StringAttr dim) {
  if (!layout.hasInDim(dim))
    return false;
  unsigned dimBits = layout.getInDimSizeLog2(dim);
  for (unsigned idx = 0; idx < dimBits; ++idx) {
    if (llvm::all_of(layout.getBasis(dim, idx),
                     [](int32_t value) { return value == 0; }))
      return true;
  }
  return false;
}

static bool isUnsupportedRowZeroM64F32ToI16PhysicalBitcast(
    gpu::MemDescType srcTy, ArrayRef<int64_t> dstShape, int64_t dstBitwidth) {
  if (!srcTy || srcTy.getRank() != 2 || srcTy.getElementTypeBitWidth() != 32 ||
      dstBitwidth != 16 || dstShape.size() != 2 || srcTy.getShape()[0] != 64 ||
      dstShape[0] != srcTy.getShape()[0] ||
      dstShape[1] != srcTy.getShape()[1] * 2 ||
      !isTensorMemoryEncoding(srcTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding())) {
    return false;
  }

  auto linear = dyn_cast<TensorMemoryLinearEncodingAttr>(srcTy.getEncoding());
  if (!linear)
    return false;

  auto *ctx = srcTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  LinearLayout layout = linear.getLinearLayout();
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol) ||
      !hasZeroBasisAlongDim(layout, kRow) ||
      hasZeroBasisAlongDim(layout, kCol))
    return false;

  LinearLayout activeRows = layout.removeZeroBasesAlongDim(kRow);
  return activeRows.hasInDim(kRow) &&
         activeRows.getInDimSize(kRow) == srcTy.getShape()[0] &&
         layout.getInDimSize(kCol) == srcTy.getShape()[1];
}

bool isTMemDescriptorSubviewType(gpu::MemDescType memTy) {
  if (!memTy || !isTensorMemoryEncoding(memTy.getEncoding()) ||
      !isa<LayoutEncodingTrait>(memTy.getEncoding())) {
    return false;
  }
  auto layoutRank =
      static_cast<size_t>(cast<LayoutEncodingTrait>(memTy.getEncoding()).getRank());
  if (memTy.getShape().size() < layoutRank ||
      memTy.getAllocShape().size() < layoutRank) {
    return false;
  }
  return memTy.getShape().take_back(layoutRank) !=
         memTy.getAllocShape().take_back(layoutRank);
}

bool hasSelfContainedTMemSubviewLayout(gpu::MemDescType memTy) {
  if (!memTy || !isTensorMemoryEncoding(memTy.getEncoding()) ||
      isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()) ||
      !isa<TensorMemoryLinearEncodingAttr>(memTy.getEncoding())) {
    return false;
  }
  if (!isTMemDescriptorSubviewType(memTy))
    return false;
  if (getCanonicalTMemLinearEncoding(memTy, /*error=*/nullptr))
    return true;

  std::string error;
  auto maybeLayout =
      getTMemViewAnalysisLinearLayout(memTy.getShape(), memTy.getEncoding(),
                                      &error);
  if (!maybeLayout)
    return false;
  return tmemLinearLayoutHasActiveLogicalShape(memTy, *maybeLayout) &&
         tmemLinearLayoutFitsAllocShape(memTy, *maybeLayout);
}

bool hasTypeLocalTMemLdStLayout(gpu::MemDescType memTy) {
  return hasSelfContainedTMemSubviewLayout(memTy) ||
         isTypeLocalTMemScalesDescriptorView(memTy);
}

static bool hasTypeLocalTMemCopyLayout(gpu::MemDescType memTy) {
  // Direct roots may still require a wider support query than their raw
  // current layout, for example dense 256-row copy roots. Keep this predicate
  // limited to descriptor classes whose current type encodes the copy image.
  return hasSelfContainedTMemSubviewLayout(memTy) ||
         isTypeLocalTMemScalesDescriptorView(memTy);
}

gpu::MemDescType getSelfContainedTMemSubviewPlanningType(gpu::MemDescType memTy) {
  if (!hasSelfContainedTMemSubviewLayout(memTy))
    return memTy;
  if (!getCanonicalTMemLinearEncoding(memTy, /*error=*/nullptr))
    return memTy;
  return gpu::MemDescType::get(memTy.getShape(), memTy.getElementType(),
                               memTy.getEncoding(), memTy.getMemorySpace(),
                               memTy.getMutableMemory(), memTy.getShape());
}

gpu::MemDescType getTMemLdStPlanningType(gpu::MemDescType memTy) {
  if (!memTy)
    return memTy;
  if (!isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
    if (auto storageTy = getMMAv5ScaleStorageType(memTy))
      return *storageTy;
  }
  if (hasSelfContainedTMemSubviewLayout(memTy))
    return getSelfContainedTMemSubviewPlanningType(memTy);
  return memTy;
}

llvm::SmallVector<gpu::MemDescType>
getTypeLocalTMemLdStQueryTypes(gpu::MemDescType memTy) {
  llvm::SmallVector<gpu::MemDescType> queryTypes;
  if (!memTy)
    return queryTypes;

  auto add = [&](gpu::MemDescType ty) {
    if (llvm::none_of(queryTypes, [&](gpu::MemDescType existing) {
          return existing == ty;
        })) {
      queryTypes.push_back(ty);
    }
  };

  if (isTypeLocalTMemScalesDescriptorView(memTy)) {
    if (auto storageType = getMMAv5ScaleStorageType(memTy))
      add(*storageType);
    add(memTy);
    return queryTypes;
  }

  if (isTMemDescriptorSubviewType(memTy) &&
      !hasSelfContainedTMemSubviewLayout(memTy))
    return queryTypes;

  bool hasCompactTypeLocalLayout =
      getCanonicalTMemLinearEncoding(memTy, /*error=*/nullptr).has_value();
  memTy = getSelfContainedTMemSubviewPlanningType(memTy);

  auto rowPlan = getTMemLdStRowPlanForType(memTy);
  if (hasCompactTypeLocalLayout) {
    if (auto surrogate =
            getCanonicalTMemLdStSurrogateType(memTy, rowPlan,
                                              /*error=*/nullptr)) {
      add(*surrogate);
    }
  }
  add(memTy);
  return queryTypes;
}

static void canonicalizeTMemLdStQueryOutDims(TMemLdStQueryLayout &query,
                                             MLIRContext *ctx) {
  auto outDimNames = standardOutDimNames(ctx, query.layout.getNumOutDims());
  SmallVector<std::pair<StringAttr, int32_t>> outDims;
  outDims.reserve(query.layout.getNumOutDims());
  for (auto [idx, size] : llvm::enumerate(query.layout.getOutDimSizes()))
    outDims.emplace_back(outDimNames[idx], static_cast<int32_t>(size));
  query.layout = LinearLayout(query.layout.getBases(), std::move(outDims),
                              query.layout.isSurjective());
}

static FailureOr<TMemLdStQueryLayout>
inferTypeLocalTMemScalesDescriptorViewLdStQueryLayout(MemDescType memTy,
                                                      std::string *error) {
  if (!isTypeLocalTMemScalesDescriptorView(memTy)) {
    if (error)
      *error = "expected tensor memory scales descriptor-view type";
    return failure();
  }
  auto maybeTwoCTAs = getTensorMemoryTwoCTAs(memTy.getEncoding());
  if (!maybeTwoCTAs) {
    if (error)
      *error = "expected tensor memory layout encoding";
    return failure();
  }

  auto rawLayout = toLinearLayout(memTy);
  TMemLdStQueryLayout query{
      rawLayout, *maybeTwoCTAs,
      SmallVector<int32_t>(rawLayout.getNumInDims(), 0)};
  canonicalizeTMemLdStQueryOutDims(query, memTy.getContext());
  return query;
}

FailureOr<TMemLdStQueryLayout>
inferTypeLocalTMemLdStQueryLayout(MemDescType memTy, std::string *error) {
  if (isTypeLocalTMemScalesDescriptorView(memTy))
    return inferTypeLocalTMemScalesDescriptorViewLdStQueryLayout(memTy, error);
  if (!memTy ||
      memTy.getMemorySpace() != TensorMemorySpaceAttr::get(memTy.getContext())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  auto encoding = memTy.getEncoding();
  if (!isTensorMemoryEncoding(encoding)) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  if (isTMemDescriptorSubviewType(memTy) &&
      !hasSelfContainedTMemSubviewLayout(memTy)) {
    if (error)
      *error = "unsupported tensor memory descriptor view: current memdesc "
               "type does not encode a self-contained ld/st layout";
    return failure();
  }

  auto maybeQuery =
      getTMemViewAnalysisLayout(memTy.getShape(), encoding, error);
  if (!maybeQuery)
    return failure();

  TMemLdStQueryLayout query{maybeQuery->layout, maybeQuery->twoCTAs,
                            SmallVector<int32_t>(
                                maybeQuery->layout.getNumInDims(), 0)};
  canonicalizeTMemLdStQueryOutDims(query, memTy.getContext());
  return query;
}

std::optional<TMemLdStSupportQueryPlan>
getTypeLocalTMemLdStSupportQueryPlan(MemDescType memTy, std::string *error) {
  MemDescType planningTy = isTypeLocalTMemScalesDescriptorView(memTy)
                               ? memTy
                               : getSelfContainedTMemSubviewPlanningType(memTy);
  FailureOr<TMemLdStQueryLayout> maybeQuery =
      inferTypeLocalTMemLdStQueryLayout(planningTy, error);
  if (failed(maybeQuery))
    return std::nullopt;

  auto rowPlan = getTMemLdStRowPlanForType(planningTy);
  if (!rowPlan)
    rowPlan = getTMemLdStRowPlan(maybeQuery->layout);
  return TMemLdStSupportQueryPlan{*maybeQuery, rowPlan};
}

std::optional<TMemLdStRowPlan>
getTMemLdStRowPlanForSupportQuery(MemDescType fallbackTy,
                                  const TMemLdStSupportQueryPlan &supportPlan) {
  if (supportPlan.rowPlan)
    return supportPlan.rowPlan;
  if (auto rowPlan = getTMemLdStRowPlanForType(fallbackTy))
    return rowPlan;
  return getTMemLdStRowPlan(supportPlan.query.layout);
}

static bool hasExactBasisSequence(
    const LinearLayout &layout, StringAttr dim,
    ArrayRef<std::array<int32_t, 2>> expected) {
  if (!layout.hasInDim(dim) ||
      layout.getInDimSizeLog2(dim) != expected.size()) {
    return false;
  }
  for (auto [idx, expectedBasis] : llvm::enumerate(expected)) {
    auto basis = layout.getBasis(dim, idx);
    if (basis.size() != 2 || basis[0] != expectedBasis[0] ||
        basis[1] != expectedBasis[1]) {
      return false;
    }
  }
  return true;
}

bool isTwoCTAScalesDescriptorViewTMemLdStQuery(MemDescType memTy,
                                               const LinearLayout &queryLayout) {
  if (!memTy || memTy.getRank() != 2 || memTy.getElementTypeBitWidth() != 8 ||
      !llvm::is_contained(ArrayRef<int64_t>{128, 256}, memTy.getShape()[0]) ||
      memTy.getShape()[1] < 4 || memTy.getShape()[1] > 128 ||
      !llvm::isPowerOf2_64(memTy.getShape()[1])) {
    return false;
  }
  auto linear =
      dyn_cast<TensorMemoryLinearEncodingAttr>(memTy.getEncoding());
  if (!linear || !linear.getTwoCTAs())
    return false;

  auto outDimSizes = llvm::to_vector(queryLayout.getOutDimSizes());
  if (outDimSizes.size() != 2 || outDimSizes[0] != memTy.getShape()[0] ||
      outDimSizes[1] != memTy.getShape()[1]) {
    return false;
  }

  auto *ctx = memTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  int32_t rows = static_cast<int32_t>(memTy.getShape()[0]);
  int32_t cols = static_cast<int32_t>(memTy.getShape()[1]);
  SmallVector<std::array<int32_t, 2>> activeRows = {
      {rows / 2, 0}, {1, 0}, {2, 0}, {4, 0}, {8, 0}};
  SmallVector<std::array<int32_t, 2>> rowsWithZeroTail = activeRows;
  rowsWithZeroTail.push_back({0, 0});
  rowsWithZeroTail.push_back({0, 0});
  SmallVector<std::array<int32_t, 2>> expectedCols = {{0, 1}, {0, 2}};
  SmallVector<std::array<int32_t, 2>> expectedBlock = {{rows / 4, 0}};
  for (int32_t rowCarry = 16; rowCarry < rows / 4; rowCarry <<= 1)
    expectedCols.push_back({rowCarry, 0});
  for (int32_t col = 4; col < cols; col <<= 1)
    expectedCols.push_back({0, col});

  bool rowsMatch = hasExactBasisSequence(queryLayout, kRow, activeRows) ||
                   hasExactBasisSequence(queryLayout, kRow, rowsWithZeroTail);
  return rowsMatch &&
         hasExactBasisSequence(queryLayout, kCol, expectedCols) &&
         hasExactBasisSequence(queryLayout, kBlock, expectedBlock);
}

std::optional<LinearLayout>
getTwoCTAScalesDescriptorViewTMemLdStLayout(MemDescType memTy,
                                            TMemAccessAtom atom,
                                            unsigned numWarps,
                                            const LinearLayout &queryLayout) {
  if (numWarps != 4 ||
      !isTwoCTAScalesDescriptorViewTMemLdStQuery(memTy, queryLayout)) {
    return std::nullopt;
  }

  auto *ctx = memTy.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto dims = standardOutDimNames(ctx, 2);
  int32_t rows = static_cast<int32_t>(memTy.getShape()[0]);
  int32_t cols = static_cast<int32_t>(memTy.getShape()[1]);

  LinearLayout::BasesT bases;
  if (atom != TMemAccessAtom::I32x32b) {
    if (atom != TMemAccessAtom::I16x32bx2 &&
        atom != TMemAccessAtom::I16x64b &&
        atom != TMemAccessAtom::I16x128b &&
        atom != TMemAccessAtom::I16x256b) {
      return std::nullopt;
    }

    if (!queryLayout.hasInDim(kRow) || !queryLayout.hasInDim(kCol))
      return std::nullopt;
    int32_t physicalRows = queryLayout.getInDimSize(kRow);
    int32_t physicalCols = queryLayout.getInDimSize(kCol);
    auto queryInDims = llvm::to_vector(queryLayout.getInDimNames());
    auto liftPhysicalBasis =
        [&](ArrayRef<int32_t> rowCol) -> std::optional<std::vector<int32_t>> {
      assert(rowCol.size() == 2);
      if (rowCol[0] < 0 || rowCol[0] >= physicalRows || rowCol[1] < 0 ||
          rowCol[1] >= physicalCols)
        return std::nullopt;
      auto logicalCoords = queryLayout.apply(makeFullLinearLayoutCoords(
          queryInDims, {{kRow, rowCol[0]}, {kCol, rowCol[1]}}));
      std::vector<int32_t> logicalBasis;
      logicalBasis.reserve(dims.size());
      for (StringAttr dim : dims)
        logicalBasis.push_back(lookupLinearLayoutCoord(logicalCoords, dim));
      return logicalBasis;
    };
    auto appendBasis = [&](StringAttr dim, ArrayRef<int32_t> rowCol) {
      auto lifted = liftPhysicalBasis(rowCol);
      if (!lifted)
        return false;
      bases[dim].push_back(std::move(*lifted));
      return true;
    };

    // The scales value type is i8, so the direct-lowering query first packs
    // four logical columns into a 32-bit physical column.  Build the pre-packed
    // physical packet layout, then lift each physical TMEM basis through the
    // exact descriptor-view query.  This is the same algebra as
    // regLayout.invertAndCompose(queryLayout), just solved in the other
    // direction for the register layout.
    if (!appendBasis(kRegister, {0, 1}) ||
        !appendBasis(kRegister, {0, 2}))
      return std::nullopt;

    int32_t prepackTileCols = 0;
    switch (atom) {
    case TMemAccessAtom::I16x32bx2:
      prepackTileCols = 4;
      // The 16x32bx2 second half must stay on lane=16.  In this
      // two-CTA scales view, physical row16 lifts to logical row8; appending
      // it again as register repetition would duplicate that row basis and
      // make the layout non-invertible.
      if (!appendBasis(kRegister, {32, 0}) ||
          !appendBasis(kLane, {1, 0}) || !appendBasis(kLane, {2, 0}) ||
          !appendBasis(kLane, {4, 0}) || !appendBasis(kLane, {8, 0}) ||
          !appendBasis(kLane, {16, 0}))
        return std::nullopt;
      break;
    case TMemAccessAtom::I16x64b:
      prepackTileCols = 8;
      if (!appendBasis(kLane, {8, 0}) || !appendBasis(kLane, {0, 4}) ||
          !appendBasis(kLane, {1, 0}) || !appendBasis(kLane, {2, 0}) ||
          !appendBasis(kLane, {4, 0}))
        return std::nullopt;
      break;
    case TMemAccessAtom::I16x128b:
      prepackTileCols = 16;
      if (!appendBasis(kRegister, {8, 0}) ||
          !appendBasis(kLane, {0, 4}) || !appendBasis(kLane, {0, 8}) ||
          !appendBasis(kLane, {1, 0}) || !appendBasis(kLane, {2, 0}) ||
          !appendBasis(kLane, {4, 0}))
        return std::nullopt;
      break;
    case TMemAccessAtom::I16x256b:
      prepackTileCols = 32;
      if (!appendBasis(kRegister, {0, 4}) ||
          !appendBasis(kRegister, {8, 0}) ||
          !appendBasis(kLane, {0, 8}) || !appendBasis(kLane, {0, 16}) ||
          !appendBasis(kLane, {1, 0}) || !appendBasis(kLane, {2, 0}) ||
          !appendBasis(kLane, {4, 0}))
        return std::nullopt;
      break;
    case TMemAccessAtom::I32x32b:
      llvm_unreachable("handled above");
    }

    for (int32_t col = prepackTileCols; col < physicalCols; col <<= 1) {
      if (!appendBasis(kRegister, {0, col}))
        return std::nullopt;
    }
    if (atom != TMemAccessAtom::I16x32bx2 && !appendBasis(kRegister, {16, 0}))
      return std::nullopt;

    bases[kWarp] = {{0, 0}, {0, 0}};
    if (queryLayout.hasInDim(kBlock))
      bases[kBlock] = queryLayout.getBases().lookup(kBlock);
    return LinearLayout(std::move(bases), {{dims[0], rows}, {dims[1], cols}},
                        /*requireSurjective=*/false);
  }

  bases[kRegister] = {{0, 1}, {0, 2}};
  for (int32_t rowCarry = 16; rowCarry < rows / 4; rowCarry <<= 1)
    bases[kRegister].push_back({rowCarry, 0});
  for (int32_t col = 4; col < cols; col <<= 1)
    bases[kRegister].push_back({0, col});
  bases[kLane] = {{rows / 2, 0}, {1, 0}, {2, 0}, {4, 0}, {8, 0}};
  bases[kWarp] = {{0, 0}, {0, 0}};
  bases[kBlock] = {{rows / 4, 0}};
  return LinearLayout(std::move(bases), {{dims[0], rows}, {dims[1], cols}},
                      /*requireSurjective=*/false);
}

FailureOr<TMemPhysicalQuery>
inferTypeLocalTMemPhysicalQuery(MemDescType memTy, std::string *error) {
  if (!memTy ||
      memTy.getMemorySpace() != TensorMemorySpaceAttr::get(memTy.getContext())) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  auto encoding = memTy.getEncoding();
  if (!isTensorMemoryEncoding(encoding)) {
    if (error)
      *error = "expected a tensor memory descriptor";
    return failure();
  }
  if (isTMemDescriptorSubviewType(memTy) &&
      !hasTypeLocalTMemCopyLayout(memTy)) {
    if (error)
      *error = "unsupported tensor memory descriptor view for tcgen05.copy: "
               "current memdesc type does not encode a self-contained "
               "copy layout";
    return failure();
  }

  auto maybeQuery =
      getTMemViewAnalysisLayout(memTy.getShape(), encoding, error);
  if (!maybeQuery)
    return failure();

  return TMemPhysicalQuery{
      memTy,
      llvm::to_vector(memTy.getShape()),
      llvm::to_vector(memTy.getAllocShape()),
      static_cast<unsigned>(memTy.getElementTypeBitWidth()),
      maybeQuery->layout,
      maybeQuery->twoCTAs,
      SmallVector<int32_t>(maybeQuery->layout.getNumInDims(), 0)};
}

bool canInvertAndComposeLayouts(const LinearLayout &inner,
                                const LinearLayout &outer) {
  return canInvertAndComposeSafely(inner, outer);
}

FailureOr<LinearLayout>
getTMemCopySourceConversion(const TMemPhysicalQuery &query,
                            const LinearLayout &shmemLl, std::string *error) {
  if (!canInvertAndComposeLayouts(query.layout, shmemLl)) {
    if (error) {
      *error = "unsupported tensor memory descriptor view for tcgen05.copy: "
               "the source shared-memory layout image is not contained in "
               "the selected tensor-memory descriptor view image";
    }
    return failure();
  }
  return query.layout.invertAndCompose(shmemLl);
}

static bool isPurePositiveColumnBasis(ArrayRef<int32_t> basis) {
  return basis.size() >= 2 && basis[0] == 0 && basis[1] > 0;
}

static std::optional<TMemPhysicalQuery>
getDirectTMemCopyFoldedRootQuery(const TMemPhysicalQuery &exactQuery,
                                 const LinearLayout &shmemLl) {
  if (isTMemDescriptorSubviewType(exactQuery.memTy))
    return std::nullopt;
  auto foldedQuery = getExpandedRowFoldedTMemQueryLayout(exactQuery.memTy);
  if (!foldedQuery)
    return std::nullopt;

  TMemPhysicalQuery query = exactQuery;
  query.layout = foldedQuery->layout;
  query.twoCTAs = foldedQuery->twoCTAs;
  query.origin = foldedQuery->origin;

  std::string conversionError;
  if (failed(getTMemCopySourceConversion(query, shmemLl, &conversionError)))
    return std::nullopt;
  return query;
}

static std::optional<TMemPhysicalQuery>
getDirectTMemCopyColumnCanonicalSourceQuery(const TMemPhysicalQuery &exactQuery,
                                            const LinearLayout &shmemLl) {
  if (isTMemDescriptorSubviewType(exactQuery.memTy))
    return std::nullopt;

  auto *ctx = exactQuery.memTy.getContext();
  auto kCol = StringAttr::get(ctx, "col");
  if (!exactQuery.layout.hasInDim(kCol))
    return std::nullopt;

  auto bases = exactQuery.layout.getBases();
  auto colIt = bases.find(kCol);
  if (colIt == bases.end() || colIt->second.empty())
    return std::nullopt;
  if (!llvm::all_of(colIt->second, isPurePositiveColumnBasis))
    return std::nullopt;

  auto sortedColBases = colIt->second;
  std::stable_sort(sortedColBases.begin(), sortedColBases.end(),
                   [](ArrayRef<int32_t> lhs, ArrayRef<int32_t> rhs) {
                     return lhs[1] < rhs[1];
                   });
  if (llvm::equal(sortedColBases, colIt->second))
    return std::nullopt;

  bases[kCol] = std::move(sortedColBases);
  TMemPhysicalQuery supportQuery = exactQuery;
  supportQuery.layout = LinearLayout(std::move(bases),
                                     llvm::to_vector(exactQuery.layout.getOutDims()),
                                     exactQuery.layout.isSurjective());

  std::string conversionError;
  if (failed(getTMemCopySourceConversion(supportQuery, shmemLl,
                                         &conversionError)))
    return std::nullopt;
  return supportQuery;
}

FailureOr<TMemCopyPhysicalQuerySelection>
selectTMemCopyPhysicalQuery(MemDescType memTy, const LinearLayout &shmemLl,
                            std::string *error) {
  bool debug = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  TMemCopyPhysicalQuerySelection selection;

  if (auto maybeTypeLocal =
          inferTypeLocalTMemPhysicalQuery(memTy, &selection.typeLocalError);
      succeeded(maybeTypeLocal)) {
    selection.typeLocal = *maybeTypeLocal;
  }

  if (debug) {
    if (selection.typeLocal) {
      llvm::errs() << "[tmem-copy] candidate type-local query\n"
                   << selection.typeLocal->layout.toString() << "\n";
    } else if (!selection.typeLocalError.empty()) {
      llvm::errs() << "[tmem-copy] type-local destination query failed: "
                   << selection.typeLocalError << "\n";
    }
  }

  if (!selection.typeLocal) {
    if (error) {
      if (!selection.typeLocalError.empty())
        *error = selection.typeLocalError;
      else
        *error = "unsupported tensor memory descriptor for tcgen05.copy";
    }
    return failure();
  }

  selection.query = *selection.typeLocal;
  selection.usedTypeLocal = true;
  if (auto foldedQuery =
          getDirectTMemCopyFoldedRootQuery(*selection.typeLocal, shmemLl)) {
    selection.typeLocal = *foldedQuery;
    selection.query = std::move(*foldedQuery);
  } else if (auto sourceQuery =
                 getDirectTMemCopyColumnCanonicalSourceQuery(*selection.typeLocal,
                                                             shmemLl)) {
    selection.query = std::move(*sourceQuery);
    selection.usedTypeLocal = false;
  }

  std::string conversionError;
  if (failed(getTMemCopySourceConversion(*selection.query, shmemLl,
                                         &conversionError))) {
    if (error) {
      *error = conversionError.empty()
                   ? "unsupported tensor memory descriptor for tcgen05.copy: "
                     "the source shared-memory layout image is not contained "
                     "in the current tensor-memory descriptor layout"
                   : conversionError;
    }
    return failure();
  }

  if (debug && !selection.usedTypeLocal) {
    llvm::errs() << "[tmem-copy] using canonical source-support query; "
                    "destination addressing remains type-local\n"
                 << selection.query->layout.toString() << "\n";
  }

  return selection;
}

uint32_t
getTMemPhysicalQueryOriginBaseOffset(const TMemPhysicalQuery &query) {
  return getTMemOriginBaseOffset(query.layout, query.origin,
                                 query.elementBitWidth);
}

bool preserveTMemLdStSupportQueryBaseOffset(
    const TMemLdStQueryLayout &supportQuery) {
  return llvm::any_of(supportQuery.origin,
                      [](int32_t value) { return value != 0; });
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

  auto srcQuery = inferTypeLocalTMemLdStQueryLayout(srcTy, error);
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
  if (isUnsupportedRowZeroM64F32ToI16PhysicalBitcast(srcTy, dstShape,
                                                     dstBitwidth)) {
    if (error)
      *error = "unsupported tensor memory physical bitcast: row-zero M64 "
               "f32-to-16-bit descriptor views require a split-N packet "
               "mapping that is not directly representable by the current "
               "memdesc type-local ld/st lowering";
    return failure();
  }
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
  auto tryMakeOwnershipPreferred = [&]() {
    if (!twoCTAs || !hadBlock || !ll.hasInDim(kBlock))
      return std::optional<TensorMemoryLinearEncodingAttr>();
    auto bases = ll.getBases();
    auto blockIt = bases.find(kBlock);
    if (blockIt == bases.end())
      return std::optional<TensorMemoryLinearEncodingAttr>();

    bool changed = false;
    if (blockIt->second.empty()) {
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
        auto outDims = llvm::to_vector(ll.getOutDimNames());
        if (static_cast<size_t>(nonZeroDim) >= outDims.size())
          continue;
        int32_t dimSize = ll.getOutDimSize(outDims[nonZeroDim]);
        if (candidate[nonZeroDim] * 2 != dimSize)
          continue;
        blockIt->second.push_back(candidate.vec());
        dimIt->second.pop_back();
        changed = true;
        break;
      }
    }
    for (StringAttr dim : {kRow, kCol}) {
      auto dimIt = bases.find(dim);
      if (dimIt == bases.end() || dimIt->second.empty())
        continue;
      while (!dimIt->second.empty()) {
        ArrayRef<int32_t> candidate = dimIt->second.back();
        bool duplicatesBlock =
            !llvm::all_of(candidate,
                          [](int32_t value) { return value == 0; }) &&
            llvm::any_of(blockIt->second, [&](ArrayRef<int32_t> blockBasis) {
              return candidate == blockBasis;
            });
        if (!duplicatesBlock)
          break;
        dimIt->second.pop_back();
        changed = true;
      }
    }
    if (!changed)
      return std::optional<TensorMemoryLinearEncodingAttr>();

    auto preferred =
        LinearLayout(std::move(bases), ll.getOutDims(), ll.isSurjective());
    std::string preferredError;
    if (auto enc = tryMakeTensorMemoryLinearEncoding(
            ctx, std::move(preferred), /*twoCTAs=*/true, &preferredError)) {
      return std::optional<TensorMemoryLinearEncodingAttr>(*enc);
    }
    return std::optional<TensorMemoryLinearEncodingAttr>();
  };
  if (auto enc = tryMakeOwnershipPreferred())
    return enc;
  if (twoCTAs)
    return std::nullopt;
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
    // representable as self-contained TMEM-linear layouts. Preserve the source
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
                /*srcOrigin=*/{}, /*preserveInteriorZeroBases=*/false,
                &leadingUnitError)) {
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
  std::string error;
  auto result =
      inferTMemSubsliceEncoding(srcShape, srcEncoding, dstShape, offsets,
                                &error);
  if (succeeded(result) &&
      canMaterializeTMemViewEncoding(ctx, dstShape, srcAllocShape, *result)) {
    dstEncoding = *result;
    return success();
  }

  // Preserving the parent encoding is a compatibility fallback for views that
  // are valid pointer transformations but cannot yet be represented as an
  // active descriptor-relative TMEM-linear layout. Prefer the inferred active
  // layout whenever possible so the result type is self-contained.
  if (auto preserved = tryPreserveExactTMemViewEncoding(
          ctx, dstShape, srcAllocShape, srcEncoding, /*error=*/nullptr)) {
    dstEncoding = *preserved;
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
          offsets.drop_front(extraRank), /*srcOrigin=*/{},
          /*preserveInteriorZeroBases=*/false, error)) {
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
    auto narrowedWindowFitsDstShape = [&]() {
      std::string inverseError;
      auto maybeInv = computeLeftInverseLayout(ll, &inverseError);
      if (failed(maybeInv))
        return true;

      SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
      encodedOffsets.reserve(layoutRank);
      for (auto [dim, offset] : llvm::enumerate(offsets))
        encodedOffsets.push_back({logicalDims[dim], offset});
      auto baseCoords =
          maybeInv->apply(makeFullLinearLayoutCoords(logicalDims, encodedOffsets));
      auto physOutDimNames = llvm::to_vector(maybeInv->getOutDimNames());
      SmallVector<uint32_t> activePhysMasks(physOutDimNames.size(), 0);
      for (int64_t dim = 0; dim < static_cast<int64_t>(dstShape.size());
           ++dim) {
        for (int64_t step = 1; step < dstShape[dim]; step <<= 1) {
          auto point = encodedOffsets;
          point[dim].second += static_cast<int32_t>(step);
          auto pointCoords =
              maybeInv->apply(makeFullLinearLayoutCoords(logicalDims, point));
          for (auto [physIdx, physDim] : llvm::enumerate(physOutDimNames)) {
            int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                            lookupLinearLayoutCoord(baseCoords, physDim);
            if (delta < 0)
              return false;
            activePhysMasks[physIdx] |= static_cast<uint32_t>(delta);
          }
        }
      }
      for (auto [physIdx, mask] : llvm::enumerate(activePhysMasks)) {
        int32_t activePhysSize = 1;
        while (activePhysSize <= static_cast<int32_t>(mask))
          activePhysSize <<= 1;
        if (physIdx >= dstShape.size())
          continue;
        if (activePhysSize > dstShape[physIdx])
          return false;
      }
      return true;
    };

    auto narrowedWindowHasNonNegativePhysicalDeltas = [&]() {
      std::string inverseError;
      auto maybeInv = computeLeftInverseLayout(ll, &inverseError);
      if (failed(maybeInv))
        return false;

      SmallVector<std::pair<StringAttr, int32_t>> encodedOffsets;
      encodedOffsets.reserve(layoutRank);
      for (auto [dim, offset] : llvm::enumerate(offsets))
        encodedOffsets.push_back({logicalDims[dim], offset});
      auto baseCoords =
          maybeInv->apply(makeFullLinearLayoutCoords(logicalDims, encodedOffsets));
      auto physOutDimNames = llvm::to_vector(maybeInv->getOutDimNames());
      for (int64_t dim = 0; dim < static_cast<int64_t>(dstShape.size());
           ++dim) {
        for (int64_t step = 1; step < dstShape[dim]; step <<= 1) {
          auto point = encodedOffsets;
          point[dim].second += static_cast<int32_t>(step);
          auto pointCoords =
              maybeInv->apply(makeFullLinearLayoutCoords(logicalDims, point));
          for (auto physDim : physOutDimNames) {
            int32_t delta = lookupLinearLayoutCoord(pointCoords, physDim) -
                            lookupLinearLayoutCoord(baseCoords, physDim);
            if (delta < 0)
              return false;
          }
        }
      }
      return true;
    };

    if (narrowedWindowFitsDstShape()) {
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

    if (narrowedWindowHasNonNegativePhysicalDeltas()) {
      auto activeLayout = ll;
      if (activeLayout.getInDimSize(kCol) != dstShape[1])
        activeLayout = activeLayout.resizeInDim(kCol, dstShape[1]);
      auto result = tryMakeTMemViewEncoding(ctx, activeLayout,
                                            maybeSrcLayout->twoCTAs, error);
      if (!result)
        return failure();
      return *result;
    }
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
          setTMemSubsliceNonAffineWindowError(error, dim, offsets[dim],
                                              dstDimSize, step);
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
    activePhysOutDims.push_back(std::make_pair(physDim, activePhysSize));
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
        srcPoint.push_back(std::make_pair(
            logicalDim, logicalDim == srcLogicalDims[dim]
                            ? static_cast<int32_t>(step)
                            : int32_t{0}));
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
    activePhysOutDims.push_back(std::make_pair(physDim, activePhysSize));
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
  if (!result) {
    return failure();
  }
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

  SmallVector<int64_t> srcShape(srcTy.getShape().begin(), srcTy.getShape().end());
  SmallVector<int64_t> dstShape = llvm::to_vector(srcTy.getShape().drop_front());
  SmallVector<int64_t> dstAllocShape =
      getTMemMemDescIndexResultAllocShape(srcTy);

  Attribute dstEncoding;
  if (failed(inferTMemIndexOpEncoding(srcShape, dstShape, dstAllocShape,
                                      srcTy.getEncoding(), dstEncoding))) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_index view";
    return failure();
  }

  auto resultTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                       srcTy.getElementType(), dstEncoding,
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
  SmallVector<int64_t> srcShape(srcTy.getShape().begin(), srcTy.getShape().end());
  SmallVector<int64_t> srcAllocShape(srcTy.getAllocShape().begin(),
                                     srcTy.getAllocShape().end());
  SmallVector<int64_t> dstShapeVec(dstShape.begin(), dstShape.end());

  Attribute dstEncoding;
  if (failed(inferTMemSubsliceOpEncoding(srcShape, srcAllocShape,
                                         srcTy.getEncoding(), dstShapeVec,
                                         offsets, dstEncoding))) {
    if (error && error->empty())
      *error = "unsupported tensor memory memdesc_subslice view";
    return failure();
  }

  auto resultTy = tryCreateMemDescType(srcTy.getContext(), dstShape,
                                       srcTy.getElementType(), dstEncoding,
                                       srcTy.getMemorySpace(),
                                       srcTy.getMutableMemory(), srcAllocShape,
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

  auto stripLeadingUnitDims = [&](ArrayRef<int64_t> shape,
                                  int64_t layoutRank) {
    while (!shape.empty() && shape.front() == 1 &&
           (static_cast<int64_t>(shape.size()) > layoutRank ||
            product<int64_t>(shape.drop_front()) >= layoutElems))
      shape = shape.drop_front();
    return shape;
  };
  layoutSrcShape = stripLeadingUnitDims(layoutSrcShape,
                                        maybeSrcLayout->layout.getNumOutDims());
  layoutDstShape = stripLeadingUnitDims(layoutDstShape,
                                        maybeSrcLayout->layout.getNumOutDims());
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
      std::optional<uint32_t> secondHalfOffset =
          packTMemRowColOffset(row, col);
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
  bool twoCTAs = getTensorMemoryTwoCTAs(memTy.getEncoding()).value_or(false);
  LinearLayout originalMemLayout =
      foldCanonicalSingleCTABlockRowsForAnalysis(
          squeezeTrivialBlock(toLinearLayout(memTy)), twoCTAs);
  memLayout = foldCanonicalSingleCTABlockRowsForAnalysis(
      squeezeTrivialBlock(std::move(memLayout)), twoCTAs);
  if (memLayout.getNumOutDims() == 0)
    return failure();
  auto hasZeroBasisAlong = [](const LinearLayout &layout, StringAttr dim) {
    return hasZeroBasisAlongDim(layout, dim);
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

  if (isUnsupportedOriginChangingTMemRowSubview(memTy)) {
    if (emitError) {
      emitError()
          << "unsupported tensor memory row-slice load/store: the current "
             "descriptor may start at a non-zero TMEM row, but direct "
             "tcgen05 load/store packets address a fixed row footprint. "
             "Represent the row selection in the tensor-memory layout, or "
             "operate on a descriptor whose current row extent matches its "
             "allocation row extent.";
    }
    return failure();
  }

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
      return static_cast<int32_t>(packTMemRowColOffset(row, col * colScale));
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
          packTMemRowColOffset(rowPlan->warpRow0, /*col=*/0);
      info->warpBaseOffset1 =
          packTMemRowColOffset(rowPlan->warpRow1, /*col=*/0);
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

  auto tryCanonicalAnchoredExactFamily = [&]()
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

    auto queryLayout =
        squeezeTrivialBlock(toLinearLayout(memTy.getShape(), memTy.getEncoding()));
    auto queryFamilyLayout = queryLayout;
    if (bitwidth < 32 && queryFamilyLayout.hasInDim(kCol))
      queryFamilyLayout = queryFamilyLayout.removeZeroBasesAlongDim(kCol);
    if (!queryLayout.hasInDim(kRow) || !queryLayout.hasInDim(kCol))
      return std::nullopt;
    if (queryLayout.hasInDim(kBlock) && queryLayout.getInDimSize(kBlock) > 1)
      return std::nullopt;
    if (queryLayout.getInDimSizeLog2(kRow) != 7)
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
          if (canonicalLayout == queryLayout) {
            matchedCanonicalLayout = canonicalLayout;
            matchedColStride = colStride;
            matchedExactCanonicalLayout = true;
            break;
          }
          auto compareLayout = canonicalLayout;
          if (bitwidth < 32 && compareLayout.hasInDim(kCol))
            compareLayout = compareLayout.removeZeroBasesAlongDim(kCol);
          if (!matchedCanonicalLayout && compareLayout == queryFamilyLayout) {
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
    if (!canInvertAndComposeSafely(regLayout, *matchedCanonicalLayout))
      return std::nullopt;
    auto canonicalCvt = regLayout.invertAndCompose(*matchedCanonicalLayout);
    canonicalCvt = squeezeTrivialBlock(std::move(canonicalCvt));
    bool canonicalHasBlockIn = canonicalCvt.hasInDim(kBlock);
    bool canonicalHasBlockOut = canonicalCvt.hasOutDim(kBlock);
    if (canonicalHasBlockIn != canonicalHasBlockOut)
      return std::nullopt;
    if (canonicalHasBlockIn) {
      auto maybeSublayout = canonicalCvt.quotient({kBlock});
      if (!maybeSublayout)
        return std::nullopt;
      canonicalCvt = *maybeSublayout;
    }
    auto canonicalCvtBases = canonicalCvt.getBases();
    canonicalCvtBases[kWarp][0] = {32, 0};
    canonicalCvtBases[kWarp][1] = {64, 0};
    canonicalCvt =
        LinearLayout(std::move(canonicalCvtBases), canonicalCvt.getOutDims(),
                     /*isSurjective=*/canonicalCvt.isSurjective());
    auto info = lowerTMemLdSt(canonicalCvt, maxnreg, bitwidth,
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
  if (auto info = tryCanonicalAnchoredExactFamily())
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
  auto hasRegisterHalfNBasis = [&]() {
    if (!regLayout.hasInDim(kReg) || regLayout.getNumOutDims() != 2 ||
        logicalCols < 2 || !llvm::isPowerOf2_64(logicalCols))
      return false;
    SmallVector<int32_t> halfNBasis(regLayout.getNumOutDims(), 0);
    halfNBasis.back() = static_cast<int32_t>(logicalCols / 2);
    for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(kReg); ++idx) {
      if (regLayout.getBasis(kReg, idx) == ArrayRef<int32_t>(halfNBasis))
        return true;
    }
    return false;
  };
  auto hasZeroRegisterBasis = [&]() {
    if (!regLayout.hasInDim(kReg))
      return false;
    for (unsigned idx = 0; idx < regLayout.getInDimSizeLog2(kReg); ++idx) {
      if (isZeroBasis(regLayout.getBasis(kReg, idx)))
        return true;
    }
    return false;
  };
  // Descriptor-view queries over scales TMEM arrive here as
  // TensorMemoryLinear, so the root scales preference path above cannot see
  // the requested 16x32bx2 split.  The lifted candidate carries a zero
  // register basis for the folded physical row32 bit and a register half-N
  // basis for packet repetition; together those distinguish it from the
  // explicit 32x32b candidate for the same view.
  bool prefersTwoCTAScalesDescriptorViewI16x32bx2 =
      bitwidth == 8 &&
      isTwoCTAScalesDescriptorViewTMemLdStQuery(memTy, memLayout) &&
      hasRegisterHalfNBasis() && hasZeroRegisterBasis();
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
                            rowPlan->rowSpan,
                            prefersCanonicalM64SplitN ||
                                prefersTwoCTAScalesDescriptorViewI16x32bx2);
  if (failed(info))
    return failure();
  auto packetRepetitionTouchesRow = [&]() {
    if (!info->reps.hasInDim(kReg) || !info->reps.hasOutDim(kRow))
      return false;
    for (unsigned idx = 0; idx < info->reps.getInDimSizeLog2(kReg); ++idx)
      if (info->reps.getBasis(kReg, idx, kRow) != 0)
        return true;
    return false;
  };
  if (info->atom == TMemAccessAtom::I32x32b &&
      info->numRegsPerMessage > getElementsPerThread(info->atom) &&
      packetRepetitionTouchesRow()) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: vectorized 32x32b "
                     "packet repetition must not advance through TMEM rows. "
                     "Use a register layout whose packet repetition stays in "
                     "columns, or scalarize the row-discontiguous view.";
    }
    return failure();
  }
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
  bool isF32M64SplitNSubview =
      bitwidth == 32 && logicalRows == 64 && physicalRows == 128 &&
      physicalCols == logicalCols && hasZeroBasisAlong(memLayout, kRow) &&
      !hasZeroBasisAlong(memLayout, kCol) &&
      info->atom == TMemAccessAtom::I16x32bx2;
  int64_t maxSplitNRegsPerMessage =
      logicalCols == 2 ? 2 : std::max<int64_t>(1, logicalCols / 4);
  if (isF32M64SplitNSubview && info->atom != TMemAccessAtom::I16x32bx2) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: M64 split-N "
                     "descriptor views must use the 16x32bx2 atom family.";
    }
    return failure();
  }
  if (isF32M64SplitNSubview &&
      info->numRegsPerMessage > maxSplitNRegsPerMessage) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: selected M64 "
                     "split-N packet is wider than the folded row layout "
                     "can address correctly. Use the canonical split-N "
                     "register layout for this descriptor.";
    }
    return failure();
  }
  auto inferCanonicalFamilyColStride = [&]() -> std::optional<unsigned> {
    if (bitwidth != 16 || memTy.getShape() != memTy.getAllocShape())
      return std::nullopt;

    auto twoCTAs = getTensorMemoryTwoCTAs(memTy);
    if (!twoCTAs)
      return std::nullopt;

    auto queryLayout =
        squeezeTrivialBlock(toLinearLayout(memTy.getShape(), memTy.getEncoding()));
    auto queryFamilyLayout = queryLayout;
    if (queryFamilyLayout.hasInDim(kCol))
      queryFamilyLayout = queryFamilyLayout.removeZeroBasesAlongDim(kCol);

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
          if (canonicalLayout == queryLayout)
            return colStride;
          auto compareLayout = canonicalLayout;
          if (compareLayout.hasInDim(kCol))
            compareLayout = compareLayout.removeZeroBasesAlongDim(kCol);
          if (!fallbackColStride && compareLayout == queryFamilyLayout)
            fallbackColStride = colStride;
        }
      }
    }
    return fallbackColStride;
  };
  auto canonicalFamilyColStride = inferCanonicalFamilyColStride();
  bool isCanonicalFamilyUnpackedFullShape =
      canonicalFamilyColStride && *canonicalFamilyColStride > 1 && bitwidth == 16 &&
      memTy.getShape() == memTy.getAllocShape();
  if (isCanonicalFamilyUnpackedFullShape)
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

  auto layoutAddressesInvalidTMemRow = [&](const LinearLayout &layout) {
    std::optional<StringAttr> physicalRowDim;
    if (layout.hasOutDim(kRow)) {
      physicalRowDim = kRow;
    } else if (layout.getNumOutDims() == 2) {
      physicalRowDim = *layout.getOutDimNames().begin();
    }
    if (!physicalRowDim)
      return false;
    for (auto inDim : layout.getInDimNames()) {
      if (!layout.hasInDim(inDim))
        continue;
      for (unsigned idx = 0; idx < layout.getInDimSizeLog2(inDim); ++idx) {
        if (layout.getBasis(inDim, idx, *physicalRowDim) >= 128)
          return true;
      }
    }
    return false;
  };
  bool hasInvalidDirectRowAddress =
      tmemPackedOffsetAddressesRow(info->baseOffset, /*rowLimit=*/128) ||
      tmemPackedOffsetAddressesRow(info->warpBaseOffset0, /*rowLimit=*/128) ||
      tmemPackedOffsetAddressesRow(info->warpBaseOffset1, /*rowLimit=*/128) ||
      (info->secondHalfOffset &&
       tmemPackedOffsetAddressesRow(*info->secondHalfOffset,
                                    /*rowLimit=*/128)) ||
      layoutAddressesInvalidTMemRow(info->reps);
  if (!hasInvalidDirectRowAddress) {
    for (int32_t packetOffset : info->packetOffsets) {
      if (tmemPackedOffsetAddressesRow(static_cast<uint32_t>(packetOffset),
                                       /*rowLimit=*/128)) {
        hasInvalidDirectRowAddress = true;
        break;
      }
    }
  }
  if (hasInvalidDirectRowAddress) {
    if (emitError) {
      emitError() << "Failed to lower TMEM load/store: selected packet "
                     "schedule addresses TMEM row >= 128. Expanded logical "
                     "row selectors must be represented through a folded "
                     "physical query or packet column offsets.";
    }
    return failure();
  }

  bool isI32RowZeroM64DirectView =
      bitwidth == 32 && logicalRows == 64 && logicalCols == physicalCols &&
      physicalRows == 128 && hasZeroBasisAlong(originalMemLayout, kRow) &&
      !hasZeroBasisAlong(originalMemLayout, kCol);
  if (isI32RowZeroM64DirectView && info->atom == TMemAccessAtom::I32x32b &&
      info->warpBaseOffset0 == getTMemPackedOffsetRowBase(32u) &&
      info->warpBaseOffset1 == getTMemPackedOffsetRowBase(64u)) {
    info->warpBaseOffset0 =
        divideTMemPackedOffsetRow(info->warpBaseOffset0, /*divisor=*/2);
    info->warpBaseOffset1 =
        divideTMemPackedOffsetRow(info->warpBaseOffset1, /*divisor=*/2);
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
      info->warpBaseOffset0 == getTMemPackedOffsetRowBase(32u) &&
      info->warpBaseOffset1 == getTMemPackedOffsetRowBase(64u)) {
    info->warpBaseOffset0 =
        divideTMemPackedOffsetRow(info->warpBaseOffset0, /*divisor=*/2);
    info->warpBaseOffset1 =
        divideTMemPackedOffsetRow(info->warpBaseOffset1, /*divisor=*/2);
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
  bool twoCTAs = getTensorMemoryTwoCTAs(memTy.getEncoding()).value_or(false);
  LinearLayout memLayout = [&]() -> LinearLayout {
    if (isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
      return squeezeTrivialBlock(toLinearLayout(memTy));
    if (isFullShapeM64TensorMemoryDescriptor(memTy))
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
      return foldCanonicalSingleCTABlockRowsForAnalysis(
          squeezeTrivialBlock(
              toLinearLayout(memTy.getShape(), memTy.getEncoding())),
          twoCTAs);
    }

    std::string analysisError;
    auto maybeAnalysisLayout = getTMemViewAnalysisLinearLayout(
        memTy.getShape(), memTy.getEncoding(), &analysisError);
    if (maybeAnalysisLayout) {
      return foldCanonicalSingleCTABlockRowsForAnalysis(
          squeezeTrivialBlock(completeTensorMemorySubviewRowBasesForAnalysis(
              memTy.getShape(), normalizeTensorMemoryLinearLayoutForAnalysis(
                                    *maybeAnalysisLayout))),
          twoCTAs);
    }

    std::string canonicalError;
    auto maybeCanonical = getCanonicalTMemLinearEncoding(memTy, &canonicalError);
    if (maybeCanonical) {
      return foldCanonicalSingleCTABlockRowsForAnalysis(
          squeezeTrivialBlock(completeTensorMemorySubviewRowBasesForAnalysis(
              memTy.getShape(), normalizeTensorMemoryLinearLayoutForAnalysis(
                                    maybeCanonical->getLinearLayout()))),
          twoCTAs);
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
  return getTMemOriginBaseOffset(query.layout, query.origin, bitwidth);
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
  return lookupTMemLdStQueryOrigin(query, kRow) == memTy.getShape()[0] &&
         lookupTMemLdStQueryOrigin(query, kCol) == 0;
}

static void adjustTMemLdStInfoForQueryLayout(
    TMemLdStEncodingInfo &info, RankedTensorType regTy, MemDescType memTy,
    const TMemLdStQueryLayout &query, int maxnreg,
    std::optional<TMemLdStRowPlan> rowPlanOverride) {
  // Query-origin translated 32x32 TMEM views can require per-register
  // immediate decomposition even when the local tile shape itself is
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

  scalarInfo->baseOffset = info.baseOffset;
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
  info->baseOffset += getTMemLdStQueryOriginBaseOffset(
      queryLayout, memTy.getElementTypeBitWidth());
  // Projected row-half support queries on 32-bit 64x64 tiles can pick the
  // 16x32bx2 family. That direct path should inherit only the lifted row
  // origin; any packed low-bit base offset would skew the selected 64-row
  // window before the per-message column immediates are applied.
  if (info->atom == TMemAccessAtom::I16x32bx2 && memTy.getRank() == 2 &&
      memTy.getElementTypeBitWidth() == 32 && memTy.getShape()[0] == 64 &&
      lookupTMemLdStQueryOrigin(queryLayout, kRow) > 0 &&
      lookupTMemLdStQueryOrigin(queryLayout, kCol) == 0) {
    info->baseOffset = getTMemPackedOffsetRowBaseOffset(info->baseOffset);
  }
  adjustTMemLdStInfoForQueryLayout(*info, regTy, memTy, queryLayout, maxnreg,
                                   rowPlanOverride);
  return info;
}

static TMemLdStEncodingInfo refineTMemLdStQueryTypeEncodingInfoImpl(
    bool viewLike, RankedTensorType regTy, MemDescType queryTy, int maxnreg,
    std::optional<TMemLdStRowPlan> rowPlanOverride, TMemLdStEncodingInfo info) {
  if (!viewLike || regTy.getRank() != 2 || regTy.getShape()[0] != 32 ||
      regTy.getShape()[1] != 32 || info.atom != TMemAccessAtom::I32x32b ||
      info.numRegsPerMessage <= 1) {
    return info;
  }

  auto scalarInfo = computeTMemLdStEncodingInfo(
      regTy, queryTy, /*maxnreg=*/std::min(maxnreg, 2), /*emitError=*/{},
      rowPlanOverride);
  if (succeeded(scalarInfo) && scalarInfo->atom == info.atom &&
      scalarInfo->numRegsPerMessage == 1) {
    return *scalarInfo;
  }

  info.numRegsPerMessage = 1;
  return info;
}

TMemLdStEncodingInfo refineTMemLdStQueryTypeEncodingInfo(
    MemDescType memTy, RankedTensorType regTy, MemDescType queryTy, int maxnreg,
    std::optional<TMemLdStRowPlan> rowPlanOverride, TMemLdStEncodingInfo info) {
  return refineTMemLdStQueryTypeEncodingInfoImpl(
      hasTypeLocalTMemLdStLayout(memTy), regTy, queryTy, maxnreg,
      rowPlanOverride, info);
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
  *maybeMemLayout = foldCanonicalSingleCTABlockRowsForAnalysis(
      completeTensorMemorySubviewRowBasesForAnalysis(
          memTy.getShape(), std::move(*maybeMemLayout)),
      getTensorMemoryTwoCTAs(memTy.getEncoding()).value_or(false));

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

static std::optional<LinearLayout>
getTMemCopy4x256RefreshDescriptorCvt(const LinearLayout &cvt, int bitwidth);

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
  if (getTMemCopy4x256RefreshDescriptorCvt(cvt, bitwidth))
    return TMemCopyAtom{4, 256, 0};
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

std::optional<std::string>
getTMemCopyAtomFailureMessage(const LinearLayout &cvt, int bitwidth) {
  auto inDims = cvt.getInDimNames();
  if (inDims.empty())
    return std::nullopt;
  auto *ctx = inDims.begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kOffset = StringAttr::get(ctx, "offset");
  if (!cvt.hasInDim(kRow) || !cvt.hasInDim(kCol) ||
      !cvt.hasOutDim(kOffset))
    return std::nullopt;

  int rows = cvt.getInDimSize(kRow);
  int cols = cvt.getInDimSize(kCol);
  int totalBits = cols * bitwidth;
  auto makeTooNarrowMessage = [&](int requiredBits, StringRef atom) {
    std::string message;
    llvm::raw_string_ostream os(message);
    os << atom << " requires at least " << requiredBits
       << " bits of logical columns in the current tensor-memory descriptor; "
       << "got " << totalBits << " bits from " << cols << " columns x "
       << bitwidth << "-bit elements. This is a hardware copy-atom boundary: "
       << "the compiler cannot borrow hidden parent columns for a narrower "
       << "current descriptor.";
    return std::string(os.str());
  };

  if (rows == 4 && totalBits < 256)
    return makeTooNarrowMessage(256, "tcgen05.copy.4x256b");
  if (rows == 128 && totalBits < 128)
    return makeTooNarrowMessage(128, "tcgen05.copy");
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
  case TMemCopySupportFailureLayer::InstructionSchedule:
    return "instruction schedule";
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

StringRef stringifyTMemCopySourceFormat(TMemCopySourceFormat sourceFormat) {
  switch (sourceFormat) {
  case TMemCopySourceFormat::None:
    return "";
  case TMemCopySourceFormat::B8x16B6x16P32:
    return "b8x16.b6x16_p32";
  case TMemCopySourceFormat::B8x16B4x16P64:
    return "b8x16.b4x16_p64";
  }
  llvm_unreachable("unknown tcgen05.copy source format");
}

static TMemCopySupportResult getSupportedTMemCopyResult() {
  return {true, TMemCopySupportFailureLayer::None, ""};
}

static TMemCopySupportResult
getUnsupportedTMemCopyResult(TMemCopySupportFailureLayer layer, Twine message) {
  return {false, layer, message.str()};
}

TMemCopySupportResult
getTMemCopySourceFormatSupport(const TMemCopyMessagePlan &message,
                               int bitwidth) {
  if (message.sourceFormat == TMemCopySourceFormat::None)
    return getSupportedTMemCopyResult();
  if (bitwidth != 8) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::IsaAtom,
        Twine("tcgen05.copy source format .") +
            stringifyTMemCopySourceFormat(message.sourceFormat) +
            " requires 8-bit source elements.");
  }
  return getSupportedTMemCopyResult();
}

static bool hasOnlyWarpx2SharedSourceDims(const LinearLayout &shmemLl,
                                          StringAttr offsetDim,
                                          StringAttr blockDim) {
  if (!shmemLl.hasInDim(offsetDim))
    return false;
  return llvm::all_of(shmemLl.getInDimNames(), [&](StringAttr dim) {
    return dim == offsetDim || dim == blockDim;
  });
}

struct TMemCopyWarpx2SharedSourceOffsetBasisMismatch {
  unsigned basisIndex = 0;
  unsigned expectedBasisCount = 0;
  unsigned actualBasisCount = 0;
  SmallVector<int32_t, 2> actualBasis;
  SmallVector<int32_t, 2> expectedBasis;
};

static std::optional<TMemCopyWarpx2SharedSourceOffsetBasisMismatch>
getWarpx2SharedSourceOffsetBasisMismatch(const LinearLayout &shmemLl,
                                         StringAttr offsetDim) {
  constexpr int32_t expectedOffsetBases[][2] = {
      {32, 0}, {0, 1}, {0, 2}, {1, 0}, {2, 0},
      {4, 0},  {8, 0}, {16, 0}, {64, 0},
  };
  auto actualOffsetBases = shmemLl.getBases().lookup(offsetDim);
  auto makeMismatch =
      [&](unsigned idx) -> TMemCopyWarpx2SharedSourceOffsetBasisMismatch {
    TMemCopyWarpx2SharedSourceOffsetBasisMismatch mismatch;
    mismatch.basisIndex = idx;
    mismatch.expectedBasisCount = std::size(expectedOffsetBases);
    mismatch.actualBasisCount = actualOffsetBases.size();
    if (idx < actualOffsetBases.size()) {
      mismatch.actualBasis.append(actualOffsetBases[idx].begin(),
                                  actualOffsetBases[idx].end());
    }
    if (idx < std::size(expectedOffsetBases)) {
      ArrayRef<int32_t> expected(expectedOffsetBases[idx]);
      mismatch.expectedBasis.append(expected.begin(), expected.end());
    }
    return mismatch;
  };

  if (actualOffsetBases.size() != std::size(expectedOffsetBases))
    return makeMismatch(std::min<unsigned>(actualOffsetBases.size(),
                                           std::size(expectedOffsetBases)));
  for (auto [idx, actual] : llvm::enumerate(actualOffsetBases)) {
    ArrayRef<int32_t> expected(expectedOffsetBases[idx]);
    if (!llvm::equal(actual, expected))
      return makeMismatch(idx);
  }
  return std::nullopt;
}

static bool hasCanonicalWarpx2SharedSourceOffsetBases(
    const LinearLayout &shmemLl, StringAttr offsetDim) {
  return !getWarpx2SharedSourceOffsetBasisMismatch(shmemLl, offsetDim);
}

enum class TMemCopyWarpx2SharedSourceRequirementKind {
  None,
  RankAndShape,
  Encoding,
  OffsetDimension,
  ExtraDimensions,
  OffsetBasisOrder,
  SingleCtaBlockBasis,
  TwoCtaBlockBasis,
};

struct TMemCopyWarpx2SharedSourceRequirement {
  TMemCopyWarpx2SharedSourceRequirementKind kind =
      TMemCopyWarpx2SharedSourceRequirementKind::None;
  TMemCopyFamily family = TMemCopyFamily::Warpx2_01_23_64x128b;
  SmallVector<int64_t, 2> sourceShape;
  std::optional<TMemCopyWarpx2SharedSourceOffsetBasisMismatch>
      offsetBasisMismatch;
};

static std::optional<TMemCopyWarpx2SharedSourceRequirement>
getTMemCopyWarpx2SharedSourceRequirement(MemDescType srcTy,
                                         TMemCopyFamily family) {
  assert((family == TMemCopyFamily::Warpx2_01_23_64x128b ||
          family == TMemCopyFamily::Warpx2_02_13_64x128b) &&
         "warpx2 shared-source requirement only applies to warpx2 copies");

  auto makeRequirement = [&](TMemCopyWarpx2SharedSourceRequirementKind kind) {
    return TMemCopyWarpx2SharedSourceRequirement{
        /*kind=*/kind,
        /*family=*/family,
        /*sourceShape=*/SmallVector<int64_t, 2>(srcTy.getShape().begin(),
                                                srcTy.getShape().end()),
        /*offsetBasisMismatch=*/std::nullopt};
  };

  if (srcTy.getRank() != 2 || srcTy.getShape()[1] != 4 ||
      srcTy.getShape()[0] < 128 || srcTy.getShape()[0] % 128 != 0 ||
      !llvm::isPowerOf2_64(srcTy.getShape()[0] / 128)) {
    return makeRequirement(
        TMemCopyWarpx2SharedSourceRequirementKind::RankAndShape);
  }
  if (!isa<triton::gpu::SharedLinearEncodingAttr>(srcTy.getEncoding())) {
    return makeRequirement(TMemCopyWarpx2SharedSourceRequirementKind::Encoding);
  }

  auto shmemLl = toLinearLayout(srcTy);
  auto *ctx = srcTy.getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!shmemLl.hasInDim(kOffset))
    return makeRequirement(
        TMemCopyWarpx2SharedSourceRequirementKind::OffsetDimension);
  if (!hasOnlyWarpx2SharedSourceDims(shmemLl, kOffset, kBlock))
    return makeRequirement(
        TMemCopyWarpx2SharedSourceRequirementKind::ExtraDimensions);

  // This is stronger than a descriptor-representability precheck. Local
  // probes showed noncanonical dense/near-canonical shared layouts can either
  // select a descriptor and still copy the wrong logical source rows/columns,
  // or fail only after source-footprint scheduling. Keep this as the current
  // source-layout contract until warpx2 planning carries the full source
  // rematerialization schedule instead of only an MMAShared descriptor.
  if (auto mismatch =
          getWarpx2SharedSourceOffsetBasisMismatch(shmemLl, kOffset)) {
    auto requirement = makeRequirement(
        TMemCopyWarpx2SharedSourceRequirementKind::OffsetBasisOrder);
    requirement.offsetBasisMismatch = std::move(mismatch);
    return requirement;
  }

  auto blockBases = shmemLl.getBases().lookup(kBlock);
  if (srcTy.getShape()[0] == 128) {
    if (!llvm::all_of(blockBases, [](ArrayRef<int32_t> basis) {
          return llvm::all_of(basis, [](int32_t v) { return v == 0; });
        })) {
      return makeRequirement(
          TMemCopyWarpx2SharedSourceRequirementKind::SingleCtaBlockBasis);
    }
    return std::nullopt;
  }

  unsigned expectedBlockBases = llvm::Log2_64(srcTy.getShape()[0] / 128);
  if (blockBases.size() != expectedBlockBases) {
    return makeRequirement(
        TMemCopyWarpx2SharedSourceRequirementKind::TwoCtaBlockBasis);
  }
  for (auto [idx, basis] : llvm::enumerate(blockBases)) {
    int32_t expectedRow = 128 << idx;
    if (!llvm::equal(basis, ArrayRef<int32_t>{expectedRow, 0})) {
      return makeRequirement(
          TMemCopyWarpx2SharedSourceRequirementKind::TwoCtaBlockBasis);
    }
  }
  return std::nullopt;
}

static std::string getTMemCopyWarpx2SharedSourceRequirementError(
    const TMemCopyWarpx2SharedSourceRequirement &requirement) {
  auto printBasis = [](llvm::raw_ostream &os, ArrayRef<int32_t> basis) {
    os << "[";
    for (auto [idx, value] : llvm::enumerate(basis)) {
      if (idx)
        os << ", ";
      os << value;
    }
    os << "]";
  };
  switch (requirement.kind) {
  case TMemCopyWarpx2SharedSourceRequirementKind::None:
    return "";
  case TMemCopyWarpx2SharedSourceRequirementKind::RankAndShape:
    return "warpx2 tcgen05.copy currently requires a 128x4 shared tile, or a "
           "128*num_ctas by 4 multi-CTA shared tile with canonical CTA block "
           "bases.";
  case TMemCopyWarpx2SharedSourceRequirementKind::Encoding:
    return "warpx2 tcgen05.copy currently requires the canonical "
           "shared-linear source layout.";
  case TMemCopyWarpx2SharedSourceRequirementKind::OffsetDimension:
    return "warpx2 tcgen05.copy shared layout has no offset dimension.";
  case TMemCopyWarpx2SharedSourceRequirementKind::ExtraDimensions:
    return "warpx2 tcgen05.copy shared layout may only use offset and block "
           "dimensions.";
  case TMemCopyWarpx2SharedSourceRequirementKind::OffsetBasisOrder: {
    std::string reason;
    llvm::raw_string_ostream os(reason);
    os << "warpx2 tcgen05.copy currently supports only the canonical 128x4 "
          "shared-linear offset basis order for source shape ";
    for (auto [idx, extent] : llvm::enumerate(requirement.sourceShape)) {
      if (idx)
        os << "x";
      os << extent;
    }
    os << ".";
    if (requirement.offsetBasisMismatch) {
      const auto &mismatch = *requirement.offsetBasisMismatch;
      os << " The first mismatch is offset basis " << mismatch.basisIndex;
      if (mismatch.actualBasisCount != mismatch.expectedBasisCount) {
        os << "; got " << mismatch.actualBasisCount
           << " offset bases but expected " << mismatch.expectedBasisCount;
      }
      if (!mismatch.actualBasis.empty()) {
        os << "; got ";
        printBasis(os, mismatch.actualBasis);
      }
      if (!mismatch.expectedBasis.empty()) {
        os << " but expected ";
        printBasis(os, mismatch.expectedBasis);
      }
      os << ".";
    }
    os << " This is a source rematerialization boundary: descriptor "
          "representability alone is not enough because the warpx2 source "
          "message schedule assigns fixed meanings to the shared offset bases. "
          "Support needs a rematerialized canonical shared source, a different "
          "source format, or a proved schedule that preserves the requested "
          "logical source footprint.";
    return os.str();
  }
  case TMemCopyWarpx2SharedSourceRequirementKind::SingleCtaBlockBasis:
    return "single-CTA warpx2 tcgen05.copy does not support a non-zero shared "
           "block basis.";
  case TMemCopyWarpx2SharedSourceRequirementKind::TwoCtaBlockBasis:
    return "multi-CTA warpx2 tcgen05.copy requires canonical shared block "
           "bases [[128, 0], [256, 0], ...].";
  }
  llvm_unreachable("unknown warpx2 shared-source requirement kind");
}

TMemCopySupportResult
getTMemCopySharedLayoutRuntimeSupport(MemDescType srcTy,
                                      TMemCopyFamily family) {
  if (auto nvmmaEnc =
          dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(srcTy.getEncoding())) {
    if (nvmmaEnc.getTransposed() || nvmmaEnc.getFp4Padded()) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::SharedLayout,
          "tcgen05.copy source should not be transposed or padded.");
    }
    bool sourceSwizzled = nvmmaEnc.getSwizzlingByteWidth() != 0;
    bool familyUsesUnswizzledSource =
        family == TMemCopyFamily::Warpx4_32x128b;
    if (familyUsesUnswizzledSource && sourceSwizzled) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::SharedLayout,
          Twine("tcgen05.copy.") + stringifyTMemCopyFamily(family) +
              " requires an unswizzled shared-memory source layout.");
    }
    if (!familyUsesUnswizzledSource && !sourceSwizzled) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::SharedLayout,
          Twine("tcgen05.copy.") + stringifyTMemCopyFamily(family) +
              " requires a swizzled shared-memory source layout.");
    }
  }

  if (family != TMemCopyFamily::Warpx2_01_23_64x128b &&
      family != TMemCopyFamily::Warpx2_02_13_64x128b)
    return getSupportedTMemCopyResult();

  if (auto requirement =
          getTMemCopyWarpx2SharedSourceRequirement(srcTy, family)) {
    return getUnsupportedTMemCopyResult(TMemCopySupportFailureLayer::SharedLayout,
                                        getTMemCopyWarpx2SharedSourceRequirementError(
                                            *requirement));
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

static bool isDenseTMemCopyFamily(TMemCopyFamily family) {
  return family == TMemCopyFamily::Dense4x256b ||
         family == TMemCopyFamily::Dense128x128b ||
         family == TMemCopyFamily::Dense128x256b;
}

static bool isAllZeroBasis(ArrayRef<int32_t> basis) {
  return llvm::all_of(basis, [](int32_t value) { return value == 0; });
}

static bool basisEquals(ArrayRef<int32_t> basis,
                        std::initializer_list<int32_t> expected) {
  return llvm::equal(basis, ArrayRef<int32_t>(expected));
}

static bool isPureOffsetBasis(const LinearLayout &layout, StringAttr dim,
                              unsigned bit, StringAttr offsetDim,
                              int32_t expectedOffset) {
  if (!layout.hasInDim(dim) || !layout.hasOutDim(offsetDim) ||
      bit >= layout.getInDimSizeLog2(dim))
    return false;
  ArrayRef<int32_t> basis = layout.getBasis(dim, bit);
  unsigned offsetIdx = layout.getOutDimIndex(offsetDim);
  for (auto [idx, value] : llvm::enumerate(basis)) {
    int32_t expected = idx == offsetIdx ? expectedOffset : 0;
    if (value != expected)
      return false;
  }
  return true;
}

static bool isTMemCopy4x256RefreshLayout(const LinearLayout &layout,
                                         MLIRContext *ctx, int bitwidth) {
  if (bitwidth != 32)
    return false;
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!layout.hasInDim(kRow) || !layout.hasInDim(kCol) ||
      layout.getNumOutDims() != 2 || layout.getInDimSize(kRow) != 128 ||
      layout.getInDimSize(kCol) != 8)
    return false;
  int64_t expectedRows = 4;
  if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) > 1) {
    if (layout.getInDimSize(kBlock) != 2 ||
        !basisEquals(layout.getBasis(kBlock, 0), {4, 0}))
      return false;
    expectedRows = 8;
  }
  auto outDims = llvm::to_vector(layout.getOutDims());
  if (outDims[0].second != expectedRows || outDims[1].second != 8)
    return false;
  for (unsigned bit = 0; bit < 5; ++bit) {
    if (!isAllZeroBasis(layout.getBasis(kRow, bit)))
      return false;
  }
  return basisEquals(layout.getBasis(kRow, 5), {0, 1}) &&
         basisEquals(layout.getBasis(kRow, 6), {0, 2}) &&
         basisEquals(layout.getBasis(kCol, 0), {1, 0}) &&
         basisEquals(layout.getBasis(kCol, 1), {2, 0}) &&
         basisEquals(layout.getBasis(kCol, 2), {0, 4});
}

bool isTMemCopy4x256RefreshLayout(gpu::MemDescType memTy) {
  if (memTy.getRank() != 2 || memTy.getElementTypeBitWidth() != 32)
    return false;

  std::string layoutError;
  auto maybeLayout = getTMemViewAnalysisLinearLayout(
      memTy.getShape(), memTy.getEncoding(), &layoutError);
  if (!maybeLayout)
    return false;

  return isTMemCopy4x256RefreshLayout(*maybeLayout, memTy.getContext(),
                                      memTy.getElementTypeBitWidth());
}

std::string getTMemCopy4x256RefreshLdStUnsupportedMessage(
    const TMemCopy4x256RefreshImageRequirement &requirement) {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "direct TMEM load/store is unsupported for the "
        "tcgen05.copy.4x256b refresh-shaped tensor memory layout. "
        "tcgen05.ld/st packets require TMEM row anchors to be materializable "
        "as warp bases, but this "
     << requirement.logicalRows << "x" << requirement.logicalColumns
     << " refresh view stores logical row bits in TMEM columns, low logical "
        "column bits in TMEM rows "
     << requirement.lowColumnRowDelta0 << "/" << requirement.lowColumnRowDelta1
     << ", and the high logical column bit at destination dword +"
     << requirement.highColumnDwordDelta
     << ". Source columns [0, " << requirement.sourceColumnSplit << ") and ["
     << requirement.sourceColumnSplit << ", " << requirement.logicalColumns
     << ") are scheduled as separate physical refresh messages. Use "
        "tcgen05_copy from shared memory for this refresh image, or access a "
        "directly supported 128-row physical layout.";
  return os.str();
}

static std::optional<LinearLayout>
getTMemCopy4x256RefreshDescriptorCvt(const LinearLayout &cvt, int bitwidth) {
  if (bitwidth != 32)
    return std::nullopt;
  auto inDims = cvt.getInDimNames();
  if (inDims.empty())
    return std::nullopt;
  auto *ctx = inDims.begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  auto kOffset = StringAttr::get(ctx, "offset");
  if (!cvt.hasInDim(kRow) || !cvt.hasInDim(kCol) ||
      !cvt.hasOutDim(kOffset) || cvt.getInDimSize(kRow) != 128 ||
      cvt.getInDimSize(kCol) != 8)
    return std::nullopt;
  for (unsigned bit = 0; bit < 5; ++bit) {
    if (!isAllZeroBasis(cvt.getBasis(kRow, bit)))
      return std::nullopt;
  }
  if (!isPureOffsetBasis(cvt, kRow, 5, kOffset, 4) ||
      !isPureOffsetBasis(cvt, kRow, 6, kOffset, 8) ||
      !isPureOffsetBasis(cvt, kCol, 0, kOffset, 1) ||
      !isPureOffsetBasis(cvt, kCol, 1, kOffset, 2) ||
      !isPureOffsetBasis(cvt, kCol, 2, kOffset, 16))
    return std::nullopt;

  LinearLayout::BasesT bases;
  bases[kRow] = {
      std::vector<int32_t>(cvt.getBasis(kCol, 0).begin(),
                           cvt.getBasis(kCol, 0).end()),
      std::vector<int32_t>(cvt.getBasis(kCol, 1).begin(),
                           cvt.getBasis(kCol, 1).end())};
  bases[kCol] = {
      std::vector<int32_t>(cvt.getBasis(kRow, 5).begin(),
                           cvt.getBasis(kRow, 5).end()),
      std::vector<int32_t>(cvt.getBasis(kRow, 6).begin(),
                           cvt.getBasis(kRow, 6).end()),
      std::vector<int32_t>(cvt.getBasis(kCol, 2).begin(),
                           cvt.getBasis(kCol, 2).end())};
  if (cvt.hasInDim(kBlock)) {
    auto blockBases = cvt.getBases().lookup(kBlock);
    bases[kBlock] =
        std::vector<std::vector<int32_t>>(blockBases.begin(), blockBases.end());
  }
  return LinearLayout(std::move(bases), llvm::to_vector(cvt.getOutDims()),
                      /*requireSurjective=*/false);
}

static unsigned getDenseTMemCopyColumnStride(TMemCopyFamily family,
                                             unsigned bitwidth) {
  switch (family) {
  case TMemCopyFamily::Dense4x256b:
  case TMemCopyFamily::Dense128x256b:
    return 256 / bitwidth;
  case TMemCopyFamily::Dense128x128b:
    return 128 / bitwidth;
  case TMemCopyFamily::Warpx2_01_23_64x128b:
  case TMemCopyFamily::Warpx2_02_13_64x128b:
  case TMemCopyFamily::Warpx4_32x128b:
    llvm_unreachable("non-dense copy family");
  }
  llvm_unreachable("unknown copy family");
}

static std::optional<std::pair<int32_t, int32_t>>
getDenseTMemCopyDestinationTileCoord(const LinearLayout &ll, MLIRContext *ctx,
                                     int32_t logicalRow, int32_t logicalCol) {
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) || ll.getNumOutDims() != 2)
    return std::nullopt;

  auto outDims = llvm::to_vector(ll.getOutDimNames());
  auto rowCol =
      ll.pseudoinvert().apply({{outDims[0], logicalRow},
                               {outDims[1], logicalCol}});
  int32_t row = 0;
  int32_t col = 0;
  for (auto [dim, value] : rowCol) {
    if (dim == kRow) {
      row = value;
      continue;
    }
    if (dim == kCol) {
      col = value;
      continue;
    }
    if (value != 0)
      return std::nullopt;
  }
  if (row < 0 || col < 0)
    return std::nullopt;
  return std::pair<int32_t, int32_t>{row, col};
}

static bool needsDenseTMemCopyPhysicalColumnTileOffsets(
    const LinearLayout &ll, MLIRContext *ctx, TMemCopyFamily family,
    unsigned bitwidth) {
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kCol))
    return false;

  unsigned instructionCols = getDenseTMemCopyColumnStride(family, bitwidth);
  // Bases below instructionCols select columns inside one copy atom; support
  // checking separately proves those columns are physically contiguous. Pure
  // column bases at or above instructionCols select which atom-width tile is
  // addressed, so non-canonical selector order requires physical tile
  // offsets. Row-touching column bases are folded-row selectors carried by
  // the descriptor/source projection, not destination column tile selectors.
  bool sawColumnTileSelector = false;
  int32_t previousColumnTileSelector = 0;
  for (ArrayRef<int32_t> basis : ll.getBases().lookup(kCol)) {
    bool touchesRow = basis[0] != 0;
    bool touchesCol = basis[1] != 0;
    if (!touchesCol || touchesRow)
      continue;

    int32_t colBasis = std::abs(basis[1]);
    if (colBasis >= static_cast<int32_t>(instructionCols)) {
      if (sawColumnTileSelector && colBasis <= previousColumnTileSelector)
        return true;
      sawColumnTileSelector = true;
      previousColumnTileSelector = colBasis;
      continue;
    }
    if (sawColumnTileSelector)
      return true;
  }
  return false;
}

std::optional<uint32_t>
getTMemCopyDestinationTileOffset(const TMemPhysicalQuery &query,
                                 TMemCopyFamily family, int32_t logicalCol) {
  if (family == TMemCopyFamily::Dense4x256b &&
      isTMemCopy4x256RefreshLayout(query.layout, query.memTy.getContext(),
                                   query.elementBitWidth))
    return 0u;
  if (!isDenseTMemCopyFamily(family))
    return getTMemWordColumn(static_cast<uint32_t>(logicalCol),
                             query.elementBitWidth);

  const LinearLayout &ll = query.layout;
  bool needsPhysicalTileOffset = needsDenseTMemCopyPhysicalColumnTileOffsets(
      ll, query.memTy.getContext(), family, query.elementBitWidth);
  if (!needsPhysicalTileOffset)
    return getTMemWordColumn(static_cast<uint32_t>(logicalCol),
                             query.elementBitWidth);

  auto coord = getDenseTMemCopyDestinationTileCoord(
      ll, query.memTy.getContext(), 0, logicalCol);
  if (!coord)
    return std::nullopt;
  return packTMemRowColOffset(
      static_cast<uint32_t>(coord->first),
      getTMemWordColumn(static_cast<uint32_t>(coord->second),
                        query.elementBitWidth));
}

static std::optional<TMemCopyDestinationFootprint>
getTMemCopyDestinationFootprint(const TMemPhysicalQuery &query,
                                TMemCopyFamily family, int32_t logicalCol,
                                unsigned rows, unsigned columns) {
  int32_t physicalRow = 0;
  int32_t physicalCol = logicalCol;
  const LinearLayout &ll = query.layout;
  if (family == TMemCopyFamily::Dense4x256b &&
      isTMemCopy4x256RefreshLayout(query.layout, query.memTy.getContext(),
                                   query.elementBitWidth)) {
    physicalCol = 0;
  } else if (isDenseTMemCopyFamily(family) &&
             needsDenseTMemCopyPhysicalColumnTileOffsets(
                 ll, query.memTy.getContext(), family,
                 query.elementBitWidth)) {
    auto coord = getDenseTMemCopyDestinationTileCoord(
        ll, query.memTy.getContext(), 0, logicalCol);
    if (!coord)
      return std::nullopt;
    physicalRow = coord->first;
    physicalCol = coord->second;
  }

  auto offset = getTMemCopyDestinationTileOffset(query, family, logicalCol);
  if (!offset)
    return std::nullopt;
  return TMemCopyDestinationFootprint{
      /*logicalRow=*/0,
      /*logicalCol=*/logicalCol,
      /*physicalRow=*/physicalRow,
      /*physicalCol=*/physicalCol,
      /*rows=*/rows,
      /*columns=*/columns,
      /*offset=*/static_cast<uint32_t>(*offset)};
}

std::optional<llvm::SmallVector<TMemCopyScheduledTile>>
getTMemCopyScheduledTilePlan(const TMemPhysicalQuery &query,
                             TMemCopyFamily family, unsigned rowStride,
                             unsigned colStride, int32_t logicalCols,
                             std::string *error) {
  if (rowStride == 0 || colStride == 0 || logicalCols < 0) {
    if (error)
      *error = "invalid tcgen05.copy destination tile stride";
    return std::nullopt;
  }

  llvm::SmallVector<TMemCopyScheduledTile> tiles;
  for (int32_t logicalCol = 0; logicalCol < logicalCols;
       logicalCol += static_cast<int32_t>(colStride)) {
    auto footprint = getTMemCopyDestinationFootprint(
        query, family, logicalCol, rowStride, colStride);
    if (!footprint) {
      if (error) {
        *error = "failed to compute physical tcgen05.copy destination tile "
                 "offset from the selected tensor-memory layout";
      }
      return std::nullopt;
    }
    tiles.push_back(TMemCopyScheduledTile{
        /*destination=*/ *footprint,
        /*sourceRow=*/0,
        /*sourceCol=*/logicalCol});
  }
  return tiles;
}

static std::optional<unsigned>
getTMemCopyEffectiveInstructionColumns(const TMemCopyMessagePlan &plan) {
  if (plan.instrShape.size() < 2 || plan.instrShape[1] == 0)
    return std::nullopt;
  if (plan.atom.nRow == 4 && plan.atom.bCol == 256) {
    // The refresh primitive is scheduled as two half-width messages: source
    // columns 0..3 and 4..7 land at destination dword offsets 0 and 4.
    return plan.instrShape[1] / 2;
  }
  return plan.instrShape[1];
}

static std::optional<TMemCopySourceFootprint>
getTMemCopySourceFootprint(const TMemCopyScheduledMessage &message,
                           const TMemCopyScheduledTile &tile,
                           std::string *error) {
  const TMemCopyMessagePlan &plan = message.plan;
  int64_t sourceRow = static_cast<int64_t>(plan.smemRow) + tile.sourceRow;
  int64_t sourceCol =
      static_cast<int64_t>(plan.smemColOffset) + tile.sourceCol;
  if (sourceRow < 0 || sourceCol < 0 ||
      sourceRow > std::numeric_limits<int32_t>::max() ||
      sourceCol > std::numeric_limits<int32_t>::max()) {
    if (error)
      *error = "tcgen05.copy source footprint has an invalid source "
               "coordinate.";
    return std::nullopt;
  }
  if (plan.instrShape.size() < 2 || plan.instrShape[0] == 0 ||
      plan.instrShape[1] == 0) {
    if (error)
      *error =
          "tcgen05.copy source footprint requires a non-empty instruction "
          "shape.";
    return std::nullopt;
  }
  auto footprintColumns = getTMemCopyEffectiveInstructionColumns(plan);
  if (!footprintColumns || *footprintColumns == 0) {
    if (error)
      *error =
          "tcgen05.copy source footprint requires a non-empty effective "
          "instruction width.";
    return std::nullopt;
  }
  unsigned columns = *footprintColumns;
  TMemCopySourceCoordinateSpace coordinateSpace =
      TMemCopySourceCoordinateSpace::DescriptorLoader;
  if (plan.useDirectSeedDescriptor)
    coordinateSpace = TMemCopySourceCoordinateSpace::DirectSeedImmediate;
  return TMemCopySourceFootprint{
      /*row=*/static_cast<int32_t>(sourceRow),
      /*col=*/static_cast<int32_t>(sourceCol),
      /*rows=*/plan.instrShape[0],
      /*columns=*/columns,
      /*coordinateSpace=*/coordinateSpace};
}

static std::optional<TMemCopyDestinationFootprint>
getTMemCopyInstructionDestinationFootprint(
    const TMemCopyScheduledMessage &message, const TMemCopyScheduledTile &tile,
    std::string *error) {
  const TMemCopyMessagePlan &plan = message.plan;
  if (plan.tmemRowDelta < 0 || plan.tmemDwordDelta < 0) {
    if (error)
      *error = "tcgen05.copy destination footprint has a negative "
               "destination delta.";
    return std::nullopt;
  }
  if (plan.instrShape.size() < 2 || plan.instrShape[0] == 0 ||
      plan.instrShape[1] == 0 || plan.atom.bCol <= 0 ||
      plan.atom.bCol % static_cast<int>(plan.instrShape[1]) != 0) {
    if (error)
      *error = "tcgen05.copy destination footprint requires a valid "
               "instruction shape and copy atom width.";
    return std::nullopt;
  }

  int32_t elementBitWidth = plan.atom.bCol / plan.instrShape[1];
  int64_t dwordDeltaBits = static_cast<int64_t>(plan.tmemDwordDelta) * 32;
  if (elementBitWidth <= 0 || dwordDeltaBits % elementBitWidth != 0) {
    if (error)
      *error = "tcgen05.copy destination dword delta does not align to an "
               "element column.";
    return std::nullopt;
  }

  int64_t physicalRow =
      static_cast<int64_t>(tile.destination.physicalRow) + plan.tmemRowDelta;
  int64_t physicalCol = static_cast<int64_t>(tile.destination.physicalCol) +
                        dwordDeltaBits / elementBitWidth;
  int64_t offset = static_cast<int64_t>(tile.destination.offset) +
                   getTMemPackedOffsetRowBase(
                       static_cast<uint32_t>(plan.tmemRowDelta)) +
                   plan.tmemDwordDelta;
  if (physicalRow > std::numeric_limits<int32_t>::max() ||
      physicalCol > std::numeric_limits<int32_t>::max() ||
      offset > std::numeric_limits<uint32_t>::max()) {
    if (error)
      *error = "tcgen05.copy destination footprint exceeds the supported "
               "coordinate range.";
    return std::nullopt;
  }
  auto footprintColumns = getTMemCopyEffectiveInstructionColumns(plan);
  if (!footprintColumns || *footprintColumns == 0) {
    if (error)
      *error =
          "tcgen05.copy destination footprint requires a non-empty effective "
          "instruction width.";
    return std::nullopt;
  }
  unsigned columns = *footprintColumns;

  return TMemCopyDestinationFootprint{
      /*logicalRow=*/tile.destination.logicalRow,
      /*logicalCol=*/tile.destination.logicalCol,
      /*physicalRow=*/static_cast<int32_t>(physicalRow),
      /*physicalCol=*/static_cast<int32_t>(physicalCol),
      /*rows=*/tile.destination.rows,
      /*columns=*/columns,
      /*offset=*/static_cast<uint32_t>(offset)};
}

std::optional<llvm::SmallVector<TMemCopyScheduledInstruction>>
getTMemCopyInstructionSchedule(ArrayRef<TMemCopyScheduledMessage> messages,
                               ArrayRef<TMemCopyScheduledTile> tiles,
                               std::string *error) {
  if (messages.empty()) {
    if (error)
      *error = "tcgen05.copy instruction scheduling requires at least one "
               "selected message.";
    return std::nullopt;
  }

  llvm::SmallVector<TMemCopyScheduledInstruction> instructions;
  for (const TMemCopyScheduledTile &tile : tiles) {
    for (unsigned messageIdx = 0, e = messages.size(); messageIdx < e;
         ++messageIdx) {
      auto source = getTMemCopySourceFootprint(messages[messageIdx], tile,
                                               error);
      if (!source)
        return std::nullopt;
      auto destination =
          getTMemCopyInstructionDestinationFootprint(messages[messageIdx],
                                                     tile, error);
      if (!destination)
        return std::nullopt;
      instructions.push_back(TMemCopyScheduledInstruction{
          /*messageIndex=*/messageIdx,
          /*tile=*/tile,
          /*source=*/ *source,
          /*destination=*/ *destination});
    }
  }
  return instructions;
}

static bool intervalsOverlap(int64_t lhsBegin, int64_t lhsEnd,
                             int64_t rhsBegin, int64_t rhsEnd) {
  return lhsBegin < rhsEnd && rhsBegin < lhsEnd;
}

static TMemCopySupportResult getTMemCopySourceFootprintSupport(
    MemDescType srcTy, TMemCopyFamily family,
    ArrayRef<TMemCopyScheduledMessage> messages,
    ArrayRef<TMemCopyScheduledInstruction> instructions) {
  for (const TMemCopyScheduledInstruction &instruction : instructions) {
    const TMemCopySourceFootprint &source = instruction.source;
    if (instruction.messageIndex >= messages.size()) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::InstructionSchedule,
          "tcgen05.copy instruction schedule references an unknown source "
          "message.");
    }
    const auto &message = messages[instruction.messageIndex];
    if (source.coordinateSpace ==
        TMemCopySourceCoordinateSpace::DirectSeedImmediate) {
      if (srcTy.getRank() != 2) {
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::PhysicalQuery,
            "tcgen05.copy direct-seed source-footprint bounds checking "
            "requires a rank-2 shared-memory source tile.");
      }
      if (source.row != 0) {
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::InstructionSchedule,
            "tcgen05.copy direct-seed source footprint has a non-zero source "
            "row, but the direct seed descriptor only carries a base source "
            "offset.");
      }
      int64_t sourceColBits =
          static_cast<int64_t>(source.col) * srcTy.getElementTypeBitWidth();
      if (sourceColBits % 128 != 0) {
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::InstructionSchedule,
            "tcgen05.copy direct-seed source footprint has a source column "
            "that is not aligned to a 128-bit descriptor offset.");
      }
      int64_t sourceOffsetBits =
          (static_cast<int64_t>(message.plan.directSourceOffsetB128) +
           sourceColBits / 128) *
          128;
      int64_t sourceFootprintBits = static_cast<int64_t>(source.rows) *
                                    source.columns *
                                    srcTy.getElementTypeBitWidth();
      int64_t sourceTileBits = static_cast<int64_t>(srcTy.getShape()[0]) *
                               srcTy.getShape()[1] *
                               srcTy.getElementTypeBitWidth();
      if (sourceOffsetBits < 0 ||
          sourceOffsetBits + sourceFootprintBits > sourceTileBits) {
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::InstructionSchedule,
            Twine("tcgen05.copy.") + stringifyTMemCopyFamily(family) +
                " direct-seed instruction schedule reads source bit range [" +
                Twine(sourceOffsetBits) + ", " +
                Twine(sourceOffsetBits + sourceFootprintBits) +
                ") outside source tile bit range [0, " +
                Twine(sourceTileBits) + ").");
      }
      continue;
    }

    assert(source.coordinateSpace ==
               TMemCopySourceCoordinateSpace::DescriptorLoader &&
           "unknown tcgen05.copy source coordinate space");
    if (!message.descriptorLayout) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::InstructionSchedule,
          "tcgen05.copy descriptor-loader source footprint has no selected "
          "descriptor layout.");
    }
    const LinearLayout &descriptorLayout = message.descriptorLayout->layout;
    auto descriptorDims = llvm::to_vector(descriptorLayout.getInDimNames());
    if (descriptorDims.size() != 2) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::InstructionSchedule,
          "tcgen05.copy descriptor-loader source footprint requires a "
          "selected two-dimensional descriptor layout.");
    }
    int64_t sourceRows = descriptorLayout.getInDimSize(descriptorDims[0]);
    int64_t sourceCols = descriptorLayout.getInDimSize(descriptorDims[1]);

    int64_t rowEnd = static_cast<int64_t>(source.row) + source.rows;
    int64_t colEnd = static_cast<int64_t>(source.col) + source.columns;
    if (source.row < 0 || source.col < 0 || rowEnd > sourceRows ||
        colEnd > sourceCols) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::InstructionSchedule,
          Twine("tcgen05.copy.") + stringifyTMemCopyFamily(family) +
              " instruction schedule reads descriptor-loader source "
              "footprint [row " +
              Twine(source.row) + ", " + Twine(rowEnd) + ") x [col " +
              Twine(source.col) + ", " + Twine(colEnd) +
              ") outside source coordinate bounds [" + Twine(sourceRows) +
              ", " + Twine(sourceCols) + "].");
    }
  }
  return getSupportedTMemCopyResult();
}

static TMemCopySupportResult getTMemCopyDestinationFootprintSupport(
    TMemCopyFamily family, ArrayRef<TMemCopyScheduledInstruction> instructions) {
  for (auto [lhsIdx, lhsInstruction] : llvm::enumerate(instructions)) {
    const TMemCopyDestinationFootprint &lhs = lhsInstruction.destination;
    int64_t lhsRowBegin = lhs.physicalRow;
    int64_t lhsRowEnd = lhsRowBegin + lhs.rows;
    int64_t lhsColBegin = lhs.physicalCol;
    int64_t lhsColEnd = lhsColBegin + lhs.columns;
    for (const auto &rhsInstruction : instructions.drop_front(lhsIdx + 1)) {
      const TMemCopyDestinationFootprint &rhs = rhsInstruction.destination;
      int64_t rhsRowBegin = rhs.physicalRow;
      int64_t rhsRowEnd = rhsRowBegin + rhs.rows;
      int64_t rhsColBegin = rhs.physicalCol;
      int64_t rhsColEnd = rhsColBegin + rhs.columns;
      if (intervalsOverlap(lhsRowBegin, lhsRowEnd, rhsRowBegin, rhsRowEnd) &&
          intervalsOverlap(lhsColBegin, lhsColEnd, rhsColBegin, rhsColEnd)) {
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::InstructionSchedule,
            Twine("tcgen05.copy.") + stringifyTMemCopyFamily(family) +
                " instruction schedule writes overlapping destination "
                "footprints. The planner must prove a non-overlapping "
                "destination schedule or use an ISA atom with an explicit "
                "destination mask before this layout can be supported.");
      }
    }
  }
  return getSupportedTMemCopyResult();
}

static std::optional<std::pair<unsigned, unsigned>>
getTMemCopyColumnSelectionRunAndPeriod(unsigned logicalColBit) {
  if (logicalColBit >= std::numeric_limits<unsigned>::digits - 1)
    return std::nullopt;
  return std::pair<unsigned, unsigned>{1u << logicalColBit,
                                       1u << (logicalColBit + 1)};
}

static void appendTMemCopyDestinationMaskScheduleGap(
    llvm::raw_ostream &os,
    const TMemCopyDestinationMaskRequirement &requirement,
    StringRef requirementName, StringRef footprintScope);

static unsigned getDenseTMemCopyInstructionRows(TMemCopyFamily family);

enum class TMemCopyDestinationRowOrderRequirementKind {
  DenseRowBases,
  DenseRowRepetitionBases,
  MulticastNonBroadcastRowBases,
};

struct TMemCopyRowBasisStep {
  unsigned bit = 0;
  int32_t physicalRow = 0;
};

struct TMemCopyDestinationRowOrderRequirement {
  TMemCopyDestinationRowOrderRequirementKind kind =
      TMemCopyDestinationRowOrderRequirementKind::DenseRowBases;
  TMemCopyFamily family = TMemCopyFamily::Dense128x128b;
  unsigned instructionRows = 0;
  unsigned instructionColumns = 0;
  llvm::SmallVector<TMemCopyRowBasisStep, 8> steps;
};

enum class TMemCopyMixedBasisRequirementKind {
  RowBasis,
  ColumnBasis,
};

struct TMemCopyMixedBasisRequirement {
  TMemCopyMixedBasisRequirementKind kind =
      TMemCopyMixedBasisRequirementKind::RowBasis;
  TMemCopyFamily family = TMemCopyFamily::Dense128x128b;
  unsigned bit = 0;
  int32_t physicalRow = 0;
  int32_t physicalCol = 0;
};

struct TMemCopyColumnFootprintRequirement {
  TMemCopyFamily family = TMemCopyFamily::Warpx2_01_23_64x128b;
  unsigned elementBitwidth = 0;
  unsigned instructionColumns = 0;
  unsigned physicalDwordColumns = 0;
  unsigned lanesPerDword = 1;
  unsigned availableColumnBasisBits = 0;
  unsigned requiredColumnBasisBits = 0;
};

static std::string getTMemCopyMixedBasisRequirementError(
    const TMemCopyMixedBasisRequirement &requirement) {
  StringRef basisKind;
  switch (requirement.kind) {
  case TMemCopyMixedBasisRequirementKind::RowBasis:
    basisKind = "row";
    break;
  case TMemCopyMixedBasisRequirementKind::ColumnBasis:
    basisKind = "column";
    break;
  }

  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "direct tcgen05.copy." << stringifyTMemCopyFamily(requirement.family)
     << " does not support TMEM " << basisKind
     << " bases that mix row and column contributions. The offending "
     << basisKind << " basis bit " << requirement.bit
     << " maps to physical TMEM delta [" << requirement.physicalRow << ", "
     << requirement.physicalCol
     << "]. Public copy atoms expose one tensor-memory address per "
        "instruction; mixed row/column bases need a rematerialized view or a "
        "multi-instruction schedule whose source and destination footprints "
        "are proved equivalent before this layout can be supported.";
  return os.str();
}

static TMemCopySupportResult getTMemCopyColumnFootprintFailure(
    const TMemCopyColumnFootprintRequirement &requirement) {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "direct tcgen05.copy." << stringifyTMemCopyFamily(requirement.family)
     << " requires enough TMEM column bases to cover the copy instruction "
        "width. The destination-column footprint requirement exposes "
     << requirement.availableColumnBasisBits << " column basis bit"
     << (requirement.availableColumnBasisBits == 1 ? "" : "s") << ", but the "
     << requirement.instructionColumns << "-column copy instruction requires "
     << requirement.requiredColumnBasisBits << " logical column basis bit"
     << (requirement.requiredColumnBasisBits == 1 ? "" : "s");
  if (requirement.elementBitwidth)
    os << " for " << requirement.elementBitwidth << "-bit elements";
  os << ".";
  if (requirement.physicalDwordColumns) {
    os << " Those logical columns occupy "
       << requirement.physicalDwordColumns
       << " physical 32-bit dword column"
       << (requirement.physicalDwordColumns == 1 ? "" : "s");
    if (requirement.lanesPerDword > 1) {
      os << " with " << requirement.lanesPerDword
         << " packed lane"
         << (requirement.lanesPerDword == 1 ? "" : "s") << " per word";
    }
    os << ".";
  }
  os << " Support needs a packed-lane source/destination storage model that "
        "carries lane selection through descriptor synthesis, source footprint "
        "planning, and the tcgen05.copy instruction schedule; descriptor "
        "footprint coverage alone is not a correctness proof.";
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::PhysicalQuery, os.str());
}

static std::optional<unsigned> findFirstNonAscendingRowBasis(
    ArrayRef<TMemCopyRowBasisStep> steps) {
  if (steps.size() < 2)
    return std::nullopt;
  for (unsigned idx = 1; idx < steps.size(); ++idx) {
    if (steps[idx].physicalRow <= steps[idx - 1].physicalRow)
      return idx;
  }
  return std::nullopt;
}

static std::optional<TMemCopyDestinationMaskRequirement>
getTMemCopyDestinationMaskRequirement(
    const TMemCopyDestinationRowOrderRequirement &requirement) {
  if (requirement.kind ==
      TMemCopyDestinationRowOrderRequirementKind::DenseRowRepetitionBases)
    return std::nullopt;

  auto offendingIdx = findFirstNonAscendingRowBasis(requirement.steps);
  if (!offendingIdx)
    return std::nullopt;
  unsigned logicalRowBit = requirement.steps[*offendingIdx].bit;
  if (logicalRowBit >= std::numeric_limits<unsigned>::digits - 1)
    return std::nullopt;

  return TMemCopyDestinationMaskRequirement{
      /*axis=*/TMemCopyDestinationMaskAxis::Row,
      /*instructionRows=*/requirement.instructionRows,
      /*instructionColumns=*/requirement.instructionColumns,
      /*selectedRun=*/1u << logicalRowBit,
      /*selectionPeriod=*/1u << (logicalRowBit + 1)};
}

static StringRef stringifyTMemCopyDestinationRowOrderBasis(
    TMemCopyDestinationRowOrderRequirementKind kind) {
  switch (kind) {
  case TMemCopyDestinationRowOrderRequirementKind::DenseRowBases:
    return "row bases";
  case TMemCopyDestinationRowOrderRequirementKind::DenseRowRepetitionBases:
    return "row-repetition bases stored in the column address space";
  case TMemCopyDestinationRowOrderRequirementKind::MulticastNonBroadcastRowBases:
    return "non-broadcast TMEM row bases";
  }
  llvm_unreachable("unknown tcgen05.copy row-order requirement kind");
}

static TMemCopySupportResult getDenseTMemCopyRowOrderFailure(
    const TMemCopyDestinationRowOrderRequirement &requirement) {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  StringRef basisKind =
      stringifyTMemCopyDestinationRowOrderBasis(requirement.kind);
  os << "direct tcgen05.copy requires TMEM " << basisKind
     << " to stay in ascending physical row order until the planner can "
        "derive an explicit source-row projection schedule with a "
        "destination-row mask, row-partitioned atom, or equivalent smaller "
        "copy footprint for row-permuted destinations.";
  if (auto offendingIdx = findFirstNonAscendingRowBasis(requirement.steps)) {
    const auto &previous = requirement.steps[*offendingIdx - 1];
    const auto &current = requirement.steps[*offendingIdx];
    os << " The first non-ascending basis is bit " << current.bit
       << " mapping to physical row " << current.physicalRow
       << " after bit " << previous.bit << " mapped to physical row "
       << previous.physicalRow << ".";
  }
  os << " Current dense copy atoms write the full physical row footprint in "
        "basis order.";
  if (auto maskRequirement = getTMemCopyDestinationMaskRequirement(requirement))
    appendTMemCopyDestinationMaskScheduleGap(
        os, *maskRequirement, "destination-row order requirement",
        "for each emitted instruction");
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

static TMemCopySupportResult
getDenseTMemCopyRowProjectionSupport(const LinearLayout &ll, MLIRContext *ctx,
                                     TMemCopyFamily family,
                                     unsigned bitwidth) {
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");

  unsigned instructionRows = getDenseTMemCopyInstructionRows(family);
  unsigned instructionColumns = getDenseTMemCopyColumnStride(family, bitwidth);

  SmallVector<TMemCopyRowBasisStep> rowBasisValues;
  for (auto [idx, basis] : llvm::enumerate(ll.getBases().lookup(kRow))) {
    if (basis[0] == 0 || basis[1] != 0) {
      TMemCopyMixedBasisRequirement requirement;
      requirement.kind = TMemCopyMixedBasisRequirementKind::RowBasis;
      requirement.family = family;
      requirement.bit = static_cast<unsigned>(idx);
      requirement.physicalRow = basis[0];
      requirement.physicalCol = basis[1];
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          getTMemCopyMixedBasisRequirementError(requirement));
    }
    rowBasisValues.push_back(
        TMemCopyRowBasisStep{static_cast<unsigned>(idx), std::abs(basis[0])});
  }
  if (findFirstNonAscendingRowBasis(rowBasisValues)) {
    TMemCopyDestinationRowOrderRequirement requirement;
    requirement.kind =
        TMemCopyDestinationRowOrderRequirementKind::DenseRowBases;
    requirement.family = family;
    requirement.instructionRows = instructionRows;
    requirement.instructionColumns = instructionColumns;
    requirement.steps = std::move(rowBasisValues);
    return getDenseTMemCopyRowOrderFailure(requirement);
  }

  SmallVector<TMemCopyRowBasisStep> rowRepetitionBasisValues;
  for (auto [idx, basis] : llvm::enumerate(ll.getBases().lookup(kCol))) {
    bool touchesRow = basis[0] != 0;
    bool touchesCol = basis[1] != 0;
    if (touchesRow && touchesCol) {
      TMemCopyMixedBasisRequirement requirement;
      requirement.kind = TMemCopyMixedBasisRequirementKind::ColumnBasis;
      requirement.family = family;
      requirement.bit = static_cast<unsigned>(idx);
      requirement.physicalRow = basis[0];
      requirement.physicalCol = basis[1];
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          getTMemCopyMixedBasisRequirementError(requirement));
    }
    if (touchesRow && !touchesCol)
      rowRepetitionBasisValues.push_back(
          TMemCopyRowBasisStep{static_cast<unsigned>(idx),
                               std::abs(basis[0])});
  }
  if (findFirstNonAscendingRowBasis(rowRepetitionBasisValues)) {
    TMemCopyDestinationRowOrderRequirement requirement;
    requirement.kind =
        TMemCopyDestinationRowOrderRequirementKind::DenseRowRepetitionBases;
    requirement.family = family;
    requirement.instructionRows = instructionRows;
    requirement.instructionColumns = instructionColumns;
    requirement.steps = std::move(rowRepetitionBasisValues);
    return getDenseTMemCopyRowOrderFailure(requirement);
  }
  return getSupportedTMemCopyResult();
}

static unsigned getDenseTMemCopyInstructionRows(TMemCopyFamily family) {
  switch (family) {
  case TMemCopyFamily::Dense4x256b:
    return 4;
  case TMemCopyFamily::Dense128x128b:
  case TMemCopyFamily::Dense128x256b:
    return 128;
  case TMemCopyFamily::Warpx2_01_23_64x128b:
  case TMemCopyFamily::Warpx2_02_13_64x128b:
  case TMemCopyFamily::Warpx4_32x128b:
    llvm_unreachable("non-dense copy family");
  }
  llvm_unreachable("unknown dense copy family");
}

static std::optional<TMemCopyInstructionColumnPermutationRequirement>
getDenseTMemCopyDestinationColumnPermutationRequirement(
    const LinearLayout &ll, MLIRContext *ctx, TMemCopyFamily family,
    unsigned bitwidth) {
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kCol))
    return std::nullopt;

  unsigned instructionColumns =
      getDenseTMemCopyColumnStride(family, bitwidth);
  if (!llvm::isPowerOf2_32(instructionColumns) || instructionColumns <= 1)
    return std::nullopt;

  unsigned instructionColumnBits = llvm::Log2_32(instructionColumns);
  auto colBases = ll.getBases().lookup(kCol);
  if (colBases.size() < instructionColumnBits)
    return std::nullopt;

  for (unsigned bit = 0; bit < instructionColumnBits; ++bit) {
    ArrayRef<int32_t> basis = colBases[bit];
    if (basis.size() < 2)
      return std::nullopt;
    int32_t actualPhysicalRowDelta = basis[0];
    int32_t actualPhysicalColumnDelta = basis[1];
    int32_t expectedPhysicalColumnDelta = 1 << bit;
    if (actualPhysicalRowDelta == 0 &&
        actualPhysicalColumnDelta == expectedPhysicalColumnDelta)
      continue;
    if (actualPhysicalRowDelta != 0)
      return std::nullopt;
    auto runAndPeriod = getTMemCopyColumnSelectionRunAndPeriod(bit);
    if (!runAndPeriod)
      return std::nullopt;
    TMemCopyInstructionColumnPermutationRequirement requirement;
    requirement.instructionRows = getDenseTMemCopyInstructionRows(family);
    requirement.instructionColumns = instructionColumns;
    requirement.logicalColBit = bit;
    requirement.selectedColumnRun = runAndPeriod->first;
    requirement.columnSelectionPeriod = runAndPeriod->second;
    requirement.actualOffset = actualPhysicalColumnDelta;
    requirement.expectedOffset = expectedPhysicalColumnDelta;
    return requirement;
  }
  return std::nullopt;
}

static TMemCopySupportResult getDenseTMemCopyColumnPermutationFailure(
    TMemCopyFamily family,
    const TMemCopyInstructionColumnPermutationRequirement &requirement) {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "direct tcgen05.copy." << stringifyTMemCopyFamily(family)
     << " requires each logical column tile to remain contiguous in physical "
        "TMEM column order. Within one "
     << requirement.instructionColumns
     << "-column copy instruction, destination column bit "
     << requirement.logicalColBit << " maps to physical column delta "
     << requirement.actualOffset << " instead of contiguous physical column "
        "delta "
     << requirement.expectedOffset << ".";
  if (auto maskRequirement =
          getTMemCopyDestinationMaskRequirement(requirement)) {
    appendTMemCopyDestinationMaskScheduleGap(
        os, *maskRequirement, "destination-column permutation requirement",
        "for each emitted instruction");
  }
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

static TMemCopySupportResult getTMemCopyDestinationBlockOwnershipSupport(
    const LinearLayout &ll, MLIRContext *ctx, TMemCopyFamily family,
    bool twoCTAs, int32_t expectedBlockRow,
    bool allowBroadcastBlockOwnership = false) {
  if (!twoCTAs)
    return getSupportedTMemCopyResult();

  auto kBlock = StringAttr::get(ctx, "block");
  auto makeFailure = [&](Twine detail) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::CtaOwnership,
        Twine("direct tcgen05.copy.") + stringifyTMemCopyFamily(family) +
            " two-CTA destination layouts require the canonical TMEM block "
            "basis [[" +
            Twine(expectedBlockRow) + ", 0]]. " + detail);
  };

  if (!ll.hasInDim(kBlock))
    return makeFailure("The destination query does not expose any "
                       "two-CTA block-selection bits.");

  auto blockBases = ll.getBases().lookup(kBlock);
  unsigned canonicalBasisCount = 0;
  bool allOtherBlockBasesCompatible = true;
  for (ArrayRef<int32_t> basis : blockBases) {
    bool isCanonical = basis.size() >= 2 && basis[0] == expectedBlockRow &&
                       basis[1] == 0;
    bool isBroadcast =
        llvm::all_of(basis, [](int32_t value) { return value == 0; });
    bool isOuterRowOwnership =
        allowBroadcastBlockOwnership && basis.size() >= 2 &&
        basis[0] > expectedBlockRow && basis[0] % expectedBlockRow == 0 &&
        basis[1] == 0 &&
        llvm::all_of(basis.drop_front(2),
                     [](int32_t value) { return value == 0; });
    canonicalBasisCount += isCanonical ? 1 : 0;
    allOtherBlockBasesCompatible &=
        isCanonical || isBroadcast || isOuterRowOwnership;
  }
  bool hasCanonicalBlockBasis =
      canonicalBasisCount == 1 && allOtherBlockBasesCompatible;
  bool hasBroadcastBlockBasis =
      allowBroadcastBlockOwnership && canonicalBasisCount == 0 &&
      allOtherBlockBasesCompatible;
  if (!hasCanonicalBlockBasis && !hasBroadcastBlockBasis) {
    return makeFailure("CTA ownership is part of the instruction schedule; "
                       "without this basis the current copy atom cannot "
                       "select the second CTA's physical row half correctly.");
  }
  return getSupportedTMemCopyResult();
}

static LinearLayout canonicalizeTMemCopyLayoutDimsForAnalysis(
    LinearLayout layout, MLIRContext *ctx) {
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto kBlock = StringAttr::get(ctx, "block");
  SmallVector<StringAttr> canonicalInDims;
  for (StringAttr dim : {kRow, kCol, kBlock}) {
    if (layout.hasInDim(dim))
      canonicalInDims.push_back(dim);
  }
  if (!canonicalInDims.empty())
    layout = layout.transposeIns(canonicalInDims);
  return layout.transposeOuts(standardOutDimNames(ctx, layout.getNumOutDims()));
}

static TMemCopySupportResult getTMemCopyMulticastDestinationRowOrderFailure(
    const TMemCopyDestinationRowOrderRequirement &requirement) {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "direct tcgen05.copy." << stringifyTMemCopyFamily(requirement.family)
     << " requires non-broadcast TMEM row bases to stay in ascending physical "
        "row order. Public multicast copy atoms write a fixed physical row "
        "footprint for each instruction, so row-permuted destinations need an "
        "explicit source-row projection schedule with a destination-row mask, "
        "row-partitioned atom, or equivalent smaller copy footprint.";
  if (auto offendingIdx = findFirstNonAscendingRowBasis(requirement.steps)) {
    const auto &previous = requirement.steps[*offendingIdx - 1];
    const auto &current = requirement.steps[*offendingIdx];
    os << " The first non-ascending non-broadcast basis is bit "
       << current.bit << " mapping to physical row " << current.physicalRow
       << " after bit " << previous.bit << " mapped to physical row "
       << previous.physicalRow << ".";
  }
  if (auto maskRequirement = getTMemCopyDestinationMaskRequirement(requirement))
    appendTMemCopyDestinationMaskScheduleGap(
        os, *maskRequirement, "destination-row order requirement",
        "for each emitted instruction");
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

static std::optional<unsigned>
getTMemCopyMulticastBroadcastMask(TMemCopyFamily family) {
  switch (family) {
  case TMemCopyFamily::Warpx2_01_23_64x128b:
    return 1u;
  case TMemCopyFamily::Warpx2_02_13_64x128b:
    return 2u;
  case TMemCopyFamily::Warpx4_32x128b:
    return 3u;
  case TMemCopyFamily::Dense4x256b:
  case TMemCopyFamily::Dense128x128b:
  case TMemCopyFamily::Dense128x256b:
    return std::nullopt;
  }
  llvm_unreachable("unknown tcgen05.copy family");
}

static TMemCopySupportResult getMulticastTMemCopyDestinationLayoutSupport(
    const LinearLayout &layout, MLIRContext *ctx, TMemCopyFamily family,
    unsigned bitwidth, bool twoCTAs) {
  auto broadcastMask = getTMemCopyMulticastBroadcastMask(family);
  if (!broadcastMask)
    return getSupportedTMemCopyResult();

  auto ll = canonicalizeTMemCopyLayoutDimsForAnalysis(layout, ctx);
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) || ll.getNumOutDims() != 2) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery,
        "direct tcgen05.copy multicast destinations currently require a "
        "rank-2 TMEM view with explicit row/col bases.");
  }

  SmallVector<TMemCopyRowBasisStep> nonBroadcastRowBases;
  auto rowBases = ll.getBases().lookup(kRow);
  constexpr unsigned kRow32Bit = 5;
  constexpr unsigned kRow64Bit = 6;
  for (auto [idx, basis] : llvm::enumerate(rowBases)) {
    bool expectedBroadcast =
        (idx == kRow32Bit && (*broadcastMask & 1u)) ||
        (idx == kRow64Bit && (*broadcastMask & 2u));
    if (expectedBroadcast) {
      if (!isAllZeroBasis(basis)) {
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::PhysicalQuery,
            Twine("direct tcgen05.copy.") + stringifyTMemCopyFamily(family) +
                " requires the multicast row basis at logical row bit " +
                Twine(idx) + " to be a zero/broadcast basis.");
      }
      continue;
    }
    if (basis.size() < 2 || basis[0] == 0 || basis[1] != 0) {
      if (basis.size() >= 2 && basis[0] != 0 && basis[1] != 0) {
        TMemCopyMixedBasisRequirement requirement;
        requirement.kind = TMemCopyMixedBasisRequirementKind::RowBasis;
        requirement.family = family;
        requirement.bit = static_cast<unsigned>(idx);
        requirement.physicalRow = basis[0];
        requirement.physicalCol = basis[1];
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::PhysicalQuery,
            getTMemCopyMixedBasisRequirementError(requirement));
      }
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          Twine("direct tcgen05.copy.") + stringifyTMemCopyFamily(family) +
              " requires non-broadcast row bases to map only to physical "
              "TMEM rows.");
    }
    nonBroadcastRowBases.push_back(
        TMemCopyRowBasisStep{static_cast<unsigned>(idx), std::abs(basis[0])});
  }
  if (findFirstNonAscendingRowBasis(nonBroadcastRowBases)) {
    TMemCopyDestinationRowOrderRequirement requirement;
    requirement.kind =
        TMemCopyDestinationRowOrderRequirementKind::MulticastNonBroadcastRowBases;
    requirement.family = family;
    requirement.instructionRows =
        family == TMemCopyFamily::Warpx4_32x128b ? 32 : 64;
    requirement.instructionColumns = bitwidth == 0 ? 0 : 128 / bitwidth;
    requirement.steps = std::move(nonBroadcastRowBases);
    return getTMemCopyMulticastDestinationRowOrderFailure(requirement);
  }

  if (bitwidth == 0 || 128 % bitwidth != 0) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::IsaAtom,
        Twine("direct tcgen05.copy.") + stringifyTMemCopyFamily(family) +
            " requires an element bitwidth that divides the 128-bit "
            "multicast instruction width.");
  }
  unsigned instructionColumns = 128 / bitwidth;
  if (!llvm::isPowerOf2_32(instructionColumns)) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::IsaAtom,
        Twine("direct tcgen05.copy.") + stringifyTMemCopyFamily(family) +
            " requires a power-of-two multicast instruction column count.");
  }
  auto colBases = ll.getBases().lookup(kCol);
  unsigned instructionColumnBits = llvm::Log2_32(instructionColumns);
  auto getBroadcastRowMaskForColumnBasis = [&](ArrayRef<int32_t> basis)
      -> std::optional<unsigned> {
    if (basis.size() < 2 || basis[1] != 0)
      return std::nullopt;
    if ((*broadcastMask & 1u) && basis[0] == 32)
      return 1u;
    if ((*broadcastMask & 2u) && basis[0] == 64)
      return 2u;
    return std::nullopt;
  };
  if (colBases.size() < instructionColumnBits) {
    unsigned lanesPerDword =
        (bitwidth > 0 && 32 % bitwidth == 0) ? 32 / bitwidth : 1;
    unsigned physicalDwordColumns =
        bitwidth > 0 ? llvm::divideCeil(instructionColumns * bitwidth, 32u)
                     : 0u;
    TMemCopyColumnFootprintRequirement requirement;
    requirement.family = family;
    requirement.elementBitwidth = bitwidth;
    requirement.instructionColumns = instructionColumns;
    requirement.physicalDwordColumns = physicalDwordColumns;
    requirement.lanesPerDword = lanesPerDword;
    requirement.availableColumnBasisBits = colBases.size();
    requirement.requiredColumnBasisBits = instructionColumnBits;
    return getTMemCopyColumnFootprintFailure(requirement);
  }
  unsigned consumedBroadcastColumnRows = 0;
  unsigned rowLiftedColumnBits = 0;
  for (unsigned bit = 0; bit < instructionColumnBits; ++bit) {
    ArrayRef<int32_t> basis = colBases[bit];
    int32_t expectedCol = 1 << (bit - rowLiftedColumnBits);
    if (basis.size() >= 2 && basis[0] == 0 && basis[1] == expectedCol)
      continue;
    if (auto rowMask = getBroadcastRowMaskForColumnBasis(basis)) {
      if ((consumedBroadcastColumnRows & *rowMask) == 0) {
        consumedBroadcastColumnRows |= *rowMask;
        ++rowLiftedColumnBits;
        continue;
      }
    }
    auto runAndPeriod = getTMemCopyColumnSelectionRunAndPeriod(bit);
    TMemCopyInstructionColumnPermutationRequirement requirement;
    requirement.instructionRows =
        family == TMemCopyFamily::Warpx4_32x128b ? 32 : 64;
    requirement.instructionColumns = instructionColumns;
    requirement.logicalColBit = bit;
    if (runAndPeriod) {
      requirement.selectedColumnRun = runAndPeriod->first;
      requirement.columnSelectionPeriod = runAndPeriod->second;
    }
    requirement.actualOffset = basis.size() >= 2 ? basis[1] : 0;
    requirement.expectedOffset = expectedCol;
    return getDenseTMemCopyColumnPermutationFailure(family, requirement);
  }

  int32_t expectedBlockRow = 128;
  if (twoCTAs) {
    auto outDims = llvm::to_vector(ll.getOutDimNames());
    if (!outDims.empty() && ll.getOutDimSize(outDims.front()) >= 2)
      expectedBlockRow =
          std::min<int32_t>(ll.getOutDimSize(outDims.front()) / 2, 128);
  }
  auto blockOwnershipSupport = getTMemCopyDestinationBlockOwnershipSupport(
      ll, ctx, family, twoCTAs, expectedBlockRow,
      /*allowBroadcastBlockOwnership=*/true);
  if (!blockOwnershipSupport)
    return blockOwnershipSupport;

  return getSupportedTMemCopyResult();
}

std::string getTMemCopy4x256RefreshImageRequirementError(
    const TMemCopy4x256RefreshImageRequirement &requirement) {
  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "tcgen05.copy.4x256b is recognized by the ISA, but Triton cannot "
        "expose it as an ordinary contiguous four-row ttng.tmem_copy "
        "lowering. The instruction writes a refresh-shaped destination view "
        "for a "
     << requirement.logicalRows << "x" << requirement.logicalColumns
     << " logical tile: logical row bits are stored in TMEM columns, low "
        "logical column bits are stored in TMEM rows "
     << requirement.lowColumnRowDelta0 << "/"
     << requirement.lowColumnRowDelta1
     << ", and the high logical column bit is stored at destination dword +"
     << requirement.highColumnDwordDelta
     << ". Source columns [0, " << requirement.sourceColumnSplit << ") and ["
     << requirement.sourceColumnSplit << ", " << requirement.logicalColumns
     << ") are scheduled as separate physical refresh messages. Ordinary "
        "contiguous tensor-memory layouts need an explicit refresh-image "
        "view/remap plus a load/store contract before this can be supported.";
  return os.str();
}

static TMemCopySupportResult getUnsupportedTMemCopy4x256RefreshImageResult(
    TMemCopySupportFailureLayer layer,
    const TMemCopy4x256RefreshImageRequirement &requirement =
        TMemCopy4x256RefreshImageRequirement{}) {
  return getUnsupportedTMemCopyResult(
      layer, getTMemCopy4x256RefreshImageRequirementError(requirement));
}

static TMemCopySupportResult
getDirectTMemCopyLayoutSupportForLayout(const LinearLayout &layout,
                                        MLIRContext *ctx,
                                        TMemCopyFamily family,
                                        unsigned bitwidth, bool twoCTAs) {
  if (!isDenseTMemCopyFamily(family))
    return getMulticastTMemCopyDestinationLayoutSupport(layout, ctx, family,
                                                        bitwidth, twoCTAs);

  if (family == TMemCopyFamily::Dense4x256b) {
    if (isTMemCopy4x256RefreshLayout(layout, ctx, bitwidth))
      return getSupportedTMemCopyResult();
    return getUnsupportedTMemCopy4x256RefreshImageResult(
        TMemCopySupportFailureLayer::PhysicalQuery);
  }

  auto ll = normalizeTensorMemoryLinearLayoutForAnalysis(layout);
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kRow) || !ll.hasInDim(kCol) || ll.getNumOutDims() != 2) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery,
        "direct tcgen05.copy currently requires a rank-2 TMEM view "
        "with explicit row/col bases.");
  }

  auto blockOwnershipSupport =
      getTMemCopyDestinationBlockOwnershipSupport(ll, ctx, family, twoCTAs,
                                                  /*expectedBlockRow=*/128);
  if (!blockOwnershipSupport)
    return blockOwnershipSupport;

  auto rowProjectionSupport =
      getDenseTMemCopyRowProjectionSupport(ll, ctx, family, bitwidth);
  if (!rowProjectionSupport)
    return rowProjectionSupport;

  auto outDims = llvm::to_vector(ll.getOutDimNames());
  unsigned colStride = getDenseTMemCopyColumnStride(family, bitwidth);
  int32_t colSize = ll.getOutDimSize(outDims[1]);
  SmallVector<uint32_t> visitedTileOffsets;
  for (int32_t logicalCol = 0; logicalCol < colSize;
       logicalCol += colStride) {
    auto tileOrigin =
        getDenseTMemCopyDestinationTileCoord(ll, ctx, 0, logicalCol);
    if (!tileOrigin || tileOrigin->second % static_cast<int32_t>(colStride)) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          "direct tcgen05.copy requires each logical column tile to start at a "
          "physical column aligned to the copy instruction width.");
    }
    uint32_t tileOffset =
        packTMemRowColOffset(static_cast<uint32_t>(tileOrigin->first),
                             getTMemWordColumn(
                                 static_cast<uint32_t>(tileOrigin->second),
                                 bitwidth));
    if (llvm::is_contained(visitedTileOffsets, tileOffset)) {
      return getUnsupportedTMemCopyResult(
          TMemCopySupportFailureLayer::PhysicalQuery,
          "direct tcgen05.copy requires every logical column tile to map to a "
          "unique physical TMEM destination tile.");
    }
    visitedTileOffsets.push_back(tileOffset);
    for (unsigned i = 1; i < colStride && logicalCol + i < colSize; ++i) {
      auto tileCoord =
          getDenseTMemCopyDestinationTileCoord(ll, ctx, 0, logicalCol + i);
      if (!tileCoord || tileCoord->first != tileOrigin->first ||
          tileCoord->second != tileOrigin->second + static_cast<int32_t>(i)) {
        if (auto permutationRequirement =
                getDenseTMemCopyDestinationColumnPermutationRequirement(
                    ll, ctx, family, bitwidth))
          return getDenseTMemCopyColumnPermutationFailure(
              family, *permutationRequirement);
        return getUnsupportedTMemCopyResult(
            TMemCopySupportFailureLayer::PhysicalQuery,
            "direct tcgen05.copy requires each logical column tile to remain "
            "contiguous in physical TMEM column order.");
      }
    }
  }
  return getSupportedTMemCopyResult();
}

TMemCopySupportResult getDirectTMemCopyLayoutSupport(MemDescType memTy,
                                                     TMemCopyFamily family) {
  std::string layoutError;
  auto maybeAnalysis =
      getTMemViewAnalysisLayout(memTy.getShape(), memTy.getEncoding(),
                                &layoutError);
  if (!maybeAnalysis) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::PhysicalQuery, layoutError);
  }
  return getDirectTMemCopyLayoutSupportForLayout(
      maybeAnalysis->layout, memTy.getContext(), family,
      memTy.getElementTypeBitWidth(), maybeAnalysis->twoCTAs);
}

TMemCopySupportResult
getDirectTMemCopyLayoutSupport(const TMemPhysicalQuery &query,
                               TMemCopyFamily family) {
  return getDirectTMemCopyLayoutSupportForLayout(
      query.layout, query.memTy.getContext(), family, query.elementBitWidth,
      query.twoCTAs);
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
getTMemCopySharedDescriptorPlanSupport(gpu::MemDescType srcTy,
                                       const LinearLayout &shmemLl,
                                       const LinearLayout &cvt,
                                       const TMemCopyPlan &plan,
                                       int bitwidth);

static std::pair<std::optional<TMemCopyExecutablePlan>, TMemCopySupportResult>
getTMemCopySharedDescriptorPlanRealization(gpu::MemDescType srcTy,
                                           const LinearLayout &shmemLl,
                                           const LinearLayout &cvt,
                                           const TMemCopyPlan &plan,
                                           int bitwidth);

static std::pair<std::optional<TMemCopyExecutablePlan>, TMemCopySupportResult>
getTMemCopyPlanRealization(MemDescType srcTy,
                           const TMemPhysicalQuery &dstQuery,
                           const LinearLayout &shmemLl, const LinearLayout &cvt,
                           const TMemCopyPlan &plan, int bitwidth) {
  auto layoutSupport = getDirectTMemCopyLayoutSupport(dstQuery, plan.family);
  if (!layoutSupport)
    return {std::nullopt, layoutSupport};

  auto sharedLayoutSupport =
      getTMemCopySharedLayoutRuntimeSupport(srcTy, plan.family);
  if (!sharedLayoutSupport)
    return {std::nullopt, sharedLayoutSupport};

  auto [executablePlan, descriptorSupport] =
      getTMemCopySharedDescriptorPlanRealization(srcTy, shmemLl, cvt, plan,
                                                 bitwidth);
  if (!descriptorSupport)
    return {std::nullopt, descriptorSupport};
  assert(executablePlan &&
         "supported tcgen05.copy descriptor plan must carry an executable "
         "schedule");
  assert(!executablePlan->messages.empty() &&
         "supported tcgen05.copy plan must contain at least one message");

  auto inDims = cvt.getInDimNames();
  if (inDims.empty()) {
    return {std::nullopt,
            getUnsupportedTMemCopyResult(
                TMemCopySupportFailureLayer::PhysicalQuery,
                "tcgen05.copy destination tile planning has no input "
                "dimensions.")};
  }
  auto kCol = StringAttr::get(inDims.begin()->getContext(), "col");
  if (!cvt.hasInDim(kCol)) {
    return {std::nullopt,
            getUnsupportedTMemCopyResult(
                TMemCopySupportFailureLayer::PhysicalQuery,
                "tcgen05.copy destination tile planning requires a column "
                "dimension.")};
  }
  const unsigned rowStride = executablePlan->messages.front().plan.instrShape[0];
  const unsigned colStride = executablePlan->messages.front().plan.instrShape[1];
  std::string destinationTileError;
  auto scheduledTiles = getTMemCopyScheduledTilePlan(
      dstQuery, executablePlan->family, rowStride, colStride,
      cvt.getInDimSize(kCol), &destinationTileError);
  if (!scheduledTiles) {
    return {std::nullopt,
            getUnsupportedTMemCopyResult(
                TMemCopySupportFailureLayer::PhysicalQuery,
                destinationTileError.empty()
                    ? "failed to compute physical tcgen05.copy destination "
                      "tile plan from the selected tensor-memory layout"
                    : destinationTileError)};
  }
  std::string instructionScheduleError;
  auto instructions = getTMemCopyInstructionSchedule(
      executablePlan->messages, *scheduledTiles, &instructionScheduleError);
  if (!instructions) {
    return {std::nullopt,
            getUnsupportedTMemCopyResult(
                TMemCopySupportFailureLayer::InstructionSchedule,
                instructionScheduleError.empty()
                    ? "failed to build tcgen05.copy instruction schedule"
                    : instructionScheduleError)};
  }
  auto sourceFootprintSupport = getTMemCopySourceFootprintSupport(
      srcTy, executablePlan->family, executablePlan->messages, *instructions);
  if (!sourceFootprintSupport)
    return {std::nullopt, sourceFootprintSupport};
  auto destinationFootprintSupport = getTMemCopyDestinationFootprintSupport(
      executablePlan->family, *instructions);
  if (!destinationFootprintSupport)
    return {std::nullopt, destinationFootprintSupport};
  executablePlan->instructions = std::move(*instructions);
  return {std::move(*executablePlan), descriptorSupport};
}

TMemCopySupportResult
getTMemCopyPlanSupport(MemDescType srcTy, const TMemPhysicalQuery &dstQuery,
                       const LinearLayout &shmemLl, const LinearLayout &cvt,
                       const TMemCopyPlan &plan, int bitwidth) {
  return getTMemCopyPlanRealization(srcTy, dstQuery, shmemLl, cvt, plan,
                                    bitwidth)
      .second;
}

TMemCopyPlanSelection selectTMemCopyPlan(MemDescType srcTy,
                                         const TMemPhysicalQuery &dstQuery,
                                         const LinearLayout &shmemLl,
                                         const LinearLayout &cvt,
                                         ArrayRef<TMemCopyPlan> plans,
                                         int bitwidth) {
  TMemCopyPlanSelection selection;
  for (const TMemCopyPlan &plan : plans) {
    auto [executablePlan, support] =
        getTMemCopyPlanRealization(srcTy, dstQuery, shmemLl, cvt, plan,
                                   bitwidth);
    if (support) {
      assert(executablePlan.has_value() &&
             "supported tcgen05.copy plan must carry a realized schedule");
      selection.plan = std::move(*executablePlan);
      return selection;
    }
    selection.failures.push_back(support);
    if (!selection.firstFailure)
      selection.firstFailure = std::move(support);
  }
  return selection;
}

void attachTMemCopyPlanFailureNotes(InFlightDiagnostic &diag,
                                    const TMemCopyPlanSelection &selection) {
  bool attached = false;
  SmallVector<std::string> attachedMessages;
  for (const TMemCopySupportResult &failure : selection.failures) {
    if (failure.message.empty())
      continue;
    if (llvm::is_contained(attachedMessages, failure.message))
      continue;
    diag.attachNote() << failure.message;
    attachedMessages.push_back(failure.message);
    attached = true;
  }
  if (!attached && selection.firstFailure &&
      !selection.firstFailure->message.empty()) {
    diag.attachNote() << selection.firstFailure->message;
  }
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
  if (!hasOnlyWarpx2SharedSourceDims(shmemLl, kOffset, kBlock))
    return std::nullopt;
  if (shmemLl.hasInDim(kBlock) &&
      !llvm::all_of(shmemLl.getBases().lookup(kBlock), [](ArrayRef<int32_t> b) {
        return llvm::all_of(b, [](int32_t v) { return v == 0; });
      })) {
    return std::nullopt;
  }

  if (!hasCanonicalWarpx2SharedSourceOffsetBases(shmemLl, kOffset))
    return std::nullopt;

  constexpr uint64_t kCanonicalWarpx2DirectSeedModeField = 1ULL << 46;
  constexpr uint64_t kCanonicalWarpx2DirectSeedStrideField = 8ULL << 32;
  constexpr uint64_t kCanonicalWarpx2DirectSeedDescriptorImm =
      kCanonicalWarpx2DirectSeedModeField |
      kCanonicalWarpx2DirectSeedStrideField;
  return kCanonicalWarpx2DirectSeedDescriptorImm;
}

struct TMemCopyWarpx2TwoCTASourceColumnRequirement {
  TMemCopySourceRowSplitRequirement rowSplit;
  unsigned affineSourceRowStrideRows = 8;
  unsigned directSeedSourceOffsetB128 = 32;
};

static std::string getTMemCopyWarpx2TwoCTASourceColumnRequirementNote(
    const TMemCopyWarpx2TwoCTASourceColumnRequirement &requirement) {
  std::string note;
  llvm::raw_string_ostream os(note);
  os << "The two-CTA warpx2::02_13 source-column preservation requirement "
        "remains unsupported until Triton can synthesize a cta_group::2 "
        "descriptor/address schedule that preserves the high source-column "
        "bit. The descriptor path fails because logical row bit "
     << requirement.rowSplit.logicalRowBit
     << " maps to a one-dword source offset rather than an affine "
     << requirement.affineSourceRowStrideRows
     << "-row source stride; the direct-seed cta_group::2 path with source "
        "offset "
     << requirement.directSeedSourceOffsetB128
     << " emits the opcode and writes the correct low destination columns, "
        "but duplicates that low source-column pair into the high destination "
        "columns. Non-zero subaligned destination dword deltas fault, and "
        "aligned deltas that complete the single-CTA schedule read zeros under "
        "cta_group::2. Decomposing this tensor-memory view into cta_group::1 "
        "copies is not valid because two-CTA TMEM allocation uses cta_group::2 "
        "granularity.";
  return os.str();
}

static std::optional<std::string>
getKnownTMemCopySourceRowSplitProbeEvidence(
    MemDescType srcTy, const LinearLayout &cvt, const TMemCopyPlan &plan,
    const TMemCopyMessagePlan &message, int bitwidth,
    const TMemCopySourceRowProjectionFailure &failure) {
  if (plan.family != TMemCopyFamily::Warpx2_02_13_64x128b)
    return std::nullopt;
  if (message.useDirectSeedDescriptor)
    return std::nullopt;
  if (bitwidth != 32 || srcTy.getRank() != 2 || srcTy.getShape()[0] != 256 ||
      srcTy.getShape()[1] != 4)
    return std::nullopt;

  auto *ctx = srcTy.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kOffset = StringAttr::get(ctx, "offset");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!cvt.hasInDim(kRow) || !cvt.hasOutDim(kOffset) ||
      !cvt.hasInDim(kBlock) || cvt.getInDimSize(kBlock) != 2)
    return std::nullopt;
  if (failure.kind !=
          TMemCopySourceRowProjectionFailureKind::NonAffineRowBit ||
      failure.logicalRowBit != llvm::Log2_32(32) ||
      failure.actualOffset != 1)
    return std::nullopt;
  if (failure.sourceRowStride <= 0 ||
      failure.actualOffset == failure.expectedOffset)
    return std::nullopt;

  auto splitRequirement =
      getTMemCopySourceRowSplitRequirement(failure, message);
  if (!splitRequirement)
    return std::nullopt;
  TMemCopyWarpx2TwoCTASourceColumnRequirement requirement;
  requirement.rowSplit = *splitRequirement;
  return getTMemCopyWarpx2TwoCTASourceColumnRequirementNote(requirement);
}

std::optional<TMemCopySourceRowSplitRequirement>
getTMemCopySourceRowSplitRequirement(
    const TMemCopySourceRowProjectionFailure &failure,
    const TMemCopyMessagePlan &message) {
  if (failure.kind != TMemCopySourceRowProjectionFailureKind::NonAffineRowBit ||
      failure.sourceRowStride == 0 ||
      failure.actualOffset == failure.expectedOffset ||
      failure.logicalRowBit >= std::numeric_limits<unsigned>::digits - 1)
    return std::nullopt;

  TMemCopySourceRowSplitRequirement requirement;
  requirement.instructionRows =
      message.instrShape.empty() ? 0 : message.instrShape[0];
  requirement.instructionColumns =
      message.instrShape.size() < 2 ? 0 : message.instrShape[1];
  requirement.logicalRowBit = failure.logicalRowBit;
  requirement.selectedRowRun = 1u << failure.logicalRowBit;
  requirement.rowSelectionPeriod = 1u << (failure.logicalRowBit + 1);
  requirement.actualOffset = failure.actualOffset;
  requirement.expectedOffset = failure.expectedOffset;
  requirement.sourceRowStride = failure.sourceRowStride;
  return requirement;
}

std::optional<TMemCopyDestinationMaskRequirement>
getTMemCopyDestinationMaskRequirement(
    const TMemCopySourceRowSplitRequirement &requirement) {
  if (requirement.selectedRowRun == 0 || requirement.rowSelectionPeriod == 0)
    return std::nullopt;
  return TMemCopyDestinationMaskRequirement{
      /*axis=*/TMemCopyDestinationMaskAxis::Row,
      /*instructionRows=*/requirement.instructionRows,
      /*instructionColumns=*/requirement.instructionColumns,
      /*selectedRun=*/requirement.selectedRowRun,
      /*selectionPeriod=*/requirement.rowSelectionPeriod};
}

static StringRef stringifyTMemCopyDestinationMaskAxis(
    TMemCopyDestinationMaskAxis axis) {
  switch (axis) {
  case TMemCopyDestinationMaskAxis::Row:
    return "row";
  case TMemCopyDestinationMaskAxis::Column:
    return "column";
  }
  llvm_unreachable("unknown TMEM copy destination mask axis");
}

static unsigned getTMemCopyDestinationMaskInstructionFootprint(
    const TMemCopyDestinationMaskRequirement &requirement) {
  switch (requirement.axis) {
  case TMemCopyDestinationMaskAxis::Row:
    return requirement.instructionRows;
  case TMemCopyDestinationMaskAxis::Column:
    return requirement.instructionColumns;
  }
  llvm_unreachable("unknown TMEM copy destination mask axis");
}

static void appendTMemCopyDestinationMaskScheduleGap(
    llvm::raw_ostream &os,
    const TMemCopyDestinationMaskRequirement &requirement,
    StringRef requirementName, StringRef footprintScope) {
  unsigned instructionFootprint =
      getTMemCopyDestinationMaskInstructionFootprint(requirement);
  if (requirement.selectedRun == 0 || requirement.selectionPeriod == 0 ||
      instructionFootprint == 0 ||
      requirement.selectedRun >= instructionFootprint)
    return;

  StringRef axis = stringifyTMemCopyDestinationMaskAxis(requirement.axis);
  os << " The derived " << requirementName
     << " would need to update only " << requirement.selectedRun
     << " of every " << requirement.selectionPeriod << " destination " << axis
     << "s, but this copy atom writes the full " << instructionFootprint << "-"
     << axis << " destination footprint";
  if (!footprintScope.empty())
    os << " " << footprintScope;
  os << ". A multi-message schedule for this split would therefore overwrite "
     << axis
     << "s owned by the complementary split unless the ISA provides a narrower "
        "atom, source format, or destination "
     << axis << " mask.";
}

static std::optional<TMemCopySupportResult>
getTMemCopySourceRowSplitScheduleSupport(
    MemDescType srcTy, const LinearLayout &cvt, const TMemCopyPlan &plan,
    unsigned messageIdx, const TMemCopyMessagePlan &message, int bitwidth,
    const TMemCopySourceRowProjectionFailure &failure,
    StringRef rowProjectionError) {
  auto splitRequirement =
      getTMemCopySourceRowSplitRequirement(failure, message);
  if (!splitRequirement)
    return std::nullopt;
  auto maskRequirement =
      getTMemCopyDestinationMaskRequirement(*splitRequirement);

  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "tcgen05.copy." << stringifyTMemCopyFamily(plan.family)
     << " descriptor message " << messageIdx
     << " has an unsupported source-row projection. "
     << rowProjectionError
     << " The derived row-selected source-offset requirement would need "
        "logical row bit "
     << splitRequirement->logicalRowBit << " to select shared offset "
     << splitRequirement->actualOffset << " instead of affine row-stride "
     << splitRequirement->expectedOffset << " for ";
  if (maskRequirement) {
    os << maskRequirement->selectedRun << "-row destination runs every "
       << maskRequirement->selectionPeriod << " rows";
  } else {
    os << splitRequirement->selectedRowRun
       << "-row destination runs every "
       << splitRequirement->rowSelectionPeriod << " rows";
  }
  if (maskRequirement && maskRequirement->instructionRows > 0) {
    os << " within the "
       << maskRequirement->instructionRows
       << "-row copy-instruction footprint";
  }
  os << ". Current tcgen05.copy scheduling can change the shared descriptor "
        "or source address per emitted instruction, but it has no proved "
        "row-selected source-offset schedule.";
  if (maskRequirement) {
    appendTMemCopyDestinationMaskScheduleGap(
        os, *maskRequirement, "row-selected source-offset requirement",
        "for each emitted instruction");
  }

  if (auto knownGap = getKnownTMemCopySourceRowSplitProbeEvidence(
          srcTy, cvt, plan, message, bitwidth, failure))
    os << " " << *knownGap;

  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

std::optional<TMemCopySourceRowProjection>
getTMemCopySourceRowProjectionPlan(const LinearLayout &cvt,
                                   const TMemCopyMessagePlan &message,
                                   std::string *error,
                                   TMemCopySourceRowProjectionFailure *failure) {
  TMemCopySourceRowProjection projection;
  const TMemCopyAtom &atom = message.atom;
  if (message.useDirectSeedDescriptor || atom.nRow == 4)
    return projection;

  auto inDims = cvt.getInDimNames();
  if (inDims.empty()) {
    if (error)
      *error = "tcgen05.copy source row projection has no input dimensions.";
    return std::nullopt;
  }
  auto *ctx = inDims.begin()->getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kOffset = StringAttr::get(ctx, "offset");
  if (!cvt.hasInDim(kRow) || !cvt.hasOutDim(kOffset)) {
    if (failure)
      failure->kind = TMemCopySourceRowProjectionFailureKind::MissingDimensions;
    if (error) {
      *error =
        "tcgen05.copy source row projection requires row and offset "
        "dimensions.";
    }
    return std::nullopt;
  }

  unsigned strideBit = llvm::Log2_32(8);
  if (strideBit >= cvt.getInDimSizeLog2(kRow)) {
    if (failure)
      failure->kind = TMemCopySourceRowProjectionFailureKind::MissingStride;
    if (error) {
      *error = "tcgen05.copy source row projection requires the 8-row source "
               "stride to be explicit in the row dimension.";
    }
    return std::nullopt;
  }
  projection.sourceRowStride = cvt.getBasis(kRow, strideBit, kOffset);

  auto appendAffineRowStep = [&](unsigned rowBit, int32_t expectedMultiplier,
                                 StringRef message) -> bool {
    int32_t expectedOffset = projection.sourceRowStride * expectedMultiplier;
    int32_t actualOffset = 0;
    bool hasRowBit = rowBit < cvt.getInDimSizeLog2(kRow);
    if (hasRowBit)
      actualOffset = cvt.getBasis(kRow, rowBit, kOffset);
    if (!hasRowBit || actualOffset != expectedOffset) {
      if (failure) {
        failure->kind =
            TMemCopySourceRowProjectionFailureKind::NonAffineRowBit;
        failure->logicalRowBit = rowBit;
        failure->actualOffset = actualOffset;
        failure->expectedOffset = expectedOffset;
        failure->sourceRowStride = projection.sourceRowStride;
      }
      if (error)
        *error = message.str();
      return false;
    }
    projection.steps.push_back(TMemCopySourceRowProjectionStep{
        rowBit, cvt.getBasis(kRow, rowBit, kOffset)});
    return true;
  };

  if ((atom.multicast & 1) == 0 &&
      !appendAffineRowStep(
          llvm::Log2_32(32), 32 / 8,
          "tcgen05.copy source row projection requires logical row bit 5 to "
          "remain an affine multiple of the 8-row source stride for this copy "
          "atom."))
    return std::nullopt;
  if (atom.multicast != 1 && (atom.multicast & 2) == 0 &&
      !appendAffineRowStep(
          llvm::Log2_32(64), 64 / 8,
          "tcgen05.copy source row projection requires logical row bit 6 to "
          "remain an affine multiple of the 8-row source stride for this copy "
          "atom."))
    return std::nullopt;

  return projection;
}

TMemCopySupportResult
getTMemCopySourceRowProjectionSupport(const LinearLayout &cvt,
                                      const TMemCopyMessagePlan &message) {
  std::string error;
  if (!getTMemCopySourceRowProjectionPlan(cvt, message, &error)) {
    return getUnsupportedTMemCopyResult(
        TMemCopySupportFailureLayer::InstructionSchedule, error);
  }
  return getSupportedTMemCopyResult();
}

llvm::SmallVector<TMemCopyPlan, 4> getTMemCopyPlans(const LinearLayout &cvt,
                                                    int bitwidth) {
  auto atom = getTMemCopyAtom(cvt, bitwidth);
  if (!atom)
    return {};

  auto *ctx = cvt.getInDimNames().begin()->getContext();
  auto kBlock = StringAttr::get(ctx, "block");

  auto makePlanForAtom = [&](const TMemCopyAtom &planAtom,
                             ArrayRef<std::tuple<unsigned, unsigned, unsigned,
                                                 int>>
                                 messageSpecs) {
    TMemCopyPlan plan;
    plan.family = getTMemCopyFamily(planAtom);
    for (auto [descriptorRows, sourceWarpGroups, instrRows, smemRow] :
         messageSpecs) {
      TMemCopyMessagePlan message;
      message.atom = planAtom;
      message.descriptorRows = descriptorRows;
      message.sourceWarpGroups = sourceWarpGroups;
      message.descriptorShape = {
          descriptorRows, static_cast<unsigned>(planAtom.bCol / bitwidth)};
      message.instrShape = {instrRows,
                            static_cast<unsigned>(planAtom.bCol / bitwidth)};
      message.smemRow = smemRow;
      plan.messages.push_back(std::move(message));
    }
    return plan;
  };
  auto makePlan = [&](ArrayRef<std::tuple<unsigned, unsigned, unsigned, int>>
                          messageSpecs) {
    return makePlanForAtom(*atom, messageSpecs);
  };

  llvm::SmallVector<TMemCopyPlan, 4> plans;
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
    if (bitwidth < 32)
      appendWarpx2Plan(/*descriptorRows=*/64u, /*sourceWarpGroups=*/2u,
                       /*directSeed=*/false, /*tmemDwordDelta=*/0,
                       /*directSourceOffsetB128=*/0);
    return plans;
  }
  if (atom->multicast == 2) {
    bool isSingleCTA = !cvt.hasInDim(kBlock) || cvt.getInDimSize(kBlock) == 1;
    if (isSingleCTA) {
      appendWarpx2Plan(/*descriptorRows=*/64u, /*sourceWarpGroups=*/2u,
                       /*directSeed=*/true, /*tmemDwordDelta=*/0,
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
    auto plan = makePlan({std::tuple{4u, 1u, 4u, 0}});
    if (auto descriptorCvt =
            getTMemCopy4x256RefreshDescriptorCvt(cvt, bitwidth)) {
      assert(plan.family == TMemCopyFamily::Dense4x256b);
      auto descriptorProjection = *descriptorCvt;
      plan.messages.front().descriptorCvt = descriptorProjection;
      TMemCopyMessagePlan highColumns = plan.messages.front();
      highColumns.smemColOffset = 4;
      highColumns.tmemDwordDelta = 4;
      plan.messages.push_back(std::move(highColumns));
    }
    plans.push_back(std::move(plan));
    return plans;
  }

  auto appendDensePlan = [&](const TMemCopyAtom &planAtom) {
    plans.push_back(makePlanForAtom(planAtom, {std::tuple{32u, 4u, 32u, 0}}));
    // Dense families sometimes admit a 64x2 descriptor factorization in
    // addition to the canonical 32x4 split. Keep the canonical plan first and
    // only fall back to the 64x2 variant when descriptor synthesis rejects the
    // canonical one.
    if (planAtom.multicast == 0 && planAtom.nRow == 128)
      plans.push_back(
          makePlanForAtom(planAtom, {std::tuple{64u, 2u, 64u, 0}}));
  };

  appendDensePlan(*atom);
  // A narrower dense copy atom can avoid shared-descriptor swizzle boundaries
  // that a wider atom would cross inside one instruction.
  if (atom->multicast == 0 && atom->nRow == 128 && atom->bCol > 128)
    appendDensePlan(TMemCopyAtom{128, 128, 0});
  return plans;
}

llvm::SmallVector<LinearLayout>
getTMemCopyDescriptorLayouts(MemDescType srcTy,
                             const LinearLayout &shmemLl,
                             const LinearLayout &cvt,
                             const TMemCopyMessagePlan &message) {
  const LinearLayout &descriptorCvt =
      message.descriptorCvt ? *message.descriptorCvt : cvt;
  assert(to_vector(descriptorCvt.getOutDimNames()) ==
         to_vector(cvt.getOutDimNames()) &&
         "tcgen05.copy descriptor projection must target the same source "
         "address space as the full copy conversion");
  auto inDims = descriptorCvt.getInDimNames();
  assert(!inDims.empty());
  auto *ctx = inDims.begin()->getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  auto kCol = StringAttr::get(ctx, "col");
  auto kRow = StringAttr::get(ctx, "row");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto makeLayout = [&](unsigned descriptorRows, unsigned sourceWarpGroups,
                        unsigned descriptorCols) -> std::optional<LinearLayout> {
    int64_t reshapeVolume = static_cast<int64_t>(descriptorRows) *
                            sourceWarpGroups * descriptorCols *
                            descriptorCvt.getInDimSize(kBlock);
    if (descriptorCvt.getTotalInDimSize() != reshapeVolume)
      return std::nullopt;
    return descriptorCvt
        .reshapeIns({{kRow, static_cast<int32_t>(descriptorRows)},
                     {kWarp, static_cast<int32_t>(sourceWarpGroups)},
                     {kCol, static_cast<int32_t>(descriptorCols)},
                     {kBlock, descriptorCvt.getInDimSize(kBlock)}})
        .sublayout({kRow, kCol}, to_vector(descriptorCvt.getOutDimNames()));
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
          if (auto layout = makeLayout(message.descriptorRows << rowFoldBits,
                                       residualWarpGroups,
                                       descriptorCvt.getInDimSize(kCol)
                                           << colFoldBits))
            pushUnique(*layout);
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
  if (auto layout = makeLayout(message.descriptorRows,
                               message.sourceWarpGroups,
                               descriptorCvt.getInDimSize(kCol)))
    pushUnique(*layout);
  auto directDataLayout = shmemLl.pseudoinvert();
  if (directDataLayout.getNumInDims() == 2)
    pushUnique(directDataLayout);
  if (auto directSharedLayout =
          makeSharedSeedLayout(message.descriptorRows,
                               message.sourceWarpGroups,
                               descriptorCvt.getInDimSize(kCol))) {
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

static std::optional<TMemCopyDescriptorLayoutSelection>
selectTMemCopyDescriptorLayout(ArrayRef<LinearLayout> srcDescLayouts,
                               ArrayRef<unsigned> descriptorShape,
                               TMemCopyFamily family, int bitwidth) {
  bool allowTransposed = family == TMemCopyFamily::Dense4x256b;
  static constexpr unsigned kDescriptorOrientations[] = {0u, 1u};
  for (const LinearLayout &srcDescLayout : srcDescLayouts) {
    for (unsigned mnDim : kDescriptorOrientations) {
      if (canRepresentAsMMASmemDescriptor(srcDescLayout, descriptorShape,
                                          bitwidth, mnDim, 5,
                                          allowTransposed)) {
        return TMemCopyDescriptorLayoutSelection{srcDescLayout, mnDim};
      }
    }
  }
  return std::nullopt;
}

static StringRef stringifyTMemCopyInstructionColumnProjectionFailureKind(
    TMemCopyInstructionColumnProjectionFailureKind kind) {
  switch (kind) {
  case TMemCopyInstructionColumnProjectionFailureKind::None:
    return "none";
  case TMemCopyInstructionColumnProjectionFailureKind::PackedLaneState:
    return "packed-lane-state";
  case TMemCopyInstructionColumnProjectionFailureKind::NonContiguousOffset:
    return "non-contiguous-offset";
  case TMemCopyInstructionColumnProjectionFailureKind::DescriptorRowStrideSelection:
    return "descriptor-row-stride-selection";
  case TMemCopyInstructionColumnProjectionFailureKind::NonOffsetComponent:
    return "non-offset-component";
  }
  llvm_unreachable("unknown tcgen05.copy instruction-column failure kind");
}

std::optional<TMemCopyDescriptorRowSplitRequirement>
getTMemCopyDescriptorRowSplitRequirement(
    const TMemCopyInstructionColumnProjectionFailure &failure) {
  if (failure.kind != TMemCopyInstructionColumnProjectionFailureKind::
                          DescriptorRowStrideSelection ||
      !failure.descriptorRowStride || !failure.descriptorRowDelta ||
      *failure.descriptorRowStride == 0)
    return std::nullopt;
  if (failure.logicalColBit >= std::numeric_limits<unsigned>::digits - 1)
    return std::nullopt;

  TMemCopyDescriptorRowSplitRequirement requirement;
  requirement.instructionRows = failure.instructionRows;
  requirement.instructionColumns = failure.instructionColumns;
  requirement.logicalColBit = failure.logicalColBit;
  requirement.selectedColumnRun = 1u << failure.logicalColBit;
  requirement.columnSelectionPeriod = 1u << (failure.logicalColBit + 1);
  requirement.actualOffset = failure.actualOffset;
  requirement.expectedOffset = failure.expectedOffset;
  requirement.descriptorRowStride = *failure.descriptorRowStride;
  requirement.descriptorRowDelta = *failure.descriptorRowDelta;
  requirement.spansInstructionRows =
      failure.descriptorRowDeltaSpansInstructionRows;
  return requirement;
}

std::optional<TMemCopyInstructionColumnPermutationRequirement>
getTMemCopyInstructionColumnPermutationRequirement(
    const TMemCopyInstructionColumnProjectionFailure &failure) {
  if (failure.kind !=
          TMemCopyInstructionColumnProjectionFailureKind::NonContiguousOffset ||
      failure.actualOffset == failure.expectedOffset)
    return std::nullopt;
  auto runAndPeriod =
      getTMemCopyColumnSelectionRunAndPeriod(failure.logicalColBit);
  if (!runAndPeriod)
    return std::nullopt;

  TMemCopyInstructionColumnPermutationRequirement requirement;
  requirement.instructionRows = failure.instructionRows;
  requirement.instructionColumns = failure.instructionColumns;
  requirement.logicalColBit = failure.logicalColBit;
  requirement.selectedColumnRun = runAndPeriod->first;
  requirement.columnSelectionPeriod = runAndPeriod->second;
  requirement.actualOffset = failure.actualOffset;
  requirement.expectedOffset = failure.expectedOffset;
  return requirement;
}

std::optional<TMemCopyDestinationMaskRequirement>
getTMemCopyDestinationMaskRequirement(
    const TMemCopyDescriptorRowSplitRequirement &requirement) {
  if (requirement.selectedColumnRun == 0 ||
      requirement.columnSelectionPeriod == 0)
    return std::nullopt;
  return TMemCopyDestinationMaskRequirement{
      /*axis=*/TMemCopyDestinationMaskAxis::Column,
      /*instructionRows=*/requirement.instructionRows,
      /*instructionColumns=*/requirement.instructionColumns,
      /*selectedRun=*/requirement.selectedColumnRun,
      /*selectionPeriod=*/requirement.columnSelectionPeriod};
}

std::optional<TMemCopyDestinationMaskRequirement>
getTMemCopyDestinationMaskRequirement(
    const TMemCopyInstructionColumnPermutationRequirement &requirement) {
  if (requirement.selectedColumnRun == 0 ||
      requirement.columnSelectionPeriod == 0)
    return std::nullopt;
  return TMemCopyDestinationMaskRequirement{
      /*axis=*/TMemCopyDestinationMaskAxis::Column,
      /*instructionRows=*/requirement.instructionRows,
      /*instructionColumns=*/requirement.instructionColumns,
      /*selectedRun=*/requirement.selectedColumnRun,
      /*selectionPeriod=*/requirement.columnSelectionPeriod};
}

std::optional<TMemCopyPackedLaneRequirement> getTMemCopyPackedLaneRequirement(
    const TMemCopyInstructionColumnProjectionFailure &failure) {
  if (failure.kind !=
          TMemCopyInstructionColumnProjectionFailureKind::PackedLaneState ||
      failure.packedLaneBits == 0 ||
      failure.packedLaneBits >= std::numeric_limits<unsigned>::digits)
    return std::nullopt;

  TMemCopyPackedLaneRequirement requirement;
  requirement.instructionRows = failure.instructionRows;
  requirement.instructionColumns = failure.instructionColumns;
  requirement.laneBits = failure.packedLaneBits;
  requirement.lanesPerDword = 1u << failure.packedLaneBits;
  requirement.physicalInstructionColumns =
      failure.instructionColumns / requirement.lanesPerDword;
  if (failure.packedLaneProjection) {
    requirement.hasPhysicalProjection = true;
    requirement.laneBits = failure.packedLaneProjection->laneBits;
    requirement.lanesPerDword = failure.packedLaneProjection->lanesPerDword;
    requirement.physicalInstructionColumns =
        failure.packedLaneProjection->physicalInstructionColumns;
  }
  return requirement;
}

static std::string formatTMemCopyInstructionColumnProjectionFailure(
    const TMemCopyInstructionColumnProjectionFailure &failure) {
  std::string note;
  llvm::raw_string_ostream os(note);
  if (failure.kind ==
      TMemCopyInstructionColumnProjectionFailureKind::PackedLaneState) {
    os << "Within one " << failure.instructionColumns
       << "-column tcgen05.copy instruction, source column bit 0 maps to no "
          "shared offset. This projection carries sub-32-bit packed lane "
          "state outside the LinearLayout offset dimension";
    if (auto packedLaneRequirement =
            getTMemCopyPackedLaneRequirement(failure)) {
      os << " (" << packedLaneRequirement->laneBits << " lane bit"
         << (packedLaneRequirement->laneBits == 1 ? "" : "s") << "; "
         << packedLaneRequirement->lanesPerDword
         << " logical source columns per 32-bit shared-memory word)";
    }
    if (failure.packedLaneProjection) {
      os << "; after those lane bits, the physical dword-column projection is "
            "contiguous over "
         << failure.packedLaneProjection->physicalInstructionColumns
         << " column"
         << (failure.packedLaneProjection->physicalInstructionColumns == 1 ? ""
                                                                          : "s");
    }
    os << ". Current copy scheduling cannot synthesize packed-lane tcgen05.copy "
          "descriptors or tile extents from that projection; the planner needs "
          "a lane-aware physical query that separates physical dword columns "
          "from packed sub-dword lanes. Use an unpacked "
          "TensorMemoryLinearLayout for dense subword copies, or a "
          "tmem.store/tmem.load path until packed lane copy semantics are "
          "modeled explicitly.";
    return os.str();
  }

  os << "Within one " << failure.instructionColumns
     << "-column tcgen05.copy instruction, source column bit "
     << failure.logicalColBit << " maps to ";
  if (failure.actualOffset == 0)
    os << "no shared offset";
  else
    os << "shared offset " << failure.actualOffset;
  if (failure.hasNonOffsetContribution)
    os << " plus a non-offset component";
  if (auto splitRequirement =
          getTMemCopyDescriptorRowSplitRequirement(failure)) {
    os << " (" << splitRequirement->descriptorRowDelta
       << " descriptor-row stride"
       << (splitRequirement->descriptorRowDelta == 1 ? "" : "s")
       << "), which would require this column bit to select descriptor row +"
       << splitRequirement->descriptorRowDelta << " for "
       << splitRequirement->selectedColumnRun
       << "-column destination runs every "
       << splitRequirement->columnSelectionPeriod
       << " columns within the same " << failure.instructionColumns
       << "-column instruction";
    if (splitRequirement->spansInstructionRows) {
      os << "; that row delta spans a "
         << splitRequirement->instructionRows
         << "-row source footprint";
      if (splitRequirement->instructionRows > 0)
        os << ", which is an entire tcgen05.copy instruction row footprint";
    }
  }
  if (auto permutationRequirement =
          getTMemCopyInstructionColumnPermutationRequirement(failure)) {
    os << ", which would require this source-column bit to select shared "
          "offset "
       << permutationRequirement->actualOffset << " for "
       << permutationRequirement->selectedColumnRun
       << "-column destination runs every "
       << permutationRequirement->columnSelectionPeriod
       << " columns within the same " << failure.instructionColumns
       << "-column instruction";
  }
  os << " instead of contiguous shared offset " << failure.expectedOffset
     << ". Public tcgen05.copy takes one tensor-memory address and one shared "
        "descriptor per instruction and has no per-column destination mask, "
        "so this projection needs a different copy atom, source format, or a "
        "proven multi-message schedule that avoids overwriting unrelated "
        "destination columns before it can be supported.";
  return os.str();
}

static std::optional<TMemCopySupportResult>
getTMemCopyDescriptorRowSplitScheduleSupport(
    TMemCopyFamily family, unsigned messageIdx,
    const TMemCopyInstructionColumnProjectionFailure &failure,
    StringRef instructionProjectionError) {
  auto splitRequirement = getTMemCopyDescriptorRowSplitRequirement(failure);
  if (!splitRequirement)
    return std::nullopt;

  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "tcgen05.copy." << stringifyTMemCopyFamily(family)
     << " descriptor message " << messageIdx
     << " has an unsupported instruction-column projection. "
     << instructionProjectionError;
  auto maskRequirement =
      getTMemCopyDestinationMaskRequirement(*splitRequirement);
  if (maskRequirement)
    appendTMemCopyDestinationMaskScheduleGap(
        os, *maskRequirement, "descriptor-row split",
        "for each descriptor row");
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

static std::optional<TMemCopySupportResult>
getTMemCopyInstructionColumnPermutationScheduleSupport(
    TMemCopyFamily family, unsigned messageIdx,
    const TMemCopyInstructionColumnProjectionFailure &failure,
    StringRef instructionProjectionError) {
  auto permutationRequirement =
      getTMemCopyInstructionColumnPermutationRequirement(failure);
  if (!permutationRequirement)
    return std::nullopt;

  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "tcgen05.copy." << stringifyTMemCopyFamily(family)
     << " descriptor message " << messageIdx
     << " has an unsupported instruction-column projection. "
     << instructionProjectionError;
  os << " The derived source-column permutation requirement would need "
        "logical column bit "
     << permutationRequirement->logicalColBit << " to select shared offset "
     << permutationRequirement->actualOffset
     << " instead of contiguous shared offset "
     << permutationRequirement->expectedOffset << " for "
     << permutationRequirement->selectedColumnRun
     << "-column destination runs every "
     << permutationRequirement->columnSelectionPeriod
     << " columns within the "
     << permutationRequirement->instructionColumns
     << "-column copy-instruction footprint. Current tcgen05.copy scheduling "
        "can change the shared descriptor or source address per emitted "
        "instruction, but it has no proved source-column-selected source-offset "
        "schedule inside one instruction.";
  if (auto maskRequirement =
          getTMemCopyDestinationMaskRequirement(*permutationRequirement)) {
    appendTMemCopyDestinationMaskScheduleGap(
        os, *maskRequirement, "source-column permutation requirement",
        "for each emitted instruction");
  }
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

static std::optional<TMemCopySupportResult>
getTMemCopyPackedLaneScheduleSupport(
    TMemCopyFamily family, unsigned messageIdx,
    const TMemCopyInstructionColumnProjectionFailure &failure,
    StringRef instructionProjectionError) {
  auto packedLaneRequirement = getTMemCopyPackedLaneRequirement(failure);
  if (!packedLaneRequirement)
    return std::nullopt;

  std::string reason;
  llvm::raw_string_ostream os(reason);
  os << "tcgen05.copy." << stringifyTMemCopyFamily(family)
     << " descriptor message " << messageIdx
     << " has an unsupported instruction-column projection. "
     << instructionProjectionError;
  os << " The derived packed-lane source-storage requirement has "
     << packedLaneRequirement->laneBits << " lane bit"
     << (packedLaneRequirement->laneBits == 1 ? "" : "s") << " and "
     << packedLaneRequirement->lanesPerDword
     << " logical source columns per 32-bit shared-memory word.";
  if (packedLaneRequirement->hasPhysicalProjection) {
    os << " The high column bits form a contiguous physical dword-column "
          "stream over "
       << packedLaneRequirement->physicalInstructionColumns << " dword column"
       << (packedLaneRequirement->physicalInstructionColumns == 1 ? ""
                                                                  : "s")
       << " for the " << packedLaneRequirement->instructionColumns
       << "-column logical instruction.";
  }
  os << " A descriptor/tile schedule that drops those lane bits can address "
        "the dword stream but aliases the packed lanes, so it would copy only "
        "one lane from each packed source word instead of all logical source "
        "columns. Support needs an explicit packed source-storage model that "
        "carries lane selection through MMAShared descriptor synthesis, source "
        "footprint planning, and the tcgen05.copy instruction schedule.";
  return getUnsupportedTMemCopyResult(
      TMemCopySupportFailureLayer::InstructionSchedule, os.str());
}

static std::optional<TMemCopyPackedLaneProjection>
getTMemCopyPackedLaneProjectionPlan(const LinearLayout &descriptorCvt,
                                    StringAttr colDim, StringAttr offsetDim,
                                    unsigned instrCols, int bitwidth) {
  if (bitwidth <= 0 || bitwidth >= 32 || 32 % bitwidth != 0 ||
      !descriptorCvt.hasInDim(colDim) || !descriptorCvt.hasOutDim(offsetDim) ||
      !llvm::isPowerOf2_32(instrCols) || instrCols == 0)
    return std::nullopt;

  unsigned laneBits = llvm::Log2_32(32 / bitwidth);
  unsigned instrColBits = llvm::Log2_32(instrCols);
  if (laneBits == 0 || laneBits >= instrColBits)
    return std::nullopt;

  auto colBases = descriptorCvt.getBases().lookup(colDim);
  if (colBases.size() < instrColBits)
    return std::nullopt;

  auto isAllZero = [](ArrayRef<int32_t> basis) {
    return llvm::all_of(basis, [](int32_t value) { return value == 0; });
  };
  unsigned offsetDimIndex = descriptorCvt.getOutDimIndex(offsetDim);
  auto isPureOffset = [&](ArrayRef<int32_t> basis, int32_t expectedOffset) {
    for (auto [idx, value] : llvm::enumerate(basis)) {
      int32_t expected = idx == offsetDimIndex ? expectedOffset : 0;
      if (value != expected)
        return false;
    }
    return true;
  };

  TMemCopyPackedLaneProjection projection;
  projection.laneBits = laneBits;
  projection.lanesPerDword = 1u << laneBits;
  projection.physicalInstructionColumns = instrCols >> laneBits;
  for (unsigned bit = 0; bit < laneBits; ++bit) {
    if (!isAllZero(descriptorCvt.getBasis(colDim, bit)))
      return std::nullopt;
  }
  for (unsigned bit = laneBits; bit < instrColBits; ++bit) {
    int32_t expectedOffset = 1 << (bit - laneBits);
    if (!isPureOffset(descriptorCvt.getBasis(colDim, bit), expectedOffset))
      return std::nullopt;
    projection.physicalSteps.push_back(
        TMemCopyInstructionColumnProjectionStep{bit, expectedOffset});
  }
  return projection;
}

std::optional<TMemCopyInstructionColumnProjection>
getTMemCopyInstructionColumnProjectionPlan(
    const LinearLayout &cvt, const TMemCopyMessagePlan &message, int bitwidth,
    std::string *error,
    TMemCopyInstructionColumnProjectionFailure *failure) {
  if (failure)
    *failure = TMemCopyInstructionColumnProjectionFailure{};
  TMemCopyInstructionColumnProjection projection;
  const LinearLayout &descriptorCvt =
      message.descriptorCvt ? *message.descriptorCvt : cvt;
  auto inDims = descriptorCvt.getInDimNames();
  if (inDims.empty() || message.instrShape.size() < 2)
    return projection;

  auto *ctx = inDims.begin()->getContext();
  auto kCol = StringAttr::get(ctx, "col");
  auto kRow = StringAttr::get(ctx, "row");
  auto kOffset = StringAttr::get(ctx, "offset");
  if (!descriptorCvt.hasInDim(kCol) || !descriptorCvt.hasOutDim(kOffset))
    return projection;

  unsigned instrCols = message.instrShape[1];
  projection.instructionColumns = instrCols;
  if (!llvm::isPowerOf2_32(instrCols) || instrCols <= 1)
    return projection;
  unsigned instrColBits = llvm::Log2_32(instrCols);

  auto colBases = descriptorCvt.getBases().lookup(kCol);
  if (colBases.size() < instrColBits || colBases.empty())
    return projection;

  int32_t unitOffset = descriptorCvt.getBasis(kCol, 0, kOffset);
  projection.unitSourceOffset = unitOffset;
  if (unitOffset == 0 && bitwidth < 32) {
    TMemCopyInstructionColumnProjectionFailure failureInfo;
    failureInfo.kind =
        TMemCopyInstructionColumnProjectionFailureKind::PackedLaneState;
    failureInfo.instructionRows =
        message.instrShape.empty() ? 0 : message.instrShape[0];
    failureInfo.instructionColumns = instrCols;
    failureInfo.logicalColBit = 0;
    failureInfo.actualOffset = 0;
    failureInfo.expectedOffset = 1;
    if (bitwidth > 0 && bitwidth < 32 && 32 % bitwidth == 0)
      failureInfo.packedLaneBits = llvm::Log2_32(32 / bitwidth);
    failureInfo.packedLaneProjection = getTMemCopyPackedLaneProjectionPlan(
        descriptorCvt, kCol, kOffset, instrCols, bitwidth);
    if (failureInfo.packedLaneProjection)
      failureInfo.packedLaneBits =
          failureInfo.packedLaneProjection->laneBits;
    if (failure)
      *failure = failureInfo;
    if (error)
      *error = formatTMemCopyInstructionColumnProjectionFailure(failureInfo);
    return std::nullopt;
  }
  if (unitOffset <= 0)
    return projection;
  unsigned offsetDimIndex = descriptorCvt.getOutDimIndex(kOffset);
  std::optional<int32_t> descriptorRowStride;
  if (descriptorCvt.hasInDim(kRow) &&
      descriptorCvt.getInDimSizeLog2(kRow) > 0) {
    int32_t stride = descriptorCvt.getBasis(kRow, 0, kOffset);
    if (stride > 0)
      descriptorRowStride = stride;
  }

  auto hasNonOffsetContribution = [&](ArrayRef<int32_t> basis) {
    for (auto [idx, value] : llvm::enumerate(basis)) {
      if (idx != offsetDimIndex && value != 0)
        return true;
    }
    return false;
  };

  for (unsigned bit = 0; bit < instrColBits; ++bit) {
    ArrayRef<int32_t> basis = descriptorCvt.getBasis(kCol, bit);
    int32_t actualOffset = basis[offsetDimIndex];
    int32_t expectedOffset = unitOffset << bit;
    bool nonOffsetContribution = hasNonOffsetContribution(basis);
    if (actualOffset == expectedOffset && !nonOffsetContribution) {
      projection.steps.push_back(
          TMemCopyInstructionColumnProjectionStep{bit, actualOffset});
      continue;
    }

    TMemCopyInstructionColumnProjectionFailure failureInfo;
    failureInfo.instructionRows =
        message.instrShape.empty() ? 0 : message.instrShape[0];
    failureInfo.instructionColumns = instrCols;
    failureInfo.logicalColBit = bit;
    failureInfo.actualOffset = actualOffset;
    failureInfo.expectedOffset = expectedOffset;
    failureInfo.descriptorRowStride = descriptorRowStride;
    failureInfo.hasNonOffsetContribution = nonOffsetContribution;
    if (nonOffsetContribution) {
      failureInfo.kind =
          TMemCopyInstructionColumnProjectionFailureKind::NonOffsetComponent;
    } else if (descriptorRowStride && actualOffset > 0 &&
               actualOffset % *descriptorRowStride == 0 &&
               actualOffset != expectedOffset) {
      failureInfo.kind = TMemCopyInstructionColumnProjectionFailureKind::
          DescriptorRowStrideSelection;
      failureInfo.descriptorRowDelta = actualOffset / *descriptorRowStride;
      failureInfo.descriptorRowDeltaSpansInstructionRows =
          failureInfo.descriptorRowDelta &&
          *failureInfo.descriptorRowDelta >=
              static_cast<int32_t>(message.instrShape[0]);
    } else {
      failureInfo.kind =
          TMemCopyInstructionColumnProjectionFailureKind::NonContiguousOffset;
    }
    if (failure)
      *failure = failureInfo;
    if (error)
      *error = formatTMemCopyInstructionColumnProjectionFailure(failureInfo);
    return std::nullopt;
  }
  return projection;
}

static std::optional<std::string> getTMemCopyInstructionColumnProjectionNote(
    const LinearLayout &cvt, const TMemCopyMessagePlan &message,
    int bitwidth) {
  std::string error;
  if (!getTMemCopyInstructionColumnProjectionPlan(cvt, message, bitwidth,
                                                  &error))
    return error;
  return std::nullopt;
}

static std::optional<std::string>
getTMemCopySubwordSourceStorageNote(const LinearLayout &cvt,
                                    const TMemCopyMessagePlan &message,
                                    int bitwidth) {
  if (bitwidth >= 32 || message.instrShape.size() < 2)
    return std::nullopt;
  auto inDims = cvt.getInDimNames();
  if (inDims.empty())
    return std::nullopt;
  auto *ctx = inDims.begin()->getContext();
  auto kCol = StringAttr::get(ctx, "col");
  if (!cvt.hasInDim(kCol))
    return std::nullopt;
  auto footprintColumns = getTMemCopyEffectiveInstructionColumns(message);
  if (!footprintColumns)
    return std::nullopt;
  int64_t logicalSourceColumns = cvt.getInDimSize(kCol);
  if (*footprintColumns <= static_cast<uint64_t>(logicalSourceColumns))
    return std::nullopt;

  std::string note;
  llvm::raw_string_ostream os(note);
  os << "The subword tcgen05.copy instruction source footprint spans "
     << *footprintColumns << " logical element columns, but the linear source "
     << "view exposes only " << logicalSourceColumns
     << " column(s). The extra sub-dword lanes must be represented by a "
        "packed source-storage or descriptor semantic-equivalence model before "
        "this schedule can be lowered safely; descriptor footprint coverage "
        "alone is not a correctness proof.";
  return os.str();
}

std::optional<TMemCopyDescriptorLayoutSelection>
selectTMemCopyDescriptorLayout(gpu::MemDescType srcTy,
                               const LinearLayout &shmemLl,
                               const LinearLayout &cvt,
                               const TMemCopyMessagePlan &message,
                               TMemCopyFamily family, int bitwidth) {
  return selectTMemCopyDescriptorLayout(
      getTMemCopyDescriptorLayouts(srcTy, shmemLl, cvt, message),
      message.descriptorShape, family, bitwidth);
}

TMemCopySupportResult
getTMemCopySharedDescriptorPlanSupport(gpu::MemDescType srcTy,
                                       const LinearLayout &shmemLl,
                                       const LinearLayout &cvt,
                                       const TMemCopyPlan &plan,
                                       int bitwidth) {
  return getTMemCopySharedDescriptorPlanRealization(srcTy, shmemLl, cvt, plan,
                                                    bitwidth)
      .second;
}

static std::pair<std::optional<TMemCopyExecutablePlan>, TMemCopySupportResult>
getTMemCopySharedDescriptorPlanRealization(gpu::MemDescType srcTy,
                                           const LinearLayout &shmemLl,
                                           const LinearLayout &cvt,
                                           const TMemCopyPlan &plan,
                                           int bitwidth) {
  TMemCopyExecutablePlan executablePlan;
  executablePlan.family = plan.family;
  bool debugTMemQuery = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  if (plan.family == TMemCopyFamily::Dense4x256b &&
      llvm::any_of(plan.messages, [](const TMemCopyMessagePlan &message) {
        return !message.descriptorCvt.has_value();
      })) {
    return {std::nullopt,
            getUnsupportedTMemCopy4x256RefreshImageResult(
                TMemCopySupportFailureLayer::DescriptorSynthesis)};
  }
  for (auto [messageIdx, message] : llvm::enumerate(plan.messages)) {
    TMemCopyScheduledMessage scheduledMessage;
    scheduledMessage.plan = message;
    auto sourceFormatSupport =
        getTMemCopySourceFormatSupport(message, bitwidth);
    if (!sourceFormatSupport)
      return {std::nullopt, sourceFormatSupport};
    std::string rowProjectionError;
    TMemCopySourceRowProjectionFailure rowProjectionFailure;
    auto rowProjection =
        getTMemCopySourceRowProjectionPlan(cvt, message, &rowProjectionError,
                                           &rowProjectionFailure);
    if (!rowProjection) {
      if (auto splitSupport = getTMemCopySourceRowSplitScheduleSupport(
              srcTy, cvt, plan, messageIdx, message, bitwidth,
              rowProjectionFailure, rowProjectionError))
        return {std::nullopt, *splitSupport};
      return {std::nullopt,
              getUnsupportedTMemCopyResult(
                  TMemCopySupportFailureLayer::InstructionSchedule,
                  rowProjectionError)};
    }
    scheduledMessage.sourceRowProjection = std::move(*rowProjection);
    if (message.useDirectSeedDescriptor) {
      if (auto seedDescriptor =
              getDirectTMemCopySeedDescriptorImm(srcTy, plan.family)) {
        scheduledMessage.directSeedDescriptorImm = *seedDescriptor;
        executablePlan.messages.push_back(std::move(scheduledMessage));
        continue;
      }
      return {std::nullopt,
              getUnsupportedTMemCopyResult(
                  TMemCopySupportFailureLayer::InstructionSchedule,
                  Twine("tcgen05.copy.") +
                      stringifyTMemCopyFamily(plan.family) +
                      " direct-seed descriptor plan requires a shared-memory "
                      "source layout whose base and offset bits can be encoded "
                      "in the immediate seed descriptor; this source layout "
                      "must use a descriptor-loader plan instead.")};
    }
    std::string instructionProjectionError;
    TMemCopyInstructionColumnProjectionFailure instructionProjectionFailure;
    auto instructionProjection = getTMemCopyInstructionColumnProjectionPlan(
        cvt, message, bitwidth, &instructionProjectionError,
        &instructionProjectionFailure);
    if (!instructionProjection) {
      if (debugTMemQuery &&
          instructionProjectionFailure.kind !=
              TMemCopyInstructionColumnProjectionFailureKind::None) {
        llvm::errs() << "[tmem-copy] instruction-column failure kind="
                     << stringifyTMemCopyInstructionColumnProjectionFailureKind(
                            instructionProjectionFailure.kind)
                     << " bit="
                     << instructionProjectionFailure.logicalColBit
                     << " actual="
                     << instructionProjectionFailure.actualOffset
                     << " expected="
                     << instructionProjectionFailure.expectedOffset;
        if (instructionProjectionFailure.descriptorRowDelta)
          llvm::errs() << " descriptorRowDelta="
                       << *instructionProjectionFailure.descriptorRowDelta;
        if (instructionProjectionFailure
                .descriptorRowDeltaSpansInstructionRows)
          llvm::errs() << " spansInstructionRows=1";
        if (auto splitRequirement = getTMemCopyDescriptorRowSplitRequirement(
                instructionProjectionFailure)) {
          llvm::errs() << " selectedColumnRun="
                       << splitRequirement->selectedColumnRun
                       << " columnSelectionPeriod="
                       << splitRequirement->columnSelectionPeriod;
        }
        if (auto packedLaneRequirement = getTMemCopyPackedLaneRequirement(
                instructionProjectionFailure)) {
          llvm::errs() << " packedLaneBits="
                       << packedLaneRequirement->laneBits
                       << " lanesPerDword="
                       << packedLaneRequirement->lanesPerDword;
        }
        if (instructionProjectionFailure.packedLaneProjection)
          llvm::errs() << " physicalColumns="
                       << instructionProjectionFailure.packedLaneProjection
                              ->physicalInstructionColumns;
        llvm::errs() << "\n";
      }
      if (auto splitSupport = getTMemCopyDescriptorRowSplitScheduleSupport(
              plan.family, messageIdx, instructionProjectionFailure,
              instructionProjectionError))
        return {std::nullopt, *splitSupport};
      if (auto permutationSupport =
              getTMemCopyInstructionColumnPermutationScheduleSupport(
                  plan.family, messageIdx, instructionProjectionFailure,
                  instructionProjectionError))
        return {std::nullopt, *permutationSupport};
      if (auto packedLaneSupport = getTMemCopyPackedLaneScheduleSupport(
              plan.family, messageIdx, instructionProjectionFailure,
              instructionProjectionError))
        return {std::nullopt, *packedLaneSupport};
      std::string reason;
      llvm::raw_string_ostream os(reason);
      os << "tcgen05.copy." << stringifyTMemCopyFamily(plan.family)
         << " descriptor message " << messageIdx
         << " has an unsupported instruction-column projection. "
         << instructionProjectionError;
      return {std::nullopt,
              getUnsupportedTMemCopyResult(
                  TMemCopySupportFailureLayer::InstructionSchedule, os.str())};
    }
    scheduledMessage.instructionColumnProjection =
        std::move(*instructionProjection);

    auto srcDescLayouts =
        getTMemCopyDescriptorLayouts(srcTy, shmemLl, cvt, message);
    if (debugTMemQuery) {
      llvm::errs() << "[tmem-copy] descriptor family="
                   << stringifyTMemCopyFamily(plan.family)
                   << " message=" << messageIdx << " descriptorShape=["
                   << message.descriptorShape[0] << ", "
                   << message.descriptorShape[1] << "] instrShape=["
                   << message.instrShape[0] << ", " << message.instrShape[1]
                   << "] candidates=" << srcDescLayouts.size() << "\n";
      if (message.descriptorCvt)
        llvm::errs() << "[tmem-copy] message descriptor projection:\n"
                     << message.descriptorCvt->toString() << "\n";
      for (auto [layoutIdx, layout] : llvm::enumerate(srcDescLayouts))
        llvm::errs() << "[tmem-copy] descriptor candidate " << layoutIdx
                     << ":\n"
                     << layout.toString() << "\n";
    }
    if (auto descriptorLayout =
            selectTMemCopyDescriptorLayout(srcDescLayouts,
                                           message.descriptorShape,
                                           plan.family, bitwidth)) {
      if (debugTMemQuery) {
        llvm::errs() << "[tmem-copy] selected descriptor mnDim="
                     << descriptorLayout->mnDim << ":\n"
                     << descriptorLayout->layout.toString() << "\n";
      }
      scheduledMessage.descriptorLayout = std::move(*descriptorLayout);
      executablePlan.messages.push_back(std::move(scheduledMessage));
      continue;
    }
    std::string reason;
    llvm::raw_string_ostream os(reason);
    os << "tcgen05.copy." << stringifyTMemCopyFamily(plan.family)
       << " descriptor message " << messageIdx
       << " has no representable MMAv5 shared-memory descriptor; tried "
       << srcDescLayouts.size() << " candidate layout(s) for descriptor shape ["
       << message.descriptorShape[0] << ", " << message.descriptorShape[1]
       << "] and instruction shape [" << message.instrShape[0] << ", "
       << message.instrShape[1] << "].";
    if (auto projectionNote =
            getTMemCopyInstructionColumnProjectionNote(cvt, message, bitwidth))
      os << " " << *projectionNote;
    if (auto storageNote =
            getTMemCopySubwordSourceStorageNote(cvt, message, bitwidth))
      os << " " << *storageNote;
    return {std::nullopt,
            getUnsupportedTMemCopyResult(
                TMemCopySupportFailureLayer::DescriptorSynthesis, os.str())};
  }
  return {std::move(executablePlan), getSupportedTMemCopyResult()};
}

bool canSynthesizeTMemCopySharedDescriptorPlan(gpu::MemDescType srcTy,
                                               const LinearLayout &shmemLl,
                                               const LinearLayout &cvt,
                                               const TMemCopyPlan &plan,
                                               int bitwidth) {
  return getTMemCopySharedDescriptorPlanSupport(srcTy, shmemLl, cvt, plan,
                                                bitwidth)
      .supported;
}

} // namespace mlir::triton::nvidia_gpu

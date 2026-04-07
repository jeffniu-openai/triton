#include "Dialect/NVGPU/IR/Dialect.h"
#include "DotOpToLLVM/MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::NVIDIA;

// The maximum number of tensor memory registers that can be accessed
// by a single message regardless of shape or repetitions
static constexpr int largestTmemLoadStore = 128;
// The maximum number of thread registers that can be populated by
// multiple messages
static constexpr int maxRegisters = 256;

namespace {

Value advanceTensorMemoryBase(Location loc, ConversionPatternRewriter &rewriter,
                              Value base, uint32_t offset) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value newBase = b.add(b.ptrtoint(i32_ty, base), b.i32_val(offset));
  return b.inttoptr(ptr_ty(rewriter.getContext(), 3), newBase);
}

static bool preserveTMemLdStSupportQueryBaseOffset(
    MemDescType memTy, const TMemLdStQueryLayout &supportQuery) {
  (void)memTy;
  return llvm::any_of(supportQuery.origin,
                      [](int32_t value) { return value != 0; });
}

static uint32_t getAlreadyAdjustedTMemSubviewBaseOffset(Value memDescValue) {
  if (!memDescValue)
    return 0;

  auto recurse = [&](Value src) {
    return getAlreadyAdjustedTMemSubviewBaseOffset(src);
  };

  if (auto reinterpret = dyn_cast_if_present<triton::gpu::MemDescReinterpretOp>(
          memDescValue.getDefiningOp())) {
    return recurse(reinterpret.getSrc());
  }

  if (auto reshape = dyn_cast_if_present<triton::gpu::MemDescReshapeOp>(
          memDescValue.getDefiningOp())) {
    return recurse(reshape.getSrc());
  }

  if (auto trans = dyn_cast_if_present<triton::gpu::MemDescTransOp>(
          memDescValue.getDefiningOp())) {
    return recurse(trans.getSrc());
  }

  if (auto subslice = dyn_cast_if_present<triton::gpu::MemDescSubsliceOp>(
          memDescValue.getDefiningOp())) {
    auto srcTy = dyn_cast<MemDescType>(subslice.getSrc().getType());
    auto dstTy = dyn_cast<MemDescType>(subslice.getType());
    if (!srcTy || !dstTy || srcTy.getRank() != 2 || dstTy.getRank() != 2 ||
        !isTensorMemoryEncoding(srcTy.getEncoding()) ||
        isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding()))
      return 0;
    SmallVector<int32_t> offsets(subslice.getOffsets().begin(),
                                 subslice.getOffsets().end());
    return recurse(subslice.getSrc()) +
           triton::nvidia_gpu::getTMemViewOffset(srcTy, offsets);
  }

  if (auto subslice = dyn_cast_if_present<TMEMSubSliceOp>(
          memDescValue.getDefiningOp())) {
    auto srcTy = dyn_cast<MemDescType>(subslice.getSrc().getType());
    if (!srcTy)
      return 0;
    return recurse(subslice.getSrc()) + getTMemSubSliceOffset(srcTy, subslice.getN());
  }

  if (auto index = dyn_cast_if_present<triton::gpu::MemDescIndexOp>(
          memDescValue.getDefiningOp())) {
    auto srcTy = dyn_cast<MemDescType>(index.getSrc().getType());
    if (!srcTy)
      return 0;
    APInt indexValue;
    if (!matchPattern(index.getIndex(), m_ConstantInt(&indexValue)))
      return 0;
    SmallVector<int32_t> offsets(srcTy.getRank(), 0);
    offsets.front() = indexValue.getSExtValue();
    return recurse(index.getSrc()) +
           triton::nvidia_gpu::getTMemViewOffset(srcTy, offsets);
  }

  return 0;
}

static LinearLayout getTMemCopyAddressLayout(MemDescType memDescType,
                                             TMemCopyFamily family) {
  LinearLayout ll = [&]() {
    std::string error;
    if (isTensorMemoryEncoding(memDescType.getEncoding()) &&
        !isa<TensorMemoryScalesEncodingAttr>(memDescType.getEncoding())) {
      if (auto maybeAnalysis = getTMemViewAnalysisLinearLayout(
              memDescType.getShape(), memDescType.getEncoding(), &error)) {
        return normalizeTensorMemoryLinearLayoutForAnalysis(*maybeAnalysis);
      }
    }
    return normalizeTensorMemoryLinearLayoutForAnalysis(
        triton::gpu::toLinearLayout(memDescType));
  }();

  if (family != TMemCopyFamily::Dense128x128b &&
      family != TMemCopyFamily::Dense128x256b)
    return ll;

  auto *ctx = memDescType.getContext();
  auto kCol = StringAttr::get(ctx, "col");
  if (!ll.hasInDim(kCol) || ll.getNumOutDims() != 2)
    return ll;

  auto bases = ll.getBases();
  auto &colBases = bases[kCol];
  llvm::stable_sort(colBases, [&](ArrayRef<int32_t> lhs, ArrayRef<int32_t> rhs) {
    bool lhsTouchesRow = lhs[0] != 0;
    bool lhsTouchesCol = lhs[1] != 0;
    bool rhsTouchesRow = rhs[0] != 0;
    bool rhsTouchesCol = rhs[1] != 0;
    auto classify = [](bool touchesRow, bool touchesCol) {
      if (touchesCol && !touchesRow)
        return 0;
      if (touchesRow && !touchesCol)
        return 1;
      return 2;
    };
    return classify(lhsTouchesRow, lhsTouchesCol) <
           classify(rhsTouchesRow, rhsTouchesCol);
  });
  return LinearLayout(std::move(bases), ll.getOutDims(),
                      /*requireSurjective=*/ll.isSurjective());
}

static uint32_t getTMemCopyViewOffset(MemDescType memDescType,
                                      ArrayRef<int32_t> offsets,
                                      TMemCopyFamily family) {
  assert(offsets.size() == memDescType.getRank());
  auto *ctx = memDescType.getContext();
  auto kRow = StringAttr::get(ctx, "row");
  auto kCol = StringAttr::get(ctx, "col");
  auto ll = getTMemCopyAddressLayout(memDescType, family);
  unsigned memRank = memDescType.getRank();
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
    auto linearizePrefixOffsets = [](ArrayRef<int64_t> shape,
                                     ArrayRef<int32_t> prefixOffsets) {
      assert(shape.size() == prefixOffsets.size());
      int64_t linearized = 0;
      int64_t stride = 1;
      for (auto [size, offset] :
           llvm::reverse(llvm::zip_equal(shape, prefixOffsets))) {
        linearized += static_cast<int64_t>(offset) * stride;
        stride *= size;
      }
      return linearized;
    };
    auto singleBufferCols = ll.getInDimSize(kCol) / (32 / bitwidth);
    offsetCol += linearizePrefixOffsets(
                     memDescType.getShape().take_front(extraRank),
                     offsets.take_front(extraRank)) *
                 singleBufferCols;
  }
  return offsetCol | offsetRow << 16;
}

SmallVector<Value> pack(ArrayRef<Value> values, Type outType, Location loc,
                        ConversionPatternRewriter &rewriter, bool pad = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type inType = values[0].getType();
  if (inType == outType) {
    return to_vector(values);
  }

  auto inbitwidth = inType.getIntOrFloatBitWidth();
  auto outbitwidth = outType.getIntOrFloatBitWidth();
  assert(inbitwidth <= outbitwidth);
  SmallVector<Value> packedValues;
  if (inbitwidth == outbitwidth) {
    for (auto &val : values) {
      packedValues.push_back(b.bitcast(val, outType));
    }
    return packedValues;
  }

  auto vecSize = outbitwidth / inbitwidth;
  auto vecTy = vec_ty(inType, vecSize);

  auto elemsPerVec = pad ? 1 : vecSize;
  assert(values.size() % elemsPerVec == 0);
  for (int i = 0; i < values.size(); i += elemsPerVec) {
    Value packed = b.undef(vecTy);
    for (int j = 0; j < elemsPerVec; j++) {
      Value val = values[i + j];
      packed = b.insert_element(vecTy, packed, val, b.i32_val(j));
    }
    packed = b.bitcast(packed, outType);
    packedValues.emplace_back(std::move(packed));
  }
  return packedValues;
}

SmallVector<Value> unpack(ArrayRef<Value> packedValues, Type outType,
                          Location loc, ConversionPatternRewriter &rewriter,
                          bool pad = false) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type inType = packedValues[0].getType();
  if (inType == outType) {
    return to_vector(packedValues);
  }

  auto inbitwidth = inType.getIntOrFloatBitWidth();
  auto outbitwidth = outType.getIntOrFloatBitWidth();
  assert(inbitwidth >= outbitwidth);
  SmallVector<Value> unpackedValues;
  if (inbitwidth == outbitwidth) {
    for (auto val : packedValues) {
      unpackedValues.push_back(b.bitcast(val, outType));
    }
    return unpackedValues;
  }
  auto vecSize = inbitwidth / outbitwidth;
  auto vecTy = vec_ty(outType, vecSize);

  auto elemsPerVec = pad ? 1 : vecSize;
  for (auto val : packedValues) {
    Value packed = b.bitcast(val, vecTy);
    for (int j = 0; j < elemsPerVec; j++) {
      unpackedValues.push_back(
          b.extract_element(outType, packed, b.i32_val(j)));
    }
  }
  return unpackedValues;
}

void createTensorMemoryStore(Location loc, Value address, int colOffset,
                             SmallVector<Value> &srcs,
                             std::optional<int> secondHalfOffset, Value pred,
                             bool unpacked, TMemAccessAtom atom,
                             ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  std::string packedStr = unpacked ? ".unpack::16b" : "";
  unsigned numRepeats = srcs.size() / getElementsPerThread(atom);
  std::string opcode = "@$0 tcgen05.st.sync.aligned.";
  opcode += getOpShape(atom);
  opcode += ".x" + std::to_string(numRepeats) + packedStr;
  opcode += ".b32 [$1 + " + std::to_string(colOffset) + "], ";
  if (secondHalfOffset)
    opcode += std::to_string(*secondHalfOffset) + ", {";
  else
    opcode += "{";

  SmallVector<PTXInstr::Operand *> operands;
  operands.push_back(ptxBuilder.newOperand(pred, "b"));
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  for (int i = 0; i < srcs.size(); i++) {
    opcode += "$" + std::to_string(i + 2);
    auto *resultOp = ptxBuilder.newOperand(srcs[i], "r");
    operands.push_back(resultOp);
    if (i < srcs.size() - 1)
      opcode += ", ";
  }
  opcode += "};";

  auto &st = *ptxBuilder.create(opcode);
  st(operands, /*onlyAttachMLIRArgs=*/true);
  Type voidTy = void_ty(rewriter.getContext());
  ptxBuilder.launch(rewriter, loc, voidTy);
}

// Returns {loadResult, redvalResult} where redvalResult is null if no reduction
std::pair<Value, Value>
createTensorMemoryLoad(Location loc, MLIRContext *ctx, Value address,
                       int colOffset, std::optional<int> secondHalfOffset,
                       bool unpacked, int numRegPerMessage, TMemAccessAtom atom,
                       std::optional<TMEMLoadReduceModifier> redOp, bool useAbs,
                       bool useNaN, Type elemTy,
                       ConversionPatternRewriter &rewriter) {
  PTXBuilder ptxBuilder;
  // If the memory is unpacked we need to pack on the fly when loading.
  std::string packedStr = unpacked ? ".pack::16b" : "";
  unsigned numRepeats = numRegPerMessage / getElementsPerThread(atom);

  std::string opcode = std::string("tcgen05.ld.") + (redOp ? "red." : "");
  opcode += "sync.aligned.";
  opcode += getOpShape(atom);
  opcode += ".x" + std::to_string(numRepeats);

  if (redOp) {
    if (unpacked) {
      llvm_unreachable("Unpacked is unsupported with TMEM reduction");
    }
    // Add reduction modifier: .min or .max
    switch (*redOp) {
    case TMEMLoadReduceModifier::MIN:
      opcode += ".min";
      break;
    case TMEMLoadReduceModifier::MAX:
      opcode += ".max";
      break;
    default:
      llvm_unreachable("Unsupported reduction modifier");
    }
    if (useAbs)
      opcode += ".abs";
    if (useNaN)
      opcode += ".NaN";

    std::string redStr;
    if (elemTy.isF32()) {
      redStr = ".f32";
    } else {
      llvm_unreachable("Unsupported type for TMEM reduction");
    }
    opcode += redStr;
  } else {
    opcode += packedStr + ".b32";
  }

  opcode += " {";

  SmallVector<PTXInstr::Operand *> operands;
  for (int i = 0; i < numRegPerMessage; i++) {
    opcode += "$" + std::to_string(i);
    auto *resultOp = ptxBuilder.newOperand("=r");
    operands.push_back(resultOp);
    if (i < numRegPerMessage - 1)
      opcode += ", ";
  }
  opcode += "}";

  int nextOperandIdx = numRegPerMessage;

  // Add redval output operand if reduction is enabled
  if (redOp) {
    opcode += ", {$" + std::to_string(nextOperandIdx) + "}";
    auto *redvalOp = ptxBuilder.newOperand("=r");
    operands.push_back(redvalOp);
    nextOperandIdx++;
  }

  opcode += ", [$" + std::to_string(nextOperandIdx) + " + " +
            std::to_string(colOffset) + "]";
  if (secondHalfOffset)
    opcode += ", " + std::to_string(*secondHalfOffset);
  opcode += ";";
  operands.push_back(ptxBuilder.newOperand(address, "r"));
  auto &ld = *ptxBuilder.create(opcode);
  ld(operands, /*onlyAttachMLIRArgs=*/true);

  // Build return type: data registers + optional redval register
  int totalResults = numRegPerMessage + (redOp ? 1 : 0);
  Type retTy;
  if (totalResults == 1) {
    retTy = i32_ty;
  } else {
    SmallVector<Type> elemTypes(totalResults, i32_ty);
    retTy = struct_ty(elemTypes);
  }
  Value ret = ptxBuilder.launch(rewriter, loc, retTy);

  // Extract load result and redval if needed
  Value loadResult = ret;
  Value redvalResult = nullptr;

  if (redOp) {
    // Per PTX spec: .num must be at least .x2 when .red is specified,
    // so numRegPerMessage >= 2 * getElementsPerThread(atom) >= 2.
    // ret is a struct with numRegPerMessage + 1 elements: {loadVals..., redval}
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    SmallVector<Type> loadElemTypes(numRegPerMessage, i32_ty);
    Type loadStructTy = struct_ty(loadElemTypes);
    Value loadStruct = b.undef(loadStructTy);
    for (int i = 0; i < numRegPerMessage; i++) {
      Value elem = b.extract_val(i32_ty, ret, i);
      loadStruct = b.insert_val(loadStructTy, loadStruct, elem, i);
    }
    loadResult = loadStruct;
    redvalResult = b.extract_val(i32_ty, ret, numRegPerMessage);
    // Bitcast redval from i32 to the target element type
    if (redvalResult && elemTy != i32_ty) {
      redvalResult = b.bitcast(redvalResult, elemTy);
    }
  }

  return {loadResult, redvalResult};
}

static SmallVector<Value> unpackResults(Value packedValues, Type elemTy,
                                        int numCols, Location loc,
                                        ConversionPatternRewriter &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> resultVals;
  int numElementsPer32B = 32 / elemTy.getIntOrFloatBitWidth();
  Type packedType = elemTy;
  if (numElementsPer32B > 1)
    packedType = vec_ty(elemTy, numElementsPer32B);

  auto unpackElement = [&](Value result) {
    result = b.bitcast(result, packedType);
    if (numElementsPer32B > 1) {
      for (int j = 0; j < numElementsPer32B; j++) {
        Value elem = b.extract_element(elemTy, result, b.i32_val(j));
        resultVals.push_back(elem);
      }
    } else {
      resultVals.push_back(result);
    }
  };

  if (isa<LLVM::LLVMStructType>(packedValues.getType())) {
    for (int i = 0; i < numCols; i++) {
      Value result = b.extract_val(i32_ty, packedValues, i);
      unpackElement(result);
    }
  } else {
    unpackElement(packedValues);
  }
  return resultVals;
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction.
// Reduction produces exactly one value per thread; if multiple messages
// contribute partial reductions, they are combined into one.
std::pair<SmallVector<Value>, SmallVector<Value>> lowerTMemLdSt(
    Location loc, ConversionPatternRewriter &rewriter, const LinearLayout &reps,
    ArrayRef<Value> vals, TMemAccessAtom atom, Type llvmElemTy, Value tmemBase,
    Value pred, int valsPerMessage, bool unpacked,
    std::optional<uint32_t> secondHalfOffset, uint32_t baseOffset,
    uint32_t warpBaseOffset0, uint32_t warpBaseOffset1,
    ArrayRef<int32_t> packetOffsets,
    std::optional<TMEMLoadReduceModifier> redOp, bool useAbs, bool useNaN) {
  auto *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");

  auto kCol = str_attr("col");
  auto kRow = str_attr("row");
  bool isStore = !vals.empty();

  tmemBase = b.ptrtoint(i32_ty, tmemBase);
  if (baseOffset != 0)
    tmemBase = b.add(tmemBase, b.i32_val(baseOffset));

  assert(to_vector(reps.getOutDimNames()) ==
         SmallVector<StringAttr>({kRow, kCol}));
  auto getRowCol = [kRow, kCol](const auto &rowCol) {
    assert(rowCol.size() == 2);
    assert(std::get<0>(rowCol[0]) == kRow);
    assert(std::get<0>(rowCol[1]) == kCol);
    return std::make_pair(std::get<1>(rowCol[0]), std::get<1>(rowCol[1]));
  };
  auto allRegBasesZero = [&]() {
    if (!reps.hasInDim(kReg))
      return false;
    for (unsigned idx = 0; idx < reps.getInDimSizeLog2(kReg); ++idx) {
      if (!llvm::all_of(reps.getBasis(kReg, idx),
                        [](int32_t value) { return value == 0; })) {
        return false;
      }
    }
    return true;
  };
  auto regColBasesAreContiguous = [&]() {
    if (!reps.hasInDim(kReg) || !reps.hasOutDim(kCol))
      return false;
    for (unsigned idx = 0; idx < reps.getInDimSizeLog2(kReg); ++idx) {
      if (reps.getBasis(kReg, idx, kCol) != (1 << idx))
        return false;
      for (StringAttr outDim : reps.getOutDimNames()) {
        if (outDim == kCol)
          continue;
        if (reps.getBasis(kReg, idx, outDim) != 0)
          return false;
      }
    }
    return true;
  };
  bool recoverScalar32x32ColSteps =
      atom == TMemAccessAtom::I32x32b && valsPerMessage == 1 &&
      reps.hasOutDim(kCol) && reps.getOutDimSize(kCol) > 1 &&
      reps.hasInDim(kReg) &&
      reps.getInDimSize(kReg) == reps.getOutDimSize(kCol) &&
      allRegBasesZero();
  bool recoverScalarI16x32bx2ColSteps =
      atom == TMemAccessAtom::I16x32bx2 && unpacked && valsPerMessage == 1 &&
      reps.hasOutDim(kCol) && reps.getOutDimSize(kCol) > 1 &&
      reps.hasInDim(kReg) &&
      reps.getOutDimSize(kCol) == 2 * reps.getInDimSize(kReg);
  bool recoverPackedI16x32bx2X2ColSteps =
      atom == TMemAccessAtom::I16x32bx2 && unpacked && valsPerMessage == 2 &&
      reps.hasOutDim(kCol) && reps.getOutDimSize(kCol) > 1 &&
      reps.hasInDim(kReg) &&
      reps.getOutDimSize(kCol) == 2 * reps.getInDimSize(kReg) &&
      regColBasesAreContiguous();

  Value warpId = WarpIdOp::create(rewriter, loc);
  // The first warp-group anchors are part of the lowering plan and may map to
  // lifted TMEM row/col offsets instead of the legacy row-only 32/64 pair.
  auto warpIdInGroup = b.and_(warpId, b.i32_val(3));
  Value warpBaseOffset = b.i32_val(0);
  if (warpBaseOffset0 != 0) {
    Value warpBit0 = b.and_(warpIdInGroup, b.i32_val(1));
    warpBaseOffset =
        b.add(warpBaseOffset, b.mul(warpBit0, b.i32_val(warpBaseOffset0)));
  }
  if (warpBaseOffset1 != 0) {
    Value warpBit1 = b.lshr(b.and_(warpIdInGroup, b.i32_val(2)), b.i32_val(1));
    warpBaseOffset =
        b.add(warpBaseOffset, b.mul(warpBit1, b.i32_val(warpBaseOffset1)));
  }
  tmemBase = b.add(tmemBase, warpBaseOffset);
  // The block offset is already added to the tmemBase
  // Add warp groups to tmemBase
  if (reps.getInDimSize(kWarp) > 4) {
    // The explicit warp-base anchors above already cover the low two warp bits
    // (the first four warps in each warp-group). Only add the higher warp
    // group contribution here; otherwise packed TMEM paths can reapply the
    // in-group row anchor and misaddress transformed support/query reps.
    Value warpGroupId =
        b.shl(b.lshr(warpId, b.i32_val(2)), b.i32_val(2));
    auto rowCol = applyLinearLayout(
        loc, rewriter, reps,
        {{kReg, b.i32_val(0)}, {kLane, b.i32_val(0)}, {kWarp, warpGroupId}});
    auto [row, col] = getRowCol(rowCol);
    tmemBase = b.add(tmemBase,
                     b.or_(b.shl(row, b.i32_val(16)), col, /*disjoint*/ true));
  }

  SmallVector<Value> resultVals, redvalVals;
  if (!packetOffsets.empty()) {
    assert(static_cast<int>(packetOffsets.size()) ==
               reps.getInDimSize(kReg) / valsPerMessage &&
           "packetOffsets must match the lowered TMEM message count");
  }
  for (int i = 0; i < reps.getInDimSize(kReg); i += valsPerMessage) {
    int staticOffset = 0;
    if (!packetOffsets.empty()) {
      staticOffset = packetOffsets[i / valsPerMessage];
    } else {
      auto [row, col] =
          getRowCol(reps.apply({{kReg, i}, {kLane, 0}, {kWarp, 0}}));
      if (recoverScalar32x32ColSteps)
        // Scalarized 32x32b.x1 packets cover one 32-bit TMEM column per
        // message. PTX col immediates for `.b32` packets are byte-addressed,
        // so recovered support/query reps whose register bases collapsed to
        // zero still need a 4-byte stride per scalar packet.
        col = i * 4;
      if (recoverScalarI16x32bx2ColSteps)
        // Scalarized unpacked 16x32bx2.x1 packets advance in packed-dword
        // column steps, but transformed descriptor-query reps can still carry
        // higher-order TMEM band bits in the raw column coordinate. Preserve
        // those bits and only add the intra-band packet stride.
        col += i * 4;
      if (recoverPackedI16x32bx2X2ColSteps)
        // Recovered unpacked 16x32bx2.x2 paths advance by packed-dword
        // message index. Preserve any raw TMEM band bits from the query reps
        // and only repair the intra-band stride.
        col += (i / valsPerMessage) * 2;
      // Encode row into the base address and pass col as an immediate
      // colOffset.
      staticOffset = col | (row << 16);
    }
    if (isStore) {
      auto chunk = to_vector(vals.slice(i, valsPerMessage));
      createTensorMemoryStore(loc, tmemBase, /*colOffset=*/staticOffset, chunk,
                              /*secondHalfOffset=*/secondHalfOffset, pred,
                              /*unpacked=*/unpacked, atom, rewriter);
    } else {
      auto [outVals, redval] =
          createTensorMemoryLoad(loc, ctx, tmemBase, /*colOffset=*/staticOffset,
                                 /*secondHalfOffset=*/secondHalfOffset,
                                 /*unpacked=*/unpacked,
                                 /*numRegPerMessage=*/valsPerMessage, atom,
                                 redOp, useAbs, useNaN, llvmElemTy, rewriter);
      resultVals.append(
          unpackResults(outVals, llvmElemTy, valsPerMessage, loc, rewriter));
      if (redval)
        redvalVals.push_back(redval);
    }
  }

  return {resultVals, redvalVals};
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction
static FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
lowerTMemLdStFromInfo(Location loc, ConversionPatternRewriter &rewriter,
                      TMemLdStEncodingInfo &info, Value pred, Type llvmElemTy,
                      ArrayRef<Value> vals, Value tmemBase,
                      std::optional<TMEMLoadReduceModifier> redOp, bool useAbs,
                      bool useNaN) {
  bool isStore = !vals.empty();
  if (info.broadcast) {
    auto removeBroadcast = std::move(info.broadcast.value());
    info.broadcast = std::nullopt;

    auto inVals = to_vector(vals);
    if (isStore) {
      inVals = removeBroadcast.apply(inVals);
    }
    auto outOr = lowerTMemLdStFromInfo(loc, rewriter, info, pred, llvmElemTy,
                                       inVals, tmemBase, redOp, useAbs,
                                       useNaN);
    if (failed(outOr))
      return failure();
    auto [outVals, redvalVals] = *outOr;
    if (!isStore) {
      auto kReg = *info.reps.getInDimNames().begin();
      uint32_t broadcastMask = info.reps.getFreeVariableMasks().lookup(kReg);
      size_t expectedSize =
          info.reps.getInDimSize(kReg) / (1 << llvm::popcount(broadcastMask));
      if (expectedSize != outVals.size()) {
        emitError(loc)
            << "unsupported broadcasted TMEM lowering for this view; "
               "reshape or permute so TMEM columns stay contiguous, or use a "
               "different TMEM register layout";
        return failure();
      }
      outVals = broadcastAs(outVals, info.reps);
    }
    return std::make_pair(std::move(outVals), std::move(redvalVals));
  }
  if (llvmElemTy.getIntOrFloatBitWidth() < 32) {
    unsigned bitwidth = llvmElemTy.getIntOrFloatBitWidth();
    bool padding = false;
    Type packedElemTy;
    if (info.vec > 1) {
      // There are contiguous elements along kCol, so we can pack them into a
      // larger dtype
      packedElemTy = int_ty(bitwidth * info.vec);
      info.vec = 1;
    } else {
      padding = info.padding;
      assert(info.unpacked || info.padding);
      packedElemTy = i32_ty;
    }
    SmallVector<Value> inVals = to_vector(vals);
    if (isStore) {
      inVals = pack(inVals, packedElemTy, loc, rewriter, padding);
    }
    auto outOr = lowerTMemLdStFromInfo(loc, rewriter, info, pred, packedElemTy,
                                       inVals, tmemBase, redOp, useAbs,
                                       useNaN);
    if (failed(outOr))
      return failure();
    auto [outVals, redvalVals] = *outOr;
    if (!isStore) {
      outVals = unpack(outVals, llvmElemTy, loc, rewriter, padding);
    }
    return std::make_pair(std::move(outVals), std::move(redvalVals));
  }

  SmallVector<Value> inVals = to_vector(vals);
  if (isStore) {
    inVals = info.perm.apply(inVals);
  }
  auto [outVals, redvalVals] =
      lowerTMemLdSt(loc, rewriter, info.reps, inVals, info.atom, llvmElemTy,
                    tmemBase, pred, info.numRegsPerMessage, info.unpacked,
                    info.secondHalfOffset, info.baseOffset, info.warpBaseOffset0,
                    info.warpBaseOffset1, info.packetOffsets, redOp, useAbs,
                    useNaN);
  if (!isStore) {
    outVals = info.perm.inverse().apply(outVals);
  }
  return std::make_pair(std::move(outVals), std::move(redvalVals));
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction.
static FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
lowerTMemLdStFromTypes(
    Location loc, ConversionPatternRewriter &rewriter, RankedTensorType regTy,
    MemDescType memTy, Value memDescValue, Value tmemBase, int maxnreg,
    Value pred, Type llvmElemTy, ArrayRef<Value> vals,
    std::optional<TMEMLoadReduceModifier> redOp = std::nullopt,
    bool useAbs = false, bool useNaN = false) {
  auto diag = [loc]() { return emitError(loc); };
  bool debugQuerySelection = std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr;
  bool traceQuerySelection =
      std::getenv("TRITON_TRACE_TMEM_QUERY_LOWERING_FILE") != nullptr;
  if (debugQuerySelection && memDescValue && memDescValue.getDefiningOp())
    llvm::errs() << "[tmem-ldst] defOp="
                 << memDescValue.getDefiningOp()->getName().getStringRef()
                 << " memTy=" << memTy << "\n";
  auto appendTrace = [&](const Twine &msg) {
    if (!traceQuerySelection)
      return;
    std::error_code ec;
    llvm::raw_fd_ostream os("/tmp/tmem_query_lowering_trace.log", ec,
                            llvm::sys::fs::OF_Append);
    if (ec)
      return;
    os << msg << "\n";
  };
  if (memDescValue) {
    std::string unsupportedDescriptorViewError;
    if (isUnsupportedDirectTMemLdStDescriptorView(
            memDescValue, &unsupportedDescriptorViewError)) {
      if (!unsupportedDescriptorViewError.empty())
        emitError(loc) << unsupportedDescriptorViewError;
      return failure();
    }
  }
  bool isViewLikeMemDesc =
      memDescValue &&
      isa_and_nonnull<triton::gpu::MemDescIndexOp,
                      TMEMSubSliceOp, triton::gpu::MemDescSubsliceOp,
                      triton::gpu::MemDescReshapeOp,
                      triton::gpu::MemDescTransOp,
                      triton::gpu::MemDescReinterpretOp>(
          memDescValue.getDefiningOp());
  bool disallowSupportRescueFor32x32Subview =
      isViewLikeMemDesc && memTy.getRank() == 2 && memTy.getShape()[0] == 32 &&
      memTy.getShape()[1] == 32;
  auto queryTypes =
      memDescValue ? triton::nvidia_gpu::getTMemLdStQueryTypes(memDescValue)
                   : SmallVector<MemDescType>{memTy};
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
  auto kRow = StringAttr::get(rewriter.getContext(), "row");
  auto kCol = StringAttr::get(rewriter.getContext(), "col");
  bool disallowQueryTypeRescueForRowZeroLiftedReinterpret = [&]() {
    if (!memDescValue ||
        !isa_and_nonnull<triton::gpu::MemDescReinterpretOp>(
            memDescValue.getDefiningOp()) ||
        memTy.getRank() != 2)
      return false;
    auto memLayout = toLinearLayout(memTy);
    int bitwidth = memTy.getElementTypeBitWidth();
    int64_t logicalRows = memTy.getShape()[0];
    int64_t logicalCols = memTy.getShape()[1];
    auto activeMemLayout = memLayout;
    if (activeMemLayout.hasInDim(kRow))
      activeMemLayout = activeMemLayout.removeZeroBasesAlongDim(kRow);
    int64_t activePhysicalRows =
        activeMemLayout.hasInDim(kRow) ? activeMemLayout.getInDimSize(kRow)
                                       : logicalRows;
    int64_t physicalCols =
        memLayout.hasInDim(kCol) ? memLayout.getInDimSize(kCol) / (32 / bitwidth)
                                 : logicalCols;
    return bitwidth == 16 && hasZeroBasisAlong(memLayout, kRow) &&
           !hasZeroBasisAlong(memLayout, kCol) &&
           logicalRows == activePhysicalRows &&
           logicalCols == physicalCols * 2;
  }();
  bool preferWidenedRootRowPlanForRowZeroM64Alloc = [&]() {
    if (!memDescValue ||
        !isa_and_nonnull<triton::nvidia_gpu::TMEMAllocOp>(
            memDescValue.getDefiningOp()) ||
        memTy.getRank() != 2 || memTy.getElementTypeBitWidth() != 32 ||
        memTy.getShape()[0] != 64)
      return false;
    auto memLayout = toLinearLayout(memTy);
    return hasZeroBasisAlong(memLayout, kRow) &&
           !hasZeroBasisAlong(memLayout, kCol);
  }();
  std::optional<TMemLdStQueryLayout> rawQueryLayout;
  std::optional<TMemLdStRowPlan> rawRowPlan;
  if (memDescValue) {
    bool disableSupportQuery =
        std::getenv("TRITON_DISABLE_TMEM_SUPPORT_QUERY_LOWERING") != nullptr;
    auto trySupportQuery = [&](const TMemLdStQueryLayout &supportQuery,
                               std::optional<TMemLdStRowPlan> supportRowPlan)
        -> FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>> {
      if (!supportRowPlan)
        supportRowPlan = getTMemLdStRowPlanForQuery(memDescValue, memTy);
      if (!supportRowPlan)
        supportRowPlan = getBackingTMemLdStRowPlan(memDescValue);
      if (!supportRowPlan)
        supportRowPlan = getTMemLdStRowPlan(supportQuery.layout);
      std::string supportDetails;
      auto encodingInfoOr = [&]() -> FailureOr<TMemLdStEncodingInfo> {
        llvm::raw_string_ostream os(supportDetails);
        ScopedDiagnosticHandler handler(
            rewriter.getContext(), [&](Diagnostic &diag) { diag.print(os); });
        return computeTMemLdStEncodingInfo(
            regTy, memTy, supportQuery, maxnreg, /*emitError=*/{},
            supportRowPlan);
      }();
      appendTrace(Twine("supportQuery ") +
                  (succeeded(encodingInfoOr)
                       ? (Twine("ok atom=") +
                          Twine(static_cast<int>(encodingInfoOr->atom)) +
                          " regsPerMsg=" +
                          Twine(encodingInfoOr->numRegsPerMessage) +
                          " baseOffset=" + Twine(encodingInfoOr->baseOffset) +
                          " warpBase0=" +
                          Twine(encodingInfoOr->warpBaseOffset0) +
                          " warpBase1=" +
                          Twine(encodingInfoOr->warpBaseOffset1) +
                          " reps=" + encodingInfoOr->reps.toString())
                       : (Twine("fail details=") + supportDetails)));
      if (debugQuerySelection) {
        llvm::errs() << "[tmem-ldst] supportQuery -> "
                     << (succeeded(encodingInfoOr)
                             ? ("ok atom=" +
                                llvm::Twine(static_cast<int>(encodingInfoOr->atom)))
                                   .str()
                             : ("fail details=" + supportDetails))
                     << "\n";
      }
      if (succeeded(encodingInfoOr)) {
        auto &encodingInfo = *encodingInfoOr;
        uint32_t alreadyAdjustedBase =
            getAlreadyAdjustedTMemSubviewBaseOffset(memDescValue);
        if (!preserveTMemLdStSupportQueryBaseOffset(memTy, supportQuery))
          encodingInfo.baseOffset = 0;
        if (alreadyAdjustedBase != 0)
          encodingInfo.baseOffset =
              encodingInfo.baseOffset > alreadyAdjustedBase
                  ? encodingInfo.baseOffset - alreadyAdjustedBase
                  : 0;
        return lowerTMemLdStFromInfo(
            loc, rewriter, encodingInfo, pred, llvmElemTy, vals, tmemBase,
            redOp, useAbs, useNaN);
      }
      return failure();
    };
    std::string supportError;
    if (auto supportPlan =
            getTMemLdStSubviewSupportPlan(memDescValue, &supportError)) {
      if (auto lowered =
              trySupportQuery(supportPlan->query, supportPlan->rowPlan);
          succeeded(lowered)) {
        return *lowered;
      }
    }
    if (!disableSupportQuery && !disallowSupportRescueFor32x32Subview) {
      if (auto supportQuery =
              getTMemLdStSupportQueryLayout(memDescValue, &supportError)) {
        if (auto lowered = trySupportQuery(*supportQuery, std::nullopt);
            succeeded(lowered)) {
          return *lowered;
        }
      }
    }
    std::string rawError;
    if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
            memDescValue, /*preserveNonCanonicalView=*/true, &rawError);
        succeeded(rawQuery)) {
      rawQueryLayout = *rawQuery;
      if (isa_and_nonnull<triton::gpu::MemDescReinterpretOp>(memDescValue.getDefiningOp()) &&
          memTy.getRank() == 2 && memTy.getElementTypeBitWidth() == 32 &&
          memTy.getShape()[0] == 64 && memTy.getShape()[1] == 128) {
        rawQueryLayout->layout = toLinearLayout(memTy);
      }
      MemDescType rawMemTy = memTy;
      if (!(isa_and_nonnull<triton::gpu::MemDescReinterpretOp>(memDescValue.getDefiningOp()) &&
            memTy.getRank() == 2 && memTy.getElementTypeBitWidth() == 32 &&
            memTy.getShape()[0] == 64 && memTy.getShape()[1] == 128)) {
        if (auto maybeStandaloneTy =
                inferStandaloneTMemRegLayoutQueryType(memDescValue, /*error=*/nullptr);
            succeeded(maybeStandaloneTy))
          rawMemTy = *maybeStandaloneTy;
      }
      rawRowPlan = disallowSupportRescueFor32x32Subview
                       ? std::optional<TMemLdStRowPlan>{}
                       : getBackingTMemLdStRowPlan(memDescValue);
      if (!rawRowPlan && !disallowSupportRescueFor32x32Subview)
        rawRowPlan = getTMemLdStRowPlanForQuery(memDescValue, rawMemTy);
      if (preferWidenedRootRowPlanForRowZeroM64Alloc) {
        rawRowPlan = TMemLdStRowPlan{/*warpRow0=*/32, /*warpRow1=*/64,
                                     /*rowSpan=*/128};
      }
      if (!disallowSupportRescueFor32x32Subview &&
          isa_and_nonnull<triton::gpu::MemDescReinterpretOp>(memDescValue.getDefiningOp()) &&
          memTy.getRank() == 2 && memTy.getElementTypeBitWidth() == 32 &&
          memTy.getShape()[0] == 64 && memTy.getShape()[1] == 128) {
        rawRowPlan = TMemLdStRowPlan{/*warpRow0=*/16, /*warpRow1=*/32,
                                     /*rowSpan=*/64};
      }
      if (debugQuerySelection) {
        llvm::errs() << "[tmem-ldst] raw memTy=" << memTy << " rawRowPlan="
                     << (rawRowPlan ? llvm::Twine(rawRowPlan->rowSpan).str()
                                    : std::string("none"))
                     << "\n";
      }
      std::string rawDetails;
      auto rawEncodingInfoOr = [&]() -> FailureOr<TMemLdStEncodingInfo> {
        llvm::raw_string_ostream os(rawDetails);
        ScopedDiagnosticHandler handler(
            rewriter.getContext(), [&](Diagnostic &diag) { diag.print(os); });
        return computeTMemLdStEncodingInfo(
            regTy, rawMemTy, *rawQueryLayout, maxnreg,
            debugQuerySelection ? diag : std::function<InFlightDiagnostic()>{},
            rawRowPlan);
      }();
      appendTrace(Twine("rawQuery ") +
                  (succeeded(rawEncodingInfoOr)
                       ? (Twine("ok atom=") +
                          Twine(static_cast<int>(rawEncodingInfoOr->atom)) +
                          " regsPerMsg=" +
                          Twine(rawEncodingInfoOr->numRegsPerMessage) +
                          " baseOffset=" + Twine(rawEncodingInfoOr->baseOffset) +
                          " warpBase0=" +
                          Twine(rawEncodingInfoOr->warpBaseOffset0) +
                          " warpBase1=" +
                          Twine(rawEncodingInfoOr->warpBaseOffset1) +
                          " reps=" + rawEncodingInfoOr->reps.toString())
                       : (Twine("fail details=") + rawDetails)));
      if (debugQuerySelection) {
        llvm::errs() << "[tmem-ldst] rawQuery -> "
                     << (succeeded(rawEncodingInfoOr)
                             ? ("ok atom=" +
                                llvm::Twine(static_cast<int>(rawEncodingInfoOr->atom)))
                                   .str()
                             : ("fail details=" + rawDetails))
                     << "\n";
      }
      if (succeeded(rawEncodingInfoOr)) {
        auto &encodingInfoOr = rawEncodingInfoOr;
        uint32_t alreadyAdjustedBase =
            getAlreadyAdjustedTMemSubviewBaseOffset(memDescValue);
        // Subview ops that already advanced the TMEM base pointer should only
        // keep the portion of the raw-query baseOffset that remains relative
        // to the lowered base, rather than re-applying the full view origin.
        if (alreadyAdjustedBase != 0)
          encodingInfoOr->baseOffset =
              encodingInfoOr->baseOffset > alreadyAdjustedBase
                  ? encodingInfoOr->baseOffset - alreadyAdjustedBase
                  : 0;
        return lowerTMemLdStFromInfo(
            loc, rewriter, *encodingInfoOr, pred, llvmElemTy, vals, tmemBase,
            redOp, useAbs, useNaN);
      }
    } else if (debugQuerySelection && !rawError.empty()) {
      llvm::errs() << "[tmem-ldst] rawQuery fail: " << rawError << "\n";
    }
  }
  std::optional<MemDescType> firstQueryTy;
  std::optional<TMemLdStRowPlan> firstQueryRowPlan;
  if (!disallowQueryTypeRescueForRowZeroLiftedReinterpret)
    for (MemDescType queryTy : queryTypes) {
    auto rowPlan = disallowSupportRescueFor32x32Subview
                       ? std::optional<TMemLdStRowPlan>{}
                       : (memDescValue ? getTMemLdStRowPlanForQuery(memDescValue,
                                                                    queryTy)
                                       : getTMemLdStRowPlanForType(queryTy));
    if (debugQuerySelection) {
      llvm::errs() << "[tmem-ldst] queryTy=" << queryTy << " rowPlan="
                   << (rowPlan ? llvm::Twine(rowPlan->rowSpan).str()
                               : std::string("none"))
                   << "\n";
    }
    if (!firstQueryTy) {
      firstQueryTy = queryTy;
      firstQueryRowPlan = rowPlan;
    }
    auto encodingInfoOr =
        computeTMemLdStEncodingInfo(regTy, queryTy, maxnreg, /*emitError=*/{},
                                    rowPlan);
    appendTrace(Twine("queryType ") +
                Twine(queryTy.getShape()[0]) + "x" +
                Twine(queryTy.getShape()[1]) + " " +
                (succeeded(encodingInfoOr)
                     ? (Twine("ok atom=") +
                        Twine(static_cast<int>(encodingInfoOr->atom)) +
                        " regsPerMsg=" +
                        Twine(encodingInfoOr->numRegsPerMessage) +
                        " baseOffset=" + Twine(encodingInfoOr->baseOffset) +
                        " warpBase0=" +
                        Twine(encodingInfoOr->warpBaseOffset0) +
                        " warpBase1=" +
                        Twine(encodingInfoOr->warpBaseOffset1) +
                        " reps=" + encodingInfoOr->reps.toString())
                     : Twine("fail")));
    if (succeeded(encodingInfoOr)) {
      if (memDescValue &&
          isa_and_nonnull<TMEMSubSliceOp, triton::gpu::MemDescSubsliceOp,
                          triton::gpu::MemDescIndexOp,
                          triton::gpu::MemDescReshapeOp>(
              memDescValue.getDefiningOp()) &&
          regTy.getRank() == 2 && regTy.getShape()[0] == 32 &&
          regTy.getShape()[1] == 32 &&
          encodingInfoOr->atom == TMemAccessAtom::I32x32b &&
          encodingInfoOr->numRegsPerMessage > 1) {
        if (auto scalarEncodingInfoOr = computeTMemLdStEncodingInfo(
                regTy, queryTy, /*maxnreg=*/std::min(maxnreg, 2),
                /*emitError=*/{}, rowPlan);
            succeeded(scalarEncodingInfoOr) &&
            scalarEncodingInfoOr->atom == encodingInfoOr->atom &&
            scalarEncodingInfoOr->numRegsPerMessage == 1) {
          encodingInfoOr = std::move(scalarEncodingInfoOr);
        } else {
          encodingInfoOr->numRegsPerMessage = 1;
        }
      }
      return lowerTMemLdStFromInfo(
          loc, rewriter, *encodingInfoOr, pred, llvmElemTy, vals, tmemBase,
          redOp, useAbs, useNaN);
    }
  }
  if (rawQueryLayout) {
    (void)computeTMemLdStEncodingInfo(regTy, memTy, rawQueryLayout->layout,
                                      maxnreg, diag, rawRowPlan);
  } else if (firstQueryTy) {
    (void)computeTMemLdStEncodingInfo(regTy, *firstQueryTy, maxnreg, diag,
                                      firstQueryRowPlan);
  }
  if (queryTypes.empty())
    return failure();
  return failure();
}

// Combine partial reductions into one value per thread via tree reduction.
static void combinePartialReductions(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     SmallVector<Value> &redvalVals,
                                     TMEMLoadReduceModifier redOp,
                                     bool useNaN) {
  if (redvalVals.size() <= 1)
    return;
  auto isMin = redOp == TMEMLoadReduceModifier::MIN;
  auto applyMinMax = [&](Value lhs, Value rhs) {
    return useNaN ? (isMin ? LLVM::MinimumOp::create(rewriter, loc, lhs, rhs)
                           : LLVM::MaximumOp::create(rewriter, loc, lhs, rhs))
                        ->getResult(0)
                  : (isMin ? LLVM::MinNumOp::create(rewriter, loc, lhs, rhs)
                           : LLVM::MaxNumOp::create(rewriter, loc, lhs, rhs))
                        ->getResult(0);
  };
  // Use tree reduction: pair up elements at each level
  while (redvalVals.size() > 1) {
    SmallVector<Value> reduced;
    assert(redvalVals.size() % 2 == 0 && "redvalVals must be a multiple of 2");
    for (size_t i = 0; i < redvalVals.size(); i += 2) {
      reduced.push_back(applyMinMax(redvalVals[i], redvalVals[i + 1]));
    }
    redvalVals = std::move(reduced);
  }
}

struct TensorMemoryLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = op.getContext();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getSrc().getType().getElementType());
    auto tmemBase = adaptor.getSrc();
    auto regTy = cast<RankedTensorType>(op.getType());
    auto memTy = cast<MemDescType>(op.getSrc().getType());

    // Extract reduction attributes
    auto redOp = op.getRedOp();
    auto useAbs = op.getAbs().value_or(false);
    auto useNaN = op.getNaN().value_or(false);
    if (redOp) {
      auto redTy = cast<RankedTensorType>(op.getRed().getType());
      assert(getTotalElemsPerThread(redTy) == 1 &&
             "reduction layout must produce exactly one value per thread");
    }

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto maxnreg = getContextualMaxNReg(op);
    auto lowered = lowerTMemLdStFromTypes(
        loc, rewriter, regTy, memTy, op.getSrc(), tmemBase, maxnreg,
        b.i1_val(true),
        llvmElemTy, {}, redOp, useAbs, useNaN);
    if (failed(lowered))
      return failure();
    auto [resultVals, redvalVals] = *lowered;

    Type structTy = getTypeConverter()->convertType(op.getType());
    Value resultStruct =
        packLLElements(loc, getTypeConverter(), resultVals, rewriter, structTy);
    // Wait insertion could be moved to the TTGIR level if needed.
    NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::LOAD);

    // tcgen05.ld.red is async, redval registers aren't valid until the wait
    if (redOp)
      combinePartialReductions(loc, rewriter, redvalVals, *redOp, useNaN);

    // Handle reduction output if present
    SmallVector<Value> results = {resultStruct};
    if (redOp) {
      // Pack redval values into the red tensor result
      Type redStructTy = getTypeConverter()->convertType(op.getRed().getType());
      Value redStruct = packLLElements(loc, getTypeConverter(), redvalVals,
                                       rewriter, redStructTy);
      results.push_back(redStruct);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct TensorMemoryStoreOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMStoreOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto ctx = op.getContext();
    auto llvmElemTy =
        getTypeConverter()->convertType(op.getDst().getType().getElementType());

    auto tmemBase = adaptor.getDst();
    Value pred = adaptor.getPred();
    auto memTy = cast<MemDescType>(op.getDst().getType());
    auto regTy = cast<RankedTensorType>(op.getSrc().getType());
    auto b = TritonLLVMOpBuilder(loc, rewriter);

    SmallVector<Value> srcValues =
        unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto maxnreg = getContextualMaxNReg(op);
    if (failed(lowerTMemLdStFromTypes(loc, rewriter, regTy, memTy, op.getDst(),
                                      tmemBase, maxnreg, pred, llvmElemTy,
                                      srcValues)))
      return failure();
    NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::STORE);

    // Emit a barrier to ensure all threads have finished writing to tensor
    // memory before any use of the tensor memory.
    // Can be AddrSpace::TensorWrite if we emit
    // NVVM::Tcgen05WaitKind::STORE during barrier lowering
    b.barrier(triton::gpu::AddrSpace::None);

    rewriter.eraseOp(op);
    return success();
  }
};

struct TensorMemoryAllocOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMAllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto ctx = op.getContext();
    Value base = nvgpu::TensorMemoryBaseAddress::create(rewriter, loc);
    Value baseInt = b.ptrtoint(i32_ty, base);
    int colOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_col_offset"))
                        .getValue()
                        .getZExtValue();
    int rowOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_row_offset"))
                        .getValue()
                        .getZExtValue();
    Value allocAddress = b.add(baseInt, b.i32_val(colOffset | rowOffset << 16));
    SmallVector<unsigned> order(op.getType().getRank());
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());
    auto shape = op.getType().getShape();

    if (op.getSrc()) {
      auto regTy = cast<RankedTensorType>(op.getSrc().getType());
      auto memTy = cast<MemDescType>(op.getResult().getType());
      auto llvmElemTy = getTypeConverter()->convertType(regTy.getElementType());
      auto maxnreg = getContextualMaxNReg(op);
      SmallVector<Value> srcValues =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value ptr = b.inttoptr(base.getType(), allocAddress);
      // Initialized allocs still need the real memdesc-value query path so
      // TMEM-linear allocations keep their backing-row support form instead of
      // collapsing to a narrower standalone query.
      if (failed(lowerTMemLdStFromTypes(loc, rewriter, regTy, memTy,
                                        /*memDescValue=*/op.getResult(),
                                        ptr, maxnreg, b.i1_val(true),
                                        llvmElemTy, srcValues)))
        return failure();
      NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::STORE);
      // Emit a barrier to ensure all threads have finished writing to tensor
      // memory before any use of the tensor memory.
      // Can be AddrSpace::TensorWrite if we emit
      // NVVM::Tcgen05WaitKind::STORE during barrier lowering
      b.barrier(triton::gpu::AddrSpace::None);
    }
    // Cast to address space 3 as the shared memory object uses 3.
    // TODO: clean this up and use either a int or ptr address space 6
    auto ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 3);
    Value ptr = b.inttoptr(ptrTy, allocAddress);
    rewriter.replaceOp(op, ptr);
    return success();
  }
};

static void createCommit(ConversionPatternRewriter &rewriter, Location loc,
                         Value barrier, Value pred, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  auto *barrierOperand = ptxBuilder.newAddrOperand(barrier, "r");
  std::string opcode =
      "tcgen05.commit.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".mbarrier::arrive::one.shared::cluster.b64";
  auto &barrierOp = *ptxBuilder.create(opcode);
  barrierOp(barrierOperand).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createTcgen05Cp(ConversionPatternRewriter &rewriter, Location loc,
                            Value tmem_address, Value src_desc, Value pred,
                            TMemCopyAtom atom, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  auto dst = ptxBuilder.newAddrOperand(tmem_address, "r");
  auto src = ptxBuilder.newOperand(src_desc, "l");
  std::string warp;
  if (atom.multicast == 1) {
    warp = ".warpx2::01_23";
  } else if (atom.multicast == 2) {
    warp = ".warpx2::02_13";
  } else if (atom.multicast == 3) {
    warp = ".warpx4";
  }
  std::string opcode =
      "tcgen05.cp.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + warp + "." +
      std::to_string(atom.nRow) + "x" + std::to_string(atom.bCol) + "b";
  auto &op = *ptxBuilder.create(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static LogicalResult copySharedToTmem(ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      const TypeConverter *typeConverter,
                                      triton::nvidia_gpu::TMEMCopyOp op,
                                      Value src, Value baseDst, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto *ctx = op.getContext();
  auto kOffset = str_attr("offset");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");
  auto kBlock = str_attr("block");

  MemDescType srcTy = op.getSrc().getType();
  MemDescType dstTy = op.getDst().getType();
  auto shmemLl = toLinearLayout(srcTy);
  std::string tmemError;
  auto maybeStandaloneDstTy = inferStandaloneTMemViewType(op.getDst(), &tmemError);
  if (failed(maybeStandaloneDstTy)) {
    return op->emitOpError(tmemError.empty()
                               ? "unsupported tensor memory descriptor view "
                                 "for tcgen05.copy lowering"
                               : tmemError);
  }
  auto tmemLl = toLinearLayout(*maybeStandaloneDstTy);
  bool isScales = isa<TensorMemoryScalesEncodingAttr>(dstTy.getEncoding());

  // This subtlely handles subviews
  auto cvt = tmemLl.invertAndCompose(shmemLl);

  auto bitwidth = srcTy.getElementType().getIntOrFloatBitWidth();
  auto copyPlans = getTMemCopyPlans(cvt, bitwidth);
  if (copyPlans.empty()) {
    return op->emitOpError("failed to classify tcgen05.copy family from "
                           "shared memory descriptor ")
           << srcTy << " to tensor memory descriptor " << dstTy;
  }
  // Get shmem ptr
  Type elemTy = typeConverter->convertType(srcTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, elemTy, rewriter);
  auto smemBase = smemObj.getShmemAffineBase(loc, rewriter, srcTy);

  struct PlannedCopyMessage {
    TMemCopyMessagePlan plan;
    std::optional<DotOpMmaSmemLoader> loader;
    std::optional<uint64_t> directSeedDescriptorImm;
  };
  SmallVector<PlannedCopyMessage, 2> plannedMessages;
  std::optional<TMemCopyPlan> selectedPlan;
  for (const auto &plan : copyPlans) {
    if (!isScales &&
        !isDirectTMemCopyLayoutSupported(*maybeStandaloneDstTy, plan.family))
      continue;
    SmallVector<PlannedCopyMessage, 2> candidateMessages;
    candidateMessages.reserve(plan.messages.size());
    bool validPlan = true;
    for (const auto &message : plan.messages) {
      if (message.useDirectSeedDescriptor) {
        if (auto seedDescImm =
                getDirectTMemCopySeedDescriptorImm(srcTy, plan.family)) {
          candidateMessages.push_back(
              PlannedCopyMessage{message, std::nullopt, *seedDescImm});
          continue;
        }
      }
      bool foundDescriptorLayout = false;
      for (const auto &srcDescLayout :
           getTMemCopyDescriptorLayouts(srcTy, shmemLl, cvt, message)) {
        for (unsigned mnDim : {0u, 1u}) {
          auto loader = DotOpMmaSmemLoader::build(
              loc, rewriter, srcDescLayout, bitwidth, smemBase,
              message.descriptorShape, mnDim, 5);
          if (failed(loader) || loader->getDescriptor().transposed)
            continue;
          PlannedCopyMessage plannedMessage{message, *loader, std::nullopt};
          candidateMessages.push_back(std::move(plannedMessage));
          foundDescriptorLayout = true;
          break;
        }
        if (foundDescriptorLayout)
          break;
      }
      if (!foundDescriptorLayout) {
        validPlan = false;
        break;
      }
    }
    if (!validPlan)
      continue;
    selectedPlan = plan;
    plannedMessages = std::move(candidateMessages);
    break;
  }
  if (!selectedPlan) {
    if (isScales) {
      StringRef family = stringifyTMemCopyFamily(copyPlans.front().family);
      auto diag =
          op->emitOpError("The source shared layout maps to tcgen05.copy.")
          << family
          << ", but Triton could not synthesize a compatible shared-memory "
             "descriptor plan for tensor memory scales.";
      diag.attachNote()
          << "Use a shared layout that lowers to tcgen05.copy." << family
          << ", or reshape / permute the shared tile until it lowers to the "
             "same descriptor family.";
      diag.attachNote()
          << "This is reported during lowering because the final "
             "shared-memory descriptor layout is only known after shared "
             "memory allocation.";
      return failure();
    }
    return op->emitOpError("failed to find valid tcgen05.copy layout from "
                           "shared memory descriptor ")
           << srcTy << " to tensor memory descriptor " << dstTy;
  }

  bool twoCTAs = getModuleTwoCTAs(op);
  // Check correct lbo/sbo along the multicast
  bool usesDirectSeedDescriptor =
      llvm::any_of(plannedMessages, [](const PlannedCopyMessage &message) {
        return message.directSeedDescriptorImm.has_value();
      });
  auto strideRow = cvt.getBasis(kRow, llvm::Log2_32(8), kOffset);
  const auto &copyAtom = plannedMessages.front().plan.atom;
  if (!usesDirectSeedDescriptor) {
    if ((copyAtom.multicast & 1) == 0) {
      assert(cvt.getBasis(kRow, llvm::Log2_32(32), kOffset) ==
             strideRow * (32 / 8));
    }
    if (copyAtom.multicast != 1 && (copyAtom.multicast & 2) == 0) {
      assert(cvt.getBasis(kRow, llvm::Log2_32(64), kOffset) ==
             strideRow * (64 / 8));
    }
  }

  const unsigned colStride = plannedMessages.front().plan.instrShape[1];
  for (int col = 0; col < cvt.getInDimSize(kCol); col += colStride) {
    for (const auto &message : plannedMessages) {
      Value desc;
      if (message.directSeedDescriptorImm) {
        uint64_t sourceOffsetB128 =
            message.plan.directSourceOffsetB128 +
            ((col + message.plan.smemColOffset) * bitwidth) / 128;
        uint64_t descImm = *message.directSeedDescriptorImm;
        descImm &= ~(((1ULL << 14) - 1) | (0x7ULL << 49));
        descImm |= sourceOffsetB128;
        descImm |= ((sourceOffsetB128 >> 3) & 0x7ULL) << 49;
        Value baseSrcb128 =
            b.lshr(b.ptrtoint(i32_ty, smemBase), b.i32_val(4));
        Value baseb128 =
            b.zext(i64_ty, b.and_(baseSrcb128, b.i32_val(0x3FFF)));
        desc = b.add(b.int_val(64, descImm), baseb128);
      } else {
        desc = message.loader->smemLoad(
            message.plan.smemRow, col + message.plan.smemColOffset, rewriter,
            loc);
      }
      auto tmemAddr = b.add(
          b.ptrtoint(i32_ty, baseDst),
          b.i32_val(message.plan.tmemDwordDelta + col * bitwidth / 32));
      createTcgen05Cp(rewriter, loc, tmemAddr, desc, pred, message.plan.atom,
                      twoCTAs);
    }
  }
  return success();
}

struct TensorMemoryCopyOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMCopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMCopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);
    bool twoCTAs = getModuleTwoCTAs(op);
    // Similar to twoCTA tcgen05.mma, the 2CTA version of this op should only be
    // emitted from the lead CTA.
    if (twoCTAs) {
      Value cluster0 = LLVM::NVIDIA::createLeadCTAPredicate(loc, rewriter);
      pred = TritonLLVMOpBuilder(loc, rewriter).and_(pred, cluster0);
    }

    if (failed(copySharedToTmem(rewriter, loc, typeConverter, op,
                                adaptor.getSrc(), adaptor.getDst(), pred)))
      return failure();

    if (op.getBarrier()) {
      auto barrier = LLVM::getSharedMemoryObjectFromStruct(
          op.getLoc(), adaptor.getBarrier(), i64_ty, rewriter);
      createCommit(rewriter, loc, barrier.getBase(), pred, twoCTAs);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct MemDescIndexOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::MemDescIndexOp> {
  using ConvertOpToLLVMPattern<
      triton::gpu::MemDescIndexOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::gpu::MemDescIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto *ctx = op->getContext();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto dstTy = op.getResult().getType();
    if (!isTensorMemoryEncoding(srcTy.getEncoding()) ||
        isa<TensorMemoryScalesEncodingAttr>(srcTy.getEncoding()) ||
        !isTensorMemoryEncoding(dstTy.getEncoding()) ||
        isa<TensorMemoryScalesEncodingAttr>(dstTy.getEncoding())) {
      return failure();
    }

    auto ll = triton::nvidia_gpu::getCanonicalTensorMemoryLinearLayout(srcTy);
    int layoutRank = ll.getNumOutDims();
    Value tmemBase = adaptor.getSrc();
    uint32_t bitwidth = srcTy.getElementTypeBitWidth();
    if (srcTy.getRank() > layoutRank) {
      auto kCol = StringAttr::get(ctx, "col");
      int singleBufferCols = ll.getInDimSize(kCol) / (32 / bitwidth);
      int64_t prefixStride = product<int64_t>(srcTy.getShape().drop_front().take_front(
          srcTy.getRank() - layoutRank - 1));
      Value offset =
          b.mul(op.getIndex(), b.i32_val(singleBufferCols * prefixStride));
      Value newBase = b.add(b.ptrtoint(i32_ty, tmemBase), offset);
      rewriter.replaceOp(op, b.inttoptr(ptr_ty(ctx, 3), newBase));
      return success();
    }

    APInt index;
    if (!matchPattern(op.getIndex(), m_ConstantInt(&index))) {
      return rewriter.notifyMatchFailure(
          op, "dynamic tensor memory indexing is only supported for the "
              "unencoded leading buffer dimension");
    }

    SmallVector<int32_t> offsets(srcTy.getRank(), 0);
    offsets.front() = index.getSExtValue();
    rewriter.replaceOp(
        op, advanceTensorMemoryBase(loc, rewriter, tmemBase,
                                    triton::nvidia_gpu::getTMemViewOffset(
                                        srcTy, offsets)));
    return success();
  }
};

class MemDescReinterpretOpConversion
    : public ConvertOpToLLVMPattern<MemDescReinterpretOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MemDescReinterpretOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = op.getSrc().getType();
    auto tmem =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(srcTy.getContext());
    if (srcTy.getMemorySpace() != tmem) {
      return failure();
    }
    rewriter.replaceOp(op, adaptor.getSrc());
    return success();
  }
};

struct TMEMSubSliceOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMSubSliceOp> {
  using ConvertOpToLLVMPattern<
      triton::nvidia_gpu::TMEMSubSliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMSubSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    // Physical TMEM pointer arithmetic is defined in the source tile's address
    // space. Using the narrowed result type can erase high-order column bits
    // for N-half views (for example 128x256 -> 128x128), collapsing distinct
    // subslices onto the same base address.
    uint32_t offset = getTMemSubSliceOffset(srcTy, op.getN());

    Value tmemBase = adaptor.getSrc();
    Value offsetVal = b.i32_val(offset);
    Value newBase = b.add(b.ptrtoint(i32_ty, tmemBase), offsetVal);
    auto elemPtrTy = ptr_ty(rewriter.getContext(), 3);
    rewriter.replaceOp(op, b.inttoptr(elemPtrTy, newBase));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<TensorMemoryCopyOpConversion, TensorMemoryLoadOpConversion,
               TensorMemoryStoreOpConversion, TensorMemoryAllocOpConversion>(
      typeConverter, benefit);
}

void mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescReinterpretOpConversion, MemDescIndexOpConversion,
               TMEMSubSliceOpConversion>(typeConverter, benefit);
}

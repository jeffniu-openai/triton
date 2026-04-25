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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::NVIDIA;

namespace {

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

static Value getTMemSubwordPhase(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Value tmemBase,
                                 uint32_t tmemElementBitwidth) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value base = b.ptrtoint(i32_ty, tmemBase);
  Value elementCol = b.and_(base, b.i32_val(kTMemPackedOffsetColMask));
  uint32_t elementsPerWord = getTMemElementsPerWord(tmemElementBitwidth);
  return b.and_(elementCol, b.i32_val(elementsPerWord - 1));
}

static SmallVector<Value>
realignPackedSubwordLoadWords(Location loc,
                              ConversionPatternRewriter &rewriter,
                              ArrayRef<Value> words, Value phase,
                              uint32_t tmemElementBitwidth) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(words.size() >= 2 &&
         "phase-aware subword load needs one tail word");
  Value phaseBits = b.mul(phase, b.i32_val(tmemElementBitwidth));
  Value phaseNonZero = b.icmp_ne(phase, b.i32_val(0));
  Value inverseShift = b.sub(b.i32_val(32), phaseBits);
  Value inverseShiftSafe =
      b.select(phaseNonZero, inverseShift, b.i32_val(0));

  SmallVector<Value> realigned;
  realigned.reserve(words.size() - 1);
  for (unsigned i = 0, e = words.size() - 1; i < e; ++i) {
    Value low = b.lshr(words[i], phaseBits);
    Value high = b.shl(words[i + 1], inverseShiftSafe);
    Value shifted = b.or_(low, high);
    realigned.push_back(b.select(phaseNonZero, shifted, words[i]));
  }
  return realigned;
}

static std::pair<SmallVector<Value>, Value>
realignPackedSubwordStoreWords(Location loc,
                               ConversionPatternRewriter &rewriter,
                               ArrayRef<Value> words, Value oldFirstWord,
                               Value oldTailWord, Value phase,
                               uint32_t tmemElementBitwidth) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(!words.empty() && "phase-aware subword store needs data words");
  Value phaseBits = b.mul(phase, b.i32_val(tmemElementBitwidth));
  Value phaseNonZero = b.icmp_ne(phase, b.i32_val(0));
  Value inverseShift = b.sub(b.i32_val(32), phaseBits);
  Value inverseShiftSafe =
      b.select(phaseNonZero, inverseShift, b.i32_val(0));
  Value lowMask = b.sub(b.shl(b.i32_val(1), phaseBits), b.i32_val(1));
  Value highMask = b.xor_(lowMask, b.i32_val(-1));

  SmallVector<Value> realigned;
  realigned.reserve(words.size());
  Value first = b.or_(b.and_(oldFirstWord, lowMask),
                      b.shl(words.front(), phaseBits));
  realigned.push_back(b.select(phaseNonZero, first, words.front()));
  for (unsigned i = 1, e = words.size(); i < e; ++i) {
    Value fromPrev = b.lshr(words[i - 1], inverseShiftSafe);
    Value fromCur = b.shl(words[i], phaseBits);
    Value shifted = b.or_(fromPrev, fromCur);
    realigned.push_back(b.select(phaseNonZero, shifted, words[i]));
  }

  Value tail = b.or_(b.and_(oldTailWord, highMask),
                     b.lshr(words.back(), inverseShiftSafe));
  return {std::move(realigned), tail};
}

static FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
lowerContiguousPackedSubwordPhaseAwareLdSt(
    Location loc, ConversionPatternRewriter &rewriter,
    const TMemLdStEncodingInfo &info, Value pred, Type llvmElemTy,
    uint32_t tmemElementBitwidth, ArrayRef<Value> vals, Value tmemBase,
    std::optional<TMEMLoadReduceModifier> redOp) {
  auto unsupported = [&]() -> LogicalResult {
    emitError(loc)
        << "unsupported sub-32-bit TMEM view origin for this ld/st layout: "
           "phase-aware lowering currently supports contiguous packed "
           "32x32b load/store plans";
    return failure();
  };
  if (tmemElementBitwidth >= 32 || llvmElemTy.getIntOrFloatBitWidth() != 32 ||
      redOp || info.atom != TMemAccessAtom::I32x32b || info.unpacked ||
      info.secondHalfOffset || !info.packetOffsets.empty()) {
    (void)unsupported();
    return failure();
  }

  auto *ctx = rewriter.getContext();
  auto kReg = str_attr("register");
  auto kWarp = str_attr("warp");
  if (!info.reps.hasInDim(kReg) ||
      info.reps.getInDimSize(kReg) != info.numRegsPerMessage ||
      info.numRegsPerMessage <= 0 || info.baseOffset != 0 ||
      info.warpBaseOffset0 != getTMemPackedOffsetRowBase(32u) ||
      info.warpBaseOffset1 != getTMemPackedOffsetRowBase(64u) ||
      info.reps.getInDimSize(kWarp) > 4) {
    (void)unsupported();
    return failure();
  }

  bool isStore = !vals.empty();
  if (isStore && static_cast<int>(vals.size()) != info.numRegsPerMessage) {
    (void)unsupported();
    return failure();
  }

  Value phase = getTMemSubwordPhase(loc, rewriter, tmemBase,
                                    tmemElementBitwidth);
  tmemBase = LLVM::NVIDIA::projectTMemElementBaseToWordBase(
      loc, rewriter, tmemBase, tmemElementBitwidth);

  // TMEM base values are encoded in hardware word columns after projection, so
  // the scalar tail message advances by one word-column per packed register.
  uint32_t tailColOffset = static_cast<uint32_t>(info.numRegsPerMessage);
  if (!isStore) {
    auto [packed, _] = createTensorMemoryLoad(
        loc, rewriter.getContext(), tmemBase, /*colOffset=*/0,
        /*secondHalfOffset=*/std::nullopt, /*unpacked=*/false,
        info.numRegsPerMessage, TMemAccessAtom::I32x32b, /*redOp=*/std::nullopt,
        /*useAbs=*/false, /*useNaN=*/false, llvmElemTy, rewriter);
    SmallVector<Value> words =
        unpackResults(packed, llvmElemTy, info.numRegsPerMessage, loc,
                      rewriter);
    auto [tail, __] = createTensorMemoryLoad(
        loc, rewriter.getContext(), tmemBase,
        /*colOffset=*/static_cast<int>(tailColOffset),
        /*secondHalfOffset=*/std::nullopt, /*unpacked=*/false,
        /*numRegPerMessage=*/1, TMemAccessAtom::I32x32b,
        /*redOp=*/std::nullopt, /*useAbs=*/false, /*useNaN=*/false, llvmElemTy,
        rewriter);
    words.push_back(tail);
    return std::make_pair(
        realignPackedSubwordLoadWords(loc, rewriter, words, phase,
                                      tmemElementBitwidth),
        SmallVector<Value>{});
  }

  auto [oldFirst, _] = createTensorMemoryLoad(
      loc, rewriter.getContext(), tmemBase, /*colOffset=*/0,
      /*secondHalfOffset=*/std::nullopt, /*unpacked=*/false,
      /*numRegPerMessage=*/1, TMemAccessAtom::I32x32b, /*redOp=*/std::nullopt,
      /*useAbs=*/false, /*useNaN=*/false, llvmElemTy, rewriter);
  auto [oldTail, __] = createTensorMemoryLoad(
      loc, rewriter.getContext(), tmemBase,
      /*colOffset=*/static_cast<int>(tailColOffset),
      /*secondHalfOffset=*/std::nullopt, /*unpacked=*/false,
      /*numRegPerMessage=*/1, TMemAccessAtom::I32x32b, /*redOp=*/std::nullopt,
      /*useAbs=*/false, /*useNaN=*/false, llvmElemTy, rewriter);
  NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::LOAD);

  auto [realignedWords, tailWord] = realignPackedSubwordStoreWords(
      loc, rewriter, vals, oldFirst, oldTail, phase, tmemElementBitwidth);
  createTensorMemoryStore(loc, tmemBase, /*colOffset=*/0, realignedWords,
                          /*secondHalfOffset=*/std::nullopt, pred,
                          /*unpacked=*/false, TMemAccessAtom::I32x32b,
                          rewriter);
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value tailPred = b.and_(pred, b.icmp_ne(phase, b.i32_val(0)));
  SmallVector<Value> tailWords = {tailWord};
  createTensorMemoryStore(loc, tmemBase,
                          /*colOffset=*/static_cast<int>(tailColOffset),
                          tailWords, /*secondHalfOffset=*/std::nullopt,
                          tailPred, /*unpacked=*/false,
                          TMemAccessAtom::I32x32b, rewriter);
  return std::make_pair(SmallVector<Value>{}, SmallVector<Value>{});
}

static bool hasZeroBasisAlong(const LinearLayout &layout, StringAttr dim) {
  if (!layout.hasInDim(dim))
    return false;
  for (unsigned idx = 0; idx < layout.getInDimSizeLog2(dim); ++idx) {
    if (llvm::all_of(layout.getBasis(dim, idx),
                     [](int32_t value) { return value == 0; }))
      return true;
  }
  return false;
}

static Value packDynamicTMemRowElementCol(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         Value row, Value col) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  return b.or_(b.shl(row, b.i32_val(16)), col, /*disjoint=*/true);
}

static Value zextSubwordValueToI32(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Value value, Type elemTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  Value bits = elemTy.isInteger() ? value : b.bitcast(value, int_ty(bitwidth));
  return bitwidth == 32 ? bits : b.zext(i32_ty, bits);
}

static uint32_t getSubwordValueMask(unsigned bitwidth) {
  assert(bitwidth > 0 && bitwidth < 32 &&
         "subword value masks are only valid below one hardware word");
  return (1u << bitwidth) - 1u;
}

static Value extractSubwordValue(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Value word, Value phase, Type elemTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  uint32_t lowMask = getSubwordValueMask(bitwidth);
  Value phaseBits = b.mul(phase, b.i32_val(bitwidth));
  Value shifted = b.lshr(word, phaseBits);
  Value masked = b.and_(shifted, b.i32_val(lowMask));
  Value narrowed = b.trunc(int_ty(bitwidth), masked);
  return elemTy.isInteger() ? narrowed : b.bitcast(narrowed, elemTy);
}

static Value mergeSubwordValue(Location loc,
                               ConversionPatternRewriter &rewriter,
                               Value oldWord, Value newValue, Value phase,
                               Type elemTy) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned bitwidth = elemTy.getIntOrFloatBitWidth();
  uint32_t lowMask = getSubwordValueMask(bitwidth);
  Value phaseBits = b.mul(phase, b.i32_val(bitwidth));
  Value laneMask = b.shl(b.i32_val(lowMask), phaseBits);
  Value valueBits = b.shl(b.and_(zextSubwordValueToI32(loc, rewriter, newValue,
                                                       elemTy),
                                b.i32_val(lowMask)),
                          phaseBits);
  return b.or_(b.and_(oldWord, b.xor_(laneMask, b.i32_val(-1))), valueBits);
}

// Fallback for legal packed-subword layouts that cannot be grouped into a
// contiguous tcgen05 subword message. The layout still describes each element's
// physical row/element-column; this path scalarizes through 32-bit RMW words and
// derives the active packed-lane phase from the current runtime taddr.
static FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>>
lowerElementwisePackedSubwordLdSt(
    Location loc, ConversionPatternRewriter &rewriter, RankedTensorType regTy,
    const TMemLdStQueryLayout &query, Value pred, Type llvmElemTy,
    uint32_t tmemElementBitwidth, ArrayRef<Value> vals, Value tmemBase,
    std::optional<TMEMLoadReduceModifier> redOp) {
  if (tmemElementBitwidth >= 32 || redOp)
    return failure();
  if (tmemElementBitwidth != llvmElemTy.getIntOrFloatBitWidth())
    return failure();
  if (!query.origin.empty() &&
      llvm::any_of(query.origin, [](int32_t value) { return value != 0; }))
    return failure();

  auto *ctx = rewriter.getContext();
  auto kReg = str_attr("register");
  auto kLane = str_attr("lane");
  auto kWarp = str_attr("warp");
  auto kBlock = str_attr("block");
  auto kRow = str_attr("row");
  auto kCol = str_attr("col");
  if (!hasZeroBasisAlong(query.layout, kCol))
    return failure();

  auto squeezeTrivialBlock = [&](LinearLayout layout) {
    if (layout.hasInDim(kBlock) && layout.getInDimSize(kBlock) == 1)
      layout = layout.squeezeIns(kBlock);
    if (layout.hasOutDim(kBlock) && layout.getOutDimSize(kBlock) == 1)
      layout = layout.squeezeOuts(kBlock);
    return layout;
  };
  LinearLayout regLayout =
      squeezeTrivialBlock(toLinearEncoding(regTy).getLinearLayout());
  LinearLayout memLayout = squeezeTrivialBlock(query.layout);
  if (!canInvertAndComposeLayouts(regLayout, memLayout))
    return failure();
  LinearLayout cvt = squeezeTrivialBlock(regLayout.invertAndCompose(memLayout));

  if (!cvt.hasOutDim(kRow) || !cvt.hasOutDim(kCol) || !cvt.hasInDim(kReg) ||
      !cvt.hasInDim(kLane))
    return failure();
  if (cvt.getInDimSize(kReg) != getTotalElemsPerThread(regTy))
    return failure();
  if (!vals.empty() && static_cast<int64_t>(vals.size()) != cvt.getInDimSize(kReg))
    return failure();
  if (cvt.getInDimSizeLog2(kLane) < 5)
    return failure();
  for (unsigned idx = 0; idx < 5; ++idx) {
    if (cvt.getBasis(kLane, idx, kRow) != static_cast<int32_t>(1u << idx) ||
        cvt.getBasis(kLane, idx, kCol) != 0)
      return failure();
  }
  if (cvt.hasInDim(kBlock) && cvt.getInDimSize(kBlock) != 1)
    return failure();

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool isStore = !vals.empty();
  SmallVector<Value> resultVals;
  if (!isStore)
    resultVals.reserve(cvt.getInDimSize(kReg));

  Value warpId = cvt.hasInDim(kWarp) ? WarpIdOp::create(rewriter, loc)
                                     : b.i32_val(0);
  auto getRowCol = [&](int regIdx) -> std::optional<std::pair<Value, Value>> {
    SmallVector<std::pair<StringAttr, Value>> inputs;
    inputs.reserve(cvt.getNumInDims());
    for (StringAttr inDim : cvt.getInDimNames()) {
      if (inDim == kReg)
        inputs.push_back({inDim, b.i32_val(regIdx)});
      else if (inDim == kLane)
        inputs.push_back({inDim, b.i32_val(0)});
      else if (inDim == kWarp)
        inputs.push_back({inDim, warpId});
      else if (inDim == kBlock)
        inputs.push_back({inDim, b.i32_val(0)});
      else
        return std::nullopt;
    }
    Value row;
    Value col;
    for (auto [outDim, value] : applyLinearLayout(loc, rewriter, cvt, inputs)) {
      if (outDim == kRow)
        row = value;
      else if (outDim == kCol)
        col = value;
    }
    if (!row || !col)
      return std::nullopt;
    return std::make_pair(row, col);
  };

  for (int regIdx = 0; regIdx < cvt.getInDimSize(kReg); ++regIdx) {
    auto rowCol = getRowCol(regIdx);
    if (!rowCol)
      return failure();
    auto [row, col] = *rowCol;
    Value elementBase = advanceTensorMemoryBase(
        loc, rewriter, tmemBase,
        packDynamicTMemRowElementCol(loc, rewriter, row, col));
    Value phase = getTMemSubwordPhase(loc, rewriter, elementBase,
                                      tmemElementBitwidth);
    Value wordBase = LLVM::NVIDIA::projectTMemElementBaseToWordBase(
        loc, rewriter, elementBase, tmemElementBitwidth);
    auto [oldWord, _] = createTensorMemoryLoad(
        loc, ctx, wordBase, /*colOffset=*/0, /*secondHalfOffset=*/std::nullopt,
        /*unpacked=*/false, /*numRegPerMessage=*/1, TMemAccessAtom::I32x32b,
        /*redOp=*/std::nullopt, /*useAbs=*/false, /*useNaN=*/false, i32_ty,
        rewriter);
    NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::LOAD);
    if (!isStore) {
      resultVals.push_back(
          extractSubwordValue(loc, rewriter, oldWord, phase, llvmElemTy));
      continue;
    }
    SmallVector<Value> merged = {
        mergeSubwordValue(loc, rewriter, oldWord, vals[regIdx], phase,
                          llvmElemTy)};
    createTensorMemoryStore(loc, wordBase, /*colOffset=*/0, merged,
                            /*secondHalfOffset=*/std::nullopt, pred,
                            /*unpacked=*/false, TMemAccessAtom::I32x32b,
                            rewriter);
    NVVM::Tcgen05WaitOp::create(rewriter, loc, NVVM::Tcgen05WaitKind::STORE);
  }

  return std::make_pair(std::move(resultVals), SmallVector<Value>{});
}

// Returns {resultVals, redvalVals} where redvalVals is empty if no reduction.
// Reduction produces exactly one value per thread; if multiple messages
// contribute partial reductions, they are combined into one.
std::pair<SmallVector<Value>, SmallVector<Value>> lowerTMemLdSt(
    Location loc, ConversionPatternRewriter &rewriter, const LinearLayout &reps,
    ArrayRef<Value> vals, TMemAccessAtom atom, Type llvmElemTy,
    uint32_t tmemElementBitwidth, Value tmemBase, Value pred,
    int valsPerMessage, bool unpacked,
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

  tmemBase = LLVM::NVIDIA::projectTMemElementBaseToWordBase(
      loc, rewriter, tmemBase, tmemElementBitwidth);
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
  // lifted TMEM row/col offsets instead of the canonical row-only 32/64 pair.
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
      staticOffset = static_cast<int>(packTMemRowColOffset(row, col));
    }
    uint32_t packetOffset = static_cast<uint32_t>(staticOffset);
    uint32_t rowBaseOffset = getTMemPackedOffsetRowBaseOffset(packetOffset);
    int colImmediate = static_cast<int>(getTMemPackedOffsetCol(packetOffset));
    Value packetBase = tmemBase;
    if (rowBaseOffset != 0)
      packetBase = b.add(packetBase, b.i32_val(rowBaseOffset));

    if (isStore) {
      auto chunk = to_vector(vals.slice(i, valsPerMessage));
      createTensorMemoryStore(loc, packetBase, /*colOffset=*/colImmediate, chunk,
                              /*secondHalfOffset=*/secondHalfOffset, pred,
                              /*unpacked=*/unpacked, atom, rewriter);
    } else {
      auto [outVals, redval] = createTensorMemoryLoad(
          loc, ctx, packetBase, /*colOffset=*/colImmediate,
          /*secondHalfOffset=*/secondHalfOffset, /*unpacked=*/unpacked,
          /*numRegPerMessage=*/valsPerMessage, atom, redOp, useAbs, useNaN,
          llvmElemTy, rewriter);
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
                      uint32_t tmemElementBitwidth, ArrayRef<Value> vals,
                      Value tmemBase,
                      std::optional<TMEMLoadReduceModifier> redOp, bool useAbs,
                      bool useNaN, bool useSubwordPhasePath) {
  bool isStore = !vals.empty();
  if (info.broadcast) {
    auto removeBroadcast = std::move(info.broadcast.value());
    info.broadcast = std::nullopt;

    auto inVals = to_vector(vals);
    if (isStore) {
      inVals = removeBroadcast.apply(inVals);
    }
    auto outOr = lowerTMemLdStFromInfo(
        loc, rewriter, info, pred, llvmElemTy, tmemElementBitwidth, inVals,
        tmemBase, redOp, useAbs, useNaN, useSubwordPhasePath);
    if (failed(outOr))
      return failure();
    auto [outVals, redvalVals] = *outOr;
    if (!isStore) {
      outVals = removeBroadcast.applyInverseWithBroadcast(outVals);
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
    auto outOr = lowerTMemLdStFromInfo(
        loc, rewriter, info, pred, packedElemTy, tmemElementBitwidth, inVals,
        tmemBase, redOp, useAbs, useNaN, useSubwordPhasePath);
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
  if (redOp && getTMemLdStReductionRepeats(info) < 2) {
    emitError(loc)
        << "failed to lower TMEM reduction: tcgen05.ld.red requires at "
           "least an .x2 message shape, but the selected direct layout "
           "scalarizes to .x1 packets";
    return failure();
  }
  if (useSubwordPhasePath) {
    return lowerContiguousPackedSubwordPhaseAwareLdSt(
        loc, rewriter, info, pred, llvmElemTy, tmemElementBitwidth, inVals,
        tmemBase, redOp);
  }
  auto [outVals, redvalVals] =
      lowerTMemLdSt(loc, rewriter, info.reps, inVals, info.atom, llvmElemTy,
                    tmemElementBitwidth, tmemBase, pred,
                    info.numRegsPerMessage, info.unpacked,
                    info.secondHalfOffset, info.baseOffset,
                    info.warpBaseOffset0, info.warpBaseOffset1,
                    info.packetOffsets, redOp, useAbs, useNaN);
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
  if (debugQuerySelection)
    llvm::errs() << "[tmem-ldst] memTy=" << memTy << "\n";

  if (isUnsupportedOriginChangingTMemRowSubview(memTy)) {
    emitError(loc)
        << "unsupported tensor memory row-slice load/store: the current "
           "descriptor may start at a non-zero TMEM row, but direct "
           "tcgen05 load/store packets address a fixed row footprint. "
           "Represent the row selection in the tensor-memory layout, or "
           "operate on a descriptor whose current row extent matches its "
           "allocation row extent.";
    return failure();
  }

  std::optional<MemDescType> typeLocalScalesStorageTy;
  if (!isa<TensorMemoryScalesEncodingAttr>(memTy.getEncoding())) {
    if (auto storageTy = getMMAv5ScaleStorageType(memTy))
      typeLocalScalesStorageTy = *storageTy;
  }

  bool hasTypeLocalSubviewLayout = hasSelfContainedTMemSubviewLayout(memTy);
  MemDescType planningMemTy = memTy;
  if (typeLocalScalesStorageTy) {
    planningMemTy = *typeLocalScalesStorageTy;
  } else if (hasTypeLocalSubviewLayout) {
    planningMemTy = getSelfContainedTMemSubviewPlanningType(memTy);
  }

  auto queryTypes = triton::nvidia_gpu::getTypeLocalTMemLdStQueryTypes(memTy);
  if (queryTypes.empty()) {
    if (isTMemDescriptorSubviewType(memTy)) {
      emitError(loc)
          << "unsupported tensor memory descriptor view: current memdesc "
             "type does not encode a self-contained ld/st layout";
      return failure();
    }
    queryTypes.push_back(planningMemTy);
  }

  bool useSubwordPhasePath = false;
  if (memTy.getElementTypeBitWidth() < 32 && memDescValue) {
    TMemSubwordPhaseStatus phaseStatus = getTMemSubwordPhaseStatus(memDescValue);
    if (debugQuerySelection)
      llvm::errs() << "[tmem-ldst] typeLocalSubwordPhaseStatus="
                   << static_cast<int>(phaseStatus) << "\n";
    useSubwordPhasePath = phaseStatus != TMemSubwordPhaseStatus::KnownZero;
  }

  if (redOp) {
    auto encodingInfoOr = computeTMemLoadReductionEncodingInfo(
        regTy, memTy, maxnreg, diag);
    if (failed(encodingInfoOr))
      return failure();
    auto lowered = lowerTMemLdStFromInfo(
        loc, rewriter, *encodingInfoOr, pred, llvmElemTy,
        memTy.getElementTypeBitWidth(), vals, tmemBase, redOp, useAbs, useNaN,
        useSubwordPhasePath);
    if (failed(lowered))
      return failure();
    return *lowered;
  }

  auto trySupportQuery = [&](const TMemLdStSupportQueryPlan &supportPlan)
      -> FailureOr<std::pair<SmallVector<Value>, SmallVector<Value>>> {
    auto supportRowPlan = supportPlan.rowPlan;
    if (!supportRowPlan)
      supportRowPlan = getTMemLdStRowPlanForType(planningMemTy);
    if (!supportRowPlan)
      supportRowPlan = getTMemLdStRowPlan(supportPlan.query.layout);

    auto encodingInfoOr = computeTMemLdStEncodingInfo(
        regTy, planningMemTy, supportPlan.query, maxnreg,
        debugQuerySelection ? diag : std::function<InFlightDiagnostic()>{},
        supportRowPlan);
    if (succeeded(encodingInfoOr)) {
      auto &encodingInfo = *encodingInfoOr;
      if (!preserveTMemLdStSupportQueryBaseOffset(planningMemTy,
                                                  supportPlan.query))
        encodingInfo.baseOffset = 0;
      return lowerTMemLdStFromInfo(
          loc, rewriter, encodingInfo, pred, llvmElemTy,
          memTy.getElementTypeBitWidth(), vals, tmemBase, redOp, useAbs, useNaN,
          useSubwordPhasePath);
    }

    auto sparseLowered = lowerElementwisePackedSubwordLdSt(
        loc, rewriter, regTy, supportPlan.query, pred, llvmElemTy,
        memTy.getElementTypeBitWidth(), vals, tmemBase, redOp);
    if (succeeded(sparseLowered))
      return *sparseLowered;
    return failure();
  };

  if (auto supportPlan = getTypeLocalTMemLdStSupportQueryPlan(planningMemTy)) {
    if (auto lowered = trySupportQuery(*supportPlan); succeeded(lowered))
      return *lowered;
  }

  std::optional<MemDescType> firstQueryTy;
  std::optional<TMemLdStRowPlan> firstQueryRowPlan;
  for (MemDescType queryTy : queryTypes) {
    auto rowPlan = getTMemLdStRowPlanForType(queryTy);
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
    if (succeeded(encodingInfoOr)) {
      *encodingInfoOr = refineTMemLdStQueryTypeEncodingInfo(
          memTy, regTy, queryTy, maxnreg, rowPlan, *encodingInfoOr);
      return lowerTMemLdStFromInfo(
          loc, rewriter, *encodingInfoOr, pred, llvmElemTy,
          memTy.getElementTypeBitWidth(), vals, tmemBase, redOp, useAbs, useNaN,
          useSubwordPhasePath);
    }
  }

  if (firstQueryTy) {
    (void)computeTMemLdStEncodingInfo(regTy, *firstQueryTy, maxnreg, diag,
                                      firstQueryRowPlan);
  }
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

static void combineLaneSplitReduction(Location loc,
                                      ConversionPatternRewriter &rewriter,
                                      SmallVector<Value> &redvalVals,
                                      RankedTensorType regTy,
                                      TMEMLoadReduceModifier redOp,
                                      bool useNaN) {
  auto laneSplitMask = getTmemLoadReductionLaneSplitMask(regTy);
  if (!laneSplitMask || *laneSplitMask == 0)
    return;
  assert(redvalVals.size() == 1 &&
         "lane-split reduction combine expects per-thread reductions to be "
         "combined first");
  auto isMin = redOp == TMEMLoadReduceModifier::MIN;
  Value peer =
      LLVM::NVIDIA::shuffleXor(loc, rewriter, redvalVals.front(),
                               static_cast<int>(*laneSplitMask));
  redvalVals.front() =
      useNaN ? (isMin ? LLVM::MinimumOp::create(rewriter, loc,
                                                redvalVals.front(), peer)
                      : LLVM::MaximumOp::create(rewriter, loc,
                                                redvalVals.front(), peer))
                     ->getResult(0)
             : (isMin ? LLVM::MinNumOp::create(rewriter, loc,
                                               redvalVals.front(), peer)
                      : LLVM::MaxNumOp::create(rewriter, loc,
                                               redvalVals.front(), peer))
                     ->getResult(0);
}

struct TensorMemoryLoadOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
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
    if (redOp) {
      combinePartialReductions(loc, rewriter, redvalVals, *redOp, useNaN);
      combineLaneSplitReduction(loc, rewriter, redvalVals, regTy, *redOp,
                                useNaN);
    }

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

struct TensorMemoryAddressOpConversion
    : public ConvertOpToLLVMPattern<triton::nvidia_gpu::TMEMAddressOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::nvidia_gpu::TMEMAddressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    rewriter.replaceOp(op, b.ptrtoint(i32_ty, adaptor.getSrc()));
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
    Value base = nvgpu::TensorMemoryBaseAddress::create(rewriter, loc);
    Value baseInt = b.ptrtoint(i32_ty, base);
    int colOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_col_offset"))
                        .getValue()
                        .getZExtValue();
    int rowOffset = cast<IntegerAttr>(op->getAttr("tensor_memory_row_offset"))
                        .getValue()
                        .getZExtValue();
    auto memTy = cast<MemDescType>(op.getResult().getType());
    baseInt = LLVM::NVIDIA::projectTMemWordBaseToElementBase(
        loc, rewriter, baseInt, memTy.getElementTypeBitWidth());
    uint32_t elementsPerWord =
        getTMemElementsPerWord(memTy.getElementTypeBitWidth());
    uint32_t elementColOffset =
        static_cast<uint32_t>(colOffset) * elementsPerWord;
    Value allocAddress = b.add(
        baseInt, b.i32_val(packTMemRowColOffset(rowOffset, elementColOffset)));
    SmallVector<unsigned> order(op.getType().getRank());
    std::iota(order.begin(), order.end(), 0);
    std::reverse(order.begin(), order.end());

    if (op.getSrc()) {
      auto regTy = cast<RankedTensorType>(op.getSrc().getType());
      auto llvmElemTy = getTypeConverter()->convertType(regTy.getElementType());
      auto maxnreg = getContextualMaxNReg(op);
      SmallVector<Value> srcValues =
          unpackLLElements(loc, adaptor.getSrc(), rewriter);
      Value ptr = b.inttoptr(base.getType(), allocAddress);
      // Initialized allocs still need the real memdesc-value query path so
      // TMEM-linear allocations keep their backing-row support form instead of
      // collapsing to a narrower normalized query.
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

static void createTcgen05Cp(ConversionPatternRewriter &rewriter, Location loc,
                            Value tmem_address, Value src_desc, Value pred,
                            TMemCopyAtom atom,
                            TMemCopySourceFormat sourceFormat, bool twoCTAs) {
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
  std::string formatSuffix;
  if (sourceFormat != TMemCopySourceFormat::None)
    formatSuffix = ("." + stringifyTMemCopySourceFormat(sourceFormat)).str();
  std::string opcode =
      "tcgen05.cp.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + warp + "." +
      std::to_string(atom.nRow) + "x" + std::to_string(atom.bCol) + "b" +
      formatSuffix;
  auto &op = *ptxBuilder.create(opcode);
  op({dst, src}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createTcgen05Commit(ConversionPatternRewriter &rewriter,
                                Location loc, Value barrier, Value pred,
                                bool twoCTAs) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  PTXBuilder ptxBuilder;
  auto *predOperand = ptxBuilder.newOperand(pred, "b");
  barrier = b.ptrtoint(i32_ty, barrier);
  auto *barrierOperand = ptxBuilder.newOperand(barrier, "r");
  std::string opcode =
      "@$0 tcgen05.commit.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".mbarrier::arrive::one.shared::cluster.b64 [$1];";
  auto &commit = *ptxBuilder.create(opcode);
  commit({predOperand, barrierOperand}, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static LogicalResult copySharedToTmem(ConversionPatternRewriter &rewriter,
                                      Location loc,
                                      const TypeConverter *typeConverter,
                                      triton::nvidia_gpu::TMEMCopyOp op,
                                      Value src, Value baseDst, Value pred) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  MemDescType srcTy = op.getSrc().getType();
  MemDescType dstTy = op.getDst().getType();
  auto shmemLl = toLinearLayout(srcTy);
  std::string tmemError;
  auto maybeQuerySelection =
      selectTMemCopyPhysicalQuery(dstTy, shmemLl, &tmemError);
  if (failed(maybeQuerySelection)) {
    return op->emitOpError(tmemError.empty()
                               ? "unsupported tensor memory descriptor view "
                                 "for tcgen05.copy lowering"
                               : tmemError);
  }
  assert(maybeQuerySelection->query &&
         "successful tcgen05.copy query selection must carry a query");
  const TMemPhysicalQuery &supportDstQuery = *maybeQuerySelection->query;
  std::string conversionError;
  auto maybeCvt =
      getTMemCopySourceConversion(supportDstQuery, shmemLl, &conversionError);
  if (failed(maybeCvt)) {
    return op->emitOpError(conversionError.empty()
                               ? "unsupported tensor memory descriptor view "
                                 "for tcgen05.copy lowering"
                               : conversionError);
  }
  auto cvt = *maybeCvt;

  auto bitwidth = srcTy.getElementType().getIntOrFloatBitWidth();
  auto copyPlans = getTMemCopyPlans(cvt, bitwidth);
  if (copyPlans.empty()) {
    auto diag = op->emitOpError("failed to classify tcgen05.copy family from "
                                "shared memory descriptor ")
                << srcTy << " to tensor memory descriptor " << dstTy;
    if (auto atomFailure = getTMemCopyAtomFailureMessage(cvt, bitwidth))
      diag.attachNote() << *atomFailure;
    return failure();
  }
  if (bitwidth < 32 &&
      getTMemSubwordPhaseStatus(op.getDst()) !=
          TMemSubwordPhaseStatus::KnownZero) {
    return op->emitOpError()
           << "unsupported sub-32-bit tensor memory destination origin for "
              "tcgen05.copy: the current descriptor may start inside a "
              "32-bit hardware column";
  }
  Value wordBaseDst = LLVM::NVIDIA::projectTMemElementBaseToWordBase(
      loc, rewriter, baseDst, bitwidth);
  // Get shmem ptr
  Type elemTy = typeConverter->convertType(srcTy.getElementType());
  auto smemObj =
      LLVM::getSharedMemoryObjectFromStruct(loc, src, elemTy, rewriter);
  auto smemBase = smemObj.getShmemAffineBase(loc, rewriter, srcTy);

  struct PlannedCopyMessage {
    TMemCopyScheduledMessage schedule;
    std::optional<DotOpMmaSmemLoader> loader;
  };
  SmallVector<PlannedCopyMessage, 2> plannedMessages;
  auto planSelection =
      selectTMemCopyPlan(srcTy, supportDstQuery, shmemLl, cvt, copyPlans,
                         bitwidth);
  if (planSelection) {
    plannedMessages.reserve(planSelection.plan->messages.size());
    for (const auto &message : planSelection.plan->messages) {
      if (message.directSeedDescriptorImm) {
        plannedMessages.push_back(PlannedCopyMessage{message, std::nullopt});
        continue;
      }
      assert(message.descriptorLayout &&
             "non-direct tcgen05.copy message must carry a descriptor layout");
      auto loader = DotOpMmaSmemLoader::build(
          loc, rewriter, message.descriptorLayout->layout, bitwidth, smemBase,
          message.plan.descriptorShape, message.descriptorLayout->mnDim, 5);
      if (failed(loader) ||
          (loader->getDescriptor().transposed &&
           planSelection.plan->family != TMemCopyFamily::Dense4x256b)) {
        return op->emitOpError(
            "failed to realize selected tcgen05.copy descriptor plan during "
            "lowering");
      }
      PlannedCopyMessage plannedMessage{message, *loader};
      plannedMessages.push_back(std::move(plannedMessage));
    }
  }
  if (!planSelection) {
    StringRef family = stringifyTMemCopyFamily(copyPlans.front().family);
    auto diag =
        op->emitOpError("The source shared layout maps to tcgen05.copy.")
        << family
        << ", but Triton could not synthesize a compatible shared-memory "
           "descriptor plan for it.";
    attachTMemCopyPlanFailureNotes(diag, planSelection);
    diag.attachNote()
        << "Use the canonical shared layout for tcgen05.copy." << family
        << ", or reshape / permute the shared tile until it lowers to the "
           "same descriptor family.";
    diag.attachNote()
        << "This is reported during lowering because the final shared-memory "
           "descriptor layout is only known after shared memory allocation.";
    return failure();
  }

  bool twoCTAs = getModuleTwoCTAs(op);
  uint32_t destinationBaseOffset =
      getTMemPhysicalQueryOriginBaseOffset(supportDstQuery);

  for (const TMemCopyScheduledInstruction &instruction :
       planSelection.plan->instructions) {
    assert(instruction.messageIndex < plannedMessages.size() &&
           "tcgen05.copy instruction schedule references an unknown message");
    const auto &message = plannedMessages[instruction.messageIndex];
    const TMemCopyScheduledTile &tile = instruction.tile;
    Value desc;
    const auto &messagePlan = message.schedule.plan;
    if (message.schedule.directSeedDescriptorImm) {
      uint64_t sourceOffsetB128 =
          messagePlan.directSourceOffsetB128 +
          (instruction.source.col * bitwidth) / 128;
      uint64_t descImm = *message.schedule.directSeedDescriptorImm;
      descImm &= ~(((1ULL << 14) - 1) | (0x7ULL << 49));
      descImm |= sourceOffsetB128;
      descImm |= ((sourceOffsetB128 >> 3) & 0x7ULL) << 49;
      Value baseSrcb128 = b.lshr(b.ptrtoint(i32_ty, smemBase), b.i32_val(4));
      Value baseb128 = b.zext(i64_ty, b.and_(baseSrcb128, b.i32_val(0x3FFF)));
      desc = b.add(b.int_val(64, descImm), baseb128);
    } else {
      desc = message.loader->smemLoad(instruction.source.row,
                                      instruction.source.col, rewriter, loc);
    }
    uint32_t messageDestinationOffset =
        destinationBaseOffset + instruction.destination.offset;
    auto tmemAddr = b.add(wordBaseDst, b.i32_val(messageDestinationOffset));
    createTcgen05Cp(rewriter, loc, tmemAddr, desc, pred, messagePlan.atom,
                    messagePlan.sourceFormat, twoCTAs);
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
      auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
          loc, adaptor.getBarrier(), rewriter.getI64Type(), rewriter);
      createTcgen05Commit(rewriter, loc, smemObj.getBase(), pred, twoCTAs);
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
    if (srcTy.getRank() > layoutRank) {
      auto kCol = StringAttr::get(ctx, "col");
      int singleBufferCols = ll.getInDimSize(kCol);
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
      Value dynamicOffset = buildDynamicTensorMemoryIndexOffset(
          loc, rewriter, adaptor.getIndex(), srcTy);
      rewriter.replaceOp(
          op, advanceTensorMemoryBase(loc, rewriter, tmemBase, dynamicOffset));
      return success();
    }

    SmallVector<int32_t> offsets(srcTy.getRank(), 0);
    offsets.front() = index.getSExtValue();
    rewriter.replaceOp(
        op, advanceTensorMemoryBase(loc, rewriter, tmemBase,
                                    triton::nvidia_gpu::getTMemViewElementOffset(
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
    Location loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto tmem =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(srcTy.getContext());
    if (srcTy.getMemorySpace() != tmem) {
      return failure();
    }
    MemDescType dstTy = op.getType();
    Value base = adaptor.getSrc();
    if (op->hasAttr("tmem_physical_bitcast")) {
      base = reinterpretTensorMemoryBase(loc, rewriter, base,
                                         srcTy.getElementTypeBitWidth(),
                                         dstTy.getElementTypeBitWidth());
    }
    rewriter.replaceOp(op, base);
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
    auto srcTy = cast<MemDescType>(op.getSrc().getType());
    // Physical TMEM pointer arithmetic is defined in the source tile's address
    // space. Using the narrowed result type can erase high-order column bits
    // for N-half views (for example 128x256 -> 128x128), collapsing distinct
    // subslices onto the same base address.
    SmallVector<int32_t> offsets(srcTy.getRank(), 0);
    offsets.back() = op.getN();
    uint32_t offset = getTMemViewElementOffset(srcTy, offsets);

    rewriter.replaceOp(
        op, advanceTensorMemoryBase(loc, rewriter, adaptor.getSrc(), offset));
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateTensorMemoryOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<TensorMemoryCopyOpConversion, TensorMemoryLoadOpConversion,
               TensorMemoryStoreOpConversion, TensorMemoryAddressOpConversion,
               TensorMemoryAllocOpConversion>(typeConverter, benefit);
}

void mlir::triton::NVIDIA::populateTensorMemorySubviewOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescReinterpretOpConversion, MemDescIndexOpConversion,
               TMEMSubSliceOpConversion>(typeConverter, benefit);
}

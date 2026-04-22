#include "Dialect/NVGPU/IR/Dialect.h"
#include "MMAHelpers.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::NVIDIA;
namespace ttng = mlir::triton::nvidia_gpu;

using ::mlir::triton::gpu::NVMMASharedEncodingAttr;
using ::mlir::triton::gpu::SharedLinearEncodingAttr;

DotOpMmaV5TmemLoader mlir::triton::NVIDIA::DotOpMmaV5TmemLoader::build(
    Location loc, RewriterBase &rewriter, gpu::MemDescType memTy,
    Value memDescValue, Value tmemBase, bool useRawWordColumns) {
  auto ll = ttng::getMMAv5TMemAddressLayout(memTy, memDescValue);
  auto bitwidth = memTy.getElementTypeBitWidth();
  auto tb = TritonLLVMOpBuilder(loc, rewriter);
  Value address = tb.ptrtoint(i32_ty, tmemBase);
  return DotOpMmaV5TmemLoader(ll.pseudoinvert(), address, bitwidth,
                              useRawWordColumns);
}

MemDescOperand mlir::triton::NVIDIA::DotOpMmaV5TmemLoader::tmemLoad(
    int a, int b, ConversionPatternRewriter &rewriter, Location loc) const {
  auto dims = to_vector(ll.getInDimNames());
  auto rowCol = ll.apply({{dims[0], a}, {dims[1], b}});
  int row = rowCol[0].second;
  int col = rowCol[1].second;
  // MMAv5 accumulators use raw 32-bit TMEM word columns on the destination
  // side. Packed f16 accumulator tiles therefore advance by word columns even
  // though the logical element layout uses colStride=2. TMEM lhs operands
  // still use typed-column addressing, so keep the old scaling there.
  if (!useRawWordColumns)
    col = col * bitwidth / 32;
  int offset = col | (row << 16);
  return {address, offset};
}

static SmallVector<int>
getSortedTMemTileOrder(Value memDescValue, MemDescType memTy, int varyingDim,
                       int numRep, int tileSize) {
  SmallVector<std::pair<uint32_t, int>> offsets;
  offsets.reserve(numRep);
  for (int rep = 0; rep < numRep; ++rep) {
    SmallVector<int32_t> logicalOffsets(memTy.getRank(), 0);
    logicalOffsets[memTy.getRank() - 2 + varyingDim] = rep * tileSize;
    uint32_t offset = ttng::getMMAv5TMemViewOffsetForLowering(
        memDescValue, memTy, logicalOffsets);
    offsets.emplace_back(offset, rep);
  }
  llvm::sort(offsets, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  SmallVector<int> order;
  order.reserve(numRep);
  for (auto [_, rep] : offsets)
    order.push_back(rep);
  return order;
}

namespace {

//===----------------------------------------------------------------------===//
// InstDescriptor
//===----------------------------------------------------------------------===//

bool isTransposed(Value operand) {
  auto tensorTy = cast<MemDescType>(operand.getType());
  auto enc = tensorTy.getEncoding();
  if (auto shared = dyn_cast<NVMMASharedEncodingAttr>(enc))
    return shared.getTransposed();
  if (ttng::isTensorMemoryEncoding(enc) &&
      !isa<ttng::TensorMemoryScalesEncodingAttr>(enc))
    return false;
  if (auto sharedLinear = dyn_cast<SharedLinearEncodingAttr>(enc)) {
    // Hack. We should refactor the lowering to be able to use the
    // result from the memory descriptor
    auto *ctx = sharedLinear.getContext();
    auto kOffset = StringAttr::get(ctx, "offset");
    auto dim0 = StringAttr::get(ctx, "dim0");
    return sharedLinear.getLinearLayout().getBasis(kOffset, 0, dim0) != 0;
  }
  return false;
}

static Value createInstDescriptor(ConversionPatternRewriter &rewriter,
                                  ttng::TCGen5MMAOp op, int M, int N,
                                  bool transposeA, bool transposeB) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t : 1;
      uint32_t dType : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 3;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t : 1;
      uint32_t M : 5;
      uint32_t : 1;
      uint32_t shift : 2;
    };
  };
  auto getTypeEncoding = [&](Type type) {
    if (type.isF16())
      return 0;
    if (type.isBF16())
      return 1;
    if (type.isF32())
      return 2;
    if (llvm::isa<Float8E4M3FNType>(type))
      return 0;
    if (llvm::isa<Float8E5M2Type>(type))
      return 1;
    // For 8-bit integer types, signed arithmetic is 1, unsigned arithmetic is
    // 0. TODO: PTX supports separate A/B signedness and integer saturation
    // descriptor bits; expose them in the IR/frontend if needed.
    if (type.isInteger(8))
      return op.getIsUnsigned() ? 0 : 1;
    llvm_unreachable("Unsupported type.");
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 4;
  desc.N = N >> 3;
  desc.aType = getTypeEncoding(op.getA().getType().getElementType());
  desc.bType = getTypeEncoding(op.getB().getType().getElementType());
  Type dstElType = op.getD().getType().getElementType();
  assert(dstElType.isF16() || dstElType.isF32() || dstElType.isInteger(32));
  if (dstElType.isInteger(32)) {
    desc.dType = 2;
  } else {
    desc.dType = dstElType.isF16() ? 0 : 1;
  }
  return b.int_val(32, desc.descriptor);
}

static Value createScaleInstDescriptor(ConversionPatternRewriter &rewriter,
                                       ttng::TCGen5MMAScaledOp op, int M, int N,
                                       bool transposeA, bool transposeB,
                                       int scaleFactorsubIdxA,
                                       int scaleFactorsubIdxB,
                                       ttng::MMAv5ScaledMxfpKind mxfpInstKind) {
  Location loc = op.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  union TCGen5InstructionDescriptor {
    uint32_t descriptor;
    struct {
      uint32_t sparsitySelector : 2;
      uint32_t sparsity : 1;
      uint32_t : 1;
      uint32_t BScaleFactor : 2;
      uint32_t : 1;
      uint32_t aType : 3;
      uint32_t bType : 3;
      uint32_t negateA : 1;
      uint32_t negateB : 1;
      uint32_t transposeA : 1;
      uint32_t transposeB : 1;
      uint32_t N : 6;
      uint32_t scaleType : 1;
      uint32_t M : 5;
      uint32_t AScaleFactor : 2;
      uint32_t : 1;
    };
  };
  auto getTypeEncoding = [](ScaleDotElemType type, bool isMXF4) {
    switch (type) {
    case ScaleDotElemType::E4M3:
      return 0;
    case ScaleDotElemType::E5M2:
      return 1;
    case ScaleDotElemType::E2M3:
      return 3;
    case ScaleDotElemType::E3M2:
      return 4;
    case ScaleDotElemType::E2M1:
      return !isMXF4 ? 5 : 1;
    default:
      break;
    }
    llvm_unreachable("Unsupported type.");
  };
  static_assert(sizeof(TCGen5InstructionDescriptor) == 4,
                "instruction descriptor size should be 32 bits.");
  TCGen5InstructionDescriptor desc;
  desc.descriptor = 0;
  desc.transposeA = transposeA;
  desc.transposeB = transposeB;
  desc.M = M >> 4;
  desc.N = N >> 3;
  desc.aType = getTypeEncoding(op.getAType(),
                               ttng::isMMAv5ScaledMxfp4(mxfpInstKind));
  desc.bType = getTypeEncoding(op.getBType(),
                               ttng::isMMAv5ScaledMxfp4(mxfpInstKind));
  desc.AScaleFactor = scaleFactorsubIdxA;
  desc.BScaleFactor = scaleFactorsubIdxB;
  // Hardcoded UE8M0 scale type.
  desc.scaleType = 1;

  if (ttng::isMMAv5ScaledMxfp4(mxfpInstKind)) {
    assert(desc.aType == 1 && desc.bType == 1);
    assert(desc.AScaleFactor <= 1 && desc.BScaleFactor <= 1);
    assert(desc.transposeA == 0 &&
           "MMAv5 with kind=mxf4 does not support transpose");
    assert(desc.transposeB == 0 &&
           "MMAv5 with kind=mxf4 does not support transpose");
    if (mxfpInstKind == ttng::MMAv5ScaledMxfpKind::Mxf4) {
      desc.AScaleFactor *= 2;
      desc.BScaleFactor *= 2;
      assert(desc.AScaleFactor == 0 ||
             desc.AScaleFactor == 2 &&
                 "MMAv5 with kind=mxf4 only supports SFA_ID 0 or 2");
      assert(desc.BScaleFactor == 0 ||
             desc.BScaleFactor == 2 &&
                 "MMAv5 with kind=mxf4 only supports SFB_ID 0 or 2");
    } else if (mxfpInstKind == ttng::MMAv5ScaledMxfpKind::Mxf4NvF4) {
      desc.scaleType = 0; // UE4M3
      assert(desc.AScaleFactor == 0 &&
             "MMAv5 with kind=mxf4nvf4 currently only supports SFA_ID 0");
      assert(desc.BScaleFactor == 0 &&
             "MMAv5 with kind=mxf4nvf4 currently only supports SFB_ID 0");
    }
  }

  return b.int_val(32, desc.descriptor);
}

//===----------------------------------------------------------------------===//
// tcgen05 instructions
//===----------------------------------------------------------------------===//

void createGen5MMA(ConversionPatternRewriter &rewriter, Location loc,
                   ttng::TCGen5MMAOp op, MemDescOperand a, Value b,
                   MemDescOperand d, Value pred, Value instDescriptor,
                   Value useInitAcc, bool aInTMem, bool twoCTAs) {
  PTXBuilder ptxBuilder;
  std::string opcode =
      "tcgen05.mma.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + ".kind::";
  Type srcElementTy = op.getA().getType().getElementType();
  if (srcElementTy.isF16() || srcElementTy.isBF16()) {
    opcode += "f16";
  } else if (srcElementTy.isF32()) {
    opcode += "tf32";
  } else if (llvm::isa<Float8E4M3FNType, Float8E5M2Type>(srcElementTy)) {
    opcode += "f8f6f4";
  } else if (op.getD().getType().getElementType().isInteger(32)) {
    // PTX uses "i8" for integer operations (both signed and unsigned)
    // The signed/unsigned distinction is encoded in the instruction descriptor
    opcode += "i8";
  } else {
    assert(0 && "Unsupported type.");
  }
  auto *accOp = ptxBuilder.newAddrOperand(d.base, "r", *d.offset);
  assert(a.offset.has_value() == aInTMem);
  auto *aOp = aInTMem ? ptxBuilder.newAddrOperand(a.base, "r", *a.offset)
                      : ptxBuilder.newOperand(a.base, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, useInitAccOp}).predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

static void createScaledGen5MMA(ConversionPatternRewriter &rewriter,
                                Location loc, ttng::TCGen5MMAScaledOp op,
                                MemDescOperand a, Value b, MemDescOperand d,
                                Value scaleA, Value scaleB, Value pred,
                                Value instDescriptor, Value useInitAcc,
                                bool aInTmem,
                                ttng::MMAv5ScaledMxfpKind mxfpInstKind,
                                bool twoCTAs) {
  PTXBuilder ptxBuilder;
  std::string opcode =
      "tcgen05.mma.cta_group::" + std::to_string(twoCTAs ? 2 : 1) + ".kind::";
  if (mxfpInstKind == ttng::MMAv5ScaledMxfpKind::Mxf8f6f4) {
    opcode += "mxf8f6f4.block_scale.scale_vec::1X";
  } else if (mxfpInstKind == ttng::MMAv5ScaledMxfpKind::Mxf4) {
    opcode += "mxf4.block_scale.scale_vec::2X";
  } else if (mxfpInstKind == ttng::MMAv5ScaledMxfpKind::Mxf4NvF4) {
    opcode += "mxf4nvf4.block_scale.scale_vec::4X";
  } else {
    assert(0 && "Unsupported mxfp kind.");
  }
  auto *accOp = ptxBuilder.newAddrOperand(d.base, "r", *d.offset);
  assert(aInTmem == a.offset.has_value());
  auto *aOp = aInTmem ? ptxBuilder.newAddrOperand(a.base, "r", *a.offset)
                      : ptxBuilder.newOperand(a.base, "l");
  auto *bOp = ptxBuilder.newOperand(b, "l");
  auto *instDescOp = ptxBuilder.newOperand(instDescriptor, "r");
  auto *scaleAOp = ptxBuilder.newAddrOperand(scaleA, "r");
  auto *scaleBOp = ptxBuilder.newAddrOperand(scaleB, "r");
  auto *useInitAccOp = ptxBuilder.newOperand(useInitAcc, "b");
  auto &mmaOp = *ptxBuilder.create(opcode);
  mmaOp({accOp, aOp, bOp, instDescOp, scaleAOp, scaleBOp, useInitAccOp})
      .predicate(pred);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

void createMMACommit(ConversionPatternRewriter &rewriter, Location loc,
                     Value barrier, Value pred, bool twoCTAs,
                     ValueRange descs) {
  PTXBuilder ptxBuilder;
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value mask;
  for (uint16_t broadcastBits : ttng::getCTABroadcastMasks(twoCTAs, descs)) {
    Value descMask =
        LLVM::NVIDIA::createTMAMulticastMask(loc, rewriter, broadcastBits);
    mask = mask ? b.or_(descMask, mask) : descMask;
  }

  SmallVector<PTXBuilder::Operand *> ptxOperands;
  auto *predOperand = ptxBuilder.newOperand(pred, "b");
  ptxOperands.push_back(predOperand);
  barrier = b.ptrtoint(i32_ty, barrier);
  auto *barrierOperand = ptxBuilder.newOperand(barrier, "r");
  ptxOperands.push_back(barrierOperand);
  std::string opcode =
      "@$0 tcgen05.commit.cta_group::" + std::to_string(twoCTAs ? 2 : 1) +
      ".mbarrier::arrive::one.shared::cluster";
  if (mask)
    opcode += ".multicast::cluster";
  opcode += ".b64 [$1]";
  if (mask) {
    opcode += ", $2";
    auto *maskOperand = ptxBuilder.newOperand(mask, "h");
    ptxOperands.push_back(maskOperand);
  }
  opcode += ";";
  auto &barrierOp = *ptxBuilder.create(opcode);
  barrierOp(ptxOperands, /*onlyAttachMLIRArgs=*/true);
  ptxBuilder.launch(rewriter, loc, void_ty(rewriter.getContext()));
}

//===----------------------------------------------------------------------===//
// MMAv5 Conversion
//===----------------------------------------------------------------------===//

// Information about how to lower a dot operation, shared between regular and
// scaled dot.
struct DotConversion {
  struct InstDesc {
    unsigned mmaSizeM;
    unsigned mmaSizeN;
    struct {
      int numRepM;
      int numRepN;
      int numRepK;
    } repShape;
    bool transA;
    bool transB;
    bool aInTmem;
  };

  using GetAccAddressFn = std::function<MemDescOperand(
      ConversionPatternRewriter &, Location, int, int, const InstDesc &)>;
  using GetAccumulatorInfoFn = std::function<
      std::optional<ttng::MMAv5AccumulatorLayoutInfo>(MemDescType)>;
  using CreateMMAInstFn = std::function<void(
      ConversionPatternRewriter &, Location, MemDescOperand, MemDescOperand,
      Value, Value, Value, const InstDesc &, int, int, int)>;

  struct {
    unsigned M;
    unsigned N;
    unsigned K;
  } shape;
  int mmaSizeK;
  SmallVector<int64_t> shapeA;
  SmallVector<int64_t> shapeB;
  int numBitsPerElementA;
  int numBitsPerElementB;
  GetAccumulatorInfoFn getAccumulatorInfo;
  GetAccAddressFn getAccAddress;
  CreateMMAInstFn createMMAInst;
};

LogicalResult convertDotImpl(const LLVMTypeConverter &typeConverter,
                             ConversionPatternRewriter &rewriter, Location loc,
                             Value a, Value b, Value d, Value loadedA,
                             Value loadedB, MemDescType dTensorTy,
                             Value useDFlag, Value pred, ValueRange barriers,
                             ValueRange barrierPreds, bool twoCTAs,
                             ValueRange commitDescs,
                             bool opKindIsMXFP4, const DotConversion &op) {
  auto tb = TritonLLVMOpBuilder(loc, rewriter);

  // Only run mma on one thread. We currently use elect as ptxas is not able to
  // detect that tid.x == 0 is true only for 1 thread.
  Value warpId = mlir::triton::gpu::WarpIdOp::create(rewriter, loc);
  Value isWarp0 = tb.icmp_eq(warpId, tb.i32_val(0));
  if (twoCTAs) {
    Value cluster0 = LLVM::NVIDIA::createLeadCTAPredicate(loc, rewriter);
    pred = tb.and_(pred, cluster0);
  }
  pred = tb.and_(pred, isWarp0);

  // Synchronize the current partition before branching into the MMA block.
  if (!barriers.empty())
    BarrierOp::create(rewriter, loc, AddrSpace::Local);

  // Wrap the whole mma code sequence within a IF block.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *mmaBlock = rewriter.createBlock(curBlock->getParent(),
                                        std::next(Region::iterator(curBlock)));
  rewriter.setInsertionPointToEnd(curBlock);
  LLVM::CondBrOp::create(rewriter, loc, pred, mmaBlock, endBlock);
  // Emit the rest in mmaBlock
  rewriter.setInsertionPointToEnd(mmaBlock);

  Value elect = LLVM::NVIDIA::createElectPredicate(loc, rewriter);

  auto aTensorTy = cast<MemDescType>(a.getType());
  auto bTensorTy = cast<MemDescType>(b.getType());
  bool aInTmem = ttng::getMMAv5LhsLayoutInfo(aTensorTy).has_value();

  Value baseA = loadedA;
  if (!aInTmem) {
    baseA = getOffsetedBase(loadedA, aTensorTy, &typeConverter, rewriter, loc);
  }
  Value baseB =
      getOffsetedBase(loadedB, bTensorTy, &typeConverter, rewriter, loc);

  auto [M, N, K] = op.shape;

  auto tensorMemInfo = op.getAccumulatorInfo
                           ? op.getAccumulatorInfo(dTensorTy)
                           : ttng::getMMAv5AccumulatorLayoutInfo(dTensorTy);
  if (!tensorMemInfo) {
    return mlir::emitError(
               loc, "failed to normalize TMEM accumulator encoding for MMAv5")
           << dTensorTy;
  }
  unsigned mmaSizeM = tensorMemInfo->mmaSizeM;
  // Account for subslices
  unsigned mmaSizeN = std::min<unsigned>(tensorMemInfo->mmaSizeN, N);
  // Checked in the verifier
  assert(mmaSizeN <= 256 &&
         "The maximum size of an MMA instruction is 128x256");
  unsigned mmaSizeK = op.mmaSizeK;
  int numRepM = ceil<unsigned>(M, mmaSizeM);
  int numRepN = ceil<unsigned>(N, mmaSizeN);
  if (twoCTAs && numRepN > 1) {
    return mlir::emitError(
               loc,
               "MMAv5 two-CTA lowering requires the accumulator tile to fit "
               "a single instruction along N")
           << " after MMAv5 accumulator normalization";
  }
  int numRepK = ceil<unsigned>(K, mmaSizeK);

  SmallVector<int64_t> shapeA = op.shapeA;
  SmallVector<int64_t> shapeB = op.shapeB;
  // In A * B = C
  // For M=64 twoCTAs, B and C have the same split and A has a split half of C
  // along M.
  SmallVector<unsigned> aOperandShape = {mmaSizeM, mmaSizeK};
  // For M=128 twoCTAs, A and C have the same split and B has a split half of C
  // along N.
  SmallVector<unsigned> bOperandShape = {mmaSizeK,
                                         mmaSizeN / (twoCTAs ? 2 : 1)};

  SmallVector<int> nRepOrder(numRepN);
  std::iota(nRepOrder.begin(), nRepOrder.end(), 0);
  if (isa<ttng::TensorMemoryEncodingAttr, ttng::TensorMemoryLinearEncodingAttr>(
          dTensorTy.getEncoding())) {
    nRepOrder = getSortedTMemTileOrder(d, dTensorTy,
                                       /*varyingDim=*/1, numRepN, mmaSizeN);
  }

  std::unique_ptr<DotOpMmaMemLoader> aLoader;
  bool transA = false;
  SmallVector<int> kRepOrder(numRepK);
  std::iota(kRepOrder.begin(), kRepOrder.end(), 0);
  unsigned aTMemTileK = aOperandShape[1];
  if (aInTmem) {
    // Scaled MMA K coordinates are in logical operand elements, while TMEM A
    // descriptors for fp4 are packed into byte storage columns. Convert the
    // per-instruction K step into descriptor storage coordinates before
    // ordering or addressing the TMEM tiles.
    unsigned storageBitwidth = aTensorTy.getElementTypeBitWidth();
    unsigned logicalBitwidth = op.numBitsPerElementA;
    if (storageBitwidth > logicalBitwidth &&
        storageBitwidth % logicalBitwidth == 0) {
      unsigned logicalPerStorage = storageBitwidth / logicalBitwidth;
      assert(aTMemTileK % logicalPerStorage == 0 &&
             "MMAv5 TMEM A K tile must align to packed storage columns");
      aTMemTileK /= logicalPerStorage;
    }

    aLoader = std::make_unique<DotOpMmaV5TmemLoader>(
        DotOpMmaV5TmemLoader::build(loc, rewriter, aTensorTy, a,
                                    baseA));
    kRepOrder = getSortedTMemTileOrder(a, aTensorTy,
                                       /*varyingDim=*/1, numRepK,
                                       aTMemTileK);
  } else {
    auto isFp4a = op.numBitsPerElementA == 4;
    auto loader = DotOpMmaSmemLoader::build(loc, rewriter, aTensorTy, baseA,
                                            aOperandShape, 0, 5, isFp4a);
    if (failed(loader)) {
      return mlir::emitError(loc, "failed to find valid tcgen05.mma layout for "
                                  "operand A in shared memory ")
             << aTensorTy << " for MMAv5 instruction shape [" << mmaSizeM
             << ", " << mmaSizeK << "]";
    }
    aLoader = std::make_unique<DotOpMmaSmemLoader>(std::move(*loader));
    transA = ((DotOpMmaSmemLoader *)aLoader.get())->getDescriptor().transposed;
  }

  auto isFp4b = op.numBitsPerElementB == 4;
  auto bLoader = DotOpMmaSmemLoader::build(loc, rewriter, bTensorTy, baseB,
                                           bOperandShape, 1, 5, isFp4b);
  if (failed(bLoader)) {
    return mlir::emitError(loc, "failed to find valid tcgen05.mma layout for "
                                "operand B in shared memory ")
           << bTensorTy << " for MMAv5 instruction shape [" << mmaSizeK << ", "
           << mmaSizeN << "]";
  }
  bool transB = !bLoader->getDescriptor().transposed;

  if (aTensorTy.getElementType().isF32() && (transA || transB)) {
    return mlir::emitError(loc, "tcgen05.mma does not support transposed "
                                "float32 operands in shared memory");
  }

  DotConversion::InstDesc desc{mmaSizeM, mmaSizeN, {numRepM, numRepN, numRepK},
                               transA,   transB,   aInTmem};
  for (int m = 0; m < numRepM; m++) {
    for (int nIdx = 0; nIdx < numRepN; nIdx++) {
      int n = nRepOrder[nIdx];
      Value useInitAcc = useDFlag;
      MemDescOperand accAddress = op.getAccAddress(rewriter, loc, m, n, desc);
      for (int kIdx = 0; kIdx < numRepK; kIdx++) {
        int k = kRepOrder[kIdx];
        unsigned aTileK = desc.aInTmem ? aTMemTileK : aOperandShape[1];
        MemDescOperand a = aLoader->memLoad(m * aOperandShape[0], k * aTileK,
                                            rewriter, loc);
        Value b = bLoader->smemLoad(k * bOperandShape[0], n * bOperandShape[1],
                                    rewriter, loc);
        op.createMMAInst(rewriter, loc, accAddress, a, b, elect, useInitAcc,
                         desc, m, n, k);
        useInitAcc = tb.i1_val(1);
      }
    }
  }

  for (auto [barrier, barrierPred] : llvm::zip(barriers, barrierPreds)) {
    Value commitPred = tb.and_(barrierPred, elect);
    auto smemObj =
        LLVM::getSharedMemoryObjectFromStruct(loc, barrier, i64_ty, rewriter);
    createMMACommit(rewriter, loc, smemObj.getBase(), commitPred, twoCTAs,
                    commitDescs);
  }
  LLVM::BrOp::create(rewriter, loc, endBlock);
  return success();
}

LogicalResult convertDot(const LLVMTypeConverter &typeConverter,
                         ConversionPatternRewriter &rewriter, Location loc,
                         ttng::TCGen5MMAOp op,
                         ttng::TCGen5MMAOpAdaptor &adaptor) {
  MemDescType aTensorTy = op.getA().getType();
  MemDescType bTensorTy = op.getB().getType();
  MemDescType dTensorTy = op.getD().getType();
  bool twoCTAs = ttng::getModuleTwoCTAs(op);
  assert(twoCTAs == op.getTwoCtas());
  SmallVector<Value> commitDescs = op.getCompletionDescs();

  DotConversion dot;

  SmallVector<int64_t> dstPerCTA = triton::gpu::getShapePerCTA(dTensorTy);
  dot.shape.M = dstPerCTA[0];
  dot.shape.N = dstPerCTA[1];
  dot.shape.K = aTensorTy.getDimSize(1);
  dot.mmaSizeK = 256 / aTensorTy.getElementTypeBitWidth();

  dot.shapeA = getShapePerCTA(aTensorTy);
  dot.shapeB = getShapePerCTA(bTensorTy);
  dot.numBitsPerElementA = aTensorTy.getElementTypeBitWidth();
  dot.numBitsPerElementB = bTensorTy.getElementTypeBitWidth();

  DotOpMmaV5TmemLoader dLoader =
      DotOpMmaV5TmemLoader::build(loc, rewriter, dTensorTy, op.getD(),
                                  adaptor.getD(),
                                  /*useRawWordColumns=*/true);
  dot.getAccAddress = [&](ConversionPatternRewriter &rewriter, Location loc,
                          int m, int n, const DotConversion::InstDesc &desc) {
    return dLoader.tmemLoad(m * desc.mmaSizeM, n * desc.mmaSizeN, rewriter,
                            loc);
  };

  dot.createMMAInst = [&](ConversionPatternRewriter &rewriter, Location loc,
                          MemDescOperand accAddress, MemDescOperand a, Value b,
                          Value pred, Value useInitAcc,
                          const DotConversion::InstDesc &desc, int m, int n,
                          int k) {
    // mmaSizeM/N is the per-cta size M/N, while the 2CTA instruction expects
    // the 2CTA size mmaSize is always 64 / 128 so we double it for 2CTA
    auto mmaSizeM = twoCTAs ? desc.mmaSizeM * 2 : desc.mmaSizeM;
    auto mmaSizeN = desc.mmaSizeN;
    assert(desc.mmaSizeM == 64 || desc.mmaSizeM == 128);
    Value instDescriptor = createInstDescriptor(
        rewriter, op, mmaSizeM, mmaSizeN, desc.transA, desc.transB);
    createGen5MMA(rewriter, loc, op, a, b, accAddress, pred, instDescriptor,
                  useInitAcc, desc.aInTmem, twoCTAs);
  };

  return convertDotImpl(
      typeConverter, rewriter, loc, op.getA(), op.getB(), op.getD(),
      adaptor.getA(), adaptor.getB(), dTensorTy, adaptor.getUseD(),
      adaptor.getPred(),
      adaptor.getBarriers(), adaptor.getBarrierPreds(), twoCTAs, commitDescs,
      /*opKindIsMXFP4=*/false, dot);
}

LogicalResult convertScaledDot(const LLVMTypeConverter &typeConverter,
                               ConversionPatternRewriter &rewriter,
                               Location loc, ttng::TCGen5MMAScaledOp op,
                               ttng::TCGen5MMAScaledOpAdaptor &adaptor) {
  MemDescType aTensorTy = op.getA().getType();
  MemDescType bTensorTy = op.getB().getType();
  MemDescType dTensorTy = op.getD().getType();

  auto scaledInfo = ttng::getMMAv5ScaledInstructionInfo(
      op.getAType(), op.getBType(), op.getAScale().getType().getElementType(),
      op.getBScale().getType().getElementType(),
      isTransposed(op.getA()) || !isTransposed(op.getB()));
  auto mxfpInstKind = scaledInfo.kind;
  bool opKindIsMXFP4 = scaledInfo.isMxfp4;

  DotConversion dot;

  // Use per-CTA shape to correctly handle 2-CTA mode. getBlockM/N() returns
  // the full block shape (e.g., 256 for 2-CTA), but we need the per-CTA shape
  // (e.g., 128) to compute the correct number of MMA repetitions.
  SmallVector<int64_t> dstPerCTA = triton::gpu::getShapePerCTA(dTensorTy);
  dot.shape.M = dstPerCTA[0];
  dot.shape.N = dstPerCTA[1];
  dot.shape.K = op.getBlockK(); // K is not split across CTAs
  dot.mmaSizeK = scaledInfo.mmaSizeK;
  auto accSupport = ttng::getMMAv5ScaledAccumulatorSupport(dTensorTy);
  auto bScaleStorageTy =
      ttng::getMMAv5ScaledBScaleStorageTypeThroughViews(op.getBScale());
  MemDescType bScaleTyForPlanning =
      bScaleStorageTy.value_or(op.getBScale().getType());
  if (accSupport.narrowNScaleFragmentRequirement &&
      !ttng::isMMAv5ScaledNarrowNBScaleStorageSupported(
          bScaleTyForPlanning,
          *accSupport.narrowNScaleFragmentRequirement)) {
    return mlir::emitError(
        loc, ttng::getMMAv5ScaledNarrowNScaleFragmentError(
                 *accSupport.narrowNScaleFragmentRequirement));
  }
  if (accSupport.repeatedN32ScaleFragmentRequirement &&
      !ttng::isMMAv5ScaledRepeatedN32BScaleStorageSupported(
          bScaleTyForPlanning,
          *accSupport.repeatedN32ScaleFragmentRequirement)) {
    return mlir::emitError(
        loc, ttng::getMMAv5ScaledRepeatedN32ScaleFragmentError(
                 *accSupport.repeatedN32ScaleFragmentRequirement));
  }

  dot.shapeA = triton::gpu::getAllocationShapePerCTA(aTensorTy);
  dot.shapeB = triton::gpu::getAllocationShapePerCTA(bTensorTy);
  if (opKindIsMXFP4) {
    dot.shapeA[1] *= 2;
    dot.shapeB[0] *= 2;
  }

  dot.numBitsPerElementA = scaledInfo.numBitsPerElementA;
  dot.numBitsPerElementB = scaledInfo.numBitsPerElementB;

  TritonLLVMOpBuilder tb(loc, rewriter);
  Value baseScaleA = tb.ptrtoint(i32_ty, adaptor.getAScale());
  Value baseScaleB = tb.ptrtoint(i32_ty, adaptor.getBScale());
  auto aScaleTy = cast<MemDescType>(op.getAScale().getType());
  MemDescType bScaleTy = bScaleTyForPlanning;
  bool twoCTAs = ttng::getModuleTwoCTAs(op);
  SmallVector<Value> commitDescs = op.getCompletionDescs();
  // Use the layout-aware TMEM loader for all scaled MMAv5 accumulators, not
  // just non-legacy layouts. This keeps scaled lowering on the same physical
  // TMEM address model as plain MMAv5 and preserves whole-tile permutations
  // and descriptor-view offsets without a separate block-id schedule.
  DotOpMmaV5TmemLoader dLoader =
      DotOpMmaV5TmemLoader::build(loc, rewriter, dTensorTy, op.getD(),
                                  adaptor.getD(),
                                  /*useRawWordColumns=*/true);
  dot.getAccumulatorInfo = [dTensorTy, accInfo = accSupport.layoutInfo](
                               MemDescType memTy) {
    if (memTy == dTensorTy)
      return accInfo;
    return ttng::getMMAv5ScaledAccumulatorLayoutInfo(memTy);
  };
  dot.getAccAddress = [&](ConversionPatternRewriter &rewriter, Location loc,
                          int m, int n, const DotConversion::InstDesc &desc) {
    return dLoader.tmemLoad(m * desc.mmaSizeM, n * desc.mmaSizeN, rewriter,
                            loc);
  };

  dot.createMMAInst = [&](ConversionPatternRewriter &rewriter, Location loc,
                          MemDescOperand accAddress, MemDescOperand a, Value b,
                          Value pred, Value useInitAcc,
                          const DotConversion::InstDesc &desc, int m, int n,
                          int k) {
    auto [numRepM, numRepN, numRepK] = desc.repShape;
    auto scaleAFragment = ttng::getMMAv5ScaleFactorFragment(
        m, k, numRepM, numRepK, ttng::getTmemAllocSizes(aScaleTy).numCols,
        scaledInfo.scaleFactorColsPerSet, /*minColsPerScaleBlock=*/1);
    auto scaleBFragment = ttng::getMMAv5ScaleFactorFragment(
        n, k, numRepN, numRepK, ttng::getTmemAllocSizes(bScaleTy).numCols,
        scaledInfo.scaleFactorColsPerSet, /*minColsPerScaleBlock=*/2);
    Value scaleA =
        tb.add(baseScaleA, tb.i32_val(scaleAFragment.tmemColumnOffset));
    Value scaleB =
        tb.add(baseScaleB, tb.i32_val(scaleBFragment.tmemColumnOffset));
    Value instDescriptor = createScaleInstDescriptor(
        rewriter, op, twoCTAs ? desc.mmaSizeM * 2 : desc.mmaSizeM,
        desc.mmaSizeN, desc.transA, desc.transB, scaleAFragment.subColumnId,
        scaleBFragment.subColumnId, mxfpInstKind);
    createScaledGen5MMA(rewriter, loc, op, a, b, accAddress, scaleA, scaleB,
                        pred, instDescriptor, useInitAcc, desc.aInTmem,
                        mxfpInstKind, twoCTAs);
  };

  return convertDotImpl(typeConverter, rewriter, loc, op.getA(), op.getB(),
                        op.getD(), adaptor.getA(), adaptor.getB(), dTensorTy,
                        adaptor.getUseD(), adaptor.getPred(),
                        adaptor.getBarriers(), adaptor.getBarrierPreds(),
                        twoCTAs, commitDescs, opKindIsMXFP4, dot);
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

struct TCGen5MMAOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5MMAOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5MMAOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(convertDot(*getTypeConverter(), rewriter, op.getLoc(), op,
                          adaptor)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct TCGen5MMAScaledOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5MMAScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5MMAScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(convertScaledDot(*getTypeConverter(), rewriter, op.getLoc(), op,
                                adaptor)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct TCGen5CommitOpConversion
    : public ConvertOpToLLVMPattern<ttng::TCGen5CommitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ttng::TCGen5CommitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    TritonLLVMOpBuilder b(loc, rewriter);

    // Because this operation can signal other partitions we need to synchronize
    // the current partition first.
    BarrierOp::create(rewriter, loc, AddrSpace::Local);

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(
        loc, adaptor.getBarrier(), rewriter.getI64Type(), rewriter);
    Value pred = LLVM::NVIDIA::createElectPredicateWarp0(loc, rewriter);

    if (adaptor.getPred())
      pred = b.and_(adaptor.getPred(), pred);

    bool twoCTAs = ttng::getModuleTwoCTAs(op);
    if (twoCTAs) {
      Value cluster0 = LLVM::NVIDIA::createLeadCTAPredicate(loc, rewriter);
      pred = b.and_(pred, cluster0);
    }

    createMMACommit(rewriter, op.getLoc(), smemObj.getBase(), pred, twoCTAs,
                    op.getDescs());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace NVIDIA {

void populateTCGen5MMAOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
  patterns.add<TCGen5MMAOpConversion, TCGen5MMAScaledOpConversion,
               TCGen5CommitOpConversion>(typeConverter, benefit);
}

} // namespace NVIDIA
} // namespace triton
} // namespace mlir

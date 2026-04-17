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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.cpp.inc"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

using namespace mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

// -- WarpGroupDotOp --
LogicalResult WarpGroupDotOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<RankedTensorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);

  // verify encodings
  auto aEnc = cast<TensorOrMemDesc>(operands[0].getType()).getEncoding();
  auto bEnc = cast<MemDescType>(operands[1].getType()).getEncoding();
  auto retEnc = accTy.getEncoding();
  if (aEnc) {
    assert(bEnc);
    Dialect &dialect = aEnc.getDialect();
    auto interface = cast<DialectInferLayoutInterface>(&dialect);
    if (interface->inferDotOpEncoding(aEnc, 0, retEnc, location).failed())
      return failure();
    if (interface->inferDotOpEncoding(bEnc, 1, retEnc, location).failed())
      return failure();
  }
  return success();
}

LogicalResult WarpGroupDotOp::verify() {
  auto resTy = getD().getType();
  auto nvmmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(resTy.getEncoding());
  if (!nvmmaEnc || !nvmmaEnc.isHopper())
    return emitOpError("WGMMA result layout must be Hopper NVMMA");

  if (!isa<NVMMASharedEncodingAttr, DotOperandEncodingAttr,
           SharedLinearEncodingAttr>(getA().getType().getEncoding()))
    return emitOpError("WGMMA A operand must have NVMMA shared or dot layout");
  if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(
          getB().getType().getEncoding()))
    return emitOpError("WGMMA B operand must have NVMMA shared layout");

  auto numWarps = gpu::lookupNumWarps(getOperation());
  if (numWarps % 4)
    return emitOpError("WGMMA requires num_warps to be divisible by 4");

  auto retShapePerCTA = getShapePerCTA(resTy);
  int rank = retShapePerCTA.size();
  if (rank != 2)
    return emitOpError("WGMMA result shape must be 2D");
  if (retShapePerCTA[0] % 64 != 0)
    return emitOpError("WGMMA result M dimension must be divisible by 64");
  if (retShapePerCTA[1] % 8 != 0)
    return emitOpError("WGMMA result N dimension must be divisible by 8");

  // Verify MMA version is supported for operands.
  int mmaVersion = nvmmaEnc.getVersionMajor();
  if (!supportMMA(getA(), mmaVersion) || !supportMMA(getB(), mmaVersion))
    return emitOpError("unsupported MMA version for the given operands");

  auto aElemTy = getA().getType().getElementType();
  if (getMaxNumImpreciseAcc() < 32 &&
      (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(aElemTy)) &&
      resTy.getElementType().isF32()) {
    return emitOpError("Cannot use F32 as the accumulator element type when "
                       "the max_num_imprecise_acc is less than 32");
  }

  if (auto aTensorTy = dyn_cast<RankedTensorType>(getA().getType())) {
    auto aDotOpEnc = cast<DotOperandEncodingAttr>(aTensorTy.getEncoding());
    unsigned kWidth = 32 / aTensorTy.getElementTypeBitWidth();
    if (aDotOpEnc.getKWidth() != kWidth) {
      return emitOpError("in-register LHS operand must have a kWidth of ")
             << kWidth << " but got " << aDotOpEnc.getKWidth();
    }
  }

  return success();
}

void WarpGroupDotOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto &a = getAMutable();
  auto &b = getBMutable();
  if (isa<MemDescType>(a.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &a, SharedMemory::get());
  if (isa<MemDescType>(b.get().getType()))
    effects.emplace_back(MemoryEffects::Read::get(), &b, SharedMemory::get());
}

bool WarpGroupDotOp::needsPartialAccumulator() {
  const auto &a = getA();
  const auto &d = getD();
  auto aTensorTy = cast<triton::gpu::TensorOrMemDesc>(a.getType());
  auto aElTy = cast<triton::gpu::TensorOrMemDesc>(a.getType()).getElementType();
  bool isFP8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(aElTy);
  bool accFP32 =
      cast<triton::gpu::TensorOrMemDesc>(d.getType()).getElementType().isF32();
  uint32_t maxNumImpreciseAcc = getMaxNumImpreciseAcc();
  return isFP8 && accFP32 && maxNumImpreciseAcc <= aTensorTy.getShape()[1];
}

bool WarpGroupDotOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

// -- WarpGroupDotWaitOp --
LogicalResult WarpGroupDotWaitOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  for (Value operand : operands)
    inferredReturnTypes.push_back(operand.getType());
  return success();
}

LogicalResult WarpGroupDotWaitOp::verify() {
  if (getOperands().empty())
    return emitOpError("expected to be waiting on at least one dependency");
  return success();
}

// -- InitBarrierOp --
LogicalResult InitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  auto barrierTy = cast<MemDescType>(getAlloc().getType());
  // We cannot place cluster barriers inside warp-specialize regions, and we
  // need to place a relaxed cluster barrier between barrier.init and the first
  // barrier use.
  bool crossCTA = barrierTy.getShape()[0] != gpu::lookupNumCTAs(getOperation());
  if (crossCTA &&
      getOperation()->getParentOfType<mlir::triton::gpu::WarpSpecializeOp>())
    return emitOpError("cannot be used inside `ttg.warp_specialize`");
  return success();
}

// -- InvalBarrierOp --
LogicalResult InvalBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- BarrierExpectOp --
LogicalResult BarrierExpectOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- WaitBarrierOp --
LogicalResult WaitBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  return success();
}

// -- ArriveBarrierOp --
LogicalResult ArriveBarrierOp::verify() {
  if (failed(verifyBarrierType(*this, getAlloc().getType())))
    return failure();
  if (getCount() < 1)
    return emitOpError("count must be greater than or equal to 1");
  return success();
}

// -- FenceMBarrierInitReleaseClusterOp --
LogicalResult FenceMBarrierInitReleaseClusterOp::verify() {
  int numCTAs = triton::gpu::lookupNumCTAs(getOperation());
  if (numCTAs <= 1)
    return emitOpError("requires ttg.num-ctas > 1");
  return success();
}

static LogicalResult verifyClusterSyncOp(Operation *op) {
  int numCTAs = triton::gpu::lookupNumCTAs(op);
  if (numCTAs <= 1)
    return op->emitOpError("requires ttg.num-ctas > 1");
  if (op->getParentOfType<mlir::triton::gpu::WarpSpecializeOp>())
    return op->emitOpError("cannot be used inside `ttg.warp_specialize`");
  return success();
}

// -- ClusterArriveOp --
LogicalResult ClusterArriveOp::verify() {
  return verifyClusterSyncOp(getOperation());
}

// -- ClusterWaitOp --
LogicalResult ClusterWaitOp::verify() {
  return verifyClusterSyncOp(getOperation());
}

// -- ClusterBarrierOp --
LogicalResult ClusterBarrierOp::verify() {
  return verifyClusterSyncOp(getOperation());
}

// -- TMA operation verifiers --
static LogicalResult verifyTMAEncoding(Operation *op, TensorDescInterface desc,
                                       Attribute enc) {
  auto nvmma = dyn_cast<NVMMASharedEncodingAttr>(enc);
  if (!nvmma)
    return op->emitOpError("TMA descriptor must have NVMMA shared layout");
  auto descEnc = dyn_cast_if_present<NVMMASharedEncodingAttr>(
      desc.getBlockType().getEncoding());
  // NOTE: Cannot do descEnc != enc as the encodings may differ in rank for
  // rank-reducing loads
  if (!descEnc || descEnc.getTransposed() != nvmma.getTransposed() ||
      descEnc.getSwizzlingByteWidth() != nvmma.getSwizzlingByteWidth() ||
      descEnc.getElementBitWidth() != nvmma.getElementBitWidth() ||
      descEnc.getFp4Padded() != nvmma.getFp4Padded()) {
    return op->emitOpError("TMA descriptor layout must match shared layout, "
                           "but got descriptor layout ")
           << descEnc << " and shared memory layout " << nvmma;
  }
  if (nvmma.getTransposed())
    return op->emitOpError("TMA descriptor layout must not be transposed");
  return success();
}

static LogicalResult verifyAsyncTMALoadOp(Operation *op,
                                          TensorDescInterface desc,
                                          TypedValue<MemDescType> barrier,
                                          MemDescType resultType) {
  if (failed(verifyBarrierType(op, barrier.getType())))
    return failure();
  if (!resultType.getMutableMemory())
    return op->emitOpError("cannot store into immutable memory");
  if (failed(verifyTMAEncoding(op, desc, resultType.getEncoding())))
    return failure();
  return success();
}

static LogicalResult verifyAsyncTMAStoreOp(Operation *op,
                                           TypedValue<TensorDescType> desc,
                                           MemDescType srcType) {
  Attribute srcEnc = srcType.getEncoding();
  // `cp.async.bulk.tensor` to global memory and `cp.reduce.async.bulk.tensor`
  // do not support fp4_padded operands.
  if (isFp4Padded(srcEnc))
    return op->emitOpError("does not support fp4_padded operands");
  return verifyTMAEncoding(op, desc.getType(), srcEnc);
}

// Helper to determine if the descriptor type is for im2col mode
static bool isIm2ColDescriptor(Type descType) {
  return isa<TensorDescIm2ColType>(descType);
}

static LogicalResult verifyAsyncTMACoords(Operation *op, ValueRange coords,
                                          TensorDescInterface desc,
                                          bool isIm2Col) {
  unsigned blockRank = desc.getBlockType().getRank();

  if (isIm2Col) {
    // For IM2COL mode, coordinates are for the full tensor (3D-5D)
    // not the 2D block shape
    if (coords.size() < 3)
      return op->emitOpError(
                 "IM2COL mode requires at least 3D coordinates, but got ")
             << coords.size() << "D";
    if (coords.size() > 5)
      return op->emitOpError(
                 "IM2COL mode supports at most 5D coordinates, but got ")
             << coords.size() << "D";
  } else {
    // For TILED mode, coordinates must match the block rank
    if (coords.size() != blockRank) {
      return op->emitOpError("expected ")
             << blockRank << " coordinates, but got " << coords.size();
    }
    if (coords.size() < 1 || coords.size() > 5)
      return op->emitOpError("must have between 1 and 5 coordinates");
  }
  return success();
}

static LogicalResult verifyTMAMode(Operation *op, bool isIm2Col,
                                   ValueRange coords, ValueRange offsets) {
  if (isIm2Col) {
    if (offsets.empty())
      return op->emitOpError("IM2COL mode requires offsets to be provided");

    // For IM2COL mode, the number of offsets should be coord.size() - 2
    // 4D tensors (4 coords) need 2 offsets, 5D tensors (5 coords) need 3
    // offsets
    size_t expectedOffsets = coords.size() - 2;
    if (offsets.size() != expectedOffsets) {
      return op->emitOpError("IM2COL mode with ")
             << coords.size() << "D coordinates requires " << expectedOffsets
             << " offsets, but got " << offsets.size();
    }
  } else {
    // TILED mode should not have offsets
    if (!offsets.empty())
      return op->emitOpError("TILED mode does not support offsets");
  }
  return success();
}

// -- AsyncTMACopyGlobalToLocalOp --
LogicalResult AsyncTMACopyGlobalToLocalOp::verify() {
  auto descType = getDesc().getType();
  bool isIm2Col = isIm2ColDescriptor(descType);
  auto descInterface = cast<TensorDescInterface>(descType);

  if (failed(verifyAsyncTMACoords(*this, getCoord(), descInterface, isIm2Col)))
    return failure();
  auto resultType = getResult().getType();
  if (failed(verifyDescriptorLoadStoreOp(*this, descType, resultType)))
    return failure();
  if (failed(verifyAsyncTMALoadOp(*this, descInterface, getBarrier(),
                                  getResult().getType())))
    return failure();
  if (failed(verifyTMAMode(*this, isIm2Col, getCoord(), getOffsets())))
    return failure();
  if (getMulticast() && !hasCGABroadcast(resultType))
    return emitOpError(
        "multicast requires the shared layout to broadcast across CTAs");
  return success();
}

// -- AsyncTMACopyLocalToGlobalOp --
LogicalResult AsyncTMACopyLocalToGlobalOp::verify() {
  // Store ops only support TILED mode
  if (failed(verifyAsyncTMACoords(*this, getCoord(), getDesc().getType(),
                                  /*isIm2Col=*/false)))
    return failure();
  MemDescType srcType = getSrc().getType();
  if (failed(verifyDescriptorLoadStoreOp(*this, getDesc().getType(), srcType)))
    return failure();
  return verifyAsyncTMAStoreOp(*this, getDesc(), srcType);
}

// -- AsyncTMAReduceOp --
LogicalResult AsyncTMAReduceOp::verify() {
  // Reduce ops only support TILED mode
  if (failed(verifyAsyncTMACoords(*this, getCoord(), getDesc().getType(),
                                  /*isIm2Col=*/false)))
    return failure();
  MemDescType srcType = getSrc().getType();
  if (failed(verifyDescriptorLoadStoreOp(*this, getDesc().getType(), srcType)))
    return failure();
  return verifyAsyncTMAStoreOp(*this, getDesc(), srcType);
}

// -- AsyncTMAGatherOp --
LogicalResult AsyncTMAGatherOp::verify() {
  auto resultType = getResult().getType();
  if (failed(verifyAsyncTMALoadOp(*this, getDesc().getType(), getBarrier(),
                                  resultType)))
    return failure();
  // `tile::gather4` does not support fp4_padded operands.
  if (isFp4Padded(getResult().getType().getEncoding()))
    return emitOpError("does not support fp4_padded operands");
  return verifyGatherScatterOp(*this,
                               getDesc().getType().getSignlessBlockType(),
                               resultType, getXOffsets().getType());
}

// -- AsyncTMAScatter --
LogicalResult AsyncTMAScatterOp::verify() {
  auto srcType = getSrc().getType();
  if (failed(verifyAsyncTMAStoreOp(*this, getDesc(), srcType)))
    return failure();
  return verifyGatherScatterOp(*this,
                               getDesc().getType().getSignlessBlockType(),
                               srcType, getXOffsets().getType());
}

// -- TCGen5MMAOp --

// barrier-and-pred := `,` ssa-value `[` ssa-value `]`
// barriers-and-preds := (barrier-and-pred)*
static ParseResult
parseBarriersAndPreds(OpAsmParser &p,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &barriers,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &preds) {
  while (succeeded(p.parseOptionalComma())) {
    if (p.parseOperand(barriers.emplace_back()) || p.parseLSquare() ||
        p.parseOperand(preds.emplace_back()) || p.parseRSquare())
      return failure();
  }
  return success();
}
static void printBarriersAndPreds(OpAsmPrinter &p, Operation *op,
                                  OperandRange barriers, OperandRange preds) {
  assert(barriers.size() == preds.size());
  for (auto [barrier, pred] : llvm::zip(barriers, preds)) {
    p << ", " << barrier << '[' << pred << ']';
  }
}

// token := `[` (ssa-value (`,` ssa-value)*)? `]`
// dep-operand := token?
static ParseResult
parseToken(OpAsmParser &p, std::optional<OpAsmParser::UnresolvedOperand> &dep,
           Type &token) {
  if (failed(p.parseOptionalLSquare()))
    return success();
  token = p.getBuilder().getType<AsyncTokenType>();
  if (succeeded(p.parseOptionalRSquare()))
    return success();
  if (p.parseOperand(dep.emplace()) || p.parseRSquare())
    return failure();
  return success();
}
static void printToken(OpAsmPrinter &p, Operation *op, Value dep, Type token) {
  if (!token)
    return;
  p << '[';
  if (dep)
    p << dep;
  p << ']';
}

namespace {
enum class MMADTypeKind { tf32, f16, f8f6f4, i8 };
} // namespace

static std::string strMMADTypeKind(MMADTypeKind kind) {
  switch (kind) {
  case MMADTypeKind::tf32:
    return "tf32";
  case MMADTypeKind::f16:
    return "f16";
  case MMADTypeKind::f8f6f4:
    return "f8f6f4";
  case MMADTypeKind::i8:
    return "i8";
  }
  llvm_unreachable("unknown mma dtype kind");
}

static std::optional<std::pair<MMADTypeKind, SmallVector<Type>>>
getMMAv5DTypeKindAndAcc(Type t) {
  MLIRContext *ctx = t.getContext();
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-kind-shapes
  if (t.isF32()) {
    return {{MMADTypeKind::tf32, {Float32Type::get(ctx)}}};
  }
  if (t.isF16()) {
    return {
        {MMADTypeKind::f16, {Float16Type::get(ctx), Float32Type::get(ctx)}}};
  }
  if (t.isBF16()) {
    return {{MMADTypeKind::f16, {Float32Type::get(ctx)}}};
  }
  // TODO: float6 and explicit float4 types are not supported yet.
  // FIXME: i8 is used to represent float4 types.
  if (isa<FloatType>(t) && llvm::is_contained(std::array<unsigned, 3>{4, 6, 8},
                                              t.getIntOrFloatBitWidth())) {
    return {
        {MMADTypeKind::f8f6f4, {Float16Type::get(ctx), Float32Type::get(ctx)}}};
  }
  if (t.isInteger(8)) {
    return {{MMADTypeKind::i8, {IntegerType::get(ctx, 32)}}};
  }
  return std::nullopt;
}

static LogicalResult verifyMMADType(Operation *op, Type a, Type b, Type d) {
  auto akind = getMMAv5DTypeKindAndAcc(a);
  auto bkind = getMMAv5DTypeKindAndAcc(b);
  if (!akind)
    return op->emitOpError("unsupported LHS operand dtype: ") << a;
  if (!bkind)
    return op->emitOpError("unsupported RHS operand dtype: ") << b;
  if (akind->first != bkind->first) {
    return op->emitOpError(
               "LHS and RHS operand dtypes kinds don't match: LHS kind is ")
           << strMMADTypeKind(akind->first) << " but RHS kind is "
           << strMMADTypeKind(bkind->first);
  }
  if (!llvm::is_contained(akind->second, d) ||
      !llvm::is_contained(bkind->second, d)) {
    InFlightDiagnostic diag =
        op->emitOpError("unsupported accumulator dtype for operand types ")
        << a << " and " << b << ", accumulator dtype is " << d
        << " but must be one of [";
    llvm::interleaveComma(akind->second, diag, [&](Type t) { diag << t; });
    diag << "]";
    return diag;
  }
  return success();
}

static std::string formatCGALayout(CGAEncodingAttr cgaLayout) {
  std::string str;
  llvm::raw_string_ostream os(str);
  auto kBlock = StringAttr::get(cgaLayout.getContext(), "block");
  os << "[";
  llvm::interleaveComma(cgaLayout.getLinearLayout().getBases().lookup(kBlock),
                        os, [&](const auto &basis) {
                          os << "[";
                          llvm::interleaveComma(basis, os);
                          os << "]";
                        });
  os << "]";
  return os.str();
}

static LogicalResult verifyCompletionBarrierLayout(Operation *op,
                                                   Value barrier) {
  auto barrierTy = cast<MemDescType>(barrier.getType());
  auto expectedCGALayout =
      CGAEncodingAttr::get1DLayout(op->getContext(), gpu::lookupNumCTAs(op));
  auto actualCGALayout = getCGALayout(barrierTy.getEncoding());
  if (actualCGALayout != expectedCGALayout)
    return op->emitOpError("completion barrier cga_layout must be ")
           << formatCGALayout(expectedCGALayout) << ", got "
           << formatCGALayout(actualCGALayout);
  return success();
}

LogicalResult TCGen5MMAOp::verify() {
  if (!getIsAsync() && !getBarriers().empty()) {
    return emitOpError("The op is synchronous but a barrier is present.");
  }
  for (auto barrier : getBarriers()) {
    auto barrierTy = cast<MemDescType>(barrier.getType());
    if (failed(verifyBarrierType(*this, barrierTy)))
      return failure();
    if (failed(verifyCompletionBarrierLayout(getOperation(), barrier)))
      return failure();
  }
  Type atype = getA().getType().getElementType();
  Type btype = getB().getType().getElementType();
  Type dtype = getD().getType().getElementType();
  if (failed(verifyMMADType(*this, atype, btype, dtype)))
    return failure();

  if (getA().getType().getRank() != 2)
    return emitOpError("LHS operand must have a rank-2 tensor");
  if (getB().getType().getRank() != 2)
    return emitOpError("RHS operand must have a rank-2 tensor");
  if (getD().getType().getRank() != 2)
    return emitOpError("Return operand must have a rank-2 tensor");

  auto aEnc = getA().getType().getEncoding();
  if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr,
           TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr>(aEnc))
    return emitOpError(
        "LHS operand must have a NVMMAShared or TensorMemory encoding");
  auto bEnc = getB().getType().getEncoding();
  if (!isa<NVMMASharedEncodingAttr, SharedLinearEncodingAttr>(bEnc))
    return emitOpError("RHS operand must have a NVMMAShared encoding");
  if (atype.isF32()) {
    if (auto aShared = dyn_cast<NVMMASharedEncodingAttr>(aEnc)) {
      if (aShared.getTransposed())
        return emitOpError(
            "tcgen05.mma does not support transposed float32 operands in "
            "shared memory");
    }
    if (auto bShared = dyn_cast<NVMMASharedEncodingAttr>(bEnc)) {
      if (!bShared.getTransposed())
        return emitOpError(
            "tcgen05.mma does not support transposed float32 operands in "
            "shared memory");
    }
  }
  auto retType = getD().getType();
  auto emitUnsupportedTMemLayout =
      [&](const MMAv5TMemInstructionTileRequirement &requirement) {
    InFlightDiagnostic diag =
        emitOpError() << getMMAv5TMemInstructionTileRequirementError(
            requirement);
    diag.attachNote()
        << getMMAv5TMemInstructionTileRequirementNote(requirement);
    return diag;
  };
  auto lhsTy = getA().getType();
  auto aTmemInfo = isa<TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr>(
                       aEnc)
                       ? getMMAv5LhsLayoutInfo(getA().getType())
                       : std::optional<MMAv5LhsLayoutInfo>{};
  if (isa<TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr>(aEnc) &&
      !aTmemInfo) {
    if (auto requirement = getMMAv5TMemInstructionTileRequirement(
            lhsTy, MMAv5TMemOperandKind::LHS))
      return emitUnsupportedTMemLayout(*requirement);
  }
  auto retInfo = getMMAv5AccumulatorLayoutInfo(retType);
  if (!retInfo) {
    if (auto requirement = getMMAv5TMemInstructionTileRequirement(
            retType, MMAv5TMemOperandKind::Accumulator))
      return emitUnsupportedTMemLayout(*requirement);
  }

  // Check colStride of TMEM operands
  if (aTmemInfo) {
    if (aTmemInfo->colStride != 1)
      return emitOpError("The col stride of the LHS operand must be 1");
  }
  if (retInfo->colStride != 32 / retType.getElementTypeBitWidth())
    return emitOpError("The col stride of the return operand must be 32 / ")
           << retType.getElementTypeBitWidth() << " but got "
           << retInfo->colStride;
  // The maximum size of a MMA instruction is 128x256
  auto ctaShape =
      getShapePerCTA(getCGALayout(retType.getEncoding()).getCTASplitNum(),
                     retType.getShape());
  auto instrSizeN = std::min<unsigned>(retInfo->mmaSizeN, ctaShape[1]);
  if (instrSizeN > 256)
    return emitOpError("The block size of the return operand must be less than "
                       "or equal to 256");
  auto emitTwoCTARepeatNError = [&]() {
    return emitOpError(
        "We don't allow to emit more than one mma instruction along N. "
        "Reduce the block or increase the number of warps or CTAs along N");
  };
  if (getTwoCtas() && (ctaShape[1] + instrSizeN - 1) / instrSizeN > 1)
    return emitTwoCTARepeatNError();

  auto aCGA = getCGALayout(aEnc).getLinearLayout();
  auto bCGA = getCGALayout(bEnc).getLinearLayout();
  auto outDims = standardOutDimNames(getContext(), 2);
  if (aCGA.getOutDimSize(outDims[1]) != 1) {
    return emitOpError("LHS CTASplit along K should be 1, but got ")
           << aCGA.getOutDimSize(outDims[1]);
  }
  if (bCGA.getOutDimSize(outDims[0]) != 1) {
    return emitOpError("RHS CTASplit along K should be 1, but got ")
           << bCGA.getOutDimSize(outDims[0]);
  }

  auto kBlock = StringAttr::get(getContext(), "block");
  if (getTwoCtas()) {
    if (bCGA.getBasis(kBlock, 0) != ArrayRef{0, 1}) {
      return emitOpError("twoCTA mode expects the first basis of the "
                         "cga_layout of the RHS to be [0, 1]");
    }
    // [Note: numRepN > 1 and two_ctas]
    // Consider, just as an example, num_ctas=16, and a huge tile of shape
    // MNK = 512x64x2048
    // This is an example of layout with numRepN=2 and two_ctas=true:
    // Layout RHS:
    // #ttg.memdesc<64x2048xf16,
    //   #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true,
    //                      elementBitWidth = 16,
    //                      CGALayout = [[0, 1], [0, 2], [0, 4], [0, 0]]}>>
    //
    // As a LinearLayout:
    // offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [8, 2],
    //           [16, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [32, 0]]
    // block = [[0, 256], [0, 512], [0, 1024], [0, 0]]
    //
    // The issue is that the data from the CTA1 should be next to that of the
    // first part of the instruction. Now, the max instruction size is 128x256,
    // so the layout we should use is
    // offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1], [8, 2],
    //           [16, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 256], [32, 0]]
    // block = [[0, 128], [0, 512], [0, 1024], [0, 0]]
    // (note how we swapped the bases [0, 256] and [0, 128])
    // The issue with this layout is that it breaks the invariant that the
    // CGALayout splits the CGA tile into contiguous CTA tiles,
    // i.e. total_layout = cta_layout * cga_layout.
    // This is used all over the place, to the point that for all legacy layouts
    // we represent the CGALayout as the `cga_layout` we have to multiply on the
    // right.
    // We could allow with a bit of effort SharedLinearLayouts that did not
    // divide on the right by a CGALayout, but for now we throw a lovely error.
    auto dCGA = getCGALayout(retType.getEncoding()).getLinearLayout();
    auto nPerCTA = retType.getDimSize(1) / dCGA.getOutDimSize(outDims[1]);
    if (nPerCTA > 256)
      return emitTwoCTARepeatNError();
  }
  if (retInfo->twoCTAs != getTwoCtas()) {
    return emitOpError("The returned value's encoding must have twoCTA=")
           << getTwoCtas() << " to be used in a "
           << (getTwoCtas() ? "twoCTA" : "non-twoCTA") << " kernel";
  }
  if (aTmemInfo) {
    if (aTmemInfo->twoCTAs != getTwoCtas()) {
      return emitOpError("The LHS operand's encoding must have twoCTA=")
             << getTwoCtas() << " to be used in a "
             << (getTwoCtas() ? "twoCTA" : "non-twoCTA") << " kernel";
    }
  }

  auto aLayout = toLinearLayout(getA().getType());
  auto bLayout = toLinearLayout(getB().getType());
  auto dLayout = toLinearLayout(retType);
  auto log2nCTAs = dLayout.hasInDim(kBlock) ? dLayout.getInDimSizeLog2(kBlock)
                                            : 0;
  auto getBasisOrZero = [&](const LinearLayout &layout, int idx,
                            StringAttr outDim) -> int32_t {
    if (!layout.hasInDim(kBlock))
      return 0;
    return layout.getBasis(kBlock, idx, outDim);
  };
  for (int i = 0; i < log2nCTAs; i++) {
    std::vector<int32_t> basis = {getBasisOrZero(aLayout, i, outDims[0]),
                                  getBasisOrZero(bLayout, i, outDims[1])};
    if (getTwoCtas() && i == 0) {
      basis[1] = 0;
    }
    if (dLayout.getBasis(kBlock, i) != ArrayRef<int32_t>(basis)) {
      return emitOpError("expected block basis ")
             << basis << " at result index " << i << ", but got "
             << dLayout.getBasis(kBlock, i);
    }
  }
  return success();
}

void TCGen5MMAOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The op reads the accumulator if `useD` is not known to be false.
  APInt useD;
  if (!matchPattern(getUseD(), m_ConstantInt(&useD)) || !useD.isZero()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());

  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
  for (auto &barrierMutable : getBarriersMutable())
    effects.emplace_back(MemoryEffects::Write::get(), &barrierMutable,
                         SharedMemory::get());
}

bool TCGen5MMAOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  return aShape[aShape.size() - 1] == bShape[aShape.size() - 2];
}

Value TCGen5MMAOp::useAccumulator() { return getUseD(); }

void TCGen5MMAOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

ValueRange TCGen5MMAOp::getCompletionBarriers() { return getBarriers(); }
ValueRange TCGen5MMAOp::getCompletionBarrierPreds() {
  return getBarrierPreds();
}

void TCGen5MMAOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAOp::getAccumulator() { return getD(); }

void TCGen5MMAOp::setAccumulator(Value accum) { getDMutable().assign(accum); }

Value TCGen5MMAOp::getPredicate() { return getPred(); }

void TCGen5MMAOp::setPredicate(Value pred) { getPredMutable().assign(pred); }

void TCGen5MMAOp::build(OpBuilder &builder, OperationState &state, Type token,
                        Value a, Value b, Value d, Value accDep, Value useD,
                        Value pred, bool twoCtas, bool multicast,
                        ValueRange barriers, ValueRange barrierPreds,
                        bool isAsync, bool isUnsigned) {
  if (!barriers.empty()) {
    isAsync = true;
  }
  build(builder, state, token, a, b, d, accDep, useD, pred, barriers,
        barrierPreds, isAsync ? builder.getUnitAttr() : UnitAttr(),
        twoCtas ? builder.getUnitAttr() : UnitAttr(),
        multicast ? builder.getUnitAttr() : UnitAttr(),
        isUnsigned ? builder.getUnitAttr() : UnitAttr());
}

bool TCGen5MMAOp::isAsync() { return getIsAsync(); }

// -- TCGen5CommitOp --
LogicalResult TCGen5CommitOp::verify() {
  auto numDescs = getDescs().size();
  if (numDescs > 2)
    return emitOpError("expected 0, 1, or 2 descriptors, got ") << numDescs;
  for (auto desc : getDescs()) {
    auto descTy = cast<MemDescType>(desc.getType());
    if (!isa<triton::gpu::SharedMemorySpaceAttr>(descTy.getMemorySpace()))
      return emitOpError("descriptor operands must be shared memory "
                         "descriptors");
  }
  auto barrierTy = getBarrier().getType();
  if (failed(verifyBarrierType(*this, barrierTy)))
    return failure();
  if (failed(verifyCompletionBarrierLayout(getOperation(), getBarrier())))
    return failure();
  return success();
}

// -- TCGen5MMAScaledOp --

static Type getScaledMMAOperandType(Type elementType,
                                    ScaleDotElemType scaleType) {
  MLIRContext *ctx = elementType.getContext();
  if (isa<FloatType>(elementType))
    return elementType;
  switch (scaleType) {
  case ScaleDotElemType::E4M3:
    return Float8E4M3FNType::get(ctx);
  case ScaleDotElemType::E5M2:
    return Float8E5M2Type::get(ctx);
  case ScaleDotElemType::E2M3:
    return Float6E2M3FNType::get(ctx);
  case ScaleDotElemType::E3M2:
    return Float6E3M2FNType::get(ctx);
  case ScaleDotElemType::E2M1:
    return Float4E2M1FNType::get(ctx);
  case ScaleDotElemType::BF16:
    return BFloat16Type::get(ctx);
  case ScaleDotElemType::FP16:
    return Float16Type::get(ctx);
  }
  llvm_unreachable("Unsupported type.");
};

LogicalResult TCGen5MMAScaledOp::verify() {
  if (!getIsAsync() && !getBarriers().empty()) {
    return emitOpError("The op is synchronous but a barrier is present.");
  }
  for (auto barrier : getBarriers()) {
    auto barrierTy = cast<MemDescType>(barrier.getType());
    if (failed(verifyBarrierType(*this, barrierTy)))
      return failure();
    if (failed(verifyCompletionBarrierLayout(getOperation(), barrier)))
      return failure();
  }
  Type atype =
      getScaledMMAOperandType(getA().getType().getElementType(), getAType());
  Type btype =
      getScaledMMAOperandType(getB().getType().getElementType(), getBType());
  Type dtype = getD().getType().getElementType();
  auto aEnc = getA().getType().getEncoding();
  bool aInTmem =
      isa<TensorMemoryEncodingAttr, TensorMemoryLinearEncodingAttr>(aEnc);
  auto aTmemInfo = aInTmem ? getMMAv5LhsLayoutInfo(getA().getType())
                           : std::optional<MMAv5LhsLayoutInfo>{};
  if (failed(verifyMMADType(*this, atype, btype, dtype)))
    return failure();
  if (aInTmem && !aTmemInfo) {
    if (auto requirement = getMMAv5TMemInstructionTileRequirement(
            getA().getType(), MMAv5TMemOperandKind::LHS)) {
      InFlightDiagnostic diag =
          emitOpError() << getMMAv5TMemInstructionTileRequirementError(
              *requirement);
      diag.attachNote()
          << getMMAv5TMemInstructionTileRequirementNote(*requirement);
      return diag;
    }
  }
  if (aTmemInfo && aTmemInfo->colStride != 1)
    return emitOpError("The col stride of the LHS operand must be 1");
  if (aTmemInfo) {
    if (auto requirement = getMMAv5ScaledMixedFp4ATMemRequirement(
            getA().getType(), getAType(), getBType())) {
      return emitOpError()
             << getMMAv5ScaledMixedFp4ATMemError(*requirement);
    }
  }
  auto accSupport = getMMAv5ScaledAccumulatorSupport(getD().getType());
  auto info = accSupport.layoutInfo;
  if (!info) {
    if (accSupport.narrowNScaleFragmentRequirement) {
      return emitOpError() << getMMAv5ScaledNarrowNScaleFragmentError(
                 *accSupport.narrowNScaleFragmentRequirement);
    }
    return emitOpError()
           << "expected accumulator layout to be directly supported MMAv5 "
              "block-scaled tensor memory, but got "
           << getD().getType().getEncoding()
           << ". Block-scaled tcgen05.mma currently requires a directly "
              "supported MMAv5 tensor-memory linear layout; tile-permuted "
              "accumulator layouts are not directly representable.";
  }
  if (info->mmaSizeM != 128)
    return emitOpError("only supports instruction shape blockM=128");
  auto ctaShape =
      getShapePerCTA(getCGALayout(getD().getType().getEncoding()).getCTASplitNum(),
                     getD().getType().getShape());
  auto instrSizeN = std::min<unsigned>(info->mmaSizeN, ctaShape[1]);
  auto bScaleStorageType =
      getMMAv5ScaledBScaleStorageTypeThroughViews(getBScale());
  MemDescType bScaleTypeForRepeatedN32 =
      bScaleStorageType.value_or(getBScale().getType());
  if (accSupport.repeatedN32ScaleFragmentRequirement &&
      !isMMAv5ScaledRepeatedN32BScaleStorageSupported(
          bScaleTypeForRepeatedN32,
          *accSupport.repeatedN32ScaleFragmentRequirement) &&
      !getMMAv5ScaledRepeatedN32BScaleRematerializedShape(
          bScaleTypeForRepeatedN32,
          *accSupport.repeatedN32ScaleFragmentRequirement)) {
    return emitOpError() << getMMAv5ScaledRepeatedN32ScaleFragmentError(
               *accSupport.repeatedN32ScaleFragmentRequirement);
  }
  if (getTwoCtas() && (ctaShape[1] + instrSizeN - 1) / instrSizeN > 1) {
    return emitOpError(
        "We don't allow to emit more than one mma instruction along N. "
        "Reduce the block or increase the number of warps or CTAs along N");
  }
  return success();
}

void TCGen5MMAScaledOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The op reads the accumulator if `useD` is not known to be false.
  APInt useD;
  if (!matchPattern(getUseD(), m_ConstantInt(&useD)) || !useD.isZero()) {
    effects.emplace_back(MemoryEffects::Read::get(), &getDMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Write::get(), &getDMutable(),
                       TensorMemory::get());

  if (isa<SharedMemorySpaceAttr>(getA().getType().getMemorySpace())) {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         SharedMemory::get());

  } else {
    effects.emplace_back(MemoryEffects::Read::get(), &getAMutable(),
                         TensorMemory::get());
  }
  effects.emplace_back(MemoryEffects::Read::get(), &getBMutable(),
                       SharedMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getAScaleMutable(),
                       TensorMemory::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getBScaleMutable(),
                       TensorMemory::get());
  for (auto &barrierMutable : getBarriersMutable())
    effects.emplace_back(MemoryEffects::Write::get(), &barrierMutable,
                         SharedMemory::get());
}

bool TCGen5MMAScaledOp::verifyDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();

  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  auto aKdim = aShape[aShape.size() - 1];
  auto bKdim = bShape[aShape.size() - 2];
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    aKdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && !transB)
    bKdim *= 2;

  return aKdim == bKdim;
}

bool TCGen5MMAScaledOp::verifyOutputDims() {
  auto aShape = this->getA().getType().getShape();
  auto bShape = this->getB().getType().getShape();
  auto cShape = this->getD().getType().getShape();
  auto oMdim = cShape[cShape.size() - 2];
  auto oNdim = cShape[cShape.size() - 1];

  int aMdim = aShape[aShape.size() - 2];
  int bNdim = bShape[bShape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    aMdim *= 2;
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    bNdim *= 2;

  if (aMdim != oMdim || bNdim != oNdim)
    return false;
  return true;
}

Value TCGen5MMAScaledOp::useAccumulator() { return getUseD(); }

void TCGen5MMAScaledOp::setUseAccumulator(Value flag) {
  getUseDMutable().assign(flag);
}

ValueRange TCGen5MMAScaledOp::getCompletionBarriers() { return getBarriers(); }
ValueRange TCGen5MMAScaledOp::getCompletionBarrierPreds() {
  return getBarrierPreds();
}

void TCGen5MMAScaledOp::addCompletionBarrier(Value barrier, Value pred) {
  getBarrierPredsMutable().append(pred);
  getBarriersMutable().append(barrier);
}

TypedValue<MemDescType> TCGen5MMAScaledOp::getAccumulator() { return getD(); }

void TCGen5MMAScaledOp::setAccumulator(Value accum) {
  getDMutable().assign(accum);
}

Value TCGen5MMAScaledOp::getPredicate() { return getPred(); }

void TCGen5MMAScaledOp::setPredicate(Value pred) {
  getPredMutable().assign(pred);
}

int64_t TCGen5MMAScaledOp::getBlockM() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockM = shape[shape.size() - 2];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && transA)
    blockM *= 2;
  return blockM;
}

int64_t TCGen5MMAScaledOp::getBlockN() {
  ArrayRef<int64_t> shape = getB().getType().getShape();
  int64_t blockN = shape[shape.size() - 1];
  bool transB = false;
  if (auto bSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getB().getType().getEncoding())) {
    transB = !bSharedLayout.getTransposed();
  }
  if (this->getBType() == ScaleDotElemType::E2M1 && transB)
    blockN *= 2;
  return blockN;
}

int64_t TCGen5MMAScaledOp::getBlockK() {
  ArrayRef<int64_t> shape = getA().getType().getShape();
  int64_t blockK = shape[shape.size() - 1];
  bool transA = false;
  if (auto aSharedLayout = dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
          getA().getType().getEncoding())) {
    transA = aSharedLayout.getTransposed();
  }
  if (this->getAType() == ScaleDotElemType::E2M1 && !transA)
    blockK *= 2;
  return blockK;
}

void TCGen5MMAScaledOp::build(OpBuilder &builder, OperationState &state,
                              Type token, Value a, Value b, Value d,
                              Value accDep, Value aScale, Value bScale,
                              ScaleDotElemType aType, ScaleDotElemType bType,
                              Value useD, Value pred, ValueRange barriers,
                              ValueRange barrierPreds, bool twoCTAs,
                              bool isAsync) {
  MLIRContext *ctx = builder.getContext();
  if (!barriers.empty()) {
    isAsync = true;
  }
  build(builder, state, token, a, b, d, accDep, aScale, bScale,
        ScaleDotElemTypeAttr::get(ctx, aType),
        ScaleDotElemTypeAttr::get(ctx, bType), useD, pred, barriers,
        barrierPreds, twoCTAs ? builder.getUnitAttr() : UnitAttr(),
        isAsync ? builder.getUnitAttr() : UnitAttr());
}

bool TCGen5MMAScaledOp::isAsync() { return getIsAsync(); }

static bool isOptimizerReplayableTMemLdSt(Operation *op, Value memdescValue) {
  if (auto load = dyn_cast<TMEMLoadOp>(op)) {
    if (load.getRedOp())
      return false;
  } else if (!isa<TMEMStoreOp>(op)) {
    return false;
  }
  return isTMemLdStReplayableHalfSliceView(memdescValue) ||
         isTMemLdStReplayableFullView(memdescValue);
}

static LogicalResult
verifyTMEMOperandPreconditions(Operation *op, RankedTensorType type,
                               MemDescType memdesc, Value memdescValue,
                               StringRef regName) {
  if (type.getRank() != 2)
    return op->emitOpError(regName) << " must be a 2D tensor";
  if (!type.getEncoding())
    return success();
  if (isa<gluon::AutoEncodingAttr>(type.getEncoding())) {
    return op->emitOpError(regName)
           << " must have a concrete distributed layout, but got "
           << type.getEncoding()
           << ". Insert set_auto_layout or convert_layout before using a "
              "TMEM load/store.";
  }

  std::string unsupportedDescriptorViewError;
  if (isUnsupportedDirectTMemLdStDescriptorView(memdescValue,
                                                &unsupportedDescriptorViewError)) {
    if (isOptimizerReplayableTMemLdSt(op, memdescValue))
      return success();
    InFlightDiagnostic diag =
        op->emitOpError(regName) << " has no supported register layout";
    if (!unsupportedDescriptorViewError.empty())
      diag.attachNote() << unsupportedDescriptorViewError;
    return diag;
  }
  return success();
}

static LogicalResult verifyTMEMOperand(Operation *op, RankedTensorType type,
                                       MemDescType memdesc, Value memdescValue,
                                       StringRef regName) {
  if (failed(verifyTMEMOperandPreconditions(op, type, memdesc, memdescValue,
                                            regName)))
    return failure();
  if (isOptimizerReplayableTMemLdSt(op, memdescValue))
    return success();

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
  auto kRow = StringAttr::get(op->getContext(), "row");
  auto kCol = StringAttr::get(op->getContext(), "col");
  bool disallowQueryTypeRescueForRowZeroLiftedReinterpret = [&]() {
    if (!isa_and_nonnull<gpu::MemDescReinterpretOp>(memdescValue.getDefiningOp()) ||
        memdesc.getRank() != 2)
      return false;
    auto memLayout = toLinearLayout(memdesc);
    int bitwidth = memdesc.getElementTypeBitWidth();
    int64_t logicalRows = memdesc.getShape()[0];
    int64_t logicalCols = memdesc.getShape()[1];
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

  auto maxnreg = getContextualMaxNReg(op);
  if (!disallowQueryTypeRescueForRowZeroLiftedReinterpret) {
    auto directRowPlan = getTMemLdStRowPlanForQuery(memdescValue, memdesc);
    if (succeeded(computeTMemLdStEncodingInfo(type, memdesc, maxnreg,
                                              /*emitError=*/{},
                                              directRowPlan))) {
      return success();
    }
  }

  auto queryTypes = triton::nvidia_gpu::getTMemLdStQueryTypes(memdescValue);
  if (!disallowQueryTypeRescueForRowZeroLiftedReinterpret) {
    for (MemDescType queryTy : queryTypes) {
      auto rowPlan = getTMemLdStRowPlanForQuery(memdescValue, queryTy);
      if (succeeded(computeTMemLdStEncodingInfo(type, queryTy, maxnreg,
                                                /*emitError=*/{}, rowPlan))) {
        return success();
      }
    }
  }
  std::string rawQueryError;
  if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
          memdescValue, /*preserveNonCanonicalView=*/true, &rawQueryError);
      succeeded(rawQuery)) {
    auto rowPlan =
        getTMemLdStRowPlanForRawQuery(memdescValue, memdesc, *rawQuery);
    if (succeeded(computeTMemLdStEncodingInfo(type, memdesc, *rawQuery, maxnreg,
                                              /*emitError=*/{}, rowPlan))) {
      return success();
    }
  }
  std::string supportQueryError;
  auto trySupportQuery = [&](const TMemLdStQueryLayout &supportQuery,
                             std::optional<TMemLdStRowPlan> rowPlan) {
    rowPlan = getTMemLdStRowPlanForSupportQuery(memdescValue, memdesc,
                                                supportQuery, rowPlan);
    return succeeded(computeTMemLdStEncodingInfo(type, memdesc, supportQuery,
                                                 maxnreg, /*emitError=*/{},
                                                 rowPlan));
  };
  if (auto supportPlan =
          getTMemLdStSupportQueryPlan(memdescValue, &supportQueryError)) {
    if (trySupportQuery(supportPlan->query, supportPlan->rowPlan))
      return success();
  }

  std::string standaloneError;
  if (auto standaloneTy =
          inferStandaloneTMemViewType(memdescValue, &standaloneError);
      succeeded(standaloneTy)) {
    if (auto maybePlan = getTMemLdStPhysicalSupportPlan(
            *standaloneTy, lookupNumWarps(op), maxnreg);
        maybePlan && maybePlan->regTy == type) {
      return success();
    }
  }

  std::string requestedLayoutDetails;
  {
    llvm::raw_string_ostream os(requestedLayoutDetails);
    ScopedDiagnosticHandler handler(op->getContext(),
                                    [&](Diagnostic &diag) { diag.print(os); });
    std::string supportError;
    if (auto supportPlan =
            getTMemLdStSupportQueryPlan(memdescValue, &supportError)) {
      auto rowPlan = supportPlan->rowPlan;
      rowPlan = getTMemLdStRowPlanForSupportQuery(
          memdescValue, memdesc, supportPlan->query, rowPlan);
      (void)computeTMemLdStEncodingInfo(type, memdesc, supportPlan->query,
                                        maxnreg,
                                        [&]() {
                                          return mlir::emitError(op->getLoc());
                                        },
                                        rowPlan);
    }
    std::string rawError;
    if (requestedLayoutDetails.empty()) {
      if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
              memdescValue, /*preserveNonCanonicalView=*/true, &rawError);
          succeeded(rawQuery)) {
        auto rowPlan =
            getTMemLdStRowPlanForRawQuery(memdescValue, memdesc, *rawQuery);
        (void)computeTMemLdStEncodingInfo(type, memdesc, *rawQuery, maxnreg,
                                          [&]() {
                                            return mlir::emitError(op->getLoc());
                                          },
                                          rowPlan);
      }
    }
    if (requestedLayoutDetails.empty() &&
        !disallowQueryTypeRescueForRowZeroLiftedReinterpret) {
      for (MemDescType queryTy : queryTypes) {
        auto rowPlan = getTMemLdStRowPlanForQuery(memdescValue, queryTy);
        (void)computeTMemLdStEncodingInfo(
            type, queryTy, maxnreg,
            [&]() { return mlir::emitError(op->getLoc()); }, rowPlan);
        if (!requestedLayoutDetails.empty())
          break;
      }
    }
  }

  SmallVector<DistributedEncodingTrait> layouts =
      getTmemCompatibleLayouts(op, type, memdesc);

  InFlightDiagnostic diag =
      op->emitOpError(regName) << " has no supported register layout";
  diag.attachNote() << "Got: " << type.getEncoding();
  for (Attribute layout : layouts)
    diag.attachNote() << "potential TMEM layout: " << layout;
  if (!requestedLayoutDetails.empty()) {
    diag.attachNote()
        << "requested layout direct-lowering details:\n"
        << StringRef(requestedLayoutDetails).trim();
  }
  if (layouts.empty()) {
    diag.attachNote()
        << "No TMEM-compatible register layout exists for this operand. "
           "reshape or permute so TMEM columns stay contiguous.";
  } else {
    diag.attachNote()
        << "Use one of the potential TMEM layouts above, or insert "
           "convert_layout explicitly.";
  }
  return diag;
}

LogicalResult TMEMStoreOp::verify() {
  if (!triton::nvidia_gpu::isTensorMemoryEncoding(
          getDst().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot store into an immutable alloc");
  }
  if (failed(
          verifyTMEMOperand(*this, getSrc().getType(), getDst().getType(), getDst(),
                            "source")))
    return failure();
  return triton::gpu::verifyMemoryOpTypes(*this, getSrc().getType(),
                                          getDst().getType());
}

// -- TMEMLoadOp --
LogicalResult TMEMLoadOp::verify() {
  if (!isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("source must be a tensor memory buffer.");
  if (!isTensorMemoryEncoding(getSrc().getType().getEncoding()))
    return emitOpError("should use tensor memory encoding.");

  // Validate reduction-related attributes early so reduction loads can use the
  // reduction verifier as their single direct-layout proof instead of paying
  // for the generic load/store proof and then recomputing the same TMEM
  // encoding info for tcgen05.ld.red.
  auto redOp = getRedOp();
  bool hasRed = getRed() != nullptr;
  bool useAbs = getAbs().value_or(false);
  bool useNaN = getNaN().value_or(false);

  if (redOp) {
    if (failed(verifyTMEMOperandPreconditions(
            *this, getType(), getSrc().getType(), getSrc(), "result")))
      return failure();
  } else if (failed(verifyTMEMOperand(*this, getType(), getSrc().getType(),
                                      getSrc(), "result"))) {
    return failure();
  }

  if (isa<TensorMemoryScalesEncodingAttr>(getSrc().getType().getEncoding()) &&
      getSrc().getType().getElementTypeBitWidth() < 32) {
    auto kReg = StringAttr::get(getContext(), "register");
    if (toLinearLayout(getType()).getFreeVariableMasks().lookup(kReg) != 0) {
      return emitOpError("tmem_load on tensor-memory scales does not support "
                         "register-broadcasted layouts; use "
                         "instr_variant=\"16x32bx2\" for narrow scales tiles, "
                         "or reshape/permute so TMEM columns stay contiguous");
    }
  }

  // redOp and red result must be consistent
  if (redOp && !hasRed)
    return emitOpError("redOp is set but 'red' result is not present");
  if (hasRed && !redOp)
    return emitOpError("'red' result is present but redOp is not set");

  // abs and NaN require redOp
  if (useAbs && !redOp)
    return emitOpError("'abs' requires 'redOp' to be set");
  if (useNaN && !redOp)
    return emitOpError("'NaN' requires 'redOp' to be set");

  // abs and NaN require floating-point element type
  Type elemTy = getSrc().getType().getElementType();
  if (useAbs && !elemTy.isF32())
    return emitOpError("'abs' requires floating-point element type (f32)");
  if (useNaN && !elemTy.isF32())
    return emitOpError("'NaN' requires floating-point element type (f32)");

  // Validate reduction conditions
  if (redOp) {
    if (!elemTy.isF32())
      return emitOpError(
          "tmem_load reduction currently requires f32 element type");
    if (isa<TensorMemoryScalesEncodingAttr>(getSrc().getType().getEncoding()))
      return emitOpError(
          "tmem_load reduction is not supported for tensor memory scales.");
    if (!isReductionFriendlyTmemSourceLayout(getSrc().getType()))
      return emitOpError(
          "tmem_load reduction source layout is not directly "
          "tcgen05.ld.red-compatible; use tmem.load(...)+tt.reduce(...) "
          "explicitly for software reduction");
    auto regTy = getType();
    auto maxnreg = getContextualMaxNReg(*this);
    auto srcMemTy = cast<MemDescType>(getSrc().getType());
    std::string encodingDetails;
    auto encodingInfoOr = [&]() -> FailureOr<TMemLdStEncodingInfo> {
      llvm::raw_string_ostream os(encodingDetails);
      ScopedDiagnosticHandler handler(getContext(),
                                      [&](Diagnostic &diag) { diag.print(os); });
      auto directRowPlan = getTMemLdStRowPlanForQuery(getSrc(), srcMemTy);
      if (auto maybeInfo = computeTMemLdStEncodingInfo(
              regTy, srcMemTy, maxnreg, /*emitError=*/{}, directRowPlan);
          succeeded(maybeInfo) && isTMemLdStReductionCompatible(*maybeInfo)) {
        return maybeInfo;
      }

      auto queryTypes = triton::nvidia_gpu::getTMemLdStQueryTypes(getSrc());
      for (MemDescType queryTy : queryTypes) {
        auto rowPlan = getTMemLdStRowPlanForQuery(getSrc(), queryTy);
        if (auto maybeInfo = computeTMemLdStEncodingInfo(
                regTy, queryTy, maxnreg,
                [&]() { return mlir::emitError(getOperation()->getLoc()); },
                rowPlan);
            succeeded(maybeInfo) &&
            isTMemLdStReductionCompatible(*maybeInfo)) {
          return maybeInfo;
        }
        if (!encodingDetails.empty())
          break;
      }
      std::string supportError;
      if (auto supportPlan = getTMemLdStSupportQueryPlan(getSrc(),
                                                         &supportError)) {
        auto rowPlan = supportPlan->rowPlan;
        if (!rowPlan)
          rowPlan = getTMemLdStRowPlanForQuery(getSrc(), srcMemTy);
        if (!rowPlan)
          rowPlan = getBackingTMemLdStRowPlan(getSrc());
        if (auto maybeInfo = computeTMemLdStEncodingInfo(
                regTy, srcMemTy, supportPlan->query, maxnreg,
                [&]() { return mlir::emitError(getOperation()->getLoc()); },
                rowPlan);
            succeeded(maybeInfo)) {
          return maybeInfo;
        }
      }
      std::string rawError;
      if (auto rawQuery = inferStandaloneTMemLdStQueryLayout(
              getSrc(), /*preserveNonCanonicalView=*/true, &rawError);
          succeeded(rawQuery)) {
        auto rowPlan = getTMemLdStRowPlanForQuery(getSrc(), srcMemTy);
        if (!rowPlan)
          rowPlan = getBackingTMemLdStRowPlan(getSrc());
        if (auto maybeInfo = computeTMemLdStEncodingInfo(
                regTy, srcMemTy, *rawQuery, maxnreg,
                [&]() { return mlir::emitError(getOperation()->getLoc()); },
                rowPlan);
            succeeded(maybeInfo)) {
          return maybeInfo;
        }
      }
      for (MemDescType queryTy : queryTypes) {
        auto rowPlan = getTMemLdStRowPlanForQuery(getSrc(), queryTy);
        if (auto maybeInfo = computeTMemLdStEncodingInfo(
                regTy, queryTy, maxnreg,
                [&]() { return mlir::emitError(getOperation()->getLoc()); },
                rowPlan);
            succeeded(maybeInfo)) {
          return maybeInfo;
        }
        if (!encodingDetails.empty())
          break;
      }
      return failure();
    }();
    if (failed(encodingInfoOr)) {
      InFlightDiagnostic diag = emitOpError(
          "failed to compute TMEM encoding info for reduction");
      if (!encodingDetails.empty()) {
        diag.attachNote()
            << "requested layout direct-lowering details:\n"
            << StringRef(encodingDetails).trim();
      }
      return diag;
    }

    if (encodingInfoOr->unpacked)
      return emitOpError(
          "tmem_load reduction requires packed format (unpacked=false)");
    unsigned reductionRepeats = getTMemLdStReductionRepeats(*encodingInfoOr);
    if (reductionRepeats < 2) {
      InFlightDiagnostic diag = emitOpError(
          "tmem_load reduction selected a scalar tcgen05.ld.red message, "
          "but tcgen05.ld.red requires at least an .x2 message shape.");
      diag.attachNote()
          << "The selected direct layout would lower to .x1 packets. Use a "
             "reduction-compatible TMEM register layout such as "
             "instr_variant=\"auto\" or instr_variant=\"16x32bx2\", or use "
             "tmem.load(...)+tt.reduce(...) explicitly for software reduction.";
      return diag;
    }

    // Verify that the N dimension is directly reducible: either entirely in
    // registers, or split only across lane bit 4 where lowering combines the
    // two tcgen05.ld.red partial reductions with a warp shuffle. Broader
    // cross-thread/warp reductions still need an explicit software reduce.
    auto reductionLayoutSupport =
        getTmemLoadReductionLayoutSupport(regTy, toLinearLayout(regTy));
    if (!reductionLayoutSupport) {
      InFlightDiagnostic diag = emitOpError(
          "tmem_load reduction register layout is not directly supported by "
          "tcgen05.ld.red lowering.");
      diag.attachNote()
          << "Direct tcgen05.ld.red lowering requires M to be unsharded and "
             "all N elements to reside in the register dimension, except for a "
             "single lane-16 split of N that lowering can combine after the "
             "partial tcgen05.ld.red results.";
      if (!reductionLayoutSupport.unsupportedReason.empty())
        diag.attachNote() << reductionLayoutSupport.unsupportedReason;
      auto regLayout = toLinearLayout(regTy);
      diag.attachNote() << "Got register layout:\n" << regLayout.toString();
      return diag;
    }
  }

  return triton::gpu::verifyMemoryOpTypes(*this, getSrc().getType(), getType());
}

// -- TMEMAllocOp --
LogicalResult TMEMAllocOp::verify() {
  if (!isTensorMemoryEncoding(getType().getEncoding()))
    return emitOpError("should use tensor memory encoding");
  if (getSrc() &&
      failed(
          verifyTMEMOperand(*this, getSrc().getType(), getType(), getResult(),
                            "source")))
    return failure();
  return triton::gpu::verifyAllocOp(*this, getSrc(), getType());
}

void TMEMAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("tensor_memory_col_offset"))
    return;
  OpResult alloc = getOperation()->getOpResult(0);
  effects.emplace_back(MemoryEffects::Allocate::get(), alloc,
                       TensorMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(), alloc,
                         TensorMemory::get());
}

// -- TMEMCopyOp --
void TMEMCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       SharedMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       TensorMemory::get());
  for (auto &barrierMutable : getBarrierMutable())
    effects.emplace_back(MemoryEffects::Write::get(), &barrierMutable,
                         SharedMemory::get());
}

LogicalResult TMEMCopyOp::verify() {
  if (!isa<triton::gpu::SharedMemorySpaceAttr>(
          getSrc().getType().getMemorySpace()))
    return emitOpError("The source must be a shared memory buffer");

  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto dstTy = cast<triton::gpu::MemDescType>(getDst().getType());
  if (srcTy.getShape() != dstTy.getShape())
    return emitOpError("source shape ")
           << srcTy.getShape() << " must match destination shape "
           << dstTy.getShape();

  if (getBarrier() && !isa<triton::gpu::SharedMemorySpaceAttr>(
                          getBarrier().getType().getMemorySpace())) {
    return emitOpError("The optional barrier should be a shared memory buffer");
  }
  if (getBarrier()) {
    auto barrierTy = getBarrier().getType();
    if (failed(verifyBarrierType(*this, barrierTy)))
      return failure();
    if (failed(verifyCompletionBarrierLayout(getOperation(), getBarrier())))
      return failure();
  }
  if (!getDst().getType().getMutableMemory()) {
    return emitOpError("Cannot copy into an immutable alloc");
  }
  auto sharedEnc =
      dyn_cast<triton::gpu::SharedEncodingTrait>(srcTy.getEncoding());
  if (sharedEnc.getAlignment() < 16) {
    return emitOpError("Source must have at least 16-byte alignment to be "
                       "representable in a matrix descriptor.");
  }
  auto shmemLl = toLinearLayout(srcTy);
  std::string tmemError;
  auto maybeQuerySelection =
      selectTMemCopyPhysicalQuery(getDst(), shmemLl, &tmemError);
  if (failed(maybeQuerySelection)) {
    return emitOpError(tmemError.empty()
                           ? "unsupported tensor memory descriptor view for "
                             "tcgen05.copy"
                           : tmemError);
  }
  auto &querySelection = *maybeQuerySelection;
  assert(querySelection.query &&
         "successful tcgen05.copy query selection must carry a query");
  const TMemPhysicalQuery &supportDstQuery = *querySelection.query;
  auto tmemLl = supportDstQuery.layout;
  if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
    if (!querySelection.standalone) {
      llvm::errs() << "[tmem-copy] standalone destination query failed: "
                   << querySelection.standaloneError << "\n";
    }
    if (!querySelection.exact) {
      llvm::errs() << "[tmem-copy] exact destination query failed: "
                   << querySelection.exactError << "\n";
    } else if (querySelection.standalone) {
      if (auto difference = getFirstTMemPhysicalQueryDifference(
              *querySelection.standalone, *querySelection.exact)) {
        auto printOrigin = [](StringRef label, ArrayRef<int32_t> origin) {
          llvm::errs() << label;
          for (int32_t value : origin)
            llvm::errs() << " " << value;
          llvm::errs() << "\n";
        };

        llvm::errs() << "[tmem-copy] destination standalone/exact query "
                        "divergence: "
                     << stringifyTMemPhysicalQueryDifference(*difference)
                     << "\n";
        llvm::errs() << "[tmem-copy] standalone layout:\n"
                     << querySelection.standalone->layout.toString() << "\n";
        printOrigin("[tmem-copy] standalone origin:",
                    querySelection.standalone->origin);
        llvm::errs() << "[tmem-copy] exact layout:\n"
                     << querySelection.exact->layout.toString() << "\n";
        printOrigin("[tmem-copy] exact origin:",
                    querySelection.exact->origin);
      }
    }
    if (querySelection.usedExact) {
      llvm::errs() << "[tmem-copy] using exact destination query\n";
    } else {
      llvm::errs() << "[tmem-copy] using standalone destination query\n";
    }
  }

  auto kBlock = StringAttr::get(srcTy.getContext(), "block");
  auto kRow = StringAttr::get(srcTy.getContext(), "row");
  std::string conversionError;
  auto maybeCvt =
      getTMemCopySourceConversion(supportDstQuery, shmemLl, &conversionError);
  if (failed(maybeCvt))
    return emitOpError(conversionError.empty()
                           ? "unsupported tensor memory descriptor view for "
                             "tcgen05.copy"
                           : conversionError);
  auto cvt = *maybeCvt;
  if (std::getenv("TRITON_DEBUG_TMEM_QUERY") != nullptr) {
    llvm::errs() << "[tmem-copy] destination-to-source conversion:\n"
                 << cvt.toString() << "\n";
  }
  if (!cvt.isTrivialOver(kBlock))
    return emitOpError("The source and destination must have the same cga "
                       "layout. Got source: ")
           << shmemLl.toString() << " and destination: " << tmemLl.toString();

  // Fp4 we could lift if we needed
  auto nvmmaEnc =
      dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(srcTy.getEncoding());
  int bitwidth = srcTy.getElementType().getIntOrFloatBitWidth();
  auto copyPlans = getTMemCopyPlans(cvt, bitwidth);
  if (nvmmaEnc && (nvmmaEnc.getTransposed() || nvmmaEnc.getFp4Padded())) {
    return emitOpError("The source should not be transposed or padded");
  }
  if (supportDstQuery.isScales) {
    if (copyPlans.empty()) {
      auto diag = emitOpError(
          "The source shared layout does not match any supported "
          "tcgen05.copy family for tensor memory scales.");
      diag.attachNote()
          << "Recognized scales copy families are warpx2::01_23.64x128b, "
             "warpx2::02_13.64x128b, and warpx4.32x128b.";
      return failure();
    }
    if (nvmmaEnc && nvmmaEnc.getSwizzlingByteWidth() != 0) {
      return emitOpError("The source should not be swizzled for now");
    }
    auto planSelection = selectTMemCopyPlan(
        srcTy, supportDstQuery, shmemLl, cvt, copyPlans, bitwidth,
        TMemCopyPlanSupportKind::TensorMemoryScales);
    if (!planSelection) {
      StringRef family = stringifyTMemCopyFamily(copyPlans.front().family);
      auto diag = emitOpError("The source shared layout maps to tcgen05.copy.")
                  << family
                  << ", but Triton could not synthesize a compatible "
                     "shared-memory descriptor plan for tensor memory scales.";
      attachTMemCopyPlanFailureNotes(diag, planSelection);
      if (querySelection.standalone && querySelection.exact) {
        if (auto note = getTMemCopyExactViewScheduleNote(
                *querySelection.standalone, *querySelection.exact))
          diag.attachNote() << *note;
      }
      diag.attachNote()
          << "Use a shared layout that lowers to tcgen05.copy." << family
          << ", or reshape / permute the shared tile until it lowers to the "
             "same descriptor family.";
      diag.attachNote()
          << "This is reported as cleanly unsupported instead of falling "
             "through to late LLVM lowering.";
      return failure();
    }
  } else {
    if (getSrc().getType().getShape() != getDst().getType().getShape()) {
      return emitOpError(
          "The source and destination must have the same shape.");
    }
    if (nvmmaEnc && nvmmaEnc.getSwizzlingByteWidth() == 0) {
      return emitOpError("Source layout should be swizzled.");
    }
    if (copyPlans.empty()) {
      auto diag = emitOpError(
          "The source shared layout does not match any recognized "
          "tcgen05.copy family for non-scales tensor memory copies.");
      if (cvt.hasInDim(kRow) && cvt.getInDimSize(kRow) > 128) {
        diag.attachNote()
            << "This projection has " << cvt.getInDimSize(kRow)
            << " logical source rows. Dense tcgen05.copy planning currently "
               "atomizes one 128-row row group per message; supporting this "
               "shape needs a first-class multi-message row-group schedule "
               "that preserves the extra row selector as descriptor "
               "projection plus source and destination row offsets.";
      }
      diag.attachNote()
          << "Recognized tcgen05.copy families are 4x256b, 128x128b, "
             "128x256b, warpx2::01_23.64x128b, "
             "warpx2::02_13.64x128b, and "
             "warpx4.32x128b.";
      diag.attachNote()
          << "Use the canonical shared layout for your intended family, or "
             "reshape / permute the shared tile until it lowers to one of "
             "those families.";
      return failure();
    }
    auto planSelection =
        selectTMemCopyPlan(srcTy, supportDstQuery, shmemLl, cvt, copyPlans,
                           bitwidth, TMemCopyPlanSupportKind::TensorMemory);
    if (!planSelection) {
      StringRef family = stringifyTMemCopyFamily(copyPlans.front().family);
      auto diag =
          emitOpError("The source shared layout maps to tcgen05.copy.")
          << family
          << ", but Triton could not synthesize a compatible shared-memory "
             "descriptor plan for it.";
      attachTMemCopyPlanFailureNotes(diag, planSelection);
      diag.attachNote()
          << "Use the canonical shared layout for tcgen05.copy." << family
          << ", or reshape / permute the shared tile until it lowers to the "
             "same descriptor family.";
      diag.attachNote()
          << "This is reported as cleanly unsupported instead of falling "
             "through to late LLVM lowering.";
      return failure();
    }
  }
  // Given that we want to support flexible input SMEM shapes, kinds of shape
  // checking we can do here are limited. For simplicity, shape checking is
  // omitted.
  return success();
}

// -- TMEMSubSliceOp --
LogicalResult TMEMSubSliceOp::verify() {
  auto srcTy = cast<triton::gpu::MemDescType>(getSrc().getType());
  auto dstTy = cast<triton::gpu::MemDescType>(getResult().getType());
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  if (!isTensorMemoryEncoding(srcLayout) ||
      isa<TensorMemoryScalesEncodingAttr>(srcLayout))
    return emitOpError("The source must be a tensor memory buffer.");
  if (!isTensorMemoryEncoding(dstLayout) ||
      isa<TensorMemoryScalesEncodingAttr>(dstLayout))
    return emitOpError("The destination must be a tensor memory buffer.");
  if (srcTy.getElementType() != dstTy.getElementType())
    return emitOpError(
        "The source and result must have the same element type.");
  if (srcTy.getAllocShape() != dstTy.getAllocShape())
    return emitOpError("The source and result must have the same alloc shape.");
  if (srcTy.getRank() != 2)
    return emitOpError("The result must be a 2D tensor memory buffer.");
  if (dstTy.getRank() != 2)
    return emitOpError("The result must be a 2D tensor memory buffer.");
  if (dstTy.getDimSize(0) != srcTy.getDimSize(0))
    return emitOpError("The result must have the same number of rows as the "
                       "source.");
  auto offset = getN();
  if (offset < 0 || offset + dstTy.getDimSize(1) > srcTy.getDimSize(1)) {
    return emitError("The split offset may not exceed the source shape");
  }

  if (dstLayout == srcLayout)
    return success();

  if (isa<TensorMemoryEncodingAttr>(srcLayout) &&
      isa<TensorMemoryEncodingAttr>(dstLayout)) {
    return emitOpError("Legacy TMEM subviews must preserve the source TMEM "
                       "encoding sugar. Expected ")
           << srcLayout << " but got " << dstLayout;
  }

  SmallVector<int32_t> offsets = {0, static_cast<int32_t>(offset)};
  std::string expectedError;
  auto expectedCanonical = inferTMemSubsliceEncoding(
      srcTy.getShape(), srcLayout, dstTy.getShape(), offsets, &expectedError);
  if (failed(expectedCanonical)) {
    if (dstLayout == srcLayout)
      return success();
    return emitOpError() << expectedError;
  }

  if (auto dstLinear = dyn_cast<TensorMemoryLinearEncodingAttr>(dstLayout)) {
    std::string dstError;
    auto dstCanonical = getCanonicalTMemLinearEncoding(dstTy, &dstError);
    if (!dstCanonical)
      return emitOpError() << dstError;
    auto *inferLayoutInterface =
        cast<triton::DialectInferLayoutInterface>(&dstLinear.getDialect());
    if (succeeded(inferLayoutInterface->verifyLayoutsAreEqual(
            dstTy.getShape(), *expectedCanonical, *dstCanonical, getLoc())))
      return success();
    return emitOpError("The destination must preserve the canonical TMEM "
                       "physical encoding ")
           << *expectedCanonical << " but got " << dstTy.getEncoding();
  }

  return emitOpError("The destination must preserve the canonical TMEM "
                     "physical encoding ")
         << *expectedCanonical << " but got " << dstLayout;
}

void TMEMSubSliceOp::build(OpBuilder &builder, OperationState &state,
                           Value alloc, int offset, int size) {
  auto allocTy = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape(allocTy.getShape());
  shape.back() = size;
  SmallVector<int32_t> offsets(shape.size(), 0);
  offsets.back() = offset;
  auto maybeEncoding = inferTMemSubsliceEncoding(
      allocTy.getShape(), allocTy.getEncoding(), shape, offsets);
  Attribute encoding =
      succeeded(maybeEncoding) ? Attribute(*maybeEncoding) : allocTy.getEncoding();
  auto subsliceType = triton::gpu::MemDescType::get(
      shape, allocTy.getElementType(), encoding, allocTy.getMemorySpace(),
      allocTy.getMutableMemory(), allocTy.getAllocShape());
  build(builder, state, subsliceType, alloc, offset);
}

// -- TensormapCreateOp --
LogicalResult TensormapCreateOp::verify() {
  auto rank = getBoxDim().size();
  if (getGlobalDim().size() != rank) {
    return emitError("Rank mismatch for global dim. Got ")
           << getGlobalDim().size() << " but expected " << rank;
  }
  if (getGlobalStride().size() + 1 != rank) {
    return emitError("Rank mismatch for global stride. Got ")
           << getGlobalStride().size() << " but expected " << rank - 1;
  }
  if (getElementStride().size() != rank) {
    return emitError("Rank mismatch for element stride. Got ")
           << getElementStride().size() << " but expected " << rank;
  }
  return success();
}

// -- CLCTryCancelOp --
static LogicalResult verifyCLCResultMemdesc(Location loc, MemDescType desc) {
  auto int_ty = dyn_cast<IntegerType>(desc.getElementType());
  if (!int_ty || int_ty.getWidth() != 64) {
    return emitError(loc)
           << "Expected CLC result buffer to have type int64, but got"
           << desc.getElementType();
  }
  auto layout = desc.getEncoding();
  auto rank = desc.getRank();
  if (rank != 1 || desc.getDimSize(0) != 2) {
    return emitError(loc) << "Expected CLC result buffer to have rank 1 and a "
                             "single dimension equal to 2, but got "
                          << desc.getShape() << ".";
  }
  auto cgaLayout = getCGALayout(layout);
  auto kBlock = StringAttr::get(cgaLayout.getContext(), "block");
  if (!llvm::all_of(cgaLayout.getLinearLayout().getBases().lookup(kBlock),
                    [](const auto &basis) {
                      return llvm::all_of(basis,
                                          [](auto base) { return base == 0; });
                    }))
    return emitError(loc) << "Expected CLC result buffer cga_layout bases to "
                             "be all zeros. Got "
                          << formatCGALayout(cgaLayout);
  return success();
}

LogicalResult CLCTryCancelOp::verify() {
  if (failed(verifyCLCResultMemdesc(getLoc(), getResult().getType())))
    return failure();
  if (failed(verifyBarrierType(*this, getMbarrier().getType())))
    return failure();
  return verifyCompletionBarrierLayout(getOperation(), getMbarrier());
}

LogicalResult CLCLoadResultOp::verify() {
  return verifyCLCResultMemdesc(getLoc(), getSrc().getType());
}

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.cpp.inc"

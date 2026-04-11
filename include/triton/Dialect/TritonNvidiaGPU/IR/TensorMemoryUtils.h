#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <functional>
#include <optional>

namespace mlir::triton::nvidia_gpu {

// Get the maximum number of registers per thread based on the context. This is
// by default 256, but it can be overridden by `ttg.maxnreg` set on the module
// or a contextual register limit set by the compiler on partitions.
int getContextualMaxNReg(Operation *op);

struct TMemLdStRowPlan {
  int32_t warpRow0;
  int32_t warpRow1;
  int32_t rowSpan;
  uint32_t baseOffset = 0;
};

struct TMemLdStQueryLayout {
  LinearLayout layout;
  bool twoCTAs;
  llvm::SmallVector<int32_t> origin;
};

struct TMemLdStSupportQueryPlan {
  TMemLdStQueryLayout query;
  std::optional<TMemLdStRowPlan> rowPlan;
};

struct TMemLdStEncodingInfo {
  TMemAccessAtom atom;
  LinearLayout reps;
  ColumnAction perm;
  int numRegsPerMessage;
  std::optional<uint32_t> secondHalfOffset;
  uint32_t baseOffset = 0;
  uint32_t warpBaseOffset0 = 32u << 16;
  uint32_t warpBaseOffset1 = 64u << 16;
  int32_t warpRow0 = 32;
  int32_t warpRow1 = 64;
  std::optional<ColumnAction> broadcast = std::nullopt;
  bool unpacked = false;
  unsigned vec = 1;
  bool padding = false;
  llvm::SmallVector<int32_t> packetOffsets = {};
};

struct TMemLdStPhysicalSupportPlan {
  gpu::MemDescType memTy;
  RankedTensorType regTy;
  TMemAccessAtom atom;
};

struct TMemCopyAtom {
  int nRow;
  int bCol;
  // a multicast of n represents that warps with (warpId & n) != 0 are
  // broadcasted
  int multicast;
};

enum class TMemCopyFamily {
  Dense128x128b,
  Dense128x256b,
  Warpx2_01_23_64x128b,
  Warpx2_02_13_64x128b,
  Warpx4_32x128b,
};

struct TMemCopyMessagePlan {
  TMemCopyAtom atom;
  unsigned descriptorRows;
  unsigned sourceWarpGroups;
  llvm::SmallVector<unsigned> descriptorShape;
  llvm::SmallVector<unsigned> instrShape;
  int smemRow = 0;
  int smemColOffset = 0;
  int tmemDwordDelta = 0;
  bool useDirectSeedDescriptor = false;
  int directSourceOffsetB128 = 0;
};

struct TMemCopyPlan {
  TMemCopyFamily family;
  llvm::SmallVector<TMemCopyMessagePlan> messages;
};

std::optional<TMemLdStRowPlan> getTMemLdStRowPlanForType(gpu::MemDescType memTy);

std::optional<TMemLdStRowPlan> getMMAv5RootRowPlan(gpu::MemDescType memTy);

Value getTMemForwardingSource(Value memDesc);

void setExplicitTMemLdStRowPlan(TMEMAllocOp op, const TMemLdStRowPlan &plan,
                                bool overwriteExisting = false);
void setExplicitMMAv5RootRowPlanIfNeeded(TMEMAllocOp op);

void setExplicitTMemPhysicalLayout(TMEMAllocOp op, const LinearLayout &layout,
                                   bool twoCTAs,
                                   bool overwriteExisting = false);

void copyExplicitTMemLdStRowPlan(TMEMAllocOp dst, TMEMAllocOp src);
void copyExplicitTMemPhysicalLayout(TMEMAllocOp dst, TMEMAllocOp src);

std::optional<TMemLdStRowPlan> getBackingTMemLdStRowPlan(Value memDesc);

std::optional<TMemLdStRowPlan> getTMemLdStRowPlanForQuery(Value memDesc,
                                                          gpu::MemDescType queryTy);
std::optional<TMemLdStRowPlan>
getTMemLdStRowPlanForQueryLayout(Value memDesc, gpu::MemDescType queryTy,
                                 const TMemLdStQueryLayout &queryLayout);

llvm::SmallVector<gpu::MemDescType> getTMemLdStQueryTypes(Value memDesc);

uint32_t getTMemViewOffsetForLowering(Value memDesc, ArrayRef<int32_t> offsets);

uint32_t getTMemSubviewOffsetForLowering(gpu::MemDescSubsliceOp op);

FailureOr<gpu::MemDescType>
inferStandaloneTMemRegLayoutQueryType(Value memDesc,
                                      std::string *error = nullptr);

FailureOr<TMemLdStQueryLayout>
inferStandaloneTMemLdStQueryLayout(Value memDesc,
                                   bool preserveNonCanonicalView = true,
                                   std::string *error = nullptr);

std::optional<TMemLdStSupportQueryPlan>
getTMemLdStSupportQueryPlan(Value memDesc, std::string *error = nullptr);

std::optional<TMemLdStQueryLayout> getTMemLdStSupportQueryLayout(
    Value memDesc, std::string *error = nullptr);

bool isUnsupportedDirectTMemLdStDescriptorView(
    Value memDesc, std::string *error = nullptr);

FailureOr<gpu::MemDescType>
inferStandaloneTMemViewType(Value memDesc, std::string *error = nullptr);

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, gpu::MemDescType memTy,
                            int maxnreg,
                            std::function<InFlightDiagnostic()> emitError = {},
                            std::optional<TMemLdStRowPlan> rowPlanOverride =
                                std::nullopt);

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, gpu::MemDescType memTy,
                            const LinearLayout &queryLayout, int maxnreg,
                            std::function<InFlightDiagnostic()> emitError = {},
                            std::optional<TMemLdStRowPlan> rowPlanOverride =
                                std::nullopt);

FailureOr<TMemLdStEncodingInfo>
computeTMemLdStEncodingInfo(RankedTensorType regTy, gpu::MemDescType memTy,
                            const TMemLdStQueryLayout &queryLayout, int maxnreg,
                            std::function<InFlightDiagnostic()> emitError = {},
                            std::optional<TMemLdStRowPlan> rowPlanOverride =
                                std::nullopt);

std::optional<TMemLdStPhysicalSupportPlan>
getTMemLdStPhysicalSupportPlan(gpu::MemDescType memTy, unsigned numWarps,
                               int maxnreg);

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps,
                                std::optional<TMemLdStRowPlan> rowPlanOverride,
                                std::optional<LinearLayout> queryLayoutOverride);

std::optional<TensorMemoryLinearEncodingAttr>
tryMakeTMemViewEncoding(MLIRContext *ctx, LinearLayout ll, bool twoCTAs,
                        std::string *error = nullptr);

std::optional<llvm::SmallVector<int64_t>>
getTMemAllocShapeForEncoding(ArrayRef<int64_t> shape, Attribute encoding,
                             std::string *error = nullptr);

LogicalResult inferTMemReshapeOpEncoding(ArrayRef<int64_t> srcShape,
                                         Attribute srcEncoding,
                                         ArrayRef<int64_t> dstShape,
                                         Attribute &dstEncoding,
                                         std::optional<Location> loc = {});

LogicalResult inferTMemIndexOpEncoding(ArrayRef<int64_t> srcShape,
                                       ArrayRef<int64_t> dstShape,
                                       ArrayRef<int64_t> dstAllocShape,
                                       Attribute srcEncoding,
                                       Attribute &dstEncoding,
                                       std::optional<Location> loc = {});

LogicalResult inferTMemSubsliceOpEncoding(ArrayRef<int64_t> srcShape,
                                          ArrayRef<int64_t> srcAllocShape,
                                          Attribute srcEncoding,
                                          ArrayRef<int64_t> dstShape,
                                          ArrayRef<int32_t> offsets,
                                          Attribute &dstEncoding,
                                          std::optional<Location> loc = {});

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemIndexEncoding(gpu::MemDescType srcTy, gpu::MemDescType dstTy);

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemIndexEncoding(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape,
                       ArrayRef<int64_t> dstAllocShape, Attribute srcEncoding,
                       std::string *error = nullptr);

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemSubsliceEncoding(gpu::MemDescType srcTy, gpu::MemDescType dstTy,
                          ArrayRef<int64_t> offsets);

FailureOr<TensorMemoryLinearEncodingAttr>
inferTMemSubsliceEncoding(ArrayRef<int64_t> srcShape, Attribute srcEncoding,
                          ArrayRef<int64_t> dstShape,
                          ArrayRef<int32_t> offsets,
                          std::string *error = nullptr);

FailureOr<gpu::MemDescType>
inferTMemIndexOpType(gpu::MemDescType srcTy, std::string *error = nullptr);

FailureOr<gpu::MemDescType>
inferTMemSubsliceOpType(gpu::MemDescType srcTy, ArrayRef<int64_t> dstShape,
                        ArrayRef<int32_t> offsets,
                        std::string *error = nullptr);

FailureOr<gpu::MemDescType>
inferTMemReshapeOpType(gpu::MemDescType srcTy, ArrayRef<int64_t> dstShape,
                       std::string *error = nullptr);

TMemCopyFamily getTMemCopyFamily(const TMemCopyAtom &atom);

StringRef stringifyTMemCopyFamily(TMemCopyFamily family);

bool isDirectTMemCopyLayoutSupported(gpu::MemDescType memTy,
                                     TMemCopyFamily family,
                                     std::string *error = nullptr);

std::optional<uint64_t>
getDirectTMemCopySeedDescriptorImm(gpu::MemDescType srcTy,
                                   TMemCopyFamily family);

std::optional<TMemCopyAtom> getTMemCopyAtom(const LinearLayout &cvt,
                                            int bitwidth);

llvm::SmallVector<TMemCopyPlan> getTMemCopyPlans(const LinearLayout &cvt,
                                                 int bitwidth);

llvm::SmallVector<LinearLayout>
getTMemCopyDescriptorLayouts(gpu::MemDescType srcTy, const LinearLayout &shmemLl,
                             const LinearLayout &cvt,
                             const TMemCopyMessagePlan &message);

bool canRepresentAsMMASmemDescriptor(const LinearLayout &ll,
                                     llvm::ArrayRef<unsigned> instrShape,
                                     int bitwidth, unsigned MNdim,
                                     int mmaVersion);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_TENSORMEMORYUTILS_H_

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

#ifndef TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>

// TritonNvidiaGPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Types.h.inc"

namespace mlir::triton::nvidia_gpu::impl {
LogicalResult verifyMMAv5Op(Operation *op);
} // namespace mlir::triton::nvidia_gpu::impl

#include "triton/Dialect/TritonNvidiaGPU/IR/OpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUAttrDefs.h.inc"

#include "triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/IR/Ops.h.inc"

namespace mlir::triton::nvidia_gpu {

constexpr static char AttrTwoCTAsName[] = "ttng.two-ctas";

struct TMemLdStRowPlan;

inline bool getModuleTwoCTAs(ModuleOp mod) {
  auto attr = mod->getAttrOfType<BoolAttr>(AttrTwoCTAsName);
  return attr ? attr.getValue() : false;
}

inline bool getModuleTwoCTAs(Operation *op) {
  return getModuleTwoCTAs(op->getParentOfType<ModuleOp>());
}

struct TensorMemory : public SideEffects::Resource::Base<TensorMemory> {
  StringRef getName() const final { return "<TensorMemory>"; }
};

struct TMemAllocation {
  TMemAllocation(int numRows, int numCols)
      : numRows(numRows), numCols(numCols) {}
  int numRows;
  int numCols;
};

// Used to describe the layout of the TMEM load/store instructions
enum class TMemAccessAtom { I32x32b, I16x64b, I16x128b, I16x256b, I16x32bx2 };

inline int getElementsPerThread(TMemAccessAtom atom) {
  switch (atom) {
  case TMemAccessAtom::I32x32b:
  case TMemAccessAtom::I16x64b:
  case TMemAccessAtom::I16x32bx2:
    return 1;
  case TMemAccessAtom::I16x128b:
    return 2;
  case TMemAccessAtom::I16x256b:
    return 4;
  }
  llvm_unreachable("Unknown TMemAccessAtom");
}

inline const char *getOpShape(TMemAccessAtom atom) {
  switch (atom) {
  case TMemAccessAtom::I32x32b:
    return "32x32b";
  case TMemAccessAtom::I16x64b:
    return "16x64b";
  case TMemAccessAtom::I16x128b:
    return "16x128b";
  case TMemAccessAtom::I16x256b:
    return "16x256b";
  case TMemAccessAtom::I16x32bx2:
    return "16x32bx2";
  }
  llvm_unreachable("Unknown TMemAccessAtom");
}

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp, int32_t warpRow0 = 32,
                           int32_t warpRow1 = 64, int32_t rowSpan = 128);

LinearLayout getTileLayout(MLIRContext *ctx, TMemAccessAtom atom, bool unpacked,
                           bool withWarp, ArrayRef<int32_t> warpBasis0,
                           ArrayRef<int32_t> warpBasis1,
                           int32_t rowSpan = 128);

TMemAllocation getTmemAllocSizes(gpu::MemDescType memDescType);

uint32_t getTMemSubSliceOffset(gpu::MemDescType memDescType, int32_t nOffset);

uint32_t getTMemViewOffset(const LinearLayout &layout,
                           ArrayRef<int32_t> offsets, uint32_t bitwidth,
                           ArrayRef<int64_t> prefixShape = {});

uint32_t getTMemViewOffset(gpu::MemDescType memDescType,
                           ArrayRef<int32_t> offsets);

bool isTensorMemoryEncoding(Attribute layout);

std::optional<bool> getTensorMemoryTwoCTAs(Attribute layout);

std::optional<bool> getTensorMemoryTwoCTAs(Type type);

std::optional<Attribute>
tryGetCanonicalTensorMemoryEncoding(ArrayRef<int64_t> shape, Attribute layout,
                                    std::string *error = nullptr);

std::optional<Attribute>
tryGetCanonicalTensorMemoryEncoding(gpu::MemDescType memDescType,
                                    std::string *error = nullptr);

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(gpu::MemDescType type,
                               std::string *error = nullptr);

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(ArrayRef<int64_t> shape, Attribute encoding,
                               std::string *error = nullptr);

std::optional<TensorMemoryLinearEncodingAttr>
getCanonicalTMemLinearEncoding(ArrayRef<int64_t> shape, unsigned blockM,
                               unsigned blockN, unsigned colStride,
                               gpu::CGAEncodingAttr cgaLayout, bool twoCTAs,
                               std::string *error = nullptr);

std::optional<LinearLayout>
getTMemViewAnalysisLinearLayout(ArrayRef<int64_t> shape, Attribute encoding,
                                std::string *error = nullptr);

Attribute getCanonicalTensorMemoryEncoding(ArrayRef<int64_t> shape,
                                           Attribute layout);

Attribute getCanonicalTensorMemoryEncoding(gpu::MemDescType memDescType);

std::optional<LinearLayout>
tryGetCanonicalTensorMemoryLinearLayout(ArrayRef<int64_t> shape,
                                        Attribute layout,
                                        std::string *error = nullptr);

std::optional<LinearLayout>
tryGetCanonicalTensorMemoryLinearLayout(gpu::MemDescType memDescType,
                                        std::string *error = nullptr);

bool tensorMemoryLinearLayoutMatchesShape(const LinearLayout &layout,
                                          ArrayRef<int64_t> shape);

LinearLayout normalizeTensorMemoryLinearLayoutForAnalysis(LinearLayout layout);

LinearLayout foldCanonicalSingleCTABlockRowsForAnalysis(LinearLayout layout,
                                                        bool twoCTAs);

LinearLayout getCanonicalTensorMemoryLinearLayout(ArrayRef<int64_t> shape,
                                                  Attribute layout);

LinearLayout getCanonicalTensorMemoryLinearLayout(gpu::MemDescType memDescType);

std::optional<LinearLayout>
getCanonicalM64SplitNLayout(MLIRContext *ctx, int64_t n, unsigned numWarps);

std::optional<LinearLayout>
getCanonicalM64SplitNLayout(gpu::MemDescType memType, unsigned numWarps);

std::optional<TensorMemoryLinearEncodingAttr>
tryMakeTensorMemoryLinearEncoding(MLIRContext *ctx, LinearLayout linearLayout,
                                  bool twoCTAs,
                                  std::string *error = nullptr);

struct MMAv5LhsLayoutInfo {
  LinearLayout familyLayout;
  unsigned mmaSizeM;
  unsigned mmaSizeN;
  unsigned colStride;
  bool twoCTAs;
};

struct MMAv5AccumulatorLayoutInfo {
  LinearLayout familyLayout;
  unsigned mmaSizeM;
  unsigned mmaSizeN;
  unsigned colStride;
  bool twoCTAs;
  bool interleavedM64;
};

struct MMAv5ScaledRepeatedN32ScaleFragmentRequirement {
  Attribute accumulatorEncoding;
  unsigned instrSizeN;
  unsigned ctaColumns;
  unsigned nInstructionCount;
};

std::optional<MMAv5LhsLayoutInfo>
getMMAv5LhsLayoutInfo(gpu::MemDescType memDescType);

std::optional<MMAv5AccumulatorLayoutInfo>
getMMAv5AccumulatorLayoutInfo(gpu::MemDescType memDescType);

std::optional<MMAv5AccumulatorLayoutInfo>
getMMAv5ScaledAccumulatorLayoutInfo(gpu::MemDescType memDescType);

std::optional<MMAv5ScaledRepeatedN32ScaleFragmentRequirement>
getMMAv5ScaledRepeatedN32ScaleFragmentRequirement(
    gpu::MemDescType memDescType);

std::optional<std::string>
getMMAv5ScaledRepeatedN32ScaleFragmentError(gpu::MemDescType memDescType);

SmallVector<gpu::DistributedEncodingTrait>
getTmemCompatibleLayouts(gpu::MemDescType memType, unsigned numWarps,
                         ArrayRef<int64_t> ctaSplit = {1, 1});

std::optional<gpu::DistributedEncodingTrait>
getTmemLoadLayoutSplitLongM(RankedTensorType tensorType,
                            gpu::MemDescType memType, int numWarps);

std::optional<gpu::DistributedEncodingTrait>
getTmemLoadReductionLayout(RankedTensorType tensorType,
                           gpu::MemDescType memType, int numWarps);

bool isReductionFriendlyTmemSourceLayout(gpu::MemDescType memType);

bool isReductionFriendlyTmemLoadLayout(RankedTensorType tensorType,
                                       const LinearLayout &regLayout);

// Returns 0 when the load layout keeps the full reduction dimension in
// registers. Returns a lane-id xor mask when the layout has one supported
// lane-local split of the reduction dimension that lowering must combine after
// tcgen05.ld.red. Returns std::nullopt for layouts that need unsupported
// cross-thread/warp reduction.
std::optional<unsigned>
getTmemLoadReductionLaneSplitMask(RankedTensorType tensorType,
                                  const LinearLayout &regLayout);

std::optional<unsigned>
getTmemLoadReductionLaneSplitMask(RankedTensorType tensorType);

SmallVector<gpu::DistributedEncodingTrait>
getTmemCompatibleLayouts(Operation *op, RankedTensorType tensorType,
                         gpu::MemDescType memType);

bool isDistributedLayoutTMemCompatible(Operation *op,
                                       RankedTensorType tensorType,
                                       gpu::MemDescType memType);

gpu::DistributedEncodingTrait
getDefaultLayoutForTmemLdSt(gpu::MemDescType memType, unsigned numWarps);

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps);

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(gpu::MemDescType memType, TMemAccessAtom atom,
                                unsigned numWarps,
                                std::optional<TMemLdStRowPlan> rowPlanOverride,
                                std::optional<LinearLayout> queryLayoutOverride =
                                    std::nullopt);

std::optional<LinearLayout>
getDistributedLayoutForTmemLdSt(const LinearLayout &memLayout,
                                TMemAccessAtom atom, unsigned numWarps,
                                int bitwidth,
                                const TMemLdStRowPlan &rowPlan,
                                bool allowSplitNFastPath = true);

std::optional<TMemLdStRowPlan> getTMemLdStRowPlan(const LinearLayout &ll);

} // namespace mlir::triton::nvidia_gpu

#endif // TRITON_DIALECT_TRITONNVIDIAGPU_IR_DIALECT_H_

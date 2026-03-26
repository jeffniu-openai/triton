#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"

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
} // namespace

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
  auto memLayout = squeezeTrivialBlock(toLinearLayout(memTy));
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
  // Warps 0-3 must map to row=32 and row=64 whether with broadcasting or not
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

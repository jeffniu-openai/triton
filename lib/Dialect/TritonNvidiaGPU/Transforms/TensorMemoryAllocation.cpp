#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Traits.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/MapVector.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

namespace ttg = triton::gpu;

#define GEN_PASS_DEF_TRITONTENSORMEMORYALLOCATIONPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Granularity of row allocations.
static constexpr int allocGranularity = 64;
struct TMemChunk {
  int startRow;
  int startCol;
  int numCols;
  int numRows;
};

// Use a simple bitmap to track memory usage. This is a slow but it allows us to
// handle 2D memory without extra algorithmic complexity. The number of
// allocations is expected to be small so the compile time is unlikely to be a
// problem.
struct MemoryBitMap {
  MemoryBitMap() : elements(512 * kNumRows, false) {}
  void free(const TMemChunk &chunk) {
    for (int i = 0; i < chunk.numCols; i++) {
      for (int j = 0; j < chunk.numRows; j++) {
        setUsed(chunk.startRow + j, chunk.startCol + i, false);
      }
    }
  }
  void alloc(const TMemChunk &chunk) {
    // Ensure the underlying data fits the allocation.
    while ((chunk.startCol + chunk.numCols) * kNumRows >= elements.size())
      elements.resize(2 * elements.size(), false);

    for (int i = 0; i < chunk.numCols; i++) {
      for (int j = 0; j < chunk.numRows; j++) {
        setUsed(chunk.startRow + j, chunk.startCol + i, true);
      }
    }
  }

  TMemChunk findFirstFit(TMemAllocation allocSize,
                         std::optional<int> rowIdConstraint,
                         int columnAlignment) const {
    int numRows = allocSize.numRows / allocGranularity;
    assert(kNumRows - numRows >= 0);
    assert(allocSize.numRows % allocGranularity == 0);
    int startCol = 0;
    while (1) {
      // Skip to the next aligned address.
      if (startCol % columnAlignment != 0) {
        startCol = (startCol / columnAlignment + 1) * columnAlignment;
      }
      // Iterate over possible starting rows
      for (int startRow = 0; startRow <= kNumRows - numRows; ++startRow) {
        if (rowIdConstraint && *rowIdConstraint != startRow)
          continue;
        bool fits = true;

        // Check if the block starting at (startRow, startCol) is free
        for (int i = 0; i < allocSize.numCols && fits; ++i) {
          for (int j = 0; j < numRows; ++j) {
            if (isUsed(startRow + j, startCol + i)) {
              fits = false;
              break;
            }
          }
        }

        // If a suitable block is found, return it
        if (fits) {
          TMemChunk chunk;
          chunk.startRow = startRow;
          chunk.startCol = startCol;
          chunk.numRows = numRows;
          chunk.numCols = allocSize.numCols;
          return chunk;
        }
      }
      startCol++;
    }
    return TMemChunk();
  }

private:
  bool isUsed(int row, int col) const {
    if (row + col * kNumRows >= elements.size())
      return false;
    return elements[row + col * kNumRows];
  }
  void setUsed(int row, int col, bool used) {
    assert(row + col * kNumRows < elements.size());
    elements[row + col * kNumRows] = used;
  }

  static constexpr int kNumRows = 2;
  std::vector<bool> elements;
};

static Interval<int> getLiveIntervals(Value value, Liveness &liveness,
                                      DenseMap<Operation *, int> &operationId,
                                      ArrayRef<Operation *> extraLiveUsers) {
  auto liveOperations = liveness.resolveLiveness(value);
  liveOperations.insert(liveOperations.end(), extraLiveUsers.begin(),
                        extraLiveUsers.end());
  // Merge the alloc liverange with the liverange of any view derived from the
  // allocation so we do not reuse the backing rows/cols while a later
  // materialization load/store still needs the parent allocation.
  DenseSet<Value> seenValues;
  SmallVector<Value> worklist{value};
  auto addAliasedResult = [&](Value result) {
    if (!isa<ttg::MemDescType>(result.getType()))
      return;
    auto userLiveness = liveness.resolveLiveness(result);
    liveOperations.insert(liveOperations.end(), userLiveness.begin(),
                          userLiveness.end());
    worklist.push_back(result);
  };
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seenValues.insert(current).second)
      continue;
    for (Operation *user : current.getUsers()) {
      // Keep the interval conservative even when MLIR liveness does not look
      // through tensor-memory descriptor aliases carried by control flow.
      liveOperations.push_back(user);
      if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
        addAliasedResult(selectOp.getResult());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        unsigned numControlOperands = forOp.getNumControlOperands();
        for (OpOperand &operand : forOp->getOpOperands()) {
          if (operand.getOperandNumber() < numControlOperands ||
              operand.get() != current)
            continue;
          unsigned iterArgIdx =
              operand.getOperandNumber() - numControlOperands;
          if (iterArgIdx < forOp.getRegionIterArgs().size())
            addAliasedResult(forOp.getRegionIterArgs()[iterArgIdx]);
          if (iterArgIdx < forOp.getNumResults())
            addAliasedResult(forOp.getResult(iterArgIdx));
        }
        continue;
      }
      if (user->hasTrait<OpTrait::MemDescViewTrait>() ||
          isa<TMEMSubSliceOp>(user)) {
        addAliasedResult(user->getResult(0));
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yieldOp->getParentOp();
        unsigned resultIdx = llvm::find(yieldOp.getResults(), current) -
                             yieldOp.getResults().begin();
        if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
          if (resultIdx < ifOp.getNumResults())
            addAliasedResult(ifOp.getResult(resultIdx));
        } else if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
          if (resultIdx < forOp.getNumResults())
            addAliasedResult(forOp.getResult(resultIdx));
        }
      }
    }
  }
  auto minId = std::numeric_limits<int>::max();
  auto maxId = std::numeric_limits<int>::min();
  std::for_each(liveOperations.begin(), liveOperations.end(),
                [&](Operation *liveOp) {
                  if (operationId[liveOp] < minId) {
                    minId = operationId[liveOp];
                  }
                  if ((operationId[liveOp] + 1) > maxId) {
                    maxId = operationId[liveOp] + 1;
                  }
                });
  return Interval(minId, maxId);
}

static void updateMap(MemoryBitMap &memoryMap, Interval<int> liveInterval,
                      std::multimap<int, TMemChunk> &intervalLiverangeEnd) {
  int start = liveInterval.start();
  // Add any dead liverange to the list of free intervals.
  for (auto it = intervalLiverangeEnd.begin();
       it != intervalLiverangeEnd.end();) {
    if (it->first > start)
      break;
    memoryMap.free(it->second);
    it = intervalLiverangeEnd.erase(it);
  }
}

static TMemChunk allocFirstFit(MemoryBitMap &memoryMap,
                               TMemAllocation allocSize,
                               std::optional<int> rowIdConstraint,
                               ArrayRef<TMemChunk> coexistingChunks,
                               int columnAlignment) {
  // `coexistingChunks` are all the allocations that might need to be live at
  // the same time as the current allocation plus what is known to be currently
  // live. Union those allocations with a copy of the current memory map and use
  // that to find the actual offsets.
  MemoryBitMap mapForAlloc = memoryMap;
  for (const TMemChunk &chunk : coexistingChunks)
    mapForAlloc.alloc(chunk);
  TMemChunk chunk =
      mapForAlloc.findFirstFit(allocSize, rowIdConstraint, columnAlignment);

  // Mark this chunk as allocated in the actual memory map.
  memoryMap.alloc(chunk);
  return chunk;
}

static SmallVector<Operation *> getAlloc(Value value) {
  SmallVector<Operation *> allocs;
  DenseSet<Value> seen;
  SmallVector<Value> worklist{value};

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;

    // Handle block arguments.
    if (auto arg = dyn_cast<BlockArgument>(v)) {
      Block *block = arg.getOwner();
      Operation *parentOp = block->getParentOp();

      // Handle block with predecessors.
      if (!block->isEntryBlock()) {
        for (Block *pred : block->getPredecessors()) {
          Operation *predOp = pred->getTerminator();
          auto br = dyn_cast<BranchOpInterface>(predOp);
          if (!br) {
            llvm::report_fatal_error("unhandled branch op: " +
                                     predOp->getName().getStringRef());
          }
          SmallVector<Attribute> operands(br->getNumOperands());
          auto it = llvm::find(br->getSuccessors(), block);
          unsigned idx = std::distance(br->getSuccessors().begin(), it);
          SuccessorOperands args = br.getSuccessorOperands(idx);
          Value operand =
              args.getForwardedOperands()[arg.getArgNumber() -
                                          args.getProducedOperandCount()];
          worklist.push_back(operand);
        }
        continue;
      }

      // Handle region entry arguments.
      if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(parentOp)) {
        worklist.push_back(wsOp.getExplicitCaptures()[arg.getArgNumber()]);
      } else if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        unsigned idx = arg.getArgNumber() - 1;
        worklist.push_back(forOp.getYieldedValues()[idx]);
        worklist.push_back(forOp.getInits()[idx]);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
        unsigned idx = arg.getArgNumber();
        if (arg.getParentRegion() == &whileOp.getAfter()) {
          worklist.push_back(whileOp.getConditionOp().getArgs()[idx]);
        } else {
          worklist.push_back(whileOp.getYieldedValues()[idx]);
          worklist.push_back(whileOp.getInits()[idx]);
        }
      } else if (isa<triton::FuncOp>(parentOp)) {
        // Function-entry arguments can be externally provided TMEM descriptors
        // with no local allocation to chase.
        continue;
      } else {
        llvm::report_fatal_error(
            "unhandled parent op when looking for TMEM alloc: " +
            parentOp->getName().getStringRef());
      }
      continue;
    }

    Operation *defOp = v.getDefiningOp();
    unsigned idx = cast<OpResult>(v).getResultNumber();
    if (isa<TMEMAllocOp>(defOp)) {
      allocs.push_back(defOp);
    } else if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      worklist.push_back(defOp->getOperand(0));
    } else if (auto sliceOp = dyn_cast<TMEMSubSliceOp>(defOp)) {
      worklist.push_back(sliceOp.getSrc());
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOp)) {
      worklist.push_back(selectOp.getTrueValue());
      worklist.push_back(selectOp.getFalseValue());
    } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      worklist.push_back(ifOp.thenYield().getOperand(idx));
      worklist.push_back(ifOp.elseYield().getOperand(idx));
    } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      worklist.push_back(forOp.getYieldedValues()[idx]);
      worklist.push_back(forOp.getInits()[idx]);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
      worklist.push_back(whileOp.getConditionOp().getArgs()[idx]);
    } else {
      llvm::report_fatal_error("unhandled op when looking for TMEM alloc: " +
                               defOp->getName().getStringRef());
    }
  }

  return allocs;
}

static SmallVector<Value> getMemDescViewAliasChain(Value value) {
  SmallVector<Value> aliases;
  DenseSet<Value> seen;
  Value current = value;
  while (current && seen.insert(current).second) {
    aliases.push_back(current);
    Operation *defOp = current.getDefiningOp();
    if (!defOp || !defOp->hasTrait<OpTrait::MemDescViewTrait>() ||
        defOp->getNumOperands() == 0)
      break;
    current = defOp->getOperand(0);
  }
  return aliases;
}

static bool isAliasViewChainEdge(Operation *user,
                                 const DenseSet<Value> &aliases) {
  if (!user->hasTrait<OpTrait::MemDescViewTrait>() ||
      user->getNumResults() != 1)
    return false;
  return aliases.contains(user->getResult(0));
}

static FailureOr<Value> applyAliasViewChainToTensor(
    PatternRewriter &rewriter, Location loc, Value tensor, Value sourceAlias,
    Value targetAlias, ArrayRef<Value> aliases) {
  auto sourceIt = llvm::find(aliases, sourceAlias);
  auto targetIt = llvm::find(aliases, targetAlias);
  if (sourceIt == aliases.end() || targetIt == aliases.end())
    return failure();

  unsigned sourceIdx = std::distance(aliases.begin(), sourceIt);
  unsigned targetIdx = std::distance(aliases.begin(), targetIt);
  if (sourceIdx < targetIdx)
    return failure();

  SmallVector<Operation *> viewOps;
  for (unsigned idx = sourceIdx; idx > targetIdx; --idx) {
    Operation *viewOp = aliases[idx - 1].getDefiningOp();
    if (!viewOp || viewOp->getNumOperands() == 0 ||
        viewOp->getOperand(0) != aliases[idx])
      return failure();
    if (!isa<ttg::MemDescReshapeOp, ttg::MemDescTransOp>(viewOp))
      return failure();
    viewOps.push_back(viewOp);
  }

  Value current = tensor;
  for (Operation *viewOp : viewOps) {
    if (auto reshapeOp = dyn_cast<ttg::MemDescReshapeOp>(viewOp)) {
      auto resultTy = cast<ttg::MemDescType>(reshapeOp.getType());
      current = triton::ReshapeOp::create(rewriter, loc, resultTy.getShape(),
                                          current, /*allowReorder=*/false);
      continue;
    }
    auto transOp = cast<ttg::MemDescTransOp>(viewOp);
    current = triton::TransOp::create(rewriter, loc, current,
                                      transOp.getOrder());
  }
  return current;
}

static std::optional<ttg::MemDescType>
getMMAv5ScaleStorageTypeThroughViews(Value scale) {
  auto scaleType = dyn_cast<ttg::MemDescType>(scale.getType());
  if (!scaleType)
    return std::nullopt;
  if (auto typeLocal = getMMAv5ScaleStorageType(scaleType))
    return typeLocal;

  Value current = scale;
  while (Operation *defOp = current.getDefiningOp()) {
    if (!defOp->hasTrait<OpTrait::MemDescViewTrait>() ||
        defOp->getNumOperands() == 0)
      return std::nullopt;

    current = defOp->getOperand(0);
    auto currentType = dyn_cast<ttg::MemDescType>(current.getType());
    if (!currentType)
      return std::nullopt;
    if (!isa<TensorMemoryScalesEncodingAttr>(currentType.getEncoding()))
      continue;
    if (currentType.getElementType() != scaleType.getElementType() ||
        currentType.getMemorySpace() != scaleType.getMemorySpace())
      return std::nullopt;
    return ttg::MemDescType::get(scaleType.getShape(),
                                 scaleType.getElementType(),
                                 currentType.getEncoding(),
                                 scaleType.getMemorySpace(),
                                 scaleType.getMutableMemory());
  }
  return std::nullopt;
}

// This pass may inspect scale view producers because it rewrites the IR before
// lowering. Verifier and LLVM lowering must use the current MemDescType only.
static std::optional<ttg::MemDescType>
getMMAv5ScaledBScaleStorageTypeThroughViews(Value bScale) {
  auto bScaleType = dyn_cast<ttg::MemDescType>(bScale.getType());
  if (!bScaleType)
    return std::nullopt;
  if (auto typeLocal = getMMAv5ScaledBScaleStorageType(bScaleType))
    return typeLocal;

  Value current = bScale;
  while (Operation *defOp = current.getDefiningOp()) {
    if (!defOp->hasTrait<OpTrait::MemDescViewTrait>() ||
        defOp->getNumOperands() == 0)
      return std::nullopt;

    current = defOp->getOperand(0);
    auto currentType = dyn_cast<ttg::MemDescType>(current.getType());
    if (!currentType)
      return std::nullopt;
    if (!isa<TensorMemoryScalesEncodingAttr>(currentType.getEncoding()))
      continue;
    if (currentType.getElementType() != bScaleType.getElementType() ||
        currentType.getMemorySpace() != bScaleType.getMemorySpace())
      return std::nullopt;
    return ttg::MemDescType::get(bScaleType.getShape(),
                                 bScaleType.getElementType(),
                                 currentType.getEncoding(),
                                 bScaleType.getMemorySpace(),
                                 bScaleType.getMutableMemory());
  }
  return std::nullopt;
}

struct TMemScaleStoreInfo {
  SmallVector<Value> aliases;
  DenseSet<Value> aliasSet;
  TMEMAllocOp allocOp;
  TMEMStoreOp storeOp;
  Value storeAlias;
  bool hasOtherUsers = false;
};

struct DeferredScaleCleanup {
  TMemScaleStoreInfo info;
  Value scale;
};

static FailureOr<TMemScaleStoreInfo>
getSingleTMemScaleStoreInfo(Value scale, Operation *consumer,
                            unsigned ignoredConsumerOperand) {
  auto scaleType = cast<ttg::MemDescType>(scale.getType());
  SmallVector<Operation *> allocs = getAlloc(scale);
  if (allocs.size() != 1)
    return failure();
  auto allocOp = dyn_cast<TMEMAllocOp>(allocs.front());
  if (!allocOp || allocOp.getSrc())
    return failure();
  auto allocType = cast<ttg::MemDescType>(allocOp.getType());
  if (allocType.getElementType() != scaleType.getElementType() ||
      allocType.getMemorySpace() != scaleType.getMemorySpace())
    return failure();

  TMemScaleStoreInfo info;
  info.allocOp = allocOp;
  info.aliases = getMemDescViewAliasChain(scale);
  for (Value alias : info.aliases)
    info.aliasSet.insert(alias);

  for (Value alias : info.aliases) {
    for (Operation *user : alias.getUsers()) {
      auto candidate = dyn_cast<TMEMStoreOp>(user);
      if (!candidate || candidate.getDst() != alias)
        continue;
      if (info.storeOp)
        return failure();
      info.storeOp = candidate;
      info.storeAlias = alias;
    }
  }
  if (!info.storeOp)
    return failure();
  if (info.storeOp->getBlock() != consumer->getBlock() ||
      !info.storeOp->isBeforeInBlock(consumer))
    return failure();

  for (Value alias : info.aliases) {
    for (Operation *user : llvm::make_early_inc_range(alias.getUsers())) {
      if (user == consumer || user == info.storeOp.getOperation() ||
          isAliasViewChainEdge(user, info.aliasSet))
        continue;
      info.hasOtherUsers = true;
    }
  }
  for (OpOperand &operand : consumer->getOpOperands()) {
    if (operand.getOperandNumber() == ignoredConsumerOperand)
      continue;
    if (info.aliasSet.contains(operand.get()))
      info.hasOtherUsers = true;
  }

  auto storedType =
      dyn_cast<RankedTensorType>(info.storeOp.getSrc().getType());
  auto storeAliasType = cast<ttg::MemDescType>(info.storeAlias.getType());
  if (!storedType || storedType.getShape() != storeAliasType.getShape())
    return failure();
  return info;
}

static void cleanupScaleAliasChain(PatternRewriter &rewriter,
                                   TMemScaleStoreInfo &info, Value scale) {
  if (!info.hasOtherUsers) {
    rewriter.eraseOp(info.storeOp);
    Value unusedView = scale;
    while (Operation *defOp = unusedView.getDefiningOp()) {
      if (defOp == info.allocOp.getOperation() || !defOp->use_empty() ||
          !defOp->hasTrait<OpTrait::MemDescViewTrait>())
        break;
      unusedView = defOp->getOperand(0);
      rewriter.eraseOp(defOp);
    }
  }
  if (info.allocOp->use_empty())
    rewriter.eraseOp(info.allocOp);
}

class MaterializeSharedMMAScalesToTMem
    : public OpRewritePattern<TCGen5MMAScaledOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult materializeScale(OpOperand &operand, int64_t rows,
                                 PatternRewriter &rewriter) const {
    auto scaleType = cast<ttg::MemDescType>(operand.get().getType());
    std::optional<ttg::MemDescType> tmemScaleType =
        getMMAv5ScaleTMemTypeForSharedScale(scaleType, rows);
    if (!tmemScaleType) {
      return operand.getOwner()->emitError()
             << "cannot materialize shared scale operand with shape "
             << scaleType.getShape()
             << " into a tensor-memory scales layout with " << rows
             << " MMA rows";
    }

    Location loc = operand.getOwner()->getLoc();
    Value tmemAlloc =
        TMEMAllocOp::create(rewriter, loc, *tmemScaleType, Value());
    TMEMCopyOp::create(rewriter, loc, operand.get(), tmemAlloc,
                       /*barrier=*/Value());
    operand.set(tmemAlloc);
    return success();
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    auto aScaleType = mmaOp.getAScale().getType();
    if (isa<ttg::SharedMemorySpaceAttr>(aScaleType.getMemorySpace())) {
      if (failed(materializeScale(mmaOp.getAScaleMutable(), mmaOp.getBlockM(),
                                  rewriter)))
        return failure();
      changed = true;
    }

    auto bScaleType = mmaOp.getBScale().getType();
    if (isa<ttg::SharedMemorySpaceAttr>(bScaleType.getMemorySpace())) {
      if (failed(materializeScale(mmaOp.getBScaleMutable(), mmaOp.getBlockN(),
                                  rewriter)))
        return failure();
      changed = true;
    }

    return success(changed);
  }
};

class RematerializeScaledMmaBScaleFragments
    : public OpRewritePattern<TCGen5MMAScaledOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  FailureOr<Value> rematerializeDirectBScale(
      Value bScale, ttg::MemDescType bScaleStorageType,
      ArrayRef<int64_t> rematerializedShape, unsigned ctaColumns,
      unsigned instrSizeN, Operation *consumer, unsigned ignoredConsumerOperand,
      PatternRewriter &rewriter,
      SmallVectorImpl<DeferredScaleCleanup> *deferredCleanups = nullptr) const {
    auto bScaleType = cast<ttg::MemDescType>(bScale.getType());
    FailureOr<TMemScaleStoreInfo> maybeInfo =
        getSingleTMemScaleStoreInfo(bScale, consumer, ignoredConsumerOperand);
    if (failed(maybeInfo))
      return failure();
    TMemScaleStoreInfo &info = *maybeInfo;

    if (bScaleType.getRank() != 2)
      return failure();

    rewriter.setInsertionPoint(info.storeOp);
    Value stored = info.storeOp.getSrc();
    if (info.storeAlias != bScale) {
      FailureOr<Value> logicalStored = applyAliasViewChainToTensor(
          rewriter, info.storeOp.getLoc(), stored, info.storeAlias, bScale,
          info.aliases);
      if (failed(logicalStored))
        return failure();
      stored = *logicalStored;
    }

    auto storedType = cast<RankedTensorType>(stored.getType());
    assert(storedType.getShape() == bScaleType.getShape() &&
           "alias view rematerialization should materialize the B-scale logical shape");

    if (storedType.getShape()[0] != static_cast<int64_t>(ctaColumns) ||
        instrSizeN == 0 || ctaColumns % instrSizeN != 0)
      return failure();

    int64_t paddingFactor =
        rematerializedShape[0] / storedType.getShape()[0];
    if (paddingFactor <= 1 ||
        rematerializedShape[0] % storedType.getShape()[0] != 0)
      return failure();

    int64_t instructionCount = ctaColumns / instrSizeN;
    SmallVector<int64_t> groupedShape{
        instructionCount, 1, static_cast<int64_t>(instrSizeN),
        storedType.getShape()[1]};

    Location loc = info.storeOp.getLoc();
    Value grouped =
        triton::ReshapeOp::create(rewriter, loc, groupedShape, stored,
                                  /*allowReorder=*/false);
    auto groupedType = cast<RankedTensorType>(grouped.getType());
    SmallVector<int64_t> broadcastShape(groupedType.getShape().begin(),
                                        groupedType.getShape().end());
    broadcastShape[1] = paddingFactor;
    Value broadcasted = triton::BroadcastOp::create(
        rewriter, loc, groupedType.clone(broadcastShape), grouped);
    Value rematerialized =
        triton::ReshapeOp::create(rewriter, loc, rematerializedShape,
                                  broadcasted, /*allowReorder=*/false);

    auto rematerializedType = ttg::MemDescType::get(
        rematerializedShape, bScaleStorageType.getElementType(),
        bScaleStorageType.getEncoding(), bScaleStorageType.getMemorySpace(),
        bScaleStorageType.getMutableMemory());
    auto rematerializedTensorType =
        cast<RankedTensorType>(rematerialized.getType());
    if (!isDistributedLayoutTMemCompatible(info.storeOp.getOperation(),
                                           rematerializedTensorType,
                                           rematerializedType)) {
      SmallVector<ttg::DistributedEncodingTrait> layouts =
          getTmemCompatibleLayouts(info.storeOp.getOperation(),
                                   rematerializedTensorType,
                                   rematerializedType);
      if (layouts.empty())
        return failure();
      auto convertedType = rematerializedTensorType.cloneWithEncoding(layouts[0]);
      rematerialized = ttg::ConvertLayoutOp::create(rewriter, loc,
                                                    convertedType,
                                                    rematerialized);
    }

    auto rematerializedAlloc =
        TMEMAllocOp::create(rewriter, loc, rematerializedType, Value());
    TMEMStoreOp::create(rewriter, loc, rematerializedAlloc.getResult(),
                        rematerialized, info.storeOp.getPred());
    if (deferredCleanups) {
      deferredCleanups->push_back(DeferredScaleCleanup{std::move(info), bScale});
      return rematerializedAlloc.getResult();
    }
    cleanupScaleAliasChain(rewriter, info, bScale);
    return rematerializedAlloc.getResult();
  }

  FailureOr<Value> rematerializeBScale(
      Value bScale, ttg::MemDescType bScaleStorageType,
      ArrayRef<int64_t> rematerializedShape, unsigned ctaColumns,
      unsigned instrSizeN, Operation *consumer, unsigned ignoredConsumerOperand,
      PatternRewriter &rewriter,
      SmallVectorImpl<DeferredScaleCleanup> &deferredCleanups,
      SmallVectorImpl<Operation *> &deadOps) const {
    if (auto selectOp = bScale.getDefiningOp<arith::SelectOp>()) {
      // Split B-scale rematerialization through dynamic selection. Single-use
      // selects can be erased after the MMA is rewritten, which lets us clean
      // up the original unpadded branch stores. Multi-use selects keep the
      // original descriptor for other consumers and materialize a new padded
      // select for this MMA only.
      bool canCleanupSelect = bScale.hasOneUse();
      SmallVector<DeferredScaleCleanup> branchCleanups;
      SmallVectorImpl<DeferredScaleCleanup> *cleanupSink =
          canCleanupSelect ? &branchCleanups : nullptr;
      Operation *branchConsumer =
          canCleanupSelect ? selectOp.getOperation() : consumer;
      unsigned trueOperand = canCleanupSelect
                                 ? selectOp.getTrueValueMutable()
                                       .getOperandNumber()
                                 : ignoredConsumerOperand;
      unsigned falseOperand = canCleanupSelect
                                  ? selectOp.getFalseValueMutable()
                                        .getOperandNumber()
                                  : ignoredConsumerOperand;
      FailureOr<Value> trueScale = rematerializeDirectBScale(
          selectOp.getTrueValue(), bScaleStorageType, rematerializedShape,
          ctaColumns, instrSizeN, branchConsumer, trueOperand, rewriter,
          cleanupSink);
      if (failed(trueScale))
        return failure();
      FailureOr<Value> falseScale = rematerializeDirectBScale(
          selectOp.getFalseValue(), bScaleStorageType, rematerializedShape,
          ctaColumns, instrSizeN, branchConsumer, falseOperand, rewriter,
          cleanupSink);
      if (failed(falseScale))
        return failure();

      rewriter.setInsertionPoint(selectOp);
      Value selected = arith::SelectOp::create(
                           rewriter, selectOp.getLoc(),
                           selectOp.getCondition(), *trueScale, *falseScale)
                           .getResult();
      if (canCleanupSelect) {
        llvm::move(branchCleanups, std::back_inserter(deferredCleanups));
        deadOps.push_back(selectOp.getOperation());
      }
      return selected;
    }

    return rematerializeDirectBScale(
        bScale, bScaleStorageType, rematerializedShape, ctaColumns, instrSizeN,
        consumer, ignoredConsumerOperand, rewriter);
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    auto accSupport = getMMAv5ScaledAccumulatorSupport(mmaOp.getD().getType());
    if (!accSupport.repeatedN32ScaleFragmentRequirement &&
        !accSupport.narrowNScaleFragmentRequirement)
      return failure();

    Value bScale = mmaOp.getBScale();
    std::optional<ttg::MemDescType> bScaleStorageType =
        getMMAv5ScaledBScaleStorageTypeThroughViews(bScale);
    if (!bScaleStorageType)
      return failure();

    std::optional<SmallVector<int64_t>> rematerializedShape = std::nullopt;
    unsigned ctaColumns = 0;
    unsigned instrSizeN = 0;
    if (accSupport.narrowNScaleFragmentRequirement) {
      const MMAv5ScaledNarrowNScaleFragmentRequirement &requirement =
          *accSupport.narrowNScaleFragmentRequirement;
      if (isMMAv5ScaledNarrowNBScaleStorageSupported(*bScaleStorageType,
                                                     requirement))
        return failure();
      rematerializedShape =
          getMMAv5ScaledNarrowNBScaleRematerializedShape(*bScaleStorageType,
                                                         requirement);
      ctaColumns = requirement.ctaColumns;
      instrSizeN = requirement.instrSizeN;
    } else {
      const MMAv5ScaledRepeatedN32ScaleFragmentRequirement &requirement =
          *accSupport.repeatedN32ScaleFragmentRequirement;
      if (isMMAv5ScaledRepeatedN32BScaleStorageSupported(*bScaleStorageType,
                                                         requirement))
        return failure();
      rematerializedShape =
          getMMAv5ScaledRepeatedN32BScaleRematerializedShape(*bScaleStorageType,
                                                            requirement);
      ctaColumns = requirement.ctaColumns;
      instrSizeN = requirement.instrSizeN;
    }
    if (!rematerializedShape)
      return failure();

    SmallVector<DeferredScaleCleanup> deferredCleanups;
    SmallVector<Operation *> deadOps;
    FailureOr<Value> rematerializedBScale = rematerializeBScale(
        bScale, *bScaleStorageType, *rematerializedShape, ctaColumns,
        instrSizeN, mmaOp.getOperation(),
        mmaOp.getBScaleMutable().getOperandNumber(), rewriter,
        deferredCleanups, deadOps);
    if (failed(rematerializedBScale))
      return failure();

    rewriter.modifyOpInPlace(mmaOp, [&] {
      mmaOp.getBScaleMutable().assign(*rematerializedBScale);
    });
    for (Operation *deadOp : deadOps) {
      if (deadOp->use_empty())
        rewriter.eraseOp(deadOp);
    }
    for (DeferredScaleCleanup &cleanup : deferredCleanups)
      cleanupScaleAliasChain(rewriter, cleanup.info, cleanup.scale);
    return success();
  }
};

class RematerializeScaledMmaScaleDescriptorViews
    : public OpRewritePattern<TCGen5MMAScaledOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  FailureOr<Value> rematerializeDirectScale(
      Value scale, ttg::MemDescType storageType, Operation *consumer,
      unsigned ignoredConsumerOperand, PatternRewriter &rewriter,
      SmallVectorImpl<DeferredScaleCleanup> *deferredCleanups = nullptr) const {
    auto scaleType = cast<ttg::MemDescType>(scale.getType());
    if (isa<TensorMemoryScalesEncodingAttr>(scaleType.getEncoding()))
      return failure();

    FailureOr<TMemScaleStoreInfo> maybeInfo =
        getSingleTMemScaleStoreInfo(scale, consumer, ignoredConsumerOperand);
    if (failed(maybeInfo))
      return failure();
    TMemScaleStoreInfo &info = *maybeInfo;

    rewriter.setInsertionPoint(info.storeOp);
    Value stored = info.storeOp.getSrc();
    if (info.storeAlias != scale) {
      FailureOr<Value> logicalStored = applyAliasViewChainToTensor(
          rewriter, info.storeOp.getLoc(), stored, info.storeAlias, scale,
          info.aliases);
      if (failed(logicalStored))
        return failure();
      stored = *logicalStored;
    }

    auto storedType = cast<RankedTensorType>(stored.getType());
    assert(storedType.getShape() == storageType.getShape() &&
           "alias view rematerialization should materialize the scale logical shape");
    Value materialized = stored;
    if (!isDistributedLayoutTMemCompatible(info.storeOp.getOperation(),
                                           storedType, storageType)) {
      SmallVector<ttg::DistributedEncodingTrait> layouts =
          getTmemCompatibleLayouts(info.storeOp.getOperation(), storedType,
                                   storageType);
      if (layouts.empty())
        return failure();
      auto convertedType = storedType.cloneWithEncoding(layouts[0]);
      materialized = ttg::ConvertLayoutOp::create(
          rewriter, info.storeOp.getLoc(), convertedType, materialized);
    }

    auto rematerializedAlloc = TMEMAllocOp::create(
        rewriter, info.storeOp.getLoc(), storageType, Value());
    TMEMStoreOp::create(rewriter, info.storeOp.getLoc(),
                        rematerializedAlloc.getResult(), materialized,
                        info.storeOp.getPred());
    if (deferredCleanups) {
      deferredCleanups->push_back(DeferredScaleCleanup{std::move(info), scale});
      return rematerializedAlloc.getResult();
    }
    cleanupScaleAliasChain(rewriter, info, scale);
    return rematerializedAlloc.getResult();
  }

  FailureOr<Value> rematerializeScaleValue(
      Value scale, ttg::MemDescType storageType, Operation *consumer,
      unsigned ignoredConsumerOperand, PatternRewriter &rewriter,
      SmallVectorImpl<DeferredScaleCleanup> &deferredCleanups,
      SmallVectorImpl<Operation *> &deadOps) const {
    if (auto selectOp = scale.getDefiningOp<arith::SelectOp>()) {
      bool canCleanupSelect = scale.hasOneUse();
      SmallVector<DeferredScaleCleanup> branchCleanups;
      SmallVectorImpl<DeferredScaleCleanup> *cleanupSink =
          canCleanupSelect ? &branchCleanups : nullptr;
      Operation *branchConsumer =
          canCleanupSelect ? selectOp.getOperation() : consumer;
      unsigned trueOperand = canCleanupSelect
                                 ? selectOp.getTrueValueMutable()
                                       .getOperandNumber()
                                 : ignoredConsumerOperand;
      unsigned falseOperand = canCleanupSelect
                                  ? selectOp.getFalseValueMutable()
                                        .getOperandNumber()
                                  : ignoredConsumerOperand;
      FailureOr<Value> trueScale = rematerializeDirectScale(
          selectOp.getTrueValue(), storageType, branchConsumer, trueOperand,
          rewriter, cleanupSink);
      if (failed(trueScale))
        return failure();
      FailureOr<Value> falseScale = rematerializeDirectScale(
          selectOp.getFalseValue(), storageType, branchConsumer, falseOperand,
          rewriter, cleanupSink);
      if (failed(falseScale))
        return failure();

      rewriter.setInsertionPoint(selectOp);
      Value selected = arith::SelectOp::create(
                           rewriter, selectOp.getLoc(),
                           selectOp.getCondition(), *trueScale, *falseScale)
                           .getResult();
      if (canCleanupSelect) {
        llvm::move(branchCleanups, std::back_inserter(deferredCleanups));
        deadOps.push_back(selectOp.getOperation());
      }
      return selected;
    }

    return rematerializeDirectScale(scale, storageType, consumer,
                                    ignoredConsumerOperand, rewriter);
  }

  LogicalResult rematerializeScale(OpOperand &operand,
                                   TCGen5MMAScaledOp mmaOp,
                                   PatternRewriter &rewriter) const {
    Value scale = operand.get();
    auto scaleType = cast<ttg::MemDescType>(scale.getType());
    if (isa<TensorMemoryScalesEncodingAttr>(scaleType.getEncoding()))
      return failure();

    std::optional<ttg::MemDescType> storageType =
        getMMAv5ScaleStorageTypeThroughViews(scale);
    if (!storageType)
      return failure();

    SmallVector<DeferredScaleCleanup> deferredCleanups;
    SmallVector<Operation *> deadOps;
    FailureOr<Value> rematerializedScale = rematerializeScaleValue(
        scale, *storageType, mmaOp.getOperation(), operand.getOperandNumber(),
        rewriter, deferredCleanups, deadOps);
    if (failed(rematerializedScale))
      return failure();

    rewriter.modifyOpInPlace(mmaOp, [&] {
      operand.assign(*rematerializedScale);
    });
    for (Operation *deadOp : deadOps) {
      if (deadOp->use_empty())
        rewriter.eraseOp(deadOp);
    }
    for (DeferredScaleCleanup &cleanup : deferredCleanups)
      cleanupScaleAliasChain(rewriter, cleanup.info, cleanup.scale);
    return success();
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp mmaOp,
                                PatternRewriter &rewriter) const override {
    bool changed = false;
    if (succeeded(rematerializeScale(mmaOp.getAScaleMutable(), mmaOp,
                                     rewriter)))
      changed = true;
    if (succeeded(rematerializeScale(mmaOp.getBScaleMutable(), mmaOp,
                                     rewriter)))
      changed = true;
    return success(changed);
  }
};

class RowIdConstraints {
  llvm::EquivalenceClasses<Operation *> dependentAllocs;
  llvm::SmallDenseMap<Operation *, int> rowIndex;

public:
  void joinOps(Operation *op1, Operation *op2) {
    dependentAllocs.unionSets(op1, op2);
  }

  std::optional<int> getRowIdConstraint(Operation *op) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return std::nullopt;
    auto rowIt = rowIndex.find(*it);
    if (rowIt == rowIndex.end())
      return std::nullopt;
    return rowIt->second;
  }

  void addConstraints(Operation *op, int rowId) {
    auto it = dependentAllocs.findLeader(op);
    if (it == dependentAllocs.member_end())
      return;
    rowIndex[*it] = rowId;
  }
};

static int
allocateTMem(Operation *parentOp,
             DenseMap<triton::nvidia_gpu::TMEMAllocOp, int> &offsets) {
  SmallVector<triton::nvidia_gpu::TMEMAllocOp> allocs;
  DenseMap<Operation *, int> operationId;
  DenseMap<Operation *, SmallVector<Operation *>> extraLiveUsers;
  RowIdConstraints rowIdConstraints;
  parentOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    operationId[op] = operationId.size();
    if (auto alloc = dyn_cast<triton::nvidia_gpu::TMEMAllocOp>(op)) {
      allocs.push_back(alloc);
    }
    if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
      auto aTMemInfo = getMMAv5LhsLayoutInfo(mmaOp.getA().getType());
      if (aTMemInfo) {
        TMemAllocation allocSize = getTmemAllocSizes(mmaOp.getA().getType());
        if (allocSize.numRows == 64) {
          // HW restriction, the A alloc and accumulator needs to be in the same
          // rows.
          SmallVector<Operation *> lhsAllocs = getAlloc(mmaOp.getA());
          SmallVector<Operation *> accAllocs = getAlloc(mmaOp.getAccumulator());
          for (Operation *lhsAlloc : lhsAllocs)
            for (Operation *accAlloc : accAllocs)
              rowIdConstraints.joinOps(lhsAlloc, accAlloc);
        }
      }
    }
  });
  parentOp->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto memDescType = dyn_cast<ttg::MemDescType>(operand.getType());
      if (!memDescType ||
          !isa<TensorMemorySpaceAttr>(memDescType.getMemorySpace()))
        continue;
      // Consumers can see a selected/yielded memdesc while the actual storage
      // comes from multiple root TMEM allocations. Keep all roots live through
      // that consumer so the allocator cannot reuse their backing columns.
      for (Operation *alloc : getAlloc(operand))
        extraLiveUsers[alloc].push_back(op);
    }
  });
  int totalMemorySize = 0;
  MemoryBitMap memoryMap;
  Liveness liveness(parentOp);
  std::multimap<int, TMemChunk> intervalLiverangeEnd;
  DenseMap<TMEMAllocOp, TMemChunk> allocChunks;
  // Implement a linear scan first fit algorithm. We expect that fragmentation
  // won't be a problem, if it is this should be revisited.
  for (auto it = allocs.begin(), e = allocs.end(); it != e; ++it) {
    TMEMAllocOp alloc = *it;

    // Find all allocations in code that may execute at the same time. Only look
    // at processed allocations.
    SmallVector<TMemChunk> coexistingChunks;
    if (auto ws = alloc->getParentOfType<triton::gpu::WarpSpecializeOp>()) {
      for (auto prevIt = allocs.begin(); prevIt != it; ++prevIt) {
        TMEMAllocOp prevAlloc = *prevIt;
        auto prevWs =
            prevAlloc->getParentOfType<triton::gpu::WarpSpecializeOp>();
        if (prevWs && prevWs == ws &&
            alloc->getParentRegion() != prevAlloc->getParentRegion())
          coexistingChunks.push_back(allocChunks.at(prevAlloc));
      }
    }

    auto extraIt = extraLiveUsers.find(alloc.getOperation());
    ArrayRef<Operation *> allocExtraLiveUsers =
        extraIt == extraLiveUsers.end() ? ArrayRef<Operation *>()
                                        : ArrayRef<Operation *>(extraIt->second);
    Interval<int> liveInterval =
        getLiveIntervals(alloc, liveness, operationId, allocExtraLiveUsers);
    auto memDescType = alloc.getType();
    TMemAllocation allocSize = getTmemAllocSizes(memDescType);
    updateMap(memoryMap, liveInterval, intervalLiverangeEnd);

    std::optional<int> rowIdConstraint =
        rowIdConstraints.getRowIdConstraint(alloc);
    // TODO: clarify the alignment requirements for different allocations. For
    // now enforce an alignment of 4 columns.
    const int columnAlignment = 4;
    TMemChunk chunkAllocated =
        allocFirstFit(memoryMap, allocSize, rowIdConstraint, coexistingChunks,
                      columnAlignment);
    allocChunks.insert({alloc, chunkAllocated});
    // currently naively constraint allocs based on the first one we find.
    rowIdConstraints.addConstraints(alloc, chunkAllocated.startRow);
    intervalLiverangeEnd.insert({liveInterval.end(), chunkAllocated});
    int colOffset = chunkAllocated.startCol;
    int rowOffset = chunkAllocated.startRow * 16;

    alloc->setAttr(
        "tensor_memory_col_offset",
        IntegerAttr::get(IntegerType::get(parentOp->getContext(), 32),
                         colOffset));
    alloc->setAttr(
        "tensor_memory_row_offset",
        IntegerAttr::get(IntegerType::get(parentOp->getContext(), 32),
                         rowOffset));
    totalMemorySize = std::max(totalMemorySize, colOffset + allocSize.numCols);
  }
  return totalMemorySize;
}

} // anonymous namespace

class TritonTensorMemoryAllocationPass
    : public impl::TritonTensorMemoryAllocationPassBase<
          TritonTensorMemoryAllocationPass> {
public:
  IntegerAttr getI32Attr(int32_t value) {
    return Builder(&getContext()).getI32IntegerAttr(value);
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();

    DenseMap<triton::nvidia_gpu::TMEMAllocOp, int> offsets;
    RewritePatternSet patterns(ctx);
    patterns.add<MaterializeSharedMMAScalesToTMem,
                 RematerializeScaledMmaBScaleFragments,
                 RematerializeScaledMmaScaleDescriptorViews>(ctx);
    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();

    // TODO: handle cases with multiple function with TMEMAllocOp.
    int totalMemorySize = allocateTMem(mod, offsets);

    std::array<int, 6> possibleAllocations = {0, 32, 64, 128, 256, 512};
    // NOTE: if totalMemorySize > 512 we exceeded the maximum amount of tensor
    // memory, but we let the compilation finish so that we can raise an
    // exception in python for the auto-tuner.
    if (totalMemorySize <= 512) {
      for (int size : possibleAllocations) {
        if (totalMemorySize <= size) {
          totalMemorySize = size;
          break;
        }
      }
    }
    if (totalMemorySize > 0) {
      // We use a small smem allocation to get the tensor memory base address
      // from tcgen05.alloc, ensure the block has at least 4 bytes of smem
      int shared = 0;
      if (auto sharedAttr = mod->getAttr("ttg.shared")) {
        shared = cast<IntegerAttr>(sharedAttr).getInt();
      }
      if (shared < 4) {
        mod->setAttr("ttg.shared", getI32Attr(4));
      }
    }
    mod->setAttr("ttg.tensor_memory_size", getI32Attr(totalMemorySize));
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

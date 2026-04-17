#include "ir.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include <cstdlib>
#include <numeric>
#include <optional>
#include <stdexcept>

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Types.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/TensorMemoryUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
namespace py = pybind11;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace gluon = mlir::triton::gluon;
namespace ttag = mlir::triton::amdgpu;

static ttg::CGAEncodingAttr
buildCgaLayoutAttr(MLIRContext *ctx,
                   const std::vector<std::vector<int32_t>> &layout,
                   unsigned rank) {
  auto kBlock = StringAttr::get(ctx, "block");
  tt::LinearLayout::BasesT bases;
  bases[kBlock] = layout;
  auto outDims = tt::standardOutDimNames(ctx, rank);
  tt::LinearLayout ll(std::move(bases), outDims);
  return ttg::CGAEncodingAttr::get(ctx, std::move(ll));
}

static std::vector<std::vector<int32_t>>
getCgaLayoutBases(ttg::CGAEncodingAttr layout) {
  std::vector<std::vector<int32_t>> result;
  auto ctx = layout.getContext();
  auto block = StringAttr::get(ctx, "block");
  const auto &basesMap = layout.getLinearLayout().getBases();
  auto it = basesMap.find(block);
  assert(it != basesMap.end());
  return it->second;
}

// Helper to check if an MLIR type or attribute has a verifier method.
template <typename AttrOrType>
static constexpr auto hasVerifier(AttrOrType t)
    -> decltype(t.verifyInvariants, true) {
  return true;
}
static constexpr auto hasVerifier(...) { return false; }

// Print a diagnostic without its location. The frontend will attach the AST
// location to the error message.
static void printDiagStr(llvm::raw_ostream &os, const Diagnostic &diag) {
  for (const DiagnosticArgument &arg : diag.getArguments())
    arg.print(os);
  os << "\n";
  for (const Diagnostic &note : diag.getNotes())
    printDiagStr(os, note);
}

struct GluonOpBuilder : public TritonOpBuilder {
  using TritonOpBuilder::TritonOpBuilder;
  // Construct an attribute or type while calling its verifier. Error messages
  // are intercepted and sent back to Python via a C++ exception.
  template <typename AttrOrType, typename... ArgTs>
  std::enable_if_t<hasVerifier(AttrOrType()), AttrOrType>
  getChecked(ArgTs &&...args) {
    // Set up a scoped handler to intercept errors.
    std::string msg;
    llvm::raw_string_ostream os(msg);
    ScopedDiagnosticHandler handler(
        getContext(), [&](Diagnostic &diag) { printDiagStr(os, diag); });

    auto result =
        AttrOrType::getChecked([&] { return mlir::emitError(getLastLoc()); },
                               std::forward<ArgTs>(args)...);
    if (!result)
      throw std::runtime_error(os.str());
    return result;
  }

  // A variant of the above due to issues with C++ overload resolution and how
  // MLIR sets up the default `getChecked` implementation.
  template <typename AttrOrType, typename... ArgTs>
  std::enable_if_t<hasVerifier(AttrOrType()), AttrOrType>
  getChecked(MLIRContext *ctx, ArgTs &&...args) {
    // Set up a scoped handler to intercept errors.
    std::string msg;
    llvm::raw_string_ostream os(msg);
    ScopedDiagnosticHandler handler(
        getContext(), [&](Diagnostic &diag) { printDiagStr(os, diag); });

    if (failed(AttrOrType::verifyInvariants(
            [&] { return mlir::emitError(getLastLoc()); }, args...)))
      throw std::runtime_error(os.str());

    return AttrOrType::get(ctx, std::forward<ArgTs>(args)...);
  }

  // Fallback method for types or attributes that do not have a verifier.
  template <typename AttrOrType, typename... ArgTs>
  std::enable_if_t<!hasVerifier(AttrOrType()), AttrOrType>
  getChecked(ArgTs &&...args) {
    return AttrOrType::get(std::forward<ArgTs>(args)...);
  }
};

template <typename CreateFn>
static auto createCheckedOrThrow(GluonOpBuilder &builder,
                                 llvm::StringRef message,
                                 CreateFn &&createFn) {
  std::string diagStr;
  llvm::raw_string_ostream diagOs(diagStr);
  ScopedDiagnosticHandler handler(
      builder.getContext(),
      [&](Diagnostic &diag) { printDiagStr(diagOs, diag); });

  auto result = createFn();
  if (failed(result)) {
    if (diagStr.empty())
      throw py::value_error(message.str().c_str());
    std::string error = message.str();
    error += "\n";
    error += diagOs.str();
    throw py::value_error(error.c_str());
  }
  return *result;
}

struct GluonLayouts {
  py::handle AutoLayout;
  py::handle CoalescedLayout;
  py::handle BlockedLayout;
  py::handle SliceLayout;
  py::handle DistributedLinearLayout;
  py::handle DotOperandLayout;
  py::handle NVMMADistributedLayout;
  py::handle TensorMemoryScalesLayout;
  py::handle TensorMemoryLayout;
  py::handle TensorMemoryLinearLayout;
  py::handle NVMMASharedLayout;
  py::handle SwizzledSharedLayout;
  py::handle SharedLinearLayout;
  py::handle AMDMFMALayout;
  py::handle AMDWMMALayout;
  py::handle PaddedSharedLayout;
  py::handle PartitionedSharedLayout;

  GluonLayouts() {
    auto layouts =
        py::module::import("triton.experimental.gluon.language._layouts");
    auto amdLayouts =
        py::module::import("triton.experimental.gluon.language.amd._layouts");
    auto blackwellLayouts = py::module::import(
        "triton.experimental.gluon.language.nvidia.blackwell");
    AutoLayout = py::object(layouts.attr("AutoLayout")).release();
    CoalescedLayout = py::object(layouts.attr("CoalescedLayout")).release();
    BlockedLayout = py::object(layouts.attr("BlockedLayout")).release();
    SliceLayout = py::object(layouts.attr("SliceLayout")).release();
    DistributedLinearLayout =
        py::object(layouts.attr("DistributedLinearLayout")).release();
    DotOperandLayout = py::object(layouts.attr("DotOperandLayout")).release();
    NVMMADistributedLayout =
        py::object(layouts.attr("NVMMADistributedLayout")).release();
    TensorMemoryScalesLayout =
        py::object(blackwellLayouts.attr("TensorMemoryScalesLayout")).release();
    TensorMemoryLayout =
        py::object(blackwellLayouts.attr("TensorMemoryLayout")).release();
    TensorMemoryLinearLayout =
        py::object(blackwellLayouts.attr("TensorMemoryLinearLayout"))
            .release();
    NVMMASharedLayout = py::object(layouts.attr("NVMMASharedLayout")).release();
    SwizzledSharedLayout =
        py::object(layouts.attr("SwizzledSharedLayout")).release();
    SharedLinearLayout =
        py::object(layouts.attr("SharedLinearLayout")).release();
    AMDMFMALayout = py::object(amdLayouts.attr("AMDMFMALayout")).release();
    AMDWMMALayout = py::object(amdLayouts.attr("AMDWMMALayout")).release();
    PaddedSharedLayout =
        py::object(layouts.attr("PaddedSharedLayout")).release();
    auto gfx1250Layouts = py::module::import(
        "triton.experimental.gluon.language.amd.gfx1250._layouts");
    PartitionedSharedLayout =
        py::object(gfx1250Layouts.attr("PartitionedSharedLayout")).release();

    auto core = py::module::import("triton.language.core");
  }
};

static bool isConvertLayoutTrivial(RankedTensorType dstTy, Value value) {
  auto srcTy = cast<RankedTensorType>(value.getType());
  if (srcTy.getEncoding() == dstTy.getEncoding())
    return true;
  // Fail safe on unresolved layouts.
  if (isa<gluon::AutoEncodingAttr>(srcTy.getEncoding()))
    return false;
  if (isa<gluon::AutoEncodingAttr>(dstTy.getEncoding()))
    return false;

  // Check concrete layouts.
  triton::LinearLayout cvt = minimalCvtLayout(srcTy, dstTy);
  auto dims = llvm::to_vector(cvt.getInDimNames());
  return dims.empty() || (dims.size() == 1 && dims.front() == "register");
}

template <typename R>
std::vector<llvm::ValueTypeFromRangeType<R>> toStdVector(R &&range) {
  return {range.begin(), range.end()};
}

py::object layoutToGluon(Attribute layout) {
  static GluonLayouts layouts;
  if (auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(blocked.getCGALayout());
    return layouts.BlockedLayout(toStdVector(blocked.getSizePerThread()),
                                 toStdVector(blocked.getThreadsPerWarp()),
                                 toStdVector(blocked.getWarpsPerCTA()),
                                 toStdVector(blocked.getOrder()), cgaBases);
  } else if (auto sliced = dyn_cast<ttg::SliceEncodingAttr>(layout)) {
    return layouts.SliceLayout(sliced.getDim(),
                               layoutToGluon(sliced.getParent()));
  } else if (auto linear = dyn_cast<ttg::LinearEncodingAttr>(layout)) {
    const auto &ll = linear.getLinearLayout();
    auto ctx = layout.getContext();
    auto kReg = mlir::StringAttr::get(ctx, "register");
    auto kLane = mlir::StringAttr::get(ctx, "lane");
    auto kWarp = mlir::StringAttr::get(ctx, "warp");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    return layouts.DistributedLinearLayout(
        ll.getBases().lookup(kReg), ll.getBases().lookup(kLane),
        ll.getBases().lookup(kWarp), ll.getBases().lookup(kBlock),
        toStdVector(ll.getOutDimSizes()));
  } else if (auto dotOp = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    return layouts.DotOperandLayout(
        dotOp.getOpIdx(), layoutToGluon(dotOp.getParent()), dotOp.getKWidth());
  } else if (auto mma = dyn_cast<ttg::NvidiaMmaEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(mma.getCGALayout());
    return layouts.NVMMADistributedLayout(
        std::vector<unsigned>{mma.getVersionMajor(), mma.getVersionMinor()},
        toStdVector(mma.getWarpsPerCTA()), toStdVector(mma.getInstrShape()),
        cgaBases);
  } else if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(layout)) {
    auto cgaLayout = nvmma.getCGALayout();
    auto cgaBases = getCgaLayoutBases(cgaLayout);
    return layouts.NVMMASharedLayout(nvmma.getSwizzlingByteWidth(),
                                     nvmma.getElementBitWidth(),
                                     cgaLayout.getRank(), nvmma.getTransposed(),
                                     nvmma.getFp4Padded(), cgaBases);
  } else if (auto swizzled =
                 dyn_cast<ttg::SwizzledSharedEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(swizzled.getCGALayout());
    return layouts.SwizzledSharedLayout(
        swizzled.getVec(), swizzled.getPerPhase(), swizzled.getMaxPhase(),
        toStdVector(swizzled.getOrder()), cgaBases);
  } else if (auto sharedLl = dyn_cast<ttg::SharedLinearEncodingAttr>(layout)) {
    const auto &ll = sharedLl.getLinearLayout();
    auto ctx = layout.getContext();
    auto kOffset = mlir::StringAttr::get(ctx, "offset");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    return layouts.SharedLinearLayout(
        toStdVector(ll.getBases().lookup(kOffset)),
        toStdVector(ll.getBases().lookup(kBlock)), sharedLl.getAlignment());
  } else if (auto autoEnc = dyn_cast<gluon::AutoEncodingAttr>(layout)) {
    return layouts.AutoLayout();
  } else if (auto autoEnc = dyn_cast<gluon::CoalescedEncodingAttr>(layout)) {
    return layouts.CoalescedLayout();
  } else if (auto amdMfma = dyn_cast<ttg::AMDMfmaEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(amdMfma.getCGALayout());
    return layouts.AMDMFMALayout(
        amdMfma.getVersion(), toStdVector(amdMfma.getInstrShape()),
        amdMfma.getIsTransposed(), toStdVector(amdMfma.getWarpsPerCTA()),
        amdMfma.getElementBitWidth(), toStdVector(amdMfma.getTilesPerWarp()),
        cgaBases);
  } else if (auto amdWmma = dyn_cast<ttg::AMDWmmaEncodingAttr>(layout)) {
    auto cgaBases = getCgaLayoutBases(amdWmma.getCGALayout());
    const auto &ctaLayout = amdWmma.getCtaLayout();
    auto ctx = layout.getContext();
    auto kReg = mlir::StringAttr::get(ctx, "register");
    auto kWarp = mlir::StringAttr::get(ctx, "warp");
    return layouts.AMDWMMALayout(
        amdWmma.getVersion(), amdWmma.getIsTransposed(),
        ctaLayout.getBases().lookup(kWarp), ctaLayout.getBases().lookup(kReg),
        toStdVector(amdWmma.getInstrShape()), cgaBases, amdWmma.getRank());
  } else if (auto paddedShared =
                 dyn_cast<ttg::PaddedSharedEncodingAttr>(layout)) {
    auto *ctx = paddedShared.getContext();
    std::vector<std::pair<unsigned, unsigned>> intervalPaddingPairs;
    for (auto [interval, padding] :
         llvm::zip(paddedShared.getIntervals(), paddedShared.getPaddings())) {
      intervalPaddingPairs.push_back({interval, padding});
    }
    auto kOffset = mlir::StringAttr::get(ctx, "offset");
    auto kBlock = mlir::StringAttr::get(ctx, "block");
    const auto &ll = paddedShared.getLinearComponent();
    auto shape = toStdVector(ll.getOutDimSizes());
    return layouts.PaddedSharedLayout(intervalPaddingPairs,
                                      ll.getBases().lookup(kOffset),
                                      ll.getBases().lookup(kBlock), shape);
  } else if (auto partitioned =
                 dyn_cast<ttg::PartitionedSharedEncodingAttr>(layout)) {
    py::object partitionLayout =
        layoutToGluon(partitioned.getPartitionLayout());
    return layouts.PartitionedSharedLayout(
        partitioned.getNumPartitions(), partitioned.getNumGroups(),
        partitioned.getPartitionDim(), partitionLayout);
  } else if (auto tmemScales =
                 dyn_cast<ttng::TensorMemoryScalesEncodingAttr>(layout)) {
    return layouts.TensorMemoryScalesLayout(
        getCgaLayoutBases(tmemScales.getCGALayout()));
  } else if (auto tmemLinear =
                 dyn_cast<ttng::TensorMemoryLinearEncodingAttr>(layout)) {
    auto ll = tmemLinear.getLinearLayout();
    auto ctx = layout.getContext();
    auto bases = ll.getBases();
    auto rowBases = bases[mlir::StringAttr::get(ctx, "row")];
    auto colBases = bases[mlir::StringAttr::get(ctx, "col")];
    auto blockBases = bases[mlir::StringAttr::get(ctx, "block")];
    auto outDims = ll.getOutDims();
    std::vector<int64_t> shape;
    shape.reserve(outDims.size());
    for (auto &od : outDims)
      shape.push_back(od.second);
    return layouts.TensorMemoryLinearLayout(rowBases, colBases, shape,
                                            blockBases,
                                            tmemLinear.getTwoCTAs());
  } else if (auto tmem = dyn_cast<ttng::TensorMemoryEncodingAttr>(layout)) {
    return layouts.TensorMemoryLayout(
        std::vector<unsigned>{tmem.getBlockM(), tmem.getBlockN()},
        tmem.getColStride(), getCgaLayoutBases(tmem.getCGALayout()),
        tmem.getTwoCTAs());
  }

  throw py::value_error("Unhandled encoding encountered");
}

template <typename CondT> static void check(CondT &&cond, const char *msg) {
  if (!std::forward<CondT>(cond))
    throw py::value_error(msg);
}

void init_gluon_ir(py::module &&m) {
  using ret = py::return_value_policy;

  py::enum_<ttng::TMEMLoadReduceModifier>(m, "TMEM_LOAD_REDUCE_MODIFIER",
                                          py::module_local())
      .value("MIN", ttng::TMEMLoadReduceModifier::MIN)
      .value("MAX", ttng::TMEMLoadReduceModifier::MAX)
      .export_values();

  py::class_<GluonOpBuilder, TritonOpBuilder>(
      m, "GluonOpBuilder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *>())
      .def("get_op_builder", &GluonOpBuilder::getBuilder, ret::reference)
      .def("get_distributed_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout) -> Type {
             return self.getChecked<RankedTensorType>(shape, elementType,
                                                      layout);
           })
      .def("get_shared_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return self.getChecked<ttg::MemDescType>(
                 shape, elementType, layout,
                 ttg::SharedMemorySpaceAttr::get(ctx),
                 /*mutableMemory=*/true,
                 /*allocShape=*/allocShape);
           })
      .def("get_tensor_mem_desc_ty",
           [](GluonOpBuilder &self, Type &elementType,
              std::vector<int64_t> &shape, Attribute layout,
              std::vector<int64_t> &allocShape) -> Type {
             auto ctx = self.getContext();
             return self.getChecked<ttg::MemDescType>(
                 shape, elementType, layout,
                 ttng::TensorMemorySpaceAttr::get(ctx),
                 /*mutableMemory=*/true,
                 /*allocShape=*/allocShape);
           })
      .def("get_tmem_alloc_shape",
           [](GluonOpBuilder &self, std::vector<int64_t> &shape,
              Attribute layout) -> std::vector<int64_t> {
             std::string error;
             auto maybeAllocShape =
                 ttng::getTMemAllocShapeForEncoding(shape, layout, &error);
             check(maybeAllocShape, error.c_str());
             return std::vector<int64_t>(maybeAllocShape->begin(),
                                         maybeAllocShape->end());
           })
      .def("get_blocked_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &sizePerThread,
              std::vector<unsigned> &threadsPerWarp,
              std::vector<unsigned> &warpsPerCta, std::vector<unsigned> &order,
              std::vector<std::vector<int32_t>> &cgaBases) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = order.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::BlockedEncodingAttr>(
                 ctx, sizePerThread, threadsPerWarp, warpsPerCta, order,
                 cgaLayout);
           })
      .def("get_slice_layout",
           [](GluonOpBuilder &self, unsigned dim,
              Attribute parent) -> Attribute {
             auto ctx = self.getContext();
             auto dist = cast<ttg::DistributedEncodingTrait>(parent);
             return self.getChecked<ttg::SliceEncodingAttr>(ctx, dim, dist);
           })
      .def("get_distributed_linear_layout",
           [](GluonOpBuilder &self, std::vector<std::vector<int>> regBases,
              std::vector<std::vector<int>> laneBases,
              std::vector<std::vector<int>> warpBases,
              std::vector<std::vector<int>> blockBases,
              std::vector<int64_t> shape) -> Attribute {
             auto ctx = self.getContext();
             auto kReg = mlir::StringAttr::get(ctx, "register");
             auto kLane = mlir::StringAttr::get(ctx, "lane");
             auto kWarp = mlir::StringAttr::get(ctx, "warp");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto outDims = tt::standardOutDimPairs(ctx, shape);
             tt::LinearLayout::BasesT bases;
             bases[kReg] = std::move(regBases);
             bases[kLane] = std::move(laneBases);
             bases[kWarp] = std::move(warpBases);
             bases[kBlock] = std::move(blockBases);
             std::string error;
             auto maybeLL = tt::LinearLayout::tryCreate(
                 std::move(bases), outDims, /*requireSurjective=*/true, &error);
             if (!maybeLL)
               throw py::value_error(error);
             return ttg::LinearEncodingAttr::get(ctx, std::move(*maybeLL));
           })
      .def("to_linear_layout",
           [](GluonOpBuilder &self, Attribute layout,
              std::vector<int64_t> &shape) -> py::object {
             auto ctx = self.getContext();

             if (ttng::isTensorMemoryEncoding(layout)) {
               std::string error;
               auto maybeLinear =
                   ttng::getCanonicalTMemLinearEncoding(shape, layout, &error);
               if (!maybeLinear)
                 throw std::runtime_error(error);
               return layoutToGluon(*maybeLinear);
             }

             auto linearLayout = ttg::toLinearLayout(shape, layout);

             if (isa<ttg::DistributedEncodingTrait>(layout)) {
               auto attr =
                   ttg::LinearEncodingAttr::get(ctx, std::move(linearLayout));
               return layoutToGluon(attr);
             }
             if (isa<ttg::SharedEncodingTrait>(layout)) {
               auto alignment =
                   cast<ttg::SharedEncodingTrait>(layout).getAlignment();
               auto attr = ttg::SharedLinearEncodingAttr::get(
                   ctx, std::move(linearLayout), alignment);
               return layoutToGluon(attr);
             }
             throw std::invalid_argument("Unsupported layout in to_linear_layout");
           })
      .def("get_dot_operand_layout",
           [](GluonOpBuilder &self, unsigned opIdx, Attribute parent,
              unsigned kWidth) -> Attribute {
             return self.getChecked<ttg::DotOperandEncodingAttr>(
                 self.getContext(), opIdx, parent, kWidth);
           })
      .def("get_mma_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &version,
              std::vector<unsigned> &warpsPerCta,
              std::vector<std::vector<int32_t>> &cgaBases,
              std::vector<unsigned> &instrShape) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = warpsPerCta.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::NvidiaMmaEncodingAttr>(
                 ctx, version[0], version[1], warpsPerCta, cgaLayout,
                 instrShape);
           })
      .def("get_amd_mfma_layout",
           [](GluonOpBuilder &self, unsigned version,
              std::vector<unsigned> &warpsPerCta,
              std::vector<unsigned> &instrShape, bool transposed,
              std::vector<std::vector<int32_t>> &cgaBases,
              std::vector<unsigned> &tilesPerWarp,
              unsigned elementBitWidth) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = warpsPerCta.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return ttg::AMDMfmaEncodingAttr::get(
                 ctx, version, warpsPerCta, instrShape, transposed, cgaLayout,
                 tilesPerWarp, elementBitWidth);
           })
      .def("get_amd_wmma_layout",
           [](GluonOpBuilder &self, unsigned version, bool transposed,
              std::vector<std::vector<int32_t>> &warpBases,
              std::vector<std::vector<int32_t>> &regBases,
              std::vector<std::vector<int32_t>> &cgaBases,
              std::vector<unsigned> &instrShape, unsigned rank) -> Attribute {
             auto ctx = self.getContext();
             auto kReg = mlir::StringAttr::get(ctx, "register");
             auto kWarp = mlir::StringAttr::get(ctx, "warp");
             auto ctaLayout =
                 tt::LinearLayout({{kReg, regBases}, {kWarp, warpBases}},
                                  tt::standardOutDimNames(ctx, rank));
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return ttg::AMDWmmaEncodingAttr::get(
                 ctx, version, ctaLayout, transposed, cgaLayout, instrShape);
           })
      .def("get_padded_shared_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &intervals,
              std::vector<unsigned> &paddings,
              std::vector<std::vector<int>> &offsetBases,
              std::vector<std::vector<int>> &blockBases,
              std::vector<int64_t> &shape) -> Attribute {
             auto ctx = self.getContext();
             auto rank = shape.size();
             auto kOffset = mlir::StringAttr::get(ctx, "offset");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto outDims = tt::standardOutDimNames(ctx, rank);
             auto ll = tt::LinearLayout({{kOffset, offsetBases}}, outDims) *
                       tt::LinearLayout({{kBlock, blockBases}}, outDims);
             return ttg::PaddedSharedEncodingAttr::get(ctx, intervals, paddings,
                                                       std::move(ll));
           })
      .def("get_shared_linear_layout",
           [](GluonOpBuilder &self, std::vector<std::vector<int>> &offsetBases,
              std::vector<std::vector<int>> &blockBases,
              unsigned alignment) -> Attribute {
             auto ctx = self.getContext();
             auto kOffset = mlir::StringAttr::get(ctx, "offset");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto outDims = tt::standardOutDimNames(ctx, offsetBases[0].size());
             auto ll = tt::LinearLayout(
                 {{kOffset, offsetBases}, {kBlock, blockBases}}, outDims);
             return self.getChecked<ttg::SharedLinearEncodingAttr>(
                 ctx, std::move(ll), alignment);
           })
      .def("get_nvmma_shared_layout",
           [](GluonOpBuilder &self, unsigned swizzleByteWidth,
              unsigned elementBitwidth, bool transposed, bool fp4Padded,
              std::vector<std::vector<int32_t>> &cgaBases,
              unsigned rank) -> Attribute {
             auto ctx = self.getContext();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::NVMMASharedEncodingAttr>(
                 ctx, swizzleByteWidth, transposed, elementBitwidth, fp4Padded,
                 cgaLayout);
           })
      .def("get_auto_layout",
           [](GluonOpBuilder &self) -> Attribute {
             return self.getChecked<gluon::AutoEncodingAttr>(self.getContext());
           })
      .def("get_coalesced_layout",
           [](GluonOpBuilder &self) -> Attribute {
             return self.getChecked<gluon::CoalescedEncodingAttr>(
                 self.getContext());
           })
      .def("get_swizzled_shared_layout",
           [](GluonOpBuilder &self, int vec, int perPhase, int maxPhase,
              std::vector<unsigned> &order,
              std::vector<std::vector<int32_t>> &cgaBases) -> Attribute {
             auto ctx = self.getContext();
             unsigned rank = order.size();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, rank);
             return self.getChecked<ttg::SwizzledSharedEncodingAttr>(
                 ctx, vec, perPhase, maxPhase, order, cgaLayout);
           })
      .def("get_partitioned_shared_layout",
           [](GluonOpBuilder &self, unsigned numPartitions, unsigned numGroups,
              unsigned partitionDim, Attribute partitionLayout) -> Attribute {
             auto ctx = self.getContext();
             auto sharedLayout =
                 cast<ttg::SharedEncodingTrait>(partitionLayout);
             return self.getChecked<ttg::PartitionedSharedEncodingAttr>(
                 ctx, numPartitions, numGroups, partitionDim, sharedLayout);
           })
      .def("get_tensor_memory_layout",
           [](GluonOpBuilder &self, std::vector<unsigned> &block,
              unsigned colStride, std::vector<std::vector<int32_t>> &cgaBases,
              bool twoCTAs) -> Attribute {
             auto ctx = self.getContext();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, /*rank=*/2);
             return self.getChecked<ttng::TensorMemoryEncodingAttr>(
                 ctx, block[0], block[1], colStride, cgaLayout, twoCTAs);
           })
      .def("get_tensor_memory_linear_layout",
           [](GluonOpBuilder &self, std::vector<std::vector<int32_t>> &rowBases,
              std::vector<std::vector<int32_t>> &colBases,
              std::vector<std::vector<int32_t>> &blockBases,
              std::vector<int64_t> &shape, bool twoCTAs) -> Attribute {
             auto ctx = self.getContext();
             auto kRow = mlir::StringAttr::get(ctx, "row");
             auto kCol = mlir::StringAttr::get(ctx, "col");
             auto kBlock = mlir::StringAttr::get(ctx, "block");
             auto outDims = tt::standardOutDimPairs(ctx, shape);
             tt::LinearLayout::BasesT bases;
             bases[kRow] = rowBases;
             bases[kCol] = colBases;
             if (!blockBases.empty())
               bases[kBlock] = blockBases;
             auto ll = tt::LinearLayout(std::move(bases), outDims,
                                        /*requiresSurjective=*/false);
             return self.getChecked<ttng::TensorMemoryLinearEncodingAttr>(
                 ctx, std::move(ll), twoCTAs);
           })
      .def("get_tensor_memory_scales_layout",
           [](GluonOpBuilder &self,
              std::vector<std::vector<int32_t>> &cgaBases) -> Attribute {
             auto ctx = self.getContext();
             auto cgaLayout = buildCgaLayoutAttr(ctx, cgaBases, /*rank=*/2);
             return self.getChecked<ttng::TensorMemoryScalesEncodingAttr>(
                 ctx, cgaLayout);
           })
      .def("get_shape_from_tensor",
           [](GluonOpBuilder &self, Value tensor) -> std::vector<int64_t> {
             auto ty = dyn_cast<RankedTensorType>(tensor.getType());
             return ty.getShape();
           })
      .def("get_gluon_layout_from_tensor",
           [](GluonOpBuilder &self, Value tensor) -> py::object {
             auto ty = dyn_cast<RankedTensorType>(tensor.getType());
             check(ty.getEncoding(), "expected a tensor with an encoding");
             return layoutToGluon(ty.getEncoding());
           })
      .def("get_gluon_layout_from_memdesc",
           [](GluonOpBuilder &self, Value memdesc) -> py::object {
             auto ty = dyn_cast<ttg::MemDescType>(memdesc.getType());
             check(ty.getEncoding(), "expected a memdesc with an encoding");
             return layoutToGluon(ty.getEncoding());
           })
      .def("get_tensor_descriptor_layout_type",
           [](GluonOpBuilder &self, Type blockType, bool isSigned,
              Attribute layout) -> Type {
             auto ctx = self.getContext();
             auto blockTy = cast<RankedTensorType>(blockType);
             auto blockTyLayout = blockTy.cloneWithEncoding(layout);
             return triton::TensorDescType::get(ctx, blockTyLayout, isSigned);
           })
      .def("get_tensor_descriptor_im2col_layout_type",
           [](GluonOpBuilder &self, Type blockType, bool isSigned,
              Attribute layout) -> Type {
             auto ctx = self.getContext();
             auto blockTy = cast<RankedTensorType>(blockType);
             auto blockTyLayout = blockTy.cloneWithEncoding(layout);
             return triton::nvidia_gpu::TensorDescIm2ColType::get(
                 ctx, blockTyLayout);
           })
      .def("is_convert_layout_trivial",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> bool {
             auto dstTy = cast<RankedTensorType>(resultTy);
             return isConvertLayoutTrivial(dstTy, value);
           })
      .def("create_histogram",
           [](GluonOpBuilder &self, Value operand, int numBins,
              std::optional<Value> mask, Attribute layout) -> Value {
             auto *ctx = self.getContext();
             auto resultTy =
                 RankedTensorType::get({static_cast<int64_t>(numBins)},
                                       IntegerType::get(ctx, 32), layout);
             if (!mask) {
               return self.create<triton::HistogramOp>(resultTy, operand);
             } else {
               return self.create<triton::HistogramOp>(resultTy, operand,
                                                       *mask);
             }
           })
      .def("create_cat",
           [](GluonOpBuilder &self, Value &lhs, Value &rhs,
              Type retType) -> Value {
             return self.create<triton::CatOp>(retType, lhs, rhs);
           })
      .def("create_fp4_to_fp",
           [](GluonOpBuilder &self, Value src, Type elemType,
              int axis) -> Value {
             return self.create<ttg::Fp4ToFpOp>(
                 cast<TypedValue<RankedTensorType>>(src), elemType, axis);
           })
      .def("create_async_copy_global_to_local",
           [](GluonOpBuilder &self, Value smem, Value pointer, Value mask,
              Value other, tt::CacheModifier cacheModifier,
              tt::EvictionPolicy evictionPolicy, bool isVolatile) {
             self.create<ttg::AsyncCopyGlobalToLocalOp>(
                 pointer, smem, mask, other, cacheModifier, evictionPolicy,
                 isVolatile);
           })
      .def("create_async_copy_local_to_global",
           [](GluonOpBuilder &self, Value smem, Value pointer, Value mask,
              tt::CacheModifier cacheModifier,
              tt::EvictionPolicy evictionPolicy) {
             self.create<ttag::AsyncCopyLocalToGlobalOp>(
                 smem, pointer, mask, cacheModifier, evictionPolicy);
           })
      .def("create_async_copy_mbarrier_arrive",
           [](GluonOpBuilder &self, Value mbarrier, bool incrementCount) {
             self.create<ttng::AsyncCopyMbarrierArriveOp>(mbarrier,
                                                          !incrementCount);
           })
      .def("create_async_commit_group",
           [](GluonOpBuilder &self) {
             ValueRange tokens;
             self.create<ttg::AsyncCommitGroupOp>(tokens);
           })
      .def("create_async_wait_group",
           [](GluonOpBuilder &self, int num) {
             ValueRange tokens;
             self.create<ttg::AsyncWaitOp>(tokens, num);
           })
      .def("create_convert_layout",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::ConvertLayoutOp>(resultTy, value);
           })
      .def("create_local_alloc",
           [](GluonOpBuilder &self, Type resultTy) -> Value {
             return self.create<ttg::LocalAllocOp>(resultTy);
           })
      .def("create_local_alloc",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> Value {
             return self.create<ttg::LocalAllocOp>(resultTy, value);
           })
      .def("create_local_store",
           [](GluonOpBuilder &self, Value memDesc, Value value) {
             self.create<ttg::LocalStoreOp>(value, memDesc);
           })
      .def("create_local_load",
           [](GluonOpBuilder &self, Type resultTy, Value memDesc) -> Value {
             return self.create<ttg::LocalLoadOp>(resultTy, memDesc);
           })
      .def("create_local_gather",
           [](GluonOpBuilder &self, Type resultTy, Value memDesc, Value indices,
              int32_t axis) -> Value {
             auto ctx = self.getContext();
             auto i32Ty = IntegerType::get(ctx, 32);
             auto axisAttr = IntegerAttr::get(i32Ty, axis);
             return self.create<ttg::LocalGatherOp>(resultTy, memDesc, indices,
                                                    axisAttr);
           })
      .def("create_local_scatter",
           [](GluonOpBuilder &self, Value memDesc, Value values, Value indices,
              int32_t axis) {
             auto ctx = self.getContext();
             auto i32Ty = IntegerType::get(ctx, 32);
             auto axisAttr = IntegerAttr::get(i32Ty, axis);
             self.create<ttg::LocalScatterOp>(memDesc, values, indices,
                                              axisAttr);
           })
      .def("get_shared_bank_conflicts",
           [](GluonOpBuilder &self, Attribute regLayoutAttr,
              Attribute sharedLayoutAttr, std::vector<int64_t> &shape,
              int bitwidth) -> int {
             auto regLayout = ttg::toLinearLayout(shape, regLayoutAttr);
             auto smemLayout = ttg::toLinearLayout(shape, sharedLayoutAttr);
             return ttg::bankConflictsMemDesc(regLayout, smemLayout, bitwidth);
           })
      .def("create_local_dealloc",
           [](GluonOpBuilder &self, Value memDesc) -> Operation * {
             return self.create<ttg::LocalDeallocOp>(memDesc);
           })

      .def("create_memdesc_index",
           [](GluonOpBuilder &self, Type resultType, Value src,
              Value index) -> Value {
             return self.create<ttg::MemDescIndexOp>(resultType, src, index);
           })
      .def("create_memdesc_index",
           [](GluonOpBuilder &self, Value src, Value index) -> Value {
             auto op = createCheckedOrThrow(
                 self, "failed to infer memdesc_index result type",
                 [&] {
                   return ttg::MemDescIndexOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src, index);
                 });
             return op.getResult();
           })
      .def("create_memdesc_subslice",
           [](GluonOpBuilder &self, Type resultType, Value src,
              std::vector<int32_t> &offsets) -> Value {
             return self.create<ttg::MemDescSubsliceOp>(resultType, src,
                                                        offsets);
           })
      .def("create_memdesc_subslice",
           [](GluonOpBuilder &self, Value src, std::vector<int64_t> &shape,
              std::vector<int32_t> &offsets) -> Value {
             auto op = createCheckedOrThrow(
                 self, "failed to infer memdesc_subslice result type",
                 [&] {
                   return ttg::MemDescSubsliceOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src, shape,
                       offsets);
                 });
             return op.getResult();
           })
      .def("create_memdesc_trans",
           [](GluonOpBuilder &self, Value src,
              std::vector<int> &order) -> Value {
             SmallVector<int32_t> orderAttr(order.begin(), order.end());
             auto op = createCheckedOrThrow(
                 self, "failed to infer memdesc_trans result type",
                 [&] {
                   return ttg::MemDescTransOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src, orderAttr);
                 });
             return op.getResult();
           })
      .def("create_memdesc_reshape",
           [](GluonOpBuilder &self, Value src,
              std::vector<int64_t> &shape) -> Value {
             auto op = createCheckedOrThrow(
                 self, "failed to infer memdesc_reshape result type",
                 [&] {
                   return ttg::MemDescReshapeOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src, shape);
                 });
             return op.getResult();
           })
      .def("create_memdesc_reinterpret",
           [](GluonOpBuilder &self, Type resultType, Value src) -> Value {
             auto op = createCheckedOrThrow(
                 self, "failed to infer memdesc_reinterpret result type",
                 [&] {
                   return ttg::MemDescReinterpretOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src,
                       cast<ttg::MemDescType>(resultType));
                 });
             return op.getResult();
           })
      .def("create_tmem_memdesc_bitcast",
           [](GluonOpBuilder &self, Value src, Type dstElementType,
              std::vector<int64_t> &dstShape) -> Value {
             std::string error;
             auto maybeResultTy = ttng::inferTMemBitcastType(
                 src, dstShape, dstElementType, &error);
             if (failed(maybeResultTy)) {
               if (error.empty())
                 error = "failed to infer tensor memory bitcast result type";
               throw py::value_error(error.c_str());
             }
             auto resultTy = *maybeResultTy;
             auto op = createCheckedOrThrow(
                 self, "failed to create tensor memory bitcast", [&] {
                   return ttg::MemDescReinterpretOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src, resultTy);
                 });
             op->setAttr("tmem_physical_bitcast",
                         mlir::UnitAttr::get(self.getBuilder().getContext()));
             return op.getResult();
           })
      .def("create_tmem_memdesc_bitcast_with_layout",
           [](GluonOpBuilder &self, Type resultType, Value src) -> Value {
             auto resultTy = cast<ttg::MemDescType>(resultType);
             std::string error;
             auto maybeInferredTy = ttng::inferTMemBitcastType(
                 src, resultTy.getShape(), resultTy.getElementType(), &error);
             if (failed(maybeInferredTy)) {
               if (error.empty())
                 error = "failed to infer tensor memory bitcast result type";
               throw py::value_error(error.c_str());
             }
             auto op = createCheckedOrThrow(
                 self, "failed to create tensor memory bitcast", [&] {
                   return ttg::MemDescReinterpretOp::createChecked(
                       self.getBuilder(), self.getLastLoc(), src, resultTy);
                 });
             op->setAttr("tmem_physical_bitcast",
                         mlir::UnitAttr::get(self.getBuilder().getContext()));
             return op.getResult();
           })
      .def("create_set_auto_layout",
           [](GluonOpBuilder &self, Attribute layout, Value value) -> Value {
             return self.create<gluon::SetAutoLayoutOp>(layout, value);
           })
      .def("create_split",
           [](GluonOpBuilder &self, Value &a) -> py::tuple {
             auto argTy = cast<RankedTensorType>(a.getType());
             auto ctx = argTy.getContext();
             auto enc = ttg::SliceEncodingAttr::get(
                 ctx, argTy.getRank() - 1,
                 cast<ttg::DistributedEncodingTrait>(argTy.getEncoding()));
             auto resTy =
                 RankedTensorType::get(ArrayRef(argTy.getShape()).drop_back(),
                                       argTy.getElementType(), enc);
             auto op = self.create<triton::SplitOp>(TypeRange{resTy, resTy}, a);
             return py::make_tuple(op->getResult(0), op->getResult(1));
           })
      .def("create_warpgroup_mma",
           [](GluonOpBuilder &self, Value a, Value b, Value acc, Value useAcc,
              triton::InputPrecision precision = triton::InputPrecision::IEEE,
              int maxNumImpreciseAcc = 0, bool isAsync = false) -> Value {
             return self.create<ttng::WarpGroupDotOp>(
                 a, b, acc, useAcc, precision, maxNumImpreciseAcc, isAsync);
           })
      .def("create_warpgroup_mma_wait",
           [](GluonOpBuilder &self, std::vector<Value> &deps, int pendings) {
             std::vector<Value> results;
             auto wait = self.create<ttng::WarpGroupDotWaitOp>(deps, pendings);
             llvm::append_range(results, wait.getResults());
             return results;
           })
      .def("create_tmem_alloc",
           [](GluonOpBuilder &self, Type resultTy, Value value) -> Value {
             auto op = self.create<ttng::TMEMAllocOp>(resultTy, value);
             return op.getResult();
           })
      .def("create_tmem_alloc",
           [](GluonOpBuilder &self, Type resultTy, py::none value) -> Value {
             auto op = self.create<ttng::TMEMAllocOp>(resultTy, Value{});
             return op.getResult();
           })
      .def("create_tmem_store",
           [](GluonOpBuilder &self, Value memDesc, Value value, Value pred) {
             self.create<ttng::TMEMStoreOp>(memDesc, value, pred);
           })
      .def(
          "create_tmem_load",
          [](GluonOpBuilder &self, Type resultTy, Value memDesc,
             std::optional<ttng::TMEMLoadReduceModifier> redOp, bool useAbs,
             tt::PropagateNan propagateNan, unsigned numWarps) -> py::object {
            ttng::TMEMLoadReduceModifierAttr redOpAttr = nullptr;
            BoolAttr absAttr = nullptr;
            BoolAttr nanAttr = nullptr;

            if (redOp) {
              if (auto rankedTy = dyn_cast<RankedTensorType>(resultTy))
                resultTy = ttng::canonicalizeTMemLoadReductionType(
                    rankedTy, memDesc, numWarps);
              redOpAttr = ttng::TMEMLoadReduceModifierAttr::get(
                  self.getContext(), redOp.value());
              if (useAbs)
                absAttr = self.getBuilder().getBoolAttr(true);
              if (propagateNan != tt::PropagateNan::NONE)
                nanAttr = self.getBuilder().getBoolAttr(true);
            }

            auto op = self.create<ttng::TMEMLoadOp>(
                resultTy, /*token=*/Type(), memDesc, /*dep=*/Value(), redOpAttr,
                absAttr, nanAttr);

            if (redOp) {
              Value result = op.getResult();
              Value red = op.getRed();
              auto redTy = cast<RankedTensorType>(red.getType());
              py::object redLayout = layoutToGluon(redTy.getEncoding());
              return py::make_tuple(result, red, redLayout);
            }
            Value result = op.getResult();
            return py::cast(result);
          },
          py::arg("resultTy"), py::arg("memDesc"),
          py::arg("redOp") = py::none(), py::arg("useAbs") = false,
          py::arg("propagateNan") = tt::PropagateNan::NONE,
          py::arg("numWarps") = 0)
      .def("create_tmem_copy",
           [](GluonOpBuilder &self, Value src, Value dst) {
             self.create<ttng::TMEMCopyOp>(src, dst, /*barrier=*/Value());
           })
      .def("create_tmem_subslice",
           [](GluonOpBuilder &self, Type resultTy, Value memDesc,
              int N) -> Value {
             return self.create<ttng::TMEMSubSliceOp>(resultTy, memDesc, N);
           })
      .def("create_mbarrier_init",
           [](GluonOpBuilder &self, Value memDesc, int count) {
             self.create<ttng::InitBarrierOp>(memDesc, count);
           })
      .def("create_mbarrier_inval",
           [](GluonOpBuilder &self, Value memDesc) {
             self.create<ttng::InvalBarrierOp>(memDesc);
           })
      .def("create_mbarrier_expect",
           [](GluonOpBuilder &self, Value memDesc, int bytes, Value pred) {
             self.create<ttng::BarrierExpectOp>(memDesc, bytes, pred);
           })
      .def("create_mbarrier_wait",
           [](GluonOpBuilder &self, Value memDesc, Value phase, Value pred,
              std::vector<Value> &deps) {
             self.create<ttng::WaitBarrierOp>(memDesc, phase, pred, deps);
           })
      .def("create_mbarrier_arrive",
           [](GluonOpBuilder &self, Value memDesc, int count, Value pred) {
             self.create<ttng::ArriveBarrierOp>(memDesc, count, pred);
           })
      .def("create_fence_mbarrier_init_release_cluster",
           [](GluonOpBuilder &self) {
             self.create<ttng::FenceMBarrierInitReleaseClusterOp>();
           })
      .def(
          "create_cluster_arrive",
          [](GluonOpBuilder &self,
             bool relaxed) { self.create<ttng::ClusterArriveOp>(relaxed); },
          py::arg("relaxed") = false)
      .def("create_cluster_wait",
           [](GluonOpBuilder &self) { self.create<ttng::ClusterWaitOp>(); })
      .def(
          "create_cluster_barrier",
          [](GluonOpBuilder &self,
             bool relaxed) { self.create<ttng::ClusterBarrierOp>(relaxed); },
          py::arg("relaxed") = false)
      // CLC (Cluster Launch Control) ops - SM100+
      .def("create_clc_try_cancel",
           [](GluonOpBuilder &self, Value result, Value mbarrier,
              bool multicast) {
             self.create<ttng::CLCTryCancelOp>(result, mbarrier, multicast);
           })
      .def("create_clc_load_result",
           [](GluonOpBuilder &self, Value result) -> Value {
             auto i64Ty = self.getBuilder().getI64Type();
             return self.create<ttng::CLCLoadResultOp>(result);
           })
      .def("create_clc_is_canceled",
           [](GluonOpBuilder &self, Value clcResult) -> Value {
             auto i1Ty = self.getBuilder().getI1Type();
             return self.create<ttng::CLCIsCanceledOp>(clcResult);
           })
      .def("create_clc_get_program_id",
           [](GluonOpBuilder &self, Value clcResult, int dim) -> Value {
             auto i32Ty = self.getBuilder().getI32Type();
             return self.create<ttng::CLCGetProgramIdOp>(clcResult, dim);
           })
      .def("create_tcgen05_mma",
           [](GluonOpBuilder &self, Value a, Value b, Value acc, Value useAcc,
             Value pred, std::vector<Value> &mbarriers,
             std::vector<Value> &mbarrier_preds, bool two_ctas,
             bool multicast) {
             Value accDep;
             auto tokType = self.getBuilder().getType<ttg::AsyncTokenType>();
             self.create<ttng::TCGen5MMAOp>(tokType, a, b, acc, accDep, useAcc,
                                            pred, two_ctas, multicast,
                                            mbarriers, mbarrier_preds);
           })
      .def("create_tcgen05_mma_scaled",
           [](GluonOpBuilder &self, Value a, Value b, Value acc, Value aScale,
              Value bScale, tt::ScaleDotElemType aType,
              tt::ScaleDotElemType bType, Value useAcc, Value pred,
              std::vector<Value> &mbarriers, std::vector<Value> &mbarrier_preds,
              bool two_ctas) {
             Value accDep;
             auto tokType = self.getBuilder().getType<ttg::AsyncTokenType>();
             self.create<ttng::TCGen5MMAScaledOp>(
                 tokType, a, b, acc, accDep, aScale, bScale, aType, bType,
                 useAcc, pred, mbarriers, mbarrier_preds, two_ctas);
           })
      .def("create_tcgen05_commit",
           [](GluonOpBuilder &self, Value &barrier, Value &pred,
              std::vector<Value> &descs) {
             self.create<ttng::TCGen5CommitOp>(barrier, pred, descs);
           })

      .def(
          "create_async_tma_copy_global_to_local",
          [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &coord,
             Value barrier, Value result, Value pred, bool multicast,
             std::optional<std::vector<Value>> offsets) {
            multicast &=
                ttng::hasCGABroadcast(cast<ttg::MemDescType>(result.getType()));
            ValueRange offsetsRange =
                offsets.has_value() ? ValueRange(*offsets) : ValueRange{};
            self.create<ttng::AsyncTMACopyGlobalToLocalOp>(
                descPtr, coord, offsetsRange, barrier, result, pred, multicast);
          })
      .def("create_async_tma_copy_local_to_global",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &coord,
              Value src) {
             self.create<ttng::AsyncTMACopyLocalToGlobalOp>(descPtr, coord,
                                                            src);
           })
      .def("create_async_tma_reduce",
           [](GluonOpBuilder &self, triton::DescriptorReduceKind kind,
              Value descPtr, std::vector<Value> &coord, Value src) {
             self.create<ttng::AsyncTMAReduceOp>(kind, descPtr, coord, src);
           })
      .def("create_async_tma_store_wait",
           [](GluonOpBuilder &self, int pendings) {
             self.create<ttng::TMAStoreWaitOp>(pendings);
           })
      .def("create_async_tma_gather",
           [](GluonOpBuilder &self, Value descPtr, Value xOffsets,
              Value yOffset, Value barrier, Value result, Value pred) {
             self.create<ttng::AsyncTMAGatherOp>(descPtr, xOffsets, yOffset,
                                                 barrier, result, pred);
           })
      .def("create_async_tma_scatter",
           [](GluonOpBuilder &self, Value descPtr, Value xOffsets,
              Value yOffset, Value src) {
             self.create<ttng::AsyncTMAScatterOp>(descPtr, xOffsets, yOffset,
                                                  src);
           })
      .def("create_fence_async_shared",
           [](GluonOpBuilder &self, bool bCluster) -> OpState {
             return self.create<ttng::FenceAsyncSharedOp>(bCluster);
           })

      .def("create_broadcast",
           [](TritonOpBuilder &self, Value &arg, Type retTy) -> Value {
             return self.create<tt::BroadcastOp>(retTy, arg);
           })
      .def("create_warp_return",
           [](GluonOpBuilder &self) -> Operation * {
             return self.create<ttg::WarpReturnOp>();
           })
      .def("create_warp_yield",
           [](GluonOpBuilder &self, std::vector<Value> &values) -> Operation * {
             return self.create<ttg::WarpYieldOp>(values);
           })
      .def("create_warp_specialize_partitions",
           [](GluonOpBuilder &self, std::vector<Value> &explicitCaptures,
              int numPartitions) -> Operation * {
             return self.create<ttg::WarpSpecializePartitionsOp>(
                 explicitCaptures, numPartitions);
           })
      .def("create_warp_specialize",
           [](GluonOpBuilder &self, std::vector<Type> &resultTypes,
              std::vector<int> &partitionNumWarps) {
             return self.create<ttg::WarpSpecializeOp>(resultTypes,
                                                       partitionNumWarps);
           })
      .def("create_buffer_load",
           [](GluonOpBuilder &self, Type resultType, Value ptr, Value offsets,
              Value mask, Value other, tt::CacheModifier cache) -> Value {
             return self.create<ttag::BufferLoadOp>(resultType, ptr, offsets,
                                                    Value() /*stride*/, cache,
                                                    mask, other);
           })
      .def("create_buffer_store",
           [](GluonOpBuilder &self, Value storedValue, Value ptr, Value offsets,
              Value mask, tt::CacheModifier cache) {
             self.create<ttag::BufferStoreOp>(storedValue, ptr, offsets,
                                              Value() /*stride*/, cache, mask);
           })
      .def("create_buffer_atomic_rmw",
           [](GluonOpBuilder &self, tt::RMWOp op, Value ptr, Value offsets,
              Value value, tt::MemSemantic sem, tt::MemSyncScope scope,
              Value mask) -> Value {
             return self.create<ttag::BufferAtomicRMWOp>(
                 value.getType(), op, ptr, offsets, value, Value() /*stride*/,
                 sem, scope, mask);
           })
      .def("create_buffer_load_to_local",
           [](GluonOpBuilder &self, Value dest, Value ptr, Value offsets,
              Value mask, Value other, Value stride,
              tt::CacheModifier cacheModifier) {
             self.create<ttag::BufferLoadToLocalOp>(
                 dest, ptr, offsets, mask, other, stride, cacheModifier);
           })
      .def("create_make_tensor_descriptor",
           [](TritonOpBuilder &self, Type resultTy, Value &base,
              std::vector<Value> &shape, std::vector<Value> &strides,
              tt::PaddingOption paddingOption) -> Value {
             return self.create<tt::MakeTensorDescOp>(resultTy, base, shape,
                                                      strides, paddingOption);
           })
      .def("create_async_tdm_copy_global_to_local",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &indices,
              Value result, Value pred, Value barrier) {
             self.create<ttag::AsyncTDMCopyGlobalToLocalOp>(
                 descPtr, indices, result, pred, barrier);
           })
      .def("create_async_tdm_copy_local_to_global",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &indices,
              Value src, Value barrier) {
             self.create<ttag::AsyncTDMCopyLocalToGlobalOp>(descPtr, indices,
                                                            src, barrier);
           })
      .def("create_async_tdm_scatter",
           [](GluonOpBuilder &self, Value descPtr, Value dstRowIndices,
              Value dstColOffset, Value src, Value barrier) {
             self.create<ttag::AsyncTDMScatterOp>(descPtr, dstRowIndices,
                                                  dstColOffset, src, barrier);
           })
      .def("create_async_tdm_gather",
           [](GluonOpBuilder &self, Value descPtr, Value srcRowIndices,
              Value srcColOffset, Value dst, Value pred, Value barrier) {
             self.create<ttag::AsyncTDMGatherOp>(
                 descPtr, srcRowIndices, srcColOffset, dst, pred, barrier);
           })
      .def("create_tdm_prefetch",
           [](GluonOpBuilder &self, Value descPtr, std::vector<Value> &indices,
              Value pred, bool speculative, bool returnOffsets) -> Value {
             auto op = self.create<ttag::TDMPrefetchOp>(
                 descPtr, indices, pred, speculative,
                 returnOffsets ? UnitAttr::get(self.getContext()) : nullptr);
             return returnOffsets ? op->getResult(0) : nullptr;
           })
      .def("create_async_tdm_wait",
           [](GluonOpBuilder &self, int num) {
             ValueRange tokens;
             self.create<ttag::AsyncTDMWait>(tokens, num);
           })
      .def("create_async_copy_lds_barrier_arrive",
           [](GluonOpBuilder &self, Value mbarrier) {
             self.create<ttag::AsyncCopyMbarrierArriveOp>(mbarrier);
           })
      .def("create_lds_barrier_init",
           [](GluonOpBuilder &self, Value memDesc, int count) {
             self.create<ttag::InitBarrierOp>(memDesc, count);
           })
      .def("create_lds_barrier_wait",
           [](GluonOpBuilder &self, Value memDesc, Value phase) {
             self.create<ttag::WaitBarrierOp>(memDesc, phase);
           })
      .def("create_lds_barrier_arrive",
           [](GluonOpBuilder &self, Value memDesc, int count) -> Value {
             return self.create<ttag::ArriveBarrierOp>(memDesc, count);
           })
      .def("create_amd_cluster_arrive",
           [](GluonOpBuilder &self) {
             self.create<ttag::ClusterBarrierArriveOp>();
           })
      .def("create_amd_cluster_wait",
           [](GluonOpBuilder &self) {
             self.create<ttag::ClusterBarrierWaitOp>();
           })
      .def("create_warp_pipeline_border",
           [](GluonOpBuilder &self, const std::string &marker, int priority) {
             auto border = self.create<ROCDL::SchedBarrier>(0);
             auto ctx = self.getContext();
             border->setAttr("triton.warp_pipeline.border",
                             StringAttr::get(ctx, marker));
             if (priority > -1) {
               auto i32Ty = IntegerType::get(ctx, 32);
               border->setAttr("triton.warp_pipeline.priority",
                               IntegerAttr::get(i32Ty, priority));
             }
           });

  m.def(
      "compute_tmem_reg_layout",
      [](py::object elementTyObj, std::vector<int64_t> shape,
         std::vector<int64_t> allocShape, py::object layoutObj,
         unsigned numWarps, const std::string &atomName) -> py::object {
        DialectRegistry registry;
        registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                        ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
        MLIRContext context(MLIRContext::Threading::DISABLED);
        context.appendDialectRegistry(registry);
        context.loadAllAvailableDialects();

        GluonOpBuilder builder(&context);
        auto builderObj =
            py::cast(&builder, py::return_value_policy::reference);

        auto elementType = elementTyObj.attr("to_ir")(builderObj).cast<Type>();
        auto layoutAttr =
            layoutObj.attr("_to_ir")(builderObj).cast<Attribute>();
        auto ctx = builder.getContext();
        auto memDescTy = builder.getChecked<ttg::MemDescType>(
            shape, elementType, layoutAttr,
            ttng::TensorMemorySpaceAttr::get(ctx),
            /*mutableMemory=*/true, allocShape);
        if (auto reason = ttng::getUnsupportedDirectTMemLdStReason(memDescTy))
          throw std::invalid_argument(*reason);
        auto matchesDesiredAtom =
            [&](ttg::MemDescType queryTy,
                std::optional<ttng::TMemAccessAtom> desiredAtom,
                ttng::TMemAccessAtom actualAtom) {
              return ttng::isTMemAccessAtomCompatibleWithRequest(
                  queryTy, desiredAtom, actualAtom);
            };
        auto firstLegalLayoutForType =
            [&](ttg::MemDescType queryTy,
                ArrayRef<ttg::DistributedEncodingTrait> layouts,
                std::optional<ttng::TMemAccessAtom> desiredAtom) -> py::object {
          for (auto candidateLayout : layouts) {
            auto regTy =
                RankedTensorType::get(shape, elementType, candidateLayout);
            auto maybeInfo = ttng::computeTMemLdStEncodingInfo(
                regTy, queryTy, /*maxnreg=*/256);
            if (succeeded(maybeInfo) &&
                matchesDesiredAtom(queryTy, desiredAtom, maybeInfo->atom)) {
              return layoutToGluon(candidateLayout);
            }
          }
          return py::none();
        };
        auto getBlockedFallbackLayouts =
            [&](ttg::MemDescType queryTy, ArrayRef<int64_t> tensorShape)
                -> SmallVector<ttg::DistributedEncodingTrait> {
          return ttng::getTMemLdStBlockedFallbackLayouts(queryTy, tensorShape,
                                                         numWarps);
        };
        auto physicalSupportLayout =
            [&](ttg::MemDescType queryTy,
                std::optional<ttng::TMemAccessAtom> desiredAtom) -> py::object {
          auto maybePlan = ttng::getTMemLdStPhysicalSupportPlan(
              queryTy, numWarps, /*maxnreg=*/256);
          if (!maybePlan ||
              !llvm::equal(maybePlan->regTy.getShape(), ArrayRef<int64_t>(shape)))
            return py::none();
          if (!matchesDesiredAtom(queryTy, desiredAtom, maybePlan->atom))
            return py::none();
          return layoutToGluon(maybePlan->regTy.getEncoding());
        };
        auto firstLegalLayoutForCanonicalType =
            [&](ttg::MemDescType queryTy,
                std::optional<ttng::TMemAccessAtom> desiredAtom) -> py::object {
          std::string canonicalError;
          auto maybeCanonicalEncoding =
              ttng::getCanonicalTMemLinearEncoding(queryTy, &canonicalError);
          if (!maybeCanonicalEncoding ||
              cast<Attribute>(*maybeCanonicalEncoding) ==
                  cast<Attribute>(queryTy.getEncoding())) {
            return py::none();
          }
          auto canonicalTy = builder.getChecked<ttg::MemDescType>(
              llvm::to_vector(queryTy.getShape()), queryTy.getElementType(),
              *maybeCanonicalEncoding, queryTy.getMemorySpace(),
              queryTy.getMutableMemory(),
              llvm::to_vector(queryTy.getAllocShape()));
          if (py::object layout = firstLegalLayoutForType(
                  canonicalTy,
                  ttng::getTmemCompatibleLayouts(canonicalTy, numWarps),
                  desiredAtom);
              !layout.is_none()) {
            return layout;
          }
          return physicalSupportLayout(canonicalTy, desiredAtom);
        };

        if (atomName == "auto") {
          auto layouts = ttng::getTmemCompatibleLayouts(memDescTy, numWarps);
          if (layouts.empty()) {
            std::string analysisError;
            if (auto maybeLayout = ttng::getTMemViewAnalysisLinearLayout(
                    shape, layoutAttr, &analysisError)) {
              auto twoCTAs = ttng::getTensorMemoryTwoCTAs(layoutAttr).value_or(false);
              if (auto maybeEncoding = ttng::tryMakeTMemViewEncoding(ctx, *maybeLayout, twoCTAs, &analysisError)) {
                auto canonicalTy = builder.getChecked<ttg::MemDescType>(
                    shape, elementType, *maybeEncoding,
                    ttng::TensorMemorySpaceAttr::get(ctx),
                    /*mutableMemory=*/true, allocShape);
                layouts = ttng::getTmemCompatibleLayouts(canonicalTy, numWarps);
              }
            }
          }
          if (!layouts.empty())
            return layoutToGluon(layouts.front());
          if (py::object blockedLayout = firstLegalLayoutForType(
                  memDescTy, getBlockedFallbackLayouts(memDescTy, shape),
                  /*desiredAtom=*/std::nullopt);
              !blockedLayout.is_none()) {
            return blockedLayout;
          }
          return physicalSupportLayout(memDescTy, /*desiredAtom=*/std::nullopt);
        }

        auto maybeAtomOr = ttng::parseTMemAccessAtomName(
            atomName, /*allowAuto=*/false, /*splitNAsPacked=*/false);
        if (failed(maybeAtomOr) || !maybeAtomOr->has_value())
          throw std::invalid_argument("unknown TMEM access atom: " + atomName);
        auto atom = **maybeAtomOr;
        if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
          throw std::invalid_argument(
              "numWarps must be a power of two and >= 4");
        if (ttng::shouldTryCanonicalTMemLdStLayoutForM64DirectAtom(
                memDescTy, numWarps, atom)) {
          if (py::object layout =
                  firstLegalLayoutForCanonicalType(memDescTy, atom);
              !layout.is_none()) {
            return layout;
          }
        }

        if (py::object layout = firstLegalLayoutForType(
                memDescTy, ttng::getTmemCompatibleLayouts(memDescTy, numWarps),
                atom);
            !layout.is_none()) {
          return layout;
        }

        std::string analysisError;
        if (auto maybeLayout =
                ttng::getTMemViewAnalysisLinearLayout(shape, layoutAttr,
                                                      &analysisError)) {
          auto twoCTAs =
              ttng::getTensorMemoryTwoCTAs(layoutAttr).value_or(false);
          if (auto maybeEncoding =
                  ttng::tryMakeTMemViewEncoding(ctx, *maybeLayout, twoCTAs,
                                               &analysisError)) {
            auto canonicalTy = builder.getChecked<ttg::MemDescType>(
                shape, elementType, *maybeEncoding,
                ttng::TensorMemorySpaceAttr::get(ctx),
                /*mutableMemory=*/true, allocShape);
            if (py::object layout = firstLegalLayoutForType(
                    canonicalTy,
                    ttng::getTmemCompatibleLayouts(canonicalTy, numWarps),
                    atom);
                !layout.is_none()) {
              return layout;
            }
          }
        }

        auto layout =
            ttng::getDistributedLayoutForTmemLdSt(memDescTy, atom, numWarps);
        if (!layout)
          return physicalSupportLayout(memDescTy, atom);

        auto attr =
            builder.getChecked<ttg::LinearEncodingAttr>(ctx, std::move(*layout));
        auto regTy = RankedTensorType::get(shape, elementType, attr);
        // The frontend must not promise a TMEM register layout that the actual
        // lowering later rejects. Use the default maxnreg budget here because
        // parsing happens without an op context.
        if (failed(ttng::computeTMemLdStEncodingInfo(regTy, memDescTy,
                                                     /*maxnreg=*/256)))
          layout.reset();
        else
          return layoutToGluon(attr);

        if (py::object blockedLayout =
                firstLegalLayoutForType(memDescTy,
                                        getBlockedFallbackLayouts(memDescTy, shape), atom);
            !blockedLayout.is_none()) {
          return blockedLayout;
        }
        if (py::object supportLayout = physicalSupportLayout(memDescTy, atom);
            !supportLayout.is_none()) {
          return supportLayout;
        }
        return py::none();
      });

  m.def(
      "infer_standalone_tmem_reg_layout_query_type_from_memdesc",
      [](Value memDesc) -> py::object {
        auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
        if (!memDescTy)
          throw std::invalid_argument("expected a memdesc value");
        std::string error;
        auto maybeTy =
            ttng::inferStandaloneTMemRegLayoutQueryType(memDesc, &error);
        if (failed(maybeTy))
          return py::none();
        if (*maybeTy == memDescTy)
          return py::none();
        std::string tyStr;
        llvm::raw_string_ostream os(tyStr);
        os << *maybeTy;
        return py::str(os.str());
      });

  m.def(
      "get_tmem_view_offset_from_memdesc",
      [](Value memDesc, std::vector<int32_t> offsets) -> uint32_t {
        auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
        if (!memDescTy)
          throw std::invalid_argument("expected a memdesc value");
        if (offsets.size() != static_cast<size_t>(memDescTy.getRank()))
          throw std::invalid_argument("offset rank mismatch");
        return ttng::getTMemViewOffset(memDescTy, offsets);
      });

  m.def(
      "compute_tmem_reg_layout_from_memdesc",
      [](Value memDesc, unsigned numWarps,
         const std::string &atomName) -> py::object {
        auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
        if (!memDescTy)
          throw std::invalid_argument("expected a memdesc value");
        bool debug = std::getenv("TRITON_DEBUG_TMEM_REG_LAYOUT") != nullptr;
        bool traceToFile =
            std::getenv("TRITON_TRACE_TMEM_REG_LAYOUT_FILE") != nullptr;
        std::string debugLogStr;
        llvm::raw_string_ostream debugLog(debugLogStr);
        auto appendTrace = [&](const Twine &msg) {
          if (!traceToFile)
            return;
          std::error_code ec;
          llvm::raw_fd_ostream os("/tmp/tmem_reg_layout_trace.log", ec,
                                  llvm::sys::fs::OF_Append);
          if (ec)
            return;
          os << msg << "\n";
        };
        auto ctx = memDesc.getContext();
        auto matchesDesiredAtom =
            [&](ttg::MemDescType queryTy,
                std::optional<ttng::TMemAccessAtom> desiredAtom,
                ttng::TMemAccessAtom actualAtom) {
              return ttng::isTMemAccessAtomCompatibleWithRequest(
                  queryTy, desiredAtom, actualAtom);
            };
        auto normalizeRegLayoutForAttr =
            [&](tt::LinearLayout layout) -> std::optional<tt::LinearLayout> {
          auto outDimNames =
              mlir::triton::standardOutDimNames(ctx, layout.getNumOutDims());
          SmallVector<std::pair<StringAttr, int32_t>> outDims;
          outDims.reserve(layout.getNumOutDims());
          for (auto [idx, size] : llvm::enumerate(layout.getOutDimSizes()))
            outDims.emplace_back(outDimNames[idx], static_cast<int32_t>(size));
          return tt::LinearLayout::tryCreate(layout.getBases(), std::move(outDims),
                                             layout.isSurjective(),
                                             /*error=*/nullptr);
        };
        auto createLinearRegAttr = [&](tt::LinearLayout layout)
            -> std::optional<ttg::LinearEncodingAttr> {
          std::string verifyDetails;
          llvm::raw_string_ostream os(verifyDetails);
          ScopedDiagnosticHandler handler(
              ctx, [&](Diagnostic &diag) { printDiagStr(os, diag); });
          if (failed(ttg::LinearEncodingAttr::verifyInvariants(
                  [&]() { return mlir::emitError(mlir::UnknownLoc::get(ctx)); },
                  layout))) {
            if (debug) {
              debugLog << "[tmem-reg-layout] invalid linear layout:\n"
                       << layout.toString() << "\n";
              if (!verifyDetails.empty())
                debugLog << "[tmem-reg-layout] invalid linear attr: "
                         << verifyDetails;
            }
            if (traceToFile && !verifyDetails.empty())
              appendTrace(Twine("invalid linear attr: ") + verifyDetails);
            return std::nullopt;
          }
          return ttg::LinearEncodingAttr::get(ctx, std::move(layout));
        };
        auto getCompatibleLayouts = [&](Value queryMemDesc,
                                       ttg::MemDescType queryTy) {
          SmallVector<ttg::DistributedEncodingTrait> layouts;
          auto addAttr = [&](ttg::DistributedEncodingTrait attr) {
            if (llvm::none_of(layouts, [&](ttg::DistributedEncodingTrait existing) {
                  return cast<Attribute>(existing) == cast<Attribute>(attr);
              })) {
              layouts.push_back(attr);
            }
          };
          auto addLayout = [&](tt::LinearLayout layout) {
            auto normalizedLayout =
                normalizeRegLayoutForAttr(std::move(layout));
            if (!normalizedLayout)
              return;
            auto attr = createLinearRegAttr(std::move(*normalizedLayout));
            if (!attr)
              return;
            addAttr(*attr);
          };

          for (auto candidate : ttng::getTMemLdStCandidateLayoutsForQuery(
                   queryMemDesc, queryTy, numWarps, atomName)) {
            if (traceToFile) {
              appendTrace(Twine("getCompatibleLayouts rowPlan atom=") +
                          Twine(static_cast<int>(candidate.atom)) +
                          " layout=" + candidate.layout.toString());
            }
            addLayout(std::move(candidate.layout));
          }
          for (auto layout : ttng::getTMemLdStGenericCompatibleLayouts(
                   queryMemDesc, queryTy, numWarps, atomName)) {
            addAttr(layout);
          }
          return layouts;
        };
        auto getBlockedFallbackLayouts =
            [&](ttg::MemDescType queryTy, ArrayRef<int64_t> tensorShape)
                -> SmallVector<ttg::DistributedEncodingTrait> {
          return ttng::getTMemLdStBlockedFallbackLayouts(queryTy, tensorShape,
                                                         numWarps);
        };
        auto firstLegalLayoutForType =
            [&](ttg::MemDescType queryTy,
                ArrayRef<ttg::DistributedEncodingTrait> layouts,
                std::optional<ttng::TMemAccessAtom> desiredAtom) -> py::object {
          auto shape = llvm::to_vector(queryTy.getShape());
          auto elementType = queryTy.getElementType();
          for (auto candidateLayout : layouts) {
            auto regTy =
                RankedTensorType::get(shape, elementType, candidateLayout);
            auto maybeInfo = ttng::computeTMemLdStEncodingInfo(
                regTy, queryTy, /*maxnreg=*/256);
            if (succeeded(maybeInfo) &&
                matchesDesiredAtom(queryTy, desiredAtom, maybeInfo->atom)) {
              appendTrace(Twine("firstLegalLayoutForType atomName=") + atomName +
                          " matchedAtom=" +
                          Twine(static_cast<int>(maybeInfo->atom)));
              return layoutToGluon(candidateLayout);
            }
          }
          return py::none();
        };
        auto firstLegalLayout = [&](Value queryMemDesc, ttg::MemDescType queryTy,
                                    ArrayRef<ttg::DistributedEncodingTrait> layouts,
                                    std::optional<ttng::TMemAccessAtom> desiredAtom)
            -> py::object {
          auto queryMemDescTy = cast<ttg::MemDescType>(queryMemDesc.getType());
          auto shape = llvm::to_vector(
              queryMemDescTy.getShape().take_back(queryMemDescTy.getRank()));
          auto elementType = queryMemDescTy.getElementType();
          std::optional<ttng::TMemLdStRowPlan> rowPlan;
          if (auto supportPlan =
                  ttng::getTMemLdStSupportQueryPlan(queryMemDesc,
                                                    /*error=*/nullptr)) {
            rowPlan = supportPlan->rowPlan;
            if (!rowPlan)
              rowPlan = ttng::getTMemLdStRowPlan(supportPlan->query.layout);
          }
          if (!rowPlan) {
            if (auto rawQuery = ttng::inferStandaloneTMemLdStQueryLayout(
                    queryMemDesc, /*preserveNonCanonicalView=*/true,
                    /*error=*/nullptr);
                succeeded(rawQuery)) {
              rowPlan = ttng::getTMemLdStRowPlan(rawQuery->layout);
            }
          }
          if (!rowPlan)
            rowPlan = ttng::getTMemLdStRowPlanForQuery(queryMemDesc, queryTy);
          if (debug) {
            debugLog << "[tmem-reg-layout] queryTy=" << queryTy
                     << " atom=" << atomName
                     << " layouts=" << layouts.size() << "\n";
          }
          for (auto candidateLayout : layouts) {
            auto regTy =
                RankedTensorType::get(shape, elementType, candidateLayout);
            std::string candidateDetails;
            auto maybeInfo = [&]() -> FailureOr<ttng::TMemLdStEncodingInfo> {
              llvm::raw_string_ostream os(candidateDetails);
              ScopedDiagnosticHandler handler(
                  ctx, [&](Diagnostic &diag) { diag.print(os); });
              return ttng::computeTMemLdStEncodingInfo(
                  regTy, queryTy, /*maxnreg=*/256,
                  [&]() { return mlir::emitError(mlir::UnknownLoc::get(ctx)); },
                  rowPlan);
            }();
            if (debug) {
              debugLog << "[tmem-reg-layout] candidate="
                       << cast<Attribute>(candidateLayout) << " -> "
                       << (succeeded(maybeInfo)
                               ? ("ok atom=" +
                                  llvm::Twine(static_cast<int>(maybeInfo->atom)))
                                     .str()
                               : ("fail details=" + candidateDetails))
                       << "\n";
            }
            if (traceToFile) {
              appendTrace(Twine("firstLegalLayout atomName=") + atomName +
                          " candidate=" +
                          (succeeded(maybeInfo)
                               ? (Twine("ok matchedAtom=") +
                                  Twine(static_cast<int>(maybeInfo->atom)))
                               : (Twine("fail details=") + candidateDetails)));
            }
            if (succeeded(maybeInfo) &&
                matchesDesiredAtom(queryTy, desiredAtom, maybeInfo->atom))
              return layoutToGluon(candidateLayout);
          }
          return py::none();
        };
        auto physicalSupportLayout =
            [&](Value queryMemDesc,
                std::optional<ttng::TMemAccessAtom> desiredAtom) -> py::object {
          auto maybePlan = ttng::getTMemLdStPhysicalSupportPlan(
              queryMemDesc, numWarps, /*maxnreg=*/256);
          if (!maybePlan) {
            std::string error;
            auto standaloneTy =
                ttng::inferStandaloneTMemViewType(queryMemDesc, &error);
            if (failed(standaloneTy))
              return py::none();
            maybePlan = ttng::getTMemLdStPhysicalSupportPlan(
                *standaloneTy, numWarps, /*maxnreg=*/256);
          }
          if (!maybePlan)
            return py::none();
          auto queryTy = cast<ttg::MemDescType>(queryMemDesc.getType());
          if (!matchesDesiredAtom(queryTy, desiredAtom, maybePlan->atom))
            return py::none();
          auto queryShape = llvm::to_vector(
              cast<ttg::MemDescType>(queryMemDesc.getType()).getShape());
          if (!llvm::equal(maybePlan->regTy.getShape(),
                           ArrayRef<int64_t>(queryShape))) {
            auto countElems = [](auto shape) {
              return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                     std::multiplies<int64_t>());
            };
            if (countElems(maybePlan->regTy.getShape()) !=
                countElems(queryShape)) {
              return py::none();
            }
            auto reshapedRegLayout =
                mlir::triton::reshapeLayout(ctx,
                                            ttg::toLinearLayout(maybePlan->regTy),
                                            queryShape);
            auto normalizedLayout =
                normalizeRegLayoutForAttr(std::move(reshapedRegLayout));
            if (!normalizedLayout)
              return py::none();
            auto attr = createLinearRegAttr(std::move(*normalizedLayout));
            if (!attr)
              return py::none();
            return layoutToGluon(*attr);
          }
          return layoutToGluon(maybePlan->regTy.getEncoding());
        };
        auto reshapeRegLayoutToQueryShape =
            [&](const tt::LinearLayout &layout,
                ArrayRef<int64_t> queryShape)
                -> std::optional<tt::LinearLayout> {
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
          auto countElems = [](ArrayRef<int64_t> shape) {
            return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                   std::multiplies<int64_t>());
          };
          if (countElems(layoutShape) != countElems(queryShape))
            return std::nullopt;
          return mlir::triton::reshapeLayout(ctx, layout, queryShape);
        };
        auto inferRawQueryLayout =
            [&](Value queryMemDesc) -> std::optional<ttng::TMemLdStQueryLayout> {
          std::string error;
          auto maybeQueryLayout = ttng::inferStandaloneTMemLdStQueryLayout(
              queryMemDesc, /*preserveNonCanonicalView=*/true, &error);
          if (failed(maybeQueryLayout)) {
            if (traceToFile && !error.empty()) {
              appendTrace(Twine("inferRawQueryLayout atomName=") + atomName +
                          " error=" + error);
            }
            if (debug && !error.empty()) {
              debugLog << "[tmem-reg-layout] raw query layout failed: " << error
                       << "\n";
            }
            return std::nullopt;
          }
          return *maybeQueryLayout;
        };
        auto firstLegalLayoutForQueryLayout =
            [&](Value queryMemDesc, const ttng::TMemLdStQueryLayout &queryLayout,
                std::optional<ttng::TMemAccessAtom> desiredAtom,
                std::optional<ttng::TMemLdStRowPlan> rowPlanOverride =
                    std::nullopt) -> py::object {
          auto queryTy = cast<ttg::MemDescType>(queryMemDesc.getType());
          bool disableRawRowPlanOverride =
              ttng::disallowTMemLdStRawQueryRowPlanOverride(queryMemDesc);
          std::optional<ttng::TMemLdStRowPlan> rowPlan = rowPlanOverride;
          if (!rowPlan && !disableRawRowPlanOverride) {
            rowPlan = ttng::getTMemLdStRowPlanForQueryLayout(queryMemDesc,
                                                             queryTy,
                                                             queryLayout);
            if (!rowPlan)
              rowPlan = ttng::getBackingTMemLdStRowPlan(queryMemDesc);
          }
          if (!rowPlan)
            rowPlan = ttng::getTMemLdStRowPlan(queryLayout.layout);
          if (traceToFile) {
            appendTrace(Twine("firstLegalLayoutForQueryLayout atomName=") +
                        atomName + " rowPlan=" +
                        (rowPlan
                             ? Twine("{") + Twine(rowPlan->warpRow0) + "," +
                                   Twine(rowPlan->warpRow1) + ";span=" +
                                   Twine(rowPlan->rowSpan) + ";base=" +
                                   Twine(rowPlan->baseOffset) + "}"
                             : Twine("none")) +
                        " layout=" + queryLayout.layout.toString());
          }

          auto tryAtom = [&](ttng::TMemAccessAtom atom) -> py::object {
            auto maybeLayout = ttng::getDistributedLayoutForTmemLdSt(
                queryTy, atom, numWarps, rowPlan, queryLayout.layout);
            if (debug) {
              debugLog << "[tmem-reg-layout] raw atom="
                       << static_cast<int>(atom) << " -> "
                       << (maybeLayout ? "layout" : "none") << "\n";
            }
            if (traceToFile && !maybeLayout)
              appendTrace(Twine("rawQuery atomName=") + atomName + " tryAtom=" +
                          Twine(static_cast<int>(atom)) + " no-layout");
            if (!maybeLayout)
              return py::none();
            auto reshapedLayout =
                reshapeRegLayoutToQueryShape(*maybeLayout, queryTy.getShape());
            if (!reshapedLayout)
              return py::none();
            auto normalizedLayout =
                normalizeRegLayoutForAttr(std::move(*reshapedLayout));
            if (!normalizedLayout)
              return py::none();
            auto attr = createLinearRegAttr(std::move(*normalizedLayout));
            if (!attr)
              return py::none();
            auto regTy = RankedTensorType::get(
                queryTy.getShape(), queryTy.getElementType(), *attr);
            std::string rawDetails;
            auto maybeInfo = [&]() -> FailureOr<ttng::TMemLdStEncodingInfo> {
              llvm::raw_string_ostream os(rawDetails);
              ScopedDiagnosticHandler handler(
                  ctx, [&](Diagnostic &diag) { diag.print(os); });
              return ttng::computeTMemLdStEncodingInfo(
                  regTy, queryTy, queryLayout, /*maxnreg=*/256,
                  [&]() { return mlir::emitError(mlir::UnknownLoc::get(ctx)); },
                  rowPlan);
            }();
            if (debug) {
              debugLog << "[tmem-reg-layout] raw candidate="
                       << cast<Attribute>(*attr) << " -> "
                       << (succeeded(maybeInfo)
                               ? ("ok atom=" +
                                  llvm::Twine(static_cast<int>(maybeInfo->atom)))
                                     .str()
                               : ("fail details=" + rawDetails))
                       << "\n";
            }
            if (traceToFile) {
              appendTrace(Twine("rawQuery atomName=") + atomName + " tryAtom=" +
                          Twine(static_cast<int>(atom)) + " candidate=" +
                          (succeeded(maybeInfo)
                               ? (Twine("ok matchedAtom=") +
                                  Twine(static_cast<int>(maybeInfo->atom)))
                               : (Twine("fail details=") + rawDetails)));
            }
            if (succeeded(maybeInfo) &&
                matchesDesiredAtom(queryTy, desiredAtom, maybeInfo->atom)) {
              appendTrace(Twine("firstLegalLayoutForQueryLayout atomName=") +
                          atomName + " matchedAtom=" +
                          Twine(static_cast<int>(maybeInfo->atom)));
              return layoutToGluon(*attr);
            }
            return py::none();
          };

          for (auto atom : ttng::getTMemLdStAtomSearchOrder(desiredAtom)) {
            py::object layout = tryAtom(atom);
            if (!layout.is_none())
              return layout;
          }
          return py::none();
        };

        auto findDirectLayoutForMemDesc =
            [&](Value queryMemDesc,
                std::optional<ttng::TMemAccessAtom> desiredAtom) -> py::object {
          auto queryMemDescTy = dyn_cast<ttg::MemDescType>(queryMemDesc.getType());
          if (!queryMemDescTy)
            return py::none();
          std::string unsupportedDescriptorViewError;
          if (ttng::isUnsupportedDirectTMemLdStDescriptorView(
                  queryMemDesc, &unsupportedDescriptorViewError)) {
            if (traceToFile && !unsupportedDescriptorViewError.empty()) {
              appendTrace(Twine("findDirectLayoutForMemDesc unsupportedView=") +
                          unsupportedDescriptorViewError);
            }
            if (debug && !unsupportedDescriptorViewError.empty()) {
              debugLog << "[tmem-reg-layout] unsupported descriptor view: "
                       << unsupportedDescriptorViewError << "\n";
            }
            return py::none();
          }
          bool isViewLikeMemDesc =
              isa_and_nonnull<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
                              ttg::MemDescReshapeOp, ttg::MemDescTransOp,
                              ttg::MemDescReinterpretOp>(
                  queryMemDesc.getDefiningOp());
          bool disableTypeOnlyFallback =
              std::getenv("TRITON_DISABLE_TYPE_ONLY_TMEM_REG_LAYOUT_FALLBACK") !=
              nullptr;
          auto queryTypes = ttng::getTMemLdStQueryTypes(queryMemDesc);
          auto preferQueryTypeLayoutsBeforeRawQuery =
              ttng::shouldPreferTMemLdStQueryTypeLayoutsBeforeRawQuery(
                  queryMemDesc, numWarps, desiredAtom);
          auto tryQueryTypeLayouts = [&]() -> py::object {
            for (ttg::MemDescType queryTy : queryTypes) {
              auto layouts = getCompatibleLayouts(queryMemDesc, queryTy);
              py::object layout = firstLegalLayout(queryMemDesc, queryTy, layouts,
                                                   desiredAtom);
              if (!layout.is_none()) {
                appendTrace("findDirectLayoutForMemDesc queryTy layouts");
                return layout;
              }
            }
            return py::none();
          };
          std::string supportError;
          auto trySupportLayout =
              [&](const ttng::TMemLdStQueryLayout &supportQuery,
                  std::optional<ttng::TMemLdStRowPlan> supportRowPlan)
              -> py::object {
            if (!supportRowPlan)
              supportRowPlan = ttng::getTMemLdStRowPlanForQueryLayout(
                  queryMemDesc, queryMemDescTy, supportQuery);
            if (!supportRowPlan)
              supportRowPlan = ttng::getBackingTMemLdStRowPlan(queryMemDesc);
            if (debug) {
              debugLog << "[tmem-reg-layout] support rowPlan="
                       << (supportRowPlan
                               ? ("{" +
                                  std::to_string(supportRowPlan->warpRow0) +
                                  "," +
                                  std::to_string(supportRowPlan->warpRow1) +
                                  ";span=" +
                                  std::to_string(supportRowPlan->rowSpan) +
                                  ";base=" +
                                  std::to_string(supportRowPlan->baseOffset) +
                                  "}")
                               : std::string("none"))
                       << "\n";
            }
            auto trySupportAtom = [&](ttng::TMemAccessAtom atom) -> py::object {
              if (!supportRowPlan) {
                if (traceToFile)
                  appendTrace(Twine("supportQuery atomName=") + atomName +
                              " tryAtom=" + Twine(static_cast<int>(atom)) +
                              " no-row-plan");
                return py::none();
              }
              auto maybeLayout = ttng::getDistributedLayoutForTmemLdSt(
                  queryMemDescTy, atom, numWarps, supportRowPlan,
                  supportQuery.layout);
              if (debug) {
                debugLog << "[tmem-reg-layout] support atom="
                         << static_cast<int>(atom) << " -> "
                         << (maybeLayout ? "layout" : "none") << "\n";
              }
              if (!maybeLayout) {
                if (traceToFile)
                  appendTrace(Twine("supportQuery atomName=") + atomName +
                              " tryAtom=" + Twine(static_cast<int>(atom)) +
                              " no-layout");
                return py::none();
              }
              auto reshapedLayout = reshapeRegLayoutToQueryShape(
                  *maybeLayout, queryMemDescTy.getShape());
              if (!reshapedLayout) {
                if (traceToFile)
                  appendTrace(Twine("supportQuery atomName=") + atomName +
                              " tryAtom=" + Twine(static_cast<int>(atom)) +
                              " reshape-failed");
                return py::none();
              }
              auto normalizedLayout =
                  normalizeRegLayoutForAttr(std::move(*reshapedLayout));
              if (!normalizedLayout) {
                if (traceToFile)
                  appendTrace(Twine("supportQuery atomName=") + atomName +
                              " tryAtom=" + Twine(static_cast<int>(atom)) +
                              " normalize-failed");
                return py::none();
              }
              auto attr = createLinearRegAttr(std::move(*normalizedLayout));
              if (!attr) {
                if (traceToFile)
                  appendTrace(Twine("supportQuery atomName=") + atomName +
                              " tryAtom=" + Twine(static_cast<int>(atom)) +
                              " invalid-linear-attr");
                return py::none();
              }
              auto regTy = RankedTensorType::get(
                  queryMemDescTy.getShape(), queryMemDescTy.getElementType(),
                  *attr);
              std::string supportDetails;
              auto maybeInfo = [&]() -> FailureOr<ttng::TMemLdStEncodingInfo> {
                llvm::raw_string_ostream os(supportDetails);
                ScopedDiagnosticHandler handler(
                    ctx, [&](Diagnostic &diag) { diag.print(os); });
                return ttng::computeTMemLdStEncodingInfo(
                    regTy, queryMemDescTy, supportQuery, /*maxnreg=*/256,
                    [&]() { return mlir::emitError(mlir::UnknownLoc::get(ctx)); },
                    supportRowPlan);
              }();
              if (traceToFile) {
                appendTrace(Twine("supportQuery atomName=") + atomName +
                            " tryAtom=" + Twine(static_cast<int>(atom)) +
                            " candidate=" +
                            (succeeded(maybeInfo)
                                 ? (Twine("ok matchedAtom=") +
                                    Twine(static_cast<int>(maybeInfo->atom)))
                                 : (Twine("fail details=") + supportDetails)));
              }
              if (succeeded(maybeInfo) &&
                  matchesDesiredAtom(queryMemDescTy, desiredAtom,
                                     maybeInfo->atom)) {
                return layoutToGluon(*attr);
              }
              return py::none();
            };
            py::object layout = py::none();
            for (auto atom : ttng::getTMemLdStAtomSearchOrder(desiredAtom)) {
              layout = trySupportAtom(atom);
              if (!layout.is_none())
                break;
            }
            if (layout.is_none()) {
              layout = firstLegalLayoutForQueryLayout(
                  queryMemDesc, supportQuery, desiredAtom, supportRowPlan);
            }
            if (!layout.is_none()) {
              appendTrace("findDirectLayoutForMemDesc supportQuery");
              return layout;
            }
            if (debug)
              debugLog << "[tmem-reg-layout] reshaped support query failed\n";
            return py::none();
          };
          auto tryCanonicalM64SplitNRawQuery =
              [&](const ttng::TMemLdStQueryLayout &rawQueryLayout)
              -> py::object {
            // The generic exact-query search still fails to expose the
            // canonical split-N user layout for this simple M64 image: for
            // 32-bit rows it rejects the unused half tile as a zero row basis,
            // while for 16-bit rows it can validate the hardware message with
            // the high-N split left in lanes. Once the raw query proves the
            // exact simple M64 image, return the canonical split-N register
            // layout that load/store lowering already accepts for the same
            // physical TMEM data.
            auto canonical =
                ttng::getCanonicalM64SplitNLayoutForRawQueryRequest(
                    queryMemDescTy, rawQueryLayout, numWarps, atomName,
                    desiredAtom, /*allow16Bit=*/true);
            if (!canonical)
              return py::none();
            auto normalizedLayout =
                normalizeRegLayoutForAttr(std::move(*canonical));
            if (!normalizedLayout)
              return py::none();
            auto attr = createLinearRegAttr(std::move(*normalizedLayout));
            if (!attr)
              return py::none();
            if (traceToFile) {
              appendTrace(Twine("canonicalM64SplitNRawQuery atomName=") +
                          atomName + " recognized");
            }
            appendTrace("findDirectLayoutForMemDesc canonicalM64SplitNRawQuery");
            return layoutToGluon(*attr);
          };
          if (auto supportPlan =
                  ttng::getTMemLdStSupportQueryPlan(queryMemDesc,
                                                    &supportError)) {
            py::object layout =
                trySupportLayout(supportPlan->query, supportPlan->rowPlan);
            if (!layout.is_none())
              return layout;
            py::object supportFallback =
                physicalSupportLayout(queryMemDesc, desiredAtom);
            if (!supportFallback.is_none()) {
              appendTrace(
                  "findDirectLayoutForMemDesc physicalSupportLayout-after-support");
              return supportFallback;
            }
          } else if (debug && !supportError.empty()) {
            debugLog << "[tmem-reg-layout] support query unavailable: "
                     << supportError << "\n";
          }
          if (preferQueryTypeLayoutsBeforeRawQuery) {
            py::object layout = tryQueryTypeLayouts();
            if (!layout.is_none())
              return layout;
          }
          if (auto rawQueryLayout = inferRawQueryLayout(queryMemDesc)) {
            if (atomName == "auto") {
              py::object layout = tryCanonicalM64SplitNRawQuery(*rawQueryLayout);
              if (!layout.is_none())
                return layout;
            }
            py::object layout = firstLegalLayoutForQueryLayout(
                queryMemDesc, *rawQueryLayout, desiredAtom);
            if (!layout.is_none()) {
              appendTrace("findDirectLayoutForMemDesc rawQuery");
              return layout;
            }
            layout = tryCanonicalM64SplitNRawQuery(*rawQueryLayout);
            if (!layout.is_none())
              return layout;
          }
          if (ttng::isTMemLdStHalfRowsDescriptorView(queryMemDesc)) {
            if (traceToFile)
              appendTrace("findDirectLayoutForMemDesc halfRows exact-lowering-required");
            if (debug) {
              debugLog << "[tmem-reg-layout] half-rows descriptor view requires exact support/raw-query lowering; refusing type-only fallback\n";
            }
            return py::none();
          }
          if (disableTypeOnlyFallback && isViewLikeMemDesc) {
            return py::none();
          }
          py::object supportFallback = py::none();
          supportFallback = physicalSupportLayout(queryMemDesc, desiredAtom);
          if (!supportFallback.is_none()) {
            appendTrace("findDirectLayoutForMemDesc physicalSupportLayout");
            return supportFallback;
          }
          std::string typeOnlyFallbackReason;
          if (ttng::disallowTMemLdStTypeOnlyFallback(
                  queryMemDesc, &typeOnlyFallbackReason)) {
            if (traceToFile)
              appendTrace(
                  "findDirectLayoutForMemDesc twoCTA-int8 exact-query-required");
            if (debug) {
              debugLog << "[tmem-reg-layout] " << typeOnlyFallbackReason
                       << "; refusing type-only fallback\n";
            }
            return py::none();
          }
          if (!preferQueryTypeLayoutsBeforeRawQuery) {
            py::object layout = tryQueryTypeLayouts();
            if (!layout.is_none())
              return layout;
          }
          for (ttg::MemDescType queryTy : queryTypes) {
            auto shape = llvm::to_vector(
                queryMemDescTy.getShape().take_back(queryMemDescTy.getRank()));
            auto blockedLayouts = getBlockedFallbackLayouts(queryTy, shape);
            py::object layout =
                firstLegalLayout(queryMemDesc, queryTy, blockedLayouts,
                                 desiredAtom);
            if (!layout.is_none()) {
              appendTrace("findDirectLayoutForMemDesc blocked fallback");
              return layout;
            }
          }
          py::object fallbackLayout = py::none();
          fallbackLayout = physicalSupportLayout(queryMemDesc, desiredAtom);
          if (!fallbackLayout.is_none()) {
            appendTrace("findDirectLayoutForMemDesc physicalSupportLayout");
            return fallbackLayout;
          }
          if (desiredAtom) {
            if (isViewLikeMemDesc)
              return py::none();
            if (auto maybeLayout = ttng::getDistributedLayoutForTmemLdSt(
                    memDescTy, *desiredAtom, numWarps)) {
              auto normalizedLayout =
                  normalizeRegLayoutForAttr(std::move(*maybeLayout));
              if (!normalizedLayout)
                return py::none();
              auto attr = createLinearRegAttr(std::move(*normalizedLayout));
              if (!attr)
                return py::none();
              auto regTy = RankedTensorType::get(
                  memDescTy.getShape(), memDescTy.getElementType(), *attr);
              if (succeeded(ttng::computeTMemLdStEncodingInfo(
                      regTy, memDescTy, /*maxnreg=*/256))) {
                return layoutToGluon(*attr);
              }
            }
          } else {
            fallbackLayout = firstLegalLayoutForType(
                memDescTy, getCompatibleLayouts(memDesc, memDescTy), desiredAtom);
            if (!fallbackLayout.is_none())
              return fallbackLayout;
          }
          return py::none();
        };

        auto maybeAtomOr = ttng::getTMemLdStRequestedAtomForMemDesc(
            memDescTy, atomName, numWarps);
        if (failed(maybeAtomOr))
          throw std::invalid_argument("unknown TMEM access atom: " + atomName);
        auto maybeAtom = *maybeAtomOr;
        if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
          throw std::invalid_argument(
              "numWarps must be a power of two and >= 4");

        if (ttng::shouldPreferLegacyTMemLdStI32x32bForAuto(memDescTy,
                                                           atomName)) {
          py::object legacyLayout =
              findDirectLayoutForMemDesc(memDesc, ttng::TMemAccessAtom::I32x32b);
          if (!legacyLayout.is_none()) {
            if (debug)
              llvm::errs() << debugLog.str();
            return legacyLayout;
          }
        }

        py::object layout = findDirectLayoutForMemDesc(memDesc, maybeAtom);
        if (!layout.is_none()) {
          if (debug)
            llvm::errs() << debugLog.str();
          return layout;
        }
          if (debug)
            llvm::errs() << debugLog.str();
          return py::none();
      });

  m.def(
      "get_tmem_ldst_unsupported_reason_from_memdesc",
      [](Value memDesc) -> py::object {
        auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
        if (!memDescTy)
          throw std::invalid_argument("expected a memdesc value");
        std::string reason;
        if (ttng::isUnsupportedDirectTMemLdStDescriptorView(memDesc, &reason) &&
            !reason.empty()) {
          return py::str(reason);
        }
        return py::none();
      });

  m.def(
      "get_tmem_ldst_unsupported_reason_from_memdesc_for_variant",
      [](Value memDesc, unsigned numWarps,
         const std::string &atomName) -> py::object {
        if (!isa<ttg::MemDescType>(memDesc.getType()))
          throw std::invalid_argument("expected a memdesc value");
        std::string reason;
        if (ttng::isUnsupportedDirectTMemLdStDescriptorView(memDesc, &reason) &&
            !reason.empty()) {
          return py::str(reason);
        }

        auto maybeAtom = ttng::parseTMemAccessAtomName(
            atomName, /*allowAuto=*/false, /*splitNAsPacked=*/true);
        if (succeeded(maybeAtom) && maybeAtom->has_value()) {
          if (auto atomReason = ttng::getUnsupportedDirectTMemLdStVariantReason(
                  memDesc, **maybeAtom, numWarps)) {
            return py::str(*atomReason);
          }
        }
        return py::none();
      });

  m.def(
      "compute_tmem_reduce_reg_layout_from_memdesc",
      [](Value memDesc, unsigned numWarps) -> py::object {
        auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
        if (!memDescTy)
          throw std::invalid_argument("expected a memdesc value");
        if (numWarps < 4 || !llvm::isPowerOf2_32(numWarps))
          throw std::invalid_argument(
              "numWarps must be a power of two and >= 4");
        if (auto layout =
                ttng::getTMemLoadReductionLayoutForMemDesc(memDesc, numWarps))
          return layoutToGluon(*layout);
        return py::none();
      });

  m.def("is_tmem_load_reduction_reg_layout_supported", [](Type resultTy) {
    auto rankedTy = dyn_cast<RankedTensorType>(resultTy);
    if (!rankedTy)
      throw std::invalid_argument("expected a ranked tensor result type");
    return static_cast<bool>(ttng::getTmemLoadReductionLayoutSupport(
        rankedTy, ttg::toLinearLayout(rankedTy)));
  });

  m.def(
      "make_cga_layout",
      [](std::vector<unsigned> ctasPerCga, std::vector<unsigned> ctaSplitNum,
         std::vector<unsigned> ctaOrder) -> std::vector<std::vector<int32_t>> {
        DialectRegistry registry;
        registry.insert<triton::TritonDialect, ttg::TritonGPUDialect>();
        MLIRContext ctx(MLIRContext::Threading::DISABLED);
        ctx.appendDialectRegistry(registry);
        ctx.loadAllAvailableDialects();
        auto attr = ttg::CGAEncodingAttr::fromSplitParams(
            &ctx, ctasPerCga, ctaSplitNum, ctaOrder);
        return getCgaLayoutBases(attr);
      });

  m.def("get_amd_mfma_scale_layout",
        [](unsigned opIdx, std::vector<int64_t> &shape, unsigned mfmaMDim,
           std::vector<unsigned> &tilesPerWarp,
           std::vector<unsigned> &warpsPerCTA) -> py::object {
          DialectRegistry registry;
          registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                          ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
          MLIRContext ctx(MLIRContext::Threading::DISABLED);
          ctx.appendDialectRegistry(registry);
          ctx.loadAllAvailableDialects();

          auto ll = ttg::chooseScaledMfmaScaleLayout(
              &ctx, opIdx, shape, mfmaMDim, tilesPerWarp, warpsPerCTA);
          auto attr = ttg::LinearEncodingAttr::get(&ctx, std::move(ll));
          return layoutToGluon(attr);
        });

  m.def("get_amd_wmma_scale_layout",
        [](unsigned opIdx, std::vector<int64_t> &shape, unsigned wmmaMDim,
           unsigned scaleFactor, std::vector<std::vector<int32_t>> &regBases,
           std::vector<std::vector<int32_t>> &warpBases,
           std::vector<std::vector<int32_t>> &cgaBases) -> py::object {
          DialectRegistry registry;
          registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                          ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
          MLIRContext ctx(MLIRContext::Threading::DISABLED);
          ctx.appendDialectRegistry(registry);
          ctx.loadAllAvailableDialects();

          auto rank = shape.size();
          auto kReg = mlir::StringAttr::get(&ctx, "register");
          auto kWarp = mlir::StringAttr::get(&ctx, "warp");
          auto ctaLayout =
              tt::LinearLayout({{kReg, regBases}, {kWarp, warpBases}},
                               tt::standardOutDimNames(&ctx, rank));
          auto cgaLayout = buildCgaLayoutAttr(&ctx, cgaBases, rank);
          auto ll = ttg::chooseScaledWmmaScaleLayout(
              &ctx, opIdx, shape, wmmaMDim, scaleFactor, ctaLayout, cgaLayout);
          auto attr = ttg::LinearEncodingAttr::get(&ctx, ll);
          return layoutToGluon(attr);
        });

  m.def("get_layout_view",
        [](py::object layout, std::vector<int64_t> shape,
           bool useHwView) -> std::string {
          DialectRegistry registry;
          registry.insert<triton::TritonDialect, ttg::TritonGPUDialect,
                          ttng::TritonNvidiaGPUDialect, gluon::GluonDialect>();
          MLIRContext ctx(MLIRContext::Threading::DISABLED);
          ctx.appendDialectRegistry(registry);
          ctx.loadAllAvailableDialects();

          GluonOpBuilder builder(&ctx);
          auto builderObj =
              py::cast(&builder, py::return_value_policy::reference);
          Attribute attr = layout.attr("_to_ir")(builderObj).cast<Attribute>();

          if (isa<gluon::AutoEncodingAttr>(attr))
            throw py::value_error("AutoLayout cannot be visualized");
          if (isa<gluon::CoalescedEncodingAttr>(attr))
            throw py::value_error("CoalescedLayout cannot be visualized");
          if (isa<ttg::PaddedSharedEncodingAttr>(attr))
            throw py::value_error("PaddedSharedLayout cannot be visualized: "
                                  "toLinearLayout not implemented");

          auto ll = ttg::toLinearLayout(shape, attr);
          if (isa<ttg::DistributedEncodingTrait>(attr)) {
            return ttg::getDistributedLayoutStr(ll, useHwView);
          } else {
            return ttg::getSharedLayoutStr(ll, useHwView);
          }
        });

  py::class_<ttg::WarpSpecializeOp, OpState>(m, "WarpSpecializeOp",
                                             py::module_local())
      .def("get_default_region", &ttg::WarpSpecializeOp::getDefaultRegion,
           ret::reference)
      .def("get_partition_op_holder",
           &ttg::WarpSpecializeOp::getPartitionOpHolder, ret::reference)
      .def("set_requested_registers", [](ttg::WarpSpecializeOp &self,
                                         std::vector<int> &requestedRegisters) {
        self.setRequestedRegisters(requestedRegisters);
      });
}

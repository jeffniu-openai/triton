#include <cuda.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void checkCuda(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;

  const char *name = nullptr;
  const char *message = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);

  std::ostringstream os;
  os << what << " failed";
  if (name)
    os << " (" << name << ")";
  if (message)
    os << ": " << message;
  throw std::runtime_error(os.str());
}

} // namespace

void launch_ptx(const std::string &ptx, const std::string &kernelName,
                torch::Tensor a, torch::Tensor b, torch::Tensor out,
                int64_t dynamicSharedBytes) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(a.scalar_type() == torch::kInt8, "a must be torch.int8");
  TORCH_CHECK(b.scalar_type() == torch::kInt8, "b must be torch.int8");
  TORCH_CHECK(out.scalar_type() == torch::kInt32, "out must be torch.int32");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

  checkCuda(cuInit(0), "cuInit");
  CUcontext context = nullptr;
  checkCuda(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  TORCH_CHECK(context != nullptr, "no current CUDA context; allocate a CUDA torch tensor before launching");

  CUmodule module = nullptr;
  checkCuda(cuModuleLoadData(&module, ptx.c_str()), "cuModuleLoadData");

  try {
    CUfunction function = nullptr;
    checkCuda(cuModuleGetFunction(&function, module, kernelName.c_str()), "cuModuleGetFunction");

    void *aPtr = reinterpret_cast<void *>(a.data_ptr<int8_t>());
    void *bPtr = reinterpret_cast<void *>(b.data_ptr<int8_t>());
    void *outPtr = reinterpret_cast<void *>(out.data_ptr<int32_t>());
    uint64_t zero0 = 0;
    uint64_t zero1 = 0;
    void *params[] = {&aPtr, &bPtr, &outPtr, &zero0, &zero1};

    checkCuda(cuLaunchKernel(function,
                             1, 1, 1,
                             128, 1, 1,
                             static_cast<unsigned int>(dynamicSharedBytes),
                             nullptr,
                             params,
                             nullptr),
              "cuLaunchKernel");
    checkCuda(cuCtxSynchronize(), "cuCtxSynchronize");
  } catch (...) {
    cuModuleUnload(module);
    throw;
  }

  checkCuda(cuModuleUnload(module), "cuModuleUnload");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_ptx", &launch_ptx, "Launch generated sm100 i8 tcgen05.mma PTX");
}

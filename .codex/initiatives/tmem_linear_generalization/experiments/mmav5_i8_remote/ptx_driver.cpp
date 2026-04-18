#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <cstdint>
#include <dlfcn.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

using CUresult = int;
using CUmodule = void *;
using CUfunction = void *;
using CUcontext = void *;
using CUstream = void *;

constexpr CUresult CUDA_SUCCESS = 0;

class CudaDriver {
public:
  CudaDriver() {
    handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error(std::string("dlopen(libcuda.so.1) failed: ") + dlerror());
    load(cuInit, "cuInit");
    load(cuCtxGetCurrent, "cuCtxGetCurrent");
    load(cuModuleLoadData, "cuModuleLoadData");
    load(cuModuleGetFunction, "cuModuleGetFunction");
    load(cuLaunchKernel, "cuLaunchKernel");
    load(cuCtxSynchronize, "cuCtxSynchronize");
    load(cuModuleUnload, "cuModuleUnload");
    load(cuGetErrorName, "cuGetErrorName");
    load(cuGetErrorString, "cuGetErrorString");
  }

  ~CudaDriver() {
    if (handle)
      dlclose(handle);
  }

  CudaDriver(const CudaDriver &) = delete;
  CudaDriver &operator=(const CudaDriver &) = delete;

  CUresult (*cuInit)(unsigned int) = nullptr;
  CUresult (*cuCtxGetCurrent)(CUcontext *) = nullptr;
  CUresult (*cuModuleLoadData)(CUmodule *, const void *) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int, unsigned int,
                             CUstream, void **, void **) = nullptr;
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuModuleUnload)(CUmodule) = nullptr;
  CUresult (*cuGetErrorName)(CUresult, const char **) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;

private:
  template <typename T> void load(T &fn, const char *name) {
    fn = reinterpret_cast<T>(dlsym(handle, name));
    if (!fn)
      throw std::runtime_error(std::string("dlsym(") + name + ") failed");
  }

  void *handle = nullptr;
};

void checkCuda(CudaDriver &driver, CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;

  const char *name = nullptr;
  const char *message = nullptr;
  driver.cuGetErrorName(result, &name);
  driver.cuGetErrorString(result, &message);

  std::ostringstream os;
  os << what << " failed";
  if (name)
    os << " (" << name << ")";
  if (message)
    os << ": " << message;
  throw std::runtime_error(os.str());
}

} // namespace

void launch_ptx(const std::string &moduleImage, const std::string &kernelName,
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

  CudaDriver driver;
  checkCuda(driver, driver.cuInit(0), "cuInit");
  CUcontext context = nullptr;
  checkCuda(driver, driver.cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  TORCH_CHECK(context != nullptr, "no current CUDA context; allocate a CUDA torch tensor before launching");

  CUmodule module = nullptr;
  checkCuda(driver, driver.cuModuleLoadData(&module, moduleImage.data()), "cuModuleLoadData");

  try {
    CUfunction function = nullptr;
    checkCuda(driver, driver.cuModuleGetFunction(&function, module, kernelName.c_str()), "cuModuleGetFunction");

    void *aPtr = reinterpret_cast<void *>(a.data_ptr<int8_t>());
    void *bPtr = reinterpret_cast<void *>(b.data_ptr<int8_t>());
    void *outPtr = reinterpret_cast<void *>(out.data_ptr<int32_t>());
    uint64_t zero0 = 0;
    uint64_t zero1 = 0;
    void *params[] = {&aPtr, &bPtr, &outPtr, &zero0, &zero1};

    checkCuda(driver, driver.cuLaunchKernel(function,
                                            1, 1, 1,
                                            128, 1, 1,
                                            static_cast<unsigned int>(dynamicSharedBytes),
                                            nullptr,
                                            params,
                                            nullptr),
              "cuLaunchKernel");
    checkCuda(driver, driver.cuCtxSynchronize(), "cuCtxSynchronize");
  } catch (...) {
    driver.cuModuleUnload(module);
    throw;
  }

  checkCuda(driver, driver.cuModuleUnload(module), "cuModuleUnload");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("launch_ptx", &launch_ptx, "Launch generated sm100 i8 tcgen05.mma PTX");
}

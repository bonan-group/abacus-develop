#include "source_base/module_device/memory_op.h"
#include "source_base/memory_recorder.h"

#include <base/macros/macros.h>
#include <hip/hip_runtime.h>
#include <thrust/complex.h>

#include <complex>
#include <type_traits>
#include <unordered_map>

#define THREADS_PER_BLOCK 256

namespace base_device
{
namespace memory
{
namespace
{
std::unordered_map<const void*, size_t> gpu_allocation_sizes;
}

template <typename FPTYPE_out, typename FPTYPE_in>
__global__ void cast_memory(FPTYPE_out* out, const FPTYPE_in* in, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    out[idx] = static_cast<FPTYPE_out>(in[idx]);
}

template <typename FPTYPE_out, typename FPTYPE_in>
__global__ void cast_memory(std::complex<FPTYPE_out>* out, const std::complex<FPTYPE_in>* in, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    auto* _out = reinterpret_cast<thrust::complex<FPTYPE_out>*>(out);
    const auto* _in = reinterpret_cast<const thrust::complex<FPTYPE_in>*>(in);
    _out[idx] = static_cast<thrust::complex<FPTYPE_out>>(_in[idx]);
}

template <typename FPTYPE>
void resize_memory_op<FPTYPE, base_device::DEVICE_GPU>::operator()(FPTYPE*& arr,
                                                                   const size_t size,
                                                                   const char* record_in)
{
    if (arr != nullptr)
    {
        delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(arr);
    }
    const size_t bytes = sizeof(FPTYPE) * size;
    hipErrcheck(hipMalloc((void**)&arr, bytes));
    gpu_allocation_sizes[arr] = bytes;
    const std::string record_string = record_in == nullptr ? "no_record" : record_in;
    ModuleBase::Memory::record_gpu_alloc(record_string, bytes);
}

template <typename FPTYPE>
void set_memory_op<FPTYPE, base_device::DEVICE_GPU>::operator()(FPTYPE* arr,
                                                                const int var,
                                                                const size_t size)
{
    hipErrcheck(hipMemset(arr, var, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, base_device::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size)
{
    hipErrcheck(hipMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, hipMemcpyDeviceToHost));
}

template <typename FPTYPE>
void synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_CPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size)
{
    hipErrcheck(hipMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, hipMemcpyHostToDevice));
}

template <typename FPTYPE>
void synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size)
{
    hipErrcheck(hipMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, hipMemcpyDeviceToDevice));
}

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, base_device::DEVICE_GPU, base_device::DEVICE_GPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {

        if (size == 0) {return;}
        const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hipLaunchKernelGGL(cast_memory, dim3(block), dim3(THREADS_PER_BLOCK), 0, 0, arr_out, arr_in, size);
        hipCheckOnDebug();
    }
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, base_device::DEVICE_GPU, base_device::DEVICE_CPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {

        if (size == 0) {return;}
        // No need to cast the memory if the data types are the same.
        if (std::is_same<FPTYPE_out, FPTYPE_in>::value)
        {
            synchronize_memory_op<FPTYPE_out, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(arr_out,
                                                                                                  reinterpret_cast<const FPTYPE_out*>(arr_in),
                                                                                                  size);
            return;
        }
        FPTYPE_in * arr = nullptr;
        hipErrcheck(hipMalloc((void **)&arr, sizeof(FPTYPE_in) * size));
        hipErrcheck(hipMemcpy(arr, arr_in, sizeof(FPTYPE_in) * size, hipMemcpyHostToDevice));
        const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        hipLaunchKernelGGL(cast_memory, dim3(block), dim3(THREADS_PER_BLOCK), 0, 0, arr_out, arr, size);
        hipCheckOnDebug();
        hipErrcheck(hipFree(arr));
    }
};

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, base_device::DEVICE_CPU, base_device::DEVICE_GPU> {
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size) {

        if (size == 0) {return;}
        // No need to cast the memory if the data types are the same.
        if (std::is_same<FPTYPE_out, FPTYPE_in>::value)
        {
            synchronize_memory_op<FPTYPE_out, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(arr_out,
                                                                                                  reinterpret_cast<const FPTYPE_out*>(arr_in),
                                                                                                  size);
            return;
        }
        auto * arr = (FPTYPE_in*) malloc(sizeof(FPTYPE_in) * size);
        hipErrcheck(hipMemcpy(arr, arr_in, sizeof(FPTYPE_in) * size, hipMemcpyDeviceToHost));
        for (int ii = 0; ii < size; ii++) {
            arr_out[ii] = static_cast<FPTYPE_out>(arr[ii]);
        }
        free(arr);
    }
};

template <typename FPTYPE>
void delete_memory_op<FPTYPE, base_device::DEVICE_GPU>::operator()(FPTYPE* arr, const size_t bytes)
{
    if (arr == nullptr)
    {
        return;
    }
    size_t free_bytes = bytes;
    if (free_bytes == 0)
    {
        const auto it = gpu_allocation_sizes.find(arr);
        if (it != gpu_allocation_sizes.end())
        {
            free_bytes = it->second;
        }
    }
    hipErrcheck(hipFree(arr));
    gpu_allocation_sizes.erase(arr);
    if (free_bytes > 0)
    {
        ModuleBase::Memory::record_gpu_free(free_bytes);
    }
}

template struct resize_memory_op<int, base_device::DEVICE_GPU>;
template struct resize_memory_op<float, base_device::DEVICE_GPU>;
template struct resize_memory_op<double, base_device::DEVICE_GPU>;
template struct resize_memory_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>;

template struct set_memory_op<int, base_device::DEVICE_GPU>;
template struct set_memory_op<float, base_device::DEVICE_GPU>;
template struct set_memory_op<double, base_device::DEVICE_GPU>;
template struct set_memory_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct set_memory_op<std::complex<double>, base_device::DEVICE_GPU>;

template struct synchronize_memory_op<int, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<float, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<float, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_op<float, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<float>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<float>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_op<std::complex<float>, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<double>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;

template struct cast_memory_op<float, float, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<double, double, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<float, double, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<double, float, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>,
                               std::complex<float>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>,
                               std::complex<double>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>,
                               std::complex<double>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>,
                               std::complex<float>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>, float, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>, double, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<float, float, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct cast_memory_op<double, double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct cast_memory_op<float, double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct cast_memory_op<double, float, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct cast_memory_op<std::complex<float>,
                               std::complex<float>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_CPU>;
template struct cast_memory_op<std::complex<double>,
                               std::complex<double>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_CPU>;
template struct cast_memory_op<std::complex<float>,
                               std::complex<double>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_CPU>;
template struct cast_memory_op<std::complex<double>,
                               std::complex<float>,
                               base_device::DEVICE_GPU,
                               base_device::DEVICE_CPU>;
template struct cast_memory_op<float, float, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<double, double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<float, double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<double, float, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>,
                               std::complex<float>,
                               base_device::DEVICE_CPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>,
                               std::complex<double>,
                               base_device::DEVICE_CPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<float>,
                               std::complex<double>,
                               base_device::DEVICE_CPU,
                               base_device::DEVICE_GPU>;
template struct cast_memory_op<std::complex<double>,
                               std::complex<float>,
                               base_device::DEVICE_CPU,
                               base_device::DEVICE_GPU>;

template struct delete_memory_op<int, base_device::DEVICE_GPU>;
template struct delete_memory_op<float, base_device::DEVICE_GPU>;
template struct delete_memory_op<double, base_device::DEVICE_GPU>;
template struct delete_memory_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct delete_memory_op<float*, base_device::DEVICE_GPU>;
template struct delete_memory_op<double*, base_device::DEVICE_GPU>;
template struct delete_memory_op<std::complex<float>*, base_device::DEVICE_GPU>;
template struct delete_memory_op<std::complex<double>*, base_device::DEVICE_GPU>;

} // namespace memory
} // end of namespace base_device

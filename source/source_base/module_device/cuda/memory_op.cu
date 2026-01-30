#include "source_base/module_device/memory_op.h"
#include "source_base/memory.h"

#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <complex>
#include <type_traits>
#include <cstdio>
#include <execinfo.h>
#include <cxxabi.h>
#include <cstdlib>

// Set to 1 to enable GPU memory allocation debugging
#define DEBUG_GPU_MEMORY_ALLOC 1

#if DEBUG_GPU_MEMORY_ALLOC
// Helper function to print a short backtrace (caller info)
static void print_caller_info(int max_frames = 6)
{
    void* callstack[16];
    int frames = backtrace(callstack, max_frames + 2);  // +2 to skip this function and resize_memory_op
    char** symbols = backtrace_symbols(callstack, frames);

    if (symbols == nullptr) {
        fprintf(stderr, "    [backtrace unavailable]\n");
        return;
    }

    // Skip first 2 frames (print_caller_info and resize_memory_op)
    for (int i = 2; i < frames; i++)
    {
        // Try to demangle the symbol
        char* symbol = symbols[i];
        char* mangled_start = nullptr;
        char* mangled_end = nullptr;

        // Find the mangled name between '(' and '+'
        for (char* p = symbol; *p; ++p) {
            if (*p == '(') {
                mangled_start = p + 1;
            } else if (*p == '+' && mangled_start) {
                mangled_end = p;
                break;
            }
        }

        if (mangled_start && mangled_end && mangled_end > mangled_start)
        {
            *mangled_end = '\0';
            int status = 0;
            char* demangled = abi::__cxa_demangle(mangled_start, nullptr, nullptr, &status);
            if (status == 0 && demangled) {
                fprintf(stderr, "    #%d %s\n", i - 2, demangled);
                free(demangled);
            } else {
                *mangled_end = '+';  // restore
                fprintf(stderr, "    #%d %s\n", i - 2, symbol);
            }
        }
        else
        {
            fprintf(stderr, "    #%d %s\n", i - 2, symbol);
        }
    }

    free(symbols);
}
#endif

#define THREADS_PER_BLOCK 256

namespace base_device
{
namespace memory
{

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

template <typename FPTYPE_out, typename FPTYPE_in>
__global__ void cast_memory(std::complex<FPTYPE_out>* out, const FPTYPE_in* in, const int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size)
    {
        return;
    }
    auto* _out = reinterpret_cast<thrust::complex<FPTYPE_out>*>(out);
    _out[idx] = static_cast<thrust::complex<FPTYPE_out>>(in[idx]);
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

    const size_t alloc_bytes = sizeof(FPTYPE) * size;
    const char* record_name = record_in ? record_in : "unknown";

#if DEBUG_GPU_MEMORY_ALLOC
    // Query GPU memory status before allocation
    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);

    // Print allocation attempt info
    fprintf(stderr, "[GPU_ALLOC] %s: requesting %.2f MB (free: %.2f MB / %.2f MB total)\n",
            record_name,
            alloc_bytes / (1024.0 * 1024.0),
            free_mem / (1024.0 * 1024.0),
            total_mem / (1024.0 * 1024.0));

    // Print caller backtrace
    print_caller_info(4);
    fflush(stderr);

    // Check if allocation will likely fail
    if (alloc_bytes > free_mem)
    {
        fprintf(stderr, "[GPU_ALLOC] WARNING: %s allocation (%.2f MB) exceeds free memory (%.2f MB)!\n",
                record_name,
                alloc_bytes / (1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0));
        fprintf(stderr, "  Call stack at failure:\n");
        print_caller_info(8);
        fflush(stderr);
    }
#endif

    cudaError_t err = cudaMalloc((void**)&arr, alloc_bytes);

#if DEBUG_GPU_MEMORY_ALLOC
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[GPU_ALLOC] FAILED: %s - %s (requested %.2f MB, free was %.2f MB)\n",
                record_name,
                cudaGetErrorString(err),
                alloc_bytes / (1024.0 * 1024.0),
                free_mem / (1024.0 * 1024.0));
        fprintf(stderr, "  Call stack at failure:\n");
        print_caller_info(10);
        fflush(stderr);
    }
#endif

    cudaErrcheck(err);

    std::string record_string;
    if (record_in != nullptr)
    {
        record_string = record_in;
    }
    else
    {
        record_string = "no_record";
    }

    if (record_string != "no_record")
    {
        ModuleBase::Memory::record_gpu(record_string, alloc_bytes);
    }
}

template <typename FPTYPE>
void set_memory_op<FPTYPE, base_device::DEVICE_GPU>::operator()(FPTYPE* arr,
                                                                const int var,
                                                                const size_t size)
{
    cudaErrcheck(cudaMemset(arr, var, sizeof(FPTYPE) * size));
}

template <typename FPTYPE>
void set_memory_2d_op<FPTYPE, base_device::DEVICE_GPU>::operator()(FPTYPE* arr,
                                                                   const size_t pitch,
                                                                   const int var,
                                                                   const size_t width,
                                                                   const size_t height)
{
    cudaErrcheck(cudaMemset2D(arr, sizeof(FPTYPE) * pitch , var, sizeof(FPTYPE) * width, height));
}

template <typename FPTYPE>
void synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, base_device::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size)
{
    cudaErrcheck(cudaMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, cudaMemcpyDeviceToHost));
}

template <typename FPTYPE>
void synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_CPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size)
{
    cudaErrcheck(cudaMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, cudaMemcpyHostToDevice));
}

template <typename FPTYPE>
void synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const FPTYPE* arr_in,
    const size_t size)
{
    cudaErrcheck(cudaMemcpy(arr_out, arr_in, sizeof(FPTYPE) * size, cudaMemcpyDeviceToDevice));
}

template <typename FPTYPE>
void synchronize_memory_2d_op<FPTYPE, base_device::DEVICE_CPU, base_device::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const size_t dpitch,
    const FPTYPE* arr_in,
    const size_t spitch,
    const size_t width,
    const size_t height)
{
    cudaErrcheck(cudaMemcpy2D(arr_out, dpitch * sizeof(FPTYPE), arr_in, spitch * sizeof(FPTYPE), width * sizeof(FPTYPE), height, cudaMemcpyDeviceToHost));
}

template <typename FPTYPE>
void synchronize_memory_2d_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_CPU>::operator()(
    FPTYPE* arr_out,
    const size_t dpitch,
    const FPTYPE* arr_in,
    const size_t spitch,
    const size_t width,
    const size_t height)
{
    cudaErrcheck(cudaMemcpy2D(arr_out, dpitch * sizeof(FPTYPE), arr_in, spitch * sizeof(FPTYPE), width * sizeof(FPTYPE), height, cudaMemcpyHostToDevice));
}

template <typename FPTYPE>
void synchronize_memory_2d_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_GPU>::operator()(
    FPTYPE* arr_out,
    const size_t dpitch,
    const FPTYPE* arr_in,
    const size_t spitch,
    const size_t width,
    const size_t height)
{
    cudaErrcheck(cudaMemcpy2D(arr_out, dpitch * sizeof(FPTYPE), arr_in, spitch * sizeof(FPTYPE), width * sizeof(FPTYPE), height, cudaMemcpyDeviceToDevice));
}

template <typename FPTYPE_out, typename FPTYPE_in>
struct cast_memory_op<FPTYPE_out, FPTYPE_in, base_device::DEVICE_GPU, base_device::DEVICE_GPU>
{
    void operator()(FPTYPE_out* arr_out,
                    const FPTYPE_in* arr_in,
                    const size_t size)
    {
        if (size == 0)
        {
            return;
        }
        const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cast_memory<<<block, THREADS_PER_BLOCK>>>(arr_out, arr_in, size);

        cudaCheckOnDebug();
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
        cudaErrcheck(cudaMalloc((void **)&arr, sizeof(FPTYPE_in) * size));
        cudaErrcheck(cudaMemcpy(arr, arr_in, sizeof(FPTYPE_in) * size, cudaMemcpyHostToDevice));
        const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        cast_memory<<<block, THREADS_PER_BLOCK>>>(arr_out, arr, size);
        cudaCheckOnDebug();
        cudaErrcheck(cudaFree(arr));
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
        cudaErrcheck(cudaMemcpy(arr, arr_in, sizeof(FPTYPE_in) * size, cudaMemcpyDeviceToHost));
        for (int ii = 0; ii < size; ii++) {
            arr_out[ii] = static_cast<FPTYPE_out>(arr[ii]);
        }
        free(arr);
    }
};

template <typename FPTYPE>
void delete_memory_op<FPTYPE, base_device::DEVICE_GPU>::operator()(FPTYPE* arr)
{
    cudaErrcheck(cudaFree(arr));
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

template struct set_memory_2d_op<int, base_device::DEVICE_GPU>;
template struct set_memory_2d_op<float, base_device::DEVICE_GPU>;
template struct set_memory_2d_op<double, base_device::DEVICE_GPU>;
template struct set_memory_2d_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct set_memory_2d_op<std::complex<double>, base_device::DEVICE_GPU>;

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

template struct synchronize_memory_2d_op<int, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_2d_op<int, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<float, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<float, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_2d_op<float, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_2d_op<double, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<std::complex<float>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<std::complex<float>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_2d_op<std::complex<float>, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<std::complex<double>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
template struct synchronize_memory_2d_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
template struct synchronize_memory_2d_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;

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
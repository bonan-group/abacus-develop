#include "source_base/module_mixing/kernels/mixing_op.h"
#include "source_base/module_device/memory_op.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <base/macros/macros.h>

#define THREADS_PER_BLOCK 256

namespace mixing {

// Get or create cuBLAS handle (defined in math_kernel_op.cu)
namespace {
    cublasHandle_t& get_cublas_handle() {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        return handle;
    }
}

//==========================================================
// CUDA Kernels
//==========================================================

// Vector subtraction: out[i] = a[i] - b[i]
template <typename FPTYPE>
__global__ void vector_subtract_kernel(
    FPTYPE* out,
    const FPTYPE* a,
    const FPTYPE* b,
    const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        out[i] = a[i] - b[i];
    }
}

// Vector AXPY: y[i] = x[i] + alpha * z[i]
template <typename FPTYPE>
__global__ void vector_axpy_kernel(
    FPTYPE* y,
    const FPTYPE* x,
    const FPTYPE alpha,
    const FPTYPE* z,
    const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        y[i] = x[i] + alpha * z[i];
    }
}

// Complex Vector AXPY: y[i] = x[i] + alpha * z[i]
// This kernel handles aliasing (y == z) correctly by reading z[i] before writing y[i]
template <typename FPTYPE>
__global__ void vector_axpy_complex_kernel(
    thrust::complex<FPTYPE>* y,
    const thrust::complex<FPTYPE>* x,
    const thrust::complex<FPTYPE> alpha,
    const thrust::complex<FPTYPE>* z,
    const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        // Read z[i] first (important when y == z to avoid aliasing bug)
        thrust::complex<FPTYPE> z_val = z[i];
        y[i] = x[i] + alpha * z_val;
    }
}

// Vector scale: x[i] *= factor
template <typename FPTYPE>
__global__ void vector_scale_kernel(
    FPTYPE* x,
    const FPTYPE factor,
    const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        x[i] *= factor;
    }
}

// Vector accumulate subtraction: out[i] -= in[i]
template <typename FPTYPE>
__global__ void vector_acc_subtract_kernel(
    FPTYPE* out,
    const FPTYPE* in,
    const int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length)
    {
        out[i] -= in[i];
    }
}

// Inner product kernel with block reduction
template <typename FPTYPE>
__global__ void inner_product_real_kernel(
    const FPTYPE* a,
    const FPTYPE* b,
    FPTYPE* partial_sums,
    const int length)
{
    __shared__ FPTYPE sdata[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute local product
    FPTYPE local_sum = (i < length) ? (a[i] * b[i]) : FPTYPE(0);
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
    {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Complex inner product kernel: sum_i conj(a[i]) * b[i]
template <typename FPTYPE>
__global__ void inner_product_complex_kernel(
    const thrust::complex<FPTYPE>* a,
    const thrust::complex<FPTYPE>* b,
    FPTYPE* partial_real,
    FPTYPE* partial_imag,
    const int length)
{
    __shared__ FPTYPE sdata_real[THREADS_PER_BLOCK];
    __shared__ FPTYPE sdata_imag[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and compute local product: conj(a) * b
    FPTYPE local_real = FPTYPE(0);
    FPTYPE local_imag = FPTYPE(0);
    if (i < length)
    {
        thrust::complex<FPTYPE> prod = thrust::conj(a[i]) * b[i];
        local_real = prod.real();
        local_imag = prod.imag();
    }
    sdata_real[tid] = local_real;
    sdata_imag[tid] = local_imag;
    __syncthreads();

    // Block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata_real[tid] += sdata_real[tid + s];
            sdata_imag[tid] += sdata_imag[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
    {
        partial_real[blockIdx.x] = sdata_real[0];
        partial_imag[blockIdx.x] = sdata_imag[0];
    }
}

//==========================================================
// GPU Operator Implementations
//==========================================================

// Vector subtraction
template <typename FPTYPE>
void vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    FPTYPE* out,
    const FPTYPE* a,
    const FPTYPE* b,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_subtract_kernel<<<block, THREADS_PER_BLOCK>>>(out, a, b, length);
    cudaCheckOnDebug();
}

// Vector AXPY
template <typename FPTYPE>
void vector_axpy_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    FPTYPE* y,
    const FPTYPE* x,
    const FPTYPE alpha,
    const FPTYPE* z,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_axpy_kernel<<<block, THREADS_PER_BLOCK>>>(y, x, alpha, z, length);
    cudaCheckOnDebug();
}

// Vector scale
template <typename FPTYPE>
void vector_scale_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    FPTYPE* x,
    const FPTYPE factor,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_scale_kernel<<<block, THREADS_PER_BLOCK>>>(x, factor, length);
    cudaCheckOnDebug();
}

// Vector accumulate subtraction
template <typename FPTYPE>
void vector_acc_subtract_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    FPTYPE* out,
    const FPTYPE* in,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_acc_subtract_kernel<<<block, THREADS_PER_BLOCK>>>(out, in, length);
    cudaCheckOnDebug();
}

// Vector copy
template <typename FPTYPE>
void vector_copy_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    FPTYPE* out,
    const FPTYPE* in,
    const int length)
{
    cudaMemcpy(out, in, length * sizeof(FPTYPE), cudaMemcpyDeviceToDevice);
    cudaCheckOnDebug();
}

// Inner product for real double
template <>
double inner_product_op<double, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const double* a,
    const double* b,
    const int length,
    double* workspace)
{
    const int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel for partial sums
    inner_product_real_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(a, b, workspace, length);
    cudaCheckOnDebug();

    // Final reduction using thrust
    thrust::device_ptr<double> dev_ptr(workspace);
    return thrust::reduce(dev_ptr, dev_ptr + num_blocks);
}

// Inner product for real float
template <>
float inner_product_op<float, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const float* a,
    const float* b,
    const int length,
    float* workspace)
{
    const int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel for partial sums
    inner_product_real_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(a, b, workspace, length);
    cudaCheckOnDebug();

    // Final reduction using thrust
    thrust::device_ptr<float> dev_ptr(workspace);
    return thrust::reduce(dev_ptr, dev_ptr + num_blocks);
}

// Inner product for complex double
template <>
std::complex<double> inner_product_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const std::complex<double>* a,
    const std::complex<double>* b,
    const int length,
    std::complex<double>* workspace)
{
    const int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Use workspace for partial sums (real and imag parts)
    double* partial_real = reinterpret_cast<double*>(workspace);
    double* partial_imag = partial_real + num_blocks;

    // Launch kernel for partial sums
    inner_product_complex_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const thrust::complex<double>*>(a),
        reinterpret_cast<const thrust::complex<double>*>(b),
        partial_real, partial_imag, length);
    cudaCheckOnDebug();

    // Final reduction using thrust
    thrust::device_ptr<double> dev_real(partial_real);
    thrust::device_ptr<double> dev_imag(partial_imag);
    double real_sum = thrust::reduce(dev_real, dev_real + num_blocks);
    double imag_sum = thrust::reduce(dev_imag, dev_imag + num_blocks);

    return std::complex<double>(real_sum, imag_sum);
}

// Inner product for complex float
template <>
std::complex<float> inner_product_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const std::complex<float>* a,
    const std::complex<float>* b,
    const int length,
    std::complex<float>* workspace)
{
    const int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Use workspace for partial sums (real and imag parts)
    float* partial_real = reinterpret_cast<float*>(workspace);
    float* partial_imag = partial_real + num_blocks;

    // Launch kernel for partial sums
    inner_product_complex_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const thrust::complex<float>*>(a),
        reinterpret_cast<const thrust::complex<float>*>(b),
        partial_real, partial_imag, length);
    cudaCheckOnDebug();

    // Final reduction using thrust
    thrust::device_ptr<float> dev_real(partial_real);
    thrust::device_ptr<float> dev_imag(partial_imag);
    float real_sum = thrust::reduce(dev_real, dev_real + num_blocks);
    float imag_sum = thrust::reduce(dev_imag, dev_imag + num_blocks);

    return std::complex<float>(real_sum, imag_sum);
}

// GEMV for double
template <>
void gemv_op<double, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const char trans,
    const int m,
    const int n,
    const double alpha,
    const double* A,
    const int lda,
    const double* x,
    const int incx,
    const double beta,
    double* y,
    const int incy)
{
    cublasOperation_t cu_trans = (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasHandle_t& handle = get_cublas_handle();
    cublasDgemv(handle, cu_trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    cudaCheckOnDebug();
}

// GEMV for float
template <>
void gemv_op<float, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const char trans,
    const int m,
    const int n,
    const float alpha,
    const float* A,
    const int lda,
    const float* x,
    const int incx,
    const float beta,
    float* y,
    const int incy)
{
    cublasOperation_t cu_trans = (trans == 'N' || trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasHandle_t& handle = get_cublas_handle();
    cublasSgemv(handle, cu_trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
    cudaCheckOnDebug();
}

// GEMV for complex double
template <>
void gemv_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const char trans,
    const int m,
    const int n,
    const std::complex<double> alpha,
    const std::complex<double>* A,
    const int lda,
    const std::complex<double>* x,
    const int incx,
    const std::complex<double> beta,
    std::complex<double>* y,
    const int incy)
{
    cublasOperation_t cu_trans;
    if (trans == 'N' || trans == 'n') cu_trans = CUBLAS_OP_N;
    else if (trans == 'T' || trans == 't') cu_trans = CUBLAS_OP_T;
    else cu_trans = CUBLAS_OP_C; // Conjugate transpose

    cuDoubleComplex cu_alpha = make_cuDoubleComplex(alpha.real(), alpha.imag());
    cuDoubleComplex cu_beta = make_cuDoubleComplex(beta.real(), beta.imag());

    cublasHandle_t& handle = get_cublas_handle();
    cublasZgemv(handle, cu_trans, m, n, &cu_alpha,
                reinterpret_cast<const cuDoubleComplex*>(A), lda,
                reinterpret_cast<const cuDoubleComplex*>(x), incx,
                &cu_beta, reinterpret_cast<cuDoubleComplex*>(y), incy);
    cudaCheckOnDebug();
}

// GEMV for complex float
template <>
void gemv_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const char trans,
    const int m,
    const int n,
    const std::complex<float> alpha,
    const std::complex<float>* A,
    const int lda,
    const std::complex<float>* x,
    const int incx,
    const std::complex<float> beta,
    std::complex<float>* y,
    const int incy)
{
    cublasOperation_t cu_trans;
    if (trans == 'N' || trans == 'n') cu_trans = CUBLAS_OP_N;
    else if (trans == 'T' || trans == 't') cu_trans = CUBLAS_OP_T;
    else cu_trans = CUBLAS_OP_C; // Conjugate transpose

    cuFloatComplex cu_alpha = make_cuFloatComplex(alpha.real(), alpha.imag());
    cuFloatComplex cu_beta = make_cuFloatComplex(beta.real(), beta.imag());

    cublasHandle_t& handle = get_cublas_handle();
    cublasCgemv(handle, cu_trans, m, n, &cu_alpha,
                reinterpret_cast<const cuFloatComplex*>(A), lda,
                reinterpret_cast<const cuFloatComplex*>(x), incx,
                &cu_beta, reinterpret_cast<cuFloatComplex*>(y), incy);
    cudaCheckOnDebug();
}

//==========================================================
// Explicit Template Instantiations
//==========================================================

// Complex type specializations for kernels that need thrust::complex
template <>
void vector_subtract_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<double>* out,
    const std::complex<double>* a,
    const std::complex<double>* b,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_subtract_kernel<<<block, THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<double>*>(out),
        reinterpret_cast<const thrust::complex<double>*>(a),
        reinterpret_cast<const thrust::complex<double>*>(b),
        length);
    cudaCheckOnDebug();
}

template <>
void vector_subtract_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<float>* out,
    const std::complex<float>* a,
    const std::complex<float>* b,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_subtract_kernel<<<block, THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<float>*>(out),
        reinterpret_cast<const thrust::complex<float>*>(a),
        reinterpret_cast<const thrust::complex<float>*>(b),
        length);
    cudaCheckOnDebug();
}

template <>
void vector_axpy_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<double>* y,
    const std::complex<double>* x,
    const std::complex<double> alpha,
    const std::complex<double>* z,
    const int length)
{
    // Use direct CUDA kernel to handle aliasing (y == z) correctly
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    thrust::complex<double> thrust_alpha(alpha.real(), alpha.imag());
    vector_axpy_complex_kernel<<<block, THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<double>*>(y),
        reinterpret_cast<const thrust::complex<double>*>(x),
        thrust_alpha,
        reinterpret_cast<const thrust::complex<double>*>(z),
        length);
    cudaCheckOnDebug();
}

template <>
void vector_axpy_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<float>* y,
    const std::complex<float>* x,
    const std::complex<float> alpha,
    const std::complex<float>* z,
    const int length)
{
    // Use direct CUDA kernel to handle aliasing (y == z) correctly
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    thrust::complex<float> thrust_alpha(alpha.real(), alpha.imag());
    vector_axpy_complex_kernel<<<block, THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<float>*>(y),
        reinterpret_cast<const thrust::complex<float>*>(x),
        thrust_alpha,
        reinterpret_cast<const thrust::complex<float>*>(z),
        length);
    cudaCheckOnDebug();
}

template <>
void vector_scale_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<double>* x,
    const std::complex<double> factor,
    const int length)
{
    cuDoubleComplex cu_factor = make_cuDoubleComplex(factor.real(), factor.imag());
    cublasHandle_t& handle = get_cublas_handle();
    cublasZscal(handle, length, &cu_factor, reinterpret_cast<cuDoubleComplex*>(x), 1);
    cudaCheckOnDebug();
}

template <>
void vector_scale_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<float>* x,
    const std::complex<float> factor,
    const int length)
{
    cuFloatComplex cu_factor = make_cuFloatComplex(factor.real(), factor.imag());
    cublasHandle_t& handle = get_cublas_handle();
    cublasCscal(handle, length, &cu_factor, reinterpret_cast<cuFloatComplex*>(x), 1);
    cudaCheckOnDebug();
}

template <>
void vector_acc_subtract_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<double>* out,
    const std::complex<double>* in,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_acc_subtract_kernel<<<block, THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<double>*>(out),
        reinterpret_cast<const thrust::complex<double>*>(in),
        length);
    cudaCheckOnDebug();
}

template <>
void vector_acc_subtract_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<float>* out,
    const std::complex<float>* in,
    const int length)
{
    const int block = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_acc_subtract_kernel<<<block, THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<float>*>(out),
        reinterpret_cast<const thrust::complex<float>*>(in),
        length);
    cudaCheckOnDebug();
}

template <>
void vector_copy_op<std::complex<double>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<double>* out,
    const std::complex<double>* in,
    const int length)
{
    cudaMemcpy(out, in, length * sizeof(std::complex<double>), cudaMemcpyDeviceToDevice);
    cudaCheckOnDebug();
}

template <>
void vector_copy_op<std::complex<float>, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<float>* out,
    const std::complex<float>* in,
    const int length)
{
    cudaMemcpy(out, in, length * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice);
    cudaCheckOnDebug();
}

// Explicit template instantiations for real types
template struct vector_subtract_op<double, base_device::DEVICE_GPU>;
template struct vector_subtract_op<float, base_device::DEVICE_GPU>;

template struct vector_axpy_op<double, base_device::DEVICE_GPU>;
template struct vector_axpy_op<float, base_device::DEVICE_GPU>;

template struct vector_scale_op<double, base_device::DEVICE_GPU>;
template struct vector_scale_op<float, base_device::DEVICE_GPU>;

template struct vector_acc_subtract_op<double, base_device::DEVICE_GPU>;
template struct vector_acc_subtract_op<float, base_device::DEVICE_GPU>;

template struct vector_copy_op<double, base_device::DEVICE_GPU>;
template struct vector_copy_op<float, base_device::DEVICE_GPU>;

} // namespace mixing

#include "mixing_op.h"
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mixing {

//==========================================================
// CPU Implementations
//==========================================================

// Vector subtraction: out[i] = a[i] - b[i]
template <typename FPTYPE>
void vector_subtract_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    FPTYPE* out,
    const FPTYPE* a,
    const FPTYPE* b,
    const int length)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
    for (int i = 0; i < length; ++i)
    {
        out[i] = a[i] - b[i];
    }
}

// Vector AXPY: y[i] = x[i] + alpha * z[i]
template <typename FPTYPE>
void vector_axpy_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    FPTYPE* y,
    const FPTYPE* x,
    const FPTYPE alpha,
    const FPTYPE* z,
    const int length)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
    for (int i = 0; i < length; ++i)
    {
        y[i] = x[i] + alpha * z[i];
    }
}

// Vector scale: x[i] *= factor
template <typename FPTYPE>
void vector_scale_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    FPTYPE* x,
    const FPTYPE factor,
    const int length)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
    for (int i = 0; i < length; ++i)
    {
        x[i] *= factor;
    }
}

// Vector accumulate subtraction: out[i] -= in[i]
template <typename FPTYPE>
void vector_acc_subtract_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    FPTYPE* out,
    const FPTYPE* in,
    const int length)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
    for (int i = 0; i < length; ++i)
    {
        out[i] -= in[i];
    }
}

// Vector copy: out[i] = in[i]
template <typename FPTYPE>
void vector_copy_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    FPTYPE* out,
    const FPTYPE* in,
    const int length)
{
    std::memcpy(out, in, length * sizeof(FPTYPE));
}

// Inner product for real types
template <>
double inner_product_op<double, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const double* a,
    const double* b,
    const int length,
    double* workspace)
{
    double result = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:result)
#endif
    for (int i = 0; i < length; ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

template <>
float inner_product_op<float, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const float* a,
    const float* b,
    const int length,
    float* workspace)
{
    float result = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:result)
#endif
    for (int i = 0; i < length; ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

// Inner product for complex types: sum_i conj(a[i]) * b[i]
template <>
std::complex<double> inner_product_op<std::complex<double>, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<double>* a,
    const std::complex<double>* b,
    const int length,
    std::complex<double>* workspace)
{
    double real_part = 0.0;
    double imag_part = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:real_part, imag_part)
#endif
    for (int i = 0; i < length; ++i)
    {
        std::complex<double> prod = std::conj(a[i]) * b[i];
        real_part += prod.real();
        imag_part += prod.imag();
    }
    return std::complex<double>(real_part, imag_part);
}

template <>
std::complex<float> inner_product_op<std::complex<float>, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<float>* a,
    const std::complex<float>* b,
    const int length,
    std::complex<float>* workspace)
{
    float real_part = 0.0f;
    float imag_part = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:real_part, imag_part)
#endif
    for (int i = 0; i < length; ++i)
    {
        std::complex<float> prod = std::conj(a[i]) * b[i];
        real_part += prod.real();
        imag_part += prod.imag();
    }
    return std::complex<float>(real_part, imag_part);
}

// GEMV: y = alpha * A * x + beta * y
// A is column-major [m x n], x is [n], y is [m]
template <typename FPTYPE>
void gemv_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const char trans,
    const int m,
    const int n,
    const FPTYPE alpha,
    const FPTYPE* A,
    const int lda,
    const FPTYPE* x,
    const int incx,
    const FPTYPE beta,
    FPTYPE* y,
    const int incy)
{
    // Simple implementation for small matrices
    // For large matrices, use BLAS
    if (trans == 'N' || trans == 'n')
    {
        // y = alpha * A * x + beta * y
        for (int i = 0; i < m; ++i)
        {
            FPTYPE sum = FPTYPE(0);
            for (int j = 0; j < n; ++j)
            {
                sum += A[i + j * lda] * x[j * incx];
            }
            y[i * incy] = alpha * sum + beta * y[i * incy];
        }
    }
    else
    {
        // y = alpha * A^T * x + beta * y
        for (int j = 0; j < n; ++j)
        {
            FPTYPE sum = FPTYPE(0);
            for (int i = 0; i < m; ++i)
            {
                sum += A[i + j * lda] * x[i * incx];
            }
            y[j * incy] = alpha * sum + beta * y[j * incy];
        }
    }
}

// Explicit instantiations
template struct vector_subtract_op<double, base_device::DEVICE_CPU>;
template struct vector_subtract_op<float, base_device::DEVICE_CPU>;
template struct vector_subtract_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct vector_subtract_op<std::complex<float>, base_device::DEVICE_CPU>;

template struct vector_axpy_op<double, base_device::DEVICE_CPU>;
template struct vector_axpy_op<float, base_device::DEVICE_CPU>;
template struct vector_axpy_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct vector_axpy_op<std::complex<float>, base_device::DEVICE_CPU>;

template struct vector_scale_op<double, base_device::DEVICE_CPU>;
template struct vector_scale_op<float, base_device::DEVICE_CPU>;
template struct vector_scale_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct vector_scale_op<std::complex<float>, base_device::DEVICE_CPU>;

template struct vector_acc_subtract_op<double, base_device::DEVICE_CPU>;
template struct vector_acc_subtract_op<float, base_device::DEVICE_CPU>;
template struct vector_acc_subtract_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct vector_acc_subtract_op<std::complex<float>, base_device::DEVICE_CPU>;

template struct vector_copy_op<double, base_device::DEVICE_CPU>;
template struct vector_copy_op<float, base_device::DEVICE_CPU>;
template struct vector_copy_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct vector_copy_op<std::complex<float>, base_device::DEVICE_CPU>;

template struct gemv_op<double, base_device::DEVICE_CPU>;
template struct gemv_op<float, base_device::DEVICE_CPU>;
template struct gemv_op<std::complex<double>, base_device::DEVICE_CPU>;
template struct gemv_op<std::complex<float>, base_device::DEVICE_CPU>;

} // namespace mixing

#ifndef MIXING_OP_H
#define MIXING_OP_H

#include <complex>
#include "source_base/module_device/types.h"

namespace mixing {

//==========================================================
// GPU Mixing Kernel Operators
// These operators provide device-portable implementations
// of common mixing operations
//==========================================================

/// Vector subtraction: out[i] = a[i] - b[i]
template <typename FPTYPE, typename Device>
struct vector_subtract_op {
    void operator()(
        const Device* ctx,
        FPTYPE* out,
        const FPTYPE* a,
        const FPTYPE* b,
        const int length);
};

/// Vector AXPY: y[i] = x[i] + alpha * z[i]
template <typename FPTYPE, typename Device>
struct vector_axpy_op {
    void operator()(
        const Device* ctx,
        FPTYPE* y,
        const FPTYPE* x,
        const FPTYPE alpha,
        const FPTYPE* z,
        const int length);
};

/// Vector scale: x[i] *= factor
template <typename FPTYPE, typename Device>
struct vector_scale_op {
    void operator()(
        const Device* ctx,
        FPTYPE* x,
        const FPTYPE factor,
        const int length);
};

/// Vector accumulate subtraction: out[i] -= in[i]
template <typename FPTYPE, typename Device>
struct vector_acc_subtract_op {
    void operator()(
        const Device* ctx,
        FPTYPE* out,
        const FPTYPE* in,
        const int length);
};

/// Vector copy: out[i] = in[i]
template <typename FPTYPE, typename Device>
struct vector_copy_op {
    void operator()(
        const Device* ctx,
        FPTYPE* out,
        const FPTYPE* in,
        const int length);
};

/// Inner product: sum_i conj(a[i]) * b[i]
/// For complex types, returns complex result
/// For real types, returns real result
template <typename FPTYPE, typename Device>
struct inner_product_op {
    FPTYPE operator()(
        const Device* ctx,
        const FPTYPE* a,
        const FPTYPE* b,
        const int length,
        FPTYPE* workspace);  // GPU reduction workspace
};

/// GEMV: y = alpha * A * x + beta * y
/// A is column-major [m x n], x is [n], y is [m]
template <typename FPTYPE, typename Device>
struct gemv_op {
    void operator()(
        const Device* ctx,
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
        const int incy);
};

//==========================================================
// CPU Specializations (default)
//==========================================================

template <typename FPTYPE>
struct vector_subtract_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        FPTYPE* out,
        const FPTYPE* a,
        const FPTYPE* b,
        const int length);
};

template <typename FPTYPE>
struct vector_axpy_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        FPTYPE* y,
        const FPTYPE* x,
        const FPTYPE alpha,
        const FPTYPE* z,
        const int length);
};

template <typename FPTYPE>
struct vector_scale_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        FPTYPE* x,
        const FPTYPE factor,
        const int length);
};

template <typename FPTYPE>
struct vector_acc_subtract_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        FPTYPE* out,
        const FPTYPE* in,
        const int length);
};

template <typename FPTYPE>
struct vector_copy_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        FPTYPE* out,
        const FPTYPE* in,
        const int length);
};

template <typename FPTYPE>
struct inner_product_op<FPTYPE, base_device::DEVICE_CPU> {
    FPTYPE operator()(
        const base_device::DEVICE_CPU* ctx,
        const FPTYPE* a,
        const FPTYPE* b,
        const int length,
        FPTYPE* workspace);
};

template <typename FPTYPE>
struct gemv_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
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
        const int incy);
};

//==========================================================
// GPU Specializations
//==========================================================

#if __CUDA || __UT_USE_CUDA

template <typename FPTYPE>
struct vector_subtract_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        FPTYPE* out,
        const FPTYPE* a,
        const FPTYPE* b,
        const int length);
};

template <typename FPTYPE>
struct vector_axpy_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        FPTYPE* y,
        const FPTYPE* x,
        const FPTYPE alpha,
        const FPTYPE* z,
        const int length);
};

template <typename FPTYPE>
struct vector_scale_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        FPTYPE* x,
        const FPTYPE factor,
        const int length);
};

template <typename FPTYPE>
struct vector_acc_subtract_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        FPTYPE* out,
        const FPTYPE* in,
        const int length);
};

template <typename FPTYPE>
struct vector_copy_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        FPTYPE* out,
        const FPTYPE* in,
        const int length);
};

template <typename FPTYPE>
struct inner_product_op<FPTYPE, base_device::DEVICE_GPU> {
    FPTYPE operator()(
        const base_device::DEVICE_GPU* ctx,
        const FPTYPE* a,
        const FPTYPE* b,
        const int length,
        FPTYPE* workspace);
};

template <typename FPTYPE>
struct gemv_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
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
        const int incy);
};

#endif // __CUDA || __UT_USE_CUDA

} // namespace mixing

#endif // MIXING_OP_H

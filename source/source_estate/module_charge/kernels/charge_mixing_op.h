#ifndef CHARGE_MIXING_OP_H
#define CHARGE_MIXING_OP_H

#include <complex>
#include "source_base/module_device/types.h"

namespace elecstate {

/// Kerker screening in reciprocal space: drhog[ig] *= gg[ig]/(gg[ig]+gg0)
/// This applies a high-pass filter to the charge density difference to accelerate SCF convergence.
template <typename FPTYPE, typename Device>
struct kerker_screen_recip_op {
    void operator()(
        const Device* ctx,
        std::complex<FPTYPE>* drhog,  // [nspin * npw] - charge density difference in reciprocal space
        const FPTYPE* gg,             // [npw] - |G|^2 values
        const FPTYPE gg0,             // Kerker screening parameter
        const FPTYPE gg0_min,         // Minimum filter coefficient
        const int npw,                // Number of plane waves
        const int nspin);             // Number of spin channels
};

/// Batched inner products with 1/G^2 weight for Hartree-like functional.
/// Computes result[i, j] = <lhs_i, rhs_j> for contiguous vector slots.
template <typename FPTYPE, typename Device>
struct inner_product_recip_hartree_batch_op {
    void operator()(
        const Device* ctx,
        const std::complex<FPTYPE>* lhs,      // [nlhs * npw] - left vector slots
        const std::complex<FPTYPE>* rhs,      // [nrhs * npw] - right vector slots
        const FPTYPE* gg,                     // [npw] - |G|^2 values
        const int npw,                        // Number of plane waves
        const int nlhs,                       // Number of left vector slots
        const int nrhs,                       // Number of right vector slots
        const int ig_gge0,                    // Index of G=0 (to skip)
        const FPTYPE tpiba2,                  // (2*pi/a)^2 prefactor
        FPTYPE* result,                       // [nlhs * nrhs] output on device
        FPTYPE* workspace);                   // [nlhs * nrhs * num_blocks]
};

/// Spin-aware batched inner products matching Charge_Mixing::inner_product_recip_hartree
/// for packed reciprocal mixing vectors.
template <typename FPTYPE, typename Device>
struct inner_product_recip_hartree_spin_batch_op {
    void operator()(
        const Device* ctx,
        const std::complex<FPTYPE>* lhs,
        const std::complex<FPTYPE>* rhs,
        const FPTYPE* gg,
        const int npw,
        const int nspin,
        const int nlhs,
        const int nrhs,
        const int ig_gge0,
        const bool gamma_only,
        const bool include_magnetism,
        const FPTYPE charge_fac,
        const FPTYPE mag_fac,
        FPTYPE* result,
        FPTYPE* workspace);
};

/// Pack spin-resolved reciprocal densities into the CPU mixing basis.
/// nspin=2: {rho_up, rho_down} -> {rho_up + rho_down, rho_up - rho_down}.
/// nspin=4: {rho, mx, my, mz} is already in the mixing basis and is copied.
template <typename FPTYPE, typename Device>
struct pack_spin_recip_op {
    void operator()(
        const Device* ctx,
        std::complex<FPTYPE>* packed,
        const std::complex<FPTYPE>* spin_data,
        const int npw,
        const int nspin);
};

/// Unpack reciprocal mixing-basis data back into spin-resolved densities.
/// nspin=2: {rho, mag} -> {0.5 * (rho + mag), 0.5 * (rho - mag)}.
/// nspin=4: {rho, mx, my, mz} is copied back unchanged.
template <typename FPTYPE, typename Device>
struct unpack_spin_recip_op {
    void operator()(
        const Device* ctx,
        std::complex<FPTYPE>* spin_data,
        const std::complex<FPTYPE>* packed,
        const int npw,
        const int nspin);
};

/// Split spin-resolved dense-grid reciprocal data into contiguous smooth and
/// high-frequency buffers.
template <typename FPTYPE, typename Device>
struct split_double_grid_recip_op {
    void operator()(
        const Device* ctx,
        std::complex<FPTYPE>* smooth,
        std::complex<FPTYPE>* high_frequency,
        const std::complex<FPTYPE>* dense,
        const int smooth_npw,
        const int dense_npw,
        const int nspin);
};

/// Combine contiguous smooth and high-frequency buffers back into spin-resolved
/// dense-grid reciprocal data.
template <typename FPTYPE, typename Device>
struct combine_double_grid_recip_op {
    void operator()(
        const Device* ctx,
        std::complex<FPTYPE>* dense,
        const std::complex<FPTYPE>* smooth,
        const std::complex<FPTYPE>* high_frequency,
        const int smooth_npw,
        const int dense_npw,
        const int nspin);
};

// CPU specializations
template <typename FPTYPE>
struct kerker_screen_recip_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        std::complex<FPTYPE>* drhog,
        const FPTYPE* gg,
        const FPTYPE gg0,
        const FPTYPE gg0_min,
        const int npw,
        const int nspin);
};

template <typename FPTYPE>
struct pack_spin_recip_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        std::complex<FPTYPE>* packed,
        const std::complex<FPTYPE>* spin_data,
        const int npw,
        const int nspin);
};

template <typename FPTYPE>
struct unpack_spin_recip_op<FPTYPE, base_device::DEVICE_CPU> {
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        std::complex<FPTYPE>* spin_data,
        const std::complex<FPTYPE>* packed,
        const int npw,
        const int nspin);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
// GPU specializations
template <typename FPTYPE>
struct kerker_screen_recip_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        std::complex<FPTYPE>* drhog,
        const FPTYPE* gg,
        const FPTYPE gg0,
        const FPTYPE gg0_min,
        const int npw,
        const int nspin);
};

template <typename FPTYPE>
struct pack_spin_recip_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        std::complex<FPTYPE>* packed,
        const std::complex<FPTYPE>* spin_data,
        const int npw,
        const int nspin);
};

template <typename FPTYPE>
struct unpack_spin_recip_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        std::complex<FPTYPE>* spin_data,
        const std::complex<FPTYPE>* packed,
        const int npw,
        const int nspin);
};

template <typename FPTYPE>
struct split_double_grid_recip_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        std::complex<FPTYPE>* smooth,
        std::complex<FPTYPE>* high_frequency,
        const std::complex<FPTYPE>* dense,
        const int smooth_npw,
        const int dense_npw,
        const int nspin);
};

template <typename FPTYPE>
struct combine_double_grid_recip_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        std::complex<FPTYPE>* dense,
        const std::complex<FPTYPE>* smooth,
        const std::complex<FPTYPE>* high_frequency,
        const int smooth_npw,
        const int dense_npw,
        const int nspin);
};

#if __CUDA || __UT_USE_CUDA
template <typename FPTYPE>
struct inner_product_recip_hartree_batch_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        const std::complex<FPTYPE>* lhs,
        const std::complex<FPTYPE>* rhs,
        const FPTYPE* gg,
        const int npw,
        const int nlhs,
        const int nrhs,
        const int ig_gge0,
        const FPTYPE tpiba2,
        FPTYPE* result,
        FPTYPE* workspace);
};

template <typename FPTYPE>
struct inner_product_recip_hartree_spin_batch_op<FPTYPE, base_device::DEVICE_GPU> {
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        const std::complex<FPTYPE>* lhs,
        const std::complex<FPTYPE>* rhs,
        const FPTYPE* gg,
        const int npw,
        const int nspin,
        const int nlhs,
        const int nrhs,
        const int ig_gge0,
        const bool gamma_only,
        const bool include_magnetism,
        const FPTYPE charge_fac,
        const FPTYPE mag_fac,
        FPTYPE* result,
        FPTYPE* workspace);
};
#endif
#endif

} // namespace elecstate

#endif // CHARGE_MIXING_OP_H

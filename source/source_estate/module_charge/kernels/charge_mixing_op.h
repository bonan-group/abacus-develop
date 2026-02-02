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

/// Inner product with 1/G^2 weight for Hartree-like functional:
/// sum_G conj(rhog1[G]) * rhog2[G] / gg[G] * tpiba2
/// This computes the Hartree energy contribution from two charge densities.
template <typename FPTYPE, typename Device>
struct inner_product_recip_hartree_op {
    FPTYPE operator()(
        const Device* ctx,
        const std::complex<FPTYPE>* rhog1,  // [npw] - first charge density in reciprocal space
        const std::complex<FPTYPE>* rhog2,  // [npw] - second charge density in reciprocal space
        const FPTYPE* gg,                    // [npw] - |G|^2 values
        const int npw,                       // Number of plane waves
        const int ig_gge0,                   // Index of G=0 (to skip)
        const FPTYPE tpiba2,                 // (2*pi/a)^2 prefactor
        FPTYPE* workspace);                  // GPU reduction workspace [num_blocks]
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
struct inner_product_recip_hartree_op<FPTYPE, base_device::DEVICE_CPU> {
    FPTYPE operator()(
        const base_device::DEVICE_CPU* ctx,
        const std::complex<FPTYPE>* rhog1,
        const std::complex<FPTYPE>* rhog2,
        const FPTYPE* gg,
        const int npw,
        const int ig_gge0,
        const FPTYPE tpiba2,
        FPTYPE* workspace);
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
struct inner_product_recip_hartree_op<FPTYPE, base_device::DEVICE_GPU> {
    FPTYPE operator()(
        const base_device::DEVICE_GPU* ctx,
        const std::complex<FPTYPE>* rhog1,
        const std::complex<FPTYPE>* rhog2,
        const FPTYPE* gg,
        const int npw,
        const int ig_gge0,
        const FPTYPE tpiba2,
        FPTYPE* workspace);
};
#endif

} // namespace elecstate

#endif // CHARGE_MIXING_OP_H

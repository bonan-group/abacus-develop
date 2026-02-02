#include "charge_mixing_op.h"
#include <cmath>
#include <algorithm>

namespace elecstate {

template <typename FPTYPE>
void kerker_screen_recip_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    std::complex<FPTYPE>* drhog,
    const FPTYPE* gg,
    const FPTYPE gg0,
    const FPTYPE gg0_min,
    const int npw,
    const int nspin)
{
    // Apply Kerker screening: drhog[ig] *= gg[ig]/(gg[ig]+gg0)
    // This is a high-pass filter that accelerates convergence by damping long-wavelength fluctuations
    for (int is = 0; is < nspin; ++is)
    {
        const int offset = is * npw;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int ig = 0; ig < npw; ++ig)
        {
            // Compute filter coefficient: gg/(gg+gg0), bounded by gg0_min
            FPTYPE filter_g = std::max(gg[ig] / (gg[ig] + gg0), gg0_min);
            drhog[offset + ig] *= filter_g;
        }
    }
}

template <typename FPTYPE>
FPTYPE inner_product_recip_hartree_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<FPTYPE>* rhog1,
    const std::complex<FPTYPE>* rhog2,
    const FPTYPE* gg,
    const int npw,
    const int ig_gge0,
    const FPTYPE tpiba2,
    FPTYPE* workspace)
{
    // Compute inner product with 1/G^2 weight:
    // sum_G conj(rhog1[G]) * rhog2[G] / gg[G]
    // This is the Hartree-like energy functional used in mixing algorithms
    FPTYPE result = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:result)
#endif
    for (int ig = 0; ig < npw; ++ig)
    {
        // Skip G=0 (divergent term)
        if (ig == ig_gge0) continue;

        // Compute conj(rhog1) * rhog2 / gg
        std::complex<FPTYPE> prod = std::conj(rhog1[ig]) * rhog2[ig];
        result += prod.real() / gg[ig];
    }
    return result * tpiba2;
}

// Explicit template instantiations
template struct kerker_screen_recip_op<float, base_device::DEVICE_CPU>;
template struct kerker_screen_recip_op<double, base_device::DEVICE_CPU>;
template struct inner_product_recip_hartree_op<float, base_device::DEVICE_CPU>;
template struct inner_product_recip_hartree_op<double, base_device::DEVICE_CPU>;

} // namespace elecstate

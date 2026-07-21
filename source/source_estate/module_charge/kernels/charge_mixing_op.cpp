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
void pack_spin_recip_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    std::complex<FPTYPE>* packed,
    const std::complex<FPTYPE>* spin_data,
    const int npw,
    const int nspin)
{
    if (nspin == 2)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int ig = 0; ig < npw; ++ig)
        {
            packed[ig] = spin_data[ig] + spin_data[npw + ig];
            packed[npw + ig] = spin_data[ig] - spin_data[npw + ig];
        }
    }
    else if (nspin == 4)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int i = 0; i < nspin * npw; ++i)
        {
            packed[i] = spin_data[i];
        }
    }
}

template <typename FPTYPE>
void unpack_spin_recip_op<FPTYPE, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    std::complex<FPTYPE>* spin_data,
    const std::complex<FPTYPE>* packed,
    const int npw,
    const int nspin)
{
    if (nspin == 2)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int ig = 0; ig < npw; ++ig)
        {
            spin_data[ig] = static_cast<FPTYPE>(0.5) * (packed[ig] + packed[npw + ig]);
            spin_data[npw + ig] = static_cast<FPTYPE>(0.5) * (packed[ig] - packed[npw + ig]);
        }
    }
    else if (nspin == 4)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int i = 0; i < nspin * npw; ++i)
        {
            spin_data[i] = packed[i];
        }
    }
}

// Explicit template instantiations
template struct kerker_screen_recip_op<float, base_device::DEVICE_CPU>;
template struct kerker_screen_recip_op<double, base_device::DEVICE_CPU>;
template struct pack_spin_recip_op<float, base_device::DEVICE_CPU>;
template struct pack_spin_recip_op<double, base_device::DEVICE_CPU>;
template struct unpack_spin_recip_op<float, base_device::DEVICE_CPU>;
template struct unpack_spin_recip_op<double, base_device::DEVICE_CPU>;

} // namespace elecstate

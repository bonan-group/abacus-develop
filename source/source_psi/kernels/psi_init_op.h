#ifndef SOURCE_PSI_KERNELS_PSI_INIT_OP_H
#define SOURCE_PSI_KERNELS_PSI_INIT_OP_H

#include "source_base/macros.h"
#include "source_base/module_device/types.h"

#include <complex>

namespace psi
{

template <typename T, typename Device>
struct init_random_op
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const Device* ctx,
                    T* psi,
                    const int nbands,
                    const int npwk,
                    const int npwk_max,
                    const int npol,
                    const int ik,
                    const int ik_tot,
                    const int seed,
                    const Real* gk2,
                    const int* igl2isz,
                    const int nst,
                    const int nz);
};

#if __CUDA || __UT_USE_CUDA
template <typename T>
struct init_random_op<T, base_device::DEVICE_GPU>
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const base_device::DEVICE_GPU* ctx,
                    T* psi,
                    const int nbands,
                    const int npwk,
                    const int npwk_max,
                    const int npol,
                    const int ik,
                    const int ik_tot,
                    const int seed,
                    const Real* gk2,
                    const int* igl2isz,
                    const int nst,
                    const int nz);
};
#endif

} // namespace psi

#endif

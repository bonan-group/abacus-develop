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
                    const int* is2fftixy,
                    const int fftnxy,
                    const int nz);
};

template <typename Real, typename Device>
struct build_gk_op
{
    void operator()(const Device* ctx,
                    Real* gk,
                    const Real* gcar,
                    const Real* kvec_c,
                    const int ik,
                    const int npwk,
                    const int npwk_max);
};

template <typename T, typename Device>
struct init_atomic_op
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const Device* ctx,
                    T* psi,
                    const int natomwfc,
                    const int npwk,
                    const int npwk_max,
                    const int total_lm,
                    const int nchi_max,
                    const int nqx,
                    const Real dq,
                    const Real tpiba,
                    const Real* gk,
                    const Real* ylm,
                    const T* sk,
                    const Real* table,
                    const int* iw2iat,
                    const int* iw2it,
                    const int* iw2ic,
                    const int* iw2lm,
                    const int* iw2l);
};

template <typename T, typename Device>
struct perturb_atomic_op
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
                    const Real mixing_coef,
                    const Real* gk2,
                    const int* igl2isz,
                    const int* is2fftixy,
                    const int fftnxy,
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
                    const int* is2fftixy,
                    const int fftnxy,
                    const int nz);
};

template <typename Real>
struct build_gk_op<Real, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    Real* gk,
                    const Real* gcar,
                    const Real* kvec_c,
                    const int ik,
                    const int npwk,
                    const int npwk_max);
};

template <typename T>
struct init_atomic_op<T, base_device::DEVICE_GPU>
{
    using Real = typename GetTypeReal<T>::type;
    void operator()(const base_device::DEVICE_GPU* ctx,
                    T* psi,
                    const int natomwfc,
                    const int npwk,
                    const int npwk_max,
                    const int total_lm,
                    const int nchi_max,
                    const int nqx,
                    const Real dq,
                    const Real tpiba,
                    const Real* gk,
                    const Real* ylm,
                    const T* sk,
                    const Real* table,
                    const int* iw2iat,
                    const int* iw2it,
                    const int* iw2ic,
                    const int* iw2lm,
                    const int* iw2l);
};

template <typename T>
struct perturb_atomic_op<T, base_device::DEVICE_GPU>
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
                    const Real mixing_coef,
                    const Real* gk2,
                    const int* igl2isz,
                    const int* is2fftixy,
                    const int fftnxy,
                    const int nz);
};
#endif

} // namespace psi

#endif

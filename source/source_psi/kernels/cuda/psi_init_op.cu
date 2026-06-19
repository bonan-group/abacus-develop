#include "source_psi/kernels/psi_init_op.h"

#include "source_base/module_device/device_check.h"

#include <cuda_runtime.h>
#include <thrust/complex.h>

#include <complex>
#include <cstdint>

namespace psi
{
namespace
{

constexpr int THREADS_PER_BLOCK = 256;

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

template <typename Real>
__device__ __forceinline__ Real unit_random(const std::uint64_t key)
{
    constexpr double norm = 1.0 / 9007199254740992.0; // 2^-53
    return static_cast<Real>(static_cast<double>(splitmix64(key) >> 11) * norm);
}

template <typename Real>
__global__ void init_random_kernel(thrust::complex<Real>* psi,
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
                                   const int nz)
{
    const int total = nbands * npwk_max * npol;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
    {
        return;
    }

    const int ig = idx % npwk_max;
    const int ipol = (idx / npwk_max) % npol;
    const int iband = idx / (npwk_max * npol);
    if (ig >= npwk)
    {
        psi[idx] = thrust::complex<Real>(0.0, 0.0);
        return;
    }

    const std::uint64_t seed_key = seed > 0 ? static_cast<std::uint64_t>(seed) : 0x6a09e667f3bcc909ULL;
    std::uint64_t draw_index = 0;
    if (seed > 0 && igl2isz != nullptr && nst > 0 && nz > 0)
    {
        const int isz = igl2isz[ik * npwk_max + ig];
        const int is = isz / nz;
        const int iz = isz - is * nz;
        const std::uint64_t sequence_index =
            ((static_cast<std::uint64_t>(iband) * static_cast<std::uint64_t>(npol)
              + static_cast<std::uint64_t>(ipol))
                 * static_cast<std::uint64_t>(nst)
             + static_cast<std::uint64_t>(is))
                * static_cast<std::uint64_t>(nz)
            + static_cast<std::uint64_t>(iz);
        draw_index = 2ULL * sequence_index;
    }
    else
    {
        draw_index =
            2ULL
            * (((static_cast<std::uint64_t>(iband) * static_cast<std::uint64_t>(npol)
                 + static_cast<std::uint64_t>(ipol))
                    * static_cast<std::uint64_t>(npwk_max))
               + static_cast<std::uint64_t>(ig));
    }

    const std::uint64_t ik_key = static_cast<std::uint64_t>(ik_tot + 1) * 0x9e3779b97f4a7c15ULL;
    const std::uint64_t rr_key = seed_key ^ ik_key ^ ((draw_index + 1ULL) * 0xd2b74407b1ce6e93ULL);
    const std::uint64_t arg_key = seed_key ^ ik_key ^ ((draw_index + 2ULL) * 0xd2b74407b1ce6e93ULL);
    const Real rr = unit_random<Real>(rr_key);
    const Real arg = static_cast<Real>(6.283185307179586476925286766559)
                     * unit_random<Real>(arg_key);
    const Real inv_gk2 = static_cast<Real>(1.0) / (gk2[ik * npwk_max + ig] + static_cast<Real>(1.0));
    const Real damping = seed > 0 ? inv_gk2 : inv_gk2 * inv_gk2;
    psi[idx] = thrust::complex<Real>(rr * cos(arg) * damping, rr * sin(arg) * damping);
}

template <typename Real>
void launch_init_random(std::complex<Real>* psi,
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
                        const int nz)
{
    const int total = nbands * npwk_max * npol;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_random_kernel<Real><<<blocks, THREADS_PER_BLOCK>>>(reinterpret_cast<thrust::complex<Real>*>(psi),
                                                            nbands,
                                                            npwk,
                                                            npwk_max,
                                                            npol,
                                                            ik,
                                                            ik_tot,
                                                            seed,
                                                            gk2,
                                                            igl2isz,
                                                            nst,
                                                            nz);
    CHECK_CUDA_SYNC();
}

} // namespace

template <typename T>
void init_random_op<T, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
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
                                                            const int nz)
{
    launch_init_random<typename GetTypeReal<T>::type>(
        psi, nbands, npwk, npwk_max, npol, ik, ik_tot, seed, gk2, igl2isz, nst, nz);
}

template struct init_random_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct init_random_op<std::complex<double>, base_device::DEVICE_GPU>;

} // namespace psi

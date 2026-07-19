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

__device__ __forceinline__ std::uint64_t random_draw_index(const int seed,
                                                           const int iband,
                                                           const int ipol,
                                                           const int ig,
                                                           const int npwk_max,
                                                           const int npol,
                                                           const int ik,
                                                           const int* igl2isz,
                                                           const int* is2fftixy,
                                                           const int fftnxy,
                                                           const int nz)
{
    if (seed > 0 && igl2isz != nullptr && is2fftixy != nullptr && fftnxy > 0 && nz > 0)
    {
        const int isz = igl2isz[ik * npwk_max + ig];
        const int is = isz / nz;
        const int iz = isz - is * nz;
        const int global_is = is2fftixy[is];
        const std::uint64_t sequence_index =
            ((static_cast<std::uint64_t>(iband) * static_cast<std::uint64_t>(npol)
              + static_cast<std::uint64_t>(ipol))
                 * static_cast<std::uint64_t>(fftnxy)
             + static_cast<std::uint64_t>(global_is))
                * static_cast<std::uint64_t>(nz)
            + static_cast<std::uint64_t>(iz);
        return 2ULL * sequence_index;
    }
    return 2ULL
           * (((static_cast<std::uint64_t>(iband) * static_cast<std::uint64_t>(npol)
                + static_cast<std::uint64_t>(ipol))
                   * static_cast<std::uint64_t>(npwk_max))
              + static_cast<std::uint64_t>(ig));
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
                                   const int* is2fftixy,
                                   const int fftnxy,
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
    const std::uint64_t draw_index = random_draw_index(
        seed, iband, ipol, ig, npwk_max, npol, ik, igl2isz, is2fftixy, fftnxy, nz);

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
__global__ void build_gk_kernel(Real* gk,
                                const Real* gcar,
                                const Real* kvec_c,
                                const int ik,
                                const int npwk,
                                const int npwk_max)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npwk * 3)
    {
        return;
    }
    const int component = idx % 3;
    gk[idx] = gcar[ik * npwk_max * 3 + idx] + kvec_c[ik * 3 + component];
}

template <typename Real>
__device__ __forceinline__ Real interpolate(const Real* table,
                                            const int nqx,
                                            const Real dq,
                                            const Real x)
{
    const Real position = x / dq;
    const int iq = static_cast<int>(position);
    if (iq < 0 || iq > nqx - 4)
    {
        return static_cast<Real>(0.0);
    }
    const Real x0 = position - static_cast<Real>(iq);
    const Real x1 = static_cast<Real>(1.0) - x0;
    const Real x2 = static_cast<Real>(2.0) - x0;
    const Real x3 = static_cast<Real>(3.0) - x0;
    return table[iq] * x1 * x2 * x3 / static_cast<Real>(6.0)
           + table[iq + 1] * x0 * x2 * x3 / static_cast<Real>(2.0)
           - table[iq + 2] * x1 * x0 * x3 / static_cast<Real>(2.0)
           + table[iq + 3] * x1 * x2 * x0 / static_cast<Real>(6.0);
}

template <typename Real>
__device__ __forceinline__ thrust::complex<Real> minus_i_to_l(const int l)
{
    switch (l & 3)
    {
    case 0:
        return thrust::complex<Real>(1.0, 0.0);
    case 1:
        return thrust::complex<Real>(0.0, -1.0);
    case 2:
        return thrust::complex<Real>(-1.0, 0.0);
    default:
        return thrust::complex<Real>(0.0, 1.0);
    }
}

template <typename Real>
__global__ void init_atomic_kernel(thrust::complex<Real>* psi,
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
                                   const thrust::complex<Real>* sk,
                                   const Real* table,
                                   const int* iw2iat,
                                   const int* iw2it,
                                   const int* iw2ic,
                                   const int* iw2lm,
                                   const int* iw2l)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = natomwfc * npwk_max;
    if (idx >= total)
    {
        return;
    }
    const int iw = idx / npwk_max;
    const int ig = idx - iw * npwk_max;
    if (ig >= npwk)
    {
        psi[idx] = thrust::complex<Real>(0.0, 0.0);
        return;
    }

    const int lm = iw2lm[iw];
    if (lm < 0 || lm >= total_lm)
    {
        psi[idx] = thrust::complex<Real>(0.0, 0.0);
        return;
    }
    const Real gx = gk[ig * 3];
    const Real gy = gk[ig * 3 + 1];
    const Real gz = gk[ig * 3 + 2];
    const Real q = sqrt(gx * gx + gy * gy + gz * gz) * tpiba;
    const int table_offset = (iw2it[iw] * nchi_max + iw2ic[iw]) * nqx;
    const Real radial = interpolate(table + table_offset, nqx, dq, q);
    const Real angular = ylm[lm * npwk + ig];
    psi[idx] = minus_i_to_l<Real>(iw2l[iw]) * sk[iw2iat[iw] * npwk + ig] * (angular * radial);
}

template <typename Real>
__global__ void perturb_atomic_kernel(thrust::complex<Real>* psi,
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
                                      const int nz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nbands * npwk_max * npol;
    if (idx >= total)
    {
        return;
    }
    const int ig = idx % npwk_max;
    if (ig >= npwk)
    {
        return;
    }
    const int ipol = (idx / npwk_max) % npol;
    const int iband = idx / (npwk_max * npol);
    const std::uint64_t seed_key = seed > 0 ? static_cast<std::uint64_t>(seed) : 0x6a09e667f3bcc909ULL;
    const std::uint64_t draw_index = random_draw_index(
        seed, iband, ipol, ig, npwk_max, npol, ik, igl2isz, is2fftixy, fftnxy, nz);
    const std::uint64_t ik_key = static_cast<std::uint64_t>(ik_tot + 1) * 0x9e3779b97f4a7c15ULL;
    const std::uint64_t rr_key = seed_key ^ ik_key ^ ((draw_index + 1ULL) * 0xd2b74407b1ce6e93ULL);
    const std::uint64_t arg_key = seed_key ^ ik_key ^ ((draw_index + 2ULL) * 0xd2b74407b1ce6e93ULL);
    const Real rr = unit_random<Real>(rr_key);
    const Real arg = static_cast<Real>(6.283185307179586476925286766559) * unit_random<Real>(arg_key);
    const Real damping = seed > 0 ? static_cast<Real>(1.0)
                                  : static_cast<Real>(1.0) / (gk2[ik * npwk_max + ig] + static_cast<Real>(1.0));
    const thrust::complex<Real> random_value(rr * cos(arg) * damping, rr * sin(arg) * damping);
    psi[idx] *= thrust::complex<Real>(1.0, 0.0) + mixing_coef * random_value;
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
                        const int* is2fftixy,
                        const int fftnxy,
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
                                                            is2fftixy,
                                                            fftnxy,
                                                            nz);
    CHECK_CUDA_SYNC();
}

template <typename Real>
void launch_build_gk(Real* gk,
                     const Real* gcar,
                     const Real* kvec_c,
                     const int ik,
                     const int npwk,
                     const int npwk_max)
{
    const int total = npwk * 3;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    build_gk_kernel<Real><<<blocks, THREADS_PER_BLOCK>>>(gk, gcar, kvec_c, ik, npwk, npwk_max);
    CHECK_CUDA_SYNC();
}

template <typename Real>
void launch_init_atomic(std::complex<Real>* psi,
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
                        const std::complex<Real>* sk,
                        const Real* table,
                        const int* iw2iat,
                        const int* iw2it,
                        const int* iw2ic,
                        const int* iw2lm,
                        const int* iw2l)
{
    const int total = natomwfc * npwk_max;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_atomic_kernel<Real><<<blocks, THREADS_PER_BLOCK>>>(reinterpret_cast<thrust::complex<Real>*>(psi),
                                                            natomwfc,
                                                            npwk,
                                                            npwk_max,
                                                            total_lm,
                                                            nchi_max,
                                                            nqx,
                                                            dq,
                                                            tpiba,
                                                            gk,
                                                            ylm,
                                                            reinterpret_cast<const thrust::complex<Real>*>(sk),
                                                            table,
                                                            iw2iat,
                                                            iw2it,
                                                            iw2ic,
                                                            iw2lm,
                                                            iw2l);
    CHECK_CUDA_SYNC();
}

template <typename Real>
void launch_perturb_atomic(std::complex<Real>* psi,
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
                           const int nz)
{
    const int total = nbands * npwk_max * npol;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    perturb_atomic_kernel<Real><<<blocks, THREADS_PER_BLOCK>>>(reinterpret_cast<thrust::complex<Real>*>(psi),
                                                               nbands,
                                                               npwk,
                                                               npwk_max,
                                                               npol,
                                                               ik,
                                                               ik_tot,
                                                               seed,
                                                               mixing_coef,
                                                               gk2,
                                                               igl2isz,
                                                               is2fftixy,
                                                               fftnxy,
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
                                                            const int* is2fftixy,
                                                            const int fftnxy,
                                                            const int nz)
{
    launch_init_random<typename GetTypeReal<T>::type>(
        psi, nbands, npwk, npwk_max, npol, ik, ik_tot, seed, gk2, igl2isz, is2fftixy, fftnxy, nz);
}

template <typename Real>
void build_gk_op<Real, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                            Real* gk,
                                                            const Real* gcar,
                                                            const Real* kvec_c,
                                                            const int ik,
                                                            const int npwk,
                                                            const int npwk_max)
{
    launch_build_gk(gk, gcar, kvec_c, ik, npwk, npwk_max);
}

template <typename T>
void init_atomic_op<T, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
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
                                                            const int* iw2l)
{
    launch_init_atomic(psi,
                       natomwfc,
                       npwk,
                       npwk_max,
                       total_lm,
                       nchi_max,
                       nqx,
                       dq,
                       tpiba,
                       gk,
                       ylm,
                       sk,
                       table,
                       iw2iat,
                       iw2it,
                       iw2ic,
                       iw2lm,
                       iw2l);
}

template <typename T>
void perturb_atomic_op<T, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
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
                                                               const int nz)
{
    launch_perturb_atomic(psi,
                          nbands,
                          npwk,
                          npwk_max,
                          npol,
                          ik,
                          ik_tot,
                          seed,
                          mixing_coef,
                          gk2,
                          igl2isz,
                          is2fftixy,
                          fftnxy,
                          nz);
}

template struct init_random_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct init_random_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct build_gk_op<float, base_device::DEVICE_GPU>;
template struct build_gk_op<double, base_device::DEVICE_GPU>;
template struct init_atomic_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct init_atomic_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct perturb_atomic_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct perturb_atomic_op<std::complex<double>, base_device::DEVICE_GPU>;

} // namespace psi

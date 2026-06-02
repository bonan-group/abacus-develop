#include "source_pw/module_pwdft/kernels/structure_factor_op.h"

#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#define THREADS_PER_BLOCK 256

namespace structure_factor_op
{

template <typename FPTYPE>
__global__ void compute_struc_fac_kernel(const int ntype,
                                         const FPTYPE* tau,
                                         const int* atom_index,
                                         const int ngm,
                                         const FPTYPE* gcar,
                                         const FPTYPE two_pi,
                                         thrust::complex<FPTYPE>* struc_fac)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ntype * ngm;
    if (idx >= total)
    {
        return;
    }

    const int it = idx / ngm;
    const int ig = idx % ngm;
    const FPTYPE gx = gcar[ig * 3 + 0];
    const FPTYPE gy = gcar[ig * 3 + 1];
    const FPTYPE gz = gcar[ig * 3 + 2];
    const int ia_begin = atom_index[it];
    const int ia_end = atom_index[it + 1];

    thrust::complex<FPTYPE> sum_phase(0.0, 0.0);
    for (int ia = ia_begin; ia < ia_end; ++ia)
    {
        const FPTYPE arg = -two_pi * (gx * tau[ia * 3 + 0] + gy * tau[ia * 3 + 1] + gz * tau[ia * 3 + 2]);
        FPTYPE sin_arg = 0;
        FPTYPE cos_arg = 0;
        sincos(arg, &sin_arg, &cos_arg);
        sum_phase += thrust::complex<FPTYPE>(cos_arg, sin_arg);
    }
    struc_fac[idx] = sum_phase;
}

template <typename FPTYPE>
void compute_struc_fac_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                       const int ntype,
                                                                       const FPTYPE* tau,
                                                                       const int* atom_index,
                                                                       const int ngm,
                                                                       const FPTYPE* gcar,
                                                                       const FPTYPE two_pi,
                                                                       std::complex<FPTYPE>* struc_fac)
{
    const int total = ntype * ngm;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_struc_fac_kernel<FPTYPE><<<blocks, THREADS_PER_BLOCK>>>(
        ntype,
        tau,
        atom_index,
        ngm,
        gcar,
        two_pi,
        reinterpret_cast<thrust::complex<FPTYPE>*>(struc_fac));
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void compute_eigts1_kernel(const int nat,
                                      const FPTYPE* gtau,
                                      const int nx,
                                      const FPTYPE two_pi,
                                      thrust::complex<FPTYPE>* eigts1)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = 2 * nx + 1;
    const int total = nat * width;
    if (idx >= total)
    {
        return;
    }

    const int iat = idx / width;
    const int n1 = idx % width - nx;
    const FPTYPE arg = -two_pi * n1 * gtau[iat * 3 + 0];
    FPTYPE sin_arg = 0;
    FPTYPE cos_arg = 0;
    sincos(arg, &sin_arg, &cos_arg);
    eigts1[idx] = thrust::complex<FPTYPE>(cos_arg, sin_arg);
}

template <typename FPTYPE>
__global__ void compute_eigts2_kernel(const int nat,
                                      const FPTYPE* gtau,
                                      const int ny,
                                      const FPTYPE two_pi,
                                      thrust::complex<FPTYPE>* eigts2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = 2 * ny + 1;
    const int total = nat * width;
    if (idx >= total)
    {
        return;
    }

    const int iat = idx / width;
    const int n2 = idx % width - ny;
    const FPTYPE arg = -two_pi * n2 * gtau[iat * 3 + 1];
    FPTYPE sin_arg = 0;
    FPTYPE cos_arg = 0;
    sincos(arg, &sin_arg, &cos_arg);
    eigts2[idx] = thrust::complex<FPTYPE>(cos_arg, sin_arg);
}

template <typename FPTYPE>
__global__ void compute_eigts3_kernel(const int nat,
                                      const FPTYPE* gtau,
                                      const int nz,
                                      const FPTYPE two_pi,
                                      thrust::complex<FPTYPE>* eigts3)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int width = 2 * nz + 1;
    const int total = nat * width;
    if (idx >= total)
    {
        return;
    }

    const int iat = idx / width;
    const int n3 = idx % width - nz;
    const FPTYPE arg = -two_pi * n3 * gtau[iat * 3 + 2];
    FPTYPE sin_arg = 0;
    FPTYPE cos_arg = 0;
    sincos(arg, &sin_arg, &cos_arg);
    eigts3[idx] = thrust::complex<FPTYPE>(cos_arg, sin_arg);
}

template <typename FPTYPE>
void compute_eigts_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                   const int nat,
                                                                   const FPTYPE* gtau,
                                                                   const int nx,
                                                                   const int ny,
                                                                   const int nz,
                                                                   const FPTYPE two_pi,
                                                                   std::complex<FPTYPE>* eigts1,
                                                                   std::complex<FPTYPE>* eigts2,
                                                                   std::complex<FPTYPE>* eigts3)
{
    const int total1 = nat * (2 * nx + 1);
    const int total2 = nat * (2 * ny + 1);
    const int total3 = nat * (2 * nz + 1);
    compute_eigts1_kernel<FPTYPE><<<(total1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        nat,
        gtau,
        nx,
        two_pi,
        reinterpret_cast<thrust::complex<FPTYPE>*>(eigts1));
    compute_eigts2_kernel<FPTYPE><<<(total2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        nat,
        gtau,
        ny,
        two_pi,
        reinterpret_cast<thrust::complex<FPTYPE>*>(eigts2));
    compute_eigts3_kernel<FPTYPE><<<(total3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        nat,
        gtau,
        nz,
        two_pi,
        reinterpret_cast<thrust::complex<FPTYPE>*>(eigts3));
    CHECK_CUDA_SYNC();
}

template struct compute_struc_fac_op<float, base_device::DEVICE_GPU>;
template struct compute_struc_fac_op<double, base_device::DEVICE_GPU>;
template struct compute_eigts_op<float, base_device::DEVICE_GPU>;
template struct compute_eigts_op<double, base_device::DEVICE_GPU>;

} // namespace structure_factor_op

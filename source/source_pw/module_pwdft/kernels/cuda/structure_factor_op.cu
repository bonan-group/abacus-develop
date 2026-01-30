#include "source_pw/module_pwdft/kernels/structure_factor_op.h"

#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>

#include <base/macros/macros.h>

#define THREADS_PER_BLOCK 256

namespace structure_factor_op {

/**
 * @brief CUDA kernel to compute structure factors for all atom types and G-vectors
 *
 * Each thread computes one (type, G-vector) pair:
 * S(G, type) = Σ_atoms exp(-i·2π·G·τ)
 *
 * @tparam FPTYPE Floating point type (float or double)
 * @param ntype Number of atom types
 * @param tau Atom positions in Cartesian coordinates [nat_total * 3]
 * @param atom_index Cumulative atom indices [ntype+1], where atom_index[it+1] - atom_index[it] = number of atoms of type it
 * @param ngm Number of G-vectors
 * @param gcar G-vector Cartesian coordinates [ngm * 3]
 * @param TWO_PI 2π constant
 * @param strucFac Output structure factors [ntype * ngm]
 */
template <typename FPTYPE>
__global__ void compute_struc_fac_kernel(
    const int ntype,
    const FPTYPE* tau,
    const int* atom_index,
    const int ngm,
    const FPTYPE* gcar,
    const FPTYPE TWO_PI,
    thrust::complex<FPTYPE>* strucFac)
{
    // Each thread computes one (type, G-vector) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = ntype * ngm;

    if (idx >= total_threads) return;

    // Decode thread index to (type, G-vector)
    const int it = idx / ngm;  // Atom type
    const int ig = idx % ngm;  // G-vector index

    // Load G-vector (coalesced read for consecutive G-vectors)
    const FPTYPE gx = gcar[ig * 3 + 0];
    const FPTYPE gy = gcar[ig * 3 + 1];
    const FPTYPE gz = gcar[ig * 3 + 2];

    // Compute structure factor for this (type, G-vector) pair
    thrust::complex<FPTYPE> sum_phase(0.0, 0.0);

    // Get atom range for this type
    const int ia_start = atom_index[it];
    const int ia_end = atom_index[it + 1];

    // Loop over atoms of this type (serial reduction within thread)
    for (int ia = ia_start; ia < ia_end; ia++)
    {
        // Load atom position
        const FPTYPE tau_x = tau[ia * 3 + 0];
        const FPTYPE tau_y = tau[ia * 3 + 1];
        const FPTYPE tau_z = tau[ia * 3 + 2];

        // Compute phase: -2π * G · τ
        const FPTYPE arg = -TWO_PI * (gx * tau_x + gy * tau_y + gz * tau_z);

        // S(G) = Σ exp(-i·2π·G·τ)
        // Use sincos for simultaneous sin/cos computation (GPU optimized)
        FPTYPE cos_arg, sin_arg;
        sincos(arg, &sin_arg, &cos_arg);
        sum_phase += thrust::complex<FPTYPE>(cos_arg, sin_arg);
    }

    // Write result (no write conflicts - each thread writes unique location)
    strucFac[it * ngm + ig] = sum_phase;
}

// Host wrapper for float precision
template <>
void compute_struc_fac_op<float, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int ntype,
    const float* tau,
    const int* atom_index,
    const int ngm,
    const float* gcar,
    const float TWO_PI,
    std::complex<float>* strucFac)
{
    const int total_threads = ntype * ngm;
    const int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_struc_fac_kernel<float><<<blocks, THREADS_PER_BLOCK>>>(
        ntype,
        tau,
        atom_index,
        ngm,
        gcar,
        TWO_PI,
        reinterpret_cast<thrust::complex<float>*>(strucFac));

    cudaCheckOnDebug();
}

// Host wrapper for double precision
template <>
void compute_struc_fac_op<double, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int ntype,
    const double* tau,
    const int* atom_index,
    const int ngm,
    const double* gcar,
    const double TWO_PI,
    std::complex<double>* strucFac)
{
    const int total_threads = ntype * ngm;
    const int blocks = (total_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    compute_struc_fac_kernel<double><<<blocks, THREADS_PER_BLOCK>>>(
        ntype,
        tau,
        atom_index,
        ngm,
        gcar,
        TWO_PI,
        reinterpret_cast<thrust::complex<double>*>(strucFac));

    cudaCheckOnDebug();
}

/**
 * @brief CUDA kernel to compute eigts1 phase factors
 *
 * Each thread computes one (atom, n1) pair:
 * eigts1[iat, n1+nx] = exp(i·2π·n1·gtau.x)
 */
template <typename FPTYPE>
__global__ void compute_eigts1_kernel(
    const int nat,
    const FPTYPE* gtau,
    const int nx,
    const FPTYPE TWO_PI,
    thrust::complex<FPTYPE>* eigts1)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nat * (2 * nx + 1);
    if (idx >= total) return;

    // Decode thread index to (atom, n1)
    const int iat = idx / (2 * nx + 1);
    const int n1 = (idx % (2 * nx + 1)) - nx;  // Map [0, 2·nx] → [-nx, nx]

    // Compute phase: -2π * n1 * gtau.x (negative sign to match CPU: exp(-i·2π·n·gtau))
    const FPTYPE arg = -TWO_PI * n1 * gtau[iat * 3 + 0];

    // eigts1 = exp(i·arg) = exp(-i·2π·n1·gtau.x)
    FPTYPE cos_arg, sin_arg;
    sincos(arg, &sin_arg, &cos_arg);
    eigts1[idx] = thrust::complex<FPTYPE>(cos_arg, sin_arg);
}

/**
 * @brief CUDA kernel to compute eigts2 phase factors
 */
template <typename FPTYPE>
__global__ void compute_eigts2_kernel(
    const int nat,
    const FPTYPE* gtau,
    const int ny,
    const FPTYPE TWO_PI,
    thrust::complex<FPTYPE>* eigts2)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nat * (2 * ny + 1);
    if (idx >= total) return;

    const int iat = idx / (2 * ny + 1);
    const int n2 = (idx % (2 * ny + 1)) - ny;

    // Compute phase: -2π * n2 * gtau.y (negative sign to match CPU)
    const FPTYPE arg = -TWO_PI * n2 * gtau[iat * 3 + 1];

    FPTYPE cos_arg, sin_arg;
    sincos(arg, &sin_arg, &cos_arg);
    eigts2[idx] = thrust::complex<FPTYPE>(cos_arg, sin_arg);
}

/**
 * @brief CUDA kernel to compute eigts3 phase factors
 */
template <typename FPTYPE>
__global__ void compute_eigts3_kernel(
    const int nat,
    const FPTYPE* gtau,
    const int nz,
    const FPTYPE TWO_PI,
    thrust::complex<FPTYPE>* eigts3)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = nat * (2 * nz + 1);
    if (idx >= total) return;

    const int iat = idx / (2 * nz + 1);
    const int n3 = (idx % (2 * nz + 1)) - nz;

    // Compute phase: -2π * n3 * gtau.z (negative sign to match CPU)
    const FPTYPE arg = -TWO_PI * n3 * gtau[iat * 3 + 2];

    FPTYPE cos_arg, sin_arg;
    sincos(arg, &sin_arg, &cos_arg);
    eigts3[idx] = thrust::complex<FPTYPE>(cos_arg, sin_arg);
}

// Host wrapper for eigts computation (float)
template <>
void compute_eigts_op<float, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int nat,
    const float* gtau,
    const int nx,
    const int ny,
    const int nz,
    const float TWO_PI,
    std::complex<float>* eigts1,
    std::complex<float>* eigts2,
    std::complex<float>* eigts3)
{
    // Launch eigts1 kernel
    const int total1 = nat * (2 * nx + 1);
    const int blocks1 = (total1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_eigts1_kernel<float><<<blocks1, THREADS_PER_BLOCK>>>(
        nat, gtau, nx, TWO_PI,
        reinterpret_cast<thrust::complex<float>*>(eigts1));

    // Launch eigts2 kernel
    const int total2 = nat * (2 * ny + 1);
    const int blocks2 = (total2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_eigts2_kernel<float><<<blocks2, THREADS_PER_BLOCK>>>(
        nat, gtau, ny, TWO_PI,
        reinterpret_cast<thrust::complex<float>*>(eigts2));

    // Launch eigts3 kernel
    const int total3 = nat * (2 * nz + 1);
    const int blocks3 = (total3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_eigts3_kernel<float><<<blocks3, THREADS_PER_BLOCK>>>(
        nat, gtau, nz, TWO_PI,
        reinterpret_cast<thrust::complex<float>*>(eigts3));

    cudaCheckOnDebug();
}

// Host wrapper for eigts computation (double)
template <>
void compute_eigts_op<double, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int nat,
    const double* gtau,
    const int nx,
    const int ny,
    const int nz,
    const double TWO_PI,
    std::complex<double>* eigts1,
    std::complex<double>* eigts2,
    std::complex<double>* eigts3)
{
    // Launch eigts1 kernel
    const int total1 = nat * (2 * nx + 1);
    const int blocks1 = (total1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_eigts1_kernel<double><<<blocks1, THREADS_PER_BLOCK>>>(
        nat, gtau, nx, TWO_PI,
        reinterpret_cast<thrust::complex<double>*>(eigts1));

    // Launch eigts2 kernel
    const int total2 = nat * (2 * ny + 1);
    const int blocks2 = (total2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_eigts2_kernel<double><<<blocks2, THREADS_PER_BLOCK>>>(
        nat, gtau, ny, TWO_PI,
        reinterpret_cast<thrust::complex<double>*>(eigts2));

    // Launch eigts3 kernel
    const int total3 = nat * (2 * nz + 1);
    const int blocks3 = (total3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_eigts3_kernel<double><<<blocks3, THREADS_PER_BLOCK>>>(
        nat, gtau, nz, TWO_PI,
        reinterpret_cast<thrust::complex<double>*>(eigts3));

    cudaCheckOnDebug();
}

}  // namespace structure_factor_op

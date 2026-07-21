#include "source_estate/module_charge/kernels/charge_mixing_op.h"
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <base/macros/macros.h>
#include "source_base/tool_quit.h"

#include <algorithm>
#include <limits>
#include <string>

namespace elecstate {

namespace {

constexpr int KERKER_THREADS_PER_BLOCK = 256;
constexpr int MIN_REDUCTION_THREADS = 128;
constexpr int DEFAULT_REDUCTION_THREADS = 256;
constexpr int MAX_REDUCTION_THREADS = 1024;

int ceil_div(const int n, const int d)
{
    return (n + d - 1) / d;
}

long long ceil_div_ll(const long long n, const long long d)
{
    return (n + d - 1) / d;
}

int round_down_power_of_two(int value)
{
    int result = 1;
    while (result <= value / 2)
    {
        result *= 2;
    }
    return result;
}

cudaDeviceProp current_device_properties()
{
    int device_id = 0;
    CHECK_CUDA(cudaGetDevice(&device_id));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
    return prop;
}

int choose_reduction_threads(const int npw, const int npairs, const cudaDeviceProp& prop)
{
    const int device_max_threads = round_down_power_of_two(
        std::max(32, std::min(MAX_REDUCTION_THREADS, prop.maxThreadsPerBlock)));
    const int min_threads = std::min(MIN_REDUCTION_THREADS, device_max_threads);
    int threads = std::min(DEFAULT_REDUCTION_THREADS, device_max_threads);
    threads = std::max(min_threads, round_down_power_of_two(threads));

    const long long max_grid_x = static_cast<long long>(prop.maxGridSize[0]);
    while (threads < device_max_threads)
    {
        const long long num_blocks = ceil_div_ll(npw, threads);
        const long long total_blocks = num_blocks * static_cast<long long>(npairs);
        if (total_blocks <= max_grid_x)
        {
            break;
        }
        threads *= 2;
    }

    return threads;
}

void validate_reduction_launch(const int npw,
                               const int npairs,
                               const int threads,
                               const cudaDeviceProp& prop,
                               const char* name)
{
    const long long num_blocks = ceil_div_ll(npw, threads);
    const long long total_blocks = num_blocks * static_cast<long long>(npairs);
    if (num_blocks > static_cast<long long>(std::numeric_limits<int>::max())
        || total_blocks > static_cast<long long>(prop.maxGridSize[0]))
    {
        ModuleBase::WARNING_QUIT(
            name,
            "reduction launch is too large for this CUDA device: npw="
                + std::to_string(npw) + ", pairs=" + std::to_string(npairs)
                + ", threads=" + std::to_string(threads)
                + ", blocks=" + std::to_string(num_blocks)
                + ", grid.x=" + std::to_string(total_blocks)
                + ", maxGridSize[0]=" + std::to_string(prop.maxGridSize[0]));
    }
}

} // namespace

// CUDA kernel for Kerker screening
template <typename FPTYPE>
__global__ void kerker_screen_kernel(
    thrust::complex<FPTYPE>* drhog,
    const FPTYPE* gg,
    const FPTYPE gg0,
    const FPTYPE gg0_min,
    const int npw,
    const int nspin)
{
    int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw) return;

    // Compute filter coefficient once
    FPTYPE filter_g = max(gg[ig] / (gg[ig] + gg0), gg0_min);

    // Apply to all spin channels
    for (int is = 0; is < nspin; ++is)
    {
        drhog[is * npw + ig] *= filter_g;
    }
}

template <typename FPTYPE>
__global__ void inner_product_hartree_batch_partial_kernel(
    const thrust::complex<FPTYPE>* lhs,
    const thrust::complex<FPTYPE>* rhs,
    const FPTYPE* gg,
    FPTYPE* partial_sums,
    const int npw,
    const int nlhs,
    const int nrhs,
    const int num_blocks,
    const int ig_gge0,
    const FPTYPE tpiba2)
{
    extern __shared__ unsigned char shared_raw[];
    FPTYPE* sdata = reinterpret_cast<FPTYPE*>(shared_raw);
    const int tid = threadIdx.x;
    const long long linear_block = static_cast<long long>(blockIdx.x);
    const int pair = static_cast<int>(linear_block / num_blocks);
    const int block = static_cast<int>(linear_block - static_cast<long long>(pair) * num_blocks);
    const int lhs_idx = pair / nrhs;
    const int rhs_idx = pair - lhs_idx * nrhs;
    const int ig = block * blockDim.x + threadIdx.x;

    FPTYPE local_sum = 0.0;
    if (lhs_idx < nlhs && ig < npw && ig != ig_gge0)
    {
        const thrust::complex<FPTYPE>* lhs_vec = lhs + lhs_idx * npw;
        const thrust::complex<FPTYPE>* rhs_vec = rhs + rhs_idx * npw;
        thrust::complex<FPTYPE> prod = thrust::conj(lhs_vec[ig]) * rhs_vec[ig];
        local_sum = prod.real() / gg[ig] * tpiba2;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partial_sums[pair * num_blocks + block] = sdata[0];
    }
}

template <typename FPTYPE>
__global__ void inner_product_hartree_batch_final_kernel(
    const FPTYPE* partial_sums,
    FPTYPE* result,
    const int npairs,
    const int num_blocks)
{
    extern __shared__ unsigned char shared_raw[];
    FPTYPE* sdata = reinterpret_cast<FPTYPE*>(shared_raw);
    const int tid = threadIdx.x;
    const int pair = blockIdx.x;

    FPTYPE local_sum = 0.0;
    for (int ib = tid; ib < num_blocks; ib += blockDim.x)
    {
        local_sum += partial_sums[pair * num_blocks + ib];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && pair < npairs)
    {
        result[pair] = sdata[0];
    }
}

template <typename FPTYPE>
__global__ void inner_product_hartree_spin_batch_partial_kernel(
    const thrust::complex<FPTYPE>* lhs,
    const thrust::complex<FPTYPE>* rhs,
    const FPTYPE* gg,
    FPTYPE* partial_sums,
    const int npw,
    const int nspin,
    const int nlhs,
    const int nrhs,
    const int num_blocks,
    const int ig_gge0,
    const bool gamma_only,
    const bool include_magnetism,
    const FPTYPE charge_fac,
    const FPTYPE mag_fac)
{
    extern __shared__ unsigned char shared_raw[];
    FPTYPE* sdata = reinterpret_cast<FPTYPE*>(shared_raw);
    const int tid = threadIdx.x;
    const long long linear_block = static_cast<long long>(blockIdx.x);
    const int pair = static_cast<int>(linear_block / num_blocks);
    const int block = static_cast<int>(linear_block - static_cast<long long>(pair) * num_blocks);
    const int lhs_idx = pair / nrhs;
    const int rhs_idx = pair - lhs_idx * nrhs;
    const int ig = block * blockDim.x + threadIdx.x;
    const int stride = nspin * npw;

    FPTYPE local_sum = 0.0;
    if (lhs_idx < nlhs && ig < npw)
    {
        const thrust::complex<FPTYPE>* lhs_vec = lhs + lhs_idx * stride;
        const thrust::complex<FPTYPE>* rhs_vec = rhs + rhs_idx * stride;

        if (ig != ig_gge0)
        {
            FPTYPE charge = (thrust::conj(lhs_vec[ig]) * rhs_vec[ig]).real() / gg[ig] * charge_fac;
            if (nspin == 2 && gamma_only)
            {
                charge *= static_cast<FPTYPE>(2.0);
            }
            local_sum += charge;
        }

        if (include_magnetism && nspin == 2)
        {
            const FPTYPE mag_value = (thrust::conj(lhs_vec[npw + ig]) * rhs_vec[npw + ig]).real();
            FPTYPE mag = mag_value * mag_fac;
            if (gamma_only)
            {
                mag *= static_cast<FPTYPE>(2.0);
            }
            local_sum += mag;
            if (ig == 0)
            {
                local_sum += mag_value * mag_fac;
            }
        }
        else if (include_magnetism && nspin == 4)
        {
            FPTYPE mag_value = 0.0;
            for (int is = 1; is < 4; ++is)
            {
                mag_value += (thrust::conj(lhs_vec[is * npw + ig]) * rhs_vec[is * npw + ig]).real();
            }
            if (ig == ig_gge0)
            {
                if (ig_gge0 > 0)
                {
                    local_sum += mag_value * mag_fac;
                }
            }
            else
            {
                FPTYPE mag = mag_value * mag_fac;
                if (gamma_only)
                {
                    mag *= static_cast<FPTYPE>(2.0);
                }
                local_sum += mag;
            }
        }
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        partial_sums[pair * num_blocks + block] = sdata[0];
    }
}

template <typename FPTYPE>
__global__ void pack_spin_recip_kernel(
    thrust::complex<FPTYPE>* packed,
    const thrust::complex<FPTYPE>* spin_data,
    const int npw,
    const int nspin)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
    {
        return;
    }

    if (nspin == 2)
    {
        const thrust::complex<FPTYPE> rho_up = spin_data[ig];
        const thrust::complex<FPTYPE> rho_down = spin_data[npw + ig];
        packed[ig] = rho_up + rho_down;
        packed[npw + ig] = rho_up - rho_down;
    }
    else if (nspin == 4)
    {
        for (int is = 0; is < 4; ++is)
        {
            packed[is * npw + ig] = spin_data[is * npw + ig];
        }
    }
}

template <typename FPTYPE>
__global__ void unpack_spin_recip_kernel(
    thrust::complex<FPTYPE>* spin_data,
    const thrust::complex<FPTYPE>* packed,
    const int npw,
    const int nspin)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
    {
        return;
    }

    if (nspin == 2)
    {
        spin_data[ig] = static_cast<FPTYPE>(0.5) * (packed[ig] + packed[npw + ig]);
        spin_data[npw + ig] = static_cast<FPTYPE>(0.5) * (packed[ig] - packed[npw + ig]);
    }
    else if (nspin == 4)
    {
        for (int is = 0; is < 4; ++is)
        {
            spin_data[is * npw + ig] = packed[is * npw + ig];
        }
    }
}

template <typename FPTYPE>
__global__ void split_double_grid_recip_kernel(
    thrust::complex<FPTYPE>* smooth,
    thrust::complex<FPTYPE>* high_frequency,
    const thrust::complex<FPTYPE>* dense,
    const int smooth_npw,
    const int dense_npw,
    const int nspin)
{
    const int hf_npw = dense_npw - smooth_npw;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int smooth_size = nspin * smooth_npw;
    const int total = nspin * dense_npw;
    if (idx >= total)
    {
        return;
    }

    if (idx < smooth_size)
    {
        const int is = idx / smooth_npw;
        const int ig = idx - is * smooth_npw;
        smooth[idx] = dense[is * dense_npw + ig];
    }
    else
    {
        const int hf_idx = idx - smooth_size;
        const int is = hf_idx / hf_npw;
        const int ig = hf_idx - is * hf_npw;
        high_frequency[hf_idx] = dense[is * dense_npw + smooth_npw + ig];
    }
}

template <typename FPTYPE>
__global__ void combine_double_grid_recip_kernel(
    thrust::complex<FPTYPE>* dense,
    const thrust::complex<FPTYPE>* smooth,
    const thrust::complex<FPTYPE>* high_frequency,
    const int smooth_npw,
    const int dense_npw,
    const int nspin)
{
    const int hf_npw = dense_npw - smooth_npw;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int smooth_size = nspin * smooth_npw;
    const int total = nspin * dense_npw;
    if (idx >= total)
    {
        return;
    }

    if (idx < smooth_size)
    {
        const int is = idx / smooth_npw;
        const int ig = idx - is * smooth_npw;
        dense[is * dense_npw + ig] = smooth[idx];
    }
    else
    {
        const int hf_idx = idx - smooth_size;
        const int is = hf_idx / hf_npw;
        const int ig = hf_idx - is * hf_npw;
        dense[is * dense_npw + smooth_npw + ig] = high_frequency[hf_idx];
    }
}

// Kerker screening operator implementation
template <typename FPTYPE>
void kerker_screen_recip_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<FPTYPE>* drhog,
    const FPTYPE* gg,
    const FPTYPE gg0,
    const FPTYPE gg0_min,
    const int npw,
    const int nspin)
{
    const int block = ceil_div(npw, KERKER_THREADS_PER_BLOCK);
    kerker_screen_kernel<<<block, KERKER_THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<FPTYPE>*>(drhog),
        gg, gg0, gg0_min, npw, nspin);

    CHECK_LAST_CUDA_ERROR("kerker_screen_kernel launch");
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void pack_spin_recip_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<FPTYPE>* packed,
    const std::complex<FPTYPE>* spin_data,
    const int npw,
    const int nspin)
{
    const int block = ceil_div(npw, KERKER_THREADS_PER_BLOCK);
    pack_spin_recip_kernel<<<block, KERKER_THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<FPTYPE>*>(packed),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(spin_data),
        npw,
        nspin);
    CHECK_LAST_CUDA_ERROR("pack_spin_recip_kernel launch");
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void unpack_spin_recip_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<FPTYPE>* spin_data,
    const std::complex<FPTYPE>* packed,
    const int npw,
    const int nspin)
{
    const int block = ceil_div(npw, KERKER_THREADS_PER_BLOCK);
    unpack_spin_recip_kernel<<<block, KERKER_THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<FPTYPE>*>(spin_data),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(packed),
        npw,
        nspin);
    CHECK_LAST_CUDA_ERROR("unpack_spin_recip_kernel launch");
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void split_double_grid_recip_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<FPTYPE>* smooth,
    std::complex<FPTYPE>* high_frequency,
    const std::complex<FPTYPE>* dense,
    const int smooth_npw,
    const int dense_npw,
    const int nspin)
{
    const int hf_npw = dense_npw - smooth_npw;
    if (smooth_npw <= 0 || hf_npw < 0 || nspin <= 0)
    {
        return;
    }
    const int block = ceil_div(nspin * dense_npw, KERKER_THREADS_PER_BLOCK);
    split_double_grid_recip_kernel<<<block, KERKER_THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<FPTYPE>*>(smooth),
        reinterpret_cast<thrust::complex<FPTYPE>*>(high_frequency),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(dense),
        smooth_npw,
        dense_npw,
        nspin);
    CHECK_LAST_CUDA_ERROR("split_double_grid_recip_kernel launch");
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void combine_double_grid_recip_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    std::complex<FPTYPE>* dense,
    const std::complex<FPTYPE>* smooth,
    const std::complex<FPTYPE>* high_frequency,
    const int smooth_npw,
    const int dense_npw,
    const int nspin)
{
    const int hf_npw = dense_npw - smooth_npw;
    if (smooth_npw <= 0 || hf_npw < 0 || nspin <= 0)
    {
        return;
    }
    const int block = ceil_div(nspin * dense_npw, KERKER_THREADS_PER_BLOCK);
    combine_double_grid_recip_kernel<<<block, KERKER_THREADS_PER_BLOCK>>>(
        reinterpret_cast<thrust::complex<FPTYPE>*>(dense),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(smooth),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(high_frequency),
        smooth_npw,
        dense_npw,
        nspin);
    CHECK_LAST_CUDA_ERROR("combine_double_grid_recip_kernel launch");
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void inner_product_recip_hartree_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
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
    FPTYPE* workspace)
{
    if (npw <= 0 || nlhs <= 0 || nrhs <= 0)
    {
        return;
    }

    const cudaDeviceProp prop = current_device_properties();
    const long long npairs_ll = static_cast<long long>(nlhs) * static_cast<long long>(nrhs);
    if (npairs_ll > static_cast<long long>(std::numeric_limits<int>::max()))
    {
        ModuleBase::WARNING_QUIT(
            "inner_product_recip_hartree_batch_op",
            "too many batched inner-product pairs: nlhs=" + std::to_string(nlhs)
                + ", nrhs=" + std::to_string(nrhs));
    }
    const int npairs = static_cast<int>(npairs_ll);
    const int threads = choose_reduction_threads(npw, npairs, prop);
    validate_reduction_launch(npw, npairs, threads, prop, "inner_product_recip_hartree_batch_op");
    const int num_blocks = ceil_div(npw, threads);
    const long long total_partial_blocks = static_cast<long long>(npairs) * num_blocks;
    const std::size_t shared_bytes = static_cast<std::size_t>(threads) * sizeof(FPTYPE);

    inner_product_hartree_batch_partial_kernel<<<static_cast<unsigned int>(total_partial_blocks), threads, shared_bytes>>>(
        reinterpret_cast<const thrust::complex<FPTYPE>*>(lhs),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(rhs),
        gg,
        workspace,
        npw,
        nlhs,
        nrhs,
        num_blocks,
        ig_gge0,
        tpiba2);
    CHECK_LAST_CUDA_ERROR("inner_product_hartree_batch_partial_kernel launch");
    CHECK_CUDA_SYNC();

    inner_product_hartree_batch_final_kernel<<<npairs, threads, shared_bytes>>>(
        workspace,
        result,
        npairs,
        num_blocks);
    CHECK_LAST_CUDA_ERROR("inner_product_hartree_batch_final_kernel launch");
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void inner_product_recip_hartree_spin_batch_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
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
    FPTYPE* workspace)
{
    if (npw <= 0 || nspin <= 0 || nlhs <= 0 || nrhs <= 0)
    {
        return;
    }

    const cudaDeviceProp prop = current_device_properties();
    const long long npairs_ll = static_cast<long long>(nlhs) * static_cast<long long>(nrhs);
    if (npairs_ll > static_cast<long long>(std::numeric_limits<int>::max()))
    {
        ModuleBase::WARNING_QUIT(
            "inner_product_recip_hartree_spin_batch_op",
            "too many batched inner-product pairs: nlhs=" + std::to_string(nlhs)
                + ", nrhs=" + std::to_string(nrhs));
    }
    const int npairs = static_cast<int>(npairs_ll);
    const int threads = choose_reduction_threads(npw, npairs, prop);
    validate_reduction_launch(npw, npairs, threads, prop, "inner_product_recip_hartree_spin_batch_op");
    const int num_blocks = ceil_div(npw, threads);
    const long long total_partial_blocks = static_cast<long long>(npairs) * num_blocks;
    const std::size_t shared_bytes = static_cast<std::size_t>(threads) * sizeof(FPTYPE);

    inner_product_hartree_spin_batch_partial_kernel<<<static_cast<unsigned int>(total_partial_blocks),
                                                      threads,
                                                      shared_bytes>>>(
        reinterpret_cast<const thrust::complex<FPTYPE>*>(lhs),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(rhs),
        gg,
        workspace,
        npw,
        nspin,
        nlhs,
        nrhs,
        num_blocks,
        ig_gge0,
        gamma_only,
        include_magnetism,
        charge_fac,
        mag_fac);
    CHECK_LAST_CUDA_ERROR("inner_product_hartree_spin_batch_partial_kernel launch");
    CHECK_CUDA_SYNC();

    inner_product_hartree_batch_final_kernel<<<npairs, threads, shared_bytes>>>(
        workspace,
        result,
        npairs,
        num_blocks);
    CHECK_LAST_CUDA_ERROR("inner_product_hartree_batch_final_kernel launch");
    CHECK_CUDA_SYNC();
}

// Explicit template instantiations
template struct kerker_screen_recip_op<float, base_device::DEVICE_GPU>;
template struct kerker_screen_recip_op<double, base_device::DEVICE_GPU>;
template struct pack_spin_recip_op<float, base_device::DEVICE_GPU>;
template struct pack_spin_recip_op<double, base_device::DEVICE_GPU>;
template struct unpack_spin_recip_op<float, base_device::DEVICE_GPU>;
template struct unpack_spin_recip_op<double, base_device::DEVICE_GPU>;
template struct split_double_grid_recip_op<float, base_device::DEVICE_GPU>;
template struct split_double_grid_recip_op<double, base_device::DEVICE_GPU>;
template struct combine_double_grid_recip_op<float, base_device::DEVICE_GPU>;
template struct combine_double_grid_recip_op<double, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_batch_op<float, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_spin_batch_op<float, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_spin_batch_op<double, base_device::DEVICE_GPU>;

} // namespace elecstate

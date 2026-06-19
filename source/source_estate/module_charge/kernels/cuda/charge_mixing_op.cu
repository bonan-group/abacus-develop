#include "source_estate/module_charge/kernels/charge_mixing_op.h"
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
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

// CUDA kernel for inner product with 1/G^2 weight
template <typename FPTYPE>
__global__ void inner_product_hartree_kernel(
    const thrust::complex<FPTYPE>* rhog1,
    const thrust::complex<FPTYPE>* rhog2,
    const FPTYPE* gg,
    FPTYPE* partial_sums,
    const int npw,
    const int ig_gge0,
    const FPTYPE tpiba2)
{
    extern __shared__ unsigned char shared_raw[];
    FPTYPE* sdata = reinterpret_cast<FPTYPE*>(shared_raw);
    int tid = threadIdx.x;
    int ig = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute local contribution, skipping G=0
    FPTYPE local_sum = 0.0;
    if (ig < npw && ig != ig_gge0)
    {
        thrust::complex<FPTYPE> prod = thrust::conj(rhog1[ig]) * rhog2[ig];
        local_sum = prod.real() / gg[ig] * tpiba2;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    // Block reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0)
    {
        partial_sums[blockIdx.x] = sdata[0];
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

    CHECK_CUDA_SYNC();
}

// Inner product operator implementation
template <typename FPTYPE>
FPTYPE inner_product_recip_hartree_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const std::complex<FPTYPE>* rhog1,
    const std::complex<FPTYPE>* rhog2,
    const FPTYPE* gg,
    const int npw,
    const int ig_gge0,
    const FPTYPE tpiba2,
    FPTYPE* workspace)
{
    const cudaDeviceProp prop = current_device_properties();
    const int threads = choose_reduction_threads(npw, 1, prop);
    validate_reduction_launch(npw, 1, threads, prop, "inner_product_recip_hartree_op");
    const int num_blocks = ceil_div(npw, threads);
    const std::size_t shared_bytes = static_cast<std::size_t>(threads) * sizeof(FPTYPE);

    // Launch kernel for partial sums
    inner_product_hartree_kernel<<<num_blocks, threads, shared_bytes>>>(
        reinterpret_cast<const thrust::complex<FPTYPE>*>(rhog1),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(rhog2),
        gg, workspace, npw, ig_gge0, tpiba2);

    CHECK_CUDA_SYNC();

    // Final reduction using thrust
    thrust::device_ptr<FPTYPE> dev_ptr(workspace);
    return thrust::reduce(dev_ptr, dev_ptr + num_blocks);
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
    CHECK_CUDA_SYNC();

    inner_product_hartree_batch_final_kernel<<<npairs, threads, shared_bytes>>>(
        workspace,
        result,
        npairs,
        num_blocks);
    CHECK_CUDA_SYNC();
}

// Explicit template instantiations
template struct kerker_screen_recip_op<float, base_device::DEVICE_GPU>;
template struct kerker_screen_recip_op<double, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_op<float, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_op<double, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_batch_op<float, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>;

} // namespace elecstate

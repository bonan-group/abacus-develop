#include "source_estate/module_charge/kernels/charge_mixing_op.h"
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <base/macros/macros.h>

#define THREADS_PER_BLOCK 256

namespace elecstate {

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
    __shared__ FPTYPE sdata[THREADS_PER_BLOCK];
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
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    kerker_screen_kernel<<<block, THREADS_PER_BLOCK>>>(
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
    const int num_blocks = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel for partial sums
    inner_product_hartree_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        reinterpret_cast<const thrust::complex<FPTYPE>*>(rhog1),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(rhog2),
        gg, workspace, npw, ig_gge0, tpiba2);

    CHECK_CUDA_SYNC();

    // Final reduction using thrust
    thrust::device_ptr<FPTYPE> dev_ptr(workspace);
    return thrust::reduce(dev_ptr, dev_ptr + num_blocks);
}

// Explicit template instantiations
template struct kerker_screen_recip_op<float, base_device::DEVICE_GPU>;
template struct kerker_screen_recip_op<double, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_op<float, base_device::DEVICE_GPU>;
template struct inner_product_recip_hartree_op<double, base_device::DEVICE_GPU>;

} // namespace elecstate

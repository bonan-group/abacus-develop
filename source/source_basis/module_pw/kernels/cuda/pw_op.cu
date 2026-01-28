#include "source_basis/module_pw/kernels/pw_op.h"

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <base/macros/macros.h>

namespace ModulePW {

#define THREADS_PER_BLOCK 256

template<class FPTYPE>
__global__ void set_3d_fft_box(
    const int npwk,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < npwk)
    {
        int xx = box_index[idx];
        out[xx] = in[idx];
    }
}

template<class FPTYPE>
__global__ void set_3d_fft_box_batch(
    const int npwk,
    const int nxyz,
    const int in_stride,
    const int* box_index,
    const int nbatch,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int ib = 0; ib < nbatch; ib++)
    {
        for (int i = idx; i < npwk; i += stride)
        {
            int xx = box_index[i];
            out[xx] = in[i];

        }
        // Increment pointer
        out = out + nxyz;
        in = in + in_stride;
    }
}



template<class FPTYPE>
__global__ void set_recip_to_real_output(
    const int nrxx,
    const bool add,
    const FPTYPE factor,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nrxx) {return;}
    if(add) {
        out[idx] += factor * in[idx];
    }
    else {
        out[idx] = in[idx];
    }
}

template<class FPTYPE>
__global__ void set_recip_to_real_output(
    const int nrxx,
    const bool add,
    const FPTYPE factor,
    const thrust::complex<FPTYPE>* in,
    FPTYPE* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= nrxx) {return;}
    if(add) {
        out[idx] += factor * in[idx].real();
    }
    else {
        out[idx] = in[idx].real();
    }
}

template<class FPTYPE>
__global__ void set_recip_to_real_output_batch(
    const int nrxx,
    const int npw,
    const int nbatch,
    const int add,
    const FPTYPE* factor,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (add) {
        for (int ib = 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < nrxx; i += stride)
            {
                out[i] += factor[ib] * in[i];
            }
            out = out + nrxx;
            in = in + npw;

        }
    }
    else
    {
        for (int ib = 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < nrxx; i += stride)
            {
                out[i] =  in[i];
            }
            out = out + nrxx;
            in = in + npw;
        }
    }
}

template<class FPTYPE>
__global__ void set_recip_to_real_output_batch(
    const int nrxx,
    const int npw,
    const int nbatch,
    const int add,
    const FPTYPE* factor,
    const thrust::complex<FPTYPE>* in,
    FPTYPE* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (add) {
        for (int ib = 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < nrxx; i += stride)
            {
                out[i] += factor[ib] * in[i].real();
            }
            out = out + nrxx;
            in = in + npw;
        }
    }
    else
    {
        for (int ib = 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < nrxx; i += stride)
            {
                out[i] =  in[i].real();
            }
            out = out + nrxx;
            in = in + npw;

        }
    }
}


template<class FPTYPE>
__global__ void set_real_to_recip_output(
    const int npwk,
    const FPTYPE one_over_nxyz,
    const bool add,
    const FPTYPE factor,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= npwk) {return;}
    if(add) {
        out[idx] += factor * one_over_nxyz * in[box_index[idx]];
    }
    else {
        out[idx] = in[box_index[idx]] * one_over_nxyz;
    }
}

template<class FPTYPE>
__global__ void set_real_to_recip_output(
    const int npwk,
    const FPTYPE one_over_nxyz,
    const bool add,
    const FPTYPE factor,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    FPTYPE* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= npwk) {return;}
    if(add) {
        out[idx] += factor * one_over_nxyz * in[box_index[idx]].real();
    }
    else {
        out[idx] = in[box_index[idx]].real() * one_over_nxyz;
    }
}

template<class FPTYPE>
__global__ void set_real_to_recip_output_batch(
    const int npwk,
    const int nxyz,
    const int out_stride,
    const FPTYPE one_over_nxyz,
    const int nbatch,
    const int add,
    const FPTYPE* factor,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    thrust::complex<FPTYPE>* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (add) {
        for (int ib= 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < npwk; i += stride)
            {
                out[i] += factor[ib] * one_over_nxyz * in[box_index[i]];
            }
            in = in + nxyz;
            out = out + out_stride;

        }
    }
    else
    {
        for (int ib= 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < npwk; i += stride)
            {
                out[i] = in[box_index[i]] * one_over_nxyz;
            }
            in = in + nxyz;
            out = out + out_stride;
        }
    }
}

template<class FPTYPE>
__global__ void set_real_to_recip_output_batch(
    const int npwk,
    const int nxyz,
    const int out_stride,
    const FPTYPE one_over_nxyz,
    const int nbatch,
    const int add,
    const FPTYPE* factor,
    const int* box_index,
    const thrust::complex<FPTYPE>* in,
    FPTYPE* out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (add) {
        for (int ib= 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < npwk; i += stride)
            {
                out[i] += factor[ib] * one_over_nxyz * in[box_index[i]].real();
            }
            in = in + nxyz;
            out = out + out_stride;

        }
    }
    else
    {
        for (int ib= 0; ib < nbatch; ib++)
        {
            for (int i = idx; i < npwk; i += stride)
            {
                out[i] = in[box_index[i]].real() * one_over_nxyz;
            }
            in = in + nxyz;
            out = out + out_stride;
        }
    }
}


template <typename FPTYPE>
void set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const int npwk,
                                                                    const int* box_index,
                                                                    const std::complex<FPTYPE>* in,
                                                                    std::complex<FPTYPE>* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_3d_fft_box<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>::operator_batch(const int npwk,
                                                                    const int nxyz,
                                                                    const int in_stride,
                                                                    const int* box_index,
                                                                    const int nbatch,
                                                                    const std::complex<FPTYPE>* in,
                                                                    std::complex<FPTYPE>* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_3d_fft_box_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        nxyz,
        in_stride,
        box_index,
        nbatch,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const int nrxx,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const std::complex<FPTYPE>* in,
                                                                              std::complex<FPTYPE>* out)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_recip_to_real_output<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        nrxx,
        add,
        factor,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>::operator_batch(const int nrxx,
                                                                              const int npw,
                                                                              const int nbatch,
                                                                              const bool add,
                                                                              const FPTYPE* factor,
                                                                              const std::complex<FPTYPE>* in,
                                                                              std::complex<FPTYPE>* out)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_recip_to_real_output_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        nrxx,
        npw,
        nbatch,
        add,
        factor,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}



template <typename FPTYPE>
void set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const int nrxx,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const std::complex<FPTYPE>* in,
                                                                              FPTYPE* out)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_recip_to_real_output<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        nrxx,
        add,
        factor,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<FPTYPE*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>::operator_batch(const int nrxx,
                                                                              const int npw,
                                                                              const int nbatch,
                                                                              const bool add,
                                                                              const FPTYPE* factor,
                                                                              const std::complex<FPTYPE>* in,
                                                                              FPTYPE* out)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_recip_to_real_output_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        nrxx,
        npw,
        nbatch,
        add,
        factor,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<FPTYPE*>(out)
    );

    cudaCheckOnDebug();
}



template <typename FPTYPE>
void set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const int npwk,
                                                                              const int nxyz,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const int* box_index,
                                                                              const std::complex<FPTYPE>* in,
                                                                              std::complex<FPTYPE>* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_real_to_recip_output<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        static_cast<FPTYPE>(1.0/nxyz),
        add,
        factor,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>::operator_batch(const int npwk,
                                                                              const int nxyz,
                                                                              const int out_stride,
                                                                              const int nbatch,
                                                                              const bool add,
                                                                              const FPTYPE* factor,
                                                                              const int* box_index,
                                                                              const std::complex<FPTYPE>* in,
                                                                              std::complex<FPTYPE>* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_real_to_recip_output_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        nxyz,
        out_stride,
        static_cast<FPTYPE>(1.0/nxyz),
        nbatch,
        add,
        factor,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<thrust::complex<FPTYPE>*>(out));

    cudaCheckOnDebug();
}

template <typename FPTYPE>
void set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const int npwk,
                                                                              const int nxyz,
                                                                              const bool add,
                                                                              const FPTYPE factor,
                                                                              const int* box_index,
                                                                              const std::complex<FPTYPE>* in,
                                                                              FPTYPE* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_real_to_recip_output<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        static_cast<FPTYPE>(1.0/nxyz),
        add,
        factor,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<FPTYPE*>(out));

    cudaCheckOnDebug();
}


template <typename FPTYPE>
void set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>::operator_batch(const int npwk,
                                                                              const int nxyz,
                                                                              const int out_stride,
                                                                              const int nbatch,
                                                                              const bool add,
                                                                              const FPTYPE* factor,
                                                                              const int* box_index,
                                                                              const std::complex<FPTYPE>* in,
                                                                              FPTYPE* out)
{
    const int block = (npwk + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    set_real_to_recip_output_batch<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npwk,
        nxyz,
        out_stride,
        static_cast<FPTYPE>(1.0/nxyz),
        nbatch,
        add,
        factor,
        box_index,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
        reinterpret_cast<FPTYPE*>(out));

    cudaCheckOnDebug();
}



template struct set_3d_fft_box_op<float, base_device::DEVICE_GPU>;
template struct set_recip_to_real_output_op<float, base_device::DEVICE_GPU>;
template struct set_real_to_recip_output_op<float, base_device::DEVICE_GPU>;
template struct set_3d_fft_box_op<double, base_device::DEVICE_GPU>;
template struct set_recip_to_real_output_op<double, base_device::DEVICE_GPU>;
template struct set_real_to_recip_output_op<double, base_device::DEVICE_GPU>;

}  // namespace ModulePW

#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"
#include "source_base/module_device/device_check.h"
#include "source_basis/module_pw/pw_basis_k.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thrust/complex.h>

namespace hamilt
{
namespace
{
__device__ int exx_positive_mod(int value, int modulus)
{
    const int result = value % modulus;
    return result < 0 ? result + modulus : result;
}
}

template <typename FPTYPE>
__global__ void exx_rotate_realspace_kernel(const thrust::complex<FPTYPE>* representative_real,
                                            thrust::complex<FPTYPE>* full_real,
                                            std::size_t nrxx,
                                            std::size_t total_size,
                                            int nx,
                                            int ny,
                                            int nz,
                                            bool time_reversal,
                                            double e11,
                                            double e12,
                                            double e13,
                                            double e21,
                                            double e22,
                                            double e23,
                                            double e31,
                                            double e32,
                                            double e33,
                                            double tx,
                                            double ty,
                                            double tz,
                                            double rep_kx,
                                            double rep_ky,
                                            double rep_kz,
                                            double full_kx,
                                            double full_ky,
                                            double full_kz)
{
    const std::size_t first_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t index = first_index; index < total_size; index += stride)
    {
        const std::size_t ib = index / nrxx;
        const std::size_t ir = index - ib * nrxx;
        const int iz = static_cast<int>(ir % static_cast<std::size_t>(nz));
        const std::size_t ixy = ir / static_cast<std::size_t>(nz);
        const int iy = static_cast<int>(ixy % static_cast<std::size_t>(ny));
        const int ix = static_cast<int>(ixy / static_cast<std::size_t>(ny));

        const double sign = time_reversal ? -1.0 : 1.0;
        const double source_x = sign * static_cast<double>(ix) / nx;
        const double source_y = sign * static_cast<double>(iy) / ny;
        const double source_z = sign * static_cast<double>(iz) / nz;
        const double rep_x = source_x * e11 + source_y * e21 + source_z * e31 + tx;
        const double rep_y = source_x * e12 + source_y * e22 + source_z * e32 + ty;
        const double rep_z = source_x * e13 + source_y * e23 + source_z * e33 + tz;

        const int rep_ix = exx_positive_mod(static_cast<int>(llround(rep_x * nx)), nx);
        const int rep_iy = exx_positive_mod(static_cast<int>(llround(rep_y * ny)), ny);
        const int rep_iz = exx_positive_mod(static_cast<int>(llround(rep_z * nz)), nz);
        const std::size_t rep_ir = static_cast<std::size_t>(rep_iz)
                                   + static_cast<std::size_t>(rep_iy) * nz
                                   + static_cast<std::size_t>(rep_ix) * ny * nz;

        const double phase_arg = 2.0 * 3.14159265358979323846
                                 * (rep_kx * rep_x + rep_ky * rep_y + rep_kz * rep_z
                                    - full_kx * source_x - full_ky * source_y - full_kz * source_z);
        thrust::complex<FPTYPE> value
            = thrust::complex<FPTYPE>(static_cast<FPTYPE>(cos(phase_arg)), static_cast<FPTYPE>(sin(phase_arg)))
              * representative_real[ib * nrxx + rep_ir];
        if (time_reversal)
        {
            value = thrust::conj(value);
        }
        full_real[index] = value;
    }
}

template <typename FPTYPE>
struct exx_rotate_realspace_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const ModulePW::PW_Basis_K* wfcpw,
                    const K_Vectors::ExxFullPoint& full_point,
                    int rep_spin_index,
                    const T* representative_real,
                    T* full_real,
                    int batch_count)
    {
        if (wfcpw->poolnproc != 1)
        {
            throw std::runtime_error("GPU PW EXX real-space symmetry rotation requires poolnproc=1");
        }
        validate_exx_realspace_symmetry_grid(wfcpw, full_point);
        std::size_t total_size = 0;
        if (batch_count <= 0
            || !checked_exx_size_product(static_cast<std::size_t>(batch_count),
                                         static_cast<std::size_t>(wfcpw->nrxx),
                                         total_size))
        {
            throw std::overflow_error("GPU PW EXX real-space rotation size is invalid");
        }
        const int threads_per_block = 256;
        const std::size_t required_blocks = (total_size + threads_per_block - 1) / threads_per_block;
        const int num_blocks = static_cast<int>(std::min<std::size_t>(required_blocks, 65535));
        const ModuleBase::Matrix3& matrix = full_point.gmatrix;
        const ModuleBase::Vector3<double>& rep_kvec = wfcpw->kvec_d[rep_spin_index];
        exx_rotate_realspace_kernel<FPTYPE><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const thrust::complex<FPTYPE>*>(representative_real),
            reinterpret_cast<thrust::complex<FPTYPE>*>(full_real),
            static_cast<std::size_t>(wfcpw->nrxx),
            total_size,
            wfcpw->nx,
            wfcpw->ny,
            wfcpw->nz,
            full_point.time_reversal,
            matrix.e11,
            matrix.e12,
            matrix.e13,
            matrix.e21,
            matrix.e22,
            matrix.e23,
            matrix.e31,
            matrix.e32,
            matrix.e33,
            full_point.gtrans.x,
            full_point.gtrans.y,
            full_point.gtrans.z,
            rep_kvec.x,
            rep_kvec.y,
            rep_kvec.z,
            full_point.full_kvec_d.x,
            full_point.full_kvec_d.y,
            full_point.full_kvec_d.z);
        CHECK_LAST_CUDA_ERROR("exx_rotate_realspace_kernel");
        CHECK_CUDA_SYNC();
    }
};

template struct exx_rotate_realspace_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_rotate_realspace_op<std::complex<double>, base_device::DEVICE_GPU>;

template <typename FPTYPE>
__global__ void exx_conjugate_real_kernel(const thrust::complex<FPTYPE>* in,
                                          thrust::complex<FPTYPE>* out,
                                          std::size_t nrxx)
{
    const std::size_t first_ir = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t ir = first_ir; ir < nrxx; ir += stride)
    {
        out[ir] = thrust::conj(in[ir]);
    }
}

template <typename FPTYPE>
struct exx_conjugate_real_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T* in, T* out, std::size_t nrxx)
    {
        if (nrxx == 0)
        {
            return;
        }
        const int threads_per_block = 256;
        const std::size_t required_blocks = (nrxx + threads_per_block - 1) / threads_per_block;
        const int num_blocks = static_cast<int>(std::min<std::size_t>(required_blocks, 65535));
        exx_conjugate_real_kernel<FPTYPE><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
            reinterpret_cast<thrust::complex<FPTYPE>*>(out),
            nrxx);

        CHECK_LAST_CUDA_ERROR("exx_conjugate_real_kernel");
        CHECK_CUDA_SYNC();
    }
};

template struct exx_conjugate_real_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_conjugate_real_op<std::complex<double>, base_device::DEVICE_GPU>;

template <typename FPTYPE>
__global__ void exx_gather_recip_kernel(const thrust::complex<FPTYPE>* in,
                                        thrust::complex<FPTYPE>* out,
                                        const int* map,
                                        int nout)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < nout)
    {
        const int src = map[ig];
        out[ig] = src >= 0 ? in[src] : thrust::complex<FPTYPE>(0, 0);
    }
}

template <typename FPTYPE>
__global__ void exx_scatter_add_recip_kernel(const thrust::complex<FPTYPE>* in,
                                             thrust::complex<FPTYPE>* out,
                                             const int* map,
                                             int nin,
                                             thrust::complex<FPTYPE> factor)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig < nin)
    {
        const int dst = map[ig];
        if (dst >= 0)
        {
            out[dst] += factor * in[ig];
        }
    }
}

template <typename FPTYPE>
struct exx_gather_recip_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T* in, T* out, const int* map, int nout)
    {
        const int threads_per_block = 256;
        const int num_blocks = (nout + threads_per_block - 1) / threads_per_block;
        exx_gather_recip_kernel<FPTYPE><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
            reinterpret_cast<thrust::complex<FPTYPE>*>(out),
            map,
            nout);
        CHECK_LAST_CUDA_ERROR("exx_gather_recip_kernel");
        CHECK_CUDA_SYNC();
    }
};

template <typename FPTYPE>
struct exx_scatter_add_recip_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T* in, T* out, const int* map, int nin, T factor)
    {
        const int threads_per_block = 256;
        const int num_blocks = (nin + threads_per_block - 1) / threads_per_block;
        exx_scatter_add_recip_kernel<FPTYPE><<<num_blocks, threads_per_block>>>(
            reinterpret_cast<const thrust::complex<FPTYPE>*>(in),
            reinterpret_cast<thrust::complex<FPTYPE>*>(out),
            map,
            nin,
            thrust::complex<FPTYPE>(static_cast<FPTYPE>(factor.real()), static_cast<FPTYPE>(factor.imag())));
        CHECK_LAST_CUDA_ERROR("exx_scatter_add_recip_kernel");
        CHECK_CUDA_SYNC();
    }
};

template struct exx_gather_recip_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_gather_recip_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct exx_scatter_add_recip_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_scatter_add_recip_op<std::complex<double>, base_device::DEVICE_GPU>;
} // namespace hamilt

#include "source_pw/module_pwdft/kernels/nonlocal_op.h"

#include <complex>
#include <thrust/complex.h>

#include <hip/hip_runtime.h>
#include <base/macros/macros.h>

#include "source_pw/module_pwdft/kernels/nonlocal_op_gpu_kernels.h"

using namespace hamilt;

#define THREADS_PER_BLOCK 256

template <typename FPTYPE>
void hamilt::uspp_qgm_build_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*dev*/,
    int ntype,
    int nh_tot,
    int npw,
    int lmaxq,
    int radial_pair_count,
    int nqxq,
    int max_terms,
    FPTYPE dq,
    FPTYPE tpiba,
    const FPTYPE* gcar,
    const int* pair_term_count,
    const int* pair_radial_index,
    const int* pair_l,
    const int* pair_lm,
    const std::complex<FPTYPE>* pair_coefficient,
    const FPTYPE* qrad,
    const FPTYPE* ylm,
    std::complex<FPTYPE>* qgm)
{
    const int count = ntype * nh_tot * npw;
    if (count == 0)
    {
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::uspp_qgm_build<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       ntype,
                       nh_tot,
                       npw,
                       lmaxq,
                       radial_pair_count,
                       nqxq,
                       max_terms,
                       dq,
                       tpiba,
                       gcar,
                       pair_term_count,
                       pair_radial_index,
                       pair_l,
                       pair_lm,
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(pair_coefficient),
                       qrad,
                       ylm,
                       reinterpret_cast<thrust::complex<FPTYPE>*>(qgm));
    hipCheckOnDebug();
}

template <typename FPTYPE>
void hamilt::uspp_overlap_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*dev*/,
    int atom_count,
    int nbands,
    int nh,
    int nhm,
    int nkb,
    int projector_offset,
    bool projector_major,
    const FPTYPE* qq,
    std::complex<FPTYPE>* ps,
    const std::complex<FPTYPE>* becp)
{
    const int count = atom_count * nbands * nh;
    if (count == 0)
    {
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::uspp_overlap<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       atom_count,
                       nbands,
                       nh,
                       nhm,
                       nkb,
                       projector_offset,
                       projector_major,
                       qq,
                       reinterpret_cast<thrust::complex<FPTYPE>*>(ps),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(becp));
    hipCheckOnDebug();
}

template <typename FPTYPE>
void hamilt::uspp_deeq_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*dev*/,
    int nspin,
    int atom_count,
    int nh,
    int nhm,
    int npw,
    int atom_offset,
    int nat,
    FPTYPE omega,
    bool gamma_only,
    int g0_index,
    const std::complex<FPTYPE>* vaux,
    const std::complex<FPTYPE>* qgm,
    const std::complex<FPTYPE>* phase,
    const FPTYPE* dvan,
    FPTYPE* deeq)
{
    const int count = nspin * atom_count * nh * (nh + 1) / 2;
    if (count == 0)
    {
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::uspp_deeq<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       nspin,
                       atom_count,
                       nh,
                       nhm,
                       npw,
                       atom_offset,
                       nat,
                       omega,
                       gamma_only,
                       g0_index,
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(vaux),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(qgm),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(phase),
                       dvan,
                       deeq);
    hipCheckOnDebug();
}

template <typename FPTYPE>
void hamilt::uspp_force_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*dev*/,
    int nspin,
    int atom_count,
    int nij,
    int npw,
    int atom_offset,
    int nat,
    int nh_tot,
    FPTYPE omega,
    FPTYPE tpiba,
    const std::complex<FPTYPE>* vaux,
    const std::complex<FPTYPE>* qgm,
    const std::complex<FPTYPE>* phase,
    const FPTYPE* gcar,
    const FPTYPE* becsum,
    FPTYPE* force)
{
    const int count = atom_count * 3;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::uspp_force<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       nspin,
                       atom_count,
                       nij,
                       npw,
                       atom_offset,
                       nat,
                       nh_tot,
                       omega,
                       tpiba,
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(vaux),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(qgm),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(phase),
                       gcar,
                       becsum,
                       force);
    hipCheckOnDebug();
}

template <typename FPTYPE>
void hamilt::uspp_stress_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*dev*/,
    int nspin,
    int atom_count,
    int nij,
    int npw,
    int atom_offset,
    int nat,
    int nh_tot,
    int ipol,
    FPTYPE tpiba,
    const std::complex<FPTYPE>* vaux,
    const std::complex<FPTYPE>* dqgm,
    const std::complex<FPTYPE>* phase,
    const FPTYPE* gcar,
    const FPTYPE* becsum,
    FPTYPE* stress)
{
    const int count = atom_count * 3;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::uspp_stress<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       nspin,
                       atom_count,
                       nij,
                       npw,
                       atom_offset,
                       nat,
                       nh_tot,
                       ipol,
                       tpiba,
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(vaux),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(dqgm),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(phase),
                       gcar,
                       becsum,
                       stress);
    hipCheckOnDebug();
}

template <typename FPTYPE>
void hamilt::nonlocal_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                                         const int& l1,
                                                                         const int& l2,
                                                                         const int& l3,
                                                                         int& sum,
                                                                         int& iat,
                                                                         const int& spin,
                                                                         const int& nkb,
                                                                         const int& deeq_x,
                                                                         const int& deeq_y,
                                                                         const int& deeq_z,
                                                                         const FPTYPE* deeq,
                                                                         std::complex<FPTYPE>* ps,
                                                                         const std::complex<FPTYPE>* becp)
{
  // denghui implement 20221019
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::nonlocal_pw<FPTYPE>), dim3(l1 * l2), dim3(THREADS_PER_BLOCK), 0, 0,
    l1, l2, l3, // loop size
    sum, iat, spin, nkb,   // control params
    deeq_x, deeq_y, deeq_z, deeq,  // deeq realArray operator()
    reinterpret_cast<thrust::complex<FPTYPE>*>(ps), // array of data
    reinterpret_cast<const thrust::complex<FPTYPE>*>(becp)); // array of data

   hipCheckOnDebug();

  iat += l1;
  sum += l1 * l3;
}

template <typename FPTYPE>
void hamilt::nonlocal_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* dev,
                                                                         const int& l1,
                                                                         const int& l2,
                                                                         const int& l3,
                                                                         int& sum,
                                                                         int& iat,
                                                                         const int& nkb,
                                                                         const int& deeq_x,
                                                                         const int& deeq_y,
                                                                         const int& deeq_z,
                                                                         const std::complex<FPTYPE>* deeq_nc,
                                                                         std::complex<FPTYPE>* ps,
                                                                         const std::complex<FPTYPE>* becp)
{
  // denghui implement 20221109
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hamilt::nonlocal_gpu_detail::nonlocal_pw<FPTYPE>), dim3(l1 * l2 / 2), dim3(THREADS_PER_BLOCK), 0, 0,
    l1, l2, l3, // loop size
    sum, iat, nkb,   // control params
    deeq_x, deeq_y, deeq_z,
    reinterpret_cast<const thrust::complex<FPTYPE>*>(deeq_nc),  // deeq realArray operator()
    reinterpret_cast<thrust::complex<FPTYPE>*>(ps), // array of data
    reinterpret_cast<const thrust::complex<FPTYPE>*>(becp)); // array of data

   hipCheckOnDebug();

  iat += l1;
  sum += l1 * l3;
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
}

namespace hamilt{
template struct nonlocal_pw_op<float, base_device::DEVICE_GPU>;
template struct nonlocal_pw_op<double, base_device::DEVICE_GPU>;
template struct uspp_qgm_build_op<float, base_device::DEVICE_GPU>;
template struct uspp_qgm_build_op<double, base_device::DEVICE_GPU>;
template struct uspp_overlap_op<float, base_device::DEVICE_GPU>;
template struct uspp_overlap_op<double, base_device::DEVICE_GPU>;
template struct uspp_deeq_op<float, base_device::DEVICE_GPU>;
template struct uspp_deeq_op<double, base_device::DEVICE_GPU>;
template struct uspp_force_op<float, base_device::DEVICE_GPU>;
template struct uspp_force_op<double, base_device::DEVICE_GPU>;
template struct uspp_stress_op<float, base_device::DEVICE_GPU>;
template struct uspp_stress_op<double, base_device::DEVICE_GPU>;
}

#include "source_estate/kernels/elecstate_op.h"
#include <thrust/complex.h>

#include <hip/hip_runtime.h>
#include <base/macros/macros.h>

#define THREADS_PER_BLOCK 256

namespace elecstate {

template <typename FPTYPE>
__global__ void uspp_atom_phase(const int atom_count,
                                const int npw,
                                const FPTYPE* gcar,
                                const FPTYPE* tau,
                                thrust::complex<FPTYPE>* phase)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atom_count * npw)
    {
        return;
    }
    const int ia = index / npw;
    const int ig = index % npw;
    const FPTYPE two_pi = static_cast<FPTYPE>(6.283185307179586476925286766559);
    const FPTYPE arg = two_pi * (gcar[3 * ig] * tau[3 * ia]
                                 + gcar[3 * ig + 1] * tau[3 * ia + 1]
                                 + gcar[3 * ig + 2] * tau[3 * ia + 2]);
    phase[index] = thrust::complex<FPTYPE>(cos(arg), -sin(arg));
}

template <typename FPTYPE>
__global__ void uspp_pack_becsum(const int atom_count,
                                 const int nij,
                                 const int nat,
                                 const int nh_tot,
                                 const int spin,
                                 const int atom_offset,
                                 const FPTYPE* becsum,
                                 thrust::complex<FPTYPE>* packed)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atom_count * nij)
    {
        return;
    }
    const int ia = index / nij;
    const int ij = index % nij;
    packed[index] = thrust::complex<FPTYPE>(
        becsum[spin * nat * nh_tot + (atom_offset + ia) * nh_tot + ij], 0.0);
}

template <typename FPTYPE>
__global__ void uspp_accumulate_rhog(const int npw,
                                     const int nij,
                                     const thrust::complex<FPTYPE>* qgm,
                                     const thrust::complex<FPTYPE>* aux,
                                     thrust::complex<FPTYPE>* rhog)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
    {
        return;
    }
    thrust::complex<FPTYPE> value(0.0, 0.0);
    for (int ij = 0; ij < nij; ++ij)
    {
        value += qgm[ij * npw + ig] * aux[ij * npw + ig];
    }
    rhog[ig] += value;
}

template <typename FPTYPE>
void uspp_atom_phase_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*ctx*/,
    int atom_count,
    int npw,
    const FPTYPE* gcar,
    const FPTYPE* tau,
    std::complex<FPTYPE>* phase)
{
    const int count = atom_count * npw;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(uspp_atom_phase<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       atom_count,
                       npw,
                       gcar,
                       tau,
                       reinterpret_cast<thrust::complex<FPTYPE>*>(phase));
    hipCheckOnDebug();
}

template <typename FPTYPE>
void uspp_pack_becsum_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*ctx*/,
    int atom_count,
    int nij,
    int nat,
    int nh_tot,
    int spin,
    int atom_offset,
    const FPTYPE* becsum,
    std::complex<FPTYPE>* packed)
{
    const int count = atom_count * nij;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(uspp_pack_becsum<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       atom_count,
                       nij,
                       nat,
                       nh_tot,
                       spin,
                       atom_offset,
                       becsum,
                       reinterpret_cast<thrust::complex<FPTYPE>*>(packed));
    hipCheckOnDebug();
}

template <typename FPTYPE>
void uspp_accumulate_rhog_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*ctx*/,
    int npw,
    int nij,
    const std::complex<FPTYPE>* qgm,
    const std::complex<FPTYPE>* aux,
    std::complex<FPTYPE>* rhog)
{
    hipLaunchKernelGGL(HIP_KERNEL_NAME(uspp_accumulate_rhog<FPTYPE>),
                       dim3((npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       npw,
                       nij,
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(qgm),
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(aux),
                       reinterpret_cast<thrust::complex<FPTYPE>*>(rhog));
    hipCheckOnDebug();
}

template <typename FPTYPE>
__global__ void uspp_becsum(const int atom_count,
                            const int nbands,
                            const int nh,
                            const int nkb,
                            const int projector_offset,
                            const int atom_offset,
                            const int nat,
                            const int nh_tot,
                            const int spin,
                            const FPTYPE* weights,
                            const thrust::complex<FPTYPE>* becp,
                            FPTYPE* becsum)
{
    const int nij = nh * (nh + 1) / 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atom_count * nij)
    {
        return;
    }
    const int ia = index / nij;
    int pair = index % nij;
    int ih = 0;
    int row_size = nh;
    while (pair >= row_size)
    {
        pair -= row_size;
        ++ih;
        --row_size;
    }
    const int jh = ih + pair;
    const int atom_projector_offset = projector_offset + ia * nh;
    FPTYPE value = 0.0;
    for (int ib = 0; ib < nbands; ++ib)
    {
        value += weights[ib]
                 * (conj(becp[ib * nkb + atom_projector_offset + ih])
                    * becp[ib * nkb + atom_projector_offset + jh])
                       .real();
    }
    const int output = spin * nat * nh_tot + (atom_offset + ia) * nh_tot + index % nij;
    becsum[output] += ih == jh ? value : 2.0 * value;
}

template <typename FPTYPE>
void uspp_becsum_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* /*ctx*/,
    int atom_count,
    int nbands,
    int nh,
    int nkb,
    int projector_offset,
    int atom_offset,
    int nat,
    int nh_tot,
    int spin,
    const FPTYPE* weights,
    const std::complex<FPTYPE>* becp,
    FPTYPE* becsum)
{
    const int count = atom_count * nh * (nh + 1) / 2;
    if (count == 0)
    {
        return;
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(uspp_becsum<FPTYPE>),
                       dim3((count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK),
                       dim3(THREADS_PER_BLOCK),
                       0,
                       0,
                       atom_count,
                       nbands,
                       nh,
                       nkb,
                       projector_offset,
                       atom_offset,
                       nat,
                       nh_tot,
                       spin,
                       weights,
                       reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                       becsum);
    hipCheckOnDebug();
}

template<typename FPTYPE>
__global__ void elecstate_pw(
    const int spin,
    const int nrxx,
    const FPTYPE w1,
    FPTYPE* rho,
    const thrust::complex<FPTYPE>* wfcr)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= nrxx) {return;}
  rho[spin * nrxx + idx] += w1 * norm(wfcr[idx]);
}

template<typename FPTYPE>
__global__ void elecstate_pw(
    const bool DOMAG,
    const bool DOMAG_Z,
    const int nrxx,
    const FPTYPE w1,
    FPTYPE* rho,
    const thrust::complex<FPTYPE>* wfcr,
    const thrust::complex<FPTYPE>* wfcr_another_spin)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= nrxx) {return;}
  rho[0 * nrxx + idx] += w1 * (norm(wfcr[idx]) + norm(wfcr_another_spin[idx]));

  if (DOMAG) {
    rho[1 * nrxx + idx] += w1 * 2.0
                  * (wfcr[idx].real() * wfcr_another_spin[idx].real()
                  +  wfcr[idx].imag() * wfcr_another_spin[idx].imag());
    rho[2 * nrxx + idx] += w1 * 2.0
                  * (wfcr[idx].real() * wfcr_another_spin[idx].imag()
                  - wfcr_another_spin[idx].real() * wfcr[idx].imag());
    rho[3 * nrxx + idx] += w1 * (norm(wfcr[idx]) - norm(wfcr_another_spin[idx]));
  }
  else if(DOMAG_Z) {
    rho[1 * nrxx + idx] = 0;
    rho[2 * nrxx + idx] = 0;
    rho[3 * nrxx + idx] += w1 * (norm(wfcr[idx]) - norm(wfcr_another_spin[idx]));
  }
  else {
    rho[0 * nrxx + idx] = 0;
    rho[1 * nrxx + idx] = 0;
    rho[2 * nrxx + idx] = 0;
    rho[3 * nrxx + idx] = 0;
  }
}

template <typename FPTYPE>
void elecstate_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                  const int& spin,
                                                                  const int& nrxx,
                                                                  const FPTYPE& w1,
                                                                  FPTYPE** rho,
                                                                  const std::complex<FPTYPE>* wfcr)
{
  const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(elecstate_pw<FPTYPE>), dim3(block), dim3(THREADS_PER_BLOCK), 0, 0,
    spin, nrxx, w1, rho[0],
    reinterpret_cast<const thrust::complex<FPTYPE>*>(wfcr)
  );

   hipCheckOnDebug();
}

template <typename FPTYPE>
void elecstate_pw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                  const bool& DOMAG,
                                                                  const bool& DOMAG_Z,
                                                                  const int& nrxx,
                                                                  const FPTYPE& w1,
                                                                  FPTYPE** rho,
                                                                  const std::complex<FPTYPE>* wfcr,
                                                                  const std::complex<FPTYPE>* wfcr_another_spin)
{
  const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(elecstate_pw<FPTYPE>), dim3(block), dim3(THREADS_PER_BLOCK), 0, 0,
    DOMAG, DOMAG_Z, nrxx, w1, rho[0],
    reinterpret_cast<const thrust::complex<FPTYPE>*>(wfcr),
    reinterpret_cast<const thrust::complex<FPTYPE>*>(wfcr_another_spin)
  );

   hipCheckOnDebug();
}

template struct elecstate_pw_op<float, base_device::DEVICE_GPU>;
template struct elecstate_pw_op<double, base_device::DEVICE_GPU>;
template struct uspp_becsum_op<float, base_device::DEVICE_GPU>;
template struct uspp_becsum_op<double, base_device::DEVICE_GPU>;
template struct uspp_atom_phase_op<float, base_device::DEVICE_GPU>;
template struct uspp_atom_phase_op<double, base_device::DEVICE_GPU>;
template struct uspp_pack_becsum_op<float, base_device::DEVICE_GPU>;
template struct uspp_pack_becsum_op<double, base_device::DEVICE_GPU>;
template struct uspp_accumulate_rhog_op<float, base_device::DEVICE_GPU>;
template struct uspp_accumulate_rhog_op<double, base_device::DEVICE_GPU>;
}

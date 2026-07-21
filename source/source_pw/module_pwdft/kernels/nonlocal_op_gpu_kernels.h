#ifndef NONLOCAL_OP_GPU_KERNELS_H
#define NONLOCAL_OP_GPU_KERNELS_H

namespace hamilt
{
namespace nonlocal_gpu_detail
{

template <typename FPTYPE>
__global__ void uspp_qgm_build(const int ntype,
                               const int nh_tot,
                               const int npw,
                               const int lmaxq,
                               const int radial_pair_count,
                               const int nqxq,
                               const int max_terms,
                               const FPTYPE dq,
                               const FPTYPE tpiba,
                               const FPTYPE* gcar,
                               const int* pair_term_count,
                               const int* pair_radial_index,
                               const int* pair_l,
                               const int* pair_lm,
                               const thrust::complex<FPTYPE>* pair_coefficient,
                               const FPTYPE* qrad,
                               const FPTYPE* ylm,
                               thrust::complex<FPTYPE>* qgm)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = ntype * nh_tot * npw;
    if (index >= count)
    {
        return;
    }
    const int ig = index % npw;
    const int pair = index / npw;
    const int it = pair / nh_tot;
    const FPTYPE gx = gcar[3 * ig];
    const FPTYPE gy = gcar[3 * ig + 1];
    const FPTYPE gz = gcar[3 * ig + 2];
    const FPTYPE position = sqrt(gx * gx + gy * gy + gz * gz) * tpiba / dq;
    const int iq = static_cast<int>(position);
    thrust::complex<FPTYPE> value(0.0, 0.0);
    if (iq <= nqxq - 4)
    {
        const FPTYPE x0 = position - iq;
        for (int term = 0; term < pair_term_count[pair]; ++term)
        {
            const int term_index = pair * max_terms + term;
            const int qrad_offset
                = (((it * lmaxq + pair_l[term_index]) * radial_pair_count + pair_radial_index[pair]) * nqxq + iq);
            const FPTYPE work = qrad[qrad_offset] * (1 - x0) * (2 - x0) * (3 - x0) / 6
                              + qrad[qrad_offset + 1] * x0 * (2 - x0) * (3 - x0) / 2
                              - qrad[qrad_offset + 2] * (1 - x0) * x0 * (3 - x0) / 2
                              + qrad[qrad_offset + 3] * (1 - x0) * (2 - x0) * x0 / 6;
            value += pair_coefficient[term_index] * work * ylm[pair_lm[term_index] * npw + ig];
        }
    }
    qgm[index] = value;
}

template <typename FPTYPE>
__global__ void uspp_overlap(const int atom_count,
                             const int nbands,
                             const int nh,
                             const int nhm,
                             const int nkb,
                             const int projector_offset,
                             const bool projector_major,
                             const FPTYPE* qq,
                             thrust::complex<FPTYPE>* ps,
                             const thrust::complex<FPTYPE>* becp)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = atom_count * nbands * nh;
    if (index >= count)
    {
        return;
    }
    const int ih = index % nh;
    const int atom_band = index / nh;
    const int ib = atom_band % nbands;
    const int ia = atom_band / nbands;
    const int atom_offset = projector_offset + ia * nh;
    thrust::complex<FPTYPE> value(0.0, 0.0);
    for (int jh = 0; jh < nh; ++jh)
    {
        value += qq[jh * nhm + ih] * becp[ib * nkb + atom_offset + jh];
    }
    const int projector = atom_offset + ih;
    ps[projector_major ? projector * nbands + ib : ib * nkb + projector] = value;
}

template <typename FPTYPE>
__global__ void uspp_deeq(const int nspin,
                          const int atom_count,
                          const int nh,
                          const int nhm,
                          const int npw,
                          const int atom_offset,
                          const int nat,
                          const FPTYPE omega,
                          const bool gamma_only,
                          const int g0_index,
                          const thrust::complex<FPTYPE>* vaux,
                          const thrust::complex<FPTYPE>* qgm,
                          const thrust::complex<FPTYPE>* phase,
                          const FPTYPE* dvan,
                          FPTYPE* deeq)
{
    const int nij = nh * (nh + 1) / 2;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nspin * atom_count * nij)
    {
        return;
    }
    const int ij = index % nij;
    const int atom_spin = index / nij;
    const int ia = atom_spin % atom_count;
    const int is = atom_spin / atom_count;
    int pair = ij;
    int ih = 0;
    int row_size = nh;
    while (pair >= row_size)
    {
        pair -= row_size;
        ++ih;
        --row_size;
    }
    const int jh = ih + pair;
    FPTYPE integral = 0.0;
    for (int ig = 0; ig < npw; ++ig)
    {
        integral += (vaux[is * npw + ig] * conj(qgm[ij * npw + ig] * phase[ia * npw + ig])).real();
    }
    if (gamma_only)
    {
        integral *= 2.0;
        if (g0_index >= 0)
        {
            integral -= (vaux[is * npw + g0_index]
                         * conj(qgm[ij * npw + g0_index] * phase[ia * npw + g0_index]))
                            .real();
        }
    }
    const FPTYPE value = omega * integral + dvan[ih * nhm + jh];
    const int iat = atom_offset + ia;
    deeq[((is * nat + iat) * nhm + ih) * nhm + jh] = value;
    deeq[((is * nat + iat) * nhm + jh) * nhm + ih] = value;
}

template <typename FPTYPE>
__global__ void uspp_force(const int nspin,
                           const int atom_count,
                           const int nij,
                           const int npw,
                           const int atom_offset,
                           const int nat,
                           const int nh_tot,
                           const FPTYPE omega,
                           const FPTYPE tpiba,
                           const thrust::complex<FPTYPE>* vaux,
                           const thrust::complex<FPTYPE>* qgm,
                           const thrust::complex<FPTYPE>* phase,
                           const FPTYPE* gcar,
                           const FPTYPE* becsum,
                           FPTYPE* force)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atom_count * 3)
    {
        return;
    }
    const int ia = index / 3;
    const int ipol = index % 3;
    const int iat = atom_offset + ia;
    FPTYPE value = 0.0;
    for (int is = 0; is < nspin; ++is)
    {
        for (int ij = 0; ij < nij; ++ij)
        {
            FPTYPE integral = 0.0;
            for (int ig = 0; ig < npw; ++ig)
            {
                const thrust::complex<FPTYPE> product
                    = vaux[is * npw + ig] * conj(qgm[ij * npw + ig] * phase[ia * npw + ig]);
                integral += product.imag() * tpiba * gcar[3 * ig + ipol];
            }
            value += omega * integral * becsum[is * nat * nh_tot + iat * nh_tot + ij];
        }
    }
    force[iat * 3 + ipol] += value;
}

template <typename FPTYPE>
__global__ void uspp_stress(const int nspin,
                            const int atom_count,
                            const int nij,
                            const int npw,
                            const int atom_offset,
                            const int nat,
                            const int nh_tot,
                            const int ipol,
                            const FPTYPE tpiba,
                            const thrust::complex<FPTYPE>* vaux,
                            const thrust::complex<FPTYPE>* dqgm,
                            const thrust::complex<FPTYPE>* phase,
                            const FPTYPE* gcar,
                            const FPTYPE* becsum,
                            FPTYPE* stress)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= atom_count * 3)
    {
        return;
    }
    const int ia = index / 3;
    const int jpol = index % 3;
    const int iat = atom_offset + ia;
    FPTYPE value = 0.0;
    for (int is = 0; is < nspin; ++is)
    {
        for (int ij = 0; ij < nij; ++ij)
        {
            FPTYPE integral = 0.0;
            for (int ig = 0; ig < npw; ++ig)
            {
                const thrust::complex<FPTYPE> product
                    = vaux[is * npw + ig] * conj(dqgm[ij * npw + ig] * phase[ia * npw + ig]);
                integral += product.real() * tpiba * gcar[3 * ig + jpol];
            }
            value += integral * becsum[is * nat * nh_tot + iat * nh_tot + ij];
        }
    }
    atomicAdd(&stress[jpol * 3 + ipol], value);
}

template <typename FPTYPE>
__global__ void nonlocal_pw(
    const int l1,
    const int l2,
    const int l3,
    const int sum,
    const int iat,
    const int spin,
    const int nkb,
    const int deeq_x,
    const int deeq_y,
    const int deeq_z,
    const FPTYPE* deeq,
    thrust::complex<FPTYPE>* ps,
    const thrust::complex<FPTYPE>* becp)
{
  const int ii = blockIdx.x / l2;
  const int jj = blockIdx.x % l2;
  for (int kk = threadIdx.x; kk < l3; kk += blockDim.x) {
    thrust::complex<FPTYPE> res(0.0, 0.0);
    for (int xx = 0; xx < l3; xx++) {
      res
        += deeq[((spin * deeq_x + iat + ii) * deeq_y + xx) * deeq_z + kk]
        *  becp[jj * nkb + sum + ii * l3 + xx];
    }
    ps[(sum + ii * l3 + kk) * l2 + jj] += res;
  }
}

template <typename FPTYPE>
__global__ void nonlocal_pw(
    const int l1,
    const int l2,
    const int l3,
    const int sum,
    const int iat,
    const int nkb,
    const int deeq_x,
    const int deeq_y,
    const int deeq_z,
    const thrust::complex<FPTYPE>* deeq_nc,
    thrust::complex<FPTYPE>* ps,
    const thrust::complex<FPTYPE>* becp)
{
  const int ii = blockIdx.x * 2 / l2;
  const int jj = blockIdx.x * 2 % l2;
  for (int kk = threadIdx.x; kk < l3; kk += blockDim.x) {
    thrust::complex<FPTYPE> res1(0.0, 0.0);
    thrust::complex<FPTYPE> res2(0.0, 0.0);
    int psind = (sum + ii * l3 + kk) * l2 + jj;
    for (int xx = 0; xx < l3; xx++) {
      int becpind = jj * nkb + sum + ii * l3 + xx;
      thrust::complex<FPTYPE> becp1 = becp[becpind];
      thrust::complex<FPTYPE> becp2 = becp[becpind + nkb];
      res1 += deeq_nc[((0 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp1
                   + deeq_nc[((1 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp2;
      res2 += deeq_nc[((2 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp1
                       + deeq_nc[((3 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp2;
    }
    ps[psind] += res1;
    ps[psind + 1] += res2;
  }
}

} // namespace nonlocal_gpu_detail
} // namespace hamilt

#endif

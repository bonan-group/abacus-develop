#include "source_pw/module_pwdft/kernels/nonlocal_op.h"

#include <cmath>

namespace hamilt {

template <typename FPTYPE>
struct uspp_qgm_build_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
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
        const int pair_count = ntype * nh_tot;
        for (int pair = 0; pair < pair_count; ++pair)
        {
            const int it = pair / nh_tot;
            for (int ig = 0; ig < npw; ++ig)
            {
                const FPTYPE gx = gcar[3 * ig];
                const FPTYPE gy = gcar[3 * ig + 1];
                const FPTYPE gz = gcar[3 * ig + 2];
                const FPTYPE position = std::sqrt(gx * gx + gy * gy + gz * gz) * tpiba / dq;
                const int iq = static_cast<int>(position);
                std::complex<FPTYPE> value(0.0, 0.0);
                if (iq <= nqxq - 4)
                {
                    const FPTYPE x0 = position - iq;
                    for (int term = 0; term < pair_term_count[pair]; ++term)
                    {
                        const int term_index = pair * max_terms + term;
                        const int qrad_offset
                            = (((it * lmaxq + pair_l[term_index]) * radial_pair_count
                                + pair_radial_index[pair])
                               * nqxq
                               + iq);
                        const FPTYPE work
                            = qrad[qrad_offset] * (1 - x0) * (2 - x0) * (3 - x0) / 6
                              + qrad[qrad_offset + 1] * x0 * (2 - x0) * (3 - x0) / 2
                              - qrad[qrad_offset + 2] * (1 - x0) * x0 * (3 - x0) / 2
                              + qrad[qrad_offset + 3] * (1 - x0) * (2 - x0) * x0 / 6;
                        value += pair_coefficient[term_index] * work * ylm[pair_lm[term_index] * npw + ig];
                    }
                }
                qgm[pair * npw + ig] = value;
            }
        }
    }
};

template <typename FPTYPE>
struct uspp_overlap_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
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
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int ia = 0; ia < atom_count; ++ia)
    {
        for (int ib = 0; ib < nbands; ++ib)
        {
            for (int ih = 0; ih < nh; ++ih)
            {
                const int atom_offset = projector_offset + ia * nh;
                std::complex<FPTYPE> value(0.0, 0.0);
                for (int jh = 0; jh < nh; ++jh)
                {
                    value += qq[jh * nhm + ih] * becp[ib * nkb + atom_offset + jh];
                }
                const int projector = atom_offset + ih;
                ps[projector_major ? projector * nbands + ib : ib * nkb + projector] = value;
            }
        }
    }
    }
};

template <typename FPTYPE>
struct uspp_deeq_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
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
        const int nij = nh * (nh + 1) / 2;
        for (int is = 0; is < nspin; ++is)
        {
            for (int ia = 0; ia < atom_count; ++ia)
            {
                for (int ij = 0; ij < nij; ++ij)
                {
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
                        integral += std::real(vaux[is * npw + ig]
                                              * std::conj(qgm[ij * npw + ig]
                                                          * phase[ia * npw + ig]));
                    }
                    if (gamma_only)
                    {
                        integral *= 2.0;
                        if (g0_index >= 0)
                        {
                            integral -= std::real(vaux[is * npw + g0_index]
                                                  * std::conj(qgm[ij * npw + g0_index]
                                                              * phase[ia * npw + g0_index]));
                        }
                    }
                    const FPTYPE value = omega * integral + dvan[ih * nhm + jh];
                    const int iat = atom_offset + ia;
                    deeq[((is * nat + iat) * nhm + ih) * nhm + jh] = value;
                    deeq[((is * nat + iat) * nhm + jh) * nhm + ih] = value;
                }
            }
        }
    }
};

template <typename FPTYPE>
struct uspp_stress_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
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
        for (int ia = 0; ia < atom_count; ++ia)
        {
            const int iat = atom_offset + ia;
            for (int jpol = 0; jpol < 3; ++jpol)
            {
                FPTYPE value = 0.0;
                for (int is = 0; is < nspin; ++is)
                {
                    for (int ij = 0; ij < nij; ++ij)
                    {
                        FPTYPE integral = 0.0;
                        for (int ig = 0; ig < npw; ++ig)
                        {
                            const std::complex<FPTYPE> product
                                = vaux[is * npw + ig]
                                  * std::conj(dqgm[ij * npw + ig] * phase[ia * npw + ig]);
                            integral += std::real(product) * tpiba * gcar[3 * ig + jpol];
                        }
                        value += integral * becsum[is * nat * nh_tot + iat * nh_tot + ij];
                    }
                }
                stress[jpol * 3 + ipol] += value;
            }
        }
    }
};

template <typename FPTYPE>
struct nonlocal_pw_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*dev*/,
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
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int ii = 0; ii < l1; ii++) {
      // each atom has nproj, means this is with structure factor;
      // each projector (each atom) must multiply coefficient
      // with all the other projectors.
      for (int jj = 0; jj < l2; ++jj) 
        for (int kk = 0; kk < l3; kk++) 
          for (int xx = 0; xx < l3; xx++) 
            ps[(sum + ii * l3 + kk) * l2 + jj]
              += deeq[((spin * deeq_x + iat + ii) * deeq_y + xx) * deeq_z + kk] 
              *  becp[jj * nkb + sum + ii * l3 + xx];
    }
    sum += l1 * l3;
    iat += l1;
  }

  void operator()(const base_device::DEVICE_CPU* dev,
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
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
    for (int ii = 0; ii < l1; ii++)
      // each atom has nproj, means this is with structure factor;
      // each projector (each atom) must multiply coefficient
      // with all the other projectors.
      for (int jj = 0; jj < l2; jj+=2)
        for (int kk = 0; kk < l3; kk++)
          for (int xx = 0; xx < l3; xx++)
          {
              int psind = (sum + ii * l3 + kk) * l2 + jj;
              int becpind = jj * nkb + sum + ii * l3 + xx;
              auto &becp1 = becp[becpind];
              auto &becp2 = becp[becpind + nkb];
              ps[psind] += deeq_nc[((0 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp1
                           + deeq_nc[((1 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp2;
              ps[psind + 1] += deeq_nc[((2 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp1
                               + deeq_nc[((3 * deeq_x + iat + ii) * deeq_y + kk) * deeq_z + xx] * becp2;
          } // end jj
    iat += l1;
    sum += l1 * l3;
  }
};

template struct nonlocal_pw_op<float, base_device::DEVICE_CPU>;
template struct nonlocal_pw_op<double, base_device::DEVICE_CPU>;
template struct uspp_qgm_build_op<float, base_device::DEVICE_CPU>;
template struct uspp_qgm_build_op<double, base_device::DEVICE_CPU>;
template struct uspp_overlap_op<float, base_device::DEVICE_CPU>;
template struct uspp_overlap_op<double, base_device::DEVICE_CPU>;
template struct uspp_deeq_op<float, base_device::DEVICE_CPU>;
template struct uspp_deeq_op<double, base_device::DEVICE_CPU>;
template struct uspp_stress_op<float, base_device::DEVICE_CPU>;
template struct uspp_stress_op<double, base_device::DEVICE_CPU>;

}  // namespace hamilt

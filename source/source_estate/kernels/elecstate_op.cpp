#include "source_estate/kernels/elecstate_op.h"

namespace elecstate{

template <typename FPTYPE>
struct uspp_atom_phase_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*ctx*/,
                    int atom_count,
                    int npw,
                    const FPTYPE* gcar,
                    const FPTYPE* tau,
                    std::complex<FPTYPE>* phase)
    {
        const FPTYPE two_pi = static_cast<FPTYPE>(6.283185307179586476925286766559);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int ia = 0; ia < atom_count; ++ia)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                const FPTYPE arg = two_pi * (gcar[3 * ig] * tau[3 * ia]
                                             + gcar[3 * ig + 1] * tau[3 * ia + 1]
                                             + gcar[3 * ig + 2] * tau[3 * ia + 2]);
                phase[ia * npw + ig] = std::complex<FPTYPE>(std::cos(arg), -std::sin(arg));
            }
        }
    }
};

template <typename FPTYPE>
struct uspp_pack_becsum_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*ctx*/,
                    int atom_count,
                    int nij,
                    int nat,
                    int nh_tot,
                    int spin,
                    int atom_offset,
                    const FPTYPE* becsum,
                    std::complex<FPTYPE>* packed)
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int ia = 0; ia < atom_count; ++ia)
        {
            for (int ij = 0; ij < nij; ++ij)
            {
                packed[ia * nij + ij]
                    = std::complex<FPTYPE>(becsum[spin * nat * nh_tot
                                                   + (atom_offset + ia) * nh_tot + ij],
                                            0.0);
            }
        }
    }
};

template <typename FPTYPE>
struct uspp_accumulate_rhog_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*ctx*/,
                    int npw,
                    int nij,
                    const std::complex<FPTYPE>* qgm,
                    const std::complex<FPTYPE>* aux,
                    std::complex<FPTYPE>* rhog)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ig = 0; ig < npw; ++ig)
        {
            std::complex<FPTYPE> value(0.0, 0.0);
            for (int ij = 0; ij < nij; ++ij)
            {
                value += qgm[ij * npw + ig] * aux[ij * npw + ig];
            }
            rhog[ig] += value;
        }
    }
};

template <typename FPTYPE>
struct uspp_becsum_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*ctx*/,
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ia = 0; ia < atom_count; ++ia)
        {
            const int atom_projector_offset = projector_offset + ia * nh;
            int ijh = 0;
            for (int ih = 0; ih < nh; ++ih)
            {
                for (int jh = ih; jh < nh; ++jh)
                {
                    FPTYPE value = 0.0;
                    for (int ib = 0; ib < nbands; ++ib)
                    {
                        value += weights[ib]
                                 * std::real(std::conj(becp[ib * nkb + atom_projector_offset + ih])
                                             * becp[ib * nkb + atom_projector_offset + jh]);
                    }
                    const int index = spin * nat * nh_tot + (atom_offset + ia) * nh_tot + ijh;
                    becsum[index] += ih == jh ? value : 2.0 * value;
                    ++ijh;
                }
            }
        }
    }
};

template <typename FPTYPE>
struct elecstate_pw_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* /*ctx*/,
                    const int& spin,
                    const int& nrxx,
                    const FPTYPE& w1,
                    FPTYPE** rho,
                    const std::complex<FPTYPE>* wfcr)
    {
      // for (int ir = 0; ir < nrxx; ir++)
      // {
      //   rho[spin][ir] += weight * norm(wfcr[ir]);
      // }
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int ir = 0; ir < nrxx; ir++)
      {
        rho[spin][ir] += w1 * norm(wfcr[ir]);
      }
    }

    void operator()(const base_device::DEVICE_CPU* ctx,
                    const bool& DOMAG,
                    const bool& DOMAG_Z,
                    const int& nrxx,
                    const FPTYPE& w1,
                    FPTYPE** rho,
                    const std::complex<FPTYPE>* wfcr,
                    const std::complex<FPTYPE>* wfcr_another_spin)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for (int ir = 0; ir < nrxx; ir++) {
          rho[0][ir] += w1 * (norm(wfcr[ir]) + norm(wfcr_another_spin[ir]));
      }
      // In this case, calculate the three components of the magnetization
      if (DOMAG)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int ir = 0; ir < nrxx; ir++) {
              rho[1][ir] += w1 * 2.0
                                          * (wfcr[ir].real() * wfcr_another_spin[ir].real()
                                             + wfcr[ir].imag() * wfcr_another_spin[ir].imag());
              rho[2][ir] += w1 * 2.0
                                          * (wfcr[ir].real() * wfcr_another_spin[ir].imag()
                                             - wfcr_another_spin[ir].real() * wfcr[ir].imag());
              rho[3][ir] += w1 * (norm(wfcr[ir]) - norm(wfcr_another_spin[ir]));
          }
      }
      else if (DOMAG_Z)
      {
#ifdef _OPENMP
#pragma omp parallel for
#endif
          for (int ir = 0; ir < nrxx; ir++)
          {
              rho[1][ir] = 0;
              rho[2][ir] = 0;
              rho[3][ir] += w1 * (norm(wfcr[ir]) - norm(wfcr_another_spin[ir]));
          }
      }
      else {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
          for (int is = 1; is < 4; is++)
          {
              for (int ir = 0; ir < nrxx; ir++)
                  rho[is][ir] = 0;
          }
      }
    }
};

template struct elecstate_pw_op<float, base_device::DEVICE_CPU>;
template struct elecstate_pw_op<double, base_device::DEVICE_CPU>;
template struct uspp_becsum_op<float, base_device::DEVICE_CPU>;
template struct uspp_becsum_op<double, base_device::DEVICE_CPU>;
template struct uspp_atom_phase_op<float, base_device::DEVICE_CPU>;
template struct uspp_atom_phase_op<double, base_device::DEVICE_CPU>;
template struct uspp_pack_becsum_op<float, base_device::DEVICE_CPU>;
template struct uspp_pack_becsum_op<double, base_device::DEVICE_CPU>;
template struct uspp_accumulate_rhog_op<float, base_device::DEVICE_CPU>;
template struct uspp_accumulate_rhog_op<double, base_device::DEVICE_CPU>;
}  // namespace elecstate

#include "source_pw/module_pwdft/kernels/vnl_op.h"

#include "vnl_tools.hpp"
#include "source_base/timer.h"

namespace hamilt
{

template <typename FPTYPE>
struct cal_vnl_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& ntype,
                    const int& npw,
                    const int& npwx,
                    const int& nhm,
                    const int& tab_2,
                    const int& tab_3,
                    const int* atom_na,
                    const int* atom_nb,
                    const int* atom_nh,
                    const FPTYPE& DQ,
                    const FPTYPE& tpiba,
                    const std::complex<FPTYPE>& NEG_IMAG_UNIT,
                    const FPTYPE* gk,
                    const FPTYPE* ylm,
                    const FPTYPE* indv,
                    const FPTYPE* nhtol,
                    const FPTYPE* nhtolm,
                    const FPTYPE* tab,
                    FPTYPE* vkb1,
                    const std::complex<FPTYPE>* sk,
                    std::complex<FPTYPE>* vkb_in)
    {
        ModuleBase::timer::tick("Operator", "cal_vnl_op");
        const int imag_pow_period = 4;
        // result table of pow(0-1i, int)
        static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};
#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            int jkb = 0, iat = 0;
            FPTYPE vq = 0.0;
            for (int it = 0; it < ntype; it++)
            {
                // calculate beta in G-space using an interpolation table
                const int nh = atom_nh[it];
                const int nbeta = atom_nb[it];

                for (int nb = 0; nb < nbeta; nb++)
                {
#ifdef _OPENMP
#pragma omp for
#endif
                    for (int ig = 0; ig < npw; ig++)
                    {
                        const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0] + gk[ig * 3 + 1] * gk[ig * 3 + 1]
                                                  + gk[ig * 3 + 2] * gk[ig * 3 + 2])
                                             * tpiba;

                        vq = _polynomial_interpolation(tab, it, nb, tab_2, tab_3, DQ, gnorm);

                        // add spherical harmonic part
                        for (int ih = 0; ih < nh; ih++)
                        {
                            if (nb == indv[it * nhm + ih])
                            {
                                const int lm = static_cast<int>(nhtolm[it * nhm + ih]);
                                vkb1[ih * npw + ig] = ylm[lm * npw + ig] * vq;
                            }
                        } // end ih
                    }
                } // end nbeta

                // vkb1 contains all betas including angular part for type nt
                // now add the structure factor and factor (-i)^l
                for (int ia = 0; ia < atom_na[it]; ia++)
                {
                    for (int ih = 0; ih < nh; ih++)
                    {
                        // std::complex<FPTYPE> pref = pow(NEG_IMAG_UNIT, nhtol[it * nhm + ih]);    //?
                        std::complex<FPTYPE> pref = pref_tab[int(nhtol[it * nhm + ih]) % imag_pow_period];
                        std::complex<FPTYPE>* pvkb = vkb_in + jkb * npwx;
#ifdef _OPENMP
#pragma omp for
#endif
                        for (int ig = 0; ig < npw; ig++)
                        {
                            pvkb[ig] = vkb1[ih * npw + ig] * sk[iat * npw + ig] * pref;
                        }
                        ++jkb;
                    } // end ih
                    iat++;
                } // end ia
            }     // enddo
#ifdef _OPENMP
        }
#endif
        ModuleBase::timer::tick("Operator", "cal_vnl_op");
    }
};

template struct cal_vnl_op<float, base_device::DEVICE_CPU>;
template struct cal_vnl_op<double, base_device::DEVICE_CPU>;

/// @brief CPU implementation of cal_vnl_atoms_op for computing vkb for a range of atoms
template <typename FPTYPE>
struct cal_vnl_atoms_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& ntype,
                    const int& npw,
                    const int& npwx,
                    const int& nhm,
                    const int& tab_2,
                    const int& tab_3,
                    const int* atom_na,
                    const int* atom_nb,
                    const int* atom_nh,
                    const int& atom_start,
                    const int& atom_end,
                    const FPTYPE& DQ,
                    const FPTYPE& tpiba,
                    const std::complex<FPTYPE>& NEG_IMAG_UNIT,
                    const FPTYPE* gk,
                    const FPTYPE* ylm,
                    const FPTYPE* indv,
                    const FPTYPE* nhtol,
                    const FPTYPE* nhtolm,
                    const FPTYPE* tab,
                    FPTYPE* vkb1,
                    const std::complex<FPTYPE>* sk,
                    const int* iat2it,
                    std::complex<FPTYPE>* vkb_out)
    {
        const int imag_pow_period = 4;
        // result table of pow(0-1i, int)
        static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

        // Track which atom types we need to process
        // First, determine which types have atoms in range [atom_start, atom_end)
        int last_type = -1;  // Track last processed type to avoid redundant vkb1 computation
        int jkb_out = 0;     // Output projector index (starts at 0)

#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            for (int iat = atom_start; iat < atom_end; iat++)
            {
                const int it = iat2it[iat];
                const int nh = atom_nh[it];
                const int nbeta = atom_nb[it];

                // Compute vkb1 for this type if not already computed
#ifdef _OPENMP
#pragma omp single
#endif
                {
                    if (it != last_type)
                    {
                        for (int nb = 0; nb < nbeta; nb++)
                        {
                            for (int ig = 0; ig < npw; ig++)
                            {
                                const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0]
                                                        + gk[ig * 3 + 1] * gk[ig * 3 + 1]
                                                        + gk[ig * 3 + 2] * gk[ig * 3 + 2])
                                                     * tpiba;

                                FPTYPE vq = _polynomial_interpolation(tab, it, nb, tab_2, tab_3, DQ, gnorm);

                                // add spherical harmonic part
                                for (int ih = 0; ih < nh; ih++)
                                {
                                    if (nb == indv[it * nhm + ih])
                                    {
                                        const int lm = static_cast<int>(nhtolm[it * nhm + ih]);
                                        vkb1[ih * npw + ig] = ylm[lm * npw + ig] * vq;
                                    }
                                }
                            }
                        }
                        last_type = it;
                    }
                }

                // Add structure factor and (-i)^l factor for this atom
                for (int ih = 0; ih < nh; ih++)
                {
                    std::complex<FPTYPE> pref = pref_tab[int(nhtol[it * nhm + ih]) % imag_pow_period];
                    std::complex<FPTYPE>* pvkb = vkb_out + jkb_out * npwx;
#ifdef _OPENMP
#pragma omp for
#endif
                    for (int ig = 0; ig < npw; ig++)
                    {
                        pvkb[ig] = vkb1[ih * npw + ig] * sk[iat * npw + ig] * pref;
                    }
#ifdef _OPENMP
#pragma omp single
#endif
                    {
                        ++jkb_out;
                    }
                }
            }
#ifdef _OPENMP
        }
#endif
    }
};

template struct cal_vnl_atoms_op<float, base_device::DEVICE_CPU>;
template struct cal_vnl_atoms_op<double, base_device::DEVICE_CPU>;

/// @brief CPU implementation of cal_vnl_atoms_cached_op for computing vkb using cached gk/ylm/sk_all
/// The sk_all array contains structure factors for ALL atoms (indexed by global iat)
template <typename FPTYPE>
struct cal_vnl_atoms_cached_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& ntype,
                    const int& npw,
                    const int& npwx,
                    const int& nhm,
                    const int& tab_2,
                    const int& tab_3,
                    const int* atom_na,
                    const int* atom_nb,
                    const int* atom_nh,
                    const int& atom_start,
                    const int& atom_end,
                    const FPTYPE& DQ,
                    const FPTYPE& tpiba,
                    const std::complex<FPTYPE>& NEG_IMAG_UNIT,
                    const FPTYPE* gk,         // Pre-computed (cached)
                    const FPTYPE* ylm,        // Pre-computed (cached)
                    const FPTYPE* indv,
                    const FPTYPE* nhtol,
                    const FPTYPE* nhtolm,
                    const FPTYPE* tab,
                    FPTYPE* vkb1,
                    const std::complex<FPTYPE>* sk_all,  // All-atom sk indexed by global iat
                    const int* iat2it,
                    std::complex<FPTYPE>* vkb_out)
    {
        const int imag_pow_period = 4;
        // result table of pow(0-1i, int)
        static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

        // Track which atom types we need to process
        int last_type = -1;  // Track last processed type to avoid redundant vkb1 computation
        int jkb_out = 0;     // Output projector index (starts at 0)

#ifdef _OPENMP
#pragma omp parallel
        {
#endif
            for (int iat = atom_start; iat < atom_end; iat++)
            {
                const int it = iat2it[iat];
                const int nh = atom_nh[it];
                const int nbeta = atom_nb[it];

                // Compute vkb1 for this type if not already computed
#ifdef _OPENMP
#pragma omp single
#endif
                {
                    if (it != last_type)
                    {
                        for (int nb = 0; nb < nbeta; nb++)
                        {
                            for (int ig = 0; ig < npw; ig++)
                            {
                                const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0]
                                                        + gk[ig * 3 + 1] * gk[ig * 3 + 1]
                                                        + gk[ig * 3 + 2] * gk[ig * 3 + 2])
                                                     * tpiba;

                                FPTYPE vq = _polynomial_interpolation(tab, it, nb, tab_2, tab_3, DQ, gnorm);

                                // add spherical harmonic part (ylm is pre-computed)
                                for (int ih = 0; ih < nh; ih++)
                                {
                                    if (nb == indv[it * nhm + ih])
                                    {
                                        const int lm = static_cast<int>(nhtolm[it * nhm + ih]);
                                        vkb1[ih * npw + ig] = ylm[lm * npw + ig] * vq;
                                    }
                                }
                            }
                        }
                        last_type = it;
                    }
                }

                // Add structure factor and (-i)^l factor for this atom
                // Note: sk_all is indexed by global atom index (iat), not chunk-local
                for (int ih = 0; ih < nh; ih++)
                {
                    std::complex<FPTYPE> pref = pref_tab[int(nhtol[it * nhm + ih]) % imag_pow_period];
                    std::complex<FPTYPE>* pvkb = vkb_out + jkb_out * npwx;
#ifdef _OPENMP
#pragma omp for
#endif
                    for (int ig = 0; ig < npw; ig++)
                    {
                        pvkb[ig] = vkb1[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
                    }
#ifdef _OPENMP
#pragma omp single
#endif
                    {
                        ++jkb_out;
                    }
                }
            }
#ifdef _OPENMP
        }
#endif
    }
};

template struct cal_vnl_atoms_cached_op<float, base_device::DEVICE_CPU>;
template struct cal_vnl_atoms_cached_op<double, base_device::DEVICE_CPU>;

} // namespace hamilt

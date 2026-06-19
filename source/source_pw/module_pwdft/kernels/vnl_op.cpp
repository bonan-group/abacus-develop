#include "source_pw/module_pwdft/kernels/vnl_op.h"

#include "vnl_tools.hpp"

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
    }
};

template struct cal_vnl_op<float, base_device::DEVICE_CPU>;
template struct cal_vnl_op<double, base_device::DEVICE_CPU>;

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
                    const FPTYPE* gk,
                    const FPTYPE* ylm,
                    const FPTYPE* indv,
                    const FPTYPE* nhtol,
                    const FPTYPE* nhtolm,
                    const FPTYPE* tab,
                    FPTYPE* vkb1,
                    const std::complex<FPTYPE>* sk_all,
                    const int* iat2it,
                    std::complex<FPTYPE>* vkb_out)
    {
        const int imag_pow_period = 4;
        static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

        int last_type = -1;
        int jkb_out = 0;
        for (int iat = atom_start; iat < atom_end; ++iat)
        {
            const int it = iat2it[iat];
            const int nh = atom_nh[it];
            const int nbeta = atom_nb[it];

            if (it != last_type)
            {
                for (int nb = 0; nb < nbeta; ++nb)
                {
#ifdef _OPENMP
#pragma omp parallel for
#endif
                    for (int ig = 0; ig < npw; ++ig)
                    {
                        const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0]
                                                  + gk[ig * 3 + 1] * gk[ig * 3 + 1]
                                                  + gk[ig * 3 + 2] * gk[ig * 3 + 2])
                                             * tpiba;
                        const FPTYPE vq = _polynomial_interpolation(tab, it, nb, tab_2, tab_3, DQ, gnorm);

                        for (int ih = 0; ih < nh; ++ih)
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

            for (int ih = 0; ih < nh; ++ih)
            {
                const std::complex<FPTYPE> pref = pref_tab[int(nhtol[it * nhm + ih]) % imag_pow_period];
                std::complex<FPTYPE>* pvkb = vkb_out + jkb_out * npwx;
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (int ig = 0; ig < npw; ++ig)
                {
                    pvkb[ig] = vkb1[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
                }
                ++jkb_out;
            }
        }
    }
};

template struct cal_vnl_atoms_cached_op<float, base_device::DEVICE_CPU>;
template struct cal_vnl_atoms_cached_op<double, base_device::DEVICE_CPU>;

template <typename FPTYPE>
struct cal_vkb1_cache_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& ntype,
                    const int& npw,
                    const int& nhm,
                    const int& tab_2,
                    const int& tab_3,
                    const int* atom_nb,
                    const int* atom_nh,
                    const FPTYPE& DQ,
                    const FPTYPE& tpiba,
                    const FPTYPE* gk,
                    const FPTYPE* ylm,
                    const FPTYPE* indv,
                    const FPTYPE* nhtolm,
                    const FPTYPE* tab,
                    FPTYPE* vkb1_cache)
    {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ig = 0; ig < npw; ++ig)
        {
            const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0]
                                      + gk[ig * 3 + 1] * gk[ig * 3 + 1]
                                      + gk[ig * 3 + 2] * gk[ig * 3 + 2])
                                 * tpiba;
            for (int it = 0; it < ntype; ++it)
            {
                const int nh = atom_nh[it];
                const int nbeta = atom_nb[it];
                FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
                for (int nb = 0; nb < nbeta; ++nb)
                {
                    const FPTYPE vq = _polynomial_interpolation(tab, it, nb, tab_2, tab_3, DQ, gnorm);
                    for (int ih = 0; ih < nh; ++ih)
                    {
                        if (nb == indv[it * nhm + ih])
                        {
                            const int lm = static_cast<int>(nhtolm[it * nhm + ih]);
                            vkb1_type[ih * npw + ig] = ylm[lm * npw + ig] * vq;
                        }
                    }
                }
            }
        }
    }
};

template <typename FPTYPE>
struct cal_vnl_from_vkb1_cache_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& npw,
                    const int& npwx,
                    const int& nhm,
                    const int* atom_nh,
                    const int& atom_start,
                    const int& atom_end,
                    const FPTYPE* nhtol,
                    const FPTYPE* vkb1_cache,
                    const std::complex<FPTYPE>* sk_all,
                    const int* iat2it,
                    std::complex<FPTYPE>* vkb_out)
    {
        const int imag_pow_period = 4;
        static const std::complex<FPTYPE> pref_tab[imag_pow_period] = {{1, 0}, {0, -1}, {-1, 0}, {0, 1}};

        int jkb_out = 0;
        for (int iat = atom_start; iat < atom_end; ++iat)
        {
            const int it = iat2it[iat];
            const int nh = atom_nh[it];
            const FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
            for (int ih = 0; ih < nh; ++ih)
            {
                const std::complex<FPTYPE> pref = pref_tab[int(nhtol[it * nhm + ih]) % imag_pow_period];
                std::complex<FPTYPE>* pvkb = vkb_out + jkb_out * npwx;
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for (int ig = 0; ig < npw; ++ig)
                {
                    pvkb[ig] = vkb1_type[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
                }
                ++jkb_out;
            }
        }
    }
};

template struct cal_vkb1_cache_op<float, base_device::DEVICE_CPU>;
template struct cal_vkb1_cache_op<double, base_device::DEVICE_CPU>;
template struct cal_vnl_from_vkb1_cache_op<float, base_device::DEVICE_CPU>;
template struct cal_vnl_from_vkb1_cache_op<double, base_device::DEVICE_CPU>;

template <typename FPTYPE>
struct cal_becp_from_vkb1_cache_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& npw,
                    const int& npwx,
                    const int& nbands,
                    const int& nkb,
                    const int& nhm,
                    const int* jkb_to_iat,
                    const int* jkb_to_it,
                    const int* jkb_to_ih,
                    const FPTYPE* jkb_pref_sign,
                    const FPTYPE* vkb1_cache,
                    const std::complex<FPTYPE>* sk_all,
                    const std::complex<FPTYPE>* psi,
                    std::complex<FPTYPE>* becp)
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int ib = 0; ib < nbands; ++ib)
        {
            for (int jkb = 0; jkb < nkb; ++jkb)
            {
                const int iat = jkb_to_iat[jkb];
                const int it = jkb_to_it[jkb];
                const int ih = jkb_to_ih[jkb];
                const FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
                const std::complex<FPTYPE> pref(jkb_pref_sign[2 * jkb], jkb_pref_sign[2 * jkb + 1]);
                std::complex<FPTYPE> sum(0, 0);
                for (int ig = 0; ig < npw; ++ig)
                {
                    const std::complex<FPTYPE> vkb = vkb1_type[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
                    sum += std::conj(vkb) * psi[ib * npwx + ig];
                }
                becp[ib * nkb + jkb] = sum;
            }
        }
    }
};

template <typename FPTYPE>
struct cal_hpsi_from_vkb1_cache_op<FPTYPE, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    const int& npw,
                    const int& npwx,
                    const int& nbands,
                    const int& nkb,
                    const int& nhm,
                    const int* jkb_to_iat,
                    const int* jkb_to_it,
                    const int* jkb_to_ih,
                    const FPTYPE* jkb_pref_sign,
                    const FPTYPE* vkb1_cache,
                    const std::complex<FPTYPE>* sk_all,
                    const std::complex<FPTYPE>* ps,
                    std::complex<FPTYPE>* hpsi)
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
        for (int ib = 0; ib < nbands; ++ib)
        {
            for (int ig = 0; ig < npw; ++ig)
            {
                std::complex<FPTYPE> sum(0, 0);
                for (int jkb = 0; jkb < nkb; ++jkb)
                {
                    const int iat = jkb_to_iat[jkb];
                    const int it = jkb_to_it[jkb];
                    const int ih = jkb_to_ih[jkb];
                    const FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
                    const std::complex<FPTYPE> pref(jkb_pref_sign[2 * jkb], jkb_pref_sign[2 * jkb + 1]);
                    const std::complex<FPTYPE> vkb = vkb1_type[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
                    sum += vkb * ps[jkb * nbands + ib];
                }
                hpsi[ib * npwx + ig] += sum;
            }
        }
    }
};

template struct cal_becp_from_vkb1_cache_op<float, base_device::DEVICE_CPU>;
template struct cal_becp_from_vkb1_cache_op<double, base_device::DEVICE_CPU>;
template struct cal_hpsi_from_vkb1_cache_op<float, base_device::DEVICE_CPU>;
template struct cal_hpsi_from_vkb1_cache_op<double, base_device::DEVICE_CPU>;

} // namespace hamilt

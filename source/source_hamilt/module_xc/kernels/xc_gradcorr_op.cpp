#include "source_hamilt/module_xc/kernels/xc_gradcorr_op.h"

#include "source_hamilt/module_xc/kernels/xc_builtin_formula.h"

#include <cmath>
#include <complex>
#include <vector>

namespace hamilt
{

namespace
{

template <typename FPTYPE>
void xc_gcx_pbe_spin(const FPTYPE rhoup,
                     const FPTYPE rhodw,
                     const FPTYPE grhoup2,
                     const FPTYPE grhodw2,
                     const int iflag,
                     FPTYPE& sx,
                     FPTYPE& v1xup,
                     FPTYPE& v1xdw,
                     FPTYPE& v2xup,
                     FPTYPE& v2xdw)
{
    const FPTYPE small = 1.0e-10;
    sx = 0.0;
    v1xup = 0.0;
    v1xdw = 0.0;
    v2xup = 0.0;
    v2xdw = 0.0;
    if (rhoup + rhodw <= small)
    {
        return;
    }

    FPTYPE sxup = 0.0;
    FPTYPE sxdw = 0.0;
    if (rhoup > small && std::sqrt(std::abs(grhoup2)) > small)
    {
        xc_builtin::pbex(iflag,
                         static_cast<FPTYPE>(2.0) * rhoup,
                         static_cast<FPTYPE>(4.0) * grhoup2,
                         sxup,
                         v1xup,
                         v2xup);
    }
    if (rhodw > small && std::sqrt(std::abs(grhodw2)) > small)
    {
        xc_builtin::pbex(iflag,
                         static_cast<FPTYPE>(2.0) * rhodw,
                         static_cast<FPTYPE>(4.0) * grhodw2,
                         sxdw,
                         v1xdw,
                         v2xdw);
    }
    sx = 0.5 * (sxup + sxdw);
    v2xup *= 2.0;
    v2xdw *= 2.0;
}

template <typename FPTYPE>
void xc_gcc_pbe_spin(const FPTYPE rho,
                     FPTYPE zeta,
                     const FPTYPE grho,
                     const int iflag,
                     FPTYPE& sc,
                     FPTYPE& v1cup,
                     FPTYPE& v1cdw,
                     FPTYPE& v2c)
{
    const FPTYPE small = 1.0e-10;
    const FPTYPE epsr = 1.0e-6;
    sc = 0.0;
    v1cup = 0.0;
    v1cdw = 0.0;
    v2c = 0.0;
    if (std::abs(zeta) - 1.0 > small || rho <= small || std::sqrt(std::abs(grho)) <= small)
    {
        return;
    }
    const FPTYPE x = std::min(std::abs(zeta), static_cast<FPTYPE>(1.0) - epsr);
    zeta = zeta > 0.0 ? x : -x;
    xc_builtin::pbec_spin(rho, zeta, grho, iflag == 2 ? 2 : 1, sc, v1cup, v1cdw, v2c);
}

} // namespace

template <typename FPTYPE, typename Device>
void xc_scalar_pbe_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                     const int nrxx,
                                                     const FPTYPE e2,
                                                     const FPTYPE epsr,
                                                     const FPTYPE* rho,
                                                     const FPTYPE* rho_core,
                                                     FPTYPE* rho_total,
                                                     FPTYPE* v,
                                                     FPTYPE* etxc,
                                                     FPTYPE* vtxc)
{
    *etxc = 0.0;
    *vtxc = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        const FPTYPE rhox = rho[ir] + rho_core[ir];
        rho_total[ir] = rhox;
        v[ir] = 0.0;
        const FPTYPE arho = std::abs(rhox);
        if (arho <= epsr)
        {
            continue;
        }
        FPTYPE exc = 0.0;
        FPTYPE vxc = 0.0;
        xc_builtin::scalar_pbe(arho, exc, vxc);
        v[ir] = e2 * vxc;
        *etxc += e2 * exc * rhox;
        *vtxc += e2 * vxc * rho[ir];
    }
}

template <typename FPTYPE, typename Device>
void xc_scalar_lda_spin_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                       const int nrxx,
                                                       const int correlation,
                                                       const FPTYPE e2,
                                                       const FPTYPE epsr,
                                                       const FPTYPE* rho_up,
                                                       const FPTYPE* rho_dw,
                                                       const FPTYPE* rho_core,
                                                       FPTYPE* rho_up_total,
                                                       FPTYPE* rho_dw_total,
                                                       FPTYPE* v,
                                                       FPTYPE* etxc,
                                                       FPTYPE* vtxc)
{
    *etxc = 0.0;
    *vtxc = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        const FPTYPE rhoup = rho_up[ir] + 0.5 * rho_core[ir];
        const FPTYPE rhodw = rho_dw[ir] + 0.5 * rho_core[ir];
        const FPTYPE rhox = rhoup + rhodw;
        rho_up_total[ir] = rhoup;
        rho_dw_total[ir] = rhodw;
        v[ir] = 0.0;
        v[nrxx + ir] = 0.0;
        const FPTYPE arho = std::abs(rhox);
        if (arho <= epsr)
        {
            continue;
        }
        FPTYPE zeta = (rho_up[ir] - rho_dw[ir]) / arho;
        if (std::abs(zeta) > 1.0)
        {
            zeta = zeta > 0.0 ? 1.0 : -1.0;
        }
        FPTYPE exc = 0.0;
        FPTYPE vup = 0.0;
        FPTYPE vdw = 0.0;
        xc_builtin::scalar_lda_spin(arho, zeta, correlation, exc, vup, vdw);
        v[ir] = e2 * vup;
        v[nrxx + ir] = e2 * vdw;
        *etxc += e2 * exc * rhox;
        *vtxc += e2 * (vup * rho_up[ir] + vdw * rho_dw[ir]);
    }
}

template <typename FPTYPE, typename Device>
void xc_gradcorr_pbe_grid_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                         const int nrxx,
                                                         const int iflag,
                                                         const FPTYPE e2,
                                                         const FPTYPE epsr,
                                                         const FPTYPE* rho,
                                                         const FPTYPE* rho_core,
                                                         const FPTYPE* gdr,
                                                         FPTYPE* v,
                                                         FPTYPE* h,
                                                         FPTYPE* etxc,
                                                         FPTYPE* vtxc)
{
    *etxc = 0.0;
    *vtxc = 0.0;

    for (int ir = 0; ir < nrxx; ++ir)
    {
        v[ir] = 0.0;
        h[3 * ir + 0] = 0.0;
        h[3 * ir + 1] = 0.0;
        h[3 * ir + 2] = 0.0;

        const FPTYPE arho = std::abs(rho[ir]);
        if (arho <= epsr)
        {
            continue;
        }

        const FPTYPE gx = gdr[3 * ir + 0];
        const FPTYPE gy = gdr[3 * ir + 1];
        const FPTYPE gz = gdr[3 * ir + 2];
        const FPTYPE grho = gx * gx + gy * gy + gz * gz;
        if (grho < static_cast<FPTYPE>(1.0e-10))
        {
            continue;
        }

        FPTYPE sx = 0.0;
        FPTYPE v1x = 0.0;
        FPTYPE v2x = 0.0;
        FPTYPE sc = 0.0;
        FPTYPE v1c = 0.0;
        FPTYPE v2c = 0.0;
        xc_builtin::pbex(iflag, arho, grho, sx, v1x, v2x);
        xc_builtin::pbec(iflag, arho, grho, sc, v1c, v2c);

        const FPTYPE sxc = sx + sc;
        const FPTYPE v1xc = v1x + v1c;
        const FPTYPE v2xc = v2x + v2c;
        const FPTYPE segno = rho[ir] >= 0.0 ? 1.0 : -1.0;
        v[ir] = e2 * v1xc;
        h[3 * ir + 0] = e2 * v2xc * gx;
        h[3 * ir + 1] = e2 * v2xc * gy;
        h[3 * ir + 2] = e2 * v2xc * gz;
        *vtxc += e2 * v1xc * (rho[ir] - rho_core[ir]);
        *etxc += e2 * sxc * segno;
    }
}

template <typename FPTYPE, typename Device>
void xc_gradcorr_pbe_grid_resident_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                                  const int nrxx,
                                                                  const int iflag,
                                                                  const FPTYPE e2,
                                                                  const FPTYPE epsr,
                                                                  const FPTYPE* rho,
                                                                  const FPTYPE* rho_core,
                                                                  const FPTYPE* gdr,
                                                                  FPTYPE* v,
                                                                  FPTYPE* h,
                                                                  FPTYPE* etxc,
                                                                  FPTYPE* vtxc)
{
    std::vector<FPTYPE> delta_v(nrxx, 0.0);
    xc_gradcorr_pbe_grid_op<FPTYPE, Device>()(ctx,
                                              nrxx,
                                              iflag,
                                              e2,
                                              epsr,
                                              rho,
                                              rho_core,
                                              gdr,
                                              delta_v.data(),
                                              h,
                                              etxc,
                                              vtxc);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        v[ir] += delta_v[ir];
    }
}

template <typename FPTYPE, typename Device>
void xc_gradcorr_pbe_spin_grid_resident_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                                       const int nrxx,
                                                                       const int iflag,
                                                                       const FPTYPE e2,
                                                                       const FPTYPE epsr,
                                                                       const FPTYPE* rho_up,
                                                                       const FPTYPE* rho_dw,
                                                                       const FPTYPE* rho_core,
                                                                       const FPTYPE* gdr_up,
                                                                       const FPTYPE* gdr_dw,
                                                                       FPTYPE* v,
                                                                       FPTYPE* h_up,
                                                                       FPTYPE* h_dw,
                                                                       FPTYPE* etxc,
                                                                       FPTYPE* vtxc)
{
    *etxc = 0.0;
    *vtxc = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        h_up[3 * ir + 0] = 0.0;
        h_up[3 * ir + 1] = 0.0;
        h_up[3 * ir + 2] = 0.0;
        h_dw[3 * ir + 0] = 0.0;
        h_dw[3 * ir + 1] = 0.0;
        h_dw[3 * ir + 2] = 0.0;

        const FPTYPE rhoup = rho_up[ir];
        const FPTYPE rhodw = rho_dw[ir];
        const FPTYPE rh = rhoup + rhodw;
        const FPTYPE gxup = gdr_up[3 * ir + 0];
        const FPTYPE gyup = gdr_up[3 * ir + 1];
        const FPTYPE gzup = gdr_up[3 * ir + 2];
        const FPTYPE gxdw = gdr_dw[3 * ir + 0];
        const FPTYPE gydw = gdr_dw[3 * ir + 1];
        const FPTYPE gzdw = gdr_dw[3 * ir + 2];
        const FPTYPE grho2up = gxup * gxup + gyup * gyup + gzup * gzup;
        const FPTYPE grho2dw = gxdw * gxdw + gydw * gydw + gzdw * gzdw;

        FPTYPE sx = 0.0;
        FPTYPE sc = 0.0;
        FPTYPE v1xup = 0.0;
        FPTYPE v1xdw = 0.0;
        FPTYPE v2xup = 0.0;
        FPTYPE v2xdw = 0.0;
        FPTYPE v1cup = 0.0;
        FPTYPE v1cdw = 0.0;
        FPTYPE v2c = 0.0;
        xc_gcx_pbe_spin(rhoup, rhodw, grho2up, grho2dw, iflag, sx, v1xup, v1xdw, v2xup, v2xdw);
        if (rh > epsr)
        {
            FPTYPE zeta = (rhoup - rhodw) / rh;
            const FPTYPE grh2 = (gxup + gxdw) * (gxup + gxdw) + (gyup + gydw) * (gyup + gydw)
                                + (gzup + gzdw) * (gzup + gzdw);
            xc_gcc_pbe_spin(rh, zeta, grh2, iflag, sc, v1cup, v1cdw, v2c);
        }

        const FPTYPE v1up = v1xup + v1cup;
        const FPTYPE v1dw = v1xdw + v1cdw;
        v[ir] += e2 * v1up;
        v[nrxx + ir] += e2 * v1dw;
        h_up[3 * ir + 0] = e2 * ((v2xup + v2c) * gxup + v2c * gxdw);
        h_up[3 * ir + 1] = e2 * ((v2xup + v2c) * gyup + v2c * gydw);
        h_up[3 * ir + 2] = e2 * ((v2xup + v2c) * gzup + v2c * gzdw);
        h_dw[3 * ir + 0] = e2 * ((v2xdw + v2c) * gxdw + v2c * gxup);
        h_dw[3 * ir + 1] = e2 * ((v2xdw + v2c) * gydw + v2c * gyup);
        h_dw[3 * ir + 2] = e2 * ((v2xdw + v2c) * gzdw + v2c * gzup);
        *vtxc += e2 * v1up * (rhoup - 0.5 * rho_core[ir]);
        *vtxc += e2 * v1dw * (rhodw - 0.5 * rho_core[ir]);
        *etxc += e2 * (sx + sc);
    }
}

template <typename FPTYPE, typename Device>
void xc_gradcorr_pbe_stress_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                           const int nrxx,
                                                           const int iflag,
                                                           const FPTYPE e2,
                                                           const FPTYPE epsr,
                                                           const FPTYPE* rho,
                                                           const FPTYPE* gdr,
                                                           FPTYPE* stress)
{
    for (int i = 0; i < 9; ++i)
    {
        stress[i] = 0.0;
    }

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const FPTYPE arho = std::abs(rho[ir]);
        if (arho <= epsr)
        {
            continue;
        }

        const FPTYPE gx = gdr[3 * ir + 0];
        const FPTYPE gy = gdr[3 * ir + 1];
        const FPTYPE gz = gdr[3 * ir + 2];
        const FPTYPE grho = gx * gx + gy * gy + gz * gz;
        if (grho < static_cast<FPTYPE>(1.0e-10))
        {
            continue;
        }

        FPTYPE sx = 0.0;
        FPTYPE v1x = 0.0;
        FPTYPE v2x = 0.0;
        FPTYPE sc = 0.0;
        FPTYPE v1c = 0.0;
        FPTYPE v2c = 0.0;
        xc_builtin::pbex(iflag, arho, grho, sx, v1x, v2x);
        xc_builtin::pbec(iflag, arho, grho, sc, v1c, v2c);

        const FPTYPE grad[3] = {gx, gy, gz};
        const FPTYPE v2xc = v2x + v2c;
        for (int l = 0; l < 3; ++l)
        {
            for (int m = 0; m <= l; ++m)
            {
                stress[l * 3 + m] += grad[l] * grad[m] * e2 * v2xc;
            }
        }
    }
}

template <typename FPTYPE, typename Device>
void xc_gradcorr_pbe_spin_stress_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                                const int nrxx,
                                                                const int iflag,
                                                                const FPTYPE e2,
                                                                const FPTYPE epsr,
                                                                const FPTYPE* rho_up,
                                                                const FPTYPE* rho_dw,
                                                                const FPTYPE* gdr_up,
                                                                const FPTYPE* gdr_dw,
                                                                FPTYPE* stress)
{
    for (int i = 0; i < 9; ++i)
    {
        stress[i] = 0.0;
    }

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const FPTYPE rhoup = rho_up[ir];
        const FPTYPE rhodw = rho_dw[ir];
        const FPTYPE rh = rhoup + rhodw;
        const FPTYPE gxup = gdr_up[3 * ir + 0];
        const FPTYPE gyup = gdr_up[3 * ir + 1];
        const FPTYPE gzup = gdr_up[3 * ir + 2];
        const FPTYPE gxdw = gdr_dw[3 * ir + 0];
        const FPTYPE gydw = gdr_dw[3 * ir + 1];
        const FPTYPE gzdw = gdr_dw[3 * ir + 2];
        const FPTYPE grho2up = gxup * gxup + gyup * gyup + gzup * gzup;
        const FPTYPE grho2dw = gxdw * gxdw + gydw * gydw + gzdw * gzdw;

        FPTYPE sx = 0.0;
        FPTYPE sc = 0.0;
        FPTYPE v1xup = 0.0;
        FPTYPE v1xdw = 0.0;
        FPTYPE v2xup = 0.0;
        FPTYPE v2xdw = 0.0;
        FPTYPE v1cup = 0.0;
        FPTYPE v1cdw = 0.0;
        FPTYPE v2c = 0.0;
        xc_gcx_pbe_spin(rhoup, rhodw, grho2up, grho2dw, iflag, sx, v1xup, v1xdw, v2xup, v2xdw);
        if (rh > epsr)
        {
            FPTYPE zeta = (rhoup - rhodw) / rh;
            const FPTYPE grh2 = (gxup + gxdw) * (gxup + gxdw) + (gyup + gydw) * (gyup + gydw)
                                + (gzup + gzdw) * (gzup + gzdw);
            xc_gcc_pbe_spin(rh, zeta, grh2, iflag, sc, v1cup, v1cdw, v2c);
        }

        const FPTYPE grad_up[3] = {gxup, gyup, gzup};
        const FPTYPE grad_dw[3] = {gxdw, gydw, gzdw};
        for (int l = 0; l < 3; ++l)
        {
            for (int m = 0; m <= l; ++m)
            {
                stress[l * 3 + m] += grad_up[l] * grad_up[m] * e2 * v2xup
                                     + grad_dw[l] * grad_dw[m] * e2 * v2xdw;
                stress[l * 3 + m] += (grad_up[l] * grad_up[m] * v2c
                                      + grad_dw[l] * grad_dw[m] * v2c
                                      + (grad_up[l] * grad_dw[m] + grad_dw[l] * grad_up[m]) * v2c)
                                     * e2;
            }
        }
    }
}

template <typename FPTYPE, typename Device>
void xc_apply_dh_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                const int nrxx,
                                                const FPTYPE* rho,
                                                const FPTYPE* rho_core,
                                                const FPTYPE* dh,
                                                FPTYPE* v,
                                                FPTYPE* vtxc_delta)
{
    *vtxc_delta = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        v[ir] -= dh[ir];
        *vtxc_delta -= dh[ir] * (rho[ir] - rho_core[ir]);
    }
}

template <typename FPTYPE, typename Device>
void xc_add_potential_op<FPTYPE, Device>::operator()(const Device* ctx, const int size, const FPTYPE* src, FPTYPE* dst)
{
    for (int i = 0; i < size; ++i)
    {
        dst[i] += src[i];
    }
}

template <typename FPTYPE, typename Device>
void xc_apply_dh_spin_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                     const int nrxx,
                                                     const FPTYPE* rho,
                                                     const FPTYPE* rho_core,
                                                     const FPTYPE* dh,
                                                     FPTYPE* v,
                                                     FPTYPE* vtxc_delta)
{
    *vtxc_delta = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        v[ir] -= dh[ir];
        *vtxc_delta -= dh[ir] * (rho[ir] - 0.5 * rho_core[ir]);
    }
}

template <typename FPTYPE, typename Device>
void xc_noncolin_rho_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                    const int nrxx,
                                                    const bool lsign,
                                                    const FPTYPE* rho0,
                                                    const FPTYPE* rho1,
                                                    const FPTYPE* rho2,
                                                    const FPTYPE* rho3,
                                                    const FPTYPE* ux,
                                                    FPTYPE* rho_up,
                                                    FPTYPE* rho_dw,
                                                    FPTYPE* neg)
{
    for (int ir = 0; ir < nrxx; ++ir)
    {
        neg[ir] = 1.0;
        if (lsign)
        {
            const FPTYPE projection = rho1[ir] * ux[0] + rho2[ir] * ux[1] + rho3[ir] * ux[2];
            neg[ir] = projection > 0.0 ? 1.0 : -1.0;
        }
        const FPTYPE amag = std::sqrt(rho1[ir] * rho1[ir] + rho2[ir] * rho2[ir] + rho3[ir] * rho3[ir]);
        rho_up[ir] = 0.5 * (rho0[ir] + neg[ir] * amag);
        rho_dw[ir] = 0.5 * (rho0[ir] - neg[ir] * amag);
    }
}

template <typename FPTYPE, typename Device>
void xc_noncolin_rotate_potential_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                                 const int nrxx,
                                                                 const FPTYPE* rho1,
                                                                 const FPTYPE* rho2,
                                                                 const FPTYPE* rho3,
                                                                 const FPTYPE* neg,
                                                                 const FPTYPE* v_up,
                                                                 const FPTYPE* v_dw,
                                                                 FPTYPE* v0,
                                                                 FPTYPE* v1,
                                                                 FPTYPE* v2,
                                                                 FPTYPE* v3)
{
    for (int ir = 0; ir < nrxx; ++ir)
    {
        const FPTYPE vavg = 0.5 * (v_up[ir] + v_dw[ir]);
        const FPTYPE vdiff = 0.5 * (v_up[ir] - v_dw[ir]);
        v0[ir] += vavg;
        const FPTYPE amag = std::sqrt(rho1[ir] * rho1[ir] + rho2[ir] * rho2[ir] + rho3[ir] * rho3[ir]);
        if (amag > 1.0e-12)
        {
            const FPTYPE factor = neg[ir] * vdiff / amag;
            v1[ir] += factor * rho1[ir];
            v2[ir] += factor * rho2[ir];
            v3[ir] += factor * rho3[ir];
        }
    }
}

template <typename FPTYPE, typename Device>
void xc_multiply_iG_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                   const int npw,
                                                   const int ipol,
                                                   const FPTYPE* gcar,
                                                   const std::complex<FPTYPE>* rhog,
                                                   std::complex<FPTYPE>* porter)
{
    const std::complex<FPTYPE> imaginary(0.0, 1.0);
    for (int ig = 0; ig < npw; ++ig)
    {
        porter[ig] = imaginary * rhog[ig] * gcar[3 * ig + ipol];
    }
}

template <typename FPTYPE, typename Device>
void xc_accumulate_iG_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                     const int npw,
                                                     const int ipol,
                                                     const FPTYPE* gcar,
                                                     const std::complex<FPTYPE>* rhog,
                                                     std::complex<FPTYPE>* accum,
                                                     const bool zero_first)
{
    const std::complex<FPTYPE> imaginary(0.0, 1.0);
    for (int ig = 0; ig < npw; ++ig)
    {
        const std::complex<FPTYPE> term = imaginary * rhog[ig] * gcar[3 * ig + ipol];
        accum[ig] = zero_first ? term : accum[ig] + term;
    }
}

template <typename FPTYPE, typename Device>
void xc_set_component_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                     const int nrxx,
                                                     const int ipol,
                                                     const FPTYPE* component,
                                                     FPTYPE* interleaved)
{
    for (int ir = 0; ir < nrxx; ++ir)
    {
        interleaved[3 * ir + ipol] = component[ir];
    }
}

template <typename FPTYPE, typename Device>
void xc_extract_component_op<FPTYPE, Device>::operator()(const Device* ctx,
                                                         const int nrxx,
                                                         const int ipol,
                                                         const FPTYPE* interleaved,
                                                         FPTYPE* component)
{
    for (int ir = 0; ir < nrxx; ++ir)
    {
        component[ir] = interleaved[3 * ir + ipol];
    }
}

template struct xc_scalar_pbe_op<float, base_device::DEVICE_CPU>;
template struct xc_scalar_pbe_op<double, base_device::DEVICE_CPU>;
template struct xc_scalar_lda_spin_op<float, base_device::DEVICE_CPU>;
template struct xc_scalar_lda_spin_op<double, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_resident_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_resident_op<double, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_spin_grid_resident_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_spin_grid_resident_op<double, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_stress_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_spin_stress_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_spin_stress_op<double, base_device::DEVICE_CPU>;
template struct xc_apply_dh_op<float, base_device::DEVICE_CPU>;
template struct xc_apply_dh_op<double, base_device::DEVICE_CPU>;
template struct xc_add_potential_op<float, base_device::DEVICE_CPU>;
template struct xc_add_potential_op<double, base_device::DEVICE_CPU>;
template struct xc_apply_dh_spin_op<float, base_device::DEVICE_CPU>;
template struct xc_apply_dh_spin_op<double, base_device::DEVICE_CPU>;
template struct xc_noncolin_rho_op<float, base_device::DEVICE_CPU>;
template struct xc_noncolin_rho_op<double, base_device::DEVICE_CPU>;
template struct xc_noncolin_rotate_potential_op<float, base_device::DEVICE_CPU>;
template struct xc_noncolin_rotate_potential_op<double, base_device::DEVICE_CPU>;
template struct xc_multiply_iG_op<float, base_device::DEVICE_CPU>;
template struct xc_multiply_iG_op<double, base_device::DEVICE_CPU>;
template struct xc_accumulate_iG_op<float, base_device::DEVICE_CPU>;
template struct xc_accumulate_iG_op<double, base_device::DEVICE_CPU>;
template struct xc_set_component_op<float, base_device::DEVICE_CPU>;
template struct xc_set_component_op<double, base_device::DEVICE_CPU>;
template struct xc_extract_component_op<float, base_device::DEVICE_CPU>;
template struct xc_extract_component_op<double, base_device::DEVICE_CPU>;

} // namespace hamilt

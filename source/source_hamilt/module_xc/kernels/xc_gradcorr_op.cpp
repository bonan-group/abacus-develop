#include "source_hamilt/module_xc/kernels/xc_gradcorr_op.h"

#include "source_base/constants.h"

#include <cmath>
#include <complex>
#include <vector>

namespace hamilt
{

namespace
{

template <typename FPTYPE>
void xc_pw(const FPTYPE rs, const int iflag, FPTYPE& ec, FPTYPE& vc)
{
    const FPTYPE a = 0.0310910;
    const FPTYPE b1 = 7.59570;
    const FPTYPE b2 = 3.58760;
    const FPTYPE a1[2] = {0.213700, 0.0264810};
    const FPTYPE b3[2] = {1.63820, -0.466470};
    const FPTYPE b4[2] = {0.492940, 0.133540};

    const FPTYPE rs12 = std::sqrt(rs);
    const FPTYPE rs32 = rs * rs12;
    const FPTYPE rs2 = rs * rs;
    const FPTYPE om = 2.0 * a * (b1 * rs12 + b2 * rs + b3[iflag] * rs32 + b4[iflag] * rs2);
    const FPTYPE dom = 2.0 * a * (0.50 * b1 * rs12 + b2 * rs + 1.50 * b3[iflag] * rs32
                                   + 2.0 * b4[iflag] * rs2);
    const FPTYPE olog = std::log(1.0 + 1.0 / om);
    ec = -2.0 * a * (1.0 + a1[iflag] * rs) * olog;
    vc = -2.0 * a * (1.0 + 2.0 / 3.0 * a1[iflag] * rs) * olog
         - 2.0 / 3.0 * a * (1.0 + a1[iflag] * rs) * dom / (om * (om + 1.0));
}

template <typename FPTYPE>
void xc_pbex(const FPTYPE rho, const FPTYPE grho, const int iflag, FPTYPE& sx, FPTYPE& v1x, FPTYPE& v2x)
{
    const FPTYPE third = 1.0 / 3.0;
    const FPTYPE c1 = 0.750 / static_cast<FPTYPE>(ModuleBase::PI);
    const FPTYPE c2 = 3.0936677262801360;
    const FPTYPE c5 = 4.0 * third;
    const FPTYPE k[3] = {0.8040, 1.24500, 0.8040};
    const FPTYPE mu[3] = {0.2195149727645171, 0.2195149727645171, 0.12345679012345679};

    const FPTYPE agrho = std::sqrt(grho);
    const FPTYPE kf = c2 * std::pow(rho, third);
    const FPTYPE dsg = 0.50 / kf;
    const FPTYPE s1 = agrho * dsg / rho;
    const FPTYPE s2 = s1 * s1;
    const FPTYPE ds = -c5 * s1;

    const FPTYPE f1 = s2 * mu[iflag] / k[iflag];
    const FPTYPE f2 = 1.0 + f1;
    const FPTYPE f3 = k[iflag] / f2;
    const FPTYPE fx = k[iflag] - f3;
    const FPTYPE exunif = -c1 * kf;
    sx = exunif * fx;

    const FPTYPE dxunif = exunif * third;
    const FPTYPE dfx1 = f2 * f2;
    const FPTYPE dfx = 2.0 * mu[iflag] * s1 / dfx1;

    v1x = sx + dxunif * fx + exunif * dfx * ds;
    v2x = exunif * dfx * dsg / agrho;
    sx *= rho;
}

template <typename FPTYPE>
void xc_pbec(const FPTYPE rho, const FPTYPE grho, const int iflag, FPTYPE& sc, FPTYPE& v1c, FPTYPE& v2c)
{
    const FPTYPE ga = 0.0310906908696548950;
    const FPTYPE be[2] = {0.06672455060314922, 0.046};
    const FPTYPE third = 1.0 / 3.0;
    const FPTYPE pi34 = 0.62035049089940;
    const FPTYPE xkf = 1.9191582926775130;
    const FPTYPE xks = 1.1283791670955130;

    FPTYPE ec = 0.0;
    FPTYPE vc = 0.0;
    const FPTYPE rs = pi34 / std::pow(rho, third);
    xc_pw(rs, 0, ec, vc);

    const FPTYPE kf = xkf / rs;
    const FPTYPE ks = xks * std::sqrt(kf);
    const FPTYPE t = std::sqrt(grho) / (2.0 * ks * rho);
    const FPTYPE expe = std::exp(-ec / ga);
    const FPTYPE af = be[iflag] / ga * (1.0 / (expe - 1.0));
    const FPTYPE bf = expe * (vc - ec);
    const FPTYPE y = af * t * t;
    const FPTYPE xy = (1.0 + y) / (1.0 + y + y * y);
    const FPTYPE x = 1.0 + y + y * y;
    const FPTYPE qy = y * y * (2.0 + y) / (x * x);
    const FPTYPE s1 = 1.0 + be[iflag] / ga * t * t * xy;
    const FPTYPE h0 = ga * std::log(s1);
    const FPTYPE dh0 = be[iflag] * t * t / s1 * (-7.0 / 3.0 * xy - qy * (af * bf / be[iflag] - 7.0 / 3.0));
    const FPTYPE ddh0 = be[iflag] / (2.0 * ks * ks * rho) * (xy - qy) / s1;

    sc = rho * h0;
    v1c = h0 + dh0;
    v2c = ddh0;
}

template <typename FPTYPE>
void xc_slater_rs(const FPTYPE rs, FPTYPE& ex, FPTYPE& vx)
{
    const FPTYPE f = -0.687247939924714;
    const FPTYPE alpha = 2.0 / 3.0;
    ex = f * alpha / rs;
    vx = 4.0 / 3.0 * f * alpha / rs;
}

template <typename FPTYPE>
void xc_scalar_pbe(const FPTYPE rho, FPTYPE& exc, FPTYPE& vxc)
{
    const FPTYPE pi34 = 0.62035049089940;
    const FPTYPE rs = pi34 / std::pow(rho, 1.0 / 3.0);
    FPTYPE ex = 0.0;
    FPTYPE vx = 0.0;
    FPTYPE ec = 0.0;
    FPTYPE vc = 0.0;
    xc_slater_rs(rs, ex, vx);
    xc_pw(rs, 0, ec, vc);
    exc = ex + ec;
    vxc = vx + vc;
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
        xc_scalar_pbe(arho, exc, vxc);
        v[ir] = e2 * vxc;
        *etxc += e2 * exc * rhox;
        *vtxc += e2 * vxc * rho[ir];
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
        xc_pbex(arho, grho, iflag, sx, v1x, v2x);
        xc_pbec(arho, grho, iflag == 2 ? 1 : iflag, sc, v1c, v2c);

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
template struct xc_gradcorr_pbe_grid_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_resident_op<float, base_device::DEVICE_CPU>;
template struct xc_gradcorr_pbe_grid_resident_op<double, base_device::DEVICE_CPU>;
template struct xc_apply_dh_op<float, base_device::DEVICE_CPU>;
template struct xc_apply_dh_op<double, base_device::DEVICE_CPU>;
template struct xc_multiply_iG_op<float, base_device::DEVICE_CPU>;
template struct xc_multiply_iG_op<double, base_device::DEVICE_CPU>;
template struct xc_accumulate_iG_op<float, base_device::DEVICE_CPU>;
template struct xc_accumulate_iG_op<double, base_device::DEVICE_CPU>;
template struct xc_set_component_op<float, base_device::DEVICE_CPU>;
template struct xc_set_component_op<double, base_device::DEVICE_CPU>;
template struct xc_extract_component_op<float, base_device::DEVICE_CPU>;
template struct xc_extract_component_op<double, base_device::DEVICE_CPU>;

} // namespace hamilt

#ifndef SOURCE_HAMILT_MODULE_XC_KERNELS_XC_BUILTIN_FORMULA_H_
#define SOURCE_HAMILT_MODULE_XC_KERNELS_XC_BUILTIN_FORMULA_H_

#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ABACUS_XC_HD __host__ __device__ inline
#else
#define ABACUS_XC_HD inline
#endif

namespace hamilt
{
namespace xc_builtin
{

template <typename T>
ABACUS_XC_HD void slater(const T rs, T& ex, T& vx)
{
    const T f = -0.687247939924714;
    const T alpha = 2.0 / 3.0;
    ex = f * alpha / rs;
    vx = 4.0 / 3.0 * f * alpha / rs;
}

template <typename T>
ABACUS_XC_HD void slater_spin(const T rho, const T zeta, T& ex, T& vxup, T& vxdw)
{
    const T f = -1.107838149573033610;
    const T alpha = 2.0 / 3.0;
    const T third = 1.0 / 3.0;
    const T p43 = 4.0 / 3.0;

    T rho13 = pow((1.0 + zeta) * rho, third);
    const T exup = f * alpha * rho13;
    vxup = p43 * f * alpha * rho13;
    rho13 = pow((1.0 - zeta) * rho, third);
    const T exdw = f * alpha * rho13;
    vxdw = p43 * f * alpha * rho13;
    ex = 0.5 * ((1.0 + zeta) * exup + (1.0 - zeta) * exdw);
}

template <typename T>
ABACUS_XC_HD void pw_interpolation(const T rs, const int iflag, T& ec, T& vc)
{
    const T a = 0.0310910;
    const T b1 = 7.59570;
    const T b2 = 3.58760;
    const T a1[2] = {0.213700, 0.0264810};
    const T b3[2] = {1.63820, -0.466470};
    const T b4[2] = {0.492940, 0.133540};

    const T rs12 = sqrt(rs);
    const T rs32 = rs * rs12;
    const T rs2 = rs * rs;
    const T om = 2.0 * a * (b1 * rs12 + b2 * rs + b3[iflag] * rs32 + b4[iflag] * rs2);
    const T dom = 2.0 * a * (0.50 * b1 * rs12 + b2 * rs + 1.50 * b3[iflag] * rs32
                              + 2.0 * b4[iflag] * rs2);
    const T olog = log(1.0 + 1.0 / om);
    ec = -2.0 * a * (1.0 + a1[iflag] * rs) * olog;
    vc = -2.0 * a * (1.0 + 2.0 / 3.0 * a1[iflag] * rs) * olog
         - 2.0 / 3.0 * a * (1.0 + a1[iflag] * rs) * dom / (om * (om + 1.0));
}

template <typename T>
ABACUS_XC_HD void pw_interpolation(const T rs, T& ec, T& vc)
{
    pw_interpolation(rs, 0, ec, vc);
}

template <typename T>
ABACUS_XC_HD void pz(const T rs, const int iflag, T& ec, T& vc)
{
    const T a[2] = {0.0311, 0.031091};
    const T b[2] = {-0.048, -0.046644};
    const T c[2] = {0.0020, 0.00419};
    const T d[2] = {-0.0116, -0.00983};
    const T gc[2] = {-0.1423, -0.103756};
    const T b1[2] = {1.0529, 0.56371};
    const T b2[2] = {0.3334, 0.27358};

    if (rs < 1.0)
    {
        const T lnrs = log(rs);
        ec = a[iflag] * lnrs + b[iflag] + c[iflag] * rs * lnrs + d[iflag] * rs;
        vc = a[iflag] * lnrs + (b[iflag] - a[iflag] / 3.0) + 2.0 / 3.0 * c[iflag] * rs * lnrs
             + (2.0 * d[iflag] - c[iflag]) / 3.0 * rs;
    }
    else
    {
        const T rs12 = sqrt(rs);
        const T ox = 1.0 + b1[iflag] * rs12 + b2[iflag] * rs;
        const T dox = 1.0 + 7.0 / 6.0 * b1[iflag] * rs12 + 4.0 / 3.0 * b2[iflag] * rs;
        ec = gc[iflag] / ox;
        vc = ec * dox / ox;
    }
}

template <typename T>
ABACUS_XC_HD void pz_polarized(const T rs, T& ec, T& vc)
{
    const T a = 0.015550;
    const T b = -0.02690;
    const T c = 0.00070;
    const T d = -0.00480;
    const T gc = -0.08430;
    const T b1 = 1.39810;
    const T b2 = 0.26110;

    if (rs < 1.0)
    {
        const T lnrs = log(rs);
        ec = a * lnrs + b + c * rs * lnrs + d * rs;
        vc = a * lnrs + (b - a / 3.0) + 2.0 / 3.0 * c * rs * lnrs + (2.0 * d - c) / 3.0 * rs;
    }
    else
    {
        const T rs12 = sqrt(rs);
        const T ox = 1.0 + b1 * rs12 + b2 * rs;
        const T dox = 1.0 + 7.0 / 6.0 * b1 * rs12 + 4.0 / 3.0 * b2 * rs;
        ec = gc / ox;
        vc = ec * dox / ox;
    }
}

template <typename T>
ABACUS_XC_HD void pz_spin(const T rs, const T zeta, T& ec, T& vcup, T& vcdw)
{
    T ecu = 0.0;
    T vcu = 0.0;
    T ecp = 0.0;
    T vcp = 0.0;
    const T p43 = 4.0 / 3.0;
    const T third = 1.0 / 3.0;
    pz(rs, 0, ecu, vcu);
    pz_polarized(rs, ecp, vcp);
    const T denom = pow(2.0, p43) - 2.0;
    const T fz = (pow(1.0 + zeta, p43) + pow(1.0 - zeta, p43) - 2.0) / denom;
    const T dfz = p43 * (pow(1.0 + zeta, third) - pow(1.0 - zeta, third)) / denom;
    ec = ecu + fz * (ecp - ecu);
    vcup = vcu + fz * (vcp - vcu) + (ecp - ecu) * dfz * (1.0 - zeta);
    vcdw = vcu + fz * (vcp - vcu) + (ecp - ecu) * dfz * (-1.0 - zeta);
}

template <typename T>
ABACUS_XC_HD void pw_spin(const T rs, const T zeta, T& ec, T& vcup, T& vcdw)
{
    const T a = 0.0310910;
    const T a1 = 0.213700;
    const T b1 = 7.59570;
    const T b2 = 3.58760;
    const T b3 = 1.63820;
    const T b4 = 0.492940;
    const T ap = 0.0155450;
    const T a1p = 0.205480;
    const T b1p = 14.11890;
    const T b2p = 6.19770;
    const T b3p = 3.36620;
    const T b4p = 0.625170;
    const T aa = 0.0168870;
    const T a1a = 0.111250;
    const T b1a = 10.3570;
    const T b2a = 3.62310;
    const T b3a = 0.880260;
    const T b4a = 0.496710;
    const T fz0 = 1.7099210;

    const T zeta2 = zeta * zeta;
    const T zeta3 = zeta2 * zeta;
    const T zeta4 = zeta3 * zeta;
    const T rs12 = sqrt(rs);
    const T rs32 = rs * rs12;
    const T rs2 = rs * rs;

    T om = 2.0 * a * (b1 * rs12 + b2 * rs + b3 * rs32 + b4 * rs2);
    T dom = 2.0 * a * (0.50 * b1 * rs12 + b2 * rs + 1.50 * b3 * rs32 + 2.0 * b4 * rs2);
    T olog = log(1.0 + 1.0 / om);
    const T epwc = -2.0 * a * (1.0 + a1 * rs) * olog;
    const T vpwc = -2.0 * a * (1.0 + 2.0 / 3.0 * a1 * rs) * olog
                   - 2.0 / 3.0 * a * (1.0 + a1 * rs) * dom / (om * (om + 1.0));

    om = 2.0 * ap * (b1p * rs12 + b2p * rs + b3p * rs32 + b4p * rs2);
    dom = 2.0 * ap * (0.50 * b1p * rs12 + b2p * rs + 1.50 * b3p * rs32 + 2.0 * b4p * rs2);
    olog = log(1.0 + 1.0 / om);
    const T epwcp = -2.0 * ap * (1.0 + a1p * rs) * olog;
    const T vpwcp = -2.0 * ap * (1.0 + 2.0 / 3.0 * a1p * rs) * olog
                    - 2.0 / 3.0 * ap * (1.0 + a1p * rs) * dom / (om * (om + 1.0));

    om = 2.0 * aa * (b1a * rs12 + b2a * rs + b3a * rs32 + b4a * rs2);
    dom = 2.0 * aa * (0.50 * b1a * rs12 + b2a * rs + 1.50 * b3a * rs32 + 2.0 * b4a * rs2);
    olog = log(1.0 + 1.0 / om);
    const T alpha = 2.0 * aa * (1.0 + a1a * rs) * olog;
    const T vpwca = 2.0 * aa * (1.0 + 2.0 / 3.0 * a1a * rs) * olog
                    + 2.0 / 3.0 * aa * (1.0 + a1a * rs) * dom / (om * (om + 1.0));

    const T denom = pow(2.0, 4.0 / 3.0) - 2.0;
    const T fz = (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0) - 2.0) / denom;
    const T dfz = (pow(1.0 + zeta, 1.0 / 3.0) - pow(1.0 - zeta, 1.0 / 3.0)) * 4.0 / (3.0 * denom);
    const T common = alpha / fz0 * (dfz * (1.0 - zeta4) - 4.0 * fz * zeta3)
                     + (epwcp - epwc) * (dfz * zeta4 + 4.0 * fz * zeta3);
    ec = epwc + alpha * fz * (1.0 - zeta4) / fz0 + (epwcp - epwc) * fz * zeta4;
    vcup = vpwc + vpwca * fz * (1.0 - zeta4) / fz0 + (vpwcp - vpwc) * fz * zeta4
           + common * (1.0 - zeta);
    vcdw = vpwc + vpwca * fz * (1.0 - zeta4) / fz0 + (vpwcp - vpwc) * fz * zeta4
           - common * (1.0 + zeta);
}

template <typename T>
ABACUS_XC_HD void pbex(const int iflag, const T rho, const T grho, T& sx, T& v1x, T& v2x)
{
    const T third = 1.0 / 3.0;
    const T c1 = 0.750 / static_cast<T>(3.14159265358979323846);
    const T c2 = 3.0936677262801360;
    const T c5 = 4.0 * third;
    const T k[3] = {0.8040, 1.24500, 0.8040};
    const T mu[3] = {0.2195149727645171, 0.2195149727645171, 0.12345679012345679};

    const T agrho = sqrt(grho);
    const T kf = c2 * pow(rho, third);
    const T dsg = 0.50 / kf;
    const T s1 = agrho * dsg / rho;
    const T s2 = s1 * s1;
    const T ds = -c5 * s1;
    const T f1 = s2 * mu[iflag] / k[iflag];
    const T f2 = 1.0 + f1;
    const T f3 = k[iflag] / f2;
    const T fx = k[iflag] - f3;
    const T exunif = -c1 * kf;
    sx = exunif * fx;
    const T dxunif = exunif * third;
    const T dfx1 = f2 * f2;
    const T dfx = 2.0 * mu[iflag] * s1 / dfx1;
    v1x = sx + dxunif * fx + exunif * dfx * ds;
    v2x = exunif * dfx * dsg / agrho;
    sx *= rho;
}

template <typename T>
ABACUS_XC_HD void pbec(const int iflag, const T rho, const T grho, T& sc, T& v1c, T& v2c)
{
    const T ga = 0.0310906908696548950;
    const T be[2] = {0.06672455060314922, 0.046};
    const int parameter = iflag == 2 ? 1 : iflag;
    const T third = 1.0 / 3.0;
    const T pi34 = 0.62035049089940;
    const T xkf = 1.9191582926775130;
    const T xks = 1.1283791670955130;

    T ec = 0.0;
    T vc = 0.0;
    const T rs = pi34 / pow(rho, third);
    pw_interpolation(rs, ec, vc);
    const T kf = xkf / rs;
    const T ks = xks * sqrt(kf);
    const T t = sqrt(grho) / (2.0 * ks * rho);
    const T expe = exp(-ec / ga);
    const T af = be[parameter] / ga * (1.0 / (expe - 1.0));
    const T bf = expe * (vc - ec);
    const T y = af * t * t;
    const T xy = (1.0 + y) / (1.0 + y + y * y);
    const T x = 1.0 + y + y * y;
    const T qy = y * y * (2.0 + y) / (x * x);
    const T s1 = 1.0 + be[parameter] / ga * t * t * xy;
    const T h0 = ga * log(s1);
    const T dh0 = be[parameter] * t * t / s1
                  * (-7.0 / 3.0 * xy - qy * (af * bf / be[parameter] - 7.0 / 3.0));
    const T ddh0 = be[parameter] / (2.0 * ks * ks * rho) * (xy - qy) / s1;
    sc = rho * h0;
    v1c = h0 + dh0;
    v2c = ddh0;
}

template <typename T>
ABACUS_XC_HD void pbec_spin(const T rho,
                            const T zeta,
                            const T grho,
                            const int iflag,
                            T& sc,
                            T& v1cup,
                            T& v1cdw,
                            T& v2c)
{
    const T ga = 0.0310910;
    const T be[3] = {0.0, 0.06672455060314922, 0.0460000};
    const T third = 1.0 / 3.0;
    const T pi34 = 0.62035049089940;
    const T xkf = 1.9191582926775130;
    const T xks = 1.1283791670955130;
    const T rs = pi34 / pow(rho, third);
    T ec = 0.0;
    T vcup = 0.0;
    T vcdw = 0.0;
    pw_spin(rs, zeta, ec, vcup, vcdw);

    const T kf = xkf / rs;
    const T ks = xks * sqrt(kf);
    const T fz = 0.5 * (pow(1.0 + zeta, 2.0 / 3.0) + pow(1.0 - zeta, 2.0 / 3.0));
    const T fz2 = fz * fz;
    const T fz3 = fz2 * fz;
    const T dfz = (pow(1.0 + zeta, -1.0 / 3.0) - pow(1.0 - zeta, -1.0 / 3.0)) / 3.0;
    const T t = sqrt(grho) / (2.0 * fz * ks * rho);
    const T expe = exp(-ec / (fz3 * ga));
    const T af = be[iflag] / ga * (1.0 / (expe - 1.0));
    const T bfup = expe * (vcup - ec) / fz3;
    const T bfdw = expe * (vcdw - ec) / fz3;
    const T y = af * t * t;
    const T xy = (1.0 + y) / (1.0 + y + y * y);
    const T qy = y * y * (2.0 + y) / pow(1.0 + y + y * y, 2);
    const T s1 = 1.0 + be[iflag] / ga * t * t * xy;
    const T h0 = fz3 * ga * log(s1);
    const T dh0up = be[iflag] * t * t * fz3 / s1
                    * (-7.0 / 3.0 * xy - qy * (af * bfup / be[iflag] - 7.0 / 3.0));
    const T dh0dw = be[iflag] * t * t * fz3 / s1
                    * (-7.0 / 3.0 * xy - qy * (af * bfdw / be[iflag] - 7.0 / 3.0));
    const T dh0z = 3.0 * h0 / fz
                   - be[iflag] * t * t * fz2 / s1
                         * (2.0 * xy - qy * (3.0 * af * expe * ec / fz3 / be[iflag] + 2.0));
    const T dh0zup = dh0z * dfz * (1.0 - zeta);
    const T dh0zdw = -dh0z * dfz * (1.0 + zeta);
    const T ddh0 = be[iflag] * fz / (2.0 * ks * ks * rho) * (xy - qy) / s1;
    sc = rho * h0;
    v1cup = h0 + dh0up + dh0zup;
    v1cdw = h0 + dh0dw + dh0zdw;
    v2c = ddh0;
}

template <typename T>
ABACUS_XC_HD void scalar_pbe(const T rho, T& exc, T& vxc)
{
    const T pi34 = 0.62035049089940;
    const T rs = pi34 / pow(rho, 1.0 / 3.0);
    T ex = 0.0;
    T vx = 0.0;
    T ec = 0.0;
    T vc = 0.0;
    slater(rs, ex, vx);
    pw_interpolation(rs, ec, vc);
    exc = ex + ec;
    vxc = vx + vc;
}

template <typename T>
ABACUS_XC_HD void scalar_lda_spin(const T rho,
                                  const T zeta,
                                  const int correlation,
                                  T& exc,
                                  T& vup,
                                  T& vdw)
{
    const T pi34 = 0.62035049089940;
    const T rs = pi34 / pow(rho, 1.0 / 3.0);
    T ex = 0.0;
    T vxup = 0.0;
    T vxdw = 0.0;
    T ec = 0.0;
    T vcup = 0.0;
    T vcdw = 0.0;
    slater_spin(rho, zeta, ex, vxup, vxdw);
    if (correlation == 0)
    {
        pz_spin(rs, zeta, ec, vcup, vcdw);
    }
    else
    {
        pw_spin(rs, zeta, ec, vcup, vcdw);
    }
    exc = ex + ec;
    vup = vxup + vcup;
    vdw = vxdw + vcdw;
}

} // namespace xc_builtin
} // namespace hamilt

#undef ABACUS_XC_HD

#endif // SOURCE_HAMILT_MODULE_XC_KERNELS_XC_BUILTIN_FORMULA_H_

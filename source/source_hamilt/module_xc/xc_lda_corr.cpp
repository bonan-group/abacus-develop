// This file contains realization of LDA correlation functionals
// Spin unpolarized ones:
//  1. pw : Perdew-Wang LDA correlation
//  2. pz : Perdew-Zunger LDA correlation
//  3. lyp : Lee-Yang-Parr correlation
//  4. vwn : Vosko-Wilk-Nusair LDA correlation
//  5. wigner : Wigner
//  6. hl : Hedin-Lunqvist
//  7. gl : Gunnarson-Lunqvist
// And some of their spin polarized counterparts:
//  1. pw_spin
//  2. pz_spin, which calls pz_polarized

#include "xc_functional.h"
#include "source_hamilt/module_xc/kernels/xc_builtin_formula.h"

void XC_Functional::pw(
    const double &rs,
    const int &iflag,
    double &ec,
    double &vc)
{
    const double c0 = 0.0310910;
    const double c1 = 0.0466440;
    const double c2 = 0.006640;
    const double c3 = 0.010430;
    const double d0 = 0.43350;
    const double d1 = 1.44080;

    // Keep the CPU-only high- and low-density iflag==1 branches.
    if (rs < 1 && iflag == 1)
    {
        const double lnrs = log(rs);
        ec = c0 * lnrs - c1 + c2 * rs * lnrs - c3 * rs;
        vc = c0 * lnrs - (c1 + c0 / 3.0) + 2.0 / 3.0 * c2 * rs * lnrs
             - (2.0 * c3 + c2) / 3.0 * rs;
    }
    else if (rs > 100.0 && iflag == 1)
    {
        ec = -d0 / rs + d1 / pow(rs, 1.50);
        vc = -4.0 / 3.0 * d0 / rs + 1.50 * d1 / pow(rs, 1.50);
    }
    else
    {
        hamilt::xc_builtin::pw_interpolation(rs, iflag, ec, vc);
    }
}

//LDA parameterization form Monte Carlo data
//iflag=0: J.P. Perdew and A. Zunger, PRB 23, 5048 (1981)
//iflag=1: G. Ortiz and P. Ballone, PRB 50, 1391 (1994)
void XC_Functional::pz(
    const double &rs,
    const int &iflag,
    double &ec,
    double &vc)
{
    hamilt::xc_builtin::pz(rs, iflag, ec, vc);
}

// C. Lee, W. Yang, and R.G. Parr, PRB 37, 785 (1988)
// LDA part only
void XC_Functional::lyp(
    const double &rs,
    double &ec,
    double &vc)
{
    const double a = 0.04918e0;
    const double b = 0.1320 * 2.87123400018819108e0;
    // pi43 = (4pi/3)^(1/3)
    const double pi43 = 1.61199195401647e0;
    const double c = 0.2533e0 * pi43;
    const double d = 0.349e0 * pi43;
    double ecrs = 0.0;
    double ox = 0.0;

    ecrs = b * exp(- c * rs);
    ox = 1.0 / (1.0 + d * rs);
    ec = - a * ox * (1.0 + ecrs);
    vc = ec - rs / 3.0 * a * ox * (d * ox + ecrs * (d * ox + c));

    return;
}

// S.H. Vosko, L. Wilk, and M. Nusair, Can. J. Phys. 58, 1200 (1980)
void XC_Functional::vwn(
    const double &rs,
    double &ec,
    double &vc)
{
    const double a = 0.0310907;
    const double b = 3.72744;
    const double c = 12.9352;
    const double x0 = -0.10498;
    double q = 0.0;
    double f1 = 0.0;
    double f2 = 0.0;
    double f3 = 0.0;
    double rs12 = 0.0;
    double fx = 0.0;
    double qx = 0.0;
    double tx = 0.0;
    double tt = 0.0;

    q = sqrt(4.0 * c - b * b);
    f1 = 2.0 * b / q;
    f2 = b * x0 / (x0 * x0 + b * x0 + c);
    f3 = 2.0 * (2.0 * x0 + b) / q;
    rs12 = sqrt(rs);
    fx = rs + b * rs12 + c;
    qx = atan(q / (2.0 * rs12 + b));

    double x = 0.0;
    x = (rs12 - x0);
    ec = a * (log(rs / fx) + f1 * qx - f2 * (log((x * x) /
              fx) + f3 * qx));
    tx = 2.0 * rs12 + b;
    tt = tx * tx + q * q;

    vc = ec - rs12 * a / 6.0 * (2.0 / rs12 - tx / fx - 4.0 * b /
        tt - f2 * (2.0 / (rs12 - x0) - tx / fx - 4.0 * (2.0 * x0 + b) / tt));

    return;
}

void XC_Functional::wigner(
    const double &rs,
    double &ec,
    double &vc)
{
    const double pi34 = 0.6203504908994e0;
    // pi34=(3/4pi)^(1/3), rho13=rho^(1/3)

    double rho13 = pi34 / rs;
    double x = 1.0 + 12.570 * rho13;
    vc = - rho13 * ((0.9436560 + 8.89630 * rho13) / (x * x));
    ec = - 0.7380 * rho13 * (0.9590 / (1.0 + 12.570 * rho13));
    return;
}

// L. Hedin and  B.I. Lundqvist,  J. Phys. C 4, 2064 (1971)
void XC_Functional::hl(
    const double &rs,
    double &ec,
    double &vc)
{
    double a = 0.0;
    double x = 0.0;
    a = log(1.00 + 21.0 / rs);
    x = rs / 21.00;
    ec = a + (pow(x, 3) * a - x * x) + x / 2.0 - 1.00 / 3.00;
    ec = - 0.02250 * ec;
    vc = - 0.02250 * a;
    return;
}

// O. Gunnarsson and B. I. Lundqvist, PRB 13, 4274 (1976)
void XC_Functional::gl(
    const double &rs,
    double &ec,
    double &vc)
{
    const double c = 0.0333;
    const double r = 11.4;
    // c=0.0203, r=15.9 for the paramagnetic case
    double x = rs / r;
    vc = - c * log(1.0 + 1.0 / x);
    ec = - c * ((1.0 + pow(x, 3)) * log(1.0 + 1.0 / x) - 1.00 /
                3.00 + x * (0.50 - x));
    return;
}

// J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
void XC_Functional::pw_spin(
    const double &rs,
    const double &zeta,
    double &ec,
    double &vcup,
    double &vcdw)
{
    hamilt::xc_builtin::pw_spin(rs, zeta, ec, vcup, vcdw);
}

// J.P. Perdew and Y. Wang, PRB 45, 13244 (1992)
void XC_Functional::pz_spin(
    const double &rs,
    const double &zeta,
    double &ec,
    double &vcup,
    double &vcdw)
{
    hamilt::xc_builtin::pz_spin(rs, zeta, ec, vcup, vcdw);
}

// J.P. Perdew and A. Zunger, PRB 23, 5048 (1981)
void XC_Functional::pz_polarized(
    const double &rs,
    double &ec,
    double &vc)
{
    hamilt::xc_builtin::pz_polarized(rs, ec, vc);
}

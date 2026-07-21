// This file contains interface to xc_functional class:
// 1. v_xc : which takes rho as input, and v_xc as output
// 2. v_xc_libxc : which does the same thing as v_xc, but calling libxc
// NOTE : it is only used for nspin = 1 and 2, the nspin = 4 case is treated in v_xc
// 3. v_xc_meta : which takes rho and tau as input, and v_xc as output

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_resident_gpu.h"
#include "source_io/module_parameter/parameter.h"
#include "xc_functional.h"

#include <vector>

#ifdef USE_LIBXC
#include "libxc_abacus.h"
#ifdef __EXX
#include "source_hamilt/module_xc/exx_info.h"
#endif
#endif

// [etxc, vtxc, v] = XC_Functional::v_xc(...)
std::tuple<double, double, ModuleBase::matrix> XC_Functional::v_xc(
    const int& nrxx,
    const Charge* const chr,
    const UnitCell* ucell)
{
    const double hybrid_alpha = XC_Functional::get_hybrid_alpha();
#ifdef __EXX
    const double hse_omega = XC_Functional::get_hse_omega();
#else
    const double hse_omega = 0.0;
#endif
    return XC_Functional::v_xc(nrxx,
                               chr,
                               ucell,
                               "cpu",
                               PARAM.inp.nspin,
                               PARAM.globalv.domag,
                               PARAM.globalv.domag_z,
                               hybrid_alpha,
                               hse_omega);
}

std::tuple<double, double, ModuleBase::matrix> XC_Functional::v_xc(
    const int& nrxx,
    const Charge* const chr,
    const UnitCell* ucell,
    const int nspin,
    const bool domag,
    const bool domag_z,
    const double hybrid_alpha,
    const double hse_omega)
{
    return XC_Functional::v_xc(
        nrxx, chr, ucell, "cpu", nspin, domag, domag_z, hybrid_alpha, hse_omega);
}

std::tuple<double, double, ModuleBase::matrix> XC_Functional::v_xc(
    const int& nrxx,
    const Charge* const chr,
    const UnitCell* ucell,
    const std::string& device,
    const int nspin,
    const bool domag,
    const bool domag_z,
    const double hybrid_alpha,
    const double hse_omega)
{
    ModuleBase::TITLE("XC_Functional", "v_xc");

    if (use_libxc)
    {
#ifdef USE_LIBXC
        return XC_Functional_Libxc::v_xc_libxc(XC_Functional::get_func_id(),
                                               nrxx,
                                               ucell->omega,
                                               ucell->tpiba,
                                               chr,
                                               nspin,
                                               domag,
                                               domag_z,
                                               &(scaling_factor_xc),
                                               hybrid_alpha,
                                               hse_omega);
#else
        ModuleBase::WARNING_QUIT("v_xc", "compile with LIBXC");
#endif
    }

    ModuleBase::timer::start("XC_Functional", "v_xc");

    //Exchange-Correlation potential Vxc(r) from n(r)
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(nspin, nrxx);

    // the square of the e charge
    // in Rydeberg unit, so * 2.0.
    double e2 = 2.0;
    double vanishing_charge = 1.0e-10;

    XC_Functional_GPU::XcGpuRequest gpu_request;
    gpu_request.device = device;
    gpu_request.nrxx = nrxx;
    gpu_request.nspin = nspin;
    gpu_request.use_libxc = use_libxc;
    gpu_request.has_kinetic_energy_density = XC_Functional::get_ked_flag();
    gpu_request.functional_ids = &func_id;
    gpu_request.charge = chr;
    gpu_request.rho_basis = chr == nullptr ? nullptr : chr->rhopw;
    gpu_request.unit_cell = ucell;
    gpu_request.rho_up = chr == nullptr ? nullptr : chr->get_rho_d(0);
    gpu_request.rho_down = chr == nullptr || nspin != 2 ? nullptr : chr->get_rho_d(1);
    gpu_request.host_potential = &v;
    const XC_Functional_GPU::XcGpuResult gpu_result = XC_Functional_GPU::evaluate_resident_xc(gpu_request);
    if (gpu_result.used)
    {
        etxc = gpu_result.energy;
        vtxc = gpu_result.potential_sum;
#ifdef __MPI
        Parallel_Reduce::reduce_pool(etxc);
        Parallel_Reduce::reduce_pool(vtxc);
#endif
        etxc *= ucell->omega / chr->rhopw->nxyz;
        vtxc *= ucell->omega / chr->rhopw->nxyz;

        ModuleBase::timer::end("XC_Functional", "v_xc");
        return std::make_tuple(etxc, vtxc, std::move(v));
    }

    ModuleBase::timer::start("XC_Functional", "xc_builtin_eval");
    if (nspin == 1 || ( nspin ==4 && !domag && !domag_z))
    {
        // spin-unpolarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            // total electron charge density
            double rhox = chr->rho[0][ir] + chr->rho_core[ir];
            double arhox = std::abs(rhox);
            if (arhox > vanishing_charge)
            {
                double exc = 0.0;
                double vxc = 0.0;
                XC_Functional::xc(arhox, exc, vxc);
                v(0,ir) = e2 * vxc;
                // consider the total charge density
                etxc += e2 * exc * rhox;
                // only consider chr->rho
                vtxc += v(0, ir) * chr->rho[0][ir];
            } // endif
        } //enddo
    }
    else if(nspin ==2)
    {
        // spin-polarized case
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for (int ir = 0;ir < nrxx;ir++)
        {
            double rhox = chr->rho[0][ir] + chr->rho[1][ir] + chr->rho_core[ir];
            double arhox = std::abs(rhox);

            if (arhox > vanishing_charge)
            {
                double zeta = (chr->rho[0][ir] - chr->rho[1][ir]) / arhox;

                if (std::abs(zeta)  > 1.0)
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }

                double rhoup = arhox * (1.0+zeta) / 2.0;
                double rhodw = arhox * (1.0-zeta) / 2.0;
                double exc = 0.0;
                double vxc[2];
                XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);

                for (int is = 0;is < nspin;is++)
                {
                    v(is, ir) = e2 * vxc[is];
                }

                etxc += e2 * exc * rhox;
                vtxc += v(0, ir) * chr->rho[0][ir] + v(1, ir) * chr->rho[1][ir];
            }
        }
    }
    else if(nspin == 4)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:etxc) reduction(+:vtxc)
#endif
        for(int ir = 0;ir<nrxx; ir++)
        {
            double amag = sqrt( pow(chr->rho[1][ir],2) + pow(chr->rho[2][ir],2) + pow(chr->rho[3][ir],2) );
            double rhox = chr->rho[0][ir] + chr->rho_core[ir];
            double arhox = std::abs( rhox );

            if ( arhox > vanishing_charge )
            {
                double zeta = amag / arhox;
                double exc = 0.0;
                double vxc[2];

                if ( std::abs( zeta ) > 1.0 )
                {
                    zeta = (zeta > 0.0) ? 1.0 : (-1.0);
                }

                if(use_libxc)
                {
#ifdef USE_LIBXC
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional_Libxc::xc_spin_libxc(XC_Functional::get_func_id(), rhoup, rhodw, exc, vxc[0], vxc[1], hybrid_alpha, hse_omega);
#else
                    ModuleBase::WARNING_QUIT("v_xc", "compile with LIBXC");
#endif
                }
                else
                {
                    double rhoup = arhox * (1.0+zeta) / 2.0;
                    double rhodw = arhox * (1.0-zeta) / 2.0;
                    XC_Functional::xc_spin(arhox, zeta, exc, vxc[0], vxc[1]);
                }

                etxc += e2 * exc * rhox;

                v(0, ir) = e2*( 0.5 * ( vxc[0] + vxc[1]) );
                vtxc += v(0,ir) * chr->rho[0][ir];

                double vs = 0.5 * ( vxc[0] - vxc[1] );
                if ( amag > vanishing_charge )
                {
                    for(int ipol = 1;ipol< 4;ipol++)
                    {
                        v(ipol, ir) = e2 * vs * chr->rho[ipol][ir] / amag;
                        vtxc += v(ipol,ir) * chr->rho[ipol][ir];
                    }
                }
            }
        }
    }
    ModuleBase::timer::end("XC_Functional", "xc_builtin_eval");
    // energy terms, local-density contributions

    // add gradient corrections (if any)
    // mohan modify 2009-12-15

    // the dummy variable dum contains gradient correction to stress
    // which is not used here
    std::vector<double> dum;
    ModuleBase::timer::start("XC_Functional", "gradcorr");
    gradcorr(etxc,
             vtxc,
             v,
             chr,
             chr->rhopw,
             ucell,
             dum,
             false,
             nspin,
             domag,
             domag_z,
             hybrid_alpha,
             hse_omega,
             device);
    ModuleBase::timer::end("XC_Functional", "gradcorr");

    // parallel code : collect vtxc,etxc
    // mohan add 2008-06-01
#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc);
    Parallel_Reduce::reduce_pool(vtxc);
#endif
    etxc *= ucell->omega / chr->rhopw->nxyz;
    vtxc *= ucell->omega / chr->rhopw->nxyz;

    ModuleBase::timer::end("XC_Functional", "v_xc");
    return std::make_tuple(etxc, vtxc, std::move(v));
}

bool XC_Functional::add_v_xc_to_device(const int& nrxx,
                                       const Charge* const chr,
                                       const UnitCell* ucell,
                                       const std::string& device,
                                       const int nspin,
                                       double* d_v_eff,
                                       double& etxc,
                                       double& vtxc)
{
    etxc = 0.0;
    vtxc = 0.0;
    if (d_v_eff == nullptr)
    {
        return false;
    }

    ModuleBase::timer::start("XC_Functional", "v_xc_device_add");
    XC_Functional_GPU::XcGpuRequest request;
    request.device = device;
    request.nrxx = nrxx;
    request.nspin = nspin;
    request.use_libxc = use_libxc;
    request.has_kinetic_energy_density = XC_Functional::get_ked_flag();
    request.functional_ids = &func_id;
    request.charge = chr;
    request.rho_basis = chr == nullptr ? nullptr : chr->rhopw;
    request.unit_cell = ucell;
    request.rho_up = chr == nullptr ? nullptr : chr->get_rho_d(0);
    request.rho_down = chr == nullptr || nspin != 2 ? nullptr : chr->get_rho_d(1);
    request.device_potential = d_v_eff;
    const XC_Functional_GPU::XcGpuResult result = XC_Functional_GPU::evaluate_resident_xc(request);
    if (result.used)
    {
        etxc = result.energy;
        vtxc = result.potential_sum;
#ifdef __MPI
        Parallel_Reduce::reduce_pool(etxc);
        Parallel_Reduce::reduce_pool(vtxc);
#endif
        etxc *= ucell->omega / chr->rhopw->nxyz;
        vtxc *= ucell->omega / chr->rhopw->nxyz;
    }
    ModuleBase::timer::end("XC_Functional", "v_xc_device_add");
    return result.used;
}

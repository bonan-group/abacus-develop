// This file contains interface to xc_functional class:
// 1. v_xc : which takes rho as input, and v_xc as output
// 2. v_xc_libxc : which does the same thing as v_xc, but calling libxc
// NOTE : it is only used for nspin = 1 and 2, the nspin = 4 case is treated in v_xc
// 3. v_xc_meta : which takes rho and tau as input, and v_xc as output

#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_base/module_device/memory_op.h"
#include "source_hamilt/module_xc/kernels/xc_gradcorr_op.h"
#include "source_io/module_parameter/parameter.h"
#include "xc_functional.h"

#include <complex>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef USE_LIBXC
#include "libxc_abacus.h"
#endif

namespace
{

#if __CUDA || __UT_USE_CUDA
bool try_v_xc_pbe_resident_gpu(const int nrxx,
                               const Charge* const chr,
                               const UnitCell* const ucell,
                               const std::string& device,
                               const std::vector<int>& func_id,
                               ModuleBase::matrix& v,
                               double& etxc,
                               double& vtxc)
{
    const char* xc_gpu_env = std::getenv("ABACUS_XC_GPU");
    const bool xc_gpu_enabled = xc_gpu_env != nullptr && std::string(xc_gpu_env) == "1";
    const bool is_pbe = func_id.size() == 2 && func_id[0] == XC_GGA_X_PBE && func_id[1] == XC_GGA_C_PBE;
    const bool is_pbesol = func_id.size() == 2 && func_id[0] == XC_GGA_X_PBE_SOL && func_id[1] == XC_GGA_C_PBE_SOL;
    ModulePW::PW_Basis* rhopw = chr != nullptr ? chr->rhopw : nullptr;
    if (!xc_gpu_enabled || device != "gpu" || PARAM.inp.nspin != 1 || !(is_pbe || is_pbesol) || chr == nullptr
        || ucell == nullptr || rhopw == nullptr || chr->get_device() != "gpu" || rhopw->get_device() != "gpu"
        || rhopw->poolnproc != 1 || rhopw->nrxx != nrxx || chr->get_rho_d(0) == nullptr)
    {
        return false;
    }

    using complex_t = std::complex<double>;
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using resmem_complex_op = base_device::memory::resize_memory_op<complex_t, base_device::DEVICE_GPU>;
    using delmem_complex_op = base_device::memory::delete_memory_op<complex_t, base_device::DEVICE_GPU>;
    using syncmem_double_h2d_op
        = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_double_d2h_op
        = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
    using setmem_double_op = base_device::memory::set_memory_op<double, base_device::DEVICE_GPU>;

    double* d_rho_core = nullptr;
    double* d_rho_total = nullptr;
    double* d_v = nullptr;
    double* d_sums = nullptr;
    double* d_gcar = nullptr;
    double* d_gdr = nullptr;
    double* d_grad_r = nullptr;
    double* d_h = nullptr;
    double* d_h_comp = nullptr;
    double* d_dh = nullptr;
    double* d_dh_sum = nullptr;
    complex_t* d_rhog_total = nullptr;
    complex_t* d_grad_g = nullptr;
    complex_t* d_aux_g = nullptr;
    complex_t* d_gaux = nullptr;

    const auto cleanup = [&]() {
        delmem_double_op()(d_rho_core);
        delmem_double_op()(d_rho_total);
        delmem_double_op()(d_v);
        delmem_double_op()(d_sums);
        delmem_double_op()(d_gcar);
        delmem_double_op()(d_gdr);
        delmem_double_op()(d_grad_r);
        delmem_double_op()(d_h);
        delmem_double_op()(d_h_comp);
        delmem_double_op()(d_dh);
        delmem_double_op()(d_dh_sum);
        delmem_complex_op()(d_rhog_total);
        delmem_complex_op()(d_grad_g);
        delmem_complex_op()(d_aux_g);
        delmem_complex_op()(d_gaux);
    };

    ModuleBase::timer::start("XC_Functional", "v_xc_resident_gpu");

    const int iflag = is_pbesol ? 2 : 0;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    const int npw = rhopw->npw;

    std::vector<double> gcar_flat(3 * npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        gcar_flat[3 * ig + 0] = rhopw->gcar[ig].x;
        gcar_flat[3 * ig + 1] = rhopw->gcar[ig].y;
        gcar_flat[3 * ig + 2] = rhopw->gcar[ig].z;
    }

    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_rho_total, nrxx);
    resmem_double_op()(d_v, nrxx);
    resmem_double_op()(d_sums, 2);
    resmem_double_op()(d_gcar, 3 * npw);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_grad_r, nrxx);
    resmem_double_op()(d_h, 3 * nrxx);
    resmem_double_op()(d_h_comp, nrxx);
    resmem_double_op()(d_dh, nrxx);
    resmem_double_op()(d_dh_sum, 1);
    resmem_complex_op()(d_rhog_total, npw);
    resmem_complex_op()(d_grad_g, npw);
    resmem_complex_op()(d_aux_g, npw);
    resmem_complex_op()(d_gaux, npw);

    syncmem_double_h2d_op()(d_rho_core, chr->rho_core, nrxx);
    syncmem_double_h2d_op()(d_gcar, gcar_flat.data(), 3 * npw);

    ModuleBase::timer::start("XC_Functional", "v_xc_resident_scalar");
    hamilt::xc_scalar_pbe_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                nrxx,
                                                                e2,
                                                                1.0e-10,
                                                                chr->get_rho_d(0),
                                                                d_rho_core,
                                                                d_rho_total,
                                                                d_v,
                                                                d_sums,
                                                                &etxc,
                                                                &vtxc);
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_scalar");

    ModuleBase::timer::start("XC_Functional", "v_xc_resident_grad_rho");
    rhopw->real_to_recip<double, complex_t, base_device::DEVICE_GPU>(d_rho_total, d_rhog_total);
    for (int ipol = 0; ipol < 3; ++ipol)
    {
        hamilt::xc_multiply_iG_op<double, base_device::DEVICE_GPU>()(
            nullptr, npw, ipol, d_gcar, d_rhog_total, d_grad_g);
        setmem_double_op()(d_grad_r, 0, nrxx);
        rhopw->recip_to_real<complex_t, double, base_device::DEVICE_GPU>(d_grad_g, d_grad_r, true, ucell->tpiba);
        hamilt::xc_set_component_op<double, base_device::DEVICE_GPU>()(nullptr, nrxx, ipol, d_grad_r, d_gdr);
    }
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_grad_rho");

    double etxcgc = 0.0;
    double vtxcgc = 0.0;
    ModuleBase::timer::start("XC_Functional", "v_xc_resident_eval_grid");
    hamilt::xc_gradcorr_pbe_grid_resident_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                                nrxx,
                                                                                iflag,
                                                                                e2,
                                                                                epsr,
                                                                                d_rho_total,
                                                                                d_rho_core,
                                                                                d_gdr,
                                                                                d_v,
                                                                                d_h,
                                                                                d_sums,
                                                                                &etxcgc,
                                                                                &vtxcgc);
    etxc += etxcgc;
    vtxc += vtxcgc;
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_eval_grid");

    ModuleBase::timer::start("XC_Functional", "v_xc_resident_grad_dot");
    for (int ipol = 0; ipol < 3; ++ipol)
    {
        hamilt::xc_extract_component_op<double, base_device::DEVICE_GPU>()(nullptr, nrxx, ipol, d_h, d_h_comp);
        rhopw->real_to_recip<double, complex_t, base_device::DEVICE_GPU>(d_h_comp, d_aux_g);
        hamilt::xc_accumulate_iG_op<double, base_device::DEVICE_GPU>()(
            nullptr, npw, ipol, d_gcar, d_aux_g, d_gaux, ipol == 0);
    }
    setmem_double_op()(d_dh, 0, nrxx);
    rhopw->recip_to_real<complex_t, double, base_device::DEVICE_GPU>(d_gaux, d_dh, true, ucell->tpiba);
    double vtxc_delta = 0.0;
    hamilt::xc_apply_dh_op<double, base_device::DEVICE_GPU>()(
        nullptr, nrxx, d_rho_total, d_rho_core, d_dh, d_v, d_dh_sum, &vtxc_delta);
    vtxc += vtxc_delta;
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_grad_dot");

    ModuleBase::timer::start("XC_Functional", "v_xc_resident_d2h");
    syncmem_double_d2h_op()(v.c, d_v, nrxx);
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_d2h");

    cleanup();
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_gpu");
    return true;
}
#endif

} // namespace

// [etxc, vtxc, v] = XC_Functional::v_xc(...)
std::tuple<double, double, ModuleBase::matrix> XC_Functional::v_xc(
    const int& nrxx,
    const Charge* const chr,
    const UnitCell* ucell)
{
    return XC_Functional::v_xc(nrxx, chr, ucell, "cpu");
}

std::tuple<double, double, ModuleBase::matrix> XC_Functional::v_xc(
    const int& nrxx,
    const Charge* const chr,
    const UnitCell* ucell,
    const std::string& device)
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
                                               &(scaling_factor_xc));
#else
        ModuleBase::WARNING_QUIT("v_xc", "compile with LIBXC");
#endif
    }

    ModuleBase::timer::start("XC_Functional", "v_xc");

    //Exchange-Correlation potential Vxc(r) from n(r)
    double etxc = 0.0;
    double vtxc = 0.0;
    ModuleBase::matrix v(PARAM.inp.nspin, nrxx);

    // the square of the e charge
    // in Rydeberg unit, so * 2.0.
    double e2 = 2.0;
    double vanishing_charge = 1.0e-10;

#if __CUDA || __UT_USE_CUDA
    if (try_v_xc_pbe_resident_gpu(nrxx, chr, ucell, device, func_id, v, etxc, vtxc))
    {
#ifdef __MPI
        Parallel_Reduce::reduce_pool(etxc);
        Parallel_Reduce::reduce_pool(vtxc);
#endif
        etxc *= ucell->omega / chr->rhopw->nxyz;
        vtxc *= ucell->omega / chr->rhopw->nxyz;

        ModuleBase::timer::end("XC_Functional", "v_xc");
        return std::make_tuple(etxc, vtxc, std::move(v));
    }
#endif

    ModuleBase::timer::start("XC_Functional", "xc_builtin_eval");
    if (PARAM.inp.nspin == 1 || ( PARAM.inp.nspin ==4 && !PARAM.globalv.domag && !PARAM.globalv.domag_z))
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
    else if(PARAM.inp.nspin ==2)
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

                for (int is = 0;is < PARAM.inp.nspin;is++)
                {
                    v(is, ir) = e2 * vxc[is];
                }

                etxc += e2 * exc * rhox;
                vtxc += v(0, ir) * chr->rho[0][ir] + v(1, ir) * chr->rho[1][ir];
            }
        }
    }
    else if(PARAM.inp.nspin == 4)
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
                    XC_Functional_Libxc::xc_spin_libxc(XC_Functional::get_func_id(), rhoup, rhodw, exc, vxc[0], vxc[1]);
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
    gradcorr(etxc, vtxc, v, chr, chr->rhopw, ucell, dum, false, device);
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

// This file contains subroutines realted to gradient calculations
// it contains 5 subroutines:
// 1. gradcorr, which calculates gradient correction
// 2. grad_wfc, which calculates gradient of wavefunction
//		it is used in stress_func_mgga.cpp
// 3. grad_rho, which calculates gradient of density
// 4. grad_dot, which calculates divergence of something
// 5. noncolin_rho, which diagonalizes the spin density matrix
//  and gives the spin up and spin down components of the charge.

#include "xc_functional.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_io/module_parameter/parameter.h"
#include <ATen/core/tensor.h>
#include <ATen/core/tensor_map.h>
#include <ATen/core/tensor_types.h>
#include <source_hamilt/module_xc/kernels/xc_functional_op.h>
#include <source_hamilt/module_xc/kernels/xc_gradcorr_op.h>
#include "source_hamilt/module_xc/xc_gpu_policy.h"
#include "source_base/module_device/memory_op.h"

#include <cstdlib>
#include <string>
#include <vector>

#ifdef USE_LIBXC
#include "libxc_abacus.h"
#ifdef __EXX
#include "source_hamilt/module_xc/exx_info.h"
#endif
#endif

bool XC_Functional::gradcorr_stress_gpu(const Charge* const chr,
                                        ModulePW::PW_Basis* rhopw,
                                        const UnitCell* ucell,
                                        std::vector<double>& stress_gga,
                                        const std::string& device)
{
#if __CUDA || __UT_USE_CUDA
    const char* xc_gpu_env = std::getenv("ABACUS_XC_GPU");
    const bool xc_gpu_enabled = xc_gpu_env != nullptr && std::string(xc_gpu_env) == "1";
    const bool is_pbe = func_id.size() == 2 && func_id[0] == XC_GGA_X_PBE && func_id[1] == XC_GGA_C_PBE;
    const bool is_pbesol = func_id.size() == 2 && func_id[0] == XC_GGA_X_PBE_SOL && func_id[1] == XC_GGA_C_PBE_SOL;
    const std::string xc_name = is_pbesol ? "PBEsol" : "PBE";
    const int nspin = PARAM.inp.nspin;
    if (!XC_Functional_GPU::xc_gpu_stress_policy(device == "gpu", !xc_gpu_enabled, nspin, xc_name)
        || use_libxc || func_type != 2 || !(is_pbe || is_pbesol)
        || chr == nullptr || rhopw == nullptr || ucell == nullptr || chr->get_device() != "gpu"
        || rhopw->get_device() != "gpu" || rhopw->poolnproc != 1 || !(nspin == 1 || nspin == 2)
        || chr->get_rho_d(0) == nullptr || (nspin == 2 && chr->get_rho_d(1) == nullptr))
    {
        return false;
    }

    chr->sync_realspace_density_to_device();

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

    const int nrxx = rhopw->nrxx;
    const int npw = rhopw->npw;
    const int iflag = is_pbesol ? 2 : 0;
    const double e2 = ModuleBase::e2;
    const double epsr = 1.0e-6;

    double* d_rho_core = nullptr;
    double* d_rho_total = nullptr;
    double* d_rho_dw_total = nullptr;
    double* d_gcar = nullptr;
    double* d_gdr = nullptr;
    double* d_gdr_dw = nullptr;
    double* d_grad_r = nullptr;
    double* d_dummy_v = nullptr;
    double* d_sums = nullptr;
    double* d_stress = nullptr;
    complex_t* d_rhog_total = nullptr;
    complex_t* d_grad_g = nullptr;

    const auto cleanup = [&]() {
        delmem_double_op()(d_rho_core);
        delmem_double_op()(d_rho_total);
        delmem_double_op()(d_rho_dw_total);
        delmem_double_op()(d_gcar);
        delmem_double_op()(d_gdr);
        delmem_double_op()(d_gdr_dw);
        delmem_double_op()(d_grad_r);
        delmem_double_op()(d_dummy_v);
        delmem_double_op()(d_sums);
        delmem_double_op()(d_stress);
        delmem_complex_op()(d_rhog_total);
        delmem_complex_op()(d_grad_g);
    };

    ModuleBase::timer::start("XC_Functional", "gradcorr_stress_gpu");

    std::vector<double> gcar_flat(3 * npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        gcar_flat[3 * ig + 0] = rhopw->gcar[ig].x;
        gcar_flat[3 * ig + 1] = rhopw->gcar[ig].y;
        gcar_flat[3 * ig + 2] = rhopw->gcar[ig].z;
    }

    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_rho_total, nrxx);
    resmem_double_op()(d_gcar, 3 * npw);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_grad_r, nrxx);
    resmem_double_op()(d_stress, 9);
    resmem_complex_op()(d_rhog_total, npw);
    resmem_complex_op()(d_grad_g, npw);
    syncmem_double_h2d_op()(d_rho_core, chr->rho_core, nrxx);
    syncmem_double_h2d_op()(d_gcar, gcar_flat.data(), 3 * npw);

    const auto build_grad = [&](double* d_total, double* d_gdr_out) {
        rhopw->real_to_recip<double, complex_t, base_device::DEVICE_GPU>(d_total, d_rhog_total);
        for (int ipol = 0; ipol < 3; ++ipol)
        {
            hamilt::xc_multiply_iG_op<double, base_device::DEVICE_GPU>()(
                nullptr, npw, ipol, d_gcar, d_rhog_total, d_grad_g);
            setmem_double_op()(d_grad_r, 0, nrxx);
            rhopw->recip_to_real<complex_t, double, base_device::DEVICE_GPU>(d_grad_g, d_grad_r, true, ucell->tpiba);
            hamilt::xc_set_component_op<double, base_device::DEVICE_GPU>()(nullptr, nrxx, ipol, d_grad_r, d_gdr_out);
        }
    };

    if (nspin == 1)
    {
        double etxc_dummy = 0.0;
        double vtxc_dummy = 0.0;
        resmem_double_op()(d_dummy_v, nrxx);
        resmem_double_op()(d_sums, 2);
        hamilt::xc_scalar_pbe_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                    nrxx,
                                                                    e2,
                                                                    1.0e-10,
                                                                    chr->get_rho_d(0),
                                                                    d_rho_core,
                                                                    d_rho_total,
                                                                    d_dummy_v,
                                                                    d_sums,
                                                                    &etxc_dummy,
                                                                    &vtxc_dummy);
        build_grad(d_rho_total, d_gdr);
        hamilt::xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_GPU>()(
            nullptr, nrxx, iflag, e2, epsr, d_rho_total, d_gdr, d_stress);
    }
    else
    {
        double etxc_dummy = 0.0;
        double vtxc_dummy = 0.0;
        resmem_double_op()(d_rho_dw_total, nrxx);
        resmem_double_op()(d_gdr_dw, 3 * nrxx);
        resmem_double_op()(d_dummy_v, 2 * nrxx);
        resmem_double_op()(d_sums, 2);
        hamilt::xc_scalar_lda_spin_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                         nrxx,
                                                                         1,
                                                                         e2,
                                                                         1.0e-10,
                                                                         chr->get_rho_d(0),
                                                                         chr->get_rho_d(1),
                                                                         d_rho_core,
                                                                         d_rho_total,
                                                                         d_rho_dw_total,
                                                                         d_dummy_v,
                                                                         d_sums,
                                                                         &etxc_dummy,
                                                                         &vtxc_dummy);
        build_grad(d_rho_total, d_gdr);
        build_grad(d_rho_dw_total, d_gdr_dw);
        hamilt::xc_gradcorr_pbe_spin_stress_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                                  nrxx,
                                                                                  iflag,
                                                                                  e2,
                                                                                  epsr,
                                                                                  d_rho_total,
                                                                                  d_rho_dw_total,
                                                                                  d_gdr,
                                                                                  d_gdr_dw,
                                                                                  d_stress);
    }

    stress_gga.assign(9, 0.0);
    syncmem_double_d2h_op()(stress_gga.data(), d_stress, 9);
    cleanup();
    ModuleBase::timer::end("XC_Functional", "gradcorr_stress_gpu");
    return true;
#else
    return false;
#endif
}

void XC_Functional::gradcorr(
    double &etxc,
    double &vtxc,
    ModuleBase::matrix &v,
    const Charge* const chr,
    ModulePW::PW_Basis* rhopw,
    const UnitCell *ucell,
    std::vector<double> &stress_gga,
    const bool is_stress,
    const int nspin,
    const bool domag,
    const bool domag_z)
{
    XC_Functional::gradcorr(etxc, vtxc, v, chr, rhopw, ucell, stress_gga, is_stress, nspin, domag, domag_z, "cpu");
}

void XC_Functional::gradcorr(
    double &etxc,
    double &vtxc,
    ModuleBase::matrix &v,
    const Charge* const chr,
    ModulePW::PW_Basis* rhopw,
    const UnitCell *ucell,
    std::vector<double> &stress_gga,
    const bool is_stress,
    const std::string& device)
{
    XC_Functional::gradcorr(etxc,
                            vtxc,
                            v,
                            chr,
                            rhopw,
                            ucell,
                            stress_gga,
                            is_stress,
                            PARAM.inp.nspin,
                            PARAM.globalv.domag,
                            PARAM.globalv.domag_z,
                            device);
}

void XC_Functional::gradcorr(
    double &etxc,
    double &vtxc,
    ModuleBase::matrix &v,
    const Charge* const chr,
    ModulePW::PW_Basis* rhopw,
    const UnitCell *ucell,
    std::vector<double> &stress_gga,
    const bool is_stress,
    const int nspin,
    const bool domag,
    const bool domag_z,
    const std::string& device)
{
    ModuleBase::TITLE("XC_Functional","gradcorr");

    if((func_type == 3 || func_type == 5) && nspin==4)
    {
        ModuleBase::WARNING_QUIT("gradcorr","meta-GGA has not been implemented for nspin = 4 yet");
    }

    if(func_type == 0 || func_type == 1)
    {
        return;
    }

    bool igcc_is_lyp = false;
    if( func_id[1] == XC_GGA_C_LYP)
    {
        igcc_is_lyp = true;
    }

    int nspin0 = nspin;
    if(nspin==4)
    {
        nspin0 =1;
    }
    if(nspin==4&&(domag||domag_z))
    {
        nspin0 = 2;
    }

    assert(nspin0>0);
    const double fac = 1.0/ nspin0;

    if(is_stress)
    {
        stress_gga.resize(9);
        for(int i=0;i<9;i++)
        {
            stress_gga[i] = 0.0;
        }
    }

    // doing FFT to get rho in G space: rhog1
    ModuleBase::timer::start("XC_Functional", "gradcorr_rho_fft");
    rhopw->real2recip(chr->rho[0], chr->rhog[0]);
    if(nspin==2)
    {
        rhopw->real2recip(chr->rho[1], chr->rhog[1]);
    }
    rhopw->real2recip(chr->rho_core, chr->rhog_core);
    ModuleBase::timer::end("XC_Functional", "gradcorr_rho_fft");

    // sum up (rho_core+rho) for each spin in real space
    // and reciprocal space.
    double* rhotmp1 = nullptr;
    double* rhotmp2 = nullptr;
    std::complex<double>* rhogsum1 = nullptr;
    std::complex<double>* rhogsum2 = nullptr;
    ModuleBase::Vector3<double>* gdr1 = nullptr;
    ModuleBase::Vector3<double>* gdr2 = nullptr;
    ModuleBase::Vector3<double>* h1 = nullptr;
    ModuleBase::Vector3<double>* h2 = nullptr;
    double* neg = nullptr;
    double** vsave = nullptr;
    double** vgg = nullptr;

    // for spin unpolarized case,
    // calculate the gradient of (rho_core+rho) in reciprocal space.
    rhotmp1 = new double[rhopw->nrxx];
    rhogsum1 = new std::complex<double>[rhopw->npw];
    ModuleBase::timer::start("XC_Functional", "gradcorr_pack");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int ir=0; ir<rhopw->nrxx; ir++)
    {
        rhotmp1[ir] = chr->rho[0][ir] + fac * chr->rho_core[ir];
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int ig=0; ig<rhopw->npw; ig++)
    {
        rhogsum1[ig] = chr->rhog[0][ig] + fac * chr->rhog_core[ig];
    }
    ModuleBase::timer::end("XC_Functional", "gradcorr_pack");

    gdr1 = new ModuleBase::Vector3<double>[rhopw->nrxx];
    if(!is_stress)
    {
        h1 = new ModuleBase::Vector3<double>[rhopw->nrxx];
    }

    ModuleBase::timer::start("XC_Functional", "gradcorr_grad_rho");
    XC_Functional::grad_rho( rhogsum1 , gdr1, rhopw, ucell->tpiba);
    ModuleBase::timer::end("XC_Functional", "gradcorr_grad_rho");

    // for spin polarized case;
    // calculate the gradient of (rho_core+rho) in reciprocal space.
    if(nspin==2)
    {
        rhotmp2 = new double[rhopw->nrxx];
        rhogsum2 = new std::complex<double>[rhopw->npw];
        ModuleBase::timer::start("XC_Functional", "gradcorr_pack");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<rhopw->nrxx; ir++)
        {
            rhotmp2[ir] = chr->rho[1][ir] + fac * chr->rho_core[ir];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ig=0; ig<rhopw->npw; ig++)
        {
            rhogsum2[ig] = chr->rhog[1][ig] + fac * chr->rhog_core[ig];
        }
        ModuleBase::timer::end("XC_Functional", "gradcorr_pack");

        gdr2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        if(!is_stress)
        {
            h2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        }

        ModuleBase::timer::start("XC_Functional", "gradcorr_grad_rho");
        XC_Functional::grad_rho( rhogsum2 , gdr2, rhopw, ucell->tpiba);
        ModuleBase::timer::end("XC_Functional", "gradcorr_grad_rho");
    }

    if(nspin == 4&&(domag||domag_z))
    {
        rhotmp2 = new double[rhopw->nrxx];
        rhogsum2 = new std::complex<double>[rhopw->npw];
        neg = new double [rhopw->nrxx];
        ModuleBase::timer::start("XC_Functional", "gradcorr_pack");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<rhopw->nrxx; ir++)
        {
            rhotmp1[ir] = 0.0;
            rhotmp2[ir] = 0.0;
            neg[ir] = 0.0;
        }
        ModuleBase::timer::end("XC_Functional", "gradcorr_pack");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ig=0; ig<rhopw->npw; ig++)
        {
            rhogsum1[ig] = 0.0;
            rhogsum2[ig] = 0.0;
        }
        if(!is_stress)
        {
            vsave = new double* [nspin];
            for(int is = 0;is<nspin;is++)
            {
                vsave[is]= new double [rhopw->nrxx];
            }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
            for(int is = 0;is<nspin;is++)
            {
                for(int ir =0;ir<rhopw->nrxx;ir++)
                {
                    vsave[is][ir] = v(is,ir);
                    v(is,ir) = 0;
                }
            }
            vgg = new double* [nspin0];
            for(int is = 0;is<nspin0;is++)
            {
                vgg[is] = new double[rhopw->nrxx];
            }
        }
        noncolin_rho(rhotmp1, rhotmp2, neg, chr->rho, rhopw->nrxx, ucell->magnet.ux_, ucell->magnet.lsign_);
        rhopw->real2recip(rhotmp1, rhogsum1);
        rhopw->real2recip(rhotmp2, rhogsum2);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<rhopw->nrxx; ir++)
        {
            rhotmp2[ir] += fac * chr->rho_core[ir];
            rhotmp1[ir] += fac * chr->rho_core[ir];
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ig=0; ig<rhopw->npw; ig++)
        {
            rhogsum2[ig] += fac * chr->rhog_core[ig];
            rhogsum1[ig] += fac * chr->rhog_core[ig];
        }

        gdr2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        if(!is_stress)
        {
            h2 = new ModuleBase::Vector3<double>[rhopw->nrxx];
        }

        ModuleBase::timer::start("XC_Functional", "gradcorr_grad_rho");
        XC_Functional::grad_rho( rhogsum1 , gdr1, rhopw, ucell->tpiba);
        XC_Functional::grad_rho( rhogsum2 , gdr2, rhopw, ucell->tpiba);
        ModuleBase::timer::end("XC_Functional", "gradcorr_grad_rho");
    }

    const double epsr = 1.0e-6;
    const double epsg = 1.0e-10;

    double vtxcgc = 0.0;
    double etxcgc = 0.0;

    ModuleBase::timer::start("XC_Functional", "gradcorr_eval_grid");
    bool gpu_gradcorr_grid_done = false;
#if __CUDA || __UT_USE_CUDA
    const char* xc_gpu_env = std::getenv("ABACUS_XC_GPU");
    const bool xc_gpu_enabled = xc_gpu_env != nullptr && std::string(xc_gpu_env) == "1";
    const bool is_pbe = func_id.size() == 2 && func_id[0] == XC_GGA_X_PBE && func_id[1] == XC_GGA_C_PBE;
    const bool is_pbesol = func_id.size() == 2 && func_id[0] == XC_GGA_X_PBE_SOL && func_id[1] == XC_GGA_C_PBE_SOL;
    const bool use_gpu_gradcorr_grid = XC_Functional_GPU::xc_gpu_policy(device == "gpu",
                                                                         !xc_gpu_enabled,
                                                                         PARAM.inp.nspin,
                                                                         is_pbesol ? "PBEsol" : "PBE")
                                        && !use_libxc && !is_stress && nspin0 == 1 && (is_pbe || is_pbesol);
    if (use_gpu_gradcorr_grid)
    {
        using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
        using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
        using syncmem_h2d_op
            = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
        using syncmem_d2h_op
            = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

        ModuleBase::timer::start("XC_Functional", "gradcorr_eval_grid_gpu");
        std::vector<double> gdr1_flat(3 * rhopw->nrxx);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            gdr1_flat[3 * ir + 0] = gdr1[ir].x;
            gdr1_flat[3 * ir + 1] = gdr1[ir].y;
            gdr1_flat[3 * ir + 2] = gdr1[ir].z;
        }

        double* d_rhotmp1 = nullptr;
        double* d_rho_core = nullptr;
        double* d_gdr1 = nullptr;
        double* d_v = nullptr;
        double* d_h1 = nullptr;
        double* d_sums = nullptr;
        resmem_double_op()(d_rhotmp1, rhopw->nrxx);
        resmem_double_op()(d_rho_core, rhopw->nrxx);
        resmem_double_op()(d_gdr1, 3 * rhopw->nrxx);
        resmem_double_op()(d_v, rhopw->nrxx);
        resmem_double_op()(d_h1, 3 * rhopw->nrxx);
        resmem_double_op()(d_sums, 2);
        syncmem_h2d_op()(d_rhotmp1, rhotmp1, rhopw->nrxx);
        syncmem_h2d_op()(d_rho_core, chr->rho_core, rhopw->nrxx);
        syncmem_h2d_op()(d_gdr1, gdr1_flat.data(), 3 * rhopw->nrxx);

        const int iflag = is_pbesol ? 2 : 0;
        hamilt::xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                           rhopw->nrxx,
                                                                           iflag,
                                                                           ModuleBase::e2,
                                                                           epsr,
                                                                           d_rhotmp1,
                                                                           d_rho_core,
                                                                           d_gdr1,
                                                                           d_v,
                                                                           d_h1,
                                                                           d_sums,
                                                                           &etxcgc,
                                                                           &vtxcgc);
        std::vector<double> v_gpu(rhopw->nrxx);
        std::vector<double> h_gpu(3 * rhopw->nrxx);
        syncmem_d2h_op()(v_gpu.data(), d_v, rhopw->nrxx);
        syncmem_d2h_op()(h_gpu.data(), d_h1, 3 * rhopw->nrxx);
        for (int ir = 0; ir < rhopw->nrxx; ++ir)
        {
            v(0, ir) += v_gpu[ir];
            h1[ir].x = h_gpu[3 * ir + 0];
            h1[ir].y = h_gpu[3 * ir + 1];
            h1[ir].z = h_gpu[3 * ir + 2];
        }

        delmem_double_op()(d_rhotmp1);
        delmem_double_op()(d_rho_core);
        delmem_double_op()(d_gdr1);
        delmem_double_op()(d_v);
        delmem_double_op()(d_h1);
        delmem_double_op()(d_sums);
        ModuleBase::timer::end("XC_Functional", "gradcorr_eval_grid_gpu");
        gpu_gradcorr_grid_done = true;
    }
#endif
    if (!gpu_gradcorr_grid_done)
    {
#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<double> local_stress_gga;
        double local_vtxcgc = 0.0;
        double local_etxcgc = 0.0;

        if(is_stress)
        {
            local_stress_gga.resize(9);
            for(int i=0;i<9;i++)
            {
                local_stress_gga[i] = 0.0;
            }
        }
#else
    std::vector<double> &local_stress_gga = stress_gga;
    double &local_vtxcgc = vtxcgc;
    double &local_etxcgc = etxcgc;
#endif

        double grho2a = 0.0;
        double grho2b = 0.0;
        double sxc = 0.0;
        double v1xc = 0.0;
        double v2xc = 0.0;

        if(nspin0==1)
        {
            double segno = 0.0;
#ifdef _OPENMP
#pragma omp for
#endif
            for(int ir=0; ir<rhopw->nrxx; ir++)
            {
                const double arho = std::abs( rhotmp1[ir] );
                if(!is_stress)
                {
                    h1[ir].x = 0.0;
                    h1[ir].y = 0.0;
                    h1[ir].z = 0.0;
                }

                if(arho > epsr)
                {
                    grho2a = gdr1[ir].norm2();

                    //normally values in rhotmp can either be >= 0 or < 0.
                    if( rhotmp1[ir] >= 0.0 )
                    {
                        segno = 1.0;
                    }
                    else
                    {
                        segno = -1.0;
                    }
                    if (use_libxc && is_stress)
                    {
#ifdef USE_LIBXC
                        if(func_type == 3 || func_type == 5)
                        {
                            double v3xc = 0.0;
                            double atau = chr->kin_r[0][ir]/2.0;
                            double hybrid_alpha = 0.0;
#ifdef __EXX
                            hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
#endif
                            XC_Functional_Libxc::tau_xc( func_id, arho, grho2a, atau, sxc, v1xc, v2xc, v3xc, hybrid_alpha);
                        }
                        else
                        {
                            XC_Functional_Libxc::gcxc_libxc( func_id, arho, grho2a, sxc, v1xc, v2xc);
                        }
#endif
                    }
                    else
                    {
                        XC_Functional::gcxc( arho, grho2a, sxc, v1xc, v2xc);
                    }
                    if(is_stress)
                    {
                        double tt[3];
                        tt[0] = gdr1[ir].x;
                        tt[1] = gdr1[ir].y;
                        tt[2] = gdr1[ir].z;
                        for(int l = 0;l< 3;l++)
                        {
                            for(int m = 0;m< l+1;m++)
                            {
                                int ind = l*3 + m;
                                local_stress_gga[ind] += tt[l] * tt[m] * ModuleBase::e2 * v2xc;
                            }
                        }
                    }
                    else
                    {
                        // first term of the gradient correction:
                        // D(rho*Exc)/D(rho)
                        v(0, ir) += ModuleBase::e2 * v1xc;

                        // h contains
                        // D(rho*Exc) / D(|grad rho|) * (grad rho) / |grad rho|
                        h1[ir] = ModuleBase::e2 * v2xc * gdr1[ir];

                        local_vtxcgc += ModuleBase::e2* v1xc * ( rhotmp1[ir] - chr->rho_core[ir] );
                        local_etxcgc += ModuleBase::e2* sxc  * segno;
                    }
                }
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp for
#endif
            for(int ir=0; ir<rhopw->nrxx; ir++)
            {
                if(use_libxc)
                {
#ifdef USE_LIBXC
                    double sxc = 0.0;
                    double v1xcup = 0.0;
                    double v1xcdw = 0.0;
                    double v2xcup = 0.0;
                    double v2xcdw = 0.0;
                    double v2xcud = 0.0;
                    if(func_type == 3 || func_type == 5)
                    {
                        double v3xcup = 0.0;
                        double v3xcdw = 0.0;
                        double atau1 = chr->kin_r[0][ir]/2.0;
                        double atau2 = chr->kin_r[1][ir]/2.0;
                        double hybrid_alpha = 0.0;
#ifdef __EXX
                        hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
#endif
                        XC_Functional_Libxc::tau_xc_spin(
                            func_id,
                            rhotmp1[ir], rhotmp2[ir], gdr1[ir], gdr2[ir],
                            atau1, atau2, sxc, v1xcup, v1xcdw, v2xcup, v2xcdw, v2xcud, v3xcup, v3xcdw, hybrid_alpha);
                    }
                    else
                    {
                        XC_Functional_Libxc::gcxc_spin_libxc(
                            func_id,
                            rhotmp1[ir], rhotmp2[ir], gdr1[ir], gdr2[ir],
                            sxc, v1xcup, v1xcdw, v2xcup, v2xcdw, v2xcud);
                    }
                    if(is_stress)
                    {
                        double tt1[3],tt2[3];
                        {
                            tt1[0] = gdr1[ir].x;
                            tt1[1] = gdr1[ir].y;
                            tt1[2] = gdr1[ir].z;
                            tt2[0] = gdr2[ir].x;
                            tt2[1] = gdr2[ir].y;
                            tt2[2] = gdr2[ir].z;
                        }
                        for(int l = 0;l< 3;l++)
                        {
                            for(int m = 0;m< l+1;m++)
                            {
                                int ind = l*3 + m;
                                local_stress_gga [ind] += ( tt1[l] * tt1[m] * v2xcup +
                                    tt2[l] * tt2[m] * v2xcdw +
                                    (tt1[l] * tt2[m] + tt2[l] * tt1[m] ) * v2xcud ) * ModuleBase::e2;
                            }
                        }
                    }
                    else
                    {
                        // first term of the gradient correction : D(rho*Exc)/D(rho)
                        v(0,ir) += ModuleBase::e2 * v1xcup;
                        v(1,ir) += ModuleBase::e2 * v1xcdw;

                        // h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
                        h1[ir] += ModuleBase::e2 * ( v2xcup * gdr1[ir] + v2xcud * gdr2[ir] );
                        h2[ir] += ModuleBase::e2 * ( v2xcdw * gdr2[ir] + v2xcud * gdr1[ir] );

                        local_vtxcgc = local_vtxcgc + ModuleBase::e2 * v1xcup * ( rhotmp1[ir] - chr->rho_core[ir] * fac );
                        local_vtxcgc = local_vtxcgc + ModuleBase::e2 * v1xcdw * ( rhotmp2[ir] - chr->rho_core[ir] * fac );
                        local_etxcgc = local_etxcgc + ModuleBase::e2 * sxc;
                    }
#endif
                }
                else
                {
                    double v1cup = 0.0;
                    double v1cdw = 0.0;
                    double v2cup = 0.0;
                    double v2cdw = 0.0;
                    double v1xup = 0.0;
                    double v1xdw = 0.0;
                    double v2xup = 0.0;
                    double v2xdw = 0.0;
                    double v2cud = 0.0;
                    double v2c = 0.0;
                    double sx = 0.0;
                    double sc = 0.0;
                    double rh = rhotmp1[ir] + rhotmp2[ir];
                    grho2a = gdr1[ir].norm2();
                    grho2b = gdr2[ir].norm2();
                    XC_Functional::gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
                        sx, v1xup, v1xdw, v2xup, v2xdw);

                    if(rh > epsr)
                    {
                        if(igcc_is_lyp)
                        {
                            ModuleBase::WARNING_QUIT("XC_Functional","igcc_is_lyp is not available now.");
                        }
                        else
                        {
                            double zeta = ( rhotmp1[ir] - rhotmp2[ir] ) / rh;
                            if(nspin==4&&(domag||domag_z))
                            {
                                zeta = fabs(zeta) * neg[ir];
                            }
                            const double grh2 = (gdr1[ir]+gdr2[ir]).norm2();
                            XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
                            v2cup = v2c;
                            v2cdw = v2c;
                            v2cud = v2c;
                        }
                    }
                    else
                    {
                        sc = 0.0;
                        v1cup = 0.0;
                        v1cdw = 0.0;
                        v2c = 0.0;
                        v2cup = 0.0;
                        v2cdw = 0.0;
                        v2cud = 0.0;
                    }

                    if(is_stress)
                    {
                        double tt1[3],tt2[3];
                        {
                            tt1[0] = gdr1[ir].x;
                            tt1[1] = gdr1[ir].y;
                            tt1[2] = gdr1[ir].z;
                            tt2[0] = gdr2[ir].x;
                            tt2[1] = gdr2[ir].y;
                            tt2[2] = gdr2[ir].z;
                        }
                        for(int l = 0;l< 3;l++)
                        {
                            for(int m = 0;m< l+1;m++)
                            {
                                int ind = l*3 + m;
                                // exchange
                                local_stress_gga [ind] += tt1[l] * tt1[m] * ModuleBase::e2 * v2xup +
                                    tt2[l] * tt2[m] * ModuleBase::e2 * v2xdw;
                                // correlation
                                local_stress_gga [ind] += ( tt1[l] * tt1[m] * v2cup +
                                    tt2[l] * tt2[m] * v2cdw +
                                    (tt1[l] * tt2[m] + tt2[l] * tt1[m] ) * v2cud ) * ModuleBase::e2;
                            }
                        }
                    }
                    else
                    {
                        // first term of the gradient correction : D(rho*Exc)/D(rho)
                        v(0,ir) = v(0,ir) + ModuleBase::e2 * ( v1xup + v1cup );
                        v(1,ir) = v(1,ir) + ModuleBase::e2 * ( v1xdw + v1cdw );

                        // h contains D(rho*Exc)/D(|grad rho|) * (grad rho) / |grad rho|
                        h1[ir] = ModuleBase::e2 * ( ( v2xup + v2cup ) * gdr1[ir] + v2cud * gdr2[ir] );
                        h2[ir] = ModuleBase::e2 * ( ( v2xdw + v2cdw ) * gdr2[ir] + v2cud * gdr1[ir] );

                        local_vtxcgc = local_vtxcgc + ModuleBase::e2 * ( v1xup + v1cup ) * ( rhotmp1[ir] - chr->rho_core[ir] * fac );
                        local_vtxcgc = local_vtxcgc + ModuleBase::e2 * ( v1xdw + v1cdw ) * ( rhotmp2[ir] - chr->rho_core[ir] * fac );
                        local_etxcgc = local_etxcgc + ModuleBase::e2 * ( sx + sc );
                    }
                }
            }
        }
#ifdef _OPENMP
    #pragma omp critical(xc_grad_reduce)
    {
        if(is_stress)
        {
            for(int l = 0;l< 3;l++)
            {
                for(int m = 0;m< l+1;m++)
                {
                    int ind = l*3 + m;
                    stress_gga [ind] += local_stress_gga [ind];
                }
            }
        }
        else
        {
            vtxcgc += local_vtxcgc;
            etxcgc += local_etxcgc;
        }
    }
}
#endif
    }
    ModuleBase::timer::end("XC_Functional", "gradcorr_eval_grid");

    if(!is_stress)
    {
        ModuleBase::timer::start("XC_Functional", "gradcorr_apply");
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<rhopw->nrxx; ir++)
        {
            rhotmp1[ir] -= fac * chr->rho_core[ir];
        }
        if(nspin0==2)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ir=0; ir<rhopw->nrxx; ir++)
            {
                rhotmp2[ir] -= fac * chr->rho_core[ir];
            }
        }

        // second term of the gradient correction :
        // sum_alpha (D / D r_alpha) ( D(rho*Exc)/D(grad_alpha rho) )

        // dh is in real sapce.
        double* dh = new double[rhopw->nrxx];

        for(int is=0; is<nspin0; is++)
        {
            if(is==0)
            {
                ModuleBase::timer::end("XC_Functional", "gradcorr_apply");
                ModuleBase::timer::start("XC_Functional", "gradcorr_grad_dot");
                XC_Functional::grad_dot(h1,dh,rhopw,ucell->tpiba);
                ModuleBase::timer::end("XC_Functional", "gradcorr_grad_dot");
                ModuleBase::timer::start("XC_Functional", "gradcorr_apply");
            }
            if(is==1)
            {
                ModuleBase::timer::end("XC_Functional", "gradcorr_apply");
                ModuleBase::timer::start("XC_Functional", "gradcorr_grad_dot");
                XC_Functional::grad_dot(h2,dh,rhopw,ucell->tpiba);
                ModuleBase::timer::end("XC_Functional", "gradcorr_grad_dot");
                ModuleBase::timer::start("XC_Functional", "gradcorr_apply");
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ir=0; ir<rhopw->nrxx; ir++)
            {
                v(is, ir) -= dh[ir];
            }

            double sum = 0.0;
            if(is==0)
            {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static, 256)
#endif
                for(int ir=0; ir<rhopw->nrxx; ir++)
                {
                    sum += dh[ir] * rhotmp1[ir];
                }
            }
            else if(is==1)
            {
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum) schedule(static, 256)
#endif
                for(int ir=0; ir<rhopw->nrxx; ir++)
                {
                    sum += dh[ir] * rhotmp2[ir];
                }
            }
            vtxcgc -= sum;
        }
        ModuleBase::timer::end("XC_Functional", "gradcorr_apply");

        delete[] dh;

        vtxc += vtxcgc;
        etxc += etxcgc;

        if(nspin == 4 && (domag||domag_z))
        {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
            for(int is=0;is<nspin;is++)
            {
                for(int ir=0;ir<rhopw->nrxx;ir++)
                {
                    if(is<nspin0)
                    {
                        vgg[is][ir] = v(is,ir);
                    }
                    v(is,ir) = vsave[is][ir];
                }
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ir=0;ir<rhopw->nrxx;ir++)
            {
                v(0,ir) += 0.5 * (vgg[0][ir] + vgg[1][ir]);
                double amag = sqrt(pow(chr->rho[1][ir],2)+pow(chr->rho[2][ir],2)+pow(chr->rho[3][ir],2));
                if(amag>1e-12)
                {
                    for(int i=1;i<4;i++)
                    {
                        v(i,ir)+= neg[ir] * 0.5 *(vgg[0][ir]-vgg[1][ir])*chr->rho[i][ir]/amag;
                    }
                }
            }
        }
    }
    // deacllocate
    delete[] rhotmp1;
    delete[] rhogsum1;
    delete[] gdr1;
    if(!is_stress)
    {
        delete[] h1;
    }

    if(nspin==2)
    {
        delete[] rhotmp2;
        delete[] rhogsum2;
        delete[] gdr2;
        if(!is_stress)
        {
            delete[] h2;
        }
    }
    if(nspin == 4 && (domag||domag_z))
    {
        delete[] neg;
        if(!is_stress)
        {
            for(int i=0; i<nspin0; i++)
            {
                delete[] vgg[i];
            }
            delete[] vgg;
            for(int i=0; i<nspin; i++)
            {
                delete[] vsave[i];
            }
            delete[] vsave;
            delete[] h2;
        }
        delete[] rhotmp2;
        delete[] rhogsum2;
        delete[] gdr2;
    }

    return;
}

template <typename T, typename Device, typename Real>
void XC_Functional::grad_wfc(
    const int ik,
    const Real tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const T* rhog,
    T* grad)
{
    using ct_Device = typename ct::PsiToContainer<Device>::type;
    const int npw_k = wfc_basis->npwk[ik];

    auto porter = std::move(ct::Tensor(
        ct::DataTypeToEnum<T>::value, ct::DeviceTypeToEnum<ct_Device>::value, {wfc_basis->nmaxgr}));
    auto gcar = ct::TensorMap(
        &wfc_basis->gcar[0][0], ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {wfc_basis->nks * wfc_basis->npwk_max, 3}).to_device<ct_Device>();
    auto kvec_c = ct::TensorMap(
        &wfc_basis->kvec_c[0][0],ct::DataType::DT_DOUBLE, ct::DeviceType::CpuDevice, {wfc_basis->nks, 3}).to_device<ct_Device>();

    auto xc_functional_grad_wfc_solver
        = hamilt::xc_functional_grad_wfc_op<T, Device>();

    for(int ipol=0; ipol<3; ipol++)
    {
        xc_functional_grad_wfc_solver(
            ik, ipol, npw_k, wfc_basis->npwk_max,
            tpiba,
            gcar.template data<Real>(),
            kvec_c.template data<Real>(),
            rhog, porter.data<T>());

        // bring the gdr from G --> R
        Device * ctx = nullptr;
        wfc_basis->recip_to_real(ctx, porter.data<T>(), porter.data<T>(), ik);

        xc_functional_grad_wfc_solver(
            ipol, wfc_basis->nrxx,
            porter.data<T>(), grad);
    }
}


void XC_Functional::grad_rho(
    const std::complex<double>* rhog,
    ModuleBase::Vector3<double>* gdr,
    const ModulePW::PW_Basis* rho_basis,
    const double tpiba)
{
    std::complex<double> *gdrtmp = new std::complex<double>[rho_basis->nmaxgr];

    // the formula is : rho(r)^prime = int iG * rho(G)e^{iGr} dG
    for(int i = 0 ; i < 3 ; ++i)
    {
        // calculate the charge density gradient in reciprocal space.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ig=0; ig<rho_basis->npw; ig++)
        {
            gdrtmp[ig] = ModuleBase::IMAG_UNIT * rhog[ig] * rho_basis->gcar[ig][i];
        }

        // bring the gdr from G --> R
        rho_basis->recip2real(gdrtmp, gdrtmp);

        // remember to multily 2pi/a0, which belongs to G vectors.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir=0; ir<rho_basis->nrxx; ir++)
        {
            gdr[ir][i] = gdrtmp[ir].real() * tpiba;
        }
    }

    delete[] gdrtmp;
    return;
}


void XC_Functional::grad_dot(
    const ModuleBase::Vector3<double>* h,
    double* dh,
    const ModulePW::PW_Basis* rho_basis,
    const double tpiba)
{
    std::complex<double> *aux = new std::complex<double>[rho_basis->nmaxgr];
    std::complex<double> *gaux = new std::complex<double>[rho_basis->npw];

    for(int i = 0 ; i < 3 ; ++i)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir = 0; ir < rho_basis->nrxx; ++ir)
        {
            aux[ir] = std::complex<double>( h[ir][i], 0.0);
        }

        // bring to G space.
        rho_basis->real2recip(aux,aux);
        if (i == 0)
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ig = 0; ig < rho_basis->npw; ++ig)
            {
                gaux[ig] =  ModuleBase::IMAG_UNIT * aux[ig] * rho_basis->gcar[ig][i];
            }
        }
        else
        {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for(int ig = 0; ig < rho_basis->npw; ++ig)
            {
                gaux[ig] +=  ModuleBase::IMAG_UNIT * aux[ig] * rho_basis->gcar[ig][i];
            }
        }
    }

    // bring back to R space
    rho_basis->recip2real(gaux,aux);

#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int ir=0; ir<rho_basis->nrxx; ir++)
    {
        dh[ir] = aux[ir].real() * tpiba;
    }

    delete[] aux;
    delete[] gaux;
    return;
}

void XC_Functional::noncolin_rho(
    double *rhoout1,
    double *rhoout2,
    double *neg,
    const double*const*const rho,
    const int nrxx,
    const double* ux_,
    const bool lsign_)
{
    //this function diagonalizes the spin density matrix and gives as output the
    //spin up and spin down components of the charge.
    //If lsign is true up and dw are with respect to the fixed quantization axis
    //ux, otherwise rho + |m| is always rhoup and rho-|m| is always rhodw.
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for(int ir = 0;ir<nrxx;ir++)
    {
        neg[ir] = 1.0;
    }
    if(lsign_)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for(int ir = 0;ir<nrxx;ir++)
        {
            if(rho[1][ir]*ux_[0] + rho[2][ir]*ux_[1] + rho[3][ir]*ux_[2]>0)
            {
                neg[ir] = 1.0;
            }
            else
            {
                neg[ir] = -1.0;
            }
        }
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int ir = 0;ir<nrxx;ir++)
    {
        double amag = sqrt(pow(rho[1][ir],2)+pow(rho[2][ir],2)+pow(rho[3][ir],2));
        rhoout1[ir] = 0.5 * (rho[0][ir] + neg[ir] * amag);
        rhoout2[ir] = 0.5 * (rho[0][ir] - neg[ir] * amag);
    }
    return;
}

template void XC_Functional::grad_wfc<std::complex<double>, base_device::DEVICE_CPU, double>(
    const int ik,
    const double tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const std::complex<double>* rhog,
    std::complex<double>* grad);
#if __CUDA || __ROCM
template void XC_Functional::grad_wfc<std::complex<double>, base_device::DEVICE_GPU, double>(
    const int ik,
    const double tpiba,
    const ModulePW::PW_Basis_K* wfc_basis,
    const std::complex<double>* rhog,
    std::complex<double>* grad);
#endif

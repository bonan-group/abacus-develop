#include "esolver_ks_pw.h"

#include "source_estate/cal_ux.h"
#include "source_estate/elecstate_pw.h"
#include "source_estate/module_charge/symmetry_rho.h"

#include "source_hsolver/diago_iter_assist.h"
#include "source_hsolver/hsolver_pw.h"
#include "source_hsolver/diago_params.h"

#include "source_hsolver/kernels/hegvd_op.h"
#include "source_io/module_parameter/parameter.h"
#include "source_lcao/module_deltaspin/spin_constrain.h"
#include "source_pw/module_pwdft/onsite_proj.h"
#include "source_lcao/module_dftu/dftu.h"
#include "source_pw/module_pwdft/vsep_pw.h"
#include "source_pw/module_pwdft/hamilt_pw.h"

#include "source_pw/module_pwdft/forces.h"
#include "source_pw/module_pwdft/stress_pw.h"
#include "source_hamilt/module_xc/xc_functional.h" // use XC_Functional

#ifdef __DSP
#include "source_base/kernels/dsp/dsp_connector.h"
#endif

#include "source_pw/module_pwdft/setup_pot.h" // mohan add 20250929
#include "source_estate/setup_estate_pw.h" // mohan add 20251005
#include "source_io/module_ctrl/ctrl_output_pw.h" // mohan add 20250927
#include "source_estate/module_charge/chgmixing.h" // use charge mixing, mohan add 20251006 
#include "source_estate/update_pot.h" // mohan add 20251016
#include "source_pw/module_pwdft/update_cell_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/dftu_pw.h" // mohan add 20250309
#include "source_pw/module_pwdft/deltaspin_pw.h" // mohan add 20250309

#include "source_hamilt/module_xc/exx_info.h" // use GlobalC::exx_info
#include "source_io/module_energy/write_bands.h"
#include "source_pw/module_pwdft/op_pw_exx.h"
#include "source_pw/module_pwdft/setup_pwwfc.h"

#include <algorithm>

namespace ModuleESolver
{

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::ESolver_KS_PW()
{
    this->classname = "ESolver_KS_PW";
    this->basisname = "PW";
}

template <typename T, typename Device>
ESolver_KS_PW<T, Device>::~ESolver_KS_PW()
{
    //****************************************************
    // do not add any codes in this deconstructor funcion
    //****************************************************
    // delete Hamilt
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }

    // delete exx_helper
    if (this->exx_helper != nullptr)
    {
        delete this->exx_helper;
        this->exx_helper = nullptr;
    }

    // mohan add 2025-10-12
    this->stp.clean();
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::allocate_hamilt(const UnitCell& ucell)
{
	this->p_hamilt = new hamilt::HamiltPW<T, Device>(
			this->pelec->pot, 
			this->pw_wfc, 
			&this->kv, 
			&this->ppcell, 
			&this->dftu,
			&ucell);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::validate_mixed_band_targets(const Input_para& inp) const
{
    if (inp.mem_saver == 1 && inp.calculation == "scf" && !this->kv.has_band_kpoints())
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mem_saver=1 for scf PW calculations requires K_POINTS_BAND target k-points");
    }
    if (!this->kv.has_band_kpoints())
    {
        return;
    }

    if (inp.calculation != "scf")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets require calculation scf");
    }
    if (inp.basis_type != "pw")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets require basis_type pw");
    }
    if (!inp.out_band[0])
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "K_POINTS_BAND requires out_band 1");
    }
    if (!GlobalC::exx_info.info_global.cal_exx || !inp.exxace || !GlobalC::exx_info.info_global.separate_loop)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets require PW hybrid EXX with exx_ace 1 and exx_separate_loop 1");
    }
    if (GlobalV::KPAR > 1)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets do not support KPAR>1 in v1");
    }
    if (inp.mem_saver == 1 && inp.ks_solver == "cg")
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mem_saver=1 for mixed hybrid band targets does not support ks_solver cg; "
                                 "use dav, dav_subspace, or bpcg");
    }
    if (inp.mem_saver == 1 && inp.out_wfc_pw != 0)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::validate_mixed_band_targets",
                                 "mixed hybrid band targets write eigenvalues only and do not support out_wfc_pw "
                                 "with mem_saver=1");
    }
}



template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_all_runners(UnitCell& ucell, const Input_para& inp)
{
    ESolver_KS::before_all_runners(ucell, inp);
    this->validate_mixed_band_targets(inp);

    //! setup and allocation for pelec, potentials, etc. 
    elecstate::setup_estate_pw(ucell, this->kv, this->sf, this->pelec, this->chr,
      this->locpp, this->ppcell, this->vsep_cell, this->pw_wfc, this->pw_rho,
      this->pw_rhod, this->pw_big, this->solvent, inp);

    this->stp.before_runner(ucell, this->kv, this->sf, *this->pw_wfc, this->ppcell, PARAM.inp);

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "INIT BASIS");

    //! Create exx_helper based on device and precision
    const bool is_gpu = (inp.device == "gpu");
    const bool is_single = (inp.precision == "single");

#if ((defined __CUDA) || (defined __ROCM))
    if (is_gpu)
    {
        if (is_single)
        {
            this->exx_helper = new Exx_Helper<std::complex<float>, base_device::DEVICE_GPU>();
        }
        else
        {
            this->exx_helper = new Exx_Helper<std::complex<double>, base_device::DEVICE_GPU>();
        }
    }
    else
#endif
    {
        if (is_single)
        {
            this->exx_helper = new Exx_Helper<std::complex<float>, base_device::DEVICE_CPU>();
        }
        else
        {
            this->exx_helper = new Exx_Helper<std::complex<double>, base_device::DEVICE_CPU>();
        }
    }

    //! Initialize exx pw
    this->exx_helper->init(ucell, inp, this->pelec->wg);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::before_scf(UnitCell& ucell, const int istep)
{
    ModuleBase::TITLE("ESolver_KS_PW", "before_scf");
    ModuleBase::timer::start("ESolver_KS_PW", "before_scf");

    ESolver_KS::before_scf(ucell, istep);

    //! Init variables (once the cell has changed)
    pw::update_cell_pw(ucell, this->ppcell, this->kv, this->pw_wfc, PARAM.inp);

    if (ucell.cell_parameter_updated)
    {
        this->stp.p_psi_init->prepare_init(PARAM.inp.pw_seed);
    }

    //! Init Hamiltonian (cell changed)
    //! Operators in HamiltPW should be reallocated once cell changed
    //! delete Hamilt if not first scf
    if (this->p_hamilt != nullptr)
    {
        delete this->p_hamilt;
        this->p_hamilt = nullptr;
    }

    //! Allocate HamiltPW
    this->allocate_hamilt(ucell);

    //! Setup potentials (local, non-local, sc, +U, DFT-1/2)
    // note: init DFT+U is done here for pw basis for every scf iteration, however, 
    // init DFT+U is done in "before_all_runners" in LCAO basis. This should be refactored, mohan note 2025-11-06
    pw::setup_pot(istep, ucell, this->kv, this->sf, this->pelec, this->Pgrid,
              this->chr, this->locpp, this->ppcell, this->dftu, this->vsep_cell,
              this->stp.template get_psi_t<T, Device>(), 
	      this->p_hamilt, 
	      this->pw_wfc, this->pw_rhod, PARAM.globalv.global_out_dir, PARAM.inp);

    // setup psi (electronic wave functions)
    this->stp.init(this->p_hamilt);

    //! Setup EXX helper for Hamiltonian and psi
    exx_helper->before_scf(this->p_hamilt, this->stp.template get_psi_t<T, Device>(), PARAM.inp);

    ModuleBase::timer::end("ESolver_KS_PW", "before_scf");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_init(UnitCell& ucell, const int istep, const int iter)
{
    ESolver_KS::iter_init(ucell, istep, iter);

    module_charge::chgmixing_ks_pw(iter, this->p_chgmix, this->dftu, PARAM.inp);

    // mohan move harris functional here, 2012-06-05
    // use 'rho(in)' and 'v_h and v_xc'(in)
    this->pelec->f_en.deband_harris = this->pelec->cal_delta_eband(ucell);

    // update local occupations for DFT+U
    // should before lambda loop in DeltaSpin
    pw::iter_init_dftu_pw(iter, istep, this->dftu, this->stp.template get_psi_t<T, Device>(), this->pelec->wg, ucell, this->p_chgmix, this->kv.isk.data());
}

// Temporary, it should be replaced by hsolver later.
template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::hamilt2rho_single(UnitCell& ucell, const int istep, const int iter, const double ethr)
{
    ModuleBase::timer::start("ESolver_KS_PW", "hamilt2rho_single");

    // reset energy
    this->pelec->f_en.eband = 0.0;
    this->pelec->f_en.demet = 0.0;

    // setup diagonalization parameters
    hsolver::setup_diago_params_pw<T, Device>(istep, iter, ethr, PARAM.inp);

    bool skip_charge = PARAM.inp.calculation == "nscf" ? true : false;

    // run the inner lambda loop to contrain atomic moments with the DeltaSpin method
    bool skip_solve = pw::run_deltaspin_lambda_loop(iter - 1, this->drho, PARAM.inp);

    if (!skip_solve)
    {
        hsolver::HSolverPW<T, Device> hsolver_pw_obj(this->pw_wfc,
                                                     PARAM.inp.calculation,
                                                     PARAM.inp.basis_type,
                                                     PARAM.inp.ks_solver,
                                                     PARAM.globalv.use_uspp,
                                                     PARAM.inp.nspin,
                                                     hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                     hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                     hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                     PARAM.inp.use_k_continuity);

        hsolver_pw_obj.solve(static_cast<hamilt::Hamilt<T, Device>*>(this->p_hamilt), *this->stp.template get_psi_t<T, Device>(), this->pelec, this->pelec->ekb.c,
          GlobalV::RANK_IN_POOL, GlobalV::NPROC_IN_POOL, skip_charge, ucell.tpiba, ucell.nat);
    }

    // symmetrize the charge density
    Symmetry_rho::symmetrize_rho(PARAM.inp.nspin, this->chr, this->pw_rhod, ucell.symm);

    ModuleBase::timer::end("ESolver_KS_PW", "hamilt2rho_single");
}


template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::iter_finish(UnitCell& ucell, const int istep, int& iter, bool& conv_esolver)
{
    // Related to EXX
    if (GlobalC::exx_info.info_global.cal_exx && !exx_helper->get_op_first_iter())
    {
        this->pelec->set_exx(exx_helper->cal_exx_energy(this->stp.template get_psi_t<T, Device>()));
    }

    // deband is calculated from "output" charge density
    this->pelec->f_en.deband = this->pelec->cal_delta_eband(ucell);

    // Call iter_finish() of ESolver_KS
    ESolver_KS::iter_finish(ucell, istep, iter, conv_esolver);

    // D in USPP needs vloc, thus needs update when veff updated
    // calculate the effective coefficient matrix for non-local
    // pp projectors, liuyu 2023-10-24
    if (PARAM.globalv.use_uspp)
    {
        ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
        this->ppcell.cal_effective_D(veff, this->pw_rhod, ucell);
    }

    // Handle EXX-related operations after SCF iteration
    exx_helper->iter_finish(this->pelec, &this->chr, this->stp.template get_psi_t<T, Device>(), ucell, PARAM.inp, conv_esolver, iter);

    // check if oscillate for delta_spin method
    pw::check_deltaspin_oscillation(iter, this->drho, this->p_chgmix, PARAM.inp);

    // the output quantities
    ModuleIO::ctrl_iter_pw(istep, iter, conv_esolver, this->stp.psi_cpu, 
              this->kv, this->pw_wfc, PARAM.inp);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_scf(UnitCell& ucell, const int istep, const bool conv_esolver)
{
    ModuleBase::TITLE("ESolver_KS_PW", "after_scf");
    ModuleBase::timer::start("ESolver_KS_PW", "after_scf");

    // Calculate kinetic energy density tau for ELF if needed
    if (PARAM.inp.out_elf[0] > 0)
    {
        auto* elec_pw = static_cast<elecstate::ElecStatePW<T, Device>*>(this->pelec);
        auto& psi = *this->stp.template get_psi_t<T, Device>();
        elec_pw->cal_tau(psi);
    }

    ESolver_KS::after_scf(ucell, istep, conv_esolver);

    // Output quantities
    ModuleIO::ctrl_scf_pw<T, Device>(istep, ucell, this->pelec, this->chr, this->kv, this->pw_wfc,
              this->pw_rho, this->pw_rhod, this->pw_big, this->stp,
              this->Pgrid, PARAM.inp);

    if (conv_esolver)
    {
        this->solve_mixed_band_targets(ucell);
    }

    ModuleBase::timer::end("ESolver_KS_PW", "after_scf");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::solve_mixed_band_targets(UnitCell& ucell)
{
    if (!this->kv.has_band_kpoints())
    {
        return;
    }

    ModuleBase::TITLE("ESolver_KS_PW", "solve_mixed_band_targets");
    ModuleBase::timer::start("ESolver_KS_PW", "mixed_band_targets");

    auto* source_exx = static_cast<hamilt::OperatorEXXPW<T, Device>*>(this->exx_helper->get_op_exx());
    if (source_exx == nullptr)
    {
        ModuleBase::WARNING_QUIT("ESolver_KS_PW::solve_mixed_band_targets",
                                 "SCF EXX source operator is not initialized");
    }

    if (PARAM.inp.mem_saver == 1)
    {
        this->solve_mixed_band_targets_mem_saver(ucell, source_exx);
    }
    else
    {
        this->solve_mixed_band_targets_full(ucell, source_exx);
    }

    ModuleBase::timer::end("ESolver_KS_PW", "mixed_band_targets");
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::solve_mixed_band_targets_full(UnitCell& ucell,
                                                             hamilt::OperatorEXXPW<T, Device>* source_exx)
{
    K_Vectors band_kv = this->kv.make_band_target_kvectors(PARAM.inp.nspin);
    ModulePW::PW_Basis_K* band_pw_wfc = nullptr;
    pw::setup_pwwfc(PARAM.inp, ucell, *this->pw_rho, band_kv, band_pw_wfc);

    pseudopot_cell_vnl band_ppcell;
    band_ppcell.init(ucell, &this->sf, band_pw_wfc);
    band_ppcell.init_vnl(ucell, this->pw_rhod);
    ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
    band_ppcell.cal_effective_D(veff, this->pw_rhod, ucell);

    Setup_Psi_pw band_stp;
    band_stp.before_runner(ucell, band_kv, this->sf, *band_pw_wfc, band_ppcell, PARAM.inp);

    auto* band_hamilt = new hamilt::HamiltPW<T, Device>(this->pelec->pot,
                                                         band_pw_wfc,
                                                         &band_kv,
                                                         &band_ppcell,
                                                         &this->dftu,
                                                         &ucell,
                                                         source_exx);
    band_stp.init(band_hamilt);

    elecstate::ElecStatePW<T, Device> band_elec(band_pw_wfc,
                                                &this->chr,
                                                &band_kv,
                                                &ucell,
                                                &band_ppcell,
                                                this->pw_rho,
                                                this->pw_big);
    band_elec.pot = this->pelec->pot;
    band_elec.skip_weights = true;
    band_elec.wg.zero_out();

    const double target_ethr = std::max(PARAM.inp.scf_thr, 1.0e-8);
    hsolver::setup_diago_params_pw<T, Device>(0, 1, target_ethr, PARAM.inp);
    hsolver::HSolverPW<T, Device> hsolver_pw_obj(band_pw_wfc,
                                                 "nscf",
                                                 PARAM.inp.basis_type,
                                                 PARAM.inp.ks_solver,
                                                 PARAM.globalv.use_uspp,
                                                 PARAM.inp.nspin,
                                                 hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                 hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                 false);

    hsolver_pw_obj.solve(static_cast<hamilt::Hamilt<T, Device>*>(band_hamilt),
                         *band_stp.template get_psi_t<T, Device>(),
                         &band_elec,
                         band_elec.ekb.c,
                         GlobalV::RANK_IN_POOL,
                         GlobalV::NPROC_IN_POOL,
                         true,
                         ucell.tpiba,
                         ucell.nat);

    ModuleIO::write_bands(PARAM.inp, band_elec.ekb, band_kv);
    band_elec.pot = nullptr;
    delete band_hamilt;
    band_stp.clean();
    pw::teardown_pwwfc(band_pw_wfc);
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::solve_mixed_band_targets_mem_saver(UnitCell& ucell,
                                                                  hamilt::OperatorEXXPW<T, Device>* source_exx)
{
    K_Vectors band_kv = this->kv.make_band_target_kvectors(PARAM.inp.nspin);
    ModulePW::PW_Basis_K* band_pw_wfc = nullptr;
    pw::setup_pwwfc(PARAM.inp, ucell, *this->pw_rho, band_kv, band_pw_wfc);

    pseudopot_cell_vnl band_ppcell;
    band_ppcell.init(ucell, &this->sf, band_pw_wfc);
    band_ppcell.init_vnl(ucell, this->pw_rhod);

    ModuleBase::matrix band_ekb(band_kv.get_nks(), PARAM.globalv.nbands_l);
    const double target_ethr = std::max(PARAM.inp.scf_thr, 1.0e-8);
    ModuleBase::matrix veff = this->pelec->pot->get_eff_v();
    band_ppcell.cal_effective_D(veff, this->pw_rhod, ucell);

    Setup_Psi_pw band_stp;
    band_stp.before_runner(ucell, band_kv, this->sf, *band_pw_wfc, band_ppcell, PARAM.inp, true);

    auto* band_hamilt = new hamilt::HamiltPW<T, Device>(this->pelec->pot,
                                                         band_pw_wfc,
                                                         &band_kv,
                                                         &band_ppcell,
                                                         &this->dftu,
                                                         &ucell,
                                                         source_exx);
    elecstate::ElecStatePW<T, Device> band_elec(band_pw_wfc,
                                                &this->chr,
                                                &band_kv,
                                                &ucell,
                                                &band_ppcell,
                                                this->pw_rho,
                                                this->pw_big);
    band_elec.pot = this->pelec->pot;
    band_elec.skip_weights = true;
    band_elec.wg.zero_out();

    hsolver::setup_diago_params_pw<T, Device>(0, 1, target_ethr, PARAM.inp);
    hsolver::HSolverPW<T, Device> hsolver_pw_obj(band_pw_wfc,
                                                 "nscf",
                                                 PARAM.inp.basis_type,
                                                 PARAM.inp.ks_solver,
                                                 PARAM.globalv.use_uspp,
                                                 PARAM.inp.nspin,
                                                 hsolver::DiagoIterAssist<T, Device>::SCF_ITER,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_NMAX,
                                                 hsolver::DiagoIterAssist<T, Device>::PW_DIAG_THR,
                                                 hsolver::DiagoIterAssist<T, Device>::need_subspace,
                                                 false);

    for (int ik = 0; ik < band_pw_wfc->nks; ++ik)
    {
        band_stp.init_ik(band_hamilt, ik);
        hsolver_pw_obj.solve_ik(static_cast<hamilt::Hamilt<T, Device>*>(band_hamilt),
                                *band_stp.template get_psi_t<T, Device>(),
                                &band_elec,
                                band_ekb.c + ik * PARAM.globalv.nbands_l,
                                ik,
                                GlobalV::RANK_IN_POOL,
                                GlobalV::NPROC_IN_POOL,
                                true);
    }

    band_elec.pot = nullptr;
    delete band_hamilt;
    band_stp.clean();
    pw::teardown_pwwfc(band_pw_wfc);

    ModuleIO::write_bands(PARAM.inp, band_ekb, band_kv);
}

template <typename T, typename Device>
double ESolver_KS_PW<T, Device>::cal_energy()
{
    return this->pelec->f_en.etot;
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_force(UnitCell& ucell, ModuleBase::matrix& force)
{
    Forces<double, Device> ff(ucell.nat);

    // mohan add 2025-10-12
    this->stp.update_psi_d();

    // Calculate forces
    ff.cal_force(ucell, force, *this->pelec, this->pw_rhod, &ucell.symm,
                 &this->sf, this->solvent, &this->dftu, &this->locpp, &this->ppcell, 
                 &this->kv, this->pw_wfc, this->stp.template get_psi_d<T, Device>());
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::cal_stress(UnitCell& ucell, ModuleBase::matrix& stress)
{
    Stress_PW<double, Device> ss(this->pelec);

    // mohan add 2025-10-12
    this->stp.update_psi_d();

    ss.cal_stress(stress, ucell, this->dftu, this->locpp, this->ppcell, this->pw_rhod,
                  &ucell.symm, &this->sf, &this->kv, this->pw_wfc, this->stp.template get_psi_d<T, Device>());

    // external stress
    double unit_transform = 0.0;
    unit_transform = ModuleBase::RYDBERG_SI / pow(ModuleBase::BOHR_RADIUS_SI, 3) * 1.0e-8;
    double external_stress[3] = {PARAM.inp.press1, PARAM.inp.press2, PARAM.inp.press3};
    for (int i = 0; i < 3; i++)
    {
        stress(i, i) -= external_stress[i] / unit_transform;
    }
}

template <typename T, typename Device>
void ESolver_KS_PW<T, Device>::after_all_runners(UnitCell& ucell)
{
    ESolver_KS::after_all_runners(ucell);

    ModuleIO::ctrl_runner_pw<T, Device>(ucell, this->pelec, this->pw_wfc, 
            this->pw_rho, this->pw_rhod, this->chr, this->kv, this->stp, 
            this->sf, this->ppcell, this->solvent, this->Pgrid, PARAM.inp); 

    elecstate::teardown_estate_pw(this->pelec, this->vsep_cell);
    
}

template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif
} // namespace ModuleESolver

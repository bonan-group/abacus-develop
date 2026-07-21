#include "elecstate_pw.h"

#include "source_base/constants.h"
#include "source_base/libm/libm.h"
#include "source_base/math_ylmreal.h"
#include "source_base/module_device/device.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_device.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/vnl_pw.h"

#include <algorithm>

namespace elecstate {

template <typename T, typename Device>
ElecStatePW<T, Device>::ElecStatePW(ModulePW::PW_Basis_K* wfc_basis_in,
                                    Charge* chr_in,
                                    K_Vectors* pkv_in,
                                    UnitCell* ucell_in,
                                    pseudopot_cell_vnl* ppcell_in,
                                    ModulePW::PW_Basis* rhopw_in,
                                    ModulePW::PW_Basis_Big* bigpw_in)
    : basis(wfc_basis_in)
{
    this->classname = "ElecStatePW";
    this->rhopw_smooth = rhopw_in;
    this->ppcell = ppcell_in;
    this->ucell = ucell_in;
    this->init_ks(chr_in, pkv_in, pkv_in->get_nks(), bigpw_in);
}

template<typename T, typename Device>
ElecStatePW<T, Device>::~ElecStatePW() 
{
    if (PARAM.inp.device == "gpu" || PARAM.inp.precision == "single")
    {
        delmem_var_op()(this->rho_data);
        delete[] this->rho;

        if (PARAM.globalv.double_grid || PARAM.globalv.use_uspp)
        {
            delmem_complex_op()(this->rhog_data);
            delete[] this->rhog;
        }
        if (this->charge->kin_density())
        {
            delmem_var_op()(this->kin_r_data);
            delete[] this->kin_r;
        }
    }
    if (this->kin_r_smooth != nullptr)
    {
        delmem_var_op()(this->kin_r_smooth_data);
        delmem_complex_op()(this->tau_g_data);
        delete[] this->kin_r_smooth;
    }
    if (PARAM.globalv.use_uspp)
    {
        delmem_var_op()(this->becsum);
    }
    delmem_complex_op()(this->wfcr);
    delmem_complex_op()(this->wfcr_another_spin);
    delmem_complex_op()(this->uspp_aux);
    delmem_complex_op()(this->uspp_packed_becsum);
}

template<typename T, typename Device>
void ElecStatePW<T, Device>::init_rho_data()
{
    if (this->init_rho)
    {
        return;
    }

    // Set device mode on Charge object (hybrid pattern like PW_Basis)
    if (std::is_same<Device, base_device::DEVICE_GPU>::value)
    {
        this->charge->set_device("gpu");
    }

    if (PARAM.inp.device == "gpu" || PARAM.inp.precision == "single")
    {
        this->rho = new Real*[this->charge->nspin];
        resmem_var_op()(this->rho_data, this->charge->nspin * this->charge->nrxx);
        for (int ii = 0; ii < this->charge->nspin; ii++)
        {
            this->rho[ii] = this->rho_data + ii * this->charge->nrxx;
        }
        if (PARAM.globalv.double_grid || PARAM.globalv.use_uspp)
        {
            this->rhog = new T*[this->charge->nspin];
            resmem_complex_op()(this->rhog_data, this->charge->nspin * this->charge->rhopw->npw);
            for (int ii = 0; ii < this->charge->nspin; ii++)
            {
                this->rhog[ii] = this->rhog_data + ii * this->charge->rhopw->npw;
            }
        }
        if (this->charge->kin_density())
        {
            this->kin_r = new Real*[this->charge->nspin];
            resmem_var_op()(this->kin_r_data, this->charge->nspin * this->charge->nrxx);
            for (int ii = 0; ii < this->charge->nspin; ii++) {
                this->kin_r[ii] = this->kin_r_data + ii * this->charge->nrxx;
            }
            if (this->charge->rhopw != this->rhopw_smooth)
            {
                this->kin_r_smooth = new Real*[this->charge->nspin];
                resmem_var_op()(this->kin_r_smooth_data,
                                this->charge->nspin * this->rhopw_smooth->nrxx,
                                "ElecSPW::kin_r_smooth");
                resmem_complex_op()(this->tau_g_data,
                                    this->charge->nspin * this->charge->rhopw->npw,
                                    "ElecSPW::tau_g");
                for (int ii = 0; ii < this->charge->nspin; ++ii)
                {
                    this->kin_r_smooth[ii]
                        = this->kin_r_smooth_data + ii * this->rhopw_smooth->nrxx;
                }
            }
        }
    }
    else
    {
        this->rho = reinterpret_cast<Real **>(this->charge->rho);
        if (PARAM.globalv.double_grid || PARAM.globalv.use_uspp)
        {
            this->rhog = reinterpret_cast<T**>(this->charge->rhog);
        }
        if (this->charge->kin_density())
        {
            this->kin_r = reinterpret_cast<Real **>(this->charge->kin_r);
        }
    }
    if (this->charge->kin_density() && this->charge->rhopw != this->rhopw_smooth
        && this->kin_r_smooth == nullptr)
    {
        this->kin_r_smooth = new Real*[this->charge->nspin];
        resmem_var_op()(this->kin_r_smooth_data,
                        this->charge->nspin * this->rhopw_smooth->nrxx,
                        "ElecSPW::kin_r_smooth");
        resmem_complex_op()(this->tau_g_data,
                            this->charge->nspin * this->charge->rhopw->npw,
                            "ElecSPW::tau_g");
        for (int ii = 0; ii < this->charge->nspin; ++ii)
        {
            this->kin_r_smooth[ii] = this->kin_r_smooth_data + ii * this->rhopw_smooth->nrxx;
        }
    }
    resmem_complex_op()(this->wfcr, this->basis->nmaxgr, "ElecSPW::wfcr");
    resmem_complex_op()(this->wfcr_another_spin, this->basis->nrxx, "ElecSPW::wfcr_a");
    if (this->ppcell != nullptr && this->ppcell->has_qgm_cache())
    {
        int max_nij = 0;
        int max_atoms = 0;
        for (int it = 0; it < this->ucell->ntype; ++it)
        {
            const int nh = this->ucell->atoms[it].ncpp.nh;
            max_nij = std::max(max_nij, nh * (nh + 1) / 2);
            max_atoms = std::max(max_atoms, this->ucell->atoms[it].na);
        }
        resmem_complex_op()(this->uspp_aux,
                            max_nij * this->charge->rhopw->npw,
                            "ElecSPW::uspp_aux");
        resmem_complex_op()(this->uspp_packed_becsum,
                            this->charge->nspin * max_atoms * max_nij,
                            "ElecSPW::uspp_packed_becsum");
    }
    this->init_rho = true;
}

template<typename T, typename Device>
void ElecStatePW<T, Device>::psiToRho(const psi::Psi<T, Device>& psi)
{
    ModuleBase::TITLE("ElecStatePW", "psiToRho");
    ModuleBase::timer::start("ElecStatePW", "psiToRho");

    this->init_rho_data();

    for(int is=0; is<PARAM.inp.nspin; is++)
	{
        // denghui replaced at 20221110
		// ModuleBase::GlobalFunc::ZEROS(this->rho[is], this->charge->nrxx);
        setmem_var_op()(this->rho[is], 0,  this->charge->nrxx);
        if (XC_Functional::get_ked_flag())
        {
            // ModuleBase::GlobalFunc::ZEROS(this->charge->kin_r[is], this->charge->nrxx);
            setmem_var_op()(this->kin_r[is], 0,  this->charge->nrxx);
            if (this->kin_r_smooth != nullptr)
            {
                setmem_var_op()(this->kin_r_smooth[is], 0, this->rhopw_smooth->nrxx);
            }
        }
        if (PARAM.globalv.double_grid || PARAM.globalv.use_uspp)
        {
            setmem_complex_op()(this->rhog[is], 0, this->charge->rhopw->npw);
        }
    }

    for (int ik = 0; ik < psi.get_nk(); ++ik)
    {
        psi.fix_k(ik);
        this->updateRhoK(psi);
    }

    this->finalize_tau_double_grid();

    this->add_usrho(psi);

    if (PARAM.inp.device == "gpu" || PARAM.inp.precision == "single")
    {
        for (int ii = 0; ii < PARAM.inp.nspin; ii++)
        {
            castmem_var_d2h_op()(this->charge->rho[ii], this->rho[ii], this->charge->nrxx);
            if (XC_Functional::get_ked_flag())
            {
                castmem_var_d2h_op()(this->charge->kin_r[ii], this->kin_r[ii], this->charge->nrxx);
            }
        }
        // Sync rho from host to Charge's device memory for use by other modules (e.g., Charge_Mixing)
        if (std::is_same<Device, base_device::DEVICE_GPU>::value)
        {
            this->charge->sync_rho_to_device();
            if (XC_Functional::get_ked_flag())
            {
                this->charge->sync_kin_r_and_save_to_device();
            }
        }
    }
    this->parallelK();
    this->charge->sync_rho_to_device();
    this->charge->sync_kin_r_and_save_to_device();
    ModuleBase::timer::end("ElecStatePW", "psiToRho");
}

template<typename T, typename Device>
void ElecStatePW<T, Device>::updateRhoK(const psi::Psi<T, Device>& psi)
{
    this->rhoBandK(psi);
}

template<typename T, typename Device>
void ElecStatePW<T, Device>::parallelK()
{
#ifdef __MPI
    this->charge->rho_mpi();
#endif
}

template<typename T, typename Device>
void ElecStatePW<T, Device>::rhoBandK(const psi::Psi<T, Device>& psi)
{
    ModuleBase::TITLE("ElecStatePW", "rhoBandK");

    // moved by denghui to constructor at 20221110
    // used for plane wavefunction FFT3D to real space
    // static std::vector<T> wfcr;
    // wfcr.resize(this->basis->nmaxgr);
    // used for plane wavefunction FFT3D to real space, non-collinear spin case
    // static std::vector<std::complex<double>> wfcr_another_spin;
    // if (PARAM.inp.nspin == 4)
    //     wfcr_another_spin.resize(this->charge->nrxx);

    this->init_rho_data();
    int ik = psi.get_current_k();
    int npw = psi.get_current_ngk();
    int current_spin = 0;
    if (PARAM.inp.nspin == 2)
    {
        current_spin = this->klist->isk[ik];
    }
    int nbands = psi.get_nbands();
    //  here we compute the band energy: the sum of the eigenvalues
    if (PARAM.inp.nspin == 4)
    {
        int npwx = npw / 2;
        for (int ibnd = 0; ibnd < nbands; ibnd++)
        {
            ///
            /// only occupied band should be calculated.
            /// be care of when smearing_sigma is large, wg would less than 0
            ///

            this->basis->recip_to_real(this->ctx, &psi(ibnd,0), this->wfcr, ik);

            this->basis->recip_to_real(this->ctx, &psi(ibnd,npwx), this->wfcr_another_spin, ik);

            const auto w1 = static_cast<Real>(this->wg(ik, ibnd) / ucell->omega);

            if (w1 != 0.0)
            {
                // replaced by denghui at 20221110
                elecstate_pw_op()(this->ctx,
                                  PARAM.globalv.domag,
                                  PARAM.globalv.domag_z,
                                  this->basis->nrxx,
                                  w1,
                                  this->rho,
                                  this->wfcr,
                                  this->wfcr_another_spin);
            }
        }
    }
    else
    {
        for (int ibnd = 0; ibnd < nbands; ibnd++)
        {
            ///
            /// only occupied band should be calculated.
            ///

            this->basis->recip_to_real(this->ctx, &psi(ibnd,0), this->wfcr, ik);

            const auto w1 = static_cast<Real>(this->wg(ik, ibnd) / ucell->omega);

            if (w1 != 0.0)
            {
                // replaced by denghui at 20221110
                elecstate_pw_op()(this->ctx, current_spin, this->basis->nrxx, w1, this->rho, this->wfcr);
            }

            // kinetic energy density
            if (XC_Functional::get_ked_flag())
            {
                for (int j = 0; j < 3; j++)
                {
                    setmem_complex_op()(this->wfcr, 0, this->basis->nmaxgr);

                    meta_op()(this->ctx,
                              ik,
                              j,
                              npw,
                              this->basis->npwk_max,
                              static_cast<Real>(ucell->tpiba),
                              this->basis->template get_gcar_data<Real>(),
                              this->basis->template get_kvec_c_data<Real>(),
                              &psi(ibnd, 0),
                              this->wfcr);

                    this->basis->recip_to_real(this->ctx, this->wfcr, this->wfcr, ik);

                    Real** tau = this->kin_r_smooth != nullptr ? this->kin_r_smooth : this->kin_r;
                    const int tau_nrxx
                        = this->kin_r_smooth != nullptr ? this->rhopw_smooth->nrxx : this->charge->nrxx;
                    elecstate_pw_op()(this->ctx, current_spin, tau_nrxx, w1, tau, this->wfcr);
                }
            }
        }
    }
}

template <typename T, typename Device>
void ElecStatePW<T, Device>::finalize_tau_double_grid()
{
    if (!XC_Functional::get_ked_flag() || this->kin_r_smooth == nullptr)
    {
        return;
    }
    for (int is = 0; is < this->charge->nspin; ++is)
    {
        T* tau_g = this->tau_g_data + is * this->charge->rhopw->npw;
        setmem_complex_op()(tau_g, 0, this->charge->rhopw->npw);
        this->rhopw_smooth->real_to_recip<Real, T, Device>(this->kin_r_smooth[is], tau_g);
        this->charge->rhopw->recip_to_real<T, Real, Device>(tau_g, this->kin_r[is]);
    }
}

template <typename T, typename Device>
void ElecStatePW<T, Device>::cal_becsum(const psi::Psi<T, Device>& psi)
{
    const T one{1, 0};
    const T zero{0, 0};
    const int npol = psi.get_npol();
    const int npwx = psi.get_nbasis() / npol;
    const int nbands = psi.get_nbands() * npol;
    const int nkb = this->ppcell->nkb;
    this->vkb = this->ppcell->template get_vkb_data<Real>();
    T* becp = nullptr;
    Real* weights = nullptr;
    resmem_complex_op()(becp, nbands * nkb, "ElecState<PW>::becp");
    resmem_var_op()(weights, nbands, "ElecState<PW>::weights");
    const int nh_tot = this->ppcell->nhm * (this->ppcell->nhm + 1) / 2;
    resmem_var_op()(becsum, nh_tot * ucell->nat * PARAM.inp.nspin, "ElecState<PW>::becsum");
    setmem_var_op()(becsum, 0, nh_tot * ucell->nat * PARAM.inp.nspin);

    for (int ik = 0; ik < psi.get_nk(); ++ik)
    {
        psi.fix_k(ik);
        const T* psi_now = psi.get_pointer();
        const int currect_spin = this->klist->isk[ik];
        const int npw = psi.get_current_ngk();

        const bool has_full_vkb = sizeof(T) == sizeof(std::complex<float>)
                                      ? this->ppcell->has_full_float_vkb
                                      : this->ppcell->has_full_double_vkb;
        if (!has_full_vkb && std::is_same<Device, base_device::DEVICE_GPU>::value)
        {
            this->ppcell->cal_becp_matrix_free<Real, Device>(
                this->ctx, *ucell, ik, npw, npwx, nbands, psi_now, becp);
        }
        else
        {
            // get |beta>
            if (this->ppcell->nkb > 0)
            {
                this->ppcell->getvnl(this->ctx, *ucell,ik, this->vkb);
            }

            // becp = <beta|psi>
            if (nbands == 1)
            {
                const int inc = 1;
                gemv_op()('C',
                          npw,
                          this->ppcell->nkb,
                          &one,
                          this->vkb,
                          this->ppcell->vkbnc,
                          psi_now,
                          inc,
                          &zero,
                          becp,
                          inc);
            }
            else
            {
                gemm_op()('C',
                          'N',
                          this->ppcell->nkb,
                          nbands,
                          npw,
                          &one,
                          this->vkb,
                          this->ppcell->vkbnc,
                          psi_now,
                          npwx,
                          &zero,
                          becp,
                          this->ppcell->nkb);
            }
        }
        #ifdef __MPI
        Parallel_Common::reduce_dev<T, Device>(becp, this->ppcell->nkb * nbands, POOL_WORLD);
        #endif

        std::vector<Real> host_weights(nbands);
        for (int ib = 0; ib < nbands; ++ib)
        {
            host_weights[ib] = static_cast<Real>(this->wg(ik, ib));
        }
        syncmem_var_h2d_op()(weights, host_weights.data(), nbands);

        // sum over bands: \sum_i <psi_i|beta_l><beta_m|psi_i> w_i
        for (int it = 0; it < ucell->ntype; it++)
        {
            Atom* atom = &ucell->atoms[it];
            if (atom->ncpp.tvanp)
            {
                const int first_iat = ucell->itia2iat(it, 0);
                uspp_becsum_op<Real, Device>()(this->ctx,
                                               atom->na,
                                               nbands,
                                               atom->ncpp.nh,
                                               this->ppcell->nkb,
                                               this->ppcell->indv_ijkb0[first_iat],
                                               first_iat,
                                               ucell->nat,
                                               nh_tot,
                                               currect_spin,
                                               weights,
                                               becp,
                                               becsum);
            }
        }
    }
    delmem_var_op()(weights);
    delmem_complex_op()(becp);
}

template <typename T, typename Device>
void ElecStatePW<T, Device>::add_usrho(const psi::Psi<T, Device>& psi)
{
    if (PARAM.globalv.use_uspp)
    {
        this->cal_becsum(psi);
    }

    // transform soft charge to recip space using smooth grids
    if (PARAM.globalv.double_grid || PARAM.globalv.use_uspp)
    {
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            this->rhopw_smooth->real_to_recip<Real, T, Device>(this->rho[is], this->rhog[is]);
        }
    }

    // \sum_lm Q_lm(r) \sum_i <psi_i|beta_l><beta_m|psi_i> w_i
    // add to the charge density in reciprocal space the part which is due to the US augmentation.
    if (PARAM.globalv.use_uspp)
    {
        this->addusdens_g(becsum, rhog);
    }
    // transform back to real space using dense grids
    if (PARAM.globalv.double_grid || PARAM.globalv.use_uspp)
    {
        for (int is = 0; is < PARAM.inp.nspin; is++)
        {
            this->charge->rhopw->recip_to_real<T, Real, Device>(this->rhog[is], this->rho[is]);
        }
    }
}

template <typename T, typename Device>
void ElecStatePW<T, Device>::addusdens_g(const Real* becsum, T** rhog)
{
    const T one{1, 0};
    const T zero{0, 0};
    const int npw = this->charge->rhopw->npw;
    const int nh_tot = this->ppcell->nhm * (this->ppcell->nhm + 1) / 2;
    const T* qgm_all = this->ppcell->template get_qgm_data<Real>();
    const T* phase_all = this->ppcell->template get_qgm_phase_data<Real>();

    for (int it = 0; it < ucell->ntype; it++)
    {
        Atom* atom = &ucell->atoms[it];
        if (atom->ncpp.tvanp)
        {
            // nij = max number of (ih,jh) pairs per atom type nt
            const int nij = atom->ncpp.nh * (atom->ncpp.nh + 1) / 2;

            T* aux2 = this->uspp_aux;
            T* tbecsum = this->uspp_packed_becsum;
            const int first_iat = ucell->itia2iat(it, 0);
            const T* skk = phase_all + first_iat * npw;

            for (int is = 0; is < PARAM.inp.nspin; is++)
            {
                uspp_pack_becsum_op<Real, Device>()(this->ctx,
                                                    atom->na,
                                                    nij,
                                                    ucell->nat,
                                                    nh_tot,
                                                    is,
                                                    first_iat,
                                                    becsum,
                                                    &tbecsum[is * atom->na * nij]);
                // sum over atoms
                char transa = 'N';
                char transb = 'T';
                gemm_op()(transa,
                          transb,
                          npw,
                          nij,
                          atom->na,
                          &one,
                          skk,
                          npw,
                          &tbecsum[is * atom->na * nij],
                          nij,
                          &zero,
                          aux2,
                          npw);

                uspp_accumulate_rhog_op<Real, Device>()(this->ctx,
                                                        npw,
                                                        nij,
                                                        qgm_all + it * nh_tot * npw,
                                                        aux2,
                                                        rhog[is]);
            }
        }
    }
}

template class ElecStatePW<std::complex<float>, base_device::DEVICE_CPU>;
template class ElecStatePW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ElecStatePW<std::complex<float>, base_device::DEVICE_GPU>;
template class ElecStatePW<std::complex<double>, base_device::DEVICE_GPU>;
#endif 

} // namespace elecstate

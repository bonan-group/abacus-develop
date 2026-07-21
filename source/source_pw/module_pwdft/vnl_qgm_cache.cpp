#include "vnl_pw.h"

#include "source_base/math_ylmreal.h"
#include "source_estate/kernels/elecstate_op.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "source_pw/module_pwdft/nonlocal_maths.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

void pseudopot_cell_vnl::prepare_qgm_cache(UnitCell& cell,
                                           const ModulePW::PW_Basis* rho_basis,
                                           bool prepare_stress,
                                           int nqxq,
                                           double dq)
{
    this->qgm_cache_ready = false;
    if (this->use_gpu_)
    {
        this->prepare_qgm_cache_gpu(cell, rho_basis, prepare_stress, nqxq, dq);
    }
    else
    {
        this->prepare_qgm_cache_cpu(cell, rho_basis, nqxq, dq);
    }
    this->qgm_cache_ready = true;
}

void pseudopot_cell_vnl::prepare_qgm_cache_gpu(UnitCell& cell,
                                               const ModulePW::PW_Basis* rho_basis,
                                               bool prepare_stress,
                                               int nqxq,
                                               double dq)
{
    const int npw = rho_basis->npw;
    const int nh_tot = this->nhm * (this->nhm + 1) / 2;
    const int pair_count = cell.ntype * nh_tot;
    const int radial_pair_count = this->nbetam * (this->nbetam + 1) / 2;
    const int max_terms = this->lmaxq * this->lmaxq;
    const size_t qgm_size = static_cast<size_t>(pair_count) * npw;
    const size_t phase_size = static_cast<size_t>(cell.nat) * npw;

    std::vector<int> pair_term_count(pair_count, 0);
    std::vector<int> pair_radial_index(pair_count, 0);
    std::vector<int> pair_l(pair_count * max_terms, 0);
    std::vector<int> pair_lm(pair_count * max_terms, 0);
    std::vector<std::complex<double>> pair_coefficient(pair_count * max_terms, {0.0, 0.0});
    for (int it = 0; it < cell.ntype; ++it)
    {
        const Atom_pseudo& upf = cell.atoms[it].ncpp;
        if (!upf.tvanp)
        {
            continue;
        }
        int ijh = 0;
        for (int ih = 0; ih < upf.nh; ++ih)
        {
            for (int jh = ih; jh < upf.nh; ++jh, ++ijh)
            {
                const int pair = it * nh_tot + ijh;
                const int nb = static_cast<int>(this->indv(it, ih));
                const int mb = static_cast<int>(this->indv(it, jh));
                const int larger = std::max(nb, mb);
                const int smaller = std::min(nb, mb);
                pair_radial_index[pair] = larger * (larger + 1) / 2 + smaller;
                const int ivl = static_cast<int>(this->nhtolm(it, ih));
                const int jvl = static_cast<int>(this->nhtolm(it, jh));
                pair_term_count[pair] = this->lpx(ivl, jvl);
                for (int term = 0; term < pair_term_count[pair]; ++term)
                {
                    const int term_index = pair * max_terms + term;
                    const int lm = this->lpl(ivl, jvl, term);
                    int l = 0;
                    while ((l + 1) * (l + 1) <= lm)
                    {
                        ++l;
                    }
                    pair_l[term_index] = l;
                    pair_lm[term_index] = lm;
                    pair_coefficient[term_index]
                        = std::pow(ModuleBase::NEG_IMAG_UNIT, l) * this->ap(lm, ivl, jvl);
                }
            }
        }
    }

    std::vector<double> tau(cell.nat * 3);
    for (int it = 0; it < cell.ntype; ++it)
    {
        for (int ia = 0; ia < cell.atoms[it].na; ++ia)
        {
            const int iat = cell.itia2iat(it, ia);
            for (int ipol = 0; ipol < 3; ++ipol)
            {
                tau[3 * iat + ipol] = cell.atoms[it].tau[ia][ipol];
            }
        }
    }

    int* d_pair_term_count = nullptr;
    int* d_pair_radial_index = nullptr;
    int* d_pair_l = nullptr;
    int* d_pair_lm = nullptr;
    double* d_qrad = nullptr;
    double* d_ylm = nullptr;
    double* d_tau = nullptr;
    std::complex<double>* d_pair_coefficient = nullptr;
    using resmem_int_op = base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>;
    using delmem_int_op = base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>;
    using syncmem_int_op
        = base_device::memory::synchronize_memory_op<int,
                                                     base_device::DEVICE_GPU,
                                                     base_device::DEVICE_CPU>;
    using castmem_z2c_d2d_op
        = base_device::memory::cast_memory_op<std::complex<float>,
                                              std::complex<double>,
                                              base_device::DEVICE_GPU,
                                              base_device::DEVICE_GPU>;

    resmem_dd_op()(this->d_qgm_gcar, npw * 3, "VNL::d_qgm_gcar");
    syncmem_d2d_h2d_op()(this->d_qgm_gcar,
                         reinterpret_cast<const double*>(rho_basis->gcar),
                         npw * 3);
    resmem_zd_op()(this->z_qgm, qgm_size, "VNL::z_qgm");
    resmem_zd_op()(this->z_qgm_phase, phase_size, "VNL::z_qgm_phase");
    resmem_int_op()(d_pair_term_count, pair_term_count.size(), "VNL::qgm_term_count");
    resmem_int_op()(d_pair_radial_index, pair_radial_index.size(), "VNL::qgm_radial_index");
    resmem_int_op()(d_pair_l, pair_l.size(), "VNL::qgm_term_l");
    resmem_int_op()(d_pair_lm, pair_lm.size(), "VNL::qgm_term_lm");
    resmem_zd_op()(d_pair_coefficient, pair_coefficient.size(), "VNL::qgm_coefficient");
    resmem_dd_op()(d_qrad, this->qrad.getSize(), "VNL::qrad_setup");
    resmem_dd_op()(d_ylm, max_terms * npw, "VNL::ylm_setup");
    resmem_dd_op()(d_tau, tau.size(), "VNL::tau_setup");
    syncmem_int_op()(d_pair_term_count, pair_term_count.data(), pair_term_count.size());
    syncmem_int_op()(d_pair_radial_index, pair_radial_index.data(), pair_radial_index.size());
    syncmem_int_op()(d_pair_l, pair_l.data(), pair_l.size());
    syncmem_int_op()(d_pair_lm, pair_lm.data(), pair_lm.size());
    syncmem_z2z_h2d_op()(d_pair_coefficient, pair_coefficient.data(), pair_coefficient.size());
    syncmem_d2d_h2d_op()(d_qrad, this->qrad.ptr, this->qrad.getSize());
    syncmem_d2d_h2d_op()(d_tau, tau.data(), tau.size());

    ModuleBase::YlmReal::Ylm_Real(gpu_ctx,
                                  max_terms,
                                  npw,
                                  this->d_qgm_gcar,
                                  d_ylm);
    hamilt::uspp_qgm_build_op<double, base_device::DEVICE_GPU>()(gpu_ctx,
                                                                 cell.ntype,
                                                                 nh_tot,
                                                                 npw,
                                                                 this->lmaxq,
                                                                 radial_pair_count,
                                                                 nqxq,
                                                                 max_terms,
                                                                 dq,
                                                                 cell.tpiba,
                                                                 this->d_qgm_gcar,
                                                                 d_pair_term_count,
                                                                 d_pair_radial_index,
                                                                 d_pair_l,
                                                                 d_pair_lm,
                                                                 d_pair_coefficient,
                                                                 d_qrad,
                                                                 d_ylm,
                                                                 this->z_qgm);
    elecstate::uspp_atom_phase_op<double, base_device::DEVICE_GPU>()(gpu_ctx,
                                                                     cell.nat,
                                                                     npw,
                                                                     this->d_qgm_gcar,
                                                                     d_tau,
                                                                     this->z_qgm_phase);

    if (this->s_deeq != nullptr)
    {
        resmem_cd_op()(this->c_qgm, qgm_size, "VNL::c_qgm");
        castmem_z2c_d2d_op()(this->c_qgm, this->z_qgm, qgm_size);
        resmem_cd_op()(this->c_qgm_phase, phase_size, "VNL::c_qgm_phase");
        castmem_z2c_d2d_op()(this->c_qgm_phase, this->z_qgm_phase, phase_size);
    }

    if (prepare_stress)
    {
        this->dqgm.create(3, cell.ntype, nh_tot, npw);
        ModuleBase::matrix ylmk0(max_terms, npw);
        ModuleBase::YlmReal::Ylm_Real(max_terms, npw, rho_basis->gcar, ylmk0);
        std::vector<double> qnorm(npw);
        for (int ig = 0; ig < npw; ++ig)
        {
            qnorm[ig] = rho_basis->gcar[ig].norm() * cell.tpiba;
        }
        ModuleBase::matrix dylmk0(max_terms, npw);
        for (int ipol = 0; ipol < 3; ++ipol)
        {
            hamilt::Nonlocal_maths<double, base_device::DEVICE_CPU>::dylmr2(
                max_terms,
                npw,
                reinterpret_cast<const double*>(rho_basis->gcar),
                dylmk0.c,
                ipol);
            for (int it = 0; it < cell.ntype; ++it)
            {
                const Atom_pseudo& upf = cell.atoms[it].ncpp;
                if (!upf.tvanp)
                {
                    continue;
                }
                int ijh = 0;
                for (int ih = 0; ih < upf.nh; ++ih)
                {
                    for (int jh = ih; jh < upf.nh; ++jh, ++ijh)
                    {
                        this->radial_fft_dq_explicit(npw,
                                                     ih,
                                                     jh,
                                                     it,
                                                     ipol,
                                                     rho_basis->gcar,
                                                     qnorm.data(),
                                                     cell.tpiba,
                                                     ylmk0,
                                                     dylmk0,
                                                     nqxq,
                                                     dq,
                                                     &this->dqgm(ipol, it, ijh, 0));
                    }
                }
            }
        }
        resmem_zd_op()(this->z_dqgm, this->dqgm.getSize(), "VNL::z_dqgm");
        syncmem_z2z_h2d_op()(this->z_dqgm, this->dqgm.ptr, this->dqgm.getSize());
    }

    delmem_int_op()(d_pair_term_count);
    delmem_int_op()(d_pair_radial_index);
    delmem_int_op()(d_pair_l);
    delmem_int_op()(d_pair_lm);
    delmem_zd_op()(d_pair_coefficient);
    delmem_dd_op()(d_qrad);
    delmem_dd_op()(d_ylm);
    delmem_dd_op()(d_tau);
}

void pseudopot_cell_vnl::prepare_qgm_cache_cpu(UnitCell& cell,
                                               const ModulePW::PW_Basis* rho_basis,
                                               int nqxq,
                                               double dq)
{
    const int npw = rho_basis->npw;
    const int nh_tot = this->nhm * (this->nhm + 1) / 2;
    this->qgm.create(cell.ntype, nh_tot, npw);
    this->qgm.zero_out();
    this->qgm_phase.create(cell.nat, npw);
    this->qgm_gcar.create(npw, 3);
    ModuleBase::matrix ylmk0(lmaxq * lmaxq, npw);
    ModuleBase::YlmReal::Ylm_Real(lmaxq * lmaxq, npw, rho_basis->gcar, ylmk0);
    std::vector<double> qnorm(npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        qnorm[ig] = rho_basis->gcar[ig].norm() * cell.tpiba;
        for (int ipol = 0; ipol < 3; ++ipol)
        {
            this->qgm_gcar(ig, ipol) = rho_basis->gcar[ig][ipol];
        }
    }
    for (int it = 0; it < cell.ntype; ++it)
    {
        const Atom_pseudo& upf = cell.atoms[it].ncpp;
        if (upf.tvanp)
        {
            int ijh = 0;
            for (int ih = 0; ih < upf.nh; ++ih)
            {
                for (int jh = ih; jh < upf.nh; ++jh, ++ijh)
                {
                    this->radial_fft_q_explicit(npw,
                                                ih,
                                                jh,
                                                it,
                                                qnorm.data(),
                                                ylmk0,
                                                nqxq,
                                                dq,
                                                &this->qgm(it, ijh, 0));
                }
            }
        }
        for (int ia = 0; ia < cell.atoms[it].na; ++ia)
        {
            const int iat = cell.itia2iat(it, ia);
            for (int ig = 0; ig < npw; ++ig)
            {
                const std::complex<double> exponent
                    = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI
                      * (rho_basis->gcar[ig] * cell.atoms[it].tau[ia]);
                this->qgm_phase(iat, ig) = std::exp(exponent);
            }
        }
    }
    if (this->s_deeq != nullptr)
    {
        resmem_ch_op()(this->c_qgm, this->qgm.getSize(), "VNL::c_qgm");
        castmem_z2c_h2h_op()(this->c_qgm, this->qgm.ptr, this->qgm.getSize());
        resmem_ch_op()(this->c_qgm_phase, this->qgm_phase.size, "VNL::c_qgm_phase");
        castmem_z2c_h2h_op()(this->c_qgm_phase, this->qgm_phase.c, this->qgm_phase.size);
    }
    this->z_qgm = this->qgm.ptr;
    this->z_qgm_phase = this->qgm_phase.c;
    this->d_qgm_gcar = this->qgm_gcar.c;
}

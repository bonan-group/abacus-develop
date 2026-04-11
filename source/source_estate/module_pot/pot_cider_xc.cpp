#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_hamilt/module_xc/xc_functional_libxc.h"
#include "source_io/module_parameter/parameter.h"

namespace elecstate
{

PotCiderXC::PotCiderXC(
    const ModulePW::PW_Basis* rho_basis_in,
    const UnitCell* ucell_in,
    double* etxc_in,
    double* vtxc_in,
    ModuleBase::matrix* vofk_in)
    : etxc_(etxc_in), vtxc_(vtxc_in), vofk_(vofk_in)
{
    this->rho_basis_ = rho_basis_in;
    this->dynamic_mode = true;
    this->fixed_mode = false;

    // Grid dimensions
    const int N_c[3] = {rho_basis_in->nx, rho_basis_in->ny, rho_basis_in->nz};

    // Lattice vectors in row-major order (Bohr units)
    const auto& lat = ucell_in->latvec;
    double cell_cv[9] = {
        lat.e11, lat.e12, lat.e13,
        lat.e21, lat.e22, lat.e23,
        lat.e31, lat.e32, lat.e33
    };

    const std::string& model_path = PARAM.inp.cider_model;
    if (model_path.empty()) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "cider_model path is empty");
    }

    ctx_ = cider_bridge_create(model_path.c_str(), N_c, cell_cv,
                               PARAM.inp.nspin, PARAM.inp.cider_xmix);
    if (!ctx_) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "Failed to create CIDER bridge context");
    }
    is_mgga_ = cider_bridge_is_mgga(ctx_);
}

PotCiderXC::~PotCiderXC()
{
    if (ctx_) {
        cider_bridge_destroy(&ctx_);
    }
}

void PotCiderXC::cal_v_eff(
    const Charge* const chg,
    const UnitCell* const ucell,
    ModuleBase::matrix& v_eff)
{
    ModuleBase::TITLE("PotCiderXC", "cal_v_eff");
    ModuleBase::timer::tick("PotCiderXC", "cal_v_eff");

    const int nspin = chg->nspin;
    const std::size_t nrxx = chg->nrxx;
    const double tpiba = ucell->tpiba;

    // === 1. Build rho in interleaved layout for gradient computation ===
    // (ABACUS libxc functions use interleaved: rho[ir*nspin+is])
    std::vector<double> rho_interleaved(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            rho_interleaved[ir * nspin + is] = chg->rho[is][ir] + 1.0 / nspin * chg->rho_core[ir];
        }
    }

    // === 2. Build sigma using ABACUS gradient machinery ===
    auto gdr = XC_Functional_Libxc::cal_gdr(nspin, nrxx, rho_interleaved, tpiba, chg);
    auto sigma_interleaved = XC_Functional_Libxc::convert_sigma(gdr);

    // === 3. Build tau in interleaved layout ===
    std::vector<double> tau_interleaved;
    if (is_mgga_ && chg->kin_r != nullptr) {
        tau_interleaved.resize(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                tau_interleaved[ir * nspin + is] = chg->kin_r[is][ir];
            }
        }
    }

    // === 4. Convert to spin-major layout for the CIDER bridge ===
    // Python expects: rho[is*ngrids+ir], sigma[isig*ngrids+ir], tau[is*ngrids+ir]
    const int nsigma = (nspin == 1) ? 1 : 3;

    std::vector<double> rho_sm(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            rho_sm[is * nrxx + ir] = rho_interleaved[ir * nspin + is];
        }
    }

    std::vector<double> sigma_sm(nrxx * nsigma);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int isig = 0; isig < nsigma; ++isig) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            sigma_sm[isig * nrxx + ir] = sigma_interleaved[ir * nsigma + isig];
        }
    }

    std::vector<double> tau_sm;
    double* tau_sm_ptr = nullptr;
    if (!tau_interleaved.empty()) {
        tau_sm.resize(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                tau_sm[is * nrxx + ir] = tau_interleaved[ir * nspin + is];
            }
        }
        tau_sm_ptr = tau_sm.data();
    }

    // === 5. Allocate output arrays (spin-major) ===
    std::vector<double> exc(nrxx, 0.0);
    std::vector<double> vrho_sm(nrxx * nspin, 0.0);
    std::vector<double> vsigma_sm(nrxx * nsigma, 0.0);
    std::vector<double> vtau_sm;
    double* vtau_sm_ptr = nullptr;
    if (is_mgga_) {
        vtau_sm.resize(nrxx * nspin, 0.0);
        vtau_sm_ptr = vtau_sm.data();
    }

    // === 6. Call CIDER bridge ===
    int err = cider_bridge_evaluate(
        ctx_, nspin, nrxx,
        rho_sm.data(), sigma_sm.data(), tau_sm_ptr,
        exc.data(), vrho_sm.data(), vsigma_sm.data(), vtau_sm_ptr);

    if (err != 0) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "cider_bridge_evaluate failed");
    }

    // === 7. Convert vrho back to interleaved, compute etxc and accumulate v_eff ===
    double etxc_local = 0.0;
    double vtxc_local = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+ : etxc_local, vtxc_local) schedule(static, 256)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const double rho_val = rho_interleaved[ir * nspin + is];
            // exc is per-particle, not per-spin -- but the bridge returns it as a single array
            // Sum contribution from each spin channel
            const double v_tmp = ModuleBase::e2 * vrho_sm[is * nrxx + ir];
            v_eff(is, ir) += v_tmp;
            vtxc_local += v_tmp * rho_val;
        }
    }
    // etxc: exc is per-particle energy density, integrate over total density
    for (std::size_t ir = 0; ir < nrxx; ++ir) {
        double rho_total = 0.0;
        for (int is = 0; is < nspin; ++is) {
            rho_total += rho_interleaved[ir * nspin + is];
        }
        etxc_local += ModuleBase::e2 * exc[ir] * rho_total;
    }
    *(this->etxc_) += etxc_local;

    // === 8. Convert vsigma back to interleaved for gradient correction ===
    std::vector<double> vsigma_int(nrxx * nsigma);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int isig = 0; isig < nsigma; ++isig) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            vsigma_int[ir * nsigma + isig] = vsigma_sm[isig * nrxx + ir];
        }
    }

    // Gradient correction: h = 2 * vsigma * grad_rho, v_eff -= div(h)
    {
        std::vector<std::vector<ModuleBase::Vector3<double>>> h(
            nspin, std::vector<ModuleBase::Vector3<double>>(nrxx));

        if (nspin == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                h[0][ir] = 2.0 * gdr[0][ir] * vsigma_int[ir] * 2.0;
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                h[0][ir] = 2.0 * (gdr[0][ir] * vsigma_int[ir * 3] * 2.0
                                   + gdr[1][ir] * vsigma_int[ir * 3 + 1]);
                h[1][ir] = 2.0 * (gdr[1][ir] * vsigma_int[ir * 3 + 2] * 2.0
                                   + gdr[0][ir] * vsigma_int[ir * 3 + 1]);
            }
        }

        std::vector<std::vector<double>> dh(nspin, std::vector<double>(nrxx));
        for (int is = 0; is < nspin; ++is) {
            XC_Functional::grad_dot(h[is].data(), dh[is].data(), chg->rhopw, tpiba);
        }

        double rvtxc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+ : rvtxc) schedule(static, 256)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                rvtxc += dh[is][ir] * rho_interleaved[ir * nspin + is];
                v_eff(is, ir) -= ModuleBase::e2 * dh[is][ir];
            }
        }
        vtxc_local -= rvtxc;
    }

    *(this->vtxc_) += vtxc_local;

    // === 9. MGGA: convert vtau back to interleaved, add into vofk ===
    if (is_mgga_ && vofk_ != nullptr && vtau_sm_ptr != nullptr) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                (*vofk_)(is, ir) += ModuleBase::e2 * vtau_sm[is * nrxx + ir];
            }
        }
    }

    ModuleBase::timer::tick("PotCiderXC", "cal_v_eff");
}

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC

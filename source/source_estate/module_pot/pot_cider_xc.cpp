#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_hamilt/module_xc/xc_functional_libxc.h"
#include "source_io/module_parameter/parameter.h"

#include <algorithm>
#include <cmath>
#ifdef __MPI
#include <mpi.h>
#endif

namespace
{

void log_same_density_xc_comparison(
    const Charge* const chg,
    const UnitCell* const ucell,
    const double bridge_etxc,
    const double bridge_vtxc,
    const ModuleBase::matrix& bridge_v,
    const ModuleBase::matrix* bridge_vofk = nullptr)
{
    const auto native_xc = XC_Functional_Libxc::v_xc_libxc(
        XC_Functional::get_func_id(),
        chg->nrxx,
        ucell->omega,
        ucell->tpiba,
        chg,
        nullptr);

    const double native_etxc = std::get<0>(native_xc);
    const double native_vtxc = std::get<1>(native_xc);
    const ModuleBase::matrix& native_v = std::get<2>(native_xc);

    double max_abs_v_diff = 0.0;
    double rms_v_diff = 0.0;
    double max_abs_v_native = 0.0;
    const double denom = static_cast<double>(bridge_v.nr * bridge_v.nc);

    for (int is = 0; is < bridge_v.nr; ++is) {
        for (int ir = 0; ir < bridge_v.nc; ++ir) {
            const double diff = bridge_v(is, ir) - native_v(is, ir);
            max_abs_v_diff = std::max(max_abs_v_diff, std::abs(diff));
            rms_v_diff += diff * diff;
            max_abs_v_native = std::max(max_abs_v_native, std::abs(native_v(is, ir)));
        }
    }
    rms_v_diff = std::sqrt(rms_v_diff / std::max(denom, 1.0));

    GlobalV::ofs_running
        << "PotCiderXC: same-density native-libxc comparison"
        << " etxc_bridge=" << bridge_etxc
        << " etxc_native=" << native_etxc
        << " delta_etxc=" << (bridge_etxc - native_etxc)
        << " vtxc_bridge=" << bridge_vtxc
        << " vtxc_native=" << native_vtxc
        << " delta_vtxc=" << (bridge_vtxc - native_vtxc)
        << " max_abs_v_diff=" << max_abs_v_diff
        << " rms_v_diff=" << rms_v_diff
        << " max_abs_v_native=" << max_abs_v_native
        << std::endl;

    GlobalV::ofs_running
        << "PotCiderXC: same-density sample"
        << " bridge_v00=" << bridge_v(0, 0)
        << " native_v00=" << native_v(0, 0)
        << " delta_v00=" << (bridge_v(0, 0) - native_v(0, 0))
        << std::endl;

    if (bridge_vofk != nullptr)
    {
        double max_abs_vofk = 0.0;
        double rms_vofk = 0.0;
        const double vofk_denom = static_cast<double>(bridge_vofk->nr * bridge_vofk->nc);
        for (int is = 0; is < bridge_vofk->nr; ++is)
        {
            for (int ir = 0; ir < bridge_vofk->nc; ++ir)
            {
                const double value = (*bridge_vofk)(is, ir);
                max_abs_vofk = std::max(max_abs_vofk, std::abs(value));
                rms_vofk += value * value;
            }
        }
        rms_vofk = std::sqrt(rms_vofk / std::max(vofk_denom, 1.0));

        GlobalV::ofs_running
            << "PotCiderXC: same-density vofk comparison"
            << " max_abs_vofk_diff=" << max_abs_vofk
            << " rms_vofk_diff=" << rms_vofk
            << " bridge_vofk00=" << ((*bridge_vofk).nr > 0 && (*bridge_vofk).nc > 0 ? (*bridge_vofk)(0, 0) : 0.0)
            << std::endl;
    }
}

std::pair<double, ModuleBase::matrix> reconstruct_bridge_vxc_with_native_helper(
    const int nspin,
    const std::size_t nrxx,
    const std::vector<double>& rho_interleaved,
    const std::vector<std::vector<ModuleBase::Vector3<double>>>& gdr,
    const std::vector<double>& vrho_interleaved,
    const std::vector<double>& vsigma_interleaved,
    const double tpiba,
    const Charge* const chg)
{
    const int xc_polarized = (1 == nspin) ? XC_UNPOLARIZED : XC_POLARIZED;
    std::vector<xc_func_type> funcs = XC_Functional_Libxc::init_func(
        XC_Functional::get_func_id(), xc_polarized);
    if (funcs.empty()) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "No XC functional ids available for bridge reconstruction");
    }
    std::vector<double> sgn(nrxx * nspin, 1.0);
    auto result = XC_Functional_Libxc::convert_vtxc_v(
        funcs.front(),
        nspin,
        nrxx,
        sgn,
        rho_interleaved,
        gdr,
        vrho_interleaved,
        vsigma_interleaved,
        tpiba,
        chg);
    XC_Functional_Libxc::finish_func(funcs);
    return result;
}

std::vector<double> gather_spin_major_to_global(
    const ModulePW::PW_Basis* const rho_basis,
    const std::vector<double>& local,
    const int nchannels)
{
    const int nrxx_local = rho_basis->nrxx;
    const int nxyz = rho_basis->nxyz;
    if (nchannels <= 0) {
        return {};
    }

#ifdef __MPI
    if (GlobalV::NPROC_IN_POOL > 1 && POOL_WORLD != MPI_COMM_NULL) {
        const int ncxy = rho_basis->nx * rho_basis->ny;
        std::vector<double> global(nchannels * nxyz, 0.0);
        std::vector<double> gathered(nchannels * nxyz, 0.0);
        std::vector<int> rec(GlobalV::NPROC_IN_POOL);
        std::vector<int> dis(GlobalV::NPROC_IN_POOL);
        for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip) {
            rec[ip] = rho_basis->numz[ip] * ncxy;
            dis[ip] = rho_basis->startz[ip] * ncxy;
        }

        for (int ich = 0; ich < nchannels; ++ich) {
            MPI_Allgatherv(local.data() + ich * nrxx_local,
                           nrxx_local,
                           MPI_DOUBLE,
                           gathered.data() + ich * nxyz,
                           rec.data(),
                           dis.data(),
                           MPI_DOUBLE,
                           POOL_WORLD);

            double* global_ch = global.data() + ich * nxyz;
            const double* gathered_ch = gathered.data() + ich * nxyz;
            for (int ip = 0; ip < GlobalV::NPROC_IN_POOL; ++ip) {
                for (int ixy = 0; ixy < ncxy; ++ixy) {
                    for (int iz = 0; iz < rho_basis->numz[ip]; ++iz) {
                        global_ch[rho_basis->nz * ixy + rho_basis->startz[ip] + iz] =
                            gathered_ch[rho_basis->numz[ip] * ixy
                                        + rho_basis->startz[ip] * ncxy
                                        + iz];
                    }
                }
            }
        }
        return global;
    }
#endif

    return local;
}

std::vector<double> scatter_spin_major_from_global(
    const ModulePW::PW_Basis* const rho_basis,
    const std::vector<double>& global,
    const int nchannels)
{
    const int nrxx_local = rho_basis->nrxx;
    std::vector<double> local(nchannels * nrxx_local, 0.0);
    if (nchannels <= 0) {
        return local;
    }

    const int ncxy = rho_basis->nx * rho_basis->ny;
    for (int ich = 0; ich < nchannels; ++ich) {
        const double* global_ch = global.data() + ich * rho_basis->nxyz;
        double* local_ch = local.data() + ich * nrxx_local;
        for (int ixy = 0; ixy < ncxy; ++ixy) {
            for (int iz = 0; iz < rho_basis->nplane; ++iz) {
                local_ch[ixy * rho_basis->nplane + iz] =
                    global_ch[rho_basis->nz * ixy + rho_basis->startz_current + iz];
            }
        }
    }
    return local;
}

} // namespace

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

    // Lattice vectors in row-major order (Bohr units). ABACUS stores
    // latvec in units of lat0, so scale explicitly before passing to
    // the bridge FFT/NLDF backend.
    const auto& lat = ucell_in->latvec;
    const double lat0 = ucell_in->lat0;
    double cell_cv[9] = {
        lat0 * lat.e11, lat0 * lat.e12, lat0 * lat.e13,
        lat0 * lat.e21, lat0 * lat.e22, lat0 * lat.e23,
        lat0 * lat.e31, lat0 * lat.e32, lat0 * lat.e33
    };

    const std::string& model_path = PARAM.inp.cider_model;
    if (model_path.empty()) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "cider_model path is empty");
    }

    GlobalV::ofs_running
        << "PotCiderXC: constructing bridge-owned XC context"
        << " model=" << model_path
        << " xmix=" << PARAM.inp.cider_xmix
        << " nspin=" << PARAM.inp.nspin
        << " grid=(" << N_c[0] << "," << N_c[1] << "," << N_c[2] << ")"
        << std::endl;

    ctx_ = cider_bridge_create(model_path.c_str(), N_c, cell_cv,
                               PARAM.inp.nspin, PARAM.inp.cider_xmix);
    if (!ctx_) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "Failed to create CIDER bridge context");
    }
    is_mgga_ = cider_bridge_is_mgga(ctx_);
    GlobalV::ofs_running
        << "PotCiderXC: bridge context ready"
        << " is_mgga=" << (is_mgga_ ? "true" : "false")
        << std::endl;
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
    const std::size_t nxyz = chg->rhopw->nxyz;
    const double tpiba = ucell->tpiba;

    static int eval_count = 0;
    eval_count += 1;
    GlobalV::ofs_running
        << "PotCiderXC: cal_v_eff call=" << eval_count
        << " nrxx=" << nrxx
        << " nspin=" << nspin
        << " tpiba=" << tpiba
        << std::endl;

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
                tau_interleaved[ir * nspin + is] = chg->kin_r[is][ir] / 2.0;
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
    const std::vector<double> rho_sm_global = gather_spin_major_to_global(this->rho_basis_, rho_sm, nspin);
    const std::vector<double> sigma_sm_global = gather_spin_major_to_global(this->rho_basis_, sigma_sm, nsigma);
    const std::vector<double> tau_sm_global = tau_sm.empty()
                                                  ? std::vector<double>()
                                                  : gather_spin_major_to_global(this->rho_basis_, tau_sm, nspin);

    std::vector<double> exc_global(nxyz, 0.0);
    std::vector<double> vrho_sm_global(nxyz * nspin, 0.0);
    std::vector<double> vsigma_sm_global(nxyz * nsigma, 0.0);
    std::vector<double> vtau_sm_global;
    double* tau_sm_global_ptr = nullptr;
    double* vtau_sm_global_ptr = nullptr;
    if (!tau_sm_global.empty()) {
        tau_sm_global_ptr = const_cast<double*>(tau_sm_global.data());
        vtau_sm_global.resize(nxyz * nspin, 0.0);
        vtau_sm_global_ptr = vtau_sm_global.data();
    }

    int err = cider_bridge_evaluate(
        ctx_, nspin, static_cast<int>(nxyz),
        rho_sm_global.data(), sigma_sm_global.data(), tau_sm_global_ptr,
        exc_global.data(), vrho_sm_global.data(), vsigma_sm_global.data(), vtau_sm_global_ptr);

    if (err != 0) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "cider_bridge_evaluate failed");
    }

    exc = scatter_spin_major_from_global(this->rho_basis_, exc_global, 1);
    vrho_sm = scatter_spin_major_from_global(this->rho_basis_, vrho_sm_global, nspin);
    vsigma_sm = scatter_spin_major_from_global(this->rho_basis_, vsigma_sm_global, nsigma);
    if (vtau_sm_global_ptr != nullptr) {
        vtau_sm = scatter_spin_major_from_global(this->rho_basis_, vtau_sm_global, nspin);
    }

    GlobalV::ofs_running
        << "PotCiderXC: bridge evaluate returned"
        << " exc0=" << (exc.empty() ? 0.0 : exc[0])
        << " vrho0=" << (vrho_sm.empty() ? 0.0 : vrho_sm[0])
        << " vsigma0=" << (vsigma_sm.empty() ? 0.0 : vsigma_sm[0])
        << std::endl;

    // === 7. Convert vrho back to interleaved, compute etxc and accumulate v_eff ===
    double etxc_local = 0.0;
    // etxc: exc is per-particle energy density, integrate over total density
    for (std::size_t ir = 0; ir < nrxx; ++ir) {
        double rho_total = 0.0;
        for (int is = 0; is < nspin; ++is) {
            rho_total += rho_interleaved[ir * nspin + is];
        }
        etxc_local += ModuleBase::e2 * exc[ir] * rho_total;
    }

    // === 8. Convert vrho/vsigma back to interleaved and use native ABACUS
    // helper machinery to reconstruct the local XC potential. This keeps the
    // divergence/gradient conventions aligned with the normal libxc path.
    std::vector<double> vrho_int(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            vrho_int[ir * nspin + is] = vrho_sm[is * nrxx + ir];
        }
    }

    std::vector<double> vsigma_int(nrxx * nsigma);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int isig = 0; isig < nsigma; ++isig) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            vsigma_int[ir * nsigma + isig] = vsigma_sm[isig * nrxx + ir];
        }
    }

    auto vtxc_v_bridge = reconstruct_bridge_vxc_with_native_helper(
        nspin,
        nrxx,
        rho_interleaved,
        gdr,
        vrho_int,
        vsigma_int,
        tpiba,
        chg);
    double vtxc_local = std::get<0>(vtxc_v_bridge);
    ModuleBase::matrix v_bridge = std::get<1>(vtxc_v_bridge);

#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc_local);
    Parallel_Reduce::reduce_pool(vtxc_local);
#endif

    const double grid_weight = ucell->omega / chg->rhopw->nxyz;
    *(this->etxc_) = etxc_local * grid_weight;
    *(this->vtxc_) = vtxc_local * grid_weight;

    ModuleBase::matrix bridge_vofk;
    if (is_mgga_ && vtau_sm_ptr != nullptr)
    {
        bridge_vofk.create(nspin, nrxx);
        for (int is = 0; is < nspin; ++is)
        {
            for (std::size_t ir = 0; ir < nrxx; ++ir)
            {
                bridge_vofk(is, ir) = vtau_sm[is * nrxx + ir];
            }
        }
    }

    if (std::abs(PARAM.inp.cider_xmix) < 1e-14) {
        log_same_density_xc_comparison(
            chg,
            ucell,
            *(this->etxc_),
            *(this->vtxc_),
            v_bridge,
            bridge_vofk.nc > 0 ? &bridge_vofk : nullptr);
    }

    v_eff += v_bridge;

    GlobalV::ofs_running
        << "PotCiderXC: completed call=" << eval_count
        << " etxc=" << *(this->etxc_)
        << " vtxc=" << *(this->vtxc_)
        << " grid_weight=" << grid_weight
        << std::endl;

    // === 9. MGGA: convert vtau back to interleaved, add into vofk ===
    if (is_mgga_ && vofk_ != nullptr && vtau_sm_ptr != nullptr) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                (*vofk_)(is, ir) += vtau_sm[is * nrxx + ir];
            }
        }
    }

    ModuleBase::timer::tick("PotCiderXC", "cal_v_eff");
}

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC

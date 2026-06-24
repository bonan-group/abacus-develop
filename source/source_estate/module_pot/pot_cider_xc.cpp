#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_cider_xc.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_global.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_hamilt/module_xc/libxc_abacus.h"
#include "source_io/module_parameter/parameter.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <string>
#ifdef __MPI
#include <mpi.h>
#endif

namespace
{

constexpr double CIDER_BRIDGE_GGA_RHO_THRESHOLD = 1.0e-6;
constexpr double CIDER_BRIDGE_GGA_GRHO_THRESHOLD = 1.0e-10;
constexpr double CIDER_BRIDGE_MGGA_RHO_THRESHOLD = 1.0e-8;
constexpr double CIDER_BRIDGE_MGGA_GRHO_THRESHOLD = 1.0e-12;
constexpr double CIDER_BRIDGE_MGGA_TAU_THRESHOLD = 1.0e-8;

ModuleBase::matrix g_last_cider_feature_v;
bool g_last_cider_feature_v_valid = false;

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

std::vector<double> build_bridge_sgn(
    const int nspin,
    const std::size_t nrxx,
    const std::vector<double>& rho_interleaved,
    const std::vector<double>& sigma_interleaved,
    const std::vector<double>& tau_interleaved,
    const bool is_mgga)
{
    std::vector<double> sgn(nrxx * nspin, 1.0);
    const double rho_threshold =
        is_mgga ? CIDER_BRIDGE_MGGA_RHO_THRESHOLD : CIDER_BRIDGE_GGA_RHO_THRESHOLD;
    const double grho_threshold =
        is_mgga ? CIDER_BRIDGE_MGGA_GRHO_THRESHOLD : CIDER_BRIDGE_GGA_GRHO_THRESHOLD;

    if (nspin == 1) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const bool low_density = rho_interleaved[ir] < rho_threshold;
            const bool low_gradient =
                is_mgga
                && !sigma_interleaved.empty()
                && std::sqrt(std::abs(sigma_interleaved[ir])) < grho_threshold;
            const bool low_tau =
                is_mgga
                && !tau_interleaved.empty()
                && std::abs(tau_interleaved[ir]) < CIDER_BRIDGE_MGGA_TAU_THRESHOLD;
            if (low_density || low_gradient || low_tau) {
                sgn[ir] = 0.0;
            }
        }
    } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const std::size_t up = ir * 2;
            const std::size_t dw = up + 1;
            const bool low_density_up =
                rho_interleaved[up] < rho_threshold;
            const bool low_density_dw =
                rho_interleaved[dw] < rho_threshold;
            const bool low_gradient_up =
                is_mgga
                && !sigma_interleaved.empty()
                && std::sqrt(std::abs(sigma_interleaved[ir * 3])) < grho_threshold;
            const bool low_gradient_dw =
                is_mgga
                && !sigma_interleaved.empty()
                && std::sqrt(std::abs(sigma_interleaved[ir * 3 + 2])) < grho_threshold;
            const bool low_tau_up =
                is_mgga
                && !tau_interleaved.empty()
                && std::abs(tau_interleaved[up]) < CIDER_BRIDGE_MGGA_TAU_THRESHOLD;
            const bool low_tau_dw =
                is_mgga
                && !tau_interleaved.empty()
                && std::abs(tau_interleaved[dw]) < CIDER_BRIDGE_MGGA_TAU_THRESHOLD;
            if (low_density_up || low_gradient_up || low_tau_up) {
                sgn[up] = 0.0;
            }
            if (low_density_dw || low_gradient_dw || low_tau_dw) {
                sgn[dw] = 0.0;
            }
        }
    }

    return sgn;
}

std::pair<double, ModuleBase::matrix> reconstruct_bridge_vxc_with_native_helper(
    const int nspin,
    const std::size_t nrxx,
    const std::vector<double>& sgn,
    const std::vector<double>& rho_interleaved,
    const std::vector<std::vector<ModuleBase::Vector3<double>>>& gdr,
    const std::vector<double>& vrho_interleaved,
    const std::vector<double>& vsigma_interleaved,
    const double tpiba,
    const Charge* const chg)
{
    double vtxc = 0.0;
    ModuleBase::matrix v(nspin, nrxx);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:vtxc) schedule(static, 256)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const std::size_t index = ir * nspin + is;
            const double v_tmp = ModuleBase::e2 * vrho_interleaved[index] * sgn[index];
            v(is, ir) += v_tmp;
            vtxc += v_tmp * rho_interleaved[index];
        }
    }

    if (!vsigma_interleaved.empty()) {
        const std::vector<std::vector<double>> dh = XC_Functional_Libxc::cal_dh(
            nspin, nrxx, sgn, gdr, vsigma_interleaved, tpiba, chg);

        double rvtxc = 0.0;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) reduction(+:rvtxc) schedule(static, 256)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                rvtxc += dh[is][ir] * rho_interleaved[ir * nspin + is];
                v(is, ir) -= dh[is][ir];
            }
        }
        vtxc -= rvtxc;
    }

    return std::make_pair(vtxc, std::move(v));
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

std::vector<double> interleaved_to_spin_major(
    const std::vector<double>& interleaved,
    const int nchannels,
    const std::size_t nrxx)
{
    std::vector<double> spin_major(nchannels * nrxx, 0.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int ich = 0; ich < nchannels; ++ich) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            spin_major[ich * nrxx + ir] = interleaved[ir * nchannels + ich];
        }
    }
    return spin_major;
}

std::vector<double> spin_major_to_interleaved(
    const std::vector<double>& spin_major,
    const int nchannels,
    const std::size_t nrxx)
{
    std::vector<double> interleaved(nchannels * nrxx, 0.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int ich = 0; ich < nchannels; ++ich) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            interleaved[ir * nchannels + ich] = spin_major[ich * nrxx + ir];
        }
    }
    return interleaved;
}

void write_debug_array(std::ofstream& out, const std::string& name, const std::vector<double>& values)
{
    const std::uint32_t name_size = static_cast<std::uint32_t>(name.size());
    const std::uint64_t value_size = static_cast<std::uint64_t>(values.size());
    out.write(reinterpret_cast<const char*>(&name_size), sizeof(name_size));
    out.write(name.data(), name.size());
    out.write(reinterpret_cast<const char*>(&value_size), sizeof(value_size));
    if (!values.empty()) {
        out.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(double)));
    }
}

void maybe_write_abacus_cider_debug_dump(
    const int eval_count,
    const int nspin,
    const int nsigma,
    const std::size_t nrxx,
    const double grid_weight,
    const double xmix,
    const std::vector<double>& rho_baseline_sm,
    const std::vector<double>& sigma_baseline_sm,
    const std::vector<double>& tau_baseline_sm,
    const std::vector<double>& rho_feature_sm,
    const std::vector<double>& sigma_feature_sm,
    const std::vector<double>& tau_feature_sm,
    const std::vector<double>& exc_baseline,
    const std::vector<double>& exc_feature,
    const std::vector<double>& vrho_baseline_sm,
    const std::vector<double>& vsigma_baseline_sm,
    const std::vector<double>& vtau_baseline_sm,
    const std::vector<double>& vrho_feature_sm,
    const std::vector<double>& vsigma_feature_sm,
    const std::vector<double>& vtau_feature_sm,
    const ModuleBase::matrix& v_baseline,
    const ModuleBase::matrix& v_feature,
    const ModuleBase::matrix& v_total,
    const double etxc_baseline_ry,
    const double etxc_feature_ry,
    const double vtxc_baseline_ry,
    const double vtxc_feature_ry)
{
    const char* prefix = std::getenv("ABACUS_CIDER_DEBUG_DUMP_PREFIX");
    if (prefix == nullptr || std::string(prefix).empty()) {
        return;
    }

    const std::string mode =
        std::getenv("ABACUS_CIDER_DEBUG_DUMP_MODE") == nullptr
            ? "all"
            : std::getenv("ABACUS_CIDER_DEBUG_DUMP_MODE");
    if (mode == "first" && eval_count != 1) {
        return;
    }

    char filename[4096];
    std::snprintf(filename, sizeof(filename), "%s_call%04d.bin", prefix, eval_count);
    std::ofstream out(filename, std::ios::binary);
    if (!out.good()) {
        GlobalV::ofs_warning << "PotCiderXC: failed to open debug dump " << filename << std::endl;
        return;
    }

    const char magic[8] = {'A', 'C', 'D', 'D', 'U', 'M', 'P', '1'};
    out.write(magic, sizeof(magic));
    const std::uint32_t version = 1;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    const std::uint32_t nspin_u = static_cast<std::uint32_t>(nspin);
    const std::uint32_t nsigma_u = static_cast<std::uint32_t>(nsigma);
    const std::uint64_t nrxx_u = static_cast<std::uint64_t>(nrxx);
    out.write(reinterpret_cast<const char*>(&nspin_u), sizeof(nspin_u));
    out.write(reinterpret_cast<const char*>(&nsigma_u), sizeof(nsigma_u));
    out.write(reinterpret_cast<const char*>(&nrxx_u), sizeof(nrxx_u));
    out.write(reinterpret_cast<const char*>(&grid_weight), sizeof(grid_weight));
    out.write(reinterpret_cast<const char*>(&xmix), sizeof(xmix));
    out.write(reinterpret_cast<const char*>(&etxc_baseline_ry), sizeof(etxc_baseline_ry));
    out.write(reinterpret_cast<const char*>(&etxc_feature_ry), sizeof(etxc_feature_ry));
    out.write(reinterpret_cast<const char*>(&vtxc_baseline_ry), sizeof(vtxc_baseline_ry));
    out.write(reinterpret_cast<const char*>(&vtxc_feature_ry), sizeof(vtxc_feature_ry));

    std::vector<std::pair<std::string, const std::vector<double>*>> arrays = {
        {"rho_baseline_sg", &rho_baseline_sm},
        {"sigma_baseline_xg", &sigma_baseline_sm},
        {"rho_feature_sg", &rho_feature_sm},
        {"sigma_feature_xg", &sigma_feature_sm},
        {"exc_baseline_g", &exc_baseline},
        {"exc_feature_g", &exc_feature},
        {"vrho_baseline_sg", &vrho_baseline_sm},
        {"vsigma_baseline_xg", &vsigma_baseline_sm},
        {"vrho_feature_sg", &vrho_feature_sm},
        {"vsigma_feature_xg", &vsigma_feature_sm},
    };
    if (!tau_baseline_sm.empty()) {
        arrays.push_back({"tau_baseline_sg", &tau_baseline_sm});
        arrays.push_back({"vtau_baseline_sg", &vtau_baseline_sm});
    }
    if (!tau_feature_sm.empty()) {
        arrays.push_back({"tau_feature_sg", &tau_feature_sm});
        arrays.push_back({"vtau_feature_sg", &vtau_feature_sm});
    }

    std::vector<double> v_baseline_sm(nspin * nrxx, 0.0);
    std::vector<double> v_feature_sm(nspin * nrxx, 0.0);
    std::vector<double> v_total_sm(nspin * nrxx, 0.0);
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            v_baseline_sm[is * nrxx + ir] = v_baseline(is, ir);
            v_feature_sm[is * nrxx + ir] = v_feature(is, ir);
            v_total_sm[is * nrxx + ir] = v_total(is, ir);
        }
    }
    arrays.push_back({"v_baseline_sg", &v_baseline_sm});
    arrays.push_back({"v_feature_sg", &v_feature_sm});
    arrays.push_back({"v_total_sg", &v_total_sm});

    const std::uint32_t array_count = static_cast<std::uint32_t>(arrays.size());
    out.write(reinterpret_cast<const char*>(&array_count), sizeof(array_count));
    for (const auto& item : arrays) {
        write_debug_array(out, item.first, *item.second);
    }

    GlobalV::ofs_running << "PotCiderXC: wrote debug dump " << filename << std::endl;
}

double integrate_role_exc(
    const std::vector<double>& exc,
    const std::vector<double>& sgn,
    const std::vector<double>& rho_interleaved,
    const int nspin,
    const std::size_t nrxx)
{
    double etxc_local = 0.0;
    for (std::size_t ir = 0; ir < nrxx; ++ir) {
        double rho_total = 0.0;
        for (int is = 0; is < nspin; ++is) {
            const std::size_t index = ir * nspin + is;
            rho_total += rho_interleaved[index] * sgn[index];
        }
        etxc_local += ModuleBase::e2 * exc[ir] * rho_total;
    }
    return etxc_local;
}

} // namespace

namespace elecstate
{

const ModuleBase::matrix* PotCiderXC::debug_last_feature_potential()
{
    return g_last_cider_feature_v_valid ? &g_last_cider_feature_v : nullptr;
}

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
    ModuleBase::timer::start("PotCiderXC", "cal_v_eff");

    const int nspin = chg->nspin;
    const std::size_t nrxx = chg->nrxx;
    const std::size_t nxyz = chg->rhopw->nxyz;
    const double tpiba = ucell->tpiba;

    if (nspin != 1 && nspin != 2) {
        ModuleBase::WARNING_QUIT(
            "PotCiderXC",
            "CIDER bridge currently supports only nspin=1 or nspin=2");
    }
    if (is_mgga_ && chg->kin_r == nullptr) {
        ModuleBase::WARNING_QUIT(
            "PotCiderXC",
            "MGGA CIDER bridge requires kinetic-energy density chg->kin_r");
    }

    static int eval_count = 0;
    eval_count += 1;
    GlobalV::ofs_running
        << "PotCiderXC: cal_v_eff call=" << eval_count
        << " nrxx=" << nrxx
        << " nspin=" << nspin
        << " tpiba=" << tpiba
        << std::endl;

    // === 1. Build role-specific densities in ABACUS interleaved layout ===
    // Baseline XC follows the normal NLCC path: valence + pseudo-core.
    // CIDER ML exchange features follow the training density: valence-only.
    std::vector<double> rho_baseline_interleaved(nrxx * nspin);
    std::vector<double> rho_feature_interleaved(nrxx * nspin);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
    for (int is = 0; is < nspin; ++is) {
        for (std::size_t ir = 0; ir < nrxx; ++ir) {
            const double rho_valence = chg->rho[is][ir];
            rho_feature_interleaved[ir * nspin + is] = rho_valence;
            rho_baseline_interleaved[ir * nspin + is] =
                rho_valence + chg->rho_core[ir] / static_cast<double>(nspin);
        }
    }

    // === 2. Build role-specific sigma using ABACUS gradient machinery ===
    auto gdr_baseline = XC_Functional_Libxc::cal_gdr(
        nspin, nrxx, rho_baseline_interleaved, tpiba, chg);
    auto sigma_baseline_interleaved = XC_Functional_Libxc::convert_sigma(gdr_baseline);
    auto gdr_feature = XC_Functional_Libxc::cal_gdr(
        nspin, nrxx, rho_feature_interleaved, tpiba, chg);
    auto sigma_feature_interleaved = XC_Functional_Libxc::convert_sigma(gdr_feature);

    // === 3. Build role-specific tau in interleaved layout ===
    std::vector<double> tau_baseline_interleaved;
    std::vector<double> tau_feature_interleaved;
    if (is_mgga_ && chg->kin_r != nullptr) {
        tau_baseline_interleaved.resize(nrxx * nspin);
        tau_feature_interleaved.resize(nrxx * nspin);
        const bool use_tf_core = PARAM.inp.cider_tf_tau;
        constexpr double TF_FACTOR = (3.0 / 10.0) * std::pow(3.0 * M_PI * M_PI, 2.0 / 3.0);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                const double tau_valence = chg->kin_r[is][ir] / 2.0;
                double tau_baseline = tau_valence;
                if (use_tf_core) {
                    const double rho_cps = std::max(chg->rho_core[ir] / nspin, 0.0);
                    tau_baseline += TF_FACTOR * std::pow(rho_cps, 5.0 / 3.0);
                }
                tau_feature_interleaved[ir * nspin + is] = tau_valence;
                tau_baseline_interleaved[ir * nspin + is] = tau_baseline;
            }
        }
    }

    static bool density_policy_logged = false;
    if (!density_policy_logged) {
        GlobalV::ofs_running
            << "PotCiderXC: dual-density policy"
            << " baseline_rho_sigma=valence_plus_core"
            << " feature_rho_sigma=valence_only"
            << " tau_feature=valence_only"
            << " tau_baseline="
            << (PARAM.inp.cider_tf_tau ? "valence_plus_tf_core" : "valence_only")
            << std::endl;
        density_policy_logged = true;
    }

    // === 4. Convert to spin-major layout for the CIDER bridge ===
    // Python expects: rho[is*ngrids+ir], sigma[isig*ngrids+ir], tau[is*ngrids+ir]
    const int nsigma = (nspin == 1) ? 1 : 3;

    const std::vector<double> rho_baseline_sm =
        interleaved_to_spin_major(rho_baseline_interleaved, nspin, nrxx);
    const std::vector<double> sigma_baseline_sm =
        interleaved_to_spin_major(sigma_baseline_interleaved, nsigma, nrxx);
    const std::vector<double> rho_feature_sm =
        interleaved_to_spin_major(rho_feature_interleaved, nspin, nrxx);
    const std::vector<double> sigma_feature_sm =
        interleaved_to_spin_major(sigma_feature_interleaved, nsigma, nrxx);

    std::vector<double> tau_baseline_sm;
    std::vector<double> tau_feature_sm;
    if (!tau_baseline_interleaved.empty()) {
        tau_baseline_sm = interleaved_to_spin_major(
            tau_baseline_interleaved, nspin, nrxx);
        tau_feature_sm = interleaved_to_spin_major(
            tau_feature_interleaved, nspin, nrxx);
    }

    // === 5. Allocate output arrays (spin-major) ===
    std::vector<double> exc_baseline(nrxx, 0.0);
    std::vector<double> exc_feature(nrxx, 0.0);
    std::vector<double> vrho_baseline_sm(nrxx * nspin, 0.0);
    std::vector<double> vrho_feature_sm(nrxx * nspin, 0.0);
    std::vector<double> vsigma_baseline_sm(nrxx * nsigma, 0.0);
    std::vector<double> vsigma_feature_sm(nrxx * nsigma, 0.0);
    std::vector<double> vtau_baseline_sm;
    std::vector<double> vtau_feature_sm;
    std::vector<double> vtau_total_sm;
    if (is_mgga_) {
        vtau_baseline_sm.resize(nrxx * nspin, 0.0);
        vtau_feature_sm.resize(nrxx * nspin, 0.0);
        vtau_total_sm.resize(nrxx * nspin, 0.0);
    }

    // === 6. Call CIDER bridge ===
    const std::vector<double> rho_baseline_sm_global =
        gather_spin_major_to_global(this->rho_basis_, rho_baseline_sm, nspin);
    const std::vector<double> sigma_baseline_sm_global =
        gather_spin_major_to_global(this->rho_basis_, sigma_baseline_sm, nsigma);
    const std::vector<double> rho_feature_sm_global =
        gather_spin_major_to_global(this->rho_basis_, rho_feature_sm, nspin);
    const std::vector<double> sigma_feature_sm_global =
        gather_spin_major_to_global(this->rho_basis_, sigma_feature_sm, nsigma);
    const std::vector<double> tau_baseline_sm_global =
        tau_baseline_sm.empty()
            ? std::vector<double>()
            : gather_spin_major_to_global(this->rho_basis_, tau_baseline_sm, nspin);
    const std::vector<double> tau_feature_sm_global =
        tau_feature_sm.empty()
            ? std::vector<double>()
            : gather_spin_major_to_global(this->rho_basis_, tau_feature_sm, nspin);

    std::vector<double> exc_baseline_global(nxyz, 0.0);
    std::vector<double> exc_feature_global(nxyz, 0.0);
    std::vector<double> vrho_baseline_sm_global(nxyz * nspin, 0.0);
    std::vector<double> vrho_feature_sm_global(nxyz * nspin, 0.0);
    std::vector<double> vsigma_baseline_sm_global(nxyz * nsigma, 0.0);
    std::vector<double> vsigma_feature_sm_global(nxyz * nsigma, 0.0);
    std::vector<double> vtau_baseline_sm_global;
    std::vector<double> vtau_feature_sm_global;
    double* vtau_baseline_sm_global_ptr = nullptr;
    double* vtau_feature_sm_global_ptr = nullptr;
    if (is_mgga_) {
        vtau_baseline_sm_global.resize(nxyz * nspin, 0.0);
        vtau_feature_sm_global.resize(nxyz * nspin, 0.0);
        vtau_baseline_sm_global_ptr = vtau_baseline_sm_global.data();
        vtau_feature_sm_global_ptr = vtau_feature_sm_global.data();
    }

    const double* tau_baseline_sm_global_ptr =
        tau_baseline_sm_global.empty() ? nullptr : tau_baseline_sm_global.data();
    const double* tau_feature_sm_global_ptr =
        tau_feature_sm_global.empty() ? nullptr : tau_feature_sm_global.data();

    int err = cider_bridge_evaluate_dual(
        ctx_, nspin, static_cast<int>(nxyz),
        rho_baseline_sm_global.data(),
        sigma_baseline_sm_global.data(),
        tau_baseline_sm_global_ptr,
        rho_feature_sm_global.data(),
        sigma_feature_sm_global.data(),
        tau_feature_sm_global_ptr,
        exc_baseline_global.data(),
        vrho_baseline_sm_global.data(),
        vsigma_baseline_sm_global.data(),
        vtau_baseline_sm_global_ptr,
        exc_feature_global.data(),
        vrho_feature_sm_global.data(),
        vsigma_feature_sm_global.data(),
        vtau_feature_sm_global_ptr);

    if (err != 0) {
        ModuleBase::WARNING_QUIT("PotCiderXC", "cider_bridge_evaluate_dual failed");
    }

    exc_baseline = scatter_spin_major_from_global(this->rho_basis_, exc_baseline_global, 1);
    exc_feature = scatter_spin_major_from_global(this->rho_basis_, exc_feature_global, 1);
    vrho_baseline_sm = scatter_spin_major_from_global(
        this->rho_basis_, vrho_baseline_sm_global, nspin);
    vrho_feature_sm = scatter_spin_major_from_global(
        this->rho_basis_, vrho_feature_sm_global, nspin);
    vsigma_baseline_sm = scatter_spin_major_from_global(
        this->rho_basis_, vsigma_baseline_sm_global, nsigma);
    vsigma_feature_sm = scatter_spin_major_from_global(
        this->rho_basis_, vsigma_feature_sm_global, nsigma);
    if (is_mgga_) {
        vtau_baseline_sm = scatter_spin_major_from_global(
            this->rho_basis_, vtau_baseline_sm_global, nspin);
        vtau_feature_sm = scatter_spin_major_from_global(
            this->rho_basis_, vtau_feature_sm_global, nspin);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < vtau_total_sm.size(); ++i) {
            vtau_total_sm[i] = vtau_baseline_sm[i] + vtau_feature_sm[i];
        }
    }

    GlobalV::ofs_running
        << "PotCiderXC: bridge evaluate_dual returned"
        << " exc_baseline0=" << (exc_baseline.empty() ? 0.0 : exc_baseline[0])
        << " exc_feature0=" << (exc_feature.empty() ? 0.0 : exc_feature[0])
        << " vrho_baseline0=" << (vrho_baseline_sm.empty() ? 0.0 : vrho_baseline_sm[0])
        << " vrho_feature0=" << (vrho_feature_sm.empty() ? 0.0 : vrho_feature_sm[0])
        << std::endl;

    const std::vector<double> sgn_baseline = build_bridge_sgn(
        nspin,
        nrxx,
        rho_baseline_interleaved,
        sigma_baseline_interleaved,
        tau_baseline_interleaved,
        is_mgga_);
    const std::vector<double> sgn_feature = build_bridge_sgn(
        nspin,
        nrxx,
        rho_feature_interleaved,
        sigma_feature_interleaved,
        tau_feature_interleaved,
        is_mgga_);

    // === 7. Integrate each energy density with its owning density ===
    double etxc_baseline_local = integrate_role_exc(
        exc_baseline, sgn_baseline, rho_baseline_interleaved, nspin, nrxx);
    double etxc_feature_local = integrate_role_exc(
        exc_feature, sgn_feature, rho_feature_interleaved, nspin, nrxx);

    // === 8. Convert vrho/vsigma back to interleaved and use native ABACUS
    // helper machinery to reconstruct the local XC potential. This keeps the
    // divergence/gradient conventions aligned with the normal libxc path.
    const std::vector<double> vrho_baseline_int =
        spin_major_to_interleaved(vrho_baseline_sm, nspin, nrxx);
    const std::vector<double> vsigma_baseline_int =
        spin_major_to_interleaved(vsigma_baseline_sm, nsigma, nrxx);
    const std::vector<double> vrho_feature_int =
        spin_major_to_interleaved(vrho_feature_sm, nspin, nrxx);
    const std::vector<double> vsigma_feature_int =
        spin_major_to_interleaved(vsigma_feature_sm, nsigma, nrxx);

    auto vtxc_v_baseline = reconstruct_bridge_vxc_with_native_helper(
        nspin,
        nrxx,
        sgn_baseline,
        rho_baseline_interleaved,
        gdr_baseline,
        vrho_baseline_int,
        vsigma_baseline_int,
        tpiba,
        chg);
    auto vtxc_v_feature = reconstruct_bridge_vxc_with_native_helper(
        nspin,
        nrxx,
        sgn_feature,
        rho_feature_interleaved,
        gdr_feature,
        vrho_feature_int,
        vsigma_feature_int,
        tpiba,
        chg);
    const double vtxc_baseline_local = std::get<0>(vtxc_v_baseline);
    const double vtxc_feature_local = std::get<0>(vtxc_v_feature);
    double vtxc_local = vtxc_baseline_local + vtxc_feature_local;
    ModuleBase::matrix v_bridge = std::get<1>(vtxc_v_baseline);
    const ModuleBase::matrix& v_baseline = std::get<1>(vtxc_v_baseline);
    const ModuleBase::matrix& v_feature = std::get<1>(vtxc_v_feature);
    g_last_cider_feature_v = v_feature;
    g_last_cider_feature_v_valid = true;
    for (int is = 0; is < v_bridge.nr; ++is) {
        for (int ir = 0; ir < v_bridge.nc; ++ir) {
            v_bridge(is, ir) += v_feature(is, ir);
        }
    }

#ifdef __MPI
    Parallel_Reduce::reduce_pool(etxc_baseline_local);
    Parallel_Reduce::reduce_pool(etxc_feature_local);
    Parallel_Reduce::reduce_pool(vtxc_local);
#endif

    const double grid_weight = ucell->omega / chg->rhopw->nxyz;
    const double etxc_baseline_ry = etxc_baseline_local * grid_weight;
    const double etxc_feature_ry = etxc_feature_local * grid_weight;
    const double etxc_total_ry = etxc_baseline_ry + etxc_feature_ry;
    *(this->etxc_) = etxc_total_ry;
    *(this->vtxc_) = vtxc_local * grid_weight;

    maybe_write_abacus_cider_debug_dump(
        eval_count,
        nspin,
        nsigma,
        nrxx,
        grid_weight,
        PARAM.inp.cider_xmix,
        rho_baseline_sm,
        sigma_baseline_sm,
        tau_baseline_sm,
        rho_feature_sm,
        sigma_feature_sm,
        tau_feature_sm,
        exc_baseline,
        exc_feature,
        vrho_baseline_sm,
        vsigma_baseline_sm,
        vtau_baseline_sm,
        vrho_feature_sm,
        vsigma_feature_sm,
        vtau_feature_sm,
        v_baseline,
        v_feature,
        v_bridge,
        etxc_baseline_ry,
        etxc_feature_ry,
        vtxc_baseline_local * grid_weight,
        vtxc_feature_local * grid_weight);

    const double ry_to_ha = 0.5;
    const double xmix = PARAM.inp.cider_xmix;
    GlobalV::ofs_running
        << "PotCiderXC: component energy diagnostic"
        << " baseline_ry=" << etxc_baseline_ry
        << " baseline_Ha=" << (etxc_baseline_ry * ry_to_ha)
        << " feature_scaled_ry=" << etxc_feature_ry
        << " feature_scaled_Ha=" << (etxc_feature_ry * ry_to_ha);
    if (std::abs(xmix) > 1.0e-14) {
        const double etxc_feature_unmixed_ry = etxc_feature_ry / xmix;
        GlobalV::ofs_running
            << " feature_unmixed_ry=" << etxc_feature_unmixed_ry
            << " feature_unmixed_Ha=" << (etxc_feature_unmixed_ry * ry_to_ha);
    } else {
        GlobalV::ofs_running
            << " feature_unmixed_ry=nan"
            << " feature_unmixed_Ha=nan";
    }
    GlobalV::ofs_running
        << " total_ry=" << etxc_total_ry
        << " total_Ha=" << (etxc_total_ry * ry_to_ha)
        << " xmix=" << xmix
        << std::endl;

    ModuleBase::matrix bridge_vofk;
    if (is_mgga_ && !vtau_total_sm.empty())
    {
        const std::vector<double> vtau_baseline_int =
            spin_major_to_interleaved(vtau_baseline_sm, nspin, nrxx);
        const std::vector<double> vtau_feature_int =
            spin_major_to_interleaved(vtau_feature_sm, nspin, nrxx);
        bridge_vofk.create(nspin, nrxx);
        for (int is = 0; is < nspin; ++is)
        {
            for (std::size_t ir = 0; ir < nrxx; ++ir)
            {
                const std::size_t index = ir * nspin + is;
                bridge_vofk(is, ir) =
                    vtau_baseline_int[index] * sgn_baseline[index]
                    + vtau_feature_int[index] * sgn_feature[index];
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

    // === 9. MGGA: assign summed role derivatives into vofk ===
    if (is_mgga_ && vofk_ != nullptr && !vtau_total_sm.empty()) {
        const std::vector<double> vtau_baseline_int =
            spin_major_to_interleaved(vtau_baseline_sm, nspin, nrxx);
        const std::vector<double> vtau_feature_int =
            spin_major_to_interleaved(vtau_feature_sm, nspin, nrxx);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 1024)
#endif
        for (int is = 0; is < nspin; ++is) {
            for (std::size_t ir = 0; ir < nrxx; ++ir) {
                const std::size_t index = ir * nspin + is;
                (*vofk_)(is, ir) =
                    vtau_baseline_int[index] * sgn_baseline[index]
                    + vtau_feature_int[index] * sgn_feature[index];
            }
        }
    }

    ModuleBase::timer::end("PotCiderXC", "cal_v_eff");
}

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC

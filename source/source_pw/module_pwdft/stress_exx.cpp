#include "op_pw_exx.h"
#include "source_pw/module_pwdft/kernels/cal_density_real_op.h"
#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"
#include "source_pw/module_pwdft/kernels/exx_stress_op.h"
#include "source_pw/module_pwdft/exx_wave_redistributor.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_comm.h" // use KP_WORLD
#include "source_base/parallel_device.h"
#include "source_base/parallel_reduce.h"
#include "stress_pw.h"

#include <cmath>
#include <algorithm>
#include <iomanip>
#include <map>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
std::size_t max_active_exx_star_size(const K_Vectors& kv)
{
    std::map<std::pair<int, int>, std::size_t> star_sizes;
    std::size_t max_size = 1;
    for (const auto& point: kv.exx_full_k_map)
    {
        if (point.active)
        {
            max_size = std::max(max_size, ++star_sizes[std::make_pair(point.rep_pool, point.rep_local_index)]);
        }
    }
    return max_size;
}

bool estimate_stress_fixed_bytes(std::size_t nrxx,
                                 std::size_t npw,
                                 std::size_t npwk_max,
                                 std::size_t nks,
                                 std::size_t complex_bytes,
                                 std::size_t real_bytes,
                                 std::size_t& result)
{
    std::size_t complex_count = 0;
    std::size_t real_count = 0;
    std::size_t map_count = 0;
    std::size_t offset_count = 0;
    std::size_t complex_storage = 0;
    std::size_t real_storage = 0;
    std::size_t map_storage = 0;
    std::size_t offset_storage = 0;
    return hamilt::checked_exx_size_sum({npwk_max, nrxx, npw}, complex_count)
           && hamilt::checked_exx_size_product(6, npw, real_count)
           && hamilt::checked_exx_size_sum({real_count, 6}, real_count)
           && hamilt::checked_exx_size_product({2, nks, npwk_max}, map_count)
           && hamilt::checked_exx_size_sum({nks, 1}, offset_count)
           && hamilt::checked_exx_size_product(complex_count, complex_bytes, complex_storage)
           && hamilt::checked_exx_size_product(real_count, real_bytes, real_storage)
           && hamilt::checked_exx_size_product(map_count, sizeof(int), map_storage)
           && hamilt::checked_exx_size_product(offset_count, sizeof(int), offset_storage)
           && hamilt::checked_exx_size_sum({complex_storage, real_storage, map_storage, offset_storage}, result);
}
} // namespace

template <typename FPTYPE, typename Device>
void Stress_PW<FPTYPE, Device>::stress_exx(ModuleBase::matrix& sigma,
                                           const ModuleBase::matrix& wg,
                                           ModulePW::PW_Basis* rhopw,
                                           ModulePW::PW_Basis_K* wfcpw,
                                           const K_Vectors *p_kv,
                                           const psi::Psi <std::complex<FPTYPE>, Device>* d_psi_in,
                                           const UnitCell& ucell,
                                           const hamilt::ExxOperatorOptions& exx_options_in,
                                           const hamilt::ExxExecutionContext& exx_execution_context)
{
    hamilt::ExxOperatorOptions exx_options = exx_options_in;
    bool gamma_extrapolation = exx_options.gamma_extrapolation;
    bool is_mp = p_kv->get_is_mp();
#ifdef __MPI
    Parallel_Common::bcast_bool(is_mp);
#endif
    if (!is_mp)
    {
        gamma_extrapolation = false;
    }

    // T is complex of FPTYPE, if FPTYPE is double, T is std::complex<double>
    // but if FPTYPE is std::complex<double>, T is still std::complex<double>
    using T = std::complex<FPTYPE>;
    using Real = FPTYPE;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using resmem_real_op = base_device::memory::resize_memory_op<Real, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using delmem_real_op = base_device::memory::delete_memory_op<Real, Device>;
    using syncmem_complex_op = base_device::memory::synchronize_memory_op<T, Device, Device>;
    using syncmem_real_h2d_op = base_device::memory::synchronize_memory_op<Real, Device, base_device::DEVICE_CPU>;
    using syncmem_real_d2h_op = base_device::memory::synchronize_memory_op<Real, base_device::DEVICE_CPU, Device>;

    if (exx_execution_context.kpar != 1 && !(exx_options.exxace && exx_options.separate_loop))
    {
        ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                 "PW EXX KPAR stress is supported only with exxace=1 and exx_separate_loop=1");
    }
    if (!std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
    {
        ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                 "GPU PW EXX stress requires poolnproc=1 because GPU PW FFT does not support "
                                 "intra-pool MPI distribution");
    }

    const int nspin_fac = exx_options.nspin == 2 ? 2 : 1;
    const Real k_spin_degeneracy = exx_options.nspin == 1 ? 2.0 : 1.0;
    double omega = ucell.omega;
    double tpiba = ucell.tpiba;
    double tpiba2 = ucell.tpiba2;

    // allocate space
    T* density_real = nullptr;
    T* density_recip = nullptr;
    Real* pot_tile = nullptr;
    Real* pot_stress_tile = nullptr;
    Real* sigma_exx_device = nullptr;
    Real* gcar_flat = nullptr;
    ModulePW::PW_Basis* rhopw_exx_owned = nullptr;
    ModulePW::PW_Basis* rhopw_exx = rhopw;
    ModulePW::PW_Basis_K* wfcpw_exx = nullptr;
    T* psi_exx_recip = nullptr;
    std::vector<int> exx_to_wfc_map_host;
    std::vector<int> exx_to_wfc_offsets;
    int* exx_to_wfc_map_device = nullptr;
    hamilt::ExxWaveRedistributorCpu<T> exx_wave_redistributor;
#if defined(__ROCM) && !defined(__CUDA)
    ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                             "GPU q-tile PW EXX stress path is not implemented for ROCm yet");
#endif
    {
        double ecut_exx = exx_options.ecutexx;
        if (ecut_exx == 0.0)
        {
            ecut_exx = exx_options.ecutrho;
        }
        const std::string exx_precision = std::is_same<FPTYPE, float>::value ? "single" : "double";
        rhopw_exx_owned = new ModulePW::PW_Basis(wfcpw->get_device(), exx_precision);
        rhopw_exx_owned->fft_bundle.setfft(wfcpw->get_device(), exx_precision);
#ifdef __MPI
        rhopw_exx_owned->initmpi(rhopw->poolnproc, rhopw->poolrank, rhopw->pool_world);
#endif
        rhopw_exx_owned->initgrids(rhopw->lat0, rhopw->latvec, ecut_exx);
        rhopw_exx_owned->initparameters(rhopw->gamma_only, ecut_exx, rhopw->distribution_type, rhopw->xprime);

        wfcpw_exx = new ModulePW::PW_Basis_K(wfcpw->get_device(), exx_precision);
        wfcpw_exx->fft_bundle.setfft(wfcpw->get_device(), exx_precision);
#ifdef __MPI
        wfcpw_exx->initmpi(wfcpw->poolnproc, wfcpw->poolrank, wfcpw->pool_world);
#endif
        wfcpw_exx->initgrids(wfcpw->lat0, wfcpw->latvec, ecut_exx);
        wfcpw_exx->initparameters(wfcpw->gamma_only,
                                  ecut_exx,
                                  wfcpw->nks,
                                  wfcpw->kvec_d,
                                  wfcpw->distribution_type,
                                  wfcpw->xprime);
        int active_q_count = 0;
        for (const auto& point: p_kv->exx_full_q_map)
        {
            active_q_count += point.active ? 1 : 0;
        }
        hamilt::ExxTilePolicyInput tile_input;
        tile_input.auto_tiling = exx_options.auto_tiling;
        tile_input.workload = hamilt::ExxTileWorkload::stress;
        tile_input.device = wfcpw->get_device();
        tile_input.precision = exx_precision;
        tile_input.memory_budget_mb = exx_options.tile_memory_budget_mb;
        tile_input.nbands = std::max(1, exx_options.configured_nbands);
        tile_input.active_q_count = std::max(1, active_q_count);
        tile_input.max_star_size = max_active_exx_star_size(*p_kv);
        tile_input.nrxx = static_cast<std::size_t>(rhopw_exx_owned->nrxx);
        tile_input.npw = static_cast<std::size_t>(rhopw_exx_owned->npw);
        tile_input.npwk_max = static_cast<std::size_t>(wfcpw_exx->npwk_max);
        tile_input.target_npwk_max = static_cast<std::size_t>(wfcpw->npwk_max);
        tile_input.fft_nx = rhopw_exx_owned->nx;
        tile_input.fft_ny = rhopw_exx_owned->ny;
        tile_input.fft_nz = rhopw_exx_owned->nz;
        tile_input.requested_batch_fft_size = exx_options.batch_fft_size;
        tile_input.requested_band_tile_size = exx_options.band_tile_size;
        tile_input.requested_q_tile_size = exx_options.q_tile_size;
        if (!estimate_stress_fixed_bytes(tile_input.nrxx,
                                         tile_input.npw,
                                         tile_input.npwk_max,
                                         static_cast<std::size_t>(wfcpw_exx->nks),
                                         sizeof(std::complex<FPTYPE>),
                                         sizeof(FPTYPE),
                                         tile_input.fixed_scratch_bytes))
        {
            ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                     "PW EXX stress fixed memory estimate overflows size_t");
        }
        const hamilt::ExxTilePolicyResult tile_result = hamilt::choose_exx_tiles(tile_input);
        if (!tile_result.fits)
        {
            const std::size_t required_bytes = tile_result.requested_bytes > 0
                                                   ? tile_result.requested_bytes
                                                   : tile_result.minimum_required_bytes;
            ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                     "PW EXX stress tile memory budget is too small: budget bytes = "
                                         + std::to_string(tile_result.budget_bytes)
                                         + ", required bytes = " + std::to_string(required_bytes));
        }
        exx_options.batch_fft_size = tile_result.batch_fft_size;
        exx_options.band_tile_size = tile_result.band_tile_size;
        exx_options.q_tile_size = tile_result.q_tile_size;
        exx_options.potential_cache_mode = tile_result.cache_mode;
        exx_options.tile_budget_bytes = tile_result.budget_bytes;
        exx_options.tile_estimated_peak_bytes = tile_result.estimated_peak_bytes;

        rhopw_exx_owned->setuptransform(exx_options.batch_fft_size);
        rhopw_exx_owned->collect_local_pw();
        rhopw_exx = rhopw_exx_owned;
        wfcpw_exx->setuptransform(exx_options.batch_fft_size);
        wfcpw_exx->collect_local_pw();
        if (exx_execution_context.my_rank == 0 && exx_execution_context.running_log != nullptr)
        {
            const double bytes_per_mib = 1024.0 * 1024.0;
            *exx_execution_context.running_log
                << " PW EXX stress tiling: mode = " << (exx_options.auto_tiling ? "automatic" : "manual")
                << ", budget = " << std::fixed << std::setprecision(1)
                << static_cast<double>(exx_options.tile_budget_bytes) / bytes_per_mib
                << " MiB, batch = " << exx_options.batch_fft_size
                << ", band = " << exx_options.band_tile_size
                << ", q = " << exx_options.q_tile_size
                << ", cache = " << hamilt::exx_potential_cache_mode_name(exx_options.potential_cache_mode)
                << ", estimated peak = "
                << static_cast<double>(exx_options.tile_estimated_peak_bytes) / bytes_per_mib
                << " MiB" << std::defaultfloat << std::endl;
        }
        if (rhopw_exx->nrxx != wfcpw_exx->nrxx)
        {
            ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                     "EXX stress density and wavefunction real-space layouts differ: rhopw_exx nrxx = "
                                         + std::to_string(rhopw_exx->nrxx)
                                         + ", wfcpw_exx nrxx = "
                                         + std::to_string(wfcpw_exx->nrxx));
        }
        if (std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
        {
            exx_wave_redistributor.setup(wfcpw, wfcpw_exx);
        }

        exx_to_wfc_offsets.assign(wfcpw_exx->nks + 1, 0);
        std::size_t map_capacity = 0;
        if (!hamilt::checked_exx_size_product(static_cast<std::size_t>(wfcpw_exx->npwk_max),
                                              static_cast<std::size_t>(wfcpw_exx->nks),
                                              map_capacity))
        {
            ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                     "PW EXX stress wavefunction map capacity overflows size_t");
        }
        exx_to_wfc_map_host.reserve(map_capacity);
        for (int ik = 0; ik < wfcpw_exx->nks; ++ik)
        {
            if (!hamilt::checked_exx_size_to_int(exx_to_wfc_map_host.size(), exx_to_wfc_offsets[ik]))
            {
                ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                         "PW EXX stress wavefunction map offset exceeds INT_MAX");
            }
            std::map<std::tuple<int, int, int>, int> wfc_g_to_igl;
            for (int igl = 0; igl < wfcpw->npwk[ik]; ++igl)
            {
                const auto g = wfcpw->getgdirect(ik, igl);
                wfc_g_to_igl.emplace(std::make_tuple(static_cast<int>(std::lround(g.x)),
                                                     static_cast<int>(std::lround(g.y)),
                                                     static_cast<int>(std::lround(g.z))),
                                     igl);
            }
            for (int igl = 0; igl < wfcpw_exx->npwk[ik]; ++igl)
            {
                const auto g = wfcpw_exx->getgdirect(ik, igl);
                const auto it = wfc_g_to_igl.find(std::make_tuple(static_cast<int>(std::lround(g.x)),
                                                                  static_cast<int>(std::lround(g.y)),
                                                                  static_cast<int>(std::lround(g.z))));
                exx_to_wfc_map_host.push_back(it == wfc_g_to_igl.end() ? -1 : it->second);
            }
        }
        if (!hamilt::checked_exx_size_to_int(exx_to_wfc_map_host.size(), exx_to_wfc_offsets[wfcpw_exx->nks]))
        {
            ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                     "PW EXX stress wavefunction map size exceeds INT_MAX");
        }
        if (!std::is_same<Device, base_device::DEVICE_CPU>::value)
        {
            base_device::memory::resize_memory_op<int, Device>()(exx_to_wfc_map_device, exx_to_wfc_map_host.size());
            base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>()(
                exx_to_wfc_map_device,
                exx_to_wfc_map_host.data(),
                exx_to_wfc_map_host.size());
        }
    }

    resmem_complex_op()(psi_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(density_real, rhopw_exx->nrxx);
    resmem_complex_op()(density_recip, rhopw_exx->npw);
    resmem_real_op()(sigma_exx_device, 6);

    std::vector<Real> gcar_host(static_cast<std::size_t>(rhopw_exx->npw) * 3);
    for (int ig = 0; ig < rhopw_exx->npw; ++ig)
    {
        gcar_host[ig * 3] = static_cast<Real>(rhopw_exx->gcar[ig].x);
        gcar_host[ig * 3 + 1] = static_cast<Real>(rhopw_exx->gcar[ig].y);
        gcar_host[ig * 3 + 2] = static_cast<Real>(rhopw_exx->gcar[ig].z);
    }
    resmem_real_op()(gcar_flat, gcar_host.size());
    syncmem_real_h2d_op()(gcar_flat, gcar_host.data(), gcar_host.size());
    base_device::memory::set_memory_op<Real, Device>()(sigma_exx_device, 0, 6);

    auto wave_recip_to_exx_recip = [&](const T* psi_recip, int ik_local) -> const T* {
        if (std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
        {
            exx_wave_redistributor.wfc_to_exx(ik_local, psi_recip, psi_exx_recip);
            return psi_exx_recip;
        }
        const int offset = exx_to_wfc_offsets[ik_local];
        const int* map = std::is_same<Device, base_device::DEVICE_CPU>::value
                             ? exx_to_wfc_map_host.data() + offset
                             : exx_to_wfc_map_device + offset;
        hamilt::exx_gather_recip_op<T, Device>()(psi_recip, psi_exx_recip, map, wfcpw_exx->npwk[ik_local]);
        return psi_exx_recip;
    };

    auto wave_recip_to_exx_real = [&](const T* psi_recip, T* out, int ik_local) {
        const T* exx_recip = wave_recip_to_exx_recip(psi_recip, ik_local);
        wfcpw_exx->template recip_to_real<T, Device>(exx_recip, out, ik_local);
    };

    auto load_q_real = [&](const K_Vectors::ExxFullQPoint& qpoint, int ispin_in, int mband, T* out) {
        const int iq_rep_spin = p_kv->exx_rep_spin_index(qpoint, ispin_in);
        d_psi_in->fix_kb(iq_rep_spin, mband);
        T* psi_mq = d_psi_in->get_pointer();
        if (qpoint.identity || qpoint.conjugate_only)
        {
            wave_recip_to_exx_real(psi_mq, out, iq_rep_spin);
            if (qpoint.conjugate_only)
            {
                hamilt::exx_conjugate_real_op<T, Device>()(out, out, wfcpw_exx->nrxx);
            }
        }
        else
        {
            const T* psi_mq_exx = wave_recip_to_exx_recip(psi_mq, iq_rep_spin);
            if (std::is_same<Device, base_device::DEVICE_CPU>::value)
            {
                wfcpw_exx->template recip_to_real<T, Device>(psi_mq_exx, out, iq_rep_spin);
                hamilt::rotate_exx_realspace_symmetry_cpu(wfcpw_exx, qpoint, iq_rep_spin, out, out);
            }
            else
            {
                const auto remap = hamilt::build_exx_symmetry_remap(wfcpw_exx, qpoint, iq_rep_spin, true);
                if (qpoint.time_reversal)
                {
                    wfcpw_exx->recip2real_remapped_conjugate(psi_mq_exx,
                                                             out,
                                                             static_cast<int>(remap.rep_igl.size()),
                                                             remap.rep_igl.data(),
                                                             remap.fft_isz.data(),
                                                             remap.phase.data(),
                                                             false,
                                                             Real(1.0));
                }
                else
                {
                    wfcpw_exx->recip2real_remapped(psi_mq_exx,
                                                   out,
                                                   static_cast<int>(remap.rep_igl.size()),
                                                   remap.rep_igl.data(),
                                                   remap.fft_isz.data(),
                                                   remap.phase.data(),
                                                   false,
                                                   Real(1.0));
                }
            }
        }
    };

    auto load_k_real = [&](const K_Vectors::ExxFullKPoint& kpoint, int ispin_in, int nband, T* out) {
        const int ik_rep_spin = p_kv->exx_rep_spin_index(kpoint, ispin_in);
        d_psi_in->fix_kb(ik_rep_spin, nband);
        T* psi_nk = d_psi_in->get_pointer();
        if (kpoint.identity || kpoint.conjugate_only)
        {
            wave_recip_to_exx_real(psi_nk, out, ik_rep_spin);
            if (kpoint.conjugate_only)
            {
                hamilt::exx_conjugate_real_op<T, Device>()(out, out, wfcpw_exx->nrxx);
            }
        }
        else
        {
            const T* psi_nk_exx = wave_recip_to_exx_recip(psi_nk, ik_rep_spin);
            if (std::is_same<Device, base_device::DEVICE_CPU>::value)
            {
                wfcpw_exx->template recip_to_real<T, Device>(psi_nk_exx, out, ik_rep_spin);
                hamilt::rotate_exx_realspace_symmetry_cpu(wfcpw_exx, kpoint, ik_rep_spin, out, out);
            }
            else
            {
                const auto remap = hamilt::build_exx_symmetry_remap(wfcpw_exx, kpoint, ik_rep_spin, true);
                if (kpoint.time_reversal)
                {
                    wfcpw_exx->recip2real_remapped_conjugate(psi_nk_exx,
                                                             out,
                                                             static_cast<int>(remap.rep_igl.size()),
                                                             remap.rep_igl.data(),
                                                             remap.fft_isz.data(),
                                                             remap.phase.data(),
                                                             false,
                                                             Real(1.0));
                }
                else
                {
                    wfcpw_exx->recip2real_remapped(psi_nk_exx,
                                                   out,
                                                   static_cast<int>(remap.rep_igl.size()),
                                                   remap.rep_igl.data(),
                                                   remap.fft_isz.data(),
                                                   remap.phase.data(),
                                                   false,
                                                   Real(1.0));
                }
            }
        }
    };

    const std::vector<const K_Vectors::ExxFullQPoint*> q_points = [&]() {
        std::vector<const K_Vectors::ExxFullQPoint*> points;
        for (const auto& qpoint: p_kv->exx_full_q_map)
        {
            if (qpoint.active)
            {
                points.push_back(&qpoint);
            }
        }
        return points;
    }();

    const int nbands_psi = d_psi_in->get_nbands();
    const int target_tile_size = std::max(1, std::min(exx_options.band_tile_size, nbands_psi));
    const int source_tile_size = target_tile_size;
    const int q_tile_size = q_points.empty() ? 1
                                             : std::max(1, std::min(exx_options.q_tile_size,
                                                                    static_cast<int>(q_points.size())));
    std::size_t target_count = 0;
    std::size_t q_count = 0;
    std::size_t weight_count = 0;
    std::size_t potential_count = 0;
    if (!hamilt::checked_exx_qtile_workspace_counts(static_cast<std::size_t>(target_tile_size),
                                                    static_cast<std::size_t>(source_tile_size),
                                                    static_cast<std::size_t>(q_tile_size),
                                                    static_cast<std::size_t>(wfcpw_exx->nrxx),
                                                    target_count,
                                                    q_count,
                                                    weight_count)
        || !hamilt::checked_exx_size_product(static_cast<std::size_t>(q_tile_size),
                                             static_cast<std::size_t>(rhopw_exx->npw),
                                             potential_count))
    {
        ModuleBase::WARNING_QUIT("Stress_PW::stress_exx", "PW EXX stress q-tile allocation overflows size_t");
    }
    if (!hamilt::checked_exx_allocation_bytes(target_count, sizeof(T))
        || !hamilt::checked_exx_allocation_bytes(q_count, sizeof(T))
        || !hamilt::checked_exx_allocation_bytes(weight_count, sizeof(Real))
        || !hamilt::checked_exx_allocation_bytes(potential_count, sizeof(Real)))
    {
        ModuleBase::WARNING_QUIT("Stress_PW::stress_exx",
                                 "PW EXX stress q-tile allocation byte size overflows size_t");
    }
    T* target_real_tile = nullptr;
    T* q_real_tile = nullptr;
    std::vector<Real> target_weights;
    std::vector<Real> q_weights;
    resmem_complex_op()(target_real_tile, target_count);
    resmem_complex_op()(q_real_tile, q_count);
    resmem_real_op()(pot_tile, potential_count);
    resmem_real_op()(pot_stress_tile, potential_count);
    target_weights.resize(static_cast<std::size_t>(target_tile_size), 0);
    q_weights.resize(weight_count, 0);

    for (int ispin = 0; ispin < nspin_fac; ++ispin)
    {
        for (const auto& kpoint: p_kv->exx_full_k_map)
        {
            if (!kpoint.active)
            {
                continue;
            }
            const int ik_rep_spin = p_kv->exx_rep_spin_index(kpoint, ispin);
            const bool own_kpoint = kpoint.rep_pool == exx_execution_context.my_pool;
            {
                for (int n_start = 0; n_start < nbands_psi; n_start += target_tile_size)
                {
                    const int n_count = std::min(target_tile_size, nbands_psi - n_start);
                    setmem_complex_op()(target_real_tile,
                                        0,
                                        static_cast<std::size_t>(target_tile_size)
                                            * static_cast<std::size_t>(wfcpw_exx->nrxx));
                    std::fill(target_weights.begin(), target_weights.end(), Real(0));
                    bool has_active_target = false;
                    for (int n_local = 0; n_local < n_count; ++n_local)
                    {
                        const int nband = n_start + n_local;
                        double wg_nkb = 0.0;
                        double wk_nk = 0.0;
                        if (own_kpoint)
                        {
                            wg_nkb = wg(ik_rep_spin, nband);
                            wk_nk = p_kv->wk[ik_rep_spin];
                        }
#ifdef __MPI
                        MPI_Bcast(&wg_nkb, 1, MPI_DOUBLE, p_kv->para_k.get_startpro_pool(kpoint.rep_pool), MPI_COMM_WORLD);
                        MPI_Bcast(&wk_nk, 1, MPI_DOUBLE, p_kv->para_k.get_startpro_pool(kpoint.rep_pool), MPI_COMM_WORLD);
#endif
                        if (wg_nkb >= 1e-12)
                        {
                            target_weights[n_local] = static_cast<Real>(wg_nkb / wk_nk);
                            has_active_target = true;
                            if (own_kpoint)
                            {
                                load_k_real(kpoint,
                                            ispin,
                                            nband,
                                            target_real_tile
                                                + static_cast<std::size_t>(n_local)
                                                      * static_cast<std::size_t>(wfcpw_exx->nrxx));
                            }
                        }
                    }

                    if (!has_active_target)
                    {
                        continue;
                    }

                    for (int q_start = 0; q_start < static_cast<int>(q_points.size()); q_start += q_tile_size)
                    {
                        const int q_count = std::min(q_tile_size, static_cast<int>(q_points.size()) - q_start);
                        for (int q_local = 0; q_local < q_count; ++q_local)
                        {
                            const auto* qpoint = q_points[q_start + q_local];
                            if (own_kpoint)
                            {
                                hamilt::get_exx_potential<Real, Device>(
                                    p_kv,
                                    wfcpw,
                                    rhopw_exx,
                                    pot_tile + static_cast<std::size_t>(q_local) * static_cast<std::size_t>(rhopw_exx->npw),
                                    tpiba,
                                    gamma_extrapolation ? hamilt::ExxSingularCorrectionMode::ScfMp
                                                        : hamilt::ExxSingularCorrectionMode::None,
                                    nullptr,
                                    nullptr,
                                    omega,
                                    kpoint,
                                    *qpoint,
                                    true,
                                    exx_options.coulomb_param);
                                hamilt::get_exx_stress_potential<Real, Device>(
                                    p_kv,
                                    wfcpw,
                                    rhopw_exx,
                                    pot_stress_tile
                                        + static_cast<std::size_t>(q_local) * static_cast<std::size_t>(rhopw_exx->npw),
                                    tpiba,
                                    gamma_extrapolation,
                                    omega,
                                    kpoint,
                                    *qpoint,
                                    exx_options.coulomb_param);
                            }
                        }

                        for (int m_start = 0; m_start < nbands_psi; m_start += source_tile_size)
                        {
                            const int m_count = std::min(source_tile_size, nbands_psi - m_start);
                            setmem_complex_op()(q_real_tile,
                                                0,
                                                static_cast<std::size_t>(q_tile_size)
                                                    * static_cast<std::size_t>(source_tile_size)
                                                    * static_cast<std::size_t>(wfcpw_exx->nrxx));
                            std::fill(q_weights.begin(), q_weights.end(), Real(0));

                            for (int q_local = 0; q_local < q_count; ++q_local)
                            {
                                const auto* qpoint = q_points[q_start + q_local];
                                const int iq_rep_spin = p_kv->exx_rep_spin_index(*qpoint, ispin);
                                const bool own_qpoint = qpoint->rep_pool == exx_execution_context.my_pool;
                                for (int m_local = 0; m_local < m_count; ++m_local)
                                {
                                    const int mband = m_start + m_local;
                                    double wg_mqb = 0.0;
                                    double wk_mq = 0.0;
                                    if (own_qpoint)
                                    {
                                        wg_mqb = wg(iq_rep_spin, mband);
                                        wk_mq = p_kv->wk[iq_rep_spin];
                                    }
#ifdef __MPI
                                    if (exx_execution_context.kpar > 1)
                                    {
                                        MPI_Bcast(&wg_mqb,
                                                  1,
                                                  MPI_DOUBLE,
                                                  p_kv->para_k.get_startpro_pool(qpoint->rep_pool),
                                                  MPI_COMM_WORLD);
                                        MPI_Bcast(&wk_mq,
                                                  1,
                                                  MPI_DOUBLE,
                                                  p_kv->para_k.get_startpro_pool(qpoint->rep_pool),
                                                  MPI_COMM_WORLD);
                                    }
#endif
                                    const std::size_t tile_state = static_cast<std::size_t>(q_local) * source_tile_size
                                                                   + m_local;
                                    if (wg_mqb >= 1e-12)
                                    {
                                        q_weights[tile_state] = static_cast<Real>(wg_mqb / wk_mq * qpoint->weight);
                                        if (own_qpoint)
                                        {
                                            load_q_real(*qpoint,
                                                        ispin,
                                                        mband,
                                                        q_real_tile
                                                            + tile_state * static_cast<std::size_t>(wfcpw_exx->nrxx));
                                        }
#ifdef __MPI
                                        if (exx_execution_context.kpar > 1)
                                        {
                                            Parallel_Common::bcast_dev<T, Device>(
                                                q_real_tile + tile_state * static_cast<std::size_t>(wfcpw_exx->nrxx),
                                                wfcpw_exx->nrxx,
                                                KP_WORLD,
                                                qpoint->rep_pool);
                                        }
#endif
                                    }
                                }
                            }

                            if (!own_kpoint)
                            {
                                continue;
                            }

                            for (int q_local = 0; q_local < q_count; ++q_local)
                            {
                                const auto* qpoint = q_points[q_start + q_local];
                                const Real* q_pot
                                    = pot_tile + static_cast<std::size_t>(q_local) * static_cast<std::size_t>(rhopw_exx->npw);
                                const Real* q_pot_stress
                                    = pot_stress_tile
                                      + static_cast<std::size_t>(q_local) * static_cast<std::size_t>(rhopw_exx->npw);

                                for (int m_local = 0; m_local < m_count; ++m_local)
                                {
                                    const Real q_weight = q_weights[static_cast<std::size_t>(q_local) * source_tile_size
                                                                    + m_local];
                                    if (std::abs(q_weight) < 1e-12)
                                    {
                                        continue;
                                    }
                                    T* psi_mq_real_tile
                                        = q_real_tile
                                          + (static_cast<std::size_t>(q_local) * source_tile_size + m_local)
                                                * static_cast<std::size_t>(wfcpw_exx->nrxx);

                                    for (int n_local = 0; n_local < n_count; ++n_local)
                                    {
                                        const Real k_occ = target_weights[n_local];
                                        if (std::abs(k_occ) < 1e-12)
                                        {
                                            continue;
                                        }
                                        const T* psi_nk_real_tile
                                            = target_real_tile
                                              + static_cast<std::size_t>(n_local)
                                                    * static_cast<std::size_t>(wfcpw_exx->nrxx);

                                        // overlap density in real space
                                        setmem_complex_op()(density_real, 0.0, rhopw_exx->nrxx);
                                        hamilt::cal_density_real_op<T, Device>()(psi_nk_real_tile,
                                                                                  psi_mq_real_tile,
                                                                                  density_real,
                                                                                  omega,
                                                                                  wfcpw_exx->nrxx);

                                        // density in reciprocal space
                                        rhopw_exx->template real_to_recip<T, T, Device>(density_real, density_recip);

                                        // 0.5 in the scalar is caused by 2x in the potential.
                                        const Real scalar = static_cast<Real>(-exx_options.hybrid_alpha
                                                                              * 0.25 * k_occ * kpoint.weight
                                                                              * k_spin_degeneracy * q_weight);
                                        hamilt::exx_stress_accumulate_op<T, Device>()(
                                            density_recip,
                                            q_pot,
                                            q_pot_stress,
                                            gcar_flat,
                                            static_cast<Real>(kpoint.full_kvec_c.x - qpoint->full_kvec_c.x),
                                            static_cast<Real>(kpoint.full_kvec_c.y - qpoint->full_kvec_c.y),
                                            static_cast<Real>(kpoint.full_kvec_c.z - qpoint->full_kvec_c.z),
                                            static_cast<Real>(tpiba),
                                            scalar,
                                            rhopw_exx->npw,
                                            sigma_exx_device);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (int l = 0; l < 3; l++)
    {
        for (int m = l + 1; m < 3; m++)
        {
            sigma(m, l) = sigma(l, m);
        }
    }
    Real sigma_exx_host[6] = {0, 0, 0, 0, 0, 0};
    syncmem_real_d2h_op()(sigma_exx_host, sigma_exx_device, 6);
    int idx = 0;
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        for (int beta = alpha; beta < 3; ++beta)
        {
            sigma(alpha, beta) += sigma_exx_host[idx++];
        }
    }
    for (int l = 0; l < 3; l++)
    {
        for (int m = l + 1; m < 3; m++)
        {
            sigma(m, l) = sigma(l, m);
        }
    }

    // Full-k terms are accumulated only on their representative owning pool,
    // while each pool's FFT/G work remains distributed over ranks.
    Parallel_Reduce::reduce_all(sigma.c, sigma.nr * sigma.nc);


    delmem_complex_op()(psi_exx_recip);
    delmem_complex_op()(density_real);
    delmem_complex_op()(density_recip);
    delmem_real_op()(pot_tile);
    delmem_real_op()(pot_stress_tile);
    delmem_real_op()(sigma_exx_device);
    delmem_real_op()(gcar_flat);
    delmem_complex_op()(target_real_tile);
    delmem_complex_op()(q_real_tile);
    base_device::memory::delete_memory_op<int, Device>()(exx_to_wfc_map_device);
    delete rhopw_exx_owned;
    delete wfcpw_exx;
}

template class Stress_PW<double, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Stress_PW<double, base_device::DEVICE_GPU>;
#endif

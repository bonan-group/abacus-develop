#include "op_pw_exx.h"

#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_device.h"
#include "source_base/parallel_comm.h" // use KP_WORLD
#include "source_base/parallel_reduce.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_cell/klist.h"
#include "source_hamilt/operator.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/kernels/cal_density_real_op.h"
#include "source_pw/module_pwdft/kernels/exx_cal_energy_op.h"
#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"
#include "source_pw/module_pwdft/kernels/mul_potential_op.h"
#include "source_pw/module_pwdft/kernels/vec_mul_vec_complex_op.h"
#include "source_pw/module_pwdft/exx_wave_redistributor.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if defined(__ROCM) && !defined(__CUDA)
#error "PW EXX q-tile GPU implementation is not implemented for ROCm in this merge."
#endif

namespace hamilt
{

namespace
{
bool consecutive_integers(const int* arr, std::size_t size)
{
    if (size == 0)
    {
        return false;
    }
    for (std::size_t i = 1; i < size; ++i)
    {
        if (arr[i] != arr[i - 1] + 1)
        {
            return false;
        }
    }
    return true;
}
}

template <typename T, typename Device>
std::vector<typename GetTypeReal<T>::type> OperatorEXXPW<T, Device>::fock_div = {};

template <typename T, typename Device>
std::vector<typename GetTypeReal<T>::type> OperatorEXXPW<T, Device>::erfc_div = {};

template <typename T, typename Device>
OperatorEXXPW<T, Device>::OperatorEXXPW(const int* isk_in,
                                        const ModulePW::PW_Basis_K* wfcpw_in,
                                        const ModulePW::PW_Basis* rhopw_in,
                                        K_Vectors *kv_in,
                                        const UnitCell *ucell,
                                        const ExxOperatorOptions& options_in)
    : isk(isk_in), wfcpw(wfcpw_in), rhopw(rhopw_in), kv(kv_in), options(options_in), ucell(ucell)
{
    if (GlobalV::KPAR != 1 && !(options.exxace && options.separate_loop))
    {
        // GlobalV::ofs_running << "EXX Calculation does not support k-point parallelism" << std::endl;
        ModuleBase::WARNING_QUIT("OperatorEXXPW",
                                 "PW EXX KPAR is supported only with exxace=1 and exx_separate_loop=1");
    }
    if (!std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW",
                                 "GPU PW EXX requires poolnproc=1 because GPU PW FFT does not support "
                                 "intra-pool MPI distribution");
    }
    gamma_extrapolation = options.gamma_extrapolation;
    bool is_mp = kv_in->get_is_mp();
#ifdef __MPI
    Parallel_Common::bcast_bool(is_mp);
#endif
    if (!is_mp)
    {
        gamma_extrapolation = false;
    }
    singular_correction_mode = gamma_extrapolation ? ExxSingularCorrectionMode::ScfMp
                                                   : ExxSingularCorrectionMode::None;

    this->classname = "OperatorEXXPW";
    this->ctx = nullptr;
    this->cpu_ctx = nullptr;
    this->cal_type = hamilt::calculation_type::pw_exx;

    int nks = wfcpw->nks;
    int nk_fac = options.nspin == 2 ? 2 : 1;

    tpiba = ucell->tpiba;
    Real tpiba2 = tpiba * tpiba;

    // initialize rhopw_dev
    double ecut_exx = options.ecutexx;
    if (ecut_exx == 0.0)
    {
        ecut_exx = options.ecutrho;
    }

    const std::string exx_precision = std::is_same<Real, float>::value ? "single" : "double";
    rhopw_dev = new ModulePW::PW_Basis(wfcpw->get_device(), exx_precision);
    rhopw_dev->fft_bundle.setfft(wfcpw->get_device(), exx_precision);
#ifdef __MPI
    rhopw_dev->initmpi(rhopw->poolnproc, rhopw->poolrank, rhopw->pool_world);
#endif
    // here we can actually use different ecut to init the grids
    rhopw_dev->initgrids(rhopw->lat0, rhopw->latvec, ecut_exx);
    rhopw_dev->initparameters(rhopw->gamma_only, ecut_exx, rhopw->distribution_type, rhopw->xprime);
    rhopw_dev->setuptransform(options.batch_fft_size);
    rhopw_dev->collect_local_pw();

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
    wfcpw_exx->setuptransform(options.batch_fft_size);
    wfcpw_exx->collect_local_pw();
    if (rhopw_dev->nrxx != wfcpw_exx->nrxx)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW",
                                 "EXX density and wavefunction real-space layouts differ: rhopw_dev nrxx = "
                                     + std::to_string(rhopw_dev->nrxx)
                                     + ", wfcpw_exx nrxx = "
                                     + std::to_string(wfcpw_exx->nrxx));
    }
    if (std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
    {
        exx_wave_redistributor.reset(new ExxWaveRedistributorCpu<T>());
        exx_wave_redistributor->setup(wfcpw, wfcpw_exx);
    }

    if (GlobalV::MY_RANK == 0)
    {
        int active_full_q_count = 0;
        for (const auto& qpoint: kv->exx_full_q_map)
        {
            if (qpoint.active)
            {
                ++active_full_q_count;
            }
        }
        GlobalV::ofs_running << " EXX effective ecutexx = " << ecut_exx
                             << " Ry, charge FFT = " << rhopw->nx << " " << rhopw->ny << " " << rhopw->nz
                             << ", wfc FFT = " << wfcpw->nx << " " << wfcpw->ny << " " << wfcpw->nz
                             << ", EXX FFT = " << rhopw_dev->nx << " " << rhopw_dev->ny << " " << rhopw_dev->nz
                             << ", EXX npw = " << rhopw_dev->npw << std::endl;
        GlobalV::ofs_running << " EXX q-tile path = on"
                             << ", batch FFT size = " << options.batch_fft_size
                             << ", band tile size = " << options.band_tile_size
                             << ", q tile size = " << options.q_tile_size
                             << ", q ownership = " << (GlobalV::KPAR > 1 ? "owner-local" : "local")
                             << ", reduced k = " << wfcpw->nks / nk_fac
                             << ", full q = " << active_full_q_count << std::endl;
    }

    // allocate real-space work buffers on the actual EXX grid
    resmem_complex_op()(psi_nk_real, wfcpw_exx->nrxx);
    resmem_complex_op()(psi_mq_real, wfcpw_exx->nrxx);
    resmem_complex_op()(h_psi_recip, wfcpw->npwk_max);
    resmem_complex_op()(density_real, rhopw_dev->nrxx);
    resmem_complex_op()(h_psi_real, rhopw_dev->nrxx);
    resmem_complex_op()(density_recip, rhopw_dev->npw);
    resmem_complex_op()(psi_nk_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(psi_mq_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(h_psi_exx_recip, wfcpw_exx->npwk_max);
    resmem_real_op()(pot, rhopw_dev->npw);

    int batch_fft_size = this->wfcpw_exx->fft_bundle.get_batch_size<Real>();
    if (batch_fft_size <= 0 && wfcpw->get_device() == "gpu")
    {
        batch_fft_size = 1;
    }
    if (batch_fft_size > 0)
    {
        resmem_complex_op()(psi_mq_batch_real, batch_fft_size * wfcpw_exx->nrxx);
        resmem_complex_op()(psi_mq_batch_recip, batch_fft_size * wfcpw_exx->npwk_max);
        resmem_complex_op()(density_real_batch, batch_fft_size * rhopw_dev->nrxx);
        resmem_complex_op()(density_recip_batch, batch_fft_size * rhopw_dev->npw);
        resmem_real_op()(density_norm_batch, batch_fft_size * rhopw_dev->npw);
        resmem_real_op()(energy_batch, batch_fft_size);
    }

    fock_div.clear();
    erfc_div.clear();
    for (const auto& param: options.fock_params)
    {
        fock_div.push_back(exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type::Fock,
                                          0.0,
                                          kv,
                                          wfcpw,
                                          rhopw_dev,
                                          tpiba,
                                          singular_correction_mode,
                                          ucell->omega));
    }
    for (const auto& param: options.erfc_params)
    {
        erfc_div.push_back(exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type::Erfc,
                                          std::stod(param.at("omega")),
                                          kv,
                                          wfcpw,
                                          rhopw_dev,
                                          tpiba,
                                          singular_correction_mode,
                                          ucell->omega));
    }

}   // end of constructor

template <typename T, typename Device>
OperatorEXXPW<T, Device>::~OperatorEXXPW()
{
    // use delete_memory_op to delete the allocated pws
    delmem_complex_op()(psi_nk_real);
    delmem_complex_op()(psi_mq_real);
    delmem_complex_op()(density_real);
    delmem_complex_op()(h_psi_real);
    delmem_complex_op()(density_recip);
    delmem_complex_op()(h_psi_recip);
    delmem_complex_op()(psi_nk_exx_recip);
    delmem_complex_op()(psi_mq_exx_recip);
    delmem_complex_op()(h_psi_exx_recip);
    delmem_complex_op()(psi_mq_batch_real);
    delmem_complex_op()(psi_mq_batch_recip);
    delmem_complex_op()(density_real_batch);
    delmem_complex_op()(density_recip_batch);
    delmem_real_op()(density_norm_batch);
    delmem_real_op()(energy_batch);
    delmem_complex_op()(alpha_all_device);
    delmem_real_op()(weight_real_device);
    base_device::memory::delete_memory_op<int, Device>()(exx_to_wfc_map_device);
    delmem_complex_op()(qtile_target_real);
    delmem_complex_op()(qtile_h_real);
    delmem_complex_op()(qtile_q_real);

    clear_exx_potential_cache();
    clear_remap_device_cache();
    delmem_real_op()(pot);
    delmem_real_op()(qtile_energy_device);

    delmem_complex_op()(h_psi_ace);
    delmem_complex_op()(psi_h_psi_ace);
    delmem_complex_op()(L_ace);
    delmem_complex_op()(Xi_psi_ace);
    for (auto &Xi_ace: Xi_ace_k)
    {
        delmem_complex_op()(Xi_ace);
    }
    Xi_ace_k.clear();
    if (owns_rhopw_dev)
    {
        delete rhopw_dev;
    }
    if (owns_wfcpw_exx)
    {
        delete wfcpw_exx;
    }
}

template <typename T>
inline bool is_finite(const T &val)
{
    return std::isfinite(val);
}

template <>
inline bool is_finite(const std::complex<float> &val)
{
    return std::isfinite(val.real()) && std::isfinite(val.imag());
}

template <>
inline bool is_finite(const std::complex<double> &val)
{
    return std::isfinite(val.real()) && std::isfinite(val.imag());
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act(const int nbands,
                                   const int nbasis,
                                   const int npol,
                                   const T *tmpsi_in,
                                   T *tmhpsi,
                                   const int ngk_ik,
                                   const bool is_first_node) const
{
    if (first_iter) return;
    // std::cout << cal_exx_energy_ace(&psi) << " EXX energy" << std::endl;
    // MPI_Abort(MPI_COMM_WORLD, 0);
    // return;

    if (is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis*nbands/npol);
    }

    if (source_op_for_target != nullptr)
    {
        const int nspin_fac = options.nspin == 2 ? 2 : 1;
        const int nks_no_spin = wfcpw->nks / nspin_fac;
        const int ispin = this->ik < nks_no_spin ? 0 : 1;
        const int ik_no_spin = this->ik - ispin * nks_no_spin;
        const int ik_global_no_spin = target_kv_for_target != nullptr
                                          && ik_no_spin >= 0
                                          && ik_no_spin < static_cast<int>(target_kv_for_target->ik2iktot.size())
                                          ? target_kv_for_target->ik2iktot[ik_no_spin]
                                          : ik_no_spin;
        if (target_kv_for_target == nullptr || ik_global_no_spin < 0
            || ik_global_no_spin >= static_cast<int>(target_kv_for_target->exx_full_k_map.size()))
        {
            ModuleBase::WARNING_QUIT("OperatorEXXPW::act", "target EXX k-point descriptor is unavailable");
        }
        const auto& target_kpoint = target_kv_for_target->exx_full_k_map[ik_global_no_spin];
        act_op_qtile(nbands,
                     nbasis,
                     npol,
                     tmpsi_in,
                     tmhpsi,
                     ngk_ik,
                     is_first_node,
                     true,
                     ispin,
                     &target_kpoint,
                     this->ik);
        return;
    }

    if (options.exxace && options.separate_loop)
    {
        act_op_ace(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
    }
    else
    {
        act_op(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
    }
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op(const int nbands,
                                   const int nbasis,
                                   const int npol,
                                   const T *tmpsi_in,
                                   T *tmhpsi,
                                   const int ngk_ik,
                                   const bool is_first_node) const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        if (GlobalV::KPAR > 1)
        {
            ModuleBase::WARNING_QUIT("OperatorEXXPW::act_op_qtile_cpu",
                                     "direct noACE KPAR q-tile PW EXX is not synchronization-safe; "
                                     "use exxace=1 and exx_separate_loop=1");
        }
        act_op_qtile_cpu(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node, true, -1);
        return;
    }

    if (GlobalV::KPAR > 1)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::act_op_qtile_gpu",
                                 "direct noACE GPU q-tile PW EXX supports KPAR=1 only");
    }
    act_op_qtile_gpu(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node, true, -1);
}

template <typename T, typename Device>
int OperatorEXXPW<T, Device>::resolve_qtile_chunk_size() const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        return 1;
    }
    const int batch_size = this->get_batch_fft_size();
    return batch_size > 1 ? batch_size : 1;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::ensure_qtile_workspace(std::size_t target_size,
                                                      std::size_t q_size,
                                                      int batch_limit) const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        return;
    }
    if (qtile_target_real_size < target_size)
    {
        delmem_complex_op()(qtile_target_real);
        qtile_target_real = nullptr;
        resmem_complex_op()(qtile_target_real, target_size);
        qtile_target_real_size = target_size;
    }
    if (qtile_h_real_size < target_size)
    {
        delmem_complex_op()(qtile_h_real);
        qtile_h_real = nullptr;
        resmem_complex_op()(qtile_h_real, target_size);
        qtile_h_real_size = target_size;
    }
    if (qtile_q_real_size < q_size)
    {
        delmem_complex_op()(qtile_q_real);
        qtile_q_real = nullptr;
        resmem_complex_op()(qtile_q_real, q_size);
        qtile_q_real_size = q_size;
    }
    if (batch_limit > 0 && weight_real_capacity < static_cast<std::size_t>(batch_limit))
    {
        delmem_real_op()(weight_real_device);
        weight_real_device = nullptr;
        resmem_real_op()(weight_real_device, batch_limit);
        weight_real_capacity = batch_limit;
    }
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::fill_target_tile(const T* tmpsi_in,
                                                int nbasis,
                                                int n_start,
                                                int n_count,
                                                T* target_real) const
{
    const std::size_t real_size = static_cast<std::size_t>(wfcpw_exx->nrxx);
    for (int n_local = 0; n_local < n_count; ++n_local)
    {
        const int n_iband = n_start + n_local;
        const T* psi_nk = tmpsi_in + static_cast<std::size_t>(n_iband) * nbasis;
        wave_recip_to_exx_real(psi_nk, target_real + static_cast<std::size_t>(n_local) * real_size, this->ik);
    }
}

template <typename T, typename Device>
typename OperatorEXXPW<T, Device>::Real
OperatorEXXPW<T, Device>::fill_q_tile_states(const std::vector<const K_Vectors::ExxFullQPoint*>& q_points,
                                             int q_start,
                                             int q_count,
                                             int ispin,
                                             int m_start,
                                             int m_count,
                                             int source_tile_size,
                                             T* q_real,
                                             Real* q_weights,
                                             bool load_wavefunctions) const
{
    ModuleBase::timer::start("OperatorEXXPW", "q_tile_fetch");

    const std::size_t real_size = static_cast<std::size_t>(wfcpw_exx->nrxx);
    Real weight_sum = 0;
    const int chunk_size = resolve_qtile_chunk_size();
    for (int q_local = 0; q_local < q_count; ++q_local)
    {
        const auto* qpoint = q_points[q_start + q_local];
        ensure_full_point_supported(*qpoint);
        const int iq_rep_spin = rep_spin_index(*qpoint, ispin);
        const int iq_pool = qpoint->rep_pool;
        const bool own_qpoint = iq_pool == GlobalV::MY_POOL;

        for (int m_local = 0; m_local < m_count; ++m_local)
        {
            const int m_iband = m_start + m_local;
            double wg_mqb_real = 0.0;
            double wk_iq_real = 0.0;
            if (own_qpoint)
            {
                wg_mqb_real = (*wg)(iq_rep_spin, m_iband);
                wk_iq_real = kv->wk[iq_rep_spin];
            }
#ifdef __MPI
            if (GlobalV::KPAR > 1)
            {
            MPI_Bcast(&wg_mqb_real, 1, MPI_DOUBLE, kv->para_k.get_startpro_pool(iq_pool), MPI_COMM_WORLD);
            MPI_Bcast(&wk_iq_real, 1, MPI_DOUBLE, kv->para_k.get_startpro_pool(iq_pool), MPI_COMM_WORLD);
            }
#endif
            const std::size_t tile_state = static_cast<std::size_t>(q_local) * source_tile_size + m_local;
            q_weights[tile_state] = 0;
            if (wg_mqb_real < 1e-12)
            {
                continue;
            }
            q_weights[tile_state] = static_cast<Real>(wg_mqb_real / wk_iq_real * qpoint->weight);
            weight_sum += std::abs(q_weights[tile_state]);

            if (!load_wavefunctions)
            {
                continue;
            }

            T* q_ptr = q_real + tile_state * real_size;
            // KPAR q-state fetch still loads/broadcasts one state at a time,
            // but the assembled q tile can be consumed by batched GPU apply.
            const bool use_gpu_batch_load = source_op_for_target == nullptr
                                            && !std::is_same<Device, base_device::DEVICE_CPU>::value
                                            && GlobalV::KPAR == 1 && chunk_size > 1;
            if (own_qpoint && !use_gpu_batch_load)
            {
                if (source_op_for_target != nullptr)
                {
                    source_op_for_target->load_full_point_real(*qpoint, ispin, m_iband, q_ptr);
                }
                else
                {
                    load_full_point_real(*qpoint, ispin, m_iband, q_ptr);
                }
            }
#ifdef __MPI
            if (GlobalV::KPAR > 1)
            {
                if (std::is_same<Device, base_device::DEVICE_CPU>::value)
                {
                    Parallel_Common::bcast_data(q_ptr, wfcpw_exx->nrxx, KP_WORLD, iq_pool);
                }
                else
                {
                    Parallel_Common::bcast_dev<T, Device>(q_ptr, wfcpw_exx->nrxx, KP_WORLD, iq_pool);
                }
            }
#endif
        }

        if (!load_wavefunctions || source_op_for_target != nullptr || !own_qpoint
            || std::is_same<Device, base_device::DEVICE_CPU>::value
            || GlobalV::KPAR > 1 || chunk_size <= 1)
        {
            continue;
        }

        int m_local = 0;
        while (m_local < m_count)
        {
            while (m_local < m_count)
            {
                const std::size_t tile_state = static_cast<std::size_t>(q_local) * source_tile_size + m_local;
                if (std::abs(q_weights[tile_state]) >= std::numeric_limits<Real>::epsilon())
                {
                    break;
                }
                ++m_local;
            }
            if (m_local >= m_count)
            {
                break;
            }
            const int run_start = m_local;
            const int run_limit = std::min(m_count, run_start + chunk_size);
            while (m_local < run_limit)
            {
                const std::size_t tile_state = static_cast<std::size_t>(q_local) * source_tile_size + m_local;
                if (std::abs(q_weights[tile_state]) < std::numeric_limits<Real>::epsilon())
                {
                    break;
                }
                ++m_local;
            }
            const int run_count = m_local - run_start;
            if (run_count <= 0)
            {
                continue;
            }

            T* q_ptr = q_real + (static_cast<std::size_t>(q_local) * source_tile_size + run_start) * real_size;
            const int first_band = m_start + run_start;
            if ((qpoint->identity || qpoint->conjugate_only) && psi.get_k_first())
            {
                for (int ib = 0; ib < run_count; ++ib)
                {
                    wave_recip_to_exx_recip(get_pw(first_band + ib, iq_rep_spin),
                                            iq_rep_spin,
                                            psi_mq_batch_recip + static_cast<std::size_t>(ib) * wfcpw_exx->npwk_max);
                }
                wfcpw_exx->recip_to_real_batch<Real, Device>(ctx,
                                                             psi_mq_batch_recip,
                                                             q_ptr,
                                                             iq_rep_spin,
                                                             run_count,
                                                             false,
                                                             Real(1.0));
                if (qpoint->conjugate_only)
                {
                    exx_conjugate_real_op<T, Device>()(q_ptr, q_ptr, static_cast<std::size_t>(run_count) * wfcpw_exx->nrxx);
                }
            }
            else if (!qpoint->identity && !qpoint->conjugate_only)
            {
                batch_actual_band_idx_cache.resize(run_count);
                for (int ib = 0; ib < run_count; ++ib)
                {
                    batch_actual_band_idx_cache[ib] = first_band + ib;
                }
                load_full_point_real_batch(*qpoint, ispin, batch_actual_band_idx_cache.data(), run_count, q_ptr);
            }
            else
            {
                for (int ib = 0; ib < run_count; ++ib)
                {
                    load_full_point_real(*qpoint,
                                         ispin,
                                         first_band + ib,
                                         q_ptr + static_cast<std::size_t>(ib) * real_size);
                }
            }
        }
    }

    ModuleBase::timer::end("OperatorEXXPW", "q_tile_fetch");
    return weight_sum;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::process_qtile_apply_tile(const K_Vectors::ExxFullKPoint& local_kpoint,
                                                        const K_Vectors::ExxFullQPoint& qpoint,
                                                        const T* target_real,
                                                        T* h_real,
                                                        const T* q_real,
                                                        const Real* q_weights,
                                                        const T* q_weights_device,
                                                        int q_local,
                                                        int n_count,
                                                        int m_count,
                                                        int source_tile_size,
                                                        int chunk_size) const
{
    ModuleBase::timer::start("OperatorEXXPW", "q_tile_pair");

    const bool use_batch = !std::is_same<Device, base_device::DEVICE_CPU>::value && chunk_size > 1;
    const std::size_t real_size = static_cast<std::size_t>(wfcpw_exx->nrxx);
    Real* pot_ik_iq = get_exx_potential_cached(local_kpoint, qpoint);

    for (int n_local = 0; n_local < n_count; ++n_local)
    {
        T* h_ptr = h_real + static_cast<std::size_t>(n_local) * real_size;
        const T* psi_nk_ptr = target_real + static_cast<std::size_t>(n_local) * real_size;

        if (!use_batch)
        {
            for (int m_local = 0; m_local < m_count; ++m_local)
            {
                const Real pair_weight = q_weights[static_cast<std::size_t>(q_local) * source_tile_size + m_local];
                if (std::abs(pair_weight) < std::numeric_limits<Real>::epsilon())
                {
                    continue;
                }

                const T* psi_mq_ptr
                    = q_real + (static_cast<std::size_t>(q_local) * source_tile_size + m_local) * real_size;
                cal_density_recip(psi_nk_ptr, psi_mq_ptr, ucell->omega);
                mul_potential_op<T, Device>()(pot_ik_iq,
                                              density_recip,
                                              rhopw_dev->npw,
                                              wfcpw->nks,
                                              this->ik,
                                              qpoint.full_index);
                rho_recip2real(density_recip, density_real, false, Real(1.0));
                vec_mul_vec_complex_op<T, Device>()(density_real, psi_mq_ptr, density_real, wfcpw_exx->nrxx);

                const T pair_weight_t = pair_weight;
                axpy_complex_op()(wfcpw_exx->nrxx, &pair_weight_t, density_real, 1, h_ptr, 1);
            }
            continue;
        }

        for (int m_batch_start = 0; m_batch_start < m_count; m_batch_start += chunk_size)
        {
            const int m_batch_end = std::min(m_batch_start + chunk_size, m_count);
            int first_active = m_batch_end;
            int last_active = m_batch_start;
            for (int m_local = m_batch_start; m_local < m_batch_end; ++m_local)
            {
                const Real pair_weight = q_weights[static_cast<std::size_t>(q_local) * source_tile_size + m_local];
                if (std::abs(pair_weight) >= std::numeric_limits<Real>::epsilon())
                {
                    first_active = std::min(first_active, m_local);
                    last_active = m_local + 1;
                }
            }
            if (first_active == m_batch_end)
            {
                continue;
            }

            const int batch_count = last_active - first_active;
            const T* alpha_weights = nullptr;
            if (q_weights_device != nullptr)
            {
                alpha_weights = q_weights_device + static_cast<std::size_t>(q_local) * source_tile_size + first_active;
            }
            else
            {
                alpha_all_host_work.assign(batch_count, T{});
                for (int ib = 0; ib < batch_count; ++ib)
                {
                    alpha_all_host_work[ib]
                        = static_cast<T>(q_weights[static_cast<std::size_t>(q_local) * source_tile_size + first_active + ib]);
                }
                if (alpha_all_capacity < static_cast<std::size_t>(batch_count))
                {
                    delmem_complex_op()(alpha_all_device);
                    alpha_all_device = nullptr;
                    resmem_complex_op()(alpha_all_device, batch_count);
                    alpha_all_capacity = batch_count;
                }
                syncmem_complex_c2d_op()(alpha_all_device, alpha_all_host_work.data(), batch_count);
                alpha_weights = alpha_all_device;
            }

            const T* psi_mq_batch
                = q_real + (static_cast<std::size_t>(q_local) * source_tile_size + first_active) * real_size;
            cal_density_recip_batch(psi_nk_ptr,
                                    const_cast<T*>(psi_mq_batch),
                                    density_real_batch,
                                    density_recip_batch,
                                    batch_count,
                                    qpoint.rep_local_index,
                                    ucell->omega);
            mul_potential_op<T, Device>().operator_batch(pot_ik_iq, density_recip_batch, rhopw_dev->npw, batch_count);
            rhopw_dev->recip_to_real_batch<Real, Device>(this->ctx,
                                                         density_recip_batch,
                                                         density_real_batch,
                                                         batch_count,
                                                         false,
                                                         Real(1.0));
            vec_mul_vec_complex_op<T, Device>().operator_batch(density_real_batch,
                                                               const_cast<T*>(psi_mq_batch),
                                                               density_real_batch,
                                                               wfcpw_exx->nrxx,
                                                               batch_count);

            const T one{1, 0};
            const int inc{1};
            ModuleBase::gemv_op<T, Device>()('N',
                                             wfcpw_exx->nrxx,
                                             batch_count,
                                             &one,
                                             density_real_batch,
                                             wfcpw_exx->nrxx,
                                             alpha_weights,
                                             inc,
                                             &one,
                                             h_ptr,
                                             inc);
        }
    }

    ModuleBase::timer::end("OperatorEXXPW", "q_tile_pair");
}

template <typename T, typename Device>
double OperatorEXXPW<T, Device>::process_qtile_energy_tile(const K_Vectors::ExxFullKPoint& kpoint,
                                                           const K_Vectors::ExxFullQPoint& qpoint,
                                                           const T* psi_nk_real_in,
                                                           const T* q_real,
                                                           const Real* q_weights,
                                                           const Real* q_weights_device,
                                                           Real k_weight,
                                                           int q_local,
                                                           int m_count,
                                                           int source_tile_size,
                                                           int chunk_size) const
{
    const bool use_batch = !std::is_same<Device, base_device::DEVICE_CPU>::value && chunk_size > 1;
    const std::size_t real_size = static_cast<std::size_t>(wfcpw_exx->nrxx);
    double energy = 0.0;

    Real* pot_ik_iq = get_exx_potential_cached(kpoint, qpoint);

    if (!use_batch)
    {
        for (int m_local = 0; m_local < m_count; ++m_local)
        {
            const Real q_weight = q_weights[static_cast<std::size_t>(q_local) * source_tile_size + m_local];
            if (std::abs(q_weight) < std::numeric_limits<Real>::epsilon())
            {
                continue;
            }
            const T* psi_mq_ptr = q_real + (static_cast<std::size_t>(q_local) * source_tile_size + m_local) * real_size;
            cal_density_recip(psi_nk_real_in, psi_mq_ptr, ucell->omega);
            energy += exx_cal_energy_op<T, Device>()(density_recip, pot_ik_iq, k_weight * q_weight, rhopw_dev->npw);
        }
        return energy;
    }

    for (int m_batch_start = 0; m_batch_start < m_count; m_batch_start += chunk_size)
    {
        const int m_batch_end = std::min(m_batch_start + chunk_size, m_count);
        int first_active = m_batch_end;
        int last_active = m_batch_start;
        for (int m_local = m_batch_start; m_local < m_batch_end; ++m_local)
        {
            const Real q_weight = q_weights[static_cast<std::size_t>(q_local) * source_tile_size + m_local];
            if (std::abs(q_weight) >= std::numeric_limits<Real>::epsilon())
            {
                first_active = std::min(first_active, m_local);
                last_active = m_local + 1;
            }
        }
        if (first_active == m_batch_end)
        {
            continue;
        }

        const int batch_count = last_active - first_active;
        const Real* batch_weight_device = nullptr;
        if (q_weights_device != nullptr)
        {
            batch_weight_device = q_weights_device + static_cast<std::size_t>(q_local) * source_tile_size + first_active;
        }
        else
        {
            weight_real_host_cache.assign(batch_count, Real(0));
            for (int ib = 0; ib < batch_count; ++ib)
            {
                weight_real_host_cache[ib]
                    = k_weight * q_weights[static_cast<std::size_t>(q_local) * source_tile_size + first_active + ib];
            }
            syncmem_real_c2d_op()(weight_real_device, weight_real_host_cache.data(), batch_count);
            batch_weight_device = weight_real_device;
        }

        const T* psi_mq_batch = q_real + (static_cast<std::size_t>(q_local) * source_tile_size + first_active) * real_size;
        cal_density_recip_batch(psi_nk_real_in,
                                const_cast<T*>(psi_mq_batch),
                                density_real_batch,
                                density_recip_batch,
                                batch_count,
                                qpoint.rep_local_index,
                                ucell->omega);
        energy += static_cast<double>(exx_density_potential_mul_op<T, Device>()(density_recip_batch,
                                                                                density_norm_batch,
                                                                                pot_ik_iq,
                                                                                energy_batch,
                                                                                const_cast<Real*>(batch_weight_device),
                                                                                rhopw_dev->npw,
                                                                                batch_count));
    }

    return energy;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op_qtile(const int nbands,
                                            const int nbasis,
                                            const int npol,
                                            const T* tmpsi_in,
                                            T* tmhpsi,
                                            const int ngk_ik,
                                            const bool is_first_node,
                                            bool accumulate_hpsi,
                                            int ispin_override,
                                            const K_Vectors::ExxFullKPoint* target_kpoint_override,
                                            int target_ik_override) const
{
    ModuleBase::timer::start("OperatorEXXPW", "act_op_qtile");

    const bool is_cpu = std::is_same<Device, base_device::DEVICE_CPU>::value;
    const bool synchronized_ace_call = ispin_override >= 0;
    if (!is_cpu && GlobalV::KPAR > 1 && !synchronized_ace_call)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::act_op_qtile",
                                 "direct noACE GPU q-tile PW EXX supports KPAR=1 only; "
                                 "KPAR>1 is supported only inside synchronized ACE construction");
    }

    const int nspin_fac = options.nspin == 2 ? 2 : 1;
    const int ispin = ispin_override >= 0 ? ispin_override : (this->ik < (wfcpw->nks / nspin_fac) ? 0 : 1);
    const K_Vectors::ExxFullKPoint* local_kpoint = nullptr;
    if (accumulate_hpsi)
    {
        local_kpoint = target_kpoint_override != nullptr ? target_kpoint_override
                                                         : &local_representative_kpoint(this->ik, ispin);
    }
    const int target_ik = target_ik_override >= 0 ? target_ik_override : this->ik;
    // Mixed band target mode uses the target outer k point, but its exchange
    // source q states and weights always come from the converged SCF EXX map.
    auto q_points = source_op_for_target != nullptr ? source_op_for_target->get_active_q_points()
                                                    : get_q_points(this->ik);

    const int nbands_psi = psi.get_nbands();
    const int requested_target_tile_size = std::max(1, std::min(options.band_tile_size, nbands));
    const int target_tile_size = (!is_cpu && GlobalV::KPAR > 1 && synchronized_ace_call) ? 1
                                                                                         : requested_target_tile_size;
    const int source_tile_size = is_cpu ? std::max(1, std::min(target_tile_size, nbands_psi))
                                        : std::max(1, std::min(options.band_tile_size, nbands_psi));
    const int q_tile_size = std::max(1, std::min(options.q_tile_size, static_cast<int>(q_points.size())));
    const int chunk_size = std::min(resolve_qtile_chunk_size(), source_tile_size);
    const std::size_t real_size = static_cast<std::size_t>(wfcpw_exx->nrxx);
    const std::size_t target_size = static_cast<std::size_t>(target_tile_size) * real_size;
    const std::size_t q_size = static_cast<std::size_t>(q_tile_size) * static_cast<std::size_t>(source_tile_size)
                               * real_size;
    const std::size_t q_weight_count = static_cast<std::size_t>(q_tile_size) * static_cast<std::size_t>(source_tile_size);

    ensure_qtile_workspace(target_size, q_size, std::max(chunk_size, static_cast<int>(q_weight_count)));

    std::vector<T> target_real_host;
    std::vector<T> h_real_host;
    std::vector<T> q_real_host;
    if (is_cpu)
    {
        target_real_host.resize(target_size);
        h_real_host.resize(target_size);
        q_real_host.resize(q_size);
    }
    T* target_real = is_cpu ? target_real_host.data() : qtile_target_real;
    T* h_real = is_cpu ? h_real_host.data() : qtile_h_real;
    T* q_real = is_cpu ? q_real_host.data() : qtile_q_real;
    std::vector<Real> q_weights(q_weight_count, 0);

    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    for (int n_start = 0; n_start < nbands; n_start += target_tile_size)
    {
        const int n_count = std::min(target_tile_size, nbands - n_start);
        if (is_cpu)
        {
            std::fill(target_real_host.begin(), target_real_host.end(), T(0));
            std::fill(h_real_host.begin(), h_real_host.end(), T(0));
        }
        else
        {
            setmem_complex_op()(qtile_target_real, 0, target_size);
            setmem_complex_op()(qtile_h_real, 0, target_size);
        }

        if (accumulate_hpsi)
        {
            ModuleBase::timer::start("OperatorEXXPW", "target_tile");
            fill_target_tile(tmpsi_in, nbasis, n_start, n_count, target_real);
            ModuleBase::timer::end("OperatorEXXPW", "target_tile");
        }

        for (int q_start = 0; q_start < static_cast<int>(q_points.size()); q_start += q_tile_size)
        {
            const int q_count = std::min(q_tile_size, static_cast<int>(q_points.size()) - q_start);
            for (int m_start = 0; m_start < nbands_psi; m_start += source_tile_size)
            {
                const int m_count = std::min(source_tile_size, nbands_psi - m_start);
                if (is_cpu)
                {
                    std::fill(q_real_host.begin(), q_real_host.end(), T(0));
                }
                else
                {
                    setmem_complex_op()(qtile_q_real, 0, q_size);
                }
                std::fill(q_weights.begin(), q_weights.end(), Real(0));

                const Real tile_weight_sum = fill_q_tile_states(q_points,
                                                                q_start,
                                                                q_count,
                                                                ispin,
                                                                m_start,
                                                                m_count,
                                                                source_tile_size,
                                                                q_real,
                                                                q_weights.data(),
                                                                true);
                if (tile_weight_sum < std::numeric_limits<Real>::epsilon() || !accumulate_hpsi)
                {
                    continue;
                }

                const T* q_weights_device = nullptr;
                if (!is_cpu)
                {
                    alpha_all_host_work.assign(q_weight_count, T{});
                    for (std::size_t iw = 0; iw < q_weight_count; ++iw)
                    {
                        alpha_all_host_work[iw] = static_cast<T>(q_weights[iw]);
                    }
                    if (alpha_all_capacity < q_weight_count)
                    {
                        delmem_complex_op()(alpha_all_device);
                        alpha_all_device = nullptr;
                        resmem_complex_op()(alpha_all_device, q_weight_count);
                        alpha_all_capacity = q_weight_count;
                    }
                    syncmem_complex_c2d_op()(alpha_all_device, alpha_all_host_work.data(), q_weight_count);
                    q_weights_device = alpha_all_device;
                }

                for (int q_local = 0; q_local < q_count; ++q_local)
                {
                    process_qtile_apply_tile(*local_kpoint,
                                             *q_points[q_start + q_local],
                                             target_real,
                                             h_real,
                                             q_real,
                                             q_weights.data(),
                                             q_weights_device,
                                             q_local,
                                             n_count,
                                             m_count,
                                             source_tile_size,
                                             chunk_size);
                }
            }
        }

        if (accumulate_hpsi)
        {
            ModuleBase::timer::start("OperatorEXXPW", "target_tile_out");
            for (int n_local = 0; n_local < n_count; ++n_local)
            {
                const int n_iband = n_start + n_local;
                T* h_psi_nk = tmhpsi + static_cast<std::size_t>(n_iband) * nbasis;
                const Real hybrid_alpha = options.hybrid_alpha;
                exx_real_to_wave_recip(h_real + static_cast<std::size_t>(n_local) * real_size,
                                       h_psi_nk,
                                       target_ik,
                                       hybrid_alpha);
            }
            ModuleBase::timer::end("OperatorEXXPW", "target_tile_out");
        }
    }

    ModuleBase::timer::end("OperatorEXXPW", "act_op_qtile");
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op_qtile_cpu(const int nbands,
                                                const int nbasis,
                                                const int npol,
                                                const T *tmpsi_in,
                                                T *tmhpsi,
                                                const int ngk_ik,
                                                const bool is_first_node,
                                                bool accumulate_hpsi,
                                                int ispin_override) const
{
    if (!std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::act_op_qtile_cpu", "CPU q-tile PW EXX wrapper called on GPU");
    }
    act_op_qtile(nbands,
                 nbasis,
                 npol,
                 tmpsi_in,
                 tmhpsi,
                 ngk_ik,
                 is_first_node,
                 accumulate_hpsi,
                 ispin_override,
                 nullptr,
                 -1);
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op_qtile_gpu(const int nbands,
                                                const int nbasis,
                                                const int npol,
                                                const T *tmpsi_in,
                                                T *tmhpsi,
                                                const int ngk_ik,
                                                const bool is_first_node,
                                                bool accumulate_hpsi,
                                                int ispin_override) const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::act_op_qtile_gpu", "GPU q-tile PW EXX wrapper called on CPU");
    }
    act_op_qtile(nbands,
                 nbasis,
                 npol,
                 tmpsi_in,
                 tmhpsi,
                 ngk_ik,
                 is_first_node,
                 accumulate_hpsi,
                 ispin_override,
                 nullptr,
                 -1);
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::set_psi(psi::Psi<T, Device>& psi_in) const
{
    set_psi_for_cache(psi_in);
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::set_psi_for_cache(const psi::Psi<T, Device>& psi_in) const
{
    psi = psi_in;
}

template <typename T, typename Device>
std::vector<const K_Vectors::ExxFullQPoint*> OperatorEXXPW<T, Device>::get_q_points(const int ik) const
{
    // stored in q_points
    if (q_points.find(ik) != q_points.end())
    {
        return q_points.find(ik)->second;
    }

    std::vector<const K_Vectors::ExxFullQPoint*> q_points_ik;
    for (const auto& qpoint: kv->exx_full_q_map)
    {
        if (!qpoint.active)
        {
            continue;
        }
        q_points_ik.push_back(&qpoint);
    }

    if (q_points_ik.empty())
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::get_q_points", "empty EXX full-q map");
    }

    q_points[ik] = q_points_ik;
    return q_points_ik;
}

template <typename T, typename Device>
std::vector<const K_Vectors::ExxFullQPoint*> OperatorEXXPW<T, Device>::get_active_q_points() const
{
    std::vector<const K_Vectors::ExxFullQPoint*> q_points_active;
    for (const auto& qpoint: kv->exx_full_q_map)
    {
        if (qpoint.active)
        {
            q_points_active.push_back(&qpoint);
        }
    }

    if (q_points_active.empty())
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::get_active_q_points", "empty EXX full-q map");
    }

    return q_points_active;
}

template <typename T, typename Device>
std::vector<const K_Vectors::ExxFullKPoint*> OperatorEXXPW<T, Device>::get_k_points() const
{
    if (!k_points.empty())
    {
        return k_points;
    }

    for (const auto& kpoint: kv->exx_full_k_map)
    {
        if (!kpoint.active)
        {
            continue;
        }
        k_points.push_back(&kpoint);
    }

    if (k_points.empty())
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::get_k_points", "empty EXX full-k map");
    }

    return k_points;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::multiply_potential(T *density_recip, int ik, int iq) const
{
    ModuleBase::timer::start("OperatorEXXPW", "multiply_potential");
    int npw = rhopw_dev->npw;
    int nks = wfcpw->nks;
    mul_potential_op<T, Device>()(pot, density_recip, npw, nks, ik, iq);

    ModuleBase::timer::end("OperatorEXXPW", "multiply_potential");
}

template <typename T, typename Device>
typename OperatorEXXPW<T, Device>::Real*
OperatorEXXPW<T, Device>::get_exx_potential_cached(const K_Vectors::ExxFullKPoint& kpoint,
                                                   const K_Vectors::ExxFullQPoint& qpoint) const
{
    if (cached_potential_ik != this->ik)
    {
        clear_exx_potential_cache();
        cached_potential_ik = this->ik;
    }

    const auto cache_key = std::make_pair(kpoint.full_index, qpoint.full_index);
    auto cache_it = pot_cache.find(cache_key);
    if (cache_it != pot_cache.end())
    {
        return cache_it->second;
    }

    Real* pot_new = nullptr;
    resmem_real_op()(pot_new, rhopw_dev->npw);
    get_exx_potential<Real, Device>(kv,
                                    wfcpw,
                                    rhopw_dev,
                                    pot_new,
                                    tpiba,
                                    singular_correction_mode,
                                    fock_div_local.empty() ? nullptr : &fock_div_local,
	                                    erfc_div_local.empty() ? nullptr : &erfc_div_local,
	                                    ucell->omega,
	                                    kpoint,
	                                    qpoint,
	                                    false);
    pot_cache[cache_key] = pot_new;
    return pot_new;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::clear_exx_potential_cache() const
{
    for (auto& entry: pot_cache)
    {
        delmem_real_op()(entry.second);
    }
    pot_cache.clear();
}

template <typename T, typename Device>
const T *OperatorEXXPW<T, Device>::get_pw(const int m, const int ik_local) const
{
    psi.fix_kb(ik_local, m);
    T* psi_mq = psi.get_pointer();
    return psi_mq;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::ensure_exx_wave_mapping() const
{
    if (!exx_to_wfc_offsets.empty())
    {
        return;
    }

    exx_to_wfc_offsets.assign(wfcpw_exx->nks + 1, 0);
    exx_to_wfc_map_host.clear();
    exx_to_wfc_map_host.reserve(static_cast<std::size_t>(wfcpw_exx->npwk_max) * wfcpw_exx->nks);
    for (int ik = 0; ik < wfcpw_exx->nks; ++ik)
    {
        exx_to_wfc_offsets[ik] = static_cast<int>(exx_to_wfc_map_host.size());
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
    exx_to_wfc_offsets[wfcpw_exx->nks] = static_cast<int>(exx_to_wfc_map_host.size());

    if (!std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        if (exx_to_wfc_map_device_capacity < exx_to_wfc_map_host.size())
        {
            base_device::memory::delete_memory_op<int, Device>()(exx_to_wfc_map_device);
            exx_to_wfc_map_device = nullptr;
            base_device::memory::resize_memory_op<int, Device>()(exx_to_wfc_map_device, exx_to_wfc_map_host.size());
            exx_to_wfc_map_device_capacity = exx_to_wfc_map_host.size();
        }
        base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>()(
            exx_to_wfc_map_device,
            exx_to_wfc_map_host.data(),
            exx_to_wfc_map_host.size());
    }
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::wave_recip_to_exx_real(const T* psi_recip, T* psi_real, int ik_local) const
{
    const T* exx_recip = wave_recip_to_exx_recip(psi_recip, ik_local, psi_nk_exx_recip);
    wfcpw_exx->recip_to_real(ctx, exx_recip, psi_real, ik_local);
}

template <typename T, typename Device>
const T* OperatorEXXPW<T, Device>::wave_recip_to_exx_recip(const T* psi_recip, int ik_local, T* scratch) const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
    {
        exx_wave_redistributor->wfc_to_exx(ik_local, psi_recip, scratch);
        return scratch;
    }
    ensure_exx_wave_mapping();
    const int offset = exx_to_wfc_offsets[ik_local];
    const int* map = std::is_same<Device, base_device::DEVICE_CPU>::value
                         ? exx_to_wfc_map_host.data() + offset
                         : exx_to_wfc_map_device + offset;
    exx_gather_recip_op<T, Device>()(psi_recip, scratch, map, wfcpw_exx->npwk[ik_local]);
    return scratch;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::exx_real_to_wave_recip(const T* psi_real,
                                                      T* h_psi_recip_out,
                                                      int ik_local,
                                                      Real factor) const
{
    setmem_complex_op()(h_psi_exx_recip, 0, wfcpw_exx->npwk_max);
    wfcpw_exx->real_to_recip(ctx, psi_real, h_psi_exx_recip, ik_local, false, Real(1.0));
    if (std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
    {
        exx_wave_redistributor->exx_to_wfc_add(ik_local, h_psi_exx_recip, h_psi_recip_out, T(factor, 0));
        return;
    }
    ensure_exx_wave_mapping();
    const int offset = exx_to_wfc_offsets[ik_local];
    const int* map = std::is_same<Device, base_device::DEVICE_CPU>::value
                         ? exx_to_wfc_map_host.data() + offset
                         : exx_to_wfc_map_device + offset;
    exx_scatter_add_recip_op<T, Device>()(h_psi_exx_recip,
                                          h_psi_recip_out,
                                          map,
                                          wfcpw_exx->npwk[ik_local],
                                          T(factor, 0));
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::ensure_full_point_supported(const K_Vectors::ExxFullPoint& point) const
{
}

template <typename T, typename Device>
int OperatorEXXPW<T, Device>::rep_spin_index(const K_Vectors::ExxFullPoint& point, int ispin) const
{
    return kv->exx_rep_spin_index(point, ispin);
}

template <typename T, typename Device>
const K_Vectors::ExxFullKPoint& OperatorEXXPW<T, Device>::local_representative_kpoint(int ik_local, int ispin) const
{
    auto cache_it = local_kpoint_cache.find(ik_local);
    if (cache_it != local_kpoint_cache.end())
    {
        return cache_it->second;
    }

    const int nspin_fac = options.nspin == 2 ? 2 : 1;
    const int nk_local_no_spin = wfcpw->nks / nspin_fac;
    const int ik_local_no_spin = ik_local % nk_local_no_spin;

    for (const auto& kpoint: kv->exx_full_k_map)
    {
        if (!kpoint.active || kpoint.rep_pool != GlobalV::MY_POOL || kpoint.rep_local_index != ik_local_no_spin)
        {
            continue;
        }
        if (kpoint.full_index == kpoint.rep_index)
        {
            auto inserted = local_kpoint_cache.emplace(ik_local, kpoint);
            return inserted.first->second;
        }
    }

    K_Vectors::ExxFullKPoint point;
    const int ik_global_no_spin = ik_local_no_spin < static_cast<int>(kv->ik2iktot.size())
                                      ? kv->ik2iktot[ik_local_no_spin]
                                      : ik_local_no_spin;
    point.full_index = ik_global_no_spin;
    point.rep_index = point.full_index;
    point.rep_local_index = ik_local_no_spin;
    point.rep_pool = GlobalV::MY_POOL;
    point.symop = 0;
    point.time_reversal = false;
    point.conjugate_only = false;
    point.identity = true;
    point.active = true;
    point.weight = ik_local < static_cast<int>(kv->wk.size()) ? kv->wk[ik_local] : 1.0;
    point.full_kvec_d = wfcpw->kvec_d[ik_local];
    point.full_kvec_c = wfcpw->kvec_c[ik_local];
    point.gmatrix = ModuleBase::Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    point.kgmatrix = ModuleBase::Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1);
    point.gtrans = ModuleBase::Vector3<double>(0, 0, 0);

    auto inserted = local_kpoint_cache.emplace(ik_local, point);
    return inserted.first->second;
}

template <typename T, typename Device>
const typename OperatorEXXPW<T, Device>::FullPointSpatialRemap&
OperatorEXXPW<T, Device>::point_spatial_remap(const K_Vectors::ExxFullPoint& point,
                                              int point_rep_spin) const
{
    const auto cache_key = std::make_pair(point.full_index, point_rep_spin);
    auto cache_it = point_gmaps.find(cache_key);
    if (cache_it != point_gmaps.end())
    {
        return cache_it->second;
    }

    FullPointSpatialRemap remap = build_exx_symmetry_remap(wfcpw_exx,
                                                           point,
                                                           point_rep_spin,
                                                           wfcpw_exx->get_device() == "gpu");

    auto inserted = point_gmaps.emplace(cache_key, std::move(remap));
    return inserted.first->second;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::ensure_remap_device_cache(const FullPointSpatialRemap& remap) const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        return;
    }
#if defined(__CUDA)
    const std::size_t npw_full = remap.rep_igl.size();
    if (remap.rep_igl_device == nullptr)
    {
        base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>()(remap.rep_igl_device, npw_full);
        base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            remap.rep_igl_device,
            remap.rep_igl.data(),
            npw_full);
    }
    if (remap.fft_ixyz_device == nullptr)
    {
        base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>()(remap.fft_ixyz_device, npw_full);
        base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            remap.fft_ixyz_device,
            remap.fft_ixyz.data(),
            npw_full);
    }
    if (std::is_same<Real, float>::value)
    {
        if (remap.phase_float_device == nullptr)
        {
            std::vector<std::complex<float>> phase_float(npw_full);
            for (std::size_t ig = 0; ig < npw_full; ++ig)
            {
                phase_float[ig] = std::complex<float>(static_cast<float>(remap.phase[ig].real()),
                                                      static_cast<float>(remap.phase[ig].imag()));
            }
            base_device::memory::resize_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(
                remap.phase_float_device,
                npw_full);
            base_device::memory::synchronize_memory_op<std::complex<float>,
                                                       base_device::DEVICE_GPU,
                                                       base_device::DEVICE_CPU>()(
                remap.phase_float_device,
                phase_float.data(),
                npw_full);
        }
    }
    else
    {
        if (remap.phase_double_device == nullptr)
        {
            std::vector<std::complex<double>> phase_double(npw_full);
            for (std::size_t ig = 0; ig < npw_full; ++ig)
            {
                phase_double[ig] = std::complex<double>(remap.phase[ig].real(), remap.phase[ig].imag());
            }
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
                remap.phase_double_device,
                npw_full);
            base_device::memory::synchronize_memory_op<std::complex<double>,
                                                       base_device::DEVICE_GPU,
                                                       base_device::DEVICE_CPU>()(
                remap.phase_double_device,
                phase_double.data(),
                npw_full);
        }
    }
#endif
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::clear_remap_device_cache() const
{
#if defined(__CUDA)
    for (auto& entry: point_gmaps)
    {
        auto& remap = entry.second;
        base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>()(remap.rep_igl_device);
        base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>()(remap.fft_ixyz_device);
        base_device::memory::delete_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(remap.phase_float_device);
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            remap.phase_double_device);
        remap.rep_igl_device = nullptr;
        remap.fft_ixyz_device = nullptr;
        remap.phase_float_device = nullptr;
        remap.phase_double_device = nullptr;
    }
#endif
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::load_full_point_real(const K_Vectors::ExxFullPoint& point,
                                                    int ispin,
                                                    int iband,
                                                    T* out) const
{
    load_full_point_real_uncached(point, ispin, iband, out);
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::load_full_point_real_uncached(const K_Vectors::ExxFullPoint& point,
                                                             int ispin,
                                                             int iband,
                                                             T* out) const
{
    ensure_full_point_supported(point);
    const int point_rep_spin = rep_spin_index(point, ispin);
    const T* psi_point = get_pw(iband, point_rep_spin);
    if (point.identity || point.conjugate_only)
    {
        wave_recip_to_exx_real(psi_point, out, point_rep_spin);
        if (point.conjugate_only)
        {
            exx_conjugate_real_op<T, Device>()(out, out, wfcpw_exx->nrxx);
        }
    }
    else
    {
        const auto& remap = point_spatial_remap(point, point_rep_spin);
        const T* psi_point_exx = wave_recip_to_exx_recip(psi_point, point_rep_spin, psi_mq_exx_recip);
        if (std::is_same<Device, base_device::DEVICE_CPU>::value)
        {
            if (point.time_reversal)
            {
                wfcpw_exx->recip2real_remapped_conjugate(psi_point_exx,
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
                wfcpw_exx->recip2real_remapped(psi_point_exx,
                                               out,
                                               static_cast<int>(remap.rep_igl.size()),
                                               remap.rep_igl.data(),
                                               remap.fft_isz.data(),
                                               remap.phase.data(),
                                               false,
                                               Real(1.0));
            }
        }
        else
        {
            if (point.time_reversal)
            {
                wfcpw_exx->recip2real_remapped_conjugate(psi_point_exx,
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
                wfcpw_exx->recip2real_remapped(psi_point_exx,
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
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::load_full_point_real_batch(const K_Vectors::ExxFullPoint& point,
                                                          int ispin,
                                                          const int* band_indices,
                                                          int batch_count,
                                                          T* out) const
{
    if (std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        for (int ib = 0; ib < batch_count; ++ib)
        {
            load_full_point_real(point, ispin, band_indices[ib], out + static_cast<std::size_t>(ib) * wfcpw_exx->nrxx);
        }
        return;
    }
    ensure_full_point_supported(point);
    const int point_rep_spin = rep_spin_index(point, ispin);
    const auto& remap = point_spatial_remap(point, point_rep_spin);
    const bool direct_source = consecutive_integers(band_indices, batch_count) && psi.get_k_first();
    const T* in_batch = nullptr;
    if (direct_source && (point.identity || point.conjugate_only))
    {
        for (int ib = 0; ib < batch_count; ++ib)
        {
            wave_recip_to_exx_recip(get_pw(band_indices[ib], point_rep_spin),
                                    point_rep_spin,
                                    psi_mq_batch_recip + static_cast<std::size_t>(ib) * wfcpw_exx->npwk_max);
        }
        wfcpw_exx->recip_to_real_batch<Real, Device>(ctx,
                                                     psi_mq_batch_recip,
                                                     out,
                                                     point_rep_spin,
                                                     batch_count,
                                                     false,
                                                     Real(1.0));
        if (point.conjugate_only)
        {
            exx_conjugate_real_op<T, Device>()(out, out, static_cast<std::size_t>(batch_count) * wfcpw_exx->nrxx);
        }
        return;
    }

    {
        for (int ib = 0; ib < batch_count; ++ib)
        {
            const T* psi_point = get_pw(band_indices[ib], point_rep_spin);
            wave_recip_to_exx_recip(psi_point,
                                    point_rep_spin,
                                    psi_mq_batch_recip + static_cast<std::size_t>(ib) * wfcpw_exx->npwk_max);
        }
        in_batch = psi_mq_batch_recip;
    }

    ensure_remap_device_cache(remap);
    const T* phase_device = nullptr;
#if defined(__CUDA)
    if (!std::is_same<Device, base_device::DEVICE_CPU>::value)
    {
        if (std::is_same<Real, float>::value)
        {
            phase_device = reinterpret_cast<const T*>(remap.phase_float_device);
        }
        else
        {
            phase_device = reinterpret_cast<const T*>(remap.phase_double_device);
        }
    }
#endif
    wfcpw_exx->recip2real_remapped_batch<Real, Device>(ctx,
                                                       in_batch,
                                                       out,
                                                       static_cast<int>(remap.rep_igl.size()),
                                                       remap.rep_igl_device == nullptr ? remap.rep_igl.data()
                                                                                       : remap.rep_igl_device,
                                                       remap.fft_isz.data(),
                                                       remap.fft_ixyz_device,
                                                       remap.phase.data(),
                                                       reinterpret_cast<const std::complex<Real>*>(phase_device),
                                                       batch_count,
                                                       point.time_reversal,
                                                       false,
                                                       Real(1.0));
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::accumulate_full_point_recip(const K_Vectors::ExxFullPoint& point,
                                                           int ispin,
                                                           const T* full_real,
                                                           T* rep_recip,
                                                           Real factor) const
{
    ensure_full_point_supported(point);
    const int point_rep_spin = rep_spin_index(point, ispin);
    if (point.identity)
    {
        exx_real_to_wave_recip(full_real, rep_recip, point_rep_spin, factor);
    }
    else if (point.conjugate_only)
    {
        exx_conjugate_real_op<T, Device>()(full_real, density_real, wfcpw_exx->nrxx);
        exx_real_to_wave_recip(density_real, rep_recip, point_rep_spin, factor);
    }
    else
    {
        const auto& remap = point_spatial_remap(point, point_rep_spin);
        setmem_complex_op()(h_psi_exx_recip, 0, wfcpw_exx->npwk_max);
        if (std::is_same<Device, base_device::DEVICE_CPU>::value)
        {
            if (point.time_reversal)
            {
                wfcpw_exx->real2recip_remapped_conjugate(full_real,
                                                         h_psi_exx_recip,
                                                         static_cast<int>(remap.rep_igl.size()),
                                                         remap.rep_igl.data(),
                                                         remap.fft_isz.data(),
                                                         remap.phase.data(),
                                                         false,
                                                         Real(1.0));
            }
            else
            {
                wfcpw_exx->real2recip_remapped(full_real,
                                               h_psi_exx_recip,
                                               static_cast<int>(remap.rep_igl.size()),
                                               remap.rep_igl.data(),
                                               remap.fft_isz.data(),
                                               remap.phase.data(),
                                               false,
                                               Real(1.0));
            }
        }
        else
        {
            if (point.time_reversal)
            {
                wfcpw_exx->real2recip_remapped_conjugate(full_real,
                                                         h_psi_exx_recip,
                                                         static_cast<int>(remap.rep_igl.size()),
                                                         remap.rep_igl.data(),
                                                         remap.fft_isz.data(),
                                                         remap.phase.data(),
                                                         false,
                                                         Real(1.0));
            }
            else
            {
                wfcpw_exx->real2recip_remapped(full_real,
                                               h_psi_exx_recip,
                                               static_cast<int>(remap.rep_igl.size()),
                                               remap.rep_igl.data(),
                                               remap.fft_isz.data(),
                                               remap.phase.data(),
                                               false,
                                               Real(1.0));
            }
        }
        ensure_exx_wave_mapping();
        const int offset = exx_to_wfc_offsets[point_rep_spin];
        const int* map = std::is_same<Device, base_device::DEVICE_CPU>::value
                             ? exx_to_wfc_map_host.data() + offset
                             : exx_to_wfc_map_device + offset;
        exx_scatter_add_recip_op<T, Device>()(h_psi_exx_recip,
                                              rep_recip,
                                              map,
                                              wfcpw_exx->npwk[point_rep_spin],
                                              T(factor, 0));
    }
}

template <typename T, typename Device>
template <typename T_in, typename Device_in>
OperatorEXXPW<T, Device>::OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op)
{
    // copy all the datas
    this->isk = op->isk;
    this->wfcpw = op->wfcpw;
    this->wfcpw_exx = op->wfcpw_exx;
    this->rhopw = op->rhopw;
    this->rhopw_dev = op->rhopw_dev;
    this->kv = op->kv;
    this->ucell = op->ucell;
    this->tpiba = op->tpiba;
    this->owns_wfcpw_exx = false;
    this->owns_rhopw_dev = false;
    this->psi = op->psi;
    this->ctx = op->ctx;
    this->cpu_ctx = op->cpu_ctx;
    this->gamma_extrapolation = op->gamma_extrapolation;
    this->singular_correction_mode = op->singular_correction_mode;
    this->fock_div_local = op->fock_div_local;
    this->erfc_div_local = op->erfc_div_local;
    if (std::is_same<Device, base_device::DEVICE_CPU>::value && wfcpw->poolnproc > 1)
    {
        exx_wave_redistributor.reset(new ExxWaveRedistributorCpu<T>());
        exx_wave_redistributor->setup(wfcpw, wfcpw_exx);
    }
    resmem_complex_op()(this->ctx, psi_nk_real, wfcpw_exx->nrxx);
    resmem_complex_op()(this->ctx, psi_mq_real, wfcpw_exx->nrxx);
    resmem_complex_op()(this->ctx, density_real, rhopw_dev->nrxx);
    resmem_complex_op()(this->ctx, h_psi_real, rhopw_dev->nrxx);
    resmem_complex_op()(this->ctx, density_recip, rhopw_dev->npw);
    resmem_complex_op()(this->ctx, h_psi_recip, wfcpw->npwk_max);
    resmem_complex_op()(this->ctx, psi_nk_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(this->ctx, psi_mq_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(this->ctx, h_psi_exx_recip, wfcpw_exx->npwk_max);
//    this->pws.resize(wfcpw->nks);


}

template <typename T, typename Device>
OperatorEXXPW<T, Device>::OperatorEXXPW(const OperatorEXXPW<T, Device>* source_op,
                                        const int* target_isk,
                                        const ModulePW::PW_Basis_K* target_wfcpw,
                                        const K_Vectors* target_kv,
                                        const ExxOperatorOptions& options_in)
{
    this->isk = target_isk;
    this->wfcpw = target_wfcpw;
    this->rhopw = source_op->rhopw;
    this->rhopw_dev = source_op->rhopw_dev;
    this->owns_wfcpw_exx = true;
    this->owns_rhopw_dev = false;
    this->source_op_for_target = source_op;
    this->target_kv_for_target = target_kv;
    this->kv = source_op->kv;
    this->options = options_in;
    this->ucell = source_op->ucell;
    this->tpiba = source_op->tpiba;
    this->psi = source_op->psi;
    this->wg = source_op->wg;
    this->p_wg = source_op->p_wg;
    this->first_iter = false;
    this->gamma_extrapolation = false;
    this->singular_correction_mode = ExxSingularCorrectionMode::SmoothTarget;
    this->classname = "OperatorEXXPW";
    this->cal_type = hamilt::calculation_type::pw_exx;
    this->ctx = nullptr;
    this->cpu_ctx = nullptr;

    const std::string exx_precision = std::is_same<Real, float>::value ? "single" : "double";
    this->wfcpw_exx = new ModulePW::PW_Basis_K(target_wfcpw->get_device(), exx_precision);
    this->wfcpw_exx->fft_bundle.setfft(target_wfcpw->get_device(), exx_precision);
#ifdef __MPI
    this->wfcpw_exx->initmpi(target_wfcpw->poolnproc, target_wfcpw->poolrank, target_wfcpw->pool_world);
#endif
    this->wfcpw_exx->initgrids(target_wfcpw->lat0,
                               target_wfcpw->latvec,
                               source_op->rhopw_dev->nx,
                               source_op->rhopw_dev->ny,
                               source_op->rhopw_dev->nz);
    this->wfcpw_exx->initparameters(target_wfcpw->gamma_only,
                                    source_op->rhopw_dev->ggecut,
                                    target_wfcpw->nks,
                                    target_wfcpw->kvec_d,
                                    target_wfcpw->distribution_type,
                                    target_wfcpw->xprime);
    this->wfcpw_exx->setuptransform(options.batch_fft_size);
    this->wfcpw_exx->collect_local_pw();

    if (this->wfcpw_exx->nrxx != source_op->wfcpw_exx->nrxx
        || this->rhopw_dev->nrxx != source_op->rhopw_dev->nrxx)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::OperatorEXXPW",
                                 "mixed band target EXX requires source and target EXX FFT grids to match; "
                                 "check ecutwfc/ecutexx and target PW setup");
    }
    if (std::is_same<Device, base_device::DEVICE_CPU>::value && target_wfcpw->poolnproc > 1)
    {
        exx_wave_redistributor.reset(new ExxWaveRedistributorCpu<T>());
        exx_wave_redistributor->setup(target_wfcpw, wfcpw_exx);
    }

    for (const auto& param: options.fock_params)
    {
        fock_div_local.push_back(exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type::Fock,
                                                0.0,
                                                this->kv,
                                                this->wfcpw,
                                                this->rhopw_dev,
                                                this->tpiba,
                                                this->singular_correction_mode,
                                                this->ucell->omega));
    }
    for (const auto& param: options.erfc_params)
    {
        erfc_div_local.push_back(exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type::Erfc,
                                                std::stod(param.at("omega")),
                                                this->kv,
                                                this->wfcpw,
                                                this->rhopw_dev,
                                                this->tpiba,
                                                this->singular_correction_mode,
                                                this->ucell->omega));
    }
    if (GlobalV::MY_RANK == 0)
    {
        GlobalV::ofs_running << " Mixed band target EXX singular correction = smooth target-k"
                             << " (source q mesh fixed, MP gamma mask disabled for target k)" << std::endl;
    }

    resmem_complex_op()(psi_nk_real, wfcpw_exx->nrxx);
    resmem_complex_op()(psi_mq_real, wfcpw_exx->nrxx);
    resmem_complex_op()(h_psi_recip, target_wfcpw->npwk_max);
    resmem_complex_op()(density_real, rhopw_dev->nrxx);
    resmem_complex_op()(h_psi_real, rhopw_dev->nrxx);
    resmem_complex_op()(density_recip, rhopw_dev->npw);
    resmem_complex_op()(psi_nk_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(psi_mq_exx_recip, wfcpw_exx->npwk_max);
    resmem_complex_op()(h_psi_exx_recip, wfcpw_exx->npwk_max);
    resmem_real_op()(pot, rhopw_dev->npw);

    int batch_fft_size = this->wfcpw_exx->fft_bundle.get_batch_size<Real>();
    if (batch_fft_size <= 0 && target_wfcpw->get_device() == "gpu")
    {
        batch_fft_size = 1;
    }
    if (batch_fft_size > 0)
    {
        resmem_complex_op()(psi_mq_batch_real, batch_fft_size * wfcpw_exx->nrxx);
        resmem_complex_op()(psi_mq_batch_recip, batch_fft_size * wfcpw_exx->npwk_max);
        resmem_complex_op()(density_real_batch, batch_fft_size * rhopw_dev->nrxx);
        resmem_complex_op()(density_recip_batch, batch_fft_size * rhopw_dev->npw);
        resmem_real_op()(density_norm_batch, batch_fft_size * rhopw_dev->npw);
        resmem_real_op()(energy_batch, batch_fft_size);
    }
}

template <typename T, typename Device>
double OperatorEXXPW<T, Device>::cal_exx_energy(psi::Psi<T, Device> *psi_) const
{
    if (options.exxace && options.separate_loop)
    {
        return cal_exx_energy_ace(psi_);
    }
    return cal_exx_energy_op_qtile(psi_);
}

template <typename T, typename Device>
double OperatorEXXPW<T, Device>::cal_exx_energy_op_qtile(psi::Psi<T, Device> *ppsi_) const
{
    ModuleBase::timer::start("OperatorEXXPW", "cal_exx_energy_qtile");

    const bool is_cpu = std::is_same<Device, base_device::DEVICE_CPU>::value;
    if (!is_cpu && GlobalV::KPAR > 1)
    {
        ModuleBase::WARNING_QUIT("OperatorEXXPW::cal_exx_energy_op_qtile",
                                 "GPU q-tile PW EXX energy supports KPAR=1 only in this milestone");
    }

    const psi::Psi<T, Device> psi_saved = psi;
    set_psi_for_cache(*ppsi_);

    setmem_complex_op()(psi_nk_real, 0, wfcpw_exx->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw_exx->nrxx);
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    if (wg == nullptr)
    {
        set_psi_for_cache(psi_saved);
        ModuleBase::timer::end("OperatorEXXPW", "cal_exx_energy_qtile");
        return 0.0;
    }

    double Eexx_ik_real = 0.0;
    const int nspin_fac = options.nspin == 2 ? 2 : 1;
    const Real k_spin_degeneracy = options.nspin == 1 ? 2.0 : 1.0;
    const auto k_points = get_k_points();
    auto q_points = get_q_points(0);
    const int nbands_psi = psi.get_nbands();
    const int source_tile_size = std::max(1, std::min(options.band_tile_size, nbands_psi));
    const int q_tile_size = std::max(1, std::min(options.q_tile_size, static_cast<int>(q_points.size())));
    const int chunk_size = std::min(resolve_qtile_chunk_size(), source_tile_size);
    const std::size_t real_size = static_cast<std::size_t>(wfcpw_exx->nrxx);
    const std::size_t q_size = static_cast<std::size_t>(q_tile_size) * static_cast<std::size_t>(source_tile_size)
                               * real_size;
    const std::size_t q_weight_count = static_cast<std::size_t>(q_tile_size) * static_cast<std::size_t>(source_tile_size);

    ensure_qtile_workspace(0, q_size, std::max(chunk_size, static_cast<int>(q_weight_count)));

    std::vector<T> q_real;
    if (is_cpu)
    {
        q_real.resize(q_size);
    }
    std::vector<Real> q_weights(q_weight_count, 0);
    std::vector<Real> q_weight_scaled_host;
    if (!is_cpu)
    {
        q_weight_scaled_host.resize(q_weight_count);
    }
    T* q_real_data = is_cpu ? q_real.data() : qtile_q_real;

    for (int ispin = 0; ispin < nspin_fac; ++ispin)
    {
        for (const auto* kpoint: k_points)
        {
            ensure_full_point_supported(*kpoint);
            const int ik_rep_spin = rep_spin_index(*kpoint, ispin);
            const bool own_kpoint = kpoint->rep_pool == GlobalV::MY_POOL;
            for (int q_start = 0; q_start < static_cast<int>(q_points.size()); q_start += q_tile_size)
            {
                const int q_count = std::min(q_tile_size, static_cast<int>(q_points.size()) - q_start);
                for (int m_start = 0; m_start < nbands_psi; m_start += source_tile_size)
                {
                    const int m_count = std::min(source_tile_size, nbands_psi - m_start);
                    if (is_cpu)
                    {
                        std::fill(q_real.begin(), q_real.end(), T(0));
                    }
                    else
                    {
                        setmem_complex_op()(qtile_q_real, 0, q_size);
                    }
                    std::fill(q_weights.begin(), q_weights.end(), Real(0));
                    const Real tile_weight_sum = fill_q_tile_states(q_points,
                                                                    q_start,
                                                                    q_count,
                                                                    ispin,
                                                                    m_start,
                                                                    m_count,
                                                                    source_tile_size,
                                                                    q_real_data,
                                                                    q_weights.data(),
                                                                    true);
                    if (tile_weight_sum < std::numeric_limits<Real>::epsilon())
                    {
                        continue;
                    }

                    for (int n_iband = 0; n_iband < psi.get_nbands(); n_iband++)
                    {
                        double wg_ikb_real = 0.0;
                        double wk_ik_real = 0.0;
                        if (own_kpoint)
                        {
                            wg_ikb_real = (*wg)(ik_rep_spin, n_iband);
                            wk_ik_real = kv->wk[ik_rep_spin];
                        }
#ifdef __MPI
                        MPI_Bcast(&wg_ikb_real,
                                  1,
                                  MPI_DOUBLE,
                                  kv->para_k.get_startpro_pool(kpoint->rep_pool),
                                  MPI_COMM_WORLD);
                        MPI_Bcast(&wk_ik_real,
                                  1,
                                  MPI_DOUBLE,
                                  kv->para_k.get_startpro_pool(kpoint->rep_pool),
                                  MPI_COMM_WORLD);
#endif
                        const bool active_k = wg_ikb_real >= 1e-12;
                        if (!active_k || !own_kpoint)
                        {
                            continue;
                        }
                        const Real k_weight = static_cast<Real>(wg_ikb_real / wk_ik_real * kpoint->weight
                                                                * k_spin_degeneracy);
                        load_full_point_real(*kpoint, ispin, n_iband, psi_nk_real);

                        const Real* q_weights_device = nullptr;
                        if (!is_cpu)
                        {
                            for (std::size_t iw = 0; iw < q_weight_count; ++iw)
                            {
                                q_weight_scaled_host[iw] = k_weight * q_weights[iw];
                            }
                            syncmem_real_c2d_op()(weight_real_device, q_weight_scaled_host.data(), q_weight_count);
                            q_weights_device = weight_real_device;
                        }

                        for (int q_local = 0; q_local < q_count; ++q_local)
                        {
                            Eexx_ik_real += process_qtile_energy_tile(*kpoint,
                                                                      *q_points[q_start + q_local],
                                                                      psi_nk_real,
                                                                      q_real_data,
                                                                      q_weights.data(),
                                                                      q_weights_device,
                                                                      k_weight,
                                                                      q_local,
                                                                      m_count,
                                                                      source_tile_size,
                                                                      chunk_size);
                        }
                    }
                }
            }
        }
    }

    Eexx_ik_real *= 0.5 * ucell->omega;
    Parallel_Reduce::reduce_all(Eexx_ik_real);

    setmem_complex_op()(psi_nk_real, 0, wfcpw_exx->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw_exx->nrxx);
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    set_psi_for_cache(psi_saved);
    ModuleBase::timer::end("OperatorEXXPW", "cal_exx_energy_qtile");
    return Eexx_ik_real;
}

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::cal_density_recip(const std::complex<double>* psi_nk_real,
                                                                                const std::complex<double>* psi_mq_real,
                                                                                double omega) const
{
    base_device::memory::set_memory_op<std::complex<double>, base_device::DEVICE_CPU>()(
        density_real,
        0,
        rhopw_dev->nrxx);
    cal_density_real_op<std::complex<double>, base_device::DEVICE_CPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw_exx->nrxx);
    rhopw_dev->real2recip(density_real, density_recip);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::cal_density_recip(const std::complex<float>* psi_nk_real,
                                                                              const std::complex<float>* psi_mq_real,
                                                                              double omega) const
{
    base_device::memory::set_memory_op<std::complex<float>, base_device::DEVICE_CPU>()(
        density_real,
        0,
        rhopw_dev->nrxx);
    cal_density_real_op<std::complex<float>, base_device::DEVICE_CPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw_exx->nrxx);
    rhopw_dev->real2recip(density_real, density_recip);
}

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::cal_density_recip_batch(
    const std::complex<double>* psi_nk_real,
    std::complex<double>* psi_mq_real_batch,
    std::complex<double>* density_real_batch,
    std::complex<double>* density_recip_batch,
    int batch_size,
    int ik,
    double omega) const
{
    base_device::memory::set_memory_op<std::complex<double>, base_device::DEVICE_CPU>()(
        density_real_batch,
        0,
        static_cast<std::size_t>(batch_size) * rhopw_dev->nrxx);
    for (int ib = 0; ib < batch_size; ib++)
    {
        std::complex<double>* psi_mq_ib = psi_mq_real_batch + static_cast<std::size_t>(ib) * wfcpw_exx->nrxx;
        std::complex<double>* density_real_ib = density_real_batch + static_cast<std::size_t>(ib) * rhopw_dev->nrxx;
        std::complex<double>* density_recip_ib = density_recip_batch + static_cast<std::size_t>(ib) * rhopw_dev->npw;

        cal_density_real_op<std::complex<double>, base_device::DEVICE_CPU>()(
            psi_nk_real, psi_mq_ib, density_real_ib, omega, wfcpw_exx->nrxx);
        rhopw_dev->real2recip(density_real_ib, density_recip_ib);
    }
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::cal_density_recip_batch(
    const std::complex<float>* psi_nk_real,
    std::complex<float>* psi_mq_real_batch,
    std::complex<float>* density_real_batch,
    std::complex<float>* density_recip_batch,
    int batch_size,
    int ik,
    double omega) const
{
    base_device::memory::set_memory_op<std::complex<float>, base_device::DEVICE_CPU>()(
        density_real_batch,
        0,
        static_cast<std::size_t>(batch_size) * rhopw_dev->nrxx);
    for (int ib = 0; ib < batch_size; ib++)
    {
        std::complex<float>* psi_mq_ib = psi_mq_real_batch + static_cast<std::size_t>(ib) * wfcpw_exx->nrxx;
        std::complex<float>* density_real_ib = density_real_batch + static_cast<std::size_t>(ib) * rhopw_dev->nrxx;
        std::complex<float>* density_recip_ib = density_recip_batch + static_cast<std::size_t>(ib) * rhopw_dev->npw;

        cal_density_real_op<std::complex<float>, base_device::DEVICE_CPU>()(
            psi_nk_real, psi_mq_ib, density_real_ib, omega, wfcpw_exx->nrxx);
        rhopw_dev->real2recip(density_real_ib, density_recip_ib);
    }
}

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::rho_recip2real(const std::complex<double>* rho_recip,
                                                                             std::complex<double>* rho_real,
                                                                             bool add,
                                                                             double factor) const
{
    rhopw_dev->recip2real(rho_recip, rho_real, add, factor);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::rho_recip2real(const std::complex<float>* rho_recip,
                                                                             std::complex<float>* rho_real,
                                                                             bool add,
                                                                             float factor) const
{
    rhopw_dev->recip2real(rho_recip, rho_real, add, factor);
}

template class OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>;
template class OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>;
template class OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>;

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>::cal_density_recip(const std::complex<double>* psi_nk_real,
                                                                                const std::complex<double>* psi_mq_real,
                                                                                double omega) const
{
    base_device::memory::set_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        density_real,
        0,
        rhopw_dev->nrxx);
    cal_density_real_op<std::complex<double>, base_device::DEVICE_GPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw_exx->nrxx);
    rhopw_dev->real2recip_gpu(density_real, density_recip);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::cal_density_recip(const std::complex<float>* psi_nk_real,
                                                                                     const std::complex<float>* psi_mq_real,
                                                                                     double omega) const
{
    base_device::memory::set_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(
        density_real,
        0,
        rhopw_dev->nrxx);
    cal_density_real_op<std::complex<float>, base_device::DEVICE_GPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw_exx->nrxx);
    rhopw_dev->real2recip_gpu(density_real, density_recip);
}

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>::cal_density_recip_batch(
    const std::complex<double>* psi_nk_real,
    std::complex<double>* psi_mq_real_batch,
    std::complex<double>* density_real_batch,
    std::complex<double>* density_recip_batch,
    int batch_size,
    int ik,
    double omega) const
{
    cal_density_real_op<std::complex<double>, base_device::DEVICE_GPU>().operator_batch(
        psi_nk_real,
        psi_mq_real_batch,
        density_real_batch,
        omega,
        wfcpw_exx->nrxx,
        batch_size);
    rhopw_dev->real_to_recip_batch<double, base_device::DEVICE_GPU>(
        this->ctx,
        density_real_batch,
        density_recip_batch,
        batch_size,
        false,
        1.0);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::cal_density_recip_batch(
    const std::complex<float>* psi_nk_real,
    std::complex<float>* psi_mq_real_batch,
    std::complex<float>* density_real_batch,
    std::complex<float>* density_recip_batch,
    int batch_size,
    int ik,
    double omega) const
{
    cal_density_real_op<std::complex<float>, base_device::DEVICE_GPU>().operator_batch(
        psi_nk_real,
        psi_mq_real_batch,
        density_real_batch,
        omega,
        wfcpw_exx->nrxx,
        batch_size);
    rhopw_dev->real_to_recip_batch<float, base_device::DEVICE_GPU>(
        this->ctx,
        density_real_batch,
        density_recip_batch,
        batch_size,
        false,
        1.0f);
}

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>::rho_recip2real(const std::complex<double>* rho_recip,
                                                                             std::complex<double>* rho_real,
                                                                             bool add,
                                                                             double factor) const
{
    rhopw_dev->recip2real_gpu(rho_recip, rho_real, add, factor);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::rho_recip2real(const std::complex<float>* rho_recip,
                                                                             std::complex<float>* rho_real,
                                                                             bool add,
                                                                             float factor) const
{
    rhopw_dev->recip2real_gpu(rho_recip, rho_real, add, factor);
}

#endif

template <typename T, typename Device>
int OperatorEXXPW<T, Device>::get_batch_fft_size() const
{
    return this->wfcpw_exx->fft_bundle.get_batch_size<Real>();
}

template int OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::get_batch_fft_size() const;
template int OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::get_batch_fft_size() const;
#if defined(__CUDA) || defined(__ROCM)
template int OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::get_batch_fft_size() const;
template int OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>::get_batch_fft_size() const;
#endif

} // namespace hamilt

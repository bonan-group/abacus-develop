#include "op_exx_pw.h"

#include "source_base/constants.h"
#include "source_base/global_variable.h"
#include "source_base/parallel_common.h"
#include "source_base/parallel_reduce.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_cell/klist.h"
#include "source_hamilt/operator.h"
#include "source_psi/psi.h"
#include "source_pw/module_pwdft/global.h"
#include "source_pw/module_pwdft/kernels/cal_density_real_op.h"
#include "source_pw/module_pwdft/kernels/exx_cal_energy_op.h"
#include "source_pw/module_pwdft/kernels/mul_potential_op.h"
#include "source_pw/module_pwdft/kernels/vec_mul_vec_complex_op.h"
#include "source_pw/module_pwdft/kernels/axpy_batch_op.h"
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <complex>
#include <cstdlib>
#include <utility>

namespace hamilt
{

bool consecutive_integers(int* arr, size_t size)
{
    if (size == 0) return false;
    for (size_t i = 1; i < size; i++)
    {
        if (arr[i] != arr[i - 1] + 1)
        {
            return false;
        }
    }
    return true;
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
                                        const UnitCell *ucell)
    : isk(isk_in), wfcpw(wfcpw_in), rhopw(rhopw_in), kv(kv_in), ucell(ucell)
{
    if (GlobalV::KPAR != 1 && PARAM.inp.exxace == false)
    {
        // GlobalV::ofs_running << "EXX Calculation does not support k-point parallelism" << std::endl;
        ModuleBase::WARNING_QUIT("OperatorEXXPW", "EXX Calculation does not support k-point parallelism when exxace is set to false");
    }
    gamma_extrapolation = PARAM.inp.exx_gamma_extrapolation;
    bool is_mp = kv_in->get_is_mp();
#ifdef __MPI
    Parallel_Common::bcast_bool(is_mp);
#endif
    if (!is_mp)
    {
        gamma_extrapolation = false;
    }

    this->classname = "OperatorEXXPW";
    this->ctx = nullptr;
    this->cpu_ctx = nullptr;
    this->cal_type = hamilt::calculation_type::pw_exx;

    // allocate real space memory
    // assert(wfcpw->nrxx == rhopw->nrxx);
    resmem_complex_op()(psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(density_real, rhopw->nrxx);
    resmem_complex_op()(h_psi_real, rhopw->nrxx);
    // allocate density recip space memory
    resmem_complex_op()(density_recip, rhopw->npw);
    // allocate h_psi recip space memory
    resmem_complex_op()(h_psi_recip, wfcpw->npwk_max);
    // resmem_complex_op()(this->ctx, psi_all_real, wfcpw->nrxx * GlobalV::NBANDS);

    // Query batch size from FFT infrastructure
    const int batch_fft_size = this->wfcpw->fft_bundle.get_batch_size<typename GetTypeReal<T>::type>();

    // Allocate batch FFT buffers (for batch recip_to_real transform)
    resmem_complex_op()(psi_mq_batch_real, batch_fft_size * wfcpw->nrxx);
    resmem_complex_op()(density_real_batch, batch_fft_size * rhopw->nrxx);
    resmem_complex_op()(density_recip_batch, batch_fft_size * rhopw->npw);
    resmem_real_op()(energy_batch, batch_fft_size);

    // Alpha buffers allocated lazily in act_op_batch (need psi.get_nbands())
    // Just allocate the batch buffer here since batch_fft_size is known
    resmem_complex_op()(alpha_batch_device, batch_fft_size);

    int nks = wfcpw->nks;
    int nk_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    resmem_real_op()(pot, rhopw->npw);
    pot_original = pot;  // Save original allocation for proper cleanup
    cached_ik = -1;  // Initialize to invalid k-point

    tpiba = ucell->tpiba;
    Real tpiba2 = tpiba * tpiba;

    // initialize rhopw_dev
    double ecut_exx = PARAM.inp.ecutexx;
    if (ecut_exx == 0.0)
    {
        ecut_exx = PARAM.inp.ecutrho;
    }

    rhopw_dev = new ModulePW::PW_Basis(wfcpw->get_device(), rhopw->get_precision());
    rhopw_dev->fft_bundle.setfft(wfcpw->get_device(), rhopw->get_precision());
#ifdef __MPI
    rhopw_dev->initmpi(rhopw->poolnproc, rhopw->poolrank, rhopw->pool_world);
#endif
    // here we can actually use different ecut to init the grids
    rhopw_dev->initgrids(rhopw->lat0, rhopw->latvec, ecut_exx);
    rhopw_dev->initgrids(rhopw->lat0, rhopw->latvec, rhopw->nx, rhopw->ny, rhopw->nz);
    rhopw_dev->initparameters(rhopw->gamma_only, ecut_exx, rhopw->distribution_type, rhopw->xprime);
    rhopw_dev->setuptransform();
    rhopw_dev->collect_local_pw();

    auto param_fock = GlobalC::exx_info.info_global.coulomb_param[Conv_Coulomb_Pot_K::Coulomb_Type::Fock];
    for (auto param: param_fock)
    {
        fock_div.push_back(exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type::Fock,
                                          0.0,
                                          kv,
                                          wfcpw,
                                          rhopw_dev,
                                          tpiba,
                                          gamma_extrapolation,
                                          ucell->omega));
    }
    auto param_erfc = GlobalC::exx_info.info_global.coulomb_param[Conv_Coulomb_Pot_K::Coulomb_Type::Erfc];
    for (auto param: param_erfc)
    {
        erfc_div.push_back(exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type::Erfc,
                                          std::stod(param["omega"]),
                                          kv,
                                          wfcpw,
                                          rhopw_dev,
                                          tpiba,
                                          gamma_extrapolation,
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

    // Free batch FFT buffers
    delmem_complex_op()(psi_mq_batch_real);
    delmem_complex_op()(density_real_batch);
    delmem_complex_op()(density_recip_batch);
    delmem_real_op()(energy_batch);

    // Free alpha value buffers (with null checks for lazy allocation)
    if (alpha_all_device != nullptr)
    {
        delmem_complex_op()(alpha_all_device);
    }
    if (alpha_batch_device != nullptr)
    {
        delmem_complex_op()(alpha_batch_device);
    }
    if (m_iband_map != nullptr)
    {
        delete[] m_iband_map;
    }

    // Clean up EXX potential cache
    clear_exx_potential_cache();

    // Free the original pot allocation
    if (enable_pot_cache && pot_original != nullptr)
    {
        delmem_real_op()(pot_original);
    }
    else if (!enable_pot_cache)
    {
        delmem_real_op()(pot);
    }

    delmem_complex_op()(h_psi_ace);
    delmem_complex_op()(psi_h_psi_ace);
    delmem_complex_op()(L_ace);
    for (auto &Xi_ace: Xi_ace_k)
    {
        delmem_complex_op()(Xi_ace);
    }
    Xi_ace_k.clear();
    delete rhopw_dev;
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

    if (PARAM.inp.exxace && GlobalC::exx_info.info_global.separate_loop)
    {
        act_op_ace(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
    }
    else
    {
        // Try batch FFT first, fallback to sequential if unavailable or disabled
        if (PARAM.inp.exx_batch_fft && wfcpw->fft_bundle.is_batch_fft_available<Real>())
        {
            act_op_batch(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
        }
        else
        {
            act_op(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
        }
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
    ModuleBase::timer::tick("OperatorEXXPW", "act_op");

    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);

    auto q_points = get_q_points(this->ik);
    // std::cout << "kpoint " << this->ik << ", qpoints: ";
    // for (auto iq: q_points)
    //     std::cout << iq << ", ";
    // std::cout << std::endl;
    int nk_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    int nk = wfcpw->nks / nk_fac;

    // ik fixed here, select band n
    for (int n_iband = 0; n_iband < nbands; n_iband++)
    {
        const T *psi_nk = tmpsi_in + n_iband * nbasis;
        // retrieve \psi_nk in real space
        wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, this->ik);

        // for \psi_nk, get the pw of iq and band m

        Real nqs = q_points.size();
        for (int iq: q_points)
        {
            ModuleBase::timer::tick("act_op", "get_exx_potential");
            Real* pot_ik_iq = get_exx_potential_cached(this->ik, iq % nk);
            pot = pot_ik_iq;  // Update member variable for multiply_potential
            ModuleBase::timer::tick("act_op", "get_exx_potential");
            for (int m_iband = 0; m_iband < psi.get_nbands(); m_iband++)
            {
                // double wg_mqb_real = GlobalC::exx_helper.wg(iq, m_iband);
                double wg_mqb_real = (*wg)(this->ik, m_iband);
                T wg_mqb = wg_mqb_real;
                if (wg_mqb_real < 1e-12)
                {
                    continue;
                }

                const T* psi_mq = get_pw(m_iband, iq);
                ModuleBase::timer::tick("act_op", "recip_to_real");
                wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                ModuleBase::timer::tick("act_op", "recip_to_real");

                // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                ModuleBase::timer::tick("act_op", "cal_density_recip");
                cal_density_recip(psi_nk_real, psi_mq_real, ucell->omega);
                ModuleBase::timer::tick("act_op", "cal_density_recip");

                // multiply the density with the potential in recip space
                ModuleBase::timer::tick("act_op", "multiply_potential");
                multiply_potential(density_recip, this->ik, iq);
                ModuleBase::timer::tick("act_op", "multiply_potential");

                // bring the potential back to real space
                ModuleBase::timer::tick("act_op", "multiply_potential");
                rho_recip2real(density_recip, density_real);
                ModuleBase::timer::tick("act_op", "multiply_potential");

                if (false)
                {
                    // do nothing
                }
                else
                {
                    ModuleBase::timer::tick("act_op", "vec_mul_vec_complex_op");
                    vec_mul_vec_complex_op<T, Device>()(density_real, psi_mq_real, density_real, wfcpw->nrxx);
                    ModuleBase::timer::tick("act_op", "vec_mul_vec_complex_op");
                }

                T wk_iq = kv->wk[iq];

                T tmp_scalar = wg_mqb / wk_iq / nqs;
                ModuleBase::timer::tick("act_op", "axpy_complex_op");
                axpy_complex_op()(wfcpw->nrxx,
                                  &tmp_scalar,
                                  density_real,
                                  1,
                                  h_psi_real,
                                  1);
                ModuleBase::timer::tick("act_op", "axpy_complex_op");

            } // end of m_iband
            setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw_dev->npw);
            setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);

        } // end of iq
        T* h_psi_nk = tmhpsi + n_iband * nbasis;
        Real hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
        wfcpw->real_to_recip(ctx, h_psi_real, h_psi_nk, this->ik, true, hybrid_alpha);
        setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);

    }

    ModuleBase::timer::tick("OperatorEXXPW", "act_op");

}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op_batch(const int nbands,
                                            const int nbasis,
                                            const int npol,
                                            const T *tmpsi_in,
                                            T *tmhpsi,
                                            const int ngk_ik,
                                            const bool is_first_node) const
{
    nvtxMark("Entering act_op_batch");
    ModuleBase::timer::tick("OperatorEXXPW", "act_op_batch");

    // Initialize buffers (same as act_op)
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);

    const int batch_fft_size = this->get_batch_fft_size();
    setmem_complex_op()(psi_mq_batch_real, 0, batch_fft_size * wfcpw->nrxx);
    setmem_complex_op()(density_real_batch, 0, batch_fft_size * rhopw_dev->nrxx);
    setmem_complex_op()(density_recip_batch, 0, batch_fft_size * rhopw_dev->npw);
    setmem_real_op()(energy_batch, 0, batch_fft_size);

    auto q_points = get_q_points(this->ik);
    int nk_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    int nk = wfcpw->nks / nk_fac;

    // Precompute the weighting factors used for axpy call
    int nqb = q_points.size() * psi.get_nbands();
    std::vector<T> alpha_all_host(nqb); // Temporary host buffer for all alpha values
    int nqs = q_points.size();
    int local_band_index = 0;  // counter for the valid bands (non-negligible weight)
    int local_q_idx = 0;
    const T one{1, 0}; // Scalar one for gemv
    const int inc{1}; // Increment for gemv 

    // Precompute the weighting factor array used for inner loop so 
    // blas GEMV can be used for fast operation
    for (int iband = 0; iband < nbands; iband++)
    {                
        local_q_idx = 0;
        double wg_mqb_real = (*wg)(this->ik, iband);
        if (wg_mqb_real < 1e-12)
        {
                continue;  // Skip negligible weights
        }
        double wg_m = (*wg)(this->ik, iband);
        for (int iq: q_points)
        {
            T wk_iq = kv->wk[iq];
            alpha_all_host[local_q_idx * nbands + local_band_index] = static_cast<T>(wg_m) / wk_iq / static_cast<T>(nqs);
            local_q_idx++;
        }
        local_band_index++; 
    }

    // Copy to device ONCE per n_iband iteration (only for GPU)
    alpha_all_device = nullptr;
    resmem_complex_op()(alpha_all_device, nqb);
    syncmem_complex_c2d_op()(alpha_all_device, alpha_all_host.data(), nqb);

    // Outer loop over bands (same structure as act_op)
    for (int n_iband = 0; n_iband < nbands; n_iband++)
    {
        local_q_idx = 0;
        const T *psi_nk = tmpsi_in + n_iband * nbasis;
        wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, this->ik);

        Real nqs = q_points.size();

        // Now loop over q-points
        for (int iq: q_points)
        {
            ModuleBase::timer::tick("act_op_batch", "get_exx_potential");
            Real* pot_ik_iq = get_exx_potential_cached(this->ik, iq % nk);
            pot = pot_ik_iq;
            ModuleBase::timer::tick("act_op_batch", "get_exx_potential");

            // === BATCH FFT SECTION ===
            // Accumulate m_iband iterations into batches
            int batch_idx = 0;

            // Heap-allocated arrays (std::vector provides dynamic sizing)
            std::vector<const T*> psi_mq_ptrs(batch_fft_size);      // Pointers to input wavefunctions
            std::vector<int> ik_batch(batch_fft_size);              // k-point indices for batch
            std::vector<T> wg_batch(batch_fft_size);                // Weights for each element (converted to T type)
            std::vector<int> batch_local_band_idx(batch_fft_size);  // Track which valid indices are in this batch
            std::vector<int> batch_actual_band_idx(batch_fft_size); // Track actual band indices for alpha retrieval
            local_band_index =0; // reset valid band counter
            for (int m_iband = 0; m_iband < psi.get_nbands(); m_iband++)
            {
                double wg_mqb_real = (*wg)(this->ik, m_iband);
                if (wg_mqb_real < 1e-12)
                {
                    continue;  // Skip negligible weights
                }

                // Add to batch
                psi_mq_ptrs[batch_idx] = get_pw(m_iband, iq);
                ik_batch[batch_idx] = iq;
                wg_batch[batch_idx] = wg_mqb_real;  // Implicit conversion to T
                batch_local_band_idx[batch_idx] = local_band_index;  // Track valid index
                batch_actual_band_idx[batch_idx] = m_iband;  // Track valid index
                batch_idx++;

                // Flush batch when full OR at last m_iband
                if (batch_idx == batch_fft_size || m_iband == psi.get_nbands() - 1)
                {
                    nvtxRangePush("Flushing batch");
                    // Copy batch inputs to contiguous buffer
                    ModuleBase::timer::tick("act_op_batch", "prepare_batch");
                    if (consecutive_integers(batch_actual_band_idx.data(), batch_idx) && (psi.get_k_first()))
                    {
                       ModuleBase::timer::tick("act_op_batch", "recip_to_real_batch direct");
                        wfcpw->recip_to_real_batch<Real, Device>(
                            ctx,
                            psi_mq_ptrs[0],              // Input: Pointer to the first wavefunction in the batch
                            psi_mq_batch_real,           // Output: batch_idx × nrxx (reuse buffer)
                            ik_batch.data(),
                            batch_idx,
                            false,
                            Real(1.0));
                       ModuleBase::timer::tick("act_op_batch", "recip_to_real_batch direct");
                    }
                    else
                    {
                        for (int ib = 0; ib < batch_idx; ib++)
                        {
                            // Use the batch buffer for in-place FFT
                            syncmem_complex_op()(
                                psi_mq_batch_real + ib * wfcpw->npwk_max,
                                psi_mq_ptrs[ib],
                                wfcpw->npwk_max);
                        }
                        ModuleBase::timer::tick("act_op_batch", "prepare_batch");

                        // Batch recip_to_real transform
                        ModuleBase::timer::tick("act_op_batch", "recip_to_real_batch");
                        wfcpw->recip_to_real_batch<Real, Device>(
                            ctx,
                            psi_mq_batch_real,           // Input: batch_idx × npwk_max
                            psi_mq_batch_real,           // Output: batch_idx × nrxx (reuse buffer)
                            ik_batch.data(),
                            batch_idx,
                            false,
                            Real(1.0));
                        ModuleBase::timer::tick("act_op_batch", "recip_to_real_batch");
                    }

                    // === STAGE 1: Batch density calculation (element-wise + real2recip FFT) ===
                    ModuleBase::timer::tick("act_op_batch", "cal_density_recip_batch");
                    cal_density_recip_batch(
                        psi_nk_real,           // psi_nk (same for all batch elements)
                        psi_mq_batch_real,     // batch of psi_mq (batch_idx × nrxx)
                        density_real_batch,    // output batch (batch_idx × nrxx)
                        density_recip_batch,   // output batch (batch_idx × npw)
                        batch_idx,
                        ik_batch.data(),
                        ucell->omega);
                    ModuleBase::timer::tick("act_op_batch", "cal_density_recip_batch");

                    // === STAGE 2: Batch multiply with potential ===
                    ModuleBase::timer::tick("act_op_batch", "multiply_potential_batch");

                    // Call batched operator (works for both CPU and GPU)
                    // - GPU: Single batched kernel launch for efficiency
                    // - CPU: Sequential processing with OpenMP per element
                    mul_potential_op<T, Device>().operator_batch(
                        pot,                     // Same potential for all batch elements
                        density_recip_batch,     // Batch input/output (batch_idx × npw)
                        rhopw_dev->npw,
                        batch_idx);

                    ModuleBase::timer::tick("act_op_batch", "multiply_potential_batch");

                    // === STAGE 3: Batch recip2real FFT for densities ===
                    ModuleBase::timer::tick("act_op_batch", "recip_to_real_batch_density");

                    // Now we can call batch method directly (works for both CPU and GPU)
                    rhopw_dev->recip_to_real_batch<Real, Device>(
                        this->ctx,
                        density_recip_batch,     // Input: batch_idx × npw
                        density_real_batch,      // Output: batch_idx × nrxx
                        batch_idx,
                        false,
                        Real(1.0));

                    ModuleBase::timer::tick("act_op_batch", "recip_to_real_batch_density");

                    // === STAGE 4: Batch element-wise multiply ===
                    ModuleBase::timer::tick("act_op_batch", "vec_mul_vec_batch");

                    // Call batched operator (works for both CPU and GPU)
                    // - GPU: Single batched kernel launch
                    // - CPU: Sequential loop with OpenMP per element
                    vec_mul_vec_complex_op<T, Device>().operator_batch(
                        density_real_batch,      // Input batch 1 (batch_idx × nrxx)
                        psi_mq_batch_real,       // Input batch 2 (batch_idx × nrxx)
                        density_real_batch,      // Output batch (batch_idx × nrxx, in-place)
                        wfcpw->nrxx,
                        batch_idx);

                    ModuleBase::timer::tick("act_op_batch", "vec_mul_vec_batch");

                    // === STAGE 6: Batched accumulation with precomputed alpha values ===
                    // The axpy is replaced with GEMV call as new we are doing matrix vector product
                    // alpha_all_device is a vector, density_real_batch is a matrix, and we accumulate into 
                    // a single vector h_psi_real
                    // std::cout << "Before gemv accumulation, batch size: " << batch_idx << std::endl;
                    // std::cout << local_q_idx << " " << nbands << " " << batch_local_band_idx[0] << std::endl;
                    ModuleBase::timer::tick("act_op_batch", "accumulate");
                    ModuleBase::gemv_op<T, Device>()(
                        'N',
                        wfcpw->nrxx,          // m
                        batch_idx,           // n
                        &one,            // alpha
                        density_real_batch,   // matrix (batch_idx, nrxx), memory layout as (nrxx, batch_idx) in column major
                        wfcpw->nrxx,         // lda
                        alpha_all_device + (local_q_idx * nbands + batch_local_band_idx[0]),  // vector (batch_idx, )
                        inc,                   // incx
                        &one,            // beta
                        h_psi_real,          // output vector (nrxx)
                        inc);                  // incy
                    ModuleBase::timer::tick("act_op_batch", "accumulate");

                    // Reset batch
                    batch_idx = 0;
                    nvtxRangePop();
                }
                local_band_index++;  // Increment valid band counter
            } // end m_iband loop

            // Clear buffers for next iq
            setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw_dev->npw);
            local_q_idx++;
        } // end iq loop

        // Transform h_psi back to reciprocal space
        T* h_psi_nk = tmhpsi + n_iband * nbasis;
        Real hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
        wfcpw->real_to_recip(ctx, h_psi_real, h_psi_nk, this->ik, true, hybrid_alpha);
        setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);

    } // end n_iband loop

    ModuleBase::timer::tick("OperatorEXXPW", "act_op_batch");
    nvtxMark("Exiting act_op_batch");
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op_kpar(const int nbands,
                                   const int nbasis,
                                   const int npol,
                                   const T *tmpsi_in,
                                   T *tmhpsi,
                                   const int ngk_ik,
                                   const bool is_first_node) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "act_op_kpar");

    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);
    // setmem_complex_op()(psi_all_real, 0, wfcpw->nrxx * GlobalV::NBANDS);
    // std::map<std::pair<int, int>, bool> has_real;
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);
    int nqs = kv->get_nkstot_full();
    int nspin_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    int ispin = this->ik < (wfcpw->nks / nspin_fac) ? 0 : 1;

    // ik fixed here, select band n
    for (int iq = 0; iq < nqs; iq++)
    {
        // for \psi_nk, get the pw of iq and band m
        Real* pot_ik_iq = get_exx_potential_cached(this->ik, iq);
        pot = pot_ik_iq;  // Update member variable for multiply_potential

        // decide which pool does the iq belong to
        int iq_pool = kv->para_k.whichpool[iq];
        int iq_loc  = iq - kv->para_k.startk_pool[iq_pool];
        if (ispin == 1)
        {
            iq_loc += wfcpw->nks / nspin_fac;
        }

        for (int m_iband = 0; m_iband < psi.get_nbands(); m_iband++)
        {
            double wg_mqb = 0;
            if (iq_pool == GlobalV::MY_POOL)
            {
                wg_mqb = (*wg)(iq_loc, m_iband);
            }
#ifdef __MPI
            MPI_Bcast(&wg_mqb, 1, MPI_DOUBLE, kv->para_k.get_startpro_pool(iq_pool), MPI_COMM_WORLD);
#endif
            if (wg_mqb < 1e-12)
                continue;

            if (iq_pool == GlobalV::MY_POOL)
            {
                const T* psi_mq = get_pw(m_iband, iq_loc + ispin * wfcpw->nks / nspin_fac);
                wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq_loc);
                // send
            }
#ifdef __MPI
#ifdef __CUDA_MPI
            MPI_Bcast(psi_mq_real, wfcpw->nrxx, MPI_DOUBLE_COMPLEX, iq_pool, KP_WORLD);
#else
            if (PARAM.inp.device == "cpu")
            {
                MPI_Bcast(psi_mq_real, wfcpw->nrxx, MPI_DOUBLE_COMPLEX, iq_pool, KP_WORLD);
            }
            else if (PARAM.inp.device == "gpu")
            {
                // need to copy to cpu first
                T* psi_mq_real_cpu = new T[wfcpw->nrxx];
                syncmem_complex_d2c_op()(psi_mq_real_cpu, psi_mq_real, wfcpw->nrxx);
                MPI_Bcast(psi_mq_real_cpu, wfcpw->nrxx, MPI_DOUBLE_COMPLEX, iq_pool, KP_WORLD);
                syncmem_complex_c2d_op()(psi_mq_real, psi_mq_real_cpu, wfcpw->nrxx);
                delete[] psi_mq_real_cpu;
            }
            else
            {
                ModuleBase::WARNING_QUIT("OperatorEXXPW", "construct_ace: unknown device");
            }
#endif
#endif
            for (int n_iband = 0; n_iband < nbands; n_iband++)
            {
                const T* psi_nk = tmpsi_in + n_iband * nbasis;
                // retrieve \psi_nk in real space
                wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, this->ik);


                // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                cal_density_recip(psi_nk_real, psi_mq_real, ucell->omega);

                mul_potential_op<T, Device>()(pot, density_recip, rhopw_dev->npw, wfcpw->nks, this->ik, iq);

                // bring the potential back to real space
                rho_recip2real(density_recip, density_real);

                if (false)
                {
                    // do nothing
                }
                else
                {
                    vec_mul_vec_complex_op<T, Device>()(density_real, psi_mq_real, density_real, wfcpw->nrxx);
                }


                Real wk_iq = kv->wk[iq];
                Real wk_ik = kv->wk[this->ik];

                Real tmp_scalar = wg_mqb / wk_ik / nqs; // wk_ik works for now, but wrong for symmetry.

                T* h_psi_nk = tmhpsi + n_iband * nbasis;
                Real hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
                wfcpw->real_to_recip(ctx, density_real, h_psi_nk, this->ik, true, hybrid_alpha * tmp_scalar);


            } // end of m_iband
            setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw_dev->npw);
            setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);

        } // end of iq

    }

    ModuleBase::timer::tick("OperatorEXXPW", "act_op_kpar");

}

template <typename T, typename Device>
std::vector<int> OperatorEXXPW<T, Device>::get_q_points(const int ik) const
{
    // stored in q_points
    if (q_points.find(ik) != q_points.end())
    {
        return q_points.find(ik)->second;
    }

    std::vector<int> q_points_ik;

    // if () // downsampling
    {
        for (int iq = 0; iq < wfcpw->nks; iq++)
        {
            if (PARAM.inp.nspin ==1 )
            {
                q_points_ik.push_back(iq);
            }
            else if (PARAM.inp.nspin == 2)
            {
                int nk_fac = 2;
                int nk = wfcpw->nks / nk_fac;
                if (iq / nk == ik / nk)
                {
                    q_points_ik.push_back(iq);
                }
            }
            else
            {
                ModuleBase::WARNING_QUIT("OperatorEXXPW", "nspin == 4 not supported");
            }
        }
    }
    // else
    // {
    //     for (int iq = 0; iq < wfcpw->nks; iq++)
    //     {
    //         kv->
    //     }
    // }

    q_points[ik] = q_points_ik;
    return q_points_ik;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::multiply_potential(T *density_recip, int ik, int iq) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "multiply_potential");
    int npw = rhopw_dev->npw;
    int nks = wfcpw->nks;
    int nk_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    int nk = nks / nk_fac;

    mul_potential_op<T, Device>()(pot, density_recip, npw, nks, ik, iq);

    ModuleBase::timer::tick("OperatorEXXPW", "multiply_potential");
}

template <typename T, typename Device>
typename GetTypeReal<T>::type* OperatorEXXPW<T, Device>::get_exx_potential_cached(int ik, int iq) const
{
    using Real = typename GetTypeReal<T>::type;

    // Cache disabled - fallback to original behavior
    if (!enable_pot_cache)
    {
        get_exx_potential<Real, Device>(kv, wfcpw, rhopw_dev, pot, tpiba,
                                       gamma_extrapolation, ucell->omega, ik, iq);
        return pot;
    }

    // Check if k-point changed - invalidate cache if needed
    if (cached_ik != ik)
    {
        // K-point changed - clear old cache
        for (auto& entry : pot_cache)
        {
            delmem_real_op()(entry.second);
        }
        pot_cache.clear();
        cached_ik = ik;  // Update cached k-point
    }

    // Check cache for current k-point
    auto it = pot_cache.find(iq);

    if (it != pot_cache.end())
    {
        // Cache hit - return existing potential for this iq
        return it->second;
    }

    // Cache miss - allocate and compute
    Real* pot_new = nullptr;
    resmem_real_op()(pot_new, rhopw_dev->npw);

    get_exx_potential<Real, Device>(kv, wfcpw, rhopw_dev, pot_new, tpiba,
                                   gamma_extrapolation, ucell->omega, ik, iq);

    // Store in cache (for current k-point)
    pot_cache[iq] = pot_new;

    return pot_new;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::clear_exx_potential_cache()
{
    using Real = typename GetTypeReal<T>::type;
    for (auto& kv : pot_cache)
    {
        delmem_real_op()(kv.second);
    }
    pot_cache.clear();
}

template <typename T, typename Device>
const T *OperatorEXXPW<T, Device>::get_pw(const int m, const int iq) const
{
    // return pws[iq].get() + m * wfcpw->npwk[iq];
    psi.fix_kb(iq, m);
    T* psi_mq = psi.get_pointer();
    return psi_mq;
}

template <typename T, typename Device>
template <typename T_in, typename Device_in>
OperatorEXXPW<T, Device>::OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op)
{
    // copy all the datas
    this->isk = op->isk;
    this->wfcpw = op->wfcpw;
    this->rhopw = op->rhopw;
    this->rhopw_dev = op->rhopw_dev;
    this->psi = op->psi;
    this->ctx = op->ctx;
    this->cpu_ctx = op->cpu_ctx;
    resmem_complex_op()(this->ctx, psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, density_real, rhopw_dev->nrxx);
    resmem_complex_op()(this->ctx, h_psi_real, rhopw_dev->nrxx);
    resmem_complex_op()(this->ctx, density_recip, rhopw_dev->npw);
    resmem_complex_op()(this->ctx, h_psi_recip, wfcpw->npwk_max);
//    this->pws.resize(wfcpw->nks);


}

template <typename T, typename Device>
double OperatorEXXPW<T, Device>::cal_exx_energy(psi::Psi<T, Device> *psi_) const
{
    if (PARAM.inp.exxace && GlobalC::exx_info.info_global.separate_loop)
    {
        return cal_exx_energy_ace(psi_);
    }
    else
    {
        // Try batch FFT version if both enabled in INPUT and available at compile time
        if (PARAM.inp.exx_batch_fft && wfcpw->fft_bundle.is_batch_fft_available<Real>())
        {
            return cal_exx_energy_batch(psi_);
        }
        else
        {
            return cal_exx_energy_op(psi_);
        }
    }
}

template <typename T, typename Device>
double OperatorEXXPW<T, Device>::cal_exx_energy_op(psi::Psi<T, Device> *ppsi_) const
{
    nvtxRangePush("cal exx_energy_op");
    psi::Psi<T, Device> psi_ = *ppsi_;

    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    if (wg == nullptr) return 0.0;
    const int nk_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    double Eexx_ik_real = 0.0;
    for (int ik = 0; ik < wfcpw->nks; ik++)
    {
        //        auto k = this->pw_wfc->kvec_c[ik];
        //        std::cout << k << std::endl;
        for (int n_iband = 0; n_iband < psi.get_nbands(); n_iband++)
        {
            setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
            setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

            // double wg_ikb_real = GlobalC::exx_helper.wg(this->ik, n_iband);
            double wg_ikb_real = (*wg)(ik, n_iband);
            T wg_ikb = wg_ikb_real;
            if (wg_ikb_real < 1e-12)
            {
                continue;
            }

            // const T *psi_nk = get_pw(n_iband, ik);
            psi.fix_kb(ik, n_iband);
            const T* psi_nk = psi.get_pointer();
            // retrieve \psi_nk in real space
            wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, ik);

            // for \psi_nk, get the pw of iq and band m
            // q_points is a vector of integers, 0 to nks-1
            std::vector<int> q_points;
            if (PARAM.inp.nspin == 1)
            {
                for (int iq = 0; iq < wfcpw->nks; iq++)
                {
                    q_points.push_back(iq);
                }
            }
            else if (PARAM.inp.nspin == 2)
            {
                int nk = wfcpw->nks / nk_fac;
                int k_spin = ik / nk;
                for (int iq = 0; iq < wfcpw->nks; iq++)
                {
                    int q_spin = iq / nk;
                    if (k_spin == q_spin)
                    {
                        q_points.push_back(iq);
                    }
                }
            }
            else
            {
                ModuleBase::WARNING_QUIT("OperatorEXXPW", "nspin == 4 not supported");
            }
            double nqs = q_points.size();

            for (int iq: q_points)
            {
                int nk = wfcpw->nks / nk_fac;
                Real* pot_ik_iq = get_exx_potential_cached(ik, iq % nk);
                pot = pot_ik_iq;  // Update member variable for multiply_potential
                for (int m_iband = 0; m_iband < psi.get_nbands(); m_iband++)
                {
                    // double wg_f = GlobalC::exx_helper.wg(iq, m_iband);
                    double wg_iqb_real = (*wg)(iq, m_iband);
                    T wg_iqb = wg_iqb_real;
                    if (wg_iqb_real < 1e-12)
                    {
                        continue;
                    }

                    psi_.fix_kb(iq, m_iband);
                    const T* psi_mq = psi_.get_pointer();
                    // const T* psi_mq = get_pw(m_iband, iq);
                    wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);

                    cal_density_recip(psi_nk_real, psi_mq_real, ucell->omega);

                    int nks = wfcpw->nks;
                    int npw = rhopw_dev->npw;
                    // int nk = nks / nk_fac;
                    Eexx_ik_real += exx_cal_energy_op<T, Device>()(density_recip, pot, wg_iqb_real / nqs * wg_ikb_real / kv->wk[ik], npw);

                } // m_iband

            } // iq

        } // n_iband

    } // ik
    Eexx_ik_real *= 0.5 * ucell->omega;
    Parallel_Reduce::reduce_pool(Eexx_ik_real);
    //    std::cout << "omega = " << this_->pelec->omega << " tpiba = " << this_->pw_rho->tpiba2 << " exx_div = " << exx_div << std::endl;

    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    nvtxRangePop();
    return Eexx_ik_real;
}

template <typename T, typename Device>
double OperatorEXXPW<T, Device>::cal_exx_energy_batch(psi::Psi<T, Device> *ppsi_) const
{
    nvtxRangePush("cal_exx_energy_batch");
    ModuleBase::timer::tick("OperatorEXXPW", "cal_exx_energy_batch");

    psi::Psi<T, Device> psi_ = *ppsi_;
    int npw = rhopw_dev->npw;

    // === INITIALIZATION (reuse from cal_exx_energy_op) ===
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    // === BATCH BUFFERS (reuse from act_op_batch) ===
    const int batch_fft_size = this->get_batch_fft_size();
    setmem_complex_op()(psi_mq_batch_real, 0, batch_fft_size * wfcpw->nrxx);
    setmem_complex_op()(density_real_batch, 0, batch_fft_size * rhopw_dev->nrxx);
    setmem_complex_op()(density_recip_batch, 0, batch_fft_size * rhopw_dev->npw);

    if (wg == nullptr) return 0.0;
    const int nk_fac = PARAM.inp.nspin == 2 ? 2 : 1;
    double Eexx_ik_real = 0.0;
    assert(npw < rhopw_dev.nrxx *2 && "realspaced grid too small for reusing density_real_batch as buffer"); // make sure potential buffer is large enough

    // === OUTER LOOPS: ik, n_iband (same as cal_exx_energy_op) ===
    for (int ik = 0; ik < wfcpw->nks; ik++)
        {

        // For precomputing alpha values for accumulation
        auto q_points = get_q_points(ik);
        int nbands = psi_.get_nbands();
        int nqs = q_points.size();
        int nqb = nqs * psi_.get_nbands();
        int local_band_index = 0;  // counter for the valid bands (non-negligible weight)
        int local_q_idx = 0;
        std::vector<Real> weight_real_host(nqb); // Temporary host buffer for all weighting factors
        Real* weight_real_device = nullptr;
        resmem_real_op()(weight_real_device, nqb);

        for (int n_iband = 0; n_iband < psi.get_nbands(); n_iband++)
        {
            // Reset buffers (from cal_exx_energy_op lines 938-941)
            setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
            setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

            double wg_ikb_real = (*wg)(ik, n_iband);
            if (wg_ikb_real < 1e-12) continue;  // Skip negligible weights

            // Precompute the weighting factor array used for inner iq, m_iband loop so 
            // blas GEMV can be used for fast operation
            local_q_idx = 0;
            for (int iq: q_points)
            {                
                local_band_index = 0;
                for (int m_iband = 0; m_iband < psi_.get_nbands(); m_iband++)
                {
                    double wg_mqb_real = (*wg)(iq, m_iband);
                    if (wg_mqb_real < 1e-12) continue;  // Skip negligible
                    weight_real_host[local_q_idx * nbands + local_band_index] = wg_mqb_real / nqs * wg_ikb_real / kv->wk[iq];
                    local_band_index++;
                }
                local_q_idx++; 
            }
            // Copy to device
            syncmem_real_c2d_op()(weight_real_device, weight_real_host.data(), nqb);


            psi.fix_kb(ik, n_iband);
            const T* psi_nk = psi.get_pointer();
            wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, ik);

            // === Q-POINT LOOP ===
            local_q_idx = 0;
            for (int iq: q_points)
            {
                int nk = wfcpw->nks / nk_fac;
                Real* pot_ik_iq = get_exx_potential_cached(ik, iq % nk);
                pot = pot_ik_iq;

                // === BATCHING SECTION: m_iband loop ===
                int batch_idx = 0;
                std::vector<const T*> psi_mq_ptrs(batch_fft_size);
                std::vector<int> ik_batch(batch_fft_size);
                std::vector<int> batch_actual_band_idx(batch_fft_size);
                std::vector<int> batch_local_band_idx(batch_fft_size);

                local_band_index = 0;  // counter for the valid bands (non-negligible weight)
                for (int m_iband = 0; m_iband < psi_.get_nbands(); m_iband++)
                {
                    double wg_iqb_real = (*wg)(iq, m_iband);
                    if (wg_iqb_real < 1e-12) continue;  // Skip negligible

                    // Accumulate into batch
                    psi_.fix_kb(iq, m_iband);
                    psi_mq_ptrs[batch_idx] = psi_.get_pointer();
                    ik_batch[batch_idx] = iq;
                    batch_actual_band_idx[batch_idx] = m_iband;
                    batch_local_band_idx[batch_idx] = local_band_index;
                    batch_idx++;

                    // Flush when batch full OR last iteration
                    if (batch_idx == batch_fft_size || m_iband == psi_.get_nbands() - 1)
                    {

                        nvtxRangePush("Flushing batch calc_exx_energy");
                        ModuleBase::timer::tick("cal_exx_energy_batch", "process_batch");

                        // === STAGE 1: Batch FFT (recip_to_real for psi_mq) ===
                        // Check if bands are consecutive (optimization from act_op_batch)
                        if (consecutive_integers(batch_actual_band_idx.data(), batch_idx) && psi_.get_k_first())
                        {
                            // Direct batch transform (no copy needed)
                            wfcpw->recip_to_real_batch<Real, Device>(
                                ctx, psi_mq_ptrs[0], psi_mq_batch_real,
                                ik_batch.data(), batch_idx, false, Real(1.0));
                        }
                        else
                        {
                            // Copy to batch buffer first
                            using syncmem_complex_op = base_device::memory::synchronize_memory_op<T, Device, Device>;
                            for (int ib = 0; ib < batch_idx; ib++) {
                                syncmem_complex_op()(psi_mq_batch_real + ib * wfcpw->npwk_max,
                                                     psi_mq_ptrs[ib], wfcpw->npwk_max);
                            }
                            wfcpw->recip_to_real_batch<Real, Device>(
                                ctx, psi_mq_batch_real, psi_mq_batch_real,
                                ik_batch.data(), batch_idx, false, Real(1.0));
                        }

                        // === STAGE 2: Batch density calculation ===
                        cal_density_recip_batch(
                            psi_nk_real, psi_mq_batch_real,
                            density_real_batch, density_recip_batch,
                            batch_idx, ik_batch.data(), ucell->omega);

                        // === STAGE 3: Energy reduction (sequential per batch element) ===
                        // NOTE: exx_cal_energy_op applies potential internally, so no need to multiply here
                        ModuleBase::timer::tick("cal_exx_energy_batch", "energy_reduction");
                        // baseline squential operation
                        // for (int ib = 0; ib < batch_idx; ib++)
                        // {
                        //     T* density_ib = density_recip_batch + ib * rhopw_dev->npw;
                        //     double E_ib = exx_cal_energy_op<T, Device>()(
                        //         density_ib, pot, scalar_batch[ib], rhopw_dev->npw);
                        //     Eexx_ik_real += E_ib;
                        // }
                        // Compute the squared norms for the density, use the 
                        // TODO this is not working ! We need to pre-compute scalar-batch outside
                        // the loop and pass it to the kernel as the final energy needs to be weighted!!
                        Eexx_ik_real += static_cast<double>(
                            exx_vector_elementwise_norm_squared_op<T, Device>()(
                            density_recip_batch,
                            reinterpret_cast<Real*>(density_real_batch),
                            pot,
                            energy_batch, 
                            weight_real_device + (local_q_idx * nbands + batch_local_band_idx[0]),
                            npw,
                            batch_idx)
                        );

                        ModuleBase::timer::tick("cal_exx_energy_batch", "energy_reduction");

                        ModuleBase::timer::tick("cal_exx_energy_batch", "process_batch");

                        // Reset batch
                        batch_idx = 0;
                        nvtxRangePop();
                    }
                    local_band_index++;  // Increment valid band counter
                } // m_iband loop
            local_q_idx++;
            } // iq loop
        } // n_iband loop
    } // ik loop

    // === FINALIZATION (same as cal_exx_energy_op) ===
    Eexx_ik_real *= 0.5 * ucell->omega;
    Parallel_Reduce::reduce_pool(Eexx_ik_real);

    // Cleanup buffers
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);
    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_real, 0, rhopw_dev->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw_dev->npw);

    ModuleBase::timer::tick("OperatorEXXPW", "cal_exx_energy_batch");
    nvtxRangePop();
    return Eexx_ik_real;
}

template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::cal_density_recip(const std::complex<double>* psi_nk_real,
                                                                                const std::complex<double>* psi_mq_real,
                                                                                double omega) const
{
    cal_density_real_op<std::complex<double>, base_device::DEVICE_CPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw->nrxx);
    rhopw_dev->real2recip(density_real, density_recip);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::cal_density_recip(const std::complex<float>* psi_nk_real,
                                                                                const std::complex<float>* psi_mq_real,
                                                                                double omega) const
{
    cal_density_real_op<std::complex<float>, base_device::DEVICE_CPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw->nrxx);
    rhopw_dev->real2recip(density_real, density_recip);
}

// Batch density calculation (CPU double precision)
template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::cal_density_recip_batch(
    const std::complex<double>* psi_nk_real,
    std::complex<double>* psi_mq_real_batch,
    std::complex<double>* density_real_batch,
    std::complex<double>* density_recip_batch,
    int batch_size,
    const int* ik_batch,
    double omega) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");

    // Process each batch element sequentially on CPU
    for (int ib = 0; ib < batch_size; ib++)
    {
        std::complex<double>* psi_mq_ib = psi_mq_real_batch + ib * wfcpw->nrxx;
        std::complex<double>* density_real_ib = density_real_batch + ib * rhopw_dev->nrxx;
        std::complex<double>* density_recip_ib = density_recip_batch + ib * rhopw_dev->npw;

        // Element-wise multiply: density = psi_nk * conj(psi_mq) / omega
        cal_density_real_op<std::complex<double>, base_device::DEVICE_CPU>()(
            psi_nk_real, psi_mq_ib, density_real_ib, omega, wfcpw->nrxx);

        // FFT: real -> recip
        rhopw_dev->real2recip(density_real_ib, density_recip_ib);
    }

    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");
}

// Batch density calculation (CPU single precision)
template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::cal_density_recip_batch(
    const std::complex<float>* psi_nk_real,
    std::complex<float>* psi_mq_real_batch,
    std::complex<float>* density_real_batch,
    std::complex<float>* density_recip_batch,
    int batch_size,
    const int* ik_batch,
    double omega) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");

    // Process each batch element sequentially on CPU
    for (int ib = 0; ib < batch_size; ib++)
    {
        std::complex<float>* psi_mq_ib = psi_mq_real_batch + ib * wfcpw->nrxx;
        std::complex<float>* density_real_ib = density_real_batch + ib * rhopw_dev->nrxx;
        std::complex<float>* density_recip_ib = density_recip_batch + ib * rhopw_dev->npw;

        // Element-wise multiply: density = psi_nk * conj(psi_mq) / omega
        cal_density_real_op<std::complex<float>, base_device::DEVICE_CPU>()(
            psi_nk_real, psi_mq_ib, density_real_ib, omega, wfcpw->nrxx);

        // FFT: real -> recip
        rhopw_dev->real2recip(density_real_ib, density_recip_ib);
    }

    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");
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
    cal_density_real_op<std::complex<double>, base_device::DEVICE_GPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw->nrxx);
    rhopw_dev->real2recip_gpu(density_real, density_recip);
}

template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::cal_density_recip(const std::complex<float>* psi_nk_real,
                                                                                const std::complex<float>* psi_mq_real,
                                                                                double omega) const
{
    cal_density_real_op<std::complex<float>, base_device::DEVICE_GPU>()(psi_nk_real, psi_mq_real, density_real, omega, wfcpw->nrxx);
    rhopw_dev->real2recip_gpu(density_real, density_recip);
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

// Batch density calculation (GPU double precision)
// For now, use sequential transforms for real->recip (focus optimization on recip->real bottleneck)
template <>
void OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>::cal_density_recip_batch(
    const std::complex<double>* psi_nk_real,
    std::complex<double>* psi_mq_real_batch,
    std::complex<double>* density_real_batch,
    std::complex<double>* density_recip_batch,
    int batch_size,
    const int* ik_batch,
    double omega) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");

    // Step 1: Batched element-wise multiply
    ModuleBase::timer::tick("cal_density_recip_batch", "element_wise_batch");
    cal_density_real_op<std::complex<double>, base_device::DEVICE_GPU>().operator_batch(
        psi_nk_real,           // Constant (nrxx)
        psi_mq_real_batch,     // Batch input (batch_size × nrxx)
        density_real_batch,    // Batch output (batch_size × nrxx)
        omega,
        wfcpw->nrxx,
        batch_size);
    ModuleBase::timer::tick("cal_density_recip_batch", "element_wise_batch");

    // Step 2: Batch FFT real → recip
    ModuleBase::timer::tick("cal_density_recip_batch", "real2recip_batch");
    rhopw_dev->real_to_recip_batch<double, base_device::DEVICE_GPU>(
        this->ctx,
        density_real_batch,      // Input: batch_size × nrxx
        density_recip_batch,     // Output: batch_size × npw
        batch_size,
        false,
        1.0);
    ModuleBase::timer::tick("cal_density_recip_batch", "real2recip_batch");

    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");
}

// Batch density calculation (GPU single precision)
template <>
void OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::cal_density_recip_batch(
    const std::complex<float>* psi_nk_real,
    std::complex<float>* psi_mq_real_batch,
    std::complex<float>* density_real_batch,
    std::complex<float>* density_recip_batch,
    int batch_size,
    const int* ik_batch,
    double omega) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");

    // Step 1: Batched element-wise multiply
    ModuleBase::timer::tick("cal_density_recip_batch", "element_wise_batch");
    cal_density_real_op<std::complex<float>, base_device::DEVICE_GPU>().operator_batch(
        psi_nk_real,           // Constant (nrxx)
        psi_mq_real_batch,     // Batch input (batch_size × nrxx)
        density_real_batch,    // Batch output (batch_size × nrxx)
        omega,
        wfcpw->nrxx,
        batch_size);
    ModuleBase::timer::tick("cal_density_recip_batch", "element_wise_batch");

    // Step 2: Batch FFT real → recip
    ModuleBase::timer::tick("cal_density_recip_batch", "real2recip_batch");
    rhopw_dev->real_to_recip_batch<float, base_device::DEVICE_GPU>(
        this->ctx,
        density_real_batch,      // Input: batch_size × nrxx
        density_recip_batch,     // Output: batch_size × npw
        batch_size,
        false,
        1.0f);
    ModuleBase::timer::tick("cal_density_recip_batch", "real2recip_batch");

    ModuleBase::timer::tick("OperatorEXXPW", "cal_density_recip_batch");
}

// ============================================================================
// Batch FFT size accessor
// ============================================================================

template <typename T, typename Device>
int OperatorEXXPW<T, Device>::get_batch_fft_size() const
{
    return this->wfcpw->fft_bundle.get_batch_size<typename GetTypeReal<T>::type>();
}

// Explicit template instantiations
template int OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>::get_batch_fft_size() const;
template int OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>::get_batch_fft_size() const;
#if defined(__CUDA) || defined(__ROCM)
template int OperatorEXXPW<std::complex<float>, base_device::DEVICE_GPU>::get_batch_fft_size() const;
template int OperatorEXXPW<std::complex<double>, base_device::DEVICE_GPU>::get_batch_fft_size() const;
#endif

#endif


} // namespace hamilt

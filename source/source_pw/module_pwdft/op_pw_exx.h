#ifndef OPEXXPW_H
#define OPEXXPW_H

#include "op_pw.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/macros.h"
#include "source_base/matrix.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_cell/klist.h"
#include "source_lcao/module_ri/conv_coulomb_pot_k.h"
#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"
#include "source_psi/psi.h"
#include "source_base/module_container/ATen/kernels/lapack.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <utility>
#include <vector>

namespace hamilt
{

enum class ExxSingularCorrectionMode
{
    None,
    ScfMp,
    SmoothTarget
};

template <typename T>
class ExxWaveRedistributorCpu;

struct ExxOperatorOptions
{
    int batch_fft_size = 1;
    int band_tile_size = 1;
    int q_tile_size = 1;
    int nspin = 1;
    double ecutexx = 0.0;
    double ecutrho = 0.0;
    bool gamma_extrapolation = false;
    bool exxace = false;
    bool separate_loop = false;
    double hybrid_alpha = 0.0;
    std::vector<std::map<std::string, std::string>> fock_params;
    std::vector<std::map<std::string, std::string>> erfc_params;
};

struct ExxLocalEnergyKPoint
{
    int ik_rep_spin = -1;
    int ispin = 0;
    K_Vectors::ExxFullKPoint kpoint;
};

namespace exx_energy_k_policy
{
inline int spin_channel_count(int nspin)
{
    return nspin == 2 ? 2 : 1;
}

inline bool uses_reduced_k_mesh(const K_Vectors& kv)
{
    return kv.get_nkstot_full() > 0 && kv.get_nkstot() > 0 && kv.get_nkstot_full() > kv.get_nkstot();
}

struct ExxQkPair
{
    int q_index = -1;
    const K_Vectors::ExxFullQPoint* qpoint = nullptr;
    int multiplier = 0;
};

inline int representative_spin_index(const K_Vectors& kv, const K_Vectors::ExxFullKPoint& point, int ispin, int nspin)
{
    if (point.rep_pool < 0)
    {
        ModuleBase::WARNING_QUIT("exx_energy_k_policy::representative_spin_index",
                                 "negative EXX representative pool");
    }
    if (point.rep_local_index < 0)
    {
        ModuleBase::WARNING_QUIT("exx_energy_k_policy::representative_spin_index",
                                 "negative EXX representative local index");
    }

    int nk_local_no_spin = kv.get_nks();
    if (nspin == 2)
    {
        if (point.rep_pool >= static_cast<int>(kv.para_k.nks_pool.size()))
        {
            ModuleBase::WARNING_QUIT("exx_energy_k_policy::representative_spin_index",
                                     "EXX representative pool is out of range");
        }
        nk_local_no_spin = kv.para_k.nks_pool[point.rep_pool];
    }
    if (ispin < 0 || ispin >= spin_channel_count(nspin))
    {
        ModuleBase::WARNING_QUIT("exx_energy_k_policy::representative_spin_index", "invalid EXX spin index");
    }
    return point.rep_local_index + ispin * nk_local_no_spin;
}

inline std::vector<ExxLocalEnergyKPoint> choose_active_full_k_points(const K_Vectors& kv, int nspin)
{
    const int nspin_fac = spin_channel_count(nspin);
    std::vector<ExxLocalEnergyKPoint> points;
    points.reserve(static_cast<std::size_t>(nspin_fac) * kv.exx_full_k_map.size());

    for (int ispin = 0; ispin < nspin_fac; ++ispin)
    {
        for (const auto& kpoint: kv.exx_full_k_map)
        {
            if (!kpoint.active)
            {
                continue;
            }
            ExxLocalEnergyKPoint point;
            point.ik_rep_spin = representative_spin_index(kv, kpoint, ispin, nspin);
            point.ispin = ispin;
            point.kpoint = kpoint;
            points.push_back(point);
        }
    }
    return points;
}

inline std::vector<ExxLocalEnergyKPoint> choose_local_representative_k_points(const K_Vectors& kv,
                                                                              int local_nks,
                                                                              int nspin,
                                                                              int my_pool)
{
    const int nspin_fac = spin_channel_count(nspin);
    const int local_nk_no_spin = local_nks / nspin_fac;
    std::vector<ExxLocalEnergyKPoint> points;
    points.reserve(std::max(0, local_nks));

    for (int ispin = 0; ispin < nspin_fac; ++ispin)
    {
        for (int ik_local = 0; ik_local < local_nk_no_spin; ++ik_local)
        {
            const int ik_rep_spin = ik_local + ispin * local_nk_no_spin;
            for (const auto& kpoint: kv.exx_full_k_map)
            {
                if (!kpoint.active || kpoint.rep_pool != my_pool || kpoint.rep_local_index != ik_local)
                {
                    continue;
                }
                if (kpoint.full_index == kpoint.rep_index)
                {
                    ExxLocalEnergyKPoint point;
                    point.ik_rep_spin = ik_rep_spin;
                    point.ispin = ispin;
                    point.kpoint = kpoint;
                    points.push_back(point);
                    break;
                }
            }
        }
    }
    return points;
}

inline std::vector<ExxLocalEnergyKPoint> choose_star_member_k_points(const K_Vectors& kv,
                                                                     const ExxLocalEnergyKPoint& representative)
{
    std::vector<ExxLocalEnergyKPoint> points;
    for (const auto& kpoint: kv.exx_full_k_map)
    {
        if (!kpoint.active || kpoint.rep_pool != representative.kpoint.rep_pool
            || kpoint.rep_local_index != representative.kpoint.rep_local_index)
        {
            continue;
        }
        ExxLocalEnergyKPoint point;
        point.ik_rep_spin = representative.ik_rep_spin;
        point.ispin = representative.ispin;
        point.kpoint = kpoint;
        points.push_back(point);
    }
    return points;
}

inline int representative_cache_scope(const ExxLocalEnergyKPoint& point)
{
    return point.kpoint.rep_index;
}

inline double scaled_full_point_occupation(double occupied_weight,
                                           double representative_weight,
                                           double full_point_weight)
{
    if (occupied_weight < 1.0e-12 || std::abs(full_point_weight) < 1.0e-14)
    {
        return 0.0;
    }
    if (std::abs(representative_weight) < 1.0e-14)
    {
        ModuleBase::WARNING_QUIT("exx_energy_k_policy::scaled_full_point_occupation",
                                 "nonzero EXX full-point occupation has zero representative weight");
    }
    return occupied_weight / representative_weight * full_point_weight;
}

inline std::size_t direct_potential_cache_entry_limit(std::size_t star_size, std::size_t q_tile_count)
{
    std::size_t result = 0;
    if (!checked_exx_size_product(star_size, q_tile_count, result))
    {
        ModuleBase::WARNING_QUIT("exx_energy_k_policy::direct_potential_cache_entry_limit",
                                 "PW EXX direct potential-cache size overflows size_t");
    }
    return result;
}

inline std::vector<ExxQkPair> build_qk_pair_map(const K_Vectors::ExxFullKPoint& kpoint,
                                                const std::vector<const K_Vectors::ExxFullQPoint*>& q_points)
{
    (void)kpoint;
    std::vector<ExxQkPair> qk_map;
    qk_map.reserve(q_points.size());
    for (int iq = 0; iq < static_cast<int>(q_points.size()); ++iq)
    {
        const K_Vectors::ExxFullQPoint* qpoint = q_points[iq];
        if (qpoint == nullptr || !qpoint->active)
        {
            continue;
        }
        ExxQkPair pair;
        pair.q_index = iq;
        pair.qpoint = qpoint;
        pair.multiplier = 1;
        qk_map.push_back(pair);
    }
    return qk_map;
}
} // namespace exx_energy_k_policy

template <typename T, typename Device>
class OperatorEXXPW : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    OperatorEXXPW(const int* isk_in,
                  const ModulePW::PW_Basis_K* wfcpw_in,
                  const ModulePW::PW_Basis* rhopw_in,
                  K_Vectors* kv_in,
                  const UnitCell* ucell,
                  const ExxOperatorOptions& options_in);

    template <typename T_in, typename Device_in = Device>
    explicit OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op_exx);

    OperatorEXXPW(const OperatorEXXPW<T, Device>* source_op,
                  const int* target_isk,
                  const ModulePW::PW_Basis_K* target_wfcpw,
                  const K_Vectors* target_kv,
                  const ExxOperatorOptions& options_in);

    virtual ~OperatorEXXPW();

    virtual void act(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T *tmpsi_in,
                     T *tmhpsi,
                     const int ngk_ik = 0,
                     const bool is_first_node = false) const override;

    double cal_exx_energy(psi::Psi<T, Device> *psi_) const;

    double cal_exx_energy_exact(psi::Psi<T, Device> *psi_) const;

    void set_psi(psi::Psi<T, Device> &psi_in) const;

    void set_wg(const ModuleBase::matrix *wg_in) { wg = wg_in; }

    int get_batch_fft_size() const;

    void construct_ace() const;

    bool first_iter = true;

    static std::vector<Real> fock_div, erfc_div;

  private:
    const int* isk = nullptr;
    const ModulePW::PW_Basis_K* wfcpw = nullptr;
    const ModulePW::PW_Basis* rhopw = nullptr;
    ModulePW::PW_Basis_K* wfcpw_exx = nullptr; // k-dependent EXX grid
    ModulePW::PW_Basis* rhopw_dev = nullptr; // for device
    const OperatorEXXPW<T, Device>* source_op_for_target = nullptr;
    const K_Vectors* target_kv_for_target = nullptr;
    bool owns_wfcpw_exx = true;
    bool owns_rhopw_dev = true;
    const UnitCell *ucell = nullptr;
    Real tpiba = 0;
    
    std::vector<const K_Vectors::ExxFullQPoint*> get_q_points(const int ik) const;
    std::vector<const K_Vectors::ExxFullQPoint*> get_active_q_points() const;
    std::vector<const K_Vectors::ExxFullKPoint*> get_k_points() const;
    const T *get_pw(const int m, const int ik_local) const;
    int rep_spin_index(const K_Vectors::ExxFullPoint& point, int ispin) const;
    const K_Vectors::ExxFullKPoint& local_representative_kpoint(int ik_local, int ispin) const;
    void ensure_full_point_supported(const K_Vectors::ExxFullPoint& point) const;
    using FullPointSpatialRemap = ExxSymmetryRemap;
    const FullPointSpatialRemap& point_spatial_remap(const K_Vectors::ExxFullPoint& point, int rep_spin_index) const;
    void load_full_point_real(const K_Vectors::ExxFullPoint& point, int ispin, int iband, T* out) const;
    void load_full_point_real_uncached(const K_Vectors::ExxFullPoint& point, int ispin, int iband, T* out) const;
    void load_full_point_real_batch(const K_Vectors::ExxFullPoint& point,
                                    int ispin,
                                    const int* band_indices,
                                    int batch_count,
                                    T* out) const;
    void set_psi_for_cache(const psi::Psi<T, Device>& psi_in) const;
    void wave_recip_to_exx_real(const T* psi_recip, T* psi_real, int ik_local) const;
    void exx_real_to_wave_recip(const T* psi_real, T* h_psi_recip, int ik_local, Real factor) const;
    const T* wave_recip_to_exx_recip(const T* psi_recip, int ik_local, T* scratch) const;
    void ensure_exx_wave_mapping() const;
    void ensure_remap_device_cache(const FullPointSpatialRemap& remap) const;
    void clear_remap_device_cache() const;
    Real* get_exx_potential_cached(const K_Vectors::ExxFullKPoint& kpoint,
                                   const K_Vectors::ExxFullQPoint& qpoint) const;
    void clear_exx_potential_cache() const;
    void reset_exx_potential_cache_for_scope(int cache_scope) const;
    int resolve_qtile_chunk_size() const;
    void ensure_qtile_workspace(std::size_t target_size, std::size_t q_size, std::size_t batch_limit) const;
    void allocate_batch_workspace(int batch_fft_size);
    void fill_target_tile(const T* tmpsi_in,
                          int nbasis,
                          int n_start,
                          int n_count,
                          T* target_real) const;
    Real fill_q_tile_states(const std::vector<const K_Vectors::ExxFullQPoint*>& q_points,
                            int q_start,
                            int q_count,
                            int ispin,
                            int m_start,
                            int m_count,
                            int source_tile_size,
                            T* q_real,
                            Real* q_weights,
                            bool load_wavefunctions) const;
    void act_op_qtile(const int nbands,
                      const int nbasis,
                      const int npol,
                      const T* tmpsi_in,
                      T* tmhpsi,
                      const int ngk_ik,
                      const bool is_first_node,
                      bool accumulate_hpsi,
                      int ispin_override,
                      const K_Vectors::ExxFullKPoint* target_kpoint_override,
                      int target_ik_override) const;
    void process_qtile_apply_tile(const K_Vectors::ExxFullKPoint& local_kpoint,
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
                                  int chunk_size) const;
    double process_qtile_energy_tile(const K_Vectors::ExxFullKPoint& kpoint,
                                     const K_Vectors::ExxFullQPoint& qpoint,
                                     const T* psi_nk_real_in,
                                     const T* q_real,
                                     const Real* q_weights,
                                     const Real* q_weights_device,
                                     Real k_weight,
                                     int q_local,
                                     int m_count,
                                     int source_tile_size,
                                     int chunk_size) const;
    void accumulate_full_point_recip(const K_Vectors::ExxFullPoint& point,
                                     int ispin,
                                     const T* full_real,
                                     T* rep_recip,
                                     Real factor) const;

    void multiply_potential(T *density_recip, int ik, int iq) const;

    void act_op(const int nbands,
                const int nbasis,
                const int npol,
                const T *tmpsi_in,
                T *tmhpsi,
                const int ngk_ik = 0,
                const bool is_first_node = false) const;

    void act_op_qtile_cpu(const int nbands,
                          const int nbasis,
                          const int npol,
                          const T *tmpsi_in,
                          T *tmhpsi,
                          const int ngk_ik,
                          const bool is_first_node,
                          bool accumulate_hpsi,
                          int ispin_override) const;

    void act_op_qtile_gpu(const int nbands,
                          const int nbasis,
                          const int npol,
                          const T *tmpsi_in,
                          T *tmhpsi,
                          const int ngk_ik,
                          const bool is_first_node,
                          bool accumulate_hpsi,
                          int ispin_override) const;

    void act_op_ace(const int nbands,
                    const int nbasis,
                    const int npol,
                    const T *tmpsi_in,
                    T *tmhpsi,
                    const int ngk_ik = 0,
                    const bool is_first_node = false) const;

    double cal_exx_energy_op_qtile(psi::Psi<T, Device> *psi_) const;

    double cal_exx_energy_ace(psi::Psi<T, Device> *psi_) const;

    void cal_density_recip(const T* psi_nk_real, const T* psi_mq_real, double omega) const;
    void cal_density_recip_batch(const T* psi_nk_real,
                                 T* psi_mq_real_batch,
                                 T* density_real_batch,
                                 T* density_recip_batch,
                                 int batch_size,
                                 int ik,
                                 double omega) const;

    void rho_recip2real(const T* rho_recip, T* rho_real, bool add, Real factor) const;

    mutable int cnt = 0;

    mutable bool potential_got = false;
    
    // pws
//    mutable std::vector<std::unique_ptr<T[]>> pws;

    // k vectors
    K_Vectors *kv = nullptr;
    ExxOperatorOptions options;

    // psi
    mutable psi::Psi<T, Device> psi;
    const ModuleBase::matrix* wg;

    // real space memory
    T *psi_nk_real = nullptr;
    T *psi_mq_real = nullptr;
    T *density_real = nullptr;
    T *h_psi_real = nullptr;
    // density recip space memory
    T *density_recip = nullptr;
    // h_psi recip space memory
    T *h_psi_recip = nullptr;
    T *psi_nk_exx_recip = nullptr;
    T *psi_mq_exx_recip = nullptr;
    T *h_psi_exx_recip = nullptr;
    Real *pot = nullptr;

    T *psi_mq_batch_real = nullptr;
    T *psi_mq_batch_recip = nullptr;
    T *density_real_batch = nullptr;
    T *density_recip_batch = nullptr;
    Real *density_norm_batch = nullptr;
    Real *energy_batch = nullptr;
    mutable T *alpha_all_device = nullptr;
    mutable std::size_t alpha_all_capacity = 0;
    mutable int alpha_cached_ik = std::numeric_limits<int>::min();
    mutable int alpha_cached_nbands = 0;
    mutable std::size_t alpha_cached_q_count = 0;
    mutable std::vector<T> alpha_all_host_cache;
    mutable std::vector<T> alpha_all_host_work;
    mutable std::vector<const T*> psi_mq_ptrs_cache;
    mutable std::vector<int> batch_local_band_idx_cache;
    mutable std::vector<int> batch_actual_band_idx_cache;
    mutable std::vector<int> exx_to_wfc_map_host;
    mutable std::vector<int> exx_to_wfc_offsets;
    mutable int* exx_to_wfc_map_device = nullptr;
    mutable std::size_t exx_to_wfc_map_device_capacity = 0;
    mutable std::unique_ptr<ExxWaveRedistributorCpu<T>> exx_wave_redistributor;
    mutable std::vector<Real> weight_real_host_cache;
    mutable Real* weight_real_device = nullptr;
    mutable std::size_t weight_real_capacity = 0;
    mutable std::map<std::pair<int, int>, Real*> pot_cache;
    mutable int cached_potential_scope = std::numeric_limits<int>::min();

    // Lin Lin's ACE memory, 10.1021/acs.jctc.6b00092
    mutable T* h_psi_ace = nullptr; // H \Psi, W in the paper
    mutable T* psi_h_psi_ace = nullptr; // \Psi^{\dagger} H \Psi, M in the paper
    mutable T* L_ace = nullptr; // cholesky(-M).L, L in the paper
    mutable T* Xi_psi_ace = nullptr; // temporary Xi * psi in ACE application
    mutable std::vector<T*> Xi_ace_k; // L^{-1} (H \Psi)^{\dagger}, \Xi in the paper
    mutable int ace_scratch_nk = 0;
    mutable int ace_scratch_nbands = 0;
    mutable int ace_scratch_nbasis = 0;
    mutable int ace_apply_scratch_nbands_tot = 0;
    mutable int ace_apply_scratch_nbands = 0;
//    mutable T* Xi_ace = nullptr; // L^{-1} (H \Psi)^{\dagger}, \Xi in the paper

    mutable std::map<int, std::vector<const K_Vectors::ExxFullQPoint*>> q_points;
    mutable std::vector<const K_Vectors::ExxFullKPoint*> k_points;
    mutable std::map<int, K_Vectors::ExxFullKPoint> local_kpoint_cache;
    mutable std::map<std::pair<int, int>, FullPointSpatialRemap> point_gmaps;
    mutable T* qtile_target_real = nullptr;
    mutable T* qtile_h_real = nullptr;
    mutable T* qtile_q_real = nullptr;
    mutable Real* qtile_energy_device = nullptr;
    mutable std::size_t qtile_target_real_size = 0;
    mutable std::size_t qtile_h_real_size = 0;
    mutable std::size_t qtile_q_real_size = 0;

    // occupational number
    const ModuleBase::matrix *p_wg;

//    mutable bool update_psi = false;

    Device *ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::AbacusDevice_t device = {};

    using ct_Device = typename ct::PsiToContainer<Device>::type;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using setmem_real_op = base_device::memory::set_memory_op<Real, Device>;
    using setmem_real_cpu_op = base_device::memory::set_memory_op<Real, base_device::DEVICE_CPU>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using syncmem_complex_op = base_device::memory::synchronize_memory_op<T, Device, Device>;
    using resmem_real_op = base_device::memory::resize_memory_op<Real, Device>;
    using delmem_real_op = base_device::memory::delete_memory_op<Real, Device>;
    using gemm_complex_op = ModuleBase::gemm_op<T, Device>;
    using axpy_complex_op = ModuleBase::axpy_op<T, Device>;
    using vec_add_vec_complex_op = ModuleBase::vector_add_vector_op<T, Device>;
    using dot_op = ModuleBase::dot_real_op<T, Device>;
    using syncmem_complex_c2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;
    using syncmem_complex_d2c_op = base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, Device>;
    using syncmem_real_c2d_op = base_device::memory::synchronize_memory_op<Real, Device, base_device::DEVICE_CPU>;
    using syncmem_real_d2c_op = base_device::memory::synchronize_memory_op<Real, base_device::DEVICE_CPU, Device>;
    using lapack_potrf = container::kernels::lapack_potrf<T, ct_Device>;
    using lapack_trtri = container::kernels::lapack_trtri<T, ct_Device>;

    bool gamma_extrapolation = true;
    ExxSingularCorrectionMode singular_correction_mode = ExxSingularCorrectionMode::ScfMp;
    std::vector<Real> fock_div_local;
    std::vector<Real> erfc_div_local;

};

template <typename Real, typename Device>
void get_exx_potential(const K_Vectors* kv,
                       const ModulePW::PW_Basis_K* wfcpw,
                       ModulePW::PW_Basis* rhopw_dev,
                       Real* pot,
                       double tpiba,
                       ExxSingularCorrectionMode singular_correction_mode,
                       const std::vector<Real>* fock_div_override,
                       const std::vector<Real>* erfc_div_override,
                       double ucell_omega,
                       int ik,
                       int iq,
                       bool is_stress);

template <typename Real, typename Device>
void get_exx_potential(const K_Vectors* kv,
                       const ModulePW::PW_Basis_K* wfcpw,
                       ModulePW::PW_Basis* rhopw_dev,
                       Real* pot,
                       double tpiba,
                       ExxSingularCorrectionMode singular_correction_mode,
                       const std::vector<Real>* fock_div_override,
                       const std::vector<Real>* erfc_div_override,
                       double ucell_omega,
                       const K_Vectors::ExxFullKPoint& kpoint,
                       const K_Vectors::ExxFullQPoint& qpoint,
                       bool is_stress);

template <typename Real, typename Device>
void get_exx_stress_potential(const K_Vectors* kv,
                              const ModulePW::PW_Basis_K* wfcpw,
                              ModulePW::PW_Basis* rhopw_dev,
                              Real* pot,
                              double tpiba,
                              bool gamma_extrapolation,
                              double ucell_omega,
                              int ik,
                              int iq);

template <typename Real, typename Device>
void get_exx_stress_potential(const K_Vectors* kv,
                              const ModulePW::PW_Basis_K* wfcpw,
                              ModulePW::PW_Basis* rhopw_dev,
                              Real* pot,
                              double tpiba,
                              bool gamma_extrapolation,
                              double ucell_omega,
                              const K_Vectors::ExxFullKPoint& kpoint,
                              const K_Vectors::ExxFullQPoint& qpoint);

double exx_divergence(Conv_Coulomb_Pot_K::Coulomb_Type coulomb_type,
                      double erfc_omega,
                      const K_Vectors* kv,
                      const ModulePW::PW_Basis_K* wfcpw,
                      ModulePW::PW_Basis* rhopw_dev,
                      double tpiba,
                      ExxSingularCorrectionMode singular_correction_mode,
                      double ucell_omega);

} // namespace hamilt

#endif // OPEXXPW_H

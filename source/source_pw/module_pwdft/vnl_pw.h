#ifndef VNL_IN_PW_H
#define VNL_IN_PW_H

#include "source_base/complexarray.h"
#include "source_base/complexmatrix.h"
#include "source_base/intarray.h"
#include "source_base/realarray.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/soc.h"
#include "source_pw/module_pwdft/structure_factor.h"
#include "source_psi/psi.h"
#include <algorithm>
#include <cstdlib>
#include <string>
#if defined(__CUDA) || defined(__UT_USE_CUDA)
#include <cuda_runtime.h>
#endif
#ifdef __LCAO
#include "source_basis/module_ao/ORB_gaunt_table.h"
#endif

inline int get_chunked_vnl_override()
{
    const char* env = std::getenv("ABACUS_VNL_CHUNKED");
    if (env == nullptr)
    {
        return -1;
    }
    return std::string(env) == "1" ? 1 : 0;
}

inline int get_vnl_chunk_size_override()
{
    const char* env = std::getenv("ABACUS_VNL_CHUNK_SIZE");
    return env == nullptr ? 0 : std::max(0, std::atoi(env));
}

inline size_t get_vnl_chunk_memory_budget_override()
{
    const char* env = std::getenv("ABACUS_VNL_CHUNK_BUDGET_MB");
    if (env == nullptr)
    {
        return 0;
    }
    const long budget_mb = std::atol(env);
    return budget_mb > 0 ? static_cast<size_t>(budget_mb) * 1024ULL * 1024ULL : 0;
}

inline size_t vnl_chunking_memory_budget_bytes()
{
    const size_t override_budget = get_vnl_chunk_memory_budget_override();
    if (override_budget > 0)
    {
        return override_budget;
    }

    const size_t gib = 1024ULL * 1024ULL * 1024ULL;
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes > 0)
    {
        return std::max<size_t>(gib, free_bytes / 2);
    }
#endif
    return 4ULL * gib;
}

inline bool vnl_chunking_enabled(const int nkb, const int npwx, const size_t element_size)
{
#if !defined(__CUDA) && !defined(__UT_USE_CUDA)
    return false;
#else
    const int override_val = get_chunked_vnl_override();
    if (override_val == 1)
    {
        return true;
    }
    if (override_val == 0)
    {
        return false;
    }
    const size_t vkb_size = static_cast<size_t>(nkb) * static_cast<size_t>(npwx) * element_size;
    return vkb_size > vnl_chunking_memory_budget_bytes();
#endif
}

inline bool force_stress_should_use_chunked_vnl(const bool is_gpu,
                                                const bool has_full_vkb,
                                                const bool chunked_vnl_enabled,
                                                const int nkb)
{
    return is_gpu && nkb > 0 && (chunked_vnl_enabled || !has_full_vkb);
}

//==========================================================
// Calculate the non-local pseudopotential in reciprocal
// space using plane wave as basis set.
//==========================================================
class pseudopot_cell_vnl
{

  public:
    pseudopot_cell_vnl();
    ~pseudopot_cell_vnl();
    void init(const UnitCell& cell,
              Structure_Factor* psf_in,
              const ModulePW::PW_Basis_K* wfc_basis = nullptr,
              const bool allocate_vkb = true);

    double cell_factor = 0.0; // LiuXh add 20180619

    int nkb = 0; // total number of beta functions considering all atoms

    int lmaxkb = 0; // max angular momentum for non-local projectors

    void init_vnl(UnitCell& cell,
                  const ModulePW::PW_Basis* rho_basis,
                  bool prepare_uspp_stress);

    void rescale_vnl(const double& omega_in);

    template <typename FPTYPE, typename Device>
    void getvnl(Device* ctx, const UnitCell& ucell, const int& ik, std::complex<FPTYPE>* vkb_in) const;

    template <typename FPTYPE, typename Device>
    void cal_becp_matrix_free(Device* ctx,
                              const UnitCell& ucell,
                              int ik,
                              int npw,
                              int npwx,
                              int nbands,
                              const std::complex<FPTYPE>* psi,
                              std::complex<FPTYPE>* becp) const;

    template <typename FPTYPE, typename Device>
    void getvnl_atoms_cached(Device* ctx,
                             const UnitCell& ucell,
                             const int& ik,
                             int atom_start,
                             int atom_end,
                             const FPTYPE* gk,
                             const FPTYPE* ylm,
                             const std::complex<FPTYPE>* sk_all,
                             std::complex<FPTYPE>* vkb_out) const;

    // void getvnl_alpha(const int &ik);

    void init_vnl_alpha(const UnitCell& cell);

    void initgradq_vnl(const UnitCell& cell);

    void getgradq_vnl(const UnitCell& ucell, const int ik);

    //===============================================================
    // MEMBER VARIABLES :
    // NAME : nqx(number of interpolation points)
    // NAME : nqxq(size of interpolation table)
    // NAME : nhm(max number of different beta functions per atom)
    // NAME : lmaxq
    // NAME : dq(space between points in the pseudopotential tab)
    //===============================================================
    // private:

    int nhm = 0;
    int nbetam = 0; // max number of beta functions

    int lmaxq = 0;

    ModuleBase::matrix indv;   // indes linking  atomic beta's to beta's in the solid
    ModuleBase::matrix nhtol;  // correspondence n <-> angular momentum l
    ModuleBase::matrix nhtolm; // correspondence n <-> combined lm index for (l,m)
    ModuleBase::matrix nhtoj;  // new added

    ModuleBase::realArray dvan;       //(:,:,:),  the D functions of the solid
    ModuleBase::ComplexArray dvan_so; //(:,:,:),  spin-orbit case,  added by zhengdy-soc

    ModuleBase::realArray tab; //(:,:,:), interpolation table for PPs: 4pi/sqrt(V) * \int betar(r)jl(qr)rdr
    ModuleBase::realArray tab_alpha;
    ModuleBase::realArray tab_at; //(:,:,:), interpolation table for atomic wfc
    ModuleBase::realArray tab_dq; // 4pi/sqrt(V) * \int betar(r)*djl(qr)/d(qr)*r^2 dr

    ModuleBase::realArray deeq; //(:,:,:,:), the integral of V_eff and Q_{nm}
    bool multi_proj = false;
    float* s_deeq = nullptr;
    double* d_deeq = nullptr;
    double* d_dvan = nullptr;
    ModuleBase::ComplexArray deeq_nc;          //(:,:,:,:), the spin-orbit case
    std::complex<float>* c_deeq_nc = nullptr;  // GPU array of deeq_nc
    std::complex<double>* z_deeq_nc = nullptr; // GPU array of deeq_nc

    // liuyu add 2023-10-03
    // uspp
    int* indv_ijkb0 = nullptr;      // first beta (index in the solid) for each atom
    ModuleBase::IntArray ijtoh;     // correspondence beta indexes ih,jh -> composite index ijh
    ModuleBase::realArray qq_at;    // the integral of q functions in the solid (ONE PER ATOM)
    ModuleBase::realArray qq_nt;    // the integral of q functions in the solid (ONE PER NTYP) used to be the qq array
    ModuleBase::ComplexArray qq_so; // Q_{nm} for spin-orbit case
    ModuleBase::realArray ap;       // the expansion coefficients
    ModuleBase::IntArray lpx;       // for each input limi,ljmj is the number of LM in the sum
    ModuleBase::IntArray lpl;       // for each input limi,ljmj points to the allowed LM
    ModuleBase::realArray qrad;     // radial FT of Q functions
    ModuleBase::ComplexArray qgm;   // packed Q functions on the dense reciprocal grid
    ModuleBase::ComplexArray dqgm;  // Cartesian derivatives of packed Q functions
    ModuleBase::ComplexMatrix qgm_phase; // atom phases on the dense reciprocal grid
    ModuleBase::matrix qgm_gcar; // dense reciprocal vectors for USPP device contractions

    float* s_qq_nt = nullptr;
    double* d_qq_nt = nullptr;
    std::complex<float>* c_qq_so = nullptr;  // GPU array of qq_so
    std::complex<double>* z_qq_so = nullptr; // GPU array of qq_so
    std::complex<float>* c_qgm = nullptr;
    std::complex<double>* z_qgm = nullptr;
    std::complex<double>* z_dqgm = nullptr;
    std::complex<float>* c_qgm_phase = nullptr;
    std::complex<double>* z_qgm_phase = nullptr;
    double* d_qgm_gcar = nullptr;

    mutable ModuleBase::ComplexMatrix vkb;    // all beta functions in reciprocal space
    mutable ModuleBase::ComplexArray gradvkb; // gradient of beta functions
    std::complex<double>*** vkb1_alpha;
    std::complex<double>*** vkb_alpha;
    Structure_Factor* psf = nullptr;

    // Column dimension of vkb matrix (= npwx), used as leading dimension in gemm/gemv.
    // On GPU path vkb ComplexMatrix is not allocated to save CPU memory; this stores the dimension.
    int vkbnc = 0;
    bool has_full_float_vkb = false;
    bool has_full_double_vkb = false;

    mutable double* cached_vkb1_double = nullptr;
    mutable float* cached_vkb1_float = nullptr;
    mutable int cached_vkb1_double_nhm = 0;
    mutable int cached_vkb1_double_npw = 0;
    mutable int cached_vkb1_float_nhm = 0;
    mutable int cached_vkb1_float_npw = 0;

    // other variables
    std::complex<double> Cal_C(int alpha, int lu, int mu, int L, int M);

    double CG(int l1, int m1, int l2, int m2, int L, int M);

    void print_vnl(std::ofstream& ofs);

    /**
     * @brief Compute the radial Fourier transform of the Q functions
     *
     * The interpolation table for the radial Fourier transform is stored in qrad.
     *
     * The formula implemented here is:
     *   \[ q(g,i,j) = \sum_\text{lm} (-i)^l \text{ap}(\text{lm},i,j)
     *   \text{yr}_\text{lm}(g) \text{qrad}(g,l,i,j) \]
     *
     * @param ng [in] the number of G vectors
     * @param ih [in] the first index of Q
     * @param jh [in] the second index of Q
     * @param itype [in] the atomic type
     * @param qnorm [in] the norm of q+g vectors
     * @param ylm [in] the real spherical harmonics
     * @param qg [out] the Fourier transform of interest
     */
    void radial_fft_q(const int ng,
                      const int ih,
                      const int jh,
                      const int itype,
                      const double* qnorm,
                      const ModuleBase::matrix ylm,
                      std::complex<double>* qg) const;

    /**
     * @brief Compute one Cartesian derivative of a packed USPP Q function.
     *
     * The radial interpolation and spherical-harmonic derivative are shared by
     * the host stress path and the derivative-Q device cache.
     */
    void radial_fft_dq(const int ng,
                       const int ih,
                       const int jh,
                       const int itype,
                       const int ipol,
                       const ModuleBase::Vector3<double>* g,
                       const double* qnorm,
                       const double tpiba,
                       const ModuleBase::matrix& ylm,
                       const ModuleBase::matrix& dylm,
                       std::complex<double>* dqg) const;
    template <typename FPTYPE, typename Device>
    void radial_fft_q(Device* ctx,
                      const int ng,
                      const int ih,
                      const int jh,
                      const int itype,
                      const FPTYPE* qnorm,
                      const FPTYPE* ylm,
                      std::complex<FPTYPE>* qg) const;

    /**
     * @brief calculate the effective coefficient matrix for non-local pseudopotential projectors
     *
     * @param veff effective potential
     * @param rho_basis potential FFT grids
     * @param cell UnitCell
     */
    void cal_effective_D(const ModuleBase::matrix& veff, const ModulePW::PW_Basis* rho_basis, UnitCell& cell);
    void cal_effective_D_gpu(const double* veff,
                             const ModulePW::PW_Basis* rho_basis,
                             UnitCell& cell);
#ifdef __LCAO
    ORB_gaunt_table MGT;
#endif

    template <typename FPTYPE>
    FPTYPE* get_nhtol_data() const;
    template <typename FPTYPE>
    FPTYPE* get_nhtolm_data() const;
    template <typename FPTYPE>
    FPTYPE* get_indv_data() const;
    template <typename FPTYPE>
    FPTYPE* get_tab_data() const;
    template <typename FPTYPE>
    FPTYPE* get_deeq_data() const;
    template <typename FPTYPE>
    FPTYPE* get_qq_nt_data() const;
    template <typename FPTYPE>
    std::complex<FPTYPE>* get_qq_so_data() const;
    template <typename FPTYPE>
    std::complex<FPTYPE>* get_vkb_data() const;
    template <typename FPTYPE>
    std::complex<FPTYPE>* get_qgm_data() const;
    template <typename FPTYPE>
    std::complex<FPTYPE>* get_qgm_phase_data() const;
    const std::complex<double>* get_dqgm_data() const
    {
        return this->z_dqgm;
    }
    const double* get_qgm_gcar_data() const
    {
        return this->d_qgm_gcar;
    }
    bool has_qgm_cache() const
    {
        return this->qgm_cache_ready;
    }
    template <typename FPTYPE>
    std::complex<FPTYPE>* get_deeq_nc_data() const;

    void release_memory();

  private:
    bool memory_released = false;
    bool qgm_cache_ready = false;
    float* s_nhtol = nullptr;
    float* s_nhtolm = nullptr;
    float* s_indv = nullptr;
    float* s_tab = nullptr;
    std::complex<float>* c_vkb = nullptr;

    double* d_nhtol = nullptr;
    double* d_nhtolm = nullptr;
    double* d_indv = nullptr;
    double* d_tab = nullptr;
    std::complex<double>* z_vkb = nullptr;

    const ModulePW::PW_Basis_K* wfcpw = nullptr;

    Soc soc;

    double omega_old = 0;
    bool use_gpu_ = false;

    /**
     * @brief Compute interpolation table qrad
     *
     * Compute interpolation table qrad(i,nm,l,nt) = Q^{(L)}_{nm,nt}(q_i)
     * of angular momentum L, for atom of type nt, on grid q_i, where
     * nm = combined index for n,m=1,nh(nt)
     *
     * @param cell UnitCell
     */
    void compute_qrad(UnitCell& cell);

    /**
     * @brief computes the integral of the effective potential with the Q function
     *
     * @param veff effective potential
     * @param rho_basis potential FFT grids
     * @param cell UnitCell
     */
    void newq(const ModuleBase::matrix& veff, const ModulePW::PW_Basis* rho_basis, UnitCell& cell);

    /**
     * @brief calculate D functions in the soc case when tvanp is true
     *
     * @param iat the index of atom
     * @param cell UnitCell
     */
    void newd_so(const int& iat, UnitCell& cell);

    /**
     * @brief calculate D functions in the noncolin case when tvanp is true
     *
     * @param iat the index of atom
     * @param cell UnitCell
     */
    void newd_nc(const int& iat, UnitCell& cell);
};

#endif // VNL_IN_PW

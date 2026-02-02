#ifndef NONLOCALPW_H
#define NONLOCALPW_H

#include "op_pw.h"

#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "source_base/kernels/math_kernel_op.h"
#include "source_base/module_device/device.h"

//#include "source_pw/module_pwdft/global.h"

#include <cstdlib>  // for getenv
#include <string>
#include <iostream>
#include "source_pw/module_pwdft/vnl_pw.h"

namespace hamilt {

#ifndef NONLOCALTEMPLATE_H
#define NONLOCALTEMPLATE_H

template<class T> class Nonlocal : public T {};
// template<typename Real, typename Device = base_device::DEVICE_CPU>
// class Nonlocal : public OperatorPW<T, Device> {};

#endif

/// @brief Check if chunked VNL is explicitly set via environment variable
/// @return 1 if explicitly enabled, 0 if explicitly disabled, -1 if not set (auto mode)
inline int get_chunked_vnl_override()
{
    static int cached = -2;  // -2 means not yet checked
    if (cached == -2)
    {
        const char* env = std::getenv("ABACUS_VNL_CHUNKED");
        if (env == nullptr)
        {
            cached = -1;  // Auto mode
        }
        else
        {
            cached = (std::string(env) == "1") ? 1 : 0;
        }
    }
    return cached;
}

/// @brief Get chunk size override from environment variable
/// @return Chunk size if set, 0 if not set (use automatic sizing)
inline int get_chunk_size_override()
{
    static int cached = -1;
    if (cached == -1)
    {
        const char* env = std::getenv("ABACUS_VNL_CHUNK_SIZE");
        if (env != nullptr)
        {
            cached = std::atoi(env);
            if (cached <= 0) cached = 0;
        }
        else
        {
            cached = 0;
        }
    }
    return cached;
}

/// @brief Threshold for automatic chunked VNL activation (4 GB)
constexpr size_t VKB_SIZE_THRESHOLD_BYTES = 4ULL * 1024 * 1024 * 1024;

/// @brief Calculate the total number of nonlocal projectors (nkb) from UnitCell
/// @param ucell The unit cell containing atom information
/// @return Total number of projectors (sum of nh * na for all atom types)
inline int calculate_nkb(const UnitCell& ucell)
{
    int nkb = 0;
    for (int it = 0; it < ucell.ntype; it++)
    {
        nkb += ucell.atoms[it].ncpp.nh * ucell.atoms[it].na;
    }
    return nkb;
}

/// @brief Determine if chunked VNL processing should be used
///
/// Logic:
/// 1. If ABACUS_VNL_CHUNKED=1, always enable chunked (user override)
/// 2. If ABACUS_VNL_CHUNKED=0, always disable chunked (user override)
/// 3. Auto mode (no env var set):
///    - On CPU: never use chunked (no memory benefit, adds overhead)
///    - On GPU: enable chunked if vkb size > 4GB threshold
///
/// @tparam Device The device type (CPU or GPU)
/// @param nkb Total number of projectors
/// @param npwx Maximum number of plane waves
/// @param element_size Size of each element in bytes (sizeof(std::complex<T>))
/// @return true if chunked VNL should be used
template<typename Device>
inline bool use_chunked_vnl(int nkb, int npwx, size_t element_size)
{
    // 1. Check for explicit user override via environment variable
    int override_val = get_chunked_vnl_override();
    if (override_val == 1)
    {
        return true;  // Explicitly enabled by user
    }
    if (override_val == 0)
    {
        return false;  // Explicitly disabled by user
    }

    // 2. Auto mode: device-dependent logic
    // On CPU: never use chunked (no memory benefit, adds overhead)
    Device* ctx = {};
    if (base_device::get_device_type<Device>(ctx) != base_device::GpuDevice)
    {
        return false;
    }

    // On GPU: enable chunked if vkb would exceed threshold (4 GB)
    size_t vkb_size = static_cast<size_t>(nkb) * npwx * element_size;
    return vkb_size > VKB_SIZE_THRESHOLD_BYTES;
}

/// @brief Determine if chunked VNL processing should be used (convenience overload)
///
/// This overload computes nkb from UnitCell and uses the PW basis for npwx.
///
/// @tparam T The complex type (std::complex<float> or std::complex<double>)
/// @tparam Device The device type (CPU or GPU)
/// @param ucell The unit cell containing atom information
/// @param wfcpw The plane wave basis for wave functions
/// @return true if chunked VNL should be used
template<typename T, typename Device>
inline bool use_chunked_vnl(const UnitCell& ucell, const ModulePW::PW_Basis_K* wfcpw)
{
    int nkb = calculate_nkb(ucell);
    int npwx = wfcpw->npwk_max;
    return use_chunked_vnl<Device>(nkb, npwx, sizeof(T));
}

template<typename T, typename Device>
class Nonlocal<OperatorPW<T, Device>> : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;
  public:
    Nonlocal(const int* isk_in,
             const pseudopot_cell_vnl* ppcell_in,
             const UnitCell* ucell_in,
             const ModulePW::PW_Basis_K* wfc_basis);

    template<typename T_in, typename Device_in = Device>
    explicit Nonlocal(const Nonlocal<OperatorPW<T_in, Device_in>>* nonlocal);

    virtual ~Nonlocal();

    virtual void init(const int ik_in)override;

    virtual void act(const int nbands,
        const int nbasis,
        const int npol,
        const T* tmpsi_in,
        T* tmhpsi,
        const int ngk_ik = 0,
        const bool is_first_node = false)const override;

    const int *get_isk() const {return this->isk;}
    const pseudopot_cell_vnl *get_ppcell() const {return this->ppcell;}
    const UnitCell *get_ucell() const {return this->ucell;}
    T* get_vkb() const
    {
        return this->vkb;
    }
    T* get_becp() const
    {
        return this->becp;
    }

  private:
    void add_nonlocal_pp(T *hpsi_in, const T *becp, const int m) const;

    // =========== Chunked VNL processing methods ===========

    /// @brief Perform nonlocal operator action using chunked processing
    /// This processes atoms in chunks to reduce peak GPU memory usage for vkb.
    /// Falls back to standard act() if chunking is not enabled.
    void act_chunked(
        const int nbands,
        const int nbasis,
        const int npol,
        const T* tmpsi_in,
        T* tmhpsi,
        const int ngk_ik,
        const bool is_first_node) const;

    /// @brief Calculate optimal chunk size based on available GPU memory
    /// @param npw Number of plane waves for current k-point
    /// @param nkb Total number of projectors
    /// @param nbands Number of bands being processed
    /// @return Optimal number of projectors per chunk
    int calculate_optimal_chunk_size(int npw, int nkb, int nbands) const;

    /// @brief Ensure chunk buffers are large enough
    /// @param chunk_nkb Number of projectors in chunk
    /// @param npw Number of plane waves
    /// @param nbands Number of bands
    void ensure_chunk_buffers(int chunk_nkb, int npw, int nbands) const;

    /// @brief Process a single atom chunk
    /// @param psi Input wavefunctions
    /// @param hpsi Output Hamiltonian * psi (contributions accumulated)
    /// @param nbands Number of bands
    /// @param npw Number of plane waves
    /// @param atom_start First atom index (inclusive)
    /// @param atom_end Last atom index (exclusive)
    /// @param ikb_offset Global projector index of first projector in chunk
    /// @param chunk_nkb Number of projectors in this chunk
    void process_atom_chunk(
        const T* psi,
        T* hpsi,
        int nbands,
        int npw,
        int atom_start,
        int atom_end,
        int ikb_offset,
        int chunk_nkb) const;

    /// @brief Apply nonlocal PP for a range of atoms using chunked buffers
    void add_nonlocal_pp_chunk(
        T* hpsi_in,
        const T* becp_chunk,
        int atom_start,
        int atom_end,
        int ikb_offset,
        int chunk_nkb,
        int m) const;

    // =========== End chunked VNL processing ===========

    mutable int max_npw = 0;

    mutable int npw = 0;

    mutable int npol = 0;

    mutable size_t nkb_m = 0;

    const int* isk = nullptr;

    const pseudopot_cell_vnl* ppcell = nullptr;

    const UnitCell* ucell = nullptr;

    const ModulePW::PW_Basis_K* wfcpw = nullptr;

    mutable T *ps = nullptr;
    mutable T *vkb = nullptr;
    mutable T *becp = nullptr;
    Device* ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    Real * deeq = nullptr;
    T * deeq_nc = nullptr;

    // =========== Chunk buffers for memory-efficient processing ===========
    mutable T* vkb_chunk = nullptr;    ///< Chunk vkb buffer [chunk_nkb × npwx]
    mutable T* becp_chunk = nullptr;   ///< Chunk becp buffer [chunk_nkb × nbands]
    mutable T* ps_chunk = nullptr;     ///< Chunk ps buffer [chunk_nkb × nbands]
    mutable int chunk_buffer_capacity = 0;  ///< Current capacity of chunk buffers (in nkb units)
    mutable int chunk_npw_capacity = 0;     ///< Current npw capacity of vkb_chunk
    mutable int chunk_nbands_capacity = 0;  ///< Current nbands capacity for becp/ps chunks
    // =========== End chunk buffers ===========

    // =========== K-point level caches (computed once per k-point, reused across all chunks) ===========
    /// @brief Ensure k-point caches are valid for current k-point
    /// Computes gk and ylm arrays if not already cached for this k-point
    void ensure_kpoint_caches(int ik, int npw) const;

    /// @brief Invalidate k-point caches (called in destructor or when needed)
    void invalidate_kpoint_caches() const;

    mutable Real* cached_gk = nullptr;      ///< [npw × 3] k+G vectors for current k-point
    mutable Real* cached_ylm = nullptr;     ///< [(lmaxkb+1)² × npw] spherical harmonics for current k-point
    mutable T* cached_sk = nullptr;         ///< [nat × npw] structure factors for ALL atoms (cached at k-point level)
    mutable int cached_ik = -1;             ///< Which k-point is currently cached (-1 = none)
    mutable int cached_npw = 0;             ///< Number of plane waves for cached arrays
    mutable int cached_ylm_size = 0;        ///< Size of ylm: (lmaxkb+1)²
    // =========== End k-point level caches ===========

    // using nonlocal_op = nonlocal_pw_op<Real, Device>;
    using gemv_op = ModuleBase::gemv_op<T, Device>;
    using gemm_op = ModuleBase::gemm_op<T, Device>;
    using nonlocal_op = nonlocal_pw_op<Real, Device>;
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    #ifdef __DSP
    using resmem_complex_op = base_device::memory::resize_memory_op_mt<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op_mt<T, Device>;
    #else
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    #endif
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;

    // Memory operations for Real type (used for cached gk, ylm arrays)
    using resmem_real_op = base_device::memory::resize_memory_op<Real, Device>;
    using delmem_real_op = base_device::memory::delete_memory_op<Real, Device>;

    T one{1, 0};
    T zero{0, 0};
};

} // namespace hamilt

#endif
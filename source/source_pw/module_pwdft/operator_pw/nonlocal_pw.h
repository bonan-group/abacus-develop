#ifndef NONLOCALPW_H
#define NONLOCALPW_H

#include "operator_pw.h"

#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "source_base/kernels/math_kernel_op.h"

#include "source_pw/module_pwdft/VNL_in_pw.h"

#include <cstdlib>  // for getenv
#include <string>

namespace hamilt {

#ifndef NONLOCALTEMPLATE_H
#define NONLOCALTEMPLATE_H

template<class T> class Nonlocal : public T {};
// template<typename Real, typename Device = base_device::DEVICE_CPU>
// class Nonlocal : public OperatorPW<T, Device> {};

#endif

/// @brief Check if chunked VNL processing is enabled via environment variable
/// @return true if ABACUS_VNL_CHUNKED=1, false otherwise
inline bool use_chunked_vnl()
{
    static int cached = -1;
    if (cached < 0)
    {
        const char* env = std::getenv("ABACUS_VNL_CHUNKED");
        cached = (env && std::string(env) == "1") ? 1 : 0;
    }
    return cached == 1;
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

    T one{1, 0};
    T zero{0, 0};
};

} // namespace hamilt

#endif
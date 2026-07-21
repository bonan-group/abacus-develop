#ifndef NONLOCALPW_H
#define NONLOCALPW_H

#include "op_pw.h"

#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/kernels/nonlocal_op.h"
#include "source_base/kernels/math_kernel_op.h"

#include "source_pw/module_pwdft/vnl_pw.h"

#include <type_traits>

namespace hamilt {

#ifndef NONLOCALTEMPLATE_H
#define NONLOCALTEMPLATE_H

template<class T> class Nonlocal : public T {};
// template<typename Real, typename Device = base_device::DEVICE_CPU>
// class Nonlocal : public OperatorPW<T, Device> {};

#endif

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

    void apply_uspp_overlap(const T* psi, T* spsi, int nrow, int npw, int nbands) const;

  private:
    void add_nonlocal_pp(T* hpsi_in, int nbands) const;
    void build_nonlocal_coefficients(int nbands) const;

    void act_chunked(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T* tmpsi_in,
                     T* tmhpsi,
                     const int ngk_ik,
                     const bool is_first_node) const;

    int calculate_optimal_chunk_size(int nkb) const;

    void ensure_chunk_buffer(int chunk_nkb, int npw) const;
    void materialize_atom_chunk(int npw, int atom_start, int atom_end, int chunk_nkb) const;

    void ensure_kpoint_caches(int ik, int npw) const;
    void invalidate_kpoint_caches() const;
    void ensure_type_metadata_cache() const;

    void invalidate_becp_cache() const;
    void cache_becp_for(const T* psi, int nrow, int nbands) const;
    bool becp_cache_matches(const T* psi, int nrow, int nbands) const;
    void ensure_becp_capacity(int nbands) const;
    void ensure_ps_capacity(int nbands) const;
    void compute_overlap_becp(const T* psi, int nrow, int nbands, std::false_type) const;
    void compute_overlap_becp(const T* psi, int nrow, int nbands, std::true_type) const;
    void add_overlap_from_projectors(T* spsi, int nrow, int nbands, std::false_type) const;
    void add_overlap_from_projectors(T* spsi, int nrow, int nbands, std::true_type) const;
    void add_overlap_chunked(T* spsi, int nrow, int nbands) const;
    void build_overlap_coefficients(int nbands, bool projector_major) const;

    mutable int max_npw = 0;

    mutable int npw = 0;

    mutable int npol = 0;

    mutable size_t ps_capacity = 0;
    mutable size_t becp_capacity = 0;
    mutable const T* becp_psi = nullptr;
    mutable int becp_ik = -1;
    mutable int becp_npw = 0;
    mutable int becp_nrow = 0;
    mutable int becp_nbands = 0;

    const int* isk = nullptr;

    const pseudopot_cell_vnl* ppcell = nullptr;

    const UnitCell* ucell = nullptr;

    const ModulePW::PW_Basis_K* wfcpw = nullptr;

    mutable T *ps = nullptr;
    mutable T *vkb = nullptr;
    mutable T *becp = nullptr;
    mutable T* vkb_chunk = nullptr;
    mutable bool full_vkb_ready = false;
    mutable int full_vkb_ready_ik = -1;
    mutable int chunk_buffer_capacity = 0;
    mutable int chunk_npw_capacity = 0;

    mutable Real* cached_gk = nullptr;
    mutable Real* cached_ylm = nullptr;
    mutable Real* cached_vkb1 = nullptr;
    mutable T* cached_sk = nullptr;
    mutable int* cached_atom_nh = nullptr;
    mutable int* cached_atom_nb = nullptr;
    mutable int* cached_iat2it = nullptr;
    mutable int* cached_jkb_to_iat = nullptr;
    mutable int* cached_jkb_to_it = nullptr;
    mutable int* cached_jkb_to_ih = nullptr;
    mutable Real* cached_jkb_pref_sign = nullptr;
    mutable int cached_ik = -1;
    mutable int cached_npw = 0;
    mutable int cached_ylm_size = 0;
    mutable int cached_vkb1_ntype = 0;
    mutable int cached_vkb1_nhm = 0;
    mutable int cached_metadata_ntype = 0;
    mutable int cached_metadata_nat = 0;
    mutable int cached_metadata_nkb = 0;

    Device* ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    Real * deeq = nullptr;
    T * deeq_nc = nullptr;
    // using nonlocal_op = nonlocal_pw_op<Real, Device>;
    using gemv_op = ModuleBase::gemv_op<T, Device>;
    using gemm_op = ModuleBase::gemm_op<T, Device>;
    using nonlocal_op = nonlocal_pw_op<Real, Device>;
#ifdef __DSP
    using setmem_complex_op = base_device::memory::set_memory_op_mt<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op_mt<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op_mt<T, Device>;
#else
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
#endif
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<T, Device, base_device::DEVICE_CPU>;
    using resmem_real_op = base_device::memory::resize_memory_op<Real, Device>;
    using delmem_real_op = base_device::memory::delete_memory_op<Real, Device>;
    using resmem_int_op = base_device::memory::resize_memory_op<int, Device>;
    using delmem_int_op = base_device::memory::delete_memory_op<int, Device>;

    T one{1, 0};
    T zero{0, 0};
};

} // namespace hamilt

#endif

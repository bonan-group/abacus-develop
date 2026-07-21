#include "op_pw_nl.h"

#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_base/parallel_reduce.h"
#include "source_base/parallel_comm.h"
#include "source_base/parallel_device.h"
#include "source_base/tool_quit.h"
#include "source_base/math_ylmreal.h"
#include "source_pw/module_pwdft/kernels/vnl_op.h"

#include <cstring>
#include <limits>
#include <vector>

namespace hamilt {

namespace
{

struct ProjectorRange
{
    int atom_begin;
    int atom_end;
    int projector_begin;
    int projector_end;
};

ProjectorRange next_projector_range(const UnitCell& ucell,
                                    const int atom_begin,
                                    const int projector_begin,
                                    const int projector_limit)
{
    ProjectorRange range = {atom_begin, atom_begin, projector_begin, projector_begin};
    const int effective_limit = std::max(1, projector_limit);
    while (range.atom_end < ucell.nat)
    {
        const int it = ucell.iat2it[range.atom_end];
        const int projectors_per_atom = ucell.atoms[it].ncpp.nh;
        if (range.projector_end > range.projector_begin
            && range.projector_end + projectors_per_atom > range.projector_begin + effective_limit)
        {
            break;
        }
        range.projector_end += projectors_per_atom;
        ++range.atom_end;
    }
    return range;
}

} // namespace

template<typename T, typename Device>
Nonlocal<OperatorPW<T, Device>>::Nonlocal(const int* isk_in,
                                               const pseudopot_cell_vnl* ppcell_in,
                                               const UnitCell* ucell_in,
                                               const ModulePW::PW_Basis_K* wfc_basis)
{
    if( isk_in == nullptr || ppcell_in == nullptr || ucell_in == nullptr)
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "Constuctor of Operator::NonlocalPW is failed, please check your code!");
    }
    this->classname = "Nonlocal";
    this->cal_type = calculation_type::pw_nonlocal;
    this->wfcpw = wfc_basis;
    this->isk = isk_in;
    this->ppcell = ppcell_in;
    this->ucell = ucell_in;
    this->deeq = this->ppcell->template get_deeq_data<Real>();
    this->deeq_nc = this->ppcell->template get_deeq_nc_data<Real>();
    this->vkb = this->ppcell->template get_vkb_data<Real>();

}

template<typename T, typename Device>
Nonlocal<OperatorPW<T, Device>>::~Nonlocal() {
    delmem_complex_op()(this->ps);
    delmem_complex_op()(this->becp);
    delmem_complex_op()(this->vkb_chunk);
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    invalidate_kpoint_caches();
#endif
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::init(const int ik_in)
{
    ModuleBase::timer::start("Nonlocal", "getvnl");
    this->ik = ik_in;
    this->invalidate_becp_cache();
    this->full_vkb_ready = false;
    this->full_vkb_ready_ik = -1;
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    invalidate_kpoint_caches();
#endif
    // Calculate nonlocal pseudopotential vkb
	if(this->ppcell->nkb > 0) //xiaohui add 2013-09-02. Attention...
	{
#if defined(__CUDA) || defined(__UT_USE_CUDA)
        if (!this->ppcell->vnl_chunk_policy().should_chunk(this->ppcell->nkb,
                                                           this->wfcpw->npwk_max,
                                                           sizeof(T)))
#endif
        {
            this->ppcell->getvnl(this->ctx, *this->ucell, this->ik, this->vkb);
            this->full_vkb_ready = true;
            this->full_vkb_ready_ik = this->ik;
        }
	}

    if(this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }

    ModuleBase::timer::end("Nonlocal", "getvnl");
}

//--------------------------------------------------------------------------
// this function sum up each non-local pseudopotential located on each atom,
//--------------------------------------------------------------------------
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::build_nonlocal_coefficients(int nbands) const
{
    const int nkb = this->ppcell->nkb;
    this->ensure_ps_capacity(nbands);
    setmem_complex_op()(this->ps, 0, static_cast<size_t>(nkb) * nbands);

    int projector_offset = 0;
    int atom_offset = 0;
    if (this->npol == 1)
    {
        const int current_spin = this->isk[this->ik];
        for (int it = 0; it < this->ucell->ntype; ++it)
        {
            nonlocal_op()(this->ctx,
                          this->ucell->atoms[it].na,
                          nbands,
                          this->ucell->atoms[it].ncpp.nh,
                          projector_offset,
                          atom_offset,
                          current_spin,
                          nkb,
                          this->ppcell->deeq.getBound2(),
                          this->ppcell->deeq.getBound3(),
                          this->ppcell->deeq.getBound4(),
                          this->deeq,
                          this->ps,
                          this->becp);
        }
        return;
    }

    for (int it = 0; it < this->ucell->ntype; ++it)
    {
        nonlocal_op()(this->ctx,
                      this->ucell->atoms[it].na,
                      nbands,
                      this->ucell->atoms[it].ncpp.nh,
                      projector_offset,
                      atom_offset,
                      nkb,
                      this->ppcell->deeq_nc.getBound2(),
                      this->ppcell->deeq_nc.getBound3(),
                      this->ppcell->deeq_nc.getBound4(),
                      this->deeq_nc,
                      this->ps,
                      this->becp);
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_nonlocal_pp(T* hpsi_in, int m) const
{
    ModuleBase::timer::start("Nonlocal", "add_nonlocal_pp");

    const int nkb = this->ppcell->nkb;
    this->build_nonlocal_coefficients(m);

    // use simple method.
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    // qianrui optimize 2021-3-31
    char transa = 'N';
    char transb = 'T';
    if (m == 1)
    {
        int inc = 1;
        // denghui replace 2022-10-20
        // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        gemv_op()(
            transa,
            this->npw,
            this->ppcell->nkb,
            &this->one,
            this->vkb,
            this->ppcell->vkbnc,
            this->ps,
            inc,
            &this->one,
            hpsi_in,
            inc);
    }
    else
    {
        int npm = m;
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // denghui replace 2022-10-20
        #ifdef __DSP
            ModuleBase::gemm_op_mt<T, Device>()
        #else
            gemm_op()
        #endif
            (
            transa,
            transb,
            this->npw,
            npm,
            this->ppcell->nkb,
            &this->one,
            this->vkb,
            this->ppcell->vkbnc,
            this->ps,
            npm,
            &this->one,
            hpsi_in,
            this->max_npw
        );
    }
    ModuleBase::timer::end("Nonlocal", "add_nonlocal_pp");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node)const
{
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    const bool full_vkb_available = sizeof(T) == sizeof(std::complex<float>) ? this->ppcell->has_full_float_vkb
                                                                             : this->ppcell->has_full_double_vkb;
    const bool full_vkb_ready = full_vkb_available && this->full_vkb_ready && this->full_vkb_ready_ik == this->ik;
    if (this->ppcell->nkb > 0
        && (!full_vkb_ready
            || this->ppcell->vnl_chunk_policy().should_chunk(this->ppcell->nkb,
                                                             this->wfcpw->npwk_max,
                                                             sizeof(T))))
    {
        this->act_chunked(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
        return;
    }
#endif

    ModuleBase::timer::start("Operator", "nonlocal_pw");
    if(is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis*nbands/npol);
    }

    this->npw = ngk_ik;
    this->max_npw = nbasis / npol;
    this->npol = npol;

    if (this->ppcell->nkb > 0)
    {
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // qianrui optimize 2021-3-31
        int nkb = this->ppcell->nkb;
        const size_t becp_size = static_cast<size_t>(nbands) * static_cast<size_t>(nkb);
        if (becp_size > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            ModuleBase::WARNING_QUIT("NonlocalPW", "nkb * nbands exceeds the reduction size limit.");
        }
        if (this->becp_capacity < becp_size)
        {
            resmem_complex_op()(this->becp, becp_size, "Nonlocal<PW>::becp");
            this->becp_capacity = becp_size;
        }
        // ModuleBase::ComplexMatrix becp(nbands, nkb, false);
        char transa = 'C';
        char transb = 'N';
        if (nbands == 1)
        {
            int inc = 1;
            // denghui replace 2022-10-20
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            gemv_op()(
                transa,
                this->npw,
                nkb,
                &this->one,
                this->vkb,
                this->ppcell->vkbnc,
                tmpsi_in,
                inc,
                &this->zero,
                this->becp,
                inc);
        }
        else
        {
            int npm = nbands;
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            // denghui replace 2022-10-20
            #ifdef __DSP
            ModuleBase::gemm_op_mt<T, Device>()
            #else
            gemm_op()
            #endif
            (
                transa,
                transb,
                nkb,
                npm,
                this->npw,
                &this->one,
                this->vkb,
                this->ppcell->vkbnc,
                tmpsi_in,
                max_npw,
                &this->zero,
                this->becp,
                nkb
            );
        }

        Parallel_Reduce::reduce_pool(becp, static_cast<int>(becp_size));

        this->cache_becp_for(tmpsi_in, this->max_npw, nbands);

        this->add_nonlocal_pp(tmhpsi, nbands);
    }

    ModuleBase::timer::end("Operator", "nonlocal_pw");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::invalidate_becp_cache() const
{
    this->becp_psi = nullptr;
    this->becp_ik = -1;
    this->becp_npw = 0;
    this->becp_nrow = 0;
    this->becp_nbands = 0;
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::cache_becp_for(const T* psi, int nrow, int nbands) const
{
    this->becp_psi = psi;
    this->becp_ik = this->ik;
    this->becp_npw = this->npw;
    this->becp_nrow = nrow;
    this->becp_nbands = nbands;
}

template<typename T, typename Device>
bool Nonlocal<OperatorPW<T, Device>>::becp_cache_matches(const T* psi, int nrow, int nbands) const
{
    return this->becp != nullptr && this->becp_psi == psi && this->becp_ik == this->ik
           && this->becp_npw == this->npw && this->becp_nrow == nrow && this->becp_nbands == nbands;
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_becp_capacity(int nbands) const
{
    const size_t required = static_cast<size_t>(this->ppcell->nkb) * static_cast<size_t>(nbands);
    if (this->becp_capacity < required)
    {
        delmem_complex_op()(this->becp);
        this->becp = nullptr;
        resmem_complex_op()(this->becp, required, "Nonlocal<PW>::becp");
        this->becp_capacity = required;
        this->invalidate_becp_cache();
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_ps_capacity(int nbands) const
{
    const size_t required = static_cast<size_t>(this->ppcell->nkb) * static_cast<size_t>(nbands);
    if (this->ps_capacity < required)
    {
        delmem_complex_op()(this->ps);
        this->ps = nullptr;
        resmem_complex_op()(this->ps, required, "Nonlocal<PW>::ps");
        this->ps_capacity = required;
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::compute_overlap_becp(const T* psi,
                                                           int nrow,
                                                           int nbands,
                                                           std::false_type) const
{
    this->ensure_becp_capacity(nbands);
    if (nbands == 1)
    {
        const int inc = 1;
        gemv_op()('C',
                  this->npw,
                  this->ppcell->nkb,
                  &this->one,
                  this->vkb,
                  this->ppcell->vkbnc,
                  psi,
                  inc,
                  &this->zero,
                  this->becp,
                  inc);
    }
    else
    {
        gemm_op()('C',
                  'N',
                  this->ppcell->nkb,
                  nbands,
                  this->npw,
                  &this->one,
                  this->vkb,
                  this->ppcell->vkbnc,
                  psi,
                  nrow,
                  &this->zero,
                  this->becp,
                  this->ppcell->nkb);
    }
#ifdef __MPI
    Parallel_Common::reduce_dev<T, Device>(this->becp, this->ppcell->nkb * nbands, POOL_WORLD);
#endif
    this->cache_becp_for(psi, nrow, nbands);
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::compute_overlap_becp(const T* psi,
                                                           int nrow,
                                                           int nbands,
                                                           std::true_type) const
{
    const bool matrix_free = !this->full_vkb_ready
                             || this->ppcell->vnl_chunk_policy().should_chunk(this->ppcell->nkb,
                                                                              this->wfcpw->npwk_max,
                                                                              sizeof(T));
    if (!matrix_free)
    {
        this->compute_overlap_becp(psi, nrow, nbands, std::false_type());
        return;
    }

    this->ensure_kpoint_caches(this->ik, this->npw);
    this->ensure_becp_capacity(nbands);
    hamilt::cal_becp_from_vkb1_cache_op<Real, base_device::DEVICE_GPU>()(
        nullptr,
        this->npw,
        nrow,
        nbands,
        this->ppcell->nkb,
        this->ppcell->nhm,
        this->cached_jkb_to_iat,
        this->cached_jkb_to_it,
        this->cached_jkb_to_ih,
        this->cached_jkb_pref_sign,
        this->cached_vkb1,
        this->cached_sk,
        psi,
        this->becp);
#ifdef __MPI
    Parallel_Common::reduce_dev<T, Device>(this->becp, this->ppcell->nkb * nbands, POOL_WORLD);
#endif
    this->cache_becp_for(psi, nrow, nbands);
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::build_overlap_coefficients(int nbands, bool projector_major) const
{
    this->ensure_ps_capacity(nbands);
    setmem_complex_op()(this->ps, 0, static_cast<size_t>(this->ppcell->nkb) * nbands);
    const Real* qq_nt = this->ppcell->template get_qq_nt_data<Real>();
    for (int it = 0; it < this->ucell->ntype; ++it)
    {
        const Atom& atoms = this->ucell->atoms[it];
        if (!atoms.ncpp.tvanp)
        {
            continue;
        }
        const int first_iat = this->ucell->itia2iat(it, 0);
        uspp_overlap_op<Real, Device>()(this->ctx,
                                        atoms.na,
                                        nbands,
                                        atoms.ncpp.nh,
                                        this->ppcell->nhm,
                                        this->ppcell->nkb,
                                        this->ppcell->indv_ijkb0[first_iat],
                                        projector_major,
                                        qq_nt + it * this->ppcell->nhm * this->ppcell->nhm,
                                        this->ps,
                                        this->becp);
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_overlap_from_projectors(T* spsi,
                                                                  int nrow,
                                                                  int nbands,
                                                                  std::false_type) const
{
    this->build_overlap_coefficients(nbands, false);
    if (nbands == 1)
    {
        const int inc = 1;
        gemv_op()('N',
                  this->npw,
                  this->ppcell->nkb,
                  &this->one,
                  this->vkb,
                  this->ppcell->vkbnc,
                  this->ps,
                  inc,
                  &this->one,
                  spsi,
                  inc);
    }
    else
    {
        gemm_op()('N',
                  'N',
                  this->npw,
                  nbands,
                  this->ppcell->nkb,
                  &this->one,
                  this->vkb,
                  this->ppcell->vkbnc,
                  this->ps,
                  this->ppcell->nkb,
                  &this->one,
                  spsi,
                  nrow);
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_overlap_from_projectors(T* spsi,
                                                                  int nrow,
                                                                  int nbands,
                                                                  std::true_type) const
{
    const bool matrix_free = !this->full_vkb_ready
                             || this->ppcell->vnl_chunk_policy().should_chunk(this->ppcell->nkb,
                                                                              this->wfcpw->npwk_max,
                                                                              sizeof(T));
    if (!matrix_free)
    {
        this->add_overlap_from_projectors(spsi, nrow, nbands, std::false_type());
        return;
    }

    this->add_overlap_chunked(spsi, nrow, nbands);
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::apply_uspp_overlap(const T* psi,
                                                         T* spsi,
                                                         int nrow,
                                                         int npw,
                                                         int nbands) const
{
    bool has_uspp = false;
    for (int it = 0; it < this->ucell->ntype; ++it)
    {
        has_uspp = has_uspp || this->ucell->atoms[it].ncpp.tvanp;
    }
    if (!has_uspp || this->ppcell->nkb <= 0 || nbands <= 0)
    {
        return;
    }
    if (PARAM.inp.noncolin)
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "noncollinear USPP overlap is not implemented");
    }
    if (npw <= 0 || nrow < npw)
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "invalid plane-wave dimensions for USPP overlap");
    }
    if (this->npw != npw || this->max_npw != nrow)
    {
        this->npw = npw;
        this->max_npw = nrow;
        this->invalidate_becp_cache();
    }

#if defined(__CUDA) || defined(__UT_USE_CUDA)
    typedef typename std::is_same<Device, base_device::DEVICE_GPU>::type gpu_tag;
#else
    typedef std::false_type gpu_tag;
#endif
    if (!this->becp_cache_matches(psi, nrow, nbands))
    {
        this->compute_overlap_becp(psi, nrow, nbands, gpu_tag());
    }
    this->add_overlap_from_projectors(spsi, nrow, nbands, gpu_tag());
    this->invalidate_becp_cache();
}

#if defined(__CUDA) || defined(__UT_USE_CUDA)
template<typename T, typename Device>
int Nonlocal<OperatorPW<T, Device>>::calculate_optimal_chunk_size(const int nkb) const
{
    return std::max(1, std::min(this->ppcell->vnl_chunk_policy().projector_limit, nkb));
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_chunk_buffer(int chunk_nkb, int npw) const
{
    const bool need_realloc = chunk_nkb > this->chunk_buffer_capacity || npw > this->chunk_npw_capacity;
    if (!need_realloc)
    {
        return;
    }

    delmem_complex_op()(this->vkb_chunk);
    this->vkb_chunk = nullptr;

    this->chunk_buffer_capacity = std::max(chunk_nkb, this->chunk_buffer_capacity);
    this->chunk_npw_capacity = std::max(npw, this->chunk_npw_capacity);

    resmem_complex_op()(this->vkb_chunk,
                        static_cast<size_t>(this->chunk_buffer_capacity) * static_cast<size_t>(this->wfcpw->npwk_max),
                        "Nonlocal<PW>::vkb_chunk");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::materialize_atom_chunk(int npw,
                                                             int atom_start,
                                                             int atom_end,
                                                             int chunk_nkb) const
{
    this->ensure_chunk_buffer(chunk_nkb, npw);
    hamilt::cal_vnl_from_vkb1_cache_op<Real, Device>()(this->ctx,
                                                       npw,
                                                       this->wfcpw->npwk_max,
                                                       this->ppcell->nhm,
                                                       this->cached_atom_nh,
                                                       atom_start,
                                                       atom_end,
                                                       this->ppcell->template get_nhtol_data<Real>(),
                                                       this->cached_vkb1,
                                                       this->cached_sk,
                                                       this->cached_iat2it,
                                                       this->vkb_chunk);
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act_chunked(const int nbands,
                                                  const int nbasis,
                                                  const int npol,
                                                  const T* tmpsi_in,
                                                  T* tmhpsi,
                                                  const int ngk_ik,
                                                  const bool is_first_node) const
{
    ModuleBase::timer::start("Operator", "nonlocal_pw_chunked");
    this->invalidate_becp_cache();
    if (is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis * nbands / npol);
    }

    if (nbands <= 0 || ngk_ik <= 0 || this->ppcell->nkb <= 0)
    {
        ModuleBase::timer::end("Operator", "nonlocal_pw_chunked");
        return;
    }

    this->npw = ngk_ik;
    this->max_npw = nbasis / npol;
    this->npol = npol;

    ensure_kpoint_caches(this->ik, ngk_ik);

    const int nkb = this->ppcell->nkb;
    const size_t coeff_size = static_cast<size_t>(nkb) * static_cast<size_t>(nbands);
    if (coeff_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "nkb * nbands exceeds the reduction size limit.");
    }
    this->ensure_becp_capacity(nbands);
    const int target_chunk = calculate_optimal_chunk_size(nkb);

    int atom_begin = 0;
    int ikb_begin = 0;
    while (ikb_begin < nkb)
    {
        const ProjectorRange range = next_projector_range(*this->ucell, atom_begin, ikb_begin, target_chunk);
        const int chunk_nkb = range.projector_end - range.projector_begin;
        this->materialize_atom_chunk(ngk_ik, range.atom_begin, range.atom_end, chunk_nkb);
        if (nbands == 1)
        {
            const int inc = 1;
            gemv_op()('C',
                      ngk_ik,
                      chunk_nkb,
                      &this->one,
                      this->vkb_chunk,
                      this->wfcpw->npwk_max,
                      tmpsi_in,
                      inc,
                      &this->zero,
                      this->becp + ikb_begin,
                      inc);
        }
        else
        {
            gemm_op()('C',
                      'N',
                      chunk_nkb,
                      nbands,
                      ngk_ik,
                      &this->one,
                      this->vkb_chunk,
                      this->wfcpw->npwk_max,
                      tmpsi_in,
                      this->max_npw,
                      &this->zero,
                      this->becp + ikb_begin,
                      nkb);
        }

        atom_begin = range.atom_end;
        ikb_begin = range.projector_end;
    }

    Parallel_Reduce::reduce_pool(this->becp, static_cast<int>(coeff_size));
    this->cache_becp_for(tmpsi_in, this->max_npw, nbands);
    this->build_nonlocal_coefficients(nbands);

    atom_begin = 0;
    ikb_begin = 0;
    while (ikb_begin < nkb)
    {
        const ProjectorRange range = next_projector_range(*this->ucell, atom_begin, ikb_begin, target_chunk);
        const int chunk_nkb = range.projector_end - range.projector_begin;
        this->materialize_atom_chunk(ngk_ik, range.atom_begin, range.atom_end, chunk_nkb);
        if (nbands == 1)
        {
            const int inc = 1;
            gemv_op()('N',
                      ngk_ik,
                      chunk_nkb,
                      &this->one,
                      this->vkb_chunk,
                      this->wfcpw->npwk_max,
                      this->ps + ikb_begin,
                      inc,
                      &this->one,
                      tmhpsi,
                      inc);
        }
        else
        {
            gemm_op()('N',
                      'T',
                      ngk_ik,
                      nbands,
                      chunk_nkb,
                      &this->one,
                      this->vkb_chunk,
                      this->wfcpw->npwk_max,
                      this->ps + static_cast<size_t>(ikb_begin) * nbands,
                      nbands,
                      &this->one,
                      tmhpsi,
                      this->max_npw);
        }

        atom_begin = range.atom_end;
        ikb_begin = range.projector_end;
    }

    ModuleBase::timer::end("Operator", "nonlocal_pw_chunked");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_overlap_chunked(T* spsi, int nrow, int nbands) const
{
    this->build_overlap_coefficients(nbands, false);
    this->ensure_kpoint_caches(this->ik, this->npw);

    const int nkb = this->ppcell->nkb;
    const int target_chunk = this->calculate_optimal_chunk_size(nkb);
    int atom_begin = 0;
    int ikb_begin = 0;
    while (ikb_begin < nkb)
    {
        const ProjectorRange range = next_projector_range(*this->ucell, atom_begin, ikb_begin, target_chunk);
        const int chunk_nkb = range.projector_end - range.projector_begin;
        this->materialize_atom_chunk(this->npw, range.atom_begin, range.atom_end, chunk_nkb);

        if (nbands == 1)
        {
            const int inc = 1;
            gemv_op()('N',
                      this->npw,
                      chunk_nkb,
                      &this->one,
                      this->vkb_chunk,
                      this->wfcpw->npwk_max,
                      this->ps + ikb_begin,
                      inc,
                      &this->one,
                      spsi,
                      inc);
        }
        else
        {
            gemm_op()('N',
                      'N',
                      this->npw,
                      nbands,
                      chunk_nkb,
                      &this->one,
                      this->vkb_chunk,
                      this->wfcpw->npwk_max,
                      this->ps + ikb_begin,
                      nkb,
                      &this->one,
                      spsi,
                      nrow);
        }

        atom_begin = range.atom_end;
        ikb_begin = range.projector_end;
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::invalidate_kpoint_caches() const
{
    delmem_real_op()(this->cached_gk);
    delmem_real_op()(this->cached_ylm);
    delmem_real_op()(this->cached_vkb1);
    delmem_complex_op()(this->cached_sk);
    delmem_int_op()(this->cached_atom_nh);
    delmem_int_op()(this->cached_atom_nb);
    delmem_int_op()(this->cached_iat2it);
    delmem_int_op()(this->cached_jkb_to_iat);
    delmem_int_op()(this->cached_jkb_to_it);
    delmem_int_op()(this->cached_jkb_to_ih);
    delmem_real_op()(this->cached_jkb_pref_sign);
    this->cached_gk = nullptr;
    this->cached_ylm = nullptr;
    this->cached_vkb1 = nullptr;
    this->cached_sk = nullptr;
    this->cached_atom_nh = nullptr;
    this->cached_atom_nb = nullptr;
    this->cached_iat2it = nullptr;
    this->cached_jkb_to_iat = nullptr;
    this->cached_jkb_to_it = nullptr;
    this->cached_jkb_to_ih = nullptr;
    this->cached_jkb_pref_sign = nullptr;
    this->cached_ik = -1;
    this->cached_npw = 0;
    this->cached_ylm_size = 0;
    this->cached_vkb1_ntype = 0;
    this->cached_vkb1_nhm = 0;
    this->cached_structure_generation = 0;
    this->cached_metadata_ntype = 0;
    this->cached_metadata_nat = 0;
    this->cached_metadata_nkb = 0;
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_kpoint_caches(int ik, int npw) const
{
    const int x1 = (this->ppcell->lmaxkb + 1) * (this->ppcell->lmaxkb + 1);
    if (this->cached_ik == ik && this->cached_npw == npw && this->cached_ylm_size == x1
        && this->cached_vkb1_ntype == this->ucell->ntype && this->cached_vkb1_nhm == this->ppcell->nhm
        && this->cached_structure_generation == this->ppcell->structure_generation_)
    {
        return;
    }

    delmem_real_op()(this->cached_gk);
    delmem_real_op()(this->cached_ylm);
    delmem_real_op()(this->cached_vkb1);
    delmem_complex_op()(this->cached_sk);
    this->cached_gk = nullptr;
    this->cached_ylm = nullptr;
    this->cached_vkb1 = nullptr;
    this->cached_sk = nullptr;

    using castmem_real_h2d_op = base_device::memory::cast_memory_op<Real, double, Device, base_device::DEVICE_CPU>;
    using castmem_real_h2h_op
        = base_device::memory::cast_memory_op<Real, double, base_device::DEVICE_CPU, base_device::DEVICE_CPU>;

    resmem_real_op()(this->cached_gk, static_cast<size_t>(npw) * 3, "Nonlocal<PW>::cached_gk");
    resmem_real_op()(this->cached_ylm, static_cast<size_t>(x1) * static_cast<size_t>(npw), "Nonlocal<PW>::cached_ylm");
    resmem_real_op()(this->cached_vkb1,
                     static_cast<size_t>(this->ucell->ntype) * static_cast<size_t>(this->ppcell->nhm)
                         * static_cast<size_t>(npw),
                     "Nonlocal<PW>::cached_vkb1");
    resmem_complex_op()(this->cached_sk,
                        static_cast<size_t>(this->ucell->nat) * static_cast<size_t>(npw),
                        "Nonlocal<PW>::cached_sk");

    ModuleBase::Vector3<double>* gk_host = new ModuleBase::Vector3<double>[npw];
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 4096 / sizeof(Real))
#endif
    for (int ig = 0; ig < npw; ++ig)
    {
        gk_host[ig] = this->wfcpw->getgpluskcar(ik, ig);
    }

    if (std::is_same<Device, base_device::DEVICE_GPU>::value)
    {
        castmem_real_h2d_op()(this->cached_gk, reinterpret_cast<double*>(gk_host), npw * 3);
    }
    else if (std::is_same<Real, float>::value)
    {
        castmem_real_h2h_op()(this->cached_gk, reinterpret_cast<double*>(gk_host), npw * 3);
    }
    else
    {
        std::memcpy(this->cached_gk, gk_host, npw * 3 * sizeof(double));
    }
    delete[] gk_host;

    ModuleBase::YlmReal::Ylm_Real(this->ctx, x1, npw, this->cached_gk, this->cached_ylm);
    ensure_type_metadata_cache();
    hamilt::cal_vkb1_cache_op<Real, Device>()(this->ctx,
                                              this->ucell->ntype,
                                              npw,
                                              this->ppcell->nhm,
                                              this->ppcell->tab.getBound2(),
                                              this->ppcell->tab.getBound3(),
                                              this->cached_atom_nb,
                                              this->cached_atom_nh,
                                              static_cast<Real>(PARAM.globalv.dq),
                                              static_cast<Real>(this->ucell->tpiba),
                                              this->cached_gk,
                                              this->cached_ylm,
                                              this->ppcell->template get_indv_data<Real>(),
                                              this->ppcell->template get_nhtolm_data<Real>(),
                                              this->ppcell->template get_tab_data<Real>(),
                                              this->cached_vkb1);
    this->ppcell->psf->get_sk(this->ctx, ik, this->wfcpw, this->cached_sk);

    this->cached_ik = ik;
    this->cached_npw = npw;
    this->cached_ylm_size = x1;
    this->cached_vkb1_ntype = this->ucell->ntype;
    this->cached_vkb1_nhm = this->ppcell->nhm;
    this->cached_structure_generation = this->ppcell->structure_generation_;
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_type_metadata_cache() const
{
    if (this->cached_metadata_ntype == this->ucell->ntype && this->cached_metadata_nat == this->ucell->nat
        && this->cached_metadata_nkb == this->ppcell->nkb && this->cached_atom_nh != nullptr
        && this->cached_atom_nb != nullptr && this->cached_iat2it != nullptr && this->cached_jkb_to_iat != nullptr
        && this->cached_jkb_to_it != nullptr && this->cached_jkb_to_ih != nullptr
        && this->cached_jkb_pref_sign != nullptr)
    {
        return;
    }

    delmem_int_op()(this->cached_atom_nh);
    delmem_int_op()(this->cached_atom_nb);
    delmem_int_op()(this->cached_iat2it);
    delmem_int_op()(this->cached_jkb_to_iat);
    delmem_int_op()(this->cached_jkb_to_it);
    delmem_int_op()(this->cached_jkb_to_ih);
    delmem_real_op()(this->cached_jkb_pref_sign);
    this->cached_atom_nh = nullptr;
    this->cached_atom_nb = nullptr;
    this->cached_iat2it = nullptr;
    this->cached_jkb_to_iat = nullptr;
    this->cached_jkb_to_it = nullptr;
    this->cached_jkb_to_ih = nullptr;
    this->cached_jkb_pref_sign = nullptr;

    resmem_int_op()(this->cached_atom_nh, this->ucell->ntype, "Nonlocal<PW>::cached_atom_nh");
    resmem_int_op()(this->cached_atom_nb, this->ucell->ntype, "Nonlocal<PW>::cached_atom_nb");
    resmem_int_op()(this->cached_iat2it, this->ucell->nat, "Nonlocal<PW>::cached_iat2it");
    resmem_int_op()(this->cached_jkb_to_iat, this->ppcell->nkb, "Nonlocal<PW>::cached_jkb_to_iat");
    resmem_int_op()(this->cached_jkb_to_it, this->ppcell->nkb, "Nonlocal<PW>::cached_jkb_to_it");
    resmem_int_op()(this->cached_jkb_to_ih, this->ppcell->nkb, "Nonlocal<PW>::cached_jkb_to_ih");
    resmem_real_op()(this->cached_jkb_pref_sign, 2 * this->ppcell->nkb, "Nonlocal<PW>::cached_jkb_pref_sign");

    using syncmem_int_op = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;
    using syncmem_real_op = base_device::memory::synchronize_memory_op<Real, Device, base_device::DEVICE_CPU>;

    std::vector<int> atom_nh(this->ucell->ntype);
    std::vector<int> atom_nb(this->ucell->ntype);
    for (int it = 0; it < this->ucell->ntype; ++it)
    {
        atom_nh[it] = this->ucell->atoms[it].ncpp.nh;
        atom_nb[it] = this->ucell->atoms[it].ncpp.nbeta;
    }

    std::vector<int> jkb_to_iat(this->ppcell->nkb);
    std::vector<int> jkb_to_it(this->ppcell->nkb);
    std::vector<int> jkb_to_ih(this->ppcell->nkb);
    std::vector<Real> jkb_pref_sign(2 * this->ppcell->nkb);
    int jkb = 0;
    for (int iat = 0; iat < this->ucell->nat; ++iat)
    {
        const int it = this->ucell->iat2it[iat];
        const int nh = this->ucell->atoms[it].ncpp.nh;
        for (int ih = 0; ih < nh; ++ih)
        {
            jkb_to_iat[jkb] = iat;
            jkb_to_it[jkb] = it;
            jkb_to_ih[jkb] = ih;
            const int lmod = static_cast<int>(this->ppcell->nhtol(it, ih)) % 4;
            const Real pref_re[4] = {1, 0, -1, 0};
            const Real pref_im[4] = {0, -1, 0, 1};
            jkb_pref_sign[2 * jkb] = pref_re[lmod];
            jkb_pref_sign[2 * jkb + 1] = pref_im[lmod];
            ++jkb;
        }
    }

    syncmem_int_op()(this->cached_atom_nh, atom_nh.data(), atom_nh.size());
    syncmem_int_op()(this->cached_atom_nb, atom_nb.data(), atom_nb.size());
    syncmem_int_op()(this->cached_iat2it, this->ucell->iat2it, this->ucell->nat);
    syncmem_int_op()(this->cached_jkb_to_iat, jkb_to_iat.data(), jkb_to_iat.size());
    syncmem_int_op()(this->cached_jkb_to_it, jkb_to_it.data(), jkb_to_it.size());
    syncmem_int_op()(this->cached_jkb_to_ih, jkb_to_ih.data(), jkb_to_ih.size());
    syncmem_real_op()(this->cached_jkb_pref_sign, jkb_pref_sign.data(), jkb_pref_sign.size());

    this->cached_metadata_ntype = this->ucell->ntype;
    this->cached_metadata_nat = this->ucell->nat;
    this->cached_metadata_nkb = this->ppcell->nkb;
}
#endif

template<typename T, typename Device>
template<typename T_in, typename Device_in>
hamilt::Nonlocal<OperatorPW<T, Device>>::Nonlocal(const Nonlocal<OperatorPW<T_in, Device_in>> *nonlocal)
{
    this->classname = "Nonlocal";
    this->cal_type = calculation_type::pw_nonlocal;
    this->ik = nonlocal->get_ik();
    this->isk = nonlocal->get_isk();
    this->ppcell = nonlocal->get_ppcell();
    this->ucell = nonlocal->get_ucell();
    this->deeq = this->ppcell->d_deeq;
    this->deeq_nc = this->ppcell->template get_deeq_nc_data<Real>();
    this->vkb = this->ppcell->template get_vkb_data<Real>();
    if( this->isk == nullptr || this->ppcell == nullptr || this->ucell == nullptr)
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "Constuctor of Operator::NonlocalPW is failed, please check your code!");
    }
}

template class Nonlocal<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;
// template Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>> *nonlocal);
#if ((defined __CUDA) || (defined __ROCM))
template class Nonlocal<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
// template Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *nonlocal); template
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>> *nonlocal); template
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>::Nonlocal(const
// Nonlocal<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *nonlocal);
#endif
} // namespace hamilt

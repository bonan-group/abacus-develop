#include "op_pw_nl.h"

#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"
#include "source_base/math_ylmreal.h"
#include "source_pw/module_pwdft/kernels/vnl_op.h"

#include <cstring>
#include <limits>
#include <vector>

namespace hamilt {

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
    delmem_complex_op()(this->becp_chunk);
    delmem_complex_op()(this->ps_chunk);
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    invalidate_kpoint_caches();
#endif
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::init(const int ik_in)
{
    ModuleBase::timer::start("Nonlocal", "getvnl");
    this->ik = ik_in;
    this->full_vkb_ready = false;
    this->full_vkb_ready_ik = -1;
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    invalidate_kpoint_caches();
#endif
    // Calculate nonlocal pseudopotential vkb
	if(this->ppcell->nkb > 0) //xiaohui add 2013-09-02. Attention...
	{
#if defined(__CUDA) || defined(__UT_USE_CUDA)
        if (!use_chunked_vnl<Device>(this->ppcell->nkb, this->wfcpw->npwk_max, sizeof(T)))
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
void Nonlocal<OperatorPW<T, Device>>::add_nonlocal_pp(T *hpsi_in, const T *becp, const int m) const
{
    ModuleBase::timer::start("Nonlocal", "add_nonlocal_pp");

    // number of projectors
    int nkb = this->ppcell->nkb;

    // T *ps = new T[nkb * m];
    // ModuleBase::GlobalFunc::ZEROS(ps, m * nkb);
    const size_t ps_size = static_cast<size_t>(nkb) * static_cast<size_t>(m);
    if (this->ps_capacity < ps_size)
    {
        resmem_complex_op()(this->ps, ps_size, "Nonlocal<PW>::ps");
        this->ps_capacity = ps_size;
    }
    setmem_complex_op()(this->ps, 0, ps_size);

    int sum = 0;
    int iat = 0;
    if (this->npol == 1)
    {
        const int current_spin = this->isk[this->ik];
        for (int it = 0; it < this->ucell->ntype; it++)
        {
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            // denghui replace 2022-10-20
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            nonlocal_op()(
                this->ctx,   // device context
                this->ucell->atoms[it].na, m, nproj, // four loop size
                sum, iat, current_spin, nkb,   // additional index params
                this->ppcell->deeq.getBound2(), this->ppcell->deeq.getBound3(), this->ppcell->deeq.getBound4(), // realArray operator()
                this->deeq, // array of data
                this->ps, this->becp); //  array of data
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            // for (int ia = 0; ia < this->ucell->atoms[it].na; ia++)
            // {
            //     // each atom has nproj, means this is with structure factor;
            //     // each projector (each atom) must multiply coefficient
            //     // with all the other projectors.
            //     for (int ib = 0; ib < m; ++ib)
            //     {
            //         for (int ip2 = 0; ip2 < nproj; ip2++)
            //         {
            //             for (int ip = 0; ip < nproj; ip++)
            //             {
            //                 this->ps[(sum + ip2) * m + ib]
            //                     += this->ppcell->deeq(current_spin, iat, ip, ip2) * this->becp[ib * nkb + sum + ip];
            //             } // end ib
            //         } // end ih
            //     } // end jh
            //     sum += nproj;
            //     ++iat;
            // } // end na
        } // end nt
    }
    else
    {
        for (int it = 0; it < this->ucell->ntype; it++)
        {
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            // added by denghui at 20221109
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            nonlocal_op()(
                this->ctx,   // device context
                this->ucell->atoms[it].na, m, nproj, // four loop size
                sum, iat, nkb,   // additional index params
                this->ppcell->deeq_nc.getBound2(), this->ppcell->deeq_nc.getBound3(), this->ppcell->deeq_nc.getBound4(), // realArray operator()
                this->deeq_nc, // array of data
                this->ps, this->becp); //  array of data
            // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            // for (int ia = 0; ia < this->ucell->atoms[it].na; ia++)
            // {
            //     // each atom has nproj, means this is with structure factor;
            //     // each projector (each atom) must multiply coefficient
            //     // with all the other projectors.
            //     for (int ib = 0; ib < m; ib+=2)
            //     {
            //         for (int ip2 = 0; ip2 < nproj; ip2++)
            //         {
            //             for (int ip = 0; ip < nproj; ip++)
            //             {
            //                 psind = (sum + ip2) * m + ib;
            //                 becpind = ib * nkb + sum + ip;
            //                 becp1 = becp[becpind];
            //                 becp2 = becp[becpind + nkb];
            //                 ps[psind] += this->ppcell->deeq_nc(0, iat, ip2, ip) * becp1
            //                              + this->ppcell->deeq_nc(1, iat, ip2, ip) * becp2;
            //                 ps[psind + 1] += this->ppcell->deeq_nc(2, iat, ip2, ip) * becp1
            //                                  + this->ppcell->deeq_nc(3, iat, ip2, ip) * becp2;
            //             } // end ib
            //         } // end ih
            //     } // end jh
            //     sum += nproj;
            //     ++iat;
            // } // end na
        } // end nt
    }

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
        && (!full_vkb_ready || use_chunked_vnl<Device>(this->ppcell->nkb, this->wfcpw->npwk_max, sizeof(T))))
    {
        if (vnl_matrix_free_enabled() && npol == 1
            && vnl_matrix_free_memory_available(this->ppcell->nkb, nbands, sizeof(T)))
        {
            this->act_matrix_free(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
            return;
        }
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

        this->add_nonlocal_pp(tmhpsi, becp, nbands);
    }

    ModuleBase::timer::end("Operator", "nonlocal_pw");
}

#if defined(__CUDA) || defined(__UT_USE_CUDA)
template<typename T, typename Device>
int Nonlocal<OperatorPW<T, Device>>::calculate_optimal_chunk_size(int npw, int nkb, int nbands) const
{
    int target_chunk = 64;
    const int env_chunk = get_vnl_chunk_size_override();
    if (env_chunk > 0)
    {
        target_chunk = env_chunk;
    }
    return std::max(1, std::min(target_chunk, nkb));
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_chunk_buffers(int chunk_nkb, int npw, int nbands) const
{
    const bool need_realloc = chunk_nkb > this->chunk_buffer_capacity
                              || npw > this->chunk_npw_capacity
                              || nbands > this->chunk_nbands_capacity;
    if (!need_realloc)
    {
        return;
    }

    delmem_complex_op()(this->vkb_chunk);
    delmem_complex_op()(this->becp_chunk);
    delmem_complex_op()(this->ps_chunk);
    this->vkb_chunk = nullptr;
    this->becp_chunk = nullptr;
    this->ps_chunk = nullptr;

    this->chunk_buffer_capacity = std::max(chunk_nkb, this->chunk_buffer_capacity);
    this->chunk_npw_capacity = std::max(npw, this->chunk_npw_capacity);
    this->chunk_nbands_capacity = std::max(nbands, this->chunk_nbands_capacity);

    resmem_complex_op()(this->vkb_chunk,
                        static_cast<size_t>(this->chunk_buffer_capacity) * static_cast<size_t>(this->wfcpw->npwk_max),
                        "Nonlocal<PW>::vkb_chunk");
    resmem_complex_op()(this->becp_chunk,
                        static_cast<size_t>(this->chunk_buffer_capacity)
                            * static_cast<size_t>(this->chunk_nbands_capacity),
                        "Nonlocal<PW>::becp_chunk");
    resmem_complex_op()(this->ps_chunk,
                        static_cast<size_t>(this->chunk_buffer_capacity)
                            * static_cast<size_t>(this->chunk_nbands_capacity),
                        "Nonlocal<PW>::ps_chunk");
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
    const int target_chunk = calculate_optimal_chunk_size(ngk_ik, nkb, nbands);

    int atom_begin = 0;
    int ikb_begin = 0;
    while (ikb_begin < nkb)
    {
        int atom_end = atom_begin;
        int ikb_end = ikb_begin;
        while (atom_end < this->ucell->nat)
        {
            const int it = this->ucell->iat2it[atom_end];
            const int nh = this->ucell->atoms[it].ncpp.nh;
            if (ikb_end > ikb_begin && ikb_end + nh > ikb_begin + target_chunk)
            {
                break;
            }
            ikb_end += nh;
            ++atom_end;
        }

        process_atom_chunk(tmpsi_in,
                           tmhpsi,
                           nbands,
                           ngk_ik,
                           atom_begin,
                           atom_end,
                           ikb_end - ikb_begin);

        atom_begin = atom_end;
        ikb_begin = ikb_end;
    }

    ModuleBase::timer::end("Operator", "nonlocal_pw_chunked");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act_matrix_free(const int nbands,
                                                      const int nbasis,
                                                      const int npol,
                                                      const T* tmpsi_in,
                                                      T* tmhpsi,
                                                      const int ngk_ik,
                                                      const bool is_first_node) const
{
    ModuleBase::timer::start("Operator", "nonlocal_pw_matrix_free");
    if (is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis * nbands / npol);
    }

    if (nbands <= 0 || ngk_ik <= 0 || this->ppcell->nkb <= 0)
    {
        ModuleBase::timer::end("Operator", "nonlocal_pw_matrix_free");
        return;
    }

    this->npw = ngk_ik;
    this->max_npw = nbasis / npol;
    this->npol = npol;

    ensure_kpoint_caches(this->ik, ngk_ik);

    const int nkb = this->ppcell->nkb;
    const size_t coeff_size = static_cast<size_t>(nbands) * static_cast<size_t>(nkb);
    if (coeff_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "nkb * nbands exceeds the reduction size limit.");
    }
    if (this->becp_capacity < coeff_size)
    {
        delmem_complex_op()(this->becp);
        this->becp = nullptr;
        resmem_complex_op()(this->becp, coeff_size, "Nonlocal<PW>::becp");
        this->becp_capacity = coeff_size;
    }
    if (this->ps_capacity < coeff_size)
    {
        delmem_complex_op()(this->ps);
        this->ps = nullptr;
        resmem_complex_op()(this->ps, coeff_size, "Nonlocal<PW>::ps");
        this->ps_capacity = coeff_size;
    }

    hamilt::cal_becp_from_vkb1_cache_op<Real, Device>()(this->ctx,
                                                        this->npw,
                                                        this->max_npw,
                                                        nbands,
                                                        nkb,
                                                        this->ppcell->nhm,
                                                        this->cached_jkb_to_iat,
                                                        this->cached_jkb_to_it,
                                                        this->cached_jkb_to_ih,
                                                        this->cached_jkb_pref_sign,
                                                        this->cached_vkb1,
                                                        this->cached_sk,
                                                        tmpsi_in,
                                                        this->becp);

    Parallel_Reduce::reduce_pool(this->becp, static_cast<int>(coeff_size));

    setmem_complex_op()(this->ps, 0, coeff_size);

    int sum = 0;
    int iat = 0;
    const int current_spin = this->isk[this->ik];
    for (int it = 0; it < this->ucell->ntype; it++)
    {
        const int nproj = this->ucell->atoms[it].ncpp.nh;
        nonlocal_op()(this->ctx,
                      this->ucell->atoms[it].na,
                      nbands,
                      nproj,
                      sum,
                      iat,
                      current_spin,
                      nkb,
                      this->ppcell->deeq.getBound2(),
                      this->ppcell->deeq.getBound3(),
                      this->ppcell->deeq.getBound4(),
                      this->deeq,
                      this->ps,
                      this->becp);
    }

    hamilt::cal_hpsi_from_vkb1_cache_op<Real, Device>()(this->ctx,
                                                        this->npw,
                                                        this->max_npw,
                                                        nbands,
                                                        nkb,
                                                        this->ppcell->nhm,
                                                        this->cached_jkb_to_iat,
                                                        this->cached_jkb_to_it,
                                                        this->cached_jkb_to_ih,
                                                        this->cached_jkb_pref_sign,
                                                        this->cached_vkb1,
                                                        this->cached_sk,
                                                        this->ps,
                                                        tmhpsi);

    ModuleBase::timer::end("Operator", "nonlocal_pw_matrix_free");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::process_atom_chunk(const T* psi,
                                                         T* hpsi,
                                                         int nbands,
                                                         int npw,
                                                         int atom_start,
                                                         int atom_end,
                                                         int chunk_nkb) const
{
    ModuleBase::timer::start("Nonlocal", "process_chunk");
    if (nbands <= 0 || npw <= 0 || chunk_nkb <= 0)
    {
        ModuleBase::timer::end("Nonlocal", "process_chunk");
        return;
    }

    const size_t chunk_coeff_size = static_cast<size_t>(chunk_nkb) * static_cast<size_t>(nbands);
    if (chunk_coeff_size > static_cast<size_t>(std::numeric_limits<int>::max()))
    {
        ModuleBase::WARNING_QUIT("NonlocalPW", "chunk_nkb * nbands exceeds the reduction size limit.");
    }

    ensure_chunk_buffers(chunk_nkb, npw, nbands);

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

    char transa = 'C';
    char transb = 'N';
    if (nbands == 1)
    {
        int inc = 1;
        gemv_op()(transa,
                  npw,
                  chunk_nkb,
                  &this->one,
                  this->vkb_chunk,
                  this->wfcpw->npwk_max,
                  psi,
                  inc,
                  &this->zero,
                  this->becp_chunk,
                  inc);
    }
    else
    {
#ifdef __DSP
        ModuleBase::gemm_op_mt<T, Device>()
#else
        gemm_op()
#endif
            (transa,
             transb,
             chunk_nkb,
             nbands,
             npw,
             &this->one,
             this->vkb_chunk,
             this->wfcpw->npwk_max,
             psi,
             this->max_npw,
             &this->zero,
             this->becp_chunk,
             chunk_nkb);
    }

    Parallel_Reduce::reduce_pool(this->becp_chunk, static_cast<int>(chunk_coeff_size));
    add_nonlocal_pp_chunk(hpsi, this->becp_chunk, atom_start, atom_end, chunk_nkb, nbands);
    ModuleBase::timer::end("Nonlocal", "process_chunk");
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_nonlocal_pp_chunk(T* hpsi_in,
                                                            const T* becp_chunk,
                                                            int atom_start,
                                                            int atom_end,
                                                            int chunk_nkb,
                                                            int m) const
{
    ModuleBase::timer::start("Nonlocal", "add_nonlocal_pp_chunk");
    setmem_complex_op()(this->ps_chunk, 0, chunk_nkb * m);

    int sum = 0;
    if (this->npol == 1)
    {
        const int current_spin = this->isk[this->ik];
        for (int iat = atom_start; iat < atom_end; ++iat)
        {
            const int it = this->ucell->iat2it[iat];
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            int global_iat = iat;
            nonlocal_op()(this->ctx,
                          1,
                          m,
                          nproj,
                          sum,
                          global_iat,
                          current_spin,
                          chunk_nkb,
                          this->ppcell->deeq.getBound2(),
                          this->ppcell->deeq.getBound3(),
                          this->ppcell->deeq.getBound4(),
                          this->deeq,
                          this->ps_chunk,
                          becp_chunk);
        }
    }
    else
    {
        for (int iat = atom_start; iat < atom_end; ++iat)
        {
            const int it = this->ucell->iat2it[iat];
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            int global_iat = iat;
            nonlocal_op()(this->ctx,
                          1,
                          m,
                          nproj,
                          sum,
                          global_iat,
                          chunk_nkb,
                          this->ppcell->deeq_nc.getBound2(),
                          this->ppcell->deeq_nc.getBound3(),
                          this->ppcell->deeq_nc.getBound4(),
                          this->deeq_nc,
                          this->ps_chunk,
                          becp_chunk);
        }
    }

    char transa = 'N';
    char transb = 'T';
    if (m == 1)
    {
        int inc = 1;
        gemv_op()(transa,
                  this->npw,
                  chunk_nkb,
                  &this->one,
                  this->vkb_chunk,
                  this->wfcpw->npwk_max,
                  this->ps_chunk,
                  inc,
                  &this->one,
                  hpsi_in,
                  inc);
    }
    else
    {
#ifdef __DSP
        ModuleBase::gemm_op_mt<T, Device>()
#else
        gemm_op()
#endif
            (transa,
             transb,
             this->npw,
             m,
             chunk_nkb,
             &this->one,
             this->vkb_chunk,
             this->wfcpw->npwk_max,
             this->ps_chunk,
             m,
             &this->one,
             hpsi_in,
             this->max_npw);
    }
    ModuleBase::timer::end("Nonlocal", "add_nonlocal_pp_chunk");
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
    this->cached_metadata_ntype = 0;
    this->cached_metadata_nat = 0;
    this->cached_metadata_nkb = 0;
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_kpoint_caches(int ik, int npw) const
{
    const int x1 = (this->ppcell->lmaxkb + 1) * (this->ppcell->lmaxkb + 1);
    if (this->cached_ik == ik && this->cached_npw == npw && this->cached_ylm_size == x1
        && this->cached_vkb1_ntype == this->ucell->ntype && this->cached_vkb1_nhm == this->ppcell->nhm)
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

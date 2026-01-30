#include "nonlocal_pw.h"

#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_base/parallel_reduce.h"
#include "source_base/tool_quit.h"
#include "source_base/module_device/nvtx_helper.h"


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
    // Clean up chunk buffers
    delmem_complex_op()(this->vkb_chunk);
    delmem_complex_op()(this->becp_chunk);
    delmem_complex_op()(this->ps_chunk);
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::init(const int ik_in)
{
    NVTX_RANGE_PUSH("Nonlocal::init");
    ModuleBase::timer::tick("Nonlocal", "getvnl");
    this->ik = ik_in;
    // Calculate nonlocal pseudopotential vkb
	if(this->ppcell->nkb > 0) //xiaohui add 2013-09-02. Attention...
	{
		NVTX_RANGE_PUSH("getvnl");
		this->ppcell->getvnl(this->ctx, *this->ucell, this->ik, this->vkb);
		NVTX_RANGE_POP();
	}

    if(this->next_op != nullptr)
    {
        this->next_op->init(ik_in);
    }

    ModuleBase::timer::tick("Nonlocal", "getvnl");
    NVTX_RANGE_POP();
}

//--------------------------------------------------------------------------
// this function sum up each non-local pseudopotential located on each atom,
//--------------------------------------------------------------------------
template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_nonlocal_pp(T *hpsi_in, const T *becp, const int m) const
{
    NVTX_RANGE_PUSH("Nonlocal::add_nonlocal_pp");
    ModuleBase::timer::tick("Nonlocal", "add_nonlocal_pp");

    // number of projectors
    int nkb = this->ppcell->nkb;

    // T *ps = new T[nkb * m];
    // ModuleBase::GlobalFunc::ZEROS(ps, m * nkb);
    if (this->nkb_m < m * nkb) {
        resmem_complex_op()(this->ps, nkb * m, "Nonlocal<PW>::ps");
        this->nkb_m = m * nkb;
    }
    setmem_complex_op()(this->ps, 0, nkb * m);

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
            NVTX_RANGE_PUSH("nonlocal_op");
            nonlocal_op()(
                this->ctx,   // device context
                this->ucell->atoms[it].na, m, nproj, // four loop size
                sum, iat, current_spin, nkb,   // additional index params
                this->ppcell->deeq.getBound2(), this->ppcell->deeq.getBound3(), this->ppcell->deeq.getBound4(), // realArray operator()
                this->deeq, // array of data
                this->ps, this->becp); //  array of data
            NVTX_RANGE_POP();
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
            NVTX_RANGE_PUSH("nonlocal_op_npol2");
            nonlocal_op()(
                this->ctx,   // device context
                this->ucell->atoms[it].na, m, nproj, // four loop size
                sum, iat, nkb,   // additional index params
                this->ppcell->deeq_nc.getBound2(), this->ppcell->deeq_nc.getBound3(), this->ppcell->deeq_nc.getBound4(), // realArray operator()
                this->deeq_nc, // array of data
                this->ps, this->becp); //  array of data
            NVTX_RANGE_POP();
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
        NVTX_RANGE_PUSH("gemv");
        gemv_op()(
            transa,
            this->npw,
            this->ppcell->nkb,
            &this->one,
            this->vkb,
            this->ppcell->vkb.nc,
            this->ps,
            inc,
            &this->one,
            hpsi_in,
            inc);
        NVTX_RANGE_POP();
    }
    else
    {
        int npm = m;
        //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        // denghui replace 2022-10-20
        NVTX_RANGE_PUSH("gemm");
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
            this->ppcell->vkb.nc,
            this->ps,
            npm,
            &this->one,
            hpsi_in,
            this->max_npw
        );
        NVTX_RANGE_POP();
    }
    ModuleBase::timer::tick("Nonlocal", "add_nonlocal_pp");
    NVTX_RANGE_POP();
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
    // Check if chunked processing is enabled
    if (use_chunked_vnl() && this->ppcell->nkb > 0)
    {
        this->act_chunked(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
        return;
    }

    NVTX_RANGE_PUSH("Nonlocal::act");
    ModuleBase::timer::tick("Operator", "nonlocal_pw");
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
        if (this->nkb_m < nbands * nkb)
        {
            resmem_complex_op()(this->becp, nbands * nkb, "Nonlocal<PW>::becp");
        }
        // ModuleBase::ComplexMatrix becp(nbands, nkb, false);
        char transa = 'C';
        char transb = 'N';
        if (nbands == 1)
        {
            int inc = 1;
            // denghui replace 2022-10-20
            // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            NVTX_RANGE_PUSH("becp_gemv");
            gemv_op()(
                transa,
                this->npw,
                nkb,
                &this->one,
                this->vkb,
                this->ppcell->vkb.nc,
                tmpsi_in,
                inc,
                &this->zero,
                this->becp,
                inc);
            NVTX_RANGE_POP();
        }
        else
        {
            int npm = nbands;
            //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            // denghui replace 2022-10-20
            NVTX_RANGE_PUSH("becp_gemm");
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
                this->ppcell->vkb.nc,
                tmpsi_in,
                max_npw,
                &this->zero,
                this->becp,
                nkb
            );
            NVTX_RANGE_POP();
        }

        NVTX_RANGE_PUSH("parallel_reduce");
        Parallel_Reduce::reduce_pool(becp, nkb * nbands);
        NVTX_RANGE_POP();

        this->add_nonlocal_pp(tmhpsi, becp, nbands);
    }

    ModuleBase::timer::tick("Operator", "nonlocal_pw");
    NVTX_RANGE_POP();
}

// =========== Chunked VNL processing implementation ===========

template<typename T, typename Device>
int Nonlocal<OperatorPW<T, Device>>::calculate_optimal_chunk_size(int npw, int nkb, int nbands) const
{
    // Default target chunk size: 64 projectors
    // This is a conservative value that works well for most cases
    int target_chunk = 64;

    // Check for environment variable override
    const char* env = std::getenv("ABACUS_VNL_CHUNK_SIZE");
    if (env)
    {
        int env_chunk = std::atoi(env);
        if (env_chunk > 0)
        {
            target_chunk = env_chunk;
        }
    }

    // Clamp to reasonable range [8, nkb]
    target_chunk = std::max(8, std::min(target_chunk, nkb));

    return target_chunk;
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::ensure_chunk_buffers(int chunk_nkb, int npw, int nbands) const
{
    const int npwx = this->wfcpw->npwk_max;

    // Check if we need to reallocate
    bool need_realloc = (chunk_nkb > this->chunk_buffer_capacity)
                     || (npw > this->chunk_npw_capacity)
                     || (nbands > this->chunk_nbands_capacity);

    if (need_realloc)
    {
        // Free old buffers
        delmem_complex_op()(this->vkb_chunk);
        delmem_complex_op()(this->becp_chunk);
        delmem_complex_op()(this->ps_chunk);

        // Allocate new buffers with some headroom
        int new_nkb_cap = std::max(chunk_nkb, this->chunk_buffer_capacity);
        int new_npw_cap = std::max(npw, this->chunk_npw_capacity);
        int new_nbands_cap = std::max(nbands, this->chunk_nbands_capacity);

        resmem_complex_op()(this->vkb_chunk, new_nkb_cap * npwx, "Nonlocal::vkb_chunk");
        resmem_complex_op()(this->becp_chunk, new_nkb_cap * new_nbands_cap, "Nonlocal::becp_chunk");
        resmem_complex_op()(this->ps_chunk, new_nkb_cap * new_nbands_cap, "Nonlocal::ps_chunk");

        this->chunk_buffer_capacity = new_nkb_cap;
        this->chunk_npw_capacity = new_npw_cap;
        this->chunk_nbands_capacity = new_nbands_cap;
    }
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::act_chunked(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node) const
{
    NVTX_RANGE_PUSH("Nonlocal::act_chunked");
    ModuleBase::timer::tick("Operator", "nonlocal_pw_chunked");

    if (is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis * nbands / npol);
    }

    this->npw = ngk_ik;
    this->max_npw = nbasis / npol;
    this->npol = npol;

    const int nkb = this->ppcell->nkb;
    const int nat = this->ucell->nat;

    // Calculate optimal chunk size
    int target_chunk = calculate_optimal_chunk_size(ngk_ik, nkb, nbands);

    // Process atoms in chunks, aligned to atom boundaries
    int iat = 0;  // Current atom index
    int ikb_processed = 0;

    while (ikb_processed < nkb)
    {
        // Find atom range for this chunk
        int chunk_start_iat = iat;
        int chunk_ikb_start = ikb_processed;
        int chunk_ikb_end = chunk_ikb_start;

        // Add atoms until we exceed target_chunk or run out of atoms
        while (iat < nat)
        {
            int it = this->ucell->iat2it[iat];
            int nh = this->ucell->atoms[it].ncpp.nh;

            if (chunk_ikb_end + nh > chunk_ikb_start + target_chunk
                && chunk_ikb_end > chunk_ikb_start)
            {
                break;  // Would exceed chunk size, stop here
            }

            chunk_ikb_end += nh;
            iat++;
        }

        int chunk_nkb = chunk_ikb_end - chunk_ikb_start;
        int chunk_end_iat = iat;

        // Process this chunk
        process_atom_chunk(tmpsi_in, tmhpsi, nbands, ngk_ik,
                           chunk_start_iat, chunk_end_iat,
                           chunk_ikb_start, chunk_nkb);

        ikb_processed = chunk_ikb_end;
    }

    ModuleBase::timer::tick("Operator", "nonlocal_pw_chunked");
    NVTX_RANGE_POP();
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::process_atom_chunk(
    const T* psi,
    T* hpsi,
    int nbands,
    int npw,
    int atom_start,
    int atom_end,
    int ikb_offset,
    int chunk_nkb) const
{
    NVTX_RANGE_PUSH("Nonlocal::process_atom_chunk");

    const int npwx = this->wfcpw->npwk_max;

    // Ensure chunk buffers are large enough
    ensure_chunk_buffers(chunk_nkb, npw, nbands);

    // 1. Compute vkb for atoms [atom_start, atom_end)
    NVTX_RANGE_PUSH("getvnl_atoms");
    this->ppcell->template getvnl_atoms<Real>(this->ctx, *this->ucell, this->ik,
                                               atom_start, atom_end, this->vkb_chunk);
    NVTX_RANGE_POP();

    // 2. GEMM1: becp_chunk = vkb_chunk^H × psi
    char transa = 'C';
    char transb = 'N';
    if (nbands == 1)
    {
        int inc = 1;
        NVTX_RANGE_PUSH("becp_chunk_gemv");
        gemv_op()(
            transa,
            npw,
            chunk_nkb,
            &this->one,
            this->vkb_chunk,
            npwx,
            psi,
            inc,
            &this->zero,
            this->becp_chunk,
            inc);
        NVTX_RANGE_POP();
    }
    else
    {
        NVTX_RANGE_PUSH("becp_chunk_gemm");
        #ifdef __DSP
        ModuleBase::gemm_op_mt<T, Device>()
        #else
        gemm_op()
        #endif
        (
            transa,
            transb,
            chunk_nkb,
            nbands,
            npw,
            &this->one,
            this->vkb_chunk,
            npwx,
            psi,
            this->max_npw,
            &this->zero,
            this->becp_chunk,
            chunk_nkb
        );
        NVTX_RANGE_POP();
    }

    // 3. Parallel reduce becp_chunk
    NVTX_RANGE_PUSH("parallel_reduce_chunk");
    Parallel_Reduce::reduce_pool(this->becp_chunk, chunk_nkb * nbands);
    NVTX_RANGE_POP();

    // 4. Apply deeq and GEMM2: hpsi += vkb_chunk × ps_chunk
    add_nonlocal_pp_chunk(hpsi, this->becp_chunk, atom_start, atom_end,
                          ikb_offset, chunk_nkb, nbands);

    NVTX_RANGE_POP();
}

template<typename T, typename Device>
void Nonlocal<OperatorPW<T, Device>>::add_nonlocal_pp_chunk(
    T* hpsi_in,
    const T* becp_chunk,
    int atom_start,
    int atom_end,
    int ikb_offset,
    int chunk_nkb,
    int m) const
{
    NVTX_RANGE_PUSH("Nonlocal::add_nonlocal_pp_chunk");
    ModuleBase::timer::tick("Nonlocal", "add_nonlocal_pp_chunk");

    // Zero the ps_chunk buffer
    setmem_complex_op()(this->ps_chunk, 0, chunk_nkb * m);

    int sum = 0;  // Local projector index within chunk
    if (this->npol == 1)
    {
        const int current_spin = this->isk[this->ik];

        // Process atoms in the chunk
        for (int iat = atom_start; iat < atom_end; iat++)
        {
            int it = this->ucell->iat2it[iat];
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            int local_iat = iat;  // The kernel expects reference and will modify it

            // Apply deeq for this atom
            // Note: we use the kernel with adjusted indices
            // The becp_chunk is indexed from 0, but deeq uses global atom index
            NVTX_RANGE_PUSH("nonlocal_op_chunk");
            nonlocal_op()(
                this->ctx,
                1,  // na = 1 (processing one atom at a time)
                m,
                nproj,
                sum,  // local sum (projector offset in chunk) - modified by kernel
                local_iat,  // global atom index for deeq lookup - modified by kernel
                current_spin,
                chunk_nkb,  // nkb for this chunk
                this->ppcell->deeq.getBound2(),
                this->ppcell->deeq.getBound3(),
                this->ppcell->deeq.getBound4(),
                this->deeq,
                this->ps_chunk,
                becp_chunk);
            NVTX_RANGE_POP();
            // Note: sum and local_iat are updated by the kernel, no need to do it here
        }
    }
    else
    {
        // Spin-orbit / non-collinear case
        for (int iat = atom_start; iat < atom_end; iat++)
        {
            int it = this->ucell->iat2it[iat];
            const int nproj = this->ucell->atoms[it].ncpp.nh;
            int local_iat = iat;

            NVTX_RANGE_PUSH("nonlocal_op_chunk_npol2");
            nonlocal_op()(
                this->ctx,
                1,  // na = 1
                m,
                nproj,
                sum,
                local_iat,
                chunk_nkb,
                this->ppcell->deeq_nc.getBound2(),
                this->ppcell->deeq_nc.getBound3(),
                this->ppcell->deeq_nc.getBound4(),
                this->deeq_nc,
                this->ps_chunk,
                becp_chunk);
            NVTX_RANGE_POP();
            // Note: sum and local_iat are updated by the kernel
        }
    }

    // GEMM2: hpsi += vkb_chunk × ps_chunk^T
    const int npwx = this->wfcpw->npwk_max;
    char transa = 'N';
    char transb = 'T';

    if (m == 1)
    {
        int inc = 1;
        NVTX_RANGE_PUSH("hpsi_chunk_gemv");
        gemv_op()(
            transa,
            this->npw,
            chunk_nkb,
            &this->one,
            this->vkb_chunk,
            npwx,
            this->ps_chunk,
            inc,
            &this->one,  // accumulate into hpsi
            hpsi_in,
            inc);
        NVTX_RANGE_POP();
    }
    else
    {
        NVTX_RANGE_PUSH("hpsi_chunk_gemm");
        #ifdef __DSP
        ModuleBase::gemm_op_mt<T, Device>()
        #else
        gemm_op()
        #endif
        (
            transa,
            transb,
            this->npw,
            m,
            chunk_nkb,
            &this->one,
            this->vkb_chunk,
            npwx,
            this->ps_chunk,
            m,
            &this->one,  // accumulate into hpsi
            hpsi_in,
            this->max_npw
        );
        NVTX_RANGE_POP();
    }

    ModuleBase::timer::tick("Nonlocal", "add_nonlocal_pp_chunk");
    NVTX_RANGE_POP();
}

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

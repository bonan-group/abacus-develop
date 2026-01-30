#include "veff_pw.h"

#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/module_device/nvtx_helper.h"

namespace hamilt {

template<typename T, typename Device>
Veff<OperatorPW<T, Device>>::Veff(const int* isk_in,
                                       const Real* veff_in,
                                       const int veff_row,
                                       const int veff_col,
                                       const ModulePW::PW_Basis_K* wfcpw_in)
{
    if (isk_in == nullptr || wfcpw_in == nullptr) 
    {
        ModuleBase::WARNING_QUIT("VeffPW", "Constuctor of Operator::VeffPW is failed, please check your code!");
    }

    this->classname = "Veff";
    this->cal_type = calculation_type::pw_veff;
    this->isk = isk_in;
    this->veff = veff_in;
    //note: "veff = nullptr" means that this core does not treat potential but still treats wf. 
    this->veff_row = veff_row;
    this->veff_col = veff_col;
    this->wfcpw = wfcpw_in;
    resmem_complex_op()(this->porter, this->wfcpw->nmaxgr, "Veff<PW>::porter");
    resmem_complex_op()(this->porter1, this->wfcpw->nmaxgr, "Veff<PW>::porter1");

}

template<typename T, typename Device>
Veff<OperatorPW<T, Device>>::~Veff()
{
    delmem_complex_op()(this->porter);
    delmem_complex_op()(this->porter1);
}

template<typename T, typename Device>
void Veff<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node)const
{
    NVTX_RANGE_PUSH("Veff::act");
    ModuleBase::timer::tick("Operator", "veff_pw");
    if(is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis*nbands/npol);
    }
    int max_npw = nbasis / npol;
    const int current_spin = this->isk[this->ik];
    const int psi_offset= max_npw * npol;
#ifdef __DSP
    if (npol == 1)
    {
        ModuleBase::FFT_Guard guard(wfcpw->fft_bundle);
        for (int ib = 0; ib < nbands; ib += npol)
        {
            NVTX_RANGE_PUSH("convolution");
            wfcpw->convolution(this->ctx,
                               this->ik,
                               this->veff_col,
                               tmpsi_in,
                               this->veff + current_spin * this->veff_col,
                               tmhpsi,
                               true);
            NVTX_RANGE_POP();
            tmhpsi   += psi_offset;
            tmpsi_in += psi_offset;
        }
    }else if (npol == 2)
    {
        const Real* current_veff[4]={nullptr};
        for (int is = 0; is < 4; is++)
        {
            current_veff[is] = this->veff + is * this->veff_col;
        }
        for (int ib = 0; ib < nbands; ib += npol)
        {
            NVTX_RANGE_PUSH("recip_to_real_npol2");
            wfcpw->recip_to_real<T, Device>(tmpsi_in, this->porter, this->ik);
            wfcpw->recip_to_real<T, Device>(tmpsi_in + max_npw, this->porter1, this->ik);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("veff_op_npol2");
            veff_op()(this->ctx, this->veff_col, this->porter, this->porter1, current_veff);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("real_to_recip_npol2");
            wfcpw->real_to_recip<T, Device>(this->porter, tmhpsi, this->ik, true);
            wfcpw->real_to_recip<T, Device>(this->porter1, tmhpsi + max_npw, this->ik, true);
            NVTX_RANGE_POP();
            tmhpsi   += psi_offset;
            tmpsi_in += psi_offset;
        }
    }else{
        ModuleBase::WARNING_QUIT("VeffPW", "npol should be 1 or 2 or veff_col equal to 0\n");
    }
#else
    if (npol == 1)
    {
        for (int ib = 0; ib < nbands; ib += npol)
        {
            NVTX_RANGE_PUSH("recip_to_real");
            wfcpw->recip_to_real<T, Device>(tmpsi_in, this->porter, this->ik);
            NVTX_RANGE_POP();
            // NOTICE: when MPI threads are larger than the number of Z grids
            // veff would contain nothing, and nothing should be done in real space
            // but the 3DFFT can not be skipped, it will cause hanging
            NVTX_RANGE_PUSH("veff_op");
            veff_op()(this->ctx, this->veff_col, this->porter, this->veff + current_spin * this->veff_col);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("real_to_recip");
            wfcpw->real_to_recip<T, Device>(this->porter, tmhpsi, this->ik, true);
            NVTX_RANGE_POP();
            tmhpsi   += psi_offset;
            tmpsi_in += psi_offset;
        }
    }
    else if (npol == 2)
    {
        const Real* current_veff[4]={nullptr};
        for (int is = 0; is < 4; is++)
        {
            current_veff[is] = this->veff + is * this->veff_col;
        }
        for (int ib = 0; ib < nbands; ib += npol)
        {
            // FFT to real space and do things.
            NVTX_RANGE_PUSH("recip_to_real_npol2");
            wfcpw->recip_to_real<T, Device>(tmpsi_in, this->porter, this->ik);
            wfcpw->recip_to_real<T, Device>(tmpsi_in + max_npw, this->porter1, this->ik);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("veff_op_npol2");
            veff_op()(this->ctx, this->veff_col, this->porter, this->porter1, current_veff);
            NVTX_RANGE_POP();
            // FFT back to G space.
            NVTX_RANGE_PUSH("real_to_recip_npol2");
            wfcpw->real_to_recip<T, Device>(this->porter, tmhpsi, this->ik, true);
            wfcpw->real_to_recip<T, Device>(this->porter1, tmhpsi + max_npw, this->ik, true);
            NVTX_RANGE_POP();
            tmhpsi   += psi_offset;
            tmpsi_in += psi_offset;
        }
    }else{
        ModuleBase::WARNING_QUIT("VeffPW", "npol should be 1 or 2 or veff_col equal to 0\n");
    }
#endif
    ModuleBase::timer::tick("Operator", "veff_pw");
    NVTX_RANGE_POP();
}

template<typename T, typename Device>
template<typename T_in, typename Device_in>
hamilt::Veff<OperatorPW<T, Device>>::Veff(const Veff<OperatorPW<T_in, Device_in>> *veff) {
    this->classname = "Veff";
    this->cal_type = calculation_type::pw_veff;
    this->ik = veff->get_ik();
    this->isk = veff->get_isk();
    this->veff_col = veff->get_veff_col();
    this->veff_row = veff->get_veff_row();
    this->wfcpw = veff->get_wfcpw();
    resmem_complex_op()(this->porter, this->wfcpw->nmaxgr);
    resmem_complex_op()(this->porter1, this->wfcpw->nmaxgr);
    this->veff = veff->get_veff();
    if (this->isk == nullptr || this->veff == nullptr || this->wfcpw == nullptr) {
        ModuleBase::WARNING_QUIT("VeffPW", "Constuctor of Operator::VeffPW is failed, please check your code!");
    }
}

template class Veff<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class Veff<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;
// template Veff<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Veff(const
// Veff<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>> *veff);
#if ((defined __CUDA) || (defined __ROCM))
template class Veff<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class Veff<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
// template Veff<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>::Veff(const
// Veff<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *veff);
#endif
} // namespace hamilt

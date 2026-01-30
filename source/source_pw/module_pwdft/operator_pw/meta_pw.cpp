#include "meta_pw.h"

#include "source_base/timer.h"
#include "source_pw/module_pwdft/global.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_base/tool_quit.h"
#include "source_base/module_device/nvtx_helper.h"

namespace hamilt {

template<typename T, typename Device>
Meta<OperatorPW<T, Device>>::Meta(Real tpiba_in,
                                       const int* isk_in,
                                       const Real* vk_in,
                                       const int vk_row,
                                       const int vk_col,
                                       const ModulePW::PW_Basis_K* wfcpw_in)
{
    if(isk_in == nullptr || tpiba_in < 1e-10 || wfcpw_in == nullptr)
    {
        ModuleBase::WARNING_QUIT("MetaPW", "Constuctor of Operator::MetaPW is failed, please check your code!");
    }
    this->classname = "Meta";
    this->cal_type = calculation_type::pw_meta;
    this->isk = isk_in;
    this->tpiba = tpiba_in;
    this->vk = vk_in;
    this->vk_row = vk_row;
    this->vk_col = vk_col;
    this->wfcpw = wfcpw_in;
    resmem_complex_op()(this->porter, this->wfcpw->nmaxgr, "Meta<PW>::porter");

}

template<typename T, typename Device>
Meta<OperatorPW<T, Device>>::~Meta()
{
    delmem_complex_op()(this->porter);
}

template<typename T, typename Device>
void Meta<OperatorPW<T, Device>>::act(
    const int nbands,
    const int nbasis,
    const int npol,
    const T* tmpsi_in,
    T* tmhpsi,
    const int ngk_ik,
    const bool is_first_node)const
{
    if (XC_Functional::get_func_type() != 3)
    {
        return;
    }

    NVTX_RANGE_PUSH("Meta::act");
    ModuleBase::timer::tick("Operator", "MetaPW");
    if(is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis*nbands/npol);
    }

    const int current_spin = this->isk[this->ik];
    int max_npw = nbasis / npol;
    //npol == 2 case has not been considered

    for (int ib = 0; ib < nbands; ++ib)
    {
        for (int j = 0; j < 3; j++)
        {
            NVTX_RANGE_PUSH("meta_op_forward");
            meta_op()(this->ctx, this->ik, j, ngk_ik, this->wfcpw->npwk_max, this->tpiba, wfcpw->get_gcar_data<Real>(), wfcpw->get_kvec_c_data<Real>(), tmpsi_in, this->porter);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("recip_to_real");
            wfcpw->recip_to_real(this->ctx, this->porter, this->porter, this->ik);
            NVTX_RANGE_POP();

            if(this->vk_col != 0) {
                NVTX_RANGE_PUSH("vector_mul_vector");
                vector_mul_vector_op()(this->vk_col, this->porter, this->porter, this->vk + current_spin * this->vk_col);
                NVTX_RANGE_POP();
            }

            NVTX_RANGE_PUSH("real_to_recip");
            wfcpw->real_to_recip(this->ctx, this->porter, this->porter, this->ik);
            NVTX_RANGE_POP();
            NVTX_RANGE_PUSH("meta_op_backward");
            meta_op()(this->ctx, this->ik, j, ngk_ik, this->wfcpw->npwk_max, this->tpiba, wfcpw->get_gcar_data<Real>(), wfcpw->get_kvec_c_data<Real>(), this->porter, tmhpsi, true);
            NVTX_RANGE_POP();

        } // x,y,z directions
        tmhpsi += max_npw;
        tmpsi_in += max_npw;
    }
    ModuleBase::timer::tick("Operator", "MetaPW");
    NVTX_RANGE_POP();
}

template<typename T, typename Device>
template<typename T_in, typename Device_in>
Meta<OperatorPW<T, Device>>::Meta(const Meta<OperatorPW<T_in, Device_in>> *meta) {
    this->classname = "Meta";
    this->cal_type = calculation_type::pw_meta;
    this->ik = meta->get_ik();
    this->isk = meta->get_isk();
    this->tpiba = meta->get_tpiba();
    this->vk = meta->get_vk();
    this->vk_row = meta->get_vk_row();
    this->vk_col = meta->get_vk_col();
    this->wfcpw = meta->get_wfcpw();
    if(this->isk == nullptr || this->tpiba < 1e-10 || this->vk == nullptr || this->wfcpw == nullptr)
    {
        ModuleBase::WARNING_QUIT("MetaPW", "Constuctor of Operator::MetaPW is failed, please check your code!");
    }
}

template class Meta<OperatorPW<std::complex<float>, base_device::DEVICE_CPU>>;
template class Meta<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>;
// template Meta<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Meta(const
// Meta<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>> *meta);
#if ((defined __CUDA) || (defined __ROCM))
template class Meta<OperatorPW<std::complex<float>, base_device::DEVICE_GPU>>;
template class Meta<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>;
// template Meta<OperatorPW<std::complex<double>, base_device::DEVICE_CPU>>::Meta(const
// Meta<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *meta); template
// Meta<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>>::Meta(const Meta<OperatorPW<std::complex<double>,
// base_device::DEVICE_CPU>> *meta); template Meta<OperatorPW<std::complex<double>,
// base_device::DEVICE_GPU>>::Meta(const Meta<OperatorPW<std::complex<double>, base_device::DEVICE_GPU>> *meta);
#endif
} // namespace hamilt
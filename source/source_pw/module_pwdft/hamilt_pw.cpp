#include "hamilt_pw.h"

#include "op_pw_ekin.h"
#include "op_pw_exx.h"
#include "op_pw_meta.h"
#include "op_pw_nl.h"
#include "op_pw_proj.h"
#include "op_pw_veff.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_hamilt/module_xc/exx_info.h" // use GlobalC::exx_info
#include "source_io/module_parameter/parameter.h"

namespace hamilt
{

template <typename T, typename Device>
HamiltPW<T, Device>::HamiltPW(elecstate::Potential* pot_in,
                              ModulePW::PW_Basis_K* wfc_basis,
                              K_Vectors* pkv,
                              pseudopot_cell_vnl* nlpp,
                              Plus_U* p_dftu, // mohan add 2025-11-06
                              const UnitCell* ucell)
{
    this->classname = "HamiltPW";
    this->ppcell = nlpp;
    const auto tpiba2 = static_cast<Real>(ucell->tpiba2);
    const auto tpiba = static_cast<Real>(ucell->tpiba);
    const int* isk = pkv->isk.data();
    const Real* gk2 = wfc_basis->get_gk2_data<Real>();

    if (PARAM.inp.t_in_h)
    {
        // Operator<double>* ekinetic = new Ekinetic<OperatorLCAO<double>>
        Operator<T, Device>* ekinetic
            = new Ekinetic<OperatorPW<T, Device>>(tpiba2, gk2, wfc_basis->nks, wfc_basis->npwk_max);
        if (this->ops == nullptr)
        {
            this->ops = ekinetic;
        }
        else
        {
            this->ops->add(ekinetic);
        }
    }
    if (PARAM.inp.vl_in_h)
    {
        std::vector<std::string> pot_register_in;
        if (PARAM.inp.vion_in_h)
        {
            pot_register_in.push_back("local");
        }
        if (PARAM.inp.vh_in_h)
        {
            pot_register_in.push_back("hartree");
        }
        // no variable can choose xc, maybe it is necessary
        pot_register_in.push_back("xc");
        if (PARAM.inp.imp_sol)
        {
            pot_register_in.push_back("surchem");
        }
        if (PARAM.inp.efield_flag)
        {
            pot_register_in.push_back("efield");
        }
        if (PARAM.inp.gate_flag)
        {
            pot_register_in.push_back("gatefield");
        }
        if (PARAM.inp.ml_exx) // sunliang
        {
            pot_register_in.push_back("ml_exx");
        }
        // DFT-1/2
        if (PARAM.inp.dfthalf_type == 1)
        {
            pot_register_in.push_back("dfthalf");
        }
        // only Potential is not empty, Veff and Meta are available
        if (pot_register_in.size() > 0)
        {
            // register Potential by gathered operator
            pot_in->pot_register(pot_register_in);
            Operator<T, Device>* veff = new Veff<OperatorPW<T, Device>>(isk,
                                                                        pot_in->get_veff_smooth_data<Real>(),
                                                                        pot_in->get_veff_smooth_nr(),
                                                                        pot_in->get_veff_smooth_nc(),
                                                                        wfc_basis);
            if (this->ops == nullptr)
            {
                this->ops = veff;
            }
            else
            {
                this->ops->add(veff);
            }
            Operator<T, Device>* meta = new Meta<OperatorPW<T, Device>>(tpiba,
                                                                        isk,
                                                                        pot_in->get_vofk_smooth_data<Real>(),
                                                                        pot_in->get_vofk_smooth_nr(),
                                                                        pot_in->get_vofk_smooth_nc(),
                                                                        wfc_basis);
            this->ops->add(meta);
        }
    }
    if (PARAM.inp.vnl_in_h)
    {
        Operator<T, Device>* nonlocal = new Nonlocal<OperatorPW<T, Device>>(isk, this->ppcell, ucell, wfc_basis);
        this->nonlocal_op = nonlocal;
        this->nonlocal_op_in_chain = true;
        if (this->ops == nullptr)
        {
            this->ops = nonlocal;
        }
        else
        {
            this->ops->add(nonlocal);
        }
    }
    else if (PARAM.globalv.use_uspp)
    {
        this->nonlocal_op = new Nonlocal<OperatorPW<T, Device>>(isk, this->ppcell, ucell, wfc_basis);
    }
    if (PARAM.inp.sc_mag_switch || PARAM.inp.dft_plus_u)
    {
        Operator<T, Device>* onsite_proj = new OnsiteProj<OperatorPW<T, Device>>(isk,
                                                                                 ucell,
                                                                                 p_dftu,
                                                                                 PARAM.inp.sc_mag_switch,
                                                                                 (PARAM.inp.dft_plus_u > 0));
        this->ops->add(onsite_proj);
    }
    if (GlobalC::exx_info.info_global.cal_exx)
    {
        bool separate_loop = GlobalC::exx_info.info_global.separate_loop;
        double hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
        auto coulomb_param = GlobalC::exx_info.info_global.coulomb_param;
        auto exx = new OperatorEXXPW<T, Device>(isk, wfc_basis, pot_in->get_rho_basis(), pkv, ucell, separate_loop, hybrid_alpha, coulomb_param);
        if (this->ops == nullptr)
        {
            this->ops = exx;
        }
        else
        {
            this->ops->add(exx);
            // exx->set_psi(&this->psi);
        }
    }
    return;
}

template <typename T, typename Device>
HamiltPW<T, Device>::~HamiltPW()
{
    if (!this->nonlocal_op_in_chain)
    {
        delete this->nonlocal_op;
    }
    if (this->ops != nullptr)
    {
        delete this->ops;
    }
}

template <typename T, typename Device>
void HamiltPW<T, Device>::updateHk(const int ik)
{
    ModuleBase::TITLE("HamiltPW", "updateHk");
    this->ops->init(ik);
    if (!this->nonlocal_op_in_chain && this->nonlocal_op != nullptr)
    {
        this->nonlocal_op->init(ik);
    }
    ModuleBase::TITLE("HamiltPW", "updateHk");
}

// This routine applies the S matrix to m wavefunctions psi and puts
// the results in spsi.
// Requires the products of psi with all beta functions in array
// becp(nkb,m) (maybe calculated in hPsi).
template <typename T, typename Device>
void HamiltPW<T, Device>::sPsi(const T* psi_in, // psi
                               T* spsi,         // spsi
                               const int nrow,  // dimension of spsi: nbands * nrow
                               const int npw,   // number of plane waves
                               const int nbands // number of bands
) const
{
    ModuleBase::TITLE("HamiltPW", "sPsi");
    syncmem_op()(spsi, psi_in, static_cast<size_t>(nbands * nrow));
    if (this->nonlocal_op != nullptr)
    {
        static_cast<Nonlocal<OperatorPW<T, Device>>*>(this->nonlocal_op)
            ->apply_uspp_overlap(psi_in, spsi, nrow, npw, nbands);
    }
}

template <typename T, typename Device>
void HamiltPW<T, Device>::set_exx_helper(Exx_Helper<T, Device>& exx_helper)
{
    auto op = this->ops;
    while (op != nullptr)
    {
        if (op->get_cal_type() == calculation_type::pw_exx)
        {
            exx_helper.op_exx = reinterpret_cast<OperatorEXXPW<T, Device>*>(op);
            exx_helper.set_op();
        }
        op = op->next_op;
    }
}

template class HamiltPW<std::complex<float>, base_device::DEVICE_CPU>;
template class HamiltPW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class HamiltPW<std::complex<float>, base_device::DEVICE_GPU>;
template class HamiltPW<std::complex<double>, base_device::DEVICE_GPU>;
#endif

} // namespace hamilt

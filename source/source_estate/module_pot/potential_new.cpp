#include "potential_new.h"

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/memory_recorder.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_base/tool_title.h"
#include "source_base/module_device/memory_op.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"
#include "pot_ml_exx.h"

#include <map>

namespace elecstate
{

Potential::Potential(const ModulePW::PW_Basis* rho_basis_in,
                     const ModulePW::PW_Basis* rho_basis_smooth_in,
                     const UnitCell* ucell_in,
                     const ModuleBase::matrix* vloc_in,
                     Structure_Factor* structure_factors_in,
                     surchem* solvent_in,
                     double* etxc_in,
                     double* vtxc_in,
                     VSep* vsep_cell_in)
    : ucell_(ucell_in), vloc_(vloc_in), structure_factors_(structure_factors_in), 
      solvent_(solvent_in), vsep_cell(vsep_cell_in), etxc_(etxc_in),
      vtxc_(vtxc_in)
{
    this->rho_basis_ = rho_basis_in;
    this->rho_basis_smooth_ = rho_basis_smooth_in;
    this->fixed_mode = true;
    this->dynamic_mode = true;
    this->use_gpu_ = (PARAM.inp.basis_type == "pw" && PARAM.inp.device == "gpu");

    // allocate memory for Potential.
    this->allocate();
}

Potential::~Potential()
{
    if (this->components.size() > 0)
    {
        for (auto comp: this->components)
        {
            delete comp;
        }
        this->components.clear();
    }
    this->component_names_.clear();
    if (use_gpu_)
    {
        delmem_sd_op()(s_veff_smooth);
        delmem_sd_op()(s_vofk_smooth);
        delmem_dd_op()(d_veff_smooth);
        delmem_dd_op()(d_vofk_smooth);
    }
    else
    {
        delmem_sh_op()(s_veff_smooth);
        delmem_sh_op()(s_vofk_smooth);
    }
}

void Potential::pot_register(const std::vector<std::string>& components_list)
{
    ModuleBase::TITLE("Potential", "pot_register");
    // delete old components first.
    if (this->components.size() > 0)
    {
        for (auto comp: this->components)
        {
            delete comp;
        }
        this->components.clear();
    }
    this->component_names_.clear();

    // register components
    //---------------------------
    // mapping for register
    //---------------------------
    for (auto comp: components_list)
    {
        PotBase* tmp = this->get_pot_type(comp);
        this->components.push_back(tmp);
        this->component_names_.push_back(comp);
    }

    // after register, reset fixed_done to false
    this->fixed_done = false;

    return;
}

void Potential::allocate()
{
    ModuleBase::TITLE("Potential", "allocate");

    const int nspin = PARAM.inp.nspin;
    assert(nspin==1 || nspin==2 || nspin==4);

    const int nrxx = this->rho_basis_->nrxx;
    const int nrxx_smooth = this->rho_basis_smooth_->nrxx;

    if (nrxx == 0)
	{
		return;
	}
	if (nrxx_smooth == 0)
	{
		return;
	}

    this->v_eff_fixed.resize(nrxx);
    ModuleBase::Memory::record("Pot::veff_fix", sizeof(double) * nrxx);

    this->v_eff.create(nspin, nrxx);
    ModuleBase::Memory::record("Pot::veff", sizeof(double) * nspin * nrxx);

    this->veff_smooth.create(nspin, nrxx_smooth);
    ModuleBase::Memory::record("Pot::veff_smooth", sizeof(double) * nspin * nrxx_smooth);

    if (XC_Functional::get_ked_flag())
    {
        this->vofk_eff.create(nspin, nrxx);
        ModuleBase::Memory::record("Pot::vofk", sizeof(double) * nspin * nrxx);

        this->vofk_smooth.create(nspin, nrxx_smooth);
        ModuleBase::Memory::record("Pot::vofk_smooth", sizeof(double) * nspin * nrxx_smooth);
    }
    if (use_gpu_)
    {
        if (PARAM.globalv.has_float_data)
        {
            resmem_sd_op()(s_veff_smooth, nspin * nrxx_smooth);
            resmem_sd_op()(s_vofk_smooth, nspin * nrxx_smooth);
        }
        if (PARAM.globalv.has_double_data)
        {
            resmem_dd_op()(d_veff_smooth, nspin * nrxx_smooth);
            resmem_dd_op()(d_vofk_smooth, nspin * nrxx_smooth);
        }
        else
        {
            resmem_dd_op()(d_veff_smooth, nspin * nrxx_smooth);
        }
    }
    else
    {
        if (PARAM.globalv.has_float_data)
        {
            resmem_sh_op()(s_veff_smooth, nspin * nrxx_smooth, "POT::sveff_smooth");
            resmem_sh_op()(s_vofk_smooth, nspin * nrxx_smooth, "POT::svofk_smooth");
        }
        if (PARAM.globalv.has_double_data)
        {
            this->d_veff_smooth = this->veff_smooth.c;
            this->d_vofk_smooth = this->vofk_smooth.c;
        }
        // There's no need to allocate memory for double precision pointers while in a CPU environment
    }
}

void Potential::update_from_charge(const Charge*const chg, const UnitCell*const ucell)
{
    ModuleBase::TITLE("Potential", "update_from_charge");
    //ModuleBase::timer::start("Potential", "update_from_charge");

    if (!this->fixed_done)
    {
        this->cal_fixed_v(this->v_eff_fixed.data());
        this->fixed_done = true;
        this->v_eff_host_stale_ = false;
    }

    if (this->update_from_charge_resident_gpu(chg, ucell))
    {
        return;
    }

    this->cal_v_eff(chg, ucell, this->v_eff);

    // interpolate potential on the smooth mesh if necessary
    this->interpolate_vrs();

    if (this->use_gpu_)
    {
        if (PARAM.globalv.has_float_data)
        {
            castmem_d2s_h2d_op()(s_veff_smooth, this->veff_smooth.c, this->veff_smooth.nr * this->veff_smooth.nc);
            if (this->vofk_smooth.nc > 0)
            {
                castmem_d2s_h2d_op()(s_vofk_smooth, this->vofk_smooth.c, this->vofk_smooth.nr * this->vofk_smooth.nc);
            }
        }
        if (PARAM.globalv.has_double_data)
        {
            syncmem_d2d_h2d_op()(d_veff_smooth, this->veff_smooth.c, this->veff_smooth.nr * this->veff_smooth.nc);
            if (this->vofk_smooth.nc > 0)
            {
                syncmem_d2d_h2d_op()(d_vofk_smooth, this->vofk_smooth.c, this->vofk_smooth.nr * this->vofk_smooth.nc);
            }
        }
    }
    else
    {
        if (PARAM.globalv.has_float_data)
        {
            castmem_d2s_h2h_op()(s_veff_smooth, this->veff_smooth.c, this->veff_smooth.nr * this->veff_smooth.nc);
            if (this->vofk_smooth.nc > 0)
            {
                castmem_d2s_h2h_op()(s_vofk_smooth, this->vofk_smooth.c, this->vofk_smooth.nr * this->vofk_smooth.nc);
            }
        }
        // There's no need to synchronize memory for double precision pointers while in a CPU environment
    }

    //ModuleBase::timer::end("Potential", "update_from_charge");
}

bool Potential::supports_resident_gpu_update() const
{
    if (!this->use_gpu_ || PARAM.globalv.double_grid || XC_Functional::get_ked_flag()
        || this->rho_basis_ == nullptr || this->rho_basis_smooth_ == nullptr
        || this->rho_basis_->nrxx != this->rho_basis_smooth_->nrxx)
    {
        return false;
    }

    bool has_xc = false;
    for (const std::string& name : this->component_names_)
    {
        if (name == "xc")
        {
            has_xc = true;
        }
        else if (name != "local" && name != "hartree")
        {
            return false;
        }
    }
    return has_xc;
}

bool Potential::update_from_charge_resident_gpu(const Charge*const chg, const UnitCell*const ucell)
{
#if __CUDA || __UT_USE_CUDA
    if (!this->supports_resident_gpu_update() || chg == nullptr || ucell == nullptr || this->d_veff_smooth == nullptr)
    {
        return false;
    }

    const int nspin = this->v_eff.nr;
    const int nrxx = this->v_eff.nc;
    if (!(nspin == 1 || nspin == 2) || nrxx <= 0)
    {
        return false;
    }

    ModuleBase::TITLE("Potential", "update_resident_gpu");
    ModuleBase::timer::start("Potential", "update_resident_gpu");

    this->v_eff.zero_out();
    this->v_eff_host_stale_ = false;
    for (int is = 0; is < nspin; ++is)
    {
        if (is == 0 || nspin == 2)
        {
            ModuleBase::GlobalFunc::COPYARRAY(this->v_eff_fixed.data(), &(this->v_eff(is, 0)), nrxx);
        }
    }
    for (size_t i = 0; i < this->components.size(); ++i)
    {
        if (this->component_names_[i] != "xc" && this->components[i]->dynamic_mode)
        {
            this->components[i]->cal_v_eff(chg, ucell, this->v_eff);
        }
    }
    this->interpolate_vrs();

    syncmem_d2d_h2d_op()(this->d_veff_smooth, this->veff_smooth.c, nspin * nrxx);

    double etxc = 0.0;
    double vtxc = 0.0;
    const bool used_resident_xc = XC_Functional::add_v_xc_to_device(nrxx, chg, ucell, "gpu", this->d_veff_smooth, etxc, vtxc);
    if (!used_resident_xc)
    {
        ModuleBase::timer::end("Potential", "update_resident_gpu");
        return false;
    }

    *(this->etxc_) = etxc;
    *(this->vtxc_) = vtxc;
    this->v_eff_host_stale_ = true;

    if (PARAM.globalv.has_float_data)
    {
        using castmem_d2s_d2d_op
            = base_device::memory::cast_memory_op<float, double, base_device::DEVICE_GPU, base_device::DEVICE_GPU>;
        castmem_d2s_d2d_op()(this->s_veff_smooth, this->d_veff_smooth, nspin * nrxx);
    }

    ModuleBase::timer::end("Potential", "update_resident_gpu");
    return true;
#else
    return false;
#endif
}

void Potential::materialize_eff_v_host() const
{
    if (!this->v_eff_host_stale_ || this->d_veff_smooth == nullptr || this->v_eff.nc == 0)
    {
        return;
    }

    const int size = this->v_eff.nr * this->v_eff.nc;
    syncmem_d2d_d2h_op()(this->v_eff.c, this->d_veff_smooth, size);
    this->veff_smooth = this->v_eff;
    this->v_eff_host_stale_ = false;
}

void Potential::cal_fixed_v(double* vl_pseudo)
{
    ModuleBase::TITLE("Potential", "cal_fixed_v");
    ModuleBase::timer::start("Potential", "cal_fixed_v");

    this->v_eff_fixed.assign(this->v_eff_fixed.size(), 0.0);
    for (size_t i = 0; i < this->components.size(); i++)
    {
        if (this->components[i]->fixed_mode)
        {
            this->components[i]->cal_fixed_v(vl_pseudo);
        }
    }

    ModuleBase::timer::end("Potential", "cal_fixed_v");
}

void Potential::cal_v_eff(const Charge*const chg, const UnitCell*const ucell, ModuleBase::matrix& v_eff)
{
    ModuleBase::TITLE("Potential", "cal_veff");
    ModuleBase::timer::start("Potential", "cal_veff");

    const int nspin_current = this->v_eff.nr;
    const int nrxx = this->v_eff.nc;
    // first of all, set v_eff to zero.
    this->v_eff.zero_out();
    this->v_eff_host_stale_ = false;

    // add fixed potential components
    // nspin = 2, add fixed components for all
    // nspin = 4, add fixed components on first colomn
    for (int i = 0; i < nspin_current; i++)
    {
        if (i == 0 || nspin_current == 2)
        {
            ModuleBase::GlobalFunc::COPYARRAY(this->v_eff_fixed.data(), &(this->v_eff(i, 0)), nrxx);
        }
    }

    // cal eff by every components
    for (size_t i = 0; i < this->components.size(); i++)
    {
        if (this->components[i]->dynamic_mode)
        {
            this->components[i]->cal_v_eff(chg, ucell, v_eff);
        }
    }

    ModuleBase::timer::end("Potential", "cal_veff");
}

void Potential::init_pot(const Charge*const chg)
{
    ModuleBase::TITLE("Potential", "init_pot");
    ModuleBase::timer::start("Potential", "init_pot");

    // fixed components only calculated in the beginning of SCF
    this->fixed_done = false;

    this->update_from_charge(chg, this->ucell_);

    ModuleBase::timer::end("Potential", "init_pot");
    return;
}

void Potential::get_vnew(const Charge* chg, ModuleBase::matrix& vnew)
{
    ModuleBase::TITLE("Potential", "get_vnew");
    vnew.create(this->v_eff.nr, this->v_eff.nc);
    this->materialize_eff_v_host();
    vnew = this->v_eff;

    this->update_from_charge(chg, this->ucell_);
    this->materialize_eff_v_host();
    //(used later for scf correction to the forces )
    for (int iter = 0; iter < vnew.nr * vnew.nc; ++iter)
    {
        vnew.c[iter] = this->v_eff.c[iter] - vnew.c[iter];
    }

    return;
}

void Potential::interpolate_vrs(void)
{
    ModuleBase::TITLE("Potential", "interpolate_vrs");
    ModuleBase::timer::start("Potential", "interpolate_vrs");

    const int nspin = PARAM.inp.nspin;
    assert(nspin==1 || nspin==2 || nspin==4);

    if (PARAM.globalv.double_grid)
    {
        if (rho_basis_->gamma_only != rho_basis_smooth_->gamma_only)
        {
            ModuleBase::WARNING_QUIT("Potential::interpolate_vrs", "gamma_only is not consistent");
        }

        ModuleBase::ComplexMatrix vrs(nspin, rho_basis_->npw);
        for (int is = 0; is < nspin; is++)
        {
            rho_basis_->real2recip(&v_eff(is, 0), &vrs(is, 0));
            rho_basis_smooth_->recip2real(&vrs(is, 0), &veff_smooth(is, 0));
        }

        if (XC_Functional::get_ked_flag())
        {
            ModuleBase::ComplexMatrix vrs_ofk(nspin, rho_basis_->npw);
            for (int is = 0; is < nspin; is++)
            {
                rho_basis_->real2recip(&vofk_eff(is, 0), &vrs_ofk(is, 0));
                rho_basis_smooth_->recip2real(&vrs_ofk(is, 0), &vofk_smooth(is, 0));
            }
        }
    }
    else
    {
        this->veff_smooth = this->v_eff;
        this->vofk_smooth = this->vofk_eff;
    }

    ModuleBase::timer::end("Potential", "interpolate_vrs");
}

template <>
float* Potential::get_veff_smooth_data()
{
    return this->veff_smooth.nc > 0 ? this->s_veff_smooth : nullptr;
}

template <>
double* Potential::get_veff_smooth_data()
{
    return this->veff_smooth.nc > 0 ? this->d_veff_smooth : nullptr;
}

template <>
float* Potential::get_vofk_smooth_data()
{
    return this->vofk_smooth.nc > 0 ? this->s_vofk_smooth : nullptr;
}

template <>
double* Potential::get_vofk_smooth_data()
{
    return this->vofk_smooth.nc > 0 ? this->d_vofk_smooth : nullptr;
}

double Potential::get_ml_exx_energy() const
{
#ifdef __MLALGO
    for (size_t i = 0; i < this->components.size(); i++)
    {
        PotML_EXX* pot_ml_exx = dynamic_cast<PotML_EXX*>(this->components[i]);
        if (pot_ml_exx != nullptr)
        {
            return pot_ml_exx->get_energy();
        }
    }
    return 0.0;
#else
    return 0.0;
#endif
}

} // namespace elecstate

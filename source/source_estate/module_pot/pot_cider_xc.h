#ifndef POT_CIDER_XC_H
#define POT_CIDER_XC_H

#if defined(ENABLE_CIDER) && defined(USE_LIBXC)

#include "pot_base.h"
#include "cider_bridge.h"

namespace elecstate
{

class PotCiderXC : public PotBase
{
  public:
    PotCiderXC(const ModulePW::PW_Basis* rho_basis_in,
               const UnitCell* ucell_in,
               double* etxc_in,
               double* vtxc_in,
               ModuleBase::matrix* vofk_in = nullptr);

    ~PotCiderXC() override;

    void cal_v_eff(const Charge* const chg,
                   const UnitCell* const ucell,
                   ModuleBase::matrix& v_eff) override;

    static const ModuleBase::matrix* debug_last_feature_potential();

  private:
    cider_bridge_ctx* ctx_ = nullptr;
    double* etxc_ = nullptr;
    double* vtxc_ = nullptr;
    ModuleBase::matrix* vofk_ = nullptr;
    bool is_mgga_ = false;
    int nspin_ = 1;
};

} // namespace elecstate

#endif // ENABLE_CIDER && USE_LIBXC
#endif // POT_CIDER_XC_H

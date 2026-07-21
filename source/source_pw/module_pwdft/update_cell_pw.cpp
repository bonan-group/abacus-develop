#include "source_pw/module_pwdft/update_cell_pw.h"
#include "source_base/global_variable.h"
#include "source_base/global_function.h"
#include "source_io/module_parameter/parameter.h"

namespace pw
{

void update_cell_pw(UnitCell& ucell,
                    pseudopot_cell_vnl& ppcell,
                    const K_Vectors& kv,
                    ModulePW::PW_Basis_K* pw_wfc,
                    const ModulePW::PW_Basis* pw_rhod,
                    bool prepare_uspp_stress,
                    int nqxq,
                    double dq,
                    const Input_para& inp)
{
    ModuleBase::TITLE("pw", "update_cell_pw");

    if (!ucell.cell_parameter_updated && !ucell.ionic_position_updated)
    {
        return;
    }

    ppcell.update_after_structure_change(ucell, pw_rhod, prepare_uspp_stress, nqxq, dq);
    if (!ucell.cell_parameter_updated)
    {
        return;
    }

    ModuleBase::GlobalFunc::DONE(GlobalV::ofs_running, "NON-LOCAL POTENTIAL");

    pw_wfc->initgrids(ucell.lat0, ucell.latvec, pw_wfc->nx, pw_wfc->ny, pw_wfc->nz);
    pw_wfc->initparameters(false, inp.ecutwfc, kv.get_nks(), kv.kvec_d.data());
    pw_wfc->collect_local_pw(inp.erf_ecut, inp.erf_height, inp.erf_sigma);
}

}

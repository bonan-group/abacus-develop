#ifndef TRAINING_DUMP_H
#define TRAINING_DUMP_H

#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_estate/elecstate.h"

#include <string>

class Charge;

namespace ModuleIO
{

void write_training_dump(const UnitCell& ucell,
                         const elecstate::ElecState& elec,
                         const ModulePW::PW_Basis& rho_basis,
                         const Charge& chr,
                         const int istep,
                         const std::string& output_root);

} // namespace ModuleIO

#endif

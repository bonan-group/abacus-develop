#ifndef SOURCE_HAMILT_MODULE_XC_XC_RESIDENT_GPU_H
#define SOURCE_HAMILT_MODULE_XC_XC_RESIDENT_GPU_H

#include "source_hamilt/module_xc/kernels/xc_resident_selector.h"

#include <cstddef>
#include <string>
#include <vector>

class Charge;
class UnitCell;

namespace ModuleBase
{
class matrix;
}

namespace ModulePW
{
class PW_Basis;
}

namespace XC_Functional_GPU
{

struct XcGpuRequest
{
    XcGpuRequest();

    std::string device;
    int nrxx;
    int nspin;
    bool use_libxc;
    bool has_kinetic_energy_density;
    const std::vector<int>* functional_ids;
    const Charge* charge;
    ModulePW::PW_Basis* rho_basis;
    const UnitCell* unit_cell;
    ModuleBase::matrix* host_potential;
    double* device_potential;
    std::size_t potential_size;
};

struct XcGpuResult
{
    bool used;
    double energy;
    double potential_sum;
};

XcGpuMode select_xc_gpu_mode(const XcGpuRequest& request);
XcGpuResult evaluate_resident_xc(const XcGpuRequest& request);
bool evaluate_resident_xc_stress(const XcGpuRequest& request, std::vector<double>& stress);

} // namespace XC_Functional_GPU

#endif // SOURCE_HAMILT_MODULE_XC_XC_RESIDENT_GPU_H

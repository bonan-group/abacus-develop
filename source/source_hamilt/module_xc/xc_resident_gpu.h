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
    std::string device = "cpu";
    int nrxx = 0;
    int nspin = 0;
    bool use_libxc = false;
    bool has_kinetic_energy_density = false;
    const std::vector<int>* functional_ids = nullptr;
    const Charge* charge = nullptr;
    ModulePW::PW_Basis* rho_basis = nullptr;
    const UnitCell* unit_cell = nullptr;
    ModuleBase::matrix* host_potential = nullptr;
    double* device_potential = nullptr;
    std::size_t potential_size = 0;
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

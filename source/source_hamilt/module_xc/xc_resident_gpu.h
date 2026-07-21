#ifndef SOURCE_HAMILT_MODULE_XC_XC_RESIDENT_GPU_H
#define SOURCE_HAMILT_MODULE_XC_XC_RESIDENT_GPU_H

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

enum class XcGpuMode
{
    Unsupported,
    LdaPzSpin,
    LdaPwSpin,
    Pbe,
    PbeSol,
    SpinPbe,
    SpinPbeSol
};

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
    const double* rho_up;
    const double* rho_down;
    ModuleBase::matrix* host_potential;
    double* device_potential;
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

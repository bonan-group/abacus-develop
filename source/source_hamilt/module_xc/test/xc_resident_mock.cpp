#include "source_hamilt/module_xc/xc_resident_gpu.h"

namespace XC_Functional_GPU
{

XcGpuRequest::XcGpuRequest()
    : device("cpu"),
      nrxx(0),
      nspin(0),
      use_libxc(false),
      has_kinetic_energy_density(false),
      functional_ids(nullptr),
      charge(nullptr),
      rho_basis(nullptr),
      unit_cell(nullptr),
      host_potential(nullptr),
      device_potential(nullptr),
      potential_size(0)
{
}

XcGpuMode select_xc_gpu_mode(const XcGpuRequest&)
{
    return XcGpuMode::Unsupported;
}

XcGpuResult evaluate_resident_xc(const XcGpuRequest&)
{
    const XcGpuResult result = {false, 0.0, 0.0};
    return result;
}

bool evaluate_resident_xc_stress(const XcGpuRequest&, std::vector<double>&)
{
    return false;
}

} // namespace XC_Functional_GPU

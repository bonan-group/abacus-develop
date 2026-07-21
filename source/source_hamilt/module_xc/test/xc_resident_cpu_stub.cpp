#include "source_hamilt/module_xc/xc_resident_gpu.h"

#include <stdexcept>

namespace XC_Functional_GPU
{

namespace
{

void require_cpu_request(const XcGpuRequest& request)
{
    if (request.device != "cpu")
    {
        throw std::logic_error("CPU XC test stub received a non-CPU request");
    }
}

} // namespace

XcGpuResult evaluate_resident_xc(const XcGpuRequest& request)
{
    require_cpu_request(request);
    const XcGpuResult result = {false, 0.0, 0.0};
    return result;
}

bool evaluate_resident_xc_stress(const XcGpuRequest& request, std::vector<double>&)
{
    require_cpu_request(request);
    return false;
}

} // namespace XC_Functional_GPU

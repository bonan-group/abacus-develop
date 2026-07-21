#ifndef SOURCE_HAMILT_MODULE_XC_KERNELS_XC_RESIDENT_SELECTOR_H
#define SOURCE_HAMILT_MODULE_XC_KERNELS_XC_RESIDENT_SELECTOR_H

#include <vector>

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

XcGpuMode select_xc_gpu_mode(const std::vector<int>& functional_ids, int nspin, int poolnproc);
bool is_xc_gpu_evaluator_available();

} // namespace XC_Functional_GPU

#endif // SOURCE_HAMILT_MODULE_XC_KERNELS_XC_RESIDENT_SELECTOR_H

#include "source_hamilt/module_xc/xc_resident_gpu.h"

#ifdef USE_LIBXC
#include <xc.h>
#else
#include "source_hamilt/module_xc/xc_ids.h"
#endif

namespace XC_Functional_GPU
{

bool is_xc_gpu_evaluator_available()
{
#if __CUDA || __UT_USE_CUDA
    return true;
#else
    return false;
#endif
}

XcGpuMode select_xc_gpu_mode(const std::vector<int>& functional_ids, const int nspin, const int poolnproc)
{
    const bool is_pz
        = functional_ids.size() == 2 && functional_ids[0] == XC_LDA_X && functional_ids[1] == XC_LDA_C_PZ;
    const bool is_pw
        = functional_ids.size() == 2 && functional_ids[0] == XC_LDA_X && functional_ids[1] == XC_LDA_C_PW;
    if (nspin == 2 && (is_pz || is_pw))
    {
        return is_pz ? XcGpuMode::LdaPzSpin : XcGpuMode::LdaPwSpin;
    }

    if (poolnproc != 1)
    {
        return XcGpuMode::Unsupported;
    }

    const bool is_pbe = functional_ids.size() == 2 && functional_ids[0] == XC_GGA_X_PBE
                        && functional_ids[1] == XC_GGA_C_PBE;
    const bool is_pbesol = functional_ids.size() == 2 && functional_ids[0] == XC_GGA_X_PBE_SOL
                           && functional_ids[1] == XC_GGA_C_PBE_SOL;
    if (nspin == 1)
    {
        return is_pbe ? XcGpuMode::Pbe : (is_pbesol ? XcGpuMode::PbeSol : XcGpuMode::Unsupported);
    }
    if (nspin == 2)
    {
        return is_pbe ? XcGpuMode::SpinPbe : (is_pbesol ? XcGpuMode::SpinPbeSol : XcGpuMode::Unsupported);
    }
    return XcGpuMode::Unsupported;
}

} // namespace XC_Functional_GPU

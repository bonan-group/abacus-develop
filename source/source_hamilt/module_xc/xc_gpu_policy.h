#ifndef SOURCE_HAMILT_MODULE_XC_XC_GPU_POLICY_H
#define SOURCE_HAMILT_MODULE_XC_XC_GPU_POLICY_H

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace XC_Functional_GPU
{

inline bool xc_gpu_disabled_by_env()
{
    const char* xc_gpu_env = std::getenv("ABACUS_XC_GPU");
    if (xc_gpu_env == nullptr)
    {
        return false;
    }

    std::string value = xc_gpu_env;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });

    return value == "0" || value == "OFF" || value == "FALSE" || value == "NO";
}

inline bool xc_gpu_policy(const bool is_gpu,
                          const bool cpu_debug,
                          const int nspin,
                          const std::string& xc_func)
{
    std::string xc_func_upper = xc_func;
    std::transform(xc_func_upper.begin(), xc_func_upper.end(), xc_func_upper.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });

    return is_gpu && !cpu_debug && nspin == 1
           && (xc_func_upper == "PBE" || xc_func_upper == "PBESOL" || xc_func_upper == "LDA"
               || xc_func_upper == "PZ");
}

inline bool xc_gpu_stress_policy(const bool is_gpu,
                                 const bool cpu_debug,
                                 const int nspin,
                                 const std::string& xc_func)
{
    std::string xc_func_upper = xc_func;
    std::transform(xc_func_upper.begin(), xc_func_upper.end(), xc_func_upper.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });

    return is_gpu && !cpu_debug && (nspin == 1 || nspin == 2)
           && (xc_func_upper == "PBE" || xc_func_upper == "PBESOL");
}

} // namespace XC_Functional_GPU

#endif // SOURCE_HAMILT_MODULE_XC_XC_GPU_POLICY_H

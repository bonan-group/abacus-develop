#ifndef SOURCE_PSI_PSI_INIT_POLICY_H
#define SOURCE_PSI_PSI_INIT_POLICY_H

#include <string>

namespace psi
{

inline bool gpu_random_init_policy(const bool is_gpu,
                                   const bool cpu_debug,
                                   const std::string& ks_solver,
                                   const bool has_initializer,
                                   const std::string& initializer_method)
{
    return is_gpu && !cpu_debug && ks_solver != "bpcg" && has_initializer && initializer_method == "random";
}

} // namespace psi

#endif

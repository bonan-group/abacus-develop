#ifndef SOURCE_PSI_PSI_INIT_POLICY_H
#define SOURCE_PSI_PSI_INIT_POLICY_H

#include <string>

namespace psi
{

inline bool gpu_resident_init_policy(const bool is_gpu,
                                     const std::string& ks_solver,
                                     const bool has_initializer,
                                     const std::string& initializer_method,
                                     const int npol)
{
    if (!is_gpu || ks_solver == "bpcg" || !has_initializer)
    {
        return false;
    }
    if (initializer_method == "random")
    {
        return true;
    }
    return npol == 1 && (initializer_method == "atomic" || initializer_method == "atomic+random");
}

} // namespace psi

#endif

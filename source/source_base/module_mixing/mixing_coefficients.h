#ifndef MIXING_COEFFICIENTS_H_
#define MIXING_COEFFICIENTS_H_

#include <vector>

namespace ModuleBase
{
class matrix;
}

namespace Base_Mixing
{
void solve_broyden_system(ModuleBase::matrix& gram, std::vector<double>& rhs);
void solve_pulay_system(ModuleBase::matrix& gram, std::vector<double>& coefficients);
} // namespace Base_Mixing

#endif // MIXING_COEFFICIENTS_H_

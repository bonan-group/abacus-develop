#include "mixing_coefficients.h"

#include "source_base/matrix.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/tool_quit.h"

namespace Base_Mixing
{
void solve_broyden_system(ModuleBase::matrix& gram, std::vector<double>& rhs)
{
    const int ndim = gram.nr;
    double* work = new double[ndim];
    int* iwork = new int[ndim];
    char uu = 'U';
    int info = 0;
    int m = 1;

    dsysv_(&uu, &ndim, &m, gram.c, &ndim, iwork, rhs.data(), &ndim, work, &ndim, &info);

    if (info != 0)
    {
        ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when DSYSV.");
    }

    delete[] work;
    delete[] iwork;
}

void solve_pulay_system(ModuleBase::matrix& gram, std::vector<double>& coefficients)
{
    const int ndim = gram.nr;
    double* work = new double[ndim];
    int* iwork = new int[ndim];
    char uu = 'U';
    int info;

    dsytrf_(&uu, &ndim, gram.c, &ndim, iwork, work, &ndim, &info);
    if (info != 0)
    {
        ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when factorizing beta.");
    }
    dsytri_(&uu, &ndim, gram.c, &ndim, iwork, work, &info);
    if (info != 0)
    {
        ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when DSYTRI beta.");
    }
    for (int i = 0; i < ndim; ++i)
    {
        for (int j = i + 1; j < ndim; ++j)
        {
            gram(i, j) = gram(j, i);
        }
    }

    double sum_beta = 0.0;
    for (int i = 0; i < ndim; ++i)
    {
        for (int j = 0; j < ndim; ++j)
        {
            sum_beta += gram(j, i);
        }
    }
    for (int i = 0; i < ndim; ++i)
    {
        coefficients[i] = 0.0;
        for (int j = 0; j < ndim; ++j)
        {
            coefficients[i] += gram(i, j);
        }
        coefficients[i] /= sum_beta;
    }

    delete[] work;
    delete[] iwork;
}
} // namespace Base_Mixing

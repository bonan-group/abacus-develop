#ifndef H_EWALD_PW_H
#define H_EWALD_PW_H

#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_cell/unitcell.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_pw/module_pwdft/forces.h"
#include "source_pw/module_pwdft/stress_func.h"
#include "source_pw/module_pwdft/structure_factor.h"

class H_Ewald_pw
{
  public:
    H_Ewald_pw();
    ~H_Ewald_pw();

    // compute the Ewald energy
    // Original interface for backward compatibility
    static double compute_ewald(const UnitCell& cell,
                                const ModulePW::PW_Basis* rho_basis,
                                const ModuleBase::ComplexMatrix& strucFac);

    // New interface with GPU support
    // Pass Structure_Factor reference to access GPU data when available
    static double compute_ewald(const UnitCell& cell,
                                const ModulePW::PW_Basis* rho_basis,
                                const Structure_Factor& sf,
                                const std::string& device);

  public:
    static void rgen(
        const ModuleBase::Vector3<double> &dtau,
        const double &rmax,
        int *irr,
        const ModuleBase::Matrix3 &at,
        const ModuleBase::Matrix3 &bg,
        ModuleBase::Vector3<double> *r,
        double *r2,
        int  &nrm
    );

	// the coefficient of ewald method
	static double alpha;
    static int mxr;

  private:
#if defined(__CUDA) || defined(__ROCM)
    // GPU implementation of Ewald energy calculation
    static double compute_ewald_gpu(const UnitCell& cell,
                                     const ModulePW::PW_Basis* rho_basis,
                                     const Structure_Factor& sf);
#endif
};

#endif //ewald energy

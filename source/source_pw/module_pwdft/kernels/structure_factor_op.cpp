#include "source_pw/module_pwdft/kernels/structure_factor_op.h"

#include "source_base/libm/libm.h"

namespace structure_factor_op
{

void compute_struc_fac_op<double, base_device::DEVICE_CPU>::operator()(const base_device::DEVICE_CPU* ctx,
                                                                       const int ntype,
                                                                       const double* tau,
                                                                       const int* atom_index,
                                                                       const int ngm,
                                                                       const double* gcar,
                                                                       const double two_pi,
                                                                       std::complex<double>* struc_fac)
{
    const std::complex<double> ci_tpi(0.0, -two_pi);
    for (int it = 0; it < ntype; ++it)
    {
        const int ia_begin = atom_index[it];
        const int ia_end = atom_index[it + 1];
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int ig = 0; ig < ngm; ++ig)
        {
            const double gx = gcar[ig * 3 + 0];
            const double gy = gcar[ig * 3 + 1];
            const double gz = gcar[ig * 3 + 2];
            std::complex<double> sum_phase(0.0, 0.0);
            for (int ia = ia_begin; ia < ia_end; ++ia)
            {
                const double gdotr = gx * tau[ia * 3 + 0] + gy * tau[ia * 3 + 1] + gz * tau[ia * 3 + 2];
                sum_phase += ModuleBase::libm::exp(ci_tpi * gdotr);
            }
            struc_fac[it * ngm + ig] = sum_phase;
        }
    }
}

void compute_eigts_op<double, base_device::DEVICE_CPU>::operator()(const base_device::DEVICE_CPU* ctx,
                                                                   const int nat,
                                                                   const double* gtau,
                                                                   const int nx,
                                                                   const int ny,
                                                                   const int nz,
                                                                   const double two_pi,
                                                                   std::complex<double>* eigts1,
                                                                   std::complex<double>* eigts2,
                                                                   std::complex<double>* eigts3)
{
    const std::complex<double> ci_tpi(0.0, -two_pi);
    for (int iat = 0; iat < nat; ++iat)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 16)
#endif
        for (int n1 = -nx; n1 <= nx; ++n1)
        {
            eigts1[iat * (2 * nx + 1) + n1 + nx] = ModuleBase::libm::exp(ci_tpi * (n1 * gtau[iat * 3 + 0]));
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 16)
#endif
        for (int n2 = -ny; n2 <= ny; ++n2)
        {
            eigts2[iat * (2 * ny + 1) + n2 + ny] = ModuleBase::libm::exp(ci_tpi * (n2 * gtau[iat * 3 + 1]));
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 16)
#endif
        for (int n3 = -nz; n3 <= nz; ++n3)
        {
            eigts3[iat * (2 * nz + 1) + n3 + nz] = ModuleBase::libm::exp(ci_tpi * (n3 * gtau[iat * 3 + 2]));
        }
    }
}

} // namespace structure_factor_op

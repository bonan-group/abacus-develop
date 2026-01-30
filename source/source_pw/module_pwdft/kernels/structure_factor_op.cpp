#include "source_pw/module_pwdft/kernels/structure_factor_op.h"
#include "source_base/constants.h"
#include "source_base/libm/libm.h"

#include <complex>

namespace structure_factor_op {

/**
 * @brief CPU implementation of structure factor computation (float)
 */
void compute_struc_fac_op<float, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const int ntype,
    const float* tau,
    const int* atom_index,
    const int ngm,
    const float* gcar,
    const float TWO_PI,
    std::complex<float>* strucFac)
{
    const std::complex<float> ci_tpi = std::complex<float>(0.0f, -TWO_PI);

    for (int it = 0; it < ntype; it++)
    {
        const int ia_start = atom_index[it];
        const int ia_end = atom_index[it + 1];

        for (int ig = 0; ig < ngm; ig++)
        {
            const float gx = gcar[ig * 3 + 0];
            const float gy = gcar[ig * 3 + 1];
            const float gz = gcar[ig * 3 + 2];

            std::complex<float> sum_phase(0.0f, 0.0f);

            for (int ia = ia_start; ia < ia_end; ia++)
            {
                const float tau_x = tau[ia * 3 + 0];
                const float tau_y = tau[ia * 3 + 1];
                const float tau_z = tau[ia * 3 + 2];

                const float gdotr = gx * tau_x + gy * tau_y + gz * tau_z;
                sum_phase += ModuleBase::libm::exp(ci_tpi * gdotr);
            }

            strucFac[it * ngm + ig] = sum_phase;
        }
    }
}

/**
 * @brief CPU implementation of structure factor computation (double)
 */
void compute_struc_fac_op<double, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const int ntype,
    const double* tau,
    const int* atom_index,
    const int ngm,
    const double* gcar,
    const double TWO_PI,
    std::complex<double>* strucFac)
{
    const std::complex<double> ci_tpi = std::complex<double>(0.0, -TWO_PI);

    for (int it = 0; it < ntype; it++)
    {
        const int ia_start = atom_index[it];
        const int ia_end = atom_index[it + 1];

        for (int ig = 0; ig < ngm; ig++)
        {
            const double gx = gcar[ig * 3 + 0];
            const double gy = gcar[ig * 3 + 1];
            const double gz = gcar[ig * 3 + 2];

            std::complex<double> sum_phase(0.0, 0.0);

            for (int ia = ia_start; ia < ia_end; ia++)
            {
                const double tau_x = tau[ia * 3 + 0];
                const double tau_y = tau[ia * 3 + 1];
                const double tau_z = tau[ia * 3 + 2];

                const double gdotr = gx * tau_x + gy * tau_y + gz * tau_z;
                sum_phase += ModuleBase::libm::exp(ci_tpi * gdotr);
            }

            strucFac[it * ngm + ig] = sum_phase;
        }
    }
}

/**
 * @brief CPU implementation of eigts computation (float)
 */
void compute_eigts_op<float, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const int nat,
    const float* gtau,
    const int nx,
    const int ny,
    const int nz,
    const float TWO_PI,
    std::complex<float>* eigts1,
    std::complex<float>* eigts2,
    std::complex<float>* eigts3)
{
    const std::complex<float> ci_tpi = std::complex<float>(0.0f, -TWO_PI);  // -i * 2π

    // Compute eigts1
    for (int iat = 0; iat < nat; iat++)
    {
        const float gtau_x = gtau[iat * 3 + 0];
        for (int n1 = -nx; n1 <= nx; n1++)
        {
            const float arg = n1 * gtau_x;
            eigts1[iat * (2 * nx + 1) + (n1 + nx)] = ModuleBase::libm::exp(ci_tpi * arg);
        }
    }

    // Compute eigts2
    for (int iat = 0; iat < nat; iat++)
    {
        const float gtau_y = gtau[iat * 3 + 1];
        for (int n2 = -ny; n2 <= ny; n2++)
        {
            const float arg = n2 * gtau_y;
            eigts2[iat * (2 * ny + 1) + (n2 + ny)] = ModuleBase::libm::exp(ci_tpi * arg);
        }
    }

    // Compute eigts3
    for (int iat = 0; iat < nat; iat++)
    {
        const float gtau_z = gtau[iat * 3 + 2];
        for (int n3 = -nz; n3 <= nz; n3++)
        {
            const float arg = n3 * gtau_z;
            eigts3[iat * (2 * nz + 1) + (n3 + nz)] = ModuleBase::libm::exp(ci_tpi * arg);
        }
    }
}

/**
 * @brief CPU implementation of eigts computation (double)
 */
void compute_eigts_op<double, base_device::DEVICE_CPU>::operator()(
    const base_device::DEVICE_CPU* ctx,
    const int nat,
    const double* gtau,
    const int nx,
    const int ny,
    const int nz,
    const double TWO_PI,
    std::complex<double>* eigts1,
    std::complex<double>* eigts2,
    std::complex<double>* eigts3)
{
    const std::complex<double> ci_tpi = std::complex<double>(0.0, -TWO_PI);  // -i * 2π

    // Compute eigts1
    for (int iat = 0; iat < nat; iat++)
    {
        const double gtau_x = gtau[iat * 3 + 0];
        for (int n1 = -nx; n1 <= nx; n1++)
        {
            const double arg = n1 * gtau_x;
            eigts1[iat * (2 * nx + 1) + (n1 + nx)] = ModuleBase::libm::exp(ci_tpi * arg);
        }
    }

    // Compute eigts2
    for (int iat = 0; iat < nat; iat++)
    {
        const double gtau_y = gtau[iat * 3 + 1];
        for (int n2 = -ny; n2 <= ny; n2++)
        {
            const double arg = n2 * gtau_y;
            eigts2[iat * (2 * ny + 1) + (n2 + ny)] = ModuleBase::libm::exp(ci_tpi * arg);
        }
    }

    // Compute eigts3
    for (int iat = 0; iat < nat; iat++)
    {
        const double gtau_z = gtau[iat * 3 + 2];
        for (int n3 = -nz; n3 <= nz; n3++)
        {
            const double arg = n3 * gtau_z;
            eigts3[iat * (2 * nz + 1) + (n3 + nz)] = ModuleBase::libm::exp(ci_tpi * arg);
        }
    }
}

}  // namespace structure_factor_op

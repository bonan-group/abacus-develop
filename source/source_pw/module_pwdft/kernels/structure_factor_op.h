#ifndef STRUCTURE_FACTOR_OP_H
#define STRUCTURE_FACTOR_OP_H

#include "source_base/module_device/device.h"

#include <complex>

namespace structure_factor_op
{

template <typename FPTYPE, typename Device>
struct compute_struc_fac_op
{
    void operator()(const Device* ctx,
                    int ntype,
                    const FPTYPE* tau,
                    const int* atom_index,
                    int ngm,
                    const FPTYPE* gcar,
                    FPTYPE two_pi,
                    std::complex<FPTYPE>* struc_fac);
};

template <>
struct compute_struc_fac_op<double, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    int ntype,
                    const double* tau,
                    const int* atom_index,
                    int ngm,
                    const double* gcar,
                    double two_pi,
                    std::complex<double>* struc_fac);
};

template <typename FPTYPE, typename Device>
struct compute_eigts_op
{
    void operator()(const Device* ctx,
                    int nat,
                    const FPTYPE* gtau,
                    int nx,
                    int ny,
                    int nz,
                    FPTYPE two_pi,
                    std::complex<FPTYPE>* eigts1,
                    std::complex<FPTYPE>* eigts2,
                    std::complex<FPTYPE>* eigts3);
};

template <>
struct compute_eigts_op<double, base_device::DEVICE_CPU>
{
    void operator()(const base_device::DEVICE_CPU* ctx,
                    int nat,
                    const double* gtau,
                    int nx,
                    int ny,
                    int nz,
                    double two_pi,
                    std::complex<double>* eigts1,
                    std::complex<double>* eigts2,
                    std::complex<double>* eigts3);
};

#if __CUDA || __UT_USE_CUDA
template <typename FPTYPE>
struct compute_struc_fac_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int ntype,
                    const FPTYPE* tau,
                    const int* atom_index,
                    int ngm,
                    const FPTYPE* gcar,
                    FPTYPE two_pi,
                    std::complex<FPTYPE>* struc_fac);
};

template <typename FPTYPE>
struct compute_eigts_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nat,
                    const FPTYPE* gtau,
                    int nx,
                    int ny,
                    int nz,
                    FPTYPE two_pi,
                    std::complex<FPTYPE>* eigts1,
                    std::complex<FPTYPE>* eigts2,
                    std::complex<FPTYPE>* eigts3);
};
#endif

} // namespace structure_factor_op

#endif // STRUCTURE_FACTOR_OP_H

#ifndef STRUCTURE_FACTOR_OP_H
#define STRUCTURE_FACTOR_OP_H

#include <complex>
#include "source_base/module_device/device.h"

namespace structure_factor_op {

/**
 * @brief GPU kernel wrapper for computing structure factors
 *
 * Computes S(G, type) = Σ_atoms exp(-i·2π·G·τ) for each atom type and G-vector
 *
 * @tparam FPTYPE Floating point type (float or double)
 * @param ntype Number of atom types
 * @param tau Atom positions [nat_total * 3] (flattened x,y,z)
 * @param atom_index Cumulative atom indices per type [ntype+1]
 * @param ngm Number of G-vectors
 * @param gcar G-vector Cartesian coordinates [ngm * 3] (flattened x,y,z)
 * @param TWO_PI 2π constant
 * @param strucFac Output structure factors [ntype * ngm]
 */
template <typename FPTYPE, typename Device>
struct compute_struc_fac_op
{
    void operator()(
        const Device* ctx,
        const int ntype,
        const FPTYPE* tau,
        const int* atom_index,
        const int ngm,
        const FPTYPE* gcar,
        const FPTYPE TWO_PI,
        std::complex<FPTYPE>* strucFac);
};

// CPU specializations
template <>
struct compute_struc_fac_op<float, base_device::DEVICE_CPU>
{
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        const int ntype,
        const float* tau,
        const int* atom_index,
        const int ngm,
        const float* gcar,
        const float TWO_PI,
        std::complex<float>* strucFac);
};

template <>
struct compute_struc_fac_op<double, base_device::DEVICE_CPU>
{
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        const int ntype,
        const double* tau,
        const int* atom_index,
        const int ngm,
        const double* gcar,
        const double TWO_PI,
        std::complex<double>* strucFac);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
template <typename FPTYPE>
struct compute_struc_fac_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        const int ntype,
        const FPTYPE* tau,
        const int* atom_index,
        const int ngm,
        const FPTYPE* gcar,
        const FPTYPE TWO_PI,
        std::complex<FPTYPE>* strucFac);
};
#endif

/**
 * @brief Compute eigts1/2/3 phase factors for structure factor
 *
 * Computes eigts[iat, n] = exp(i·2π·n·gtau) for each atom and n value
 * where gtau = G^T · τ (reciprocal lattice matrix times atom position)
 *
 * @tparam FPTYPE Floating point type (float or double)
 * @param nat Total number of atoms
 * @param gtau G^T · τ for each atom [nat * 3] (gtau.x, gtau.y, gtau.z)
 * @param nx, ny, nz FFT grid dimensions
 * @param TWO_PI 2π constant
 * @param eigts1, eigts2, eigts3 Output arrays [nat * (2*nx+1)], [nat * (2*ny+1)], [nat * (2*nz+1)]
 */
template <typename FPTYPE, typename Device>
struct compute_eigts_op
{
    void operator()(
        const Device* ctx,
        const int nat,
        const FPTYPE* gtau,
        const int nx,
        const int ny,
        const int nz,
        const FPTYPE TWO_PI,
        std::complex<FPTYPE>* eigts1,
        std::complex<FPTYPE>* eigts2,
        std::complex<FPTYPE>* eigts3);
};

// CPU specializations
template <>
struct compute_eigts_op<float, base_device::DEVICE_CPU>
{
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        const int nat,
        const float* gtau,
        const int nx,
        const int ny,
        const int nz,
        const float TWO_PI,
        std::complex<float>* eigts1,
        std::complex<float>* eigts2,
        std::complex<float>* eigts3);
};

template <>
struct compute_eigts_op<double, base_device::DEVICE_CPU>
{
    void operator()(
        const base_device::DEVICE_CPU* ctx,
        const int nat,
        const double* gtau,
        const int nx,
        const int ny,
        const int nz,
        const double TWO_PI,
        std::complex<double>* eigts1,
        std::complex<double>* eigts2,
        std::complex<double>* eigts3);
};

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
template <typename FPTYPE>
struct compute_eigts_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nat,
        const FPTYPE* gtau,
        const int nx,
        const int ny,
        const int nz,
        const FPTYPE TWO_PI,
        std::complex<FPTYPE>* eigts1,
        std::complex<FPTYPE>* eigts2,
        std::complex<FPTYPE>* eigts3);
};
#endif

}  // namespace structure_factor_op

#endif  // STRUCTURE_FACTOR_OP_H

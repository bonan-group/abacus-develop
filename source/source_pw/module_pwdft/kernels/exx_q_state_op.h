#ifndef EXX_Q_STATE_OP_H
#define EXX_Q_STATE_OP_H

#include "source_base/macros.h"
#include "source_base/vector3.h"
#include "source_base/module_device/types.h"
#include "source_cell/klist.h"

#include <cstddef>
#include <complex>
#include <vector>

namespace ModulePW
{
class PW_Basis_K;
}

namespace hamilt
{
bool checked_exx_size_product(std::size_t lhs, std::size_t rhs, std::size_t& result);

bool is_exx_realspace_symmetry_grid_compatible(const ModulePW::PW_Basis_K* wfcpw,
                                               const K_Vectors::ExxFullPoint& full_point);

void validate_exx_realspace_symmetry_grid(const ModulePW::PW_Basis_K* wfcpw,
                                          const K_Vectors::ExxFullPoint& full_point);

struct ExxSymmetryRemap
{
    std::vector<int> rep_igl;
    std::vector<int> fft_isz;
    std::vector<int> fft_ixyz;
    std::vector<std::complex<double>> phase;
    mutable int* rep_igl_device = nullptr;
    mutable int* fft_ixyz_device = nullptr;
    mutable std::complex<float>* phase_float_device = nullptr;
    mutable std::complex<double>* phase_double_device = nullptr;
};

ExxSymmetryRemap build_exx_symmetry_remap(const ModulePW::PW_Basis_K* wfcpw,
                                          const K_Vectors::ExxFullPoint& full_point,
                                          int rep_spin_index,
                                          bool need_gpu_fft_index);

template <typename T>
void rotate_exx_realspace_symmetry_cpu(const ModulePW::PW_Basis_K* wfcpw,
                                       const K_Vectors::ExxFullPoint& full_point,
                                       int rep_spin_index,
                                       const T* representative_real,
                                       T* full_real);

template <typename T>
void rotate_exx_realspace_symmetry_adjoint_cpu(const ModulePW::PW_Basis_K* wfcpw,
                                               const K_Vectors::ExxFullPoint& full_point,
                                               int rep_spin_index,
                                               const T* full_real,
                                               T* representative_real);

template <typename T, typename Device>
struct exx_conjugate_real_op
{
    void operator()(const T* in, T* out, std::size_t nrxx);
};

template <typename T, typename Device>
struct exx_rotate_realspace_op
{
    void operator()(const ModulePW::PW_Basis_K* wfcpw,
                    const K_Vectors::ExxFullPoint& full_point,
                    int rep_spin_index,
                    const T* representative_real,
                    T* full_real,
                    int batch_count);
};

template <typename T, typename Device>
struct exx_gather_recip_op
{
    void operator()(const T* in, T* out, const int* map, int nout);
};

template <typename T, typename Device>
struct exx_scatter_add_recip_op
{
    void operator()(const T* in, T* out, const int* map, int nin, T factor);
};
} // namespace hamilt

#endif // EXX_Q_STATE_OP_H

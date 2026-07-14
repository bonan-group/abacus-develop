#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"

#include "source_base/tool_quit.h"

namespace hamilt
{
namespace
{
void unsupported_rocm_exx_q_state()
{
    ModuleBase::WARNING_QUIT("exx_q_state_op", "PW EXX q-state GPU operations are not implemented for ROCm");
}
} // namespace

template <typename FPTYPE>
struct exx_rotate_realspace_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const ModulePW::PW_Basis_K*,
                    const K_Vectors::ExxFullPoint&,
                    int,
                    const T*,
                    T*,
                    int)
    {
        unsupported_rocm_exx_q_state();
    }
};

template <typename FPTYPE>
struct exx_conjugate_real_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T*, T*, std::size_t)
    {
        unsupported_rocm_exx_q_state();
    }
};

template <typename FPTYPE>
struct exx_gather_recip_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T*, T*, const int*, int)
    {
        unsupported_rocm_exx_q_state();
    }
};

template <typename FPTYPE>
struct exx_scatter_add_recip_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T*, T*, const int*, int, T)
    {
        unsupported_rocm_exx_q_state();
    }
};

template struct exx_rotate_realspace_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_rotate_realspace_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct exx_conjugate_real_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_conjugate_real_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct exx_gather_recip_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_gather_recip_op<std::complex<double>, base_device::DEVICE_GPU>;
template struct exx_scatter_add_recip_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct exx_scatter_add_recip_op<std::complex<double>, base_device::DEVICE_GPU>;
} // namespace hamilt

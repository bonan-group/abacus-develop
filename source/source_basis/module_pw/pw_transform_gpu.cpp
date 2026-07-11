#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "pw_basis.h"

#include <string>
namespace ModulePW
{
#if (defined(__CUDA) || defined(__ROCM))
namespace
{
void check_gpu_fft_poolnproc(const int poolnproc, const std::string& caller)
{
    if (poolnproc > 1)
    {
        ModuleBase::WARNING_QUIT(caller,
                                 "GPU FFT with poolnproc > 1 is not supported. "
                                 "Use one MPI rank per pool for GPU PW runs.");
    }
}
} // namespace

template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const FPTYPE* in, std::complex<FPTYPE>* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "real_to_recip gpu");
    check_gpu_fft_poolnproc(this->poolnproc, "PW_Basis::real2recip_gpu");
    const size_t size = this->nrxx;
    base_device::memory::cast_memory_op<std::complex<FPTYPE>, FPTYPE,base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        in,
        size);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::end(this->classname, "real_to_recip gpu");
}
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "real_to_recip gpu");
    check_gpu_fft_poolnproc(this->poolnproc, "PW_Basis::real2recip_gpu");
    base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                          in,
                                                                          this->nrxx);
    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::end(this->classname, "real_to_recip gpu");
}

template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in, FPTYPE* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "recip_to_real gpu");
    check_gpu_fft_poolnproc(this->poolnproc, "PW_Basis::recip2real_gpu");
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<FPTYPE>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);
    if (this->gamma_only)
    {
#if defined(__ROCM)
        ModuleBase::WARNING_QUIT("PW_Basis::recip2real_gpu",
                                 "ROCm gamma-only GPU recip_to_real is not supported. "
                                 "Run this case with device=cpu or use a non-gamma-only GPU path.");
#else
        set_3d_fft_box_gamma_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nx,
                                                                   this->ny,
                                                                   this->nz,
                                                                   this->xprime,
                                                                   this->ig2ixyz_gpu,
                                                                   in,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());
#endif
    }
    else
    {
        set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                             this->ig2ixyz_gpu,
                                                             in,
                                                             this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    }
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::end(this->classname, "recip_to_real gpu");
}
template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::start(this->classname, "recip_to_real gpu");
    check_gpu_fft_poolnproc(this->poolnproc, "PW_Basis::recip2real_gpu");
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<double>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);

    if (this->gamma_only)
    {
#if defined(__ROCM)
        ModuleBase::WARNING_QUIT("PW_Basis::recip2real_gpu",
                                 "ROCm gamma-only GPU recip_to_real is not supported. "
                                 "Run this case with device=cpu or use a non-gamma-only GPU path.");
#else
        set_3d_fft_box_gamma_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nx,
                                                                   this->ny,
                                                                   this->nz,
                                                                   this->xprime,
                                                                   this->ig2ixyz_gpu,
                                                                   in,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());
#endif
    }
    else
    {
        set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                             this->ig2ixyz_gpu,
                                                             in,
                                                             this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    }
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::end(this->classname, "recip_to_real gpu");
}
template void PW_Basis::real2recip_gpu<double>(const double* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const float* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::real2recip_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               double* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              float* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

#endif
} // namespace ModulePW

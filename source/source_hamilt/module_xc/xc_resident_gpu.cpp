#include "source_hamilt/module_xc/xc_resident_gpu.h"

#include "source_base/constants.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/timer.h"
#include "source_hamilt/module_xc/kernels/xc_gradcorr_op.h"
#include "source_hamilt/module_xc/xc_functional.h"

#include <cmath>
#include <complex>
#include <vector>

namespace XC_Functional_GPU
{

XcGpuRequest::XcGpuRequest()
    : device("cpu"),
      nrxx(0),
      nspin(0),
      use_libxc(false),
      has_kinetic_energy_density(false),
      functional_ids(nullptr),
      charge(nullptr),
      rho_basis(nullptr),
      unit_cell(nullptr),
      host_potential(nullptr),
      device_potential(nullptr),
      potential_size(0)
{
}

namespace
{

XcGpuMode select_available_mode(const XcGpuRequest& request, const bool require_potential_output)
{
    if (!is_xc_gpu_evaluator_available())
    {
        (void)request;
        (void)require_potential_output;
        return XcGpuMode::Unsupported;
    }
#if __CUDA || __UT_USE_CUDA
    if (request.device != "gpu" || request.use_libxc || request.has_kinetic_energy_density
        || request.functional_ids == nullptr || request.charge == nullptr || request.rho_basis == nullptr
        || request.unit_cell == nullptr || request.nrxx <= 0 || request.charge->nrxx != request.nrxx
        || request.charge->rhopw != request.rho_basis || request.rho_basis->nrxx != request.nrxx
        || request.rho_basis->nxyz <= 0 || request.charge->nspin < request.nspin
        || request.charge->rho_core == nullptr || request.charge->get_rho_d(0) == nullptr
        || request.charge->get_device() != "gpu" || request.rho_basis->get_device() != "gpu")
    {
        return XcGpuMode::Unsupported;
    }

    if (require_potential_output)
    {
        const bool host_output = request.host_potential != nullptr;
        const bool device_output = request.device_potential != nullptr;
        const std::size_t expected_size = static_cast<std::size_t>(request.nspin) * request.nrxx;
        if (host_output == device_output || !std::isfinite(request.unit_cell->omega) || request.unit_cell->omega <= 0.0)
        {
            return XcGpuMode::Unsupported;
        }
        if (host_output
            && (request.potential_size != 0 || request.host_potential->nr != request.nspin
                || request.host_potential->nc != request.nrxx || request.host_potential->c == nullptr))
        {
            return XcGpuMode::Unsupported;
        }
        if (device_output && request.potential_size != expected_size)
        {
            return XcGpuMode::Unsupported;
        }
    }

    const XcGpuMode mode
        = select_xc_gpu_mode(*request.functional_ids, request.nspin, request.rho_basis->poolnproc);
    if (mode == XcGpuMode::Unsupported || (request.nspin == 2 && request.charge->get_rho_d(1) == nullptr))
    {
        return XcGpuMode::Unsupported;
    }
    if (mode == XcGpuMode::LdaPzSpin || mode == XcGpuMode::LdaPwSpin)
    {
        return mode;
    }

    if (request.rho_basis->poolnproc != 1 || request.rho_basis->npw <= 0 || request.rho_basis->gcar == nullptr
        || !std::isfinite(request.unit_cell->tpiba) || request.unit_cell->tpiba <= 0.0)
    {
        return XcGpuMode::Unsupported;
    }

    return mode;
#else
    return XcGpuMode::Unsupported;
#endif
}

} // namespace

XcGpuMode select_xc_gpu_mode(const XcGpuRequest& request)
{
    return select_available_mode(request, true);
}

#if __CUDA || __UT_USE_CUDA
namespace
{

using complex_t = std::complex<double>;
using sync_double_h2d_op
    = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using sync_double_d2h_op
    = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using set_double_op = base_device::memory::set_memory_op<double, base_device::DEVICE_GPU>;

template <typename T>
class DeviceBuffer
{
  public:
    DeviceBuffer() : data_(nullptr) {}

    ~DeviceBuffer()
    {
        base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(data_);
    }

    void resize(const std::size_t size)
    {
        base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(data_, size);
    }

    T* get() { return data_; }
    const T* get() const { return data_; }

  private:
    DeviceBuffer(const DeviceBuffer&);
    DeviceBuffer& operator=(const DeviceBuffer&);

    T* data_;
};

class XcGpuWorkspace
{
  public:
    explicit XcGpuWorkspace(const XcGpuRequest& request) : request_(request) {}

    void allocate_lda()
    {
        const int nrxx = request_.nrxx;
        rho_core_.resize(nrxx);
        rho_up_.resize(nrxx);
        rho_down_.resize(nrxx);
        potential_.resize(2 * nrxx);
        sums_.resize(2);
        sync_double_h2d_op()(rho_core_.get(), request_.charge->rho_core, nrxx);
    }

    void allocate_gga(const bool spin)
    {
        const int nrxx = request_.nrxx;
        const int npw = request_.rho_basis->npw;
        rho_core_.resize(nrxx);
        rho_up_.resize(nrxx);
        if (spin)
        {
            rho_down_.resize(nrxx);
        }
        potential_.resize(spin ? 2 * nrxx : nrxx);
        sums_.resize(2);
        gcar_.resize(3 * npw);
        grad_up_.resize(3 * nrxx);
        if (spin)
        {
            grad_down_.resize(3 * nrxx);
        }
        grad_scratch_.resize(nrxx);
        h_up_.resize(3 * nrxx);
        if (spin)
        {
            h_down_.resize(3 * nrxx);
        }
        h_component_.resize(nrxx);
        divergence_.resize(nrxx);
        divergence_sum_.resize(1);
        rho_g_.resize(npw);
        grad_g_.resize(npw);
        aux_g_.resize(npw);
        divergence_g_.resize(npw);
        sync_double_h2d_op()(rho_core_.get(), request_.charge->rho_core, nrxx);
        upload_geometry();
    }

    void allocate_stress(const bool spin)
    {
        const int nrxx = request_.nrxx;
        const int npw = request_.rho_basis->npw;
        rho_core_.resize(nrxx);
        rho_up_.resize(nrxx);
        if (spin)
        {
            rho_down_.resize(nrxx);
        }
        potential_.resize(spin ? 2 * nrxx : nrxx);
        sums_.resize(2);
        gcar_.resize(3 * npw);
        grad_up_.resize(3 * nrxx);
        if (spin)
        {
            grad_down_.resize(3 * nrxx);
        }
        grad_scratch_.resize(nrxx);
        stress_.resize(9);
        rho_g_.resize(npw);
        grad_g_.resize(npw);
        sync_double_h2d_op()(rho_core_.get(), request_.charge->rho_core, nrxx);
        upload_geometry();
    }

    void build_gradient(double* density, double* gradient)
    {
        const int nrxx = request_.nrxx;
        const int npw = request_.rho_basis->npw;
        request_.rho_basis->real_to_recip<double, complex_t, base_device::DEVICE_GPU>(density, rho_g_.get());
        for (int ipol = 0; ipol < 3; ++ipol)
        {
            hamilt::xc_multiply_iG_op<double, base_device::DEVICE_GPU>()(
                nullptr, npw, ipol, gcar_.get(), rho_g_.get(), grad_g_.get());
            set_double_op()(grad_scratch_.get(), 0, nrxx);
            request_.rho_basis->recip_to_real<complex_t, double, base_device::DEVICE_GPU>(
                grad_g_.get(), grad_scratch_.get(), true, request_.unit_cell->tpiba);
            hamilt::xc_set_component_op<double, base_device::DEVICE_GPU>()(
                nullptr, nrxx, ipol, grad_scratch_.get(), gradient);
        }
    }

    void apply_divergence(const double* h,
                          const double* density,
                          double* potential,
                          const bool spin,
                          double& potential_sum)
    {
        const int nrxx = request_.nrxx;
        const int npw = request_.rho_basis->npw;
        for (int ipol = 0; ipol < 3; ++ipol)
        {
            hamilt::xc_extract_component_op<double, base_device::DEVICE_GPU>()(
                nullptr, nrxx, ipol, h, h_component_.get());
            request_.rho_basis->real_to_recip<double, complex_t, base_device::DEVICE_GPU>(
                h_component_.get(), aux_g_.get());
            hamilt::xc_accumulate_iG_op<double, base_device::DEVICE_GPU>()(
                nullptr, npw, ipol, gcar_.get(), aux_g_.get(), divergence_g_.get(), ipol == 0);
        }
        set_double_op()(divergence_.get(), 0, nrxx);
        request_.rho_basis->recip_to_real<complex_t, double, base_device::DEVICE_GPU>(
            divergence_g_.get(), divergence_.get(), true, request_.unit_cell->tpiba);
        double delta = 0.0;
        if (spin)
        {
            hamilt::xc_apply_dh_spin_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                           nrxx,
                                                                           density,
                                                                           rho_core_.get(),
                                                                           divergence_.get(),
                                                                           potential,
                                                                           divergence_sum_.get(),
                                                                           &delta);
        }
        else
        {
            hamilt::xc_apply_dh_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                      nrxx,
                                                                      density,
                                                                      rho_core_.get(),
                                                                      divergence_.get(),
                                                                      potential,
                                                                      divergence_sum_.get(),
                                                                      &delta);
        }
        potential_sum += delta;
    }

    void finalize(const int size)
    {
        if (request_.device_potential != nullptr)
        {
            ModuleBase::timer::start("XC_Functional", "v_xc_resident_device_add");
            hamilt::xc_add_potential_op<double, base_device::DEVICE_GPU>()(
                nullptr, size, potential_.get(), request_.device_potential);
            ModuleBase::timer::end("XC_Functional", "v_xc_resident_device_add");
        }
        else
        {
            ModuleBase::timer::start("XC_Functional", "v_xc_resident_d2h");
            sync_double_d2h_op()(request_.host_potential->c, potential_.get(), size);
            ModuleBase::timer::end("XC_Functional", "v_xc_resident_d2h");
        }
    }

    double* rho_core() { return rho_core_.get(); }
    double* rho_up() { return rho_up_.get(); }
    double* rho_down() { return rho_down_.get(); }
    double* potential() { return potential_.get(); }
    double* sums() { return sums_.get(); }
    double* grad_up() { return grad_up_.get(); }
    double* grad_down() { return grad_down_.get(); }
    double* h_up() { return h_up_.get(); }
    double* h_down() { return h_down_.get(); }
    double* stress() { return stress_.get(); }

  private:
    XcGpuWorkspace(const XcGpuWorkspace&);
    XcGpuWorkspace& operator=(const XcGpuWorkspace&);

    void upload_geometry()
    {
        const int npw = request_.rho_basis->npw;
        std::vector<double> flat(3 * npw);
        for (int ig = 0; ig < npw; ++ig)
        {
            flat[3 * ig] = request_.rho_basis->gcar[ig].x;
            flat[3 * ig + 1] = request_.rho_basis->gcar[ig].y;
            flat[3 * ig + 2] = request_.rho_basis->gcar[ig].z;
        }
        sync_double_h2d_op()(gcar_.get(), flat.data(), 3 * npw);
    }

    const XcGpuRequest& request_;
    DeviceBuffer<double> rho_core_;
    DeviceBuffer<double> rho_up_;
    DeviceBuffer<double> rho_down_;
    DeviceBuffer<double> potential_;
    DeviceBuffer<double> sums_;
    DeviceBuffer<double> gcar_;
    DeviceBuffer<double> grad_up_;
    DeviceBuffer<double> grad_down_;
    DeviceBuffer<double> grad_scratch_;
    DeviceBuffer<double> h_up_;
    DeviceBuffer<double> h_down_;
    DeviceBuffer<double> h_component_;
    DeviceBuffer<double> divergence_;
    DeviceBuffer<double> divergence_sum_;
    DeviceBuffer<double> stress_;
    DeviceBuffer<complex_t> rho_g_;
    DeviceBuffer<complex_t> grad_g_;
    DeviceBuffer<complex_t> aux_g_;
    DeviceBuffer<complex_t> divergence_g_;
};

XcGpuResult evaluate_lda(const XcGpuRequest& request, const XcGpuMode mode, XcGpuWorkspace& workspace)
{
    XcGpuResult result = {true, 0.0, 0.0};
    workspace.allocate_lda();
    hamilt::xc_scalar_lda_spin_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                     request.nrxx,
                                                                     mode == XcGpuMode::LdaPzSpin ? 0 : 1,
                                                                     2.0,
                                                                     1.0e-10,
                                                                     request.charge->get_rho_d(0),
                                                                     request.charge->get_rho_d(1),
                                                                     workspace.rho_core(),
                                                                     workspace.rho_up(),
                                                                     workspace.rho_down(),
                                                                     workspace.potential(),
                                                                     workspace.sums(),
                                                                     &result.energy,
                                                                     &result.potential_sum);
    workspace.finalize(2 * request.nrxx);
    return result;
}

XcGpuResult evaluate_pbe(const XcGpuRequest& request, const XcGpuMode mode, XcGpuWorkspace& workspace)
{
    XcGpuResult result = {true, 0.0, 0.0};
    const bool spin = mode == XcGpuMode::SpinPbe || mode == XcGpuMode::SpinPbeSol;
    const bool pbesol = mode == XcGpuMode::PbeSol || mode == XcGpuMode::SpinPbeSol;
    const int iflag = pbesol ? 2 : 0;
    workspace.allocate_gga(spin);

    if (spin)
    {
        hamilt::xc_scalar_lda_spin_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                         request.nrxx,
                                                                         1,
                                                                         2.0,
                                                                         1.0e-10,
                                                                         request.charge->get_rho_d(0),
                                                                         request.charge->get_rho_d(1),
                                                                         workspace.rho_core(),
                                                                         workspace.rho_up(),
                                                                         workspace.rho_down(),
                                                                         workspace.potential(),
                                                                         workspace.sums(),
                                                                         &result.energy,
                                                                         &result.potential_sum);
        workspace.build_gradient(workspace.rho_up(), workspace.grad_up());
        workspace.build_gradient(workspace.rho_down(), workspace.grad_down());
        double energy_gradient = 0.0;
        double potential_gradient = 0.0;
        hamilt::xc_gradcorr_pbe_spin_grid_resident_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            request.nrxx,
            iflag,
            2.0,
            1.0e-6,
            workspace.rho_up(),
            workspace.rho_down(),
            workspace.rho_core(),
            workspace.grad_up(),
            workspace.grad_down(),
            workspace.potential(),
            workspace.h_up(),
            workspace.h_down(),
            workspace.sums(),
            &energy_gradient,
            &potential_gradient);
        result.energy += energy_gradient;
        result.potential_sum += potential_gradient;
        workspace.apply_divergence(
            workspace.h_up(), workspace.rho_up(), workspace.potential(), true, result.potential_sum);
        workspace.apply_divergence(workspace.h_down(),
                                   workspace.rho_down(),
                                   workspace.potential() + request.nrxx,
                                   true,
                                   result.potential_sum);
        workspace.finalize(2 * request.nrxx);
    }
    else
    {
        hamilt::xc_scalar_pbe_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                    request.nrxx,
                                                                    2.0,
                                                                    1.0e-10,
                                                                    request.charge->get_rho_d(0),
                                                                    workspace.rho_core(),
                                                                    workspace.rho_up(),
                                                                    workspace.potential(),
                                                                    workspace.sums(),
                                                                    &result.energy,
                                                                    &result.potential_sum);
        workspace.build_gradient(workspace.rho_up(), workspace.grad_up());
        double energy_gradient = 0.0;
        double potential_gradient = 0.0;
        hamilt::xc_gradcorr_pbe_grid_resident_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                                     request.nrxx,
                                                                                     iflag,
                                                                                     2.0,
                                                                                     1.0e-6,
                                                                                     workspace.rho_up(),
                                                                                     workspace.rho_core(),
                                                                                     workspace.grad_up(),
                                                                                     workspace.potential(),
                                                                                     workspace.h_up(),
                                                                                     workspace.sums(),
                                                                                     &energy_gradient,
                                                                                     &potential_gradient);
        result.energy += energy_gradient;
        result.potential_sum += potential_gradient;
        workspace.apply_divergence(
            workspace.h_up(), workspace.rho_up(), workspace.potential(), false, result.potential_sum);
        workspace.finalize(request.nrxx);
    }
    return result;
}

} // namespace
#endif

XcGpuResult evaluate_resident_xc(const XcGpuRequest& request)
{
    const XcGpuMode mode = select_xc_gpu_mode(request);
    XcGpuResult unsupported = {false, 0.0, 0.0};
    if (mode == XcGpuMode::Unsupported)
    {
        return unsupported;
    }

#if __CUDA || __UT_USE_CUDA
    ModuleBase::timer::start("XC_Functional", "v_xc_resident_gpu");
    XcGpuWorkspace workspace(request);
    XcGpuResult result = mode == XcGpuMode::LdaPzSpin || mode == XcGpuMode::LdaPwSpin
                             ? evaluate_lda(request, mode, workspace)
                             : evaluate_pbe(request, mode, workspace);
    ModuleBase::timer::end("XC_Functional", "v_xc_resident_gpu");
    return result;
#else
    return unsupported;
#endif
}

bool evaluate_resident_xc_stress(const XcGpuRequest& request, std::vector<double>& stress)
{
    const XcGpuMode mode = select_available_mode(request, false);
    const bool spin = mode == XcGpuMode::SpinPbe || mode == XcGpuMode::SpinPbeSol;
    const bool scalar = mode == XcGpuMode::Pbe || mode == XcGpuMode::PbeSol;
    if (!spin && !scalar)
    {
        return false;
    }

#if __CUDA || __UT_USE_CUDA
    request.charge->sync_rho_to_device();
    request.charge->sync_kin_r_to_device();
    ModuleBase::timer::start("XC_Functional", "gradcorr_stress_gpu");
    XcGpuWorkspace workspace(request);
    workspace.allocate_stress(spin);
    double unused_energy = 0.0;
    double unused_potential_sum = 0.0;
    if (spin)
    {
        hamilt::xc_scalar_lda_spin_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                         request.nrxx,
                                                                         1,
                                                                         ModuleBase::e2,
                                                                         1.0e-10,
                                                                         request.charge->get_rho_d(0),
                                                                         request.charge->get_rho_d(1),
                                                                         workspace.rho_core(),
                                                                         workspace.rho_up(),
                                                                         workspace.rho_down(),
                                                                         workspace.potential(),
                                                                         workspace.sums(),
                                                                         &unused_energy,
                                                                         &unused_potential_sum);
        workspace.build_gradient(workspace.rho_up(), workspace.grad_up());
        workspace.build_gradient(workspace.rho_down(), workspace.grad_down());
        hamilt::xc_gradcorr_pbe_spin_stress_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            request.nrxx,
            mode == XcGpuMode::SpinPbeSol ? 2 : 0,
            ModuleBase::e2,
            1.0e-6,
            workspace.rho_up(),
            workspace.rho_down(),
            workspace.grad_up(),
            workspace.grad_down(),
            workspace.stress());
    }
    else
    {
        hamilt::xc_scalar_pbe_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                    request.nrxx,
                                                                    ModuleBase::e2,
                                                                    1.0e-10,
                                                                    request.charge->get_rho_d(0),
                                                                    workspace.rho_core(),
                                                                    workspace.rho_up(),
                                                                    workspace.potential(),
                                                                    workspace.sums(),
                                                                    &unused_energy,
                                                                    &unused_potential_sum);
        workspace.build_gradient(workspace.rho_up(), workspace.grad_up());
        hamilt::xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                              request.nrxx,
                                                                              mode == XcGpuMode::PbeSol ? 2 : 0,
                                                                              ModuleBase::e2,
                                                                              1.0e-6,
                                                                              workspace.rho_up(),
                                                                              workspace.grad_up(),
                                                                              workspace.stress());
    }
    stress.assign(9, 0.0);
    sync_double_d2h_op()(stress.data(), workspace.stress(), 9);
    ModuleBase::timer::end("XC_Functional", "gradcorr_stress_gpu");
    return true;
#else
    return false;
#endif
}

} // namespace XC_Functional_GPU

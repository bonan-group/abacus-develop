#include "charge_mixing.h"

#if __CUDA

#include "source_base/kernels/math_kernel_op.h"
#include "source_base/timer.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_mixing/gpu_mixing.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "kernels/charge_mixing_op.h"

#include <algorithm>

struct ChargeMixingGpuWorkspace
{
    Base_Mixing::GpuMixingData<std::complex<double>>* rho_history = nullptr;
    Base_Mixing::GpuMixingData<std::complex<double>>* tau_history = nullptr;
    double* batch_reduction = nullptr;
    double* batch_result = nullptr;
    std::complex<double>* tau = nullptr;
    std::complex<double>* tau_save = nullptr;
    std::complex<double>* rho_mix_in = nullptr;
    std::complex<double>* rho_mix_out = nullptr;
    std::complex<double>* rho_smooth_in = nullptr;
    std::complex<double>* rho_smooth_out = nullptr;
    std::complex<double>* rho_high_frequency_in = nullptr;
    std::complex<double>* rho_high_frequency_out = nullptr;
    std::complex<double>* plain_residual = nullptr;

    ~ChargeMixingGpuWorkspace()
    {
        delete this->rho_history;
        delete this->tau_history;
        this->delete_device(this->batch_reduction);
        this->delete_device(this->batch_result);
        this->delete_device(this->tau);
        this->delete_device(this->tau_save);
        this->delete_device(this->rho_mix_in);
        this->delete_device(this->rho_mix_out);
        this->delete_device(this->rho_smooth_in);
        this->delete_device(this->rho_smooth_out);
        this->delete_device(this->rho_high_frequency_in);
        this->delete_device(this->rho_high_frequency_out);
        this->delete_device(this->plain_residual);
    }

  private:
    template <typename T>
    void delete_device(T*& pointer)
    {
        if (pointer != nullptr)
        {
            base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(pointer);
            pointer = nullptr;
        }
    }
};

namespace
{
void build_recip_hartree_beta_row_gpu(ChargeMixingGpuWorkspace& workspace,
                                      const ModulePW::PW_Basis* rhopw,
                                      const double tpiba,
                                      const double omega,
                                      const double mixing_angle,
                                      const std::complex<double>* vectors_d,
                                      const int nvec,
                                      const int row,
                                      const int nspin,
                                      const bool gamma_only,
                                      const bool include_magnetism,
                                      ModuleBase::matrix& beta)
{
    if (nvec <= 0 || row < 0 || row >= nvec)
    {
        return;
    }

    const int npw = rhopw->npw;
    const int components = (nspin == 4 && mixing_angle > 0.0) ? 2 : nspin;
    const int vector_length = components * npw;
    const double charge_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (tpiba * tpiba);
    if (components == 1)
    {
        elecstate::inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d + row * vector_length,
            vectors_d,
            rhopw->get_gg_d(),
            npw,
            1,
            nvec,
            rhopw->ig_gge0,
            charge_fac,
            workspace.batch_result,
            workspace.batch_reduction);
    }
    else
    {
        const double mag_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (ModuleBase::TWO_PI * ModuleBase::TWO_PI);
        elecstate::inner_product_recip_hartree_spin_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d + row * vector_length,
            vectors_d,
            rhopw->get_gg_d(),
            npw,
            components,
            1,
            nvec,
            rhopw->ig_gge0,
            gamma_only,
            include_magnetism,
            charge_fac,
            mag_fac,
            workspace.batch_result,
            workspace.batch_reduction);
    }

    std::vector<double> row_values(nvec);
    base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        row_values.data(), workspace.batch_result, nvec);
#ifdef __MPI
    Parallel_Reduce::reduce_pool(row_values.data(), nvec);
#endif
    for (int i = 0; i < nvec; ++i)
    {
        const double value = row_values[i] * omega * 0.5;
        beta(row, i) = value;
        beta(i, row) = value;
    }
}

void build_recip_hartree_gamma_gpu(ChargeMixingGpuWorkspace& workspace,
                                   const ModulePW::PW_Basis* rhopw,
                                   const double tpiba,
                                   const double omega,
                                   const double mixing_angle,
                                   const std::complex<double>* vectors_d,
                                   const std::complex<double>* rhs_d,
                                   const int nvec,
                                   const int nspin,
                                   const bool gamma_only,
                                   const bool include_magnetism,
                                   std::vector<double>& gamma)
{
    gamma.assign(nvec, 0.0);
    if (nvec <= 0)
    {
        return;
    }

    const int npw = rhopw->npw;
    const int components = (nspin == 4 && mixing_angle > 0.0) ? 2 : nspin;
    const int vector_length = components * npw;
    const double charge_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (tpiba * tpiba);
    if (components == 1)
    {
        elecstate::inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d,
            rhs_d,
            rhopw->get_gg_d(),
            npw,
            nvec,
            1,
            rhopw->ig_gge0,
            charge_fac,
            workspace.batch_result,
            workspace.batch_reduction);
    }
    else
    {
        const double mag_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (ModuleBase::TWO_PI * ModuleBase::TWO_PI);
        elecstate::inner_product_recip_hartree_spin_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d,
            rhs_d,
            rhopw->get_gg_d(),
            npw,
            components,
            nvec,
            1,
            rhopw->ig_gge0,
            gamma_only,
            include_magnetism,
            charge_fac,
            mag_fac,
            workspace.batch_result,
            workspace.batch_reduction);
    }

    base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        gamma.data(), workspace.batch_result, nvec);
#ifdef __MPI
    Parallel_Reduce::reduce_pool(gamma.data(), nvec);
#endif
    for (double& value : gamma)
    {
        value *= omega * 0.5;
    }
}
} // namespace

void Charge_Mixing::init_mixing_gpu(const int nspin)
{
    if (this->gpu_workspace_ == nullptr)
    {
        this->gpu_workspace_ = new ChargeMixingGpuWorkspace;
    }

    int resize_tmp = 1;
    if (nspin == 4 && this->mixing_angle > 0)
    {
        resize_tmp = 2;
    }
    const std::size_t length = this->rhopw->npw * nspin / resize_tmp;

    if (this->mixing_mode != "plain" && this->mixing_gpu == nullptr)
    {
        const Base_Mixing::MixingAlgorithm algorithm = this->mixing_mode == "pulay"
                                                           ? Base_Mixing::MixingAlgorithm::Pulay
                                                           : Base_Mixing::MixingAlgorithm::Broyden;
        this->mixing_gpu = new Base_Mixing::GpuMixing<std::complex<double>>(
            algorithm, this->mixing_ndim, this->mixing_beta);
        this->mixing_gpu->reset(length);
    }
    if (this->mixing_gpu != nullptr && this->gpu_workspace_->rho_history == nullptr)
    {
        this->gpu_workspace_->rho_history = new Base_Mixing::GpuMixingData<std::complex<double>>(
            this->mixing_gpu->history_capacity(), length);
    }
    if ((XC_Functional::get_ked_flag()) && this->mixing_tau && this->mixing_gpu != nullptr
        && this->gpu_workspace_->tau_history == nullptr)
    {
        const std::size_t tau_length = this->rhopw->npw * nspin;
        this->gpu_workspace_->tau_history = new Base_Mixing::GpuMixingData<std::complex<double>>(
            this->mixing_gpu->history_capacity(), tau_length);
    }

    if (this->gpu_workspace_->batch_reduction == nullptr || this->gpu_workspace_->batch_result == nullptr)
    {
        const int max_npw = std::max(this->rhopw->npw, this->rhodpw->npw);
        constexpr int min_reduction_threads = 128;
        const int max_blocks = (max_npw + min_reduction_threads - 1) / min_reduction_threads;
        const int max_nvec = std::max(1, this->mixing_ndim);
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->batch_reduction,
            max_nvec * max_nvec * max_blocks,
            "charge_mixing_batch_workspace");
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->batch_result, max_nvec * max_nvec, "charge_mixing_batch_result");
    }
    if ((XC_Functional::get_ked_flag()) && this->mixing_tau && this->gpu_workspace_->tau == nullptr)
    {
        const std::size_t tau_length = this->rhodpw->npw * nspin;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->tau, tau_length, "charge_mixing_tau_g");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->tau_save, tau_length, "charge_mixing_tau_g_save");
    }
    if (this->gpu_workspace_->plain_residual == nullptr)
    {
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->plain_residual,
            nspin * this->rhodpw->npw,
            "charge_mixing_plain_residual");
    }
    if (nspin > 1 && this->gpu_workspace_->rho_mix_in == nullptr)
    {
        const int rho_mix_length = nspin * this->rhopw->npw;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->rho_mix_in, rho_mix_length, "charge_mixing_spin_rhog_in");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->rho_mix_out, rho_mix_length, "charge_mixing_spin_rhog_out");
    }
    if (nspin > 1 && this->rhopw != this->rhodpw && this->gpu_workspace_->rho_smooth_in == nullptr)
    {
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->rho_smooth_in,
            nspin * this->rhopw->npw,
            "charge_mixing_spin_smooth_in");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            this->gpu_workspace_->rho_smooth_out,
            nspin * this->rhopw->npw,
            "charge_mixing_spin_smooth_out");
        const int high_frequency_npw = this->rhodpw->npw - this->rhopw->npw;
        if (high_frequency_npw > 0)
        {
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
                this->gpu_workspace_->rho_high_frequency_in,
                nspin * high_frequency_npw,
                "charge_mixing_spin_high_frequency_in");
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
                this->gpu_workspace_->rho_high_frequency_out,
                nspin * high_frequency_npw,
                "charge_mixing_spin_high_frequency_out");
        }
    }
}

void Charge_Mixing::free_mixing_gpu()
{
    delete this->mixing_gpu;
    this->mixing_gpu = nullptr;
    delete this->gpu_workspace_;
    this->gpu_workspace_ = nullptr;
}

void Charge_Mixing::mix_rho_recip_gpu(Charge* chr, const bool include_magnetism)
{
    ModuleBase::TITLE("Charge_Mixing", "mix_rho_recip_gpu");
    ModuleBase::timer::start("Charge_Mixing", "mix_rho_recip_gpu");

    const int nspin = chr->nspin;
    const bool double_grid = (this->rhopw != this->rhodpw);

    assert(nspin == 1 || nspin == 2 || nspin == 4);
    assert(nspin != 4 || this->mixing_angle <= 0.0);
    assert(mixing_mode == "plain" || mixing_mode == "broyden" || mixing_mode == "pulay");

    // Initialize GPU mixing resources
    init_mixing_gpu(nspin);

    const int npw = this->rhopw->npw;

    // get_drho() prepares the dense reciprocal rho and rho_save buffers immediately before reciprocal mixing.

    const bool spinful_mixing = (nspin > 1);
    const bool spinful_double_grid = (spinful_mixing && double_grid);
    const int rho_components = spinful_mixing ? nspin : 1;
    const int dense_npw = this->rhodpw->npw;
    const int high_frequency_npw = dense_npw - npw;
    std::complex<double>* rhogs_in_d = this->gpu_workspace_->rho_smooth_in;
    std::complex<double>* rhogs_out_d = this->gpu_workspace_->rho_smooth_out;
    std::complex<double>* rhoghf_in_d = this->gpu_workspace_->rho_high_frequency_in;
    std::complex<double>* rhoghf_out_d = this->gpu_workspace_->rho_high_frequency_out;
    if (spinful_double_grid)
    {
        elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr, rhogs_in_d, rhoghf_in_d, chr->get_rhog_save_d(0), npw, dense_npw, nspin);
        elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr, rhogs_out_d, rhoghf_out_d, chr->get_rhog_d(0), npw, dense_npw, nspin);
    }
    std::complex<double>* rhog_in_mix_d
        = spinful_mixing ? this->gpu_workspace_->rho_mix_in : chr->get_rhog_save_d(0);
    std::complex<double>* rhog_out_mix_d
        = spinful_mixing ? this->gpu_workspace_->rho_mix_out : chr->get_rhog_d(0);
    if (spinful_mixing)
    {
        const std::complex<double>* spin_rhog_in_d = spinful_double_grid ? rhogs_in_d : chr->get_rhog_save_d(0);
        const std::complex<double>* spin_rhog_out_d = spinful_double_grid ? rhogs_out_d : chr->get_rhog_d(0);
        elecstate::pack_spin_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr, rhog_in_mix_d, spin_rhog_in_d, npw, nspin);
        elecstate::pack_spin_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr, rhog_out_mix_d, spin_rhog_out_d, npw, nspin);
    }

    // Step 2: GPU Kerker screening function
    auto screen_gpu = [this](std::complex<double>* drhog_d, const int npw_in, const int components) {
        if (this->mixing_gg0 <= 0.0 || this->mixing_beta <= 0.1)
        {
            return;
        }
        const double gg0 = std::pow(this->mixing_gg0 * ModuleBase::BOHR_TO_A / *this->tpiba, 2);
        const double gg0_min = this->mixing_gg0_min / this->mixing_beta;
        elecstate::kerker_screen_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr,  // ctx
            drhog_d,
            this->rhopw->get_gg_d(),
            gg0,
            gg0_min,
            npw_in,
            1);  // nspin=1
        if (components > 1 && this->mixing_gg0_mag > 0.0001 && this->mixing_beta_mag > 0.1)
        {
            const double gg0_mag = std::pow(this->mixing_gg0_mag * ModuleBase::BOHR_TO_A / *this->tpiba, 2);
            const double gg0_min_mag = this->mixing_gg0_min / this->mixing_beta_mag;
            elecstate::kerker_screen_recip_op<double, base_device::DEVICE_GPU>()(
                nullptr,
                drhog_d + npw_in,
                this->rhopw->get_gg_d(),
                gg0_mag,
                gg0_min_mag,
                npw_in,
                components - 1);
        }
    };
    auto screen_rho_gpu = [&screen_gpu, npw, rho_components](std::complex<double>* drhog_d) {
        screen_gpu(drhog_d, npw, rho_components);
    };

    auto plain_mix_gpu = [this](std::complex<double>* data_out_d,
                                const std::complex<double>* data_in_d,
                                const int npw_in,
                                const int components,
                                const bool use_magnetic_beta,
                                std::function<void(std::complex<double>*)> screen) {
        const int length = npw_in * components;
        ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(
            length,
            this->gpu_workspace_->plain_residual,
            data_out_d,
            1.0,
            data_in_d,
            -1.0);
        if (screen != nullptr)
        {
            screen(this->gpu_workspace_->plain_residual);
        }
        if (!use_magnetic_beta || components == 1)
        {
            ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(
                length,
                data_out_d,
                data_in_d,
                1.0,
                this->gpu_workspace_->plain_residual,
                this->mixing_beta);
            return;
        }
        ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(
            npw_in,
            data_out_d,
            data_in_d,
            1.0,
            this->gpu_workspace_->plain_residual,
            this->mixing_beta);
        ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(
            (components - 1) * npw_in,
            data_out_d + npw_in,
            data_in_d + npw_in,
            1.0,
            this->gpu_workspace_->plain_residual + npw_in,
            this->mixing_beta_mag);
    };
    auto rho_history_mix_gpu = [this, npw, rho_components](std::complex<double>* data_out_d,
                                                           const std::complex<double>* data_in_d,
                                                           const std::complex<double>* residual_d) {
        ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(
            npw,
            data_out_d,
            data_in_d,
            1.0,
            residual_d,
            this->mixing_beta);
        if (rho_components > 1)
        {
            ModuleBase::vector_add_vector_op<std::complex<double>, base_device::DEVICE_GPU>()(
                (rho_components - 1) * npw,
                data_out_d + npw,
                data_in_d + npw,
                1.0,
                residual_d + npw,
                this->mixing_beta_mag);
        }
    };

    // Step 3: GPU batched inner product builders.
    const bool gamma_only = this->rhopw->gamma_only;
    auto build_beta_gpu = [this, nspin, gamma_only, include_magnetism](
                              const std::complex<double>* vectors_d,
                              int nvec,
                              int row,
                              ModuleBase::matrix& beta) {
        build_recip_hartree_beta_row_gpu(*this->gpu_workspace_,
                                         this->rhopw,
                                         *this->tpiba,
                                         *this->omega,
                                         this->mixing_angle,
                                         vectors_d,
                                         nvec,
                                         row,
                                         nspin,
                                         gamma_only,
                                         include_magnetism,
                                         beta);
    };
    auto build_gamma_gpu = [this, nspin, gamma_only, include_magnetism](
                               const std::complex<double>* vectors_d,
                               const std::complex<double>* rhs_d,
                               int nvec,
                               std::vector<double>& gamma) {
        build_recip_hartree_gamma_gpu(*this->gpu_workspace_,
                                      this->rhopw,
                                      *this->tpiba,
                                      *this->omega,
                                      this->mixing_angle,
                                      vectors_d,
                                      rhs_d,
                                      nvec,
                                      nspin,
                                      gamma_only,
                                      include_magnetism,
                                      gamma);
    };

    // Step 4-6: Mixing mode specific operations
    if (mixing_mode == "plain")
    {
        plain_mix_gpu(rhog_out_mix_d, rhog_in_mix_d, npw, rho_components, spinful_mixing, screen_rho_gpu);
        if (spinful_mixing)
        {
            std::complex<double>* spin_rhog_out_d = spinful_double_grid ? rhogs_out_d : chr->get_rhog_d(0);
            elecstate::unpack_spin_recip_op<double, base_device::DEVICE_GPU>()(
                nullptr, spin_rhog_out_d, rhog_out_mix_d, npw, nspin);
        }
    }
    else if (mixing_mode == "broyden" || mixing_mode == "pulay")
    {
        this->mixing_gpu->push_data(*this->gpu_workspace_->rho_history,
                                    rhog_in_mix_d,
                                    rhog_out_mix_d,
                                    screen_rho_gpu,
                                    rho_history_mix_gpu,
                                    true);
        this->mixing_gpu->update_coefficients(
            *this->gpu_workspace_->rho_history, build_beta_gpu, build_gamma_gpu);
        this->mixing_gpu->mix_data(*this->gpu_workspace_->rho_history, rhog_out_mix_d);
        if (spinful_mixing)
        {
            std::complex<double>* spin_rhog_out_d = spinful_double_grid ? rhogs_out_d : chr->get_rhog_d(0);
            elecstate::unpack_spin_recip_op<double, base_device::DEVICE_GPU>()(
                nullptr, spin_rhog_out_d, rhog_out_mix_d, npw, nspin);
        }
    }

    if (double_grid)
    {
        if (spinful_double_grid)
        {
            if (high_frequency_npw > 0)
            {
                plain_mix_gpu(rhoghf_out_d, rhoghf_in_d, high_frequency_npw, nspin, false, nullptr);
            }
            elecstate::combine_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
                nullptr, chr->get_rhog_d(0), rhogs_out_d, rhoghf_out_d, npw, dense_npw, nspin);
        }
        else if (high_frequency_npw > 0)
        {
            plain_mix_gpu(chr->get_rhog_d(0) + this->rhopw->npw,
                          chr->get_rhog_save_d(0) + this->rhopw->npw,
                          high_frequency_npw,
                          1,
                          false,
                          nullptr);
        }
    }

    const bool mix_tau = (XC_Functional::get_ked_flag()) && mixing_tau;
    if (mix_tau)
    {
        chr->sync_kin_r_to_device();
        chr->sync_kin_r_save_to_device();
        for (int is = 0; is < nspin; ++is)
        {
            this->rhodpw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
                chr->get_kin_r_d(is), this->gpu_workspace_->tau + is * this->rhodpw->npw);
            this->rhodpw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
                chr->get_kin_r_save_d(is), this->gpu_workspace_->tau_save + is * this->rhodpw->npw);
        }

        std::complex<double>* tau_smooth_in_d = this->gpu_workspace_->tau_save;
        std::complex<double>* tau_smooth_out_d = this->gpu_workspace_->tau;
        std::complex<double>* tau_high_frequency_in_d = nullptr;
        std::complex<double>* tau_high_frequency_out_d = nullptr;
        if (double_grid)
        {
            if (spinful_mixing)
            {
                elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
                    nullptr,
                    this->gpu_workspace_->rho_smooth_in,
                    this->gpu_workspace_->rho_high_frequency_in,
                    this->gpu_workspace_->tau_save,
                    npw,
                    dense_npw,
                    nspin);
                elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
                    nullptr,
                    this->gpu_workspace_->rho_smooth_out,
                    this->gpu_workspace_->rho_high_frequency_out,
                    this->gpu_workspace_->tau,
                    npw,
                    dense_npw,
                    nspin);
                tau_smooth_in_d = this->gpu_workspace_->rho_smooth_in;
                tau_smooth_out_d = this->gpu_workspace_->rho_smooth_out;
                tau_high_frequency_in_d = this->gpu_workspace_->rho_high_frequency_in;
                tau_high_frequency_out_d = this->gpu_workspace_->rho_high_frequency_out;
            }
            else
            {
                tau_high_frequency_in_d = this->gpu_workspace_->tau_save + npw;
                tau_high_frequency_out_d = this->gpu_workspace_->tau + npw;
            }
        }

        if (mixing_mode == "plain")
        {
            plain_mix_gpu(tau_smooth_out_d, tau_smooth_in_d, npw, nspin, false, nullptr);
        }
        else if (mixing_mode == "broyden" || mixing_mode == "pulay")
        {
            this->mixing_gpu->push_data(*this->gpu_workspace_->tau_history,
                                        tau_smooth_in_d,
                                        tau_smooth_out_d,
                                        nullptr,
                                        nullptr,
                                        false);
            this->mixing_gpu->mix_data(*this->gpu_workspace_->tau_history, tau_smooth_out_d);
        }

        if (double_grid)
        {
            if (high_frequency_npw > 0)
            {
                plain_mix_gpu(tau_high_frequency_out_d,
                              tau_high_frequency_in_d,
                              high_frequency_npw,
                              nspin,
                              false,
                              nullptr);
            }
            if (spinful_mixing)
            {
                elecstate::combine_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
                    nullptr,
                    this->gpu_workspace_->tau,
                    tau_smooth_out_d,
                    tau_high_frequency_out_d,
                    npw,
                    dense_npw,
                    nspin);
            }
        }
    }

    // Step 7: GPU FFT: rhog_d -> rho_d
    for (int is = 0; is < nspin; ++is)
    {
        this->rhodpw->recip_to_real<std::complex<double>, double, base_device::DEVICE_GPU>(
            chr->get_rhog_d(is), chr->get_rho_d(is));
    }
    if (mix_tau)
    {
        for (int is = 0; is < nspin; ++is)
        {
            this->rhodpw->recip_to_real<std::complex<double>, double, base_device::DEVICE_GPU>(
                this->gpu_workspace_->tau + is * this->rhodpw->npw, chr->get_kin_r_d(is));
        }
    }

    // Step 8: Sync final result to CPU
    chr->sync_rho_to_host();
    chr->sync_rhog_to_host();
    if (mix_tau)
    {
        chr->sync_kin_r_to_host();
    }
    ModuleBase::timer::end("Charge_Mixing", "mix_rho_recip_gpu");
}

#endif // __CUDA

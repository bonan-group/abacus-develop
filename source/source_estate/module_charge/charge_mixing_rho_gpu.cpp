#include "charge_mixing.h"

#if __CUDA

#include "source_base/timer.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_mixing/kernels/mixing_op.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "kernels/charge_mixing_op.h"

#include <algorithm>

void Charge_Mixing::init_mixing_gpu(const int nspin)
{
    // Initialize GPU mixing data if not already done
    if (rho_mdata_gpu == nullptr)
    {
        // For nspin=4 with mixing_angle > 0, use 2 instead of 4
        int resize_tmp = 1;
        if (nspin == 4 && this->mixing_angle > 0)
        {
            resize_tmp = 2;
        }
        const std::size_t length = this->rhopw->npw * nspin / resize_tmp;

        // Pulay uses mixing_ndim, Broyden uses mixing_ndim+1
        int data_ndim = (mixing_mode == "pulay") ? this->mixing_ndim : this->mixing_ndim + 1;
        rho_mdata_gpu = new Base_Mixing::Mixing_Data_GPU<std::complex<double>>(
            data_ndim, length);
    }
    if ((XC_Functional::get_ked_flag()) && mixing_tau && tau_mdata_gpu == nullptr)
    {
        const std::size_t tau_length = this->rhopw->npw * nspin;
        const int data_ndim = (mixing_mode == "pulay") ? this->mixing_ndim : this->mixing_ndim + 1;
        tau_mdata_gpu = new Base_Mixing::Mixing_Data_GPU<std::complex<double>>(
            data_ndim, tau_length);
    }

    // Initialize GPU Broyden mixing if not already done
    if (mixing_gpu == nullptr && mixing_mode == "broyden")
    {
        mixing_gpu = new Base_Mixing::Broyden_Mixing_GPU<std::complex<double>>(
            this->mixing_ndim, this->mixing_beta);
        mixing_gpu->init(rho_mdata_gpu->length);
    }

    // Initialize GPU Pulay mixing if not already done
    if (mixing_pulay_gpu == nullptr && mixing_mode == "pulay")
    {
        mixing_pulay_gpu = new Base_Mixing::Pulay_Mixing_GPU<std::complex<double>>(
            this->mixing_ndim, this->mixing_beta);
        mixing_pulay_gpu->init(rho_mdata_gpu->length);
    }

    // Initialize GPU workspace for inner products
    if (gpu_workspace_d == nullptr)
    {
        const int max_npw = std::max(this->rhopw->npw, this->rhodpw->npw);
        constexpr int min_reduction_threads = 128;
        const int max_blocks = (max_npw + min_reduction_threads - 1) / min_reduction_threads;
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            gpu_workspace_d, max_blocks, "charge_mixing_workspace");
    }
    if (gpu_batch_workspace_d == nullptr || gpu_batch_result_d == nullptr)
    {
        const int max_npw = std::max(this->rhopw->npw, this->rhodpw->npw);
        constexpr int min_reduction_threads = 128;
        const int max_blocks = (max_npw + min_reduction_threads - 1) / min_reduction_threads;
        const int max_nvec = std::max(1, this->mixing_ndim);
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            gpu_batch_workspace_d, max_nvec * max_nvec * max_blocks, "charge_mixing_batch_workspace");
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            gpu_batch_result_d, max_nvec * max_nvec, "charge_mixing_batch_result");
    }
    if ((XC_Functional::get_ked_flag()) && mixing_tau && tau_g_d == nullptr)
    {
        const std::size_t tau_length = this->rhodpw->npw * nspin;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            tau_g_d, tau_length, "charge_mixing_tau_g");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            tau_g_save_d, tau_length, "charge_mixing_tau_g_save");
    }
    if (plain_residual_d == nullptr)
    {
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            plain_residual_d, nspin * this->rhodpw->npw, "charge_mixing_plain_residual");
    }
    if (nspin > 1 && rho_mix_in_d == nullptr)
    {
        const int rho_mix_length = nspin * this->rhopw->npw;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            rho_mix_in_d, rho_mix_length, "charge_mixing_spin_rhog_in");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            rho_mix_out_d, rho_mix_length, "charge_mixing_spin_rhog_out");
    }
    if (nspin > 1 && this->rhopw != this->rhodpw && rho_smooth_in_d == nullptr)
    {
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            rho_smooth_in_d, nspin * this->rhopw->npw, "charge_mixing_spin_smooth_in");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            rho_smooth_out_d, nspin * this->rhopw->npw, "charge_mixing_spin_smooth_out");
        const int high_frequency_npw = this->rhodpw->npw - this->rhopw->npw;
        if (high_frequency_npw > 0)
        {
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
                rho_high_frequency_in_d,
                nspin * high_frequency_npw,
                "charge_mixing_spin_high_frequency_in");
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
                rho_high_frequency_out_d,
                nspin * high_frequency_npw,
                "charge_mixing_spin_high_frequency_out");
        }
    }
}

void Charge_Mixing::free_mixing_gpu()
{
    if (rho_mdata_gpu != nullptr)
    {
        delete rho_mdata_gpu;
        rho_mdata_gpu = nullptr;
    }
    if (tau_mdata_gpu != nullptr)
    {
        delete tau_mdata_gpu;
        tau_mdata_gpu = nullptr;
    }
    if (mixing_gpu != nullptr)
    {
        delete mixing_gpu;
        mixing_gpu = nullptr;
    }
    if (mixing_pulay_gpu != nullptr)
    {
        delete mixing_pulay_gpu;
        mixing_pulay_gpu = nullptr;
    }
    if (gpu_workspace_d != nullptr)
    {
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(gpu_workspace_d);
        gpu_workspace_d = nullptr;
    }
    if (gpu_batch_workspace_d != nullptr)
    {
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(gpu_batch_workspace_d);
        gpu_batch_workspace_d = nullptr;
    }
    if (gpu_batch_result_d != nullptr)
    {
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(gpu_batch_result_d);
        gpu_batch_result_d = nullptr;
    }
    if (tau_g_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(tau_g_d);
        tau_g_d = nullptr;
    }
    if (tau_g_save_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(tau_g_save_d);
        tau_g_save_d = nullptr;
    }
    if (rho_mix_in_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(rho_mix_in_d);
        rho_mix_in_d = nullptr;
    }
    if (rho_mix_out_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(rho_mix_out_d);
        rho_mix_out_d = nullptr;
    }
    if (rho_smooth_in_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(rho_smooth_in_d);
        rho_smooth_in_d = nullptr;
    }
    if (rho_smooth_out_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(rho_smooth_out_d);
        rho_smooth_out_d = nullptr;
    }
    if (rho_high_frequency_in_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            rho_high_frequency_in_d);
        rho_high_frequency_in_d = nullptr;
    }
    if (rho_high_frequency_out_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            rho_high_frequency_out_d);
        rho_high_frequency_out_d = nullptr;
    }
    if (plain_residual_d != nullptr)
    {
        base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(plain_residual_d);
        plain_residual_d = nullptr;
    }
}

double Charge_Mixing::inner_product_recip_hartree_gpu(
    const std::complex<double>* rhog1_d,
    const std::complex<double>* rhog2_d)
{
    // Use GPU kernel for inner product with 1/G^2 weighting
    static const double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / ((*this->tpiba) * (*this->tpiba));
    const int npw = this->rhopw->npw;
    const int ig_gge0 = this->rhopw->ig_gge0;

    // Call GPU inner product kernel
    double result = elecstate::inner_product_recip_hartree_op<double, base_device::DEVICE_GPU>()(
        nullptr,  // ctx
        rhog1_d,
        rhog2_d,
        this->rhopw->get_gg_d(),
        npw,
        ig_gge0,
        fac,  // tpiba2 factor already included
        gpu_workspace_d);

#ifdef __MPI
    Parallel_Reduce::reduce_pool(result);
#endif

    result *= *this->omega * 0.5;
    return result;
}

void Charge_Mixing::build_recip_hartree_beta_row_gpu(
    const std::complex<double>* vectors_d,
    int nvec,
    int row,
    int nspin,
    bool gamma_only,
    bool include_magnetism,
    ModuleBase::matrix& beta)
{
    if (nvec <= 0 || row < 0 || row >= nvec)
    {
        return;
    }

    const int npw = this->rhopw->npw;
#if __CUDA
    const int components = (nspin == 4 && this->mixing_angle > 0.0) ? 2 : nspin;
    const int vector_length = components * npw;
    const double charge_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / ((*this->tpiba) * (*this->tpiba));
    if (components == 1)
    {
        elecstate::inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d + row * vector_length,
            vectors_d,
            this->rhopw->get_gg_d(),
            npw,
            1,
            nvec,
            this->rhopw->ig_gge0,
            charge_fac,
            gpu_batch_result_d,
            gpu_batch_workspace_d);
    }
    else
    {
        const double mag_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (ModuleBase::TWO_PI * ModuleBase::TWO_PI);
        elecstate::inner_product_recip_hartree_spin_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d + row * vector_length,
            vectors_d,
            this->rhopw->get_gg_d(),
            npw,
            components,
            1,
            nvec,
            this->rhopw->ig_gge0,
            gamma_only,
            include_magnetism,
            charge_fac,
            mag_fac,
            gpu_batch_result_d,
            gpu_batch_workspace_d);
    }

    std::vector<double> row_values(nvec);
    base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        row_values.data(), gpu_batch_result_d, nvec);
#ifdef __MPI
    Parallel_Reduce::reduce_pool(row_values.data(), nvec);
#endif
    for (int i = 0; i < nvec; ++i)
    {
        const double value = row_values[i] * (*this->omega) * 0.5;
        beta(row, i) = value;
        beta(i, row) = value;
    }
#else
    for (int i = 0; i < nvec; ++i)
    {
        const double value = this->inner_product_recip_hartree_gpu(vectors_d + row * npw, vectors_d + i * npw);
        beta(row, i) = value;
        beta(i, row) = value;
    }
#endif
}

void Charge_Mixing::build_recip_hartree_gamma_gpu(
    const std::complex<double>* vectors_d,
    const std::complex<double>* rhs_d,
    int nvec,
    int nspin,
    bool gamma_only,
    bool include_magnetism,
    std::vector<double>& gamma)
{
    gamma.assign(nvec, 0.0);
    if (nvec <= 0)
    {
        return;
    }

    const int npw = this->rhopw->npw;
#if __CUDA
    const int components = (nspin == 4 && this->mixing_angle > 0.0) ? 2 : nspin;
    const int vector_length = components * npw;
    const double charge_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / ((*this->tpiba) * (*this->tpiba));
    if (components == 1)
    {
        elecstate::inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d,
            rhs_d,
            this->rhopw->get_gg_d(),
            npw,
            nvec,
            1,
            this->rhopw->ig_gge0,
            charge_fac,
            gpu_batch_result_d,
            gpu_batch_workspace_d);
    }
    else
    {
        const double mag_fac = ModuleBase::e2 * ModuleBase::FOUR_PI / (ModuleBase::TWO_PI * ModuleBase::TWO_PI);
        elecstate::inner_product_recip_hartree_spin_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            vectors_d,
            rhs_d,
            this->rhopw->get_gg_d(),
            npw,
            components,
            nvec,
            1,
            this->rhopw->ig_gge0,
            gamma_only,
            include_magnetism,
            charge_fac,
            mag_fac,
            gpu_batch_result_d,
            gpu_batch_workspace_d);
    }

    base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        gamma.data(), gpu_batch_result_d, nvec);
#ifdef __MPI
    Parallel_Reduce::reduce_pool(gamma.data(), nvec);
#endif
    for (double& value : gamma)
    {
        value *= (*this->omega) * 0.5;
    }
#else
    for (int i = 0; i < nvec; ++i)
    {
        gamma[i] = this->inner_product_recip_hartree_gpu(vectors_d + i * npw, rhs_d);
    }
#endif
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
    std::complex<double>* rhogs_in_d = rho_smooth_in_d;
    std::complex<double>* rhogs_out_d = rho_smooth_out_d;
    std::complex<double>* rhoghf_in_d = rho_high_frequency_in_d;
    std::complex<double>* rhoghf_out_d = rho_high_frequency_out_d;
    if (spinful_double_grid)
    {
        elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr, rhogs_in_d, rhoghf_in_d, chr->get_rhog_save_d(0), npw, dense_npw, nspin);
        elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr, rhogs_out_d, rhoghf_out_d, chr->get_rhog_d(0), npw, dense_npw, nspin);
    }
    std::complex<double>* rhog_in_mix_d = spinful_mixing ? rho_mix_in_d : chr->get_rhog_save_d(0);
    std::complex<double>* rhog_out_mix_d = spinful_mixing ? rho_mix_out_d : chr->get_rhog_d(0);
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
        const base_device::DEVICE_GPU* ctx = nullptr;
        const int length = npw_in * components;
        mixing::vector_subtract_op<std::complex<double>, base_device::DEVICE_GPU>()(
            ctx, this->plain_residual_d, data_out_d, data_in_d, length);
        if (screen != nullptr)
        {
            screen(this->plain_residual_d);
        }
        if (!use_magnetic_beta || components == 1)
        {
            mixing::vector_axpy_op<std::complex<double>, base_device::DEVICE_GPU>()(
                ctx,
                data_out_d,
                data_in_d,
                std::complex<double>(this->mixing_beta, 0.0),
                this->plain_residual_d,
                length);
            return;
        }
        mixing::vector_axpy_op<std::complex<double>, base_device::DEVICE_GPU>()(
            ctx,
            data_out_d,
            data_in_d,
            std::complex<double>(this->mixing_beta, 0.0),
            this->plain_residual_d,
            npw_in);
        mixing::vector_axpy_op<std::complex<double>, base_device::DEVICE_GPU>()(
            ctx,
            data_out_d + npw_in,
            data_in_d + npw_in,
            std::complex<double>(this->mixing_beta_mag, 0.0),
            this->plain_residual_d + npw_in,
            (components - 1) * npw_in);
    };
    auto rho_history_mix_gpu = [this, npw, rho_components](std::complex<double>* data_out_d,
                                                           const std::complex<double>* data_in_d,
                                                           const std::complex<double>* residual_d) {
        const base_device::DEVICE_GPU* ctx = nullptr;
        mixing::vector_axpy_op<std::complex<double>, base_device::DEVICE_GPU>()(
            ctx,
            data_out_d,
            data_in_d,
            std::complex<double>(this->mixing_beta, 0.0),
            residual_d,
            npw);
        if (rho_components > 1)
        {
            mixing::vector_axpy_op<std::complex<double>, base_device::DEVICE_GPU>()(
                ctx,
                data_out_d + npw,
                data_in_d + npw,
                std::complex<double>(this->mixing_beta_mag, 0.0),
                residual_d + npw,
                (rho_components - 1) * npw);
        }
    };

    // Step 3: GPU batched inner product builders.
    const bool gamma_only = this->rhopw->gamma_only;
    auto build_beta_gpu = [this, nspin, gamma_only, include_magnetism](
                              const std::complex<double>* vectors_d,
                              int nvec,
                              int row,
                              ModuleBase::matrix& beta) {
        this->build_recip_hartree_beta_row_gpu(
            vectors_d, nvec, row, nspin, gamma_only, include_magnetism, beta);
    };
    auto build_gamma_gpu = [this, nspin, gamma_only, include_magnetism](
                               const std::complex<double>* vectors_d,
                               const std::complex<double>* rhs_d,
                               int nvec,
                               std::vector<double>& gamma) {
        this->build_recip_hartree_gamma_gpu(
            vectors_d, rhs_d, nvec, nspin, gamma_only, include_magnetism, gamma);
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
    else if (mixing_mode == "broyden")
    {
        // Broyden mixing path
        mixing_gpu->push_data(
            *rho_mdata_gpu, rhog_in_mix_d, rhog_out_mix_d, screen_rho_gpu, rho_history_mix_gpu, true);
        mixing_gpu->cal_coef_from_beta_gamma(*rho_mdata_gpu, build_beta_gpu, build_gamma_gpu);
        mixing_gpu->mix_data(*rho_mdata_gpu, rhog_out_mix_d);
        if (spinful_mixing)
        {
            std::complex<double>* spin_rhog_out_d = spinful_double_grid ? rhogs_out_d : chr->get_rhog_d(0);
            elecstate::unpack_spin_recip_op<double, base_device::DEVICE_GPU>()(
                nullptr, spin_rhog_out_d, rhog_out_mix_d, npw, nspin);
        }
    }
    else if (mixing_mode == "pulay")
    {
        // Pulay mixing path
        mixing_pulay_gpu->push_data(
            *rho_mdata_gpu, rhog_in_mix_d, rhog_out_mix_d, screen_rho_gpu, rho_history_mix_gpu, true);
        mixing_pulay_gpu->cal_coef_from_beta(*rho_mdata_gpu, build_beta_gpu);
        mixing_pulay_gpu->mix_data(*rho_mdata_gpu, rhog_out_mix_d);
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
        for (int is = 0; is < nspin; ++is)
        {
            this->rhodpw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
                chr->get_kin_r_d(is), tau_g_d + is * this->rhodpw->npw);
            this->rhodpw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
                chr->get_kin_r_save_d(is), tau_g_save_d + is * this->rhodpw->npw);
        }

        std::complex<double>* tau_smooth_in_d = tau_g_save_d;
        std::complex<double>* tau_smooth_out_d = tau_g_d;
        std::complex<double>* tau_high_frequency_in_d = nullptr;
        std::complex<double>* tau_high_frequency_out_d = nullptr;
        if (double_grid)
        {
            if (spinful_mixing)
            {
                elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
                    nullptr,
                    rho_smooth_in_d,
                    rho_high_frequency_in_d,
                    tau_g_save_d,
                    npw,
                    dense_npw,
                    nspin);
                elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
                    nullptr,
                    rho_smooth_out_d,
                    rho_high_frequency_out_d,
                    tau_g_d,
                    npw,
                    dense_npw,
                    nspin);
                tau_smooth_in_d = rho_smooth_in_d;
                tau_smooth_out_d = rho_smooth_out_d;
                tau_high_frequency_in_d = rho_high_frequency_in_d;
                tau_high_frequency_out_d = rho_high_frequency_out_d;
            }
            else
            {
                tau_high_frequency_in_d = tau_g_save_d + npw;
                tau_high_frequency_out_d = tau_g_d + npw;
            }
        }

        if (mixing_mode == "plain")
        {
            plain_mix_gpu(tau_smooth_out_d, tau_smooth_in_d, npw, nspin, false, nullptr);
        }
        else if (mixing_mode == "broyden")
        {
            mixing_gpu->push_data(
                *tau_mdata_gpu, tau_smooth_in_d, tau_smooth_out_d, nullptr, nullptr, false);
            mixing_gpu->mix_data(*tau_mdata_gpu, tau_smooth_out_d);
        }
        else if (mixing_mode == "pulay")
        {
            mixing_pulay_gpu->push_data(
                *tau_mdata_gpu, tau_smooth_in_d, tau_smooth_out_d, nullptr, nullptr, false);
            mixing_pulay_gpu->mix_data(*tau_mdata_gpu, tau_smooth_out_d);
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
                    tau_g_d,
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
                tau_g_d + is * this->rhodpw->npw, chr->get_kin_r_d(is));
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

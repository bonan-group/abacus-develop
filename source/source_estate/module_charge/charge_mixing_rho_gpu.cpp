#include "charge_mixing.h"

#if __CUDA

#include "source_io/module_parameter/parameter.h"
#include "source_base/timer.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "kernels/charge_mixing_op.h"

#include <algorithm>

void Charge_Mixing::init_mixing_gpu()
{
    // Initialize GPU mixing data if not already done
    if (rho_mdata_gpu == nullptr)
    {
        const int nspin = PARAM.inp.nspin;
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
        const std::size_t tau_length = this->rhodpw->npw * PARAM.inp.nspin;
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
        const std::size_t tau_length = this->rhodpw->npw * PARAM.inp.nspin;
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            tau_g_d, tau_length, "charge_mixing_tau_g");
        base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
            tau_g_save_d, tau_length, "charge_mixing_tau_g_save");
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
    ModuleBase::matrix& beta)
{
    if (nvec <= 0 || row < 0 || row >= nvec)
    {
        return;
    }

    const int npw = this->rhopw->npw;
#if __CUDA
    static const double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / ((*this->tpiba) * (*this->tpiba));
    elecstate::inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>()(
        nullptr,
        vectors_d + row * npw,
        vectors_d,
        this->rhopw->get_gg_d(),
        npw,
        1,
        nvec,
        this->rhopw->ig_gge0,
        fac,
        gpu_batch_result_d,
        gpu_batch_workspace_d);

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
    std::vector<double>& gamma)
{
    gamma.assign(nvec, 0.0);
    if (nvec <= 0)
    {
        return;
    }

    const int npw = this->rhopw->npw;
#if __CUDA
    static const double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / ((*this->tpiba) * (*this->tpiba));
    elecstate::inner_product_recip_hartree_batch_op<double, base_device::DEVICE_GPU>()(
        nullptr,
        vectors_d,
        rhs_d,
        this->rhopw->get_gg_d(),
        npw,
        nvec,
        1,
        this->rhopw->ig_gge0,
        fac,
        gpu_batch_result_d,
        gpu_batch_workspace_d);

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

void Charge_Mixing::mix_rho_recip_gpu(Charge* chr)
{
    ModuleBase::TITLE("Charge_Mixing", "mix_rho_recip_gpu");
    ModuleBase::timer::start("Charge_Mixing", "mix_rho_recip_gpu");

    const int nspin = PARAM.inp.nspin;

    assert(nspin == 1);
    assert(mixing_mode == "broyden" || mixing_mode == "pulay");

    // Initialize GPU mixing resources
    init_mixing_gpu();

    const int npw = this->rhopw->npw;

    // Step 1: Ensure rho is on GPU and perform FFT to get rhog
    // (This should already be done in get_drho, but we ensure it here)
    chr->sync_rho_to_device<base_device::DEVICE_GPU>();
    chr->sync_rho_save_to_device<base_device::DEVICE_GPU>();

    // FFT: rho_d -> rhog_d and rho_save_d -> rhog_save_d
    chr->rhopw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
        chr->get_rho_d(0), chr->get_rhog_d(0));
    chr->rhopw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
        chr->get_rho_save_d(0), chr->get_rhog_save_d(0));

    std::complex<double>* rhog_in_d = chr->get_rhog_save_d(0);
    std::complex<double>* rhog_out_d = chr->get_rhog_d(0);

    // Step 2: GPU Kerker screening function
    const double gg0 = std::pow(this->mixing_gg0 * ModuleBase::BOHR_TO_A / *this->tpiba, 2);
    const double gg0_min = this->mixing_gg0_min / this->mixing_beta;

    auto screen_gpu = [this, npw, gg0, gg0_min](std::complex<double>* drhog_d) {
        if (this->mixing_gg0 <= 0.0 || this->mixing_beta <= 0.1)
        {
            return;
        }
        elecstate::kerker_screen_recip_op<double, base_device::DEVICE_GPU>()(
            nullptr,  // ctx
            drhog_d,
            this->rhopw->get_gg_d(),
            gg0,
            gg0_min,
            npw,
            1);  // nspin=1
    };

    // Step 3: GPU batched inner product builders.
    auto build_beta_gpu = [this](const std::complex<double>* vectors_d,
                                  int nvec,
                                  int row,
                                  ModuleBase::matrix& beta) {
        this->build_recip_hartree_beta_row_gpu(vectors_d, nvec, row, beta);
    };
    auto build_gamma_gpu = [this](const std::complex<double>* vectors_d,
                                   const std::complex<double>* rhs_d,
                                   int nvec,
                                   std::vector<double>& gamma) {
        this->build_recip_hartree_gamma_gpu(vectors_d, rhs_d, nvec, gamma);
    };

    // Step 4-6: Mixing mode specific operations
    if (mixing_mode == "broyden")
    {
        // Broyden mixing path
        mixing_gpu->push_data(*rho_mdata_gpu, rhog_in_d, rhog_out_d, screen_gpu, true);
        mixing_gpu->cal_coef_from_beta_gamma(*rho_mdata_gpu, build_beta_gpu, build_gamma_gpu);
        mixing_gpu->mix_data(*rho_mdata_gpu, rhog_out_d);
    }
    else if (mixing_mode == "pulay")
    {
        // Pulay mixing path
        mixing_pulay_gpu->push_data(*rho_mdata_gpu, rhog_in_d, rhog_out_d, screen_gpu, true);
        mixing_pulay_gpu->cal_coef_from_beta(*rho_mdata_gpu, build_beta_gpu);
        mixing_pulay_gpu->mix_data(*rho_mdata_gpu, rhog_out_d);
    }

    const bool mix_tau = (XC_Functional::get_ked_flag()) && mixing_tau;
    if (mix_tau)
    {
        chr->sync_kin_r_to_device<base_device::DEVICE_GPU>();
        chr->sync_kin_r_save_to_device<base_device::DEVICE_GPU>();
        this->rhodpw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
            chr->get_kin_r_d(0), tau_g_d);
        this->rhodpw->real_to_recip<double, std::complex<double>, base_device::DEVICE_GPU>(
            chr->get_kin_r_save_d(0), tau_g_save_d);

        if (mixing_mode == "broyden")
        {
            mixing_gpu->push_data(*tau_mdata_gpu, tau_g_save_d, tau_g_d, nullptr, false);
            mixing_gpu->mix_data(*tau_mdata_gpu, tau_g_d);
        }
        else if (mixing_mode == "pulay")
        {
            mixing_pulay_gpu->push_data(*tau_mdata_gpu, tau_g_save_d, tau_g_d, nullptr, false);
            mixing_pulay_gpu->mix_data(*tau_mdata_gpu, tau_g_d);
        }
    }

    // Step 7: GPU FFT: rhog_d -> rho_d
    this->rhodpw->recip_to_real<std::complex<double>, double, base_device::DEVICE_GPU>(
        rhog_out_d, chr->get_rho_d(0));
    if (mix_tau)
    {
        this->rhodpw->recip_to_real<std::complex<double>, double, base_device::DEVICE_GPU>(
            tau_g_d, chr->get_kin_r_d(0));
    }

    // Step 8: Sync final result to CPU
    chr->sync_rho_to_host<base_device::DEVICE_GPU>();
    chr->sync_rhog_to_host<base_device::DEVICE_GPU>();
    if (mix_tau)
    {
        chr->sync_kin_r_to_host<base_device::DEVICE_GPU>();
    }

    ModuleBase::timer::end("Charge_Mixing", "mix_rho_recip_gpu");
}

#endif // __CUDA

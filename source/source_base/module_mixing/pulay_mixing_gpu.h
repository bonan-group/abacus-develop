#ifndef PULAY_MIXING_GPU_H_
#define PULAY_MIXING_GPU_H_

#include <complex>
#include <vector>
#include <functional>
#include <algorithm>
#include <cmath>
#include "mixing_data_gpu.h"
#include "source_base/matrix.h"
#include "source_base/module_device/types.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_external/lapack_connector.h"
#include "source_base/tool_quit.h"
#include "kernels/mixing_op.h"

namespace Base_Mixing
{

/**
 * @brief GPU-accelerated Pulay mixing (DIIS)
 *
 * This class implements the Pulay mixing method with all heavy
 * computations on GPU.
 *
 * Ref: Pulay P. Chemical Physics Letters, 1980, 73(2): 393-398.
 *
 * Key differences from Broyden:
 * - Stores F (residuals), not dF (differences)
 * - data_ndim = mixing_ndim (not mixing_ndim + 1)
 * - Uses matrix inversion for coefficients
 *
 * Formula:
 *   F = n_out - n_in
 *   alpha{ij} = <F{i}, F{j}>
 *   beta{ij} = inv(alpha){ij}
 *   coef{i} = sum_j(beta{ij}) / sum_{k,j}(beta{kj})
 *   mixing_data{i} = n_in{i} + mixing_beta*F{i}
 *   n{m+1} = sum_i(coef{i} * mixing_data{i})
 */
template <typename FPTYPE>
class Pulay_Mixing_GPU
{
  public:
    /**
     * @brief Construct with mixing parameters
     * @param mixing_ndim Number of iterations to keep in history
     * @param mixing_beta Mixing parameter beta
     */
    Pulay_Mixing_GPU(const int& mixing_ndim, const FPTYPE& mixing_beta = 0.7)
        : mixing_ndim(mixing_ndim),
          data_ndim(mixing_ndim),  // Note: NOT mixing_ndim+1 like Broyden
          mixing_beta(mixing_beta)
    {
        coef.resize(mixing_ndim);
        beta_matrix = ModuleBase::matrix(mixing_ndim, mixing_ndim, true);
    }

    ~Pulay_Mixing_GPU()
    {
        free_gpu_memory();
    }

    /**
     * @brief Initialize GPU memory for given data length
     * @param length Length of mixing vectors
     */
    void init(const std::size_t length)
    {
        if (this->length != length || F_d == nullptr)
        {
            free_gpu_memory();
            this->length = length;
            alloc_gpu_memory();
        }
    }

    /**
     * @brief Reset mixing state
     */
    void reset()
    {
        ndim_cal_F = 0;
        start_F = 0;
        address = nullptr;
    }

    /**
     * @brief Push data and compute residual on GPU
     *
     * @param mdata GPU mixing data storage
     * @param data_in_d Input data on GPU (x_in)
     * @param data_out_d Output data on GPU (x_out = f(x_in))
     * @param screen GPU screening function (e.g., Kerker)
     * @param need_calcoef Whether to prepare for coefficient calculation
     */
    void push_data(Mixing_Data_GPU<FPTYPE>& mdata,
                   const FPTYPE* data_in_d,
                   const FPTYPE* data_out_d,
                   std::function<void(FPTYPE*)> screen,
                   const bool& need_calcoef);

    /**
     * @brief Calculate mixing coefficients
     *
     * Uses inner products computed on GPU, coefficient solve on CPU.
     *
     * @param mdata GPU mixing data
     * @param inner_product_gpu GPU inner product function (returns double)
     */
    void cal_coef(const Mixing_Data_GPU<FPTYPE>& mdata,
                  std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu);

    /**
     * @brief Calculate coefficients from a batched GPU Gram-matrix builder.
     *
     * The builder receives the contiguous residual slots F_d and updates the
     * row/column for the overwritten raw slot.
     */
    void cal_coef_from_beta(
        const Mixing_Data_GPU<FPTYPE>& mdata,
        std::function<void(const FPTYPE*, int, int, ModuleBase::matrix&)> build_beta_gpu);

    /**
     * @brief Mix data using computed coefficients
     *
     * result_d = sum_i coef[i] * mdata[i]
     *
     * @param mdata GPU mixing data
     * @param result_d Output on GPU
     */
    void mix_data(const Mixing_Data_GPU<FPTYPE>& mdata, FPTYPE* result_d);

    // Public accessors
    int get_data_ndim() const { return data_ndim; }
    const std::vector<double>& get_coef() const { return coef; }

  private:
    void alloc_gpu_memory();
    void free_gpu_memory();
    void set_single_coefficient(const int slot);
    void sync_coefficients_to_device();
    void set_identity_slot_map(const int ndim_use);

  private:
    // Parameters
    int mixing_ndim = -1;      // Number of history iterations
    int data_ndim = -1;        // = mixing_ndim (NOT mixing_ndim+1 like Broyden)
    FPTYPE mixing_beta = 0.7;  // Mixing parameter
    std::size_t length = 0;    // Vector length

    // GPU memory
    FPTYPE* F_d = nullptr;     // F vectors [mixing_ndim * length]
    FPTYPE* temp_d = nullptr;  // Temporary buffer [length]
    FPTYPE* workspace_d = nullptr; // Workspace for inner products
    FPTYPE* coef_d = nullptr;  // Coefficients mirrored to GPU for final mix

    // CPU memory for coefficient calculation
    ModuleBase::matrix beta_matrix;  // Inner product matrix <F_i, F_j>
    std::vector<double> coef;        // Mixing coefficients
    std::vector<int> coef_to_gpu_slot; // CPU coefficient index -> GPU history slot

    // State
    Mixing_Data_GPU<FPTYPE>* address = nullptr;
    int start_F = 0;       // Current position in F circular buffer
    int ndim_cal_F = 0;    // Number of F vectors stored (up to mixing_ndim)
};

//==========================================================
// Implementation
//==========================================================

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::alloc_gpu_memory()
{
    if (length > 0)
    {
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            F_d, mixing_ndim * length, "Pulay_F");
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            temp_d, length, "Pulay_temp");
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            coef_d, mixing_ndim, "Pulay_coef");

        // Workspace for reductions (need enough blocks)
        const int max_blocks = (length + 255) / 256;
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            workspace_d, 2 * max_blocks, "Pulay_workspace");
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::free_gpu_memory()
{
    if (F_d != nullptr)
    {
        base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(F_d);
        F_d = nullptr;
    }
    if (temp_d != nullptr)
    {
        base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(temp_d);
        temp_d = nullptr;
    }
    if (workspace_d != nullptr)
    {
        base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(workspace_d);
        workspace_d = nullptr;
    }
    if (coef_d != nullptr)
    {
        base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(coef_d);
        coef_d = nullptr;
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::set_single_coefficient(const int slot)
{
    std::fill(coef.begin(), coef.end(), 0.0);
    if (slot >= 0 && slot < static_cast<int>(coef.size()))
    {
        coef[slot] = 1.0;
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::set_identity_slot_map(const int ndim_use)
{
    coef_to_gpu_slot.resize(ndim_use);
    for (int i = 0; i < ndim_use; ++i)
    {
        coef_to_gpu_slot[i] = i;
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::sync_coefficients_to_device()
{
    std::vector<FPTYPE> coef_fp(coef.size());
    for (std::size_t i = 0; i < coef.size(); ++i)
    {
        coef_fp[i] = static_cast<FPTYPE>(coef[i]);
    }
    base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
        coef_d, coef_fp.data(), coef_fp.size());
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::push_data(
    Mixing_Data_GPU<FPTYPE>& mdata,
    const FPTYPE* data_in_d,
    const FPTYPE* data_out_d,
    std::function<void(FPTYPE*)> screen,
    const bool& need_calcoef)
{
    const std::size_t len = mdata.length;
    const base_device::DEVICE_GPU* ctx = nullptr;
    const int len_i = static_cast<int>(len);

    // F = data_out - data_in (on GPU)
    mixing::vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, temp_d, data_out_d, data_in_d, len_i);

    // Apply Kerker screening (on GPU)
    if (screen != nullptr)
    {
        screen(temp_d);
    }

    // mixed = data_in + mixing_beta * F (on GPU)
    // Store in temp buffer first, then push to mdata
    mixing::vector_axpy_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, temp_d, data_in_d, mixing_beta, temp_d, len_i);

    // Push mixed data to history
    mdata.push(temp_d);

    if (!need_calcoef)
        return;

    // Verify we're bound to this mdata
    if (address != &mdata && address != nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "Pulay_Mixing",
            "One Pulay_Mixing object can only bind one Mixing_Data object to calculate coefficients");
    }

    // Recompute F = data_out - data_in for coefficient calculation
    mixing::vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, temp_d, data_out_d, data_in_d, len_i);

    if (screen != nullptr)
    {
        screen(temp_d);
    }

    if (mdata.ndim_use == 1)
    {
        address = &mdata;
        start_F = mdata.start;
        ndim_cal_F = 1;
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_d + start_F * len, temp_d, len_i);
    }
    else
    {
        start_F = mdata.start;
        ndim_cal_F = std::min(ndim_cal_F + 1, mixing_ndim);
        FPTYPE* F_slot = F_d + start_F * len;
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_slot, temp_d, len_i);
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::cal_coef(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu)
{
    if (address != &mdata && address != nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "Pulay_mixing",
            "One Pulay_Mixing object can only bind one Mixing_Data object to calculate coefficients");
    }

    const std::size_t len = mdata.length;
    const int ndim_use = mdata.ndim_use;

    if (ndim_use > 1)
    {
        ModuleBase::matrix beta_tmp(ndim_use, ndim_use);
        set_identity_slot_map(ndim_use);

        for (int i = 0; i < ndim_use; ++i)
        {
            const FPTYPE* Fi = F_d + i * len;
            for (int j = i; j < ndim_use; ++j)
            {
                if (i != start_F && j != start_F)
                {
                    beta_tmp(i, j) = beta_matrix(i, j);
                }
                else
                {
                    const FPTYPE* Fj = F_d + j * len;
                    double result = inner_product_gpu(Fi, Fj);
                    beta_matrix(i, j) = result;
                    beta_tmp(i, j) = result;
                }
                if (j != i)
                {
                    beta_tmp(j, i) = beta_tmp(i, j);
                }
            }
        }

        // Invert beta matrix on CPU using LAPACK
        // dsytrf: Factorize symmetric matrix
        // dsytri: Compute inverse from factorization
        double* work = new double[ndim_use];
        int* iwork = new int[ndim_use];
        char uu = 'U';
        int info = 0;

        dsytrf_(&uu, &ndim_use, beta_tmp.c, &ndim_use, iwork, work, &ndim_use, &info);
        if (info != 0)
        {
            delete[] work;
            delete[] iwork;
            ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when factorizing beta.");
        }

        dsytri_(&uu, &ndim_use, beta_tmp.c, &ndim_use, iwork, work, &info);
        if (info != 0)
        {
            delete[] work;
            delete[] iwork;
            ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when DSYTRI beta.");
        }

        // Fill lower triangle from upper (symmetric)
        for (int i = 0; i < ndim_use; ++i)
        {
            for (int j = i + 1; j < ndim_use; ++j)
            {
                beta_tmp(i, j) = beta_tmp(j, i);
            }
        }

        // Compute normalized coefficients
        // coef[i] = sum_j(beta_inv(i,j)) / sum_{k,j}(beta_inv(k,j))
        double sum_all = 0.0;
        for (int k = 0; k < ndim_use; ++k)
        {
            for (int j = 0; j < ndim_use; ++j)
            {
                sum_all += beta_tmp(k, j);
            }
        }
        std::fill(coef.begin(), coef.end(), 0.0);
        for (int i = 0; i < ndim_use; ++i)
        {
            double sum_row = 0.0;
            for (int j = 0; j < ndim_use; ++j)
            {
                sum_row += beta_tmp(i, j);
            }
            coef[coef_to_gpu_slot[i]] = sum_row / sum_all;
        }

        delete[] work;
        delete[] iwork;
        sync_coefficients_to_device();
    }
    else
    {
        beta_matrix(0, 0) = inner_product_gpu(F_d + start_F * len, F_d + start_F * len);
        set_single_coefficient(mdata.start);
        sync_coefficients_to_device();
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::cal_coef_from_beta(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    std::function<void(const FPTYPE*, int, int, ModuleBase::matrix&)> build_beta_gpu)
{
    if (address != &mdata && address != nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "Pulay_mixing",
            "One Pulay_Mixing object can only bind one Mixing_Data object to calculate coefficients");
    }

    const int ndim_use = mdata.ndim_use;

    if (ndim_use > 1)
    {
        ModuleBase::matrix beta_tmp(ndim_use, ndim_use);
        set_identity_slot_map(ndim_use);

        for (int i = 0; i < ndim_use; ++i)
        {
            for (int j = i; j < ndim_use; ++j)
            {
                if (i != start_F && j != start_F)
                {
                    beta_tmp(i, j) = beta_matrix(i, j);
                }
                if (j != i)
                {
                    beta_tmp(j, i) = beta_tmp(i, j);
                }
            }
        }

        build_beta_gpu(F_d, ndim_use, start_F, beta_tmp);

        for (int i = 0; i < ndim_use; ++i)
        {
            beta_matrix(start_F, i) = beta_tmp(start_F, i);
            beta_matrix(i, start_F) = beta_tmp(i, start_F);
        }
        for (int i = 0; i < ndim_use; ++i)
        {
            for (int j = i + 1; j < ndim_use; ++j)
            {
                beta_tmp(j, i) = beta_tmp(i, j);
            }
        }

        double* work = new double[ndim_use];
        int* iwork = new int[ndim_use];
        char uu = 'U';
        int info = 0;

        dsytrf_(&uu, &ndim_use, beta_tmp.c, &ndim_use, iwork, work, &ndim_use, &info);
        if (info != 0)
        {
            delete[] work;
            delete[] iwork;
            ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when factorizing beta.");
        }

        dsytri_(&uu, &ndim_use, beta_tmp.c, &ndim_use, iwork, work, &info);
        if (info != 0)
        {
            delete[] work;
            delete[] iwork;
            ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when DSYTRI beta.");
        }

        for (int i = 0; i < ndim_use; ++i)
        {
            for (int j = i + 1; j < ndim_use; ++j)
            {
                beta_tmp(i, j) = beta_tmp(j, i);
            }
        }

        double sum_all = 0.0;
        for (int k = 0; k < ndim_use; ++k)
        {
            for (int j = 0; j < ndim_use; ++j)
            {
                sum_all += beta_tmp(k, j);
            }
        }
        std::fill(coef.begin(), coef.end(), 0.0);
        for (int i = 0; i < ndim_use; ++i)
        {
            double sum_row = 0.0;
            for (int j = 0; j < ndim_use; ++j)
            {
                sum_row += beta_tmp(i, j);
            }
            coef[coef_to_gpu_slot[i]] = sum_row / sum_all;
        }

        delete[] work;
        delete[] iwork;
        sync_coefficients_to_device();
    }
    else
    {
        set_single_coefficient(mdata.start);
        sync_coefficients_to_device();
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::mix_data(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    FPTYPE* result_d)
{
    const std::size_t len = mdata.length;
    const base_device::DEVICE_GPU* ctx = nullptr;
    const int ndim_use = mdata.ndim_use;
    const int len_i = static_cast<int>(len);

    if (len == 0 || ndim_use <= 0)
    {
        return;
    }
    if (ndim_use == 1)
    {
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, result_d, mdata.get_data_d(0), len_i);
        return;
    }

    mixing::gemv_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, 'N', len_i, mdata.ndim_tot, static_cast<FPTYPE>(1.0), mdata.data_d, len_i,
        coef_d, 1, static_cast<FPTYPE>(0.0), result_d, 1);
}

// Explicit instantiations
template class Pulay_Mixing_GPU<double>;
template class Pulay_Mixing_GPU<float>;
template class Pulay_Mixing_GPU<std::complex<double>>;
template class Pulay_Mixing_GPU<std::complex<float>>;

} // namespace Base_Mixing

#endif // PULAY_MIXING_GPU_H_

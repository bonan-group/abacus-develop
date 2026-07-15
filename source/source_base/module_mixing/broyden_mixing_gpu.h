#ifndef BROYDEN_MIXING_GPU_H_
#define BROYDEN_MIXING_GPU_H_

#include <complex>
#include <vector>
#include <functional>
#include <algorithm>
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
 * @brief GPU-accelerated Broyden mixing
 *
 * This class implements the simplified modified Broyden mixing method
 * with all heavy computations on GPU.
 *
 * Key operations on GPU:
 * - Vector subtraction (F = out - in)
 * - AXPY operations (mixed = in + beta * F)
 * - Inner products for coefficient calculation
 * - Final mixing GEMV
 *
 * Coefficient calculation (small matrix solve) stays on CPU.
 */
template <typename FPTYPE>
class Broyden_Mixing_GPU
{
  public:
    /**
     * @brief Construct with mixing parameters
     * @param mixing_ndim Number of iterations to keep in history
     * @param mixing_beta Mixing parameter beta
     */
    Broyden_Mixing_GPU(const int& mixing_ndim)
        : Broyden_Mixing_GPU(mixing_ndim, static_cast<FPTYPE>(0.7))
    {
    }

    Broyden_Mixing_GPU(const int& mixing_ndim, const FPTYPE& mixing_beta)
        : mixing_ndim(mixing_ndim),
          data_ndim(mixing_ndim + 1),
          mixing_beta(mixing_beta)
    {
        coef.resize(mixing_ndim + 1);
        beta_matrix = ModuleBase::matrix(mixing_ndim, mixing_ndim, true);
    }

    ~Broyden_Mixing_GPU()
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
        ndim_cal_dF = 0;
        start_dF = -1;
        address = nullptr;
    }

    /**
     * @brief Push data and compute residual on GPU
     *
     * @param mdata GPU mixing data storage
     * @param data_in_d Input data on GPU (x_in)
     * @param data_out_d Output data on GPU (x_out = f(x_in))
     * @param screen GPU screening function (e.g., Kerker)
     * @param mix Optional component-aware mixing function
     * @param need_calcoef Whether to prepare for coefficient calculation
     */
    void push_data(Mixing_Data_GPU<FPTYPE>& mdata,
                   const FPTYPE* data_in_d,
                   const FPTYPE* data_out_d,
                   std::function<void(FPTYPE*)> screen,
                   std::function<void(FPTYPE*, const FPTYPE*, const FPTYPE*)> mix,
                   const bool& need_calcoef);

    /**
     * @brief Calculate mixing coefficients
     *
     * Uses inner products computed on GPU, coefficient solve on CPU.
     *
     * @param mdata GPU mixing data
     * @param inner_product_gpu GPU inner product function (returns double, not FPTYPE)
     */
    void cal_coef(const Mixing_Data_GPU<FPTYPE>& mdata,
                  std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu);

    /**
     * @brief Calculate coefficients from batched GPU beta/gamma builders.
     *
     * The beta builder updates the overwritten dF row/column in raw slot order.
     * The gamma builder fills gamma[i]=<dF_i,F> for active raw dF slots.
     */
    void cal_coef_from_beta_gamma(
        const Mixing_Data_GPU<FPTYPE>& mdata,
        std::function<void(const FPTYPE*, int, int, ModuleBase::matrix&)> build_beta_gpu,
        std::function<void(const FPTYPE*, const FPTYPE*, int, std::vector<double>&)> build_gamma_gpu);

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
    void use_latest_coefficients(const Mixing_Data_GPU<FPTYPE>& mdata);
    void sync_coefficients_to_device();
    void set_identity_slot_map(const int ndim_use);

    int dFindex_move(const int& index) const
    {
        return (start_dF + index + mixing_ndim) % mixing_ndim;
    }

  private:
    // Parameters
    int mixing_ndim = -1;      // Number of history iterations
    int data_ndim = -1;        // mixing_ndim + 1
    FPTYPE mixing_beta = 0.7;  // Mixing parameter
    std::size_t length = 0;    // Vector length

    // GPU memory
    FPTYPE* F_d = nullptr;     // Current residual [length]
    FPTYPE* dF_d = nullptr;    // Residual differences [mixing_ndim * length]
    FPTYPE* temp_d = nullptr;  // Temporary buffer [length]
    FPTYPE* workspace_d = nullptr; // Workspace for inner products
    FPTYPE* coef_d = nullptr;  // Coefficients mirrored to GPU for final mix

    // CPU memory for coefficient calculation
    ModuleBase::matrix beta_matrix;  // Inner product matrix
    std::vector<double> coef;        // Mixing coefficients
    std::vector<int> coef_to_gpu_slot; // CPU coefficient index -> GPU history slot

    // State
    Mixing_Data_GPU<FPTYPE>* address = nullptr;
    int start_dF = -1;
    int ndim_cal_dF = 0;
};

//==========================================================
// Implementation
//==========================================================

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::alloc_gpu_memory()
{
    if (length > 0)
    {
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            F_d, length, "Broyden_F");
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            dF_d, mixing_ndim * length, "Broyden_dF");
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            temp_d, length, "Broyden_temp");
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            coef_d, data_ndim, "Broyden_coef");

        // Workspace for reductions (need enough blocks)
        const int max_blocks = (length + 255) / 256;
        base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
            workspace_d, 2 * max_blocks, "Broyden_workspace");
    }
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::free_gpu_memory()
{
    if (F_d != nullptr)
    {
        base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(F_d);
        F_d = nullptr;
    }
    if (dF_d != nullptr)
    {
        base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(dF_d);
        dF_d = nullptr;
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
void Broyden_Mixing_GPU<FPTYPE>::use_latest_coefficients(const Mixing_Data_GPU<FPTYPE>& mdata)
{
    std::fill(coef.begin(), coef.end(), 0.0);
    if (mdata.start >= 0 && mdata.start < static_cast<int>(coef.size()))
    {
        coef[mdata.start] = 1.0;
    }
    else if (!coef.empty())
    {
        coef[0] = 1.0;
    }
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::set_identity_slot_map(const int ndim_use)
{
    coef_to_gpu_slot.resize(ndim_use);
    for (int i = 0; i < ndim_use; ++i)
    {
        coef_to_gpu_slot[i] = i;
    }
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::sync_coefficients_to_device()
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
void Broyden_Mixing_GPU<FPTYPE>::push_data(
    Mixing_Data_GPU<FPTYPE>& mdata,
    const FPTYPE* data_in_d,
    const FPTYPE* data_out_d,
    std::function<void(FPTYPE*)> screen,
    std::function<void(FPTYPE*, const FPTYPE*, const FPTYPE*)> mix,
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

    if (mix != nullptr)
    {
        mix(temp_d, data_in_d, temp_d);
    }
    else
    {
        mixing::vector_axpy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, temp_d, data_in_d, mixing_beta, temp_d, len_i);
    }

    // Push mixed data to history
    mdata.push(temp_d);

    if (!need_calcoef)
        return;

    // Verify we're bound to this mdata
    if (address != &mdata && address != nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "Broyden_Mixing",
            "One Broyden_Mixing object can only bind one Mixing_Data object to calculate coefficients");
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
        // First iteration: just store F
        address = &mdata;
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_d, temp_d, len_i);
    }
    else
    {
        // Compute dF = F_prev - F_curr = -(F_curr - F_prev)
        ndim_cal_dF = std::min(ndim_cal_dF + 1, mixing_ndim);
        start_dF = (start_dF + 1) % mixing_ndim;

        FPTYPE* dF_slot = dF_d + start_dF * len;

        // dF = F_prev - F_curr (stored as dF = old_F - new_F)
        mixing::vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, dF_slot, F_d, temp_d, len_i);

        // Update F for next iteration
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_d, temp_d, len_i);
    }
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::cal_coef(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu)
{
    if (address != &mdata && address != nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "Broyden_mixing",
            "One Broyden_Mixing object can only bind one Mixing_Data object to calculate coefficients");
    }

    const std::size_t len = mdata.length;

    if (ndim_cal_dF > 0)
    {
        ModuleBase::matrix beta_tmp(ndim_cal_dF, ndim_cal_dF);

        // Compute beta(i, j) = <dF_i, dF_j> using GPU inner products
        for (int i = 0; i < ndim_cal_dF; ++i)
        {
            const FPTYPE* dFi = dF_d + i * len;
            for (int j = i; j < ndim_cal_dF; ++j)
            {
                if (i != start_dF && j != start_dF)
                {
                    // Reuse cached value
                    beta_tmp(i, j) = beta_matrix(i, j);
                }
                else
                {
                    // Compute new inner product on GPU
                    const FPTYPE* dFj = dF_d + j * len;
                    double result = inner_product_gpu(dFi, dFj);
                    beta_matrix(i, j) = result;
                    beta_tmp(i, j) = result;
                }
                if (j != i)
                {
                    beta_tmp(j, i) = beta_tmp(i, j);
                }
            }
        }

        // Compute gamma = <dF_i, F> for all i
        std::vector<double> gamma(ndim_cal_dF);
        for (int i = 0; i < ndim_cal_dF; ++i)
        {
            const FPTYPE* dFi = dF_d + i * len;
            gamma[i] = inner_product_gpu(dFi, F_d);
        }

        // Solve beta * gamma = c on CPU (small matrix)
        double* work = new double[ndim_cal_dF];
        int* iwork = new int[ndim_cal_dF];
        char uu = 'U';
        int info = 0;
        int m = 1;

        // Use LAPACK dsysv
        dsysv_(&uu, &ndim_cal_dF, &m, beta_tmp.c, &ndim_cal_dF, iwork,
               gamma.data(), &ndim_cal_dF, work, &ndim_cal_dF, &info);
        if (info != 0)
        {
            delete[] work;
            delete[] iwork;
            ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when DSYSV.");
        }

        // Compute final coefficients
        std::fill(coef.begin(), coef.end(), 0.0);
        set_identity_slot_map(mdata.ndim_use);
        coef[mdata.start] = 1.0 + gamma[dFindex_move(0)];
        for (int i = 1; i < ndim_cal_dF; ++i)
        {
            coef[mdata.index_move(-i)] = gamma[dFindex_move(-i)] - gamma[dFindex_move(-i + 1)];
        }
        coef[mdata.index_move(-ndim_cal_dF)] = -gamma[dFindex_move(-ndim_cal_dF + 1)];

        delete[] work;
        delete[] iwork;
        sync_coefficients_to_device();
    }
    else
    {
        use_latest_coefficients(mdata);
        set_identity_slot_map(mdata.ndim_use);
        sync_coefficients_to_device();
    }

    // Prepare dF for next iteration
    FPTYPE* dFnext = dF_d + dFindex_move(1) * len;
    const base_device::DEVICE_GPU* ctx = nullptr;
    mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, dFnext, F_d, static_cast<int>(len));
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::cal_coef_from_beta_gamma(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    std::function<void(const FPTYPE*, int, int, ModuleBase::matrix&)> build_beta_gpu,
    std::function<void(const FPTYPE*, const FPTYPE*, int, std::vector<double>&)> build_gamma_gpu)
{
    if (address != &mdata && address != nullptr)
    {
        ModuleBase::WARNING_QUIT(
            "Broyden_mixing",
            "One Broyden_Mixing object can only bind one Mixing_Data object to calculate coefficients");
    }

    const std::size_t len = mdata.length;

    if (ndim_cal_dF > 0)
    {
        ModuleBase::matrix beta_tmp(ndim_cal_dF, ndim_cal_dF);

        for (int i = 0; i < ndim_cal_dF; ++i)
        {
            for (int j = i; j < ndim_cal_dF; ++j)
            {
                if (i != start_dF && j != start_dF)
                {
                    beta_tmp(i, j) = beta_matrix(i, j);
                }
                if (j != i)
                {
                    beta_tmp(j, i) = beta_tmp(i, j);
                }
            }
        }

        build_beta_gpu(dF_d, ndim_cal_dF, start_dF, beta_tmp);
        for (int i = 0; i < ndim_cal_dF; ++i)
        {
            beta_matrix(start_dF, i) = beta_tmp(start_dF, i);
            beta_matrix(i, start_dF) = beta_tmp(i, start_dF);
        }
        for (int i = 0; i < ndim_cal_dF; ++i)
        {
            for (int j = i + 1; j < ndim_cal_dF; ++j)
            {
                beta_tmp(j, i) = beta_tmp(i, j);
            }
        }

        std::vector<double> gamma(ndim_cal_dF);
        build_gamma_gpu(dF_d, F_d, ndim_cal_dF, gamma);

        double* work = new double[ndim_cal_dF];
        int* iwork = new int[ndim_cal_dF];
        char uu = 'U';
        int info = 0;
        int m = 1;

        dsysv_(&uu, &ndim_cal_dF, &m, beta_tmp.c, &ndim_cal_dF, iwork,
               gamma.data(), &ndim_cal_dF, work, &ndim_cal_dF, &info);
        if (info != 0)
        {
            delete[] work;
            delete[] iwork;
            ModuleBase::WARNING_QUIT("Charge_Mixing", "Error when DSYSV.");
        }

        std::fill(coef.begin(), coef.end(), 0.0);
        set_identity_slot_map(mdata.ndim_use);
        coef[mdata.start] = 1.0 + gamma[dFindex_move(0)];
        for (int i = 1; i < ndim_cal_dF; ++i)
        {
            coef[mdata.index_move(-i)] = gamma[dFindex_move(-i)] - gamma[dFindex_move(-i + 1)];
        }
        coef[mdata.index_move(-ndim_cal_dF)] = -gamma[dFindex_move(-ndim_cal_dF + 1)];

        delete[] work;
        delete[] iwork;
        sync_coefficients_to_device();
    }
    else
    {
        use_latest_coefficients(mdata);
        set_identity_slot_map(mdata.ndim_use);
        sync_coefficients_to_device();
    }

    FPTYPE* dFnext = dF_d + dFindex_move(1) * len;
    const base_device::DEVICE_GPU* ctx = nullptr;
    mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, dFnext, F_d, static_cast<int>(len));
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::mix_data(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    FPTYPE* result_d)
{
    const std::size_t len = mdata.length;
    const base_device::DEVICE_GPU* ctx = nullptr;
    const int len_i = static_cast<int>(len);

    if (len == 0 || mdata.ndim_use <= 0)
    {
        return;
    }
    if (mdata.ndim_use == 1)
    {
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, result_d, mdata.get_data_d(0), len_i);
        return;
    }

    mixing::gemv_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, 'N', len_i, mdata.ndim_tot, static_cast<FPTYPE>(1.0), mdata.data_d,
        len_i, coef_d, 1, static_cast<FPTYPE>(0.0), result_d, 1);
}

// Explicit instantiations
template class Broyden_Mixing_GPU<double>;
template class Broyden_Mixing_GPU<float>;
template class Broyden_Mixing_GPU<std::complex<double>>;
template class Broyden_Mixing_GPU<std::complex<float>>;

} // namespace Base_Mixing

#endif // BROYDEN_MIXING_GPU_H_

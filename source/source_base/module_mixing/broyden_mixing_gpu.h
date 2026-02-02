#ifndef BROYDEN_MIXING_GPU_H_
#define BROYDEN_MIXING_GPU_H_

#include <complex>
#include <vector>
#include <functional>
#include "mixing_data_gpu.h"
#include "source_base/matrix.h"
#include "source_base/module_device/types.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_external/lapack_connector.h"
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
    Broyden_Mixing_GPU(const int& mixing_ndim, const FPTYPE& mixing_beta = 0.7)
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
     * @param inner_product_gpu GPU inner product function (returns double, not FPTYPE)
     */
    void cal_coef(const Mixing_Data_GPU<FPTYPE>& mdata,
                  std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu);

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

    // CPU memory for coefficient calculation
    ModuleBase::matrix beta_matrix;  // Inner product matrix
    std::vector<double> coef;        // Mixing coefficients

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
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::push_data(
    Mixing_Data_GPU<FPTYPE>& mdata,
    const FPTYPE* data_in_d,
    const FPTYPE* data_out_d,
    std::function<void(FPTYPE*)> screen,
    const bool& need_calcoef)
{
    const std::size_t len = mdata.length;
    const base_device::DEVICE_GPU* ctx = nullptr;

    // F = data_out - data_in (on GPU)
    mixing::vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, temp_d, data_out_d, data_in_d, static_cast<int>(len));

    // Apply Kerker screening (on GPU)
    if (screen != nullptr)
    {
        screen(temp_d);
    }

    // mixed = data_in + mixing_beta * F (on GPU)
    // Store in temp buffer first, then push to mdata
    mixing::vector_axpy_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, temp_d, data_in_d, mixing_beta, temp_d, static_cast<int>(len));

    // Push mixed data to history
    mdata.push(temp_d);

    if (!need_calcoef)
        return;

    // Verify we're bound to this mdata
    if (address != &mdata && address != nullptr)
    {
        // Error: trying to use with different mdata
        return;
    }

    // Recompute F = data_out - data_in for coefficient calculation
    mixing::vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, temp_d, data_out_d, data_in_d, static_cast<int>(len));

    if (screen != nullptr)
    {
        screen(temp_d);
    }

    if (mdata.ndim_use == 1)
    {
        // First iteration: just store F
        address = &mdata;
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_d, temp_d, static_cast<int>(len));
    }
    else
    {
        // Compute dF = F_prev - F_curr = -(F_curr - F_prev)
        ndim_cal_dF = std::min(ndim_cal_dF + 1, mixing_ndim);
        start_dF = (start_dF + 1) % mixing_ndim;

        FPTYPE* dF_slot = dF_d + start_dF * len;

        // dF = F_prev - F_curr (stored as dF = old_F - new_F)
        mixing::vector_subtract_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, dF_slot, F_d, temp_d, static_cast<int>(len));

        // Update F for next iteration
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_d, temp_d, static_cast<int>(len));
    }
}

template <typename FPTYPE>
void Broyden_Mixing_GPU<FPTYPE>::cal_coef(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu)
{
    if (address != &mdata && address != nullptr)
    {
        return;
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

        // Compute final coefficients
        coef[mdata.start] = 1.0 + gamma[dFindex_move(0)];
        for (int i = 1; i < ndim_cal_dF; ++i)
        {
            coef[mdata.index_move(-i)] = gamma[dFindex_move(-i)] - gamma[dFindex_move(-i + 1)];
        }
        coef[mdata.index_move(-ndim_cal_dF)] = -gamma[dFindex_move(-ndim_cal_dF + 1)];

        delete[] work;
        delete[] iwork;
    }
    else
    {
        coef[0] = 1.0;
    }

    // Prepare dF for next iteration
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

    // result = sum_i coef[i] * mdata[i]
    // Start with coef[start] * mdata[start]
    const FPTYPE* first_data = mdata.get_data_d(0);

    // Initialize result = coef[0] * mdata[0]
    mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, result_d, first_data, static_cast<int>(len));
    mixing::vector_scale_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, result_d, static_cast<FPTYPE>(coef[mdata.start]), static_cast<int>(len));

    // Add contributions from other history vectors
    for (int i = 1; i < mdata.ndim_use; ++i)
    {
        int idx = mdata.index_move(-i);
        const FPTYPE* hist_data = mdata.data_d + idx * len;

        // result += coef[idx] * hist_data
        // Using temp buffer for scaled version, then add
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, temp_d, hist_data, static_cast<int>(len));
        mixing::vector_scale_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, temp_d, static_cast<FPTYPE>(coef[idx]), static_cast<int>(len));

        // result += temp (no direct add kernel, so use axpy with alpha=1)
        mixing::vector_axpy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, result_d, result_d, static_cast<FPTYPE>(1.0), temp_d, static_cast<int>(len));
    }
}

// Explicit instantiations
template class Broyden_Mixing_GPU<double>;
template class Broyden_Mixing_GPU<float>;
template class Broyden_Mixing_GPU<std::complex<double>>;
template class Broyden_Mixing_GPU<std::complex<float>>;

} // namespace Base_Mixing

#endif // BROYDEN_MIXING_GPU_H_

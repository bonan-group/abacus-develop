#ifndef PULAY_MIXING_GPU_H_
#define PULAY_MIXING_GPU_H_

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

    // Map from local F index to storage index (circular buffer)
    int Findex(const int& index) const
    {
        return (start_F + index + mixing_ndim) % mixing_ndim;
    }

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

    // CPU memory for coefficient calculation
    ModuleBase::matrix beta_matrix;  // Inner product matrix <F_i, F_j>
    std::vector<double> coef;        // Mixing coefficients

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
        // First iteration: just store F at position 0
        address = &mdata;
        start_F = 0;
        ndim_cal_F = 1;
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_d, temp_d, static_cast<int>(len));
    }
    else
    {
        // Store F in circular buffer
        // Note: Unlike Broyden which computes dF=F_old-F_new,
        // Pulay just stores F directly
        start_F = (start_F + 1) % mixing_ndim;
        ndim_cal_F = std::min(ndim_cal_F + 1, mixing_ndim);

        FPTYPE* F_slot = F_d + start_F * len;
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, F_slot, temp_d, static_cast<int>(len));
    }
}

template <typename FPTYPE>
void Pulay_Mixing_GPU<FPTYPE>::cal_coef(
    const Mixing_Data_GPU<FPTYPE>& mdata,
    std::function<double(const FPTYPE*, const FPTYPE*)> inner_product_gpu)
{
    if (address != &mdata && address != nullptr)
    {
        return;
    }

    const std::size_t len = mdata.length;
    const int ndim_use = mdata.ndim_use;

    if (ndim_use > 1)
    {
        ModuleBase::matrix beta_tmp(ndim_use, ndim_use);

        // Compute beta(i, j) = <F_i, F_j> using GPU inner products
        // Note: Pulay uses <F, F> while Broyden uses <dF, dF>
        for (int i = 0; i < ndim_use; ++i)
        {
            // Map from ndim_use index to F storage index
            // i=0 is most recent (at start_F), i=1 is previous, etc.
            int idx_i = (start_F - i + mixing_ndim) % mixing_ndim;
            const FPTYPE* Fi = F_d + idx_i * len;

            for (int j = i; j < ndim_use; ++j)
            {
                if (i != 0 && j != 0)
                {
                    // Reuse cached value (only new F at position 0)
                    beta_tmp(i, j) = beta_matrix(i, j);
                }
                else
                {
                    // Compute new inner product on GPU
                    int idx_j = (start_F - j + mixing_ndim) % mixing_ndim;
                    const FPTYPE* Fj = F_d + idx_j * len;
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
            // Matrix factorization failed, fall back to simple mixing
            for (int i = 0; i < ndim_use; ++i)
            {
                coef[i] = (i == 0) ? 1.0 : 0.0;
            }
            delete[] work;
            delete[] iwork;
            return;
        }

        dsytri_(&uu, &ndim_use, beta_tmp.c, &ndim_use, iwork, work, &info);
        if (info != 0)
        {
            // Matrix inversion failed, fall back to simple mixing
            for (int i = 0; i < ndim_use; ++i)
            {
                coef[i] = (i == 0) ? 1.0 : 0.0;
            }
            delete[] work;
            delete[] iwork;
            return;
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

        // Map coefficients to mdata indices
        // coef[i] corresponds to mdata.index_move(-i)
        for (int i = 0; i < ndim_use; ++i)
        {
            double sum_row = 0.0;
            for (int j = 0; j < ndim_use; ++j)
            {
                sum_row += beta_tmp(i, j);
            }
            coef[i] = sum_row / sum_all;
        }

        delete[] work;
        delete[] iwork;
    }
    else
    {
        // First iteration: just use coefficient 1.0
        coef[0] = 1.0;
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

    // result = sum_i coef[i] * mdata[i]
    // Note: coef[i] corresponds to mdata at position -i (0=most recent)

    // Start with coef[0] * mdata[0] (most recent)
    const FPTYPE* first_data = mdata.get_data_d(0);

    // Initialize result = coef[0] * mdata[0]
    mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, result_d, first_data, static_cast<int>(len));
    mixing::vector_scale_op<FPTYPE, base_device::DEVICE_GPU>()(
        ctx, result_d, static_cast<FPTYPE>(coef[0]), static_cast<int>(len));

    // Add contributions from other history vectors
    for (int i = 1; i < ndim_use; ++i)
    {
        int idx = mdata.index_move(-i);
        const FPTYPE* hist_data = mdata.data_d + idx * len;

        // result += coef[i] * hist_data
        // Using temp buffer for scaled version, then add
        mixing::vector_copy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, temp_d, hist_data, static_cast<int>(len));
        mixing::vector_scale_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, temp_d, static_cast<FPTYPE>(coef[i]), static_cast<int>(len));

        // result += temp (no direct add kernel, so use axpy with alpha=1)
        mixing::vector_axpy_op<FPTYPE, base_device::DEVICE_GPU>()(
            ctx, result_d, result_d, static_cast<FPTYPE>(1.0), temp_d, static_cast<int>(len));
    }
}

// Explicit instantiations
template class Pulay_Mixing_GPU<double>;
template class Pulay_Mixing_GPU<float>;
template class Pulay_Mixing_GPU<std::complex<double>>;
template class Pulay_Mixing_GPU<std::complex<float>>;

} // namespace Base_Mixing

#endif // PULAY_MIXING_GPU_H_

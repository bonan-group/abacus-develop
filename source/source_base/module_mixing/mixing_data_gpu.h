#ifndef MIXING_DATA_GPU_H_
#define MIXING_DATA_GPU_H_

#include <cstddef>
#include "source_base/module_device/types.h"
#include "source_base/module_device/memory_op.h"

namespace Base_Mixing
{

/**
 * @brief GPU-resident mixing data for full GPU mixing path
 *
 * This class manages mixing history data directly on GPU memory,
 * eliminating CPU-GPU transfers during mixing iterations.
 * The data is organized as a circular buffer similar to the CPU Mixing_Data class.
 */
template <typename FPTYPE>
class Mixing_Data_GPU
{
  public:
    Mixing_Data_GPU() = default;

    /**
     * @brief Construct and allocate GPU memory
     * @param ndim Number of vectors to store for mixing (history depth)
     * @param length Length of each vector
     */
    Mixing_Data_GPU(const int& ndim, const std::size_t& length)
    {
        resize(ndim, length);
    }

    /**
     * @brief Destroy and free GPU memory
     */
    ~Mixing_Data_GPU()
    {
        if (data_d != nullptr)
        {
            base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(data_d);
            data_d = nullptr;
        }
    }

    /**
     * @brief Resize/reallocate GPU memory
     * @param ndim Number of vectors to store for mixing
     * @param length Length of each vector
     */
    void resize(const int& ndim, const std::size_t& length)
    {
        // Free existing memory if any
        if (data_d != nullptr)
        {
            base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(data_d);
            data_d = nullptr;
        }

        // Reset state
        this->ndim_tot = ndim;
        this->length = length;
        this->start = -1;
        this->ndim_use = 0;
        this->ndim_history = 0;

        // Allocate GPU memory
        if (ndim > 0 && length > 0)
        {
            base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
                data_d, ndim * length, "Mixing_Data_GPU");

            // Zero-initialize
            base_device::memory::set_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
                data_d, 0, ndim * length);
        }
    }

    /**
     * @brief Push new data (GPU->GPU copy)
     * @param data_in_d Pointer to GPU data to push
     */
    void push(const FPTYPE* data_in_d)
    {
        this->start = (this->start + 1) % this->ndim_tot;
        this->ndim_use = std::min(this->ndim_use + 1, this->ndim_tot);
        ++this->ndim_history;

        // GPU-to-GPU copy
        FPTYPE* dest_d = data_d + this->start * this->length;
        base_device::memory::synchronize_memory_op<FPTYPE,
            base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
            dest_d, data_in_d, this->length);
    }

    /**
     * @brief Push data from CPU (CPU->GPU copy)
     * @param data_in_h Pointer to CPU data to push
     */
    void push_from_host(const FPTYPE* data_in_h)
    {
        this->start = (this->start + 1) % this->ndim_tot;
        this->ndim_use = std::min(this->ndim_use + 1, this->ndim_tot);
        ++this->ndim_history;

        // CPU-to-GPU copy
        FPTYPE* dest_d = data_d + this->start * this->length;
        base_device::memory::synchronize_memory_op<FPTYPE,
            base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            dest_d, data_in_h, this->length);
    }

    /**
     * @brief Get pointer to GPU data at given index
     * @param index Index in history (0 = most recent)
     * @return Pointer to GPU data
     */
    FPTYPE* get_data_d(int index) const
    {
        int actual_idx = index_move(index);
        return data_d + actual_idx * this->length;
    }

    /**
     * @brief Get pointer to current (most recent) GPU data
     */
    FPTYPE* get_current_d() const
    {
        return data_d + this->start * this->length;
    }

    /**
     * @brief Copy data to CPU
     * @param out_h CPU destination buffer
     * @param index Index in history (0 = most recent)
     */
    void copy_to_host(FPTYPE* out_h) const
    {
        copy_to_host(out_h, 0);
    }

    void copy_to_host(FPTYPE* out_h, int index) const
    {
        const FPTYPE* src_d = get_data_d(index);
        base_device::memory::synchronize_memory_op<FPTYPE,
            base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            out_h, src_d, this->length);
    }

    /**
     * @brief Reset mixing state
     */
    void reset()
    {
        this->ndim_use = 0;
        this->ndim_history = 0;
        this->start = -1;
    }

    /**
     * @brief Get the actual index for i-th vector relative to current position
     * @param n Relative index (0=current, -1=previous, etc.)
     */
    int index_move(const int& n) const
    {
        return (n + this->start + ndim_tot) % ndim_tot;
    }

  public:
    // GPU data buffer [ndim_tot * length]
    FPTYPE* data_d = nullptr;
    // Total number of vectors allocated
    int ndim_tot = 0;
    // Length of each vector
    std::size_t length = 0;
    // Current position in circular buffer
    int start = -1;
    // Number of vectors currently in use
    int ndim_use = 0;
    // Total number of vectors pushed (can exceed ndim_tot)
    int ndim_history = 0;
};

} // namespace Base_Mixing

#endif // MIXING_DATA_GPU_H_

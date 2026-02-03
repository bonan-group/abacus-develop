#ifndef MEMORY_H
#define MEMORY_H

#include <chrono>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace ModuleBase
{

/**
 * @brief Record memory consumption during computation.
 * @author Mohan
 * @note 8 bit  = 1 Byte; 1024 Byte = 1 KB;
 * 1024 KB   = 1 MB; 1024 MB   = 1 GB
 *
 */
class Memory
{
  public:
    Memory();
    ~Memory();

    /**
     * @brief Record memory consumed during computation
     *
     * @param class_name The name of a class
     * @param name The name of a quantity
     * @param n The number of the quantity
     * @param type The type of data
     * @param accumulate Useless, always set false
     * @return double
     */
    static double record(const std::string &class_name,
                         const std::string &name,
                         const long &n,
                         const std::string &type,
                         const bool accumulate = false);

    /**
     * @brief Record memory consumed during computation
     *
     * @param name The name of a quantity
     * @param n The number of the quantity
     * @param accumulate Useless, always set false
     */
    static void record(
      const std::string &name_in,
      const size_t &n_in,
      const bool accumulate = false
    );

#if defined(__CUDA) || defined(__ROCM)

    /**
     * @brief Record memory consumed on gpu during computation
     *
     * @param class_name The name of a class
     * @param name The name of a quantity
     * @param n The number of the quantity
     * @param type The type of data
     * @param accumulate Useless, always set false
     * @return double
     */
    static double record_gpu(const std::string &class_name,
                         const std::string &name,
                         const long &n,
                         const std::string &type,
                         const bool accumulate = false);

    /**
     * @brief Record memory consumed on gpu during computation
     *
     * @param name The name of a quantity
     * @param n The number of the quantity
     * @param accumulate Useless, always set false
     */
    static void record_gpu(
      const std::string &name_in,
      const size_t &n_in,
      const bool accumulate = false
    );

    /**
     * @brief Record GPU memory allocation with pointer tracking for later deallocation
     *
     * @param ptr The pointer to the allocated memory
     * @param name The name of the quantity
     * @param n The size in bytes
     */
    static void record_gpu_alloc(
      const void* ptr,
      const std::string &name_in,
      const size_t &n_in
    );

    /**
     * @brief Register a pointer for deallocation tracking (call after successful allocation)
     *
     * @param ptr The pointer to register
     * @param name The name of the allocation
     * @param size_bytes The size in bytes
     */
    static void register_gpu_pointer(
      const void* ptr,
      const std::string &name,
      const size_t size_bytes
    );

    /**
     * @brief Record GPU memory deallocation for leak detection
     *
     * @param ptr The pointer being freed (used to look up allocation info)
     */
    static void record_gpu_free(const void* ptr);

    /**
     * @brief Record GPU memory deallocation for leak detection (without pointer tracking)
     *
     * @param name The name of the quantity being freed
     * @param n The size in bytes being freed
     */
    static void record_gpu_free(
      const std::string &name_in,
      const size_t &n_in
    );

#endif

    static double &get_total(void)
    {
        return total;
    }

    static void finish(std::ofstream &ofs);

    /**
     * @brief Print memory consumed (> 1 MB) in a file
     *
     * @param ofs The output file stream for print out memory records
     */
    static void print_all(std::ofstream &ofs);

    static void print(const int find_in);

    /**
     * @brief Calculate memory requirements for various
     * types of data
     *
     * @param n The number of a type of data
     * @param type The type of data
     * @return double
     */
    static double calculate_mem(const long &n, const std::string &type);

    /**
     * @brief Initialize memory stream output
     *
     * @param out_dir The output directory (e.g., "OUT.ABACUS/")
     * @param enable Whether to enable streaming output
     */
    static void init_stream(const std::string &out_dir, bool enable);

    /**
     * @brief Close the memory stream file
     */
    static void close_stream();

    /**
     * @brief Check if memory stream is enabled
     */
    static bool is_stream_enabled() { return stream_enabled; }

  private:
    static double total;

    // Dynamic arrays for CPU memory tracking
    static std::vector<std::string> name_vec;
    static std::vector<std::string> class_name_vec;
    static std::vector<double> consume_vec;
    static bool init_flag;

    // Peak memory tracking
    static double peak_cpu;

#if defined(__CUDA) || defined(__ROCM)
    static double total_gpu;

    // Dynamic arrays for GPU memory tracking
    static std::vector<std::string> name_gpu_vec;
    static std::vector<std::string> class_name_gpu_vec;
    static std::vector<double> consume_gpu_vec;
    static bool init_flag_gpu;

    // Peak and deallocation tracking for GPU
    static double peak_gpu;
    static int alloc_count_gpu;
    static int free_count_gpu;
    static double freed_total_gpu;

    // Pointer tracking for proper deallocation tracking
    // Maps pointer address -> (name, size_in_bytes)
    struct AllocInfo {
        std::string name;
        size_t size_bytes;
    };
    static std::map<const void*, AllocInfo> gpu_alloc_map;
#endif

    // NDJSON stream output
    static std::ofstream ofs_mem_stream;
    static bool stream_enabled;
    static std::chrono::steady_clock::time_point start_time;

    static int complex_matrix_memory; //(16 Byte)
    static int double_memory; //(8 Byte)
    static int int_memory; //(4 Byte)
    static int bool_memory;
    static int short_memory; //(2 Byte)
    static int float_memory; //(4 Byte)

    /**
     * @brief Get the size in bytes for one element of the given type (pure function, no side effects)
     *
     * @param type The type of data
     * @return size_t Size in bytes per element
     */
    static size_t get_type_size(const std::string &type);

    /**
     * @brief Write an allocation event to the NDJSON stream
     */
    static void write_stream_event(const std::string &event_type,
                                   const std::string &device,
                                   const std::string &name,
                                   double size_mb,
                                   size_t size_bytes,
                                   double total_mb);

    /**
     * @brief Initialize internal arrays if not already done
     */
    static void init_cpu_arrays();

#if defined(__CUDA) || defined(__ROCM)
    /**
     * @brief Initialize GPU arrays if not already done
     */
    static void init_gpu_arrays();
#endif
};

} // namespace ModuleBase

#endif

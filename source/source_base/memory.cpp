//==========================================================
// AUTHOR : mohan
// DATE : 2008-11-18
//==========================================================
#include <cassert>
#include <iomanip>
#include <algorithm>
#include "memory.h"
#include "global_variable.h"
#include "source_base/parallel_reduce.h"

namespace ModuleBase
{
//    8 bit  = 1 Byte
// 1024 Byte = 1 KB
// 1024 KB   = 1 MB
// 1024 MB   = 1 GB
double Memory::total = 0.0;
int Memory::complex_matrix_memory = 2*sizeof(double); // 16 byte
int Memory::double_memory = sizeof(double); // 8 byte
int Memory::int_memory = sizeof(int); // 4.0 Byte
int Memory::bool_memory = sizeof(bool); // 1.0 Byte
int Memory::float_memory = sizeof(float); // 4.0 Byte
int Memory::short_memory = sizeof(short); // 2.0 Byte

bool Memory::init_flag = false;

// Dynamic arrays for CPU memory tracking
std::vector<std::string> Memory::name_vec;
std::vector<std::string> Memory::class_name_vec;
std::vector<double> Memory::consume_vec;

// Peak memory tracking
double Memory::peak_cpu = 0.0;

#if defined(__CUDA) || defined(__ROCM)

double Memory::total_gpu = 0.0;
bool Memory::init_flag_gpu = false;

// Dynamic arrays for GPU memory tracking
std::vector<std::string> Memory::name_gpu_vec;
std::vector<std::string> Memory::class_name_gpu_vec;
std::vector<double> Memory::consume_gpu_vec;

// Peak and deallocation tracking for GPU
double Memory::peak_gpu = 0.0;
int Memory::alloc_count_gpu = 0;
int Memory::free_count_gpu = 0;
double Memory::freed_total_gpu = 0.0;

// Pointer tracking for proper deallocation tracking
std::map<const void*, Memory::AllocInfo> Memory::gpu_alloc_map;

#endif

// NDJSON stream output
std::ofstream Memory::ofs_mem_stream;
bool Memory::stream_enabled = false;
std::chrono::steady_clock::time_point Memory::start_time;

Memory::Memory()
{
}

Memory::~Memory()
{
}

void Memory::init_cpu_arrays()
{
    if (!Memory::init_flag)
    {
        // Reserve some initial capacity to reduce reallocations
        name_vec.reserve(100);
        class_name_vec.reserve(100);
        consume_vec.reserve(100);
        Memory::init_flag = true;
    }
}

#if defined(__CUDA) || defined(__ROCM)
void Memory::init_gpu_arrays()
{
    if (!Memory::init_flag_gpu)
    {
        // Reserve some initial capacity to reduce reallocations
        name_gpu_vec.reserve(100);
        class_name_gpu_vec.reserve(100);
        consume_gpu_vec.reserve(100);
        Memory::init_flag_gpu = true;
    }
}
#endif

void Memory::init_stream(const std::string &out_dir, bool enable)
{
    if (enable)
    {
        std::string stream_file = out_dir + "memory_stream.jsonl";
        ofs_mem_stream.open(stream_file, std::ios::out | std::ios::trunc);
        if (ofs_mem_stream.is_open())
        {
            stream_enabled = true;
            start_time = std::chrono::steady_clock::now();
        }
        else
        {
            stream_enabled = false;
            std::cerr << "Warning: Could not open memory stream file: " << stream_file << std::endl;
        }
    }
    else
    {
        stream_enabled = false;
    }
}

void Memory::close_stream()
{
    if (ofs_mem_stream.is_open())
    {
        ofs_mem_stream.close();
    }
    stream_enabled = false;
}

void Memory::write_stream_event(const std::string &event_type,
                                 const std::string &device,
                                 const std::string &name,
                                 double size_mb,
                                 size_t size_bytes,
                                 double total_mb)
{
    if (!stream_enabled || !ofs_mem_stream.is_open())
    {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
    long long timestamp_ms = duration.count();

    // Write NDJSON line (each line is a complete JSON object)
    // Using fixed precision for consistent formatting
    ofs_mem_stream << std::fixed << std::setprecision(2);
    ofs_mem_stream << "{\"event\":\"" << event_type << "\""
                   << ",\"device\":\"" << device << "\""
                   << ",\"name\":\"" << name << "\""
                   << ",\"size_mb\":" << size_mb
                   << ",\"size_bytes\":" << size_bytes
                   << ",\"total_" << device << "_mb\":" << total_mb
                   << ",\"timestamp_ms\":" << timestamp_ms
                   << "}" << std::endl;

    // Flush to ensure real-time output (important for streaming)
    ofs_mem_stream.flush();
}

size_t Memory::get_type_size(const std::string &type)
{
    // Pure function that returns bytes per element for a given type
    // No side effects on Memory::total
    if (type == "ModuleBase::ComplexMatrix" || type == "complexmatrix" || type == "cdouble")
    {
        return complex_matrix_memory;
    }
    else if (type == "real" || type == "double")
    {
        return double_memory;
    }
    else if (type == "int")
    {
        return int_memory;
    }
    else if (type == "bool")
    {
        return bool_memory;
    }
    else if (type == "short")
    {
        return short_memory;
    }
    else if (type == "float")
    {
        return float_memory;
    }
    else if (type == "AtomLink")
    {
        return int_memory * 2 + double_memory * 3;
    }
    else if (type == "ModuleBase::Vector3<double>")
    {
        return 3 * double_memory;
    }
    else
    {
        std::cout << "not this type in memory storage : " << type << std::endl;
        return 0;
    }
}

double Memory::calculate_mem(const long &n_in, const std::string &type)
{
    // This function updates CPU total and peak - only use for CPU memory tracking
    double n = static_cast<double>(n_in);
    size_t type_size = get_type_size(type);
    double factor = 1.0 / 1024.0 / 1024.0;
    double mem = type_size * factor;

    total += n * mem;

    // Update peak CPU memory
    if (total > peak_cpu)
    {
        peak_cpu = total;
    }

    return n * mem;
}


double Memory::record
(
    const std::string &class_name_in,
    const std::string &name_in,
    const long &n_in,
    const std::string &type,
    const bool accumulate
)
{
    init_cpu_arrays();

    // Find existing record or create new one
    int find = -1;
    for (size_t i = 0; i < name_vec.size(); i++)
    {
        if (name_in == name_vec[i])
        {
            find = static_cast<int>(i);
            break;
        }
    }

    // Create new record if not found
    if (find < 0)
    {
        name_vec.push_back(name_in);
        class_name_vec.push_back(class_name_in);
        consume_vec.push_back(0.0);
        find = static_cast<int>(name_vec.size() - 1);
    }

    consume_vec[find] = Memory::calculate_mem(n_in, type);

    if (consume_vec[find] > 5)
    {
        print(find);
    }

    // Stream output for CPU allocations
    if (stream_enabled)
    {
        size_t size_bytes = static_cast<size_t>(n_in) * get_type_size(type);
        write_stream_event("alloc", "cpu", name_in, consume_vec[find], size_bytes, total);
    }

    return consume_vec[find];
}

void Memory::record
(
    const std::string &name_in,
    const size_t &n_in,
    const bool accumulate
)
{
    init_cpu_arrays();

    // Find existing record or create new one
    int find = -1;
    for (size_t i = 0; i < name_vec.size(); i++)
    {
        if (name_in == name_vec[i])
        {
            find = static_cast<int>(i);
            break;
        }
    }

    // Create new record if not found
    if (find < 0)
    {
        name_vec.push_back(name_in);
        class_name_vec.push_back("");
        consume_vec.push_back(0.0);
        find = static_cast<int>(name_vec.size() - 1);
    }

    const double factor = 1.0 / 1024.0 / 1024.0;
    double size_mb = n_in * factor;

    if (accumulate)
    {
        consume_vec[find] += size_mb;
        Memory::total += size_mb;
    }
    else
    {
        if (consume_vec[find] < size_mb)
        {
            Memory::total += size_mb - consume_vec[find];
            consume_vec[find] = size_mb;
            if (consume_vec[find] > 5)
            {
                print(find);
            }
        }
    }

    // Update peak CPU memory
    if (total > peak_cpu)
    {
        peak_cpu = total;
    }

    // Stream output for CPU allocations
    if (stream_enabled)
    {
        write_stream_event("alloc", "cpu", name_in, size_mb, n_in, total);
    }

    return;
}

#if defined(__CUDA) || defined(__ROCM)

double Memory::record_gpu
(
    const std::string &class_name_in,
    const std::string &name_in,
    const long &n_in,
    const std::string &type,
    const bool accumulate
)
{
    init_gpu_arrays();

    // Find existing record or create new one
    int find = -1;
    for (size_t i = 0; i < name_gpu_vec.size(); i++)
    {
        if (name_in == name_gpu_vec[i])
        {
            find = static_cast<int>(i);
            break;
        }
    }

    // Create new record if not found
    if (find < 0)
    {
        name_gpu_vec.push_back(name_in);
        class_name_gpu_vec.push_back(class_name_in);
        consume_gpu_vec.push_back(0.0);
        find = static_cast<int>(name_gpu_vec.size() - 1);
        alloc_count_gpu++;
    }

    // Calculate GPU memory separately (no side effects on CPU total)
    size_t type_size = get_type_size(type);
    size_t size_bytes = static_cast<size_t>(n_in) * type_size;
    double factor = 1.0 / 1024.0 / 1024.0;
    double size_mb = size_bytes * factor;

    consume_gpu_vec[find] = size_mb;
    Memory::total_gpu += size_mb;

    // Update peak GPU memory
    if (total_gpu > peak_gpu)
    {
        peak_gpu = total_gpu;
    }

    if (consume_gpu_vec[find] > 5)
    {
        print(find);
    }

    // Stream output for GPU allocations
    if (stream_enabled)
    {
        write_stream_event("alloc", "gpu", name_in, size_mb, size_bytes, total_gpu);
    }

    return consume_gpu_vec[find];
}

void Memory::record_gpu
(
    const std::string &name_in,
    const size_t &n_in,
    const bool accumulate
)
{
    init_gpu_arrays();

    // Find existing record or create new one
    int find = -1;
    for (size_t i = 0; i < name_gpu_vec.size(); i++)
    {
        if (name_in == name_gpu_vec[i])
        {
            find = static_cast<int>(i);
            break;
        }
    }

    // Create new record if not found
    if (find < 0)
    {
        name_gpu_vec.push_back(name_in);
        class_name_gpu_vec.push_back("");
        consume_gpu_vec.push_back(0.0);
        find = static_cast<int>(name_gpu_vec.size() - 1);
        alloc_count_gpu++;
    }

    const double factor = 1.0 / 1024.0 / 1024.0;
    double size_mb = n_in * factor;

    if (accumulate)
    {
        consume_gpu_vec[find] += size_mb;
        Memory::total_gpu += size_mb;
    }
    else
    {
        if (consume_gpu_vec[find] < size_mb)
        {
            Memory::total_gpu += size_mb - consume_gpu_vec[find];
            consume_gpu_vec[find] = size_mb;
            if (consume_gpu_vec[find] > 5)
            {
                print(find);
            }
        }
    }

    // Update peak GPU memory
    if (total_gpu > peak_gpu)
    {
        peak_gpu = total_gpu;
    }

    // Stream output for GPU allocations
    if (stream_enabled)
    {
        write_stream_event("alloc", "gpu", name_in, size_mb, n_in, total_gpu);
    }

    return;
}

void Memory::record_gpu_free
(
    const std::string &name_in,
    const size_t &n_in
)
{
    if (!init_flag_gpu)
    {
        return;  // Nothing to free if GPU tracking wasn't initialized
    }

    const double factor = 1.0 / 1024.0 / 1024.0;
    double size_mb = n_in * factor;

    // Find the record and update
    for (size_t i = 0; i < name_gpu_vec.size(); i++)
    {
        if (name_in == name_gpu_vec[i])
        {
            // Update the consumption record
            if (consume_gpu_vec[i] >= size_mb)
            {
                consume_gpu_vec[i] -= size_mb;
            }
            else
            {
                consume_gpu_vec[i] = 0.0;
            }
            break;
        }
    }

    // Update total GPU memory
    Memory::total_gpu -= size_mb;
    if (Memory::total_gpu < 0)
    {
        Memory::total_gpu = 0;
    }

    // Update freed tracking
    free_count_gpu++;
    freed_total_gpu += size_mb;

    // Stream output for GPU deallocations
    if (stream_enabled)
    {
        write_stream_event("free", "gpu", name_in, size_mb, n_in, total_gpu);
    }
}

void Memory::record_gpu_alloc
(
    const void* ptr,
    const std::string &name_in,
    const size_t &n_in
)
{
    if (ptr == nullptr)
    {
        return;
    }

    // First, record the allocation using existing method
    record_gpu(name_in, n_in, false);

    // Then, store the pointer -> allocation info mapping for later deallocation tracking
    gpu_alloc_map[ptr] = {name_in, n_in};
}

void Memory::register_gpu_pointer
(
    const void* ptr,
    const std::string &name,
    const size_t size_bytes
)
{
    if (ptr == nullptr)
    {
        return;
    }
    gpu_alloc_map[ptr] = {name, size_bytes};
}

void Memory::record_gpu_free(const void* ptr)
{
    if (ptr == nullptr)
    {
        return;
    }

    std::string name;
    size_t size_bytes = 0;

    auto it = gpu_alloc_map.find(ptr);
    if (it != gpu_alloc_map.end())
    {
        name = it->second.name;
        size_bytes = it->second.size_bytes;
        gpu_alloc_map.erase(it);
    }
    else
    {
        // Pointer not found in our tracking map
        // This can happen for allocations that weren't tracked
        // or for double-frees
        return;
    }

    // Now call the name/size version
    record_gpu_free(name, size_bytes);
}

#endif

void Memory::print(const int find)
{
    // Check if output stream is open before writing (may not be initialized during early setup)
    // Note: This function is now mostly a no-op since we use streaming output instead
    // Keeping it for backwards compatibility with code that calls it
    return;
}


void Memory::finish(std::ofstream &ofs)
{
    print_all(ofs);

    // Clear CPU arrays
    if (init_flag)
    {
        name_vec.clear();
        class_name_vec.clear();
        consume_vec.clear();
        init_flag = false;
    }

#if defined(__CUDA) || defined(__ROCM)
    // Clear GPU arrays
    if (init_flag_gpu)
    {
        name_gpu_vec.clear();
        class_name_gpu_vec.clear();
        consume_gpu_vec.clear();
        gpu_alloc_map.clear();
        init_flag_gpu = false;
    }
#endif

    // Close stream file
    close_stream();

    return;
}

void Memory::print_all(std::ofstream &ofs)
{
    if (!init_flag)
    {
        return;
    }

    const double small = 1.0; // unit is MB

#ifdef __MPI
    Parallel_Reduce::reduce_all(Memory::total);
    Parallel_Reduce::reduce_all(Memory::peak_cpu);
#if defined(__CUDA) || defined(__ROCM)
    Parallel_Reduce::reduce_all(Memory::total_gpu);
    Parallel_Reduce::reduce_all(Memory::peak_gpu);
#endif
#endif

    // Print enhanced memory summary header
    ofs << "\n ======================== MEMORY SUMMARY ========================" << std::endl;

    // CPU Memory Section
    ofs << "\n CPU Memory:" << std::endl;
    ofs << "   Peak Usage:     " << std::fixed << std::setprecision(1) << peak_cpu << " MB" << std::endl;
    ofs << "   Current Usage:  " << std::fixed << std::setprecision(1) << Memory::total << " MB" << std::endl;

    ofs << "\n   Top Allocations (>= " << small << " MB):" << std::endl;
    ofs << "   NAME                              SIZE (MB)" << std::endl;
    ofs << "   -----------------------------------------------" << std::endl;

    // Sort by size and print CPU allocations
    std::vector<std::pair<double, size_t>> sorted_cpu;
    for (size_t i = 0; i < consume_vec.size(); i++)
    {
#ifdef __MPI
        Parallel_Reduce::reduce_all(consume_vec[i]);
#endif
        if (consume_vec[i] >= small)
        {
            sorted_cpu.push_back({consume_vec[i], i});
        }
    }
    std::sort(sorted_cpu.begin(), sorted_cpu.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    for (const auto& item : sorted_cpu)
    {
        ofs << std::setw(35) << name_vec[item.second]
            << std::setw(12) << std::fixed << std::setprecision(1) << item.first << std::endl;
    }

#if defined(__CUDA) || defined(__ROCM)
    if (init_flag_gpu)
    {
        // GPU Memory Section
        ofs << "\n GPU Memory:" << std::endl;
        ofs << "   Peak Usage:     " << std::fixed << std::setprecision(1) << peak_gpu << " MB" << std::endl;
        ofs << "   Current Usage:  " << std::fixed << std::setprecision(1) << Memory::total_gpu << " MB";

        // Calculate potential leak
        double potential_leak = Memory::total_gpu;
        if (potential_leak > 0.1)  // Only report if significant
        {
            ofs << "  (potential leak: " << std::fixed << std::setprecision(1) << potential_leak << " MB not freed)";
        }
        ofs << std::endl;

        ofs << "   Allocations:    " << alloc_count_gpu << " total";
        if (free_count_gpu > 0)
        {
            ofs << ", " << free_count_gpu << " freed";
        }
        int active = alloc_count_gpu - free_count_gpu;
        if (active > 0)
        {
            ofs << ", " << active << " active";
        }
        ofs << std::endl;

        ofs << "\n   Top Allocations (>= " << small << " MB):" << std::endl;
        ofs << "   NAME                              SIZE (MB)" << std::endl;
        ofs << "   -----------------------------------------------" << std::endl;

        // Sort by size and print GPU allocations
        std::vector<std::pair<double, size_t>> sorted_gpu;
        for (size_t i = 0; i < consume_gpu_vec.size(); i++)
        {
#ifdef __MPI
            Parallel_Reduce::reduce_all(consume_gpu_vec[i]);
#endif
            if (consume_gpu_vec[i] >= small)
            {
                sorted_gpu.push_back({consume_gpu_vec[i], i});
            }
        }
        std::sort(sorted_gpu.begin(), sorted_gpu.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        for (const auto& item : sorted_gpu)
        {
            ofs << std::setw(35) << name_gpu_vec[item.second]
                << std::setw(12) << std::fixed << std::setprecision(1) << item.first << std::endl;
        }
    }
#endif

    ofs << "\n ================================================================" << std::endl;
    ofs << " (allocations < " << small << " MB have been omitted)" << std::endl;
    ofs << " ================================================================" << std::endl;

    return;
}

}

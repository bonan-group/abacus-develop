#include "source_pw/module_pwdft/vnl_chunk_policy.h"

#include <algorithm>
#include <limits>

#if defined(__CUDA) || defined(__UT_USE_CUDA)
#include <cuda_runtime.h>
#endif

bool VnlChunkPolicy::should_chunk(const int nkb,
                                  const int npwx,
                                  const std::size_t element_size) const
{
    if (!this->supported || nkb <= 0 || npwx <= 0 || element_size == 0)
    {
        return false;
    }

    const std::size_t wave_count = static_cast<std::size_t>(npwx);
    if (wave_count > std::numeric_limits<std::size_t>::max() / element_size)
    {
        return true;
    }
    const std::size_t bytes_per_projector = wave_count * element_size;
    return static_cast<std::size_t>(nkb) > this->memory_budget_bytes / bytes_per_projector;
}

int VnlChunkPolicy::atoms_in_chunk(const int remaining_atoms,
                                   const int projectors_per_atom) const
{
    if (remaining_atoms <= 0 || projectors_per_atom <= 0)
    {
        return 0;
    }
    const int atom_limit = std::max(1, this->projector_limit / projectors_per_atom);
    return std::min(remaining_atoms, atom_limit);
}

VnlChunkPolicy make_default_vnl_chunk_policy(const bool use_gpu)
{
    const std::size_t gib = 1024ULL * 1024ULL * 1024ULL;
    std::size_t memory_budget_bytes = 4ULL * gib;
    bool supported = false;
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    supported = use_gpu;
    if (use_gpu)
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && free_bytes > 0)
        {
            memory_budget_bytes = std::max(gib, free_bytes / 2);
        }
    }
#endif
    const VnlChunkPolicy policy = {supported, memory_budget_bytes, 64};
    return policy;
}

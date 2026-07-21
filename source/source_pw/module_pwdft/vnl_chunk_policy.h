#ifndef ABACUS_SOURCE_PW_MODULE_PWDFT_VNL_CHUNK_POLICY_H
#define ABACUS_SOURCE_PW_MODULE_PWDFT_VNL_CHUNK_POLICY_H

#include <cstddef>

struct VnlChunkPolicy
{
    bool supported;
    std::size_t memory_budget_bytes;
    int projector_limit;

    bool should_chunk(int nkb, int npwx, std::size_t element_size) const;
    int atoms_in_chunk(int remaining_atoms, int projectors_per_atom) const;
};

VnlChunkPolicy make_default_vnl_chunk_policy(bool use_gpu);

#endif // ABACUS_SOURCE_PW_MODULE_PWDFT_VNL_CHUNK_POLICY_H

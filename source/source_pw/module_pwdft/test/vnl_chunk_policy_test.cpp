#include "source_pw/module_pwdft/vnl_chunk_policy.h"

#include <gtest/gtest.h>

#include <cstddef>

namespace
{

TEST(VnlChunkPolicy, UnsupportedBackendNeverChunks)
{
    const VnlChunkPolicy policy = {false, 1, 64};

    EXPECT_FALSE(policy.should_chunk(1, 2, sizeof(double)));
}

TEST(VnlChunkPolicy, ChunksOnlyAboveMemoryBudget)
{
    const VnlChunkPolicy policy = {true, 64, 64};

    EXPECT_FALSE(policy.should_chunk(2, 4, 4));
    EXPECT_FALSE(policy.should_chunk(2, 8, 4));
    EXPECT_TRUE(policy.should_chunk(2, 9, 4));
}

TEST(VnlChunkPolicy, EmptyProjectorSetNeverChunks)
{
    const VnlChunkPolicy policy = {true, 0, 64};

    EXPECT_FALSE(policy.should_chunk(0, 128, sizeof(double)));
}

TEST(VnlChunkPolicy, AtomChunksRoundDownToWholeAtoms)
{
    const VnlChunkPolicy policy = {true, 1024, 10};

    EXPECT_EQ(policy.atoms_in_chunk(8, 3), 3);
    EXPECT_EQ(policy.atoms_in_chunk(2, 3), 2);
}

TEST(VnlChunkPolicy, AtomChunksAlwaysAdmitOneOversizedAtom)
{
    const VnlChunkPolicy policy = {true, 1024, 2};

    EXPECT_EQ(policy.atoms_in_chunk(4, 3), 1);
}

TEST(VnlChunkPolicy, InvalidAtomCountsOrProjectorsProduceEmptyChunk)
{
    const VnlChunkPolicy policy = {true, 1024, 64};

    EXPECT_EQ(policy.atoms_in_chunk(0, 4), 0);
    EXPECT_EQ(policy.atoms_in_chunk(3, 0), 0);
}

TEST(VnlChunkPolicy, DefaultPolicyKeepsProjectorLimitAndUnsupportedCpu)
{
    const VnlChunkPolicy policy = make_default_vnl_chunk_policy(false);
    const std::size_t gib = 1024ULL * 1024ULL * 1024ULL;

    EXPECT_FALSE(policy.supported);
    EXPECT_EQ(policy.memory_budget_bytes, 4ULL * gib);
    EXPECT_EQ(policy.projector_limit, 64);
}

#if !defined(__CUDA) && !defined(__UT_USE_CUDA)
TEST(VnlChunkPolicy, CpuOnlyBuildRejectsRequestedGpuPolicy)
{
    const VnlChunkPolicy policy = make_default_vnl_chunk_policy(true);

    EXPECT_FALSE(policy.supported);
}
#endif

} // namespace

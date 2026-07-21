#include "source_base/parallel_global.h"

#include <gtest/gtest.h>

int main(int argc, char** argv)
{
#ifdef __MPI
    int nproc = 1;
    int rank = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int nproc_in_band_group = 1;
    int rank_in_band_group = 0;
    int band_group = 0;
    int nproc_in_pool = 1;
    int rank_in_pool = 0;
    int pool = 0;
    Parallel_Global::init_pools(nproc,
                                rank,
                                1,
                                1,
                                nproc_in_band_group,
                                rank_in_band_group,
                                band_group,
                                nproc_in_pool,
                                rank_in_pool,
                                pool);
#endif

    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();

#ifdef __MPI
    MPI_Finalize();
#endif
    return result;
}

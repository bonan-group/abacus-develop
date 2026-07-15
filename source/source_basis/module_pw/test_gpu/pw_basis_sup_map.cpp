#include "source_base/matrix3.h"
#include "source_base/module_device/memory_op.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_sup.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

namespace
{
void expect_gpu_map_matches_fft_layout(const ModulePW::PW_Basis& basis)
{
    ASSERT_LT(basis.fftny, basis.ny);
    ASSERT_NE(basis.ig2ixyz_gpu, nullptr);

    std::vector<int> gpu_map(basis.npw);
    base_device::memory::synchronize_memory_op<int, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        gpu_map.data(), basis.ig2ixyz_gpu, basis.npw);

    for (int ig = 0; ig < basis.npw; ++ig)
    {
        const int isz = basis.ig2isz[ig];
        const int iz = isz % basis.nz;
        const int is = isz / basis.nz;
        const int ixy = basis.is2fftixy[is];
        const int ix = ixy / basis.fftny;
        const int iy = ixy % basis.fftny;
        const int expected = iz + iy * basis.nz + ix * basis.ny * basis.nz;
        EXPECT_EQ(gpu_map[ig], expected) << "ig=" << ig << ", ixy=" << ixy;
    }
}
} // namespace

TEST(PWBasisSupGpuTest, GammaYReducedIndexMapMatchesFftLayout)
{
    const bool gamma_only = true;
    const bool xprime = false;
    const int distribution_type = 1;
    const double lat0 = 1.0;
    const double smooth_ecut = 40.0;
    const double dense_ecut = 100.0;
    const ModuleBase::Matrix3 latvec(8.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 6.0);

    ModulePW::PW_Basis smooth("gpu", "double");
    smooth.initmpi(1, 0, MPI_COMM_WORLD);
    smooth.initgrids(lat0, latvec, smooth_ecut);
    smooth.initparameters(gamma_only, smooth_ecut, distribution_type, xprime);
    smooth.setuptransform();
    smooth.collect_local_pw();

    ModulePW::PW_Basis_Sup dense("gpu", "double");
    dense.initmpi(1, 0, MPI_COMM_WORLD);
    dense.initgrids(lat0, latvec, dense_ecut);
    dense.initparameters(gamma_only, dense_ecut, distribution_type, xprime);
    dense.setuptransform(&smooth);
    dense.collect_local_pw();

    expect_gpu_map_matches_fft_layout(smooth);
    expect_gpu_map_matches_fft_layout(dense);
}

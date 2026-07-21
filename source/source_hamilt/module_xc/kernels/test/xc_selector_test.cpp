#include "source_hamilt/module_xc/xc_resident_gpu.h"

#ifdef USE_LIBXC
#include <xc.h>
#else
#include "source_hamilt/module_xc/xc_ids.h"
#endif

#include <gtest/gtest.h>

#include <vector>

namespace
{

TEST(XcGpuMetadataSelectorTest, SelectsLdaSpinIndependentOfPoolSize)
{
    const std::vector<int> pz_ids = {XC_LDA_X, XC_LDA_C_PZ};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pz_ids, 2, 4),
              XC_Functional_GPU::XcGpuMode::LdaPzSpin);

    const std::vector<int> pw_ids = {XC_LDA_X, XC_LDA_C_PW};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pw_ids, 2, 7),
              XC_Functional_GPU::XcGpuMode::LdaPwSpin);
}

TEST(XcGpuMetadataSelectorTest, SelectsPbeAndPbesolScalarAndSpinModes)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_ids, 1, 1), XC_Functional_GPU::XcGpuMode::Pbe);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_ids, 2, 1), XC_Functional_GPU::XcGpuMode::SpinPbe);

    const std::vector<int> pbesol_ids = {XC_GGA_X_PBE_SOL, XC_GGA_C_PBE_SOL};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbesol_ids, 1, 1), XC_Functional_GPU::XcGpuMode::PbeSol);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbesol_ids, 2, 1),
              XC_Functional_GPU::XcGpuMode::SpinPbeSol);
}

TEST(XcGpuMetadataSelectorTest, RejectsUnsupportedMetadataAndMultiRankGga)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_ids, 4, 1),
              XC_Functional_GPU::XcGpuMode::Unsupported);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_ids, 1, 2),
              XC_Functional_GPU::XcGpuMode::Unsupported);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_ids, 2, 2),
              XC_Functional_GPU::XcGpuMode::Unsupported);

    const std::vector<int> unsupported_ids = {XC_LDA_X};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(unsupported_ids, 1, 1),
              XC_Functional_GPU::XcGpuMode::Unsupported);
}

TEST(XcGpuMetadataSelectorTest, ReportsCompiledEvaluatorAvailability)
{
#if __CUDA || __UT_USE_CUDA
    EXPECT_TRUE(XC_Functional_GPU::is_xc_gpu_evaluator_available());
#else
    EXPECT_FALSE(XC_Functional_GPU::is_xc_gpu_evaluator_available());
#endif
}

} // namespace

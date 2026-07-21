#include "source_hamilt/module_xc/xc_resident_gpu.h"
#include "source_hamilt/module_xc/xc_functional.h"

#ifdef USE_LIBXC
#include <xc.h>
#else
#include "source_hamilt/module_xc/xc_ids.h"
#endif

#include <gtest/gtest.h>

#include <vector>

namespace
{

XC_Functional_GPU::XcGpuRequest supported_request(const std::vector<int>& functional_ids, const int nspin)
{
    static double rho_up = 0.0;
    static double rho_down = 0.0;
    static Charge charge;
    static ModulePW::PW_Basis basis;
    static UnitCell unit_cell;
    static ModuleBase::matrix host_potential;

    XC_Functional_GPU::XcGpuRequest request;
    request.device = "gpu";
    request.nrxx = 8;
    request.nspin = nspin;
    request.functional_ids = &functional_ids;
    request.charge = &charge;
    request.rho_basis = &basis;
    request.unit_cell = &unit_cell;
    request.rho_up = &rho_up;
    request.rho_down = nspin == 2 ? &rho_down : nullptr;
    request.host_potential = &host_potential;
    charge.set_device("gpu");
    basis.set_device("gpu");
    basis.nrxx = request.nrxx;
    basis.poolnproc = 1;
    return request;
}

TEST(XcGpuSelectorTest, RejectsCpuDevice)
{
    const std::vector<int> functional_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    XC_Functional_GPU::XcGpuRequest request = supported_request(functional_ids, 1);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Pbe);

    request.device = "cpu";
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
}

TEST(XcGpuSelectorTest, SelectsLdaSpinIndependentOfPoolSize)
{
    const std::vector<int> pz_ids = {XC_LDA_X, XC_LDA_C_PZ};
    XC_Functional_GPU::XcGpuRequest request = supported_request(pz_ids, 2);
    request.rho_basis->poolnproc = 4;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::LdaPzSpin);

    const std::vector<int> pw_ids = {XC_LDA_X, XC_LDA_C_PW};
    request = supported_request(pw_ids, 2);
    request.rho_basis->poolnproc = 7;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::LdaPwSpin);
}

TEST(XcGpuSelectorTest, SelectsPbeAndPbeSolScalarAndSpinModes)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(supported_request(pbe_ids, 1)),
              XC_Functional_GPU::XcGpuMode::Pbe);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(supported_request(pbe_ids, 2)),
              XC_Functional_GPU::XcGpuMode::SpinPbe);

    const std::vector<int> pbesol_ids = {XC_GGA_X_PBE_SOL, XC_GGA_C_PBE_SOL};
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(supported_request(pbesol_ids, 1)),
              XC_Functional_GPU::XcGpuMode::PbeSol);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(supported_request(pbesol_ids, 2)),
              XC_Functional_GPU::XcGpuMode::SpinPbeSol);
}

TEST(XcGpuSelectorTest, RejectsUnsupportedFunctionalSpinAndLayout)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    XC_Functional_GPU::XcGpuRequest request = supported_request(pbe_ids, 1);

    request.rho_basis->nrxx = request.nrxx + 1;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    request = supported_request(pbe_ids, 4);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    const std::vector<int> unsupported_ids = {XC_LDA_X};
    request = supported_request(unsupported_ids, 1);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    request = supported_request(pbe_ids, 2);
    request.rho_down = nullptr;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    request = supported_request(pbe_ids, 1);
    request.rho_basis->set_device("cpu");
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    request = supported_request(pbe_ids, 1);
    request.use_libxc = true;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
}

TEST(XcGpuSelectorTest, RejectsMultiRankPbeLayouts)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    XC_Functional_GPU::XcGpuRequest request = supported_request(pbe_ids, 1);
    request.rho_basis->poolnproc = 2;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    request = supported_request(pbe_ids, 2);
    request.rho_basis->poolnproc = 2;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
}

} // namespace

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

struct SelectorFixture
{
    SelectorFixture(const std::vector<int>& functional_ids, const int nspin) : host_potential(nspin, 8)
    {
        basis.nrxx = 8;
        basis.nxyz = 8;
        basis.npw = 1;
        basis.poolnproc = 1;
        basis.gcar = new ModuleBase::Vector3<double>[1];
        basis.set_device("gpu");
        charge.set_rhopw(&basis);
        charge.set_device("gpu");
        charge.allocate(nspin, false);
        unit_cell.omega = 1.0;
        unit_cell.tpiba = 1.0;
        request.device = "gpu";
        request.nrxx = 8;
        request.nspin = nspin;
        request.functional_ids = &functional_ids;
        request.charge = &charge;
        request.rho_basis = &basis;
        request.unit_cell = &unit_cell;
        request.host_potential = &host_potential;
    }

    ModulePW::PW_Basis basis;
    Charge charge;
    UnitCell unit_cell;
    ModuleBase::matrix host_potential;
    XC_Functional_GPU::XcGpuRequest request;
};

TEST(XcGpuSelectorTest, AvailabilityMatchesCompiledEvaluator)
{
    const std::vector<int> functional_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    SelectorFixture fixture(functional_ids, 1);
    XC_Functional_GPU::XcGpuRequest& request = fixture.request;
#if __CUDA || __UT_USE_CUDA
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Pbe);
#else
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
#endif
}

#if __CUDA || __UT_USE_CUDA
TEST(XcGpuSelectorTest, RejectsCpuDevice)
{
    const std::vector<int> functional_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    SelectorFixture fixture(functional_ids, 1);
    XC_Functional_GPU::XcGpuRequest& request = fixture.request;
    request.device = "cpu";
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
}

TEST(XcGpuSelectorTest, SelectsLdaSpinIndependentOfPoolSize)
{
    const std::vector<int> pz_ids = {XC_LDA_X, XC_LDA_C_PZ};
    SelectorFixture pz_fixture(pz_ids, 2);
    XC_Functional_GPU::XcGpuRequest& request = pz_fixture.request;
    request.rho_basis->poolnproc = 4;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::LdaPzSpin);

    const std::vector<int> pw_ids = {XC_LDA_X, XC_LDA_C_PW};
    SelectorFixture pw_fixture(pw_ids, 2);
    pw_fixture.request.rho_basis->poolnproc = 7;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pw_fixture.request), XC_Functional_GPU::XcGpuMode::LdaPwSpin);
}

TEST(XcGpuSelectorTest, SelectsPbeAndPbeSolScalarAndSpinModes)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    SelectorFixture pbe_scalar(pbe_ids, 1);
    SelectorFixture pbe_spin(pbe_ids, 2);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_scalar.request),
              XC_Functional_GPU::XcGpuMode::Pbe);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbe_spin.request),
              XC_Functional_GPU::XcGpuMode::SpinPbe);

    const std::vector<int> pbesol_ids = {XC_GGA_X_PBE_SOL, XC_GGA_C_PBE_SOL};
    SelectorFixture pbesol_scalar(pbesol_ids, 1);
    SelectorFixture pbesol_spin(pbesol_ids, 2);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbesol_scalar.request),
              XC_Functional_GPU::XcGpuMode::PbeSol);
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(pbesol_spin.request),
              XC_Functional_GPU::XcGpuMode::SpinPbeSol);
}

TEST(XcGpuSelectorTest, RejectsUnsupportedFunctionalSpinAndLayout)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    SelectorFixture layout_fixture(pbe_ids, 1);
    XC_Functional_GPU::XcGpuRequest request = layout_fixture.request;

    request.rho_basis->nrxx = request.nrxx + 1;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture spin_fixture(pbe_ids, 4);
    request = spin_fixture.request;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    const std::vector<int> unsupported_ids = {XC_LDA_X};
    SelectorFixture functional_fixture(unsupported_ids, 1);
    request = functional_fixture.request;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture density_fixture(pbe_ids, 2);
    request = density_fixture.request;
    Charge no_density;
    double rho_core[8] = {};
    no_density.nrxx = request.nrxx;
    no_density.nspin = request.nspin;
    no_density.rho_core = rho_core;
    no_density.set_rhopw(request.rho_basis);
    no_density.set_device("gpu");
    request.charge = &no_density;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture basis_fixture(pbe_ids, 1);
    request = basis_fixture.request;
    request.rho_basis->set_device("cpu");
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture libxc_fixture(pbe_ids, 1);
    request = libxc_fixture.request;
    request.use_libxc = true;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
}

TEST(XcGpuSelectorTest, RejectsMultiRankPbeLayouts)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    SelectorFixture scalar_fixture(pbe_ids, 1);
    XC_Functional_GPU::XcGpuRequest request = scalar_fixture.request;
    request.rho_basis->poolnproc = 2;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture spin_fixture(pbe_ids, 2);
    request = spin_fixture.request;
    request.rho_basis->poolnproc = 2;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
}

TEST(XcGpuSelectorTest, RejectsInvalidPotentialTargetsAndLayouts)
{
    const std::vector<int> pbe_ids = {XC_GGA_X_PBE, XC_GGA_C_PBE};
    SelectorFixture host_fixture(pbe_ids, 1);
    XC_Functional_GPU::XcGpuRequest request = host_fixture.request;

    ModuleBase::matrix empty_host;
    request.host_potential = &empty_host;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture both_fixture(pbe_ids, 1);
    request = both_fixture.request;
    request.device_potential = request.charge->get_rho_d(0);
    request.potential_size = request.nrxx;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    request.host_potential = nullptr;
    --request.potential_size;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture charge_fixture(pbe_ids, 1);
    request = charge_fixture.request;
    Charge* charge = const_cast<Charge*>(request.charge);
    charge->nrxx = request.nrxx - 1;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
    charge->nrxx = request.nrxx;

    SelectorFixture npw_fixture(pbe_ids, 1);
    request = npw_fixture.request;
    request.rho_basis->npw = 0;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);

    SelectorFixture gcar_fixture(pbe_ids, 1);
    request = gcar_fixture.request;
    ModuleBase::Vector3<double>* gcar = request.rho_basis->gcar;
    request.rho_basis->gcar = nullptr;
    EXPECT_EQ(XC_Functional_GPU::select_xc_gpu_mode(request), XC_Functional_GPU::XcGpuMode::Unsupported);
    request.rho_basis->gcar = gcar;
}
#endif

} // namespace

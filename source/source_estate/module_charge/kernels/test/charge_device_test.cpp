#include "source_estate/module_charge/charge.h"

#include "source_base/module_device/memory_op.h"

#include <base/utils/gtest.h>

#include <algorithm>
#include <complex>
#include <type_traits>
#include <vector>

namespace
{

using sync_double_d2h_op
    = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using sync_double_h2d_op
    = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using sync_complex_d2h_op = base_device::memory::synchronize_memory_op<std::complex<double>,
                                                                       base_device::DEVICE_CPU,
                                                                       base_device::DEVICE_GPU>;
using sync_complex_h2d_op = base_device::memory::synchronize_memory_op<std::complex<double>,
                                                                       base_device::DEVICE_GPU,
                                                                       base_device::DEVICE_CPU>;

void initialize_basis(ModulePW::PW_Basis& basis)
{
    basis.nrxx = 5;
    basis.nxyz = 5;
    basis.npw = 3;
    basis.nmaxgr = 5;
}

void expect_double_data(const std::vector<double>& expected, const double* actual)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]);
    }
}

void expect_complex_data(const std::vector<std::complex<double>>& expected,
                         const std::complex<double>* actual)
{
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(expected[i], actual[i]);
    }
}

TEST(ChargeDeviceTest, AllocatesWhenHostStorageExistsFirst)
{
    ModulePW::PW_Basis basis;
    initialize_basis(basis);

    Charge charge;
    charge.set_rhopw(&basis);
    charge.allocate(2, false);
    EXPECT_EQ(charge.get_rho_d(), nullptr);

    charge.set_device("gpu");

    EXPECT_NE(charge.get_rho_d(0), nullptr);
    EXPECT_NE(charge.get_rho_d(1), nullptr);
    EXPECT_NE(charge.get_rhog_d(0), nullptr);
    EXPECT_NE(charge.get_rho_save_d(0), nullptr);
    EXPECT_NE(charge.get_rhog_save_d(0), nullptr);
    EXPECT_EQ(charge.get_kin_r_d(), nullptr);
}

TEST(ChargeDeviceTest, AllocatesWhenGpuIsEnabledFirst)
{
    ModulePW::PW_Basis basis;
    initialize_basis(basis);

    Charge charge;
    charge.set_rhopw(&basis);
    charge.set_device("gpu");
    EXPECT_EQ(charge.get_rho_d(), nullptr);

    charge.allocate(2, true);

    EXPECT_NE(charge.get_rho_d(0), nullptr);
    EXPECT_NE(charge.get_rho_d(1), nullptr);
    EXPECT_NE(charge.get_rhog_d(0), nullptr);
    EXPECT_NE(charge.get_kin_r_d(0), nullptr);
    EXPECT_NE(charge.get_kin_r_save_d(0), nullptr);
}

TEST(ChargeDeviceTest, RhoAndRhogRoundTrip)
{
    ModulePW::PW_Basis basis;
    initialize_basis(basis);
    Charge charge;
    charge.set_rhopw(&basis);
    charge.allocate(2, false);
    charge.set_device("gpu");

    const std::vector<double> rho = {0.1, 0.2, 0.3, 0.4, 0.5,
                                     1.1, 1.2, 1.3, 1.4, 1.5};
    const std::vector<std::complex<double>> rhog = {{0.1, -0.1}, {0.2, -0.2}, {0.3, -0.3},
                                                    {1.1, -1.1}, {1.2, -1.2}, {1.3, -1.3}};
    std::copy(rho.begin(), rho.end(), charge.rho[0]);
    std::copy(rhog.begin(), rhog.end(), charge.rhog[0]);

    charge.sync_rho_to_device();
    charge.sync_rhog_to_device();
    std::fill(charge.rho[0], charge.rho[0] + rho.size(), 0.0);
    std::fill(charge.rhog[0], charge.rhog[0] + rhog.size(), std::complex<double>());
    charge.sync_rho_to_host();
    charge.sync_rhog_to_host();

    expect_double_data(rho, charge.rho[0]);
    expect_complex_data(rhog, charge.rhog[0]);
}

TEST(ChargeDeviceTest, SavedDensityUploadAndRhogDownload)
{
    ModulePW::PW_Basis basis;
    initialize_basis(basis);
    Charge charge;
    charge.set_rhopw(&basis);
    charge.allocate(2, false);
    charge.set_device("gpu");

    const std::vector<double> rho_save = {0.5, 0.4, 0.3, 0.2, 0.1,
                                          1.5, 1.4, 1.3, 1.2, 1.1};
    const std::vector<std::complex<double>> rhog_save = {{0.4, 0.1}, {0.3, 0.2}, {0.2, 0.3},
                                                         {1.4, 1.1}, {1.3, 1.2}, {1.2, 1.3}};
    std::copy(rho_save.begin(), rho_save.end(), charge.rho_save[0]);
    std::copy(rhog_save.begin(), rhog_save.end(), charge.rhog_save[0]);

    charge.sync_saved_density_to_device();

    std::vector<double> uploaded_rho(rho_save.size(), 0.0);
    std::vector<std::complex<double>> uploaded_rhog(rhog_save.size());
    sync_double_d2h_op()(uploaded_rho.data(), charge.get_rho_save_d(0), uploaded_rho.size());
    sync_complex_d2h_op()(uploaded_rhog.data(), charge.get_rhog_save_d(0), uploaded_rhog.size());
    expect_double_data(rho_save, uploaded_rho.data());
    expect_complex_data(rhog_save, uploaded_rhog.data());

    const std::vector<std::complex<double>> device_update = {{2.4, -0.1}, {2.3, -0.2}, {2.2, -0.3},
                                                              {3.4, -1.1}, {3.3, -1.2}, {3.2, -1.3}};
    sync_complex_h2d_op()(charge.get_rhog_save_d(0), device_update.data(), device_update.size());
    charge.sync_rhog_save_to_host();
    expect_complex_data(device_update, charge.rhog_save[0]);
}

TEST(ChargeDeviceTest, OptionalKineticDensityRoundTripAndSavedUpload)
{
    ModulePW::PW_Basis basis;
    initialize_basis(basis);
    Charge charge;
    charge.set_rhopw(&basis);
    charge.set_device("gpu");
    charge.allocate(2, true);

    const std::vector<double> kinetic = {0.01, 0.02, 0.03, 0.04, 0.05,
                                         1.01, 1.02, 1.03, 1.04, 1.05};
    const std::vector<double> kinetic_save = {0.05, 0.04, 0.03, 0.02, 0.01,
                                              1.05, 1.04, 1.03, 1.02, 1.01};
    std::copy(kinetic.begin(), kinetic.end(), charge.kin_r[0]);
    std::copy(kinetic_save.begin(), kinetic_save.end(), charge.kin_r_save[0]);

    charge.sync_kin_r_to_device();
    charge.sync_saved_density_to_device();
    std::fill(charge.kin_r[0], charge.kin_r[0] + kinetic.size(), 0.0);
    charge.sync_kin_r_to_host();

    expect_double_data(kinetic, charge.kin_r[0]);
    std::vector<double> uploaded_save(kinetic_save.size(), 0.0);
    sync_double_d2h_op()(uploaded_save.data(), charge.get_kin_r_save_d(0), uploaded_save.size());
    expect_double_data(kinetic_save, uploaded_save.data());
}

TEST(ChargeDeviceTest, OrdinarySyncMethodsAreSafeWithoutDeviceStorage)
{
    Charge charge;

    EXPECT_NO_THROW(charge.sync_rho_to_device());
    EXPECT_NO_THROW(charge.sync_rho_to_host());
    EXPECT_NO_THROW(charge.sync_rhog_to_device());
    EXPECT_NO_THROW(charge.sync_rhog_to_host());
    EXPECT_NO_THROW(charge.sync_kin_r_to_device());
    EXPECT_NO_THROW(charge.sync_kin_r_to_host());
    EXPECT_NO_THROW(charge.sync_saved_density_to_device());
    EXPECT_NO_THROW(charge.sync_rhog_save_to_host());
}

TEST(ChargeDeviceTest, ChargeCannotBeCopied)
{
    EXPECT_FALSE(std::is_copy_constructible<Charge>::value);
    EXPECT_FALSE(std::is_copy_assignable<Charge>::value);
}

} // namespace

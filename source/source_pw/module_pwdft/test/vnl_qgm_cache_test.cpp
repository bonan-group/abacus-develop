#include <gtest/gtest.h>

#include "source_pw/module_pwdft/vnl_pw.h"

class VnlGpuCacheTestAccess
{
  public:
#if defined(__CUDA) || defined(__UT_USE_CUDA)
    static void configure_gpu_mirrors(pseudopot_cell_vnl& vnl)
    {
        vnl.use_gpu_ = true;
        base_device::memory::resize_memory_op<float, base_device::DEVICE_GPU>()(vnl.s_tab,
                                                                                vnl.tab.getSize());
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(vnl.d_tab,
                                                                                 vnl.tab.getSize());
        base_device::memory::resize_memory_op<float, base_device::DEVICE_GPU>()(vnl.s_deeq, 1);
    }
#endif
};

namespace
{

class VnlQgmCacheFixture : public testing::Test
{
  protected:
    void SetUp() override
    {
        cell.ntype = 1;
        cell.nat = 1;
        cell.omega = 8.0;
        cell.tpiba = 2.0;
        cell.atoms = new Atom[1];
        cell.set_atom_flag = true;
        cell.atoms[0].na = 1;
        cell.atoms[0].tau.push_back(ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
        cell.itia2iat.create(1, 1);
        cell.itia2iat(0, 0) = 0;

        basis.npw = 1;
        basis.gcar = new ModuleBase::Vector3<double>[1];
        basis.gcar[0] = ModuleBase::Vector3<double>(1.0, 0.0, 0.0);

        vnl.nhm = 1;
        vnl.lmaxq = 1;
        vnl.qgm.create(1, 1, 1);
        vnl.qgm_phase.create(1, 1);
        vnl.qgm_gcar.create(1, 3);
        vnl.qgm_phase(0, 0) = std::complex<double>(1.0, 0.0);
        vnl.qgm_gcar(0, 0) = 1.0;
        vnl.z_qgm = vnl.qgm.ptr;
        vnl.z_qgm_phase = vnl.qgm_phase.c;
        vnl.d_qgm_gcar = vnl.qgm_gcar.c;

        cell.cell_parameter_updated = true;
        vnl.update_after_structure_change(cell, &basis, false, 4, 0.5);
        cell.cell_parameter_updated = false;
    }

    UnitCell cell;
    ModulePW::PW_Basis basis;
    pseudopot_cell_vnl vnl;
};

TEST(VnlQgmCache, StartsUnready)
{
    pseudopot_cell_vnl vnl;

    EXPECT_FALSE(vnl.has_qgm_cache());
}

TEST(VnlQgmCache, CpuAliasesSurviveReleaseAndDestruction)
{
    pseudopot_cell_vnl vnl;
    vnl.nhm = 1;
    vnl.qgm.create(1, 1, 1);
    vnl.qgm_phase.create(1, 1);
    vnl.qgm_gcar.create(1, 3);
    vnl.z_qgm = vnl.qgm.ptr;
    vnl.z_qgm_phase = vnl.qgm_phase.c;
    vnl.d_qgm_gcar = vnl.qgm_gcar.c;

    vnl.release_memory();

    EXPECT_FALSE(vnl.has_qgm_cache());
    EXPECT_EQ(vnl.z_qgm, vnl.qgm.ptr);
    EXPECT_EQ(vnl.z_qgm_phase, vnl.qgm_phase.c);
    EXPECT_EQ(vnl.d_qgm_gcar, vnl.qgm_gcar.c);
}

TEST_F(VnlQgmCacheFixture, IonicPositionUpdateRefreshesAtomPhases)
{
    ASSERT_EQ(vnl.qgm_phase(0, 0), std::complex<double>(1.0, 0.0));

    cell.atoms[0].tau[0] = ModuleBase::Vector3<double>(0.25, 0.0, 0.0);
    cell.ionic_position_updated = true;
    vnl.update_after_structure_change(cell, &basis, false, 4, 0.5);

    const std::complex<double>* phase = vnl.get_qgm_phase_data<double>();
    EXPECT_NEAR(phase[0].real(), 0.0, 1.0e-12);
    EXPECT_NEAR(phase[0].imag(), -1.0, 1.0e-12);
}

TEST_F(VnlQgmCacheFixture, CellUpdateRefreshesTablesAndDenseBasisCaches)
{
    vnl.tab.create(1, 1, 1);
    vnl.tab(0, 0, 0) = 4.0;
    vnl.qgm(0, 0, 0) = std::complex<double>(7.0, 0.0);

    cell.omega = 32.0;
    cell.cell_parameter_updated = true;
    cell.atoms[0].tau[0] = ModuleBase::Vector3<double>(0.0, 0.25, 0.0);
    basis.gcar[0] = ModuleBase::Vector3<double>(0.0, 1.0, 0.0);
    vnl.update_after_structure_change(cell, &basis, false, 4, 0.5);

    EXPECT_DOUBLE_EQ(vnl.tab(0, 0, 0), 2.0);
    EXPECT_TRUE(vnl.has_qgm_cache());
    EXPECT_EQ(vnl.get_qgm_data<double>()[0], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(vnl.get_qgm_gcar_data()[0], 0.0);
    EXPECT_EQ(vnl.get_qgm_gcar_data()[1], 1.0);
    EXPECT_NEAR(vnl.get_qgm_phase_data<double>()[0].real(), 0.0, 1.0e-12);
    EXPECT_NEAR(vnl.get_qgm_phase_data<double>()[0].imag(), -1.0, 1.0e-12);
}

#if defined(__CUDA) || defined(__UT_USE_CUDA)

class VnlQgmGpuCacheFixture : public VnlQgmCacheFixture
{
  protected:
    void SetUp() override
    {
        cell.ntype = 1;
        cell.nat = 1;
        cell.omega = 8.0;
        cell.tpiba = 2.0;
        cell.atoms = new Atom[1];
        cell.set_atom_flag = true;
        cell.atoms[0].na = 1;
        cell.atoms[0].tau.push_back(ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
        cell.itia2iat.create(1, 1);
        cell.itia2iat(0, 0) = 0;

        basis.npw = 1;
        basis.gcar = new ModuleBase::Vector3<double>[1];
        basis.gcar[0] = ModuleBase::Vector3<double>(1.0, 0.0, 0.0);

        vnl.nhm = 1;
        vnl.lmaxq = 1;
        vnl.qgm_phase.create(1, 1);
        vnl.tab.create(1, 1, 1);
        vnl.tab(0, 0, 0) = 4.0;
        VnlGpuCacheTestAccess::configure_gpu_mirrors(vnl);

        cell.cell_parameter_updated = true;
        vnl.update_after_structure_change(cell, &basis, false, 4, 0.5);
        cell.cell_parameter_updated = false;
    }

    void TearDown() override
    {
        vnl.release_memory();
    }

    template <typename T>
    static std::vector<T> copy_from_gpu(const T* source, size_t count)
    {
        std::vector<T> host(count);
        base_device::memory::synchronize_memory_op<T,
                                                   base_device::DEVICE_CPU,
                                                   base_device::DEVICE_GPU>()(host.data(), source, count);
        return host;
    }
};

TEST_F(VnlQgmGpuCacheFixture, IonicPositionRefreshesDoubleAndFloatDevicePhases)
{
    cell.atoms[0].tau[0] = ModuleBase::Vector3<double>(0.25, 0.0, 0.0);
    cell.ionic_position_updated = true;
    vnl.update_after_structure_change(cell, &basis, false, 4, 0.5);

    const std::vector<std::complex<double>> phase_double
        = copy_from_gpu(vnl.get_qgm_phase_data<double>(), 1);
    const std::vector<std::complex<float>> phase_float
        = copy_from_gpu(vnl.get_qgm_phase_data<float>(), 1);
    EXPECT_NEAR(phase_double[0].real(), 0.0, 1.0e-12);
    EXPECT_NEAR(phase_double[0].imag(), -1.0, 1.0e-12);
    EXPECT_NEAR(phase_float[0].real(), 0.0f, 1.0e-6f);
    EXPECT_NEAR(phase_float[0].imag(), -1.0f, 1.0e-6f);
}

TEST_F(VnlQgmGpuCacheFixture, CellRefreshesDeviceTablesQgmPhasesAndGcar)
{
    const std::complex<double> stale_qgm(7.0, 0.0);
    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_CPU>()(vnl.get_qgm_data<double>(),
                                                                         &stale_qgm,
                                                                         1);

    cell.omega = 32.0;
    cell.cell_parameter_updated = true;
    cell.atoms[0].tau[0] = ModuleBase::Vector3<double>(0.0, 0.25, 0.0);
    basis.gcar[0] = ModuleBase::Vector3<double>(0.0, 1.0, 0.0);
    vnl.update_after_structure_change(cell, &basis, false, 4, 0.5);

    const std::vector<double> table_double = copy_from_gpu(vnl.get_tab_data<double>(), 1);
    const std::vector<float> table_float = copy_from_gpu(vnl.get_tab_data<float>(), 1);
    const std::vector<std::complex<double>> qgm_double = copy_from_gpu(vnl.get_qgm_data<double>(), 1);
    const std::vector<std::complex<float>> qgm_float = copy_from_gpu(vnl.get_qgm_data<float>(), 1);
    const std::vector<std::complex<double>> phase_double
        = copy_from_gpu(vnl.get_qgm_phase_data<double>(), 1);
    const std::vector<std::complex<float>> phase_float
        = copy_from_gpu(vnl.get_qgm_phase_data<float>(), 1);
    const std::vector<double> gcar = copy_from_gpu(vnl.get_qgm_gcar_data(), 3);

    EXPECT_DOUBLE_EQ(table_double[0], 2.0);
    EXPECT_FLOAT_EQ(table_float[0], 2.0f);
    EXPECT_EQ(qgm_double[0], std::complex<double>(0.0, 0.0));
    EXPECT_EQ(qgm_float[0], std::complex<float>(0.0f, 0.0f));
    EXPECT_NEAR(phase_double[0].imag(), -1.0, 1.0e-12);
    EXPECT_NEAR(phase_float[0].imag(), -1.0f, 1.0e-6f);
    EXPECT_DOUBLE_EQ(gcar[0], 0.0);
    EXPECT_DOUBLE_EQ(gcar[1], 1.0);
    EXPECT_DOUBLE_EQ(gcar[2], 0.0);
}

#endif

} // namespace

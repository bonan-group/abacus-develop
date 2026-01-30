#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>
#include <cmath>
#include <complex>
#include "source_cell/unitcell.h"
#include "source_estate/module_dm/test/prepare_unitcell.h"
#define private public
#include "source_io/module_parameter/parameter.h"
#include "source_pw/module_pwdft/structure_factor.h"
#undef private
/************************************************
 *  unit test of class Structure_factor and 
 ***********************************************/

/**
 * - Tested Functions:
 *   - Fcoef::create to create a 5 dimensional array of complex numbers
 *   - Soc::set_fcoef to set the fcoef array
 *   - Soc::spinor to calculate the spinor
 *   - Soc::rot_ylm to calculate the rotation matrix
 *   - Soc::sph_ind to calculate the m index of the spherical harmonics
*/

//compare two complex by using EXPECT_DOUBLE_EQ()
InfoNonlocal::InfoNonlocal()
{
}
InfoNonlocal::~InfoNonlocal()
{
}

Magnetism::Magnetism()
{
}
Magnetism::~Magnetism()
{
}

class StructureFactorTest : public testing::Test
{
protected:
    Structure_Factor SF;
    std::string output;
    ModulePW::PW_Basis* rho_basis;
    UnitCell* ucell;
    UcellTestPrepare utp = UcellTestLib["Si"];
    Parallel_Grid* pgrid;
    std::vector<int> nw = {13};
    int nlocal = 0;
void SetUp()
{
    rho_basis=new ModulePW::PW_Basis;
    ucell = utp.SetUcellInfo(nw, nlocal);
    ucell->set_iat2iwt(1);
    pgrid = new Parallel_Grid;
    rho_basis->npw=10;
    rho_basis->gcar=new ModuleBase::Vector3<double>[10];
    // for (int ig=0;ig<rho_basis->npw;ig++)
    // {
    //     rho_basis->gcar[ig]=1.0;
    // }
}
};

TEST_F(StructureFactorTest, set)
{
    const ModulePW::PW_Basis* rho_basis_in;
    const int nbspline_in =10;
    SF.set(rho_basis_in,nbspline_in);
    EXPECT_EQ(nbspline_in, 10);
}


TEST_F(StructureFactorTest, setup_structure_factor_double)
{
    rho_basis->npw = 10;
    SF.setup(ucell,*pgrid,rho_basis);  

    for (int i=0;i< ucell->nat * (2 * rho_basis->nx + 1);i++) 
    {
       EXPECT_EQ(SF.z_eigts1[i].real(),1);
       EXPECT_EQ(SF.z_eigts1[i].imag(),0);
    }

    for (int i=0;i< ucell->nat * (2 * rho_basis->ny + 1);i++) 
    {
       EXPECT_EQ(SF.z_eigts2[i].real(),1);
       EXPECT_EQ(SF.z_eigts2[i].imag(),0);
    }

    for (int i=0;i< ucell->nat * (2 * rho_basis->nz + 1);i++) 
    {
       EXPECT_EQ(SF.z_eigts3[i].real(),1);
       EXPECT_EQ(SF.z_eigts3[i].imag(),0);
    }
}

TEST_F(StructureFactorTest, setup_structure_factor_float)
{
    PARAM.sys.has_float_data = true;
    rho_basis->npw = 10;
    SF.setup(ucell,*pgrid,rho_basis);

    for (int i=0;i< ucell->nat * (2 * rho_basis->nx + 1);i++)
    {
       EXPECT_EQ(SF.c_eigts1[i].real(),1);
       EXPECT_EQ(SF.c_eigts1[i].imag(),0);
    }

    for (int i=0;i< ucell->nat * (2 * rho_basis->ny + 1);i++)
    {
       EXPECT_EQ(SF.c_eigts2[i].real(),1);
       EXPECT_EQ(SF.c_eigts2[i].imag(),0);
    }

    for (int i=0;i< ucell->nat * (2 * rho_basis->nz + 1);i++)
    {
       EXPECT_EQ(SF.c_eigts3[i].real(),1);
       EXPECT_EQ(SF.c_eigts3[i].imag(),0);
    }
}

TEST_F(StructureFactorTest, setup_structure_factor_gpu_float)
{
#if !(__CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM)
    GTEST_SKIP() << "GPU not available, skipping GPU test";
#endif

    // Save original settings
    std::string original_device = PARAM.inp.device;
    std::string original_basis = PARAM.inp.basis_type;

    // Enable GPU and single precision
    const_cast<std::string&>(PARAM.inp.basis_type) = "pw";  // Required for GPU
    PARAM.sys.has_float_data = true;

    // CPU reference computation
    const_cast<std::string&>(PARAM.inp.device) = "cpu";
    Structure_Factor SF_cpu;
    SF_cpu.set(rho_basis, -1);  // Use -1 to disable bspline mode
    rho_basis->npw = 10;
    SF_cpu.setup(ucell, *pgrid, rho_basis);

    // GPU computation
    const_cast<std::string&>(PARAM.inp.device) = "gpu";
    Structure_Factor SF_gpu;
    SF_gpu.set(rho_basis, -1);  // Use -1 to disable bspline mode
    SF_gpu.setup(ucell, *pgrid, rho_basis);

    // Restore original settings
    const_cast<std::string&>(PARAM.inp.device) = original_device;
    const_cast<std::string&>(PARAM.inp.basis_type) = original_basis;

    // Compare strucFac arrays (complex<double>)
    const int strucFac_size = ucell->ntype * rho_basis->npw;
    double max_error_strucFac = 0.0;
    for (int i = 0; i < strucFac_size; i++)
    {
        double err_real = std::abs(SF_gpu.strucFac.c[i].real() - SF_cpu.strucFac.c[i].real());
        double err_imag = std::abs(SF_gpu.strucFac.c[i].imag() - SF_cpu.strucFac.c[i].imag());
        max_error_strucFac = std::max(max_error_strucFac, std::max(err_real, err_imag));
    }

    // Compare eigts1/2/3 arrays (use double precision host arrays, since c_eigts are device pointers)
    // For single precision, we still compare the double host arrays which were computed by GPU
    const int eigts1_size = ucell->nat * (2 * rho_basis->nx + 1);
    const int eigts2_size = ucell->nat * (2 * rho_basis->ny + 1);
    const int eigts3_size = ucell->nat * (2 * rho_basis->nz + 1);

    double max_error_eigts1 = 0.0;
    for (int i = 0; i < eigts1_size; i++)
    {
        double err_real = std::abs(SF_gpu.eigts1.c[i].real() - SF_cpu.eigts1.c[i].real());
        double err_imag = std::abs(SF_gpu.eigts1.c[i].imag() - SF_cpu.eigts1.c[i].imag());
        max_error_eigts1 = std::max(max_error_eigts1, std::max(err_real, err_imag));
    }

    double max_error_eigts2 = 0.0;
    for (int i = 0; i < eigts2_size; i++)
    {
        double err_real = std::abs(SF_gpu.eigts2.c[i].real() - SF_cpu.eigts2.c[i].real());
        double err_imag = std::abs(SF_gpu.eigts2.c[i].imag() - SF_cpu.eigts2.c[i].imag());
        max_error_eigts2 = std::max(max_error_eigts2, std::max(err_real, err_imag));
    }

    double max_error_eigts3 = 0.0;
    for (int i = 0; i < eigts3_size; i++)
    {
        double err_real = std::abs(SF_gpu.eigts3.c[i].real() - SF_cpu.eigts3.c[i].real());
        double err_imag = std::abs(SF_gpu.eigts3.c[i].imag() - SF_cpu.eigts3.c[i].imag());
        max_error_eigts3 = std::max(max_error_eigts3, std::max(err_real, err_imag));
    }

    // Assert within tolerance (using double precision tolerance since we're comparing double arrays)
    double tolerance = 1e-10;
    EXPECT_LT(max_error_strucFac, tolerance)
        << "strucFac max error: " << max_error_strucFac << " exceeds tolerance: " << tolerance;
    EXPECT_LT(max_error_eigts1, tolerance)
        << "eigts1 max error: " << max_error_eigts1 << " exceeds tolerance: " << tolerance;
    EXPECT_LT(max_error_eigts2, tolerance)
        << "eigts2 max error: " << max_error_eigts2 << " exceeds tolerance: " << tolerance;
    EXPECT_LT(max_error_eigts3, tolerance)
        << "eigts3 max error: " << max_error_eigts3 << " exceeds tolerance: " << tolerance;
}

TEST_F(StructureFactorTest, setup_structure_factor_gpu_double)
{
#if !(__CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM)
    GTEST_SKIP() << "GPU not available, skipping GPU test";
#endif

    // Save original settings
    std::string original_device = PARAM.inp.device;
    std::string original_basis = PARAM.inp.basis_type;

    // Enable GPU and double precision
    const_cast<std::string&>(PARAM.inp.basis_type) = "pw";  // Required for GPU
    PARAM.sys.has_float_data = false;

    // CPU reference computation
    const_cast<std::string&>(PARAM.inp.device) = "cpu";
    Structure_Factor SF_cpu;
    SF_cpu.set(rho_basis, -1);  // Use -1 to disable bspline mode
    rho_basis->npw = 10;
    SF_cpu.setup(ucell, *pgrid, rho_basis);

    // GPU computation
    const_cast<std::string&>(PARAM.inp.device) = "gpu";
    Structure_Factor SF_gpu;
    SF_gpu.set(rho_basis, -1);  // Use -1 to disable bspline mode
    SF_gpu.setup(ucell, *pgrid, rho_basis);

    // Restore original settings
    const_cast<std::string&>(PARAM.inp.device) = original_device;
    const_cast<std::string&>(PARAM.inp.basis_type) = original_basis;

    // Compare strucFac arrays (complex<double>)
    const int strucFac_size = ucell->ntype * rho_basis->npw;
    double max_error_strucFac = 0.0;
    for (int i = 0; i < strucFac_size; i++)
    {
        double err_real = std::abs(SF_gpu.strucFac.c[i].real() - SF_cpu.strucFac.c[i].real());
        double err_imag = std::abs(SF_gpu.strucFac.c[i].imag() - SF_cpu.strucFac.c[i].imag());
        max_error_strucFac = std::max(max_error_strucFac, std::max(err_real, err_imag));
    }

    // Compare eigts1/2/3 arrays (use host arrays, since z_eigts are device pointers for GPU)
    const int eigts1_size = ucell->nat * (2 * rho_basis->nx + 1);
    const int eigts2_size = ucell->nat * (2 * rho_basis->ny + 1);
    const int eigts3_size = ucell->nat * (2 * rho_basis->nz + 1);

    double max_error_eigts1 = 0.0;
    for (int i = 0; i < eigts1_size; i++)
    {
        double err_real = std::abs(SF_gpu.eigts1.c[i].real() - SF_cpu.eigts1.c[i].real());
        double err_imag = std::abs(SF_gpu.eigts1.c[i].imag() - SF_cpu.eigts1.c[i].imag());
        max_error_eigts1 = std::max(max_error_eigts1, std::max(err_real, err_imag));
    }

    double max_error_eigts2 = 0.0;
    for (int i = 0; i < eigts2_size; i++)
    {
        double err_real = std::abs(SF_gpu.eigts2.c[i].real() - SF_cpu.eigts2.c[i].real());
        double err_imag = std::abs(SF_gpu.eigts2.c[i].imag() - SF_cpu.eigts2.c[i].imag());
        max_error_eigts2 = std::max(max_error_eigts2, std::max(err_real, err_imag));
    }

    double max_error_eigts3 = 0.0;
    for (int i = 0; i < eigts3_size; i++)
    {
        double err_real = std::abs(SF_gpu.eigts3.c[i].real() - SF_cpu.eigts3.c[i].real());
        double err_imag = std::abs(SF_gpu.eigts3.c[i].imag() - SF_cpu.eigts3.c[i].imag());
        max_error_eigts3 = std::max(max_error_eigts3, std::max(err_real, err_imag));
    }

    // Assert within tolerance (double precision: 1e-10)
    double tolerance = 1e-10;
    EXPECT_LT(max_error_strucFac, tolerance)
        << "strucFac max error: " << max_error_strucFac << " exceeds tolerance: " << tolerance;
    EXPECT_LT(max_error_eigts1, tolerance)
        << "eigts1 max error: " << max_error_eigts1 << " exceeds tolerance: " << tolerance;
    EXPECT_LT(max_error_eigts2, tolerance)
        << "eigts2 max error: " << max_error_eigts2 << " exceeds tolerance: " << tolerance;
    EXPECT_LT(max_error_eigts3, tolerance)
        << "eigts3 max error: " << max_error_eigts3 << " exceeds tolerance: " << tolerance;
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

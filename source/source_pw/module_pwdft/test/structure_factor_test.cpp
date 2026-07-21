#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include <string>
#include <cmath>
#include <complex>
#include "source_cell/setup_nonlocal.h"
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

namespace
{

Parameter& mutable_test_parameter()
{
    return PARAM;
}

class ScopedStructureFactorParameterState
{
  public:
    ScopedStructureFactorParameterState()
        : parameter(mutable_test_parameter()),
          device(this->parameter.input.device),
          has_float_data(this->parameter.sys.has_float_data)
    {
    }

    ~ScopedStructureFactorParameterState()
    {
        this->parameter.input.device = this->device;
        this->parameter.sys.has_float_data = this->has_float_data;
    }

  private:
    Parameter& parameter;
    std::string device;
    bool has_float_data;
};

} // namespace

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
    ScopedStructureFactorParameterState parameter_state;
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

#if defined(__CUDA) || defined(__UT_USE_CUDA)
TEST_F(StructureFactorTest, setup_keeps_typed_device_eigts_alive_until_destruction)
{
    ScopedStructureFactorParameterState parameter_state;
    Parameter& parameter = mutable_test_parameter();
    parameter.input.device = "gpu";
    parameter.sys.has_float_data = true;

    {
        Structure_Factor gpu_sf;
        gpu_sf.setup(ucell, *pgrid, rho_basis);

        EXPECT_NE(gpu_sf.get_eigts1_data<double>(), nullptr);
        EXPECT_NE(gpu_sf.get_eigts2_data<double>(), nullptr);
        EXPECT_NE(gpu_sf.get_eigts3_data<double>(), nullptr);
        EXPECT_NE(gpu_sf.get_eigts1_data<float>(), nullptr);
        EXPECT_NE(gpu_sf.get_eigts2_data<float>(), nullptr);
        EXPECT_NE(gpu_sf.get_eigts3_data<float>(), nullptr);
    }
}
#endif

int main()
{
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}

#include "source_hamilt/module_xc/kernels/xc_builtin_formula.h"
#include "source_hamilt/module_xc/kernels/xc_gradcorr_op.h"
#include "source_hamilt/module_xc/xc_functional.h"

#include <base/utils/gtest.h>

#include <vector>

namespace hamilt
{
namespace
{

TEST(XCBuiltinFormulaTest, SlaterAndPwInterpolationMatchFixedReferences)
{
    double energy = 0.0;
    double potential = 0.0;

    xc_builtin::slater(2.0, energy, potential);
    EXPECT_NEAR(energy, -0.22908264664157132, 1.0e-15);
    EXPECT_NEAR(potential, -0.30544352885542841, 1.0e-15);

    double down = 0.0;
    xc_builtin::slater_spin(0.7, 0.4, energy, potential, down);
    EXPECT_NEAR(energy, -0.67945046330498049, 1.0e-15);
    EXPECT_NEAR(potential, -0.97813579571063958, 1.0e-15);
    EXPECT_NEAR(down, -0.7374629802528655, 1.0e-15);

    xc_builtin::pw_interpolation(2.0, 0, energy, potential);
    EXPECT_NEAR(energy, -0.044759590030785945, 1.0e-15);
    EXPECT_NEAR(potential, -0.051492941313303925, 1.0e-15);
}

TEST(XCBuiltinFormulaTest, PublicPwPreservesIflagOneDensityBranches)
{
    double energy = 0.0;
    double potential = 0.0;

    XC_Functional::pw(0.5, 1, energy, potential);
    EXPECT_NEAR(energy, -0.075710887630248275, 1.0e-15);
    EXPECT_NEAR(potential, -0.084675804750428588, 1.0e-15);

    XC_Functional::pw(200.0, 1, energy, potential);
    EXPECT_NEAR(energy, -0.0016581002748332113, 1.0e-15);
    EXPECT_NEAR(potential, -0.0021259004122498168, 1.0e-15);
}

struct SpinReference
{
    double zeta;
    double pz_energy;
    double pz_up;
    double pz_down;
    double pw_energy;
    double pw_up;
    double pw_down;
};

TEST(XCBuiltinFormulaTest, PzAndPwSpinMatchUnpolarizedOrdinaryAndNearPolarizedReferences)
{
    const SpinReference references[] = {
        {0.0,
         -0.045091213633848361,
         -0.051812941923196104,
         -0.051812941923196104,
         -0.044759590030785945,
         -0.051492941313303925,
         -0.051492941313303925},
        {0.4,
         -0.042173382699745195,
         -0.039546672674751017,
         -0.069200710702592455,
         -0.042224770023135939,
         -0.040823293239436273,
         -0.066774988812374439},
        {0.999999,
         -0.02408982895580588,
         -0.02755654080239028,
         -0.16221359768066954,
         -0.023909441964858471,
         -0.027355265953446256,
         -0.18243446302294963},
        {-0.999999,
         -0.02408982895580588,
         -0.16221359768066954,
         -0.02755654080239028,
         -0.023909441964858471,
         -0.18243446302294963,
         -0.027355265953446256}};

    for (const SpinReference& reference : references)
    {
        double energy = 0.0;
        double up = 0.0;
        double down = 0.0;
        xc_builtin::pz_spin(2.0, reference.zeta, energy, up, down);
        EXPECT_NEAR(energy, reference.pz_energy, 1.0e-15);
        EXPECT_NEAR(up, reference.pz_up, 1.0e-15);
        EXPECT_NEAR(down, reference.pz_down, 1.0e-15);

        xc_builtin::pw_spin(2.0, reference.zeta, energy, up, down);
        EXPECT_NEAR(energy, reference.pw_energy, 1.0e-15);
        EXPECT_NEAR(up, reference.pw_up, 1.0e-15);
        EXPECT_NEAR(down, reference.pw_down, 1.0e-15);
    }
}

TEST(XCBuiltinFormulaTest, PbeAndPbesolGradientCorrectionsMatchFixedReferences)
{
    double energy = 0.0;
    double v1 = 0.0;
    double v2 = 0.0;

    xc_builtin::pbex(0, 0.7, 0.13, energy, v1, v2);
    EXPECT_NEAR(energy, -0.00088364965109673092, 1.0e-15);
    EXPECT_NEAR(v1, 0.0016750823649456215, 1.0e-15);
    EXPECT_NEAR(v2, -0.013562060713024477, 1.0e-15);

    xc_builtin::pbex(2, 0.7, 0.13, energy, v1, v2);
    EXPECT_NEAR(energy, -0.00049749211386265963, 1.0e-15);
    EXPECT_NEAR(v1, 0.00094504933801475876, 1.0e-15);
    EXPECT_NEAR(v2, -0.0076434078178492905, 1.0e-15);

    xc_builtin::pbec(0, 0.7, 0.13, energy, v1, v2);
    EXPECT_NEAR(energy, 0.00086819772950828627, 1.0e-15);
    EXPECT_NEAR(v1, -0.0015965812890975639, 1.0e-15);
    EXPECT_NEAR(v2, 0.013093302289672195, 1.0e-15);

    xc_builtin::pbec(2, 0.7, 0.13, energy, v1, v2);
    EXPECT_NEAR(energy, 0.00060223405888618119, 1.0e-15);
    EXPECT_NEAR(v1, -0.0011195374287538086, 1.0e-15);
    EXPECT_NEAR(v2, 0.0091378980790997513, 1.0e-15);
}

TEST(XCBuiltinFormulaTest, GridSkipsZeroGradientAndVanishingDensity)
{
    const int nrxx = 2;
    const double epsr = 1.0e-6;
    const std::vector<double> rho = {0.7, 1.0e-12};
    const std::vector<double> rho_core(nrxx, 0.0);
    const std::vector<double> gradient = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    std::vector<double> potential(nrxx, 1.0);
    std::vector<double> h(3 * nrxx, 1.0);
    double etxc = 1.0;
    double vtxc = 1.0;

    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                               nrxx,
                                                               0,
                                                               1.0,
                                                               epsr,
                                                               rho.data(),
                                                               rho_core.data(),
                                                               gradient.data(),
                                                               potential.data(),
                                                               h.data(),
                                                               &etxc,
                                                               &vtxc);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_DOUBLE_EQ(potential[ir], 0.0);
    }
    for (int i = 0; i < 3 * nrxx; ++i)
    {
        EXPECT_DOUBLE_EQ(h[i], 0.0);
    }
    EXPECT_DOUBLE_EQ(etxc, 0.0);
    EXPECT_DOUBLE_EQ(vtxc, 0.0);
}

} // namespace
} // namespace hamilt

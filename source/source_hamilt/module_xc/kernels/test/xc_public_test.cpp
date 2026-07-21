#include "source_hamilt/module_xc/xc_functional.h"

#include <gtest/gtest.h>

TEST(XCBuiltinPublicBoundaryTest, PwPreservesIflagOneDensityBranches)
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

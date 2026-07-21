#include "source_pw/module_pwdft/vnl_pw.h"

#include <gtest/gtest.h>

namespace
{

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

} // namespace

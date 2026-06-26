#include "../pw_basis_k.h"
#include "depend_mock.h"
#include "pw_test.h"
#include "source_base/constants.h"
#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"

#ifdef __MPI
#include "mpi.h"
#include "test_tool.h"
#endif

#include <array>
#include <cmath>
#include <complex>
#include <vector>

extern int nproc_in_pool, rank_in_pool;
extern std::string precision_flag, device_flag;

namespace
{
using complexd = std::complex<double>;

void init_exx_pw(ModulePW::PW_Basis_K& wfcpw, const std::vector<ModuleBase::Vector3<double>>& kvecs)
{
    const double lat0 = 1.8897261254578281;
    const ModuleBase::Matrix3 latvec(10.0, 0.0, 0.0,
                                     0.0, 10.0, 0.0,
                                     0.0, 0.0, 10.0);
    wfcpw.initgrids(lat0, latvec, 10.0);
    wfcpw.initparameters(false, 2.0, static_cast<int>(kvecs.size()), kvecs.data(), 1, true);
    wfcpw.setuptransform();
    wfcpw.collect_local_pw();
}

std::vector<ModuleBase::Vector3<double>> unshifted_2x2x2_kmesh()
{
    return {ModuleBase::Vector3<double>(0.0, 0.0, 0.0),
            ModuleBase::Vector3<double>(0.5, 0.0, 0.0),
            ModuleBase::Vector3<double>(0.0, 0.5, 0.0),
            ModuleBase::Vector3<double>(0.0, 0.0, 0.5),
            ModuleBase::Vector3<double>(0.5, 0.5, 0.0),
            ModuleBase::Vector3<double>(0.5, 0.0, 0.5),
            ModuleBase::Vector3<double>(0.0, 0.5, 0.5),
            ModuleBase::Vector3<double>(0.5, 0.5, 0.5)};
}

std::vector<ModuleBase::Vector3<double>> shifted_2x2x2_kmesh()
{
    return {ModuleBase::Vector3<double>(0.25, 0.25, 0.25),
            ModuleBase::Vector3<double>(0.25, 0.25, -0.25),
            ModuleBase::Vector3<double>(0.25, -0.25, 0.25),
            ModuleBase::Vector3<double>(0.25, -0.25, -0.25),
            ModuleBase::Vector3<double>(-0.25, 0.25, 0.25),
            ModuleBase::Vector3<double>(-0.25, 0.25, -0.25),
            ModuleBase::Vector3<double>(-0.25, -0.25, 0.25),
            ModuleBase::Vector3<double>(-0.25, -0.25, -0.25)};
}

K_Vectors::ExxFullPoint rotation_2x2x2_point()
{
    K_Vectors::ExxFullPoint point;
    point.identity = false;
    point.conjugate_only = false;
    point.time_reversal = false;
    point.full_kvec_d = ModuleBase::Vector3<double>(0.5, 0.0, 0.0);
    point.gmatrix = ModuleBase::Matrix3(0.0, 1.0, 0.0,
                                        -1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0);
    point.kgmatrix = point.gmatrix;
    point.gtrans = ModuleBase::Vector3<double>(0.0, 0.5, 0.0);
    return point;
}

K_Vectors::ExxFullPoint time_reversal_rotation_2x2x2_point()
{
    K_Vectors::ExxFullPoint point;
    point.identity = false;
    point.conjugate_only = false;
    point.time_reversal = true;
    point.full_kvec_d = ModuleBase::Vector3<double>(0.25, 0.25, 0.25);
    point.gmatrix = ModuleBase::Matrix3(0.0, -1.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 0.0, -1.0);
    point.kgmatrix = point.gmatrix;
    point.gtrans = ModuleBase::Vector3<double>(0.5, 0.0, 0.0);
    return point;
}

void expect_complex_arrays_near(const std::vector<complexd>& got,
                                const std::vector<complexd>& expected,
                                const double tolerance)
{
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(got[i].real(), expected[i].real(), tolerance) << "index " << i;
        EXPECT_NEAR(got[i].imag(), expected[i].imag(), tolerance) << "index " << i;
    }
}
} // namespace

TEST_F(PWTEST, exx_realspace_symmetry_rotation_matches_reciprocal_remap)
{
    if (nproc_in_pool > 1)
    {
        GTEST_SKIP() << "reciprocal-remap reference is single-rank only";
    }

    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    const ModuleBase::Vector3<double> kvec_d[1] = {ModuleBase::Vector3<double>(0.0, 0.0, 0.0)};
    init_exx_pw(wfcpw, std::vector<ModuleBase::Vector3<double>>(kvec_d, kvec_d + 1));

    K_Vectors::ExxFullPoint point;
    point.identity = false;
    point.conjugate_only = false;
    point.time_reversal = false;
    point.full_kvec_d = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    point.full_kvec_c = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    point.gmatrix = ModuleBase::Matrix3(0.0, -1.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0);
    point.kgmatrix = point.gmatrix;
    point.gtrans = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);

    std::vector<complexd> recip(wfcpw.npwk[0]);
    for (int ig = 0; ig < wfcpw.npwk[0]; ++ig)
    {
        recip[ig] = complexd(0.1 * (ig + 1), -0.03 * (ig % 7));
    }

    const auto remap = hamilt::build_exx_symmetry_remap(&wfcpw, point, 0, false);
    std::vector<complexd> expected(wfcpw.nrxx);
    wfcpw.recip2real_remapped(recip.data(),
                              expected.data(),
                              static_cast<int>(remap.rep_igl.size()),
                              remap.rep_igl.data(),
                              remap.fft_isz.data(),
                              remap.phase.data(),
                              false,
                              1.0);

    std::vector<complexd> representative_real(wfcpw.nrxx);
    wfcpw.recip_to_real(static_cast<const base_device::DEVICE_CPU*>(nullptr),
                        recip.data(),
                        representative_real.data(),
                        0,
                        false,
                        1.0);

    std::vector<complexd> rotated(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_cpu(&wfcpw, point, 0, representative_real.data(), rotated.data());

    expect_complex_arrays_near(rotated, expected, 1e-10);
}

TEST_F(PWTEST, exx_realspace_symmetry_adjoint_matches_reciprocal_remap)
{
    if (nproc_in_pool > 1)
    {
        GTEST_SKIP() << "reciprocal-remap reference is single-rank only";
    }

    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    const ModuleBase::Vector3<double> kvec_d[1] = {ModuleBase::Vector3<double>(0.0, 0.0, 0.0)};
    init_exx_pw(wfcpw, std::vector<ModuleBase::Vector3<double>>(kvec_d, kvec_d + 1));

    K_Vectors::ExxFullPoint point;
    point.identity = false;
    point.conjugate_only = false;
    point.time_reversal = false;
    point.full_kvec_d = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    point.full_kvec_c = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    point.gmatrix = ModuleBase::Matrix3(0.0, -1.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0);
    point.kgmatrix = point.gmatrix;
    point.gtrans = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);

    std::vector<complexd> full_real(wfcpw.nrxx);
    for (int ir = 0; ir < wfcpw.nrxx; ++ir)
    {
        full_real[ir] = complexd(0.02 * (ir + 1), -0.04 * (ir % 5));
    }

    const auto remap = hamilt::build_exx_symmetry_remap(&wfcpw, point, 0, false);
    std::vector<complexd> expected(wfcpw.npwk[0], complexd(0.0, 0.0));
    wfcpw.real2recip_remapped(full_real.data(),
                              expected.data(),
                              static_cast<int>(remap.rep_igl.size()),
                              remap.rep_igl.data(),
                              remap.fft_isz.data(),
                              remap.phase.data(),
                              false,
                              1.0);

    std::vector<complexd> representative_real(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_adjoint_cpu(&wfcpw,
                                                      point,
                                                      0,
                                                      full_real.data(),
                                                      representative_real.data());

    std::vector<complexd> rotated_recip(wfcpw.npwk[0], complexd(0.0, 0.0));
    wfcpw.real_to_recip(static_cast<const base_device::DEVICE_CPU*>(nullptr),
                        representative_real.data(),
                        rotated_recip.data(),
                        0,
                        false,
                        1.0);

    expect_complex_arrays_near(rotated_recip, expected, 1e-10);
}

TEST_F(PWTEST, exx_realspace_symmetry_rotation_matches_reciprocal_remap_for_2x2x2_kmesh)
{
    if (nproc_in_pool > 1)
    {
        GTEST_SKIP() << "reciprocal-remap reference is single-rank only";
    }

    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    init_exx_pw(wfcpw, unshifted_2x2x2_kmesh());
    const int rep_spin_index = 2;
    K_Vectors::ExxFullPoint point = rotation_2x2x2_point();
    point.full_kvec_c = point.full_kvec_d * wfcpw.G;

    std::vector<complexd> recip(wfcpw.npwk[rep_spin_index]);
    for (int ig = 0; ig < wfcpw.npwk[rep_spin_index]; ++ig)
    {
        recip[ig] = complexd(0.07 * (ig + 1), -0.02 * ((ig + 3) % 7));
    }

    const auto remap = hamilt::build_exx_symmetry_remap(&wfcpw, point, rep_spin_index, false);
    std::vector<complexd> expected(wfcpw.nrxx);
    wfcpw.recip2real_remapped(recip.data(),
                              expected.data(),
                              static_cast<int>(remap.rep_igl.size()),
                              remap.rep_igl.data(),
                              remap.fft_isz.data(),
                              remap.phase.data(),
                              false,
                              1.0);

    std::vector<complexd> representative_real(wfcpw.nrxx);
    wfcpw.recip_to_real(static_cast<const base_device::DEVICE_CPU*>(nullptr),
                        recip.data(),
                        representative_real.data(),
                        rep_spin_index,
                        false,
                        1.0);

    std::vector<complexd> rotated(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_cpu(&wfcpw,
                                              point,
                                              rep_spin_index,
                                              representative_real.data(),
                                              rotated.data());

    expect_complex_arrays_near(rotated, expected, 1e-10);
}

TEST_F(PWTEST, exx_realspace_symmetry_time_reversal_matches_reciprocal_remap_for_2x2x2_kmesh)
{
    if (nproc_in_pool > 1)
    {
        GTEST_SKIP() << "reciprocal-remap reference is single-rank only";
    }

    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    init_exx_pw(wfcpw, shifted_2x2x2_kmesh());
    const int rep_spin_index = 3;
    K_Vectors::ExxFullPoint point = time_reversal_rotation_2x2x2_point();
    point.full_kvec_c = point.full_kvec_d * wfcpw.G;

    std::vector<complexd> recip(wfcpw.npwk[rep_spin_index]);
    for (int ig = 0; ig < wfcpw.npwk[rep_spin_index]; ++ig)
    {
        recip[ig] = complexd(-0.04 * (ig + 2), 0.06 * ((ig + 5) % 9));
    }

    const auto remap = hamilt::build_exx_symmetry_remap(&wfcpw, point, rep_spin_index, false);
    std::vector<complexd> expected(wfcpw.nrxx);
    wfcpw.recip2real_remapped_conjugate(recip.data(),
                                        expected.data(),
                                        static_cast<int>(remap.rep_igl.size()),
                                        remap.rep_igl.data(),
                                        remap.fft_isz.data(),
                                        remap.phase.data(),
                                        false,
                                        1.0);

    std::vector<complexd> representative_real(wfcpw.nrxx);
    wfcpw.recip_to_real(static_cast<const base_device::DEVICE_CPU*>(nullptr),
                        recip.data(),
                        representative_real.data(),
                        rep_spin_index,
                        false,
                        1.0);

    std::vector<complexd> rotated(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_cpu(&wfcpw,
                                              point,
                                              rep_spin_index,
                                              representative_real.data(),
                                              rotated.data());

    expect_complex_arrays_near(rotated, expected, 1e-10);
}

TEST_F(PWTEST, exx_realspace_symmetry_adjoint_matches_reciprocal_remap_for_2x2x2_kmesh)
{
    if (nproc_in_pool > 1)
    {
        GTEST_SKIP() << "reciprocal-remap reference is single-rank only";
    }

    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    init_exx_pw(wfcpw, unshifted_2x2x2_kmesh());
    const int rep_spin_index = 2;
    K_Vectors::ExxFullPoint point = rotation_2x2x2_point();
    point.full_kvec_c = point.full_kvec_d * wfcpw.G;

    std::vector<complexd> full_real(wfcpw.nrxx);
    for (int ir = 0; ir < wfcpw.nrxx; ++ir)
    {
        full_real[ir] = complexd(0.015 * (ir + 1), -0.025 * ((ir + 4) % 11));
    }

    const auto remap = hamilt::build_exx_symmetry_remap(&wfcpw, point, rep_spin_index, false);
    std::vector<complexd> expected(wfcpw.npwk[rep_spin_index], complexd(0.0, 0.0));
    wfcpw.real2recip_remapped(full_real.data(),
                              expected.data(),
                              static_cast<int>(remap.rep_igl.size()),
                              remap.rep_igl.data(),
                              remap.fft_isz.data(),
                              remap.phase.data(),
                              false,
                              1.0);

    std::vector<complexd> representative_real(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_adjoint_cpu(&wfcpw,
                                                      point,
                                                      rep_spin_index,
                                                      full_real.data(),
                                                      representative_real.data());

    std::vector<complexd> rotated_recip(wfcpw.npwk[rep_spin_index], complexd(0.0, 0.0));
    wfcpw.real_to_recip(static_cast<const base_device::DEVICE_CPU*>(nullptr),
                        representative_real.data(),
                        rotated_recip.data(),
                        rep_spin_index,
                        false,
                        1.0);

    expect_complex_arrays_near(rotated_recip, expected, 1e-10);
}

TEST_F(PWTEST, exx_realspace_symmetry_time_reversal_adjoint_matches_reciprocal_remap_for_2x2x2_kmesh)
{
    if (nproc_in_pool > 1)
    {
        GTEST_SKIP() << "reciprocal-remap reference is single-rank only";
    }

    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    init_exx_pw(wfcpw, shifted_2x2x2_kmesh());
    const int rep_spin_index = 3;
    K_Vectors::ExxFullPoint point = time_reversal_rotation_2x2x2_point();
    point.full_kvec_c = point.full_kvec_d * wfcpw.G;

    std::vector<complexd> full_real(wfcpw.nrxx);
    for (int ir = 0; ir < wfcpw.nrxx; ++ir)
    {
        full_real[ir] = complexd(-0.018 * ((ir + 1) % 17), 0.021 * ((ir + 2) % 13));
    }

    const auto remap = hamilt::build_exx_symmetry_remap(&wfcpw, point, rep_spin_index, false);
    std::vector<complexd> expected(wfcpw.npwk[rep_spin_index], complexd(0.0, 0.0));
    wfcpw.real2recip_remapped_conjugate(full_real.data(),
                                        expected.data(),
                                        static_cast<int>(remap.rep_igl.size()),
                                        remap.rep_igl.data(),
                                        remap.fft_isz.data(),
                                        remap.phase.data(),
                                        false,
                                        1.0);

    std::vector<complexd> representative_real(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_adjoint_cpu(&wfcpw,
                                                      point,
                                                      rep_spin_index,
                                                      full_real.data(),
                                                      representative_real.data());

    std::vector<complexd> rotated_recip(wfcpw.npwk[rep_spin_index], complexd(0.0, 0.0));
    wfcpw.real_to_recip(static_cast<const base_device::DEVICE_CPU*>(nullptr),
                        representative_real.data(),
                        rotated_recip.data(),
                        rep_spin_index,
                        false,
                        1.0);

    expect_complex_arrays_near(rotated_recip, expected, 1e-10);
}

TEST_F(PWTEST, exx_realspace_symmetry_rotation_is_consistent_across_pool_ranks)
{
    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    init_exx_pw(wfcpw, unshifted_2x2x2_kmesh());
    K_Vectors::ExxFullPoint point = rotation_2x2x2_point();
    point.full_kvec_c = point.full_kvec_d * wfcpw.G;

    std::vector<complexd> representative_real(wfcpw.nrxx);
    for (int ir = 0; ir < wfcpw.nrxx; ++ir)
    {
        const int local_iz = ir % wfcpw.nplane;
        const int ixy = ir / wfcpw.nplane;
        const int global_iz = wfcpw.startz_current + local_iz;
        const int global_ir = ixy * wfcpw.nz + global_iz;
        representative_real[ir] = complexd(0.01 * (global_ir + 1), -0.02 * (global_ir % 11));
    }

    std::vector<complexd> rotated(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_cpu(&wfcpw, point, 2, representative_real.data(), rotated.data());

    int mismatch_count = 0;
    for (int ir = 0; ir < wfcpw.nrxx; ++ir)
    {
        const int local_iz = ir % wfcpw.nplane;
        const int ixy = ir / wfcpw.nplane;
        const int global_iz = wfcpw.startz_current + local_iz;
        const int full_global_ir = ixy * wfcpw.nz + global_iz;
        const int iz = full_global_ir % wfcpw.nz;
        const int ixy_full = full_global_ir / wfcpw.nz;
        const int iy = ixy_full % wfcpw.ny;
        const int ix = ixy_full / wfcpw.ny;
        const int rep_ix = (wfcpw.nx - iy) % wfcpw.nx;
        const int rep_iy = (ix + wfcpw.ny / 2) % wfcpw.ny;
        const int rep_global_ir = iz + rep_iy * wfcpw.nz + rep_ix * wfcpw.ny * wfcpw.nz;
        const complexd phase(0.0, 1.0);
        const complexd expected = phase * complexd(0.01 * (rep_global_ir + 1), -0.02 * (rep_global_ir % 11));
        if (std::abs(rotated[ir].real() - expected.real()) > 1e-12
            || std::abs(rotated[ir].imag() - expected.imag()) > 1e-12)
        {
            if (mismatch_count < 8)
            {
                EXPECT_NEAR(rotated[ir].real(), expected.real(), 1e-12);
                EXPECT_NEAR(rotated[ir].imag(), expected.imag(), 1e-12);
            }
            ++mismatch_count;
        }
    }
    EXPECT_EQ(mismatch_count, 0);
}

TEST_F(PWTEST, exx_realspace_symmetry_adjoint_roundtrip_is_consistent_across_pool_ranks)
{
    ModulePW::PW_Basis_K wfcpw(device_flag, precision_flag);
#ifdef __MPI
    wfcpw.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
#endif

    init_exx_pw(wfcpw, unshifted_2x2x2_kmesh());
    K_Vectors::ExxFullPoint point = rotation_2x2x2_point();
    point.full_kvec_c = point.full_kvec_d * wfcpw.G;

    std::vector<complexd> representative_real(wfcpw.nrxx);
    for (int ir = 0; ir < wfcpw.nrxx; ++ir)
    {
        const int local_iz = ir % wfcpw.nplane;
        const int ixy = ir / wfcpw.nplane;
        const int global_iz = wfcpw.startz_current + local_iz;
        const int global_ir = ixy * wfcpw.nz + global_iz;
        representative_real[ir] = complexd(0.03 * std::sin(0.13 * global_ir),
                                           -0.02 * std::cos(0.07 * global_ir));
    }

    std::vector<complexd> rotated(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_cpu(&wfcpw, point, 2, representative_real.data(), rotated.data());

    std::vector<complexd> recovered(wfcpw.nrxx);
    hamilt::rotate_exx_realspace_symmetry_adjoint_cpu(&wfcpw, point, 2, rotated.data(), recovered.data());

    expect_complex_arrays_near(recovered, representative_real, 1e-12);
}

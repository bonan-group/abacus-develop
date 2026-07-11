#include <source_hamilt/module_xc/kernels/xc_functional_op.h>
#include <source_hamilt/module_xc/kernels/xc_gradcorr_op.h>
#include <source_hamilt/module_xc/xc_functional.h>
#include <source_hamilt/module_xc/xc_gpu_policy.h>
#define private public
#include <source_io/module_parameter/parameter.h>
#undef private
#include <source_base/global_variable.h>
#include <source_base/parallel_comm.h>

#include <base/utils/gtest.h>
#include <source_base/module_device/memory_op.h>
#include <source_cell/unitcell.h>
#include <ATen/core/tensor.h>

#include <complex>
#include <cstdlib>
#include <string>

#ifdef __MPI
#include <mpi.h>
#endif

namespace hamilt {

namespace
{

class ScopedMpiInit
{
  public:
    ScopedMpiInit()
    {
#ifdef __MPI
        int initialized = 0;
        MPI_Initialized(&initialized);
        if (initialized == 0)
        {
            MPI_Init(nullptr, nullptr);
            owns_mpi_ = true;
        }
        old_pool_world_ = POOL_WORLD;
        old_nproc_ = GlobalV::NPROC;
        old_my_rank_ = GlobalV::MY_RANK;
        old_kpar_ = GlobalV::KPAR;
        old_my_pool_ = GlobalV::MY_POOL;
        old_nproc_in_pool_ = GlobalV::NPROC_IN_POOL;
        old_rank_in_pool_ = GlobalV::RANK_IN_POOL;
        MPI_Comm_rank(MPI_COMM_WORLD, &GlobalV::MY_RANK);
        MPI_Comm_size(MPI_COMM_WORLD, &GlobalV::NPROC);
        GlobalV::KPAR = 1;
        GlobalV::MY_POOL = 0;
        GlobalV::NPROC_IN_POOL = GlobalV::NPROC;
        GlobalV::RANK_IN_POOL = GlobalV::MY_RANK;
        MPI_Comm_split(MPI_COMM_WORLD, 0, GlobalV::MY_RANK, &POOL_WORLD);
        owns_pool_world_ = true;
#endif
    }

    ~ScopedMpiInit()
    {
#ifdef __MPI
        if (owns_pool_world_)
        {
            MPI_Comm_free(&POOL_WORLD);
            POOL_WORLD = old_pool_world_;
            GlobalV::NPROC = old_nproc_;
            GlobalV::MY_RANK = old_my_rank_;
            GlobalV::KPAR = old_kpar_;
            GlobalV::MY_POOL = old_my_pool_;
            GlobalV::NPROC_IN_POOL = old_nproc_in_pool_;
            GlobalV::RANK_IN_POOL = old_rank_in_pool_;
        }
        int finalized = 0;
        MPI_Finalized(&finalized);
        if (owns_mpi_ && finalized == 0)
        {
            MPI_Finalize();
        }
#endif
    }

  private:
    bool owns_mpi_ = false;
    bool owns_pool_world_ = false;
#ifdef __MPI
    MPI_Comm old_pool_world_ = MPI_COMM_NULL;
    int old_nproc_ = 1;
    int old_my_rank_ = 0;
    int old_kpar_ = 1;
    int old_my_pool_ = 0;
    int old_nproc_in_pool_ = 1;
    int old_rank_in_pool_ = 0;
#endif
};

} // namespace

template<typename T>
class XC_FunctionalOpTest : public testing::Test {
public:
    XC_FunctionalOpTest() = default;

    ~XC_FunctionalOpTest() override = default;
};

TYPED_TEST_SUITE(XC_FunctionalOpTest, base::utils::ComplexTypes);

TEST(XCFunctionGpuPolicyTest, GuardsSupportedBuiltins)
{
    using XC_Functional_GPU::xc_gpu_policy;
    using XC_Functional_GPU::xc_gpu_stress_policy;

    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PBE"));
    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PZ"));
    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "pbesol"));
    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "LDA"));
    EXPECT_FALSE(xc_gpu_policy(false, false, 1, "PBE"));
    EXPECT_FALSE(xc_gpu_policy(true, true, 1, "PBE"));
    EXPECT_FALSE(xc_gpu_policy(true, false, 2, "PBE"));
    EXPECT_FALSE(xc_gpu_policy(true, false, 1, "SCAN"));

    EXPECT_TRUE(xc_gpu_stress_policy(true, false, 1, "PBE"));
    EXPECT_TRUE(xc_gpu_stress_policy(true, false, 2, "pbesol"));
    EXPECT_FALSE(xc_gpu_stress_policy(false, false, 1, "PBE"));
    EXPECT_FALSE(xc_gpu_stress_policy(true, true, 1, "PBE"));
    EXPECT_FALSE(xc_gpu_stress_policy(true, false, 4, "PBE"));
    EXPECT_FALSE(xc_gpu_stress_policy(true, false, 1, "LDA"));
    EXPECT_FALSE(xc_gpu_stress_policy(true, false, 1, "SCAN"));
}

TEST(XCFunctionGpuPolicyTest, EnvDisablesGpuOnlyWhenExplicitlyOff)
{
    using XC_Functional_GPU::xc_gpu_disabled_by_env;

    const char* old_env_value = std::getenv("ABACUS_XC_GPU");
    const bool had_xc_gpu_env = old_env_value != nullptr;
    const std::string old_xc_gpu_env = had_xc_gpu_env ? std::string(old_env_value) : std::string();

    unsetenv("ABACUS_XC_GPU");
    EXPECT_FALSE(xc_gpu_disabled_by_env());

    setenv("ABACUS_XC_GPU", "0", 1);
    EXPECT_TRUE(xc_gpu_disabled_by_env());

    setenv("ABACUS_XC_GPU", "off", 1);
    EXPECT_TRUE(xc_gpu_disabled_by_env());

    setenv("ABACUS_XC_GPU", "FALSE", 1);
    EXPECT_TRUE(xc_gpu_disabled_by_env());

    setenv("ABACUS_XC_GPU", "1", 1);
    EXPECT_FALSE(xc_gpu_disabled_by_env());

    if (had_xc_gpu_env)
    {
        setenv("ABACUS_XC_GPU", old_xc_gpu_env.c_str(), 1);
    }
    else
    {
        unsetenv("ABACUS_XC_GPU");
    }
}

TEST(XCGradcorrOpTest, PbeGridCpuMatchesBuiltinReferenceValues)
{
    const int nrxx = 5;
    const int iflag = 0;
    const double e2 = 1.0;
    const double epsr = 1.0e-6;
    const std::vector<double> rho = {0.17E+01, 0.17E+01, 0.15E+01, 0.88E-01, 0.18E+04};
    const std::vector<double> grho = {0.81E-11, 0.17E+01, 0.36E+02, 0.87E-01, 0.55E+00};
    const std::vector<double> gdr = {std::sqrt(grho[0]), 0.0, 0.0,
                                     std::sqrt(grho[1]), 0.0, 0.0,
                                     std::sqrt(grho[2]), 0.0, 0.0,
                                     std::sqrt(grho[3]), 0.0, 0.0,
                                     std::sqrt(grho[4]), 0.0, 0.0};
    const std::vector<double> rho_core(nrxx, 0.0);
    const std::vector<double> ref_sxc = {0.0, -0.000103750, -0.0328708695, -0.0032277985, 0.0};
    const std::vector<double> ref_v1xc = {0.0, 0.00021536874, 0.04931694948, 0.05374316118, 0.0};
    const std::vector<double> ref_v2xc = {0.0, -0.0002386176, -0.0025842562, -0.0825164089, 0.0};

    std::vector<double> v(nrxx, 0.0);
    std::vector<double> h(3 * nrxx, 0.0);
    double etxc = 0.0;
    double vtxc = 0.0;
    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                               nrxx,
                                                               iflag,
                                                               e2,
                                                               epsr,
                                                               rho.data(),
                                                               rho_core.data(),
                                                               gdr.data(),
                                                               v.data(),
                                                               h.data(),
                                                               &etxc,
                                                               &vtxc);

    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(v[ir], ref_v1xc[ir], 1.0e-8);
        EXPECT_NEAR(h[3 * ir], ref_v2xc[ir] * gdr[3 * ir], 1.0e-8);
        EXPECT_DOUBLE_EQ(h[3 * ir + 1], 0.0);
        EXPECT_DOUBLE_EQ(h[3 * ir + 2], 0.0);
        ref_etxc += ref_sxc[ir];
        ref_vtxc += ref_v1xc[ir] * rho[ir];
    }
    EXPECT_NEAR(etxc, ref_etxc, 1.0e-8);
    EXPECT_NEAR(vtxc, ref_vtxc, 1.0e-8);
}

#if __CUDA || __UT_USE_CUDA
TEST(XCGradcorrOpTest, PbeGridGpuMatchesCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 6;
    const int iflag = 0;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;

    const std::vector<double> rho = {0.018, 0.024, 0.031, 0.045, 0.052, 0.063};
    const std::vector<double> rho_core = {0.002, 0.0015, 0.001, 0.0025, 0.003, 0.002};
    const std::vector<double> gdr = {0.011, 0.017, 0.019, -0.009, 0.021, 0.013,
                                     0.015, -0.014, 0.026, 0.010, -0.018, 0.022,
                                     -0.020, 0.012, 0.016, 0.024, 0.019, -0.011};

    std::vector<double> ref_v(nrxx, 0.0);
    std::vector<double> ref_h(3 * nrxx, 0.0);
    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                               nrxx,
                                                               iflag,
                                                               e2,
                                                               epsr,
                                                               rho.data(),
                                                               rho_core.data(),
                                                               gdr.data(),
                                                               ref_v.data(),
                                                               ref_h.data(),
                                                               &ref_etxc,
                                                               &ref_vtxc);

    double* d_rho = nullptr;
    double* d_rho_core = nullptr;
    double* d_gdr = nullptr;
    double* d_v = nullptr;
    double* d_h = nullptr;
    double* d_sums = nullptr;
    resmem_double_op()(d_rho, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_v, nrxx);
    resmem_double_op()(d_h, 3 * nrxx);
    resmem_double_op()(d_sums, 2);

    syncmem_h2d_op()(d_rho, rho.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);
    syncmem_h2d_op()(d_gdr, gdr.data(), 3 * nrxx);

    double gpu_etxc = 0.0;
    double gpu_vtxc = 0.0;
    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                               nrxx,
                                                               iflag,
                                                               e2,
                                                               epsr,
                                                               d_rho,
                                                               d_rho_core,
                                                               d_gdr,
                                                               d_v,
                                                               d_h,
                                                               d_sums,
                                                               &gpu_etxc,
                                                               &gpu_vtxc);

    std::vector<double> gpu_v(nrxx, 0.0);
    std::vector<double> gpu_h(3 * nrxx, 0.0);
    syncmem_d2h_op()(gpu_v.data(), d_v, nrxx);
    syncmem_d2h_op()(gpu_h.data(), d_h, 3 * nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(gpu_v[ir], ref_v[ir], 1.0e-10);
    }
    for (int ir = 0; ir < 3 * nrxx; ++ir)
    {
        EXPECT_NEAR(gpu_h[ir], ref_h[ir], 1.0e-10);
    }
    EXPECT_NEAR(gpu_etxc, ref_etxc, 1.0e-10);
    EXPECT_NEAR(gpu_vtxc, ref_vtxc, 1.0e-10);

    delmem_double_op()(d_rho);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_gdr);
    delmem_double_op()(d_v);
    delmem_double_op()(d_h);
    delmem_double_op()(d_sums);
}

TEST(XCGradcorrOpTest, PbesolGridGpuMatchesCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 6;
    const int iflag = 2;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;

    const std::vector<double> rho = {0.018, 0.024, 0.031, 0.045, 0.052, 0.063};
    const std::vector<double> rho_core = {0.002, 0.0015, 0.001, 0.0025, 0.003, 0.002};
    const std::vector<double> gdr = {0.011, 0.017, 0.019, -0.009, 0.021, 0.013,
                                     0.015, -0.014, 0.026, 0.010, -0.018, 0.022,
                                     -0.020, 0.012, 0.016, 0.024, 0.019, -0.011};

    std::vector<double> ref_v(nrxx, 0.0);
    std::vector<double> ref_h(3 * nrxx, 0.0);
    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                               nrxx,
                                                               iflag,
                                                               e2,
                                                               epsr,
                                                               rho.data(),
                                                               rho_core.data(),
                                                               gdr.data(),
                                                               ref_v.data(),
                                                               ref_h.data(),
                                                               &ref_etxc,
                                                               &ref_vtxc);

    double* d_rho = nullptr;
    double* d_rho_core = nullptr;
    double* d_gdr = nullptr;
    double* d_v = nullptr;
    double* d_h = nullptr;
    double* d_sums = nullptr;
    resmem_double_op()(d_rho, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_v, nrxx);
    resmem_double_op()(d_h, 3 * nrxx);
    resmem_double_op()(d_sums, 2);

    syncmem_h2d_op()(d_rho, rho.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);
    syncmem_h2d_op()(d_gdr, gdr.data(), 3 * nrxx);

    double gpu_etxc = 0.0;
    double gpu_vtxc = 0.0;
    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                               nrxx,
                                                               iflag,
                                                               e2,
                                                               epsr,
                                                               d_rho,
                                                               d_rho_core,
                                                               d_gdr,
                                                               d_v,
                                                               d_h,
                                                               d_sums,
                                                               &gpu_etxc,
                                                               &gpu_vtxc);

    std::vector<double> gpu_v(nrxx, 0.0);
    std::vector<double> gpu_h(3 * nrxx, 0.0);
    syncmem_d2h_op()(gpu_v.data(), d_v, nrxx);
    syncmem_d2h_op()(gpu_h.data(), d_h, 3 * nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(gpu_v[ir], ref_v[ir], 1.0e-10);
    }
    for (int ir = 0; ir < 3 * nrxx; ++ir)
    {
        EXPECT_NEAR(gpu_h[ir], ref_h[ir], 1.0e-10);
    }
    EXPECT_NEAR(gpu_etxc, ref_etxc, 1.0e-10);
    EXPECT_NEAR(gpu_vtxc, ref_vtxc, 1.0e-10);

    delmem_double_op()(d_rho);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_gdr);
    delmem_double_op()(d_v);
    delmem_double_op()(d_h);
    delmem_double_op()(d_sums);
}

TEST(XCResidentOpTest, ScalarPbeGpuMatchesReferenceValues)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 5;
    const double e2 = 2.0;
    const double epsr = 1.0e-10;
    const std::vector<double> rho = {0.17E+01, 0.17E+01, 0.15E+01, 0.88E-01, 0.18E+04};
    const std::vector<double> rho_core(nrxx, 0.0);
    const std::vector<double> ref_vxc = {-1.259358955, -1.259358955, -1.210234884, -0.4975768336, -12.12952188};

    std::vector<double> ref_total(nrxx, 0.0);
    std::vector<double> ref_v(nrxx, 0.0);
    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    xc_scalar_pbe_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                        nrxx,
                                                        e2,
                                                        epsr,
                                                        rho.data(),
                                                        rho_core.data(),
                                                        ref_total.data(),
                                                        ref_v.data(),
                                                        &ref_etxc,
                                                        &ref_vtxc);

    double* d_rho = nullptr;
    double* d_rho_core = nullptr;
    double* d_total = nullptr;
    double* d_v = nullptr;
    double* d_sums = nullptr;
    resmem_double_op()(d_rho, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_total, nrxx);
    resmem_double_op()(d_v, nrxx);
    resmem_double_op()(d_sums, 2);
    syncmem_h2d_op()(d_rho, rho.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);

    double etxc = 0.0;
    double vtxc = 0.0;
    xc_scalar_pbe_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                           nrxx,
                                                           e2,
                                                           epsr,
                                                           d_rho,
                                                           d_rho_core,
                                                           d_total,
                                                           d_v,
                                                           d_sums,
                                                           &etxc,
                                                           &vtxc);

    std::vector<double> v(nrxx, 0.0);
    std::vector<double> total(nrxx, 0.0);
    syncmem_d2h_op()(v.data(), d_v, nrxx);
    syncmem_d2h_op()(total.data(), d_total, nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        const double rhox = rho[ir] + rho_core[ir];
        EXPECT_NEAR(total[ir], rhox, 1.0e-14);
        EXPECT_NEAR(v[ir], ref_v[ir], 1.0e-12);
        EXPECT_NEAR(v[ir], e2 * ref_vxc[ir], 1.0e-8);
    }
    EXPECT_NEAR(etxc, ref_etxc, 1.0e-10);
    EXPECT_NEAR(vtxc, ref_vtxc, 1.0e-10);

    delmem_double_op()(d_rho);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_total);
    delmem_double_op()(d_v);
    delmem_double_op()(d_sums);
}

TEST(XCResidentOpTest, ScalarLdaSpinCpuMatchesReferenceValues)
{
    const int nrxx = 5;
    const double e2 = 2.0;
    const double epsr = 1.0e-10;
    const std::vector<double> rho = {0.17E+01, 0.17E+01, 0.15E+01, 0.88E-01, 0.18E+04};
    const std::vector<double> zeta = {0.0, 0.2, 0.5, 0.8, 1.0};
    const std::vector<double> rho_core(nrxx, 0.0);
    std::vector<double> rho_up(nrxx, 0.0);
    std::vector<double> rho_dw(nrxx, 0.0);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        rho_up[ir] = rho[ir] * (1.0 + zeta[ir]) * 0.5;
        rho_dw[ir] = rho[ir] * (1.0 - zeta[ir]) * 0.5;
    }

    const std::vector<double> pz_exc = {-0.95651735756525513,
                                        -0.96317186117476516,
                                        -0.95998479865326902,
                                        -0.41690422190075471,
                                        -11.392818565518368};
    const std::vector<double> pz_vup = {-1.2588131365004438,
                                        -1.3213867444561467,
                                        -1.3486549996482768,
                                        -0.56794400635336095,
                                        -15.170993784117668};
    const std::vector<double> pz_vdw = {-1.2588131365004438,
                                        -1.1877924103859072,
                                        -1.0155330910438201,
                                        -0.37092937962238148,
                                        -0.51758342826640091};
    const std::vector<double> pw_exc = {-0.9570906378, -0.9639575384, -0.961793486, -0.4179657552, -11.39185886};
    const std::vector<double> pw_vup = {-1.259358955, -1.323886052, -1.352864095, -0.5692699308, -15.16995135};
    const std::vector<double> pw_vdw = {-1.259358955, -1.186042827, -1.011084744, -0.3739376001, -0.6598499399};

    for (const int correlation : {0, 1})
    {
        std::vector<double> total_up(nrxx, 0.0);
        std::vector<double> total_dw(nrxx, 0.0);
        std::vector<double> v(2 * nrxx, 0.0);
        double etxc = 0.0;
        double vtxc = 0.0;
        xc_scalar_lda_spin_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                                 nrxx,
                                                                 correlation,
                                                                 e2,
                                                                 epsr,
                                                                 rho_up.data(),
                                                                 rho_dw.data(),
                                                                 rho_core.data(),
                                                                 total_up.data(),
                                                                 total_dw.data(),
                                                                 v.data(),
                                                                 &etxc,
                                                                 &vtxc);

        double ref_etxc = 0.0;
        double ref_vtxc = 0.0;
        for (int ir = 0; ir < nrxx; ++ir)
        {
            const auto& ref_exc = correlation == 0 ? pz_exc : pw_exc;
            const auto& ref_vup = correlation == 0 ? pz_vup : pw_vup;
            const auto& ref_vdw = correlation == 0 ? pz_vdw : pw_vdw;
            EXPECT_NEAR(total_up[ir], rho_up[ir], 1.0e-14);
            EXPECT_NEAR(total_dw[ir], rho_dw[ir], 1.0e-14);
            EXPECT_NEAR(v[ir], e2 * ref_vup[ir], 1.0e-8);
            EXPECT_NEAR(v[nrxx + ir], e2 * ref_vdw[ir], 1.0e-8);
            ref_etxc += e2 * ref_exc[ir] * rho[ir];
            ref_vtxc += e2 * ref_vup[ir] * rho_up[ir] + e2 * ref_vdw[ir] * rho_dw[ir];
        }
        EXPECT_NEAR(etxc, ref_etxc, 1.0e-4);
        EXPECT_NEAR(vtxc, ref_vtxc, 1.0e-4);
    }
}

TEST(XCResidentOpTest, ScalarLdaSpinGpuMatchesCpuWithCore)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 6;
    const double e2 = 2.0;
    const double epsr = 1.0e-10;
    const int correlation = 1;
    const std::vector<double> rho_up = {0.018, 0.024, 0.031, 0.045, 0.052, 0.063};
    const std::vector<double> rho_dw = {0.011, 0.019, 0.010, 0.026, 0.040, 0.003};
    const std::vector<double> rho_core = {0.002, 0.0015, 0.001, 0.0025, 0.003, 0.002};

    std::vector<double> ref_total_up(nrxx, 0.0);
    std::vector<double> ref_total_dw(nrxx, 0.0);
    std::vector<double> ref_v(2 * nrxx, 0.0);
    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    xc_scalar_lda_spin_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                             nrxx,
                                                             correlation,
                                                             e2,
                                                             epsr,
                                                             rho_up.data(),
                                                             rho_dw.data(),
                                                             rho_core.data(),
                                                             ref_total_up.data(),
                                                             ref_total_dw.data(),
                                                             ref_v.data(),
                                                             &ref_etxc,
                                                             &ref_vtxc);

    double* d_rho_up = nullptr;
    double* d_rho_dw = nullptr;
    double* d_rho_core = nullptr;
    double* d_total_up = nullptr;
    double* d_total_dw = nullptr;
    double* d_v = nullptr;
    double* d_sums = nullptr;
    resmem_double_op()(d_rho_up, nrxx);
    resmem_double_op()(d_rho_dw, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_total_up, nrxx);
    resmem_double_op()(d_total_dw, nrxx);
    resmem_double_op()(d_v, 2 * nrxx);
    resmem_double_op()(d_sums, 2);
    syncmem_h2d_op()(d_rho_up, rho_up.data(), nrxx);
    syncmem_h2d_op()(d_rho_dw, rho_dw.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);

    double gpu_etxc = 0.0;
    double gpu_vtxc = 0.0;
    xc_scalar_lda_spin_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                             nrxx,
                                                             correlation,
                                                             e2,
                                                             epsr,
                                                             d_rho_up,
                                                             d_rho_dw,
                                                             d_rho_core,
                                                             d_total_up,
                                                             d_total_dw,
                                                             d_v,
                                                             d_sums,
                                                             &gpu_etxc,
                                                             &gpu_vtxc);

    std::vector<double> total_up(nrxx, 0.0);
    std::vector<double> total_dw(nrxx, 0.0);
    std::vector<double> v(2 * nrxx, 0.0);
    syncmem_d2h_op()(total_up.data(), d_total_up, nrxx);
    syncmem_d2h_op()(total_dw.data(), d_total_dw, nrxx);
    syncmem_d2h_op()(v.data(), d_v, 2 * nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(total_up[ir], ref_total_up[ir], 1.0e-14);
        EXPECT_NEAR(total_dw[ir], ref_total_dw[ir], 1.0e-14);
        EXPECT_NEAR(v[ir], ref_v[ir], 1.0e-12);
        EXPECT_NEAR(v[nrxx + ir], ref_v[nrxx + ir], 1.0e-12);
    }
    EXPECT_NEAR(gpu_etxc, ref_etxc, 1.0e-12);
    EXPECT_NEAR(gpu_vtxc, ref_vtxc, 1.0e-12);

    delmem_double_op()(d_rho_up);
    delmem_double_op()(d_rho_dw);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_total_up);
    delmem_double_op()(d_total_dw);
    delmem_double_op()(d_v);
    delmem_double_op()(d_sums);
}

TEST(XCResidentOpTest, GridGpuAccumulatesIntoResidentPotential)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 4;
    const int iflag = 0;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    const std::vector<double> rho = {0.018, 0.024, 0.031, 0.045};
    const std::vector<double> rho_core = {0.002, 0.0015, 0.001, 0.0025};
    const std::vector<double> gdr = {0.011, 0.017, 0.019, -0.009, 0.021, 0.013,
                                     0.015, -0.014, 0.026, 0.010, -0.018, 0.022};
    const std::vector<double> initial_v = {1.0, 2.0, 3.0, 4.0};

    std::vector<double> ref_v_delta(nrxx, 0.0);
    std::vector<double> ref_h(3 * nrxx, 0.0);
    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                               nrxx,
                                                               iflag,
                                                               e2,
                                                               epsr,
                                                               rho.data(),
                                                               rho_core.data(),
                                                               gdr.data(),
                                                               ref_v_delta.data(),
                                                               ref_h.data(),
                                                               &ref_etxc,
                                                               &ref_vtxc);

    double* d_rho = nullptr;
    double* d_rho_core = nullptr;
    double* d_gdr = nullptr;
    double* d_v = nullptr;
    double* d_h = nullptr;
    double* d_sums = nullptr;
    resmem_double_op()(d_rho, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_v, nrxx);
    resmem_double_op()(d_h, 3 * nrxx);
    resmem_double_op()(d_sums, 2);
    syncmem_h2d_op()(d_rho, rho.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);
    syncmem_h2d_op()(d_gdr, gdr.data(), 3 * nrxx);
    syncmem_h2d_op()(d_v, initial_v.data(), nrxx);

    double etxc = 0.0;
    double vtxc = 0.0;
    xc_gradcorr_pbe_grid_resident_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                        nrxx,
                                                                        iflag,
                                                                        e2,
                                                                        epsr,
                                                                        d_rho,
                                                                        d_rho_core,
                                                                        d_gdr,
                                                                        d_v,
                                                                        d_h,
                                                                        d_sums,
                                                                        &etxc,
                                                                        &vtxc);

    std::vector<double> v(nrxx, 0.0);
    std::vector<double> h(3 * nrxx, 0.0);
    syncmem_d2h_op()(v.data(), d_v, nrxx);
    syncmem_d2h_op()(h.data(), d_h, 3 * nrxx);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(v[ir], initial_v[ir] + ref_v_delta[ir], 1.0e-10);
    }
    for (int ir = 0; ir < 3 * nrxx; ++ir)
    {
        EXPECT_NEAR(h[ir], ref_h[ir], 1.0e-10);
    }
    EXPECT_NEAR(etxc, ref_etxc, 1.0e-10);
    EXPECT_NEAR(vtxc, ref_vtxc, 1.0e-10);

    delmem_double_op()(d_rho);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_gdr);
    delmem_double_op()(d_v);
    delmem_double_op()(d_h);
    delmem_double_op()(d_sums);
}

TEST(XCResidentOpTest, StressPbeGpuMatchesCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 6;
    const int iflag = 0;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    const std::vector<double> rho_total = {0.020, 0.0255, 0.032, 0.0475, 0.055, 0.065};
    const std::vector<double> gdr = {0.011, 0.017, 0.019, -0.009, 0.021, 0.013,
                                     0.015, -0.014, 0.026, 0.010, -0.018, 0.022,
                                     -0.020, 0.012, 0.016, 0.024, 0.019, -0.011};

    std::vector<double> ref_stress(9, 0.0);
    xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                                 nrxx,
                                                                 iflag,
                                                                 e2,
                                                                 epsr,
                                                                 rho_total.data(),
                                                                 gdr.data(),
                                                                 ref_stress.data());

    double* d_rho_total = nullptr;
    double* d_gdr = nullptr;
    double* d_stress = nullptr;
    resmem_double_op()(d_rho_total, nrxx);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_stress, 9);
    syncmem_h2d_op()(d_rho_total, rho_total.data(), nrxx);
    syncmem_h2d_op()(d_gdr, gdr.data(), 3 * nrxx);

    xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                 nrxx,
                                                                 iflag,
                                                                 e2,
                                                                 epsr,
                                                                 d_rho_total,
                                                                 d_gdr,
                                                                 d_stress);

    std::vector<double> stress(9, 0.0);
    syncmem_d2h_op()(stress.data(), d_stress, 9);
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(stress[i], ref_stress[i], 1.0e-11);
    }

    delmem_double_op()(d_rho_total);
    delmem_double_op()(d_gdr);
    delmem_double_op()(d_stress);
}

TEST(XCResidentOpTest, StressPbesolGpuMatchesCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 6;
    const int iflag = 2;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    const std::vector<double> rho_total = {0.020, 0.0255, 0.032, 0.0475, 0.055, 0.065};
    const std::vector<double> gdr = {0.011, 0.017, 0.019, -0.009, 0.021, 0.013,
                                     0.015, -0.014, 0.026, 0.010, -0.018, 0.022,
                                     -0.020, 0.012, 0.016, 0.024, 0.019, -0.011};

    std::vector<double> ref_stress(9, 0.0);
    xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                                 nrxx,
                                                                 iflag,
                                                                 e2,
                                                                 epsr,
                                                                 rho_total.data(),
                                                                 gdr.data(),
                                                                 ref_stress.data());

    double* d_rho_total = nullptr;
    double* d_gdr = nullptr;
    double* d_stress = nullptr;
    resmem_double_op()(d_rho_total, nrxx);
    resmem_double_op()(d_gdr, 3 * nrxx);
    resmem_double_op()(d_stress, 9);
    syncmem_h2d_op()(d_rho_total, rho_total.data(), nrxx);
    syncmem_h2d_op()(d_gdr, gdr.data(), 3 * nrxx);

    xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                 nrxx,
                                                                 iflag,
                                                                 e2,
                                                                 epsr,
                                                                 d_rho_total,
                                                                 d_gdr,
                                                                 d_stress);

    std::vector<double> stress(9, 0.0);
    syncmem_d2h_op()(stress.data(), d_stress, 9);
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_NEAR(stress[i], ref_stress[i], 1.0e-11);
    }

    delmem_double_op()(d_rho_total);
    delmem_double_op()(d_gdr);
    delmem_double_op()(d_stress);
}

TEST(XCResidentOpTest, StressSpinPbeGpuMatchesCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 5;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    const std::vector<double> rho_up_total = {0.018, 0.024, 0.031, 0.045, 0.052};
    const std::vector<double> rho_dw_total = {0.011, 0.019, 0.010, 0.026, 0.040};
    const std::vector<double> gdr_up = {0.011, 0.017, 0.019, -0.009, 0.021, 0.013,
                                        0.015, -0.014, 0.026, 0.010, -0.018, 0.022,
                                        -0.020, 0.012, 0.016};
    const std::vector<double> gdr_dw = {-0.010, 0.013, 0.016, 0.018, -0.012, 0.017,
                                        0.014, 0.011, -0.019, -0.015, 0.016, 0.020,
                                        0.013, -0.017, 0.012};

    double* d_rho_up_total = nullptr;
    double* d_rho_dw_total = nullptr;
    double* d_gdr_up = nullptr;
    double* d_gdr_dw = nullptr;
    double* d_stress = nullptr;
    resmem_double_op()(d_rho_up_total, nrxx);
    resmem_double_op()(d_rho_dw_total, nrxx);
    resmem_double_op()(d_gdr_up, 3 * nrxx);
    resmem_double_op()(d_gdr_dw, 3 * nrxx);
    resmem_double_op()(d_stress, 9);
    syncmem_h2d_op()(d_rho_up_total, rho_up_total.data(), nrxx);
    syncmem_h2d_op()(d_rho_dw_total, rho_dw_total.data(), nrxx);
    syncmem_h2d_op()(d_gdr_up, gdr_up.data(), 3 * nrxx);
    syncmem_h2d_op()(d_gdr_dw, gdr_dw.data(), 3 * nrxx);

    for (const int iflag: {0, 2})
    {
        std::vector<double> ref_stress(9, 0.0);
        xc_gradcorr_pbe_spin_stress_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                                          nrxx,
                                                                          iflag,
                                                                          e2,
                                                                          epsr,
                                                                          rho_up_total.data(),
                                                                          rho_dw_total.data(),
                                                                          gdr_up.data(),
                                                                          gdr_dw.data(),
                                                                          ref_stress.data());

        xc_gradcorr_pbe_spin_stress_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                          nrxx,
                                                                          iflag,
                                                                          e2,
                                                                          epsr,
                                                                          d_rho_up_total,
                                                                          d_rho_dw_total,
                                                                          d_gdr_up,
                                                                          d_gdr_dw,
                                                                          d_stress);

        std::vector<double> stress(9, 0.0);
        syncmem_d2h_op()(stress.data(), d_stress, 9);
        for (int i = 0; i < 9; ++i)
        {
            EXPECT_NEAR(stress[i], ref_stress[i], 1.0e-11);
        }
    }

    delmem_double_op()(d_rho_up_total);
    delmem_double_op()(d_rho_dw_total);
    delmem_double_op()(d_gdr_up);
    delmem_double_op()(d_gdr_dw);
    delmem_double_op()(d_stress);
}

TEST(XCResidentOpTest, AddPotentialGpuAccumulatesIntoDeviceBuffer)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int size = 6;
    const std::vector<double> src = {0.1, -0.2, 0.3, 0.4, -0.5, 0.6};
    const std::vector<double> dst0 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    double* d_src = nullptr;
    double* d_dst = nullptr;
    resmem_double_op()(d_src, size);
    resmem_double_op()(d_dst, size);
    syncmem_h2d_op()(d_src, src.data(), size);
    syncmem_h2d_op()(d_dst, dst0.data(), size);

    xc_add_potential_op<double, base_device::DEVICE_GPU>()(nullptr, size, d_src, d_dst);

    std::vector<double> dst(size, 0.0);
    syncmem_d2h_op()(dst.data(), d_dst, size);
    for (int i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(dst[i], dst0[i] + src[i]);
    }

    delmem_double_op()(d_src);
    delmem_double_op()(d_dst);
}

TEST(XCResidentOpTest, ChargeRealspaceDensitySyncIsNoopOnCpuDevice)
{
    constexpr int nrxx = 4;
    ModulePW::PW_Basis rhopw;
    rhopw.nrxx = nrxx;
    rhopw.npw = nrxx;
    rhopw.nmaxgr = nrxx;
    rhopw.nxyz = nrxx;
    rhopw.set_device("cpu");

    Charge chr;
    chr.set_rhopw(&rhopw);
    chr.set_device("cpu");
    chr.allocate(2, false);
    for (int ir = 0; ir < nrxx; ++ir)
    {
        chr.rho[0][ir] = 0.1 * (ir + 1);
        chr.rho[1][ir] = 0.05 * (ir + 1);
    }

    chr.sync_realspace_density_to_device();

    EXPECT_EQ(chr.get_device(), "cpu");
    EXPECT_EQ(chr.get_rho_d(0), nullptr);
    EXPECT_EQ(chr.get_rho_d(1), nullptr);
}

TEST(XCResidentOpTest, FullVxcLdaSpinResidentGpuMatchesCpu)
{
    ScopedMpiInit scoped_mpi;
    const int old_nspin = PARAM.input.nspin;
    const char* old_env_value = std::getenv("ABACUS_XC_GPU");
    const bool had_xc_gpu_env = old_env_value != nullptr;
    const std::string old_xc_gpu_env = had_xc_gpu_env ? std::string(old_env_value) : std::string();

    constexpr int nrxx = 6;
    ModulePW::PW_Basis rhopw;
    rhopw.nrxx = nrxx;
    rhopw.npw = nrxx;
    rhopw.nmaxgr = nrxx;
    rhopw.nxyz = nrxx;
    rhopw.set_device("gpu");

    UnitCell ucell;
    ucell.omega = 12.0;
    ucell.tpiba = 1.0;

    Charge chr;
    chr.set_rhopw(&rhopw);
    chr.set_device("gpu");
    chr.allocate(2, false);

    const std::vector<double> rho_up = {0.018, 0.024, 0.031, 0.045, 0.052, 0.063};
    const std::vector<double> rho_dw = {0.011, 0.019, 0.010, 0.026, 0.040, 0.003};
    const std::vector<double> rho_core = {0.002, 0.0015, 0.001, 0.0025, 0.003, 0.002};
    for (int ir = 0; ir < nrxx; ++ir)
    {
        chr.rho[0][ir] = rho_up[ir];
        chr.rho[1][ir] = rho_dw[ir];
        chr.rho_core[ir] = rho_core[ir];
    }
    chr.sync_realspace_density_to_device();

    PARAM.input.nspin = 2;
    XC_Functional::set_xc_type("PZ");

    setenv("ABACUS_XC_GPU", "0", 1);
    const auto cpu_result = XC_Functional::v_xc(nrxx, &chr, &ucell, "cpu");

    setenv("ABACUS_XC_GPU", "1", 1);
    const auto gpu_result = XC_Functional::v_xc(nrxx, &chr, &ucell, "gpu");

    const double cpu_etxc = std::get<0>(cpu_result);
    const double cpu_vtxc = std::get<1>(cpu_result);
    const ModuleBase::matrix& cpu_v = std::get<2>(cpu_result);
    const double gpu_etxc = std::get<0>(gpu_result);
    const double gpu_vtxc = std::get<1>(gpu_result);
    const ModuleBase::matrix& gpu_v = std::get<2>(gpu_result);

    EXPECT_NEAR(gpu_etxc, cpu_etxc, 1.0e-12);
    EXPECT_NEAR(gpu_vtxc, cpu_vtxc, 1.0e-12);
    for (int is = 0; is < 2; ++is)
    {
        for (int ir = 0; ir < nrxx; ++ir)
        {
            EXPECT_NEAR(gpu_v(is, ir), cpu_v(is, ir), 1.0e-12);
        }
    }

    PARAM.input.nspin = old_nspin;
    if (had_xc_gpu_env)
    {
        setenv("ABACUS_XC_GPU", old_xc_gpu_env.c_str(), 1);
    }
    else
    {
        unsetenv("ABACUS_XC_GPU");
    }
}

TEST(XCResidentOpTest, SpinPbeGridGpuMatchesCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 5;
    const int iflag = 0;
    const double e2 = 2.0;
    const double epsr = 1.0e-6;
    const std::vector<double> rho_up = {0.018, 0.024, 0.031, 0.045, 0.052};
    const std::vector<double> rho_dw = {0.011, 0.019, 0.010, 0.026, 0.004};
    const std::vector<double> rho_core = {0.002, 0.0015, 0.001, 0.0025, 0.003};
    const std::vector<double> gdr_up = {0.011, 0.017, 0.019, -0.009, 0.021,
                                        0.013, 0.015, -0.014, 0.026, 0.010,
                                        -0.018, 0.022, -0.020, 0.012, 0.016};
    const std::vector<double> gdr_dw = {-0.006, 0.010, 0.012, 0.014, -0.011,
                                        0.008, 0.017, 0.019, -0.007, 0.009,
                                        0.004, -0.013, 0.018, 0.006, -0.010};
    const std::vector<double> initial_v = {1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0};

    std::vector<double> ref_v = initial_v;
    std::vector<double> ref_h_up(3 * nrxx, 0.0);
    std::vector<double> ref_h_dw(3 * nrxx, 0.0);
    double ref_etxc = 0.0;
    double ref_vtxc = 0.0;
    xc_gradcorr_pbe_spin_grid_resident_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                                             nrxx,
                                                                             iflag,
                                                                             e2,
                                                                             epsr,
                                                                             rho_up.data(),
                                                                             rho_dw.data(),
                                                                             rho_core.data(),
                                                                             gdr_up.data(),
                                                                             gdr_dw.data(),
                                                                             ref_v.data(),
                                                                             ref_h_up.data(),
                                                                             ref_h_dw.data(),
                                                                             &ref_etxc,
                                                                             &ref_vtxc);

    double* d_rho_up = nullptr;
    double* d_rho_dw = nullptr;
    double* d_rho_core = nullptr;
    double* d_gdr_up = nullptr;
    double* d_gdr_dw = nullptr;
    double* d_v = nullptr;
    double* d_h_up = nullptr;
    double* d_h_dw = nullptr;
    double* d_sums = nullptr;
    resmem_double_op()(d_rho_up, nrxx);
    resmem_double_op()(d_rho_dw, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_gdr_up, 3 * nrxx);
    resmem_double_op()(d_gdr_dw, 3 * nrxx);
    resmem_double_op()(d_v, 2 * nrxx);
    resmem_double_op()(d_h_up, 3 * nrxx);
    resmem_double_op()(d_h_dw, 3 * nrxx);
    resmem_double_op()(d_sums, 2);
    syncmem_h2d_op()(d_rho_up, rho_up.data(), nrxx);
    syncmem_h2d_op()(d_rho_dw, rho_dw.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);
    syncmem_h2d_op()(d_gdr_up, gdr_up.data(), 3 * nrxx);
    syncmem_h2d_op()(d_gdr_dw, gdr_dw.data(), 3 * nrxx);
    syncmem_h2d_op()(d_v, initial_v.data(), 2 * nrxx);

    double gpu_etxc = 0.0;
    double gpu_vtxc = 0.0;
    xc_gradcorr_pbe_spin_grid_resident_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                                             nrxx,
                                                                             iflag,
                                                                             e2,
                                                                             epsr,
                                                                             d_rho_up,
                                                                             d_rho_dw,
                                                                             d_rho_core,
                                                                             d_gdr_up,
                                                                             d_gdr_dw,
                                                                             d_v,
                                                                             d_h_up,
                                                                             d_h_dw,
                                                                             d_sums,
                                                                             &gpu_etxc,
                                                                             &gpu_vtxc);

    std::vector<double> v(2 * nrxx, 0.0);
    std::vector<double> h_up(3 * nrxx, 0.0);
    std::vector<double> h_dw(3 * nrxx, 0.0);
    syncmem_d2h_op()(v.data(), d_v, 2 * nrxx);
    syncmem_d2h_op()(h_up.data(), d_h_up, 3 * nrxx);
    syncmem_d2h_op()(h_dw.data(), d_h_dw, 3 * nrxx);

    for (int ir = 0; ir < 2 * nrxx; ++ir)
    {
        EXPECT_NEAR(v[ir], ref_v[ir], 1.0e-10);
    }
    for (int ir = 0; ir < 3 * nrxx; ++ir)
    {
        EXPECT_NEAR(h_up[ir], ref_h_up[ir], 1.0e-10);
        EXPECT_NEAR(h_dw[ir], ref_h_dw[ir], 1.0e-10);
    }
    EXPECT_NEAR(gpu_etxc, ref_etxc, 1.0e-10);
    EXPECT_NEAR(gpu_vtxc, ref_vtxc, 1.0e-10);

    delmem_double_op()(d_rho_up);
    delmem_double_op()(d_rho_dw);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_gdr_up);
    delmem_double_op()(d_gdr_dw);
    delmem_double_op()(d_v);
    delmem_double_op()(d_h_up);
    delmem_double_op()(d_h_dw);
    delmem_double_op()(d_sums);
}

TEST(XCResidentOpTest, ApplyDhGpuUpdatesPotentialAndVtxc)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 5;
    const std::vector<double> rho = {0.18, 0.24, 0.31, 0.45, 0.52};
    const std::vector<double> rho_core = {0.02, 0.015, 0.01, 0.025, 0.03};
    const std::vector<double> dh = {0.5, -0.25, 0.125, -0.0625, 0.03125};
    const std::vector<double> initial_v = {1.0, 1.5, 2.0, 2.5, 3.0};

    double* d_rho = nullptr;
    double* d_rho_core = nullptr;
    double* d_dh = nullptr;
    double* d_v = nullptr;
    double* d_sum = nullptr;
    resmem_double_op()(d_rho, nrxx);
    resmem_double_op()(d_rho_core, nrxx);
    resmem_double_op()(d_dh, nrxx);
    resmem_double_op()(d_v, nrxx);
    resmem_double_op()(d_sum, 1);
    syncmem_h2d_op()(d_rho, rho.data(), nrxx);
    syncmem_h2d_op()(d_rho_core, rho_core.data(), nrxx);
    syncmem_h2d_op()(d_dh, dh.data(), nrxx);
    syncmem_h2d_op()(d_v, initial_v.data(), nrxx);

    double vtxc_delta = 0.0;
    xc_apply_dh_op<double, base_device::DEVICE_GPU>()(nullptr,
                                                      nrxx,
                                                      d_rho,
                                                      d_rho_core,
                                                      d_dh,
                                                      d_v,
                                                      d_sum,
                                                      &vtxc_delta);

    std::vector<double> v(nrxx, 0.0);
    syncmem_d2h_op()(v.data(), d_v, nrxx);
    double ref_vtxc_delta = 0.0;
    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(v[ir], initial_v[ir] - dh[ir], 1.0e-12);
        ref_vtxc_delta -= dh[ir] * (rho[ir] - rho_core[ir]);
    }
    EXPECT_NEAR(vtxc_delta, ref_vtxc_delta, 1.0e-12);

    delmem_double_op()(d_rho);
    delmem_double_op()(d_rho_core);
    delmem_double_op()(d_dh);
    delmem_double_op()(d_v);
    delmem_double_op()(d_sum);
}

TEST(XCResidentOpTest, ReciprocalGradientHelpersMatchCpuFormula)
{
    using complex_t = std::complex<double>;
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using resmem_complex_op = base_device::memory::resize_memory_op<complex_t, base_device::DEVICE_GPU>;
    using delmem_complex_op = base_device::memory::delete_memory_op<complex_t, base_device::DEVICE_GPU>;
    using sync_double_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using sync_double_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
    using sync_complex_h2d_op = base_device::memory::synchronize_memory_op<complex_t, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using sync_complex_d2h_op = base_device::memory::synchronize_memory_op<complex_t, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int npw = 4;
    const int nrxx = 3;
    const std::vector<double> gcar = {0.2, 0.3, -0.1, 1.1, -0.7, 0.4,
                                      -0.5, 0.9, 0.8, 0.0, -1.2, 0.6};
    const std::vector<complex_t> rhog = {{0.5, -0.25}, {-0.75, 0.125}, {1.25, 0.5}, {-0.125, -0.875}};
    const std::vector<double> interleaved = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    const std::vector<double> component = {-1.0, -2.0, -3.0};

    double* d_gcar = nullptr;
    double* d_interleaved = nullptr;
    double* d_component = nullptr;
    complex_t* d_rhog = nullptr;
    complex_t* d_porter = nullptr;
    complex_t* d_accum = nullptr;
    resmem_double_op()(d_gcar, 3 * npw);
    resmem_double_op()(d_interleaved, 3 * nrxx);
    resmem_double_op()(d_component, nrxx);
    resmem_complex_op()(d_rhog, npw);
    resmem_complex_op()(d_porter, npw);
    resmem_complex_op()(d_accum, npw);
    sync_double_h2d_op()(d_gcar, gcar.data(), 3 * npw);
    sync_double_h2d_op()(d_interleaved, interleaved.data(), 3 * nrxx);
    sync_double_h2d_op()(d_component, component.data(), nrxx);
    sync_complex_h2d_op()(d_rhog, rhog.data(), npw);

    xc_multiply_iG_op<double, base_device::DEVICE_GPU>()(nullptr, npw, 1, d_gcar, d_rhog, d_porter);
    xc_accumulate_iG_op<double, base_device::DEVICE_GPU>()(nullptr, npw, 0, d_gcar, d_rhog, d_accum, true);
    xc_accumulate_iG_op<double, base_device::DEVICE_GPU>()(nullptr, npw, 2, d_gcar, d_rhog, d_accum, false);
    xc_set_component_op<double, base_device::DEVICE_GPU>()(nullptr, nrxx, 1, d_component, d_interleaved);
    xc_extract_component_op<double, base_device::DEVICE_GPU>()(nullptr, nrxx, 1, d_interleaved, d_component);

    std::vector<complex_t> porter(npw);
    std::vector<complex_t> accum(npw);
    std::vector<double> out_component(nrxx);
    std::vector<double> out_interleaved(3 * nrxx);
    sync_complex_d2h_op()(porter.data(), d_porter, npw);
    sync_complex_d2h_op()(accum.data(), d_accum, npw);
    sync_double_d2h_op()(out_component.data(), d_component, nrxx);
    sync_double_d2h_op()(out_interleaved.data(), d_interleaved, 3 * nrxx);

    const complex_t imaginary(0.0, 1.0);
    for (int ig = 0; ig < npw; ++ig)
    {
        EXPECT_NEAR(porter[ig].real(), (imaginary * rhog[ig] * gcar[3 * ig + 1]).real(), 1.0e-14);
        EXPECT_NEAR(porter[ig].imag(), (imaginary * rhog[ig] * gcar[3 * ig + 1]).imag(), 1.0e-14);
        const complex_t ref_accum = imaginary * rhog[ig] * (gcar[3 * ig] + gcar[3 * ig + 2]);
        EXPECT_NEAR(accum[ig].real(), ref_accum.real(), 1.0e-14);
        EXPECT_NEAR(accum[ig].imag(), ref_accum.imag(), 1.0e-14);
    }
    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(out_component[ir], component[ir], 1.0e-14);
        EXPECT_NEAR(out_interleaved[3 * ir], interleaved[3 * ir], 1.0e-14);
        EXPECT_NEAR(out_interleaved[3 * ir + 1], component[ir], 1.0e-14);
        EXPECT_NEAR(out_interleaved[3 * ir + 2], interleaved[3 * ir + 2], 1.0e-14);
    }

    delmem_double_op()(d_gcar);
    delmem_double_op()(d_interleaved);
    delmem_double_op()(d_component);
    delmem_complex_op()(d_rhog);
    delmem_complex_op()(d_porter);
    delmem_complex_op()(d_accum);
}

TEST(XCResidentOpTest, NoncolinRhoAndRotateGpuMatchCpu)
{
    using resmem_double_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
    using delmem_double_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
    using syncmem_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
    using syncmem_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;

    const int nrxx = 5;
    const std::vector<double> rho0 = {1.0, 1.2, 0.9, 1.5, 0.8};
    const std::vector<double> rho1 = {0.20, -0.10, 0.00, 1.0e-14, -0.30};
    const std::vector<double> rho2 = {0.10, 0.15, 0.00, -1.0e-14, 0.05};
    const std::vector<double> rho3 = {-0.05, 0.05, 0.00, 1.0e-14, 0.20};
    const std::vector<double> ux = {0.5773502691896258, 0.5773502691896258, 0.5773502691896258};
    const std::vector<double> v_up = {-0.4, -0.3, -0.2, -0.1, -0.05};
    const std::vector<double> v_dw = {-0.5, -0.45, -0.25, -0.15, -0.10};
    const std::vector<double> initial_v0 = {1.0, 2.0, 3.0, 4.0, 5.0};
    const std::vector<double> initial_v1 = {0.5, 0.4, 0.3, 0.2, 0.1};
    const std::vector<double> initial_v2 = {-0.5, -0.4, -0.3, -0.2, -0.1};
    const std::vector<double> initial_v3 = {0.1, 0.2, 0.3, 0.4, 0.5};

    std::vector<double> ref_up(nrxx, 0.0);
    std::vector<double> ref_dw(nrxx, 0.0);
    std::vector<double> ref_neg(nrxx, 0.0);
    xc_noncolin_rho_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                          nrxx,
                                                          true,
                                                          rho0.data(),
                                                          rho1.data(),
                                                          rho2.data(),
                                                          rho3.data(),
                                                          ux.data(),
                                                          ref_up.data(),
                                                          ref_dw.data(),
                                                          ref_neg.data());
    std::vector<double> ref_v0 = initial_v0;
    std::vector<double> ref_v1 = initial_v1;
    std::vector<double> ref_v2 = initial_v2;
    std::vector<double> ref_v3 = initial_v3;
    xc_noncolin_rotate_potential_op<double, base_device::DEVICE_CPU>()(nullptr,
                                                                       nrxx,
                                                                       rho1.data(),
                                                                       rho2.data(),
                                                                       rho3.data(),
                                                                       ref_neg.data(),
                                                                       v_up.data(),
                                                                       v_dw.data(),
                                                                       ref_v0.data(),
                                                                       ref_v1.data(),
                                                                       ref_v2.data(),
                                                                       ref_v3.data());

    double* d_rho0 = nullptr;
    double* d_rho1 = nullptr;
    double* d_rho2 = nullptr;
    double* d_rho3 = nullptr;
    double* d_ux = nullptr;
    double* d_up = nullptr;
    double* d_dw = nullptr;
    double* d_neg = nullptr;
    double* d_v_up = nullptr;
    double* d_v_dw = nullptr;
    double* d_v0 = nullptr;
    double* d_v1 = nullptr;
    double* d_v2 = nullptr;
    double* d_v3 = nullptr;
    resmem_double_op()(d_rho0, nrxx);
    resmem_double_op()(d_rho1, nrxx);
    resmem_double_op()(d_rho2, nrxx);
    resmem_double_op()(d_rho3, nrxx);
    resmem_double_op()(d_ux, 3);
    resmem_double_op()(d_up, nrxx);
    resmem_double_op()(d_dw, nrxx);
    resmem_double_op()(d_neg, nrxx);
    resmem_double_op()(d_v_up, nrxx);
    resmem_double_op()(d_v_dw, nrxx);
    resmem_double_op()(d_v0, nrxx);
    resmem_double_op()(d_v1, nrxx);
    resmem_double_op()(d_v2, nrxx);
    resmem_double_op()(d_v3, nrxx);
    syncmem_h2d_op()(d_rho0, rho0.data(), nrxx);
    syncmem_h2d_op()(d_rho1, rho1.data(), nrxx);
    syncmem_h2d_op()(d_rho2, rho2.data(), nrxx);
    syncmem_h2d_op()(d_rho3, rho3.data(), nrxx);
    syncmem_h2d_op()(d_ux, ux.data(), 3);
    syncmem_h2d_op()(d_v_up, v_up.data(), nrxx);
    syncmem_h2d_op()(d_v_dw, v_dw.data(), nrxx);
    syncmem_h2d_op()(d_v0, initial_v0.data(), nrxx);
    syncmem_h2d_op()(d_v1, initial_v1.data(), nrxx);
    syncmem_h2d_op()(d_v2, initial_v2.data(), nrxx);
    syncmem_h2d_op()(d_v3, initial_v3.data(), nrxx);

    xc_noncolin_rho_op<double, base_device::DEVICE_GPU>()(
        nullptr, nrxx, true, d_rho0, d_rho1, d_rho2, d_rho3, d_ux, d_up, d_dw, d_neg);
    xc_noncolin_rotate_potential_op<double, base_device::DEVICE_GPU>()(
        nullptr, nrxx, d_rho1, d_rho2, d_rho3, d_neg, d_v_up, d_v_dw, d_v0, d_v1, d_v2, d_v3);

    std::vector<double> up(nrxx, 0.0);
    std::vector<double> dw(nrxx, 0.0);
    std::vector<double> neg(nrxx, 0.0);
    std::vector<double> out_v0(nrxx, 0.0);
    std::vector<double> out_v1(nrxx, 0.0);
    std::vector<double> out_v2(nrxx, 0.0);
    std::vector<double> out_v3(nrxx, 0.0);
    syncmem_d2h_op()(up.data(), d_up, nrxx);
    syncmem_d2h_op()(dw.data(), d_dw, nrxx);
    syncmem_d2h_op()(neg.data(), d_neg, nrxx);
    syncmem_d2h_op()(out_v0.data(), d_v0, nrxx);
    syncmem_d2h_op()(out_v1.data(), d_v1, nrxx);
    syncmem_d2h_op()(out_v2.data(), d_v2, nrxx);
    syncmem_d2h_op()(out_v3.data(), d_v3, nrxx);

    for (int ir = 0; ir < nrxx; ++ir)
    {
        EXPECT_NEAR(up[ir], ref_up[ir], 1.0e-14);
        EXPECT_NEAR(dw[ir], ref_dw[ir], 1.0e-14);
        EXPECT_NEAR(neg[ir], ref_neg[ir], 1.0e-14);
        EXPECT_NEAR(out_v0[ir], ref_v0[ir], 1.0e-14);
        EXPECT_NEAR(out_v1[ir], ref_v1[ir], 1.0e-14);
        EXPECT_NEAR(out_v2[ir], ref_v2[ir], 1.0e-14);
        EXPECT_NEAR(out_v3[ir], ref_v3[ir], 1.0e-14);
    }

    delmem_double_op()(d_rho0);
    delmem_double_op()(d_rho1);
    delmem_double_op()(d_rho2);
    delmem_double_op()(d_rho3);
    delmem_double_op()(d_ux);
    delmem_double_op()(d_up);
    delmem_double_op()(d_dw);
    delmem_double_op()(d_neg);
    delmem_double_op()(d_v_up);
    delmem_double_op()(d_v_dw);
    delmem_double_op()(d_v0);
    delmem_double_op()(d_v1);
    delmem_double_op()(d_v2);
    delmem_double_op()(d_v3);
}
#endif

TYPED_TEST(XC_FunctionalOpTest, xc_functional_grad_wfc_op) {
    using Type = typename std::tuple_element<0, decltype(TypeParam())>::type;
    using Device = typename std::tuple_element<1, decltype(TypeParam())>::type;
    using Real = typename GetTypeReal<Type>::type;
    
    const int ik = 0;
    const int ipol = 0;
    const int npw = 4;
    const int npwx = 4;
    const int nrxx = 4;
    const Real tpiba = 3.3249187157734292;

    ct::Tensor gcar = std::move(ct::Tensor(
               {static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(-0.46354790605367302),
                static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(-0.30903193736911533),
                static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(-0.15451596868455766),
                static_cast<Real>(0), static_cast<Real>(0), static_cast<Real>(0)}).to_device<Device>());

    ct::Tensor kvec_c = std::move(ct::Tensor({static_cast<Real>(0)}).to_device<Device>());

    ct::Tensor rhog = std::move(ct::Tensor(
               {static_cast<Type>(-3.0017116103913875e-05, 7.1056289872527673e-08),
                static_cast<Type>(-7.5546991753311607e-05, -2.3428181621642512e-07),
                static_cast<Type>(-0.00092404657246742273, 1.604946087155834e-06),
                static_cast<Type>(0.98692408978015989, 1.8132214954316574e-09)}).to_device<Device>());

    ct::Tensor expected_porter = std::move(ct::Tensor(
               {static_cast<Type>(0),
                static_cast<Type>(0),
                static_cast<Type>(0),
                static_cast<Type>(0)}).to_device<Device>());
    
    auto porter = expected_porter;

    ct::Tensor porter_after = std::move(ct::Tensor(
               {static_cast<Type>(1.909712182465086e-07, 3.6064630015014698e-16),
                static_cast<Type>(-1.6872327080003407e-07, -3.4645680421774294e-16),
                static_cast<Type>(-2.3926040075122384e-07, -5.37330596683816e-16),
                static_cast<Type>(1.5465312289181674e-08, -8.7169854667834556e-17)}).to_device<Device>());

    ct::Tensor expected_grad = porter;
    auto grad = expected_grad;
    grad.zero();

    auto xc_functional_grad_wfc_solver = xc_functional_grad_wfc_op<Type, typename ct::ContainerToPsi<Device>::type>();
    
    xc_functional_grad_wfc_solver(
        ik, ipol, npw, npwx, // Integers
		tpiba,	// Double
        gcar.data<Real>(),   // Array of Real
        kvec_c.data<Real>(), // Array of double
		rhog.data<Type>(), porter.data<Type>());    // Array of std::complex<double>

    EXPECT_EQ(porter, expected_porter);


	xc_functional_grad_wfc_solver(
        ipol, nrxx,	// Integers
		porter_after.data<Type>(), grad.data<Type>());	// Array of std::complex<double>

    EXPECT_EQ(grad, expected_grad);
}

} // namespace hamilt

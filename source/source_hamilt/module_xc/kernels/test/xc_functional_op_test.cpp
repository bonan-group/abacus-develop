#include <source_hamilt/module_xc/kernels/xc_functional_op.h>
#include <source_hamilt/module_xc/kernels/xc_gradcorr_op.h>
#include <source_hamilt/module_xc/xc_gpu_policy.h>

#include <base/utils/gtest.h>
#include <source_base/module_device/memory_op.h>
#include <ATen/core/tensor.h>

namespace hamilt {

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

    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PBE"));
    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PZ"));
    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "pbesol"));
    EXPECT_TRUE(xc_gpu_policy(true, false, 1, "LDA"));
    EXPECT_FALSE(xc_gpu_policy(false, false, 1, "PBE"));
    EXPECT_FALSE(xc_gpu_policy(true, true, 1, "PBE"));
    EXPECT_FALSE(xc_gpu_policy(true, false, 2, "PBE"));
    EXPECT_FALSE(xc_gpu_policy(true, false, 1, "SCAN"));
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

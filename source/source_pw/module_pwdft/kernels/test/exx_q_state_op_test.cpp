#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"
#include "source_pw/module_pwdft/kernels/mul_potential_op.h"
#include "source_pw/module_pwdft/kernels/vec_mul_vec_complex_op.h"

#include <complex>
#include <gtest/gtest.h>
#include <vector>

class TestModuleHamiltExxQState : public ::testing::Test
{
  protected:
    using complexd = std::complex<double>;
    using conjugate_cpu_op = hamilt::exx_conjugate_real_op<complexd, base_device::DEVICE_CPU>;
    using gather_cpu_op = hamilt::exx_gather_recip_op<complexd, base_device::DEVICE_CPU>;
    using scatter_cpu_op = hamilt::exx_scatter_add_recip_op<complexd, base_device::DEVICE_CPU>;
    using mul_potential_cpu_op = hamilt::mul_potential_op<complexd, base_device::DEVICE_CPU>;
    using vec_mul_cpu_op = hamilt::vec_mul_vec_complex_op<complexd, base_device::DEVICE_CPU>;
};

TEST_F(TestModuleHamiltExxQState, conjugate_gather_and_scatter_cpu)
{
    const std::vector<complexd> in = {{1.0, 2.0}, {3.0, -4.0}, {-2.0, 0.5}, {5.0, 1.0}};
    std::vector<complexd> conjugated(in.size());
    conjugate_cpu_op()(in.data(), conjugated.data(), in.size());

    EXPECT_EQ(conjugated[0], complexd(1.0, -2.0));
    EXPECT_EQ(conjugated[2], complexd(-2.0, -0.5));

    const std::vector<int> gather_map = {2, -1, 0};
    std::vector<complexd> gathered(gather_map.size(), {9.0, 9.0});
    gather_cpu_op()(conjugated.data(), gathered.data(), gather_map.data(), gather_map.size());

    EXPECT_EQ(gathered[0], complexd(-2.0, -0.5));
    EXPECT_EQ(gathered[1], complexd(0.0, 0.0));
    EXPECT_EQ(gathered[2], complexd(1.0, -2.0));

    const std::vector<int> scatter_map = {1, -1, 3};
    std::vector<complexd> out(4, {1.0, 1.0});
    scatter_cpu_op()(gathered.data(), out.data(), scatter_map.data(), scatter_map.size(), complexd(2.0, 0.0));

    EXPECT_EQ(out[0], complexd(1.0, 1.0));
    EXPECT_EQ(out[1], complexd(-3.0, 0.0));
    EXPECT_EQ(out[2], complexd(1.0, 1.0));
    EXPECT_EQ(out[3], complexd(3.0, -3.0));
}

TEST_F(TestModuleHamiltExxQState, gather_and_scatter_cpu_support_float_maps)
{
    using complexf = std::complex<float>;
    using gather_float_cpu_op = hamilt::exx_gather_recip_op<complexf, base_device::DEVICE_CPU>;
    using scatter_float_cpu_op = hamilt::exx_scatter_add_recip_op<complexf, base_device::DEVICE_CPU>;

    const std::vector<complexf> in = {{1.0f, -1.0f}, {2.0f, 0.5f}, {-3.0f, 2.0f}};
    const std::vector<int> gather_map = {1, -1, 2, 0};
    std::vector<complexf> gathered(gather_map.size(), {9.0f, 9.0f});
    gather_float_cpu_op()(in.data(), gathered.data(), gather_map.data(), gather_map.size());

    EXPECT_EQ(gathered[0], complexf(2.0f, 0.5f));
    EXPECT_EQ(gathered[1], complexf(0.0f, 0.0f));
    EXPECT_EQ(gathered[2], complexf(-3.0f, 2.0f));
    EXPECT_EQ(gathered[3], complexf(1.0f, -1.0f));

    const std::vector<int> scatter_map = {2, -1, 0, 1};
    std::vector<complexf> out(3, {0.25f, -0.25f});
    scatter_float_cpu_op()(gathered.data(), out.data(), scatter_map.data(), scatter_map.size(), complexf(-0.5f, 0.25f));

    EXPECT_EQ(out[0], complexf(1.25f, -2.0f));
    EXPECT_EQ(out[1], complexf(0.0f, 0.5f));
    EXPECT_EQ(out[2], complexf(-0.875f, 0.0f));
}

TEST_F(TestModuleHamiltExxQState, batch_elementwise_cpu_matches_scalar_calls)
{
    const int npw = 3;
    const int batch_size = 2;
    const std::vector<double> pot = {2.0, -1.0, 0.5};
    std::vector<complexd> density = {{1.0, 1.0}, {2.0, -1.0}, {-4.0, 2.0},
                                     {0.5, 0.0}, {-3.0, 1.0}, {6.0, -2.0}};
    std::vector<complexd> expected = density;
    for (int ib = 0; ib < batch_size; ++ib)
    {
        mul_potential_cpu_op()(pot.data(), expected.data() + ib * npw, npw, 0, 0, 0);
    }

    mul_potential_cpu_op().operator_batch(pot.data(), density.data(), npw, batch_size);

    EXPECT_EQ(density, expected);

    const std::vector<complexd> left = {{1.0, 2.0}, {3.0, 0.0}, {-1.0, 1.0},
                                        {0.0, 2.0}, {2.0, -1.0}, {4.0, 0.5}};
    const std::vector<complexd> right = {{2.0, 0.0}, {0.5, -1.0}, {3.0, 2.0},
                                         {1.0, -1.0}, {-2.0, 0.0}, {0.0, 2.0}};
    std::vector<complexd> batched(left.size());
    std::vector<complexd> scalar(left.size());
    for (int ib = 0; ib < batch_size; ++ib)
    {
        vec_mul_cpu_op()(left.data() + ib * npw, right.data() + ib * npw, scalar.data() + ib * npw, npw);
    }

    vec_mul_cpu_op().operator_batch(left.data(), right.data(), batched.data(), npw, batch_size);

    EXPECT_EQ(batched, scalar);
}

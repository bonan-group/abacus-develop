#include "source_base/module_device/device.h"
#include "source_base/module_device/memory_op.h"
#include "source_estate/module_charge/kernels/charge_mixing_op.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

namespace
{
using Complex = std::complex<double>;

template <typename T>
T* copy_to_gpu(const std::vector<T>& host, const char* label)
{
    T* device = nullptr;
    base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(device, host.size(), label);
    base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
        device, host.data(), host.size());
    return device;
}

template <typename T>
std::vector<T> copy_to_host(const T* device, const int size)
{
    std::vector<T> host(size);
    base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        host.data(), device, size);
    return host;
}

double spin_hartree_reference(const Complex* lhs,
                              const Complex* rhs,
                              const std::vector<double>& gg,
                              const int npw,
                              const int nspin,
                              const int ig_gge0,
                              const bool gamma_only,
                              const bool include_magnetism,
                              const double charge_fac,
                              const double mag_fac)
{
    double result = 0.0;
    for (int ig = 0; ig < npw; ++ig)
    {
        if (ig != ig_gge0)
        {
            double charge = std::real(std::conj(lhs[ig]) * rhs[ig]) / gg[ig] * charge_fac;
            if (nspin == 2 && gamma_only)
            {
                charge *= 2.0;
            }
            result += charge;
        }
    }

    if (!include_magnetism)
    {
        return result;
    }
    if (nspin == 2)
    {
        result += std::real(std::conj(lhs[npw]) * rhs[npw]) * mag_fac;
        for (int ig = 0; ig < npw; ++ig)
        {
            double magnetic = std::real(std::conj(lhs[npw + ig]) * rhs[npw + ig]) * mag_fac;
            if (gamma_only)
            {
                magnetic *= 2.0;
            }
            result += magnetic;
        }
    }
    else if (nspin == 4)
    {
        for (int ig = 0; ig < npw; ++ig)
        {
            double magnetic = 0.0;
            for (int is = 1; is < nspin; ++is)
            {
                magnetic += std::real(std::conj(lhs[is * npw + ig]) * rhs[is * npw + ig]);
            }
            if (ig == ig_gge0)
            {
                if (ig_gge0 > 0)
                {
                    result += magnetic * mag_fac;
                }
            }
            else
            {
                result += magnetic * mag_fac * (gamma_only ? 2.0 : 1.0);
            }
        }
    }
    return result;
}
} // namespace

TEST(ChargeMixingGpuKernelsTest, SpinHartreeBatchMatchesCpuReference)
{
    if (!base_device::information::probe_gpu_availability())
    {
        GTEST_SKIP() << "No GPU device is available.";
    }

    struct Case
    {
        int nspin;
        int ig_gge0;
        bool gamma_only;
        bool include_magnetism;
    };
    const std::vector<Case> cases = {
        {2, 0, false, true}, {2, 0, true, true}, {4, 2, true, true}, {4, 2, false, false}};
    const int npw = 4;
    const int nlhs = 2;
    const int nrhs = 2;
    const double charge_fac = 3.25;
    const double mag_fac = 0.75;

    for (const Case& test_case : cases)
    {
        SCOPED_TRACE(::testing::Message() << "nspin=" << test_case.nspin
                                          << ", ig_gge0=" << test_case.ig_gge0
                                          << ", gamma_only=" << test_case.gamma_only
                                          << ", include_magnetism=" << test_case.include_magnetism);
        std::vector<double> gg = {1.0, 1.5, 2.5, 4.0};
        gg[test_case.ig_gge0] = 0.0;
        std::vector<Complex> lhs(nlhs * test_case.nspin * npw);
        std::vector<Complex> rhs(nrhs * test_case.nspin * npw);
        for (std::size_t i = 0; i < lhs.size(); ++i)
        {
            lhs[i] = Complex(0.2 * (i + 1), -0.05 * (i + 2));
        }
        for (std::size_t i = 0; i < rhs.size(); ++i)
        {
            rhs[i] = Complex(-0.1 * (i + 3), 0.08 * (i + 1));
        }

        Complex* lhs_d = copy_to_gpu(lhs, "charge_mixing_test_hartree_lhs");
        Complex* rhs_d = copy_to_gpu(rhs, "charge_mixing_test_hartree_rhs");
        double* gg_d = copy_to_gpu(gg, "charge_mixing_test_hartree_gg");
        double* result_d = nullptr;
        double* workspace_d = nullptr;
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            result_d, nlhs * nrhs, "charge_mixing_test_hartree_result");
        base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(
            workspace_d, nlhs * nrhs * npw, "charge_mixing_test_hartree_workspace");

        elecstate::inner_product_recip_hartree_spin_batch_op<double, base_device::DEVICE_GPU>()(
            nullptr,
            lhs_d,
            rhs_d,
            gg_d,
            npw,
            test_case.nspin,
            nlhs,
            nrhs,
            test_case.ig_gge0,
            test_case.gamma_only,
            test_case.include_magnetism,
            charge_fac,
            mag_fac,
            result_d,
            workspace_d);
        const std::vector<double> result = copy_to_host(result_d, nlhs * nrhs);
        for (int ilhs = 0; ilhs < nlhs; ++ilhs)
        {
            for (int irhs = 0; irhs < nrhs; ++irhs)
            {
                const double expected = spin_hartree_reference(
                    lhs.data() + ilhs * test_case.nspin * npw,
                    rhs.data() + irhs * test_case.nspin * npw,
                    gg,
                    npw,
                    test_case.nspin,
                    test_case.ig_gge0,
                    test_case.gamma_only,
                    test_case.include_magnetism,
                    charge_fac,
                    mag_fac);
                EXPECT_NEAR(result[ilhs * nrhs + irhs], expected, 1e-11);
            }
        }

        base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(lhs_d);
        base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(rhs_d);
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(gg_d);
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(result_d);
        base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(workspace_d);
    }
}

TEST(ChargeMixingGpuKernelsTest, PackAndUnpackNspin2)
{
    if (!base_device::information::probe_gpu_availability())
    {
        GTEST_SKIP() << "No GPU device is available.";
    }

    const int npw = 3;
    const int nspin = 2;
    const std::vector<Complex> spin_data = {
        Complex(1.0, 0.5), Complex(2.0, -0.5), Complex(3.0, 1.0),
        Complex(0.25, -0.5), Complex(0.5, 0.25), Complex(1.0, -1.0)};
    Complex* spin_d = copy_to_gpu(spin_data, "charge_mixing_test_spin");
    Complex* packed_d = nullptr;
    Complex* unpacked_d = nullptr;
    base_device::memory::resize_memory_op<Complex, base_device::DEVICE_GPU>()(
        packed_d, nspin * npw, "charge_mixing_test_packed");
    base_device::memory::resize_memory_op<Complex, base_device::DEVICE_GPU>()(
        unpacked_d, nspin * npw, "charge_mixing_test_unpacked");

    elecstate::pack_spin_recip_op<double, base_device::DEVICE_GPU>()(
        nullptr, packed_d, spin_d, npw, nspin);
    const std::vector<Complex> packed = copy_to_host(packed_d, nspin * npw);
    for (int ig = 0; ig < npw; ++ig)
    {
        EXPECT_EQ(packed[ig], spin_data[ig] + spin_data[npw + ig]);
        EXPECT_EQ(packed[npw + ig], spin_data[ig] - spin_data[npw + ig]);
    }

    elecstate::unpack_spin_recip_op<double, base_device::DEVICE_GPU>()(
        nullptr, unpacked_d, packed_d, npw, nspin);
    EXPECT_EQ(copy_to_host(unpacked_d, nspin * npw), spin_data);

    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(spin_d);
    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(packed_d);
    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(unpacked_d);
}

class DoubleGridGpuKernelsTest : public ::testing::TestWithParam<int>
{
};

TEST_P(DoubleGridGpuKernelsTest, SplitAndCombinePreserveSpinOrdering)
{
    if (!base_device::information::probe_gpu_availability())
    {
        GTEST_SKIP() << "No GPU device is available.";
    }

    const int nspin = GetParam();
    const int smooth_npw = 3;
    const int dense_npw = 5;
    const int hf_npw = dense_npw - smooth_npw;
    std::vector<Complex> dense(nspin * dense_npw);
    for (int is = 0; is < nspin; ++is)
    {
        for (int ig = 0; ig < dense_npw; ++ig)
        {
            dense[is * dense_npw + ig] = Complex(100.0 * is + ig, -10.0 * is - ig);
        }
    }

    Complex* dense_d = copy_to_gpu(dense, "charge_mixing_test_dense");
    Complex* smooth_d = nullptr;
    Complex* high_frequency_d = nullptr;
    base_device::memory::resize_memory_op<Complex, base_device::DEVICE_GPU>()(
        smooth_d, nspin * smooth_npw, "charge_mixing_test_smooth");
    base_device::memory::resize_memory_op<Complex, base_device::DEVICE_GPU>()(
        high_frequency_d, nspin * hf_npw, "charge_mixing_test_high_frequency");

    elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
        nullptr, smooth_d, high_frequency_d, dense_d, smooth_npw, dense_npw, nspin);
    const std::vector<Complex> smooth = copy_to_host(smooth_d, nspin * smooth_npw);
    const std::vector<Complex> high_frequency = copy_to_host(high_frequency_d, nspin * hf_npw);
    for (int is = 0; is < nspin; ++is)
    {
        for (int ig = 0; ig < smooth_npw; ++ig)
        {
            EXPECT_EQ(smooth[is * smooth_npw + ig], dense[is * dense_npw + ig]);
        }
        for (int ig = 0; ig < hf_npw; ++ig)
        {
            EXPECT_EQ(high_frequency[is * hf_npw + ig], dense[is * dense_npw + smooth_npw + ig]);
        }
    }

    base_device::memory::set_memory_op<Complex, base_device::DEVICE_GPU>()(
        dense_d, 0, nspin * dense_npw);
    elecstate::combine_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
        nullptr, dense_d, smooth_d, high_frequency_d, smooth_npw, dense_npw, nspin);
    EXPECT_EQ(copy_to_host(dense_d, nspin * dense_npw), dense);

    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(dense_d);
    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(smooth_d);
    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(high_frequency_d);
}

INSTANTIATE_TEST_SUITE_P(Spinful,
                         DoubleGridGpuKernelsTest,
                         ::testing::Values(2, 4));

TEST(ChargeMixingGpuKernelsTest, EqualSizeDoubleGridAllowsNullHighFrequencyBuffer)
{
    if (!base_device::information::probe_gpu_availability())
    {
        GTEST_SKIP() << "No GPU device is available.";
    }

    const int npw = 3;
    const int nspin = 2;
    const std::vector<Complex> dense = {
        {1.0, 0.0}, {2.0, 0.5}, {3.0, 1.0}, {4.0, -0.5}, {5.0, -1.0}, {6.0, -1.5}};
    Complex* dense_d = copy_to_gpu(dense, "charge_mixing_test_equal_dense");
    Complex* smooth_d = nullptr;
    base_device::memory::resize_memory_op<Complex, base_device::DEVICE_GPU>()(
        smooth_d, nspin * npw, "charge_mixing_test_equal_smooth");

    elecstate::split_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
        nullptr, smooth_d, nullptr, dense_d, npw, npw, nspin);
    EXPECT_EQ(copy_to_host(smooth_d, nspin * npw), dense);

    base_device::memory::set_memory_op<Complex, base_device::DEVICE_GPU>()(dense_d, 0, nspin * npw);
    elecstate::combine_double_grid_recip_op<double, base_device::DEVICE_GPU>()(
        nullptr, dense_d, smooth_d, nullptr, npw, npw, nspin);
    EXPECT_EQ(copy_to_host(dense_d, nspin * npw), dense);

    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(dense_d);
    base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>()(smooth_d);
}

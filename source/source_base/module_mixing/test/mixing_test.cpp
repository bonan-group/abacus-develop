#ifdef _OPENMP
#include <omp.h>
#endif

#include "../broyden_mixing.h"
#include "../plain_mixing.h"
#include "../pulay_mixing.h"
#if __UT_USE_CUDA
#include "../broyden_mixing_gpu.h"
#include "../mixing_data_gpu.h"
#include "../pulay_mixing_gpu.h"
#include "source_base/module_device/device.h"
#include "source_base/module_device/memory_op.h"
#endif
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#define DOUBLETHRESHOLD 1e-8
double ext_inner_product_mock(double* x1, double* x2)
{
    return 0.0;
}
class Mixing_Test : public testing::Test
{
  protected:
    Mixing_Test()
    {
    }
    ~Mixing_Test()
    {
        delete this->mixing;
    }
    const double mixing_beta = 0.6;
    const int mixing_ndim = 3;
    Base_Mixing::Mixing_Data xdata;
    Base_Mixing::Mixing* mixing = nullptr;
    double thr = 1e-8;
    int niter = 0;
    int maxiter = 10;
    std::vector<double> xd_ref = {0.0, 0.0, 0.0};
    std::vector<std::complex<double>> xc_ref = {
        {0.0, 1.0},
        {1.0, 0.0},
        0.0
    };
    void init_method(std::string method)
    {
        if (method == "broyden")
        {
            this->mixing = new Base_Mixing::Broyden_Mixing(this->mixing_ndim, this->mixing_beta);
        }
        else if (method == "pulay")
        {
            this->mixing = new Base_Mixing::Pulay_Mixing(this->mixing_ndim, this->mixing_beta);
        }
        else if (method == "plain")
        {
            this->mixing = new Base_Mixing::Plain_Mixing(this->mixing_beta);
        }
    }

    void clear()
    {
        delete this->mixing;
        this->mixing = nullptr;
    }

    /**
     * @brief sover linear equation:
     *        [ 8 -3  2 ][x1]   [20]       [3]
     *        [ 4 11 -1 ][x2] = [33]   x = [2]
     *        [ 6  3 12 ][x3]   [36]       [1]
     *
     *         [x1]   [ 3/8  -2/8   20/8 ][x1]
     *         [x2] = [-4/11  1/11 -33/11][x2]
     *         [x3]   [-6/12 -3/12  36/12][x3]
     */
    template <typename FPTYPE>
    void solve_linear_eq(FPTYPE* x_in, FPTYPE* x_out, bool diff_beta = false)
    {
        this->mixing->init_mixing_data(xdata, 3, sizeof(FPTYPE));
        std::vector<FPTYPE> delta_x(3);

        auto screen = std::bind(&Mixing_Test::Kerker_mock<FPTYPE>, this, std::placeholders::_1);
        auto inner_product
            = std::bind(static_cast<double (Mixing_Test::*)(FPTYPE*, FPTYPE*)>(&Mixing_Test::inner_product_mock),
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2);

        double residual = 10.;
        this->niter = 0;
        while (niter < maxiter)
        {
            x_out[0] = (3. * x_in[1] - 2. * x_in[2] + 20.) / 8.;
            x_out[1] = (-4. * x_out[0] + 1. * x_in[2] + 33.) / 11.;
            x_out[2] = (-6. * x_out[0] - 3. * x_out[1] + 36.) / 12.;

            niter++;

            for (int i = 0; i < 3; ++i)
            {
                delta_x[i] = x_out[i] - x_in[i];
            }
            residual = this->inner_product_mock(delta_x.data(), delta_x.data());
            if (residual <= thr)
            {
                break;
            }
            if (diff_beta)
            {
                this->mixing->push_data(
                    this->xdata,
                    x_in,
                    x_out,
                    screen,
                    // mixing can use different mixing_beta for one vector
                    [](FPTYPE* out, const FPTYPE* in, const FPTYPE* sres) {
                        out[0] = in[0] + 0.5 * sres[0];
                        out[1] = in[1] + 0.6 * sres[1];
                        out[2] = in[2] + 0.5 * sres[2];
                    },
                    true);
            }
            else
            {
                this->mixing->push_data(this->xdata, x_in, x_out, screen, true);
            }

            this->mixing->cal_coef(this->xdata, inner_product);

            this->mixing->mix_data(this->xdata, x_in);
        }
    }

    template <typename FPTYPE>
    void Kerker_mock(FPTYPE* drho)
    {
    }

    double inner_product_mock(double* x1, double* x2)
    {
        double xnorm = 0.0;
        for (int ir = 0; ir < 3; ++ir)
        {
            xnorm += x1[ir] * x2[ir];
        }
        return xnorm;
    }
    double inner_product_mock(std::complex<double>* x1, std::complex<double>* x2)
    {
        double xnorm = 0.0;
        for (int ir = 0; ir < 3; ++ir)
        {
            xnorm += x1[ir].real() * x2[ir].real() + x1[ir].imag() * x2[ir].imag();
        }
        return xnorm;
    }
};

TEST_F(Mixing_Test, BroydenSolveLinearEq)
{
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    init_method("broyden");
    std::vector<double> x_in = xd_ref;
    std::vector<double> x_out(3);
    solve_linear_eq<double>(x_in.data(), x_out.data(), true);
    EXPECT_NEAR(x_out[0], 3.0, DOUBLETHRESHOLD);
    EXPECT_NEAR(x_out[1], 2.0, DOUBLETHRESHOLD);
    EXPECT_NEAR(x_out[2], 1.0, DOUBLETHRESHOLD);
    ASSERT_EQ(niter, 5);

    this->mixing->reset();
    xdata.reset();

    std::vector<std::complex<double>> xc_in = xc_ref;
    std::vector<std::complex<double>> xc_out(3);
    solve_linear_eq<std::complex<double>>(xc_in.data(), xc_out.data(), true);
    EXPECT_NEAR(xc_out[0].real(), 3.0, DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_out[1].real(), 2.0, DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_out[2].real(), 1.0, DOUBLETHRESHOLD);
    ASSERT_EQ(niter, 5);
    std::string output;
    Base_Mixing::Mixing_Data testdata;
    this->mixing->init_mixing_data(testdata, 3, sizeof(double));

    testing::internal::CaptureStdout();
    EXPECT_EXIT(this->mixing->push_data(testdata, x_in.data(), x_out.data(), nullptr, true),
                ::testing::ExitedWithCode(1),
                "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(
        output,
        testing::HasSubstr("One Broyden_Mixing object can only bind one Mixing_Data object to calculate coefficients"));

    testing::internal::CaptureStdout();
    EXPECT_EXIT(this->mixing->cal_coef(testdata, ext_inner_product_mock), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(
        output,
        testing::HasSubstr("One Broyden_Mixing object can only bind one Mixing_Data object to calculate coefficients"));

    clear();
}

TEST_F(Mixing_Test, PulaySolveLinearEq)
{
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    init_method("pulay");
    std::vector<double> x_in = xd_ref;
    std::vector<double> x_out(3);
    solve_linear_eq<double>(x_in.data(), x_out.data());
    EXPECT_NEAR(x_out[0], 2.9999959638248037, DOUBLETHRESHOLD);
    EXPECT_NEAR(x_out[1], 2.0000002552633349, DOUBLETHRESHOLD);
    EXPECT_NEAR(x_out[2], 1.0000019542717642, DOUBLETHRESHOLD);
    ASSERT_EQ(niter, 6);

    this->mixing->reset();
    xdata.reset();

    std::vector<std::complex<double>> xc_in = xc_ref;
    std::vector<std::complex<double>> xc_out(3);
    solve_linear_eq<std::complex<double>>(xc_in.data(), xc_out.data());
    EXPECT_NEAR(xc_out[0].real(), 3.0000063220482565, DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_out[1].real(), 1.9999939191147462, DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_out[2].real(), 0.99999835919718549, DOUBLETHRESHOLD);
    ASSERT_EQ(niter, 6);

    std::string output;
    Base_Mixing::Mixing_Data testdata;
    this->mixing->init_mixing_data(testdata, 3, sizeof(double));

    testing::internal::CaptureStdout();
    EXPECT_EXIT(this->mixing->push_data(testdata, x_in.data(), x_out.data(), nullptr, true),
                ::testing::ExitedWithCode(1),
                "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(
        output,
        testing::HasSubstr("One Pulay_Mixing object can only bind one Mixing_Data object to calculate coefficients"));

    testing::internal::CaptureStdout();
    EXPECT_EXIT(this->mixing->cal_coef(testdata, ext_inner_product_mock), ::testing::ExitedWithCode(1), "");
    output = testing::internal::GetCapturedStdout();
    EXPECT_THAT(
        output,
        testing::HasSubstr("One Pulay_Mixing object can only bind one Mixing_Data object to calculate coefficients"));

    clear();
}

TEST_F(Mixing_Test, PlainSolveLinearEq)
{
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    init_method("plain");
    std::vector<double> x_in = xd_ref;
    std::vector<double> x_out(3);
    solve_linear_eq<double>(x_in.data(), x_out.data());
    EXPECT_NEAR(x_out[0], 2.9999613068687698, DOUBLETHRESHOLD);
    EXPECT_NEAR(x_out[1], 2.0000472873362103, DOUBLETHRESHOLD);
    EXPECT_NEAR(x_out[2], 1.0000075247315625, DOUBLETHRESHOLD);
    ASSERT_EQ(niter, 10);

    this->mixing->reset();
    xdata.reset();

    std::vector<std::complex<double>> xc_in = xc_ref;
    std::vector<std::complex<double>> xc_out(3);
    solve_linear_eq<std::complex<double>>(xc_in.data(), xc_out.data());
    EXPECT_NEAR(xc_out[0].real(), 2.9999418982632711, DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_out[1].real(), 2.0000317031363761, DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_out[2].real(), 1.0000211250842703, DOUBLETHRESHOLD);
    ASSERT_EQ(niter, 10);

    // test mix_data of plain_mixing
    std::vector<double> x_tmp(3);
    this->mixing->push_data(this->xdata, x_in.data(), x_out.data(), nullptr, true);
    this->mixing->mix_data(this->xdata, x_tmp.data());
    Base_Mixing::Plain_Mixing plain_mix(mixing_beta);
    plain_mix.plain_mix(x_in.data(), x_in.data(), x_out.data(), 3, [](double* x) {});
    EXPECT_NEAR(x_tmp[0], x_in[0], DOUBLETHRESHOLD);
    EXPECT_NEAR(x_tmp[1], x_in[1], DOUBLETHRESHOLD);
    EXPECT_NEAR(x_tmp[2], x_in[2], DOUBLETHRESHOLD);
    
    std::vector<std::complex<double>> xc_tmp(3);
    this->mixing->push_data(this->xdata, xc_in.data(), xc_out.data(), nullptr, true);
    this->mixing->mix_data(this->xdata, xc_tmp.data());
    plain_mix.plain_mix(xc_in.data(), xc_in.data(), xc_out.data(), 3, nullptr);
    EXPECT_NEAR(xc_tmp[0].real(), xc_in[0].real(), DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_tmp[1].real(), xc_in[1].real(), DOUBLETHRESHOLD);
    EXPECT_NEAR(xc_tmp[2].real(), xc_in[2].real(), DOUBLETHRESHOLD);
    
    this->mixing->reset();

    clear();
}

TEST_F(Mixing_Test, OtherCover)
{
    this->mixing = new Base_Mixing::Broyden_Mixing(2, 0.7);
    Base_Mixing::Mixing_Data nodata;
    this->mixing->init_mixing_data(nodata, 0, sizeof(double));
    this->mixing->push_data(nodata, (double*)nullptr, (double*)nullptr, nullptr, false);
    this->mixing->push_data(nodata, (double*)nullptr, (double*)nullptr, nullptr, false);
    this->mixing->mix_data(nodata, (double*)nullptr);
    this->mixing->mix_data(nodata, (std::complex<double>*)nullptr);
    EXPECT_EQ(nodata.length, 0);

    clear();
}

#if __UT_USE_CUDA
namespace
{
template <typename FPTYPE>
double gpu_inner_product(const FPTYPE* a, const FPTYPE* b, const int length, FPTYPE* workspace)
{
    return mixing::inner_product_op<FPTYPE, base_device::DEVICE_GPU>()(
        nullptr, a, b, length, workspace);
}

template <>
double gpu_inner_product<std::complex<double>>(const std::complex<double>* a,
                                               const std::complex<double>* b,
                                               const int length,
                                               std::complex<double>* workspace)
{
    return mixing::inner_product_op<std::complex<double>, base_device::DEVICE_GPU>()(
        nullptr, a, b, length, workspace).real();
}

template <typename FPTYPE>
double cpu_inner_product(const FPTYPE* a, const FPTYPE* b, const int length)
{
    double result = 0.0;
    for (int i = 0; i < length; ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

template <>
double cpu_inner_product<std::complex<double>>(const std::complex<double>* a,
                                               const std::complex<double>* b,
                                               const int length)
{
    double result = 0.0;
    for (int i = 0; i < length; ++i)
    {
        result += (std::conj(a[i]) * b[i]).real();
    }
    return result;
}

template <typename FPTYPE>
void expect_near_value(const FPTYPE& actual, const FPTYPE& expected, const double tol)
{
    EXPECT_NEAR(actual, expected, tol);
}

template <>
void expect_near_value<std::complex<double>>(const std::complex<double>& actual,
                                             const std::complex<double>& expected,
                                             const double tol)
{
    EXPECT_NEAR(actual.real(), expected.real(), tol);
    EXPECT_NEAR(actual.imag(), expected.imag(), tol);
}

template <typename MixerCpu, typename MixerGpu, typename FPTYPE>
void compare_cpu_gpu_mixing_history()
{
    constexpr int length = 4;
    constexpr int mixing_ndim = 3;
    constexpr double mixing_beta = 0.6;
    const std::vector<std::vector<FPTYPE>> inputs = {
        {FPTYPE(0.1), FPTYPE(-0.2), FPTYPE(0.3), FPTYPE(0.7)},
        {FPTYPE(0.4), FPTYPE(0.1), FPTYPE(-0.5), FPTYPE(0.2)},
        {FPTYPE(-0.3), FPTYPE(0.6), FPTYPE(0.8), FPTYPE(-0.4)},
        {FPTYPE(0.9), FPTYPE(-0.7), FPTYPE(0.2), FPTYPE(0.5)},
        {FPTYPE(-0.6), FPTYPE(0.3), FPTYPE(-0.1), FPTYPE(0.4)},
    };
    const std::vector<std::vector<FPTYPE>> outputs = {
        {FPTYPE(0.6), FPTYPE(0.0), FPTYPE(0.1), FPTYPE(1.1)},
        {FPTYPE(0.2), FPTYPE(0.8), FPTYPE(-0.1), FPTYPE(-0.3)},
        {FPTYPE(0.5), FPTYPE(0.2), FPTYPE(1.0), FPTYPE(0.1)},
        {FPTYPE(1.2), FPTYPE(-0.4), FPTYPE(-0.6), FPTYPE(0.9)},
        {FPTYPE(-0.2), FPTYPE(0.9), FPTYPE(0.4), FPTYPE(-0.8)},
    };

    MixerCpu cpu_mixer(mixing_ndim, mixing_beta);
    Base_Mixing::Mixing_Data cpu_data;
    cpu_mixer.init_mixing_data(cpu_data, length, sizeof(FPTYPE));

    MixerGpu gpu_mixer(mixing_ndim, static_cast<FPTYPE>(mixing_beta));
    Base_Mixing::Mixing_Data_GPU<FPTYPE> gpu_data(gpu_mixer.get_data_ndim(), length);
    gpu_mixer.init(length);

    FPTYPE* input_d = nullptr;
    FPTYPE* output_d = nullptr;
    FPTYPE* mixed_d = nullptr;
    const int max_blocks = (length + 255) / 256;
    FPTYPE* workspace_d = nullptr;
    base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(input_d, length, "mixing_test_in");
    base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(output_d, length, "mixing_test_out");
    base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(mixed_d, length, "mixing_test_mix");
    base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(
        workspace_d, 2 * max_blocks, "mixing_test_workspace");

    std::vector<FPTYPE> cpu_mixed(length);
    std::vector<FPTYPE> gpu_mixed(length);
    auto cpu_mix = [](FPTYPE* out, const FPTYPE* in, const FPTYPE* residual) {
        for (int i = 0; i < length; ++i)
        {
            out[i] = in[i] + static_cast<FPTYPE>(mixing_beta) * residual[i];
        }
    };
    for (std::size_t step = 0; step < inputs.size(); ++step)
    {
        cpu_mixer.push_data(cpu_data, inputs[step].data(), outputs[step].data(), nullptr, cpu_mix, true);
        cpu_mixer.cal_coef(cpu_data,
                           [](FPTYPE* a, FPTYPE* b) { return cpu_inner_product(a, b, length); });
        cpu_mixer.mix_data(cpu_data, cpu_mixed.data());

        base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            input_d, inputs[step].data(), length);
        base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            output_d, outputs[step].data(), length);
        gpu_mixer.push_data(gpu_data, input_d, output_d, nullptr, true);
        gpu_mixer.cal_coef(gpu_data,
                           [workspace_d](const FPTYPE* a, const FPTYPE* b) {
                               return gpu_inner_product(a, b, length, workspace_d);
                           });
        gpu_mixer.mix_data(gpu_data, mixed_d);
        base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            gpu_mixed.data(), mixed_d, length);

        for (int i = 0; i < length; ++i)
        {
            expect_near_value(gpu_mixed[i], cpu_mixed[i], 1e-8);
        }
    }

    base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(input_d);
    base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(output_d);
    base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(mixed_d);
    base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(workspace_d);
}
} // namespace

TEST(MixingGpuTest, PulayMatchesCpuAcrossHistoryWrap)
{
    if (!base_device::information::probe_gpu_availability())
    {
        GTEST_SKIP() << "No GPU device is available for CUDA mixing parity tests.";
    }
    compare_cpu_gpu_mixing_history<Base_Mixing::Pulay_Mixing,
                                   Base_Mixing::Pulay_Mixing_GPU<double>,
                                   double>();
    compare_cpu_gpu_mixing_history<Base_Mixing::Pulay_Mixing,
                                   Base_Mixing::Pulay_Mixing_GPU<std::complex<double>>,
                                   std::complex<double>>();
}

TEST(MixingGpuTest, BroydenMatchesCpuAcrossHistoryWrap)
{
    if (!base_device::information::probe_gpu_availability())
    {
        GTEST_SKIP() << "No GPU device is available for CUDA mixing parity tests.";
    }
    compare_cpu_gpu_mixing_history<Base_Mixing::Broyden_Mixing,
                                   Base_Mixing::Broyden_Mixing_GPU<double>,
                                   double>();
    compare_cpu_gpu_mixing_history<Base_Mixing::Broyden_Mixing,
                                   Base_Mixing::Broyden_Mixing_GPU<std::complex<double>>,
                                   std::complex<double>>();
}
#endif // __UT_USE_CUDA

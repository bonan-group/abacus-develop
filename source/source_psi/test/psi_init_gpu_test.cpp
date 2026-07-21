#include "source_base/module_device/memory_op.h"
#include "source_psi/kernels/psi_init_op.h"
#include "source_psi/psi_init_atomic_device.h"
#include "source_psi/psi_init_policy.h"

#include <gtest/gtest.h>

#include <complex>
#include <cmath>
#include <type_traits>
#include <vector>

namespace
{

using Complex = std::complex<float>;

template <typename T>
class DeviceBuffer
{
  public:
    explicit DeviceBuffer(const std::size_t size) : size_(size)
    {
        base_device::memory::resize_memory_op<T, base_device::DEVICE_GPU>()(data_, size_);
    }

    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size())
    {
        copy_from(host);
    }

    ~DeviceBuffer()
    {
        base_device::memory::delete_memory_op<T, base_device::DEVICE_GPU>()(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data()
    {
        return data_;
    }

    const T* data() const
    {
        return data_;
    }

    void copy_from(const std::vector<T>& host)
    {
        ASSERT_EQ(host.size(), size_);
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(
            data_, host.data(), size_);
    }

    std::vector<T> copy_to_host() const
    {
        std::vector<T> host(size_);
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
            host.data(), data_, size_);
        return host;
    }

  private:
    T* data_ = nullptr;
    std::size_t size_;
};

const Complex seeded_sequence_golden_0(0.31092915f, -0.85787631f);
const Complex seeded_sequence_golden_1(-0.30231641f, -0.02610150f);

std::vector<Complex> run_random_init(const int seed)
{
    constexpr int nbands = 2;
    constexpr int npwk = 3;
    constexpr int npwk_max = 5;
    constexpr int npol = 2;
    constexpr int total = nbands * npwk_max * npol;

    std::vector<float> h_gk2 = {0.0f, 1.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.0f};
    DeviceBuffer<float> d_gk2(h_gk2);
    DeviceBuffer<Complex> d_psi(total);

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi.data(), nbands, npwk, npwk_max, npol, 1, 1, seed, d_gk2.data(), nullptr, nullptr, 0, 0);

    return d_psi.copy_to_host();
}

std::vector<Complex> run_stick_mapped_random_init()
{
    constexpr int nbands = 1;
    constexpr int npwk = 3;
    constexpr int npwk_max = 5;
    constexpr int npol = 1;
    constexpr int total = nbands * npwk_max * npol;

    std::vector<float> h_gk2 = {0.0f, 1.0f, 1.0f, 0.0f, 0.0f};
    std::vector<int> h_igl2isz = {0, 7, 7, 0, 0};
    std::vector<int> h_is2fftixy = {0, 3, 7, 11};
    DeviceBuffer<float> d_gk2(h_gk2);
    DeviceBuffer<int> d_igl2isz(h_igl2isz);
    DeviceBuffer<int> d_is2fftixy(h_is2fftixy);
    DeviceBuffer<Complex> d_psi(total);

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr,
        d_psi.data(),
        nbands,
        npwk,
        npwk_max,
        npol,
        0,
        0,
        17,
        d_gk2.data(),
        d_igl2isz.data(),
        d_is2fftixy.data(),
        16,
        4);

    return d_psi.copy_to_host();
}

std::vector<Complex> run_seeded_sequence_random_init()
{
    constexpr int nbands = 2;
    constexpr int npwk = 1;
    constexpr int npwk_max = 1;
    constexpr int npol = 1;
    constexpr int total = nbands * npwk_max * npol;

    std::vector<float> h_gk2 = {0.0f};
    std::vector<int> h_igl2isz = {0};
    std::vector<int> h_is2fftixy = {0};
    DeviceBuffer<float> d_gk2(h_gk2);
    DeviceBuffer<int> d_igl2isz(h_igl2isz);
    DeviceBuffer<int> d_is2fftixy(h_is2fftixy);
    DeviceBuffer<Complex> d_psi(total);

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr,
        d_psi.data(),
        nbands,
        npwk,
        npwk_max,
        npol,
        0,
        2,
        17,
        d_gk2.data(),
        d_igl2isz.data(),
        d_is2fftixy.data(),
        1,
        1);

    return d_psi.copy_to_host();
}

std::vector<Complex> run_seeded_global_k_random_init(const int ik_tot)
{
    constexpr int nbands = 1;
    constexpr int npwk = 1;
    constexpr int npwk_max = 1;
    constexpr int npol = 1;
    constexpr int total = nbands * npwk_max * npol;

    std::vector<float> h_gk2 = {0.0f};
    std::vector<int> h_igl2isz = {0};
    std::vector<int> h_is2fftixy = {0};
    DeviceBuffer<float> d_gk2(h_gk2);
    DeviceBuffer<int> d_igl2isz(h_igl2isz);
    DeviceBuffer<int> d_is2fftixy(h_is2fftixy);
    DeviceBuffer<Complex> d_psi(total);

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr,
        d_psi.data(),
        nbands,
        npwk,
        npwk_max,
        npol,
        0,
        ik_tot,
        17,
        d_gk2.data(),
        d_igl2isz.data(),
        d_is2fftixy.data(),
        1,
        1);

    return d_psi.copy_to_host();
}

Complex run_decomposed_seeded_random(const int local_stick, const std::vector<int>& h_is2fftixy)
{
    constexpr int nz = 4;
    const std::vector<float> h_gk2 = {0.0f};
    const std::vector<int> h_igl2isz = {local_stick * nz + 2};
    DeviceBuffer<float> d_gk2(h_gk2);
    DeviceBuffer<int> d_igl2isz(h_igl2isz);
    DeviceBuffer<int> d_is2fftixy(h_is2fftixy);
    DeviceBuffer<Complex> d_psi(1);

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(nullptr,
                                                            d_psi.data(),
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            0,
                                                            0,
                                                            17,
                                                            d_gk2.data(),
                                                            d_igl2isz.data(),
                                                            d_is2fftixy.data(),
                                                            16,
                                                            nz);

    return d_psi.copy_to_host()[0];
}

} // namespace

TEST(PsiInitGpuRandom, SeededInitializationIsDeterministicAndZeroPads)
{
    const std::vector<Complex> first = run_random_init(17);
    const std::vector<Complex> second = run_random_init(17);
    ASSERT_EQ(first.size(), second.size());
    for (int i = 0; i < static_cast<int>(first.size()); ++i)
    {
        EXPECT_EQ(first[i], second[i]) << "index " << i;
    }

    constexpr int nbands = 2;
    constexpr int npwk = 3;
    constexpr int npwk_max = 5;
    constexpr int npol = 2;
    for (int ib = 0; ib < nbands; ++ib)
    {
        for (int ipol = 0; ipol < npol; ++ipol)
        {
            const int offset = ib * npwk_max * npol + ipol * npwk_max;
            for (int ig = 0; ig < npwk; ++ig)
            {
                EXPECT_NE(first[offset + ig], Complex(0.0f, 0.0f));
            }
            for (int ig = npwk; ig < npwk_max; ++ig)
            {
                EXPECT_EQ(first[offset + ig], Complex(0.0f, 0.0f));
            }
        }
    }
}

TEST(PsiInitGpuRandomPolicy, AllowsUnseededGpuRandomInitialization)
{
    EXPECT_TRUE(psi::gpu_resident_init_policy(true, "dav_subspace", true, "random", 1));
}

TEST(PsiInitGpuPolicy, AllowsCollinearAtomicInitializers)
{
    EXPECT_TRUE(psi::gpu_resident_init_policy(true, "dav_subspace", true, "atomic", 1));
    EXPECT_TRUE(psi::gpu_resident_init_policy(true, "dav_subspace", true, "atomic+random", 1));
}

TEST(PsiInitGpuPolicy, KeepsUnsupportedAtomicInitializersOnCpu)
{
    EXPECT_FALSE(psi::gpu_resident_init_policy(true, "bpcg", true, "atomic", 1));
    EXPECT_FALSE(psi::gpu_resident_init_policy(true, "dav_subspace", true, "atomic", 2));
    EXPECT_FALSE(psi::gpu_resident_init_policy(false, "dav_subspace", true, "atomic+random", 1));
}

TEST(PsiInitGpuOwnership, DeviceOwnerAndKernelViewAreNonowningValueInterfaces)
{
    EXPECT_FALSE(std::is_copy_constructible<psi::AtomicGpuInitializer<Complex>>::value);
    EXPECT_FALSE(std::is_copy_assignable<psi::AtomicGpuInitializer<Complex>>::value);
    EXPECT_TRUE(std::is_standard_layout<psi::AtomicInitTableView<float>>::value);
    EXPECT_TRUE(std::is_trivially_copyable<psi::AtomicInitTableView<float>>::value);
}

TEST(PsiInitGpuAtomic, BuildsCartesianGPlusKOnDevice)
{
    const std::vector<float> h_gcar = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f,
                                       4.0f, 5.0f, 6.0f,  7.0f,  8.0f,  9.0f};
    const std::vector<float> h_kvec = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, 0.25f};
    DeviceBuffer<float> d_gcar(h_gcar);
    DeviceBuffer<float> d_kvec(h_kvec);
    DeviceBuffer<float> d_gk(6);

    psi::build_gk_op<float, base_device::DEVICE_GPU>()(
        nullptr, d_gk.data(), d_gcar.data(), d_kvec.data(), 1, 2, 2);

    const std::vector<float> h_gk = d_gk.copy_to_host();
    EXPECT_EQ(h_gk, (std::vector<float>{3.0f, 7.0f, 6.25f, 6.0f, 10.0f, 9.25f}));
}

TEST(PsiInitGpuAtomic, AssemblesAtomicOrbitalsAndZeroPads)
{
    constexpr int natomwfc = 2;
    constexpr int npw = 2;
    constexpr int npwk_max = 3;
    constexpr int total_lm = 2;
    constexpr int nchi_max = 2;
    constexpr int nqx = 5;

    const std::vector<float> h_gk = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    const std::vector<float> h_ylm = {0.5f, 0.25f, 1.5f, -0.5f};
    const std::vector<Complex> h_sk = {Complex(1.0f, 1.0f), Complex(2.0f, -1.0f),
                                       Complex(-1.0f, 0.5f), Complex(0.5f, 2.0f)};
    const std::vector<float> h_table = {2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
    const std::vector<int> h_iw2iat = {0, 1};
    const std::vector<int> h_iw2it = {0, 0};
    const std::vector<int> h_iw2ic = {0, 1};
    const std::vector<int> h_iw2lm = {0, 1};
    const std::vector<int> h_iw2l = {0, 1};

    DeviceBuffer<float> d_gk(h_gk);
    DeviceBuffer<float> d_ylm(h_ylm);
    DeviceBuffer<float> d_table(h_table);
    DeviceBuffer<Complex> d_sk(h_sk);
    DeviceBuffer<Complex> d_psi(natomwfc * npwk_max);
    DeviceBuffer<int> d_iw2iat(h_iw2iat);
    DeviceBuffer<int> d_iw2it(h_iw2it);
    DeviceBuffer<int> d_iw2ic(h_iw2ic);
    DeviceBuffer<int> d_iw2lm(h_iw2lm);
    DeviceBuffer<int> d_iw2l(h_iw2l);
    const psi::AtomicInitTableView<float> table_view = {total_lm,
                                                       nchi_max,
                                                       nqx,
                                                       1.0f,
                                                       1.0f,
                                                       d_table.data(),
                                                       d_iw2iat.data(),
                                                       d_iw2it.data(),
                                                       d_iw2ic.data(),
                                                       d_iw2lm.data(),
                                                       d_iw2l.data()};

    psi::init_atomic_op<Complex, base_device::DEVICE_GPU>()(nullptr,
                                                            d_psi.data(),
                                                            natomwfc,
                                                            npw,
                                                            npwk_max,
                                                            d_gk.data(),
                                                            d_ylm.data(),
                                                            d_sk.data(),
                                                            table_view);

    const std::vector<Complex> h_psi = d_psi.copy_to_host();
    EXPECT_NEAR(h_psi[0].real(), 1.0f, 1.0e-6f);
    EXPECT_NEAR(h_psi[0].imag(), 1.0f, 1.0e-6f);
    EXPECT_NEAR(h_psi[1].real(), 1.5f, 1.0e-6f);
    EXPECT_NEAR(h_psi[1].imag(), -0.75f, 1.0e-6f);
    EXPECT_EQ(h_psi[2], Complex(0.0f, 0.0f));
    EXPECT_NEAR(h_psi[3].real(), 5.25f, 1.0e-6f);
    EXPECT_NEAR(h_psi[3].imag(), 10.5f, 1.0e-6f);
    EXPECT_NEAR(h_psi[4].real(), -8.0f, 1.0e-6f);
    EXPECT_NEAR(h_psi[4].imag(), 2.0f, 1.0e-6f);
    EXPECT_EQ(h_psi[5], Complex(0.0f, 0.0f));

}

TEST(PsiInitGpuAtomicRandom, PerturbsAtomicOrbitalsInPlaceDeterministically)
{
    constexpr int nbands = 2;
    constexpr int npw = 2;
    constexpr int npwk_max = 3;
    const std::vector<Complex> h_input = {Complex(1.0f, 0.0f), Complex(2.0f, -1.0f), Complex(0.0f, 0.0f),
                                          Complex(-1.0f, 0.5f), Complex(0.5f, 2.0f), Complex(0.0f, 0.0f)};
    const std::vector<float> h_gk2 = {0.0f, 2.0f, 0.0f};
    DeviceBuffer<float> d_gk2(h_gk2);
    DeviceBuffer<Complex> d_first(h_input);
    DeviceBuffer<Complex> d_second(h_input);

    psi::perturb_atomic_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_first.data(), nbands, npw, npwk_max, 1, 0, 3, 17, 0.05f, d_gk2.data(), nullptr, nullptr, 0, 0);
    psi::perturb_atomic_op<Complex, base_device::DEVICE_GPU>()(
        nullptr,
        d_second.data(),
        nbands,
        npw,
        npwk_max,
        1,
        0,
        3,
        17,
        0.05f,
        d_gk2.data(),
        nullptr,
        nullptr,
        0,
        0);

    const std::vector<Complex> h_first = d_first.copy_to_host();
    const std::vector<Complex> h_second = d_second.copy_to_host();
    EXPECT_EQ(h_first, h_second);
    EXPECT_NE(h_first[0], h_input[0]);
    EXPECT_NE(h_first[1], h_input[1]);
    EXPECT_NE(h_first[3], h_input[3]);
    EXPECT_NE(h_first[4], h_input[4]);
    EXPECT_EQ(h_first[2], Complex(0.0f, 0.0f));
    EXPECT_EQ(h_first[5], Complex(0.0f, 0.0f));

}

TEST(PsiInitGpuRandom, DifferentSeedChangesValues)
{
    const std::vector<Complex> first = run_random_init(17);
    const std::vector<Complex> second = run_random_init(18);
    bool found_difference = false;
    for (int i = 0; i < static_cast<int>(first.size()); ++i)
    {
        if (first[i] != second[i])
        {
            found_difference = true;
            break;
        }
    }
    EXPECT_TRUE(found_difference);
}

TEST(PsiInitGpuRandom, UnseededPathAppliesCpuCompatibleExtraGk2Damping)
{
    const std::vector<Complex> unseeded = run_random_init(0);
    const std::vector<Complex> seeded = run_random_init(17);

    constexpr int npwk_max = 5;
    constexpr int npol = 2;
    const int first_band_second_pol = npwk_max;

    EXPECT_LT(std::abs(unseeded[1]), 0.5f);
    EXPECT_LT(std::abs(unseeded[2]), 0.25f);
    EXPECT_LT(std::abs(unseeded[first_band_second_pol + 1]), 0.5f);
    EXPECT_LT(std::abs(unseeded[first_band_second_pol + 2]), 0.25f);

    EXPECT_GT(std::abs(seeded[1]), 0.0f);
    EXPECT_GT(std::abs(seeded[2]), 0.0f);
}

TEST(PsiInitGpuRandom, SeededInitializationUsesStickMappedRandomValues)
{
    const std::vector<Complex> psi = run_stick_mapped_random_init();

    ASSERT_EQ(psi.size(), 5);
    EXPECT_NE(psi[1], Complex(0.0f, 0.0f));
    EXPECT_EQ(psi[1], psi[2]);
}

TEST(PsiInitGpuRandom, SeededInitializationUsesGlobalStickDrawSlots)
{
    const std::vector<Complex> psi = run_seeded_sequence_random_init();

    ASSERT_EQ(psi.size(), 2);
    EXPECT_NE(psi[0], psi[1]);
    EXPECT_NEAR(psi[0].real(), seeded_sequence_golden_0.real(), 1.0e-6f);
    EXPECT_NEAR(psi[0].imag(), seeded_sequence_golden_0.imag(), 1.0e-6f);
    EXPECT_NEAR(psi[1].real(), seeded_sequence_golden_1.real(), 1.0e-6f);
    EXPECT_NEAR(psi[1].imag(), seeded_sequence_golden_1.imag(), 1.0e-6f);
}

TEST(PsiInitGpuRandom, DoublePrecisionUsesFixedSequenceGoldens)
{
    using DoubleComplex = std::complex<double>;
    const std::vector<double> h_gk2(1, 0.0);
    const std::vector<int> h_igl2isz(1, 0);
    const std::vector<int> h_is2fftixy(1, 0);
    DeviceBuffer<double> d_gk2(h_gk2);
    DeviceBuffer<int> d_igl2isz(h_igl2isz);
    DeviceBuffer<int> d_is2fftixy(h_is2fftixy);
    DeviceBuffer<DoubleComplex> d_psi(2);

    psi::init_random_op<DoubleComplex, base_device::DEVICE_GPU>()(nullptr,
                                                                  d_psi.data(),
                                                                  2,
                                                                  1,
                                                                  1,
                                                                  1,
                                                                  0,
                                                                  2,
                                                                  17,
                                                                  d_gk2.data(),
                                                                  d_igl2isz.data(),
                                                                  d_is2fftixy.data(),
                                                                  1,
                                                                  1);

    const std::vector<DoubleComplex> values = d_psi.copy_to_host();
    ASSERT_EQ(values.size(), 2);
    EXPECT_NEAR(values[0].real(), 0.31092915, 1.0e-6);
    EXPECT_NEAR(values[0].imag(), -0.85787631, 1.0e-6);
    EXPECT_NEAR(values[1].real(), -0.30231641, 1.0e-6);
    EXPECT_NEAR(values[1].imag(), -0.02610150, 1.0e-6);
}

TEST(PsiInitGpuRandom, SeededInitializationUsesGlobalKIndexInRandomKey)
{
    const std::vector<Complex> local_zero_global_two = run_seeded_global_k_random_init(2);
    const std::vector<Complex> local_zero_global_three = run_seeded_global_k_random_init(3);

    ASSERT_EQ(local_zero_global_two.size(), 1);
    ASSERT_EQ(local_zero_global_three.size(), 1);
    EXPECT_NE(local_zero_global_two[0], local_zero_global_three[0]);
    EXPECT_NEAR(local_zero_global_two[0].real(), seeded_sequence_golden_0.real(), 1.0e-6f);
    EXPECT_NEAR(local_zero_global_two[0].imag(), seeded_sequence_golden_0.imag(), 1.0e-6f);
}

TEST(PsiInitGpuRandom, SeededInitializationIsInvariantToLocalStickDecomposition)
{
    const Complex first = run_decomposed_seeded_random(0, std::vector<int>{7});
    const Complex second = run_decomposed_seeded_random(2, std::vector<int>{1, 4, 7});
    EXPECT_EQ(first, second);
}

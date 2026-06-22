#include "source_base/module_device/memory_op.h"
#include "source_psi/kernels/psi_init_op.h"
#include "source_psi/psi_init_policy.h"

#include <gtest/gtest.h>

#include <complex>
#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

using Complex = std::complex<float>;
using sync_h2d_op = base_device::memory::synchronize_memory_op<float, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using sync_int_h2d_op = base_device::memory::synchronize_memory_op<int, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using sync_d2h_op = base_device::memory::synchronize_memory_op<Complex, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using resize_float_op = base_device::memory::resize_memory_op<float, base_device::DEVICE_GPU>;
using resize_int_op = base_device::memory::resize_memory_op<int, base_device::DEVICE_GPU>;
using resize_complex_op = base_device::memory::resize_memory_op<Complex, base_device::DEVICE_GPU>;
using delete_float_op = base_device::memory::delete_memory_op<float, base_device::DEVICE_GPU>;
using delete_int_op = base_device::memory::delete_memory_op<int, base_device::DEVICE_GPU>;
using delete_complex_op = base_device::memory::delete_memory_op<Complex, base_device::DEVICE_GPU>;

std::uint64_t splitmix64_host(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

float unit_random_host(const std::uint64_t key)
{
    constexpr double norm = 1.0 / 9007199254740992.0;
    return static_cast<float>(static_cast<double>(splitmix64_host(key) >> 11) * norm);
}

Complex expected_sequence_value(const int seed,
                                const int ik_tot,
                                const int iband,
                                const int ipol,
                                const int isz,
                                const int nst,
                                const int nz,
                                const float gk2)
{
    const int is = isz / nz;
    const int iz = isz - is * nz;
    const std::uint64_t sequence_index =
        ((static_cast<std::uint64_t>(iband) * 1ULL + static_cast<std::uint64_t>(ipol))
             * static_cast<std::uint64_t>(nst)
         + static_cast<std::uint64_t>(is))
            * static_cast<std::uint64_t>(nz)
        + static_cast<std::uint64_t>(iz);
    const std::uint64_t seed_key = static_cast<std::uint64_t>(seed);
    const std::uint64_t ik_key = static_cast<std::uint64_t>(ik_tot + 1) * 0x9e3779b97f4a7c15ULL;
    const std::uint64_t rr_key = seed_key ^ ik_key ^ ((2ULL * sequence_index + 1ULL) * 0xd2b74407b1ce6e93ULL);
    const std::uint64_t arg_key = seed_key ^ ik_key ^ ((2ULL * sequence_index + 2ULL) * 0xd2b74407b1ce6e93ULL);
    const float rr = unit_random_host(rr_key);
    const float arg =
        static_cast<float>(6.283185307179586476925286766559) * unit_random_host(arg_key);
    const float damping = 1.0f / (gk2 + 1.0f);
    return Complex(rr * std::cos(arg) * damping, rr * std::sin(arg) * damping);
}

std::vector<Complex> run_random_init(const int seed)
{
    constexpr int nbands = 2;
    constexpr int npwk = 3;
    constexpr int npwk_max = 5;
    constexpr int npol = 2;
    constexpr int total = nbands * npwk_max * npol;

    std::vector<float> h_gk2 = {0.0f, 1.0f, 3.0f, 0.0f, 0.0f, 0.0f, 1.0f, 3.0f, 0.0f, 0.0f};
    float* d_gk2 = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 1, 1, seed, d_gk2, nullptr, 0, 0);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_complex_op()(d_psi);
    return h_psi;
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
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 0, 0, 17, d_gk2, d_igl2isz, 4, 4);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_complex_op()(d_psi);
    return h_psi;
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
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 0, 2, 17, d_gk2, d_igl2isz, 1, 1);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_complex_op()(d_psi);
    return h_psi;
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
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 0, ik_tot, 17, d_gk2, d_igl2isz, 1, 1);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_complex_op()(d_psi);
    return h_psi;
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
    EXPECT_TRUE(psi::gpu_random_init_policy(true, false, "dav_subspace", true, "random"));
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

TEST(PsiInitGpuRandom, SeededInitializationUsesCpuTraversalDrawSlots)
{
    const std::vector<Complex> psi = run_seeded_sequence_random_init();

    ASSERT_EQ(psi.size(), 2);
    EXPECT_NE(psi[0], psi[1]);
    EXPECT_NEAR(psi[0].real(), expected_sequence_value(17, 2, 0, 0, 0, 1, 1, 0.0f).real(), 1.0e-6f);
    EXPECT_NEAR(psi[0].imag(), expected_sequence_value(17, 2, 0, 0, 0, 1, 1, 0.0f).imag(), 1.0e-6f);
    EXPECT_NEAR(psi[1].real(), expected_sequence_value(17, 2, 1, 0, 0, 1, 1, 0.0f).real(), 1.0e-6f);
    EXPECT_NEAR(psi[1].imag(), expected_sequence_value(17, 2, 1, 0, 0, 1, 1, 0.0f).imag(), 1.0e-6f);
}

TEST(PsiInitGpuRandom, SeededInitializationUsesGlobalKIndexInRandomKey)
{
    const std::vector<Complex> local_zero_global_two = run_seeded_global_k_random_init(2);
    const std::vector<Complex> local_zero_global_three = run_seeded_global_k_random_init(3);

    ASSERT_EQ(local_zero_global_two.size(), 1);
    ASSERT_EQ(local_zero_global_three.size(), 1);
    EXPECT_NE(local_zero_global_two[0], local_zero_global_three[0]);
    EXPECT_NEAR(local_zero_global_two[0].real(), expected_sequence_value(17, 2, 0, 0, 0, 1, 1, 0.0f).real(), 1.0e-6f);
    EXPECT_NEAR(local_zero_global_two[0].imag(), expected_sequence_value(17, 2, 0, 0, 0, 1, 1, 0.0f).imag(), 1.0e-6f);
}

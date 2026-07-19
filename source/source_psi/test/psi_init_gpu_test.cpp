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
using sync_complex_h2d_op
    = base_device::memory::synchronize_memory_op<Complex, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using sync_d2h_op = base_device::memory::synchronize_memory_op<Complex, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using sync_float_d2h_op
    = base_device::memory::synchronize_memory_op<float, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
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
                                const int global_stick,
                                const int fftnxy,
                                const int nz,
                                const float gk2)
{
    const int iz = 0;
    const std::uint64_t sequence_index =
        ((static_cast<std::uint64_t>(iband) * 1ULL + static_cast<std::uint64_t>(ipol))
             * static_cast<std::uint64_t>(fftnxy)
         + static_cast<std::uint64_t>(global_stick))
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
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 1, 1, seed, d_gk2, nullptr, nullptr, 0, 0);

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
    std::vector<int> h_is2fftixy = {0, 3, 7, 11};
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    int* d_is2fftixy = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_int_op()(d_is2fftixy, h_is2fftixy.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());
    sync_int_h2d_op()(d_is2fftixy, h_is2fftixy.data(), h_is2fftixy.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 0, 0, 17, d_gk2, d_igl2isz, d_is2fftixy, 16, 4);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_int_op()(d_is2fftixy);
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
    std::vector<int> h_is2fftixy = {0};
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    int* d_is2fftixy = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_int_op()(d_is2fftixy, h_is2fftixy.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());
    sync_int_h2d_op()(d_is2fftixy, h_is2fftixy.data(), h_is2fftixy.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 0, 2, 17, d_gk2, d_igl2isz, d_is2fftixy, 1, 1);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_int_op()(d_is2fftixy);
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
    std::vector<int> h_is2fftixy = {0};
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    int* d_is2fftixy = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_int_op()(d_is2fftixy, h_is2fftixy.size());
    resize_complex_op()(d_psi, total);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());
    sync_int_h2d_op()(d_is2fftixy, h_is2fftixy.data(), h_is2fftixy.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_psi, nbands, npwk, npwk_max, npol, 0, ik_tot, 17, d_gk2, d_igl2isz, d_is2fftixy, 1, 1);

    std::vector<Complex> h_psi(total);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_int_op()(d_is2fftixy);
    delete_complex_op()(d_psi);
    return h_psi;
}

Complex run_decomposed_seeded_random(const int local_stick, const std::vector<int>& h_is2fftixy)
{
    constexpr int nz = 4;
    const std::vector<float> h_gk2 = {0.0f};
    const std::vector<int> h_igl2isz = {local_stick * nz + 2};
    float* d_gk2 = nullptr;
    int* d_igl2isz = nullptr;
    int* d_is2fftixy = nullptr;
    Complex* d_psi = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_int_op()(d_igl2isz, h_igl2isz.size());
    resize_int_op()(d_is2fftixy, h_is2fftixy.size());
    resize_complex_op()(d_psi, 1);
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_int_h2d_op()(d_igl2isz, h_igl2isz.data(), h_igl2isz.size());
    sync_int_h2d_op()(d_is2fftixy, h_is2fftixy.data(), h_is2fftixy.size());

    psi::init_random_op<Complex, base_device::DEVICE_GPU>()(nullptr,
                                                            d_psi,
                                                            1,
                                                            1,
                                                            1,
                                                            1,
                                                            0,
                                                            0,
                                                            17,
                                                            d_gk2,
                                                            d_igl2isz,
                                                            d_is2fftixy,
                                                            16,
                                                            nz);

    Complex h_psi;
    sync_d2h_op()(&h_psi, d_psi, 1);
    delete_float_op()(d_gk2);
    delete_int_op()(d_igl2isz);
    delete_int_op()(d_is2fftixy);
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

TEST(PsiInitGpuPolicy, AllowsCollinearAtomicInitializers)
{
    EXPECT_TRUE(psi::gpu_resident_init_policy(true, false, "dav_subspace", true, "atomic", 1));
    EXPECT_TRUE(psi::gpu_resident_init_policy(true, false, "dav_subspace", true, "atomic+random", 1));
}

TEST(PsiInitGpuPolicy, KeepsUnsupportedAtomicInitializersOnCpu)
{
    EXPECT_FALSE(psi::gpu_resident_init_policy(true, false, "bpcg", true, "atomic", 1));
    EXPECT_FALSE(psi::gpu_resident_init_policy(true, false, "dav_subspace", true, "atomic", 2));
    EXPECT_FALSE(psi::gpu_resident_init_policy(false, false, "dav_subspace", true, "atomic+random", 1));
    EXPECT_FALSE(psi::gpu_resident_init_policy(true, true, "dav_subspace", true, "atomic+random", 1));
}

TEST(PsiInitGpuAtomic, BuildsCartesianGPlusKOnDevice)
{
    const std::vector<float> h_gcar = {1.0f, 2.0f, 3.0f, -1.0f, -2.0f, -3.0f,
                                       4.0f, 5.0f, 6.0f,  7.0f,  8.0f,  9.0f};
    const std::vector<float> h_kvec = {0.5f, -0.5f, 1.0f, -1.0f, 2.0f, 0.25f};
    float* d_gcar = nullptr;
    float* d_kvec = nullptr;
    float* d_gk = nullptr;
    resize_float_op()(d_gcar, h_gcar.size());
    resize_float_op()(d_kvec, h_kvec.size());
    resize_float_op()(d_gk, 6);
    sync_h2d_op()(d_gcar, h_gcar.data(), h_gcar.size());
    sync_h2d_op()(d_kvec, h_kvec.data(), h_kvec.size());

    psi::build_gk_op<float, base_device::DEVICE_GPU>()(nullptr, d_gk, d_gcar, d_kvec, 1, 2, 2);

    std::vector<float> h_gk(6);
    sync_float_d2h_op()(h_gk.data(), d_gk, h_gk.size());
    EXPECT_EQ(h_gk, (std::vector<float>{3.0f, 7.0f, 6.25f, 6.0f, 10.0f, 9.25f}));

    delete_float_op()(d_gcar);
    delete_float_op()(d_kvec);
    delete_float_op()(d_gk);
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

    float* d_gk = nullptr;
    float* d_ylm = nullptr;
    float* d_table = nullptr;
    Complex* d_sk = nullptr;
    Complex* d_psi = nullptr;
    int* d_iw2iat = nullptr;
    int* d_iw2it = nullptr;
    int* d_iw2ic = nullptr;
    int* d_iw2lm = nullptr;
    int* d_iw2l = nullptr;
    resize_float_op()(d_gk, h_gk.size());
    resize_float_op()(d_ylm, h_ylm.size());
    resize_float_op()(d_table, h_table.size());
    resize_complex_op()(d_sk, h_sk.size());
    resize_complex_op()(d_psi, natomwfc * npwk_max);
    resize_int_op()(d_iw2iat, h_iw2iat.size());
    resize_int_op()(d_iw2it, h_iw2it.size());
    resize_int_op()(d_iw2ic, h_iw2ic.size());
    resize_int_op()(d_iw2lm, h_iw2lm.size());
    resize_int_op()(d_iw2l, h_iw2l.size());
    sync_h2d_op()(d_gk, h_gk.data(), h_gk.size());
    sync_h2d_op()(d_ylm, h_ylm.data(), h_ylm.size());
    sync_h2d_op()(d_table, h_table.data(), h_table.size());
    sync_complex_h2d_op()(d_sk, h_sk.data(), h_sk.size());
    sync_int_h2d_op()(d_iw2iat, h_iw2iat.data(), h_iw2iat.size());
    sync_int_h2d_op()(d_iw2it, h_iw2it.data(), h_iw2it.size());
    sync_int_h2d_op()(d_iw2ic, h_iw2ic.data(), h_iw2ic.size());
    sync_int_h2d_op()(d_iw2lm, h_iw2lm.data(), h_iw2lm.size());
    sync_int_h2d_op()(d_iw2l, h_iw2l.data(), h_iw2l.size());

    psi::init_atomic_op<Complex, base_device::DEVICE_GPU>()(nullptr,
                                                            d_psi,
                                                            natomwfc,
                                                            npw,
                                                            npwk_max,
                                                            total_lm,
                                                            nchi_max,
                                                            nqx,
                                                            1.0f,
                                                            1.0f,
                                                            d_gk,
                                                            d_ylm,
                                                            d_sk,
                                                            d_table,
                                                            d_iw2iat,
                                                            d_iw2it,
                                                            d_iw2ic,
                                                            d_iw2lm,
                                                            d_iw2l);

    std::vector<Complex> h_psi(natomwfc * npwk_max);
    sync_d2h_op()(h_psi.data(), d_psi, h_psi.size());
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

    delete_float_op()(d_gk);
    delete_float_op()(d_ylm);
    delete_float_op()(d_table);
    delete_complex_op()(d_sk);
    delete_complex_op()(d_psi);
    delete_int_op()(d_iw2iat);
    delete_int_op()(d_iw2it);
    delete_int_op()(d_iw2ic);
    delete_int_op()(d_iw2lm);
    delete_int_op()(d_iw2l);
}

TEST(PsiInitGpuAtomicRandom, PerturbsAtomicOrbitalsInPlaceDeterministically)
{
    constexpr int nbands = 2;
    constexpr int npw = 2;
    constexpr int npwk_max = 3;
    const std::vector<Complex> h_input = {Complex(1.0f, 0.0f), Complex(2.0f, -1.0f), Complex(0.0f, 0.0f),
                                          Complex(-1.0f, 0.5f), Complex(0.5f, 2.0f), Complex(0.0f, 0.0f)};
    const std::vector<float> h_gk2 = {0.0f, 2.0f, 0.0f};
    float* d_gk2 = nullptr;
    Complex* d_first = nullptr;
    Complex* d_second = nullptr;
    resize_float_op()(d_gk2, h_gk2.size());
    resize_complex_op()(d_first, h_input.size());
    resize_complex_op()(d_second, h_input.size());
    sync_h2d_op()(d_gk2, h_gk2.data(), h_gk2.size());
    sync_complex_h2d_op()(d_first, h_input.data(), h_input.size());
    sync_complex_h2d_op()(d_second, h_input.data(), h_input.size());

    psi::perturb_atomic_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_first, nbands, npw, npwk_max, 1, 0, 3, 17, 0.05f, d_gk2, nullptr, nullptr, 0, 0);
    psi::perturb_atomic_op<Complex, base_device::DEVICE_GPU>()(
        nullptr, d_second, nbands, npw, npwk_max, 1, 0, 3, 17, 0.05f, d_gk2, nullptr, nullptr, 0, 0);

    std::vector<Complex> h_first(h_input.size());
    std::vector<Complex> h_second(h_input.size());
    sync_d2h_op()(h_first.data(), d_first, h_first.size());
    sync_d2h_op()(h_second.data(), d_second, h_second.size());
    EXPECT_EQ(h_first, h_second);
    EXPECT_NE(h_first[0], h_input[0]);
    EXPECT_NE(h_first[1], h_input[1]);
    EXPECT_NE(h_first[3], h_input[3]);
    EXPECT_NE(h_first[4], h_input[4]);
    EXPECT_EQ(h_first[2], Complex(0.0f, 0.0f));
    EXPECT_EQ(h_first[5], Complex(0.0f, 0.0f));

    delete_float_op()(d_gk2);
    delete_complex_op()(d_first);
    delete_complex_op()(d_second);
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

TEST(PsiInitGpuRandom, SeededInitializationIsInvariantToLocalStickDecomposition)
{
    const Complex first = run_decomposed_seeded_random(0, std::vector<int>{7});
    const Complex second = run_decomposed_seeded_random(2, std::vector<int>{1, 4, 7});
    EXPECT_EQ(first, second);
}

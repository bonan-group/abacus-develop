#include "cuda_runtime.h"
#include "source_basis/module_pw/pw_basis_k.h"
#include "source_pw/module_pwdft/kernels/exx_q_state_op.h"

#include <complex>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace
{
using complexd = std::complex<double>;

struct CudaComplexDeleter
{
    void operator()(complexd* ptr) const
    {
        if (ptr != nullptr)
        {
            cudaFree(ptr);
        }
    }
};

void init_pw(ModulePW::PW_Basis_K& wfcpw)
{
    const double lat0 = 1.8897261254578281;
    const ModuleBase::Matrix3 latvec(10.0, 0.0, 0.0,
                                     0.0, 10.0, 0.0,
                                     0.0, 0.0, 10.0);
    const ModuleBase::Vector3<double> kvec_d[1] = {ModuleBase::Vector3<double>(0.0, 0.0, 0.0)};
    wfcpw.initmpi(1, 0, MPI_COMM_WORLD);
    wfcpw.initgrids(lat0, latvec, 10.0);
    wfcpw.initparameters(false, 2.0, 1, kvec_d, 1, true);
    wfcpw.setuptransform();
    wfcpw.collect_local_pw();
}

K_Vectors::ExxFullPoint make_point(bool time_reversal)
{
    K_Vectors::ExxFullPoint point;
    point.identity = false;
    point.conjugate_only = false;
    point.time_reversal = time_reversal;
    point.full_kvec_d = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    point.full_kvec_c = ModuleBase::Vector3<double>(0.0, 0.0, 0.0);
    point.gmatrix = ModuleBase::Matrix3(0.0, -1.0, 0.0,
                                        1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0);
    point.kgmatrix = point.gmatrix;
    point.gtrans = ModuleBase::Vector3<double>(0.25, 0.0, 0.0);
    return point;
}

void expect_gpu_matches_host(bool time_reversal)
{
    ModulePW::PW_Basis_K wfcpw("gpu", "double");
    wfcpw.fft_bundle.setfft("gpu", "double");
    init_pw(wfcpw);

    const int batch_count = 2;
    const std::size_t state_size = static_cast<std::size_t>(wfcpw.nrxx);
    std::vector<complexd> representative(batch_count * state_size);
    for (std::size_t i = 0; i < representative.size(); ++i)
    {
        representative[i] = complexd(0.013 * (i + 1), -0.007 * ((i + 5) % 17));
    }

    const K_Vectors::ExxFullPoint point = make_point(time_reversal);
    std::vector<complexd> expected(batch_count * state_size);
    for (int ib = 0; ib < batch_count; ++ib)
    {
        hamilt::rotate_exx_realspace_symmetry_cpu(&wfcpw,
                                                   point,
                                                   0,
                                                   representative.data() + ib * state_size,
                                                   expected.data() + ib * state_size);
    }

    complexd* representative_raw = nullptr;
    complexd* full_raw = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&representative_raw), representative.size() * sizeof(complexd)),
              cudaSuccess);
    std::unique_ptr<complexd, CudaComplexDeleter> representative_device(representative_raw);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&full_raw), expected.size() * sizeof(complexd)), cudaSuccess);
    std::unique_ptr<complexd, CudaComplexDeleter> full_device(full_raw);
    ASSERT_EQ(cudaMemcpy(representative_device.get(),
                         representative.data(),
                         representative.size() * sizeof(complexd),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    hamilt::exx_rotate_realspace_op<complexd, base_device::DEVICE_GPU>()(&wfcpw,
                                                                         point,
                                                                         0,
                                                                         representative_device.get(),
                                                                         full_device.get(),
                                                                         batch_count);

    std::vector<complexd> actual(expected.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), full_device.get(), actual.size() * sizeof(complexd), cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(actual[i].real(), expected[i].real(), 1.0e-12) << "index " << i;
        EXPECT_NEAR(actual[i].imag(), expected[i].imag(), 1.0e-12) << "index " << i;
    }
}
} // namespace

TEST(ExxRealspaceSymmetryGpuTest, BatchedRotationMatchesHostReference)
{
    expect_gpu_matches_host(false);
}

TEST(ExxRealspaceSymmetryGpuTest, BatchedTimeReversalMatchesHostReference)
{
    expect_gpu_matches_host(true);
}

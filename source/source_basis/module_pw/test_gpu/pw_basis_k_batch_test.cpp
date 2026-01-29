#include "mpi.h"
#include "cuda_runtime.h"
#include "source_base/module_device/device.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/vector3.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_basis_k.h"

#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <typeinfo>

using namespace std;

/**
 * @brief Test suite for PW_Basis_K batch FFT transforms
 *
 * Tests batch versions of recip_to_real and real_to_recip against
 * sequential implementations to verify correctness.
 */
template <typename T>
class PW_BASIS_K_BATCH_TEST : public ::testing::Test
{
  public:
    using Real = T;
    using Device = base_device::DEVICE_GPU;

    ModulePW::PW_Basis_K pwtest;

    // Device arrays for batch testing
    complex<T>* d_in_batch = nullptr;
    complex<T>* d_out_batch_seq = nullptr;
    complex<T>* d_out_batch = nullptr;
    complex<T>* d_tmp_seq = nullptr;

    // Host arrays for verification
    complex<T>* h_out_seq = nullptr;
    complex<T>* h_out_batch = nullptr;

    int nks = 0;
    int npwk_max = 0;
    int nrxx = 0;
    int npw = 0;

    void SetUp() override
    {
        // Set device to GPU before initialization
        pwtest.set_device("gpu");

        // Set precision based on template parameter T
        if (typeid(T) == typeid(float))
        {
            pwtest.set_precision("single");
        }
        else
        {
            pwtest.set_precision("double");
        }

        // IMPORTANT: Also set device/precision on fft_bundle directly
        // (PW_Basis_K::set_device/set_precision don't propagate to fft_bundle)
        pwtest.fft_bundle.setfft(pwtest.get_device(), pwtest.get_precision());

        // Initialize PW_Basis_K with multiple k-points
        // Strategy: Small grid + very high cutoff = keep most/all modes
        ModuleBase::Matrix3 latvec(1, 0, 0, 0, 1, 0, 0, 0, 1);
        T lat0 = 10.0;  // Large lat0 makes grid smaller for given cutoff
        T wfcecut = 100.0;  // Very high cutoff to keep all modes in small grid
        bool gamma_only = false;
        int distribution_type = 1;
        bool xprime = false;

        // Setup 8 k-points with small values
        nks = 8;
        ModuleBase::Vector3<double>* kvec_d = new ModuleBase::Vector3<double>[nks];
        for (int ik = 0; ik < nks; ik++)
        {
            kvec_d[ik].set(0.001 * ik, 0.001 * ik, 0.001 * ik);
        }

        // Initialize MPI (required even for single-process tests)
        const int mypool = 0;
        const int key = 1;
        const int nproc_in_pool = 1;
        const int rank_in_pool = 0;
        MPI_Comm POOL_WORLD;
        MPI_Comm_split(MPI_COMM_WORLD, mypool, key, &POOL_WORLD);

        // Initialize PW_Basis_K
        pwtest.initmpi(nproc_in_pool, rank_in_pool, POOL_WORLD);
        pwtest.initgrids(lat0, latvec, wfcecut);
        pwtest.initparameters(gamma_only, wfcecut, nks, kvec_d, distribution_type, xprime);
        pwtest.setuptransform();
        pwtest.collect_local_pw();

        // Setup batch FFT
        pwtest.fft_bundle.setupBatchFFT();

        // Store dimensions
        npwk_max = pwtest.npwk_max;
        npw = pwtest.npw;
        nrxx = pwtest.nrxx;

        // Allocate device memory for batch operations using template-aware operations
        const int batch_size = 8;
        base_device::memory::resize_memory_op<complex<T>, Device>()(d_in_batch, batch_size * nrxx);
        base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_batch_seq, batch_size * npwk_max);
        base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_batch, batch_size * npwk_max);
        base_device::memory::resize_memory_op<complex<T>, Device>()(d_tmp_seq, nrxx);

        // Allocate host memory for verification
        h_out_seq = new complex<T>[batch_size * npwk_max];
        h_out_batch = new complex<T>[batch_size * npwk_max];

        delete[] kvec_d;
    }

    void TearDown() override
    {
        if (d_in_batch) base_device::memory::delete_memory_op<complex<T>, Device>()(d_in_batch);
        if (d_out_batch_seq) base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_batch_seq);
        if (d_out_batch) base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_batch);
        if (d_tmp_seq) base_device::memory::delete_memory_op<complex<T>, Device>()(d_tmp_seq);

        delete[] h_out_seq;
        delete[] h_out_batch;
    }

    void FillRandomData(complex<T>* data_device, int size)
    {
        // Generate random data on host
        std::vector<complex<T>> h_data(size);
        std::mt19937 gen(12345); // Fixed seed for reproducibility
        std::uniform_real_distribution<T> dis(-1.0, 1.0);

        for (int i = 0; i < size; i++)
        {
            h_data[i] = complex<T>(dis(gen), dis(gen));
        }

        // Copy to device
        cudaMemcpy(data_device, h_data.data(), size * sizeof(complex<T>), cudaMemcpyHostToDevice);
    }

    T ComputeMaxError(const complex<T>* h_seq, const complex<T>* h_batch, int size)
    {
        T max_error = 0.0;
        for (int i = 0; i < size; i++)
        {
            T error_real = std::abs(h_seq[i].real() - h_batch[i].real());
            T error_imag = std::abs(h_seq[i].imag() - h_batch[i].imag());
            max_error = std::max(max_error, std::max(error_real, error_imag));
        }
        return max_error;
    }
};

using TestTypes = ::testing::Types<double, float>;
TYPED_TEST_SUITE(PW_BASIS_K_BATCH_TEST, TestTypes);

TYPED_TEST(PW_BASIS_K_BATCH_TEST, RecipToRealBatch)
{
    using T = typename TestFixture::Real;
    using Device = typename TestFixture::Device;

    const int batch_count = 8;
    const int nrxx = this->nrxx;
    const int npwk_max = this->npwk_max;

    // All batch transforms use the same k-point (batch function takes single ik)
    const int ik = 0;

    // Allocate and fill input data (reciprocal space, sparse)
    complex<T>* d_in_recip_batch = nullptr;
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_in_recip_batch, batch_count * npwk_max);
    this->FillRandomData(d_in_recip_batch, batch_count * npwk_max);

    complex<T>* d_out_real_seq = nullptr;
    complex<T>* d_out_real_batch = nullptr;
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_real_seq, batch_count * nrxx);
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_real_batch, batch_count * nrxx);

    // Sequential transforms (using template version like existing tests)
    for (int ib = 0; ib < batch_count; ib++)
    {
        this->pwtest.template recip_to_real<std::complex<T>, Device>(
            d_in_recip_batch + ib * npwk_max,
            d_out_real_seq + ib * nrxx,
            ik,
            false,
            T(1.0));
    }

    // Batch transform (using template version)
    this->pwtest.template recip_to_real_batch<T, Device>(
        nullptr,  // ctx parameter
        d_in_recip_batch,
        d_out_real_batch,
        ik,
        batch_count,
        false,
        T(1.0));

    // Copy results to host
    complex<T>* h_out_seq = new complex<T>[batch_count * nrxx];
    complex<T>* h_out_batch = new complex<T>[batch_count * nrxx];
    cudaMemcpy(h_out_seq, d_out_real_seq, batch_count * nrxx * sizeof(complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_batch, d_out_real_batch, batch_count * nrxx * sizeof(complex<T>), cudaMemcpyDeviceToHost);

    // Verify correctness
    T max_error = this->ComputeMaxError(h_out_seq, h_out_batch, batch_count * nrxx);

    // For double precision, expect very high accuracy (< 1e-10)
    // For single precision, expect reasonable accuracy (< 1e-4)
    T tolerance = (typeid(T) == typeid(double)) ? T(1e-10) : T(1e-4);

    EXPECT_LT(max_error, tolerance)
        << "Max error: " << max_error
        << " exceeds tolerance: " << tolerance;

    // Cleanup
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_in_recip_batch);
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_real_seq);
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_real_batch);
    delete[] h_out_seq;
    delete[] h_out_batch;
}

TYPED_TEST(PW_BASIS_K_BATCH_TEST, RealToRecipBatch)
{
    using T = typename TestFixture::Real;
    using Device = typename TestFixture::Device;

    const int batch_count = 8;
    const int nrxx = this->nrxx;
    const int npwk_max = this->npwk_max;

    // All batch transforms use the same k-point (batch function takes single ik)
    const int ik = 0;

    // Allocate and fill input data (real space, dense)
    complex<T>* d_in_real_batch = nullptr;
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_in_real_batch, batch_count * nrxx);
    this->FillRandomData(d_in_real_batch, batch_count * nrxx);

    complex<T>* d_out_recip_seq = nullptr;
    complex<T>* d_out_recip_batch = nullptr;
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_recip_seq, batch_count * npwk_max);
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_recip_batch, batch_count * npwk_max);

    // Sequential transforms (using template version like existing tests)
    for (int ib = 0; ib < batch_count; ib++)
    {
        this->pwtest.template real_to_recip<std::complex<T>, Device>(
            d_in_real_batch + ib * nrxx,
            d_out_recip_seq + ib * npwk_max,
            ik,
            false,
            T(1.0));
    }

    // Batch transform
    this->pwtest.template real_to_recip_batch<T, Device>(
        nullptr,  // ctx parameter
        d_in_real_batch,
        d_out_recip_batch,
        ik,
        batch_count,
        false,
        T(1.0));

    // Copy results to host
    complex<T>* h_out_seq = new complex<T>[batch_count * npwk_max];
    complex<T>* h_out_batch = new complex<T>[batch_count * npwk_max];
    cudaMemcpy(h_out_seq, d_out_recip_seq, batch_count * npwk_max * sizeof(complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_batch, d_out_recip_batch, batch_count * npwk_max * sizeof(complex<T>), cudaMemcpyDeviceToHost);

    // Verify correctness
    T max_error = this->ComputeMaxError(h_out_seq, h_out_batch, batch_count * npwk_max);

    T tolerance = (typeid(T) == typeid(double)) ? T(1e-10) : T(1e-4);

    EXPECT_LT(max_error, tolerance)
        << "Max error: " << max_error
        << " exceeds tolerance: " << tolerance;

    // Cleanup
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_in_real_batch);
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_recip_seq);
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_recip_batch);
    delete[] h_out_seq;
    delete[] h_out_batch;
}

TYPED_TEST(PW_BASIS_K_BATCH_TEST, PartialBatch)
{
    using T = typename TestFixture::Real;
    using Device = typename TestFixture::Device;

    // Test with partial batch (5 transforms instead of 8)
    const int batch_count = 5;
    const int nrxx = this->nrxx;
    const int npwk_max = this->npwk_max;

    // All batch transforms use the same k-point (batch function takes single ik)
    const int ik = 0;

    complex<T>* d_in_recip_batch = nullptr;
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_in_recip_batch, batch_count * npwk_max);
    this->FillRandomData(d_in_recip_batch, batch_count * npwk_max);

    complex<T>* d_out_real_seq = nullptr;
    complex<T>* d_out_real_batch = nullptr;
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_real_seq, batch_count * nrxx);
    base_device::memory::resize_memory_op<complex<T>, Device>()(d_out_real_batch, batch_count * nrxx);

    // Sequential
    for (int ib = 0; ib < batch_count; ib++)
    {
        this->pwtest.template recip_to_real<std::complex<T>, Device>(
            d_in_recip_batch + ib * npwk_max,
            d_out_real_seq + ib * nrxx,
            ik);
    }

    // Batch with partial count
    this->pwtest.template recip_to_real_batch<T, Device>(
        nullptr,
        d_in_recip_batch,
        d_out_real_batch,
        ik,
        batch_count);

    // Verify
    complex<T>* h_out_seq = new complex<T>[batch_count * nrxx];
    complex<T>* h_out_batch = new complex<T>[batch_count * nrxx];
    cudaMemcpy(h_out_seq, d_out_real_seq, batch_count * nrxx * sizeof(complex<T>), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_batch, d_out_real_batch, batch_count * nrxx * sizeof(complex<T>), cudaMemcpyDeviceToHost);

    T max_error = this->ComputeMaxError(h_out_seq, h_out_batch, batch_count * nrxx);
    T tolerance = (typeid(T) == typeid(double)) ? T(1e-10) : T(1e-4);

    EXPECT_LT(max_error, tolerance);

    // Cleanup
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_in_recip_batch);
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_real_seq);
    base_device::memory::delete_memory_op<complex<T>, Device>()(d_out_real_batch);
    delete[] h_out_seq;
    delete[] h_out_batch;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
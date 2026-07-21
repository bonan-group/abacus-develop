#include "charge.h"

#include "source_base/module_device/memory_op.h"

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
using resmem_d_gpu_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
using resmem_z_gpu_op = base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>;
using delmem_d_gpu_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
using delmem_z_gpu_op = base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>;
using syncmem_d_h2d_op
    = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using syncmem_d_d2h_op
    = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using syncmem_z_h2d_op = base_device::memory::synchronize_memory_op<std::complex<double>,
                                                                  base_device::DEVICE_GPU,
                                                                  base_device::DEVICE_CPU>;
using syncmem_z_d2h_op = base_device::memory::synchronize_memory_op<std::complex<double>,
                                                                  base_device::DEVICE_CPU,
                                                                  base_device::DEVICE_GPU>;
#endif

class ChargeDeviceStorage
{
  public:
    ChargeDeviceStorage() = default;

    ~ChargeDeviceStorage()
    {
        release();
    }

    ChargeDeviceStorage(const ChargeDeviceStorage&) = delete;
    ChargeDeviceStorage& operator=(const ChargeDeviceStorage&) = delete;

    void allocate(const int nspin, const int nrxx, const int ngmc, const bool allocate_kinetic)
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (nspin > 0 && nrxx > 0)
        {
            resmem_d_gpu_op()(rho_, nspin * nrxx, "Charge::rho_d");
            resmem_d_gpu_op()(rho_save_, nspin * nrxx, "Charge::rho_save_d");
        }
        if (nspin > 0 && ngmc > 0)
        {
            resmem_z_gpu_op()(rhog_, nspin * ngmc, "Charge::rhog_d");
            resmem_z_gpu_op()(rhog_save_, nspin * ngmc, "Charge::rhog_save_d");
        }
        if (allocate_kinetic && nspin > 0 && nrxx > 0)
        {
            resmem_d_gpu_op()(kin_r_, nspin * nrxx, "Charge::kin_r_d");
            resmem_d_gpu_op()(kin_r_save_, nspin * nrxx, "Charge::kin_r_save_d");
        }
#endif
    }

    void release()
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rho_ != nullptr)
        {
            delmem_d_gpu_op()(rho_);
            rho_ = nullptr;
        }
        if (rho_save_ != nullptr)
        {
            delmem_d_gpu_op()(rho_save_);
            rho_save_ = nullptr;
        }
        if (rhog_ != nullptr)
        {
            delmem_z_gpu_op()(rhog_);
            rhog_ = nullptr;
        }
        if (rhog_save_ != nullptr)
        {
            delmem_z_gpu_op()(rhog_save_);
            rhog_save_ = nullptr;
        }
        if (kin_r_ != nullptr)
        {
            delmem_d_gpu_op()(kin_r_);
            kin_r_ = nullptr;
        }
        if (kin_r_save_ != nullptr)
        {
            delmem_d_gpu_op()(kin_r_save_);
            kin_r_save_ = nullptr;
        }
#endif
    }

    double* rho(const int is, const int nrxx) const
    {
        return rho_ == nullptr ? nullptr : rho_ + is * nrxx;
    }

    double* rho_save(const int is, const int nrxx) const
    {
        return rho_save_ == nullptr ? nullptr : rho_save_ + is * nrxx;
    }

    std::complex<double>* rhog(const int is, const int ngmc) const
    {
        return rhog_ == nullptr ? nullptr : rhog_ + is * ngmc;
    }

    std::complex<double>* rhog_save(const int is, const int ngmc) const
    {
        return rhog_save_ == nullptr ? nullptr : rhog_save_ + is * ngmc;
    }

    double* kin_r(const int is, const int nrxx) const
    {
        return kin_r_ == nullptr ? nullptr : kin_r_ + is * nrxx;
    }

    double* kin_r_save(const int is, const int nrxx) const
    {
        return kin_r_save_ == nullptr ? nullptr : kin_r_save_ + is * nrxx;
    }

    void upload_rho(const double* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rho_ != nullptr && host != nullptr)
        {
            syncmem_d_h2d_op()(rho_, host, size);
        }
#endif
    }

    void download_rho(double* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rho_ != nullptr && host != nullptr)
        {
            syncmem_d_d2h_op()(host, rho_, size);
        }
#endif
    }

    void upload_rhog(const std::complex<double>* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rhog_ != nullptr && host != nullptr)
        {
            syncmem_z_h2d_op()(rhog_, host, size);
        }
#endif
    }

    void download_rhog(std::complex<double>* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rhog_ != nullptr && host != nullptr)
        {
            syncmem_z_d2h_op()(host, rhog_, size);
        }
#endif
    }

    void upload_kin_r(const double* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (kin_r_ != nullptr && host != nullptr)
        {
            syncmem_d_h2d_op()(kin_r_, host, size);
        }
#endif
    }

    void download_kin_r(double* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (kin_r_ != nullptr && host != nullptr)
        {
            syncmem_d_d2h_op()(host, kin_r_, size);
        }
#endif
    }

    void upload_saved(const double* rho_host,
                      const std::complex<double>* rhog_host,
                      const double* kin_r_host,
                      const int rho_size,
                      const int rhog_size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rho_save_ != nullptr && rho_host != nullptr)
        {
            syncmem_d_h2d_op()(rho_save_, rho_host, rho_size);
        }
        if (rhog_save_ != nullptr && rhog_host != nullptr)
        {
            syncmem_z_h2d_op()(rhog_save_, rhog_host, rhog_size);
        }
        if (kin_r_save_ != nullptr && kin_r_host != nullptr)
        {
            syncmem_d_h2d_op()(kin_r_save_, kin_r_host, rho_size);
        }
#endif
    }

    void download_rhog_save(std::complex<double>* host, const int size) const
    {
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
        if (rhog_save_ != nullptr && host != nullptr)
        {
            syncmem_z_d2h_op()(host, rhog_save_, size);
        }
#endif
    }

  private:
    double* rho_ = nullptr;
    double* rho_save_ = nullptr;
    std::complex<double>* rhog_ = nullptr;
    std::complex<double>* rhog_save_ = nullptr;
    double* kin_r_ = nullptr;
    double* kin_r_save_ = nullptr;
};

void Charge::set_device(const std::string& device_in)
{
    if (device_in == device_)
    {
        return;
    }
    if (device_ == "gpu")
    {
        free_device_memory();
    }
    device_ = device_in;
    if (device_ == "gpu" && allocate_rho)
    {
        allocate_device_memory();
    }
}

void Charge::allocate_device_memory()
{
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
    if (device_ != "gpu")
    {
        return;
    }
    device_storage_ = new ChargeDeviceStorage;
    device_storage_->allocate(nspin, nrxx, ngmc, kin_r != nullptr);
#endif
}

void Charge::free_device_memory()
{
    delete device_storage_;
    device_storage_ = nullptr;
}

double* Charge::get_rho_d(const int is) const
{
    return device_ == "gpu" && device_storage_ != nullptr ? device_storage_->rho(is, nrxx) : nullptr;
}

double* Charge::get_rho_save_d(const int is) const
{
    return device_ == "gpu" && device_storage_ != nullptr ? device_storage_->rho_save(is, nrxx) : nullptr;
}

std::complex<double>* Charge::get_rhog_d(const int is) const
{
    return device_ == "gpu" && device_storage_ != nullptr ? device_storage_->rhog(is, ngmc) : nullptr;
}

std::complex<double>* Charge::get_rhog_save_d(const int is) const
{
    return device_ == "gpu" && device_storage_ != nullptr ? device_storage_->rhog_save(is, ngmc) : nullptr;
}

double* Charge::get_kin_r_d(const int is) const
{
    return device_ == "gpu" && device_storage_ != nullptr ? device_storage_->kin_r(is, nrxx) : nullptr;
}

double* Charge::get_kin_r_save_d(const int is) const
{
    return device_ == "gpu" && device_storage_ != nullptr ? device_storage_->kin_r_save(is, nrxx) : nullptr;
}

void Charge::sync_rho_to_device() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->upload_rho(_space_rho, nspin * nrxx);
    }
}

void Charge::sync_rho_to_host() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->download_rho(_space_rho, nspin * nrxx);
    }
}

void Charge::sync_rhog_to_device() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->upload_rhog(_space_rhog, nspin * ngmc);
    }
}

void Charge::sync_rhog_to_host() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->download_rhog(_space_rhog, nspin * ngmc);
    }
}

void Charge::sync_kin_r_to_device() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->upload_kin_r(_space_kin_r, nspin * nrxx);
    }
}

void Charge::sync_kin_r_to_host() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->download_kin_r(_space_kin_r, nspin * nrxx);
    }
}

void Charge::sync_saved_density_to_device() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->upload_saved(
            _space_rho_save, _space_rhog_save, _space_kin_r_save, nspin * nrxx, nspin * ngmc);
    }
}

void Charge::sync_rhog_save_to_host() const
{
    if (device_ == "gpu" && device_storage_ != nullptr)
    {
        device_storage_->download_rhog_save(_space_rhog_save, nspin * ngmc);
    }
}

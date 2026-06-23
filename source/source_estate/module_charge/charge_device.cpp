#include "charge.h"
#include "source_base/module_device/memory_op.h"

//==========================================================
// Device memory management implementation
// Following hybrid pattern like PW_Basis
//==========================================================

// Type aliases for memory operations
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
using resmem_d_gpu_op = base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>;
using resmem_z_gpu_op = base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>;
using delmem_d_gpu_op = base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>;
using delmem_z_gpu_op = base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>;
using syncmem_d_h2d_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using syncmem_d_d2h_op = base_device::memory::synchronize_memory_op<double, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
using syncmem_z_h2d_op = base_device::memory::synchronize_memory_op<std::complex<double>, base_device::DEVICE_GPU, base_device::DEVICE_CPU>;
using syncmem_z_d2h_op = base_device::memory::synchronize_memory_op<std::complex<double>, base_device::DEVICE_CPU, base_device::DEVICE_GPU>;
#endif

void Charge::set_device(const std::string& device_in)
{
    // No change needed if device is the same
    if (device_in == device_)
    {
        return;
    }

    // Free existing device memory if switching from GPU
    if (device_ == "gpu")
    {
        free_device_memory();
    }

    device_ = device_in;

    // Allocate device memory if switching to GPU and CPU memory is already allocated
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

    // Allocate rho arrays
    if (nspin > 0 && nrxx > 0)
    {
        resmem_d_gpu_op()(rho_d_, nspin * nrxx, "Charge::rho_d");
        resmem_d_gpu_op()(rho_save_d_, nspin * nrxx, "Charge::rho_save_d");
    }

    // Allocate rhog arrays
    if (nspin > 0 && ngmc > 0)
    {
        resmem_z_gpu_op()(rhog_d_, nspin * ngmc, "Charge::rhog_d");
        resmem_z_gpu_op()(rhog_save_d_, nspin * ngmc, "Charge::rhog_save_d");
    }

    // Allocate kin_r arrays if meta-GGA is used
    if (kin_r != nullptr && nspin > 0 && nrxx > 0)
    {
        resmem_d_gpu_op()(kin_r_d_, nspin * nrxx, "Charge::kin_r_d");
        resmem_d_gpu_op()(kin_r_save_d_, nspin * nrxx, "Charge::kin_r_save_d");
    }
#endif
}

void Charge::free_device_memory()
{
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
    if (rho_d_ != nullptr)
    {
        delmem_d_gpu_op()(rho_d_);
        rho_d_ = nullptr;
    }
    if (rho_save_d_ != nullptr)
    {
        delmem_d_gpu_op()(rho_save_d_);
        rho_save_d_ = nullptr;
    }
    if (rhog_d_ != nullptr)
    {
        delmem_z_gpu_op()(rhog_d_);
        rhog_d_ = nullptr;
    }
    if (rhog_save_d_ != nullptr)
    {
        delmem_z_gpu_op()(rhog_save_d_);
        rhog_save_d_ = nullptr;
    }
    if (kin_r_d_ != nullptr)
    {
        delmem_d_gpu_op()(kin_r_d_);
        kin_r_d_ = nullptr;
    }
    if (kin_r_save_d_ != nullptr)
    {
        delmem_d_gpu_op()(kin_r_save_d_);
        kin_r_save_d_ = nullptr;
    }
#endif
}

//==========================================================
// Device pointer accessors
//==========================================================

double* Charge::get_rho_d(int is) const
{
    if (device_ != "gpu" || rho_d_ == nullptr)
    {
        return nullptr;
    }
    return rho_d_ + is * nrxx;
}

double* Charge::get_rho_save_d(int is) const
{
    if (device_ != "gpu" || rho_save_d_ == nullptr)
    {
        return nullptr;
    }
    return rho_save_d_ + is * nrxx;
}

std::complex<double>* Charge::get_rhog_d(int is) const
{
    if (device_ != "gpu" || rhog_d_ == nullptr)
    {
        return nullptr;
    }
    return rhog_d_ + is * ngmc;
}

std::complex<double>* Charge::get_rhog_save_d(int is) const
{
    if (device_ != "gpu" || rhog_save_d_ == nullptr)
    {
        return nullptr;
    }
    return rhog_save_d_ + is * ngmc;
}

double* Charge::get_kin_r_d(int is) const
{
    if (device_ != "gpu" || kin_r_d_ == nullptr)
    {
        return nullptr;
    }
    return kin_r_d_ + is * nrxx;
}

double* Charge::get_kin_r_save_d(int is) const
{
    if (device_ != "gpu" || kin_r_save_d_ == nullptr)
    {
        return nullptr;
    }
    return kin_r_save_d_ + is * nrxx;
}

//==========================================================
// GPU template specializations for sync methods
//==========================================================

#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

template <>
void Charge::sync_rho_to_device<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rho_d_ == nullptr || _space_rho == nullptr)
    {
        return;
    }
    syncmem_d_h2d_op()(rho_d_, _space_rho, nspin * nrxx);
}

template <>
void Charge::sync_rho_to_host<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rho_d_ == nullptr || _space_rho == nullptr)
    {
        return;
    }
    syncmem_d_d2h_op()(_space_rho, rho_d_, nspin * nrxx);
}

template <>
void Charge::sync_rhog_to_device<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rhog_d_ == nullptr || _space_rhog == nullptr)
    {
        return;
    }
    syncmem_z_h2d_op()(rhog_d_, _space_rhog, nspin * ngmc);
}

template <>
void Charge::sync_rhog_to_host<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rhog_d_ == nullptr || _space_rhog == nullptr)
    {
        return;
    }
    syncmem_z_d2h_op()(_space_rhog, rhog_d_, nspin * ngmc);
}

template <>
void Charge::sync_kin_r_to_device<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || kin_r_d_ == nullptr || _space_kin_r == nullptr)
    {
        return;
    }
    syncmem_d_h2d_op()(kin_r_d_, _space_kin_r, nspin * nrxx);
}

template <>
void Charge::sync_kin_r_to_host<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || kin_r_d_ == nullptr || _space_kin_r == nullptr)
    {
        return;
    }
    syncmem_d_d2h_op()(_space_kin_r, kin_r_d_, nspin * nrxx);
}

template <>
void Charge::sync_kin_r_save_to_device<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || kin_r_save_d_ == nullptr || _space_kin_r_save == nullptr)
    {
        return;
    }
    syncmem_d_h2d_op()(kin_r_save_d_, _space_kin_r_save, nspin * nrxx);
}

template <>
void Charge::sync_rho_save_to_device<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rho_save_d_ == nullptr || _space_rho_save == nullptr)
    {
        return;
    }
    syncmem_d_h2d_op()(rho_save_d_, _space_rho_save, nspin * nrxx);
}

template <>
void Charge::sync_rhog_save_to_device<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rhog_save_d_ == nullptr || _space_rhog_save == nullptr)
    {
        return;
    }
    syncmem_z_h2d_op()(rhog_save_d_, _space_rhog_save, nspin * ngmc);
}

template <>
void Charge::sync_rhog_save_to_host<base_device::DEVICE_GPU>()
{
    if (device_ != "gpu" || rhog_save_d_ == nullptr || _space_rhog_save == nullptr)
    {
        return;
    }
    syncmem_z_d2h_op()(_space_rhog_save, rhog_save_d_, nspin * ngmc);
}

#endif // __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM

void Charge::sync_realspace_density_to_device()
{
#if __CUDA || __UT_USE_CUDA || __ROCM || __UT_USE_ROCM
    if (device_ != "gpu")
    {
        return;
    }
    this->sync_rho_to_device<base_device::DEVICE_GPU>();
    if (kin_r_d_ != nullptr)
    {
        this->sync_kin_r_to_device<base_device::DEVICE_GPU>();
    }
#endif
}

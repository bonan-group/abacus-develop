#include "source_base/memory.h"
#include "source_base/timer.h"
#include "source_pw/module_pwdft/global.h"
#include "source_pw/module_pwdft/kernels/wf_op.h"
#include "source_base/module_device/device.h"
#include "structure_factor.h"
std::complex<double>* Structure_Factor::get_sk(const int ik,
                                               const int it,
                                               const int ia,
                                               const ModulePW::PW_Basis_K* wfc_basis) const
{
    ModuleBase::timer::tick("Structure_Factor", "get_sk");
    const double arg = (wfc_basis->kvec_c[ik] * ucell->atoms[it].tau[ia]) * ModuleBase::TWO_PI;
    const std::complex<double> kphase = std::complex<double>(cos(arg), -sin(arg));
    const int npw = wfc_basis->npwk[ik];
    std::complex<double> *sk = new std::complex<double>[npw];
    const int nx = wfc_basis->nx, ny = wfc_basis->ny, nz = wfc_basis->nz;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int igl = 0; igl < npw; ++igl)
    {
        const int isz = wfc_basis->getigl2isz(ik, igl);
        int iz = isz % nz;
        const int is = isz / nz;
        const int ixy = wfc_basis->is2fftixy[is];
        int ix = ixy / wfc_basis->fftny;
        int iy = ixy % wfc_basis->fftny;
        if (ix >= int(nx / 2) + 1) 
        {
            ix -= nx;
        }
        if (iy >= int(ny / 2) + 1) 
        {
            iy -= ny;
        }
        if (iz >= int(nz / 2) + 1) 
        {
            iz -= nz;
        }
        ix += this->rho_basis->nx;
        iy += this->rho_basis->ny;
        iz += this->rho_basis->nz;
        const int iat = ucell->itia2iat(it, ia);
        sk[igl] = kphase * this->eigts1(iat, ix) * this->eigts2(iat, iy) * this->eigts3(iat, iz);
    }
    ModuleBase::timer::tick("Structure_Factor", "get_sk");
    return sk;
}

template <typename FPTYPE, typename Device>
void Structure_Factor::get_sk(Device* ctx,
                              const int ik,
                              const ModulePW::PW_Basis_K* wfc_basis,
                              std::complex<FPTYPE>* sk) const
{
    ModuleBase::timer::tick("Structure_Factor", "get_sk");

    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::AbacusDevice_t device = base_device::get_device_type<Device>(ctx);
    using cal_sk_op = hamilt::cal_sk_op<FPTYPE, Device>;
    using resmem_int_op = base_device::memory::resize_memory_op<int, Device>;
    using delmem_int_op = base_device::memory::delete_memory_op<int, Device>;
    using syncmem_int_op = base_device::memory::synchronize_memory_op<int, Device, base_device::DEVICE_CPU>;

    using resmem_var_op = base_device::memory::resize_memory_op<FPTYPE, Device>;
    using delmem_var_op = base_device::memory::delete_memory_op<FPTYPE, Device>;
    using syncmem_var_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;

    int iat = 0, _npw = wfc_basis->npwk[ik], eigts1_nc = this->eigts1.nc, eigts2_nc = this->eigts2.nc,
            eigts3_nc = this->eigts3.nc;
    int *igl2isz = nullptr, *is2fftixy = nullptr, *atom_na = nullptr, *h_atom_na = new int[ucell->ntype];
    FPTYPE *atom_tau = nullptr, *h_atom_tau = new FPTYPE[ucell->nat * 3], *kvec = wfc_basis->get_kvec_c_data<FPTYPE>();
    std::complex<FPTYPE> *eigts1 = this->get_eigts1_data<FPTYPE>(), *eigts2 = this->get_eigts2_data<FPTYPE>(),
            *eigts3 = this->get_eigts3_data<FPTYPE>();
    for (int it = 0; it < ucell->ntype; it++)
    {
        h_atom_na[it] = ucell->atoms[it].na;
    }
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int iat = 0; iat < ucell->nat; iat++)
    {
        int it = ucell->iat2it[iat];
        int ia = ucell->iat2ia[iat];
        auto *tau = reinterpret_cast<double *>(ucell->atoms[it].tau.data());
        h_atom_tau[iat * 3 + 0] = static_cast<FPTYPE>(tau[ia * 3 + 0]);
        h_atom_tau[iat * 3 + 1] = static_cast<FPTYPE>(tau[ia * 3 + 1]);
        h_atom_tau[iat * 3 + 2] = static_cast<FPTYPE>(tau[ia * 3 + 2]);
    }
    if (device == base_device::GpuDevice)
    {
        resmem_int_op()(atom_na, ucell->ntype);
        syncmem_int_op()(atom_na, h_atom_na, ucell->ntype);

        resmem_var_op()(atom_tau, ucell->nat * 3);
        syncmem_var_op()(atom_tau, h_atom_tau, ucell->nat * 3);

        igl2isz = wfc_basis->d_igl2isz_k;
        is2fftixy = wfc_basis->d_is2fftixy;
    }
    else
    {
        atom_na = h_atom_na;
        atom_tau = h_atom_tau;
        igl2isz = wfc_basis->igl2isz_k;
        is2fftixy = wfc_basis->is2fftixy;
    }

    cal_sk_op()(ctx,
                ik,
                ucell->ntype,
                wfc_basis->nx,
                wfc_basis->ny,
                wfc_basis->nz,
                this->rho_basis->nx,
                this->rho_basis->ny,
                this->rho_basis->nz,
                _npw,
                wfc_basis->npwk_max,
                wfc_basis->fftny,
                eigts1_nc,
                eigts2_nc,
                eigts3_nc,
                atom_na,
                igl2isz,
                is2fftixy,
                ModuleBase::TWO_PI,
                kvec,
                atom_tau,
                eigts1,
                eigts2,
                eigts3,
                sk);
    if (device == base_device::GpuDevice)
    {
        delmem_int_op()(atom_na);
        delmem_var_op()(atom_tau);
    }
    delete[] h_atom_na;
    delete[] h_atom_tau;
    ModuleBase::timer::tick("Structure_Factor", "get_sk");
}

std::complex<double>* Structure_Factor::get_skq(int ik,
                                                const int it,
                                                const int ia,
                                                const ModulePW::PW_Basis_K* wfc_basis,
                                                ModuleBase::Vector3<double> q) // pengfei 2016-11-23
{
    const int npw = wfc_basis->npwk[ik];
    std::complex<double> *skq = new std::complex<double>[npw];

    for (int ig = 0; ig < npw; ig++)
    {
        ModuleBase::Vector3<double> qkq = wfc_basis->getgpluskcar(ik, ig) + q;
        double arg = (qkq * ucell->atoms[it].tau[ia]) * ModuleBase::TWO_PI;
        skq[ig] = std::complex<double>(cos(arg), -sin(arg));
    }

    return skq;
}

template void Structure_Factor::get_sk<float, base_device::DEVICE_CPU>(base_device::DEVICE_CPU*,
                                                                       int,
                                                                       const ModulePW::PW_Basis_K*,
                                                                       std::complex<float>*) const;
template void Structure_Factor::get_sk<double, base_device::DEVICE_CPU>(base_device::DEVICE_CPU*,
                                                                        int,
                                                                        const ModulePW::PW_Basis_K*,
                                                                        std::complex<double>*) const;
#if defined(__CUDA) || defined(__ROCM)
template void Structure_Factor::get_sk<float, base_device::DEVICE_GPU>(base_device::DEVICE_GPU*,
                                                                       int,
                                                                       const ModulePW::PW_Basis_K*,
                                                                       std::complex<float>*) const;
template void Structure_Factor::get_sk<double, base_device::DEVICE_GPU>(base_device::DEVICE_GPU*,
                                                                        int,
                                                                        const ModulePW::PW_Basis_K*,
                                                                        std::complex<double>*) const;
#endif

/// @brief Compute structure factor for a single atom
/// This version computes sk for a single atom, useful for chunked processing
/// to avoid computing sk for all atoms when only a subset is needed.
template <typename FPTYPE, typename Device>
void Structure_Factor::get_sk(Device* ctx,
                              const int ik,
                              const int iat,
                              const ModulePW::PW_Basis_K* wfc_basis,
                              std::complex<FPTYPE>* sk) const
{
    ModuleBase::timer::tick("Structure_Factor", "get_sk_single");

    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::AbacusDevice_t device = base_device::get_device_type<Device>(ctx);
    using resmem_var_op = base_device::memory::resize_memory_op<FPTYPE, Device>;
    using delmem_var_op = base_device::memory::delete_memory_op<FPTYPE, Device>;
    using syncmem_var_op = base_device::memory::synchronize_memory_op<FPTYPE, Device, base_device::DEVICE_CPU>;
    using syncmem_complex_d2h_op = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>;
    using syncmem_complex_h2d_op = base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, Device, base_device::DEVICE_CPU>;

    const int _npw = wfc_basis->npwk[ik];
    const int nx = wfc_basis->nx, ny = wfc_basis->ny, nz = wfc_basis->nz;
    const int it = ucell->iat2it[iat];
    const int ia = ucell->iat2ia[iat];

    // Get tau for this atom
    FPTYPE h_atom_tau[3];
    auto* tau = reinterpret_cast<double*>(ucell->atoms[it].tau.data());
    h_atom_tau[0] = static_cast<FPTYPE>(tau[ia * 3 + 0]);
    h_atom_tau[1] = static_cast<FPTYPE>(tau[ia * 3 + 1]);
    h_atom_tau[2] = static_cast<FPTYPE>(tau[ia * 3 + 2]);

    // Compute kphase = e^{-i k·tau}
    FPTYPE* kvec = wfc_basis->get_kvec_c_data<FPTYPE>();
    // For GPU, kvec is already on device; for CPU, we need to get the host pointer
    FPTYPE kvec_host[3];
    if (device == base_device::GpuDevice)
    {
        // Copy kvec from device to host for kphase computation
        base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, Device>()(
            kvec_host, kvec + ik * 3, 3);
    }
    else
    {
        kvec_host[0] = kvec[ik * 3 + 0];
        kvec_host[1] = kvec[ik * 3 + 1];
        kvec_host[2] = kvec[ik * 3 + 2];
    }

    const FPTYPE arg = (kvec_host[0] * h_atom_tau[0] +
                        kvec_host[1] * h_atom_tau[1] +
                        kvec_host[2] * h_atom_tau[2]) * static_cast<FPTYPE>(ModuleBase::TWO_PI);
    const std::complex<FPTYPE> kphase = std::complex<FPTYPE>(cos(arg), -sin(arg));

    // Get eigts pointers
    std::complex<FPTYPE>* eigts1 = this->get_eigts1_data<FPTYPE>();
    std::complex<FPTYPE>* eigts2 = this->get_eigts2_data<FPTYPE>();
    std::complex<FPTYPE>* eigts3 = this->get_eigts3_data<FPTYPE>();
    const int eigts1_nc = this->eigts1.nc;
    const int eigts2_nc = this->eigts2.nc;
    const int eigts3_nc = this->eigts3.nc;

    if (device == base_device::GpuDevice)
    {
        // For GPU, we need to compute on device
        // Allocate temporary buffer on host, compute, then copy to device
        // This is suboptimal but simple - a proper GPU kernel should be added later
        std::complex<FPTYPE>* sk_host = new std::complex<FPTYPE>[_npw];

        // We need eigts on host for this computation
        std::complex<FPTYPE>* eigts1_host = new std::complex<FPTYPE>[this->eigts1.nr * eigts1_nc];
        std::complex<FPTYPE>* eigts2_host = new std::complex<FPTYPE>[this->eigts2.nr * eigts2_nc];
        std::complex<FPTYPE>* eigts3_host = new std::complex<FPTYPE>[this->eigts3.nr * eigts3_nc];

        base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>()(
            eigts1_host, eigts1, this->eigts1.nr * eigts1_nc);
        base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>()(
            eigts2_host, eigts2, this->eigts2.nr * eigts2_nc);
        base_device::memory::synchronize_memory_op<std::complex<FPTYPE>, base_device::DEVICE_CPU, Device>()(
            eigts3_host, eigts3, this->eigts3.nr * eigts3_nc);

        const int* igl2isz = wfc_basis->igl2isz_k;  // Use host version
        const int* is2fftixy = wfc_basis->is2fftixy;  // Use host version

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int igl = 0; igl < _npw; ++igl)
        {
            const int isz = igl2isz[ik * wfc_basis->npwk_max + igl];
            int iz = isz % nz;
            const int is = isz / nz;
            const int ixy = is2fftixy[is];
            int ix = ixy / wfc_basis->fftny;
            int iy = ixy % wfc_basis->fftny;
            if (ix >= int(nx / 2) + 1) { ix -= nx; }
            if (iy >= int(ny / 2) + 1) { iy -= ny; }
            if (iz >= int(nz / 2) + 1) { iz -= nz; }
            ix += this->rho_basis->nx;
            iy += this->rho_basis->ny;
            iz += this->rho_basis->nz;
            sk_host[igl] = kphase * eigts1_host[iat * eigts1_nc + ix]
                                  * eigts2_host[iat * eigts2_nc + iy]
                                  * eigts3_host[iat * eigts3_nc + iz];
        }

        // Copy result to device
        syncmem_complex_h2d_op()(sk, sk_host, _npw);

        delete[] sk_host;
        delete[] eigts1_host;
        delete[] eigts2_host;
        delete[] eigts3_host;
    }
    else
    {
        // CPU computation
        const int* igl2isz = wfc_basis->igl2isz_k;
        const int* is2fftixy = wfc_basis->is2fftixy;

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int igl = 0; igl < _npw; ++igl)
        {
            const int isz = igl2isz[ik * wfc_basis->npwk_max + igl];
            int iz = isz % nz;
            const int is = isz / nz;
            const int ixy = is2fftixy[is];
            int ix = ixy / wfc_basis->fftny;
            int iy = ixy % wfc_basis->fftny;
            if (ix >= int(nx / 2) + 1) { ix -= nx; }
            if (iy >= int(ny / 2) + 1) { iy -= ny; }
            if (iz >= int(nz / 2) + 1) { iz -= nz; }
            ix += this->rho_basis->nx;
            iy += this->rho_basis->ny;
            iz += this->rho_basis->nz;
            sk[igl] = kphase * eigts1[iat * eigts1_nc + ix]
                            * eigts2[iat * eigts2_nc + iy]
                            * eigts3[iat * eigts3_nc + iz];
        }
    }

    ModuleBase::timer::tick("Structure_Factor", "get_sk_single");
}

template void Structure_Factor::get_sk<float, base_device::DEVICE_CPU>(base_device::DEVICE_CPU*,
                                                                        int,
                                                                        int,
                                                                        const ModulePW::PW_Basis_K*,
                                                                        std::complex<float>*) const;
template void Structure_Factor::get_sk<double, base_device::DEVICE_CPU>(base_device::DEVICE_CPU*,
                                                                         int,
                                                                         int,
                                                                         const ModulePW::PW_Basis_K*,
                                                                         std::complex<double>*) const;
#if defined(__CUDA) || defined(__ROCM)
template void Structure_Factor::get_sk<float, base_device::DEVICE_GPU>(base_device::DEVICE_GPU*,
                                                                        int,
                                                                        int,
                                                                        const ModulePW::PW_Basis_K*,
                                                                        std::complex<float>*) const;
template void Structure_Factor::get_sk<double, base_device::DEVICE_GPU>(base_device::DEVICE_GPU*,
                                                                         int,
                                                                         int,
                                                                         const ModulePW::PW_Basis_K*,
                                                                         std::complex<double>*) const;
#endif
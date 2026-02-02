#include "source_base/timer.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "pw_basis_k.h"
#include "pw_gatherscatter.h"

#include <cassert>
#include <complex>

namespace ModulePW
{

/**
 * @brief transform real space to reciprocal space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform f'(r) to c(k,g).
 * @param in: (nplane,ny,nx), std::complex<double> data
 * @param out: (nz, ns),  std::complex<double> data
 */
template <typename FPTYPE>
void PW_Basis_K::real2recip(const std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");

    assert(this->gamma_only == false);
    auto* auxr = this->fft_bundle.get_auxr_data<FPTYPE>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ir = 0; ir < this->nrxx; ++ir)
    {
        auxr[ir] = in[ir];
    }
    this->fft_bundle.fftxyfor(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());

    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
    if (add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] += tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] = tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
}

/**
 * @brief transform real space to reciprocal space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform f'(r) to c(k,g).
 * @param in: (nplane,ny,nx), double data
 * @param out: (nz, ns),  std::complex<double> data
 */
template <typename FPTYPE>
void PW_Basis_K::real2recip(const FPTYPE* in,
                            std::complex<FPTYPE>* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real2recip");
    assert(this->gamma_only == true);
    // for(int ir = 0 ; ir < this->nrxx ; ++ir)
    // {
    //     this->fft_bundle.get_rspace_data<FPTYPE>()[ir] = in[ir];
    // }
    // r2c in place
    const int npy = this->ny * this->nplane;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int ix = 0; ix < this->nx; ++ix)
    {
        for (int ipy = 0; ipy < npy; ++ipy)
        {
            this->fft_bundle.get_rspace_data<FPTYPE>()[ix * npy + ipy] = in[ix * npy + ipy];
        }
    }

    this->fft_bundle.fftxyr2c(fft_bundle.get_rspace_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());

    this->gatherp_scatters(this->fft_bundle.get_auxr_data<FPTYPE>(), this->fft_bundle.get_auxg_data<FPTYPE>());

    this->fft_bundle.fftzfor(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
    if (add)
    {
        FPTYPE tmpfac = factor / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] += tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    else
    {
        FPTYPE tmpfac = 1.0 / FPTYPE(this->nxyz);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int igl = 0; igl < npwk; ++igl)
        {
            out[igl] = tmpfac * auxg[this->igl2isz_k[igl + startig]];
        }
    }
    ModuleBase::timer::tick(this->classname, "real2recip");
    return;
}

/**
 * @brief transform reciprocal space to real space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform c(k,g) to f'(r).
 * @param in: (nz,ns), std::complex<double>
 * @param out: (nplane, ny, nx), std::complex<double>
 */
template <typename FPTYPE>
void PW_Basis_K::recip2real(const std::complex<FPTYPE>* in,
                            std::complex<FPTYPE>* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip2real");
    assert(this->gamma_only == false);
    ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxg_data<FPTYPE>(), this->nst * this->nz);

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int igl = 0; igl < npwk; ++igl)
    {
        auxg[this->igl2isz_k[igl + startig]] = in[igl];
    }
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    this->fft_bundle.fftxybac(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_auxr_data<FPTYPE>());
    auto* auxr = this->fft_bundle.get_auxr_data<FPTYPE>();
    if (add)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            out[ir] += factor * auxr[ir];
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int ir = 0; ir < this->nrxx; ++ir)
        {
            out[ir] = auxr[ir];
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

/**
 * @brief transform reciprocal space to real space
 * @details real wave function f(k,r):
 *          f(k,r)=1/V*\sum_{g} c(k,g)*exp(i(g+k)*r) \equiv exp(ikr)f'(k.r)
 *          c(k,g)=\int dr*f(k,r)*exp(-i(g+k)*r)
 *          However, we use f'(k,r)!!! :
 *          f'(k,r)=1/V*\sum_{g} c(k,g)*exp(ig*r)
 *          c(k,g)=\int dr*f'(k,r)*exp(-ig*r)
 *
 *          This function tranform c(k,g) to f'(r).
 * @param in: (nz,ns), std::complex<double>
 * @param out: (nplane, ny, nx), double
 */
template <typename FPTYPE>
void PW_Basis_K::recip2real(const std::complex<FPTYPE>* in,
                            FPTYPE* out,
                            const int ik,
                            const bool add,
                            const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip2real");
    assert(this->gamma_only == true);
    ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxg_data<FPTYPE>(), this->nst * this->nz);

    const int startig = ik * this->npwk_max;
    const int npwk = this->npwk[ik];
    auto* auxg = this->fft_bundle.get_auxg_data<FPTYPE>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int igl = 0; igl < npwk; ++igl)
    {
        auxg[this->igl2isz_k[igl + startig]] = in[igl];
    }
    this->fft_bundle.fftzbac(fft_bundle.get_auxg_data<FPTYPE>(), fft_bundle.get_auxg_data<FPTYPE>());

    this->gathers_scatterp(this->fft_bundle.get_auxg_data<FPTYPE>(), this->fft_bundle.get_auxr_data<FPTYPE>());

    this->fft_bundle.fftxyc2r(fft_bundle.get_auxr_data<FPTYPE>(), fft_bundle.get_rspace_data<FPTYPE>());

    // for(int ir = 0 ; ir < this->nrxx ; ++ir)
    // {
    //     out[ir] = this->fft_bundle.get_rspace_data<FPTYPE>()[ir] / this->nxyz;
    // }

    // r2c in place
    const int npy = this->ny * this->nplane;
    auto* rspace = this->fft_bundle.get_rspace_data<FPTYPE>();
    if (add)
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int ix = 0; ix < this->nx; ++ix)
        {
            for (int ipy = 0; ipy < npy; ++ipy)
            {
                out[ix * npy + ipy] += factor * rspace[ix * npy + ipy];
            }
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int ix = 0; ix < this->nx; ++ix)
        {
            for (int ipy = 0; ipy < npy; ++ipy)
            {
                out[ix * npy + ipy] = rspace[ix * npy + ipy];
            }
        }
    }
    ModuleBase::timer::tick(this->classname, "recip2real");
}

template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    this->real2recip(in, out, ik, add, factor);
}
template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    #if defined(__DSP)
        this->real2recip_dsp(in,out,ik,add,factor);
    #else
        this->real2recip(in, out, ik, add, factor);
    #endif
}

template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    this->recip2real(in, out, ik, add, factor);
}
template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_CPU* /*dev*/,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    #if defined(__DSP)
        this->recip2real_dsp(in,out,ik,add,factor);
    #else
        this->recip2real(in, out, ik, add, factor);
    #endif
}

#if (defined(__CUDA) || defined(__ROCM))
template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_GPU* ctx,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    base_device::memory::synchronize_memory_op<std::complex<float>, base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<float>(),
        in,
        this->nrxx);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<float>(), this->fft_bundle.get_auxr_3d_data<float>());

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<float, base_device::DEVICE_GPU>()(npw_k,
                                                                  this->nxyz,
                                                                  add,
                                                                  factor,
                                                                  this->ig2ixyz_k + startig,
                                                                  this->fft_bundle.get_auxr_3d_data<float>(),
                                                                  out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}
template <>
void PW_Basis_K::real_to_recip(const base_device::DEVICE_GPU* ctx,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    base_device::memory::synchronize_memory_op<std::complex<double>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<double>(),
                                                                          in,
                                                                          this->nrxx);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<double>(), this->fft_bundle.get_auxr_3d_data<double>());

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<double, base_device::DEVICE_GPU>()(npw_k,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_k + startig,
                                                                   this->fft_bundle.get_auxr_3d_data<double>(),
                                                                   out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}

template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_GPU* ctx,
                               const std::complex<float>* in,
                               std::complex<float>* out,
                               const int ik,
                               const bool add,
                               const float factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<float>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<float>(),
        0,
        this->nxyz);

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<float, base_device::DEVICE_GPU>()(npw_k,
                                                        this->ig2ixyz_k + startig,
                                                        in,
                                                        this->fft_bundle.get_auxr_3d_data<float>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<float>(), this->fft_bundle.get_auxr_3d_data<float>());

    set_recip_to_real_output_op<float, base_device::DEVICE_GPU>()(this->nrxx,
                                                                  add,
                                                                  factor,
                                                                  this->fft_bundle.get_auxr_3d_data<float>(),
                                                                  out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}
template <>
void PW_Basis_K::recip_to_real(const base_device::DEVICE_GPU* ctx,
                               const std::complex<double>* in,
                               std::complex<double>* out,
                               const int ik,
                               const bool add,
                               const double factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<double>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<double>(),
        0,
        this->nxyz);

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<double, base_device::DEVICE_GPU>()(npw_k,
                                                         this->ig2ixyz_k + startig,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<double>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<double>(), this->fft_bundle.get_auxr_3d_data<double>());

    set_recip_to_real_output_op<double, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<double>(),
                                                                   out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

template <typename FPTYPE>
void PW_Basis_K::real2recip_gpu(const std::complex<FPTYPE>* in,
                               std::complex<FPTYPE>* out,
                               const int ik,
                               const bool add,
                               const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                          in,
                                                                          this->nrxx);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(), this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];
    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw_k,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_k + startig,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}
template <typename FPTYPE>
void PW_Basis_K::recip2real_gpu(const std::complex<FPTYPE>* in,
                               std::complex<FPTYPE>* out,
                               const int ik,
                               const bool add,
                               const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<FPTYPE>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);

    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw_k,
                                                         this->ig2ixyz_k + startig,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(), this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

template void PW_Basis_K::real2recip_gpu<float>(const std::complex<float>*,
                                                std::complex<float>*,
                                                const int,
                                                const bool,
                                                const float) const;

template void PW_Basis_K::real2recip_gpu<double>(const std::complex<double>*,
                                                 std::complex<double>*,
                                                 const int,
                                                 const bool,
                                                 const double) const;

template void PW_Basis_K::recip2real_gpu<float>(const std::complex<float>*,
                                                std::complex<float>*,
                                                const int,
                                                const bool,
                                                const float) const;

template void PW_Basis_K::recip2real_gpu<double>(const std::complex<double>*,
                                                 std::complex<double>*,
                                                 const int,
                                                 const bool,
                                                 const double) const;

#endif

template void PW_Basis_K::real2recip<float>(const float* in,
                                            std::complex<float>* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::real2recip<float>(const std::complex<float>* in,
                                            std::complex<float>* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::recip2real<float>(const std::complex<float>* in,
                                            float* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis_K::recip2real<float>(const std::complex<float>* in,
                                            std::complex<float>* out,
                                            const int ik,
                                            const bool add,
                                            const float factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)

template void PW_Basis_K::real2recip<double>(const double* in,
                                             std::complex<double>* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::real2recip<double>(const std::complex<double>* in,
                                             std::complex<double>* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nplane,nx*ny)  ; out(nz, ns)
template void PW_Basis_K::recip2real<double>(const std::complex<double>* in,
                                             double* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)
template void PW_Basis_K::recip2real<double>(const std::complex<double>* in,
                                             std::complex<double>* out,
                                             const int ik,
                                             const bool add,
                                             const double factor) const; // in:(nz, ns)  ; out(nplane,nx*ny)

// ============================================================================
// Batch Transform Implementations
// ============================================================================

#if defined(__CUDA) || defined(__ROCM)

template <typename FPTYPE, typename Device>
void PW_Basis_K::real_to_recip_batch(const Device* ctx,
                                     const std::complex<FPTYPE>* in_batch,
                                     std::complex<FPTYPE>* out_batch,
                                     const int ik,
                                     int batch_count,
                                     const bool add,
                                     const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip_batch gpu");

    // Check if batch FFT is available
    if (!this->fft_bundle.is_batch_fft_available<FPTYPE>())
    {
        // Fallback to sequential transforms using GPU-specific template overload
        for (int ib = 0; ib < batch_count; ++ib)
        {
            this->real_to_recip<std::complex<FPTYPE>, Device>(
                               in_batch + ib * this->nrxx,
                               out_batch + ib * this->npwk_max,
                               ik,
                               add,
                               factor);
        }
        ModuleBase::timer::tick(this->classname, "real_to_recip_batch gpu");
        return;
    }

    // Batch FFT path
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    // Get batch output buffer from FFT bundle
    std::complex<FPTYPE>* batch_out = this->fft_bundle.get_batch_output_buffer<FPTYPE>();
    const int nxyz = this->nxyz;

    // Perform batch forward FFT directly on input (out-of-place FFT doesn't modify input)
    // const_cast is safe because cuFFT out-of-place C2C transform only reads from input
    this->fft_bundle.fft3D_forward_batch(const_cast<std::complex<FPTYPE>*>(in_batch), batch_out, batch_count);

    // Extract results using batch operator (all batch elements share same ik)
    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    // Create factor vector
    std::vector<FPTYPE> factor_vec(batch_count, factor);

    // Allocate and copy to GPU
    FPTYPE* d_factors = nullptr;
    base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(d_factors, batch_count);
    base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        d_factors, factor_vec.data(), batch_count);

    // Call operator_batch once
    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>().operator_batch(
        npw_k,
        nxyz,
        this->npwk_max,     // out_stride: matches output buffer layout
        batch_count,
        add,
        d_factors,
        this->ig2ixyz_k + startig,
        batch_out,
        out_batch);

    // Free memory
    base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(d_factors);

    ModuleBase::timer::tick(this->classname, "real_to_recip_batch gpu");
}

template <typename FPTYPE, typename Device>
void PW_Basis_K::recip_to_real_batch(const Device* ctx,
                                     const std::complex<FPTYPE>* in_batch,
                                     std::complex<FPTYPE>* out_batch,
                                     const int ik,
                                     int batch_count,
                                     const bool add,
                                     const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real_batch gpu");

    // Check if batch FFT is available
    if (!this->fft_bundle.is_batch_fft_available<FPTYPE>())
    {
        // Fallback to sequential transforms using GPU-specific template overload
        for (int ib = 0; ib < batch_count; ++ib)
        {
            this->recip_to_real<std::complex<FPTYPE>, Device>(
                               in_batch + ib * this->npwk_max,
                               out_batch + ib * this->nrxx,
                               ik,
                               add,
                               factor);
        }
        ModuleBase::timer::tick(this->classname, "recip_to_real_batch gpu");
        return;
    }

    // Batch FFT path
    assert(this->gamma_only == false);
    assert(this->poolnproc == 1);

    // Get batch buffers from FFT bundle
    std::complex<FPTYPE>* batch_in = this->fft_bundle.get_batch_input_buffer<FPTYPE>();
    std::complex<FPTYPE>* batch_out = this->fft_bundle.get_batch_output_buffer<FPTYPE>();
    const int nxyz = this->nxyz;

    // Zero batch input buffer
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        batch_in,
        0,
        batch_count * nxyz);

    // Populate FFT input for all batch elements (all share same ik)
    const int startig = ik * this->npwk_max;
    const int npw_k = this->npwk[ik];

    // Use batched kernel with separate count (npw_k) and stride (npwk_max)
    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>().operator_batch(
        npw_k,              // count: actual plane waves for this k-point
        nxyz,
        this->npwk_max,     // in_stride: matches input data layout
        this->ig2ixyz_k + startig,
        batch_count,
        in_batch,
        batch_in);

    // Perform batch backward FFT
    this->fft_bundle.fft3D_backward_batch(batch_in, batch_out, batch_count);

    // Extract results using batch operator
    // Create factor vector
    std::vector<FPTYPE> factor_vec(batch_count, factor);

    // Allocate and copy to GPU
    FPTYPE* d_factors = nullptr;
    base_device::memory::resize_memory_op<FPTYPE, base_device::DEVICE_GPU>()(d_factors, batch_count);
    base_device::memory::synchronize_memory_op<FPTYPE, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(
        d_factors, factor_vec.data(), batch_count);

    // Call batch operator
    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>().operator_batch(
        this->nrxx,
        nxyz,
        batch_count,
        add,
        d_factors,
        batch_out,
        out_batch);

    // Free memory
    base_device::memory::delete_memory_op<FPTYPE, base_device::DEVICE_GPU>()(d_factors);

    ModuleBase::timer::tick(this->classname, "recip_to_real_batch gpu");
}

// Template instantiations for batch transforms
template void PW_Basis_K::real_to_recip_batch<float, base_device::DEVICE_GPU>(
    const base_device::DEVICE_GPU*,
    const std::complex<float>*,
    std::complex<float>*,
    const int,
    int,
    const bool,
    const float) const;

template void PW_Basis_K::real_to_recip_batch<double, base_device::DEVICE_GPU>(
    const base_device::DEVICE_GPU*,
    const std::complex<double>*,
    std::complex<double>*,
    const int,
    int,
    const bool,
    const double) const;

template void PW_Basis_K::recip_to_real_batch<float, base_device::DEVICE_GPU>(
    const base_device::DEVICE_GPU*,
    const std::complex<float>*,
    std::complex<float>*,
    const int,
    int,
    const bool,
    const float) const;

template void PW_Basis_K::recip_to_real_batch<double, base_device::DEVICE_GPU>(
    const base_device::DEVICE_GPU*,
    const std::complex<double>*,
    std::complex<double>*,
    const int,
    int,
    const bool,
    const double) const;

#else
// CPU fallback (not implemented - just use sequential)
template <typename FPTYPE, typename Device>
void PW_Basis_K::real_to_recip_batch(const Device* ctx,
                                     const std::complex<FPTYPE>* in_batch,
                                     std::complex<FPTYPE>* out_batch,
                                     const int ik,
                                     int batch_count,
                                     const bool add,
                                     const FPTYPE factor) const
{
    // Fallback to sequential transforms on CPU
    for (int ib = 0; ib < batch_count; ++ib)
    {
        this->real_to_recip(ctx,
                           in_batch + ib * this->nrxx,
                           out_batch + ib * this->npwk_max,
                           ik,
                           add,
                           factor);
    }
}

template <typename FPTYPE, typename Device>
void PW_Basis_K::recip_to_real_batch(const Device* ctx,
                                     const std::complex<FPTYPE>* in_batch,
                                     std::complex<FPTYPE>* out_batch,
                                     const int ik,
                                     int batch_count,
                                     const bool add,
                                     const FPTYPE factor) const
{
    // Fallback to sequential transforms on CPU
    for (int ib = 0; ib < batch_count; ++ib)
    {
        this->recip_to_real(ctx,
                           in_batch + ib * this->npwk_max,
                           out_batch + ib * this->nrxx,
                           ik,
                           add,
                           factor);
    }
}
#endif

// Explicit template instantiations for CPU batch transforms
template void PW_Basis_K::real_to_recip_batch<float, base_device::DEVICE_CPU>(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<float>* in_batch,
    std::complex<float>* out_batch,
    const int ik,
    int batch_count,
    const bool add,
    const float factor) const;

template void PW_Basis_K::real_to_recip_batch<double, base_device::DEVICE_CPU>(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<double>* in_batch,
    std::complex<double>* out_batch,
    const int ik,
    int batch_count,
    const bool add,
    const double factor) const;

template void PW_Basis_K::recip_to_real_batch<float, base_device::DEVICE_CPU>(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<float>* in_batch,
    std::complex<float>* out_batch,
    const int ik,
    int batch_count,
    const bool add,
    const float factor) const;

template void PW_Basis_K::recip_to_real_batch<double, base_device::DEVICE_CPU>(
    const base_device::DEVICE_CPU* ctx,
    const std::complex<double>* in_batch,
    std::complex<double>* out_batch,
    const int ik,
    int batch_count,
    const bool add,
    const double factor) const;

} // namespace ModulePW

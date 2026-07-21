#include "source_hamilt/module_xc/kernels/xc_gradcorr_op.h"

#include "source_hamilt/module_xc/kernels/xc_builtin_formula.h"
#include "source_base/module_device/types.h"

#include <base/macros/macros.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>

#define THREADS_PER_BLOCK 256

namespace hamilt
{

template <typename FPTYPE>
__device__ void xc_gcx_pbe_spin(const FPTYPE rhoup,
                                const FPTYPE rhodw,
                                const FPTYPE grhoup2,
                                const FPTYPE grhodw2,
                                const int iflag,
                                FPTYPE& sx,
                                FPTYPE& v1xup,
                                FPTYPE& v1xdw,
                                FPTYPE& v2xup,
                                FPTYPE& v2xdw)
{
    const FPTYPE small = 1.0e-10;
    sx = 0.0;
    v1xup = 0.0;
    v1xdw = 0.0;
    v2xup = 0.0;
    v2xdw = 0.0;
    if (rhoup + rhodw <= small)
    {
        return;
    }

    FPTYPE sxup = 0.0;
    FPTYPE sxdw = 0.0;
    if (rhoup > small && sqrt(fabs(grhoup2)) > small)
    {
        xc_builtin::pbex(iflag,
                         static_cast<FPTYPE>(2.0) * rhoup,
                         static_cast<FPTYPE>(4.0) * grhoup2,
                         sxup,
                         v1xup,
                         v2xup);
    }
    if (rhodw > small && sqrt(fabs(grhodw2)) > small)
    {
        xc_builtin::pbex(iflag,
                         static_cast<FPTYPE>(2.0) * rhodw,
                         static_cast<FPTYPE>(4.0) * grhodw2,
                         sxdw,
                         v1xdw,
                         v2xdw);
    }
    sx = 0.5 * (sxup + sxdw);
    v2xup *= 2.0;
    v2xdw *= 2.0;
}

template <typename FPTYPE>
__device__ void xc_gcc_pbe_spin(const FPTYPE rho,
                                FPTYPE zeta,
                                const FPTYPE grho,
                                const int iflag,
                                FPTYPE& sc,
                                FPTYPE& v1cup,
                                FPTYPE& v1cdw,
                                FPTYPE& v2c)
{
    const FPTYPE small = 1.0e-10;
    const FPTYPE epsr = 1.0e-6;
    sc = 0.0;
    v1cup = 0.0;
    v1cdw = 0.0;
    v2c = 0.0;
    if (fabs(zeta) - 1.0 > small || rho <= small || sqrt(fabs(grho)) <= small)
    {
        return;
    }
    const FPTYPE x = fmin(fabs(zeta), static_cast<FPTYPE>(1.0) - epsr);
    zeta = zeta > 0.0 ? x : -x;
    xc_builtin::pbec_spin(rho, zeta, grho, iflag == 2 ? 2 : 1, sc, v1cup, v1cdw, v2c);
}

template <typename FPTYPE>
__global__ void xc_scalar_pbe_kernel(const int nrxx,
                                     const FPTYPE e2,
                                     const FPTYPE epsr,
                                     const FPTYPE* rho,
                                     const FPTYPE* rho_core,
                                     FPTYPE* rho_total,
                                     FPTYPE* v,
                                     FPTYPE* sums)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    const FPTYPE rhox = rho[ir] + rho_core[ir];
    rho_total[ir] = rhox;
    v[ir] = 0.0;
    const FPTYPE arho = fabs(rhox);
    if (arho <= epsr)
    {
        return;
    }

    FPTYPE exc = 0.0;
    FPTYPE vxc = 0.0;
    xc_builtin::scalar_pbe(arho, exc, vxc);
    v[ir] = e2 * vxc;
    atomicAdd(sums, e2 * exc * rhox);
    atomicAdd(sums + 1, e2 * vxc * rho[ir]);
}

template <typename FPTYPE>
__global__ void xc_scalar_lda_spin_kernel(const int nrxx,
                                          const int correlation,
                                          const FPTYPE e2,
                                          const FPTYPE epsr,
                                          const FPTYPE* rho_up,
                                          const FPTYPE* rho_dw,
                                          const FPTYPE* rho_core,
                                          FPTYPE* rho_up_total,
                                          FPTYPE* rho_dw_total,
                                          FPTYPE* v,
                                          FPTYPE* sums)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    const FPTYPE rhoup = rho_up[ir] + 0.5 * rho_core[ir];
    const FPTYPE rhodw = rho_dw[ir] + 0.5 * rho_core[ir];
    const FPTYPE rhox = rhoup + rhodw;
    rho_up_total[ir] = rhoup;
    rho_dw_total[ir] = rhodw;
    v[ir] = 0.0;
    v[nrxx + ir] = 0.0;
    const FPTYPE arho = fabs(rhox);
    if (arho <= epsr)
    {
        return;
    }

    FPTYPE zeta = (rho_up[ir] - rho_dw[ir]) / arho;
    if (fabs(zeta) > 1.0)
    {
        zeta = zeta > 0.0 ? 1.0 : -1.0;
    }
    FPTYPE exc = 0.0;
    FPTYPE vup = 0.0;
    FPTYPE vdw = 0.0;
    xc_builtin::scalar_lda_spin(arho, zeta, correlation, exc, vup, vdw);
    v[ir] = e2 * vup;
    v[nrxx + ir] = e2 * vdw;
    atomicAdd(sums, e2 * exc * rhox);
    atomicAdd(sums + 1, e2 * (vup * rho_up[ir] + vdw * rho_dw[ir]));
}

template <typename FPTYPE>
__global__ void xc_gradcorr_pbe_grid_kernel(const int nrxx,
                                            const int iflag,
                                            const FPTYPE e2,
                                            const FPTYPE epsr,
                                            const FPTYPE* rho,
                                            const FPTYPE* rho_core,
                                            const FPTYPE* gdr,
                                            FPTYPE* v,
                                            FPTYPE* h,
                                            FPTYPE* sums)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    v[ir] = 0.0;
    h[3 * ir + 0] = 0.0;
    h[3 * ir + 1] = 0.0;
    h[3 * ir + 2] = 0.0;

    const FPTYPE arho = fabs(rho[ir]);
    if (arho <= epsr)
    {
        return;
    }

    const FPTYPE gx = gdr[3 * ir + 0];
    const FPTYPE gy = gdr[3 * ir + 1];
    const FPTYPE gz = gdr[3 * ir + 2];
    const FPTYPE grho = gx * gx + gy * gy + gz * gz;
    if (grho < static_cast<FPTYPE>(1.0e-10))
    {
        return;
    }

    FPTYPE sx = 0.0;
    FPTYPE v1x = 0.0;
    FPTYPE v2x = 0.0;
    FPTYPE sc = 0.0;
    FPTYPE v1c = 0.0;
    FPTYPE v2c = 0.0;
    xc_builtin::pbex(iflag, arho, grho, sx, v1x, v2x);
    xc_builtin::pbec(iflag, arho, grho, sc, v1c, v2c);

    const FPTYPE sxc = sx + sc;
    const FPTYPE v1xc = v1x + v1c;
    const FPTYPE v2xc = v2x + v2c;
    const FPTYPE segno = rho[ir] >= 0.0 ? 1.0 : -1.0;

    v[ir] = e2 * v1xc;
    h[3 * ir + 0] = e2 * v2xc * gx;
    h[3 * ir + 1] = e2 * v2xc * gy;
    h[3 * ir + 2] = e2 * v2xc * gz;
    atomicAdd(sums, e2 * sxc * segno);
    atomicAdd(sums + 1, e2 * v1xc * (rho[ir] - rho_core[ir]));
}

template <typename FPTYPE>
__global__ void xc_gradcorr_pbe_grid_resident_kernel(const int nrxx,
                                                     const int iflag,
                                                     const FPTYPE e2,
                                                     const FPTYPE epsr,
                                                     const FPTYPE* rho,
                                                     const FPTYPE* rho_core,
                                                     const FPTYPE* gdr,
                                                     FPTYPE* v,
                                                     FPTYPE* h,
                                                     FPTYPE* sums)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    h[3 * ir + 0] = 0.0;
    h[3 * ir + 1] = 0.0;
    h[3 * ir + 2] = 0.0;

    const FPTYPE arho = fabs(rho[ir]);
    if (arho <= epsr)
    {
        return;
    }

    const FPTYPE gx = gdr[3 * ir + 0];
    const FPTYPE gy = gdr[3 * ir + 1];
    const FPTYPE gz = gdr[3 * ir + 2];
    const FPTYPE grho = gx * gx + gy * gy + gz * gz;
    if (grho < static_cast<FPTYPE>(1.0e-10))
    {
        return;
    }

    FPTYPE sx = 0.0;
    FPTYPE v1x = 0.0;
    FPTYPE v2x = 0.0;
    FPTYPE sc = 0.0;
    FPTYPE v1c = 0.0;
    FPTYPE v2c = 0.0;
    xc_builtin::pbex(iflag, arho, grho, sx, v1x, v2x);
    xc_builtin::pbec(iflag, arho, grho, sc, v1c, v2c);

    const FPTYPE sxc = sx + sc;
    const FPTYPE v1xc = v1x + v1c;
    const FPTYPE v2xc = v2x + v2c;
    const FPTYPE segno = rho[ir] >= 0.0 ? 1.0 : -1.0;

    v[ir] += e2 * v1xc;
    h[3 * ir + 0] = e2 * v2xc * gx;
    h[3 * ir + 1] = e2 * v2xc * gy;
    h[3 * ir + 2] = e2 * v2xc * gz;
    atomicAdd(sums, e2 * sxc * segno);
    atomicAdd(sums + 1, e2 * v1xc * (rho[ir] - rho_core[ir]));
}

template <typename FPTYPE>
__global__ void xc_gradcorr_pbe_spin_grid_resident_kernel(const int nrxx,
                                                          const int iflag,
                                                          const FPTYPE e2,
                                                          const FPTYPE epsr,
                                                          const FPTYPE* rho_up,
                                                          const FPTYPE* rho_dw,
                                                          const FPTYPE* rho_core,
                                                          const FPTYPE* gdr_up,
                                                          const FPTYPE* gdr_dw,
                                                          FPTYPE* v,
                                                          FPTYPE* h_up,
                                                          FPTYPE* h_dw,
                                                          FPTYPE* sums)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    h_up[3 * ir + 0] = 0.0;
    h_up[3 * ir + 1] = 0.0;
    h_up[3 * ir + 2] = 0.0;
    h_dw[3 * ir + 0] = 0.0;
    h_dw[3 * ir + 1] = 0.0;
    h_dw[3 * ir + 2] = 0.0;

    const FPTYPE rhoup = rho_up[ir];
    const FPTYPE rhodw = rho_dw[ir];
    const FPTYPE rh = rhoup + rhodw;
    const FPTYPE gxup = gdr_up[3 * ir + 0];
    const FPTYPE gyup = gdr_up[3 * ir + 1];
    const FPTYPE gzup = gdr_up[3 * ir + 2];
    const FPTYPE gxdw = gdr_dw[3 * ir + 0];
    const FPTYPE gydw = gdr_dw[3 * ir + 1];
    const FPTYPE gzdw = gdr_dw[3 * ir + 2];
    const FPTYPE grho2up = gxup * gxup + gyup * gyup + gzup * gzup;
    const FPTYPE grho2dw = gxdw * gxdw + gydw * gydw + gzdw * gzdw;

    FPTYPE sx = 0.0;
    FPTYPE sc = 0.0;
    FPTYPE v1xup = 0.0;
    FPTYPE v1xdw = 0.0;
    FPTYPE v2xup = 0.0;
    FPTYPE v2xdw = 0.0;
    FPTYPE v1cup = 0.0;
    FPTYPE v1cdw = 0.0;
    FPTYPE v2c = 0.0;
    xc_gcx_pbe_spin(rhoup, rhodw, grho2up, grho2dw, iflag, sx, v1xup, v1xdw, v2xup, v2xdw);
    if (rh > epsr)
    {
        FPTYPE zeta = (rhoup - rhodw) / rh;
        const FPTYPE grh2 = (gxup + gxdw) * (gxup + gxdw) + (gyup + gydw) * (gyup + gydw)
                            + (gzup + gzdw) * (gzup + gzdw);
        xc_gcc_pbe_spin(rh, zeta, grh2, iflag, sc, v1cup, v1cdw, v2c);
    }

    const FPTYPE v1up = v1xup + v1cup;
    const FPTYPE v1dw = v1xdw + v1cdw;
    v[ir] += e2 * v1up;
    v[nrxx + ir] += e2 * v1dw;
    h_up[3 * ir + 0] = e2 * ((v2xup + v2c) * gxup + v2c * gxdw);
    h_up[3 * ir + 1] = e2 * ((v2xup + v2c) * gyup + v2c * gydw);
    h_up[3 * ir + 2] = e2 * ((v2xup + v2c) * gzup + v2c * gzdw);
    h_dw[3 * ir + 0] = e2 * ((v2xdw + v2c) * gxdw + v2c * gxup);
    h_dw[3 * ir + 1] = e2 * ((v2xdw + v2c) * gydw + v2c * gyup);
    h_dw[3 * ir + 2] = e2 * ((v2xdw + v2c) * gzdw + v2c * gzup);
    atomicAdd(sums, e2 * (sx + sc));
    atomicAdd(sums + 1, e2 * v1up * (rhoup - 0.5 * rho_core[ir]));
    atomicAdd(sums + 1, e2 * v1dw * (rhodw - 0.5 * rho_core[ir]));
}

template <typename FPTYPE>
__global__ void xc_gradcorr_pbe_stress_kernel(const int nrxx,
                                              const int iflag,
                                              const FPTYPE e2,
                                              const FPTYPE epsr,
                                              const FPTYPE* rho,
                                              const FPTYPE* gdr,
                                              FPTYPE* stress)
{
    __shared__ FPTYPE block_stress[6 * THREADS_PER_BLOCK];
    FPTYPE local_stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const int tid = threadIdx.x;
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir < nrxx)
    {
        const FPTYPE arho = fabs(rho[ir]);
        const FPTYPE gx = gdr[3 * ir + 0];
        const FPTYPE gy = gdr[3 * ir + 1];
        const FPTYPE gz = gdr[3 * ir + 2];
        const FPTYPE grho = gx * gx + gy * gy + gz * gz;

        if (arho > epsr && grho >= static_cast<FPTYPE>(1.0e-10))
        {
            FPTYPE sx = 0.0;
            FPTYPE v1x = 0.0;
            FPTYPE v2x = 0.0;
            FPTYPE sc = 0.0;
            FPTYPE v1c = 0.0;
            FPTYPE v2c = 0.0;
            xc_builtin::pbex(iflag, arho, grho, sx, v1x, v2x);
            xc_builtin::pbec(iflag, arho, grho, sc, v1c, v2c);

            const FPTYPE factor = e2 * (v2x + v2c);
            local_stress[0] = gx * gx * factor;
            local_stress[1] = gy * gx * factor;
            local_stress[2] = gy * gy * factor;
            local_stress[3] = gz * gx * factor;
            local_stress[4] = gz * gy * factor;
            local_stress[5] = gz * gz * factor;
        }
    }

    for (int i = 0; i < 6; ++i)
    {
        block_stress[i * blockDim.x + tid] = local_stress[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int i = 0; i < 6; ++i)
            {
                block_stress[i * blockDim.x + tid] += block_stress[i * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < 6)
    {
        const int stress_index[6] = {0, 3, 4, 6, 7, 8};
        atomicAdd(stress + stress_index[tid], block_stress[tid * blockDim.x]);
    }
}

template <typename FPTYPE>
__global__ void xc_gradcorr_pbe_spin_stress_kernel(const int nrxx,
                                                   const int iflag,
                                                   const FPTYPE e2,
                                                   const FPTYPE epsr,
                                                   const FPTYPE* rho_up,
                                                   const FPTYPE* rho_dw,
                                                   const FPTYPE* gdr_up,
                                                   const FPTYPE* gdr_dw,
                                                   FPTYPE* stress)
{
    __shared__ FPTYPE block_stress[6 * THREADS_PER_BLOCK];
    FPTYPE local_stress[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    const int tid = threadIdx.x;
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir < nrxx)
    {
        const FPTYPE rhoup = rho_up[ir];
        const FPTYPE rhodw = rho_dw[ir];
        const FPTYPE rh = rhoup + rhodw;
        const FPTYPE gxup = gdr_up[3 * ir + 0];
        const FPTYPE gyup = gdr_up[3 * ir + 1];
        const FPTYPE gzup = gdr_up[3 * ir + 2];
        const FPTYPE gxdw = gdr_dw[3 * ir + 0];
        const FPTYPE gydw = gdr_dw[3 * ir + 1];
        const FPTYPE gzdw = gdr_dw[3 * ir + 2];
        const FPTYPE grho2up = gxup * gxup + gyup * gyup + gzup * gzup;
        const FPTYPE grho2dw = gxdw * gxdw + gydw * gydw + gzdw * gzdw;

        FPTYPE sx = 0.0;
        FPTYPE sc = 0.0;
        FPTYPE v1xup = 0.0;
        FPTYPE v1xdw = 0.0;
        FPTYPE v2xup = 0.0;
        FPTYPE v2xdw = 0.0;
        FPTYPE v1cup = 0.0;
        FPTYPE v1cdw = 0.0;
        FPTYPE v2c = 0.0;
        xc_gcx_pbe_spin(rhoup, rhodw, grho2up, grho2dw, iflag, sx, v1xup, v1xdw, v2xup, v2xdw);
        if (rh > epsr)
        {
            FPTYPE zeta = (rhoup - rhodw) / rh;
            const FPTYPE grh2 = (gxup + gxdw) * (gxup + gxdw) + (gyup + gydw) * (gyup + gydw)
                                + (gzup + gzdw) * (gzup + gzdw);
            xc_gcc_pbe_spin(rh, zeta, grh2, iflag, sc, v1cup, v1cdw, v2c);
        }

        const FPTYPE grad_up[3] = {gxup, gyup, gzup};
        const FPTYPE grad_dw[3] = {gxdw, gydw, gzdw};
        int index = 0;
        for (int l = 0; l < 3; ++l)
        {
            for (int m = 0; m <= l; ++m)
            {
                const FPTYPE exchange = grad_up[l] * grad_up[m] * e2 * v2xup
                                        + grad_dw[l] * grad_dw[m] * e2 * v2xdw;
                const FPTYPE correlation = (grad_up[l] * grad_up[m] * v2c
                                            + grad_dw[l] * grad_dw[m] * v2c
                                            + (grad_up[l] * grad_dw[m] + grad_dw[l] * grad_up[m]) * v2c)
                                           * e2;
                local_stress[index++] = exchange + correlation;
            }
        }
    }

    for (int i = 0; i < 6; ++i)
    {
        block_stress[i * blockDim.x + tid] = local_stress[i];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            for (int i = 0; i < 6; ++i)
            {
                block_stress[i * blockDim.x + tid] += block_stress[i * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < 6)
    {
        const int stress_index[6] = {0, 3, 4, 6, 7, 8};
        atomicAdd(stress + stress_index[tid], block_stress[tid * blockDim.x]);
    }
}

template <typename FPTYPE>
__global__ void xc_apply_dh_kernel(const int nrxx,
                                   const FPTYPE* rho,
                                   const FPTYPE* rho_core,
                                   const FPTYPE* dh,
                                   FPTYPE* v,
                                   FPTYPE* sum)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    v[ir] -= dh[ir];
    atomicAdd(sum, -dh[ir] * (rho[ir] - rho_core[ir]));
}

template <typename FPTYPE>
__global__ void xc_add_potential_kernel(const int size, const FPTYPE* src, FPTYPE* dst)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
    {
        dst[i] += src[i];
    }
}

template <typename FPTYPE>
__global__ void xc_apply_dh_spin_kernel(const int nrxx,
                                        const FPTYPE* rho,
                                        const FPTYPE* rho_core,
                                        const FPTYPE* dh,
                                        FPTYPE* v,
                                        FPTYPE* sum)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    v[ir] -= dh[ir];
    atomicAdd(sum, -dh[ir] * (rho[ir] - 0.5 * rho_core[ir]));
}

template <typename FPTYPE>
__global__ void xc_noncolin_rho_kernel(const int nrxx,
                                       const bool lsign,
                                       const FPTYPE* rho0,
                                       const FPTYPE* rho1,
                                       const FPTYPE* rho2,
                                       const FPTYPE* rho3,
                                       const FPTYPE* ux,
                                       FPTYPE* rho_up,
                                       FPTYPE* rho_dw,
                                       FPTYPE* neg)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    neg[ir] = 1.0;
    if (lsign)
    {
        const FPTYPE projection = rho1[ir] * ux[0] + rho2[ir] * ux[1] + rho3[ir] * ux[2];
        neg[ir] = projection > 0.0 ? 1.0 : -1.0;
    }
    const FPTYPE amag = sqrt(rho1[ir] * rho1[ir] + rho2[ir] * rho2[ir] + rho3[ir] * rho3[ir]);
    rho_up[ir] = 0.5 * (rho0[ir] + neg[ir] * amag);
    rho_dw[ir] = 0.5 * (rho0[ir] - neg[ir] * amag);
}

template <typename FPTYPE>
__global__ void xc_noncolin_rotate_potential_kernel(const int nrxx,
                                                    const FPTYPE* rho1,
                                                    const FPTYPE* rho2,
                                                    const FPTYPE* rho3,
                                                    const FPTYPE* neg,
                                                    const FPTYPE* v_up,
                                                    const FPTYPE* v_dw,
                                                    FPTYPE* v0,
                                                    FPTYPE* v1,
                                                    FPTYPE* v2,
                                                    FPTYPE* v3)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir >= nrxx)
    {
        return;
    }

    const FPTYPE vavg = 0.5 * (v_up[ir] + v_dw[ir]);
    const FPTYPE vdiff = 0.5 * (v_up[ir] - v_dw[ir]);
    v0[ir] += vavg;
    const FPTYPE amag = sqrt(rho1[ir] * rho1[ir] + rho2[ir] * rho2[ir] + rho3[ir] * rho3[ir]);
    if (amag > 1.0e-12)
    {
        const FPTYPE factor = neg[ir] * vdiff / amag;
        v1[ir] += factor * rho1[ir];
        v2[ir] += factor * rho2[ir];
        v3[ir] += factor * rho3[ir];
    }
}

template <typename FPTYPE>
__global__ void xc_multiply_iG_kernel(const int npw,
                                      const int ipol,
                                      const FPTYPE* gcar,
                                      const thrust::complex<FPTYPE>* rhog,
                                      thrust::complex<FPTYPE>* porter)
{
    const int ig = threadIdx.x + blockIdx.x * blockDim.x;
    if (ig >= npw)
    {
        return;
    }
    porter[ig] = thrust::complex<FPTYPE>(0.0, gcar[3 * ig + ipol]) * rhog[ig];
}

template <typename FPTYPE>
__global__ void xc_accumulate_iG_kernel(const int npw,
                                        const int ipol,
                                        const FPTYPE* gcar,
                                        const thrust::complex<FPTYPE>* rhog,
                                        thrust::complex<FPTYPE>* accum,
                                        const bool zero_first)
{
    const int ig = threadIdx.x + blockIdx.x * blockDim.x;
    if (ig >= npw)
    {
        return;
    }
    const thrust::complex<FPTYPE> term = thrust::complex<FPTYPE>(0.0, gcar[3 * ig + ipol]) * rhog[ig];
    accum[ig] = zero_first ? term : accum[ig] + term;
}

template <typename FPTYPE>
__global__ void xc_set_component_kernel(const int nrxx, const int ipol, const FPTYPE* component, FPTYPE* interleaved)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir < nrxx)
    {
        interleaved[3 * ir + ipol] = component[ir];
    }
}

template <typename FPTYPE>
__global__ void xc_extract_component_kernel(const int nrxx, const int ipol, const FPTYPE* interleaved, FPTYPE* component)
{
    const int ir = threadIdx.x + blockIdx.x * blockDim.x;
    if (ir < nrxx)
    {
        component[ir] = interleaved[3 * ir + ipol];
    }
}

template <typename FPTYPE>
void xc_scalar_pbe_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                      const int nrxx,
                                                                      const FPTYPE e2,
                                                                      const FPTYPE epsr,
                                                                      const FPTYPE* rho,
                                                                      const FPTYPE* rho_core,
                                                                      FPTYPE* rho_total,
                                                                      FPTYPE* v,
                                                                      FPTYPE* sums,
                                                                      FPTYPE* etxc,
                                                                      FPTYPE* vtxc)
{
    cudaMemset(sums, 0, 2 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_scalar_pbe_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, e2, epsr, rho, rho_core, rho_total, v, sums);
    CHECK_CUDA_SYNC();
    cudaMemcpy(etxc, sums, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(vtxc, sums + 1, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_scalar_lda_spin_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                        const int nrxx,
                                                                        const int correlation,
                                                                        const FPTYPE e2,
                                                                        const FPTYPE epsr,
                                                                        const FPTYPE* rho_up,
                                                                        const FPTYPE* rho_dw,
                                                                        const FPTYPE* rho_core,
                                                                        FPTYPE* rho_up_total,
                                                                        FPTYPE* rho_dw_total,
                                                                        FPTYPE* v,
                                                                        FPTYPE* sums,
                                                                        FPTYPE* etxc,
                                                                        FPTYPE* vtxc)
{
    cudaMemset(sums, 0, 2 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_scalar_lda_spin_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx,
                                                                    correlation,
                                                                    e2,
                                                                    epsr,
                                                                    rho_up,
                                                                    rho_dw,
                                                                    rho_core,
                                                                    rho_up_total,
                                                                    rho_dw_total,
                                                                    v,
                                                                    sums);
    CHECK_CUDA_SYNC();
    cudaMemcpy(etxc, sums, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(vtxc, sums + 1, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_gradcorr_pbe_grid_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                          const int nrxx,
                                                                          const int iflag,
                                                                          const FPTYPE e2,
                                                                          const FPTYPE epsr,
                                                                          const FPTYPE* rho,
                                                                          const FPTYPE* rho_core,
                                                                          const FPTYPE* gdr,
                                                                          FPTYPE* v,
                                                                          FPTYPE* h,
                                                                          FPTYPE* sums,
                                                                          FPTYPE* etxc,
                                                                          FPTYPE* vtxc)
{
    cudaMemset(sums, 0, 2 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_gradcorr_pbe_grid_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, iflag, e2, epsr, rho, rho_core, gdr, v, h, sums);
    CHECK_CUDA_SYNC();
    cudaMemcpy(etxc, sums, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(vtxc, sums + 1, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_gradcorr_pbe_grid_resident_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int nrxx,
    const int iflag,
    const FPTYPE e2,
    const FPTYPE epsr,
    const FPTYPE* rho,
    const FPTYPE* rho_core,
    const FPTYPE* gdr,
    FPTYPE* v,
    FPTYPE* h,
    FPTYPE* sums,
    FPTYPE* etxc,
    FPTYPE* vtxc)
{
    cudaMemset(sums, 0, 2 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_gradcorr_pbe_grid_resident_kernel<FPTYPE>
        <<<block, THREADS_PER_BLOCK>>>(nrxx, iflag, e2, epsr, rho, rho_core, gdr, v, h, sums);
    CHECK_CUDA_SYNC();
    cudaMemcpy(etxc, sums, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(vtxc, sums + 1, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_gradcorr_pbe_spin_grid_resident_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int nrxx,
    const int iflag,
    const FPTYPE e2,
    const FPTYPE epsr,
    const FPTYPE* rho_up,
    const FPTYPE* rho_dw,
    const FPTYPE* rho_core,
    const FPTYPE* gdr_up,
    const FPTYPE* gdr_dw,
    FPTYPE* v,
    FPTYPE* h_up,
    FPTYPE* h_dw,
    FPTYPE* sums,
    FPTYPE* etxc,
    FPTYPE* vtxc)
{
    cudaMemset(sums, 0, 2 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_gradcorr_pbe_spin_grid_resident_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx,
                                                                                    iflag,
                                                                                    e2,
                                                                                    epsr,
                                                                                    rho_up,
                                                                                    rho_dw,
                                                                                    rho_core,
                                                                                    gdr_up,
                                                                                    gdr_dw,
                                                                                    v,
                                                                                    h_up,
                                                                                    h_dw,
                                                                                    sums);
    CHECK_CUDA_SYNC();
    cudaMemcpy(etxc, sums, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(vtxc, sums + 1, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_gradcorr_pbe_stress_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                            const int nrxx,
                                                                            const int iflag,
                                                                            const FPTYPE e2,
                                                                            const FPTYPE epsr,
                                                                            const FPTYPE* rho,
                                                                            const FPTYPE* gdr,
                                                                            FPTYPE* stress)
{
    cudaMemset(stress, 0, 9 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_gradcorr_pbe_stress_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, iflag, e2, epsr, rho, gdr, stress);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_gradcorr_pbe_spin_stress_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                                 const int nrxx,
                                                                                 const int iflag,
                                                                                 const FPTYPE e2,
                                                                                 const FPTYPE epsr,
                                                                                 const FPTYPE* rho_up,
                                                                                 const FPTYPE* rho_dw,
                                                                                 const FPTYPE* gdr_up,
                                                                                 const FPTYPE* gdr_dw,
                                                                                 FPTYPE* stress)
{
    cudaMemset(stress, 0, 9 * sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_gradcorr_pbe_spin_stress_kernel<FPTYPE>
        <<<block, THREADS_PER_BLOCK>>>(nrxx, iflag, e2, epsr, rho_up, rho_dw, gdr_up, gdr_dw, stress);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_apply_dh_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                 const int nrxx,
                                                                 const FPTYPE* rho,
                                                                 const FPTYPE* rho_core,
                                                                 const FPTYPE* dh,
                                                                 FPTYPE* v,
                                                                 FPTYPE* sum,
                                                                 FPTYPE* vtxc_delta)
{
    cudaMemset(sum, 0, sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_apply_dh_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, rho, rho_core, dh, v, sum);
    CHECK_CUDA_SYNC();
    cudaMemcpy(vtxc_delta, sum, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_add_potential_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                      const int size,
                                                                      const FPTYPE* src,
                                                                      FPTYPE* dst)
{
    const int block = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_add_potential_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(size, src, dst);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_apply_dh_spin_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                      const int nrxx,
                                                                      const FPTYPE* rho,
                                                                      const FPTYPE* rho_core,
                                                                      const FPTYPE* dh,
                                                                      FPTYPE* v,
                                                                      FPTYPE* sum,
                                                                      FPTYPE* vtxc_delta)
{
    cudaMemset(sum, 0, sizeof(FPTYPE));
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_apply_dh_spin_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, rho, rho_core, dh, v, sum);
    CHECK_CUDA_SYNC();
    cudaMemcpy(vtxc_delta, sum, sizeof(FPTYPE), cudaMemcpyDeviceToHost);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_noncolin_rho_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                     const int nrxx,
                                                                     const bool lsign,
                                                                     const FPTYPE* rho0,
                                                                     const FPTYPE* rho1,
                                                                     const FPTYPE* rho2,
                                                                     const FPTYPE* rho3,
                                                                     const FPTYPE* ux,
                                                                     FPTYPE* rho_up,
                                                                     FPTYPE* rho_dw,
                                                                     FPTYPE* neg)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_noncolin_rho_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, lsign, rho0, rho1, rho2, rho3, ux, rho_up, rho_dw, neg);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_noncolin_rotate_potential_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                                 const int nrxx,
                                                                                 const FPTYPE* rho1,
                                                                                 const FPTYPE* rho2,
                                                                                 const FPTYPE* rho3,
                                                                                 const FPTYPE* neg,
                                                                                 const FPTYPE* v_up,
                                                                                 const FPTYPE* v_dw,
                                                                                 FPTYPE* v0,
                                                                                 FPTYPE* v1,
                                                                                 FPTYPE* v2,
                                                                                 FPTYPE* v3)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_noncolin_rotate_potential_kernel<FPTYPE>
        <<<block, THREADS_PER_BLOCK>>>(nrxx, rho1, rho2, rho3, neg, v_up, v_dw, v0, v1, v2, v3);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_multiply_iG_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                    const int npw,
                                                                    const int ipol,
                                                                    const FPTYPE* gcar,
                                                                    const std::complex<FPTYPE>* rhog,
                                                                    std::complex<FPTYPE>* porter)
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto rhog_ = reinterpret_cast<const thrust::complex<FPTYPE>*>(rhog);
    auto porter_ = reinterpret_cast<thrust::complex<FPTYPE>*>(porter);
    xc_multiply_iG_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(npw, ipol, gcar, rhog_, porter_);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_accumulate_iG_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                      const int npw,
                                                                      const int ipol,
                                                                      const FPTYPE* gcar,
                                                                      const std::complex<FPTYPE>* rhog,
                                                                      std::complex<FPTYPE>* accum,
                                                                      const bool zero_first)
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    auto rhog_ = reinterpret_cast<const thrust::complex<FPTYPE>*>(rhog);
    auto accum_ = reinterpret_cast<thrust::complex<FPTYPE>*>(accum);
    xc_accumulate_iG_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(npw, ipol, gcar, rhog_, accum_, zero_first);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_set_component_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                      const int nrxx,
                                                                      const int ipol,
                                                                      const FPTYPE* component,
                                                                      FPTYPE* interleaved)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_set_component_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, ipol, component, interleaved);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void xc_extract_component_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                          const int nrxx,
                                                                          const int ipol,
                                                                          const FPTYPE* interleaved,
                                                                          FPTYPE* component)
{
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    xc_extract_component_kernel<FPTYPE><<<block, THREADS_PER_BLOCK>>>(nrxx, ipol, interleaved, component);
    CHECK_CUDA_SYNC();
}

template struct xc_scalar_pbe_op<float, base_device::DEVICE_GPU>;
template struct xc_scalar_pbe_op<double, base_device::DEVICE_GPU>;
template struct xc_scalar_lda_spin_op<float, base_device::DEVICE_GPU>;
template struct xc_scalar_lda_spin_op<double, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_grid_op<float, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_grid_op<double, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_grid_resident_op<float, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_grid_resident_op<double, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_spin_grid_resident_op<float, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_spin_grid_resident_op<double, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_stress_op<float, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_stress_op<double, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_spin_stress_op<float, base_device::DEVICE_GPU>;
template struct xc_gradcorr_pbe_spin_stress_op<double, base_device::DEVICE_GPU>;
template struct xc_apply_dh_op<float, base_device::DEVICE_GPU>;
template struct xc_apply_dh_op<double, base_device::DEVICE_GPU>;
template struct xc_add_potential_op<float, base_device::DEVICE_GPU>;
template struct xc_add_potential_op<double, base_device::DEVICE_GPU>;
template struct xc_apply_dh_spin_op<float, base_device::DEVICE_GPU>;
template struct xc_apply_dh_spin_op<double, base_device::DEVICE_GPU>;
template struct xc_noncolin_rho_op<float, base_device::DEVICE_GPU>;
template struct xc_noncolin_rho_op<double, base_device::DEVICE_GPU>;
template struct xc_noncolin_rotate_potential_op<float, base_device::DEVICE_GPU>;
template struct xc_noncolin_rotate_potential_op<double, base_device::DEVICE_GPU>;
template struct xc_multiply_iG_op<float, base_device::DEVICE_GPU>;
template struct xc_multiply_iG_op<double, base_device::DEVICE_GPU>;
template struct xc_accumulate_iG_op<float, base_device::DEVICE_GPU>;
template struct xc_accumulate_iG_op<double, base_device::DEVICE_GPU>;
template struct xc_set_component_op<float, base_device::DEVICE_GPU>;
template struct xc_set_component_op<double, base_device::DEVICE_GPU>;
template struct xc_extract_component_op<float, base_device::DEVICE_GPU>;
template struct xc_extract_component_op<double, base_device::DEVICE_GPU>;

} // namespace hamilt

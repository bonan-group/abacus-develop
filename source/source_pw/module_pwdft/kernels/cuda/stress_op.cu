#include "source_pw/module_pwdft/kernels/stress_op.h"
#include "source_base/constants.h"
#include "source_base/kernels/math_ylm_op.h"
#include "source_base/module_device/device.h"
#include "source_base/module_device/kernel_compat.h"
#include "vnl_tools_cu.hpp"
#include "source_base/module_device/types.h"

#include <complex>
#include <thrust/complex.h>
#include "source_base/module_device/device_check.h"

#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define WARP_SIZE 32

namespace hamilt{

template <typename FPTYPE>
__forceinline__
__device__
void warp_reduce(FPTYPE & val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
}
template <typename T>
__device__ static inline 
thrust::complex<T> conj(thrust::complex<T>& in) {
    return thrust::conj(in);
}
template <typename T>
__global__ void cal_stress_mgga(
    const int spin,
    const int nrxx,
    const T w1,
    const thrust::complex<T> * gradwfc,
    T * crosstaus)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= nrxx) { return; }
    int ipol = 0;
    for (int ix = 0; ix < 3; ix++) {
        for (int iy = 0; iy < ix + 1; iy++) {
            crosstaus[spin * nrxx * 6 + ipol * nrxx + idx]
                += 2.0 * w1
                * (gradwfc[ix * nrxx + idx].real() * gradwfc[iy*nrxx + idx].real()
                +  gradwfc[ix * nrxx + idx].imag() * gradwfc[iy*nrxx + idx].imag());
            ipol += 1;
        }
    }
}

template <typename FPTYPE>
__global__ void set_stress_ewa_diag_kernel(const FPTYPE charge, const FPTYPE alpha, const FPTYPE omega, FPTYPE* stress)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < 7; ++i)
        {
            stress[i] = 0.0;
        }
        stress[6] = ModuleBase::TWO_PI * ModuleBase::e2 / 4.0 / alpha * (charge / omega) * (charge / omega);
    }
}

template <typename FPTYPE>
__device__ void stress_ewa_add_lower_triangle(const FPTYPE fac,
                                              const FPTYPE rx,
                                              const FPTYPE ry,
                                              const FPTYPE rz,
                                              FPTYPE& xx,
                                              FPTYPE& yx,
                                              FPTYPE& yy,
                                              FPTYPE& zx,
                                              FPTYPE& zy,
                                              FPTYPE& zz)
{
    xx += fac * rx * rx;
    yx += fac * ry * rx;
    yy += fac * ry * ry;
    zx += fac * rz * rx;
    zy += fac * rz * ry;
    zz += fac * rz * rz;
}

template <typename FPTYPE>
__global__ void stress_ewa_g_kernel(const int nat,
                                    const int npw,
                                    const int ig0,
                                    const FPTYPE alpha,
                                    const FPTYPE omega,
                                    const FPTYPE tpiba2,
                                    const FPTYPE fact,
                                    const FPTYPE* tau,
                                    const FPTYPE* atom_z,
                                    const FPTYPE* gcar,
                                    const FPTYPE* gg,
                                    FPTYPE* partial)
{
    const int tid = threadIdx.x;
    FPTYPE xx = 0.0;
    FPTYPE yx = 0.0;
    FPTYPE yy = 0.0;
    FPTYPE zx = 0.0;
    FPTYPE zy = 0.0;
    FPTYPE zz = 0.0;
    FPTYPE sdewald = 0.0;
    for (int ig = blockIdx.x * blockDim.x + tid; ig < npw; ig += blockDim.x * gridDim.x)
    {
        if (ig == ig0)
        {
            continue;
        }
        const FPTYPE gx = gcar[ig * 3];
        const FPTYPE gy = gcar[ig * 3 + 1];
        const FPTYPE gz = gcar[ig * 3 + 2];
        const FPTYPE g2 = gg[ig] * tpiba2;
        const FPTYPE g2a = g2 / 4.0 / alpha;
        FPTYPE rho_real = 0.0;
        FPTYPE rho_imag = 0.0;
        for (int iat = 0; iat < nat; ++iat)
        {
            const FPTYPE arg = ModuleBase::TWO_PI * (gx * tau[iat * 3] + gy * tau[iat * 3 + 1]
                                                     + gz * tau[iat * 3 + 2]);
            FPTYPE sinp = 0.0;
            FPTYPE cosp = 0.0;
            sincos(arg, &sinp, &cosp);
            rho_real += atom_z[iat] * cosp;
            rho_imag += atom_z[iat] * sinp;
        }
        rho_real /= omega;
        rho_imag /= omega;
        const FPTYPE sewald = fact * ModuleBase::TWO_PI * ModuleBase::e2 * exp(-g2a) / g2
                              * (rho_real * rho_real + rho_imag * rho_imag);
        sdewald -= sewald;
        const FPTYPE tensor_fac = sewald * tpiba2 * 2.0 / g2 * (g2a + 1.0);
        stress_ewa_add_lower_triangle(tensor_fac, gx, gy, gz, xx, yx, yy, zx, zy, zz);
    }

    warp_reduce(xx);
    warp_reduce(yx);
    warp_reduce(yy);
    warp_reduce(zx);
    warp_reduce(zy);
    warp_reduce(zz);
    warp_reduce(sdewald);

    __shared__ FPTYPE warp_sums[7][THREADS_PER_BLOCK / WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0)
    {
        warp_sums[0][warp_id] = xx;
        warp_sums[1][warp_id] = yx;
        warp_sums[2][warp_id] = yy;
        warp_sums[3][warp_id] = zx;
        warp_sums[4][warp_id] = zy;
        warp_sums[5][warp_id] = zz;
        warp_sums[6][warp_id] = sdewald;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        FPTYPE vals[7];
        for (int i = 0; i < 7; ++i)
        {
            vals[i] = lane_id < (blockDim.x / WARP_SIZE) ? warp_sums[i][lane_id] : 0.0;
            warp_reduce(vals[i]);
            if (lane_id == 0)
            {
                partial[blockIdx.x * 7 + i] = vals[i];
            }
        }
    }
}

template <typename FPTYPE>
__global__ void stress_ewa_r_kernel(const int nat,
                                    const int nm1,
                                    const int nm2,
                                    const int nm3,
                                    const FPTYPE alpha,
                                    const FPTYPE omega,
                                    const FPTYPE lat0,
                                    const FPTYPE rmax,
                                    const FPTYPE* tau,
                                    const FPTYPE* atom_z,
                                    const FPTYPE* latvec,
                                    FPTYPE* partial)
{
    const int tid = threadIdx.x;
    FPTYPE xx = 0.0;
    FPTYPE yx = 0.0;
    FPTYPE yy = 0.0;
    FPTYPE zx = 0.0;
    FPTYPE zy = 0.0;
    FPTYPE zz = 0.0;
    const long long npairs = static_cast<long long>(nat) * nat;
    const FPTYPE sqa = sqrt(alpha);
    const FPTYPE sq8a_2pi = sqrt(8.0 * alpha / ModuleBase::TWO_PI);
    const FPTYPE rmax2 = rmax * rmax;
    for (long long pair = blockIdx.x * blockDim.x + tid; pair < npairs; pair += blockDim.x * gridDim.x)
    {
        const int iat = pair / nat;
        const int jat = pair - static_cast<long long>(iat) * nat;
        const FPTYPE dtau_x = tau[iat * 3] - tau[jat * 3];
        const FPTYPE dtau_y = tau[iat * 3 + 1] - tau[jat * 3 + 1];
        const FPTYPE dtau_z = tau[iat * 3 + 2] - tau[jat * 3 + 2];
        for (int ia = -nm1; ia <= nm1; ++ia)
        {
            for (int ib = -nm2; ib <= nm2; ++ib)
            {
                for (int ic = -nm3; ic <= nm3; ++ic)
                {
                    const FPTYPE rx = ia * latvec[0] + ib * latvec[3] + ic * latvec[6] - dtau_x;
                    const FPTYPE ry = ia * latvec[1] + ib * latvec[4] + ic * latvec[7] - dtau_y;
                    const FPTYPE rz = ia * latvec[2] + ib * latvec[5] + ic * latvec[8] - dtau_z;
                    const FPTYPE r2 = rx * rx + ry * ry + rz * rz;
                    if (r2 > rmax2 || fabs(r2) <= 1.0e-10)
                    {
                        continue;
                    }
                    const FPTYPE rr = sqrt(r2) * lat0;
                    const FPTYPE fac = -ModuleBase::e2 / 2.0 / omega * lat0 * lat0 * atom_z[iat] * atom_z[jat]
                                       / (rr * rr * rr)
                                       * (erfc(sqa * rr) + rr * sq8a_2pi * exp(-alpha * rr * rr));
                    stress_ewa_add_lower_triangle(fac, rx, ry, rz, xx, yx, yy, zx, zy, zz);
                }
            }
        }
    }

    warp_reduce(xx);
    warp_reduce(yx);
    warp_reduce(yy);
    warp_reduce(zx);
    warp_reduce(zy);
    warp_reduce(zz);

    __shared__ FPTYPE warp_sums[6][THREADS_PER_BLOCK / WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0)
    {
        warp_sums[0][warp_id] = xx;
        warp_sums[1][warp_id] = yx;
        warp_sums[2][warp_id] = yy;
        warp_sums[3][warp_id] = zx;
        warp_sums[4][warp_id] = zy;
        warp_sums[5][warp_id] = zz;
    }
    __syncthreads();
    if (warp_id == 0)
    {
        FPTYPE vals[6];
        for (int i = 0; i < 6; ++i)
        {
            vals[i] = lane_id < (blockDim.x / WARP_SIZE) ? warp_sums[i][lane_id] : 0.0;
            warp_reduce(vals[i]);
            if (lane_id == 0)
            {
                partial[blockIdx.x * 7 + i] = vals[i];
            }
        }
        if (lane_id == 0)
        {
            partial[blockIdx.x * 7 + 6] = 0.0;
        }
    }
}

template <typename FPTYPE>
__global__ void stress_ewa_final_reduce_kernel(const FPTYPE* partial, const int blocks, FPTYPE* stress)
{
    const int tid = threadIdx.x;
    FPTYPE vals[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int ib = tid; ib < blocks; ib += blockDim.x)
    {
        for (int i = 0; i < 7; ++i)
        {
            vals[i] += partial[ib * 7 + i];
        }
    }
    for (int i = 0; i < 7; ++i)
    {
        warp_reduce(vals[i]);
    }
    __shared__ FPTYPE warp_sums[7][THREADS_PER_BLOCK / WARP_SIZE];
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    if (lane_id == 0)
    {
        for (int i = 0; i < 7; ++i)
        {
            warp_sums[i][warp_id] = vals[i];
        }
    }
    __syncthreads();
    if (warp_id == 0)
    {
        for (int i = 0; i < 7; ++i)
        {
            FPTYPE val = lane_id < (blockDim.x / WARP_SIZE) ? warp_sums[i][lane_id] : 0.0;
            warp_reduce(val);
            if (lane_id == 0)
            {
                stress[i] += val;
            }
        }
    }
}

template <typename FPTYPE>
__global__ void cal_dbecp_noevc_nl(
        const int ipol,
        const int jpol,
        const int npw,
        const int npwx,
        const int ik,
        const FPTYPE tpiba,
        const FPTYPE *gcar,
        const FPTYPE *kvec_c,
        thrust::complex<FPTYPE> *vkbi,
        thrust::complex<FPTYPE> *vkbj,
        thrust::complex<FPTYPE> *vkb,
        thrust::complex<FPTYPE> *vkb1,
        thrust::complex<FPTYPE> *vkb2,
        thrust::complex<FPTYPE> *dbecp_noevc)
{
    int i = blockIdx.x;
    const thrust::complex<FPTYPE>* pvkb0i = vkbi + i * npwx;
    const thrust::complex<FPTYPE>* pvkb0j = vkbj + i * npwx;
    thrust::complex<FPTYPE>* pvkb = nullptr;
    thrust::complex<FPTYPE>* pdbecp_noevc = dbecp_noevc + i * npwx;
    // third term of dbecp_noevc
    //std::complex<FPTYPE>* pvkb = &vkb2(i,0);
    //std::complex<FPTYPE>* pdbecp_noevc = &dbecp_noevc(i, 0);
    FPTYPE qvec[3] = {0, 0, 0};
    for (int ig = threadIdx.x; ig < npw; ig += blockDim.x)
    {
        pvkb = vkb1 + i * npwx;
        qvec[ipol] = gcar[(ik * npwx + ig) * 3 + ipol] + kvec_c[ik * 3 + ipol];
        qvec[jpol] = gcar[(ik * npwx + ig) * 3 + jpol] + kvec_c[ik * 3 + jpol];
        pvkb[ig] += 0.5 * qvec[ipol] * pvkb0j[ig] +
                    0.5 * qvec[jpol] * pvkb0i[ig];
        pdbecp_noevc[ig] -= 2.0 * pvkb[ig];
        if (ipol == jpol) {
            pvkb = vkb + i * npwx;
            pdbecp_noevc[ig] -= pvkb[ig];
        }
        pvkb = vkb2 + i * npwx;
        for (int ii = 0; ii < 3; ii++) {
            qvec[ii] = gcar[(ik * npwx + ig) * 3 + ii] + kvec_c[ik * 3 + ii];
        }
        FPTYPE qvec_norm2 = qvec[0] * qvec[0] + qvec[1] * qvec[1] + qvec[2] * qvec[2];
        FPTYPE qm1 = qvec_norm2 > 1e-16 ? 1.0 / sqrt(qvec_norm2) : 0;
        pdbecp_noevc[ig] -= 2.0 * pvkb[ig] * qvec[ipol] *
                            qvec[jpol] * qm1 *	tpiba;
    } // end ig
}

template <typename FPTYPE>
__global__ void cal_stress_nl(
        const bool nondiagonal,
        const int ipol,
        const int jpol,
        const int nkb,
        const int ntype,
        const int spin,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE *d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const FPTYPE *deeq,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *stress)
{
    int ib = blockIdx.x / ntype;
    int it = blockIdx.x % ntype;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    FPTYPE stress_var = 0;
    FPTYPE fac;
    if (occ)
    {
        fac = d_wg[ib];
    }
    else
    {
        fac = d_wg[0];
    }
    FPTYPE ekb_now = 0.0;
    if (d_ekb != nullptr)
    {
        ekb_now = d_ekb[ib];
    }
    const int nproj = atom_nh[it];
    for (int ia = 0; ia < atom_na[it]; ia++)
    {
        for (int ii = threadIdx.x; ii < nproj * nproj; ii += blockDim.x) {
            const int ip1 = ii / nproj, ip2 = ii % nproj;
            if(!nondiagonal && ip1 != ip2) {
                continue;
            }
            FPTYPE ps_qq = 0;
            if (ekb_now != 0)
            {
                ps_qq = -ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2];
            }
            const FPTYPE ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
            const int inkb1 = sum + ip1;
            const int inkb2 = sum + ip2;
            //out<<"\n ps = "<<ps;
            const FPTYPE dbb = ( conj( dbecp[ ib * nkb + inkb1] ) * becp[ ib * nkb + inkb2] ).real();
            stress_var -= ps * fac * dbb;
        }
        ++iat;
        sum+=nproj;
    }//ia
    __syncwarp();
    warp_reduce(stress_var);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(stress + ipol * 3 + jpol, stress_var);
    }
}

template <typename FPTYPE>
__global__ void cal_multi_dot(const int npw,
                              const FPTYPE fac,
                              const FPTYPE* gk1,
                              const FPTYPE* gk2,
                              const FPTYPE* d_kfac,
                              const thrust::complex<FPTYPE>* psi,
                              FPTYPE* sum)
{
    __shared__ FPTYPE s_sum[THREADS_PER_BLOCK];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheid = threadIdx.x;
    FPTYPE local_sum = 0;
    while (tid < npw) {
        local_sum += fac * gk1[tid] * gk2[tid] * d_kfac[tid] * thrust::norm(psi[tid]);
        tid += blockDim.x * gridDim.x;
    }
    s_sum[cacheid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (cacheid < s) {
            s_sum[cacheid] += s_sum[cacheid + s];
        }
        __syncthreads();
    }
    if (cacheid == 0) {
        atomicAdd(sum, s_sum[0]);
    }
}

template <typename FPTYPE>
__global__ void cal_kinetic_stress(const int npw,
                                   const int npwk_max,
                                   const int npol,
                                   const int nbands,
                                   const FPTYPE* band_weight,
                                   const bool occ,
                                   const FPTYPE k_weight,
                                   const FPTYPE* gk,
                                   const FPTYPE* kfac,
                                   const thrust::complex<FPTYPE>* psi,
                                   FPTYPE* stress)
{
    const int component = blockIdx.x;
    const int ib = blockIdx.y;
    const int pairs_l[6] = {0, 1, 1, 2, 2, 2};
    const int pairs_m[6] = {0, 0, 1, 0, 1, 2};
    const int l = pairs_l[component];
    const int m = pairs_m[component];
    const FPTYPE fac = occ ? band_weight[ib] : k_weight;
    if (fac == 0.0)
    {
        return;
    }

    __shared__ FPTYPE s_sum[THREADS_PER_BLOCK];
    FPTYPE local_sum = 0;
    const FPTYPE* gkl = gk + l * npwk_max;
    const FPTYPE* gkm = gk + m * npwk_max;
    for (int ipol = 0; ipol < npol; ++ipol)
    {
        const thrust::complex<FPTYPE>* ppsi = psi + (ib * npol + ipol) * npwk_max;
        for (int ig = threadIdx.x; ig < npw; ig += blockDim.x)
        {
            local_sum += fac * gkl[ig] * gkm[ig] * kfac[ig] * thrust::norm(ppsi[ig]);
        }
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        atomicAdd(stress + l * 3 + m, s_sum[0]);
    }
}

template <typename FPTYPE>
void cal_dbecp_noevc_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                        const int& ipol,
                                                                        const int& jpol,
                                                                        const int& nkb,
                                                                        const int& npw,
                                                                        const int& npwx,
                                                                        const int& ik,
                                                                        const FPTYPE& tpiba,
                                                                        const FPTYPE* gcar,
                                                                        const FPTYPE* kvec_c,
                                                                        std::complex<FPTYPE>* vkbi,
                                                                        std::complex<FPTYPE>* vkbj,
                                                                        std::complex<FPTYPE>* vkb,
                                                                        std::complex<FPTYPE>* vkb1,
                                                                        std::complex<FPTYPE>* vkb2,
                                                                        std::complex<FPTYPE>* dbecp_noevc)
{
    cal_dbecp_noevc_nl<FPTYPE><<<nkb, THREADS_PER_BLOCK>>>(
            ipol,
            jpol,
            npw,
            npwx,
            ik,
            tpiba,
            gcar,
            kvec_c,
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkbi),
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkbj),
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkb),
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkb1),
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkb2),
            reinterpret_cast<thrust::complex<FPTYPE>*>(dbecp_noevc));

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void cal_stress_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                   const bool& nondiagonal,
                                                                   const int& ipol,
                                                                   const int& jpol,
                                                                   const int& nkb,
                                                                   const int& nbands_occ,
                                                                   const int& ntype,
                                                                   const int& spin,
                                                                   const int& deeq_2,
                                                                   const int& deeq_3,
                                                                   const int& deeq_4,
                                                                   const int* atom_nh,
                                                                   const int* atom_na,
                                                                   const FPTYPE* d_wg,
                                                                   const bool& occ,
                                                                   const FPTYPE* d_ekb,
                                                                   const FPTYPE* qq_nt,
                                                                   const FPTYPE* deeq,
                                                                   const std::complex<FPTYPE>* becp,
                                                                   const std::complex<FPTYPE>* dbecp,
                                                                   FPTYPE* stress)
{
     cal_stress_nl<FPTYPE><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
             nondiagonal,
             ipol,
             jpol,
             nkb,
             ntype,
             spin,
             deeq_2,
             deeq_3,
             deeq_4,
             atom_nh,
             atom_na,
             d_wg,
             occ,
             d_ekb,
             qq_nt,
             deeq,
             reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
             reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
             stress);// array of data

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void cal_stress_nl_chunk(
        const bool nondiagonal,
        const int ipol,
        const int jpol,
        const int chunk_nkb,
        const int spin,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int it,
        const int atom_start,
        const int atom_count,
        const int nproj,
        const FPTYPE *d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const FPTYPE *deeq,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *stress)
{
    int ib = blockIdx.x;

    FPTYPE stress_var = 0;
    FPTYPE fac;
    if (occ)
    {
        fac = d_wg[ib];
    }
    else
    {
        fac = d_wg[0];
    }
    FPTYPE ekb_now = 0.0;
    if (d_ekb != nullptr)
    {
        ekb_now = d_ekb[ib];
    }
    for (int ia = 0; ia < atom_count; ia++)
    {
        const int iat = atom_start + ia;
        const int sum = ia * nproj;
        for (int ii = threadIdx.x; ii < nproj * nproj; ii += blockDim.x) {
            const int ip1 = ii / nproj, ip2 = ii % nproj;
            if(!nondiagonal && ip1 != ip2) {
                continue;
            }
            FPTYPE ps_qq = 0;
            if (ekb_now != 0)
            {
                ps_qq = -ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2];
            }
            const FPTYPE ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
            const int inkb1 = sum + ip1;
            const int inkb2 = sum + ip2;
            const FPTYPE dbb = (conj(dbecp[ib * chunk_nkb + inkb1]) * becp[ib * chunk_nkb + inkb2]).real();
            stress_var -= ps * fac * dbb;
        }
    }
    __syncwarp();
    warp_reduce(stress_var);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(stress + ipol * 3 + jpol, stress_var);
    }
}

template <typename FPTYPE>
void cal_stress_nl_op<FPTYPE, base_device::DEVICE_GPU>::chunk(const base_device::DEVICE_GPU* ctx,
                                                              const bool& nondiagonal,
                                                              const int& ipol,
                                                              const int& jpol,
                                                              const int& chunk_nkb,
                                                              const int& nbands_occ,
                                                              const int& spin,
                                                              const int& deeq_2,
                                                              const int& deeq_3,
                                                              const int& deeq_4,
                                                              const int& it,
                                                              const int& atom_start,
                                                              const int& atom_count,
                                                              const int& nproj,
                                                              const FPTYPE* d_wg,
                                                              const bool& occ,
                                                              const FPTYPE* d_ekb,
                                                              const FPTYPE* qq_nt,
                                                              const FPTYPE* deeq,
                                                              const std::complex<FPTYPE>* becp,
                                                              const std::complex<FPTYPE>* dbecp,
                                                              FPTYPE* stress)
{
    cal_stress_nl_chunk<FPTYPE><<<nbands_occ, THREADS_PER_BLOCK>>>(
        nondiagonal,
        ipol,
        jpol,
        chunk_nkb,
        spin,
        deeq_2,
        deeq_3,
        deeq_4,
        it,
        atom_start,
        atom_count,
        nproj,
        d_wg,
        occ,
        d_ekb,
        qq_nt,
        deeq,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
        stress);

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void build_stress_nl_reordered_r_chunk(
        const bool nondiagonal,
        const int chunk_nkb,
        const int nbands_occ,
        const int spin,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int it,
        const int atom_start,
        const int atom_count,
        const int nproj,
        const FPTYPE* d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const FPTYPE* deeq,
        const thrust::complex<FPTYPE>* becp,
        thrust::complex<FPTYPE>* r_chunk)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int total = nbands_occ * chunk_nkb;
    if (idx >= total)
    {
        return;
    }

    const int ib = idx / chunk_nkb;
    const int inkb1 = idx - ib * chunk_nkb;
    const int ia = inkb1 / nproj;
    const int ip1 = inkb1 - ia * nproj;
    if (ia >= atom_count)
    {
        r_chunk[idx] = thrust::complex<FPTYPE>(0.0, 0.0);
        return;
    }

    const int iat = atom_start + ia;
    const FPTYPE fac = occ ? d_wg[ib] : d_wg[0];
    const FPTYPE ekb_now = d_ekb != nullptr ? d_ekb[ib] : 0.0;
    thrust::complex<FPTYPE> sum(0.0, 0.0);
    for (int ip2 = 0; ip2 < nproj; ++ip2)
    {
        if (!nondiagonal && ip1 != ip2)
        {
            continue;
        }
        FPTYPE ps_qq = 0.0;
        if (ekb_now != 0.0)
        {
            ps_qq = -ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2];
        }
        const FPTYPE ps = deeq[((spin * deeq_2 + iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
        const int inkb2 = ia * nproj + ip2;
        sum += fac * ps * becp[ib * chunk_nkb + inkb2];
    }
    r_chunk[idx] = sum;
}

template <typename FPTYPE>
void build_stress_nl_reordered_r_op<FPTYPE, base_device::DEVICE_GPU>::chunk(
        const base_device::DEVICE_GPU* ctx,
        const bool& nondiagonal,
        const int& chunk_nkb,
        const int& nbands_occ,
        const int& spin,
        const int& deeq_2,
        const int& deeq_3,
        const int& deeq_4,
        const int& it,
        const int& atom_start,
        const int& atom_count,
        const int& nproj,
        const FPTYPE* d_wg,
        const bool& occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const FPTYPE* deeq,
        const std::complex<FPTYPE>* becp,
        std::complex<FPTYPE>* r_chunk)
{
    const int total = chunk_nkb * nbands_occ;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    build_stress_nl_reordered_r_chunk<FPTYPE><<<blocks, THREADS_PER_BLOCK>>>(
        nondiagonal,
        chunk_nkb,
        nbands_occ,
        spin,
        deeq_2,
        deeq_3,
        deeq_4,
        it,
        atom_start,
        atom_count,
        nproj,
        d_wg,
        occ,
        d_ekb,
        qq_nt,
        deeq,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
        reinterpret_cast<thrust::complex<FPTYPE>*>(r_chunk));

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void build_stress_nl_reordered_r_chunk_nc(
        const int chunk_nkb,
        const int nbands_occ,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int it,
        const int atom_start,
        const int atom_offset_in_type,
        const int atom_count,
        const int nproj,
        const FPTYPE* d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const thrust::complex<FPTYPE>* deeq_nc,
        const thrust::complex<FPTYPE>* becp,
        thrust::complex<FPTYPE>* r_chunk)
{
    const int total = nbands_occ * 2 * chunk_nkb;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total)
    {
        return;
    }

    const int ib_spinor = idx / chunk_nkb;
    const int ib = ib_spinor / 2;
    const int is = ib_spinor - ib * 2;
    const int inkb1 = idx - ib_spinor * chunk_nkb;
    const int ia = inkb1 / nproj;
    const int ip1 = inkb1 - ia * nproj;
    if (ia >= atom_count)
    {
        r_chunk[idx] = thrust::complex<FPTYPE>(0.0, 0.0);
        return;
    }

    const int chunk_type_start = atom_start - atom_offset_in_type;
    const int deeq_iat = chunk_type_start + atom_offset_in_type + ia;
    const FPTYPE fac = occ ? d_wg[ib] : d_wg[0];
    const FPTYPE ekb_now = d_ekb != nullptr ? d_ekb[ib] : 0.0;
    thrust::complex<FPTYPE> sum(0.0, 0.0);
    for (int ip2 = 0; ip2 < nproj; ++ip2)
    {
        thrust::complex<FPTYPE> ps_qq(0.0, 0.0);
        if (ekb_now != 0.0)
        {
            ps_qq = thrust::complex<FPTYPE>(
                -ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2],
                0.0);
        }
        const thrust::complex<FPTYPE> ps0
            = deeq_nc[((0 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
        const thrust::complex<FPTYPE> ps1
            = deeq_nc[((1 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2];
        const thrust::complex<FPTYPE> ps2
            = deeq_nc[((2 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2];
        const thrust::complex<FPTYPE> ps3
            = deeq_nc[((3 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
        const int inkb2 = ia * nproj + ip2;
        const thrust::complex<FPTYPE> becp_up = becp[(ib * 2) * chunk_nkb + inkb2];
        const thrust::complex<FPTYPE> becp_dn = becp[(ib * 2 + 1) * chunk_nkb + inkb2];
        if (is == 0)
        {
            sum += fac * (ps0 * becp_up + ps1 * becp_dn);
        }
        else
        {
            sum += fac * (ps2 * becp_up + ps3 * becp_dn);
        }
    }
    r_chunk[idx] = sum;
}

template <typename FPTYPE>
void build_stress_nl_reordered_r_op<FPTYPE, base_device::DEVICE_GPU>::chunk(
        const base_device::DEVICE_GPU* ctx,
        const int& chunk_nkb,
        const int& nbands_occ,
        const int& deeq_2,
        const int& deeq_3,
        const int& deeq_4,
        const int& it,
        const int& atom_start,
        const int& atom_offset_in_type,
        const int& atom_count,
        const int& nproj,
        const FPTYPE* d_wg,
        const bool& occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const std::complex<FPTYPE>* deeq_nc,
        const std::complex<FPTYPE>* becp,
        std::complex<FPTYPE>* r_chunk)
{
    const int total = chunk_nkb * nbands_occ * 2;
    const int blocks = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    build_stress_nl_reordered_r_chunk_nc<FPTYPE><<<blocks, THREADS_PER_BLOCK>>>(
        chunk_nkb,
        nbands_occ,
        deeq_2,
        deeq_3,
        deeq_4,
        it,
        atom_start,
        atom_offset_in_type,
        atom_count,
        nproj,
        d_wg,
        occ,
        d_ekb,
        qq_nt,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(deeq_nc),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
        reinterpret_cast<thrust::complex<FPTYPE>*>(r_chunk));

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void cal_stress_nl_reordered_chunk(const int ipol,
                                              const int jpol,
                                              const int npw,
                                              const int chunk_nkb,
                                              const thrust::complex<FPTYPE>* y_chunk,
                                              const thrust::complex<FPTYPE>* vkb_deri_chunk,
                                              FPTYPE* stress)
{
    __shared__ FPTYPE partial[THREADS_PER_BLOCK];
    FPTYPE local = 0.0;
    const long long total = static_cast<long long>(npw) * chunk_nkb;
    for (long long idx = threadIdx.x + static_cast<long long>(blockIdx.x) * blockDim.x;
         idx < total;
         idx += static_cast<long long>(blockDim.x) * gridDim.x)
    {
        const thrust::complex<FPTYPE> y = y_chunk[idx];
        const thrust::complex<FPTYPE> d = vkb_deri_chunk[idx];
        local -= (d * thrust::conj(y)).real();
    }
    partial[threadIdx.x] = local;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(stress + ipol * 3 + jpol, partial[0]);
    }
}

template <typename FPTYPE>
void cal_stress_nl_reordered_op<FPTYPE, base_device::DEVICE_GPU>::chunk(
        const base_device::DEVICE_GPU* ctx,
        const int& ipol,
        const int& jpol,
        const int& npw,
        const int& chunk_nkb,
        const std::complex<FPTYPE>* y_chunk,
        const std::complex<FPTYPE>* vkb_deri_chunk,
        FPTYPE* stress)
{
    const long long total = static_cast<long long>(npw) * chunk_nkb;
    const int blocks = std::min(4096LL, std::max(1LL, (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK));
    cal_stress_nl_reordered_chunk<FPTYPE><<<blocks, THREADS_PER_BLOCK>>>(ipol,
                                                                        jpol,
                                                                        npw,
                                                                        chunk_nkb,
                                                                        reinterpret_cast<const thrust::complex<FPTYPE>*>(y_chunk),
                                                                        reinterpret_cast<const thrust::complex<FPTYPE>*>(vkb_deri_chunk),
                                                                        stress);
    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void cal_stress_nl_chunk_nc(
        const int ipol,
        const int jpol,
        const int chunk_nkb,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int it,
        const int atom_start,
        const int atom_offset_in_type,
        const int atom_count,
        const int nproj,
        const FPTYPE *d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const thrust::complex<FPTYPE> *deeq_nc,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *stress)
{
    const int ib = blockIdx.x;
    const int ib2 = ib * 2;

    FPTYPE stress_var = 0;
    FPTYPE fac;
    if (occ)
    {
        fac = d_wg[ib];
    }
    else
    {
        fac = d_wg[0];
    }
    FPTYPE ekb_now = 0.0;
    if (d_ekb != nullptr)
    {
        ekb_now = d_ekb[ib];
    }
    const int chunk_type_start = atom_start - atom_offset_in_type;
    int sum = 0;
    for (int ia = 0; ia < atom_count; ia++)
    {
        for (int ii = threadIdx.x; ii < nproj * nproj; ii += blockDim.x) {
            const int ip1 = ii / nproj;
            const int ip2 = ii % nproj;
            const int deeq_iat = chunk_type_start + atom_offset_in_type + ia;
            thrust::complex<FPTYPE> ps_qq = 0;
            if (ekb_now != 0)
            {
                ps_qq = thrust::complex<FPTYPE>(-ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2],
                                                0.0);
            }
            const thrust::complex<FPTYPE> ps0 = deeq_nc[((0 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2]
                                                + ps_qq;
            const thrust::complex<FPTYPE> ps1 = deeq_nc[((1 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2];
            const thrust::complex<FPTYPE> ps2 = deeq_nc[((2 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2];
            const thrust::complex<FPTYPE> ps3 = deeq_nc[((3 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2]
                                                + ps_qq;
            const int inkb1 = sum + ip1;
            const int inkb2 = sum + ip2;
            const thrust::complex<FPTYPE> dbb0 = conj(dbecp[ib2 * chunk_nkb + inkb1])
                                                 * becp[ib2 * chunk_nkb + inkb2];
            const thrust::complex<FPTYPE> dbb1 = conj(dbecp[ib2 * chunk_nkb + inkb1])
                                                 * becp[(ib2 + 1) * chunk_nkb + inkb2];
            const thrust::complex<FPTYPE> dbb2 = conj(dbecp[(ib2 + 1) * chunk_nkb + inkb1])
                                                 * becp[ib2 * chunk_nkb + inkb2];
            const thrust::complex<FPTYPE> dbb3 = conj(dbecp[(ib2 + 1) * chunk_nkb + inkb1])
                                                 * becp[(ib2 + 1) * chunk_nkb + inkb2];
            stress_var -= fac * (ps0 * dbb0 + ps1 * dbb1 + ps2 * dbb2 + ps3 * dbb3).real();
        }
        sum += nproj;
    }
    __syncwarp();
    warp_reduce(stress_var);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(stress + ipol * 3 + jpol, stress_var);
    }
}

template <typename FPTYPE>
void cal_stress_nl_op<FPTYPE, base_device::DEVICE_GPU>::chunk(const base_device::DEVICE_GPU* ctx,
                                                              const int& ipol,
                                                              const int& jpol,
                                                              const int& chunk_nkb,
                                                              const int& nbands_occ,
                                                              const int& deeq_2,
                                                              const int& deeq_3,
                                                              const int& deeq_4,
                                                              const int& it,
                                                              const int& atom_start,
                                                              const int& atom_offset_in_type,
                                                              const int& atom_count,
                                                              const int& nproj,
                                                              const FPTYPE* d_wg,
                                                              const bool& occ,
                                                              const FPTYPE* d_ekb,
                                                              const FPTYPE* qq_nt,
                                                              const std::complex<FPTYPE>* deeq_nc,
                                                              const std::complex<FPTYPE>* becp,
                                                              const std::complex<FPTYPE>* dbecp,
                                                              FPTYPE* stress)
{
    cal_stress_nl_chunk_nc<FPTYPE><<<nbands_occ, THREADS_PER_BLOCK>>>(
        ipol,
        jpol,
        chunk_nkb,
        deeq_2,
        deeq_3,
        deeq_4,
        it,
        atom_start,
        atom_offset_in_type,
        atom_count,
        nproj,
        d_wg,
        occ,
        d_ekb,
        qq_nt,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(deeq_nc),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
        stress);

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
__global__ void cal_stress_nl(
        const int ipol,
        const int jpol,
        const int nkb,
        const int ntype,
        const int deeq_2,
        const int deeq_3,
        const int deeq_4,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE *d_wg,
        const bool occ,
        const FPTYPE* d_ekb,
        const FPTYPE* qq_nt,
        const thrust::complex<FPTYPE> *deeq_nc,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *stress)
{
    const int ib = blockIdx.x / ntype; // index of loop-nbands
    const int ib2  = ib * 2;
    const int it = blockIdx.x % ntype; // index of loop-ntype

    int iat = 0; // calculate the begin of atomic index
    int sum = 0; // calculate the begin of atomic-orbital index
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    FPTYPE stress_var = 0;
    FPTYPE fac;
    if (occ)
    {
        fac = d_wg[ib];
    }
    else
    {
        fac = d_wg[0];
    }
    FPTYPE ekb_now = 0.0;
    if (d_ekb != nullptr)
    {
        ekb_now = d_ekb[ib];
    }
    const int nproj = atom_nh[it];
    for (int ia = 0; ia < atom_na[it]; ia++)
    {
        for (int ii = threadIdx.x; ii < nproj * nproj; ii += blockDim.x) {
            const int ip1 = ii / nproj;
	        const int ip2 = ii % nproj;
            thrust::complex<FPTYPE> ps_qq = 0;
            if(ekb_now != 0)
            {
                ps_qq = thrust::complex<FPTYPE>(- ekb_now * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2], 0.0);
            }
            const thrust::complex<FPTYPE> ps0 = deeq_nc[((iat + ia) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
            const thrust::complex<FPTYPE> ps1 = deeq_nc[((1 * deeq_2 + iat + ia) * deeq_3 + ip1) * deeq_4 + ip2];
            const thrust::complex<FPTYPE> ps2 = deeq_nc[((2 * deeq_2 + iat + ia) * deeq_3 + ip1) * deeq_4 + ip2];
            const thrust::complex<FPTYPE> ps3 = deeq_nc[((3 * deeq_2 + iat + ia) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq;
            const int inkb1 = sum + ip1;
            const int inkb2 = sum + ip2;
            //out<<"\n ps = "<<ps;
            const thrust::complex<FPTYPE> dbb0 = conj(dbecp[ib2 * nkb + inkb1]) * becp[ib2 * nkb + inkb2];
            const thrust::complex<FPTYPE> dbb1 = conj(dbecp[ib2 * nkb + inkb1]) * becp[(ib2+1) * nkb + inkb2];
            const thrust::complex<FPTYPE> dbb2 = conj(dbecp[(ib2+1) * nkb + inkb1]) * becp[ib2 * nkb + inkb2];
            const thrust::complex<FPTYPE> dbb3 = conj(dbecp[(ib2+1) * nkb + inkb1]) * becp[(ib2+1) * nkb + inkb2];
            stress_var -= fac * (ps0 * dbb0 + ps1 * dbb1 + ps2 * dbb2 + ps3 * dbb3).real();
        }
        ++iat;
        sum+=nproj;
    }//ia
    __syncwarp();
    warp_reduce(stress_var);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(stress + ipol * 3 + jpol, stress_var);
    }
}

template <typename FPTYPE>
void cal_stress_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                   const int& ipol,
                                                                   const int& jpol,
                                                                   const int& nkb,
                                                                   const int& nbands_occ,
                                                                   const int& ntype,
                                                                   const int& deeq_2,
                                                                   const int& deeq_3,
                                                                   const int& deeq_4,
                                                                   const int* atom_nh,
                                                                   const int* atom_na,
                                                                   const FPTYPE* d_wg,
                                                                   const bool& occ,
                                                                   const FPTYPE* d_ekb,
                                                                   const FPTYPE* qq_nt,
                                                                   const std::complex<FPTYPE>* deeq_nc,
                                                                   const std::complex<FPTYPE>* becp,
                                                                   const std::complex<FPTYPE>* dbecp,
                                                                   FPTYPE* stress)
{
     cal_stress_nl<FPTYPE><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
             ipol,
             jpol,
             nkb,
             ntype,
             deeq_2,
             deeq_3,
             deeq_4,
             atom_nh,
             atom_na,
             d_wg,
             occ,
             d_ekb,
             qq_nt,
             reinterpret_cast<const thrust::complex<FPTYPE>*>(deeq_nc),
             reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
             reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
             stress);// array of data

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
FPTYPE cal_multi_dot_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const int& npw,
                                                                   const FPTYPE& fac,
                                                                   const FPTYPE* gk1,
                                                                   const FPTYPE* gk2,
                                                                   const FPTYPE* d_kfac,
                                                                   const std::complex<FPTYPE>* psi)
{
    FPTYPE* d_sum = nullptr;
    cudaMalloc(&d_sum, sizeof(FPTYPE) * 1);
    cudaMemset(d_sum, 0, sizeof(FPTYPE) * 1);
    int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cal_multi_dot<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npw, fac, gk1, gk2, d_kfac, reinterpret_cast<const thrust::complex<FPTYPE>*>(psi), d_sum);
    FPTYPE sum;
    cudaMemcpy(&sum, d_sum, sizeof(FPTYPE) * 1, cudaMemcpyDeviceToHost);
    cudaFree(d_sum);

    CHECK_CUDA_SYNC();
    return sum;
}

template <typename FPTYPE>
void cal_kinetic_stress_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                        const int& npw,
                                                                        const int& npwk_max,
                                                                        const int& npol,
                                                                        const int& nbands,
                                                                        const FPTYPE* band_weight,
                                                                        const bool& occ,
                                                                        const FPTYPE& k_weight,
                                                                        const FPTYPE* gk,
                                                                        const FPTYPE* kfac,
                                                                        const std::complex<FPTYPE>* psi,
                                                                        FPTYPE* stress)
{
    cudaMemset(stress, 0, sizeof(FPTYPE) * 9);
    if (npw == 0 || nbands == 0)
    {
        return;
    }
    dim3 grid(6, nbands);
    cal_kinetic_stress<FPTYPE><<<grid, THREADS_PER_BLOCK>>>(npw,
                                                            npwk_max,
                                                            npol,
                                                            nbands,
                                                            band_weight,
                                                            occ,
                                                            k_weight,
                                                            gk,
                                                            kfac,
                                                            reinterpret_cast<const thrust::complex<FPTYPE>*>(psi),
                                                            stress);

    CHECK_CUDA_SYNC();
}

template <typename T, typename Device>
void cal_stress_mgga_op<T, Device>::operator()(
    const int& spin,
    const int& nrxx,
    const Real& w1,
    const T * gradwfc,
    Real * crosstaus)
{
    auto gradwfc_ = reinterpret_cast<const thrust::complex<Real>*>(gradwfc);
    const int block = (nrxx + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cal_stress_mgga<Real><<<block, THREADS_PER_BLOCK>>>(
        spin, nrxx, w1, gradwfc_, crosstaus);

    CHECK_CUDA_SYNC();
}




template <typename FPTYPE>
__global__ void cal_vkb(
    const int npw,
    const int* indexes,
    const FPTYPE* vqs_in,
    const FPTYPE* ylms_in,
    const thrust::complex<FPTYPE>* sk_in,
    const thrust::complex<FPTYPE>* pref_in,
    thrust::complex<FPTYPE>* vkbs_out
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ih =  blockIdx.y;

    thrust::complex<FPTYPE>* vkb_ptr = vkbs_out + ih * npw;
    const FPTYPE* ylm_ptr = ylms_in + indexes[ih*4] * npw;
    const FPTYPE* vq_ptr = vqs_in + indexes[ih*4+1] * npw;
    if(idx<npw) vkb_ptr[idx] = ylm_ptr[idx] * vq_ptr[idx] * sk_in[idx] * pref_in[ih];              
    
}

template <typename FPTYPE>
__global__ void cal_vkb_deri(
        const int npw,
        const int ipol,
        const int jpol,
        const int* indexes,
        const FPTYPE* vqs_in, const FPTYPE* vqs_deri_in,
        const FPTYPE* ylms_in, const FPTYPE* ylms_deri_in,
        const thrust::complex<FPTYPE>* sk_in,
        const thrust::complex<FPTYPE>* pref_in,
        const FPTYPE* gk_in,
        thrust::complex<FPTYPE>* vkbs_out
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ih =  blockIdx.y;

    thrust::complex<FPTYPE>* vkb_ptr = vkbs_out + ih * npw;
    const FPTYPE* ylm_ptr = ylms_in + indexes[ih*4] * npw;
    const FPTYPE* vq_ptr = vqs_in + indexes[ih*4 + 1] * npw;

    const FPTYPE* ylm_deri_ptr1 = ylms_deri_in + indexes[ih*4+2] * npw;
    const FPTYPE* ylm_deri_ptr2 = ylms_deri_in + indexes[ih*4+3] * npw;
    const FPTYPE* vq_deri_ptr = vqs_deri_in + indexes[ih*4+1] * npw;
    const FPTYPE* gkn = &gk_in[4 * npw];
    const FPTYPE* gk = &gk_in[idx * 3];

    if(idx<npw) {
        vkb_ptr[idx] = thrust::complex<FPTYPE>(0.0, 0.0);
        if(ipol == jpol)
        {
            vkb_ptr[idx] -= ylm_ptr[idx] * vq_ptr[idx] * sk_in[idx] * pref_in[ih];
        }
        vkb_ptr[idx] -= (gk[ipol] * ylm_deri_ptr2[idx] 
                        + gk[jpol] * ylm_deri_ptr1[idx]) 
                        * vq_ptr[idx] * sk_in[idx] * pref_in[ih];

        vkb_ptr[idx] -= 2.0 * ylm_ptr[idx] * vq_deri_ptr[idx] * sk_in[idx] * pref_in[ih]
                    * gk[ipol] * gk[jpol] * gkn[idx];  
    }
}


template <typename FPTYPE>
__global__ void cal_vq(
        const FPTYPE* tab,
        int it, const FPTYPE* gk, int npw,
        const int tab_2,const int tab_3,  const FPTYPE table_interval, 
        const int nbeta, FPTYPE* vq
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ib =  blockIdx.y;

    FPTYPE* vq_ptr = &vq[ib * npw];
    const FPTYPE* gnorm = &gk[3 * npw];
    if(idx<npw) vq_ptr[idx] = _polynomial_interpolation(
        tab, it, ib, tab_2, tab_3, table_interval, gnorm[idx]);
}

template <typename FPTYPE>
__global__ void cal_vq_deri(
        const FPTYPE* tab,
        int it, const FPTYPE* gk, int npw,
        const int tab_2,const int tab_3,  const FPTYPE table_interval, 
        const int nbeta, FPTYPE* vq
){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int ib =  blockIdx.y;

    FPTYPE* vq_ptr = &vq[ib * npw];
    const FPTYPE* gnorm = &gk[3 * npw];
    if(idx<npw) vq_ptr[idx] = _polynomial_interpolation_nl(
        tab, it, ib, tab_2, tab_3, table_interval, gnorm[idx]);
}

template <typename FPTYPE>
__global__ void prepare_ylm_deri_g(
        const int npw,
        const int ipol,
        const int sign,
        const FPTYPE* gk,
        FPTYPE* shifted_gk)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= npw)
    {
        return;
    }

    const int base = idx * 3;
    const FPTYPE gx = gk[base];
    const FPTYPE gy = gk[base + 1];
    const FPTYPE gz = gk[base + 2];
    const FPTYPE dg = static_cast<FPTYPE>(1e-6) * sqrt(gx * gx + gy * gy + gz * gz);

    shifted_gk[base] = gx;
    shifted_gk[base + 1] = gy;
    shifted_gk[base + 2] = gz;
    shifted_gk[base + ipol] += static_cast<FPTYPE>(sign) * dg;
}

template <typename FPTYPE>
__global__ void cal_ylm_deri_from_pm(
        const int npw,
        const int nylm,
        const FPTYPE* gk,
        const FPTYPE* ylm_plus,
        const FPTYPE* ylm_minus,
        FPTYPE* ylm_deri)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int size = npw * nylm;
    if (idx >= size)
    {
        return;
    }

    const int ig = idx % npw;
    const int base = ig * 3;
    const FPTYPE gx = gk[base];
    const FPTYPE gy = gk[base + 1];
    const FPTYPE gz = gk[base + 2];
    const FPTYPE dg = static_cast<FPTYPE>(1e-6) * sqrt(gx * gx + gy * gy + gz * gz);
    const FPTYPE scale = dg > static_cast<FPTYPE>(1e-15) ? static_cast<FPTYPE>(0.5) / dg : static_cast<FPTYPE>(0.0);
    ylm_deri[idx] = (ylm_plus[idx] - ylm_minus[idx]) * scale;
}

template <typename FPTYPE>
__global__ void cal_stress_drhoc_aux0(
        const FPTYPE* r, const FPTYPE* rhoc, 
        const FPTYPE *gx_arr, const FPTYPE *rab, FPTYPE *drhocg, 
        const int mesh, const int igl0, const int ngg, const double omega
){
    const double FOUR_PI =  4.0 * 3.14159265358979323846;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= ngg) {return;}

    FPTYPE rhocg1=0.0;
    FPTYPE gx = gx_arr[idx];

    auto aux = [](FPTYPE r, FPTYPE rhoc, FPTYPE gx, FPTYPE rab) -> FPTYPE{
        return r * rhoc * (r * cos (gx * r) / gx - sin (gx * r) / (gx * gx)) * rab;
    };

    FPTYPE f_0 = aux(r[0],rhoc[0], gx, rab[0]);
    for( int ir = 1 ; ir< mesh - 2; ir+=2)
    {
        rhocg1 += 2 * aux(r[ir],rhoc[ir], gx, rab[ir]) + aux(r[ir+1],rhoc[ir+1], gx, rab[ir+1]);
    }//ir
    FPTYPE f_2 = aux(r[mesh - 2],rhoc[mesh - 2], gx, rab[mesh - 2]);
    FPTYPE f_1 = aux(r[mesh - 1],rhoc[mesh - 1], gx, rab[mesh - 1]);

    rhocg1 += f_2+f_2;
    rhocg1 += rhocg1;
    rhocg1 += f_0 + f_1;
    rhocg1/=3.0;

    drhocg [idx] = FOUR_PI / omega * rhocg1;
}

template <typename FPTYPE>
__global__ void cal_stress_drhoc_aux1(
        const FPTYPE* r, const FPTYPE* rhoc, 
        const FPTYPE *gx_arr, const FPTYPE *rab, FPTYPE *drhocg, 
        const int mesh, const int igl0, const int ngg, const double omega
){
    const double FOUR_PI =  4.0 * 3.14159265358979323846;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= ngg) {return;}

    FPTYPE rhocg1=0.0;
    FPTYPE gx = gx_arr[idx];

    auto aux = [](FPTYPE r, FPTYPE rhoc, FPTYPE gx, FPTYPE rab) -> FPTYPE{
        return sin (gx * r) / (gx * r) * r * r * rhoc * rab;
    };

    FPTYPE f_0 = r[0] * r[0] * rhoc[0] * rab[0];
    for( int ir = 1 ; ir< mesh - 2; ir+=2)
    {
        rhocg1 += 2 * aux(r[ir],rhoc[ir], gx, rab[ir]) + aux(r[ir+1],rhoc[ir+1], gx, rab[ir+1]);
    }//ir
    
    FPTYPE f_2 = aux(r[mesh - 2],rhoc[mesh - 2], gx, rab[mesh - 2]);
    FPTYPE f_1 = aux(r[mesh - 1],rhoc[mesh - 1], gx, rab[mesh - 1]);

    rhocg1 += f_2+f_2;
    rhocg1 += rhocg1;
    rhocg1 += f_0 + f_1;
    rhocg1/=3.0;

    drhocg [idx] = FOUR_PI * rhocg1 / omega;
}


template <typename FPTYPE>
__global__ void cal_stress_drhoc_aux2(
        const FPTYPE* r, const FPTYPE* rhoc, 
        const FPTYPE *gx_arr, const FPTYPE *rab, FPTYPE *drhocg, 
        const int mesh, const int igl0, const int ngg, const double omega
){


    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= ngg) {return;}

    FPTYPE rhocg1=0.0;
    FPTYPE gx = gx_arr[idx];    

    auto aux = [](FPTYPE r, FPTYPE rhoc, FPTYPE gx, FPTYPE rab) -> FPTYPE{
        return r < 1.0e-8 ? rab * rhoc : rab * rhoc * sin(gx * r) / (gx * r);
    };


    FPTYPE f_0 = r[0] * r[0] * rhoc[0] * rab[0];
    for( int ir = 1 ; ir< mesh - 2; ir+=2)
    {
        rhocg1 += 2 * aux(r[ir],rhoc[ir], gx, rab[ir]) + aux(r[ir+1],rhoc[ir+1], gx, rab[ir+1]);
    }//ir
    FPTYPE f_2 = aux(r[mesh - 2],rhoc[mesh - 2], gx, rab[mesh - 2]);
    FPTYPE f_1 = aux(r[mesh - 1],rhoc[mesh - 1], gx, rab[mesh - 1]);
    
    rhocg1 += f_2+f_2;
    rhocg1 += rhocg1;
    rhocg1 += f_0 + f_1;
    rhocg1/=3.0;

    drhocg [idx] = rhocg1;
}


template <typename FPTYPE>
__global__ void cal_stress_drhoc_aux3(
        const FPTYPE* r, const FPTYPE* rhoc, 
        const FPTYPE *gx_arr, const FPTYPE *rab, FPTYPE *drhocg, 
        const int mesh, const int igl0, const int ngg, const double omega
){
    const double FOUR_PI =  4.0 * 3.14159265358979323846;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx >= ngg) {return;}

    FPTYPE rhocg1=0.0;
    FPTYPE gx = gx_arr[idx];    
    const FPTYPE pow_gx = gx * gx;

    auto aux = [](FPTYPE r, FPTYPE rhoc, FPTYPE gx, FPTYPE rab) -> FPTYPE{
        return rab * rhoc * (r * cos(gx * r)/gx - sin(gx * r)/(gx * gx));
    };

    FPTYPE f_0 = r[0] * r[0] * rhoc[0] * rab[0];
    for( int ir = 1 ; ir< mesh - 2; ir+=2)
    {
        rhocg1 += 2 * aux(r[ir],rhoc[ir], gx, rab[ir]) + aux(r[ir+1],rhoc[ir+1], gx, rab[ir+1]);
    }//ir
    FPTYPE f_2 = aux(r[mesh - 2],rhoc[mesh - 2], gx, rab[mesh - 2]);
    FPTYPE f_1 = aux(r[mesh - 1],rhoc[mesh - 1], gx, rab[mesh - 1]);
    
    rhocg1 += f_2+f_2;
    rhocg1 += rhocg1;
    rhocg1 += f_0 + f_1;
    rhocg1/=3.0;

    // calculations after Simpson Integral
    const double g2a = pow_gx / 4.0;
    rhocg1 *= FOUR_PI / omega / 2.0 / gx;
    rhocg1 += FOUR_PI / omega * gx_arr[ngg] * exp(-g2a) * (g2a + 1) / (pow_gx*pow_gx);
    drhocg [idx] = rhocg1;
}



template <typename FPTYPE>
__global__ void cal_force_npw(
        const thrust::complex<FPTYPE> *psiv,
        const FPTYPE* gv,
        const FPTYPE* rhocgigg_vec,
        FPTYPE* force,
        const FPTYPE* tau,
        const int npw,
        const FPTYPE omega, const FPTYPE tpiba
){
    int ia = blockIdx.x;
    int tid = threadIdx.x;
    if(tid > npw) return;

    FPTYPE pos_x = tau[ia * 3];
    FPTYPE pos_y = tau[ia * 3 + 1];
    FPTYPE pos_z = tau[ia * 3 + 2];
    FPTYPE t_force0 = 0;
    FPTYPE t_force1 = 0;
    FPTYPE t_force2 = 0;
    for(int ig = tid; ig<npw;ig += blockDim.x) {
        const thrust::complex<FPTYPE> psiv_conj = conj(psiv[ig]);

        const FPTYPE arg = ModuleBase::TWO_PI * (gv[ig * 3] * pos_x + gv[ig * 3 + 1] * pos_y + gv[ig * 3 + 2] * pos_z);
        FPTYPE sinp, cosp;
        sincos(arg, &sinp, &cosp);
        const thrust::complex<FPTYPE> expiarg = thrust::complex<FPTYPE>(sinp, cosp);

        const thrust::complex<FPTYPE> tmp_var = psiv_conj * expiarg * tpiba * omega * rhocgigg_vec[ig];

        const thrust::complex<FPTYPE> ipol0 = tmp_var * gv[ig * 3];
        t_force0 += ipol0.real();

        const thrust::complex<FPTYPE> ipol1 = tmp_var * gv[ig * 3 + 1];
        t_force1 += ipol1.real();

        const thrust::complex<FPTYPE> ipol2 = tmp_var * gv[ig * 3 + 2];
        t_force2 += ipol2.real();
    }
    __syncwarp();
    warp_reduce(t_force0);
    warp_reduce(t_force1);
    warp_reduce(t_force2);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(&force[ia * 3], t_force0);
        atomicAdd(&force[ia * 3 + 1], t_force1);
        atomicAdd(&force[ia * 3 + 2], t_force2);
    }
}

template <typename FPTYPE>
void cal_vkb_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nh,
        const int npw,
        const int* indexes,
        const FPTYPE* vqs_in,
        const FPTYPE* ylms_in,
        const std::complex<FPTYPE>* sk_in,
        const std::complex<FPTYPE>* pref_in,
        std::complex<FPTYPE>* vkbs_out
    )
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 gridsize(block,nh);

    cal_vkb<FPTYPE><<<gridsize,THREADS_PER_BLOCK>>>(
        npw, indexes, vqs_in, ylms_in,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(sk_in), 
        reinterpret_cast<const thrust::complex<FPTYPE>*>(pref_in), 
        reinterpret_cast<thrust::complex<FPTYPE>*>(vkbs_out)
        
    );

}

template <typename FPTYPE>
void cal_vkb_deri_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU* ctx,
        const int nh,
        const int npw,
        const int ipol,
        const int jpol,
        const int* indexes,
        const FPTYPE* vqs_in,
        const FPTYPE* vqs_deri_in,
        const FPTYPE* ylms_in,
        const FPTYPE* ylms_deri_in,
        const std::complex<FPTYPE>* sk_in,
        const std::complex<FPTYPE>* pref_in,
        const FPTYPE* gk_in,
        std::complex<FPTYPE>* vkbs_out)
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 gridsize(block,nh);

    cal_vkb_deri<FPTYPE><<<gridsize,THREADS_PER_BLOCK>>>(
        npw, ipol, jpol, indexes,
        vqs_in, vqs_deri_in, ylms_in, ylms_deri_in,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(sk_in), 
        reinterpret_cast<const thrust::complex<FPTYPE>*>(pref_in),       
        gk_in,
        reinterpret_cast<thrust::complex<FPTYPE>*>(vkbs_out)
    );
}

template <typename FPTYPE>
void cal_vq_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU *ctx,
        const FPTYPE* tab,
        int it, const FPTYPE* gk, int npw,
        const int tab_2, const int tab_3, const FPTYPE table_interval, 
        const int nbeta, FPTYPE* vq
    )
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 gridsize(block,nbeta);

    cal_vq<FPTYPE><<<gridsize,THREADS_PER_BLOCK>>>(
        tab, it, gk, npw, tab_2, tab_3,
        table_interval, nbeta, vq
    );
}


template <typename FPTYPE>
void cal_vq_deri_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
        const base_device::DEVICE_GPU *ctx,
        const FPTYPE* tab,
        int it, const FPTYPE* gk, int npw,
        const int tab_2, const int tab_3, const FPTYPE table_interval, 
        const int nbeta, FPTYPE* vq
    )
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 gridsize(block,nbeta);

    cal_vq_deri<FPTYPE><<<gridsize,THREADS_PER_BLOCK>>>(
        tab, it, gk, npw, tab_2, tab_3,
        table_interval, nbeta, vq
    );

    return ;
}

template <typename FPTYPE>
void cal_ylm_deri_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                  const int& nylm,
                                                                  const int& npw,
                                                                  const FPTYPE* gk,
                                                                  FPTYPE* ylm_deri)
{
    const int lmax = static_cast<int>(sqrt(static_cast<double>(nylm))) - 1;
    if ((lmax + 1) * (lmax + 1) != nylm)
    {
        return;
    }

    FPTYPE* shifted_gk = nullptr;
    FPTYPE* ylm_scratch = nullptr;
    FPTYPE* ylm_plus = nullptr;
    FPTYPE* ylm_minus = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&shifted_gk), static_cast<size_t>(npw) * 3 * sizeof(FPTYPE)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ylm_scratch), static_cast<size_t>(nylm) * npw * sizeof(FPTYPE)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ylm_plus), static_cast<size_t>(nylm) * npw * sizeof(FPTYPE)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ylm_minus), static_cast<size_t>(nylm) * npw * sizeof(FPTYPE)));

    const int g_block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const int ylm_size = npw * nylm;
    const int ylm_block = (ylm_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    for (int ipol = 0; ipol < 3; ++ipol)
    {
        prepare_ylm_deri_g<FPTYPE><<<g_block, THREADS_PER_BLOCK>>>(npw, ipol, 1, gk, shifted_gk);
        ModuleBase::cal_ylm_real_op<FPTYPE, base_device::DEVICE_GPU>()(ctx,
                                                                       npw,
                                                                       lmax,
                                                                       ModuleBase::SQRT2,
                                                                       ModuleBase::PI,
                                                                       ModuleBase::PI_HALF,
                                                                       ModuleBase::FOUR_PI,
                                                                       ModuleBase::SQRT_INVERSE_FOUR_PI,
                                                                       shifted_gk,
                                                                       ylm_scratch,
                                                                       ylm_plus);
        prepare_ylm_deri_g<FPTYPE><<<g_block, THREADS_PER_BLOCK>>>(npw, ipol, -1, gk, shifted_gk);
        ModuleBase::cal_ylm_real_op<FPTYPE, base_device::DEVICE_GPU>()(ctx,
                                                                       npw,
                                                                       lmax,
                                                                       ModuleBase::SQRT2,
                                                                       ModuleBase::PI,
                                                                       ModuleBase::PI_HALF,
                                                                       ModuleBase::FOUR_PI,
                                                                       ModuleBase::SQRT_INVERSE_FOUR_PI,
                                                                       shifted_gk,
                                                                       ylm_scratch,
                                                                       ylm_minus);
        cal_ylm_deri_from_pm<FPTYPE><<<ylm_block, THREADS_PER_BLOCK>>>(npw,
                                                                       nylm,
                                                                       gk,
                                                                       ylm_plus,
                                                                       ylm_minus,
                                                                       ylm_deri + ipol * ylm_size);
    }

    CHECK_CUDA(cudaFree(shifted_gk));
    CHECK_CUDA(cudaFree(ylm_scratch));
    CHECK_CUDA(cudaFree(ylm_plus));
    CHECK_CUDA(cudaFree(ylm_minus));
    CHECK_CUDA_SYNC();
}


/**
 * The implementation of this operator is detailed in stress_op.h.
 * The function is called by the module as follows
 *      Type = 0 -> stress_cc
 *      Type = 1 -> stress_cc, force_cc
 *      Type = 2 -> force_scc
 *      Type = 3 -> stress_loc
 */
template <typename FPTYPE>
void cal_stress_drhoc_aux_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
        const FPTYPE* r, const FPTYPE* rhoc,  
        const FPTYPE *gx_arr, const FPTYPE *rab, FPTYPE *drhocg, 
        const int mesh, const int igl0, const int ngg, const double omega,
        int type
    )
{
    const int block = (ngg + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    if(type == 0) {
        cal_stress_drhoc_aux0<FPTYPE><<<block,THREADS_PER_BLOCK>>>(
            r,rhoc,gx_arr,rab,drhocg,mesh,igl0,ngg,omega
        );
    } else if(type == 1 ){
        cal_stress_drhoc_aux1<FPTYPE><<<block,THREADS_PER_BLOCK>>>(
            r,rhoc,gx_arr,rab,drhocg,mesh,igl0,ngg,omega
        );        
    } else if(type == 2 ){
        cal_stress_drhoc_aux2<FPTYPE><<<block,THREADS_PER_BLOCK>>>(
            r,rhoc,gx_arr,rab,drhocg,mesh,igl0,ngg,omega
        );        
    } else if(type == 3 ){
        cal_stress_drhoc_aux3<FPTYPE><<<block,THREADS_PER_BLOCK>>>(
            r,rhoc,gx_arr,rab,drhocg,mesh,igl0,ngg,omega
        );        
    }
    return ;
}


template <typename FPTYPE>
void cal_force_npw_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
        const std::complex<FPTYPE> *psiv,
        const FPTYPE* gv,
        const FPTYPE* rhocgigg_vec,
        FPTYPE* force,
        const FPTYPE* tau,
        const int npw,
        const FPTYPE omega, const FPTYPE tpiba, const int na
    )
{
    cal_force_npw <<<na, THREADS_PER_BLOCK >>> (
        reinterpret_cast<const thrust::complex<FPTYPE>*>(psiv),
        gv, rhocgigg_vec, force, tau,
        npw, omega, tpiba
    ); 
    return ;
}


template <>
void pointer_array_malloc<base_device::DEVICE_GPU>::operator()(
    void **ptr,
    const int n
)
{
    CHECK_CUDA(cudaMalloc(ptr, n * sizeof(void*)));
}

template struct pointer_array_malloc<base_device::DEVICE_GPU>;

template <>
void synchronize_ptrs<base_device::DEVICE_GPU>::operator()(
    void **ptr_out,
    const void **ptr_in,
    const int size)
{
    cudaMemcpy(ptr_out, ptr_in, sizeof(void*) * size, cudaMemcpyHostToDevice);
}

template <typename FPTYPE, int npol>
__global__ void cal_stress_onsite(
        const int nkb,
        const int ntype,
        const int wg_nc,
        const int ik,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE *d_wg,
        const thrust::complex<FPTYPE> *vu,
        const int* orbital_corr,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *stress)
{
    const int ib = blockIdx.x / ntype;
    const int it = blockIdx.x % ntype;
    if(orbital_corr[it] == -1) return;
    const int orbital_l = orbital_corr[it];
    const int ip_begin = orbital_l * orbital_l;
    const int tlp1 = 2 * orbital_l + 1;
    const int tlp1_2 = tlp1 * tlp1;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
        vu += npol * npol * tlp1_2 * atom_na[ii];
    }

    FPTYPE stress_var = 0;
    const FPTYPE fac = d_wg[ik * wg_nc + ib];
    const int nprojs = atom_nh[it];
    const int ib2 = ib * npol;
    for (int ia = 0; ia < atom_na[it]; ia++)
    {
        for (int mm = threadIdx.x; mm < tlp1_2; mm += blockDim.x) {
            const int m1 = mm / tlp1;
            const int m2 = mm % tlp1;
            const int ip1 = ip_begin + m1;
            const int ip2 = ip_begin + m2;
            const int inkb1 = sum + ip1 + ib2 * nkb;
            const int inkb2 = sum + ip2 + ib2 * nkb;
            if (npol == 2)
            {
                thrust::complex<FPTYPE> ps[4] = {vu[mm], vu[mm + tlp1_2], vu[mm + 2 * tlp1_2], vu[mm + 3 * tlp1_2]};
                const thrust::complex<FPTYPE> dbb0 = conj(dbecp[inkb1]) * becp[inkb2];
                const thrust::complex<FPTYPE> dbb1 = conj(dbecp[inkb1]) * becp[inkb2 + nkb];
                const thrust::complex<FPTYPE> dbb2 = conj(dbecp[inkb1 + nkb]) * becp[inkb2];
                const thrust::complex<FPTYPE> dbb3 = conj(dbecp[inkb1 + nkb]) * becp[inkb2 + nkb];
                stress_var -= fac * (ps[0] * dbb0 + ps[1] * dbb1 + ps[2] * dbb2 + ps[3] * dbb3).real();
            }
            else
            {
                stress_var -= fac * (vu[mm] * (conj(dbecp[inkb1]) * becp[inkb2])).real();
            }
        }
        ++iat;
        sum+=nprojs;
        vu += npol * npol * tlp1_2;
    }//ia
    __syncwarp();
    warp_reduce(stress_var);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(stress, stress_var);
    }
}

template <typename FPTYPE, int npol>
__global__ void cal_stress_onsite(
        const int nkb,
        const int ntype,
        const int wg_nc,
        const int ik,
        int spin_sign,
        const int *atom_nh,
        const int *atom_na,
        const FPTYPE *d_wg,
        const double* lambda,
        const thrust::complex<FPTYPE> *becp,
        const thrust::complex<FPTYPE> *dbecp,
        FPTYPE *stress)
{
    const int ib = blockIdx.x / ntype;
    const int it = blockIdx.x % ntype;

    int iat = 0;
    int sum = 0;
    for (int ii = 0; ii < it; ii++) {
        iat += atom_na[ii];
        sum += atom_na[ii] * atom_nh[ii];
    }

    FPTYPE stress_var = 0;
    const FPTYPE fac = d_wg[ik * wg_nc + ib];
    const int nprojs = atom_nh[it];
    const int ib2 = ib * npol;
    for (int ia = 0; ia < atom_na[it]; ia++)
    {
        if (npol == 2)
        {
            const thrust::complex<FPTYPE> coefficients0(lambda[iat*3+2], 0.0);
            const thrust::complex<FPTYPE> coefficients1(lambda[iat*3] , lambda[iat*3+1]);
            const thrust::complex<FPTYPE> coefficients2(lambda[iat*3] , -1 * lambda[iat*3+1]);
            const thrust::complex<FPTYPE> coefficients3(-1 * lambda[iat*3+2], 0.0);
            for (int ip = threadIdx.x; ip < nprojs; ip += blockDim.x) {
                const int inkb1 = sum + ip + ib2 * nkb;
                const thrust::complex<FPTYPE> dbb0 = conj(dbecp[inkb1]) * becp[inkb1];
                const thrust::complex<FPTYPE> dbb1 = conj(dbecp[inkb1]) * becp[nkb + inkb1];
                const thrust::complex<FPTYPE> dbb2 = conj(dbecp[nkb + inkb1]) * becp[inkb1];
                const thrust::complex<FPTYPE> dbb3 = conj(dbecp[nkb + inkb1]) * becp[nkb + inkb1];
                stress_var -= fac * (coefficients0 * dbb0 + coefficients1 * dbb1 + coefficients2 * dbb2 + coefficients3 * dbb3).real();
            }
        }
        else
        {
            const FPTYPE lambda_z = lambda[iat*3+2] * spin_sign;
            for (int ip = threadIdx.x; ip < nprojs; ip += blockDim.x) {
                const int inkb = sum + ip + ib2 * nkb;
                const FPTYPE dbb = (conj(dbecp[inkb]) * becp[inkb]).real();
                stress_var -= fac * lambda_z * dbb;
            }
        }
        ++iat;
        sum+=nprojs;
    }//ia
    __syncwarp();
    warp_reduce(stress_var);
    if (threadIdx.x % WARP_SIZE == 0) {
        atomicAdd(stress, stress_var);
    }
}

//kernel for DFTU stress
template <typename FPTYPE>
void cal_stress_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                    const int& nkb,
                    const int& nbands_occ,
                    const int& ntype,
                    const int& wg_nc,
                    const int& ik,
                    const int& npol,
                    const int* atom_nh,
                    const int* atom_na,
                    const FPTYPE* d_wg,
                    const std::complex<FPTYPE>* vu,
                    const int* orbital_corr,
                    const std::complex<FPTYPE>* becp,
                    const std::complex<FPTYPE>* dbecp,
                    FPTYPE* stress)
{
    if (npol == 1)
    {
        cal_stress_onsite<FPTYPE, 1><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
                 nkb,
                 ntype,
                 wg_nc,
                 ik,
                 atom_nh,
                 atom_na,
                 d_wg,
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(vu),
                 orbital_corr,
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                 stress);
    }
    else
    {
        cal_stress_onsite<FPTYPE, 2><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
                 nkb,
                 ntype,
                 wg_nc,
                 ik,
                 atom_nh,
                 atom_na,
                 d_wg,
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(vu),
                 orbital_corr,
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                 stress);
    }

    CHECK_CUDA_SYNC();
}
// kernel for DeltaSpin stress
template <typename FPTYPE>
void cal_stress_nl_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                    const int& nkb,
                    const int& nbands_occ,
                    const int& ntype,
                    const int& wg_nc,
                    const int& ik,
                    const int& npol,
                    const int* atom_nh,
                    const int* atom_na,
                    const FPTYPE* d_wg,
                    const double* lambda,
                    const int* isk,
                    const std::complex<FPTYPE>* becp,
                    const std::complex<FPTYPE>* dbecp,
                    FPTYPE* stress)
{
    int spin_sign = 1;
    if (isk != nullptr && npol == 1)
    {
        spin_sign = (isk[ik] == 0) ? 1 : -1;
    }
    if (npol == 1)
    {
        cal_stress_onsite<FPTYPE, 1><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
                 nkb,
                 ntype,
                 wg_nc,
                 ik,
                 spin_sign,
                 atom_nh,
                 atom_na,
                 d_wg,
                 lambda,
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                 stress);
    }
    else
    {
        cal_stress_onsite<FPTYPE, 2><<<nbands_occ * ntype, THREADS_PER_BLOCK>>>(
                 nkb,
                 ntype,
                 wg_nc,
                 ik,
                 spin_sign,
                 atom_nh,
                 atom_na,
                 d_wg,
                 lambda,
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(becp),
                 reinterpret_cast<const thrust::complex<FPTYPE>*>(dbecp),
                 stress);
    }

    CHECK_CUDA_SYNC();
}

template <typename FPTYPE>
void cal_stress_ewa_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                    const int nat,
                                                                    const int npw,
                                                                    const int ig0,
                                                                    const int do_real_space,
                                                                    const int nm1,
                                                                    const int nm2,
                                                                    const int nm3,
                                                                    const FPTYPE alpha,
                                                                    const FPTYPE omega,
                                                                    const FPTYPE tpiba2,
                                                                    const FPTYPE lat0,
                                                                    const FPTYPE fact,
                                                                    const FPTYPE rmax,
                                                                    const FPTYPE charge,
                                                                    const FPTYPE* tau,
                                                                    const FPTYPE* atom_z,
                                                                    const FPTYPE* gcar,
                                                                    const FPTYPE* gg,
                                                                    const FPTYPE* latvec,
                                                                    FPTYPE* stress)
{
    if (nat <= 0 || npw <= 0)
    {
        return;
    }

    set_stress_ewa_diag_kernel<FPTYPE><<<1, 1>>>(charge, alpha, omega, stress);

    const int g_blocks = std::min(1024, std::max(1, (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK));
    FPTYPE* partial = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&partial), static_cast<size_t>(g_blocks) * 7 * sizeof(FPTYPE)));
    stress_ewa_g_kernel<FPTYPE><<<g_blocks, THREADS_PER_BLOCK>>>(nat,
                                                                 npw,
                                                                 ig0,
                                                                 alpha,
                                                                 omega,
                                                                 tpiba2,
                                                                 fact,
                                                                 tau,
                                                                 atom_z,
                                                                 gcar,
                                                                 gg,
                                                                 partial);
    stress_ewa_final_reduce_kernel<FPTYPE><<<1, THREADS_PER_BLOCK>>>(partial, g_blocks, stress);
    CHECK_CUDA(cudaFree(partial));

    if (do_real_space)
    {
        const long long npairs = static_cast<long long>(nat) * nat;
        const int r_blocks = std::min(1024,
                                      std::max(1, static_cast<int>((npairs + THREADS_PER_BLOCK - 1)
                                                                   / THREADS_PER_BLOCK)));
        partial = nullptr;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&partial), static_cast<size_t>(r_blocks) * 7 * sizeof(FPTYPE)));
        stress_ewa_r_kernel<FPTYPE><<<r_blocks, THREADS_PER_BLOCK>>>(nat,
                                                                     nm1,
                                                                     nm2,
                                                                     nm3,
                                                                     alpha,
                                                                     omega,
                                                                     lat0,
                                                                     rmax,
                                                                     tau,
                                                                     atom_z,
                                                                     latvec,
                                                                     partial);
        stress_ewa_final_reduce_kernel<FPTYPE><<<1, THREADS_PER_BLOCK>>>(partial, r_blocks, stress);
        CHECK_CUDA(cudaFree(partial));
    }

    CHECK_CUDA_SYNC();
}

template struct synchronize_ptrs<base_device::DEVICE_GPU>;

template struct cal_stress_mgga_op<std::complex<float>, base_device::DEVICE_GPU>;
template struct cal_stress_mgga_op<std::complex<double>, base_device::DEVICE_GPU>;

template struct cal_dbecp_noevc_nl_op<float, base_device::DEVICE_GPU>;
template struct cal_dbecp_noevc_nl_op<double, base_device::DEVICE_GPU>;

template struct cal_stress_nl_op<float, base_device::DEVICE_GPU>;
template struct cal_stress_nl_op<double, base_device::DEVICE_GPU>;

template struct build_stress_nl_reordered_r_op<float, base_device::DEVICE_GPU>;
template struct build_stress_nl_reordered_r_op<double, base_device::DEVICE_GPU>;

template struct cal_stress_nl_reordered_op<float, base_device::DEVICE_GPU>;
template struct cal_stress_nl_reordered_op<double, base_device::DEVICE_GPU>;

template struct cal_stress_ewa_op<float, base_device::DEVICE_GPU>;
template struct cal_stress_ewa_op<double, base_device::DEVICE_GPU>;


template struct cal_vq_op<double, base_device::DEVICE_GPU>;
template struct cal_vq_op<float, base_device::DEVICE_GPU>;

template struct cal_vq_deri_op<double, base_device::DEVICE_GPU>;
template struct cal_vq_deri_op<float, base_device::DEVICE_GPU>;

template struct cal_ylm_deri_op<double, base_device::DEVICE_GPU>;
template struct cal_ylm_deri_op<float, base_device::DEVICE_GPU>;

template struct cal_vkb_op<double, base_device::DEVICE_GPU>;
template struct cal_vkb_op<float, base_device::DEVICE_GPU>;

template struct cal_vkb_deri_op<double, base_device::DEVICE_GPU>;
template struct cal_vkb_deri_op<float, base_device::DEVICE_GPU>;

template struct cal_stress_drhoc_aux_op<double, base_device::DEVICE_GPU>;
template struct cal_stress_drhoc_aux_op<float, base_device::DEVICE_GPU>;

template struct cal_force_npw_op<double, base_device::DEVICE_GPU>;
template struct cal_force_npw_op<float, base_device::DEVICE_GPU>;

template struct cal_multi_dot_op<double, base_device::DEVICE_GPU>;
template struct cal_multi_dot_op<float, base_device::DEVICE_GPU>;

template struct cal_kinetic_stress_op<double, base_device::DEVICE_GPU>;
template struct cal_kinetic_stress_op<float, base_device::DEVICE_GPU>;

// template struct prepare_vkb_deri_ptr_op<double, base_device::DEVICE_GPU>;
// template struct prepare_vkb_deri_ptr_op<float, base_device::DEVICE_GPU>;
}  // namespace hamilt

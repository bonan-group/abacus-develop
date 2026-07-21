#include "source_pw/module_pwdft/kernels/vnl_op.h"
#include "vnl_tools_cu.hpp"

#include <complex>

#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <base/macros/macros.h>

#define THREADS_PER_BLOCK 256

namespace hamilt {

template<typename FPTYPE>
__global__ void cal_vnl(
    const int ntype,
    const int npw,
    const int npwx,
    const int nhm,
    const int tab_2,
    const int tab_3,
    const int * atom_na,
    const int * atom_nb,
    const int * atom_nh,
    const FPTYPE DQ,
    const FPTYPE tpiba,
    const thrust::complex<FPTYPE> NEG_IMAG_UNIT,
    const FPTYPE *gk,
    const FPTYPE *ylm,
    const FPTYPE *indv,
    const FPTYPE *nhtol,
    const FPTYPE *nhtolm,
    const FPTYPE *tab,
    FPTYPE *vkb1,
    const thrust::complex<FPTYPE> *sk,
    thrust::complex<FPTYPE> *vkb_in)
{
    FPTYPE vq = 0.0;
    int iat = 0, jkb = 0, ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw) {return;}
    for (int it = 0; it < ntype; it++) {
        // calculate beta in G-space using an interpolation table
        const int nh = atom_nh[it];
        const int nbeta = atom_nb[it];

        for (int nb = 0; nb < nbeta; nb++) {
            const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0] + gk[ig * 3 + 1] * gk[ig * 3 + 1] +
                                      gk[ig * 3 + 2] * gk[ig * 3 + 2]) * tpiba;

            vq = _polynomial_interpolation(
                    tab, it, nb, tab_2, tab_3, DQ, gnorm);

            // add spherical harmonic part
            for (int ih = 0; ih < nh; ih++) {
                if (nb == indv[it * nhm + ih]) {
                    const int lm = static_cast<int>(nhtolm[it * nhm + ih]);
                    vkb1[ih * npw + ig] = ylm[lm * npw + ig] * vq;
                }
            } // end ih
        } // end nbeta

        // vkb1 contains all betas including angular part for type nt
        // now add the structure factor and factor (-i)^l
        for (int ia = 0; ia < atom_na[it]; ia++) {
            for (int ih = 0; ih < nh; ih++) {
                thrust::complex<FPTYPE> pref = pow(NEG_IMAG_UNIT, nhtol[it * nhm + ih]);    //?
                thrust::complex<FPTYPE> *pvkb = vkb_in + jkb * npwx;
                pvkb[ig] = vkb1[ih * npw + ig] * sk[iat * npw + ig] * pref;
                ++jkb;
            } // end ih
            iat++;
        } // end ia
    } // enddo
}

template <typename FPTYPE>
void cal_vnl_op<FPTYPE, base_device::DEVICE_GPU>::operator() (
    const base_device::DEVICE_GPU *ctx,
    const int &ntype,
    const int &npw,
    const int &npwx,
    const int &nhm,
    const int &tab_2,
    const int &tab_3,
    const int * atom_na,
    const int * atom_nb,
    const int * atom_nh,
    const FPTYPE &DQ,
    const FPTYPE &tpiba,
    const std::complex<FPTYPE> &NEG_IMAG_UNIT,
    const FPTYPE *gk,
    const FPTYPE *ylm,
    const FPTYPE *indv,
    const FPTYPE *nhtol,
    const FPTYPE *nhtolm,
    const FPTYPE *tab,
    FPTYPE *vkb1,
    const std::complex<FPTYPE> *sk,
    std::complex<FPTYPE> *vkb_in)
{
    int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cal_vnl<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
            ntype, npw, npwx, nhm, tab_2, tab_3,
            atom_na, atom_nb, atom_nh,
            DQ, tpiba,
            static_cast<thrust::complex<FPTYPE>>(NEG_IMAG_UNIT),
            gk, ylm, indv, nhtol, nhtolm, tab, vkb1,
            reinterpret_cast<const thrust::complex<FPTYPE>*>(sk),
            reinterpret_cast<thrust::complex<FPTYPE>*>(vkb_in));

    CHECK_CUDA_SYNC();
}

template struct cal_vnl_op<float, base_device::DEVICE_GPU>;
template struct cal_vnl_op<double, base_device::DEVICE_GPU>;

template<typename FPTYPE>
__global__ void cal_vkb1_cache(const int ntype,
                               const int npw,
                               const int nhm,
                               const int tab_2,
                               const int tab_3,
                               const int* atom_nb,
                               const int* atom_nh,
                               const FPTYPE DQ,
                               const FPTYPE tpiba,
                               const FPTYPE* gk,
                               const FPTYPE* ylm,
                               const FPTYPE* indv,
                               const FPTYPE* nhtolm,
                               const FPTYPE* tab,
                               FPTYPE* vkb1_cache)
{
    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
    {
        return;
    }

    const FPTYPE gnorm = sqrt(gk[ig * 3 + 0] * gk[ig * 3 + 0]
                              + gk[ig * 3 + 1] * gk[ig * 3 + 1]
                              + gk[ig * 3 + 2] * gk[ig * 3 + 2])
                         * tpiba;

    for (int it = 0; it < ntype; ++it)
    {
        const int nh = atom_nh[it];
        const int nbeta = atom_nb[it];
        FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
        for (int nb = 0; nb < nbeta; ++nb)
        {
            const FPTYPE vq = _polynomial_interpolation(tab, it, nb, tab_2, tab_3, DQ, gnorm);
            for (int ih = 0; ih < nh; ++ih)
            {
                if (nb == indv[it * nhm + ih])
                {
                    const int lm = static_cast<int>(nhtolm[it * nhm + ih]);
                    vkb1_type[ih * npw + ig] = ylm[lm * npw + ig] * vq;
                }
            }
        }
    }
}

template <typename FPTYPE>
void cal_vkb1_cache_op<FPTYPE, base_device::DEVICE_GPU>::operator()(const base_device::DEVICE_GPU* ctx,
                                                                    const int& ntype,
                                                                    const int& npw,
                                                                    const int& nhm,
                                                                    const int& tab_2,
                                                                    const int& tab_3,
                                                                    const int* atom_nb,
                                                                    const int* atom_nh,
                                                                    const FPTYPE& DQ,
                                                                    const FPTYPE& tpiba,
                                                                    const FPTYPE* gk,
                                                                    const FPTYPE* ylm,
                                                                    const FPTYPE* indv,
                                                                    const FPTYPE* nhtolm,
                                                                    const FPTYPE* tab,
                                                                    FPTYPE* vkb1_cache)
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cal_vkb1_cache<FPTYPE><<<block, THREADS_PER_BLOCK>>>(ntype,
                                                         npw,
                                                         nhm,
                                                         tab_2,
                                                         tab_3,
                                                         atom_nb,
                                                         atom_nh,
                                                         DQ,
                                                         tpiba,
                                                         gk,
                                                         ylm,
                                                         indv,
                                                         nhtolm,
                                                         tab,
                                                         vkb1_cache);
    CHECK_CUDA_SYNC();
}

template<typename FPTYPE>
__global__ void cal_vnl_from_vkb1_cache(const int npw,
                                        const int npwx,
                                        const int nhm,
                                        const int* atom_nh,
                                        const int atom_start,
                                        const int atom_end,
                                        const FPTYPE* nhtol,
                                        const FPTYPE* vkb1_cache,
                                        const thrust::complex<FPTYPE>* sk_all,
                                        const int* iat2it,
                                        thrust::complex<FPTYPE>* vkb_out)
{
    const thrust::complex<FPTYPE> pref_tab[4] = {
        thrust::complex<FPTYPE>(1, 0),
        thrust::complex<FPTYPE>(0, -1),
        thrust::complex<FPTYPE>(-1, 0),
        thrust::complex<FPTYPE>(0, 1)};

    const int ig = blockIdx.x * blockDim.x + threadIdx.x;
    if (ig >= npw)
    {
        return;
    }

    int jkb_out = 0;
    for (int iat = atom_start; iat < atom_end; ++iat)
    {
        const int it = iat2it[iat];
        const int nh = atom_nh[it];
        const FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
        for (int ih = 0; ih < nh; ++ih)
        {
            const thrust::complex<FPTYPE> pref = pref_tab[int(nhtol[it * nhm + ih]) % 4];
            thrust::complex<FPTYPE>* pvkb = vkb_out + jkb_out * npwx;
            pvkb[ig] = vkb1_type[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
            ++jkb_out;
        }
    }
}

template <typename FPTYPE>
void cal_vnl_from_vkb1_cache_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int& npw,
    const int& npwx,
    const int& nhm,
    const int* atom_nh,
    const int& atom_start,
    const int& atom_end,
    const FPTYPE* nhtol,
    const FPTYPE* vkb1_cache,
    const std::complex<FPTYPE>* sk_all,
    const int* iat2it,
    std::complex<FPTYPE>* vkb_out)
{
    const int block = (npw + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cal_vnl_from_vkb1_cache<FPTYPE><<<block, THREADS_PER_BLOCK>>>(
        npw,
        npwx,
        nhm,
        atom_nh,
        atom_start,
        atom_end,
        nhtol,
        vkb1_cache,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(sk_all),
        iat2it,
        reinterpret_cast<thrust::complex<FPTYPE>*>(vkb_out));
    CHECK_CUDA_SYNC();
}

template struct cal_vkb1_cache_op<float, base_device::DEVICE_GPU>;
template struct cal_vkb1_cache_op<double, base_device::DEVICE_GPU>;
template struct cal_vnl_from_vkb1_cache_op<float, base_device::DEVICE_GPU>;
template struct cal_vnl_from_vkb1_cache_op<double, base_device::DEVICE_GPU>;

template<typename FPTYPE>
__global__ void cal_becp_from_vkb1_cache(const int npw,
                                         const int npwx,
                                         const int nbands,
                                         const int nkb,
                                         const int nhm,
                                         const int* jkb_to_iat,
                                         const int* jkb_to_it,
                                         const int* jkb_to_ih,
                                         const FPTYPE* jkb_pref_sign,
                                         const FPTYPE* vkb1_cache,
                                         const thrust::complex<FPTYPE>* sk_all,
                                         const thrust::complex<FPTYPE>* psi,
                                         thrust::complex<FPTYPE>* becp)
{
    extern __shared__ __align__(sizeof(thrust::complex<double>)) unsigned char shared_raw[];
    thrust::complex<FPTYPE>* partial = reinterpret_cast<thrust::complex<FPTYPE>*>(shared_raw);

    const int jkb = blockIdx.x;
    const int ib = blockIdx.y;
    const int iat = jkb_to_iat[jkb];
    const int it = jkb_to_it[jkb];
    const int ih = jkb_to_ih[jkb];
    const FPTYPE* vkb1_type = vkb1_cache + it * nhm * npw;
    const thrust::complex<FPTYPE> pref(jkb_pref_sign[2 * jkb], jkb_pref_sign[2 * jkb + 1]);

    thrust::complex<FPTYPE> sum(0, 0);
    for (int ig = threadIdx.x; ig < npw; ig += blockDim.x)
    {
        const thrust::complex<FPTYPE> vkb = vkb1_type[ih * npw + ig] * sk_all[iat * npw + ig] * pref;
        sum += thrust::conj(vkb) * psi[ib * npwx + ig];
    }
    partial[threadIdx.x] = sum;
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
        becp[ib * nkb + jkb] = partial[0];
    }
}

template <typename FPTYPE>
void cal_becp_from_vkb1_cache_op<FPTYPE, base_device::DEVICE_GPU>::operator()(
    const base_device::DEVICE_GPU* ctx,
    const int& npw,
    const int& npwx,
    const int& nbands,
    const int& nkb,
    const int& nhm,
    const int* jkb_to_iat,
    const int* jkb_to_it,
    const int* jkb_to_ih,
    const FPTYPE* jkb_pref_sign,
    const FPTYPE* vkb1_cache,
    const std::complex<FPTYPE>* sk_all,
    const std::complex<FPTYPE>* psi,
    std::complex<FPTYPE>* becp)
{
    const dim3 grid(nkb, nbands);
    const size_t shared_size = THREADS_PER_BLOCK * sizeof(thrust::complex<FPTYPE>);
    cal_becp_from_vkb1_cache<FPTYPE><<<grid, THREADS_PER_BLOCK, shared_size>>>(
        npw,
        npwx,
        nbands,
        nkb,
        nhm,
        jkb_to_iat,
        jkb_to_it,
        jkb_to_ih,
        jkb_pref_sign,
        vkb1_cache,
        reinterpret_cast<const thrust::complex<FPTYPE>*>(sk_all),
        reinterpret_cast<const thrust::complex<FPTYPE>*>(psi),
        reinterpret_cast<thrust::complex<FPTYPE>*>(becp));
    CHECK_CUDA_SYNC();
}

template struct cal_becp_from_vkb1_cache_op<float, base_device::DEVICE_GPU>;
template struct cal_becp_from_vkb1_cache_op<double, base_device::DEVICE_GPU>;

}  // namespace hamilt

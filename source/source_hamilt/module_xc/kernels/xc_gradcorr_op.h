#ifndef SOURCE_HAMILT_MODULE_XC_KERNELS_XC_GRADCORR_OP_H
#define SOURCE_HAMILT_MODULE_XC_KERNELS_XC_GRADCORR_OP_H

#include "source_base/module_device/types.h"

#include <complex>

namespace hamilt
{

template <typename FPTYPE, typename Device>
struct xc_gradcorr_pbe_grid_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* gdr,
                    FPTYPE* v,
                    FPTYPE* h,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE, typename Device>
struct xc_scalar_pbe_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    FPTYPE* rho_total,
                    FPTYPE* v,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE>
struct xc_scalar_pbe_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    FPTYPE* rho_total,
                    FPTYPE* v,
                    FPTYPE* sums,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE>
struct xc_gradcorr_pbe_grid_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* gdr,
                    FPTYPE* v,
                    FPTYPE* h,
                    FPTYPE* sums,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE, typename Device>
struct xc_gradcorr_pbe_grid_resident_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* gdr,
                    FPTYPE* v,
                    FPTYPE* h,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE>
struct xc_gradcorr_pbe_grid_resident_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* gdr,
                    FPTYPE* v,
                    FPTYPE* h,
                    FPTYPE* sums,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE, typename Device>
struct xc_apply_dh_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* dh,
                    FPTYPE* v,
                    FPTYPE* vtxc_delta);
};

template <typename FPTYPE>
struct xc_apply_dh_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* dh,
                    FPTYPE* v,
                    FPTYPE* sum,
                    FPTYPE* vtxc_delta);
};

template <typename FPTYPE, typename Device>
struct xc_multiply_iG_op
{
    void operator()(const Device* ctx,
                    int npw,
                    int ipol,
                    const FPTYPE* gcar,
                    const std::complex<FPTYPE>* rhog,
                    std::complex<FPTYPE>* porter);
};

template <typename FPTYPE>
struct xc_multiply_iG_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int npw,
                    int ipol,
                    const FPTYPE* gcar,
                    const std::complex<FPTYPE>* rhog,
                    std::complex<FPTYPE>* porter);
};

template <typename FPTYPE, typename Device>
struct xc_accumulate_iG_op
{
    void operator()(const Device* ctx,
                    int npw,
                    int ipol,
                    const FPTYPE* gcar,
                    const std::complex<FPTYPE>* rhog,
                    std::complex<FPTYPE>* accum,
                    bool zero_first);
};

template <typename FPTYPE>
struct xc_accumulate_iG_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int npw,
                    int ipol,
                    const FPTYPE* gcar,
                    const std::complex<FPTYPE>* rhog,
                    std::complex<FPTYPE>* accum,
                    bool zero_first);
};

template <typename FPTYPE, typename Device>
struct xc_set_component_op
{
    void operator()(const Device* ctx, int nrxx, int ipol, const FPTYPE* component, FPTYPE* interleaved);
};

template <typename FPTYPE>
struct xc_set_component_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int ipol,
                    const FPTYPE* component,
                    FPTYPE* interleaved);
};

template <typename FPTYPE, typename Device>
struct xc_extract_component_op
{
    void operator()(const Device* ctx, int nrxx, int ipol, const FPTYPE* interleaved, FPTYPE* component);
};

template <typename FPTYPE>
struct xc_extract_component_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int ipol,
                    const FPTYPE* interleaved,
                    FPTYPE* component);
};

} // namespace hamilt

#endif // SOURCE_HAMILT_MODULE_XC_KERNELS_XC_GRADCORR_OP_H

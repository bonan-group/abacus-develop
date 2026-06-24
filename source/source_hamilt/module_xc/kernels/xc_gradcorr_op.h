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

template <typename FPTYPE, typename Device>
struct xc_scalar_lda_spin_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    int correlation,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho_up,
                    const FPTYPE* rho_dw,
                    const FPTYPE* rho_core,
                    FPTYPE* rho_up_total,
                    FPTYPE* rho_dw_total,
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
struct xc_scalar_lda_spin_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int correlation,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho_up,
                    const FPTYPE* rho_dw,
                    const FPTYPE* rho_core,
                    FPTYPE* rho_up_total,
                    FPTYPE* rho_dw_total,
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

template <typename FPTYPE, typename Device>
struct xc_gradcorr_pbe_spin_grid_resident_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho_up,
                    const FPTYPE* rho_dw,
                    const FPTYPE* rho_core,
                    const FPTYPE* gdr_up,
                    const FPTYPE* gdr_dw,
                    FPTYPE* v,
                    FPTYPE* h_up,
                    FPTYPE* h_dw,
                    FPTYPE* etxc,
                    FPTYPE* vtxc);
};

template <typename FPTYPE, typename Device>
struct xc_gradcorr_pbe_stress_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* gdr,
                    FPTYPE* stress);
};

template <typename FPTYPE, typename Device>
struct xc_gradcorr_pbe_spin_stress_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho_up,
                    const FPTYPE* rho_dw,
                    const FPTYPE* gdr_up,
                    const FPTYPE* gdr_dw,
                    FPTYPE* stress);
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

template <typename FPTYPE>
struct xc_gradcorr_pbe_spin_grid_resident_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
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
                    FPTYPE* vtxc);
};

template <typename FPTYPE>
struct xc_gradcorr_pbe_stress_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho,
                    const FPTYPE* gdr,
                    FPTYPE* stress);
};

template <typename FPTYPE>
struct xc_gradcorr_pbe_spin_stress_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    int iflag,
                    FPTYPE e2,
                    FPTYPE epsr,
                    const FPTYPE* rho_up,
                    const FPTYPE* rho_dw,
                    const FPTYPE* gdr_up,
                    const FPTYPE* gdr_dw,
                    FPTYPE* stress);
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

template <typename FPTYPE, typename Device>
struct xc_add_potential_op
{
    void operator()(const Device* ctx, int size, const FPTYPE* src, FPTYPE* dst);
};

template <typename FPTYPE, typename Device>
struct xc_apply_dh_spin_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    const FPTYPE* rho,
                    const FPTYPE* rho_core,
                    const FPTYPE* dh,
                    FPTYPE* v,
                    FPTYPE* vtxc_delta);
};

template <typename FPTYPE, typename Device>
struct xc_noncolin_rho_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    bool lsign,
                    const FPTYPE* rho0,
                    const FPTYPE* rho1,
                    const FPTYPE* rho2,
                    const FPTYPE* rho3,
                    const FPTYPE* ux,
                    FPTYPE* rho_up,
                    FPTYPE* rho_dw,
                    FPTYPE* neg);
};

template <typename FPTYPE, typename Device>
struct xc_noncolin_rotate_potential_op
{
    void operator()(const Device* ctx,
                    int nrxx,
                    const FPTYPE* rho1,
                    const FPTYPE* rho2,
                    const FPTYPE* rho3,
                    const FPTYPE* neg,
                    const FPTYPE* v_up,
                    const FPTYPE* v_dw,
                    FPTYPE* v0,
                    FPTYPE* v1,
                    FPTYPE* v2,
                    FPTYPE* v3);
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

template <typename FPTYPE>
struct xc_add_potential_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx, int size, const FPTYPE* src, FPTYPE* dst);
};

template <typename FPTYPE>
struct xc_apply_dh_spin_op<FPTYPE, base_device::DEVICE_GPU>
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

template <typename FPTYPE>
struct xc_noncolin_rho_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    bool lsign,
                    const FPTYPE* rho0,
                    const FPTYPE* rho1,
                    const FPTYPE* rho2,
                    const FPTYPE* rho3,
                    const FPTYPE* ux,
                    FPTYPE* rho_up,
                    FPTYPE* rho_dw,
                    FPTYPE* neg);
};

template <typename FPTYPE>
struct xc_noncolin_rotate_potential_op<FPTYPE, base_device::DEVICE_GPU>
{
    void operator()(const base_device::DEVICE_GPU* ctx,
                    int nrxx,
                    const FPTYPE* rho1,
                    const FPTYPE* rho2,
                    const FPTYPE* rho3,
                    const FPTYPE* neg,
                    const FPTYPE* v_up,
                    const FPTYPE* v_dw,
                    FPTYPE* v0,
                    FPTYPE* v1,
                    FPTYPE* v2,
                    FPTYPE* v3);
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

#ifndef SOURCE_HAMILT_MODULE_XC_KERNELS_XC_GRADCORR_OP_H
#define SOURCE_HAMILT_MODULE_XC_KERNELS_XC_GRADCORR_OP_H

#include "source_base/module_device/types.h"

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

} // namespace hamilt

#endif // SOURCE_HAMILT_MODULE_XC_KERNELS_XC_GRADCORR_OP_H

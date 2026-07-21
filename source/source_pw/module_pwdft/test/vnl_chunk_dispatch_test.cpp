#include "source_pw/module_pwdft/fs_nonlocal_tools.h"

#include <type_traits>

typedef void (pseudopot_cell_vnl::*VnlInitSignature)(const UnitCell&,
                                                     Structure_Factor*,
                                                     const ModulePW::PW_Basis_K*,
                                                     bool,
                                                     const VnlChunkPolicy&,
                                                     bool);

static_assert(std::is_same<decltype(&pseudopot_cell_vnl::init), VnlInitSignature>::value,
              "GPU memory ownership must be explicit and independent from chunk-kernel support");

static_assert(!hamilt::detail::VnlChunkKernelSupport<base_device::DEVICE_CPU>::value,
              "CPU force/stress paths must not dispatch VNL chunk kernels");

#if defined(__CUDA) || defined(__UT_USE_CUDA)
static_assert(hamilt::detail::VnlChunkKernelSupport<base_device::DEVICE_GPU>::value,
              "CUDA force/stress paths must dispatch VNL chunk kernels");
#endif

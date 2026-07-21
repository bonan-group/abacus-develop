#include "source_pw/module_pwdft/fs_nonlocal_tools.h"

#include <type_traits>

static_assert(!hamilt::detail::VnlChunkKernelSupport<base_device::DEVICE_CPU>::value,
              "CPU force/stress paths must not dispatch VNL chunk kernels");

#if defined(__CUDA) || defined(__UT_USE_CUDA)
static_assert(hamilt::detail::VnlChunkKernelSupport<base_device::DEVICE_GPU>::value,
              "CUDA force/stress paths must dispatch VNL chunk kernels");
#endif

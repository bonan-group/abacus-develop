/**
 * @file nvtx_helper.h
 * @brief Macro wrappers for NVTX (NVIDIA Tools Extension) profiling markers.
 *
 * This header provides convenient macros for adding NVTX profiling markers
 * that enable GPU profiling with tools like NVIDIA Nsight Systems.
 *
 * When compiled with CUDA and __USE_NVTX defined, the macros expand to
 * actual NVTX API calls. Otherwise, they compile to no-ops.
 *
 * Usage:
 *   NVTX_RANGE_PUSH("SectionName");  // Start a named range
 *   // ... code to profile ...
 *   NVTX_RANGE_POP();                // End the range
 *
 *   NVTX_MARK("EventName");          // Mark a single point in time
 */

#ifndef NVTX_HELPER_H_
#define NVTX_HELPER_H_

#if defined(__CUDA) && defined(__USE_NVTX)
#include "source_base/module_device/cuda_compat.h"

#define NVTX_RANGE_PUSH(name) nvtxRangePushA(name)
#define NVTX_RANGE_POP() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)

#else
// No-op macros for CPU-only builds or when NVTX is disabled
#define NVTX_RANGE_PUSH(name) do {} while(0)
#define NVTX_RANGE_POP() do {} while(0)
#define NVTX_MARK(name) do {} while(0)
#endif

#endif // NVTX_HELPER_H_

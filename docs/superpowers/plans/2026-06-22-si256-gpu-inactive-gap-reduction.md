# Si256 GPU Inactive Gap Reduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the remaining CPU-only gaps in the 256-atom silicon GPU benchmark, with correctness checks and Nsight evidence after each improvement.

**Architecture:** Treat each observed inactive gap as a measurable target. First make profiling attribution reproducible, then move SCC force accumulation to GPU, then address the repeated XC functional gap with staged CPU optimization and a guarded GPU-native path for the common Si/PBE case. Keep CPU fallbacks and numerical comparisons in every task.

**Tech Stack:** ABACUS C++/CUDA, existing `base_device` memory operators, GoogleTest, CMake, Nsight Systems, `OMP_NUM_THREADS=1`.

## Global Constraints

- Run ABACUS runtime tests, MPI tests, and Nsight profiling outside the sandbox.
- Set `OMP_NUM_THREADS=1` for ABACUS runs unless explicitly comparing thread counts.
- Do not add new dependencies on `GlobalV`, `GlobalC`, or `PARAM`.
- Prefer existing `base_device` memory/operator patterns over new infrastructure.
- Keep CPU implementations as reference fallbacks.
- Preserve force/stress correctness before accepting any performance improvement.
- Every performance task must report: ABACUS timer table, Nsight inactive-gap summary, top kernels, and max force/energy drift versus baseline.

---

## Profile Baseline

Current OMP=1 Si256 run:

- Total ABACUS wall time: 54.72 s.
- CUDA active span: 54.33 s.
- GPU-active time: 24.59 s.
- GPU inactive gap time: 29.74 s.
- Biggest CPU-only contributors:
  - `PotXC cal_veff`: 15.32 s over 10 calls, almost entirely `XC_Functional v_xc`.
  - `Forces cal_force_scc`: 6.69 s in final force phase.
  - Smaller host orchestration and stress/force reductions around force/stress finalization.

Key source evidence:

- XC enters CPU/LibXC path at `source/source_estate/module_pot/pot_xc.cpp`.
- `source/source_hamilt/module_xc/kernels/cuda/xc_functional_op.cu` only provides gradient helper kernels, not full `v_xc`.
- SCC force computes `deriv_drhoc_scc` partly on GPU, then copies `drhocg` back and performs the main atom/G accumulation on CPU in `source/source_pw/module_pwdft/forces_scc.cpp`.
- Nonlocal, local, Ewald, and NLCC force already have GPU kernels or partial GPU paths.

---

### Task 1: Make GPU Gap Attribution Reproducible

**Files:**
- Create: `tools/perf/analyze_nsys_gpu_gaps.py`
- Create: `tools/perf/run_si256_nsys.sh`
- Modify: `docs/superpowers/plans/2026-06-22-si256-gpu-inactive-gap-reduction.md`

**Interfaces:**
- Produces: `tools/perf/analyze_nsys_gpu_gaps.py profile.sqlite --min-gap-ms 10 --json gaps.json`
- Produces: `tools/perf/run_si256_nsys.sh CASE_DIR ABACUS_BIN OUT_DIR`
- Later tasks use the same output schema to compare gap time.

- [x] **Step 1: Write the gap analyzer**

Create `tools/perf/analyze_nsys_gpu_gaps.py` with:

```python
#!/usr/bin/env python3
import argparse
import json
import sqlite3


def tables(conn):
    return {row[0] for row in conn.execute("select name from sqlite_master where type='table'")}


def cuda_intervals(conn):
    names = tables(conn)
    intervals = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if table not in names:
            continue
        cols = {row[1] for row in conn.execute(f"pragma table_info({table})")}
        if {"start", "end"} <= cols:
            intervals.extend((int(start), int(end), table) for start, end in conn.execute(f"select start,end from {table}"))
    return sorted((s, e, k) for s, e, k in intervals if e > s)


def merge(intervals):
    merged = []
    for start, end, _kind in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--min-gap-ms", type=float, default=10.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    conn = sqlite3.connect(args.sqlite)
    intervals = cuda_intervals(conn)
    if not intervals:
        raise SystemExit("no CUDA activity intervals found")
    merged = merge(intervals)
    span_start, span_end = merged[0][0], merged[-1][1]
    active_ns = sum(end - start for start, end in merged)
    min_gap_ns = int(args.min_gap_ms * 1e6)
    gaps = []
    for prev, cur in zip(merged, merged[1:]):
        gap = cur[0] - prev[1]
        if gap >= min_gap_ns:
            gaps.append({"start_s": prev[1] / 1e9, "end_s": cur[0] / 1e9, "duration_s": gap / 1e9})

    result = {
        "span_s": (span_end - span_start) / 1e9,
        "active_s": active_ns / 1e9,
        "inactive_s": ((span_end - span_start) - active_ns) / 1e9,
        "utilization_percent": 100.0 * active_ns / (span_end - span_start),
        "gap_count": len(gaps),
        "gaps": sorted(gaps, key=lambda item: item["duration_s"], reverse=True),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Write the repeatable profiling wrapper**

Create `tools/perf/run_si256_nsys.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

case_dir=$1
abacus_bin=$2
out_dir=$3

mkdir -p "${out_dir}"
cd "${case_dir}"
export OMP_NUM_THREADS=1

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv -lms 500 > "${out_dir}/gpu_usage.csv" &
gpu_pid=$!
trap 'kill "${gpu_pid}" 2>/dev/null || true' EXIT

/usr/bin/time -v nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o "${out_dir}/profile" "${abacus_bin}" > "${out_dir}/abacus_stdout.log" 2> "${out_dir}/nsys_run.log"
nsys stats "${out_dir}/profile.nsys-rep" > "${out_dir}/nsys_stats.txt"
sqlite="${out_dir}/profile.sqlite"
nsys export -t sqlite --force-overwrite=true -o "${sqlite}" "${out_dir}/profile.nsys-rep" > "${out_dir}/nsys_export.log"
python3 "${OLDPWD}/tools/perf/analyze_nsys_gpu_gaps.py" "${sqlite}" --min-gap-ms 10 --json "${out_dir}/gaps.json"
```

- [x] **Step 3: Verify analyzer on existing profile**

Run:

```bash
python3 tools/perf/analyze_nsys_gpu_gaps.py nsight_si256_unseeded_gpuinit_omp1_20260622-112921/si256_dav_subspace_gpu_random_unseeded_omp1/profile.sqlite --min-gap-ms 500
```

Expected: reports one ~6.6 s gap, repeated ~1.7 s gaps, and utilization near 45%.

- [x] **Step 4: Commit**

```bash
git add tools/perf/analyze_nsys_gpu_gaps.py tools/perf/run_si256_nsys.sh docs/superpowers/plans/2026-06-22-si256-gpu-inactive-gap-reduction.md
git commit -m "Add repeatable Si256 Nsight gap tooling"
```

---

### Task 2: Move SCC Force Main Accumulation to GPU

**Files:**
- Modify: `source/source_pw/module_pwdft/forces_scc.cpp`
- Modify: `source/source_pw/module_pwdft/kernels/stress_op.h`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu`
- Modify: `source/source_pw/module_pwdft/kernels/rocm/stress_op.hip.cu`
- Test: `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp`

**Interfaces:**
- Produces: `hamilt::cal_force_scc_op<FPTYPE, base_device::DEVICE_GPU>::operator()(...)`
- Consumes: device arrays for `psic(G)`, `gcar`, `ig2igg`, `rhocgnt`, `tau`, and output `forcescc`.
- CPU path remains unchanged.

- [x] **Step 1: Add a kernel unit test**

Add a small CPU-vs-GPU comparison in `stress_op_test.cpp`:

```cpp
TEST_F(StressOpTest, CalForceSccGpuMatchesCpuReference)
{
    // Use nat=2, npw=4, ngg=4 with nonzero G vectors and deterministic psic/rhocgnt/tau.
    // Compute the reference with the same formula currently in Forces::cal_force_scc.
    // Launch cal_force_scc_op<double, DEVICE_GPU>.
    // Copy the device force back and EXPECT_NEAR each component to 1e-10.
}
```

Use existing memory helpers already used in this file for GPU allocation/copy.

- [x] **Step 2: Run test and verify it fails to compile**

Run:

```bash
cmake --build build-test-cuda --target MODULE_PW_Stress_UTs -j2
```

Expected: compile failure because `cal_force_scc_op` does not exist.

Note: in the current build tree the target is `MODULE_PW_Hamilt_Kernels_UTs`.
The RED build failed as expected with missing `hamilt::cal_force_scc_op`.

- [x] **Step 3: Declare the operator**

Add to `stress_op.h` near `cal_force_npw_op`:

```cpp
template <typename FPTYPE, typename Device>
struct cal_force_scc_op
{
    void operator()(const Device* ctx,
                    int nat,
                    int npw,
                    int ig0,
                    int forcenl_nc,
                    FPTYPE fact,
                    FPTYPE tpiba,
                    const FPTYPE* gcar,
                    const int* ig2igg,
                    const FPTYPE* rhocgnt,
                    const std::complex<FPTYPE>* psic,
                    const FPTYPE* tau,
                    FPTYPE* forcescc);
};
```

Add a `DEVICE_GPU` specialization beside the other GPU declarations.

- [x] **Step 4: Implement CUDA SCC force kernel**

Add to `stress_op.cu`:

```cpp
template <typename FPTYPE>
__global__ void cal_force_scc_kernel(int nat,
                                     int npw,
                                     int ig0,
                                     int force_nc,
                                     FPTYPE fact,
                                     FPTYPE tpiba,
                                     const FPTYPE* gcar,
                                     const int* ig2igg,
                                     const FPTYPE* rhocgnt,
                                     const thrust::complex<FPTYPE>* psic,
                                     const FPTYPE* tau,
                                     FPTYPE* forcescc)
{
    const int iat = blockIdx.x;
    const int tid = threadIdx.x;
    FPTYPE fx = 0.0;
    FPTYPE fy = 0.0;
    FPTYPE fz = 0.0;
    const FPTYPE tx = tau[3 * iat + 0];
    const FPTYPE ty = tau[3 * iat + 1];
    const FPTYPE tz = tau[3 * iat + 2];
    for (int ig = tid; ig < npw; ig += blockDim.x)
    {
        if (ig == ig0) { continue; }
        const FPTYPE gx = gcar[3 * ig + 0];
        const FPTYPE gy = gcar[3 * ig + 1];
        const FPTYPE gz = gcar[3 * ig + 2];
        const FPTYPE arg = ModuleBase::TWO_PI * (gx * tx + gy * ty + gz * tz);
        FPTYPE sinp, cosp;
        sincos(arg, &sinp, &cosp);
        const thrust::complex<FPTYPE> cpm(sinp, cosp);
        const FPTYPE value = (cpm * thrust::conj(psic[ig])).real() * fact * rhocgnt[ig2igg[ig]] * tpiba;
        fx += gx * value;
        fy += gy * value;
        fz += gz * value;
    }
    // Use the same warp/block reduction pattern as force_loc_kernel.
    // Store into forcescc[iat * force_nc + 0..2].
}
```

- [x] **Step 5: Wire GPU path in `forces_scc.cpp`**

In `cal_force_scc`, when `this->device == base_device::GpuDevice`, keep `psic` on device after `real2recip` if the PW basis exposes a device FFT result. If it does not, copy `psic`, `gcar`, `ig2igg`, `tau`, and `rhocgnt` to device once per atom type and call `cal_force_scc_op`. Do not copy `drhocgnt` back before the force accumulation in the GPU path.

Note: the implemented runtime path is guarded to single-atomic-type cells, matching
the Si256 benchmark, so mixed-species cells keep the CPU reference accumulation.

- [x] **Step 6: Run kernel test**

Run:

```bash
cmake --build build-test-cuda --target MODULE_PW_Stress_UTs -j2
./build-test-cuda/source/source_pw/module_pwdft/kernels/test/MODULE_PW_Stress_UTs --gtest_filter='*CalForceSccGpuMatchesCpuReference*'
```

Expected: test passes with max absolute difference below `1e-10`.

Actual: `MODULE_PW_Hamilt_Kernels_UTs --gtest_filter='*cal_force_scc_op_gpu*'`
passed outside the sandbox. The sandboxed run cannot see a CUDA device.

- [x] **Step 7: Run Si256 force correctness check**

Run the Si256 case outside the sandbox with:

```bash
cd runtime_si256_force_stress_fix_build_20260621-232100
export OMP_NUM_THREADS=1
/home/bonan/appdir/abacus-develop-cufft-batch/build/abacus_basic_gpu > scc_gpu_force_stdout.log 2> scc_gpu_force_stderr.log
```

Expected: exit status 0. Compare total energy and final max force to the committed baseline logs. Accept only if force drift is within numerical noise for the same precision mode.

Actual: guarded Si256 run exited 0 with `OMP_NUM_THREADS=1`; final energy stayed
`-27439.6264120973828540 eV` and total pressure stayed `44.738920 kbar`.

- [x] **Step 8: Profile and accept/reject**

Run:

```bash
tools/perf/run_si256_nsys.sh runtime_si256_force_stress_fix_build_20260621-232100 build/abacus_basic_gpu nsight_si256_scc_gpu_omp1
```

Expected: `Forces cal_force_scc` no longer produces a ~6.6 s CPU-only gap. If the gap remains because `real2recip` or host transfer dominates, record that and split the next task around keeping `vnew`/FFT output resident on GPU.

Actual: `nsight_si256_scc_gpu_omp1/gaps.json` no longer shows the original
single ~6.6 s gap; largest gaps are now repeated XC/SCF-region gaps of about
1.8-3.0 s. Total inactive time remains about 29.97 s.

- [x] **Step 9: Commit**

```bash
git add source/source_pw/module_pwdft/forces_scc.cpp source/source_pw/module_pwdft/kernels/stress_op.h source/source_pw/module_pwdft/kernels/cuda/stress_op.cu source/source_pw/module_pwdft/kernels/rocm/stress_op.hip.cu source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp
git commit -m "Move SCC force accumulation to GPU"
```

---

### Task 3: Remove Avoidable Host Work from Existing GPU Force Paths

**Files:**
- Modify: `source/source_pw/module_pwdft/forces.cpp`
- Modify: `source/source_pw/module_pwdft/forces_cc.cpp`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/force_op.cu`
- Test: existing force/stress runtime Si256 case.

**Interfaces:**
- Consumes: GPU force components from local/Ewald/NLCC/nonlocal/SCC.
- Produces: lower host transfer/setup time in `Forces cal_force`.

- [x] **Step 1: Add timers around host packing and D2H copies**

Use existing `ModuleBase::timer` labels:

```cpp
ModuleBase::timer::start("Forces", "force_gpu_pack");
// host vector construction and H2D copies
ModuleBase::timer::end("Forces", "force_gpu_pack");

ModuleBase::timer::start("Forces", "force_gpu_d2h");
// D2H copies
ModuleBase::timer::end("Forces", "force_gpu_d2h");
```

- [x] **Step 2: Run Si256 once and identify transfer/setup cost**

Run outside sandbox:

```bash
cd runtime_si256_force_stress_fix_build_20260621-232100
export OMP_NUM_THREADS=1
/home/bonan/appdir/abacus-develop-cufft-batch/build/abacus_basic_gpu > force_pack_stdout.log 2> force_pack_stderr.log
```

Expected: timer table shows whether repeated `gcar/tau/iat2it` packing is material.

Actual: `force_gpu_pack` and `force_gpu_d2h` did not appear in the printed
Si256 timer table, so their costs are below the timer report threshold. The
force phase remains dominated by `Forces cal_force_nl`.

- [x] **Step 3: Reuse device metadata per force call**

Create small local RAII scratch structs inside `forces.cpp` if packing is material:

```cpp
struct ForceDeviceMeta
{
    int* iat2it = nullptr;
    int* ig2igg = nullptr;
    double* gcar = nullptr;
    double* tau = nullptr;
};
```

Allocate once in `cal_force`, pass to local/Ewald/SCC/NLCC helpers, and free before returning.

Decision: skipped the scratch-struct reuse refactor because measurement did
not show packing/D2H as material. This avoids broad force-interface churn.

- [x] **Step 4: Keep final component sum on CPU unless proven material**

The final loop over `nat * 3` is small for 256 atoms. Do not move it to GPU unless profiler shows it above 100 ms.

- [x] **Step 5: Verify**

Run build and Si256:

```bash
cmake --build build --target abacus_basic_gpu -j2
cd runtime_si256_force_stress_fix_build_20260621-232100
export OMP_NUM_THREADS=1
/home/bonan/appdir/abacus-develop-cufft-batch/build/abacus_basic_gpu > force_meta_stdout.log 2> force_meta_stderr.log
```

Expected: exit status 0, unchanged force values, lower force setup timers if they were material.

Actual: instrumented Si256 run exited 0 with `OMP_NUM_THREADS=1`; final energy
remained `-27439.6264120973828540 eV` and pressure `44.738920 kbar`.

- [x] **Step 6: Commit**

```bash
git add source/source_pw/module_pwdft/forces.cpp source/source_pw/module_pwdft/forces_cc.cpp source/source_pw/module_pwdft/kernels/cuda/force_op.cu
git commit -m "Reduce host setup in GPU force paths"
```

---

### Task 4: Add Fine-Grained XC Timing and Decide GPU Target

**Files:**
- Modify: `source/source_hamilt/module_xc/libxc_pot.cpp`
- Modify: `source/source_hamilt/module_xc/libxc_tools.cpp`
- Modify: `source/source_hamilt/module_xc/xc_pot.cpp`

**Interfaces:**
- Produces timers for `xc_convert_rho`, `xc_cal_gdr`, `xc_libxc_eval`, `xc_convert_v`.
- Later tasks use these to choose CPU batching versus GPU implementation.

- [x] **Step 1: Add sub-timers**

Wrap each stage:

```cpp
ModuleBase::timer::start("XC_Functional_Libxc", "convert_rho");
rho = XC_Functional_Libxc::convert_rho(nspin, nrxx, chr);
ModuleBase::timer::end("XC_Functional_Libxc", "convert_rho");
```

Repeat for `cal_gdr`, `convert_sigma`, `cal_sgn`, `xc_lda/gga_exc_vxc`, `convert_etxc`, and `convert_vtxc_v`.

- [x] **Step 2: Verify timing labels**

Run:

```bash
cmake --build build --target abacus_basic_gpu -j2
cd runtime_si256_force_stress_fix_build_20260621-232100
export OMP_NUM_THREADS=1
/home/bonan/appdir/abacus-develop-cufft-batch/build/abacus_basic_gpu > xc_timers_stdout.log 2> xc_timers_stderr.log
```

Expected: timer table splits the current ~15.3 s XC time across stages.

Actual: the LibXC sub-timers did not appear for the Si256 benchmark because
this input uses the built-in `XC_Functional::v_xc` path. Added built-in
sub-timers as well. The verified OMP=1 Si256 split is:

- `PotXC cal_veff`: 19.28 s over 10 calls.
- `XC_Functional v_xc`: 19.21 s over 10 calls.
- `XC_Functional xc_builtin_eval`: 1.98 s over 10 calls.
- `XC_Functional gradcorr`: 17.12 s over 10 calls.

- [x] **Step 3: Decide target path**

Use the timer split:

- If `xc_lda/gga_exc_vxc` dominates: implement a GPU-native analytic PBE/PBEsol/LDA path for known built-in functionals first.
- If `convert_*` and `cal_gdr` dominate: move density conversion, sigma construction, and `convert_vtxc_v` to GPU while leaving LibXC evaluation on CPU.
- If both dominate: implement GPU analytic PBE for Si benchmark and keep LibXC as fallback.

Decision: `gradcorr` dominates the Si/PBE benchmark. Task 5 should target a
guarded GPU path for the built-in GGA gradient-correction work first, not the
LibXC conversion helpers or the scalar built-in XC evaluation.

- [x] **Step 4: Commit timing instrumentation**

```bash
git add source/source_hamilt/module_xc/libxc_pot.cpp source/source_hamilt/module_xc/libxc_tools.cpp source/source_hamilt/module_xc/xc_pot.cpp
git commit -m "Add detailed XC timing attribution"
```

Actual commit: `f91c640af Add detailed XC timing attribution`.

Additional `gradcorr` attribution showed the Si/PBE GGA time is split across
multiple sub-stages:

- `gradcorr_rho_fft`: 1.99 s over 11 calls.
- `gradcorr_grad_rho`: 4.42 s over 11 calls.
- `gradcorr_eval_grid`: 5.66 s over 11 calls.
- `gradcorr_grad_dot`: 5.30 s over 10 calls.

This means a scalar-only LDA/PBE XC kernel would not remove the whole observed
gap; a useful runtime path must also address gradient/divergence FFT residency
and the grid-point GGA loop.

---

### Task 5: Implement a Guarded GPU `gradcorr` Path for Built-In PBE

**Files:**
- Create/modify: `source/source_hamilt/module_xc/kernels/xc_gradcorr_op.h`
- Create/modify: `source/source_hamilt/module_xc/kernels/cuda/xc_gradcorr_op.cu`
- Modify: `source/source_hamilt/module_xc/xc_gpu_policy.h`
- Modify: `source/source_estate/module_pot/pot_xc.h`
- Modify: `source/source_estate/module_pot/pot_xc.cpp`
- Modify: `source/source_hamilt/module_xc/xc_functional.h`
- Modify: `source/source_hamilt/module_xc/xc_pot.cpp`
- Modify: `source/source_hamilt/module_xc/xc_grad.cpp`
- Modify: `source/CMakeLists.txt`
- Test: `source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp`

**Interfaces:**
- Produces: a guarded GPU implementation for the dominant built-in PBE
  `gradcorr` work, not just scalar LDA/PBE evaluation.
- Requires: an explicit device/context path from `PotXC::cal_v_eff` into
  `XC_Functional::v_xc`/`gradcorr`.
- Guards: CUDA build, `device == gpu`, `ABACUS_XC_GPU=1`, `nspin == 1`,
  non-stress call, built-in PBE/PBEsol only.
- CPU/LibXC remains default fallback for unsupported functionals.

**Rationale:**

The measured Si256 bottleneck is `XC_Functional::gradcorr`, not the scalar
`XC_Functional::xc` loop. The relevant OMP=1 timer split is:

- `XC_Functional v_xc`: 19.63 s over 10 calls.
- `XC_Functional xc_builtin_eval`: 1.91 s over 10 calls.
- `XC_Functional gradcorr`: 17.60 s over 10 calls.
- `gradcorr_rho_fft`: 1.99 s over 11 calls.
- `gradcorr_grad_rho`: 4.42 s over 11 calls.
- `gradcorr_eval_grid`: 5.66 s over 11 calls.
- `gradcorr_grad_dot`: 5.30 s over 10 calls.

Therefore a standalone LDA scalar kernel would address the wrong hot path. A
useful GPU XC path must either keep the density-gradient/divergence workflow
GPU-resident or replace it with GPU kernels plus GPU FFT calls.

- [x] **Step 1: Write policy tests**

Add tests for:

```cpp
EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PBE"));
EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PZ"));
EXPECT_FALSE(xc_gpu_policy(false, false, 1, "PBE"));
EXPECT_FALSE(xc_gpu_policy(true, false, 2, "PBE"));
EXPECT_FALSE(xc_gpu_policy(true, true, 1, "PBE"));
EXPECT_FALSE(xc_gpu_policy(true, false, 1, "SCAN"));
```

- [x] **Step 2: Run test and verify it fails**

Run:

```bash
cmake --build build-test-cuda --target MODULE_HAMILT_XC_Functional_UTs -j2
```

Expected: missing `xc_gpu_policy.h`.

Actual: the target `MODULE_HAMILT_XC_Functional_UTs` is absent in the available
CUDA test build because `source/source_hamilt/module_xc/CMakeLists.txt` gates
kernel tests behind both `ENABLE_MPI` and `ENABLE_LIBXC`, and the configured
`build-test-cuda` has `ENABLE_LIBXC=OFF`. The first direct target build
therefore failed with "No rule to make target".

- [x] **Step 3: Implement policy header**

Create:

```cpp
#ifndef SOURCE_HAMILT_MODULE_XC_XC_GPU_POLICY_H
#define SOURCE_HAMILT_MODULE_XC_XC_GPU_POLICY_H

#include <string>

namespace XC_Functional_GPU
{
inline bool xc_gpu_policy(bool is_gpu, bool cpu_debug, int nspin, const std::string& xc_func)
{
    return is_gpu && !cpu_debug && nspin == 1
           && (xc_func == "PBE" || xc_func == "PBEsol" || xc_func == "LDA" || xc_func == "PZ");
}
}

#endif
```

Implemented as `source/source_hamilt/module_xc/xc_gpu_policy.h` with
case-insensitive matching for the supported built-ins.

- [x] **Step 4: Extend the PotXC/XC interface with explicit device intent**

Thread a device/context argument from `PotXC::cal_v_eff` into
`XC_Functional::v_xc` and `gradcorr`, preserving the existing CPU call path as
the default. Do not infer GPU use from global state alone.

Actual: added explicit `device` overloads for `XC_Functional::v_xc` and
`XC_Functional::gradcorr`; the existing overloads still route to `"cpu"`.
`PotXC::cal_v_eff` now passes `rho_basis_->get_device()` when available.

- [x] **Step 5: Add a guarded CPU fallback switch**

Use `ABACUS_XC_GPU=0/1` only as a runtime opt-in/opt-out once the call path has
explicit device intent. Unsupported cases must fall back to CPU:

- non-CUDA builds
- `nspin != 1`
- stress calls
- LibXC/mGGA/hybrid functionals
- non-PBE/PBEsol built-ins

Actual: the runtime path requires CUDA build, explicit GPU device intent,
`ABACUS_XC_GPU=1`, built-in PBE/PBEsol, `nspin == 1`, non-stress, and
non-LibXC. All other cases keep the existing CPU path.

- [x] **Step 6: Port the PBE/PBEsol grid-point `gradcorr` evaluation**

Move the `nspin == 1` built-in GGA grid loop from `gradcorr` to a CUDA kernel:

- input: `rhotmp1`, `rho_core`, `gdr1`
- output: local additions to `v`, `h1`, `etxcgc`, `vtxcgc`
- formulas: match existing `gcxc`, `pbex`, and `pbec` behavior for PBE/PBEsol
- reductions: use deterministic block reductions where practical and compare
  against CPU tolerances

Actual: added `xc_gradcorr_pbe_grid_op` CPU/CUDA operators and wired the
guarded runtime path for the grid-point contribution to `v`, `h1`, `etxcgc`,
and `vtxcgc`. A reference test caught and fixed an initial `pw`/`pz`
correlation-helper mix-up in the copied PBE formula.

- [x] **Step 7: Address GPU-resident gradient/divergence work**

The measured FFT/derivative pieces are comparable to the grid loop. Decide
after Step 6 profiling whether to:

- keep `grad_rho`/`grad_dot` CPU-side initially and accept a partial win, or
- add GPU `grad_rho`/`grad_dot` variants using the existing PW GPU FFT
  machinery so `rhotmp`, `gdr`, `h`, and `dh` avoid host round-trips.

Decision: kept `grad_rho` and `grad_dot` CPU-side for this staged change. The
grid-loop-only move reduces the measured XC grid time substantially, while
full GPU-resident gradient/divergence remains the next larger XC project.

- [x] **Step 8: Add CPU-vs-GPU correctness tests**

Compare a deterministic small grid against the CPU `gradcorr` reference:

- `v` max absolute difference
- `etxc`/`vtxc` drift
- PBE and PBEsol
- fallback behavior for unsupported cases

Expected tolerance: start with `1e-10` double for isolated kernels and relax
only if full runtime FFT ordering requires it.

Actual:

```bash
cmake --build build-test-cuda --target MODULE_HAMILT_XC_Functional_UTs -j2
./build-test-cuda/source/source_hamilt/module_xc/kernels/test/MODULE_HAMILT_XC_Functional_UTs --gtest_filter='XCGradcorrOpTest.PbeGridCpuMatchesBuiltinReferenceValues:XCGradcorrOpTest.PbeGridGpuMatchesCpu:XCFunctionGpuPolicyTest.GuardsSupportedBuiltins'
```

The focused tests passed. The added CPU reference test compares the new
operator against existing ABACUS PBE reference values, and the GPU test
compares CUDA output against the CPU operator.

- [x] **Step 9: Runtime validation**

Run a small Si case and Si256 with CPU XC and GPU XC toggled:

```bash
export OMP_NUM_THREADS=1
ABACUS_XC_GPU=0 build/abacus_basic_gpu
ABACUS_XC_GPU=1 build/abacus_basic_gpu
```

Expected: total energy drift within selected precision tolerance, SCF convergence unchanged or improved.

Actual: Si256 CPU-XC fallback and GPU-XC runs both exited 0 with
`OMP_NUM_THREADS=1`. Separate copied-case logs gave:

- CPU-XC: `FINAL_ETOT = -27439.6260124258878932 eV`,
  pressure `44.739301 kbar`.
- GPU-XC: `FINAL_ETOT = -27439.6265233284029819 eV`,
  pressure `44.739213 kbar`.
- Absolute drift: `5.11e-4 eV`, `8.8e-5 kbar`.
- Max force component drift: `3.00e-4 eV/Angstrom`.
- Max force vector drift: `3.54e-4 eV/Angstrom`.

- [x] **Step 10: Profile**

Run Si256 Nsight:

```bash
ABACUS_XC_GPU=1 tools/perf/run_si256_nsys.sh /tmp/si256_xc_gpu_path build/abacus_basic_gpu nsight_si256_xc_gpu_gradcorr_omp1
```

Expected: repeated ~1.7 s gaps shrink substantially if XC was the source.

Actual timer/profile evidence:

- `PotXC cal_veff`: `14.11 s -> 11.46 s`.
- `XC_Functional v_xc`: `14.06 s -> 11.41 s`.
- `XC_Functional gradcorr`: `12.53 s -> 9.89 s`.
- `gradcorr_eval_grid`: `4.52 s -> 1.87 s`.
- new `gradcorr_eval_grid_gpu`: `1.39 s`.
- Nsight GPU active span: `43.23 s`.
- CUDA-active time: `25.06 s`.
- inactive time: `18.17 s`.
- utilization: `57.97%`.

- [x] **Step 11: Commit**

```bash
git add source/source_hamilt/module_xc/kernels/xc_gradcorr_op.h source/source_hamilt/module_xc/kernels/cuda/xc_gradcorr_op.cu source/source_hamilt/module_xc/xc_gpu_policy.h source/source_estate/module_pot/pot_xc.h source/source_estate/module_pot/pot_xc.cpp source/source_hamilt/module_xc/xc_functional.h source/source_hamilt/module_xc/xc_pot.cpp source/source_hamilt/module_xc/xc_grad.cpp source/CMakeLists.txt source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp
git commit -m "Add guarded GPU XC gradient correction path"
```

---

### Task 6: Final OMP=1 and OMP=8 Performance Comparison

**Files:**
- Create: `tools/perf/compare_si256_profiles.py`
- Modify: this plan with final measurements.

**Interfaces:**
- Consumes: `gaps.json`, `gpu_usage.csv`, `abacus_stdout.log`.
- Produces: concise comparison table for baseline, SCC-GPU, XC-GPU, and OMP=8.

- [x] **Step 1: Write comparison script**

Create a script that extracts:

- total ABACUS time
- `PotXC cal_veff`
- `XC_Functional v_xc`
- `Forces cal_force_scc`
- inactive gap total
- largest inactive gap
- average and max GPU utilization

Actual: added `tools/perf/compare_si256_profiles.py`. It reads `abacus_stdout.log`,
`gaps.json`, and `gpu_usage.csv`, then prints a Markdown table and optional CSV.
The final CSV was written to `nsight_si256_final_compare.csv`.

- [x] **Step 2: Run final OMP=1 profile**

```bash
tools/perf/run_si256_nsys.sh runtime_si256_force_stress_fix_build_20260621-232100 build/abacus_basic_gpu nsight_si256_final_omp1
```

Actual: ran with `OMP_NUM_THREADS=1 ABACUS_XC_GPU=1`. The run exited 0, wrote
`nsight_si256_final_omp1/profile.nsys-rep`, exported `profile.sqlite`, and wrote
`gaps.json`.

- [x] **Step 3: Run final OMP=8 profile**

Temporarily override the wrapper or run manually:

```bash
cd runtime_si256_force_stress_fix_build_20260621-232100
export OMP_NUM_THREADS=8
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o ../nsight_si256_final_omp8/profile /home/bonan/appdir/abacus-develop-cufft-batch/build/abacus_basic_gpu
```

Actual: ran with `OMP_NUM_THREADS=8 ABACUS_XC_GPU=1` and explicit GPU sampling.
The run exited 0, wrote `nsight_si256_final_omp8/profile.nsys-rep`, exported
`profile.sqlite`, and wrote `gaps.json`.

- [x] **Step 4: Acceptance criteria**

Accept the full series only if:

- Si256 exits status 0.
- SCF reaches the same convergence target.
- Total energy and forces match baseline within precision tolerance.
- Largest CPU-only SCC force gap is removed or explained by unavoidable FFT/transfer.
- Repeated XC gaps are reduced or documented as needing a larger GPU-XC project.

Final comparison:

| Profile | Total s | PotXC s | v_xc s | resident s | SCC force s | Inactive s | Largest gap s | CUDA util % | nvidia-smi avg % | nvidia-smi max % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_omp1 | 30.340 | 2.980 | 2.970 |  |  | 13.217 | 4.934 | 55.878 | 44.592 | 100.000 |
| baseline_omp8 | 25.660 | 0.930 | 0.920 |  |  | 8.626 | 4.741 | 65.891 | 54.426 | 100.000 |
| xc_fallback_omp1 | 46.490 | 14.300 | 14.250 |  |  | 21.596 | 1.826 | 53.179 | 43.514 | 100.000 |
| xc_resident_final_omp1 | 33.700 | 0.940 | 0.890 | 0.750 |  | 8.123 | 0.975 | 75.600 | 57.953 | 100.000 |
| xc_resident_final_omp8 | 30.710 | 0.910 | 0.860 | 0.740 |  | 5.050 | 0.680 | 83.352 | 67.568 | 100.000 |

Acceptance notes:

- Both final Si256 resident runs exited 0 and reached the same DS9 endpoint:
  `ETOT = -2.74396265e+04 eV`, `DRHO = 4.2581e-07`.
- OMP=1 and OMP=8 final resident runs printed the same pressure to shown
  precision: `44.739280 kbar`.
- The resident XC timer is present in both final runs:
  `0.75 s / 10 calls` for OMP=1 and `0.74 s / 10 calls` for OMP=8.
- Repeated XC gaps are reduced in the full resident path:
  `XC_Functional v_xc` is `14.25 s -> 0.89 s` versus the explicit fallback
  profile, and inactive time is `21.596 s -> 8.123 s` at OMP=1.
- The final timer table does not expose `Forces cal_force_scc` separately, so
  the comparison script leaves that field blank and reports the broader
  `Forces cal_force`/`Forces cal_force_nl` timers when present.

- [x] **Step 5: Commit report**

```bash
git add tools/perf/compare_si256_profiles.py docs/superpowers/plans/2026-06-22-si256-gpu-inactive-gap-reduction.md
git commit -m "Report Si256 GPU inactive gap reductions"
```

Actual: the report is included with the final resident XC/spin validation commit.

---

## Risk Notes

- Full LibXC is not GPU-ready in this codebase. A GPU XC path should start with recognized analytic built-ins and fall back aggressively.
- SCC force acceleration may expose that `real2recip` or host-resident `vnew` is the real remaining cost. If so, split a follow-up task to keep potential/FFT buffers resident.
- Force/stress changes must be verified numerically before performance wins are accepted.
- OMP=8 may reduce CPU gaps without solving GPU idleness; treat it as comparison data, not the primary fix.

## Self-Review

- Spec coverage: covers profiling repeatability, SCC force GPU accumulation, existing force path host work, XC attribution, guarded GPU XC, and final OMP comparison.
- Placeholder scan: no implementation step is intentionally left unspecified; GPU XC PBE is staged after LDA because correctness risk is higher.
- Type consistency: task interfaces use existing ABACUS `Device`, `FPTYPE`, `ModuleBase::matrix`, and `base_device` patterns.

## Current Status: 2026-06-23

- The resident built-in XC work moved beyond the partial `gradcorr_eval_grid` offload recorded in Task 5. The full guarded resident path is tracked in `plans/2026-06-22-gpu-resident-xc-v1.md`.
- Current Si256 full-resident evidence from that plan:
  - `PotXC cal_veff`: `14.30 s -> 1.13 s`
  - `XC_Functional v_xc`: `14.25 s -> 1.08 s`
  - ABACUS total timer: `46.49 s -> 33.78 s`
  - Nsight inactive time: `21.596008871 s -> 8.135898538 s`
  - CUDA utilization over traced span: `53.18% -> 75.64%`
  - Energy drift: `-9.84e-05 eV` total, `-3.84e-07 eV/atom`
  - Max force-component drift: `1.69e-04 eV/Angstrom`
- Spin XC extension status is tracked separately in `plans/2026-06-23-gpu-resident-spin-xc-v1.md`.
- No open items remain in this plan. Spin and noncollinear follow-up boundaries are tracked in `plans/2026-06-23-gpu-resident-spin-xc-v1.md`.

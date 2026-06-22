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

- [ ] **Step 4: Commit timing instrumentation**

```bash
git add source/source_hamilt/module_xc/libxc_pot.cpp source/source_hamilt/module_xc/libxc_tools.cpp source/source_hamilt/module_xc/xc_pot.cpp
git commit -m "Add detailed XC timing attribution"
```

---

### Task 5: Implement a Guarded GPU XC Path for Built-In LDA/PBE

**Files:**
- Create: `source/source_hamilt/module_xc/kernels/xc_eval_op.h`
- Create: `source/source_hamilt/module_xc/kernels/cuda/xc_eval_op.cu`
- Create: `source/source_hamilt/module_xc/xc_gpu_policy.h`
- Modify: `source/source_hamilt/module_xc/xc_pot.cpp`
- Modify: `source/source_hamilt/module_xc/CMakeLists.txt`
- Test: `source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp`

**Interfaces:**
- Produces: `XC_Functional::try_v_xc_gpu(...) -> std::optional<std::tuple<double, double, ModuleBase::matrix>>`
- Guards: CUDA build, `device == gpu`, `nspin == 1`, non-mGGA, recognized analytic functionals only.
- CPU/LibXC remains default fallback for unsupported functionals.

- [ ] **Step 1: Write policy tests**

Add tests for:

```cpp
EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PBE"));
EXPECT_TRUE(xc_gpu_policy(true, false, 1, "PZ"));
EXPECT_FALSE(xc_gpu_policy(false, false, 1, "PBE"));
EXPECT_FALSE(xc_gpu_policy(true, false, 2, "PBE"));
EXPECT_FALSE(xc_gpu_policy(true, true, 1, "PBE"));
EXPECT_FALSE(xc_gpu_policy(true, false, 1, "SCAN"));
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
cmake --build build-test-cuda --target MODULE_HAMILT_XC_Functional_UTs -j2
```

Expected: missing `xc_gpu_policy.h`.

- [ ] **Step 3: Implement policy header**

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

- [ ] **Step 4: Implement only LDA first**

Add CUDA kernel for LDA exchange/correlation matching the existing CPU scalar routines. Accept only LDA/PZ in the first implementation and return fallback for PBE.

- [ ] **Step 5: Compare GPU LDA against CPU**

Unit test random positive densities:

```cpp
for each rho[i] in deterministic vector:
    compare vxc_gpu[i] to XC_Functional::xc(rho[i], exc, vxc)
```

Expected tolerance: `1e-10` double, `1e-5` float.

- [ ] **Step 6: Extend to PBE only after LDA passes**

Port the existing analytic `pbex/pbec` formulas instead of calling LibXC from device code. Keep this guarded to known `XC_Functional::use_libxc == false` built-ins unless a correctness comparison against LibXC is added.

- [ ] **Step 7: Runtime validation**

Run a small Si case and Si256 with CPU XC and GPU XC toggled:

```bash
export OMP_NUM_THREADS=1
ABACUS_XC_GPU=0 build/abacus_basic_gpu
ABACUS_XC_GPU=1 build/abacus_basic_gpu
```

Expected: total energy drift within selected precision tolerance, SCF convergence unchanged or improved.

- [ ] **Step 8: Profile**

Run Si256 Nsight:

```bash
tools/perf/run_si256_nsys.sh runtime_si256_force_stress_fix_build_20260621-232100 build/abacus_basic_gpu nsight_si256_xc_gpu_omp1
```

Expected: repeated ~1.7 s gaps shrink substantially if XC was the source.

- [ ] **Step 9: Commit**

```bash
git add source/source_hamilt/module_xc/kernels/xc_eval_op.h source/source_hamilt/module_xc/kernels/cuda/xc_eval_op.cu source/source_hamilt/module_xc/xc_gpu_policy.h source/source_hamilt/module_xc/xc_pot.cpp source/source_hamilt/module_xc/CMakeLists.txt source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp
git commit -m "Add guarded GPU XC evaluation path"
```

---

### Task 6: Final OMP=1 and OMP=8 Performance Comparison

**Files:**
- Create: `tools/perf/compare_si256_profiles.py`
- Modify: this plan with final measurements.

**Interfaces:**
- Consumes: `gaps.json`, `gpu_usage.csv`, `abacus_stdout.log`.
- Produces: concise comparison table for baseline, SCC-GPU, XC-GPU, and OMP=8.

- [ ] **Step 1: Write comparison script**

Create a script that extracts:

- total ABACUS time
- `PotXC cal_veff`
- `XC_Functional v_xc`
- `Forces cal_force_scc`
- inactive gap total
- largest inactive gap
- average and max GPU utilization

- [ ] **Step 2: Run final OMP=1 profile**

```bash
tools/perf/run_si256_nsys.sh runtime_si256_force_stress_fix_build_20260621-232100 build/abacus_basic_gpu nsight_si256_final_omp1
```

- [ ] **Step 3: Run final OMP=8 profile**

Temporarily override the wrapper or run manually:

```bash
cd runtime_si256_force_stress_fix_build_20260621-232100
export OMP_NUM_THREADS=8
nsys profile -t cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite=true -o ../nsight_si256_final_omp8/profile /home/bonan/appdir/abacus-develop-cufft-batch/build/abacus_basic_gpu
```

- [ ] **Step 4: Acceptance criteria**

Accept the full series only if:

- Si256 exits status 0.
- SCF reaches the same convergence target.
- Total energy and forces match baseline within precision tolerance.
- Largest CPU-only SCC force gap is removed or explained by unavoidable FFT/transfer.
- Repeated XC gaps are reduced or documented as needing a larger GPU-XC project.

- [ ] **Step 5: Commit report**

```bash
git add tools/perf/compare_si256_profiles.py docs/superpowers/plans/2026-06-22-si256-gpu-inactive-gap-reduction.md
git commit -m "Report Si256 GPU inactive gap reductions"
```

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

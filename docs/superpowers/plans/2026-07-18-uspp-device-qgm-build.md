# USPP Device QGM Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build persistent USPP `qgm` and atom-phase caches directly on CUDA/ROCm devices, eliminating full host cache materialization and reducing GPU startup time.

**Architecture:** `pseudopot_cell_vnl` will pack projector-pair angular metadata on the host, upload only compact metadata plus the existing `qrad` table, and invoke a bulk device kernel over `(type, pair, G)`. The existing atom-phase device kernel will fill `qgm_phase`; CPU and stress-derivative paths remain unchanged, and existing cache getters keep their layouts.

**Tech Stack:** C++11, CUDA, HIP/ROCm, ABACUS device memory APIs, existing real-spherical-harmonic kernels, GoogleTest, CMake/CTest, Nsight Systems.

## Global Constraints

- Preserve CPU cache construction and numerical behavior.
- Preserve `qgm[type][pair][G]` and `qgm_phase[atom][G]` layouts.
- Keep `dqgm` host-built in this slice.
- Add no INPUT parameters or default arguments.
- Keep the PR-level `GlobalV`/`GlobalC`/`PARAM` dependency budget non-increasing.
- Keep C++11 compatibility.
- Check that the GPU is free before every GPU test, sanitizer run, profile, or benchmark.

---

### Task 1: Bulk Device QGM Kernel

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/nonlocal_op.h`
- Modify: `source/source_pw/module_pwdft/kernels/nonlocal_op.cpp`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/nonlocal_op.cu`
- Modify: `source/source_pw/module_pwdft/kernels/rocm/nonlocal_op.hip.cu`
- Test: `source/source_pw/module_pwdft/kernels/test/nonlocal_op_test.cpp`

**Interfaces:**
- Consumes: flattened `qrad[type][l][radial_pair][iq]`, `ylm[lm][G]`, and packed per-augmentation-pair term metadata.
- Produces: `uspp_qgm_build_op<FPTYPE, Device>::operator()` filling `qgm[type][augmentation_pair][G]`.

- [ ] **Step 1: Add an independent CPU reference and a compile-red GPU test**

Create `TestModuleHamiltUsppQgm.gpu_matches_independent_reference` with two atom types, inactive padded pairs, multiple angular terms, nonuniform reciprocal vectors, and four-point interpolation values. The test must construct expected values directly:

```cpp
for (int term = 0; term < term_count[pair]; ++term)
{
    const int iq = static_cast<int>(qnorm / dq);
    const double x0 = qnorm / dq - iq;
    const double work = qrad0 * (1-x0) * (2-x0) * (3-x0) / 6
                      + qrad1 * x0 * (2-x0) * (3-x0) / 2
                      - qrad2 * (1-x0) * x0 * (3-x0) / 2
                      + qrad3 * (1-x0) * (2-x0) * x0 / 6;
    expected += coefficient[pair_term] * work * ylm[lm * npw + ig];
}
```

- [ ] **Step 2: Build to verify the new test fails**

Run:

```bash
cmake --build build-cuda-charge-mixing-compile --target MODULE_PW_Hamilt_Kernels_UTs -j2
```

Expected: compilation fails because `uspp_qgm_build_op` is not declared.

- [ ] **Step 3: Declare the explicit bulk-builder interface**

Add CPU and GPU specializations with no default arguments:

```cpp
template <typename FPTYPE, typename Device>
struct uspp_qgm_build_op
{
    void operator()(const Device* ctx,
                    int ntype,
                    int nh_tot,
                    int npw,
                    int lmaxq,
                    int radial_pair_count,
                    int nqxq,
                    int max_terms,
                    FPTYPE dq,
                    FPTYPE tpiba,
                    const FPTYPE* gcar,
                    const int* pair_term_count,
                    const int* pair_radial_index,
                    const int* pair_l,
                    const int* pair_lm,
                    const std::complex<FPTYPE>* pair_coefficient,
                    const FPTYPE* qrad,
                    const FPTYPE* ylm,
                    std::complex<FPTYPE>* qgm);
};
```

- [ ] **Step 4: Implement CPU, CUDA, and ROCm parity kernels**

Map one device thread to one output `(type, pair, G)`. Compute `qnorm = |G| * tpiba`, perform the exact four-point interpolation used by `PolyInt`, sum the packed angular terms, and write zero for padded inactive pairs. Guard `iq > nqxq - 4` by writing zero, matching the pointer interpolation helper's out-of-range behavior.

- [ ] **Step 5: Run the focused kernel tests**

After an idle-GPU check, run:

```bash
./build-cuda-charge-mixing-compile/source/source_pw/module_pwdft/kernels/test/MODULE_PW_Hamilt_Kernels_UTs \
  --gtest_filter='TestModuleHamiltUsppQgm.*:TestModuleHamiltUsppOverlap.*'
```

Expected: all selected CPU/CUDA tests pass.

---

### Task 2: Direct Device Cache Construction

**Files:**
- Modify: `source/source_pw/module_pwdft/vnl_pw.h`
- Modify: `source/source_pw/module_pwdft/vnl_pw.cpp`
- Modify: `source/source_pw/module_pwdft/setup_pot.cpp`
- Modify: `source/source_estate/elecstate_pw.cpp`
- Test: `source/source_pw/module_pwdft/kernels/test/nonlocal_op_test.cpp`

**Interfaces:**
- Consumes: `uspp_qgm_build_op<double, DEVICE_GPU>` and `uspp_atom_phase_op<double, DEVICE_GPU>`.
- Produces: persistent `z_qgm`, `z_qgm_phase`, and `d_qgm_gcar`; `has_qgm_cache()` reports availability independently of host `qgm` size.

- [ ] **Step 1: Add cache-availability coverage**

Add a small policy test proving GPU cache availability can be true while host `qgm` is empty. Introduce the intended query:

```cpp
bool pseudopot_cell_vnl::has_qgm_cache() const;
```

Expected red result: compilation fails before the query exists.

- [ ] **Step 2: Add explicit cache state**

Add `bool qgm_cache_ready = false` and implement:

```cpp
bool has_qgm_cache() const { return qgm_cache_ready; }
```

Set it only after all active cache pointers are populated. Clear it in initialization, release, and destruction paths. Replace `qgm.getSize() > 0` availability checks in `setup_pot.cpp` and `elecstate_pw.cpp`.

- [ ] **Step 3: Pack compact pair metadata**

For `pair_count = ntype * nh_tot` and `max_terms = lmaxq * lmaxq`, build host vectors:

```cpp
std::vector<int> term_count(pair_count, 0);
std::vector<int> radial_index(pair_count, 0);
std::vector<int> term_l(pair_count * max_terms, 0);
std::vector<int> term_lm(pair_count * max_terms, 0);
std::vector<std::complex<double>> coefficient(pair_count * max_terms, {0.0, 0.0});
```

Fill actual symmetric `(ih,jh)` pairs using `indv`, `nhtolm`, `lpx`, `lpl`, and `ap`; keep padded pairs inactive. Precompute `(-i)^l * ap` on the host.

- [ ] **Step 4: Build double caches directly on GPU**

In the `use_gpu_` branch:

1. allocate `d_qgm_gcar`, `z_qgm`, and `z_qgm_phase`;
2. copy only `gcar`, compact metadata, `qrad`, and fractional atom coordinates;
3. compute device Ylm with `ModuleBase::YlmReal::Ylm_Real`;
4. invoke `uspp_qgm_build_op<double, DEVICE_GPU>`;
5. invoke `uspp_atom_phase_op<double, DEVICE_GPU>`;
6. cast device double caches to float when `s_deeq != nullptr`;
7. free all setup-only device buffers.

Do not call `qgm.create` or `qgm_phase.create` in this branch.

- [ ] **Step 5: Preserve stress-only host preparation**

When `prepare_uspp_stress` is true, retain host `ylmk0`, `qnorm`, and `dqgm` construction and copy only `dqgm` to `z_dqgm`. This branch must not materialize host `qgm` or host phases.

- [ ] **Step 6: Preserve the CPU branch**

Keep existing host `qgm`, phase, and pointer-alias behavior under `!use_gpu_`. Set `qgm_cache_ready` after construction.

- [ ] **Step 7: Build the executable and focused target**

Run:

```bash
cmake --build build-cuda-charge-mixing-compile --target abacus_pw_gpu MODULE_PW_Hamilt_Kernels_UTs -j2
```

Expected: both targets link successfully.

---

### Task 3: Numerical, Memory, And Performance Verification

**Files:**
- Update: `.planning/2026-07-13-mixing-support-matrix/findings.md`
- Update: `.planning/2026-07-13-mixing-support-matrix/progress.md`

**Interfaces:**
- Consumes: final direct-device cache implementation.
- Produces: parity, memory-safety, and startup-performance evidence.

- [ ] **Step 1: Run focused CUDA tests**

After checking the GPU is idle, run the QGM, phase, overlap, effective-D, force, and stress test filters. Expected: all pass.

- [ ] **Step 2: Run a host-built reference and device-built Si32 case**

Use the existing 32-atom 30/120 Ry, 2x2x2 USPP case. Retain a temporary test-only environment switch for the host reference only if required, and remove it before completion. Compare final energy, 432 eigenvalues, occupations, augmentation density restart, force, and effective-D observables within existing double-precision tolerances.

- [ ] **Step 3: Validate retained stress behavior**

Run a small USPP stress calculation with the device-built QGM path and host-built `dqgm`. Compare the stress tensor with the pre-change reference.

- [ ] **Step 4: Run compute-sanitizer**

After another idle-GPU check, run:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  env OMP_NUM_THREADS=1 /absolute/path/to/abacus_pw_gpu
```

Expected: `ERROR SUMMARY: 0 errors`.

- [ ] **Step 5: Verify host-cache elimination**

Confirm ABACUS memory reporting and a debugger/profile trace show no GPU-path allocations for full host `qgm` or `qgm_phase`, and no host-to-device copies of those arrays. Device `z_qgm` and `z_qgm_phase` remain persistent by design.

- [ ] **Step 6: Benchmark Si256 startup**

After confirming the GPU is idle, rerun the exact 256-atom 30/120 Ry, 2x2x2, 520-band `dav_subspace` benchmark. Record `init_vnl`, total startup, electronic iteration, and total wall time against the 13.25-second and 399.08-second baselines.

- [ ] **Step 7: Run final repository checks**

Run:

```bash
git diff --check
python3 tools/03_code_analysis/agent_governance_check.py --staged
```

Use a temporary index for the scoped governance check. Expected: no errors; document migration-neutral warnings and why no user documentation update is required.

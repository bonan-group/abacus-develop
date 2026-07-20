# GPU Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the confirmed GPU correctness, API, CI, and build-profile defects while preserving one-rank-per-pool GPU FFT and the global float-FFTW default.

**Architecture:** Keep unsupported distributed GPU FFT behavior explicit through shared policy helpers, repair numerical ownership and sizing at their existing boundaries, and exercise each bug through the closest existing unit or workflow test. Avoid new global dependencies and avoid broader refactoring.

**Tech Stack:** ABACUS C++11, CUDA/ROCm, CMake/CTest, GoogleTest, MPI, existing PW GPU integration harness.

## Global Constraints

- GPU plane-wave FFT remains limited to one MPI rank per pool.
- `ENABLE_FLOAT_FFTW` remains globally `OFF`.
- GPU profiles supporting `precision single` or `precision mixing` explicitly enable float FFTW.
- Do not add `GlobalV`, `GlobalC`, or `PARAM` references.
- Run GPU and MPI tests outside the sandbox with `OMP_NUM_THREADS=1`.
- Do not commit changes unless the user asks for a commit.

---

### Task 1: Explicit GPU PW/USPP rank policy

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/test/vnl_op_test.cpp`
- Modify: `source/source_pw/module_pwdft/vnl_pw.h`
- Modify: `source/source_pw/module_pwdft/vnl_pw.cpp:2092-2103`

**Interfaces:**
- Produces: `gpu_pw_fft_pool_supported(int poolnproc) -> bool`
- Consumes: `ModulePW::PW_Basis::poolnproc`

- [x] **Step 1: Add the failing policy test**

```cpp
TEST(TestSrcPWVnlPolicy, gpuPwFftRequiresOneRankPerPool)
{
    EXPECT_TRUE(gpu_pw_fft_pool_supported(1));
    EXPECT_FALSE(gpu_pw_fft_pool_supported(2));
}
```

- [x] **Step 2: Build the focused target and verify RED**

Run: `cmake --build build-cuda-charge-mixing-compile --target MODULE_PW_Hamilt_Kernels_UTs -j2`

Expected: compile failure because `gpu_pw_fft_pool_supported` is not declared.

- [x] **Step 3: Add the C++11 inline policy and use it before GPU effective-D allocation**

```cpp
inline bool gpu_pw_fft_pool_supported(const int poolnproc)
{
    return poolnproc == 1;
}
```

At the start of `cal_effective_D_gpu`, reject a null basis or unsupported pool layout with the existing one-rank-per-pool diagnostic before allocating `vaux` or launching kernels.

- [x] **Step 4: Rebuild and run the focused policy test**

Run the target above, then:

`build-cuda-charge-mixing-compile/source/source_pw/module_pwdft/kernels/test/MODULE_PW_Hamilt_Kernels_UTs --gtest_filter=TestSrcPWVnlPolicy.gpuPwFftRequiresOneRankPerPool`

Expected: PASS.

### Task 2: Safe zero-projector chunk sizing

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/test/vnl_op_test.cpp`
- Modify: `source/source_pw/module_pwdft/vnl_pw.h`
- Modify: `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1173-1181,1392-1400`

**Interfaces:**
- Produces: `projector_atom_chunk_size(int remaining_atoms, int target_chunk, int nh) -> int`
- Consumes: force/stress chunk-loop remaining atom count, target projector count, projectors per atom.

- [x] **Step 1: Add the failing chunk-policy test**

```cpp
TEST(TestSrcPWVnlPolicy, zeroProjectorTypeHasNoChunk)
{
    EXPECT_EQ(projector_atom_chunk_size(3, 64, 0), 0);
    EXPECT_EQ(projector_atom_chunk_size(3, 64, 4), 3);
    EXPECT_EQ(projector_atom_chunk_size(20, 32, 4), 8);
}
```

- [x] **Step 2: Rebuild and verify RED**

Expected: compile failure because the helper is absent.

- [x] **Step 3: Implement the helper and use it in both loops**

```cpp
inline int projector_atom_chunk_size(const int remaining_atoms,
                                     const int target_chunk,
                                     const int nh)
{
    if (remaining_atoms <= 0 || nh <= 0)
    {
        return 0;
    }
    return std::max(1, std::min(remaining_atoms, target_chunk / nh));
}
```

Both force and stress loops skip the atom type when the helper returns zero; neither evaluates `target_chunk / nh` directly.

- [x] **Step 4: Rebuild and run both VNL policy tests**

Expected: PASS.

### Task 3: Smooth-grid cast bounds

**Files:**
- Modify: `source/source_estate/test/potential_new_test.cpp`
- Modify: `source/source_estate/module_pot/potential_new.h`
- Modify: `source/source_estate/module_pot/potential_new.cpp:335-340`
- Modify: `tests/11_PW_GPU/mixing_plain_spin2_double_grid/INPUT`

**Interfaces:**
- Produces: private `Potential::smooth_potential_size() const -> size_t`
- Consumes: `veff_smooth.nr` and `veff_smooth.nc`

- [ ] **Step 1: Add a failing unit test with different dense and smooth sizes**

Construct `Potential(rhodpw, rhopw, ...)` with `rhodpw->nrxx = 100` and `rhopw->nrxx = 40`, then assert `smooth_potential_size() == nspin * 40`.

- [ ] **Step 2: Build `MODULE_ESTATE_potentials_new` and verify RED**

Expected: compile failure because the helper is absent.

- [ ] **Step 3: Implement the helper and replace the dense cast count**

```cpp
size_t Potential::smooth_potential_size() const
{
    return static_cast<size_t>(this->veff_smooth.nr) * this->veff_smooth.nc;
}
```

Use it for the line-339 device cast.

- [ ] **Step 4: Make one existing double-grid GPU workflow single precision**

Change `mixing_plain_spin2_double_grid/INPUT` from `precision double` to `precision single`, retaining other double-grid cases for double coverage.

- [ ] **Step 5: Run the unit test and later the CUDA workflow**

Expected: unit PASS; the integration completes without CUDA illegal access and retains its reference tolerance.

### Task 4: CUDA Ewald G=0 ownership

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:2310-2317`

**Interfaces:**
- Consumes: existing `ig0`; no signature change.
- Produces: zero diagonal initialization on ranks where `ig0 < 0`.

- [ ] **Step 1: Add a failing non-owner CUDA operator test**

Call the operator with `ig0 = -1`, `do_real_space = 0`, and a minimal nonzero-G list; assert the standalone diagonal constant is absent.

- [ ] **Step 2: Run only the new gtest and verify RED**

Expected: line-2315 initialization produces a nonzero diagonal value.

- [ ] **Step 3: Gate the diagonal initializer**

Zero the output on every rank, then call `set_stress_ewa_diag_kernel` only when `ig0 >= 0`. Preserve G-space work on every rank and real-space ownership through `do_real_space`.

- [ ] **Step 4: Run owner and non-owner Ewald tests**

Expected: both PASS.

### Task 5: Explicit-spin resident XC dispatch

**Files:**
- Modify: `source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp`
- Modify: `source/source_hamilt/module_xc/xc_pot.cpp:60-165,326-345,630-658,840-875`

**Interfaces:**
- Add `const int nspin` to the three anonymous resident-selector functions.
- Callers pass their explicit overload argument, not `PARAM.inp.nspin`.

- [ ] **Step 1: Add a failing explicit/global spin mismatch test**

Set global spin to 2, construct a two-channel resident `Charge`, then call the explicit overload with `nspin = 1` for built-in PBE. Compare against the explicit CPU result and assert a one-row matrix.

- [ ] **Step 2: Run the new test and verify RED**

Expected: current spin selector follows global spin and either overwrites the one-row result or disagrees with CPU.

- [ ] **Step 3: Thread explicit spin through all selectors**

Replace each `PARAM.inp.nspin` eligibility check with the selector's new `nspin` parameter and update both matrix-returning and device-accumulating call sites.

- [ ] **Step 4: Run mismatch plus existing resident LDA/PBE tests**

Expected: all PASS.

### Task 6: Remove undefined API and execute GPU unit tests in CI

**Files:**
- Modify: `source/source_hamilt/module_xc/xc_functional.h:244-252`
- Modify: `.github/workflows/cuda.yml:53-56`

**Interfaces:**
- Remove only the undefined short `gradcorr` declaration.
- Preserve the full explicit and device overloads.

- [ ] **Step 1: Confirm no caller or definition exists**

Run: `rg -n "gradcorr\\(" source --glob '*.{h,cpp,cu}'`

- [ ] **Step 2: Remove the declaration and its default argument**

- [ ] **Step 3: Add `MODULE_HAMILT_XC_Functional_UTs` and `MODULE_PSI_init_gpu_test` to the CUDA CTest regex**

- [ ] **Step 4: Validate YAML and list the selected CTest targets**

Expected: both names are selected by the workflow expression.

### Task 6a: Restore ROCm gamma-only inverse parity

**Files:**
- Modify: `source/source_basis/module_pw/kernels/pw_op.h`
- Modify: `source/source_basis/module_pw/kernels/rocm/pw_op.hip.cu`
- Modify: `source/source_basis/module_pw/kernels/test/pw_op_test.cpp`
- Modify: `source/source_basis/module_pw/pw_transform_gpu.cpp`

- [x] Make the existing gamma reconstruction tests backend-neutral.
- [x] Implement and instantiate the HIP gamma reconstruction operator.
- [x] Route both ROCm inverse overloads through the gamma operator.
- [x] Build and run the CUDA parity tests; record that ROCm hardware/toolchain verification is unavailable locally.

### Task 7: Float-FFTW GPU profiles and documentation

**Files:**
- Modify: `CMakeLists.txt` near feature options
- Modify: `Dockerfile.cuda`
- Modify: `.github/workflows/build_test_cmake.yml`
- Modify: `toolchain/README.md` CUDA configuration examples
- Modify: relevant GPU build documentation discovered during implementation

**Interfaces:**
- No option default change.
- Configure-time warning only when `(USE_CUDA OR USE_ROCM) AND NOT ENABLE_FLOAT_FFTW`.

- [ ] **Step 1: Capture current configuration output without float FFTW**

Configure a fresh `/tmp` build with `USE_CUDA=ON`, `ENABLE_FLOAT_FFTW=OFF`, and `GIT_SUBMODULE=OFF`; verify the intended actionable warning is absent.

- [ ] **Step 2: Add the warning**

The message states that double precision remains available and that single/mixing precision requires `-DENABLE_FLOAT_FFTW=ON`.

- [ ] **Step 3: Enable float FFTW in CUDA profiles advertising general GPU support**

Add `-DENABLE_FLOAT_FFTW=ON` to `Dockerfile.cuda`, CUDA CMake CI, and the documented CUDA toolchain commands. Apply the same rule to any concrete ROCm profile found; do not invent a new profile.

- [ ] **Step 4: Reconfigure and verify the warning/default behavior**

Expected: warning appears with the option off, disappears with it on, and the cache default remains `OFF`.

### Task 8: Verification and handoff

**Files:**
- Update: `.planning/2026-07-20-gpu-review-fixes/{task_plan,findings,progress}.md`

- [ ] **Step 1: Run focused CPU builds/tests**

Run the potential and policy tests available without a GPU.

- [ ] **Step 2: Run focused CUDA tests outside the sandbox**

Run Ewald owner/non-owner, XC explicit-spin mismatch plus resident neighbors, psi initialization, mixing, and PW kernel tests with `OMP_NUM_THREADS=1`.

- [ ] **Step 3: Run one-rank GPU workflow cases outside the sandbox**

Run `CASES_MIXING_GPU.txt` and `CASES_USPP_MIXING_GPU.txt` with `-n 1`.

- [ ] **Step 4: Run static and governance checks**

Run `git diff --check` and the merge-base-scoped governance checker. Record the pre-existing branch-wide global dependency result separately from any newly introduced delta.

- [ ] **Step 5: Review the final diff and report exact evidence**

State every command, pass/fail result, and any unavailable ROCm verification.

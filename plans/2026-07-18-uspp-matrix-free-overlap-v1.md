# USPP Matrix-Free Overlap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PW USPP `S*psi` reuse nonlocal projector coefficients, support matrix-free projectors, and avoid per-call overlap allocations.

**Architecture:** The PW nonlocal operator owns one-shot projector coefficient validity and USPP overlap application because it already owns full and on-the-fly projector state. `HamiltPW::sPsi` copies the identity contribution and delegates augmentation to that operator with explicit active and leading dimensions. Full-projector reconstruction uses the persistent `becp` and `ps` buffers; memory-saving reconstruction materializes bounded atom-aligned projector chunks and uses GEMM without allocating the full projector matrix.

**Tech Stack:** C++11, CUDA/ROCm device kernels, cuBLAS/rocBLAS-backed ABACUS math operators, GoogleTest, CMake/CTest, ABACUS PW GPU runtime.

## Global Constraints

- Keep C++11 compatibility and avoid new default arguments.
- Do not add new `GlobalV`, `GlobalC`, or `PARAM` control dependencies.
- Keep the CPU full-projector path numerically equivalent.
- Run all GPU and MPI-linked runtime validation outside the sandbox after confirming the GPU is idle.
- Do not optimize host `qgm` construction in this slice; it is a separate one-time startup concern.

---

### Task 1: Projector-Major USPP Coefficient Layout

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/nonlocal_op.h`
- Modify: `source/source_pw/module_pwdft/kernels/nonlocal_op.cpp`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/nonlocal_op.cu`
- Modify: `source/source_pw/module_pwdft/kernels/rocm/nonlocal_op.hip.cu`
- Test: `source/source_pw/module_pwdft/kernels/test/nonlocal_op_test.cpp`

**Interfaces:**
- Consumes: band-major `becp[ib*nkb+jkb]`.
- Produces: `uspp_overlap_op(..., bool projector_major, ps, becp)`, writing either `ps[ib*nkb+jkb]` or `ps[jkb*nbands+ib]`.

- [x] Add a test that calls `uspp_overlap_op` with `projector_major=true` and compares every value with an independently transposed CPU reference.
- [x] Build and run the focused kernel test to verify the new call fails before the interface exists.
- [x] Add the explicit layout flag to CPU, CUDA, and ROCm declarations and implementations.
- [x] Update existing callers/tests to pass `false` for the legacy band-major layout.
- [x] Run the focused CPU/CUDA kernel tests and verify both layouts pass.

### Task 2: Operator-Owned USPP Overlap

**Files:**
- Modify: `source/source_pw/module_pwdft/op_pw_nl.h`
- Modify: `source/source_pw/module_pwdft/op_pw_nl.cpp`
- Modify: `source/source_pw/module_pwdft/hamilt_pw.h`
- Modify: `source/source_pw/module_pwdft/hamilt_pw.cpp`
- Runtime regression: `/tmp/si32-uspp-nc-profile/uspp-matrix-free`

**Interfaces:**
- Produces: `Nonlocal<OperatorPW<T, Device>>::apply_uspp_overlap(const T* psi, T* spsi, int nrow, int npw, int nbands) const`.
- Maintains: one-shot cached coefficient identity `(psi pointer, ik, npw, nrow, nbands)` and invalidates it after overlap consumption, on k-point changes, and when explicit dimensions change.
- Consumes: existing persistent `becp`, `ps`, full `vkb`, and matrix-free projector caches.

- [x] Re-run the forced matrix-free Si32 case and retain the observed `CUBLAS_STATUS_INVALID_VALUE` as the red integration result.
- [x] Record valid `becp` metadata after full and chunked nonlocal H application; invalidate it after one overlap use and on k-point/dimension changes.
- [x] Add helpers that recompute `becp` only when one-shot metadata does not match, using full GEMM or bounded projector chunks as appropriate.
- [x] Apply `qq` into persistent `ps`, using band-major output for full and chunked GEMM reconstruction.
- [x] Store the nonlocal operator as its existing base pointer in `HamiltPW`, then delegate `sPsi` augmentation to the typed operator in the implementation file.
- [x] Remove the duplicate local `becp`/`ps` allocation and direct full-`vkb` implementation from `HamiltPW::sPsi`.
- [x] Rebuild the CUDA PW executable and focused kernel target.
- [x] Run the forced matrix-free Si32 case and verify it completes without cuBLAS errors.
- [x] Compare full and matrix-free final energy, `DRHO`, eigenvalues, and charge-density restart data within existing double-precision tolerances.

### Task 3: Regression And Performance Verification

**Files:**
- Update evidence: `.planning/2026-07-13-mixing-support-matrix/findings.md`
- Update evidence: `.planning/2026-07-13-mixing-support-matrix/progress.md`

**Interfaces:**
- Consumes: the fixed full and matrix-free overlap paths.
- Produces: numerical and performance evidence for deciding whether further kernel work is justified.

- [x] Run focused CPU and CUDA kernel tests.
- [x] Run the existing USPP mixing/runtime cases affected by PW Hamiltonian overlap.
- [x] Run `git diff --check` and the governance checker.
- [x] Check the GPU is idle, then rerun matched Si32 USPP full, USPP matrix-free, and NC baselines.
- [x] Capture a fresh Nsight Systems USPP profile and compare GEMM, synchronous-copy API, and total solver time against the pre-fix report.
- [x] Run the 256-atom `30/120 Ry`, `2x2x2`, USPP benchmark only after the smaller-case numerical and profiling checks pass.

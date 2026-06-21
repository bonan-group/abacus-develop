# CUDA Nonlocal Projector Stress Acceleration Plan

## Objective

Reduce the remaining `stress_nl` cost in the Si256 `dav_subspace` CUDA profile after the Ewald port. The current post-Ewald profile shows total runtime around 47.5 s and `stress_nl` around 9.4 s, making nonlocal projector stress the dominant remaining stress component. The goal is to keep the nonlocal projector stress path device-resident where practical, reuse existing CUDA kernels first, and avoid large new custom kernels until profiling proves they are needed.

## Current Understanding

- `Stress_Func::stress_nl` has a CUDA chunked path and falls back to the full path only when chunking is unavailable. The chunked call is selected before the old six-component full-projector path.
- The chunked implementation already performs the final nonlocal stress accumulation through `cal_stress_nl_op::chunk`, so the largest remaining cost is unlikely to be the final stress contraction alone.
- The chunked path still performs important preparation work before and inside the six stress-component loop: `g+k` construction, structure factors, Ylm, Ylm derivatives, radial interpolation, projector construction, GEMM, MPI pool reduction, and final contraction.
- Existing CUDA support already covers several pieces we should reuse:
  - `cal_vkb`, `cal_vkb_deri`, `cal_vq`, and `cal_vq_deri` kernels in `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu`.
  - `cal_stress_nl_op::chunk` interfaces in `source/source_pw/module_pwdft/kernels/stress_op.h`.
  - `ModuleBase::YlmReal::Ylm_Real(ctx, ...)`, which dispatches to `cal_ylm_real_op` and already has CUDA tests.
  - The matrix-free nonlocal Hamiltonian path already caches `g+k`, Ylm, `vkb1`, structure factors, and projector metadata per k-point in `Nonlocal<OperatorPW<T, Device>>::ensure_kpoint_caches`.

## Code References

- Main stress entry: `source/source_pw/module_pwdft/stress_nl.cpp:12`
- CUDA chunk dispatch: `source/source_pw/module_pwdft/stress_nl.cpp:54`
- Full fallback six-component loop: `source/source_pw/module_pwdft/stress_nl.cpp:65`
- Chunked stress implementation: `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1271`
- Current CPU `g+k`, Ylm, and Ylm derivative helpers: `source/source_pw/module_pwdft/nonlocal_maths.hpp:128`, `source/source_pw/module_pwdft/nonlocal_maths.hpp:152`, `source/source_pw/module_pwdft/nonlocal_maths.hpp:177`
- Current per-chunk radial/projector setup: `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:241`, `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:294`
- Existing CUDA projector kernels: `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:1055`, `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:1075`, `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:1117`, `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:1133`
- Existing chunk contraction interface: `source/source_pw/module_pwdft/kernels/stress_op.h:377`
- Existing CUDA Ylm dispatch: `source/source_base/math_ylmreal.cpp:301`
- Existing nonlocal Hamiltonian k-point cache: `source/source_pw/module_pwdft/op_pw_nl.cpp:780`

## Implementation Plan

- [ ] Add fine-grained timing for the chunked nonlocal stress path.
  - [ ] Time `cal_gk`, structure-factor setup, `cal_ylm`, `cal_ylm_deri`, `cal_vkb_type_chunk`, base `becp` GEMM, pool reduction, each derivative-projector build, each derivative GEMM, and final `cal_stress_nl_op::chunk`.
  - [ ] Keep timers behind existing ABACUS timer mechanisms so they can be left in place or removed cleanly after profiling.
  - [ ] Re-run the Si256 `dav_subspace` stress profile to confirm whether the dominant part is Ylm derivative setup, radial interpolation/projector setup, GEMM, MPI reduction, or final contraction.

- [ ] Reuse and hoist invariant data inside `FS_Nonlocal_tools::cal_stress_chunked`.
  - [ ] Upload or cache `d_vq_tab` once for the stress call instead of recasting or copying it through repeated helper paths.
  - [ ] Move base `cal_vq` work out of repeated derivative-component paths when the data is invariant for a given type and k-point.
  - [ ] Move derivative radial interpolation `cal_vq_deri` out of the six tensor-component loop when it is independent of `ipol` and `jpol`.
  - [ ] Split component-dependent index/prefactor preparation from component-independent radial interpolation so the cheap scalar metadata remains on CPU only where necessary.
  - [ ] Verify that the hoisted data is valid for all atom chunks of the same type and is not accidentally tied to `atom_start`.

- [ ] Replace CPU Ylm generation in the chunked stress path with the existing CUDA Ylm dispatch.
  - [ ] Change `Nonlocal_maths::cal_ylm` or add a stress-specific helper so the CUDA path calls `ModuleBase::YlmReal::Ylm_Real(gpu_ctx, ...)` directly on device `g+k`.
  - [ ] Remove the current CPU temporary plus host-to-device copy for GPU stress when the input `q` is already device-resident.
  - [ ] Add focused tests comparing CPU and GPU Ylm output for small deterministic `g+k` arrays across the `lmaxkb` range used by pseudopotentials.

- [ ] Add a CUDA path for Ylm derivatives using the lowest-risk reuse strategy first.
  - [ ] Implement device finite-difference derivative generation by reusing existing CUDA Ylm evaluation on perturbed `g+k` arrays.
  - [ ] Generate the three derivative directions once per k-point, not inside each of the six stress tensor components.
  - [ ] Store derivatives in the same layout currently consumed by `cal_vkb_deri`.
  - [ ] Preserve the CPU finite-difference implementation as the reference path and avoid changing CPU behavior.
  - [ ] Add tests comparing GPU derivative output against the CPU `dylmr2` path within existing single/double GPU tolerances.

- [ ] Move `g+k` construction toward device residency.
  - [ ] First, reuse the matrix-free nonlocal Hamiltonian cache if the same k-point projector cache is available and valid for stress.
  - [ ] If the Hamiltonian cache cannot be shared safely, add a small CUDA helper that builds `g+k` and its norm array directly from device or staged reciprocal-grid data.
  - [ ] Keep the CPU `getgpluskcar` loop as fallback for CPU and non-CUDA paths.
  - [ ] Verify the zero-vector and Gamma-point handling matches the current CPU helper.

- [ ] Reuse nonlocal Hamiltonian projector caches where ownership is clean.
  - [ ] Audit whether `Nonlocal<OperatorPW>::ensure_kpoint_caches` can expose cached `g+k`, Ylm, `vkb1`, structure factors, and projector metadata to stress without introducing hidden mutable state or lifetime hazards.
  - [ ] Prefer a small shared cache object or helper owned by the PW nonlocal module over duplicating long-lived buffers inside stress.
  - [ ] Keep cache invalidation explicit on k-point, `npw`, pseudopotential metadata, atom count, and precision.
  - [ ] Do not route stress through Hamiltonian application code if that would blur responsibilities; share only projector preparation data and kernels.

- [ ] Reduce six-component repeated projector work after the preparation path is device-resident.
  - [ ] Profile again after Ylm/Ylm-derivative and radial interpolation hoisting.
  - [ ] If derivative GEMMs dominate, add an optional six-component batched mode using existing GEMM/batched-GEMM wrappers where available.
  - [ ] If memory pressure is acceptable, allocate derivative-projector and `dbecp` workspaces for multiple tensor components at once.
  - [ ] If memory pressure is high for large cells, keep the sequential component loop but reuse all invariant device buffers.

- [ ] Clean up allocation behavior in nonlocal stress workspaces.
  - [ ] Ensure `vkb_chunk`, `becp_chunk`, `dbecp_chunk`, Ylm, derivative Ylm, radial tables, and index/prefactor buffers are allocated through reusable workspace ownership rather than per-component allocation.
  - [ ] Check the Nsight Systems profile for `cudaMalloc` or synchronous copy noise inside `stress_nl`.
  - [ ] Keep the existing chunk-size override behavior for controlled memory/performance testing.

## Verification Plan

- [ ] Run focused kernel/unit tests for Ylm CPU/GPU agreement.
- [ ] Add or extend tests for GPU Ylm derivative generation against CPU `dylmr2`.
- [ ] Run existing `MODULE_PW_Hamilt_Kernels_UTs` after every kernel-interface change.
- [ ] Run a small deterministic PW stress case with `cal_stress=1` and compare nonlocal stress against the current CPU/CUDA baseline.
- [ ] Run the Si256 `dav_subspace`, `precision single`, `cal_force=1`, `cal_stress=1` benchmark and compare final energy, force, and stress against the post-Ewald baseline within existing GPU tolerances.
- [ ] Re-run Nsight Systems for the same Si256 case and confirm:
  - [ ] `stress_nl` is reduced by at least 30% in the first optimization pass.
  - [ ] no new GPU-idle gap above 1 s appears in stress.
  - [ ] allocator and synchronous-copy events inside `stress_nl` are reduced or eliminated.

## Risks And Mitigations

- [ ] Ylm derivative correctness is the largest mathematical risk.
  - Mitigation: start with GPU finite differences using the existing Ylm CUDA implementation, compare directly against CPU `dylmr2`, and defer analytic derivatives until correctness and profile data justify them.
- [ ] Cache sharing with the Hamiltonian nonlocal path can introduce lifetime and invalidation bugs.
  - Mitigation: make cache ownership and invalidation explicit, and prefer read-only reuse of prepared arrays over calling Hamiltonian application routines from stress.
- [ ] MPI pool reduction may remain a visible CPU synchronization point.
  - Mitigation: profile it separately before changing it; only optimize after projector preparation and GEMM costs are understood.
- [ ] Six-component batching may increase memory footprint.
  - Mitigation: add it only after profiling, make it conditional on workspace capacity, and keep the current sequential loop as fallback.
- [ ] Noncollinear and ultrasoft paths may exercise different metadata and tensor shapes.
  - Mitigation: keep the first implementation for the existing chunked CUDA path, retain fallback behavior, and add tests for `npol=1` before expanding coverage.

## Acceptance Criteria

- [ ] CUDA behavior is unchanged for cases that do not enter the chunked nonlocal stress path.
- [ ] CPU behavior is unchanged.
- [ ] Existing nonlocal stress tests and PW Hamiltonian kernel tests pass.
- [ ] Si256 stress results remain within current GPU tolerances.
- [ ] `stress_nl` drops from the current roughly 9.4 s post-Ewald baseline by at least 30% after the first implementation pass.
- [ ] The plan does not require new third-party dependencies.

## Follow-Up Options

- [ ] Implement analytic CUDA Ylm derivatives if finite-difference Ylm derivative generation remains a bottleneck.
- [ ] Add cuBLAS strided-batched derivative GEMM if the six derivative GEMMs dominate after setup is moved to GPU.
- [ ] Port the same design to ROCm after CUDA correctness and performance are established.
- [ ] Investigate GPU-aware or overlapped pool reductions if MPI synchronization becomes the next largest gap.

# CUDA Ewald Stress Acceleration

## Objective

Move the plane-wave Ewald stress contribution from host/OpenMP execution to a CUDA-first implementation while preserving the existing CPU path and avoiding unnecessary new GPU kernels. The target workload is the Si256 `dav_subspace` GPU run where `Stress::stress_ewa` costs about 6.7 seconds, making it the largest remaining CPU-only stress item after nonlocal stress.

The implementation should reuse existing ABACUS device-memory utilities, stress-kernel dispatch patterns, and the existing Ewald force CUDA packing/kernel style before introducing new CUDA code. The expected outcome is a CUDA Ewald stress path behind the existing stress interface, with only the final small stress tensor copied back to host before MPI pool reduction.

Relevant current code:

- `Stress_PW::cal_stress` calls Ewald stress through `this->stress_ewa(...)` at `source/source_pw/module_pwdft/stress_pw.cpp:83`.
- `Stress_Func::stress_ewa` is templated for CPU and GPU but currently executes host/OpenMP loops at `source/source_pw/module_pwdft/stress_ewa.cpp:15`.
- The reciprocal-space Ewald stress loop recomputes ionic phase sums over all atoms for each G vector at `source/source_pw/module_pwdft/stress_ewa.cpp:78`.
- The real-space Ewald stress loop calls `H_Ewald_pw::rgen` for every atom pair at `source/source_pw/module_pwdft/stress_ewa.cpp:123`.
- Final sign, symmetry fill, and pool reduction happen on the host at `source/source_pw/module_pwdft/stress_ewa.cpp:183`.
- Existing Ewald force CUDA code packs atom/G-vector data and launches `force_ew_kernel` at `source/source_pw/module_pwdft/kernels/cuda/force_op.cu:1045`.
- The Ewald force CUDA dispatch interface is declared at `source/source_pw/module_pwdft/kernels/force_op.h:421`.

## Implementation Plan

- [ ] 1. Establish an Ewald stress CPU-reference test before changing behavior.
  Add or extend a focused stress-kernel/unit test that calls the current CPU `stress_ewa` path for a small deterministic PW cell and records the 3x3 tensor as the reference. The test should cover at least one gamma-only PW case, one case where `rho_basis->ig_gge0 >= 0`, and one case where `ig_gge0 < 0` if the existing test fixtures make that practical. This anchors the CUDA port against the current math in `source/source_pw/module_pwdft/stress_ewa.cpp:31`, `source/source_pw/module_pwdft/stress_ewa.cpp:61`, and `source/source_pw/module_pwdft/stress_ewa.cpp:183`.

- [ ] 2. Add a small Ewald stress device-op interface in the stress kernel layer.
  Introduce a CUDA-only internal op, conceptually parallel to the existing Ewald force op in `source/source_pw/module_pwdft/kernels/force_op.h:421`, but place it with stress operations if that better matches existing ownership in `source/source_pw/module_pwdft/kernels/stress_op.h`. The interface should accept flattened atom positions, atom charges or atom-type charge data, flattened G vectors, G magnitudes, scalar Ewald parameters, `ig_gge0`, and a device output buffer for the lower-triangle stress plus the scalar diagonal term. Keep the CPU implementation unchanged and avoid exposing any new user input.

- [ ] 3. Reuse the existing Ewald force host packing pattern for the first CUDA integration.
  In the GPU specialization path for Ewald stress, pack `tau`, atom type or per-atom charge, `iat2it`, `gcar`, and `gg` using the same local style already used by `cal_force_ew` around `source/source_pw/module_pwdft/forces.cpp:563`. This first pass may allocate temporary device buffers inside `stress_ewa`, matching the force path, so correctness lands before broader workspace reuse. Do not add persistent workspace ownership until the CUDA math is validated.

- [ ] 4. Implement the reciprocal-space Ewald stress contribution as the first CUDA subpath.
  Port the loop at `source/source_pw/module_pwdft/stress_ewa.cpp:78` to CUDA. Prefer one compact custom reduction op over several small kernels: each thread or block should process G vectors, compute the same `rhostar`, `sewald`, six lower-triangle components, and `sdewald` contribution, then reduce to a small device buffer. This is the most GPU-regular section and should preserve the gamma-only factor from `source/source_pw/module_pwdft/stress_ewa.cpp:61`, the `G=0` skip from `source/source_pw/module_pwdft/stress_ewa.cpp:81`, the phase convention from `source/source_pw/module_pwdft/stress_ewa.cpp:93`, and the unit factors from `source/source_pw/module_pwdft/stress_ewa.cpp:100`.

- [ ] 5. Evaluate whether existing structure-factor data can replace per-G atom phase recomputation.
  Before optimizing further, compare the CUDA reciprocal implementation that directly matches CPU math with a design that reuses `Structure_Factor` data already available in `Stress_PW::cal_stress` through `p_sf`. Use this only if sign/conjugation and type-summed charge conventions match the current `rhostar` construction exactly. If not exact, keep the direct CUDA phase sum for phase 1 and defer structure-factor reuse to a later cleanup.

- [ ] 6. Implement the real-space Ewald stress contribution with minimal custom CUDA logic.
  Port the atom-pair/image-shell loop from `source/source_pw/module_pwdft/stress_ewa.cpp:123` only after the reciprocal path is correct. Avoid a GPU sort because the stress accumulation does not require sorted image vectors. Instead, reproduce the `rgen` inclusion rule from `source/source_hamilt/module_ewald/H_Ewald_pw.cpp:300` directly inside a CUDA kernel: loop over the bounded image box, filter by `rmax`, skip near-zero vectors, and accumulate the six lower-triangle terms. Launch this only when `ig_gge0 >= 0`, preserving the current rank ownership rule from `source/source_pw/module_pwdft/stress_ewa.cpp:123`.

- [ ] 7. Keep host finalization and MPI reduction unchanged in the first CUDA version.
  After the CUDA op writes the lower-triangle tensor and scalar diagonal term, copy back only the small result buffer. Reuse the existing host finalization in `source/source_pw/module_pwdft/stress_ewa.cpp:183`: add the diagonal scalar, apply the sign convention, call `Parallel_Reduce::reduce_pool`, and mirror the tensor. This keeps MPI behavior and symmetry of the output conservative while removing the large CPU loops.

- [ ] 8. Gate the CUDA path by device type and preserve CPU behavior exactly.
  Dispatch to the CUDA Ewald stress path only when `Device` is `base_device::DEVICE_GPU` and CUDA is enabled. CPU builds, ROCm builds, LCAO callers, and any unsupported edge cases should continue through the current implementation. ROCm parity should be deferred until the CUDA path is validated.

- [ ] 9. Add focused correctness tests for the CUDA Ewald stress op.
  Extend `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp` or the nearest existing stress test target with CUDA tests comparing the device result to the CPU reference for the same deterministic cell. Use tolerances that account for changed reduction order but are tight enough to catch sign, unit, `G=0`, and gamma-factor mistakes.

- [ ] 10. Run the large Si256 `dav_subspace` benchmark and profile the result.
  Rebuild `abacus_basic_gpu`, then run the Si256 Nsight workflow outside the sandbox with `OMP_NUM_THREADS=1`. Compare `Stress stress_ewa` against the baseline profile where Ewald stress was about 6.69 seconds and total stress was about 17.24 seconds. Confirm that no new long GPU-idle gap is introduced and that `stress_nl` remains the next major target.

- [ ] 11. Add a second-pass cleanup only after correctness and performance are demonstrated.
  If temporary allocation/copy overhead is visible, move packed atom/G-vector buffers to a reusable workspace owned by the stress or device-memory layer. If the reciprocal phase-sum kernel remains expensive, revisit structure-factor reuse. If real-space dominates, tune atom-pair/image parallelism and reduction strategy before adding more kernels.

## Verification Criteria

- CPU-only builds and CPU stress calculations continue to use the existing `stress_ewa` implementation with no behavior change.
- CUDA unit tests compare Ewald stress tensors against CPU references for deterministic small PW cells within agreed GPU stress tolerances.
- Si256 `dav_subspace` final energy, force, and stress remain within existing GPU tolerances versus the pre-port baseline.
- The fresh Si256 profile shows `Stress stress_ewa` reduced substantially from the current 6.69 second baseline, with an initial target of at least 50 percent reduction.
- Nsight Systems shows CUDA activity during the former Ewald stress idle interval and no newly introduced GPU-idle gap above 1 second attributable to Ewald stress setup.
- Only a small final stress buffer is copied from device to host for the CUDA Ewald path.

## Potential Risks and Mitigations

1. **Phase convention mismatch in reciprocal space**
   Mitigation: Start with a direct CUDA translation of the current `rhostar` formula instead of immediately reusing `Structure_Factor`; add tests that catch conjugation/sign mistakes.

2. **Incorrect handling of `G=0` ownership across MPI ranks**
   Mitigation: Preserve the existing `ig_gge0 >= 0` rules for the diagonal scalar and real-space contribution, and keep the existing host `Parallel_Reduce::reduce_pool` finalization.

3. **Real-space image-shell differences from replacing `rgen`**
   Mitigation: Reproduce the same image bounds, radius filter, and near-zero exclusion rule. Do not require sorted image vectors because the stress sum is order-independent except for floating-point roundoff.

4. **GPU reduction order changes stress by more than expected**
   Mitigation: Use double precision for the Ewald stress CUDA path initially, even in single-precision PW runs, matching the current `Stress_PW<double, DEVICE_GPU>` instantiation. Validate tolerances before considering narrower precision.

5. **Temporary device allocation overhead hides kernel speedup**
   Mitigation: Accept temporary buffers for the first correctness pass, then profile and move to reusable workspaces only if allocation/copy overhead is significant.

6. **Excessive custom kernels make the port hard to maintain**
   Mitigation: Limit phase 1 to one reciprocal reduction op and one real-space reduction op, using existing memory utilities and launch/reduction idioms from the force and stress kernel layers.

7. **ROCm behavior regresses**
   Mitigation: Keep ROCm on the current host path until the CUDA implementation is validated, then port the proven op shape separately.

## Alternative Approaches

1. **Reciprocal-only CUDA first**
   This is the lowest-risk CUDA increment because the G-space loop is regular and clearly GPU-portable. It may leave a large fraction of the 6.69 seconds in the CPU real-space loop, so it is useful as a staged milestone but may not meet the full performance target alone.

2. **Structure-factor reuse first**
   This could reduce both CPU and GPU work by avoiding per-G atom phase recomputation, and it may reuse existing GPU structure-factor infrastructure. The risk is sign/conjugation and type-charge convention mismatch, so it should follow the direct CUDA reference path unless inspection proves exact equivalence.

3. **Full Ewald stress CUDA in one pass**
   This has the best chance of eliminating the whole Ewald idle interval, but it touches both reciprocal and real-space math at once. It should only be chosen if the implementation budget allows careful unit tests before running large-cell profiles.

## Assumptions

- CUDA is the first target because the measured profile was collected with Nsight on a CUDA build.
- `Stress_PW<double, base_device::DEVICE_GPU>` remains the active stress type for the Si256 GPU run.
- The Si256 benchmark remains `dav_subspace`, `precision single`, `cal_force=1`, and `cal_stress=1`.
- The implementation should not change SCF math, input syntax, solver selection, or CPU stress behavior.

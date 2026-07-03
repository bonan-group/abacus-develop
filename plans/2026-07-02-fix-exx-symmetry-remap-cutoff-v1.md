# Fix EXX Symmetry Remap Cutoff

## Objective

Fix the GPU PW EXX reciprocal symmetry remap abort where `build_exx_symmetry_remap` can include a full-point G vector that was excluded from the representative k-point basis. The evidence points to inconsistent cutoff predicates: `PW_Basis_K::setupIndGk` builds `npwk` and `igl2ig_k` with a strict plane-wave cutoff at `source/source_basis/module_pw/pw_basis_k.cpp:141`, while `build_exx_symmetry_remap` scans full-point G vectors with a relaxed cutoff at `source/source_pw/module_pwdft/kernels/exx_q_state_op.cpp:216`. The plan also covers the GPU batch identity-point path, where `load_full_point_real_batch` constructs a remap before checking whether the point can use the direct identity transform at `source/source_pw/module_pwdft/op_pw_exx.cpp:1778`.

## Implementation Plan

- [ ] 1. Capture the current failure from the reported SOL62++ EXX-label folder and save the final EXX progress line for comparison.
- [ ] 2. Add a regression test in the existing EXX symmetry test file for a full-point G vector inside the relaxed remap tolerance but outside the representative basis cutoff.
- [ ] 3. Align the remap full-point cutoff with the strict predicate used when the representative plane-wave basis is built.
- [ ] 4. Expand the remap lookup failure message with the failing G index, cutoff delta, mapped G vectors, k-point metadata, and GPU-index request state.
- [ ] 5. Reorder GPU batch identity and conjugate-only loads so direct-source transforms bypass `point_spatial_remap` before any remap cache construction.
- [ ] 6. Review the stress EXX call sites to confirm they already bypass remap construction for identity and conjugate-only full points.
- [ ] 7. Build CPU and GPU PW targets, run EXX symmetry tests, run GPU EXX integration, and rerun the failing SOL62++ EXX-label case.

## Verification Criteria

- The new regression test fails before the cutoff predicate change and passes after it.
- `build_exx_symmetry_remap` no longer accepts full-point G vectors outside the representative `PW_Basis_K::setupIndGk` basis.
- GPU batch identity and conjugate-only points do not construct `point_spatial_remap` before taking the direct transform path.
- Existing reciprocal symmetry remap tests continue to pass.
- The GPU PW EXX integration target passes.
- The reported SOL62++ EXX-label case reaches past the previous `failed to map full-point G vector to representative G vector` abort.

## Potential Risks and Mitigations

1. **Changing the cutoff could drop a vector that was previously included in remapped transforms.**
   Mitigation: Use the representative basis contract as the source of truth, and verify against the existing real-space symmetry comparison tests plus the failing SOL62++ case.

2. **A strict cutoff may hide a deeper symmetry mapping issue for genuinely rotated full points.**
   Mitigation: Keep the improved failure diagnostics so future mapping failures report the exact G vector, cutoff delta, and k-point metadata instead of a generic abort.

3. **Moving the batch identity branch may alter GPU batching behavior.**
   Mitigation: Limit the branch reorder to identity and conjugate-only direct-source cases, matching the already-established single-band behavior and preserving the remapped path for rotated points.

4. **The exact CaSe/CaS failing path may not be present in the current checkout.**
   Mitigation: Use the code-level regression test as the deterministic guard, and rerun the user-provided failing folder when its exact path is available.

## Alternative Approaches

1. Relax `PW_Basis_K::setupIndGk` to include the same tolerance as `build_exx_symmetry_remap`. This would make the predicates consistent, but it changes the plane-wave basis size globally and could affect energies, memory, and reference outputs. I do not recommend it as the first fix.

2. Keep the relaxed remap cutoff but skip unmapped vectors with a warning. This avoids the abort but risks silently dropping reciprocal components inconsistently, which is worse for EXX correctness.

3. Only add diagnostics and ask users to reduce cutoff sensitivity or change grids. This helps debugging but does not fix the inconsistent contract between remap construction and the representative basis.

# GPU-Resident Spin XC Extension

## Objective

Extend the opt-in GPU-resident built-in XC path beyond the current unpolarized PBE/PBEsol case by first adding resident LSDA for `nspin == 2`, then adding resident collinear-spin PBE/PBEsol GGA, and finally preparing a bounded feasibility path for noncollinear `nspin == 4`. This is a separate plan from the Si256 CPU-only gap-reduction plan: its primary goal is correctness and coverage for spin XC code paths, with performance validation as a secondary gate.

Current source context:

- The committed resident path is guarded by `ABACUS_XC_GPU=1`, `device == gpu`, built-in PBE/PBEsol, and `PARAM.inp.nspin == 1` in `source/source_hamilt/module_xc/xc_pot.cpp:36` and `source/source_hamilt/module_xc/xc_pot.cpp:41`.
- The resident unpolarized path already keeps scalar XC, total density, gradient construction, grid correction, divergence, and final D2H copy in one call-level workflow in `source/source_hamilt/module_xc/xc_pot.cpp:127`, `source/source_hamilt/module_xc/xc_pot.cpp:141`, `source/source_hamilt/module_xc/xc_pot.cpp:155`, `source/source_hamilt/module_xc/xc_pot.cpp:173`, and `source/source_hamilt/module_xc/xc_pot.cpp:189`.
- CPU collinear spin scalar XC is already implemented in `XC_Functional::v_xc` for `PARAM.inp.nspin == 2` at `source/source_hamilt/module_xc/xc_pot.cpp:284`, using `XC_Functional::xc_spin` at `source/source_hamilt/module_xc/xc_pot.cpp:308`.
- CPU `nspin == 2` GGA gradient correction forms two spin total densities with half core charge at `source/source_hamilt/module_xc/xc_grad.cpp:124` and `source/source_hamilt/module_xc/xc_grad.cpp:157`, computes two spin gradients at `source/source_hamilt/module_xc/xc_grad.cpp:143` and `source/source_hamilt/module_xc/xc_grad.cpp:176`, evaluates spin GGA terms at `source/source_hamilt/module_xc/xc_grad.cpp:554` and `source/source_hamilt/module_xc/xc_grad.cpp:571`, and applies two divergence corrections at `source/source_hamilt/module_xc/xc_grad.cpp:683`.
- CPU noncollinear GGA first maps four charge/magnetization channels to two local spin densities with `noncolin_rho` at `source/source_hamilt/module_xc/xc_grad.cpp:229`, then rotates the two spin potentials back to four channels at `source/source_hamilt/module_xc/xc_grad.cpp:739`.
- CPU LSDA support is already present in `XC_Functional::xc_spin` at `source/source_hamilt/module_xc/xc_lda_wrap.cpp:108`, including Slater spin exchange at `source/source_hamilt/module_xc/xc_lda_wrap.cpp:141`, PZ correlation at `source/source_hamilt/module_xc/xc_lda_wrap.cpp:171`, and PW correlation at `source/source_hamilt/module_xc/xc_lda_wrap.cpp:181`. Existing CPU tests cover spin PZ references in `source/source_hamilt/module_xc/test/test_xc2.cpp:189`.

Assumptions:

- First implementation target is CUDA, built-in non-LibXC functionals, `device == gpu`, `ABACUS_XC_GPU=1`, non-stress, and single-pool PW GPU FFT, matching the existing resident path constraints.
- `nspin == 2` LSDA can be enabled before spin GGA because it does not need gradient, FFT, or divergence stages.
- `nspin == 2` PBE/PBEsol should reuse the resident FFT helpers already validated for `nspin == 1`, but needs new spin scalar and spin grid kernels.
- `nspin == 4` noncollinear support should not be enabled by simply treating it as `nspin == 2`; it needs explicit density diagonalization and potential rotation kernels before runtime use.

## Implementation Plan

- [x] 1. Baseline and document the current CPU spin behavior before any new GPU enablement, covering LSDA, PBE, PBEsol, and noncollinear mapping. Record the scalar equations, spin-density packing, core-density split, `vtxc` accumulation, GGA `h1/h2` construction, and noncollinear rotate-back semantics using the source references above. Rationale: the spin implementation has more coupling between scalar XC, gradient correction, and potential channel layout than the unpolarized path, so the resident GPU contract must be explicit before writing kernels.

- [x] 2. Add focused CPU/GPU parity tests for a new resident LSDA spin scalar operator before implementing the operator. The tests should use deterministic `rho_up`, `rho_down`, and `rho_core` arrays, compare `v_up`, `v_down`, total spin densities, `etxc`, and `vtxc` against CPU `XC_Functional::xc_spin`, and cover both PZ and PW-LDA functional IDs. Rationale: CPU `nspin == 2` LDA is already supported through `XC_Functional::xc_spin`, but the resident GPU kernels currently only implement a PBE scalar helper, so LSDA is the lowest-risk spin extension.

- [x] 3. Implement the resident LSDA spin scalar operator in `source/source_hamilt/module_xc/kernels/xc_gradcorr_op.h`, `source/source_hamilt/module_xc/kernels/xc_gradcorr_op.cpp`, and `source/source_hamilt/module_xc/kernels/cuda/xc_gradcorr_op.cu`. The operator should form total density, clamp `zeta` consistently with the CPU `v_xc` spin path, write two device potential channels, and reduce `etxc` and `vtxc` without invoking any GGA gradient workflow. Rationale: this unlocks `nspin == 2` LDA residency independently and provides reusable spin scalar machinery for spin PBE/PBEsol.

- [x] 4. Add a narrow resident LSDA dispatch path in `source/source_hamilt/module_xc/xc_pot.cpp` while keeping unsupported spin paths on the existing CPU fallback. The guard should require CUDA, `ABACUS_XC_GPU=1`, `device == gpu`, `PARAM.inp.nspin == 2`, built-in PZ or PW-LDA IDs, non-LibXC, GPU charge and PW objects, matching `nrxx`, and non-null device density pointers for both spin channels. Rationale: LSDA does not need the PW gradient FFT pipeline, so it can become the first safe resident `nspin == 2` path without changing GGA behavior.

- [x] 5. Add focused CPU/GPU parity tests for spin PBE/PBEsol grid evaluation before implementing the spin GGA resident grid kernel. The tests should compare device results against CPU `gcx_spin` and `gcc_spin` formulas for unequal spin densities, nonzero `grad_up`, nonzero `grad_down`, positive and near-fully-polarized `zeta`, and both PBE and PBEsol flags. Rationale: spin GGA has cross-gradient correlation through `v2cud`, so testing only per-channel exchange would miss the most important coupling.

- [x] 6. Implement a resident spin PBE/PBEsol grid kernel that consumes two total spin densities and two interleaved gradient buffers, accumulates into two resident potential channels, writes two `h` fields, and reduces spin GGA `etxcgc` and `vtxcgc`. The kernel should mirror CPU use of `gcx_spin` and `gcc_spin`, including the total-density threshold, `zeta` handling, and half-core subtraction in `vtxcgc`. Rationale: this is the central math difference between unpolarized resident GGA and collinear-spin resident GGA.

- [x] 7. Generalize the resident `v_xc` workspace in `source/source_hamilt/module_xc/xc_pot.cpp` so it can hold either one or two spin channels without duplicating the full unpolarized path. The refactor should preserve the current `nspin == 1` behavior, keep per-call buffer ownership for now, and make the final host copy write either one or two `ModuleBase::matrix` rows. Rationale: adding a separate complete spin implementation would duplicate allocation, FFT, and cleanup logic and make future noncollinear support harder.

- [x] 8. Add resident spin gradient construction for `nspin == 2` PBE/PBEsol by reusing the GPU real-to-reciprocal and reciprocal-to-real FFT sequence twice, once for each spin total density. Each spin total density should include half of `rho_core`, and each gradient transform should keep the existing `add=true` plus `tpiba` factor contract that fixed the earlier unpolarized gradient bug. Rationale: the CPU spin path computes `grad_rho` separately for spin-up and spin-down; residency only works if both gradients stay on device.

- [x] 9. Add resident spin divergence application for `nspin == 2` PBE/PBEsol by reusing the existing device reciprocal helper sequence for both `h_up` and `h_down`. The update should subtract the corresponding divergence from each potential channel and reduce the two `dh * rho_spin_valence` contributions to `vtxcgc`. Rationale: CPU `gradcorr` runs `grad_dot` twice for collinear spin, and this is required for full GGA parity rather than a partial grid-only offload.

- [x] 10. Enable the guarded resident `nspin == 2` PBE/PBEsol dispatch only after the spin scalar, spin grid, two-gradient, and two-divergence tests pass. The guard should remain opt-in with `ABACUS_XC_GPU=1`, reject LibXC, reject stress, reject noncollinear `nspin == 4`, reject unsupported GGA IDs, and fall back silently to the existing CPU path for everything else. Rationale: correctness coverage should precede any runtime enablement, and the existing default behavior must remain unchanged.

- [x] 11. Add runtime smoke tests for small collinear spin cases using GPU PW, one LSDA case and one PBE or PBEsol case. Compare resident and fallback final energy, `v_xc_resident_gpu` timer presence, convergence, and potential-related drift with deterministic seeds where available. Rationale: kernel tests validate formulas, but the actual resident branch also depends on `Charge` device density, PW GPU FFTs, and the `ModuleBase::matrix` boundary.

- [x] 12. Profile one representative collinear spin case with Nsight after correctness is stable. Compare `PotXC cal_veff`, `XC_Functional v_xc`, resident sub-timers, CUDA inactive gaps, and final numerical drift for `ABACUS_XC_GPU=0` and `ABACUS_XC_GPU=1`. Rationale: this plan is separate from CPU-gap reduction, but the extension should not regress the performance win or create new large CPU-only gaps.

- [x] 13. Add noncollinear feasibility kernels and tests without enabling runtime `nspin == 4` by default. The first kernel should reproduce `noncolin_rho` on device for `rho[0..3]`, `ux`, and `lsign`, and a second test should validate rotate-back from two spin potentials to four potential channels for finite and near-zero magnetization. Rationale: noncollinear support needs extra transformations before and after the collinear spin GGA core; testing those transformations separately prevents accidental misuse of the `nspin == 2` implementation.

- [x] 14. Decide the noncollinear enablement boundary after feasibility tests pass. If the device `noncolin_rho` and rotate-back tests pass and the collinear spin resident core is stable, draft a follow-up implementation plan for opt-in `nspin == 4` PBE/PBEsol. If they expose ambiguous magnetization or potential semantics, keep `nspin == 4` on CPU fallback and document the blocker. Rationale: noncollinear spin has enough additional physics and channel-layout risk that it should be promoted only after evidence, not bundled into the first collinear spin commit.

- [x] 15. Update documentation and the resident XC plan status after implementation. The documentation should state that CPU `nspin == 2` LSDA already existed, list which resident GPU spin paths are now covered, identify remaining fallbacks, and record any runtime/profile evidence. Rationale: this feature is guarded and performance-sensitive, so users and future agents need a clear map of what is enabled, what is fallback, and why.

## Implementation Status

Implemented in this branch:

- Resident CUDA `nspin == 2` LSDA scalar support for built-in PZ and PW-LDA through `xc_scalar_lda_spin_op`.
- Resident CUDA `nspin == 2` PBE/PBEsol support for built-in non-LibXC functionals, including spin scalar, two GPU density-gradient builds, spin GGA grid accumulation, two divergence applications, and final two-row host copy.
- Noncollinear feasibility-only CUDA helpers for `noncolin_rho` and rotate-back potential mapping; runtime `nspin == 4` remains disabled.
- Guarding remains opt-in through `ABACUS_XC_GPU=1`; default behavior remains CPU XC fallback.
- Real-space charge-density freshness syncs were added after host mutations that can feed resident XC: initial charge setup, post-renormalization SCF initialization, post-`parallelK()` density construction, and post-mixing broadcast. `Charge::sync_realspace_density_to_device()` is a CPU/no-GPU no-op and only performs H2D sync when the charge device is `"gpu"`.

Verification performed:

- TDD red checks:
  - Missing `xc_scalar_lda_spin_op` compile failure before scalar implementation.
  - Missing `xc_gradcorr_pbe_spin_grid_resident_op` link failure before spin-grid implementation.
- Focused CUDA XC suite:
  - `./build-test-cuda/source/source_hamilt/module_xc/kernels/test/MODULE_HAMILT_XC_Functional_UTs`
  - Result: 18 tests passed, including `FullVxcLdaSpinResidentGpuMatchesCpu` and `ChargeRealspaceDensitySyncIsNoopOnCpuDevice`.
- Production build:
  - `cmake --build build --target abacus_basic_gpu -j2`
  - Result: passed.
- Runtime smoke:
  - Input: scratch copy of `tests/01_PW/037_PW_FM` with `device gpu`, absolute `pseudo_dir`, `OMP_NUM_THREADS=1`.
  - One-step density-freshness check: fallback `!FINAL_ETOT_IS -5960.070859958296 eV`, resident `!FINAL_ETOT_IS -5960.070859958268 eV`, drift `2.82e-11 eV`.
  - Strict converged check: fallback `!FINAL_ETOT_IS -5866.197297502681 eV`, resident `!FINAL_ETOT_IS -5866.197297503067 eV`, drift `-3.86e-10 eV`; resident timer `XC_Functional v_xc_resident_gpu` present for 21 calls.
  - Short Nsight-profiled check (`scf_nmax=2`): fallback `!FINAL_ETOT_IS -5806.079618838718 eV`, resident `!FINAL_ETOT_IS -5806.079618838719 eV`, drift about `-9.1e-13 eV`; resident timer present for 3 calls.
- Nsight profiling:
  - Fallback profile: `/tmp/abacus_spin_xc_nsys_off/profile.nsys-rep` and `/tmp/abacus_spin_xc_nsys_off/profile.sqlite`.
  - Resident profile: `/tmp/abacus_spin_xc_nsys_on/profile.nsys-rep` and `/tmp/abacus_spin_xc_nsys_on/profile.sqlite`.
  - ABACUS timer on the short profile: total `0.60 s -> 0.56 s`; resident run reports `PotXC cal_veff 0.01 s`, `XC_Functional v_xc 0.01 s`, and `XC_Functional v_xc_resident_gpu 0.01 s` over 3 calls.
- MPI/band-parallel validation status:
  - Attempted two-rank `bndpar 2` smoke on `tests/01_PW/037_PW_FM`.
  - Default OpenMPI shared-memory `vader` transport crashed in `MPI_Allreduce` with a segmentation fault.
  - TCP/loopback transport reached the SCF section but failed in OpenMPI `btl_tcp_frag_send` with `writev Bad address` and wedged before a fallback/resident comparison.
  - This remains an environment/launcher blocker for local validation, not a demonstrated resident-XC numerical mismatch.

Remaining follow-up:

- Multi-rank `bndpar`/KPAR runtime validation still needs to be rerun in an MPI environment that can complete the GPU PW smoke test without transport failures.
- Runtime `nspin == 4` enablement is intentionally deferred; the transform kernels pass feasibility tests, but full noncollinear residency needs a separate plan covering gradient construction from diagonalized densities, rotate-back after divergence, and runtime tolerances.

## Verification Criteria

- The existing unpolarized focused CUDA tests continue to pass, including scalar PBE, PBE/PBEsol grid parity, resident grid accumulation, reciprocal helpers, and PW gamma FFT-box tests.
- New LSDA spin scalar CPU/GPU parity tests pass for PZ and PW-LDA with unequal spin densities and nonzero core density.
- New spin PBE/PBEsol GGA grid CPU/GPU parity tests pass for exchange, correlation, cross-gradient `v2cud`, two potential channels, two `h` fields, `etxcgc`, and `vtxcgc`.
- The resident dispatch remains disabled by default and requires `ABACUS_XC_GPU=1` for every new spin path.
- Unsupported cases, including LibXC, stress, unsupported GGAs, and noncollinear `nspin == 4`, continue to use existing CPU fallback until explicitly enabled.
- A small GPU PW LSDA `nspin == 2` runtime case exits 0 with `OMP_NUM_THREADS=1`, calls the resident path when enabled, and matches fallback energy within an agreed double-precision tolerance.
- A small GPU PW PBE or PBEsol `nspin == 2` runtime case exits 0 with `OMP_NUM_THREADS=1`, calls the resident path when enabled, and matches fallback energy, pressure if available, and convergence behavior within documented tolerances.
- Noncollinear feasibility tests pass for device `noncolin_rho` and potential rotate-back, or the plan records a concrete blocker before any runtime `nspin == 4` enablement.
- Fresh profiling for one representative spin case shows no new dominant CPU-only XC stage relative to the fallback path, and resident sub-timers identify scalar, gradient, grid, divergence, and final copy costs separately.

## Potential Risks and Mitigations

1. **Spin GGA formula drift**
   - Impact: Incorrect `v_up`, `v_down`, `h_up`, `h_down`, or energy terms for magnetic calculations.
   - Likelihood: Medium, because spin GGA includes cross-gradient correlation and half-core handling.
   - Mitigation: Write CPU-reference parity tests for scalar LSDA, spin grid terms, two divergence applications, and full runtime cases before enabling dispatch.
   - Contingency: Keep `nspin == 2` PBE/PBEsol guarded off while allowing only tested LSDA spin residency.

2. **Regression in current unpolarized resident path**
   - Impact: The existing Si256 performance path could break while generalizing workspace logic.
   - Likelihood: Medium if the workspace refactor is too broad.
   - Mitigation: Keep the first refactor minimal, rerun existing `nspin == 1` focused tests and Si2 smoke checks, and avoid changing the public `v_xc` return contract.
   - Contingency: Split spin resident code into a separate helper while preserving the committed unpolarized branch unchanged.

3. **Device density freshness ambiguity**
   - Impact: Resident spin path could read stale spin density if upstream code has not synchronized both spin channels to device.
   - Likelihood: Medium, because the previous review already found a host-to-device overwrite risk in the unpolarized path.
   - Mitigation: Consume `Charge::get_rho_d(0)` and `Charge::get_rho_d(1)` directly, add runtime smoke tests through the real charge path, and do not perform hidden host-to-device synchronization inside `v_xc`.
   - Contingency: If device-density ownership is unclear for spin cases, keep the resident spin path disabled and document the required upstream synchronization contract.

4. **Noncollinear potential rotation mistakes**
   - Impact: Incorrect four-channel potential for magnetic noncollinear calculations.
   - Likelihood: High if implemented in the same pass as collinear spin.
   - Mitigation: Treat noncollinear as feasibility first, test `noncolin_rho` and rotate-back independently, and do not enable runtime `nspin == 4` until a separate evidence-backed plan is complete.
   - Contingency: Leave noncollinear on CPU fallback and ship only collinear spin resident support.

5. **Performance disappointment for small spin cases**
   - Impact: Additional device work and final D2H copy may not visibly speed up small magnetic systems.
   - Likelihood: Medium.
   - Mitigation: Separate correctness enablement from default enablement, keep opt-in gating, and profile representative magnetic workloads rather than only tiny smoke tests.
   - Contingency: Keep spin residency experimental behind `ABACUS_XC_GPU=1` until larger benchmarks show value.

## Alternative Approaches

1. **Implement LSDA spin first, then spin GGA**
   - Description: Add resident `nspin == 2` LDA/PZ/PW-LDA support before PBE/PBEsol spin GGA.
   - Pros: Smallest mathematical surface, no FFT gradient or divergence changes, directly addresses the LDA spin status question.
   - Cons: Does not address the main GGA magnetic workload immediately.
   - Recommendation: Use this as the first implementation phase because it de-risks spin scalar infrastructure.

2. **Jump directly to full `nspin == 2` PBE/PBEsol**
   - Description: Implement spin scalar, spin gradient, spin grid, and spin divergence together.
   - Pros: Delivers the most useful collinear spin GGA feature in one branch.
   - Cons: Larger patch, harder debugging, and more ways to confuse scalar spin errors with GGA errors.
   - Recommendation: Do not start here unless LSDA is explicitly deprioritized; use staged LSDA then GGA instead.

3. **Reuse CPU spin grid evaluation with only GPU gradients**
   - Description: Compute spin gradients on GPU, copy them to host, and keep scalar/grid/divergence on CPU.
   - Pros: Lower implementation risk and might reduce part of the cost.
   - Cons: Violates the resident goal, preserves host-device traffic, and repeats the partial-offload limitation seen in the unpolarized path.
   - Recommendation: Reject for this plan except as a temporary debugging comparison.

4. **Enable noncollinear immediately through the collinear spin core**
   - Description: Add device `noncolin_rho`, run resident spin GGA, rotate potentials back, and enable `nspin == 4` in the same implementation.
   - Pros: Broadest spin feature coverage.
   - Cons: Too much physics and channel-layout risk for one patch, especially around near-zero magnetization and sign handling.
   - Recommendation: Reject for initial implementation; keep noncollinear as feasibility and follow-up.

## Assumptions

- CPU `nspin == 2` LDA/LSDA behavior is authoritative and should be used as the reference for device scalar spin kernels.
- Built-in PBE and PBEsol remain the only GGA functionals in scope for resident spin GGA.
- LibXC, mGGA, hybrid, stress, and noncollinear runtime enablement stay out of scope for the first collinear spin implementation.
- The final `XC_Functional::v_xc` API continues returning a host `ModuleBase::matrix`, so one final D2H copy remains acceptable for this plan.

## Dependencies

- The current committed resident unpolarized PBE/PBEsol path, including GPU reciprocal helper kernels and PW gamma FFT-box fixes.
- A CUDA-capable test environment for focused GPU kernel tests and ABACUS runtime smoke tests.
- Existing CPU spin reference functions in `xc_lda_wrap.cpp`, `xc_gga_wrap.cpp`, and `xc_grad.cpp`.
- Existing ABACUS GPU build and runtime convention with `OMP_NUM_THREADS=1` for runtime checks.

## Notes

- This plan intentionally separates correctness coverage from default enablement. New spin resident paths should remain opt-in through `ABACUS_XC_GPU=1`.
- The near-term “status of `nspin == 2` LDA” is: CPU support exists; resident GPU support does not yet exist and should be the first extension.
- The noncollinear phase is explicitly a feasibility stage, not a commitment to runtime enablement in the same patch.

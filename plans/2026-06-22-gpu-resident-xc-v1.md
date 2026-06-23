# GPU-Resident Built-In XC Path

## Objective

Make the common built-in PBE/PBEsol exchange-correlation path for GPU PW calculations resident on the GPU across scalar XC evaluation, GGA gradient construction, grid-point gradient correction, divergence application, and final potential contribution. The target is to remove avoidable host-device round trips in `XC_Functional::v_xc` and `XC_Functional::gradcorr`, not merely accelerate one grid loop.

Current evidence and source context:

- `PotXC::cal_v_eff` now passes explicit device intent from the density basis into XC, but the return value is still a host `ModuleBase::matrix` that is added to host-side `v_eff` at `source/source_estate/module_pot/pot_xc.cpp:38`.
- Built-in scalar XC still loops on host density arrays and fills a host matrix before calling `gradcorr` at `source/source_hamilt/module_xc/xc_pot.cpp:59` and `source/source_hamilt/module_xc/xc_pot.cpp:186`.
- The current GPU `gradcorr` path copies `rhotmp1` and flattened `gdr1` to device, launches only the grid evaluator, then copies `v` and `h` back to host at `source/source_hamilt/module_xc/xc_grad.cpp:267`.
- `grad_rho` and `grad_dot` remain CPU workflows around host FFT calls at `source/source_hamilt/module_xc/xc_grad.cpp:857` and `source/source_hamilt/module_xc/xc_grad.cpp:895`.
- The charge object already exposes device density and reciprocal-density pointers through `get_rho_d` and `get_rhog_d` at `source/source_estate/module_charge/charge.h:184`.
- `PW_Basis` already has GPU real-to-reciprocal and reciprocal-to-real FFT helpers at `source/source_basis/module_pw/pw_transform_gpu.cpp:8` and `source/source_basis/module_pw/pw_transform_gpu.cpp:57`.

Assumptions:

- First production target remains CUDA, `nspin == 1`, non-stress, built-in PBE/PBEsol, and single-pool GPU PW, matching the current benchmark and `PW_Basis::real2recip_gpu` assertion at `source/source_basis/module_pw/pw_transform_gpu.cpp:11`.
- LibXC, mGGA, spin-polarized, noncollinear, and stress paths stay on CPU until the resident built-in PBE path is proven correct.
- The output may still be copied back once at the boundary if the broader potential pipeline requires host `ModuleBase::matrix`; the plan’s goal is to remove repeated internal XC transfers first, then move the potential boundary.

## Implementation Plan

- [x] Establish a precise transfer and residency baseline for the current partial GPU XC path.
  Record timers around H2D copies of `rhotmp1`, `rho_core`, and `gdr1`, D2H copies of `v` and `h`, CPU `grad_rho`, CPU `grad_dot`, and the final `v_eff += v` boundary. This separates true kernel time from transfer and host formatting time. Affected files: `source/source_hamilt/module_xc/xc_grad.cpp`, `source/source_hamilt/module_xc/xc_pot.cpp`, and the existing Nsight wrapper. Rationale: the current measured reduction is modest because only the middle grid evaluation moved; we need measurements that expose the full cost of non-residency.

- [x] Define a GPU-resident XC workspace for the guarded built-in path.
  Introduce a narrow internal workspace that owns or borrows device buffers for total density, core density, reciprocal total density, gradient components, `h` components, divergence scratch, potential contribution, and reduction scalars. Keep allocation lifetime at the `v_xc` call level initially, then consider reuse across SCF iterations after correctness is stable. Affected files: `source/source_hamilt/module_xc/xc_functional.h`, `source/source_hamilt/module_xc/xc_pot.cpp`, and new files under `source/source_hamilt/module_xc/kernels/`. Rationale: ad hoc allocations inside `gradcorr_eval_grid` force transfers and make it hard to keep the whole pipeline resident.

- [x] Build total density and scalar LDA contribution directly on device.
  Add a CUDA operator that reads `Charge::get_rho_d(0)` and device core density, forms the total density, evaluates the scalar built-in XC contribution, initializes the device potential, and reduces scalar `etxc` and `vtxc`. Use the existing CPU scalar loop at `source/source_hamilt/module_xc/xc_pot.cpp:59` as the reference. Rationale: scalar XC is not the largest cost, but leaving it on CPU forces an early host matrix and breaks residency before `gradcorr`.

- [x] Add GPU `grad_rho` for the resident path using existing PW GPU FFT helpers.
  Compute reciprocal total density on device, multiply by each Cartesian `iG` on device, call `PW_Basis::recip2real_gpu`, and store three real-space gradient components in device buffers. Reuse `PW_Basis::gcar` data through an explicit device copy or a persistent device mirror; note that `PW_Basis` already tracks `npw`, `nrxx`, and `gg_d`-style device metadata at `source/source_basis/module_pw/pw_basis.h:115` and `source/source_basis/module_pw/pw_basis.h:120`. Rationale: CPU `grad_rho` was around `3.08 s` in the latest validation and currently forces `gdr` back to host before the GPU grid kernel can run.

- [x] Replace the current copied-input grid evaluator with an in-workspace device operator.
  Refactor `xc_gradcorr_pbe_grid_op` so it consumes resident total-density and gradient buffers and accumulates directly into the resident potential and `h` buffers. Keep the existing isolated CPU and CUDA parity tests, but add a residency-mode test that proves no host intermediate is required between `grad_rho_gpu` and grid evaluation. Affected files: `source/source_hamilt/module_xc/kernels/xc_gradcorr_op.h`, `source/source_hamilt/module_xc/kernels/cuda/xc_gradcorr_op.cu`, and `source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp`. Rationale: the current operator is correct but staged around host-owned `gdr1` and host-owned `v`.

- [x] Add GPU `grad_dot` for the resident path using existing PW GPU FFT helpers.
  For each `h` component, transform device real-space `h` to reciprocal space, multiply by `iG`, accumulate the reciprocal divergence on device, transform back to real space, subtract the divergence from the resident potential, and reduce the `dh * rho` contribution to `vtxcgc`. Use the CPU algorithm at `source/source_hamilt/module_xc/xc_grad.cpp:895` as the reference. Rationale: CPU `grad_dot` was around `3.47 s` in the latest validation and is the remaining large piece after grid evaluation moved.

- [x] Return or apply the resident XC potential without an internal round trip.
  Add a device-aware XC result path so `PotXC` can add the XC potential to the device-side effective potential when the caller is GPU-resident. If the current `Potential` interface still requires host `ModuleBase::matrix`, perform one final D2H copy at the boundary and make that copy visible with a timer. Affected files: `source/source_estate/module_pot/pot_xc.cpp`, `source/source_estate/module_pot/potential_new.cpp`, and any existing `Veff<PW>` device porter integration. Rationale: fully resident XC only matters if its result feeds the next GPU consumer without immediately returning to host.

- [x] Keep the guarded fallback policy strict and observable.
  Preserve the current CPU fallback for unsupported cases and add a runtime timer or one-line debug marker that reports whether the resident path, partial path, or CPU path was selected. The guard should require CUDA, explicit GPU device intent, `ABACUS_XC_GPU=1`, built-in PBE/PBEsol, `nspin == 1`, non-stress, non-LibXC, and compatible PW GPU FFT conditions. Rationale: silent fallback makes performance results ambiguous and risks accidentally routing unsupported physics through the resident path.

- [x] Build layered correctness tests from kernels to full `v_xc`.
  Add deterministic tests for scalar XC, GPU `grad_rho`, resident grid evaluation, GPU `grad_dot`, and full resident `v_xc` against the CPU path. Use small synthetic grids for isolated kernels and at least one real PW basis test for FFT ordering. Rationale: the previous formula-helper bug showed that GPU-vs-GPU-mirror tests are insufficient; every stage needs a canonical CPU reference. Progress: scalar/grid/divergence helper tests pass, a gamma-only FFT-box mirror test was added, debug subdivision demonstrated real PW-basis `grad_rho` plus resident grid parity at roundoff, and a compact full `XC_Functional::v_xc` fixture now covers the resident dispatch boundary with real `Charge` device density.

- [x] Validate Si256 numerics and performance with toggles.
  Run Si256 with `ABACUS_XC_GPU=0`, current partial GPU path, and full resident GPU path. Compare final energy, pressure, max force component drift, SCF convergence, `PotXC cal_veff`, `XC_Functional v_xc`, `gradcorr_grad_rho`, `gradcorr_eval_grid`, `gradcorr_grad_dot`, and Nsight inactive time. Rationale: the goal is not a fast isolated kernel; it is reducing the repeated XC inactive gaps in the benchmark.

- [ ] Decide whether to extend to stress and spin after the resident `nspin == 1` path is stable.
  If the resident path meets correctness and performance criteria, plan separate follow-ups for stress GGA, `nspin == 2`, PZ/LDA scalar-only resident mode, and eventually LibXC/mGGA strategy. Rationale: mixing these cases into the first resident PBE path would blur risk and slow down validation.

## Verification Criteria

- Focused CUDA tests pass for scalar XC, PBE/PBEsol grid evaluation, resident grid accumulation, reciprocal helpers, and `grad_dot` helpers against CPU references.
- Runtime parity passes for the guarded resident `v_xc` branch on seeded Si2; a compact automated full-`v_xc` unit/integration fixture remains follow-up.
- `cmake --build build-test-cuda --target MODULE_HAMILT_XC_Functional_UTs -j2` passes.
- `cmake --build build --target abacus_basic_gpu -j2` passes.
- Si256 exits status 0 for CPU fallback and resident GPU path with `OMP_NUM_THREADS=1`.
- Si256 final total-energy drift is at or below the tolerance established by the current partial GPU path, and max force component drift is no worse than the current `3.00e-4 eV/Angstrom` reference unless a tighter or looser precision-specific threshold is explicitly justified.
- `PotXC cal_veff` drops materially below the current partial GPU value of about `11.46 s`.
- `gradcorr_grad_rho` and `gradcorr_grad_dot` no longer appear as dominant CPU-only stages in the timer table.
- Nsight inactive time improves materially from the current post-partial-XC value of `18.17 s`, with the repeated XC-region gaps either removed or clearly attributed to remaining non-XC work.
- The plan implementation leaves unsupported functionals and spin/stress modes on the existing CPU path.

## Potential Risks and Mitigations

1. **GPU FFT helper limitations**
   Mitigation: start with the existing single-pool condition used by `PW_Basis::real2recip_gpu`, guard fallback when conditions are not met, and add a real PW-basis test before enabling runtime use.

2. **Numerical drift from reduction and FFT ordering**
   Mitigation: compare each stage independently against CPU references, keep reductions explicit and measurable, and set separate tolerances for isolated double-precision kernels and full single-precision Si256 runtime.

3. **Device memory pressure**
   Mitigation: allocate a workspace once per `v_xc` call, reuse buffers between stages, measure peak GPU memory in `device.log`, and avoid retaining SCF-lifetime buffers until per-call memory is understood.

4. **Interface churn across `PotXC`, `XC_Functional`, and `Potential`**
   Mitigation: add a narrow resident path in parallel with the existing host-returning API first, then move the potential boundary only after the internal XC pipeline is correct.

5. **Physics coverage gaps**
   Mitigation: keep guards strict and visible; treat spin, stress, LibXC, mGGA, and hybrid paths as follow-up work rather than weakening the first resident path.

## Alternative Approaches

1. Continue incremental partial offload: move only `grad_rho` or only `grad_dot` next. This is lower risk, but it preserves intermediate host-owned data structures and may leave most of the residency problem intact.

2. Build a separate fused PBE-only XC driver that bypasses `XC_Functional::gradcorr` entirely. This is likely faster and cleaner for Si/PBE, but it duplicates more logic and requires a stronger validation suite before it is acceptable.

3. Try to route through LibXC-compatible GPU abstractions. This has broader long-term functional coverage, but the current bottleneck is built-in PBE `gradcorr`, and the codebase does not currently have a GPU LibXC backend.

4. Use OpenMP/CPU optimization for the remaining `grad_rho` and `grad_dot`. This may reduce wall time on CPU-heavy machines, but it does not solve GPU idleness or the host-device residency issue you called out.

## Implementation Status: 2026-06-23

Implemented:

- Added CUDA/CPU scalar PBE, resident grid-accumulation, divergence-application, and reciprocal helper ops in `source/source_hamilt/module_xc/kernels/xc_gradcorr_op.*`.
- Added focused CUDA tests for scalar PBE, resident grid accumulation, `dh` application, and reciprocal helper formulas in `source/source_hamilt/module_xc/kernels/test/xc_functional_op_test.cpp`.
- Added a guarded `XC_Functional::v_xc` resident path for CUDA, `ABACUS_XC_GPU=1`, `device == gpu`, built-in PBE/PBEsol, `nspin == 1`, non-LibXC, and single-pool PW GPU FFTs.
- The resident path keeps scalar XC, total density, reciprocal density, gradients, grid evaluation, `h`, divergence, and potential on device, then performs one final D2H copy because `v_xc` still returns a host `ModuleBase::matrix`.

Verification completed:

- `cmake --build build-test-cuda --target MODULE_HAMILT_XC_Functional_UTs -j2` passes.
- Focused CUDA test filter `XCResidentOpTest.*:XCGradcorrOpTest.PbeGridGpuMatchesCpu:XCGradcorrOpTest.PbeGridCpuMatchesBuiltinReferenceValues:XCFunctionGpuPolicyTest.GuardsSupportedBuiltins` passes 7 tests.
- `cmake --build build-test-cuda --target abacus_basic_gpu -j2` passes.
- Si2 PBE GPU smoke run with `OMP_NUM_THREADS=1 ABACUS_XC_GPU=1` exits 0 and the timer table confirms `XC_Functional v_xc_resident_gpu` is called.

Debug finding and fix:

- The seeded Si2 resident path originally failed parity because GPU `grad_rho` and `grad_dot` called `PW_Basis::recip_to_real(..., add=false, factor=ucell->tpiba)`. The GPU output helper only applies `factor` when `add=true`, matching the existing helper contract, so the resident gradients lost the `2*pi/a0` scale. The debug prints showed `grad_rho` was smaller than CPU by approximately `2*pi`, while scalar XC and reciprocal total density already matched.
- The resident path now zeroes the output buffers and calls `recip_to_real(..., add=true, factor=ucell->tpiba)` for the scaled gradient/divergence transforms.
- The debug reference now fresh-FFTs valence density instead of comparing against possibly stale `chr->rhog[0]` during initialization.
- A gamma-only GPU inverse FFT box-fill helper was also added so `PW_Basis::recip2real_gpu` reconstructs Hermitian conjugate partners before the full complex inverse. This hardens the GPU path for gamma-only reciprocal data and is covered by focused kernel tests for `xprime`, `yprime`, and reduced-axis boundary planes.
- Post-review cleanup removed the resident path's internal `sync_rho_to_device()` call. The resident branch now consumes `Charge::get_rho_d(0)` directly instead of overwriting device density from the host mirror at entry.

Verification completed after the fix:

- `cmake --build build-test-cuda --target MODULE_PW_PW_Kernels_UTs -j2` passes for the focused gamma-box and baseline box-fill tests.
- `cmake --build build-test-cuda --target MODULE_HAMILT_XC_Functional_UTs -j2` passes.
- Focused CUDA PW filter `TestModulePWPWMultiDevice.set_3d_fft_box_gamma_op_gpu_fills_conjugate_partners:TestModulePWPWMultiDevice.set_3d_fft_box_gamma_op_gpu_supports_yprime_half_spectrum:TestModulePWPWMultiDevice.set_3d_fft_box_gamma_op_gpu_avoids_boundary_pair_races:TestModulePWPWMultiDevice.set_3d_fft_box_op_gpu` passes 4 tests.
- Focused CUDA XC filter `XCResidentOpTest.*:XCGradcorrOpTest.PbeGridGpuMatchesCpu:XCGradcorrOpTest.PbesolGridGpuMatchesCpu:XCGradcorrOpTest.PbeGridCpuMatchesBuiltinReferenceValues:XCFunctionGpuPolicyTest.GuardsSupportedBuiltins` passes 8 tests.
- Seeded Si2 PBE clean runtime parity is restored:
  - `ABACUS_XC_GPU=1`: `!FINAL_ETOT_IS -171.2119808601766 eV`
  - `ABACUS_XC_GPU=0`: `!FINAL_ETOT_IS -171.2119808601695 eV`
- With temporary debug subdivision, scalar density/potential, reciprocal total density, `grad_rho`, resident grid potential, `h`, grid `etxc`, and grid `vtxc` matched CPU references at roundoff in the Si2 case. The debug-print code has since been removed from production sources.

Si256 validation after cleanup:

- Debug-print scaffolding was removed from `source/source_hamilt/module_xc/xc_pot.cpp`; production files no longer contain `ABACUS_XC_GPU_DEBUG` or `XC_GPU_DEBUG`.
- Fresh Si256 profiles were run with the normal `build/abacus_basic_gpu` executable because `build-test-cuda/abacus_basic_gpu` is not configured with float FFTW and cannot run the single-precision Si256 case.
- Profile outputs:
  - Fallback: `nsight_si256_xc_clean_off_build_20260623/`
  - Resident: `nsight_si256_xc_clean_on_build_20260623/`
- Numerical drift between `ABACUS_XC_GPU=0` and `ABACUS_XC_GPU=1`:
  - Final energy delta: `-9.83687641565e-05 eV`, or `-3.84252984986e-07 eV/atom`
  - Max force-component delta: `1.693478e-04 eV/Angstrom`
  - Max stress-component delta: `4.728356e-04 kbar`
  - Pressure delta: `3.6e-04 kbar`
- ABACUS timer reduction:
  - `PotXC cal_veff`: `14.30 s` -> `1.13 s`
  - `XC_Functional v_xc`: `14.25 s` -> `1.08 s`
  - Total ABACUS timer: `46.49 s` -> `33.78 s`
  - `/usr/bin/time` wall time under Nsight: `52.63 s` -> `39.82 s`
- Nsight CUDA activity gap reduction:
  - Inactive time: `21.596008871 s` -> `8.135898538 s`
  - Active time: `24.528896285 s` -> `25.261329452 s`
  - CUDA utilization over traced span: `53.179288286968415%` -> `75.63900051694081%`
  - Largest inactive gap: `1.826373815 s` -> `1.039062187 s`

Remaining follow-up:

- The resident path still performs one final device-to-host copy because `XC_Functional::v_xc` returns a host `ModuleBase::matrix`. Moving the `PotXC`/effective-potential boundary fully onto the device remains the next residency improvement.
- A compact automated full-interface fixture now covers the resident `XC_Functional::v_xc` dispatch boundary for LSDA spin:
  - Test: `XCResidentOpTest.FullVxcLdaSpinResidentGpuMatchesCpu`
  - It creates a real `Charge` object with GPU density buffers, syncs host real-space spin density to device, toggles `ABACUS_XC_GPU`, calls `XC_Functional::v_xc(..., "cpu")` and `XC_Functional::v_xc(..., "gpu")`, and compares `etxc`, `vtxc`, and the two-row potential matrix.
  - The CUDA XC suite now also includes `XCResidentOpTest.ChargeRealspaceDensitySyncIsNoopOnCpuDevice`.
  - Fresh result: `./build-test-cuda/source/source_hamilt/module_xc/kernels/test/MODULE_HAMILT_XC_Functional_UTs` passes 18 tests outside the sandbox with GPU access.
- The full-interface fixture intentionally targets LSDA spin rather than PBE/PBEsol because the compact PBE resident path depends on a fully initialized GPU PW FFT basis. PBE/PBEsol full-path coverage remains through runtime smoke/profile validation plus focused stage tests.

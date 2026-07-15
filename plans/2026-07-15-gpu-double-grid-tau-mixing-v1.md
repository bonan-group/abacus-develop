# GPU Reciprocal Double-Grid Tau Mixing

## Objective

Enable CUDA GPU-resident reciprocal tau mixing for double-grid PW calculations while preserving the existing CPU algorithm exactly: mix smooth tau coefficients with the selected plain, Broyden, or Pulay workflow; plain-mix high-frequency tau coefficients; recombine the dense reciprocal vector; and inverse transform every spin channel on GPU.

## Pre-Implementation Algorithm and Reuse Boundary

The rho stage in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:383` transforms dense spin-strided reciprocal data, splits it into persistent smooth and high-frequency buffers, mixes the smooth representation, plain-mixes high frequencies, and recombines the dense vector. The persistent buffers are allocated in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:85` and become available for reuse after rho recombination at `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:600`.

The CPU tau reference in `source/source_estate/module_charge/charge_mixing_rho.cpp:396` performs the same double-grid split, smooth history mixing, high-frequency plain mixing, recombination, and dense inverse FFT. Tau deliberately has no Kerker screening and does not use the rho spin packing representation.

Before this work, the GPU tau stage transformed all spin channels but treated the dense spin-strided arrays as a contiguous smooth vector. The support predicate, fallback message, and GPU assertion therefore rejected double-grid tau.

## Implementation Plan

- [ ] 1. Add and run a failing double-grid tau eligibility regression in `source/source_estate/test/charge_mixing_test.cpp`.

  Cover plain, Broyden, and Pulay with `nspin=1`, `nspin=2`, and traditional `nspin=4`. Confirm failure is caused by the support predicate.

- [ ] 2. Remove the double-grid tau gate, fallback reason, and GPU assertion.

  Modify `source/source_estate/module_charge/charge_mixing.cpp`, `charge_mixing_rho.cpp`, and `charge_mixing_rho_gpu.cpp`. Retain all unrelated restrictions.

- [ ] 3. Split dense tau into reusable smooth and high-frequency GPU views.

  In `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:622`, use direct offsets for `nspin=1` and the persistent rho scratch buffers plus CUDA split for `nspin>1`.

- [ ] 4. Mix smooth and high-frequency tau with CPU-equivalent semantics.

  Use `rhopw->npw * nspin` for smooth tau, no screening, rho-derived DIIS coefficients, and scalar `mixing_beta` for high frequencies.

- [ ] 5. Recombine tau and inverse transform every dense-grid spin channel.

  Use CUDA combine for spinful data. Leave `nspin=1` in place because its smooth and high-frequency regions are already contiguous.

- [ ] 6. Update all `mixing_gpu` support-matrix documentation.

  Modify `docs/parameters.yaml`, `docs/advanced/input_files/input-main.md`, and built-in help so only noncollinear angle mixing remains a CPU fallback.

- [ ] 7. Run the CPU and focused CUDA regression suites.

  Require 21/21 CPU tests, 4/4 CUDA CTests, and a clean `git diff --check`.

- [ ] 8. Run a LibXC SCAN double-grid tau GPU smoke.

  Use `nspin=2`, three SCF iterations, `mixing_tau=true`, and `OMP_NUM_THREADS=1`. Require GPU mixing calls plus finite `DRHO` and `DKIN`.

- [ ] 9. Run GPU availability and compute-sanitizer checks.

  Check `nvidia-smi` immediately before runtime validation and require memcheck to report zero errors.

- [ ] 10. Run the ABACUS governance checker through an isolated temporary index.

  Require a non-increasing global dependency budget and record any migration-neutral warnings.

## Verification Criteria

- The eligibility regression passes for all nine combinations of three supported mixing methods and three supported spin configurations with double grid and active tau mixing.
- Smooth tau mixing uses `nspin * rhopw->npw` contiguous coefficients; dense spin strides never enter `Mixing_Data_GPU` directly.
- High-frequency tau coefficients are plain mixed with `mixing_beta` and recombined for every spin channel.
- Existing rho double-grid behavior and workspace lifetime remain unchanged.
- CPU charge-mixing tests pass 21/21.
- Focused CUDA CTest passes 4/4.
- The LibXC SCAN double-grid tau smoke completes three GPU-resident mixing iterations and reports finite `DRHO` and `DKIN` values.
- Compute-sanitizer reports zero CUDA memory errors.
- Documentation and built-in INPUT help describe the same support matrix.
- The governance checker exits successfully and reports a non-increasing global dependency budget.

## Potential Risks and Mitigations

1. **Dense spin-stride corruption**
   Mitigation: Require split/combine for `nspin>1` and mix only contiguous smooth views; rely on the existing nspin=2/4 split/combine CUDA numerical tests.

2. **Workspace aliasing with rho data**
   Mitigation: Reuse buffers only after rho high-frequency recombination has completed; rho and tau retain separate history objects and dense storage.

3. **Incorrect high-frequency beta semantics**
   Mitigation: Match the CPU `mixing_highf` behavior exactly by using scalar `mixing_beta` for all tau high-frequency channels.

4. **Zero high-frequency size**
   Mitigation: Keep the existing `high_frequency_npw > 0` guards and allow combine operations to receive a null high-frequency buffer only when the dense and smooth sizes are equal.

5. **Meta-GGA runtime unavailable in the local CUDA profile**
   Mitigation: Use the already verified system LibXC configuration and record the executable build identity used for the SCAN smoke.

## Alternative Approaches

1. Allocate dedicated tau smooth and high-frequency buffers. This makes ownership visually explicit but duplicates memory whose lifetimes do not overlap with rho scratch use.

2. Introduce a generic double-grid mixing workspace abstraction for rho and tau. This could reduce future duplication but expands the refactor and testing surface without being required for this feature.

3. Keep smooth tau on GPU but copy high-frequency tau to CPU. This is simpler than full GPU support but adds synchronization and violates the requested resident path.

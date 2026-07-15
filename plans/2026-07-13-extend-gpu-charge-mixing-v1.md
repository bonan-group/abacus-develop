# Extend GPU Charge Mixing Coverage

## Objective

Extend ABACUS full GPU-resident reciprocal charge mixing beyond the current CUDA-only `nspin == 1` Broyden/Pulay/no-double-grid path. The target coverage is full GPU plain mixing, GPU spinful `nspin == 2` and `nspin == 4` Broyden/Pulay/plain mixing, GPU double-grid mixing, and `mixing_angle > 0` with double grid. The implementation should preserve the current CPU fallback behavior until each combination has focused CPU/GPU parity evidence.

Current source context:

- Full GPU-resident charge mixing is currently gated to CUDA, `device == "gpu"`, GPU-resident charge, `nspin == 1`, `mixing_mode` in `{broyden, pulay}`, and no double grid in `source/source_estate/module_charge/charge_mixing_rho.cpp:21`.
- Unsupported GPU-resident cases currently fall back with reasons for disabled `mixing_gpu`, non-GPU charge, non-`nspin == 1`, non-Broyden/Pulay mode, or double grid in `source/source_estate/module_charge/charge_mixing_rho.cpp:32`.
- CPU reciprocal mixing already covers `nspin == 1`, `nspin == 2`, traditional `nspin == 4`, and `nspin == 4` angle mixing in `source/source_estate/module_charge/charge_mixing_rho.cpp:83`, `source/source_estate/module_charge/charge_mixing_rho.cpp:92`, `source/source_estate/module_charge/charge_mixing_rho.cpp:163`, and `source/source_estate/module_charge/charge_mixing_rho.cpp:192`.
- CPU double grid splits smooth and high-frequency reciprocal data, mixes the smooth part with the selected method, and mixes high-frequency data with plain mixing in `source/source_estate/module_charge/charge_mixing_rho.cpp:71` and `source/source_estate/module_charge/charge_mixing_rho.cpp:284`.
- `mixing_angle > 0` currently rejects double grid in the reciprocal path in `source/source_estate/module_charge/charge_mixing_rho.cpp:197`, then transforms `{rho, |m|}` through real and reciprocal space in `source/source_estate/module_charge/charge_mixing_rho.cpp:201`.
- The current GPU path allocates `Mixing_Data_GPU`, Broyden/Pulay GPU mixers, inner-product workspaces, and optional tau workspaces in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:14`.
- The current full GPU path asserts `nspin == 1` and Broyden/Pulay in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:263`, then performs GPU FFTs, Kerker screening, Broyden/Pulay coefficient calculation, optional tau mixing, final inverse FFT, and host sync in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:278`, `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:291`, `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:320`, `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:336`, and `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:358`.
- GPU Kerker kernels already accept an `nspin` argument, but the current CPU-facing preconditioner only uses the GPU shortcut for simple same-parameter cases and otherwise falls back for spinful magnetic-parameter handling in `source/source_estate/module_charge/kernels/charge_mixing_op.h:11` and `source/source_estate/module_charge/charge_mixing_preconditioner.cpp:34`.
- CPU reciprocal Broyden/Pulay coefficients use spin-aware Hartree-like inner products with different density and magnetization weighting for `nspin == 2`, traditional `nspin == 4`, and angle-mixing `nspin == 4` in `source/source_estate/module_charge/charge_mixing_residual.cpp:338`, `source/source_estate/module_charge/charge_mixing_residual.cpp:373`, `source/source_estate/module_charge/charge_mixing_residual.cpp:419`, and `source/source_estate/module_charge/charge_mixing_residual.cpp:462`.
- GPU Broyden and Pulay implementations can consume arbitrary contiguous `Mixing_Data_GPU` vectors and batched coefficient builders, but they depend on caller-provided inner products matching CPU semantics in `source/source_base/module_mixing/broyden_mixing_gpu.h:437` and `source/source_base/module_mixing/pulay_mixing_gpu.h:432`.
- Low-level GPU vector kernels already provide subtract, axpy, scale, copy, and gemv-style operations that can support a GPU plain path without a separate history mixer in `source/source_base/module_mixing/kernels/cuda/mixing_op.cu:192`.

Assumptions:

- CUDA is the first implementation target. ROCm continues to use the existing fallback until a separate ROCm plan adds equivalent kernels.
- This plan covers reciprocal-space SCF mixing, because the current full GPU-resident path is reciprocal and the requested features are the current reciprocal-path gaps.
- CPU behavior is authoritative for spin packing, Kerker application, coefficient inner products, double-grid splitting, and angle-mixing semantics.
- `mixing_tau` should remain supported in the combinations where rho mixing is enabled, but tau enablement must not be the gating feature for the first rho parity tests.

## Implementation Plan

- [ ] 1. Baseline current fallback behavior and add failing runtime-gate tests.

- [ ] 2. Implement GPU plain reciprocal mixing for `nspin == 1` without double grid.

- [ ] 3. Add CPU/GPU parity tests for GPU plain mixing, Kerker behavior, and optional tau.

- [ ] 4. Introduce explicit GPU packing and unpacking helpers for spinful reciprocal data.

- [ ] 5. Add spin-aware GPU Kerker support for density and magnetic components.

- [ ] 6. Add spin-aware GPU reciprocal inner-product builders for Broyden and Pulay.

- [ ] 7. Add parity tests for spin packing, spin Kerker, and spin inner products.

- [ ] 8. Activate collinear `nspin == 2` GPU plain, Broyden, and Pulay mixing.

- [ ] 9. Activate traditional `nspin == 4` GPU plain, Broyden, and Pulay mixing.

- [ ] 10. Refactor GPU workspace sizing around an explicit active-mixing descriptor.

- [ ] 11. Add GPU double-grid split and combine support for rho and tau.

- [ ] 12. Enable full GPU double-grid support for `nspin == 1`.

- [ ] 13. Enable GPU double-grid support for `nspin == 2` and traditional `nspin == 4`.

- [ ] 14. Define CPU reference semantics for `mixing_angle > 0` with double grid.

- [ ] 15. Implement CPU and GPU support for angle mixing with double grid.

- [ ] 16. Add parity tests for angle mixing with double grid and near-zero magnetization.

- [ ] 17. Extend full GPU tau mixing to the newly enabled spin and double-grid cases.

- [ ] 18. Update fallback logging, input documentation, and support-matrix docs.

- [ ] 19. Add runtime smoke tests and fallback comparisons for each newly enabled class.

- [ ] 20. Add Nsight and ABACUS timer profiling after correctness is stable.

## Verification Criteria

- Focused CPU/GPU parity tests pass for `nspin == 1` plain GPU mixing with Kerker-on, Kerker-off, low beta, and tau-on cases.
- Spin packing and unpacking GPU tests match CPU layouts for `nspin == 2`, traditional `nspin == 4`, and angle-mixing `nspin == 4`.
- Spin-aware GPU Kerker tests match CPU `Kerker_screen_recip` for charge and magnetic parameter combinations.
- Spin-aware GPU inner-product tests match CPU `inner_product_recip_hartree` for `nspin == 2`, traditional `nspin == 4`, and angle-mixing `nspin == 4`.
- Full GPU Broyden/Pulay/plain runtime tests match CPU fallback rho/rhog output and SCF behavior for `nspin == 2` and traditional `nspin == 4` without double grid.
- GPU double-grid tests match CPU smooth/high-frequency split, high-frequency plain mixing, recombination, and dense-grid inverse FFT behavior for `nspin == 1`, `nspin == 2`, and traditional `nspin == 4`.
- `mixing_angle > 0` with double grid no longer hard exits and matches the defined CPU reference behavior before full GPU runtime enablement.
- `mixing_tau` remains correct for each newly enabled resident combination or is explicitly gated with a clear fallback reason.
- Documentation states the new `mixing_gpu` availability matrix and remaining unsupported cases.
- ABACUS runtime smoke tests are run with `OMP_NUM_THREADS=1`, record the executable identity, and report exact pass/fail status and numerical tolerances.
- Nsight or ABACUS timer evidence shows no new dominant host-device synchronization path after resident support expands.

## Potential Risks and Mitigations

1. **Spinful coefficient metric drift**
   - Impact: Broyden/Pulay may converge differently or incorrectly for magnetic calculations.
   - Likelihood: High if the current `nspin == 1` GPU inner product is reused without spin-aware weighting.
   - Mitigation: Add dedicated GPU inner-product builders that match CPU `inner_product_recip_hartree` before enabling spinful Broyden/Pulay.
   - Contingency: Enable spinful GPU plain first and keep spinful Broyden/Pulay fallback until coefficient parity is proven.

2. **Double-grid memory layout mistakes**
   - Impact: Smooth and high-frequency reciprocal components could be mixed with the wrong method or recombined into the wrong dense-grid slots.
   - Likelihood: Medium, because CPU `divide_data` handles `nspin == 1` and spinful layouts differently.
   - Mitigation: Test split, high-frequency plain mixing, and recombination separately before full runtime enablement.
   - Contingency: Land `nspin == 1` double-grid first and defer spinful double-grid if layout parity is not clear.

3. **Angle-mixing semantics under double grid**
   - Impact: Incorrect magnetization modulus or direction rescaling for noncollinear calculations.
   - Likelihood: High, because this is not currently supported even on the CPU reciprocal path.
   - Mitigation: Define and test the CPU reference behavior before enabling the GPU path; include near-zero magnetization tests.
   - Contingency: Keep `mixing_angle > 0` plus double grid unsupported and document the blocker if the physical semantics are ambiguous.

4. **Overly broad runtime gate**
   - Impact: Unsupported combinations could silently enter an untested GPU path.
   - Likelihood: Medium during gate expansion.
   - Mitigation: Replace the current single conditional with an explicit support predicate that records method, spin, double-grid, angle, tau, build, and residency decisions.
   - Contingency: Default to existing CPU fallback on any unknown combination.

5. **Performance regression from extra packing or synchronization**
   - Impact: Expanded GPU support could be correct but slower than the current CPU fallback.
   - Likelihood: Medium for small systems and spinful double-grid cases.
   - Mitigation: Keep feature opt-in through `mixing_gpu`, profile after correctness, and avoid host round trips in pack, split, coefficient, and combine stages.
   - Contingency: Keep newly supported combinations behind opt-in documentation and recommend fallback where profiling shows no benefit.

## Alternative Approaches

1. **Recommended: staged kernel parity and gated runtime enablement**
   - Description: Implement plain, spinful kernels, double grid, and angle double-grid in separate phases, enabling each runtime combination only after focused parity tests pass.
   - Pros: Lowest correctness risk, aligns with existing CPU behavior, and allows partial delivery if later phases expose ambiguity.
   - Cons: More test scaffolding and more intermediate gates.

2. **Fast-path only for plain and `nspin == 2`**
   - Description: Implement GPU plain and collinear spin first, leaving `nspin == 4`, double grid, and angle double-grid for later.
   - Pros: Delivers useful coverage quickly and avoids the hardest noncollinear/double-grid interactions.
   - Cons: Does not satisfy the full requested matrix in one plan.

3. **Single generic GPU vector engine for all combinations**
   - Description: Build a fully generic component-layout descriptor and route all methods/spins/grids through it at once.
   - Pros: Potentially cleaner long-term abstraction.
   - Cons: Higher initial blast radius, harder to prove parity, and more risk of changing current `nspin == 1` behavior.

4. **Partial GPU offload with CPU coefficients**
   - Description: Keep GPU FFT and residual formation, copy spinful vectors to host for CPU Broyden/Pulay coefficients, then copy mixed results back.
   - Pros: Lower implementation effort for spinful inner products.
   - Cons: Not full GPU-resident, adds host-device traffic, and does not address the support gap as requested.

## Implementation Notes

- The first code change should be tests for the current missing runtime gates and plain GPU parity, not a gate expansion.
- The support predicate should be easy to log and unit test; it should describe why a case uses GPU or falls back.
- Plain GPU mixing can avoid a new `Plain_Mixing_GPU` class by using existing GPU vector operations, but the final implementation should choose the smallest design that keeps rho, tau, and high-frequency plain mixing easy to share.
- Spinful Broyden/Pulay should not be enabled until the GPU coefficient builders use CPU-equivalent spin metrics.
- `mixing_angle > 0` plus double grid should be treated as a CPU semantics feature first and a GPU acceleration feature second.

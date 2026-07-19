# USPP Low-Rank Davidson Overlap

## Objective

Eliminate the standalone full-size `V(Q becp)` back-projection from the PW USPP `dav_subspace` hot path while preserving the generalized eigenproblem, existing CPU/GPU numerical behavior, and fallback support for every solver and Hamiltonian that does not expose the required low-rank projector capability.

The target formulation uses the USPP structure `H = H0 + V D V^H` and `S = I + V Q V^H`. Davidson will retain beta coefficients for its basis, construct the reduced H and S matrices from low-rank terms, and form each unconverged residual with one combined `V(D - epsilon Q)` back-projection. This is intentionally broader than adding a residual callback: the current solver requires separate H and S basis blocks in `source/source_hsolver/diago_dav_subspace.cpp:143-154`, `source/source_hsolver/diago_dav_subspace.cpp:480-483`, and `source/source_hsolver/diago_dav_subspace.cpp:490-590`, so the reduced-matrix and refresh logic must be updated together.

## Current Evidence and Scope

- The matched Si256 GPU run uses 4608 USPP projectors and spends 385.91 seconds versus 137.76 seconds for NCP. Forcing the already allocated full projector path reduces USPP to 337.29 seconds.
- A matched one-k-point trace at the same Si256 matrix dimensions contains exactly 12 standalone overlap back-projection GEMMs, each taking about 1.11 seconds. The preceding `Q becp` kernel takes only 1.16 milliseconds across all 12 calls.
- The existing implementation already reuses one `V^H psi` result between adjacent H and S calls through `source/source_pw/module_pwdft/op_pw_nl.cpp:301-395` and `source/source_pw/module_pwdft/op_pw_nl.cpp:521-562`.
- Initial scope is PW, collinear USPP, `ks_solver=dav_subspace`, CPU and GPU template compatibility, full and chunked projector storage, double and float compilation, and serial plus pool-MPI correctness. Existing noncollinear USPP remains unsupported.
- CG, BPCG, legacy Davidson, NCP, LCAO, EXX, DFT+U, onsite projectors, and Hamiltonians without the low-rank capability continue through their current interfaces. No INPUT parameter or user documentation change is planned.

## Architectural Decisions

Introduce a generic low-rank generalized-operator capability at the Hamiltonian/solver boundary rather than making `source_hsolver` depend on PW pseudopotential classes. The capability exposes projector rank, projection of a vector block, D and Q coefficient transforms, back-projection, and application of H with one identified nonlocal operator excluded. `HamiltPW` supplies this capability from its existing `Nonlocal` instance; the base Hamiltonian reports it unavailable.

The solver owns basis-lifetime coefficient storage with column-major shape projector-rank by Davidson-basis-size. This avoids relying on the current one-shot pointer cache after a wavefunction buffer is mutated. Globally reduced beta coefficients must not be reduced again when their low-rank contribution is added to reduced H or S matrices.

Delivery is split into two numerical checkpoints. The overlap-only checkpoint replaces stored explicit S basis blocks with `psi^H psi + B^H Q B`, retaining the current full H basis blocks. The final checkpoint stores `H0 psi`, constructs `Hcc` with `B^H D B`, and forms residuals with one combined back-projection. The first checkpoint is retained as a review and debugging boundary, not as a separate permanent mode.

## Affected Files

- `source/source_hamilt/hamilt.h`: optional generic low-rank generalized-operator capability.
- `source/source_hamilt/operator.h` and `source/source_hamilt/operator.cpp`: identity-based operator-chain exclusion for applying H0 without accidentally excluding other projector operators.
- `source/source_pw/module_pwdft/hamilt_pw.h` and `source/source_pw/module_pwdft/hamilt_pw.cpp`: PW capability adapter and H0 application.
- `source/source_pw/module_pwdft/op_pw_nl.h` and `source/source_pw/module_pwdft/op_pw_nl.cpp`: stable projection, D/Q transform, and full/chunked back-projection operations.
- `source/source_pw/module_pwdft/vnl_pw.h` and `source/source_pw/module_pwdft/vnl_pw.cpp`: immutable full-versus-chunked allocation policy used by all later calls.
- `source/source_hsolver/hsolver_pw.cpp`: capability discovery and explicit Davidson callback/workspace wiring.
- `source/source_hsolver/diago_dav_subspace.h` and `source/source_hsolver/diago_dav_subspace.cpp`: basis coefficient storage, reduced matrices, residual, refresh, and fallback path.
- `source/source_hsolver/test/`, `source/source_pw/module_pwdft/kernels/test/`, and their `CMakeLists.txt` files: algebra, callback, heterogeneous-kernel, and fallback tests.
- `tests/11_PW_GPU/` and `.github/workflows/cuda.yml`: focused runtime and CI coverage using the existing USPP cases.

## Implementation Plan

- [ ] 1. Capture the current full and chunked Si32 and one-k-point Si256 references, including energies, eigenvalues, occupations, Davidson call counts, projector GEMM counts, peak memory, and the exact executable identity, so every later checkpoint has a stable correctness and performance baseline.
- [ ] 2. Freeze the projector storage policy selected in `source/source_pw/module_pwdft/vnl_pw.cpp:265-343`; replace later free-memory re-evaluation in `source/source_pw/module_pwdft/op_pw_nl.cpp:204-213`, `source/source_pw/module_pwdft/op_pw_nl.cpp:398-433`, and `source/source_pw/module_pwdft/op_pw_nl.cpp:503-518` with the immutable allocated-storage state, and test that a run cannot retain full `vkb` while switching to chunk reconstruction.
- [ ] 3. Add CPU algebra tests for a small synthetic low-rank generalized problem, comparing explicit dense H and S against reduced formulas for diagonal blocks, off-diagonal basis expansion, Ritz residuals, and basis refresh; include nontrivial complex coefficients, multiple unconverged vectors, and Hermitian non-diagonal D and Q blocks.
- [ ] 4. Define an optional generic low-rank capability in `source/source_hamilt/hamilt.h` without default arguments or global state, with explicit operations for projector rank, globally reduced beta projection, D/Q coefficient transforms, coefficient back-projection, and H0 application; keep the existing H/S interface unchanged for callers without the capability.
- [ ] 5. Add an identity-based operator-chain traversal in `source/source_hamilt/operator.h` and `source/source_hamilt/operator.cpp` that excludes only the owned `Nonlocal` instance, then verify with a mock chain that kinetic, local, meta, EXX, DFT+U, and onsite contributions remain included even if another operator shares a calculation-type label.
- [ ] 6. Refactor `Nonlocal` in `source/source_pw/module_pwdft/op_pw_nl.h` and `source/source_pw/module_pwdft/op_pw_nl.cpp` into stable primitives that project `B = V^H psi`, construct `D B` and `Q B`, and back-project arbitrary coefficient blocks through either full or chunked V; preserve current pool reduction semantics and reuse existing D/Q kernels instead of duplicating pseudopotential indexing.
- [ ] 7. Wire the PW capability through `source/source_pw/module_pwdft/hamilt_pw.h`, `source/source_pw/module_pwdft/hamilt_pw.cpp`, and `source/source_hsolver/hsolver_pw.cpp`, enabling it only for supported collinear USPP `dav_subspace` calculations and passing an explicit unavailable capability for all fallback paths.
- [ ] 8. Implement the overlap-only Davidson checkpoint in `source/source_hsolver/diago_dav_subspace.h` and `source/source_hsolver/diago_dav_subspace.cpp`: store B and Q B for each basis block, remove explicit S basis storage on the capability path, update Scc as the pool-reduced local `psi^H psi` term plus one non-reduced `B^H Q B` term, form the overlap part of each residual from Ritz-space coefficients, and transform B/Q B during refresh.
- [ ] 9. Validate the overlap-only checkpoint against the explicit-S fallback on CPU, CUDA, full-projector, chunked-projector, one-rank, and two-rank cases; require identical Davidson convergence decisions and demonstrate that standalone overlap back-projection calls fall from two per `diag_once` to one residual-time call before proceeding.
- [ ] 10. Complete the low-rank H checkpoint by storing H0 basis blocks, adding `B^H D B` to Hcc after the local pool reduction, and replacing separate nonlocal-H and overlap residual contributions with one band-dependent `V(D B C - Q B C epsilon)` back-projection; update refresh to transform H0 and B while leaving the generalized reduced eigensolve unchanged.
- [ ] 11. Remove capability-path H/S workspaces that are no longer read, add dedicated timers and profiler-visible ranges for projection, D/Q transforms, reduced low-rank updates, and combined back-projection, and verify that fallback builds retain their previous allocation and timer behavior.
- [ ] 12. Extend focused unit and integration coverage for nspin 1 and 2, full and chunked storage, equal and changing unconverged-band counts, Davidson refresh, `vnl_in_h` true and false, unsupported noncollinear fallback, CPU/CUDA numerical parity, MPI coefficient ownership, sanitizer safety, and the three existing USPP GPU mixing cases.
- [ ] 13. Run final performance validation on idle GPU hardware: compare the one-k-point Si256-sized Nsight trace and the full six-k-point Si256 benchmark against NCP, explicit-S USPP, and forced-full USPP references; retain the optimization only if standalone full-size `V(Q becp)` GEMMs are absent and numerical acceptance criteria pass.
- [ ] 14. Run the focused CTest sets, CUDA memcheck, exact runtime matrix, `git diff --check`, and the scoped agent-governance checker; record that no INPUT documentation changed and document any remaining low-rank global-reduction or unsupported-spin limitations in the change summary.

## Verification Criteria

- Synthetic low-rank reduced H, reduced S, and residual results agree with explicit dense H/S evaluation within `1e-12` for complex double CPU tests and the existing heterogeneous-kernel tolerance for complex float.
- Capability and fallback Davidson paths produce the same converged eigenvalues, occupations, total energy, iteration decisions, and final wavefunctions up to phase/subspace equivalence on small deterministic USPP cases.
- Si32 full versus chunked GPU runs retain the established energy agreement, and CPU/GPU Si2 force and stress references remain within their existing tolerances after the solver change.
- One-rank and two-rank tests agree, proving that globally reduced B coefficients are not multiplied by the pool size when low-rank Hcc/Scc terms are added.
- NCP, CG, BPCG, legacy Davidson, noncollinear rejection, `vnl_in_h=false`, and Hamiltonians without the capability follow the unchanged fallback path.
- The one-k-point Si256-sized trace contains zero standalone full-size `V(Q becp)` overlap GEMMs and at most one combined nonlocal back-projection per Davidson gradient expansion.
- The optimized one-k-point USPP GPU-kernel time improves by at least 15 percent from the 54.895-second forced-full reference, and the six-k-point Si256 total improves from 337.29 seconds without increasing peak memory above the forced-full reference.
- Compute-sanitizer reports zero memory errors for the optimized full and chunked paths, and focused CUDA/CPU CTest targets pass without relaxed references.
- The global dependency budget remains non-increasing, all additions are C++11-compatible, new test sources are explicitly registered, and `git diff --check` plus the scoped governance checker pass.

## Potential Risks and Mitigations

1. **Direct residual alone is algebraically insufficient for current Davidson.** The solver also requires separate reduced H and S matrices. Mitigation: update reduced-matrix construction and refresh from the same low-rank basis coefficients before enabling the combined residual.

2. **MPI low-rank terms may be over-counted.** B is globally reduced and replicated, unlike local real-space/PW dot products. Mitigation: reduce only local `psi^H H0 psi` and `psi^H psi` terms, then add `B^H D B` and `B^H Q B` once after reduction; exercise at least two pool ranks.

3. **One-shot beta cache identity is unsafe for mutable wavefunction buffers.** Pointer equality does not imply unchanged data. Mitigation: copy each accepted basis block into solver-owned coefficient storage immediately after projection and never use pointer identity across a solver mutation.

4. **Refresh can desynchronize psi, H0 psi, B, and Q B.** Mitigation: transform all capability-path basis objects with the same Vcc operation and validate refresh explicitly against dense recomputation.

5. **Excluding nonlocal H may accidentally exclude another projector-like operator.** Mitigation: skip the exact owned `Nonlocal` pointer, not every operator with the same calculation type, and test mixed operator chains.

6. **Band-dependent epsilon makes combined coefficients easy to lay out incorrectly.** Mitigation: define one column-major projector-by-vector contract and test unequal eigenvalues, non-proportional vectors, and multiple unconverged-band orderings.

7. **Chunked back-projection may regress through small GEMM fragmentation.** Mitigation: freeze storage policy first, use the existing atom-aligned chunk machinery, and validate with both 64 and memory-budgeted larger chunks.

8. **The optimization could change convergence despite matching one application.** Mitigation: compare complete Davidson iteration decisions, refresh behavior, eigenpairs, and converged SCF results rather than relying only on kernel parity.

9. **Removing S workspace may make rollback difficult during implementation.** Mitigation: retain the explicit-S fallback until all capability tests and performance gates pass, and land the overlap-only checkpoint before removing dead capability-path storage.

10. **ROCm or float template instantiation may break despite a CUDA-only performance target.** Mitigation: keep algebra device-generic, avoid CUDA types in public interfaces, compile CPU/CUDA/ROCm instantiations where available, and gate only runtime enablement rather than compilation.

## Alternative Approaches

1. **Fused H/S wide GEMM:** Keep explicit Hpsi and Spsi but concatenate D B and Q B for one wider back-projection. This is lower risk and may reuse V reads, but it preserves nearly all projector FLOPs and the large S workspace.

2. **Overlap-only low-rank path as the final design:** Construct Scc from B and Q and use one residual-time Q back-projection while retaining full Hpsi. This removes about half the measured overlap GEMMs with less operator-chain work, but leaves the D and Q residual contributions separate.

3. **Transform to a standard eigenproblem with an S inverse or square root:** This could remove explicit S applications from Davidson, but constructing and applying the transformation for distributed low-rank USPP overlap is more invasive and introduces conditioning risks.

4. **Mixed-precision projector contractions:** Complex-float projector GEMMs with double correction could improve throughput and memory independently, but they do not remove repeated work and require a separate numerical-accuracy plan.

5. **Keep explicit S and tune chunking only:** Freezing policy and increasing chunk size already improve the baseline, but the one-k-point trace shows the standalone overlap GEMM alone costs 13.32 seconds, so chunk tuning cannot close the full USPP/NCP gap.

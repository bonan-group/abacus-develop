# CUDA Device-Resident Ultrasoft Pseudopotentials

## Objective

Make the scalar and collinear-spin PW ultrasoft pseudopotential (USPP) path correct on CUDA and keep its overlap, projector occupations, augmentation density, dense reciprocal charge, and effective projector coefficients on the GPU. Connect that path to the branch's existing CUDA random-wavefunction initialization, reciprocal double-grid rho/tau mixing, resident potential work, and chunked nonlocal force/stress kernels. Preserve CPU behavior and explicit fallbacks for unsupported configurations.

The first supported envelope is `nspin=1` and `nspin=2`, double precision and mixed precision, reciprocal-space mixing, one MPI rank per FFT pool, and LDA/GGA SCF. Existing noncollinear USPP is not included because overlap application exits and the occupation branches are empty in `source/source_pw/module_pwdft/hamilt_pw.cpp:235` and `source/source_estate/elecstate_pw.cpp:367`. Meta-GGA tau mixing is included, but meta-GGA potential evaluation may retain its documented host fallback until a separate resident meta-GGA implementation exists.

## Current Support Audit

| Optimized path | Current USPP status | Planned action |
| --- | --- | --- |
| CUDA random wavefunction initialization | The random coefficients are produced on device, but CG subspace diagonalization reaches the unsafe USPP overlap loop in `source/source_psi/psi_prepare.cpp:279` and `source/source_pw/module_pwdft/hamilt_pw.cpp:247`. | Fix overlap application first and add random-plus-CG USPP coverage. |
| Batched/chunked nonlocal Hamiltonian | The nonlocal kernels consume device `deeq`; the chunked path already treats USPP as non-diagonal in `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:38`. | Retain and regression-test; do not duplicate this implementation. |
| USPP overlap `S|psi>` | Host loops pack device `qq_nt` into device `qqc` in `source/source_pw/module_pwdft/hamilt_pw.cpp:253`. | Replace the packing loop and per-atom tiny GEMMs with a CUDA overlap contraction using explicit atom/projector metadata. |
| Projector occupations `becsum` | GPU GEMM uses the unallocated host matrix leading dimension, then host loops dereference device `becp`, scratch, and `becsum` in `source/source_estate/elecstate_pw.cpp:327` and `source/source_estate/elecstate_pw.cpp:373`. | Use `vkbnc` and a CUDA occupation contraction that writes packed symmetric occupations directly. |
| Augmentation density | Device allocations are filled and accumulated by host loops; the templated radial transform also dereferences device buffers on the host in `source/source_estate/elecstate_pw.cpp:494` and `source/source_pw/module_pwdft/vnl_pw.cpp:1187`. | Cache static augmentation form factors and phases, then contract and accumulate entirely on CUDA. |
| Reciprocal double-grid rho/tau mixing | The CUDA mixer already supports dense/smooth splitting and tau in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:382`, but USPP crashes before reaching it. | Reuse unchanged where possible and add USPP workflow assertions to CI. |
| Resident XC/potential update | The support predicate rejects every double-grid case in `source/source_estate/module_pot/potential_new.cpp:232`; interpolation is host-only in `source/source_estate/module_pot/potential_new.cpp:425`. | Add dense-device potential storage and CUDA dense-to-smooth interpolation for supported LDA/GGA components. |
| USPP effective projector coefficients | `newq` rebuilds the same augmentation form factors and contracts effective potential on the CPU every SCF iteration in `source/source_pw/module_pwdft/vnl_pw.cpp:1582`. | Reuse the augmentation cache and compute `deeq` from device effective potential, copying only when a host consumer requests it. |
| GPU structure factors | Device kernels exist, but atom metadata is allocated and copied on each call in `source/source_pw/module_pwdft/structure_factor_k.cpp:69`. | Reuse the device phase machinery and move stable metadata into the USPP workspace. |
| Chunked nonlocal force/stress | CUDA kernels already consume `qq_nt`, `deeq`, occupations, and eigenvalues on device in `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1271`. | Keep this path and validate USPP numerics after SCF support is restored. |
| USPP augmentation force/stress | Separate routines read device `becsum` from CPU code and rebuild augmentation data on host in `source/source_pw/module_pwdft/forces_us.cpp:32` and `source/source_pw/module_pwdft/stress_us.cpp:26`. | Port these contractions in a later gate using the shared augmentation workspace. |
| Symmetry and multi-pool density reduction | Density symmetry and pool reduction consume host `Charge::rho` in `source/source_estate/module_charge/symmetry_rho.cpp:49` and `source/source_estate/module_charge/charge_mpi.cpp:129`. | Keep an explicit materialization boundary initially; add a device-aware path only after single-rank correctness. |

## Implementation Plan

- [ ] 1. Freeze the USPP CUDA support contract and add failing reproductions before changing production code.

  Add focused tests for `nspin=1` and `nspin=2`, mixed species with `nh < nhm`, multiple bands, Gamma and non-Gamma k points, and both atomic and random initialization. Record the current failures at overlap application, `becsum`, augmentation density, and effective-D assembly. Use the supplied Si USPP for local acceptance and a repository-owned USPP fixture for CI. Cite the support boundary in test names so noncollinear USPP cannot be mistaken for a regression.

- [ ] 2. Introduce a narrowly owned reusable USPP device workspace and explicit refresh points.

  Place projector-pair metadata, atom/type offsets, stable atom positions, dense-grid dimensions, cached species form factors, atom phase factors, and reusable contraction scratch behind a C++11-compatible owner associated with `pseudopot_cell_vnl` or `ElecStatePW`. Prepare or refresh it explicitly when the cell, ionic positions, pseudopotentials, precision, or dense PW basis changes; do not infer workflow changes from unrelated mutable flags. Keep headers minimal and wire every new translation unit through `source/source_pw/module_pwdft/CMakeLists.txt` or `source/source_estate/CMakeLists.txt` as appropriate.

- [ ] 3. Replace the unsafe USPP overlap packing with a dedicated CPU/CUDA operation.

  Add an overlap contraction beside the existing nonlocal kernels in `source/source_pw/module_pwdft/kernels/nonlocal_op.h`. Consume real `qq_nt`, per-type `nh`, global `nhm`, atom projector offsets, band-major `becp`, and write `ps` without host access or temporary complex matrices. Preserve the existing BLAS projection and final `vkb * ps` operations in `source/source_pw/module_pwdft/hamilt_pw.cpp:194`. Test float and double precision against an independent CPU reference, including unequal `nh` and `nhm`, one and many bands, and zero-projector types.

- [ ] 4. Compute packed projector occupations directly on CUDA.

  Correct the `vkb` leading dimension to `vkbnc` in `source/source_estate/elecstate_pw.cpp:316`, stage band weights once per k point, and launch a contraction over atoms and upper-triangular projector pairs. Accumulate the diagonal once and off-diagonal real part twice into the existing spin/atom/pair layout, matching `source/source_estate/elecstate_pw.cpp:410`. Use device-aware pool reduction for `becp` or `becsum` through the existing `Parallel_Common` abstraction rather than passing device pointers to the host-only `Parallel_Reduce` interface.

- [ ] 5. Build and cache dense-grid augmentation form factors and atom phases.

  Compute each USPP species' packed augmentation form factors once per dense-grid/cell refresh using the existing CPU polynomial reference in `source/source_pw/module_pwdft/vnl_pw.cpp:1026`, then upload the immutable result, or provide a CUDA builder only if profiling shows refresh cost matters. Generate or reuse device atom phase factors from the structure-factor kernels in `source/source_pw/module_pwdft/structure_factor_k.cpp:50`. Share this cache between charge augmentation, effective-D assembly, and later force/stress work so `radial_fft_q` is not repeated every SCF iteration.

- [ ] 6. Assemble USPP augmentation charge into dense `rhog` entirely on CUDA.

  Replace the host packing, phase, and pointwise accumulation loops in `source/source_estate/elecstate_pw.cpp:511` with device packing/casting, batched atom contraction, and a fused projector-pair accumulation. Keep the existing smooth-grid forward FFT and dense-grid inverse FFT in `source/source_estate/elecstate_pw.cpp:448`, but make the destination alias `Charge` device storage in double precision and use device-to-device conversion for mixed precision. Avoid per-spin and per-atom allocations inside the SCF loop by reusing workspace capacity.

- [ ] 7. Add a CUDA path for USPP effective-D assembly using the same cache.

  Refactor `cal_effective_D` so the CPU implementation remains the reference while CUDA accepts dense device effective potential, performs its reciprocal transform, contracts it with cached phases and augmentation form factors, handles the Gamma correction, adds `dvan`, mirrors the symmetric projector matrix, and leaves `deeq` resident. Preserve an explicit host-materialization accessor for output and legacy force paths. Validate the device result against `newq` across multiple atoms, spins, projector counts, and Gamma modes before switching the SCF call at `source/source_esolver/esolver_ks_pw.cpp:258`.

- [ ] 8. Make charge ownership and synchronization boundaries explicit through the SCF loop.

  Remove the unconditional device-to-host-to-device round trip in `source/source_estate/elecstate_pw.cpp:149` for one-rank, symmetry-disabled CUDA runs. Materialize host `rho`, `rhog`, `kin_r`, `becsum`, or `deeq` only for a named host consumer such as symmetry, multi-pool reduction, reporting, or a fallback component, then restore the device copy through one ownership API. Preserve the current one-rank-per-FFT-pool restriction in `source/source_estate/module_charge/charge_mixing_residual.cpp:10`; treat multi-pool support as a correctness boundary, not an implicit CUDA-aware MPI assumption.

- [ ] 9. Connect USPP to existing reciprocal double-grid rho and tau mixing.

  Reuse the split, packed-spin, history, high-frequency plain mixing, recombination, and dense inverse FFT already implemented in `source/source_estate/module_charge/charge_mixing_rho_gpu.cpp:399`. Verify that USPP-produced dense `rhog` and tau buffers enter this path without a host reconstruction and that plain, Broyden, and Pulay retain CPU-equivalent convergence semantics. Keep `mixing_angle` plus USPP unsupported with `nspin=4`, consistent with the wider noncollinear USPP limitation.

- [ ] 10. Extend the resident LDA/GGA potential path to double-grid USPP calculations.

  Allocate a dense device effective-potential buffer, add fixed, Hartree, and CUDA XC contributions on the dense grid, and interpolate dense reciprocal coefficients to `d_veff_smooth` with the existing device PW transforms instead of `Potential::interpolate_vrs` host calls. Introduce a CUDA Hartree reciprocal kernel or retain a clearly measured host Hartree boundary until that kernel lands; do not label the latter end-to-end resident. Continue to reject unsupported potential components and KED/meta-GGA in `supports_resident_gpu_update` with a single explicit fallback message.

- [ ] 11. Port the separate USPP augmentation force and stress contributions after SCF correctness is stable.

  Reuse cached augmentation form factors, phases, device `becsum`, and device effective potential to replace host construction and contraction in `source/source_pw/module_pwdft/forces_us.cpp:36` and `source/source_pw/module_pwdft/stress_us.cpp:30`. Return only the final `3 * nat` force and `3 * 3` stress reductions to host. Test these terms independently, then together with the already optimized chunked nonlocal force/stress path; keep this gate separable from the SCF merge if review or sanitizer failures appear.

- [ ] 12. Add CUDA unit, integration, sanitizer, and CI coverage for the complete supported matrix.

  Wire CUDA kernel tests alongside the existing estate and PW kernel tests, and add compact GPU workflows using repository USPP data: plain `nspin=1`, Broyden `nspin=2`, and Pulay plus tau `nspin=2`, all with an actual dense/smooth grid split. Require logs to show USPP detection, CUDA reciprocal mixing, finite SCF residuals, and unequal smooth/dense plane-wave counts. Add random-plus-CG coverage for overlap and a force/stress case when gate 11 lands. Extend `.github/workflows/cuda.yml:53` without replacing the existing norm-conserving matrix.

- [ ] 13. Validate numerical equivalence, memory safety, residency, and performance in increasing system sizes.

  Run CPU versus CUDA comparisons for total energy, band energy, integrated charge, augmentation occupations, forces, and stress on a small system before the 256-atom Si case. Run compute-sanitizer memcheck on the small CUDA case. Use Nsight Systems or CUDA event attribution to verify no host dereference of device buffers and no per-iteration rebuild of static augmentation data. Before any performance run, confirm the requested GPU has no unrelated compute process with `nvidia-smi`; skip and report the benchmark if it is occupied. Finish with `git diff --check`, focused CTest/integration results, and `tools/03_code_analysis/agent_governance_check.py` against the branch base.

## Verification Criteria

- A small scalar Si calculation using `/home/bonan/appdir/CASTEP_UPF/Si_C19MK2_PBE_OTF.upf`, `ecutwfc=30 Ry`, an explicit dense cutoff, and a `2 2 2` k mesh completes at least three SCF iterations on CUDA with finite energy and residuals.
- CPU and CUDA overlap output, packed `becsum`, augmentation `rhog`, and `deeq` agree within precision-appropriate tolerances for mixed projector counts and multiple atoms.
- The CUDA run reaches `INFO: Using GPU-resident reciprocal charge mixing.` for plain, Broyden, and Pulay USPP workflows; the tau case also reports finite `DKIN`.
- Smooth and dense plane-wave counts differ in every double-grid workflow, and the dense high-frequency augmentation coefficients are nonzero for the USPP fixture.
- Random CUDA initialization with CG reaches and completes subspace diagonalization without `ABACUS_PSI_INIT_CPU_DEBUG`.
- Compute-sanitizer memcheck reports zero invalid accesses for the small USPP SCF case and, after gate 11, the force/stress case.
- Host/device transfer attribution shows static augmentation form factors and atom metadata are not rebuilt or uploaded each SCF iteration.
- With symmetry disabled and one rank per FFT pool, no bulk `becp`, `becsum`, augmentation `rhog`, or `deeq` device-to-host transfer occurs inside the USPP kernels.
- With symmetry or a legacy host consumer enabled, results remain correct and the materialization boundary is explicit and tested.
- USPP energy, charge, force, and stress match the CPU reference at the integration test tolerances; norm-conserving CUDA tests remain unchanged and passing.
- The compact USPP CUDA matrix is additive to the existing norm-conserving CI matrix in `.github/workflows/cuda.yml`.
- Performance is reported only from an otherwise idle GPU and includes executable commit, pseudopotential identity, MPI/OpenMP layout, cutoffs, k mesh, and timer/profile evidence.

## Potential Risks and Mitigations

1. **Incorrect packed projector indexing across species**
   - Impact: Silent charge, overlap, or effective-D corruption when `nh` differs from `nhm`.
   - Likelihood: High without dedicated coverage.
   - Mitigation: Use explicit type/atom offset arrays and independent references with mixed species and unequal projector counts.
   - Contingency: Keep the CPU implementation selectable as a diagnostic reference while isolating the failing operation.

2. **Workspace invalidation after ionic or cell updates**
   - Impact: Stale phases or augmentation form factors produce wrong MD, relaxation, force, or stress results.
   - Likelihood: Medium.
   - Mitigation: Refresh through explicit setup/update calls keyed to cell, positions, basis, and precision; test a displaced-atom refresh.
   - Contingency: Rebuild at each ionic step until a narrower invalidation contract is proven.

3. **Gamma-only factor or conjugation mismatch**
   - Impact: Correct-looking but systematically wrong augmentation density and `deeq`.
   - Likelihood: Medium.
   - Mitigation: Compare Gamma and non-Gamma CPU references and isolate the G=0 correction in tests.
   - Contingency: Keep the existing CPU `newq` path for Gamma-only until parity is demonstrated.

4. **MPI receives device pointers through host-only collectives**
   - Impact: Runtime failure or corruption on installations without CUDA-aware MPI.
   - Likelihood: Medium.
   - Mitigation: Use `Parallel_Common` device wrappers with host staging fallback and retain one rank per FFT pool for the first release.
   - Contingency: Gate multi-rank CUDA USPP with a clear error instead of silently assuming CUDA-aware MPI.

5. **Host mirror staleness in legacy consumers**
   - Impact: Energies, symmetry, output, or forces consume an old density or potential.
   - Likelihood: High if synchronization remains ad hoc.
   - Mitigation: Centralize named materialization operations and add transition tests around symmetry, mixing, and effective-D calls.
   - Contingency: Materialize at the outer SCF boundary while preserving device-resident USPP internals, then remove transfers incrementally.

6. **USPP cache memory growth on large dense grids**
   - Impact: Cached form factors consume substantial GPU memory for many species/projector pairs.
   - Likelihood: Medium for large pseudopotentials.
   - Mitigation: Record allocations, size by actual packed `nij`, share the cache across consumers, and add a configurable chunked contraction when capacity is exceeded.
   - Contingency: Stream one species at a time while retaining device-resident dynamic data.

7. **Force/stress expansion destabilizes the core SCF change**
   - Impact: A broad patch becomes difficult to review and validate.
   - Likelihood: Medium.
   - Mitigation: Land force/stress as a separate gate after overlap, density, mixing, and effective-D tests pass.
   - Contingency: Merge SCF support with an explicit host force/stress fallback and follow immediately with gate 11.

## Alternative Approaches

1. **Patch each host loop with temporary device copies**
   - Description: Copy `qq_nt`, `becp`, scratch arrays, and `becsum` to host around the existing loops.
   - Pros: Smallest correctness patch and useful as a diagnostic oracle.
   - Cons: Repeated synchronization, high memory traffic, no device residency, and no reuse by effective-D or force/stress.
   - Recommendation: Use only for debugging, not as the supported implementation.

2. **Port `radial_fft_q` polynomial interpolation directly to CUDA every iteration**
   - Description: Upload `qrad` and metadata, then generate each projector-pair form factor on demand.
   - Pros: Keeps all setup and evaluation on GPU and follows the current function boundary.
   - Cons: Repeats static work each SCF step and complicates irregular interpolation metadata.
   - Recommendation: Prefer cached CPU-built form factors first; revisit a CUDA builder only if ionic-step profiling justifies it.

3. **Use batched BLAS for every atom and projector pair**
   - Description: Pack compact matrices and issue strided batched GEMMs for overlap, occupations, and augmentation.
   - Pros: Relies on tuned vendor libraries.
   - Cons: Requires packing, many small matrices, awkward variable projector counts, and extra workspace.
   - Recommendation: Use BLAS for the large atom/form-factor contractions, but use direct CUDA kernels for irregular packed indexing and fused accumulation.

4. **Attempt end-to-end host-free PW SCF in one change**
   - Description: Port symmetry, convergence norms, Hartree, all potential components, reporting, and USPP together.
   - Pros: Clean final residency model.
   - Cons: Very large blast radius and obscures whether failures come from USPP physics or general SCF ownership changes.
   - Recommendation: Follow the staged gates above; make every remaining host boundary visible and measurable.

## Assumptions

- The first release targets CUDA; CPU remains the numerical reference and ROCm keeps a correct fallback until equivalent kernels are implemented.
- Scalar and collinear USPP are the supported physics envelope. Noncollinear and spin-orbit USPP require a separate design because current reference branches are incomplete.
- Reciprocal-space mixing remains the only optimized mixing target for PW USPP.
- The CUDA PW FFT restriction of one MPI rank per pool remains in force for the initial implementation.
- Repository CI uses repository-owned pseudopotentials; the supplied CASTEP Si USPP is a local acceptance and performance fixture, not a new committed dependency.
- Static augmentation form factors may be prepared on CPU at cell/basis refresh as long as SCF iterations do not rebuild or transfer them.

## Dependencies

- A CUDA build with the existing PW, cuFFT, cuBLAS, LibXC, and GPU test targets enabled.
- An idle NVIDIA GPU for runtime, sanitizer, and performance validation; MPI and GPU commands must run outside the restricted sandbox.
- Existing device PW transforms and double-grid index maps in `source/source_basis/module_pw/pw_transform_gpu.cpp` and `source/source_basis/module_pw/pw_basis_sup.cpp`.
- Existing device memory and collective abstractions in `source/source_base/module_device/memory_op.h` and `source/source_base/parallel_device.h`.
- Existing repository USPP integration fixtures under `tests/01_PW` and `tests/PP_ORB` for CI construction.

## Notes

- No INPUT behavior change is required for the core implementation. If a diagnostic or fallback control is added, update both `docs/parameters.yaml` and `docs/advanced/input_files/input-main.md`.
- The earlier 256-atom Si comparison cannot produce a fair timing until overlap, occupations, augmentation density, and effective-D assembly all complete. Repeat it only after the small-system criteria pass.
- A successful GPU reciprocal mixer log proves only the mixing stage is resident. End-to-end claims require transfer profiling across density, symmetry, potential, effective-D, and reporting boundaries.

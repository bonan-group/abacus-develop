# GPU Review Fixes Design

## Goal

Resolve the confirmed correctness, API, and CUDA-CI findings from the review of
`codex/gpu-resident-mixing` while preserving two intentional support boundaries:

- GPU plane-wave FFT remains limited to one MPI rank per pool.
- Single/mixed-precision GPU builds require float FFTW, while the repository-wide
  `ENABLE_FLOAT_FFTW` default remains `OFF`.

## Scope

The change will address:

1. The resident-potential smooth-grid cast overrun.
2. Ambiguous multi-rank GPU USPP effective-D behavior.
3. Multi-rank CUDA Ewald-stress overcounting.
4. Explicit-spin XC dispatch using global spin state.
5. Zero-projector atom types dividing CUDA chunk sizing by zero.
6. The undefined `XC_Functional::gradcorr` convenience declaration.
7. CUDA CI not executing the new XC and psi GPU unit tests.
8. Build-profile and documentation visibility for the float-FFTW requirement.
9. ROCm gamma-only inverse FFT Hermitian reconstruction parity with CUDA.

The branch-wide global-dependency budget is recorded as a separate governance
cleanup. This correctness patch must not add new `GlobalV`, `GlobalC`, or
`PARAM` references, but removing all existing branch additions is outside this
focused repair because it would require broader interface redesign across the
optimization series.

## Design

### Smooth-grid resident potential

Use the owned smooth-potential matrix element count for the double-to-float
device cast. The count must be `veff_smooth.nr * veff_smooth.nc`, not the dense
`v_eff` size. A focused size-selection test will distinguish dense and smooth
grid sizes before the production line is changed.

### GPU USPP effective-D rank guard

Add an explicit `rho_basis->poolnproc == 1` precondition at the beginning of
`cal_effective_D_gpu`. This mirrors `PW_Basis::real2recip_gpu`, makes the support
boundary visible before allocations and kernel launches, and avoids adding an
MPI reduction to a path whose prerequisite FFT is intentionally not distributed.
The test will verify that a multi-rank layout is rejected with the documented
one-rank-per-pool diagnostic.

### CUDA Ewald stress rank ownership

Pass G=0 ownership into the CUDA Ewald operator. Every rank will still zero its
output, but only the rank with `ig0 >= 0` will initialize the scalar diagonal
term. Existing pool reduction will then reproduce the CPU ownership model.
A focused operator test will cover both owner and non-owner inputs.

### Explicit XC spin dispatch

Pass the overload's explicit `nspin` value into each resident XC selector and
use it for eligibility checks. No selector in the explicit overload may consult
`PARAM.inp.nspin`. Tests will deliberately make the explicit and global values
different and verify that dispatch follows the explicit argument without
overwriting a differently shaped matrix.

### Zero-projector CUDA chunking

Skip atom types with `nh == 0` before computing `target_chunk / nh` in both
chunked force and stress paths. Mixed local-only/nonlocal species tests will
exercise each loop.

### `gradcorr` API cleanup

Remove the unused short `gradcorr` declaration. It has no definition or caller,
and retaining it would expose a link-failing public API and a forbidden default
argument. Existing explicit overloads remain unchanged.

### ROCm gamma-only inverse FFT

Implement the existing gamma half-spectrum reconstruction operator for HIP and
use it from both real and complex ROCm inverse transforms. Reuse the CUDA
indexing and reduced-boundary rules, and run the same GPU operator tests under
either backend.

### CUDA CI and float FFTW

Extend the CUDA CTest expression to execute
`MODULE_HAMILT_XC_Functional_UTs` and `MODULE_PSI_init_gpu_test`. Add a
single-precision double-grid workflow case that reaches the resident-potential
cast.

Keep `ENABLE_FLOAT_FFTW` globally `OFF`. CUDA and ROCm CI/toolchain profiles
that advertise single/mixed precision will pass `-DENABLE_FLOAT_FFTW=ON`.
User documentation will state that requirement, and CMake will emit an
actionable warning for GPU builds configured without float FFTW rather than
making double-only GPU builds fail configuration.

## Testing Strategy

Each production fix follows a red-green cycle:

1. Add or extend the smallest relevant unit/integration test.
2. Run it against the current implementation and record the expected failure.
3. Apply the minimal production change.
4. Re-run the focused test and its neighboring suite.

Final verification will include applicable CPU unit tests, CUDA-focused tests,
the one-rank GPU mixing workflow matrix, `git diff --check`, and the agent
governance checker. GPU and MPI commands will run outside the sandbox with
`OMP_NUM_THREADS=1`.

## Expected Side Effects

- Double-only GPU builds remain free of an `fftw3f` requirement.
- Single/mixed GPU profiles acquire an explicit `fftw3f` build and runtime
  dependency, with modest added build/link time and host FFT storage.
- GPU multi-rank-per-pool execution remains unsupported and fails earlier with
  a clearer diagnostic.
- No INPUT parameter behavior changes are introduced, so no parameter metadata
  update is required.

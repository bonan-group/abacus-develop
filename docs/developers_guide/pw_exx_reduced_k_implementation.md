# Reduced-k Plane-wave Exact Exchange Implementation

## Scope

This report describes the reduced-k plane-wave exact-exchange (PW EXX)
implementation on branch `batch-exx-pr`, including the exact-exchange energy,
direct operator, ACE path, memory-defensive q-point tiling, and GPU symmetry
reconstruction. It records the mathematical choices, their relationship to
Quantum ESPRESSO (QE), the implementation boundaries, and the validation
performed for HSE calculations.

The reference point for the original implementation is `official/develop` at
commit `a76c946288fb5d4a1844caccf05f818ccb0e8011`. That implementation iterates
only stored k-points and contains an explicit warning in the direct operator:
the existing `wk_ik` weighting is "wrong for symmetry." It has no full-k/q map
and no q-tile controls.

On the RTX 5090 test host, untouched official/develop also has a separate GPU
compatibility defect: the private EXX FFT basis inherits density-basis precision
instead of the EXX template precision and can fail with CUDA `invalid argument`.
Performance comparisons therefore distinguish untouched official/develop from
a patched-official baseline containing only the three-line FFT precision fix.
That compatibility patch does not include any reduced-k, qk-map, tiling, energy,
operator, or ACE change.

## Problem

For semilocal density functionals, symmetry-equivalent k-points can be replaced
by one irreducible representative and its integration weight. Exact exchange is
nonlocal in k-space: for each target k, the Fock operator integrates over every
partner q. The pair `(k,q)`, the momentum transfer, occupations, singular
correction, and symmetry transformation of the partner wavefunction must remain
consistent.

The original reduced-k implementation treated the stored irreducible list as
the q integration grid. This loses the individual full-grid partner points and
their momentum transfers. A representative occupation weight is not sufficient
to recover those pair-dependent quantities. Consequently, reduced-k direct EXX
energy and operator application were not guaranteed to reproduce the ordered
full-k double sum.

The GPU implementation also reconstructed a symmetry-related state through a
reciprocal-space G-vector remap. Because the destination k-point has its own
finite plane-wave cutoff, that operation can impose a second k-dependent
projection. QE and the ABACUS CPU implementation instead rotate the state on
the complete FFT real-space grid, where a space-group operation is a unitary
permutation plus a Bloch phase and optional time-reversal conjugation.

## Mathematical Model

The implementation distinguishes three objects:

1. A stored irreducible representative wavefunction `psi(rep(k))`.
2. A full-grid target or partner point carrying its Cartesian/fractional k,
   full-grid weight, representative index, symmetry operation, reciprocal
   shift, and time-reversal flag.
3. An ordered qk pair connecting one full target descriptor to one full partner
   descriptor.

For a representative partner orbital, the physical occupation is recovered as

```text
occupation(q,m) = wg(rep(q),m) / wk(rep(q)).
```

The partner-grid integration weight is stored separately in the full-point
descriptor. Thus an ordered partner contribution uses

```text
[wg(rep(q),m) / wk(rep(q))] * weight(q_full).
```

The direct scalar energy is evaluated as an ordered full-pair sum and multiplied
by one half. The current correctness-first energy path explicitly enumerates
the full target star and builds an ordered qk map for each target member. It
does not fold `(k,q)` and `(q,k)` into an unordered triangular pair. This avoids
assuming exact pair symmetry across finite FFT cutoffs, symmetry phases, and
singular corrections.

The Hamiltonian action is required only at stored representative target k-points.
It integrates over every active full partner point using the same full-point
descriptors and partner weights. With a unitary symmetry transformation this is
covariant with the explicit full-target scalar sum.

QE expresses the same physics with a transfer-q loop and the maps `index_xkq`,
`index_xk`, and signed `index_sym`. ABACUS loops over absolute full partner k
points instead. On a closed uniform mesh, the two loops are related by a
target-dependent permutation. The ABACUS qk map records that ordered relation;
it is not an unordered-pair compression.

## Full-k And qk Maps

`KVectorUtils::kvec_ibz_kpoint()` now preserves a descriptor for each original
full-grid point in `K_Vectors::exx_full_k_map` and `exx_full_q_map`. Each
`ExxFullPoint` contains:

- full and representative indices;
- representative pool/local ownership;
- full fractional and Cartesian k vectors;
- normalized full-grid integration weight;
- real-space and reciprocal-space symmetry matrices;
- fractional translation;
- time-reversal, conjugate-only, identity, and active flags.

The map is built after the final max-norm irreducible representative has been
selected, so wavefunction basis construction and EXX symmetry reconstruction
refer to the same representative. The map is broadcast with the k-point data
and remains available to both operator and energy paths.

The energy policy in `op_pw_exx.h` selects representative targets, enumerates
their full star members, and creates an ordered `ExxQkPair` entry for every
active partner. No `unordered_pair_multiplier` is used for the final direct
energy quadrature.

## Descriptor-aware Coulomb Kernel

The EXX Coulomb potential is generated from full target and partner descriptors,
not only local irreducible integer indices. Therefore the momentum difference,
HSE screened kernel, and singular correction use the physical full-grid
`k-q` pair even though the underlying wavefunction is stored at an irreducible
representative.

Potential caching is scoped to a target calculation and keyed by full target
and partner indices. The cache is reset between target scopes instead of being
allowed to grow across the complete ordered double sum.

## Memory-defensive q Tiling

The direct operator, direct energy, and ACE construction share the q-tile path.
Two independent controls bound the workspace:

- `exx_q_tile_size`: number of full partner points reconstructed together;
- `exx_band_tile_size`: number of partner bands reconstructed together.

For FFT-grid size `Nr`, the dominant partner-state workspace scales as

```text
O(exx_q_tile_size * exx_band_tile_size * Nr),
```

not `O(Nq_full * Nband * Nr)`. The workspace is allocated once and reused.
Potential cache lifetime is also bounded by the current target scope. This is
the memory-defensive strategy inherited from the `abacus-cider` work: recompute
or reload bounded tiles rather than retain every full-q real-space state.

`exx_batch_fft_size` controls FFT batching independently of the q and band
tiles. Changing a tile size changes execution granularity but not the ordered
quadrature or its weights.

## GPU Real-space Symmetry Reconstruction

The final GPU reconstruction is implemented by
`exx_rotate_realspace_op<T, DEVICE_GPU>` in
`source/source_pw/module_pwdft/kernels/cuda/exx_q_state_op.cu`.

For each representative state:

1. Inverse FFT the representative reciprocal coefficients on the EXX FFT grid.
2. For each destination grid point, map the integer FFT coordinate through the
   stored space-group operation and fractional translation.
3. Gather the corresponding representative-grid sample.
4. Apply the Bloch phase

   ```text
   exp[i 2*pi * (k_rep . r_rep - k_full . source)].
   ```

5. For time reversal, use `source = -r` and conjugate the gathered value.
6. Write the full-point real-space state directly into the bounded q-tile
   workspace.

The kernel supports batched states and follows the ABACUS convention
`r_rep = source * gmatrix + gtrans`. Both scalar and batched q-state loading use
the same operation. There is no forward transform through a second full-k
plane-wave cutoff, so the symmetry action remains a permutation of the complete
FFT grid, matching QE and the existing CPU semantic reference.

Identity and conjugate-only cases retain their cheaper specialized paths. GPU
PW EXX currently requires `poolnproc=1`, consistent with the surrounding GPU
PW FFT limitation.

## ACE

ACE construction was not changed to average target stars. For each irreducible
target k, ABACUS applies the representative direct Fock operator to the projector
orbitals and builds one ACE block, matching QE's organization.

Two invariants were measured during debugging:

- projected Fock Hermiticity:
  `||Phi^dagger Vx Phi - (Phi^dagger Vx Phi)^dagger|| / ||Phi^dagger Vx Phi||`
  was approximately `1e-16`;
- ACE interpolation:
  `||V_ACE Phi - V_direct Phi|| / ||V_direct Phi||` was approximately `1e-15`.

These results ruled out star-averaged ACE projectors and an ACE factorization
change. An earlier apparent ACE mismatch came from unconverged 30 Ry outer-loop
trajectories and a temporary reduced-only convergence diagnostic; that
diagnostic was removed.

## Tests

Focused CPU tests cover:

- full-point map and ordered qk policy;
- ordinary, translated, reciprocal-shift, and time-reversal transformations;
- ABACUS's fractional-translation convention;
- forward/adjoint transformation consistency.

Focused CUDA tests compare batched ordinary and time-reversal rotations against
the CPU implementation component by component at `1e-12` tolerance. The final
test results were 13/13 focused CPU tests and 2/2 CUDA rotation tests.

## GPU Validation

All production calculations used a LibXC-enabled CUDA Release build,
`OMP_NUM_THREADS=1`, HSE, and `ecutwfc=60 Ry` on an RTX 5090.

For primitive GaAs on a 2x2x2 mesh:

- ACE reduced/full differences were `3.20e-7 eV` in total energy and
  `2.35e-7 eV` in EXX energy;
- direct no-ACE reduced/full differences were `3.94e-6 eV` in total energy and
  `1.56e-5 eV` in EXX energy;
- reduced-k wall-time speedups were `1.93x` for ACE and `1.71x` for direct
  no-ACE.

For a matched-work 4x4x4 ACE calculation, 64 full points reduced to eight
representatives. Three repetitions gave:

- median wall time `633.43 s` full and `82.81 s` reduced (`7.65x` speedup);
- median ACE construction `604.82 s` full and `77.39 s` reduced (`7.82x`);
- ABACUS-reported host memory `62.99 MB` full and `11.64 MB` reduced (`5.41x`);
- identical reported q-state GPU allocation of `10.18 MB`;
- reduced/full sixth-update differences of `2.89e-8 eV` total and
  `1.68e-7 eV` EXX.

A 6x6x6 reduced-k pilot used 16 representatives for 216 full points. Reported
host memory was `19.02 MB`, while q-state GPU allocation remained `10.18 MB`.
This confirms that the q-state workspace is bounded by tile dimensions rather
than the full k-grid size.

### Post-review hardening

The direct scalar-energy path has a second memory component beyond the q-state
workspace: Coulomb potentials cached by `(full_k, full_q)`. The reduced-energy
loop now clears this cache after every q tile and enforces a peak entry count of
`star_size * q_count`. Potentials remain reusable across all source and target
bands inside the tile. The Hamiltonian-application cache retains its existing
scope because its target-band-first loop revisits q points and benefits from
that reuse without the full-star multiplier.

Additional edge-case hardening includes:

- pool ownership and KPAR values are read from the explicitly supplied
  `K_Vectors::para_k` state; the pool-local exact-energy loop performs no world
  collectives;
- CPU and CUDA share the same FFT-grid symmetry compatibility check;
- zero-raw-weight stars receive equal active-member fallback weights;
- zero full-point weights are skipped before representative-weight division;
- batch allocation products are checked in `size_t`, and CUDA rotation and
  conjugation use 64-bit grid-stride indexing;
- ROCm has explicit link definitions that report the unsupported q-state
  operation until a validated HIP implementation is available.

Final post-review tests passed 42 k-list, 16 EXX policy/radial, 15 CPU symmetry,
and 2 CUDA symmetry cases. Final GaAs 2x2x2 HSE GPU results were
`-1944.990717307951 eV` for reduced ACE and `-1944.990701496733 eV` for tightly
converged direct q2. The latter differs from the accepted direct q4 result by
`5.63e-8 eV` total and `1.16e-7 eV` EXX.

## official/develop A/B

The untouched reference was `official/develop` commit
`a76c946288fb5d4a1844caccf05f818ccb0e8011`, rebuilt as a Release PW CUDA
executable with LibXC and the same CUDA architecture list as the current build.
Both binaries report ABACUS `v3.11.0-beta.5`. Inputs were identical after
removing the three current-only tile controls and output suffix.

The reference has no production reduced-k PW EXX mode. Its input conversion
emits `EXX PW works only with symmetry=-1` and changes a requested
`symmetry=1` run to full k. This was confirmed by the 2x2x2 GaAs cases: both
official symmetry settings used eight k points and produced identical energies.
The comparison therefore uses official full k as the physics reference and as
the production fallback timing for requested symmetry reduction.

For converged 2x2x2 ACE, current full k matched official total energy within
`1.4e-11 eV`; current reduced k matched official within `3.20e-7 eV` total and
`2.35e-7 eV` EXX. Tightly converged direct no-ACE comparisons used
`scf_thr=1e-10` and `pw_diag_thr=1e-10`: current full and reduced total energies
were within `7.88e-6 eV` and `8.92e-6 eV` of official, respectively.

Three-repeat equivalent full-k medians show that the current implementation is
also faster before applying symmetry reduction:

- ACE wall time: `11.51 s` official and `9.58 s` current (`1.20x`);
- direct wall time: `152.38 s` official and `34.61 s` current (`4.40x`);
- direct operator time: `111.60 s` official and `21.54 s` current (`5.18x`).

For the fixed-work 4x4x4 ACE case, current full-k wall time was `633.43 s`
versus `761.05 s` official (`1.20x`). Current reduced k used eight
representatives and took `82.81 s`, a `9.19x` speedup over the official full-k
fallback. The fixed-work current reduced and official energies differed by
`1.46e-7 eV` total and `1.67e-7 eV` EXX.

Current full-k bookkeeping increases ABACUS-reported host memory: `62.99 MB`
versus `33.04 MB` official on 4x4x4. In the intended reduced mode it falls to
`11.64 MB`, `2.84x` below the official fallback. Reported GPU allocation stays
near `10 MB`. Full raw outputs, hashes, parsed rows, and derived ratios are in
`validation_artifacts/exx_official_develop_ab/`.

## Performance Interpretation

The real-space rotation kernel averaged `7.174 us` over 2,000 small-grid test
launches, with `0.022 us` standard deviation. Production timing is dominated by
ordered Fock pair work, not the launch overhead.

The 4x4 reduced calculation approaches the ideal eightfold speedup because the
number of target representatives falls from 64 to eight while the full partner
quadrature is preserved. The final direct no-ACE scalar energy evaluation does
not receive the same target reduction because it explicitly evaluates the full
ordered target/partner sum; on the 2x2 case that stage took `9.76 s` full and
`9.62 s` reduced. Most no-ACE end-to-end speedup comes from repeated
representative-target Hamiltonian applications.

## Limitations And Follow-up

- GPU PW EXX supports `KPAR=1`/`poolnproc=1` in this implementation milestone.
- ROCm does not yet implement the q-tile GPU path.
- The direct scalar energy favors exact full-pair equivalence over target-star
  compression; this is deliberately more expensive than the representative
  operator path.
- A complete 6x6 q-tile-size sweep remains useful for tuning defaults, but does
  not affect the correctness of the ordered sum.
- The unfiltered `pw_test_gpu` executable has a pre-existing CUDA
  `invalid argument` failure in `PW_BASIS_C2R_GPU_TEST/0.Mixing`; the focused
  EXX CUDA tests pass.

## Explicitly Rejected Designs

- Do not build ACE projectors from a star-averaged target Fock action. QE does
  not do this, and the ACE interpolation invariant already passes.
- Do not triangularize the direct energy into unordered k/q pairs unless every
  discrete phase, cutoff, singular correction, and symmetry covariance is
  proven pair-symmetric.
- Do not reconstruct GPU full-point states through a second finite full-k
  reciprocal cutoff. Use the unitary real-space FFT-grid transformation.

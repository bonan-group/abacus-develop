# USPP Device-Built QGM Cache Design

## Goal

Reduce PW GPU startup time and host memory use by constructing the persistent
USPP augmentation-function cache (`qgm`) and atom translation phases
(`qgm_phase`) directly on the GPU.

The optimized path must preserve the existing reciprocal-space layouts and
consumer interfaces so augmentation density, effective-D assembly, and USPP
forces continue to use persistent device-resident arrays without per-SCF host
materialization or transfers.

## Context

For each atom type and symmetric projector pair, ABACUS stores the Fourier
transform of the USPP augmentation function on the dense reciprocal grid:

```text
qgm[type, pair, G] = Q_type,pair(G)
```

For each atom, it separately stores the translation factor:

```text
qgm_phase[atom, G] = exp(-i 2 pi G . tau_atom)
```

Their product gives the augmentation function centered on an atom. The current
GPU setup first creates both full arrays on the host, then allocates and copies
device versions. In the 256-atom silicon benchmark, the complete nonlocal setup
takes about 13.25 seconds and temporarily requires both host and device storage.

## Scope

This slice will:

- retain host construction for CPU calculations;
- retain the compact host `qrad` radial interpolation table;
- build double-precision `z_qgm` directly on the GPU;
- build double-precision `z_qgm_phase` directly on the GPU;
- derive float caches from the double device caches when mixed precision is
  active;
- avoid allocating the full host `qgm` and `qgm_phase` arrays on the GPU path;
- preserve the current persistent device layouts and getter interfaces;
- update cache-availability checks so they do not depend on host-array size.

This slice will not:

- make `qgm` or phases on demand during every SCF operation;
- reduce persistent device memory through chunked consumers;
- move construction of the stress-only derivative cache `dqgm` to the GPU;
- change INPUT parameters, numerical formulas, or CPU behavior.

## Architecture

### Cache ownership

`pseudopot_cell_vnl` remains the owner of all augmentation caches. On the GPU
path, `z_qgm` and `z_qgm_phase` become the authoritative copies. Host `qgm` and
`qgm_phase` remain empty unless a host-only path explicitly requires them.

The existing getters continue to return the active precision's device pointer.
A cache-availability query will represent whether augmentation data exists,
instead of using `qgm.getSize()` as an indirect signal.

### Static host metadata

The CPU setup will pack the small metadata needed by the bulk GPU builder:

- projector-pair radial index;
- projector-pair angular indices;
- nonzero spherical-harmonic indices and coefficients;
- `qrad` dimensions and interpolation spacing;
- atom type offsets and projector-pair counts.

This metadata is proportional to atom types and projector channels, not to the
dense reciprocal grid or atom count. It may be uploaded as temporary setup
buffers and released after cache construction.

### Device construction flow

1. Allocate and upload dense reciprocal vectors once. Reuse this allocation as
   the persistent `qgm_gcar` cache.
2. Compute `|G| * tpiba` on the GPU.
3. Compute real spherical harmonics on the GPU using the existing device Ylm
   implementation.
4. Launch a bulk QGM builder over `(type, symmetric pair, G)`. Each output
   element performs the same polynomial interpolation and finite angular sum as
   the host `radial_fft_q` implementation.
5. Upload fractional atom coordinates and launch the existing atom-phase kernel
   over `(atom, G)` to fill `z_qgm_phase`.
6. When float data is required, cast the completed double device arrays into
   `c_qgm` and `c_qgm_phase` on device.
7. Release setup-only metadata, norms, spherical harmonics, and atom-coordinate
   buffers.

No full `qgm` or phase array crosses the host/device boundary.

### Stress boundary

When USPP stress preparation is requested, the existing host `dqgm` construction
remains active and its result is copied to `z_dqgm`. It may reuse host norms and
spherical harmonics, but it must not force host construction of `qgm` or
`qgm_phase`.

Moving `dqgm` construction to the device is a separate follow-up because its
Cartesian derivative formula and validation surface are larger.

## Compatibility And Failure Handling

- The CPU path remains byte-for-byte structurally unchanged.
- CUDA and ROCm receive equivalent bulk-builder interfaces and implementations.
- The implementation remains compatible with C++11.
- Existing cache pointers remain null for non-USPP calculations.
- Allocation or kernel failures use the repository's existing device error
  handling; setup must not silently fall back to allocating multi-gigabyte host
  caches.
- Unit-cell or grid reinitialization destroys and rebuilds the device caches as
  it does today.

## Testing

### Kernel tests

- Compare CPU and GPU bulk QGM generation for multiple atom types, projector
  pairs, angular channels, and nonuniform G norms.
- Compare GPU atom phases with the existing independent CPU reference.
- Cover G=0, zero angular coefficients, and both float and double output caches.

The bulk-QGM test must fail before the new device builder exists.

### Runtime validation

- Rebuild the CUDA PW executable and focused kernel targets.
- Run a small USPP silicon calculation and compare total energy, eigenvalues,
  augmentation charge, effective-D values, and forces against the host-built
  reference within existing precision tolerances.
- Run an unchanged USPP stress case to confirm the retained `dqgm` path.
- Run compute-sanitizer memcheck on the small GPU case.
- Confirm memory reporting no longer includes full host `qgm` or
  `qgm_phase` allocations on the GPU path.

### Performance acceptance

After confirming the GPU is idle, rerun the 256-atom silicon case with 30 Ry
wavefunction cutoff, 120 Ry density cutoff, and a 2x2x2 k-point mesh.

Acceptance requires:

- numerical parity with the host-built cache path;
- a material reduction from the previous 13.25-second `init_vnl` time;
- no regression in the electronic iteration or force timings;
- elimination of full host `qgm` and `qgm_phase` materialization for GPU runs.

## Risks

- Polynomial interpolation indexing must exactly match the host implementation;
  a small indexing difference can perturb augmentation charge globally.
- Mixed precision still needs double caches for effective-D and force assembly.
- Host code currently uses `qgm.getSize()` as an availability proxy; every such
  check must be replaced before host arrays can remain empty.
- Device construction reduces startup and host memory but intentionally does not
  reduce persistent GPU memory. On-demand chunking remains a possible later
  optimization if GPU capacity becomes the limiting factor.

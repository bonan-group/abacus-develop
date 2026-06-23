# Building and Running ABACUS-CIDER

This directory is a vendored ABACUS tree inside CiderPress. It is not a plain upstream checkout: it includes CIDER XC integration and EXX training-data dump support.

The commands below assume you are starting in:

```bash
cd /home/bonan/appdir/CiderPress/abacus-cider
```

## Runtime Environment

Use the parent CiderPress Python environment for CIDER-enabled runs, so the embedded Python interpreter can import `ciderpress`:

```bash
source ../.venv/bin/activate
```

For all ABACUS tests and small validation runs, keep OpenMP fixed:

```bash
export OMP_NUM_THREADS=1
```

When using the local ELPA and CiderPress bridge libraries, set:

```bash
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
export PKG_CONFIG_PATH=/home/bonan/appdir/elpa-2026.02.001/lib/pkgconfig:${PKG_CONFIG_PATH}
```

Run ABACUS runtime tests, MPI tests, integration tests, and GPU tests outside the sandbox. MPI launch, process visibility, local sockets, and GPU discovery can otherwise be unreliable.

## Build CiderPress Bridge

CIDER-enabled ABACUS needs the CiderPress bridge library and headers from the parent repository, normally under `../ciderpress/lib`.

If the bridge is missing or stale:

```bash
cd /home/bonan/appdir/CiderPress/ciderpress/lib
cmake -B build -DBUILD_WITH_MPI=OFF -DBUILD_FFTW=ON -DBUILD_LIBXC=ON
cmake --build build -j$(nproc)
```

Return to ABACUS:

```bash
cd /home/bonan/appdir/CiderPress/abacus-cider
```

## Full CPU Build With ELPA, LibRI, Libxc, and CIDER

This build produces the general MPI CPU binary used for most tests and CIDER training-data generation:

```bash
cmake -S . -B /tmp/abacus-cider-elpa-libri-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx \
  -DENABLE_MPI=ON \
  -DUSE_OPENMP=ON \
  -DENABLE_LCAO=ON \
  -DENABLE_LIBRI=ON \
  -DUSE_ELPA=ON \
  -DELPA_DIR=/home/bonan/appdir/elpa-2026.02.001 \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -DCIDERPRESS_DIR=/home/bonan/appdir/CiderPress \
  -DBUILD_TESTING=ON

cmake --build /tmp/abacus-cider-elpa-libri-build -j$(nproc)
```

Main binary:

```bash
/tmp/abacus-cider-elpa-libri-build/abacus_std_para
```

Quick binary check:

```bash
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
/tmp/abacus-cider-elpa-libri-build/abacus_std_para --help
```

## PW CUDA Build for GPU EXX Paths

Use this build when you need the PW CUDA binary and the upstream batched FFT, q-tile, and q-cache optimized EXX path. GPU execution requires a working NVIDIA driver visible to the runtime.

```bash
cmake -S . -B /tmp/abacus-cider-pw-gpu-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/mpicxx \
  -DUSE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DENABLE_MPI=ON \
  -DUSE_OPENMP=ON \
  -DENABLE_LCAO=OFF \
  -DENABLE_LIBRI=OFF \
  -DUSE_ELPA=OFF \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -DCIDERPRESS_DIR=/home/bonan/appdir/CiderPress

cmake --build /tmp/abacus-cider-pw-gpu-build -j$(nproc)
```

Main binary:

```bash
/tmp/abacus-cider-pw-gpu-build/abacus_pw_gpu
```

Before running GPU cases, verify the driver is visible:

```bash
nvidia-smi
```

If `nvidia-smi` cannot communicate with the driver, GPU ABACUS tests will fail even if the CUDA build itself succeeds.

## Running the Compiled Binary

Run from an ABACUS input directory containing `INPUT`, `STRU`, and `KPT`:

```bash
cd /path/to/case
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
/tmp/abacus-cider-elpa-libri-build/abacus_std_para
```

For MPI runs:

```bash
cd /path/to/case
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
mpirun -np 4 /tmp/abacus-cider-elpa-libri-build/abacus_std_para
```

For GPU runs, use the GPU binary and set `device gpu` in `INPUT`:

```bash
cd /path/to/gpu-case
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
mpirun -np 1 /tmp/abacus-cider-pw-gpu-build/abacus_pw_gpu
```

## Generating PW Training Data

Training-data generation is enabled by `out_training_data 1`. The current workflow writes a PW NLDX/CIDER training record and evaluates a one-shot PBE0 full-range EXX label on the converged PBE density and PW wavefunctions.

Minimum relevant `INPUT` settings:

```text
calculation       scf
basis_type        pw
dft_functional    PBE
device            cpu
precision         double

out_training_data 1
out_wfc_pw        2
out_band          1
```

Current restrictions:

- `basis_type` must be `pw`.
- `dft_functional` must be `PBE` for the PBE-base EXX label workflow.
- `nspin` must be `1` or `2`.
- `kpar` must be `1`.
- `bndpar` must be `1`.
- Pseudopotentials must be norm-conserving.
- `ENABLE_LIBXC=ON` is required.

The example inputs are:

```bash
examples/training/si_pw/primitive/pbe_base
examples/training/si_pw/shaken_2x2x2/pbe_base
```

### Smoke Test: Primitive Si Training Dump

This keeps generated files out of the worktree:

```bash
rm -rf /tmp/abacus-training-dump-smoke
mkdir -p /tmp/abacus-training-dump-smoke
cp examples/training/si_pw/primitive/pbe_base/INPUT /tmp/abacus-training-dump-smoke/
cp examples/training/si_pw/primitive/pbe_base/KPT /tmp/abacus-training-dump-smoke/
cp examples/training/si_pw/primitive/pbe_base/STRU /tmp/abacus-training-dump-smoke/
sed -i 's#pseudo_dir        ../../../../../tests/PP_ORB#pseudo_dir        /home/bonan/appdir/CiderPress/abacus-cider/tests/PP_ORB#' \
  /tmp/abacus-training-dump-smoke/INPUT

cd /tmp/abacus-training-dump-smoke
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
/tmp/abacus-cider-elpa-libri-build/abacus_std_para
```

Successful output includes:

```text
training_pbe0_exx_label: hybrid-scaled EXX energy = ... Ha
write_training_dump: wrote OUT.TRAIN_PRIM/training_dump/step_0
```

The dump directory should contain:

```text
OUT.TRAIN_PRIM/training_dump/step_0/record.json
OUT.TRAIN_PRIM/training_dump/step_0/rho_valence_sg.npy
OUT.TRAIN_PRIM/training_dump/step_0/rho_sg.npy
OUT.TRAIN_PRIM/training_dump/step_0/rho_core_g.npy
OUT.TRAIN_PRIM/training_dump/step_0/sigma_xg.npy
OUT.TRAIN_PRIM/training_dump/step_0/ekb_kb.npy
OUT.TRAIN_PRIM/training_dump/step_0/occ_kb.npy
OUT.TRAIN_PRIM/training_dump/step_0/kpt_cart_kv.npy
OUT.TRAIN_PRIM/training_dump/step_0/kpt_direct_kv.npy
OUT.TRAIN_PRIM/training_dump/step_0/wk_k.npy
OUT.TRAIN_PRIM/training_dump/step_0/kspin_k.npy
```

`record.json` records units explicitly. Energies and EXX labels are in Hartree:

```json
{
  "energy_unit": "Ha",
  "exx_label": {
    "present": true,
    "energy_Ha": -0.5326364970338882,
    "hybrid_scaled": true,
    "hybrid_alpha": 0.25
  }
}
```

The exact energy value depends on the input and build, but the unit key and `energy_Ha` field should be present.

## Running Internal Tests

Build with `BUILD_TESTING=ON`, then run CTest outside the sandbox:

```bash
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
ctest --test-dir /tmp/abacus-cider-elpa-libri-build --output-on-failure -j 4
```

Focused EXX test:

```bash
export OMP_NUM_THREADS=1
export LD_LIBRARY_PATH=/home/bonan/appdir/elpa-2026.02.001/lib:/home/bonan/appdir/CiderPress/ciderpress/lib:${LD_LIBRARY_PATH}
ctest --test-dir /tmp/abacus-cider-elpa-libri-build -R '^08_EXX$' --output-on-failure -j 1
```

Known environment-sensitive cases:

- GPU suites fail if the NVIDIA driver is not visible, even when CUDA compilation succeeds.
- Some OFDFT ML-KEDF tests require `ENABLE_MLALGO=ON`.
- Single-precision PW CPU cases require `ENABLE_FLOAT_FFTW=ON`.

# AGENTS.md

These instructions apply inside `abacus-cider/`.

## Project

- This directory is a vendored ABACUS tree inside the CiderPress repository.
- It is not a plain upstream checkout: this copy includes CIDER NLDF exchange-correlation integration.
- Keep ABACUS-style C++/CMake conventions when editing this tree, and avoid broad upstream refactors unless the task requires them.
- Online ABACUS documentation lives at https://abacus.deepmodeling.com/.

## Environment

- The parent CiderPress virtual environment is at `../.venv/` and is managed by `uv`.
- Activate it, or use `../.venv/bin/python3`, before running CIDER-enabled ABACUS examples so the embedded Python interpreter can import `ciderpress`.
- Prefer `uv` for Python package management from the parent repository. Use `uv pip install <package>` to add dependencies, and run `uv pip` commands outside the sandbox so the uv cache can be used.
- CIDER-enabled builds need the CiderPress bridge library and headers from the parent tree, normally under `../ciderpress/lib/`.

## Build

ABACUS is built with CMake. Common options in this checkout include:

- `ENABLE_MPI=ON` by default
- `USE_OPENMP=ON` by default
- `ENABLE_LCAO=ON` by default
- `USE_ELPA=ON` by default when MPI is enabled
- `ENABLE_LIBXC=OFF` by default
- `ENABLE_CIDER=OFF` by default; requires `ENABLE_LIBXC=ON`
- `ENABLE_MLALGO=OFF` by default
- `ENABLE_LIBRI=OFF` by default
- `USE_CUDA=OFF`, `USE_ROCM=OFF`, `USE_DSP=OFF` by default
- `BUILD_TESTING=OFF` by default
- `ENABLE_ASAN=OFF`, `ENABLE_COVERAGE=OFF` by default

Basic CMake build:

```bash
cmake -B build -DENABLE_MPI=ON -DENABLE_LCAO=ON
cmake --build build -j$(nproc)
```

CIDER-enabled build from `abacus-cider/`:

```bash
source ../.venv/bin/activate
cmake -B build \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -DCIDERPRESS_DIR=..
cmake --build build -j$(nproc)
```

If CMake cannot find the bridge, point it directly at the CiderPress library directory:

```bash
cmake -B build \
  -DENABLE_LIBXC=ON \
  -DENABLE_CIDER=ON \
  -DCIDERPRESS_LIB_DIR=../ciderpress/lib
```

If the CiderPress bridge libraries are missing, build them from the parent repository first:

```bash
cd ../ciderpress/lib
cmake -B build -DBUILD_WITH_MPI=OFF -DBUILD_FFTW=ON -DBUILD_LIBXC=ON
cmake --build build -j$(nproc)
```

The toolchain flow is still available for dependency-heavy builds:

```bash
cd toolchain
./toolchain_gnu.sh
./build_abacus_gnu.sh
source install/setup
source abacus_env.sh
```

Other toolchain variants present here include `toolchain_intel.sh`, `toolchain_gcc-aocl.sh`, and `toolchain_aocc-aocl.sh`.

## Executables

This CMake configuration names ABACUS binaries by enabled feature class, not by the simple upstream aliases. Examples:

- `abacus_1p`: PW-only MPI CPU build
- `abacus_1s`: PW-only serial CPU build
- `abacus_2p`: LCAO/PW MPI CPU build without LibRI or MLALGO
- `abacus_2s`: LCAO/PW serial CPU build without LibRI or MLALGO
- `abacus_2g`: LCAO/PW MPI CUDA build without LibRI or MLALGO
- `abacus_3*`, `abacus_4*`, `abacus_5*`: LibRI and/or MLALGO variants

The current local `build/` directory contains `abacus_1p` and `abacus_2p`.

## CIDER Integration

CIDER support is wired through:

- CMake option: `ENABLE_CIDER`
- Bridge library/header lookup: `libciderbridge` and `cider_bridge.h`
- Potential implementation: `source/source_estate/module_pot/pot_cider_xc.*`
- PW Hamiltonian registration: `source/source_pw/module_pwdft/hamilt_pw.cpp`
- Input parameters: `cider_model`, `cider_xmix`, and `cider_tf_tau`

Current CIDER restrictions enforced by input conversion:

- `basis_type` must be `pw`
- `nspin` must be `1` or `2`
- `cider_xmix` must be in `[0, 1]`
- Do not combine `cider_model` with native hybrid/EXX `dft_functional` settings
- MGGA CIDER models request kinetic-energy density and may need NLCC-aware handling

CIDER examples are under:

- `examples/02_scf/pw_Si2-cider*`
- `examples/bandgap/pw_*`

The parent repository also documents a quick smoke test:

```bash
source ../.venv/bin/activate
cd examples/02_scf/pw_Si2-cider
export OMP_NUM_THREADS=1
./abacus
```

Use the executable or wrapper that exists in the example/build you are testing.

## Testing

Always run ABACUS internal tests with `OMP_NUM_THREADS=1` so MPI test cases do not oversubscribe the machine and results are comparable to the reference data.
Run ABACUS runtime tests, MPI tests, integration tests, and any test where process visibility matters outside the sandbox using an escalated command. This avoids sandbox interference with MPI launch, local sockets, process inspection, and runtime behavior.

Unit tests are CMake/CTest-based when configured with `BUILD_TESTING=ON`:

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build -j$(nproc)
cd build
export OMP_NUM_THREADS=1
ctest -N
ctest -R cell -V
```

Integration tests are organized in `tests/`:

- `01_PW`: PW basis
- `02_NAO_Gamma`: NAO gamma-only
- `03_NAO_multik`: NAO k-point
- `04_FF`: force fields
- `05_rtTDDFT`: real-time TDDFT
- `06_SDFT`: stochastic DFT
- `07_OFDFT`: orbital-free DFT
- `08_EXX`: hybrid and LR-TDDFT
- `09_DeePKS`: DeePKS
- `10_others`: miscellaneous
- `11_PW_GPU`, `12_NAO_Gamma_GPU`, `13_NAO_multik_GPU`, `15_rtTDDFT_GPU`, `16_SDFT_GPU`: GPU tests
- `PP_ORB`: pseudopotentials and numerical orbitals used by tests
- `integrate`: integration-test scripts
- `libxc`, `performance`: specialized examples/tests

Common integration commands:

```bash
cd tests/integrate
export OMP_NUM_THREADS=1
bash Autotest.sh
```

```bash
cd tests/01_PW/101_PW_MD_1O
export OMP_NUM_THREADS=1
bash ../../integrate/Single.sh
```

Check `tests/README` before changing test layout or naming.

## Source Layout

Important directories in `source/`:

- `source_base`: math, containers, device abstraction, FFT/grid/mixing utilities
- `source_basis`: AO, NAO, and PW basis modules
- `source_cell`: cell, symmetry, and neighbor handling
- `source_estate`: charge density, electronic state, and potentials
- `source_hamilt`: Hamiltonian, XC, VDW, and surface chemistry modules
- `source_hsolver`: eigensolvers including genelpa and PEXSI hooks
- `source_pw`: PW DFT, OFDFT, and stochastic DFT
- `source_lcao`: LCAO modules including gint, hcontainer, RI, LR/RT, DeePKS, DFT+U, and RDMFT
- `source_io`: input/output and parameter handling
- `source_md`, `source_relax`, `source_esolver`, `source_psi`, `source_main`

GPU/device kernels follow the local pattern:

- CPU kernels: `kernels/*.cpp`
- CUDA kernels: `kernels/cuda/*.cu`
- ROCm kernels: `kernels/rocm/*.hip.cu`

## Development Workflow

- Use existing module boundaries and CMake target patterns.
- Keep CIDER changes focused on the bridge, input validation, potential registration, and PW paths unless broader behavior is explicitly requested.
- Formatting configuration is present in `.clang-format`, `.clang-tidy`, and `.pre-commit-config.yaml`.
- CI workflow files live in `.github/workflows/`, including CMake build tests, integration tests, CUDA, coverage, pytest, Doxygen, and performance workflows.
- The Python package under `python/pyabacus/` uses `scikit-build-core`, `pybind11`, and pytest settings in its own `pyproject.toml`.

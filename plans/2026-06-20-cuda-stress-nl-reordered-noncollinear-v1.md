# CUDA Noncollinear Stress NL Reordered Contraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the CUDA reordered nonlocal stress contraction from scalar `npol == 1` to noncollinear `npol == 2` PW calculations.

**Architecture:** Keep the existing chunked CUDA stress infrastructure. Add a noncollinear `R_chunk` builder that folds the four `deeq_nc` spin blocks and spin-diagonal `ekb * qq_nt` terms into two spinor-weighted projector coefficient columns per band, then reuse the existing `Psi * R^H` GEMM and derivative-dot stress reducer.

**Tech Stack:** ABACUS PW CUDA backend, cuBLAS GEMM through `ModuleBase::gemm_op`, existing `FS_Nonlocal_tools` chunk buffers, GoogleTest CUDA kernel tests.

## Global Constraints

- CUDA first; CPU behavior unchanged.
- Scope is `npol == 2` / noncollinear chunked CUDA stress only.
- Preserve current noncollinear math, including `deeq_nc` block order: `0=up/up`, `1=up/down`, `2=down/up`, `3=down/down`.
- Preserve current `qq_nt` behavior: apply `-ekb * qq_nt` only to spin-diagonal blocks 0 and 3.
- Do not change SCF math, solver selection, convergence criteria, input syntax, or CPU paths.
- Run ABACUS runtime tests and CUDA runtime tests outside the sandbox with `OMP_NUM_THREADS=1`.

---

## File Structure

- Modify `source/source_pw/module_pwdft/kernels/stress_op.h`
  - Add a CUDA noncollinear overload to `build_stress_nl_reordered_r_op`.
- Modify `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu`
  - Add the noncollinear `R_chunk` builder kernel and overload implementation.
- Modify `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp`
  - Use reordered contraction for `npol == 2` in `cal_stress_chunked`.
- Modify `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp`
  - Add a deterministic CUDA unit test comparing reordered noncollinear contraction against the old `dbecp` formula.

---

## Task 1: Add Failing Noncollinear Reordered Stress Test

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp`

**Interfaces:**
- Consumes existing scalar interfaces:
  - `build_stress_nl_reordered_r_op<double, base_device::DEVICE_GPU>().chunk(...)`
  - `cal_stress_nl_reordered_op<double, base_device::DEVICE_GPU>().chunk(...)`
- Produces a failing test that requires the new noncollinear `R` builder overload.

- [ ] **Step 1: Add test fixture**

Add a CUDA test named:

```cpp
TEST(TestSrcPWStressMultiDevice, cal_stress_nl_reordered_chunk_nc_gpu)
```

Use:

```cpp
constexpr int chunk_nkb = 4;
constexpr int nproj = 2;
constexpr int atom_count = 2;
constexpr int atom_start = 1;
constexpr int atom_offset_in_type = 1;
constexpr int nbands_occ = 3;
constexpr int npol = 2;
constexpr int npw = 5;
constexpr int deeq_2 = 4;
constexpr int deeq_3 = 2;
constexpr int deeq_4 = 2;
constexpr int it = 0;
constexpr int ipol = 1;
constexpr int jpol = 0;
```

Create deterministic complex arrays:

- `becp` layout: `(ib * npol + spinor) * chunk_nkb + iproj`
- `psi` layout: `(ib * npol + spinor) * npw + ig`
- `vkb_deri` layout: `iproj * npw + ig`
- `deeq_nc` layout: `((block * deeq_2 + iat) * deeq_3 + ip1) * deeq_4 + ip2`

- [ ] **Step 2: Add CPU reference using old formula**

Compute:

```cpp
dbecp[(ib * npol + is) * chunk_nkb + p] =
    sum_ig conj(vkb_deri[p * npw + ig]) * psi[(ib * npol + is) * npw + ig]
```

Then for each band, atom, `ip1`, `ip2`:

```cpp
ps0 = deeq_nc[0,iat,ip1,ip2] - ekb[ib] * qq_nt[it,ip1,ip2]
ps1 = deeq_nc[1,iat,ip1,ip2]
ps2 = deeq_nc[2,iat,ip1,ip2]
ps3 = deeq_nc[3,iat,ip1,ip2] - ekb[ib] * qq_nt[it,ip1,ip2]

stress[ipol * 3 + jpol] -= wg[ib] * real(
    ps0 * conj(dbecp_up[p1]) * becp_up[p2]
  + ps1 * conj(dbecp_up[p1]) * becp_down[p2]
  + ps2 * conj(dbecp_down[p1]) * becp_up[p2]
  + ps3 * conj(dbecp_down[p1]) * becp_down[p2]);
```

- [ ] **Step 3: Call the desired new GPU API**

The test should call:

```cpp
hamilt::build_stress_nl_reordered_r_op<double, base_device::DEVICE_GPU>().chunk(
    gpu_ctx,
    chunk_nkb,
    nbands_occ,
    deeq_2,
    deeq_3,
    deeq_4,
    it,
    atom_start,
    atom_offset_in_type,
    atom_count,
    nproj,
    d_wg,
    true,
    d_ekb,
    d_qq_nt,
    d_deeq_nc,
    d_becp,
    d_r);
```

Then:

```cpp
ModuleBase::gemm_op<std::complex<double>, base_device::DEVICE_GPU>()(
    'N', 'C', npw, chunk_nkb, nbands_occ * npol,
    &one, d_psi, npw, d_r, chunk_nkb, &zero, d_y, npw);
hamilt::cal_stress_nl_reordered_op<double, base_device::DEVICE_GPU>().chunk(
    gpu_ctx, ipol, jpol, npw, chunk_nkb, d_y, d_vkb_deri, d_stress);
```

- [ ] **Step 4: Verify red**

Run:

```bash
cmake --build build-test-cuda --target MODULE_PW_Hamilt_Kernels_UTs -j 8
```

Expected: link or compile failure because the new noncollinear overload is not implemented yet.

---

## Task 2: Add CUDA Noncollinear R Builder Interface and Kernel

**Files:**
- Modify: `source/source_pw/module_pwdft/kernels/stress_op.h`
- Modify: `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu`

**Interfaces:**
- Produces overload:

```cpp
void build_stress_nl_reordered_r_op<FPTYPE, base_device::DEVICE_GPU>::chunk(
    const base_device::DEVICE_GPU* ctx,
    const int& chunk_nkb,
    const int& nbands_occ,
    const int& deeq_2,
    const int& deeq_3,
    const int& deeq_4,
    const int& it,
    const int& atom_start,
    const int& atom_offset_in_type,
    const int& atom_count,
    const int& nproj,
    const FPTYPE* d_wg,
    const bool& occ,
    const FPTYPE* d_ekb,
    const FPTYPE* qq_nt,
    const std::complex<FPTYPE>* deeq_nc,
    const std::complex<FPTYPE>* becp,
    std::complex<FPTYPE>* r_chunk);
```

- [ ] **Step 1: Add declaration to the generic and GPU specializations**

The generic declaration may be an empty no-op body, matching existing CPU/generic chunk patterns.

- [ ] **Step 2: Implement CUDA kernel**

One thread computes one `(band, spinor, projector)` entry:

```cpp
idx -> ib_spinor, inkb1
ib = ib_spinor / 2
is = ib_spinor - ib * 2
ia = inkb1 / nproj
ip1 = inkb1 - ia * nproj
```

For each `ip2`:

```cpp
becp_up = becp[(ib * 2 + 0) * chunk_nkb + inkb2]
becp_dn = becp[(ib * 2 + 1) * chunk_nkb + inkb2]
ps_qq = -ekb[ib] * qq_nt[it * deeq_3 * deeq_4 + ip1 * deeq_4 + ip2]
ps0 = deeq_nc[((0 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq
ps1 = deeq_nc[((1 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2]
ps2 = deeq_nc[((2 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2]
ps3 = deeq_nc[((3 * deeq_2 + deeq_iat) * deeq_3 + ip1) * deeq_4 + ip2] + ps_qq
```

Then:

```cpp
if (is == 0) sum += fac * (ps0 * becp_up + ps1 * becp_dn);
else         sum += fac * (ps2 * becp_up + ps3 * becp_dn);
```

- [ ] **Step 3: Instantiate float and double**

Add explicit template instantiations if needed through the existing struct instantiation.

- [ ] **Step 4: Verify green for unit target**

Run:

```bash
cmake --build build-test-cuda --target MODULE_PW_Hamilt_Kernels_UTs -j 8
OMP_NUM_THREADS=1 ./build-test-cuda/source/source_pw/module_pwdft/kernels/test/MODULE_PW_Hamilt_Kernels_UTs
```

Expected: all tests pass outside the sandbox.

---

## Task 3: Integrate Noncollinear Reordered Path in Chunked Stress

**Files:**
- Modify: `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp`

**Interfaces:**
- Consumes the noncollinear `R` builder from Task 2.
- Reuses `stress_r_chunk`, `stress_y_chunk`, and `cal_stress_nl_reordered_op`.

- [ ] **Step 1: Build `R` for `npol == 2` after `becp_chunk` reduction**

In `FS_Nonlocal_tools::cal_stress_chunked`, after the existing scalar `if (npol == 1)` block, add an `else` branch:

```cpp
hamilt::build_stress_nl_reordered_r_op<FPTYPE, Device>().chunk(
    this->ctx,
    chunk_nkb,
    npm,
    this->nlpp_->deeq_nc.getBound2(),
    this->nlpp_->deeq_nc.getBound3(),
    this->nlpp_->deeq_nc.getBound4(),
    it,
    atom_type_start + ia_begin,
    ia_begin,
    atom_count,
    nh,
    d_wg_ik,
    occ,
    d_ekb_ik,
    qq_nt,
    this->nlpp_->template get_deeq_nc_data<FPTYPE>(),
    this->becp_chunk,
    this->stress_r_chunk);
```

- [ ] **Step 2: Compute `Y = Psi * R^H` for `npol == 2`**

Use:

```cpp
gemm_op()('N', 'C',
          npw, chunk_nkb, npm_npol,
          &ModuleBase::ONE,
          ppsi, this->max_npw,
          this->stress_r_chunk, chunk_nkb,
          &ModuleBase::ZERO,
          this->stress_y_chunk, npw);
```

- [ ] **Step 3: Remove derivative GEMM for `npol == 2`**

In the component loop, call:

```cpp
hamilt::cal_stress_nl_reordered_op<FPTYPE, Device>().chunk(
    this->ctx, ipol, jpol, npw, chunk_nkb,
    this->stress_y_chunk, this->vkb_chunk, stress);
```

Keep the old `cal_stress_nl_op` noncollinear path only as fallback if a future guard is added; the default CUDA chunked path should use reordered.

- [ ] **Step 4: Build production target**

Run:

```bash
cmake --build build --target abacus_basic_gpu -j 8
```

Expected: build succeeds.

---

## Task 4: Runtime and Profile Verification

**Files:**
- No code changes unless a verification failure exposes a bug.

- [ ] **Step 1: Run CUDA unit tests outside sandbox**

Run:

```bash
OMP_NUM_THREADS=1 ./build-test-cuda/source/source_pw/module_pwdft/kernels/test/MODULE_PW_Hamilt_Kernels_UTs
```

Expected: all tests pass.

- [ ] **Step 2: Run scalar Si256 regression**

Run the existing Si256 `dav_subspace`, `precision single`, `cal_force=1`, `cal_stress=1` benchmark with `build/abacus_basic_gpu`.

Expected:
- Final energy remains `-27439.5567724336033280 eV` for the existing staged input.
- Scalar `Stress stress_nl` remains near the current reordered value, around `2.9 s`.

- [ ] **Step 3: Run or stage a noncollinear PW stress case**

Search existing tests for a small `nspin 4`, `basis_type pw`, `device gpu`, `cal_stress 1` case. If none exists, stage one from an existing PW SOC/noncollinear input and run with `OMP_NUM_THREADS=1`.

Expected:
- Run completes.
- Stress tensor is finite.
- If a baseline can be run by forcing old path or using CPU, compare stress within existing GPU tolerance.

- [ ] **Step 4: Profile representative noncollinear case**

Use Nsight Systems outside sandbox:

```bash
OMP_NUM_THREADS=1 nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true -o profile /path/to/build/abacus_basic_gpu
```

Expected:
- No six derivative `dbecp` ZGEMM groups in noncollinear `stress_nl`.
- New `build_stress_nl_reordered_r_chunk_nc` or equivalent appears.
- `cal_stress_nl_reordered_chunk` appears for the six stress components.

---

## Self-Review

- Spec coverage: covers API, CUDA kernel, integration, tests, and runtime/profile verification.
- Placeholder scan: no `TBD`, `TODO`, or unspecified test commands.
- Type consistency: noncollinear overload consistently uses `deeq_nc` as `std::complex<FPTYPE>*`, scalar dimensions as `int`, and `becp/r_chunk` spinor layout as `(ib * 2 + spinor) * chunk_nkb + p`.

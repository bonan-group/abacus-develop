# CUDA Stress NL Reordered Contraction

## Objective

Reduce the CUDA PW nonlocal stress bottleneck by replacing the six repeated derivative projector projections with a reordered contraction for `npol == 1`. The optimized path should keep CPU and noncollinear behavior unchanged, preserve current stress results within GPU tolerances, and materially reduce `Stress stress_nl` for the Si256 `dav_subspace` case.

## Current Code Path

The current chunked CUDA stress path is in `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1287`. It builds `becp_chunk` with one GEMM at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1359`, then loops over the six stress tensor components at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1379`. Inside that loop it builds the derivative projector chunk at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1383`, computes `dbecp_chunk` with another GEMM at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1390`, and contracts `becp_chunk` with `dbecp_chunk` via `cal_stress_nl_op::chunk` at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1406`.

The current CUDA stress chunk contraction kernel is declared in `source/source_pw/module_pwdft/kernels/stress_op.h:410` and implemented in `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:680`. Its existing algebra is visible in `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:651`, where each band, atom, and projector pair contributes a weighted real product of `dbecp` and `becp`.

The existing GEMM wrapper for complex double calls cuBLAS `cublasZgemm` in `source/source_base/kernels/cuda/math_kernel_op.cu:300`. The profile confirms that these GEMMs dominate `stress_nl`, while the final stress contraction kernel is tiny.

## Implementation Plan

- [ ] 1. Add focused CUDA unit tests for the new reordered algebra before implementation. Extend `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp` near the existing stress tests at `source/source_pw/module_pwdft/kernels/test/stress_op_test.cpp:303`. Use a small deterministic dataset with one atom type, two atoms, `npol == 1`, nonzero `wg`, nonzero `ekb`, nonzero `qq_nt`, both diagonal-only and `nondiagonal` cases, and complex `becp`, `psi`, and derivative projector data. The reference should compute the existing formula on the host and compare it to the new two-stage GPU result. This proves the sign, conjugation, occupation weighting, and `ekb * qq_nt` handling before touching the production path.

- [ ] 2. Add stress-op CUDA interfaces for building the compact weighted coefficient matrix and reducing derivative projectors against the reordered field. Modify `source/source_pw/module_pwdft/kernels/stress_op.h:386` to add CUDA-only declarations behind the existing `base_device::DEVICE_GPU` specialization style. Keep the interfaces separate from `cal_stress_nl_op::chunk` so the existing path remains available. The first interface should consume `becp_chunk`, `wg`, `ekb`, `qq_nt`, `deeq`, atom/type metadata, and produce `R_chunk`. The second interface should consume `Y_chunk`, `vkb_deri_chunk`, component indices, and accumulate directly into the 3x3 stress tensor.

- [ ] 3. Implement the weighted coefficient builder in `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu`. For each band and projector row in the current chunk, compute the same projector-pair weighting currently assembled inside `cal_stress_nl_chunk` at `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:655`. Preserve `nondiagonal` behavior by skipping off-diagonal projector-pair contributions when the existing path would skip them. Include `wg` when `occ` is true, use the k-point weight when `occ` is false, and include the eigenvalue overlap correction when `d_ekb` is present. The output layout must be chosen to feed the next cuBLAS call without extra transposes.

- [ ] 4. Implement the derivative-dot stress reduction kernel in `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu`. This kernel should compute the stress component contribution by reducing over `G` and chunk projector index for the already generated `vkb_deri_chunk` and the reordered field `Y_chunk`. Match the current sign and conjugation convention by validating against the unit test from task 1. Accumulate only the requested lower-triangular stress component, matching the current `(ipol, jpol)` loop in `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1379`.

- [ ] 5. Add reusable chunk scratch buffers to `FS_Nonlocal_tools`. Modify `source/source_pw/module_pwdft/fs_nonlocal_tools.h:246` to add GPU chunk scratch for `R_chunk` and `Y_chunk`, plus capacities. Extend `ensure_chunk_memory` in `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:208` so the new buffers are allocated once per required chunk/band/plane-wave shape and reused across stress components. Keep the existing `dbecp_chunk` buffer for force and fallback stress paths.

- [ ] 6. Integrate the optimized path into `FS_Nonlocal_tools::cal_stress_chunked` for CUDA `npol == 1` only. Replace the inner `dbecp_chunk` GEMM and `cal_stress_nl_op::chunk` sequence at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1390` through `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1427` with the reordered path when running on CUDA and `npol == 1`. The integrated flow should keep the existing `becp_chunk` GEMM at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1359`, build `R_chunk`, perform one `Y_chunk` GEMM per atom/projector chunk, then loop over six stress components to generate `vkb_deri_chunk` and run the reduction kernel. Leave `npol == 2`, noncollinear, CPU, and unsupported cases on the existing path.

- [ ] 7. Ensure MPI pool behavior remains equivalent. The current path reduces `becp_chunk` across `POOL_WORLD` at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1372`. The optimized path must preserve this reduction before building `R_chunk`. If `Y_chunk` depends on local `psi` and reduced `R_chunk`, no new MPI reduction should be introduced inside the derivative loop. Confirm this with a small MPI-linked smoke run outside the sandbox if an MPI test case is available.

- [ ] 8. Add ABACUS timer or NVTX markers around the optimized substeps. Add short, removable or permanent scoped timing around weighted coefficient build, reordered GEMM, derivative projector generation, and derivative-dot reduction inside the CUDA `npol == 1` stress path. The goal is to distinguish remaining GEMM time from derivative generation and reduction time in the next Nsight run.

- [ ] 9. Build and run focused verification. Build `MODULE_PW_Hamilt_Kernels_UTs` and run it outside the sandbox because CUDA tests require device visibility. Confirm the new unit tests fail before implementation and pass after implementation. Then build `abacus_basic_gpu` and confirm the binary links cleanly.

- [ ] 10. Run Si256 correctness and performance verification. Re-run the Si256 `dav_subspace` profile with `cal_force=1`, `cal_stress=1`, `precision single`, and stress double behavior unchanged. Compare final stress against the baseline in `/tmp/abacus_si256_vnl_end_profile_20260620-111920/si256_dav_subspace` and the chunked profile in `/tmp/abacus_si256_stress_chunked_profile_20260620-131057/si256_dav_subspace`. Confirm `Stress stress_nl` decreases materially, Nsight no longer shows six derivative `dbecp` ZGEMM groups, and no new long GPU-idle gap appears.

## Verification Criteria

- The new CUDA unit tests compare the reordered contraction against the existing stress formula for diagonal and non-diagonal projector behavior.
- `MODULE_PW_Hamilt_Kernels_UTs` passes outside the sandbox with CUDA device visibility.
- `cmake --build build --target abacus_basic_gpu -j 8` succeeds.
- The Si256 `dav_subspace` run completes with final stress matching the current GPU baseline within existing tolerance.
- Nsight shows one reordered `Y_chunk` GEMM per stress chunk instead of six derivative `dbecp` GEMMs per stress chunk.
- `Stress stress_nl` is materially below the current chunked value of about 9.70 s and the previous full-path value of about 9.40 s.

## Potential Risks and Mitigations

1. **Conjugation or sign mismatch**
   Mitigation: Make the first test compare the reordered GPU path to the existing host formula using nontrivial complex data. Include at least one case where imaginary terms would expose a reversed conjugation.

2. **Incorrect projector-pair indexing inside chunks**
   Mitigation: Use the same `atom_start`, `atom_count`, `nproj`, `it`, and `deeq` indexing convention as `cal_stress_nl_chunk` in `source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:651`. Test with two atoms in one chunk so atom offsets are exercised.

3. **MPI pool semantics change**
   Mitigation: Keep the existing `becp_chunk` pool reduction before the new coefficient build. Do not add a derivative-side pool reduction unless the existing mathematical data ownership requires it.

4. **Extra memory pressure from `Y_chunk`**
   Mitigation: Reuse existing chunk sizing and allocate `Y_chunk` as `npw * chunk_nkb`. This is similar in size to `vkb_chunk`, about 260 MB for the Si256 default chunk size of 64 projectors. If memory pressure is too high, reduce chunk size with the existing `ABACUS_VNL_CHUNK_SIZE` mechanism.

5. **The new reduction becomes memory-bandwidth bound**
   Mitigation: This is expected but should still be much cheaper than six `P * N * G` ZGEMMs. If reduction time dominates, optimize reduction layout and block sizing after correctness is proven.

6. **Noncollinear path accidentally changes**
   Mitigation: Gate the optimized path strictly on `npol == 1` and CUDA. Preserve the existing `npol == 2` branch at `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1431`.

## Alternative Approaches

1. **Grouped or fused six-component GEMM**
   This would reduce launch overhead but preserve nearly all dominant FLOPs. It is not the preferred path because the profile shows compute time, not launch time, dominates.

2. **Full `vkb` large-GEMM path**
   This is slightly faster than chunking in the current profile, but it does not reduce the six repeated derivative projections and can require very large full-projector storage.

3. **Delay derivative projector generation until reduction**
   This could avoid writing `vkb_deri_chunk`, but it requires fusing the derivative-projector formula into the reduction kernel. That is a good second-phase optimization after the reordered contraction is correct.

## Assumptions

- CUDA is the first implementation target.
- The first optimized path is limited to `npol == 1`.
- CPU and noncollinear behavior remain unchanged.
- Stress remains double precision.
- The current chunked projector generation is retained initially to minimize algorithmic and correctness risk.

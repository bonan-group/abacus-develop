# GPU Accelerated PW Stress Calculation Plan

## Objective

Accelerate the PW stress calculation for the CUDA `Si256 dav_subspace` path by moving the dominant host-side stress terms and fine-grained synchronized reductions onto the GPU, while preserving the current CPU behavior and stress math. The first target is the stress-stage GPU idle gap seen in Nsight; ROCm parity should follow only after the CUDA design is verified.

## Current Code Path And GPU Coverage

`ESolver_KS_PW::cal_stress` updates device psi and calls `Stress_PW<double, Device>::cal_stress` with the device psi object (`source/source_esolver/esolver_ks_pw.cpp:318`). `Stress_PW::cal_stress` then executes stress terms serially: kinetic, Hartree, Ewald, GGA/mGGA, local, NLCC, nonlocal, and optional USPP/VDW/onsite/EXX terms (`source/source_pw/module_pwdft/stress_pw.cpp:77`, `source/source_pw/module_pwdft/stress_pw.cpp:135`). The final assembly is host-side `ModuleBase::matrix` addition (`source/source_pw/module_pwdft/stress_pw.cpp:135`).

The existing GPU support is partial:

- Kinetic stress has a GPU `cal_multi_dot_op`, but `FS_Kin_tools::cal_gk` forms `g+k` on the host and copies it to the device for each k-point (`source/source_pw/module_pwdft/fs_kin_tools.cpp:51`, `source/source_pw/module_pwdft/fs_kin_tools.cpp:73`). `cal_stress_kin` calls one dot reduction per band, spin channel, and lower-triangle tensor component (`source/source_pw/module_pwdft/fs_kin_tools.cpp:93`, `source/source_pw/module_pwdft/fs_kin_tools.cpp:110`). The CUDA operator allocates one scalar, copies it device-to-host, frees it, and synchronizes on every call (`source/source_pw/module_pwdft/kernels/cuda/stress_op.cu:665`).
- Hartree stress is effectively host-side: it allocates a host `aux`, packs density with OpenMP, calls `rho_basis->real2recip(aux, aux)`, and reduces over `rho_basis->npw` on the CPU (`source/source_pw/module_pwdft/stress_har.cpp:20`, `source/source_pw/module_pwdft/stress_har.cpp:58`, `source/source_pw/module_pwdft/stress_har.cpp:69`).
- Ewald stress is host-side: alpha selection, G-space structure-factor accumulation, R-space neighbor shell generation, and tensor reductions all run in CPU/OpenMP loops (`source/source_pw/module_pwdft/stress_ewa.cpp:23`, `source/source_pw/module_pwdft/stress_ewa.cpp:68`, `source/source_pw/module_pwdft/stress_ewa.cpp:123`).
- GGA stress is host-side through `XC_Functional::gradcorr`: it performs real-to-recip transforms and allocates host arrays (`source/source_hamilt/module_xc/xc_grad.cpp:72`, `source/source_hamilt/module_xc/xc_grad.cpp:96`), then accumulates stress in CPU/OpenMP grid loops (`source/source_hamilt/module_xc/xc_grad.cpp:233`, `source/source_hamilt/module_xc/xc_grad.cpp:312`, `source/source_hamilt/module_xc/xc_grad.cpp:381`).
- mGGA stress is mixed: gradient/crosstau accumulation uses device tensors and a GPU operator (`source/source_pw/module_pwdft/stress_mgga.cpp:37`, `source/source_pw/module_pwdft/stress_mgga.cpp:67`), but it copies `crosstaus` back to CPU and performs the final real-space grid reduction on the host (`source/source_pw/module_pwdft/stress_mgga.cpp:76`, `source/source_pw/module_pwdft/stress_mgga.cpp:96`).
- Local stress is mixed but host-dominant: it packs density and transforms on host (`source/source_pw/module_pwdft/stress_loc.cpp:33`, `source/source_pw/module_pwdft/stress_loc.cpp:64`), accumulates local energy and stress contractions on CPU (`source/source_pw/module_pwdft/stress_loc.cpp:71`, `source/source_pw/module_pwdft/stress_loc.cpp:113`), while only the radial `dvloc_of_g` helper has a GPU path that allocates/copies temporary arrays and immediately copies results back (`source/source_pw/module_pwdft/stress_loc.cpp:242`, `source/source_pw/module_pwdft/stress_loc.cpp:261`).
- NLCC stress is mixed but host-dominant: it recomputes `vxc`, packs and transforms `psic` on host (`source/source_pw/module_pwdft/stress_cc.cpp:52`, `source/source_pw/module_pwdft/stress_cc.cpp:76`, `source/source_pw/module_pwdft/stress_cc.cpp:101`), performs diagonal and non-diagonal contractions on CPU (`source/source_pw/module_pwdft/stress_cc.cpp:124`, `source/source_pw/module_pwdft/stress_cc.cpp:151`), while `deriv_drhoc` has the same allocate-copy-kernel-copy-back GPU helper pattern (`source/source_pw/module_pwdft/stress_cc.cpp:298`, `source/source_pw/module_pwdft/stress_cc.cpp:314`).
- Nonlocal stress is the most GPU-ready term: it accumulates a device stress buffer (`source/source_pw/module_pwdft/stress_nl.cpp:31`), uses `FS_Nonlocal_tools` (`source/source_pw/module_pwdft/stress_nl.cpp:36`), supports a chunked CUDA path (`source/source_pw/module_pwdft/stress_nl.cpp:54`, `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:1271`), and calls CUDA kernels for VNL derivatives and stress contractions (`source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:608`, `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:684`). It still has host-side setup/copies for `g+k`, interpolation tables, prefactors, and structure factors (`source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:393`, `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:404`, `source/source_pw/module_pwdft/fs_nonlocal_tools.cpp:430`).
- USPP, onsite, and EXX stress paths are host-heavy (`source/source_pw/module_pwdft/stress_us.cpp:30`, `source/source_pw/module_pwdft/stress_onsite.cpp:51`, `source/source_pw/module_pwdft/stress_exx.cpp:87`), but they are conditional and should not lead the first CUDA stress pass for norm-conserving Si.

## Implementation Plan

- [ ] Add temporary stress-term profiling boundaries for CUDA validation.
  - [ ] Add scoped timing/NVTX around `stress_kin`, `stress_har`, `stress_ewa`, `stress_gga`, `stress_mgga`, `stress_loc`, `stress_cc`, and `stress_nl`.
  - [ ] Keep profiling changes separable from the optimization patches so they can be dropped after the Nsight comparison.
  - [ ] Capture baseline per-term CPU wall time and GPU idle spans for the exact `Si256 dav_subspace` input.

- [ ] Introduce a CUDA stress workspace and device-resident tensor accumulator.
  - [ ] Add a CUDA-only internal workspace owned by the PW stress layer or a helper object, containing device buffers for 9 stress values, 6 lower-triangle values, per-block reduction scratch, and reusable radial/reciprocal temporaries.
  - [ ] Keep the public `ModuleBase::matrix` interface unchanged and copy only the final 9 values back for MPI reduction and existing print/output paths.
  - [ ] Preserve CPU template behavior unchanged; specialize only the CUDA `Device` path behind existing `Stress_Func<FPTYPE, DEVICE_GPU>` and kernel operator patterns.

- [ ] Fix kinetic stress granularity first.
  - [ ] Replace per-band/per-component `cal_multi_dot_op` scalar reductions with a batched CUDA kernel that accumulates all six independent tensor components for one k-point, all occupied bands, and both spinor components when present.
  - [ ] Reuse the kinetic workspace instead of `cudaMalloc/cudaFree` inside every dot reduction.
  - [ ] Move or cache `g+k` and `kfac` generation in device memory so `FS_Kin_tools::cal_gk` does not rebuild host arrays and copy them for every stress call when the same k-basis data is already available or can be generated on device.
  - [ ] Leave final pool/all reductions on host initially; copy one 9-value tensor per term rather than one scalar per dot.

- [ ] Port Ewald stress to CUDA.
  - [ ] Split Ewald into explicit CUDA helpers for alpha/constant setup, G-space stress, and R-space stress.
  - [ ] Pre-upload atom positions, type offsets, valence charges, lattice vectors, and compact atom-type metadata to device workspace.
  - [ ] For G-space, compute ionic structure factors and six tensor components in a single reduction over `rho_basis->npw`, skipping `ig_gge0` on device.
  - [ ] For R-space, start with a conservative hybrid design: generate the bounded neighbor image list or pair-shell descriptors on CPU once per cell, upload the compact list, and reduce pair contributions on GPU. This avoids immediately porting `H_Ewald_pw::rgen` control flow while removing the dominant pair contribution loop from the stress gap.
  - [ ] Once validated, evaluate a full GPU `rgen`/pair enumeration only if CPU shell-list generation remains visible in Nsight.

- [ ] Move Hartree stress to device-resident reciprocal density and GPU reductions.
  - [ ] Reuse existing device density/FFT outputs where available instead of repacking `chr->rho` into a host `aux`.
  - [ ] If the density is only resident in real space, add a CUDA path to sum spin densities on device and call the device FFT route used by `PW_Basis`.
  - [ ] Add a CUDA reduction over G-vectors for the six lower-triangle Hartree components and the diagonal energy term adjustment.
  - [ ] Keep the current gamma-only scaling and `ig_gge0` handling exactly matched to the CPU formula.

- [ ] Make local stress and NLCC device-resident.
  - [ ] Pre-upload radial grids, `rab`, local potential tables, core charge tables, and `gx_arr` per atom type into persistent CUDA workspace buffers.
  - [ ] Change `dvloc_of_g` and `deriv_drhoc` CUDA paths so they write device outputs that remain on device; remove immediate device-to-host copies after `cal_stress_drhoc_aux_op`.
  - [ ] Add GPU contraction kernels for `(rho(G), structure factor, dvloc/drhocg, gcar)` to produce the local and NLCC stress tensors directly on device.
  - [ ] Keep only a 9-value final device-to-host transfer per component family.
  - [ ] For the first Si256 target, gate the NLCC path so it returns early without extra work when no atom type has NLCC, matching the current `judge==0` behavior.

- [ ] Port GGA and finish mGGA reductions.
  - [ ] For GGA, isolate the stress-only path in `XC_Functional::gradcorr` and add a CUDA implementation for gradient-density stress accumulation.
  - [ ] Reuse existing GPU XC functional kernels if they can provide `v2xc`; otherwise add a minimal stress-only CUDA kernel for the specific functionals used in the benchmark first, then generalize.
  - [ ] Keep LibXC-heavy or unsupported functional combinations on the CPU fallback path until GPU functional parity is explicit.
  - [ ] For mGGA, keep `grad_wfc` and `cal_stress_mgga_op` on device and move the final `v_ofk * (kin_r + crosstaus)` grid reduction to CUDA, avoiding `crosstaus.to_device<ct::DEVICE_CPU>()` in the GPU path.

- [ ] Clean up nonlocal stress after the CPU-dominant terms.
  - [ ] Confirm the chunked VNL stress path is active for the target run and does not fall back to full `vkb` allocation.
  - [ ] Batch all six tensor derivative contractions where practical so `cal_vkb_deri_s`, GEMM, and `cal_stress_nl_op` launch patterns are not repeated more than necessary.
  - [ ] Cache `g+k`, `vq_tab`, `pref`, and derivative index device buffers across tensor components and k-points when valid.
  - [ ] Preserve the current final 9-value transfer and MPI reduction unless MPI device collectives are introduced separately.

- [ ] Integrate term results through a consistent host/device boundary.
  - [ ] Define a single stress assembly point that combines CUDA term tensors before one final copy, or copies one tensor per term only for temporary debugging.
  - [ ] Mirror lower-triangle to full tensor and apply symmetry in the same order as the CPU path unless a deliberate, tested reordering is needed.
  - [ ] Keep optional USPP, onsite, EXX, and VDW stress on existing paths for the first CUDA milestone, with explicit fallbacks documented in code comments or timer labels.

- [ ] Add focused correctness tests.
  - [ ] Add small deterministic CUDA-vs-CPU tests for kinetic, Ewald, Hartree, local, and NLCC stress tensors.
  - [ ] Add or extend a PW stress integration test that checks total stress against CPU reference within existing GPU tolerances.
  - [ ] Include gamma-only and non-gamma coverage for `ig_gge0` and scaling behavior.
  - [ ] Run MPI-linked stress tests outside the sandbox with `OMP_NUM_THREADS=1`.

- [ ] Re-profile the target benchmark after each tranche.
  - [ ] Re-run Nsight Systems on `Si256 dav_subspace` after kinetic batching, after Ewald GPU, and after local/Hartree/GGA residency.
  - [ ] Track per-term wall time, largest GPU idle gap, number of synchronous D2H copies, and `cudaMalloc/cudaFree` counts during stress.
  - [ ] Remove temporary profiling markers once the optimized code path is stable.

## Verification Criteria

- CPU stress behavior remains unchanged for non-GPU builds.
- CUDA stress agrees with CPU reference for component tests within established ABACUS GPU tolerances.
- The `Si256 dav_subspace` benchmark preserves final energy, forces, and total stress within current accepted tolerances.
- The stress-stage idle gap from the original profile is reduced by at least 50%.
- No new GPU idle interval above 1 second is introduced during stress.
- Stress computation does not perform per-band or per-component `cudaMalloc/cudaFree` or scalar D2H synchronization in the CUDA path.
- MPI-linked ABACUS runtime verification is run outside the sandbox with `OMP_NUM_THREADS=1`.

## Potential Risks And Mitigations

- Numerical reduction order will change on GPU and may shift low bits of stress components. Mitigate with component-level tolerances, deterministic small-cell tests, and CPU fallback comparisons.
- Ewald R-space pair enumeration has complex boundary behavior through `H_Ewald_pw::rgen`. Mitigate by first generating a CPU shell list and only moving arithmetic reductions to GPU.
- GGA stress may depend on LibXC paths that are not currently GPU-ready. Mitigate with a CPU fallback for unsupported functionals and a first CUDA implementation for the benchmark functional set.
- Device memory pressure can rise if all stress workspaces are allocated simultaneously. Mitigate with a reusable workspace sized by `npw`, `ngg`, `nrxx`, atom metadata, and term-local scratch reuse.
- Existing `PW_Basis` FFT ownership may make density residency non-obvious. Mitigate by auditing current device FFT APIs before changing Hartree/local/GGA setup.
- MPI reductions are currently host reductions. Mitigate by keeping final tensor MPI reduction on host in phase 1 and treating GPU-aware MPI as a separate optimization.

## Alternative Approaches

- Port only Ewald first. This directly attacks the clearly CPU-only term, but leaves kinetic scalar synchronizations and local/Hartree/GGA host work in place.
- Optimize only allocator/synchronization noise first. This is lower risk and helps kinetic/local/NLCC helpers, but may not reduce the large stress gap enough if Ewald and GGA dominate.
- Build a full device-resident stress subsystem in one pass. This gives the cleanest final architecture, but has a larger correctness and review surface; it should be approached as staged patches even if the end state is shared.
- Leave GGA/LibXC on CPU initially. This is pragmatic for the Si256 target if it uses a simpler functional path, but a complete GPU stress solution will eventually need a supported GPU GGA stress path or an explicit fallback policy.

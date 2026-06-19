# Extracting Band Structure

In ABACUS, in order to obtain the eigenvalues of Hamiltonian, or generally called band structure, examples can be found in [examples/band](https://github.com/deepmodeling/abacus-develop/tree/develop/examples/band).
Similar to the [DOS case](https://abacus-rtd.readthedocs.io/en/latest/advanced/elec_properties/dos.html), one first needs to perform a ground-state energy calculation ***with one additional keyword "[out_chg](https://abacus-rtd.readthedocs.io/en/latest/advanced/input_files/input-main.html#out-chg)" in the INPUT file***:

```
out_chg 1
```

With this input parameter, the converged charge density will be output in the files such as `chgs1.cube`, `chgs2.cube`, etc.
Then, one can use the same `STRU` file, pseudopotential files and atomic orbital files (and the local density matrix file onsite.dm if DFT+U is used) to do a non-self-consistent (NSCF) calculation. In this example, the potential is constructed from the ground-state charge density from the proceeding calculation. Now the INPUT file is like:

```
INPUT_PARAMETERS
#Parameters (General)
nbands        8
calculation   nscf
basis_type    lcao
read_file_dir ./

#Parameters (Accuracy)
ecutwfc       60
scf_nmax      50
scf_thr       1.0e-9
pw_diag_thr   1.0e-7

#Parameters (File)
init_chg      file
out_band      1
out_proj_band 1

#Parameters (Smearing)
smearing_method gaussian
smearing_sigma  0.02
```

Here is a relevant k-point file KPT (in LINE mode):

```
K_POINTS # keyword for start
6 # number of high symmetry lines
Line # line-mode
0.5 0.0 0.5 20 # X
0.0 0.0 0.0 20 # G
0.5 0.5 0.5 20 # L
0.5 0.25 0.75 20 # W
0.375 0.375 0.75 20 # K
0.0 0.0 0.0 1 # G
```

This means we are using the following k-points:

- 6 k points, here means 6 k points:
  (0.5, 0.0, 0.5) (0.0, 0.0, 0.0) (0.5, 0.5, 0.5) (0.5, 0.25, 0.75) (0.375, 0.375, 0.75) (0.0, 0.0,
  0.0)
- 20/1 number of k points along the segment line, which is constructed by two adjacent k
  points.

Next, run ABACUS and you will see a file named `eigs1.txt` in the output directory. 
Plot it and you will obtain the energy band structure!

If "out_proj_band" set 1, it will also produce the projected band structure in a file called PBAND_1 in xml format.

The PBAND_1 file starts with number of atomic orbitals in the system, the text contents of element `<band structure>` is the same as data in the BANDS_1.dat file, such as:

```
<pband>
<nspin>1</nspin>
<norbitals>153</norbitals>
<band_structure nkpoints="96" nbands="50" units="eV">
...

```

The rest of the files arranged in sections, each section with a header such as below:

```
<orbital
 index="                                   1"
 atom_index="                              1"
 species="Si"
 l="                                       0"
 m="                                       0"
 z="                                       1"
>
<data>
...
</data>

```

The shape of text contents of element `<data>` is (Number of k-points, Number of bands)

## Hybrid PW Band Structure With EXX

For plane-wave hybrid-functional calculations, ABACUS can calculate band eigenvalues on a separate band path after the hybrid SCF calculation converges. This workflow is different from the ordinary NSCF workflow above: keep `calculation` set to `scf`, and provide the SCF k-point mesh plus an additional `K_POINTS_BAND` section in the `KPT` file.

The INPUT file should include the PW hybrid settings and request band output, for example:

```
INPUT_PARAMETERS
calculation             scf
basis_type              pw
ks_solver               dav
out_band                1

exx_hybrid_step         1
exxace                  1
exx_separate_loop       1
```

The `KPT` file first defines the k-point mesh used by the SCF calculation, then defines the target band path:

```
K_POINTS
0
Gamma
2 2 2 0 0 0

K_POINTS_BAND
4
Line
0.0 0.0 0.0 10
0.5 0.0 0.0 10
0.5 0.5 0.0 10
0.0 0.0 0.0 1
```

`K_POINTS_BAND` supports the following coordinate modes:

- `Line`, `L`, or `Line_Direct`: line-mode interpolation in direct coordinates. Each special point line contains `kx ky kz n`, where `n` is the number of points generated from that special point toward the next one.
- `Line_Cartesian`: line-mode interpolation in Cartesian coordinates, with the same `kx ky kz n` format.
- `Direct` or `D`: explicit target k-point list in direct coordinates.
- `Cartesian` or `C`: explicit target k-point list in Cartesian coordinates.

For explicit `Direct` or `Cartesian` lists, only `kx ky kz` is used. A trailing weight value may be present for compatibility with ordinary `K_POINTS` files, but it is ignored:

```
K_POINTS_BAND
2
Direct
0.0 0.0 0.0
0.5 0.0 0.0
```

After the SCF calculation converges, ABACUS reuses the converged hybrid potential and EXX operator to solve the Hamiltonian on the `K_POINTS_BAND` path and writes the band eigenvalues through the usual `out_band` output files.

Current restrictions:

- This workflow requires `calculation scf`, not `calculation nscf`.
- It is implemented for `basis_type pw` hybrid EXX calculations.
- `exxace 1` and `exx_separate_loop 1` are required.
- `out_band 1` is required when `K_POINTS_BAND` is present.
- `KPAR > 1` is not supported for this workflow.

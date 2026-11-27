# Hybrid quick sphere

For the complete dual-toolkit build and ABI verification, see the [8.1 build instructions](../../docs/CLUSTER81.md).

`anat_fix_topology` stages this directory. Only the repeated quick-sphere edge distances,
triangle metrics and normals use CUDA. The mesh connectivity and device buffers are reused;
coordinates and metric fields cross the host/device boundary. The step search, gradient
averaging, genetic repair and subsequent intersection cleanup remain native FreeSurfer.
No research profiler, PyTorch worker, license file or unversioned system `mris_sphere` is required.

`native/qsphere` embeds the upstream sphere entry point and links `geometry.so` via `$ORIGIN`
and the installed `fsbindings.cpython-38-x86_64-linux-gnu.so`. Supported build stamp:
`freesurfer-linux-ubuntu20_x86_64-8.1.0-20250718-f30dfed`, Linux x86-64, CUDA multi-architecture binaries (see root README). Builds target all ordinary SMs reported by nvcc plus baseline PTX unless --arch is specified.
Select `--topology_backend cpu` to run the unmodified distribution executable instead.
GPU mode aborts on an incompatible ABI or device, missing CUDA evidence, or a changed
mesh identity/volume geometry; it does not silently switch to the CPU executable.

To rebuild with matching FreeSurfer sources, GCC/OpenMP, CUDA, `xxd`, and FreeSurfer's
dependency headers (including ITK), run from this directory:

```bash
python build.py "$FS_SOURCE" "$FREESURFER_HOME" native \
    --include "$ITK_INCLUDE" --include "$DEPENDENCY_INCLUDE" --nvcc "$NVCC"
```

The build extracts the native metric helpers from the official tree. No copy of the installed
bindings or of a user's FreeSurfer license is distributed. See `../../THIRD_PARTY.md`.

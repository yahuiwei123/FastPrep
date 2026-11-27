# T1 placement backend

For the complete dual-toolkit build and ABI verification, see the [8.1 build instructions](../../../docs/CLUSTER81.md).

Only the T1 branches of FreeSurfer 8.1.0 `mris_place_surface --white` and `--pial` are supported. The entry point is assembled from externally supplied FreeSurfer source; the multimodal branch (not exported by the binding) is replaced by an explicit error. Upstream code performs border search, native forces, step reduction and final processing. The CUDA Jacobi smoother preserves the native neighbor order, signed float dot test, double accumulation, frozen vertices and the scratch gradients. `orderedMove` links the *same* movement bridge and uses the same persistent ordered PyTorch worker and GPU triangle tests as the T2 backend. It needs zero momentum, a current face hash and the native 0.01 mm near-neighbor limit. No CUDA kernel runs in the subject's FreeSurfer installation.

Rebuild only against the matching stamped FreeSurfer installation/bindings, source headers, ITK and CUDA toolkit. On a matching system, from the FastPrep root, build into a **fresh** directory, then deploy its three files together as `vendor/t2_pial/t1/native/` after validation:

```bash
python vendor/t2_pial/t1/build.py "$FS_SOURCE" "$FREESURFER_HOME" /path/to/fresh-t1-runtime \
  --include "$ITK_INCLUDE" --include "$DEPENDENCY_INCLUDE" --nvcc "$CUDA_HOME/bin/nvcc"
```

The shared `../native/classify.so` and `../native/runtime.json` must match. Both placement drivers use protocol version 2: deploy the T1 and T2 drivers together with their metadata and the shared worker. At the first movement, the bridge writes the loaded `MRIS` one-ring (`vertices_topology[vertex].v[0:vnum]`) as CSR offsets and vertex IDs. The worker checks that it agrees with the faces, then keeps the native edge order on the GPU across all movement attempts. There is no external topology snapshot or reconstruction of neighbor order from faces. `place`, `signed.so` and `runtime.json` are stage assets, not a general replacement for the FreeSurfer executable. The configured BFC Python must provide NumPy and CUDA-enabled PyTorch. Use the workflow selector, not `LD_PRELOAD`. In FastPrep's scratch directory, each movement is exchanged with the worker; that temporary state is not published. No T2, T2 field builder or histogram data is used by T1 placement. See `../../../README.md` for mode selection and the distinction between checked and unchecked execution.

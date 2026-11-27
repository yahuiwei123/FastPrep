# FastPrep QC

In the 8.1 workflow, the default `report_only` mode also applies to the scientific stages that measure surface defects. Finite, corresponding results with residual intersections or folded sphere faces continue downstream. Published `outputs.json` records `quality_warnings` by stage and preserves each QC verdict in `qc_stages`.


FastPrep QC is a standalone report and release gate. It reads completed
`fastprep.stage/1` manifests, writes measurements, images, JSON reports and an
HTML index, and can forward only the ports declared by the selected QC node.
It does not run preprocessing or claim scientific approval.

## Run

From the project directory:

```bash
python bin/qc.py --spec spec.json --runtime runtime.json \
  --blueprint blueprint.json --mode report_only
```

The spec identifies `analysis_unit_id` (or `subject_id`), `node_id` (or
`qc_node_id`) and upstream stage result directories. The runtime supplies the
QC interpreters, configured command prefixes, environment, resource paths and
`session_input_plan`. Paths can be absolute for the current run or relative to
the project directory. The blueprint contains the selected QC node and its
policy. `report_only` is the default: a fail or unknown result is forwarded as
provisional. `strict` returns a nonzero status and does not publish
`result/manifest.json` unless the report passes and policy status is approved.

In the Nextflow workflow, `report_only` schedules QC as reporting branches:
scientific consumers bind directly to completed scientific producers and can
run while QC reports are generated. Final assembly explicitly joins all eleven
completed QC branches, including full cortical-pair measurements, before
publishing. Failed QC execution still fails the run; a reported fail or unknown
remains provisional. `strict` waits for the original QC gates before releasing
downstream work. No geometry validation inside scientific commands is bypassed.
This DAG revision requires a newly prepared 51-node blueprint and a subject
boundary when upgrading a running cohort; do not mix it with a 50-node attempt.

The runtime uses `fastprep.runtime/1`, `package_root`, and a `qc` section with
`render_python`, `pair_python`, `fs_home` and `vendor_root`. `vendor_root` is
the directory containing `neuroqc/` (normally `<package_root>/vendor`). The
renderer imports that package, and rules are read from
`vendor_root/neuroqc/references`. `pair_python` supplies FCL/trimesh;
`render_python` supplies VTK for full-vertex geometry and image rendering.
`commands` maps command identifiers to `argv_prefix`, optional `environment`
and optional file path strings. `resources` entries have `id` and `path`.

### Comparing volume QC implementations

New Nextflow QC tasks pass `--volume-engine batched`. To reproduce a QC gate
with `bin/qc.py`, use the same upstream results, blueprint, runtime and mode,
but a **fresh working directory per run**. Select one of:

- `reference`: original per-label assignment and a separate renderer per PNG.
  This remains the default for CLI calls that omit the flag.
- `mapped`: same rendering as `reference`; compact the sorted nonzero labels
  in one pass with a lookup table (or a sorted lookup for sparse ranges).
  Both optimized modes reject more than 255 foreground codes rather than
  losing labels in the `uint8` QC volume. The input segmentation is untouched.
- `batched`: same mapping as `mapped`; one renderer per requested MRI/label
  volume pair loads the voxel arrays and reuses the canonical axial slices,
  display aspect and intensity window. The cache is not shared between pairs,
  sessions or processes. A pair with no selected visual outputs does not start
  a renderer. Segmentation metrics still run independently of rendering.

This is a CPU optimization; it does not require a GPU or drop any views.
Each selected view retains its PNG and JSON sidecar, including the original
label definition for contours. The extra `*-batch.json` lists completed views
and per-view errors; missing required evidence remains missing. For new runs
using `mapped` or `batched`, `provenance.qc_commands[].elapsed_seconds` in the
stage report measures each subprocess; it does not include in-process work
such as label mapping. Compare decoded PNG pixels and scientific report fields,
not command lines, run paths or timestamps, when validating parity.

### Multi-view surface rendering

New Nextflow runs use `--qc_surface_engine reuse`. For consecutive views of
the same hemisphere, mesh and scalar layer, one offscreen VTK window retains
the geometry, actors and lookup tables. Each view applies surfplot's original
hemisphere orientation and an independently fitted parallel camera. The
600×600 viewports, lighting, shared feature scale and original PNG montage are
unchanged. Meshes and scalar layers are not shared across subjects or commands;
`sulc` and `curv` remain separate evidence. This changes drawing only, not mesh
metrics, surface reconstruction, view selection or approval rules.

Use `--qc_surface_engine reference` in Nextflow to restore one window per view.
For a standalone QC gate (`bin/qc.py`), surface rendering command or
`bin/qc_spherical.py`, the equivalent option is `--surface-engine reference|reuse`;
omitting it retains the original renderer. Surface and volume engines can be
compared independently. When validating a new VTK/graphics environment, use
identical meshes, feature arrays, policy and plotting versions and compare the
decoded pixels as well as view order, orientation, color scale and report status.

## Input and stage contracts

Upstream and forwarded manifests use `fastprep.stage/1` and contain `subject`,
`node_id`, `outputs`, `metrics` and `commands`. Output paths are relative to
their result directory. Collections retain their main JSON record path and
declared `members`; scalar outputs do not gain a synthetic members list.
Forwarded contracts come from the selected QC node declaration. No content
hashes are read, generated or required.

Session input plans provide `analysis_unit_id`, `subject_id`, `session_id`, a
`reference`, and `companions`. Each item declares `modality` (`T1w`, `T2w` or
`FLAIR`), `structural_status`, `visual_review`, `path`, identity fields and
`probe.facts` (`shape`, `finite`, `nonzero`, `affine`, `voxel_size_mm` and
`spatial_units`). A structural pass only means these structural facts passed.
Input visual quality remains pending until a person reviews the generated
images. BIDS-declared modality is displayed as declared; it is not inferred
from image intensities.

## Hemisphere execution

Independent hemisphere chains run concurrently within each native stage. This
covers tessellation, topology correction, intersection repair, gray/white
intensity estimation, white and pial placement, both cortex labels, smoothing
and inflation, curvature statistics, sphere generation, SUGAR registration,
Jacobian, cortical parcellation, percent contrast, parcellation statistics,
sphere correspondence validation and both fsaverage6 surface resamplings.
The workflow graph and output ports are preserved. Shared input validation,
T2 preparation, template staging and manifest assembly complete outside the
hemisphere workers. Hypointensity relabeling, surf2vol and brain-volume
statistics retain their joint-volume execution. QC jobs allocated one CPU
retain their serial hemisphere checks and rendering.

The node parameter `hemisphere_workers` defaults to 2 and accepts 1 or 2.
Workers divide the allocated CPU count, rounding down. White/pial placement
caps parallel workers at two threads each; GPU placement supports 2, 4 and
8 threads, with at least four allocated CPUs needed for two GPU hemisphere
workers. With one allocated CPU, ordinary hemisphere work runs serially;
explicit GPU placement retains its unsupported-thread error. GPU stages also
accept `gpu_hemisphere_workers=1` to limit concurrent hemisphere tasks on the
selected device. Two GPU workers use the same selected device and require
memory for both processes; device selection does not reserve a GPU or its
memory in a scheduler.

Each worker has isolated command records and metric dictionaries. Hemisphere
chains run their scientific commands as separate processes, keep intra-chain
dependencies and read their own most recent command log for topology checks.
`pctsurfcon --nolog` avoids its shared internal log; stdout and stderr remain
in each worker's command logs. The parent merges records after both workers
finish, rejects colliding metric keys and propagates any worker failure.
Failed executions retain command journals and hemisphere execution records,
and do not publish a successful stage manifest.

`metrics.hemisphere_execution` and `hemisphere_execution.json` record actual
workers, allocated CPUs, per-worker threads, each hemisphere's wall time and
status, and combined wall time. `logs/lh/` and `logs/rh/` contain distinct
command logs and journals; root `commands.json` contains both hemispheres.
BLAS, OpenMP, ITK, NumExpr and VTK thread limits respect each worker's budget.
Geometry, identity, finite-value, topology and correspondence checks remain
required. A performance comparison does not replace anatomical QC.

## Ribbon and cortical-pair execution

`anat_ribbon` enables native `mris_volmask --parallel` when at least four CPUs
are allocated, then publishes the ribbon volumes. `anat_cortical_pair` reads
the completed ribbon and starts left and right cortical-pair workers as
separate processes, with at most two workers and a divided CPU thread budget.
A one-CPU task runs serially. The parameters `ribbon_parallel=false` in
`anat_ribbon` and `metrics_parallel=false` in `anat_cortical_pair` select
the serial reference execution explicitly.
Both parameters default to true for new executions; existing running snapshots
keep their own implementation.

Every cortical vertex and every eligible noncontact cortical white triangle
is still checked. Numerical contact tolerance, pinned/contact exclusions,
thickness values and ribbon label definitions are unchanged. A failed worker
blocks stage completion, with separate logs retained for both hemispheres.
The downstream pial/ribbon QC gate reuses these results as before.

The ribbon manifest records rasterization in `metrics.ribbon_execution`.
The pair manifest records measurement wall time, total time and worker count
in `metrics.cortical_pair_execution`, with both JSON measurements registered
as a hemisphere collection. Each `metrics.lh` / `metrics.rh`
record contains `timings_seconds` for input validation, pial mesh validation,
vertex containment, numerical-contact distances, intersection preparation,
triangle-pair intersections and summary statistics. CLI workers also record
input reading, ribbon coverage and worker total time; phase completion is
flushed to stderr. `commands.json` records per-process duration and thread
budget. Timing/provenance fields are excluded when comparing scientific
measurements between serial and parallel executions; scientific fields must
still agree. These execution records do not grant anatomical approval.

## Outputs and interpretation

`qc_result/approval.json` records the execution mode, policy state and whether
release was justified. `qc_result/<node>/report.json`, measurements, evidence
sidecars, command logs and PNGs preserve unknown and fail statuses.
`qc_result/report/` contains a JSON summary and an HTML index whose links are
relative to the index location. Forwarded outputs and QC command logs are
copied to `result/` with paths relative to that result directory, then
registered in `result/manifest.json`. Existing different destination files
are not overwritten.

An approval is true only when every required check passes and the blueprint
policy is explicitly approved. Missing measurements, required images, visual
review or protocol decisions remain unknown. Report generation never creates
visual acceptance or policy approval.

## Collection

```bash
python bin/collect_qc.py --directories qc_result --output qc_index
```

This indexes existing reports without changing their decisions.

The collector copies each gate's complete evidence directory to
`<output>/<subject>/<gate>/`, including PNG sidecars and command logs, and
writes relative links. It collects one report per subject/gate; nested
stage reports are retained as evidence rather than counted again.

The bundled renderers provide matched before/after bias-field comparisons,
T1/T2 surface contours, six-view meshes, and native/registered/template
spherical feature comparisons. Native cortical-pair measurements use the
native thickness and cortex labels; fsaverage6 pairs use their own resampled
white/pial geometry and template anatomical reference. Whole-mesh medial-wall
contact is not automatically classified as cortical penetration.

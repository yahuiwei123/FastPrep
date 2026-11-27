# Validation record

## Hemisphere and ribbon parallel execution (2026-11-23)

Nineteen native stages now dispatch independent left/right chains with
isolated command journals and metrics. White/pial placement uses two threads
per concurrent hemisphere, including the GPU backend. One-CPU ordinary
stages and explicit one-worker settings retain serial execution. Shared
volume operations and one-CPU QC jobs retain their existing execution.
The complete suite ran 184 tests in 62.086 seconds: 179 passed and five
optional tests skipped. Coverage includes actual concurrent subprocesses,
intra-hemisphere ordering, isolation of the last log used by topology checks,
nested metric merging, failure propagation with sibling log retention,
resource limits, and one-worker fallback. The ribbon suite was subsequently
rerun after adding the two-CPU rasterizer fallback case: all six tests passed.

Isolated production-node harnesses reused the original upstream inputs for
PT001_ClevelandCCF / sub-01 / ses-01. Eleven parallel nodes matched the
previous serial scientific outputs exactly: tessellation, both cortex-label
nodes, smoothing/inflation, curvature statistics, white Jacobian, cortical
parcellation, percent contrast, parcellation statistics, and white/pial
fsaverage6 resampling. Checks compare coordinates, faces, volume geometry,
morphometry arrays, label records, annotation names/tables and numerical
statistics, including `# Measure` statistics. Output-directory provenance
and nonnumerical command/date headers are excluded; physical geometry is
retained. The shared FreeSurfer `pctsurfcon` internal log is disabled, while
both command streams remain captured separately.

Representative node wall times follow. Serial values are the archived
Nextflow task realtime; parallel values include stage setup, execution,
checks and output handling in the isolated harness. Different system load
and wrapper overhead limit the precision of these comparisons.

| Node | Serial seconds | Parallel seconds |
| --- | ---: | ---: |
| White placement, GPU | 230.243 | 137.961 |
| Pial placement, GPU | 274.848 | 165.801 |
| Smoothing/inflation | 47.089 | 29.273 |
| Cortex label | 26.307 | 15.998 |
| Cortex + hippocampus/amygdala label | 26.312 | 15.665 |
| Cortical parcellation | 22.852 | 15.199 |
| Percent contrast | 8.913 | 6.265 |
| Parcellation statistics | 27.645 | 26.139 |
| White fsaverage6 resampling | 10.262 | 6.104 |
| Pial fsaverage6 resampling | 10.225 | 6.821 |

The two-thread GPU white- and pial-placement runs retained exactly the
archived four-thread GPU coordinates, faces, physical volume geometry,
curvature, area and, for pial, thickness for both hemispheres. All
post-placement gates measured Euler=2, zero defects and zero
self-intersecting vertices. Both hemispheres selected the CUDA backend.
This session had no T2; native two-thread multimodal T2 refinement was not
benchmarked, while its dispatch and post-placement gates remain covered by
the existing portable tests. A separate official CPU
two-thread white-placement run matched the GPU left surface exactly, while
the right surface differed by a maximum 0.0135093 mm (RMS 0.0000697 mm).
This is not an exact CPU/GPU match; the strict `checked` backend's comparison
and CPU fallback policy remain unchanged. No tolerance was relaxed to accept
that comparison. The source copy's native executable permissions were
restored for the placement drivers, distance-field helper and quick sphere;
the archived running snapshot was left intact.

An actual two-thread `checked` T1 pial run also completed all production
geometry gates and exercised its strict comparison: right-hemisphere pial,
targets, rip flags and physical geometry were exact CPU/GPU matches. On the
left, targets, rip flags and physical geometry matched, while pial vertices
differed by maximum 0.296585 mm and RMS 0.000959 mm (95th percentile zero).
The backend consequently selected the official CPU result for the left
hemisphere and CUDA for the right, as required by its existing policy. The
validation harness then deliberately reported the nonexact comparison as a
failure; it is not counted as an exact-match validation. This checked run
took 417.138 seconds because it executed both backends. These CPU/GPU
differences are distinct from the exact match between the new GPU execution
and the archived GPU result; no anatomical interpretation is inferred from
either comparison.

For the same session, the ribbon node fell from 798.362 to 294.388 seconds
(2.71 times faster). Native `mris_volmask --parallel` took 153.728 seconds;
the two cortical-pair subprocesses together took 139.715 seconds of wall
time. All three ribbon volumes had identical grids and zero changed voxels.
All scientific metric fields matched for both hemispheres, excluding timing
and provenance. Vertex containment, all eligible cortical triangle checks,
contact exclusions and numerical tolerances were preserved. The original
unknown/failure findings remain in the candidate metrics.

These are stage-level performance and numerical comparisons. They do not
establish end-to-end subject runtime, anatomical approval, or equivalence for
every dataset. The existing batch snapshot and historical results were not
modified or migrated by this validation.

## Default FireANTs MNI registration (2026-11-23)

New preparations default to FireANTs 1.5.0 affine MI followed by masked-CC Greedy compositive registration on CUDA. ANTs affine/SyN remains an explicit `--mni-backend ants` option; old blueprints lacking a backend field retain ANTs. FireANTs uses a GPU affine-resampled fitting image, then composes the physical affine after the template-grid deformation. Final structural outputs sample each original image once using a full pull displacement, without applying the affine twice. CUDA failures do not select a CPU backend.

The final suite ran 172 tests in 58.257 seconds: 167 passed and 5 optional tests skipped. All eight tests in `test_fireants_register.py` ran, including both actual CUDA integrations on an NVIDIA L40. Coverage includes different anisotropic/oblique image grids and origins, physical affine Jacobian determinants, noncommuting affine/deformation composition, forward/inverse field geometry and landmark closure, a separate rotated companion transform, original companion input selection and single-pass resampling, refusal of CPU/unavailable CUDA, backend selection and inverse nonconvergence. Native ANTs affine/SyN, native FLIRT interoperability, model-bundle and workflow contracts also passed. Optional skips are not claimed as verified. To enable the GPU integrations, set `FASTPREP_TEST_FIREANTS_CUDA=cuda:0` with FireANTs installed in the scientific test interpreter and `antsApplyTransforms` available on PATH.

One real T1w, PT001_ClevelandCCF / sub-01 / ses-01, was registered to MNI152NLin2009cAsym res-01 (193 × 229 × 193). Its software configuration passed the full preparation environment probes. An isolated three-node harness executed the production input selection, empty-companion alignment and MNI `NativeStage` output handling; upstream QC gates, the full Nextflow reconstruction and batch migration were not run. Measured MNI node wall time was 29.061 seconds, with 23.373 seconds for the registration subprocess and 3.334 seconds for final resampling. Other time covers validation and output handling. These are MNI timings, not complete FastPrep subject runtimes.

The real run used CUDA0, PyTorch 2.8.0+cu128 and FireANTs 1.5.0 without optional fused operations; the baseline PyTorch implementation still ran on CUDA. Peak allocated GPU memory was 7,936,071,680 bytes. The full physical LPS pull Jacobian had zero nonpositive voxels across the grid; its template-brain range was 0.437970–1.057498. Jacobians are computed with both source and template physical coordinate frames, rather than treating image-index axes as physical axes. Numerical inverse verification checked 4,129,254 interior native-grid points (52.506% of that grid), with maximum residual 0.000919 mm and RMS 0.000030 mm. Boundary and out-of-FOV points are not included in those residual statistics.

Earlier development candidates failed inverse or Jacobian checks and were retained separately; none was promoted to a successful result. The final output retains its affine, composed forward/inverse fields, physical Jacobian image, backend/device/version/parameters, separate registration/resampling/total timings and `scientific_approval=not_performed`. A sparse three-plane overlay was inspected as an implementation check; comprehensive anatomical/ROI review, full-resolution inverse interpolation validation on this subject, batch generalization and equivalence to ANTs SyN remain unestablished. Changing this source does not change an already running frozen pipeline snapshot; use a new preparation and work/output directories to adopt it.

## MNI template normalization (2026-11-20)

The TemplateFlow default `tpl-MNI152NLin2009cAsym`, `res-01`, was retrieved with its brain mask and verified as two readable volumes on the same 193 × 229 × 193, 1 mm grid. A default prepare on a selected real session resolved this exact template without a template flag.

Native ANTs tests exercised affine + SyN and single-pass resampling of a synthetic T1w and its companion. A noncommuting rotation, displacement and FLIRT translation gave the expected landmark intensities with the production transform order. A separate native FLIRT comparison used different anisotropic grids and opposite handedness. Wrong source/target binding is rejected before registration. Nextflow stub execution assembled all 50 nodes for each of two units. The full suite ran 136 tests (132 passed, 4 optional skips). These are geometry, ordering and execution checks, not a scientific approval of any subject's normalization.

## Session alignment with FSL FLIRT (2026-11-20)

Native FLIRT 6.0, 6 DOF, mutual information for search and fitting (32 bins). A known translation of a synthetic landmark remained correct after changing the handedness and orientation of the reconstruction reference. Legacy ANTs manifests and invalid, reflected or scaled fitted matrices are rejected. The full suite ran 132 tests (128 passed, 4 optional skips).

On the five NKI sessions, three had no companions. For the three real pairs, the before/after QC mutual information (64 bins) was 0.47786/0.54873 (ses-02 T2w), 0.49421/0.55291 (ses-03 T2w), and 0.44989/0.45285 (ses-03 FLAIR). Relative to archived ANTs results, the fixed-to-moving mapping differed by RMS 0.163, 0.235 and 1.693 mm respectively at sampled brain-mask voxel centres. Re-expressing either T2w transform for the reconstruction grid changed sampled physical positions by less than 0.000001 mm. These are transform comparisons, not anatomical ground truth or surface distances; no new pial surfaces were evaluated. Historical results below predate the backend switch.

## Reliable-core BFC revision (2026-11-18)

The updated suite passed 54 tests with 2 optional tests skipped, using the configured scientific Python environment (2026-11-18); no skipped interfaces are claimed as verified.

Portable regressions cover physical-distance per-class erosion, robust within-class log-MAD exclusion of low-intensity WM outliers, exclusion of
CSF/unknown/zero-signal samples, zero soft-label leakage, actual nearest donors,
finite positive fields, unmodeled-region correction without clipping,
constant-tissue identity, invariance to excluded intensities, and stable RBF
normalization with finite gradients and constant far-field interpolation.
Application-domain validation permits unit replacement of unused background
estimates but rejects nonpositive fields anywhere in the brain, including label zero.
Synthetic T1/T2 integration retains geometry, input labels and WM=110, checks
deep-region field application, and rejects overwriting an existing output.
The previous release evidence below is retained, not retroactively applied to
this revised algorithm. Implementation tests do not establish anatomical truth
or whole-brain scientific QC; the adapter is not the unmodified upstream CLI.

Validation applies to the standalone source and model bundle. It does not establish anatomical accuracy on real subjects.

The complete test suite passed: **23 tests**, with the optional real FreeSurfer, QC rendering and FCL/VTK interfaces enabled (78.2 seconds). The tests used existing environments; no dependencies were installed.

## Workflow and native interfaces

- The explicit Nextflow DSL2 graph contains 49 stages, 241 connections and 18 published output contracts. Two topology-only analysis units completed all 49 stages and final assembly using Nextflow 26.04.3. The stub tasks produced no scientific images or approval.
- `nextflow lint` passed for `main.nf`, `modules/stages.nf` and `nextflow.config`.
- The final topology run completed 102 tasks (two analysis units plus shared preflight/collection). An immediate `-resume` reused all 102 cached tasks and submitted zero new tasks.
- Copying the complete release payload to a new directory whose name contains a space passed both the five model-bundle tests and the full two-unit topology run. The copied workflow used its own project root and bundled models.
- Pipeline contract tests cover invalid graph connections, escaped artifact paths, collection assembly with empty stderr logs, refusal to overwrite existing subject results, valid empty Python package files, missing script interpreters, and the use of final T2 pial surfaces for morphometry.
- Real FreeSurfer 7.3.2 `mris_place_surface --curv-map`, `--area-map` and `--thickness` commands passed on two concentric 642-vertex synthetic surfaces. Outputs were finite, with the expected vertex count and a median thickness between 1 and 3 mm. This is an interface/geometry check, not a cortical reconstruction benchmark.
- All six configured scientific Python environments loaded their required modules. ANTs reported 2.6.5, and `mri_convert` reported FreeSurfer 7.3.2. Resolution of 45 command recipes and four template resources succeeded before script-interpreter validation was added. The full environment probe then stopped at `recon-all`: this host lacks `/bin/tcsh`. Preparation now detects that missing interpreter explicitly. No system packages were installed.

## Synthetic tissue-guided BFC and BIDS selection

The portable suite `tests/test_inputs_and_bias.py` was run with the existing `mars` Python 3.11 environment using `unittest`; all four tests passed. The suite generated temporary BIDS and MGH data locally and exercised two 64³ synthetic bias-field cases, one each for T1 and T2.

For both modalities the test verified that the white-matter coefficient of variation (`standard deviation / mean`) was lower after correction. The test did not print or retain the before/after numeric CV values, so this record reports the measured direction of change only. It also verified finite output data and affine/voxel geometry, zero values outside the brain mask, WM median intensity 110, and unchanged input labels.

The BIDS cases covered filtering to requested subjects and multiple sessions, explicit T1/T2 selection, rejection of unselected multiple T1/T2 acquisitions, and rejection of a T2 selection from another session or when T2 use is disabled.

## QC and surface-pair environment evidence

The existing `macapipe` environment loaded VTK 9.6.2 and surfplot 0.2.0. The existing `sugar-source` environment loaded FCL/python-fcl 0.7.0.11, trimesh 4.12.2, and rtree 1.4.1. The pipeline uses these in separate processes: the geometry profile performs the FCL work, and the QC/render profile provides the VTK child process. The `mars` environment lacks these packages.

Actual CLI integration tests generated matched bias-correction comparisons, anatomical white/pial contours, six-view surface images, and native/registered/template spherical-feature figures with PNG metadata sidecars. Full cortical-pair measurement passed on concentric synthetic surfaces: no white vertices outside pial, no cortical triangle intersections, and no negative supplied thickness. Its reported QC status correctly remained `unknown` pending review.

Artifact tests verified collection primary manifests, member paths, copied command logs, strict-mode blocking, and self-contained collected HTML reports with retained sidecars. Nested stage reports were not counted as duplicate gate reports.

## Limits

No complete real-subject reconstruction, T2 versus T1 anatomical accuracy comparison, model inference benchmark, GPU benchmark, or clean-machine Conda installation was run for this release. The deployment recipes require target-machine verification, especially native SUGAR and FCL dependencies. Input structural checks and generated QC images never constitute scientific approval.

## BFC toolkit and default RBF integration (2026-11-18)

This amendment concerns `bin/tissue_bfc/` version 1.0.0, the unchanged reliable-core-v2 sampling recipe, and the new default RBF wiring. It does not rewrite the historical limits or prior failed scientific attempts.

Final run on 2026-11-18: `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 FASTPREP_TEST_BFC_CUDA=cuda:0 python -B -m unittest discover -s fastprep-anat/tests -v` ran 60 tests in 75.878 seconds: 58 passed and 2 skipped. The RBF test executed both synthetic T1 and T2 on CUDA0 (NVIDIA L20); it was not skipped. The skipped tests were the optional native FreeSurfer interface and real QC-renderer integration suites. The final log is `docs/bfc-toolkit-tests-20261118.log`.

- Both upstream sources are included unchanged. The standalone archive can be extracted to a different directory and its Gaussian CLI can run a synthetic volume without `reference_runtime.py`, Nextflow, an external source path or a FreeSurfer installation.
- T1 and optional T2 both default to RBF. `prepare.py --bfc-method gaussian` explicitly changes both. The Nextflow device defaults to `cuda:0`; a RBF CUDA request fails rather than falling back to CPU.
- The real CUDA integration test runs synthetic T1 and T2 with the configured `FASTPREP_TEST_BFC_CUDA` device. It checks output geometry, finite/nonnegative intensities, preserved labels and excluded deep signals, bias reduction, WM110 calibration, and the exact reciprocal relationship between the saved RBF divisor and published multiplication factor. These are interface/numerical tests, not anatomical approval.
- Toolkit tests also cover default dispatch without an explicit method, unavailable CUDA, visible device indexing, standalone package contents and overwrite refusal. The existing optional native FreeSurfer and full QC renderer integration tests remain separate and are skipped unless their environment variables are configured.
- The prior real-data repair comparison remains limited to one T1 session. Its Gaussian candidate failed local image QC; RBF reduced the old hotspot and label-shaped seam but is not fully approved for all brain regions or real T2 data. No new full anatomical run or batch replacement is implied by changing defaults. See `docs/BIAS_CORRECTION.md` for the diagnosis and limitations.

### Local post-placement pial intersection recovery

Pial placement can finish with residual self intersections at the boundary of
cortex and vertices pinned to white, including with official CPU placement.
After validating vertex/face identity and volume geometry, the runtime permits
one `mris_remove_intersection` attempt for pial only, if Euler is already 2 and
the defect index is zero. White placement and invalid topology remain strict.
Both the initial T1 pial and any T2-refined pial pass the same final gate before
curvature, area, thickness, ribbon or resampling is computed.

Recovery is deliberately local: at most 32 intersecting/changed vertices,
maximum displacement 3 mm overall and 1 mm on cortex. These are engineering
limits on automatic edits, not anatomical QC acceptance thresholds. A missing
measurement, exceeded limit, changed indexing/geometry or residual intersection
still fails the stage. Cleanup can move affected medial-wall vertices that were
previously pinned to white; this is recorded, never described as preserving the
pinning. Clean pial outputs do not run cleanup.

Raw and candidate surfaces, changed vertex IDs/displacements and recovery JSON
are retained as members of the pial refinement provenance and published with
the final outputs. Full cortical-pair and anatomical QC remain required.
Zero self intersections do not establish white/pial containment or scientific
approval; `report_only` continues to retain failed and unknown QC findings.

### Fixed medial-wall boundary: validation and authorized rollout

The optional `medial_wall_strategy` modes `fixed-during-placement` and
`fixed-on-intersection` keep noncortical vertices at white throughout native
pial positioning by using the cortex label for both activity and pinning.
Inputs and outputs must preserve exact fixed coordinates, mesh correspondence,
finite coordinates and volume geometry. Initial smoothing is disallowed because
native presmoothing precedes ripping. A T2 recovery first regenerates pinned T1
pial. Failed originals, candidates, commands and displacement measurements remain
in provenance; unsupported topology and missing measurements never trigger this
recovery. The normal path remains unchanged when no intersections are detected.

On PT023_Infant_Development_Brain/sub-001 and PT024_M4Raw/sub-001, native CPU
trials and CUDA stage replays reduced right-pial self-intersecting vertex counts
from 10 and 6 to zero. Euler remained 2, with zero defect index and exact fixed
medial coordinates. PT023 CPU/CUDA placement was exact in both hemispheres.
PT024 was not exact: maximum pial coordinate distance was 0.00866 mm on the left
and 0.59398 mm on the right; checked mode retained its official CPU fallback.

These candidates are **not qualified for batch deployment**. The right-sided
white/pial crossing-face-pair count increased from 36 to 78 on PT023 and from
474 to 595 on PT024; outside white vertex counts changed from 2 to 13 and from
25 to 41, respectively. The original pial meshes have self intersections, so
their containment is a diagnostic baseline, not a valid anatomical reference.
The initial validation kept this candidate disabled. Subsequently, on 2026-11-24,
the user explicitly requested continuation of affected failures and use of the
latest method for remaining subjects after these regressions were disclosed.
New workflow defaults now select `fixed-on-intersection`; the batch continues in
the previously authorized `report_only` mode. This operational authorization does
not qualify these meshes anatomically or change their recorded QC failures.
Self-intersection, topology, correspondence and geometry gates are retained.

A pial-only task override and isolated helper directory support resuming old
Nextflow sessions while retaining their original upstream inputs and cache
validation. Reused upstream work is not included in the new resume segment's
wall-clock time; resumed subjects remain excluded from uninterrupted runtimes.

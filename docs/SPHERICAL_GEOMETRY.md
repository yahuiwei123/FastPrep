# Spherical orientation and limited repair

In strict mode, FastPrep accepts only serialized, consistently oriented, closed genus-zero
spheres with no reversed or degenerate triangles. Coordinates are evaluated in
float64 after float32 quantization and again after reading the saved surface.
Every face must also exceed a derivative-based float32 roundoff margin. The
spherical mapping degree must be one, and faces spanning a nonlocal hemisphere
are rejected. In the default `report_only` mode, bounded repair is attempted.
If a finite, corresponding sphere still exceeds the quality bounds, the pipeline
retains the computed coordinates, measures the serialized faces, and records
the finding in the stage manifest and published `quality_warnings`.
Template interpolation still requires valid source triangles and query coverage. They do not certify anatomical correctness or sulcal alignment.

## Prevention at the source

1. After initialization intersection cleanup and before white placement, a dual
   adjacency graph detects inconsistent face winding. At most 64 misoriented
   face rows may be corrected, preserving their three vertex IDs, row identity,
   all coordinates and volume metadata. The original surface is retained as
   `.before-winding`. The orientation decision does not depend on spherical
   coordinate signs, so this cannot disguise a geometric fold. All downstream
   white, pial, inflated and sphere meshes inherit the same oriented faces.
2. Native `mris_sphere` output is retained as `.sphere.raw`, checked against the
   inflated surface's triangle identity, and validated or locally repaired.
3. SUGAR rigid rotation is applied in float64. A single global rotation preserves
   orientation mathematically, but almost degenerate faces can reverse during
   float32 coordinate arithmetic or storage. Input conditioning and final file
   validation handle this explicitly.
4. Nonlinear model output is retained and checked on its template mesh before
   transfer to native vertices. Transfer uses barycentric coordinates of actual
   containing mesh triangles, computed in float64 with adjacency walking.
   Unmatched points fail; nearest-three-vertices extrapolation is not allowed.
5. Native output is checked again. A continuous template map can still yield
   folded thin triangles when sampled onto another triangulation whose edges
   cross template edges. Limited coordinate repair handles those residuals.

## Coordinate repair limits

The constrained optimizer minimizes tangent displacement, moving only vertices
belonging to unsafe triangles and keeping all other vertices fixed. All faces
incident to movable vertices are constrained, including neighboring initially
valid faces. Vertex order, triangle rows and connectivity are unchanged.

Default limits are 64 unsafe faces, 192 movable vertices, 100 optimizer
iterations, and 0.5 mm maximum chord displacement on a radius-100 mm sphere.
This is a spherical coordinate bound, not an anatomical white/pial displacement.
Final serialized geometry, rather than an optimizer success flag, determines
acceptance. Extensive defects, wrong topology, identity changes, excessive
displacement or remaining unsafe faces fail strict validation. Raw model outputs, repair metrics
and file-readback metrics are retained.

## Real-data validation, 2026-11-26

Three failed subjects were replayed through real CUDA rigid and nonlinear
registration, independently for both hemispheres. All six saved outputs have
zero reversed, zero degenerate and zero unsafe faces, with degree one.

| Case | Hemisphere | Native nonlinear flips before local repair | After | Maximum local movement on radius-100 sphere |
| --- | --- | ---: | ---: | ---: |
| ds001132 | left | 31 | 0 | 0.0303 mm |
| ds001132 | right | 12 | 0 | 0.2900 mm |
| ds001228 | left | 19 | 0 | 0.0697 mm |
| ds001228 | right | 26 | 0 | 0.3011 mm |
| ds000113 | left | 23 | 0 | 0.1506 mm |
| ds000113 | right | 31 | 0 | 0.0111 mm |

For ds001132 left, the original template output had no flips; the previous
nearest-three interpolation introduced 68. Containing-triangle interpolation
reduced this to 31 before bounded local repair. The original rigid failure was
separately reproduced as float32 rotation of an extremely thin native face.
For ds001228 and ds000113, a single wrong face winding was traced back to the
topology-correction output; it was already present before white and sphere.

FreeSurfer `mris_info --intersections` still marked 30, 6 and 6 vertices on three
of the six repaired spheres. These are marked vertices, not flipped-face counts.
An independent all-candidate separating-axis audit found no nonadjacent triangle
intersections in any of the six files. Every overlapping bounding-box candidate
had a strict separating plane; the smallest separation over the six meshes was
0.000241 mm. For the four pairs flagged by the bundled conservative FreeSurfer
triangle predicate, strict separation was additionally certified with exact
rational arithmetic on the stored coordinates. Native checker reports are
retained; they are not rewritten to zero or treated as a cortical surface gate.

The replay is a geometry and implementation check, not full anatomical approval.
Native/registered/fixed six-view sulcal montages are recorded separately.

## Resuming older attempts

Use `--sphere_helpers /path/to/new-snapshot/bin` for native sphere generation and
SUGAR registration. This also binds the matching registration wrapper, while
retaining the recorded interpreter, model weights and template identity.
Use `--intersection_helpers /path/to/new-snapshot/bin` to recompute winding
correction before white/pial. Retain upstream tessellation and topology cache;
recompute affected descendants. Old Nextflow entrypoints must support the helper
override. Do not repair only the face ordering of an exported sphere while
leaving its associated cortical meshes inconsistent.

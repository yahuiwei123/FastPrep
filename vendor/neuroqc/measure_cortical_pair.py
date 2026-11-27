"""Measure matching FreeSurfer white/pial, native thickness and rasterized ribbon.

This reports measurements, never a QC approval. Normal displacement is not
cortical thickness; ribbon/aseg overlap has no implicit acceptance threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import nibabel as nib
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy


def polydata(vertices, faces):
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(vertices, dtype=np.float64), deep=True))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(
        np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel(), deep=True))
    mesh = vtk.vtkPolyData()
    mesh.SetPoints(points)
    mesh.SetPolys(cells)
    return mesh


def measure(white, pial, thickness, cortex_indices, *, tolerance_mm=1e-6, progress=False):
    started = phase_started = time.perf_counter()
    timings = {}

    def finish_phase(name):
        nonlocal phase_started
        finished = time.perf_counter()
        timings[name] = finished - phase_started
        phase_started = finished
        if progress:
            print(json.dumps(dict(phase=name, seconds=timings[name])), file=sys.stderr, flush=True)

    wv, wf = white
    pv, pf = pial
    for vertices, faces in (white, pial):
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError('finite 3D vertices required')
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError('nonempty triangle faces required')
        if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('invalid face indices')
    if wv.shape != pv.shape or not np.array_equal(wf, pf):
        raise ValueError('white/pial vertex and face correspondence required')
    if thickness.shape != (len(wv),) or not np.isfinite(thickness).all():
        raise ValueError('finite native thickness for every vertex required')
    ids = np.asarray(cortex_indices)
    if ids.ndim != 1 or not len(ids) or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('nonempty cortex vertex indices required')
    if ids.min() < 0 or ids.max() >= len(wv) or len(np.unique(ids)) != len(ids):
        raise ValueError('invalid or duplicate cortex vertex indices')
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0:
        raise ValueError('nonnegative finite numerical tolerance required')
    finish_phase('input_validation')

    outer = polydata(pv, pf)
    boundaries = vtk.vtkFeatureEdges()
    boundaries.SetInputData(outer)
    boundaries.BoundaryEdgesOn()
    boundaries.NonManifoldEdgesOn()
    boundaries.FeatureEdgesOff()
    boundaries.ManifoldEdgesOff()
    boundaries.Update()
    if boundaries.GetOutput().GetNumberOfCells():
        raise ValueError('pial must be closed and manifold for containment')
    finish_phase('pial_mesh_validation')
    # Check every cortical vertex, with no sampling. VTK containment does not
    # depend on the face-normal orientation. Exact coincident vertices and
    # nearest-face distances only explain numerical contact, not anatomical slack.
    queries = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(np.ascontiguousarray(wv[ids], dtype=np.float64), deep=True))
    queries.SetPoints(points)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(queries)
    enclosed.SetSurfaceData(outer)
    enclosed.SetTolerance(0.0)
    enclosed.Update()
    inside = vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray('SelectedPoints')).astype(bool)
    finish_phase('cortex_vertex_containment')
    coincident = np.linalg.norm(pv[ids] - wv[ids], axis=1) <= tolerance_mm
    on_surface = coincident.copy()
    distance = vtk.vtkImplicitPolyDataDistance()
    distance.SetInput(outer)
    candidates = np.flatnonzero(~inside & ~coincident)
    for i in candidates:
        on_surface[i] = abs(distance.EvaluateFunction(wv[ids[i]])) <= tolerance_mm
    contained = inside | on_surface
    scalar = thickness[ids]
    outside_ids = ids[~contained]
    finish_phase('numerical_contact_distances')
    cortical = np.zeros(len(wv), dtype=bool)
    cortical[ids] = True
    contact = np.linalg.norm(pv - wv, axis=1) <= tolerance_mm
    # Pinned white/pial contact is expected outside cortex. Check the full
    # pial against white faces wholly in noncontact cortex; keep the excluded
    # boundary faces explicit for anatomical review.
    face_mask = cortical[wf].all(axis=1) & ~contact[wf].any(axis=1)
    candidate_ids = np.flatnonzero(face_mask)
    finish_phase('intersection_mesh_preparation')
    crossing_pairs = []
    if len(candidate_ids):
        collisions = vtk.vtkCollisionDetectionFilter()
        collisions.SetInputData(0, polydata(wv, wf[face_mask]))
        collisions.SetInputData(1, outer)
        for i in (0, 1):
            collisions.SetTransform(i, vtk.vtkTransform())
        collisions.SetCollisionModeToAllContacts()
        collisions.SetBoxTolerance(0.0)
        collisions.SetCellTolerance(0.0)
        collisions.Update()
        count = collisions.GetNumberOfContacts()
        if count:
            white_cells = vtk_to_numpy(collisions.GetContactCells(0))
            pial_cells = vtk_to_numpy(collisions.GetContactCells(1))
            crossing_pairs = np.column_stack((candidate_ids[white_cells], pial_cells)).tolist()
    finish_phase('triangle_pair_intersections')
    report = {
        'vertex_correspondence': True,
        'vertex_count': len(wv), 'cortex_vertex_count': len(ids),
        'negative_thickness_vertices': int((scalar < 0).sum()),
        'zero_thickness_vertices': int((scalar == 0).sum()),
        'all_vertices_negative_thickness': int((thickness < 0).sum()),
        'all_vertices_zero_thickness': int((thickness == 0).sum()),
        'thickness_percentiles_mm': dict(zip(('min', 'p1', 'p5', 'p50', 'p95', 'p99', 'max'),
                                            np.percentile(scalar, [0, 1, 5, 50, 95, 99, 100]).tolist())),
        'thickness_method': 'native mris_place_surface --thickness; supplied scalar unchanged',
        'white_inside_pial_fraction': float(contained.mean()),
        'white_outside_pial_vertices': int(len(outside_ids)),
        'white_outside_pial_vertex_ids': outside_ids.tolist(),
        'cortex_contact_vertices': int(on_surface.sum()),
        'noncortex_vertex_count': int(len(wv) - len(ids)),
        'containment_method': 'vtkSelectEnclosedPoints on every cortex vertex, plus numerical surface contact',
        'numerical_contact_tolerance_mm': tolerance_mm,
        'cortical_triangle_pair_intersections': len(crossing_pairs),
        'intersecting_face_pairs': crossing_pairs,
        'intersection_method': 'vtkCollisionDetectionFilter: every noncontact cortex white triangle against full pial',
        'checked_cortex_white_faces': int(len(candidate_ids)),
        'excluded_boundary_or_contact_white_faces': int(len(wf) - len(candidate_ids)),
        'missing_checks': ['pinned_and_boundary_contact_visual_review', 'visual_review',
                           'protocol_thickness_and_coverage_limits'],
    }
    finish_phase('summary_statistics')
    timings['measure_total'] = time.perf_counter() - started
    report['timings_seconds'] = timings
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('white', 'pial', 'thickness', 'cortex-label', 'ribbon', 'aseg', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--hemi', choices=('lh', 'rh'), required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    white = nib.freesurfer.read_geometry(str(args.white))
    pial = nib.freesurfer.read_geometry(str(args.pial))
    thickness = nib.freesurfer.read_morph_data(str(args.thickness))
    cortex = nib.freesurfer.read_label(str(args.cortex_label))
    input_seconds = time.perf_counter() - started
    report = measure(white, pial, thickness, cortex, progress=True)
    coverage_started = time.perf_counter()
    ribbon, aseg = nib.load(args.ribbon), nib.load(args.aseg)
    if ribbon.shape != aseg.shape or not np.allclose(ribbon.affine, aseg.affine, atol=1e-5, rtol=0):
        raise ValueError('ribbon and aseg must have matching physical grids')
    rd, ad = np.asanyarray(ribbon.dataobj), np.asanyarray(aseg.dataobj)
    if not np.isfinite(rd).all() or not np.isfinite(ad).all():
        raise ValueError('finite ribbon and aseg required')
    label = 3 if args.hemi == 'lh' else 42
    predicted, reference = rd == label, ad == label
    n_pred, n_ref = int(predicted.sum()), int(reference.sum())
    overlap = int((predicted & reference).sum())
    report['ribbon_coverage'] = {
        'method': 'same-grid native ribbon overlap with aseg cortical gray label; descriptive, no reference truth',
        'cortex_label': label, 'ribbon_voxels': n_pred, 'aseg_cortex_voxels': n_ref,
        'overlap_voxels': overlap,
        'aseg_cortex_covered_fraction': overlap / n_ref if n_ref else None,
        'ribbon_within_aseg_cortex_fraction': overlap / n_pred if n_pred else None,
    }
    report['hemi'] = args.hemi
    report['timings_seconds'].update(read_surfaces=input_seconds,
        ribbon_coverage=time.perf_counter() - coverage_started, worker_total=time.perf_counter() - started)
    report['status'] = 'unknown'
    report['provenance'] = {
        name: {'path': str(getattr(args, name).resolve())}
        for name in ('white', 'pial', 'thickness', 'cortex_label', 'ribbon', 'aseg')
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

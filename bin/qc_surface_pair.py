"""Full-vertex FCL/VTK evidence; unsigned distances are not cortical thickness."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def vtk_geometry(inner_vertices, inner_faces, outer_vertices, outer_faces):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

    def polydata(coordinates, faces=None):
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.asarray(coordinates, dtype=np.float64), deep=True))
        surface = vtk.vtkPolyData()
        surface.SetPoints(points)
        if faces is not None:
            cells = vtk.vtkCellArray()
            indices = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).astype(np.int64).reshape(-1)
            cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(indices, deep=True))
            surface.SetPolys(cells)
        return surface

    inner, outer = polydata(inner_vertices, inner_faces), polydata(outer_vertices, outer_faces)
    vtk.vtkMath.RandomSeed(1234)
    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetInputData(polydata(inner_vertices))
    enclosed.SetSurfaceData(outer)
    enclosed.SetTolerance(0.0)
    enclosed.CheckSurfaceOn()
    enclosed.Update()
    selected = vtk_to_numpy(enclosed.GetOutput().GetPointData().GetArray("SelectedPoints"))
    if len(selected) != len(inner_vertices):
        raise ValueError("Containment did not evaluate every inner vertex")
    locator = vtk.vtkStaticCellLocator()
    locator.SetDataSet(inner)
    locator.BuildLocator()
    closest = [0.0, 0.0, 0.0]
    cell_id, sub_id, distance_squared = vtk.reference(0), vtk.reference(0), vtk.reference(0.0)
    cell = vtk.vtkGenericCell()
    distances = np.empty(len(outer_vertices))
    for index, coordinates in enumerate(outer_vertices):
        locator.FindClosestPoint(coordinates, closest, cell, cell_id, sub_id, distance_squared)
        if int(cell_id) < 0:
            raise ValueError("Missing nearest triangle for an outer vertex")
        distances[index] = float(distance_squared) ** .5
    if not np.isfinite(distances).all():
        raise ValueError("Nonfinite full-vertex triangle distances")
    return {"innerOutsideVertexCount": int((selected == 0).sum()),
            "innerVerticesEvaluated": len(inner_vertices), "outerVerticesEvaluated": len(outer_vertices),
            "medianInnerDistance": float(np.median(distances)), "p99InnerDistance": float(np.quantile(distances, .99)),
            "containmentMethod": "vtkSelectEnclosedPoints; all inner vertices; tolerance=0; seed=1234",
            "distanceMethod": "vtkStaticCellLocator; every outer vertex to nearest inner triangle",
            "vtkVersion": vtk.vtkVersion.GetVTKVersion(), "sampling": "none"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inner", required=True)
    parser.add_argument("--outer", required=True)
    parser.add_argument("--coordinate-unit", required=True, choices=["mm", "meter", "micron"])
    parser.add_argument("--geometry-python")
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--thickness")
    parser.add_argument("--cortex-label")
    parser.add_argument("--ribbon")
    parser.add_argument("--aseg")
    parser.add_argument("--hemi", choices=["lh", "rh"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    full_inputs = (args.thickness, args.cortex_label, args.ribbon, args.aseg, args.hemi)
    if any(full_inputs) and not all(full_inputs):
        parser.error("full cortical-pair evidence requires thickness, cortex-label, ribbon, aseg and hemi")
    if all(full_inputs):
        if not args.geometry_python:
            parser.error("full cortical-pair evidence requires --geometry-python")
        import neuroqc
        helper = Path(neuroqc.__file__).resolve().parent / "measure_cortical_pair.py"
        output = Path(args.output)
        geometry_output = output.with_name(output.stem + "-geometry.json")
        base_command = [sys.executable, "-B", str(Path(__file__).resolve()),
                        "--inner", args.inner, "--outer", args.outer,
                        "--coordinate-unit", args.coordinate_unit,
                        "--geometry-python", args.geometry_python, "--output", str(geometry_output)]
        geometry_stdout, geometry_stderr = output.with_suffix(".geometry.stdout"), output.with_suffix(".geometry.stderr")
        with geometry_stdout.open("w", encoding="utf-8") as out_stream, geometry_stderr.open("w", encoding="utf-8") as err_stream:
            geometry_result = subprocess.run(base_command, stdout=out_stream, stderr=err_stream,
                                             env=os.environ, check=False)
        if geometry_result.returncode:
            raise RuntimeError("FCL/VTK surface geometry failed; see " + str(geometry_stderr))
        cortical_output = output.with_name(output.stem + "-cortical.json")
        command = [args.geometry_python, "-B", str(helper), "--white", args.inner, "--pial", args.outer,
                   "--thickness", args.thickness, "--cortex-label", args.cortex_label,
                   "--ribbon", args.ribbon, "--aseg", args.aseg, "--hemi", args.hemi,
                   "--output", str(cortical_output)]
        stdout, stderr = output.with_suffix(".measure.stdout"), output.with_suffix(".measure.stderr")
        with stdout.open("w", encoding="utf-8") as out_stream, stderr.open("w", encoding="utf-8") as err_stream:
            result = subprocess.run(command, stdout=out_stream, stderr=err_stream, env=os.environ, check=False)
        if result.returncode:
            raise RuntimeError("Full cortical-pair measurement failed; see " + str(stderr))
        geometry = json.loads(geometry_output.read_text(encoding="utf-8"))
        cortical = json.loads(cortical_output.read_text(encoding="utf-8"))
        cortical["geometry_pair"] = geometry
        if geometry.get("status") == "fail" or cortical.get("white_outside_pial_vertices", 0) \
                or cortical.get("cortical_triangle_pair_intersections", 0) \
                or cortical.get("negative_thickness_vertices", 0):
            cortical["status"] = "fail"
        output.write_text(json.dumps(cortical, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
        return
    import numpy as np
    from neuroqc.evidence import metric_result, provenance, write_json
    from neuroqc.images import UNIT_TO_MM
    from neuroqc.surfaces import read_surface, validate_mesh

    inner_vertices, inner_faces = read_surface(args.inner)
    outer_vertices, outer_faces = read_surface(args.outer)
    validate_mesh(inner_vertices, inner_faces)
    validate_mesh(outer_vertices, outer_faces)
    scale = UNIT_TO_MM[args.coordinate_unit]
    inner_vertices, outer_vertices = np.asarray(inner_vertices) * scale, np.asarray(outer_vertices) * scale
    if args.geometry_only:
        write_json(args.output, vtk_geometry(inner_vertices, inner_faces, outer_vertices, outer_faces))
        return
    import trimesh
    import fcl

    inner = trimesh.Trimesh(vertices=inner_vertices, faces=inner_faces, process=False)
    outer = trimesh.Trimesh(vertices=outer_vertices, faces=outer_faces, process=False)
    closed = bool(inner.is_watertight and outer.is_watertight)
    metrics = {"closedSurfacePair": closed, "nestingCheck": "FCL triangle collision and VTK full-vertex containment",
               "surfaceIntersection": None, "distanceUnit": "mm", "innerOutsideVertexCount": None,
               "medianInnerDistance": None, "p99InnerDistance": None, "vertexCorrespondence": None,
               "negativeThicknessVertices": None, "sampling": "none"}
    child = None
    if closed:
        collision = fcl.CollisionResult()
        inner_object = fcl.CollisionObject(trimesh.collision.mesh_to_BVH(inner))
        outer_object = fcl.CollisionObject(trimesh.collision.mesh_to_BVH(outer))
        fcl.collide(outer_object, inner_object, fcl.CollisionRequest(), collision)
        metrics["surfaceIntersection"] = bool(collision.is_collision)
        if not args.geometry_python:
            raise ValueError("A verified VTK-native interpreter is required")
        output = Path(args.output)
        geometry = output.with_name(output.stem + "-vtk.json")
        command = [args.geometry_python, "-B", str(Path(__file__).absolute()), "--geometry-only",
                   "--inner", args.inner, "--outer", args.outer, "--coordinate-unit", args.coordinate_unit,
                   "--output", str(geometry)]
        stdout, stderr = output.with_suffix(".vtk.stdout"), output.with_suffix(".vtk.stderr")
        with stdout.open("w") as out_stream, stderr.open("w") as err_stream:
            result = subprocess.run(command, stdout=out_stream, stderr=err_stream, env=os.environ)
        child = {"argv": command, "exit_code": result.returncode, "stdout": stdout.name, "stderr": stderr.name}
        if result.returncode:
            raise RuntimeError("VTK full-vertex worker failed; see " + str(stderr))
        metrics.update(json.loads(geometry.read_text()))
    failed = metrics["surfaceIntersection"] is True or bool(metrics["innerOutsideVertexCount"])
    record = metric_result(metrics, failed=failed, metric_status="fail" if failed else "pass" if closed else "unknown",
                           missing=["vertex_correspondence", "signed_cortical_thickness", "cortical_contact_localization", "ribbon_coverage", "visual_review"])
    record["provenance"] = provenance({"inner": args.inner, "outer": args.outer},
                   {"coordinate_unit": args.coordinate_unit, "backend": "FCL/VTK full-vertex geometry/1",
                    "containment_backend_changed_from_trimesh": True,
                    "boundary_contact_classification_requires_review": True})
    record["provenance"].update(command=[sys.executable, *sys.argv], geometry_worker=child)
    write_json(args.output, record)


if __name__ == "__main__":
    main()

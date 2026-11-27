"""Read surface geometry without changing coordinates, vertices or faces."""
from pathlib import Path
import nibabel as nib
import numpy as np

from .images import UNIT_TO_MM, affine_mm


def read_surface(path):
    path = Path(path)
    if path.suffix.lower() == '.ply':
        import trimesh
        mesh = trimesh.load_mesh(path, process=False)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    else:
        vertices, faces = nib.freesurfer.read_geometry(str(path))
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f'Surface needs nonempty 3D vertices: {path}')
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f'Surface needs nonempty triangular faces: {path}')
    return vertices, faces


def validate_mesh(vertices, faces):
    if not np.isfinite(vertices).all():
        raise ValueError('Surface vertices must be finite')
    if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Invalid triangular face indices')


def mesh_voxels(path, image, canonical, space, surface_reference=None):
    """Map mm surface coordinates to the displayed volume's voxel coordinates.

    A tkr surface needs the original FreeSurfer MGH/MGZ reference geometry.
    A NIfTI with similar dimensions is not sufficient evidence of that geometry.
    """
    vertices, faces = read_surface(path)
    validate_mesh(vertices, faces)
    if space == 'freesurfer-tkr':
        reference = nib.load(str(surface_reference)) if surface_reference is not None else image
        if not isinstance(reference, nib.MGHImage):
            raise ValueError('freesurfer-tkr requires an MGH/MGZ image or --surface-reference pointing to the original MGH/MGZ volume')
        voxels = nib.affines.apply_affine(np.linalg.inv(reference.header.get_vox2ras_tkr()), vertices)
        ras = nib.affines.apply_affine(affine_mm(reference), voxels)
    elif space == 'scanner-ras':
        if surface_reference is not None:
            raise ValueError('surface_reference is only used with freesurfer-tkr')
        ras = vertices
    else:
        raise ValueError(f'Unknown surface coordinate space: {space}')
    return nib.affines.apply_affine(np.linalg.inv(affine_mm(canonical)), ras), faces


def segments(vertices, faces, z):
    """Intersect triangles with an axial voxel plane, preserving x/y coordinates."""
    triangles = vertices[faces]
    zv = triangles[:, :, 2]
    triangles = triangles[(zv.min(1) <= z) & (zv.max(1) >= z) & (np.ptp(zv, axis=1) > 1e-6)]
    result = []
    for triangle in triangles:
        points = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[i], triangle[j]
            if abs(a[2] - z) <= 1e-8:
                points.append(a[:2])
            if (a[2] - z) * (b[2] - z) < 0:
                points.append((a + (z - a[2]) / (b[2] - a[2]) * (b - a))[:2])
        if points:
            unique = np.unique(np.asarray(points), axis=0)
            if len(unique) == 2 and np.linalg.norm(unique[0] - unique[1]) > 1e-6:
                result.append(unique)
    return np.asarray(result) if result else np.empty((0, 2, 2))


def nesting_metrics(outer_vertices, outer_faces, inner_vertices, inner_faces, *, coordinate_unit=None):
    """Test nesting geometrically when white/pial have independent topology.

    Return (geometrically_nested, measurements), NOT a WHITE_PIAL rule decision.
    Distances are unsigned nearest-triangle distances, not cortical thickness.
    Self-intersections and vertex correspondence are not tested.
    """
    import trimesh
    import fcl
    validate_mesh(outer_vertices, outer_faces)
    validate_mesh(inner_vertices, inner_faces)
    if coordinate_unit is not None and coordinate_unit not in UNIT_TO_MM:
        raise ValueError('coordinate_unit must be mm, meter, micron, or None')
    factor = UNIT_TO_MM.get(coordinate_unit, 1.0)
    outer_vertices = np.asarray(outer_vertices) * factor
    inner_vertices = np.asarray(inner_vertices) * factor
    outer = trimesh.Trimesh(vertices=outer_vertices, faces=outer_faces, process=False)
    inner = trimesh.Trimesh(vertices=inner_vertices, faces=inner_faces, process=False)
    closed = bool(outer.is_watertight and inner.is_watertight)
    metrics = {'closedSurfacePair': closed, 'nestingCheck': 'triangle_collision_and_vertex_containment',
               'distanceMethod': 'outer_vertex_to_inner_triangle', 'surfaceIntersection': None,
               'distanceUnit': 'mm' if coordinate_unit is not None else 'unknown',
               'innerOutsideVertexCount': None, 'medianInnerDistance': None, 'p99InnerDistance': None,
               'vertexCorrespondence': None, 'negativeThicknessVertices': None}
    if not closed:
        return False, metrics
    # CollisionManager treats convex meshes as solids and mislabels nesting.
    # Explicit triangle BVHs test surface contact instead of volume overlap.
    outer_object = fcl.CollisionObject(trimesh.collision.mesh_to_BVH(outer))
    inner_object = fcl.CollisionObject(trimesh.collision.mesh_to_BVH(inner))
    collision = fcl.CollisionResult()
    fcl.collide(outer_object, inner_object, fcl.CollisionRequest(), collision)
    intersects = bool(collision.is_collision)
    outside = sum(int(np.count_nonzero(~outer.contains(inner_vertices[start:start + 2048])))
                  for start in range(0, len(inner_vertices), 2048))
    distance = np.concatenate([trimesh.proximity.closest_point(inner, outer_vertices[start:start + 2048])[1]
                               for start in range(0, len(outer_vertices), 2048)])
    median, p99 = float(np.median(distance)), float(np.quantile(distance, .99))
    metrics.update(surfaceIntersection=intersects, innerOutsideVertexCount=outside,
                   medianInnerDistance=median, p99InnerDistance=p99)
    nested = bool(not intersects and outside == 0 and np.isfinite(distance).all())
    return nested, metrics

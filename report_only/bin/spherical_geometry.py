"""Report spherical quality findings and preserve finite, corresponding native outputs."""
from pathlib import Path
import json
import numpy as np


class SphericalQualityError(ValueError):
    """A measurable defect in an otherwise addressable triangular mesh."""


def check_indices(faces, count):
    faces = np.asarray(faces)
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or faces.min() < 0 or faces.max() >= count):
        raise ValueError('Invalid spherical face indices')


def unit_vertices(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError('Invalid spherical vertices')
    radius = np.linalg.norm(xyz, axis=1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or (radius == 0).any():
        raise ValueError('Invalid spherical vertices')
    if not np.allclose(radius, 100., atol=.01, rtol=0):
        raise ValueError('Expected a radius-100 sphere')
    return xyz / radius[:, None]


def orientation(unit, faces):
    a, b, c = unit[faces].transpose(1, 0, 2)
    signed = np.einsum('ij,ij->i', a, np.cross(b - a, c - a))
    epsilon = np.finfo(np.float32).eps
    # Cushion for coordinate rounding and normalization, based on the
    # first derivative of det(a,b,c); not a tolerance that accepts negatives.
    margin = 8 * epsilon * (np.linalg.norm(np.cross(a, b), axis=1)
                            + np.linalg.norm(np.cross(b, c), axis=1)
                            + np.linalg.norm(np.cross(c, a), axis=1)) + 64 * epsilon**2
    return signed, margin


def topology(faces, nvertices):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces)
    check_indices(faces, nvertices)
    if (np.diff(np.sort(faces, axis=1), axis=1) == 0).any():
        raise SphericalQualityError('Degenerate spherical face')
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise SphericalQualityError('Duplicate spherical faces')
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientation_sum = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if (counts != 2).any() or orientation_sum.any() or nvertices - len(unique) + len(faces) != 2:
        raise SphericalQualityError('Sphere must have closed, consistently oriented genus-zero topology')
    graph = coo_matrix((np.ones(len(unique)), (unique[:, 0], unique[:, 1])), shape=(nvertices, nvertices))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise SphericalQualityError('Disconnected sphere or unused vertices')
    paired = np.argsort(inverse).reshape(-1, 2)
    face_ids = np.tile(np.arange(len(faces)), 3)[paired]
    dual = coo_matrix((np.ones(len(face_ids)), (face_ids[:, 0], face_ids[:, 1])),
                      shape=(len(faces), len(faces)))
    if connected_components(dual, directed=False, return_labels=False) != 1:
        raise SphericalQualityError('Disconnected face adjacency or pinched nonmanifold vertices')
    return dict(vertices=nvertices, faces=len(faces), edges=len(unique), euler=2,
                connected_components=1, face_connected_components=1)


def canonical_faces(faces, nvertices, *, max_reoriented=64):
    """Repair sparse inconsistent winding using adjacency, never geometric sign.

    Two copies of the dual graph encode face orientation parity. An orientable
    connected closed mesh has two components; choose the one preserving the
    majority of the original winding. Triangle rows and vertex sets stay fixed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(faces)
    check_indices(faces, nvertices)
    if (np.diff(np.sort(faces, axis=1), axis=1) == 0).any():
        raise SphericalQualityError('Degenerate face in winding repair')
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    if (counts != 2).any() or nvertices - len(unique) + len(faces) != 2:
        raise SphericalQualityError('Winding repair requires a closed genus-zero mesh')
    order = np.argsort(inverse).reshape(-1, 2)
    face_ids = np.tile(np.arange(len(faces)), 3)[order]
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)[order]
    opposite = signs[:, 0] == signs[:, 1]
    count = len(faces)
    source = np.r_[face_ids[:, 0], face_ids[:, 0] + count]
    target = np.r_[face_ids[:, 1] + opposite * count, face_ids[:, 1] + (~opposite) * count]
    graph = coo_matrix((np.ones(len(source)), (source, target)), shape=(2*count, 2*count))
    components, labels = connected_components(graph, directed=False)
    if components != 2 or (labels[:count] == labels[count:]).any():
        raise SphericalQualityError('Nonorientable or disconnected mesh in winding repair')
    reverse = labels[:count] != labels[0]
    if reverse.sum() > count // 2:
        reverse = ~reverse
    if reverse.sum() > max_reoriented:
        raise SphericalQualityError('Face winding corrections exceed the bounded repair limit')
    corrected = faces.copy()
    corrected[reverse] = corrected[reverse][:, [0, 2, 1]]
    topology(corrected, nvertices)
    return corrected, dict(method='dual_graph_orientation_parity_majority',
                           reversed_face_rows=np.flatnonzero(reverse).tolist(),
                           coordinates_changed=False, triangle_vertex_sets_preserved=True)


def reported_winding(faces, nvertices):
    try:
        return canonical_faces(faces, nvertices)
    except SphericalQualityError as error:
        return faces, dict(reversed_face_rows=[], quality_warning=str(error),
                           coordinates_changed=False, triangle_vertex_sets_preserved=True)


def inspect(xyz, faces):
    check_indices(faces, len(xyz))
    unit = unit_vertices(xyz)
    signed, margin = orientation(unit, faces)
    a, b, c = unit[faces].transpose(1, 0, 2)
    denominator = 1 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
    degree = np.sum(2 * np.arctan2(signed, denominator)) / (4 * np.pi)
    return dict(negative_faces=int((signed < 0).sum()), zero_faces=int((signed == 0).sum()),
                unsafe_faces=int((signed <= margin).sum()), minimum_signed_determinant=float(signed.min()),
                minimum_margin_ratio=float((signed / margin).min()), spherical_degree=float(degree),
                nonlocal_faces=int((denominator <= 0).sum()))


def require_valid(xyz, faces):
    report = inspect(xyz, faces)
    if (report['unsafe_faces'] or report['nonlocal_faces']
            or abs(report['spherical_degree'] - 1) > 1e-6):
        raise SphericalQualityError('Invalid spherical orientation: ' + json.dumps(report))
    return report


def bounded_repair(xyz, faces, *, max_faces=64, max_vertices=192, max_displacement_mm=.5, max_iterations=100):
    """Minimize movement of unsafe-face vertices with all incident faces constrained.

    The boundary is fixed. SLSQP operates in tangent coordinates, with a
    per-vertex angular bound, and all checks repeat on quantized coordinates.
    This is a limited repair attempt, not a guarantee for arbitrary meshes.
    """
    from scipy.optimize import minimize

    original = np.asarray(xyz, dtype=np.float32).astype(np.float64)
    structure = topology(faces, len(original))
    unit = unit_vertices(original)
    signed, margin = orientation(unit, faces)
    before = inspect(original, faces)
    bad = np.flatnonzero(signed <= margin)
    report = dict(schema='fastprep.spherical_geometry/1', before=before, topology=structure,
                  method='bounded_local_tangent_constrained_minimum_displacement',
                  limits=dict(max_faces=max_faces, max_vertices=max_vertices,
                              max_displacement_mm=max_displacement_mm, max_iterations=max_iterations))
    if not len(bad):
        report.update(after=require_valid(original, faces), changed_vertices=0, max_displacement_mm=0., iterations=0)
        return original.astype(np.float32), report
    movable = np.unique(faces[bad])
    if len(bad) > max_faces or len(movable) > max_vertices or before['nonlocal_faces']:
        raise SphericalQualityError('Spherical defects exceed bounded local repair limits: ' + json.dumps(before))
    selected = np.isin(faces, movable).any(axis=1)
    local_faces = faces[selected]
    scale = np.maximum(np.abs(signed[selected]), 4 * margin[selected])
    target = 2 * margin[selected]
    lookup = np.full(len(unit), -1, dtype=int)
    lookup[movable] = np.arange(len(movable))
    local_ids = lookup[local_faces]
    origins = unit[movable]
    axis = np.eye(3)[np.argmin(np.abs(origins), axis=1)]
    tangent1 = np.cross(origins, axis)
    tangent1 /= np.linalg.norm(tangent1, axis=1, keepdims=True)
    basis = np.stack([tangent1, np.cross(origins, tangent1)], axis=2)
    angular_bound = max_displacement_mm / 100

    def evaluate(z, gradient=False):
        z = z.reshape(-1, 2)
        lifted = origins + angular_bound * np.einsum('nij,nj->ni', basis, z)
        lengths = np.linalg.norm(lifted, axis=1)
        moved = lifted / lengths[:, None]
        points = unit[local_faces].copy()
        for corner in range(3):
            valid = local_ids[:, corner] >= 0
            points[valid, corner] = moved[local_ids[valid, corner]]
        a, b, c = points.transpose(1, 0, 2)
        determinant = np.einsum('ij,ij->i', a, np.cross(b - a, c - a))
        constraints = np.r_[(determinant - target) / scale, 1 - np.sum(z*z, axis=1)]
        if not gradient:
            return constraints
        derivative = angular_bound * (basis - moved[:, :, None] * np.einsum('ni,nij->nj', moved, basis)[:, None, :]) / lengths[:, None, None]
        jac = np.zeros((len(local_faces) + len(movable), len(movable), 2))
        for corner, cross in enumerate([np.cross(b, c), np.cross(c, a), np.cross(a, b)]):
            indices = np.flatnonzero(local_ids[:, corner] >= 0)
            vertices = local_ids[indices, corner]
            jac[indices, vertices] = np.einsum('ni,nij->nj', cross[indices], derivative[vertices]) / scale[indices, None]
        jac[len(local_faces) + np.arange(len(movable)), np.arange(len(movable))] = -2*z
        return jac.reshape(len(constraints), -1)

    result = minimize(lambda z: float(z @ z), np.zeros(2 * len(movable)), jac=lambda z: 2*z,
                      constraints=[dict(type='ineq', fun=evaluate, jac=lambda z: evaluate(z, True))],
                      method='SLSQP', bounds=[(-1., 1.)] * (2 * len(movable)),
                      options=dict(maxiter=max_iterations, ftol=1e-12))
    candidate = original.copy()
    lifted = origins + angular_bound * np.einsum('nij,nj->ni', basis, result.x.reshape(-1, 2))
    candidate[movable] = 100 * lifted / np.linalg.norm(lifted, axis=1, keepdims=True)
    candidate = candidate.astype(np.float32)
    displacement = np.linalg.norm(candidate.astype(np.float64) - original, axis=1)
    if not np.isfinite(candidate).all():
        raise SphericalQualityError('Local spherical repair produced nonfinite candidate coordinates')
    if displacement.max() > max_displacement_mm:
        raise SphericalQualityError('Local spherical repair exceeded displacement bound')
    # Acceptance is based on measured geometry, even if the optimizer reports
    # a numerical stopping condition. Never accept its success flag alone.
    after = require_valid(candidate, faces)
    report.update(after=after, changed_vertices=int((displacement > 0).sum()),
                  max_displacement_mm=float(displacement.max()), iterations=int(result.nit),
                  repaired_face_ids=bad.tolist(), moved_vertex_ids=movable.tolist(),
                  optimizer_success=bool(result.success), optimizer_message=str(result.message))
    return candidate, report


def repair(xyz, faces, **limits):
    try:
        return bounded_repair(xyz, faces, **limits)
    except SphericalQualityError as error:
        original = np.asarray(xyz, dtype=np.float32)
        measured = inspect(original, faces)
        return original, dict(schema='fastprep.spherical_geometry/1', method='retain_native_coordinates',
                              before=measured, after=measured, changed_vertices=0,
                              max_displacement_mm=0., iterations=None, quality_warning=str(error))


def save_checked(destination, xyz, faces, *, volume_info=None):
    import nibabel as nib

    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refusing to overwrite sphere: ' + str(destination))
    # Validate both the quantized buffer and the actual file after writing.
    quantized = np.asarray(xyz, dtype=np.float32)
    unit_vertices(quantized)
    check_indices(faces, len(quantized))
    nib.freesurfer.write_geometry(str(destination), xyz, faces, volume_info=volume_info)
    actual, actual_faces = nib.freesurfer.read_geometry(str(destination))
    if not np.array_equal(actual_faces, faces) or not np.array_equal(actual, quantized):
        raise ValueError('Sphere serialization changed vertex/face identity')
    return inspect(actual, actual_faces)


def prepare_file(source, destination, *, topology_reference=None):
    import nibabel as nib

    xyz, faces, metadata = nib.freesurfer.read_geometry(str(source), read_metadata=True)
    winding = None
    if topology_reference is not None:
        reference_xyz, reference_faces = nib.freesurfer.read_geometry(str(topology_reference))
        if (xyz.shape != reference_xyz.shape or faces.shape != reference_faces.shape
                or not np.array_equal(np.sort(faces, axis=1), np.sort(reference_faces, axis=1))):
            raise ValueError('Sphere changed reference vertex or triangle identity')
        reference_faces, orientation_report = reported_winding(reference_faces, len(reference_xyz))
        winding = dict(reference=str(Path(topology_reference).resolve()),
                       restored_face_rows=np.flatnonzero((faces != reference_faces).any(axis=1)).tolist(),
                       reference_winding=orientation_report)
        # Restore the anatomical connectivity and adjacency-derived winding,
        # not a sign selected from the mapped coordinates. Folds still fail.
        faces = reference_faces
    candidate, report = repair(xyz, faces)
    report.update(source=str(Path(source).resolve()), output=str(Path(destination).absolute()))
    if winding is not None:
        report['topology_reference'] = winding
        if 'quality_warning' in winding['reference_winding']:
            report['winding_warning'] = winding['reference_winding']['quality_warning']
    report['readback'] = save_checked(destination, candidate, faces, volume_info=metadata or None)
    Path(str(destination) + '.geometry.json').write_text(json.dumps(report, indent=2) + '\n')
    return report

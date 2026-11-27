"""Continuous barycentric sampling on actual spherical mesh triangles.

Triangle lookup walks mesh adjacency in float64. Every sample must lie in an
actual containing triangle; there is no nearest-three-vertices fallback.
"""
import numpy as np


def interpolate(source_xyz, faces, target_xyz, values, *, max_steps=128):
    from scipy.spatial import cKDTree
    from spherical_geometry import topology, require_valid

    source = np.asarray(source_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    faces = np.asarray(faces)
    if (source.ndim != 2 or source.shape[1] != 3 or target.ndim != 2
            or target.shape[1] != 3 or len(values) != len(source)
            or not all(np.isfinite(x).all() for x in (source, target, values))):
        raise ValueError('Invalid spherical interpolation input')
    lengths = np.linalg.norm(source, axis=1)
    target_lengths = np.linalg.norm(target, axis=1)
    if (lengths == 0).any() or (target_lengths == 0).any():
        raise ValueError('Zero spherical interpolation direction')
    source = source / lengths[:, None]
    target = target / target_lengths[:, None]
    topology(faces, len(source))
    require_valid(source * 100, faces)
    triangle = source[faces]
    # In each cone, target = alpha*a + beta*b + gamma*c. Nonnegative
    # coefficients locate the ray in the triangle; normalizing their sum
    # gives planar barycentric weights at the ray/triangle intersection.
    inverse = np.linalg.inv(triangle.transpose(0, 2, 1))
    edges = faces[:, [[1, 2], [2, 0], [0, 1]]].reshape(-1, 2)
    _, inverse_edges, counts = np.unique(np.sort(edges, axis=1), axis=0,
                                         return_inverse=True, return_counts=True)
    if (counts != 2).any():
        raise ValueError('Spherical interpolation requires closed mesh adjacency')
    paired = np.argsort(inverse_edges).reshape(-1, 2)
    neighbor = np.empty(len(edges), dtype=np.int64)
    neighbor[paired[:, 0]] = paired[:, 1] // 3
    neighbor[paired[:, 1]] = paired[:, 0] // 3
    neighbor = neighbor.reshape(-1, 3)
    centers = triangle.sum(axis=1)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    chosen = cKDTree(centers).query(target, workers=1)[1]
    weights = np.empty((len(target), 3), dtype=np.float64)
    remaining = np.arange(len(target))
    steps = 0
    for steps in range(1, max_steps + 1):
        coefficients = np.einsum('nij,nj->ni', inverse[chosen[remaining]], target[remaining])
        total = coefficients.sum(axis=1)
        inside = (coefficients.min(axis=1) >= -1e-10) & (total > 0)
        # Clamp only floating point edge roundoff; extrapolation is forbidden.
        accepted = np.maximum(coefficients[inside], 0)
        weights[remaining[inside]] = accepted / accepted.sum(axis=1, keepdims=True)
        outside = remaining[~inside]
        if not len(outside):
            break
        edge = coefficients[~inside].argmin(axis=1)
        chosen[outside] = neighbor[chosen[outside], edge]
        remaining = outside
    else:
        raise ValueError(f'No containing spherical triangle for {len(remaining)} samples')
    vertex_values = values[faces[chosen]]
    output = np.einsum('ni,ni...->n...', weights, vertex_values)
    return output, dict(method='containing_mesh_triangle_float64',
                        samples=len(target), unmatched_samples=0,
                        maximum_walk_steps=steps, minimum_weight=float(weights.min()),
                        maximum_weight_sum_error=float(np.abs(weights.sum(axis=1) - 1).max()))

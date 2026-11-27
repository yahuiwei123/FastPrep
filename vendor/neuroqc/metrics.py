"""Deterministic measurements, with local checks separated from rule decisions."""
import numpy as np
from scipy import ndimage

from .evidence import metric_result, provenance
from .images import affine_mm, load_volume, same_grid
from .surfaces import read_surface


def mutual_information(a, b, bins=64):
    if a.shape != b.shape:
        raise ValueError("Mutual information requires matching arrays")
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.size < 100 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    hist = np.histogram2d(a, b, bins=bins)[0]
    p = hist / hist.sum()
    independent = p.sum(1)[:, None] * p.sum(0)[None, :]
    nonzero = p > 0
    return float(np.sum(p[nonzero] * np.log(p[nonzero] / independent[nonzero])))


def registration(fixed_path, moving_path=None, registered_path=None,
                 mi_min=None, improvement_min=None, threshold_source=None):
    if registered_path is None:
        raise ValueError("A registered image is required")
    for threshold in (mi_min, improvement_min):
        if threshold is not None and not np.isfinite(threshold):
            raise ValueError("Thresholds must be finite")
    fixed_image, fixed = load_volume(fixed_path, canonical=True)
    registered_image, registered = load_volume(registered_path, canonical=True)
    grid_ok = same_grid(fixed_image, registered_image)
    finite = bool(np.isfinite(fixed).all() and np.isfinite(registered).all())
    before, before_grid, before_finite = None, None, None
    if moving_path is not None:
        moving_image, moving = load_volume(moving_path, canonical=True)
        before_grid = same_grid(fixed_image, moving_image)
        before_finite = bool(np.isfinite(moving).all() and np.isfinite(fixed).all())
        if before_grid and before_finite:
            before = mutual_information(fixed, moving)
    after = mutual_information(fixed, registered) if grid_ok and finite else None
    improvement = after - before if before is not None and after is not None else None
    metrics = {"sameGrid": grid_ok, "finite": finite, "baselineSameGrid": before_grid,
               "baselineFinite": before_finite, "mutualInformationBefore": before,
               "mutualInformationAfter": after, "mutualInformationImprovement": improvement}
    missing = ["visual_review", "transform_metadata"]
    comparisons = [(after, mi_min), (improvement, improvement_min)]
    required = [(value, threshold) for value, threshold in comparisons if threshold is not None]
    failed = not grid_ok or not finite
    # A threshold without provenance is recorded but not used to claim a QC decision.
    if not required:
        missing.append("protocol_thresholds")
    elif not threshold_source:
        missing.append("threshold_source")
    else:
        failed |= any(value is not None and value < threshold for value, threshold in required)
    if after is None:
        missing.append("usable_after_similarity")
    if improvement_min is not None and improvement is None:
        missing.append("comparable_baseline_for_improvement")
    local = "fail" if failed else "pass" if required and threshold_source and all(
        value is not None for value, _ in required) else "unknown"
    result = metric_result(metrics, failed=failed, metric_status=local, missing=missing)
    result["provenance"] = provenance(
        {"fixed": fixed_path, "moving": moving_path, "registered": registered_path},
        {"mi_min": mi_min, "improvement_min": improvement_min,
         "threshold_source": threshold_source, "mi_bins": 64,
         "mi_log_base": "natural", "sampling": "all_voxels", "comparison": ">="})
    return result


def segmentation(image_path, labels_path, labels=None, connectivity=6):
    """Hard labels only. An empty class is measured, not judged anatomically."""
    if type(connectivity) is not int or connectivity not in (6, 18, 26):
        raise ValueError("3-D component connectivity must be 6, 18 or 26")
    image, raw = load_volume(image_path, canonical=True)
    label_image, values = load_volume(labels_path, canonical=True)
    grid_ok = same_grid(image, label_image)
    finite = bool(np.isfinite(values).all())
    image_finite = bool(np.isfinite(raw).all())
    rounded = np.rint(values)
    integer = finite and bool(np.allclose(values, rounded, atol=1e-4, rtol=0))
    present = np.unique(rounded) if integer else np.array([])
    selected = [int(x) for x in present if x != 0]
    if labels is not None:
        if any(not np.isfinite(x) or int(x) != x for x in labels):
            raise ValueError("Hard label values must be finite integers")
        selected = list(dict.fromkeys(int(x) for x in labels))
    voxel_volume = abs(float(np.linalg.det(affine_mm(label_image)[:3, :3])))
    tissue_metrics = {}
    # Noninteger data are never rounded into apparently valid measurements.
    if integer:
        # Scan the label map once, then measure each label in its complete bounding
        # box. Outside that box its mask is identically false, so components and
        # hole filling are unchanged. Border fractions still use the full image.
        encoded = np.searchsorted(present, rounded).astype(np.int32)
        encoded += 1  # include label zero when explicitly requested
        boxes = dict(zip(map(int, present), ndimage.find_objects(encoded, max_label=len(present))))
        del encoded
        structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
        for label in selected:
            box = boxes.get(label)
            voxels, count, sizes, holes, border_fraction = 0, 0, [], 0, None
            if box is not None:
                mask = rounded[box] == label
                voxels = int(mask.sum())
                components, count = ndimage.label(mask, structure=structure)
                sizes = sorted(np.bincount(components.ravel())[1:].astype(int).tolist(), reverse=True)
                holes = int((ndimage.binary_fill_holes(mask) & ~mask).sum())
                border = np.zeros(mask.shape, dtype=bool)
                for axis, extent in enumerate(box):
                    for index, touches in ((0, extent.start == 0), (-1, extent.stop == values.shape[axis])):
                        if touches:
                            edge = [slice(None)] * 3
                            edge[axis] = index
                            border[tuple(edge)] = True
                border_fraction = float((mask & border).sum() / voxels)
            tissue_metrics[str(label)] = {
                "foregroundVoxelCount": voxels, "volumeMl": voxels * voxel_volume / 1000,
                "connectedComponents": int(count), "componentConnectivity": connectivity,
                "componentSizesVoxels": sizes,
                "detachedForegroundVoxelCount": voxels - (sizes[0] if sizes else 0), "holes": holes,
                "borderVoxelFraction": border_fraction}
    result = metric_result(
        {"sameGrid": grid_ok, "finite": finite, "imageFinite": image_finite,
         "integerLabels": integer, "binaryLabels": finite and bool(np.isin(values, [0, 1]).all()),
         "labels": tissue_metrics},
        failed=not (grid_ok and finite and image_finite and integer),
        missing=["label_definitions_and_protocol", "visual_review", "coverage_assessment"])
    result["referenceMetrics"] = {"status": "unknown", "reason": "No independent reference supplied"}
    result["provenance"] = provenance({"image": image_path, "labels": labels_path},
                                      {"labels": selected, "mode": "hard_labels",
                                       "component_connectivity": connectivity,
                                       "foreground_definition": "Each selected hard label is measured separately by equality",
                                       "holes_unit": "voxels"})
    return result


def surface(path):
    vertices, faces = read_surface(path)
    finite = bool(np.isfinite(vertices).all())
    valid = bool(np.issubdtype(faces.dtype, np.integer) and faces.min() >= 0 and faces.max() < len(vertices))
    degenerate = boundary = nonmanifold = euler = duplicate = isolated = None
    if valid and finite:
        tri = vertices[faces]
        twice_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
        degenerate = int(np.count_nonzero(twice_area == 0))
        boundary = int(np.count_nonzero(counts == 1))
        nonmanifold = int(np.count_nonzero(counts > 2))
        euler = int(len(vertices) - len(unique_edges) + len(faces))
        duplicate = int(len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0)))
        isolated = int(len(vertices) - len(np.unique(faces)))
    metrics = {"vertexCount": len(vertices), "faceCount": len(faces), "finite": finite,
               "validFaceIndices": valid, "degenerateFaceCount": degenerate,
               "boundaryEdgeCount": boundary, "nonmanifoldEdgeCount": nonmanifold,
               "eulerCharacteristic": euler, "duplicateFaceCount": duplicate,
               "isolatedVertexCount": isolated, "selfIntersections": None}
    result = metric_result(metrics, failed=not (finite and valid), metric_status=(
        "fail" if not (finite and valid) else "unknown"),
        missing=["self_intersections", "protocol_topology_and_defect_limits", "six_view_visual_review"])
    result["provenance"] = provenance({"surface": path}, {"degenerate_test": "exact_zero_area"})
    return result

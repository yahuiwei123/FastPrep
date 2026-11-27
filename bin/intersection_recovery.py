"""Bounded fixed-boundary recovery for stalled initialized-surface repair."""

from pathlib import Path
import warnings


def write_coordinates_like(reference, destination, coordinates):
    """Replace only triangle-file xyz bytes, preserving the native footer exactly."""
    import io
    import struct
    import numpy as np

    payload = Path(reference).read_bytes()
    stream = io.BytesIO(payload)
    if stream.read(3) != b'\xff\xff\xfe':
        raise ValueError('Coordinate replacement requires a FreeSurfer triangle surface')
    for _ in range(2):
        if not stream.readline().endswith(b'\n'):
            raise ValueError('Truncated surface comment header')
    counts = stream.read(8)
    if len(counts) != 8:
        raise ValueError('Truncated surface counts')
    vertices, faces = struct.unpack('>ii', counts)
    start = stream.tell()
    end = start + vertices * 12
    if vertices <= 0 or faces <= 0 or end + faces * 12 > len(payload):
        raise ValueError('Invalid or truncated surface geometry')
    values = np.asarray(coordinates)
    if values.shape != (vertices, 3) or values.dtype.kind != 'f' or not np.isfinite(values).all():
        raise ValueError('Replacement coordinates must retain finite vertex correspondence')
    with np.errstate(over='ignore'):
        encoded = values.astype('>f4')
    if not np.isfinite(encoded).all():
        raise ValueError('Replacement coordinates overflow the surface storage format')
    # NiBabel serializes volume_info with reduced decimal precision. Keeping
    # these original bytes avoids changing CRAS on a later native read/write.
    with Path(destination).open('xb') as output:
        output.write(payload[:start])
        output.write(encoded.tobytes())
        output.write(payload[end:])


def fair_coordinates(coordinates, faces, seeds, *, rings=3, blend=0.25,
                     max_displacement_mm=1.0, clip=False):
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import MatrixRankWarning, spsolve

    if (coordinates.ndim != 2 or coordinates.shape[1] != 3 or coordinates.dtype.kind != "f" or not np.isfinite(coordinates).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu"
            or not len(faces) or faces.min() < 0 or faces.max() >= len(coordinates)
            or seeds.shape != (len(coordinates),) or seeds.dtype != bool or not seeds.any()):
        raise ValueError("Invalid surface or intersection vertex mask for local fairing")
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)
    adjacency = sparse.coo_matrix((np.ones(2 * len(edges)),
                                  (edges.reshape(-1), edges[:, ::-1].reshape(-1))),
                                 shape=(len(coordinates), len(coordinates))).tocsr()
    selected = seeds.copy()
    if type(rings) is not int or rings < 1 or not 0 < blend <= 1 or not max_displacement_mm > 0:
        raise ValueError("Invalid local fairing parameters")
    for ring in range(rings):
        selected[np.unique(faces[np.any(selected[faces], axis=1)])] = True
    if selected.sum() > 1000 or selected.all():
        raise ValueError("Local fairing exceeds its 1000-vertex or fixed-boundary budget")
    laplacian = sparse.diags(np.asarray(adjacency.sum(axis=1)).reshape(-1)) - adjacency
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        harmonic = spsolve(laplacian[selected][:, selected], adjacency[selected][:, ~selected] @ coordinates[~selected])
    delta = blend * (harmonic - coordinates[selected])
    if clip:
        # Scale the whole field, preserving its smooth direction. Do not clip
        # individual vertices into a discontinuous deformation at the boundary.
        delta *= min(1.0, max_displacement_mm / max(1e-30, np.linalg.norm(delta, axis=1).max()))
    result = coordinates.copy()
    result[selected] = coordinates[selected] + delta
    if not np.isfinite(result).all() or np.linalg.norm(result - coordinates, axis=1).max() > max_displacement_mm + 1e-12:
        raise ValueError(f"Local fairing exceeds its finite-coordinate or {max_displacement_mm:g} mm displacement budget")
    return result, selected


class RecoveryExhausted(ValueError):
    """No candidate satisfied every geometric and displacement constraint."""


def validate_candidate(reference, faces, metadata, candidate, selected, budget):
    import nibabel as nib
    import numpy as np

    vertices, target_faces, target_metadata = nib.freesurfer.read_geometry(candidate, read_metadata=True)
    if vertices.shape != reference.shape or not np.array_equal(target_faces, faces) or not np.isfinite(vertices).all():
        raise ValueError("Local recovery changed vertex/face identity or finite geometry")
    for key in ("volume", "voxelsize", "xras", "yras", "zras", "cras"):
        if key not in metadata or key not in target_metadata or not np.array_equal(metadata[key], target_metadata[key]):
            raise ValueError("Local recovery changed the surface volume geometry: " + key)
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if np.any(area <= 1e-12):
        raise ValueError("Local recovery created a degenerate triangle")
    displacement = np.linalg.norm(vertices - reference, axis=1)
    # FreeSurfer surface coordinates are stored as float32; allow only storage
    # rounding at the displacement boundary, not a larger scientific tolerance.
    if displacement.max() > budget + 1e-5 or np.any(displacement[~selected] != 0):
        raise ValueError("Native repair exceeded the fixed region or displacement budget")
    return displacement, float(area.min())


def recover_intersections(stage, hemi, repaired, measure):
    import nibabel as nib
    import numpy as np

    from reference_runtime import write_json

    parameters = stage.node["parameters"]
    budget = parameters.get("local_recovery_max_displacement_mm", 2.5)
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not np.isfinite(budget) or not 0 < budget <= 2.5:
        raise ValueError("Local recovery displacement budget must be positive and at most 2.5 mm")
    folder = stage.root / "intersection-recovery" / hemi / Path(repaired).name
    folder.mkdir(parents=True, exist_ok=False)
    native = folder / (hemi + ".orig.native")
    native.write_bytes(Path(repaired).read_bytes())
    mask = folder / "intersections.mgz"
    stage.run("freesurfer:mris_remove_intersection", ["-map", native, mask])
    coordinates, faces, metadata = nib.freesurfer.read_geometry(native, read_metadata=True)
    values = nib.load(mask).get_fdata().reshape(-1)
    if values.shape != (len(coordinates),) or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Invalid native intersection mask")
    seeds = values > 0
    trials = []
    details = dict(method="bounded neighborhood harmonic fairing with native cleanup",
                   seed_vertices=int(seeds.sum()), native_surface=str(native), trials=trials,
                   max_displacement_mm=budget, max_selected_vertices=1000,
                   anatomical_review="pending", scientific_approval="not_performed")
    for rings in (2, 3, 5):
        record = dict(neighborhood_rings=rings)
        trials.append(record)
        try:
            candidate, selected = fair_coordinates(coordinates, faces, seeds, rings=rings, blend=1.0,
                                                   max_displacement_mm=min(1.0, budget), clip=True)
        except ValueError as error:
            record['rejected'] = str(error)
            write_json(folder / 'recovery.json', details)
            continue
        record['selected_vertices'] = int(selected.sum())
        fairing = folder / (hemi + f".orig.fairing-r{rings}")
        write_coordinates_like(native, fairing, candidate)
        for cleanup in (False, True):
            target = fairing
            if cleanup:
                target = folder / (hemi + f".orig.cleaned-r{rings}")
                stage.run("freesurfer:mris_remove_intersection", [fairing, target])
            checks = measure(stage, target)
            result = dict(surface=str(target), measurements=checks)
            record['native_cleanup' if cleanup else 'fairing'] = result
            try:
                displacement, minimum_area = validate_candidate(coordinates, faces, metadata, target, selected, budget)
                result.update(max_displacement_mm=float(displacement.max()), minimum_area_mm2=minimum_area)
            except ValueError as error:
                result['rejected'] = str(error)
                write_json(folder / 'recovery.json', details)
                continue
            write_json(folder / 'recovery.json', details)
            if (checks['euler_number'] != parameters['required_euler_number'] or
                    checks['total_defect_index'] != parameters['required_total_defect_index'] or
                    checks['self_intersecting_vertices'] != 0):
                continue
            details.update(neighborhood_rings=rings, selected_vertices=int(selected.sum()),
                           changed_vertices=int(np.count_nonzero(displacement)),
                           displacement_mm_p50_p95_max=np.percentile(displacement, [50, 95, 100]).tolist(),
                           changed_region_tkr_mm_min=coordinates[selected].min(axis=0).tolist(),
                           changed_region_tkr_mm_max=coordinates[selected].max(axis=0).tolist(),
                           candidate_surface=str(target), measurements=checks)
            write_json(folder / 'recovery.json', details)
            Path(repaired).write_bytes(target.read_bytes())
            return checks, details
    raise RecoveryExhausted("Bounded local recovery exhausted without zero intersections within its fixed region and displacement budget: " + hemi)

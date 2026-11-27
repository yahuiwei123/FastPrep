"""Image geometry and display helpers shared by metrics and renderers."""
from pathlib import Path

import nibabel as nib
import numpy as np

PLANES = {"axial": 2, "sagittal": 0, "coronal": 1}
UNIT_TO_MM = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}


def load_volume(path, *, canonical=False):
    image = nib.load(str(path))
    if len(image.shape) != 3 or any(n == 0 for n in image.shape):
        raise ValueError(f"Expected a nonempty 3-D image: {path}")
    if not np.isfinite(image.affine).all() or np.linalg.det(image.affine[:3, :3]) == 0:
        raise ValueError(f"Invalid image affine: {path}")
    if canonical:
        image = nib.as_closest_canonical(image)
    return image, np.asanyarray(image.dataobj)


def spatial_unit(image):
    # FreeSurfer MGH/MGZ geometry is defined in millimetres.
    if isinstance(image, nib.MGHImage):
        return "mm"
    unit = image.header.get_xyzt_units()[0]
    # Workflow convention: unspecified NIfTI spatial units are millimetres.
    return "mm" if unit == "unknown" else unit


def affine_mm(image):
    unit = spatial_unit(image)
    if unit not in UNIT_TO_MM:
        raise ValueError(f"Unsupported spatial unit: {unit}")
    affine = image.affine.copy()
    affine[:3, :] *= UNIT_TO_MM[unit]
    return affine


def same_grid(a, b, *, atol_mm=1e-3):
    """Compare physical grids, interpreting unspecified NIfTI units as mm."""
    return a.shape == b.shape and bool(np.allclose(
        affine_mm(a), affine_mm(b), atol=atol_mm, rtol=0))


def require_same_grid(a, b):
    if not same_grid(a, b):
        raise ValueError("Image grid mismatch (shape or physical affine); resample explicitly")


def limits(data, window=None):
    if window is not None:
        lo, hi = map(float, window)
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            raise ValueError("Window requires finite LOW < HIGH")
        return lo, hi
    values = data[np.isfinite(data) & (data != 0)]
    if not values.size:
        values = data[np.isfinite(data)]
    lo, hi = np.percentile(values, [1, 99]) if values.size else (0, 1)
    return float(lo), float(hi if hi > lo else lo + 1)


def select_slices(data, axis, *, slices=None, slice_range=None, count=20):
    """Canonical voxel indices, selected from the source image, never a prediction.

    Full source support is an overview heuristic, not proof of brain coverage.
    Explicit slices include outlying/suspicious anatomy identified by a reviewer.
    """
    if slices is not None and slice_range is not None:
        raise ValueError("Use slices or slice_range, not both")
    if count < 1:
        raise ValueError("Slice count must be positive")
    if slices is not None:
        selected = list(slices)
        if not selected or any(int(x) != x for x in selected):
            raise ValueError("Slices must be nonempty integer indices")
    else:
        if slice_range is None:
            support = np.isfinite(data) & (data != 0)
            occupied = np.flatnonzero(np.any(support, axis=tuple(i for i in range(3) if i != axis)))
            low, high = (int(occupied[0]), int(occupied[-1])) if occupied.size else (0, data.shape[axis] - 1)
        else:
            low, high = slice_range
            if int(low) != low or int(high) != high or low > high:
                raise ValueError("Slice range requires integer LOW <= HIGH")
        if low < 0 or high >= data.shape[axis]:
            raise ValueError("Slice range outside the canonical image")
        selected = np.rint(np.linspace(low, high, min(count, high - low + 1))).astype(int).tolist()
    if any(x < 0 or x >= data.shape[axis] for x in selected):
        raise ValueError("Slice index outside the canonical image")
    return list(dict.fromkeys(int(x) for x in selected))


def panel(data, axis, index):
    """Horizontal/vertical axes are the remaining canonical axes, origin lower."""
    return np.take(data, index, axis=axis).T


def panel_aspect(image, axis):
    affine = affine_mm(image)
    vectors = affine[:3, :3]
    norms = np.linalg.norm(vectors, axis=0)
    normalized = vectors / norms
    if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError("Sheared grids require explicit resampling before slice rendering")
    remaining = [i for i in range(3) if i != axis]
    return float(norms[remaining[1]] / norms[remaining[0]])


def image_record(path, image):
    return {"source": str(Path(path).resolve()), "shape": [int(n) for n in image.shape],
            "affine": image.affine.tolist(), "spatial_unit": spatial_unit(image),
            "axis_codes": list(nib.aff2axcodes(image.affine))}

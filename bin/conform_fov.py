"""FreeSurfer conform-width selection, following sMRIPrep's FOV check."""

from pathlib import Path

import nibabel as nib
import numpy as np


POLICY = "smriprep-physical-fov-cw256"
SOURCE = "https://github.com/nipreps/smriprep/blob/master/src/smriprep/workflows/surfaces.py"


def image_geometry(path):
    image = nib.load(path)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    affine = np.asarray(image.affine, dtype=float)
    if (len(image.shape) != 3 or any(size < 1 for size in image.shape)
            or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.isfinite(affine).all()
            or abs(np.linalg.det(affine[:3, :3])) < 1e-12):
        raise ValueError("Invalid conform image geometry: " + str(path))
    return dict(path=str(Path(path).absolute()), shape=[int(size) for size in image.shape],
                voxel_size_mm=spacing.tolist(),
                fov_mm=(np.asarray(image.shape) * spacing).tolist(), affine=affine.tolist())


def motioncor_fov_policy(paths):
    """Inputs are mri_convert-produced MGZ files, whose spatial unit is mm.

    As in sMRIPrep's _check_cw256, compare shape * voxel size, not voxel
    count, with 256 mm. FreeSurfer performs the conform operation itself.
    This decision does not establish anatomical coverage after cropping.
    """
    inputs = [image_geometry(path) for path in paths]
    if not inputs:
        raise ValueError("Conform requires at least one input volume")
    force_width = any(max(item["fov_mm"]) > 256 for item in inputs)
    return dict(policy=POLICY, source=SOURCE, threshold_mm=256,
                inputs=inputs, flags=["-cw256"] if force_width else [],
                coverage_review="pending" if force_width else "no_forced_crop")


def record_conform_output(decision, path):
    output = image_geometry(path)
    image = nib.load(path)
    if not np.isfinite(np.asarray(image.dataobj)).all():
        raise ValueError("Non-finite conformed image")
    if decision["flags"] and (output["shape"] != [256, 256, 256]
                             or not np.allclose(output["voxel_size_mm"], 1, atol=1e-5, rtol=0)):
        raise ValueError("FreeSurfer -cw256 did not produce the expected 256 mm grid")
    decision["output"] = output
    inverse = np.linalg.inv(np.asarray(output["affine"]))
    decision["input_voxel_to_output_voxel"] = [
        (inverse @ np.asarray(item["affine"])).tolist() for item in decision["inputs"]]
    return decision

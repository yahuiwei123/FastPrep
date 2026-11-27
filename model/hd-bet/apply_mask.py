"""Apply an existing binary HD-BET brain mask without rerunning inference."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def apply_mask(image, mask_image):
    data = np.asarray(image.dataobj)
    mask = np.asarray(mask_image.dataobj)
    if image.shape != mask_image.shape or not np.allclose(
            image.affine, mask_image.affine, atol=1e-4, rtol=0):
        raise ValueError("Image and mask must share one physical grid")
    if not np.isfinite(data).all() or not np.isfinite(mask).all():
        raise ValueError("Image and mask must be finite")
    if not set(np.unique(mask)).issubset({0, 1}) or not np.any(mask == 1):
        raise ValueError("Mask must be nonempty and binary")
    return np.where(mask != 0, data, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    image = nib.load(str(args.image))
    mask = nib.load(str(args.mask))
    result = apply_mask(image, mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, image.affine, image.header), str(args.output))


if __name__ == "__main__":
    main()

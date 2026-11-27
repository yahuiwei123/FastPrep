"""Predict an HD-BET binary brain mask from one nifti image.

The upstream ``hdbet_predict`` entry point runs nnU-Net preprocessing and
segmentation export in spawn workers, which only works when the calling script
is guarded by ``if __name__ == '__main__'`` and when the input is a nifti file.
This adapter performs the same single-case inference through nnU-Net's
documented sequential path instead: it loads the checkpoint named by an explicit
local model directory and writes exactly the mask, not a brain-extracted image
and not any leftover plans/dataset sidecar.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nibabel as nib
import numpy as np


def existing(path, name):
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"Missing or empty {name}: {resolved}")
    return resolved


def parameter_folder(model_dir):
    """Return the parameter folder with the trained fold, as HD-BET expects."""
    root = Path(model_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Missing HD-BET parameter directory: {root}")
    if not (root / "plans.json").is_file() or not (root / "dataset.json").is_file():
        raise ValueError(f"HD-BET parameter directory lacks plans.json/dataset.json: {root}")
    fold = root / "fold_all"
    if not fold.is_dir() or not any(fold.glob("checkpoint_*.pth")):
        raise ValueError(f"HD-BET parameter directory lacks fold_all checkpoints: {root}")
    return root


def main(argv=None):
    model_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=model_root / "source",
                        help="bundled HD-BET source directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tta", action="store_true",
                        help="enable mirroring test-time augmentation")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args(argv)

    image_path = existing(args.image, "input MRI")
    parameters = parameter_folder(args.model_dir)
    source = Path(args.source_root).resolve()
    if not (source / "HD_BET" / "hd_bet_prediction.py").is_file():
        raise ValueError(f"Missing bundled HD-BET source: {source}")
    if str(image_path).endswith(".mgz"):
        raise ValueError("HD-BET reads its inputs through SimpleITK, which cannot open .mgz; supply nifti")

    os.environ["HD_BET_PARAMS"] = str(parameters)
    os.environ.setdefault("nnUNet_compile", "F")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import torch
    from HD_BET.hd_bet_prediction import get_hdbet_predictor

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested but is unavailable")
        device = torch.device(args.device)
    predictor = get_hdbet_predictor(use_tta=args.tta, device=device, verbose=False)
    if device.type == "cpu":
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))

    reference = nib.load(str(image_path))
    values = np.asanyarray(reference.dataobj)
    if values.ndim != 3:
        raise ValueError("HD-BET expects one 3D image without a time or channel axis")

    with tempfile.TemporaryDirectory(prefix="hdbet-") as scratch:
        truncated = str(Path(scratch) / "mask.nii.gz")
        predictor.predict_from_files_sequential([[str(image_path)]], [truncated],
                                                save_probabilities=False, overwrite=True)
        written = Path(truncated)
        if not written.is_file():
            raise RuntimeError("HD-BET produced no mask for the declared input")
        mask_image = nib.load(str(written))
        mask = np.asanyarray(mask_image.dataobj)

    if mask.shape != reference.shape or not np.allclose(mask_image.affine, reference.affine, atol=1e-4, rtol=0):
        raise ValueError("Predicted mask does not share the input physical grid")
    if set(np.unique(mask).tolist()) - {0, 1}:
        raise ValueError("HD-BET must emit a hard 0/1 mask")
    if not np.any(mask == 1):
        raise ValueError("HD-BET produced an empty mask")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), reference.affine), str(args.output))
    checkpoint = sorted((parameters / "fold_all").glob("checkpoint_*.pth"))
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps({
        "operation": "hd-bet:predict-mask",
        "input": str(image_path),
        "output": str(args.output.resolve()),
        "model_dir": str(parameters),
        "checkpoint": str(checkpoint[-1]),
        "source_root": str(source),
        "labels": {"0": "background", "1": "brain"},
        "device": str(device),
        "test_time_augmentation": args.tta,
        "foreground_voxels": int((mask == 1).sum()),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

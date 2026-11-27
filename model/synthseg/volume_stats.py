"""Compute SynthSeg posterior volumes without depending on the full result bundle."""

import argparse
import csv
from pathlib import Path

import numpy as np


def posterior_volumes(posterior, labels, voxel_volume_mm3, channel_axis=-1):
    posterior = np.asarray(posterior)
    moved = np.moveaxis(posterior, channel_axis, -1)
    if moved.shape[-1] != len(labels) or not np.isfinite(moved).all():
        raise ValueError("Posterior channel count must match labels and be finite")
    if voxel_volume_mm3 <= 0 or not np.isfinite(voxel_volume_mm3):
        raise ValueError("Voxel volume must be positive and finite")
    if np.any(moved < 0) or np.any(moved > 1):
        raise ValueError("Posterior values must lie in [0, 1]")
    return moved.reshape(-1, moved.shape[-1]).sum(axis=0) * voxel_volume_mm3


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", required=True, type=Path)
    parser.add_argument("--labels", required=True, nargs="+", type=int)
    parser.add_argument("--voxel-volume-mm3", required=True, type=float)
    parser.add_argument("--channel-axis", type=int, default=-1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    volumes = posterior_volumes(
        np.load(args.posterior), args.labels, args.voxel_volume_mm3, args.channel_axis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("label", "volume_mm3"))
        writer.writerows(zip(args.labels, volumes))


if __name__ == "__main__":
    main()

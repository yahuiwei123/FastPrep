"""Compute per-label posterior volumes from explicit files.

``volume_stats.py`` takes its label list as a variable-length ``--labels``
argument, which the registration contract cannot bind to one input file per
token. This adapter takes the same label schema as a JSON document and performs
the identical computation, so the operation has one declared input per port.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_labels(path):
    document = json.loads(Path(path).read_text())
    labels = document.get("labels") if isinstance(document, dict) else None
    if (not isinstance(labels, list) or not labels
            or any(not isinstance(value, int) or isinstance(value, bool) for value in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Label schema must declare a nonempty list of unique integer labels")
    return labels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--voxel-volume-mm3", required=True, type=float)
    parser.add_argument("--channel-axis", type=int, default=-1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    labels = load_labels(args.labels)
    if not args.voxel_volume_mm3 > 0 or not np.isfinite(args.voxel_volume_mm3):
        raise ValueError("Voxel volume must be positive and finite")
    # Reuse the reviewed upstream computation instead of a second implementation.
    source = Path(__file__).with_name("volume_stats.py")
    spec = importlib.util.spec_from_file_location("synthseg_volume_stats", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    volumes = module.posterior_volumes(
        np.load(args.posterior), labels, args.voxel_volume_mm3, args.channel_axis)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("label", "volume_mm3"))
        writer.writerows(zip(labels, volumes))


if __name__ == "__main__":
    main()

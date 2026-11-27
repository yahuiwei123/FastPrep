"""Check host ABI requirements and native / PTX architecture images of the 8.1 build."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

CUDA12 = [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90, 100, 101, 120]
CUDA13 = [75, 80, 86, 87, 88, 89, 90, 100, 103, 110, 120, 121]
KERNELS = [
    "topology/native/geometry.so",
    "t2_pial/native/classify.so",
    "t2_pial/native/pial-field.so",
    "t2_pial/t1/native/signed.so",
]
DRIVERS = [
    "topology/native/qsphere",
    "t2_pial/native/place",
    "t2_pial/native/distance-field",
    "t2_pial/t1/native/place",
]


def versions(path):
    text = subprocess.check_output(["readelf", "-WV", path], text=True)
    needs = text.partition("Version needs section")[2]
    return {
        tag: max(
            (
                tuple(map(int, v.split(".")))
                for v in re.findall(rf"Name: {tag}_(\d+(?:\.\d+)+)", needs)
            ),
            default=(),
        )
        for tag in ("GLIBC", "GLIBCXX", "CXXABI")
    }


def inspect(path, targets=()):
    abi = versions(path)
    for name, limit in [
        ("GLIBC", (2, 31)),
        ("GLIBCXX", (3, 4, 28)),
        ("CXXABI", (1, 3, 12)),
    ]:
        if abi[name] > limit:
            raise RuntimeError(
                f"{path} requires {name}_{abi[name]}, above Ubuntu 20.04 baseline"
            )
    report = {k: ".".join(map(str, v)) for k, v in abi.items() if v}
    rpath = subprocess.check_output(
        ["patchelf", "--print-rpath", path], text=True
    ).strip()
    if rpath not in ("", "$ORIGIN"):
        raise RuntimeError(f"{path} has external build RPATH {rpath}")
    if targets:
        for option, suffix, name, expected in [
            ("-lelf", "cubin", "sass", targets),
            ("-lptx", "ptx", "ptx", [min(targets)]),
        ]:
            text = subprocess.check_output(["cuobjdump", option, path], text=True)
            actual = sorted(
                {int(v) for v in re.findall(r"\.sm_(\d+)\." + suffix, text)}
            )
            if actual != expected:
                raise RuntimeError(
                    f"{path}: expected {name} {expected}, found {actual}"
                )
            report[name] = actual
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vendor", type=Path)
    parser.add_argument("--sugar-extension", type=Path)
    parser.add_argument("--sysroot", type=Path)
    parser.add_argument("--freesurfer", type=Path)
    args = parser.parse_args()
    report = {}
    for name in KERNELS:
        file = Path(name)
        for folder, targets in [
            (file.parent, CUDA12),
            (file.parent / "cuda13", CUDA13),
        ]:
            rel = folder / file.name
            report[str(rel)] = inspect(args.vendor / rel, targets)
    for name in DRIVERS:
        report[name] = inspect(args.vendor / name)
    if args.sysroot:
        if not args.freesurfer:
            parser.error("--freesurfer is required with --sysroot")
        root = args.sysroot.resolve()
        vendor = args.vendor.resolve()
        lib = root / "lib/x86_64-linux-gnu"
        usr = root / "usr/lib/x86_64-linux-gnu"
        fs = args.freesurfer.resolve() / "python/packages/fsbindings"
        loader = lib / "ld-linux-x86-64.so.2"
        for name in report:
            path = vendor / name
            search = os.pathsep.join(map(str, (path.parent, fs, lib, usr)))
            listing = subprocess.check_output(
                [loader, "--inhibit-cache", "--library-path", search, "--list", path],
                text=True,
            )
            for dependency in re.findall(r"=> (/[^ ]+)", listing):
                if not any(
                    Path(dependency).resolve().is_relative_to(base)
                    for base in (root, vendor, fs)
                ):
                    raise RuntimeError(
                        f"{name} resolved dependency outside target libraries: {dependency}"
                    )
            report[name]["focal_loader"] = "passed"
    if args.sugar_extension:
        report["pytorch3d/_C"] = inspect(
            args.sugar_extension, [a for a in CUDA12 if a >= 70]
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

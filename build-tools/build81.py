"""Build the 8.1 host drivers and both CUDA bundles, verify, then publish."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "compat-build"))
import drivers
import surface_cuda
import toolchain
from verify import DRIVERS, KERNELS


def publish(source, dest):
    with tempfile.NamedTemporaryFile(
        dir=dest.parent, prefix=".build-", delete=False
    ) as stream:
        tmp = Path(stream.name)
    try:
        shutil.copy2(source, tmp)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--freesurfer", type=Path, required=True, help="FreeSurfer 8.1.0 Ubuntu 20 installation")
    parser.add_argument("--source", type=Path, required=True, help="Matching FreeSurfer source tree or source-input tar archive")
    parser.add_argument("--packages", type=Path, required=True, help="Directory containing the compiler, sysroot and ITK .deb files")
    parser.add_argument("--cuda12", type=Path, required=True, help="CUDA 12.8 Conda toolchain prefix")
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--nvcc13", default="/usr/local/cuda-13.2/bin/nvcc")
    args = parser.parse_args()
    os.environ.pop("LIBRARY_PATH", None)
    work = args.work_dir.resolve()
    fs = args.freesurfer.resolve()
    workflow = HERE.parent
    src, itk, focal = toolchain.prepare(work, args.source.resolve(), args.packages.resolve())
    stage = work / "vendor"
    surface_cuda.build(stage, src / "include", toolchain.nvcc12(work, args.cuda12.resolve()), args.nvcc13)
    drivers.build_drivers(workflow, fs, src, itk, work / "drivers", stage)
    with (work / "surface-abi.json").open("w") as f:
        subprocess.run(
            [
                sys.executable,
                HERE / "compat-build/verify.py",
                stage,
                "--sysroot",
                focal,
                "--freesurfer",
                fs,
            ],
            check=True,
            stdout=f,
        )
    if not args.stage_only:
        artifacts = [Path(p) for p in DRIVERS + KERNELS]
        artifacts += [Path(p).parent / "cuda13" / Path(p).name for p in KERNELS]
        for artifact in artifacts:
            publish(stage / artifact, workflow / "vendor" / artifact)


if __name__ == "__main__":
    main()

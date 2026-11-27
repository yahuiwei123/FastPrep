"""Build the SUGAR CUDA extension against dio's installed PyTorch and its matching PyTorch3D CPU objects."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "compat-build"))
import toolchain
from build81 import publish
from verify import CUDA12, inspect
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--python-prefix", type=Path, required=True, help="CPython 3.9, PyTorch 2.8+cu128 and CUDA 12.8 prefix")
    parser.add_argument("--source", type=Path, required=True, help="PyTorch3D 0.7.9 source with matching prebuilt CPU objects")
    parser.add_argument("--install-dir", type=Path, help="Optional existing pytorch3d package directory for the verified extension")
    parser.add_argument("--jobs", type=int, default=10)
    args = parser.parse_args()
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    os.environ.pop("LIBRARY_PATH", None)
    dio = args.python_prefix.resolve()
    torch = dio / "lib/python3.9/site-packages/torch"
    source = args.source.resolve()
    original = source / "build/temp.linux-x86_64-cpython-39"
    nvcc = toolchain.nvcc12(work, dio)
    native = [sm for sm in CUDA12 if sm >= 70]
    flags = [
        "-ccbin",
        str(HERE / "compat-build/cuda-host-cxx"),
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-Xcompiler=-fPIC",
        "-DWITH_CUDA",
        "-DCUDA_HAS_FP16=1",
        "-DTHRUST_IGNORE_CUB_VERSION_CHECK",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-DTORCH_EXTENSION_NAME=_C",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
    ]
    flags.extend(
        "-I" + str(p)
        for p in [
            torch / "include",
            torch / "include/torch/csrc/api/include",
            dio / "include/python3.9",
            source / "pytorch3d/csrc",
        ]
    )
    for major in sorted({sm // 10 for sm in native}):
        group = [sm for sm in native if sm // 10 == major]
        codes = ["sm_" + str(sm) for sm in group]
        if group[0] == min(native):
            codes.append("compute_" + str(group[0]))
        flags.append(f"-gencode=arch=compute_{group[0]},code=[{','.join(codes)}]")
    lines = [
        "rule cuda",
        "  command = " + " ".join([str(nvcc), *flags, "-c $in -o $out"]),
    ]
    objects = []
    for obj in sorted(original.rglob("*.o")):
        relative = Path(str(obj).split("/pytorch3d/csrc/")[-1])
        cu = source / "pytorch3d/csrc" / relative.with_suffix(".cu")
        if cu.exists():
            (work / relative).parent.mkdir(parents=True, exist_ok=True)
            lines.append(f"build {relative}: cuda {cu}")
            objects.append(work / relative)
        else:
            objects.append(obj)
    if len(objects) != 67 or len(lines) - 2 != 31:
        raise ValueError(
            "Unexpected translation units for the installed PyTorch3D 0.7.9 source"
        )
    (work / "build.ninja").write_text("\n".join(lines) + "\n")
    subprocess.run(["ninja", "-j", str(args.jobs)], cwd=work, check=True)
    out = work / "_C.cpython-39-x86_64-linux-gnu.so"
    subprocess.run(
        [
            HERE / "compat-build/cuda-host-cxx",
            "-pthread",
            "-shared",
            "-o",
            out,
            *objects,
            "-L" + str(torch / "lib"),
            "-L" + str(dio / "targets/x86_64-linux/lib"),
            "-lc10",
            "-ltorch",
            "-ltorch_cpu",
            "-ltorch_python",
            "-lcudart",
            "-lc10_cuda",
            "-ltorch_cuda",
        ],
        check=True,
    )
    subprocess.run(["patchelf", "--remove-rpath", out], check=True)
    print(json.dumps(inspect(out, native), indent=2))
    if args.install_dir:
        publish(out, args.install_dir / out.name)


if __name__ == "__main__":
    main()

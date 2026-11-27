import concurrent.futures
import os
from pathlib import Path
import subprocess
import sys

vendor = Path(__file__).resolve().parents[2] / "vendor"
sys.path.insert(0, str(vendor))
from cuda_target import flags, targets

sources = {
    "topology/native/geometry.so": "topology/src/geometry.cu",
    "t2_pial/native/classify.so": "t2_pial/src/sync_kernel.cu",
    "t2_pial/native/pial-field.so": "t2_pial/src/pial_field.cu",
    "t2_pial/t1/native/signed.so": "t2_pial/t1/signed_smooth.cu",
}


def compile_one(stage, headers, nvcc, subdir, dest, source):
    out = stage / Path(dest).parent / subdir / Path(dest).name
    out.parent.mkdir(parents=True, exist_ok=True)
    archs = targets(nvcc)
    args = [
        nvcc,
        "-ccbin",
        str(Path(__file__).with_name("cuda-host-cxx")),
        "-I" + str(headers),
        "-O3",
        "--fmad=false",
        *flags(archs),
        "-shared",
        "-Xcompiler=-fPIC,-ffp-contract=off",
        vendor / source,
        "-o",
        out,
    ]
    with (out.with_suffix(".build.log")).open("w") as log:
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run(["patchelf", "--remove-rpath", out], check=True)
    print("finished", out, archs, flush=True)


def build(stage, headers, nvcc12, nvcc13):
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        jobs = [
            pool.submit(compile_one, stage, headers, nvcc, variant, dest, source)
            for nvcc, variant in [(nvcc12, ""), (nvcc13, "cuda13")]
            for dest, source in sources.items()
        ]
        for job in jobs:
            job.result()

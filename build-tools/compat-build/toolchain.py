"""Prepare private build inputs from the shared packages, without installing on compute nodes."""

import os
from pathlib import Path
import shutil
import subprocess
import tarfile

def prepare(work, source, package_dir):
    work.mkdir(parents=True, exist_ok=True)
    focal = work / "focal"
    itk = work / "itk"
    packages = sorted(package_dir.glob("*.deb"))
    if not packages:
        raise ValueError(f"No Ubuntu toolchain packages in {package_dir}")
    for package in packages:
        root = itk if package.name.startswith("libinsighttoolkit") else focal
        subprocess.run(["dpkg-deb", "-x", package, root], check=True)
    # Debian packages have absolute links that are interpreted relative to / on an installed system.
    for directory, dirs, files in os.walk(focal):
        for name in dirs + files:
            link = Path(directory) / name
            if link.is_symlink() and (target := Path(os.readlink(link))).is_absolute():
                link.unlink()
                link.symlink_to(
                    os.path.relpath(focal / target.relative_to("/"), link.parent)
                )
    if not source.is_dir():
        with tarfile.open(source) as archive:
            source = work / "source"
            archive.extractall(source, filter="data")
    host_lib = work / "compiler-lib"
    host_lib.mkdir(exist_ok=True)
    isl = host_lib / "libisl.so.22"
    if not isl.exists():
        isl.symlink_to(focal / "usr/lib/x86_64-linux-gnu/libisl.so.22")
    # Keep compiler-only libraries away from the scientific programs.
    os.environ["FASTPREP_FOCAL_SYSROOT"] = str(focal)
    os.environ["FASTPREP_COMPILER_LIB"] = str(host_lib)
    return source, itk / "usr/include/ITK-5.3", focal


def nvcc12(work, dio):
    root = work / "cuda12"
    dest = root / "bin"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("nvcc", "cicc"):
        src = dio / ("bin/nvcc" if name == "nvcc" else "nvvm/bin/cicc")
        out = root / src.relative_to(dio)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out)
        out.chmod(0o755)
    for source, target in [
        *(
            (dio / "bin" / n, dest / n)
            for n in (
                "nvcc.profile",
                "bin2c",
                "cudafe++",
                "fatbinary",
                "nvlink",
                "ptxas",
                "__nvcc_device_query",
                "crt",
            )
        ),
        (dio / "nvvm/libdevice", root / "nvvm/libdevice"),
        *(
            (dio / "targets/x86_64-linux" / n, root / "targets/x86_64-linux" / n)
            for n in ("include", "lib")
        ),
        (root / "nvvm", root / "targets/x86_64-linux/nvvm"),
    ]:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source)
    return dest / "nvcc"

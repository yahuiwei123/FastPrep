"""Build the Linux/FreeSurfer T2 placement runtime (never invoked by a subject job)."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cuda_target import targets, flags
import subprocess
import tempfile


STAMP = 'freesurfer-linux-ubuntu20_x86_64-8.1.0-20250718-f30dfed'
BINDING = 'fsbindings.cpython-38-x86_64-linux-gnu.so'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fs-home', type=Path, required=True)
    parser.add_argument('--fs-source', type=Path, required=True)
    parser.add_argument('--include', type=Path, action='append', default=[])
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--arch', type=int)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'native')
    args = parser.parse_args()
    args.archs = targets(args.nvcc, args.arch)
    home, source = args.fs_home.resolve(), args.fs_source.resolve()
    if (home / 'build-stamp.txt').read_text().strip() != STAMP:
        parser.error('requires the supported FreeSurfer 8.1.0 Linux build; see README')
    bindings = home / 'python/packages/fsbindings'
    if not (bindings / BINDING).is_file() or args.out.exists():
        parser.error('requires fsbindings and a fresh output directory')
    src = Path(__file__).resolve().parent / 'src'
    out = args.out.resolve()
    with tempfile.TemporaryDirectory(dir=out.parent) as directory:
        build = Path(directory) / 'native'
        build.mkdir()
        nv = [args.nvcc, '-shared', '-O3', '--fmad=false', *flags(args.archs), '-Xcompiler=-fPIC,-ffp-contract=off']
        subprocess.run([*nv, src / 'pial_field.cu', '-o', build / 'pial-field.so'], check=True)
        subprocess.run([*nv, '-I' + str(source / 'include'), src / 'sync_kernel.cu', '-o', build / 'classify.so'], check=True)
        include = [source / name for name in ('include', 'packages/nifti', 'packages/minc', 'packages/netcdf')]
        common = [args.cxx, '-O2', '-std=c++17', '-s', '-fopenmp', '-DHAVE_OPENMP', '-no-pie', '-rdynamic', '-DT2GPU_COMPACT_HEAP',
                  *['-I' + str(path.resolve()) for path in [*include, *args.include]]]
        libs = ['-L' + str(bindings), '-l:' + BINDING, home / 'python/lib/libpython3.8.a', '-ldl', '-lcrypt', '-lutil', '-lz']
        subprocess.run([*common, src / 'distance_field.cpp', src / 'lean_distance.cpp', *libs, '-o', build / 'distance-field'], check=True)
        subprocess.run([*common, '-DT2GPU_TARGET_OVERLAP', '-DT2GPU_ORDERED_PIAL', '-DT2GPU_PARALLEL_PIAL',
                        src / 'sync_bridge.cpp', src / 'place.cpp', src / 'lean_distance.cpp', *libs,
                        '-L' + str(build), '-Wl,-rpath,$ORIGIN', '-l:pial-field.so', '-o', build / 'place'], check=True)
        (build / 'runtime.json').write_text(json.dumps({'schema': 'fastprep.t2-pial/2', 'stamp': STAMP, 'ptx': True, 'arch': min(args.archs), 'sass': args.archs, 'binding': BINDING}) + '\n')
        os.rename(build, out)


if __name__ == '__main__':
    main()

"""Link the FreeSurfer 8.1.0 T1 placement branch to scoped audit/CUDA overrides."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cuda_target import targets, flags
import subprocess
import tempfile

SHARED = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'home', 'out'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--include', type=Path, action='append', default=[])
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', type=int)
    args = parser.parse_args()
    args.archs = targets(args.nvcc, args.arch)
    folder, src, home = (path.resolve() for path in (args.out, args.source, args.home))
    if (home/'build-stamp.txt').read_text().strip() != 'freesurfer-linux-ubuntu20_x86_64-8.1.0-20250718-f30dfed':
        parser.error('unsupported FreeSurfer ABI')
    if folder.exists(): parser.error('requires a fresh output directory')
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        compile_driver(args,src,home,Path(tmp),folder)


def compile_driver(args,src,home,folder,output):
    base = Path(__file__).resolve().parent
    includes = [src/name for name in ('include','packages/nifti','packages/minc','packages/netcdf')]
    common = ['g++', '-O2', '-std=c++17', '-DHAVE_OPENMP', '-fopenmp', *['-I'+str(path) for path in includes+args.include]]
    subprocess.run([args.nvcc, '-O3', '--fmad=false', *flags(args.archs), '-shared', '-Xcompiler=-fPIC',
                    base/'signed_smooth.cu', '-o', folder/'signed.so'], check=True)
    subprocess.run([*common, '-DmrisAsynchronousTimeStep=orderedMove', '-c', SHARED/'src/sync_bridge.cpp',
                    '-o', folder/'ordered.o'], check=True)
    official = src/'mris_make_surfaces/mris_place_surface.cpp'
    text = official.read_text()
    start = text.index('    else {\n      // Compute the target xyz coordinate (l_location)')
    end = text.index('    // This appears to adjust the cost weights', start)
    text = (text[:start] + '    else { fprintf(stderr,"T1-only placement driver\\n"); return 1; }\n' + text[end:])
    entry = folder/'t1_entry.cpp'
    entry.write_text(text.replace('#include "mris_multimodal_refinement.h"',''))
    subprocess.run([*common, '-I'+str(official.parent), '-s', '-no-pie', '-rdynamic', entry, base/'placement.cpp',
                    folder/'ordered.o', '-L'+str(folder), '-Wl,-rpath,$ORIGIN', '-l:signed.so',
                    '-L'+str(home/'python/packages/fsbindings'), '-l:fsbindings.cpython-38-x86_64-linux-gnu.so',
                    home/'python/lib/libpython3.8.a', '-ldl', '-lcrypt', '-lutil', '-lz', '-o', folder/'place'], check=True)
    bundle = folder/'runtime'
    bundle.mkdir()
    for name in ('signed.so','place'): (folder/name).rename(bundle/name)
    (bundle/'runtime.json').write_text(json.dumps(dict(ptx=True, arch=min(args.archs), sass=args.archs,stamp=(home/'build-stamp.txt').read_text().strip(),
                                                       schema='fastprep.t1-placement/2'))+'\n')
    bundle.rename(output)


if __name__ == '__main__':
    main()

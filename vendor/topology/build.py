"""Build the hybrid quick sphere against a matching FreeSurfer source tree and installation."""
import argparse
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
    for name in ('source', 'home', 'output'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--entry-dir', type=Path)
    parser.add_argument('--metric-dir', type=Path)
    parser.add_argument('--include', type=Path, action='append', default=[])
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--arch', type=int)
    args = parser.parse_args()
    args.archs = targets(args.nvcc, args.arch)
    source, home, output = (value.resolve() for value in (args.source, args.home, args.output))
    if (home / 'build-stamp.txt').read_text().strip() != STAMP:
        parser.error('unsupported FreeSurfer binding ABI')
    entry = (args.entry_dir or source / 'mris_sphere').resolve()
    metric = (args.metric_dir or source / 'utils').resolve()
    unit = Path(__file__).resolve().parent / 'src'
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as directory:
        work = Path(directory)
        with (work / 'mris_sphere.help.xml.h').open('wb') as header:
            subprocess.run(['xxd', '-i', 'mris_sphere.help.xml'], cwd=entry, stdout=header, check=True)
        base = (source / 'include/mrisurf_base.h').read_text()
        text = (metric / 'mrisurf_metricProperties_fast.h').read_text()
        parts = base[base.index('static float mrisNormalize('):base.index('int mris_sort_compare_float(')]
        parts += text[text.index('static void MRISMP_makeDist2'):text.index('void MRISMP_copy(')]
        parts += text[text.index('static void MRISMP_computeSurfaceDimensions'):text.index('static void mrismp_OrientEllipsoid')]
        (work / 'official_parts.inc').write_text(parts + '\n#undef COMPILING_MRIS_MP\n')
        subprocess.run([args.nvcc, '-O3', '--fmad=false', *flags(args.archs), '-shared', '-Xcompiler=-fPIC',
                        unit / 'geometry.cu', '-o', work / 'geometry.so'], check=True)
        includes = [work, metric, unit, *(source / name for name in ('include', 'packages/nifti', 'packages/minc', 'packages/netcdf')), *args.include]
        subprocess.run(['g++', '-std=c++17', '-O2', '-fopenmp', '-DHAVE_OPENMP', '-no-pie', '-rdynamic',
                        *['-I' + str(path) for path in includes], entry / 'mris_sphere.cpp', unit / 'help.cpp',
                        unit / 'metric_bridge.cpp', '-L' + str(work), '-Wl,-rpath,$ORIGIN', '-l:geometry.so',
                        '-L' + str(home / 'python/packages/fsbindings'), '-l:' + BINDING, home / 'python/lib/libpython3.8.a',
                        '-ldl', '-lcrypt', '-lutil', '-lz', '-s', '-o', work / 'qsphere'], check=True)
        for name in ('geometry.so', 'qsphere'):
            (work / name).replace(output / name)


if __name__ == '__main__':
    main()

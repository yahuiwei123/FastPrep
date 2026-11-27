"""Link the FreeSurfer 8.1 entry points against the matching host ABI."""

from pathlib import Path
import subprocess as sp
import shutil
import concurrent.futures


def build_drivers(w, fs, src, itk, build, stage):
    if (
        fs / "build-stamp.txt"
    ).read_text().strip() != "freesurfer-linux-ubuntu20_x86_64-8.1.0-20250718-f30dfed":
        raise ValueError("Expected FreeSurfer 8.1.0 f30dfed, Ubuntu 20 build")
    build.mkdir(parents=True, exist_ok=True)
    v = w / "vendor"
    includes = [
        build,
        src / "include",
        src / "utils",
        src / "mris_make_surfaces",
        src / "packages/nifti",
        src / "packages/minc",
        src / "packages/netcdf",
        itk,
    ]
    common = [
        str(Path(__file__).with_name("focal-cxx")),
        "-Wl,--as-needed",
        "-std=c++17",
        "-O2",
        "-fopenmp",
        "-DHAVE_OPENMP",
        "-no-pie",
        "-rdynamic",
        *["-I" + str(i) for i in includes],
    ]
    libs = [
        "-L" + str(fs / "python/packages/fsbindings"),
        "-l:fsbindings.cpython-38-x86_64-linux-gnu.so",
        fs / "python/lib/libpython3.8.a",
        "-ldl",
        "-lcrypt",
        "-lutil",
        "-lz",
        "-pthread",
    ]
    for folder, base in [
        ("mris_sphere", "mris_sphere"),
        ("mris_make_surfaces", "mris_place_surface"),
    ]:
        n = base + "_help_xml"
        b = (src / folder / (base + ".help.xml")).read_bytes()
        (build / (base + ".help.xml.h")).write_text(
            "unsigned char "
            + n
            + "[]={"
            + ",".join(map(str, b))
            + "};\nunsigned int "
            + n
            + "_len="
            + str(len(b))
            + ";\n"
        )

    def run(args):
        sp.run(args, check=True, cwd=build)

    top = v / "topology"
    t = v / "t2_pial"
    t1 = t / "t1"
    ts = t / "src"
    qs = top / "src"

    def sphere():
        base = (src / "include/mrisurf_base.h").read_text()
        text = (src / "utils/mrisurf_metricProperties_fast.h").read_text()
        parts = base[
            base.index("static float mrisNormalize(") : base.index(
                "int mris_sort_compare_float("
            )
        ]
        parts += text[
            text.index("static void MRISMP_makeDist2") : text.index("void MRISMP_copy(")
        ]
        parts += text[
            text.index("static void MRISMP_computeSurfaceDimensions") : text.index(
                "static void mrismp_OrientEllipsoid"
            )
        ]
        (build / "official_parts.inc").write_text(
            parts + "\n#undef COMPILING_MRIS_MP\n"
        )
        run(
            [
                *common,
                "-I" + str(qs),
                src / "mris_sphere/mris_sphere.cpp",
                qs / "help.cpp",
                qs / "metric_bridge.cpp",
                "-L" + str(stage / "topology/native"),
                "-Wl,-rpath,$ORIGIN",
                "-l:geometry.so",
                *libs,
                "-o",
                "qsphere",
            ]
        )

    def white():
        text = (src / "mris_make_surfaces/mris_place_surface.cpp").read_text()
        a = text.index(
            "    else {\n      // Compute the target xyz coordinate (l_location)"
        )
        b = text.index("    // This appears to adjust the cost weights", a)
        text = (
            text[:a]
            + '    else { fprintf(stderr,"T1-only placement driver\\n"); return 1; }\n'
            + text[b:]
        )
        (build / "entry.cpp").write_text(
            text.replace('#include "mris_multimodal_refinement.h"', "")
        )
        run(
            [
                *common,
                "-DmrisAsynchronousTimeStep=orderedMove",
                "-c",
                ts / "sync_bridge.cpp",
                "-o",
                "ordered.o",
            ]
        )
        run(
            [
                *common,
                "entry.cpp",
                t1 / "placement.cpp",
                "ordered.o",
                "-L" + str(stage / "t2_pial/t1/native"),
                "-Wl,-rpath,$ORIGIN",
                "-l:signed.so",
                *libs,
                "-o",
                "place-t1",
            ]
        )

    def pial():
        run(
            [
                *common,
                "-DT2GPU_COMPACT_HEAP",
                "-DT2GPU_TARGET_OVERLAP",
                "-DT2GPU_ORDERED_PIAL",
                "-DT2GPU_PARALLEL_PIAL",
                ts / "sync_bridge.cpp",
                ts / "place.cpp",
                ts / "lean_distance.cpp",
                *libs,
                "-L" + str(stage / "t2_pial/native"),
                "-Wl,-rpath,$ORIGIN",
                "-l:pial-field.so",
                "-o",
                "place-t2",
            ]
        )

    def distance():
        run(
            [
                *common,
                "-DT2GPU_COMPACT_HEAP",
                ts / "distance_field.cpp",
                ts / "lean_distance.cpp",
                *libs,
                "-o",
                "distance-field",
            ]
        )

    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(fn) for fn in (sphere, white, pial, distance)]:
            f.result()
    for name, dst in [
        ("qsphere", "topology/native/qsphere"),
        ("place-t1", "t2_pial/t1/native/place"),
        ("place-t2", "t2_pial/native/place"),
        ("distance-field", "t2_pial/native/distance-field"),
    ]:
        shutil.copy2(build / name, stage / dst)

"""Typed CLI adapters. Scientific imports happen only after argument parsing."""
import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _montage(parser, *, plane=True):
    if plane:
        parser.add_argument("--plane", choices=["axial", "sagittal", "coronal"], default="axial")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--slices", nargs="+", type=int, help="Explicit canonical voxel indices")
    selection.add_argument("--slice-range", nargs=2, type=int, metavar=("LOW", "HIGH"),
                           help="Inclusive canonical voxel range; default: full nonzero source-image range")
    parser.add_argument("--slice-count", type=_positive, default=20,
                        help="Automatic selection count (default 20; fewer if range is short)")
    parser.add_argument("--rows", type=_positive, default=4)
    parser.add_argument("--columns", type=_positive, default=5)
    parser.add_argument("--window", nargs=2, type=float, metavar=("LOW", "HIGH"),
                        help="Intensity window; default: finite nonzero source 1st/99th percentiles")


def build_parser(*, legacy_stdout=False):
    parser = argparse.ArgumentParser(description="Generate reusable QC evidence; complete rule decisions require protocol and review.")
    parser.add_argument("--version", action="version", version=f"neuroqc {__version__}")
    groups = parser.add_subparsers(dest="group", required=True)
    metrics = groups.add_parser("metrics", help="Measurements as JSON; status is not a complete rule pass")
    metric_commands = metrics.add_subparsers(dest="command", required=True)
    registration = metric_commands.add_parser("registration", help="MI on comparable physical grids")
    registration.add_argument("--fixed", required=True, type=Path)
    registration.add_argument("--registered", required=True, type=Path)
    registration.add_argument("--moving", type=Path, help="Optional baseline already on the fixed grid")
    registration.add_argument("--mi-min", type=float)
    registration.add_argument("--improvement-min", type=float)
    registration.add_argument("--threshold-source", help="Protocol/reference provenance for supplied thresholds")
    labels = metric_commands.add_parser("segmentation", help="Hard-label facts; empty labels are allowed")
    labels.add_argument("--image", required=True, type=Path)
    labels.add_argument("--labels", required=True, type=Path)
    labels.add_argument("--label", action="extend", nargs="+", type=int,
                        help="Label values; one list or repeated --label flags")
    labels.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=6,
                        help="3-D component adjacency: 6 (default), 18 or 26; brain masks use 26 unless their protocol specifies otherwise")
    mesh = metric_commands.add_parser("surface", help="Basic mesh facts; self-intersections are not tested")
    mesh.add_argument("--surface", required=True, type=Path)
    pair = metric_commands.add_parser("surface-pair", help="Geometric nesting and unsigned distances, not cortical thickness")
    pair.add_argument("--inner", required=True, type=Path)
    pair.add_argument("--outer", required=True, type=Path)
    pair.add_argument("--coordinate-unit", required=True, choices=["mm", "meter", "micron"])
    for command in (registration, labels, mesh, pair):
        command.add_argument("--output", type=Path, required=not legacy_stdout,
                             help="JSON output path" + ("; omit for stdout in this legacy entry point" if legacy_stdout else ""))

    render = groups.add_parser("render", help="PNG plus an always-written OUTPUT.png.json evidence sidecar")
    render_commands = render.add_subparsers(dest="command", required=True)
    image = render_commands.add_parser("image", help="Independent source-image montage")
    image.add_argument("--image", required=True, type=Path)
    _montage(image)
    comparison = render_commands.add_parser("comparison", help="Matched before/after rows with one shared intensity window")
    comparison.add_argument("--before", required=True, type=Path)
    comparison.add_argument("--after", required=True, type=Path)
    comparison.add_argument("--plane", choices=["axial", "sagittal", "coronal"], default="axial")
    selection = comparison.add_mutually_exclusive_group()
    selection.add_argument("--slices", nargs="+", type=int)
    selection.add_argument("--slice-range", nargs=2, type=int)
    comparison.add_argument("--slice-count", type=_positive, default=5)
    comparison.add_argument("--window", nargs=2, type=float)
    orthogonal = render_commands.add_parser("orthogonal", help="Source axial/sagittal/coronal views for quality and declared-modality review")
    orthogonal.add_argument("--image", required=True, type=Path)
    orthogonal.add_argument("--declared-modality", help="Caller-declared modality (e.g. T1w, T2w, FLAIR); recorded, never inferred")
    orthogonal.add_argument("--slice-count", type=_positive, default=5,
                            help="Representative slices per plane (default 5); supplement using render image --slices")
    orthogonal.add_argument("--window", nargs=2, type=float, metavar=("LOW", "HIGH"),
                            help="Shared intensity window; default: source 1st/99th percentiles")
    contour = render_commands.add_parser("contours", help="One label on source slices, including empty prediction regions")
    contour.add_argument("--image", required=True, type=Path)
    contour.add_argument("--labels", required=True, type=Path)
    contour.add_argument("--label", required=True, type=float)
    contour.add_argument("--color", default="#ff3030", help="Matplotlib contour color; default red #ff3030")
    _montage(contour)
    batch = render_commands.add_parser('label-batch', help='Reuse the axial source layout across workflow label views')
    batch.add_argument('--image', required=True, type=Path)
    batch.add_argument('--labels', required=True, type=Path)
    batch.add_argument('--source-output', type=Path)
    batch.add_argument('--contour', nargs=2, action='append', default=[], metavar=('LABEL', 'PNG'))
    batch.add_argument('--output', required=True, type=Path, help='JSON inventory of completed and failed views')
    reg = render_commands.add_parser("registration", help="Separate before/after orthogonal overlays")
    reg.add_argument("--fixed", required=True, type=Path)
    reg.add_argument("--registered", required=True, type=Path)
    reg.add_argument("--moving", type=Path, help="Optional comparable baseline; unavailable baseline is annotated")
    surface_contour = render_commands.add_parser("surface-contours", help="White/pial axial contours (red/blue)")
    surface_contour.add_argument("--image", required=True, type=Path)
    surface_contour.add_argument("--inner", action="extend", nargs="+", type=Path, default=[],
                                 help="White surfaces; at least one inner or outer surface is required")
    surface_contour.add_argument("--outer", action="extend", nargs="+", type=Path, default=[],
                                 help="Pial surfaces; at least one inner or outer surface is required")
    surface_contour.add_argument("--surface-space", choices=["freesurfer-tkr", "scanner-ras"], default="freesurfer-tkr",
                                 help="Surface coordinates in mm; tkr requires an MGH/MGZ reference")
    surface_contour.add_argument("--surface-reference", type=Path,
                                 help="Original FreeSurfer MGH/MGZ volume; required for tkr with a NIfTI image")
    surface_contour.add_argument("--title", default="Cortical surface QC")
    _montage(surface_contour, plane=False)
    surface = render_commands.add_parser("surface", help="surfplot six views; at least one hemisphere required")
    surface.add_argument("--lh", type=Path)
    surface.add_argument("--rh", type=Path)
    surface.add_argument("--title", default="Cortical surface multi-view QC")
    surface.add_argument('--surface-engine', choices=('reference', 'reuse'), default='reference')
    for command in (image, comparison, orthogonal, contour, reg, surface_contour, surface):
        command.add_argument("--output", required=True, type=Path,
                             help="PNG path (must end .png); also writes PATH.json with inputs/parameters")
    return parser


def main(argv=None, *, legacy_stdout=False):
    parser = build_parser(legacy_stdout=legacy_stdout)
    args = parser.parse_args(argv)
    options = vars(args).copy()
    group, command, output = (options.pop(key) for key in ("group", "command", "output"))
    try:
        from .evidence import metric_result, provenance, write_json
        if group == "metrics":
            from . import metrics
            if command == "registration":
                result = metrics.registration(args.fixed, args.moving, args.registered,
                                              args.mi_min, args.improvement_min, args.threshold_source)
            elif command == "segmentation":
                result = metrics.segmentation(args.image, args.labels, args.label, args.connectivity)
            elif command == "surface":
                result = metrics.surface(args.surface)
            else:
                from .surfaces import nesting_metrics, read_surface
                nested, measurements = nesting_metrics(*read_surface(args.outer), *read_surface(args.inner),
                                                       coordinate_unit=args.coordinate_unit)
                failed = measurements["surfaceIntersection"] is True or bool(measurements["innerOutsideVertexCount"])
                result = metric_result(measurements, failed=failed,
                                       metric_status="fail" if failed else "pass" if nested else "unknown",
                                       missing=["vertex_correspondence", "thickness", "ribbon_coverage", "visual_review"])
                result["provenance"] = provenance({"inner": args.inner, "outer": args.outer},
                                                   {"coordinate_unit": args.coordinate_unit})
            result["provenance"]["command"] = ["python", "-m", "neuroqc"] + list(sys.argv[1:] if argv is None else argv)
            if output is None:
                print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
            else:
                write_json(output, result)
        else:
            from . import rendering
            if command == 'label-batch':
                invocation = ['python', '-m', 'neuroqc'] + list(sys.argv[1:] if argv is None else argv)
                result = rendering.label_montages(args.image, args.labels, args.source_output, args.contour, invocation)
                write_json(output, result)
                return
            # Library keywords identify file roles; argparse uses shorter flag names.
            for key in ("image", "before", "after", "labels", "fixed", "registered", "moving"):
                if key in options:
                    options[key + "_path"] = options.pop(key)
            function = getattr(rendering, "render_" + command.replace("-", "_"))
            result = function(output=output, **options)
            result["command"] = ["python", "-m", "neuroqc"] + list(sys.argv[1:] if argv is None else argv)
            write_json(str(output) + ".json", result)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

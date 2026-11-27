"""Run one DeepPrep SUGAR hemisphere registration stage from explicit files.

Both hemispheres are supported; the bundled weights live under
``weights/<ico_level>/<hemisphere>_<stage>_904_<ico_level>.model``.
"""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import types

import nibabel as nib
import numpy as np


ICO_LEVELS = ("fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6")


# Upstream SUGAR release names; ``rigid`` is a substring of ``NoRigid``, so the
# stage cannot be told apart by substring matching and the exact name is required.
CHECKPOINT_FILENAMES = {
    "rigid": "{hemisphere}_Rigid_904_fsaverage6.model",
    "nonlinear": "{hemisphere}_NoRigid_904_{level}.model",
}


def existing_file(path, name):
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty {name}: {path}")
    return path


def load_sphere(path, name):
    path = existing_file(path, name)
    xyz, faces = nib.freesurfer.read_geometry(str(path))
    xyz = xyz.astype(np.float32)
    faces = faces.astype(np.int64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Invalid triangular sphere geometry for {name}: {path}")
    if not np.isfinite(xyz).all() or np.any(np.linalg.norm(xyz, axis=1) == 0):
        raise ValueError(f"Invalid sphere coordinates for {name}: {path}")
    return path, xyz, faces


def load_sulc(path, vertex_count, name):
    path = existing_file(path, name)
    values = nib.freesurfer.read_morph_data(str(path)).astype(np.float32)
    if values.shape != (vertex_count,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain one finite value per sphere vertex")
    if float(values.std()) == 0:
        raise ValueError(f"{name} has zero variance and cannot be normalized")
    return path, values


def normalize_sulc(values):
    normalized = (values - values.mean()) / values.std()
    low = normalized < -3
    high = normalized > 3
    normalized[low] = -3 - (1 - np.exp(3 - np.abs(normalized[low])))
    normalized[high] = 3 + (1 - np.exp(3 - np.abs(normalized[high])))
    return normalized.astype(np.float32)


def select_device(name):
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        index = 0
    else:
        index_name = name[5:] if name.startswith("cuda:") else name
        try:
            index = int(index_name)
        except ValueError as error:
            raise ValueError(
                "device must be auto, cpu, cuda, cuda:N, or a non-negative GPU index"
            ) from error
        if index < 0:
            raise ValueError("GPU index must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch sees no GPU")
    if index >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {index} was requested, but PyTorch sees "
            f"{torch.cuda.device_count()} device(s)"
        )
    return torch.device(f"cuda:{index}")


def validate_runtime(device):
    import torch

    runtime = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type != "cuda":
        return runtime
    try:
        import torch_scatter
    except ImportError as error:
        raise RuntimeError(
            "SUGAR CUDA execution requires a CUDA-enabled torch-scatter build"
        ) from error
    try:
        scatter_cuda = int(torch.ops.torch_scatter.cuda_version())
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(
            "torch-scatter has no CUDA runtime; install a build compiled for CUDA"
        ) from error
    if scatter_cuda <= 0:
        raise RuntimeError(
            "torch-scatter is CPU-only; SUGAR CUDA execution requires a CUDA build"
        )
    try:
        with torch.inference_mode():
            source = torch.tensor([1.0, 2.0], device=device)
            index = torch.tensor([0, 0], dtype=torch.long, device=device)
            result = torch_scatter.scatter_add(source, index, dim=0)
            if result.item() != 3.0:
                raise RuntimeError("unexpected torch-scatter CUDA result")
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(
            f"torch-scatter CUDA preflight failed on {device}: {error}"
        ) from error
    runtime.update({
        "torch_scatter_version": torch_scatter.__version__,
        "torch_scatter_cuda_version": scatter_cuda,
        "gpu_name": torch.cuda.get_device_name(device),
    })
    try:
        import pytorch3d
        from pytorch3d.ops import knn_points
    except ImportError as error:
        raise RuntimeError("SUGAR CUDA execution requires a CUDA-enabled PyTorch3D build") from error
    try:
        with torch.inference_mode():
            points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device=device)
            nearest = knn_points(points[:, :1], points, K=1)
            if (nearest.idx.device != device or nearest.dists.device != device
                    or nearest.idx.item() != 0 or nearest.dists.item() != 0.0):
                raise RuntimeError("unexpected PyTorch3D CUDA KNN result or device")
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(f"PyTorch3D CUDA KNN preflight failed on {device}: {error}") from error
    runtime.update({"pytorch3d_version": pytorch3d.__version__, "pytorch3d_cuda_knn_verified": True})
    return runtime


def load_source(source_root):
    source_root = Path(source_root).resolve()
    entrypoint = existing_file(
        source_root / "sphere_registrate_norigid_multi_level_multi_fixed_predict.py",
        "SUGAR inference source",
    )
    for relative, label in (
        ("gatunet_model.py", "SUGAR model source"),
        ("utils/auxi_data.py", "SUGAR auxiliary geometry source"),
        ("utils/interp_fine.py", "SUGAR interpolation source"),
        ("utils/pooling.py", "SUGAR pooling source"),
        ("utils/rotate_matrix.py", "SUGAR rotation source"),
        ("utils/negative_area_triangle.py", "SUGAR topology QC source"),
    ):
        existing_file(source_root / relative, label)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    gatunet = importlib.import_module("gatunet_model")
    pooling = importlib.import_module("utils.pooling")
    for module, expected_root in ((gatunet, source_root), (pooling, source_root / "utils")):
        module_path = Path(module.__file__).resolve()
        if expected_root not in module_path.parents:
            raise ValueError(f"SUGAR imported a conflicting module outside source-root: {module_path}")
    legacy = sys.modules.setdefault("featreg", types.ModuleType("featreg"))
    legacy.__path__ = [str(source_root)]
    legacy_utils = sys.modules.setdefault("featreg.utils", types.ModuleType("featreg.utils"))
    legacy_utils.__path__ = [str(source_root / "utils")]
    legacy.gatunet_model = gatunet
    legacy.utils = legacy_utils
    legacy_utils.pooling = pooling
    sys.modules["featreg.gatunet_model"] = gatunet
    sys.modules["featreg.utils.pooling"] = pooling
    inference = importlib.import_module("sphere_registrate_norigid_multi_level_multi_fixed_predict")
    auxi_data = importlib.import_module("utils.auxi_data")
    interpolation = importlib.import_module("utils.interp_fine")
    rotation = importlib.import_module("utils.rotate_matrix")
    negative = importlib.import_module("utils.negative_area_triangle")
    return entrypoint, inference, auxi_data, interpolation, rotation, negative


def checkpoint_filename(hemisphere, stage, level):
    if stage not in CHECKPOINT_FILENAMES:
        raise ValueError(f"Unknown SUGAR stage: {stage}")
    if hemisphere not in ("lh", "rh"):
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    if level not in ICO_LEVELS:
        raise ValueError(f"Unknown ico level: {level}")
    if stage == "rigid" and level != "fsaverage6":
        raise ValueError(f"SUGAR rigid checkpoints exist only for fsaverage6, not {level}")
    return CHECKPOINT_FILENAMES[stage].format(hemisphere=hemisphere, level=level)


def bundled_checkpoint(model_root, hemisphere, stage, level):
    return model_root / "weights" / level / checkpoint_filename(hemisphere, stage, level)


def checkpoint_identity(path, hemisphere, stage, level):
    path = existing_file(path, "SUGAR checkpoint")
    expected = checkpoint_filename(hemisphere, stage, level)
    if path.name != expected:
        raise ValueError(
            f"SUGAR {stage} checkpoint for {hemisphere}/{level} must identify the "
            f"{level} model and be named {expected}: {path.name}"
        )
    return path


def load_model(path, device, hemisphere, stage, level):
    import torch

    path = checkpoint_identity(path, hemisphere, stage, level)
    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"SUGAR checkpoint does not contain a model: {path}")
    model = checkpoint["model"].to(device)
    model.eval()
    return path, model


def tensor_data(sulc, xyz, faces, device):
    import torch

    count = len(xyz)
    zeros_float = np.zeros(count, dtype=np.float32)
    zeros_int = np.zeros(count, dtype=np.int64)
    return [
        torch.from_numpy(normalize_sulc(sulc)).unsqueeze(0).to(device),
        torch.from_numpy(zeros_float).unsqueeze(0).to(device),
        torch.from_numpy((xyz / 100).astype(np.float32)).unsqueeze(0).to(device),
        torch.from_numpy(faces).unsqueeze(0).to(device),
        torch.from_numpy(zeros_int).unsqueeze(0).to(device),
    ]


def resample_sulc(interpolation, source_xyz, target_xyz, source_sulc, device):
    import torch

    return interpolation.resample_sphere_surface_barycentric(
        torch.from_numpy(source_xyz).to(device),
        torch.from_numpy(target_xyz).to(device),
        torch.from_numpy(source_sulc).unsqueeze(1).to(device),
        device=device,
    ).squeeze(1).detach().cpu().numpy().astype(np.float32)


def load_template_metadata(path, hemisphere, sphere_path, sulc_path):
    path = existing_file(path, "SUGAR template metadata")
    metadata = json.loads(path.read_text())
    expected = {
        "template": "fsaverage6",
        "hemisphere": hemisphere,
        "vertex_count": 40962,
        "sphere": str(sphere_path),
        "sulc": str(sulc_path),
    }
    for key, value in expected.items():
        actual = metadata.get(key)
        if key in ("sphere", "sulc"):
            try:
                actual = str(Path(actual).resolve())
            except TypeError as error:
                raise ValueError(f"SUGAR template metadata has no valid {key}") from error
        if actual != value:
            raise ValueError(
                f"SUGAR template metadata {key} must be {value!r}, got {actual!r}"
            )
    return path, metadata


def validate_template(auxi_data, template_xyz, template_faces):
    internal_xyz, internal_faces = auxi_data.get_geometry_by_ico_level("fsaverage6")
    if len(template_xyz) != 40962:
        raise ValueError("SUGAR fixed template must be the 40962-vertex fsaverage6 sphere")
    if template_faces.shape != internal_faces.shape or not np.array_equal(template_faces, internal_faces):
        raise ValueError("SUGAR fixed template topology differs from its embedded fsaverage6 topology")
    if template_xyz.shape != internal_xyz.shape or not np.allclose(
        template_xyz, internal_xyz, atol=1e-4, rtol=0
    ):
        raise ValueError("SUGAR fixed template coordinates differ from its embedded fsaverage6 sphere")


def run_inference(inference, moving, fixed, models, faces, levels, device):
    import torch

    with torch.inference_mode():
        result = inference.infer(
            moving, fixed, models, faces, levels, ["sulc"] * len(levels), device
        )
    return {"moved_sphere": result[1], "euler_angle": result[6]}


def validate_output_sphere(xyz, faces, expected_vertices, name):
    xyz = np.asarray(xyz)
    if xyz.shape != (expected_vertices, 3) or not np.isfinite(xyz).all():
        raise ValueError(f"{name} must contain {expected_vertices} finite 3D vertices")
    if faces.ndim != 2 or faces.shape[1] != 3 or np.any(faces < 0) or np.any(faces >= len(xyz)):
        raise ValueError(f"{name} has invalid faces")
    radii = np.linalg.norm(xyz, axis=1)
    if not np.allclose(radii, 100.0, atol=1e-2, rtol=0):
        raise ValueError(f"{name} is not a radius-100 sphere")
    return xyz.astype(np.float32)


def require_zero_negative_faces(negative, faces, xyz, name):
    count = int(negative.count_negative_area(faces, xyz).item())
    if count:
        raise ValueError(f"{name} produced {count} flipped faces")
    return count


def write_provenance(work, payload):
    path = work / "provenance.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def add_common_arguments(parser, model_root):
    parser.add_argument("--sphere", type=Path, required=True,
                        help="native sphere; rigid.sphere for nonlinear mode")
    parser.add_argument("--sulc", type=Path, required=True,
                        help="native sulc values associated with the native sphere vertices")
    parser.add_argument("--template-sphere", type=Path, required=True,
                        help="hemisphere-matched fsaverage6 sphere")
    parser.add_argument("--template-sulc", type=Path, required=True,
                        help="hemisphere-matched fsaverage6 sulc")
    parser.add_argument("--template-metadata", type=Path, required=True,
                        help="JSON binding template id, hemisphere and exact sphere/sulc paths")
    parser.add_argument("--hemisphere", choices=("lh", "rh"), required=True)
    parser.add_argument("--source-root", type=Path, default=model_root / "source",
                        help="bundled SUGAR source directory")
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, cuda, cuda:N, or a visible non-negative GPU index")
    parser.add_argument("--work-dir", type=Path, required=True)


def parse_args(argv=None):
    model_root = Path(__file__).resolve().parents[2] / 'model/sugar'
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    rigid = subparsers.add_parser("rigid", help="estimate the global SUGAR rotation")
    add_common_arguments(rigid, model_root)
    rigid.add_argument("--checkpoint", type=Path,
                       help="defaults to weights/fsaverage6/<hemisphere>_Rigid_904_fsaverage6.model")
    nonlinear = subparsers.add_parser("nonlinear", help="estimate the four-level SUGAR deformation")
    add_common_arguments(nonlinear, model_root)
    for level in ICO_LEVELS:
        nonlinear.add_argument(f"--checkpoint-{level}", type=Path,
                               help=f"defaults to weights/{level}/"
                                    f"<hemisphere>_NoRigid_904_{level}.model")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_root = Path(__file__).resolve().parents[2] / 'model/sugar'
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from spherical_geometry import prepare_file, repair, save_checked, require_valid
    from spherical_interpolation import interpolate
    sphere_path, native_xyz, native_faces = load_sphere(args.sphere, "native sphere")
    sulc_path, native_sulc = load_sulc(args.sulc, len(native_xyz), "native sulc")
    template_path, template_xyz, template_faces = load_sphere(args.template_sphere, "fsaverage6 sphere")
    template_sulc_path, template_sulc = load_sulc(
        args.template_sulc, len(template_xyz), "fsaverage6 sulc"
    )
    template_metadata_path, _ = load_template_metadata(
        args.template_metadata,
        args.hemisphere,
        template_path,
        template_sulc_path,
    )
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    device = select_device(args.device)
    runtime = validate_runtime(device)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=False)
    prepared = work / (args.hemisphere + '.input.sphere')
    input_geometry = prepare_file(sphere_path, prepared)
    _, native_xyz, native_faces = load_sphere(prepared, 'validated input sphere')
    require_valid(template_xyz, template_faces)
    entrypoint, inference, auxi_data, interpolation, rotation, negative = load_source(args.source_root)
    validate_template(auxi_data, template_xyz, template_faces)
    xyzs, all_faces = auxi_data.get_geometry_all_level_torch()
    required_levels = ("fsaverage6",) if args.mode == "rigid" else ICO_LEVELS
    faces = {key: all_faces[key].to(device) for key in required_levels}
    moving_sulc = resample_sulc(
        interpolation, native_xyz, template_xyz, native_sulc, device
    )
    moving = tensor_data(moving_sulc, template_xyz, template_faces, device)
    fixed = tensor_data(template_sulc, template_xyz, template_faces, device)

    if args.mode == "rigid":
        checkpoint_arg = args.checkpoint or bundled_checkpoint(
            model_root, args.hemisphere, "rigid", "fsaverage6"
        )
        checkpoint, model = load_model(
            checkpoint_arg, device, args.hemisphere, "rigid", "fsaverage6"
        )
        result = run_inference(
            inference, moving, fixed, [model], faces, ["fsaverage6"], device
        )
        import torch

        # Rotation preserves orientation mathematically. Float32 coordinate
        # arithmetic can reverse extremely thin triangles, so evaluate this
        # geometric transform in float64, then verify the serialized result.
        native_unit = torch.from_numpy(native_xyz.astype(np.float64) / 100).to(device)
        angles = result['euler_angle'].to(dtype=torch.float64)
        if angles.shape != (1, 3) or not torch.isfinite(angles).all():
            raise ValueError('Rigid model must produce one finite global Euler rotation')
        registered = rotation.apply_rotate_matrix(angles, native_unit, norm=True)
        registered_xyz = validate_output_sphere(
            registered.detach().cpu().numpy() * 100, native_faces, len(native_xyz), "rigid output"
        )
        raw_output = work / f'{args.hemisphere}.rigid.sphere.raw'
        nib.freesurfer.write_geometry(str(raw_output), registered_xyz, native_faces)
        registered_xyz, output_geometry = repair(registered_xyz, native_faces)
        output = work / f"{args.hemisphere}.rigid.sphere"
        output_geometry['readback'] = save_checked(output, registered_xyz, native_faces)
        provenance = write_provenance(work, {
            "operation": "sugar-rigid-registration",
            "hemisphere": args.hemisphere,
            "native_sphere": str(sphere_path),
            "native_sulc": str(sulc_path),
            "template_sphere": str(template_path),
            "template_sulc": str(template_sulc_path),
            "template_metadata": str(template_metadata_path),
            "checkpoint": str(checkpoint),
            "source_entrypoint": str(entrypoint),
            "device": str(device),
            "runtime": runtime,
            "output": str(output),
            "negative_faces": output_geometry["readback"]["negative_faces"],
            "raw_output": str(raw_output),
            "rotation_arithmetic": "float64",
            "euler_angles_radians": angles.detach().cpu().tolist(),
            "input_geometry": input_geometry,
            "output_geometry": output_geometry,
        })
        print(json.dumps({"rigid_sphere": str(output), "provenance": str(provenance)}))
        return

    checkpoints = []
    models = []
    for level in ICO_LEVELS:
        checkpoint_arg = getattr(args, f"checkpoint_{level}") or bundled_checkpoint(
            model_root, args.hemisphere, "nonlinear", level
        )
        checkpoint, model = load_model(
            checkpoint_arg, device, args.hemisphere, "nonlinear", level
        )
        checkpoints.append(checkpoint)
        models.append(model)
    result = run_inference(
        inference, moving, fixed, models, faces, list(ICO_LEVELS), device
    )
    import torch

    template_moved = result['moved_sphere'].detach().cpu().numpy() * 100
    nib.freesurfer.write_geometry(
        str(work / f'{args.hemisphere}.template.reg.raw'),
        template_moved, template_faces)
    template_moved, template_geometry = repair(template_moved, template_faces)
    template_output = work / f'{args.hemisphere}.template.reg'
    template_geometry['readback'] = save_checked(template_output, template_moved, template_faces)
    # Nearest three vertices need not be one mesh triangle. Use a continuous
    # map on real containing triangles, with explicit failure on lookup gaps.
    # Resampling onto a different triangulation can still reverse very thin
    # native faces, so the mapped native mesh must also pass bounded repair.
    moved_native, interpolation_report = interpolate(
        template_xyz, template_faces, native_xyz, template_moved)
    moved_native *= 100 / np.linalg.norm(moved_native, axis=1, keepdims=True)
    raw_output = work / f"{args.hemisphere}.sphere.reg.raw"
    nib.freesurfer.write_geometry(
        str(raw_output), moved_native, native_faces
    )
    repaired_xyz, output_geometry = repair(moved_native, native_faces)
    repaired_xyz = validate_output_sphere(repaired_xyz, native_faces, len(native_xyz), "nonlinear output")
    output = work / f"{args.hemisphere}.sphere.reg"
    output_geometry['readback'] = save_checked(output, repaired_xyz, native_faces)
    provenance = write_provenance(work, {
        "operation": "sugar-nonlinear-registration",
        "hemisphere": args.hemisphere,
        "rigid_sphere": str(sphere_path),
        "native_sulc": str(sulc_path),
        "template_sphere": str(template_path),
        "template_sulc": str(template_sulc_path),
        "template_metadata": str(template_metadata_path),
        "checkpoints": {level: str(path) for level, path in zip(ICO_LEVELS, checkpoints)},
        "source_entrypoint": str(entrypoint),
        "device": str(device),
        "runtime": runtime,
        "raw_output": str(raw_output),
        "output": str(output),
        "negative_faces_before_repair": output_geometry['before']['negative_faces'],
        "negative_faces_after_repair": output_geometry["readback"]["negative_faces"],
        "repair_iterations": output_geometry['iterations'],
        "input_geometry": input_geometry,
        "output_geometry": output_geometry,
        "template_geometry": template_geometry,
        "interpolation": interpolation_report,
    })
    print(json.dumps({
        "raw_registered_sphere": str(raw_output),
        "registered_sphere": str(output),
        "provenance": str(provenance),
    }))


if __name__ == "__main__":
    main()

"""Prepare portable command paths and BIDS sessions for a native FastPrep run."""

import argparse
import glob
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from reference_runtime import checked_file, read_json, write_json
from session_inputs import prepare_plans
import mni_template


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value, variables=None):
    expanded = os.path.expanduser(str(value))
    for name, replacement in (variables or {}).items():
        expanded = expanded.replace("${" + name + "}", str(replacement))
    expanded = os.path.expandvars(expanded)
    if "$" in expanded:
        raise ValueError("Unresolved environment variable in path: " + expanded)
    path = Path(expanded)
    return path if path.is_absolute() else ROOT / path


def executable(value):
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    selected = shutil.which(expanded) if "/" not in expanded else None
    path = Path(selected) if selected else resolve_path(expanded)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError("Executable is unavailable: " + str(path))
    with path.open("rb") as stream:
        header = stream.readline(512)
    if header.startswith(b"#!"):
        interpreter = shlex.split(header[2:].decode("utf-8").strip())[0]
        if not Path(interpreter).is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError("Missing script interpreter " + interpreter + " required by " + str(path)
                             + "; install the tool's system dependencies before preparing a run")
    return str(path.absolute())


def resources(fs_home):
    records = []
    for hemi, letter in (("lh", "L"), ("rh", "R")):
        records.append({"id": "fsaverage6-" + letter + "-sphere",
                        "path": str(checked_file(fs_home / "subjects/fsaverage6/surf" / (hemi + ".sphere"))),
                        "contract": {"space": "fsaverage6", "hemi": letter}})
    for identity, relative in (("fsaverage6-orig", "subjects/fsaverage6/mri/orig.mgz"),
                               ("freesurfer-gca-brain", "average/RB_all_2020-01-02.gca")):
        records.append({"id": identity, "path": str(checked_file(fs_home / relative))})
    return records


def resource_files(pattern):
    matches = [checked_file(path, allow_empty=Path(path).suffix == ".py")
               for path in sorted(glob.glob(str(pattern), recursive=True)) if Path(path).is_file()]
    if not matches:
        raise ValueError("Required resource is missing: " + str(pattern))
    return [str(path) for path in matches]


def build_runtime(settings, graph):
    fs_home = resolve_path(settings["freesurfer_home"])
    fsl_dir = resolve_path(settings.get("fsl_dir", "${FSLDIR}"))
    license_path = checked_file(resolve_path(settings.get("fs_license", str(fs_home / "license.txt"))))
    interpreters = {name: executable(value) for name, value in settings["python"].items()}
    required_python = {"helper", "bfc", "synthseg", "sugar", "hdbet", "qc", "geometry"}
    if not required_python.issubset(interpreters):
        raise ValueError("Missing Python profiles: " + ", ".join(sorted(required_python - set(interpreters))))
    variables = {"FREESURFER_HOME": fs_home, "PROJECT_ROOT": ROOT}
    model_recipes = read_json(ROOT / "conf/models.json")["commands"]
    native_recipes = read_json(ROOT / "conf/native_commands.json")["commands"]
    required = {command for node in graph["nodes"] for command in node.get("commands", node.get("dependencies", []))}
    commands = {}
    for identity in sorted(required):
        if identity.startswith("freesurfer:"):
            recipe = native_recipes[identity]
            commands[identity] = {"argv_prefix": [executable(str(fs_home / "bin" / recipe["executable"]))],
                                  "environment": {"FREESURFER_HOME": str(fs_home), "FS_LICENSE": str(license_path),
                                                  "PATH": str(fs_home / "bin") + os.pathsep + os.environ.get("PATH", "")},
                                  "files": [path for relative in recipe.get("files", [])
                                            for path in resource_files(fs_home / relative)]}
        elif identity == "fsl:flirt":
            commands[identity] = {"argv_prefix": [executable(str(fsl_dir / "bin/flirt"))],
                                  "environment": {"FSLDIR": str(fsl_dir), "FSLOUTPUTTYPE": "NIFTI_GZ"}, "files": []}
        elif identity.startswith('ants:'):
            name = identity.split(':')[1]
            folder = settings.get('ants_bin', os.environ.get('ANTSPATH', ''))
            commands[identity] = {'argv_prefix': [executable(str(Path(folder) / name))], 'environment': {}, 'files': []}
        elif identity == "neuroqc:measure-cortical-pair":
            script = checked_file(ROOT / "vendor/neuroqc/measure_cortical_pair.py")
            commands[identity] = {"argv_prefix": [interpreters["qc"], str(script)], "environment": {}, "files": []}
        else:
            if identity not in model_recipes:
                raise ValueError("No portable recipe for " + identity)
            recipe = model_recipes[identity]
            dependencies = []
            for pattern in recipe.get("files", []):
                dependencies.extend(resource_files(resolve_path(pattern, variables)))
            environment = {name: str(value).replace("${PROJECT_ROOT}", str(ROOT)).replace("${FREESURFER_HOME}", str(fs_home))
                           for name, value in recipe.get("environment", {}).items()}
            profile = 'fireants' if identity == 'fireants:register' and 'fireants' in interpreters else recipe['python']
            commands[identity] = {"argv_prefix": [interpreters[profile], str(checked_file(ROOT / recipe["script"]))],
                                  "environment": environment, "files": sorted(set(dependencies))}
    return {"schema": "fastprep.runtime/1", "package_root": str(ROOT), "commands": commands,
            "resources": resources(fs_home), "helper_python": interpreters["helper"], "python": interpreters,
            "qc": {"render_python": interpreters["qc"], "pair_python": interpreters["geometry"],
                   "fs_home": str(fs_home), "vendor_root": str(ROOT / "vendor")}}


def environment_probes(runtime):
    imports = {"helper": "numpy, scipy, nibabel, SimpleITK, sklearn", "synthseg": "tensorflow, nibabel, scipy",
               "bfc": "numpy, scipy, nibabel, SimpleITK, torch",
               "sugar": "torch, torch_geometric, torch_scatter, pytorch3d, nibabel",
               "hdbet": "torch, nnunetv2.inference.predict_from_raw_data, SimpleITK",
               "qc": "numpy, scipy, nibabel, matplotlib, vtk, surfplot",
               "geometry": "numpy, scipy, nibabel, trimesh, fcl, rtree"}
    reports = {}
    for profile, modules in imports.items():
        command = [runtime["python"][profile], "-B", "-c", "import " + modules + "; import sys; print(sys.version)"]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"} if profile == "synthseg" else os.environ.copy()
        result = subprocess.run(command, capture_output=True, text=True, env=environment)
        if result.returncode:
            raise RuntimeError("Python profile " + profile + " cannot load required libraries:\n" + result.stderr[-4000:])
        reports[profile] = {"python": runtime["python"][profile], "version": result.stdout.strip(), "imports": modules}
    for identity, flag in (("freesurfer:recon-all", "--version"), ("fsl:flirt", "-version")):
        command = runtime["commands"][identity]
        result = subprocess.run([*command["argv_prefix"], flag], env={**os.environ, **command["environment"]},
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Native version probe failed: " + identity + "\n" + result.stderr[-2000:])
        reports[identity] = {"version": (result.stdout + result.stderr).strip()}
    if 'fireants:register' in runtime['commands']:
        command = runtime['commands']['fireants:register']
        result = subprocess.run([*command['argv_prefix'], '--probe'], capture_output=True, text=True,
                                env={**os.environ, **command['environment']})
        if result.returncode:
            raise RuntimeError('FireANTs environment probe failed:\n' + result.stderr[-4000:])
        reports['fireants:register'] = {'version': result.stdout.strip()}
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--software", required=True)
    parser.add_argument("--bids-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-units-json", type=Path, help="Plan these worker sessions independently")
    parser.add_argument("--subjects", default="")
    parser.add_argument("--sessions", default="")
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--selection-policy", choices=("resolution-snr", "strict"), default="resolution-snr",
                        help="Rank same-session acquisitions by resolution then estimated SNR, or require explicit choices")
    parser.add_argument("--t2-mode", choices=["auto", "require", "off"], default="auto")
    parser.add_argument("--bfc-method", choices=["rbf", "gaussian"], default="rbf")
    parser.add_argument('--mni-backend', choices=['fireants', 'ants'], default='fireants')
    parser.add_argument("--skip-environment-probes", action="store_true")
    mni_template.options(parser)
    args = parser.parse_args()
    output = Path(args.output).expanduser().absolute()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty control directory: " + str(output))
    graph = read_json(ROOT / "conf/workflow.json")
    select_bfc_method(graph, args.bfc_method)
    select_mni_backend(graph, args.mni_backend)
    runtime = build_runtime(read_json(args.software), graph)
    runtime['mni_template'] = mni_template.template_resources(args)
    mni_template.validate_template(runtime['mni_template'])
    registration = 'fireants:register' if args.mni_backend == 'fireants' else 'ants:antsRegistration'
    runtime['commands'][registration]['files'].extend(runtime['mni_template'][key] for key in ('t1w', 'mask'))
    subjects = [value.removeprefix("sub-") for value in args.subjects.split(",") if value]
    sessions = [value.removeprefix("ses-") for value in args.sessions.split(",") if value]
    if args.batch_units_json:
        import re
        from nibabel.filebasedimages import ImageFileError
        if subjects or sessions or args.selections:
            parser.error("--batch-units-json already selects the exact sessions")
        units = read_json(args.batch_units_json)
        if not units or len(set(units)) != len(units) or any(not re.fullmatch(r"sub-[A-Za-z0-9]+(?:/ses-[A-Za-z0-9]+)?", u) for u in units):
            raise ValueError("Invalid batch session list")
        if not read_json(Path(args.bids_dir) / "dataset_description.json").get("BIDSVersion"):
            raise ValueError("The BIDS dataset description must declare BIDSVersion")
        plans, t2_series, rejected = [], {}, {}
        for unit in units:
            uid = unit.replace("/", "_")
            try:
                items, series = prepare_plans(args.bids_dir, t2_mode=args.t2_mode, selection_policy=args.selection_policy, unit=unit)
                if [item["analysis_unit_id"] for item in items] != [uid]:
                    raise ValueError("Prepared session disagrees with input directory")
            except (OSError, ValueError, EOFError, ImageFileError) as exc:
                rejected[uid] = str(exc)
                print(f"Rejected input {uid}: {exc}", file=sys.stderr)
            else:
                plans.extend(items)
                t2_series.update(series)
        runtime["rejected_sessions"] = rejected
    else:
        plans, t2_series = prepare_plans(args.bids_dir, subjects, sessions, args.selections, args.t2_mode, args.selection_policy)
    runtime["session_input_plan"] = plans
    runtime["bids_dir"] = str(Path(args.bids_dir).expanduser().resolve())
    runtime["environment_checks"] = "not_run" if args.skip_environment_probes else environment_probes(runtime)
    pial = next(node for node in graph["nodes"] if node["node_id"] == "anat_pial_surface")
    pial["parameters"].update(t2_mode=args.t2_mode, t2_series_ids=t2_series)
    write_json(output / "runtime.json", runtime)
    write_json(output / "workflow.json", graph)
    write_json(output / "inputs.json", plans)
    print("Prepared " + str(len(plans)) + " sessions in " + str(output))
    print("Input structural checks passed; image quality and reconstruction remain to be reviewed.")


def select_mni_backend(graph, backend):
    if backend not in {'fireants', 'ants'}:
        raise ValueError('MNI backend must be fireants or ants')
    node = next(node for node in graph['nodes'] if node['node_id'] == 'session_normalize_mni')
    node['parameters']['backend'] = backend
    node['dependencies'] = ['fireants:register' if backend == 'fireants' else 'ants:antsRegistration',
                            'ants:antsApplyTransforms']


def select_bfc_method(graph, method):
    if method not in {"rbf", "gaussian"}:
        raise ValueError("BFC method must be rbf or gaussian")
    for node in graph["nodes"]:
        if node["node_id"] == "anat_tissue_bias_correct":
            node["parameters"]["method"] = method
        elif node["node_id"] == "anat_pial_surface":
            node["parameters"]["bias"]["method"] = method


if __name__ == "__main__":
    main()

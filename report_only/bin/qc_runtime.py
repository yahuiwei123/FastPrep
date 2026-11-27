"""Run portable, evidence-producing QC for completed FastPrep stages."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import filecmp
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from reference_runtime import artifact_path, checked_file, read_json, write_json


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MESH_PORTS = {
    "orig_nofix_surf", "orig_topofix_surf", "orig_surf", "orig_premesh_surf",
    "smoothwm_nofix_surf", "inflated_nofix_surf", "qsphere_nofix_surf",
    "white_preaparc_surf", "white_surf", "pial_surf", "pial_t1_surf",
    "smoothwm_surf", "inflated_surf", "sphere_surf", "sphere_reg_surf",
}
LABEL_PORTS = {
    "mask_nii", "mask_mgz",
    "aseg_auto_mgz", "aseg_synthseg_mgz", "aseg_auto_noccseg_mgz",
    "aseg_synthseg_withcc_mgz", "aseg_presurf_mgz", "wm_seed_mgz",
    "wm_asegedit_mgz", "wm_mgz", "aseg_presurf_hypos_mgz", "filled_mgz", "filled_pretess_mgz",
    "ribbon_mgz", "aseg_mgz", "aparc_aseg_mgz",
}


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def overall_status(checks):
    if any(item["status"] == "fail" and item["severity"] == "error" for item in checks):
        return "fail"
    if any(item["status"] == "unknown" and item["severity"] in {"error", "warning"}
           for item in checks):
        return "unknown"
    return "pass"


def _unit_id(spec):
    return spec.get("analysis_unit_id", spec.get("subject_id", spec.get("subject")))


def _session_plan(runtime, uid):
    matches = [item for item in runtime.get("session_input_plan", [])
               if item.get("analysis_unit_id") == uid]
    if len(matches) != 1:
        raise ValueError("Expected one session input plan for " + str(uid))
    plan = matches[0]
    if plan.get("subject_id", "") + ("_ses-" + plan.get("session_id", "")
                                      if plan.get("session_id") else "") != uid:
        raise ValueError("Session identity differs from BIDS subject/session")
    for item in [plan["reference"], *plan.get("companions", [])]:
        if (item.get("analysis_unit_id") != uid or item.get("subject_id") != plan.get("subject_id")
                or item.get("session_id", "") != plan.get("session_id", "")):
            raise ValueError("Acquisition identity crosses session boundary")
        if item.get("modality") not in {"T1w", "T2w", "FLAIR"}:
            raise ValueError("Unsupported BIDS-declared structural modality")
        if item.get("structural_status") != "pass" or item.get("visual_review") != "pending":
            raise ValueError("Input plan must record structural checks and pending visual review")
        facts = item.get("probe", {}).get("facts", {})
        if (len(facts.get("shape", [])) != 3 or not facts.get("finite") or not facts.get("nonzero")
                or not facts.get("affine") or not facts.get("voxel_size_mm")
                or not facts.get("spatial_units")):
            raise ValueError("Input structural facts are incomplete")
        checked_file(item["path"])
    return plan


def _manifest_directory(value):
    path = Path(value).expanduser().absolute()
    return path.parent if path.name == "manifest.json" else path


def staged_catalog(spec, runtime):
    uid = _unit_id(spec)
    if not isinstance(uid, str) or not uid:
        raise ValueError("spec must identify analysis_unit_id or subject_id")
    stages = {}
    for item in spec.get("upstream", []):
        directory = _manifest_directory(item)
        manifest = read_json(directory / "manifest.json")
        if manifest.get("schema") != "fastprep.stage/1" or manifest.get("subject") != uid:
            raise ValueError("Upstream manifest has an incompatible stage schema or subject")
        identity = manifest["node_id"]
        if identity in stages:
            raise ValueError("Duplicate upstream stage: " + identity)
        outputs = {}
        for port, record in manifest["outputs"].items():
            members = record.get("members") or [record]
            checked = []
            for member in members:
                path = artifact_path(directory, member)
                checked.append({**member, "path": str(path), "_relative_path": member["path"]})
            outputs[port] = {"contract": record["contract"], "files": checked,
                             "record_file": {**record, "path": str(artifact_path(directory, record)),
                                             "_relative_path": record["path"]},
                             "record_path": str(artifact_path(directory, record))}
        stages[identity] = {"actual_node": identity, "outputs": outputs,
                            "metrics": manifest.get("metrics", {}),
                            "commands": manifest.get("commands", []),
                            "manifest": str(directory / "manifest.json"),
                            "root": str(directory)}
    return {"subject": uid, "stages": stages, "resources": runtime.get("resources", [])}


def stage_files(catalog, identity, port):
    try:
        return catalog["stages"][identity]["outputs"][port]["files"]
    except KeyError as error:
        raise ValueError("Missing declared QC artifact " + identity + "/" + port) from error


def first(catalog, identity, port, hemi=None):
    record = catalog["stages"][identity]["outputs"][port]
    if hemi is None and record["contract"].get("data_type") == "manifest":
        return checked_file(record["record_path"])
    members = [item for item in stage_files(catalog, identity, port)
               if hemi is None or item.get("hemi") == hemi]
    if len(members) != 1:
        raise ValueError("Missing or ambiguous artifact " + identity + "/" + port)
    return checked_file(members[0]["path"])


def _path(value, base=PROJECT):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _command_prefix(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and value and all(isinstance(token, str) for token in value):
        return value
    raise ValueError("Command argv_prefix must be a nonempty string list")


def compact_labels(data, present):
    """Map sorted finite integer labels to consecutive uint8 codes; zero stays background."""
    import numpy as np

    foreground = present != 0
    count = int(np.count_nonzero(foreground))
    if count > np.iinfo(np.uint8).max:
        raise ValueError("QC grouping cannot represent more than 255 foreground labels")
    codes = np.zeros(len(present), dtype=np.uint8)
    codes[foreground] = np.arange(1, count + 1, dtype=np.intp)
    if present.size and present[0] >= 0 and present[-1] <= 65535:
        table = np.zeros(int(present[-1]) + 1, dtype=np.uint8)
        table[present.astype(np.intp)] = codes
        return table[data if data.dtype.kind in 'iu' else data.astype(np.intp)]
    return codes[np.searchsorted(present, data)]


class Evidence:
    def __init__(self, catalog, node, runtime, output, stage=None):
        self.catalog, self.node, self.runtime = catalog, node, runtime
        self.identity = stage or node.get("checks_node", node["node_id"])
        self.stage = catalog["stages"].get(self.identity)
        if self.stage is None:
            raise ValueError("QC references an absent stage: " + self.identity)
        self.checked_nodes = [stage] if stage else node.get("policy", {}).get("checked_nodes", [self.identity])
        self.directory = stage or node["node_id"].removesuffix("_qc")
        self.root = Path(output) / self.directory
        self.root.mkdir(parents=True, exist_ok=True)
        self.qc = runtime.get("qc", {})
        self.package_root = _path(runtime.get("package_root", PROJECT))
        self.vendor_root = _path(self.qc.get("vendor_root", self.package_root / "vendor"))
        self.commands, self.visuals, self.missing, self.failures = [], [], [], set()
        self.measurements = {"artifacts": [], "surface": [], "segmentation": [],
                             "source_stage_metrics": self.stage["metrics"]}
        self.started_at = utc_now()
        policy = node.get("policy", {})
        self.rules = policy.get("stage_rules", {}).get(stage, policy.get("rule_ids", [])) if stage else policy.get("rule_ids", [])
        self.env = {**os.environ, "PYTHONPATH": str(self.vendor_root), "MPLBACKEND": "Agg",
                    "QT_QPA_PLATFORM": "offscreen", "OMP_NUM_THREADS": "1"}
        for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS',
                         'VTK_SMP_MAX_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.env[variable] = '1'
        fs_home = self.qc.get("fs_home")
        if fs_home:
            self.env["FREESURFER_HOME"] = str(_path(fs_home, self.package_root))
        self.visual_contract = self._visual_scope()
        self._source_anatomy = {}

    def _visual_scope(self):
        contract = self.node.get("policy", {}).get("visual_evidence_contract")
        if contract is None:
            return None
        if contract.get("version") != "selected-qc-views/1" or self.identity not in contract.get("stages", {}):
            raise ValueError("Unsupported or incomplete selected QC visual contract")
        scope = dict(contract["stages"][self.identity])
        refinement = self.stage["metrics"].get("pial_refinement", {})
        if refinement.get("mode") == "T2_refined":
            scope["png_files"] = [*scope.get("png_files", []), *scope.get("t2_png_files", [])]
        return scope

    def _render_enabled(self, relative):
        return self.visual_contract is None or relative in self.visual_contract.get("png_files", [])

    def tool(self, relative, argv, *, pair=False, fs=False, source_anatomy=False, label_definition=None):
        if relative.endswith(".png") and not self._render_enabled(relative):
            return None
        surface_engine = self.qc.get('surface_engine', 'reference')
        if surface_engine != 'reference' and (argv[:2] == ['render', 'surface'] or argv[0] == 'spherical-features'):
            argv = [*argv, '--surface-engine', surface_engine]
        output = self.root / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        if fs:
            commands = self.runtime.get("commands", {})
            matches = [(name, record) for name, record in commands.items()
                       if name.endswith("\u003a" + argv[0]) or name == argv[0]]
            if len(matches) != 1:
                self.missing.append("Configured FreeSurfer command unavailable or ambiguous: " + argv[0])
                return None
            command_id, config = matches[0]
            command = [*_command_prefix(config["argv_prefix"]), *map(str, argv[1:])]
            env = {**self.env, **config.get("environment", {})}
        elif pair:
            executable = str(_path(self.qc.get("pair_python", self.qc.get("helper_python", sys.executable)), self.package_root))
            command = [executable, "-B", str(HERE / "qc_surface_pair.py"), *map(str, argv[1:]),
                       "--geometry-python", str(_path(self.qc.get("render_python", sys.executable), self.package_root)),
                       "--output", str(output)]
            env = self.env
            command_id = "qc_surface_pair"
        elif argv[0] == "spherical-features":
            executable = str(_path(self.qc.get("render_python", sys.executable), self.package_root))
            command = [executable, "-B", str(HERE / "qc_spherical.py"), *map(str, argv[1:]),
                       "--output", str(output)]
            env, command_id = self.env, "qc_spherical"
        else:
            executable = str(_path(self.qc.get("render_python", self.qc.get("helper_python", sys.executable)), self.package_root))
            command = [executable, "-B", "-m", "neuroqc", *map(str, argv), "--output", str(output)]
            env = self.env
            command_id = "neuroqc"
        index = len(self.commands)
        logs = self.root / "logs"
        logs.mkdir(exist_ok=True)
        stdout, stderr = logs / f"{index}.stdout", logs / f"{index}.stderr"
        started = time.monotonic()
        with stdout.open("w", encoding="utf-8") as out_stream, stderr.open("w", encoding="utf-8") as err_stream:
            result = subprocess.run(command, stdout=out_stream, stderr=err_stream, env=env, check=False)
        self.commands.append({"command_id": command_id, "argv": command, "exit_code": result.returncode,
                              "stdout": str(stdout.relative_to(self.root)), "stderr": str(stderr.relative_to(self.root))})
        if self.qc.get('volume_engine', 'reference') != 'reference' or surface_engine != 'reference':
            self.commands[-1]['elapsed_seconds'] = time.monotonic() - started
        if result.returncode:
            self.missing.append("Tool failed: " + command_id + "; see " + str(stderr.relative_to(self.root)))
            return None
        if fs:
            return stdout.read_text(encoding="utf-8", errors="replace") + "\n" + stderr.read_text(encoding="utf-8", errors="replace")
        if output.suffix == ".png":
            return self.register_visual(output, source_anatomy, label_definition)
        return read_json(output)

    def register_visual(self, output, source_anatomy, label_definition):
        checked_file(output)
        record = read_json(str(output) + ".json")
        if label_definition is not None:
            record.setdefault("parameters", {})["label_definition"] = label_definition
            write_json(str(output) + ".json", record)
        self.visuals.append({"view_type": "operation-specific", "source": str(output.relative_to(self.root)),
                             "observation": "Generated evidence; visual review pending.",
                             "layout": {**record.get("parameters", {}),
                                        **({"source_identity": "source_anatomy", "view_kind": "image"}
                                           if source_anatomy else {})},
                             "slice_selection": record.get("slice_selection", {})})
        return record

    def label_views(self, port, reference, grouped, definitions):
        key = str(Path(reference).resolve())
        source = port + '-source-axial.png'
        views = {}
        args = ['render', 'label-batch', '--image', reference, '--labels', grouped]
        if key not in self._source_anatomy and self._render_enabled(source):
            path = self.root / source
            args += ['--source-output', path]
            views[str(path)] = None
        for code, definition in definitions.items():
            relative = port + '-label-' + code + '-contours.png'
            if self._render_enabled(relative):
                path = self.root / relative
                args += ['--contour', code, path]
                views[str(path)] = definition
        if not views:
            return
        inventory = self.tool(port + '-batch.json', args)
        if inventory is None:
            return
        completed, failures = inventory['completed'], inventory['failed']
        if sorted([*completed, *failures]) != sorted(views):
            raise ValueError('Incomplete or unexpected label-batch outputs')
        for path, definition in views.items():
            if path in failures:
                self.missing.append('Render failed: ' + Path(path).name + ': ' + failures[path])
                continue
            self.register_visual(Path(path), definition is None, definition)
            if definition is None:
                self._source_anatomy[key] = source

    def source_image(self):
        if self.identity.startswith("resample_") or self.identity == "validate_fsaverage6_sphere_correspondence":
            matches = [item["path"] for item in self.runtime.get("resources", []) if item["id"] == "fsaverage6-orig"]
            if len(matches) != 1:
                raise ValueError("Missing fsaverage6 anatomical reference")
            return checked_file(matches[0])
        uid = self.catalog["subject"]
        plan = _session_plan(self.runtime, uid)
        if self.identity == "anat_get_t1w_file_in_bids":
            return checked_file(plan["reference"]["path"])
        for stage_id in ("anat_motioncor", "anat_create_subject_orig_dir"):
            if "orig_mgz" in self.catalog["stages"].get(stage_id, {}).get("outputs", {}):
                return first(self.catalog, stage_id, "orig_mgz")
        return checked_file(plan["reference"]["path"])

    def source_anatomy(self, relative, reference):
        key = str(Path(reference).resolve())
        if key not in self._source_anatomy:
            record = self.tool(relative, ["render", "image", "--image", reference, "--plane", "axial"],
                               source_anatomy=True)
            if record is not None:
                self._source_anatomy[key] = relative
        return self._source_anatomy.get(key)

    def _visual_review(self, rule):
        return [{**item, **({"modality_assessment": "indeterminate", "image_quality_assessment": "pending"}
                            if rule == "INPUT.MODALITY" else {}),
                 **({"false_negative_review": "pending", "false_positive_review": "pending",
                     "coverage_assessment": "indeterminate"} if rule.startswith("SEG.") else {})}
                for item in self.visuals]

    def volume(self, port, member):
        import nibabel as nib
        import numpy as np

        path = checked_file(member["path"])
        image = nib.load(path)
        data = np.asanyarray(image.dataobj)
        finite = bool(np.isfinite(data).all())
        facts = {"port": port, "hemi": member.get("hemi"), "path": str(path),
                 **image_geometry(image), "orientation": list(nib.aff2axcodes(image.affine)),
                 "finite": finite, "intensity_percentiles": np.percentile(data[np.isfinite(data)], [1, 5, 50, 95, 99]).tolist()
                 if np.isfinite(data).any() else []}
        self.measurements["artifacts"].append(facts)
        if data.ndim != 3 or not finite:
            self.failures.add("OUTPUT.CONSISTENCY")
            return
        if port in LABEL_PORTS:
            if not np.array_equal(data, np.rint(data)):
                self.failures.add("SEG.TISSUE")
                return
            reference = self.source_image()
            reference_image = nib.load(reference)
            if image.shape != reference_image.shape or not np.allclose(image.affine, reference_image.affine, atol=1e-4, rtol=0):
                self.failures.update({"SEG.TISSUE", "OUTPUT.CONSISTENCY"})
                self.missing.append("Image-label grids differ; no implicit resampling: " + port)
                return
            if port == 'filled_pretess_mgz':
                hemi = member.get('hemi')
                if hemi not in ('lh', 'rh'):
                    raise ValueError('Pretess initializer requires a bound hemisphere')
                label = 255 if hemi == 'lh' else 127
                facts['tessellation_foreground_label'] = label
                # The volume retains both labels; only this hemisphere's label
                # is consumed by tessellation and is the initializer under review.
                data = np.where(data == label, data, 0)
                port += '-' + hemi
            present = np.unique(data)
            labels = [int(value) for value in present]
            groups = {"labels": [v for v in labels if v != 0]}
            fast = self.qc.get('volume_engine', 'reference') != 'reference'
            encoded = compact_labels(data, present) if fast else np.zeros(data.shape, dtype=np.uint8)
            definitions = {}
            for index, label in enumerate(groups["labels"], 1):
                if not fast:
                    encoded[data == label] = index
                definitions[str(index)] = {"name": "label_" + str(label), "source_labels": [label]}
            if definitions:
                grouped = self.root / (port + "-qc-groups.nii.gz")
                nib.save(nib.Nifti1Image(encoded, image.affine), grouped)
                self.measurements["segmentation"].append({"port": port, "labels_present": labels,
                                                            "group_definitions": definitions,
                    "result": self.tool(port + "-segmentation.json", ["metrics", "segmentation", "--image", reference,
                        "--labels", grouped, "--label", *range(1, len(definitions) + 1)])})
                if self.qc.get('volume_engine') == 'batched':
                    self.label_views(port, reference, grouped, definitions)
                    return
                self.source_anatomy(port + "-source-axial.png", reference)
                for index, definition in enumerate(definitions.values(), 1):
                    self.tool(port + "-label-" + str(index) + "-contours.png", ["render", "contours", "--image", reference,
                        "--labels", grouped, "--label", index, "--plane", "axial", "--color", "#ff3030"],
                        label_definition=definition)
            return
        suffix = "-" + member["hemi"] if member.get("hemi") else ""
        self.tool(port + suffix + "-orthogonal.png", ["render", "orthogonal", "--image", path,
                  "--declared-modality", "derived:" + port])

    def mesh(self, port, member):
        import nibabel as nib
        import numpy as np
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        path, hemi = checked_file(member["path"]), member.get("hemi", "unknown")
        stem = hemi + "-" + port
        measured = self.tool(stem + "-mesh.json", ["metrics", "surface", "--surface", path])
        topology = self.tool(stem + "-euler.txt", ["mris_euler_number", path], fs=True)
        intersection = self.tool(stem + "-intersections.txt", ["mris_info", "--intersections", path], fs=True)
        counts = re.findall(r"^Found (\d+) intersections\s*$", intersection or "", re.M)
        vertices, faces = nib.freesurfer.read_geometry(path)
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        adjacency = coo_matrix((np.ones(len(edges), dtype=np.uint8),
                               (edges[:, 0], edges[:, 1])), shape=(len(vertices), len(vertices))).tocsr()
        components = int(connected_components(adjacency, directed=False, return_labels=False))
        metrics = (measured or {}).get("metrics", {})
        metrics["connectedComponents"] = components
        fixed = port not in {"orig_nofix_surf", "smoothwm_nofix_surf", "inflated_nofix_surf", "qsphere_nofix_surf", "smoothwm_surf", "inflated_surf"}
        if fixed and (components != 1 or metrics.get("boundaryEdgeCount") != 0
                      or metrics.get("nonmanifoldEdgeCount") != 0 or metrics.get("degenerateFaceCount") != 0):
            self.failures.add("SURFACE.MESH")
        if len(counts) != 1:
            self.missing.append("Independent self-intersection measurement unavailable: " + stem)
        elif int(counts[0]) and fixed and port != "orig_topofix_surf":
            self.failures.add("SURFACE.MESH")
        elif int(counts[0]):
            self.missing.append("Intermediate surface intersections require review: " + stem)
        self.measurements["surface"].append({"port": port, "hemi": hemi, "result": measured,
            "self_intersecting_vertices": int(counts[0]) if len(counts) == 1 else None,
            "self_intersection_gate": "current_stage",
            "self_intersection_status": "measured",
            "euler_tool_output": topology, "connected_components": components})
        self.tool(stem + "-six-views.png", ["render", "surface", "--" + hemi, path, "--title", stem])
        if port == "sphere_reg_surf":
            self.spherical_features(member)

    def spherical_features(self, member):
        hemi = member["hemi"]
        template_root = Path(self.qc["fs_home"]) / "subjects/fsaverage6/surf"
        template_sphere = checked_file(template_root / (hemi + ".sphere"))
        template_sulc = checked_file(template_root / (hemi + ".sulc"))
        expected = {"--template-sphere": str(template_sphere), "--template-sulc": str(template_sulc)}
        commands = [entry["argv"] for entry in self.stage["commands"] if "--hemisphere" in entry["argv"]
                    and entry["argv"][entry["argv"].index("--hemisphere") + 1] == hemi]
        if not any(all(flag in argv and argv[argv.index(flag) + 1] == value for flag, value in expected.items())
                   for argv in commands):
            self.missing.append("Spherical visualization reference differs from registration command inputs: " + hemi)
            return
        for feature, identity, port in (("sulc", "anat_smooth_inflated", "sulc_surf"),
                                        ("curv", "anat_white_surface", "curv_surf")):
            self.tool(hemi + "-sphere-registration-" + feature + ".png", ["spherical-features", "--native",
                first(self.catalog, "anat_sphere", "sphere_surf", hemi), "--registered", member["path"],
                "--template", template_sphere, "--feature", first(self.catalog, identity, port, hemi),
                "--template-feature", checked_file(template_root / (hemi + "." + feature)), "--hemi", hemi])

    def scalar(self, port, member):
        import nibabel as nib
        import numpy as np

        path, hemi = checked_file(member["path"]), member.get("hemi")
        if hemi is None:
            return
        scalar_ports = {"curv_surf", "area_surf", "curv_pial_surf", "area_pial_surf",
                        "thickness_surf", "sulc_surf", "jacobian_white", "aparc_annot", "cortex_label",
                        "cortex_hipamyg_label"}
        if port not in scalar_ports and path.suffix not in {".annot", ".label", ".mgh"}:
            if path.suffix in {".stats", ".dat"}:
                self.measurements.setdefault("native_statistics", []).append({"port": port, "hemi": hemi,
                    "path": str(path), "text": path.read_text(encoding="utf-8", errors="replace")})
            return
        if path.suffix == ".annot":
            values, _, names = nib.freesurfer.read_annot(str(path))
            names = [name.decode(errors="replace") for name in names]
        elif path.suffix == ".label":
            indices = nib.freesurfer.read_label(str(path))
            values = indices
            names = None
        elif path.suffix == ".mgh":
            values = np.asanyarray(nib.load(str(path)).dataobj).reshape(-1)
            names = None
        else:
            try:
                values = nib.freesurfer.read_morph_data(str(path))
            except (ValueError, OSError):
                return
            names = None
        mesh_candidates = [(identity, name, record) for identity, stage in self.catalog["stages"].items()
                           for name, record in stage["outputs"].items()
                           if name in {"white_surf", "white_preaparc_surf", "orig_surf"}
                           and any(item.get("hemi") == hemi for item in record["files"])]
        preferred = [(identity, name, record) for identity, name, record in mesh_candidates
                     if identity == "anat_white_surface" and name == "white_surf"]
        if preferred:
            mesh_candidates = preferred
        if len(mesh_candidates) != 1:
            self.missing.append("Scalar mesh reference unavailable or ambiguous: " + hemi + "/" + port)
            return
        identity, mesh_port, _ = mesh_candidates[0]
        mesh = first(self.catalog, identity, mesh_port, hemi)
        vertices, _ = nib.freesurfer.read_geometry(str(mesh))
        if path.suffix == ".label":
            valid_indices = bool(len(values) == 0 or (values.min() >= 0 and values.max() < len(vertices)))
            finite = valid_indices
            count = len(values)
            percentiles = None
        else:
            finite = bool(np.isfinite(values).all())
            count = len(values)
            percentiles = np.percentile(values, [1, 5, 50, 95, 99]).tolist() if finite and count else None
            valid_indices = True
        correspondence = count == len(vertices) if path.suffix != ".label" else valid_indices
        result = {"port": port, "hemi": hemi, "path": str(path), "mesh": str(mesh),
                  "value_count": count, "mesh_vertex_count": len(vertices), "finite": finite,
                  "vertex_correspondence": correspondence, "percentiles": percentiles,
                  "label_names": names}
        self.measurements.setdefault("scalar", []).append(result)
        if not finite or not correspondence:
            self.failures.add("OUTPUT.CONSISTENCY")

    def bias_metrics(self):
        import nibabel as nib
        import numpy as np

        before_node = next((name for name in ("anat_motioncor", "anat_create_subject_orig_dir")
                            if name in self.catalog["stages"]), None)
        label_node = next((name for name in ("anat_synthseg", "anat_segmentation_qc")
                           if name in self.catalog["stages"] and "aseg_synthseg_mgz" in self.catalog["stages"][name]["outputs"]), None)
        after_port = next((name for name in ("orig_nu_mgz", "nu_mgz") if name in self.stage["outputs"]), None)
        if not before_node or not label_node or not after_port:
            self.missing.append("Bias field comparison inputs unavailable")
            return
        before_path = first(self.catalog, before_node, "orig_mgz")
        after_path = first(self.catalog, self.identity, after_port)
        labels_path = first(self.catalog, label_node, "aseg_synthseg_mgz")
        before, after, labels = map(nib.load, (before_path, after_path, labels_path))
        comparable = (before.shape == after.shape == labels.shape
                      and np.allclose(before.affine, after.affine, atol=1e-4, rtol=0)
                      and np.allclose(before.affine, labels.affine, atol=1e-4, rtol=0))
        facts = {"before": str(before_path), "after": str(after_path), "labels": str(labels_path),
                 "same_grid": bool(comparable)}
        if comparable:
            mask = np.isin(np.asanyarray(labels.dataobj), [2, 41])
            for name, image in (("before", before), ("after", after)):
                samples = np.asanyarray(image.dataobj)[mask]
                mean = float(samples.mean()) if len(samples) else 0.0
                facts[name + "_white_matter_cv"] = float(samples.std() / abs(mean)) if mean else None
        else:
            self.failures.add("PREPROC.BIAS_FIELD")
            self.missing.append("Bias evidence physical grids differ")
        if "bias_factor_nii" in self.stage["outputs"]:
            factor_path = first(self.catalog, self.identity, "bias_factor_nii")
            factor = nib.load(factor_path)
            values = np.asanyarray(factor.dataobj)
            valid = (factor.shape == before.shape and np.allclose(factor.affine, before.affine, atol=1e-4, rtol=0)
                     and np.isfinite(values).all() and np.all(values > 0))
            facts["correction_factor"] = {"path": str(factor_path), "finite_positive_same_grid": bool(valid)}
            if not valid:
                self.failures.add("PREPROC.BIAS_FIELD")
        self.measurements["bias_field"] = facts
        if comparable:
            self.tool("bias-before-after-axial.png", ["render", "comparison", "--before", before_path,
                      "--after", after_path, "--plane", "axial"])

    def pial_refinement_metrics(self):
        import nibabel as nib
        import numpy as np

        record = self.stage["outputs"].get("pial_refinement_manifest")
        if not record:
            self.missing.append("Pial refinement manifest unavailable")
            return
        manifests = [item for item in record["files"] if Path(item["path"]).name == "pial_refinement.json"]
        if len(manifests) != 1:
            self.missing.append("Pial refinement manifest absent or ambiguous")
            return
        metadata = read_json(manifests[0]["path"])
        refinement = self.stage["metrics"].get("pial_refinement", {})
        if refinement.get("mode") and metadata.get("mode") != refinement["mode"]:
            self.failures.add("OUTPUT.CONSISTENCY")
        self.measurements["pial_refinement"] = metadata
        if metadata.get("mode") == "T1_only":
            return
        if metadata.get("mode") != "T2_refined":
            self.missing.append("Unknown pial refinement mode")
            return
        by_name = {Path(item["path"]).name: item["path"] for item in record["files"]}
        reference = self.source_image()
        image = by_name.get("T2.mgz")
        if not image:
            self.missing.append("Refined T2 image missing")
            return
        relation = image_geometry(nib.load(reference))
        target = image_geometry(nib.load(image))
        same_grid = relation["shape"] == target["shape"] and np.allclose(relation["affine_mm"], target["affine_mm"], atol=1e-4, rtol=0)
        self.measurements["pial_refinement"]["t2_reference_grid_match"] = bool(same_grid)
        if not same_grid:
            self.failures.add("OUTPUT.CONSISTENCY")
        elif by_name.get("T2.prenorm.nii.gz") and by_name.get("T2.norm.mgz"):
            self.tool("t2-bias-before-after-axial.png", ["render", "comparison", "--before",
                      by_name["T2.prenorm.nii.gz"], "--after", by_name["T2.norm.mgz"], "--plane", "axial"])
        for hemi in ("lh", "rh"):
            white_stage = next((name for name in self.catalog["stages"]
                                if "white_surf" in self.catalog["stages"][name]["outputs"]), None)
            if not white_stage:
                self.missing.append("Pial white-surface ancestor unavailable")
                break
            initial = first(self.catalog, self.identity, "pial_t1_surf", hemi)
            final = first(self.catalog, self.identity, "pial_surf", hemi)
            a, af = nib.freesurfer.read_geometry(str(initial))
            b, bf = nib.freesurfer.read_geometry(str(final))
            if a.shape != b.shape or not np.array_equal(af, bf):
                self.failures.add("SURFACE.WHITE_PIAL")
            else:
                self.measurements["pial_refinement"][hemi + "_displacement_mm_percentiles"] = np.percentile(
                    np.linalg.norm(b - a, axis=1), [50, 95, 99, 100]).tolist()
        white_stage = next((name for name in self.catalog["stages"]
                            if "white_surf" in self.catalog["stages"][name]["outputs"]), None)
        if same_grid and white_stage:
            self.tool("white-pial-T2-axial.png", ["render", "surface-contours", "--image", image,
                "--inner", first(self.catalog, white_stage, "white_surf", "lh"),
                first(self.catalog, white_stage, "white_surf", "rh"),
                "--outer", first(self.catalog, self.identity, "pial_surf", "lh"),
                first(self.catalog, self.identity, "pial_surf", "rh"), "--surface-space",
                "freesurfer-tkr", "--surface-reference", reference])

    def companion_registration_metrics(self):
        manifest_path = first(self.catalog, self.identity, "alignment_manifest")
        manifest = read_json(manifest_path)
        source_root = Path(manifest_path).parent
        registrations = []
        for record in manifest.get("registrations", []):
            registered = checked_file(source_root / record["registered"])
            baseline = checked_file(source_root / record["baseline"])
            fixed = checked_file(record["fixed"])
            measured = self.tool(record["series_id"] + "-metrics.json", ["metrics", "registration",
                "--fixed", fixed, "--moving", baseline, "--registered", registered,
                "--improvement-min", "0", "--threshold-source",
                "session-companion-registration/1: non-decreasing mutual information"])
            self.tool(record["series_id"] + "-registration.png", ["render", "registration", "--fixed",
                      fixed, "--moving", baseline, "--registered", registered])
            registrations.append({"series_id": record["series_id"], "result": measured})
        self.measurements["companion_registration"] = registrations
        self.measurements["companion_count"] = manifest.get("companion_count")
        if manifest.get("registered_count") != manifest.get("companion_count"):
            self.failures.add("OUTPUT.CONSISTENCY")

    def pair(self):
        import nibabel as nib
        import numpy as np

        target = self.identity == "resample_pial_surface_fsaverage6"
        white_identity = "resample_white_surface_fsaverage6" if target else "anat_white_surface"
        white_stages = [(identity, stage) for identity, stage in self.catalog["stages"].items()
                        if "white_surf" in stage["outputs"]]
        preferred = [item for item in white_stages if item[0] == white_identity]
        if preferred:
            white_stages = preferred
        contour_white, contour_pial = [], []
        for hemi in ("lh", "rh"):
            try:
                if len(white_stages) != 1:
                    raise ValueError("White surface ancestor is absent or ambiguous")
                white = first(self.catalog, white_stages[0][0], "white_surf", hemi)
                pial = first(self.catalog, self.identity, "pial_surf", hemi)
            except ValueError as error:
                self.missing.append(str(error))
                continue
            white_xyz, white_faces = nib.freesurfer.read_geometry(str(white))
            pial_xyz, pial_faces = nib.freesurfer.read_geometry(str(pial))
            correspondence = white_xyz.shape == pial_xyz.shape and np.array_equal(white_faces, pial_faces)
            if not correspondence:
                self.failures.add("SURFACE.WHITE_PIAL")
            def artifact(port, selected_hemi=None, preferred_identity=None):
                found = [(identity, item) for identity, stage in self.catalog["stages"].items()
                         for output_port, record in stage["outputs"].items() if output_port == port
                         for item in record["files"]
                         if selected_hemi is None or item.get("hemi") == selected_hemi]
                preferred = [item for identity, item in found if identity == preferred_identity]
                if preferred:
                    found = [(preferred_identity, item) for item in preferred]
                return found[0][1]["path"] if len(found) == 1 else None

            thickness = artifact("thickness_surf", hemi, self.identity)
            cortex = artifact("cortex_label", hemi, "anat_cortex_label")
            ribbon = artifact("lh_ribbon_mgz" if hemi == "lh" else "rh_ribbon_mgz", preferred_identity="anat_ribbon")
            if not ribbon:
                ribbon = artifact("ribbon_mgz", preferred_identity="anat_ribbon")
            aseg = artifact("aseg_presurf_mgz", preferred_identity="anat_synthseg_fill")
            if not aseg:
                aseg = artifact("aseg_mgz")
            pair_args = ["surface-pair", "--inner", white, "--outer", pial, "--coordinate-unit", "mm"]
            if not target and all((thickness, cortex, ribbon, aseg)):
                pair_args.extend(["--thickness", thickness, "--cortex-label", cortex,
                                  "--ribbon", ribbon, "--aseg", aseg, "--hemi", hemi])
            elif not target:
                self.missing.append(hemi + ": full cortical pair evidence lacks thickness, cortex label, ribbon or aseg")
            ribbon_stage = self.catalog["stages"].get("anat_cortical_pair", self.catalog["stages"].get("anat_ribbon", {}))
            result = None if target else ribbon_stage.get("metrics", {}).get(hemi)
            if result is None:
                result = self.tool(hemi + "-white-pial.json", pair_args, pair=True)
            pair_record = {"hemi": hemi, "vertex_correspondence": correspondence, "result": result,
                "nearest_distance_is_not_thickness": True,
                "corresponding_vertex_distance_mm_percentiles": np.percentile(
                    np.linalg.norm(pial_xyz - white_xyz, axis=1), [1, 5, 50, 95, 99]).tolist() if correspondence else None}
            thickness = next((identity for identity, stage in self.catalog["stages"].items()
                              if "thickness_surf" in stage["outputs"]), None)
            if thickness and not target:
                values = nib.freesurfer.read_morph_data(str(first(self.catalog, thickness, "thickness_surf", hemi)))
                pair_record["thickness_percentiles_mm"] = np.percentile(values, [1, 5, 50, 95, 99]).tolist()
                pair_record["negative_thickness_vertices"] = int((values < 0).sum())
                if not np.isfinite(values).all() or (values < 0).any():
                    self.failures.add("SURFACE.WHITE_PIAL")
            self.measurements.setdefault("surface_pair", []).append(pair_record)
            contour_white.append(white)
            contour_pial.append(pial)
            if result and result.get("status") == "fail" and target:
                self.missing.append(hemi + ": whole-mesh contacts require localization; pinned medial-wall contact is not automatically cortical penetration")
            if result and (result.get("white_outside_pial_vertices", 0)
                           or result.get("cortical_triangle_pair_intersections", 0)
                           or result.get("negative_thickness_vertices", 0)):
                self.failures.add("SURFACE.WHITE_PIAL")
            if result:
                self.missing.extend(hemi + ": " + item for item in result.get("missing_checks", []))
        if len(contour_white) == 2:
            image = self.source_image()
            self.tool("white-pial-axial.png", ["render", "surface-contours", "--image", image,
                "--inner", *contour_white, "--outer", *contour_pial, "--surface-space",
                "freesurfer-tkr", "--surface-reference", image])

    def _visual_scope_check(self):
        if self.visual_contract is None:
            return
        expected = set(self.visual_contract.get("png_files", []))
        actual = {item["source"] for item in self.visuals}
        if actual - expected:
            self.failures.add("REPORT.TRACEABILITY")
            self.missing.append("Generated undeclared QC images: " + ", ".join(sorted(actual - expected)))
        if expected - actual:
            self.missing.append("Required QC images could not be generated: " + ", ".join(sorted(expected - actual)))

    def collect_measurements(self, input_plan):
        import nibabel as nib
        import numpy as np

        if self.identity == "anat_get_t1w_file_in_bids":
            reference = input_plan["reference"]
            image_path = checked_file(reference["path"])
            image = nib.load(image_path)
            values = np.asanyarray(image.dataobj)
            facts = reference["probe"]["facts"]
            self.measurements["input"] = {"declared_modality": reference["modality"],
                "structural_status": reference["structural_status"], "visual_review": "pending",
                "shape": list(values.shape), "finite": bool(np.isfinite(values).all()),
                "nonzero": bool(np.any(values != 0)), "voxel_size_mm": facts["voxel_size_mm"],
                "spatial_units": facts["spatial_units"], "affine": image.affine.tolist(),
                "orientation": list(nib.aff2axcodes(image.affine))}
            if values.ndim != 3 or not np.isfinite(values).all() or not np.any(values != 0):
                self.failures.update({"INPUT.INTEGRITY", "OUTPUT.CONSISTENCY"})
            self.tool("input-orthogonal.png", ["render", "orthogonal", "--image", image_path,
                      "--declared-modality", reference["modality"]])
            self.missing.append("Raw input visual quality and modality review are pending; structural checks are not quality approval.")

        if self.identity == "session_align_companions":
            self.companion_registration_metrics()
        if "bias_factor_nii" in self.stage["outputs"] or "orig_nu_mgz" in self.stage["outputs"]:
            self.bias_metrics()
        if "pial_refinement_manifest" in self.stage["outputs"]:
            self.pial_refinement_metrics()

        for port, record in self.stage["outputs"].items():
            contract_type = record["contract"].get("data_type")
            for member in record["files"]:
                path = checked_file(member["path"])
                if port in MESH_PORTS:
                    self.mesh(port, member)
                elif contract_type == "volume" or port in LABEL_PORTS:
                    self.volume(port, member)
                elif member.get("hemi"):
                    self.scalar(port, member)
                else:
                    self.measurements["artifacts"].append({"port": port, "hemi": member.get("hemi"),
                        "path": str(path), "size_bytes": path.stat().st_size,
                        "data_type": contract_type})
        if "SURFACE.WHITE_PIAL" in self.rules:
            self.pair()
        if self.identity == 'anat_synthseg' and 'anat_mask_to_mgz' in self.catalog['stages']:
            labels = nib.load(first(self.catalog, self.identity, 'aseg_synthseg_mgz'))
            mask = nib.load(first(self.catalog, 'anat_mask_to_mgz', 'mask_mgz'))
            if labels.shape == mask.shape and np.allclose(labels.affine, mask.affine, atol=1e-4, rtol=0):
                mask_values = np.asanyarray(mask.dataobj) != 0
                count = int(mask_values.sum())
                covered = int(np.count_nonzero(mask_values & (np.asanyarray(labels.dataobj) != 0)))
                self.measurements['synthseg_hd_bet_overlap'] = {
                    'brain_mask_voxels': count, 'overlap_voxels': covered,
                    'fraction_of_mask_with_nonzero_segmentation': covered / count if count else None}
            else:
                self.missing.append('SynthSeg and HD-BET masks do not share a grid for overlap measurement')

    def generate(self):
        # Validate identity and acquisition contracts outside optional evidence collection.
        input_plan = (_session_plan(self.runtime, self.catalog["subject"])
                      if self.identity == "anat_get_t1w_file_in_bids" else None)
        try:
            self.collect_measurements(input_plan)
        except (ValueError, ArithmeticError, RuntimeError) as error:
            # The catalog has already checked session identity and resolved required artifacts.
            incomplete = f'{type(error).__name__}: {error}'
            self.missing.append('QC measurement incomplete: ' + incomplete)
            self.measurements['measurement_error'] = incomplete
        warnings = self.stage['metrics'].get('quality_warnings', {})
        if warnings:
            self.missing.extend(f'{location}/{name} requires review'
                                for location, checks in warnings.items() for name in checks)
        self._visual_scope_check()
        write_json(self.root / "measurements.json", {**self.measurements, "missing_checks": self.missing})
        self._quantitative_table()
        rules = self.vendor_root / "neuroqc" / "references"
        rule_index = read_json(rules / "qc-rules.json")
        checks = []
        for identity in self.rules:
            rule = read_json(rules / rule_index["rule_files"][identity])
            if identity in self.failures:
                status = "fail"
            elif identity == "OUTPUT.CONSISTENCY" and not self.missing:
                status = "pass"
            else:
                status = "unknown"
            checks.append({"rule_id": identity, "status": status, "severity": rule["severity"],
                "metrics": [{"source": "measurements.tsv", "definition": "Measured facts; acceptance thresholds are not inferred"}],
                "visual_evidence": self._visual_review(identity),
                "evidence": ["measurements.json", "measurements.tsv", *[item["source"] for item in self.visuals]],
                "message": "Generated evidence does not replace protocol or visual review."})
        operation = ("cortical_reconstruction" if any(item.startswith("SURFACE.") for item in self.rules)
                     else "segmentation" if any(item.startswith("SEG.") for item in self.rules)
                     else "structural_preprocessing")
        policy = self.node.get("policy", {})
        inputs = [{"role": self.identity + ":" + port, "semantic": record["contract"], "source": member["path"]}
                  for port, record in self.stage["outputs"].items() for member in record["files"]]
        status = overall_status(checks)
        if status == 'pass' and (warnings or 'measurement_error' in self.measurements):
            status = 'unknown'
        report = {"report_id": self.catalog["subject"] + ":" + self.node["node_id"], "operation": operation,
            "inputs": inputs, "checks": checks, "overall_status": status,
            "quality_warnings": warnings, "blockers": [item["rule_id"] for item in checks if item["status"] != "pass" and item["severity"] == "error"],
            "human_review": [item["rule_id"] for item in checks if item["status"] != "pass"] + self.missing,
            "provenance": {"requested_node": self.directory, "actual_source_node": self.identity,
                "source_manifest": self.stage["manifest"], "checked_nodes": self.checked_nodes,
                "source_commands": self.stage["commands"], "measurement_table": "measurements.tsv",
                "qc_commands": self.commands, "started_at": self.started_at,
                "completed_at": utc_now(), "scientific_approval": "not_performed"}}
        write_json(self.root / "report.json", report)
        self._write_evidence_index()
        return report

    def _quantitative_table(self):
        import csv

        path = self.root / "measurements.tsv"
        rows = []
        for group in ("artifacts", "surface", "segmentation"):
            for index, record in enumerate(self.measurements.get(group, [])):
                rows.append({"category": group, "index": index, "record": json.dumps(record, ensure_ascii=False,
                                                                                       allow_nan=False)})
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("category", "index", "record"), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    def _write_evidence_index(self):
        write_json(self.root / "evidence.json", {"node_id": self.node["node_id"],
            "subject": self.catalog["subject"], "files": [
                {"path": str(path.relative_to(self.root)), "size_bytes": path.stat().st_size}
                for path in sorted(self.root.rglob("*")) if path.is_file() and path.name != "evidence.json"]})


def image_geometry(image):
    unit = image.header.get_xyzt_units()[0] if hasattr(image.header, "get_xyzt_units") else "mm"
    scale = {"mm": 1, "unknown": 1, "meter": 1000, "micron": .001}.get(unit, 1)
    affine = image.affine.copy()
    affine[:3] *= scale
    return {"shape": list(map(int, image.shape)),
            "voxel_size_mm": [float(value * scale) for value in image.header.get_zooms()[:3]],
            "affine_mm": affine.tolist(), "spatial_units": unit}


def generate_group(catalog, node, runtime, output):
    from concurrent.futures import ThreadPoolExecutor

    started_at = utc_now()
    policy = node.get("policy", {})
    checked_nodes = policy.get("checked_nodes") or [node.get("checks_node", node["node_id"])]
    checks, inputs, reviews, blockers, stage_rows, commands = [], [], [], [], [], []
    root = Path(output) / node["node_id"].removesuffix("_qc")
    root.mkdir(parents=True, exist_ok=True)
    threads = runtime.get('qc', {}).get('threads', 1)
    if type(threads) is not int or threads < 1:
        raise ValueError('QC threads must be a positive integer')
    if len(set(checked_nodes)) != len(checked_nodes):
        raise ValueError('Duplicate checked stage in QC group')
    workers = min(threads, len(checked_nodes))
    started = time.monotonic()

    def generate_stage(identity):
        stage_start = time.monotonic()
        evidence = Evidence(catalog, node, runtime, root / "stages", stage=identity)
        report = evidence.generate()
        return identity, evidence, report, time.monotonic() - stage_start

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            generated = list(executor.map(generate_stage, checked_nodes))
    else:
        generated = [generate_stage(identity) for identity in checked_nodes]
    execution = {'allocated_threads': threads, 'workers': workers,
                 'stage_seconds': {identity: seconds for identity, _, _, seconds in generated},
                 'wall_seconds': time.monotonic() - started}
    quality_warnings = {}
    for identity, evidence, report, _ in generated:
        if report.get('quality_warnings'):
            quality_warnings[identity] = report['quality_warnings']
        prefix = str(evidence.root.relative_to(root)) + "/"
        for original in report["checks"]:
            check = copy.deepcopy(original)
            check["message"] = identity + ": " + check["message"]
            check["evidence"] = [prefix + path for path in check["evidence"]]
            for metric in check["metrics"]:
                metric["source"] = prefix + metric["source"]
                metric["stage_node"] = identity
            for visual in check["visual_evidence"]:
                visual["source"] = prefix + visual["source"]
            checks.append(check)
        inputs.extend(report["inputs"])
        reviews.extend(identity + ": " + message for message in report["human_review"])
        blockers.extend(identity + ": " + message for message in report["blockers"])
        for command in report["provenance"]["qc_commands"]:
            commands.append({**command, **{key: prefix + value for key, value in command.items()
                                            if key in {"stdout", "stderr"}}})
        stage_rows.append({"requested_node": identity, "actual_source_node": identity,
                           "status": report["overall_status"], "report": prefix + "report.json",
                           "measurements": prefix + "measurements.json"})
    operation = "cortical_reconstruction" if any(item["rule_id"].startswith("SURFACE.") for item in checks) else "structural_preprocessing"
    status = overall_status(checks)
    if status == 'pass' and any(row['status'] == 'unknown' for row in stage_rows):
        status = 'unknown'
    report = {"report_id": catalog["subject"] + ":" + node["node_id"], "operation": operation,
              "inputs": inputs, "checks": checks, "overall_status": status,
              "quality_warnings": quality_warnings, "blockers": blockers, "human_review": reviews,
              "provenance": {"requested_node": node["node_id"], "checked_nodes": checked_nodes,
                  "checked_stage_provenance": stage_rows, "qc_commands": commands,
                  "group_execution": execution,
                  "started_at": started_at, "completed_at": utc_now(),
                  "scientific_approval": "not_performed"}}
    write_json(root / "report.json", report)
    write_json(root / "measurements.json", {"stages": {
        identity: read_json(root / "stages" / identity / "measurements.json") for identity in checked_nodes}})
    return report


def _copy_preserving(source_root, source_record, output_root, registry):
    source_root = Path(source_root).resolve()
    source = checked_file(source_record["path"])
    if not source.resolve().is_relative_to(source_root):
        raise ValueError("Declared output source escapes its stage result directory")
    relative = Path(source_record.get("_relative_path", source_record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Output member path must be relative and contained")
    target = output_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    key = relative.as_posix()
    if key in registry:
        if not filecmp.cmp(source, registry[key], shallow=False):
            raise ValueError("Two declared outputs map to different files at " + key)
    elif target.exists():
        if not filecmp.cmp(source, target, shallow=False):
            raise ValueError("Refusing to overwrite a different existing output: " + key)
        registry[key] = target
    else:
        shutil.copy2(source, target)
        registry[key] = target
    record = {key: value for key, value in source_record.items() if key not in {"path", "_relative_path"}}
    return {**record, "path": key, "size_bytes": target.stat().st_size}


def forward_outputs(spec, runtime, blueprint, node, catalog, report, mode):
    if mode not in {"strict", "report_only"}:
        raise ValueError("mode must be report_only or strict")
    policy = node.get("policy", {})
    approved = report["overall_status"] == "pass" and policy.get("status") == "approved"
    qc_root = Path("qc_result")
    qc_root.mkdir(parents=True, exist_ok=True)
    write_json(qc_root / "approval.json", {"approved": approved, "overall_status": report["overall_status"],
        "mode": mode, "policy_status": policy.get("status", "unreviewed"),
        "scientific_approval": "not_performed"})
    qc_out = qc_root / node["node_id"]
    qc_out.mkdir(parents=True, exist_ok=True)
    result_root = Path("result")
    result_root.mkdir(parents=True, exist_ok=True)
    if not approved and mode == "strict":
        write_json(qc_root / "blocked.json", {"overall_status": report["overall_status"],
            "scientific_approval": "not_performed"})
        return
    output_sources = policy.get("output_sources", {})
    declared_outputs = {item["name"]: item for item in node.get("outputs", [])}
    outputs = {}
    copied = {}
    for port, source in output_sources.items():
        identity, source_port = source["node_id"], source["port"]
        try:
            record = catalog["stages"][identity]["outputs"][source_port]
        except KeyError as error:
            raise ValueError("QC output source missing from upstream catalog: " + identity + "/" + source_port) from error
        directory = Path(catalog["stages"][identity]["root"])
        main_record = _copy_preserving(directory, record["record_file"], result_root, copied)
        members = [_copy_preserving(directory, item, result_root, copied) for item in record["files"]]
        if port not in declared_outputs:
            raise ValueError("QC output port is not declared by the selected node: " + port)
        declared = declared_outputs[port]
        contract = declared.get("contract", {key: value for key, value in declared.items() if key != "name"})
        outputs[port] = {"contract": contract, "path": main_record["path"]}
        if "members" in record["record_file"]:
            outputs[port]["members"] = members
    command_root = Path("qc_result") / node["node_id"].removesuffix("_qc")
    copied_commands = []
    for index, command in enumerate(report["provenance"].get("qc_commands", [])):
        copied_command = dict(command)
        for key in ("stdout", "stderr"):
            relative = Path(command[key])
            source = command_root / relative
            if not source.is_file():
                raise ValueError("QC command log is missing: " + str(source))
            destination_relative = Path("qc_logs") / node["node_id"] / relative
            destination = result_root / destination_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source, destination)
            copied_command[key] = destination_relative.as_posix()
        copied_commands.append(copied_command)
    stage_manifest = {"schema": "fastprep.stage/1", "subject": catalog["subject"],
        "node_id": node["node_id"], "outputs": outputs,
        "metrics": {"qc_execution_mode": mode, "qc_status": report["overall_status"],
                    "qc_approved": approved, "qc_report_id": report["report_id"]},
        "commands": copied_commands,
        "scientific_approval": "not_performed"}
    write_json(result_root / "manifest.json", stage_manifest)
    write_json(qc_out / "manifest.json", {"schema": "fastprep.qc/1", "report": "report.json",
        "approval": "../approval.json", "overall_status": report["overall_status"],
        "forwarded_outputs": outputs})


def _write_index(directories, output):
    reports = {}
    for directory in directories:
        directory = Path(directory)
        if (directory / "report.json").is_file():
            candidates = [directory / "report.json"]
        else:
            candidates = [path for path in directory.glob("*/report.json") if path.is_file()]
        for path in candidates:
            report = read_json(path)
            report_id = report["report_id"]
            current = reports.get(report_id)
            if current is None or len(path.parts) < len(current[0].parts):
                reports[report_id] = (path, report)
    reports = sorted(reports.values(), key=lambda item: item[1]["report_id"])
    summary = {"schema": "fastprep.qc-summary/1", "report_count": len(reports),
               "overall_status": "fail" if any(r["overall_status"] == "fail" for _, r in reports)
               else "unknown" if not reports or any(r["overall_status"] == "unknown" for _, r in reports) else "pass",
               "reports": []}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, report in reports:
        subject, gate = report["report_id"].split(":", 1)
        report_rel = Path(subject) / gate / "report.json"
        source_root = path.parent
        target_root = output / report_rel.parent
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            target = target_root / source.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not filecmp.cmp(source, target, shallow=False):
                    raise ValueError("Collected QC output conflicts with an existing file: " + str(target))
            else:
                shutil.copy2(source, target)
        copied_images = []
        for check in report.get("checks", []):
            for visual in check.get("visual_evidence", []):
                relative = Path(visual["source"])
                copied = (target_root / relative).is_file()
                if copied:
                    link = (report_rel.parent / relative).as_posix()
                    copied_images.append((link, relative.name))
                else:
                    copied_images.append((None, relative.name))
        summary["reports"].append({"node_id": gate, "status": report["overall_status"],
                                    "path": report_rel.as_posix()})
        links = " ".join(
            '<a href="' + html.escape(image) + '">' + html.escape(name) + '</a>'
            if image else html.escape(name + " (missing)") for image, name in copied_images)
        rows.append("<tr><td>" + html.escape(report["report_id"]) + "</td><td>" +
                    html.escape(report["overall_status"]) + "</td><td>" + links + "</td></tr>")
    write_json(output / "summary.json", summary)
    page = ("<!doctype html><meta charset='utf-8'><title>FastPrep QC</title><h1>FastPrep QC</h1>"
            "<p>Generated evidence is not scientific approval; visual review may remain pending.</p>"
            "<table><tr><th>Report</th><th>Status</th><th>Images</th></tr>" + "".join(rows) + "</table>")
    (output / "index.html").write_text(page, encoding="utf-8")


def run(spec, runtime, blueprint, mode):
    os.environ.update(MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen")
    runtime = {**runtime, 'qc': {**runtime.get('qc', {}),
                               'threads': spec.get('threads', runtime.get('qc', {}).get('threads', 1))}}
    catalog = staged_catalog(spec, runtime)
    node_id = spec.get("qc_node_id", spec.get("node_id"))
    if not node_id:
        raise ValueError("spec must contain qc_node_id or node_id")
    nodes = [item for item in blueprint["nodes"] if item.get("kind") == "qc" and item["node_id"] == node_id]
    if len(nodes) != 1:
        raise ValueError("Blueprint must contain exactly one selected QC node: " + node_id)
    node = nodes[0]
    if "stage_rules" in node.get("policy", {}):
        report = generate_group(catalog, node, runtime, "qc_result")
    else:
        report = Evidence(catalog, node, runtime, "qc_result").generate()
    report_path = Path("qc_result") / node["node_id"].removesuffix("_qc") / "report.json"
    forward_outputs(spec, runtime, blueprint, node, catalog, report, mode)
    _write_index([report_path.parent], Path("qc_result") / "report")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--mode", choices=("report_only", "strict"), default="report_only")
    parser.add_argument("--volume-engine", choices=("reference", "mapped", "batched"), default="reference")
    parser.add_argument('--surface-engine', choices=('reference', 'reuse'), default='reference')
    args = parser.parse_args(argv)
    spec, runtime, blueprint = read_json(args.spec), read_json(args.runtime), read_json(args.blueprint)
    runtime['qc'] = {**runtime.get('qc', {}), 'volume_engine': args.volume_engine, 'surface_engine': args.surface_engine}
    report = run(spec, runtime, blueprint, args.mode)
    print(report["report_id"] + ": " + report["overall_status"])
    node_id = spec.get("qc_node_id", spec.get("node_id"))
    node = next(item for item in blueprint["nodes"] if item.get("kind") == "qc" and item["node_id"] == node_id)
    approved = report["overall_status"] == "pass" and node.get("policy", {}).get("status") == "approved"
    return 0 if args.mode == "report_only" or approved else 2


if __name__ == "__main__":
    raise SystemExit(main())

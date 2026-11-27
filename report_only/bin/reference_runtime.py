"""Native FastPrep stages with explicit inputs, geometry checks and command records."""

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


HEMISPHERES = ("lh", "rh")
VOLUMES = {
    "orig_mgz": "orig.mgz", "rawavg_mgz": "rawavg.mgz", "aseg_synthseg_mgz": "aseg.synthseg.mgz",
    "aseg_auto_noccseg_mgz": "aseg.auto_noCCseg.mgz", "mask_mgz": "mask.mgz",
    "orig_nu_mgz": "orig_nu.mgz", "bias_factor_nii": "bias.correction-factor.nii.gz", "nu_mgz": "nu.mgz", "t1_mgz": "T1.mgz",
    "norm_mgz": "norm.mgz", "brainmask_mgz": "brainmask.mgz", "aseg_auto_mgz": "aseg.auto.mgz",
    "aseg_synthseg_withcc_mgz": "aseg.synthseg.withCC.mgz", "aseg_presurf_mgz": "aseg.presurf.mgz",
    "wm_seed_mgz": "wm.seed.synthseg.mgz", "wm_asegedit_mgz": "wm.asegedit.mgz",
    "brain_mgz": "brain.mgz", "brain_finalsurfs_mgz": "brain.finalsurfs.mgz", "wm_mgz": "wm.mgz",
    "filled_mgz": "filled.mgz", "aseg_presurf_hypos_mgz": "aseg.presurf.hypos.mgz",
    "ribbon_mgz": "ribbon.mgz", "lh_ribbon_mgz": "lh.ribbon.mgz", "rh_ribbon_mgz": "rh.ribbon.mgz",
    "aseg_mgz": "aseg.mgz", "aparc_aseg_mgz": "aparc+aseg.mgz",
}
HEMI_FILES = {
    "filled_pretess_mgz": "mri/{hemi}.filled.pretess.mgz",
    "orig_surf": "surf/{hemi}.orig", "orig_premesh_surf": "surf/{hemi}.orig.premesh",
    "orig_nofix_surf": "surf/{hemi}.orig.nofix", "smoothwm_nofix_surf": "surf/{hemi}.smoothwm.nofix",
    "inflated_nofix_surf": "surf/{hemi}.inflated.nofix", "qsphere_nofix_surf": "surf/{hemi}.qsphere.nofix",
    "orig_topofix_surf": "surf/{hemi}.orig.topofix",
    "autodet_gwstats": "stats/{hemi}.autodet.gw.stats.dat",
    "white_preaparc_surf": "surf/{hemi}.white.preaparc", "white_surf": "surf/{hemi}.white",
    "curv_surf": "surf/{hemi}.curv", "area_surf": "surf/{hemi}.area",
    "cortex_label": "label/{hemi}.cortex.label", "cortex_hipamyg_label": "label/{hemi}.cortex+hipamyg.label",
    "smoothwm_surf": "surf/{hemi}.smoothwm", "inflated_surf": "surf/{hemi}.inflated",
    "sulc_surf": "surf/{hemi}.sulc", "curv_stats": "stats/{hemi}.curv.stats",
    "sphere_surf": "surf/{hemi}.sphere", "sphere_reg_surf": "surf/{hemi}.sphere.reg",
    "jacobian_white": "surf/{hemi}.jacobian_white", "aparc_annot": "label/{hemi}.aparc.annot",
    "pial_surf": "surf/{hemi}.pial", "pial_t1_surf": "surf/{hemi}.pial.T1",
    "curv_pial_surf": "surf/{hemi}.curv.pial", "area_pial_surf": "surf/{hemi}.area.pial",
    "thickness_surf": "surf/{hemi}.thickness", "w_g_pct_mgh": "surf/{hemi}.w-g.pct.mgh",
    "w_g_pct_stats": "stats/{hemi}.w-g.pct.stats", "aparc_pial_stats": "stats/{hemi}.aparc.pial.stats",
    "aparc_stats": "stats/{hemi}.aparc.stats",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked_file(path, *, allow_empty=False):
    path = Path(path).expanduser().absolute()
    if not path.is_file() or (not allow_empty and path.stat().st_size == 0):
        raise ValueError("Missing or empty file: " + str(path))
    return path


def copy_file(source, destination):
    source, destination = checked_file(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    return destination


def artifact_path(directory, record):
    directory = Path(directory).absolute()
    path = directory / record["path"]
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Upstream artifact escapes its result directory")
    return checked_file(path)


class NativeStage:
    def __init__(self, spec, runtime, blueprint, result_dir="result"):
        self.spec, self.runtime, self.blueprint = spec, runtime, blueprint
        self.node = next(node for node in blueprint["nodes"] if node["node_id"] == spec["node_id"])
        self.subject = spec["subject"]
        if not re.fullmatch(r"sub-[A-Za-z0-9]+(?:_ses-[A-Za-z0-9]+)?", self.subject):
            raise ValueError("Invalid subject/session ID")
        self.root = Path(result_dir).absolute()
        self.subjects_dir = self.root / "subjects"
        self.subject_dir = self.subjects_dir / self.subject
        for name in ("mri/transforms", "surf", "label", "stats", "scripts"):
            (self.subject_dir / name).mkdir(parents=True, exist_ok=True)
        self.inputs, self.staged_inputs, self.upstream = {}, {}, {}
        self.logs, self.metrics = [], {}
        for directory in spec.get("upstream", []):
            directory = Path(directory).absolute()
            manifest = read_json(directory / "manifest.json")
            if manifest["subject"] != self.subject or manifest["node_id"] in self.upstream:
                raise ValueError("Upstream subject or node identity mismatch")
            if manifest.get("schema") != "fastprep.stage/1":
                raise ValueError("Only native stage artifacts can enter scientific execution")
            self.upstream[manifest["node_id"]] = (directory, manifest)
        self.verify_upstream()
        for edge in blueprint["edges"]:
            if edge["target"] != self.node["node_id"] or edge["source"] == "$input":
                continue
            directory, manifest = self.upstream[edge["source"]]
            self.inputs[edge["target_port"]] = (directory, manifest["outputs"][edge["source_port"]])
        expected = {port["name"] for port in self.node["inputs"] if port["name"] != "bids"}
        if set(self.inputs) != expected:
            raise ValueError("Input port mismatch for " + self.node["node_id"])

    def path(self, port, hemi=None):
        if (port, hemi) not in self.staged_inputs:
            raise ValueError("Missing staged input: " + port + ("/" + hemi if hemi else ""))
        return self.staged_inputs[port, hemi]

    def _upstream_path(self, port, hemi=None):
        directory, record = self.inputs[port]
        if hemi:
            matches = [member for member in record["members"] if member.get("hemi") == hemi]
            if len(matches) != 1:
                raise ValueError("Missing or ambiguous hemisphere: " + port + "/" + hemi)
            record = matches[0]
        return artifact_path(directory, record)

    def verify_upstream(self):
        for directory, manifest in self.upstream.values():
            for record in manifest["outputs"].values():
                for item in [record, *record.get("members", [])]:
                    artifact_path(directory, item)

    def output(self, relative):
        path = self.subject_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def volume(self, port):
        return self.output("mri/" + VOLUMES[port])

    def hemi(self, port, hemi):
        return self.output(HEMI_FILES[port].format(hemi=hemi))

    def stage_inputs(self):
        prefix = "subjects/" + self.subject + "/"
        for node in self.blueprint["nodes"]:
            if node["node_id"] not in self.upstream:
                continue
            directory, manifest = self.upstream[node["node_id"]]
            for record in manifest["outputs"].values():
                for item in record.get("members", [record]):
                    if item["path"].startswith(prefix):
                        copy_file(directory / item["path"], self.output(item["path"][len(prefix):]))
        aliases = {"white_surf_native": "white_surf", "pial_surf_native": "pial_surf",
                   "sphere_reg_native": "sphere_reg_surf", "sulc_native": "sulc_surf"}
        for port, (directory, record) in self.inputs.items():
            name = aliases.get(port, port)
            hemispheres = [None, *(member["hemi"] for member in record.get("members", []) if member.get("hemi"))]
            for hemi in hemispheres:
                source = self._upstream_path(port, hemi)
                relative = source.relative_to(directory)
                if name in VOLUMES:
                    destination = self.volume(name)
                elif hemi and name in HEMI_FILES:
                    destination = self.hemi(name, hemi)
                    if record.get("contract", {}).get("space") == "fsaverage6":
                        destination = destination.with_name(destination.name + ".fsaverage6")
                elif port.startswith("talairach_"):
                    filename = {"talairach_auto_xfm": "talairach.auto.xfm", "talairach_xfm": "talairach.xfm",
                                "talairach_xfm_lta": "talairach.xfm.lta", "talairach_lta": "talairach.lta"}[port]
                    destination = self.output("mri/transforms/" + filename)
                elif relative.as_posix().startswith(prefix):
                    destination = self.root / relative
                else:
                    destination = self.root / "inputs" / port / source.name
                self.staged_inputs[port, hemi] = copy_file(source, destination)

    def command(self, command_id):
        command = self.runtime["commands"].get(command_id)
        if not command or not command.get("argv_prefix"):
            raise ValueError("Missing command configuration: " + command_id)
        return command

    def run(self, command_id, arguments):
        command = self.command(command_id)
        self.run_process(command_id, [*command["argv_prefix"], *arguments], command.get("environment", {}), self.subject_dir)

    def run_process(self, command_id, argv, extra_env, cwd, *, check=True):
        record = self._execute_process(command_id, argv, extra_env, cwd, len(self.logs) + 1,
                                       self.spec['threads'], check=False)
        self.logs.append(record)
        self.write_commands()
        if check and record['exit_code']:
            raise RuntimeError('Native command failed: ' + command_id + '; see ' + str(self.root / record['stderr']))
        return record['exit_code']

    def write_commands(self):
        write_json(self.root / getattr(self, 'command_journal', 'commands.json'), self.logs)

    def _execute_process(self, command_id, argv, extra_env, cwd, sequence, threads, *, check=True):
        environment = {**os.environ, **extra_env, "SUBJECTS_DIR": str(self.subjects_dir),
                       "OMP_NUM_THREADS": str(threads), "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS": str(threads)}
        for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VTK_SMP_MAX_THREADS', 'NUMEXPR_NUM_THREADS'):
            configured = environment.get(variable, str(threads))
            environment[variable] = str(min(int(threads), max(1, int(configured))))
        argv = list(map(str, argv))
        relative_directory = Path('logs') / getattr(self, 'log_namespace', '')
        directory = self.root / relative_directory
        directory.mkdir(parents=True, exist_ok=True)
        name = str(sequence).zfill(3) + "-" + command_id.replace(":", "-")
        started = time.perf_counter()
        with (directory / (name + ".stdout")).open("w") as stdout, (directory / (name + ".stderr")).open("w") as stderr:
            try:
                code = subprocess.run(argv, env=environment, cwd=cwd, stdout=stdout, stderr=stderr, check=False).returncode
            except OSError as error:
                if check:
                    raise
                stderr.write(str(error) + "\n")
                code = 127
        return {"command_id": command_id, "argv": argv, "exit_code": code,
                  'duration_seconds': time.perf_counter() - started, 'threads': int(threads),
                  "stdout": str(relative_directory / (name + '.stdout')),
                  "stderr": str(relative_directory / (name + '.stderr'))}

    def run_concurrent(self, jobs, *, max_workers, check=True):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not jobs or max_workers < 1 or int(self.spec['threads']) < 1:
            raise ValueError('Concurrent commands require jobs, positive workers and an allocated CPU budget')
        workers = min(max_workers, len(jobs), int(self.spec['threads']))
        threads = max(1, int(self.spec['threads']) // workers)
        prepared = [(identity, arguments, self.command(identity)) for identity, arguments in jobs]
        offset = len(self.logs)
        self.logs.extend(dict(command_id=identity, exit_code=None) for identity, _ in jobs)
        self.write_commands()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for index, (identity, arguments, command) in enumerate(prepared):
                environment = {**command.get('environment', {}), 'OPENBLAS_NUM_THREADS': str(threads),
                               'MKL_NUM_THREADS': str(threads), 'VTK_SMP_MAX_THREADS': str(threads)}
                future = executor.submit(self._execute_process, identity, [*command['argv_prefix'], *arguments],
                                         environment, self.subject_dir, offset + index + 1, threads, check=False)
                futures[future] = offset + index
            for future in as_completed(futures):
                self.logs[futures[future]] = future.result()
                self.write_commands()
        failed = [record for record in self.logs[offset:] if record['exit_code']]
        if failed and check:
            raise RuntimeError('Concurrent native command failed; see ' + ', '.join(str(self.root / record['stderr']) for record in failed))

    def each_hemisphere(self, operation, *, minimum_threads=1, maximum_threads=None, uses_gpu=False):
        from concurrent.futures import ThreadPoolExecutor

        allocated = int(self.spec['threads'])
        parameters = self.node.get('parameters', {})
        requested = parameters.get('hemisphere_workers', 2)
        if type(requested) is not int or requested not in (1, 2):
            raise ValueError('hemisphere_workers must be 1 or 2')
        gpu_workers = parameters.get('gpu_hemisphere_workers', 2)
        if type(gpu_workers) is not int or gpu_workers not in (1, 2):
            raise ValueError('gpu_hemisphere_workers must be 1 or 2')
        if allocated < 1 or minimum_threads < 1:
            raise ValueError('Hemisphere execution requires a positive CPU budget')
        workers = min(requested, max(1, allocated // minimum_threads), gpu_workers if uses_gpu else 2)
        worker_threads = allocated // workers
        if workers > 1 and maximum_threads is not None:
            if type(maximum_threads) is not int or maximum_threads < minimum_threads:
                raise ValueError('maximum_threads must be an integer at least minimum_threads')
            worker_threads = min(worker_threads, maximum_threads)
        started = time.perf_counter()
        execution = dict(workers=workers, allocated_threads=allocated,
                         threads_per_hemisphere=worker_threads, uses_gpu=uses_gpu, hemispheres={})
        self.metrics['hemisphere_execution'] = execution

        def execute(worker_stage, hemi):
            hemisphere_started = time.perf_counter()
            try:
                value = operation(worker_stage, hemi)
            except Exception as error:
                execution['hemispheres'][hemi] = dict(status='failed', seconds=time.perf_counter() - hemisphere_started,
                                                      error=str(error))
                raise
            execution['hemispheres'][hemi] = dict(status='completed', seconds=time.perf_counter() - hemisphere_started)
            return value

        if workers == 1:
            try:
                return {hemi: execute(self, hemi) for hemi in HEMISPHERES}
            finally:
                execution['wall_seconds'] = time.perf_counter() - started
                write_json(self.root / 'hemisphere_execution.json', execution)

        children = {}
        for hemi in HEMISPHERES:
            child = copy.copy(self)
            child.spec = dict(self.spec, threads=worker_threads)
            child.metrics, child.logs = {}, []
            child.log_namespace = hemi
            child.command_journal = 'logs/' + hemi + '/commands.json'
            children[hemi] = child
        values, errors = {}, []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {hemi: executor.submit(execute, child, hemi) for hemi, child in children.items()}
            for hemi, future in futures.items():
                try:
                    values[hemi] = future.result()
                except Exception as error:
                    errors.append(error)
        for hemi, child in children.items():
            self.logs.extend(dict(record, hemi=hemi) for record in child.logs)
        self.write_commands()
        execution['wall_seconds'] = time.perf_counter() - started
        write_json(self.root / 'hemisphere_execution.json', execution)
        if errors:
            raise errors[0]

        def merge_metrics(destination, source):
            for key, value in source.items():
                if key not in destination:
                    destination[key] = value
                elif isinstance(value, dict) and isinstance(destination[key], dict):
                    merge_metrics(destination[key], value)
                else:
                    raise ValueError('Hemisphere workers wrote the same metric: ' + key)

        merged = copy.deepcopy(self.metrics)
        for child in children.values():
            merge_metrics(merged, child.metrics)
        self.metrics = merged
        return values

    def resource_file(self, command_id, pattern):
        matches = [path for path in self.command(command_id).get("files", []) if re.fullmatch(pattern, Path(path).name)]
        if len(matches) != 1:
            raise ValueError("Missing or ambiguous command resource: " + pattern)
        return checked_file(matches[0])

    def resource(self, identity):
        matches = [record for record in self.runtime["resources"] if record["id"] == identity]
        if len(matches) != 1:
            raise ValueError("Missing resource: " + identity)
        return checked_file(matches[0]["path"])

    def file_record(self, path, **metadata):
        path = checked_file(path)
        return {"path": str(path.relative_to(self.root)), "size_bytes": path.stat().st_size, **metadata}

    def collection(self, port, paths):
        members = [self.file_record(path, hemi=hemi) for hemi, path in paths.items()]
        manifest = self.root / "collections" / (port + ".json")
        write_json(manifest, {"subject": self.subject, "members": members})
        return self.file_record(manifest, members=members)

    def finish(self, outputs):
        expected = {port["name"] for port in self.node["outputs"]}
        if set(outputs) != expected:
            raise ValueError("Output port mismatch: " + self.node["node_id"])
        records = {}
        for port, value in outputs.items():
            declared = next(item for item in self.node["outputs"] if item["name"] == port)
            if declared["cardinality"] == "both_hemispheres":
                if set(value) != set(HEMISPHERES):
                    raise ValueError("Both hemispheres are required: " + port)
                record = self.collection(port, value)
            else:
                record = self.file_record(value)
                if port in {"alignment_manifest", "pial_refinement_manifest", "mni_manifest"}:
                    record["members"] = [self.file_record(value, hemi=None), *read_json(value)["members"]]
                if port == "orig_dir":
                    record["members"] = read_json(value)["members"]
            records[port] = {**record, "contract": declared}
        write_json(self.root / "manifest.json", {"schema": "fastprep.stage/1", "subject": self.subject,
                   "node_id": self.node["node_id"], "outputs": records, "metrics": self.metrics,
                   "commands": self.logs, "scientific_approval": "not_performed"})


def volume_outputs(stage):
    return {port["name"]: stage.volume(port["name"]) for port in stage.node["outputs"]}


def hemi_outputs(stage):
    return {port["name"]: {hemi: stage.hemi(port["name"], hemi) for hemi in HEMISPHERES}
            for port in stage.node["outputs"]}


from session_inputs import session_plan, align_companions
from mni_normalize import normalize_session
from tissue_bias import tissue_bias_correct


def t1w_selection(stage):
    plan = session_plan(stage.runtime, stage.subject)
    reference = plan["reference"]
    manifest = stage.root / "t1w_manifest.json"
    write_json(manifest, {"schema": "fastprep.t1w-manifest/1", "subject_id": plan["subject_id"],
                         "session_id": plan["session_id"], "analysis_unit_id": stage.subject,
                         "scans": [{key: reference[key] for key in ("path", "entities")}],
                         "companions": plan["companions"], "visual_review": "pending",
                         **({"selection": plan["selection"]} if "selection" in plan else {})})
    stage.metrics["input_review"] = "structural checks passed; anatomical visual review pending"
    return {"t1w_manifest": manifest}


def orig_directory(stage):
    scans = read_json(stage.path("t1w_manifest"))["scans"]
    if len(scans) != 1:
        raise ValueError("Session reconstruction requires exactly one selected T1w")
    members = []
    for index, scan in enumerate(scans, start=1):
        path = stage.output(f"mri/orig/{index:03d}.mgz")
        stage.run("freesurfer:mri_convert", [checked_file(scan["path"]), path])
        members.append(stage.file_record(path, source=scan))
    manifest = stage.root / "orig_dir.json"
    write_json(manifest, {"schema": "fastprep.orig-manifest/1", "subject_id": stage.subject, "members": members})
    return {"orig_dir": manifest}


def recon(stage):
    fov = None
    if stage.node["node_id"] == "anat_motioncor":
        from conform_fov import motioncor_fov_policy

        paths = [checked_file(stage.root / member["path"])
                 for member in read_json(stage.path("orig_dir"))["members"]]
        fov = motioncor_fov_policy(paths)
        write_json(stage.root / "conform_fov.json", fov)
    stage.run("freesurfer:recon-all", ["-s", stage.subject, "-sd", stage.subjects_dir,
              "-no-isrunning", "-threads", stage.spec["threads"], "-itkthreads", stage.spec["threads"],
              *(["-no-v8"] if fov is not None else []),  # Only the motioncor stage was requested.
              *(fov["flags"] if fov else []),
              *stage.node["parameters"]["recon_all_stages"]])
    if fov is not None:
        from conform_fov import record_conform_output

        stage.metrics["conform_fov"] = record_conform_output(fov, stage.volume("orig_mgz"))
        write_json(stage.root / "conform_fov.json", fov)
    return volume_outputs(stage)


def same_grid(reference, image):
    import numpy as np

    if len(reference.shape) != 3 or image.shape != reference.shape or not np.allclose(
            reference.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("Physical grids differ; FastPrep does not implicitly resample labels")
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite scientific volume")
    return values


def checked_labels(reference, image, allowed=None):
    import numpy as np

    values = same_grid(reference, image)
    if not np.array_equal(values, np.rint(values)) or values.min() < 0:
        raise ValueError("Segmentation must have nonnegative integer labels")
    present = set(np.unique(values).astype(int).tolist())
    if allowed is not None and not present.issubset(set(allowed)):
        raise ValueError("Unexpected segmentation labels: " + str(sorted(present - set(allowed))))
    return values


def save_mgz(values, reference, destination, dtype):
    import nibabel as nib
    import numpy as np

    image = nib.MGHImage(np.asarray(values, dtype=dtype), reference.affine, header=reference.header.copy())
    image.set_data_dtype(dtype)
    nib.save(image, destination)


def synthseg(stage):
    import nibabel as nib
    import numpy as np

    original = nib.load(stage.path("orig_mgz"))
    values = same_grid(original, original)
    directory = stage.root / "synthseg"
    directory.mkdir()
    image = directory / "orig.nii.gz"
    tensor, geometry = directory / "image.npy", directory / "geometry.json"
    volumes = directory / "volumes.npy"
    runtime_info = directory / "runtime.json"
    segmentation = directory / "aseg.nii.gz"
    nib.save(nib.Nifti1Image(values.astype(np.float32), original.affine), image)
    threads = ["--threads", stage.spec["threads"]]
    stage.run("synthseg:preprocess", ["preprocess", "--image", image, "--output-image", tensor,
              "--output-geometry", geometry, *threads])
    labels = stage.resource_file("synthseg:segment", r"synthseg_segmentation_labels_2\.0\.npy")
    ensemble_options = ["--sequential-flip"] if stage.spec["device"] != "cpu" else []
    stage.run("synthseg:segment", ["segment-base", "--image", tensor, "--checkpoint",
              stage.resource_file("synthseg:segment", r"synthseg_2\.0\.h5"), "--labels", labels,
              "--denoiser-labels", stage.resource_file("synthseg:segment", r"synthseg_denoiser_labels_2\.0\.npy"),
              "--geometry", geometry, "--topology",
              stage.resource_file("synthseg:segment", r"synthseg_topological_classes_2\.0\.npy"),
              "--output-segmentation", segmentation, "--output-volumes", volumes,
              "--device", stage.spec["device"], "--runtime-info", runtime_info,
              *ensemble_options, *threads])
    recovered = checked_labels(original, nib.load(segmentation), np.unique(np.load(labels, allow_pickle=False)))
    if any(not np.any(recovered == label) for label in (2, 41)):
        raise ValueError("SynthSeg did not identify both cerebral white-matter hemispheres")
    save_mgz(recovered, original, stage.volume("aseg_synthseg_mgz"), np.int16)
    stage.metrics["synthseg"] = {"variant": "2.0-standard-base", **read_json(runtime_info),
                               "labels_present": np.unique(recovered).astype(int).tolist(),
                               "grid_recovery": "exact shape/affine; no extra label resampling",
                               "parcellation": False}
    return volume_outputs(stage)


def orig_to_nifti(stage):
    import nibabel as nib
    import numpy as np

    output = stage.root / "orig.conformed.nii.gz"
    stage.run("freesurfer:mri_convert", [stage.path("orig_mgz"), output])
    reference = nib.load(stage.path("orig_mgz"))
    converted = same_grid(reference, nib.load(output))
    if not np.array_equal(converted, np.asarray(reference.dataobj)):
        raise ValueError("HD-BET input conversion changed anatomical intensities")
    return {"orig_nii": output}


def hdbet_mask(stage):
    import nibabel as nib
    import numpy as np

    command = "hd-bet:predict-mask"
    model_dir = stage.resource_file(command, r"plans\.json").parent
    source_root = stage.resource_file(command, r"hd_bet_prediction\.py").parents[1]
    output, provenance = stage.root / "brain-mask.nii.gz", stage.root / "predict-provenance.json"
    requested = str(stage.spec["device"])
    device = "cpu" if requested == "auto" else "cuda:" + requested if requested.isdigit() else requested
    stage.run(command, ["--image", stage.path("orig_nii"), "--model-dir", model_dir,
              "--source-root", source_root, "--device", device, "--output", output, "--provenance", provenance])
    labels = checked_labels(nib.load(stage.path("orig_nii")), nib.load(output), [0, 1])
    if not np.any(labels == 1):
        raise ValueError("HD-BET produced an empty brain mask")
    stage.metrics["hdbet"] = {"device": device, "test_time_augmentation": False,
                               "foreground_voxels": int(np.count_nonzero(labels)), "same_grid": True}
    return {"mask_nii": output, "hdbet_provenance": provenance}


def mask_to_mgz(stage):
    import nibabel as nib
    import numpy as np

    stage.run("freesurfer:mri_convert", ["--out_data_type", "uchar", stage.path("mask_nii"), stage.volume("mask_mgz")])
    reference = nib.load(stage.path("orig_mgz"))
    before = checked_labels(reference, nib.load(stage.path("mask_nii")), [0, 1])
    after = checked_labels(reference, nib.load(stage.volume("mask_mgz")), [0, 1])
    if not np.array_equal(before, after) or not np.any(after):
        raise ValueError("HD-BET mask conversion changed the binary mask")
    return volume_outputs(stage)


def prepare_synthseg_aseg(stage):
    import nibabel as nib
    import numpy as np

    source = nib.load(stage.path("aseg_synthseg_mgz"))
    values = checked_labels(source, source)
    if any(not np.any(values == label) for label in (2, 41)) or values.max() > 60:
        raise ValueError("A bilateral SynthSeg base aseg is required, not DKT parcellation")
    stage.run("freesurfer:mri_convert", [stage.path("aseg_synthseg_mgz"), stage.volume("aseg_auto_noccseg_mgz")])
    recovered = checked_labels(source, nib.load(stage.volume("aseg_auto_noccseg_mgz")))
    if not np.array_equal(values, recovered):
        raise ValueError("SynthSeg aseg preparation changed base segmentation labels")
    return volume_outputs(stage)


def validate_lta(path):
    import numpy as np

    text = checked_file(path).read_text()
    match = re.search(r"(?m)^1\s+4\s+4\s*\n((?:[^\n]+\n){4})", text)
    if not match or not re.search(r"(?m)^type\s*=\s*[01]\b", text):
        raise ValueError("An affine FreeSurfer LTA is required")
    matrix = np.asarray([[float(value) for value in line.split()] for line in match[1].splitlines()])
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1]) or abs(np.linalg.det(matrix[:3, :3])) < 1e-8):
        raise ValueError("Invalid or singular GCA affine transform")
    if len(re.findall(r"(?m)^valid\s*=\s*1\b", text)) != 2:
        raise ValueError("GCA LTA must record valid source and target volume geometries")
    return matrix.tolist()


def gca_register(stage):
    import nibabel as nib

    original = nib.load(stage.path("orig_nu_mgz"))
    same_grid(original, original)
    checked_labels(original, nib.load(stage.path("mask_mgz")), [0, 1])
    atlas = stage.resource("freesurfer-gca-brain")
    output = stage.output("mri/transforms/gca.lta")
    stage.run("freesurfer:mri_em_register", ["-uns", stage.node["parameters"]["uns"],
              "-mask", stage.path("mask_mgz"), stage.path("orig_nu_mgz"), atlas, output])
    stage.metrics["gca_registration"] = {"atlas": str(atlas), "source": str(stage.path("orig_nu_mgz")),
                                          "matrix": validate_lta(output),
                                          "target": "brain GCA atlas; not MNI305 xfm"}
    return {"gca_lta": output}


def talairach(stage):
    command = "freesurfer:talairach_avi"
    auto = stage.output("mri/transforms/talairach.auto.xfm")
    transform = stage.output("mri/transforms/talairach.xfm")
    lta = stage.output("mri/transforms/talairach.xfm.lta")
    stage.run(command, ["--i", stage.path("orig_nu_mgz"), "--xfm", auto])
    copy_file(auto, transform)
    target = stage.resource_file(command, r"mni305\.cor\.mgz")
    stage.run("freesurfer:lta_convert", ["--src", stage.path("orig_mgz"), "--trg", target,
              "--inmni", transform, "--outlta", lta, "--subject", "fsaverage", "--ltavox2vox"])
    validate_lta(stage.path("gca_lta"))
    plain = copy_file(stage.path("gca_lta"), stage.output("mri/transforms/talairach.lta"))
    stage.run("freesurfer:mri_add_xform_to_header", [transform, stage.path("orig_nu_mgz"), stage.volume("nu_mgz")])
    return {"nu_mgz": stage.volume("nu_mgz"), "talairach_auto_xfm": auto, "talairach_xfm": transform,
            "talairach_xfm_lta": lta, "talairach_lta": plain}


def brainmask(stage):
    import nibabel as nib
    import numpy as np

    reference = nib.load(stage.path("nu_mgz"))
    before = same_grid(reference, reference)
    mask = checked_labels(reference, nib.load(stage.path("mask_mgz")), [0, 1])
    if not np.any(mask):
        raise ValueError("The approved HD-BET brain mask is empty")
    stage.run("freesurfer:mri_mask", [stage.path("nu_mgz"), stage.path("mask_mgz"), stage.volume("norm_mgz")])
    after = same_grid(reference, nib.load(stage.volume("norm_mgz")))
    if not np.array_equal(after, np.where(mask > 0, before, 0)):
        raise ValueError("Brain extraction must preserve in-mask corrected intensities")
    copy_file(stage.path("nu_mgz"), stage.volume("t1_mgz"))
    copy_file(stage.volume("norm_mgz"), stage.volume("brainmask_mgz"))
    stage.metrics["intensity_contract"] = {"t1_mgz": "byte-identical corrected nu_mgz; no second normalization",
                                            "norm_mgz": "HD-BET mask applied to corrected nu_mgz",
                                            "brainmask_mgz": "byte-identical norm_mgz; intensity image, not binary mask"}
    return volume_outputs(stage)


def paint_cc(stage):
    import nibabel as nib
    import numpy as np

    stage.run("freesurfer:mri_cc", ["-aseg", "aseg.auto_noCCseg.mgz", "-norm", "norm.mgz",
              "-o", "aseg.auto.mgz", "-lta", "cc_up.lta", "-sdir", stage.subjects_dir, stage.subject])
    cc = nib.load(str(stage.volume("aseg_auto_mgz")))
    source = nib.load(stage.path("aseg_synthseg_mgz"))
    values = checked_labels(source, cc, [*range(61), *range(251, 256)])
    stage.metrics["corpus_callosum"] = {"source": "SynthSeg base aseg", "labels_present": np.unique(values).astype(int).tolist()}
    copy_file(stage.volume("aseg_auto_mgz"), stage.volume("aseg_synthseg_withcc_mgz"))
    return volume_outputs(stage)


def validate_filled(reference, segmentation, filled, stage):
    import numpy as np
    from scipy.ndimage import label as connected_labels

    aseg = checked_labels(reference, segmentation)
    values = checked_labels(reference, filled, [0, 127, 255])
    facts = {"same_grid": True, "labels_present": np.unique(values).astype(int).tolist(), "hemispheres": {}}
    for hemi, seed_label, fill_value, other_value in (("lh", 2, 255, 127), ("rh", 41, 127, 255)):
        support = values == fill_value
        expected = int(np.count_nonzero(support & (aseg == seed_label)))
        crossed = int(np.count_nonzero((values == other_value) & (aseg == seed_label)))
        if not np.any(support):
            raise ValueError("Filled hemisphere is missing or inconsistent with SynthSeg cerebral WM: " + hemi)
        facts["hemispheres"][hemi] = {"value": fill_value, "voxels": int(np.count_nonzero(support)),
                                      "components_6_connected": int(connected_labels(support)[1]),
                                      "cerebral_wm_overlap": expected, "opposite_fill_overlap": crossed}
        if expected <= crossed:
            quality_warning(stage, hemi, "filled_wm_overlap", facts["hemispheres"][hemi])
    return facts


def synthseg_fill(stage):
    import nibabel as nib
    import numpy as np

    reference = nib.load(stage.path("norm_mgz"))
    same_grid(reference, reference)
    labels = checked_labels(reference, nib.load(stage.path("aseg_auto_mgz")))
    same_grid(reference, nib.load(stage.path("brainmask_mgz")))
    parameters = stage.node["parameters"]
    if (parameters["wm_seed_labels"] != [2, 41] or parameters["wm_seed_value"] != 110
            or parameters["wm_seed_dtype"] != "uint8"):
        raise ValueError("The validated FastPrep seed contract is cerebral WM 2/41 at value 110")
    if any(not np.any(labels == value) for value in parameters["wm_seed_labels"]):
        raise ValueError("SynthSeg cerebral WM is missing")
    validate_lta(stage.path("talairach_lta"))
    copy_file(stage.path("aseg_auto_mgz"), stage.volume("aseg_presurf_mgz"))
    copy_file(stage.path("norm_mgz"), stage.volume("brain_mgz"))
    stage.run("freesurfer:mri_mask", ["-T", 5, stage.volume("brain_mgz"),
              stage.path("brainmask_mgz"), stage.volume("brain_finalsurfs_mgz")])
    stage.run("freesurfer:mri_binarize", ["--i", stage.volume("aseg_presurf_mgz"),
              "--match", *parameters["wm_seed_labels"], "--binval", parameters["wm_seed_value"], "--uchar",
              "--o", stage.volume("wm_seed_mgz")])
    seed = nib.load(stage.volume("wm_seed_mgz"))
    checked_labels(reference, seed, [0, 110])
    if seed.get_data_dtype().kind != "u" or seed.get_data_dtype().itemsize != 1:
        raise ValueError("mri_edit_wm_with_aseg requires an MRI_UCHAR white-matter seed")
    stage.run("freesurfer:mri_edit_wm_with_aseg", [stage.volume("wm_seed_mgz"),
              stage.volume("brain_mgz"), stage.volume("aseg_presurf_mgz"), stage.volume("wm_asegedit_mgz")])
    edited = checked_labels(reference, nib.load(stage.volume("wm_asegedit_mgz")))
    stage.run("freesurfer:mri_pretess", [stage.volume("wm_asegedit_mgz"), "wm",
              stage.path("norm_mgz"), stage.volume("wm_mgz")])
    checked_labels(reference, nib.load(stage.volume("wm_mgz")))
    cuts = stage.output("scripts/ponscc.cut.log")
    stage.run("freesurfer:mri_fill", ["-a", cuts, "-xform", stage.path("talairach_lta"),
              "-segmentation", stage.volume("aseg_presurf_mgz"), "-topofix", stage.path("norm_mgz"),
              stage.volume("wm_mgz"), stage.volume("filled_mgz")])
    stage.metrics["filled"] = validate_filled(reference, nib.load(stage.volume("aseg_presurf_mgz")),
                                                nib.load(stage.volume("filled_mgz")), stage)
    stage.metrics["subcortical_support"] = {
        "method": "FreeSurfer mri_edit_wm_with_aseg, not a simple segmentation union",
        "supported_voxels_by_label": {str(value): int(np.count_nonzero((labels == value) & (edited >= 5)))
                                      for value in parameters["subcortical_support_labels"]}}
    for name in ("brain_mgz", "brain_finalsurfs_mgz"):
        same_grid(reference, nib.load(stage.volume(name)))
    outputs = {item["name"]: stage.volume(item["name"]) for item in stage.node["outputs"] if item["name"] != "ponscc_cut_log"}
    return {**outputs, "ponscc_cut_log": checked_file(cuts)}


def tessellate(stage):
    import nibabel as nib
    import numpy as np

    reference = nib.load(stage.path("orig_mgz"))
    values = checked_labels(reference, nib.load(stage.path("filled_mgz")), [0, 127, 255])
    same_grid(reference, nib.load(stage.path("norm_mgz")))
    if stage.node["parameters"]["hemisphere_values"] != {"lh": 255, "rh": 127}:
        raise ValueError("Direct tessellation requires the FreeSurfer filled hemisphere convention")
    voxel_to_tkr = reference.header.get_vox2ras_tkr()
    shape = np.asarray(reference.shape)
    spacing = np.asarray(reference.header.get_zooms()[:3])
    axes = reference.affine[:3, :3] / spacing
    expected_geometry = {"volume": shape, "voxelsize": spacing, "xras": axes[:, 0],
                         "yras": axes[:, 1], "zras": axes[:, 2],
                         "cras": nib.affines.apply_affine(reference.affine, shape / 2)}
    def process_hemisphere(stage, hemi):
        value = {"lh": 255, "rh": 127}[hemi]
        if not np.any(values == value):
            raise ValueError("Cannot tessellate an empty filled hemisphere: " + hemi)
        output = stage.hemi("orig_nofix_surf", hemi)
        raw = stage.root / (hemi + ".tessellated")
        # recon-all performs a second, label-specific pretess AFTER mri_fill.
        # Each hemisphere starts from the same preserved filled volume.
        initializer = stage.hemi("filled_pretess_mgz", hemi)
        stage.run("freesurfer:mri_pretess", [stage.path("filled_mgz"), value,
                                            stage.path("norm_mgz"), initializer])
        prepared = checked_labels(reference, nib.load(initializer), [0, 127, 255])
        if not np.any(prepared == value):
            raise ValueError("Pretess removed the filled hemisphere: " + hemi)
        changed = values != prepared
        other_value = 127 if value == 255 else 255
        if np.any(changed & (values != value) & (prepared != value)):
            quality_warning(stage, hemi, 'pretess_unrelated_label_changes', {
                'voxel_count': int(np.count_nonzero(changed & (values != value) & (prepared != value)))})
        transitions, counts = np.unique(np.column_stack((values[changed], prepared[changed])),
                                         axis=0, return_counts=True)
        stage.run("freesurfer:mri_tessellate", [initializer, value, raw])
        coordinates, faces = nib.freesurfer.read_geometry(raw)
        triangular = stage.root / (hemi + ".tessellated.triangle")
        nib.freesurfer.write_geometry(triangular, coordinates, faces)
        stage.run("freesurfer:mris_convert", ["--vol-geom", stage.path("orig_mgz"), triangular, output])
        coordinates, faces, geometry = nib.freesurfer.read_geometry(output, read_metadata=True)
        if (not np.isfinite(coordinates).all() or not len(faces) or faces.min() < 0
                or faces.max() >= len(coordinates) or str(geometry.get("valid", "")).split()[:1] != ["1"]
                or any(key not in geometry or not np.allclose(geometry[key], expected, atol=1e-4, rtol=0)
                       for key, expected in expected_geometry.items())):
            raise ValueError("Invalid direct filled tessellation: " + hemi)
        voxels = nib.affines.apply_affine(np.linalg.inv(voxel_to_tkr), coordinates)
        if np.any(voxels < -1) or np.any(voxels > np.asarray(reference.shape)):
            raise ValueError("Tessellation does not preserve FreeSurfer tkregister-RAS coordinates")
        stage.metrics[hemi] = {"method": "FreeSurfer label-specific mri_pretess followed by mri_tessellate",
                               "filled_value": value,
                               "initializer": str(initializer),
                               "pretess": {"source": str(stage.path("filled_mgz")),
                                           "intensity_reference": str(stage.path("norm_mgz")),
                                           "changed_voxels": int(changed.sum()),
                                           "label_transitions": [{"from": int(pair[0]), "to": int(pair[1]),
                                                                  "voxels": int(count)}
                                                                 for pair, count in zip(transitions, counts)],
                                           "other_label_to_target_voxels": int(np.count_nonzero(
                                               changed & (values == other_value) & (prepared == value))),
                                           "source_filled_preserved": True},
                               "vertices": len(coordinates), "faces": len(faces), "surface_space": "freesurfer-tkr",
                               "topology_correction": False, "levelset_inference": False,
                               "preserved_native_volume_geometry": True}
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def quality_warning(stage, location, check, details):
    stage.metrics.setdefault("quality_warnings", {}).setdefault(location, {})[check] = details
    print(f"[report_only] {stage.subject} {stage.node['node_id']} {location} {check}: "
          + json.dumps(details, ensure_ascii=False), file=sys.stderr, flush=True)


def surface_repair_checks(stage, path):
    import nibabel as nib
    import numpy as np
    from spherical_geometry import check_indices

    xyz, faces = nib.freesurfer.read_geometry(checked_file(path))
    check_indices(faces, len(xyz))
    if not np.isfinite(xyz).all():
        raise ValueError("Nonfinite surface coordinates: " + str(path))
    errors = []
    def measure(identity, arguments, patterns):
        command = stage.command(identity)
        code = stage.run_process(identity, [*command['argv_prefix'], *arguments],
                                 command.get('environment', {}), stage.subject_dir, check=False)
        record = stage.logs[-1]
        output = "\n".join((stage.root / record[stream]).read_text(errors='replace')
                           for stream in ('stdout', 'stderr'))
        values = [re.findall(pattern, output, re.M) for pattern in patterns]
        if code or any(len(matches) != 1 for matches in values):
            errors.append(dict(command=identity, exit_code=code, stdout=record['stdout'], stderr=record['stderr']))
        return [int(matches[0]) if not code and len(matches) == 1 else None for matches in values]
    euler, defects = measure('freesurfer:mris_euler_number', [path],
                             [r'^euler # =.* =\s*(-?\d+)\s*-->', r'^total defect index =\s*(-?\d+)\s*$'])
    intersections, = measure('freesurfer:mris_info', ['--intersections', path],
                             [r'^Found (\d+) intersections\s*$'])
    return dict(euler_number=euler, total_defect_index=defects, self_intersecting_vertices=intersections,
                **({'unavailable_measurements': errors} if errors else {}))


def check_topology_metrics(stage, hemi, name, measurements):
    if (measurements["euler_number"] != 2 or measurements["total_defect_index"] != 0
            or measurements["self_intersecting_vertices"] != 0):
        quality_warning(stage, hemi, name, measurements)


def surface_repair_geometry(raw, topofix, repaired):
    import nibabel as nib
    import numpy as np

    raw_coordinates, raw_faces = nib.freesurfer.read_geometry(raw)
    topofix_coordinates, topofix_faces = nib.freesurfer.read_geometry(topofix)
    coordinates, faces = nib.freesurfer.read_geometry(repaired)
    if coordinates.shape != topofix_coordinates.shape or not np.array_equal(faces, topofix_faces):
        raise ValueError("Intersection repair changed topofix vertex or face identity")
    if not all(np.isfinite(values).all() for values in (raw_coordinates, topofix_coordinates, coordinates)):
        raise ValueError("Surface repair contains nonfinite coordinates")
    displacement = np.linalg.norm(coordinates - topofix_coordinates, axis=1)
    return {"raw_vertices": len(raw_coordinates), "topofix_vertices": len(topofix_coordinates),
            "repaired_vertices": len(coordinates), "raw_faces": len(raw_faces), "repaired_faces": len(faces),
            "topofix_changed_indexing": (raw_coordinates.shape != topofix_coordinates.shape
                                         or not np.array_equal(raw_faces, topofix_faces)),
            "intersection_repair_displacement_mm_p50_p95_max": np.percentile(displacement, [50, 95, 100]).tolist(),
            "raw_surface": str(raw), "topofix_surface": str(topofix), "repaired_surface": str(repaired)}


def surface_topology_geometry(raw, topofix):
    import nibabel as nib
    import numpy as np

    raw_coordinates, raw_faces = nib.freesurfer.read_geometry(raw)
    coordinates, faces = nib.freesurfer.read_geometry(topofix)
    if not all(np.isfinite(values).all() for values in (raw_coordinates, coordinates)):
        raise ValueError("Surface topology fixing contains nonfinite coordinates")
    return {"raw_vertices": len(raw_coordinates), "topofix_vertices": len(coordinates),
            "raw_faces": len(raw_faces), "topofix_faces": len(faces),
            "topofix_changed_indexing": (raw_coordinates.shape != coordinates.shape
                                         or not np.array_equal(raw_faces, faces)),
            "raw_surface": str(raw), "topofix_surface": str(topofix)}


def fix_initialized_topology(stage, hemi):
    from topology_backend import quick_sphere

    parameters = stage.node["parameters"]
    smoothing_iterations = parameters.get("topology_smoothing_iterations", 0)
    if type(smoothing_iterations) is not int or smoothing_iterations < 0:
        raise ValueError("topology_smoothing_iterations must be a nonnegative integer")
    repair_seed = parameters.get("topology_repair_seed", parameters["topology_seed"])
    if type(repair_seed) is not int or repair_seed < 0:
        raise ValueError("topology_repair_seed must be a nonnegative integer")
    for volume in ("brain_mgz", "wm_mgz", "orig_mgz"):
        checked_file(stage.path(volume))
    raw = stage.path("orig_nofix_surf", hemi)
    smoothwm = stage.hemi("smoothwm_nofix_surf", hemi)
    inflated = stage.hemi("inflated_nofix_surf", hemi)
    qsphere = stage.hemi("qsphere_nofix_surf", hemi)
    topofix = stage.hemi("orig_topofix_surf", hemi)
    raw_checks = surface_repair_checks(stage, raw)
    stage.run("freesurfer:mris_smooth", ["-nw", "-seed", parameters["topology_seed"], raw, smoothwm])
    stage.run("freesurfer:mris_inflate", ["-no-save-sulc", smoothwm, inflated])
    sphere = quick_sphere(stage, hemi, ["-q", "-p", parameters["qsphere_passes"],
              "-a", parameters["qsphere_averages"], "-seed", parameters["topology_seed"],
              "-threads", stage.spec["threads"]], inflated, qsphere)
    # Keep the native GA surface. Intersection cleanup has its own downstream node.
    stage.run("freesurfer:mris_fix_topology", ["-mgz", "-ga", "-seed", repair_seed,
              "-threads", stage.spec["threads"], "-s", smoothing_iterations,
              "-orig", "orig.nofix", "-sphere", "qsphere.nofix",
              "-inflated", "inflated.nofix", "-out", "orig.topofix", stage.subject, hemi])
    topofix_checks = surface_repair_checks(stage, topofix)
    check_topology_metrics(stage, hemi, "topofix_before_intersection_cleanup", topofix_checks)
    stage.metrics[hemi] = {"raw": raw_checks, "topofix": topofix_checks, "quick_sphere": sphere,
                           "topology_smoothing_iterations": smoothing_iterations,
                           "topology_repair_seed": repair_seed,
                           **surface_topology_geometry(raw, topofix)}


def fix_topology(stage):
    stage.each_hemisphere(fix_initialized_topology, uses_gpu=stage.spec.get("topology_backend", "gpu") != "cpu")
    return hemi_outputs(stage)


def repair_initialized_intersections(stage, hemi):
    parameters = stage.node["parameters"]
    local_recovery = parameters.get("local_fairing_recovery", False)
    if type(local_recovery) is not bool:
        raise ValueError("local_fairing_recovery must be a boolean")
    raw = stage.path("orig_nofix_surf", hemi)
    topofix = stage.path("orig_topofix_surf", hemi)
    repaired = stage.hemi("orig_surf", hemi)
    raw_checks = surface_repair_checks(stage, raw)
    topofix_checks = surface_repair_checks(stage, topofix)
    stage.run("freesurfer:mris_remove_intersection", [topofix, repaired])
    repaired_checks = surface_repair_checks(stage, repaired)
    recovery = None
    native_checks = dict(repaired_checks)
    if (local_recovery and repaired_checks["euler_number"] == parameters["required_euler_number"]
            and repaired_checks["total_defect_index"] == parameters["required_total_defect_index"]
            and repaired_checks["self_intersecting_vertices"] is not None
            and repaired_checks["self_intersecting_vertices"] != parameters["required_self_intersecting_vertices"]):
        from intersection_recovery import recover_intersections, RecoveryExhausted
        try:
            repaired_checks, recovery = recover_intersections(stage, hemi, repaired, surface_repair_checks)
        except (RecoveryExhausted, RuntimeError) as error:
            # Optional fairing operates on separate candidates. Revalidate the retained native file below.
            recovery = dict(status='retained_native_cleanup', reason=str(error))
    check_topology_metrics(stage, hemi, "after_intersection_cleanup", repaired_checks)
    stage.metrics[hemi] = {"raw": raw_checks, "topofix": topofix_checks, "repaired": repaired_checks,
                           "self_intersection_gate": "report_only",
                           "self_intersection_status": "measured" if repaired_checks["self_intersecting_vertices"] is not None else "unavailable",
                           **surface_repair_geometry(raw, topofix, repaired)}
    if recovery is not None:
        stage.metrics[hemi].update(native_before_recovery=native_checks, local_fairing_recovery=recovery)
    stage.metrics[hemi]['face_winding'] = orient_initialized_surface(repaired)
    winding = stage.metrics[hemi]['face_winding']
    if 'quality_warning' in winding:
        quality_warning(stage, hemi, 'face_winding', winding)
    copy_file(repaired, stage.hemi("orig_premesh_surf", hemi))


def orient_initialized_surface(path):
    """Correct sparse GA face winding once, before all matched cortical meshes."""
    import nibabel as nib
    import numpy as np
    from spherical_geometry import reported_winding

    xyz, faces, metadata = nib.freesurfer.read_geometry(path, read_metadata=True)
    oriented, report = reported_winding(faces, len(xyz))
    if report['reversed_face_rows']:
        backup = path.with_name(path.name + '.before-winding')
        if backup.exists():
            raise ValueError('Refusing to replace earlier winding evidence')
        # A NiBabel metadata rewrite rounds FreeSurfer's CRAS/direction strings.
        # Only the triangle indices change here. Keep every other native byte.
        import struct
        copy_file(path, backup)
        with path.open('r+b') as stream:
            if stream.read(3) != bytes.fromhex('fffffe'):
                raise ValueError('Winding repair requires a native triangle file')
            stream.readline()
            stream.readline()
            if stream.read(8) != struct.pack('>ii', len(xyz), len(faces)):
                raise ValueError('Unexpected FreeSurfer vertex/face counts during winding repair')
            stream.seek(xyz.size * 4, os.SEEK_CUR)
            offset = stream.tell()
            if stream.read(faces.size * 4) != faces.astype('>i4').tobytes():
                raise ValueError('Unexpected FreeSurfer triangle data during winding repair')
            stream.seek(offset)
            stream.write(oriented.astype('>i4').tobytes())
        readback, read_faces, read_metadata = nib.freesurfer.read_geometry(path, read_metadata=True)
        if (not np.array_equal(readback, xyz) or not np.array_equal(read_faces, oriented)
                or metadata.keys() != read_metadata.keys()
                or any(not np.array_equal(value, read_metadata[key]) for key, value in metadata.items())):
            raise ValueError('Winding repair changed coordinates, geometry or triangle correspondence')
        report['original_surface'] = str(backup)
    return report


def remove_intersections(stage):
    stage.each_hemisphere(repair_initialized_intersections)
    return hemi_outputs(stage)


def autodetect(stage):
    def process_hemisphere(stage, hemi):
        stage.run("freesurfer:mris_autodet_gwstats", ["--o", stage.hemi("autodet_gwstats", hemi),
                  "--i", stage.path("brain_finalsurfs_mgz"), "--wm", stage.path("wm_mgz"),
                  "--" + hemi + "-surf", stage.path("orig_premesh_surf", hemi)])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def placed_surface_geometry(path, reference):
    import nibabel as nib
    import numpy as np

    vertices, faces, metadata = nib.freesurfer.read_geometry(path, read_metadata=True)
    if (vertices.shape != reference[0].shape or not len(vertices)
            or not np.isfinite(vertices).all() or not np.isfinite(reference[0]).all()
            or not np.array_equal(faces, reference[1])):
        raise ValueError("Surface placement changed vertex/face correspondence or finite coordinates: " + str(path))
    for key in ("volume", "voxelsize", "xras", "yras", "zras", "cras"):
        if (key not in metadata or key not in reference[2] or not np.isfinite(metadata[key]).all()
                or not np.array_equal(metadata[key], reference[2][key])):
            raise ValueError("Surface placement changed reference volume geometry: " + str(path) + " " + key)
    return vertices, faces, metadata


def check_placed_surface(stage, hemi, port, reference):
    path = stage.hemi(port, hemi)
    placed_surface_geometry(path, reference)
    checks = surface_repair_checks(stage, path)
    stage.metrics.setdefault("placement_checks", {}).setdefault(hemi, {})[port] = {
        **checks, "surface": str(path), "self_intersection_gate": "report_only",
        "vertex_face_correspondence": True, "reference_volume_geometry_preserved": True}
    check_topology_metrics(stage, hemi, port, checks)


def record_medial_wall(stage, hemi, port, decision):
    wall = decision.get('medial_wall', {})
    if 'quality_warning' in wall:
        quality_warning(stage, hemi, port + '_medial_wall', wall)
    for key in ('input_constraint', 'output_constraint'):
        if 'quality_warning' in wall.get(key, {}):
            quality_warning(stage, hemi, port + '_' + key, wall[key])



def place_surface(stage):
    import nibabel as nib
    from t2_pial import initial_t1_pial_profile, prepare_t2, refinement_manifest, t2_refinement_profile
    from t2_pial_backend import refine
    from pial_boundary import FIXED

    pial = stage.node["node_id"] == "anat_pial_surface"
    nsmooth = stage.node["parameters"].get("nsmooth", 0 if pial else 5)
    if type(nsmooth) is not int or nsmooth < 0:
        raise ValueError("nsmooth must be a nonnegative integer")
    if pial:
        t2_volume, refinement, refinement_files = prepare_t2(stage)
        refinement["initial_t1_pial"] = {}
        refinement["placement_backend"] = {}
        t2_overrides, refinement["t2_refinement_parameters"] = t2_refinement_profile(
            stage.node["parameters"], t2_volume is not None)
    def process_hemisphere(stage, hemi):
        source = stage.path("white_surf" if pial else "orig_surf", hemi)
        source_bytes = source.read_bytes()
        reference = nib.freesurfer.read_geometry(source, read_metadata=True)
        target_port = "pial_t1_surf" if pial else "white_preaparc_surf"
        target = stage.hemi(target_port, hemi)
        args = ["--adgws-in", stage.path("autodet_gwstats", hemi), "--wm", stage.path("wm_mgz"),
                "--invol", stage.path("brain_finalsurfs_mgz"), "--" + hemi, "--pial" if pial else "--white",
                "--i", stage.path("white_surf" if pial else "orig_surf", hemi), "--o", target,
                "--seg", stage.path("aseg_presurf_mgz"), "--nsmooth", str(nsmooth), "--threads", stage.spec["threads"]]
        if pial:
            args += ["--aparc", stage.path("aparc_annot", hemi), "--rip-label", stage.path("cortex_hipamyg_label", hemi),
                     "--pin-medial-wall", stage.path("cortex_label", hemi), "--white-surf", stage.path("white_surf", hemi),
                     "--repulse-surf", stage.path("white_surf", hemi)]
        initial_args = list(args)
        if pial:
            overrides, profile = initial_t1_pial_profile(stage.node["parameters"], t2_volume is not None,
                                                        stage.path("autodet_gwstats", hemi))
            initial_args.extend(overrides)
            refinement["initial_t1_pial"][hemi] = profile
        t1_decision = refine(stage, initial_args, hemi, 't1-pial' if pial else 't1-white')
        t1_decision["nsmooth"] = nsmooth
        stage.metrics.setdefault('t1_placement', {})[hemi] = t1_decision
        if source.read_bytes() != source_bytes:
            raise ValueError("Surface placement modified the input reference surface: " + str(source))
        record_medial_wall(stage, hemi, target_port, t1_decision)
        check_placed_surface(stage, hemi, target_port, reference)
        if pial:
            refinement['initial_t1_pial'][hemi]['placement_backend'] = t1_decision
        final = stage.hemi("pial_surf" if pial else "white_surf", hemi)
        if pial and t2_volume is not None:
            refined_args = list(args)
            refined_args[refined_args.index("--i") + 1] = target
            refined_args[refined_args.index("--o") + 1] = final
            t2_arguments = [*refined_args, "--mmvol", t2_volume, "T2", *t2_overrides]
            if t1_decision.get('medial_wall', {}).get('strategy') == FIXED:
                refinement["placement_backend"][hemi] = refine(stage, t2_arguments, hemi, medial_wall_strategy=FIXED)
            else:
                refinement["placement_backend"][hemi] = refine(stage, t2_arguments, hemi)
            refinement["placement_backend"][hemi]["nsmooth"] = nsmooth
            if source.read_bytes() != source_bytes:
                raise ValueError("Pial placement modified the white reference surface")
            record_medial_wall(stage, hemi, "pial_surf", refinement["placement_backend"][hemi])
            check_placed_surface(stage, hemi, "pial_surf", reference)
        else:
            copy_file(target, final)
        target = final
        stage.run("freesurfer:mris_place_surface", ["--curv-map", target, "2", "10",
                  stage.hemi("curv_pial_surf" if pial else "curv_surf", hemi)])
        stage.run("freesurfer:mris_place_surface", ["--area-map", target,
                  stage.hemi("area_pial_surf" if pial else "area_surf", hemi)])
        if pial:
            stage.run("freesurfer:mris_place_surface", ["--thickness", stage.path("white_surf", hemi), target,
                      "20", "5", stage.hemi("thickness_surf", hemi)])
    uses_gpu = stage.spec.get("t1_placement_backend", "gpu") != "cpu" or (
        pial and t2_volume is not None and stage.spec.get("t2_pial_backend", "gpu") != "cpu")
    stage.each_hemisphere(process_hemisphere, minimum_threads=2 if uses_gpu else 1,
                          maximum_threads=2, uses_gpu=uses_gpu)
    if pial:
        outputs = {port["name"]: {hemi: stage.hemi(port["name"], hemi) for hemi in HEMISPHERES}
                   for port in stage.node["outputs"] if port["cardinality"] == "both_hemispheres"}
        refinement["post_placement_checks"] = stage.metrics["placement_checks"]
        outputs["pial_refinement_manifest"] = refinement_manifest(stage, refinement, refinement_files)
        stage.metrics["pial_refinement"] = refinement
        return outputs
    return hemi_outputs(stage)


def cortex(stage):
    keep = stage.node["parameters"]["label_cortex_keep_hipamyg"]
    port = stage.node["outputs"][0]["name"]
    def process_hemisphere(stage, hemi):
        stage.run("freesurfer:mri_label2label-cortex", ["--label-cortex", stage.path("white_preaparc_surf", hemi),
                  stage.path("aseg_presurf_mgz"), keep, stage.hemi(port, hemi)])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def hyporelabel(stage):
    stage.run("freesurfer:mri_relabel_hypointensities", [stage.path("aseg_presurf_mgz"),
              stage.subject_dir / "surf", stage.volume("aseg_presurf_hypos_mgz")])
    return volume_outputs(stage)


def smooth(stage):
    def process_hemisphere(stage, hemi):
        stage.run("freesurfer:mris_smooth", ["-nw", stage.path("white_preaparc_surf", hemi), stage.hemi("smoothwm_surf", hemi)])
        stage.run("freesurfer:mris_inflate", [stage.hemi("smoothwm_surf", hemi), stage.hemi("inflated_surf", hemi)])
        stage.run("freesurfer:mris_info", [stage.hemi("inflated_surf", hemi)])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def curvature_stats(stage):
    def process_hemisphere(stage, hemi):
        stage.run("freesurfer:mris_curvature_stats", ["-o", stage.hemi("curv_stats", hemi), stage.subject,
                  hemi, stage.path("curv_surf", hemi)])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def sphere(stage):
    from spherical_geometry import prepare_file

    def process_hemisphere(stage, hemi):
        raw = stage.hemi("sphere_surf", hemi).with_name(hemi + '.sphere.raw')
        stage.run("freesurfer:mris_sphere", ["-seed", "1234", "-threads", stage.spec["threads"],
                  stage.path("inflated_surf", hemi), raw])
        stage.metrics[hemi] = prepare_file(raw, stage.hemi("sphere_surf", hemi),
                                         topology_reference=stage.path("inflated_surf", hemi))
        record_sphere_quality(stage, hemi, 'sphere', stage.metrics[hemi])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def record_sphere_quality(stage, hemi, name, report):
    if 'quality_warning' in report or 'winding_warning' in report:
        quality_warning(stage, hemi, name, {key: report[key] for key in
                        ('quality_warning', 'winding_warning', 'after') if key in report})


def export_registered_sphere(stage, hemi, model_output):
    import numpy as np
    from nibabel.freesurfer.io import read_geometry, write_geometry

    source = stage.path("sphere_surf", hemi)
    source_coordinates, source_faces, volume_info = read_geometry(source, read_metadata=True)
    coordinates, faces = read_geometry(model_output)
    required = {"head", "valid", "filename", "volume", "voxelsize", "xras", "yras", "zras", "cras"}
    if not required.issubset(volume_info) or str(volume_info["valid"]).split()[:1] != ["1"]:
        raise ValueError("Native sphere has missing or invalid volume geometry: " + str(source))
    for name in ("volume", "voxelsize", "xras", "yras", "zras", "cras"):
        values = np.asarray(volume_info[name])
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Invalid native sphere volume geometry field: " + name)
        if name in ("volume", "voxelsize") and values.min() <= 0:
            raise ValueError("Native sphere volume geometry must have positive " + name)
    if coordinates.shape != source_coordinates.shape or not np.array_equal(faces, source_faces):
        raise ValueError("Registered sphere changed native vertex or face identity")
    if not np.isfinite(coordinates).all():
        raise ValueError("Registered sphere has nonfinite coordinates")
    from spherical_geometry import save_checked
    destination = stage.hemi("sphere_reg_surf", hemi)
    final_geometry = save_checked(destination, coordinates, faces, volume_info=volume_info)
    stage.metrics[hemi] = {"native_sphere": str(source), "model_sphere": str(model_output),
                           "preserved_native_volume_geometry": True, "spherical_orientation": final_geometry}
    return destination


def register_sphere(stage):
    # Selective helper routing must also select the matching SUGAR wrapper;
    # model weights and the configured scientific interpreter remain bound.
    commands = dict(stage.runtime['commands'])
    for name in ('sugar:rigid', 'sugar:nonlinear'):
        command = commands[name]
        argv = list(command['argv_prefix'])
        indices = [i for i, arg in enumerate(argv) if Path(arg).name == 'run_registration.py']
        if len(indices) != 1:
            raise ValueError('Cannot bind SUGAR command to selected helper revision')
        argv[indices[0]] = str(Path(__file__).resolve().with_name('sugar_registration.py'))
        commands[name] = dict(command, argv_prefix=argv)
    stage.runtime = dict(stage.runtime, commands=commands)

    def process_hemisphere(stage, hemi):
        letter = "L" if hemi == "lh" else "R"
        command = "sugar:rigid"
        template = stage.resource("fsaverage6-" + letter + "-sphere")
        sulc = stage.resource_file(command, hemi + r"\.sulc")
        metadata = stage.root / (hemi + "-template.json")
        write_json(metadata, {"type": "SphericalTemplateMetadata", "template": "fsaverage6", "hemisphere": hemi,
                   "vertex_count": 40962, "sphere": str(template), "sulc": str(sulc)})
        common = ["--sulc", stage.path("sulc_surf", hemi), "--template-sphere", template,
                  "--template-sulc", sulc, "--template-metadata", metadata, "--hemisphere", hemi, "--device", stage.spec["device"]]
        rigid = stage.root / (hemi + "-sugar-rigid")
        stage.run(command, ["rigid", "--sphere", stage.path("sphere_surf", hemi), *common,
                  "--checkpoint", stage.resource_file(command, hemi + r"_Rigid_904_fsaverage6\.model"), "--work-dir", rigid])
        nonlinear = stage.root / (hemi + "-sugar-nonlinear")
        args = ["nonlinear", "--sphere", rigid / (hemi + ".rigid.sphere"), *common, "--work-dir", nonlinear]
        for level in (3, 4, 5, 6):
            args += [f"--checkpoint-fsaverage{level}",
                     stage.resource_file("sugar:nonlinear", hemi + rf"_NoRigid_904_fsaverage{level}\.model")]
        stage.run("sugar:nonlinear", args)
        export_registered_sphere(stage, hemi, nonlinear / (hemi + ".sphere.reg"))
        for directory in (rigid, nonlinear):
            provenance = read_json(directory / 'provenance.json')
            for field in ('input_geometry', 'template_geometry', 'output_geometry'):
                if field in provenance:
                    record_sphere_quality(stage, hemi, directory.name + '_' + field, provenance[field])
    stage.each_hemisphere(process_hemisphere, uses_gpu=stage.spec.get("device", "cpu") != "cpu")
    return hemi_outputs(stage)


def jacobian(stage):
    def process_hemisphere(stage, hemi):
        stage.run("freesurfer:mris_jacobian", [stage.path("white_preaparc_surf", hemi),
                  stage.path("sphere_reg_surf", hemi), stage.hemi("jacobian_white", hemi)])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def cortical_parcellation(stage):
    command = "freesurfer:mris_ca_label"
    def process_hemisphere(stage, hemi):
        atlas = stage.resource_file(command, hemi + r"\.curvature\.buckner40\.filled\.desikan_killiany\.2010-03-25\.gcs")
        stage.run(command, ["-sdir", stage.subjects_dir, "-orig", "smoothwm",
                  "-l", stage.hemi("cortex_label", hemi), "-aseg", stage.volume("aseg_presurf_mgz"),
                  "-seed", "1234", stage.subject, hemi, stage.hemi("sphere_reg_surf", hemi),
                  atlas, stage.hemi("aparc_annot", hemi)])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def percent_contrast(stage):
    def process_hemisphere(stage, hemi):
        stage.run("freesurfer:pctsurfcon", ["--s", stage.subject, "--" + hemi + "-only", "--nolog"])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def parcellation_stats(stage):
    def process_hemisphere(stage, hemi):
        for surface, port in (("white", "aparc_stats"), ("pial", "aparc_pial_stats")):
            stage.run("freesurfer:mris_anatomical_stats", ["-th3", "-mgz", "-cortex", stage.hemi("cortex_label", hemi),
                      "-a", stage.hemi("aparc_annot", hemi), "-f", stage.hemi(port, hemi),
                      stage.subject, hemi, surface])
    stage.each_hemisphere(process_hemisphere)
    return hemi_outputs(stage)


def make_ribbon(stage):
    parallel = int(stage.spec['threads']) >= 4 and stage.node.get('parameters', {}).get('ribbon_parallel', True)
    stage.run("freesurfer:mris_volmask", ["--sd", stage.subjects_dir, "--aseg_name", "aseg.presurf",
              "--label_left_white", "2", "--label_left_ribbon", "3", "--label_right_white", "41",
              "--label_right_ribbon", "42", "--save_ribbon", *(['--parallel'] if parallel else []), stage.subject])


def ribbon(stage):
    started = time.perf_counter()
    make_ribbon(stage)
    stage.metrics['ribbon_execution'] = dict(rasterization_seconds=time.perf_counter() - started,
        allocated_threads=int(stage.spec['threads']),
        rasterization_parallel=int(stage.spec['threads']) >= 4 and stage.node.get('parameters', {}).get('ribbon_parallel', True))
    return volume_outputs(stage)


def cortical_pair(stage):
    """Measure native surfaces/ribbon on an independent reporting branch."""
    started = time.perf_counter()
    measurements, jobs = {}, []
    for hemi in HEMISPHERES:
        measurement = stage.root / "measurements" / (hemi + "-cortical-pair.json")
        measurement.parent.mkdir(parents=True, exist_ok=True)
        measurements[hemi] = measurement
        jobs.append(("neuroqc:measure-cortical-pair", ["--white", stage.path("white_surf", hemi),
                  "--pial", stage.path("pial_surf", hemi), "--thickness", stage.path("thickness_surf", hemi),
                  "--cortex-label", stage.path("cortex_label", hemi), "--ribbon", stage.path("ribbon_mgz"),
                  "--aseg", stage.path("aseg_presurf_mgz"), "--hemi", hemi, "--output", measurement]))
    measurement_started = time.perf_counter()
    workers = min(2, int(stage.spec['threads'])) if stage.node.get('parameters', {}).get('metrics_parallel', True) else 1
    first_command = len(stage.logs)
    stage.run_concurrent(jobs, max_workers=workers, check=False)
    for (hemi, measurement), command in zip(measurements.items(), stage.logs[first_command:]):
        if command['exit_code'] or not measurement.is_file():
            unavailable = dict(status='unknown', hemi=hemi, missing_checks=['Cortical pair measurement unavailable'],
                               exit_code=command['exit_code'], stderr=command['stderr'])
            # Preserve any partial tool output alongside the diagnostic.
            if measurement.exists():
                measurement.rename(measurement.with_suffix('.partial.json'))
            write_json(measurement, unavailable)
            quality_warning(stage, hemi, 'cortical_pair_measurement', unavailable)
        stage.metrics[hemi] = read_json(measurement)
        if stage.metrics[hemi].get('status') == 'fail':
            quality_warning(stage, hemi, 'cortical_pair', {'status': 'fail', 'measurement': str(measurement.name)})
    stage.metrics['cortical_pair_execution'] = dict(
        measurement_wall_seconds=time.perf_counter() - measurement_started,
        total_seconds=time.perf_counter() - started, measurement_workers=workers,
        allocated_threads=int(stage.spec['threads']))
    return {'cortical_pair_measurements': measurements}


def surf2vol(stage):
    aparc = stage.node["node_id"] == "anat_aparc2aseg"
    port = "aparc_aseg_mgz" if aparc else "aseg_mgz"
    command = "freesurfer:mri_surf2volseg-label" if aparc else "freesurfer:mri_surf2volseg-fix"
    args = ["--i", stage.path("aseg_mgz" if aparc else "aseg_presurf_hypos_mgz"), "--o", stage.volume(port),
            "--threads", stage.spec["threads"]]
    for hemi in HEMISPHERES:
        args += ["--" + hemi + "-cortex-mask", stage.path("cortex_label", hemi),
                 "--" + hemi + "-white", stage.path("white_surf", hemi), "--" + hemi + "-pial", stage.path("pial_surf", hemi)]
    if aparc:
        args += ["--label-cortex", "--lh-annot", stage.path("aparc_annot", "lh"), "1000",
                 "--rh-annot", stage.path("aparc_annot", "rh"), "2000"]
    else:
        args += ["--fix-presurf-with-ribbon", stage.path("ribbon_mgz")]
    stage.run(command, args)
    if not aparc:
        stage.run("freesurfer:mri_brainvol_stats", ["--subject", stage.subject])
        return {"aseg_mgz": stage.volume(port), "brainvol_stats": stage.output("stats/brainvol.stats")}
    return volume_outputs(stage)


def correspondence(stage):
    import nibabel as nib
    import numpy as np

    def process_hemisphere(stage, hemi):
        letter = "L" if hemi == "lh" else "R"
        white, faces = nib.freesurfer.read_geometry(str(stage.path("white_surf_native", hemi)))
        pial, pial_faces = nib.freesurfer.read_geometry(str(stage.path("pial_surf_native", hemi)))
        sphere_xyz, sphere_faces = nib.freesurfer.read_geometry(str(stage.path("sphere_reg_native", hemi)))
        sulc = nib.freesurfer.read_morph_data(str(stage.path("sulc_native", hemi)))
        target = stage.resource("fsaverage6-" + letter + "-sphere")
        target_xyz, target_faces = nib.freesurfer.read_geometry(str(target))
        if (white.shape != pial.shape or white.shape != sphere_xyz.shape or len(sulc) != len(white)
                or not np.array_equal(faces, pial_faces) or not np.array_equal(faces, sphere_faces)
                or len(target_xyz) != 40962 or not all(np.isfinite(values).all() for values in (white, pial, sphere_xyz, sulc))):
            raise ValueError("Invalid native vertex identity or fsaverage6 density for " + hemi)
        if (np.linalg.norm(sphere_xyz, axis=1) == 0).any():
            raise ValueError("Registered sphere contains zero-radius vertices")
        return {"hemi": hemi, "source_vertices": len(white), "target_vertices": len(target_xyz),
                        "source_faces": len(faces), "target_faces": len(target_faces),
                        "target_sphere": str(target)}
    records = stage.each_hemisphere(process_hemisphere).values()
    target_volume = str(stage.resource("fsaverage6-orig"))
    output = {}
    for record in records:
        hemi = record["hemi"]
        path = stage.root / (hemi + ".sphere_correspondence.json")
        write_json(path, {"subject": stage.subject, "template": "fsaverage6", "hemispheres": [record],
                         "target_volume": target_volume})
        output[hemi] = path
    return {"sphere_correspondence": output}


def read_sphere_correspondence(stage, hemi):
    validation = read_json(stage.path("sphere_correspondence", hemi))
    records = validation.get("hemispheres", [])
    if (validation.get("subject") != stage.subject or validation.get("template") != "fsaverage6"
            or len(records) != 1 or records[0].get("hemi") != hemi):
        raise ValueError("Sphere correspondence identity mismatch for " + hemi)
    record = records[0]
    letter = "L" if hemi == "lh" else "R"
    target = stage.resource("fsaverage6-" + letter + "-sphere")
    if (record.get("target_sphere") != str(target)
            or record.get("target_vertices") != 40962
            or validation.get("target_volume") != str(stage.resource("fsaverage6-orig"))):
        raise ValueError("Sphere correspondence target mismatch for " + hemi)
    return validation


def resample(stage):
    import nibabel as nib
    import numpy as np

    surface = "pial" if stage.node["node_id"].startswith("resample_pial") else "white"
    for hemi in HEMISPHERES:
        read_sphere_correspondence(stage, hemi)
    target_subject = stage.subjects_dir / "fsaverage6"
    for hemi, letter in (("lh", "L"), ("rh", "R")):
        copy_file(stage.resource("fsaverage6-" + letter + "-sphere"), target_subject / "surf" / (hemi + ".sphere.reg"))
    target_volume = copy_file(stage.resource("fsaverage6-orig"), target_subject / "mri/orig.mgz")
    def process_hemisphere(stage, hemi):
        path = stage.output("surf/" + hemi + "." + surface + ".fsaverage6")
        stage.run("freesurfer:mri_surf2surf", ["--sd", stage.subjects_dir, "--srcsubject", stage.subject,
                  "--trgsubject", "fsaverage6", "--hemi", hemi, "--sval-tal-xyz", surface,
                  "--tval-xyz", target_volume, "--tval", path])
        checks = surface_repair_checks(stage, path)
        stage.metrics[hemi] = checks
        check_topology_metrics(stage, hemi, surface + '_fsaverage6', checks)
        xyz, faces, geometry = nib.freesurfer.read_geometry(str(path), read_metadata=True)
        letter = "L" if hemi == "lh" else "R"
        _, target_faces = nib.freesurfer.read_geometry(str(stage.resource("fsaverage6-" + letter + "-sphere")))
        target_image = nib.load(str(target_volume))
        shape, spacing = np.asarray(target_image.shape[:3]), np.asarray(target_image.header.get_zooms()[:3])
        axes = target_image.affine[:3, :3] / spacing
        center = nib.affines.apply_affine(target_image.affine, shape / 2)
        expected_geometry = {"volume": shape, "voxelsize": spacing, "xras": axes[:, 0],
                             "yras": axes[:, 1], "zras": axes[:, 2], "cras": center}
        if (len(xyz) != 40962 or not np.isfinite(xyz).all() or not np.array_equal(faces, target_faces)
                or any(key not in geometry or not np.allclose(geometry[key], value, atol=1e-4, rtol=0)
                       for key, value in expected_geometry.items())):
            raise ValueError("Invalid fsaverage6 output geometry")
        if surface == "pial":
            reference_xyz, reference_faces = nib.freesurfer.read_geometry(str(stage.path("white_surf_fsaverage6_reference", hemi)))
            if reference_xyz.shape != xyz.shape or not np.array_equal(reference_faces, faces):
                raise ValueError("Resampled white and pial vertex identities differ")
        return path
    output = stage.each_hemisphere(process_hemisphere)
    return {surface + "_surf": output}


OPERATIONS = {
    "session_normalize_mni": normalize_session,
    "session_align_companions": align_companions,
    "anat_get_t1w_file_in_bids": t1w_selection, "anat_create_subject_orig_dir": orig_directory,
    "anat_motioncor": recon, "anat_synthseg": synthseg, "anat_prepare_synthseg_aseg": prepare_synthseg_aseg,
    "anat_orig_to_nifti": orig_to_nifti, "anat_hdbet_mask": hdbet_mask, "anat_mask_to_mgz": mask_to_mgz,
    "anat_gca_register": gca_register, "anat_tissue_bias_correct": tissue_bias_correct, "anat_talairach_and_nu": talairach,
    "anat_brainmask": brainmask, "anat_paint_cc_to_aseg": paint_cc,
    "anat_synthseg_fill": synthseg_fill,
    "anat_tessellate": tessellate,
    "anat_fix_topology": fix_topology, "anat_remove_intersection": remove_intersections,
    "anat_autodet_gwstats": autodetect, "anat_white_surface": place_surface, "anat_cortex_label": cortex,
    "anat_cortex_hipamyg_label": cortex, "anat_hyporelabel": hyporelabel, "anat_smooth_inflated": smooth,
    "anat_curv_stats": curvature_stats, "anat_sphere": sphere, "anat_sphere_register": register_sphere,
    "anat_jacobian": jacobian, "anat_cortparc_aparc": cortical_parcellation, "anat_pial_surface": place_surface,
    "anat_pctsurfcon": percent_contrast, "anat_parcstats": parcellation_stats, "anat_ribbon": ribbon, "anat_cortical_pair": cortical_pair,
    "anat_apas2aseg": surf2vol, "anat_aparc2aseg": surf2vol,
    "validate_fsaverage6_sphere_correspondence": correspondence,
    "resample_white_surface_fsaverage6": resample, "resample_pial_surface_fsaverage6": resample,
}

#!/usr/bin/env python3
"""Submit FastPrep Anat BIDS session batches to AIRs. Default action: dry run."""

import argparse
import csv
import uuid
import contextlib
import copy
import fcntl
import filecmp
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


SCRIPT = Path(__file__).resolve()
WORKFLOW = SCRIPT.parent
UNIT = re.compile(r"sub-[A-Za-z0-9]+(?:/ses-[A-Za-z0-9]+)?")
INFANT_DATASETS = {"ds004776", "ds006169", "PT023_Infant_Development_Brain"}


def dataset_exclusion(dataset):
    parts = Path(dataset).parts
    if any(part.partition("_")[0] == "PT002" for part in parts):
        return "PT002 outpatient data temporarily excluded"
    if not INFANT_DATASETS.isdisjoint(parts):
        return "infant dataset excluded"
    return None


def datasets(root, selected):
    """Recognize raw dataset roots, including nested dsNNNNNN collections."""
    def visit(folder):
        if dataset_exclusion(folder) or dataset_exclusion(folder.resolve()):
            return
        if (folder / "participants.tsv").is_file() or (folder / "dataset_description.json").is_file():
            yield folder
            return
        entries = sorted(folder.iterdir())
        if any(p.name.startswith("sub-") and p.is_dir() for p in entries):
            yield folder
            return
        for child in entries:
            if re.fullmatch(r"PT\d+_.+|ds\d+", child.name) and child.is_dir():
                yield from visit(child)

    if selected:
        scope = (root / selected).resolve()
        if not scope.is_relative_to(root) or not scope.is_dir():
            raise ValueError(f"Unknown dataset selection: {selected}")
        reason = dataset_exclusion(selected) or dataset_exclusion(scope)
        if reason:
            print(f"SKIP\t{selected}\t{reason}", flush=True)
            return []
    else:
        scope = root
    if scope == root:
        # Collection-wide participants.tsv may coexist with the per-dataset tables.
        children = (p for p in root.iterdir() if re.fullmatch(r"PT\d+_.+|ds\d+", p.name) and p.is_dir())
        found = sorted(ds for child in children for ds in visit(child))
    else:
        found = sorted(visit(scope))
    if not found:
        raise ValueError(f"No datasets matched under {root} (selection: {selected!r})")
    return found


def subject_units(subject):
    """Return candidate sessions and reasons for omissions; no image data are read."""
    children = sorted(subject.iterdir())
    parents = [p for p in children if p.name.startswith("ses-") and p.is_dir()]
    if (subject / "anat").is_dir() or not parents:
        parents.insert(0, subject)
    result = []
    for parent in parents:
        unit = parent.relative_to(subject.parent).as_posix()
        anat = parent / "anat"
        if not anat.is_dir():
            result.append((unit, "no-anat"))
            continue
        with os.scandir(anat) as entries:
            eligible = any(e.name.endswith(("_T1w.nii", "_T1w.nii.gz"))
                           and not re.search(r"(?:^|_)part-(?!mag_)[^_]+_", e.name)
                           and e.is_file() for e in entries)
        reason = "eligible" if eligible else "no-magnitude-T1w"
        if not UNIT.fullmatch(unit):
            reason = "invalid-BIDS-identity"
        result.append((unit, reason))
    return result


def job_name(relative, part):
    # The PT collection ID and optional OpenNeuro ID identify this dataset.
    short = re.sub(r"^(PT\d+)_[^/]+", r"\1", relative)
    return "fastprep-" + short.replace("/", "-") + f"-{part:04d}"


def worker_command(a, relative, part, units):
    cmd = ["python3", str(SCRIPT), "worker", "--root", str(a.root), "--dataset", relative,
           "--part", str(part), "--workflow", str(a.workflow), "--software", str(a.software),
           "--work-root", str(a.work_root), "--out-root", str(a.out_root),
           "--nextflow", a.nextflow, "--threads", str(a.threads),
           "--surface-backend", a.surface_backend, "--units", ",".join(units)]
    cmd.append("--isolate-session-errors" if a.isolate_session_errors else "--no-isolate-session-errors")
    if a.templateflow_home:
        cmd.extend(["--templateflow-home", str(a.templateflow_home)])
    command = "exec " + shlex.join(cmd)
    if a.env_script:
        command = "set -e\nsource " + shlex.quote(str(a.env_script)) + "\n" + command
    return "bash -c " + shlex.quote(command)


def make_plan(a):
    found = datasets(a.root, a.dataset)
    jobs, totals = [], Counter()
    empty = 0
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        for index, ds in enumerate(found, 1):
            rel = ds.relative_to(a.root).as_posix()
            if rel == ".":
                raise ValueError("--root is the collection directory; use --dataset to select a dataset")
            subjects = sorted(p for p in ds.glob("sub-*") if p.is_dir())
            units = []
            for scanned in pool.map(subject_units, subjects):
                for unit, reason in scanned:
                    totals[reason] += 1
                    if reason == "eligible":
                        units.append(unit)
                    elif a.list_skipped:
                        print(f"SKIP\t{rel}/{unit}\t{reason}")
            if not units:
                empty += 1
                if a.list_skipped:
                    print(f"SKIP\t{rel}\tno eligible sessions")
            for i in range(0, len(units), a.sessions_per_job):
                part = i // a.sessions_per_job + 1
                batch = units[i:i + a.sessions_per_job]
                jobs.append((job_name(rel, part), worker_command(a, rel, part, batch), len(batch)))
            if time.monotonic() - last_report >= 30:
                print(f"Scanned {index}/{len(found)} datasets; {totals['eligible']} eligible sessions",
                      file=sys.stderr, flush=True)
                last_report = time.monotonic()
    if len({name for name, _, _ in jobs}) != len(jobs):
        raise ValueError("Overlapping AIRs configuration names")
    print(f"Datasets={len(found)}, without eligible sessions={empty}, "
          f"sessions={dict(totals)}, jobs={len(jobs)}", flush=True)
    return jobs[a.start_index:None if a.limit is None else a.start_index + a.limit]


@contextlib.contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is using {path.parent}") from None
        yield


def input_view(dataset, units, job, isolate_session_errors=False):
    """Use real subject/session parents so FastPrep's own BIDS walker sees each anat link."""
    signature = job / "units.json"
    if signature.exists() and json.loads(signature.read_text()) != units:
        raise ValueError(f"Session membership changed for {job}. Choose a new work root for this partition.")
    if not signature.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=job, delete=False) as tmp:
            json.dump(units, tmp)
        os.replace(tmp.name, signature)
    view = job / "bids"
    view.mkdir(exist_ok=True)
    descriptor = dataset / "dataset_description.json"
    target = view / descriptor.name
    with tempfile.TemporaryDirectory(dir=view) as tmp:
        staged = Path(tmp) / descriptor.name
        if descriptor.is_file():
            staged.symlink_to(descriptor)
        else:
            staged.write_text(json.dumps({
                "Name": f"FastPrep input view for {dataset.name}", "BIDSVersion": "1.9.0",
                "DatasetType": "raw",
                "Description": "Generated working layout; the source has no dataset_description.json."
            }, indent=2) + "\n")
        os.replace(staged, target)
    for unit in units:
        source = dataset / unit / "anat"
        if not source.is_dir():
            if isolate_session_errors:
                continue  # prepare records the missing session input separately
            raise FileNotFoundError(source)
        link = view / unit / "anat"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(source, target_is_directory=True)
        if link.resolve() != source.resolve():
            raise ValueError(f"Unexpected input link {link}")
    return view


class PartialBatchError(Exception):
    """Only identified sessions failed; the GPU worker may take another batch."""


def finish_sessions(job, out, expected, rejected, trace):
    completed = set()
    failed = {uid: [{"stage": "prepare", "reason": message}] for uid, message in rejected.items()}
    if trace is not None:
        task_counts = Counter({"CACHED": 0})
        with trace.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                task_counts[row["status"]] += 1
                match = re.fullmatch(r"([^ ]+) \((sub-[A-Za-z0-9]+(?:_ses-[A-Za-z0-9]+)?)(?::[^()]*)?\)", row["name"])
                if not match:
                    continue
                stage, uid = match.groups()
                if uid not in expected:
                    raise RuntimeError(f"Unexpected session in Nextflow trace: {uid}")
                if row["status"] in {"FAILED", "ABORTED"}:
                    failed.setdefault(uid, []).append({"stage": stage, "exit": row["exit"], "hash": row["hash"]})
                if stage == "ASSEMBLE" and row["status"] in {"COMPLETED", "CACHED"}:
                    manifest = json.loads((out / uid / "outputs.json").read_text())
                    if manifest.get("subject") != uid or manifest.get("schema") != "fastprep.outputs/1":
                        raise RuntimeError(f"Invalid published output for {uid}")
                    completed.add(uid)
        counts = " ".join(f"{status}={count}" for status, count in task_counts.items())
        print(f"Nextflow task counts for {job}: {counts}", flush=True)
    if set(expected) != completed | failed.keys() or completed.intersection(failed):
        raise RuntimeError("Nextflow did not account for every requested session; see " + str(trace))
    summary = {"completed": sorted(completed), "failed": failed, "trace": trace.name if trace else None}
    target = job / "session-status.json"
    staged = job / ".session-status.tmp"
    staged.write_text(json.dumps(summary, indent=2) + "\n")
    staged.replace(target)
    if failed:
        raise PartialBatchError(f"{len(failed)} failed, {len(completed)} completed. See {target}")


def run_worker(a):
    dataset = (a.root / a.dataset).resolve()
    if not dataset.is_relative_to(a.root) or dataset == a.root:
        raise ValueError("Dataset must be inside the input collection")
    reason = dataset_exclusion(a.dataset) or dataset_exclusion(dataset)
    if reason:
        print(f"SKIP\t{a.dataset}\t{reason}", flush=True)
        if "FASTPREP_GPU_SLOT" in os.environ:
            # Already-running rank launchers understand 3 as continue without a .done marker.
            raise SystemExit(3)
        return
    units = a.units.split(",")
    if any(not UNIT.fullmatch(u) for u in units) or len(set(units)) != len(units):
        raise ValueError("Invalid or duplicate subject/session")
    rel = dataset.relative_to(a.root)
    job = a.work_root / rel / f"part-{a.part:04d}"
    out = a.out_root / rel / job.name  # each Nextflow run publishes a batch-wide qc/ index
    settings = json.loads(a.software.read_text())
    helper = os.path.expandvars(os.path.expanduser(settings["python"]["helper"]))
    helper = shutil.which(helper) if "/" not in helper else str(a.workflow / helper)
    if not helper or not os.access(helper, os.X_OK):
        raise ValueError(f"Helper Python from {a.software} is unavailable: {settings['python']['helper']}")
    nf = shutil.which(a.nextflow)
    if not nf:
        raise ValueError(f"Nextflow executable unavailable: {a.nextflow}")
    with exclusive_lock(job / ".worker.lock"):
        history = job / ".nextflow/history"
        resume = ["-resume"] if history.is_file() and history.stat().st_size else []
        if not resume and out.is_dir() and any(out.iterdir()):
            raise ValueError(f"Nonempty output without matching Nextflow history: {out}")
        mode = "resume requested (-resume)" if resume else "new run"
        print(f"Nextflow {mode} for {rel}/{job.name}, work directory {job / 'work'}. "
              "Preparing inputs before Nextflow starts.", flush=True)
        view = input_view(dataset, units, job, a.isolate_session_errors)
        control = job / "control"
        control.mkdir(exist_ok=True)
        # prepare.py requires an empty directory. Keep unchanged control files and
        # their timestamps across retries so Nextflow can reuse its task cache.
        with tempfile.TemporaryDirectory(dir=job, prefix="prepare-") as tmp:
            prepare = [helper, "-B", str(a.workflow / "bin/prepare.py"), "--bids-dir", str(view),
                       "--software", str(a.software), "--output", tmp, "--t2-mode", "auto"]
            if a.isolate_session_errors:
                prepare.extend(["--batch-units-json", str(job / "units.json")])
            if a.templateflow_home:
                prepare.extend(["--templateflow-home", str(a.templateflow_home)])
            subprocess.run(prepare, cwd=a.workflow, check=True)
            plans = json.loads((Path(tmp) / "inputs.json").read_text())
            rejected = json.loads((Path(tmp) / "runtime.json").read_text()).get("rejected_sessions", {}) if a.isolate_session_errors else {}
            expected = [u.replace("/", "_") for u in units]
            if sorted([p["analysis_unit_id"] for p in plans] + list(rejected)) != sorted(expected):
                raise ValueError("Prepared sessions differ from the submission plan")
            updated = []
            for name in ("runtime.json", "workflow.json", "inputs.json"):
                src, dst = Path(tmp) / name, control / name
                if not dst.exists() or not filecmp.cmp(src, dst, shallow=False):
                    os.replace(src, dst)
                    updated.append(name)
            if resume:
                print("Prepared control files updated: " + (", ".join(updated) or "none"), flush=True)
        trace = None
        isolate_args = []
        if a.isolate_session_errors:
            if not plans:
                finish_sessions(job, out, expected, rejected, None)
            trace = job / f"session-trace-{uuid.uuid4().hex}.tsv"
            isolate_args = ["--isolate_session_errors", "true", "-with-trace", str(trace)]
        subprocess.run([nf, "run", str(a.workflow / "main.nf"), *resume, *isolate_args,
                        "-work-dir", str(job / "work"), "--runtime", str(control / "runtime.json"),
                        "--blueprint", str(control / "workflow.json"), "--helper_python", helper,
                        "--device", "cuda:0", "--threads", str(a.threads), "--qc_threads", str(a.threads),
                        "--topology_backend", a.surface_backend, "--t1_placement_backend", a.surface_backend,
                        "--t2_pial_backend", a.surface_backend,
                        "--qc_mode", "report_only", "--outdir", str(out)], cwd=job, check=True)
        if a.isolate_session_errors:
            finish_sessions(job, out, expected, rejected, trace)


def experiment(name):
    raw = subprocess.check_output(["airsctl", "experiment", "list", "-N", name], text=True)
    if not raw.strip():
        raise ValueError(f"AIRs experiment not found: {name}")
    info = json.loads(raw)
    if info.get("experiment_name") != name or not isinstance(info.get("advance_config_infos"), list):
        raise ValueError(f"Unexpected AIRs experiment response for {name}")
    return info


def register(a, group):
    info = experiment(a.experiment)
    configs = info["advance_config_infos"]
    by_name = {c["config_name"]: c for c in configs}
    if a.base_config not in by_name:
        raise ValueError(f"Template {a.base_config!r} not found in {a.experiment}")
    changed = False
    for name, command, _ in group:
        if name in by_name:
            if by_name[name].get("command") != command:
                raise ValueError(f"Existing AIRs config {name!r} has a different command or session list")
            continue
        item = copy.deepcopy(by_name[a.base_config])
        item.pop("conf_id", None)
        for resource in item.get("resource_config_list", []):
            resource.pop("id", None)
        item.update(config_name=name, command=command)
        configs.append(item)
        changed = True
    if changed:
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as tmp:
            json.dump(info, tmp)
            tmp.flush()
            subprocess.run(["airsctl", "experiment", "modify", "-f", tmp.name], check=True,
                           stdout=subprocess.DEVNULL)
        saved = {c["config_name"]: c.get("command") for c in experiment(a.experiment)["advance_config_infos"]}
        if any(saved.get(name) != cmd for name, cmd, _ in group):
            raise RuntimeError("AIRs did not persist the requested configs; verify the experiment")


def submit(a, jobs):
    info = experiment(a.experiment)
    state = a.work_root / ".submissions" / Path(info["experiment_id"]).name
    with exclusive_lock(state / ".lock"):
        for offset in range(0, len(jobs), 50):
            group = jobs[offset:offset + 50]
            register(a, group)
            if not a.run_jobs:
                continue
            for name, _, _ in group:
                marker = state / (name + ".status")
                if marker.exists():
                    status = marker.read_text().strip()
                    if status == "submitted":
                        print("Already submitted:", name)
                        continue
                    raise RuntimeError(f"Check AIRs job history for {name} before clearing {marker}")
                # A lost CLI response can follow a successful submission.
                # Leave this marker for reconciliation rather than launch a duplicate.
                marker.write_text("pending\n")
                cmd = ["airsctl", "job", "run", "-N", a.experiment, "-n", name]
                if a.duration_hours is not None:
                    cmd.extend(["--duration-hours", str(a.duration_hours)])
                if a.queue_priority_mode:
                    cmd.extend(["--queue-priority-mode", a.queue_priority_mode])
                subprocess.run(cmd, check=True)
                marker.write_text("submitted\n")
                print("Submitted:", name, flush=True)


def positive(value):
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    worker = argv[:1] == ["worker"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("FASTPREP_BIDS_ROOT", "smri")), help="raw dataset collection")
    parser.add_argument("--workflow", type=Path, default=WORKFLOW)
    parser.add_argument("--software", type=Path, default=os.environ.get("FASTPREP_SOFTWARE"),
                        help="FastPrep software JSON as seen by each worker")
    parser.add_argument("--work-root", type=Path, default=Path.cwd() / "fastprep-anat-work")
    parser.add_argument("--out-root", type=Path, default=Path.cwd() / "fastprep-anat-results")
    parser.add_argument("--nextflow", default=os.environ.get("FASTPREP_NEXTFLOW", "nextflow"),
                        help="worker executable name or absolute path")
    parser.add_argument("--templateflow-home", type=Path, help="shared TemplateFlow cache on workers")
    parser.add_argument("--threads", type=positive, default=8)
    parser.add_argument("--isolate-session-errors", action=argparse.BooleanOptionalAction, default=True,
                        help="continue healthy sessions when another session fails")
    parser.add_argument("--surface-backend", choices=["gpu", "cpu"], default="gpu",
                        help="FastPrep topology and white/pial placement backend; models still use cuda:0")
    parser.add_argument("--dataset", required=worker, help="relative dataset or collection, e.g. PT030_OpenNeuro/ds000001")
    if worker:
        parser.add_argument("--units", required=True)
        parser.add_argument("--part", type=positive, required=True)
    else:
        mode = parser.add_mutually_exclusive_group()
        mode.add_argument("--dry-run", "-d", action="store_true", help="preview only (default)")
        mode.add_argument("--register-only", action="store_true", help="register configs without launching jobs")
        mode.add_argument("--run-jobs", "-r", action="store_true", help="register configs and launch unsubmitted jobs")
        parser.add_argument("--experiment", "-N", help="existing AIRs experiment name")
        parser.add_argument("--base-config", "-n", default="config1", help="existing GPU resource template")
        parser.add_argument("--sessions-per-job", type=positive, default=32)
        parser.add_argument("--start-index", "-s", type=int, default=0, help="zero-based job index after dataset selection")
        parser.add_argument("--limit", "-l", type=positive, help="maximum number of AIRs jobs")
        parser.add_argument("--show-commands", action="store_true", help="print each selected worker command")
        parser.add_argument("--list-skipped", action="store_true", help="print omitted dataset/session paths and reasons")
        parser.add_argument("--env-script", type=Path, help="Bash setup file to source inside the AIRs worker")
        parser.add_argument("--duration-hours", type=positive, help="duration for AIRs public queues")
        parser.add_argument("--queue-priority-mode", choices=["normal", "high"])
    a = parser.parse_args(argv[1:] if worker else argv)
    for key in ["root", "workflow", "work_root", "out_root"]:
        setattr(a, key, getattr(a, key).expanduser().resolve())
    # Worker-only paths may appear only after its image and volumes are mounted.
    a.software = (a.software or a.workflow / "software.local.json").expanduser().absolute()
    if a.templateflow_home:
        a.templateflow_home = a.templateflow_home.expanduser().absolute()
    for p in (a.work_root, a.out_root):
        if p.is_relative_to(a.root) or a.root.is_relative_to(p):
            parser.error("work and output directories must be separate from the raw dataset tree")
    if a.work_root.is_relative_to(a.out_root) or a.out_root.is_relative_to(a.work_root):
        parser.error("work and output directories must be separate")
    if worker:
        run_worker(a)
        return
    if a.start_index < 0:
        parser.error("--start-index must be nonnegative")
    if (a.register_only or a.run_jobs) and not a.experiment:
        parser.error("-N/--experiment is required to register or run jobs")
    if a.env_script:
        a.env_script = a.env_script.expanduser().absolute()
    jobs = make_plan(a)
    print(f"Selected {len(jobs)} jobs / {sum(n for _, _, n in jobs)} sessions")
    if a.show_commands:
        for name, command, _ in jobs:
            print(name, command, sep="\t")
    if jobs and (a.register_only or a.run_jobs):
        submit(a, jobs)


if __name__ == "__main__":
    try:
        main()
    except PartialBatchError as exc:
        print(f"PARTIAL BATCH: {exc}", file=sys.stderr)
        sys.exit(3)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f"ERROR: {exc}")

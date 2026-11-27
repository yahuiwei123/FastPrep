"""BIDS session selection and rigid alignment without an external controller."""

import csv
from itertools import product
from pathlib import Path
import re


def session_plan(runtime, uid):
    plans = [plan for plan in runtime["session_input_plan"] if plan["analysis_unit_id"] == uid]
    if len(plans) != 1:
        raise ValueError("Expected exactly one input plan for " + uid)
    plan = plans[0]
    expected = plan["subject_id"] + ("_ses-" + plan["session_id"] if plan["session_id"] else "")
    if uid != expected:
        raise ValueError("Session identity disagrees with subject/session entities")
    for item in [plan["reference"], *plan["companions"]]:
        if item["analysis_unit_id"] != uid:
            raise ValueError("An acquisition crosses the session boundary")
    return plan


def foreground_snr_proxy(values):
    import numpy as np

    ends = [size - size % 2 for size in values.shape]
    strides = [2 * max(1, (end + 127) // 128) for end in ends]
    signal = detail = positive = None
    for corner in product(range(2), repeat=3):
        tile = values[tuple(slice(start, end, stride) for start, end, stride in zip(corner, ends, strides))].astype(np.float64)
        if signal is None:
            signal = np.zeros_like(tile)
            detail = np.zeros_like(tile)
            positive = np.ones(tile.shape, dtype=bool)
        signal += tile / 8
        detail += (-1 if sum(corner) % 2 else 1) * tile / np.sqrt(8)
        positive &= tile > 0
    means = signal[positive]
    if means.size < 64:
        return None
    high = np.percentile(means, 95)
    foreground = positive & (signal >= 0.2 * high) & (signal <= high)
    if np.count_nonzero(foreground) < 64:
        return None
    coefficients = detail[foreground]
    noise = np.median(np.abs(coefficients - np.median(coefficients))) / 0.67448975
    level = np.median(signal[foreground])
    if not np.isfinite(noise) or noise <= np.finfo(np.float64).eps * level * 16:
        return None
    return float(level / noise)


def image_facts(path, *, score_snr=False):
    import nibabel as nib
    import numpy as np

    image = nib.load(path)
    values = np.asarray(image.dataobj)
    units = image.header.get_xyzt_units()[0]
    if units not in {"mm", "unknown"}:
        raise ValueError("Convert spatial units to mm before processing: " + str(path))
    if (values.ndim != 3 or not np.isfinite(values).all() or not np.any(values > 0)
            or not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-10
            or any(spacing <= 0 for spacing in image.header.get_zooms()[:3])):
        raise ValueError("Input is not a finite, nonempty 3D structural image with valid geometry: " + str(path))
    facts = {"shape": list(image.shape), "voxel_size_mm": [float(value) for value in image.header.get_zooms()[:3]],
            "orientation": list(nib.aff2axcodes(image.affine)), "affine": image.affine.tolist(),
            "finite": True, "nonzero": True, "spatial_units": units,
            "unknown_units_assumed_mm": units == "unknown", "visual_review": "pending"}
    if score_snr:
        facts["snr_proxy"] = foreground_snr_proxy(values)
    return facts


def acquisitions(bids_dir, subjects=(), sessions=(), unit=None):
    from reference_runtime import read_json

    root = Path(bids_dir).expanduser().resolve()
    if not read_json(root / "dataset_description.json").get("BIDSVersion"):
        raise ValueError("dataset_description.json must declare BIDSVersion")
    groups = {}
    seen = set()
    paths = (root / unit / "anat").glob("*.nii*") if unit is not None else root.glob("sub-*/**/anat/*.nii*")
    for path in sorted(paths):
        if not path.name.endswith((".nii", ".nii.gz")):
            continue
        stem = path.name.removesuffix(".gz").removesuffix(".nii")
        tokens = stem.split("_")
        if tokens[-1] not in {"T1w", "T2w", "FLAIR"}:
            continue
        entities = dict(token.split("-", 1) for token in tokens[:-1] if "-" in token)
        subject, session = entities.get("sub", ""), entities.get("ses", "")
        if not re.fullmatch(r"[A-Za-z0-9]+", subject) or session and not re.fullmatch(r"[A-Za-z0-9]+", session):
            raise ValueError("Invalid BIDS identity: " + str(path))
        if subjects and subject not in subjects or sessions and session not in sessions:
            continue
        expected_parent = root / ("sub-" + subject)
        if session:
            expected_parent /= "ses-" + session
        if path.parent != expected_parent / "anat":
            raise ValueError("Filename subject/session differs from its BIDS directory: " + str(path))
        identity = (subject, session, tuple(sorted(entities.items())), tokens[-1])
        if identity in seen:
            raise ValueError("Duplicate image entities (.nii and .nii.gz are not distinct acquisitions): " + stem)
        seen.add(identity)
        uid = "sub-" + subject + ("_ses-" + session if session else "")
        groups.setdefault(uid, []).append({"series_id": stem, "path": str(path.resolve()), "entities": entities,
                                          "analysis_unit_id": uid, "subject_id": "sub-" + subject,
                                          "session_id": session, "modality": tokens[-1]})
    if not groups:
        raise ValueError("No selected structural BIDS acquisitions")
    return groups


def selection_rows(path, bids_dir):
    if path is None:
        return {}
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"subject", "session", "t1w", "t2w"}.issubset(reader.fieldnames or []):
            raise ValueError("Selection TSV columns: subject, session, t1w, t2w")
        for row in reader:
            subject = row["subject"].removeprefix("sub-")
            session = row["session"].removeprefix("ses-")
            uid = "sub-" + subject + ("_ses-" + session if session else "")
            if uid in result:
                raise ValueError("Duplicate selection row: " + uid)
            chosen = {}
            for key in ("t1w", "t2w"):
                if row[key]:
                    selected = Path(row[key]).expanduser()
                    chosen[key] = str((selected if selected.is_absolute() else Path(bids_dir) / selected).resolve())
            result[uid] = chosen
    return result


def choose_series(candidates, requested, policy, modality, uid):
    import numpy as np

    if requested:
        matches = [item for item in candidates if item['path'] == requested]
        if len(matches) != 1:
            raise ValueError(f"Selected {modality[:2]} is absent from the same BIDS session: {uid}")
        return matches[0], {"reason": "manual", "series_id": matches[0]['series_id']}
    if policy == 'strict' and len(candidates) > 1:
        if modality == 'T1w':
            raise ValueError('Select exactly one T1w using --selections for ' + uid)
        raise ValueError('Multiple T2w companions: specify --selections for ' + uid)
    eligible = [item for item in candidates if item['entities'].get('part', 'mag') == 'mag']
    if not eligible:
        raise ValueError(f'No magnitude {modality} candidate for {uid}')
    if len(candidates) == 1:
        return eligible[0], {"reason": "unique", "series_id": eligible[0]['series_id']}
    scores = []
    for item in eligible:
        facts = item['probe']['facts']
        axes = np.asarray(facts['affine'])[:3, :3]
        scores.append(dict(series_id=item['series_id'], voxel_volume_mm3=round(float(abs(np.linalg.det(axes))), 4),
                           largest_spacing_mm=round(float(np.linalg.norm(axes, axis=0).max()), 4), snr_proxy=facts['snr_proxy']))
    scores.sort(key=lambda row: (row['voxel_volume_mm3'], row['largest_spacing_mm'], row['snr_proxy'] is None,
                                 -(row['snr_proxy'] or 0), row['series_id']))
    winner = next(item for item in eligible if item['series_id'] == scores[0]['series_id'])
    factor = 'only_magnitude' if len(scores) == 1 else next(key for key in
              ('voxel_volume_mm3', 'largest_spacing_mm', 'snr_proxy', 'series_id') if scores[0][key] != scores[1][key])
    return winner, dict(reason='ranked', series_id=winner['series_id'], deciding_factor=factor, ranked_candidates=scores,
                        excluded_non_magnitude=[item['series_id'] for item in candidates if item['entities'].get('part', 'mag') != 'mag'])


def prepare_plans(bids_dir, subjects=(), sessions=(), selections=None, t2_mode="auto", selection_policy="resolution-snr", unit=None):
    from t2_pial import select_t2

    if selection_policy not in {'strict', 'resolution-snr'}:
        raise ValueError('Invalid selection policy: ' + str(selection_policy))
    groups = acquisitions(bids_dir, subjects, sessions, unit)
    choices = selection_rows(selections, bids_dir)
    if set(choices) - set(groups):
        raise ValueError("Selection TSV contains sessions outside the selected dataset: " + ", ".join(sorted(set(choices) - set(groups))))
    plans, t2_series = [], {}
    for uid, items in groups.items():
        requested = choices.get(uid, {})
        t1_images = [item for item in items if item["modality"] == "T1w"]
        t2_images = [item for item in items if item['modality'] == 'T2w']
        ranked = {modality for modality, candidates in [('T1w', t1_images), ('T2w', t2_images)]
                  if selection_policy == 'resolution-snr' and len(candidates) > 1 and modality.lower() not in requested
                  and (modality != 'T2w' or t2_mode != 'off')}
        for item in items:
            item["probe"] = {"facts": image_facts(item["path"], score_snr=item['modality'] in ranked and
                                                 item['entities'].get('part', 'mag') == 'mag')}
            item["structural_status"] = "pass"
            item["visual_review"] = "pending"
        reference, t1_decision = choose_series(t1_images, requested.get('t1w'), selection_policy, 'T1w', uid)
        selection = dict(policy=selection_policy, algorithm_version=1, T1w=t1_decision)
        plan = {"analysis_unit_id": uid, "subject_id": reference["subject_id"], "session_id": reference["session_id"], "selection": selection,
                "reference": reference, "companions": [item for item in items if item is not reference]}
        if 't2w' in requested or (t2_mode != 'off' and t2_images):
            chosen, selection['T2w'] = choose_series(t2_images, requested.get('t2w'), selection_policy, 'T2w', uid)
            t2_series[uid] = chosen['series_id']
        else:
            selection['T2w'] = dict(reason='disabled' if t2_mode == 'off' else 'absent', series_id=None)
        select_t2(plan, {"t2_mode": t2_mode, "t2_series_ids": t2_series})
        plans.append(plan)
    return plans, t2_series


ALIGNMENT_SCHEMA = "fastprep.session-alignment/2"
FLIRT_CONVENTION = "FSL FLIRT scaled-mm; moving to fixed"


def require_flirt_header(image, path):
    import numpy as np

    forms = [matrix for matrix, code in (image.get_sform(coded=True), image.get_qform(coded=True)) if code]
    if not forms:
        raise ValueError("FLIRT header mapping requires a coded sform or qform: " + str(path))
    signs = {np.sign(np.linalg.det(matrix[:3, :3])) for matrix in forms}
    if len(signs) != 1 or not signs.issubset({-1, 1}) or not all(np.isfinite(matrix).all() for matrix in forms):
        raise ValueError("Invalid or conflicting NIfTI handedness for FLIRT: " + str(path))


def read_flirt_matrix(path, *, rigid=False):
    import numpy as np

    matrix = np.loadtxt(path)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-10):
        raise ValueError("Invalid FLIRT matrix: " + str(path))
    rotation = matrix[:3, :3]
    if rigid and (np.linalg.det(rotation) <= 0 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)):
        raise ValueError("Expected a proper rigid FLIRT matrix: " + str(path))
    return matrix


def align_companions(stage):
    import nibabel as nib
    import numpy as np
    from reference_runtime import checked_file, read_json, write_json

    plan = session_plan(stage.runtime, stage.subject)
    fixed = checked_file(plan["reference"]["path"])
    manifest = read_json(stage.path("t1w_manifest"))
    if manifest["scans"] != [{key: plan["reference"][key] for key in ("path", "entities")}]:
        raise ValueError("Alignment reference disagrees with the selected reconstruction reference")
    fixed_image = nib.load(fixed)
    if plan["companions"]:
        require_flirt_header(fixed_image, fixed)
    fits, companions = [], []
    for item in plan["companions"]:
        moving = checked_file(item["path"])
        require_flirt_header(nib.load(moving), moving)
        directory = stage.root / "alignment" / item["series_id"]
        directory.mkdir(parents=True)
        transform = directory / "moving-to-T1w.flirt.mat"
        registered = directory / "registered.nii.gz"
        common = ["-in", moving, "-ref", fixed, "-interp", "trilinear", "-datatype", "float"]
        fits.append(("fsl:flirt", [*common, "-dof", "6", "-cost", "mutualinfo", "-searchcost", "mutualinfo",
                     "-bins", "32", "-omat", transform, "-out", registered]))
        companions.append((item, moving, common, transform, registered))

    workers = stage.spec.get("companion_workers", 1)
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError("companion_workers must be 1 or 2")
    parallel = min(workers, len(fits), int(stage.spec["threads"])) > 1
    if parallel:
        stage.run_concurrent(fits, max_workers=workers)

    members, registrations = [], []
    for (item, moving, common, transform, registered), fit in zip(companions, fits):
        if not parallel:
            stage.run(*fit)
        read_flirt_matrix(checked_file(transform), rigid=True)
        directory = registered.parent
        baseline = directory / "baseline.nii.gz"
        stage.run("fsl:flirt", [*common, "-applyxfm", "-usesqform", "-out", baseline])
        for path in (registered, baseline):
            aligned = nib.load(path)
            if (aligned.shape != fixed_image.shape or not np.allclose(aligned.affine, fixed_image.affine, atol=1e-4, rtol=0)
                    or not np.isfinite(np.asarray(aligned.dataobj)).all()):
                raise ValueError("Companion output does not match the reference physical grid: " + str(path))
        sidecar = directory / "registered.json"
        write_json(sidecar, {"Sources": [str(moving)], "SpatialReference": str(fixed),
                            "Transform": str(transform.relative_to(stage.root)),
                            "TransformConvention": FLIRT_CONVENTION, "Backend": "FSL FLIRT", "DOF": 6,
                            "Cost": "mutualinfo", "SearchCost": "mutualinfo", "HistogramBins": 32,
                            "Interpolation": "trilinear", "DeclaredModality": item["modality"], "VisualReview": "pending"})
        registrations.append({"series_id": item["series_id"], "moving": str(moving), "fixed": str(fixed),
                              "registered": str(registered.relative_to(stage.root)),
                              "baseline": str(baseline.relative_to(stage.root)), "transform": str(transform.relative_to(stage.root))})
        members.extend(stage.file_record(path, hemi=None) for path in (registered, baseline, transform, sidecar))
    result = stage.root / "alignment_manifest.json"
    write_json(result, {"schema": ALIGNMENT_SCHEMA, "transform_convention": FLIRT_CONVENTION, "analysis_unit_id": stage.subject,
                        "reference": {key: plan["reference"][key] for key in ("path", "series_id")},
                        "registrations": registrations, "members": members,
                        "companion_count": len(plan["companions"]), "registered_count": len(registrations)})
    stage.metrics.update(companion_count=len(plan["companions"]), physical_grids_verified=True)
    return {"alignment_manifest": result}

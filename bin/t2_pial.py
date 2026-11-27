"""Prepare a structurally checked same-session T2 on the reconstruction grid."""


def initial_t1_pial_profile(parameters, t2_available, stats_path):
    import math
    from pathlib import Path

    scale = parameters.get("t2_initial_t1_threshold_scale", 1.0)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("t2_initial_t1_threshold_scale must be finite and in (0, 1]")
    metadata = {"profile": "native-default", "requested_threshold_scale": scale,
                "effective_threshold_scale": 1.0, "applied": False,
                "reason": "T1_only" if not t2_available else "scale_is_one"}
    if not t2_available or scale == 1:
        return [], metadata
    fields = ("pial_border_low", "pial_border_hi", "pial_outside_hi")
    required = (*fields, "pial_outside_low", "pial_inside_hi")
    baseline = {}
    for line in Path(stats_path).read_text().splitlines():
        parts = line.split()
        if not parts or parts[0] not in required:
            continue
        if len(parts) != 2 or parts[0] in baseline:
            raise ValueError("Malformed or duplicate pial threshold in autodet_gwstats")
        baseline[parts[0]] = float(parts[1])
    if set(baseline) != set(required) or not all(math.isfinite(value) and value > 0 for value in baseline.values()):
        raise ValueError("autodet_gwstats requires finite positive pial thresholds")
    arguments = [token for field in fields for token in ("--" + field, format(baseline[field] * scale, ".9g"))]
    effective = dict(baseline)
    effective.update({field: float(arguments[index * 2 + 1]) for index, field in enumerate(fields)})
    for thresholds in (baseline, effective):
        if not (thresholds["pial_outside_low"] < thresholds["pial_border_low"]
                <= thresholds["pial_border_hi"] <= thresholds["pial_inside_hi"]
                and thresholds["pial_outside_low"] < thresholds["pial_outside_hi"] <= thresholds["pial_inside_hi"]):
            raise ValueError("Invalid pial threshold ordering; increase t2_initial_t1_threshold_scale")
    metadata.update({"profile": "t2-init-lower-t1-thresholds-v1", "applied": True,
                     "effective_threshold_scale": scale, "reason": "T2_refinement_available",
                     "baseline_thresholds": baseline, "effective_thresholds": effective,
                     "native_overrides": arguments, "scope": "initial_T1_pial_only"})
    return arguments, metadata


def t2_refinement_profile(parameters, t2_available):
    import math

    requested = parameters.get("t2_refinement_overrides", {})
    defaults = {"mm_min_inside": 110.0, "mm_max_inside": 300.0,
                "mm_min_outside": 130.0, "mm_max_outside": 300.0,
                "location": 0.5, "curv": 0.5, "nspring": 1.0}
    if not isinstance(requested, dict) or set(requested) - set(defaults):
        raise ValueError("Unsupported t2_refinement_overrides")
    effective = dict(defaults)
    for name, value in requested.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("T2 refinement overrides must be finite positive numbers")
        effective[name] = value
    if effective["nspring"] > 1:
        raise ValueError("T2 nspring must not exceed the native cap of 1")
    for region in ("inside", "outside"):
        if effective["mm_min_" + region] >= effective["mm_max_" + region]:
            raise ValueError("Invalid T2 refinement threshold ordering")
    if effective["mm_min_outside"] < effective["mm_min_inside"]:
        raise ValueError("T2 outside minimum must not be below inside minimum")
    arguments = [token for name in defaults if name in requested
                 for token in ("--" + name, format(requested[name], ".9g"))] if t2_available else []
    return arguments, {"scope": "T2_refinement_only", "requested": dict(requested),
                       "applied": bool(arguments), "native_overrides": arguments,
                       "effective_parameters": effective if t2_available else None,
                       "baseline": "FreeSurfer-8.2.0-mmvol-T2",
                       "reason": "T2_available" if t2_available else "T1_only"}


def select_t2(plan, parameters):
    mode = parameters["t2_mode"]
    if mode not in {"auto", "require", "off"}:
        raise ValueError("t2_mode must be auto, require or off")
    selected = parameters.get("t2_series_ids", {}).get(plan["analysis_unit_id"])
    if mode == "off":
        if selected:
            raise ValueError("An explicit T2 selection conflicts with t2_mode=off")
        return None
    candidates = [item for item in plan["companions"] if item["modality"] == "T2w"]
    if selected:
        candidates = [item for item in candidates if item["series_id"] == selected]
        if len(candidates) != 1:
            raise ValueError("Selected T2 series is absent or ambiguous in this session: " + selected)
    if not candidates:
        if mode == "require":
            raise ValueError("T2 pial refinement requires a same-session T2w")
        return None
    if len(candidates) != 1:
        raise ValueError("Multiple T2w companions: set t2_series_ids for " + plan["analysis_unit_id"])
    candidate = candidates[0]
    if (candidate["analysis_unit_id"] != plan["analysis_unit_id"] or candidate["structural_status"] != "pass"
            or len(candidate["probe"]["facts"]["shape"]) != 3):
        raise ValueError("T2 pial refinement requires a structurally valid 3D image in the same session")
    return candidate


def prepare_t2(stage):
    import nibabel as nib
    import numpy as np
    from reference_runtime import artifact_path, checked_file, copy_file, read_json, same_grid, session_plan
    from session_inputs import ALIGNMENT_SCHEMA, FLIRT_CONVENTION, read_flirt_matrix
    from tissue_bias import correct_image

    plan = session_plan(stage.runtime, stage.subject)
    selected = select_t2(plan, stage.node["parameters"])
    if selected is None:
        return None, {"mode": "T1_only", "reason": "disabled" if stage.node["parameters"]["t2_mode"] == "off" else "no_T2w",
                      "analysis_unit_id": stage.subject}, []
    directory, record = stage.inputs["alignment_manifest"]
    manifest = read_json(stage._upstream_path("alignment_manifest"))
    if manifest.get("schema") != ALIGNMENT_SCHEMA or manifest.get("transform_convention") != FLIRT_CONVENTION:
        raise ValueError("Expected a FLIRT alignment manifest; rerun session alignment with a new control directory")
    if (manifest["analysis_unit_id"] != stage.subject
            or manifest["reference"]["series_id"] != plan["reference"]["series_id"]):
        raise ValueError("T2 alignment belongs to a different session or reference")
    registrations = [item for item in manifest["registrations"] if item["series_id"] == selected["series_id"]]
    if len(registrations) != 1:
        raise ValueError("Selected T2 must have exactly one alignment record")
    registration = registrations[0]
    if registration["moving"] != selected["path"] or registration["fixed"] != plan["reference"]["path"]:
        raise ValueError("T2 transform source or target differs from the selected input plan")
    transforms = [member for member in record["members"] if member["path"] == registration["transform"]]
    if len(transforms) != 1:
        raise ValueError("T2 alignment transform is not bound to the released manifest")
    transform = artifact_path(directory, transforms[0])
    moving = checked_file(selected["path"])
    reference = nib.load(stage.path("brain_finalsurfs_mgz"))
    anatomy = same_grid(reference, reference)
    fixed = stage.output("mri/T2.reference.nii.gz")
    fixed_image = nib.Nifti1Image(anatomy.astype(np.float32), reference.affine)
    fixed_image.header.set_xyzt_units("mm")
    nib.save(fixed_image, fixed)
    aligned = stage.output("mri/T2.prenorm.nii.gz")
    native_to_recon = stage.output("mri/transforms/native-T1w-to-recon-grid.flirt.mat")
    t2_to_recon = stage.output("mri/transforms/native-T2w-to-recon.flirt.mat")
    native_rigid = read_flirt_matrix(transform, rigid=True)
    stage.run("fsl:flirt", ["-in", checked_file(registration["fixed"]), "-ref", fixed, "-applyxfm", "-usesqform",
              "-omat", native_to_recon])
    np.savetxt(t2_to_recon, read_flirt_matrix(native_to_recon) @ native_rigid, fmt="%.15g")
    stage.run("fsl:flirt", ["-in", moving, "-ref", fixed, "-out", aligned, "-applyxfm", "-init", t2_to_recon,
              "-interp", "trilinear", "-datatype", "float"])
    same_grid(reference, nib.load(aligned))
    corrected = stage.output("mri/T2.norm.mgz")
    paths, bias_metrics = correct_image(stage, aligned, stage.path("aseg_presurf_mgz"), stage.path("mask_mgz"),
                                       corrected, stage.root / "t2-bfc", "T2", stage.node["parameters"]["bias"])
    target = stage.output("mri/T2.mgz")
    same_grid(reference, nib.load(corrected))
    copy_file(corrected, target)
    metadata = {"mode": "T2_refined", "analysis_unit_id": stage.subject, "series_id": selected["series_id"],
                "source": str(moving), "alignment_series_id": registration["series_id"],
                "resampling": "one trilinear interpolation from native T2 using the FLIRT rigid transform rebased to the reconstruction grid",
                "transform_convention": FLIRT_CONVENTION, "resampling_transform": str(t2_to_recon.relative_to(stage.root)),
                "t2_volume": str(target.relative_to(stage.root)),
                "t2_preparation": "bias-corrected-full-head-v1", "skull_stripped": False,
                "refinement_input": "Byte-identical bias-corrected T2.norm.mgz; no brain masking",
                "coverage": "Native T2 coverage retained within the reconstruction reference grid",
                "bias": bias_metrics,
                "scientific_approval": "not_performed"}
    return target, metadata, [fixed, native_to_recon, t2_to_recon, aligned, corrected, target, paths["factor"], paths["tissues"]]


def refinement_manifest(stage, metadata, paths):
    from reference_runtime import write_json

    path = stage.root / "pial_refinement.json"
    write_json(path, {"schema": "fastprep.pial-refinement/1", **metadata,
                      "members": [stage.file_record(member, hemi=None) for member in paths]})
    return path

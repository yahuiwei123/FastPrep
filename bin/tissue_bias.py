"""Adapt FreeSurfer tissues to the reliable-core RBF/Gaussian toolkit."""

import json
from pathlib import Path


def tissue_labels(labels, mask):
    import numpy as np

    tissues = np.zeros(labels.shape, dtype=np.uint8)
    tissues[np.isin(labels, [4, 5, 14, 15, 24, 43, 44])] = 1
    tissues[np.isin(labels, [3, 42])] = 2
    tissues[np.isin(labels, [2, 41, 251, 252, 253, 254, 255])] = 3
    tissues[mask == 0] = 0
    if set(np.unique(tissues)) != {0, 1, 2, 3}:
        raise ValueError("Tissue-guided BFC requires background, CSF, cerebral GM and WM")
    for label in (2, 3, 41, 42):
        if not np.any((labels == label) & (mask > 0)):
            raise ValueError("Tissue-guided BFC requires bilateral cerebral GM and WM")
    return tissues


def correct_image(stage, image_path, labels_path, mask_path, output_path, directory, modality, parameters):
    import nibabel as nib
    import numpy as np
    from reference_runtime import checked_labels, same_grid

    directory = Path(directory)
    if Path(output_path).exists():
        raise FileExistsError("Refusing to overwrite a scientific result; use a new attempt")
    directory.mkdir(parents=True, exist_ok=False)
    original = nib.load(image_path)
    before = same_grid(original, original)
    spacing = original.header.get_zooms()[:3]
    if not np.allclose(spacing, spacing[0], rtol=0, atol=1e-5):
        raise ValueError("This adapter requires the isotropic surface reconstruction grid")
    if before.ndim != 3 or before.min() < 0 or before.max() <= before.min():
        raise ValueError("BFC input must be a nonnegative, nonconstant 3D magnitude image")
    mask = checked_labels(original, nib.load(mask_path), [0, 1])
    labels = checked_labels(original, nib.load(labels_path))
    tissues = tissue_labels(labels, mask)
    white = np.isin(labels, [2, 41]) & (mask > 0)
    if any(not np.any(before[(labels == label) & (mask > 0)] > 0) for label in (2, 3, 41, 42)):
        raise ValueError("BFC input has no positive intensity support in a required tissue")
    method = parameters.get("method", "rbf")
    if method == "TissueGuidedBiasFieldCorrection Gaussian mixture":
        method = "gaussian"
    if method not in {"rbf", "gaussian"}:
        raise ValueError("BFC method must be rbf or gaussian")
    if method == "rbf" and not parameters["preserve_contrast"]:
        raise ValueError("RBF requires preserve_contrast=true; fixed tissue targets are not implemented")
    paths = {name: directory / (name + ".nii.gz") for name in ("input", "tissues", "mask", "corrected", "field", "factor")}
    for name, values in (("input", before.astype(np.float32)), ("tissues", tissues), ("mask", mask.astype(np.uint8))):
        image = nib.Nifti1Image(values, original.affine)
        image.header.set_xyzt_units("mm")
        nib.save(image, paths[name])
    command_id = "tissue-guided-bfc:" + ("rbf-scatter" if method == "rbf" else "gaussian-mixture")
    filename = r"bfc_rbf_scatter\.py" if method == "rbf" else r"bfc_gauss_mixture\.py"
    source = stage.resource_file(command_id, filename)
    upstream_modality = "None" if parameters["preserve_contrast"] else modality
    if (parameters["bias_max_iter"] < 1 or parameters["max_kernel"] < 1
            or parameters["refine_max_iter"] != 0 or parameters["wm_target"] <= 0
            or parameters["label_soft_kernel"] <= 0
            or parameters.get("min_kernel") is not None and parameters["min_kernel"] <= 0):
        raise ValueError("Invalid BFC parameters; this adapter requires fixed tissue priors")
    arguments = ["--source-script", source, "--method", method, "--seed", parameters["seed"],
                 "--input_img", paths["input"], "--input_lab", paths["tissues"],
                 "--input_msk", paths["mask"], "--output_img", paths["corrected"],
                 "--output_bias", paths["field"], "--modality", upstream_modality]
    for name in ("bias_max_iter", "max_kernel", "refine_max_iter", "label_soft_kernel"):
        arguments += ["--" + name, parameters[name]]
    arguments += ["--gm-core-mm", parameters.get("gm_core_mm", 1.0),
                  "--wm-core-mm", parameters.get("wm_core_mm", 2.0),
                  "--core-log-mad-limit", parameters.get("core_log_mad_limit", 3.5)]
    if parameters.get("min_kernel") is not None:
        arguments += ["--min_kernel", parameters["min_kernel"]]
    if method == "rbf":
        arguments += ["--device", stage.spec.get("device", "cuda:0")]
        for name, default in (("n_points", 48), ("rbf_max_iter", 25), ("tol", 1e-6),
                              ("rbf_sigma", 0.05), ("soft_sigma", 0.5), ("k_neighbors", 48)):
            arguments += ["--" + name, parameters.get(name, default)]
    stage.run(command_id, arguments)
    corrected = same_grid(original, nib.load(paths["corrected"]))
    raw_field = same_grid(original, nib.load(paths["field"]))
    if np.any(raw_field <= 0) or np.any(corrected < 0):
        raise ValueError("Tissue-guided BFC produced nonpositive factors or negative intensities")
    factor = (1 / raw_field if method == "rbf" else raw_field).astype(np.float32)
    if not np.isfinite(factor).all() or np.any(factor <= 0):
        raise ValueError("BFC effective multiplication factor is invalid")
    expected = np.where(mask > 0, before * factor, before)
    if not np.allclose(corrected, expected, rtol=2e-4, atol=2e-4):
        raise ValueError("Corrected image disagrees with continuous brain-domain field application")
    if np.any(factor[mask == 0] != 1) or not np.array_equal(corrected[before == 0], before[before == 0]):
        raise ValueError("Correction changed outside-brain factors or zero-signal support")
    support_report = json.loads((directory / "support.json").read_text())
    representation = "divisor" if method == "rbf" else "multiplier"
    if (support_report.get("status") != "completed"
            or support_report.get("field_representation") != representation
            or support_report.get("method") != method):
        raise ValueError("Reliable-core BFC did not complete")
    factor_image = nib.Nifti1Image(factor, original.affine)
    factor_image.header.set_xyzt_units("mm")
    nib.save(factor_image, paths["factor"])
    same_grid(original, nib.load(paths["factor"]))
    median = float(np.median(corrected[white]))
    if median <= 0:
        raise ValueError("Corrected cerebral white matter must have positive median intensity")
    scale = parameters["wm_target"] / median
    result = np.where(before > 0, corrected * scale, 0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("FreeSurfer intensity calibration produced nonfinite values")
    nib.save(nib.MGHImage(result, original.affine), output_path)
    same_grid(original, nib.load(output_path))
    metrics = {"method": method, "modality": modality,
               "upstream_modality": upstream_modality,
               "same_grid": True, "tissue_voxels": {str(label): int(np.sum(tissues == label)) for label in range(4)},
               "factor_semantics": "multiplicative correction of original intensities throughout the brain; unity outside; not an N4 divisor",
               "raw_field_representation": representation, "raw_field_filename": "field.nii.gz",
               "reliable_support": support_report,
               "global_scale": scale, "white_matter_median_before": float(np.median(before[white])),
               "white_matter_median_after": float(np.median(result[white])),
               "factor_range": [float(factor.min()), float(factor.max())],
               "parameters": parameters, "scientific_approval": "not_performed"}
    return paths, metrics


def tissue_bias_correct(stage):
    from reference_runtime import copy_file, volume_outputs

    paths, metrics = correct_image(stage, stage.path("orig_mgz"), stage.path("aseg_auto_noccseg_mgz"),
                                   stage.path("mask_mgz"), stage.volume("orig_nu_mgz"),
                                   stage.root / "tissue-bfc", "T1", stage.node["parameters"])
    copy_file(paths["factor"], stage.volume("bias_factor_nii"))
    stage.metrics["tissue_bias_correct"] = metrics
    return volume_outputs(stage)

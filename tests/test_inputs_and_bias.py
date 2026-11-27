"""Portable BIDS-selection and tissue-guided BFC tests (stdlib unittest)."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
if str(BIN) not in sys.path:
    sys.path.insert(0, str(BIN))

from session_inputs import prepare_plans, require_flirt_header  # noqa: E402
from t2_pial import select_t2  # noqa: E402


def has_modules(*names):
    return all(importlib.util.find_spec(name) is not None for name in names)


@unittest.skipUnless(has_modules("numpy", "nibabel"), "BIDS image tests need NumPy and NiBabel")
class BidsInputSelectionTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        import numpy as np

        self.temp = tempfile.TemporaryDirectory(prefix="fastprep-bids-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bids"
        self.root.mkdir()
        (self.root / "dataset_description.json").write_text(json.dumps({"Name": "test", "BIDSVersion": "1.9.0"}))
        image = nib.Nifti1Image(np.arange(1, 9 * 10 * 11 + 1, dtype=np.float32).reshape(9, 10, 11), np.diag([1, 1, 1, 1]))
        image.header.set_xyzt_units("mm")
        self.files = {}
        entries = [
            ("01", "01", "acq-a", "T1w"),
            ("01", "01", "acq-b", "T1w"),
            ("01", "01", "acq-a", "T2w"),
            ("01", "01", "acq-b", "T2w"),
            ("01", "02", "", "T1w"),
            ("01", "02", "", "T2w"),
            ("02", "", "", "T1w"),
        ]
        for subject, session, acquisition, modality in entries:
            folder = self.root / f"sub-{subject}"
            if session:
                folder /= f"ses-{session}"
            folder /= "anat"
            folder.mkdir(parents=True, exist_ok=True)
            entities = f"sub-{subject}_" + (f"ses-{session}_" if session else "")
            entities += f"{acquisition}_" if acquisition else ""
            path = folder / f"{entities}{modality}.nii.gz"
            nib.save(image, path)
            self.files[(subject, session, acquisition, modality)] = path

    def test_flirt_header_requires_unambiguous_handedness(self):
        import nibabel as nib
        import numpy as np

        path = self.files['01', '02', '', 'T1w']
        image = nib.load(path)
        for code in ('qform', 'sform'):
            image.set_qform(None, code=0)
            image.set_sform(None, code=0)
            with self.assertRaisesRegex(ValueError, 'coded sform or qform'):
                require_flirt_header(image, path)
            getattr(image, 'set_' + code)(np.eye(4), code=1)
            require_flirt_header(image, path)
        image.set_qform(np.diag([-1, 1, 1, 1]), code=1)
        with self.assertRaisesRegex(ValueError, 'conflicting NIfTI handedness'):
            require_flirt_header(image, path)

    def write_selection(self, rows):
        path = Path(self.temp.name) / "selection.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("subject", "session", "t1w", "t2w"), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_filters_multiple_sessions_and_selects_requested_t1_t2(self):
        first_t1 = self.files[("01", "01", "acq-b", "T1w")]
        first_t2 = self.files[("01", "01", "acq-a", "T2w")]
        choices = self.write_selection([
            {"subject": "sub-01", "session": "ses-01", "t1w": str(first_t1.relative_to(self.root)),
             "t2w": str(first_t2.relative_to(self.root))},
            {"subject": "01", "session": "02", "t1w": "", "t2w": ""},
        ])
        plans, selected = prepare_plans(self.root, subjects=("01",), sessions=("01", "02"),
                                        selections=choices, t2_mode="auto")
        self.assertEqual([plan["analysis_unit_id"] for plan in plans], ["sub-01_ses-01", "sub-01_ses-02"])
        by_id = {plan["analysis_unit_id"]: plan for plan in plans}
        first = by_id["sub-01_ses-01"]
        second = by_id["sub-01_ses-02"]
        self.assertEqual(first["reference"]["path"], str(first_t1.resolve()))
        selected_t2 = next(item for item in first["companions"] if item["path"] == str(first_t2.resolve()))
        self.assertEqual(selected["sub-01_ses-01"], selected_t2["series_id"])
        self.assertTrue(all(item["analysis_unit_id"] == "sub-01_ses-01" for item in first["companions"]))
        self.assertEqual(second["session_id"], "02")
        self.assertIsNone(select_t2(second, {"t2_mode": "off", "t2_series_ids": {}}))
        self.assertEqual(len([item for item in second["companions"] if item["modality"] == "T2w"]), 1)
        self.assertNotIn("sub-02", {plan["analysis_unit_id"] for plan in plans})

    def test_strict_policy_rejects_unselected_multiple_t1_and_t2(self):
        with self.assertRaisesRegex(ValueError, "Select exactly one T1w"):
            prepare_plans(self.root, subjects=("01",), sessions=("01",), selection_policy="strict")
        selected_t1 = self.files[("01", "01", "acq-a", "T1w")]
        choices = self.write_selection([{"subject": "01", "session": "01",
                                         "t1w": str(selected_t1.relative_to(self.root)), "t2w": ""}])
        with self.assertRaisesRegex(ValueError, "Multiple T2w companions"):
            prepare_plans(self.root, subjects=("01",), sessions=("01",), selections=choices, selection_policy="strict")

    def write_candidate(self, acquisition, modality, noise, spacing=(1, 1, 1), gain=1):
        import nibabel as nib
        import numpy as np

        path = self.files['01', '01', acquisition, modality]
        values = np.ones((36, 36, 36)) * 100 + np.random.default_rng(19).normal(0, noise, (36, 36, 36))
        values[:4] = 0
        image = nib.Nifti1Image((values * gain).astype(np.float32), np.diag([*spacing, 1]))
        image.header.set_xyzt_units('mm')
        nib.save(image, path)
        return path

    def test_resolution_then_snr_selection_and_downstream_binding(self):
        self.write_candidate('acq-a', 'T1w', 1, (1.5, 1, 1))
        t1 = self.write_candidate('acq-b', 'T1w', 7)
        self.write_candidate('acq-a', 'T2w', 8, gain=20)
        t2 = self.write_candidate('acq-b', 'T2w', 2)
        plans, mapping = prepare_plans(self.root, subjects=('01',), sessions=('01',))
        plan, = plans
        self.assertEqual(plan['reference']['path'], str(t1))
        self.assertEqual(len(plan['companions']), 3)
        self.assertEqual(select_t2(plan, dict(t2_mode='require', t2_series_ids=mapping))['path'], str(t2))
        self.assertEqual(plan['selection']['policy'], 'resolution-snr')
        self.assertEqual(plan['selection']['T1w']['deciding_factor'], 'voxel_volume_mm3')
        self.assertEqual(plan['selection']['T2w']['deciding_factor'], 'snr_proxy')
        winner, runner_up = plan['selection']['T2w']['ranked_candidates']
        self.assertGreater(winner['snr_proxy'], runner_up['snr_proxy'] * 3)
        self.assertEqual(winner['series_id'], mapping['sub-01_ses-01'])
        choices = self.write_selection([dict(subject='01', session='01', t1w=str(self.files['01', '01', 'acq-a', 'T1w']), t2w='')])
        manual, mapping2 = prepare_plans(self.root, subjects=('01',), sessions=('01',), selections=choices)
        self.assertEqual(manual[0]['selection']['T1w']['reason'], 'manual')
        self.assertNotEqual(manual[0]['reference']['path'], str(t1))
        self.assertEqual(mapping2, mapping)

    def test_unavailable_snr_tie_non_magnitude_and_modes(self):
        for acq in ('acq-a', 'acq-b'):
            self.write_candidate(acq, 'T1w', 0)
        plans, mapping = prepare_plans(self.root, subjects=('01',), sessions=('01',), t2_mode='off')
        plan, = plans
        self.assertFalse(mapping)
        self.assertEqual(plan['selection']['T2w']['reason'], 'disabled')
        self.assertEqual(plan['reference']['entities']['acq'], 'a')
        self.assertEqual(plan['selection']['T1w']['deciding_factor'], 'series_id')
        self.assertTrue(all(row['snr_proxy'] is None for row in plan['selection']['T1w']['ranked_candidates']))
        self.assertEqual(len(plan['companions']), 3)
        phase = self.write_candidate('acq-a', 'T1w', 2, (.5, .5, .5))
        phase.rename(phase.with_name(phase.name.replace('_T1w', '_part-phase_T1w')))
        plans, _ = prepare_plans(self.root, subjects=('01',), sessions=('01',))
        self.assertEqual(plans[0]['reference']['entities']['acq'], 'b')
        self.assertEqual(len(plans[0]['selection']['T1w']['excluded_non_magnitude']), 1)
        with self.assertRaisesRegex(ValueError, 'requires a same-session T2w'):
            prepare_plans(self.root, subjects=('02',), t2_mode='require')

    def test_equal_volume_prefers_thinner_slices_and_valid_snr_over_missing(self):
        self.write_candidate('acq-a', 'T1w', 2, (.5, 1, 2))
        self.write_candidate('acq-b', 'T1w', 6)
        self.write_candidate('acq-a', 'T2w', 0)
        self.write_candidate('acq-b', 'T2w', 4)
        plans, _ = prepare_plans(self.root, subjects=('01',), sessions=('01',))
        decision = plans[0]['selection']
        self.assertEqual(decision['T1w']['deciding_factor'], 'largest_spacing_mm')
        for modality in ('T1w', 'T2w'):
            self.assertTrue(decision[modality]['series_id'].endswith('acq-b_' + modality))
        self.assertIsNone(decision['T2w']['ranked_candidates'][1]['snr_proxy'])

    def test_explicit_t2_cannot_cross_session_or_be_used_when_disabled(self):
        first_t1 = self.files[("01", "01", "acq-a", "T1w")]
        second_t2 = self.files[("01", "02", "", "T2w")]
        choices = self.write_selection([{"subject": "01", "session": "01",
                                         "t1w": str(first_t1.relative_to(self.root)),
                                         "t2w": str(second_t2.relative_to(self.root))}])
        with self.assertRaisesRegex(ValueError, "Selected T2 is absent from the same BIDS session"):
            prepare_plans(self.root, subjects=("01",), sessions=("01",), selections=choices, t2_mode="auto")
        plan = {"analysis_unit_id": "sub-01_ses-01", "companions": []}
        with self.assertRaisesRegex(ValueError, "conflicts with t2_mode=off"):
            select_t2(plan, {"t2_mode": "off", "t2_series_ids": {"sub-01_ses-01": "some-t2"}})


class _BfcStage:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.spec = {"device": os.environ.get("FASTPREP_TEST_BFC_CUDA", "cuda:0")}

    def resource_file(self, command_id, filename_pattern):
        import re

        names = {"tissue-guided-bfc:gaussian-mixture": "bfc_gauss_mixture.py",
                 "tissue-guided-bfc:rbf-scatter": "bfc_rbf_scatter.py"}
        if command_id not in names or not re.fullmatch(filename_pattern, names[command_id]):
            raise AssertionError("BFC requested an unexpected resource")
        return ROOT / "vendor/tissue_guided_bfc" / names[command_id]

    def run(self, command_id, arguments):
        if command_id not in {"tissue-guided-bfc:gaussian-mixture", "tissue-guided-bfc:rbf-scatter"}:
            raise AssertionError("BFC requested an unexpected command")
        completed = subprocess.run(
            [sys.executable, "-B", str(BIN / "run_tissue_bfc.py"), *map(str, arguments)],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        if completed.returncode:
            raise AssertionError(f"BFC subprocess failed ({completed.returncode}):\n{completed.stdout}")


BFC_MODULES = ("numpy", "scipy", "nibabel", "SimpleITK", "sklearn")
@unittest.skipUnless(has_modules(*BFC_MODULES), "Synthetic BFC needs NumPy, SciPy, NiBabel, SimpleITK and scikit-learn")
class TissueBiasIntegrationTests(unittest.TestCase):
    def test_t1_and_t2_bias_reduction_geometry_wm_scale_and_label_preservation(self):
        self.exercise_method("gaussian")

    @unittest.skipUnless(os.environ.get("FASTPREP_TEST_BFC_CUDA"), "Set FASTPREP_TEST_BFC_CUDA for real CUDA RBF integration")
    def test_rbf_cuda_t1_t2_and_divisor_to_pipeline_multiplier(self):
        self.exercise_method("rbf")

    def exercise_method(self, method):
        import nibabel as nib
        import numpy as np
        from tissue_bias import correct_image

        with tempfile.TemporaryDirectory(prefix="fastprep-bfc-") as temporary:
            root = Path(temporary)
            axes = np.indices((64, 64, 64), dtype=np.float32)
            radius = np.sqrt(np.sum((axes - 31.5) ** 2, axis=0))
            labels = np.zeros(radius.shape, dtype=np.int16)
            labels[radius < 25] = 24
            labels[radius < 22] = 3
            labels[(radius < 22) & (axes[0] > 31.5)] = 42
            labels[radius < 16] = 2
            labels[(radius < 16) & (axes[0] > 31.5)] = 41
            labels[22:27, 28:34, 28:34] = 10
            labels[37:42, 28:34, 28:34] = 49
            mask = (labels > 0).astype(np.uint8)
            affine = np.array([[-1, 0, 0, 32], [0, 0, 1, -32], [0, -1, 0, 32], [0, 0, 0, 1]], dtype=float)
            white = np.isin(labels, [2, 41])
            gray = np.isin(labels, [3, 42])
            label_path, mask_path = root / "labels.mgz", root / "mask.mgz"
            nib.save(nib.MGHImage(labels, affine), label_path)
            nib.save(nib.MGHImage(mask, affine), mask_path)
            saved_labels = np.asarray(nib.load(label_path).dataobj).copy()
            stage = _BfcStage(root)
            parameters = dict(method=method, bias_max_iter=10, max_kernel=9, min_kernel=None, refine_max_iter=0,
                              label_soft_kernel=0.4, seed=0, wm_target=110, preserve_contrast=True)
            reports = {}
            for modality, means in (("T1", (20, 60, 110)), ("T2", (220, 80, 35))):
                deep = np.isin(labels, [10, 49])
                truth = np.where(labels == 24, means[0], np.where(gray, means[1], np.where(white, means[2], np.where(deep, means[2] * 1.2, 0))))
                field = np.exp(0.4 * (axes[0] - 31.5) / 31.5 + 0.2 * (axes[2] - 31.5) / 31.5)
                noise = np.random.default_rng(17).normal(0, 0.4, truth.shape)
                source = np.where(mask, np.maximum(truth * field + noise, 0), 0).astype(np.float32)
                input_path, output_path = root / f"{modality}-input.mgz", root / f"{modality}-corrected.mgz"
                nib.save(nib.MGHImage(source, affine), input_path)
                paths, metrics = correct_image(stage, input_path, label_path, mask_path, output_path,
                                           root / f"{modality}-work", modality, parameters)
                corrected_image = nib.load(output_path)
                corrected = np.asarray(corrected_image.dataobj)
                self.assertEqual(corrected.shape, (64, 64, 64))
                self.assertTrue(np.isfinite(corrected).all())
                self.assertTrue(np.isfinite(corrected_image.affine).all())
                self.assertTrue(np.isfinite(corrected_image.header.get_zooms()[:3]).all())
                self.assertTrue(all(step > 0 for step in corrected_image.header.get_zooms()[:3]))
                self.assertNotEqual(float(np.linalg.det(corrected_image.affine[:3, :3])), 0.0)
                self.assertTrue(np.allclose(corrected_image.affine, affine, rtol=0, atol=1e-5))
                self.assertTrue(np.all(corrected[mask == 0] == 0))
                self.assertAlmostEqual(float(np.median(corrected[white])), 110.0, delta=1e-4)
                cv_before = float(source[white].std() / source[white].mean())
                cv_after = float(corrected[white].std() / corrected[white].mean())
                self.assertLess(cv_after, cv_before, (modality, cv_before, cv_after))
                self.assertAlmostEqual(metrics["white_matter_median_after"], 110.0, delta=1e-4)
                raw_corrected = nib.load(paths["corrected"]).get_fdata()
                factor = nib.load(paths["factor"]).get_fdata()
                raw_field = nib.load(paths["field"]).get_fdata()
                np.testing.assert_allclose(factor, 1 / raw_field if method == "rbf" else raw_field, rtol=1e-6)
                self.assertEqual(metrics["raw_field_representation"], "divisor" if method == "rbf" else "multiplier")
                if method == "rbf":
                    self.assertTrue(metrics["reliable_support"]["fitting"]["device"].startswith("cuda:"))
                    self.assertEqual(metrics["reliable_support"]["fitting"]["soft_weight_outside_fit_support"], 0)
                    self.assertTrue(np.all((raw_field > 0) & (raw_field <= 100)))
                    self.assertEqual(metrics["reliable_support"]["fitting"]["model_revision"], "bounded-sigmoid-v1")
                    self.assertEqual(metrics["reliable_support"]["fitting"]["field_bounds"], {"minimum": 1e-6, "maximum": 100.0})
                    with np.load(root / f"{modality}-work" / "fitted-control-points.npz") as fitted:
                        self.assertTrue(np.isfinite(fitted["point_logits"]).all())
                        self.assertTrue(np.all((fitted["point_values"] > 0) & (fitted["point_values"] <= 100)))
                np.testing.assert_allclose(raw_corrected[deep], source[deep] * factor[deep], rtol=1e-5)
                self.assertGreater(float(np.max(np.abs(raw_corrected[deep] - source[deep]))), 0.1)
                support = nib.load(root / f"{modality}-work" / "fit-mask.nii.gz").get_fdata() > 0
                self.assertFalse(np.any(support & ~(white | gray)))
                self.assertEqual(metrics["reliable_support"]["implementation"], "reliable-core-v2")
                with self.assertRaises(FileExistsError):
                    correct_image(stage, input_path, label_path, mask_path, output_path,
                                  root / f"{modality}-work", modality, parameters)
                reports[modality] = (cv_before, cv_after)
            self.assertTrue(np.array_equal(np.asarray(nib.load(label_path).dataobj), saved_labels))
            self.assertLess(reports["T1"][1], reports["T1"][0])
            self.assertLess(reports["T2"][1], reports["T2"][0])


if __name__ == "__main__":
    unittest.main()

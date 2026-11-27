"""Check the corrected, full-head T2 handoff without native image executables."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from t2_pial import prepare_t2
from session_inputs import ALIGNMENT_SCHEMA, FLIRT_CONVENTION


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("numpy", "nibabel")),
                     "T2 image tests need NumPy and NiBabel")
class T2PreparationTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        import numpy as np

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.affine = np.diag([1., 1., 1., 1.])
        self.anatomy = np.zeros((6, 5, 4), dtype=np.float32)
        self.anatomy[2:4, 1:4, 1:3] = 110
        self.source = np.arange(self.anatomy.size, dtype=np.float32).reshape(self.anatomy.shape) + 50
        self.source[0, 0, 0] = 0
        self.corrected = np.where(self.source > 0, self.source * 2 + 17, 0).astype(np.float32)
        self.brain_path = self.root / "brain.finalsurfs.mgz"
        self.source_path = self.root / "native-T2.nii.gz"
        self.fixed_path = self.root / "native-T1.nii.gz"
        nib.save(nib.Nifti1Image(self.anatomy, self.affine), self.fixed_path)
        nib.save(nib.MGHImage(self.anatomy, self.affine), self.brain_path)
        nib.save(nib.Nifti1Image(self.source, self.affine), self.source_path)
        self.plan = {
            "analysis_unit_id": "sub-test_ses-01", "reference": {"series_id": "T1", "path": str(self.fixed_path)},
            "companions": [{"analysis_unit_id": "sub-test_ses-01", "series_id": "T2", "modality": "T2w",
                            "structural_status": "pass", "probe": {"facts": {"shape": [6, 5, 4]}},
                            "path": str(self.source_path)}],
        }
        self.alignment = self.root / "alignment"
        self.alignment.mkdir()
        np.savetxt(self.alignment / "transform.mat", np.eye(4))
        self.registration = {"series_id": "T2", "moving": str(self.source_path), "fixed": str(self.fixed_path),
                             "transform": "transform.mat"}
        self.write_alignment()
        self.calls = []
        self.stage = SimpleNamespace(
            root=self.root, subject="sub-test_ses-01", runtime={},
            node={"parameters": {"t2_mode": "auto", "t2_series_ids": {}, "bias": {"method": "rbf"}}},
            inputs={"alignment_manifest": (self.alignment, {"members": [{"path": "transform.mat"}]})},
            _upstream_path=lambda port: self.alignment / "alignment.json",
            path=lambda port: self.brain_path if port == "brain_finalsurfs_mgz" else self.root / port,
            output=self.output, run=self.run_transform,
        )

    def write_alignment(self, schema=ALIGNMENT_SCHEMA):
        (self.alignment / "alignment.json").write_text(json.dumps({
            "schema": schema, "transform_convention": FLIRT_CONVENTION,
            "analysis_unit_id": "sub-test_ses-01", "reference": self.plan["reference"],
            "registrations": [self.registration],
        }))

    def output(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run_transform(self, command, arguments):
        import nibabel as nib
        import numpy as np

        self.calls.append((command, arguments))
        self.assertEqual(command, "fsl:flirt")
        self.assertIn("-applyxfm", arguments)
        if "-omat" in arguments:
            self.assertIn("-usesqform", arguments)
            self.assertNotIn("-out", arguments)
            self.assertEqual(arguments[arguments.index("-in") + 1], self.fixed_path)
            np.savetxt(arguments[arguments.index("-omat") + 1], np.eye(4))
            return
        self.assertEqual(arguments[arguments.index("-in") + 1], self.source_path)
        np.testing.assert_array_equal(np.loadtxt(arguments[arguments.index("-init") + 1]), np.eye(4))
        nib.save(nib.Nifti1Image(self.source, self.affine), arguments[arguments.index("-out") + 1])

    def correct_image(self, stage, aligned, labels, mask, output, directory, modality, parameters):
        import nibabel as nib
        import numpy as np

        self.assertEqual(stage, self.stage)
        self.assertEqual(aligned, self.root / "mri/T2.prenorm.nii.gz")
        np.testing.assert_array_equal(nib.load(aligned).get_fdata(), self.source)
        self.assertEqual(modality, "T2")
        self.assertEqual(parameters, {"method": "rbf"})
        self.assertEqual(mask, self.root / "mask_mgz")
        self.assertEqual(labels, self.root / "aseg_presurf_mgz")
        nib.save(nib.MGHImage(self.corrected, self.affine), output)
        return {"factor": directory / "factor.nii.gz", "tissues": directory / "tissues.nii.gz"}, {
            "method": "rbf", "scientific_approval": "not_performed"}

    def prepare(self, correction=None):
        with patch("reference_runtime.session_plan", return_value=self.plan), \
             patch("tissue_bias.correct_image", side_effect=correction or self.correct_image):
            return prepare_t2(self.stage)

    def test_refinement_receives_corrected_full_head_without_brain_masking(self):
        import nibabel as nib
        import numpy as np

        target, metadata, paths = self.prepare()
        values = nib.load(target).get_fdata(dtype=np.float32)
        np.testing.assert_array_equal(values, self.corrected)
        self.assertFalse(np.array_equal(values, self.source))
        self.assertGreater(np.count_nonzero(values[self.anatomy < 1]), 0)
        self.assertEqual(values[0, 0, 0], 0)
        self.assertEqual(target.read_bytes(), (self.root / "mri/T2.norm.mgz").read_bytes())
        self.assertFalse(metadata["skull_stripped"])
        self.assertEqual(metadata["t2_preparation"], "bias-corrected-full-head-v1")
        self.assertEqual(metadata["scientific_approval"], "not_performed")
        self.assertIn(target, paths)
        self.assertEqual(len(self.calls), 2)

    def test_legacy_and_nonrigid_transforms_are_rejected(self):
        import numpy as np

        self.write_alignment("fastprep.session-alignment/1")
        with self.assertRaisesRegex(ValueError, "Expected a FLIRT alignment manifest"):
            self.prepare()
        self.write_alignment()
        for invalid in (np.diag([-1, 1, 1, 1]), np.diag([1, 2, 1, 1]), np.ones((3, 4))):
            np.savetxt(self.alignment / "transform.mat", invalid)
            with self.assertRaisesRegex(ValueError, "FLIRT matrix"):
                self.prepare()
        self.assertEqual(self.calls, [])

    @unittest.skipUnless(os.environ.get("FSLDIR") and (Path(os.environ["FSLDIR"]) / "bin/flirt").is_file(),
                         "native landmark test requires FSLDIR/bin/flirt")
    def test_native_flirt_landmark_on_reoriented_reconstruction_grid(self):
        import nibabel as nib
        import numpy as np

        values = np.zeros_like(self.source)
        values[2, 2, 1] = 100
        nib.save(nib.Nifti1Image(values, self.affine), self.source_path)
        recon_affine = np.array([[0, -1, 0, 5], [1, 0, 0, 0], [0, 0, -1, 3], [0, 0, 0, 1.]])
        nib.save(nib.MGHImage(np.zeros((5, 6, 4), dtype='f4'), recon_affine), self.brain_path)
        translation = np.eye(4)
        translation[0, 3] = -1
        np.savetxt(self.alignment / "transform.mat", translation)

        def native(command, arguments):
            self.assertEqual(command, "fsl:flirt")
            subprocess.run([Path(os.environ["FSLDIR"]) / "bin/flirt", *arguments], check=True,
                           env={**os.environ, "FSLOUTPUTTYPE": "NIFTI_GZ"})

        def inspect(stage, aligned, labels, mask, output, directory, modality, parameters):
            result = nib.load(aligned)
            expected = np.zeros(result.shape)
            expected[2, 2, 2] = 100
            np.testing.assert_allclose(result.get_fdata(), expected, atol=1e-4, rtol=0)
            np.testing.assert_allclose(result.affine, recon_affine, atol=1e-5)
            nib.save(nib.MGHImage(expected.astype('f4'), recon_affine), output)
            return {"factor": directory / "factor", "tissues": directory / "tissues"}, {}

        self.stage.run = native
        self.prepare(inspect)

    def test_nonfinite_corrected_t2_is_rejected(self):
        self.corrected[1, 1, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite scientific volume"):
            self.prepare()
        self.assertFalse((self.root / "mri/T2.mgz").exists())

    def test_corrected_t2_grid_mismatch_is_rejected(self):
        def mismatched(*arguments):
            self.affine[0, 3] = 2
            return self.correct_image(*arguments)

        with self.assertRaisesRegex(ValueError, "Physical grids differ"):
            self.prepare(mismatched)
        self.assertFalse((self.root / "mri/T2.mgz").exists())

    def test_bias_failure_never_falls_back_to_uncorrected_t2(self):
        def failed(*arguments):
            raise ValueError("T2 correction failed")

        with self.assertRaisesRegex(ValueError, "T2 correction failed"):
            self.prepare(failed)
        self.assertFalse((self.root / "mri/T2.mgz").exists())

    def test_absent_t2_keeps_t1_only_mode(self):
        self.plan["companions"] = []
        target, metadata, paths = self.prepare()
        self.assertIsNone(target)
        self.assertEqual(metadata["mode"], "T1_only")
        self.assertEqual(paths, [])
        self.assertEqual(self.calls, [])

    def test_transform_from_another_input_is_rejected(self):
        self.registration["moving"] = "another-session-T2"
        self.write_alignment()
        with self.assertRaisesRegex(ValueError, "transform source or target differs"):
            self.prepare()
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()

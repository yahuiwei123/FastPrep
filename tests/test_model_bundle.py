"""Portable model-bundle contract checks using only the standard library."""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelBundleContractTests(unittest.TestCase):
    def test_synthseg_initializers_only_import_packaged_modules(self):
        for initializer in (ROOT / 'model/synthseg/source').rglob('__init__.py'):
            for statement in ast.walk(ast.parse(initializer.read_text())):
                if isinstance(statement, ast.ImportFrom) and statement.level == 1 and statement.module is None:
                    for alias in statement.names:
                        with self.subTest(initializer=initializer, module=alias.name):
                            parent = initializer.parent
                            self.assertTrue((parent / (alias.name + '.py')).is_file()
                                            or (parent / alias.name / '__init__.py').is_file())

    def test_synthseg_recipe_does_not_disable_requested_gpu(self):
        self.assertNotEqual(self.commands['synthseg:segment']['environment'].get('CUDA_VISIBLE_DEVICES'), '-1')

    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((ROOT / "conf/models.json").read_text())
        cls.commands = cls.document["commands"]

    def test_command_schema_and_local_paths(self):
        self.assertTrue(self.commands)
        for command_id, command in self.commands.items():
            with self.subTest(command=command_id):
                self.assertEqual(set(command), {"python", "script", "files", "environment"})
                self.assertIn(command["python"], {"synthseg", "sugar", "hdbet", "helper", "bfc"})
                self.assertFalse(Path(command["script"]).is_absolute())
                self.assertIsInstance(command["files"], list)
                self.assertEqual(len(command["files"]), len(set(command["files"])))
                self.assertIsInstance(command["environment"], dict)
                for relative in [command["script"], *command["files"]]:
                    self.assertTrue(relative.startswith("${FREESURFER_HOME}/") or not Path(relative).is_absolute(), relative)
                    self.assertFalse(".." in Path(relative).parts, relative)

    def test_bundled_command_files_exist_and_resource_names_are_unique(self):
        main_owned = {"bin/run_tissue_bfc.py"}
        for command_id, command in self.commands.items():
            with self.subTest(command=command_id):
                for relative in [command["script"], *command["files"]]:
                    if relative in main_owned:
                        continue
                    if relative.startswith("${FREESURFER_HOME}/"):
                        continue
                    self.assertTrue((ROOT / relative).is_file(), relative)
                # Only filenames fetched by resource_file need unique basenames.
                patterns = {
                    "synthseg:segment": ("synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy", "synthseg_denoiser_labels_2.0.npy"),
                    "synthseg:postprocess-base": ("synthseg_segmentation_labels_2.0.npy", "synthseg_topological_classes_2.0.npy"),
                    "sugar:rigid": ("lh.sulc", "rh.sulc", "lh_Rigid_904_fsaverage6.model", "rh_Rigid_904_fsaverage6.model"),
                    "sugar:nonlinear": tuple(f"{hemi}_NoRigid_904_fsaverage{level}.model" for hemi in ("lh", "rh") for level in (3, 4, 5, 6)),
                    "hd-bet:predict-mask": ("plans.json", "hd_bet_prediction.py"),
                    "tissue-guided-bfc:gaussian-mixture": ("bfc_gauss_mixture.py",),
                    "tissue-guided-bfc:rbf-scatter": ("bfc_rbf_scatter.py",),
                }.get(command_id, ())
                for filename in patterns:
                    self.assertEqual(sum(Path(path).name == filename for path in command["files"]), 1, filename)

    def test_expected_model_operations_and_specific_assets(self):
        required = {
            "synthseg:preprocess", "synthseg:segment", "synthseg:postprocess-base",
            "sugar:rigid", "sugar:nonlinear", "hd-bet:predict-mask", "hd-bet:apply-mask",
            "tissue-guided-bfc:gaussian-mixture", "tissue-guided-bfc:rbf-scatter",
        }
        self.assertTrue(required.issubset(self.commands))
        segment_files = self.commands["synthseg:segment"]["files"]
        for filename in ("synthseg_2.0.h5", "synthseg_segmentation_labels_2.0.npy",
                         "synthseg_denoiser_labels_2.0.npy"):
            self.assertEqual(sum(Path(path).name == filename for path in segment_files), 1)
        post_files = self.commands["synthseg:postprocess-base"]["files"]
        self.assertEqual(sum(Path(path).name == "synthseg_topological_classes_2.0.npy" for path in post_files), 1)
        sugar_rigid = self.commands["sugar:rigid"]["files"]
        sugar_nonlinear = self.commands["sugar:nonlinear"]["files"]
        for hemi in ("lh", "rh"):
            self.assertIn(f"{hemi}_Rigid_904_fsaverage6.model", [Path(p).name for p in sugar_rigid])
            for level in (3, 4, 5, 6):
                filename = f"{hemi}_NoRigid_904_fsaverage{level}.model"
                self.assertIn(filename, [Path(p).name for p in sugar_nonlinear])
        hdbet_files = self.commands["hd-bet:predict-mask"]["files"]
        self.assertEqual(sum(Path(path).name == "checkpoint_final.pth" for path in hdbet_files), 1)
        bfc = self.commands["tissue-guided-bfc:gaussian-mixture"]
        self.assertEqual(bfc["script"], "bin/run_tissue_bfc.py")
        self.assertIn("bin/bfc_support.py", bfc["files"])
        self.assertIn("vendor/tissue_guided_bfc/bfc_gauss_mixture.py", bfc["files"])
        rbf = self.commands["tissue-guided-bfc:rbf-scatter"]
        self.assertEqual(rbf["python"], "bfc")
        self.assertEqual(rbf["script"], "bin/run_tissue_bfc.py")
        for filename in ("__init__.py", "__main__.py", "cli.py", "support.py", "validation.py"):
            self.assertIn("bin/tissue_bfc/" + filename, rbf["files"])
            self.assertIn("bin/tissue_bfc/" + filename, bfc["files"])
        for hemi in ("lh", "rh"):
            for kind in ("sphere", "sulc"):
                expected = f"${{FREESURFER_HOME}}/subjects/fsaverage6/surf/{hemi}.{kind}"
                self.assertIn(expected, self.commands["sugar:rigid"]["files"])
                self.assertIn(expected, self.commands["sugar:nonlinear"]["files"])

    def test_copied_runtime_wrappers_have_no_content_digest_logic(self):
        wrappers = (
            "model/synthseg/atomic.py",
            "model/synthseg/volume_stats_op.py",
            "model/sugar/run_registration.py",
            "model/hd-bet/predict_mask.py",
            "model/hd-bet/apply_mask.py",
        )
        for relative in wrappers:
            with self.subTest(wrapper=relative):
                source = (ROOT / relative).read_text()
                for marker in ("hashlib", "sha256", "hexdigest", "checksum"):
                    self.assertNotIn(marker, source)

    def test_profiles_and_environment_recipes_are_present(self):
        for profile in ("synthseg", "sugar", "hdbet", "helper", "bfc"):
            with self.subTest(profile=profile):
                self.assertTrue((ROOT / "envs" / f"{profile}.yml").is_file())
                self.assertTrue((ROOT / "envs" / f"{profile}-requirements.txt").is_file())


if __name__ == "__main__":
    unittest.main()

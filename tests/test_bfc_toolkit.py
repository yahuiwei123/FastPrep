"""Default wiring, standalone packaging and CUDA fail-closed regression checks."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))


class ToolkitContractTests(unittest.TestCase):
    def test_t1_t2_defaults_and_explicit_gaussian_selection(self):
        from prepare import select_bfc_method

        graph = json.loads((ROOT / "conf/workflow.json").read_text())
        nodes = {node["node_id"]: node for node in graph["nodes"]}
        parameters = [nodes["anat_tissue_bias_correct"]["parameters"], nodes["anat_pial_surface"]["parameters"]["bias"]]
        for selected in parameters:
            self.assertEqual(selected["method"], "rbf")
            self.assertEqual(selected["k_neighbors"], selected["n_points"])
        for identity in ("anat_tissue_bias_correct", "anat_pial_surface"):
            commands = nodes[identity].get("commands", nodes[identity].get("dependencies"))
            self.assertEqual(commands.count("tissue-guided-bfc:rbf-scatter"), 1)
            self.assertEqual(commands.count("tissue-guided-bfc:gaussian-mixture"), 1)
        self.assertIn("device = 'cuda:0'", (ROOT / "nextflow.config").read_text())
        select_bfc_method(graph, "gaussian")
        self.assertTrue(all(selected["method"] == "gaussian" for selected in parameters))
        with self.assertRaises(ValueError):
            select_bfc_method(graph, "auto")

    def test_standalone_archive_is_small_complete_and_refuses_overwrite(self):
        from package_bfc import package

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bfc.tar.gz"
            package(output)
            with self.assertRaises(FileExistsError):
                package(output)
            with tarfile.open(output) as archive:
                names = archive.getnames()
            self.assertTrue(any(name.endswith("/bin/tissue_bfc/cli.py") for name in names))
            self.assertTrue(any(name.endswith("/bfc_rbf_scatter.py") for name in names))
            self.assertTrue(any(name.endswith("/bfc_gauss_mixture.py") for name in names))
            self.assertFalse(any("__pycache__" in name or name.endswith("runtime.json") for name in names))
            self.assertLess(output.stat().st_size, 200000)


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("scipy"), "CUDA checks require PyTorch and SciPy")
class DeviceTests(unittest.TestCase):
    def test_explicit_gpu_selection_and_unavailable_device_never_fall_back(self):
        from tissue_bfc.support import cuda_device

        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(cuda_device("1"), "cuda:1")
            self.assertEqual(cuda_device("cuda"), "cuda:0")
            self.assertEqual(cuda_device("cuda:1"), "cuda:1")
            with self.assertRaises(RuntimeError):
                cuda_device("cuda:2")
            for device in ("cpu", "auto", "cuda:-1"):
                with self.assertRaises(ValueError):
                    cuda_device(device)
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                cuda_device("cuda:0")

    def test_default_cli_is_rbf_and_fails_before_creating_outputs_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output"
            arguments = [sys.executable, "-B", str(ROOT / "bin/run_tissue_bfc.py")]
            for name in ("input_img", "input_lab", "input_msk"):
                arguments += ["--" + name, str(Path(directory) / "missing.nii.gz")]
            arguments += ["--output_img", str(target / "corrected.nii.gz"), "--output_bias", str(target / "field.nii.gz")]
            result = subprocess.run(arguments, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CUDA device is unavailable", result.stderr)
            self.assertFalse(target.exists())


MODULES = ("numpy", "scipy", "nibabel", "SimpleITK", "sklearn")
@unittest.skipUnless(all(importlib.util.find_spec(name) for name in MODULES), "Standalone smoke needs scientific Python")
class StandaloneTests(unittest.TestCase):
    def test_extracted_package_runs_without_fastprep_runtime_or_external_source_path(self):
        import nibabel as nib
        import numpy as np
        from package_bfc import package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "toolkit.tar.gz"
            package(output)
            with tarfile.open(output) as archive:
                archive.extractall(root, filter="data")
            toolkit = next(root.glob("fastprep-bfc-*"))
            self.assertFalse((toolkit / "bin/reference_runtime.py").exists())
            labels = np.zeros((32, 32, 32), dtype=np.uint8)
            labels[2:15, 2:30, 2:30] = 2
            labels[17:30, 2:30, 2:30] = 3
            mask = np.ones_like(labels)
            image = np.where(labels == 2, 60, np.where(labels == 3, 110, 20)).astype(np.float32)
            arguments = [sys.executable, "-B", str(toolkit / "bin/run_tissue_bfc.py"), "--method", "gaussian", "--bias_max_iter", "2"]
            for name, values in (("input_img", image), ("input_lab", labels), ("input_msk", mask)):
                path = root / f"{name}.nii.gz"
                nib.save(nib.Nifti1Image(values, np.eye(4)), path)
                arguments += ["--" + name, str(path)]
            corrected = root / "result/corrected.nii.gz"
            arguments += ["--output_img", str(corrected), "--output_bias", str(root / "result/field.nii.gz")]
            result = subprocess.run(arguments, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            np.testing.assert_allclose(nib.load(corrected).get_fdata(), image, rtol=1e-5, atol=1e-4)
            report = json.loads((root / "result/support.json").read_text())
            self.assertEqual(report["method"], "gaussian")
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["toolkit_version"], "1.1.0")


if __name__ == "__main__":
    unittest.main()

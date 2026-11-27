"""SUGAR device formats and explicit CUDA fail-fast behavior."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


class SugarDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ("numpy", "nibabel")):
            raise unittest.SkipTest("NumPy and NiBabel required")
        spec = importlib.util.spec_from_file_location("sugar_device_test", ROOT / "model/sugar/run_registration.py")
        cls.wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.wrapper)

    def torch(self, device_count=1):
        module = MagicMock()
        module.__version__ = "test"
        module.device.side_effect = lambda name: name
        module.cuda.is_available.return_value = device_count > 0
        module.cuda.device_count.return_value = device_count
        return module

    def test_cuda_and_numeric_formats_select_visible_device(self):
        module = self.torch(device_count=3)
        with patch.dict(sys.modules, {"torch": module}):
            for name, expected in (("cuda", "cuda:0"), ("0", "cuda:0"), ("cuda:0", "cuda:0"),
                                   ("2", "cuda:2"), ("cuda:2", "cuda:2")):
                with self.subTest(name=name):
                    self.assertEqual(self.wrapper.select_device(name), expected)

    def test_cpu_does_not_query_cuda(self):
        module = self.torch()
        with patch.dict(sys.modules, {"torch": module}):
            self.assertEqual(self.wrapper.select_device("cpu"), "cpu")
        module.cuda.is_available.assert_not_called()
        module.cuda.device_count.assert_not_called()

    def test_auto_retains_gpu_preference_and_cpu_fallback(self):
        for count, expected in ((0, "cpu"), (1, "cuda:0")):
            with self.subTest(count=count), patch.dict(sys.modules, {"torch": self.torch(count)}):
                self.assertEqual(self.wrapper.select_device("auto"), expected)

    def test_explicit_cuda_never_falls_back(self):
        module = self.torch(device_count=0)
        with patch.dict(sys.modules, {"torch": module}):
            for name in ("cuda", "cuda:0", "0"):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "PyTorch sees no GPU"):
                    self.wrapper.select_device(name)
        module.device.assert_not_called()

    def test_out_of_range_indices_are_rejected(self):
        with patch.dict(sys.modules, {"torch": self.torch()}):
            for name in ("1", "cuda:1"):
                with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, "CUDA device 1 was requested"):
                    self.wrapper.select_device(name)

    def test_negative_indices_are_rejected(self):
        with patch.dict(sys.modules, {"torch": self.torch()}):
            for name in ("-1", "cuda:-1"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "non-negative"):
                    self.wrapper.select_device(name)

    def test_invalid_formats_are_rejected(self):
        with patch.dict(sys.modules, {"torch": self.torch()}):
            for name in ("", "cuda:", "cuda:one", "cuda:0:1", "gpu", "mps"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "device must be"):
                    self.wrapper.select_device(name)

    def test_outer_gpu_visibility_is_not_reinterpreted(self):
        with patch.dict(sys.modules, {"torch": self.torch()}), \
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-selected-physical-device"}):
            self.assertEqual(self.wrapper.select_device("cuda:0"), "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-selected-physical-device")
            with self.assertRaisesRegex(RuntimeError, "CUDA device 3 was requested"):
                self.wrapper.select_device("cuda:3")

    def runtime_modules(self):
        device = MagicMock()
        device.type = "cuda"
        device.__str__.return_value = "cuda:0"
        module = self.torch()
        module.ops.torch_scatter.cuda_version.return_value = 12080
        scatter = MagicMock()
        scatter.__version__ = "test"
        scatter.scatter_add.return_value.item.return_value = 3.0
        ops = MagicMock()
        ops.knn_points.return_value.idx.device = device
        ops.knn_points.return_value.dists.device = device
        ops.knn_points.return_value.idx.item.return_value = 0
        ops.knn_points.return_value.dists.item.return_value = 0.0
        pytorch3d = MagicMock()
        pytorch3d.__version__ = "test"
        return device, {"torch": module, "torch_scatter": scatter,
                        "pytorch3d": pytorch3d, "pytorch3d.ops": ops}

    def test_runtime_verifies_pytorch3d_cuda_kernel(self):
        device, modules = self.runtime_modules()
        with patch.dict(sys.modules, modules):
            runtime = self.wrapper.validate_runtime(device)
        self.assertTrue(runtime["pytorch3d_cuda_knn_verified"])
        modules["pytorch3d.ops"].knn_points.assert_called_once()

    def test_cpu_only_pytorch3d_fails_preflight(self):
        device, modules = self.runtime_modules()
        modules["pytorch3d.ops"].knn_points.side_effect = RuntimeError("Not compiled with GPU support")
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "PyTorch3D CUDA KNN preflight failed.*Not compiled"):
                self.wrapper.validate_runtime(device)

    def test_pytorch3d_cpu_result_is_rejected(self):
        device, modules = self.runtime_modules()
        modules["pytorch3d.ops"].knn_points.return_value.idx.device = "cpu"
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "unexpected PyTorch3D CUDA KNN result or device"):
                self.wrapper.validate_runtime(device)

    def test_pytorch3d_wrong_result_is_rejected(self):
        device, modules = self.runtime_modules()
        modules["pytorch3d.ops"].knn_points.return_value.idx.item.return_value = 1
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "unexpected PyTorch3D CUDA KNN result or device"):
                self.wrapper.validate_runtime(device)


if __name__ == "__main__":
    unittest.main()

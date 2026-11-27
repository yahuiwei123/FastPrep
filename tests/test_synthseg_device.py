"""Device selection and fail-fast checks without requiring GPU hardware."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


class SynthSegDeviceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('numpy', 'nibabel')):
            raise unittest.SkipTest('NumPy and NiBabel required')
        spec = importlib.util.spec_from_file_location('synthseg_atomic_test', ROOT / 'model/synthseg/atomic.py')
        cls.wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.wrapper)

    def tensorflow(self, device_count=1, convolution_device='/device:GPU:0'):
        module = MagicMock()
        module.__version__ = 'test'
        module.config.list_physical_devices.return_value = [f'gpu-{index}' for index in range(device_count)]
        module.config.experimental.get_device_details.return_value = {'device_name': 'test GPU'}
        module.nn.conv3d.return_value.device = convolution_device
        return module

    def test_cpu_explicitly_hides_gpus(self):
        module = self.tensorflow()
        with patch.dict(sys.modules, {'tensorflow': module}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-selected'}):
            _, runtime = self.wrapper.configure_tensorflow('cpu', 2)
            self.assertEqual(runtime['device'], 'cpu')
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '-1')
        module.config.set_visible_devices.assert_called_once_with([], 'GPU')
        module.nn.conv3d.assert_not_called()

    def test_cuda_preserves_outer_visibility_and_verifies_convolution(self):
        module = self.tensorflow()
        with patch.dict(sys.modules, {'tensorflow': module}), patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': 'GPU-selected'}):
            _, runtime = self.wrapper.configure_tensorflow('cuda', 4)
            self.assertEqual(runtime['device'], 'cuda:0')
            self.assertEqual(runtime['cuda_visible_devices'], 'GPU-selected')
        module.config.set_visible_devices.assert_called_once_with(['gpu-0'], 'GPU')
        module.config.experimental.set_memory_growth.assert_called_once_with('gpu-0', True)
        module.nn.conv3d.return_value.numpy.assert_called_once()

    def test_visible_gpu_index_is_respected(self):
        module = self.tensorflow(device_count=2)
        with patch.dict(sys.modules, {'tensorflow': module}):
            _, runtime = self.wrapper.configure_tensorflow('cuda:1', 4)
        self.assertEqual(runtime['device'], 'cuda:1')
        module.config.set_visible_devices.assert_called_once_with(['gpu-1'], 'GPU')

    def test_explicit_gpu_never_falls_back_when_unavailable(self):
        module = self.tensorflow(device_count=0)
        with patch.dict(sys.modules, {'tensorflow': module}):
            with self.assertRaisesRegex(RuntimeError, 'no CPU fallback'):
                self.wrapper.configure_tensorflow('cuda', 4)

    def test_out_of_range_gpu_is_rejected(self):
        module = self.tensorflow()
        with patch.dict(sys.modules, {'tensorflow': module}):
            with self.assertRaisesRegex(RuntimeError, 'GPU 2 requested'):
                self.wrapper.configure_tensorflow('2', 4)

    def test_cpu_convolution_is_rejected_for_gpu_request(self):
        module = self.tensorflow(convolution_device='/device:CPU:0')
        with patch.dict(sys.modules, {'tensorflow': module}):
            with self.assertRaisesRegex(RuntimeError, 'did not execute on GPU'):
                self.wrapper.configure_tensorflow('cuda', 4)

    def test_invalid_device_and_thread_count_are_rejected(self):
        for device in ('-1', 'cuda:-1', 'invalid'):
            with self.subTest(device=device), self.assertRaises(ValueError):
                self.wrapper.configure_tensorflow(device, 4)
        with self.assertRaisesRegex(ValueError, 'threads must be positive'):
            self.wrapper.configure_tensorflow('cpu', 0)

    def test_sequential_flip_preserves_spatial_axis_channels_and_average(self):
        import numpy as np

        image = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4, 1)
        original = np.arange(72, dtype=np.float32).reshape(1, 2, 3, 4, 3)
        flipped = original * 0.25 + 7
        indices = np.array([0, 2, 1])
        net = MagicMock()
        net.predict.side_effect = [original.copy(), flipped.copy()]
        expected = 0.5 * (original + np.flip(flipped, axis=1)[..., indices])
        actual = self.wrapper.predict_sequential_flip(net, image, indices)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(net.predict.call_args_list[0].args[0], image)
        np.testing.assert_array_equal(net.predict.call_args_list[1].args[0], np.flip(image, axis=1))

    def test_sequential_flip_rejects_mismatched_channels(self):
        import numpy as np

        net = MagicMock()
        net.predict.return_value = np.zeros((1, 2, 2, 2, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'Invalid posterior shape'):
            self.wrapper.predict_sequential_flip(net, np.zeros((1, 2, 2, 2, 1)), [0, 1])


if __name__ == '__main__':
    unittest.main()

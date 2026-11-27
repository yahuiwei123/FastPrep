import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'bin'), str(ROOT / 'vendor')]
from qc_runtime import Evidence, compact_labels

try:
    import nibabel as nib
    import numpy as np
    from PIL import Image
    from neuroqc import rendering
except ImportError:
    nib = None


@unittest.skipIf(nib is None, 'requires QC scientific Python')
class FastVolumeTests(unittest.TestCase):
    def test_mapping_codes_layouts_and_sparse_extremes(self):
        for values, dtype in (([0, 2, 9, 2], '>i2'), ([7, 7, 8, 10], np.uint8),
                              ([-2, 0, 505000, 8], np.int64), ([-3, 0, 5, -3], np.float32),
                              ([0, 0, 0, 0], np.uint8), ([0, 9, 1, 5], '>f4'),
                              ([2**63 + 1, 0, 2**63+2, 3], np.uint64)):
            data = np.array(values, dtype=dtype).reshape(2, 2, 1).T
            present = np.unique(data)
            expected = np.zeros(data.shape, dtype=np.uint8)
            for code, value in enumerate(present[present != 0], 1):
                expected[data == value] = code
            actual = compact_labels(data, present)
            self.assertEqual(actual.dtype, expected.dtype)
            np.testing.assert_array_equal(actual, expected)
        codes = np.arange(256)
        np.testing.assert_array_equal(compact_labels(codes, codes), codes)
        with self.assertRaisesRegex(ValueError, '255'):
            compact_labels(codes + 1, codes + 1)

    def test_source_contours_cached_once_with_identical_pixels_and_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image, labels = root / 'src.nii.gz', root / 'labels.nii.gz'
            raw = np.arange(8*10*26, dtype=np.int16).reshape(8, 10, 26)
            affine = np.diag([-1.2, 1.7, 2.0, 1.0])
            nib.save(nib.Nifti1Image(raw, affine), image)
            selected = np.zeros_like(raw)
            selected[:4, 4:, 6:18] = 1
            nib.save(nib.Nifti1Image(selected, affine), labels)
            out = [root / (name + '.png') for name in ('source', 'label1', 'empty')]
            with patch.object(rendering, 'load_volume', wraps=rendering.load_volume) as loads, \
                    patch.object(rendering, 'select_slices', wraps=rendering.select_slices) as slices, \
                    patch.object(rendering, 'limits', wraps=rendering.limits) as limits:
                result = rendering.label_montages(image, labels, out[0], [(1, out[1]), (3, out[2])], ['test-batch'])
                self.assertEqual(loads.call_count, 2)
                self.assertEqual((slices.call_count, limits.call_count), (1, 1))
            self.assertEqual(result, {'completed': list(map(str, out)), 'failed': {}})
            for idx, actual in enumerate(out):
                reference = root / ('ref-' + actual.name)
                old = (rendering.render_image(image, reference) if idx == 0 else
                       rendering.render_contours(image, labels, float((1, 3)[idx-1]), reference))
                new = json.loads(Path(str(actual) + '.json').read_text())
                del new['command'], new['output'], old['output']
                self.assertEqual(new, old)
                with Image.open(actual) as left, Image.open(reference) as right:
                    self.assertEqual(left.size, right.size)
                    self.assertEqual(left.tobytes(), right.tobytes())
            nib.save(nib.Nifti1Image(selected, np.eye(4)), labels)
            bad = root / 'bad-grid.png'
            result = rendering.label_montages(image, labels, None, [(1, bad)], ['test-batch'])
            self.assertIn(str(bad), result['failed'])
            self.assertFalse(bad.exists())

    def test_disabled_views_do_not_launch_renderer(self):
        evidence = Evidence.__new__(Evidence)
        evidence.root = Path('qc_result')
        evidence._source_anatomy = {}
        evidence.visual_contract = {'png_files': []}
        with patch.object(evidence, 'tool', side_effect=AssertionError('unexpected process')):
            evidence.label_views('port', Path('image'), Path('labels'), {'1': {}})

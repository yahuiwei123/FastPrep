"""Exercise physical FOV boundaries and conform geometry provenance."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from conform_fov import motioncor_fov_policy, record_conform_output


class ConformFovTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def image(self, shape, spacing, name='input.mgz'):
        path = self.root / name
        affine = np.diag([*spacing, 1.])
        affine[:3, 3] = [-120, -130, -100]
        nib.save(nib.MGHImage(np.zeros(shape, dtype=np.uint8), affine), path)
        return path

    def test_physical_fov_not_voxel_count(self):
        for shape, spacing, forced in [
            ((256, 8, 8), (1, 1, 1), False),
            ((257, 8, 8), (1, 1, 1), True),
            ((200, 8, 8), (1.5, 1, 1), True),
            ((300, 8, 8), (.8, 1, 1), False),
            ((8, 8, 260), (1, 1, 1), True),
        ]:
            with self.subTest(shape=shape, spacing=spacing):
                decision = motioncor_fov_policy([self.image(shape, spacing)])
                self.assertEqual(bool(decision['flags']), forced)
                json.dumps(decision, allow_nan=False)  # Native MGZ integer types must be serialized.

    def test_rejects_wrong_forced_grid_and_nonfinite_output(self):
        decision = motioncor_fov_policy([self.image((300, 8, 8), (1, 1, 1))])
        with self.assertRaisesRegex(ValueError, 'expected 256 mm grid'):
            record_conform_output(decision, self.image((8, 8, 8), (1, 1, 1), 'wrong.mgz'))
        output = self.root / 'bad.mgz'
        nib.save(nib.MGHImage(np.full((8, 8, 8), np.nan, dtype=np.float32), np.eye(4)), output)
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            record_conform_output(decision, output)

    def test_voxel_mapping_preserves_world_coordinates(self):
        source = self.image((260, 8, 8), (1, 1, 1))
        decision = motioncor_fov_policy([source])
        affine = np.array([[-1., 0, 0, 128], [0, 0, 1., -128], [0, -1., 0, 128], [0, 0, 0, 1]])
        output = self.root / 'orig.mgz'
        nib.save(nib.MGHImage(np.zeros((256, 256, 256), dtype=np.uint8), affine), output)
        report = record_conform_output(decision, output)
        voxel = np.array([100, 4, 5, 1])
        mapped = np.asarray(report['input_voxel_to_output_voxel'][0]) @ voxel
        np.testing.assert_allclose(affine @ mapped, nib.load(source).affine @ voxel)
        self.assertEqual(report['coverage_review'], 'pending')
        json.dumps(report, allow_nan=False)


if __name__ == '__main__':
    unittest.main()

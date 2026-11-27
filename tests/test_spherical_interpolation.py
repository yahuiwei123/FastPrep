import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('numpy', 'nibabel', 'scipy')),
                     'Scientific Python required')
class SphericalInterpolationTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        import numpy as np

        self.xyz, self.faces = nib.freesurfer.read_geometry(
            ROOT / 'model/sugar/source/utils/auxi_data/fsaverage2.sphere')
        self.xyz /= np.linalg.norm(self.xyz, axis=1, keepdims=True)

    def test_identity_and_rotation_on_random_directions_and_exact_edges(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from spherical_interpolation import interpolate

        random = np.random.default_rng(2026).normal(size=(2000, 3))
        edges = self.xyz[self.faces[:, :2]].sum(axis=1)
        targets = np.concatenate([random, self.xyz, edges])
        targets /= np.linalg.norm(targets, axis=1, keepdims=True)
        matrix = Rotation.from_euler('xyz', [13, -21, 37], degrees=True).as_matrix()
        output, report = interpolate(self.xyz, self.faces, targets, self.xyz @ matrix.T)
        output /= np.linalg.norm(output, axis=1, keepdims=True)
        np.testing.assert_allclose(output, targets @ matrix.T, atol=1e-10)
        self.assertEqual(report['unmatched_samples'], 0)
        self.assertGreaterEqual(report['minimum_weight'], 0)

    def test_actual_triangle_weights_and_continuity_across_shared_edge(self):
        import numpy as np
        from spherical_interpolation import interpolate

        a, b, c = self.faces[17]
        values = np.random.default_rng(42).normal(size=len(self.xyz))
        weights = np.array([.21, .32, .47])
        target = weights @ self.xyz[[a, b, c]]
        actual, _ = interpolate(self.xyz, self.faces, target[None], values)
        np.testing.assert_allclose(actual[0], weights @ values[[a, b, c]], atol=1e-10)
        midpoint = (self.xyz[a] + self.xyz[b]) / 2
        normal = np.cross(self.xyz[a], self.xyz[b])
        targets = np.stack([midpoint + 1e-9 * normal, midpoint - 1e-9 * normal])
        actual, _ = interpolate(self.xyz, self.faces, targets, values)
        np.testing.assert_allclose(actual, (values[a] + values[b]) / 2, atol=1e-7)

    def test_invalid_map_and_missing_containment_fail_instead_of_nearest_fallback(self):
        from spherical_interpolation import interpolate

        with self.assertRaisesRegex(ValueError, 'orientation'):
            interpolate(-self.xyz, self.faces, self.xyz, self.xyz)
        with self.assertRaisesRegex(ValueError, 'containing'):
            interpolate(self.xyz, self.faces, self.xyz, self.xyz, max_steps=0)


if __name__ == '__main__':
    unittest.main()

"""Regression tests for accepting measured defects while preserving real artifacts."""

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'report_only' / 'bin'))
import spherical_geometry as reporting
from reference_runtime import orient_initialized_surface

spec = importlib.util.spec_from_file_location('strict_geometry', ROOT / 'bin' / 'spherical_geometry.py')
strict_geometry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strict_geometry)


class ReportOnlyTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / 'model' / 'sugar' / 'source' / 'utils' / 'auxi_data' / 'fsaverage2.sphere'
        xyz, self.faces = nib.freesurfer.read_geometry(source)
        self.xyz = (xyz * (100 / np.linalg.norm(xyz, axis=1, keepdims=True))).astype(np.float32)

    def test_report_only_retains_folded_sphere_and_measures_written_faces(self):
        folded = self.xyz.copy()
        folded[:, 0] *= -1
        with self.assertRaises(ValueError):
            strict_geometry.repair(folded, self.faces)
        result, metrics = reporting.repair(folded, self.faces)
        self.assertIn('quality_warning', metrics)
        np.testing.assert_array_equal(result, folded)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lh.sphere.reg'
            written = reporting.save_checked(path, result, self.faces)
            xyz, faces = nib.freesurfer.read_geometry(path)
            np.testing.assert_array_equal(faces, self.faces)
            np.testing.assert_array_equal(xyz, folded)
            self.assertEqual(written['negative_faces'], len(self.faces))
            self.assertEqual(metrics['after'], written)

    def test_clean_sphere_is_preserved_without_warning(self):
        result, metrics = reporting.repair(self.xyz, self.faces)
        self.assertNotIn('quality_warning', metrics)
        np.testing.assert_array_equal(result, self.xyz)

    def test_corrupt_coordinates_and_face_indices_still_fail(self):
        bad = self.xyz.copy()
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            reporting.repair(bad, self.faces)
        for invalid in (-1, len(self.xyz)):
            faces = self.faces.copy()
            faces[0, 0] = invalid
            with self.assertRaises(ValueError):
                reporting.repair(self.xyz, faces)

    def test_changed_reference_triangle_identity_still_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            faces = self.faces.copy()
            faces[[0, 1]] = faces[[1, 0]]
            nib.freesurfer.write_geometry(folder / 'sphere', self.xyz, self.faces)
            nib.freesurfer.write_geometry(folder / 'inflated', self.xyz, faces)
            with self.assertRaisesRegex(ValueError, 'triangle identity'):
                reporting.prepare_file(folder / 'sphere', folder / 'out', topology_reference=folder / 'inflated')
            self.assertFalse((folder / 'out').exists())

    def test_winding_repair_preserves_native_footer_verbatim(self):
        faces = self.faces.copy()
        faces[2] = faces[2, [0, 2, 1]]
        footer = struct.pack('>iii', 2, 0, 20) + (
            'valid = 1\nfilename = orig.mgz\nvolume = 256 256 256\n'
            'voxelsize = 1 1 1\nxras = -1 0 0\nyras = 0 0 -1.00000011920929\n'
            'zras = 0 1 0\ncras = -9.516181945800781 -2.612754821777344 6.568519592285156\n'
        ).encode()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'lh.orig'
            nib.freesurfer.write_geometry(path, self.xyz, faces)
            with path.open('ab') as stream:
                stream.write(footer)
            report = orient_initialized_surface(path)
            self.assertEqual(report['reversed_face_rows'], [2])
            self.assertTrue(path.read_bytes().endswith(footer))
            xyz, read_faces = nib.freesurfer.read_geometry(path)
            np.testing.assert_array_equal(xyz, self.xyz)
            np.testing.assert_array_equal(read_faces, self.faces)


if __name__ == '__main__':
    unittest.main()

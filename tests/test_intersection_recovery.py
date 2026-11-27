"""Bounded initialized-mesh recovery must retain geometry and hard gates."""

from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from intersection_recovery import fair_coordinates
from reference_runtime import repair_initialized_intersections
from test_topology_recovery import Stage, checks


SCIENCE_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "scipy"))


@unittest.skipUnless(SCIENCE_AVAILABLE and importlib.util.find_spec('nibabel'), 'Requires scientific Python')
class NativeSurfaceSerializationTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        import nibabel as nib
        import numpy as np

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'rh.native'
        self.points = np.eye(4, 3)
        self.faces = np.array([[0, 1, 2], [0, 2, 3], [1, 3, 2], [0, 3, 1]])
        meta = dict(head=[2, 0, 20], valid='1', filename='test', volume=[256]*3,
                    voxelsize=[1]*3, xras=[-1, 0, 0], yras=[0, 0, -1],
                    zras=[0, 1, 0], cras=[0]*3)
        nib.freesurfer.write_geometry(self.source, self.points, self.faces, volume_info=meta)
        prefix, _, _ = self.source.read_bytes().partition(b'cras')
        self.source.write_bytes(prefix + b'cras = 2.037284851074219 23.18820190429688 -0.5997848510742188\n')

    def test_coordinate_write_preserves_native_footer_and_indices_exactly(self):
        import nibabel as nib
        import numpy as np
        from intersection_recovery import write_coordinates_like

        original = self.source.read_bytes()
        reference = nib.freesurfer.read_geometry(self.source, read_metadata=True)
        points = reference[0].copy()
        points[0, 1] += .125
        target = self.root / 'rh.recovered'
        write_coordinates_like(self.source, target, points)
        actual = nib.freesurfer.read_geometry(target, read_metadata=True)
        np.testing.assert_array_equal(actual[0], points)
        np.testing.assert_array_equal(actual[1], reference[1])
        for key in ('volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras'):
            np.testing.assert_array_equal(actual[2][key], reference[2][key])
        self.assertEqual(original, self.source.read_bytes())
        self.assertEqual(original[original.index(b'valid'):], target.read_bytes()[target.read_bytes().index(b'valid'):])
        with self.assertRaises(FileExistsError):
            write_coordinates_like(self.source, self.source, points)
        self.assertEqual(original, self.source.read_bytes())

    def test_lossy_footer_is_rejected_even_when_difference_is_tiny(self):
        import nibabel as nib
        import numpy as np
        from intersection_recovery import validate_candidate

        xyz, faces, meta = nib.freesurfer.read_geometry(self.source, read_metadata=True)
        target = self.root / 'rh.lossy'
        nib.freesurfer.write_geometry(target, xyz, faces, volume_info=meta)
        with self.assertRaisesRegex(ValueError, 'volume geometry: cras'):
            validate_candidate(xyz, faces, meta, target, np.ones(len(xyz), dtype=bool), 2.5)

    def test_invalid_coordinates_and_truncated_source_never_publish(self):
        import numpy as np
        from intersection_recovery import write_coordinates_like

        target = self.root / 'rh.bad'
        for coordinates in (self.points[:-1], self.points * np.nan, self.points + 1e100):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                write_coordinates_like(self.source, target, coordinates)
            self.assertFalse(target.exists())
        self.source.write_bytes(b'\xff\xff\xfecomment\n\n')
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            write_coordinates_like(self.source, target, self.points)
        self.assertFalse(target.exists())


class IntersectionRecoveryTests(unittest.TestCase):
    def test_invalid_option_rejected_before_commands(self):
        stage = Stage(Path("unused"))
        for value in (0, 1, "true", None):
            stage.node["parameters"]["local_fairing_recovery"] = value
            with self.assertRaisesRegex(ValueError, "boolean"):
                repair_initialized_intersections(stage, "lh")
        self.assertEqual(stage.calls, [])

    def test_no_implicit_recovery(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            stage = Stage(Path(directory))
            with patch("reference_runtime.surface_repair_checks", side_effect=[checks(), checks(10), checks(4)]), \
                    patch("reference_runtime.surface_repair_geometry", return_value={}), \
                    patch("intersection_recovery.recover_intersections") as recovery:
                with self.assertRaisesRegex(ValueError, "zero self-intersections before placement"):
                    repair_initialized_intersections(stage, "lh")
                recovery.assert_not_called()
            self.assertFalse(stage.hemi("orig_premesh_surf", "lh").exists())

    def test_bad_topology_never_enters_fairing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            stage = Stage(Path(directory))
            stage.node["parameters"]["local_fairing_recovery"] = True
            with patch("reference_runtime.surface_repair_checks", side_effect=[checks(), checks(10), checks(4, euler=0)]), \
                    patch("intersection_recovery.recover_intersections") as recovery:
                with self.assertRaisesRegex(ValueError, "Euler=2 and zero defects"):
                    repair_initialized_intersections(stage, "lh")
                recovery.assert_not_called()

    def test_failed_recovery_never_publishes_premesh(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            stage = Stage(Path(directory))
            stage.node["parameters"]["local_fairing_recovery"] = True
            with patch("reference_runtime.surface_repair_checks", side_effect=[checks(), checks(10), checks(4)]), \
                    patch("intersection_recovery.recover_intersections", side_effect=ValueError("1 mm budget")):
                with self.assertRaisesRegex(ValueError, "budget"):
                    repair_initialized_intersections(stage, "lh")
            self.assertFalse(stage.hemi("orig_premesh_surf", "lh").exists())

    def test_explicit_recovery_retains_native_failure_measurement(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            stage = Stage(Path(directory))
            stage.node["parameters"]["local_fairing_recovery"] = True
            details = dict(method="test-only mock", scientific_approval="not_performed")
            with patch("reference_runtime.surface_repair_checks", side_effect=[checks(), checks(10), checks(4)]), \
                    patch("intersection_recovery.recover_intersections", return_value=(checks(), details)), \
                    patch("reference_runtime.orient_initialized_surface", return_value={}), \
                    patch("reference_runtime.surface_repair_geometry", return_value={}):
                repair_initialized_intersections(stage, "lh")
            self.assertEqual(stage.metrics["lh"]["native_before_recovery"], checks(4))
            self.assertEqual(stage.metrics["lh"]["repaired"], checks())
            self.assertEqual(stage.metrics["lh"]["local_fairing_recovery"], details)

    def grid(self, scale=1):
        import numpy as np

        coordinates = np.array([[column, row, 0] for row in range(15) for column in range(15)], dtype=float) * scale
        faces = []
        for row in range(14):
            for column in range(14):
                start = row * 15 + column
                faces.extend([[start, start + 1, start + 15], [start + 1, start + 16, start + 15]])
        seeds = np.zeros(len(coordinates), dtype=bool)
        seeds[112] = True
        return coordinates, np.array(faces), seeds

    @unittest.skipUnless(SCIENCE_AVAILABLE, "Requires NumPy and SciPy")
    def test_fixed_boundary_input_immutability_and_locality(self):
        import numpy as np

        coordinates, faces, seeds = self.grid()
        coordinates[112, 2] = 0.8
        original = coordinates.copy()
        result, selected = fair_coordinates(coordinates, faces, seeds)
        self.assertTrue(np.array_equal(coordinates, original))
        self.assertTrue(np.array_equal(result[~selected], original[~selected]))
        self.assertLess(result[112, 2], original[112, 2])
        self.assertLess(np.linalg.norm(result - original, axis=1).max(), 1)

    @unittest.skipUnless(SCIENCE_AVAILABLE, "Requires NumPy and SciPy")
    def test_displacement_budget_is_hard(self):
        coordinates, faces, seeds = self.grid()
        coordinates[112, 2] = 8
        with self.assertRaisesRegex(ValueError, "1 mm"):
            fair_coordinates(coordinates, faces, seeds)

    @unittest.skipUnless(SCIENCE_AVAILABLE, "Requires NumPy and SciPy")
    def test_no_fixed_boundary_rejected(self):
        coordinates, faces, seeds = self.grid()
        seeds[:] = True
        with self.assertRaisesRegex(ValueError, "fixed-boundary"):
            fair_coordinates(coordinates, faces, seeds)

    @unittest.skipUnless(SCIENCE_AVAILABLE, "Requires NumPy and SciPy")
    def test_invalid_or_empty_mask_rejected(self):
        coordinates, faces, seeds = self.grid()
        for invalid in (seeds[:-1], seeds.astype(float), seeds & False):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                fair_coordinates(coordinates, faces, invalid)

    @unittest.skipUnless(SCIENCE_AVAILABLE, "Requires NumPy and SciPy")
    def test_bounded_harmonic_field_escapes_large_step_without_moving_boundary(self):
        import numpy as np

        coordinates, faces, seeds = self.grid()
        coordinates[112, 2] = 8
        result, selected = fair_coordinates(coordinates, faces, seeds, rings=2,
                                            blend=1., max_displacement_mm=1., clip=True)
        self.assertLessEqual(np.linalg.norm(result - coordinates, axis=1).max(), 1. + 1e-12)
        self.assertTrue(np.array_equal(result[~selected], coordinates[~selected]))
        self.assertLess(result[112, 2], coordinates[112, 2])

    def test_recovery_budget_cannot_be_unbounded_or_nonfinite(self):
        from intersection_recovery import recover_intersections

        stage = Stage(Path('unused'))
        for value in (0, -1, 3., True, float('nan'), float('inf'), '2.5'):
            stage.node['parameters']['local_recovery_max_displacement_mm'] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'at most 2.5'):
                recover_intersections(stage, 'lh', Path('unused.orig'), None)
        self.assertEqual(stage.calls, [])


if __name__ == "__main__":
    unittest.main()

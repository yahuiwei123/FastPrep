import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))


@unittest.skipUnless(all(importlib.util.find_spec(x) for x in ('numpy', 'nibabel', 'scipy')),
                     'Scientific Python required')
class SphereGeometryTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        import numpy as np
        from spherical_geometry import require_valid, topology

        self.xyz, self.faces = nib.freesurfer.read_geometry(ROOT / 'model/sugar/source/utils/auxi_data/fsaverage2.sphere')
        self.xyz = (self.xyz / np.linalg.norm(self.xyz, axis=1, keepdims=True) * 100).astype(np.float32)
        topology(self.faces, len(self.xyz))
        require_valid(self.xyz, self.faces)

    def narrow_face(self, height):
        import numpy as np

        xyz = self.xyz.copy().astype(np.float64)
        a, b, c = self.faces[0]
        midpoint = xyz[a] + xyz[b]
        midpoint /= np.linalg.norm(midpoint)
        normal = np.cross(xyz[a], xyz[b])
        normal /= np.linalg.norm(normal)
        xyz[c] = midpoint + height * normal
        xyz[c] *= 100 / np.linalg.norm(xyz[c])
        return xyz.astype(np.float32)

    def test_valid_mesh_is_unchanged(self):
        import numpy as np
        from spherical_geometry import repair

        result, report = repair(self.xyz, self.faces)
        np.testing.assert_array_equal(result, self.xyz)
        self.assertEqual(report['changed_vertices'], 0)

    def test_single_local_fold_is_repaired_with_fixed_boundary(self):
        import numpy as np
        from spherical_geometry import inspect, repair

        broken = self.narrow_face(-1e-5)
        self.assertGreater(inspect(broken, self.faces)['negative_faces'], 0)
        original_faces = self.faces.copy()
        candidate, report = repair(broken, self.faces)
        self.assertEqual(report['after']['unsafe_faces'], 0)
        self.assertLessEqual(report['max_displacement_mm'], .5)
        outside = np.ones(len(broken), dtype=bool)
        outside[report['moved_vertex_ids']] = False
        np.testing.assert_array_equal(candidate[outside], broken[outside])
        np.testing.assert_array_equal(self.faces, original_faces)

    def test_nearly_degenerate_input_remains_valid_after_rotations_and_storage(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        from spherical_geometry import repair, save_checked

        candidate, report = repair(self.narrow_face(1e-9), self.faces)
        self.assertGreater(report['before']['unsafe_faces'], 0)
        with tempfile.TemporaryDirectory() as temporary:
            for i, matrix in enumerate(Rotation.random(12, random_state=2026).as_matrix()):
                rotated = (candidate.astype(np.float64) @ matrix.T).astype(np.float32)
                record = save_checked(Path(temporary)/f'{i}.sphere', rotated, self.faces)
                self.assertEqual(record['negative_faces'], 0)

    def test_winding_repair_is_topological_and_does_not_hide_geometric_fold(self):
        import numpy as np
        from spherical_geometry import canonical_faces, inspect, repair

        bad_faces = self.faces.copy()
        bad_faces[7] = bad_faces[7, [0, 2, 1]]
        restored, report = canonical_faces(bad_faces, len(self.xyz))
        np.testing.assert_array_equal(restored, self.faces)
        self.assertEqual(report['reversed_face_rows'], [7])
        broken = self.narrow_face(-1e-5)
        unchanged, report = canonical_faces(self.faces, len(broken))
        np.testing.assert_array_equal(unchanged, self.faces)
        self.assertGreater(inspect(broken, unchanged)['negative_faces'], 0)
        with self.assertRaisesRegex(ValueError, 'limits'):
            repair(broken, unchanged, max_faces=0)

    def test_reject_extensive_fold_boundary_and_changed_reference_connectivity(self):
        from spherical_geometry import repair, topology, prepare_file
        import nibabel as nib

        with self.assertRaisesRegex(ValueError, 'limits'):
            repair(-self.xyz, self.faces)
        with self.assertRaisesRegex(ValueError, 'closed'):
            topology(self.faces[:-1], len(self.xyz))
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            nib.freesurfer.write_geometry(folder/'sphere', self.xyz, self.faces)
            nib.freesurfer.write_geometry(folder/'reference', self.xyz, self.faces[::-1])
            with self.assertRaisesRegex(ValueError, 'identity'):
                prepare_file(folder/'sphere', folder/'output', topology_reference=folder/'reference')
            self.assertFalse((folder/'output').exists())

    def test_initialized_winding_correction_preserves_coordinates_and_failure_evidence(self):
        import numpy as np
        import nibabel as nib
        from reference_runtime import orient_initialized_surface

        faces = self.faces.copy()
        faces[5] = faces[5, [0, 2, 1]]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'lh.orig'
            # Anatomical geometry need not be spherical for winding correction.
            xyz = self.xyz * np.array([.7, .8, .6], np.float32)
            nib.freesurfer.write_geometry(path, xyz, faces)
            original_bytes = path.read_bytes()
            report = orient_initialized_surface(path)
            actual, actual_faces = nib.freesurfer.read_geometry(path)
            np.testing.assert_array_equal(actual, xyz)
            np.testing.assert_array_equal(actual_faces, self.faces)
            self.assertEqual(Path(report['original_surface']).read_bytes(), original_bytes)

    def test_pinched_spheres_are_rejected_even_with_euler_two_and_closed_edges(self):
        import numpy as np
        from spherical_geometry import topology

        count = len(self.xyz)
        # Two otherwise separate spheres share two nonadjacent vertices.
        # Vertex-graph connectivity, edge incidence and Euler alone miss this.
        shared = [0, int(np.argmin(self.xyz @ self.xyz[0]))]
        mapping = np.empty(count, dtype=int)
        other = np.ones(count, dtype=bool)
        other[shared] = False
        mapping[shared] = shared
        mapping[other] = np.arange(count, 2 * count - 2)
        faces = np.concatenate([self.faces, mapping[self.faces]])
        with self.assertRaisesRegex(ValueError, 'nonmanifold'):
            topology(faces, 2 * count - 2)

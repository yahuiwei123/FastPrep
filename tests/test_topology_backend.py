import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from topology_backend import BINDING, STAMP, STOCK, geometry_events, quick_sphere, verify_mesh


class TopologyBackendTests(unittest.TestCase):
    def test_trace_bound_to_mesh_and_full_geometry_mode(self):
        for log in ('', 'GEOM_PASS 3 0 1', 'GEOM_BIND 2 4 12\nGEOM_PASS 3 0 1',
                    'GEOM_BIND 4 4 12\nGEOM_PASS 2 0 1', 'GEOM_BIND 4 4 12\nGEOM_PASS 3 1 1'):
            with self.subTest(log=log), self.assertRaises(RuntimeError):
                geometry_events(log, 4, 4)

    def test_default_gpu_does_not_silently_use_cpu(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='-1'), self.assertRaisesRegex(ValueError, 'masked'):
            quick_sphere(SimpleNamespace(spec={'device': 'cuda:0'}), 'lh', [], None, None)

    def test_gpu_publication_requires_success_evidence_and_geometry(self):
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest('requires scientific Python')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, CUDA_VISIBLE_DEVICES='7,4'):
            root = Path(directory)
            (root / 'native').mkdir()
            (root / 'native/geometry.so').touch()
            (root / 'native/qsphere').touch(mode=0o755)
            (root / 'build-stamp.txt').write_text(STAMP)
            (root / 'python/packages/fsbindings').mkdir(parents=True)
            (root / 'python/packages/fsbindings' / BINDING).touch()
            source, target = root / 'lh.inflated.nofix', root / 'lh.qsphere.nofix'
            faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
            points = np.eye(4, 3)
            meta = dict(head=[2, 0, 20], valid='1', filename='orig.mgz', volume=[32]*3, voxelsize=[1]*3,
                        xras=[1, 0, 0], yras=[0, 1, 0], zras=[0, 0, 1], cras=[0]*3)
            nib.freesurfer.write_geometry(source, points, faces, volume_info=meta)
            trace = 'GEOM_BIND 4 4 12\nGEOM_PASS 3 0 1\n'

            def run(identity, argv, env, cwd):
                self.assertEqual(identity, 'fastprep:topology-geometry-lh')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '4')
                self.assertEqual((env['GEOM_STEP'], env['GEOM_CHECK']), ('3', '0'))
                (root / 'stderr').write_text(trace)
                nib.freesurfer.write_geometry(argv[-1], points + 1, faces, volume_info=meta)

            def command(identity):
                self.assertEqual(identity, STOCK)
                return {'argv_prefix': [str(root / 'bin/mris_sphere')], 'environment': {'FREESURFER_HOME': str(root)}}

            stage = SimpleNamespace(spec={'device': 'cuda:1', 'topology_runtime': root}, subject_dir=root,
                                    root=root, run_process=run, logs=[{'stderr': 'stderr'}], command=command)
            details = quick_sphere(stage, 'lh', [], source, target)
            self.assertEqual((details['backend'], details['geometry_passes']), ('gpu', 1))
            before = target.read_bytes()
            trace = ''
            with self.assertRaisesRegex(RuntimeError, 'evidence'):
                quick_sphere(stage, 'lh', [], source, target)
            self.assertEqual(before, target.read_bytes())
            (root / 'build-stamp.txt').write_text('other')
            with self.assertRaisesRegex(RuntimeError, 'incompatible'):
                quick_sphere(stage, 'lh', [], source, target)
            for moved, triangles, header in ((points, faces[::-1], meta), (points, faces, dict(meta, cras=[0, 0, 1])),
                                             (points, faces, dict(meta, filename='different.mgz')), (points * np.nan, faces, meta)):
                nib.freesurfer.write_geometry(target, moved, triangles, volume_info=header)
                with self.assertRaisesRegex(ValueError, 'geometry'):
                    verify_mesh(source, target)

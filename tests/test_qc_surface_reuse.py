import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))


@unittest.skipUnless(importlib.util.find_spec('surfplot'), 'requires QC rendering environment')
class SurfaceReuseTests(unittest.TestCase):
    def test_independent_cameras_and_scene_boundaries(self):
        from brainspace.mesh.mesh_creation import build_polydata
        import nibabel as nib
        import numpy as np
        from surfplot import Plot
        from neuroqc.rendering import _surface_pixels

        coords, faces = nib.freesurfer.read_geometry(ROOT / 'model/sugar/source/utils/auxi_data/fsaverage2.sphere')
        coords = coords * [1.2, .8, 1.5] + [13, -15, 40]
        mesh = build_polydata(coords, faces)
        other = build_polydata(coords * [.9, 1.4, 1.1], faces)
        feature = coords[:, 1]
        panels = [dict(hemisphere='lh', mesh=mesh, view='anterior'),
                  dict(hemisphere='lh', mesh=mesh, view='ventral'),
                  dict(hemisphere='lh', mesh=other, view='ventral'),
                  dict(hemisphere='rh', mesh=other, view='medial'),
                  dict(hemisphere='rh', mesh=other, view='medial', values=feature),
                  dict(hemisphere='rh', mesh=other, view='lateral', values=feature),
                  dict(hemisphere='rh', mesh=other, view='lateral', values=-feature)]
        def render(engine, calls):
            with patch.object(Plot, 'render', autospec=True, side_effect=Plot.render) as create:
                pixels = [item.copy() for item in _surface_pixels(panels, engine, 'coolwarm', (-100, 100), False)]
            self.assertEqual(create.call_count, calls)
            return pixels

        old, new = render('reference', len(panels)), render('reuse', 5)
        for before, after in zip(old, new, strict=True):
            self.assertEqual(after.shape, (600, 600, 3))
            self.assertTrue(after.any())
            np.testing.assert_array_equal(before, after)
        self.assertFalse(np.array_equal(new[-1], new[-2]))

    def test_reference_validation_survives_reuse(self):
        from brainspace.mesh.mesh_creation import build_polydata
        import numpy as np
        from neuroqc.rendering import _surface_pixels

        mesh = build_polydata(np.eye(3), np.array([[0, 1, 2]]))
        for options, expected in (({'hemisphere': 'bilateral'}, 'hemisphere'), ({'view': 'oblique'}, 'view'),
                                  ({'values': np.ones(2)}, 'vertices'), ({'values': np.full(3, np.nan)}, 'finite')):
            panel = dict(mesh=mesh, hemisphere='lh', view='dorsal')
            panel.update(options)
            with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                list(_surface_pixels([panel], 'reuse', 'viridis', None, False))

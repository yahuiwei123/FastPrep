import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from mni_geometry import prepare_image, assert_geometry
from mni_normalize import apply_args, bind_fireants_command, write_itk_pull


@unittest.skipUnless(importlib.util.find_spec('nibabel') and importlib.util.find_spec('SimpleITK'),
                     'NiBabel and SimpleITK required')
class GeometryTests(unittest.TestCase):
    def test_conflict_shear_and_uncoded_preserve_scaled_voxels(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for case in ('orthogonal_conflict', 'shear_with_qform', 'uncoded'):
                with self.subTest(case=case):
                    qform = np.diag([-1.2, 0.9, 1.7, 1.])
                    qform[:3, 3] = [11, -20, 7]
                    sform = qform.copy()
                    sform[0, 3] += 12
                    if case == 'shear_with_qform':
                        sform[0, 1] = 0.002
                    source = folder/(case + '.nii.gz')
                    img = nib.Nifti1Image(np.arange(120, dtype=np.int16).reshape(4, 5, 6), qform)
                    img.set_qform(qform, 0 if case == 'uncoded' else 1)
                    img.set_sform(sform, 0 if case == 'uncoded' else 1)
                    img.header.set_slope_inter(0.5, 17)
                    nib.save(img, source)
                    original_bytes = source.read_bytes()
                    before = nib.load(source)
                    output, record = prepare_image(source, folder/(case + '-prepared.nii.gz'))
                    after = assert_geometry(output)
                    self.assertEqual(source.read_bytes(), original_bytes)
                    np.testing.assert_array_equal(before.get_fdata(), after.get_fdata())
                    expected = {'orthogonal_conflict': before.get_sform(),
                                'shear_with_qform': before.get_qform(), 'uncoded': before.affine}[case]
                    np.testing.assert_allclose(after.affine, expected, atol=1e-4, rtol=0)
                    self.assertEqual(record['selected_form'], dict(orthogonal_conflict='sform',
                                     shear_with_qform='qform', uncoded='nibabel_fallback')[case])
                    self.assertTrue(record['warnings'])
                    with self.assertRaisesRegex(ValueError, 'new MNI geometry destination'):
                        prepare_image(source, output)

    def test_reject_shear_only_and_opposite_handedness(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for case in ('shear_only', 'opposite_handedness'):
                with self.subTest(case=case):
                    matrix = np.eye(4)
                    matrix[0, 1] = 0.2
                    image = nib.Nifti1Image(np.ones((3, 4, 5), np.float32), matrix)
                    if case == 'opposite_handedness':
                        image.set_qform(np.diag([-1., 1., 1., 1.]), 1)
                    source = folder/(case + '.nii.gz')
                    nib.save(image, source)
                    with self.assertRaisesRegex(ValueError, 'Sheared sform|handedness'):
                        prepare_image(source, folder/'prepared.nii.gz')
                    self.assertFalse((folder/'prepared.nii.gz').exists())

    @unittest.skipUnless(shutil.which('antsApplyTransforms'), 'ANTs required')
    def test_companion_original_fsl_frame_to_selected_physical_frame(self):
        import nibabel as nib
        import numpy as np
        import SimpleITK as sitk

        # Different origins, noncommuting rigid and header conflict. With an
        # identity FLIRT mapping the same voxel index must still be sampled.
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            xyz = np.indices((24, 26, 28))
            values = (500 + xyz[0] + 3*xyz[1] + 7*xyz[2]).astype(np.float32)
            originals, prepared = [], []
            for name, angle, origin in [('fixed', .2, [17, -3, 9]), ('moving', -.1, [-12, 5, 2])]:
                rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                                     [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
                qform = np.eye(4)
                qform[:3, :3] = rotation @ np.diag([1.2, 1.7, 2.])
                qform[:3, 3] = origin
                sform = qform.copy()
                sform[0, 1] += 0.003
                image = nib.Nifti1Image(values, sform)
                image.set_qform(qform, 1)
                image.set_sform(sform, 1)
                original = folder/(name + '.nii.gz')
                nib.save(image, original)
                work, _ = prepare_image(original, folder/(name + '-prepared.nii.gz'))
                originals.append(original)
                prepared.append(work)
            np.savetxt(folder/'flirt.mat', np.eye(4))
            write_itk_pull(originals[1], originals[0], folder/'flirt.mat', folder/'rigid.tfm',
                           moving_geometry=prepared[1], fixed_geometry=prepared[0])
            # A template-grid displacement before the noncommuting rigid
            # moves one native voxel in the fixed x direction.
            grid = sitk.ReadImage(str(prepared[0]))
            step = np.array(grid.GetDirection()).reshape(3, 3)[:, 0] * grid.GetSpacing()[0]
            field = sitk.GetImageFromArray(np.broadcast_to(step, (28, 26, 24, 3)).copy(), isVector=True)
            field.CopyInformation(grid)
            sitk.WriteImage(field, str(folder/'warp.nii.gz'))
            subprocess.run(['antsApplyTransforms', *map(str, apply_args(prepared[1], prepared[0],
                            folder/'out.nii.gz', [folder/'warp.nii.gz', folder/'rigid.tfm']))], check=True)
            result = nib.load(folder/'out.nii.gz').get_fdata()
            np.testing.assert_allclose(result[2:-2, 2:-2, 2:-2], values[3:-1, 2:-2, 2:-2], atol=0.001, rtol=0)


class BindingTests(unittest.TestCase):
    def test_selective_resume_binds_new_script_and_preserves_python_environment(self):
        command = dict(argv_prefix=['/env/python', '-B', '/old/bin/fireants_register.py'],
                       environment={'TEST': 'retained'})
        runtime = {'commands': {'fireants:register': command}}
        stage = SimpleNamespace(runtime=runtime)
        bind_fireants_command(stage)
        self.assertEqual(command['argv_prefix'][-1], '/old/bin/fireants_register.py')
        rebound = stage.runtime['commands']['fireants:register']
        self.assertEqual(rebound['argv_prefix'][:2], ['/env/python', '-B'])
        self.assertEqual(rebound['environment'], command['environment'])
        self.assertEqual(Path(rebound['argv_prefix'][-1]),
                         Path(__file__).resolve().parents[1]/'bin/fireants_register.py')

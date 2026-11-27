import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from mni_normalize import apply_args, normalize_session, write_itk_pull
from mni_template import options, template_resources
from reference_runtime import read_json, write_json


class TemplateTests(unittest.TestCase):
    def test_default_and_missing_mask(self):
        parser = argparse.ArgumentParser()
        options(parser)
        with tempfile.TemporaryDirectory() as temp:
            args = parser.parse_args(['--templateflow-home', temp])
            self.assertEqual((args.mni_template, args.mni_resolution), ('tpl-MNI152NLin2009cAsym', 1))
            folder = Path(temp)/args.mni_template
            folder.mkdir()
            (folder/(args.mni_template + '_res-01_T1w.nii.gz')).write_bytes(b'test')
            with self.assertRaisesRegex(ValueError, 'desc-brain_mask'):
                template_resources(args)


@unittest.skipUnless(shutil.which('antsApplyTransforms') and shutil.which('antsRegistration'), 'ANTs binaries required')
class NativeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('flirt'), 'FSL FLIRT required')
    def test_compare_flirt_on_different_anisotropic_grids(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            source, target = folder/'source.nii.gz', folder/'target.nii.gz'
            values = np.indices((30, 32, 28))
            source_header = np.array([[0, -1, 0, 40], [1.5, 0, 0, -7], [0, 0, 2, 3], [0, 0, 0, 1]])
            target_header = np.array([[-1.2, 0, 0, 15], [0, 0, -1, 12], [0, 1.6, 0, 1], [0, 0, 0, 1]])
            nib.save(nib.Nifti1Image((values[0]+3*values[1]+7*values[2]+3).astype('float32'), source_header), source)
            nib.save(nib.Nifti1Image(np.ones((34, 25, 30), dtype='float32'), target_header), target)
            cosine, sine = np.cos(0.1), np.sin(0.1)
            np.savetxt(folder/'fsl.mat', [[cosine,-sine,0,2], [sine,cosine,0,-3], [0,0,1,4], [0,0,0,1]])
            write_itk_pull(source, target, folder/'fsl.mat', folder/'itk.tfm')
            subprocess.run(['flirt', '-in', source, '-ref', target, '-applyxfm', '-init', folder/'fsl.mat',
                            '-interp', 'trilinear', '-out', folder/'fsl.nii.gz'], check=True)
            subprocess.run(['antsApplyTransforms', *apply_args(source, target, folder/'itk.nii.gz', [folder/'itk.tfm'])], check=True)
            first, second = [nib.load(folder/name).get_fdata() for name in ('fsl.nii.gz','itk.nii.gz')]
            from scipy.ndimage import binary_erosion
            interior = binary_erosion((first > 0) & (second > 0), iterations=2)
            self.assertGreater(interior.sum(), 1000)
            np.testing.assert_allclose(first[interior], second[interior], atol=5e-4, rtol=0)

    def test_noncommuting_pull_chain(self):
        import numpy as np
        import SimpleITK as sitk

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            depth, row, column = np.indices((24, 24, 24))
            image = sitk.GetImageFromArray((column + 7*row + 100*depth).astype('float32'))
            source = folder/'original.nii.gz'
            sitk.WriteImage(image, str(source))
            matrix = np.eye(4)
            matrix[0, 3] = 2
            np.savetxt(folder/'flirt.mat', matrix)
            rigid = folder/'rigid.tfm'
            write_itk_pull(source, source, folder/'flirt.mat', rigid)
            linear = sitk.AffineTransform(3)
            linear.SetCenter([12, 12, 12])
            linear.SetMatrix([0, -1, 0, 1, 0, 0, 0, 0, 1])
            affine = folder/'affine.tfm'
            sitk.WriteTransform(linear, str(affine))
            displacement = np.zeros((24, 24, 24, 3), np.float32)
            displacement[..., 1] = 1
            warp = folder/'warp.nii.gz'
            sitk.WriteImage(sitk.GetImageFromArray(displacement, isVector=True), str(warp))
            output = folder/'output.nii.gz'
            subprocess.run(['antsApplyTransforms', *apply_args(source, source, output, [warp, affine, rigid])], check=True)
            result = sitk.ReadImage(str(output))
            for point in [(6, 8, 9), (10, 13, 17), (15, 9, 8)]:
                expected = image[25-point[1], point[0], point[2]]
                self.assertAlmostEqual(result[point], expected, places=3)

    def test_native_registration_with_companion(self):
        import nibabel as nib
        import numpy as np
        from session_inputs import ALIGNMENT_SCHEMA, FLIRT_CONVENTION

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            uid = 'sub-01'
            xyz = np.indices((48, 48, 48))
            dist = ((xyz - np.array([21, 25, 24])[:, None, None, None])**2).sum(axis=0)
            signal = (80*np.exp(-dist/170) + 40*np.exp(-((xyz-15)**2).sum(axis=0)/30)).astype('float32')
            raw = folder/'sub-01_T1w.nii.gz'
            fixed = folder/'atlas.nii.gz'
            mask = folder/'mask.nii.gz'
            other = folder/'sub-01_T2w.nii.gz'
            for path, values in [(raw, signal), (fixed, np.roll(signal, 2, axis=0)),
                                 (mask, (dist < 350).astype('uint8')), (other, signal*2)]:
                nib.save(nib.Nifti1Image(values, np.eye(4)), path)
            def acquisition(path):
                return dict(path=str(path), series_id=path.name.split('.')[0], analysis_unit_id=uid)
            plan = dict(analysis_unit_id=uid, subject_id=uid, session_id='', reference=acquisition(raw), companions=[acquisition(other)])
            rigid = 'moving-to-T1w.mat'
            np.savetxt(folder/rigid, np.eye(4))
            manifest = dict(schema=ALIGNMENT_SCHEMA, transform_convention=FLIRT_CONVENTION, analysis_unit_id=uid,
                            reference=dict(path=str(raw), series_id='sub-01_T1w'), registrations=[dict(moving=str(other),
                            fixed=str(raw), series_id='sub-01_T2w', transform=rigid, registered='must-not-be-read.nii.gz')])
            write_json(folder/'alignment.json', manifest)
            (folder/'mni').mkdir()
            calls = []
            def run(identity, args):
                calls.append((identity, args))
                subprocess.run([identity.split(':')[1], *args], check=True,
                               env=dict(os.environ, ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS='2'))
            stage = SimpleNamespace(subject=uid, runtime=dict(session_input_plan=[plan], mni_template=dict(
                id='Synthetic', resolution=1, t1w=str(fixed), mask=str(mask))), metrics={}, output=lambda path: folder/path,
                run=run, file_record=lambda path: {'path': str(path.relative_to(folder))},
                inputs={'alignment_manifest': (folder, {'path': 'alignment.json', 'members': [{'path': rigid}]})})
            stage.node = {'parameters': {'backend': 'ants'}}
            manifest['registrations'][0]['fixed'] = str(other)
            write_json(folder/'alignment.json', manifest)
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                normalize_session(stage)
            self.assertFalse(calls)
            manifest['registrations'][0]['fixed'] = str(raw)
            write_json(folder/'alignment.json', manifest)
            result = read_json(normalize_session(stage)['mni_manifest'])
            self.assertEqual([name for name, _ in calls], ['ants:antsRegistration', *['ants:antsApplyTransforms']*2])
            for (_, args), original in zip(calls[1:], [raw, other]):
                working = Path(args[args.index('-i') + 1])
                self.assertNotEqual(working, original)
                np.testing.assert_array_equal(nib.load(working).get_fdata(), nib.load(original).get_fdata())
            self.assertEqual([row['source'] for row in result['images']], [str(raw), str(other)])
            self.assertEqual([len(row['ants_image_transform_list']) for row in result['images']], [2, 3])
            self.assertEqual(result['scientific_approval'], 'not_performed')

import copy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from fireants_register import compose_affine, cuda_device, inverse_coordinates, physical_jacobian, save_forward, save_inverse
from mni_normalize import normalize_session
from prepare import select_mni_backend
from reference_runtime import read_json, write_json


class BackendTests(unittest.TestCase):
    def test_default_and_explicit_native_graph(self):
        graph = read_json(ROOT / 'conf/workflow.json')
        original = copy.deepcopy(graph)
        node = next(node for node in graph['nodes'] if node['node_id'] == 'session_normalize_mni')
        self.assertEqual(node['parameters']['backend'], 'fireants')
        self.assertEqual(node['dependencies'], ['fireants:register', 'ants:antsApplyTransforms'])
        select_mni_backend(graph, 'ants')
        self.assertEqual(node['dependencies'], ['ants:antsRegistration', 'ants:antsApplyTransforms'])
        select_mni_backend(graph, 'fireants')
        self.assertEqual(graph, original)
        with self.assertRaises(ValueError):
            select_mni_backend(graph, 'unknown')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
    def test_reject_cpu_device(self):
        with self.assertRaisesRegex(ValueError, 'requires.*CUDA'):
            cuda_device('cpu')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
    def test_no_fallback_when_cuda_is_unavailable(self):
        with patch('torch.cuda.is_available', return_value=False), self.assertRaisesRegex(RuntimeError, 'CPU fallback is disabled'):
            cuda_device('cuda:0')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch required')
class InverseTests(unittest.TestCase):
    def test_affine_is_composed_after_deformation(self):
        import torch

        affine = torch.tensor([[[0, -1., 0, 2.], [1., 0, 0, -1.], [0, 0, 1., 0]]])
        displacement = torch.ones((1, 2, 2, 2, 3))
        parameters = compose_affine(dict(affine=torch.eye(3, 4).unsqueeze(0), grid=displacement), affine)
        torch.testing.assert_close(parameters['affine'], affine)
        torch.testing.assert_close(parameters['grid'][0, 0, 0, 0], torch.tensor([-1., 1., 1.]))

    def test_inverse_noncommuting_affine_and_spatially_varying_field(self):
        import torch
        from torch.nn import functional as functional

        identity = torch.eye(3, 4).unsqueeze(0)
        grid = functional.affine_grid(identity, [1, 1, 28, 30, 32], align_corners=True)
        affine = torch.tensor([[[0.97, -0.12, 0.02, 0.04], [0.1, 1.03, 0, -0.02], [0, 0.03, 1.01, 0.01]]])
        displacement = torch.stack([0.04*grid[..., 1].sin(), 0.02*grid[..., 0]*grid[..., 2], 0.03*grid[..., 0].cos()], dim=-1)
        geometry = torch.eye(4).unsqueeze(0)
        geometry[0, :3, :3] = torch.tensor([[0, -24., 0], [20., 0, 0], [0, 0, 30.]])
        inverse, report = inverse_coordinates(dict(affine=affine, grid=displacement), (25, 29, 31), geometry,
                                               tolerance_mm=0.0001)
        expected_native = functional.affine_grid(identity, [1, 1, 25, 29, 31], align_corners=True)
        analytic_displacement = torch.stack([0.04*inverse[..., 1].sin(), 0.02*inverse[..., 0]*inverse[..., 2],
                                             0.03*inverse[..., 0].cos()], dim=-1)
        composed = torch.einsum('bij,b...j->b...i', affine[:, :, :3], inverse) + affine[:, None, None, None, :, 3] + analytic_displacement
        interior = (inverse.abs() < 0.9).all(dim=-1)
        self.assertGreater(int(interior.sum()), 1000)
        torch.testing.assert_close(composed[interior], expected_native[interior], atol=3e-5, rtol=0)
        self.assertLess(report['max_residual_mm'], 0.0001)

    def test_reject_unconverged_inverse(self):
        import torch
        from torch.nn import functional as functional

        affine = torch.eye(3, 4).unsqueeze(0)
        grid = functional.affine_grid(affine, [1, 1, 16, 16, 16], align_corners=True)
        with self.assertRaisesRegex(ValueError, 'did not converge'):
            inverse_coordinates(dict(affine=affine, grid=0.3*grid), (16, 16, 16),
                                torch.eye(4).unsqueeze(0), max_iterations=10, tolerance_mm=1e-12)


@unittest.skipUnless(os.environ.get('FASTPREP_TEST_FIREANTS_CUDA') and shutil.which('antsApplyTransforms'),
                     'Set FASTPREP_TEST_FIREANTS_CUDA and provide ANTs to run GPU integration tests')
class CudaTests(unittest.TestCase):
    def test_physical_export_oblique_anisotropic_and_inverse(self):
        import nibabel as nib
        import numpy as np
        import SimpleITK as sitk
        import torch
        from torch.nn import functional as functional
        from fireants.io.image import Image, BatchedImages
        from fireants.registration.greedy import GreedyRegistration
        from mni_normalize import apply_args

        device = cuda_device(os.environ['FASTPREP_TEST_FIREANTS_CUDA'])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            fixed = sitk.GetImageFromArray(np.ones((24, 26, 28), np.float32))
            fixed.SetSpacing((1.2, 1.7, 2.0))
            fixed.SetOrigin((-15, -23, -17))
            angle = 0.13
            fixed.SetDirection((np.cos(angle), -np.sin(angle), 0, np.sin(angle), np.cos(angle), 0, 0, 0, 1))
            moving = sitk.GetImageFromArray(np.ones((26, 28, 30), np.float32))
            moving.SetSpacing((1.1, 1.6, 1.9))
            moving.SetOrigin((-17, -22, -18))
            coordinates = np.indices((26, 28, 30))[::-1].transpose(1, 2, 3, 0)
            values = coordinates @ np.diag(moving.GetSpacing()) + np.array(moving.GetOrigin())
            signal = (500 + values @ np.array([1, 3, 7])).astype('float32')
            moving = sitk.GetImageFromArray(signal)
            moving.SetSpacing((1.1, 1.6, 1.9))
            moving.SetOrigin((-17, -22, -18))
            fixed_path, moving_path = folder/'fixed.nii.gz', folder/'moving.nii.gz'
            sitk.WriteImage(fixed, str(fixed_path))
            sitk.WriteImage(moving, str(moving_path))
            fixed_images, moving_images = [BatchedImages(Image(image, device=device)) for image in (fixed, moving)]
            matrix = torch.tensor([[[1., -0.08, 0, 1.], [0.07, 1.01, 0, -0.8], [0, 0.02, 0.98, 0.5], [0, 0, 0, 1.]]], device=device)
            registration = GreedyRegistration([1], [1], fixed_images, moving_images, init_affine=matrix,
                                              smooth_warp_sigma=0, smooth_grad_sigma=0, loss_params={}, progress_bar=False)
            grid = functional.affine_grid(torch.eye(3, 4, device=device).unsqueeze(0), fixed_images.shape, align_corners=True)
            displacement = torch.stack([0.02*grid[..., 1].sin(), 0.01*grid[..., 0]*grid[..., 2], 0.01*grid[..., 0].cos()], dim=-1)
            with torch.no_grad():
                registration.warp.warp.copy_(displacement)
            forward = folder/'forward.nii.gz'
            parameters = registration.get_warp_parameters(fixed_images, moving_images)
            constant = dict(affine=parameters['affine'], grid=torch.zeros_like(parameters['grid']))
            determinants = physical_jacobian(constant, fixed_images, moving_images)
            torch.testing.assert_close(determinants, torch.linalg.det(matrix[:, :3, :3])[:, None, None, None].expand_as(determinants),
                                       atol=1e-5, rtol=0)
            save_forward(parameters, fixed_images, moving_images, forward)
            inverse, report = inverse_coordinates(parameters, moving_images.shape[2:], moving_images.get_torch2phy(), tolerance_mm=0.001)
            inverse_path = folder/'inverse.nii.gz'
            save_inverse(inverse, fixed_images, moving_images, inverse_path)
            for path, reference in [(forward, fixed_path), (inverse_path, moving_path)]:
                field, reference_image = nib.load(path), nib.load(reference)
                self.assertEqual(field.shape, (*reference_image.shape, 1, 3))
                np.testing.assert_allclose(field.affine, reference_image.affine, atol=1e-4, rtol=0)
            output = folder/'resampled.nii.gz'
            subprocess.run(['antsApplyTransforms', *apply_args(moving_path, fixed_path, output, [forward])], check=True)
            normalized = functional.affine_grid(parameters['affine'], fixed_images.shape, align_corners=True) + displacement
            physical = torch.einsum('bij,b...j->b...i', moving_images.get_torch2phy()[:, :3, :3], normalized) + moving_images.get_torch2phy()[:, None, None, None, :3, 3]
            expected = (500 + physical @ torch.tensor([1., 3., 7.], device=device)).detach().cpu().numpy()[0]
            interior = ((normalized.abs() < 0.85).all(dim=-1)).cpu().numpy()[0]
            actual = sitk.GetArrayFromImage(sitk.ReadImage(str(output)))
            np.testing.assert_allclose(actual[interior], expected[interior], atol=3e-4, rtol=0)
            rigid = sitk.AffineTransform(3)
            rigid.SetMatrix((np.cos(0.1), -np.sin(0.1), 0, np.sin(0.1), np.cos(0.1), 0, 0, 0, 1))
            rigid.SetTranslation((0.7, -0.5, 0.2))
            rigid_path = folder/'rigid.tfm'
            sitk.WriteTransform(rigid, str(rigid_path))
            subprocess.run(['antsApplyTransforms', *apply_args(moving_path, fixed_path, output, [forward, rigid_path])], check=True)
            mapped = physical.detach().cpu().numpy()[0] @ np.array(rigid.GetMatrix()).reshape(3, 3).T + np.array(rigid.GetTranslation())
            rigid_expected = 500 + mapped @ np.array([1, 3, 7])
            actual = sitk.GetArrayFromImage(sitk.ReadImage(str(output)))
            np.testing.assert_allclose(actual[interior], rigid_expected[interior], atol=4e-4, rtol=0)
            forward_transform = sitk.DisplacementFieldTransform(sitk.ReadImage(str(forward), sitk.sitkVectorFloat64))
            inverse_transform = sitk.DisplacementFieldTransform(sitk.ReadImage(str(inverse_path), sitk.sitkVectorFloat64))
            for index in [(9, 10, 10), (16, 13, 12), (12, 17, 15)]:
                point = fixed.TransformIndexToPhysicalPoint(index)
                restored = inverse_transform.TransformPoint(forward_transform.TransformPoint(point))
                np.testing.assert_allclose(restored, point, atol=0.005, rtol=0)
            self.assertLess(report['max_residual_mm'], 0.001)

    def test_pipeline_registration_with_original_companion(self):
        import nibabel as nib
        import numpy as np
        from session_inputs import ALIGNMENT_SCHEMA, FLIRT_CONVENTION

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            xyz = np.indices((40, 42, 44))
            distance = ((xyz - np.array([19, 20, 21])[:, None, None, None])**2).sum(axis=0)
            signal = ((distance < 300) * (80 + 15*np.sin(xyz[0]*0.6) + 12*np.cos(xyz[1]*0.5)
                       + 10*np.sin(xyz[2]*0.7) + 30*np.exp(-((xyz-14)**2).sum(axis=0)/25))).astype('float32')
            raw, fixed, mask, companion = [folder/name for name in ('sub-01_T1w.nii.gz', 'atlas.nii.gz', 'mask.nii.gz', 'sub-01_T2w.nii.gz')]
            for path, values in [(raw, signal), (fixed, np.roll(signal, 2, axis=0)),
                                 (mask, (distance < 280).astype('uint8')), (companion, signal*2)]:
                nib.save(nib.Nifti1Image(values, np.diag([1.1, 1.3, 1.5, 1])), path)
            # Regression: native sform has shear, ITK used qform while old
            # validation used sform. Both final inputs must be harmonized.
            for path in (raw, companion):
                image = nib.load(path)
                qform = image.affine.copy()
                sform = qform.copy()
                sform[0, 1] = 0.002
                image.set_qform(qform, 1)
                image.set_sform(sform, 1)
                nib.save(image, path)
            def acquisition(path):
                return dict(path=str(path), series_id=path.name.split('.')[0], analysis_unit_id='sub-01')
            plan = dict(analysis_unit_id='sub-01', subject_id='sub-01', session_id='',
                        reference=acquisition(raw), companions=[acquisition(companion)])
            np.savetxt(folder/'rigid.mat', np.eye(4))
            write_json(folder/'alignment.json', dict(schema=ALIGNMENT_SCHEMA, transform_convention=FLIRT_CONVENTION,
                       analysis_unit_id='sub-01', reference=dict(path=str(raw), series_id='sub-01_T1w'),
                       registrations=[dict(moving=str(companion), fixed=str(raw), series_id='sub-01_T2w',
                                           transform='rigid.mat', registered='must-not-be-read.nii.gz')]))
            (folder/'mni').mkdir()
            calls = []
            def run(identity, arguments):
                calls.append((identity, arguments))
                if identity == 'fireants:register':
                    subprocess.run([sys.executable, '-B', ROOT/'bin/fireants_register.py', *map(str, arguments),
                                    '--scales', '2', '1', '--affine-iterations', '100', '50',
                                    '--warp-iterations', '40', '20'], check=True)
                else:
                    subprocess.run(['antsApplyTransforms', *map(str, arguments)], check=True)
            stage = SimpleNamespace(subject='sub-01', node={'parameters': {'backend': 'fireants'}},
                                    spec=dict(device=os.environ['FASTPREP_TEST_FIREANTS_CUDA'], threads=2),
                                    runtime=dict(session_input_plan=[plan], mni_template=dict(id='Synthetic', resolution=1,
                                                 t1w=str(fixed), mask=str(mask))), metrics={}, output=lambda path: folder/path,
                                    run=run, file_record=lambda path: {'path': str(path.relative_to(folder))},
                                    inputs={'alignment_manifest': (folder, {'path': 'alignment.json', 'members': [{'path': 'rigid.mat'}]})})
            result = read_json(normalize_session(stage)['mni_manifest'])
            self.assertEqual([name for name, _ in calls], ['fireants:register', 'ants:antsApplyTransforms', 'ants:antsApplyTransforms'])
            self.assertEqual([len(image['ants_image_transform_list']) for image in result['images']], [1, 2])
            for (_, arguments), original in zip(calls[1:], [raw, companion]):
                working = Path(arguments[arguments.index('-i') + 1])
                self.assertNotEqual(working, original)
                np.testing.assert_array_equal(nib.load(working).get_fdata(), nib.load(original).get_fdata())
            self.assertEqual([row['source'] for row in result['images']], [str(raw), str(companion)])
            first, second = [nib.load(folder/'mni'/image['image']).get_fdata() for image in result['images']]
            np.testing.assert_allclose(second, first*2, atol=0.001, rtol=0)
            before = np.mean((signal - np.roll(signal, 2, axis=0))**2)
            after = np.mean((first - np.roll(signal, 2, axis=0))**2)
            self.assertLess(after, before)
            self.assertEqual(result['registration']['backend'], 'fireants')
            self.assertEqual(result['scientific_approval'], 'not_performed')


if __name__ == '__main__':
    unittest.main()

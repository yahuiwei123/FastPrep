"""The fused base segmentation preserves postprocessing without posterior files."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]


class SynthSegOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if any(importlib.util.find_spec(name) is None for name in ('numpy', 'nibabel', 'tensorflow')):
            raise unittest.SkipTest('NumPy, NiBabel and TensorFlow required')
        spec = importlib.util.spec_from_file_location('synthseg_outputs_test', ROOT / 'model/synthseg/atomic.py')
        cls.wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.wrapper)

    def test_fused_base_matches_file_handoff_without_saving_probabilities(self):
        import nibabel as nib
        import numpy as np

        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '-1'}):
            from SynthSeg import predict_synthseg as upstream
        resources = ROOT / 'model/synthseg/source/data/labels_classes_priors'
        labels_file = resources / 'synthseg_segmentation_labels_2.0.npy'
        labels, flip, _ = self.wrapper.label_info(labels_file)
        shape = (12, 14, 16)
        hard = np.zeros(shape, dtype=np.int32)
        hard[2:6, 2:12, 2:14] = 2
        hard[6:10, 2:12, 2:14] = 41
        hard[3:5, 4:10, 4:12] = 3
        hard[7:9, 4:10, 4:12] = 42
        post = np.full((1, *shape, len(labels)), 0.001, dtype=np.float32)
        for channel, label in enumerate(labels):
            post[0, ..., channel][hard == label] = 0.9
        post /= post.sum(axis=-1, keepdims=True)
        aff = np.diag([-1., 1., -1., 1.])
        aff[:3, 3] = [11, -2, 15]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / 'image.npy'
            np.save(image, np.zeros((1, *shape, 1)))
            geometry = root / 'geometry.json'
            geometry.write_text(json.dumps(dict(shape=list(shape), pad_idx=[0, 0, 0, *shape],
                crop_idx=None, affine=aff.tolist(), resolution=[1., 1., 1.],
                header=base64.b64encode(nib.Nifti1Header().binaryblock).decode())))
            raw = root / 'reference_raw.npy'
            np.save(raw, post)
            common = ['--geometry', str(geometry), '--labels', str(labels_file),
                      '--topology', str(resources / 'synthseg_topological_classes_2.0.npy')]
            reference = root / 'reference'
            fused = root / 'fused'
            with patch.object(self.wrapper, 'configure_tensorflow', return_value=(None, {'device': 'cpu'})):
                self.wrapper.main(['postprocess', '--posteriors', str(raw), *common,
                    '--output-segmentation', str(reference / 'aseg.nii.gz'),
                    '--output-volumes', str(reference / 'volumes.npy')])
                net = MagicMock()
                net.weights = []
                net.predict.side_effect = [post.copy(), np.flip(post[..., flip], axis=1).copy()]
                with patch.object(upstream, 'build_model', return_value=net):
                    self.wrapper.main(['segment-base', '--image', str(image),
                        '--checkpoint', str(root / 'unused_model.h5'),
                        '--denoiser-labels', str(resources / 'synthseg_denoiser_labels_2.0.npy'),
                        '--sequential-flip', *common,
                        '--output-segmentation', str(fused / 'aseg.nii.gz'),
                        '--output-volumes', str(fused / 'volumes.npy'),
                        '--runtime-info', str(fused / 'runtime.json')])
            expected, actual = nib.load(reference / 'aseg.nii.gz'), nib.load(fused / 'aseg.nii.gz')
            np.testing.assert_array_equal(np.asanyarray(actual.dataobj), np.asanyarray(expected.dataobj))
            np.testing.assert_array_equal(actual.affine, aff)
            np.testing.assert_array_equal(np.load(fused / 'volumes.npy'), np.load(reference / 'volumes.npy'))
            self.assertTrue({2, 3, 41, 42}.issubset(np.unique(np.asanyarray(actual.dataobj))))
            self.assertEqual({p.name for p in fused.iterdir()}, {'aseg.nii.gz', 'volumes.npy', 'runtime.json'})
            self.assertEqual(json.loads((fused / 'runtime.json').read_text())['posterior_storage'], 'memory_only')
            self.assertEqual(net.predict.call_count, 2)


if __name__ == '__main__':
    unittest.main()

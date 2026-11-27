"""Cropped label measurements must exactly match full-volume measurements."""

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))

try:
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    from neuroqc.metrics import segmentation
except ImportError:
    nib = None


@unittest.skipIf(nib is None, 'requires QC scientific Python')
class SegmentationBoundsTests(unittest.TestCase):
    def full_reference(self, values, label, connectivity, volume):
        mask = values == label
        voxels = int(mask.sum())
        components, count = ndimage.label(mask,
            ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity]))
        sizes = sorted(np.bincount(components.ravel())[1:].astype(int).tolist(), reverse=True)
        border = np.zeros(mask.shape, dtype=bool)
        for axis in range(3):
            for index in (0, -1):
                edge = [slice(None)] * 3
                edge[axis] = index
                border[tuple(edge)] = True
        return dict(foregroundVoxelCount=voxels, volumeMl=voxels * volume / 1000,
            connectedComponents=int(count), componentConnectivity=connectivity,
            componentSizesVoxels=sizes, detachedForegroundVoxelCount=voxels - (sizes[0] if sizes else 0),
            holes=int((ndimage.binary_fill_holes(mask) & ~mask).sum()),
            borderVoxelFraction=float((mask & border).sum() / voxels) if voxels else None)

    def test_cavities_boundaries_diagonal_contacts_absent_and_sparse_labels(self):
        hollow = np.zeros((13, 12, 11), dtype=np.int64)
        hollow[2:9, 2:9, 2:9] = 2
        hollow[3:8, 3:8, 3:8] = 0
        hollow[0, 0, 0] = hollow[1, 1, 1] = hollow[-1, -1, -1] = -7
        hollow[9:12, :4, :4] = 9000000000
        rng = np.random.default_rng(17)
        cases = [hollow, rng.choice([0, -7, 2, 9000000000], (7, 8, 9)),
                 np.ones((1, 7, 6), dtype=np.int64) * 2, np.zeros((3, 4, 5), dtype=np.int64)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case, values in enumerate(cases):
                image, labels = root / 'image.nii', root / 'labels.nii'
                affine = np.diag([1.2, 1.4, 2., 1.])
                nib.save(nib.Nifti1Image(np.ones(values.shape, np.float32), affine), image)
                nib.save(nib.Nifti1Image(values, affine, dtype=np.int64), labels)
                voxel_volume = abs(float(np.linalg.det(nib.load(labels).affine[:3, :3])))
                for connectivity in (6, 18, 26):
                    selected = [0, -7, 2, 9000000000, 51]
                    actual = segmentation(image, labels, selected, connectivity)
                    for label in selected:
                        with self.subTest(case=case, connectivity=connectivity, label=label):
                            self.assertEqual(actual['metrics']['labels'][str(label)],
                                self.full_reference(values, label, connectivity, voxel_volume))
                    self.assertEqual(actual['status'], 'unknown')
                    self.assertIn('visual_review', actual['missing_checks'])

    def test_invalid_values_and_grid_still_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image, labels = root / 'image.nii', root / 'labels.nii'
            raw = np.ones((4, 5, 6), dtype=np.float32)
            nib.save(nib.Nifti1Image(raw, np.eye(4)), image)
            for value in (0.1, np.nan, np.inf):
                data = raw.copy()
                data[2, 2, 2] = value
                nib.save(nib.Nifti1Image(data, np.eye(4)), labels)
                result = segmentation(image, labels)
                self.assertEqual(result['status'], 'fail')
                self.assertFalse(result['metrics']['integerLabels'])
                self.assertEqual(result['metrics']['labels'], {})
            nib.save(nib.Nifti1Image(raw, np.diag([2., 1., 1., 1.])), labels)
            self.assertEqual(segmentation(image, labels)['status'], 'fail')


if __name__ == '__main__':
    unittest.main()

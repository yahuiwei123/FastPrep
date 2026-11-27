"""Near-coplanar triangle ordering must not bypass a GPU movement decision."""
import os
from pathlib import Path
import sys
import unittest

try:
    import torch
except ImportError:
    torch = None

BUNDLE = Path(__file__).resolve().parents[1] / 'vendor/t2_pial'
sys.path.insert(0, str(BUNDLE))


@unittest.skipUnless(torch is not None and os.environ.get('FASTPREP_TEST_PLACEMENT_CUDA'),
                     'requires matching CUDA placement runtime')
class PriorityCommitTests(unittest.TestCase):
    def setUp(self):
        from ordered_move import PriorityCommit

        self.faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device='cuda')
        library = os.environ.get('FASTPREP_TEST_COLLISION_LIBRARY', str(BUNDLE/'native/classify.so'))
        self.engine = PriorityCommit(self.faces, library)
        # Six-point regression fixture; no subject files or external paths required.
        self.current = torch.tensor([
            [-5.757838249206543, -71.74835968017578, -27.176584243774414],
            [-6.076255798339844, -71.55376434326172, -27.924476623535156],
            [-5.58822774887085, -71.68368530273438, -26.97411346435547],
            [-5.796329975128174, -71.6254653930664, -27.45995330810547],
            [-6.236766338348389, -71.53961181640625, -28.17507553100586],
            [-5.92392110824585, -71.57408142089844, -27.695810317993164]], device='cuda')
        self.delta = torch.zeros_like(self.current)
        self.delta[3] = torch.tensor([.021541280671954155, .03513961657881737, -.007906747981905937], device='cuda')

    def test_collision_is_symmetric_at_problem_endpoint(self):
        keys = torch.tensor([1, 2], dtype=torch.int64, device='cuda')
        self.assertEqual(self.engine.hits(self.current + self.delta, keys).tolist(), [True, True])

    def test_proposal_rejects_endpoint_before_commit_without_moving_frozen_vertices(self):
        frozen = torch.ones(6, dtype=torch.bool, device='cuda')
        frozen[3] = False
        original = self.current.clone()
        end, motion, scales, trace, pairs = self.engine.step(
            self.current, self.delta, torch.zeros_like(self.delta), frozen, 1., 0., .3)
        self.assertTrue(torch.equal(self.current, original))
        self.assertTrue(torch.equal(end[frozen], original[frozen]))
        self.assertEqual(scales[3].item(), 0.)
        self.assertTrue(torch.equal(end, original))
        self.assertEqual(trace['unresolved_per_try'], [0])
        self.assertFalse(trace['global_fallback'])

    def test_clear_motion_is_not_globally_frozen(self):
        current = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],
                                [0.,0.,3.],[1.,0.,3.],[0.,1.,3.]], device='cuda')
        gradient = torch.zeros_like(current)
        gradient[3:, 2] = .1
        frozen = torch.tensor([True,True,True,False,False,False], device='cuda')
        end, motion, scales, trace, pairs = self.engine.step(
            current, gradient, torch.zeros_like(current), frozen, 1., 0., .3)
        torch.testing.assert_close(end, current + gradient)
        self.assertTrue((scales == 1).all())


if __name__ == '__main__':
    unittest.main()

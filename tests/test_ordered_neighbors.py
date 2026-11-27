import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

bundle = Path(__file__).resolve().parents[1] / 'vendor/t2_pial'
if torch is not None:
    sys.path.insert(0, str(bundle))
    from ordered_move import NeighborCommit


@unittest.skipIf(torch is None, 'requires PyTorch')
class OrderedNeighborTests(unittest.TestCase):
    def setUp(self):
        self.engine = NeighborCommit.__new__(NeighborCommit)
        self.engine.faces = torch.tensor([[0, 1, 2]])
        self.offsets = torch.tensor([0, 2, 4, 6], dtype=torch.int32)
        self.ids = torch.tensor([2, 1, 0, 2, 1, 0], dtype=torch.int32)

    def test_live_ring_is_validated_without_sorting_it(self):
        self.engine.set_one_ring(self.offsets, self.ids)
        self.assertEqual(self.engine.edges[0].tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(self.engine.edges[1].tolist(), self.ids.tolist())
        with self.assertRaisesRegex(ValueError, 'already initialized'):
            self.engine.set_one_ring(self.offsets, self.ids)
        with self.assertRaisesRegex(ValueError, 'state does not match'):
            self.engine.plan(torch.zeros(2, 3), torch.zeros(2, 3))

    def test_incomplete_wrong_or_duplicate_ring_is_rejected(self):
        for offsets, ids in (([0, 3, 2, 6], self.ids), ([0, 2, 4, 7], self.ids),
                             ([0, 2, 4, 6], [2, 1, 0, 2, 3, 0]), ([0, 2, 4, 6], [2, 2, 0, 2, 1, 0]),
                             ([0, 2, 4, 5], [2, 1, 0, 2, 1]), ([0, 2, 4, 6], [0, 1, 0, 2, 1, 0])):
            with self.subTest(offsets=offsets, ids=ids), self.assertRaises(ValueError):
                self.engine.set_one_ring(torch.as_tensor(offsets, dtype=torch.int32), torch.as_tensor(ids, dtype=torch.int32))

    def test_uses_residual_in_neighbor_order_independently_per_vertex(self):
        delta = torch.tensor([[1., 1., 0.], [1., 1., 0.], [-1., 0., 0.], [3., 2., 1.]])
        vertices = torch.tensor([0, 0, 1, 1, 2])
        relative = torch.tensor([[1., 0., 0.], [1., 1., 0.], [1., 1., 0.], [1., 0., 0.], [1., 0., 0.]])
        actual, _ = self.engine.project_neighbors(delta, vertices, relative, relative.norm(dim=1))
        torch.testing.assert_close(actual, torch.tensor([[-.5, .5, 0.], [0., 0., 0.], [-1., 0., 0.], [3., 2., 1.]]))
        self.assertEqual(delta[0].tolist(), [1., 1., 0.])

    def test_no_close_neighbors_and_coincident_guard(self):
        delta = torch.tensor([[1., 2., 3.]])
        result, changed = self.engine.project_neighbors(delta, torch.empty(0, dtype=torch.long), delta[:0], torch.empty(0))
        self.assertTrue(torch.equal(result, delta))
        self.assertEqual(changed, 0)
        with self.assertRaisesRegex(ValueError, 'coincident one-ring'):
            self.engine.project_neighbors(delta, torch.tensor([0]), torch.zeros_like(delta), torch.zeros(1))

    @unittest.skipUnless(os.environ.get('FASTPREP_TEST_PLACEMENT_CUDA'), 'requires matching CUDA placement runtime')
    def test_default_worker_reuses_uploaded_native_order(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as errors:
            root = Path(directory) / 'sync-moves'
            root.mkdir()
            np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], '<i4').tofile(root / 'faces.i32')
            np.array([0, 3, 6, 9, 12], '<i4').tofile(root / 'offsets.i32')
            np.array([2, 1, 3, 0, 2, 3, 1, 3, 0, 2, 1, 0], '<i4').tofile(root / 'neighbors.i32')
            state = np.zeros((4, 10), '<f4')
            state[:, :3] = [[0, 0, 0], [.004, .004, 0], [.004, 0, 0], [0, 0, 1]]
            state[0, 3:6] = [.002, .002, 0]
            state[1:, 9] = 1
            env = os.environ.copy()
            env.pop('SYNC_ORDERED', None)
            client, worker = socket.socketpair()
            with client, worker, subprocess.Popen([sys.executable, bundle / 'sync_worker.py', bundle / 'native/classify.so'],
                                                  cwd=directory, env=env, stdin=worker, stderr=errors) as process:
                worker.close()
                client.settimeout(60)
                try:
                    for index in range(2):
                        step = root / f'step-{index}'
                        step.mkdir()
                        state.tofile(step / 'input.f32')
                        (step / 'config.txt').write_text('4 1 0 .3')
                        client.sendall(b'M')
                        if client.recv(1) != b'M':
                            errors.seek(0)
                            self.fail(errors.read().decode())
                        output = np.fromfile(step / 'output.f32', '<f4').reshape(4, 7)
                        np.testing.assert_allclose(output[0, :3], [-.001, .001, 0], atol=1e-8)
                        np.testing.assert_array_equal(output[1:, :3], state[1:, :3])
                        if index == 0:
                            for name in ('offsets.i32', 'neighbors.i32'):
                                (root / name).rename(Path(directory) / name)
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(process.wait(timeout=30), 0)
                finally:
                    if process.poll() is None:
                        process.kill()


if __name__ == '__main__':
    unittest.main()

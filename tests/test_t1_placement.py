import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'bin'))
from t2_pial_backend import refine

try:
    import nibabel as nib
    import numpy as np
except ImportError:
    nib = None


@unittest.skipIf(nib is None, 'requires the scientific Python environment')
class PlacementTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = {'t1_placement_backend':'checked', 'threads':4, 'device':'cuda:0'}
        self.stage = SimpleNamespace(root=self.root, node={'parameters':{}}, spec=self.spec,
                                     run=self.stock, run_process=self.run_process)
        self.points = np.eye(4,3)
        self.faces = np.array([[0,1,2],[0,2,3],[1,3,2],[0,3,1]])
        self.meta = dict(head=[2,0,20], valid='1', filename='example', volume=[16]*3, voxelsize=[1]*3,
                         xras=[1,0,0],yras=[0,1,0],zras=[0,0,1],cras=[0]*3)
        self.input = self.root/'input'
        self.surface(self.input, self.points)
        self.final = self.root/'lh.final'
        self.args = ['--lh','--white','--i',self.input,'--o',self.final]
        self.change = None
        self.stock_failure = False

    def surface(self, path, points, faces=None, meta=None):
        nib.freesurfer.write_geometry(path, points, self.faces if faces is None else faces, volume_info=meta or self.meta)

    def outputs(self, args, change=None):
        for flag, kind in [('--o','surface'),('--target','targets')]:
            if flag not in args: continue
            points=self.points.copy()
            if change==kind: points[0,1]+=.125
            self.surface(args[args.index(flag)+1], points, self.faces[::-1] if change=='faces' else self.faces,
                         dict(self.meta,cras=[1,0,0]) if change=='geometry' else self.meta)
        if '--ripflag-out' in args:
            flags = np.zeros((4,1,1),dtype='uint8'); flags[1]=int(change=='flags')
            nib.save(nib.MGHImage(flags,np.eye(4)),args[args.index('--ripflag-out')+1])

    def stock(self, command, args):
        self.assertEqual(command,'freesurfer:mris_place_surface')
        if self.stock_failure: raise RuntimeError('stock failed')
        self.outputs(args)

    def run_process(self, command, args, env, cwd, check=True):
        if self.change=='error': return 2
        self.outputs(args,self.change); return 0

    def select(self):
        with patch('t2_pial_backend.configuration',return_value=({'CUDA_VISIBLE_DEVICES':'0'},None)):
            return refine(self.stage,self.args,'lh','t1-white')

    def test_checked_exact_and_fallbacks(self):
        for change in (None,'surface','targets','geometry','faces','flags','error'):
            with self.subTest(change=change):
                self.change=change; decision=self.select()
                self.assertEqual(decision['selected'],'official-cpu' if change else 'cuda-ordered-signed')
                np.testing.assert_array_equal(nib.freesurfer.read_geometry(self.final)[0],self.points)
                if change=='surface': self.assertEqual(decision['same_input_cpu_comparison']['white']['max_mm'],.125)

    def test_no_reference_or_failed_gpu_does_not_publish(self):
        self.stock_failure=True
        with self.assertRaises(RuntimeError): self.select()
        self.assertFalse(self.final.exists())
        self.spec['t1_placement_backend']='gpu'; self.change='error'
        with self.assertRaises(RuntimeError): self.select()
        self.assertFalse(self.final.exists())

    def test_default_gpu_and_explicit_cpu(self):
        self.spec.pop('t1_placement_backend')
        decision=self.select(); self.assertEqual(decision['requested'],'gpu')
        self.assertEqual(decision['selected'],'cuda-ordered-signed'); self.assertIsNone(decision['same_input_cpu_comparison'])
        self.spec['t1_placement_backend']='cpu'
        self.assertEqual(self.select()['selected'],'official-cpu')

    def test_default_gpu_does_not_fall_back(self):
        self.spec.pop('t1_placement_backend')
        with patch('t2_pial_backend.configuration',return_value=(None,'CUDA unavailable')):
            with self.assertRaisesRegex(RuntimeError,'CUDA unavailable'): refine(self.stage,self.args,'lh','t1-white')
        self.assertFalse(self.final.exists())

    def test_unsupported_threads_and_wrong_kind(self):
        self.spec['threads']=1
        self.assertEqual(refine(self.stage,self.args,'lh','t1-white')['selected'],'official-cpu')
        with self.assertRaisesRegex(ValueError,'wrong T1'): refine(self.stage,self.args,'lh','t1-pial')

    def select_t2(self):
        args=['--pial','--i',self.input,'--white-surf',self.input,'--mmvol','example.mgz','T2','--o',self.final]
        with patch('t2_pial_backend.configuration',return_value=({'CUDA_VISIBLE_DEVICES':'0'},None)):
            return refine(self.stage,args,'lh')

    def test_t2_gpu_default_and_task_override_do_not_run_stock(self):
        self.spec['t1_placement_backend'] = 'cpu'
        with patch.object(self.stage, 'run', side_effect=AssertionError('unexpected CPU placement')):
            for legacy_graph in (False, True):
                if legacy_graph:
                    self.stage.node['parameters']['t2_pial_backend'] = 'checked'
                    self.spec['t2_pial_backend'] = 'gpu'
                with self.subTest(legacy_graph=legacy_graph):
                    result = self.select_t2()
                    self.assertEqual((result['requested'], result['selected']), ('gpu', 'cuda-parallel'))
                    self.assertIsNone(result['same_input_cpu_comparison'])

    def test_t2_gpu_errors_never_fall_back(self):
        with patch.object(self.stage, 'run', side_effect=AssertionError('unexpected CPU placement')):
            for change in ('error', 'faces'):
                self.change = change
                with self.subTest(change=change), self.assertRaises(RuntimeError):
                    self.select_t2()
                self.assertFalse(self.final.exists())
            self.spec['threads'] = 1
            with self.assertRaisesRegex(RuntimeError, 'unqualified thread count'):
                refine(self.stage, [], 'lh')

    def test_t2_checked_or_cpu_only_when_explicit(self):
        self.stage.node['parameters']['t2_pial_backend'] = 'checked'
        with patch.object(self.stage, 'run', wraps=self.stock) as stock:
            result = self.select_t2()
            self.assertTrue(result['same_input_cpu_comparison']['exact'])
            stock.assert_called_once()
        self.spec['t2_pial_backend'] = 'cpu'
        with patch.object(self.stage, 'run_process', side_effect=AssertionError('unexpected CUDA placement')):
            self.assertEqual(self.select_t2()['selected'], 'official-cpu')
        self.stage.node['parameters']['t2_pial_backend'] = 'gpu'
        self.spec['t2_pial_backend'] = 'checked'
        self.change = 'surface'
        result = self.select_t2()
        self.assertEqual(result['selected'], 'official-cpu')
        self.assertFalse(result['same_input_cpu_comparison']['exact'])

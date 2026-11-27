import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from reference_runtime import NativeStage, read_json, surface_repair_checks


class HemisphereTests(unittest.TestCase):
    def stage(self, folder, threads=4, **parameters):
        stage = object.__new__(NativeStage)
        stage.root = stage.subject_dir = stage.subjects_dir = folder
        stage.spec = dict(threads=threads)
        stage.node = dict(parameters=parameters)
        stage.logs, stage.metrics = [], {}
        stage.runtime = {'commands': {'test:identity': {'argv_prefix': [sys.executable, '-B', '-c',
            'import os,sys; print(sys.argv[1], os.environ["OMP_NUM_THREADS"])']}}}
        return stage

    def test_ordered_chains_overlap_and_keep_local_last_log_and_nested_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage(Path(temporary))
            barrier = threading.Barrier(2, timeout=10)
            def operation(child, hemi):
                child.run('test:identity', [hemi + '-first'])
                barrier.wait()
                self.assertEqual((child.root / child.logs[-1]['stdout']).read_text().strip(), hemi + '-first 2')
                child.run('test:identity', [hemi + '-second'])
                child.metrics['placement_checks'] = {hemi: {'correspondence': True}}
                return hemi
            self.assertEqual(stage.each_hemisphere(operation), {'lh': 'lh', 'rh': 'rh'})
            self.assertEqual(stage.metrics['placement_checks'], {
                'lh': {'correspondence': True}, 'rh': {'correspondence': True}})
            records = read_json(stage.root / 'commands.json')
            self.assertEqual(len(records), 4)
            self.assertEqual(len({record['stdout'] for record in records}), 4)
            self.assertEqual([record['hemi'] for record in records], ['lh', 'lh', 'rh', 'rh'])
            for hemi in ('lh', 'rh'):
                self.assertEqual(len(read_json(stage.root / 'logs' / hemi / 'commands.json')), 2)

    def test_real_topology_log_parser_cannot_read_other_hemisphere(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage(Path(temporary))
            stage.runtime['commands'].update({
                'freesurfer:mris_euler_number': {'argv_prefix': [sys.executable, '-B', '-c',
                    'import sys; value=2 if sys.argv[1]=="lh" else 0; print(f"euler # = value = {value} --> holes\\ntotal defect index = 0")']},
                'freesurfer:mris_info': {'argv_prefix': [sys.executable, '-B', '-c',
                    'import sys; print("Found", 3 if sys.argv[-1]=="lh" else 7, "intersections")']}})
            barrier = threading.Barrier(2, timeout=10)
            def operation(child, hemi):
                barrier.wait()
                return surface_repair_checks(child, hemi)
            results = stage.each_hemisphere(operation)
            self.assertEqual((results['lh']['euler_number'], results['rh']['euler_number']), (2, 0))
            self.assertEqual((results['lh']['self_intersecting_vertices'], results['rh']['self_intersecting_vertices']), (3, 7))

    def test_failure_collects_sibling_and_keeps_journals_without_publishing_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage(Path(temporary))
            barrier = threading.Barrier(2, timeout=10)
            def operation(child, hemi):
                child.run('test:identity', [hemi])
                barrier.wait()
                if hemi == 'lh':
                    raise ValueError('invalid geometry')
                child.metrics['rh'] = {'checked': True}
            with self.assertRaisesRegex(ValueError, 'invalid geometry'):
                stage.each_hemisphere(operation)
            self.assertEqual(len(read_json(stage.root / 'commands.json')), 2)
            execution = read_json(stage.root / 'hemisphere_execution.json')
            self.assertEqual(execution['hemispheres']['lh']['status'], 'failed')
            self.assertEqual(execution['hemispheres']['rh']['status'], 'completed')
            self.assertNotIn('rh', stage.metrics)

    def test_serial_fallback_explicit_limits_and_cpu_budget(self):
        for threads, minimum, uses_gpu, parameters, workers, quota in [
            (1, 1, False, {}, 1, 1), (4, 1, False, {'hemisphere_workers': 1}, 1, 4),
            (4, 1, True, {'gpu_hemisphere_workers': 1}, 1, 4),
            (4, 4, True, {}, 1, 4), (8, 4, True, {}, 2, 4), (12, 4, True, {}, 2, 6),
            (16, 4, True, {}, 2, 8), (3, 1, False, {}, 2, 1)]:
            with self.subTest(threads=threads, parameters=parameters), tempfile.TemporaryDirectory() as temporary:
                stage = self.stage(Path(temporary), threads, **parameters)
                results = stage.each_hemisphere(lambda child, hemi: child.spec['threads'],
                    minimum_threads=minimum, uses_gpu=uses_gpu)
                self.assertEqual(results, {'lh': quota, 'rh': quota})
                self.assertEqual(stage.metrics['hemisphere_execution']['workers'], workers)
                self.assertLessEqual(workers * quota, threads)

    def test_placement_uses_two_threads_per_hemisphere(self):
        for threads in (4, 8, 16):
            with self.subTest(threads=threads), tempfile.TemporaryDirectory() as temporary:
                stage = self.stage(Path(temporary), threads)
                results = stage.each_hemisphere(lambda child, hemi: child.spec['threads'],
                    minimum_threads=2, maximum_threads=2, uses_gpu=True)
                self.assertEqual(results, {'lh': 2, 'rh': 2})
                self.assertEqual(stage.metrics['hemisphere_execution']['workers'], 2)

    def test_invalid_worker_limits_and_colliding_metrics_are_rejected(self):
        for parameters in ({'hemisphere_workers': 0}, {'hemisphere_workers': True}, {'gpu_hemisphere_workers': 3}):
            with tempfile.TemporaryDirectory() as temporary:
                stage = self.stage(Path(temporary), **parameters)
                with self.assertRaises(ValueError):
                    stage.each_hemisphere(lambda child, hemi: None)
        with tempfile.TemporaryDirectory() as temporary:
            stage = self.stage(Path(temporary))
            def operation(child, hemi):
                child.metrics['shared'] = hemi
            with self.assertRaisesRegex(ValueError, 'same metric'):
                stage.each_hemisphere(operation)


if __name__ == '__main__':
    unittest.main()

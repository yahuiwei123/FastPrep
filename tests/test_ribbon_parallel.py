import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bin'))
from reference_runtime import NativeStage, cortical_pair, read_json, ribbon, write_json


WORKER = """
import json
import os
from pathlib import Path
import sys
import time
identity, peers, exit_code = sys.argv[1:]
started = time.monotonic()
Path(identity + '.ready').touch()
deadline = started + 10
while len(list(Path('.').glob('*.ready'))) < int(peers):
    if time.monotonic() > deadline:
        raise RuntimeError('Concurrent peer did not start')
    time.sleep(0.01)
print(json.dumps(dict(identity=identity, started=started, ended=time.monotonic(),
    omp=os.environ['OMP_NUM_THREADS'], blas=os.environ.get('OPENBLAS_NUM_THREADS'),
    itk=os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'])), flush=True)
sys.exit(int(exit_code))
"""


class ConcurrentCommandTests(unittest.TestCase):
    def stage(self, folder, threads):
        stage = object.__new__(NativeStage)
        stage.root = stage.subjects_dir = stage.subject_dir = folder
        stage.spec = dict(threads=threads)
        stage.logs = []
        stage.runtime = {'commands': {'test:worker': {'argv_prefix': [sys.executable, '-B', '-c', WORKER]}}}
        return stage

    def test_processes_overlap_with_distinct_logs_and_shared_cpu_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            stage = self.stage(folder, 4)
            stage.run_concurrent([('test:worker', ['lh', '2', '0']), ('test:worker', ['rh', '2', '0'])], max_workers=2)
            records = read_json(folder/'commands.json')
            self.assertEqual([record['exit_code'] for record in records], [0, 0])
            self.assertEqual(len({record['stdout'] for record in records}), 2)
            results = [read_json(folder/record['stdout']) for record in records]
            self.assertLessEqual(max(result['started'] for result in results), min(result['ended'] for result in results))
            self.assertEqual([result['identity'] for result in results], ['lh', 'rh'])
            for result, record in zip(results, records):
                self.assertEqual((result['omp'], result['blas'], result['itk']), ('2', '2', '2'))
                self.assertGreater(record['duration_seconds'], 0)

    def test_one_allocated_cpu_serializes_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            stage = self.stage(folder, 1)
            stage.run_concurrent([('test:worker', ['lh', '1', '0']), ('test:worker', ['rh', '1', '0'])], max_workers=2)
            results = [read_json(folder/record['stdout']) for record in stage.logs]
            self.assertGreaterEqual(results[1]['started'], results[0]['ended'])
            self.assertEqual([result['omp'] for result in results], ['1', '1'])

    def test_failed_child_blocks_success_and_preserves_both_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            stage = self.stage(folder, 4)
            with self.assertRaisesRegex(RuntimeError, 'Concurrent native command failed'):
                stage.run_concurrent([('test:worker', ['lh', '2', '3']), ('test:worker', ['rh', '2', '0'])], max_workers=2)
            self.assertEqual([record['exit_code'] for record in read_json(folder/'commands.json')], [3, 0])
            for record in stage.logs:
                self.assertTrue((folder/record['stderr']).is_file())

    def test_launch_failure_is_recorded_and_other_child_is_collected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            stage = self.stage(folder, 4)
            stage.runtime['commands']['test:missing'] = {'argv_prefix': [str(folder/'missing-executable')]}
            with self.assertRaisesRegex(RuntimeError, 'Concurrent native command failed'):
                stage.run_concurrent([('test:missing', []), ('test:worker', ['rh', '1', '0'])], max_workers=2)
            self.assertEqual([record['exit_code'] for record in stage.logs], [127, 0])
            self.assertIn('missing-executable', (folder/stage.logs[0]['stderr']).read_text())


class RibbonDispatchTests(unittest.TestCase):
    def test_rasterization_does_not_run_pair_measurements(self):
        for threads, parameters, parallel in [(4, {}, True), (2, {}, False),
                                              (4, {'ribbon_parallel': False}, False)]:
            with self.subTest(threads=threads), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                calls = []
                stage = SimpleNamespace(root=folder, subjects_dir=folder, subject='sub-test',
                    node={'parameters': parameters, 'outputs': [{'name': 'ribbon_mgz'}]},
                    spec=dict(threads=threads), metrics={},
                    run=lambda identity, arguments: calls.append((identity, arguments)),
                    volume=lambda port: folder/(port+'.mgz'))
                self.assertEqual(ribbon(stage), {'ribbon_mgz': folder/'ribbon_mgz.mgz'})
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][0], 'freesurfer:mris_volmask')
                self.assertEqual('--parallel' in calls[0][1], parallel)

    def test_independent_pair_keeps_both_measurements_and_ribbon_input(self):
        for threads, parameters, expected_workers, parallel in [
                (4, {}, 2, True), (2, {}, 2, False), (1, {}, 1, False),
                (4, {'ribbon_parallel': False, 'metrics_parallel': False}, 1, False)]:
            with self.subTest(threads=threads, parameters=parameters), tempfile.TemporaryDirectory() as temporary:
                folder = Path(temporary)
                calls = []
                node = {'parameters': parameters, 'outputs': [{'name': 'ribbon_mgz'}]}
                def concurrent(jobs, max_workers):
                    self.assertEqual(max_workers, expected_workers)
                    self.assertEqual(len(jobs), 2)
                    for identity, arguments in jobs:
                        self.assertEqual(identity, 'neuroqc:measure-cortical-pair')
                        hemi = arguments[arguments.index('--hemi') + 1]
                        write_json(arguments[arguments.index('--output') + 1], dict(hemi=hemi, status='unknown'))
                stage = SimpleNamespace(root=folder, subjects_dir=folder, subject='sub-test', node=node,
                    spec=dict(threads=threads), metrics={}, run=lambda identity, arguments: calls.append((identity, arguments)),
                    run_concurrent=concurrent, path=lambda port, hemi=None: folder/((hemi or '')+port),
                    volume=lambda port: folder/(port+'.mgz'))
                outputs = cortical_pair(stage)
                self.assertEqual(calls, [])
                self.assertEqual(set(outputs['cortical_pair_measurements']), {'lh', 'rh'})
                self.assertEqual([stage.metrics[hemi]['hemi'] for hemi in ('lh', 'rh')], ['lh', 'rh'])
                self.assertEqual(stage.metrics['cortical_pair_execution']['measurement_workers'], expected_workers)


@unittest.skipUnless(importlib.util.find_spec('vtk') and importlib.util.find_spec('nibabel'), 'VTK and NiBabel required')
class GeometryTimingTests(unittest.TestCase):
    def test_timing_preserves_complete_geometry_checks(self):
        import nibabel as nib
        import numpy as np

        specification = importlib.util.spec_from_file_location('cortical_pair_timing', ROOT/'vendor/neuroqc/measure_cortical_pair.py')
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        vertices, faces = nib.freesurfer.read_geometry(str(ROOT/'model/sugar/source/utils/auxi_data/fsaverage2.sphere'))
        report = module.measure((vertices*.08, faces), (vertices*.10, faces), np.full(len(vertices), 2.), np.arange(len(vertices)))
        self.assertEqual(report['vertex_count'], len(vertices))
        self.assertEqual(report['checked_cortex_white_faces'], len(faces))
        self.assertEqual(report['white_outside_pial_vertices'], 0)
        self.assertEqual(report['cortical_triangle_pair_intersections'], 0)
        self.assertEqual(report['negative_thickness_vertices'], 0)
        self.assertEqual(set(report['timings_seconds']), {'input_validation', 'pial_mesh_validation',
            'cortex_vertex_containment', 'numerical_contact_distances', 'intersection_mesh_preparation',
            'triangle_pair_intersections', 'summary_statistics', 'measure_total'})
        self.assertTrue(all(seconds >= 0 for seconds in report['timings_seconds'].values()))


if __name__ == '__main__':
    unittest.main()

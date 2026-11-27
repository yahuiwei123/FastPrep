"""A fixed medial wall must survive T1/T2 placement and failed recovery."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bin'))
from pial_boundary import FIXED, LEGACY, RECOVER, prepare, verify
from reference_runtime import place_surface, read_json
import test_surface_placement_gate as placement_fixture


@unittest.skipUnless(placement_fixture.SCIENCE_AVAILABLE, 'requires scientific Python')
class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = placement_fixture.PlacementGateTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def stage(self, **kwargs):
        stage = self.fixture.stage(pial=True, **kwargs)
        stage.node['parameters']['medial_wall_strategy'] = RECOVER
        for hemi in ('lh', 'rh'):
            stage.path('cortex_hipamyg_label', hemi).write_text('#!ascii label\n3\n0 0 0 0 0\n1 0 0 0 0\n2 0 0 0 0\n')
        return stage

    def args(self, stage):
        return ['--pial', '--nsmooth', '0', '--i', stage.path('white_surf', 'lh'),
                '--white-surf', stage.path('white_surf', 'lh'),
                '--pin-medial-wall', stage.path('cortex_label', 'lh'),
                '--rip-label', stage.path('cortex_hipamyg_label', 'lh')]

    def test_fixes_complement_and_preserves_cortical_activity(self):
        stage = self.stage()
        original = self.args(stage)
        args, record = prepare(original, FIXED)
        self.assertEqual(args[args.index('--rip-label') + 1], stage.path('cortex_label', 'lh'))
        self.assertEqual(original[original.index('--rip-label') + 1], stage.path('cortex_hipamyg_label', 'lh'))
        self.assertEqual(record['newly_fixed_vertices'], 1)
        self.assertEqual(record['input_constraint']['fixed_vertices'], 2)
        for strategy in (LEGACY, RECOVER):
            self.assertEqual(prepare(original, strategy)[0], original)

    def test_unpinned_t2_input_and_unpinned_output_are_rejected(self):
        stage = self.stage()
        moved = self.fixture.vertices.copy()
        moved[2, 0] += .125
        path = self.root / 'unpinned.pial'
        self.fixture.write_mesh(path, vertices=moved)
        args = self.args(stage)
        args[args.index('--i') + 1] = path
        with self.assertRaisesRegex(ValueError, 'exactly at white'):
            prepare(args, FIXED)
        with self.assertRaisesRegex(ValueError, 'exactly at white'):
            verify(path, stage.path('white_surf', 'lh'), stage.path('cortex_label', 'lh'))

    def test_invalid_activity_geometry_and_presmoothing_are_rejected(self):
        stage = self.stage()
        args = self.args(stage)
        args[args.index('--nsmooth') + 1] = '1'
        with self.assertRaisesRegex(ValueError, 'nsmooth=0'):
            prepare(args, FIXED)
        args = self.args(stage)
        stage.path('cortex_hipamyg_label', 'lh').write_text('#!ascii label\n1\n2 0 0 0 0\n')
        with self.assertRaisesRegex(ValueError, 'contained'):
            prepare(args, FIXED)
        path = self.root / 'wrong-space'
        self.fixture.write_mesh(path, metadata=dict(self.fixture.metadata, cras=[1, 0, 0]))
        with self.assertRaisesRegex(ValueError, 'volume geometry'):
            verify(path, stage.path('white_surf', 'lh'), stage.path('cortex_label', 'lh'))

    def test_healthy_adaptive_run_uses_original_placement_once(self):
        stage = self.stage()
        place_surface(stage)
        runs = [args for _, args in stage.calls if '--o' in args]
        self.assertEqual(len(runs), 2)
        for args in runs:
            self.assertIn('cortex_hipamyg_label', str(args[args.index('--rip-label') + 1]))
        self.assertFalse((self.root / 'placement_recovery').exists())

    def allow_fixed_recovery(self, stage):
        run = stage.run
        fixed = set()

        def wrapped(command, args):
            if '--o' in args and '--rip-label' in args:
                label = Path(args[args.index('--rip-label') + 1])
                if label.name.endswith('.cortex_label'):
                    fixed.add('lh' if '--lh' in args else 'rh')
            run(command, args)
            if command == 'freesurfer:mris_info':
                path = str(args[-1])
                if any(hemi in path for hemi in fixed):
                    (stage.root / stage.logs[-1]['stdout']).write_text('Found 0 intersections\n')
        stage.run = wrapped

    def test_failed_t1_is_preserved_and_reconstructed_once_without_smoothing(self):
        stage = self.stage(bad_port='pial_t1_surf')
        self.allow_fixed_recovery(stage)
        place_surface(stage)
        recovery = self.root / 'placement_recovery/lh/pial_t1_surf-fixed-boundary'
        self.assertTrue((recovery / 'original.pial').is_file())
        record = read_json(recovery / 'recovery.json')
        self.assertEqual(record['before']['self_intersecting_vertices'], 7)
        self.assertEqual(record['after']['self_intersecting_vertices'], 0)
        self.assertEqual(record['status'], 'reconstructed_requires_downstream_qc')
        self.assertEqual(len([args for _, args in stage.calls if '--o' in args]), 3)
        self.assertFalse(any(command.endswith('mris_remove_intersection') for command, _ in stage.calls))

    def test_t2_recovery_rebuilds_t1_with_same_boundary_before_t2(self):
        stage = self.stage(t2=True, bad_port='pial_surf')
        self.allow_fixed_recovery(stage)
        place_surface(stage)
        runs = [args for _, args in stage.calls if '--o' in args and '--lh' in args]
        self.assertEqual([('--mmvol' in args) for args in runs], [False, True, False, True])
        for args in runs[2:]:
            self.assertEqual(args[args.index('--rip-label') + 1], stage.path('cortex_label', 'lh'))
        record = read_json(self.root / 'placement_recovery/lh/pial_surf-fixed-boundary/recovery.json')
        self.assertIn('fixed_t1_placement', record)
        self.assertIn('fixed_t2_placement', record)

    def test_task_override_recovers_old_blueprint_without_mutating_it(self):
        stage = self.stage(bad_port='pial_t1_surf')
        stage.node['parameters']['medial_wall_strategy'] = LEGACY
        stage.spec['medial_wall_strategy'] = RECOVER
        self.allow_fixed_recovery(stage)
        place_surface(stage)
        self.assertEqual(stage.node['parameters']['medial_wall_strategy'], LEGACY)
        record = read_json(self.root / 'placement_recovery/lh/pial_t1_surf-fixed-boundary/recovery.json')
        self.assertEqual(record['status'], 'reconstructed_requires_downstream_qc')
        self.assertEqual(record['original_t1_placement']['medial_wall']['requested_strategy'], RECOVER)

    def test_failed_constrained_rerun_stops_and_retains_failure(self):
        stage = self.stage(bad_port='pial_t1_surf')
        run = stage.run
        def always_bad(command, args):
            run(command, args)
            if command == 'freesurfer:mris_info':
                (stage.root / stage.logs[-1]['stdout']).write_text('Found 7 intersections\n')
        stage.run = always_bad
        with self.assertRaisesRegex(ValueError, 'Constrained pial reconstruction'):
            place_surface(stage)
        self.fixture.assert_no_morphometry(stage)
        record = read_json(self.root / 'placement_recovery/lh/pial_t1_surf-fixed-boundary/recovery.json')
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(record['after']['self_intersecting_vertices'], 7)

    def test_topology_or_missing_measurement_never_triggers_boundary_retry(self):
        for fault in ('euler', 'missing'):
            with self.subTest(fault=fault):
                stage = self.stage(bad_port='pial_t1_surf', fault=fault)
                with self.assertRaises(ValueError):
                    place_surface(stage)
                self.assertEqual(len([args for _, args in stage.calls if '--o' in args]), 1)
                self.assertFalse((self.root / 'placement_recovery').exists())


if __name__ == '__main__':
    unittest.main()

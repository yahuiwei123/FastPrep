"""Native topology smoothing controls retain the existing repair gates."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from reference_runtime import fix_initialized_topology, repair_initialized_intersections


class Stage:
    def __init__(self, root):
        self.root = root
        self.node = {"parameters": {"topology_seed": 1234, "qsphere_passes": 6,
                     "qsphere_averages": 128, "required_euler_number": 2,
                     "required_total_defect_index": 0, "required_self_intersecting_vertices": 0}}
        self.spec = {"threads": 2, "topology_backend": "cpu"}
        self.subject = "sub-test_ses-01"
        self.metrics = {}
        self.calls = []

    def path(self, port, hemi=None):
        path = self.root / "inputs" / ((hemi + "." if hemi else "") + port)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"input surface")
        return path

    def hemi(self, port, hemi):
        path = self.root / "outputs" / (hemi + "." + port)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, command, arguments):
        self.calls.append((command, arguments))
        if command == "freesurfer:mris_remove_intersection":
            arguments[-1].write_bytes(b"repaired surface")
        if command == "freesurfer:mris_fix_topology":
            name = arguments[arguments.index("-out") + 1]
            path = self.root / "outputs" / (arguments[-1] + "." + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("seed=" + str(arguments[arguments.index("-seed") + 1]))


def checks(intersections=0, euler=2):
    return {"euler_number": euler, "total_defect_index": 0,
            "self_intersecting_vertices": intersections}


class TopologyRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.stage = Stage(Path(self.directory.name))
        winding = patch('reference_runtime.orient_initialized_surface', return_value={})
        winding.start()
        self.addCleanup(winding.stop)

    def run_topology(self):
        with patch("reference_runtime.surface_repair_checks", return_value=checks()), \
                patch("reference_runtime.surface_topology_geometry", return_value={}):
            fix_initialized_topology(self.stage, "lh")
        return next(arguments for command, arguments in self.stage.calls
                    if command == "freesurfer:mris_fix_topology")

    def test_default_preserves_zero_smoothing(self):
        arguments = self.run_topology()
        self.assertEqual(arguments[arguments.index("-s") + 1], 0)
        self.assertEqual(self.stage.metrics["lh"]["topology_smoothing_iterations"], 0)

    def test_explicit_smoothing_forwarded_and_recorded(self):
        self.stage.node["parameters"]["topology_smoothing_iterations"] = 50
        arguments = self.run_topology()
        self.assertEqual(arguments[arguments.index("-s") + 1], 50)
        self.assertEqual(arguments[arguments.index("-seed") + 1], 1234)
        self.assertEqual(self.stage.metrics["lh"]["topology_smoothing_iterations"], 50)

    def test_invalid_smoothing_rejected_before_commands(self):
        for value in (-1, True, 1.5, "50", None):
            with self.subTest(value=value):
                self.stage.node["parameters"]["topology_smoothing_iterations"] = value
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    fix_initialized_topology(self.stage, "lh")
                self.assertEqual(self.stage.calls, [])

    def test_native_repair_seed_does_not_change_preparation_seed(self):
        self.stage.node["parameters"]["topology_repair_seed"] = 4321
        arguments = self.run_topology()
        self.assertEqual(arguments[arguments.index("-seed") + 1], 4321)
        smoothing = next(arguments for command, arguments in self.stage.calls if command == "freesurfer:mris_smooth")
        self.assertEqual(smoothing[smoothing.index("-seed") + 1], 1234)
        self.assertEqual(self.stage.metrics["lh"]["topology_repair_seed"], 4321)

    def test_invalid_native_repair_seed_rejected(self):
        for value in (-1, True, 1.5, "4321", None):
            with self.subTest(value=value):
                self.stage.node["parameters"]["topology_repair_seed"] = value
                with self.assertRaisesRegex(ValueError, "nonnegative integer"):
                    fix_initialized_topology(self.stage, "lh")
                self.assertEqual(self.stage.calls, [])

    def test_topology_gate_remains_strict(self):
        with patch("reference_runtime.surface_repair_checks", side_effect=[checks(), checks(euler=0)]):
            with self.assertRaisesRegex(ValueError, "Euler=2 and zero defects"):
                fix_initialized_topology(self.stage, "lh")
        self.assertNotIn("lh", self.stage.metrics)

    def recovery(self, measurements):
        self.stage.node["parameters"]["topology_recovery_attempts"] = [{"seed": 4321, "smoothing_iterations": 10}]
        with patch("reference_runtime.surface_repair_checks", side_effect=measurements), \
                patch("reference_runtime.surface_repair_geometry", return_value={}), \
                patch("reference_runtime.surface_topology_geometry", return_value={}):
            fix_initialized_topology(self.stage, "lh")

    def test_cleanup_success_keeps_original_seed_and_skips_retry(self):
        self.recovery([checks(), checks(70), checks()])
        self.assertEqual(sum(command == "freesurfer:mris_fix_topology" for command, _ in self.stage.calls), 1)
        self.assertEqual(self.stage.metrics["lh"]["selected_topology_attempt"], 0)
        self.assertEqual(self.stage.hemi("orig_topofix_surf", "lh").read_text(), "seed=1234")

    def test_stalled_cleanup_selects_verified_retry_and_preserves_failed_candidate(self):
        self.recovery([checks(), checks(178), checks(33), checks(), checks()])
        details = self.stage.metrics["lh"]
        self.assertEqual(details["selected_topology_attempt"], 1)
        self.assertEqual(details["topology_repair_seed"], 4321)
        self.assertEqual(details["topology_smoothing_iterations"], 10)
        self.assertEqual(details["topology_recovery_attempts"][0]["native_cleanup"], checks(33))
        self.assertEqual(Path(details["topology_recovery_attempts"][0]["surface"]).read_text(), "seed=1234")
        self.assertEqual(self.stage.hemi("orig_topofix_surf", "lh").read_text(), "seed=4321")

    def test_local_cleanup_is_attempted_before_another_ga_seed(self):
        self.stage.node['parameters']['local_fairing_recovery'] = True
        with patch('intersection_recovery.recover_intersections', return_value=(checks(), {'validated': True})) as repair:
            self.recovery([checks(), checks(178), checks(13)])
        repair.assert_called_once()
        self.assertEqual(sum(command == 'freesurfer:mris_fix_topology' for command, _ in self.stage.calls), 1)
        history = self.stage.metrics['lh']['topology_recovery_attempts']
        self.assertEqual(history[0]['native_cleanup'], checks(13))
        self.assertTrue(history[0]['local_fairing_recovery']['validated'])

    def test_exhausted_local_cleanup_still_requires_a_verified_ga_retry(self):
        from intersection_recovery import RecoveryExhausted

        self.stage.node['parameters']['local_fairing_recovery'] = True
        with patch('intersection_recovery.recover_intersections', side_effect=RecoveryExhausted('bounded failure')):
            self.recovery([checks(), checks(178), checks(13), checks(2), checks()])
        self.assertEqual(self.stage.metrics['lh']['selected_topology_attempt'], 1)
        self.assertEqual(self.stage.metrics['lh']['topology_recovery_attempts'][0]['native_cleanup'], checks(13))

    def test_exhausted_recovery_does_not_publish_selected_topology(self):
        with self.assertRaisesRegex(ValueError, "Topology recovery exhausted"):
            self.recovery([checks(), checks(178), checks(33), checks(40), checks(20)])
        self.assertFalse(self.stage.hemi("orig_topofix_surf", "lh").exists())
        self.assertEqual(len(self.stage.metrics["lh"]["topology_recovery_attempts"]), 2)

    def test_invalid_retry_configuration_fails_before_commands(self):
        for invalid in (None, {}, [{"seed": True, "smoothing_iterations": 10}],
                        [{"seed": 4321}], [{"seed": 4321, "smoothing_iterations": -1}]):
            with self.subTest(value=invalid):
                self.stage.node["parameters"]["topology_recovery_attempts"] = invalid
                with self.assertRaisesRegex(ValueError, "topology recovery attempt|topology_recovery_attempts"):
                    fix_initialized_topology(self.stage, "lh")
                self.assertEqual(self.stage.calls, [])

    def test_residual_intersections_stop_before_premesh_publication(self):
        with patch("reference_runtime.surface_repair_checks",
                   side_effect=[checks(), checks(97), checks(28)]), \
                patch("reference_runtime.surface_repair_geometry", return_value={}):
            with self.assertRaisesRegex(ValueError, "zero self-intersections before placement"):
                repair_initialized_intersections(self.stage, "lh")
        self.assertNotIn("lh", self.stage.metrics)
        self.assertTrue(self.stage.hemi("orig_surf", "lh").exists())
        self.assertFalse(self.stage.hemi("orig_premesh_surf", "lh").exists())

    def test_bad_repaired_topology_does_not_publish_premesh(self):
        for invalid in (checks(euler=0), {**checks(), "total_defect_index": 1}):
            with self.subTest(measurement=invalid), \
                    patch("reference_runtime.surface_repair_checks", side_effect=[checks(), checks(97), invalid]):
                with self.assertRaisesRegex(ValueError, "Euler=2 and zero defects"):
                    repair_initialized_intersections(self.stage, "lh")
        self.assertFalse(self.stage.hemi("orig_premesh_surf", "lh").exists())

    def test_changed_mesh_identity_does_not_publish_premesh(self):
        with patch("reference_runtime.surface_repair_checks", return_value=checks()), \
                patch("reference_runtime.surface_repair_geometry", side_effect=ValueError("changed face identity")):
            with self.assertRaisesRegex(ValueError, "face identity"):
                repair_initialized_intersections(self.stage, "lh")
        self.assertFalse(self.stage.hemi("orig_premesh_surf", "lh").exists())

    def test_zero_intersections_publish_identical_premesh(self):
        with patch("reference_runtime.surface_repair_checks", return_value=checks()), \
                patch("reference_runtime.surface_repair_geometry", return_value={}):
            repair_initialized_intersections(self.stage, "lh")
        self.assertEqual(self.stage.hemi("orig_surf", "lh").read_bytes(),
                         self.stage.hemi("orig_premesh_surf", "lh").read_bytes())


if __name__ == "__main__":
    unittest.main()

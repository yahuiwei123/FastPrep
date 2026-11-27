"""Keep the T2-specific T1 initialization isolated from native defaults."""

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from t2_pial import initial_t1_pial_profile


class PialInitializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.stats = Path(self.directory.name) / "autodet.dat"
        self.content = ("pial_border_low 60\npial_border_hi 80\npial_outside_hi 75\n"
                        "pial_outside_low 10\npial_inside_hi 110\n")
        self.stats.write_text(self.content)

    def test_t2_initial_thresholds_and_provenance(self):
        arguments, profile = initial_t1_pial_profile({"t2_initial_t1_threshold_scale": 0.9}, True, self.stats)
        self.assertEqual(arguments, ["--pial_border_low", "54", "--pial_border_hi", "72", "--pial_outside_hi", "67.5"])
        self.assertEqual(profile["scope"], "initial_T1_pial_only")
        self.assertEqual(profile["baseline_thresholds"]["pial_border_low"], 60)
        self.assertEqual(profile["effective_thresholds"]["pial_border_low"], 54)
        self.assertEqual(profile["effective_thresholds"]["pial_inside_hi"], 110)
        self.assertEqual(profile["effective_thresholds"]["pial_outside_low"], 10)
        self.assertTrue(profile["applied"])
        self.assertEqual(self.stats.read_text(), self.content)

    def test_inactive_profiles_do_not_read_stats(self):
        self.stats.unlink()
        for parameters, available in (({}, True), ({"t2_initial_t1_threshold_scale": 1}, True),
                                      ({"t2_initial_t1_threshold_scale": 0.9}, False),
                                      ({"t2_initial_t1_threshold_scale": 0.85}, False)):
            with self.subTest(parameters=parameters, available=available):
                arguments, profile = initial_t1_pial_profile(parameters, available, self.stats)
                self.assertEqual(arguments, [])
                self.assertEqual(profile["effective_threshold_scale"], 1)
                self.assertFalse(profile["applied"])

    def test_configured_default_preserves_native_initialization(self):
        import json

        graph = json.loads((ROOT / "conf/workflow.json").read_text())
        node = next(node for node in graph["nodes"] if node["node_id"] == "anat_pial_surface")
        self.assertEqual(node["parameters"]["t2_initial_t1_threshold_scale"], 1.0)
        self.stats.unlink()
        arguments, profile = initial_t1_pial_profile(node["parameters"], True, self.stats)
        self.assertEqual(arguments, [])
        self.assertEqual(profile["effective_threshold_scale"], 1.0)
        self.assertFalse(profile["applied"])

    def test_explicit_stronger_candidate_remains_available(self):
        arguments, profile = initial_t1_pial_profile({"t2_initial_t1_threshold_scale": 0.85}, True, self.stats)
        self.assertEqual(arguments, ["--pial_border_low", "51", "--pial_border_hi", "68", "--pial_outside_hi", "63.75"])
        self.assertEqual(profile["effective_threshold_scale"], 0.85)
        self.assertEqual(profile["effective_thresholds"]["pial_inside_hi"], 110)
        self.assertEqual(profile["effective_thresholds"]["pial_outside_low"], 10)
        self.assertEqual(self.stats.read_text(), self.content)

    def test_invalid_scale_fails(self):
        for scale in (0, -1, 1.1, float("nan"), float("inf"), True, "0.9", None):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "threshold_scale"):
                initial_t1_pial_profile({"t2_initial_t1_threshold_scale": scale}, True, self.stats)

    def test_invalid_stats_fail_without_fallback(self):
        for content in (self.content.replace("60", "nan"), self.content.replace("60", "-1"),
                        self.content.replace("pial_border_low 60\n", ""), self.content + "pial_border_low 60\n",
                        self.content.replace("60", "90"), self.content.replace("60", "not-a-number"),
                        self.content.replace("pial_border_low 60", "pial_border_low 60 extra")):
            with self.subTest(content=content), self.assertRaises(ValueError):
                self.stats.write_text(content)
                initial_t1_pial_profile({"t2_initial_t1_threshold_scale": 0.9}, True, self.stats)

    def test_scale_cannot_cross_unmodified_outside_floor(self):
        with self.assertRaisesRegex(ValueError, "threshold ordering"):
            initial_t1_pial_profile({"t2_initial_t1_threshold_scale": 0.1}, True, self.stats)


if __name__ == "__main__":
    unittest.main()

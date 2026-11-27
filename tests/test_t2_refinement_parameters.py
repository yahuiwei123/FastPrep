from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from t2_pial import t2_refinement_profile


class T2RefinementParametersTests(unittest.TestCase):
    def test_native_default_unchanged(self):
        arguments, profile = t2_refinement_profile({}, True)
        self.assertEqual(arguments, [])
        self.assertFalse(profile["applied"])
        self.assertEqual(profile["effective_parameters"]["mm_min_inside"], 110)
        self.assertEqual(profile["effective_parameters"]["location"], 0.5)
        self.assertEqual(profile["effective_parameters"]["curv"], 0.5)
        self.assertEqual(profile["effective_parameters"]["nspring"], 1.0)

    def test_selected_production_defaults(self):
        import json

        graph = json.loads((Path(__file__).resolve().parents[1] / "conf/workflow.json").read_text())
        parameters = next(node["parameters"] for node in graph["nodes"] if node["node_id"] == "anat_pial_surface")
        self.assertEqual(parameters["t2_initial_t1_threshold_scale"], 1.0)
        self.assertEqual(parameters["t2_refinement_overrides"], {"location": 0.6, "curv": 0.8, "nspring": 0.6})
        arguments, profile = t2_refinement_profile(parameters, True)
        self.assertEqual(arguments, ["--location", "0.6", "--curv", "0.8", "--nspring", "0.6"])
        self.assertEqual(profile["effective_parameters"], {"mm_min_inside": 110, "mm_max_inside": 300,
            "mm_min_outside": 130, "mm_max_outside": 300, "location": 0.6, "curv": 0.8, "nspring": 0.6})
        self.assertTrue(profile["applied"])
        arguments, profile = t2_refinement_profile(parameters, False)
        self.assertEqual(arguments, [])
        self.assertFalse(profile["applied"])

    def test_candidate_parameters_and_provenance(self):
        parameters = {"t2_refinement_overrides": {"mm_min_inside": 100, "location": 0.75}}
        arguments, profile = t2_refinement_profile(parameters, True)
        self.assertEqual(arguments, ["--mm_min_inside", "100", "--location", "0.75"])
        self.assertEqual(profile["effective_parameters"]["mm_max_inside"], 300)
        self.assertTrue(profile["applied"])
        self.assertEqual(profile["scope"], "T2_refinement_only")
        self.assertEqual(parameters["t2_refinement_overrides"]["mm_min_inside"], 100)

    def test_t1_only_has_no_overrides(self):
        arguments, profile = t2_refinement_profile(
            {"t2_refinement_overrides": {"mm_min_inside": 100, "location": 0.75}}, False)
        self.assertEqual(arguments, [])
        self.assertFalse(profile["applied"])
        self.assertIsNone(profile["effective_parameters"])

    def test_extreme_dark_floor_sweep_changes_only_inside_floor(self):
        for floor in (50, 60, 70, 110):
            with self.subTest(floor=floor):
                arguments, profile = t2_refinement_profile(
                    {"t2_refinement_overrides": {"mm_min_inside": floor}}, True)
                self.assertEqual(arguments, ["--mm_min_inside", str(floor)])
                self.assertEqual(profile["effective_parameters"]["location"], 0.5)
                self.assertEqual(profile["effective_parameters"]["mm_min_outside"], 130)
                self.assertEqual(profile["effective_parameters"]["mm_max_inside"], 300)

    def test_invalid_parameters_fail(self):
        for overrides in (None, [], {"unknown": 1}, {"location": True}, {"location": "0.75"},
                          {"location": 0}, {"location": float("nan")}, {"location": float("inf")},
                          {"mm_min_inside": -1}, {"mm_min_inside": 300},
                          {"mm_max_outside": 120}, {"mm_min_outside": 100},
                          {"curv": -1}, {"curv": float("nan")}, {"nspring": True},
                          {"nspring": 1.01}, {"nspring": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                t2_refinement_profile({"t2_refinement_overrides": overrides}, True)


if __name__ == "__main__":
    unittest.main()

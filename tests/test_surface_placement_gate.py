"""Self-intersections gate both initialization and written placement outputs."""

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from reference_runtime import NativeStage, place_surface, read_json, surface_repair_checks, check_placed_surface
from qc_runtime import Evidence


SCIENCE_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("numpy", "nibabel", "scipy"))


class MeasurementStage:
    each_hemisphere = NativeStage.each_hemisphere

    def __init__(self, root):
        self.root = root
        self.logs = []
        self.calls = []
        self.topology_text = "euler # = 4 - 6 + 4 = 2 --> 0 holes\ntotal defect index = 0\n"
        self.intersection_text = "Found 0 intersections\n"

    def run(self, command, arguments):
        self.calls.append((command, arguments))
        stdout, stderr = str(len(self.logs)) + ".stdout", str(len(self.logs)) + ".stderr"
        text = self.topology_text if command.endswith("mris_euler_number") else self.intersection_text
        (self.root / stdout).write_text(text)
        (self.root / stderr).write_text("")
        self.logs.append({"stdout": stdout, "stderr": stderr})


class MeasurementTests(unittest.TestCase):
    def test_missing_or_ambiguous_measurements_are_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = MeasurementStage(Path(directory))
            for field in ("intersection_text", "topology_text"):
                original = getattr(stage, field)
                for invalid in ("", original + original):
                    with self.subTest(field=field, invalid=invalid):
                        setattr(stage, field, invalid)
                        with self.assertRaisesRegex(ValueError, "Missing or ambiguous"):
                            surface_repair_checks(stage, "synthetic.white")
                setattr(stage, field, original)


@unittest.skipUnless(SCIENCE_AVAILABLE, "Requires NumPy, NiBabel and SciPy")
class PlacementGateTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], dtype=float)
        self.faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        self.metadata = dict(head=[2, 0, 20], valid="1", filename="synthetic.mgz", volume=[8, 8, 8],
                             voxelsize=[1, 1, 1], xras=[1, 0, 0], yras=[0, 1, 0], zras=[0, 0, 1], cras=[0, 0, 0])

    def write_mesh(self, path, *, vertices=None, faces=None, metadata=None):
        from nibabel.freesurfer.io import write_geometry

        write_geometry(path, self.vertices if vertices is None else vertices,
                       self.faces if faces is None else faces,
                       volume_info=self.metadata if metadata is None else metadata)

    def stage(self, *, pial=False, t2=False, bad_port=None, bad_hemi="lh", fault="intersections"):
        stage = MeasurementStage(self.root)
        identity = "anat_pial_surface" if pial else "anat_white_surface"
        stage.node = copy.deepcopy(next(node for node in read_json(ROOT / "conf/workflow.json")["nodes"]
                                        if node["node_id"] == identity))
        stage.node["parameters"].pop("t2_initial_t1_threshold_scale", None)
        # These fixtures exercise the original bounded native-cleanup policy.
        # The fixed-boundary recovery has separate integration tests.
        stage.node["parameters"]["medial_wall_strategy"] = "legacy-final-pin"
        stage.node["parameters"]["hemisphere_workers"] = 1
        stage.metrics = {}
        stage.recovery_succeeds = False
        stage.spec = {"threads": 4, "t1_placement_backend": "cpu", "t2_pial_backend": "cpu"}
        stage.path = lambda port, hemi=None: self.root / (str(hemi) + "." + port)
        stage.hemi = stage.path
        original_run = stage.run

        def run(command, arguments):
            target = next((Path(argument) for argument in arguments if str(argument).endswith(str(bad_port))), None)
            bad = target is not None and target.name.startswith(bad_hemi + ".")
            if any(str(argument).endswith(".repaired") for argument in arguments) and not stage.recovery_succeeds:
                bad = True
            stage.intersection_text = "Found 7 intersections\n" if bad and fault == "intersections" else "Found 0 intersections\n"
            if bad and fault == "missing":
                stage.intersection_text = "unrecognized output\n"
            if bad and fault == "euler":
                stage.topology_text = "euler # = 4 - 6 + 4 = 0 --> 1 holes\ntotal defect index = 1\n"
            original_run(command, arguments)
            if command == "freesurfer:mris_remove_intersection":
                import shutil
                shutil.copyfile(arguments[0], arguments[1])
            if "--o" in arguments:
                output = Path(arguments[arguments.index("--o") + 1])
                vertices, faces, metadata = self.vertices.copy(), self.faces.copy(), dict(self.metadata)
                if bad and fault == "faces":
                    faces = faces[::-1].copy()
                if bad and fault == "nonfinite":
                    vertices[0, 0] = float("nan")
                if bad and fault == "geometry":
                    metadata["cras"] = [1, 0, 0]
                self.write_mesh(output, vertices=vertices, faces=faces, metadata=metadata)
                if bad and fault == "input_changed":
                    self.write_mesh(stage.path("white_surf" if pial else "orig_surf", bad_hemi), vertices=vertices * 2)

        stage.run = run
        for hemi in ("lh", "rh"):
            self.write_mesh(stage.path("white_surf" if pial else "orig_surf", hemi))
        if pial:
            for hemi in ("lh", "rh"):
                stage.path("cortex_label", hemi).write_text("#!ascii label\n2\n0 1 0 0 0\n1 0 1 0 0\n")
            self.enterContext(patch("t2_pial.prepare_t2", return_value=(
                self.root / "T2.mgz" if t2 else None, {"mode": "T2_refined" if t2 else "T1_only"}, [])))
            self.enterContext(patch("t2_pial.refinement_manifest", return_value=self.root / "pial.json"))
        return stage

    def assert_no_morphometry(self, stage, hemi="lh"):
        self.assertFalse(any(arguments[0] in ("--curv-map", "--area-map", "--thickness")
                             and any(Path(str(value)).name.startswith(hemi + ".") for value in arguments)
                             for _, arguments in stage.calls))

    def assert_nsmooth(self, stage, expected, *, t2=False):
        placements = [arguments for _, arguments in stage.calls if "--o" in arguments]
        self.assertEqual(len(placements), 4 if t2 else 2)
        for arguments in placements:
            self.assertEqual(arguments.count("--nsmooth"), 1)
            self.assertEqual(arguments[arguments.index("--nsmooth") + 1], str(expected))
        for hemi in ("lh", "rh"):
            self.assertEqual(stage.metrics["t1_placement"][hemi]["nsmooth"], expected)
            if t2:
                self.assertEqual(stage.metrics["pial_refinement"]["placement_backend"][hemi]["nsmooth"], expected)

    def test_nsmooth_defaults_match_recon_all_stages(self):
        for pial, t2, expected in ((False, False, 5), (True, False, 0), (True, True, 0)):
            with self.subTest(pial=pial, t2=t2):
                stage = self.stage(pial=pial, t2=t2)
                self.assertEqual(stage.node["parameters"]["nsmooth"], expected)
                place_surface(stage)
                self.assert_nsmooth(stage, expected, t2=t2)

    def test_missing_nsmooth_uses_stage_default(self):
        for pial, t2, expected in ((False, False, 5), (True, False, 0), (True, True, 0)):
            with self.subTest(pial=pial, t2=t2):
                stage = self.stage(pial=pial, t2=t2)
                stage.node["parameters"].pop("nsmooth")
                place_surface(stage)
                self.assert_nsmooth(stage, expected, t2=t2)

    def test_explicit_nsmooth_is_forwarded_including_zero(self):
        for pial, t2 in ((False, False), (True, False), (True, True)):
            for value in (0, 3):
                with self.subTest(pial=pial, t2=t2, nsmooth=value):
                    stage = self.stage(pial=pial, t2=t2)
                    stage.node["parameters"]["nsmooth"] = value
                    place_surface(stage)
                    self.assert_nsmooth(stage, value, t2=t2)

    def test_invalid_nsmooth_rejected_before_processing(self):
        for pial in (False, True):
            for value in (-1, True, False, 1.5, "5", None):
                with self.subTest(pial=pial, nsmooth=value):
                    stage = self.stage(pial=pial)
                    stage.node["parameters"]["nsmooth"] = value
                    with patch("t2_pial.prepare_t2") as prepare, self.assertRaisesRegex(ValueError, "nonnegative integer"):
                        place_surface(stage)
                    prepare.assert_not_called()
                    self.assertEqual(stage.calls, [])
                    self.assertEqual(stage.metrics, {})

    def test_white_measurement_follows_placement_for_both_hemispheres(self):
        stage = self.stage()
        place_surface(stage)
        for hemi in ("lh", "rh"):
            target = stage.hemi("white_preaparc_surf", hemi)
            placement_index = next(index for index, (_, args) in enumerate(stage.calls) if "--o" in args and target in args)
            check_index = next(index for index, (command, args) in enumerate(stage.calls)
                               if command == "freesurfer:mris_info" and target in args)
            morphometry_index = next(index for index, (_, args) in enumerate(stage.calls)
                                    if args[0] == "--curv-map" and stage.hemi("white_surf", hemi) in args)
            self.assertLess(placement_index, check_index)
            self.assertLess(check_index, morphometry_index)
            self.assertEqual(target.read_bytes(), stage.hemi("white_surf", hemi).read_bytes())
            self.assertEqual(stage.metrics["placement_checks"][hemi]["white_preaparc_surf"]["self_intersecting_vertices"], 0)

    def test_bad_white_stops_before_copy_and_morphometry(self):
        for fault in ("intersections", "missing", "euler", "faces", "nonfinite", "geometry", "input_changed"):
            with self.subTest(fault=fault):
                stage = self.stage(bad_port="white_preaparc_surf", fault=fault)
                with self.assertRaises(ValueError):
                    place_surface(stage)
                self.assertFalse(stage.hemi("white_surf", "lh").exists())
                self.assert_no_morphometry(stage)

    def test_right_hemisphere_failure_is_not_hidden_by_left_success(self):
        stage = self.stage(bad_port="white_preaparc_surf", bad_hemi="rh")
        with self.assertRaisesRegex(ValueError, "zero self-intersections"):
            place_surface(stage)
        self.assertTrue(stage.hemi("white_surf", "lh").exists())
        self.assertFalse(stage.hemi("white_surf", "rh").exists())
        self.assert_no_morphometry(stage, "rh")

    def test_gpu_selected_output_has_the_same_gate(self):
        stage = self.stage(bad_port="white_preaparc_surf")
        stage.spec["t1_placement_backend"] = "gpu"

        def refine(stage, arguments, hemi, role):
            self.assertEqual(arguments[arguments.index("--nsmooth") + 1], "5")
            stage.run("fastprep:" + role + "-cuda-" + hemi, arguments)
            return {"selected": "cuda-ordered-signed"}

        with patch("t2_pial_backend.refine", side_effect=refine), self.assertRaisesRegex(ValueError, "zero self-intersections"):
            place_surface(stage)
        self.assertFalse(stage.hemi("white_surf", "lh").exists())

    def test_t1_pial_gate_precedes_t2_refinement(self):
        stage = self.stage(pial=True, t2=True, bad_port="pial_t1_surf")
        with self.assertRaisesRegex(ValueError, "zero self-intersections"):
            place_surface(stage)
        self.assertFalse(any("--mmvol" in args for _, args in stage.calls))
        self.assertFalse(stage.hemi("pial_surf", "lh").exists())
        self.assert_no_morphometry(stage)

    def test_t2_pial_gate_precedes_morphometry(self):
        stage = self.stage(pial=True, t2=True, bad_port="pial_surf")
        with self.assertRaisesRegex(ValueError, "zero self-intersections"):
            place_surface(stage)
        self.assertEqual(stage.metrics["placement_checks"]["lh"]["pial_t1_surf"]["self_intersecting_vertices"], 0)
        self.assert_no_morphometry(stage)

    def test_t1_only_pial_checks_then_copies(self):
        stage = self.stage(pial=True)
        place_surface(stage)
        for hemi in ("lh", "rh"):
            self.assertEqual(stage.hemi("pial_t1_surf", hemi).read_bytes(), stage.hemi("pial_surf", hemi).read_bytes())
            self.assertEqual(set(stage.metrics["placement_checks"][hemi]), {"pial_t1_surf"})

    def test_successful_t2_checks_both_placements_and_preserves_white(self):
        stage = self.stage(pial=True, t2=True)
        original = {hemi: stage.path("white_surf", hemi).read_bytes() for hemi in ("lh", "rh")}
        place_surface(stage)
        for hemi in ("lh", "rh"):
            self.assertEqual(stage.path("white_surf", hemi).read_bytes(), original[hemi])
            self.assertEqual(set(stage.metrics["placement_checks"][hemi]), {"pial_t1_surf", "pial_surf"})

    def test_pial_native_recovery_preserves_original_and_precedes_morphometry(self):
        stage = self.stage(pial=True, bad_port="pial_t1_surf")
        stage.recovery_succeeds = True
        place_surface(stage)
        checks = stage.metrics["placement_checks"]["lh"]["pial_t1_surf"]
        self.assertEqual(checks["self_intersecting_vertices"], 0)
        recovery = checks["intersection_recovery"]
        self.assertEqual(recovery["before"]["self_intersecting_vertices"], 7)
        self.assertEqual(recovery["scientific_approval"], "not_performed")
        self.assertTrue(Path(recovery["original_surface"]).is_file())
        self.assertTrue(Path(recovery["candidate_surface"]).is_file())
        cleanup_index = next(i for i, (cmd, _) in enumerate(stage.calls) if cmd.endswith("mris_remove_intersection"))
        metric_index = next(i for i, (_, args) in enumerate(stage.calls) if args[0] == "--thickness")
        self.assertLess(cleanup_index, metric_index)
        self.assertEqual(stage.hemi("pial_surf", "lh").read_bytes(), Path(recovery["candidate_surface"]).read_bytes())
        self.assertNotIn("intersection_recovery", stage.metrics["placement_checks"]["rh"]["pial_t1_surf"])

    def test_clean_pial_never_runs_recovery(self):
        stage = self.stage(pial=True)
        place_surface(stage)
        self.assertFalse(any(cmd.endswith("mris_remove_intersection") for cmd, _ in stage.calls))
        self.assertFalse((self.root / "placement_recovery").exists())

    def test_recovery_rejects_excess_movement(self):
        stage = self.stage(pial=True, bad_port="pial_t1_surf")
        stage.recovery_succeeds = True
        original_run = stage.run
        def run(command, arguments):
            original_run(command, arguments)
            if command.endswith("mris_remove_intersection"):
                moved = self.vertices.copy()
                moved[0, 0] += 1.1
                self.write_mesh(arguments[1], vertices=moved)
        stage.run = run
        with self.assertRaisesRegex(ValueError, "safeguards"):
            place_surface(stage)
        self.assert_no_morphometry(stage)
        raw = self.root / "placement_recovery/lh/pial_t1_surf/lh.pial_t1_surf"
        self.assertEqual(raw.read_bytes(), stage.hemi("pial_t1_surf", "lh").read_bytes())
        journal = read_json(raw.parent / "recovery.json")
        self.assertEqual(journal["status"], "failed")
        self.assertGreater(journal["max_cortical_displacement_mm"], 1)

    def test_recovery_never_accepts_corrupt_candidate_or_unmeasured_result(self):
        import nibabel as nib
        base = self.root
        for fault in ("faces", "geometry", "nonfinite", "missing"):
            with self.subTest(fault=fault):
                self.root = base / fault
                self.root.mkdir()
                stage = self.stage(pial=True, bad_port="pial_t1_surf")
                stage.recovery_succeeds = True
                original_run = stage.run
                def run(command, arguments):
                    original_run(command, arguments)
                    if command.endswith("mris_remove_intersection"):
                        vertices, faces, metadata = self.vertices.copy(), self.faces.copy(), dict(self.metadata)
                        if fault == "faces":
                            faces = faces[::-1].copy()
                        elif fault == "geometry":
                            metadata["cras"] = [1, 0, 0]
                        elif fault == "nonfinite":
                            vertices[0, 0] = float("nan")
                        self.write_mesh(arguments[1], vertices=vertices, faces=faces, metadata=metadata)
                    if fault == "missing" and command.endswith("mris_info") and str(arguments[-1]).endswith(".repaired"):
                        (stage.root / stage.logs[-1]["stdout"]).write_text("unrecognized output\n")
                stage.run = run
                with self.assertRaises(ValueError):
                    place_surface(stage)
                self.assert_no_morphometry(stage)
                self.assertFalse(stage.hemi("pial_surf", "lh").exists())

    def test_bad_topology_is_not_sent_to_local_pial_cleanup(self):
        stage = self.stage(pial=True, bad_port="pial_t1_surf", fault="euler")
        with self.assertRaisesRegex(ValueError, "Euler=2"):
            place_surface(stage)
        self.assertFalse(any(cmd.endswith("mris_remove_intersection") for cmd, _ in stage.calls))
        self.assert_no_morphometry(stage)

    def evidence(self, port, *, marker=True, identity="anat_remove_intersection", intersections="Found 7 intersections\n", bad_mesh=False):
        path = self.root / ("lh." + port)
        self.write_mesh(path)
        metrics = {"lh": {"self_intersection_gate": "post_placement"}} if marker else {}
        evidence = Evidence({"stages": {identity: {"metrics": metrics}}}, {"node_id": identity}, {}, self.root / "qc")

        def tool(relative, arguments, **kwargs):
            if arguments[0] == "metrics":
                return {"metrics": {"boundaryEdgeCount": int(bad_mesh), "nonmanifoldEdgeCount": 0, "degenerateFaceCount": 0}}
            if arguments[0] == "mris_info":
                return intersections
            return None

        with patch.object(evidence, "tool", side_effect=tool):
            evidence.mesh(port, {"path": path, "hemi": "lh"})
        return evidence

    def test_qc_rejects_initialized_intersections_even_with_old_deferral_marker(self):
        for port in ("orig_surf", "orig_premesh_surf"):
            with self.subTest(port=port):
                evidence = self.evidence(port)
                self.assertIn("SURFACE.MESH", evidence.failures)
                self.assertEqual(evidence.missing, [])
                measured = evidence.measurements["surface"][0]
                self.assertEqual(measured["self_intersecting_vertices"], 7)
                self.assertEqual(measured["self_intersection_status"], "measured")
                self.assertEqual(measured["self_intersection_gate"], "current_stage")
                self.assertIn("SURFACE.MESH", self.evidence(port, marker=False).failures)

    def test_qc_never_defers_final_meshes_or_other_sources(self):
        for port in ("white_surf", "white_preaparc_surf", "pial_t1_surf", "pial_surf"):
            with self.subTest(port=port):
                self.assertIn("SURFACE.MESH", self.evidence(port).failures)
        self.assertIn("SURFACE.MESH", self.evidence("orig_surf", identity="old_stage").failures)

    def test_init_retains_mesh_and_measurement_checks(self):
        self.assertIn("SURFACE.MESH", self.evidence("orig_surf", bad_mesh=True).failures)
        for text in ("", "Found 0 intersections\nFound 7 intersections\n"):
            with self.subTest(text=text):
                evidence = self.evidence("orig_surf", intersections=text)
                self.assertTrue(evidence.missing)
                self.assertIsNone(evidence.measurements["surface"][0]["self_intersecting_vertices"])


if __name__ == "__main__":
    unittest.main()

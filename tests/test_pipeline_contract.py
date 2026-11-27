"""Scientific handoff and portable-output regression checks."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bin"))
from assemble import assemble
from preflight import validate_graph
from prepare import executable, resource_files
from reference_runtime import artifact_path, place_surface, read_json, write_json


class PipelineContractTests(unittest.TestCase):
    def test_brainvol_stats_uses_required_subject_flag(self):
        from reference_runtime import surf2vol

        calls = []

        class Stage:
            node = {"node_id": "anat_apas2aseg"}
            spec = {"threads": 2}
            subject = "sub-test_ses-01"

            def path(self, port, hemi=None):
                return Path("inputs") / ((hemi + "." if hemi else "") + port)

            def volume(self, port):
                return Path("outputs") / port

            def output(self, relative):
                return Path("outputs") / relative

            def run(self, command, arguments):
                calls.append((command, arguments))

        stage = Stage()
        outputs = surf2vol(stage)
        self.assertIn(("freesurfer:mri_brainvol_stats", ["--subject", stage.subject]), calls)
        self.assertIn("brainvol_stats", outputs)

    def test_graph_ports_and_reject_missing_edge(self):
        graph = read_json(ROOT / "conf/workflow.json")
        self.assertEqual(len(validate_graph(graph)), 51)
        broken = copy.deepcopy(graph)
        broken["edges"].pop()
        with self.assertRaisesRegex(ValueError, "Unbound"):
            validate_graph(broken)

    def test_graph_qc_rules_exist_in_bundled_catalog(self):
        graph = read_json(ROOT / "conf/workflow.json")
        rules = ROOT / "vendor/neuroqc/references"
        index = read_json(rules / "qc-rules.json")["rule_files"]
        for node in graph["nodes"]:
            policy = node.get("policy", {})
            declared = set(policy.get("rule_ids", []))
            for values in policy.get("stage_rules", {}).values():
                declared.update(values)
            for identity in declared:
                with self.subTest(node=node["node_id"], rule=identity):
                    self.assertEqual(read_json(rules / index[identity])["id"], identity)

    def test_empty_source_allowed_but_empty_weights_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "__init__.py").touch()
            self.assertEqual(len(resource_files(root / "*.py")), 1)
            (root / "weights.model").touch()
            with self.assertRaisesRegex(ValueError, "empty"):
                resource_files(root / "*.model")

    def test_missing_script_interpreter_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "script"
            path.write_text("#!/nonexistent-fastprep-interpreter\n")
            path.chmod(0o755)
            with self.assertRaisesRegex(ValueError, "Missing script interpreter"):
                executable(path)

    def test_artifact_cannot_escape_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").write_text("data")
            result = root / "result"
            result.mkdir()
            with self.assertRaisesRegex(ValueError, "escapes"):
                artifact_path(result, {"path": "../data"})

    def test_assembly_keeps_collection_paths_and_empty_command_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage"
            members = []
            for hemi in ("lh", "rh"):
                path = f"subjects/sub-01/surf/{hemi}.white"
                (stage / path).parent.mkdir(parents=True, exist_ok=True)
                (stage / path).write_text(hemi)
                members.append({"path": path, "hemi": hemi})
            write_json(stage / "collections/white.json", {"members": members})
            (stage / "logs").mkdir()
            (stage / "logs/stdout").write_text("done")
            (stage / "logs/stderr").touch()
            write_json(stage / "manifest.json", {"schema": "fastprep.stage/1", "subject": "sub-01",
                "node_id": "white", "outputs": {"white": {"path": "collections/white.json",
                "members": members, "contract": {"space": "fsnative"}}}, "metrics": {},
                "commands": [{"stdout": "logs/stdout", "stderr": "logs/stderr"}]})
            graph = {"version": "1.0.0", "nodes": [{"node_id": "white"}],
                "output_contracts": [{"name": "white", "source": "white", "source_port": "white"}]}
            assemble({"subject": "sub-01", "upstream": [str(stage)]}, graph, {}, output=root / "out")
            destination = root / "out/sub-01"
            outputs = read_json(destination / "outputs.json")
            self.assertEqual(outputs["scientific_approval"], "not_performed")
            self.assertTrue((destination / "provenance/white/logs/stderr").is_file())
            for record in outputs["outputs"]["white"]["files"]:
                self.assertTrue((destination / record["path"]).is_file())
            with self.assertRaisesRegex(ValueError, "new empty"):
                assemble({"subject": "sub-01", "upstream": [str(stage)]}, graph, {}, output=root / "out")
            with self.assertRaisesRegex(ValueError, "Native and topology"):
                assemble({"subject": "sub-01", "upstream": [str(stage)]}, graph, {}, stub=True, output=root / "stub")

    def test_final_t2_pial_drives_morphometry(self):
        try:
            import numpy as np
            from nibabel.freesurfer.io import write_geometry
        except ImportError:
            self.skipTest("NumPy and NiBabel required")
        self.enterContext(patch("reference_runtime.surface_repair_checks", return_value={
            "euler_number": 2, "total_defect_index": 0, "self_intersecting_vertices": 0}))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]], dtype=float)
            faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
            metadata = dict(head=[2, 0, 20], valid="1", filename="synthetic.mgz", volume=[8, 8, 8],
                            voxelsize=[1, 1, 1], xras=[1, 0, 0], yras=[0, 1, 0], zras=[0, 0, 1], cras=[0, 0, 0])
            calls = []

            class Stage:
                node = next(n for n in read_json(ROOT / "conf/workflow.json")["nodes"] if n["node_id"] == "anat_pial_surface")
                spec = {"threads": 1, "t1_placement_backend": "cpu", "t2_pial_backend": "cpu"}
                metrics = {}

                def each_hemisphere(self, operation, **options):
                    return {hemi: operation(self, hemi) for hemi in ("lh", "rh")}

                def path(self, port, hemi=None):
                    return self.hemi(port, hemi)

                def hemi(self, port, hemi):
                    return root / (str(hemi) + "." + port)

                def run(self, command, args):
                    calls.append((command, args))
                    if "--o" in args:
                        target = args[args.index("--o") + 1]
                        write_geometry(target, vertices * (2 if "--mmvol" in args else 1.5), faces, volume_info=metadata)

            stage = Stage()
            stage.root = root
            for hemi in ("lh", "rh"):
                write_geometry(stage.path("white_surf", hemi), vertices, faces, volume_info=metadata)
                stage.path("autodet_gwstats", hemi).write_text(
                    "pial_border_low 60\npial_border_hi 80\npial_outside_hi 75\n"
                    "pial_outside_low 10\npial_inside_hi 110\n")
            stage.node = copy.deepcopy(Stage.node)
            stage.node["parameters"]["t2_initial_t1_threshold_scale"] = 0.85
            stage.node["parameters"].pop("t2_refinement_overrides", None)
            with patch("t2_pial.prepare_t2", return_value=(root / "T2.mgz", {"mode": "T2_refined"}, [])), \
                 patch("t2_pial.refinement_manifest", return_value=root / "pial.json"):
                outputs = place_surface(stage)
            self.assertEqual(len([args for _, args in calls if "--mmvol" in args]), 2)
            for _, args in calls:
                if "--pial" in args:
                    self.assertEqual("--pial_border_low" in args, "--mmvol" not in args)
                    if "--mmvol" not in args:
                        self.assertEqual(args[args.index("--pial_border_low") + 1], "51")
            for hemi in ("lh", "rh"):
                final = outputs["pial_surf"][hemi]
                for flag in ("--curv-map", "--area-map", "--thickness"):
                    self.assertTrue(any(args[0] == flag and final in args for _, args in calls))
                self.assertTrue(stage.metrics["pial_refinement"]["initial_t1_pial"][hemi]["applied"])
            baseline_commands = [args[:-6] for _, args in calls if "--pial" in args and "--mmvol" not in args]
            refined_commands = [args for _, args in calls if "--mmvol" in args]
            for baseline, refined in zip(baseline_commands, refined_commands):
                expected = list(baseline)
                expected[expected.index("--i") + 1] = baseline[baseline.index("--o") + 1]
                expected[expected.index("--o") + 1] = refined[refined.index("--o") + 1]
                self.assertEqual(refined, [*expected, "--mmvol", root / "T2.mgz", "T2"])
            calls.clear()
            stage.node = copy.deepcopy(Stage.node)
            with patch("t2_pial.prepare_t2", return_value=(root / "T2.mgz", {"mode": "T2_refined"}, [])), \
                 patch("t2_pial.refinement_manifest", return_value=root / "pial.json"):
                place_surface(stage)
            for _, arguments in calls:
                if "--pial" not in arguments:
                    continue
                if "--mmvol" in arguments:
                    self.assertEqual(arguments[-9:], ["--mmvol", root / "T2.mgz", "T2",
                                                     "--location", "0.6", "--curv", "0.8", "--nspring", "0.6"])
                    self.assertNotIn("--pial_border_low", arguments)
                    self.assertNotIn("--mm_min_inside", arguments)
                else:
                    self.assertNotIn("--mm_min_inside", arguments)
                    self.assertNotIn("--location", arguments)
                    self.assertNotIn("--curv", arguments)
                    self.assertNotIn("--nspring", arguments)
                    self.assertNotIn("--pial_border_low", arguments)
            recorded = stage.metrics["pial_refinement"]["t2_refinement_parameters"]
            self.assertEqual(recorded["requested"], {"location": 0.6, "curv": 0.8, "nspring": 0.6})
            self.assertTrue(recorded["applied"])
            calls.clear()
            for mode in ("auto", "off"):
                with self.subTest(t2_mode=mode):
                    stage.node = copy.deepcopy(Stage.node)
                    stage.node["parameters"]["t2_mode"] = mode
                    for hemi in ("lh", "rh"):
                        stage.path("autodet_gwstats", hemi).unlink(missing_ok=True)
                    with patch("t2_pial.prepare_t2", return_value=(None, {"mode": "T1_only"}, [])), \
                         patch("t2_pial.refinement_manifest", return_value=root / "pial.json"):
                        place_surface(stage)
                    initial_commands = [args for _, args in calls if "--pial" in args]
                    self.assertEqual(initial_commands, baseline_commands)
                    self.assertFalse(stage.metrics["pial_refinement"]["t2_refinement_parameters"]["applied"])
                    for hemi in ("lh", "rh"):
                        self.assertEqual(stage.hemi("pial_t1_surf", hemi).read_bytes(),
                                         stage.hemi("pial_surf", hemi).read_bytes())
                        self.assertFalse(stage.metrics["pial_refinement"]["initial_t1_pial"][hemi]["applied"])
                    calls.clear()


if __name__ == "__main__":
    unittest.main()

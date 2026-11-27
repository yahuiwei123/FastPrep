import json
import os
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import qc_runtime


class StandaloneQCTests(unittest.TestCase):
    def test_manifest_with_members_resolves_to_primary_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qc_runtime.write_json(root / "alignment_manifest.json", {"registrations": []})
            (root / "transform.mat").write_text("synthetic transform fixture")
            qc_runtime.write_json(root / "manifest.json", {
                "schema": "fastprep.stage/1", "subject": "sub-01", "node_id": "alignment",
                "outputs": {"alignment_manifest": {"path": "alignment_manifest.json",
                    "contract": {"data_type": "manifest"},
                    "members": [{"path": "alignment_manifest.json"}, {"path": "transform.mat"}]}},
                "metrics": {}, "commands": []})
            catalog = qc_runtime.staged_catalog({"subject": "sub-01", "upstream": [str(root)]}, {})
            self.assertEqual(qc_runtime.first(catalog, "alignment", "alignment_manifest"), root / "alignment_manifest.json")

    def test_stage_catalog_accepts_portable_manifest_without_hash_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "subjects/sub-01/mri/input.txt"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("input\n")
            qc_runtime.write_json(root / "manifest.json", {
                "schema": "fastprep.stage/1", "subject": "sub-01", "node_id": "preproc",
                "outputs": {"image": {"path": "subjects/sub-01/mri/input.txt",
                    "contract": {"data_type": "text"}, "members": []}},
                "metrics": {}, "commands": []})
            catalog = qc_runtime.staged_catalog({"analysis_unit_id": "sub-01", "upstream": [str(root)]}, {})
            self.assertEqual(qc_runtime.first(catalog, "preproc", "image").read_text(), "input\n")
            self.assertNotIn("sha256", json.dumps(catalog).lower())

    def test_structural_input_pass_never_means_visual_approval(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "t1.nii.gz"
            image.write_bytes(b"nifti placeholder")
            item = {"analysis_unit_id": "sub-01_ses-01", "subject_id": "sub-01", "session_id": "01",
                    "path": str(image), "series_id": "sub-01_ses-01_T1w", "modality": "T1w",
                    "structural_status": "pass", "visual_review": "pending",
                    "probe": {"facts": {"shape": [2, 2, 2], "finite": True, "nonzero": True,
                        "affine": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                        "voxel_size_mm": [1, 1, 1], "spatial_units": "mm"}}}
            plan = {"analysis_unit_id": "sub-01_ses-01", "subject_id": "sub-01", "session_id": "01",
                    "reference": item, "companions": []}
            self.assertIs(qc_runtime._session_plan({"session_input_plan": [plan]}, "sub-01_ses-01"), plan)
            item["visual_review"] = "pass"
            with self.assertRaises(ValueError):
                qc_runtime._session_plan({"session_input_plan": [plan]}, "sub-01_ses-01")

    def test_report_only_forwards_provisionally_and_strict_blocks_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                root = Path("upstream")
                artifact = root / "subjects/sub-01/mri/result.txt"
                artifact.parent.mkdir(parents=True)
                artifact.write_text("preserve me\n")
                qc_runtime.write_json(root / "manifest.json", {
                    "schema": "fastprep.stage/1", "subject": "sub-01", "node_id": "compute",
                    "outputs": {"native": {"path": artifact.relative_to(root).as_posix(),
                                               "contract": {"data_type": "text"}}},
                    "metrics": {}, "commands": []})
                blueprint = {"nodes": [{"node_id": "native_qc", "kind": "qc", "checks_node": "compute",
                    "outputs": [{"name": "native", "contract": {"data_type": "text"}}],
                    "policy": {"status": "approved", "rule_ids": ["REPORT.TRACEABILITY"],
                        "output_sources": {"native": {"node_id": "compute", "port": "native"}}}}]}
                spec = {"analysis_unit_id": "sub-01", "node_id": "native_qc", "upstream": ["upstream"]}
                report = qc_runtime.run(spec, {}, blueprint, "report_only")
                self.assertEqual(report["overall_status"], "unknown")
                approval = qc_runtime.read_json("qc_result/approval.json")
                self.assertFalse(approval["approved"])
                self.assertEqual(Path("result/subjects/sub-01/mri/result.txt").read_text(), "preserve me\n")
                manifest = qc_runtime.read_json("result/manifest.json")
                self.assertEqual(manifest["schema"], "fastprep.stage/1")
                self.assertEqual(manifest["outputs"]["native"]["path"],
                                 "subjects/sub-01/mri/result.txt")
                self.assertNotIn("members", manifest["outputs"]["native"])
                self.assertNotIn("sha256", json.dumps(manifest).lower())

                Path("strict-run").mkdir()
                spec["upstream"] = [str(root.absolute())]
                os.chdir("strict-run")
                report = qc_runtime.run(spec, {}, blueprint, "strict")
                self.assertEqual(report["overall_status"], "unknown")
                self.assertFalse(Path("result/manifest.json").exists())
                self.assertTrue(Path("qc_result/blocked.json").is_file())
            finally:
                os.chdir(previous)

    def test_forwarding_preserves_collection_record_and_copies_qc_logs(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                root = Path("upstream")
                member = root / "surfaces/lh.white"
                main_record = root / "collections/white.json"
                member.parent.mkdir(parents=True)
                main_record.parent.mkdir(parents=True)
                member.write_text("mesh\n")
                qc_runtime.write_json(main_record, {"members": [{"path": "surfaces/lh.white", "hemi": "lh"}]})
                qc_runtime.write_json(root / "manifest.json", {
                    "schema": "fastprep.stage/1", "subject": "sub-01", "node_id": "white",
                    "outputs": {"white_surf": {"path": "collections/white.json",
                        "contract": {"data_type": "surface", "space": "fsnative"},
                        "members": [{"path": "surfaces/lh.white", "hemi": "lh"}]}},
                    "metrics": {}, "commands": []})
                catalog = qc_runtime.staged_catalog(
                    {"analysis_unit_id": "sub-01", "upstream": ["upstream"]}, {})
                log_root = Path("qc_result/native")
                (log_root / "logs").mkdir(parents=True)
                (log_root / "logs/0.stdout").write_text("stdout\n")
                (log_root / "logs/0.stderr").write_text("stderr\n")
                node = {"node_id": "native_qc", "outputs": [{"name": "white_surf",
                    "contract": {"data_type": "surface", "space": "declared"}}],
                    "policy": {"status": "approved", "output_sources": {
                        "white_surf": {"node_id": "white", "port": "white_surf"}}}}
                report = {"overall_status": "unknown", "report_id": "sub-01:native_qc",
                    "provenance": {"qc_commands": [{"stdout": "logs/0.stdout", "stderr": "logs/0.stderr"}]}}
                qc_runtime.forward_outputs({}, {}, {}, node, catalog, report, "report_only")
                manifest = qc_runtime.read_json("result/manifest.json")
                output = manifest["outputs"]["white_surf"]
                self.assertEqual(output["path"], "collections/white.json")
                self.assertEqual(output["contract"], {"data_type": "surface", "space": "declared"})
                self.assertEqual(output["members"][0]["path"], "surfaces/lh.white")
                self.assertEqual(Path("result/collections/white.json").read_text(),
                                 main_record.read_text())
                command = manifest["commands"][0]
                self.assertEqual(Path("result", command["stdout"]).read_text(), "stdout\n")
                self.assertEqual(Path("result", command["stderr"]).read_text(), "stderr\n")
            finally:
                os.chdir(previous)

    def test_collector_html_links_resolve_from_index_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            previous = Path.cwd()
            os.chdir(temporary)
            try:
                report_dir = Path("qc_result/node")
                (report_dir / "logs").mkdir(parents=True)
                (report_dir / "view.png").write_bytes(b"png")
                (report_dir / "view.png.json").write_text("{}\n")
                (report_dir / "logs/0.stdout").write_text("command log\n")
                qc_runtime.write_json(report_dir / "report.json", {
                    "report_id": "sub-01:node", "overall_status": "unknown",
                    "checks": [{"visual_evidence": [{"source": "view.png"}]}]})
                nested = report_dir / "stages/stage"
                nested.mkdir(parents=True)
                qc_runtime.write_json(nested / "report.json", {
                    "report_id": "sub-01:node", "overall_status": "unknown", "checks": []})
                duplicate = report_dir / "report/reports/0000"
                duplicate.mkdir(parents=True)
                qc_runtime.write_json(duplicate / "report.json", {
                    "report_id": "sub-01:node", "overall_status": "unknown", "checks": []})
                qc_runtime._write_index([Path("qc_result")], Path("qc_index"))
                page = Path("qc_index/index.html").read_text()
                self.assertIn('href="sub-01/node/view.png"', page)
                summary = qc_runtime.read_json("qc_index/summary.json")
                self.assertEqual(summary["report_count"], 1)
                self.assertEqual(summary["reports"][0]["path"], "sub-01/node/report.json")
                for relative in ("report.json", "view.png", "view.png.json", "logs/0.stdout"):
                    self.assertTrue(Path("qc_index/sub-01/node", relative).is_file())
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

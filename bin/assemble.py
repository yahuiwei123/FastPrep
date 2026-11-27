"""Collect all declared base outputs, without granting scientific approval."""

import argparse
from pathlib import Path

from reference_runtime import artifact_path, copy_file, read_json, write_json


def assemble(spec, blueprint, runtime, stub=False, output="published"):
    subject = spec["subject"]
    destination = Path(output) / subject
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Assembly requires a new empty subject output directory: " + str(destination))
    stages = {}
    for directory in spec["upstream"]:
        directory = Path(directory)
        manifest = read_json(directory / "manifest.json")
        if manifest["subject"] != subject or manifest["node_id"] in stages:
            raise ValueError("Mismatched or duplicated stage result")
        required_schema = "fastprep.stub/1" if stub else "fastprep.stage/1"
        if manifest["schema"] != required_schema:
            raise ValueError("Native and topology artifacts must never be mixed")
        stages[manifest["node_id"]] = (directory, manifest)
    expected = {node["node_id"] for node in blueprint["nodes"]}
    if set(stages) != expected:
        raise ValueError("Missing or extra nodes at final assembly: " + str(expected.symmetric_difference(stages)))
    if stub:
        write_json(destination / "topology.json", {"mode": "topology_only", "subject": subject,
                   "nodes": sorted(stages), "node_count": len(stages), "scientific_outputs": [],
                   "scientific_approval": "not_performed"})
        return
    produced = {}
    prefix = "subjects/" + subject + "/"
    for node in blueprint["nodes"]:
        directory, manifest = stages[node["node_id"]]
        links = {}
        for record in manifest["outputs"].values():
            source = artifact_path(directory, record)
            if "members" in record:
                target = copy_file(source, destination / "provenance" / node["node_id"] / record["path"])
                links[record["path"]] = str(target.relative_to(destination))
            for item in record.get("members", [record]):
                source = artifact_path(directory, item)
                relative = item["path"].removeprefix(prefix)
                if not item["path"].startswith(prefix):
                    relative = "provenance/" + node["node_id"] + "/" + relative
                target = copy_file(source, destination / relative)
                produced[(node["node_id"], item["path"])] = {
                    **item, "path": str(target.relative_to(destination)), "size_bytes": target.stat().st_size}
                links[item["path"]] = str(target.relative_to(destination))
        provenance = destination / "provenance" / node["node_id"]
        copy_file(directory / "manifest.json", provenance / "manifest.json")
        write_json(provenance / "published_paths.json", links)
        for entry in manifest["commands"]:
            for stream in ("stdout", "stderr"):
                source = directory / entry[stream]
                target = provenance / entry[stream]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    outputs = {}
    for contract in blueprint["output_contracts"]:
        _, manifest = stages[contract["source"]]
        record = manifest["outputs"][contract["source_port"]]
        members = record.get("members", [record])
        outputs[contract["name"]] = {"source": contract["source"], "contract": record["contract"],
            "files": [produced[(contract["source"], item["path"])] for item in members]}
    qc_stages = {identity: manifest["metrics"] for identity, (_, manifest) in stages.items()
                 if manifest.get("metrics", {}).get("qc_execution_mode")}
    write_json(destination / "outputs.json", {"schema": "fastprep.outputs/1", "subject": subject,
               "workflow_version": blueprint["version"], "outputs": outputs,
               "qc_execution_mode": "report_only" if any(record["qc_execution_mode"] == "report_only"
                                                         for record in qc_stages.values()) else "strict",
               "qc_stages": qc_stages,
               "scientific_approval": "not_performed", "missing_review": ["visual_qc", "qc_policy_review"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()
    assemble(read_json(args.spec), read_json(args.blueprint), read_json(args.runtime), args.stub)


if __name__ == "__main__":
    main()

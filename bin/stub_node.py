"""Exercise the DAG without creating any scientific images or approvals."""

import argparse
from pathlib import Path
from reference_runtime import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    arguments = parser.parse_args()
    spec = read_json(arguments.spec)
    sources = []
    for directory in spec.get("upstream", []):
        manifest = read_json(Path(directory) / "manifest.json")
        if manifest["schema"] != "fastprep.stub/1" or manifest["subject"] != spec["subject"]:
            raise ValueError("Topology-only upstream identity mismatch")
        sources.append(manifest["node_id"])
    write_json("result/manifest.json", {"schema": "fastprep.stub/1", "subject": spec["subject"],
               "node_id": spec["node_id"], "outputs": {}, "upstream_node_ids": sources,
               "scientific_approval": "not_performed"})


if __name__ == "__main__":
    main()

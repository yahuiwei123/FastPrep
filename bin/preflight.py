"""Check workflow structure and selected sessions before Nextflow execution."""

import argparse
from graphlib import TopologicalSorter
from pathlib import Path

from reference_runtime import OPERATIONS, checked_file, copy_file, read_json, write_json
from session_inputs import session_plan


def validate_graph(graph):
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    if len(nodes) != len(graph["nodes"]):
        raise ValueError("Duplicate workflow node IDs")
    if {identity for identity, node in nodes.items() if node["kind"] != "qc"} != set(OPERATIONS):
        raise ValueError("Workflow scientific stages differ from the implementation")
    dependencies = {identity: set() for identity in nodes}
    incoming = {identity: set() for identity in nodes}
    for edge in graph["edges"]:
        target = edge["target"]
        if target not in nodes or edge["target_port"] not in {port["name"] for port in nodes[target]["inputs"]}:
            raise ValueError("Invalid target port: " + str(edge))
        if edge["target_port"] in incoming[target]:
            raise ValueError("Multiple sources for input port: " + str(edge))
        incoming[target].add(edge["target_port"])
        if edge["source"] != "$input":
            source = edge["source"]
            if source not in nodes or edge["source_port"] not in {port["name"] for port in nodes[source]["outputs"]}:
                raise ValueError("Invalid source port: " + str(edge))
            dependencies[target].add(source)
    for identity, node in nodes.items():
        if incoming[identity] != {port["name"] for port in node["inputs"]}:
            raise ValueError("Unbound input ports: " + identity)
    list(TopologicalSorter(dependencies).static_order())
    for output in graph["output_contracts"]:
        if output["source_port"] not in {port["name"] for port in nodes[output["source"]]["outputs"]}:
            raise ValueError("Invalid published output: " + output["name"])
    return nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--blueprint", required=True)
    parser.add_argument("--stub", action="store_true")
    arguments = parser.parse_args()
    runtime, graph = read_json(arguments.runtime), read_json(arguments.blueprint)
    validate_graph(graph)
    if arguments.stub:
        subjects = runtime["topology_subjects"]
    else:
        if runtime.get("schema") != "fastprep.runtime/1":
            raise ValueError("Use prepare.py to create a native runtime configuration")
        subjects = [plan["analysis_unit_id"] for plan in runtime["session_input_plan"]]
        if not subjects or len(subjects) != len(set(subjects)):
            raise ValueError("Expected unique, nonempty analysis units")
        for subject in subjects:
            plan = session_plan(runtime, subject)
            for item in [plan["reference"], *plan["companions"]]:
                checked_file(item["path"])
        for command in runtime["commands"].values():
            checked_file(command["argv_prefix"][0])
            for path in command.get("files", []):
                checked_file(path, allow_empty=Path(path).suffix == ".py")
    copy_file(arguments.runtime, "checked/runtime.json")
    copy_file(arguments.blueprint, "checked/workflow.json")
    write_json("checked/selection.json", {"subjects": subjects})
    Path("checked/subjects.txt").write_text("".join(subject + "\n" for subject in subjects), encoding="utf-8")


if __name__ == "__main__":
    main()

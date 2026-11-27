"""Execute one scientific node from the standalone workflow."""

import argparse

from reference_runtime import NativeStage, OPERATIONS, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--blueprint", required=True)
    arguments = parser.parse_args()
    stage = NativeStage(read_json(arguments.spec), read_json(arguments.runtime), read_json(arguments.blueprint))
    stage.stage_inputs()
    stage.finish(OPERATIONS[stage.node["node_id"]](stage))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create a portable summary and HTML index for FastPrep QC reports."""

import argparse
from pathlib import Path

from qc_runtime import _write_index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", nargs="+", required=True)
    parser.add_argument("--output", default="qc_index")
    args = parser.parse_args(argv)
    _write_index(args.directories, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

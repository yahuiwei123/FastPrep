"""Create a portable source-and-model archive without local runtime material."""

import argparse
import os
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = {"bin", "conf", "docs", "envs", "model", "modules", "vendor", "tests", "report_only", "cluster15", "build-tools"}
TOP_FILES = {"README.md", "AGENTS.md", "THIRD_PARTY.md", ".gitignore", ".gitattributes", "batch_fastprep_anat.py", "main.nf", "nextflow.config"}
EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".nextflow", ".venv"}


def included_files(root):
    candidates = [root / name for name in sorted(TOP_FILES) if (root / name).is_file()]
    for name in sorted(DIRECTORIES):
        for directory, children, names in os.walk(root / name):
            children[:] = sorted(child for child in children if child not in EXCLUDED_PARTS)
            if name == "cluster15":
                children[:] = [child for child in children if child != "node-debs"]
            if name == "envs":
                children[:] = []
            candidates.extend(Path(directory) / filename for filename in sorted(names))
    for path in candidates:
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts[0] not in DIRECTORIES and str(relative) not in TOP_FILES:
            continue
        if relative.parts[0] == "envs" and (len(relative.parts) != 2 or path.suffix not in {".yml", ".txt"}):
            continue
        if path.name in {"software.local.json", "license.txt", ".license", ".env"} or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink() and not (path.is_file() and path.resolve().is_relative_to(root.resolve())):
            raise ValueError("External or broken link in source bundle: " + str(relative))
        if path.is_file():
            yield path


def package(root, output):
    root, output = Path(root).resolve(), Path(output).absolute()
    files = list(included_files(root))
    required = {"main.nf", "nextflow.config", "README.md", "conf/models.json", "conf/workflow.json"}
    if not required.issubset({path.relative_to(root).as_posix() for path in files}):
        raise ValueError("The standalone project is incomplete")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz", compresslevel=1) as archive:
            for path in files:
                info = archive.gettarinfo(str(path), arcname="fastprep-anat/" + path.relative_to(root).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with path.open("rb") as source:
                    archive.addfile(info, source)
    print(str(output))
    print(f"{len(files)} files; {output.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(ROOT, args.output)


if __name__ == "__main__":
    main()

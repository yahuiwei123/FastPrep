"""Build the small standalone BFC toolkit without models, MRI data or runtime paths."""

import argparse
from pathlib import Path
import tarfile

from tissue_bfc import __version__


ROOT = Path(__file__).resolve().parents[1]


def toolkit_files(root):
    files = [(path, path.relative_to(root).as_posix()) for path in sorted((root / "bin/tissue_bfc").glob("*.py"))]
    for relative in ("bin/run_tissue_bfc.py", "bin/bfc_support.py", "envs/bfc.yml", "envs/bfc-requirements.txt",
                     "envs/helper-requirements.txt", "tests/test_bfc_support.py",
                     "vendor/tissue_guided_bfc/bfc_gauss_mixture.py", "vendor/tissue_guided_bfc/bfc_rbf_scatter.py",
                     "vendor/tissue_guided_bfc/UPSTREAM_README.md", "vendor/tissue_guided_bfc/SOURCE.md"):
        files.append((root / relative, relative))
    files.append((root / "docs/BIAS_CORRECTION.md", "README.md"))
    for source, _ in files:
        if not source.is_file() or source.is_symlink():
            raise ValueError("Missing or nonportable toolkit source: " + str(source))
    return files


def package(output, root=ROOT):
    output = Path(output)
    files = toolkit_files(Path(root))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for source, relative in files:
                info = archive.gettarinfo(str(source), arcname=f"fastprep-bfc-{__version__}/" + relative)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with source.open("rb") as content:
                    archive.addfile(info, content)
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    count = package(arguments.output)
    print(f"Created {arguments.output}: {count} files; refuses overwriting existing archives")


if __name__ == "__main__":
    main()

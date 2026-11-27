# Source this file before an 8.1 cluster run.
fastprep_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -f "$fastprep_root/.site/environment.sh" ]]; then
    source "$fastprep_root/.site/environment.sh"
fi
bash "$fastprep_root/cluster15/ensure-node.sh" || return 1
unset BASH_ENV
if [[ -z ${FREESURFER_HOME:-} ]]; then
    FREESURFER_HOME=$(python3 - "$fastprep_root" <<'PYTHON'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
config = Path(os.environ.get("FASTPREP_SOFTWARE", root / "software.local.json"))
print(root / os.path.expandvars(os.path.expanduser(json.loads(config.read_text())["freesurfer_home"])))
PYTHON
) || return 1
fi
export FREESURFER_HOME FREESURFER="$FREESURFER_HOME"
export FSFAST_HOME="$FREESURFER_HOME/fsfast" MNI_DIR="$FREESURFER_HOME/mni" FSF_OUTPUT_FORMAT=nii.gz
export PATH="$FREESURFER_HOME/bin:$FSFAST_HOME/bin:$PATH"
export LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | awk '!/python3\.12\/dist-packages\/torch/{print}' | paste -sd :)

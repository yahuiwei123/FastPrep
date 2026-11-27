#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
run_dir=$(realpath -m "${2:?absolute shared run directory}")
if [[ -f "$here/../.site/environment.sh" ]]; then source "$here/../.site/environment.sh"; fi
root=$(realpath -m "${FASTPREP_BIDS_ROOT:-smri}")
cd "$here"
export FASTPREP_BIDS_ROOT="$root"
: "${1:?relative dataset}" "${2:?absolute shared run directory}"
python3 -B - "$1" <<'PY'
import sys
import os
from pathlib import Path
from cluster import batch
dataset = sys.argv[1]
reason = batch.dataset_exclusion(dataset) or batch.dataset_exclusion((Path(os.environ.get('FASTPREP_BIDS_ROOT', 'smri')) / dataset).resolve())
if reason:
    sys.exit(f'SKIP {dataset}: {reason}')
PY
: "${FASTPREP_NODES:?set the three Slurm node names}" "${FASTPREP_PARTITION:?set Slurm partition}"
for cmd in sbatch squeue scancel sinfo scontrol; do command -v "$cmd" >/dev/null || { echo "missing $cmd" >&2; exit 1; }; done
# --nodelist would require every named host for each task. Exclude non-target hosts instead.
hosts=$(sinfo -hN -p "$FASTPREP_PARTITION" -o '%N' | sort -u)
wanted=$(scontrol show hostnames "$FASTPREP_NODES" | sort -u)
[[ $(wc -l <<<"$wanted") == 3 && -n "$hosts" ]] || { echo "Expected three target nodes" >&2; exit 2; }
while read -r n; do grep -Fxq "$n" <<<"$hosts" || { echo "$n not in partition" >&2; exit 2; }; done <<<"$wanted"
others=$(comm -23 <(printf '%s\n' "$hosts") <(printf '%s\n' "$wanted") | paste -sd ,)
export FASTPREP_NODE_FILTER=${others:+--exclude=$others}
source ./env81.sh
export NXF_ANSI_LOG=false
if [[ ! -f "$run_dir/plan.json" ]]; then python3 cluster.py plan --dataset "$1" --batch-size 1000000 --run-dir "$run_dir"; fi
exec python3 - "$1" "$run_dir" <<'PY'
from pathlib import Path
import sys,os,json
p=Path.cwd().parent;r=Path(sys.argv[2]).resolve()
jobs=json.loads((r/'plan.json').read_text())
if len(jobs)!=1 or jobs[0][0]!=sys.argv[1]:sys.exit('Expected one exact BIDS dataset; use a separate run directory per dataset')
ds,part,units=jobs[0]
os.execv(sys.executable,[sys.executable,str(p/'batch_fastprep_anat.py'),'worker','--root',os.environ['FASTPREP_BIDS_ROOT'],'--dataset',ds,'--part',str(part),'--units',','.join(units),'--isolate-session-errors','--software',os.environ.get('FASTPREP_SOFTWARE',str(p/'software.local.json')),'--workflow',str(p),'--nextflow',str(p/'cluster15/nextflow-slurm'),'--surface-backend','gpu','--work-root',str(r/'work'),'--out-root',str(r/'results')])
PY

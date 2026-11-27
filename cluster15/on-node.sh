#!/bin/bash
set -e
here=$(cd "$(dirname "$0")" && pwd)
source "$here/env81.sh"
export NXF_ANSI_LOG=false
exec python3 "$here/cluster.py" run "$@"

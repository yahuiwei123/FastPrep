#!/bin/bash
# FreeSurfer invokes /bin/tcsh directly, including in subprocesses.
set -euo pipefail
check_tcsh() { [[ -x /bin/tcsh ]] && /bin/tcsh -f -c 'exit 0'; }
check_bc() { [[ $(echo '2+2' | bc 2>/dev/null) == 4 ]]; }
packages=()
check_tcsh || packages+=(tcsh)
check_bc || packages+=(bc)
(( ${#packages[@]} )) || exit 0
printf '%s: installing missing FastPrep dependencies: %s\n' "$(hostname)" "${packages[*]}" >&2
command -v apt-get >/dev/null || { echo 'Install tcsh and bc with the node package manager before launching FastPrep.' >&2; exit 1; }
asroot=()
if (( EUID != 0 )); then
  command -v sudo >/dev/null && sudo -n true || { echo 'Node admin: install tcsh and bc, then restart this rank.' >&2; exit 1; }
  asroot=(sudo -n)
fi
# Use the local signed APT index to authenticate any shared offline package.
# Stage privately so the verified bytes are the bytes supplied to apt.
cache=${FASTPREP_NODE_DEBS:-"$(dirname "$0")/node-debs"}
tmp=$(mktemp -d)
trap 'rm -rf -- "$tmp"' EXIT
args=()
for pkg in "${packages[@]}"; do
  selected=$pkg
  for deb in "$cache"/"${pkg}"_*.deb; do
    [[ -f $deb ]] || continue
    cp "$deb" "$tmp/$pkg.deb"
    ver=$(dpkg-deb -f "$tmp/$pkg.deb" Version)
    arch=$(dpkg-deb -f "$tmp/$pkg.deb" Architecture)
    [[ $arch == "$(dpkg --print-architecture)" ]] || continue
    digest=$(sha256sum "$tmp/$pkg.deb"); digest=${digest%% *}
    if apt-cache show "$pkg=$ver" 2>/dev/null | awk '/^SHA256:/{print $2}' | grep -Fx "$digest" >/dev/null; then
      selected="$tmp/$pkg.deb"; break
    fi
  done
  args+=("$selected")
done
"${asroot[@]}" apt-get -o DPkg::Lock::Timeout=120 -y --reinstall --no-install-recommends install "${args[@]}"
check_tcsh && check_bc || { echo 'Node dependency check failed after installation' >&2; exit 1; }

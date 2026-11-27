"""Generate native code for each ordinary architecture advertised by nvcc, plus PTX."""
import re
import subprocess


def targets(nvcc, requested=None):
    output = subprocess.check_output([nvcc, '--list-gpu-code'], text=True)
    available = sorted({int(m[1]) for m in re.finditer(r'^sm_(\d+)$', output, re.M)})
    if not available or (requested is not None and requested not in available):
        raise ValueError(f'Unsupported CUDA target; {nvcc} offers {available}')
    return available if requested is None else [requested]


def flags(archs):
    return [f'-gencode=arch=compute_{a},code=sm_{a}' for a in archs] + [
        f'-gencode=arch=compute_{min(archs)},code=compute_{min(archs)}']

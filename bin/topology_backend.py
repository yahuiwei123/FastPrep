"""Run only the quick-sphere geometry on CUDA; preserve the stock serial trajectory and GA."""
import os
from pathlib import Path
import re
import tempfile
import time

from t2_pial_backend import device_mask, surface_libdir


STOCK = 'freesurfer:mris_sphere'
STAMP = 'freesurfer-linux-ubuntu20_x86_64-8.1.0-20250718-f30dfed'
BINDING = 'fsbindings.cpython-38-x86_64-linux-gnu.so'


def verify_mesh(source, candidate):
    import nibabel as nib
    import numpy as np

    points, faces, meta = nib.freesurfer.read_geometry(source, read_metadata=True)
    moved, triangles, header = nib.freesurfer.read_geometry(candidate, read_metadata=True)
    spatial = ('volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras')
    if (not len(points) or points.shape != moved.shape or not np.isfinite(points).all() or
            not np.isfinite(moved).all() or not len(faces) or faces.min() < 0 or faces.max() >= len(points) or
            not np.array_equal(faces, triangles) or not meta.get('filename') or
            any(key not in meta or not np.isfinite(meta[key]).all() for key in spatial) or
            meta.keys() != header.keys() or any(not np.array_equal(meta[key], header[key]) for key in meta)):
        raise ValueError('Quick-sphere mesh correspondence or volume geometry changed')
    return len(points), len(faces)


def geometry_events(log, vertices, faces):
    binds = re.findall(r'^GEOM_BIND (\d+) (\d+) (\d+)$', log, re.M)
    passes = re.findall(r'^GEOM_PASS (.*)$', log, re.M)
    if (not binds or any((int(count), int(triangles)) != (vertices, faces) or int(edges) <= 0
                         for count, triangles, edges in binds) or not passes or
            any(not re.fullmatch(r'3 0 \d+', row) for row in passes)):
        raise RuntimeError('Missing or inconsistent CUDA quick-sphere evidence')
    return {'geometry_passes': len(passes), 'mesh_binds': len(binds),
            'native_normal_retries': log.count('GEOM_DEGENERATE native-normal-retry')}


def quick_sphere(stage, hemi, arguments, inflated, qsphere):
    backend = stage.spec.get('topology_backend', 'gpu')
    start = time.monotonic()
    details = {'backend': backend, 'ga_backend': 'cpu'}
    if backend == 'cpu':
        stage.run(STOCK, [*arguments, inflated, qsphere])
    elif backend == 'gpu':
        mask = device_mask(stage.spec['device'], os.environ.get('CUDA_VISIBLE_DEVICES'))
        if mask is None:
            raise ValueError('Topology GPU requested but CUDA device masked or CPU selected')
        native = Path(stage.spec['topology_runtime']).resolve() / 'native'
        command = stage.command(STOCK)
        env = command['environment'].copy()
        home = Path(env['FREESURFER_HOME'])
        bindings = home / 'python/packages/fsbindings'
        if (not (native / 'geometry.so').is_file() or not os.access(native / 'qsphere', os.X_OK) or
                (home / 'build-stamp.txt').read_text().strip() != STAMP or not (bindings / BINDING).is_file() or
                command['argv_prefix'] != [str(home / 'bin/mris_sphere')]):
            raise RuntimeError('Missing hybrid quick-sphere or incompatible FreeSurfer installation')
        env.update(CUDA_VISIBLE_DEVICES=mask, GEOM_STEP='3', GEOM_CHECK='0',
                   LD_LIBRARY_PATH=os.pathsep.join(filter(None, (str(surface_libdir(native, mask)), str(bindings), env.get('LD_LIBRARY_PATH', os.getenv('LD_LIBRARY_PATH'))))))
        with tempfile.TemporaryDirectory(dir=qsphere.parent) as tmp:
            candidate = Path(tmp) / qsphere.name
            stage.run_process('fastprep:topology-geometry-' + hemi, [native / 'qsphere', *arguments, inflated, candidate], env, stage.subject_dir)
            vertices, faces = verify_mesh(inflated, candidate)
            details.update(geometry_events((stage.root / stage.logs[-1]['stderr']).read_text(), vertices, faces),
                           cuda_visible_device=mask, fs_stamp=STAMP)
            candidate.replace(qsphere)
    else:
        raise ValueError('Unknown topology backend: ' + str(backend))
    return dict(details, seconds=time.monotonic() - start)

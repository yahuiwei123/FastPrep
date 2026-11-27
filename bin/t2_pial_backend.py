"""Select and validate hemisphere placement without publishing unchecked partial outputs."""
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile


BUNDLE = Path(__file__).resolve().parents[1] / 'vendor/t2_pial'
STOCK = 'freesurfer:mris_place_surface'


def device_mask(device, inherited):
    device = str(device).lower()
    if device == 'cpu':
        return None
    if device in ('auto', 'cuda'):
        index = 0
    elif re.fullmatch(r'(?:cuda:)?[0-9]+', device):
        index = int(device.split(':')[-1])
    else:
        raise ValueError('Invalid T2 placement device: ' + device)
    if inherited is None:
        return str(index)
    devices = inherited.split(',')
    if index >= len(devices) or devices[index].strip() in ('', '-1'):
        return None
    return devices[index].strip()


def surface_libdir(native, mask):
    """Use the CUDA 13 libraries for targets absent in the broader CUDA 12 build."""
    modern = native / 'cuda13'
    if not modern.is_dir():
        return native
    # Query in a child: CUDA_VISIBLE_DEVICES must take effect before cuInit.
    import subprocess
    import sys
    probe = """import ctypes as c
d = c.CDLL('libcuda.so.1')
dev, major, minor, driver = (c.c_int() for _ in range(4))
if any((d.cuInit(0), d.cuDeviceGet(c.byref(dev),0), d.cuDriverGetVersion(c.byref(driver)),
        d.cuDeviceComputeCapability(c.byref(major),c.byref(minor),dev))):
    raise SystemExit(1)
sm = major.value * 10 + minor.value
print(int(driver.value >= 13020 and (sm in (88,103,110,121) or sm > 121)))
"""
    check = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                           env={**os.environ, 'CUDA_VISIBLE_DEVICES': mask})
    return modern if check.returncode == 0 and check.stdout.strip() == '1' else native


def configuration(stage, hemi, bundle, role='t2-pial'):
    parameters = stage.node['parameters']
    if stage.spec['threads'] not in (2, 4, 8):
        return None, 'unqualified thread count'
    overrides = parameters.get('t2_refinement_overrides', {})
    if role == 't2-pial' and overrides not in ({}, {'location': 0.6, 'curv': 0.8, 'nspring': 0.6}):
        return None, 'unqualified T2 overrides'
    mask = device_mask(stage.spec['device'], os.environ.get('CUDA_VISIBLE_DEVICES'))
    if mask is None:
        return None, 'CPU requested or CUDA device masked'
    native = bundle / 'native'
    required = [native / 'runtime.json', native / 'classify.so', bundle / 'sync_worker.py']
    if role == 't2-pial':
        required.extend(native / name for name in ('pial-field.so','place','distance-field'))
    else:
        required.extend(bundle / 't1/native' / name for name in ('runtime.json','signed.so','place'))
    if not all(path.is_file() for path in required):
        return None, 'GPU runtime not installed'
    command = stage.command(STOCK)
    env = command['environment'].copy()
    home = Path(env['FREESURFER_HOME'])
    stamp = home / 'build-stamp.txt'
    try:
        build = json.loads((native / 'runtime.json').read_text())
        if (build['schema'] != 'fastprep.t2-pial/2' or not isinstance(build['arch'],int) or
                not isinstance(build['binding'],str) or not isinstance(build['stamp'],str)):
            return None, 'invalid shared placement runtime metadata'
        if role != 't2-pial':
            driver = json.loads((bundle / 't1/native/runtime.json').read_text())
            if driver['schema'] != 'fastprep.t1-placement/2' or any(driver[key] != build[key] for key in ('stamp','arch')) or driver.get('ptx', False) != build.get('ptx', False):
                return None, 'incompatible T1 placement runtime'
    except (OSError, ValueError, TypeError, KeyError):
        return None, 'unreadable placement runtime metadata'
    bindings = home / 'python/packages/fsbindings'
    if (not stamp.is_file() or stamp.read_text().strip() != build['stamp'] or not (bindings / build['binding']).is_file() or
            command['argv_prefix'] != [str(home / 'bin/mris_place_surface')]):
        return None, 'incompatible FreeSurfer runtime'
    python = stage.runtime['python']['bfc']
    libdir = surface_libdir(native, mask)
    t1dir = bundle / 't1/native'
    if libdir != native:
        t1dir = t1dir / 'cuda13'
    env.update(CUDA_VISIBLE_DEVICES=mask, T2GPU_PYTHON=python,
               SYNC_LIBRARY=str(libdir / 'classify.so'), SYNC_WORKER=str(bundle / 'sync_worker.py'), SYNC_ORDERED='neighbor',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if role == 't2-pial':
        env.update(T2GPU_WORKER=str(bundle / 'worker.py'), T2GPU_BATCH_FIT='1', T2GPU_DISTANCE_MEMO='1',
                   T2GPU_PARALLEL_FIELDS='1', T2GPU_LEAN_MARCH='1', T2GPU_CAPTURE_WARM_PIAL='0')
    else:
        env.update(T1_MODE='combined', T1_TRACE='0')
    old_path = env.get('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH', ''))
    env['LD_LIBRARY_PATH'] = os.pathsep.join(map(str, (libdir, t1dir, bindings))) + (os.pathsep + old_path if old_path else '')
    comparison = '>=' if build.get('ptx') is True else '=='
    probe = ('import sys, torch; sys.exit(0 if torch.cuda.is_available() and '
             f'torch.cuda.get_device_capability(0) {comparison} {divmod(build["arch"], 10)!r} else 1)')
    if stage.run_process('fastprep:' + role + '-device-' + hemi, [python, '-c', probe], env, stage.subject_dir, check=False):
        return None, 'CUDA device or compute capability unavailable'
    return env, None


def endpoints(directory, hemi, white, surface='pial'):
    import nibabel as nib
    import numpy as np

    result = {}
    for key in (surface, 'targets'):
        vertices, faces, meta = nib.freesurfer.read_geometry(directory / (hemi + '.' + key), read_metadata=True)
        geom_keys = ('volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras')
        problems = []
        if vertices.shape != white[0].shape or not len(vertices):
            problems.append('vertex count/shape differs')
        if not np.array_equal(faces, white[1]):
            problems.append('face indices differ')
        if not np.isfinite(vertices).all():
            problems.append('nonfinite coordinates')
        for name in geom_keys:
            if name not in meta or not np.isfinite(meta[name]).all():
                problems.append('missing/nonfinite volume field ' + name)
            elif len(white) == 3 and (name not in white[2] or not np.array_equal(meta[name], white[2][name])):
                problems.append('reference volume field ' + name + ' differs')
        if problems:
            raise ValueError('Invalid ' + key + ' geometry or vertex correspondence: ' + '; '.join(problems))
        result[key] = vertices, {name: meta[name] for name in geom_keys}
    rip = nib.load(directory / 'rip.mgz')
    values = np.asarray(rip.dataobj)
    if values.size != len(white[0]) or not np.isin(values, [0, 1]).all() or not np.isfinite(rip.affine).all():
        raise ValueError('Invalid placement rip flags')
    result['rip'] = values, rip.affine
    return result


def differences(cpu, gpu, surface='pial'):
    import numpy as np

    report = {}
    for key in (surface, 'targets'):
        first, second = cpu[key], gpu[key]
        delta = second[0] - first[0]
        distance = np.linalg.norm(delta, axis=1)
        report[key] = {'max_mm': float(distance.max()), 'rms_mm': float(np.sqrt(np.mean(distance ** 2))), 'p95_mm': float(np.percentile(distance, 95)),
                       'min_xyz_mm': delta.min(0).tolist(), 'max_xyz_mm': delta.max(0).tolist(),
                       'same_volume_geometry': all(np.array_equal(values, second[1][name]) for name, values in first[1].items())}
    report['rip_exact'] = all(np.array_equal(first, second) for first, second in zip(cpu['rip'], gpu['rip']))
    report['exact'] = report['rip_exact'] and all(report[key]['same_volume_geometry'] and report[key]['max_mm'] == 0 for key in (surface, 'targets'))
    return report


def preserve_failure(directory, root, role, hemi, reason):
    """Keep the last replayable movement state before temporary work is removed."""
    destination = Path(tempfile.mkdtemp(prefix=role + '-' + hemi + '-failed-', dir=root))
    moves = directory / 'sync-moves'
    if moves.is_dir():
        replay = destination / 'sync-moves'
        replay.mkdir()
        for name in ('faces.i32', 'offsets.i32', 'neighbors.i32'):
            if (moves / name).is_file():
                shutil.copy2(moves / name, replay / name)
        steps = [p for p in moves.glob('step-*') if p.is_dir() and p.name[5:].isdigit()]
        if steps:
            latest = max(steps, key=lambda p: int(p.name[5:]))
            shutil.copytree(latest, replay / latest.name)
    for path in directory.iterdir():
        if path.is_file() and (path.name.startswith(hemi + '.') or path.name in ('rip.mgz', 'timing.tsv')):
            shutil.copy2(path, destination / path.name)
    (destination / 'failure.json').write_text(json.dumps(dict(role=role, hemi=hemi, reason=reason,
        source_directory=str(directory), scientific_approval='not_performed'), indent=2) + '\n')
    return destination


def refine(stage, arguments, hemi, role='t2-pial', *, medial_wall_strategy=None):
    import nibabel as nib
    from pial_boundary import prepare, strategy_for, verify_output

    if role == 't2-pial':
        mode = stage.spec.get('t2_pial_backend', stage.node['parameters'].get('t2_pial_backend', 'gpu'))
        executable, surface = 'native/place', 'pial'
    elif role in ('t1-white','t1-pial'):
        if '--mmvol' in arguments or ('--white' if role == 't1-white' else '--pial') not in arguments:
            raise ValueError('wrong T1 placement role')
        mode = stage.spec.get('t1_placement_backend','gpu')
        executable, surface = 't1/native/place', role[3:]
    else:
        raise ValueError('unknown placement role: ' + role)
    if mode not in ('checked', 'cpu', 'gpu'):
        raise ValueError('placement backend must be checked, cpu or gpu')
    decision = {'requested': mode, 'selected': 'official-cpu', 'same_input_cpu_comparison': None}
    boundary = None
    if role in ('t1-pial', 't2-pial'):
        strategy = medial_wall_strategy if medial_wall_strategy is not None else strategy_for(stage)
        arguments, boundary = prepare(arguments, strategy)
        decision['medial_wall'] = boundary
    bundle = Path(stage.spec.get('pial_runtime', BUNDLE)).resolve()
    env, reason = (None, 'CPU backend selected') if mode == 'cpu' else configuration(stage, hemi, bundle, role)
    if env is None:
        if mode == 'gpu':
            raise RuntimeError(reason)
        stage.run(STOCK, arguments)
        if boundary is not None:
            verify_output(Path(arguments[arguments.index('--o') + 1]), boundary)
        return dict(decision, reason=reason)
    decision['cuda_visible_device'] = env['CUDA_VISIBLE_DEVICES']
    output_index = arguments.index('--o') + 1
    final = Path(arguments[output_index])
    white = nib.freesurfer.read_geometry(arguments[arguments.index('--white-surf' if role == 't2-pial' else '--i') + 1],
                                         read_metadata=role != 't2-pial')
    with tempfile.TemporaryDirectory(prefix=role + '-' + hemi + '-', dir=stage.root) as work:
        def redirected(folder):
            folder.mkdir()
            args = list(arguments)
            args[output_index] = folder / (hemi + '.' + surface)
            return [*args, '--target', folder / (hemi + '.targets'), '--ripflag-out', folder / 'rip.mgz']

        gpu = Path(work) / 'gpu'
        gpu_args = redirected(gpu)
        code = stage.run_process('fastprep:' + role + '-cuda-' + hemi, [bundle / executable, *gpu_args], env, gpu, check=False)
        candidate = None
        if not code:
            try:
                candidate = endpoints(gpu, hemi, white, surface)
            except (ValueError, OSError, KeyError, IndexError, EOFError, struct.error, nib.filebasedimages.ImageFileError) as error:
                decision['reason'] = 'invalid GPU output: ' + str(error)
        else:
            decision['reason'] = f'GPU placement exited {code}'
        if mode == 'gpu' and candidate is None:
            evidence = preserve_failure(gpu, stage.root, role, hemi, decision['reason'])
            raise RuntimeError(decision['reason'] + '; replay evidence: ' + str(evidence))
        if mode == 'checked':
            stock = Path(work) / 'stock'
            stage.run(STOCK, redirected(stock))
            reference = endpoints(stock, hemi, white, surface)
            if candidate is not None:
                comparison = differences(reference, candidate, surface)
                decision['same_input_cpu_comparison'] = comparison
                if not comparison['exact']:
                    decision['reason'] = 'GPU differs from official same-input endpoint'
                    candidate = None
        if candidate is not None:
            decision['selected'] = 'cuda-parallel' if role == 't2-pial' else 'cuda-ordered-signed'
        chosen = gpu if candidate is not None else stock
        if boundary is not None:
            verify_output(chosen / (hemi + '.' + surface), boundary)
        os.replace(chosen / (hemi + '.' + surface), final)
    return decision

"""Resolve a specific TemplateFlow volumetric atlas, with an explicit prefetch CLI."""

import argparse
import os
from pathlib import Path
import re


DEFAULT_TEMPLATE = 'tpl-MNI152NLin2009cAsym'


def options(parser):
    parser.add_argument('--mni-template', default=DEFAULT_TEMPLATE)
    parser.add_argument('--mni-resolution', type=int, default=1)
    parser.add_argument('--templateflow-home', type=Path, default=Path(os.environ.get('TEMPLATEFLOW_HOME', '~/.cache/templateflow')).expanduser())


def template_resources(args, download=False):
    identifier = args.mni_template.removeprefix('tpl-')
    resolution = args.mni_resolution
    if not re.fullmatch(r'[A-Za-z0-9]+', identifier) or resolution < 1:
        raise ValueError('Invalid TemplateFlow identifier or resolution')
    home = args.templateflow_home.expanduser().absolute()
    if download:
        os.environ['TEMPLATEFLOW_HOME'] = str(home)
        from templateflow.api import get
    prefix = f'tpl-{identifier}_res-{resolution:02}'
    resources = dict(id=identifier, resolution=resolution)
    for key, suffix, desc in [('t1w', 'T1w', None), ('mask', 'mask', 'brain')]:
        name = f'{prefix}' + (f'_desc-{desc}' if desc else '') + f'_{suffix}.nii.gz'
        path = home / f'tpl-{identifier}' / name
        if download:
            fetched = get(identifier, resolution=resolution, suffix=suffix, desc=desc, extension='.nii.gz')
            if not isinstance(fetched, Path) or fetched.resolve() != path.resolve():
                raise ValueError('TemplateFlow lookup did not uniquely resolve ' + name)
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f'Missing {path}; prefetch with bin/mni_template.py and the same template options')
        resources[key] = str(path)
    return resources


def validate_template(resources):
    import nibabel as nib
    import numpy as np

    grid = nib.load(resources['t1w'])
    image = np.asarray(grid.dataobj)
    if (len(grid.shape) != 3 or not np.isfinite(grid.affine).all() or abs(np.linalg.det(grid.affine[:3, :3])) < 1e-8
            or not np.isfinite(image).all() or image.max() <= image.min()):
        raise ValueError('Invalid or empty 3D TemplateFlow T1w reference')
    mask = nib.load(resources['mask'])
    values = np.asarray(mask.dataobj)
    if (mask.shape != grid.shape or not np.allclose(mask.affine, grid.affine, atol=1e-4, rtol=0)
            or not values.any() or values.all() or not np.isin(values, [0, 1]).all()):
        raise ValueError('Expected a binary brain mask on the reference grid')
    return grid


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    options(parser)
    resources = template_resources(parser.parse_args(), download=True)
    validate_template(resources)
    print(resources)

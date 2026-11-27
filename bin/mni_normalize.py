"""Normalize native structural intensities, not the cortical reconstruction, to TemplateFlow."""

from mni_template import validate_template
from session_inputs import ALIGNMENT_SCHEMA, FLIRT_CONVENTION, read_flirt_matrix, require_flirt_header, session_plan


def flirt_frames(path, physical_path=None):
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk

    image = nib.load(path)
    require_flirt_header(image, path)
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(physical_path or path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3 or reader.GetSize() != image.shape:
        raise ValueError('Expected a 3D structural image: ' + str(path))
    scaled = np.diag([*image.header.get_zooms(), 1.0])
    if np.linalg.det(image.affine[:3, :3]) > 0:
        scaled[0, 3] = (image.shape[0] - 1) * scaled[0, 0]
        scaled[0, 0] *= -1
    physical = np.eye(4)
    physical[:3, :3] = np.array(reader.GetDirection()).reshape(3, 3) @ np.diag(reader.GetSpacing())
    physical[:3, 3] = reader.GetOrigin()
    return scaled, physical


def write_itk_pull(moving, fixed, flirt, output, *, moving_geometry=None, fixed_geometry=None):
    import numpy as np
    import SimpleITK as sitk

    matrix = read_flirt_matrix(flirt, rigid=True)
    # FLIRT matrices describe the ORIGINAL voxel arrays in FSL scaled mm.
    # Convert to the explicitly selected physical frames used by MNI/ITK.
    source_scaled, source_lps = flirt_frames(moving, moving_geometry)
    target_scaled, target_lps = flirt_frames(fixed, fixed_geometry)
    pull = source_lps @ np.linalg.inv(source_scaled) @ np.linalg.inv(matrix) @ target_scaled @ np.linalg.inv(target_lps)
    transform = sitk.AffineTransform(3)
    transform.SetMatrix(pull[:3, :3].ravel())
    transform.SetTranslation(pull[:3, 3])
    sitk.WriteTransform(transform, str(output))


def rigid_inputs(stage, plan):
    from reference_runtime import artifact_path, read_json

    directory, record = stage.inputs['alignment_manifest']
    alignment = read_json(artifact_path(directory, record))
    reference = plan['reference']
    if (alignment.get('schema') != ALIGNMENT_SCHEMA or alignment.get('transform_convention') != FLIRT_CONVENTION
            or alignment['analysis_unit_id'] != stage.subject or alignment['reference'] !=
            {key: reference[key] for key in ('path', 'series_id')}):
        raise ValueError('Alignment does not refer to this session, T1w and FLIRT convention')
    registrations = alignment['registrations']
    if sorted(item['series_id'] for item in registrations) != sorted(item['series_id'] for item in plan['companions']):
        raise ValueError('Companion list does not match the input plan')
    bound = {}
    for item in plan['companions']:
        registration = next(row for row in registrations if row['series_id'] == item['series_id'])
        members = [member for member in record['members'] if member['path'] == registration['transform']]
        if len(members) != 1 or registration['moving'] != item['path'] or registration['fixed'] != reference['path']:
            raise ValueError('Unbound companion rigid transform or mismatched moving/fixed images')
        bound[item['series_id']] = artifact_path(directory, members[0])
    return bound


def registration_args(template, reference, mask, prefix):
    if any(any(char in str(path) for char in ',[]') for path in [template, reference, mask, prefix]):
        raise ValueError('ANTs bracket syntax cannot encode a comma or square bracket in these paths')
    pair = f'{template},{reference}'
    return ['--dimensionality', '3', '--float', '1', '--output', prefix, '--random-seed', '2026',
            '--collapse-output-transforms', '1', '--write-composite-transform', '0',
            '--initial-moving-transform', f'[{pair},1]', '--transform', 'Affine[0.1]',
            '--metric', f'Mattes[{pair},1,32,Regular,0.25]', '--convergence', '[1000x500x250x50,1e-6,10]',
            '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox', '--masks', '[NULL,NULL]',
            '--transform', 'SyN[0.1,3,0]', '--metric', f'CC[{pair},1,4]', '--convergence', '[100x70x50x20,1e-6,10]',
            '--shrink-factors', '8x4x2x1', '--smoothing-sigmas', '3x2x1x0vox', '--masks', f'[{mask},NULL]']


def apply_args(source, template, output, chain):
    return ['-d', '3', '--float', '1', '-i', source, '-r', template, '-o', output, '-n', 'Linear',
            *[arg for transform in chain for arg in ('-t', transform)]]


def bind_fireants_command(stage):
    """A selectively resumed MNI node must use its own helper implementation."""
    from pathlib import Path

    if 'commands' not in stage.runtime:  # Lightweight integration-test stage.
        return
    command = stage.runtime['commands']['fireants:register']
    argv = list(command['argv_prefix'])
    scripts = [index for index, value in enumerate(argv) if Path(value).name == 'fireants_register.py']
    if len(scripts) != 1:
        raise ValueError('Cannot bind FireANTs command to the selected MNI helpers')
    argv[scripts[0]] = str(Path(__file__).resolve().with_name('fireants_register.py'))
    stage.runtime = dict(stage.runtime, commands={**stage.runtime['commands'],
                         'fireants:register': dict(command, argv_prefix=argv)})


def normalize_session(stage):
    import time
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    from reference_runtime import checked_file, read_json, write_json
    from mni_geometry import POLICY, prepare_image

    started = time.perf_counter()
    plan = session_plan(stage.runtime, stage.subject)
    rigids = rigid_inputs(stage, plan)
    atlas = stage.runtime['mni_template']
    template, mask = [checked_file(atlas[key]) for key in ('t1w', 'mask')]
    validate_template(atlas)
    reference = checked_file(plan['reference']['path'])
    original_reference = reference
    destination = stage.output('mni/manifest.json').parent
    geometry = {}
    template, geometry['template'] = prepare_image(template, destination/'geometry/template.nii.gz')
    mask, geometry['mask'] = prepare_image(mask, destination/'geometry/mask.nii.gz')
    prepared_sources = {}
    for item in [plan['reference'], *plan['companions']]:
        key = item['series_id']
        prepared_sources[key], geometry[key] = prepare_image(
            checked_file(item['path']), destination/'geometry'/(key + '.nii.gz'))
    reference = prepared_sources[plan['reference']['series_id']]
    grid = validate_template(dict(t1w=template, mask=mask))
    geometry_path = destination/'geometry.json'
    write_json(geometry_path, dict(schema=POLICY, inputs=geometry,
                                  scientific_approval='not_performed'))
    def grid_values(path):
        image = nib.load(path)
        values = np.asarray(image.dataobj)
        if (image.shape != grid.shape or not np.allclose(image.affine, grid.affine, rtol=0, atol=1e-4)
                or not np.isfinite(values).all() or not values.any()):
            raise ValueError('Invalid image content or template grid: ' + str(path))
        return values

    space = f"space-{atlas['id']}_res-{atlas['resolution']:02}"
    prefix = destination/'T1wToTemplate'
    backend = stage.node.get('parameters', {}).get('backend', 'ants')
    registration_started = time.perf_counter()
    if backend == 'fireants':
        bind_fireants_command(stage)
        stage.run('fireants:register', ['--fixed', template, '--moving', reference, '--mask', mask,
                  '--output-prefix', prefix, '--device', stage.spec['device'], '--threads', stage.spec['threads'],
                  '--prepared-inputs'])
    elif backend == 'ants':
        stage.run('ants:antsRegistration', registration_args(template, reference, mask, prefix))
    else:
        raise ValueError('Unknown MNI registration backend: ' + str(backend))
    registration_seconds = time.perf_counter() - registration_started
    affine, warp, inverse = [checked_file(str(prefix) + suffix) for suffix in
                             ('0GenericAffine.mat', '1Warp.nii.gz', '1InverseWarp.nii.gz')]
    linear = sitk.AffineTransform(sitk.ReadTransform(str(affine)))
    determinant = float(np.linalg.det(np.array(linear.GetMatrix()).reshape(3, 3)))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError('Invalid T1w-to-template affine determinant')
    field = nib.load(warp)
    if (field.shape != (*grid.shape, 1, 3) or not np.allclose(field.affine, grid.affine, atol=1e-4, rtol=0)
            or not np.isfinite(np.asarray(field.dataobj)).all()):
        raise ValueError('Invalid forward warp on template grid')
    inverse_field = nib.load(inverse)
    source_grid = nib.load(reference)
    if (inverse_field.shape != (*source_grid.shape, 1, 3)
            or not np.allclose(inverse_field.affine, source_grid.affine, atol=1e-4, rtol=0)
            or not np.isfinite(np.asarray(inverse_field.dataobj)).all()):
        raise ValueError('Invalid inverse warp on native T1w grid')
    paths, records = [affine, warp, inverse, geometry_path, template, mask, *prepared_sources.values()], []
    registration = dict(backend=backend, registration_seconds=registration_seconds,
                        forward_includes_affine=backend == 'fireants')
    if backend == 'fireants':
        provenance_path = checked_file(str(prefix) + 'fireants.json')
        provenance = read_json(provenance_path)
        if (provenance.get('schema') != 'fastprep.fireants/1' or provenance.get('backend') != 'fireants'
                or provenance.get('forward_includes_affine') is not True
                or provenance.get('inverse_includes_affine') is not True
                or not provenance.get('device', '').startswith('cuda:')
                or provenance.get('fixed') != str(template.resolve())
                or provenance.get('moving') != str(reference.resolve())
                or provenance.get('fixed_mask') != str(mask.resolve())):
            raise ValueError('FireANTs provenance does not bind these inputs, device and full pull convention')
        paths.append(provenance_path)
        jacobian_path = checked_file(str(prefix) + 'Jacobian.nii.gz')
        jacobian = grid_values(jacobian_path)
        brain = np.asarray(nib.load(mask).dataobj) > 0
        if (jacobian[brain] <= 0).any():
            raise ValueError('Non-positive FireANTs Jacobian in template brain')
        paths.append(jacobian_path)
        registration.update(provenance=provenance_path.name, algorithm=provenance['algorithm'],
                            version=provenance['version'], device=provenance['device'])
    resampling_started = time.perf_counter()
    for item in [plan['reference'], *plan['companions']]:
        source = checked_file(item['path'])
        working_source = prepared_sources[item['series_id']]
        chain = [warp] if backend == 'fireants' else [warp, affine]
        if item['series_id'] in rigids:
            rigid = destination/(item['series_id'] + '_to-T1w.tfm')
            write_itk_pull(source, original_reference, rigids[item['series_id']], rigid,
                           moving_geometry=working_source, fixed_geometry=reference)
            paths.append(rigid)
            chain.append(rigid)
        stem, modality = item['series_id'].rsplit('_', 1)
        output = destination/f'{stem}_{space}_{modality}.nii.gz'
        stage.run('ants:antsApplyTransforms', apply_args(working_source, template, output, chain))
        grid_values(output)
        paths.append(output)
        records.append(dict(source=str(source), series_id=item['series_id'], image=output.name,
                            prepared_source=str(working_source.relative_to(destination)),
                            ants_image_transform_list=[path.name for path in chain], interpolation='Linear'))
    manifest = destination/'manifest.json'
    registration.update(resampling_seconds=time.perf_counter() - resampling_started,
                        total_seconds=time.perf_counter() - started)
    write_json(manifest, dict(schema='fastprep.mni/1', analysis_unit_id=stage.subject, reference=str(original_reference),
               prepared_reference=str(reference.relative_to(destination)), geometry_policy=POLICY,
               geometry_provenance=geometry_path.name,
               native_grid_shape=list(source_grid.shape), native_grid_affine_ras_mm=source_grid.affine.tolist(),
               template=atlas, grid_shape=grid.shape, grid_affine_ras_mm=grid.affine.tolist(), images=records,
               registration=registration,
               inverse_ants_image_transform_list=[inverse.name] if backend == 'fireants' else [f'[{affine.name},1]', inverse.name],
               affine_determinant=determinant, inverse_warp=inverse.name, scientific_approval='not_performed',
               members=[stage.file_record(path) for path in paths]))
    stage.metrics.update(mni_template=atlas['id'], mni_backend=backend, normalized_images=len(records),
                         mni_registration_seconds=registration_seconds, mni_total_seconds=registration['total_seconds'],
                         visual_review='pending')
    return {'mni_manifest': manifest}

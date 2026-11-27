"""Resolve NIfTI geometry once, before ITK registration and image resampling.

Prefer a representable sform, then a coded qform. Never approximate a sheared
sform silently. Uncoded images use NiBabel's declared centered LAS fallback;
that processing convention is not evidence of anatomical orientation.
"""

from pathlib import Path

POLICY = 'fastprep.mni_geometry/1'
ATOL = 1e-4


def itk_affine(path):
    import numpy as np
    import SimpleITK as sitk

    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    if reader.GetDimension() != 3:
        raise ValueError('MNI requires a 3D image: ' + str(path))
    lps = np.eye(4)
    lps[:3, :3] = np.asarray(reader.GetDirection()).reshape(3, 3) @ np.diag(reader.GetSpacing())
    lps[:3, 3] = reader.GetOrigin()
    return np.diag([-1., -1., 1., 1.]) @ lps


def assert_geometry(path):
    import nibabel as nib
    import numpy as np

    image = nib.load(path)
    qform, qcode = image.get_qform(coded=True)
    sform, scode = image.get_sform(coded=True)
    if (not qcode or not scode or not np.allclose(qform, sform, atol=ATOL, rtol=0)
            or not np.allclose(image.affine, itk_affine(path), atol=ATOL, rtol=0)):
        raise ValueError('MNI prepared geometry differs between NIfTI/ITK readers: ' + str(path))
    return image


def prepare_image(source, destination):
    """Write a lossless header-only working copy and an auditable selection record.

    No array reorientation, resampling, or intensity conversion takes place.
    Both spatial forms are explicitly coded on the working copy. A form is
    representable only if qform round-trip preserves its affine within ATOL.
    """
    import itertools
    import nibabel as nib
    import numpy as np

    source, destination = Path(source).resolve(), Path(destination).absolute()
    if destination.exists() or source == destination:
        raise ValueError('Use a new MNI geometry destination: ' + str(destination))
    image = nib.load(source)
    if len(image.shape) != 3 or not isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)):
        raise ValueError('MNI requires a 3D NIfTI input: ' + str(source))
    unit = image.header.get_xyzt_units()[0]
    if unit not in ('mm', 'unknown'):
        raise ValueError('MNI requires mm units (or unspecified legacy units): ' + str(source))
    forms = {}
    for name in ('qform', 'sform'):
        matrix, code = getattr(image, 'get_' + name)(coded=True)
        if code and (not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-8):
            raise ValueError('Invalid coded ' + name + ': ' + str(source))
        forms[name] = dict(code=int(code), affine_ras_mm=None if matrix is None else matrix.tolist())
    if forms['qform']['code'] and forms['sform']['code']:
        if np.sign(np.linalg.det(image.get_qform()[:3, :3])) != np.sign(np.linalg.det(image.get_sform()[:3, :3])):
            raise ValueError('Conflicting qform/sform handedness: ' + str(source))

    def representable(matrix):
        trial = image.header.copy()
        trial.set_qform(matrix)
        return np.allclose(trial.get_qform(), matrix, atol=ATOL, rtol=0)

    warnings = []
    if forms['sform']['code'] and representable(image.get_sform()):
        selected, code, matrix = 'sform', forms['sform']['code'], image.get_sform()
    elif forms['qform']['code']:
        selected, code, matrix = 'qform', forms['qform']['code'], image.get_qform()
        if forms['sform']['code']:
            warnings.append('sform_not_representable_without_shear_loss; selected_coded_qform')
    elif forms['sform']['code']:
        raise ValueError('Sheared sform with no coded qform requires explicit geometric resampling: ' + str(source))
    else:
        selected, code, matrix = 'nibabel_fallback', 2, image.affine
        warnings.append('uncoded_spatial_forms; anatomical_orientation_unverified')
    if forms['sform']['code'] and forms['qform']['code'] and not np.allclose(
            image.get_sform(), image.get_qform(), atol=ATOL, rtol=0):
        warnings.append('coded_qform_sform_disagree; selection_requires_anatomical_review')
    if unit == 'unknown':
        warnings.append('unspecified_spatial_units_assumed_mm')

    # Preserve the raw stored samples and slope/intercept, including integer
    # images with nontrivial scaling. Re-saving scaled floats would be lossy.
    proxy = image.dataobj
    raw = proxy.get_unscaled()
    output = image.__class__(raw, matrix, header=image.header.copy())
    output.set_qform(matrix, code)
    chosen = output.get_qform()
    if not np.allclose(chosen, matrix, atol=ATOL, rtol=0):
        raise ValueError('Selected MNI geometry cannot be represented without shear loss')
    output.set_sform(chosen, code)
    output.header.set_xyzt_units('mm', image.header.get_xyzt_units()[1])
    output.header.set_slope_inter(proxy.slope, proxy.inter)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(output, destination)
    prepared = assert_geometry(destination)
    if (prepared.shape != image.shape or prepared.get_data_dtype() != image.get_data_dtype()
            or not np.array_equal(prepared.dataobj.get_unscaled(), raw)
            or prepared.dataobj.slope != proxy.slope or prepared.dataobj.inter != proxy.inter):
        raise ValueError('MNI geometry preparation changed voxel samples: ' + str(source))
    corners = np.array(list(itertools.product(*[(0, size - 1) for size in image.shape])))
    delta = prepared.affine - image.affine
    corner_change = np.linalg.norm(corners @ delta[:3, :3].T + delta[:3, 3], axis=1)
    return destination, dict(schema=POLICY, source=str(source), prepared=str(destination.resolve()),
                             selected_form=selected, original_forms=forms, warnings=warnings,
                             original_nibabel_affine_ras_mm=image.affine.tolist(),
                             affine_ras_mm=prepared.affine.tolist(), shape=list(image.shape),
                             max_corner_difference_from_original_nibabel_mm=float(corner_change.max()),
                             voxel_samples='unchanged', interpolation='none', spatial_units='mm',
                             original_spatial_units=unit)

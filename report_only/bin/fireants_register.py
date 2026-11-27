"""GPU FireANTs registration with composed ITK LPS pull displacement fields."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time


SCALES = [8, 4, 2, 1]
AFFINE_ITERATIONS = [200, 100, 50, 20]
WARP_ITERATIONS = [100, 70, 50, 20]
PULL_CONVENTION = 'ITK LPS mm; full fixed-to-moving pull including affine'


def cuda_device(requested):
    import torch

    requested = str(requested)
    selected = torch.device('cuda:' + requested if requested.isdigit() else requested)
    if selected.type != 'cuda':
        raise ValueError('FireANTs requires an explicitly selected CUDA device; use --mni-backend ants for CPU')
    if not torch.cuda.is_available():
        raise RuntimeError('FireANTs CUDA is unavailable; CPU fallback is disabled')
    selected = torch.device('cuda', 0 if selected.index is None else selected.index)
    torch.cuda.set_device(selected)
    probe = torch.ones((2, 2), device=selected)
    if not (probe @ probe).is_cuda:
        raise RuntimeError('FireANTs CUDA execution probe failed')
    torch.cuda.synchronize(selected)
    return torch.device('cuda', torch.cuda.current_device())


def inverse_coordinates(parameters, moving_shape, moving_t2p, max_iterations=200, tolerance_mm=0.05):
    """Invert y = A x + u(x) on the native moving grid using checked fixed point iteration."""
    import torch
    from torch.nn import functional as functional

    if max_iterations < 1:
        raise ValueError('Inverse displacement requires at least one iteration')
    with torch.no_grad():
        linear = parameters['affine']
        displacement = parameters['grid'].detach().movedim(-1, 1)
        inverse_linear = torch.linalg.inv(linear[:, :, :3])
        identity = torch.eye(3, 4, device=linear.device, dtype=linear.dtype).unsqueeze(0)
        native = functional.affine_grid(identity, [1, 1, *moving_shape], align_corners=True)
        initial = torch.einsum('bij,b...j->b...i', inverse_linear, native - linear[:, None, None, None, :, 3])
        estimate = initial.clone()
        margin = 2 / (torch.tensor(list(reversed(displacement.shape[2:])), device=linear.device) - 1)
        for iteration in range(max_iterations):
            sampled = functional.grid_sample(displacement, estimate, padding_mode='border', align_corners=True).movedim(1, -1)
            updated = initial - torch.einsum('bij,b...j->b...i', inverse_linear, sampled)
            estimate = 0.25 * estimate + 0.75 * updated
            if (iteration + 1) % 10 and iteration + 1 < max_iterations:
                continue
            sampled = functional.grid_sample(displacement, estimate, padding_mode='border', align_corners=True).movedim(1, -1)
            residual = torch.einsum('bij,b...j->b...i', linear[:, :, :3], estimate) + linear[:, None, None, None, :, 3] + sampled - native
            residual_mm = torch.linalg.vector_norm(torch.einsum('bij,b...j->b...i', moving_t2p[:, :3, :3], residual), dim=-1)
            valid = (estimate.abs() < 1 - margin).all(dim=-1)
            if not torch.isfinite(estimate).all() or not torch.isfinite(residual_mm).all():
                raise ValueError('Non-finite inverse displacement')
            errors = residual_mm[valid]
            maximum = float(errors.max()) if errors.numel() else None
            report = dict(method='damped_fixed_point_full_pull', iterations=iteration + 1,
                          tolerance_mm=tolerance_mm, max_residual_mm=maximum,
                          rms_residual_mm=float(errors.square().mean().sqrt()) if errors.numel() else None,
                          checked_voxels=int(valid.sum()), checked_fraction=float(valid.float().mean()))
            if maximum is not None and maximum <= tolerance_mm:
                return estimate, dict(report, status='converged')
        return estimate, dict(report, status='unverified' if maximum is None else 'tolerance_exceeded')


def save_inverse(coordinates, fixed_images, moving_images, destination):
    """Save moving-grid to fixed-space coordinates as an ITK LPS displacement."""
    import SimpleITK as sitk
    import torch
    from torch.nn import functional as functional

    with torch.no_grad():
        fixed_t2p, moving_t2p = fixed_images.get_torch2phy(), moving_images.get_torch2phy()
        identity = torch.eye(3, 4, device=coordinates.device, dtype=coordinates.dtype).unsqueeze(0)
        native = functional.affine_grid(identity, [1, 1, *coordinates.shape[1:-1]], align_corners=True)
        displacement = (torch.einsum('bij,b...j->b...i', fixed_t2p[:, :3, :3], coordinates)
                        + fixed_t2p[:, None, None, None, :3, 3]
                        - torch.einsum('bij,b...j->b...i', moving_t2p[:, :3, :3], native)
                        - moving_t2p[:, None, None, None, :3, 3])
        image = sitk.GetImageFromArray(displacement[0].cpu().numpy(), isVector=True)
        image.CopyInformation(moving_images.images[0].itk_image)
        sitk.WriteImage(image, str(destination))


def compose_affine(parameters, normalized_affine):
    """Compose the physical affine after a deformation fitted on its resampled grid."""
    import torch

    with torch.no_grad():
        row = torch.zeros((1, 1, 4), dtype=normalized_affine.dtype, device=normalized_affine.device)
        row[:, :, 3] = 1
        return dict(affine=normalized_affine @ torch.cat([parameters['affine'], row], dim=1),
                    grid=torch.einsum('bij,b...j->b...i', normalized_affine[:, :, :3], parameters['grid']))


def save_forward(parameters, fixed_images, moving_images, destination):
    import torch
    from torch.nn import functional as functional

    with torch.no_grad():
        coordinates = functional.affine_grid(parameters['affine'], fixed_images.shape, align_corners=True) + parameters['grid']
        save_inverse(coordinates, moving_images, fixed_images, destination)


def physical_jacobian(parameters, fixed_images, moving_images):
    """Differentiate in normalized axes, then convert determinants to physical LPS axes."""
    import torch

    with torch.no_grad():
        displacement = parameters['grid'].detach()
        derivatives = torch.gradient(displacement, spacing=tuple(2 / (size - 1) for size in displacement.shape[1:4]),
                                     dim=(1, 2, 3))
        differential = torch.stack(list(reversed(derivatives)), dim=-1) + parameters['affine'][:, None, None, None, :3, :3]
        conversion = (torch.linalg.det(moving_images.get_torch2phy()[:, :3, :3])
                      / torch.linalg.det(fixed_images.get_torch2phy()[:, :3, :3]))
        return torch.linalg.det(differential) * conversion[:, None, None, None]


def register(args):
    started = time.perf_counter()
    import nibabel as nib
    import numpy as np
    import SimpleITK as sitk
    import torch
    from torch.nn import functional as functional
    from fireants.io.image import Image, BatchedImages
    from fireants.interpolator import fireants_interpolator
    from fireants.registration.affine import AffineRegistration
    from fireants.registration.greedy import GreedyRegistration
    from fireants.registration.moments import MomentsRegistration
    from mni_template import validate_template
    from mni_geometry import assert_geometry, prepare_image

    device = cuda_device(args.device)
    torch.set_num_threads(args.threads)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(args.threads)
    torch.manual_seed(2026)
    torch.cuda.reset_peak_memory_stats(device)
    validate_template(dict(t1w=args.fixed, mask=args.mask))
    source = nib.load(args.moving)
    values = np.asarray(source.dataobj)
    if (len(source.shape) != 3 or not np.isfinite(source.affine).all()
            or abs(np.linalg.det(source.affine[:3, :3])) < 1e-8
            or not np.isfinite(values).all() or values.max() <= values.min()):
        raise ValueError('Expected a finite, nonconstant 3D moving structural image')
    prefix = str(Path(args.output_prefix).absolute())
    outputs = {key: Path(prefix + suffix) for key, suffix in (
        ('affine', '0GenericAffine.mat'), ('forward', '1Warp.nii.gz'),
        ('inverse', '1InverseWarp.nii.gz'), ('jacobian', 'Jacobian.nii.gz'), ('provenance', 'fireants.json'))}
    if any(path.exists() for path in outputs.values()):
        raise ValueError('FireANTs outputs already exist; use a new output prefix')
    outputs['affine'].parent.mkdir(parents=True, exist_ok=True)
    input_paths = dict(fixed=args.fixed, moving=args.moving, mask=args.mask)
    geometry = {}
    working = {}
    for role, path in input_paths.items():
        if getattr(args, 'prepared_inputs', False):
            assert_geometry(path)
            working[role] = path
        else:
            working[role], geometry[role] = prepare_image(path, Path(prefix + 'inputs') / (role + '.nii.gz'))
    validate_template(dict(t1w=working['fixed'], mask=working['mask']))
    fixed_image, moving_image = [Image.load_file(str(working[role]), device=device) for role in ('fixed', 'moving')]
    fixed_images, moving_images = BatchedImages(fixed_image), BatchedImages(moving_image)
    timings = {}

    def finish_phase(name, phase_start):
        torch.cuda.synchronize(device)
        timings[name] = time.perf_counter() - phase_start
        print(json.dumps(dict(phase=name, seconds=timings[name], device=str(device))), flush=True)

    phase_start = time.perf_counter()
    moments = MomentsRegistration(scale=args.scales[0], fixed_images=fixed_images, moving_images=moving_images,
                                  moments=1, transl_mode='com', loss_params={}, progress_bar=False)
    moments.optimize()
    finish_phase('moments', phase_start)
    phase_start = time.perf_counter()
    affine = AffineRegistration(args.scales, args.affine_iterations, fixed_images, moving_images,
                                loss_type='mi', loss_params={}, optimizer_lr=0.03,
                                init_rigid=moments.get_affine_init(), progress_bar=False)
    affine.optimize()
    matrix = affine.get_affine_matrix().detach()
    if not torch.isfinite(matrix).all() or float(torch.linalg.det(matrix[:, :3, :3])[0]) == 0:
        raise ValueError('Invalid FireANTs affine transform')
    linear = sitk.AffineTransform(3)
    linear.SetMatrix(matrix[0, :3, :3].cpu().numpy().astype(float).ravel().tolist())
    linear.SetTranslation(matrix[0, :3, 3].cpu().numpy().astype(float).tolist())
    sitk.WriteTransform(linear, str(outputs['affine']))
    finish_phase('affine', phase_start)
    phase_start = time.perf_counter()
    with torch.no_grad():
        normalized_affine = (moving_images.get_phy2torch() @ matrix @ fixed_images.get_torch2phy())[:, :3]
        fit_coordinates = functional.affine_grid(normalized_affine, fixed_images.shape, align_corners=True)
        fit_image = Image(fixed_image.itk_image, device=device)
        fit_image.array = functional.grid_sample(moving_images(), fit_coordinates, align_corners=True)
        fit_mask = Image(fixed_image.itk_image, device=device)
        fit_mask.array = functional.grid_sample(torch.ones_like(moving_images()), fit_coordinates, align_corners=True)
    mask_image = Image.load_file(str(working['mask']), device=device)
    fixed_image.concatenate(mask_image)
    fit_image.concatenate(fit_mask)
    fit_fixed_images, fit_moving_images = BatchedImages(fixed_image), BatchedImages(fit_image)
    finish_phase('affine_fit_resampling', phase_start)
    phase_start = time.perf_counter()
    nonlinear = GreedyRegistration(args.scales, args.warp_iterations, fit_fixed_images, fit_moving_images,
                                    loss_type='masked_cc', loss_params={}, cc_kernel_size=9,
                                    optimizer_lr=0.1, deformation_type='compositive',
                                    smooth_warp_sigma=0.5, smooth_grad_sigma=1.0, progress_bar=False)
    nonlinear.optimize()
    finish_phase('nonlinear', phase_start)
    phase_start = time.perf_counter()
    with torch.no_grad():
        parameters = compose_affine(nonlinear.get_warp_parameters(fit_fixed_images, fit_moving_images), normalized_affine)
        save_forward(parameters, fixed_images, moving_images, outputs['forward'])
        jacobian = physical_jacobian(parameters, fixed_images, moving_images)[0].cpu().numpy()
        brain = np.asarray(nib.load(working['mask']).dataobj).transpose(2, 1, 0) > 0
        jacobian_report = dict(brain_nonpositive_voxels=int((jacobian[brain] <= 0).sum()),
                               brain_min=float(jacobian[brain].min()), brain_max=float(jacobian[brain].max()),
                               full_nonpositive_voxels=int((jacobian <= 0).sum()))
        if not np.isfinite(jacobian).all():
            raise ValueError('Non-finite Jacobian in FireANTs brain deformation: ' + str(jacobian_report))
        jacobian_image = sitk.GetImageFromArray(jacobian)
        jacobian_image.CopyInformation(fixed_image.itk_image)
        sitk.WriteImage(jacobian_image, str(outputs['jacobian']))
        coordinates, inverse_check = inverse_coordinates(parameters, moving_images.shape[2:], moving_images.get_torch2phy())
        save_inverse(coordinates, fixed_images, moving_images, outputs['inverse'])
    finish_phase('export_and_inverse', phase_start)
    for key, reference in [('forward', working['fixed']), ('inverse', working['moving'])]:
        field, grid = nib.load(outputs[key]), nib.load(reference)
        if (field.shape != (*grid.shape, 1, 3) or not np.allclose(field.affine, grid.affine, atol=1e-4, rtol=0)
                or not np.isfinite(np.asarray(field.dataobj)).all()):
            raise ValueError('Invalid FireANTs displacement field: ' + key)
    provenance = dict(schema='fastprep.fireants/1', backend='fireants', version=version('fireants'),
                      algorithm='affine_mi_then_greedy_masked_cc_compositive', device=str(device),
                      device_name=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                      fused_ops=bool(fireants_interpolator.use_ffo), seed=2026, threads=args.threads,
                      fixed=str(Path(args.fixed).resolve()), moving=str(Path(args.moving).resolve()),
                      fixed_mask=str(Path(args.mask).resolve()), transform_convention=PULL_CONVENTION,
                      working_inputs={role: str(Path(path).resolve()) for role, path in working.items()},
                      input_geometry=geometry, prepared_inputs=bool(getattr(args, 'prepared_inputs', False)),
                      forward_includes_affine=True, inverse_includes_affine=True,
                      inverse_check=inverse_check, jacobian_check=jacobian_report,
                      jacobian_image=outputs['jacobian'].name,
                      composition='physical_affine_after_template_grid_greedy',
                      fitting_image='GPU affine-resampled moving; final outputs sample original once',
                      scales=args.scales, affine_iterations=args.affine_iterations,
                      warp_iterations=args.warp_iterations, cc_kernel_size=9, affine_optimizer_lr=0.03,
                      warp_optimizer_lr=0.1, smooth_warp_sigma=0.5, smooth_grad_sigma=1.0,
                      convergence_tolerance=1e-6, convergence_window=10, timings_seconds=timings,
                      total_seconds=time.perf_counter() - started,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      scientific_approval='not_performed')
    outputs['provenance'].write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--fixed', type=Path)
    parser.add_argument('--moving', type=Path)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--output-prefix', type=Path)
    parser.add_argument('--prepared-inputs', action='store_true',
                        help='Require explicitly harmonized input forms; geometry provenance is held by the caller')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--scales', type=int, nargs='+', default=SCALES)
    parser.add_argument('--affine-iterations', type=int, nargs='+', default=AFFINE_ITERATIONS)
    parser.add_argument('--warp-iterations', type=int, nargs='+', default=WARP_ITERATIONS)
    args = parser.parse_args()
    if args.probe:
        from fireants.registration.affine import AffineRegistration
        from fireants.registration.greedy import GreedyRegistration
        from fireants.registration.moments import MomentsRegistration
        import torch
        print(json.dumps(dict(fireants=version('fireants'), torch=torch.__version__, cuda_build=torch.version.cuda)))
        return
    if not all((args.fixed, args.moving, args.mask, args.output_prefix)):
        parser.error('--fixed, --moving, --mask and --output-prefix are required')
    if (args.threads < 1 or not args.scales or args.scales[-1] != 1
            or any(first <= second for first, second in zip(args.scales, args.scales[1:]))
            or any(len(counts) != len(args.scales) or any(count < 1 for count in counts)
                   for counts in (args.affine_iterations, args.warp_iterations))):
        parser.error('Require positive threads/iterations and decreasing scales ending in 1 with matching iteration counts')
    register(args)


if __name__ == '__main__':
    main()

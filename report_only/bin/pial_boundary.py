"""Keep the final medial-wall constraint active throughout native placement.

The stock final coordinate reset bypasses collision-aware movement.  Starting
from a pinned surface and ripping the same noncortical vertices makes that
reset a no-op and keeps native cleanup from releasing the constraint.  Cortical
vertices are still optimized by the native image forces and collision checks.
This does not assert anatomical quality or replace the independent mesh gate.
"""

from pathlib import Path


LEGACY = 'legacy-final-pin'
FIXED = 'fixed-during-placement'
RECOVER = 'fixed-on-intersection'
GEOMETRY_KEYS = ('volume', 'voxelsize', 'xras', 'yras', 'zras', 'cras')


def strategy_for(stage):
    """A task-specific override allows pial-only recovery of an old workflow."""
    return stage.spec.get('medial_wall_strategy',
                          stage.node['parameters'].get('medial_wall_strategy', LEGACY))


def argument(arguments, flag):
    if arguments.count(flag) != 1:
        raise ValueError('Medial-wall constraint needs exactly one ' + flag)
    index = arguments.index(flag) + 1
    if index >= len(arguments):
        raise ValueError('Missing medial-wall argument for ' + flag)
    return arguments[index]


def mask(path, count):
    import nibabel as nib
    import numpy as np

    indices = np.atleast_1d(nib.freesurfer.read_label(path))
    if (not len(indices) or indices.min() < 0 or indices.max() >= count
            or len(np.unique(indices)) != len(indices)):
        raise ValueError('Invalid medial-wall/cortex label: ' + str(path))
    selected = np.zeros(count, dtype=bool)
    selected[indices] = True
    return selected


def verify(surface, white, cortex_label):
    import nibabel as nib
    import numpy as np

    vertices, faces, metadata = nib.freesurfer.read_geometry(surface, read_metadata=True)
    reference, reference_faces, reference_metadata = nib.freesurfer.read_geometry(white, read_metadata=True)
    if (vertices.shape != reference.shape or not len(vertices) or not np.isfinite(vertices).all()
            or not np.isfinite(reference).all() or not np.array_equal(faces, reference_faces)):
        raise ValueError('Medial-wall constraint requires finite corresponding surfaces')
    if any(key not in metadata or key not in reference_metadata
           or not np.isfinite(metadata[key]).all() or not np.isfinite(reference_metadata[key]).all()
           or not np.array_equal(metadata[key], reference_metadata[key]) for key in GEOMETRY_KEYS):
        raise ValueError('Medial-wall constraint requires matching reference volume geometry')
    cortex = mask(cortex_label, len(vertices))
    distance = np.linalg.norm(vertices[~cortex] - reference[~cortex], axis=1)
    report = dict(cortical_vertices=int(cortex.sum()), fixed_vertices=len(distance),
                  fixed_vertices_equal_white=bool(len(distance) and np.all(distance == 0)),
                  max_fixed_displacement_mm=float(distance.max()) if len(distance) else None)
    if not report['fixed_vertices_equal_white']:
        report['quality_warning'] = 'Fixed medial-wall constraint not demonstrated'
    return report


def prepare(arguments, strategy):
    """Validate the initial constraint and return new arguments plus provenance."""
    import nibabel as nib
    import numpy as np

    if strategy not in (LEGACY, FIXED, RECOVER):
        raise ValueError('Unknown medial_wall_strategy: ' + str(strategy))
    if strategy in (LEGACY, RECOVER):
        return list(arguments), dict(strategy=LEGACY, requested_strategy=strategy)
    if '--pial' not in arguments or '--white' in arguments:
        raise ValueError('Fixed medial-wall constraint is only supported for pial placement')
    if str(argument(arguments, '--nsmooth')) != '0':
        raise ValueError('Fixed medial-wall placement requires nsmooth=0; initial smoothing precedes native ripping')
    source = Path(argument(arguments, '--i'))
    white = Path(argument(arguments, '--white-surf'))
    cortex = Path(argument(arguments, '--pin-medial-wall'))
    expanded = Path(argument(arguments, '--rip-label'))
    before = verify(source, white, cortex)
    vertices, _ = nib.freesurfer.read_geometry(white)
    cortical, movable = (mask(path, len(vertices)) for path in (cortex, expanded))
    excluded = int(np.count_nonzero(cortical & ~movable))
    if excluded:
        return list(arguments), dict(strategy=LEGACY, requested_strategy=FIXED,
                                     quality_warning='Fixed constraint would expand the native activity label',
                                     cortical_vertices_outside_activity_label=excluded, input_constraint=before)
    rewritten = list(arguments)
    rewritten[rewritten.index('--rip-label') + 1] = cortex
    return rewritten, dict(strategy=FIXED, input_surface=str(source), white_surface=str(white),
                           cortex_label=str(cortex), original_activity_label=str(expanded),
                           effective_activity_label=str(cortex),
                           newly_fixed_vertices=int(np.count_nonzero(movable & ~cortical)),
                           input_constraint=before, scientific_approval='not_performed',
                           note='Native placement uses the final medial-wall constraint from the start; no post-placement snapping or smoothing repair is accepted.')


def verify_output(path, record):
    if record['strategy'] == FIXED:
        record['output_constraint'] = verify(path, record['white_surface'], record['cortex_label'])

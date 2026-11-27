"""Reusable evidence rendering. No QC pass is inferred from a generated image."""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from .evidence import provenance, write_json
from .images import (PLANES, image_record, limits, load_volume, panel,
                     panel_aspect, require_same_grid, same_grid, select_slices)
from .surfaces import mesh_voxels, read_surface, segments, validate_mesh


def _figure(rows, columns, figsize):
    if rows < 1 or columns < 1:
        raise ValueError("Rows and columns must be positive")
    figure = Figure(figsize=figsize, facecolor="black")
    FigureCanvasAgg(figure)
    return figure, figure.subplots(rows, columns, squeeze=False)


def _finish(figure, output, metadata, *, tight=True):
    output = Path(output)
    if output.suffix.lower() != ".png":
        raise ValueError("Render output must have a .png extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.update(output=str(output.resolve()), evidence_kind="visual",
                    status="unknown", coverage_assessment="indeterminate")
    if tight:
        figure.tight_layout(rect=(0, 0, 1, .95), pad=.5)
    figure.savefig(output, dpi=150, facecolor="black")
    figure.clear()
    write_json(str(output) + ".json", metadata)
    return metadata


def _source_layout(image_path, plane, slices, slice_range, slice_count, rows, columns, window):
    image, raw = load_volume(image_path, canonical=True)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite image data cannot be rendered as valid visual evidence")
    axis = PLANES[plane]
    chosen = select_slices(raw, axis, slices=slices, slice_range=slice_range, count=slice_count)
    if len(chosen) > rows * columns:
        raise ValueError("Selected slices exceed layout capacity")
    parameters = {"plane": plane, "slices": chosen, "slice_count": slice_count,
                  "rows": rows, "columns": columns, "window": list(limits(raw, window)),
                  "slice_range": slice_range, "selection": "explicit" if slices is not None else
                  "explicit_range" if slice_range is not None else "source_nonzero_range",
                  "display_origin": "lower", "display_aspect": panel_aspect(image, axis)}
    return image, raw, parameters


def _label_volume(image, labels_path):
    label_image, labels = load_volume(labels_path, canonical=True)
    require_same_grid(image, label_image)
    if not np.isfinite(labels).all():
        raise ValueError("Label image and selected label must be finite")
    return labels


def _draw_montage(output, metadata, backgrounds, contours=None):
    parameters = metadata['parameters']
    rows, columns = parameters['rows'], parameters['columns']
    chosen, plane = parameters['slices'], parameters['plane']
    display_window, aspect = parameters['window'], parameters['display_aspect']
    label, color = parameters.get('label'), parameters.get('contour_color')
    figure, axes = _figure(rows, columns, (columns * 3.4, rows * 3.2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, index, background in zip(axes.flat, chosen, backgrounds):
        ax.imshow(background, cmap="gray", vmin=display_window[0],
                  vmax=display_window[1], origin="lower", aspect=aspect, interpolation="nearest")
        annotation = ""
        if contours is not None:
            contour = next(contours)
            if contour.any():
                # Padding also exposes contours for a completely filled slice.
                padded = np.pad(contour.astype(float), 1)
                ax.contour(np.arange(-1, contour.shape[1] + 1), np.arange(-1, contour.shape[0] + 1),
                           padded, levels=[.5], colors=[color], linewidths=1.15)
                ax.set_xlim(-.5, contour.shape[1] - .5)
                ax.set_ylim(-.5, contour.shape[0] - .5)
            else:
                annotation = " | no selected label"
        ax.set_title(f"{plane} index={index}{annotation}", color="white", fontsize=8)
    title = "Source image" if contours is None else f"Label={label} | contour={color}"
    figure.suptitle(f"{title} | canonical voxel planes | review coverage separately", color="white")
    return _finish(figure, output, metadata)


def _volume_montage(image_path, output, *, labels_path=None, label=None,
                    color="#ff3030", plane="axial", slices=None, slice_range=None,
                    slice_count=20, rows=4, columns=5, window=None):
    image, raw, parameters = _source_layout(image_path, plane, slices, slice_range, slice_count, rows, columns, window)
    axis, chosen = PLANES[plane], parameters['slices']
    contours = None
    if labels_path is not None:
        labels = _label_volume(image, labels_path)
        if label is None or not np.isfinite(label):
            raise ValueError("Label image and selected label must be finite")
        mask = labels == label
        parameters.update(label=label, contour_color=color, selected_label_empty=not bool(mask.any()))
        contours = (panel(mask, axis, index) for index in chosen)
    metadata = provenance({"image": image_path, "labels": labels_path}, parameters)
    metadata["display_image"] = image_record(image_path, image)
    metadata["original_image"] = image_record(image_path, load_volume(image_path)[0])
    return _draw_montage(output, metadata, (panel(raw, axis, index) for index in chosen), contours)


def label_montages(image_path, labels_path, source_output, contour_outputs, command):
    """Render the workflow's axial evidence contract; cache only within this input pair."""
    import nibabel as nib

    paths = [Path(path).resolve() for _, path in contour_outputs]
    if source_output is not None:
        paths.append(Path(source_output).resolve())
    if not paths or len(paths) != len(set(paths)) or any(path.suffix != '.png' for path in paths):
        raise ValueError('Expected distinct PNG output paths for label-batch')
    image, raw, layout = _source_layout(image_path, 'axial', None, None, 20, 4, 5, None)
    chosen = layout['slices']
    backgrounds = [np.ascontiguousarray(raw[:, :, index].T) for index in chosen]
    del raw
    headers = dict(display_image=image_record(image_path, image), original_image=image_record(image_path, nib.load(image_path)))
    outputs, failed = [], {}

    def render(path, label=None):
        try:
            parameters = layout.copy()
            contours = None
            if label is not None:
                if not np.isfinite(label):
                    raise ValueError('Selected label must be finite')
                parameters.update(label=label, contour_color='#ff3030', selected_label_empty=label not in present)
                contours = (plane == label for plane in label_panels)
            metadata = provenance({'image': image_path, 'labels': labels_path if label is not None else None}, parameters)
            record = _draw_montage(path, dict(metadata, **headers), backgrounds, contours)
            write_json(str(path) + '.json', dict(record, command=command))
            outputs.append(str(path))
        except (ValueError, OSError) as error:
            failed[str(path)] = str(error)

    if source_output is not None:
        render(source_output)
    if contour_outputs:
        try:
            labels = _label_volume(image, labels_path)
            present = set(np.unique(labels).tolist())
            label_panels = [np.ascontiguousarray(labels[:, :, index].T) for index in chosen]
        except (ValueError, OSError) as error:
            failed.update({str(path): str(error) for _, path in contour_outputs})
        else:
            for label, path in contour_outputs:
                render(path, float(label))
    return {'completed': outputs, 'failed': failed}


def render_image(image_path, output, **options):
    return _volume_montage(image_path, output, **options)


def render_comparison(before_path, after_path, output, *, plane="axial", slices=None,
                      slice_range=None, slice_count=5, window=None):
    before, before_data = load_volume(before_path, canonical=True)
    after, after_data = load_volume(after_path, canonical=True)
    require_same_grid(before, after)
    if not np.isfinite(before_data).all() or not np.isfinite(after_data).all():
        raise ValueError("Nonfinite image data cannot be rendered as valid visual evidence")
    axis = PLANES[plane]
    chosen = select_slices(before_data, axis, slices=slices, slice_range=slice_range, count=slice_count)
    display_window = limits(before_data, window)
    aspect = panel_aspect(before, axis)
    figure, axes = _figure(2, len(chosen), (len(chosen) * 3.4, 6.4))
    for row, (role, data) in enumerate((("Before", before_data), ("After", after_data))):
        for ax, index in zip(axes[row], chosen):
            ax.axis("off")
            ax.imshow(panel(data, axis, index), cmap="gray", vmin=display_window[0],
                      vmax=display_window[1], origin="lower", aspect=aspect, interpolation="nearest")
            ax.set_title(f"{role} | {plane} index={index}", color="white", fontsize=8)
    figure.suptitle("Before / after | matched slices and shared intensity window", color="white")
    parameters = {"plane": plane, "slices": chosen, "slice_count": slice_count,
                  "slice_range": slice_range, "rows": 2, "columns": len(chosen),
                  "row_roles": ["before", "after"], "window": list(display_window),
                  "intensity_rescaling": "none", "display_origin": "lower", "display_aspect": aspect,
                  "selection": "explicit" if slices is not None else
                  "explicit_range" if slice_range is not None else "before_nonzero_range"}
    metadata = provenance({"before": before_path, "after": after_path}, parameters)
    metadata["display_images"] = {"before": image_record(before_path, before),
                                  "after": image_record(after_path, after)}
    return _finish(figure, output, metadata)


def render_orthogonal(image_path, output, *, declared_modality=None, slice_count=5, window=None):
    """Source-only three-plane evidence; the declared modality is not a prediction."""
    if declared_modality is not None:
        declared_modality = declared_modality.strip()
        if not declared_modality:
            raise ValueError("Declared modality must not be empty")
    image, raw = load_volume(image_path, canonical=True)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite image data cannot be rendered as valid visual evidence")
    selections = {name: select_slices(raw, axis, count=slice_count) for name, axis in PLANES.items()}
    columns = max(map(len, selections.values()))
    display_window = limits(raw, window)
    aspects = {name: panel_aspect(image, axis) for name, axis in PLANES.items()}
    figure, axes = _figure(3, columns, (columns * 3.4, 10))
    for ax in axes.flat:
        ax.axis("off")
    for row, (name, axis) in enumerate(PLANES.items()):
        for ax, index in zip(axes[row], selections[name]):
            ax.imshow(panel(raw, axis, index), cmap="gray", vmin=display_window[0],
                      vmax=display_window[1], origin="lower", aspect=aspects[name], interpolation="nearest")
            ax.set_title(f"{name} index={index}", color="white", fontsize=9)
    figure.suptitle(f"Source image | declared modality: {declared_modality or 'unspecified'} | review pending", color="white")
    metadata = provenance({"image": image_path}, {
        "declared_modality": declared_modality, "views": list(PLANES), "slices": selections,
        "slice_count_per_plane": slice_count, "rows": 3, "columns": columns,
        "selection": "source_nonzero_range", "window": list(display_window),
        "display_origin": "lower", "display_aspect": aspects})
    metadata.update(declared_modality=declared_modality, modality_assessment="indeterminate",
                    image_quality_assessment="indeterminate", view_type="source_orthogonal",
                    display_image=image_record(image_path, image),
                    original_image=image_record(image_path, load_volume(image_path)[0]))
    return _finish(figure, output, metadata)


def render_contours(image_path, labels_path, label, output, **options):
    return _volume_montage(image_path, output, labels_path=labels_path, label=label, **options)


def render_registration(fixed_path, registered_path, output, *, moving_path=None):
    fixed_image, fixed = load_volume(fixed_path, canonical=True)
    registered_image, registered = load_volume(registered_path, canonical=True)
    require_same_grid(fixed_image, registered_image)
    if not np.isfinite(fixed).all() or not np.isfinite(registered).all():
        raise ValueError("Nonfinite fixed/registered image")
    moving, baseline_reason = None, "No baseline supplied"
    if moving_path is not None:
        moving_image, candidate = load_volume(moving_path, canonical=True)
        if same_grid(fixed_image, moving_image) and np.isfinite(candidate).all():
            moving, baseline_reason = candidate, None
        else:
            baseline_reason = "Baseline grid mismatch or nonfinite data; explicit baseline preparation required"
    windows = {"fixed": limits(fixed), "registered": limits(registered)}
    if moving is not None:
        windows["moving"] = limits(moving)
    figure, axes = _figure(3, 6, (20, 11))
    selections = {}
    for row, (name, axis) in enumerate(PLANES.items()):
        chosen = select_slices(fixed, axis, count=3)
        selections[name] = chosen
        for ax in axes[row]:
            ax.axis("off")
        for col, index in enumerate(chosen):
            f = _scaled(panel(fixed, axis, index), windows["fixed"])
            for offset, data, key, caption in ((0, moving, "moving", "before"),
                                                (3, registered, "registered", "after")):
                ax = axes[row, col + offset]
                if data is None:
                    ax.text(.5, .5, "Before unavailable\nSee metadata", ha="center", color="white")
                else:
                    other = _scaled(panel(data, axis, index), windows[key])
                    ax.imshow(np.stack((f, other, np.zeros_like(f)), axis=-1),
                              origin="lower", aspect=panel_aspect(fixed_image, axis), interpolation="nearest")
                ax.set_title(f"{caption} | {name} index={index}", color="white", fontsize=9)
    figure.suptitle("Registration | separate before / after | fixed=red, other=green, overlap=yellow", color="white")
    metadata = provenance({"fixed": fixed_path, "moving": moving_path, "registered": registered_path},
                          {"slices": selections, "windows": windows,
                           "legend": {"fixed": "red", "moving_or_registered": "green"},
                           "layout": "before_left_after_right", "display_origin": "lower"})
    metadata.update(baseline_unavailable_reason=baseline_reason,
                    display_image=image_record(fixed_path, fixed_image),
                    original_image=image_record(fixed_path, load_volume(fixed_path)[0]))
    return _finish(figure, output, metadata)


def _scaled(data, window):
    return np.clip((data - window[0]) / (window[1] - window[0]), 0, 1)


def render_surface_contours(image_path, output, *, inner=(), outer=(),
                            surface_space="freesurfer-tkr", surface_reference=None,
                            title="Cortical surface QC", slices=None, slice_range=None,
                            slice_count=20, rows=4, columns=5, window=None):
    if not inner and not outer:
        raise ValueError("At least one inner or outer surface is required")
    original, _ = load_volume(image_path)
    image, raw = load_volume(image_path, canonical=True)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite image data")
    meshes = [(*mesh_voxels(path, original, image, surface_space, surface_reference), color)
              for paths, color in ((inner, "#ff3030"), (outer, "#3080ff")) for path in paths]
    # Overview from original-image support also includes entirely missing surface regions.
    chosen = select_slices(raw, 2, slices=slices, slice_range=slice_range, count=slice_count)
    if len(chosen) > rows * columns:
        raise ValueError("Selected slices exceed layout capacity")
    display_window = limits(raw, window)
    aspect = panel_aspect(image, 2)
    figure, axes = _figure(rows, columns, (columns * 3.4, rows * 3.2))
    for ax in axes.flat:
        ax.axis("off")
    for ax, z in zip(axes.flat, chosen):
        ax.imshow(panel(raw, 2, z), cmap="gray", vmin=display_window[0], vmax=display_window[1],
                  interpolation="nearest", origin="lower", aspect=aspect)
        for vertices, faces, color in meshes:
            lines = segments(vertices, faces, z)
            if len(lines):
                ax.add_collection(LineCollection(lines, colors=color, linewidths=.65))
        ax.set_xlim(-.5, raw.shape[0] - .5)
        ax.set_ylim(-.5, raw.shape[1] - .5)
        ax.set_title(f"axial index={z}", color="white", fontsize=8)
    figure.suptitle(title + " | red=white, blue=pial", color="white")
    inputs = {"image": image_path, "surface_reference": surface_reference}
    inputs.update({f"inner_{i}": p for i, p in enumerate(inner)})
    inputs.update({f"outer_{i}": p for i, p in enumerate(outer)})
    metadata = provenance(inputs, {"surface_space": surface_space, "surface_unit": "mm",
                          "plane": "axial", "slices": chosen, "slice_range": slice_range,
                          "selection": "explicit" if slices is not None else
                          "explicit_range" if slice_range is not None else "source_nonzero_range",
                          "window": list(display_window), "rows": rows, "columns": columns,
                          "title": title, "display_aspect": aspect, "display_origin": "lower",
                          "legend": {"white": "#ff3030", "pial": "#3080ff"}})
    metadata.update(display_image=image_record(image_path, image),
                    original_image=image_record(image_path, original))
    return _finish(figure, output, metadata)


SURFACE_VIEWS = ("lateral", "medial", "anterior", "posterior", "dorsal", "ventral")
SURFACE_ZOOM = .9


def _surface_pixels(panels, surface_engine, cmap, color_range, zero_transparent):
    """Yield independent square RGB viewports, reusing only an unchanged scene."""
    from itertools import groupby
    from surfplot import Plot

    if surface_engine not in ('reference', 'reuse'):
        raise ValueError('Unknown surface rendering engine')
    for panel_spec in panels:
        if panel_spec['hemisphere'] not in ('lh', 'rh') or panel_spec['view'] not in SURFACE_VIEWS:
            raise ValueError('Unknown surface hemisphere or view')

    def scene(panel_spec):
        return panel_spec['hemisphere'], id(panel_spec['mesh']), 'values' in panel_spec, id(panel_spec.get('values'))

    groups = (group for _, group in groupby(panels, scene)) if surface_engine == 'reuse' else ([item] for item in panels)
    for group in groups:
        first, *rest = group
        mesh, hemi = first['mesh'], first['hemisphere']
        plot = Plot(surf_lh=mesh if hemi == 'lh' else None, surf_rh=mesh if hemi == 'rh' else None,
                    layout='row', views=[first['view']], mirror_views=False, size=(600, 600), zoom=SURFACE_ZOOM,
                    background=(0, 0, 0), brightness=.35)
        if 'values' in first:
            values = np.asarray(first['values'])
            if values.ndim != 1 or len(values) != mesh.n_points or not np.isfinite(values).all():
                raise ValueError('Surface feature must be finite and match the mesh vertices')
            plot.add_layer({'left' if hemi == 'lh' else 'right': values}, cmap=cmap, color_range=color_range,
                           zero_transparent=zero_transparent, cbar=False)
        plotter = plot.render(offscreen=True)
        try:
            yield plotter.to_numpy(transparent_bg=False)
            if rest:
                from surfplot.surf import orientations
                from surfplot.plotting import _set_layout

                views, _ = _set_layout(plot.surfaces.get('left'), plot.surfaces.get('right'), 'row',
                                       [item['view'] for item in rest], mirror=False)
                renderer = plotter.renderers[0][0].VTKObject
                actors = list(renderer.GetActors())
                for view in views:
                    for actor in actors:
                        actor.SetOrientation(*orientations[view])
                    renderer.SetActiveCamera(renderer.MakeCamera())
                    renderer.ResetCamera()
                    camera = renderer.GetActiveCamera()
                    camera.ParallelProjectionOn()
                    camera.Zoom(SURFACE_ZOOM)
                    yield plotter.to_numpy(transparent_bg=False)
        finally:
            plotter.close()


def build_surface_montage(panels, *, title, columns=3, cmap="viridis",
                          color_range=None, zero_transparent=False, colorbar_label=None, surface_engine='reference'):
    from contextlib import closing
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if not panels or type(columns) is not int or columns < 1:
        raise ValueError("Surface panels and a positive column count are required")
    if colorbar_label is not None and color_range is None:
        raise ValueError("A shared colorbar requires an explicit color range")
    rows = (len(panels) + columns - 1) // columns
    title_lines = max(len(panel.get("title", "").splitlines()) for panel in panels)
    header = 1.0 + .15 * max(0, title_lines - 1)
    height = rows * 3.4 + header + .6
    figure, axes = _figure(rows, columns, (columns * 3.6, height))
    figure.subplots_adjust(left=.02, right=.98, bottom=1.0 / height if colorbar_label else .02,
                           top=1 - header / height, hspace=.18, wspace=.05)
    for axis in axes.flat:
        axis.set_axis_off()
    records = []
    with closing(_surface_pixels(panels, surface_engine, cmap, color_range, zero_transparent)) as images:
        for index, (panel_spec, pixels) in enumerate(zip(panels, images, strict=True)):
            hemisphere, view = panel_spec["hemisphere"], panel_spec["view"]
            row, column = divmod(index, columns)
            axis = axes[row, column]
            axis.imshow(pixels)
            panel_title = panel_spec.get("title", f"{hemisphere} {view}")
            axis.set_title(panel_title, color="white", fontsize=10)
            records.append({"hemisphere": hemisphere, "view": view, "row": row,
                            "column": column, "title": panel_title})
    if colorbar_label is not None:
        colorbar_axis = figure.add_axes((.25, .48 / height, .5, .12 / height))
        colorbar = figure.colorbar(ScalarMappable(norm=Normalize(*color_range), cmap=cmap),
                                  cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label(colorbar_label, color="white", fontsize=9)
        colorbar.ax.tick_params(colors="white", labelsize=8)
    figure.suptitle(title, color="white", fontsize=13)
    parameters = {"renderer": "surfplot", "views": list(dict.fromkeys(panel["view"] for panel in panels)),
                  "mirror_views": False, "brightness": .35, "panel_size": [600, 600], "zoom": SURFACE_ZOOM,
                  "camera_fit": "ResetCamera per view, parallel projection, zoom-out margin",
                  "layout": "independent square viewports", "rows": rows, "columns": columns,
                  "panels": records, "title": title, "zero_transparent": zero_transparent,
                  "cmap": cmap, "color_range": list(color_range) if color_range is not None else None,
                  "colorbar_label": colorbar_label}
    return figure, parameters


def render_surface(output, *, lh=None, rh=None, title="Cortical surface multi-view QC", surface_engine='reference'):
    if lh is None and rh is None:
        raise ValueError("At least one hemisphere surface is required")
    from brainspace.mesh.mesh_creation import build_polydata
    loaded = {}
    for name, path in (("lh", lh), ("rh", rh)):
        if path is not None:
            vertices, faces = read_surface(path)
            validate_mesh(vertices, faces)
            loaded[name] = build_polydata(vertices, faces)
    panels = [{"hemisphere": hemi, "mesh": mesh, "view": view}
              for hemi, mesh in loaded.items() for view in SURFACE_VIEWS]
    figure, parameters = build_surface_montage(panels, title=title, surface_engine=surface_engine)
    return _finish(figure, output, provenance({"lh": lh, "rh": rh}, parameters), tight=False)

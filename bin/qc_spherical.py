"""Render native, registered and template spheres with a shared feature scale."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--surface-engine', choices=('reference', 'reuse'), default='reference')
    for option in ("native", "registered", "template", "feature", "template-feature", "hemi", "output"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from brainspace.mesh.mesh_creation import build_polydata
    from neuroqc.evidence import provenance, write_json
    from neuroqc.rendering import build_surface_montage

    subject_values = nib.freesurfer.read_morph_data(args.feature)
    reference_values = nib.freesurfer.read_morph_data(args.template_feature)
    views = ["lateral", "medial", "dorsal", "ventral", "anterior", "posterior"]
    panels, records = [], []
    native_faces = None
    for title, path, values in (("Native", args.native, subject_values),
                                ("Registered", args.registered, subject_values),
                                ("Fixed fsaverage6", args.template, reference_values)):
        coordinates, faces = nib.freesurfer.read_geometry(path)
        if len(values) != len(coordinates) or not np.isfinite(values).all() or not np.isfinite(coordinates).all():
            raise ValueError("Spherical feature/vertex correspondence or finite-value check failed")
        if title == "Native":
            native_faces = faces
        elif title == "Registered" and not np.array_equal(faces, native_faces):
            raise ValueError("Registered sphere changed native vertex indexing")
        mesh = build_polydata(coordinates, faces)
        panels.extend({"hemisphere": args.hemi, "mesh": mesh, "values": values,
                       "view": view, "title": title + "\n" + view} for view in views)
        records.append({"role": title, "surface": path, "vertex_count": len(coordinates)})
    limit = float(np.percentile(np.abs(np.concatenate((subject_values, reference_values))), 99))
    if limit <= 0:
        raise ValueError("A constant feature cannot demonstrate spherical registration")
    figure, layout = build_surface_montage(panels, title=args.hemi + " spherical registration | review pending",
        columns=6, cmap="coolwarm", color_range=(-limit, limit), zero_transparent=False,
        colorbar_label="Raw FreeSurfer feature; shared scale",
        **({'surface_engine': args.surface_engine} if args.surface_engine != 'reference' else {}))
    figure.savefig(args.output, dpi=140, facecolor=figure.get_facecolor())
    plt.close(figure)
    metadata = provenance({"native": args.native, "registered": args.registered, "template": args.template,
                           "feature": args.feature, "template_feature": args.template_feature},
                          {"views": views, "panels": records, "render_layout": layout,
                           "color_range": [-limit, limit], "scaling": "shared_symmetric_absolute_p99",
                           "independent_normalization": False, "visual_review": "pending"})
    write_json(str(args.output) + ".json", metadata)


if __name__ == "__main__":
    main()

"""Optional real QC rendering and geometry checks using configured interpreters."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RENDER = os.environ.get("FASTPREP_TEST_QC_PYTHON")
GEOMETRY = os.environ.get("FASTPREP_TEST_GEOMETRY_PYTHON")


@unittest.skipUnless(RENDER and GEOMETRY, "Set FASTPREP_TEST_QC_PYTHON and FASTPREP_TEST_GEOMETRY_PYTHON")
class QCIntegrationTests(unittest.TestCase):
    def test_real_renderers_and_full_cortical_pair(self):
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory(prefix="fastprep-qc-") as directory:
            root = Path(directory)
            environment = {**os.environ, "PYTHONPATH": str(ROOT / "vendor"),
                           "MPLBACKEND": "Agg", "QT_QPA_PLATFORM": "offscreen", "OMP_NUM_THREADS": "1"}
            xyz, faces = nib.freesurfer.read_geometry(str(ROOT / "model/sugar/source/utils/auxi_data/fsaverage2.sphere"))
            white, pial = root / "lh.white", root / "lh.pial"
            nib.freesurfer.write_geometry(white, xyz * .08, faces)
            nib.freesurfer.write_geometry(pial, xyz * .10, faces)
            nib.freesurfer.write_morph_data(root / "lh.thickness", np.full(len(xyz), 2, dtype=np.float32))
            nib.freesurfer.write_morph_data(root / "lh.sulc", (xyz[:, 0] / 100).astype(np.float32))
            label = root / "lh.cortex.label"
            label.write_text("#!ascii label, synthetic test\n" + str(len(xyz)) + "\n" +
                             "".join(f"{i} 0 0 0 0\n" for i in range(len(xyz))))
            axes = np.indices((32, 32, 32), dtype=np.float32)
            radius = np.sqrt(np.sum((axes - 16) ** 2, axis=0))
            values = np.where(radius < 13, 120 - radius * 4, 0).astype(np.float32)
            affine = np.array([[-1, 0, 0, 16], [0, 0, 1, -16], [0, -1, 0, 16], [0, 0, 0, 1]], float)
            image, after = root / "before.mgz", root / "after.mgz"
            nib.save(nib.MGHImage(values, affine), image)
            nib.save(nib.MGHImage(values * 1.1, affine), after)
            ribbon = np.where((radius >= 8) & (radius < 10), 3, 0).astype(np.int16)
            for name in ("ribbon.mgz", "aseg.mgz"):
                nib.save(nib.MGHImage(ribbon, affine), root / name)

            def run(command):
                result = subprocess.run(list(map(str, command)), env=environment, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, " ".join(map(str, command)) + "\n" + result.stderr[-3000:])

            for operation, arguments, name in (
                ("comparison", ["--before", image, "--after", after, "--plane", "axial"], "bias.png"),
                ("surface-contours", ["--image", image, "--inner", white, "--outer", pial,
                    "--surface-space", "freesurfer-tkr", "--surface-reference", image], "pial.png"),
                ("surface", ["--lh", pial], "surface.png")):
                run([RENDER, "-B", "-m", "neuroqc", "render", operation, *arguments, "--output", root / name])
                self.assertGreater((root / name).stat().st_size, 1000)
                self.assertTrue((root / (name + ".json")).is_file())
            run([RENDER, "-B", ROOT / "bin/qc_spherical.py", "--native", white, "--registered", white,
                 "--template", pial, "--feature", root / "lh.sulc", "--template-feature", root / "lh.sulc",
                 "--hemi", "lh", "--output", root / "sphere.png"])
            run([GEOMETRY, "-B", ROOT / "bin/qc_surface_pair.py", "--inner", white, "--outer", pial,
                 "--coordinate-unit", "mm", "--geometry-python", RENDER,
                 "--thickness", root / "lh.thickness", "--cortex-label", label,
                 "--ribbon", root / "ribbon.mgz", "--aseg", root / "aseg.mgz", "--hemi", "lh",
                 "--output", root / "pair.json"])
            record = json.loads((root / "pair.json").read_text())
            self.assertEqual(record["white_outside_pial_vertices"], 0)
            self.assertEqual(record["negative_thickness_vertices"], 0)
            self.assertEqual(record["cortical_triangle_pair_intersections"], 0)
            self.assertEqual(record["status"], "unknown")
            self.assertEqual(record["vertex_count"], len(xyz))


if __name__ == "__main__":
    unittest.main()

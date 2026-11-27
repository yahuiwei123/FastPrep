"""Optional real FreeSurfer interface test, enabled by FASTPREP_TEST_FREESURFER."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FS_HOME = os.environ.get("FASTPREP_TEST_FREESURFER")


@unittest.skipUnless(FS_HOME, "Set FASTPREP_TEST_FREESURFER to exercise installed FreeSurfer binaries")
class FreeSurferInterfaceTests(unittest.TestCase):
    def test_final_surface_morphometry_commands(self):
        import numpy as np
        from nibabel.freesurfer.io import read_geometry, read_morph_data, write_geometry

        vertices, faces = read_geometry(str(ROOT / "model/sugar/source/utils/auxi_data/fsaverage2.sphere"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            white, pial = root / "lh.white", root / "lh.pial"
            write_geometry(white, vertices * .2, faces)
            write_geometry(pial, vertices * .22, faces)
            executable = Path(FS_HOME) / "bin/mris_place_surface"
            environment = {**os.environ, "FREESURFER_HOME": FS_HOME, "OMP_NUM_THREADS": "1",
                           "FS_LICENSE": os.environ.get("FS_LICENSE", str(Path(FS_HOME) / "license.txt"))}
            commands = [(["--curv-map", pial, "2", "10", root / "lh.curv.pial"], "lh.curv.pial"),
                        (["--area-map", pial, root / "lh.area.pial"], "lh.area.pial"),
                        (["--thickness", white, pial, "20", "5", root / "lh.thickness"], "lh.thickness")]
            for arguments, output in commands:
                result = subprocess.run([str(executable), *map(str, arguments)], env=environment,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr[-2000:] + result.stdout[-2000:])
                values = read_morph_data(root / output)
                self.assertEqual(len(values), len(vertices))
                self.assertTrue(np.isfinite(values).all())
            thickness = read_morph_data(root / "lh.thickness")
            self.assertTrue(np.all(thickness >= 0))
            self.assertGreater(float(np.median(thickness)), 1)
            self.assertLess(float(np.median(thickness)), 3)


if __name__ == "__main__":
    unittest.main()

# Portable model bundle dependencies

The environments are separate because the model stacks have incompatible Python and numerical-library requirements. The YAML and requirements files are portable dependency recipes, not lockfiles; they contain no content digests. Create only the environment needed for a command, from the repository root, for example:

```sh
conda env create -p "$PWD/envs/synthseg" -f envs/synthseg.yml
```

The listed versions were checked in existing scientific environments on this host. They do not claim that every wheel is available for every Linux distribution, accelerator, or Python build.

| Profile | Observed runtime | Main packages | Notes |
| --- | --- | --- | --- |
| `synthseg` | Python 3.9.25 | TensorFlow 2.15.1, Keras 2.15.0, NumPy 1.26.4, SciPy 1.13.1, NiBabel 5.3.3 | FastPrep now defaults to `cuda:0`; pre/postprocessing stays on CPU. GPU requests require compatible CUDA/cuDNN libraries and fail rather than silently falling back. |
| `sugar` | Python 3.9.25 | PyTorch 2.8.0, PyG 2.2.0, torch-scatter 2.1.2, PyTorch3D 0.7.9, NumPy 1.26.4, SciPy 1.13.1, NiBabel 5.3.3 | The verified GPU build used CUDA 12.8 and H100 `sm_90`; torch-scatter and PyTorch3D are native extensions. Select builds matching the destination GPU/driver, or validate a CPU build before use. |
| `hdbet` | Python 3.12.13 | PyTorch 2.5.1, nnU-Net v2 2.8.1, SimpleITK 2.5.5, NiBabel 5.4.2, NumPy 2.4.6, SciPy 1.17.1 | The checked host had a CUDA 12.1 PyTorch build. Install a PyTorch build appropriate to the destination; `predict-mask` supports CPU and CUDA. |
| `helper` | Python 3.11.15 | NumPy 1.26.4, SciPy 1.15.3, NiBabel 5.2.1, SimpleITK 2.2.1, scikit-learn 1.9.0 | Controls the pipeline and the explicit CPU Gaussian BFC alternative. |
| `bfc` | Python 3.11; verified PyTorch 2.8.0+cu128 | NumPy, SciPy, NiBabel, SimpleITK, PyTorch; scikit-learn for standalone Gaussian | New profile for default RBF. NVIDIA L20 CUDA execution verified locally; install a CUDA-enabled PyTorch build matching the target driver. No CPU fallback. The new environment recipe itself was not installed from scratch. |

The listed scientific-environment package versions were queried directly from the existing `synthseg`, `sugar`, `macabrain`, and `mars` Python environments. They were not installed or modified during this work. `pytest` is absent from the `mars` Python 3.11 environment; the bundle contract tests use the Python standard library `unittest` and need no pytest installation.

`envs/sugar-requirements.txt` records Python package versions but cannot express a binary wheel compatible with every CUDA stack. Likewise, PyTorch wheel selection depends on the target platform. Treat these two native model profiles as versioned recipes that require platform-appropriate wheel selection, not as a promise that a generic `pip install -r` will build accelerator extensions.

## Other runtime requirements

- Volume normalization defaults to FireANTs 1.5.0 (external package, included in `envs/bfc-requirements.txt`) with CUDA PyTorch. It uses `python.bfc` unless optional `python.fireants` selects a separate interpreter. The observed combination is Python 3.11, PyTorch 2.8.0+cu128 and NVIDIA L40. Missing optional fused operations use PyTorch on the same GPU; actual device and fused status are recorded. There is no CPU fallback. Existing environments must install the new dependency before preparing a run. The dependency recipe has not been installed from scratch.
- Both registration backends need ANTs `antsApplyTransforms` and a populated TemplateFlow cache. Only explicit `prepare.py --mni-backend ants` requires `antsRegistration` for the original affine/SyN method. The `templateflow` download client can be installed separately; compute nodes only read the T1w template and matching brain mask.
- FreeSurfer and FSL FLIRT are external command-line dependencies of the main pipeline. Set `fsl_dir` to the FSL installation root or export `FSLDIR`. Their installation recipes and license handling belong to the main runtime configuration. No FreeSurfer license file or credential is part of this model bundle.
- `neuroqc:measure-cortical-pair` is handled by the main QC worker and its configured `neuroqc` Python package. It is not a model-bundle command in `conf/models.json`.
- The `bin/tissue_bfc/` toolkit owns both corrected methods; `bin/run_tissue_bfc.py` and `bin/bfc_support.py` remain compatibility entry points. Both unchanged upstream sources are bundled. RBF is the default for T1/T2 and uses `python.bfc`; Gaussian is an explicit CPU alternative using `python.helper`. The adapter changes fitting support/statistics and field application; it is not an unmodified upstream CLI execution. Rebuild runtime with all seven Python profiles; do not reuse an old absolute-path runtime after changing methods. Details and standalone archive instructions are in `docs/BIAS_CORRECTION.md`.

## QC rendering and surface-pair geometry

The vendor QC renderers use `numpy`, `scipy`, `nibabel`, `matplotlib`, `surfplot`, and VTK. Surface-pair processing splits across two interpreters: `pair_python` runs the FCL/trimesh worker from the `geometry` profile, then `--geometry-python` launches a VTK-only worker using `render_python` from the `qc` profile. The FCL and VTK stacks therefore remain in separate environments.

The locally verified QC/render environment uses VTK 9.6.2 and surfplot 0.2.0 (`macapipe`). The locally verified geometry environment uses python-fcl/FCL 0.7.0.11, trimesh 4.12.2, and rtree 1.4.1 (`sugar-source`). These versions are recorded in the corresponding environment YAML files; `python-fcl` is installed by pip because the conda-forge package is unavailable. PyPI metadata access timed out, so availability of a matching target-machine wheel was not verified. For the Python 3.11 profile, pip must find a wheel matching the CPython 3.11 ABI, Linux CPU architecture, and compatible manylinux/glibc tag. If no compatible wheel is available and pip falls back to a source build, the target needs a compatible native FCL/libccd toolchain. The `mars` interpreter used for tests lacks these QC/geometry packages; the profiles were not installed or resolved during this task.

Create the project-local environments with Conda's explicit prefix option:

```sh
conda env create -p "$PWD/envs/qc" -f envs/qc.yml
conda env create -p "$PWD/envs/geometry" -f envs/geometry.yml
```

The verified `mars` Python 3.11 environment contains NumPy 1.26.4, SciPy 1.15.3, NiBabel 5.2.1, and Matplotlib 3.10.0. It does not contain VTK, surfplot, trimesh, rtree, or python-fcl, so QC rendering and the pair workers were not exercised in `mars`. VTK is a large native package. `python-fcl` wheel availability and installation on other Linux targets remain unverified; a source build may require a matching native FCL/libccd development stack.

Validation evidence for the synthetic BFC and BIDS selection tests is recorded in [VALIDATION.md](VALIDATION.md).

`tests/test_inputs_and_bias.py` uses only `unittest` and temporary, generated BIDS/MGH inputs. Its BFC integration test skips when the required scientific modules are absent; it does not import or refer to any source checkout or preserved `/tmp` probe.

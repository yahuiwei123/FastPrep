# Implementation lessons

- A collection output consists of its main manifest and its members. Forward both, keep their paths relative to the new result directory, and copy real command logs. A topology-only run cannot validate this native artifact contract.
- Removing integrity locks requires replacing the associated interfaces and schemas together. Retain anatomical identity and geometry checks; do not substitute fabricated digests or file timestamps.
- A structurally valid input and a generated QC report do not constitute visual approval. User-facing text and runtime metadata must preserve this distinction.
- Empty Python package initializers are valid resources. Require nonempty scientific data and weights while permitting empty `.py` package files.
- A portable source bundle and a prepared machine-local runtime are different artifacts. Recreate the runtime after relocation; refuse direct assembly into a populated subject directory.
- An executable script can exist and have execute permissions while its shebang interpreter is absent. Check that interpreter explicitly so missing FreeSurfer `tcsh` is reported before scheduling scientific work.
- Validate vendored QC tools through their actual CLI, not only Python imports or forwarding fixtures. An older renderer can import successfully while lacking a workflow-required command such as `render comparison`.
- Native and template surfaces require their own white/pial pair and anatomical reference. Never reuse native ribbon measurements as evidence for resampled fsaverage6 geometry.

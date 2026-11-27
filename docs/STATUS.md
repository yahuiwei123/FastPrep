# Current release task

Objective: deliver a portable standalone FastPrep project with tissue-guided bias correction, optional T2 pial refinement, and no project-added content locks.

Completed: scientific stages and model bundle; standalone input/runtime preparation; explicit 49-node Nextflow workflow; two-session topology execution and full cache reuse; 23 regression/integration tests; real FreeSurfer and QC interface checks; relocation to a directory containing a space; installation and usage documentation.

Release entry points: `README.md`, `main.nf`, `bin/prepare.py`, and `bin/package.py`. The package contains source, required model weights, QC tools and deployment recipes. Target-machine software configuration is generated locally and is not part of the release payload.

Validation boundary: no complete real-subject cortical reconstruction or clean-machine environment installation has been performed for this standalone release. Generated QC evidence is not anatomical approval.

Host prerequisite: this machine lacks `/bin/tcsh`, so the full native environment probe stops before `recon-all`. Preparation reports the missing interpreter explicitly; no system packages were installed during this task.

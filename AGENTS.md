# FastPrep standalone

- This project runs independently of BrainBoot and LangGraph.
- Do not add content digests, checksum locks, fake digest functions or credential files.
- Use configured scientific Python environments for scientific commands. The control scripts and standard-library tests support Python 3.11+.
- Never edit generated scientific results or QC decisions to simulate success.
- Retain input existence, subject/session identity, spatial geometry, finite-value and mesh correspondence checks.
- Tests and portable examples must not depend on the source machine or the original BrainBoot checkout.
- Preserve third-party licenses and citations; user FreeSurfer licenses belong outside the project.

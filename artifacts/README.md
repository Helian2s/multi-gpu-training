# Run artifacts

This directory is the local collection point for generated experiment outputs.
Its contents are ignored by Git except for this file.

Use the following layout for every remote or local run:

```text
artifacts/runs/EXP-NN/<run-id>/
├── manifest.json          # Provider, commit, image digest, hardware, versions, and timestamps
├── qualification/        # Mandatory pre-run qualification output
├── config/               # Fully resolved configuration used for the run
├── raw/                   # Machine-readable metrics and application logs
├── profiles/              # Nsight, PyTorch Profiler, NCCL, or DCGM artifacts
└── analysis/              # Derived tables and plots
```

Use the catalog's canonical ID for `EXP-NN` and an immutable run ID such as
`20260711T213000Z-run01`. Hardware, provider, and precision belong in
`manifest.json` rather than being inferred only from the run ID. Never overwrite
a completed run. Copy the complete run directory to durable project storage
before stopping a Runpod Pod or stopping/terminating an EC2 instance.

Small, reviewed tables or figures may later be copied into an experiment report
and committed deliberately. Checkpoints and raw profiler traces are never
committed.

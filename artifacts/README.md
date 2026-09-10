# Run artifacts

This directory holds local copies of generated run outputs. Everything except
this README is ignored by Git. The repository's portable reading material is
in the [experiment reports](../experiments/README.md) and
[history summaries](../docs/experiment_history/README.md).

## Where the recorded evidence lives

| Source | Canonical local mirror | Recorded contents |
| --- | --- | --- |
| AWS S3 artifact prefixes | `artifacts/runs/aws-s3-mirror/` | Qualification, PyTorch and pipeline runs, queue logs, metrics, profiler output |
| Runpod Pod storage, copied before deletion | `artifacts/runs/runpod-volume-mirror/` | A100 qualification, communication and synthetic parallelism runs, queue logs and metrics |

The July handoff records an external transfer archive containing both mirrors,
about 119 MB across 347 files, with a SHA-256 checksum. See
[HANDOFF.md](../HANDOFF.md) for its historical workstation path and restoration
instructions. The archive is not part of the clone or a public download.

A new checkout does not establish access to S3, the transfer archive, or any
local mirror. The macOS source review on 2026-09-10 had none of the raw mirrors.

## Layout emitted by the shared executors

```text
EXP-NN/<run-id>/
├── plan.json                       # Selected variants, image reference, output paths
├── manifest.json                   # Execution plan and controller status
├── raw/<variant-id>/
│   ├── rank_0.json                 # Per-rank metrics; additional ranks where applicable
│   ├── subprocess.log              # Worker/controller log
│   └── ...                         # Workload-specific traces or profiler summaries
└── metrics/variant_results.jsonl    # Worker results, including rank records
```

Qualification, queue logs, timestamps, and exit-status files are collected by
provider tooling and can live under separate qualification/queue prefixes.
Communication experiments use collector-specific layouts. The manifest alone
does not contain the complete hardware or provenance record: preserve the
corresponding qualification, configuration, image, and queue evidence together.

The AWS mirror retains the S3 key below `artifacts/`, for example:

```text
artifacts/runs/aws-s3-mirror/EXP-03/runs/<run-id>/
artifacts/runs/runpod-volume-mirror/EXP-14/<run-id>/
```

## Analyze and preserve

Restore mirrors before regenerating history. `make experiment-history` writes
tracked summaries and some interim reports; see [the script guide](../scripts/README.md)
before running it. EXP-12 and EXP-14 have executable table summarizers; the
other experiment analyzers remain templates.

Use unique run IDs and keep raw evidence immutable. Copy Runpod artifacts out
before Pod deletion; the current queue does not perform that transfer
automatically. Verify transferred inputs and artifacts before relying on them.

Keep model weights, datasets, checkpoints, raw profiler traces, and credentials
out of Git. Reviewed summary tables may be committed with reports. Historical
results remain subject to the [validation findings](../docs/validation-status.md).

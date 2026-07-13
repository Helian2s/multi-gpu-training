# Tests

Shared utilities are tested under `tests/common/`. Experiment-specific tests may
live inside the experiment directory when they are inseparable from that
experiment, or under `tests/experiments/<experiment-id>/` when centralized test
execution is more useful.

Local tests must cover configuration validation, deterministic preprocessing,
metric analysis, and failure handling without requiring an NVIDIA GPU. GPU
smoke and distributed correctness tests run in the appropriate project
container on the selected provider and record their environment like any other
run.

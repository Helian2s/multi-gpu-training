# Shared experiment infrastructure

`common/` contains behavior used by more than one experiment. Shared code must
remain framework-neutral where practical; framework-specific adapters should be
clearly separated.

The planned modules are:

- `qualification/`: provider, compute host, GPU, topology, version, permission,
  and smoke checks.
- `launch/`: stable wrappers for single-process, `torchrun`, and NeMo/Megatron
  launches.
- `telemetry/`: process, CPU, GPU, memory, power, NCCL, and timing collection.
- `analysis/`: metric loading, aggregation, confidence intervals, and common
  plots.
- `schemas/`: validation contracts for manifests, metrics, and experiment YAML.

Create a module only when the first experiment needs it. Do not add empty Python
packages or speculative abstractions merely to fill this tree. Once an
interface exists, add its tests under `tests/common/`.

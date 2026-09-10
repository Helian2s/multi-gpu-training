# Local checks and test coverage

The local suite uses Python's `unittest` and runs without an NVIDIA GPU,
prepared model inputs, or cloud credentials. Tests are in this directory as
`test_*.py`; there are no separate `tests/common/` or `tests/experiments/`
packages in the current implementation.

## Run the checks

From the repository root, with Python 3.12 or later, Git, and Make installed:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install PyYAML==6.0.3
make check PYTHON=.venv/bin/python
```

For an existing environment, set `PYTHON` to its interpreter. The Makefile's
default is `python3`, so preparing `.venv` alone does not make `make check` use
it. `ModuleNotFoundError: No module named 'yaml'` means PyYAML is missing from
the selected interpreter.

`make check` compiles Python sources, checks the scaffolder CLI, discovers unit
tests, validates repository documents/configuration, and runs `git diff --check`.

## Current coverage

| Test area | Coverage |
| --- | --- |
| Input preparation | Deterministic binary token/index output, empty records, file inventory, and checksum verification |
| Shared runner | Variant/run-unit selection, plan serialization, image reference requirements, and FP16 parameter-precision selection |
| AWS launch and queues | Request construction, confirmation handling, device masks, queue ordering, generated scripts, and lifecycle configuration |
| AWS operator helpers | Resource tags, run IDs, and container naming |
| Runpod queues | Queue/configuration checks, command generation, image/storage requirements, and GPU-count constraints |

The 2026-09-10 review passed **41 tests** with Python 3.12 on macOS ARM64.

## What passing does not establish

The suite does not run CUDA kernels, test the full Qwen training objective,
compare distributed outputs/gradients with a single-rank reference, validate
measured speedups, or check live cloud readiness. Shell tests inspect generated
commands; they do not provision resources.

The [validation review](../docs/validation-status.md) records the correctness
and analysis gaps found beyond this coverage. GPU qualification and measured
reruns require the selected container on the assigned provider, with fresh
hardware/runtime evidence and authorized GPU execution.

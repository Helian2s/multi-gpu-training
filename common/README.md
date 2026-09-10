# Shared runtime

The shared code turns an experiment specification into worker processes and
recorded outputs. Provider provisioning, storage transfers, and shutdown belong
under [infra/](../infra/README.md).

## Implemented modules

| Module | Responsibility |
| --- | --- |
| [experiment_runner.py](experiment_runner.py) | Load YAML, select variants/run units, resolve output paths, validate image references, and produce dry-run or execution plans |
| [pytorch_executor.py](pytorch_executor.py) | Run Qwen training steps, GEMM and attention microbenchmarks, PyTorch profiling, and controlled fault cases |
| [megatron_executor.py](megatron_executor.py) | Run synthetic TP/SP, CP, pipeline, and TP×DP workloads using custom PyTorch distributed operations in the NeMo/Megatron image |
| [qualification/aws_gpu_smoke.py](qualification/aws_gpu_smoke.py) | Check the allocated AWS GPU runtime and record qualification output |
| [qualification/runpod_gpu_smoke.py](qualification/runpod_gpu_smoke.py) | Check Runpod GPU identity, runtime, and topology |

## Execution flow

1. An experiment's `run_expNN.py` calls the shared runner with its ID,
   configuration path, and executor.
2. The runner resolves selected variants and an immutable image reference.
   Dry-run mode prints a plan without starting GPU workers.
3. The executor launches each variant in a subprocess. Multi-rank variants use
   `torch.distributed.run` on one host, with the configured device mask.
4. Workers write rank JSON, combined JSONL metrics, and logs. The controller
   updates the execution manifest.

EXP-01 and EXP-10 use shell collection scripts for CUDA/NCCL communication
measurements rather than a training executor.

## Interpretation limits

An execution manifest marked `completed` means the worker loop returned
successfully. The current executors do not evaluate the full expected-result
contract or implement configured repetitions. The synthetic parallelism code
is not a full Megatron model-training integration.

The [validation review](../docs/validation-status.md) documents label alignment,
sample partitioning, and distributed-gradient issues. Consult it before adding
performance conclusions or promoting reports. Existing regression coverage is
listed in [the test guide](../tests/README.md).

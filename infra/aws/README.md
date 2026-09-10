# AWS execution tooling

The AWS phase ran EXP-01–09 in the PyTorch image and EXP-12 in the
NeMo/Megatron image. It used G7e hosts with RTX PRO 6000 Blackwell Server
Edition GPUs in `us-west-2`. EXP-12 is the accepted exception to the usual
Runpod placement of the synthetic parallelism experiments.

## Profiles and queues

| Profile | Recorded instance type | Physical GPUs | Queue |
| --- | --- | ---: | --- |
| AWS-A1 | `g7e.2xlarge` | 1 | AWS-A1-PyTorch: EXP-03–06 |
| AWS-A2 | `g7e.12xlarge` | 2 | AWS-A2-PyTorch: EXP-01/02/07/08/09; AWS-A2-Megatron: EXP-12 |

`A2V1` and `A2V2` are run-unit suffixes for one or two visible GPUs on the same
billed AWS-A2 host. The current catalog contains no four-GPU AWS run. Provider,
GPU-family, and scale changes follow [the project decisions](../../PROJECT_DECISIONS.md).

## Modules

| File | Responsibility |
| --- | --- |
| [exp01_preflight.py](exp01_preflight.py) | Read-only identity, quota, offerings, price, registry, storage, and configuration checks |
| [exp01_launch.py](exp01_launch.py) | Qualification launch requests, cache attachment, bootstrap scripts, confirmation, and lifetime guards |
| [exp01_ops.py](exp01_ops.py) | SSM commands/shells, status, logs, monitoring, and artifact inspection |
| [a1_queue.py](a1_queue.py) | Run the A1 queue through SSM on an already-running selected host |
| [a2_queue.py](a2_queue.py) | Plan, launch, or run the A2 PyTorch and Megatron queues |

Queue definitions are [a1_experiment_queue.yaml](a1_experiment_queue.yaml),
[a2_experiment_queue.yaml](a2_experiment_queue.yaml), and
[a2_megatron_queue.yaml](a2_megatron_queue.yaml). Their IDs and digests identify
previous execution configurations; they are not live readiness guarantees.

## Inspect without launching

Local plan and script generation, from the repository root:

```bash
make aws-a1-queue-plan PYTHON=.venv/bin/python
make aws-a2-queue-plan PYTHON=.venv/bin/python
make aws-a2-megatron-queue-plan PYTHON=.venv/bin/python
make aws-a2-queue-script PYTHON=.venv/bin/python RUN_ID=review-a2
```

The following contact AWS, require credentials, and perform read-only checks
or EC2 API dry-runs rather than creating instances:

```bash
make aws-exp01-preflight PYTHON=.venv/bin/python
make aws-a1-preflight PYTHON=.venv/bin/python
make aws-a2-queue-launch-dry-run PYTHON=.venv/bin/python RUN_ID=review-a2
make aws-a2-megatron-queue-launch-dry-run PYTHON=.venv/bin/python RUN_ID=review-pipeline
```

Launch commands require explicit approval and the exact confirmation phrase
shown by the launch tooling. Running a queue on an existing host also consumes
GPU time. Use `make help` to inspect the corresponding launch/run entry points.

## Storage and shutdown behavior

The execution design uses private ECR for images, S3 for durable inputs/results,
and retained EBS caches for Docker layers and prepared inputs. Cache volumes
must match the selected Availability Zone. Instance-store data is temporary.

A2 queues upload artifacts after each run unit and schedule a hard shutdown.
The checked-in A2 configurations set a 300-minute hard deadline. On normal
completion, a 15-minute inspection hold precedes stopping, subject to that
deadline. On failure they retain
the host for inspection until the already-scheduled deadline. The A1 and
qualification wrappers have their own settings; inspect the generated script
and selected configuration rather than assuming identical behavior.

Stopping EC2 ends compute execution but can leave billable storage. Instance
termination, volume cleanup, and registry cleanup are distinct operations.
Artifact upload must be verified before removing the resources that hold it.

## Inspect an existing host

These operator commands require current AWS access and an identified project
host; they do not create one:

```bash
make aws-exp01-status PYTHON=.venv/bin/python
make aws-exp01-logs PYTHON=.venv/bin/python
make aws-exp01-artifacts PYTHON=.venv/bin/python
make aws-exp01-host-command PYTHON=.venv/bin/python CMD='nvidia-smi'
```

Pass `INSTANCE_ID=i-...` when selecting among hosts. Container helpers also
accept `RUN_ID=...`. Interactive host/container shells require the local
Session Manager plugin, checked by `make aws-ssm-plugin-check`. A1 has matching
`aws-a1-*` operator targets.

## Historical state and rerun requirements

The [July handoff](../../HANDOFF.md) records no remaining project instances or
cache volumes after cleanup, and deletion of all images from the two project
ECR repositories on 2026-07-16. Treat that as a dated cleanup record, not a
current API observation. Old volume IDs and image digests in configuration
must be reconciled before a new launch.

A future run needs fresh identity/quota/capacity/price checks, suitable storage,
a rebuilt and published image, and host qualification. It also needs the
correctness repairs described in [validation status](../../docs/validation-status.md)
when rerunning affected training or parallelism comparisons.

Exact historical resource, image, and operational details remain in
[TOOLING.md](../TOOLING.md), the [reports](../../experiments/README.md), and
[the operational timeline](../../docs/experiment_history/operational_timeline.md).

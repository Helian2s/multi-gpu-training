# Cloud execution infrastructure

Provider tooling manages the GPU host, storage, connection, and lifecycle.
Experiment logic lives in [common/](../common/README.md) and
[experiments/](../experiments/README.md), without provider API calls.

## Execution queues used

| Queue | Host profile | Experiments |
| --- | --- | --- |
| AWS-A1-PyTorch | One-GPU AWS G7e | EXP-03–06 |
| AWS-A2-PyTorch | Two-GPU AWS G7e, with 1/2-visible-GPU phases | EXP-01, EXP-02, EXP-07–09 |
| AWS-A2-Megatron | Two-GPU AWS G7e, with 1/2-visible-GPU phases | EXP-12 |
| RUNPOD-A2-Megatron | Two A100 SXM GPUs, with 1/2-visible-GPU phases | EXP-10, EXP-11, EXP-13 |
| RUNPOD-A4-Megatron | Four A100 SXM GPUs | EXP-14 |

Each run stays on one physical host and each experiment has one provider.
Visibility masks change the worker count, not the billed resource. Hardware
and image differences are retained in reports rather than pooled as equivalent
training environments.

## Implemented lifecycle

| Operation | AWS | Runpod |
| --- | --- | --- |
| Plan and configure | YAML queues, local plans, generated host scripts | YAML queues, local plans, generated container scripts and Pod creation commands |
| Connect and execute | EC2 launch wrappers and SSM operator/queue commands | Operator-created Pod and SSH; container startup keeps it available for queue execution |
| Qualify | GPU/runtime smoke plus EXP-01 communication evidence | GPU/runtime/topology smoke plus EXP-10 communication evidence |
| Stage and collect | S3 input staging and artifact uploads; EBS caches | Pod storage and operator-managed artifact copy-out |
| Limit lifetime | Scheduled shutdown; success/failure behavior depends on launch mode and queue settings | Creation command includes a termination deadline; early stop and artifact preservation remain operator responsibilities |

The Runpod script's `stop_on_success` field does not itself stop a Pod. Do not
treat a generated plan or a zero queue exit status as proof that artifacts are
durable or billing has ended. See [the validation review](../docs/validation-status.md).

## Local inspection

With the [local environment](../tests/README.md) installed:

```bash
make aws-a1-queue-plan PYTHON=.venv/bin/python
make aws-a2-queue-plan PYTHON=.venv/bin/python
make aws-a2-megatron-queue-plan PYTHON=.venv/bin/python
make runpod-a2-megatron-queue-plan PYTHON=.venv/bin/python
make runpod-a4-megatron-queue-plan PYTHON=.venv/bin/python
```

These targets inspect configuration locally. Provider preflight and launch
API dry-runs are separate operations and require current credentials.

## Running again

The checked-in image digests, resource IDs, and provider observations describe
historical runs. The [July handoff](../HANDOFF.md) records AWS image/cache
cleanup and a Runpod image that predates the committed NCCL cleanup fix.
Recheck account access, capacity, price, storage, images, and qualification
before another execution. Billable resource changes and image publication
require explicit authorization under [the working agreements](../AGENTS.md).

Use [the AWS guide](aws/README.md) or [the Runpod guide](runpod/README.md) for
provider-specific commands. Record machine/provider observations in
[TOOLING.md](TOOLING.md); accepted constraints remain in
[PROJECT_DECISIONS.md](../PROJECT_DECISIONS.md).

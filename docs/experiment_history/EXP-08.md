# EXP-08: FSDP sharding and ZeRO-style memory trade-offs

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-08`, `pytorch`, `aws`, `AWS-A2`

## Goal And Design

Educational goal: Learn what FSDP shards, which extra collectives it adds, and how to decide whether memory savings are worth the throughput and complexity cost when the model already fits.

### Scenario

A model fits with two-rank DDP, but the team assumes FSDP must be better because
it shards model state. The experiment must show which memory categories are
reduced, which collectives are added, and whether those trade-offs help when
the unsharded model already fits.

### Question

How do replicated DDP and FSDP sharding trade persistent memory savings for
additional communication, initialization complexity, and checkpoint handling on
two visible GPUs?

### Hypothesis

FSDP full sharding will reduce persistent per-rank model-state memory relative
to DDP, but may be slower for this fixed model because parameter materialization
and extra collectives add overhead. Partial sharding should land between DDP
and full sharding on memory and communication cost.

### Decision rule

Confirm the hypothesis when sharding reduces the expected memory categories and
the report can attribute any throughput loss to added collectives,
materialization, initialization, or checkpoint overhead.

Reject it when FSDP neither reduces the expected memory categories nor provides
a coherent communication or initialization explanation.

Leave it inconclusive when memory attribution, distributed logs, checkpoint
round-trip evidence, input identity, or image identity is incomplete.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | pytorch |
| Image | `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486` |
| Provider | aws |
| Compute profile | AWS-A2 |
| Resource type | g7e.12xlarge |
| GPU type | RTX PRO 6000 Blackwell Server Edition |
| Physical GPUs | 2 |
| Normal visible GPUs | 2 |
| Workload profile | benchmark |
| Baseline | EXP-08-A2V2-ddp-replicated-baseline |
| Declared variants | 4 |

## Current Conclusion

FSDP completed on two AWS GPUs and reduced peak memory versus DDP from about 19.3 GiB to 14.5 GiB per rank while also reducing step time in this short run from about 239 ms to about 205 ms. The prefetch variant was skipped because its admission gate was not met.

## Interpretation

The measured FSDP variants sharded model/gradient/optimizer state enough to lower memory. The result is encouraging but remains a short controlled run, not a general claim that FSDP is always faster.

## Run Inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-08/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
- Manifest status: `completed`
- Run units: `EXP-08-A2V2`
- Planned variant count in manifest/plan: `4`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-08/runs/aws-a2-full-fp16fix-20260715T023252Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-08-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-08-A2V2-utc.txt`=2026-07-15T02:39:16Z
- Raw artifact file count below `raw/`: `11`

## Measured Results From `aws-a2-full-fp16fix-20260715T023252Z`

| Variant | Strategy | Precision | World | Microbatch | Accum | Global batch | Seq | Mean step ms | Peak GiB | Loss | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-08-A2V2-ddp-replicated-baseline | ddp | bf16 | 2 | 1 | 1 | 2 | 1024 | 238.609 | 19.29 | r0=6.114; r1=6.063 | true |
| EXP-08-A2V2-fsdp-shard-grad-optim | fsdp | bf16 | 2 | 1 | 1 | 2 | 1024 | 205.459 | 14.49 | r1=6.067; r0=6.116 | true |
| EXP-08-A2V2-fsdp-full-shard | fsdp | bf16 | 2 | 1 | 1 | 2 | 1024 | 205.73 | 14.49 | r0=6.117; r1=6.068 | true |
| EXP-08-A2V2-fsdp-full-shard-prefetch | None | None | 2 | None | None | None | None |  |  |  |  |

## Formal Report Status

- Report file: `experiments/exp_08_fsdp_sharding_zero_memory_tradeoffs/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Use sharding to trade extra distributed machinery for lower per-rank state memory. Validate correctness and state-dict behavior before treating FSDP as a production recipe.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.

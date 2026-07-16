# EXP-11: Tensor plus sequence parallelism

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-11`, `nemo-megatron`, `runpod`, `RUNPOD-A100-SXM2`

## Goal And Design

Educational goal: Learn which parts of a transformer layer tensor parallelism shards, how sequence parallelism reduces activation pressure, and which collectives those choices introduce.

### Scenario

A team enables TP=2 on a transformer block that already fits on one GPU and
expects speedup from using a second GPU.

### Question

When does tensor parallelism reduce memory or enable larger layers, and what
communication cost appears when sequence parallelism is added?

### Hypothesis

TP=2 will reduce per-rank MLP shard state but add an all-reduce path. Sequence
parallelism will reduce local sequence activation pressure while changing the
per-rank token geometry.

### Decision rule

The hypothesis is confirmed when TP=2 reduces per-rank shard memory but reports
non-zero collective traffic, and sequence parallelism reduces local sequence
activation footprint relative to TP=2 without sequence parallelism.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | nemo-megatron |
| Image | `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` |
| Provider | runpod |
| Compute profile | RUNPOD-A100-SXM2 |
| Resource type | secure-cloud-pod |
| GPU type | NVIDIA A100-SXM4-80GB |
| Physical GPUs | 2 |
| Normal visible GPUs | variant-dependent |
| Workload profile | benchmark |
| Baseline | EXP-11-RUNPOD-A2V1-tp1-sp-off |
| Declared variants | 3 |

## Current Conclusion

Tensor parallelism reduced per-rank memory from about 190 MiB to 106 MiB and introduced non-zero collective traffic. Sequence parallelism reduced local sequence length from 1024 to 512 and slightly lowered memory further, but step time increased in this small synthetic workload.

## Interpretation

TP=2 split the intermediate dimension and changed communication. SP changed token placement within the TP group, which is useful when sequence activation memory is the constraint, even if the small benchmark does not show a speed gain.

## Run Inventory

### runpod-a2-megatron-20260715T203057Z

- Run ID: `runpod-a2-megatron-20260715T203057Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-11/runpod-a2-megatron-20260715T203057Z`
- Image: `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:8a03f34f9cb8e101ba59a92f98d50ff8e2e0dfa8094dffc2fb977af26176f484`
- Manifest status: `completed`
- Run units: `EXP-11-RUNPOD-A2-Megatron-V2`
- Planned variant count in manifest/plan: `2`
- Metrics JSONL: `artifacts/runs/runpod-volume-mirror/EXP-11/runpod-a2-megatron-20260715T203057Z/metrics/variant_results.jsonl`
- Raw artifact file count below `raw/`: `8`

## Measured Results From `runpod-a2-megatron-20260715T203057Z`

| Variant | TP | SP | World | Local seq | Shard intermediate | Max step ms | Tokens/s | Peak MiB | Collective MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-11-RUNPOD-A2V1-tp1-sp-off | 1 | false | 1 | 1024 | 4096 | 1.291 | 792,887 | 190.3 | 0.0 |
| EXP-11-RUNPOD-A2V2-tp2-sp-off | 2 | false | 2 | 1024 | 2048 | 1.721 | 594,972 | 106.3 | 28.0 |
| EXP-11-RUNPOD-A2V2-tp2-sp-on | 2 | true | 2 | 512 | 2048 | 2.119 | 483,274 | 101.3 | 14.0 |

## Formal Report Status

- Report file: `experiments/exp_11_tensor_sequence_parallelism/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Tensor parallelism is primarily a model-sharding tool; sequence parallelism targets activation placement. Measure both memory and collective cost.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.

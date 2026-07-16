# EXP-14: TP=2 x DP=2 for model width and throughput

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-14`, `nemo-megatron`, `runpod`, `RUNPOD-A100-SXM4`

## Goal And Design

Educational goal: Learn how tensor-parallel and data-parallel process groups compose in a four-rank hybrid, and how to reason about memory, throughput, and communication trade-offs across DP=4, TP=4, and TP=2 x DP=2.

### Scenario

A team has four peer-accessible A100 SXM GPUs and must choose between
replicating the model, sharding the model, or combining the two.

### Question

When is a TP=2 x DP=2 hybrid preferable to DP=4 or TP=4 on the same four GPUs?

### Hypothesis

The hybrid layout will expose both TP and DP communication paths and should sit
between DP=4 and TP=4 in the memory/throughput trade-off, rather than being
universally best.

### Decision rule

The hypothesis is confirmed when the hybrid records TP groups `[0,1]` and
`[2,3]`, DP groups `[0,2]` and `[1,3]`, and produces memory/throughput behavior
between pure DP and pure TP for the chosen shape.

### Configuration Snapshot

| Field | Value |
| --- | --- |
| Framework/image family | nemo-megatron |
| Image | `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b` |
| Provider | runpod |
| Compute profile | RUNPOD-A100-SXM4 |
| Resource type | secure-cloud-pod |
| GPU type | NVIDIA A100-SXM4-80GB |
| Physical GPUs | 4 |
| Normal visible GPUs | 4 |
| Workload profile | benchmark |
| Baseline | EXP-14-RUNPOD-A4-dp4 |
| Declared variants | 3 |

## Current Conclusion

The four-GPU Runpod A4 hybrid run completed. DP=4 gave the highest aggregate tokens/s in the bounded workload, TP=4 used the least memory, and TP=2 x DP=2 landed between them on throughput, memory, and collective volume.

## Interpretation

The experiment validated non-trivial tensor-parallel and data-parallel process groups on one NVLink-connected four-GPU host. The result is process-group and trade-off evidence, not a full Qwen3 Megatron recipe benchmark.

## Run Inventory

### runpod-a4-megatron-20260715T210918Z

- Run ID: `runpod-a4-megatron-20260715T210918Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-14/runpod-a4-megatron-20260715T210918Z`
- Image: `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b`
- Manifest status: `running`
- Run units: `EXP-14-RUNPOD-A4-Megatron`
- Planned variant count in manifest/plan: `3`
- Metrics JSONL: `artifacts/runs/runpod-volume-mirror/EXP-14/runpod-a4-megatron-20260715T210918Z/metrics/variant_results.jsonl`
- Raw artifact file count below `raw/`: `15`

### runpod-a4-megatron-20260715T212017Z

- Run ID: `runpod-a4-megatron-20260715T212017Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-14/runpod-a4-megatron-20260715T212017Z`
- Image: `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b`
- Manifest status: `completed`
- Run units: `EXP-14-RUNPOD-A4-Megatron`
- Planned variant count in manifest/plan: `3`
- Metrics JSONL: `artifacts/runs/runpod-volume-mirror/EXP-14/runpod-a4-megatron-20260715T212017Z/metrics/variant_results.jsonl`
- Raw artifact file count below `raw/`: `15`

## Measured Results From `runpod-a4-megatron-20260715T210918Z`

| Variant | Layout | TP | DP | World | TP ranks | DP ranks | Max step ms | Tokens/s | Peak MiB | Collective MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-14-RUNPOD-A4-dp4 | dp4 | 1 | 4 | 4 | [1] | [0, 1, 2, 3] | 4.263 | 480,467 | 183.3 | 896.0 |
| EXP-14-RUNPOD-A4-tp4 | tp4 | 4 | 1 | 4 | [0, 1, 2, 3] | [1] | 2.398 | 213,531 | 60.3 | 28.0 |
| EXP-14-RUNPOD-A4-tp2-dp2 | tp2_dp2 | 2 | 2 | 4 | [2, 3] | [0, 2] | 3.114 | 328,807 | 101.3 | 238.0 |

## Measured Results From `runpod-a4-megatron-20260715T212017Z`

| Variant | Layout | TP | DP | World | TP ranks | DP ranks | Max step ms | Tokens/s | Peak MiB | Collective MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-14-RUNPOD-A4-dp4 | dp4 | 1 | 4 | 4 | [2] | [0, 1, 2, 3] | 3.832 | 534,415 | 183.3 | 896.0 |
| EXP-14-RUNPOD-A4-tp4 | tp4 | 4 | 1 | 4 | [0, 1, 2, 3] | [2] | 2.345 | 218,329 | 60.3 | 28.0 |
| EXP-14-RUNPOD-A4-tp2-dp2 | tp2_dp2 | 2 | 2 | 4 | [2, 3] | [1, 3] | 3.316 | 308,836 | 101.3 | 238.0 |

## Formal Report Status

- Report file: `experiments/exp_14_tp2_dp2_hybrid/report.md`
- Report status line: Completed from Runpod A4 run

## Exam / Study Takeaway

TP=2 x DP=2 is the smallest four-rank layout that exercises both model sharding and data replication. Choose it when a model needs some sharding but still benefits from replicated throughput.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.

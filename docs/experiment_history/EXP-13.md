# EXP-13: Context parallelism for long sequences

Generated experiment-history file for RAG and cross-workstation continuity.

Source priority: measured artifact summaries in this file, then the raw artifact paths listed here, then the formal experiment report if it has already been completed.

## Tags

`EXP-13`, `nemo-megatron`, `runpod`, `RUNPOD-A100-SXM2`

## Goal And Design

Educational goal: Learn when context parallelism becomes useful for long sequence training, and how to compare its activation-memory savings against attention communication cost.

### Scenario

A team increases context length until attention activations, not parameters,
become the memory bottleneck.

### Question

At what sequence lengths does context parallelism's activation-memory reduction
justify its attention communication?

### Hypothesis

CP=2 will reduce each rank's local sequence length and peak activation pressure
for longer contexts, while adding K/V exchange traffic that can dominate shorter
contexts.

### Decision rule

The hypothesis is confirmed when CP=2 records half-length local sequence shards
and lower or bounded memory at long context while exposing non-zero K/V exchange
traffic. The crossover is inferred from the memory and throughput tables.

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
| Baseline | EXP-13-RUNPOD-A2V1-cp1-seq1024 |
| Declared variants | 5 |

## Current Conclusion

Context parallel variants completed on Runpod A2. CP=2 halved local sequence length for the same global sequence and lowered memory versus comparable one-rank long-sequence cases, while adding collective traffic.

## Interpretation

The run demonstrates the intended CP geometry: sequence context is partitioned across ranks, reducing local token/activation pressure at the cost of communication.

## Run Inventory

### runpod-a2-megatron-20260715T203057Z

- Run ID: `runpod-a2-megatron-20260715T203057Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-13/runpod-a2-megatron-20260715T203057Z`
- Image: `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:8a03f34f9cb8e101ba59a92f98d50ff8e2e0dfa8094dffc2fb977af26176f484`
- Manifest status: `completed`
- Run units: `EXP-13-RUNPOD-A2-Megatron-V2`
- Planned variant count in manifest/plan: `2`
- Metrics JSONL: `artifacts/runs/runpod-volume-mirror/EXP-13/runpod-a2-megatron-20260715T203057Z/metrics/variant_results.jsonl`
- Raw artifact file count below `raw/`: `12`

## Measured Results From `runpod-a2-megatron-20260715T203057Z`

| Variant | CP | World | Seq | Local seq | Checkpointing | Max step ms | Tokens/s | Peak MiB | Collective MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-13-RUNPOD-A2V1-cp1-seq1024 | 1 | 1 | 1024 | 1024 | false | 1.861 | 550,353 | 54.8 | 0.0 |
| EXP-13-RUNPOD-A2V1-cp1-seq2048 | 1 | 1 | 2048 | 2048 | false | 1.515 | 1,351,872 | 111.3 | 0.0 |
| EXP-13-RUNPOD-A2V1-cp1-seq4096-checkpoint | 1 | 1 | 4096 | 4096 | true | 3.259 | 1,256,812 | 321.3 | 0.0 |
| EXP-13-RUNPOD-A2V2-cp2-seq2048 | 2 | 2 | 2048 | 1024 | false | 3.022 | 677,591 | 63.3 | 40.0 |
| EXP-13-RUNPOD-A2V2-cp2-seq4096 | 2 | 2 | 4096 | 2048 | false | 3.046 | 1,344,750 | 165.3 | 80.0 |

## Formal Report Status

- Report file: `experiments/exp_13_context_parallelism_long_sequences/report.md`
- Report status line: Raw execution complete; formal validation pending

## Exam / Study Takeaway

Use context parallelism when long context memory is the constraint; expect communication cost and validate attention/backend support.

## Raw Artifact Transfer Note

The raw files referenced above are intentionally ignored by Git. To move them to another workstation, transfer the compressed artifact archive recorded in `HANDOFF.md` or recreate the mirrors from S3/Runpod before deeper analysis.

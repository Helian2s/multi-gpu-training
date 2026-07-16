# EXP-13: Context parallelism for long sequences - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-13.md](../../docs/experiment_history/EXP-13.md).

## Executive conclusion

Context parallel variants completed on Runpod A2. CP=2 halved local sequence length for the same global sequence and lowered memory versus comparable one-rank long-sequence cases, while adding collective traffic.

## Run inventory

### runpod-a2-megatron-20260715T203057Z

- Run ID: `runpod-a2-megatron-20260715T203057Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-13/runpod-a2-megatron-20260715T203057Z`
- Image: `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:8a03f34f9cb8e101ba59a92f98d50ff8e2e0dfa8094dffc2fb977af26176f484`
- Manifest status: `completed`
- Run units: `EXP-13-RUNPOD-A2-Megatron-V2`
- Planned variant count in manifest/plan: `2`
- Metrics JSONL: `artifacts/runs/runpod-volume-mirror/EXP-13/runpod-a2-megatron-20260715T203057Z/metrics/variant_results.jsonl`
- Raw artifact file count below `raw/`: `12`

## Measured results

### `runpod-a2-megatron-20260715T203057Z` metrics

| Variant | CP | World | Seq | Local seq | Checkpointing | Max step ms | Tokens/s | Peak MiB | Collective MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-13-RUNPOD-A2V1-cp1-seq1024 | 1 | 1 | 1024 | 1024 | false | 1.861 | 550,353 | 54.8 | 0.0 |
| EXP-13-RUNPOD-A2V1-cp1-seq2048 | 1 | 1 | 2048 | 2048 | false | 1.515 | 1,351,872 | 111.3 | 0.0 |
| EXP-13-RUNPOD-A2V1-cp1-seq4096-checkpoint | 1 | 1 | 4096 | 4096 | true | 3.259 | 1,256,812 | 321.3 | 0.0 |
| EXP-13-RUNPOD-A2V2-cp2-seq2048 | 2 | 2 | 2048 | 1024 | false | 3.022 | 677,591 | 63.3 | 40.0 |
| EXP-13-RUNPOD-A2V2-cp2-seq4096 | 2 | 2 | 4096 | 2048 | false | 3.046 | 1,344,750 | 165.3 | 80.0 |

## Interpretation

The run demonstrates the intended CP geometry: sequence context is partitioned across ranks, reducing local token/activation pressure at the cost of communication.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Use context parallelism when long context memory is the constraint; expect communication cost and validate attention/backend support.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.

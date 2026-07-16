# EXP-11: Tensor plus sequence parallelism - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-11.md](../../docs/experiment_history/EXP-11.md).

## Executive conclusion

Tensor parallelism reduced per-rank memory from about 190 MiB to 106 MiB and introduced non-zero collective traffic. Sequence parallelism reduced local sequence length from 1024 to 512 and slightly lowered memory further, but step time increased in this small synthetic workload.

## Run inventory

### runpod-a2-megatron-20260715T203057Z

- Run ID: `runpod-a2-megatron-20260715T203057Z`
- Artifact directory: `artifacts/runs/runpod-volume-mirror/EXP-11/runpod-a2-megatron-20260715T203057Z`
- Image: `ghcr.io/helian2s/multi-gpu-training-nemo@sha256:8a03f34f9cb8e101ba59a92f98d50ff8e2e0dfa8094dffc2fb977af26176f484`
- Manifest status: `completed`
- Run units: `EXP-11-RUNPOD-A2-Megatron-V2`
- Planned variant count in manifest/plan: `2`
- Metrics JSONL: `artifacts/runs/runpod-volume-mirror/EXP-11/runpod-a2-megatron-20260715T203057Z/metrics/variant_results.jsonl`
- Raw artifact file count below `raw/`: `8`

## Measured results

### `runpod-a2-megatron-20260715T203057Z` metrics

| Variant | TP | SP | World | Local seq | Shard intermediate | Max step ms | Tokens/s | Peak MiB | Collective MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-11-RUNPOD-A2V1-tp1-sp-off | 1 | false | 1 | 1024 | 4096 | 1.291 | 792,887 | 190.3 | 0.0 |
| EXP-11-RUNPOD-A2V2-tp2-sp-off | 2 | false | 2 | 1024 | 2048 | 1.721 | 594,972 | 106.3 | 28.0 |
| EXP-11-RUNPOD-A2V2-tp2-sp-on | 2 | true | 2 | 512 | 2048 | 2.119 | 483,274 | 101.3 | 14.0 |

## Interpretation

TP=2 split the intermediate dimension and changed communication. SP changed token placement within the TP group, which is useful when sequence activation memory is the constraint, even if the small benchmark does not show a speed gain.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Tensor parallelism is primarily a model-sharding tool; sequence parallelism targets activation placement. Measure both memory and collective cost.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.

# EXP-09: Controlled troubleshooting and failure diagnosis - interim report

Report status: Raw execution complete; formal validation pending

This interim report replaces the old placeholder so repository search and RAG do not incorrectly report the experiment as not run. It is a derived summary from local artifact mirrors, not a final validated publication report. The detailed RAG-oriented history is in [docs/experiment_history/EXP-09.md](../../docs/experiment_history/EXP-09.md).

## Executive conclusion

Controlled failure cases produced the intended signatures: healthy smoke passed, OOM and FP16 overflow were expected failures, input starvation produced a timed wait, mismatched collectives surfaced an explicit collective fingerprint mismatch, and a timeout case recorded bounded timeout evidence.

## Run inventory

### aws-a2-full-fp16fix-20260715T023252Z

- Run ID: `aws-a2-full-fp16fix-20260715T023252Z`
- Artifact directory: `artifacts/runs/aws-s3-mirror/EXP-09/runs/aws-a2-full-fp16fix-20260715T023252Z`
- Image: `037678282394.dkr.ecr.us-west-2.amazonaws.com/multi-gpu-training-pytorch@sha256:8f7e455bc939e95bd795bbe569224cd2728903324324df7f60dcbffc9af38486`
- Manifest status: `completed`
- Run units: `EXP-09-A2V2`
- Planned variant count in manifest/plan: `3`
- Metrics JSONL: `artifacts/runs/aws-s3-mirror/EXP-09/runs/aws-a2-full-fp16fix-20260715T023252Z/metrics/variant_results.jsonl`
- Exit statuses: `exit_status-EXP-09-A2V1.txt`=0, `exit_status-EXP-09-A2V2.txt`=0
- Finished UTC markers: `finished-EXP-09-A2V1-utc.txt`=2026-07-15T02:39:31Z, `finished-EXP-09-A2V2-utc.txt`=2026-07-15T02:41:54Z
- Raw artifact file count below `raw/`: `18`

## Measured results

### `aws-a2-full-fp16fix-20260715T023252Z` metrics

| Variant | Case | World | Status | Elapsed s | Detail |
| --- | --- | --- | --- | --- | --- |
| EXP-09-A2V1-healthy-smoke | healthy_training_smoke | 1 | pass | 0.165 | healthy CUDA smoke completed |
| EXP-09-A2V1-oom-fragmentation | bounded_oom_or_fragmentation | 1 | expected_failure | 0 | controlled CUDA out of memory diagnostic |
| EXP-09-A2V1-numerical-overflow | fp16_nonfinite_gradient | 1 | expected_failure | 0.558 | controlled fp16 non-finite diagnostic |
| EXP-09-A2V1-input-starvation | input_pipeline_starvation | 1 | pass | 5 | controlled input wait of 5.0 seconds |
| EXP-09-A2V2-mismatched-collective | mismatched_collective | 2 | expected_failure | 0.965 | Detected mismatch between collectives on ranks. Rank 0 is running collective: CollectiveFingerPrint(SequenceNumber=0, OpType=ALLREDUCE, TensorShape=[1], TensorDtypes=Float, Tens... |
| EXP-09-A2V2-rank-straggler | artificial_rank_straggler | 2 | pass | 10.955 | straggler collective completed |
| EXP-09-A2V2-nccl-timeout-evidence | nccl_timeout_signature | 2 | expected_failure | 120 | controlled timeout after 120s; see /workspace/multi-gpu-training/artifacts/runs/EXP-09/aws-a2-full-fp16fix-20260715T023252Z/raw/EXP-09-A2V2-nccl-timeout-evidence/subprocess.log |

## Interpretation

The experiment created known-bad cases so future troubleshooting can distinguish memory pressure, numerical overflow, input stalls, rank stragglers, collective mismatches, and NCCL timeout symptoms.

## Limitations and anomalies

- This is a short bounded lab measurement, not model training to convergence.
- Raw artifacts remain ignored by Git; transfer the compressed artifact archive for deep reanalysis.
- Catalog lifecycle status remains `accepted` until final validation and completed report review.

## Exam takeaway

Good distributed debugging starts from the failure signature: OOM, non-finite gradients, input wait, rank mismatch, and timeout each point to different first checks.

## Reproduction

Use the experiment spec, queue configuration, and immutable image digest recorded in this report and in `docs/experiment_history/`. Restore `artifacts/runs/` from the artifact archive before rerunning local analysis.

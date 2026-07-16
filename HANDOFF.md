# Current workstation handoff

Prepared: 2026-07-16

From: Ubuntu x86_64 workstation

To: next project workstation

Branch: `main`

This is an operational handoff, not a source of project-wide decisions. If it
conflicts with `PROJECT_DECISIONS.md` or `EXPERIMENT_CATALOG.md`, those
authoritative files govern. Replace this file at the next explicit workstation
handoff instead of accumulating a historical log here.

## Receiving Codex instructions

1. Read `AGENTS.md`, `WORKFLOW.md`, `README.md`, this file,
   `PROJECT_DECISIONS.md`, the relevant catalog section, and
   `infra/TOOLING.md` before changing the repository.
2. Inspect the branch, recent commits, and worktree. Run `make check` before
   assuming the checkout is ready.
3. Do not assume Ubuntu-local credentials, Docker images, ignored artifacts,
   model/data inputs, or cloud authentication exist on the next workstation.
4. Do not create billable resources, cloud identities, registries, or publish
   images without explicit user approval.

## Current project state

- AWS EXP-01 through EXP-09 have raw execution artifacts in S3 and a local
  ignored mirror. Interim RAG-oriented reports now summarize the measured
  artifacts; final validation and completed lifecycle status remain pending.
- AWS EXP-12 is completed and reported.
- Runpod EXP-10, EXP-11, and EXP-13 have raw execution artifacts from
  `runpod-a2-megatron-20260715T203057Z`; interim RAG-oriented reports now
  summarize the measured artifacts; final validation and completed lifecycle
  status remain pending.
- Runpod EXP-14 is completed and reported from
  `runpod-a4-megatron-20260715T212017Z`.
- RAG-friendly experiment history files are tracked under
  `docs/experiment_history/`. They can be regenerated after restoring the raw
  artifact mirrors with `make experiment-history`.
- The CUDA 12.8 Runpod NeMo image used for EXP-10/11/13/14 is published to
  GHCR at digest
  `sha256:c2713c9027894f03d4da724cca04cc51256cec4bdc4b1f42b63578ff6133ac3b`.
- The EXP-14 run used a container-side hotpatch that is now committed in source:
  `common/megatron_executor.py` skips explicit distributed process-group
  destruction by default to avoid NCCL cleanup hangs in short-lived workers.
  Publish a replacement immutable image before future reproducibility runs.

## Ignored local assets

The user asked to preserve artifacts for another workstation. Raw artifacts are
not committed because repository rules exclude large experiment outputs and raw
profiler traces. Copy these ignored directories outside Git if the next
workstation needs local analysis without re-downloading:

```text
artifacts/runs/aws-s3-mirror/
artifacts/runs/runpod-volume-mirror/
```

Current local artifact mirror size is about `119M` across `347` files. It
includes AWS and Runpod raw logs, metrics, queue logs, and PyTorch profiler
trace JSON. The same AWS artifacts also remain durable in S3 under the
`artifacts/` prefixes recorded in `infra/TOOLING.md`; Runpod artifacts were
copied from Pod volumes into the local mirror before Pod deletion.

The current compressed transfer archive on this Ubuntu workstation is:

```text
/home/val/Documents/py-projects/multi-gpu-training-artifacts-20260715T214405Z.tar.zst
/home/val/Documents/py-projects/multi-gpu-training-artifacts-20260715T214405Z.tar.zst.sha256
```

SHA256:
`be457ed5f8b9d959316456ad2a32257f05b2ff71c82e6a17d98a0bac31c2af8f`

It contains the top-level `artifacts/` directory and verified byte-for-byte
against the current local mirror at the time it was created. To restore it
after cloning on another workstation:

```bash
sha256sum -c multi-gpu-training-artifacts-20260715T214405Z.tar.zst.sha256
tar --zstd -xf multi-gpu-training-artifacts-20260715T214405Z.tar.zst -C /path/to/multi-gpu-training
cd /path/to/multi-gpu-training
make check
make experiment-history
```

The pinned model, raw dataset, and processed token streams remain ignored under
`data/raw/` and `data/processed/`. They are also staged in S3 under
`inputs/qwen3-wikitext-v1/` for AWS queues.

## Provider state

- Runpod read-only checks report no active Pods, no network volumes,
  `currentSpendPerHr=0`, account balance about `$17.38`, and spend limit `$80`.
  A direct GraphQL attempt to set `spendLimit=1` failed because Runpod does not
  expose `spendLimit` in `UpdateUserSettingsInput`.
- AWS read-only cleanup checks previously found no active/stopped project EC2
  instances, no project-tagged EBS volumes, and no unattached available EBS
  volumes in `us-west-2`.
- AWS ECR repositories `multi-gpu-training-pytorch` and
  `multi-gpu-training-nemo` still exist in `us-west-2`, but an approved cleanup
  deleted all images on 2026-07-16. Future AWS launches must rebuild and
  republish the required image before use.
- Provider credentials, API keys, SSH keys, Docker registry sessions, and local
  cloud config are not in Git and must be configured separately on the next
  workstation.

## Next recommended work

1. Transfer ignored artifact mirrors outside Git if local analysis is needed on
   the next workstation.
2. Run `make check` after checkout and after any artifact transfer.
3. Review the interim reports and RAG history for AWS EXP-01 through EXP-09,
   then promote them to final validated reports when ready.
4. Review the interim reports and RAG history for Runpod EXP-10, EXP-11, and
   EXP-13, then promote them to final validated reports when ready.
5. Publish a refreshed Runpod NeMo image that includes the committed NCCL
   cleanup fix before any future rerun.
6. Rebuild and publish fresh AWS ECR images before any future AWS experiment
   launch.

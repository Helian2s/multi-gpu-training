# Current workstation handoff

Prepared: 2026-07-13

From: macOS ARM64 workstation

To: Ubuntu x86_64 workstation

Branch: `agent/document-project-decisions`

This is an operational handoff, not a source of project-wide decisions. If it
conflicts with `PROJECT_DECISIONS.md` or `EXPERIMENT_CATALOG.md`, those
authoritative files govern. Replace this file at the next explicit workstation
handoff instead of accumulating a historical log here.

## Receiving Codex instructions

1. Read `AGENTS.md`, `WORKFLOW.md`, `README.md`, this file,
   `PROJECT_DECISIONS.md`, and the relevant catalog section before changing the
   repository.
2. Inspect the branch, recent commits, and worktree. Run `make check` before
   assuming the checkout is ready.
3. Audit the Ubuntu workstation according to `WORKFLOW.md`; replace the unknown
   Ubuntu entries in `infra/TOOLING.md` with observed versions and readiness.
4. Preserve all current scope, provider, GPU-count, experiment-order, model,
   dataset, and registry decisions. A new chat session is not permission to
   revisit accepted decisions.
5. Do not create billable resources, cloud identities, registries, or publish
   images without explicit user approval.

## Data transfer

The user will transfer project inputs through Google Drive after cloning or
updating the Git repository on Ubuntu. Copy only these ignored directories into
the corresponding location in the clone:

```text
data/raw/        approximately 3.5 GiB
data/processed/  approximately 552 MiB
```

Do not transfer `data/cache/`, the macOS `.venv/`, Docker Desktop state, or any
credential. The tracked `data/README.md` arrives through Git.

After copying the directories:

```bash
make prepare-environment
make verify-inputs
git status -sb
```

`make verify-inputs` hashes all model, dataset, and canonical token-stream files
against `data/processed/.../manifest.json`. The expected result is ten model
files, six dataset files, and nine processed files. The transfer must not make
Git report the ignored data as tracked changes.

## Current asset and tooling state

- Accepted model/tokenizer: `Qwen/Qwen3-1.7B-Base` at the immutable revision in
  `configs/inputs.lock.yaml`.
- Accepted dataset: `Salesforce/wikitext`, `wikitext-103-raw-v1`, at the
  immutable revision in the same lock.
- The model, raw dataset, and canonical token streams are complete on macOS and
  are not stored in Git.
- The Ubuntu workstation has not yet been audited.
- No project PyTorch or NeMo/Megatron container image has been built.
- AWS CLI and AWS/ECR identity have not been verified for this project.
- GHCR package publication has not been tested.
- The Runpod API key previously exposed in chat must be rotated before paid
  Runpod work; the replacement must never enter the repository or chat.

## Next recommended work

1. Complete the Ubuntu audit and input verification.
2. Confirm NVIDIA NGC authentication interactively without exposing the key.
3. Compatibility-test and then accept exact NGC base references before adding
   Dockerfiles. Current candidates are `nvcr.io/nvidia/pytorch:26.06-py3` and
   `nvcr.io/nvidia/nemo:26.06`; candidate status is not an accepted pin.
4. Build the first `linux/amd64` project image only after the base and dependency
   compatibility contract is recorded.

Do not begin a numbered experiment until its implementation is accepted and
the relevant provider qualification and cost gates exist.

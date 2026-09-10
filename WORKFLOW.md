# Cross-workstation workflow

This project is expected to move between a macOS ARM64 workstation, an Ubuntu
x86_64 workstation, and temporary cloud GPU hosts. The objective of this
workflow is to make a fresh Codex session sufficient even when its local chat
history is different.

## Shared-state model

| State | Authoritative location | Transfer rule |
| --- | --- | --- |
| Project scope and accepted decisions | `PROJECT_DECISIONS.md` | Commit to Git |
| Experiment definitions and lifecycle | `EXPERIMENT_CATALOG.md` | Commit to Git |
| Agent working agreements | `AGENTS.md` | Loaded automatically by Codex from the repository |
| Current workstation transition | `HANDOFF.md` | Replace when a new explicit handoff is prepared |
| Workstation and provider readiness | `infra/TOOLING.md` | Record facts without secrets and commit to Git |
| Source, configuration, tests, and image definitions | Repository files | Commit to Git |
| Model, dataset, processed inputs, images, and run artifacts | Ignored local storage or approved durable cloud storage | Reproduce or transfer with checksum verification |
| API keys, cloud credentials, SSH private keys, and registry tokens | Local credential store or environment | Never commit or copy through project files |
| Conversation history | Local Codex session | Helpful context only; never the project record |

## Workstation roles

These roles describe convenience, not a hard restriction:

- **macOS ARM64:** documentation, source changes, data preparation, analysis,
  and lightweight `linux/amd64` build checks through Docker emulation.
- **Ubuntu x86_64:** preferred workstation for large NVIDIA image pulls,
  native `linux/amd64` image builds, registry publication preparation, and
  other CPU/RAM/disk-intensive local work.
- **AWS and Runpod GPU hosts:** qualification and measured CUDA/GPU execution
  only. A cloud host is not the primary development environment.

Neither local workstation can validate CUDA behavior unless it has a supported
NVIDIA GPU and NVIDIA Container Toolkit. The final image must therefore pass a
provider-side GPU smoke test even if the local build succeeds.

## Starting a session

Install the [minimal local-check dependencies](tests/README.md) before running
checks on a fresh workstation. `make check` uses `python3` by default; if using
the repository virtual environment, run `make check PYTHON=.venv/bin/python`
instead. The same `PYTHON` override applies to plan-generation targets.

Start Codex at the repository root. Before making changes, inspect the shared
state:

```bash
git status -sb
git branch --show-current
git log --oneline --decorate -5
uname -srm
make check
```

Then read:

1. `AGENTS.md` for durable working and safety rules.
2. `README.md` for the project overview and `docs/README.md` for document roles.
3. `HANDOFF.md` for current local-only assets, blockers, and the next action.
4. `PROJECT_DECISIONS.md` for accepted constraints.
5. The relevant part of `EXPERIMENT_CATALOG.md` for the current task.
6. `infra/TOOLING.md` for the current machine's known readiness.
7. `docs/validation-status.md` for later source-review findings that affect
   interpretation of the recorded experiments.

Do not infer that another workstation's successful check applies here. Audit a
new workstation once and update only its section in `infra/TOOLING.md`.

## Synchronizing between workstations

Sequential work on a shared branch is the simplest process:

1. Finish a coherent change on workstation A.
2. Run `make check`, review `git status` and the diff, then commit and push.
3. On workstation B, confirm its worktree is clean, fetch the remote, switch to
   the intended tracking branch, and use a fast-forward-only pull.
4. Run `make check` on workstation B before continuing.

Never use a pull to resolve an unexplained dirty worktree. If both workstations
must work at the same time, create a separate branch per task and merge or
cherry-pick reviewed commits. Do not edit the same branch concurrently and hope
that conversation history will reconcile it.

## First Ubuntu session

The first Codex session on the Ubuntu x86_64 workstation should:

1. Verify Git, GNU Make, Python, Docker Engine, Buildx, available RAM, available
   disk space, and outbound registry access.
2. Run `docker info` and `docker buildx ls`; do not assume a local NVIDIA GPU.
3. Verify GitHub authentication only if repository or GHCR operations need it.
4. Verify NGC, AWS, Runpod, ECR, or GHCR authentication separately and only
   when the next approved task requires it. Never print credential values.
5. Run `make check` and record exact tool versions and readiness in the Ubuntu
   section of `infra/TOOLING.md`.
6. Create the ignored `.venv` with `make prepare-environment` only when local
   input preparation is needed. Reproduce inputs with `make prepare-inputs` or
   transfer `data/raw/` and `data/processed/` through approved storage and run
   `make verify-inputs` against the generated manifest.

## Ending a session

Before moving to the other workstation:

```bash
make check
git status -sb
git diff --check
```

Also:

- Update the appropriate authoritative or operational document when its state
  changed.
- Confirm that credentials, model/data caches, container archives, checkpoints,
  and profiler artifacts remain ignored.
- Commit a coherent unit and push it when requested.
- In the final handoff, state the branch, commit, checks run, pending work, and
  any local-only asset that the other workstation must reproduce.

## Suggested prompt for a fresh Codex session

```text
Read AGENTS.md, WORKFLOW.md, and HANDOFF.md, then inspect README.md, git status,
the recent commits, and this workstation's section in infra/TOOLING.md.
Summarize the accepted project state and the next safe task before making
changes. Do not assume that another workstation's tools, credentials, caches,
or chat history exist here.
```

# multi-gpu-training

Experimental GPU acceleration and single-node distributed LLM training lab for
the NCP-GENL certification.

## Document map

| Document | Purpose | Authority |
| --- | --- | --- |
| [AGENTS.md](AGENTS.md) | Durable Codex working agreements, safety gates, and cross-workstation handoff rules loaded from the repository | Authoritative for agent workflow |
| [WORKFLOW.md](WORKFLOW.md) | Human-readable startup, synchronization, workstation-role, and handoff procedure | Authoritative for cross-workstation procedure |
| [HANDOFF.md](HANDOFF.md) | Current workstation-transition state, local-only assets, blockers, and receiving-Codex instructions | Operational snapshot; replace at the next explicit handoff |
| [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md) | Current snapshot and history of accepted project scope, infrastructure, frameworks, workload rules, exclusions, container strategy, tools, and reproducibility requirements | Authoritative for project-wide decisions |
| [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md) | Experiment candidates, lifecycle status, hypotheses, scenarios, measurements, estimated GPU-hours, and implementation order | Authoritative for experiment definitions and status |
| [infra/TOOLING.md](infra/TOOLING.md) | Dated per-workstation operational audit of installed tools, authentication readiness, and remaining provider checks | Informational; does not change project decisions |

This README is only the repository entry point; it intentionally does not copy
project decisions or experiment details. If the two substantive documents ever
conflict, `PROJECT_DECISIONS.md` governs project-wide constraints, while
`EXPERIMENT_CATALOG.md` governs the content and status of experiments that fit
within those constraints.

## Repository layout

```text
.
├── artifacts/             # Local run outputs; only its README is tracked
├── common/                # Shared qualification, launch, telemetry, and analysis contracts
├── configs/               # Shared workload/configuration examples
├── containers/            # PyTorch and NeMo/Megatron image definitions
├── data/                  # Dataset preparation contract; downloaded data is ignored
├── experiments/           # Accepted experiment directories and canonical template
├── infra/                 # Provider-neutral contract plus AWS and Runpod adapters
├── scripts/               # Local automation and experiment scaffolding
├── tests/                 # Tests for shared and experiment-specific code
├── AGENTS.md              # Durable Codex working agreements and safety gates
├── WORKFLOW.md            # Cross-workstation startup and handoff procedure
├── HANDOFF.md             # Current receiving-workstation instructions and state
├── Makefile               # Stable local entry points
├── requirements-preparation.txt # Pinned local input-preparation environment
├── EXPERIMENT_CATALOG.md  # Proposed experiments and lifecycle status
└── PROJECT_DECISIONS.md   # Current decisions and append-only decision log
```

Catalog entries do not receive a directory merely because they have an ID. Once
an experiment's catalog status is `accepted`, create its directory from
`experiments/_template` with `scripts/new_experiment.py`. Generated datasets,
checkpoints, raw run data, and profiler traces remain outside Git according to
the contracts in `data/` and `artifacts/`.

Run `make help` for the currently available local entry points.

# Documentation guide

Start with the [project overview](../README.md) for the engineering work,
hardware, experiment coverage, and selected observations.

## Read the work

| Document | What it explains |
| --- | --- |
| [Experiment guide](../experiments/README.md) | What each experiment ran, with links to its report |
| [Run history](experiment_history/README.md) | Recorded run IDs, extracted metrics, and per-experiment summaries |
| [Validation status](validation-status.md) | Source-review findings and the limits on interpreting results |
| [Operational timeline](experiment_history/operational_timeline.md) | Launch attempts, compatibility issues, and runtime fixes |
| [Artifact guide](../artifacts/README.md) | Where raw evidence lives and how to restore it |

The experiment-history pages are generated snapshots of earlier artifact
mirrors. Their references to local files describe the generating workstation,
not every checkout. Read them alongside the later validation review.

## Work on the repository

| Document | Role |
| --- | --- |
| [Tests](../tests/README.md) | Minimal local environment, checks, and coverage limits |
| [Shared runtime](../common/README.md) | Runner and executor responsibilities |
| [Data preparation](../data/README.md) | Pinned inputs, token streams, and checksum verification |
| [Containers](../containers/README.md) | Image families, builds, and historical image availability |
| [Infrastructure](../infra/README.md) | Provider queues and actual lifecycle behavior |
| [Workflow](../WORKFLOW.md) | Startup, synchronization, and workstation handoff |

## Document authority

The README and directory guides summarize the implemented project. They do not
change accepted scope or experiment status.

| Record | Authority |
| --- | --- |
| [PROJECT_DECISIONS.md](../PROJECT_DECISIONS.md) | Accepted project-wide scope and constraints, plus append-only decision history |
| [EXPERIMENT_CATALOG.md](../EXPERIMENT_CATALOG.md) | Canonical experiment IDs, definitions, hypotheses, and lifecycle status |
| [AGENTS.md](../AGENTS.md) | Repository working agreements for coding agents |
| [WORKFLOW.md](../WORKFLOW.md) | Cross-workstation procedure |
| [HANDOFF.md](../HANDOFF.md) | Dated operational snapshot from the last explicit handoff |
| [infra/TOOLING.md](../infra/TOOLING.md) | Dated workstation and provider readiness observations |

Project decisions govern project-wide constraints; the catalog governs
experiments within those constraints. Review findings identify gaps between
the contract and implementation without silently changing either. Generated
summaries and old handoffs are evidence of past state, not live provider checks.

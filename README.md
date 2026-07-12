# multi-gpu-training

Experimental GPU acceleration and single-node distributed LLM training lab for
the NCP-GENL certification.

## Document map

| Document | Purpose | Authority |
| --- | --- | --- |
| [PROJECT_DECISIONS.md](PROJECT_DECISIONS.md) | Current snapshot and history of accepted project scope, infrastructure, frameworks, workload rules, exclusions, container strategy, tools, and reproducibility requirements | Authoritative for project-wide decisions |
| [EXPERIMENT_CATALOG.md](EXPERIMENT_CATALOG.md) | Experiment candidates, selection status, hypotheses, scenarios, measurements, estimated GPU-hours, and implementation order | Authoritative for experiment definitions and status |

This README is only the repository entry point; it intentionally does not copy
project decisions or experiment details. If the two substantive documents ever
conflict, `PROJECT_DECISIONS.md` governs project-wide constraints, while
`EXPERIMENT_CATALOG.md` governs the content and status of experiments that fit
within those constraints.

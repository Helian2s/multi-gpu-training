# Codex working agreements

Read `README.md`, `PROJECT_DECISIONS.md`, and the relevant part of
`EXPERIMENT_CATALOG.md` before changing project scope or experiments.

- Keep `PROJECT_DECISIONS.md` as both the current authoritative snapshot and an
  append-only decision history. When a user-approved project-wide decision
  changes, update the snapshot and append a dated `PD-NNN` entry that states
  what it supersedes; do not rewrite old entries to hide the change.
- Keep canonical experiment IDs in continuous `EXP-NN` order and update summary
  tables, detailed headings, cross-references, implementation order, and session
  plans together.
- Create directories only for experiments whose catalog status is `accepted`.
  Change an experiment to `completed` only after its required artifacts and
  report have been validated.
- Assign one cloud provider to each numbered experiment. Every distributed run
  is single-node. Do not introduce a four-GPU run without the hypothesis and
  admission rule required by the current project decisions.
- Keep provider APIs out of experiment logic. Put lifecycle behavior under
  `infra/`, shared runtime contracts under `common/`, and accepted experiment
  implementations under `experiments/`.
- Never commit credentials, resolved secrets, downloaded model/data caches,
  checkpoints, raw profiler traces, or other large artifacts.
- Do not create, start, stop, or delete a billable cloud resource, change a
  quota, create cloud credentials/roles, or publish an image without explicit
  user approval. Read-only identity, quota, capacity, and status checks are
  allowed when relevant.
- Any future launch automation must require a dry-run or confirmation plus
  maximum lifetime/cost, durable stage-out, and termination-on-failure guards.
- Prefer the stable repository entry points in `Makefile`. Run `make check`
  after repository changes and add focused tests when behavior changes.
- Record dynamic workstation/provider readiness in `infra/TOOLING.md`; do not
  confuse an approved tool with an installed or authenticated tool.

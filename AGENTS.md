# Codex working agreements

Read `README.md`, `PROJECT_DECISIONS.md`, and the relevant part of
`EXPERIMENT_CATALOG.md` before changing project scope or experiments.

## Cross-workstation continuity

The Git repository is the shared project memory. Local Codex conversation
history is not authoritative and must not be required to continue the work.

- At the beginning of a session, inspect `git status -sb`, the current branch,
  and recent commits before editing. Read `WORKFLOW.md`, `HANDOFF.md`, and the
  matching workstation section in `infra/TOOLING.md`.
- Identify the current OS and architecture. Never assume that tools,
  credentials, caches, downloaded inputs, Docker images, or authentication from
  another workstation are available locally.
- Preserve unrelated or uncommitted work. Do not pull into a dirty worktree or
  switch branches in a way that could overwrite local changes.
- Treat only committed files as cross-workstation handoff state. Record durable
  decisions in `PROJECT_DECISIONS.md`, experiment state in
  `EXPERIMENT_CATALOG.md`, and machine-specific readiness in
  `infra/TOOLING.md`.
- Before handing work to another workstation, run `make check`, inspect the
  diff for secrets or large artifacts, update any affected status document,
  and clearly report uncommitted work. Commit and push only when the user asks.
- Prefer sequential work on one shared branch. If two workstations will edit
  concurrently, use separate task branches and reconcile them through Git
  rather than copying working directories.

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

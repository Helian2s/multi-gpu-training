# Experiments

This directory contains only accepted experiments plus the canonical
`_template`. Proposed experiments and their lifecycle status remain in
`../EXPERIMENT_CATALOG.md`.

## Lifecycle

1. After explicit acceptance, set the catalog's `Status` value to `accepted`.
2. Run `python3 scripts/new_experiment.py EXP-NN slug "Experiment title"`.
3. Complete the hypothesis, configuration, expected-result, and source files
   before authorizing GPU time.
4. Implement local tests and the smallest applicable GPU smoke test.
5. Run the experiment and store raw outputs under `artifacts/runs/`.
6. Implement `analyze.py` without changing the predeclared decision rule after
   observing results.
7. Complete `report.md`, including whether the hypothesis was confirmed,
   rejected, or left inconclusive.
8. After validating the required artifacts and report, set the catalog's
   `Status` value to `completed`.

Directory names use `exp_NN_short_slug`, while the canonical ID inside files
uses `EXP-NN`.

## File contract

| File | Purpose |
| --- | --- |
| `hypothesis.md` | Human-readable scenario, falsifiable hypothesis, variables, rationale, and invalidation criteria |
| `experiment.yaml` | Machine-readable environment, workload, sweep, launcher, measurement, and output configuration |
| `expected_results.yaml` | Predeclared correctness gates and decision rules used to confirm or reject expectations |
| `sources.bib` | Research papers and authoritative documentation supporting the hypothesis |
| `analyze.py` | Experiment-specific, reproducible transformation from raw metrics to derived results |
| `report.md` | Observed environment, results, interpretation, limitations, cost, and conclusion |

Training or launcher code may be added inside an experiment when it is truly
specific. Reusable functionality belongs under `common/` and is tested under
`tests/common/`.

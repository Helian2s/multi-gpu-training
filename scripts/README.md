# Local automation

Run these tools from the repository root, preferably through the
[Makefile](../Makefile). Cloud lifecycle code lives under
[infra/](../infra/README.md).

| Script | What it does | Make target |
| --- | --- | --- |
| [new_experiment.py](new_experiment.py) | Scaffold an accepted experiment from the canonical template | `make new-experiment ID=EXP-NN SLUG=slug TITLE="Title"` |
| [prepare_inputs.py](prepare_inputs.py) | Fetch pinned model/dataset snapshots, write deterministic token streams and checksums, or verify existing inputs | `make prepare-inputs`, `make verify-inputs` |
| [validate_repo.py](validate_repo.py) | Check YAML syntax, local Markdown links, canonical experiment IDs/titles, provider placement, and catalog structure | Included in `make check` |
| [build_experiment_history.py](build_experiment_history.py) | Generate per-experiment history, an index, a timeline, and interim reports from specs, curated notes, and artifact mirrors | `make experiment-history` |

## History generation

Restore the raw AWS and Runpod mirrors described in
[the artifact guide](../artifacts/README.md) before running:

```bash
make experiment-history PYTHON=.venv/bin/python
```

This command **writes tracked Markdown files**. It rebuilds the history pages
from whatever mirrors are available locally and overwrites interim reports
for experiments other than EXP-12/EXP-14 when run directories are present.
Running it without mirrors can remove historical metric tables from generated
pages. Review the diff before retaining regenerated output.

The generator combines extracted numbers with curated interpretations in
`EXPERIMENT_NOTES`; it does not independently validate hypotheses. The later
[validation review](../docs/validation-status.md) must be read alongside those
historical summaries. Do not regenerate merely to update the documentation
navigation or workstation state.

## Dependencies

Repository checks and plan generation need Python 3.12 or later and PyYAML.
Input preparation uses the larger pinned environment in
[requirements-preparation.txt](../requirements-preparation.txt). See
[tests](../tests/README.md) for minimal setup and [data](../data/README.md) for
preparation commands. `build_experiment_history.py` runs immediately when
invoked; use the description above rather than assuming it has a `--help` mode.

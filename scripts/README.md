# Automation scripts

Scripts here provide stable local entry points for scaffolding, preparation,
build, upload, run, collection, and analysis workflows. They must be safe to run
from the repository root and must not print secrets.

Currently available:

- `new_experiment.py`: instantiate an accepted experiment from the canonical
  template.
- `prepare_inputs.py`: download the accepted immutable Qwen3 and WikiText
  snapshots, then create canonical per-split token streams, document indexes,
  hashes, and a reproducibility manifest. Its `--verify-only` mode hashes the
  complete downloaded and processed asset set against that manifest without
  downloading or preprocessing it again.
- `validate_repo.py`: validate YAML syntax, local Markdown links, continuous and
  matching catalog IDs/titles, shared-workload row/status structure,
  AWS-then-Runpod ordering, single-provider placement, and compute-profile
  references in both the summary and detailed experiment sections. It also
  rejects legacy three-digit experiment IDs outside the append-only decision
  history. It prefers PyYAML and uses the workstation's Ruby YAML parser only as
  a bootstrap fallback before the Python development environment is installed.

Run a script with `--help` before use. AWS and Runpod provisioning and
stop/terminate/delete operations will be added only with explicit safeguards
against leaving billable resources running.

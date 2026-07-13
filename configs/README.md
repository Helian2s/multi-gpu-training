# Shared configuration

This directory holds configuration shared across comparable experiments.

`workload.example.yaml` is a proposal derived from the experiment catalog, not
an accepted executable configuration. After the remaining shared workload
proposals are accepted and exact revisions are pinned, create `workload.yaml`.
Recorded runs must copy the fully resolved configuration into their artifact
directory rather than relying on mutable defaults.

`inputs.lock.yaml` is accepted and pins the exact model, tokenizer, dataset, and
preprocessing identity used by `make prepare-inputs`. The large downloaded and
processed files remain under ignored `data/` paths; the generated manifest is
the portable record that must accompany later uploads.

`provider.example.yaml` defines the provider-neutral fields that AWS and Runpod
adapters must resolve for each compute session. Its `provider.profile` refers to
an accepted compute profile in `../EXPERIMENT_CATALOG.md`; the adapter expands
that profile into the concrete resource metadata preserved with the run.

Secrets, AWS credentials, Runpod API keys, GitHub tokens, and registry
credentials must never be stored here.

# Provider infrastructure

This directory isolates cloud-provider operations from experiment logic. AWS
and Runpod adapters must implement the same lifecycle:

1. Validate configuration, credentials, quota/capacity assumptions, and budget.
2. Provision or start exactly one physical GPU host.
3. Record provider metadata and run mandatory qualification.
4. Pull the same immutable image content from the provider's registry: ECR on
   AWS or the GHCR mirror on Runpod; stage the same pinned inputs.
5. Apply the accepted compute profile and any explicitly documented visibility
   mask, then launch the container.
6. Stage artifacts to durable storage and verify their manifest/checksums.
7. Stop or terminate the resource even when launch, training, or collection
   fails.

Provider adapters may resolve storage URIs and metadata, but may not alter an
experiment's model, dataset, batch geometry, precision, launcher semantics, or
measurement window.

## Shared adapter contract

Future automation under `infra/aws/` and `infra/runpod/` must expose equivalent
operations such as `validate`, `provision`, `qualify`, `stage-in`, `run`,
`stage-out`, `stop`, and `status`. Mutating operations must support a dry-run or
explicit confirmation and enforce maximum lifetime/cost safeguards.

Resolved provider configuration follows `../configs/provider.example.yaml`.
Secrets are supplied by the provider's credential mechanism and never written
to a resolved configuration or run manifest.

Every numbered experiment has exactly one provider. Related AWS PCIe and Runpod
NVLink questions use separate IDs, and results retain their complete hardware
environment. A GPU-count scaling experiment may use sequential instance sizes
from one provider, but no run spans hosts. Provider-neutral code and image
content enable portability; they do not make cross-provider numbers
interchangeable.

Current workstation readiness and credential checks are tracked separately in
[TOOLING.md](TOOLING.md); a project decision must not claim that a tool is
installed merely because it is approved.

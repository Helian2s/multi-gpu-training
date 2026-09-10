# Configuration files

Experiments combine a shared input identity with their own executable workload
variants. Provider queues select where those variants run.

| File | Role |
| --- | --- |
| [inputs.lock.yaml](inputs.lock.yaml) | Accepted model, tokenizer, dataset, and preprocessing revisions used by the preparation script |
| [workload.example.yaml](workload.example.yaml) | Proposed shared-workload example; not the configuration consumed by current experiment runners |
| [provider.example.yaml](provider.example.yaml) | Provider-neutral reference contract; not a launch-ready account configuration |

Current executable workload settings live in each experiment's
`experiment.yaml`, including overrides, variants, image references, and output
paths. AWS and Runpod queue YAML files under [infra/](../infra/README.md)
contain the concrete launch/run-unit configuration.

The accepted input lock is implemented by [prepare_inputs.py](../scripts/prepare_inputs.py).
The current synthetic EXP-11–14 workloads generate their own tensors; linking
the shared input contract does not mean those workloads train Qwen. This gap
is recorded in [validation status](../docs/validation-status.md).

Default precision and evaluation-boundary proposals remain governed by their
status in the [catalog](../EXPERIMENT_CATALOG.md). An example file does not
promote a proposal into an accepted project decision.

Treat checked-in provider resource IDs and image digests as historical
configuration until revalidated. Keep credentials and resolved secrets outside
configuration files and Git.

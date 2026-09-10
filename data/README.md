# Model and dataset preparation

[prepare_inputs.py](../scripts/prepare_inputs.py) prepares the accepted
**Qwen3-1.7B-Base** model/tokenizer and **WikiText-103 raw** train, validation,
and test splits. Exact revisions and preprocessing rules live in
[inputs.lock.yaml](../configs/inputs.lock.yaml).

## Implemented pipeline

1. Fetch the pinned model, tokenizer, and dataset snapshots.
2. Tokenize source records in order and append one EOS to each non-empty record.
3. Preserve empty records in the document index without adding tokens.
4. Write a canonical token stream, document offsets, and document hashes for
   each split.
5. Inventory downloaded and processed files with sizes and SHA-256 checksums
   in a manifest.

Tokens use little-endian `uint32`; document offsets use little-endian `uint64`.
The PyTorch training executor slices fixed-length samples from the train stream.
The current synthetic parallelism executor generates tensors directly and does
not consume this dataset or the Qwen checkpoint.

## Prepare or verify inputs

The full preparation environment is separate from the minimal local-check
setup. From the repository root:

```bash
make prepare-environment
make prepare-inputs
make verify-inputs
```

`make prepare-environment` creates or extends `.venv` using Python 3.12 and
[requirements-preparation.txt](../requirements-preparation.txt).
`make prepare-inputs` downloads model/data assets and writes processed files;
allow disk space for both source snapshots and generated outputs.

`make verify-inputs` checks every inventoried file against its size and checksum
without downloading or preprocessing again. Run it after transferring inputs
between workstations or before staging them to a GPU host. Runtime presence
checks alone do not verify the complete input inventory.

## Storage and reproducibility

Downloaded assets under `raw/` and generated assets under `processed/` are
ignored by Git. A clone contains the preparation code and lock file, not these
assets. The July AWS queues staged inputs from S3; exact locations are recorded
in provider configuration and [the handoff](../HANDOFF.md).

The preprocessing contract makes source token streams reproducible. Training
sample partitioning and label alignment have separate known issues documented
in [validation status](../docs/validation-status.md); input checksum success
does not resolve them.

# Dataset workspace

Dataset code and metadata are tracked; downloaded or generated data are not.

The accepted end-to-end input workload uses the `Salesforce/wikitext`
`wikitext-103-raw-v1` splits and the tokenizer distributed with
`Qwen/Qwen3-1.7B-Base`. Exact revisions and the preprocessing contract are
pinned in `../configs/inputs.lock.yaml`. The preparation pipeline:

1. Download exact model, tokenizer, and dataset revisions.
2. Preserve license and attribution metadata.
3. Tokenize deterministically and insert EOS between documents.
4. Concatenate the records into one canonical token stream per source split;
   fixed-length experiment samples are deterministic slices of that stream.
5. Write a manifest containing revisions, parameters, counts, and content
   hashes.
6. Expose identical sample IDs and token sequences to PyTorch and Megatron
   loaders.

Use `raw/` for immutable fetched inputs, `processed/` for canonical token
streams, and `cache/` or `downloads/` for disposable transfer caches. These
directories are ignored by Git. Durable copies live in S3 for AWS or on a
Runpod network volume; performance-sensitive working copies may be staged to
EC2 instance-store/EBS or Pod-local storage.

Run `make prepare-environment` once and then `make prepare-inputs`. Empty source
rows are retained in the document offset/hash index but contribute no tokens;
non-empty records receive one EOS. Processed output contains little-endian
`uint32` token streams, `uint64` document offsets, per-document SHA-256 values,
and a manifest with source/output checksums. Model, dataset, and generated files
are intentionally not committed.

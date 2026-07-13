# Dataset workspace

Dataset code and metadata are tracked; downloaded or generated data are not.

The proposed end-to-end workload would use the `Salesforce/wikitext`
`wikitext-103-raw-v1` splits and the tokenizer distributed with the proposed
Qwen model. If the proposal is accepted, the preparation pipeline will:

1. Download exact model, tokenizer, and dataset revisions.
2. Preserve license and attribution metadata.
3. Tokenize deterministically and insert EOS between documents.
4. Pack fixed-length sequences without project-specific curation.
5. Write a manifest containing revisions, parameters, counts, and content
   hashes.
6. Expose identical sample IDs and token sequences to PyTorch and Megatron
   loaders.

Use `downloads/` for source caches, `raw/` for immutable fetched inputs,
`processed/` for the canonical packed token stream, and `cache/` for disposable
indexes. These directories are ignored by Git. Durable copies live in S3 for
AWS or on a Runpod network volume; performance-sensitive working copies may be
staged to EC2 instance-store/EBS or Pod-local storage.

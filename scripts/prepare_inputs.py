#!/usr/bin/env python3
"""Download and tokenize the immutable shared model and WikiText inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
from array import array
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "inputs.lock.yaml"
SPLITS = ("train", "validation", "test")


class PreparationError(RuntimeError):
    """Raised when an input contract or local artifact is invalid."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pinned inputs and build canonical token streams."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Accepted immutable input lock file.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--download-only", action="store_true", help="Download without tokenizing."
    )
    mode.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Tokenize previously downloaded inputs.",
    )
    mode.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify downloaded and processed inputs against their manifest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of source records tokenized per batch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing processed output with the same identity.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise PreparationError(
            "PyYAML is required; install requirements-preparation.txt"
        ) from error

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("status") != "accepted":
        raise PreparationError(f"input lock is not accepted: {path}")
    for section in ("model", "dataset", "preprocessing"):
        if not isinstance(data.get(section), dict):
            raise PreparationError(f"input lock has no {section!r} mapping")
    return data


def safe_name(identifier: str) -> str:
    return identifier.replace("/", "--")


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    model = config["model"]
    dataset = config["dataset"]
    identity = (
        f"{safe_name(model['id'])}--{model['revision'][:12]}__"
        f"{safe_name(dataset['id'])}--{dataset['revision'][:12]}__"
        f"{dataset['configuration']}__{config['preprocessing']['version']}"
    )
    return {
        "cache": ROOT / "data" / "cache" / "huggingface",
        "model": (
            ROOT
            / "data"
            / "raw"
            / "models"
            / safe_name(model["id"])
            / model["revision"]
        ),
        "dataset": (
            ROOT
            / "data"
            / "raw"
            / "datasets"
            / safe_name(dataset["id"])
            / dataset["revision"]
        ),
        "processed": ROOT / "data" / "processed" / identity,
    }


def download_inputs(config: dict[str, Any], paths: dict[str, Path]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as error:
        raise PreparationError(
            "huggingface_hub is required; install requirements-preparation.txt"
        ) from error

    model = config["model"]
    dataset = config["dataset"]
    paths["cache"].mkdir(parents=True, exist_ok=True)
    paths["model"].mkdir(parents=True, exist_ok=True)
    paths["dataset"].mkdir(parents=True, exist_ok=True)

    print(f"Downloading model {model['id']}@{model['revision']}", flush=True)
    snapshot_download(
        repo_id=model["id"],
        revision=model["revision"],
        local_dir=paths["model"],
        cache_dir=paths["cache"],
    )

    configuration = dataset["configuration"]
    print(f"Downloading dataset {dataset['id']}@{dataset['revision']}", flush=True)
    snapshot_download(
        repo_id=dataset["id"],
        repo_type="dataset",
        revision=dataset["revision"],
        local_dir=paths["dataset"],
        cache_dir=paths["cache"],
        allow_patterns=[
            ".gitattributes",
            "README.md",
            "LICENSE*",
            f"{configuration}/*",
        ],
    )


def little_endian_bytes(values: Sequence[int], typecode: str) -> bytes:
    result = array(typecode, values)
    expected_size = {"I": 4, "Q": 8}[typecode]
    if result.itemsize != expected_size:
        raise PreparationError(
            f"unexpected array item size for {typecode}: {result.itemsize}"
        )
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def write_tokenized_split(
    text_batches: Iterable[Sequence[str]],
    tokenizer: Any,
    output_directory: Path,
    split: str,
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    final_tokens = output_directory / f"{split}.tokens.bin"
    final_offsets = output_directory / f"{split}.documents.idx"
    final_hashes = output_directory / f"{split}.document_hashes.bin"
    temporary_tokens = final_tokens.with_suffix(final_tokens.suffix + ".tmp")
    temporary_offsets = final_offsets.with_suffix(final_offsets.suffix + ".tmp")
    temporary_hashes = final_hashes.with_suffix(final_hashes.suffix + ".tmp")

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise PreparationError("the selected tokenizer has no EOS token ID")
    if not 0 <= eos_token_id <= 0xFFFFFFFF:
        raise PreparationError(f"EOS token ID does not fit uint32: {eos_token_id}")

    offsets = [0]
    record_count = 0
    empty_record_count = 0
    token_count = 0
    tokens_digest = hashlib.sha256()
    hashes_digest = hashlib.sha256()

    try:
        with temporary_tokens.open("wb") as token_handle, temporary_hashes.open(
            "wb"
        ) as hash_handle:
            for texts in text_batches:
                if any(not isinstance(text, str) for text in texts):
                    raise PreparationError(f"split {split} contains a non-string value")
                encoded = tokenizer(
                    list(texts),
                    add_special_tokens=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )["input_ids"]
                if len(encoded) != len(texts):
                    raise PreparationError("tokenizer returned the wrong batch length")

                for text, token_ids in zip(texts, encoded, strict=True):
                    if text:
                        document_token_ids = [*token_ids, eos_token_id]
                    else:
                        empty_record_count += 1
                        document_token_ids = []
                    document_bytes = little_endian_bytes(document_token_ids, "I")
                    document_hash = hashlib.sha256(document_bytes).digest()
                    token_handle.write(document_bytes)
                    hash_handle.write(document_hash)
                    tokens_digest.update(document_bytes)
                    hashes_digest.update(document_hash)
                    token_count += len(document_token_ids)
                    record_count += 1
                    offsets.append(token_count)

                if record_count and record_count % 100_000 < len(texts):
                    print(
                        f"{split}: {record_count:,} records, {token_count:,} tokens",
                        flush=True,
                    )

        offsets_bytes = little_endian_bytes(offsets, "Q")
        temporary_offsets.write_bytes(offsets_bytes)
        os.replace(temporary_tokens, final_tokens)
        os.replace(temporary_offsets, final_offsets)
        os.replace(temporary_hashes, final_hashes)
    except BaseException:
        for path in (temporary_tokens, temporary_offsets, temporary_hashes):
            path.unlink(missing_ok=True)
        raise

    return {
        "records": record_count,
        "empty_records": empty_record_count,
        "tokens_including_eos": token_count,
        "files": {
            final_tokens.name: {
                "bytes": final_tokens.stat().st_size,
                "sha256": tokens_digest.hexdigest(),
            },
            final_offsets.name: {
                "bytes": final_offsets.stat().st_size,
                "sha256": hashlib.sha256(offsets_bytes).hexdigest(),
            },
            final_hashes.name: {
                "bytes": final_hashes.stat().st_size,
                "sha256": hashes_digest.hexdigest(),
            },
        },
    }


def parquet_text_batches(
    dataset_directory: Path,
    configuration: str,
    split: str,
    text_field: str,
    batch_size: int,
) -> Iterator[list[str]]:
    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise PreparationError(
            "pyarrow is required; install requirements-preparation.txt"
        ) from error

    shard_directory = dataset_directory / configuration
    shards = sorted(shard_directory.glob(f"{split}-*.parquet"))
    if not shards:
        raise PreparationError(f"no Parquet shards found for split {split}")
    for shard in shards:
        parquet_file = parquet.ParquetFile(shard)
        if text_field not in parquet_file.schema.names:
            raise PreparationError(f"{shard} has no {text_field!r} column")
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=[text_field]
        ):
            yield batch.column(0).to_pylist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or ".cache" in path.parts:
            continue
        records.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def package_versions() -> dict[str, str]:
    packages = (
        "huggingface_hub",
        "transformers",
        "tokenizers",
        "pyarrow",
        "safetensors",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def compare_inventory(
    label: str, directory: Path, expected: Any
) -> list[dict[str, Any]]:
    if not directory.is_dir():
        raise PreparationError(f"{label} directory does not exist: {directory}")
    if not isinstance(expected, list):
        raise PreparationError(f"manifest has no valid {label} inventory")
    if any(
        not isinstance(record, dict)
        or not isinstance(record.get("path"), str)
        or not isinstance(record.get("bytes"), int)
        or not isinstance(record.get("sha256"), str)
        for record in expected
    ):
        raise PreparationError(f"manifest has an invalid {label} inventory record")
    expected_paths = [record["path"] for record in expected]
    if len(expected_paths) != len(set(expected_paths)):
        raise PreparationError(f"manifest has duplicate {label} inventory paths")
    actual = inventory(directory)
    if actual != expected:
        expected_by_path = {record["path"]: record for record in expected}
        actual_by_path = {record["path"]: record for record in actual}
        missing = sorted(set(expected_by_path) - set(actual_by_path))
        unexpected = sorted(set(actual_by_path) - set(expected_by_path))
        changed = sorted(
            path
            for path in set(expected_by_path) & set(actual_by_path)
            if expected_by_path[path] != actual_by_path[path]
        )
        details = []
        if missing:
            details.append(f"missing={missing[:3]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:3]}")
        if changed:
            details.append(f"changed={changed[:3]}")
        suffix = f": {', '.join(details)}" if details else ""
        raise PreparationError(f"{label} inventory verification failed{suffix}")
    return actual


def verify_processed_outputs(
    processed_directory: Path, splits: Any
) -> list[Path]:
    if not processed_directory.is_dir():
        raise PreparationError(
            f"processed directory does not exist: {processed_directory}"
        )
    if not isinstance(splits, dict):
        raise PreparationError("manifest has no valid split records")

    expected_files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_record = splits.get(split)
        if not isinstance(split_record, dict) or not isinstance(
            split_record.get("files"), dict
        ):
            raise PreparationError(f"manifest has no valid {split} file records")
        for filename, file_record in split_record["files"].items():
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(file_record, dict)
                or filename in expected_files
            ):
                raise PreparationError(
                    f"manifest has an invalid processed filename: {filename!r}"
                )
            expected_files[filename] = file_record

    actual_files = sorted(
        path
        for path in processed_directory.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    actual_names = {path.name for path in actual_files}
    expected_names = set(expected_files)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise PreparationError(
            "processed output set verification failed: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )

    for path in actual_files:
        expected = expected_files[path.name]
        if path.stat().st_size != expected.get("bytes"):
            raise PreparationError(f"processed output size mismatch: {path.name}")
        if sha256_file(path) != expected.get("sha256"):
            raise PreparationError(f"processed output hash mismatch: {path.name}")
    return actual_files


def verify_inputs(config: dict[str, Any], paths: dict[str, Path]) -> Path:
    manifest_path = paths["processed"] / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise PreparationError(
            f"processed manifest does not exist: {manifest_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise PreparationError(
            f"processed manifest is invalid: {manifest_path}"
        ) from error
    if not isinstance(manifest, dict) or manifest.get("input_lock") != config:
        raise PreparationError("processed manifest does not match the accepted input lock")

    source_files = manifest.get("source_files")
    if not isinstance(source_files, dict):
        raise PreparationError("manifest has no valid source-file inventories")
    model_files = compare_inventory("model", paths["model"], source_files.get("model"))
    dataset_files = compare_inventory(
        "dataset", paths["dataset"], source_files.get("dataset")
    )
    processed_files = verify_processed_outputs(
        paths["processed"], manifest.get("splits")
    )
    print(
        "Input verification passed: "
        f"{len(model_files)} model files, {len(dataset_files)} dataset files, "
        f"{len(processed_files)} processed files",
        flush=True,
    )
    return manifest_path


def preprocess_inputs(
    config: dict[str, Any],
    paths: dict[str, Path],
    batch_size: int,
    force: bool,
) -> Path:
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise PreparationError(
            "transformers is required; install requirements-preparation.txt"
        ) from error

    if batch_size <= 0:
        raise PreparationError("batch size must be positive")
    manifest_path = paths["processed"] / "manifest.json"
    if manifest_path.exists() and not force:
        print(f"Processed inputs already exist: {manifest_path}", flush=True)
        return manifest_path
    if not paths["model"].is_dir() or not paths["dataset"].is_dir():
        raise PreparationError("pinned model and dataset must be downloaded first")

    paths["processed"].mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        paths["model"], local_files_only=True, trust_remote_code=False, use_fast=True
    )
    preprocessing = config["preprocessing"]
    dataset = config["dataset"]
    split_records = {}
    for split in SPLITS:
        print(f"Tokenizing {split}", flush=True)
        split_records[split] = write_tokenized_split(
            parquet_text_batches(
                paths["dataset"],
                dataset["configuration"],
                split,
                preprocessing["text_field"],
                batch_size,
            ),
            tokenizer,
            paths["processed"],
            split,
        )

    manifest = {
        "schema_version": 1,
        "input_lock": config,
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "vocabulary_size": len(tokenizer),
        },
        "format": {
            "tokens": "headerless little-endian uint32 token IDs",
            "document_offsets": (
                "headerless little-endian uint64 cumulative offsets; "
                "record i is offsets[i]:offsets[i+1]"
            ),
            "document_hashes": (
                "32-byte SHA-256 values over each record's token bytes; "
                "non-empty records include EOS and empty records hash zero bytes"
            ),
        },
        "splits": split_records,
        "source_files": {
            "model": inventory(paths["model"]),
            "dataset": inventory(paths["dataset"]),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"Wrote manifest: {manifest_path}", flush=True)
    return manifest_path


def main() -> int:
    args = parse_args()
    try:
        config = load_config(args.config.resolve())
        paths = input_paths(config)
        if args.verify_only:
            verify_inputs(config, paths)
            return 0
        if not args.preprocess_only:
            download_inputs(config, paths)
        if not args.download_only:
            preprocess_inputs(config, paths, args.batch_size, args.force)
    except (PreparationError, OSError, ValueError) as error:
        print(f"input preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

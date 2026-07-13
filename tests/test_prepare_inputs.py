from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_inputs import (
    PreparationError,
    inventory,
    sha256_file,
    verify_inputs,
    write_tokenized_split,
)


class DummyTokenizer:
    eos_token_id = 99

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}


class WriteTokenizedSplitTest(unittest.TestCase):
    def test_preserves_empty_records_and_writes_little_endian_contract(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            result = write_tokenized_split(
                [["ab", ""]], DummyTokenizer(), output, "validation"
            )

            token_bytes = (output / "validation.tokens.bin").read_bytes()
            offsets_bytes = (output / "validation.documents.idx").read_bytes()
            document_hashes = (
                output / "validation.document_hashes.bin"
            ).read_bytes()

            self.assertEqual(struct.unpack("<3I", token_bytes), (97, 98, 99))
            self.assertEqual(struct.unpack("<3Q", offsets_bytes), (0, 3, 3))
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["empty_records"], 1)
            self.assertEqual(result["tokens_including_eos"], 3)
            self.assertEqual(
                document_hashes[:32], hashlib.sha256(token_bytes[:12]).digest()
            )
            self.assertEqual(
                document_hashes[32:], hashlib.sha256(b"").digest()
            )


class VerifyInputsTest(unittest.TestCase):
    def test_verifies_source_and_processed_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = {
                "model": root / "model",
                "dataset": root / "dataset",
                "processed": root / "processed",
            }
            for path in paths.values():
                path.mkdir()
            (paths["model"] / "weights.bin").write_bytes(b"model")
            (paths["dataset"] / "train.parquet").write_bytes(b"dataset")

            split_records = {}
            for split in ("train", "validation", "test"):
                output = paths["processed"] / f"{split}.tokens.bin"
                output.write_bytes(split.encode("utf-8"))
                split_records[split] = {
                    "files": {
                        output.name: {
                            "bytes": output.stat().st_size,
                            "sha256": sha256_file(output),
                        }
                    }
                }

            config = {
                "schema_version": 1,
                "status": "accepted",
                "model": {"id": "model", "revision": "revision"},
                "dataset": {
                    "id": "dataset",
                    "revision": "revision",
                    "configuration": "configuration",
                },
                "preprocessing": {"version": "version"},
            }
            manifest = {
                "input_lock": config,
                "source_files": {
                    "model": inventory(paths["model"]),
                    "dataset": inventory(paths["dataset"]),
                },
                "splits": split_records,
            }
            (paths["processed"] / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )

            self.assertEqual(
                verify_inputs(config, paths), paths["processed"] / "manifest.json"
            )
            (paths["model"] / "weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(
                PreparationError, "model inventory verification failed"
            ):
                verify_inputs(config, paths)


if __name__ == "__main__":
    unittest.main()

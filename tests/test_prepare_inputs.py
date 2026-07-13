from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_inputs import write_tokenized_split


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


if __name__ == "__main__":
    unittest.main()

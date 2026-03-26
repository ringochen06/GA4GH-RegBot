from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.main import RegBot


class _FakeEmbeddings:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):  # noqa: ARG002
        pass

    def encode(self, texts, normalize_embeddings=True):  # noqa: ARG002
        if isinstance(texts, str):
            n = 1
        else:
            n = len(texts)
        return _FakeEmbeddings([[1.0] * 384 for _ in range(n)])


class TestRegBotPipeline(unittest.TestCase):
    @patch("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)
    def test_ingest_retrieve_check_smoke(self) -> None:
        policy = Path(__file__).resolve().parents[1] / "examples" / "data" / "sample_ga4gh_policy_stub.txt"
        consent = Path(__file__).resolve().parents[1] / "examples" / "data" / "sample_consent_short.txt"
        self.assertTrue(policy.is_file(), "missing examples/data sample policy file")

        with tempfile.TemporaryDirectory() as tmp:
            bot = RegBot(store_dir=tmp, api_key="")
            self.assertTrue(bot.ingest_policy_documents(str(policy), reset=True))
            hits = bot.retrieve_relevant_clauses("international data sharing withdrawal", top_k=4)
            self.assertGreater(len(hits), 0)
            self.assertIn("id", hits[0])
            with open(consent, encoding="utf-8") as f:
                report = bot.check_compliance(f.read(), top_k=6)
            self.assertIn("status", report)
            self.assertIn("citations", report)


if __name__ == "__main__":
    unittest.main()

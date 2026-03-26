import unittest

from src.regbot.grounding import audit_citation_grounding, allowed_chunk_ids


class TestGrounding(unittest.TestCase):
    def test_allowed_ids(self) -> None:
        chunks = [{"id": "a1"}, {"id": "b2"}]
        self.assertEqual(allowed_chunk_ids(chunks), {"a1", "b2"})

    def test_audit_rejects_unknown_chunk(self) -> None:
        allow = {"c1", "c2"}
        report = {
            "recommendations": ["do something"],
            "citations": [{"chunk_id": "nope", "reason": "x"}],
        }
        audit = audit_citation_grounding(report, allow, strict_min_citations_vs_recommendations=True)
        self.assertFalse(audit["ok"])
        self.assertTrue(any("Invalid chunk_id" in i for i in audit["issues"]))

    def test_strict_requires_enough_citations(self) -> None:
        allow = {"c1"}
        report = {
            "recommendations": ["r1", "r2"],
            "citations": [{"chunk_id": "c1", "reason": "only one"}],
        }
        audit = audit_citation_grounding(report, allow, strict_min_citations_vs_recommendations=True)
        self.assertFalse(audit["ok"])


if __name__ == "__main__":
    unittest.main()

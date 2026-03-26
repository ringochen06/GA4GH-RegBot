import unittest

from src.main import RegBot


class TestRegBotSmoke(unittest.TestCase):
    def test_instantiates(self) -> None:
        bot = RegBot(api_key="test")
        self.assertEqual(bot.api_key, "test")

    def test_placeholder_methods_return_expected_types(self) -> None:
        bot = RegBot(api_key="test")
        self.assertIsInstance(bot.ingest_policy_documents("dummy.pdf"), bool)
        clauses = bot.retrieve_relevant_clauses("dummy query")
        self.assertIsInstance(clauses, list)
        self.assertTrue(all(isinstance(c, str) for c in clauses))
        report = bot.check_compliance("dummy consent form")
        self.assertIsInstance(report, dict)
        self.assertIn("status", report)


if __name__ == "__main__":
    unittest.main()


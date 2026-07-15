import os
import unittest
from unittest.mock import patch

from grok_search.config import config


class ConfigTests(unittest.TestCase):
    def test_grok_api_mode_accepts_supported_values(self):
        for mode in ("auto", "chat_completions", "responses"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"GROK_API_MODE": mode}):
                self.assertEqual(config.grok_api_mode, mode)

    def test_grok_api_mode_rejects_unknown_value(self):
        with patch.dict(os.environ, {"GROK_API_MODE": "legacy"}):
            with self.assertRaisesRegex(ValueError, "GROK_API_MODE"):
                _ = config.grok_api_mode

    def test_responses_read_timeout_defaults_to_300_seconds(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(config.responses_read_timeout, 300.0)

    def test_responses_read_timeout_accepts_positive_number(self):
        with patch.dict(os.environ, {"GROK_RESPONSES_READ_TIMEOUT": "450.5"}):
            self.assertEqual(config.responses_read_timeout, 450.5)

    def test_responses_read_timeout_rejects_invalid_values(self):
        for value in ("invalid", "0", "-1"):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"GROK_RESPONSES_READ_TIMEOUT": value},
            ):
                with self.assertRaisesRegex(ValueError, "GROK_RESPONSES_READ_TIMEOUT"):
                    _ = config.responses_read_timeout

    def test_responses_effort_is_unset_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(config.responses_effort)

    def test_responses_effort_accepts_supported_values(self):
        for value, expected in (("low", "low"), (" XHIGH ", "xhigh")):
            with self.subTest(value=value), patch.dict(
                os.environ,
                {"GROK_RESPONSES_EFFORT": value},
            ):
                self.assertEqual(config.responses_effort, expected)

    def test_responses_effort_rejects_unknown_value(self):
        with patch.dict(os.environ, {"GROK_RESPONSES_EFFORT": "extreme"}):
            with self.assertRaisesRegex(ValueError, "GROK_RESPONSES_EFFORT"):
                _ = config.responses_effort


if __name__ == "__main__":
    unittest.main()

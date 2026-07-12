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


if __name__ == "__main__":
    unittest.main()

import json
import unittest

from grok_search.providers.base import SearchResponse
from grok_search.providers.grok import GrokAPIError, GrokSearchProvider


class FakeStreamResponse:
    def __init__(self, lines: list[str], status_code: int = 200):
        self._lines = lines
        self.status_code = status_code

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class GrokProviderTests(unittest.TestCase):
    def test_auto_mode_selects_responses_for_multi_agent(self):
        multi_agent = GrokSearchProvider("https://example.test/v1", "key", "grok-multi-agent")
        standard = GrokSearchProvider("https://example.test/v1", "key", "grok-fast")

        self.assertEqual(multi_agent._resolved_api_mode(), "responses")
        self.assertEqual(standard._resolved_api_mode(), "chat_completions")

    def test_parse_responses_payload_extracts_text_and_deduplicated_sources(self):
        payload = {
            "status": "completed",
            "output": [
                {
                    "type": "web_search_call",
                    "action": {
                        "sources": [
                            {"url": "https://example.com/a", "title": "Source A"},
                            {"url": "https://example.com/b", "title": "Source B"},
                        ]
                    },
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "First paragraph.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url": "https://example.com/a",
                                    "title": "Duplicate A",
                                }
                            ],
                        },
                        {
                            "type": "output_text",
                            "text": " Second paragraph.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "url": "https://example.com/c",
                                        "title": "Source C",
                                    },
                                }
                            ],
                        },
                    ],
                },
            ],
        }

        result = GrokSearchProvider._parse_responses_payload(payload)

        self.assertEqual(result.content, "First paragraph. Second paragraph.")
        self.assertEqual(
            [source["url"] for source in result.sources],
            [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
        )
        self.assertTrue(all(source["provider"] == "grok" for source in result.sources))

    def test_parse_responses_payload_raises_explicit_api_error(self):
        with self.assertRaisesRegex(GrokAPIError, "rate_limit_exceeded"):
            GrokSearchProvider._parse_responses_payload(
                {
                    "error": {
                        "type": "rate_limit_exceeded",
                        "code": "rate_limit_exceeded",
                        "message": "No available accounts",
                    }
                }
            )

    def test_parse_responses_payload_rejects_empty_output(self):
        with self.assertRaisesRegex(GrokAPIError, "no output text"):
            GrokSearchProvider._parse_responses_payload(
                {
                    "status": "incomplete",
                    "output": [],
                    "incomplete_details": {"reason": "max_tokens"},
                }
            )


class GrokProviderAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_builds_responses_web_search_payload(self):
        provider = GrokSearchProvider(
            "https://example.test/v1",
            "key",
            "grok-4.20-multi-agent-0309",
            api_mode="auto",
        )
        captured = {}

        async def fake_execute(headers, payload, ctx):
            captured.update(payload)
            return SearchResponse(content="answer", sources=[])

        provider._execute_responses_with_retry = fake_execute

        result = await provider.search("latest Python news", platform="GitHub")

        self.assertEqual(result.content, "answer")
        self.assertEqual(captured["model"], "grok-4.20-multi-agent-0309")
        self.assertEqual(captured["tools"], [{"type": "web_search"}])
        self.assertIn("latest Python news", captured["input"])
        self.assertIn("GitHub", captured["input"])
        self.assertNotIn("messages", captured)

    async def test_chat_stream_ignores_null_content_and_collects_text(self):
        response = FakeStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":null}}]}',
                'data: {"choices":[{"delta":{"content":"Hello"}}]}',
                'data: {"choices":[{"delta":{"content":" world"}}]}',
                "data: [DONE]",
            ]
        )
        provider = GrokSearchProvider("https://example.test/v1", "key")

        self.assertEqual(await provider._parse_streaming_response(response), "Hello world")

    async def test_chat_stream_surfaces_embedded_error_event(self):
        error_event = {
            "error": {
                "type": "rate_limit_exceeded",
                "code": "rate_limit_exceeded",
                "message": "No available accounts",
            }
        }
        response = FakeStreamResponse([f"data: {json.dumps(error_event)}", "data: [DONE]"])
        provider = GrokSearchProvider("https://example.test/v1", "key")

        with self.assertRaisesRegex(GrokAPIError, "No available accounts"):
            await provider._parse_streaming_response(response)


if __name__ == "__main__":
    unittest.main()

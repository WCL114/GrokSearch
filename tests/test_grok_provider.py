import json
import os
import unittest
from unittest.mock import patch

import httpx

from grok_search.providers.base import SearchResponse
from grok_search.providers.grok import GrokAPIError, GrokSearchProvider


class FakeStreamResponse:
    def __init__(
        self,
        lines: list[str],
        status_code: int = 200,
        failure: Exception | None = None,
    ):
        self._lines = lines
        self.status_code = status_code
        self.failure = failure
        self.is_error = status_code >= 400

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        if self.failure:
            raise self.failure

    async def aread(self):
        return b""


class FakeStreamContext:
    def __init__(self, response: FakeStreamResponse):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeAsyncClient:
    def __init__(self, responses: list[FakeStreamResponse]):
        self.responses = responses
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def stream(self, method, url, headers, json):
        response = self.responses[self.calls]
        self.calls += 1
        return FakeStreamContext(response)


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

    def test_parse_responses_payload_returns_explicit_api_error(self):
        result = GrokSearchProvider._parse_responses_payload(
            {
                "error": {
                    "type": "rate_limit_exceeded",
                    "code": "rate_limit_exceeded",
                    "message": "No available accounts",
                }
            }
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("rate_limit_exceeded", result.error)

    def test_parse_responses_payload_preserves_incomplete_status(self):
        result = GrokSearchProvider._parse_responses_payload(
            {
                "status": "incomplete",
                "output": [],
                "incomplete_details": {"reason": "max_tokens"},
            }
        )
        self.assertEqual(result.status, "incomplete")
        self.assertIn("max_tokens", result.error)


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

        with patch.dict(os.environ, {"GROK_RESPONSES_EFFORT": ""}):
            result = await provider.search("latest Python news", platform="GitHub")

        self.assertEqual(result.content, "answer")
        self.assertEqual(captured["model"], "grok-4.20-multi-agent-0309")
        self.assertEqual(captured["tools"], [{"type": "web_search"}])
        self.assertEqual(captured["reasoning"], {"effort": "high"})
        self.assertTrue(captured["stream"])
        self.assertIn("latest Python news", captured["input"])
        self.assertIn("GitHub", captured["input"])
        self.assertNotIn("messages", captured)

    async def test_search_passes_explicit_responses_effort(self):
        provider = GrokSearchProvider(
            "https://example.test/v1",
            "key",
            "grok-multi-agent",
            api_mode="responses",
        )
        captured = {}

        async def fake_execute(headers, payload, ctx):
            captured.update(payload)
            return SearchResponse(content="answer")

        provider._execute_responses_with_retry = fake_execute

        with patch.dict(os.environ, {"GROK_RESPONSES_EFFORT": ""}):
            await provider.search("deep research", effort="xhigh")

        self.assertEqual(captured["reasoning"], {"effort": "xhigh"})

    async def test_configured_responses_effort_overrides_request_value(self):
        provider = GrokSearchProvider(
            "https://example.test/v1",
            "key",
            "grok-multi-agent",
            api_mode="responses",
        )
        captured = {}

        async def fake_execute(headers, payload, ctx):
            captured.update(payload)
            return SearchResponse(content="answer")

        provider._execute_responses_with_retry = fake_execute

        with patch.dict(os.environ, {"GROK_RESPONSES_EFFORT": "xhigh"}):
            await provider.search("deep research", effort="low")

        self.assertEqual(captured["reasoning"], {"effort": "xhigh"})

    async def test_invalid_configured_responses_effort_is_rejected_before_request(self):
        provider = GrokSearchProvider(
            "https://example.test/v1",
            "key",
            "grok-multi-agent",
            api_mode="responses",
        )
        called = False

        async def fake_execute(headers, payload, ctx):
            nonlocal called
            called = True
            return SearchResponse(content="answer")

        provider._execute_responses_with_retry = fake_execute

        with patch.dict(os.environ, {"GROK_RESPONSES_EFFORT": "extreme"}):
            with self.assertRaisesRegex(ValueError, "GROK_RESPONSES_EFFORT"):
                await provider.search("query")
        self.assertFalse(called)

    async def test_configured_responses_effort_does_not_affect_chat_mode(self):
        provider = GrokSearchProvider(
            "https://example.test/v1",
            "key",
            "grok-fast",
            api_mode="chat_completions",
        )
        captured = {}

        async def fake_execute(headers, payload, ctx):
            captured.update(payload)
            return "answer"

        provider._execute_stream_with_retry = fake_execute

        with patch.dict(os.environ, {"GROK_RESPONSES_EFFORT": "extreme"}):
            result = await provider.search("query")

        self.assertEqual(result.content, "answer")
        self.assertNotIn("reasoning", captured)

    async def test_search_rejects_invalid_effort_before_request(self):
        provider = GrokSearchProvider(
            "https://example.test/v1",
            "key",
            api_mode="responses",
        )
        called = False

        async def fake_execute(headers, payload, ctx):
            nonlocal called
            called = True
            return SearchResponse(content="answer")

        provider._execute_responses_with_retry = fake_execute

        with self.assertRaisesRegex(ValueError, "effort"):
            await provider.search("query", effort="extreme")
        self.assertFalse(called)

    async def test_responses_stream_collects_deltas_sources_and_status(self):
        response = FakeStreamResponse(
            [
                'event: response.output_text.delta',
                'data: {"type":"response.output_text.delta","delta":"Hello "}',
                'data: {"type":"response.output_text.annotation.added","annotation":{"type":"url_citation","url":"https://example.com/a","title":"Source A"}}',
                'data: {"type":"response.output_text.delta","delta":"world"}',
                'data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hello world","annotations":[{"type":"url_citation","url":"https://example.com/a"}]}]}]}}',
            ]
        )
        provider = GrokSearchProvider("https://example.test/v1", "key")

        result = await provider._parse_responses_stream(response)

        self.assertEqual(result.content, "Hello world")
        self.assertEqual(result.status, "completed")
        self.assertEqual([source["url"] for source in result.sources], ["https://example.com/a"])

    async def test_responses_stream_preserves_partial_result_after_disconnect(self):
        response = FakeStreamResponse(
            ['data: {"type":"response.output_text.delta","delta":"partial"}'],
            failure=httpx.ReadError("connection lost"),
        )
        provider = GrokSearchProvider("https://example.test/v1", "key")

        result = await provider._parse_responses_stream(response)

        self.assertEqual(result.content, "partial")
        self.assertEqual(result.status, "incomplete")
        self.assertIn("connection lost", result.error)

    async def test_responses_stream_falls_back_to_full_json(self):
        response = FakeStreamResponse(
            [
                '{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"fallback","annotations":[]}]}]}'
            ]
        )
        provider = GrokSearchProvider("https://example.test/v1", "key")

        result = await provider._parse_responses_stream(response)

        self.assertEqual(result.content, "fallback")
        self.assertEqual(result.status, "completed")

    async def test_responses_stream_collects_tool_sources(self):
        response = FakeStreamResponse(
            [
                'data: {"type":"response.web_search_call.completed","action":{"sources":[{"url":"https://example.com/tool","title":"Tool Source"}]}}',
                'data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"answer","annotations":[]}]}]}}',
            ]
        )
        provider = GrokSearchProvider("https://example.test/v1", "key")

        result = await provider._parse_responses_stream(response)

        self.assertEqual(result.status, "completed")
        self.assertEqual(
            [source["url"] for source in result.sources],
            ["https://example.com/tool"],
        )

    async def test_responses_stream_returns_terminal_failures(self):
        cases = [
            (
                [
                    'data: {"type":"response.failed","response":{"status":"failed","error":{"code":"upstream_failed","message":"request failed"},"output":[]}}'
                ],
                "upstream_failed",
            ),
            (
                [
                    'data: {"type":"error","code":"rate_limit_exceeded","message":"No available accounts"}'
                ],
                "rate_limit_exceeded",
            ),
        ]
        provider = GrokSearchProvider("https://example.test/v1", "key")

        for lines, expected_error in cases:
            with self.subTest(expected_error=expected_error):
                result = await provider._parse_responses_stream(
                    FakeStreamResponse(lines)
                )
                self.assertEqual(result.status, "failed")
                self.assertIn(expected_error, result.error)

    async def test_responses_executor_retries_only_before_first_event(self):
        provider = GrokSearchProvider("https://example.test/v1", "key")
        completed_event = (
            'data: {"type":"response.completed","response":{"status":"completed",'
            '"output":[{"type":"message","content":[{"type":"output_text",'
            '"text":"complete","annotations":[]}]}]}}'
        )
        retrying_client = FakeAsyncClient(
            [
                FakeStreamResponse([], failure=httpx.ReadError("before stream")),
                FakeStreamResponse([completed_event]),
            ]
        )

        with patch(
            "grok_search.providers.grok.httpx.AsyncClient",
            return_value=retrying_client,
        ), patch.dict(
            os.environ,
            {
                "GROK_RETRY_MAX_ATTEMPTS": "1",
                "GROK_RETRY_MULTIPLIER": "0",
                "GROK_RETRY_MAX_WAIT": "0",
            },
        ):
            result = await provider._execute_responses_with_retry({}, {"stream": True})

        self.assertEqual(result.status, "completed")
        self.assertEqual(retrying_client.calls, 2)

        interrupted_client = FakeAsyncClient(
            [
                FakeStreamResponse(
                    ['data: {"type":"response.output_text.delta","delta":"partial"}'],
                    failure=httpx.ReadError("after stream"),
                ),
                FakeStreamResponse([completed_event]),
            ]
        )
        with patch(
            "grok_search.providers.grok.httpx.AsyncClient",
            return_value=interrupted_client,
        ), patch.dict(os.environ, {"GROK_RETRY_MAX_ATTEMPTS": "1"}):
            result = await provider._execute_responses_with_retry({}, {"stream": True})

        self.assertEqual(result.status, "incomplete")
        self.assertEqual(interrupted_client.calls, 1)

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

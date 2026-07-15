import inspect
import unittest
from unittest.mock import AsyncMock, patch

from grok_search.providers.base import SearchResponse
from grok_search.server import web_search
from grok_search.sources import allocate_extra_sources


class ServerHelperTests(unittest.TestCase):
    def test_web_search_defaults_to_high_effort(self):
        effort = inspect.signature(web_search).parameters["effort"]
        self.assertEqual(effort.default, "high")

    def test_allocate_extra_sources(self):
        cases = [
            (0, True, True, (0, 0)),
            (5, True, False, (5, 0)),
            (5, False, True, (0, 5)),
            (5, True, True, (3, 2)),
            (4, True, True, (2, 2)),
            (5, False, False, (0, 0)),
        ]
        for total, has_tavily, has_firecrawl, expected in cases:
            with self.subTest(total=total, tavily=has_tavily, firecrawl=has_firecrawl):
                self.assertEqual(
                    allocate_extra_sources(total, has_tavily, has_firecrawl),
                    expected,
                )


class ServerWebSearchTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_search_passes_effort_and_returns_provider_status(self):
        provider_result = SearchResponse(
            content="partial answer",
            sources=[{"url": "https://example.com", "provider": "grok"}],
            status="incomplete",
            error="stream interrupted",
        )

        with patch.dict(
            "os.environ",
            {
                "GROK_API_URL": "https://example.test/v1",
                "GROK_API_KEY": "key",
                "GROK_MODEL": "grok-multi-agent",
                "TAVILY_ENABLED": "false",
            },
        ), patch(
            "grok_search.server.GrokSearchProvider"
        ) as provider_class:
            provider_class.return_value.search = AsyncMock(return_value=provider_result)

            result = await web_search("query", effort="high")

        provider_class.return_value.search.assert_awaited_once_with(
            "query",
            "",
            effort="high",
        )
        self.assertEqual(result["content"], "partial answer")
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["error"], "stream interrupted")
        self.assertEqual(result["sources_count"], 1)


if __name__ == "__main__":
    unittest.main()

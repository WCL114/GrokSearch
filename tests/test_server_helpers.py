import unittest

from grok_search.sources import allocate_extra_sources


class ServerHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Literal


SearchStatus = Literal["completed", "incomplete", "failed"]


@dataclass
class SearchResponse:
    """搜索 Provider 的统一结构化返回值。"""

    content: str
    sources: list[dict] = field(default_factory=list)
    status: SearchStatus = "completed"
    error: str | None = None


class SearchResult:
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "",
        published_date: str = "",
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date = published_date

    def to_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date,
        }


class BaseSearchProvider(ABC):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    @abstractmethod
    async def search(self, query: str, max_results: int = 5) -> SearchResponse:
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        pass

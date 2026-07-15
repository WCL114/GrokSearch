import httpx
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Literal, Optional
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential
from tenacity.wait import wait_base
from .base import BaseSearchProvider, SearchResponse
from ..utils import search_prompt, fetch_prompt, url_describe_prompt, rank_sources_prompt
from ..logger import log_info
from ..config import DEFAULT_RESPONSE_EFFORT, VALID_RESPONSE_EFFORTS, config


def get_local_time_info() -> str:
    """获取本地时间信息，用于注入到搜索查询中"""
    try:
        # 尝试获取系统本地时区
        local_tz = datetime.now().astimezone().tzinfo
        local_now = datetime.now(local_tz)
    except Exception:
        # 降级使用 UTC
        local_now = datetime.now(timezone.utc)

    # 格式化时间信息
    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday = weekdays_cn[local_now.weekday()]

    return (
        f"[Current Time Context]\n"
        f"- Date: {local_now.strftime('%Y-%m-%d')} ({weekday})\n"
        f"- Time: {local_now.strftime('%H:%M:%S')}\n"
        f"- Timezone: {local_now.tzname() or 'Local'}\n"
    )


def _needs_time_context(query: str) -> bool:
    """检查查询是否需要时间上下文"""
    # 中文时间相关关键词
    cn_keywords = [
        "当前", "现在", "今天", "明天", "昨天",
        "本周", "上周", "下周", "这周",
        "本月", "上月", "下月", "这个月",
        "今年", "去年", "明年",
        "最新", "最近", "近期", "刚刚", "刚才",
        "实时", "即时", "目前",
    ]
    # 英文时间相关关键词
    en_keywords = [
        "current", "now", "today", "tomorrow", "yesterday",
        "this week", "last week", "next week",
        "this month", "last month", "next month",
        "this year", "last year", "next year",
        "latest", "recent", "recently", "just now",
        "real-time", "realtime", "up-to-date",
    ]

    query_lower = query.lower()

    for keyword in cn_keywords:
        if keyword in query:
            return True

    for keyword in en_keywords:
        if keyword in query_lower:
            return True

    return False

RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
ResponseEffort = Literal["low", "medium", "high", "xhigh"]


class GrokAPIError(RuntimeError):
    """Grok 上游返回的可诊断错误。"""

    def __init__(self, message: str, status_code: int | None = None, code: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class _RetryableResponsesProtocolError(GrokAPIError):
    """Responses 流开始前出现的可重试协议错误。"""


def _api_error_from_payload(error: Any, status_code: int | None = None) -> GrokAPIError:
    if isinstance(error, dict):
        code = str(error.get("code") or error.get("type") or "")
        message = str(error.get("message") or code or "Unknown API error")
    else:
        code = ""
        message = str(error or "Unknown API error")

    prefix = f"Grok API error {status_code}" if status_code is not None else "Grok API error"
    if code:
        return GrokAPIError(f"{prefix} ({code}): {message}", status_code, code)
    return GrokAPIError(f"{prefix}: {message}", status_code, code)


def _api_error_from_response(response: httpx.Response) -> GrokAPIError:
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None

    error = payload.get("error") if isinstance(payload, dict) else None
    if error:
        return _api_error_from_payload(error, response.status_code)

    body = response.text.strip()
    if len(body) > 500:
        body = body[:500] + "..."
    return GrokAPIError(
        f"Grok API error {response.status_code}: {body or response.reason_phrase}",
        response.status_code,
    )


def _is_retryable_exception(exc) -> bool:
    """检查异常是否可重试"""
    if isinstance(exc, _RetryableResponsesProtocolError):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError, httpx.RemoteProtocolError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES
    if isinstance(exc, GrokAPIError):
        return exc.status_code in RETRYABLE_STATUS_CODES
    return False


def _append_grok_source(
    raw: Any,
    sources: list[dict],
    seen_urls: set[str],
) -> None:
    if not isinstance(raw, dict):
        return
    citation = raw.get("url_citation")
    if isinstance(citation, dict):
        raw = citation
    url = raw.get("url")
    if not isinstance(url, str) or not url.strip():
        return
    url = url.strip()
    if url in seen_urls:
        return
    seen_urls.add(url)
    item = {"url": url, "provider": "grok"}
    title = raw.get("title")
    if isinstance(title, str) and title.strip():
        item["title"] = title.strip()
    sources.append(item)


def _collect_responses_output(output_items: Any) -> tuple[str, list[dict]]:
    content_parts: list[str] = []
    sources: list[dict] = []
    seen_urls: set[str] = set()

    for output in output_items or []:
        if not isinstance(output, dict):
            continue
        if output.get("type") == "message":
            for item in output.get("content") or []:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if item.get("type") == "output_text" and isinstance(text, str):
                    content_parts.append(text)
                for annotation in item.get("annotations") or []:
                    _append_grok_source(annotation, sources, seen_urls)
        elif output.get("type") == "web_search_call":
            action = output.get("action")
            if isinstance(action, dict):
                for source in action.get("sources") or []:
                    _append_grok_source(source, sources, seen_urls)

    return "".join(content_parts).strip(), sources


def _responses_error_message(payload: dict) -> str | None:
    error = payload.get("error")
    if error:
        return str(_api_error_from_payload(error))

    details = payload.get("incomplete_details")
    if isinstance(details, dict):
        reason = details.get("reason")
        if reason:
            return f"Responses API incomplete: {reason}"
    if details:
        return f"Responses API incomplete: {details}"
    return None


def _parse_responses_payload_data(payload: dict) -> SearchResponse:
    content, sources = _collect_responses_output(payload.get("output"))
    raw_status = payload.get("status")
    error = _responses_error_message(payload)

    if raw_status == "incomplete":
        status = "incomplete"
        error = error or "Responses API returned an incomplete result"
    elif error or raw_status in {"failed", "cancelled"}:
        status = "failed"
        error = error or f"Responses API ended with status: {raw_status}"
    else:
        status = "completed"

    if not content and status == "completed":
        status_text = raw_status or "unknown"
        raise GrokAPIError(
            f"Responses API returned no output text (status={status_text})"
        )

    return SearchResponse(
        content=content,
        sources=sources,
        status=status,
        error=error,
    )


class _ResponsesStreamAccumulator:
    """累积 Responses SSE 事件，并保留可返回的部分结果。"""

    def __init__(self):
        self.received_event = False
        self.terminal = False
        self.status: str | None = None
        self.error: str | None = None
        self._content_parts: list[str] = []
        self._fallback_content = ""
        self._sources: list[dict] = []
        self._seen_urls: set[str] = set()

    def _add_source(self, raw: Any) -> None:
        _append_grok_source(raw, self._sources, self._seen_urls)

    def _add_output_item(self, item: Any) -> None:
        content, sources = _collect_responses_output([item])
        if content:
            self._fallback_content = content
        for source in sources:
            self._add_source(source)

    def _merge_result(self, result: SearchResponse) -> None:
        if result.content:
            self._fallback_content = result.content
        for source in result.sources:
            self._add_source(source)
        self.status = result.status
        self.error = result.error

    def consume(self, payload: dict) -> None:
        self.received_event = True
        event_type = payload.get("type")

        if not event_type and ("output" in payload or "status" in payload):
            self._merge_result(_parse_responses_payload_data(payload))
            self.terminal = True
            return

        if not event_type and payload.get("error"):
            self.status = "failed"
            self.error = str(_api_error_from_payload(payload["error"]))
            self.terminal = True
            return

        if event_type == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                self._content_parts.append(delta)
            return

        if event_type == "response.output_text.annotation.added":
            self._add_source(payload.get("annotation") or payload)
            return

        if event_type == "response.output_text.done":
            text = payload.get("text")
            if isinstance(text, str):
                self._fallback_content = text
            return

        if event_type == "response.content_part.done":
            part = payload.get("part")
            if isinstance(part, dict):
                text = part.get("text")
                if part.get("type") == "output_text" and isinstance(text, str):
                    self._fallback_content = text
                for annotation in part.get("annotations") or []:
                    self._add_source(annotation)
            return

        if event_type == "response.output_item.done":
            self._add_output_item(payload.get("item"))
            return

        if isinstance(event_type, str) and event_type.startswith("response.web_search_call."):
            action = payload.get("action")
            if isinstance(action, dict):
                for source in action.get("sources") or []:
                    self._add_source(source)
            self._add_output_item(payload.get("item"))
            return

        if event_type in {
            "response.completed",
            "response.incomplete",
            "response.failed",
        }:
            response_payload = payload.get("response")
            if not isinstance(response_payload, dict):
                response_payload = dict(payload)
                response_payload.setdefault("status", event_type.removeprefix("response."))
            self._merge_result(_parse_responses_payload_data(response_payload))
            self.terminal = True
            return

        if event_type == "error":
            self.status = "failed"
            self.error = str(_api_error_from_payload(payload.get("error") or payload))
            self.terminal = True

    def result(
        self,
        default_status: str = "incomplete",
        default_error: str | None = None,
    ) -> SearchResponse:
        content = "".join(self._content_parts).strip()
        if not content:
            content = self._fallback_content.strip()
        status = self.status or default_status
        error = self.error or default_error
        if status == "completed" and not content:
            raise GrokAPIError("Responses API completed without output text")
        return SearchResponse(
            content=content,
            sources=self._sources,
            status=status,
            error=error,
        )


class _WaitWithRetryAfter(wait_base):
    """等待策略：优先使用 Retry-After 头，否则使用指数退避"""

    def __init__(self, multiplier: float, max_wait: int):
        self._base_wait = wait_random_exponential(multiplier=multiplier, max=max_wait)
        self._protocol_error_base = 3.0

    def __call__(self, retry_state):
        if retry_state.outcome and retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                retry_after = self._parse_retry_after(exc.response)
                if retry_after is not None:
                    return retry_after
            if isinstance(exc, httpx.RemoteProtocolError):
                return self._base_wait(retry_state) + self._protocol_error_base
        return self._base_wait(retry_state)

    def _parse_retry_after(self, response: httpx.Response) -> Optional[float]:
        """解析 Retry-After 头（支持秒数或 HTTP 日期格式）"""
        header = response.headers.get("Retry-After")
        if not header:
            return None
        header = header.strip()

        if header.isdigit():
            return float(header)

        try:
            retry_dt = parsedate_to_datetime(header)
            if retry_dt.tzinfo is None:
                retry_dt = retry_dt.replace(tzinfo=timezone.utc)
            delay = (retry_dt - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, delay)
        except (TypeError, ValueError):
            return None


class GrokSearchProvider(BaseSearchProvider):
    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str = "grok-4-fast",
        api_mode: str = "auto",
    ):
        super().__init__(api_url, api_key)
        self.model = model
        self.api_mode = api_mode

    def get_provider_name(self) -> str:
        return "Grok"

    def _resolved_api_mode(self) -> str:
        if self.api_mode != "auto":
            return self.api_mode
        if "multi-agent" in self.model.lower():
            return "responses"
        return "chat_completions"

    async def search(
        self,
        query: str,
        platform: str = "",
        min_results: int = 3,
        max_results: int = 10,
        effort: ResponseEffort = DEFAULT_RESPONSE_EFFORT,
        ctx=None,
    ) -> SearchResponse:
        if effort not in VALID_RESPONSE_EFFORTS:
            valid = ", ".join(sorted(VALID_RESPONSE_EFFORTS))
            raise ValueError(f"effort must be one of: {valid}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        platform_prompt = ""

        if platform:
            platform_prompt = "\n\nYou should search the web for the information you need, and focus on these platform: " + platform + "\n"

        time_context = get_local_time_info() + "\n"

        await log_info(ctx, f"platform_prompt: { query + platform_prompt}", config.debug_enabled)

        user_input = time_context + query + platform_prompt
        if self._resolved_api_mode() == "responses":
            effective_effort = config.responses_effort or effort
            payload = {
                "model": self.model,
                "instructions": search_prompt,
                "input": user_input,
                "reasoning": {"effort": effective_effort},
                "tools": [{"type": "web_search"}],
                "stream": True,
            }
            return await self._execute_responses_with_retry(headers, payload, ctx)

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": search_prompt},
                {"role": "user", "content": user_input},
            ],
            "stream": True,
        }
        content = await self._execute_stream_with_retry(headers, payload, ctx)
        return SearchResponse(content=content)

    @staticmethod
    def _parse_responses_payload(payload: dict) -> SearchResponse:
        return _parse_responses_payload_data(payload)

    async def _parse_responses_stream(self, response, ctx=None) -> SearchResponse:
        accumulator = _ResponsesStreamAccumulator()
        full_json_lines: list[str] = []

        try:
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line or line.startswith("event:"):
                    continue
                if not line.startswith("data:"):
                    full_json_lines.append(line)
                    continue

                data_text = line[5:].lstrip()
                if not data_text or data_text == "[DONE]":
                    continue
                try:
                    event = json.loads(data_text)
                except json.JSONDecodeError:
                    full_json_lines.append(data_text)
                    continue
                if not isinstance(event, dict):
                    continue

                accumulator.consume(event)
                if accumulator.terminal:
                    result = accumulator.result()
                    await log_info(
                        ctx,
                        f"responses stream status: {result.status}, content length: {len(result.content)}, sources: {len(result.sources)}",
                        config.debug_enabled,
                    )
                    return result
        except httpx.RequestError as exc:
            if accumulator.received_event:
                return accumulator.result(
                    default_status="incomplete",
                    default_error=f"Responses stream interrupted: {exc}",
                )
            raise

        if accumulator.received_event:
            return accumulator.result(
                default_status="incomplete",
                default_error="Responses stream ended before a terminal event",
            )

        if full_json_lines:
            try:
                payload = json.loads("".join(full_json_lines))
            except json.JSONDecodeError as exc:
                raise _RetryableResponsesProtocolError(
                    f"Responses API returned invalid JSON before streaming: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise _RetryableResponsesProtocolError(
                    "Responses API returned a non-object JSON payload"
                )
            return self._parse_responses_payload(payload)

        raise _RetryableResponsesProtocolError(
            "Responses API returned an empty response before streaming"
        )

    async def _execute_responses_with_retry(
        self,
        headers: dict,
        payload: dict,
        ctx=None,
    ) -> SearchResponse:
        timeout = httpx.Timeout(
            connect=6.0,
            read=config.responses_read_timeout,
            write=10.0,
            pool=None,
        )
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/responses",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.is_error:
                            body = await response.aread()
                            response._content = body
                            raise _api_error_from_response(response)
                        return await self._parse_responses_stream(response, ctx)

        raise GrokAPIError("Responses API request exhausted retries")

    async def fetch(self, url: str, ctx=None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": fetch_prompt,
                },
                {"role": "user", "content": url + "\n获取该网页内容并返回其结构化Markdown格式" },
            ],
            "stream": True,
        }
        return await self._execute_stream_with_retry(headers, payload, ctx)

    async def _parse_streaming_response(self, response, ctx=None) -> str:
        content = ""
        full_body_buffer = [] 
        
        async for line in response.aiter_lines():
            line = line.strip()
            if not line:
                continue
            
            full_body_buffer.append(line)

            # 兼容 "data: {...}" 和 "data:{...}" 两种 SSE 格式
            if line.startswith("data:"):
                if line in ("data: [DONE]", "data:[DONE]"):
                    continue
                try:
                    # 去掉 "data:" 前缀，并去除可能的空格
                    json_str = line[5:].lstrip()
                    data = json.loads(json_str)
                    if data.get("error"):
                        raise _api_error_from_payload(data["error"], response.status_code)
                    choices = data.get("choices", [])
                    if choices and len(choices) > 0:
                        delta = choices[0].get("delta", {})
                        delta_content = delta.get("content")
                        if isinstance(delta_content, str):
                            content += delta_content
                except (json.JSONDecodeError, IndexError):
                    continue
                
        if not content and full_body_buffer:
            try:
                full_text = "".join(full_body_buffer)
                data = json.loads(full_text)
                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    message_content = message.get("content", "")
                    if isinstance(message_content, str):
                        content = message_content
            except json.JSONDecodeError:
                pass
        
        await log_info(ctx, f"content: {content}", config.debug_enabled)

        if not content:
            raise GrokAPIError("Chat Completions API returned no text content")
        return content

    async def _execute_stream_with_retry(self, headers: dict, payload: dict, ctx=None) -> str:
        """执行带重试机制的流式 HTTP 请求"""
        timeout = httpx.Timeout(connect=6.0, read=120.0, write=10.0, pool=None)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(config.retry_max_attempts + 1),
                wait=_WaitWithRetryAfter(config.retry_multiplier, config.retry_max_wait),
                retry=retry_if_exception(_is_retryable_exception),
                reraise=True,
            ):
                with attempt:
                    async with client.stream(
                        "POST",
                        f"{self.api_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.is_error:
                            body = await response.aread()
                            response._content = body
                            raise _api_error_from_response(response)
                        return await self._parse_streaming_response(response, ctx)

        raise GrokAPIError("Chat Completions request exhausted retries")

    async def describe_url(self, url: str, ctx=None) -> dict:
        """让 Grok 阅读单个 URL 并返回 title + extracts"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": url_describe_prompt},
                {"role": "user", "content": url},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        title, extracts = url, ""
        for line in result.strip().splitlines():
            if line.startswith("Title:"):
                title = line[6:].strip() or url
            elif line.startswith("Extracts:"):
                extracts = line[9:].strip()
        return {"title": title, "extracts": extracts, "url": url}

    async def rank_sources(self, query: str, sources_text: str, total: int, ctx=None) -> list[int]:
        """让 Grok 按查询相关度对信源排序，返回排序后的序号列表"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": rank_sources_prompt},
                {"role": "user", "content": f"Query: {query}\n\n{sources_text}"},
            ],
            "stream": True,
        }
        result = await self._execute_stream_with_retry(headers, payload, ctx)
        order: list[int] = []
        seen: set[int] = set()
        for token in result.strip().split():
            try:
                n = int(token)
                if 1 <= n <= total and n not in seen:
                    seen.add(n)
                    order.append(n)
            except ValueError:
                continue
        # 补齐遗漏的序号
        for i in range(1, total + 1):
            if i not in seen:
                order.append(i)
        return order

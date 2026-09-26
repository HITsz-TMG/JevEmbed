"""Small OpenAI-compatible chat-completions teacher client."""

import os
import json
import threading
import time


class TeacherError(RuntimeError):
    """A sanitized teacher failure (never includes headers or response bodies)."""

    def __init__(self, message, *, fatal=False):
        super().__init__(message)
        self.fatal = fatal


class ChatTeacher:
    def __init__(self, settings, *, client=None):
        import httpx

        self.settings = settings
        key_name = settings.get("api_key_env")
        if key_name and not os.getenv(key_name):
            raise TeacherError(f"Teacher API key environment variable {key_name} is unset")
        self._own_client = client is None
        self.client = client or httpx.Client(timeout=settings["timeout"], follow_redirects=False)
        self._counter_lock = threading.Lock()
        self.http_requests = 0
        self._local = threading.local()
        self.headers = {"Content-Type": "application/json"}
        if key_name:
            self.headers["Authorization"] = f"Bearer {os.environ[key_name]}"
        self.url = settings["base_url"].rstrip("/") + "/chat/completions"

    def close(self):
        if self._own_client:
            self.client.close()

    def complete(self, messages):
        """Return (assistant text, usage). Retry only transient transport/status failures."""
        import httpx

        body = {"model": self.settings["model"], "messages": messages,
                "temperature": self.settings["temperature"], "max_tokens": self.settings["max_tokens"]}
        body.update(self.settings.get("request_options", {}))
        retries = self.settings["max_retries"]
        self._local.attempts = 0
        for attempt in range(retries + 1):
            try:
                with self._counter_lock:
                    self.http_requests += 1
                self._local.attempts += 1
                with self.client.stream("POST", self.url, json=body, headers=self.headers) as response:
                    status = response.status_code
                    if status in (408, 429) or 500 <= status <= 599:
                        if attempt < retries:
                            time.sleep(min(2 ** attempt, 8))
                            continue
                    if not 200 <= status < 300:
                        raise TeacherError(f"Teacher HTTP {status}",
                                           fatal=status not in (408, 429) and status < 500)
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 8 * 1024 * 1024:
                            raise TeacherError("Teacher response exceeds 8 MiB")
                        chunks.append(chunk)
                payload = json.loads(b"".join(chunks))
                if type(payload) is not dict:
                    raise ValueError("response is not an object")
                choices = payload.get("choices")
                if type(choices) is not list or not choices or type(choices[0]) is not dict:
                    raise ValueError("choices is not a nonempty list of objects")
                choice = choices[0]
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None and type(finish_reason) is not str:
                    raise ValueError("invalid finish reason")
                if finish_reason in ("length", "content_filter", "tool_calls"):
                    raise TeacherError("Teacher response did not finish normally")
                message = choice.get("message")
                if type(message) is not dict or type(message.get("content")) is not str:
                    raise ValueError("assistant content is not text")
                content = message["content"]
                usage = payload.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict) or any(
                        key in usage and (type(usage[key]) is not int or usage[key] < 0)
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    ):
                        raise ValueError("invalid usage")
                return content, usage
            except httpx.TransportError as exc:
                if attempt < retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise TeacherError("Teacher transport failure") from exc
            except (KeyError, IndexError, AttributeError, TypeError, ValueError):
                raise TeacherError("Malformed teacher response") from None
        raise TeacherError("Teacher retry budget exhausted")

    def last_http_requests(self):
        return getattr(self._local, "attempts", 0)

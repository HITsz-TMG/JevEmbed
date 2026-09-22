import logging
import math
import os
import time

from .base import EmbeddingBatch
from ..errors import BackendError, ValidationError
from ..prompts import PromptAdapter

logger = logging.getLogger(__name__)


class HTTPEmbeddingBackend:
    def __init__(self, config, *, transport=None):
        self.config = config
        self.adapter = PromptAdapter(config.prompts)
        self.transport = transport
        if config.template_owner != "client":
            raise ValidationError("Generic /v1/embeddings supports client templates only; server templates require a dedicated adapter")
        if not config.server_enforces_length or config.overflow_policy != "error":
            raise ValidationError("HTTP requires server_enforces_length=true and overflow_policy=error; configure the server to reject oversized input")

    def encode(self, items):
        try:
            import httpx
        except ImportError as exc:
            raise BackendError("Install jevembed[http]") from exc
        cfg = self.config
        headers = {}
        if cfg.api_key_env:
            key = os.environ.get(cfg.api_key_env)
            if not key:
                raise BackendError(f"Missing API key environment variable: {cfg.api_key_env}")
            headers["Authorization"] = f"Bearer {key}"
        vectors, tokens, sources = [], 0, set()
        with httpx.Client(timeout=cfg.timeout, transport=self.transport) as client:
            for start in range(0, len(items), cfg.batch_size):
                texts = [self.adapter.render(item) for item in items[start:start+cfg.batch_size]]
                response = None
                for attempt in range(cfg.max_retries + 1):
                    try:
                        response = client.post(cfg.base_url.rstrip("/") + "/embeddings", headers=headers,
                                               json={"model": cfg.model_name_or_path, "input": texts, "encoding_format": "float"})
                        if response.status_code not in (408, 429, 500, 502, 503, 504, 529) or attempt == cfg.max_retries:
                            response.raise_for_status()
                            break
                    except (httpx.TimeoutException, httpx.NetworkError) as exc:
                        if attempt == cfg.max_retries:
                            raise BackendError(f"Embedding HTTP transport failed: {type(exc).__name__}") from exc
                    except httpx.HTTPStatusError as exc:
                        raise BackendError(f"Embedding server returned HTTP {exc.response.status_code}") from exc
                    time.sleep(min(0.25 * 2**attempt, 2))
                try:
                    data = response.json()
                    rows = data["data"]
                    indices = [row["index"] for row in rows]
                    if any(type(i) is not int for i in indices) or sorted(indices) != list(range(len(texts))):
                        raise ValueError("Missing, duplicate or invalid embedding indices")
                    vectors.extend(row["embedding"] for row in sorted(rows, key=lambda row: row["index"]))
                    usage = data.get("usage", {}).get("prompt_tokens")
                    if usage is None:
                        if cfg.usage_mode == "strict":
                            raise ValueError("Embedding response has no exact usage.prompt_tokens")
                        usage = sum(math.ceil(len(text.encode("utf-8"))/4) for text in texts)
                        sources.add("estimated_utf8_bytes_div4")
                        logger.warning("HTTP token usage estimated as ceil(UTF-8 bytes / 4) per input")
                    else:
                        sources.add("http_usage")
                    if type(usage) is not int or usage < 0:
                        raise ValueError("Invalid usage.prompt_tokens")
                    tokens += usage
                except (KeyError, TypeError, ValueError) as exc:
                    raise BackendError(f"Invalid embedding HTTP response: {exc}") from exc
        return EmbeddingBatch(vectors, tokens, "+".join(sorted(sources)),
                              {"template_owner": "client", "length_policy": "server_enforced_error"})

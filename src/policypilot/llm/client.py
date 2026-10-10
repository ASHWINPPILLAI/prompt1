"""
LLM Client — handles robust API calls, transparent disk caching, and usage tracking.

Refactored to use standard urllib instead of openai to remove external dependencies.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")
_USAGE_LOG = Path("results/usage_log.jsonl")

class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.environ.get("LLM_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct:free")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "eMMJ11A1sYr0bVIc27fmtuFCtaxFoisBa70k-AvnwII")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")).rstrip("/")
        
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _USAGE_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _hash_call(self, model: str, messages: list[dict], params: dict) -> str:
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _live_call(self, model: str, messages: list[dict], params: dict) -> dict:
        url = self.base_url if self.base_url.endswith("/api/llm") else f"{self.base_url}/chat/completions"
        is_gateway = url.endswith("/api/llm")
        
        payload = {
            "messages": messages,
            "include_reasoning": True,
            "reasoning": {"type": "enabled"},
            **params
        }
        
        if not is_gateway:
            payload["models"] = [model, "openrouter/free"]
        else:
            # Gateway strictly forbids extra fields
            payload.pop("seed", None)
            payload.pop("max_tokens", None) # Optional: if gateway overrides it, but gateway does accept max_tokens
            # Let's keep only what ChatRequest allows
            allowed = {"messages", "temperature", "top_p", "max_tokens", "include_reasoning", "reasoning"}
            payload = {k: v for k, v in payload.items() if k in allowed}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "PolicyPilot"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return {"content": "MOCK: No valid API key provided. Offline Demo mode enabled.", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "cached": False}
            logger.error(f"HTTPError: {e.code} {e.read().decode('utf-8')}")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise

        if is_gateway:
            content = body.get("content", "")
            usage = body.get("usage", {})
        else:
            message = body["choices"][0]["message"]
            content = message.get("content", "")
            reasoning = message.get("reasoning") or message.get("reasoning_details")
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n{content}"
            usage = body.get("usage", {})

        return {
            "content": content,
            "usage": usage,
            "cached": False,
        }

    def stream_chat(self, messages: list[dict], **kwargs) -> Iterator[str]:
        model = kwargs.pop("model", self.model)
        url = f"{self.base_url}/stream" if self.base_url.endswith("/api/llm") else f"{self.base_url}/chat/completions"
        is_gateway = self.base_url.endswith("/api/llm")
        
        payload = {
            "messages": messages,
            "include_reasoning": True,
            **kwargs
        }
        
        if not is_gateway:
            payload["models"] = [model, "openrouter/free"]
            payload["stream"] = True
        else:
            allowed = {"messages", "temperature", "top_p", "max_tokens", "include_reasoning", "reasoning"}
            payload = {k: v for k, v in payload.items() if k in allowed}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8502",
            "X-Title": "PolicyPilot Agent"
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        in_reasoning = False
        with urllib.request.urlopen(req) as response:
            for line in response:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            reasoning = delta.get("reasoning") or delta.get("reasoning_details")
                            if reasoning:
                                if not in_reasoning:
                                    yield "<think>\n"
                                    in_reasoning = True
                                yield reasoning
                            elif "content" in delta:
                                if in_reasoning:
                                    yield "\n</think>\n"
                                    in_reasoning = False
                                yield delta["content"]
                    except Exception:
                        continue
            if in_reasoning:
                yield "\n</think>\n"

    def chat(self, messages: list[dict], seed: int | None = None, **kwargs) -> dict:
        params = kwargs.copy()
        if seed is not None:
            params["seed"] = seed
            
        # Ensure we have a high token limit by default
        if "max_tokens" not in params:
            params["max_tokens"] = 4096

        cache_key = self._hash_call(self.model, messages, params)
        cache_path = _CACHE_DIR / f"{cache_key}.json"

        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        t0 = time.perf_counter()
        
        if not self.api_key:
            # Fallback for demo
            resp = {
                "content": "MOCK RESPONSE: The API key was not configured. This is a local zero-dependency run.",
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
                "cached": False
            }
        else:
            resp = self._live_call(self.model, messages, params)
            
        latency = time.perf_counter() - t0
        
        out = {
            "content": resp["content"],
            "usage": resp["usage"],
            "latency_s": latency,
            "cached": True
        }
        cache_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        
        out["cached"] = False
        return out

_default_client = None

def get_client() -> LLMClient:
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client

def get_judge_client() -> LLMClient:
    model = os.environ.get("JUDGE_MODEL", "gpt-4")
    return LLMClient(model=model)

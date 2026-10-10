"""PolicyPilot secure LLM gateway (reference implementation).

The browser never sees the OpenRouter key and never talks to OpenRouter.
Everything goes: browser -> this gateway -> OpenRouter.

What it enforces (see README.md for the threat model):
  * key lives only in the server environment, is never logged or returned
  * callers authenticate with a token; only SHA-256 hashes are stored server-side
  * callers cannot choose the model, the URL, tools, or oversized parameters
  * per-user, per-IP, global and daily request limits (OpenRouter free limits are account-wide)
  * only ':free' models are allowed unless you explicitly opt in to paid ones
  * provider data-collection is denied by default
  * best-effort PII masking inside the <question> part of the prompt
  * response cache + single-flight for deterministic (temperature 0) calls
  * model fallback chain with a circuit breaker, timeouts and bounded retries
  * only a minimal, sanitised result is returned; upstream errors are never leaked
  * body-size limit, CORS allowlist, security headers, logs without prompts or secrets

Run:  uvicorn gateway:app --host 127.0.0.1 --port 8000
New token:  python gateway.py newtoken <name>
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import random
import re
import secrets
import sys
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

log = logging.getLogger("gateway")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"  # fixed on purpose: never user-controlled
MODEL_ID = re.compile(r"^[A-Za-z0-9._:/-]{3,100}$")


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    api_key: str = field(default="", repr=False)          # repr=False: never printed by accident
    models: tuple = ()
    tokens: dict = field(default_factory=dict, repr=False)  # name -> sha256 hex of the caller token
    allowed_origins: tuple = ()
    allow_credentials: bool = False
    data_collection: str = "deny"                          # "deny" or "allow"
    allow_paid_models: bool = False
    upstream_url: str = OPENROUTER_URL
    max_body_bytes: int = 96_000
    max_messages: int = 24
    max_chars_per_message: int = 16_000
    max_total_chars: int = 40_000
    max_output_tokens: int = 800
    max_output_chars: int = 8_000
    user_rpm: int = 10
    ip_rpm: int = 30
    global_rpm: int = 18        # OpenRouter free models allow 20/min per account; stay under it
    daily_cap: int = 45         # 50/day without purchased credits; use 950 with >= $10 credits
    max_queue_wait_s: float = 8.0
    max_inflight: int = 4
    cache_ttl_s: int = 3600
    cache_max: int = 512
    breaker_failures: int = 3
    breaker_cooldown_s: float = 60.0
    retry_backoff_s: float = 0.4
    scrub_pii: bool = True
    hsts: bool = False
    app_url: str = ""
    app_title: str = "PolicyPilot"

    @staticmethod
    def from_env(env=os.environ) -> "Config":
        def num(name, d, cast=int):
            return cast(env.get(name, d))

        def flag(name, d):
            return str(env.get(name, d)).strip().lower() in ("1", "true", "yes")

        tokens = {}
        for part in (p.strip() for p in env.get("GATEWAY_TOKEN_HASHES", "").split(",")):
            name, _, digest = part.partition(":")
            if name and re.fullmatch(r"[0-9a-f]{64}", digest):
                tokens[name] = digest
        return Config(
            api_key=env.get("OPENROUTER_API_KEY", ""),
            models=tuple(m.strip() for m in env.get("OPENROUTER_MODELS", "").split(",") if m.strip()),
            tokens=tokens,
            allowed_origins=tuple(o.strip() for o in env.get("ALLOWED_ORIGINS", "").split(",") if o.strip()),
            allow_credentials=flag("CORS_ALLOW_CREDENTIALS", False),
            data_collection=env.get("OPENROUTER_DATA_COLLECTION", "deny"),
            allow_paid_models=flag("ALLOW_PAID_MODELS", False),
            user_rpm=num("USER_RPM", 10), ip_rpm=num("IP_RPM", 30), global_rpm=num("GLOBAL_RPM", 18),
            daily_cap=num("DAILY_CAP", 45), scrub_pii=flag("SCRUB_PII", True), hsts=flag("ENABLE_HSTS", False),
            app_url=env.get("APP_URL", ""), app_title=env.get("APP_TITLE", "PolicyPilot"),
        )

    def validate(self, *, require_key: bool = True) -> None:
        if require_key and not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set (put it in the server environment, never in code).")
        if self.api_key and not self.api_key.startswith("sk-or-"):
            log.warning("config: OPENROUTER_API_KEY does not look like an OpenRouter key")
        if not self.models:
            raise RuntimeError("OPENROUTER_MODELS is empty. List one or more model ids, comma separated.")
        for m in self.models:
            if not MODEL_ID.match(m):
                raise RuntimeError(f"Invalid model id: {m!r}")
            if not m.endswith(":free") and not self.allow_paid_models:
                raise RuntimeError(f"{m!r} is not a ':free' model. Set ALLOW_PAID_MODELS=true to allow spending.")
        if self.data_collection not in ("deny", "allow"):
            raise RuntimeError("OPENROUTER_DATA_COLLECTION must be 'deny' or 'allow'.")
        if not self.tokens:
            raise RuntimeError("GATEWAY_TOKEN_HASHES is empty. Create tokens with: python gateway.py newtoken <name>")
        if "*" in self.allowed_origins:
            raise RuntimeError("ALLOWED_ORIGINS must list exact origins, not '*'.")


# --------------------------------------------------------------------------- helpers
def redact(text: str, cfg: Config) -> str:
    return text.replace(cfg.api_key, "[redacted]") if cfg.api_key else text


def user_tag(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:10]


def log_event(**fields) -> None:
    """One JSON line per request. Never pass prompts, answers, tokens or keys here."""
    log.info(json.dumps(fields, separators=(",", ":"), default=str))


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_QUESTION = re.compile(r"(<question>)(.*?)(</question>)", re.S)


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def scrub(text: str) -> str:
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub("[SSN]", text)
    text = _PHONE.sub("[PHONE]", text)

    def card(m):
        digits = re.sub(r"\D", "", m.group(0))
        return "[CARD]" if 13 <= len(digits) <= 19 and _luhn(digits) else m.group(0)

    return _CARD.sub(card, text)


def scrub_message(role: str, content: str) -> str:
    """Mask personal data in what the user typed. Only the <question> block is touched when present,
    so official phone numbers inside retrieved documents stay intact."""
    if role != "user":
        return content
    if "<question>" in content:
        return _QUESTION.sub(lambda m: m.group(1) + scrub(m.group(2)) + m.group(3), content)
    return scrub(content)


class Window:
    """Sliding-window limiter. check() returns 0.0 if allowed, else seconds to wait."""

    def __init__(self, limit: int, per_s: float = 60.0):
        self.limit, self.per, self.hits = limit, per_s, defaultdict(deque)

    def check(self, key: str) -> float:
        now = time.monotonic()
        q = self.hits[key]
        while q and now - q[0] >= self.per:
            q.popleft()
        if len(q) >= self.limit:
            return max(0.1, self.per - (now - q[0]))
        q.append(now)
        if len(self.hits) > 20_000:  # bound memory under a flood of distinct keys
            for k in [k for k, v in self.hits.items() if not v]:
                del self.hits[k]
        return 0.0


class TTLCache:
    def __init__(self, ttl: int, max_items: int):
        self.ttl, self.max, self.d = ttl, max_items, OrderedDict()

    def get(self, key):
        item = self.d.get(key)
        if not item:
            return None
        if item[0] < time.monotonic():
            self.d.pop(key, None)
            return None
        self.d.move_to_end(key)
        return item[1]

    def set(self, key, value):
        self.d[key] = (time.monotonic() + self.ttl, value)
        self.d.move_to_end(key)
        while len(self.d) > self.max:
            self.d.popitem(last=False)


@dataclass
class Breaker:
    failures: int = 0
    open_until: float = 0.0


class Retryable(Exception):
    """Upstream problem worth trying another model for."""


# --------------------------------------------------------------------------- schemas
class Msg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """No 'model', 'tools', 'url' or 'provider' fields exist, and extras are rejected."""
    model_config = ConfigDict(extra="forbid")
    messages: list[Msg] = Field(min_length=1)
    temperature: float = Field(0.2, allow_inf_nan=False)
    top_p: float = Field(1.0, allow_inf_nan=False)
    max_tokens: int = 600
    include_reasoning: Optional[bool] = None
    reasoning: Optional[dict | bool] = None


# --------------------------------------------------------------------------- gateway core
class Gateway:
    def __init__(self, cfg: Config, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.cfg = cfg
        self.transport = transport
        self.client: Optional[httpx.AsyncClient] = None
        self.user_win, self.ip_win = Window(cfg.user_rpm), Window(cfg.ip_rpm)
        self.global_win = Window(cfg.global_rpm)
        self.cache = TTLCache(cfg.cache_ttl_s, cfg.cache_max)
        self.inflight: dict = {}
        self.breakers = {m: Breaker() for m in cfg.models}
        self.sem = asyncio.Semaphore(cfg.max_inflight)
        self.day, self.day_count = time.strftime("%Y-%m-%d", time.gmtime()), 0

    async def start(self):
        self.client = httpx.AsyncClient(
            transport=self.transport,
            timeout=httpx.Timeout(connect=5.0, read=45.0, write=10.0, pool=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=False,  # a redirect could leak the Authorization header
        )

    async def stop(self):
        if self.client:
            await self.client.aclose()

    # ---- request preparation
    def prepare(self, req: ChatRequest) -> ChatRequest:
        c = self.cfg
        if len(req.messages) > c.max_messages:
            raise HTTPException(413, "Too many messages.")
        total = 0
        for m in req.messages:
            if len(m.content) > c.max_chars_per_message:
                raise HTTPException(413, "A message is too long.")
            total += len(m.content)
        if total > c.max_total_chars:
            raise HTTPException(413, "The conversation is too long.")
        msgs = [Msg(role=m.role, content=scrub_message(m.role, m.content) if c.scrub_pii else m.content) for m in req.messages]
        return ChatRequest(
            messages=msgs,
            temperature=min(1.0, max(0.0, req.temperature)),
            top_p=min(1.0, max(0.05, req.top_p)),
            max_tokens=min(c.max_output_tokens, max(16, req.max_tokens)),
            include_reasoning=req.include_reasoning,
            reasoning=req.reasoning
        )

    def cache_key(self, req: ChatRequest) -> str:
        blob = json.dumps([self.cfg.models, [m.model_dump() for m in req.messages], req.temperature, req.top_p, req.max_tokens], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    # ---- limits
    def check_caller(self, user: str, ip: str):
        for win, key in ((self.user_win, "u:" + user), (self.ip_win, "ip:" + ip)):
            wait = win.check(key)
            if wait:
                raise HTTPException(429, "Too many requests. Slow down.", headers={"Retry-After": str(int(wait) + 1)})

    async def acquire_upstream_slot(self):
        """Global and daily limits protect the account-wide OpenRouter quota. Briefly queue before failing."""
        deadline = time.monotonic() + self.cfg.max_queue_wait_s
        while True:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if today != self.day:
                self.day, self.day_count = today, 0
            if self.day_count >= self.cfg.daily_cap:
                raise HTTPException(429, "The daily request budget is used up. Try again tomorrow.", headers={"Retry-After": "3600"})
            wait = self.global_win.check("global")
            if wait == 0.0:
                self.day_count += 1
                return
            if time.monotonic() + min(wait, 1.0) > deadline:
                raise HTTPException(429, "The service is busy. Try again in a moment.", headers={"Retry-After": str(int(wait) + 1)})
            await asyncio.sleep(min(wait, 1.0))

    # ---- upstream
    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.cfg.api_key}", "Content-Type": "application/json"}
        if self.cfg.app_url:
            h["HTTP-Referer"] = self.cfg.app_url
        if self.cfg.app_title:
            h["X-Title"] = self.cfg.app_title
        return h

    def _payload(self, model: str, req: ChatRequest, stream: bool) -> dict:
        body = {
            "model": model,
            "messages": [m.model_dump() for m in req.messages],
            "temperature": req.temperature, "top_p": req.top_p, "max_tokens": req.max_tokens,
            "provider": {"data_collection": self.cfg.data_collection},
        }
        if stream:
            body["stream"] = True
        if req.include_reasoning is not None:
            body["include_reasoning"] = req.include_reasoning
        if req.reasoning is not None:
            body["reasoning"] = req.reasoning
        return body

    def _classify(self, status: int, model: str):
        if status in (401, 402, 403):  # key revoked, no credit, or blocked: an operator problem, not a model problem
            log_event(event="upstream_config_error", status=status, model=model)
            raise HTTPException(503, "The language service is unavailable.")
        raise Retryable(f"status {status}")

    def _breaker_hit(self, model: str):
        b = self.breakers[model]
        b.failures += 1
        if b.failures >= self.cfg.breaker_failures:
            b.open_until = time.monotonic() + self.cfg.breaker_cooldown_s
            log_event(event="breaker_open", model=model)

    async def _post_once(self, model: str, req: ChatRequest) -> dict:
        async with self.sem:
            try:
                r = await self.client.post(self.cfg.upstream_url, headers=self._headers(), json=self._payload(model, req, False))
            except (httpx.TimeoutException, httpx.TransportError):
                raise Retryable("transport")
        if r.status_code != 200:
            self._classify(r.status_code, model)
        try:
            data = r.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError
            reasoning = choice["message"].get("reasoning") or choice["message"].get("reasoning_details")
            if reasoning:
                content = f"<think>\n{reasoning}\n</think>\n{content}"
        except Exception:
            raise Retryable("bad body")
        usage = data.get("usage") or {}
        return {
            "content": redact(content, self.cfg)[: self.cfg.max_output_chars],
            "model": model,
            "finish_reason": choice.get("finish_reason"),
            "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0), "completion_tokens": int(usage.get("completion_tokens") or 0)},
        }

    async def _call_chain(self, req: ChatRequest) -> dict:
        for model in self.cfg.models:
            if self.breakers[model].open_until > time.monotonic():
                continue
            for attempt in (0, 1):
                await self.acquire_upstream_slot()
                try:
                    result = await self._post_once(model, req)
                    self.breakers[model].failures = 0
                    return result
                except Retryable:
                    if attempt == 0:
                        await asyncio.sleep(self.cfg.retry_backoff_s * (1 + random.random()))
                        continue
                    self._breaker_hit(model)
        raise HTTPException(503, "The language service is busy. Try again shortly.", headers={"Retry-After": "10"})

    async def chat(self, req: ChatRequest) -> dict:
        req = self.prepare(req)
        key = self.cache_key(req) if req.temperature == 0 else None
        fut = None
        if key:
            hit = self.cache.get(key)
            if hit:
                return {**hit, "cached": True}
            waiting = self.inflight.get(key)
            if waiting:  # single-flight: identical concurrent questions share one upstream call
                return {**(await asyncio.shield(waiting)), "cached": True}
            fut = asyncio.get_running_loop().create_future()
            self.inflight[key] = fut
        try:
            result = await self._call_chain(req)
            if key:
                self.cache.set(key, result)
                fut.set_result(result)
            return {**result, "cached": False}
        except BaseException as exc:
            if fut is not None and not fut.done():
                fut.set_exception(exc)
                fut.exception()  # mark retrieved so asyncio does not warn
            raise
        finally:
            if key:
                self.inflight.pop(key, None)

    async def stream(self, req: ChatRequest) -> AsyncIterator[dict]:
        """Yield {'delta': str} events then one {'done': True, ...}. Falls back to the next model only before the first token."""
        req = self.prepare(req)
        for model in self.cfg.models:
            if self.breakers[model].open_until > time.monotonic():
                continue
            await self.acquire_upstream_slot()
            started, produced, usage = False, 0, {}
            try:
                async with self.sem:
                    async with self.client.stream("POST", self.cfg.upstream_url, headers=self._headers(), json=self._payload(model, req, True)) as r:
                        if r.status_code != 200:
                            await r.aread()
                            self._classify(r.status_code, model)
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue  # keep-alive comments
                            raw = line[5:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                obj = json.loads(raw)
                            except ValueError:
                                continue
                            if "error" in obj:
                                raise Retryable("mid-stream error")
                            usage = obj.get("usage") or usage
                            choices = obj.get("choices") or []
                            
                            delta = (choices[0].get("delta") or {}).get("content") if choices else None
                            if delta:
                                delta = redact(delta, self.cfg)
                                produced += len(delta)
                                if produced > self.cfg.max_output_chars:
                                    break
                                started = True
                                yield {"delta": delta}
                self.breakers[model].failures = 0
                yield {"done": True, "model": model, "usage": {"prompt_tokens": int(usage.get("prompt_tokens") or 0), "completion_tokens": int(usage.get("completion_tokens") or 0)}}
                return
            except Retryable:
                self._breaker_hit(model)
                if started:
                    yield {"error": "The answer was interrupted.", "done": True}
                    return
            except (httpx.TimeoutException, httpx.TransportError):
                self._breaker_hit(model)
                if started:
                    yield {"error": "The answer was interrupted.", "done": True}
                    return
        yield {"error": "The language service is busy. Try again shortly.", "done": True}


# --------------------------------------------------------------------------- ASGI plumbing
class _TooLarge(Exception):
    pass


class BodyLimit:
    """Reject oversized bodies before parsing, including chunked uploads without Content-Length."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        for k, v in scope["headers"]:
            if k == b"content-length":
                try:
                    if int(v) > self.max:
                        return await self._reject(send)
                except ValueError:
                    return await self._reject(send)
        total, started = 0, False

        async def limited():
            nonlocal total
            msg = await receive()
            if msg["type"] == "http.request":
                total += len(msg.get("body", b""))
                if total > self.max:
                    raise _TooLarge()
            return msg

        async def tracking_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited, tracking_send)
        except _TooLarge:
            if not started:
                await self._reject(send)

    async def _reject(self, send):
        body = b'{"error":"Request body too large."}'
        await send({"type": "http.response.start", "status": 413, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})


def create_app(cfg: Optional[Config] = None, transport: Optional[httpx.AsyncBaseTransport] = None, require_key: bool = True) -> FastAPI:
    cfg = cfg or Config.from_env()
    cfg.validate(require_key=require_key)
    gw = Gateway(cfg, transport)

    @asynccontextmanager
    async def lifespan(app):
        await gw.start()
        yield
        await gw.stop()

    app = FastAPI(title="PolicyPilot LLM gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)  # no public schema
    app.state.gateway = gw
    app.add_middleware(BodyLimit, max_bytes=cfg.max_body_bytes)
    if cfg.allowed_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cfg.allowed_origins), allow_credentials=cfg.allow_credentials,
                           allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["authorization", "content-type"], max_age=600)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        request.state.rid = uuid.uuid4().hex[:12]
        t0 = time.monotonic()
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        response.headers["X-Request-ID"] = request.state.rid
        if cfg.hsts:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        log_event(event="request", rid=request.state.rid, path=request.url.path, status=response.status_code, ms=int((time.monotonic() - t0) * 1000))
        return response

    @app.exception_handler(HTTPException)
    async def http_err(request: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail, "request_id": getattr(request.state, "rid", "")}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError):  # never echo the submitted input back
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
        return JSONResponse({"error": "Invalid request.", "fields": fields, "request_id": getattr(request.state, "rid", "")}, status_code=422)

    @app.exception_handler(Exception)
    async def boom(request: Request, exc: Exception):
        log_event(event="unhandled", rid=getattr(request.state, "rid", ""), kind=type(exc).__name__)  # class name only, never the message
        return JSONResponse({"error": "Something went wrong.", "request_id": getattr(request.state, "rid", "")}, status_code=500)

    async def caller(request: Request) -> str:
        auth = request.headers.get("authorization", "")
        if auth[:7].lower() != "bearer ":
            raise HTTPException(401, "Sign in required.", headers={"WWW-Authenticate": "Bearer"})
        digest = hashlib.sha256(auth[7:].strip().encode()).hexdigest()
        who = None
        for name, stored in cfg.tokens.items():  # constant-time compare against every stored hash
            if hmac.compare_digest(digest, stored):
                who = name
        if who is None:
            raise HTTPException(401, "Sign in required.", headers={"WWW-Authenticate": "Bearer"})
        ip = request.client.host if request.client else "unknown"  # behind a proxy, use its verified client-IP header instead
        gw.check_caller(who, ip)
        request.state.user = user_tag(who)
        return who

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.get("/api/llm/budget")
    async def budget(_: str = Depends(caller)):
        return {"daily_used": gw.day_count, "daily_cap": cfg.daily_cap, "global_rpm": cfg.global_rpm}

    @app.post("/api/llm")
    async def llm(req: ChatRequest, request: Request, _: str = Depends(caller)):
        t0 = time.monotonic()
        out = await gw.chat(req)
        out["latency_s"] = round(time.monotonic() - t0, 2)
        out["request_id"] = request.state.rid
        log_event(event="llm", rid=request.state.rid, user=request.state.user, model=out["model"], cached=out["cached"],
                  ptok=out["usage"]["prompt_tokens"], ctok=out["usage"]["completion_tokens"], s=out["latency_s"])
        return out

    @app.post("/api/llm/stream")
    async def llm_stream(req: ChatRequest, request: Request, _: str = Depends(caller)):
        async def sse():
            async for ev in gw.stream(req):
                yield f"data: {json.dumps(ev)}\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    return app


def _cli():
    if len(sys.argv) >= 2 and sys.argv[1] == "newtoken":
        name = sys.argv[2] if len(sys.argv) > 2 else "user"
        token = secrets.token_urlsafe(32)
        print(f"Give this token to the caller ONCE (it is not stored): {token}")
        print(f"Add this to GATEWAY_TOKEN_HASHES: {name}:{hashlib.sha256(token.encode()).hexdigest()}")
        return True
    return False


if _cli():
    sys.exit(0)

if __name__ != "__main__":
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))
    app = create_app()

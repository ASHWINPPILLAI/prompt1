"""Security and behaviour tests. A fake OpenRouter is used, so no key or network is needed.
Run:  pytest -q
"""
import hashlib
import json
import logging

import httpx
import pytest
from fastapi.testclient import TestClient

import gateway
from gateway import Config, create_app

KEY = "sk-or-v1-TESTKEY-DO-NOT-LEAK-1234567890"
TOKEN = "caller-token-abc"
HASH = hashlib.sha256(TOKEN.encode()).hexdigest()
AUTH = {"Authorization": f"Bearer {TOKEN}"}
MODELS = ("fake/model-a:free", "fake/model-b:free")


class Upstream:
    """Scripted fake OpenRouter. Records every request it receives."""

    def __init__(self, script=None):
        self.calls, self.script = [], list(script or [])

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.calls.append({"headers": dict(request.headers), "body": body, "url": str(request.url)})
        if self.script:
            item = self.script.pop(0)
            return item(request) if callable(item) else item
        return ok(f"answer from {body['model']}")


def ok(text="hello", ptok=11, ctok=7):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": ptok, "completion_tokens": ctok}, "id": "gen-secret-upstream-id"})


def cfg(**kw):
    base = dict(api_key=KEY, models=MODELS, tokens={"alice": HASH}, allowed_origins=("https://app.example.com",),
                retry_backoff_s=0.0, user_rpm=100, ip_rpm=100, global_rpm=100, daily_cap=1000, max_queue_wait_s=0.2)
    base.update(kw)
    return Config(**base)


def client(upstream=None, **kw):
    up = upstream or Upstream()
    app = create_app(cfg(**kw), transport=httpx.MockTransport(up))
    return TestClient(app), up


def body(text="How long do I have to report an address change?", **kw):
    return {"messages": [{"role": "user", "content": text}], **kw}


# ---------------------------------------------------------------- authentication
def test_requires_auth():
    c, up = client()
    with c:
        assert c.post("/api/llm", json=body()).status_code == 401
        assert c.post("/api/llm", json=body(), headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert c.post("/api/llm", json=body(), headers={"Authorization": "Basic abc"}).status_code == 401
    assert up.calls == []


def test_healthz_is_public_and_minimal():
    c, _ = client()
    with c:
        r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"ok": True}


def test_no_public_schema_or_docs():
    c, _ = client()
    with c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(path).status_code == 404


# ---------------------------------------------------------------- the key never escapes
def test_key_only_goes_upstream_and_never_comes_back():
    c, up = client()
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 200
    assert up.calls[0]["headers"]["authorization"] == f"Bearer {KEY}"
    assert KEY not in r.text and KEY not in json.dumps(dict(r.headers))
    assert "gen-secret-upstream-id" not in r.text, "raw upstream fields must not be forwarded"


def test_upstream_error_text_is_never_leaked():
    leaky = httpx.Response(500, text=f"boom key={KEY} internal stack trace")
    c, _ = client(Upstream([leaky] * 20))
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 503
    assert KEY not in r.text and "stack trace" not in r.text


def test_model_echoing_the_key_is_redacted():
    c, _ = client(Upstream([ok(f"my key is {KEY}")]))
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert KEY not in r.text and "[redacted]" in r.text


def test_logs_contain_no_prompts_or_secrets(caplog):
    caplog.set_level(logging.INFO, logger="gateway")
    secret_question = "my private question about zebra-unicorn-42"
    c, _ = client()
    with c:
        c.post("/api/llm", json=body(secret_question), headers=AUTH)
        c.post("/api/llm", json=body(secret_question), headers={"Authorization": "Bearer wrong"})
    logs = caplog.text
    assert "zebra-unicorn-42" not in logs and KEY not in logs and TOKEN not in logs


def test_config_repr_hides_secrets():
    text = repr(cfg())
    assert KEY not in text and HASH not in text


# ---------------------------------------------------------------- callers cannot steer the request
def test_client_cannot_choose_model_url_tools_or_provider():
    c, up = client()
    with c:
        for extra in ({"model": "openai/gpt-5"}, {"url": "https://evil.example"}, {"tools": [{}]}, {"provider": {"allow_fallbacks": True}}, {"stream": True}):
            assert c.post("/api/llm", json=body(**extra), headers=AUTH).status_code == 422
    assert up.calls == []


def test_upstream_url_and_model_are_fixed_by_server():
    c, up = client()
    with c:
        c.post("/api/llm", json=body(), headers=AUTH)
    assert up.calls[0]["url"] == gateway.OPENROUTER_URL
    assert up.calls[0]["body"]["model"] in MODELS


def test_bad_roles_and_extra_message_fields_rejected():
    c, up = client()
    with c:
        assert c.post("/api/llm", json={"messages": [{"role": "tool", "content": "x"}]}, headers=AUTH).status_code == 422
        assert c.post("/api/llm", json={"messages": [{"role": "user", "content": "x", "name": "n"}]}, headers=AUTH).status_code == 422
        assert c.post("/api/llm", json={"messages": []}, headers=AUTH).status_code == 422
    assert up.calls == []


def test_parameters_are_clamped():
    c, up = client()
    with c:
        c.post("/api/llm", json=body(temperature=9, top_p=0, max_tokens=999999), headers=AUTH)
    sent = up.calls[0]["body"]
    assert sent["temperature"] == 1.0 and sent["top_p"] == 0.05 and sent["max_tokens"] == 800


def test_oversize_message_rejected_before_upstream():
    c, up = client()
    with c:
        assert c.post("/api/llm", json=body("x" * 16_001), headers=AUTH).status_code == 413
        many = {"messages": [{"role": "user", "content": "hi"}] * 25}
        assert c.post("/api/llm", json=many, headers=AUTH).status_code == 413
    assert up.calls == []


def test_body_size_limit():
    c, up = client()
    with c:
        r = c.post("/api/llm", content=b"{" + b" " * 200_000 + b"}", headers={**AUTH, "content-type": "application/json"})
    assert r.status_code == 413 and up.calls == []


def test_validation_errors_do_not_echo_input():
    c, _ = client()
    with c:
        r = c.post("/api/llm", json={"messages": [{"role": "user", "content": "SECRET-INPUT"}], "model": "SECRET-MODEL"}, headers=AUTH)
    assert r.status_code == 422 and "SECRET" not in r.text


# ---------------------------------------------------------------- privacy
def test_data_collection_denied_by_default():
    c, up = client()
    with c:
        c.post("/api/llm", json=body(), headers=AUTH)
    assert up.calls[0]["body"]["provider"] == {"data_collection": "deny"}


def test_pii_masked_in_question_but_context_phone_kept():
    prompt = "<context>Call 1-800-772-1213 for help.</context>\\n<question>My SSN is 123-45-6789, email a.b@mail.com, card 4111 1111 1111 1111, phone (555) 123-4567</question>"
    c, up = client()
    with c:
        c.post("/api/llm", json=body(prompt), headers=AUTH)
    sent = up.calls[0]["body"]["messages"][0]["content"]
    assert "1-800-772-1213" in sent, "official numbers in retrieved text must survive"
    for raw in ("123-45-6789", "a.b@mail.com", "4111 1111 1111 1111", "(555) 123-4567"):
        assert raw not in sent
    assert "[SSN]" in sent and "[EMAIL]" in sent and "[CARD]" in sent and "[PHONE]" in sent


def test_ordinary_numbers_are_not_masked():
    q = "<question>If my benefit is $1,240 a month, how much over 18 months? Total 4111 1111 1111 1112</question>"  # last one fails Luhn
    c, up = client()
    with c:
        c.post("/api/llm", json=body(q), headers=AUTH)
    sent = up.calls[0]["body"]["messages"][0]["content"]
    assert "$1,240" in sent and "18 months" in sent and "4111 1111 1111 1112" in sent


# ---------------------------------------------------------------- rate limits and cost
def test_per_user_rate_limit():
    c, up = client(user_rpm=3)
    with c:
        codes = [c.post("/api/llm", json=body(f"q{i}"), headers=AUTH).status_code for i in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert len(up.calls) == 3


def test_retry_after_header_on_429():
    c, _ = client(user_rpm=1)
    with c:
        c.post("/api/llm", json=body("a"), headers=AUTH)
        r = c.post("/api/llm", json=body("b"), headers=AUTH)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_daily_cap_protects_account_quota():
    c, up = client(daily_cap=2)
    with c:
        codes = [c.post("/api/llm", json=body(f"q{i}"), headers=AUTH).status_code for i in range(4)]
    assert codes == [200, 200, 429, 429] and len(up.calls) == 2


def test_global_rpm_cap_stays_under_upstream_limit():
    c, up = client(global_rpm=2)
    with c:
        codes = [c.post("/api/llm", json=body(f"q{i}"), headers=AUTH).status_code for i in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:] and len(up.calls) == 2


def test_paid_models_refused_unless_opted_in():
    with pytest.raises(RuntimeError, match="not a ':free' model"):
        create_app(cfg(models=("fake/paid-model",)), transport=httpx.MockTransport(Upstream()))
    create_app(cfg(models=("fake/paid-model",), allow_paid_models=True), transport=httpx.MockTransport(Upstream()))


def test_startup_refuses_unsafe_config():
    with pytest.raises(RuntimeError):
        create_app(cfg(api_key=""), transport=httpx.MockTransport(Upstream()))
    with pytest.raises(RuntimeError):
        create_app(cfg(tokens={}), transport=httpx.MockTransport(Upstream()))
    with pytest.raises(RuntimeError):
        create_app(cfg(allowed_origins=("*",)), transport=httpx.MockTransport(Upstream()))
    with pytest.raises(RuntimeError):
        create_app(cfg(models=()), transport=httpx.MockTransport(Upstream()))


# ---------------------------------------------------------------- speed and resilience
def test_cache_serves_repeat_deterministic_calls_without_upstream():
    c, up = client()
    with c:
        a = c.post("/api/llm", json=body(temperature=0), headers=AUTH).json()
        b = c.post("/api/llm", json=body(temperature=0), headers=AUTH).json()
    assert len(up.calls) == 1 and a["cached"] is False and b["cached"] is True and a["content"] == b["content"]


def test_random_calls_are_not_cached():
    c, up = client()
    with c:
        c.post("/api/llm", json=body(temperature=0.7), headers=AUTH)
        c.post("/api/llm", json=body(temperature=0.7), headers=AUTH)
    assert len(up.calls) == 2


def test_fallback_to_second_model_on_429():
    c, up = client(Upstream([httpx.Response(429), httpx.Response(429)]))
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 200 and r.json()["model"] == MODELS[1]
    assert [call["body"]["model"] for call in up.calls] == [MODELS[0], MODELS[0], MODELS[1]]


def test_circuit_breaker_skips_a_failing_model():
    class OnlyAFails(Upstream):
        def __call__(self, request):
            resp = super().__call__(request)
            return httpx.Response(503) if self.calls[-1]["body"]["model"] == MODELS[0] else resp

    up = OnlyAFails()
    c, _ = client(up, breaker_failures=3, user_rpm=100)
    with c:
        for i in range(3):  # three questions each exhaust model A (two attempts) and fall back to B
            r = c.post("/api/llm", json=body(f"q{i}", temperature=0.5), headers=AUTH)
            assert r.status_code == 200 and r.json()["model"] == MODELS[1]
        up.calls.clear()
        c.post("/api/llm", json=body("later", temperature=0.5), headers=AUTH)
    assert [call["body"]["model"] for call in up.calls] == [MODELS[1]], "open breaker must skip model A entirely"


def test_all_models_down_gives_generic_503():
    c, _ = client(Upstream([httpx.Response(500)] * 30))
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 503 and "request_id" in r.json()


def test_billing_or_key_problem_is_a_generic_503_not_a_leak():
    c, _ = client(Upstream([httpx.Response(402, text=f"Insufficient credits for {KEY}")]))
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 503 and KEY not in r.text and "credits" not in r.text.lower()


def test_malformed_upstream_body_is_handled():
    c, _ = client(Upstream([httpx.Response(200, json={"weird": 1})] * 2 + [ok("fine")]))
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 200 and r.json()["content"] == "fine"


def test_single_flight_identical_concurrent_requests_share_one_call():
    import threading
    import time as _t

    class Slow(Upstream):
        def __call__(self, request):
            _t.sleep(0.3)
            return super().__call__(request)

    up = Slow()
    c, _ = client(up)
    results = []
    with c:
        def go():
            results.append(c.post("/api/llm", json=body("same question", temperature=0), headers=AUTH).status_code)
        ts = [threading.Thread(target=go) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert results == [200] * 4 and len(up.calls) == 1


# ---------------------------------------------------------------- streaming
def sse(*chunks, usage=None):
    lines = [": OPENROUTER PROCESSING\n"]
    for ch in chunks:
        lines.append("data: " + json.dumps({"id": "gen-secret", "choices": [{"delta": {"content": ch}}]}) + "\n")
    if usage:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}) + "\n")
    lines.append("data: [DONE]\n")
    return httpx.Response(200, content="\n".join(lines).encode(), headers={"content-type": "text/event-stream"})


def events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def test_streaming_yields_only_sanitised_events():
    c, _ = client(Upstream([sse("Hel", "lo", usage={"prompt_tokens": 5, "completion_tokens": 2})]))
    with c:
        r = c.post("/api/llm/stream", json=body(), headers=AUTH)
    ev = events(r)
    assert "".join(e.get("delta", "") for e in ev) == "Hello"
    assert ev[-1]["done"] is True and ev[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}
    assert "gen-secret" not in r.text and KEY not in r.text


def test_streaming_falls_back_before_first_token():
    c, _ = client(Upstream([httpx.Response(429), sse("from b")]))
    with c:
        r = c.post("/api/llm/stream", json=body(), headers=AUTH)
    ev = events(r)
    assert ev[0]["delta"] == "from b" and ev[-1]["model"] == MODELS[1]


def test_streaming_requires_auth():
    c, up = client()
    with c:
        assert c.post("/api/llm/stream", json=body()).status_code == 401
    assert up.calls == []


# ---------------------------------------------------------------- browser-facing hardening
def test_security_headers_present():
    c, _ = client()
    with c:
        r = c.post("/api/llm", json=body(), headers=AUTH)
    h = r.headers
    assert h["x-content-type-options"] == "nosniff" and h["cache-control"] == "no-store"
    assert h["referrer-policy"] == "no-referrer" and "frame-ancestors 'none'" in h["content-security-policy"]
    assert h["x-request-id"]


def test_cors_allows_only_listed_origins():
    c, _ = client()
    with c:
        good = c.options("/api/llm", headers={"Origin": "https://app.example.com", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"})
        bad = c.options("/api/llm", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
        ok_get = c.get("/healthz", headers={"Origin": "https://evil.example"})
    assert good.headers.get("access-control-allow-origin") == "https://app.example.com"
    assert "access-control-allow-origin" not in bad.headers
    assert "access-control-allow-origin" not in ok_get.headers
    assert good.headers.get("access-control-allow-credentials") is None


def test_unhandled_error_is_generic(monkeypatch):
    c, _ = client()
    async def explode(self, req):
        raise ValueError(f"internal detail {KEY}")
    monkeypatch.setattr(gateway.Gateway, "chat", explode)
    with TestClient(c.app, raise_server_exceptions=False) as tc:
        r = tc.post("/api/llm", json=body(), headers=AUTH)
    assert r.status_code == 500 and KEY not in r.text and "internal detail" not in r.text


def test_budget_endpoint_needs_auth_and_shows_no_secrets():
    c, _ = client()
    with c:
        assert c.get("/api/llm/budget").status_code == 401
        r = c.get("/api/llm/budget", headers=AUTH)
    assert r.status_code == 200 and set(r.json()) == {"daily_used", "daily_cap", "global_rpm"}


def test_token_helper_prints_hash_not_stored_token(capsys, monkeypatch):
    import runpy, sys
    monkeypatch.setattr(sys, "argv", ["gateway.py", "newtoken", "bob"])
    with pytest.raises(SystemExit):
        runpy.run_path(gateway.__file__, run_name="__main__")
    out = capsys.readouterr().out
    assert "GATEWAY_TOKEN_HASHES: bob:" in out

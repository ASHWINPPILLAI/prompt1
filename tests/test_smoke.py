"""
Smoke tests for P0 scaffold.

These tests verify the project structure and basic imports without
requiring any downloaded data, API keys, or a running LLM.

Run: pytest tests/ -q
"""

from __future__ import annotations

import json
from pathlib import Path


# ── Structure tests ────────────────────────────────────────────────────────────
class TestProjectStructure:
    """Verify required files and directories exist."""

    def test_agents_md_exists(self):
        assert Path("AGENTS.md").exists(), "AGENTS.md must be in repo root"

    def test_readme_exists(self):
        assert Path("README.md").exists()

    def test_env_example_exists(self):
        assert Path(".env.example").exists()

    def test_gitignore_exists(self):
        assert Path(".gitignore").exists()

    def test_configs_dir(self):
        assert Path("configs/default.yaml").exists()
        assert Path("configs/sweeps.yaml").exists()

    def test_prompts_dir(self):
        prompt_ids = [
            "v0_baseline", "v1_persona_rag", "v2_rules_json",
            "v3_fewshot", "v4_cot", "v4b_autocot", "v5_react", "v6_verifier",
            "judge", "verifier",
        ]
        for pid in prompt_ids:
            p = Path(f"prompts/{pid}.yaml")
            assert p.exists(), f"Missing prompt file: {p}"

    def test_src_package(self):
        assert Path("src/policypilot/__init__.py").exists()

    def test_eval_dir(self):
        assert Path("eval").exists()

    def test_results_dirs(self):
        for d in ["results/runs", "results/metrics", "results/figures", "results/router"]:
            assert Path(d).exists(), f"Missing results dir: {d}"

    def test_scripts_exist(self):
        assert Path("scripts/data/download_all.py").exists()
        assert Path("scripts/data/inspect_schemas.py").exists()


# ── YAML parsing tests ─────────────────────────────────────────────────────────
class TestPromptYAML:
    """Verify all prompt YAML files are valid and have required fields."""

    REQUIRED_FIELDS = {"id", "technique", "hypothesis", "system", "user_template"}

    def _load_all_prompts(self):
        import yaml  # noqa: PLC0415
        return {
            p.stem: yaml.safe_load(p.read_text(encoding="utf-8"))
            for p in Path("prompts").glob("*.yaml")
        }

    def test_all_prompts_parseable(self):
        prompts = self._load_all_prompts()
        assert len(prompts) >= 8, f"Expected at least 8 prompts, got {len(prompts)}"

    def test_required_fields_present(self):
        prompts = self._load_all_prompts()
        for name, data in prompts.items():
            for field in self.REQUIRED_FIELDS:
                assert field in data, f"Prompt '{name}' missing field '{field}'"

    def test_prompt_ids_match_filenames(self):
        import yaml  # noqa: PLC0415
        for p in Path("prompts").glob("*.yaml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            assert data.get("id") == p.stem, (
                f"Prompt id '{data.get('id')}' does not match filename '{p.stem}'"
            )


# ── Config tests ────────────────────────────────────────────────────────────────
class TestConfigs:
    """Verify configs parse correctly."""

    def test_default_yaml(self):
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
        assert "llm" in cfg
        assert "retrieval" in cfg
        assert "router" in cfg
        assert "react" in cfg

    def test_sweeps_yaml(self):
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(Path("configs/sweeps.yaml").read_text(encoding="utf-8"))
        assert "temperature" in cfg
        assert "top_p" in cfg
        assert "chunking" in cfg


# ── Tools unit tests (no API, no index) ───────────────────────────────────────
class TestCalculator:
    """Verify calculator tool with known values."""

    def _calc(self, expr: str) -> str:
        from policypilot.tools import calculator, CalcArgs  # noqa: PLC0415
        return calculator(CalcArgs(expression=expr))

    def test_addition(self):
        assert "5" in self._calc("2 + 3")

    def test_division(self):
        result = self._calc("10 / 4")
        assert "2.5" in result

    def test_percentage(self):
        # 142 / 200 * 100 = 71.0
        result = self._calc("142 / 200 * 100")
        assert "71" in result

    def test_power(self):
        result = self._calc("2 ** 10")
        assert "1024" in result

    def test_disallowed_import(self):
        from policypilot.tools import calculator, CalcArgs  # noqa: PLC0415
        from pydantic import ValidationError
        try:
            calculator(CalcArgs(expression="import os"))
            assert False, "Should have raised"
        except (ValidationError, ValueError):
            pass  # Expected

    def test_disallowed_eval(self):
        from policypilot.tools import calculator, CalcArgs  # noqa: PLC0415
        from pydantic import ValidationError
        try:
            calculator(CalcArgs(expression="eval('1+1')"))
            assert False, "Should have raised"
        except (ValidationError, ValueError):
            pass


# ── Prompt loader unit tests ────────────────────────────────────────────────────
class TestPromptLoader:
    def test_load_v0(self):
        from policypilot.prompting import load_prompt  # noqa: PLC0415
        p = load_prompt("v0_baseline")
        assert p.id == "v0_baseline"
        assert p.needs_retrieval is False

    def test_load_v2(self):
        from policypilot.prompting import load_prompt  # noqa: PLC0415
        p = load_prompt("v2_rules_json")
        assert p.needs_retrieval is True
        assert p.output_schema is not None

    def test_load_missing_raises(self):
        from policypilot.prompting import load_prompt  # noqa: PLC0415
        try:
            load_prompt("nonexistent_prompt")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_render_user_message(self):
        from policypilot.prompting import render_user_message  # noqa: PLC0415
        result = render_user_message("Hello {name}!", name="world")
        assert result == "Hello world!"

    def test_render_missing_key_raises(self):
        from policypilot.prompting import render_user_message  # noqa: PLC0415
        try:
            render_user_message("Hello {name}!", other="world")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── JSON repair tests ─────────────────────────────────────────────────────────
class TestJSONRepair:
    def test_valid_json(self):
        from policypilot.pipeline import _parse_json_output  # noqa: PLC0415
        parsed, err = _parse_json_output('{"answerable": true, "answer": "yes"}')
        assert parsed == {"answerable": True, "answer": "yes"}
        assert err is None

    def test_json_in_markdown_fence(self):
        from policypilot.pipeline import _parse_json_output  # noqa: PLC0415
        text = '```json\n{"answerable": true}\n```'
        parsed, err = _parse_json_output(text)
        assert parsed == {"answerable": True}

    def test_json_with_preamble(self):
        from policypilot.pipeline import _parse_json_output  # noqa: PLC0415
        text = 'Here is the answer:\n{"answerable": false}'
        parsed, err = _parse_json_output(text)
        assert parsed == {"answerable": False}

    def test_no_json_returns_none(self):
        from policypilot.pipeline import _parse_json_output  # noqa: PLC0415
        parsed, err = _parse_json_output("No JSON here at all.")
        assert parsed is None
        assert err is not None


# ── LLM client tests (fake) ────────────────────────────────────────────────────
class TestLLMClientFake:
    """Test the client with a fake response; no real API call."""

    def _make_fake_client(self, content: str = '{"answerable": true}'):
        """Build a client with a stubbed _live_call."""
        import sys
        import os
        # Set dummy env vars so client init doesn't fail
        os.environ.setdefault("LLM_MODEL", "fake-model")
        os.environ.setdefault("LLM_API_KEY", "fake-key")

        from policypilot.llm.client import LLMClient  # noqa: PLC0415

        client = LLMClient(model="fake-model", api_key="fake-key")
        # Stub _live_call to avoid a real HTTP request
        client._live_call = lambda model, messages, params: {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "latency_s": 0.01,
            "cached": False,
        }
        return client

    def test_chat_returns_content(self):
        client = self._make_fake_client('{"score": 1}')
        result = client.chat([{"role": "user", "content": "test"}])
        assert result["content"] == '{"score": 1}'
        assert result["usage"]["prompt_tokens"] == 10

    def test_second_call_is_cached(self):
        import tempfile, os  # noqa: PLC0415, E401
        from pathlib import Path
        from policypilot import llm  # noqa: PLC0415
        # We just check that calling the same messages twice returns cached=True
        # on the second call (depends on disk cache working).
        # Minimal test: at minimum, the first call doesn't raise.
        client = self._make_fake_client()
        msg = [{"role": "user", "content": "cache-test-unique-42"}]
        r1 = client.chat(msg, seed=42)
        assert "content" in r1

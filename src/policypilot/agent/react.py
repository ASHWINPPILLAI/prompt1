"""
ReAct agent loop (V5, V6).

Text-based Thought/Action/Observation loop — every step is visible in the demo.
WHY text-based rather than function-calling? The spec requires a visible trace,
and text parsing keeps the format model-agnostic (works with any OpenAI-compatible
endpoint, not just those that support function-calling).

Protocol:
  Thought: what the model needs next
  Action: tool_name[{"key": "value"}]
  Observation: (system fills in)
  ...up to max_steps...
  Final Answer: <JSON or plain text>
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from policypilot.llm.client import LLMClient
from policypilot.tools import call_tool

logger = logging.getLogger(__name__)

# Regex: exactly one Action per line, tool_name[{json}]
_ACTION_RE = re.compile(r"Action:\s*(\w+)\[(.+)\]\s*$", re.MULTILINE | re.DOTALL)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)
_TOOL_OUTPUT_CHARS = 1200  # Truncate long tool outputs (see configs/default.yaml)


class ReActAgent:
    """
    Text-based ReAct loop with a configurable step limit and trace logging.

    Each step is logged as {step, thought, action, args, observation, tokens}
    in react_trace, which the app renders and the run_query pipeline stores.
    """

    def __init__(
        self,
        client: LLMClient,
        max_steps: int = 5,
        tool_output_chars: int = _TOOL_OUTPUT_CHARS,
    ) -> None:
        self.client = client
        self.max_steps = max_steps
        self.tool_output_chars = tool_output_chars

    def run(
        self,
        messages: list[dict[str, str]],
        *,
        run_id: str = "",
        prompt_id: str = "",
    ) -> dict[str, Any]:
        """
        Execute the ReAct loop.

        Returns:
          final_answer   — string (may be JSON or plain text)
          react_trace    — list of step dicts
          total_tokens   — cumulative token count
          latency_s      — wall time
        """
        # WHY stop=["Observation:"]? Prevents the model from inventing its own
        # observations; the system always fills them in.
        conversation = list(messages)
        trace: list[dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        t0 = time.perf_counter()

        for step in range(self.max_steps):
            # On the last step, explicitly ask for Final Answer
            if step == self.max_steps - 1:
                conversation.append(
                    {"role": "user", "content": "You must now give the Final Answer."}
                )

            response = self.client.chat(
                conversation,
                stop=["Observation:"],
                temperature=0.0,  # Reproducible tool calls
                top_p=1.0,
                max_tokens=300,
                run_id=run_id,
                prompt_id=prompt_id,
            )
            model_text = response["content"].strip()
            total_prompt_tokens += response["usage"]["prompt_tokens"]
            total_completion_tokens += response["usage"]["completion_tokens"]

            # Check for Final Answer
            final_match = _FINAL_RE.search(model_text)
            if final_match:
                trace.append(
                    {
                        "step": step,
                        "type": "final",
                        "text": model_text,
                        "tokens": response["usage"],
                    }
                )
                return {
                    "final_answer": final_match.group(1).strip(),
                    "react_trace": trace,
                    "total_tokens": {
                        "prompt": total_prompt_tokens,
                        "completion": total_completion_tokens,
                    },
                    "latency_s": time.perf_counter() - t0,
                }

            # Parse Action
            action_match = _ACTION_RE.search(model_text)
            if not action_match:
                # Malformed step — log and continue
                observation = "ERROR: Could not parse Action. Use format: Action: tool_name[{json}]"
                trace.append(
                    {
                        "step": step,
                        "type": "malformed",
                        "text": model_text,
                        "observation": observation,
                        "tokens": response["usage"],
                    }
                )
            else:
                tool_name = action_match.group(1)
                raw_args_str = action_match.group(2).strip()
                try:
                    raw_args = json.loads(raw_args_str)
                except json.JSONDecodeError:
                    raw_args = {}
                    observation = f"ERROR: Invalid JSON arguments: {raw_args_str[:100]}"
                else:
                    observation = call_tool(tool_name, raw_args)
                    # Truncate long outputs and signal truncation
                    if len(observation) > self.tool_output_chars:
                        observation = observation[: self.tool_output_chars] + " [TRUNCATED]"

                trace.append(
                    {
                        "step": step,
                        "type": "action",
                        "thought": self._extract_thought(model_text),
                        "action": tool_name,
                        "args": raw_args_str,
                        "observation": observation,
                        "tokens": response["usage"],
                    }
                )

            # Append model output + observation to conversation
            conversation.append({"role": "assistant", "content": model_text})
            conversation.append(
                {"role": "user", "content": f"Observation: {observation}"}
            )

        # Exceeded max_steps without Final Answer
        fallback = "The documents do not provide enough information to answer this question."
        trace.append({"step": self.max_steps, "type": "timeout", "text": fallback})
        return {
            "final_answer": fallback,
            "react_trace": trace,
            "total_tokens": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
            },
            "latency_s": time.perf_counter() - t0,
        }

    @staticmethod
    def _extract_thought(text: str) -> str:
        """Extract Thought: line for the trace log."""
        for line in text.splitlines():
            if line.startswith("Thought:"):
                return line[8:].strip()
        return ""

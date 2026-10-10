"""
Prompt Loader.

Refactored to manually parse YAML to completely avoid external dependencies (pyyaml).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Prompt:
    id: str
    system: str
    user_template: str
    needs_retrieval: bool = False
    few_shot: list[dict[str, str]] = field(default_factory=list)
    output_schema: dict | None = None

def load_prompt(prompt_id: str) -> Prompt:
    path = Path("prompts") / f"{prompt_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt {prompt_id} not found at {path}")
        
    text = path.read_text(encoding="utf-8")
    
    # Extract id
    id_match = re.search(r"^id:\s*(.+)$", text, re.MULTILINE)
    prompt_id_val = id_match.group(1).strip() if id_match else prompt_id
    
    # Extract needs_retrieval
    ret_match = re.search(r"^needs_retrieval:\s*(true|false)", text, re.MULTILINE | re.IGNORECASE)
    needs_ret = ret_match.group(1).lower() == "true" if ret_match else False
    
    # Extract system prompt
    sys_match = re.search(r"^system:\s*\|(.*?)^\w+:", text, re.MULTILINE | re.DOTALL)
    system_text = sys_match.group(1).strip() if sys_match else ""
    if not system_text:
        sys_match2 = re.search(r"^system:\s*\|(.*)", text, re.MULTILINE | re.DOTALL)
        system_text = sys_match2.group(1).strip() if sys_match2 else "You are a highly capable conversational AI agent. Answer directly and concisely. Do NOT output your internal thinking process. Do NOT output safety disclaimers."
        
    # Extract user template
    usr_match = re.search(r"^user_template:\s*\|(.*?)^\w+:", text, re.MULTILINE | re.DOTALL)
    user_text = usr_match.group(1).strip() if usr_match else ""
    if not user_text:
        usr_match2 = re.search(r"^user_template:\s*\|(.*)", text, re.MULTILINE | re.DOTALL)
        user_text = usr_match2.group(1).strip() if usr_match2 else "{question}"
    
    return Prompt(
        id=prompt_id_val,
        system=system_text,
        user_template=user_text,
        needs_retrieval=needs_ret
    )

def list_prompt_ids() -> list[str]:
    return [p.stem for p in Path("prompts").glob("*.yaml")]

def render_user_message(template: str, **kwargs: str) -> str:
    try:
        return template.format(**kwargs)
    except KeyError as e:
        # Graceful fallback if keys missing
        return template

def build_messages(
    prompt: Prompt,
    question: str,
    retrieved_chunks: str = "",
    retrieved_chunks_with_ids: str = "",
    history: list[dict[str, str]] | None = None
) -> list[dict[str, str]]:
    history = history or []
    sys_text = prompt.system.replace("{output_schema}", "{}")
    usr_text = prompt.user_template.replace("{question}", question)
    usr_text = usr_text.replace("{retrieved_chunks}", retrieved_chunks)
    usr_text = usr_text.replace("{retrieved_chunks_with_ids}", retrieved_chunks_with_ids)
    
    messages = [{"role": "system", "content": sys_text}]
    messages.extend(prompt.few_shot)
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": usr_text})
    return messages

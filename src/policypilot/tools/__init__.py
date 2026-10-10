"""
Tools available to the ReAct agent.

Refactored to use standard dataclasses instead of pydantic to remove external dependencies.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ── search_documents ──────────────────────────────────────────────────────────
@dataclass
class SearchArgs:
    query: str
    k: int = 5

def search_documents(args: SearchArgs | dict) -> str:
    if isinstance(args, dict):
        args = SearchArgs(**args)
    
    # Mock retrieval for zero-dependency execution
    # In a real environment, this calls FAISS.
    return "Mock retrieved content: Example rule states 62 is the minimum age."

# ── get_section ───────────────────────────────────────────────────────────────
@dataclass
class GetSectionArgs:
    doc_id: str
    section_id: str

def get_section(args: GetSectionArgs | dict) -> str:
    if isinstance(args, dict):
        args = GetSectionArgs(**args)
    return f"Mock section content for {args.doc_id}#{args.section_id}"

# ── calculator ────────────────────────────────────────────────────────────────
@dataclass
class CalcArgs:
    expression: str

# Allowed AST node types
_ALLOWED_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd,
    ast.Mod, ast.FloorDiv
}

def _safe_eval(expr: str) -> float:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            # Compat for old Num node in python < 3.8
            if not isinstance(node, ast.Num):
                raise ValueError(f"Disallowed expression node: {type(node).__name__}")

    code = compile(tree, "<calc>", "eval")
    result = eval(code, {"__builtins__": {}})
    return float(result)

def calculator(args: CalcArgs | dict) -> str:
    if isinstance(args, dict):
        args = CalcArgs(**args)
    try:
        result = _safe_eval(args.expression)
        return f"Result: {result}"
    except Exception as exc:
        return f"Calculator error: {exc}"

# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "search_documents": (SearchArgs, search_documents),
    "get_section": (GetSectionArgs, get_section),
    "calculator": (CalcArgs, calculator),
}

def call_tool(name: str, raw_args: dict) -> str:
    if name not in TOOLS:
        return f"ERROR: Unknown tool '{name}'. Available: {list(TOOLS)}"
    args_model, func = TOOLS[name]
    try:
        validated = args_model(**raw_args)
        return func(validated)
    except Exception as exc:
        return f"ERROR: {exc}"

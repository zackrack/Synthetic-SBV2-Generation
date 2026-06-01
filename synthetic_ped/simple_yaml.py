"""Tiny YAML/JSON loader for the repository's simple configuration files.

Supports the subset used by the example configs: top-level mappings, one-level
nested mappings, scalars, block lists, and inline lists such as ``[TH, S]``. If
PyYAML is installed, it is used instead.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

_PENDING_CONTAINER = object()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError:
        loaded = _load_simple_yaml(text)
    else:
        loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in {path}")
    return loaded


def _strip_comment(line: str) -> str:
    in_quote: str | None = None
    for i, char in enumerate(line):
        if char in {'"', "'"}:
            in_quote = None if in_quote == char else char if in_quote is None else in_quote
        elif char == "#" and in_quote is None:
            return line[:i]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value.startswith("[") or value.startswith("{"):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            inner = value[1:-1].strip()
            return [] if not inner else [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current_key: str | None = None
    current_container: dict[str, Any] | list[Any] | None = None
    current_indent = 0

    lines = [_strip_comment(line).rstrip() for line in text.splitlines()]
    for raw_line in lines:
        if not raw_line.strip():
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        if indent == 0:
            if ":" not in stripped:
                raise ValueError(f"Unsupported YAML line: {raw_line!r}")
            key, value = stripped.split(":", 1)
            key = key.strip()
            value = value.strip()
            current_key = key
            current_indent = indent
            if value:
                root[key] = _parse_scalar(value)
                current_container = None
            else:
                # Decide list vs mapping lazily from the first indented child.
                root[key] = _PENDING_CONTAINER
                current_container = None
            continue

        if current_key is None:
            raise ValueError(f"Indented YAML line without parent: {raw_line!r}")

        if stripped.startswith("-"):
            if not isinstance(root.get(current_key), list):
                root[current_key] = []
            container = root[current_key]
            assert isinstance(container, list)
            container.append(_parse_scalar(stripped[1:].strip()))
            current_container = container
            continue

        if ":" in stripped:
            if not isinstance(root.get(current_key), dict):
                root[current_key] = {}
            container = root[current_key]
            assert isinstance(container, dict)
            key, value = stripped.split(":", 1)
            container[key.strip()] = _parse_scalar(value.strip())
            current_container = container
            continue

        raise ValueError(f"Unsupported YAML line: {raw_line!r}")

    for key, value in list(root.items()):
        if value is _PENDING_CONTAINER:
            root[key] = {}
    return root

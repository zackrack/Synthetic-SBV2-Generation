from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_ped.simple_yaml import load_config


@dataclass(frozen=True)
class ContextTemplate:
    context_type: str
    template: str

    def render(self, word: str) -> str:
        return self.template.format(word=word)


def load_context_templates(path: Path) -> list[ContextTemplate]:
    data = load_config(path)
    contexts = data.get("contexts", data)
    if not isinstance(contexts, dict):
        raise ValueError("context templates must be a mapping")
    templates = [ContextTemplate(str(key), str(value)) for key, value in contexts.items()]
    expected = {"word", "sentence", "passage"}
    missing = expected - {t.context_type for t in templates}
    if missing:
        raise ValueError(f"Missing context templates: {sorted(missing)}")
    return templates

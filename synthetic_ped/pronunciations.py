from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PronunciationDictionary:
    entries: dict[str, list[str]]

    @classmethod
    def from_path(cls, path: Path) -> "PronunciationDictionary":
        entries: dict[str, list[str]] = {}
        if not path.exists():
            return cls(entries)
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            if "(" in word and word.endswith(")"):
                word = word[: word.index("(")]
            entries.setdefault(word, [_normalize_phone(p) for p in parts[1:]])
        return cls(entries)

    def get(self, word: str) -> list[str] | None:
        phones = self.entries.get(word.lower())
        return list(phones) if phones is not None else None


def _normalize_phone(phone: str) -> str:
    # CMUdict carries stress on vowels (AH0, EY1). PED-style inventories are
    # phonemic, so stress is removed for canonical/edit operations.
    return "".join(ch for ch in phone.upper() if not ch.isdigit())


def read_words(path: Path) -> list[str]:
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            words.append(stripped.split()[0])
    return words

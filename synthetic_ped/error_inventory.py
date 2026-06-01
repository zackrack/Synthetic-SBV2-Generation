from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_ped.simple_yaml import load_config


@dataclass(frozen=True)
class ErrorInventory:
    vowels: tuple[tuple[str, str], ...]
    consonants: tuple[tuple[str, str], ...]
    consonant_set: frozenset[str]

    @classmethod
    def from_path(cls, path: Path) -> "ErrorInventory":
        data = load_config(path)
        vowels = tuple(_pairs(data.get("vowels", []), "vowels"))
        consonants = tuple(_pairs(data.get("consonants", []), "consonants"))
        consonant_set = frozenset(str(p).upper() for p in data.get("consonant_phones", []))
        if not consonant_set:
            consonant_set = frozenset({target for target, _ in consonants})
        return cls(vowels=vowels, consonants=consonants, consonant_set=consonant_set)

    @property
    def substitutions_by_target(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for target, replacement in (*self.vowels, *self.consonants):
            out.setdefault(target, []).append(replacement)
        return out

    @property
    def substitution_pairs(self) -> set[tuple[str, str]]:
        return set(self.vowels) | set(self.consonants)

    def is_consonant(self, phone: str) -> bool:
        return phone.upper() in self.consonant_set


def _pairs(value: object, name: str) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of [target, replacement] pairs")
    pairs: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid {name} pair: {item!r}")
        pairs.append((str(item[0]).upper(), str(item[1]).upper()))
    return pairs

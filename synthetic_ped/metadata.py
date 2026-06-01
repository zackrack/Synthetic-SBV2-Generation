from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ManifestRow:
    item_id: str
    source_word: str
    context_type: str
    text_prompt: str
    canonical_phones: list[str]
    modified_phones: list[str]
    canonical_ipa: str | None
    modified_ipa: str | None
    error_type: str
    target_index: int
    target_phone: str
    replacement_phone: str | None
    phone_inventory: str
    sbv2_phone_input: list[str]
    audio_path: str
    speaker_id: str | None = None
    model_id: str | None = None
    style: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    generation_status: str = "pending"
    validation_status: str = "unverified"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ManifestRow":
        return cls(**data)


def append_jsonl(path: Path, row: ManifestRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(ManifestRow.from_dict(json.loads(line)))
    return rows


def write_jsonl(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ManifestRow.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = row.to_dict()
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    data[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
            writer.writerow(data)

from __future__ import annotations

import csv
import json
from pathlib import Path

from synthetic_ped.metadata import ManifestRow

REVIEW_COLUMNS = [
    "item_id", "audio_path", "text_prompt", "canonical_phones", "modified_phones",
    "error_type", "target_phone", "replacement_phone", "reviewer_decision", "reviewer_notes",
]


def write_review_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "item_id": row.item_id,
                "audio_path": row.audio_path,
                "text_prompt": row.text_prompt,
                "canonical_phones": " ".join(row.canonical_phones),
                "modified_phones": " ".join(row.modified_phones),
                "error_type": row.error_type,
                "target_phone": row.target_phone,
                "replacement_phone": row.replacement_phone or "",
                "reviewer_decision": "",
                "reviewer_notes": "",
            })


def filter_accepted(review_csv: Path, manifest_rows: list[ManifestRow]) -> list[ManifestRow]:
    accepted: set[str] = set()
    with review_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("reviewer_decision", "").strip().lower() in {"accept", "accepted", "yes", "y"}:
                accepted.add(row["item_id"])
    return [row for row in manifest_rows if row.item_id in accepted]

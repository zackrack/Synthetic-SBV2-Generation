from __future__ import annotations

from dataclasses import dataclass

from synthetic_ped.error_inventory import ErrorInventory
from synthetic_ped.metadata import ManifestRow


@dataclass(frozen=True)
class ValidationResult:
    status: str
    notes: str = ""


def one_edit_matches(row: ManifestRow) -> bool:
    canonical = row.canonical_phones
    modified = row.modified_phones
    idx = row.target_index
    if row.error_type == "substitution":
        return (
            len(canonical) == len(modified)
            and 0 <= idx < len(canonical)
            and canonical[:idx] == modified[:idx]
            and canonical[idx + 1 :] == modified[idx + 1 :]
            and canonical[idx] == row.target_phone
            and modified[idx] == row.replacement_phone
            and canonical[idx] != modified[idx]
        )
    if row.error_type == "deletion":
        return (
            len(canonical) == len(modified) + 1
            and 0 <= idx < len(canonical)
            and canonical[idx] == row.target_phone
            and canonical[:idx] + canonical[idx + 1 :] == modified
            and row.replacement_phone is None
        )
    return False


def validate_row(row: ManifestRow, inventory: ErrorInventory) -> ValidationResult:
    missing = []
    for attr in ["text_prompt", "canonical_phones", "modified_phones", "sbv2_phone_input", "audio_path"]:
        if not getattr(row, attr):
            missing.append(attr)
    if missing:
        return ValidationResult("invalid", f"missing required fields: {', '.join(missing)}")
    if not one_edit_matches(row):
        return ValidationResult("invalid", "modified phones are not the configured one-edit change")
    if row.error_type == "deletion" and not inventory.is_consonant(row.target_phone):
        return ValidationResult("invalid", "deletion target is not a configured consonant")
    if row.error_type == "substitution" and (row.target_phone, row.replacement_phone or "") not in inventory.substitution_pairs:
        return ValidationResult("invalid", "substitution pair is not in the configured inventory")
    return ValidationResult("unverified", "passed structural validation; awaiting human/audio review")


def validate_rows(rows: list[ManifestRow], inventory: ErrorInventory) -> list[ManifestRow]:
    validated: list[ManifestRow] = []
    for row in rows:
        result = validate_row(row, inventory)
        row.validation_status = result.status
        row.notes = "; ".join(part for part in [row.notes, result.notes] if part)
        validated.append(row)
    return validated

from __future__ import annotations

import logging
from dataclasses import dataclass

from synthetic_ped.error_inventory import ErrorInventory


LOGGER = logging.getLogger("synthetic_ped")


@dataclass(frozen=True)
class Variant:
    source_word: str
    canonical_phones: list[str]
    modified_phones: list[str]
    error_type: str
    target_index: int
    target_phone: str
    replacement_phone: str | None
    notes: str = ""


def generate_variants(
    word: str, canonical_phones: list[str], inventory: ErrorInventory
) -> list[Variant]:
    variants: list[Variant] = []
    seen: set[tuple[str, ...]] = set()
    substitutions = inventory.substitutions_by_target
    for index, target in enumerate(canonical_phones):
        target = target.upper()
        for replacement in substitutions.get(target, []):
            modified = list(canonical_phones)
            modified[index] = replacement
            key = tuple(modified)
            if key in seen:
                LOGGER.info(
                    "Skipping duplicate substitution variant for %s at index %d: %s",
                    word,
                    index,
                    " ".join(modified),
                )
                continue
            seen.add(key)
            variants.append(
                Variant(
                    source_word=word,
                    canonical_phones=list(canonical_phones),
                    modified_phones=modified,
                    error_type="substitution",
                    target_index=index,
                    target_phone=target,
                    replacement_phone=replacement,
                )
            )
        if inventory.is_consonant(target):
            modified = list(canonical_phones[:index] + canonical_phones[index + 1 :])
            key = tuple(modified)
            if key in seen:
                LOGGER.info(
                    "Skipping duplicate deletion variant for %s at index %d: %s",
                    word,
                    index,
                    " ".join(modified),
                )
                continue
            seen.add(key)
            variants.append(
                Variant(
                    source_word=word,
                    canonical_phones=list(canonical_phones),
                    modified_phones=modified,
                    error_type="deletion",
                    target_index=index,
                    target_phone=target,
                    replacement_phone=None,
                )
            )
    return variants

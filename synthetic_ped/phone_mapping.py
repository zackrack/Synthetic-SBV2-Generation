from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synthetic_ped.simple_yaml import load_config


class PhoneMappingError(ValueError):
    pass


@dataclass(frozen=True)
class PhoneMapper:
    mapping: dict[str, str]
    inventory_name: str = "CMU-39-to-SBV2"

    @classmethod
    def from_path(cls, path: Path) -> "PhoneMapper":
        data = load_config(path)
        raw_mapping = data.get("mapping", data)
        if not isinstance(raw_mapping, dict):
            raise ValueError("phone mapping must be a mapping")
        inventory_name = str(data.get("phone_inventory", "CMU-39-to-SBV2"))
        return cls({str(k).upper(): str(v) for k, v in raw_mapping.items()}, inventory_name)

    def map_sequence(self, phones: list[str]) -> list[str]:
        missing = [phone for phone in phones if phone.upper() not in self.mapping]
        if missing:
            raise PhoneMappingError(f"Missing SBV2 phone mappings for: {', '.join(sorted(set(missing)))}")
        return [self.mapping[phone.upper()] for phone in phones]

import pytest

from synthetic_ped.phone_mapping import PhoneMapper, PhoneMappingError


def test_phone_mapping_failure_behavior():
    mapper = PhoneMapper({"TH": "th"})
    with pytest.raises(PhoneMappingError):
        mapper.map_sequence(["TH", "AE"])

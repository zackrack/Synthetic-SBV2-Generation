from synthetic_ped.error_inventory import ErrorInventory
from synthetic_ped.variants import generate_variants


def inventory():
    return ErrorInventory(
        vowels=(("AE", "EH"),),
        consonants=(("TH", "S"),),
        consonant_set=frozenset({"TH", "N", "K"}),
    )


def test_substitution_generation():
    variants = generate_variants("thank", ["TH", "AE", "N", "K"], inventory())
    sub = [v for v in variants if v.error_type == "substitution" and v.target_phone == "TH"]
    assert len(sub) == 1
    assert sub[0].modified_phones == ["S", "AE", "N", "K"]
    assert sub[0].replacement_phone == "S"


def test_consonant_deletion_generation():
    variants = generate_variants("thank", ["TH", "AE", "N", "K"], inventory())
    deletions = [v for v in variants if v.error_type == "deletion"]
    assert [v.target_phone for v in deletions] == ["TH", "N", "K"]
    assert deletions[0].modified_phones == ["AE", "N", "K"]


def test_no_vowel_deletion():
    variants = generate_variants("at", ["AE", "T"], inventory())
    assert not any(v.error_type == "deletion" and v.target_phone == "AE" for v in variants)

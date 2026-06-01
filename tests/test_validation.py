from synthetic_ped.error_inventory import ErrorInventory
from synthetic_ped.metadata import ManifestRow
from synthetic_ped.validation import validate_row


def inv():
    return ErrorInventory(vowels=(), consonants=(("TH", "S"),), consonant_set=frozenset({"TH", "N"}))


def row(**overrides):
    data = dict(
        item_id="thank_0001_word",
        source_word="thank",
        context_type="word",
        text_prompt="thank",
        canonical_phones=["TH", "AE", "N", "K"],
        modified_phones=["S", "AE", "N", "K"],
        canonical_ipa="θ æ n k",
        modified_ipa="s æ n k",
        error_type="substitution",
        target_index=0,
        target_phone="TH",
        replacement_phone="S",
        phone_inventory="test",
        sbv2_phone_input=["s", "ae", "n", "k"],
        audio_path="audio/word/thank.wav",
    )
    data.update(overrides)
    return ManifestRow(**data)


def test_one_edit_distance_validation():
    result = validate_row(row(), inv())
    assert result.status == "unverified"


def test_rejects_non_inventory_substitution():
    result = validate_row(row(replacement_phone="F", modified_phones=["F", "AE", "N", "K"], sbv2_phone_input=["f", "ae", "n", "k"]), inv())
    assert result.status == "invalid"


def test_deletion_only_for_consonants():
    vowel_delete = row(error_type="deletion", target_index=1, target_phone="AE", replacement_phone=None, modified_phones=["TH", "N", "K"], sbv2_phone_input=["th", "n", "k"])
    result = validate_row(vowel_delete, inv())
    assert result.status == "invalid"

from synthetic_ped.metadata import ManifestRow


def test_metadata_serialization():
    row = ManifestRow(
        item_id="id",
        source_word="word",
        context_type="word",
        text_prompt="word",
        canonical_phones=["W", "ER", "D"],
        modified_phones=["W", "ER"],
        canonical_ipa=None,
        modified_ipa=None,
        error_type="deletion",
        target_index=2,
        target_phone="D",
        replacement_phone=None,
        phone_inventory="test",
        sbv2_phone_input=["w", "er"],
        audio_path="audio/word/id.wav",
    )
    cloned = ManifestRow.from_dict(row.to_dict())
    assert cloned == row

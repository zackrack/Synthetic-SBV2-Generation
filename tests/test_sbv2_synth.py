import json
from types import SimpleNamespace

from synthetic_ped.sbv2_synth import select_compatible_model


def make_model(tmp_path, name: str, version: str):
    model_dir = tmp_path / name
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"version": version}), encoding="utf-8"
    )
    model_file = model_dir / f"{name}.safetensors"
    model_file.write_text("", encoding="utf-8")
    return SimpleNamespace(name=name, files=[str(model_file)])


def test_select_compatible_model_skips_jp_extra_for_english(tmp_path):
    jp_extra = make_model(tmp_path, "jp_extra", "2.7.0-JP-Extra")
    english = make_model(tmp_path, "english", "2.7.0")

    selection = select_compatible_model([jp_extra, english], None, "EN")

    assert selection.model_info == english
    assert selection.notes == ""


def test_select_compatible_model_reports_when_only_jp_extra_for_english(tmp_path):
    jp_extra = make_model(tmp_path, "jp_extra", "2.7.0-JP-Extra")

    selection = select_compatible_model([jp_extra], None, "EN")

    assert selection.model_info is None
    assert "no SBV2 models compatible with language EN" in selection.notes
    assert "python initialize.py --only_infer" in selection.notes


def test_select_compatible_model_rejects_explicit_jp_extra_for_english(tmp_path):
    jp_extra = make_model(tmp_path, "jp_extra", "2.7.0-JP-Extra")

    selection = select_compatible_model([jp_extra], "jp_extra", "EN")

    assert selection.model_info is None
    assert "SBV2 model jp_extra is JP-Extra" in selection.notes

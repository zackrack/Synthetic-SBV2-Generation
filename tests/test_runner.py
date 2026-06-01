from pathlib import Path

from synthetic_ped.runner import run_generation


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_run_generation_returns_summary(tmp_path):
    cfg = tmp_path / "cfg"
    out = tmp_path / "out"
    write(cfg / "words.txt", "thank\n")
    write(cfg / "pron.dict", "THANK TH AE NG K\n")
    write(
        cfg / "inventory.yaml",
        "consonants:\n  - [TH, S]\nvowels: []\nconsonant_phones:\n  - TH\n  - NG\n  - K\n",
    )
    write(
        cfg / "mapping.yaml",
        "phone_inventory: test\nmapping:\n  TH: th\n  S: s\n  AE: ae\n  NG: ng\n  K: k\n",
    )
    write(
        cfg / "contexts.yaml",
        "word: '{word}'\nsentence: 'Say {word}.'\npassage: 'Say {word} now.'\n",
    )
    write(cfg / "gen.yaml", "language: EN\nstyle: Neutral\n")

    summary = run_generation(
        words_path=cfg / "words.txt",
        pronunciations_path=cfg / "pron.dict",
        error_inventory_path=cfg / "inventory.yaml",
        phone_mapping_path=cfg / "mapping.yaml",
        context_templates_path=cfg / "contexts.yaml",
        generation_config_path=cfg / "gen.yaml",
        output_dir=out,
        dry_run=True,
        max_items=3,
        make_review_manifest=True,
    )

    assert summary.status == "completed"
    assert summary.total_rows == 3
    assert summary.generated_rows == 3
    assert summary.failed_rows == 0
    assert summary.manifest_path.exists()
    assert summary.metadata_path.exists()
    assert summary.review_manifest_path is not None
    assert summary.review_manifest_path.exists()

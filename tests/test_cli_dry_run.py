from pathlib import Path

from synthetic_ped.cli import main
from synthetic_ped.metadata import read_jsonl


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_dry_run_behavior(tmp_path):
    cfg = tmp_path / "cfg"
    out = tmp_path / "out"
    write(cfg / "words.txt", "thank\n")
    write(cfg / "pron.dict", "THANK TH AE NG K\n")
    write(cfg / "inventory.yaml", "consonants:\n  - [TH, S]\nvowels: []\nconsonant_phones:\n  - TH\n  - NG\n  - K\n")
    write(cfg / "mapping.yaml", "phone_inventory: test\nmapping:\n  TH: th\n  S: s\n  AE: ae\n  NG: ng\n  K: k\n")
    write(cfg / "contexts.yaml", "word: '{word}'\nsentence: 'Say {word}.'\npassage: 'Say {word} now.'\n")
    write(cfg / "gen.yaml", "language: EN\nstyle: Neutral\n")

    rc = main([
        "--words", str(cfg / "words.txt"),
        "--pronunciations", str(cfg / "pron.dict"),
        "--error-inventory", str(cfg / "inventory.yaml"),
        "--phone-mapping", str(cfg / "mapping.yaml"),
        "--context-templates", str(cfg / "contexts.yaml"),
        "--generation-config", str(cfg / "gen.yaml"),
        "--output-dir", str(out),
        "--dry-run",
    ])
    assert rc == 0
    rows = read_jsonl(out / "manifest.jsonl")
    assert rows
    assert all(row.generation_status == "dry_run" for row in rows)
    assert not list((out / "audio").glob("**/*.wav"))

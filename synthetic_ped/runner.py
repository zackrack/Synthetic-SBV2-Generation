from __future__ import annotations

import io
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthetic_ped.contexts import load_context_templates
from synthetic_ped.error_inventory import ErrorInventory
from synthetic_ped.ipa import phones_to_ipa
from synthetic_ped.metadata import ManifestRow, append_jsonl, read_jsonl, write_csv, write_jsonl
from synthetic_ped.phone_mapping import PhoneMapper, PhoneMappingError
from synthetic_ped.pronunciations import PronunciationDictionary, read_words
from synthetic_ped.review import write_review_manifest
from synthetic_ped.sbv2_synth import synthesize_with_sbv2
from synthetic_ped.simple_yaml import load_config
from synthetic_ped.validation import validate_rows
from synthetic_ped.variants import generate_variants

LOGGER = logging.getLogger("synthetic_ped")


@dataclass(frozen=True)
class GenerationSummary:
    status: str
    total_rows: int
    generated_rows: int
    skipped_rows: int
    failed_rows: int
    manifest_path: Path
    metadata_path: Path
    review_manifest_path: Path | None = None
    rows: list[ManifestRow] = field(default_factory=list)
    log_output: str = ""


def run_generation(
    words_path: str | Path = Path("configs/gfta_words.txt"),
    pronunciations_path: str | Path = Path("configs/pronunciations.dict"),
    error_inventory_path: str | Path = Path("configs/error_inventory.yaml"),
    phone_mapping_path: str | Path = Path("configs/phone_mapping.yaml"),
    context_templates_path: str | Path = Path("configs/context_templates.yaml"),
    generation_config_path: str | Path = Path("configs/generation_config.yaml"),
    output_dir: str | Path = Path("synthetic_ped_data"),
    speaker_id: str | None = None,
    style: str | None = None,
    max_items: int | None = None,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    validate_only: bool = False,
    make_review_manifest: bool = False,
) -> GenerationSummary:
    """Run the synthetic PED pipeline without going through argparse.

    This callable is shared by the CLI and the Gradio app. It preserves the
    CLI's output layout while returning counts, paths, rows, and captured log
    output for UI display.
    """
    log_stream = io.StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

    try:
        return _run_generation_impl(
            words_path=Path(words_path),
            pronunciations_path=Path(pronunciations_path),
            error_inventory_path=Path(error_inventory_path),
            phone_mapping_path=Path(phone_mapping_path),
            context_templates_path=Path(context_templates_path),
            generation_config_path=Path(generation_config_path),
            output_dir=Path(output_dir),
            speaker_id=speaker_id or None,
            style=style or None,
            max_items=max_items,
            dry_run=dry_run,
            resume=resume,
            overwrite=overwrite,
            validate_only=validate_only,
            make_review_manifest=make_review_manifest,
            log_stream=log_stream,
        )
    except Exception:
        LOGGER.exception("Synthetic PED generation failed")
        raise
    finally:
        LOGGER.removeHandler(handler)
        handler.close()


def _run_generation_impl(
    *,
    words_path: Path,
    pronunciations_path: Path,
    error_inventory_path: Path,
    phone_mapping_path: Path,
    context_templates_path: Path,
    generation_config_path: Path,
    output_dir: Path,
    speaker_id: str | None,
    style: str | None,
    max_items: int | None,
    dry_run: bool,
    resume: bool,
    overwrite: bool,
    validate_only: bool,
    make_review_manifest: bool,
    log_stream: io.StringIO,
) -> GenerationSummary:
    _setup_output(output_dir)
    file_handler = logging.FileHandler(output_dir / "logs" / "generation.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOGGER.addHandler(file_handler)
    try:
        inventory = ErrorInventory.from_path(error_inventory_path)
        manifest_path = output_dir / "manifest.jsonl"
        metadata_path = output_dir / "metadata.csv"
        review_path = output_dir / "review_manifest.csv"

        if validate_only:
            rows = validate_rows(read_jsonl(manifest_path), inventory)
            write_jsonl(manifest_path, rows)
            write_csv(metadata_path, rows)
            written_review_path = _maybe_write_review_manifest(make_review_manifest, review_path, rows)
            return _summary(
                status="validated",
                rows=rows,
                manifest_path=manifest_path,
                metadata_path=metadata_path,
                review_manifest_path=written_review_path,
                log_output=log_stream.getvalue(),
            )

        _copy_input_configs(
            output_dir,
            [
                words_path,
                pronunciations_path,
                error_inventory_path,
                phone_mapping_path,
                context_templates_path,
                generation_config_path,
            ],
        )
        generation_config = _load_generation_config(generation_config_path)
        rows = build_rows(
            words_path=words_path,
            pronunciations_path=pronunciations_path,
            phone_mapping_path=phone_mapping_path,
            context_templates_path=context_templates_path,
            inventory=inventory,
            generation_config=generation_config,
            speaker_id=speaker_id,
            style=style,
        )
        if resume:
            existing = {row.item_id: row for row in read_jsonl(manifest_path)}
            rows = [existing.get(row.item_id, row) for row in rows]
        rows = validate_rows(rows, inventory)

        if manifest_path.exists():
            manifest_path.unlink()
        completed: list[ManifestRow] = []
        for row in rows:
            if max_items is not None and len(completed) >= max_items:
                break
            row = generate_or_skip(
                row=row,
                output_dir=output_dir,
                dry_run=dry_run,
                resume=resume,
                overwrite=overwrite,
            )
            append_jsonl(manifest_path, row)
            completed.append(row)
        write_csv(metadata_path, completed)
        written_review_path = _maybe_write_review_manifest(make_review_manifest, review_path, completed)
        return _summary(
            status="completed",
            rows=completed,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
            review_manifest_path=written_review_path,
            log_output=log_stream.getvalue(),
        )
    finally:
        LOGGER.removeHandler(file_handler)
        file_handler.close()


def build_rows(
    *,
    words_path: Path,
    pronunciations_path: Path,
    phone_mapping_path: Path,
    context_templates_path: Path,
    inventory: ErrorInventory,
    generation_config: dict[str, Any],
    speaker_id: str | None,
    style: str | None,
) -> list[ManifestRow]:
    words = read_words(words_path)
    pronunciations = PronunciationDictionary.from_path(pronunciations_path)
    mapper = PhoneMapper.from_path(phone_mapping_path)
    templates = load_context_templates(context_templates_path)
    rows: list[ManifestRow] = []
    for word in words:
        canonical = pronunciations.get(word)
        if canonical is None:
            LOGGER.warning("Skipping %s: missing pronunciation", word)
            continue
        variants = generate_variants(word, canonical, inventory)
        for variant_number, variant in enumerate(variants, start=1):
            try:
                sbv2_phones = mapper.map_sequence(variant.modified_phones)
                mapping_status = "pending"
                notes = variant.notes
            except PhoneMappingError as exc:
                sbv2_phones = []
                mapping_status = "failed"
                notes = str(exc)
                LOGGER.warning("Skipping synthesis for %s variant %d: %s", word, variant_number, exc)
            for template in templates:
                item_id = f"{_safe_id(word)}_{variant_number:04d}_{template.context_type}"
                audio_rel = Path("audio") / template.context_type / f"{item_id}.wav"
                rows.append(
                    ManifestRow(
                        item_id=item_id,
                        source_word=word,
                        context_type=template.context_type,
                        text_prompt=template.render(word),
                        canonical_phones=variant.canonical_phones,
                        modified_phones=variant.modified_phones,
                        canonical_ipa=phones_to_ipa(variant.canonical_phones),
                        modified_ipa=phones_to_ipa(variant.modified_phones),
                        error_type=variant.error_type,
                        target_index=variant.target_index,
                        target_phone=variant.target_phone,
                        replacement_phone=variant.replacement_phone,
                        phone_inventory=mapper.inventory_name,
                        sbv2_phone_input=sbv2_phones,
                        audio_path=str(audio_rel),
                        speaker_id=speaker_id,
                        model_id=(
                            str(generation_config["model_name"])
                            if generation_config.get("model_name") is not None
                            else None
                        ),
                        style=style or generation_config.get("style"),
                        generation_config=generation_config,
                        generation_status=mapping_status,
                        validation_status="unverified",
                        notes=notes,
                    )
                )
    return rows


def generate_or_skip(
    *,
    row: ManifestRow,
    output_dir: Path,
    dry_run: bool,
    resume: bool,
    overwrite: bool,
) -> ManifestRow:
    output_path = output_dir / row.audio_path
    if row.generation_status == "failed":
        return row
    if dry_run:
        row.generation_status = "dry_run"
        return row
    if resume and output_path.exists() and row.generation_status == "success":
        return row
    if output_path.exists() and not overwrite:
        row.generation_status = "skipped_existing"
        row.notes = "; ".join(
            part for part in [row.notes, "audio exists; pass --overwrite to regenerate"] if part
        )
        return row
    result = synthesize_with_sbv2(
        text=row.text_prompt,
        phones=row.sbv2_phone_input,
        output_path=output_path,
        speaker_id=row.speaker_id,
        style=row.style,
        generation_config=row.generation_config,
    )
    row.generation_status = result.status
    if result.notes:
        row.notes = "; ".join(part for part in [row.notes, result.notes] if part)
    return row


def _summary(
    *,
    status: str,
    rows: list[ManifestRow],
    manifest_path: Path,
    metadata_path: Path,
    review_manifest_path: Path | None,
    log_output: str,
) -> GenerationSummary:
    return GenerationSummary(
        status=status,
        total_rows=len(rows),
        generated_rows=sum(1 for row in rows if row.generation_status in {"success", "dry_run"}),
        skipped_rows=sum(1 for row in rows if row.generation_status.startswith("skipped")),
        failed_rows=sum(1 for row in rows if row.generation_status == "failed"),
        manifest_path=manifest_path,
        metadata_path=metadata_path,
        review_manifest_path=review_manifest_path,
        rows=rows,
        log_output=log_output,
    )


def _maybe_write_review_manifest(
    make_review_manifest: bool, review_path: Path, rows: list[ManifestRow]
) -> Path | None:
    if not make_review_manifest:
        return None
    write_review_manifest(review_path, rows)
    return review_path


def _copy_input_configs(output_dir: Path, config_paths: list[Path]) -> None:
    destination = output_dir / "configs"
    destination.mkdir(parents=True, exist_ok=True)
    for path in config_paths:
        if path.exists():
            shutil.copy2(path, destination / path.name)


def _load_generation_config(path: Path) -> dict[str, Any]:
    return load_config(path) if path.exists() else {}


def _setup_output(output_dir: Path) -> None:
    for subdir in ["audio/word", "audio/sentence", "audio/passage", "configs", "logs"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)


def _safe_id(word: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in word).strip("_")

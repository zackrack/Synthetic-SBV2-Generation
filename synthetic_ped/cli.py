from __future__ import annotations

import argparse
import logging
from pathlib import Path

from synthetic_ped.metadata import read_jsonl, write_csv, write_jsonl
from synthetic_ped.review import filter_accepted
from synthetic_ped.runner import run_generation


LOGGER = logging.getLogger("synthetic_ped")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate PEDBench-style phonetic-error speech with Style-BERT-VITS2."
    )
    parser.add_argument("--words", type=Path, default=Path("configs/gfta_words.txt"))
    parser.add_argument("--pronunciations", type=Path, default=Path("configs/pronunciations.dict"))
    parser.add_argument("--error-inventory", type=Path, default=Path("configs/error_inventory.yaml"))
    parser.add_argument("--phone-mapping", type=Path, default=Path("configs/phone_mapping.yaml"))
    parser.add_argument("--context-templates", type=Path, default=Path("configs/context_templates.yaml"))
    parser.add_argument("--generation-config", type=Path, default=Path("configs/generation_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic_ped_data"))
    parser.add_argument("--speaker-id", default=None)
    parser.add_argument("--style", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--make-review-manifest", action="store_true")
    parser.add_argument(
        "--filter-accepted",
        type=Path,
        default=None,
        help="Review CSV to filter accepted rows from an existing manifest.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.filter_accepted is not None:
        accepted = filter_accepted(
            args.filter_accepted, read_jsonl(args.output_dir / "manifest.jsonl")
        )
        write_jsonl(args.output_dir / "accepted_manifest.jsonl", accepted)
        write_csv(args.output_dir / "accepted_metadata.csv", accepted)
        return 0

    review_only = (
        args.make_review_manifest
        and not args.validate_only
        and not args.dry_run
        and not args.resume
        and not args.overwrite
        and args.max_items is None
    )

    summary = run_generation(
        words_path=args.words,
        pronunciations_path=args.pronunciations,
        error_inventory_path=args.error_inventory,
        phone_mapping_path=args.phone_mapping,
        context_templates_path=args.context_templates,
        generation_config_path=args.generation_config,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id,
        style=args.style,
        max_items=args.max_items,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        validate_only=args.validate_only or review_only,
        make_review_manifest=args.make_review_manifest,
    )
    LOGGER.info(
        "Synthetic PED run %s: total=%d generated=%d skipped=%d failed=%d manifest=%s metadata=%s",
        summary.status,
        summary.total_rows,
        summary.generated_rows,
        summary.skipped_rows,
        summary.failed_rows,
        summary.manifest_path,
        summary.metadata_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

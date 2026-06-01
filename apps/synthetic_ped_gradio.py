#!/usr/bin/env python
from __future__ import annotations

import html
from pathlib import Path
import sys
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import gradio as gr
except ImportError as exc:  # pragma: no cover - exercised manually when gradio is absent
    raise SystemExit(
        "Gradio is required for this app. Install it with: pip install gradio"
    ) from exc

from synthetic_ped.runner import GenerationSummary, run_generation


def run_from_ui(
    words_path: str,
    pronunciations_path: str,
    error_inventory_path: str,
    phone_mapping_path: str,
    context_templates_path: str,
    generation_config_path: str,
    output_dir: str,
    speaker_id: str,
    style: str,
    max_items: float | None,
    dry_run: bool,
    resume: bool,
    overwrite: bool,
    validate_only: bool,
    make_review_manifest: bool,
):
    try:
        max_items_int = None
        if max_items is not None and str(max_items).strip() != "":
            max_items_int = int(max_items)
            if max_items_int < 1:
                max_items_int = None
        summary = run_generation(
            words_path=words_path,
            pronunciations_path=pronunciations_path,
            error_inventory_path=error_inventory_path,
            phone_mapping_path=phone_mapping_path,
            context_templates_path=context_templates_path,
            generation_config_path=generation_config_path,
            output_dir=output_dir,
            speaker_id=speaker_id or None,
            style=style or None,
            max_items=max_items_int,
            dry_run=dry_run,
            resume=resume,
            overwrite=overwrite,
            validate_only=validate_only,
            make_review_manifest=make_review_manifest,
        )
        return _summary_outputs(summary)
    except Exception as exc:  # UI boundary: render failures instead of crashing the app
        return (
            f"failed: {exc}",
            0,
            0,
            1,
            "",
            "",
            "",
            f"{type(exc).__name__}: {exc}",
            "<p>No preview available because generation failed.</p>",
        )


def _summary_outputs(summary: GenerationSummary):
    return (
        summary.status,
        summary.generated_rows,
        summary.skipped_rows,
        summary.failed_rows,
        str(summary.manifest_path),
        str(summary.metadata_path),
        str(summary.review_manifest_path or ""),
        summary.log_output,
        _preview_html(summary),
    )


def _preview_html(summary: GenerationSummary, limit: int = 8) -> str:
    playable_rows = []
    output_dir = summary.manifest_path.parent
    for row in summary.rows:
        audio_path = output_dir / row.audio_path
        if row.generation_status == "success" and audio_path.exists():
            playable_rows.append((row, audio_path))
        if len(playable_rows) >= limit:
            break
    if not playable_rows:
        return "<p>No generated audio files are available to preview.</p>"

    header = """
<table>
  <thead>
    <tr>
      <th>item_id</th><th>source_word</th><th>context_type</th><th>error_type</th>
      <th>target_phone</th><th>replacement_phone</th><th>text_prompt</th><th>audio</th>
    </tr>
  </thead>
  <tbody>
"""
    rows_html = []
    for row, audio_path in playable_rows:
        file_url = f"/file={quote(str(audio_path.resolve()))}"
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(row.item_id)}</td>"
            f"<td>{html.escape(row.source_word)}</td>"
            f"<td>{html.escape(row.context_type)}</td>"
            f"<td>{html.escape(row.error_type)}</td>"
            f"<td>{html.escape(row.target_phone)}</td>"
            f"<td>{html.escape(row.replacement_phone or '')}</td>"
            f"<td>{html.escape(row.text_prompt)}</td>"
            f"<td><audio controls src=\"{file_url}\"></audio></td>"
            "</tr>"
        )
    return header + "\n".join(rows_html) + "\n  </tbody>\n</table>"


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Synthetic PED Generator") as app:
        gr.Markdown(
            "# Synthetic PED Generator\n"
            "Configure and run the Style-BERT-VITS2 synthetic phonetic-error pipeline locally. "
            "The text prompt remains normal orthography; the mapped phone input is the source of truth."
        )
        with gr.Row():
            with gr.Column():
                words_path = gr.Textbox(label="Words file path", value="configs/gfta_words.txt")
                pronunciations_path = gr.Textbox(
                    label="Pronunciations dict path", value="configs/pronunciations.dict"
                )
                error_inventory_path = gr.Textbox(
                    label="Error inventory YAML path", value="configs/error_inventory.yaml"
                )
                phone_mapping_path = gr.Textbox(
                    label="Phone mapping YAML path", value="configs/phone_mapping.yaml"
                )
                context_templates_path = gr.Textbox(
                    label="Context templates YAML path", value="configs/context_templates.yaml"
                )
                generation_config_path = gr.Textbox(
                    label="Generation config YAML path", value="configs/generation_config.yaml"
                )
                output_dir = gr.Textbox(label="Output directory", value="synthetic_ped_data")
            with gr.Column():
                speaker_id = gr.Textbox(label="Speaker ID", value="0")
                style = gr.Textbox(label="Style", value="Neutral")
                max_items = gr.Number(label="Max items (blank/0 for no limit)", value=0, precision=0)
                dry_run = gr.Checkbox(label="Dry run", value=True)
                resume = gr.Checkbox(label="Resume", value=False)
                overwrite = gr.Checkbox(label="Overwrite existing audio", value=False)
                validate_only = gr.Checkbox(label="Validate only", value=False)
                make_review_manifest = gr.Checkbox(label="Make review manifest", value=False)
        run_button = gr.Button("Generate Synthetic PED Data", variant="primary")

        with gr.Row():
            status = gr.Textbox(label="Status")
            generated_rows = gr.Number(label="Number of generated rows", precision=0)
            skipped_rows = gr.Number(label="Number of skipped rows", precision=0)
            failed_rows = gr.Number(label="Number of failed rows", precision=0)
        manifest_path = gr.Textbox(label="Output manifest path")
        metadata_path = gr.Textbox(label="Metadata CSV path")
        review_manifest_path = gr.Textbox(label="Review manifest path")
        log_output = gr.Textbox(label="Log output", lines=12)
        preview = gr.HTML(label="Generated audio preview")

        run_button.click(
            run_from_ui,
            inputs=[
                words_path,
                pronunciations_path,
                error_inventory_path,
                phone_mapping_path,
                context_templates_path,
                generation_config_path,
                output_dir,
                speaker_id,
                style,
                max_items,
                dry_run,
                resume,
                overwrite,
                validate_only,
                make_review_manifest,
            ],
            outputs=[
                status,
                generated_rows,
                skipped_rows,
                failed_rows,
                manifest_path,
                metadata_path,
                review_manifest_path,
                log_output,
                preview,
            ],
        )
    return app


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", share=False)

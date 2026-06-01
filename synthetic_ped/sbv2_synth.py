from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    status: str
    audio_path: Path
    sample_rate: int | None = None
    notes: str = ""


@dataclass(frozen=True)
class ModelSelection:
    model_info: Any | None = None
    notes: str = ""


SYNTHETIC_PED_MODEL_SETUP_NOTE = (
    "For the default English synthetic PED config, install an English/non-JP-Extra "
    "SBV2 model by running `python initialize.py --only_infer`."
)


def synthesize_with_sbv2(
    text: str,
    phones: list[str],
    output_path: Path,
    speaker_id: str | None,
    style: str | None,
    generation_config: dict[str, Any],
) -> GenerationResult:
    """Synthesize one utterance with Style-BERT-VITS2 direct phone control.

    This is the only repository-specific integration point. The pipeline passes
    normal orthographic ``text`` because SBV2 requires text, and passes
    ``phones`` as ``given_phone`` because the phones are the intended realized
    pronunciation. ``given_tone`` defaults to a same-length zero vector unless
    supplied in ``generation_config``.
    """
    from style_bert_vits2.constants import Languages
    from style_bert_vits2.tts_model import TTSModelHolder
    from style_bert_vits2.utils import torch_device_to_onnx_providers

    model_assets_dir = Path(generation_config.get("model_assets_dir", "model_assets"))
    device = str(generation_config.get("device", "cpu"))
    model_name = generation_config.get("model_name")
    model_file = generation_config.get("model_file")
    language_name = str(generation_config.get("language", "EN"))
    language = getattr(Languages, language_name)
    line_split = bool(generation_config.get("line_split", False))
    speaker_int = int(
        speaker_id
        if speaker_id is not None and str(speaker_id).isdigit()
        else generation_config.get("speaker_id", 0)
    )
    style_name = style or str(generation_config.get("style", "Neutral"))
    given_tone = generation_config.get("given_tone")
    if given_tone is None:
        given_tone = [0] * len(phones)
    if len(given_tone) != len(phones):
        raise ValueError("given_tone length must match phones length")

    holder = TTSModelHolder(
        model_assets_dir, device, torch_device_to_onnx_providers(device)
    )
    if not holder.models_info:
        return GenerationResult(
            "failed", output_path, notes=f"no SBV2 models found in {model_assets_dir}"
        )
    if model_name is None:
        model_info = holder.models_info[0]
        model_name = model_info.name
    else:
        matches = [info for info in holder.models_info if info.name == model_name]
        if not matches:
            return GenerationResult(
                "failed", output_path, notes=f"SBV2 model not found: {model_name}"
            )
        model_info = matches[0]
    if model_file is None:
        files = [
            f
            for f in model_info.files
            if (f.endswith(".safetensors") or f.endswith(".onnx"))
            and not f.startswith(".")
        ]
        if not files:
            return GenerationResult(
                "failed", output_path, notes=f"no model file found for {model_name}"
            )
        model_file = files[0]

    model = holder.get_model(str(model_name), str(model_file))
    try:
        model.load()
        sample_rate, audio_data = model.infer(
            text,
            language=language,
            speaker_id=speaker_int,
            style=style_name,
            given_phone=phones,
            given_tone=list(given_tone),
            line_split=line_split,
            sdp_ratio=float(generation_config.get("sdp_ratio", 0.4)),
            noise=float(generation_config.get("noise", 0.667)),
            noise_w=float(generation_config.get("noise_w", 0.8)),
            length=float(generation_config.get("length", 1.0)),
            style_weight=float(generation_config.get("style_weight", 2.0)),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav(output_path, sample_rate, audio_data)
        return GenerationResult("success", output_path, sample_rate=sample_rate)
    # Capture per-row failures at the synthesis boundary for long runs.
    except Exception as exc:
        return GenerationResult(
            "failed", output_path, notes=f"SBV2 synthesis failed: {exc}"
        )


def _write_wav(path: Path, sample_rate: int, audio_data: Any) -> None:
    try:
        import numpy as np

        arr = np.asarray(audio_data)
        if arr.dtype != np.int16:
            if arr.dtype.kind == "f":
                arr = np.clip(arr, -1.0, 1.0)
                arr = (arr * 32767).astype(np.int16)
            else:
                arr = arr.astype(np.int16)
        pcm = arr.tobytes()
    except ImportError:
        pcm = bytes(audio_data)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

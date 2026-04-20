"""IndicConformer Tamil ASR wrapper.

Lazy-loads the model on first use. The model weights (~500MB) are
downloaded from HuggingFace into the user's HF cache on first call.
CPU inference works; GPU is used automatically if available.
"""

from __future__ import annotations

import glob
import io
import os
import time
from typing import Optional

import numpy as np
import soundfile as sf
import torch

# Compat shim: NeMo 2.7.x calls accelerate.utils.environment.check_fp8_capability,
# but accelerate 1.13.x exposes it as check_cuda_fp8_capability. Alias them so
# model-load doesn't explode on CPU-only machines where FP8 isn't even relevant.
try:
    from accelerate.utils import environment as _accel_env
    if not hasattr(_accel_env, "check_fp8_capability"):
        if hasattr(_accel_env, "check_cuda_fp8_capability"):
            _accel_env.check_fp8_capability = _accel_env.check_cuda_fp8_capability
        else:
            _accel_env.check_fp8_capability = lambda *a, **kw: False
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

from .romanize import romanize

_MODEL_ID = os.environ.get(
    "ASR_MODEL_ID",
    "ai4bharat/indicconformer_stt_ta_hybrid_rnnt_large",
)
_HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
_TARGET_SR = 16_000

_model = None  # module-level cache


def _load_model():
    global _model
    if _model is not None:
        return _model

    start = time.perf_counter()
    from huggingface_hub import snapshot_download
    from nemo.collections.asr.models import ASRModel

    try:
        local_dir = snapshot_download(repo_id=_MODEL_ID, token=_HF_TOKEN)
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "gated" in msg.lower() or "restricted" in msg.lower():
            raise RuntimeError(
                f"HuggingFace access denied for {_MODEL_ID}. This model is gated — "
                "visit https://huggingface.co/" + _MODEL_ID +
                " and click 'Agree and access repository', then put your HF "
                "access token (from https://huggingface.co/settings/tokens) "
                "into .env as HF_TOKEN=hf_..."
            ) from exc
        raise
    nemo_files = glob.glob(os.path.join(local_dir, "*.nemo"))
    if not nemo_files:
        raise RuntimeError(
            f"No .nemo file found under {local_dir} for {_MODEL_ID}"
        )
    model = ASRModel.restore_from(restore_path=nemo_files[0], map_location="cpu")
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    # Prefer RNNT decoding for accuracy on hybrid models.
    try:
        model.change_decoding_strategy(decoder_type="rnnt")
    except Exception:
        pass

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    print(f"✅ IndicConformer Tamil loaded ({elapsed_ms}ms)")
    _model = model
    return _model


def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes into mono float32 + sample rate."""
    with io.BytesIO(audio_bytes) as buf:
        data, sr = sf.read(buf, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def _resample(audio: np.ndarray, sr: int, target_sr: int = _TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio
    import torchaudio.functional as AF

    tensor = torch.from_numpy(audio).unsqueeze(0)
    resampled = AF.resample(tensor, sr, target_sr)
    return resampled.squeeze(0).numpy().astype(np.float32)


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe a chunk of audio (any soundfile-readable container)."""
    if not audio_bytes:
        return ""
    audio, sr = _decode_audio(audio_bytes)
    audio = _resample(audio, sr, _TARGET_SR)
    if audio.size < _TARGET_SR // 10:  # < 100ms of audio — skip
        return ""

    model = _load_model()
    with torch.inference_mode():
        hyps = model.transcribe([audio], batch_size=1)
    return _extract_text(hyps)


def transcribe_array(audio: np.ndarray, sample_rate: int) -> str:
    """Transcribe a raw float32 mono numpy array."""
    if audio is None or audio.size == 0:
        return ""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    audio = _resample(audio, sample_rate, _TARGET_SR)
    if audio.size < _TARGET_SR // 10:
        return ""
    model = _load_model()
    with torch.inference_mode():
        hyps = model.transcribe([audio], batch_size=1)
    return _extract_text(hyps)


def _extract_text(hyps) -> str:
    """NeMo's transcribe returns varied shapes across versions.

    Tamil IndicConformer outputs native Tamil script; romanize it so the
    nurse sees readable English letters in the live transcript.
    """
    if not hyps:
        return ""
    first = hyps[0]
    # NeMo >=1.22 returns list[Hypothesis] with .text
    if isinstance(first, list):
        first = first[0] if first else ""
    text = getattr(first, "text", first)
    text = (text or "").strip()
    return romanize(text)


def warmup() -> None:
    """Preload the model so the first real request is fast."""
    _load_model()

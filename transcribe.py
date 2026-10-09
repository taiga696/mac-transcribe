#!/usr/bin/env python3
"""
transcribe.py — Terminal-based audio transcription with speaker diarization.

Uses faster-whisper for transcription + pyannote.audio for speaker diarization.
Runs entirely on the local machine; no cloud upload.

Usage:
    transcribe audio.m4a
    transcribe audio.m4a --language ja --model large-v3
    transcribe audio.m4a --output transcript.md
    transcribe audio.m4a --num-speakers 2
    transcribe audio.m4a --no-diarize        # skip diarization (no HF token needed)

Requirements:
    - HuggingFace token (set HF_TOKEN env var or run `hf auth login`)
      Required for diarization only. Plain transcription works without it.
    - pyannote/speaker-diarization-3.1 gated access granted at
      https://huggingface.co/pyannote/speaker-diarization-3.1
    - pyannote/segmentation-3.0 gated access granted at
      https://huggingface.co/pyannote/segmentation-3.0
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Tuple


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS for display."""
    return str(timedelta(seconds=int(seconds)))


def _resolve_hf_token() -> str | None:
    """Find a HuggingFace token from env vars or huggingface_hub's storage.

    Tries in order:
      1. HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars
      2. huggingface_hub.get_token() (recommended in modern versions)
      3. HfApi().token (older versions)
      4. Direct read of ~/.cache/huggingface/token (universal fallback)
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    # 1. get_token() — works in huggingface_hub >= 0.16
    try:
        from huggingface_hub import get_token
        t = get_token()
        if t:
            return t
    except Exception:
        pass
    # 2. HfApi().token — fallback
    try:
        from huggingface_hub import HfApi
        t = HfApi().token
        if t:
            return t
    except Exception:
        pass
    # 3. Direct file read — universal fallback (hf auth login saves here)
    try:
        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(token_path):
            with open(token_path) as f:
                t = f.read().strip()
                if t:
                    return t
    except Exception:
        pass
    return None


def _pick_device(requested: str) -> Tuple[str, str]:
    """Pick (device, compute_type) for faster-whisper based on availability."""
    if requested != "auto":
        return requested, ("float16" if requested == "cuda" else "int8")
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
        # faster-whisper has no MPS backend; CPU with int8 is the safe default
        # on Apple Silicon. Use --engine mlx to run on the GPU instead.
        return "cpu", "int8"
    except Exception:
        return "cpu", "int8"


def _pick_engine(requested: str) -> str:
    """Pick transcription engine: 'mlx' (Apple Silicon GPU) or 'faster-whisper'."""
    if requested != "auto":
        return requested
    try:
        import mlx_whisper  # noqa: F401
        return "mlx"
    except Exception:
        return "faster-whisper"


# ----------------------------------------------------------------------------
# Transcription (faster-whisper)
# ----------------------------------------------------------------------------

def transcribe_audio(
    audio_path: Path,
    language: str,
    model_size: str,
    device: str,
    compute_type: str,
) -> Tuple[list, object]:
    """Run Whisper transcription, return (segments, info)."""
    from faster_whisper import WhisperModel

    print(f"[1/3] Loading Whisper model '{model_size}' on {device} ({compute_type})…",
          file=sys.stderr)
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    print(f"      loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    lang_display = language or "auto"
    print(f"[2/3] Transcribing {audio_path.name} (language={lang_display})…",
          file=sys.stderr)
    t0 = time.time()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        beam_size=5,
    )
    # faster-whisper returns a generator — materialise so we can iterate twice
    segments = list(segments)
    print(f"      transcribed {len(segments)} segments in {time.time() - t0:.1f}s "
          f"(detected: {info.language} @ {info.language_probability:.2f})",
          file=sys.stderr)
    return segments, info


# ----------------------------------------------------------------------------
# Transcription (mlx-whisper — Apple Silicon GPU)
# ----------------------------------------------------------------------------

class _MlxWord:
    """Adapter giving mlx-whisper word dicts a faster-whisper-like interface."""
    __slots__ = ("start", "end", "word")

    def __init__(self, d: dict):
        self.start = float(d.get("start", 0.0))
        self.end = float(d.get("end", 0.0))
        self.word = d.get("word", "")


# mlx-whisper returns token-level timestamps, so Japanese text arrives as
# per-character fragments. Merge tokens into phrase-level units (split on
# punctuation) or word-level units (split on space-prefixed tokens) so that
# speaker alignment and Markdown output stay readable.
_PHRASE_END_CHARS = "、。,.．!！?？"

def _merge_mlx_words(word_dicts: list) -> list:
    merged: list[dict] = []
    buf: dict | None = None
    for w in word_dicts:
        t = w.get("word", "")
        if not t.strip():
            continue
        starts_new = t.startswith((" ", "　"))
        if buf is None or starts_new:
            if buf is not None:
                merged.append(buf)
            buf = {
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "word": t,
            }
        else:
            buf["end"] = float(w.get("end", buf["end"]))
            buf["word"] += t
        if t.rstrip().endswith(tuple(_PHRASE_END_CHARS)):
            merged.append(buf)
            buf = None
    if buf is not None:
        merged.append(buf)
    return merged


class _MlxSegment:
    """Adapter giving mlx-whisper segment dicts a faster-whisper-like interface."""
    __slots__ = ("start", "end", "text", "words")

    def __init__(self, d: dict):
        self.start = float(d.get("start", 0.0))
        self.end = float(d.get("end", 0.0))
        self.text = d.get("text", "")
        self.words = [_MlxWord(w) for w in _merge_mlx_words(d.get("words") or [])]


class _MlxInfo:
    """Adapter for the language/probability metadata faster-whisper exposes."""
    __slots__ = ("language", "language_probability")

    def __init__(self, language: str, probability: float = 1.0):
        self.language = language
        self.language_probability = probability


def _load_audio_16k(audio_path: Path):
    """Decode audio to 16 kHz mono float32 numpy array via PyAV.

    mlx-whisper's built-in loader shells out to ffmpeg, which is not
    installed on this machine — so we decode with PyAV (already used by
    faster-whisper) and pass the waveform directly.
    """
    import av
    import numpy as np

    container = av.open(str(audio_path))
    resampler = av.audio.resampler.AudioResampler(
        format="fltp", layout="mono", rate=16000,
    )
    frames = []
    try:
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                arr = r.to_ndarray()  # (1, samples) float32
                frames.append(arr.reshape(-1))
        # flush resampler
        for r in resampler.resample(None):
            frames.append(r.to_ndarray().reshape(-1))
    finally:
        container.close()
    if not frames:
        raise RuntimeError(f"音声をデコードできませんでした: {audio_path}")
    return np.concatenate(frames).astype(np.float32)


def transcribe_audio_mlx(
    audio_path: Path,
    language: str,
    model_size: str,
) -> Tuple[list, object]:
    """Run mlx-whisper on the Apple Silicon GPU, return (segments, info)."""
    import mlx_whisper

    # Allow passing a full HF repo id; otherwise map size -> mlx-community repo
    repo = (model_size if "/" in model_size
            else f"mlx-community/whisper-{model_size}-mlx")

    lang_display = language or "auto"
    print(f"[1/2] Transcribing {audio_path.name} with mlx-whisper '{repo}' "
          f"on GPU (language={lang_display})…", file=sys.stderr)

    t0 = time.time()
    audio = _load_audio_16k(audio_path)
    duration = len(audio) / 16000.0
    print(f"      decoded {duration / 60:.1f} min of audio in "
          f"{time.time() - t0:.1f}s", file=sys.stderr)

    t0 = time.time()
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=True,
        verbose=False,
    )
    segments = [_MlxSegment(s) for s in result.get("segments", [])]
    info = _MlxInfo(result.get("language") or (language or "unknown"))
    print(f"      transcribed {len(segments)} segments in {time.time() - t0:.1f}s "
          f"(language: {info.language})",
          file=sys.stderr)
    return segments, info


# ----------------------------------------------------------------------------
# Diarization (pyannote.audio)
# ----------------------------------------------------------------------------

def _load_audio_with_pyav(audio_path: Path):
    """Load audio file via PyAV and return (waveform_tensor, sample_rate).

    Used to feed pyannote.audio with a numpy/tensor dict instead of a file
    path, so that we can avoid torchcodec (which requires system ffmpeg).

    Returns:
        (torch.Tensor of shape (1, num_samples) float32, int sample_rate)
        or (None, None) on failure.
    """
    try:
        import av
        import numpy as np
        import torch
    except ImportError:
        return None, None
    try:
        container = av.open(str(audio_path))
    except Exception as e:
        print(f"      audio open failed: {e}", file=sys.stderr)
        return None, None
    stream = container.streams.audio[0]
    sample_rate = stream.rate
    frames = []
    try:
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()  # shape: (channels, samples)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)  # to mono
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype == np.int32:
                arr = arr.astype(np.float32) / 2147483648.0
            elif arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            frames.append(arr)
    finally:
        container.close()
    if not frames:
        return None, None
    waveform = np.concatenate(frames)
    tensor = torch.from_numpy(waveform).unsqueeze(0)  # (1, samples)
    return tensor, sample_rate


def diarize_audio(
    audio_path: Path,
    hf_token: str,
    num_speakers: int | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> object:
    """Run pyannote speaker diarization, return the Annotation object."""
    import torch
    from pyannote.audio import Pipeline

    print("[3/3] Loading pyannote diarization pipeline…", file=sys.stderr)
    t0 = time.time()
    # pyannote.audio 4.x renamed use_auth_token -> token
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        # older pyannote (< 4.0) fallback
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
    # pyannote supports MPS on Apple Silicon — faster than CPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            pipeline.to(torch.device("mps"))
            print("      using MPS device", file=sys.stderr)
        except Exception:
            pass  # fall back to CPU silently
    print(f"      pipeline loaded in {time.time() - t0:.1f}s", file=sys.stderr)

    print(f"      diarizing {audio_path.name}…", file=sys.stderr)
    t0 = time.time()
    # Avoid torchcodec (requires system ffmpeg) — feed waveform via PyAV
    waveform, sample_rate = _load_audio_with_pyav(audio_path)
    if waveform is None:
        # Fallback: pass path directly (will need torchcodec / ffmpeg)
        audio_input = str(audio_path)
    else:
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}
    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers
    diarization = pipeline(audio_input, **kwargs)
    # pyannote.audio 4.x returns a DiarizeOutput NamedTuple whose
    # .speaker_diarization field holds the Annotation. Older versions
    # returned the Annotation directly. Unwrap for compatibility.
    if hasattr(diarization, "speaker_diarization") and not hasattr(diarization, "itertracks"):
        diarization = diarization.speaker_diarization
    n_turns = len(list(diarization.itertracks()))
    n_speakers = len(diarization.labels())
    print(f"      {n_turns} turns, {n_speakers} speakers in {time.time() - t0:.1f}s",
          file=sys.stderr)
    return diarization


# ----------------------------------------------------------------------------
# Alignment
# ----------------------------------------------------------------------------

def align_words_to_speakers(
    segments: list,
    diarization,
) -> List[Tuple[str, float, float, str]]:
    """Return [(speaker, start, end, word), …] aligned by best overlap."""
    # Flatten whisper output into per-word tuples
    words: List[Tuple[float, float, str]] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                txt = w.word.strip()
                if txt:
                    words.append((w.start, w.end, txt))
        else:
            # No word-level timestamps — use segment boundaries, attach full text
            words.append((seg.start, seg.end, seg.text.strip()))

    if not words:
        return []

    # Pre-extract diarization turns once for speed
    turns = [(t.start, t.end, spk) for t, _, spk in diarization.itertracks(yield_label=True)]
    turns.sort()

    aligned: List[Tuple[str, float, float, str]] = []
    for w_start, w_end, w_text in words:
        best_spk: str | None = None
        best_overlap = 0.0
        for t_start, t_end, spk in turns:
            # Early termination: turns are sorted, so once we pass the word we can stop
            if t_start > w_end:
                break
            overlap = min(w_end, t_end) - max(w_start, t_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_spk = spk
        if best_spk is None:
            best_spk = "UNKNOWN"
        aligned.append((best_spk, w_start, w_end, w_text))
    return aligned


def group_by_speaker(
    aligned: List[Tuple[str, float, float, str]],
    max_gap_sec: float = 1.5,
) -> List[Tuple[str, float, float, str]]:
    """Merge consecutive same-speaker words into utterances.

    A new utterance starts when speaker changes OR gap exceeds max_gap_sec.
    """
    if not aligned:
        return []
    merged: List[Tuple[str, float, float, str]] = []
    cur_spk, cur_start, cur_end, cur_words = aligned[0][0], aligned[0][1], aligned[0][2], [aligned[0][3]]
    for spk, start, end, word in aligned[1:]:
        if spk == cur_spk and (start - cur_end) <= max_gap_sec:
            cur_end = end
            cur_words.append(word)
        else:
            merged.append((cur_spk, cur_start, cur_end, " ".join(cur_words)))
            cur_spk, cur_start, cur_end, cur_words = spk, start, end, [word]
    merged.append((cur_spk, cur_start, cur_end, " ".join(cur_words)))
    return merged


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------

def write_markdown(
    path: Path,
    audio_name: str,
    info,
    utterances: List[Tuple[str, float, float, str]] | None,
    segments: list | None,
    with_diarization: bool,
) -> None:
    """Write the transcript to a Markdown file."""
    lines: List[str] = []
    lines.append(f"# 文字起こし: {audio_name}")
    lines.append("")
    lines.append(f"- 生成日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 言語: {info.language} (信頼度: {info.language_probability:.2f})")
    lines.append(f"- デバイアリゼーション: {'あり' if with_diarization else 'なし'}")
    lines.append("")
    lines.append("---")
    lines.append("")

    if with_diarization and utterances:
        for spk, start, end, text in utterances:
            ts = _fmt_ts(start)
            lines.append(f"**[{ts}] {spk}**: {text}")
            lines.append("")
    else:
        # Fallback: segment-level timestamps only
        for seg in (segments or []):
            ts = _fmt_ts(seg.start)
            lines.append(f"**[{ts}]**: {seg.text.strip()}")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(
    path: Path,
    audio_name: str,
    info,
    utterances: List[Tuple[str, float, float, str]] | None,
    segments: list | None,
    with_diarization: bool,
) -> None:
    """Write the transcript as JSON for downstream processing."""
    data = {
        "audio": audio_name,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": info.language,
        "language_probability": round(float(info.language_probability), 3),
        "diarization": with_diarization,
        "utterances": [],
    }
    if with_diarization and utterances:
        for spk, start, end, text in utterances:
            data["utterances"].append({
                "speaker": spk,
                "start": round(float(start), 3),
                "end": round(float(end), 3),
                "text": text,
            })
    else:
        for seg in (segments or []):
            data["utterances"].append({
                "start": round(float(seg.start), 3),
                "end": round(float(seg.end), 3),
                "text": seg.text.strip(),
            })
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="音声ファイルをテキスト化（話者分離オプション付き）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  transcribe meeting.m4a\n"
            "  transcribe meeting.m4a --num-speakers 2\n"
            "  transcribe meeting.m4a --no-diarize          # HF トークン不要\n"
            "  transcribe meeting.m4a -o notes.md --format md\n"
        ),
    )
    parser.add_argument("audio", help="音声ファイルのパス")
    parser.add_argument("-l", "--language", default="auto",
                        help="言語コード (default: auto で自動検出)。ja/en/zh 等")
    parser.add_argument("-m", "--model", default="large-v3",
                        help="Whisper モデルサイズ (default: large-v3). "
                             "tiny/base/small/medium/large-v3/large-v2 等")
    parser.add_argument("-o", "--output",
                        help="出力ファイルパス（未指定時は入力ファイルと同階層）")
    parser.add_argument("-f", "--format", choices=["md", "json"], default="md",
                        help="出力フォーマット (default: md)")
    parser.add_argument("--num-speakers", type=int,
                        help="話者数を正確に指定")
    parser.add_argument("--min-speakers", type=int, help="最小話者数")
    parser.add_argument("--max-speakers", type=int, help="最大話者数")
    parser.add_argument("--no-diarize", action="store_true",
                        help="話者分離をスキップ（HF トークン不要で動作）")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="推論デバイス (default: auto) — faster-whisper エンジン用")
    parser.add_argument("--engine", default="auto",
                        choices=["auto", "mlx", "faster-whisper"],
                        help="文字起こしエンジン (default: auto = mlx-whisper が使えればGPU、"
                             "なければ faster-whisper CPU)")
    parser.add_argument("--compute-type", default="int8",
                        help="faster-whisper 計算精度 (default: int8) — faster-whisper エンジン用")
    parser.add_argument("--max-gap", type=float, default=1.5,
                        help="同一話者の連続発話をマージする際の最大隙間秒 (default: 1.5)")
    args = parser.parse_args()

    # --- Resolve paths ---
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"ERROR: 音声ファイルが見つかりません: {audio_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        suffix = "md" if args.format == "md" else "json"
        output_path = audio_path.with_suffix(f".{suffix}")

    # --- Decide on diarization ---
    hf_token = None
    do_diarize = not args.no_diarize
    if do_diarize:
        hf_token = _resolve_hf_token()
        if not hf_token:
            print(
                "WARNING: HF_TOKEN が未設定のため話者分離をスキップします。\n"
                "         話者分離が必要な場合は以下を実行してください:\n"
                "           1. https://huggingface.co/pyannote/speaker-diarization-3.1 で利用規約に同意\n"
                "           2. https://huggingface.co/pyannote/segmentation-3.0 で利用規約に同意\n"
                "           3. https://huggingface.co/settings/tokens でアクセストークンを生成\n"
                "           4. export HF_TOKEN=hf_xxxxxxxx  (または hf auth login)\n"
                "         今回はセグメントのみの文字起こしを続行します。\n",
                file=sys.stderr,
            )
            do_diarize = False

    # --- Transcribe ---
    engine = _pick_engine(args.engine)
    try:
        # both engines expect None for auto-detection (not "auto" string)
        lang_for_whisper = None if args.language == "auto" else args.language
        if engine == "mlx":
            segments, info = transcribe_audio_mlx(
                audio_path, lang_for_whisper, args.model,
            )
        else:
            device, compute_type = _pick_device(args.device)
            if args.compute_type != "int8":
                compute_type = args.compute_type  # explicit override
            segments, info = transcribe_audio(
                audio_path, lang_for_whisper, args.model, device, compute_type,
            )
    except Exception as e:
        print(f"ERROR: Whisper 推論に失敗 ({engine}): {e}", file=sys.stderr)
        return 2

    # --- Diarize + align (optional) ---
    utterances: List[Tuple[str, float, float, str]] | None = None
    if do_diarize:
        try:
            diarization = diarize_audio(
                audio_path, hf_token,
                args.num_speakers, args.min_speakers, args.max_speakers,
            )
            aligned = align_words_to_speakers(segments, diarization)
            utterances = group_by_speaker(aligned, max_gap_sec=args.max_gap)
        except Exception as e:
            print(f"WARNING: 話者分離に失敗しました ({e})。セグメントのみで続行します。",
                  file=sys.stderr)
            do_diarize = False

    # --- Write output ---
    if args.format == "md":
        write_markdown(output_path, audio_path.name, info, utterances,
                       segments if not do_diarize else None, do_diarize)
    else:
        write_json(output_path, audio_path.name, info, utterances,
                   segments if not do_diarize else None, do_diarize)

    print(f"\n✓ 完了: {output_path}", file=sys.stderr)
    if not do_diarize:
        print("  （話者分離なし — HF_TOKEN 設定で有効化）", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

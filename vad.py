#!/usr/bin/env python3
"""
VAD: Voice Activity Detection (Silero VAD)
==========================================
Preprocesses Cantonese interview audio using Silero VAD to detect speech
segments.

Produces:
  1. A JSON segmentation report per audio file (start, end, duration, speech ratio)
  2. Individual trimmed .wav clips for each detected speech segment

Usage:
    # Auto-scan audio/ directory
    python vad.py

    # Single file
    python vad.py --input audio/interview_01.mp3

    # Custom threshold and padding
    python vad.py --threshold 0.6 --speech-pad-ms 50

    # JSON report only (no audio clips)
    python vad.py --no-clips

    # Save current parameters to config file for reuse
    python vad.py --threshold 0.3 --speech-pad-ms 200 --save-config

    # Show effective parameters and their sources (CLI / config / default)
    python vad.py --show-config

Output:
    vad/
    ├── <stem>/
    │   ├── <stem>_vad.json          Segmentation report
    │   ├── <stem>_seg_001.wav       Speech segment clip 1
    │   ├── <stem>_seg_002.wav       Speech segment clip 2
    │   └── ...
    └── ...

Setup:
    pip install torch torchaudio

    Silero VAD is loaded via torch.hub and does not require a separate
    pip package. An internet connection is needed on first run to download
    the model.
"""

import argparse
import io
import json
import logging
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vad")

# ---------------------------------------------------------------------------
# Supported audio formats
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "audio"
VAD_DIR = PROJECT_ROOT / "vad"

# Default VAD parameters (config-savable)
DEFAULTS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "window_size_samples": 512,
    "speech_pad_ms": 30,
    "sampling_rate": 16000,
}

CONFIG_PATH = PROJECT_ROOT / "vad_config.json"

# Expected types for config validation
_CONFIG_TYPES = {
    "threshold": (int, float),
    "min_speech_duration_ms": int,
    "min_silence_duration_ms": int,
    "window_size_samples": int,
    "speech_pad_ms": int,
    "sampling_rate": int,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def format_seconds(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for display."""
    hrs, rem = divmod(seconds, 3600)
    mins, secs = divmod(rem, 60)
    return f"{int(hrs):02d}:{int(mins):02d}:{secs:06.3f}"


def load_config() -> dict:
    """Load vad_config.json if it exists, returning an empty dict otherwise.

    Invalid JSON, wrong types, or unknown keys are handled gracefully:
    - Parse errors produce a warning and fall back to defaults.
    - Keys with wrong types are logged and skipped.
    - Unknown keys are silently ignored (forward-compatible).
    """
    if not CONFIG_PATH.is_file():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read %s: %s", CONFIG_PATH.name, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("%s is not a JSON object, ignoring", CONFIG_PATH.name)
        return {}

    # Validate types for known keys
    for key, expected in _CONFIG_TYPES.items():
        if key in data and not isinstance(data[key], expected):
            type_names = (
                expected.__name__ if isinstance(expected, type)
                else "/".join(t.__name__ for t in expected)
            )
            log.warning(
                "%s: %s has invalid type %s, expected %s; ignoring",
                CONFIG_PATH.name, key, type(data[key]).__name__, type_names,
            )
            del data[key]

    return data


def resolve_params(args: argparse.Namespace, config: dict) -> dict:
    """Resolve effective parameters: CLI > config file > built-in defaults.

    For each config-savable parameter, if the user did not explicitly
    provide a CLI value (detected via None), fall back to config,
    then to DEFAULTS.

    Returns a dict of parameter name -> resolved value.
    """
    params = {}
    for key, default_value in DEFAULTS.items():
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            params[key] = cli_value
        elif key in config:
            params[key] = config[key]
        else:
            params[key] = default_value
    return params


def save_config(params: dict) -> None:
    """Save parameters to vad_config.json."""
    CONFIG_PATH.write_text(
        json.dumps(params, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    log.info("Configuration saved to %s", CONFIG_PATH)


def gather_audio_files(directory: Path) -> list[Path]:
    """Collect and sort audio files from *directory*.

    Exits with an error if the directory does not exist or contains no
    supported audio files.
    """
    if not directory.exists():
        log.error("Directory not found: %s", directory)
        sys.exit(1)

    files = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        log.error("No audio files found in %s", directory)
        sys.exit(1)

    return files


def load_audio(audio_path: Path, sampling_rate: int = 16000) -> torch.Tensor:
    """Load an audio file and resample to *sampling_rate* Hz mono float32.

    Uses ffmpeg for decoding (supports all common formats) and the stdlib
    wave module for parsing.  This avoids a hard dependency on
    torchaudio / torchcodec which may fail to find FFmpeg on some systems.

    Returns a 1-D float32 tensor with values in [-1, 1].
    """
    result = subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-i", str(audio_path),
            "-f", "wav",
            "-ar", str(sampling_rate),
            "-ac", "1",          # mono
            "-loglevel", "error",
            "-",                 # output to stdout
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}): "
            f"{result.stderr.decode(errors='replace').strip()}"
        )

    wav_io = io.BytesIO(result.stdout)
    with wave.open(wav_io, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        # Determine sample width and convert to float32
        sampwidth = wf.getsampwidth()
        if sampwidth == 2:
            wav_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            wav_np = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            wav_np = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    if sr != sampling_rate:
        log.warning(
            "ffmpeg output sample rate (%d) differs from target (%d)",
            sr, sampling_rate,
        )

    return torch.from_numpy(wav_np)


def save_wav(path: str, tensor: torch.Tensor, sampling_rate: int = 16000):
    """Save a 1-D float32 tensor as a 16-bit PCM WAV file."""
    # Clamp and convert to int16
    clipped = tensor.clamp(-1.0, 1.0)
    pcm = (clipped.numpy() * 32768.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sampling_rate)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# VADProcessor
# ---------------------------------------------------------------------------
class VADProcessor:
    """Wraps Silero VAD model for speech segment detection and audio clipping."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        window_size_samples: int = 512,
        speech_pad_ms: int = 30,
        sampling_rate: int = 16000,
    ):
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.sampling_rate = sampling_rate
        self.model = None
        self.get_speech_timestamps = None

    def load(self):
        """Load the Silero VAD model from torch.hub."""
        if self.model is not None:
            return

        log.info("Loading Silero VAD model from torch.hub ...")
        start = time.time()
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        (self.get_speech_timestamps, *_rest) = utils
        elapsed = time.time() - start
        log.info("Silero VAD model loaded in %.1f s", elapsed)

    def process(self, audio_path: Path) -> dict:
        """Run VAD on *audio_path*, returning a structured result dict.

        Returns a dict with keys:
            source_file, duration_s, sample_rate, parameters, segments,
            total_speech_s, total_silence_s, speech_ratio
        """
        if self.model is None:
            self.load()

        log.info("Loading audio: %s", audio_path.name)
        audio = load_audio(audio_path, sampling_rate=self.sampling_rate)

        duration_s = audio.shape[0] / self.sampling_rate
        log.info(
            "Audio loaded: %s (%s, %d Hz)",
            format_seconds(duration_s),
            audio_path.name,
            self.sampling_rate,
        )

        # Edge case: audio shorter than one window
        if audio.shape[0] < self.window_size_samples:
            log.warning(
                "Audio too short (%d samples < window size %d), skipping VAD",
                audio.shape[0],
                self.window_size_samples,
            )
            return self._empty_result(audio_path, duration_s)

        log.info("Running VAD ...")
        start = time.time()
        speech_timestamps = self.get_speech_timestamps(
            audio,
            self.model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms,
            window_size_samples=self.window_size_samples,
            speech_pad_ms=self.speech_pad_ms,
        )
        elapsed = time.time() - start
        log.info(
            "VAD complete in %.1f s (%.1f min audio, %.1fx real-time)",
            elapsed,
            duration_s / 60,
            elapsed / max(duration_s, 0.01),
        )

        if not speech_timestamps:
            log.warning("No speech detected in %s", audio_path.name)
            return self._empty_result(audio_path, duration_s)

        # Build segments list
        segments = []
        total_speech_s = 0.0
        for i, seg in enumerate(speech_timestamps, start=1):
            start_s = seg["start"] / self.sampling_rate
            end_s = seg["end"] / self.sampling_rate
            dur = end_s - start_s
            total_speech_s += dur
            segments.append({
                "index": i,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "duration_s": round(dur, 3),
                "confidence": None,  # Silero VAD does not provide per-segment confidence
            })

        total_silence_s = duration_s - total_speech_s
        speech_ratio = round(total_speech_s / duration_s, 4) if duration_s > 0 else 0.0

        log.info(
            "Detected %d speech segment(s) — %s speech / %s total (%.0f%%)",
            len(segments),
            format_seconds(total_speech_s),
            format_seconds(duration_s),
            speech_ratio * 100,
        )

        return {
            "source_file": audio_path.name,
            "duration_s": round(duration_s, 3),
            "sample_rate": self.sampling_rate,
            "parameters": {
                "threshold": self.threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "window_size_samples": self.window_size_samples,
                "speech_pad_ms": self.speech_pad_ms,
            },
            "segments": segments,
            "total_speech_s": round(total_speech_s, 3),
            "total_silence_s": round(total_silence_s, 3),
            "speech_ratio": speech_ratio,
        }

    def _empty_result(self, audio_path: Path, duration_s: float) -> dict:
        """Return a result dict with zero segments (for short audio or no speech)."""
        return {
            "source_file": audio_path.name,
            "duration_s": round(duration_s, 3),
            "sample_rate": self.sampling_rate,
            "parameters": {
                "threshold": self.threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "window_size_samples": self.window_size_samples,
                "speech_pad_ms": self.speech_pad_ms,
            },
            "segments": [],
            "total_speech_s": 0.0,
            "total_silence_s": round(duration_s, 3),
            "speech_ratio": 0.0,
        }


# ---------------------------------------------------------------------------
# Audio clip saving
# ---------------------------------------------------------------------------
def save_segment_clips(
    audio_path: Path,
    result: dict,
    output_dir: Path,
) -> list[Path]:
    """Save each speech segment as an individual .wav clip.

    Parameters
    ----------
    audio_path : Path
        Original audio file path.
    result : dict
        VAD result dict from VADProcessor.process().
    output_dir : Path
        Per-file output directory (e.g., vad/interview_01/).

    Returns
    -------
    list[Path]
        Paths of all saved clip files.
    """
    segments = result["segments"]
    if not segments:
        return []

    sampling_rate = result["sample_rate"]

    # Re-read the audio to get a fresh tensor for clipping
    audio = load_audio(audio_path, sampling_rate=sampling_rate)

    stem = audio_path.stem
    saved_paths = []

    for seg in segments:
        idx = seg["index"]
        start_sample = int(seg["start_s"] * sampling_rate)
        end_sample = int(seg["end_s"] * sampling_rate)

        # Clamp to valid range
        start_sample = max(0, start_sample)
        end_sample = min(audio.shape[0], end_sample)

        clip_tensor = audio[start_sample:end_sample]

        clip_name = f"{stem}_seg_{idx:03d}.wav"
        clip_path = output_dir / clip_name

        save_wav(str(clip_path), clip_tensor, sampling_rate=sampling_rate)

        saved_paths.append(clip_path)
        log.info(
            "  Saved clip %s (%.3f s)",
            clip_name,
            seg["duration_s"],
        )

    return saved_paths


def save_merged_clip(
    audio_path: Path,
    result: dict,
    output_dir: Path,
) -> Path | None:
    """Concatenate all speech segments into a single merged WAV file.

    Returns the path to the merged file, or None if there are no segments.
    """
    segments = result["segments"]
    if not segments:
        return None

    sampling_rate = result["sample_rate"]

    # Re-read the audio and extract each segment
    audio = load_audio(audio_path, sampling_rate=sampling_rate)

    clips = []
    for seg in segments:
        start_sample = int(seg["start_s"] * sampling_rate)
        end_sample = int(seg["end_s"] * sampling_rate)
        start_sample = max(0, start_sample)
        end_sample = min(audio.shape[0], end_sample)
        clips.append(audio[start_sample:end_sample])

    merged = torch.cat(clips, dim=0)

    merged_name = f"{audio_path.stem}_merged.wav"
    merged_path = output_dir / merged_name
    save_wav(str(merged_path), merged, sampling_rate=sampling_rate)

    merged_dur_s = merged.shape[0] / sampling_rate
    log.info(
        "  Merged clip -> %s (%s)",
        merged_name,
        format_seconds(merged_dur_s),
    )

    return merged_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="VAD: Detect speech segments in audio using Silero VAD",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single audio file to process (default: scan audio/ directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VAD_DIR,
        help="Output directory for VAD results (default: vad/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Speech probability threshold, 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=None,
        help="Minimum speech segment duration in ms (default: 250)",
    )
    parser.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=None,
        help="Minimum silence between speech segments in ms (default: 100)",
    )
    parser.add_argument(
        "--window-size-samples",
        type=int,
        default=None,
        help="Window size in samples for 16kHz audio (default: 512; use 256 for 8kHz)",
    )
    parser.add_argument(
        "--speech-pad-ms",
        type=int,
        default=None,
        help="Padding added to each side of speech segment in ms (default: 30)",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Target sampling rate for VAD (default: 16000; allowed: 8000, 16000)",
    )
    parser.add_argument(
        "--no-clips",
        action="store_true",
        default=False,
        help="Skip saving individual .wav clip files (report only)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        default=False,
        help="Concatenate all speech segments into a single _merged.wav file",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save current parameters to vad_config.json and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        default=False,
        help="Display effective parameters (defaults + config + CLI) and exit",
    )
    args = parser.parse_args()

    # --- Load config and resolve parameters ------------------------------------
    config = load_config()
    params = resolve_params(args, config)

    # --- Handle action flags ---------------------------------------------------
    if args.show_config:
        print("Effective VAD parameters:")
        for key in DEFAULTS:
            if getattr(args, key, None) is not None:
                source = "CLI"
            elif key in config:
                source = "config"
            else:
                source = "default"
            print(f"  {key}: {params[key]}  (from {source})")
        sys.exit(0)

    if args.save_config:
        save_config(params)
        sys.exit(0)

    # --- Validate parameters ---------------------------------------------------
    if params["threshold"] < 0.0 or params["threshold"] > 1.0:
        log.error("threshold must be between 0.0 and 1.0, got %.2f", params["threshold"])
        sys.exit(1)

    if params["min_speech_duration_ms"] < 0:
        log.error(
            "min_speech_duration_ms must be non-negative, got %d",
            params["min_speech_duration_ms"],
        )
        sys.exit(1)

    if params["min_silence_duration_ms"] < 0:
        log.error(
            "min_silence_duration_ms must be non-negative, got %d",
            params["min_silence_duration_ms"],
        )
        sys.exit(1)

    if params["speech_pad_ms"] < 0:
        log.error("speech_pad_ms must be non-negative, got %d", params["speech_pad_ms"])
        sys.exit(1)

    if params["sampling_rate"] not in (8000, 16000):
        log.error("sampling_rate must be 8000 or 16000, got %d", params["sampling_rate"])
        sys.exit(1)

    if params["window_size_samples"] <= 0:
        log.error("window_size_samples must be positive, got %d", params["window_size_samples"])
        sys.exit(1)

    if params["sampling_rate"] == 8000 and params["window_size_samples"] > 256:
        log.warning(
            "For 8kHz audio, window_size_samples is typically 256 (got %d). "
            "Results may be suboptimal.",
            params["window_size_samples"],
        )
    if params["sampling_rate"] == 16000 and params["window_size_samples"] < 512:
        log.warning(
            "For 16kHz audio, window_size_samples is typically 512+ (got %d). "
            "Results may be suboptimal.",
            params["window_size_samples"],
        )

    # --- Determine input files -------------------------------------------------
    if args.input:
        if not args.input.exists():
            log.error("Input file not found: %s", args.input)
            sys.exit(1)
        audio_files = [args.input.resolve()]
    else:
        audio_files = gather_audio_files(AUDIO_DIR)
        log.info("Found %d audio file(s) in %s", len(audio_files), AUDIO_DIR)

    # --- Ensure output directory exists ----------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)

    # --- Instantiate processor (model loads lazily) ----------------------------
    processor = VADProcessor(**params)

    # --- Process each file -----------------------------------------------------
    for audio_path in audio_files:
        log.info("=" * 60)
        log.info("Processing: %s", audio_path.name)

        try:
            result = processor.process(audio_path)
        except Exception as exc:
            log.error("Failed to process %s: %s", audio_path.name, exc)
            continue

        # Per-file output directory
        file_output_dir = args.output / audio_path.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        json_name = f"{audio_path.stem}_vad.json"
        json_path = file_output_dir / json_name
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        segment_count = len(result["segments"])
        log.info(
            "Saved %d segment(s) -> %s (%s speech, %.0f%% ratio)",
            segment_count,
            json_name,
            format_seconds(result["total_speech_s"]),
            result["speech_ratio"] * 100,
        )

        # Save individual clips
        if not args.no_clips and segment_count > 0:
            clip_paths = save_segment_clips(
                audio_path, result, file_output_dir,
            )
            log.info(
                "Saved %d audio clip(s) to %s/",
                len(clip_paths),
                file_output_dir.name,
            )

        # Save merged clip (all speech segments concatenated)
        if args.merge and segment_count > 0:
            save_merged_clip(audio_path, result, file_output_dir)

    log.info("=" * 60)
    log.info("Done -- %d file(s) processed.", len(audio_files))


if __name__ == "__main__":
    main()

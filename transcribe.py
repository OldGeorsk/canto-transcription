#!/usr/bin/env python3
"""
Stage 1: Audio → Raw Transcript
================================
Converts Cantonese interview audio into verbatim transcripts using WhisperX.

Usage:
    # Auto-scan audio/ directory
    python transcribe.py

    # Single file
    python transcribe.py --input audio/interview_01.mp3

    # With speaker diarization
    python transcribe.py --hf-token hf_xxxx --diarize

    # Custom model and device
    python transcribe.py --model large-v2 --device cpu --language zh

    # Save current parameters to config file for reuse
    python transcribe.py --batch-size 8 --chunk-size 20 --save-config

    # Show effective parameters and their sources
    python transcribe.py --show-config

Setup:
    python -m venv venv
    source venv/Scripts/activate   # Windows
    pip install -r requirements.txt

Output format (per CLAUDE.md):
    [HH:MM]
    [SPEAKER_00]
    transcript text...

    [HH:MM]
    [SPEAKER_01]
    transcript text...
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import whisperx

# ---------------------------------------------------------------------------
# Load .env file (project-local secrets, e.g. HF_TOKEN)
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    """Load key=value pairs from PROJECT_ROOT/.env into os.environ (if present)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:  # don't override existing env vars
            os.environ[key] = value

_load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("transcribe")

# ---------------------------------------------------------------------------
# Supported audio formats
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus"}

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "audio"
RAW_DIR = PROJECT_ROOT / "raw"

# ---------------------------------------------------------------------------
# Default parameters and config
# ---------------------------------------------------------------------------
DEFAULT_ASR_OPTIONS = {
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "word_timestamps": False,
    "initial_prompt": None,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "beam_size": 5,
    "best_of": 5,
    "patience": 1,
    "length_penalty": 1,
    "repetition_penalty": 1,
    "no_repeat_ngram_size": 0,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "suppress_blank": True,
    "suppress_tokens": [-1],
    "max_new_tokens": None,
    "prefix": None,
    "multilingual": False,
    "suppress_numerals": False,
    "hallucination_silence_threshold": None,
    "hotwords": None,
}

DEFAULT_VAD_OPTIONS = {
    "vad_onset": 0.500,
    "vad_offset": 0.363,
}

DEFAULT_DIARIZE_OPTIONS = {
    "num_speakers": None,
    "min_speakers": None,
    "max_speakers": None,
    "threshold": None,
}

# ---------------------------------------------------------------------------
# Backend-specific defaults
# ---------------------------------------------------------------------------
WHISPERX_DEFAULTS = {
    "model": "large-v3",
    "compute_type": "float16",
    "batch_size": 16,
    "chunk_size": 30,
    "language": "yue",
    "device": "cuda",
}

QWEN_DEFAULTS = {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "batch_size": 1,
    "enable_timestamps": False,
    "return_segments": True,
    "qwen_chunk_s": 60,
    "language": "yue",
    "device": "cuda",
}

CONFIG_WHISPERX = PROJECT_ROOT / "transcribe_config_whisperx.json"
CONFIG_QWEN = PROJECT_ROOT / "transcribe_config_qwen.json"
CONFIG_LEGACY = PROJECT_ROOT / "transcribe_config.json"

# Expected types for top-level config validation
_CONFIG_TYPES: dict[str, type] = {
    "model": str,
    "compute_type": str,
    "batch_size": int,
    "chunk_size": int,
    "language": str,
    "device": str,
    "enable_timestamps": bool,
    "return_segments": bool,
    "qwen_chunk_s": int,
}


def _config_path(backend: str) -> Path:
    """Return the config file path for *backend*."""
    return CONFIG_WHISPERX if backend == "whisperx" else CONFIG_QWEN


@dataclass
class ResolvedConfig:
    """Holds fully resolved parameters for the active backend."""
    backend: str                          # "whisperx" or "qwen"
    language: str
    device: str
    backend_options: dict                 # active backend's resolved options
    diarize_options: dict
    asr_options: dict | None = None       # whisperx-specific (None for qwen)
    vad_options: dict | None = None       # whisperx-specific (None for qwen)
    _raw_config: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Helper: format seconds → [HH:MM]
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """Convert seconds to [HH:MM] format string."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"[{hours:02d}:{minutes:02d}]"


# ---------------------------------------------------------------------------
# Helper: gather audio files
# ---------------------------------------------------------------------------
def gather_audio_files(directory: Path) -> list[Path]:
    """Collect all audio files from *directory* with recognised extensions."""
    if not directory.exists():
        log.error("Audio directory not found: %s", directory)
        sys.exit(1)

    files = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not files:
        log.error("No audio files found in %s", directory)
        sys.exit(1)

    log.info("Found %d audio file(s) in %s", len(files), directory)
    return files


def save_backend_config(backend: str, config: dict) -> None:
    """Write a flat config dict to the appropriate backend-specific file."""
    path = _config_path(backend)
    path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _migrate_legacy_config() -> dict | None:
    """If the legacy shared config file exists, migrate it to backend-specific files.

    Returns the whisperx config dict from the old file, or None.
    After migration the legacy file is renamed to .bak.
    """
    if not CONFIG_LEGACY.is_file():
        return None
    try:
        data = json.loads(CONFIG_LEGACY.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read legacy %s: %s", CONFIG_LEGACY.name, exc)
        return None
    if not isinstance(data, dict):
        return None

    log.info("Migrating legacy config %s to backend-specific files", CONFIG_LEGACY.name)

    # Extract whisperx section from nested format, or use top-level for old format
    whisperx_data = data.get("whisperx", {}) if "whisperx" in data else data
    if isinstance(whisperx_data, dict) and whisperx_data:
        # Merge with language/device from shared section if present
        for k in ("language", "device"):
            if k in data and k not in whisperx_data:
                whisperx_data[k] = data[k]
        save_backend_config("whisperx", whisperx_data)

    # Extract qwen section if present
    qwen_data = data.get("qwen", {})
    if isinstance(qwen_data, dict) and qwen_data:
        for k in ("language", "device"):
            if k in data and k not in qwen_data:
                qwen_data[k] = data[k]
        save_backend_config("qwen", qwen_data)

    # Also extract diarize_options to whisperx file
    di = data.get("diarize_options", {})
    if isinstance(di, dict) and di and whisperx_data:
        whisperx_data["diarize_options"] = di
        save_backend_config("whisperx", whisperx_data)

    # Rename legacy file to backup
    try:
        CONFIG_LEGACY.rename(CONFIG_LEGACY.with_suffix(".json.bak"))
        log.info("Legacy config backed up as %s", CONFIG_LEGACY.with_suffix(".json.bak").name)
    except OSError:
        log.warning("Could not rename legacy config; please delete %s manually", CONFIG_LEGACY.name)

    return whisperx_data if isinstance(whisperx_data, dict) and whisperx_data else None


def load_config(backend: str) -> dict:
    """Load the config file for *backend*, returning a flat dict.

    Automatically migrates legacy config on first access.
    Returns an empty dict if no config file exists.
    """
    # One-time migration of legacy shared config
    _migrate_legacy_config()

    path = _config_path(backend)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Failed to read %s: %s", path.name, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("%s is not a JSON object, ignoring", path.name)
        return {}

    # Validate types for known keys
    for key, expected in _CONFIG_TYPES.items():
        if key in data and not isinstance(data[key], expected):
            log.warning(
                "%s: %s has invalid type %s, expected %s; ignoring",
                path.name, key, type(data[key]).__name__, expected.__name__,
            )
            del data[key]

    return data


def resolve_config(args: argparse.Namespace, config: dict, backend: str) -> ResolvedConfig:
    """Resolve effective parameters for *backend*.

    Priority: CLI arguments > config file > built-in defaults.

    *config* is a flat dict loaded from the backend-specific file.
    Returns a ResolvedConfig.
    """
    backend_defaults = (
        WHISPERX_DEFAULTS if backend == "whisperx" else QWEN_DEFAULTS
    )

    # --- Resolve each key: CLI > config > default ---
    backend_options = {}
    for key, default_value in backend_defaults.items():
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            backend_options[key] = cli_value
        elif key in config:
            backend_options[key] = config[key]
        else:
            backend_options[key] = default_value

    # Shared params (language, device are in backend_options too)
    language = backend_options["language"]
    device = backend_options["device"]

    # --- For WhisperX: also resolve nested asr_options and vad_options ---
    asr_options = None
    vad_options = None
    if backend == "whisperx":
        asr_opts_cfg = config.get("asr_options", {})
        if isinstance(asr_opts_cfg, dict):
            asr_options = {**DEFAULT_ASR_OPTIONS, **asr_opts_cfg}
        else:
            asr_options = dict(DEFAULT_ASR_OPTIONS)

        vad_opts_cfg = config.get("vad_options", {})
        if isinstance(vad_opts_cfg, dict):
            vad_options = {**DEFAULT_VAD_OPTIONS, **vad_opts_cfg}
        else:
            vad_options = dict(DEFAULT_VAD_OPTIONS)

    # --- Diarization options ---
    diarize_config = config.get("diarize_options", {})
    if not isinstance(diarize_config, dict):
        diarize_config = {}
    diarize_options = {}
    for key in ("num_speakers", "min_speakers", "max_speakers", "threshold"):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            diarize_options[key] = cli_val
        elif key in diarize_config:
            diarize_options[key] = diarize_config[key]
        else:
            diarize_options[key] = DEFAULT_DIARIZE_OPTIONS[key]

    return ResolvedConfig(
        backend=backend,
        language=language,
        device=device,
        backend_options=backend_options,
        diarize_options=diarize_options,
        asr_options=asr_options,
        vad_options=vad_options,
        _raw_config=config,
    )


def save_config(resolved: ResolvedConfig) -> None:
    """Save parameters to the active backend's config file.

    Writes a flat dict.  Only the active backend's file is touched.
    """
    backend = resolved.backend

    # Build flat config dict
    new_config = dict(resolved.backend_options)  # shallow copy

    # For WhisperX: embed asr_options and vad_options as sub-dicts
    if backend == "whisperx":
        asr_clean = {
            k: v for k, v in (resolved.asr_options or {}).items()
            if v is not None
        }
        vad_clean = {
            k: v for k, v in (resolved.vad_options or {}).items()
            if v is not None
        }
        new_config["asr_options"] = asr_clean
        new_config["vad_options"] = vad_clean

    # Diarization options
    di_clean = {
        k: v for k, v in resolved.diarize_options.items()
        if v is not None
    }
    if di_clean:
        new_config["diarize_options"] = di_clean

    save_backend_config(backend, new_config)
    log.info("Configuration saved to %s", _config_path(backend))


def show_config(resolved: ResolvedConfig, args: argparse.Namespace) -> None:
    """Display effective parameters grouped by section.

    Shows: [Backend: Options], [asr_options] (whisperx only),
    [vad_options] (whisperx only), [diarize_options],
    with source annotations (CLI / config / default).
    """
    raw = resolved._raw_config
    config_file = _config_path(resolved.backend)

    print(f"Effective transcribe parameters ({config_file.name}):\n")

    # --- Active backend ---
    print(f"  [{resolved.backend}: Options]")
    for key in sorted(resolved.backend_options):
        value = resolved.backend_options[key]
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            source = "CLI"
        elif key in raw:
            source = "config"
        else:
            source = "default"
        print(f"    {key:<20} {value!r:<22} (from {source})")

    # --- WhisperX: asr_options ---
    if resolved.backend == "whisperx" and resolved.asr_options:
        print()
        print(f"  [asr_options]")
        asr_cfg = raw.get("asr_options", {})
        for key in sorted(DEFAULT_ASR_OPTIONS):
            value = resolved.asr_options.get(key)
            source = "config" if isinstance(asr_cfg, dict) and key in asr_cfg else "default"
            print(f"    {key:<32} {value!r:<22} (from {source})")

    # --- WhisperX: vad_options ---
    if resolved.backend == "whisperx" and resolved.vad_options:
        print()
        print(f"  [vad_options]")
        vad_cfg = raw.get("vad_options", {})
        for key in sorted(DEFAULT_VAD_OPTIONS):
            value = resolved.vad_options.get(key)
            source = "config" if isinstance(vad_cfg, dict) and key in vad_cfg else "default"
            print(f"    {key:<32} {value!r:<22} (from {source})")

    # --- Diarization ---
    print()
    print("  [diarize_options]")
    di_cfg = raw.get("diarize_options", {})
    for key in ("num_speakers", "min_speakers", "max_speakers", "threshold"):
        value = resolved.diarize_options.get(key)
        if value is not None:
            cli_val = getattr(args, key, None)
            source = "CLI" if cli_val is not None else (
                "config" if isinstance(di_cfg, dict) and key in di_cfg else "default"
            )
        else:
            source = "default"
        print(f"    {key:<18} {value!r:<22} (from {source})")


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------
class Transcriber:
    """Multi-backend ASR transcription with optional speaker diarization.

    Backend dispatch is handled internally.  External callers use the
    same interface regardless of backend.
    """

    def __init__(
        self,
        backend: str = "whisperx",
        model_name: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "yue",
        hf_token: str | None = None,
        asr_options: dict | None = None,
        vad_options: dict | None = None,
        chunk_size: int = 30,
        batch_size: int = 16,
        # Qwen-specific (ignored for whisperx):
        enable_timestamps: bool = False,
        return_segments: bool = True,
        qwen_chunk_s: int = 60,
    ):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.hf_token = hf_token
        self.asr_options = asr_options
        self.vad_options = vad_options
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.qwen_chunk_s = qwen_chunk_s
        self.enable_timestamps = enable_timestamps
        self.return_segments = return_segments
        self.model = None
        self.processor = None   # Qwen AutoProcessor
        self.diarize_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self):
        """Load the ASR model for the active backend (lazy)."""
        if self.model is not None:
            return
        if self.backend == "whisperx":
            self._load_whisperx()
        elif self.backend == "qwen":
            self._load_qwen()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def load_diarize(self):
        """Load the pyannote diarization pipeline (backend-agnostic)."""
        if self.diarize_model is not None:
            return
        if not self.hf_token:
            raise ValueError(
                "HF token is required for diarization. "
                "Pass --hf-token or set HF_TOKEN env var."
            )
        log.info("Loading speaker diarization pipeline …")
        start = time.time()
        self.diarize_model = whisperx.diarize.DiarizationPipeline(
            token=self.hf_token,
            device=self.device,
        )
        elapsed = time.time() - start
        log.info("Diarization pipeline loaded in %.1f s", elapsed)

    def load_audio(self, audio_path: Path):
        """Load audio file, returning a numpy array at 16 kHz mono.

        Uses whisperx.load_audio() which wraps torchaudio + ffmpeg.
        This works for all backends.
        """
        return whisperx.load_audio(str(audio_path))

    def transcribe(
        self,
        audio_path: Path,
        batch_size: int = 16,
        diarize: bool = False,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        threshold: float | None = None,
    ) -> dict:
        """Run transcription on *audio_path*, returning a result dict with
        ``"segments"`` (list of dicts with start/end/text/speaker).

        If *diarize* is True, speaker labels are assigned to each segment.
        """
        self.load()
        log.info("Loading audio: %s", audio_path.name)
        audio = self.load_audio(audio_path)

        # --- ASR (backend-specific) ---
        if self.backend == "whisperx":
            result = self._transcribe_whisperx(audio, batch_size)
        elif self.backend == "qwen":
            result = self._transcribe_qwen(audio, batch_size)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        # --- Speaker diarization (shared post-processing) ---
        if diarize:
            if self.backend == "qwen" and not self.enable_timestamps:
                log.warning(
                    "Qwen diarization requires --enable-timestamps. "
                    "Skipping speaker assignment for this run."
                )
            else:
                self.load_diarize()

                # Log diarization settings
                diarize_args: dict = {}
                if num_speakers is not None:
                    diarize_args["num_speakers"] = num_speakers
                    log.info("  num_speakers = %d", num_speakers)
                if min_speakers is not None:
                    diarize_args["min_speakers"] = min_speakers
                    log.info("  min_speakers = %d", min_speakers)
                if max_speakers is not None:
                    diarize_args["max_speakers"] = max_speakers
                    log.info("  max_speakers = %d", max_speakers)
                if threshold is not None:
                    self.diarize_model.model.threshold = threshold
                    log.info("  clustering threshold = %.2f", threshold)

                log.info("Running speaker diarization …")
                start = time.time()
                diarize_segments = self.diarize_model(audio, **diarize_args)

                if self.backend == "whisperx":
                    # Word-level precision via WhisperX helper
                    result = whisperx.assign_word_speakers(
                        diarize_segments, result,
                    )
                else:
                    # Backend-independent segment-overlap fallback
                    result = self._assign_speakers_by_overlap(
                        diarize_segments, result,
                    )

                elapsed = time.time() - start
                log.info("Diarization complete in %.1f s", elapsed)

                # Count unique speakers
                speakers = set(
                    seg.get("speaker", "UNKNOWN")
                    for seg in result.get("segments", [])
                )
                log.info(
                    "Detected %d speaker(s): %s",
                    len(speakers),
                    ", ".join(sorted(speakers)),
                )

        return result

    # ------------------------------------------------------------------
    # WhisperX backend
    # ------------------------------------------------------------------

    def _load_whisperx(self):
        """Load the WhisperX model into memory."""
        log.info(
            "Loading WhisperX model '%s' on %s (compute: %s) …",
            self.model_name,
            self.device,
            self.compute_type,
        )
        start = time.time()
        load_kwargs = {}
        if self.asr_options is not None:
            load_kwargs["asr_options"] = self.asr_options
        if self.vad_options is not None:
            load_kwargs["vad_options"] = self.vad_options
        self.model = whisperx.load_model(
            self.model_name,
            self.device,
            compute_type=self.compute_type,
            language=self.language,
            **load_kwargs,
        )
        elapsed = time.time() - start
        log.info("Model loaded in %.1f s", elapsed)

    def _transcribe_whisperx(self, audio, batch_size: int) -> dict:
        """Run WhisperX ASR on *audio* (numpy array)."""
        log.info("Transcribing with WhisperX …")
        start = time.time()
        result = self.model.transcribe(
            audio,
            language=self.language,
            batch_size=batch_size,
            chunk_size=self.chunk_size,
            print_progress=True,
        )
        elapsed = time.time() - start
        duration_min = audio.shape[0] / 16000 / 60
        log.info(
            "Transcription complete in %.1f s (audio: %.1f min, x%.1f real-time)",
            elapsed,
            duration_min,
            elapsed / max(duration_min, 0.01),
        )
        return result

    # ------------------------------------------------------------------
    # Qwen backend
    # ------------------------------------------------------------------

    def _load_qwen(self):
        """Lazy-import and load Qwen3-ASR model + processor via qwen-asr.

        Qwen3-ASR is NOT loaded via vanilla transformers — it requires
        the ``qwen-asr`` package which provides the model architecture.
        """
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            log.error(
                "Qwen3-ASR requires the qwen-asr package. "
                "Install with: pip install -r requirements-qwen.txt"
            )
            raise SystemExit(1) from exc

        import torch as _torch

        log.info(
            "Loading Qwen3-ASR model '%s' on %s …",
            self.model_name, self.device,
        )
        start = time.time()

        torch_dtype = _torch.bfloat16 if self.device == "cuda" else _torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_name,
            device_map=self.device if self.device == "cuda" else "cpu",
            dtype=torch_dtype,
            max_inference_batch_size=self.batch_size,
        )
        # The underlying processor is exposed for inference
        self.processor = self.model.processor

        elapsed = time.time() - start
        log.info("Qwen3-ASR model loaded in %.1f s", elapsed)

    def _transcribe_qwen(self, audio, _batch_size: int) -> dict:
        """Run Qwen3-ASR on *audio* (numpy float32 array at 16 kHz).

        Chunks long audio into ``qwen_chunk_s`` pieces (default 60 s) to
        avoid the model's internal 20-minute chunk limit eating all VRAM.

        Returns a dict with ``"segments"`` in the common format.
        """
        import numpy as np

        log.info("Transcribing with Qwen3-ASR …")
        start = time.time()

        # Ensure mono float32 numpy array
        if isinstance(audio, np.ndarray):
            audio_data = audio.astype(np.float32)
        else:
            audio_data = np.array(audio, dtype=np.float32)
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=0)

        # --- Manual chunking to limit peak VRAM ---
        sample_rate = 16000
        total_s = len(audio_data) / sample_rate
        chunk_s = getattr(self, 'qwen_chunk_s', 60)  # default 60 s per chunk
        chunk_samples = int(chunk_s * sample_rate)

        if total_s <= chunk_s:
            # Short audio: single pass
            results = self.model.transcribe(
                audio=(audio_data, sample_rate),
                context="",
                language=self.language,
                return_time_stamps=self.enable_timestamps,
            )
            result = results[0]
        else:
            # Long audio: split into chunks
            num_chunks = (len(audio_data) + chunk_samples - 1) // chunk_samples
            log.info(
                "Audio %.1f min — splitting into %d chunk(s) of %d s",
                total_s / 60, num_chunks, chunk_s,
            )
            all_text: list[str] = []
            all_time_stamps: list = []
            for i in range(num_chunks):
                c_start = i * chunk_samples
                c_end = min(c_start + chunk_samples, len(audio_data))
                chunk_audio = audio_data[c_start:c_end]
                log.info("  Chunk %d/%d (%.1f s) …", i + 1, num_chunks, len(chunk_audio) / sample_rate)
                chunk_results = self.model.transcribe(
                    audio=(chunk_audio, sample_rate),
                    context="",
                    language=self.language,
                    return_time_stamps=self.enable_timestamps,
                )
                cr = chunk_results[0]
                all_text.append(cr.text)
                if self.enable_timestamps and cr.time_stamps:
                    # Offset timestamps by chunk start time
                    offset_s = c_start / sample_rate
                    if isinstance(cr.time_stamps[0], dict):
                        for ts in cr.time_stamps:
                            ts["start"] = ts.get("start", 0) + offset_s
                            ts["end"] = ts.get("end", 0) + offset_s
                    else:
                        for ts in cr.time_stamps:
                            ts = list(ts)
                            ts[0] = ts[0] + offset_s
                            ts[1] = ts[1] + offset_s
                    all_time_stamps.extend(cr.time_stamps)
            # Reassemble: join all text
            result_text = "".join(all_text)
            # Create a synthetic ASRTranscription-like object
            class _Result:
                text = result_text
                language = self.language
                time_stamps = all_time_stamps if self.enable_timestamps else None
            result = _Result()

        # --- Parse into common segment format ---
        if self.enable_timestamps and result.time_stamps:
            segments = self._parse_qwen_aligned(result.time_stamps)
        else:
            segments = self._parse_qwen_text(result.text)

        elapsed = time.time() - start
        audio_dur_s = len(audio_data) / 16000
        if audio_dur_s > 0:
            log.info(
                "Qwen transcription complete in %.1f s "
                "(audio: %.1f min, x%.1f real-time)",
                elapsed, audio_dur_s / 60,
                elapsed / max(audio_dur_s, 0.01),
            )
        else:
            log.info("Qwen transcription complete in %.1f s", elapsed)

        return {"segments": segments, "language": result.language}

    # ------------------------------------------------------------------
    # Qwen output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_qwen_text(text: str) -> list[dict]:
        """Convert raw Qwen ASR text into segment dicts (no timestamps).

        Splits on Cantonese/Chinese sentence-ending punctuation as a
        heuristic for segment boundaries.
        """
        import re
        segments: list[dict] = []
        parts = re.split(r"(?<=[。！？!?\.])\s*", text)
        for part in parts:
            part = part.strip()
            if part:
                segments.append({
                    "start": 0.0,
                    "end": 0.0,
                    "text": part,
                    "speaker": "SPEAKER_00",
                })
        if not segments:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": text.strip(),
                "speaker": "SPEAKER_00",
            })
        return segments

    @staticmethod
    def _parse_qwen_aligned(time_stamps) -> list[dict]:
        """Parse aligned time_stamp segments into the common format.

        *time_stamps* is a list of ``(start_s, end_s, text)`` tuples
        returned by Qwen3-ASR's forced aligner.
        """
        segments: list[dict] = []
        for ts in time_stamps:
            if isinstance(ts, dict):
                start_s = float(ts.get("start", 0))
                end_s = float(ts.get("end", 0))
                text = str(ts.get("text", ""))
            else:
                start_s, end_s = float(ts[0]), float(ts[1])
                text = str(ts[2]) if len(ts) > 2 else ""
            if text.strip():
                segments.append({
                    "start": round(start_s, 3),
                    "end": round(end_s, 3),
                    "text": text.strip(),
                    "speaker": "SPEAKER_00",
                })
        if not segments:
            segments.append({
                "start": 0.0,
                "end": 0.0,
                "text": "",
                "speaker": "SPEAKER_00",
            })
        return segments

    # ------------------------------------------------------------------
    # Diarization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_speakers_by_overlap(diarize_segments, asr_result: dict) -> dict:
        """Assign speaker labels to ASR segments by timestamp overlap.

        Backend-independent alternative to ``whisperx.assign_word_speakers()``.
        For each ASR segment, finds the diarization segment with
        maximum temporal overlap and assigns its speaker label.

        *diarize_segments* is a pyannote diarization result (iterable of
        dicts with ``start``, ``end``, ``speaker`` keys, or a pyannote
        object with ``itertotal()`` / ``itertracks()``).
        """
        # Normalise diarization segments to a simple list of dicts
        di_segs: list[dict] = []
        if hasattr(diarize_segments, "itertracks"):
            for segment, _track, speaker in diarize_segments.itertracks(yield_label=True):
                di_segs.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                })
        elif hasattr(diarize_segments, "itertotal"):
            # older pyannote API
            for dseg in diarize_segments.itertotal():
                di_segs.append({
                    "start": dseg.get("start", 0.0),
                    "end": dseg.get("end", 0.0),
                    "speaker": dseg.get("speaker", "UNKNOWN"),
                })
        else:
            # Assume list of dicts
            for d in diarize_segments:
                di_segs.append({
                    "start": d.get("start", 0.0) if isinstance(d, dict) else 0.0,
                    "end": d.get("end", 0.0) if isinstance(d, dict) else 0.0,
                    "speaker": d.get("speaker", "UNKNOWN") if isinstance(d, dict) else "UNKNOWN",
                })

        for seg in asr_result.get("segments", []):
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for dseg in di_segs:
                overlap = max(
                    0.0,
                    min(seg_end, dseg["end"]) - max(seg_start, dseg["start"]),
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = dseg["speaker"]

            seg["speaker"] = best_speaker

        return asr_result


# ---------------------------------------------------------------------------
# Format output
# ---------------------------------------------------------------------------
def format_transcript(result: dict, has_diarization: bool = False) -> str:
    """Convert WhisperX segments into the CLAUDE.md transcript format.

    With speaker diarization, uses real speaker labels (SPEAKER_00, …).
    Without it, labels alternate A/B per VAD segment.
    """
    segments = result.get("segments", [])
    if not segments:
        log.warning("No segments found in transcription result")
        return ""

    lines: list[str] = []
    speaker_toggle = True
    prev_speaker = None

    for seg in segments:
        timestamp = format_timestamp(seg["start"])

        if has_diarization:
            # Use real speaker label from pyannote
            speaker = f"[{seg.get('speaker', 'UNKNOWN')}]"
        else:
            # Fallback: alternate A / B
            speaker = "[Speaker A]" if speaker_toggle else "[Speaker B]"
            speaker_toggle = not speaker_toggle

        text = seg["text"].strip()

        if text:
            lines.append(f"{timestamp}\n{speaker}\n{text}\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Transcribe Cantonese interview audio → raw verbatim transcript",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single audio file to transcribe (default: scan audio/ directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR,
        help="Output directory for raw transcripts (default: raw/)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["whisperx", "qwen"],
        help="ASR backend to use (default: whisperx). "
        "whisperx = WhisperX + CTranslate2; qwen = Qwen3-ASR via transformers",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for the selected backend "
        "(whisperx default: large-v3; qwen default: Qwen/Qwen3-ASR-1.7B)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code for ASR (default: yue for Cantonese)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (default: cuda; allowed: cuda, cpu)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=None,
        help="Compute precision (whisperx only; default: float16; "
        "allowed: float16, float32, int8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for transcription "
        "(whisperx default: 16, qwen default: 1)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Audio chunk length in seconds for VAD merging (whisperx only; default: 30)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="HuggingFace access token for speaker diarization "
        "(or set HF_TOKEN env var). Required for --diarize.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        default=False,
        help="Enable speaker diarization (requires --hf-token)",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Exact number of speakers in the interview (helps diarization accuracy)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers to detect",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers to detect",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Clustering threshold (0.0–1.0). Lower = more speakers; "
        "higher = fewer speakers. Default ~0.5",
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        default=False,
        help="Save current parameters to transcribe_config.json and exit",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        default=False,
        help="Display effective parameters (defaults + config + CLI) and exit",
    )
    args = parser.parse_args()

    # --- Determine backend (needed before config loading) -----------------------
    backend = args.backend or "whisperx"
    if backend not in ("whisperx", "qwen"):
        log.warning("Unknown backend '%s', falling back to whisperx", backend)
        backend = "whisperx"
    # Store resolved backend on args for downstream use
    args.backend = backend

    # --- Load config and resolve parameters ------------------------------------
    config = load_config(backend)
    resolved = resolve_config(args, config, backend)

    # --- Handle action flags ---------------------------------------------------
    if args.show_config:
        show_config(resolved, args)
        sys.exit(0)

    if args.save_config:
        save_config(resolved)
        sys.exit(0)

    # --- Determine input files -------------------------------------------------
    if args.input:
        if not args.input.exists():
            log.error("Input file not found: %s", args.input)
            sys.exit(1)
        audio_files = [args.input.resolve()]
    else:
        audio_files = gather_audio_files(AUDIO_DIR)

    # --- Validate device / compute_type ----------------------------------------
    if resolved.device == "cpu" and resolved.backend == "whisperx":
        if resolved.backend_options.get("compute_type") == "float16":
            log.warning(
                "float16 is not supported on CPU — switching to float32. "
                "Use --compute-type int8 for faster CPU inference."
            )
            resolved.backend_options["compute_type"] = "float32"

    # --- Ensure output directory exists ----------------------------------------
    args.output.mkdir(parents=True, exist_ok=True)

    # --- Validate diarization settings ------------------------------------------
    if args.diarize and not args.hf_token:
        log.error(
            "--diarize requires a HuggingFace token. "
            "Pass --hf-token or set HF_TOKEN env var."
        )
        sys.exit(1)

    di_threshold = resolved.diarize_options.get("threshold")
    if di_threshold is not None and (di_threshold < 0.0 or di_threshold > 1.0):
        log.error("threshold must be between 0.0 and 1.0, got %.2f", di_threshold)
        sys.exit(1)

    # Warn if speaker params used without --diarize
    if not args.diarize and any([
        resolved.diarize_options.get("num_speakers") is not None,
        resolved.diarize_options.get("min_speakers") is not None,
        resolved.diarize_options.get("max_speakers") is not None,
        resolved.diarize_options.get("threshold") is not None,
    ]):
        log.warning(
            "Speaker parameters (num_speakers, min_speakers, max_speakers, threshold) "
            "require --diarize. Ignoring."
        )

    # --- Warn if whisperx-specific flags used with qwen backend ----------------
    if resolved.backend == "qwen":
        if args.compute_type is not None:
            log.debug("--compute-type is ignored for qwen backend")
        if args.chunk_size is not None:
            log.debug("--chunk-size is ignored for qwen backend")

    # --- Build transcriber -----------------------------------------------------
    hf_token = args.hf_token if args.diarize else None

    # --- Map shared language to backend-specific format ----------------------
    # WhisperX uses ISO codes (yue), Qwen uses full names (Cantonese)
    lang_map = {"yue": "Cantonese", "zh": "Chinese"}
    backend_lang = resolved.language
    if resolved.backend == "qwen":
        backend_lang = lang_map.get(resolved.language, resolved.language)
    elif resolved.backend == "whisperx":
        inv_map = {v: k for k, v in lang_map.items()}
        backend_lang = inv_map.get(resolved.language, resolved.language)

    if resolved.backend == "whisperx":
        asr_opts_clean = {
            k: v for k, v in (resolved.asr_options or {}).items()
            if v is not None
        }
        vad_opts_clean = {
            k: v for k, v in (resolved.vad_options or {}).items()
            if v is not None
        }
        transcriber = Transcriber(
            backend="whisperx",
            model_name=resolved.backend_options["model"],
            device=resolved.device,
            compute_type=resolved.backend_options["compute_type"],
            language=backend_lang,
            hf_token=hf_token,
            asr_options=asr_opts_clean,
            vad_options=vad_opts_clean,
            chunk_size=resolved.backend_options["chunk_size"],
            batch_size=resolved.backend_options.get("batch_size", 16),
        )
    elif resolved.backend == "qwen":
        transcriber = Transcriber(
            backend="qwen",
            model_name=resolved.backend_options["model"],
            device=resolved.device,
            language=backend_lang,
            hf_token=hf_token,
            batch_size=resolved.backend_options.get("batch_size", 1),
            enable_timestamps=resolved.backend_options.get("enable_timestamps", False),
            return_segments=resolved.backend_options.get("return_segments", True),
            qwen_chunk_s=resolved.backend_options.get("qwen_chunk_s", 60),
        )
    else:
        log.error("Unknown backend: %s", resolved.backend)
        sys.exit(1)

    # --- Process each file -----------------------------------------------------
    for audio_path in audio_files:
        log.info("=" * 60)
        log.info("Processing: %s", audio_path.name)

        result = transcriber.transcribe(
            audio_path,
            batch_size=resolved.backend_options.get("batch_size", 16),
            diarize=args.diarize,
            num_speakers=resolved.diarize_options.get("num_speakers"),
            min_speakers=resolved.diarize_options.get("min_speakers"),
            max_speakers=resolved.diarize_options.get("max_speakers"),
            threshold=resolved.diarize_options.get("threshold"),
        )
        transcript = format_transcript(result, has_diarization=args.diarize)

        output_path = args.output / f"{audio_path.stem}.txt"
        output_path.write_text(transcript, encoding="utf-8")

        segment_count = len(result.get("segments", []))
        log.info(
            "Saved %d segments → %s (%d chars)",
            segment_count,
            output_path.name,
            len(transcript),
        )

    log.info("=" * 60)
    log.info("Done — %d file(s) processed.", len(audio_files))


if __name__ == "__main__":
    main()

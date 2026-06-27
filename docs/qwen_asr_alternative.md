# Qwen3-ASR as an Alternative ASR Backend

## Purpose

This note explains how to add Qwen3-ASR as an optional backup ASR model for this project while preserving the existing WhisperX-based transcription framework.

The goal is not to replace WhisperX immediately. WhisperX should remain the default production backend because the current pipeline already depends on its audio loading, ASR output structure, VAD options, timestamped segments, and pyannote speaker assignment.

Qwen3-ASR should be introduced as an alternative backend for comparison, fallback transcription, and future optimization of Cantonese ASR quality.

## Core Principles (ratified 2026-06-17)

### Principle 1: WhisperX is the immutable core

WhisperX remains the production backbone of this project. **Any new model or backend addition must never modify, overwrite, or monkey-patch the existing WhisperX code path.** This means:

- No changes to `whisperx.load_model()` call signatures, defaults, or parameter passing
- No changes to `_transcribe_whisperx()`, `_load_whisperx()`, or `format_transcript()`
- No shared mutable state between backends
- Backend-specific attributes (e.g. `qwen_chunk_s`, `enable_timestamps`) must never be read in the WhisperX path
- Backend selection must be via a clean `if/elif` dispatch with no fallthrough

**Qwen was added without violating this principle** — the two backends are fully isolated — but future additions must maintain this discipline.

### Principle 2: Separate config files per backend family **(ratified)**

WhisperX and Qwen have fundamentally different parameter sets (e.g. `compute_type` / `asr_options` / `vad_options` vs `enable_timestamps` / `qwen_chunk_s`). They now use **independent configuration files**:

| File | Backend |
|------|---------|
| `transcribe_config_whisperx.json` | WhisperX |
| `transcribe_config_qwen.json` | Qwen |

Each file contains only its own backend's parameters. The `--backend` flag determines which file is read or written. There is no shared file and no cross-contamination.

### Principle 3: Any backend can be shelved without cleanup

If Qwen proves unsuitable for production, removing it from active use must require **zero changes** to WhisperX code. The Qwen code may remain in `transcribe.py` (behind its `if self.backend == "qwen"` branch) or be extracted to a separate file, but its presence or absence must have no effect on WhisperX transcription.

---

## Original Design Rationale

Use a backend switch instead of rewriting the whole program.

Current structure:

```text
audio file
-> WhisperX ASR
-> optional pyannote diarization
-> transcript formatter
-> raw transcript
```

Recommended structure:

```text
audio file
-> selected ASR backend: whisperx or qwen
-> optional diarization
-> shared transcript formatter
-> raw transcript
```

This keeps the existing project framework stable while allowing Qwen to be tested on real Cantonese interview recordings.

## Why Qwen Cannot Be a Direct Model Swap

In the current `transcribe.py`, the `--model` option is a WhisperX model name, for example:

```bash
python transcribe.py --model large-v3
```

This cannot safely become:

```bash
python transcribe.py --model Qwen/Qwen3-ASR-1.7B
```

because Qwen and WhisperX use different inference APIs and may return different output formats. WhisperX currently provides or supports:

- audio loading through `whisperx.load_audio()`
- ASR through `whisperx.load_model().transcribe()`
- WhisperX-specific ASR options
- WhisperX VAD options
- segment timestamps
- pyannote speaker assignment through `whisperx.assign_word_speakers()`

Qwen3-ASR should therefore be wrapped in its own backend class and made to return the same internal segment format expected by the existing formatter.

## Proposed Command-Line Interface

Add a new argument:

```bash
--backend whisperx|qwen
```

Default:

```bash
--backend whisperx
```

Example commands:

```bash
# Existing behavior
python transcribe.py

# Existing behavior, explicit backend
python transcribe.py --backend whisperx

# Use Qwen3-ASR for one file
python transcribe.py --backend qwen --input audio/interview.m4a

# Use Qwen3-ASR with explicit model
python transcribe.py --backend qwen --model Qwen/Qwen3-ASR-1.7B --input audio/interview.m4a
```

The existing WhisperX workflow should continue to work without requiring Qwen dependencies.

## Proposed Code Structure

The current `Transcriber` class can be gradually separated into backend-specific classes:

```text
BaseTranscriber
├── WhisperXTranscriber
└── QwenTranscriber
```

For a minimal first implementation, a formal base class is optional. The important point is that both backends should return a common result structure.

Recommended internal result format:

```python
{
    "segments": [
        {
            "start": 0.0,
            "end": 12.5,
            "text": "原始轉錄文字",
            "speaker": "SPEAKER_00"
        }
    ]
}
```

The existing `format_transcript()` function can then remain mostly unchanged.

## Configuration Management

The current `transcribe_config.json` is WhisperX-specific. It contains options such as:

- `compute_type`
- `chunk_size`
- `asr_options`
- `vad_options`

These options should not be blindly passed to Qwen because they are WhisperX-specific or may have different meanings in Qwen.

The recommended solution is to make the configuration backend-aware.

## Recommended Config Layout

```json
{
  "backend": "whisperx",
  "language": "yue",
  "device": "cuda",

  "whisperx": {
    "model": "large-v3",
    "compute_type": "float16",
    "batch_size": 16,
    "chunk_size": 20,
    "asr_options": {
      "condition_on_previous_text": false,
      "without_timestamps": true,
      "word_timestamps": false,
      "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
      "beam_size": 5,
      "best_of": 5,
      "no_speech_threshold": 0.6,
      "compression_ratio_threshold": 2.4
    },
    "vad_options": {
      "vad_onset": 0.5,
      "vad_offset": 0.363
    }
  },

  "qwen": {
    "model": "Qwen/Qwen3-ASR-1.7B",
    "batch_size": 1,
    "enable_timestamps": false,
    "return_segments": true
  },

  "diarize_options": {
    "threshold": 0.5
  }
}
```

## Shared Options

These options can remain top-level because they are conceptually shared:

| Option | Purpose |
| --- | --- |
| `backend` | Selects `whisperx` or `qwen` |
| `language` | Keeps Cantonese as `yue` |
| `device` | Selects `cuda` or `cpu` |

## WhisperX-Specific Options

These should live under the `whisperx` key:

| Option | Reason |
| --- | --- |
| `model` | Whisper model name, such as `large-v3` |
| `compute_type` | WhisperX/CTranslate2 precision option |
| `batch_size` | WhisperX transcription batch size |
| `chunk_size` | WhisperX chunking behavior |
| `asr_options` | WhisperX ASR decoding parameters |
| `vad_options` | WhisperX internal VAD parameters |

## Qwen-Specific Options

These should live under the `qwen` key:

| Option | Reason |
| --- | --- |
| `model` | Qwen model ID, such as `Qwen/Qwen3-ASR-1.7B` |
| `batch_size` | Qwen inference batch size, likely smaller at first |
| `enable_timestamps` | Whether to request timestamps if supported |
| `return_segments` | Whether Qwen output should be normalized into segment objects |

The exact Qwen options should be adjusted after confirming the installed Qwen package API.

## Diarization Options

`diarize_options` can remain shared because diarization is conceptually separate from ASR.

However, the current implementation uses `whisperx.assign_word_speakers()`, which is WhisperX-specific. For Qwen support, diarization should eventually be handled through a backend-independent timestamp overlap method.

Recommended rollout:

1. First version: Qwen ASR without diarization.
2. Second version: Qwen ASR with segment-level timestamps.
3. Third version: pyannote diarization assigned to Qwen segments by timestamp overlap.

## Dependency Management

Qwen should be an optional dependency.

Do not import Qwen packages at the top of `transcribe.py`. Import them only when the Qwen backend is selected.

Recommended pattern:

```python
if backend == "qwen":
    from qwen_asr import QwenASR
```

This avoids breaking the existing WhisperX workflow for team members who have not installed Qwen.

Recommended documentation approach:

```text
requirements.txt          core WhisperX workflow
requirements-qwen.txt     optional Qwen backend dependencies
```

Example:

```bash
pip install -r requirements-qwen.txt
```

## Save-Config Behavior

The `--save-config` behavior should also become backend-aware.

When saving a WhisperX config, it should update only:

```json
"backend"
"language"
"device"
"whisperx"
"diarize_options"
```

When saving a Qwen config, it should update only:

```json
"backend"
"language"
"device"
"qwen"
"diarize_options"
```

It should not delete the other backend's existing settings. This allows researchers to switch between WhisperX and Qwen without losing carefully tuned parameters.

## Suggested Migration Path

1. Keep the current `transcribe_config.json` as a backup.
2. Add `backend` with default value `whisperx`.
3. Move existing WhisperX-specific fields into a new `whisperx` object.
4. Add a new `qwen` object with conservative defaults.
5. Update `resolve_config()` so it reads only the active backend settings.
6. Update `--show-config` so it clearly displays shared, WhisperX, Qwen, and diarization settings.
7. Update `--save-config` so it preserves inactive backend settings.

## Testing Plan

Use a small set of real Cantonese interview samples before recommending Qwen to the whole research team.

Suggested comparison outputs:

```text
qa/asr_comparison/
├── sample_01_whisperx.txt
├── sample_01_qwen.txt
├── sample_02_whisperx.txt
├── sample_02_qwen.txt
└── notes.md
```

Evaluation points:

- Cantonese wording accuracy
- preservation of fillers and hesitations
- hallucination rate
- English term preservation
- timestamp usefulness
- speaker turn compatibility
- runtime
- GPU memory usage
- installation difficulty for team members

## Recommended Final Position

For this project, Qwen3-ASR should be treated as an optional backup and comparison backend first.

Recommended production default:

```json
"backend": "whisperx"
```

Recommended Qwen test setting:

```json
"backend": "qwen"
```

This gives the research team flexibility while preserving the reproducibility and stability of the existing transcription workflow.

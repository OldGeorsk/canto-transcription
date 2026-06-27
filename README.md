# 粤语访谈转录工具

> 把粤语（广东话）访谈录音自动转为研究用文本的开源工具。

本工具面向**学术访谈研究**（香港 / 跨境学生经验等场景），将录音处理为可直接用于定性分析的文本。核心目标：**高识别准确率、低幻觉、保留原意、可复现、结构化输出**。

## 三阶段流程

1. **Stage 0（可选）音频预处理** — 用 VAD（Silero）分析录音中的语音段落，过滤静音
2. **Stage 1 音频 → 逐字转录** — WhisperX + pyannote，保留粤语口语、填充词、说话人标签、时间戳
3. **Stage 2 逐字转录 → 标准书面中文** — 粤语口语规范化为繁体书面语（在 AI 平台辅助下完成，本项目不提供自动化脚本）

---

## 目录

- [快速开始](#快速开始)
- [环境要求](#环境要求)
- [安装与配置](#安装与配置)
- [使用方法](#使用方法)
  - [Stage 0 — VAD 语音检测（可选）](#stage-0--vad-语音检测可选)
  - [Stage 1 — 音频转录](#stage-1--音频转录)
  - [切换 ASR 后端（WhisperX / Qwen3-ASR）](#切换-asr-后端whisperx--qwen3-asr)
  - [Stage 2 — 转标准书面中文](#stage-2--转标准书面中文)
- [配置文件](#配置文件)
- [预期性能](#预期性能)
- [常见问题](#常见问题)
- [目录结构](#目录结构)
- [许可证](#许可证)

---

## 快速开始

5 分钟从零跑通一次转录：

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate            # Windows（macOS/Linux 用 source .venv/bin/activate）
python -m pip install -r requirements.txt

# 2. 把录音文件放进 audio/ 目录

# 3. 转录（自动扫描 audio/ 下所有音频）
python transcribe.py

# 4. 结果在 raw/ 目录，文件名与录音一致，扩展名改为 .txt
```

如果需要区分说话人（访谈研究**强烈推荐**），先按 [安装与配置 → 说话人分离](#说话人分离可选推荐) 配好 HuggingFace Token，然后：

```bash
python transcribe.py --diarize --num-speakers 2
```

> 更多场景见下方 [使用方法](#使用方法)。

---

## 环境要求

首次使用前，需在电脑上完成以下准备：

| 要求 | 说明 |
|------|------|
| Python 3.12 | 需已加入系统 PATH |
| NVIDIA GPU | **推荐 RTX 3060 及以上**；最低 RTX 2060（CUDA 加速必需） |
| ffmpeg | 需已加入系统 PATH（WhisperX 解码 m4a / mp3 等格式时依赖） |
| 网络 | 首次运行需下载模型（约 3 GB），后续离线可用 |

<details>
<summary>ffmpeg 安装提示（点击展开）</summary>

- **Windows**：用 [winget](https://github.com/BtbN/FFmpeg-Builds) 或 [Gyan Builds](https://www.gyan.dev/ffmpeg/builds/) 下载，解压后把 `bin` 目录加入系统 PATH。装完重开命令行，运行 `ffmpeg -version` 验证。
- **macOS**：`brew install ffmpeg`
- **Linux**：`sudo apt install ffmpeg`

</details>

---

## 安装与配置

1. 打开项目目录 `auto-transcription/`
2. 打开命令行（cmd），确认 Python 版本正常：

```
python --version
```
应显示 `Python 3.12.x`

### 安装依赖

推荐使用虚拟环境（venv）隔离依赖，避免 whisperx / torch 等包污染系统 Python：

```
python -m venv .venv
.venv\Scripts\activate
# macOS / Linux 改为：source .venv/bin/activate
```

激活后安装依赖：

```
python -m pip install -r requirements.txt
```

**（可选）安装 Qwen3-ASR 后端：**

如需使用 Qwen 模型作为替代 ASR 引擎：

```
pip install -r requirements-qwen.txt
```

（Qwen 模型约 3-4 GB，首次运行时会自动下载，或手动下载：`huggingface-cli download Qwen/Qwen3-ASR-1.7B`）

**注意**：如果 `whisperx` 安装后覆盖了 GPU 版 PyTorch，需补装：

```
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

### 说话人分离（可选，推荐）

要启用自动说话人分离功能：

1. 注册 HuggingFace 账号：https://huggingface.co/join
2. 接受 pyannote 模型使用条款（两个都要点）：
   - https://huggingface.co/pyannote/speaker-diarization-3.1 → **Agree and access**
   - https://huggingface.co/pyannote/segmentation-3.0 → **Agree and access**
3. 生成 Access Token：https://huggingface.co/settings/tokens → **New token** → 类型选 **Read** → 复制 token
4. 将 token 写入项目根目录的 `.env` 文件：

```
HF_TOKEN=hf_你的token
```

---

## 使用方法

### Stage 0 — VAD 语音检测（可选）

在正式转录之前，可先用 VAD 分析录音中的语音分布：

1. 将录音文件放入 `audio/` 目录
2. 运行 VAD 分析：

```
python vad.py
```

- 自动扫描 `audio/` 目录下所有音频文件
- 结果输出到 `vad/` 目录，每个文件一个子文件夹
- 输出包含：JSON 语音段落报告 + 每个语音段落的独立 .wav 片段

3. 只想分析单个文件：

```
python vad.py --input audio/访谈文件名.mp3
```

4. 仅生成 JSON 报告，不输出音频片段：

```
python vad.py --no-clips
```

**常用参数：**

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--input` | 指定单个音频文件 | 扫描 `audio/` | `--input audio/interview.mp3` |
| `--output` | 输出目录 | `vad/` | `--output my_vad/` |
| `--threshold` | 语音概率阈值（0.0–1.0） | `0.5` | `--threshold 0.6` |
| `--min-speech-duration-ms` | 最短语音段落（毫秒） | `250` | `--min-speech-duration-ms 500` |
| `--min-silence-duration-ms` | 最短静音间隔（毫秒） | `100` | `--min-silence-duration-ms 200` |
| `--speech-pad-ms` | 语音段落两侧填充（毫秒） | `30` | `--speech-pad-ms 50` |
| `--window-size-samples` | 窗口采样数 | `512` | `--window-size-samples 256`（8kHz 时） |
| `--sampling-rate` | 采样率 | `16000` | `--sampling-rate 8000` |
| `--no-clips` | 跳过音频片段输出 | 关闭 | `--no-clips` |

**调参示例：**

严格检测（只保留高置信度语音，过滤更多背景音）：
```
python vad.py --threshold 0.7 --min-speech-duration-ms 500 --min-silence-duration-ms 200
```

宽松检测（保留更多短语音片段，适用于快速对话）：
```
python vad.py --threshold 0.3 --min-speech-duration-ms 100
```

**输出格式：**

```
vad/
└── 访谈文件名/
    ├── 访谈文件名_vad.json      ← JSON 语音段落报告
    ├── 访谈文件名_seg_001.wav   ← 语音片段 1
    ├── 访谈文件名_seg_002.wav   ← 语音片段 2
    └── ...
```

**JSON 报告示例：**

```json
{
  "source_file": "interview.mp3",
  "duration_s": 1847.32,
  "sample_rate": 16000,
  "parameters": {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 100,
    "window_size_samples": 512,
    "speech_pad_ms": 30
  },
  "segments": [
    {
      "index": 1,
      "start_s": 0.53,
      "end_s": 3.48,
      "duration_s": 2.95,
      "confidence": null
    }
  ],
  "total_speech_s": 456.79,
  "total_silence_s": 1390.53,
  "speech_ratio": 0.2473
}
```

> **提示**：VAD 是完全独立的前处理步骤，不会影响后续的 `transcribe.py` 转录流程。如果不需要预处理，可直接跳到 Stage 1。

---

### Stage 1 — 音频转录

1. 将录音文件放入 `audio/` 目录（支持 mp3、wav、m4a、flac 等格式）
2. 运行转录：

```
python transcribe.py
```

- 自动扫描 `audio/` 目录下所有音频文件
- 转录结果输出到 `raw/` 目录（文件名保持不变，扩展名改为 `.txt`）

3. 如果启用了说话人分离：

```
python transcribe.py --diarize
```

4. 如果只想转录单个文件：

```
python transcribe.py --input audio/访谈文件名.mp3
```

**常用参数：**

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--input` | 指定单个音频文件 | 扫描 `audio/` | `--input audio/interview.mp3` |
| `--output` | 输出目录 | `raw/` | `--output output/` |
| `--backend` | ASR 后端引擎 | `whisperx` | `--backend qwen`（切换 Qwen 模型） |
| `--model` | 模型名称 | 视后端而定 | `--model large-v3`（WhisperX）/ `--model Qwen/Qwen3-ASR-1.7B`（Qwen） |
| `--language` | 语言代码 | `yue`（粤语） | `--language zh` |
| `--device` | 计算设备 | `cuda`（GPU） | `--device cpu` |
| `--batch-size` | 批处理大小 | 16（WhisperX）/ 1（Qwen） | `--batch-size 8`（显存不足时调小） |
| `--chunk-size` | 音频分段长度（秒） | 30（仅 WhisperX） | `--chunk-size 20` |
| `--compute-type` | 计算精度 | `float16`（仅 WhisperX） | `--compute-type int8` |
| `--diarize` | 启用说话人分离 | 关闭 | `--diarize` |
| `--num-speakers` | 精确指定说话人数量 | 自动检测 | `--num-speakers 2` |
| `--min-speakers` | 最少说话人数 | 自动检测 | `--min-speakers 2` |
| `--max-speakers` | 最多说话人数 | 自动检测 | `--max-speakers 4` |
| `--threshold` | 聚类阈值（0.0–1.0） | ~0.5 | `--threshold 0.6` |
| `--save-config` | 保存当前参数到配置文件并退出 | — | `--save-config` |
| `--show-config` | 显示生效参数及来源并退出 | — | `--show-config` |

**说话人分离调参示例：**

已知访谈有 2 个说话人（最常见场景，强烈推荐指定）：
```
python transcribe.py --diarize --num-speakers 2
```

不确定人数，但知道范围：
```
python transcribe.py --diarize --min-speakers 2 --max-speakers 4
```

说话人标签太碎（同一个说话人被拆成多个标签）→ 调高阈值：
```
python transcribe.py --diarize --num-speakers 2 --threshold 0.7
```

说话人标签太粗（不同说话人被合在一起）→ 调低阈值：
```
python transcribe.py --diarize --num-speakers 2 --threshold 0.3
```

访谈场景推荐组合：
```
python transcribe.py --diarize --num-speakers 2 --threshold 0.6
```

> **阈值说明**：默认约 0.5，范围 0.0–1.0。值越高，聚类越保守（Speaker 更少）；值越低，聚类越激进（Speaker 更多）。`--num-speakers` 是最有效的参数，能大幅减少误判。

**保存和复用参数：**

```bash
# 保存当前参数到配置文件
python transcribe.py --batch-size 8 --chunk-size 20 --save-config

# 查看当前生效的参数及来源
python transcribe.py --show-config

# 配置文件已保存后，直接运行即可复用
python transcribe.py

# CLI 参数优先于配置文件
python transcribe.py --batch-size 4 --show-config
```

**输出格式示例：**

```
[00:00]
[SPEAKER_00]
好多谢你参与我哋嘅访问 我先介绍一下自己

[00:12]
[SPEAKER_01]
我係香港土生土长 依家係一个Designer
```

> 使用 `--diarize` 时，说话人标签为 `SPEAKER_00`、`SPEAKER_01` 等（由声纹自动区分）。
> 不使用 `--diarize` 时，标签交替为 `[Speaker A]`、`[Speaker B]`（无声纹区分，不推荐）。

---

### 切换 ASR 后端（WhisperX / Qwen3-ASR）

本工具支持两种 ASR 引擎，可通过 `--backend` 切换：

| 后端 | 说明 | 模型 | 安装要求 |
|------|------|------|----------|
| `whisperx`（默认） | WhisperX + CTranslate2，速度快、生态成熟 | `large-v3` | `requirements.txt` |
| `qwen` | Qwen3-ASR，粤语识别效果更优（实验性） | `Qwen/Qwen3-ASR-1.7B` | `requirements.txt` + `requirements-qwen.txt` |

**使用 Qwen 后端：**

```bash
# 1. 安装 Qwen 依赖（首次使用）
pip install -r requirements-qwen.txt

# 2. 下载 Qwen 模型（首次使用，约 3-4 GB）
huggingface-cli download Qwen/Qwen3-ASR-1.7B

# 3. 使用 Qwen 转录
python transcribe.py --backend qwen --input audio/访谈文件名.mp3

# 4. 将 Qwen 设为默认后端
python transcribe.py --backend qwen --save-config
```

**切换回 WhisperX：**

```bash
python transcribe.py --backend whisperx
# 或恢复默认
python transcribe.py --backend whisperx --save-config
```

> **注意**：Qwen 后端目前不支持说话人分离（`--diarize`）。如果需要说话人标签，请使用默认的 WhisperX 后端。两个后端的配置独立保存（各自一个文件），切换时不会丢失对方的参数。

---

### Stage 2 — 转标准书面中文

Stage 2 通过大模型辅助完成（本项目不提供自动化脚本）。**推荐使用 DeepSeek**（中文理解较好、性价比高），其他大模型（如 ChatGPT、Claude 等）亦可。

转换内容包括：

- 去除时间戳
- 保留说话人标签
- 合并同一说话人的连续轮次
- 粤语口语 → 标准书面中文（仅繁体中文）
- 保留英文词汇和不确定标记（`[inaudible]`、`[?]`、`[cross-talk]`）

> **提示**：粤语 → 书面中文的转换参考指引见 [`skills/spoken-to-written-Canto.md`](skills/spoken-to-written-Canto.md)，可配合 AI 工具使用。转换产物输出到 `normalized/` 目录。

---

## 配置文件

转录参数可以持久保存在项目根目录的 JSON 配置文件中，下次运行无需重复输入。

**优先级：CLI 参数 > 配置文件 > 内置默认值。**

每个 ASR 后端使用**独立的配置文件**（因为两者参数集本质不同，混在一个文件会造成混淆）。`--backend` 决定读取 / 写入哪个文件：

| 后端 | 配置文件 | 写入命令 |
|------|----------|----------|
| `whisperx`（默认） | `transcribe_config_whisperx.json` | `python transcribe.py --save-config` |
| `qwen` | `transcribe_config_qwen.json` | `python transcribe.py --backend qwen --save-config` |

每次运行参数（`--input`、`--output`、`--diarize`、`--hf-token`）**不保存**到配置文件，仅对当前调用生效。

### WhisperX 配置（`transcribe_config_whisperx.json`）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `model` | string | `"large-v3"` | Whisper 模型名称 |
| `device` | string | `"cuda"` | 计算设备（`cuda` / `cpu`） |
| `compute_type` | string | `"float16"` | 量化精度（`float16` / `float32` / `int8`） |
| `language` | string | `"yue"` | 语言代码 |
| `batch_size` | int | `16` | 批处理大小 |
| `chunk_size` | int | `30` | 音频分段长度（秒） |

`asr_options` 子参数（控制 Whisper 模型行为）：

| 键 | 默认值 | 说明 |
|----|--------|------|
| `condition_on_previous_text` | `false` | 用前一段输出作为下一段提示 |
| `without_timestamps` | `true` | 禁用时间戳输出 |
| `word_timestamps` | `false` | 输出词级时间戳 |
| `initial_prompt` | `null` | 给模型的首段提示（可放领域术语） |
| `temperatures` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | 采样温度序列 |
| `beam_size` | `5` | Beam search 大小 |
| `best_of` | `5` | 采样候选数 |
| `compression_ratio_threshold` | `2.4` | 压缩比阈值 |
| `log_prob_threshold` | `-1.0` | 对数概率阈值 |
| `no_speech_threshold` | `0.6` | 无语音阈值 |
| `suppress_blank` | `true` | 抑制空白输出 |
| `suppress_numerals` | `false` | 抑制数字符号 |

`vad_options` 子参数（WhisperX 内置 VAD）：

| 键 | 默认值 | 说明 |
|----|--------|------|
| `vad_onset` | `0.500` | VAD 起始阈值 |
| `vad_offset` | `0.363` | VAD 结束阈值 |

`diarize_options` 子参数（说话人分离，两个后端共用同一组）：

| 键 | 默认值 | 说明 |
|----|--------|------|
| `num_speakers` | `null` | 精确说话人数量 |
| `min_speakers` | `null` | 最小说话人数 |
| `max_speakers` | `null` | 最大说话人数 |
| `threshold` | `null` | 聚类阈值 |

**示例文件：**

```json
{
  "model": "large-v3",
  "device": "cuda",
  "compute_type": "float16",
  "language": "yue",
  "batch_size": 16,
  "chunk_size": 30,
  "asr_options": {
    "condition_on_previous_text": false,
    "without_timestamps": true,
    "initial_prompt": null,
    "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "beam_size": 5,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.4
  },
  "vad_options": {
    "vad_onset": 0.500,
    "vad_offset": 0.363
  },
  "diarize_options": {
    "threshold": 0.5
  }
}
```

### Qwen 配置（`transcribe_config_qwen.json`）

| 键 | 类型 | 默认值 | 说明 |
|----|------|--------|------|
| `model` | string | `"Qwen/Qwen3-ASR-1.7B"` | Qwen 模型 ID 或本地路径 |
| `device` | string | `"cuda"` | 计算设备 |
| `language` | string | `"yue"` | 语言代码 |
| `batch_size` | int | `1` | 批处理大小（通常为 1） |
| `enable_timestamps` | bool | `false` | 是否启用时间戳输出 |
| `return_segments` | bool | `true` | 是否将输出规范化为分段对象 |

**示例文件：**

```json
{
  "model": "Qwen/Qwen3-ASR-1.7B",
  "device": "cuda",
  "language": "yue",
  "batch_size": 1,
  "enable_timestamps": false,
  "return_segments": true
}
```

---

## 预期性能

| 配置 | 转录速度 |
|------|----------|
| RTX 4070 | 约 1 倍实时（1 小时音频 ≈ 1 分钟处理） |
| RTX 3060（推荐起点） | 约 1.5 倍实时 |
| RTX 2060（最低门槛） | 约 2-3 倍实时，长录音显存吃紧时需调小 `--batch-size` |
| CPU（无 GPU） | 约 10-20 倍实时（1 小时音频 ≈ 10-20 小时） |

---

## 常见问题

### Q: 运行时报 "Torch not compiled with CUDA enabled"
→ 运行 [安装依赖](#安装依赖) 章节的 force-reinstall 命令重新安装 GPU 版 PyTorch。

### Q: pip 命令无响应
→ 运行失败用 `python -m pip` 替代 `pip`，例如 `python -m pip install whisperx`。

### Q: 显存不足 / Out of Memory
→ 减小 batch-size：`python transcribe.py --batch-size 8` 或 `--batch-size 4`。

### Q: 说话人标签不准确
→ 优先使用 `--num-speakers` 指定确切人数。如果同一说话人被拆成多个标签，调高 `--threshold`（如 0.7）；如果不同说话人被合在一起，调低 `--threshold`（如 0.3）。录音质量越高越好，尽量减少背景噪音，同一录音建议整段处理（不要切割）。

### Q: 如何使用 Qwen3-ASR 替代 WhisperX？
→ 安装可选依赖 `pip install -r requirements-qwen.txt`，然后运行 `python transcribe.py --backend qwen`。首次使用会自动下载模型（约 3-4 GB）。切换回 WhisperX 用 `--backend whisperx`。详见 [切换 ASR 后端](#切换-asr-后端whisperx--qwen3-asr)。

### Q: Qwen 后端支持说话人分离吗？
→ 暂不支持。Qwen 后端是实验性功能，目前仅提供纯文本转录。如需说话人标签，请使用默认的 WhisperX 后端并加上 `--diarize`。

---

## 目录结构

```
auto-transcription/
├── audio/                           ← 录音文件放这里（不随仓库发布，含隐私数据）
├── vad/                             ← Stage 0 输出：VAD 语音分析 + 音频片段
├── raw/                             ← Stage 1 输出：逐字转录
├── normalized/                      ← Stage 2 输出：书面中文转录
├── skills/                          ← Stage 2 转换参考指引
├── vad.py                           ← Stage 0 脚本（VAD 预处理）
├── transcribe.py                    ← Stage 1 脚本
├── vad_config.json                  ← （可选）VAD 参数持久化
├── transcribe_config_whisperx.json  ← （可选）WhisperX 转录参数
├── transcribe_config_qwen.json      ← （可选）Qwen3-ASR 转录参数
├── requirements.txt                 ← 核心依赖
├── requirements-qwen.txt            ← Qwen 后端依赖
├── .env                             ← HuggingFace Token（不随仓库发布）
├── .gitignore
└── LICENSE
```

> **克隆后注意**：`audio/`、`raw/`、`vad/`、`normalized/` 这四个目录**不会随仓库发布**（含受访者隐私，已被 `.gitignore` 排除）。克隆仓库后需自行创建 `audio/` 目录再放入录音：
> ```bash
> mkdir audio
> ```
>
> **未发布的内部文件**：本项目另有 `CLAUDE.md`、`AGENTS.md`（AI 协作规范）与 `DEVLOG.md`（开发日志）三个内部文档，仅供作者与 AI 工具协作使用，不包含在开源仓库中。

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

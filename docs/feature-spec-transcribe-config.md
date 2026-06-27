# Feature Spec: transcribe.py 配置文件持久化

## 背景

当前 `transcribe.py` 的所有参数只能通过命令行传入。用户每次运行都需要重新输入相同的参数，效率低且易出错。此外，WhisperX 底层支持的一些重要参数（如 `chunk_size`、`condition_on_previous_text`、`without_timestamps`）当前完全没有暴露给用户。

本需求旨在为 `transcribe.py` 引入一个 JSON 配置文件，让用户可以：
1. 持久化已有的 CLI 参数
2. 访问 WhisperX 底层的高级参数
3. 像 `vad.py` 的 `vad_config.json` 一样，保存和复用参数

---

## 当前状态

### transcribe.py 现有 CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | Path | None | 输入文件（默认扫描 audio/） |
| `--output` | Path | raw/ | 输出目录 |
| `--model` | str | `"large-v3"` | Whisper 模型 |
| `--language` | str | `"yue"` | 语言代码 |
| `--device` | str | `"cuda"` | 计算设备 |
| `--compute-type` | str | `"float16"` | 量化精度 |
| `--batch-size` | int | `16` | 批处理大小 |
| `--hf-token` | str | env | HuggingFace token |
| `--diarize` | flag | False | 启用说话人识别 |
| `--num-speakers` | int | None | 说话人数量 |
| `--min-speakers` | int | None | 最小说话人数 |
| `--max-speakers` | int | None | 最大说话人数 |
| `--threshold` | float | None | 聚类阈值 |

### WhisperX 底层未暴露的参数

**`whisperx.load_model()` 的 `asr_options` 字典**（控制 Whisper 模型行为）：

| 参数 | WhisperX 默认值 | 说明 |
|------|----------------|------|
| `condition_on_previous_text` | `False` | 是否用前一段输出作为下一段提示 |
| `without_timestamps` | `True` | 是否禁用时间戳输出 |
| `word_timestamps` | `False` | 是否输出词级时间戳 |
| `initial_prompt` | `None` | 给模型的初始提示（可放领域术语） |
| `temperature` | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | 采样温度序列 |
| `compression_ratio_threshold` | `2.4` | 压缩比阈值 |
| `log_prob_threshold` | `-1.0` | 对数概率阈值 |
| `no_speech_threshold` | `0.6` | 无语音阈值 |
| `beam_size` | `5` | Beam search 大小 |
| `best_of` | `5` | 采样候选数 |
| `patience` | `1` | Beam search 耐心因子 |
| `length_penalty` | `1` | 长度惩罚 |
| `repetition_penalty` | `1` | 重复惩罚 |
| `no_repeat_ngram_size` | `0` | 禁止重复的 ngram 大小 |
| `suppress_blank` | `True` | 抑制空白输出 |
| `suppress_tokens` | `[-1]` | 抑制的 token ID |
| `max_new_tokens` | `None` | 每 chunk 最大生成 token 数 |
| `prefix` | `None` | 第一个窗口的前缀 |
| `multilingual` | `False` | 是否对每段进行语言检测 |
| `suppress_numerals` | `False` | 是否抑制数字符号 |
| `hallucination_silence_threshold` | `None` | 幻觉静音阈值 |
| `hotwords` | `None` | 热词/提示短语 |

**`whisperx.load_model()` 的 VAD 参数**（内置 VAD，非项目 Stage 0 的 Silero VAD）：

| 参数 | WhisperX 默认值 | 说明 |
|------|----------------|------|
| `chunk_size` | `30` | VAD 合并后的音频段长度（秒） |
| `vad_onset` | `0.500` | VAD 起始阈值 |
| `vad_offset` | `0.363` | VAD 结束阈值 |

**注意**：`chunk_size` 在 `self.model.transcribe()` 调用时传入，不在 `load_model` 的 `asr_options` 中。VAD 参数（`vad_onset`、`vad_offset`）通过 `vad_options` 传入 `load_model`。

---

## 设计决策（建议）

| 决策 | 建议 | 理由 |
|------|------|------|
| 配置文件格式 | JSON | 与 `vad_config.json` 和项目已有 JSON 输出一致 |
| 配置文件位置 | `transcribe_config.json`（项目根目录） | 与 `vad_config.json` 并列，易于发现 |
| 优先级 | CLI 参数 > 配置文件 > 内置默认值 | 与 `vad.py` 一致 |
| 参数保存范围 | 仅持久化参数，不保存 per-run 参数 | `--input`、`--output` 是每次运行不同的 |
| 参数分组 | 配置文件中按层级分组 | `asr_options`、`vad_options`、`diarize_options` 各自一组 |

### 建议的配置文件 Schema

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
        "word_timestamps": false,
        "initial_prompt": null,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "suppress_blank": true,
        "suppress_tokens": [-1],
        "max_new_tokens": null,
        "prompt_reset_on_temperature": 0.5,
        "prefix": null,
        "multilingual": false,
        "suppress_numerals": false,
        "hallucination_silence_threshold": null,
        "hotwords": null
    },
    "vad_options": {
        "chunk_size": 30,
        "vad_onset": 0.500,
        "vad_offset": 0.363
    },
    "diarize_options": {
        "num_speakers": null,
        "min_speakers": null,
        "max_speakers": null,
        "threshold": null
    }
}
```

**注意**：`vad_options.chunk_size` 和顶层 `chunk_size` 是同一个参数的不同位置。WhisperX 源码中，`chunk_size` 在 `transcribe()` 调用时传入，而 `vad_options` 在 `load_model()` 时传入。实现时需要确认 WhisperX 实际使用哪个值（以 `transcribe()` 的 `chunk_size` 为准）。

### 不保存的参数（per-run）

- `--input` — 每次运行不同
- `--output` — 项目结构决定
- `--diarize` — 每次运行可能不同
- `--hf-token` — 安全敏感，已有 `.env` 管理

---

## 实现要求

### 1. 新增 `--save-config` 和 `--show-config` CLI 参数

与 `vad.py` 一致：

```bash
# 保存当前参数
python transcribe.py --save-config

# 显示生效参数及来源
python transcribe.py --show-config
```

### 2. 参数解析优先级

```
CLI 参数（非 None）> 配置文件 > 内置默认值
```

对于 `asr_options` 和 `vad_options` 等嵌套字典：
- 配置文件中有该字典 → 使用配置文件的值
- 配置文件中没有该字典 → 使用 WhisperX 默认值
- CLI 参数覆盖顶层参数（如 `--batch-size`）

### 3. 参数传递路径

需要修改两个调用点：

**调用点 A — `whisperx.load_model()`（当前行 149）**：
```python
# 当前：
self.model = whisperx.load_model(
    self.model_name,
    self.device,
    compute_type=self.compute_type,
    language=self.language,
)

# 目标：
self.model = whisperx.load_model(
    self.model_name,
    self.device,
    compute_type=self.compute_type,
    language=self.language,
    asr_options=asr_options_dict,    # 新增
    vad_options=vad_options_dict,    # 新增
)
```

**调用点 B — `self.model.transcribe()`（当前行 206）**：
```python
# 当前：
result = self.model.transcribe(
    audio,
    language=self.language,
    batch_size=batch_size,
    print_progress=True,
)

# 目标：
result = self.model.transcribe(
    audio,
    language=self.language,
    batch_size=batch_size,
    chunk_size=chunk_size,           # 新增
    print_progress=True,
)
```

### 4. 代码修改范围

| 文件 | 修改内容 |
|------|----------|
| `transcribe.py` | 新增配置加载/保存/解析逻辑；修改 argparse 默认值；新增 CLI 参数；修改两个调用点 |
| `CLAUDE.md` | 更新 Stage 1 文档，添加配置文件说明 |
| `docs/` | 本文件作为开发参考 |

### 5. 错误处理

- 配置文件不存在 → 静默使用默认值（不报错）
- 配置文件 JSON 格式错误 → 警告 + 使用默认值
- 配置文件中参数类型错误 → 警告 + 跳过该参数
- 配置文件中未知参数 → 忽略（前向兼容）
- `asr_options` 中的无效 key → 忽略（WhisperX 会忽略未知 key）

### 6. 不可破坏的约束

- **不改变** 当前不带配置文件时的任何行为
- **不改变** `format_transcript()` 的逻辑
- **不改变** diarization 流程
- **不改变** 输出文件格式（`[HH:MM]\n[SPEAKER_00]\ntext\n`）
- **不引入** 新的 pip 依赖

---

## 参考实现

`vad.py` 已有完整的配置文件实现（`vad_config.json`），包括：
- `DEFAULTS` 字典
- `CONFIG_PATH` 常量
- `_CONFIG_TYPES` 类型验证
- `load_config()` 函数
- `resolve_params()` 函数
- `save_config()` 函数
- argparse 默认值改为 `None` 的 sentinel 模式
- `--save-config` 和 `--show-config` CLI 参数

`transcribe.py` 的实现应遵循相同的模式和代码风格。

---

## 验收标准

1. `python transcribe.py --show-config` 显示所有参数及来源
2. `python transcribe.py --save-config` 生成 `transcribe_config.json`
3. `python transcribe.py --show-config` 从配置文件读取参数
4. `python transcribe.py --batch-size 8 --show-config` 显示 batch_size 来源为 CLI
5. 配置文件中 `"condition_on_previous_text": true` 生效
6. 配置文件中 `"chunk_size": 15` 生效
7. 不带配置文件运行时行为与修改前完全一致
8. 配置文件中有无效类型时警告并回退到默认值

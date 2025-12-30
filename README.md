# TRPG NER 模型 - HydroRoll 基础 NLP 模型

基于 MiniRBT (hfl/minirbt-h256) 的中文 TRPG（桌上角色扮演游戏）日志命名实体识别系统，支持训练、推理、ONNX 导出和 WebUI 可视化标注平台。

## 功能特性

- **NER 实体识别**: 自动识别 TRPG 日志中的发言者、时间戳、对话、动作、注释等实体
- **灵活训练**: 支持初次训练、增量训练、标签维度扩展/缩减
- **ONNX 导出**: 支持 ONNX 格式导出，实现 CPU 高速推理
- **WebUI 平台**: Label Studio 风格的可视化标注界面 (WIP)
- **数据转换**: 支持 Word-level/Char-level CoNLL 格式互转
- **自动修复**: 智能修复截断的时间戳和发言者名称

## 实体类型

| 标签          | 说明      | 示例                                  |
| ------------- | --------- | ------------------------------------- |
| `speaker`   | 发言者    | "风雨"                                |
| `timestamp` | 时间戳    | "2024-06-08 21:44:59"                 |
| `dialogue`  | 对话内容  | ""呜哇...""                           |
| `action`    | 动作描述  | "剧烈的疼痛从头颅深处一波波地涌出..." |
| `comment`   | 注释/旁白 | "（红木家具上刻着一行小字）"          |

## 安装

### 环境要求

- Python >= 3.12
- CUDA (可选，用于 GPU 加速)

### 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install -r requirements.txt
```

```

```

### 1. 数据准备

将训练数据准备为 CoNLL 格式，支持两种格式：

#### Char-level 格式 (推荐)

```
-DOCSTART- -X- O
风 -X- _ B-speaker
雨 -X- _ I-speaker
  -X- _ O
2 -X- _ B-timestamp
0 -X- _ I-timestamp
...
```

#### Word-level 格式

```
风雨 O O B-speaker
2024-06-08 O O B-timestamp
21:44:59 O O I-timestamp
```

### 2. 初次训练

```bash
# 基础训练（使用默认参数）
uv run main.py --train --conll ./data

# 完整参数训练
uv run main.py --train \
    --conll ./data \
    --model hfl/minirbt-h256 \
    --output ./models/trpg-final \
    --epochs 20 \
    --batch 4
```

#### 参数说明

| 参数         | 说明                 | 默认值                   |
| ------------ | -------------------- | ------------------------ |
| `--train`  | 启用训练模式         | -                        |
| `--conll`  | CoNLL 文件或目录路径 | `./data`               |
| `--model`  | 基础模型名称         | `hfl/minirbt-h256`     |
| `--output` | 模型输出目录         | `./models/trpg-ner-v1` |
| `--epochs` | 训练轮数             | `20`                   |
| `--batch`  | 批处理大小           | `4`                    |
| `--resume` | 恢复检查点路径       | `None`                 |

### 3. 推理测试

```bash
# 单文本测试
uv run main.py --test "风雨 2024-06-08 21:44:59 剧烈的疼痛从头颅深处一波波地涌出..."

# 多文本测试
uv run main.py --test \
    "莎莎 2024-06-08 21:46:26 \"呜哇...\" 下意识去拿法杖" \
    "BOT 2024-06-08 21:50:03 莎莎 的出目是 D10+7=6+7=13"

```

#### 输出格式

```json
{
  "metadata": {
    "speaker": "风雨",
    "timestamp": "2024-06-08 21:44:59"
  },
  "content": [
    {
      "type": "comment",
      "content": "剧烈的疼痛从头颅深处一波波地涌出...",
      "confidence": 0.952
    }
  ]
}
```

## 增量训练

在已有模型基础上继续训练新数据：

```bash
# 继续训练（自动加载最新检查点）
uv run main.py --train \
    --conll ./new_data \
    --output ./models/trpg-final \
    --epochs 5

# 从指定检查点恢复
uv run main.py --train \
    --conll ./new_data \
    --output ./models/trpg-final \
    --resume ./models/trpg-final/checkpoint-200 \
    --epochs 5
```

## 标签维度管理

### 添加新标签训练

在 CoNLL 数据中添加新的标签类型（如 `emotion`），模型会自动适配：

```bash
# 添加新标签后重新训练（使用 ignore_mismatched_sizes）
uv run main.py --train \
    --conll ./data_with_emotion \
    --output ./models/trpg-final \
    --epochs 15
```

### 减少标签维度训练

如果需要减少标签类型，建议从头训练：

```bash
# 从基础模型重新训练（使用缩减后的标签集）
uv run main.py --train \
    --conll ./data_reduced_labels \
    --model hfl/minirbt-h256 \
    --output ./models/trpg-reduced \
    --epochs 20
```

## ONNX 导出

将训练好的模型导出为 ONNX 格式，用于 CPU 推理加速。

> **注意**: ONNX 导出功能会自动使用推理时找到的模型目录。如果模型目录不在默认位置，请先���用 `--output` 指定正确的模型目录。

```bash
# 方式一：从默认模型目录导出（自动查找 models/trpg-final）
uv run main.py --export_onnx \
    --onnx_path ./models/trpg-final/model.onnx

# 方式二：指定模型目录导出
uv run main.py --export_onnx \
    --output ./models/trpg-final \
    --onnx_path ./models/trpg-final/model.onnx

# 方式三：导出到其他路径
uv run main.py --export_onnx \
    --onnx_path ./models/trpg-optimized.onnx
```

### ONNX 模型特点

- 支持 CPU 推理（无需 GPU）
- 模型大小约 50-100 MB
- 推理速度约 10-50 ms/句（取决于硬件）
- 兼容 Windows/Linux/macOS/Raspberry Pi

## ONNX 模型测试

### 基础推理测试

```bash
# 使用 ONNX 模型进行推理
uv run tests/onnx_infer.py "风雨 2024-06-08 21:44:59 剧烈的疼痛..."
```

#### 性能指标

```
Performance Results (n=100):
   Average latency: 25.32 ms
   P95 latency:     31.45 ms
   Max RAM usage:   120.5 MB
   Throughput:      39.5 sentences/sec
```

## 数据转换工具

### CoNLL 转 Dataset

```bash
# 转换单个文件
uv run src/utils/conll_to_dataset.py data.conll \
    --output ./dataset/trpg \
    --format jsonl \
    --validate

# 转换整个目录
uv run src/utils/conll_to_dataset.py ./data \
    --output ./dataset/trpg \
    --format both \
    --validate
```

#### 输出格式

- `--format jsonl`: 输出 JSONL 格式
- `--format dataset`: 输出 HuggingFace Dataset 格式
- `--format both`: 同时输出两种格式

### Word-level 转 Char-level CoNLL

```bash
uv run src/utils/word_conll_to_char_conll.py \
    input_word.conll \
    output_char.conll
```

## 高级功能

### 自定义标签

在 `src/webui/utils.py` 中修改 `DEFAULT_LABELS`：

```python
DEFAULT_LABELS = [
    {"name": "timestamp", "color": "#87CEEB", "type": "text"},
    {"name": "speaker", "color": "#90EE90", "type": "text"},
    {"name": "dialogue", "color": "#FFB6C1", "type": "text"},
    {"name": "action", "color": "#DDA0DD", "type": "text"},
    {"name": "comment", "color": "#FFD700", "type": "text"},
    # 添加新标签
    {"name": "emotion", "color": "#FFA07A", "type": "text"},
]
```

## 系统要求

### 训练环境

- CPU: Intel Core i5 或同等性能
- RAM: 8 GB+
- GPU: NVIDIA GPU（可选，用于加速）
- 存储: 5 GB+

### ONNX 推理环境

- CPU: Intel Core i3 或同等性能
- RAM: 2 GB+
- 存储: 100 MB
- 支持 Raspberry Pi 4 (4GB RAM)

## 开源协议

本项目采用 [AFL-3.0](COPYING) 协议开源。

## 相关链接

- [MiniRBT 模型](https://huggingface.co/hfl/minirbt-h256)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [ONNX Runtime](https://onnxruntime.ai/)

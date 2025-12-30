"""
ONNX 推理模块

提供基于 ONNX 的 TRPG 日志命名实体识别推理功能。
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    import numpy as np
    import onnxruntime as ort
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError(
        "依赖未安装。请运行: pip install onnxruntime transformers numpy"
    ) from e


# 默认模型路径（相对于包安装位置）
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "models" / "trpg-final"
# 远程模型 URL（用于自动下载）
MODEL_URL = "https://github.com/HydroRoll-Team/base-model/releases/download/v0.1.0/model.onnx"


class TRPGParser:
    """
    TRPG 日志解析器（基于 ONNX）

    Args:
        model_path: ONNX 模型路径，默认使用内置模型
        tokenizer_path: tokenizer 配置路径，默认与 model_path 相同
        device: 推理设备，"cpu" 或 "cuda"

    Examples:
        >>> parser = TRPGParser()
        >>> result = parser.parse("风雨 2024-06-08 21:44:59 剧烈的疼痛...")
        >>> print(result['metadata']['speaker'])
        '风雨'
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cpu",
    ):
        # 确定模型路径
        if model_path is None:
            model_path = self._get_default_model_path()

        if tokenizer_path is None:
            tokenizer_path = Path(model_path).parent

        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device

        # 加载模型
        self._load_model()

    def _get_default_model_path(self) -> str:
        """获取默认模型路径"""
        # 1. 尝试相对于项目根目录
        project_root = Path(__file__).parent.parent.parent.parent
        local_model = project_root / "models" / "trpg-final" / "model.onnx"
        if local_model.exists():
            return str(local_model)

        # 2. 尝试用户数据目录
        from pathlib import Path
        user_model_dir = Path.home() / ".cache" / "basemodel" / "models" / "trpg-final"
        user_model = user_model_dir / "model.onnx"
        if user_model.exists():
            return str(user_model)

        # 3. 抛出错误，提示下载
        raise FileNotFoundError(
            f"模型文件未找到。请从 {MODEL_URL} 下载模型到 {user_model_dir}\n"
            f"或运行: python -m basemodel.download_model"
        )

    def _load_model(self):
        """加载 ONNX 模型和 Tokenizer"""
        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.tokenizer_path),
            local_files_only=True,
        )

        # 加载 ONNX 模型
        providers = ["CPUExecutionProvider"]
        if self.device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers,
        )

        # 加载标签映射
        import json
        config_path = self.tokenizer_path / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                self.id2label = {int(k): v for k, v in config.get("id2label", {}).items()}
        else:
            # 默认标签
            self.id2label = {
                0: "O", 1: "B-action", 2: "I-action", 3: "B-comment", 4: "I-comment",
                5: "B-dialogue", 6: "I-dialogue", 7: "B-speaker", 8: "I-speaker",
                9: "B-timestamp", 10: "I-timestamp",
            }

    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析单条 TRPG 日志

        Args:
            text: 待解析的日志文本

        Returns:
            包含 metadata 和 content 的字典
            - metadata: speaker, timestamp
            - content: dialogue, action, comment 列表

        Examples:
            >>> parser = TRPGParser()
            >>> result = parser.parse("风雨 2024-06-08 21:44:59 剧烈的疼痛...")
            >>> result['metadata']['speaker']
            '风雨'
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        # 推理
        outputs = self.session.run(
            ["logits"],
            {
                "input_ids": inputs["input_ids"].astype(np.int64),
                "attention_mask": inputs["attention_mask"].astype(np.int64),
            },
        )

        # 后处理
        logits = outputs[0][0]
        predictions = np.argmax(logits, axis=-1)
        offsets = inputs["offset_mapping"][0]

        # 聚合实体
        entities = self._group_entities(predictions, offsets, logits)

        # 构建结果
        result = {"metadata": {}, "content": []}
        for ent in entities:
            if ent["start"] >= len(text) or ent["end"] > len(text):
                continue

            raw_text = text[ent["start"]: ent["end"]]
            clean_text = self._clean_text(raw_text, ent["type"])

            if not clean_text.strip():
                continue

            if ent["type"] in ["timestamp", "speaker"]:
                result["metadata"][ent["type"]] = clean_text
            elif ent["type"] in ["dialogue", "action", "comment"]:
                result["content"].append({
                    "type": ent["type"],
                    "content": clean_text,
                    "confidence": round(ent["score"], 3),
                })

        return result

    def _group_entities(self, predictions, offsets, logits):
        """将 token 级别的预测聚合为实体"""
        entities = []
        current = None

        for i in range(len(predictions)):
            start, end = offsets[i]
            if start == end:  # special tokens
                continue

            pred_id = int(predictions[i])
            label = self.id2label.get(pred_id, "O")

            if label == "O":
                if current:
                    entities.append(current)
                    current = None
                continue

            tag_type = label[2:] if len(label) > 2 else "O"

            if label.startswith("B-"):
                if current:
                    entities.append(current)
                current = {
                    "type": tag_type,
                    "start": int(start),
                    "end": int(end),
                    "score": float(np.max(logits[i])),
                }
            elif label.startswith("I-") and current and current["type"] == tag_type:
                current["end"] = int(end)
            else:
                if current:
                    entities.append(current)
                current = None

        if current:
            entities.append(current)

        return entities

    def _clean_text(self, text: str, group: str) -> str:
        """清理提取的文本"""
        import re

        text = text.strip()

        # 移除周围符号
        if group == "comment":
            text = re.sub(r"^[（(]+|[）)]+$", "", text)
        elif group == "dialogue":
            text = re.sub(r'^[""''「」『』]+|[""""」』『』]+$', "", text)
        elif group == "action":
            text = re.sub(r"^[*＃]+|[*＃]+$", "", text)

        # 修复时间戳
        if group == "timestamp" and text and text[0].isdigit():
            if len(text) > 2 and text[2] == "-":
                text = "20" + text

        return text

    def parse_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量解析多条日志

        Args:
            texts: 日志文本列表

        Returns:
            解析结果列表
        """
        return [self.parse(text) for text in texts]


# 便捷函数
def parse_line(text: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """
    解析单条日志的便捷函数

    Args:
        text: 日志文本
        model_path: 可选的模型路径

    Returns:
        解析结果字典
    """
    parser = TRPGParser(model_path=model_path)
    return parser.parse(text)


def parse_lines(texts: List[str], model_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    批量解析日志的便捷函数

    Args:
        texts: 日志文本列表
        model_path: 可选的模型路径

    Returns:
        解析结果列表
    """
    parser = TRPGParser(model_path=model_path)
    return parser.parse_batch(texts)


__all__ = ["TRPGParser", "parse_line", "parse_lines"]

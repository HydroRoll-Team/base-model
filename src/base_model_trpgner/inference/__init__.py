"""
ONNX 推理模块

提供基于 ONNX 的 TRPG 日志命名实体识别推理功能。
"""

import os
import json
import shutil
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


# GitHub 仓库信息
REPO_OWNER = "HydroRoll-Team"
REPO_NAME = "base-model"
# 用户数据目录
USER_MODEL_DIR = Path.home() / ".cache" / "base_model_trpgner" / "models" / "trpg-final"


def get_latest_release_url() -> str:
    """
    获取 GitHub 最新 Release 的下载 URL

    Returns:
        最新 Release 的标签名（如 v0.1.0）
    """
    import urllib.request
    import urllib.error

    try:
        # 使用 GitHub API 获取最新 release
        api_url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/releases/latest"
        with urllib.request.urlopen(api_url, timeout=10) as response:
            data = json.load(response)
            return data.get("tag_name", "v0.1.0")
    except (urllib.error.URLError, json.JSONDecodeError, KeyError):
        # 失败时返回默认版本
        return "v0.1.0"


def download_model_files(version: Optional[str] = None, force: bool = False) -> Path:
    """
    从 GitHub Release 下载模型文件

    Args:
        version: Release 版本（如 v0.1.0），None 表示最新版本
        force: 是否强制重新下载（即使文件已存在）

    Returns:
        模型文件保存目录
    """
    import urllib.request
    import urllib.error

    if version is None:
        version = get_latest_release_url()

    model_dir = USER_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    # 检查是否已下载
    marker_file = model_dir / ".version"
    if not force and marker_file.exists():
        with open(marker_file, "r") as f:
            current_version = f.read().strip()
        if current_version == version:
            print(f"模型已存在 (版本: {version})")
            return model_dir

    print(f"正在下载模型 {version}...")

    # 需要下载的文件
    base_url = f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{version}"
    files_to_download = [
        "model.onnx",
        "model.onnx.data",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]

    for filename in files_to_download:
        url = f"{base_url}/{filename}"
        dest_path = model_dir / filename

        try:
            print(f"下载 {filename}...")
            with urllib.request.urlopen(url, timeout=60) as response:
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(response, f)
        except urllib.error.HTTPError as e:
            print(f"下载 {filename} 失败: {e}")
            # model.onnx.data 不是必需的（某些小模型可能没有）
            if filename == "model.onnx.data":
                continue
            else:
                raise RuntimeError(f"无法下载模型文件: {filename}") from e

    # 写入版本标记
    with open(marker_file, "w") as f:
        f.write(version)

    print(f"✅ 模型下载完成: {model_dir}")
    return model_dir


class TRPGParser:
    """
    TRPG 日志解析器（基于 ONNX）

    首次运行时会自动从 GitHub Release 下载最新模型。

    Args:
        model_path: ONNX 模型路径，默认使用自动下载的模型
        tokenizer_path: tokenizer 配置路径，默认与 model_path 相同
        device: 推理设备，"cpu" 或 "cuda"
        auto_download: 是否自动下载模型（默认 True）

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
        auto_download: bool = True,
    ):
        # 确定模型路径
        if model_path is None:
            model_path = self._get_default_model_path(auto_download)

        if tokenizer_path is None:
            tokenizer_path = Path(model_path).parent

        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.device = device

        # 加载模型
        self._load_model()

    def _get_default_model_path(self, auto_download: bool) -> str:
        """获取默认模型路径，必要时自动下载"""
        # 1. 检查本地开发环境
        project_root = Path(__file__).parent.parent.parent.parent
        local_model = project_root / "models" / "trpg-final" / "model.onnx"
        if local_model.exists():
            return str(local_model)

        # 2. 检查用户缓存目录
        user_model = USER_MODEL_DIR / "model.onnx"
        if user_model.exists():
            return str(user_model)

        # 3. 自动下载
        if auto_download:
            print("模型未找到，正在从 GitHub Release 下载...")
            download_model_files()
            return str(user_model)

        # 4. 抛出错误
        raise FileNotFoundError(
            f"模型文件未找到。\n"
            f"请开启自动下载: TRPGParser(auto_download=True)\n"
            f"或手动下载到: {USER_MODEL_DIR}\n"
            f"下载地址: https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/latest"
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


__all__ = ["TRPGParser", "parse_line", "parse_lines", "download_model_files"]

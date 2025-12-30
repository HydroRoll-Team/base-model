#!/usr/bin/env python3
"""
模型下载脚本

自动下载预训练的 ONNX 模型到用户缓存目录。
"""

import os
import sys
from pathlib import Path
import urllib.request


def download_model(
    output_dir: str = None,
    url: str = "https://github.com/HydroRoll-Team/base-model/releases/download/v0.1.0/model.onnx"
):
    """
    下载 ONNX 模型

    Args:
        output_dir: 输出目录，默认为 ~/.cache/basemodel/models/trpg-final/
        url: 模型下载 URL
    """
    if output_dir is None:
        output_dir = Path.home() / ".cache" / "basemodel" / "models" / "trpg-final"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.onnx"

    if output_path.exists():
        print(f"✅ 模型已存在: {output_path}")
        return str(output_path)

    print(f"📥 正在下载模型到 {output_path}...")
    print(f"   URL: {url}")

    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ 模型下载成功!")
        return str(output_path)
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print(f"   请手动从以下地址下载模型:")
        print(f"   {url}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 base-model ONNX 模型")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="模型输出目录（默认: ~/.cache/basemodel/models/trpg-final/）"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://github.com/HydroRoll-Team/base-model/releases/download/v0.1.0/model.onnx",
        help="模型下载 URL"
    )

    args = parser.parse_args()

    download_model(args.output_dir, args.url)

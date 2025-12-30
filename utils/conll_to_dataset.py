"""
TRPG CoNLL 转 Dataset 工具
- 自动检测 word-level / char-level
- 生成 {"text": str, "char_labels": List[str]}
- 支持多文档、跨行实体
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datasets import Dataset

def word_to_char_labels(text: str, word_labels: List[Tuple[str, str]]) -> List[str]:
    """
    将 word-level 标注转为 char-level labels
    Args:
        text: 原始文本 (e.g., "风雨 2024-06-08")
        word_labels: [("风雨", "B-speaker"), ("2024-06-08", "B-timestamp"), ...]
    Returns:
        char_labels: ["B-speaker", "I-speaker", "O", "B-timestamp", ...]
    """
    char_labels = ["O"] * len(text)
    pos = 0
    
    for token, label in word_labels:
        if pos >= len(text):
            break
            
        # 在文本中定位 token（处理空格/换行）
        while pos < len(text) and text[pos] != token[0]:
            pos += 1
        if pos >= len(text):
            break
            
        # 匹配 token
        if text[pos:pos+len(token)] == token:
            # 标注 B/I
            for i, char in enumerate(token):
                idx = pos + i
                if idx < len(char_labels):
                    if i == 0 and label.startswith("B-"):
                        char_labels[idx] = label
                    elif label.startswith("B-"):
                        char_labels[idx] = "I" + label[1:]
                    else:
                        char_labels[idx] = label
            pos += len(token)
        else:
            pos += 1
            
    return char_labels

def parse_conll_to_samples(filepath: str) -> List[Dict[str, Any]]:
    """
    解析 .conll → [{"text": "...", "char_labels": [...]}, ...]
    自动处理：
    - -DOCSTART- 文档边界
    - 空行句子边界
    - word-level → char-level 转换
    """
    samples = []
    current_lines = []  # 存储原始行用于检测粒度
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            current_lines.append(line.rstrip('\n'))
    
    # 检测是否 word-level
    is_word_level = False
    for line in current_lines:
        if line.strip() and not line.startswith("-DOCSTART-"):
            parts = line.split()
            if len(parts) >= 4:
                token = parts[0]
                # 如果 token 长度 >1 且非标点 → 可能是 word-level
                if len(token) > 1 and not re.match(r'^[^\w\s\u4e00-\u9fff]+$', token):
                    is_word_level = True
                    break
    
    if is_word_level:
        print(f"Detected word-level CoNLL, converting to char-level...")
        return _parse_word_conll(filepath)
    else:
        print(f"Detected char-level CoNLL, parsing directly...")
        return _parse_char_conll(filepath)

def _parse_word_conll(filepath: str) -> List[Dict[str, Any]]:
    """解析 word-level .conll（如您提供的原始格式）"""
    samples = []
    current_text_parts = []
    current_word_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if current_text_parts:
                    # 合并文本
                    text = "".join(current_text_parts)
                    # 生成 char-level labels
                    char_labels = word_to_char_labels(text, current_word_labels)
                    samples.append({
                        "text": text,
                        "char_labels": char_labels
                    })
                    current_text_parts = []
                    current_word_labels = []
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
                
            token, label = parts[0], parts[3]
            current_text_parts.append(token)
            current_word_labels.append((token, label))
    
    # 处理末尾
    if current_text_parts:
        text = "".join(current_text_parts)
        char_labels = word_to_char_labels(text, current_word_labels)
        samples.append({
            "text": text,
            "char_labels": char_labels
        })
    
    return samples

def _parse_char_conll(filepath: str) -> List[Dict[str, Any]]:
    """解析 char-level .conll"""
    samples = []
    current_text = []
    current_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith("-DOCSTART-"):
                if current_text:
                    samples.append({
                        "text": "".join(current_text),
                        "char_labels": current_labels.copy()
                    })
                    current_text, current_labels = [], []
                continue
            
            if not line:
                if current_text:
                    samples.append({
                        "text": "".join(current_text),
                        "char_labels": current_labels.copy()
                    })
                    current_text, current_labels = [], []
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue
                
            char = parts[0].replace("\\n", "\n")
            label = parts[3]
            current_text.append(char)
            current_labels.append(label)
    
    # 末尾处理
    if current_text:
        samples.append({
            "text": "".join(current_text),
            "char_labels": current_labels.copy()
        })
    
    return samples

def save_dataset(samples: List[Dict[str, Any]], output_path: str, format: str = "jsonl"):
    """保存数据集"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved {len(samples)} samples to {output_path} (JSONL)")
    
    elif format == "dataset":
        dataset = Dataset.from_list(samples)
        dataset.save_to_disk(output_path)
        print(f"Saved {len(samples)} samples to {output_path} (Hugging Face Dataset)")
    
    elif format == "both":
        jsonl_path = output_path + ".jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"Saved JSONL to {jsonl_path}")
        
        dataset_path = output_path + "_dataset"
        dataset = Dataset.from_list(samples)
        dataset.save_to_disk(dataset_path)
        print(f"Saved Dataset to {dataset_path}")

def validate_samples(samples: List[Dict[str, Any]]) -> bool:
    """验证样本一致性"""
    for i, sample in enumerate(samples):
        if len(sample["text"]) != len(sample["char_labels"]):
            print(f"Sample {i}: text len={len(sample['text'])}, labels len={len(sample['char_labels'])}")
            return False
    print(f"All {len(samples)} samples validated: text & labels length match")
    return True

def main():
    parser = argparse.ArgumentParser(description="Convert CoNLL to TRPG Dataset")
    parser.add_argument("input", type=str, help="Input .conll file or directory")
    parser.add_argument("--output", type=str, default="./dataset/trpg", 
                        help="Output path (without extension)")
    parser.add_argument("--format", choices=["jsonl", "dataset", "both"], 
                        default="jsonl", help="Output format")
    parser.add_argument("--validate", action="store_true", 
                        help="Validate samples after conversion")
    
    args = parser.parse_args()
    
    filepaths = []
    if os.path.isdir(args.input):
        filepaths = sorted(Path(args.input).glob("*.conll"))
    elif args.input.endswith(".conll"):
        filepaths = [Path(args.input)]
    else:
        raise ValueError("Input must be .conll file or directory")
    
    if not filepaths:
        raise FileNotFoundError(f"No .conll files found in {args.input}")
    
    print(f"Processing {len(filepaths)} files: {[f.name for f in filepaths]}")
    
    all_samples = []
    for fp in filepaths:
        print(f"\nProcessing {fp.name}...")
        samples = parse_conll_to_samples(str(fp))
        print(f"  → {len(samples)} samples")
        all_samples.extend(samples)
    
    print(f"\nTotal: {len(all_samples)} samples")
    
    if args.validate:
        if not validate_samples(all_samples):
            exit(1)
    
    save_dataset(all_samples, args.output, args.format)
    
    label_counts = {}
    for sample in all_samples:
        for label in sample["char_labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  {label}: {label_counts[label]}")

if __name__ == "__main__":
    main()
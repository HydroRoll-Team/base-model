"""
工具模块

提供数据加载、CoNLL 格式处理等工具函数。
"""

import os
import glob
from typing import List, Dict, Any, Tuple
from datasets import Dataset
from tqdm.auto import tqdm


def word_to_char_labels(text: str, word_labels: List[Tuple[str, str]]) -> List[str]:
    """Convert word-level labels to char-level"""
    char_labels = ["O"] * len(text)
    pos = 0

    for token, label in word_labels:
        if pos >= len(text):
            break

        while pos < len(text) and text[pos] != token[0]:
            pos += 1
        if pos >= len(text):
            break

        if text[pos: pos + len(token)] == token:
            for i in range(len(token)):
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


def parse_conll_file(filepath: str) -> List[Dict[str, Any]]:
    """Parse .conll → [{"text": str, "char_labels": List[str]}]"""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f.readlines()]

    # 检测 word-level
    is_word_level = any(
        len(line.split()[0]) > 1
        for line in lines
        if line.strip() and not line.startswith("-DOCSTART-") and len(line.split()) >= 4
    )

    samples = []
    if is_word_level:
        current_text_parts = []
        current_word_labels = []

        for line in lines:
            if not line or line.startswith("-DOCSTART-"):
                if current_text_parts:
                    text = "".join(current_text_parts)
                    char_labels = word_to_char_labels(text, current_word_labels)
                    samples.append({"text": text, "char_labels": char_labels})
                    current_text_parts = []
                    current_word_labels = []
                continue

            parts = line.split()
            if len(parts) >= 4:
                token, label = parts[0], parts[3]
                current_text_parts.append(token)
                current_word_labels.append((token, label))

        if current_text_parts:
            text = "".join(current_text_parts)
            char_labels = word_to_char_labels(text, current_word_labels)
            samples.append({"text": text, "char_labels": char_labels})
    else:
        current_text = []
        current_labels = []

        for line in lines:
            if line.startswith("-DOCSTART-"):
                if current_text:
                    samples.append({
                        "text": "".join(current_text),
                        "char_labels": current_labels.copy(),
                    })
                    current_text, current_labels = [], []
                continue

            if not line:
                if current_text:
                    samples.append({
                        "text": "".join(current_text),
                        "char_labels": current_labels.copy(),
                    })
                    current_text, current_labels = [], []
                continue

            parts = line.split()
            if len(parts) >= 4:
                char = parts[0].replace("\\n", "\n")
                label = parts[3]
                current_text.append(char)
                current_labels.append(label)

        if current_text:
            samples.append({
                "text": "".join(current_text),
                "char_labels": current_labels.copy(),
            })

    return samples


def load_conll_dataset(conll_dir_or_files: str) -> Tuple[Dataset, List[str]]:
    """Load .conll files → Dataset"""
    filepaths = []
    if os.path.isdir(conll_dir_or_files):
        filepaths = sorted(glob.glob(os.path.join(conll_dir_or_files, "*.conll")))
    elif conll_dir_or_files.endswith(".conll"):
        filepaths = [conll_dir_or_files]
    else:
        raise ValueError("conll_dir_or_files must be .conll file or directory")

    if not filepaths:
        raise FileNotFoundError(f"No .conll files found in {conll_dir_or_files}")

    print(f"Loading {len(filepaths)} conll files: {filepaths}")

    all_samples = []
    label_set = {"O"}

    for fp in tqdm(filepaths, desc="Parsing .conll"):
        samples = parse_conll_file(fp)
        for s in samples:
            all_samples.append(s)
            label_set.update(s["char_labels"])

    # Build label list
    label_list = ["O"]
    for label in sorted(label_set - {"O"}):
        if label.startswith("B-") or label.startswith("I-"):
            label_list.append(label)
    for label in list(label_list):
        if label.startswith("B-"):
            i_label = "I" + label[1:]
            if i_label not in label_list:
                label_list.append(i_label)
                print(f"⚠️  Added missing {i_label} for {label}")

    print(f"✅ Loaded {len(all_samples)} samples, {len(label_list)} labels: {label_list}")
    return Dataset.from_list(all_samples), label_list


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=128):
    """Tokenize and align labels with tokenizer"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors=None,
    )

    labels = []
    for i, label_seq in enumerate(examples["char_labels"]):
        offsets = tokenized["offset_mapping"][i]
        label_ids = []
        for start, end in offsets:
            if start == end:
                label_ids.append(-100)
            else:
                label_ids.append(label2id[label_seq[start]])
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


__all__ = [
    "word_to_char_labels",
    "parse_conll_file",
    "load_conll_dataset",
    "tokenize_and_align_labels",
]

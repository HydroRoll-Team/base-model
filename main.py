"""
TRPG NER 训练与推理脚本 (Robust Edition)
- 自动探测模型路径（支持 safetensors/pytorch）
- 统一 tokenizer 行为（offset_mapping）
- 智能修复 timestamp/speaker 截断
- 兼容 transformers <4.5
"""

import os
import re
import glob
import sys
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
    logging as hf_logging,
)
from datasets import Dataset
from tqdm.auto import tqdm


hf_logging.set_verbosity_error()


def word_to_char_labels(text: str, word_labels: List[Tuple[str, str]]) -> List[str]:
    """Convert word-level labels to char-level"""
    char_labels = ["O"] * len(text)
    pos = 0

    for token, label in word_labels:
        if pos >= len(text):
            break

        # 定位 token（跳过空格/换行）
        while pos < len(text) and text[pos] != token[0]:
            pos += 1
        if pos >= len(text):
            break

        if text[pos : pos + len(token)] == token:
            for i, _ in enumerate(token):
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
        # Word-level parsing
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
        # Char-level parsing
        current_text = []
        current_labels = []

        for line in lines:
            if line.startswith("-DOCSTART-"):
                if current_text:
                    samples.append(
                        {
                            "text": "".join(current_text),
                            "char_labels": current_labels.copy(),
                        }
                    )
                    current_text, current_labels = [], []
                continue

            if not line:
                if current_text:
                    samples.append(
                        {
                            "text": "".join(current_text),
                            "char_labels": current_labels.copy(),
                        }
                    )
                    current_text, current_labels = [], []
                continue

            parts = line.split()
            if len(parts) >= 4:
                char = parts[0].replace("\\n", "\n")
                label = parts[3]
                current_text.append(char)
                current_labels.append(label)

        if current_text:
            samples.append({"text": "".join(current_text), "char_labels": current_labels.copy()})

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


# ===========================
# 2. 数据预处理（offset_mapping 对齐）
# ===========================


def tokenize_and_align_labels(examples, tokenizer, label2id, max_length=128):
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
            if start == end:  # special tokens
                label_ids.append(-100)
            else:
                label_ids.append(label2id[label_seq[start]])
        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized


# ===========================
# 3. 训练流程
# ===========================


def train_ner_model(
    conll_data: str,
    model_name_or_path: str = "hfl/minirbt-h256",
    output_dir: str = "./models/trpg-ner-v1",
    num_train_epochs: int = 15,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    resume_from_checkpoint: Optional[str] = None,
):
    # Step 1: Load data
    dataset, label_list = load_conll_dataset(conll_data)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # Step 2: Init model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.model_max_length > 1000:
        tokenizer.model_max_length = max_length

    model = AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    # Step 3: Tokenize
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id, max_length),
        batched=True,
        remove_columns=["text", "char_labels"],
    )

    # Step 4: Training args (compatible with old transformers)
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_steps=200,
        save_total_limit=2,
        do_eval=False,
        report_to="none",
        no_cuda=not torch.cuda.is_available(),
        load_best_model_at_end=False,
        push_to_hub=False,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model, tokenizer, label_list


def fix_timestamp(ts: str) -> str:
    """Fix truncated timestamp: '4-06-08' → '2024-06-08'"""
    if not ts:
        return ts
    m = re.match(r"^(\d{1,2})-(\d{2})-(\d{2})(.*)", ts)
    if m:
        year_short, month, day, rest = m.groups()
        if len(year_short) == 1:
            year = "202" + year_short
        elif len(year_short) == 2:
            year = "20" + year_short
        else:
            year = year_short
        return f"{year}-{month}-{day}{rest}"
    return ts


def fix_speaker(spk: str) -> str:
    """Fix truncated speaker name"""
    if not spk:
        return spk
    spk = re.sub(r"[^\w\s\u4e00-\u9fff]+$", "", spk)
    # Ensure at least 2 chars for Chinese names
    if len(spk) == 1 and re.match(r"^[风雷电雨雪火水木金]", spk):
        return spk + "某"
    return spk


class TRPGParser:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.id2label = self.model.config.id2label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        self.nlp = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )

    def parse_line(self, text: str) -> Dict[str, Any]:
        ents = self.nlp(text)
        print(f"[DEBUG] Raw entities: {ents}")
        out = {"metadata": {}, "content": []}

        # Merge adjacent entities
        merged = []
        for e in sorted(ents, key=lambda x: x["start"]):
            if merged and merged[-1]["entity_group"] == e["entity_group"]:
                if e["start"] <= merged[-1]["end"]:
                    merged[-1]["end"] = max(merged[-1]["end"], e["end"])
                    merged[-1]["score"] = min(merged[-1]["score"], e["score"])
                    continue
            merged.append(e)

        for e in merged:
            group = e["entity_group"]
            raw_text = text[e["start"] : e["end"]]
            clean_text = re.sub(r"^[<\[\"“「*（＃\s]+|[>\]\"”」*）\s]+$", "", raw_text).strip()
            if not clean_text:
                clean_text = raw_text

            # Special fixes
            if group == "timestamp":
                clean_text = fix_timestamp(clean_text)
            elif group == "speaker":
                clean_text = fix_speaker(clean_text)

            if group in ["timestamp", "speaker"] and clean_text:
                out["metadata"][group] = clean_text
            elif group in ["dialogue", "action", "comment"] and clean_text:
                out["content"].append(
                    {
                        "type": group,
                        "content": clean_text,
                        "confidence": round(float(e["score"]), 3),
                    }
                )
        return out

    def parse_lines(self, texts: List[str]) -> List[Dict[str, Any]]:
        return [self.parse_line(text) for text in texts]


def find_model_dir(requested_path: str, default_paths: List[str]) -> str:
    """Robustly find model directory"""
    # Check requested path first
    candidates = [requested_path] + default_paths

    for path in candidates:
        if not os.path.isdir(path):
            continue

        # Check required files
        required = ["config.json", "tokenizer.json"]
        has_required = all((Path(path) / f).exists() for f in required)

        # Check model files (safetensors or pytorch)
        model_files = ["model.safetensors", "pytorch_model.bin"]
        has_model = any((Path(path) / f).exists() for f in model_files)

        if has_required and has_model:
            return path

    # If not found, try subdirectories
    for path in candidates:
        if not os.path.isdir(path):
            continue
        for root, dirs, files in os.walk(path):
            for d in dirs:
                full_path = os.path.join(root, d)
                has_required = all(
                    (Path(full_path) / f).exists() for f in ["config.json", "tokenizer.json"]
                )
                has_model = any(
                    (Path(full_path) / f).exists()
                    for f in ["model.safetensors", "pytorch_model.bin"]
                )
                if has_required and has_model:
                    return full_path

    raise FileNotFoundError(
        f"Model not found in any of: {candidates}\n"
        "Required files: config.json, tokenizer.json, and (model.safetensors or pytorch_model.bin)\n"
        "Run training first: --train --conll ./data"
    )


def export_to_onnx(model_dir: str, onnx_path: str, max_length: int = 128):
    """Export model to ONNX format (fixed for local paths)"""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import torch
        from torch.onnx import export as onnx_export
        import os

        print(f"Exporting model from {model_dir} to {onnx_path}...")

        model_dir = os.path.abspath(model_dir)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
        model.eval()

        # Create dummy input
        dummy_text = "莎莎 2024-06-08 21:46:26"
        inputs = tokenizer(
            dummy_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

        # Export to ONNX (使用 opset 18 以兼容现代 PyTorch)
        onnx_export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"},
            },
        )

        # Verify ONNX model
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        print(f"ONNX export successful! Size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
        return True

    except Exception as e:
        print(f"ONNX export failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="TRPG NER: Train & Infer")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--conll", type=str, default="./data", help="Path to .conll files or dir")
    parser.add_argument("--model", type=str, default="hfl/minirbt-h256", help="Base model")
    parser.add_argument(
        "--output", type=str, default="./models/trpg-ner-v1", help="Model output dir"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--test", type=str, nargs="*", help="Test texts")
    parser.add_argument("--export_onnx", action="store_true", help="Export model to ONNX")
    parser.add_argument(
        "--onnx_path",
        type=str,
        default="./models/trpg-final/model.onnx",
        help="ONNX output path",
    )

    args = parser.parse_args()

    if args.train:
        try:
            train_ner_model(
                conll_data=args.conll,
                model_name_or_path=args.model,
                output_dir=args.output,
                num_train_epochs=args.epochs,
                per_device_train_batch_size=args.batch,
                resume_from_checkpoint=args.resume,
            )
            print(f"Training finished. Model saved to {args.output}")
        except Exception as e:
            print(f"Training failed: {e}")
            sys.exit(1)

    # Inference setup
    default_model_paths = [
        args.output,
        "./models/trpg-ner-v1",
        "./models/trpg-ner-v2",
        "./models/trpg-ner-v3",
        "./cvrp-ner-model",
        "./models",
    ]

    try:
        model_dir = find_model_dir(args.output, default_model_paths)
        print(f"Using model from: {model_dir}")
        parser = TRPGParser(model_dir=model_dir)
    except FileNotFoundError as e:
        print(f"{e}")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Run inference
    if args.test:
        for t in args.test:
            print(f"\nInput: {t}")
            result = parser.parse_line(t)
            print("Parse:", result)
    else:
        # Demo
        demo_texts = [
            "风雨 2024-06-08 21:44:59\n剧烈的疼痛从头颅深处一波波地涌出...",
            "莎莎 2024-06-08 21:46:26\n“呜哇...”＃下意识去拿法杖",
            "白麗 霊夢 2024-06-08 21:50:03\n莎莎 的出目是 D10+7=6+7=13",
        ]
        print("\nDemo inference (using model from", model_dir, "):")
        for i, t in enumerate(demo_texts, 1):
            print(f"\nDemo {i}: {t[:50]}...")
            result = parser.parse_line(t)
            meta = result["metadata"]
            content = result["content"]
            print(f"   Speaker: {meta.get('speaker', 'N/A')}")
            print(f"   Timestamp: {meta.get('timestamp', 'N/A')}")
            if content:
                first = content[0]
                print(
                    f"   Content: {first['type']}='{first['content'][:40]}{'...' if len(first['content'])>40 else ''}' (conf={first['confidence']})"
                )

    # ONNX export
    if args.export_onnx:
        # 智能确定模型目录：优先使用推理时找到的目录
        onnx_model_dir = model_dir
        success = export_to_onnx(
            model_dir=onnx_model_dir,
            onnx_path=args.onnx_path,
            max_length=128,
        )
        if success:
            print(f"ONNX model saved to {args.onnx_path}")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()

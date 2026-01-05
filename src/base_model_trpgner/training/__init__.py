"""
训练模块

提供 TRPG NER 模型训练功能。

注意: 使用此模块需要安装训练依赖:
    pip install base-model-trpgner[train]
"""

import os
from typing import Optional, List
from pathlib import Path


def train_ner_model(
    conll_data: str,
    model_name_or_path: str = "hfl/minirbt-h256",
    output_dir: str = "./models/trpg-ner-v1",
    num_train_epochs: int = 20,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """
    训练 NER 模型

    Args:
        conll_data: CoNLL 格式数据文件或目录
        model_name_or_path: 基础模型名称或路径
        output_dir: 模型输出目录
        num_train_epochs: 训练轮数
        per_device_train_batch_size: 批处理大小
        learning_rate: 学习率
        max_length: 最大序列长度
        resume_from_checkpoint: 恢复检查点路径

    Examples:
        >>> from base_model_trpgner.training import train_ner_model
        >>> train_ner_model(
        ...     conll_data="./data",
        ...     output_dir="./my_model",
        ...     num_train_epochs=10
        ... )
    """
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForTokenClassification,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        raise ImportError("训练依赖未安装。请运行: pip install base-model-trpgner[train]") from e

    # 导入数据处理函数
    from base_model_trpgner.utils import load_conll_dataset, tokenize_and_align_labels

    print("Starting training...")

    # 加载数据
    dataset, label_list = load_conll_dataset(conll_data)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    # 初始化模型
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

    # Tokenize
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer, label2id, max_length),
        batched=True,
        remove_columns=["text", "char_labels"],
    )

    # 训练参数
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

    # 开始训练
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型
    print("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Training finished. Model saved to {output_dir}")


def export_to_onnx(
    model_dir: str,
    onnx_path: str,
    max_length: int = 128,
) -> bool:
    """
    将训练好的模型导出为 ONNX 格式

    Args:
        model_dir: 模型目录
        onnx_path: ONNX 输出路径
        max_length: 最大序列长度

    Returns:
        是否成功
    """
    try:
        from torch.onnx import export as onnx_export
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        import onnx
    except ImportError as e:
        raise ImportError("ONNX 导出依赖未安装。请运行: pip install onnx") from e

    print(f"Exporting model from {model_dir} to {onnx_path}...")

    model_dir = os.path.abspath(model_dir)
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(model_dir, local_files_only=True)
    model.eval()

    dummy_text = "莎莎 2024-06-08 21:46:26"
    inputs = tokenizer(
        dummy_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    # 导出 ONNX
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

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"ONNX export successful! Size: {size_mb:.2f} MB")
    return True


__all__ = ["train_ner_model", "export_to_onnx"]

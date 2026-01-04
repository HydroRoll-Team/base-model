#!/usr/bin/env python3
"""
使用 LLM 对游戏日志进行自动标注（支持高并发）
标注格式：speaker、timestamp、dialogue、action、comment
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List

from dotenv import load_dotenv
from ollama import chat

load_dotenv()


def get_annotation_prompt(text: str) -> str:
    """
    构造 LLM 标注 prompt（只返回类型和文本，我们自己计算位置）
    """
    return f"""你是一个专业的文本标注助手。请对以下 TRPG 游戏日志进行标注，标注格式为 JSON。

## 标签类型及规则

1. **speaker**: 说话人/玩家名字（位于文本开头，后跟空格和时间戳）
2. **timestamp**: 时间戳（格式如：2024-06-08 21:44:59）
3. **dialogue**: 角色对话（用引号包裹的说话内容，如 ""...""）
4. **action**: 动作/指令（以 . 开头的骰子指令，如 .rd10+7、.ww12a9+1）
5. **comment**: 其他描述性内容（角色扮演描述、系统消息、动作描述等）

## 标注示例

### 示例 1：
文本：`风雨 2024-06-08 21:44:59\\n剧烈的疼痛从头颅深处一波波地涌出，仿佛每一次脉搏的跳动都在击打你的头骨。`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "风雨"}},
    {{"type": "timestamp", "text": "2024-06-08 21:44:59"}},
    {{"type": "comment", "text": "剧烈的疼痛从头颅深处一波波地涌出，仿佛每一次脉搏的跳动都在击打你的头骨。"}}
  ]
}}
```

### 示例 2：
文本：`莎莎 2024-06-08 21:46:26\\n"呜哇..."＃下意识去拿法杖，但启动施法起手后大脑里一片空白...`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "莎莎"}},
    {{"type": "timestamp", "text": "2024-06-08 21:46:26"}},
    {{"type": "dialogue", "text": ""呜哇...""}},
    {{"type": "comment", "text": "＃下意识去拿法杖，但启动施法起手后大脑里一片空白..."}}
  ]
}}
```

### 示例 3：
文本：`莎莎 2024-06-08 21:49:51\\n.rd10+7`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "莎莎"}},
    {{"type": "timestamp", "text": "2024-06-08 21:49:51"}},
    {{"type": "action", "text": ".rd10+7"}}
  ]
}}
```

### 示例 4：
文本：`白麗 霊夢 2024-06-08 21:49:51\\n莎莎 的出目是\\nD10+7=6+7=13`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "白麗 霊夢"}},
    {{"type": "timestamp", "text": "2024-06-08 21:49:51"}},
    {{"type": "comment", "text": "莎莎 的出目是\\nD10+7=6+7=13"}}
  ]
}}
```

## 注意事项

- 只返回标注的类型（type）和文本内容（text），不需要返回位置信息
- 确保标注的文本内容与原文本完全一致
- 只返回 JSON，不要添加任何其他解释性文字
- 如果文本中不包含某种标签类型，就不要包含该标签

## 待标注文本

{text}

## 请返回标注结果（只返回 JSON，不要其他内容）："""


def call_llm_api(prompt: str, index: int, total: int) -> Dict[str, Any]:
    """
    调用 Ollama 本地 LLM（带重试机制）
    """
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    messages = [
        {
            "role": "system",
            "content": "你是一个专业的文本标注助手，严格按照 JSON 格式返回标注结果，不要添加任何其他内容。",
        },
        {"role": "user", "content": prompt},
    ]

    max_retries = 3
    base_delay = 1  # 秒

    for attempt in range(max_retries):
        try:
            response = chat(
                model=ollama_model,
                messages=messages,
                think=False,
                stream=False,
            )

            content = response.message.content

            if not content:
                print(f"[{index}/{total}] API 返回空内容")
                if attempt < max_retries - 1:
                    time.sleep(base_delay)
                    continue
                return {"annotations": []}

            # 尝试解析 JSON
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            print(f"[{index}/{total}] API 调用成功")
            return json.loads(content)
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                print(f"[{index}/{total}] JSON 解析失败: {e}，重试中...")
                time.sleep(base_delay)
            else:
                print(f"[{index}/{total}] JSON 解析失败，达到最大重试次数")
                return {"annotations": []}
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[{index}/{total}] API 调用失败: {e}，重试中...")
                time.sleep(base_delay * (2**attempt))
            else:
                print(f"[{index}/{total}] API 调用失败: {e}，达到最大重试次数")
                return {"annotations": []}

    return {"annotations": []}


def calculate_annotation_positions(original_text: str, llm_annotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据 LLM 返回的标注文本计算在原文本中的位置

    Args:
        original_text: 原始文本
        llm_annotations: LLM 返回的标注列表，每个包含 type 和 text

    Returns:
        包含 start, end, type, text 的完整标注列表
    """
    result_annotations = []
    current_pos = 0

    for ann in llm_annotations:
        ann_type = ann.get("type")
        ann_text = ann.get("text", "")

        if not ann_type or not ann_text:
            continue

        # 在原文本中查找标注文本的位置
        # 从当前位置开始查找，避免重复匹配
        pos = original_text.find(ann_text, current_pos)

        if pos == -1:
            # 如果没找到，尝试从头查找（处理非顺序标注）
            pos = original_text.find(ann_text)

        if pos != -1:
            result_annotations.append({
                "type": ann_type,
                "start": pos,
                "end": pos + len(ann_text),
                "text": ann_text
            })
            # 更新当前位置为标注结束位置
            current_pos = pos + len(ann_text)

    return result_annotations


def convert_to_label_studio_format(
    task_id: int, text: str, llm_annotations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    将 LLM 标注结果转换为 Label Studio 格式
    """
    import uuid

    annotation_id = str(uuid.uuid4())

    # 计算标注���置
    annotations = calculate_annotation_positions(text, llm_annotations)

    # 构建 result 数组
    results = []
    for ann in annotations:
        if ann.get("type") is None or ann.get("text") is None:
            continue

        result_id = str(uuid.uuid4())
        results.append(
            {
                "value": {
                    "start": ann["start"],
                    "end": ann["end"],
                    "text": ann["text"],
                    "labels": [ann["type"]],
                },
                "id": result_id,
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "origin": "manual",
            }
        )

    # 构建完整的 Label Studio 任务格式
    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": task_id,
        "annotations": [
            {
                "id": task_id,
                "completed_by": 1,
                "result": results,
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": now,
                "updated_at": now,
                "draft_created_at": now,
                "lead_time": 0.0,
                "prediction": {},
                "result_count": len(results),
                "unique_id": annotation_id,
                "import_id": None,
                "last_action": None,
                "bulk_created": False,
                "task": task_id,
                "project": 2,
                "updated_by": 1,
                "parent_prediction": None,
                "parent_annotation": None,
                "last_created_by": None,
            }
        ],
        "file_upload": "llm-auto-annotated.json",
        "drafts": [],
        "predictions": [],
        "data": {"text": text},
        "meta": {},
        "created_at": now,
        "updated_at": now,
        "allow_skip": True,
        "inner_id": task_id,
        "total_annotations": 1,
        "cancelled_annotations": 0,
        "total_predictions": 0,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "last_comment_updated_at": None,
        "project": 2,
        "updated_by": 1,
        "comment_authors": [],
    }


def process_logs(input_path: str, output_path: str, concurrency: int = 5, batch_size: int = 50):
    """
    处理日志文件并进行自动标注（支持高并发）

    Args:
        input_path: 输入的 processed_logs.json 文件路径
        output_path: 输出的标注结果文件路径
        concurrency: 并发线程数
        batch_size: 批处理保存大小
    """
    # 读取输入文件
    print(f"读取输入文件: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    total = len(logs)
    print(f"总共 {total} 条日志需要标注")
    print(f"并发数: {concurrency}")

    results = []
    # 用于保持顺序的字典
    results_dict = {}

    def process_single_log(index: int, log_entry: Dict[str, Any]):
        text = log_entry.get("text", "")
        if not text:
            print(f"[{index}/{total}] 跳过空文本")
            return None

        print(f"\n[{index}/{total}] 处理文本: {text[:50]}...")

        # 构造 prompt
        prompt = get_annotation_prompt(text)

        # 调用 LLM API
        llm_result = call_llm_api(prompt, index, total)

        # 转换为 Label Studio 格式
        return convert_to_label_studio_format(
            task_id=index, text=text, llm_annotations=llm_result.get("annotations", [])
        )

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_single_log, i, log_entry): i
            for i, log_entry in enumerate(logs, 1)
        }

        # 收集完成的任务
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                if result is not None:
                    results_dict[index] = result
            except Exception as e:
                print(f"[{index}/{total}] 处理失败: {e}")

    # 按顺序整理结果
    for index in sorted(results_dict.keys()):
        results.append(results_dict[index])

        # 每处理 batch_size 条保存一次
        if index % batch_size == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n已保存 {index} 条结果到 {output_path}")

    # 保存最终结果
    print(f"\n保存最终结果到: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成！共标注 {len(results)} 条日志")


def main():
    """
    主函数
    """
    # 使用默认路径
    input_path = "dataset/processed_logs/processed_logs.json"
    output_path = f"dataset/llm_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入文件不存在: {input_path}")
        return

    # 开始处理
    process_logs(input_path, output_path, concurrency=5, batch_size=50)


if __name__ == "__main__":
    main()

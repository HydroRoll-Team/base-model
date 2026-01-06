"""
使用 LLM 对游戏日志进行自动标注（支持高并发）
标注格式：speaker、timestamp、dialogue、action、comment
"""

import json
import os
import re
import time
import threading
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
    return f"""你是一个专业的 TRPG 游戏日志标注助手。请根据**语义**对以下文本进行标注，标注格式为 JSON。

## 标签类型及语义判断规则

1. **speaker**：说话人/玩家名字
   - 通常位于文本开头
   - 后面紧跟空格和时间戳
   - 格式特征：`名字(QQ号) 时间戳`

2. **timestamp**：时间戳
   - 时间格式：`YYYY-MM-DD HH:MM:SS`
   - 时间格式也有可能是其他的变体，如 `YYYY/MM/DD HH:MM` 等
   - 紧跟在 speaker 之后

3. **action**：骰子/游戏指令或者人物动作词描述
   - 以点号 `.` 或者 `/` 或者 `!` 或者 `。` 开头的指令
   - 例如：`.rd10+7`、`。ww12a9+1`、`!roll` 等
   - 也可以是描述角色动作的简短词语
   - 例如：`站起身来`、`掏出法杖` 等

4. **dialogue**：角色对话/说话内容
   - **判断依据：是否为角色口中说出的话**
   - 可能有引号包裹（\"\"\"\"、\"\"\"\"或\"\"\"\"）
   - **也可能没有引号**，需要根据语义判断
   - 间接引语、心理独白如果明显是"说话"性质也应标注
   - 关键：这段文字是角色"说"出来的，而不是"做"的动作描述

5. **comment**：其他所有内容
   - 场景描写
   - 系统消息（如骰子结果）
   - 心理活动（非说话形式）
   - GM 描述
   - 其他不属于上述类型的内容

## 标注原则

- **按文本出现顺序标注**
- **根据语义判断类型**，不要仅依赖格式特征
- **不要遗漏任何有效内容**，文本的有效部分都必须被标注，即使是系统消息或动作描述，不过你需要自己判断哪些是 action 动作哪些是 comment 描述或者旁白，说话人后面的QQ号和括号不需要标注
- **保持文本原样**，标注的 text 必须与原文完全一致
- **注意标点符号**，标点符号是文本的一部分，必须包含在标注的 text 中

## 标注示例

### 示例 1：纯动作描述
文本：`风雨(1231287491) 2024-06-08 21:44:59\\n剧烈的疼痛从头颅深处一波波地涌出，仿佛每一次脉搏的跳动都在击打你的头骨。`
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

### 示例 2：有引号的对话 + 动作
文本：`莎莎(123125124) 2024-06-08 21:46:26\\n"呜哇..."＃下意识去拿法杖，但启动施法起手后大脑里一片空白...`
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

### 示例 3：无引号的对话（语义判断）
文本：`风雨(1231287491) 2024-06-08 21:50:15\\n我不行了，��带我离开这里`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "风雨"}},
    {{"type": "timestamp", "text": "2024-06-08 21:50:15"}},
    {{"type": "dialogue", "text": "我不行了，快带我离开这里"}}
  ]
}}
```

### 示例 4：对话 + 动作混合
文本：`白麗 霊夢(12345678921) 2024-06-08 21:51:00\\n好的，我明白了。他点点头，转身离开了房间。`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "白麗 霊夢"}},
    {{"type": "timestamp", "text": "2024-06-08 21:51:00"}},
    {{"type": "dialogue", "text": "好的，我明白了。"}},
    {{"type": "comment", "text": "他点点头，转身离开了房间。"}}
  ]
}}
```

### 示例 5：纯动作指令
文本：`莎莎(1251124512) 2024-06-08 21:49:51\\n.rd10+7`
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

### 示例 6：系统消息
文本：`白麗 霊夢(12345678921) 2024-06-08 21:49:51\\n莎莎 的出目是\\nD10+7=6+7=13`
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

### 示例 7：多段对话混合描述
文本：`白麗 霊夢(12345678921) 2024-06-08 21:52:00\\n等等，这是什么？他指着地上的物品，疑惑地问道。这是...魔法道具吗？`
标注：
```json
{{
  "annotations": [
    {{"type": "speaker", "text": "白麗 霊夢"}},
    {{"type": "timestamp", "text": "2024-06-08 21:52:00"}},
    {{"type": "dialogue", "text": "等等，这是什么？"}},
    {{"type": "comment", "text": "他指着地上的物品，疑惑地问道。"}},
    {{"type": "dialogue", "text": "这是...魔法道具吗？"}}
  ]
}}
```

## 重要提示

- **dialogue 的判断核心是"这是角色说的话吗"**，而不是"有没有引号"
- 如果文本是角色直接说出的内容，即使没有引号也应标注为 dialogue
- 如果文本是场景、心理描写等非说话内容，应标注为 comment
- 如果文本是指令或者任务的动作词，应标注为 action
- 请严格按照上述 JSON 格式返回标注结果
- 只返回 JSON，不要添加任何其他解释性文字

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


def calculate_annotation_positions(
    original_text: str, llm_annotations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
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
            result_annotations.append(
                {"type": ann_type, "start": pos, "end": pos + len(ann_text), "text": ann_text}
            )
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

    # 计算标注位置
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


def process_logs(input_path: str, output_path: str, concurrency: int = 10, batch_size: int = 50):
    """
    处理日志文件并进行自动标注（支持高并发）

    Args:
        input_path: 输入的 processed_logs.json 文件路径
        output_path: 输出的标注结果文件路径
        concurrency: 并发线程数
        batch_size: 批处理保存大小（已弃用，保留用于兼容性）
    """
    # 读取输入文件
    print(f"读取输入文件: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    total = len(logs)
    print(f"总共 {total} 条日志需要标注")
    print(f"并发数: {concurrency}")

    # 用于保持顺序的字典和线程锁
    results_dict = {}
    results_lock = threading.Lock()
    completed_count = [0]  # 使用列表以便在闭包中修改

    def write_results_to_file():
        """将当前已完成的结果按顺序写入文件"""
        with results_lock:
            sorted_results = [results_dict[i] for i in sorted(results_dict.keys())]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sorted_results, f, ensure_ascii=False, indent=2)

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
                    with results_lock:
                        results_dict[index] = result
                        completed_count[0] += 1
                        current_completed = completed_count[0]

                    # 立即写入文件
                    write_results_to_file()
                    print(f"[{index}/{total}] 已保存，完成进度: {current_completed}/{total}")
            except Exception as e:
                print(f"[{index}/{total}] 处理失败: {e}")

    print(f"\n完成！共标注 {len(results_dict)} 条日志")
    print(f"结果已保存到: {output_path}")


def post_process_labels(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    对标注结果进行后处理，去除非 speaker 和 timestamp 标签中的特殊符号并重新计算位置

    Args:
        task_data: 单个任务的数据

    Returns:
        后处理后的任务数据
    """
    original_text = task_data["data"]["text"]
    result_list = task_data["annotations"][0]["result"]

    # 定义需要去除的正则表达式模式
    pattern = r"[#“”「」『』【】]"

    for result_item in result_list:
        labels = result_item["value"]["labels"]
        text = result_item["value"]["text"]
        print(f"Processing label: {labels} with text: {text}")

        # 只处理非 speaker 和 timestamp 的标签
        if "speaker" not in labels and "timestamp" not in labels:
            # 记录原始信息
            original_start = result_item["value"]["start"]
            original_end = result_item["value"]["end"]
            print(f"Original positions: start={original_start}, end={original_end}")

            processed_text = re.sub(pattern, "", text)
            print(f"Processed text: {processed_text}")

            # 如果文本发生变化
            if processed_text != text:
                # 在原始文本中查找处理后的文本位置
                # 首先从原始位置附近开始查找
                search_start = max(0, original_start - 10)
                search_end = min(len(original_text), original_end + 10)

                # 在搜索范围内查找处理后的文本
                pos = original_text.find(processed_text, search_start, search_end)

                # 如果在附近没找到，在整个文本中查找
                if pos == -1:
                    pos = original_text.find(processed_text)

                # 更新结果
                if pos != -1:
                    result_item["value"]["text"] = processed_text
                    result_item["value"]["start"] = pos
                    result_item["value"]["end"] = pos + len(processed_text)
                else:
                    print(
                        f"警告：无法找到处理后的文本位置，原始文本: '{text}', 处理后: '{processed_text}'"
                    )

    return task_data


def process_input_file(input_path: str):
    """
    Process input file and perform post-processing on all annotation results

    Args:
        input_path: Path to the input annotation result file
    """
    print(f"Reading input file: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total {len(data)} tasks to process")

        # Process each task
        for i, task_data in enumerate(data):
            print(f"\n[{i+1}/{len(data)}] Processing task ID: {task_data.get('id', 'N/A')}")

            # Perform post-processing
            processed_task = post_process_labels(task_data)

            # Update original data
            data[i] = processed_task

            print(f"Completed, processed text length: {len(processed_task['data']['text'])}")

        # Save processed file
        output_path = input_path.replace(".json", "_post_processed.json")
        output_path = (
            f"dataset/llm_annotated_post_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\nPost-processing completed! Results saved to: {output_path}")

    except Exception as e:
        print(f"Error processing file: {e}")


def main():
    """Main function"""
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        # If file path is provided, execute post-processing
        input_file = sys.argv[1]
        print(f"File path detected, executing post-processing mode...")
        process_input_file(input_file)
    else:
        # Use default path, execute normal annotation process
        input_path = input("请输入待处理的日志文件路径（processed_logs.json）：").strip()
        output_path = f"dataset/llm_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")

        # Check if input file exists
        if not os.path.exists(input_path):
            print(f"Error: Input file does not exist: {input_path}")
            return

        # Start processing
        process_logs(input_path, output_path, concurrency=10, batch_size=50)


if __name__ == "__main__":
    main()

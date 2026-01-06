import glob
import json
import os
import random
import re


def process_g_files(directory="."):
    # 获取绝对路径
    abs_directory = os.path.abspath(directory)
    # 在指定目录中查找以g开头的txt文件
    pattern = os.path.join(abs_directory, "g*.txt")
    files = glob.glob(pattern)

    if not files:
        print("未找到以'g'开头的txt文件")
        return

    print(f"在目录 {abs_directory} 中找到 {len(files)} 个文件: {', '.join(os.path.basename(f) for f in files)}")

    all_entries = []

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                current_paragraph = []

                for line in file:
                    stripped_line = line.rstrip("\n")

                    if stripped_line.strip():
                        current_paragraph.append(stripped_line)
                    else:
                        if current_paragraph:
                            paragraph_text = "\n".join(current_paragraph)
                            all_entries.append(paragraph_text)
                            current_paragraph = []

                if current_paragraph:
                    paragraph_text = "\n".join(current_paragraph)
                    all_entries.append(paragraph_text)

            print(f"处理文件 {file_path} 完成")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    # 生成三个版本的日志文件

    # 版本1：仅去除数字标记 (原有版本)
    version1_entries = []
    for text in all_entries:
        cleaned_text = re.sub(r"\(\d+\)", "", text)
        version1_entries.append({"text": cleaned_text})

    with open("processed_logs.json", "w", encoding="utf-8") as f:
        json.dump(version1_entries, f, ensure_ascii=False, indent=2)

    print(f"已生成 processed_logs.json (仅去除数字标记)")

    # 版本2：保留所有敏感信息，仅去除数字标记
    version2_entries = []
    for text in all_entries:
        version2_entries.append({"text": text})

    with open("processed_logs_sensitive.json", "w", encoding="utf-8") as f:
        json.dump(version2_entries, f, ensure_ascii=False, indent=2)

    print(f"已生成 processed_logs_sensitive.json (保留敏感信息)")

    # 版本3：去除数字标记和所有特定标点符号
    punctuation_pattern = r'[#""''""“”‘’「」『』【】]'
    version3_entries = []
    for text in all_entries:
        # 先去除数字标记
        cleaned_text = re.sub(r"\(\d+\)", "", text)
        # 再去除特定标点符号
        cleaned_text = re.sub(punctuation_pattern, "", cleaned_text)
        version3_entries.append({"text": cleaned_text})

    with open("processed_logs_clean.json", "w", encoding="utf-8") as f:
        json.dump(version3_entries, f, ensure_ascii=False, indent=2)

    print(f"已生成 processed_logs_clean.json (去除数字标记和标点符号)")

    # 为每个版本生成20%的测试集
    random.seed(42)  # 设置随机种子确保可重复性

    # 版本1的测试集 (processed_logs.json -> processed_logs_test.json)
    test_size1 = max(1, int(len(version1_entries) * 0.2))
    test_set1 = random.sample(version1_entries, test_size1)
    train_set1 = [entry for entry in version1_entries if entry not in test_set1]

    with open("processed_logs_train.json", "w", encoding="utf-8") as f:
        json.dump(train_set1, f, ensure_ascii=False, indent=2)
    with open("processed_logs_test.json", "w", encoding="utf-8") as f:
        json.dump(test_set1, f, ensure_ascii=False, indent=2)

    print(f"已生成 processed_logs_train.json ({len(train_set1)} 条)")
    print(f"已生成 processed_logs_test.json ({len(test_set1)} 条, 20%)")

    # 版本2的测试集 (processed_logs_sensitive.json -> processed_logs_sensitive_test.json)
    test_size2 = max(1, int(len(version2_entries) * 0.2))
    test_set2 = random.sample(version2_entries, test_size2)
    train_set2 = [entry for entry in version2_entries if entry not in test_set2]

    with open("processed_logs_sensitive_train.json", "w", encoding="utf-8") as f:
        json.dump(train_set2, f, ensure_ascii=False, indent=2)
    with open("processed_logs_sensitive_test.json", "w", encoding="utf-8") as f:
        json.dump(test_set2, f, ensure_ascii=False, indent=2)

    print(f"已生成 processed_logs_sensitive_train.json ({len(train_set2)} 条)")
    print(f"已生成 processed_logs_sensitive_test.json ({len(test_set2)} 条, 20%)")

    # 版本3的测试集 (processed_logs_clean.json -> processed_logs_clean_test.json)
    test_size3 = max(1, int(len(version3_entries) * 0.2))
    test_set3 = random.sample(version3_entries, test_size3)
    train_set3 = [entry for entry in version3_entries if entry not in test_set3]

    with open("processed_logs_clean_train.json", "w", encoding="utf-8") as f:
        json.dump(train_set3, f, ensure_ascii=False, indent=2)
    with open("processed_logs_clean_test.json", "w", encoding="utf-8") as f:
        json.dump(test_set3, f, ensure_ascii=False, indent=2)

    print(f"已生成 processed_logs_clean_train.json ({len(train_set3)} 条)")
    print(f"已生成 processed_logs_clean_test.json ({len(test_set3)} 条, 20%)")

    print(f"\n处理完成! 共处理 {len(all_entries)} 个段落")
    print(f"生成了3个版本的数据集，每个版本都有对应的训练集和测试集")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        print(f"处理目录: {directory}")
        process_g_files(directory)
    else:
        print("处理当前目录")
        process_g_files(".")

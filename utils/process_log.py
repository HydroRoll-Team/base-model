import glob
import json
import re


def process_g_files():
    files = glob.glob("g*.txt")

    if not files:
        print("未找到以'g'开头的txt文件")
        return

    print(f"找到 {len(files)} 个文件: {', '.join(files)}")

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
                            cleaned_text = re.sub(r"\(\d+\)", "", paragraph_text)
                            all_entries.append({"text": cleaned_text})
                            current_paragraph = []

                if current_paragraph:
                    paragraph_text = "\n".join(current_paragraph)
                    cleaned_text = re.sub(r"\(\d+\)", "", paragraph_text)
                    all_entries.append({"text": cleaned_text})

            print(f"处理文件 {file_path} 完成")

        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")

    output_file = "processed_logs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成! 共处理 {len(all_entries)} 个段落")
    print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    process_g_files()

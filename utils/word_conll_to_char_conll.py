def word_conll_to_char_conll(word_conll_lines: list[str]) -> list[str]:
    char_lines = []
    in_new_sample = True  # 下一行是否应视为新样本开始

    for line in word_conll_lines:
        stripped = line.strip()
        if not stripped:
            # 空行 → 标记下一句为新样本
            in_new_sample = True
            char_lines.append("")
            continue

        parts = stripped.split()
        if len(parts) < 4:
            char_lines.append(line.rstrip())
            continue

        token, label = parts[0], parts[3]

        # 检测新发言：B-speaker 出现 → 新样本
        if label == "B-speaker" and in_new_sample:
            char_lines.append("-DOCSTART- -X- O")
            in_new_sample = False

        # 转换 token → char labels（同前）
        if label == "O":
            for c in token:
                char_lines.append(f"{c} -X- _ O")
        else:
            bio_prefix = label[:2]
            tag = label[2:]
            for i, c in enumerate(token):
                char_label = f"B-{tag}" if (bio_prefix == "B-" and i == 0) else f"I-{tag}"
                char_lines.append(f"{c} -X- _ {char_label}")

    return char_lines

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python word_conll_to_char_conll.py <input_word.conll> <output_char.conll>")
        sys.exit(1)

    input_fp = sys.argv[1]
    output_fp = sys.argv[2]

    with open(input_fp, "r", encoding="utf-8") as f:
        word_conll_lines = f.readlines()

    char_conll_lines = word_conll_to_char_conll(word_conll_lines)

    with open(output_fp, "w", encoding="utf-8") as f:
        f.write("\n".join(char_conll_lines) + "\n")

    print(f"Converted {input_fp} to character-level CoNLL format at {output_fp}")


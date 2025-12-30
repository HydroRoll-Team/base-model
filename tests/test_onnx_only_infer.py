"""
Minimal ONNX-only inference using only:
 - models/trpg-final/model.onnx
 - models/trpg-final/config.json

NOTE: 使用自制字符级 tokenizer（非训练时 tokenizer），结果可能与原模型输出不一致，
但可在没有 tokenizer 文件时完成端到端推理演示。
"""

import os, sys, json, re
import numpy as np
import onnxruntime as ort

MODEL_DIR = "models/trpg-final"
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")
CFG_PATH = os.path.join(MODEL_DIR, "config.json")
MAX_LEN = 128

# load id2label & vocab_size
with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)
id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}
vocab_size = int(cfg.get("vocab_size", 30000))
pad_id = int(cfg.get("pad_token_id", 0))

# simple char-level tokenizer (adds [CLS]=101, [SEP]=102, pads with pad_id)
CLS_ID = 101
SEP_ID = 102


def char_tokenize(text, max_length=MAX_LEN):
    chars = list(text)
    # reserve 2 for CLS and SEP
    max_chars = max_length - 2
    chars = chars[:max_chars]
    ids = [CLS_ID] + [100 + (ord(c) % (vocab_size - 200)) for c in chars] + [SEP_ID]
    attn = [1] * len(ids)
    # pad
    pad_len = max_length - len(ids)
    ids += [pad_id] * pad_len
    attn += [0] * pad_len
    # offsets: for CLS/SEP/pad use (0,0); for char tokens map to character positions
    offsets = [(0, 0)]
    pos = 0
    for c in chars:
        offsets.append((pos, pos + 1))
        pos += 1
    offsets.append((0, 0))  # SEP
    offsets += [(0, 0)] * pad_len
    return {
        "input_ids": np.array([ids], dtype=np.int64),
        "attention_mask": np.array([attn], dtype=np.int64),
        "offset_mapping": np.array([offsets], dtype=np.int64),
        "text": text,
    }


# onnx runtime session
providers = [
    p
    for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
    if p in ort.get_available_providers()
]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=providers)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


text = sys.argv[1] if len(sys.argv) > 1 else "风雨 2024-06-08 21:44:59 剧烈的疼痛..."
inp = char_tokenize(text, MAX_LEN)

# build feed dict matching session inputs
feed = {}
for s_in in sess.get_inputs():
    name = s_in.name
    if name in inp:
        feed[name] = inp[name]

outs = sess.run(None, feed)
logits = np.asarray(outs[0])  # (batch, seq_len, num_labels)
probs = softmax(logits)

ids = inp["input_ids"][0]
offsets = inp["offset_mapping"][0]
attn = inp["attention_mask"][0]

# reconstruct token strings (CLS, each char, SEP)
tokens = []
for i, idv in enumerate(ids):
    if i == 0:
        tokens.append("[CLS]")
    else:
        if offsets[i][0] == 0 and offsets[i][1] == 0:
            # SEP or pad
            if attn[i] == 1:
                tokens.append("[SEP]")
            else:
                tokens.append("[PAD]")
        else:
            s, e = offsets[i]
            tokens.append(text[s:e])

# print raw logits shape and a small slice for inspection
print("Raw logits shape:", logits.shape)
print("\nPer-token logits (index token -> first 6 logits):")
for i, (t, l, a) in enumerate(zip(tokens, logits[0], attn)):
    if not a:
        continue
    print(f"{i:03d} {t:>6} ->", np.around(l[:6], 3).tolist())

# predictions & probs
pred_ids = logits.argmax(-1)[0]
pred_probs = probs[0, np.arange(probs.shape[1]), pred_ids]

print("\nPer-token predictions (token \\t label \\t prob):")
for i, (t, pid, pprob, a) in enumerate(zip(tokens, pred_ids, pred_probs, attn)):
    if not a:
        continue
    lab = id2label.get(int(pid), "O")
    print(f"{t}\t{lab}\t{pprob:.3f}")

# merge BIO into entities using offsets
entities = []
cur = None
for i, (pid, pprob, off, a) in enumerate(zip(pred_ids, pred_probs, offsets, attn)):
    if not a or (off[0] == off[1] == 0):
        if cur:
            entities.append(cur)
            cur = None
        continue
    label = id2label.get(int(pid), "O")
    if label == "O":
        if cur:
            entities.append(cur)
            cur = None
        continue
    if label.startswith("B-") or cur is None or label[2:] != cur["type"]:
        if cur:
            entities.append(cur)
        cur = {
            "type": label[2:],
            "start": int(off[0]),
            "end": int(off[1]),
            "probs": [float(pprob)],
        }
    else:
        cur["end"] = int(off[1])
        cur["probs"].append(float(pprob))
if cur:
    entities.append(cur)


# small fixes (timestamp/speaker) like main.py
def fix_timestamp(ts):
    if not ts:
        return ts
    m = re.match(r"^(\d{1,2})-(\d{2})-(\d{2})(.*)", ts)
    if m:
        y, mo, d, rest = m.groups()
        if len(y) == 1:
            y = "202" + y
        elif len(y) == 2:
            y = "20" + y
        return f"{y}-{mo}-{d}{rest}"
    return ts


def fix_speaker(spk):
    if not spk:
        return spk
    spk = re.sub(r"[^\w\s\u4e00-\u9fff]+$", "", spk)
    if len(spk) == 1 and re.match(r"^[风雷电雨雪火水木金]", spk):
        return spk + "某"
    return spk


out = {"metadata": {}, "content": []}
for e in entities:
    s, e_pos = e["start"], e["end"]
    ent_text = text[s:e_pos]
    conf = round(float(np.mean(e["probs"])), 3)
    typ = e["type"]
    if typ in ("timestamp", "speaker"):
        ent_text = (
            fix_timestamp(ent_text) if typ == "timestamp" else fix_speaker(ent_text)
        )
        out["metadata"][typ] = ent_text
    else:
        out["content"].append({"type": typ, "content": ent_text, "confidence": conf})

print("\nConstructed JSON:")
print(json.dumps(out, ensure_ascii=False, indent=2))

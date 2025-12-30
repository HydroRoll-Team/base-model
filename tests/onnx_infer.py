import os, sys, json, re
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_DIR = "models/trpg-final"
ONNX_PATH = os.path.join(MODEL_DIR, "model.onnx")

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(ONNX_PATH, sess_options=so, providers=providers)

def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

text = sys.argv[1] if len(sys.argv) > 1 else "风雨 2024-06-08 21:44:59 剧烈的疼痛..."
inputs = tok(text, return_tensors="np", return_offsets_mapping=True, padding="max_length", truncation=True, max_length=128)

feed = {}
for inp in sess.get_inputs():
    if inp.name in inputs:
        feed[inp.name] = inputs[inp.name]

outs = sess.run(None, feed)
logits = np.asarray(outs[0])  # (batch, seq_len, num_labels)
probs = softmax(logits)

ids = inputs["input_ids"][0]
offsets = inputs["offset_mapping"][0]
attn = inputs["attention_mask"][0]
tokens = tok.convert_ids_to_tokens(ids)

print("Raw logits shape:", logits.shape)
# print("\nPer-token raw logits (token : [..first 8 logits..])")
# for i, (t, l, a) in enumerate(zip(tokens, logits[0], attn)):
#     if not a:
#         continue
#     print(f"{i:03d}", t, "->", np.around(l[:8], 4).tolist())

pred_ids = logits.argmax(-1)[0]
pred_probs = probs[0, np.arange(probs.shape[1]), pred_ids]

with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    cfg = json.load(f)
id2label = {int(k): v for k, v in cfg.get("id2label", {}).items()}

print("\nPer-token predictions (token \\t label \\t prob):")
for i, (t, pid, pprob, a) in enumerate(zip(tokens, pred_ids, pred_probs, attn)):
    if not a:
        continue
    lab = id2label.get(int(pid), "O")
    print(f"{t}\t{lab}\t{pprob:.3f}")

# 聚合实体
entities = []
cur = None
for i, (pid, pprob, off, a) in enumerate(zip(pred_ids, pred_probs, offsets, attn)):
    if not a or (off[0] == off[1] == 0):
        if cur:
            entities.append(cur); cur = None
        continue
    label = id2label.get(int(pid), "O")
    if label == "O":
        if cur:
            entities.append(cur); cur = None
        continue
    if label.startswith("B-") or cur is None or label[2:] != cur["type"]:
        if cur:
            entities.append(cur)
        cur = {"type": label[2:], "tokens": [i], "start": int(off[0]), "end": int(off[1]), "probs":[float(pprob)]}
    else:
        cur["tokens"].append(i)
        cur["end"] = int(off[1])
        cur["probs"].append(float(pprob))
if cur:
    entities.append(cur)

def fix_timestamp(ts):
    if not ts: return ts
    m = re.match(r"^(\d{1,2})-(\d{2})-(\d{2})(.*)", ts)
    if m:
        y, mo, d, rest = m.groups()
        if len(y)==1: y="202"+y
        elif len(y)==2: y="20"+y
        return f"{y}-{mo}-{d}{rest}"
    return ts
def fix_speaker(spk):
    if not spk: return spk
    spk = re.sub(r"[^\w\s\u4e00-\u9fff]+$", "", spk)
    if len(spk)==1 and re.match(r"^[风雷电雨雪火水木金]", spk):
        return spk+"某"
    return spk

out = {"metadata": {}, "content": []}
for e in entities:
    s, epos = e["start"], e["end"]
    ent_text = text[s:epos]
    conf = round(float(np.mean(e["probs"])), 3)
    typ = e["type"]
    if typ in ("timestamp", "speaker"):
        if typ=="timestamp":
            ent_text = fix_timestamp(ent_text)
        else:
            ent_text = fix_speaker(ent_text)
        out["metadata"][typ] = ent_text
    else:
        out["content"].append({"type": typ, "content": ent_text, "confidence": conf})

print("\nConstructed JSON:")
print(json.dumps(out, ensure_ascii=False, indent=2))
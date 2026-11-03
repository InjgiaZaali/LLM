import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--adapter", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--condition", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_new_tokens", type=int, default=30)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_examples", type=int, default=0)
args = ap.parse_args()


def parse(raw):
    """Returns (labels, malformed). No synonym mapping."""
    try:
        data = json.loads(raw.strip())
    except Exception:
        return [], True
    if not isinstance(data, list):
        return [], True
    seen = []
    for x in data:
        x = str(x).strip().lower()
        if x in EMOTIONS and x not in seen:
            seen.append(x)
    return seen, len(seen) == 0


tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, args.adapter)
model.eval()

rows = [json.loads(l) for l in open(args.data, encoding="utf-8")]
if args.max_examples:
    rows = rows[: args.max_examples]

results = []
for i in range(0, len(rows), args.batch):
    chunk = rows[i:i + args.batch]
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                     add_generation_prompt=True, tokenize=False)
             for r in chunk]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    for r, o in zip(chunk, out):
        raw = tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True)
        pred, bad = parse(raw)
        results.append({
            "id": r["id"], "lang": r["lang"], "condition": args.condition,
            "model": args.model, "seed": args.seed,
            "gold_labels": json.dumps(r["gold"]),
            "pred_labels": json.dumps(pred),
            "raw_output": raw.strip(), "malformed": int(bad)})
    print(f"{min(i + args.batch, len(rows))}/{len(rows)}", flush=True)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(args.out, sep="\t", index=False)
print("malformed rate:", sum(r["malformed"] for r in results) / len(results))
print("saved", args.out)

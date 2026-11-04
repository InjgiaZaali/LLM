import argparse
import math
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer, TrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--train", required=True)
ap.add_argument("--dev", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.05)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=2)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max_examples", type=int, default=0)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return rows[: args.max_examples] if args.max_examples else rows


def encode(ex):
    text = tok.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}],
        add_generation_prompt=True, tokenize=False)
    p = tok(text, add_special_tokens=False)["input_ids"]
    a = tok(ex["answer"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + a)[: args.max_len]
    labels = ([-100] * len(p) + a)[: args.max_len]  # loss only on the answer
    return {"input_ids": ids, "labels": labels}


def collate(batch):
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, att = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
        lab.append(b["labels"] + [-100] * pad)
        att.append([1] * len(b["input_ids"]) + [0] * pad)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
            "attention_mask": torch.tensor(att)}


train_ds = Dataset.from_list([encode(e) for e in load(args.train)])
dev_ds = Dataset.from_list([encode(e) for e in load(args.dev)])

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
model = AutoModelForCausalLM.from_pretrained(
    args.model, quantization_config=bnb, dtype=torch.bfloat16, device_map="cuda")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
    target_modules="all-linear", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

steps = math.ceil(len(train_ds) / (args.batch * args.accum)) * math.ceil(args.epochs)
warmup = max(1, int(0.03 * steps))
targs = TrainingArguments(
    output_dir=args.out, num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=args.accum, learning_rate=args.lr,
    lr_scheduler_type="cosine", warmup_steps=warmup, bf16=True,
    logging_steps=10, eval_strategy="epoch", save_strategy="epoch",
    save_total_limit=2, load_best_model_at_end=True,
    metric_for_best_model="eval_loss", greater_is_better=False,
    seed=args.seed, report_to="none", remove_unused_columns=False)

Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=collate).train()

model.save_pretrained(Path(args.out) / "best_adapter")
json.dump(vars(args), open(Path(args.out) / "run_config.json", "w"), indent=2)
print("done")

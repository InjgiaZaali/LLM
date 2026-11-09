"""Controlled Qwen zero-shot prompt-wording sensitivity test on XED DEV only.

Run from the repository root on an NVIDIA GPU:
    pip install -r requirements-portable.txt
    python scripts/prompt_sensitivity_dev.py --n 200 --seed 42

Both wordings use exactly the same sampled dev rows, model, English instruction
language, decoding, token budget, and parser. Does not read the test partition.

Outputs (created only after actually running the experiment):
    results/prompt_sensitivity/qwen_dev_prompt_sensitivity_predictions.csv
    results/prompt_sensitivity/qwen_dev_prompt_sensitivity_metrics.csv

The 200-example subset is a speed-conscious sensitivity probe; use --n 954
for the full development partition. Report the actual sample size honestly.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, jaccard_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def make_prompt(sentence, variant):
    if variant == "original":
        return (
            "Identify the emotions in this sentence.\n\n"
            "Possible emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\n"
            "Return only a JSON array of the emotions expressed in the sentence.\n"
            "Do not explain your answer.\n\n"
            f"Sentence: {sentence}\nAnswer:"
        )
    if variant == "explicit_multilabel":
        return (
            "Classify the emotions conveyed by this sentence.\n\n"
            "Allowed labels: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\n"
            "One, multiple, or no emotions may apply. Return exactly a JSON array "
            "of permitted English label names, e.g. [\"joy\", \"trust\"] or [].\n"
            "Do not add any explanation or other text.\n\n"
            f"Sentence: {sentence}\nAnswer:"
        )
    raise ValueError(f"Unknown prompt variant: {variant}")


def parse_prediction(raw):
    try:
        values = json.loads(raw.strip())
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            return [], True
        values = [v.strip().lower() for v in values]
        if not all(v in EMOTIONS for v in values):
            return [], True
        return list(dict.fromkeys(values)), False
    except (json.JSONDecodeError, TypeError, AttributeError):
        return [], True


def binarize(lists):
    return np.array([[int(label in labels) for label in EMOTIONS] for labels in lists], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200,
                    help="Number of dev pairs to sample; 0 means all 954")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=60)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Enable an NVIDIA GPU, e.g. Google Colab GPU runtime.")

    torch.manual_seed(args.seed)
    dev_path = ROOT / "data/processed/xed_dev.tsv"
    df = pd.read_csv(dev_path, sep="\t").reset_index(drop=True)
    if len(df) != 954:
        raise RuntimeError(f"Expected 954 dev rows, got {len(df)} from {dev_path}")
    if args.n < 0 or args.n > len(df):
        raise ValueError("--n must be between 0 and the full development-set size")
    if args.batch < 1:
        raise ValueError("--batch must be >= 1")

    # Sample *once*, then reuse identical records and labels for every variant.
    if args.n:
        chosen = df.sample(n=args.n, random_state=args.seed).sort_index()
    else:
        chosen = df

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16
    ).to("cuda")
    model.eval()

    output_dir = ROOT / "results/prompt_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "qwen_dev_prompt_sensitivity_predictions.csv"
    metric_path = output_dir / "qwen_dev_prompt_sensitivity_metrics.csv"

    preds, metrics = [], []
    for language, column in (("english", "english_text"), ("spanish", "es_text")):
        for variant in ("original", "explicit_multilabel"):
            condition_rows = []
            for start in range(0, len(chosen), args.batch):
                batch = chosen.iloc[start:start + args.batch]
                prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": make_prompt(row[column], variant)}],
                        tokenize=False, add_generation_prompt=True
                    )
                    for _, row in batch.iterrows()
                ]
                inputs = tokenizer(
                    prompts, padding=True, add_special_tokens=False,
                    return_tensors="pt"
                ).to("cuda")
                with torch.inference_mode():
                    tokens = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id
                    )
                prompt_length = inputs["input_ids"].shape[1]
                generated = tokenizer.batch_decode(
                    tokens[:, prompt_length:], skip_special_tokens=True
                )
                for (idx, row), raw in zip(batch.iterrows(), generated):
                    pred, malformed = parse_prediction(raw)
                    condition_rows.append({
                        "row_index": int(idx), "language": language,
                        "prompt_variant": variant, "sentence": row[column],
                        "actual": row["emotion_labels"],
                        "predicted": json.dumps(pred),
                        "raw_output": raw.strip(), "malformed": malformed,
                    })
                print(f"{language} {variant}: {min(start + args.batch,len(chosen))}/{len(chosen)}", flush=True)

            preds.extend(condition_rows)
            actual = [json.loads(row["actual"]) for row in condition_rows]
            predicted = [json.loads(row["predicted"]) for row in condition_rows]
            y, p = binarize(actual), binarize(predicted)
            metrics.append({
                "language": language,
                "prompt_variant": variant,
                "examples": len(condition_rows),
                "sample_seed": args.seed,
                "model": MODEL_NAME,
                "decoding": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "micro_f1": f1_score(y, p, average="micro", zero_division=0),
                "sample_jaccard": jaccard_score(y, p, average="samples", zero_division=0),
                "malformed_rate": np.mean([row["malformed"] for row in condition_rows]),
            })
            # Save completed condition results in case a later condition fails.
            pd.DataFrame(preds).to_csv(prediction_path, index=False)
            pd.DataFrame(metrics).to_csv(metric_path, index=False)

    print("\nDEV-ONLY controlled prompt-wording comparison:")
    print(pd.DataFrame(metrics).to_string(index=False))
    print("Saved:", prediction_path)
    print("Saved:", metric_path)
    print("The test set was not loaded or evaluated.")


if __name__ == "__main__":
    main()

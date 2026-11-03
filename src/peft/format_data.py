import argparse
import json
import random
from pathlib import Path

import pandas as pd

EMOTIONS = ["anger", "anticipation", "disgust", "fear",
            "joy", "sadness", "surprise", "trust"]

INSTRUCTION = (
    "Identify the emotions in this sentence.\n\n"
    "Possible emotions: anger, anticipation, disgust, fear, joy, sadness, surprise, trust.\n\n"
    "Return only a JSON array of the emotions expressed in the sentence.\n"
    "Do not explain your answer."
)


def make_example(row, lang):
    text = row["english_text"] if lang == "en" else row["es_text"]
    labels = json.loads(row["emotion_labels"])
    ordered = [e for e in EMOTIONS if e in labels]  # fixed label order
    return {
        "id": row["alignment_id"],
        "lang": lang,
        "prompt": f"{INSTRUCTION}\n\nSentence: {text}\nAnswer:",
        "answer": json.dumps(ordered),
        "gold": ordered,
    }


def write_jsonl(path, examples):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"{path}: {len(examples)} examples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="data/formatted")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets = {}
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(Path(args.data_dir) / f"xed_{split}.tsv", sep="\t")
        for lang in ["en", "es"]:
            exs = [make_example(r, lang) for _, r in df.iterrows()]
            sets[(split, lang)] = exs
            write_jsonl(out / f"{split}_{lang}.jsonl", exs)

    bi = sets[("train", "en")] + sets[("train", "es")]
    random.Random(args.seed).shuffle(bi)
    write_jsonl(out / "train_bilingual.jsonl", bi)


if __name__ == "__main__":
    main()

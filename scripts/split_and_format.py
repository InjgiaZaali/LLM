import argparse
import json
import re
import unicodedata
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

LABELS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

INSTRUCTION = (
    "Classify the emotions expressed in the following movie subtitle line. "
    "The permitted labels are: anger, anticipation, disgust, fear, joy, sadness, surprise, trust. "
    "One or several labels may apply. "
    "Answer only with a JSON list containing the permitted label names.\n\n"
    "Subtitle: {text}"
)

SPLITS = ("train", "dev", "test")


def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).casefold()
    return re.sub(r"\s+", " ", text).strip()


def find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        parent[x], x = root, parent[x]
    return root


def union(parent, a, b):
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def assign_groups(df):
    parent = list(range(len(df)))
    for col in ("en_key", "es_key"):
        seen = {}
        for i, key in enumerate(df[col]):
            if key in seen:
                union(parent, seen[key], i)
            else:
                seen[key] = i
    return [find(parent, i) for i in range(len(df))]


def label_matrix(label_lists):
    y = np.zeros((len(label_lists), len(LABELS)), dtype=int)
    for i, labels in enumerate(label_lists):
        for label in labels:
            y[i, LABELS.index(label)] = 1
    return y


def stratified_group_split(group_labels, seed, dev_frac, test_frac):
    y = label_matrix(group_labels)
    x = np.zeros(len(y))

    outer = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(outer.split(x, y))

    inner = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=dev_frac / (1 - test_frac), random_state=seed
    )
    train_rel, dev_rel = next(inner.split(x[trainval_idx], y[trainval_idx]))

    assignment = np.empty(len(y), dtype=object)
    assignment[trainval_idx[train_rel]] = "train"
    assignment[trainval_idx[dev_rel]] = "dev"
    assignment[test_idx] = "test"
    return assignment


def check_leakage(df):
    for col in ("alignment_id", "en_key", "es_key", "group"):
        sets = {s: set(df.loc[df["split"] == s, col]) for s in SPLITS}
        for a, b in combinations(SPLITS, 2):
            overlap = sets[a] & sets[b]
            assert not overlap, f"{col}: {len(overlap)} valores compartidos entre {a} y {b}"


def pick_few_shot(train, seed, k):
    pool = train[
        train["english_text"].str.len().between(20, 100)
        & train["es_text"].str.len().between(20, 100)
        & (train["emotion_labels"].apply(len) <= 2)
    ].sample(frac=1, random_state=seed)

    chosen, covered = [], set()
    for _ in range(k):
        best_id, best_gain = None, -1
        for row in pool.itertuples():
            if row.alignment_id in chosen:
                continue
            gain = len(set(row.emotion_labels) - covered)
            if gain > best_gain:
                best_id, best_gain = row.alignment_id, gain
        chosen.append(best_id)
        covered |= set(pool.loc[pool["alignment_id"] == best_id, "emotion_labels"].iloc[0])
    return chosen


def ordered(labels):
    return [label for label in LABELS if label in labels]


def to_record(row, lang):
    text = row.english_text if lang == "en" else row.es_text
    labels = ordered(row.emotion_labels)
    return {
        "alignment_id": row.alignment_id,
        "language": lang,
        "text": text,
        "labels": labels,
        "messages": [
            {"role": "user", "content": INSTRUCTION.format(text=text)},
            {"role": "assistant", "content": json.dumps(labels)},
        ],
    }


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_stats(df):
    stats = {}
    for s in SPLITS:
        part = df[df["split"] == s]
        counts = label_matrix(part["emotion_labels"].tolist()).sum(axis=0)
        stats[s] = {
            "rows": int(len(part)),
            "share": round(len(part) / len(df), 4),
            "groups": int(part["group"].nunique()),
            "mean_labels_per_line": round(float(counts.sum() / len(part)), 4),
            "label_counts": {label: int(c) for label, c in zip(LABELS, counts)},
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="xed_english_spanish_matched.tsv")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-frac", type=float, default=0.20)
    parser.add_argument("--test-frac", type=float, default=0.10)
    parser.add_argument("--shots", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, sep="\t", dtype=str, keep_default_na=False)
    df["emotion_labels"] = df["emotion_labels"].apply(json.loads)
    n_loaded = len(df)

    df = df.drop_duplicates(subset="alignment_id").reset_index(drop=True)
    n_unique_ids = len(df)

    df["en_key"] = df["english_text"].apply(normalize)
    df["es_key"] = df["es_text"].apply(normalize)
    df = df[(df["en_key"] != "") & (df["es_key"] != "")].reset_index(drop=True)
    df["group"] = assign_groups(df)

    groups = (
        df.groupby("group")["emotion_labels"]
        .apply(lambda s: sorted({label for labels in s for label in labels}))
        .sort_index()
    )
    assignment = stratified_group_split(groups.tolist(), args.seed, args.dev_frac, args.test_frac)
    df["split"] = df["group"].map(dict(zip(groups.index, assignment)))

    check_leakage(df)

    few_shot_ids = pick_few_shot(df[df["split"] == "train"], args.seed, args.shots)
    assert set(df.loc[df["alignment_id"].isin(few_shot_ids), "split"]) == {"train"}

    df.assign(emotion_labels=df["emotion_labels"].apply(json.dumps)).to_csv(
        out / "split_assignments.tsv", sep="\t", index=False
    )

    for s in SPLITS:
        part = df[df["split"] == s]
        for lang in ("en", "es"):
            write_jsonl(out / f"{lang}_{s}.jsonl", [to_record(r, lang) for r in part.itertuples()])

    with open(out / "few_shot_ids.json", "w", encoding="utf-8") as f:
        json.dump(few_shot_ids, f, indent=2)

    stats = split_stats(df)
    stats["meta"] = {
        "seed": args.seed,
        "rows_loaded": n_loaded,
        "rows_unique_alignment_id": n_unique_ids,
        "rows_after_cleaning": int(len(df)),
        "duplicate_groups": int((df.groupby("group").size() > 1).sum()),
        "rows_in_duplicate_groups": int(df.groupby("group")["group"].transform("size").gt(1).sum()),
        "labels": LABELS,
    }
    with open(out / "split_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps(stats["meta"], indent=2))
    for s in SPLITS:
        print(s, stats[s]["rows"], stats[s]["share"], stats[s]["label_counts"])
    print("few-shot:", few_shot_ids)


if __name__ == "__main__":
    main()
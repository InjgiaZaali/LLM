"""Create leakage-safe 70/20/10 multilingual multi-label XED splits.

Run from the folder containing xed_english_spanish_matched.tsv:
    python split_en_es_dataset.py

Dependencies: pandas, numpy (pip install pandas numpy)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Change this filename/path if your matched TSV is in a different folder.
INPUT_FILE = Path("xed_english_spanish_matched.tsv")
OUTPUT_DIR = Path(".")
RANDOM_SEED = 42
EMOTIONS = [
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust",
]


def multilabel_stratified_groups(y, proportions=(0.7, 0.2, 0.1), seed=42):
    """Iterative multi-label stratification of *groups*, never individual rows.

    y: group-by-emotion binary matrix, with one row per unique English text.
    Returns group-level split numbers: 0=train, 1=development, 2=test.
    """
    rng = np.random.RandomState(seed)
    fractions = np.asarray(proportions, dtype=float)
    desired_group_sizes = fractions * len(y)
    desired_emotion_counts = np.outer(fractions, y.sum(axis=0)).astype(float)
    assignments = np.full(len(y), -1, dtype=int)

    while True:
        remaining_counts = y[assignments == -1].sum(axis=0)
        nonzero_counts = remaining_counts[remaining_counts > 0]
        if len(nonzero_counts) == 0:
            break

        # Select the currently rarest remaining emotion.
        rarest = np.flatnonzero(remaining_counts == nonzero_counts.min())[0]
        candidates = np.flatnonzero((y[:, rarest] == 1) & (assignments == -1))
        rng.shuffle(candidates)

        for group_index in candidates:
            # Choose the split that needs this emotion most.
            remaining_need = desired_emotion_counts[:, rarest]
            options = np.flatnonzero(np.isclose(remaining_need, remaining_need.max()))
            if len(options) > 1:
                space = desired_group_sizes[options]
                options = options[np.flatnonzero(np.isclose(space, space.max()))]

            split = int(rng.choice(options))
            assignments[group_index] = split
            desired_group_sizes[split] -= 1
            desired_emotion_counts[split] -= y[group_index]

    # Handle any unlabeled groups (not expected for this XED file).
    for group_index in np.flatnonzero(assignments == -1):
        options = np.flatnonzero(
            np.isclose(desired_group_sizes, desired_group_sizes.max())
        )
        split = int(rng.choice(options))
        assignments[group_index] = split
        desired_group_sizes[split] -= 1

    return assignments


def main():
    df = pd.read_csv(INPUT_FILE, sep="\t")
    df["emotion_labels"] = df["emotion_labels"].apply(json.loads)

    # Validate the assumed one-English-text-per-alignment-ID structure.
    assert df.groupby("alignment_id")["english_text"].nunique().eq(1).all(), (
        "An alignment ID has multiple English texts; grouping must be revised."
    )
    assert all(set(labels).issubset(EMOTIONS) for labels in df["emotion_labels"])

    # Group using English text. This keeps each alignment ID together AND
    # prevents identical English sentences from crossing split boundaries.
    groups = df.groupby("english_text", sort=False)["emotion_labels"].apply(
        lambda rows: sorted({emotion for labels in rows for emotion in labels})
    ).reset_index()

    y = np.array([
        [int(emotion in labels) for emotion in EMOTIONS]
        for labels in groups["emotion_labels"]
    ], dtype=int)

    group_splits = multilabel_stratified_groups(y, seed=RANDOM_SEED)
    group_to_split = dict(zip(groups["english_text"], group_splits))
    df["split"] = df["english_text"].map(group_to_split).map(
        {0: "train", 1: "dev", 2: "test"}
    )

    # The same alignment, English text, or Spanish text must never be in
    # different partitions. Stop rather than save files if leakage is found.
    for column in ["alignment_id", "english_text", "es_text"]:
        assert df.groupby(column)["split"].nunique().max() == 1, (
            f"Data leakage detected: {column} crosses partitions!"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for part in ["train", "dev", "test"]:
        subset = df.loc[df["split"] == part].drop(columns="split").copy()
        subset["emotion_labels"] = subset["emotion_labels"].apply(json.dumps)
        path = OUTPUT_DIR / f"xed_{part}.tsv"
        subset.to_csv(path, sep="\t", index=False)
        print(f"{part.upper()}: {len(subset)} rows ({len(subset)/len(df):.1%}) -> {path}")

    print("\nLeakage check passed for alignment IDs and both languages.")
    print("Random seed:", RANDOM_SEED)
    print("Total rows:", len(df))


if __name__ == "__main__":
    main()

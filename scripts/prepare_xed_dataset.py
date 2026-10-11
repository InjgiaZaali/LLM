
import pandas as pd
import csv
import json

# 1. Load Spanish projections
rows = []

with open("es-projections.tsv", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\r\n")

        if line.count("\t") != 1:
            continue

        text, labels = line.split("\t")
        rows.append([text, labels])

es = pd.DataFrame(rows, columns=["es_text", "labels"])

print("Spanish labelled rows:", len(es))


# 2. Load English-Spanish alignment pairs
pairs = pd.read_csv(
    "pairs-es.txt",
    sep="\t",
    header=None,
    names=[
        "source_file",
        "spanish_file",
        "source_line",
        "spanish_line",
        "english_text",
        "es_text"
    ],
    dtype=str,
    quoting=csv.QUOTE_NONE
)

pairs = pairs[
    pairs["source_file"].str.startswith("en/", na=False) &
    pairs["spanish_file"].str.startswith("es/", na=False)
].copy()

print("English-Spanish pairs:", len(pairs))


# 3. Create alignment identifiers
pairs["alignment_id"] = (
    pairs["source_file"] + "::" + pairs["source_line"]
)


# 4. Remove ambiguous Spanish sentence matches
match_counts = pairs.groupby("es_text")["alignment_id"].transform("nunique")

pairs_clean = pairs[match_counts == 1].copy()

pairs_clean = pairs_clean.drop_duplicates(subset=["es_text"])


# 5. Match labelled Spanish with English subtitles
matched = es.merge(
    pairs_clean[
        ["alignment_id", "english_text", "es_text"]
    ],
    on="es_text",
    how="inner",
    validate="one_to_one"
)

print("Initially matched examples:", len(matched))


# 6. Convert numeric labels to emotion names
label_map = {
    "1": "anger",
    "2": "anticipation",
    "3": "disgust",
    "4": "fear",
    "5": "joy",
    "6": "sadness",
    "7": "surprise",
    "8": "trust"
}

def convert_labels(label_string):
    numbers = label_string.split(",")

    return list(dict.fromkeys(
        label_map[number.strip()]
        for number in numbers
    ))

matched["emotion_labels"] = matched["labels"].apply(convert_labels)


# 7. Remove conflicting labels for the same alignment ID
matched["label_key"] = matched["emotion_labels"].apply(
    lambda labels: tuple(sorted(labels))
)

consistent = matched.groupby("alignment_id")["label_key"].transform("nunique") == 1

matched = matched[consistent].copy()
matched = matched.drop(columns=["label_key"])

print("Final matched examples:", len(matched))
print("Unique alignment IDs:", matched["alignment_id"].nunique())


# 8. Save the matched dataset
output_file = "xed_english_spanish_matched.tsv"

matched.assign(
    emotion_labels=matched["emotion_labels"].apply(json.dumps)
).to_csv(
    output_file,
    sep="\t",
    index=False
)

print("Saved:", output_file)


# 9. Verify saved dataset
check = pd.read_csv(output_file, sep="\t")

check["emotion_labels"] = check["emotion_labels"].apply(json.loads)

print("Rows in saved file:", len(check))

print(
    check[
        ["alignment_id", "english_text", "es_text", "emotion_labels"]
    ].head(10)
)

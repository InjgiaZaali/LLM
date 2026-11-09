import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

E = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
SEEDS = [42, 43, 44]

def load(cond, lang, seed):
    return pd.read_csv(f"predictions/test_{cond}_1ep_{lang}_seed{seed}.tsv", sep="\t")

def vec(series):
    return np.array([[int(e in json.loads(x)) for e in E] for x in series])

# 1) Per-emotion precision / recall / F1 / support (mean over seeds)
rows = []
for cond in ["english", "spanish", "bilingual"]:
    for lang in ["en", "es"]:
        per = []
        for s in SEEDS:
            d = load(cond, lang, s)
            y, p = vec(d["gold_labels"]), vec(d["pred_labels"])
            pr, rc, f1, sup = precision_recall_fscore_support(y, p, zero_division=0)
            per.append((pr, rc, f1, sup))
        m = np.mean([x[:3] for x in per], axis=0)
        for i, e in enumerate(E):
            rows.append(dict(train=cond, eval=lang, emotion=e, precision=m[0][i],
                             recall=m[1][i], f1=m[2][i], support=int(per[0][3][i])))
pd.DataFrame(rows).round(2).to_csv("predictions/per_emotion_test.csv", index=False)
print("saved predictions/per_emotion_test.csv")

# 2) Example lines: Spanish text, English text, gold, and each model's prediction (seed 42)
texts = pd.read_csv("data/processed/xed_test.tsv", sep="\t")
base = load("english", "es", 42)[["id", "gold_labels"]].rename(columns={"id": "alignment_id"})
for cond in ["english", "spanish", "bilingual"]:
    base[cond] = load(cond, "es", 42)["pred_labels"].values
assert (texts["alignment_id"].values == base["alignment_id"].values).all()
base["es_text"] = texts["es_text"].values
base["english_text"] = texts["english_text"].values
base["models_disagree"] = base[["english", "spanish", "bilingual"]].nunique(axis=1) > 1
base["all_wrong"] = [all(json.loads(r[c]) != json.loads(r["gold_labels"]) for c in ["english", "spanish", "bilingual"])
                     for _, r in base.iterrows()]
base.to_csv("predictions/error_examples_test_es.csv", index=False)
print("lines where models disagree:", int(base["models_disagree"].sum()), "of", len(base))
print("lines where all three are wrong:", int(base["all_wrong"].sum()))
print(base[base["models_disagree"]].head(10)[["es_text", "english_text", "gold_labels", "english", "spanish", "bilingual"]].to_string())

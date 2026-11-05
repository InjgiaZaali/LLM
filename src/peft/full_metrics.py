import glob, json, re, sys
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (f1_score, jaccard_score, precision_score,
                             recall_score, hamming_loss)

E = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
split = sys.argv[1] if len(sys.argv) > 1 else "test"   # "dev" or "test"
rows = []
for path in sorted(glob.glob(f"predictions/{split}_*_1ep_*_seed*.tsv")):
    m = re.match(rf"predictions/{split}_(\w+?)_1ep_(en|es)_seed(\d+)\.tsv", path)
    cond, lang, seed = m.group(1), m.group(2), int(m.group(3))
    d = pd.read_csv(path, sep="\t")
    b = lambda c: np.array([[int(e in json.loads(x)) for e in E] for x in d[c]])
    y, p = b("gold_labels"), b("pred_labels")
    rows.append(dict(
        train=cond, eval=lang, seed=seed,
        micro_f1=f1_score(y, p, average="micro", zero_division=0),
        jaccard=jaccard_score(y, p, average="samples", zero_division=0),
        macro_f1=f1_score(y, p, average="macro", zero_division=0),
        micro_precision=precision_score(y, p, average="micro", zero_division=0),
        micro_recall=recall_score(y, p, average="micro", zero_division=0),
        hamming_loss=hamming_loss(y, p),
        exact_match=(y == p).all(axis=1).mean(),
        malformed=d["malformed"].mean()))
df = pd.DataFrame(rows)
df.to_csv(f"predictions/all_metrics_{split}_per_run.csv", index=False)
cols = ["micro_f1", "jaccard", "macro_f1", "micro_precision", "micro_recall",
        "hamming_loss", "exact_match", "malformed"]
g = df.groupby(["train", "eval"])[cols].agg(["mean", "std"]).round(3)
g.to_csv(f"predictions/all_metrics_{split}_summary.csv")
print(g.T.to_string())
print("scikit-learn", sklearn.__version__, "| zero_division=0 | std = sample std (n-1)")

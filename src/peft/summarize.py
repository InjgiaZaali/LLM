import glob, json, re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score

E = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
rows = []
for path in sorted(glob.glob("predictions/dev_*_1ep_*_seed*.tsv")):
    m = re.match(r"predictions/dev_(\w+?)_1ep_(en|es)_seed(\d+)\.tsv", path)
    cond, lang, seed = m.group(1), m.group(2), int(m.group(3))
    d = pd.read_csv(path, sep="\t")
    b = lambda c: np.array([[int(e in json.loads(x)) for e in E] for x in d[c]])
    y, p = b("gold_labels"), b("pred_labels")
    rows.append(dict(train=cond, eval=lang, seed=seed,
                     micro_f1=f1_score(y, p, average="micro", zero_division=0),
                     jaccard=jaccard_score(y, p, average="samples", zero_division=0),
                     malformed=d["malformed"].mean()))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
print()
g = df.groupby(["train", "eval"])[["micro_f1", "jaccard"]].agg(["mean", "std", "count"]).round(3)
print(g.to_string())

import json, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, jaccard_score

E = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
for path in sys.argv[1:]:
    d = pd.read_csv(path, sep="\t")
    b = lambda col: np.array([[int(e in json.loads(x)) for e in E] for x in d[col]])
    y, p = b("gold_labels"), b("pred_labels")
    print(path)
    print("  micro-F1 %.3f | sample-Jaccard %.3f | malformed %.3f" % (
        f1_score(y, p, average="micro", zero_division=0),
        jaccard_score(y, p, average="samples", zero_division=0),
        d["malformed"].mean()))

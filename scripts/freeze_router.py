"""Freeze the Part B router: multilingual-e5-small embeddings + logistic regression, fitted exactly as in
scripts/route_step4.py (2,000 training-mixture prompts per domain, SEED 20261029, C=4.0) and saved to
artifacts/router_e5lr.pkl with the embedding model revision. The router never sees eval prompts.
usage: .venv/bin/python -P scripts/freeze_router.py
"""
import hashlib, json, os, pickle, random
import numpy as np
from sklearn.linear_model import LogisticRegression

DOMAINS = ["math", "code", "sql", "tools", "translate", "general"]
N_TRAIN = 2000; SEED = 20261029; EMB = "intfloat/multilingual-e5-small"
REV = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


def user_text(msgs):
    return next((m["content"] for m in msgs if m["role"] == "user"), "")


def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMB, device="cpu", revision=REV); m.max_seq_length = 512
    return m.encode(["query: " + t[:4000] for t in texts], batch_size=64, normalize_embeddings=True, show_progress_bar=False)


rng = random.Random(SEED); X, y = [], []
for d in DOMAINS:
    lines = open(f"data/mixtures/N14M/spec/{d}.jsonl").readlines()
    for l in rng.sample(lines, min(N_TRAIN, len(lines))):
        X.append(user_text(json.loads(l)["messages"])); y.append(d)
E = embed(X)
cached = np.load("logs/step4/emb-train2000.npy")
print("max |emb - Part A cache| =", float(np.abs(E - cached).max()))
clf = LogisticRegression(max_iter=3000, C=4.0).fit(E, y)
os.makedirs("artifacts", exist_ok=True)
blob = pickle.dumps({"clf": clf, "emb_model": EMB, "emb_revision": REV, "n_train": N_TRAIN, "seed": SEED})
open("artifacts/router_e5lr.pkl", "wb").write(blob)
print("artifacts/router_e5lr.pkl sha256", hashlib.sha256(blob).hexdigest(), "train acc", clf.score(E, y))

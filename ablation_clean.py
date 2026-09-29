#!/usr/bin/env python3
"""
Clean Arc Gate ablation.

Purpose:
  Measure whether the production centroid-distance signal adds held-out
  discrimination beyond conventional prompt-injection signals.

This script does NOT import arc_gate.py and does NOT call the production proxy.
That prevents authority state, LLM judge, session tau, or routing logic from
leaking into the ablation.

Signals:
  phrase      : explicit phrase-pattern indicator
  tfidf       : production TF-IDF classifier probability
  conventional: max(phrase, tfidf)
  geometry    : angular distance from the production clean-prompt centroid
  full        : logistic stacker fit ONLY on the calibration split using
                conventional + geometry

Evaluation:
  AUROC, AUPRC, TPR@1%FPR, TPR@0.1%FPR.
  Bootstrap 95% CIs for delta(full - conventional) at fixed FPR.

Data:
  attacks: InjecAgent public test cases (all four published splits)
  benign : HuggingFaceH4/no_robots user instructions, sampled deterministically

The held-out test split is never used to choose thresholds or fit the stacker.
"""

from __future__ import annotations
import argparse, io, json, pickle, random, re, statistics, sys, urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

ARC_RAW = "https://raw.githubusercontent.com/9hannahnine-jpg/arc-gate/main"
INJEC_RAW = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"

PHRASES = [
    "ignore your guidelines","ignore all safety","ignore all previous",
    "ignore previous instructions","no longer bound","no filters",
    "ignore your safety","disregard your instructions","disregard all instructions",
    "override your instructions","bypass your safety","system override",
    "jailbreak","do anything now","previous instructions are void",
    "instructions are void","forget everything you were told",
    "unrestricted mode","restrictions lifted","all restrictions lifted",
    "no content policy","no ethical guidelines","filters are disabled",
    "safe mode off","safety off","filters off","evil ai","unrestricted ai",
    "dan mode","you are dan","act as dan","without any restrictions",
    "management override","executive override","ceo override",
    "as your operator","as your owner","as your creator","decode and follow",
    "execute the following base64","encoded instruction","base64 command",
    "[system]","### system","<!-- system:","<!-- note to ai","<!-- instruction",
    "[system note","[instruction","note for ai","by the way, ignore",
    "also, ignore","p.s. ignore","your new instructions are",
    "your instructions are now","secret instruction","hidden command",
    "covert instruction","the hidden instruction is","after translating follow",
    "bypass its own safety","steps to bypass","<information>","</information>",
    "before you can solve the task","important message from me",
    "after you do that, you can solve","signed,","this is an important message",
    "before ai had safety","before safety filters","remind me of the old days",
    "in a world where you have no restrictions","what if you had no guidelines",
    "imagine having no restrictions","if restrictions did not exist",
    "theoretically if you had no restrictions","suppose you had no restrictions",
    "imagine you have no restrictions","ignore your previous instructions",
    "set aside your guidelines","set aside your rules","violate your guidelines",
]

# Frozen copy of the production centroid prompts as of the ablation branch.
# Keeping them here makes the experiment independent of arc_gate.py imports.
GEO_CLEAN = [
    "What are your business hours?","Can I get a refund for my order?",
    "How do I reset my password?","What payment methods do you accept?",
    "Where is my order?","How long does shipping take?",
    "Can I speak to a human agent?","What is included in the pro plan?",
    "How do I contact support?","What is the capital of France?",
    "Who invented the telephone?","What is the speed of light?",
    "How does photosynthesis work?","What is the Pythagorean theorem?",
    "What are some good Python libraries for data analysis?",
    "How do I use a for loop in Python?","What is a REST API?",
    "How does HTTPS work?","What is the difference between SQL and NoSQL?",
    "Help me write a professional email declining a meeting",
    "Can you proofread this paragraph?","Can you recommend a good book?",
    "What should I pack for a trip to Japan?","How do I make pasta from scratch?",
    "How do I set up two-factor authentication?",
    "What are the best practices for API security?",
    "Explain the difference between REST and GraphQL",
    "What are the OWASP top 10 vulnerabilities?",
    "How do I write a good bug report?","What is a webhook and how do I use it?",
    "How do I generate an API key?","What is OAuth and how does it work?",
    "How do I set up a CI/CD pipeline?","What are microservices?",
    "What is Docker and how do I use it?","What is Kubernetes?",
    "How do I write unit tests?","What is test-driven development?",
    "Can you explain what inflation is?","Hi","Hello","Thanks","Thank you",
    "OK","Okay","Got it","Yes","No","Sure","Please","Help",
    "Can you help me?","I need help","Tell me more","Go on",
    "Who wrote Hamlet?","Tell me a joke","What time is it?",
]


def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def load_pickle(name: str):
    return pickle.load(io.BytesIO(get_bytes(f"{ARC_RAW}/models/{name}")))


def load_injecagent() -> list[str]:
    files = [
        "test_cases_dh_base.json","test_cases_ds_base.json",
        "test_cases_dh_enhanced.json","test_cases_ds_enhanced.json",
    ]
    out = []
    for fn in files:
        data = json.loads(get_bytes(f"{INJEC_RAW}/{fn}").decode("utf-8"))
        for row in data:
            txt = str(row.get("Tool Response", "")).strip()
            if txt:
                out.append(txt)
    # stable de-duplication
    return list(dict.fromkeys(out))


def load_benign(n: int, seed: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="train")
    pool = []
    for row in ds:
        msgs = row.get("messages") or row.get("prompt") or []
        if isinstance(msgs, str):
            txt = msgs.strip()
            if txt:
                pool.append(txt)
            continue
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    txt = str(m.get("content", "")).strip()
                    if txt:
                        pool.append(txt)
                    break
    pool = list(dict.fromkeys(pool))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def phrase_score(text: str) -> float:
    s = " ".join(text.lower().split())
    compact = s.replace(" ", "").replace("_", "")
    return float(any(p in s or p.replace(" ", "") in compact for p in PHRASES))


def tpr_at_fpr(y, score, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y, score)
    eligible = np.where(fpr <= target_fpr)[0]
    return float(np.max(tpr[eligible])) if len(eligible) else 0.0


def metrics(y, score):
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, score, 0.01),
        "tpr_at_0_1pct_fpr": tpr_at_fpr(y, score, 0.001),
    }


def bootstrap_delta(y, a, b, target_fpr, n_boot=1000, seed=123):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); a = np.asarray(a); b = np.asarray(b)
    vals = []
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        yy = y[idx]
        vals.append(tpr_at_fpr(yy, b[idx], target_fpr) -
                    tpr_at_fpr(yy, a[idx], target_fpr))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"delta": float(np.mean(vals)), "ci95": [float(lo), float(hi)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benign", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ablation_results.json")
    args = ap.parse_args()

    print("Loading independent datasets...")
    attacks = load_injecagent()
    benign = load_benign(args.benign, args.seed)
    print(f"attacks={len(attacks)} benign={len(benign)}")

    texts = np.array(attacks + benign, dtype=object)
    y = np.array([1] * len(attacks) + [0] * len(benign), dtype=int)

    idx = np.arange(len(y))
    cal_idx, test_idx = train_test_split(
        idx, test_size=0.60, random_state=args.seed, stratify=y
    )

    print("Loading production TF-IDF model...")
    char_vec = load_pickle("tfidf_char_vec.pkl")
    word_vec = load_pickle("tfidf_word_vec.pkl")
    clf = load_pickle("tfidf_clf2.pkl")
    X = hstack([char_vec.transform(texts), word_vec.transform(texts)])
    tfidf = clf.predict_proba(X)[:, 1]
    phrase = np.array([phrase_score(t) for t in texts])
    conventional = np.maximum(tfidf, phrase)

    print("Computing independent centroid-distance signal...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    clean_emb = model.encode(GEO_CLEAN, convert_to_numpy=True, normalize_embeddings=True)
    centroid = clean_emb.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
    emb = model.encode(texts.tolist(), batch_size=64, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True)
    cos = np.clip(emb @ centroid, -1 + 1e-7, 1 - 1e-7)
    geometry = np.arccos(cos)

    # Fit the combination ONLY on calibration examples.
    stack = LogisticRegression(max_iter=1000, class_weight="balanced")
    stack.fit(np.column_stack([conventional[cal_idx], geometry[cal_idx]]), y[cal_idx])
    full = stack.predict_proba(np.column_stack([conventional, geometry]))[:, 1]

    result = {
        "protocol": {
            "attack_source": "InjecAgent all four public test-case files",
            "benign_source": "HuggingFaceH4/no_robots user messages",
            "seed": args.seed,
            "calibration_fraction": 0.40,
            "test_fraction": 0.60,
            "n_attack": len(attacks),
            "n_benign": len(benign),
            "note": "No production proxy, authority state, LLM judge, tau/session state, or routing logic used.",
        },
        "calibration": {
            "full_coefficients": stack.coef_[0].tolist(),
            "full_intercept": stack.intercept_.tolist(),
        },
        "test": {},
    }

    yt = y[test_idx]
    scores = {
        "phrase": phrase[test_idx],
        "tfidf": tfidf[test_idx],
        "conventional": conventional[test_idx],
        "geometry": geometry[test_idx],
        "full": full[test_idx],
    }
    for name, score in scores.items():
        result["test"][name] = metrics(yt, score)

    result["incremental_geometry"] = {
        "delta_tpr_at_1pct_fpr": bootstrap_delta(
            yt, scores["conventional"], scores["full"], 0.01
        ),
        "delta_tpr_at_0_1pct_fpr": bootstrap_delta(
            yt, scores["conventional"], scores["full"], 0.001
        ),
        "delta_auroc": float(
            result["test"]["full"]["auroc"] - result["test"]["conventional"]["auroc"]
        ),
        "delta_auprc": float(
            result["test"]["full"]["auprc"] - result["test"]["conventional"]["auprc"]
        ),
    }

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

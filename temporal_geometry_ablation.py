#!/usr/bin/env python3
"""
Independent temporal-geometry ablation for Bendex / Arc Gate.

Goal
----
Test whether multi-turn *trajectory geometry* contains reproducible attack
information when ALL classifier-derived/security-labelled coordinates are removed.

No imports from arc_gate.py.
No TF-IDF scores.
No phrase flags.
No authority-state events.
No LLM judge.
No production tau formula.

The only observables are sentence embeddings of successive USER turns.

Temporal features
-----------------
- angular turn-to-turn step sizes
- path length
- first-to-last displacement
- tortuosity
- step acceleration
- early-vs-late drift ratio
- linear trend in step size
- direction-change / turning angle

Evaluation
----------
1. Ordinary stratified held-out test.
2. Leave-one-attack-corpus-out tests:
   train on attack corpus A + benign train, test on unseen attack corpus B + benign test.
3. Compare temporal model against a static baseline using only the FINAL user-turn
   embedding distance to the same fixed benign centroid used by production Arc Gate.

The cross-corpus test is decisive: if temporal geometry is a general attack signal,
it should survive attack-corpus shift instead of merely identifying dataset style.
"""

from __future__ import annotations
import ast, json, math, random, urllib.request
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
MAX_SESSIONS_PER_SOURCE = 2500
MIN_USER_TURNS = 3
ARC_GATE_RAW = "https://raw.githubusercontent.com/9hannahnine-jpg/arc-gate/main/arc_gate.py"

ATTACK_DATASETS = [
    ("tom_gibbs", "tom-gibbs/multi-turn_jailbreak_attack_datasets", None),
    ("harm_geometry", "yelyzavetahusieva/geometry-of-harmfulness-in-multi-turn-attacks", None),
]
BENIGN_DATASET = ("ultrachat", "HuggingFaceH4/ultrachat_200k", None)

def get_bytes(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()

def stable_sample(items, n=MAX_SESSIONS_PER_SOURCE, seed=SEED):
    if len(items) <= n:
        return items
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    return [items[i] for i in idx[:n]]

def production_clean_prompts():
    src = get_bytes(ARC_GATE_RAW).decode("utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_GEO_CLEAN_PROMPTS" for t in node.targets):
                return list(ast.literal_eval(node.value))
    raise RuntimeError("_GEO_CLEAN_PROMPTS not found")

def all_rows(name, config=None):
    from datasets import load_dataset
    ds = load_dataset(name, config) if config else load_dataset(name)
    out = []
    if hasattr(ds, "items"):
        # Prefer test/validation first to reduce accidental training-set artifacts,
        # then add train if needed.
        keys = list(ds.keys())
        order = sorted(keys, key=lambda k: (0 if "test" in k else 1 if ("valid" in k or "val" in k) else 2, k))
        for split in order:
            for row in ds[split]:
                rr = dict(row)
                rr["_split"] = split
                out.append(rr)
                if len(out) >= MAX_SESSIONS_PER_SOURCE * 4:
                    return out
    else:
        for row in ds:
            out.append(dict(row))
            if len(out) >= MAX_SESSIONS_PER_SOURCE * 4:
                break
    return out

def maybe_parse(v):
    if isinstance(v, (list, dict)):
        return v
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return v
        if s[0] in "[{":
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(s)
                except Exception:
                    pass
    return v

def extract_user_turns_from_obj(obj):
    """Recursively extract ordered user-side turns from common conversation schemas."""
    obj = maybe_parse(obj)
    if isinstance(obj, list):
        # Standard chat list
        if obj and all(isinstance(x, dict) for x in obj):
            turns = []
            for m in obj:
                role = str(m.get("role", m.get("from", m.get("speaker", "")))).lower()
                content = m.get("content", m.get("value", m.get("text", m.get("message", ""))))
                if role in {"user","human","client"} and isinstance(content, str) and content.strip():
                    turns.append(content.strip())
            if turns:
                return turns
        # Some datasets store a list of raw user prompts.
        if obj and all(isinstance(x, str) for x in obj):
            return [x.strip() for x in obj if x.strip()]
        # recurse
        best = []
        for x in obj:
            cand = extract_user_turns_from_obj(x)
            if len(cand) > len(best):
                best = cand
        return best

    if isinstance(obj, dict):
        # Common nested keys
        for key in (
            "turns","messages","conversation","conversations",
            "multi_turn_conversation","multi-turn conversation",
            "Multi-turn Conversation","dialogue","chat","history"
        ):
            if key in obj:
                cand = extract_user_turns_from_obj(obj[key])
                if cand:
                    return cand
        # dict keyed by turn number
        vals = list(obj.values())
        cand = extract_user_turns_from_obj(vals)
        if cand:
            return cand

    return []

def extract_session(row):
    # Explicit likely fields first.
    for key in (
        "Multi-turn Conversation","multi_turn_conversation","multi-turn conversation",
        "turns","messages","conversation","conversations","dialogue","chat","history"
    ):
        if key in row:
            turns = extract_user_turns_from_obj(row[key])
            if len(turns) >= MIN_USER_TURNS:
                return turns

    # Last-resort scan over structured fields.
    best = []
    for v in row.values():
        if isinstance(v, (list, dict, str)):
            cand = extract_user_turns_from_obj(v)
            if len(cand) > len(best):
                best = cand
    return best if len(best) >= MIN_USER_TURNS else None

def load_sessions(name, config=None):
    rows = all_rows(name, config)
    sessions = []
    for row in rows:
        s = extract_session(row)
        if s:
            sessions.append(s)
    # de-duplicate by first/last turns
    uniq = {}
    for s in sessions:
        key = (s[0][:200], s[-1][:200], len(s))
        uniq.setdefault(key, s)
    return stable_sample(list(uniq.values()))

def load_benign_ultrachat():
    # UltraChat has messages in standard role/content format.
    return load_sessions(BENIGN_DATASET[1], BENIGN_DATASET[2])

def angle(a, b):
    c = np.clip(float(np.dot(a, b)), -1 + 1e-7, 1 - 1e-7)
    return math.acos(c)

def temporal_features(E):
    """E: normalized embedding matrix, one row per user turn."""
    steps = np.array([angle(E[i-1], E[i]) for i in range(1, len(E))], dtype=float)
    path = float(np.sum(steps))
    endpoint = angle(E[0], E[-1])
    accel = np.diff(steps) if len(steps) > 1 else np.array([0.0])

    # step-size trend
    if len(steps) > 1:
        x = np.arange(len(steps), dtype=float)
        slope = float(np.polyfit(x, steps, 1)[0])
    else:
        slope = 0.0

    half = max(1, len(steps)//2)
    early = float(np.mean(steps[:half]))
    late = float(np.mean(steps[-half:]))
    late_early_ratio = late / (early + 1e-8)

    # Direction-change angle in ambient normalized embedding coordinates.
    turns = []
    for i in range(1, len(E)-1):
        v1 = E[i] - E[i-1]
        v2 = E[i+1] - E[i]
        n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
        if n1 > 1e-8 and n2 > 1e-8:
            turns.append(math.acos(np.clip(float(np.dot(v1,v2)/(n1*n2)), -1+1e-7, 1-1e-7)))
    turn_mean = float(np.mean(turns)) if turns else 0.0
    turn_max = float(np.max(turns)) if turns else 0.0

    return np.array([
        len(E),
        float(np.mean(steps)),
        float(np.std(steps)),
        float(np.max(steps)),
        float(steps[-1]),
        path,
        endpoint,
        path / (endpoint + 1e-8),
        float(np.mean(np.abs(accel))),
        float(np.max(np.abs(accel))),
        late_early_ratio,
        slope,
        turn_mean,
        turn_max,
    ], dtype=float)

FEATURE_NAMES = [
    "n_turns","mean_step","std_step","max_step","last_step",
    "path_length","endpoint_displacement","tortuosity",
    "mean_abs_acceleration","max_abs_acceleration",
    "late_early_ratio","step_slope","mean_turn_angle","max_turn_angle",
]

def tpr_at_fpr(y, score, target):
    fpr,tpr,_ = roc_curve(y, score)
    ok = np.where(fpr <= target)[0]
    return float(np.max(tpr[ok])) if len(ok) else 0.0

def metric_block(y, score):
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, score, 0.01),
        "tpr_at_0_1pct_fpr": tpr_at_fpr(y, score, 0.001),
    }

def fit_eval(Xtr,ytr,Xte,yte):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
    )
    model.fit(Xtr,ytr)
    score = model.predict_proba(Xte)[:,1]
    return model, score, metric_block(yte, score)

def split_indices(n, frac=0.6, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n*frac))
    return idx[:cut], idx[cut:]

def main():
    from sentence_transformers import SentenceTransformer

    print("Loading attack sessions...")
    attack_sources = {}
    failures = {}
    for key,name,config in ATTACK_DATASETS:
        try:
            ss = load_sessions(name,config)
            if ss:
                attack_sources[key] = ss
            else:
                failures[key] = "no parseable >=3-user-turn sessions"
        except Exception as e:
            failures[key] = repr(e)

    print("Loading benign sessions...")
    benign = load_benign_ultrachat()
    if not benign:
        raise RuntimeError("No benign UltraChat sessions parsed")

    print("counts", {k:len(v) for k,v in attack_sources.items()}, "benign", len(benign))
    print("failures", failures)
    if len(attack_sources) < 2:
        raise RuntimeError("Need at least two independent attack corpora for cross-corpus test")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Fixed production centroid only for static comparator.
    clean = production_clean_prompts()
    C = model.encode(clean, convert_to_numpy=True, normalize_embeddings=True).mean(axis=0)
    C = C/(np.linalg.norm(C)+1e-12)

    def featurize_sessions(sessions, label, source):
        X=[]; static=[]; meta=[]
        for j,s in enumerate(sessions):
            E = model.encode(s, batch_size=32, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
            if len(E) < MIN_USER_TURNS:
                continue
            X.append(temporal_features(E))
            static.append(angle(E[-1], C))
            meta.append({"label":label,"source":source,"n_turns":len(E)})
        return np.vstack(X), np.array(static), meta

    blocks = {}
    for k,ss in attack_sources.items():
        blocks[k] = featurize_sessions(ss,1,k)
    blocks["benign"] = featurize_sessions(benign,0,"ultrachat")

    results = {
        "protocol": {
            "observables": "MiniLM embeddings of successive user turns only",
            "forbidden_inputs": ["TF-IDF","phrase flags","authority events","LLM judge","production tau","attack labels as features"],
            "feature_names": FEATURE_NAMES,
            "attack_sources": list(attack_sources.keys()),
            "benign_source": "HuggingFaceH4/ultrachat_200k",
            "load_failures": failures,
        },
        "source_feature_means": {},
        "pooled_holdout": {},
        "cross_corpus": {},
    }

    for k,(X,S,M) in blocks.items():
        results["source_feature_means"][k] = {
            FEATURE_NAMES[i]: float(np.mean(X[:,i])) for i in range(X.shape[1])
        }
        results["source_feature_means"][k]["static_final_centroid_distance"] = float(np.mean(S))

    # Pooled stratified-ish split done within each source.
    Xtr=[];ytr=[];Str=[];Xte=[];yte=[];Ste=[]
    for k,(X,S,M) in blocks.items():
        tr,te = split_indices(len(X),0.6,SEED + len(k))
        lab = 0 if k=="benign" else 1
        Xtr.append(X[tr]); ytr.extend([lab]*len(tr)); Str.append(S[tr])
        Xte.append(X[te]); yte.extend([lab]*len(te)); Ste.append(S[te])
    Xtr=np.vstack(Xtr); Xte=np.vstack(Xte); ytr=np.array(ytr); yte=np.array(yte)
    Str=np.concatenate(Str); Ste=np.concatenate(Ste)

    _, temporal_score, temporal_m = fit_eval(Xtr,ytr,Xte,yte)
    # Static comparator: learn sign/threshold from training.
    _, static_score, static_m = fit_eval(Str.reshape(-1,1),ytr,Ste.reshape(-1,1),yte)
    combo_tr = np.column_stack([Xtr,Str])
    combo_te = np.column_stack([Xte,Ste])
    _, combo_score, combo_m = fit_eval(combo_tr,ytr,combo_te,yte)
    results["pooled_holdout"] = {
        "temporal_only": temporal_m,
        "static_final_centroid_only": static_m,
        "temporal_plus_static": combo_m,
        "delta_auroc_temporal_plus_static_vs_static": combo_m["auroc"]-static_m["auroc"],
    }

    # Cross-corpus: unseen attack source at test. Benign is independently split.
    Bx,Bs,_ = blocks["benign"]
    btr,bte = split_indices(len(Bx),0.6,SEED)
    for held in attack_sources:
        train_attack = [k for k in attack_sources if k != held]
        X_train = np.vstack([blocks[k][0] for k in train_attack] + [Bx[btr]])
        S_train = np.concatenate([blocks[k][1] for k in train_attack] + [Bs[btr]])
        y_train = np.concatenate([
            np.ones(sum(len(blocks[k][0]) for k in train_attack),dtype=int),
            np.zeros(len(btr),dtype=int)
        ])
        Xa,Sa,_ = blocks[held]
        X_test = np.vstack([Xa,Bx[bte]])
        S_test = np.concatenate([Sa,Bs[bte]])
        y_test = np.concatenate([np.ones(len(Xa),dtype=int),np.zeros(len(bte),dtype=int)])

        _,ts,tm = fit_eval(X_train,y_train,X_test,y_test)
        _,ss,sm = fit_eval(S_train.reshape(-1,1),y_train,S_test.reshape(-1,1),y_test)
        _,cs,cm = fit_eval(np.column_stack([X_train,S_train]),y_train,
                           np.column_stack([X_test,S_test]),y_test)
        results["cross_corpus"][f"held_out_{held}"] = {
            "train_attack_sources": train_attack,
            "test_attack_source": held,
            "temporal_only": tm,
            "static_final_centroid_only": sm,
            "temporal_plus_static": cm,
            "delta_auroc_temporal_vs_static": tm["auroc"]-sm["auroc"],
            "delta_auroc_combined_vs_static": cm["auroc"]-sm["auroc"],
        }

    Path("temporal_geometry_results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Multi-dataset audit of the fixed Bendex centroid-distance hypothesis.

Question:
    Are prompt-injection attacks consistently farther from the production
    benign centroid than benign prompts?

The production centroid prompt list is parsed directly from arc_gate.py on main.
No Arc Gate routing, phrase detection, TF-IDF, LLM judge, authority state,
session geometry, or thresholds are used.

Outputs:
  centroid_multidataset_results.json
  centroid_distances.csv
"""

from __future__ import annotations
import ast, csv, io, json, math, urllib.request
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

ARC_GATE_RAW = "https://raw.githubusercontent.com/9hannahnine-jpg/arc-gate/main/arc_gate.py"
INJEC_RAW = "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data"
SEED = 42
MAX_PER_SOURCE = 3000

def get_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()

def stable_unique(xs):
    return list(dict.fromkeys(x.strip() for x in xs if isinstance(x, str) and x.strip()))

def sample_cap(xs, n=MAX_PER_SOURCE, seed=SEED):
    xs = stable_unique(xs)
    if len(xs) <= n:
        return xs
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(xs), n, replace=False)
    return [xs[i] for i in idx]

def production_clean_prompts():
    src = get_bytes(ARC_GATE_RAW).decode("utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_GEO_CLEAN_PROMPTS":
                    val = ast.literal_eval(node.value)
                    return stable_unique(val)
    raise RuntimeError("_GEO_CLEAN_PROMPTS not found")

def load_injecagent_attacks():
    files = [
        "test_cases_dh_base.json","test_cases_ds_base.json",
        "test_cases_dh_enhanced.json","test_cases_ds_enhanced.json",
    ]
    out = []
    for fn in files:
        rows = json.loads(get_bytes(f"{INJEC_RAW}/{fn}").decode("utf-8"))
        out.extend(str(r.get("Tool Response","")) for r in rows)
    return sample_cap(out)

def dataset_rows(name, config=None):
    from datasets import load_dataset
    ds = load_dataset(name, config) if config else load_dataset(name)
    rows = []
    if hasattr(ds, "items"):
        for split, part in ds.items():
            for r in part:
                rr = dict(r)
                rr["_split"] = split
                rows.append(rr)
    else:
        for r in ds:
            rows.append(dict(r))
    return rows

def text_field(row):
    for k in ("text","prompt","instruction","content","query","input"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # no_robots message list
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                v = m.get("content")
                if isinstance(v, str) and v.strip():
                    return v.strip()
    return None

def load_labeled_hf(name, config=None):
    attacks, benign = [], []
    for r in dataset_rows(name, config):
        txt = text_field(r)
        if not txt:
            continue
        lab = r.get("label")
        if isinstance(lab, str):
            ll = lab.lower()
            if ll in {"1","attack","jailbreak","injection","malicious","harmful"}:
                attacks.append(txt)
            elif ll in {"0","benign","safe","normal"}:
                benign.append(txt)
        elif lab is not None:
            try:
                if int(lab) == 1:
                    attacks.append(txt)
                elif int(lab) == 0:
                    benign.append(txt)
            except Exception:
                pass
    return sample_cap(attacks), sample_cap(benign)

def load_jackhhao():
    attacks, benign = [], []
    for r in dataset_rows("jackhhao/jailbreak-classification"):
        txt = text_field(r)
        typ = str(r.get("type", "")).lower()
        if not txt:
            continue
        if typ == "jailbreak":
            attacks.append(txt)
        elif typ == "benign":
            benign.append(txt)
    return sample_cap(attacks), sample_cap(benign)

def load_no_robots():
    rows = dataset_rows("HuggingFaceH4/no_robots")
    return sample_cap([text_field(r) for r in rows if text_field(r)])

def load_alpaca():
    rows = dataset_rows("tatsu-lab/alpaca")
    return sample_cap([str(r.get("instruction","")).strip() for r in rows])

def load_dolly():
    rows = dataset_rows("databricks/databricks-dolly-15k")
    return sample_cap([str(r.get("instruction","")).strip() for r in rows])

def summary(x):
    x = np.asarray(x, dtype=float)
    qs = np.quantile(x, [0.01,0.05,0.25,0.5,0.75,0.95,0.99])
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q01": float(qs[0]), "q05": float(qs[1]), "q25": float(qs[2]),
        "median": float(qs[3]), "q75": float(qs[4]), "q95": float(qs[5]),
        "q99": float(qs[6]), "max": float(np.max(x)),
    }

def auc_pair(att, ben):
    y = np.array([1]*len(att) + [0]*len(ben))
    s = np.array(list(att)+list(ben), dtype=float)
    auc = float(roc_auc_score(y, s))
    return {
        "auc_attack_farther": auc,
        "auc_attack_closer": 1.0 - auc,
        "mean_attack_minus_benign": float(np.mean(att)-np.mean(ben)),
        "median_attack_minus_benign": float(np.median(att)-np.median(ben)),
    }

def main():
    print("Parsing production centroid prompts...")
    clean_ref = production_clean_prompts()
    print("production centroid prompts:", len(clean_ref))

    sources = {}
    failures = {}

    try:
        sources["attack_injecagent"] = load_injecagent_attacks()
    except Exception as e:
        failures["attack_injecagent"] = repr(e)

    for dsname, keyprefix, config in [
        ("deepset/prompt-injections","deepset",None),
        ("protectai/prompt-injection-validation","protectai",None),
        ("neuralchemy/Prompt-injection-dataset","neuralchemy","core"),
    ]:
        try:
            a,b = load_labeled_hf(dsname, config)
            sources[f"attack_{keyprefix}"] = a
            sources[f"benign_{keyprefix}"] = b
        except Exception as e:
            failures[keyprefix] = repr(e)

    try:
        a,b = load_jackhhao()
        sources["attack_jackhhao"] = a
        sources["benign_jackhhao"] = b
    except Exception as e:
        failures["jackhhao"] = repr(e)

    for name, loader in [
        ("benign_no_robots", load_no_robots),
        ("benign_alpaca", load_alpaca),
        ("benign_dolly", load_dolly),
    ]:
        try:
            sources[name] = loader()
        except Exception as e:
            failures[name] = repr(e)

    # Remove empty sources.
    sources = {k:v for k,v in sources.items() if v}
    print("sources:", {k:len(v) for k,v in sources.items()})
    if failures:
        print("load failures:", failures)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ref = model.encode(clean_ref, convert_to_numpy=True, normalize_embeddings=True)
    centroid = ref.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12

    distances = {}
    csv_rows = []
    for name, texts in sources.items():
        print("embedding", name, len(texts))
        emb = model.encode(texts, batch_size=64, convert_to_numpy=True,
                           normalize_embeddings=True, show_progress_bar=False)
        cos = np.clip(emb @ centroid, -1+1e-7, 1-1e-7)
        d = np.arccos(cos)
        distances[name] = d
        for i,(t,dist) in enumerate(zip(texts,d)):
            csv_rows.append({
                "source": name,
                "label": 1 if name.startswith("attack_") else 0,
                "distance": float(dist),
                "text": t.replace("\n"," ")[:500],
            })

    stats = {k: summary(v) for k,v in distances.items()}
    attack_keys = [k for k in distances if k.startswith("attack_")]
    benign_keys = [k for k in distances if k.startswith("benign_")]
    pairs = {}
    for ak in attack_keys:
        for bk in benign_keys:
            pairs[f"{ak}__vs__{bk}"] = auc_pair(distances[ak], distances[bk])

    # pooled comparisons
    all_attack = np.concatenate([distances[k] for k in attack_keys])
    all_benign = np.concatenate([distances[k] for k in benign_keys])
    pooled = auc_pair(all_attack, all_benign)

    result = {
        "hypothesis": "attacks should have larger angular distance from the production clean centroid",
        "centroid_source": "current _GEO_CLEAN_PROMPTS parsed from arc_gate.py main",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "source_counts": {k:len(v) for k,v in sources.items()},
        "load_failures": failures,
        "distance_stats": stats,
        "pairwise": pairs,
        "pooled": pooled,
        "interpretation_key": {
            "auc_attack_farther_gt_0_5": "supports intended production sign",
            "auc_attack_farther_lt_0_5": "attacks are closer to centroid than benign prompts",
            "auc_near_0_5": "centroid distance has little ranking information",
        },
    }
    Path("centroid_multidataset_results.json").write_text(json.dumps(result, indent=2))

    with open("centroid_distances.csv","w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source","label","distance","text"])
        w.writeheader()
        w.writerows(csv_rows)

    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

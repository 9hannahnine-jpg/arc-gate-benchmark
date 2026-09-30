#!/usr/bin/env python3
"""
One-class temporal geometry audit.

This is the deployment-realistic test:
- NO attack examples are used for fitting or threshold selection.
- Fit only on benign multi-turn conversations.
- Calibrate thresholds on held-out benign validation traffic.
- Evaluate realized FPR on untouched benign test traffic.
- Evaluate TPR on multiple independent attack corpora.

All conversations are reduced to exactly the final 3 user turns so conversation
length cannot leak attack labels.

Temporal observables come only from MiniLM embeddings of successive user turns.
No phrase flags, TF-IDF, authority state, LLM judge, or production tau.

Score:
  Ledoit-Wolf Mahalanobis distance in standardized temporal-feature space.

Comparators:
  static final-turn distance from the fixed production benign centroid.
"""

from __future__ import annotations
import ast, json, math, random, re, urllib.request
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

SEED = 42
MAX_PER_SOURCE = 400
MIN_TURNS = 3

ARC_GATE_RAW = "https://raw.githubusercontent.com/9hannahnine-jpg/arc-gate/main/arc_gate.py"

ATTACK_DATASETS = [
    ("tom_gibbs", "tom-gibbs/multi-turn_jailbreak_attack_datasets", None),
    ("harm_geometry", "yelyzavetahusieva/geometry-of-harmfulness-in-multi-turn-attacks", None),
]

# Try multiple independent benign conversation domains. Failures are recorded,
# not silently ignored.
BENIGN_DATASETS = [
    ("ultrachat", "HuggingFaceH4/ultrachat_200k", None),
    ("chatbot_arena", "lmsys/chatbot_arena_conversations", None),
    ("sharegpt", "Aeala/ShareGPT_Vicuna_unfiltered", None),
]

def get_bytes(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()

def production_clean_prompts():
    src = get_bytes(ARC_GATE_RAW).decode("utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "_GEO_CLEAN_PROMPTS" for t in node.targets):
                return list(ast.literal_eval(node.value))
    raise RuntimeError("_GEO_CLEAN_PROMPTS not found")

def load_dataset_rows(name, config=None, max_rows=MAX_PER_SOURCE*8):
    from datasets import load_dataset
    ds = load_dataset(name, config) if config else load_dataset(name)
    rows = []
    if hasattr(ds, "items"):
        order = sorted(ds.keys(), key=lambda k: (0 if "test" in k else 1 if ("valid" in k or "val" in k) else 2, k))
        for split in order:
            for r in ds[split]:
                rr = dict(r); rr["_split"] = split
                rows.append(rr)
                if len(rows) >= max_rows:
                    return rows
    else:
        for r in ds:
            rows.append(dict(r))
            if len(rows) >= max_rows:
                break
    return rows

def maybe_parse(v):
    if isinstance(v, (list,dict)):
        return v
    if isinstance(v,str):
        s=v.strip()
        if s and s[0] in "[{":
            for parser in (json.loads, ast.literal_eval):
                try: return parser(s)
                except Exception: pass
    return v

def turns_from_obj(obj):
    obj = maybe_parse(obj)
    if isinstance(obj,list):
        if obj and all(isinstance(x,dict) for x in obj):
            turns=[]
            for m in obj:
                role=str(m.get("role",m.get("from",m.get("speaker","")))).lower()
                content=m.get("content",m.get("value",m.get("text",m.get("message",""))))
                if role in {"user","human","client"} and isinstance(content,str) and content.strip():
                    turns.append(content.strip())
            if len(turns)>=MIN_TURNS:
                return turns
        if obj and all(isinstance(x,str) for x in obj):
            out=[x.strip() for x in obj if x.strip()]
            if len(out)>=MIN_TURNS: return out
        best=[]
        for x in obj:
            c=turns_from_obj(x)
            if len(c)>len(best): best=c
        return best
    if isinstance(obj,dict):
        for k in ("turns","messages","conversation","conversations","dialogue","chat","history","Multi-turn Conversation","multi_turn_conversation"):
            if k in obj:
                c=turns_from_obj(obj[k])
                if len(c)>=MIN_TURNS: return c
        best=[]
        for v in obj.values():
            c=turns_from_obj(v)
            if len(c)>len(best): best=c
        return best
    return []

def parse_sharegpt_text(text):
    # Fallback for flattened ShareGPT-style transcripts.
    if not isinstance(text,str): return []
    parts = re.split(r"(?:###\s*)?(?:Human|USER|User)\s*:", text)
    turns=[]
    for p in parts[1:]:
        p=re.split(r"(?:###\s*)?(?:Assistant|ASSISTANT|Assistant)\s*:",p)[0].strip()
        if p: turns.append(p)
    return turns

def extract_session(row):
    for k in ("messages","conversation","conversations","turns","dialogue","chat","history","Multi-turn Conversation","multi_turn_conversation"):
        if k in row:
            c=turns_from_obj(row[k])
            if len(c)>=MIN_TURNS: return c
    # flattened transcript fallbacks
    for k in ("text","content","prompt"):
        c=parse_sharegpt_text(row.get(k))
        if len(c)>=MIN_TURNS: return c
    best=[]
    for v in row.values():
        c=turns_from_obj(v)
        if len(c)>len(best): best=c
    return best if len(best)>=MIN_TURNS else None

def load_sessions(name, config=None, seed=SEED):
    rows=load_dataset_rows(name,config)
    ss=[]
    seen=set()
    for r in rows:
        s=extract_session(r)
        if not s: continue
        # fixed 3-turn window
        s=s[-3:]
        key=tuple(x[:160] for x in s)
        if key not in seen:
            seen.add(key); ss.append(s)
    rng=random.Random(seed); rng.shuffle(ss)
    return ss[:MAX_PER_SOURCE]

def angle(a,b):
    return math.acos(np.clip(float(np.dot(a,b)),-1+1e-7,1-1e-7))

FEATURE_NAMES = [
    "mean_step","std_step","max_step","last_step",
    "endpoint_displacement","tortuosity",
    "mean_abs_acceleration","max_abs_acceleration",
    "late_early_ratio","step_slope","mean_turn_angle","max_turn_angle",
]

def temporal_features(E):
    steps=np.array([angle(E[i-1],E[i]) for i in range(1,len(E))],float)
    path=float(np.sum(steps))
    endpoint=angle(E[0],E[-1])
    accel=np.diff(steps) if len(steps)>1 else np.array([0.0])
    x=np.arange(len(steps),dtype=float)
    slope=float(np.polyfit(x,steps,1)[0]) if len(steps)>1 else 0.0
    half=max(1,len(steps)//2)
    early=float(np.mean(steps[:half])); late=float(np.mean(steps[-half:]))
    ratio=late/(early+1e-8)
    turns=[]
    for i in range(1,len(E)-1):
        v1=E[i]-E[i-1]; v2=E[i+1]-E[i]
        n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
        if n1>1e-8 and n2>1e-8:
            turns.append(math.acos(np.clip(float(np.dot(v1,v2)/(n1*n2)),-1+1e-7,1-1e-7)))
    return np.array([
        float(np.mean(steps)),float(np.std(steps)),float(np.max(steps)),float(steps[-1]),
        endpoint,path/(endpoint+1e-8),
        float(np.mean(np.abs(accel))),float(np.max(np.abs(accel))),
        ratio,slope,float(np.mean(turns)) if turns else 0.0,float(np.max(turns)) if turns else 0.0
    ])

def featurize(model,sessions,centroid):
    flat=[t for s in sessions for t in s]
    Eall=model.encode(flat,batch_size=128,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False)
    X=[]; static=[]
    pos=0
    for s in sessions:
        E=Eall[pos:pos+3]; pos+=3
        X.append(temporal_features(E))
        static.append(angle(E[-1],centroid))
    return np.vstack(X),np.array(static)

def split3(n,seed):
    rng=np.random.default_rng(seed)
    idx=np.arange(n); rng.shuffle(idx)
    a=int(.5*n); b=int(.75*n)
    return idx[:a],idx[a:b],idx[b:]

class OneClassTemporal:
    def fit(self,X):
        self.scaler=StandardScaler().fit(X)
        Z=self.scaler.transform(X)
        self.cov=LedoitWolf().fit(Z)
        return self
    def score(self,X):
        Z=self.scaler.transform(X)
        return self.cov.mahalanobis(Z)

def quantile_threshold(scores,target_fpr):
    # threshold chosen only on benign validation traffic
    return float(np.quantile(scores,1.0-target_fpr,method="higher"))

def evaluate_at_threshold(ben_scores,attack_scores,th):
    return {
        "realized_fpr":float(np.mean(ben_scores>th)),
        "tpr":float(np.mean(attack_scores>th)),
        "threshold":float(th),
        "n_benign_test":int(len(ben_scores)),
        "n_attack":int(len(attack_scores)),
    }

def static_anomaly_scores(train_static, X):
    mu=float(np.mean(train_static)); sd=float(np.std(train_static)+1e-8)
    return np.abs((X-mu)/sd)

def main():
    from sentence_transformers import SentenceTransformer

    failures={}
    attacks={}
    benign={}

    for key,name,cfg in ATTACK_DATASETS:
        try:
            ss=load_sessions(name,cfg,SEED+len(key))
            if ss: attacks[key]=ss
            else: failures[key]="no parseable sessions"
        except Exception as e:
            failures[key]=repr(e)

    for key,name,cfg in BENIGN_DATASETS:
        try:
            ss=load_sessions(name,cfg,SEED+len(key))
            if len(ss)>=80: benign[key]=ss
            else: failures[key]=f"only {len(ss)} parseable sessions"
        except Exception as e:
            failures[key]=repr(e)

    if len(attacks)<2: raise RuntimeError("Need >=2 attack corpora")
    if len(benign)<1: raise RuntimeError("Need >=1 benign corpus")

    print("counts", {k:len(v) for k,v in attacks.items()}, {k:len(v) for k,v in benign.items()})
    print("failures",failures)

    model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    ref=model.encode(production_clean_prompts(),convert_to_numpy=True,normalize_embeddings=True)
    C=ref.mean(axis=0); C=C/(np.linalg.norm(C)+1e-12)

    A={}
    for k,ss in attacks.items(): A[k]=featurize(model,ss,C)
    B={}
    for k,ss in benign.items(): B[k]=featurize(model,ss,C)

    results={
        "protocol":{
            "fit_data":"benign only",
            "threshold_selection":"held-out benign validation only",
            "window":"exactly final 3 user turns",
            "temporal_score":"LedoitWolf Mahalanobis distance in standardized temporal feature space",
            "feature_names":FEATURE_NAMES,
            "attack_sources":list(attacks),
            "benign_sources":list(benign),
            "load_failures":failures,
        },
        "within_domain_calibration":{},
        "pooled_benign_calibration":{},
        "leave_one_benign_domain_out":{},
    }

    # Deployment-specific calibration: each benign domain independently.
    for bk,(Xb,Sb) in B.items():
        tr,va,te=split3(len(Xb),SEED+len(bk))
        det=OneClassTemporal().fit(Xb[tr])
        bval=det.score(Xb[va]); btest=det.score(Xb[te])
        sval=static_anomaly_scores(Sb[tr],Sb[va]); stest=static_anomaly_scores(Sb[tr],Sb[te])
        row={"temporal":{},"static":{}}
        for fpr in (0.01,0.001):
            th=quantile_threshold(bval,fpr)
            sth=quantile_threshold(sval,fpr)
            row["temporal"][str(fpr)]={"benign_realized_fpr":float(np.mean(btest>th)),"threshold":th,"attacks":{}}
            row["static"][str(fpr)]={"benign_realized_fpr":float(np.mean(stest>sth)),"threshold":sth,"attacks":{}}
            for ak,(Xa,Sa) in A.items():
                row["temporal"][str(fpr)]["attacks"][ak]=float(np.mean(det.score(Xa)>th))
                row["static"][str(fpr)]["attacks"][ak]=float(np.mean(static_anomaly_scores(Sb[tr],Sa)>sth))
        results["within_domain_calibration"][bk]=row

    # Pooled benign calibration.
    splitB={}
    for bk,(Xb,Sb) in B.items():
        splitB[bk]=split3(len(Xb),SEED+100+len(bk))
    Xtr=np.vstack([B[k][0][splitB[k][0]] for k in B])
    Xva=np.vstack([B[k][0][splitB[k][1]] for k in B])
    Xte=np.vstack([B[k][0][splitB[k][2]] for k in B])
    Str=np.concatenate([B[k][1][splitB[k][0]] for k in B])
    Sva=np.concatenate([B[k][1][splitB[k][1]] for k in B])
    Ste=np.concatenate([B[k][1][splitB[k][2]] for k in B])
    det=OneClassTemporal().fit(Xtr)
    bval=det.score(Xva); btest=det.score(Xte)
    sval=static_anomaly_scores(Str,Sva); stest=static_anomaly_scores(Str,Ste)
    pooled={"temporal":{},"static":{}}
    for fpr in (0.01,0.001):
        th=quantile_threshold(bval,fpr); sth=quantile_threshold(sval,fpr)
        pooled["temporal"][str(fpr)]={"benign_realized_fpr":float(np.mean(btest>th)),"attacks":{}}
        pooled["static"][str(fpr)]={"benign_realized_fpr":float(np.mean(stest>sth)),"attacks":{}}
        for ak,(Xa,Sa) in A.items():
            pooled["temporal"][str(fpr)]["attacks"][ak]=float(np.mean(det.score(Xa)>th))
            pooled["static"][str(fpr)]["attacks"][ak]=float(np.mean(static_anomaly_scores(Str,Sa)>sth))
    results["pooled_benign_calibration"]=pooled

    # Leave-one-benign-domain-out: tests FPR transfer to unseen benign style.
    if len(B)>=2:
        for held in B:
            others=[k for k in B if k!=held]
            Xtrain=np.vstack([B[k][0][:max(1,int(.65*len(B[k][0])))] for k in others])
            Xval=np.vstack([B[k][0][max(1,int(.65*len(B[k][0]))):] for k in others])
            Strain=np.concatenate([B[k][1][:max(1,int(.65*len(B[k][1])))] for k in others])
            Sval2=np.concatenate([B[k][1][max(1,int(.65*len(B[k][1]))):] for k in others])
            det2=OneClassTemporal().fit(Xtrain)
            v=det2.score(Xval)
            sv=static_anomaly_scores(Strain,Sval2)
            Xheld,Sheld=B[held]
            out={"temporal":{},"static":{}}
            for fpr in (0.01,0.001):
                th=quantile_threshold(v,fpr); sth=quantile_threshold(sv,fpr)
                out["temporal"][str(fpr)]={"unseen_benign_fpr":float(np.mean(det2.score(Xheld)>th)),"attacks":{}}
                out["static"][str(fpr)]={"unseen_benign_fpr":float(np.mean(static_anomaly_scores(Strain,Sheld)>sth)),"attacks":{}}
                for ak,(Xa,Sa) in A.items():
                    out["temporal"][str(fpr)]["attacks"][ak]=float(np.mean(det2.score(Xa)>th))
                    out["static"][str(fpr)]["attacks"][ak]=float(np.mean(static_anomaly_scores(Strain,Sa)>sth))
            results["leave_one_benign_domain_out"][held]=out

    Path("temporal_oneclass_results.json").write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=="__main__":
    main()

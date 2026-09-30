#!/usr/bin/env python3
"""
Conventional Arc Gate vs external prompt-injection baselines.

Scope
-----
This intentionally removes every Bendex geometry component.

Arc Gate offline conventional core:
  1) production phrase/normalization detector parsed from arc_gate.py
  2) production TF-IDF classifier artifacts from arc-gate/main
  3) max(phrase_hit, tfidf_probability)

Not included:
  - centroid / Fisher-Rao claims
  - session tau / trajectory geometry
  - Mahalanobis geometry
  - LLM judge (requires paid API and is not an offline detector)
  - authority state machine (stateful policy engine, not a scalar single-prompt classifier)
  - Arc Sentry BehavioralFilter (primarily harmful-content classifier; reported separately in its own repo)

External model baselines:
  - protectai/deberta-v3-base-prompt-injection-v2
  - Keshav0av/deberta-v3-prompt-injection-detector

Datasets:
  - neuralchemy/Prompt-injection-dataset core TEST split only
  - jackhhao/jailbreak-classification

Threshold-independent metrics:
  AUROC, AUPRC, TPR@1%FPR, TPR@0.1%FPR

We also report direct production Arc Gate decisions:
  phrase OR TF-IDF >= current production block threshold if recoverable,
  otherwise only score-based metrics.

Caveat:
  Public security datasets can share upstream examples with public model training
  corpora. This benchmark establishes external performance relative to Arc Gate's
  own published benchmark, not proof of zero upstream-source overlap for every
  third-party baseline.
"""

from __future__ import annotations
import ast, base64, codecs, io, json, pickle, re, unicodedata, urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

ARC_RAW = "https://raw.githubusercontent.com/9hannahnine-jpg/arc-gate/main"
BASELINES = [
    "protectai/deberta-v3-base-prompt-injection-v2",
]

def get_bytes(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()

def get_arc_source():
    return get_bytes(f"{ARC_RAW}/arc_gate.py").decode("utf-8")

def parse_assignment(src, name):
    tree=ast.parse(src)
    for node in tree.body:
        if isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            if any(isinstance(t,ast.Name) and t.id==name for t in targets):
                try: return ast.literal_eval(node.value)
                except Exception: return None
    return None

def production_phrases(src):
    val=parse_assignment(src,"INJECTION_PHRASES")
    if not isinstance(val,list):
        raise RuntimeError("Could not parse INJECTION_PHRASES")
    return [str(x).lower() for x in val]

def phrase_hit(prompt, phrases):
    def norm(s):
        return unicodedata.normalize("NFKD",s).encode("ascii","ignore").decode("ascii")
    def variants(s):
        v=[s,norm(s)]
        try:
            b=base64.b64decode(s.strip()+"==").decode("utf-8",errors="ignore")
            if len(b)>4:v.append(b)
        except Exception:pass
        try:
            h=bytes.fromhex(s.replace(" ","").replace(":","")).decode("utf-8",errors="ignore")
            if len(h)>4:v.append(h)
        except Exception:pass
        try:v.append(codecs.decode(s,"rot13"))
        except Exception:pass
        v.append(s.replace("0","o").replace("1","i").replace("3","e").replace("4","a").replace("@","a").replace("$","s"))
        return v
    for variant in variants(prompt):
        pl=variant.lower()
        pn=pl.replace(" ","").replace("_","")
        pc=" ".join(pl.split())
        for ph in phrases:
            if ph in pl or ph.replace(" ","") in pn or ph in pc:
                return 1.0
    return 0.0

def load_pickle(name):
    return pickle.load(io.BytesIO(get_bytes(f"{ARC_RAW}/models/{name}")))

def arc_scores(texts, phrases):
    print("loading Arc Gate TF-IDF artifacts")
    cv=load_pickle("tfidf_char_vec.pkl")
    wv=load_pickle("tfidf_word_vec.pkl")
    clf=load_pickle("tfidf_clf2.pkl")
    X=hstack([cv.transform(texts),wv.transform(texts)])
    tf=np.asarray(clf.predict_proba(X)[:,1],dtype=float)
    ph=np.asarray([phrase_hit(t,phrases) for t in texts],dtype=float)
    return ph,tf,np.maximum(ph,tf)

def get_text(row):
    for k in ("text","prompt","instruction","content","query","input"):
        v=row.get(k)
        if isinstance(v,str) and v.strip():return v.strip()
    return None

def stratified_cap(texts, labels, max_n=600, seed=42):
    labels=np.asarray(labels,dtype=int)
    if len(labels)<=max_n:
        return list(texts),labels
    rng=np.random.default_rng(seed)
    pos=np.where(labels==1)[0]; neg=np.where(labels==0)[0]
    npos=min(len(pos),max_n//2); nneg=min(len(neg),max_n-npos)
    # If one class is short, give remainder to the other.
    if npos+nneg<max_n:
        if len(pos)>npos:
            npos=min(len(pos),max_n-nneg)
        if npos+nneg<max_n and len(neg)>nneg:
            nneg=min(len(neg),max_n-npos)
    idx=np.concatenate([rng.choice(pos,npos,False),rng.choice(neg,nneg,False)])
    rng.shuffle(idx)
    return [texts[i] for i in idx],labels[idx]

def load_external_sets():
    from datasets import load_dataset
    out={}

    # Frozen test split only. Never touch neuralchemy train/validation.
    ds=load_dataset("neuralchemy/Prompt-injection-dataset","core",split="test")
    texts=[]; labels=[]
    for r in ds:
        t=get_text(r)
        if t is not None:
            texts.append(t); labels.append(int(r["label"]))
    texts,labels=stratified_cap(texts,labels,200,42)
    out["neuralchemy_core_test"]=(texts,np.asarray(labels,dtype=int))

    ds2=load_dataset("jackhhao/jailbreak-classification")
    texts=[];labels=[]
    parts=ds2.values() if hasattr(ds2,"values") else [ds2]
    for part in parts:
        for r in part:
            t=get_text(r)
            typ=str(r.get("type",r.get("label",""))).lower()
            if not t:continue
            if typ=="jailbreak" or typ in {"1","attack","malicious"}:
                texts.append(t);labels.append(1)
            elif typ=="benign" or typ in {"0","safe","normal"}:
                texts.append(t);labels.append(0)
    texts,labels=stratified_cap(texts,labels,200,43)
    out["jackhhao"]=(texts,np.asarray(labels,dtype=int))
    return out

def find_injection_index(model):
    labels={int(k):str(v).lower() for k,v in model.config.id2label.items()}
    for i,s in labels.items():
        if any(w in s for w in ("inject","jailbreak","malicious","attack","unsafe")):
            return i
    # Known binary convention for listed baselines.
    return 1

def transformer_scores(model_name,texts,batch=32):
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    print("loading baseline",model_name)
    tok=AutoTokenizer.from_pretrained(model_name)
    model=AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    idx=find_injection_index(model)
    scores=[]
    with torch.inference_mode():
        for i in range(0,len(texts),batch):
            enc=tok(texts[i:i+batch],padding=True,truncation=True,max_length=512,return_tensors="pt")
            logits=model(**enc).logits
            p=torch.softmax(logits,dim=-1)[:,idx]
            scores.extend(p.cpu().numpy().tolist())
    del model
    return np.asarray(scores,dtype=float)

def tpr_at_fpr(y,s,target):
    fpr,tpr,_=roc_curve(y,s)
    ok=np.where(fpr<=target)[0]
    return float(np.max(tpr[ok])) if len(ok) else 0.0

def metrics(y,s):
    return {
        "auroc":float(roc_auc_score(y,s)),
        "auprc":float(average_precision_score(y,s)),
        "tpr_at_1pct_fpr":tpr_at_fpr(y,s,0.01),
        "tpr_at_0_1pct_fpr":tpr_at_fpr(y,s,0.001),
    }

def bootstrap_delta(y,a,b,n=1000,seed=123):
    rng=np.random.default_rng(seed)
    pos=np.where(y==1)[0];neg=np.where(y==0)[0]
    out={"auroc":[],"tpr1":[]}
    for _ in range(n):
        idx=np.concatenate([rng.choice(pos,len(pos),True),rng.choice(neg,len(neg),True)])
        yy=y[idx]; aa=a[idx]; bb=b[idx]
        out["auroc"].append(roc_auc_score(yy,aa)-roc_auc_score(yy,bb))
        out["tpr1"].append(tpr_at_fpr(yy,aa,0.01)-tpr_at_fpr(yy,bb,0.01))
    def summarize(v):
        return {"mean":float(np.mean(v)),"ci95":[float(x) for x in np.quantile(v,[.025,.975])]}
    return {k:summarize(v) for k,v in out.items()}

def main():
    src=get_arc_source()
    phrases=production_phrases(src)
    datasets=load_external_sets()
    result={
        "scope":"Arc Gate offline conventional core only; all geometry removed",
        "arc_components":["production phrase detector","production TF-IDF probability","max ensemble"],
        "excluded":["all Bendex geometry","LLM judge","stateful authority engine","Arc Sentry harmful-content filter"],
        "baseline_models":BASELINES,
        "datasets":{},
    }

    for dname,(texts,y) in datasets.items():
        print("\nDATASET",dname,"n",len(y),"attacks",int(y.sum()),"benign",int((1-y).sum()))
        ph,tf,arc=arc_scores(texts,phrases)
        scores={
            "arc_phrase":ph,
            "arc_tfidf":tf,
            "arc_conventional_core":arc,
        }
        for m in BASELINES:
            try:
                scores[m]=transformer_scores(m,texts)
            except Exception as e:
                print("BASELINE FAILURE",m,repr(e))
        row={
            "n":int(len(y)),"n_attack":int(y.sum()),"n_benign":int((1-y).sum()),
            "metrics":{k:metrics(y,v) for k,v in scores.items()},
            "pairwise_arc_minus_baseline":{},
        }
        for m in BASELINES:
            if m in scores:
                row["pairwise_arc_minus_baseline"][m]=bootstrap_delta(y,arc,scores[m])
        result["datasets"][dname]=row

    Path("conventional_baseline_results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    main()

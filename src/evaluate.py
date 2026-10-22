"""Blocked, stratified group cross-validation + window / segment / alarm metrics."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold

from .data import SAMPLE_DT
from .features import feature_columns
from .pipeline import fit_bundle, predict_bundle

MODEL_ORDER = ["B0_SPC", "B1_LogReg", "M1_RF", "M2_HGB", "M3_IF", "M4_Stack"]


def run_cv(w: pd.DataFrame, seed=0, n_splits=5, group_col="block", cols=None) -> pd.DataFrame:
    """Out-of-fold probabilities/predictions for every model.

    group_col="block" is the honest split (30 s time blocks never straddle train/test).
    group_col=None gives plain shuffled KFold (leaky; only used to demonstrate the leakage).
    """
    fc = feature_columns(w)
    X, y = w[fc].reset_index(drop=True), w["y"].to_numpy()
    g = w[group_col].to_numpy() if group_col else np.arange(len(w))
    if group_col:
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, g)
    else:
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    parts = []
    for k, (tr, te) in enumerate(splits):
        b = fit_bundle(X.iloc[tr].reset_index(drop=True), y[tr], g[tr], seed, cols)
        p = predict_bundle(b, X.iloc[te])
        for m in MODEL_ORDER:
            p[m + "_pred"] = (p[m] >= b["thr"][m]).astype(int)
        p["fold"], p["idx"] = k, te
        parts.append(p)
    oof = pd.concat(parts).sort_values("idx").set_index("idx")
    oof["y"] = y
    return oof


def window_metrics(y, p, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "PR_AUC": average_precision_score(y, p), "ROC_AUC": roc_auc_score(y, p),
        "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "Precision": prec, "Recall": rec, "FPR": fp / (fp + tn) if fp + tn else 0.0,
        "Brier": brier_score_loss(y, p), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


def segment_metrics(w: pd.DataFrame, pred) -> dict:
    """Event view: a segment alarms if any of its windows is flagged.

    Latency = seconds from segment start to the end of the first flagged window.
    False alarms/hour = alarming normal segments per hour of recorded normal operation.
    """
    d = w[["seg", "y", "pos_in_seg", "win_len", "seg_len"]].copy()
    d["pred"] = np.asarray(pred)
    d["t_end"] = d["pos_in_seg"] + d["win_len"] * SAMPLE_DT
    s = d.groupby("seg").agg(y=("y", "max"), alarm=("pred", "max"), seg_len=("seg_len", "first"))
    lat = d[d.pred == 1].groupby("seg")["t_end"].min()
    a, n = s[s.y == 1], s[s.y == 0]
    normal_hours = n.seg_len.sum() * SAMPLE_DT / 3600
    return {
        "seg_recall": a.alarm.mean(), "seg_false_alarm_rate": n.alarm.mean(),
        "latency_median_s": float(lat.reindex(a.index).median()),
        "latency_max_s": float(lat.reindex(a.index).max()),
        "false_alarms_per_hour": float(n.alarm.sum() / normal_hours),
    }


def summarise(oofs: list, w: pd.DataFrame) -> pd.DataFrame:
    """Mean and std over CV repeats of window + segment metrics per model."""
    rows = []
    for m in MODEL_ORDER:
        per = []
        for oof in oofs:
            r = window_metrics(oof["y"], oof[m], oof[m + "_pred"])
            r.update(segment_metrics(w, oof[m + "_pred"].to_numpy()))
            per.append(r)
        df = pd.DataFrame(per)
        row = {"model": m}
        for c in df.columns:
            row[c] = df[c].mean()
            row[c + "_sd"] = df[c].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)

"""Fit / predict a calibrated bundle: members -> Platt calibration -> logistic stacker, F1-optimal thresholds."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .models import CLIP, STACK, STACK_MEMBERS, CurrentGuard, ModeGuard, make_models


def _clip(m):
    return np.clip(np.asarray(m, float), -CLIP, CLIP)


def best_threshold(y, p) -> float:
    """F1-maximising threshold; ties -> median of tying candidates (centre of the plateau)."""
    y = np.asarray(y)
    cand = np.unique(p)
    cand = cand[(cand >= 0.05) & (cand <= 0.95)] if ((cand >= 0.05) & (cand <= 0.95)).any() else cand
    f1 = []
    for t in cand:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        f1.append(2 * tp / (pred.sum() + (y == 1).sum()) if pred.sum() else 0.0)
    f1 = np.asarray(f1)
    return float(np.median(cand[f1 >= f1.max() - 1e-12]))


def _inner_oof(factories, X, y, g, seed):
    Z = {k: np.zeros(len(y)) for k in factories}
    folds = StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(X, y, g)
    for a, b in folds:
        for k, f in factories.items():
            Z[k][b] = f().fit(X.iloc[a], y[a]).margin(X.iloc[b])
    return pd.DataFrame({k: _clip(v) for k, v in Z.items()})


def fit_bundle(X: pd.DataFrame, y, g, seed=0, cols=None, n_modes: int = 2) -> dict:
    y = np.asarray(y)
    fac = make_models(seed, cols)
    Z = _inner_oof(fac, X, y, g, seed)
    members = {k: f().fit(X, y) for k, f in fac.items()}
    platt = {k: LogisticRegression(C=100.0).fit(Z[[k]], y) for k in fac}
    stack = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    stack.fit(Z[STACK_MEMBERS], y)
    thr = {k: best_threshold(y, platt[k].predict_proba(Z[[k]])[:, 1]) for k in fac}
    thr[STACK] = best_threshold(y, stack.predict_proba(Z[STACK_MEMBERS])[:, 1])
    guard = ModeGuard(k=n_modes, seed=seed).fit(X, y)        # deployed: per-operating-mode limits (yellow only)
    guard_global = CurrentGuard().fit(X, y)                   # kept for the comparison table: one global observed-max limit
    return {"members": members, "platt": platt, "stack": stack, "thr": thr, "guard": guard, "guard_global": guard_global,
            "features": list(X.columns), "cols": cols, "policy": {}}


def predict_bundle(b: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Calibrated anomaly probability per model, plus `unc` = disagreement (std) of stack members
    `guard` = mode-specific peak-current flag (0/1, inside a known mode's support only), `guard_global` = the old single
    observed-max flag, `new_mode` = window outside every known mode's support (approval/retrain status, not an alarm)
    and `mode` = nearest known operating mode. The guard never changes a probability; the alarm policy
    (errors.assign_level) uses `guard` to raise a window to yellow."""
    X = X[b["features"]]
    Z = pd.DataFrame({k: _clip(m.margin(X)) for k, m in b["members"].items()}, index=X.index)
    out = pd.DataFrame({k: b["platt"][k].predict_proba(Z[[k]])[:, 1] for k in Z}, index=X.index)
    out[STACK] = b["stack"].predict_proba(Z[STACK_MEMBERS])[:, 1]
    out["unc"] = out[STACK_MEMBERS].std(axis=1)
    out["guard"] = b["guard"].flag(X)
    out["guard_global"] = b["guard_global"].flag(X)
    out["new_mode"] = b["guard"].new_mode(X)
    out["mode"] = b["guard"].mode(X)
    return out

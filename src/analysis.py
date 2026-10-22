"""Model-side analyses: operating modes, importance, interactions, ablations, mode-shift stress test."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold

from .data import CH
from .errors import _rule_table, alarm_policy, assign_level, bootstrap_ci, pooled
from .evaluate import MODEL_ORDER, run_cv, summarise, window_metrics
from .features import feature_columns, invariant_columns, make_windows
from .models import CurrentGuard, ModeGuard
from .pipeline import fit_bundle, predict_bundle


def _hgb(seed=0, balanced=True):
    return HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
        class_weight="balanced" if balanced else None, random_state=seed)


def feature_groups(cols) -> dict:
    ch = {"상부진동(AI0)": "vup_", "하부진동(AI1)": "vlow_", "전류(AI2)": "cur_"}
    g = {k: [c for c in cols if c.startswith(p)] for k, p in ch.items()}
    g["채널간 관계"] = [c for c in cols if c.startswith("vib_")]
    return g


def family_groups(cols) -> dict:
    fam = {"크기(mean/std/rms/ptp)": ("_mean", "_std", "_rms", "_ptp"),
           "형상(skew/kurt/crest)": ("_skew", "_kurt", "_crest"),
           "동특성(mad/madr/zcr)": ("_mad", "_madr", "_zcr")}
    out = {k: [c for c in cols if c.endswith(s)] for k, s in fam.items()}
    return {k: v for k, v in out.items() if v}


# ---------------------------------------------------------------- raw-signal cross-correlation
_PAIRS = [(CH[0], CH[1]), (CH[0], CH[2]), (CH[1], CH[2])]


def _xcorr_curve(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised cross-correlation of a,b at lags -max_lag..+max_lag (positive lag: b shifted later)."""
    a, b = a - a.mean(), b - b.mean()
    out = np.zeros(2 * max_lag + 1)
    for i, lag in enumerate(range(-max_lag, max_lag + 1)):
        x, y = (a[:len(a) - lag], b[lag:]) if lag >= 0 else (a[-lag:], b[:len(b) + lag])
        out[i] = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
    return out


def cross_correlation(df: pd.DataFrame, max_lag=10, min_len=30):
    """Per-segment peak cross-correlation (and its lag) between each raw channel pair, pooled separately
    over normal and anomaly segments long enough (>=min_len samples) for a stable estimate.

    Returns (summary table, {(label, pair): mean lag-correlation curve} for plotting).
    """
    seg_len = df.groupby("seg").size()
    good = seg_len[seg_len >= min_len].index
    rows, curves = [], {}
    for label, y in (("정상", 0), ("이상", 1)):
        segs = [s for s in good if df.loc[df.seg == s, "Equipment_state"].iloc[0] == y]
        for a, b in _PAIRS:
            peaks, lags, cs = [], [], []
            for s in segs:
                g = df[df.seg == s]
                c = _xcorr_curve(g[a].to_numpy(), g[b].to_numpy(), max_lag)
                cs.append(c)
                i = int(np.argmax(np.abs(c)))
                peaks.append(c[i])
                lags.append(i - max_lag)
            if not cs:
                continue
            curves[(label, f"{a}~{b}")] = np.mean(cs, axis=0)
            rows.append({"segment_type": label, "pair": f"{a}~{b}", "n_segments": len(segs),
                         "peak_lag_median": float(np.median(lags)), "mean_peak_corr": float(np.mean(peaks)),
                         "sd_peak_corr": float(np.std(peaks))})
    return pd.DataFrame(rows), curves


# ---------------------------------------------------------------- operating modes
def find_modes(w: pd.DataFrame, seed=0):
    """Cluster NORMAL windows on (log lower-vibration RMS, log current RMS).

    Returns (mode id per window ordered low->high vibration level, per-mode summary, silhouette per k).
    """
    z = np.log(w[["vlow_rms", "cur_rms"]].to_numpy())
    n = w["y"].to_numpy() == 0
    mu, sd = z[n].mean(0), z[n].std(0)
    zs = (z - mu) / sd
    sil = {}
    for k in (2, 3, 4):
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(zs[n])
        sil[k] = silhouette_score(zs[n], lab, sample_size=2000, random_state=seed)
    k = max(sil, key=sil.get)
    km = KMeans(k, n_init=10, random_state=seed).fit(zs[n])
    rank = {int(c): i for i, c in enumerate(np.argsort(km.cluster_centers_[:, 0]))}
    mode = np.array([rank[int(c)] for c in km.predict(zs)])
    info = w.assign(mode=mode).groupby(["mode", "y"]).agg(
        windows=("y", "size"), vlow_rms=("vlow_rms", "median"), vup_rms=("vup_rms", "median"),
        cur_rms=("cur_rms", "median")).reset_index()
    return mode, info, sil


# ---------------------------------------------------------------- importance / interactions
def importance(w: pd.DataFrame, seed=0, cols=None):
    """Fold-wise permutation importance (AP drop) per feature / group, plus leave-group-out retrain PR-AUC."""
    fc = list(cols) if cols else feature_columns(w)
    X, y, g = w[fc].reset_index(drop=True), w["y"].to_numpy(), w["block"].to_numpy()
    rng = np.random.default_rng(seed)
    groups = {**feature_groups(fc), **family_groups(fc)}
    feat, grp = [], []
    oof_lgo = {k: np.zeros(len(y)) for k in ["전체"] + list(groups)}
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        m = _hgb(seed).fit(X.iloc[tr], y[tr])
        Xte = X.iloc[te]
        base = average_precision_score(y[te], m.predict_proba(Xte)[:, 1])
        feat.append(permutation_importance(m, Xte, y[te], scoring="average_precision",
                                           n_repeats=8, random_state=seed).importances_mean)
        row = {}
        for name, cols in groups.items():
            drops = []
            for _ in range(8):
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].to_numpy()[rng.permutation(len(te))]
                drops.append(base - average_precision_score(y[te], m.predict_proba(Xp)[:, 1]))
            row[name] = np.mean(drops)
        grp.append(row)
        oof_lgo["전체"][te] = m.predict_proba(Xte)[:, 1]
        for name, cols in groups.items():                       # retrain without the group
            keep = [c for c in fc if c not in cols]
            oof_lgo[name][te] = _hgb(seed).fit(X.iloc[tr][keep], y[tr]).predict_proba(Xte[keep])[:, 1]
    fi = pd.DataFrame({"feature": fc, "perm_importance": np.mean(feat, axis=0),
                       "perm_sd": np.std(feat, axis=0)}).sort_values("perm_importance", ascending=False)
    gi = pd.DataFrame(grp).agg(["mean", "std"]).T.rename(columns={"mean": "perm_drop_AP", "std": "sd"})
    gi = gi.reset_index().rename(columns={"index": "group"})
    lgo = pd.DataFrame({"removed_group": list(oof_lgo),
                        "PR_AUC": [average_precision_score(y, v) for v in oof_lgo.values()]})
    return fi, gi, lgo


def h_statistic(model, X: pd.DataFrame, cols, n=150, seed=0) -> pd.DataFrame:
    """Friedman H^2 pairwise interaction on log-odds (0 = additive, 1 = pure interaction)."""
    S = X.sample(n, random_state=seed).reset_index(drop=True)
    B = X.sample(n, random_state=seed + 1).reset_index(drop=True)

    def pdp(cs):
        out = np.empty(n)
        for i in range(n):
            T = B.copy()
            for c in cs:
                T[c] = S.loc[i, c]
            out[i] = model.decision_function(T).mean()
        return out - out.mean()

    single = {c: pdp([c]) for c in cols}
    rows = []
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            pab = pdp([cols[a], cols[b]])
            resid = pab - single[cols[a]] - single[cols[b]]
            rows.append({"f1": cols[a], "f2": cols[b], "H2": float((resid ** 2).sum() / ((pab ** 2).sum() + 1e-12))})
    return pd.DataFrame(rows).sort_values("H2", ascending=False)


# ---------------------------------------------------------------- ablations
def imbalance_ablation(w: pd.DataFrame, seed=0, cols=None) -> pd.DataFrame:
    """Same block folds, HGB only: no weights vs balanced weights, both at the default 0.5 threshold."""
    fc = list(cols) if cols else feature_columns(w)
    X, y, g = w[fc].reset_index(drop=True), w["y"].to_numpy(), w["block"].to_numpy()
    p_plain, p_bal = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
        p_plain[te] = _hgb(seed, False).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        p_bal[te] = _hgb(seed, True).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    rows = [{"variant": "HGB 가중치 없음 (thr=0.5)", **window_metrics(y, p_plain, p_plain >= 0.5)},
            {"variant": "HGB class_weight=balanced (thr=0.5)", **window_metrics(y, p_bal, p_bal >= 0.5)}]
    return pd.DataFrame(rows)


def window_ablation(df, sizes=(5, 20), seed=0, invariant=True) -> pd.DataFrame:
    rows = []
    for W in sizes:
        w = make_windows(df, W, max(W // 2, 1)).reset_index(drop=True)
        cols = invariant_columns(feature_columns(w)) if invariant else None
        s = summarise([run_cv(w, seed, cols=cols)], w)
        s.insert(0, "window", W)
        rows.append(s[s.model.isin(["B0_SPC", "B1_LogReg", "M2_HGB", "M4_Stack"])])
    return pd.concat(rows)


def mode_shift_test(w: pd.DataFrame, mode, seed=0, cols=None) -> pd.DataFrame:
    """Train on normals of the OTHER operating modes (+ anomaly train blocks); test on the held-out mode's normals.

    Simulates an unseen but legitimate operating change; the false-alarm rate is the quantity of interest.
    """
    fc = feature_columns(w)
    X, y, g = w[fc].reset_index(drop=True), w["y"].to_numpy(), w["block"].to_numpy()
    an = np.where(y == 1)[0]
    rows = []
    for m in sorted(set(mode[y == 0])):
        held = (mode == m) & (y == 0)
        rest = np.where(~held & (y == 0))[0]
        if held.sum() < 30 or len(rest) < 30:
            continue
        preds = {k: np.zeros(len(y), int) for k in MODEL_ORDER}
        probs = {k: np.zeros(len(y)) for k in MODEL_ORDER}
        for a_tr, a_te in StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(an, y[an], g[an]):
            tr = np.concatenate([rest, an[a_tr]])
            te = np.concatenate([np.where(held)[0], an[a_te]])
            b = fit_bundle(X.iloc[tr].reset_index(drop=True), y[tr], g[tr], seed, cols)
            p = predict_bundle(b, X.iloc[te])
            for k in MODEL_ORDER:
                probs[k][te] = p[k].to_numpy()
                preds[k][te] = (p[k] >= b["thr"][k]).astype(int)
        te_all = np.where(held | (y == 1))[0]
        for k in MODEL_ORDER:
            r = window_metrics(y[te_all], probs[k][te_all], preds[k][te_all])
            rows.append({"held_out_mode": int(m), "model": k, "n_normal": int(held.sum()),
                         "FPR_on_unseen_mode": r["FP"] / max(int(held.sum()), 1),
                         "Recall": r["Recall"], "F1": r["F1"], "PR_AUC": r["PR_AUC"]})
    return pd.DataFrame(rows)


def guard_mode_shift(w: pd.DataFrame, mode) -> pd.DataFrame:
    """Peak-current guards under the unseen-mode protocol of mode_shift_test: the guard is fitted on the OTHER modes'
    normal windows only; we then look at the held-out mode's normal windows (and at all anomaly windows).
    global_* = one observed-max limit (the first design); mode_* = ModeGuard with the known modes only (one mode left,
    so k=1), which additionally reports `new_mode_*` = share of windows outside every known mode's support.
    A high *_normal_flag is the yellow-flood failure; a high new_mode_normal share is the intended behaviour: the state is
    routed to approval instead of being read as a fault."""
    y = w["y"].to_numpy()
    rows = []
    for m in sorted(set(mode[y == 0])):
        held = (mode == m) & (y == 0)
        rest = (mode != m) & (y == 0)
        if held.sum() < 30 or rest.sum() < 30:
            continue
        fit_rows = rest | (y == 1)
        gg = CurrentGuard().fit(w.loc[fit_rows], y[fit_rows])
        mg = ModeGuard(k=int(len(set(mode[rest])))).fit(w.loc[fit_rows], y[fit_rows])
        H, Aw = w.loc[held], w.loc[y == 1]
        rows.append({"held_out_mode": int(m), "n_normal": int(held.sum()), "global_limit": gg.limit_,
                     "global_normal_flag": float(gg.flag(H).mean()), "global_anomaly_flag": float(gg.flag(Aw).mean()),
                     "mode_normal_flag": float(mg.flag(H).mean()), "mode_anomaly_flag": float(mg.flag(Aw).mean()),
                     "new_mode_normal": float(mg.new_mode(H).mean()), "new_mode_anomaly": float(mg.new_mode(Aw).mean())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- stress tests on raw signals
ALPHAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
SPEED_FACTORS = (0.5, 0.7, 1.0, 1.25, 1.5, 2.0, 2.5)


def _rebuild(parts: list, tag: str) -> pd.DataFrame:
    """Stack synthetic segments (CH arrays) into a frame make_windows accepts (state 0, fresh 10 Hz clock)."""
    rows = []
    for k, (seg, arr) in enumerate(parts):
        d = pd.DataFrame(arr, columns=CH)
        d["TimeStamp"] = pd.Timestamp("2000-01-01") + pd.to_timedelta(np.arange(len(d)) * 0.1, unit="s") + pd.Timedelta(hours=k)
        d["Equipment_state"], d["seg"], d["block"], d["source"] = 0, seg, seg, tag
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def _warp(arr: np.ndarray, f: float) -> np.ndarray:
    """Resample a segment so every frequency in it appears f times higher (f>1: faster machine)."""
    n = len(arr)
    t = np.arange(0, n - 1 + 1e-9, f)
    return np.column_stack([np.interp(t, np.arange(n), arr[:, j]) for j in range(arr.shape[1])])


def stress_tests(df: pd.DataFrame, w: pd.DataFrame, cols, thr: dict, final_model: str, seed=0, n_splits=5):
    """Two synthetic perturbations of HELD-OUT normal segments, scored by per-fold bundles with the deployed alarm
    policy `thr` (red / yellow-or-red) next to the SPC rule at its own threshold.

    1. weak fault: (1-a)*normal + a*(random chunk of a TRAIN-fold real fault segment), a = 0..1. Shows how weak a fault
       can get before the model (and the SPC rule) stops raising it. Synthetic - labelled as such in the report.
    2. speed change: time-warp of normal segments by factor f (all frequencies scaled), channels = all / current only.
       Shows how much a drifting motor speed alone costs in false alarms. Caveat: at 10 Hz sampling the apparent
       frequency is aliased, so f is an apparent-frequency factor, not a calibrated rpm change.
    Returns (weak_fault table, speed table); counts are pooled over folds.
    """
    fc = feature_columns(w)
    X, y, g = w[fc].reset_index(drop=True), w["y"].to_numpy(), w["block"].to_numpy()
    raw = {sid: grp[CH].to_numpy(float) for sid, grp in df.groupby("seg")}
    rng = np.random.default_rng(seed)
    weak, speed = {}, {}
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, g):
        b = fit_bundle(X.iloc[tr].reset_index(drop=True), y[tr], g[tr], seed, cols)
        norm_segs = sorted(set(w.seg.iloc[te][y[te] == 0]))
        fault_pool = [raw[s] for s in sorted(set(w.seg.iloc[tr][y[tr] == 1])) if len(raw[s]) >= 10]
        base = [(s, raw[s]) for s in norm_segs if len(raw[s]) >= 5]

        def hits(parts, tag):
            ww = make_windows(_rebuild(parts, tag), 10, 5).reset_index(drop=True)
            p = predict_bundle(b, ww)
            lev = assign_level(p[final_model], p["unc"], thr, p["guard"])
            return {"최종 모델(빨강)": (int((lev == "red").sum()), len(ww)),
                    "최종 모델(노랑 이상)": (int((lev != "green").sum()), len(ww)),
                    "SPC 규칙": (int((p["B0_SPC"] >= b["thr"]["B0_SPC"]).sum()), len(ww))}

        faults = []
        for s, a in base:
            f = fault_pool[rng.integers(len(fault_pool))]
            off = rng.integers(0, max(len(f) - len(a), 0) + 1)
            chunk = f[off:off + len(a)]
            faults.append(chunk if len(chunk) == len(a) else np.resize(f, a.shape))
        for al in ALPHAS:
            r = hits([(s, (1 - al) * a + al * c) for (s, a), c in zip(base, faults)], "weak")
            for m, (h, n) in r.items():
                cell = weak.setdefault((al, m), [0, 0]); cell[0] += h; cell[1] += n
        for f in SPEED_FACTORS:
            for name, chans in (("전 채널", [0, 1, 2]), ("전류만", [2])):
                parts = []
                for s, a in base:
                    wa = _warp(a, f)
                    parts.append((s, np.column_stack([wa[:, j] if j in chans else np.resize(a[:, j], len(wa)) for j in range(3)])))
                r = hits(parts, "speed")
                for m, (h, n) in r.items():
                    cell = speed.setdefault((f, name, m), [0, 0]); cell[0] += h; cell[1] += n
    wk = pd.DataFrame([{"alpha": k[0], "model": k[1], "alarm_windows": v[0], "windows": v[1], "alarm_rate": v[0] / max(v[1], 1)}
                       for k, v in weak.items()])
    sp = pd.DataFrame([{"factor": k[0], "channels": k[1], "model": k[2], "false_alarms": v[0], "windows": v[1],
                        "false_alarm_rate": v[0] / max(v[1], 1)} for k, v in speed.items()])
    return wk, sp


def severity_recall(D: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Real anomaly windows binned by how far their amplitude sits from normal (robust-z of the three RMS channels,
    the same signal an SPC chart reads): is the model still firing in the weakest quartile that SPC cannot see?"""
    N, A = D[(D.y == 0) & (D.rep == 0)], D[(D.y == 1) & (D.rep == 0)].copy()
    z = np.zeros(len(A))
    for c in ("vup_rms", "vlow_rms", "cur_rms"):
        med = N[c].median()
        mad = 1.4826 * (N[c] - med).abs().median() + 1e-9
        z = np.maximum(z, ((A[c] - med).abs() / mad).to_numpy())
    A["severity_z"] = z
    A["level"] = assign_level(A.p, A.unc, thr, A.guard)       # deployed policy: p, member disagreement and the peak-current guard
    A["bin"] = pd.qcut(A.severity_z.rank(method="first"), 4, labels=["Q1 (가장 약함)", "Q2", "Q3", "Q4 (가장 강함)"])
    return A.groupby("bin", observed=True).agg(
        windows=("y", "size"), z_median=("severity_z", "median"), spc_detect=("spc_pred", "mean"),
        model_red=("level", lambda s: (s == "red").mean()), model_yellow_or_red=("level", lambda s: (s != "green").mean())
    ).reset_index()


# ---------------------------------------------------------------- independent evaluation: nested cross-validation
N_OUTER = 3


def outer_splits(w: pd.DataFrame, n_outer: int = N_OUTER, seed: int = 0) -> list:
    """Outer folds: whole 30-second block groups (a block never straddles train/test), stratified on the anomaly label."""
    return list(StratifiedGroupKFold(n_outer, shuffle=True, random_state=seed).split(w, w["y"], w["block"]))


def nested_validation(w: pd.DataFrame, inv: list, seed: int = 0, n_outer: int = N_OUTER):
    """Selection and evaluation on disjoint data. For every outer fold the WHOLE procedure is rerun on the outer-train part
    only - operating-mode clustering, blocked CV of both feature sets x all models, the unseen-mode test, select_final,
    the alarm-policy thresholds - and the chosen model is then scored ONCE on the outer-test part, which took no part in any
    choice. Pooled over the outer folds this is an estimate of the performance of 'select, then fit' itself, free of the
    selection optimism that the main CV table carries (there the same windows picked the model and scored it).
    A single hold-out is not used: the anomaly session has only 6 time blocks.
    Returns (per-fold table, pooled metrics with block-bootstrap CI, segment-rule table, scored test windows)."""
    from .evaluate import window_metrics
    fc = feature_columns(w)
    rows, parts = [], []
    for k, (tr, te) in enumerate(outer_splits(w, n_outer, seed)):
        wtr, wte = w.iloc[tr].reset_index(drop=True), w.iloc[te].reset_index(drop=True)
        mode_tr = find_modes(wtr, seed)[0]
        wtr["mode"] = mode_tr
        sets = {"ALL": None, "INV": inv}
        oofs = {n: [run_cv(wtr, seed, cols=c)] for n, c in sets.items()}
        summ = pd.concat([summarise(oofs[n], wtr).assign(feature_set=n) for n, c in sets.items()], ignore_index=True)
        ms = pd.concat([mode_shift_test(wtr, mode_tr, seed, cols=c).assign(feature_set=n) for n, c in sets.items()], ignore_index=True)
        (fs, model), _ = select_final(summ, ms)
        _, _, thr = alarm_policy(pooled(wtr, oofs[fs], model))
        b = fit_bundle(wtr[fc].reset_index(drop=True), wtr["y"].to_numpy(), wtr["block"].to_numpy(), seed, sets[fs])
        p = predict_bundle(b, wte[fc].reset_index(drop=True))
        pred = (p[model] >= b["thr"][model]).astype(int).to_numpy()
        part = wte[["seg", "block", "y", "pos_in_seg", "win_len", "seg_len"]].copy()
        part["p"], part["pred"], part["fold"], part["t_red"] = p[model].to_numpy(), pred, k, thr["t_red"]
        part["level"] = assign_level(part.p, p["unc"].to_numpy(), thr, p["guard"].to_numpy())
        parts.append(part)
        r = window_metrics(part.y, part.p, pred)
        rows.append({"fold": k + 1, "feature_set": fs, "model": model, "test_windows": len(part), "test_anomaly_windows": int(part.y.sum()),
                     "test_blocks": int(part.block.nunique()), **{c: r[c] for c in ("F1", "Recall", "Precision", "FPR", "PR_AUC", "TP", "FP", "FN", "TN")}})
    test = pd.concat(parts, ignore_index=True)
    ci = bootstrap_ci(test.assign(rep=0))
    tp, fp, fn, tn = (int(((test.pred == a) & (test.y == b)).sum()) for a, b in ((1, 1), (1, 0), (0, 1), (0, 0)))
    ci["TP"], ci["FP"], ci["FN"], ci["TN"] = tp, fp, fn, tn
    return pd.DataFrame(rows), ci, _rule_table(test, float(test.t_red.median())), test


# ---------------------------------------------------------------- overfitting diagnostic
def overfit_check(w: pd.DataFrame, cols, seed=0, configs: dict | None = None) -> pd.DataFrame:
    """Train-vs-held-out AP gap for the deployed hyperparameters, plus a regularization sweep checked
    against the real generalization canary (unseen-mode FPR), not just held-out AP. A small gap with no
    gain from extra regularization is evidence the model is not overfitting in the classical sense; it
    would NOT prove generalization to a fault type never seen in training (see mode_shift_test)."""
    X, y, g = w[cols].reset_index(drop=True), w["y"].to_numpy(), w["block"].to_numpy()
    configs = configs or {
        "배포 설정 (depth4,lr.05,iter200,l2=1)": dict(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0),
        "규제 강화1 (depth3,lr.05,iter150,l2=3)": dict(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=3.0),
        "규제 강화2 (depth3,lr.03,iter120,l2=5)": dict(max_depth=3, learning_rate=0.03, max_iter=120, l2_regularization=5.0),
    }
    mode, _, _ = find_modes(w, seed)
    rows = []
    for name, kw in configs.items():
        tr_ap, te_ap = [], []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            m = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **kw).fit(X.iloc[tr], y[tr])
            tr_ap.append(average_precision_score(y[tr], m.predict_proba(X.iloc[tr])[:, 1]))
            te_ap.append(average_precision_score(y[te], m.predict_proba(X.iloc[te])[:, 1]))
        fpr = {}
        for held in sorted(set(mode[y == 0])):
            mask = (mode == held) & (y == 0)
            if mask.sum() < 30:
                continue
            train_rows = (~mask) & (y == 0) | (y == 1)
            mm = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **kw).fit(X[train_rows], y[train_rows])
            fpr[held] = float((mm.predict_proba(X[mask])[:, 1] >= 0.5).mean())
        rows.append({"variant": name, "train_AP": float(np.mean(tr_ap)), "oof_AP": float(np.mean(te_ap)),
                     "gap": float(np.mean(tr_ap) - np.mean(te_ap)),
                     "unseen_mode0_FPR": fpr.get(0, float("nan")), "unseen_mode1_FPR": fpr.get(1, float("nan"))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- final-model selection
def select_final(summary: pd.DataFrame, shift: pd.DataFrame, tol=0.01):
    """Pick (feature_set, model) by score = mean(F1 in known modes, F1 on an unseen mode).

    Pre-declared tie rule: every candidate within `tol` of the best score competes; the stacking ensemble wins
    such a tie (lower variance + member-disagreement uncertainty), otherwise the top score wins.
    """
    known = summary[summary.model != "B0_SPC"].set_index(["feature_set", "model"])["F1"]
    unseen = shift[shift.model != "B0_SPC"].groupby(["feature_set", "model"])["F1"].mean()
    t = pd.DataFrame({"F1_known_modes": known, "F1_unseen_mode": unseen}).dropna()
    t["score"] = (t.F1_known_modes + t.F1_unseen_mode) / 2
    pool = t[t.score >= t.score.max() - tol]
    stacks = pool[pool.index.get_level_values("model") == "M4_Stack"]
    best = (stacks if len(stacks) else pool).score.idxmax()
    return best, t.reset_index().sort_values("score", ascending=False)

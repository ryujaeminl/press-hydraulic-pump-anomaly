"""Error-condition and operations analyses on out-of-fold predictions."""
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.tree import DecisionTreeClassifier, export_text

from .data import SAMPLE_DT
from .features import feature_columns
from .models import STACK
from .pipeline import best_threshold


def pooled(w: pd.DataFrame, oofs: list, model=STACK) -> pd.DataFrame:
    """Windows x CV repeats, with the chosen model's probability / decision and the SPC baseline decision."""
    frames = []
    for r, o in enumerate(oofs):
        d = w.reset_index(drop=True).copy()
        d["p"], d["pred"], d["unc"] = o[model].to_numpy(), o[model + "_pred"].to_numpy(), o["unc"].to_numpy()
        d["spc_pred"], d["rep"] = o["B0_SPC_pred"].to_numpy(), r
        for c in ("guard", "guard_global", "new_mode"):
            d[c] = o[c].to_numpy() if c in o else 0
        frames.append(d)
    D = pd.concat(frames, ignore_index=True)
    D["err"] = np.select([(D.pred == 1) & (D.y == 0), (D.pred == 0) & (D.y == 1)], ["FP", "FN"], "OK")
    return D


PROCESSING = {"세그먼트 위치", "윈도우 길이"}
NEAR_FP, NEAR_FN = 0.05, 0.95     # "near-error": normal with p >= 0.05, anomaly with p <= 0.95


def _rate(sub: pd.DataFrame, key, err: str, name: str) -> pd.DataFrame:
    fmt = lambda v: f"{v.left:.3g}~{v.right:.3g}" if isinstance(v, pd.Interval) else str(v)
    key = pd.Series([fmt(v) for v in np.asarray(key, dtype=object)], index=sub.index)
    near = sub.p >= NEAR_FP if err == "FP" else sub.p <= NEAR_FN
    g = sub.assign(k=key, near=near).groupby("k").agg(
        n=("err", "size"), errors=("err", lambda s: (s == err).sum()), near_errors=("near", "sum"))
    g["rate"], g["near_rate"] = g["errors"] / g["n"], g["near_errors"] / g["n"]
    g.insert(0, "condition", name)
    return g.reset_index().rename(columns={"k": "level"})


def _flag(s: pd.Series, cond, yes: str, no: str):
    return np.where(cond, yes, no)


def rate_tables(D: pd.DataFrame) -> dict:
    """False-alarm rate among normal windows / miss rate among anomaly windows, by operating condition."""
    N, A = D[D.y == 0], D[D.y == 1]
    q3 = lambda s: pd.qcut(s, 3, labels=["low", "mid", "high"], duplicates="drop")
    fp = pd.concat([
        _rate(N, N["mode"].map(lambda m: f"mode{m}"), "FP", "운전모드"),
        _rate(N, _flag(N, N.pos_in_seg == 0, "첫 윈도우", "이후 윈도우"), "FP", "세그먼트 위치"),
        _rate(N, _flag(N, N.win_len < 10, "짧은(<10)", "정상(=10)"), "FP", "윈도우 길이"),
        _rate(N, q3(N["cur_rms"]), "FP", "전류 RMS 3분위"),
        _rate(N, q3(N["vlow_rms"]), "FP", "하부진동 RMS 3분위"),
    ])
    fn = pd.concat([
        _rate(A, pd.qcut(A["vup_rms"], 4, duplicates="drop"), "FN", "상부진동 RMS 4분위"),
        _rate(A, _flag(A, A.win_len < 10, "짧은(<10)", "정상(=10)"), "FN", "윈도우 길이"),
        _rate(A, _flag(A, A.pos_in_seg == 0, "첫 윈도우", "이후 윈도우"), "FN", "세그먼트 위치"),
        _rate(A, A["mode"].map(lambda m: f"mode{m}"), "FN", "가장 가까운 운전모드"),
        _rate(A, pd.qcut(A["vlow_rms"], 4, duplicates="drop"), "FN", "하부진동 RMS 4분위"),
        _rate(A, pd.qcut(A["cur_rms"], 4, duplicates="drop"), "FN", "전류 RMS 4분위"),
    ])
    for t in (fp, fn):      # machine state (what the equipment is doing) vs processing artefact (how we cut the signal)
        t.insert(1, "kind", np.where(t.condition.isin(PROCESSING), "처리 조건", "기계 상태"))
    a = A.assign(spc_miss=(A.spc_pred == 0))
    prof = a.groupby("spc_miss")[["vup_rms", "vlow_rms", "cur_rms", "cur_mean", "vup_mad", "vup_zcr", "vup_kurt"]].median()
    prof["n"] = a.groupby("spc_miss").size()
    return {"fp_conditions": fp, "fn_conditions": fn, "spc_miss_profile": prof.reset_index()}


def hard_case_rules(D: pd.DataFrame, depth=3):
    """Shallow trees on 'near-error' windows (normals with p>=0.05, anomalies with p<=0.95), first CV repeat.
    Returns (text of the trees, table with each tree's root split: feature, threshold, side holding most hard windows)."""
    d = D[D.rep == 0]
    fc = feature_columns(d)
    out, roots = [], []
    for name, sub, hard, leaf in (("정상 윈도우 중 오경보 근접(p>=0.05)", d[d.y == 0], lambda s: s.p >= NEAR_FP, 10),
                                  ("이상 윈도우 중 미탐 근접(p<=0.95)", d[d.y == 1], lambda s: s.p <= NEAR_FN, 3)):
        yy = hard(sub).astype(int)
        if yy.sum() < 3:
            out.append(f"[{name}] 해당 윈도우 {int(yy.sum())}개 - 규칙 도출 생략\n")
            continue
        t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, class_weight="balanced",
                                   random_state=0).fit(sub[fc], yy)
        out.append(f"[{name}] n={len(sub)}, hard={int(yy.sum())}\n" + export_text(t, feature_names=fc, decimals=3))
        tr = t.tree_
        frac = lambda c: tr.value[c][0][1] / max(tr.value[c][0].sum(), 1e-9)       # share of 'hard' windows in a child
        roots.append({"case": "오경보 근접" if sub.y.iloc[0] == 0 else "미탐 근접", "feature": fc[tr.feature[0]],
                      "threshold": float(tr.threshold[0]),
                      "hard_side": "<=" if frac(tr.children_left[0]) >= frac(tr.children_right[0]) else ">"})
    return "\n".join(out), pd.DataFrame(roots)


def hardest_cases(D: pd.DataFrame, k=12) -> pd.DataFrame:
    d = D[D.rep == 0]
    cols = ["seg", "pos_in_seg", "win_len", "mode", "p", "unc", "vup_rms", "vlow_rms", "cur_rms", "cur_mean", "vup_zcr"]
    fp = d[d.y == 0].nlargest(k, "p")[cols].assign(kind="정상(오경보 위험)")
    fn = d[d.y == 1].nsmallest(k, "p")[cols].assign(kind="이상(미탐 위험)")
    return pd.concat([fp, fn])


def reliability(y, p, bins=8) -> pd.DataFrame:
    """Quantile reliability table on the anomaly-probability axis."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(mean_p=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index()


def ece(y, p, bins=10) -> float:
    r = reliability(y, p, bins)
    return float((r.n * (r.mean_p - r.frac_pos).abs()).sum() / r.n.sum())


def uncertainty_table(D: pd.DataFrame) -> pd.DataFrame:
    d = D[D.rep == 0].copy()
    edges = [0, d.unc.quantile(0.5), d.unc.quantile(0.9), d.unc.quantile(0.98), d.unc.max() + 1e-9]
    d["unc_bin"] = pd.cut(d.unc, edges, labels=["하위50%", "50~90%", "90~98%", "상위2%"], include_lowest=True)
    t = d.groupby("unc_bin", observed=True).agg(n=("y", "size"), anomalies=("y", "sum"),
                                                 errors=("err", lambda s: (s != "OK").sum())).reset_index()
    t["error_rate"] = t.errors / t.n
    return t


def uncertainty_capture(D: pd.DataFrame, ks=(0.01, 0.02, 0.05, 0.10)) -> pd.DataFrame:
    """Do the model's errors concentrate in the top-k% most uncertain windows? (first CV repeat). Reports the error capture
    rate, the enrichment over chance and a binomial p-value, so the claim 'recheck the uncertain windows first' is tested
    rather than asserted."""
    d = D[D.rep == 0]
    err = (d.err != "OK").to_numpy()
    n_err = int(err.sum())
    rows = []
    for k in ks:
        top = (d.unc >= d.unc.quantile(1 - k)).to_numpy()
        cap, frac = int((top & err).sum()), float(top.mean())
        rows.append({"top_pct": round(100 * k), "windows": int(top.sum()), "errors_total": n_err, "errors_captured": cap,
                     "capture_rate": cap / max(n_err, 1), "expected_by_chance": n_err * frac,
                     "enrichment": cap / max(n_err * frac, 1e-9), "p_value": float(binom.sf(cap - 1, n_err, frac)) if cap else 1.0})
    return pd.DataFrame(rows)


def unc_informative(cap: pd.DataFrame) -> bool:
    """Pre-declared rule: uncertainty may directly raise an alarm level only if the top 2% hold >=2 errors, >=50% of all
    errors, and that is unlikely by chance (binomial p < 0.01). Otherwise it stays a monitoring metric."""
    r = cap[cap.top_pct == 2].iloc[0]
    return bool(r.errors_captured >= 2 and r.capture_rate >= 0.5 and r.p_value < 0.01)


def unc_trigger_cost(D: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Windows that are yellow ONLY because members disagree (p below t_yellow, guard silent): what the unc trigger costs."""
    d = D[D.rep == 0]
    only = (d.p < thr["t_yellow"]) & (d.guard == 0) & (d.unc >= thr["unc_high"])
    return pd.DataFrame([{"normal_windows_yellow_by_unc_only": int((only & (d.y == 0)).sum()),
                          "anomaly_windows_yellow_by_unc_only": int((only & (d.y == 1)).sum()),
                          "errors_in_unc_only_yellow": int((only & (d.err != "OK")).sum()), "errors_total": int((d.err != "OK").sum()),
                          "normal_windows": int((d.y == 0).sum()), "anomaly_windows": int((d.y == 1).sum())}])


def bootstrap_ci(D: pd.DataFrame, n_boot=1000, seed=0) -> pd.DataFrame:
    """Block-bootstrap 95% CI of window-level F1 / recall / false-positive rate (first CV repeat)."""
    d = D[D.rep == 0]
    key = d.assign(tp=(d.pred == 1) & (d.y == 1), fp=(d.pred == 1) & (d.y == 0),
                   fn=(d.pred == 0) & (d.y == 1), tn=(d.pred == 0) & (d.y == 0))
    cnt = key.groupby("block")[["tp", "fp", "fn", "tn"]].sum().to_numpy(float)
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_boot):
        tp, fp, fn, tn = cnt[rng.integers(0, len(cnt), len(cnt))].sum(0)
        res.append((2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fn, 1), fp / max(fp + tn, 1)))
    r = np.array(res)
    tp, fp, fn, tn = cnt.sum(0)
    point = (2 * tp / (2 * tp + fp + fn), tp / (tp + fn), fp / (fp + tn))
    return pd.DataFrame({"metric": ["F1", "Recall", "FPR"], "point": point,
                         "ci_low": np.percentile(r, 2.5, axis=0), "ci_high": np.percentile(r, 97.5, axis=0)})


def model_diff_ci(oof: pd.DataFrame, w: pd.DataFrame, model_a: str, model_b: str, n_boot=1000, seed=0) -> dict:
    """Block-bootstrap 95% CI for F1(model_a) - F1(model_b), paired on the SAME OOF repeat (same test windows
    for both models, same resampled blocks each draw) -> an honest "are these two models really different"
    check, not just eyeballing the F1 gap. If the interval straddles 0, the gap is not statistically supported
    at this sample size."""
    d = w[["block"]].copy()
    d["y"] = oof["y"].to_numpy()
    for tag, m in (("a", model_a), ("b", model_b)):
        pred = oof[f"{m}_pred"].to_numpy()
        d[f"tp_{tag}"] = (pred == 1) & (d.y == 1)
        d[f"fp_{tag}"] = (pred == 1) & (d.y == 0)
        d[f"fn_{tag}"] = (pred == 0) & (d.y == 1)
    cols = [f"{p}_{t}" for t in ("a", "b") for p in ("tp", "fp", "fn")]
    cnt = d.groupby("block")[cols].sum().to_numpy(float)

    def f1(tp, fp, fn):
        return 2 * tp / max(2 * tp + fp + fn, 1)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        tpa, fpa, fna, tpb, fpb, fnb = cnt[rng.integers(0, len(cnt), len(cnt))].sum(0)
        deltas.append(f1(tpa, fpa, fna) - f1(tpb, fpb, fnb))
    tpa, fpa, fna, tpb, fpb, fnb = cnt.sum(0)
    return {"model_a": model_a, "model_b": model_b, "delta_F1": f1(tpa, fpa, fna) - f1(tpb, fpb, fnb),
            "ci_low": float(np.percentile(deltas, 2.5)), "ci_high": float(np.percentile(deltas, 97.5))}


def cost_curve(D: pd.DataFrame, ratios=(1, 5, 10, 50, 100)) -> pd.DataFrame:
    """Threshold minimising  C_FN*FN + C_FP*FP  (C_FP = 1) for several assumed cost ratios (first CV repeat)."""
    d = D[D.rep == 0]
    y, p = d.y.to_numpy(), d.p.to_numpy()
    ts = np.unique(np.r_[0.0, np.quantile(p, np.linspace(0, 1, 400)), 1.0])
    fp = np.array([((p >= t) & (y == 0)).sum() for t in ts])
    fn = np.array([((p < t) & (y == 1)).sum() for t in ts])
    rows = []
    for r in ratios:
        i = int(np.argmin(r * fn + fp))
        rows.append({"C_FN/C_FP": r, "threshold": ts[i], "FP": int(fp[i]), "FN": int(fn[i]),
                     "cost": float(r * fn[i] + fp[i])})
    return pd.DataFrame(rows)


def assign_level(p, unc, thr: dict, guard=None):
    """red: p >= t_red | yellow: p >= t_yellow, ensemble members disagree (unc >= unc_high), or the peak-current
    guard fired | green. The guard can only raise a window to yellow, never to red: a red alarm is reserved for
    what the learned model itself finds anomalous (see guard_effect for what red-forcing cost)."""
    p, unc = np.asarray(p), np.asarray(unc)
    g = np.zeros(len(p), bool) if guard is None else np.asarray(guard).astype(bool)
    return np.select([p >= thr["t_red"], (p >= thr["t_yellow"]) | (unc >= thr["unc_high"]) | g], ["red", "yellow"], "green")


def _consec(flag) -> bool:
    """Two consecutive True. A one-window segment cannot show two in a row, so it counts as hit if that window is."""
    f = np.asarray(flag)
    return bool(f[0]) if len(f) == 1 else bool((f[1:] & f[:-1]).any())


def sequence_hit(level, rule: str) -> bool:
    """R1 / R2 / R3 on a time-ordered array of alarm levels. This single function scores the report's rule table AND
    predict.py's stop recommendation, so the analysed policy and the deployed one cannot drift apart."""
    lv = np.asarray(level)
    return {"R1": lambda: bool((lv == "red").any()), "R2": lambda: _consec(lv == "red"),
            "R3": lambda: _consec(lv != "green")}[rule]()


RULES = {"R1 red 1회": lambda s: sequence_hit(s.level, "R1"),
         "R2 red 2연속": lambda s: sequence_hit(s.level, "R2"),
         "R3 yellow 이상 2연속": lambda s: sequence_hit(s.level, "R3")}


def _rule_table(d: pd.DataFrame, t_red: float) -> pd.DataFrame:
    """Segment-level rules on a frame with columns seg, pos_in_seg, win_len, seg_len, y, p, level."""
    hours = d[d.y == 0].groupby("seg").seg_len.first().sum() * SAMPLE_DT / 3600
    # R4 compares the segment's mean p with the red threshold; a frame pooled over folds carries each fold's own t_red
    rules = {**RULES, "R4 세그먼트 평균 p>=t_red": lambda s: s.p.mean() >= (s["t_red"].iloc[0] if "t_red" in s else t_red)}
    y_seg = d.groupby("seg").y.max()
    d = d.assign(e=d.pos_in_seg + d.win_len * SAMPLE_DT)
    first_red = d[d.level == "red"].groupby("seg").e.min()
    rows = []
    for name, f in rules.items():
        al = d.groupby("seg").apply(f, include_groups=False).astype(bool)
        rows.append({"rule": name, "segment_recall": float(al[y_seg == 1].mean()),
                     "normal_seg_false_alarm": float(al[y_seg == 0].mean()),
                     "false_alarms_per_hour": float(al[y_seg == 0].sum() / hours),
                     "latency_median_s(first red)": float(first_red.reindex(y_seg[y_seg == 1].index).median())})
    return pd.DataFrame(rows)


def _level_table(d: pd.DataFrame) -> pd.DataFrame:
    lv = d.groupby(["y", "level"]).size().unstack(fill_value=0).reindex(columns=["green", "yellow", "red"], fill_value=0)
    lv.index = ["정상 윈도우" if i == 0 else "이상 윈도우" for i in lv.index]
    return lv.reset_index().rename(columns={"index": "구분"})


def alarm_policy(D: pd.DataFrame, cost_ratio=10) -> tuple:
    """Three-level policy + sequence rules. Returns (level table, rule table, thresholds).

    t_red = F1-optimal threshold of the MODEL probability; t_yellow = threshold minimising C_FN*FN + C_FP*FP for an
    ASSUMED cost ratio. The deployed levels additionally turn yellow where the peak-current guard fired.
    """
    d = D[D.rep == 0].sort_values(["seg", "pos_in_seg"]).copy()
    t_red = best_threshold(d.y, d.p)
    t_cost = float(cost_curve(D, (cost_ratio,)).threshold.iloc[0])
    unc_used = unc_informative(uncertainty_capture(D))
    thr = {"t_red": t_red, "t_yellow": max(t_cost if t_cost < t_red else t_red / 2, 1e-3),
           "unc_high": float(d.unc.quantile(0.98)) if unc_used else 1e9,      # 1e9 = trigger switched off
           "unc_used": unc_used, "assumed_cost_ratio": cost_ratio}
    d["level"] = assign_level(d.p, d.unc, thr, d.guard)
    return _level_table(d), _rule_table(d, t_red), thr


def guard_effect(D: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Same thresholds, same windows (first CV repeat): alarm levels WITHOUT the peak-current guard, WITH it raising
    only to yellow, and WITH it forcing red (the variant that was rejected - kept so its cost stays visible)."""
    d = D[D.rep == 0].sort_values(["seg", "pos_in_seg"]).copy()
    base = assign_level(d.p, d.unc, thr)
    variants = {"모델만": base,
                "전역 최대값 가드→노랑": assign_level(d.p, d.unc, thr, d.guard_global),
                "모드별 가드→노랑 (채택)": assign_level(d.p, d.unc, thr, d.guard),
                "전역 가드→빨강 (폐기)": np.where(d.guard_global.to_numpy() == 1, "red", base)}
    rows = []
    for name, lev in variants.items():
        d["level"] = lev
        n, a = d[d.y == 0], d[d.y == 1]
        r = _rule_table(d, thr["t_red"]).set_index("rule")
        rows.append({"variant": name,
                     "normal_windows": len(n), "normal_yellow": int((n.level == "yellow").sum()),
                     "normal_red": int((n.level == "red").sum()),
                     "normal_new_mode": int((n.new_mode == 1).sum()), "anomaly_new_mode": int((a.new_mode == 1).sum()),
                     "anomaly_windows": len(a), "anomaly_red": int((a.level == "red").sum()),
                     "anomaly_yellow": int((a.level == "yellow").sum()), "anomaly_green": int((a.level == "green").sum()),
                     "R1_recall": r.loc["R1 red 1회", "segment_recall"], "R1_FA_per_h": r.loc["R1 red 1회", "false_alarms_per_hour"],
                     "R2_recall": r.loc["R2 red 2연속", "segment_recall"], "R2_FA_per_h": r.loc["R2 red 2연속", "false_alarms_per_hour"],
                     "R3_recall": r.loc["R3 yellow 이상 2연속", "segment_recall"],
                     "R3_FA_per_h": r.loc["R3 yellow 이상 2연속", "false_alarms_per_hour"]})
    return pd.DataFrame(rows)

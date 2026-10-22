"""Score a raw sensor CSV with the trained bundle produced by run_all.py.

    python predict.py --input new_data.csv [--model outputs/model_bundle.joblib] [--out outputs/predictions_new.csv]

Input columns: TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current (Equipment_state optional, ignored).
Output: one row per window (p_anomaly, uncertainty, guard, new_mode, level = green/yellow/red) and a per-segment summary CSV.

Each run also appends one row to outputs/predict_log.csv, writes outputs/latest_run.json (the new data's timeline and
segment waveforms) and rebuilds dashboard.html, so a dashboard tab left open (it auto-refreshes, see
report/build_dashboard.py) plots every scoring run as it happens.
"""
import argparse
import csv
import json
import math
import os
import runpy
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import CH, SAMPLE_DT, read_csv, segment
from src.errors import assign_level, sequence_hit
from src.features import make_windows
from src.pipeline import predict_bundle

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
LEVEL_RANK = {"green": 0, "yellow": 1, "red": 2}
DASHBOARD_OUT = ROOT / "outputs"                       # same dir run_all.py writes to, for one shared dashboard
LOG_PATH = DASHBOARD_OUT / "predict_log.csv"
LATEST_PATH = DASHBOARD_OUT / "latest_run.json"        # what the dashboard plots for the newest scoring run
LOG_COLS = ["run_at", "input", "windows", "segments", "green", "yellow", "red", "worst_level", "max_p"]
MAX_POINTS, MAX_SEGS, MAX_WAVE = 4000, 40, 600         # keep dashboard.html small whatever the input size


def effective_policy(bundle: dict, use_unc: bool = True) -> dict:
    return bundle["policy"] if use_unc else {**bundle["policy"], "unc_high": 1e9}   # 1e9 = member-disagreement trigger off


def score(bundle: dict, seg: pd.DataFrame, use_unc: bool = True) -> pd.DataFrame:
    """`seg` = segment(read_csv(...)): raw samples with their continuous-segment id."""
    pol = effective_policy(bundle, use_unc)
    w = make_windows(seg, pol["window"], pol["stride"]).reset_index(drop=True)
    if w.empty:
        raise ValueError("no usable window: every continuous segment is shorter than 5 samples")
    p = predict_bundle(bundle, w)
    out = w[["seg", "t_start", "t_end", "pos_in_seg", "win_len"]].copy()
    out["p_anomaly"] = p[pol["final_model"]].to_numpy()
    out["uncertainty"] = p["unc"].to_numpy()
    # mode-specific peak-current guard (src/models.py:ModeGuard) only lifts a window to yellow; red stays the model's call.
    # new_mode = outside every known operating mode's support: not an alarm, but it must go to approval/retraining.
    out["guard"] = p["guard"].astype(int).to_numpy()
    out["new_mode"] = p["new_mode"].astype(int).to_numpy()
    out["level"] = assign_level(out.p_anomaly, out.uncertainty, pol, out.guard)
    return out


def summarise_segments(win: pd.DataFrame) -> pd.DataFrame:
    g = win.assign(rank=win.level.map(LEVEL_RANK)).groupby("seg")
    seg = g.agg(t_start=("t_start", "min"), windows=("level", "size"), p_max=("p_anomaly", "max"),
                worst_rank=("rank", "max"))
    seg["worst_level"] = seg.worst_rank.map({v: k for k, v in LEVEL_RANK.items()})
    # rule R2 (two consecutive red windows) = the stop recommendation the report derives its false-alarm rate for
    ordered = win.sort_values(["seg", "pos_in_seg"]).groupby("seg")
    seg["stop_recommended"] = ordered["level"].apply(lambda s: sequence_hit(s, "R2"))     # same function as the report's rule table
    seg["new_mode_share"] = ordered["new_mode"].mean()
    return seg.drop(columns="worst_rank").reset_index()


def log_run(input_path: Path, win: pd.DataFrame, seg: pd.DataFrame):
    """Append one row summarising this scoring run to outputs/predict_log.csv (creates it on first use)."""
    counts = win.level.value_counts().reindex(["green", "yellow", "red"], fill_value=0)
    worst = "red" if counts.red else "yellow" if counts.yellow else "green"
    row = {"run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "input": input_path.name,
           "windows": len(win), "segments": len(seg), "green": int(counts.green), "yellow": int(counts.yellow),
           "red": int(counts.red), "worst_level": worst, "max_p": round(float(win.p_anomaly.max()), 4)}
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if is_new:
            w.writeheader()
        w.writerow(row)


def timeline(win: pd.DataFrame, n_max: int = MAX_POINTS) -> dict:
    """p and worst level per window; max-pooled into <= n_max buckets so a long file stays a small page."""
    k = max(1, math.ceil(len(win) / n_max))
    grp = np.arange(len(win)) // k
    p = win.p_anomaly.reset_index(drop=True).groupby(grp).max()
    lv = win.level.map(LEVEL_RANK).reset_index(drop=True).groupby(grp).max()
    first = win.seg.ne(win.seg.shift()).to_numpy().nonzero()[0] // k
    n_seg = len(first)
    return {"step": k, "p": [round(float(v), 5) for v in p], "lv": [int(v) for v in lv],
            "seg_at": sorted({int(i) for i in first}) if n_seg <= 100 else []}      # boundary lines only while readable


def segment_wave(sdf: pd.DataFrame, win: pd.DataFrame, seg_id) -> dict:
    """Raw 3-channel samples of one segment (<= MAX_WAVE around its highest-probability window) + its window levels."""
    g, w = sdf[sdf.seg == seg_id], win[win.seg == seg_id]
    peak = w.loc[w.p_anomaly.idxmax()]
    centre = int(round(peak.pos_in_seg / SAMPLE_DT)) + int(peak.win_len) // 2
    lo = max(0, min(centre - MAX_WAVE // 2, len(g) - MAX_WAVE))
    hi = min(len(g), lo + MAX_WAVE)
    wins = []
    for r in w.itertuples():
        s = int(round(r.pos_in_seg / SAMPLE_DT))
        a, b = max(s, lo), min(s + int(r.win_len), hi)
        if a < b:
            wins.append([a - lo, b - lo, LEVEL_RANK[r.level]])
    cols = {f"ai{i}": [round(float(v), 4) for v in g[c].iloc[lo:hi]] for i, c in enumerate(CH)}
    return {**cols, "wins": wins, "t0": round(lo * SAMPLE_DT, 1)}


def latest_run(input_path: Path, sdf: pd.DataFrame, win: pd.DataFrame, seg: pd.DataFrame, pol: dict) -> dict:
    """Everything the dashboard needs to plot the newest scoring run (written to outputs/latest_run.json)."""
    counts = win.level.value_counts().reindex(["green", "yellow", "red"], fill_value=0)
    top = seg.assign(rank=seg.worst_level.map(LEVEL_RANK)).sort_values(["rank", "p_max"], ascending=False).head(MAX_SEGS)
    segs = [{"seg": int(r.seg), "t_start": pd.Timestamp(r.t_start).isoformat(sep=" ", timespec="milliseconds"), "windows": int(r.windows), "p_max": round(float(r.p_max), 4),
             "worst_level": r.worst_level, "stop": bool(r.stop_recommended), "new_mode_share": round(float(r.new_mode_share), 3),
             "wave": segment_wave(sdf, win, r.seg)} for r in top.itertuples()]
    return {"input": input_path.name, "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "thr": {k: float(pol[k]) for k in ("t_red", "t_yellow")}, "windows": len(win), "segments": len(seg),
            "stop": int(seg.stop_recommended.sum()), "counts": {k: int(v) for k, v in counts.items()},
            "new_mode_share": round(float(win.new_mode.mean()), 3), "timeline": timeline(win), "segs": segs}


def write_latest_run(payload: dict):
    LATEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LATEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, LATEST_PATH)                       # atomic: an auto-reloading dashboard never reads half a file


def refresh_dashboard():
    """Best-effort: rebuild dashboard.html so an open tab (auto-refreshing) picks up this run's log row.
    Silently skipped if run_all.py has never produced the tables/figures the dashboard reads from."""
    try:
        runpy.run_path(str(ROOT / "report" / "build_dashboard.py"), init_globals={"OUT_DIR": str(DASHBOARD_OUT)})
    except Exception as e:
        print(f"[warn] dashboard not refreshed: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default=str(ROOT / "outputs" / "model_bundle.joblib"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "predictions_new.csv"))
    ap.add_argument("--no-unc-trigger", action="store_true",
                    help="do not raise windows to yellow because the ensemble members disagree (costs ~1.3%% extra yellow on normal data)")
    a = ap.parse_args()
    input_path = Path(a.input)
    bundle, sdf = joblib.load(a.model), segment(read_csv(input_path))
    win = score(bundle, sdf, use_unc=not a.no_unc_trigger)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    win.to_csv(a.out, index=False, encoding="utf-8-sig")
    seg = summarise_segments(win)
    seg.to_csv(Path(a.out).with_name(Path(a.out).stem + "_segments.csv"), index=False, encoding="utf-8-sig")
    print(f"{len(win)} windows, {len(seg)} segments -> {a.out}")
    print(win.level.value_counts().reindex(["green", "yellow", "red"], fill_value=0).to_string())
    print(f"stop recommended (R2: 2 consecutive red windows) in {int(seg.stop_recommended.sum())} of {len(seg)} segments")
    if win.new_mode.mean() > 0.05:
        print(f"[notice] {100 * win.new_mode.mean():.0f}% of windows lie outside every known operating mode: either a NEW operating "
              "mode (send to approval / retraining) or a fault - read it together with the alarm levels; the guard is not applied there")
    log_run(input_path, win, seg)
    write_latest_run(latest_run(input_path, sdf, win, seg, effective_policy(bundle, not a.no_unc_trigger)))
    refresh_dashboard()


if __name__ == "__main__":
    main()

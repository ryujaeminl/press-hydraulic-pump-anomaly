"""End-to-end pipeline: preprocessing -> features -> blocked CV -> analyses -> final model -> predictions -> figures.

    python run_all.py                      # data folder = parent of this file
    python run_all.py --data <folder>      # folder holding press_data_normal.csv and outlier_data.csv
    python run_all.py --quick              # final model only (INV / M2_HGB), 1 CV repeat, no ablations: smoke test -> outputs_quick/
"""
import argparse
import json
import runpy
import time
import warnings
import webbrowser
from pathlib import Path

import joblib
import pandas as pd

from src import analysis as A
from src import errors as E
from src import figures as F
from src import result_figures as R
from src.data import load_training, quality_report
from src.evaluate import MODEL_ORDER, run_cv, summarise
from src.features import feature_columns, invariant_columns, make_windows
from src.pipeline import fit_bundle

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path(__file__).resolve().parent
SEED = 0
T0 = time.time()


def default_data_dir() -> str:
    """Where the two CSVs live when --data is not given: solution/data/ (a self-contained zip) or the parent folder."""
    for d in (ROOT / "data", ROOT.parent):
        if (d / "press_data_normal.csv").exists():
            return str(d)
    return str(ROOT.parent)


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def save(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.5g")


def print_summary(summary, ms, final_set, final_model, thr, guard_eff=None):
    """Key numbers of the run as a plain-text table (what the terminal shows when the run ends)."""
    s = summary[summary.feature_set == final_set].set_index("model")
    bar = "=" * 76
    print()
    print(bar)
    print(f" FINAL MODEL: {final_model}  (feature set {final_set})   red p>={thr['t_red']:.3f}   yellow p>={thr['t_yellow']:.3f}")
    print(bar)
    print(f" {'model':<11}{'F1':>7}{'PR-AUC':>8}{'Recall':>8}{'Prec':>7}{'FPR%':>7}{'FP':>6}{'FN':>6}")
    for m in MODEL_ORDER:
        r = s.loc[m]
        star = "   <- final" if m == final_model else ""
        print(f" {m:<11}{r.F1:7.3f}{r.PR_AUC:8.3f}{r.Recall:8.3f}{r.Precision:7.3f}{100 * r.FPR:7.2f}{r.FP:6.1f}{r.FN:6.1f}{star}")
    if len(ms):
        print(" unseen operating mode -> false-alarm rate on normal windows (held-out mode0 / mode1)")
        for fs in ("ALL", "INV"):
            v = ms[(ms.feature_set == fs) & (ms.model == final_model)].sort_values("held_out_mode").FPR_on_unseen_mode
            print(f"   {final_model} with {fs} features: " + " / ".join(f"{100 * x:.1f}%" for x in v))
    if guard_eff is not None and len(guard_eff):
        g = guard_eff.set_index("variant")
        base, glob, yel, red = g.iloc[0], g.iloc[1], g.iloc[2], g.iloc[3]
        print(f" mode-specific guard -> yellow only: normal windows yellow {base.normal_yellow:.0f}->{yel.normal_yellow:.0f} (single global limit: {glob.normal_yellow:.0f}), "
              f"red {base.normal_red:.0f}->{yel.normal_red:.0f}   (forcing red instead would give red {red.normal_red:.0f}, R1 false alarms/h {base.R1_FA_per_h:.2f}->{red.R1_FA_per_h:.2f})")
    print(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=default_data_dir(), help="folder with press_data_normal.csv and outlier_data.csv (default: solution/data, else the parent folder)")
    ap.add_argument("--out", default=None, help="default: outputs/ (outputs_quick/ with --quick)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-open", action="store_true", help="do not open the dashboard in a browser")
    a = ap.parse_args()
    out = Path(a.out) if a.out else ROOT / ("outputs_quick" if a.quick else "outputs")   # a smoke test never overwrites real results
    tab, fig = out / "tables", out / "figures"
    repeats = 1 if a.quick else a.repeats
    F.setup_font()

    # 1. data understanding -------------------------------------------------------------------
    df = load_training(a.data)
    save(quality_report(a.data), tab / "01_data_quality.csv")
    xcorr, xcorr_curves = A.cross_correlation(df)
    save(xcorr, tab / "01b_cross_correlation.csv")
    w = make_windows(df, a.window, a.stride).reset_index(drop=True)
    fc = feature_columns(w)
    inv = invariant_columns(fc)
    seg_all = df.groupby("source").seg.nunique().to_dict()
    seg_used = w.groupby("source").seg.nunique().to_dict()
    log(f"windows={len(w)} anomaly={int(w.y.sum())} features={len(fc)} (invariant {len(inv)}) segments used {seg_used} of {seg_all}")
    mode, minfo, sil = A.find_modes(w, SEED)
    w["mode"] = mode
    save(minfo, tab / "02_operating_modes.csv")
    F.fig_signals(df, fig)
    F.fig_cross_corr(xcorr_curves, fig)
    F.fig_segments(df, fig)
    F.fig_modes(w, mode, fig)
    F.fig_features(w, fig)

    # 2. two feature sets x all models: blocked CV (known modes) + unseen-mode stress test ----------
    sets = {"INV": inv} if a.quick else {"ALL": None, "INV": inv}
    oofs = {}
    for name, cols in sets.items():
        oofs[name] = [run_cv(w, s, cols=cols) for s in range(repeats)]
        log(f"CV done: feature set {name}")
    summary = pd.concat([summarise(oofs[n], w).assign(feature_set=n) for n in sets], ignore_index=True)
    summary["ECE"] = [E.ece(oofs[r.feature_set][0].y, oofs[r.feature_set][0][r.model]) for r in summary.itertuples()]
    save(summary, tab / "03_model_comparison.csv")
    ms = pd.DataFrame()
    if a.quick:
        final_set, final_model, sel = "INV", "M2_HGB", pd.DataFrame()
    else:
        ms = pd.concat([A.mode_shift_test(w, mode, SEED, cols=c).assign(feature_set=n) for n, c in sets.items()],
                       ignore_index=True)
        save(ms, tab / "04_mode_shift.csv")
        (final_set, final_model), sel = A.select_final(summary, ms)
        save(sel, tab / "05_final_selection.csv")
        F.fig_mode_shift(ms, fig)
        F.fig_feature_sets(sel, fig)
    log(f"final model: {final_model} on feature set {final_set}")
    cols, foofs = sets[final_set], oofs[final_set]

    # 2a. is the F1 gap between the final model and its closest rivals real, or noise? -----------------------
    if not a.quick:
        diffs = pd.DataFrame([E.model_diff_ci(foofs[0], w, final_model, rival)
                              for rival in ("M1_RF", "M4_Stack") if rival != final_model])
        save(diffs, tab / "05b_model_diff_ci.csv")
        log("model-diff significance check done")

    F.fig_compare(summary[summary.feature_set == final_set], fig)
    F.fig_pr(foofs[0], fig)
    F.fig_calibration(foofs[0], fig)
    if not a.quick:
        save(summarise([run_cv(w, SEED, group_col=None, cols=cols)], w), tab / "06_leaky_random_split_check.csv")
        log("leakage check done")

    # 3. importance / interaction ---------------------------------------------------------------
    for name, c in sets.items():
        fi, gi, lgo = A.importance(w, SEED, cols=c)
        save(fi, tab / f"07_perm_importance_{name}.csv")
        save(gi, tab / f"07_group_importance_{name}.csv")
        save(lgo, tab / f"07_leave_group_out_{name}.csv")
        F.fig_importance(fi, gi, lgo, fig, name="fig08_importance.png" if name == final_set else f"fig08b_importance_{name}.png")
        if name == final_set:
            X = w[cols or fc]
            hgb = A._hgb(SEED).fit(X, w["y"].to_numpy())
            h = A.h_statistic(hgb, X, list(fi.feature.head(6)), seed=SEED)
            save(h, tab / "07_interaction_H2.csv")
            F.fig_interaction(h, hgb, X, fig)
    log("importance/interaction done")

    # 4. error conditions, uncertainty, policy (final model) --------------------------------------
    D = E.pooled(w, foofs, model=final_model)
    tabs = E.rate_tables(D)
    for k, v in tabs.items():
        save(v, tab / f"08_{k}.csv")
    rules_txt, rule_roots = E.hard_case_rules(D)
    (tab / "08_hard_case_rules.txt").write_text(rules_txt, encoding="utf-8")
    save(rule_roots, tab / "08_hard_case_roots.csv")
    save(E.hardest_cases(D), tab / "08_hardest_cases.csv")
    save(E.uncertainty_table(D), tab / "09_uncertainty.csv")
    save(E.uncertainty_capture(D), tab / "09b_uncertainty_capture.csv")
    ci = E.bootstrap_ci(D)
    save(ci, tab / "09_bootstrap_ci.csv")
    save(E.cost_curve(D), tab / "10_cost_threshold.csv")
    levels, rules, thr = E.alarm_policy(D)
    save(levels, tab / "10_alarm_levels.csv")
    save(rules, tab / "10_alarm_rules.csv")
    save(E.unc_trigger_cost(D, thr), tab / "09c_unc_trigger_cost.csv")
    guard_eff = E.guard_effect(D, thr)
    save(guard_eff, tab / "12_guard_effect.csv")
    save(A.severity_recall(D, thr), tab / "14b_severity_recall.csv")
    if not a.quick:
        save(A.guard_mode_shift(w, mode), tab / "12b_guard_mode_shift.csv")
    F.fig_errors(tabs, fig)
    F.fig_error_cases(df, D, fig)
    F.fig_policy(D, thr, fig)
    log("error/policy analysis done")

    # 5. ablations ---------------------------------------------------------------------------------
    if not a.quick:
        save(A.imbalance_ablation(w, SEED, cols=cols), tab / "11_imbalance_ablation.csv")
        save(A.window_ablation(df, seed=SEED, invariant=final_set == "INV"), tab / "11_window_ablation.csv")
        save(A.overfit_check(w, cols, SEED), tab / "13_overfit_check.csv")
        weak, speed = A.stress_tests(df, w, cols, thr, final_model, SEED)
        save(weak, tab / "14_weak_fault.csv")
        save(speed, tab / "15_speed_shift.csv")
        # independent evaluation: model selection + thresholds redone inside each outer fold, outer-test windows scored once
        nest_folds, nest_pooled, nest_rules, _ = A.nested_validation(w, inv, SEED)
        save(nest_folds, tab / "16_nested_folds.csv")
        save(nest_pooled, tab / "16b_nested_pooled.csv")
        save(nest_rules, tab / "16c_nested_rules.csv")
        log("ablations done")

    # 6. final model + honest (out-of-fold) predictions ---------------------------------------------
    y = w["y"].to_numpy()
    bundle = fit_bundle(w[fc].reset_index(drop=True), y, w["block"].to_numpy(), SEED, cols)
    bundle["policy"] = {**thr, "window": a.window, "stride": a.stride, "final_model": final_model,
                        "final_set": final_set, "cur_guard_limit": bundle["guard"].limit_}
    bundle["policy"]["guard_limits"] = [float(v) for v in bundle["guard"].limits_]
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "model_bundle.joblib")
    o = foofs[0]
    pred = w[["seg", "source", "t_start", "t_end", "pos_in_seg", "win_len", "mode", "y"]].copy()
    for m in MODEL_ORDER:
        pred[f"p_{m}"] = o[m].to_numpy()
    pred["p_final"], pred["unc"] = o[final_model].to_numpy(), o["unc"].to_numpy()
    pred["guard"], pred["new_mode"] = o["guard"].to_numpy().astype(int), o["new_mode"].to_numpy().astype(int)
    pred["level"] = E.assign_level(pred.p_final, pred.unc, thr, pred.guard)
    pred["unc_flag"] = (pred.unc >= thr["unc_high"]).astype(int)
    save(pred, out / "predictions_oof.csv")

    summary_json = {
        "windows": len(w), "anomaly_windows": int(w.y.sum()), "n_features_all": len(fc), "n_features_invariant": len(inv),
        "segments_total": seg_all, "segments_used": seg_used, "modes_silhouette": {str(k): float(v) for k, v in sil.items()},
        "final_set": final_set, "final_model": final_model, "thresholds": thr,
        "bootstrap_ci": ci.set_index("metric").round(4).to_dict("index"),
        "cur_guard_limit": bundle["guard"].limit_, "guard_limits": [float(v) for v in bundle["guard"].limits_],
        "guard_effect": guard_eff.set_index("variant").round(4).to_dict("index"),
    }
    (out / "results_summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. numbers -> charts -> dashboard ------------------------------------------------------------
    print_summary(summary, ms, final_set, final_model, thr, guard_eff)
    if not a.quick:
        R.make_all(tab, fig, final_set, final_model, summary_json)
        log("result charts done")
        try:
            runpy.run_path(str(ROOT / "report" / "build_dashboard.py"), init_globals={"OUT_DIR": str(out)})
            if not a.no_open:
                webbrowser.open((out / "dashboard.html").resolve().as_uri())
        except Exception as e:      # the dashboard is a convenience; never fail the run because of it
            print(f"[warn] dashboard not built: {e}")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()

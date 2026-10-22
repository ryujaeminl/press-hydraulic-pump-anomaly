"""Report figures (matplotlib, Korean labels). Every function writes one PNG and returns its path."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from sklearn.metrics import precision_recall_curve

from .data import CH, SAMPLE_DT
from .errors import reliability
from .evaluate import MODEL_ORDER

LABEL = {"B0_SPC": "B0 SPC규칙", "B1_LogReg": "B1 로지스틱", "M1_RF": "M1 랜덤포레스트",
         "M2_HGB": "M2 HistGB", "M3_IF": "M3 IsoForest(정상전용)", "M4_Stack": "M4 스태킹앙상블"}
COLOR = {"B0_SPC": "#9e9e9e", "B1_LogReg": "#7b9acc", "M1_RF": "#5aa469", "M2_HGB": "#d98c3f",
         "M3_IF": "#a06cc4", "M4_Stack": "#c0392b"}
CH_LABEL = ["상부진동(AI0)", "하부진동(AI1)", "전류(AI2)"]


def setup_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110


def _save(fig, out: Path, name: str, top=None) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    if fig.get_layout_engine() is None:          # figures built with layout='constrained' place their own suptitle
        fig.tight_layout(rect=(0, 0, 1, top) if top else None)
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig_signals(df, out):
    fig, ax = plt.subplots(3, 2, figsize=(9, 6.2), sharex=True)
    for j, (src, title) in enumerate([("press_data_normal.csv", "정상 (state=0)"), ("outlier_data.csv", "이상 (state=1)")]):
        d = df[df.source == src]
        seg = d.groupby("seg").size()
        g = d[d.seg == seg[seg == 50].index[1]]
        t = np.arange(len(g)) * 0.1
        for i, c in enumerate(CH):
            ax[i, j].plot(t, g[c].to_numpy(), lw=1.1, color="#c0392b" if j else "#2c6fbb")
            ax[i, j].set_ylabel(CH_LABEL[i], fontsize=8)
            ax[i, j].grid(alpha=.3)
        ax[0, j].set_title(title)
        ax[2, j].set_xlabel("시간 [s] (연속 50샘플, 10Hz)")
    return _save(fig, out, "fig01_signals.png")


def fig_segments(df, out):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for src, c, lab in [("press_data_normal.csv", "#2c6fbb", "정상"), ("outlier_data.csv", "#c0392b", "이상")]:
        d = df[df.source == src]
        L = d.groupby("seg").size()
        ax[0].hist(L, bins=range(0, 55, 5), alpha=.6, color=c, label=f"{lab} ({len(L)}구간)", density=True)
        gap = d.groupby("seg").TimeStamp.min().sort_values().diff().dt.total_seconds().dropna()
        ax[1].hist(gap.clip(upper=15), bins=15, alpha=.6, color=c, label=lab, density=True)
    ax[0].set(xlabel="연속 구간 길이 [샘플]", ylabel="밀도", title="연속 수집 구간 길이(최대 50)")
    ax[1].set(xlabel="구간 시작 간격 [s]", title="구간 간 공백(15s 이상은 15로 절단)")
    ax[0].legend(); ax[1].legend()
    return _save(fig, out, "fig02_segments.png")


def fig_modes(w, mode, out):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    n = w.y.to_numpy() == 0
    for m, col in zip(sorted(set(mode[n])), ["#2c6fbb", "#5aa469", "#d98c3f", "#a06cc4"]):
        s = n & (mode == m)
        ax.scatter(w.vlow_rms[s], w.cur_rms[s], s=6, alpha=.35, color=col, label=f"정상 mode{m}")
    a = ~n
    ax.scatter(w.vlow_rms[a], w.cur_rms[a], s=16, marker="x", color="#c0392b", label="이상")
    ax.set(xlabel="하부진동 RMS", ylabel="전류 RMS", title="정상 운전모드(군집)와 이상 윈도우")
    ax.legend(fontsize=8)
    return _save(fig, out, "fig03_modes.png")


def fig_features(w, out):
    cols = [("vup_rms", "상부진동 RMS"), ("vup_zcr", "상부진동 영교차율"), ("vup_mad", "상부진동 평균|Δ|"),
            ("cur_mean", "전류 평균(DC 오프셋)"), ("vlow_rms", "하부진동 RMS"), ("vib_corr", "상·하부진동 상관")]
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.4))
    for a, (c, t) in zip(ax.ravel(), cols):
        lo, hi = np.percentile(w[c], [0.5, 99.5])
        for y, col, lab in [(0, "#2c6fbb", "정상"), (1, "#c0392b", "이상")]:
            a.hist(w.loc[w.y == y, c].clip(lo, hi), bins=30, alpha=.55, density=True, color=col, label=lab)
        a.set_title(t, fontsize=9)
    ax[0, 0].legend(fontsize=8)
    return _save(fig, out, "fig04_features.png")


def fig_cross_corr(curves, out):
    """curves: {(label, 'A~B'): mean cross-correlation vs lag array} from analysis.cross_correlation."""
    pairs = sorted({p for _, p in curves})
    max_lag = (len(next(iter(curves.values()))) - 1) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    fig, ax = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 3.4), sharey=True, layout="constrained")
    for a, pair in zip(np.atleast_1d(ax), pairs):
        for label, col in [("정상", "#2c6fbb"), ("이상", "#c0392b")]:
            if (label, pair) in curves:
                a.plot(lags, curves[(label, pair)], "o-", ms=3, color=col, label=label)
        a.axhline(0, color="#bbb", lw=.8)
        a.axvline(0, color="#bbb", lw=.8, ls="--")
        a.set_title(pair.replace("_Vibration", "").replace("_Current", ""), fontsize=9)
        a.set_xlabel("지연 [샘플, 0.1s]")
    np.atleast_1d(ax)[0].set_ylabel("정규화 상호상관")
    np.atleast_1d(ax)[0].legend(fontsize=8)
    fig.suptitle("원시 신호 채널 간 시차상관 (평균곡선; 표의 피크값은 구간별 개별 피크 평균)", fontsize=10.5)
    return _save(fig, out, "fig04b_cross_corr.png")


def fig_compare(summary, out):
    s = summary.set_index("model").loc[MODEL_ORDER]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.6))
    for a, (m, t) in zip(ax, [("F1", "F1-score (윈도우)"), ("PR_AUC", "PR-AUC"),
                              ("false_alarms_per_hour", "오경보 [건/시간, 정상 구간 기준]")]):
        a.bar([LABEL[k].split()[0] for k in s.index], s[m], yerr=s[m + "_sd"], color=[COLOR[k] for k in s.index], capsize=3)
        a.set_title(t, fontsize=10)
        a.tick_params(axis="x", labelsize=8)
    return _save(fig, out, "fig05_model_compare.png")


def fig_pr(oof, out):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for m in MODEL_ORDER:
        p, r, _ = precision_recall_curve(oof.y, oof[m])
        ax.plot(r, p, color=COLOR[m], label=LABEL[m], lw=1.6)
    ax.set(xlabel="Recall", ylabel="Precision", title="PR 곡선 (블록 교차검증 OOF)", ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=.3)
    return _save(fig, out, "fig06_pr_curves.png")


def fig_calibration(oof, out):
    """Reliability on fixed probability bins (quantile bins collapse here: only ~3% of windows are anomalies, so almost
    every quantile bin holds nothing but normals). Marker area = number of windows in the bin; empty bins are skipped."""
    edges = np.array([0, .01, .05, .2, .5, .8, .95, 1.0001])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    for m in ["B1_LogReg", "M1_RF", "M2_HGB", "M3_IF", "M4_Stack"]:
        p, y = oof[m].to_numpy(), oof.y.to_numpy()
        b = np.digitize(p, edges) - 1
        pts = [(p[b == k].mean(), y[b == k].mean(), (b == k).sum()) for k in range(len(edges) - 1) if (b == k).sum() >= 5]
        if pts:
            x, f, n = zip(*pts)
            ax.plot(x, f, "-", color=COLOR[m], lw=1, alpha=.8)
            ax.scatter(x, f, s=[12 + 60 * np.sqrt(v / max(n)) for v in n], color=COLOR[m], label=LABEL[m], zorder=3)
    ax.set(xlabel="예측 확률(보정 후)", ylabel="실제 이상 비율", title="신뢰도 곡선 (고정 확률 구간, 점 크기 = 창 수)", xlim=(-.03, 1.03), ylim=(-.03, 1.03))
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=.3)
    return _save(fig, out, "fig07_calibration.png")


def fig_importance(fi, gi, lgo, out, name="fig08_importance.png"):
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    t = fi.head(12).iloc[::-1]
    ax[0].barh(t.feature, t.perm_importance, xerr=t.perm_sd, color="#5aa469")
    ax[0].set_title("변수별 순열중요도 (AP 감소)", fontsize=10)
    g = gi.iloc[::-1]
    ax[1].barh(g["group"], g.perm_drop_AP, xerr=g.sd.fillna(0), color="#d98c3f")
    ax[1].set_title("그룹 순열중요도 (그룹 통째 섞기)", fontsize=10)
    l = lgo.iloc[::-1]
    ax[2].barh(l.removed_group, l.PR_AUC, color="#7b9acc")
    ax[2].set_xlim(l.PR_AUC.min() - 0.02, 1.005)
    ax[2].set_title("그룹 제거 후 재학습 PR-AUC", fontsize=10)
    for a in ax:
        a.tick_params(axis="y", labelsize=8)
    return _save(fig, out, name)


def fig_interaction(h, pdp_model, X, out):
    from sklearn.inspection import PartialDependenceDisplay
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    t = h.head(8).iloc[::-1]
    ax[0].barh([f"{a} × {b}" for a, b in zip(t.f1, t.f2)], t.H2, color="#a06cc4")
    ax[0].set_title("변수쌍 상호작용 강도 (Friedman H²)", fontsize=10)
    ax[0].tick_params(axis="y", labelsize=8)
    top = h.iloc[0]
    PartialDependenceDisplay.from_estimator(pdp_model, X, [(top.f1, top.f2)], ax=ax[1], grid_resolution=25)
    ax[1].set_title(f"2D 부분의존: {top.f1} × {top.f2}", fontsize=10)
    return _save(fig, out, "fig09_interaction.png")


def fig_errors(tabs, out):
    fp, fn = (tabs[k][tabs[k].kind == "기계 상태"] for k in ("fp_conditions", "fn_conditions"))   # processing rows: table only
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for a, t, ttl, col in [(ax[0], fp, "정상 윈도우(기계 상태별): 오경보 근접(p>=0.05) 비율", "#2c6fbb"),
                           (ax[1], fn, "이상 윈도우(기계 상태별): 미탐 근접(p<=0.95) 비율", "#c0392b")]:
        lab = [f"{c} | {l}" for c, l in zip(t.condition, t.level)][::-1]
        rate, n = t.near_rate.to_numpy()[::-1], t.n.to_numpy()[::-1]
        e, k = t.errors.to_numpy()[::-1], t.near_errors.to_numpy()[::-1]
        a.barh(lab, rate, color=col)
        for i in range(len(rate)):
            a.text(rate[i], i, f" 오류{int(e[i])}/근접{int(k[i])}/n={int(n[i])}", va="center", fontsize=6.5)
        a.set_title(ttl, fontsize=10)
        a.tick_params(axis="y", labelsize=7)
        a.set_xlim(0, max(rate.max() * 1.6, 0.05))
    return _save(fig, out, "fig10_error_conditions.png")


def fig_error_cases(df: pd.DataFrame, D: pd.DataFrame, out: Path):
    """Raw 3-channel waveform of the worst error windows (first CV repeat): up to 2 misses (lowest p among the anomalies the
    model left below its threshold) and 1 false alarm (highest p among the normals it flagged). The window is shaded."""
    d = D[D.rep == 0]
    cases = [("미탐(FN)", r, "#c0392b") for r in d[d.err == "FN"].nsmallest(2, "p").itertuples()]         + [("오경보(FP)", r, "#2c6fbb") for r in d[d.err == "FP"].nlargest(1, "p").itertuples()]
    if not cases:
        return None
    fig, ax = plt.subplots(3, len(cases), figsize=(4.3 * len(cases), 6.4), squeeze=False)
    for j, (kind, r, col) in enumerate(cases):
        g = df[df.seg == r.seg].reset_index(drop=True)
        t = np.arange(len(g)) * SAMPLE_DT
        for i, c in enumerate(CH):
            ax[i][j].plot(t, g[c].to_numpy(), lw=.9, color="#444")
            ax[i][j].axvspan(r.pos_in_seg, r.pos_in_seg + r.win_len * SAMPLE_DT, color=col, alpha=.22)
            ax[i][j].tick_params(labelsize=7)
            if j == 0:
                ax[i][j].set_ylabel(CH_LABEL[i], fontsize=8)
        ax[0][j].set_title(f"{kind} - 구간 {r.seg}, 모델 확률 p={r.p:.3g}", fontsize=9)
        ax[2][j].set_xlabel("구간 시작 후 시간 [s] (음영 = 판정 윈도우 1초)", fontsize=8)
    return _save(fig, out, "fig14_error_cases.png")


def fig_policy(D, thr, out):
    d = D[D.rep == 0].sort_values(["seg", "pos_in_seg"]).reset_index(drop=True)
    first_anom = int(d.index[d.y == 1][0])
    seg = d.iloc[max(first_anom - 120, 0): first_anom + 160]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = np.arange(len(seg))
    y = seg.y.to_numpy()
    ax.axhspan(0, thr["t_yellow"], color="#c8e6c9", alpha=.5)
    ax.axhspan(thr["t_yellow"], thr["t_red"], color="#fff3b0", alpha=.6)
    ax.axhspan(thr["t_red"], 1, color="#ffcdd2", alpha=.6)
    ax.plot(x, seg.p.to_numpy(), color="#444", lw=.8)
    ax.scatter(x[y == 1], seg.p.to_numpy()[y == 1], s=12, color="#c0392b", label="실제 이상 윈도우", zorder=3)
    ax.scatter(x[y == 0], seg.p.to_numpy()[y == 0], s=5, color="#2c6fbb", label="실제 정상 윈도우", zorder=3)
    ax.set(ylim=(-0.02, 1.02), xlabel="윈도우 순서(시간순)", ylabel="이상 확률",
           title="3단계 경보 (초록=정상 / 노랑=주의 / 빨강=경고) - 정상→이상 세션 전환부")
    ax.legend(fontsize=8, loc="center right")
    return _save(fig, out, "fig11_alarm_policy.png")


def fig_mode_shift(ms, out):
    sets = list(ms.feature_set.unique())
    fig, ax = plt.subplots(1, len(sets), figsize=(5.4 * len(sets), 3.6), sharey=True)
    for a, name in zip(np.atleast_1d(ax), sets):
        piv = ms[ms.feature_set == name].pivot_table(index="model", columns="held_out_mode",
                                                     values="FPR_on_unseen_mode").loc[MODEL_ORDER]
        piv.plot.bar(ax=a, color=["#2c6fbb", "#d98c3f", "#5aa469"][: piv.shape[1]], rot=20, legend=False)
        a.set_xticklabels([LABEL[m].split()[0] for m in piv.index])
        a.set(xlabel="", title={"ALL": "전체 특징(진폭 포함)", "INV": "운전모드 불변 특징"}.get(name, name))
    np.atleast_1d(ax)[0].set_ylabel("미학습 운전모드 정상 윈도우 FPR")
    np.atleast_1d(ax)[-1].legend(title="학습에서 제외한 모드", fontsize=8, loc="upper right")
    return _save(fig, out, "fig12_mode_shift.png")


def fig_feature_sets(sel, out):
    """Known-mode F1 vs unseen-mode F1 per model, for both feature sets."""
    models = [m for m in MODEL_ORDER if m != "B0_SPC"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    x = np.arange(len(models))
    for a, col, ttl in [(ax[0], "F1_known_modes", "기존 운전모드 (블록 교차검증)"), (ax[1], "F1_unseen_mode", "미학습 운전모드 외삽")]:
        for k, (name, c) in enumerate([("ALL", "#9e9e9e"), ("INV", "#c0392b")]):
            v = [sel[(sel.feature_set == name) & (sel.model == m)][col].iloc[0] for m in models]
            a.bar(x + (k - .5) * .38, v, .38, color=c, label={"ALL": "전체 특징", "INV": "불변 특징"}[name])
        a.set_xticks(x)
        a.set_xticklabels([LABEL[m].split()[0] for m in models])
        a.set_title(ttl, fontsize=10)
        a.set_ylim(0, 1.25)
    ax[0].set_ylabel("F1-score")
    ax[0].legend(fontsize=8, loc="upper center", ncol=2)
    return _save(fig, out, "fig13_feature_sets.png")

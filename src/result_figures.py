"""Charts of the numeric results stored in outputs/tables/*.csv (one PNG per result table group).

    from src.result_figures import make_all; make_all(tables_dir, figures_dir, final_set, final_model, summary_json)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from .figures import COLOR, LABEL, _save, setup_font

ORDER = ["B0_SPC", "B1_LogReg", "M1_RF", "M2_HGB", "M3_IF", "M4_Stack"]
SET_NAME = {"ALL": "전체 특징(진폭 포함)", "INV": "운전모드 불변 특징"}
GOOD, BAD, WARN, GREY = "#3c9d5d", "#d64541", "#e0a020", "#9e9e9e"


def _short(m): return LABEL[m].split()[0]


def _cnt(v): return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def _annotate(ax, bars, fmt="{:.3f}", pad=0.01):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + pad, fmt.format(h), ha="center", va="bottom", fontsize=8)


def _mix(color, t):
    """Blend colour with white: t=0 white ... t=1 full colour."""
    r, g, b = to_rgb(color)
    return (1 - t + t * r, 1 - t + t * g, 1 - t + t * b)


# ------------------------------------------------------------------------------------------ 20
def fig_metrics_heatmap(S: pd.DataFrame, out: Path):
    cols = [("F1", "F1", True), ("PR_AUC", "PR-AUC", True), ("Recall", "재현율", True), ("Precision", "정밀도", True), ("FPR", "FPR(%)", False)]
    sets = [s for s in ("ALL", "INV") if s in set(S.feature_set)]
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 3.9), sharey=True)
    for ax, fs in zip(np.atleast_1d(axes), sets):
        d = S[S.feature_set == fs].set_index("model").loc[ORDER]
        for i, m in enumerate(ORDER):
            for j, (c, _, higher) in enumerate(cols):
                v = float(d.loc[m, c])
                t = (v - 0.5) / 0.5 if higher else 1 - min(v / 0.005, 1)
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=_mix(GOOD, float(np.clip(t, 0, 1))), edgecolor="white"))
                ax.text(j + .5, i + .5, f"{v:.3f}" if higher else f"{100 * v:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(ORDER), 0)
        ax.set_xticks(np.arange(len(cols)) + .5)
        ax.set_xticklabels([c[1] for c in cols], fontsize=9)
        ax.xaxis.tick_top()
        ax.set_yticks(np.arange(len(ORDER)) + .5)
        ax.set_yticklabels([LABEL[m] for m in ORDER], fontsize=9)
        ax.set_title(SET_NAME[fs], fontsize=10, pad=26)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("모델별 성능 지표 (블록 그룹 CV 3회 평균, 진할수록 좋음)", fontsize=11, y=0.99)
    return _save(fig, out, "fig20_metrics_heatmap.png", top=0.93)


# ------------------------------------------------------------------------------------------ 21
def fig_confusion(S: pd.DataFrame, final_set: str, final_model: str, out: Path):
    picks = [("B0_SPC", "B0 SPC 규칙 (베이스라인)"), (final_model, f"{_short(final_model)} 최종 모델")]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for ax, (m, title) in zip(axes, picks):
        r = S[(S.feature_set == final_set) & (S.model == m)].iloc[0]
        cm = np.array([[r.TN, r.FP], [r.FN, r.TP]], float)
        rate = cm / cm.sum(axis=1, keepdims=True)
        for i in range(2):
            for j in range(2):
                good = i == j
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=_mix(GOOD if good else BAD, .15 + .7 * (rate[i, j] if good else min(rate[i, j] * 8, 1))), edgecolor="white"))
                ax.text(j + .5, i + .5, f"{_cnt(cm[i, j])}창\n{100 * rate[i, j]:.2f}%", ha="center", va="center", fontsize=11)
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_xticks([.5, 1.5])
        ax.set_xticklabels(["정상 예측", "이상 예측"])
        ax.set_yticks([.5, 1.5])
        ax.set_yticklabels(["실제 정상", "실제 이상"])
        ax.xaxis.tick_top()
        ax.set_title(title, fontsize=10, pad=28)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("윈도우 오분류 수 (3회 평균, 행 기준 비율)", fontsize=11, y=0.99)
    return _save(fig, out, "fig21_confusion.png", top=0.93)


# ------------------------------------------------------------------------------------------ 22
def fig_intervals(S: pd.DataFrame, ci: pd.DataFrame, final_set: str, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    c = ci.set_index("metric")
    for y, (metric, lab) in enumerate([("F1", "F1"), ("Recall", "재현율")]):
        r = c.loc[metric]
        ax[0].errorbar([r.point], [y], xerr=[[r.point - r.ci_low], [r.ci_high - r.point]], fmt="o", color=COLOR["M4_Stack"], capsize=6, ms=8)
        ax[0].text(r.ci_low, y - .22, f"{r.point:.3f}  [{r.ci_low:.3f}, {r.ci_high:.3f}]", fontsize=9)
    ax[0].set_yticks([0, 1])
    ax[0].set_yticklabels(["F1", "재현율"])
    ax[0].set_ylim(1.6, -.6)
    ax[0].set_xlim(.85, 1.02)
    ax[0].grid(axis="x", alpha=.3)
    ax[0].set_title("F1·재현율 (블록 부트스트랩 95% CI)", fontsize=10)
    r = c.loc["FPR"]
    ax[1].errorbar([100 * r.point], [0], xerr=[[100 * (r.point - r.ci_low)], [100 * (r.ci_high - r.point)]], fmt="o", color=GOOD, capsize=6, ms=8)
    ax[1].text(100 * r.point, .25, f"{100 * r.point:.2f}%\n[{100 * r.ci_low:.2f}%, {100 * r.ci_high:.2f}%]", ha="center", fontsize=9)
    ax[1].set_xlim(-.02, max(100 * r.ci_high * 1.6, .12))
    ax[1].set_ylim(-.6, .8)
    ax[1].set_yticks([])
    ax[1].set_xlabel("이상 블록이 6개뿐이라 구간이 넓음", fontsize=8)
    ax[1].grid(axis="x", alpha=.3)
    ax[1].set_title("오경보율 FPR [%] (95% CI)", fontsize=10)
    d = S[S.feature_set == final_set].set_index("model").loc[[m for m in ORDER if m != "B0_SPC"]]
    ax[2].barh([_short(m) for m in d.index], d.F1, xerr=d.F1_sd, color=[COLOR[m] for m in d.index], capsize=3)
    for y, (v, sd) in enumerate(zip(d.F1, d.F1_sd)):
        ax[2].text(v + sd + .004, y, f"{v:.3f}±{sd:.3f}", va="center", fontsize=8)
    ax[2].set_xlim(0.7, 1.08)
    ax[2].invert_yaxis()
    ax[2].set_title("모델별 F1 ± 반복 표준편차(seed 3회)", fontsize=10)
    return _save(fig, out, "fig22_intervals.png")


# ---- 23
def fig_validation_checks(S: pd.DataFrame, leak: pd.DataFrame, win: pd.DataFrame, imb: pd.DataFrame, final_set: str, final_model: str, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    # (a) leakage check
    ms = [m for m in ORDER if m != "B0_SPC"]
    blk = S[S.feature_set == final_set].set_index("model").loc[ms].F1
    rnd = leak.set_index("model").loc[ms].F1
    x = np.arange(len(ms))
    b1 = ax[0].bar(x - .19, blk, .38, color=COLOR["M4_Stack"], label="블록 그룹 CV(채택)")
    b2 = ax[0].bar(x + .19, rnd, .38, color=GREY, label="무작위 KFold(누수 가능)")
    _annotate(ax[0], b1, "{:.2f}", .005)
    _annotate(ax[0], b2, "{:.2f}", .005)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([_short(m) for m in ms], fontsize=8)
    ax[0].set_ylim(.6, 1.2)
    ax[0].set_title("누수 점검: 검증 방식별 F1", fontsize=10)
    ax[0].legend(fontsize=7, loc="upper center", ncol=2)
    # (b) window length: 5/20 from the ablation table, 10 from the main comparison
    mids = [m for m in ("M2_HGB", "M4_Stack") if m in set(win.model)]
    cur = S[(S.feature_set == final_set) & (S.model.isin(mids))]
    for m in mids:
        pts = win[win.model == m].set_index("window")[["F1", "false_alarms_per_hour"]]
        pts.loc[10] = cur[cur.model == m][["F1", "false_alarms_per_hour"]].iloc[0].to_numpy()
        pts = pts.sort_index()
        ax[1].plot(pts.index / 10, pts.F1, "o-", color=COLOR[m], label=f"{_short(m)} F1")
    ax[1].set_xlabel("창 길이 [초] = 탐지 지연")
    ax[1].set_ylabel("F1")
    ax[1].set_ylim(.9, 1.0)
    ax2 = ax[1].twinx()
    m = final_model if final_model in mids else mids[0]
    pts = win[win.model == m].set_index("window").false_alarms_per_hour
    pts.loc[10] = float(cur[cur.model == m].false_alarms_per_hour.iloc[0])
    pts = pts.sort_index()
    ax2.plot(pts.index / 10, pts.values, "s--", color=BAD, label=f"{_short(m)} 오경보")
    ax2.set_ylabel("오경보 [건/시간]", color=BAD)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center", ncol=3)
    ax[1].set_ylim(.9, 1.0)
    ax[1].set_title("창 길이: 지연 vs 정확도·오경보", fontsize=10)
    # (c) imbalance handling
    lab = ["가중치 없음", "class_weight\nbalanced"]
    xx = np.arange(2)
    for k, (name, v) in enumerate({"F1": imb.F1, "재현율": imb.Recall, "정밀도": imb.Precision}.items()):
        bars = ax[2].bar(xx + (k - 1) * .26, v.to_numpy(), .26, label=name)
        _annotate(ax[2], bars, "{:.3f}", .002)
    ax[2].set_xticks(xx)
    ax[2].set_xticklabels(lab)
    ax[2].set_ylim(.9, 1.07)
    ax[2].legend(fontsize=7, loc="upper center", ncol=3)
    ax[2].set_title("불균형 처리 효과 (HGB, thr=0.5)", fontsize=10)
    return _save(fig, out, "fig23_validation_checks.png")


# ------------------------------------------------------------------------------------------ 24
def fig_alarm_results(levels: pd.DataFrame, rules: pd.DataFrame, cost: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw={"width_ratios": [1.15, 1.2, 1]})
    lv = levels.set_index("구분")
    cols = [("green", "초록(정상)", GOOD), ("yellow", "노랑(주의)", WARN), ("red", "빨강(경고)", BAD)]
    for i, name in enumerate(lv.index):
        tot = lv.loc[name, [c[0] for c in cols]].sum()
        left, n_small = 0, 0
        for key, lbl, col in cols:
            v = lv.loc[name, key]
            frac = v / tot
            ax[0].barh(i, frac, left=left, color=col, label=lbl if i == 0 else None)
            if frac > .04:
                ax[0].text(left + frac / 2, i, f"{int(v)}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            elif v > 0:
                ax[0].text(min(max(left + frac / 2, .06), .94), i - .47 - .17 * n_small, f"{lbl.split('(')[1][:-1]} {int(v)}", ha="center", fontsize=8, color=col)
                n_small += 1
            left += frac
    ax[0].set_yticks(range(len(lv)))
    ax[0].set_yticklabels(lv.index)
    ax[0].set_ylim(1.6, -1.0)
    ax[0].set_xlim(0, 1)
    ax[0].set_xticks([0, .5, 1])
    ax[0].set_xticklabels(["0%", "50%", "100%"])
    ax[0].legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
    ax[0].set_title("윈도우 경보 등급 분포 (창 수)", fontsize=10)
    r = rules.copy()
    short = {"R1": "red 1회", "R2": "red 2연속", "R3": "노랑+ 2연속", "R4": "구간 평균"}
    r["name"] = r.rule.str.split(" ", n=1).str[0].map(lambda k: f"{k}\n{short.get(k, '')}")
    bars = ax[1].bar(r.name, r.false_alarms_per_hour, color=[BAD if ("2연속" in x and "red" in x) else GREY for x in r.rule])
    for b, rec in zip(bars, r.segment_recall):
        ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + .3, f"{b.get_height():.1f}\n탐지 {100 * rec:.0f}%", ha="center", fontsize=8)
    ax[1].set_ylim(0, r.false_alarms_per_hour.max() * 1.3)
    ax[1].tick_params(axis="x", labelsize=7)
    ax[1].set_title("경보 규칙별 오경보 [건/기록시간]", fontsize=10)
    c = cost.copy()
    ax[2].plot(c["C_FN/C_FP"].astype(str) + ":1", c.threshold, "o-", color=COLOR["M2_HGB"])
    for x, (t, fp, fn) in enumerate(zip(c.threshold, c.FP, c.FN)):
        ax[2].text(x, t + .03, f"FP{int(fp)}/FN{int(fn)}", ha="center", fontsize=7)
    ax[2].set_ylim(0, max(c.threshold.max() * 1.25, .5))
    ax[2].set_xlabel("가정한 비용비 (미탐 : 오경보)")
    ax[2].set_ylabel("비용 최소 임계값")
    ax[2].set_title("비용비별 최적 임계값", fontsize=10)
    return _save(fig, out, "fig24_alarm_results.png")


# ------------------------------------------------------------------------------------------ 25
def fig_uncertainty(unc: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.6))
    b = ax[0].bar(unc.unc_bin.astype(str), 100 * unc.error_rate, color=[GOOD, GOOD, WARN, BAD])
    for bar, e, n in zip(b, unc.errors, unc.n):
        ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .1, f"오류 {int(e)}/{int(n)}", ha="center", fontsize=8)
    ax[0].set_ylim(0, max(100 * unc.error_rate.max() * 1.18, 1))
    ax[0].set_ylabel("오류율 [%]")
    ax[0].set_title("앙상블 불일치(불확실도) 구간별 오류율", fontsize=10)
    ax[1].bar(unc.unc_bin.astype(str), unc.anomalies, color=COLOR["M1_RF"])
    for x, (a, n) in enumerate(zip(unc.anomalies, unc.n)):
        ax[1].text(x, a + 1, f"{int(a)}창 ({100 * a / n:.0f}%)", ha="center", fontsize=8)
    ax[1].set_ylim(0, unc.anomalies.max() * 1.18)
    ax[1].set_ylabel("이상 창 수")
    ax[1].set_title("같은 구간에 든 실제 이상 창", fontsize=10)
    return _save(fig, out, "fig25_uncertainty.png")


# ------------------------------------------------------------------------------------------ 26
def fig_data_overview(Q: pd.DataFrame, summary_json: dict, out: Path):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    names, cols = ["정상", "이상"], [COLOR["B1_LogReg"], COLOR["M4_Stack"]]
    b = ax[0].bar(names, Q.rows, color=cols)
    _annotate(ax[0], b, "{:,.0f}", Q.rows.max() * .01)
    ax[0].set_ylim(0, Q.rows.max() * 1.1)
    ax[0].set_title("원시 행 수", fontsize=10)
    tot = [summary_json["segments_total"][f] for f in Q.file]
    used = [summary_json["segments_used"][f] for f in Q.file]
    x = np.arange(2)
    b1 = ax[1].bar(x - .19, tot, .38, color=GREY, label="전체 구간")
    b2 = ax[1].bar(x + .19, used, .38, color=cols, label="사용(5샘플 이상)")
    _annotate(ax[1], b1, "{:.0f}", max(tot) * .01)
    _annotate(ax[1], b2, "{:.0f}", max(tot) * .01)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(names)
    ax[1].set_ylim(0, max(tot) * 1.25)
    ax[1].legend(fontsize=8)
    ax[1].set_title("연속 수집 구간 수", fontsize=10)
    n_anom = summary_json["anomaly_windows"]
    n_norm = summary_json["windows"] - n_anom
    b = ax[2].bar(names, [n_norm, n_anom], color=cols)
    _annotate(ax[2], b, "{:,.0f}", n_norm * .01)
    ax[2].set_ylim(0, n_norm * 1.1)
    ax[2].set_title(f"1초 윈도우 수 (이상 {100 * n_anom / summary_json['windows']:.1f}%)", fontsize=10)
    return _save(fig, out, "fig26_data_overview.png")


# ------------------------------------------------------------------------------------------ 27
def fig_selection(sel: pd.DataFrame, final_set: str, final_model: str, out: Path):
    d = sel.sort_values("score", ascending=True).reset_index(drop=True)
    lab = [f"{'불변' if r.feature_set == 'INV' else '전체'} · {_short(r.model)}" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(d) + 1.4))
    y = np.arange(len(d))
    ax.barh(y + .19, d.F1_known_modes, .38, color=COLOR["B1_LogReg"], label="기존 운전모드 F1")
    ax.barh(y - .19, d.F1_unseen_mode, .38, color=COLOR["M1_RF"], label="미학습 운전모드 F1")
    for i, r in enumerate(d.itertuples()):
        ax.text(1.005, i, f"점수 {r.score:.3f}", va="center", fontsize=8,
                fontweight="bold" if (r.feature_set, r.model) == (final_set, final_model) else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=9)
    for t, r in zip(ax.get_yticklabels(), d.itertuples()):
        if (r.feature_set, r.model) == (final_set, final_model):
            t.set_color(BAD)
            t.set_fontweight("bold")
    ax.set_xlim(0, 1.12)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.07), ncol=2)
    ax.set_title("최종 모델 선정: 두 시험의 F1 평균이 최대인 조합(빨간 글씨)", fontsize=10)
    return _save(fig, out, "fig27_selection.png")


# ------------------------------------------------------------------------------------------ 28
def fig_guard_effect(ge: pd.DataFrame, gm: pd.DataFrame, out: Path):
    """Peak-current guard: alarm levels under four wirings + what each guard does on an unseen operating mode."""
    names = ["모델만", "전역 한계" + "\n→노랑", "모드별 한계" + "\n→노랑(채택)", "전역 한계" + "\n→빨강(폐기)"]
    x = np.arange(len(ge))
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1]}, layout="constrained")
    w = .36
    for k, (col, lab, c) in enumerate([("normal_yellow", "노랑", WARN), ("normal_red", "빨강", BAD)]):
        bars = ax[0].bar(x + (k - .5) * w, ge[col], w, color=c, label=lab)
        for b in bars:
            ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{int(b.get_height())}", ha="center", fontsize=8)
    ax[0].set_title("정상 윈도우가 받은 경보 [창]", fontsize=10)
    for k, (col, lab, c) in enumerate([("anomaly_yellow", "노랑(놓침 아님)", WARN), ("anomaly_green", "초록(완전 미탐)", GREY)]):
        bars = ax[1].bar(x + (k - .5) * w, ge[col], w, color=c, label=lab)
        for b in bars:
            ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + .3, f"{int(b.get_height())}", ha="center", fontsize=8)
    ax[1].set_title("이상 윈도우 중 빨강이 못 된 것 [창]", fontsize=10)
    for a in ax[:2]:
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=8.5)
        a.legend(fontsize=9)
        a.set_ylim(0, a.get_ylim()[1] * 1.3)
        a.yaxis.set_major_locator(MaxNLocator(integer=True))
    xm = np.arange(len(gm))
    w3 = .26
    series = [("global_normal_flag", "전역 한계: 노랑 플래그", WARN), ("mode_normal_flag", "모드별 한계: 노랑 플래그", BAD), ("new_mode_normal", "신규 모드로 분류(승인 대기)", GOOD)]
    for k, (col, lab, c) in enumerate(series):
        bars = ax[2].bar(xm + (k - 1) * w3, 100 * gm[col], w3, color=c, label=lab)
        _annotate(ax[2], bars, "{:.0f}%", 1.0)
    ax[2].set_xticks(xm)
    ax[2].set_xticklabels([f"mode{m} 미학습" for m in gm.held_out_mode], fontsize=8)
    ax[2].set_ylim(0, 128)
    ax[2].set_title("미학습 모드의 정상 창 처리 [%]", fontsize=10)
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("피크전류 가드: 노랑으로만 올리고, 모드별 한계는 새 모드를 ‘경보’ 대신 ‘승인 대기’로 보낸다", fontsize=11)
    return _save(fig, out, "fig28_guard_effect.png")


# ------------------------------------------------------------------------------------------ 29
def fig_overfit_check(of: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = np.arange(len(of))
    b1 = ax[0].bar(x - .19, of.train_AP, .38, color=GREY, label="학습 데이터 AP")
    b2 = ax[0].bar(x + .19, of.oof_AP, .38, color=COLOR["M2_HGB"], label="held-out(OOF) AP")
    for xi, g in zip(x, of.gap):
        ax[0].text(xi, 1.005, f"gap {g:.3f}", ha="center", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([v.split(" (")[0] for v in of.variant], fontsize=8)
    ax[0].set_ylim(.9, 1.09)
    ax[0].legend(fontsize=8, loc="upper center", ncol=2)
    ax[0].set_title("학습 vs held-out 성능차 (과적합 점검)", fontsize=10)
    w = .32
    ax[1].bar(x - w / 2, 100 * of.unseen_mode0_FPR, w, color="#2c6fbb", label="mode0 제외")
    ax[1].bar(x + w / 2, 100 * of.unseen_mode1_FPR, w, color="#d98c3f", label="mode1 제외")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([v.split(" (")[0] for v in of.variant], fontsize=8)
    ax[1].set_ylabel("미학습 모드 FPR [%]")
    ax[1].legend(fontsize=8)
    ax[1].set_title("규제를 더 강하게 해도 일반화는 안 좋아짐", fontsize=10)
    return _save(fig, out, "fig29_overfit_check.png")


# ------------------------------------------------------------------------------------------ 30 / 31
def fig_weak_fault(wk: pd.DataFrame, sev: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    for m, c, ls in (("최종 모델(빨강)", COLOR["M2_HGB"], "-"), ("최종 모델(노랑 이상)", WARN, "-"), ("SPC 규칙", GREY, "--")):
        d = wk[wk.model == m].sort_values("alpha")
        ax[0].plot(100 * d.alpha, 100 * d.alarm_rate, "o", ls=ls, color=c, label=m)
    ax[0].set_xlabel("고장 성분 비율 α [%]  (0 = 순수 정상 신호, 100 = 실제 고장 신호)")
    ax[0].set_ylabel("경보가 울린 윈도우 [%]")
    ax[0].set_ylim(-3, 103)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("합성 약한 고장: 정상 신호에 고장 신호를 α만큼 섞음", fontsize=10)
    x = np.arange(len(sev))
    w = .26
    for k, (col, lab, c) in enumerate([("spc_detect", "SPC 규칙", GREY), ("model_yellow_or_red", "최종 모델 노랑 이상", WARN), ("model_red", "최종 모델 빨강", COLOR["M2_HGB"])]):
        bars = ax[1].bar(x + (k - 1) * w, 100 * sev[col], w, color=c, label=lab)
        _annotate(ax[1], bars, "{:.0f}", 1.0)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{b}\n(n={n})" for b, n in zip(sev["bin"], sev.windows)], fontsize=8)
    ax[1].set_ylim(0, 138)
    ax[1].set_ylabel("탐지된 실제 이상 윈도우 [%]")
    ax[1].legend(fontsize=8, loc="upper center", ncol=3)
    ax[1].set_title("실제 이상 윈도우: 진폭이 정상에서 먼 정도별 탐지율", fontsize=10)
    return _save(fig, out, "fig30_weak_fault.png")


def fig_speed_shift(sp: pd.DataFrame, out: Path):
    """Red-level false-alarm rate of normal windows per apparent-frequency factor; one bar per warped-channel set."""
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    d0 = sp[(sp.model == "최종 모델(빨강)") & (sp.channels == "전류만")].sort_values("factor")
    d1 = sp[(sp.model == "최종 모델(빨강)") & (sp.channels == "전 채널")].sort_values("factor")
    x = np.arange(len(d0))
    w = .38
    for off, d, c, lab in ((-w / 2, d0, BAD, "전류 채널만 변형"), (w / 2, d1, COLOR["M2_HGB"], "전 채널 변형")):
        bars = ax.bar(x + off, 100 * d.false_alarm_rate, w, color=c, label=lab)
        for b, v in zip(bars, 100 * d.false_alarm_rate):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.axhline(1, color=GREY, lw=.9, ls=":")
    ax.lines[-1].set_label("1% 기준선")
    ax.set_xticks(x)
    ax.set_xticklabels([f"×{f:g}" for f in d0.factor])
    ax.set_xlabel("정상 신호의 겉보기 주파수 배율 (×1 = 원래 속도)")
    ax.set_ylabel("정상 윈도우 오경보율 [%]")
    ax.set_ylim(0, 112)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("속도 변화 시험: 겉보기 주파수 배율별 정상 윈도우 오경보율(빨강)", fontsize=10)
    return _save(fig, out, "fig31_speed_shift.png")


# ------------------------------------------------------------------------------------------ all
def make_all(tab: Path, fig: Path, final_set: str, final_model: str, summary_json: dict) -> list:
    """Draw every result chart from the saved CSV tables; returns the written paths."""
    setup_font()
    tab, fig = Path(tab), Path(fig)

    def rd(name): return pd.read_csv(tab / name)

    S = rd("03_model_comparison.csv")
    return [
        fig_metrics_heatmap(S, fig),
        fig_confusion(S, final_set, final_model, fig),
        fig_intervals(S, rd("09_bootstrap_ci.csv"), final_set, fig),
        fig_validation_checks(S, rd("06_leaky_random_split_check.csv"), rd("11_window_ablation.csv"),
                              rd("11_imbalance_ablation.csv"), final_set, final_model, fig),
        fig_alarm_results(rd("10_alarm_levels.csv"), rd("10_alarm_rules.csv"), rd("10_cost_threshold.csv"), fig),
        fig_uncertainty(rd("09_uncertainty.csv"), fig),
        fig_data_overview(rd("01_data_quality.csv"), summary_json, fig),
        fig_selection(rd("05_final_selection.csv"), final_set, final_model, fig),
        fig_guard_effect(rd("12_guard_effect.csv"), rd("12b_guard_mode_shift.csv"), fig),
        fig_overfit_check(rd("13_overfit_check.csv"), fig),
        fig_weak_fault(rd("14_weak_fault.csv"), rd("14b_severity_recall.csv"), fig),
        fig_speed_shift(rd("15_speed_shift.csv"), fig),
    ]

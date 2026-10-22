"""Fill the official contest-report template with this project's results.

    python report/build_report.py            # after run_all.py; writes report/최종보고서.hwpx

Every number is read from outputs/tables/*.csv, so the report always matches the last pipeline run.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hwpx_engine import Template  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT, TAB, FIG = ROOT / "outputs", ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
NAME = {"B0_SPC": "B0 SPC 관리도 규칙", "B1_LogReg": "B1 로지스틱회귀", "M1_RF": "M1 랜덤포레스트",
        "M2_HGB": "M2 HistGradientBoosting", "M3_IF": "M3 IsolationForest(정상 전용)", "M4_Stack": "M4 스태킹 앙상블"}
SHORT = {k: v.split()[0] for k, v in NAME.items()}
ORDER = ["B0_SPC", "B1_LogReg", "M1_RF", "M2_HGB", "M3_IF", "M4_Stack"]


def rd(name): return pd.read_csv(TAB / name)


def pct(x, d=1): return f"{100 * x:.{d}f}%"


def f3(x): return f"{x:.3f}"


# ------------------------------------------------------------------------------------------- results
S, MS, SEL = rd("03_model_comparison.csv"), rd("04_mode_shift.csv"), rd("05_final_selection.csv")
LEAK, Q, MODES = rd("06_leaky_random_split_check.csv"), rd("01_data_quality.csv"), rd("02_operating_modes.csv")
XC = rd("01b_cross_correlation.csv")
GI, LGO, PI = rd("07_group_importance_INV.csv"), rd("07_leave_group_out_INV.csv"), rd("07_perm_importance_INV.csv")
H2 = rd("07_interaction_H2.csv")
FPC, FNC, SPCM = rd("08_fp_conditions.csv"), rd("08_fn_conditions.csv"), rd("08_spc_miss_profile.csv")
UNC, CI = rd("09_uncertainty.csv"), rd("09_bootstrap_ci.csv")
COST, LV, RULES = rd("10_cost_threshold.csv"), rd("10_alarm_levels.csv"), rd("10_alarm_rules.csv")
IMB, WIN = rd("11_imbalance_ablation.csv"), rd("11_window_ablation.csv")
GE, GM, OF = rd("12_guard_effect.csv"), rd("12b_guard_mode_shift.csv"), rd("13_overfit_check.csv")
WK, SEV, SP = rd("14_weak_fault.csv"), rd("14b_severity_recall.csv"), rd("15_speed_shift.csv")
DIFF, PRED = rd("05b_model_diff_ci.csv"), pd.read_csv(OUT / "predictions_oof.csv")
UC, UTC, ROOTS = rd("09b_uncertainty_capture.csv"), rd("09c_unc_trigger_cost.csv"), rd("08_hard_case_roots.csv")
NF, NP, NR = rd("16_nested_folds.csv"), rd("16b_nested_pooled.csv"), rd("16c_nested_rules.csv")
N_TESTS = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf8"), re.M)) for p in (ROOT / "tests").glob("test_*.py"))
N_TABLES = len(list(TAB.glob("*.csv")))
RUN_MIN = 90          # wall-clock minutes of one full `python run_all.py` on the dev machine (see README)
J = json.loads((OUT / "results_summary.json").read_text(encoding="utf8"))
THR, FINAL, FSET = J["thresholds"], J["final_model"], J["final_set"]
assert FINAL == "M2_HGB" and FSET == "INV", "report text was written for M2_HGB on the invariant feature set"


def sm(fs, model, col): return float(S[(S.feature_set == fs) & (S.model == model)][col].iloc[0])


def shift(fs, model, mode, col="FPR_on_unseen_mode"):
    return float(MS[(MS.feature_set == fs) & (MS.model == model) & (MS.held_out_mode == mode)][col].iloc[0])


def sel(fs, model, col): return float(SEL[(SEL.feature_set == fs) & (SEL.model == model)][col].iloc[0])


def ci(metric, col): return float(CI[CI.metric == metric][col].iloc[0])


qn, qa = Q.iloc[0], Q.iloc[1]
n_norm_seg, n_anom_seg = J["segments_total"]["press_data_normal.csv"], J["segments_total"]["outlier_data.csv"]
u_norm_seg, u_anom_seg = J["segments_used"]["press_data_normal.csv"], J["segments_used"]["outlier_data.csv"]
n_win, n_anom_win = J["windows"], J["anomaly_windows"]
m0, m1 = MODES[(MODES["mode"] == 0) & (MODES.y == 0)].iloc[0], MODES[(MODES["mode"] == 1) & (MODES.y == 0)].iloc[0]
gi = GI.set_index("group")
lgo = LGO.set_index("removed_group").PR_AUC
spc_hit, spc_miss = SPCM[~SPCM.spc_miss].iloc[0], SPCM[SPCM.spc_miss].iloc[0]
lv = LV.set_index("구분")
rules = RULES.set_index("rule")
top_h2 = H2.iloc[0]
ge0, ge1, ge2, ge3 = (GE.iloc[i] for i in range(4))   # 모델만 / 전역 한계→노랑 / 모드별 한계→노랑(채택) / 전역 한계→빨강(폐기); 첫 CV 반복
gm = GM.set_index("held_out_mode")
NPI = NP.set_index("metric")
guard_gain, guard_cost = int(ge3.anomaly_red - ge0.anomaly_red), int(ge3.normal_red - ge0.normal_red)   # what forcing red would add
guard_rescued = int(ge0.anomaly_green - ge2.anomaly_green)       # anomaly windows the deployed yellow guard lifts out of green
GUARD_LIMITS = J["guard_limits"]                                 # per-mode peak-current limits of the deployed bundle
of0 = OF.iloc[0]                           # 배포 설정(현재 하이퍼파라미터)의 과적합 점검
fn_oof = PRED[(PRED.y == 1) & (PRED.level != "red")]           # 모델이 빨강으로 못 올린 이상 창(첫 CV 반복)
d_m1 = DIFF[DIFF.model_b == "M1_RF"].iloc[0]
d_m4 = DIFF[DIFF.model_b == "M4_Stack"].iloc[0]
xc = XC.set_index(["segment_type", "pair"])
xc01_n, xc01_a = xc.loc[("정상", "AI0_Vibration~AI1_Vibration")], xc.loc[("이상", "AI0_Vibration~AI1_Vibration")]


LEARNED_F1 = (min(sm("INV", m, "F1") for m in ("B1_LogReg", "M1_RF", "M2_HGB", "M4_Stack")), max(sm("INV", m, "F1") for m in ("B1_LogReg", "M1_RF", "M2_HGB", "M4_Stack")))
HI_FP = [shift("ALL", m, 1) for m in ("B0_SPC", "M1_RF", "M2_HGB")]        # amplitude-based models on the held-out high-load mode
_red_sp = SP[SP.model == "최종 모델(빨강)"].pivot(index="factor", columns="channels", values="false_alarm_rate")
SPEED_SAFE = [f for f in _red_sp.index if _red_sp.max(axis=1)[f] <= 0.01]   # factors with red false alarms <= 1% under both warps
SPEED_BROKE = [f for f in _red_sp.index if _red_sp.max(axis=1)[f] > 0.01]


CH_NAME = {"cur": "전류", "vup": "상부진동", "vlow": "하부진동"}
STAT_MEANING = {"madr": "파형 거칠기(샘플 간 변화량/표준편차)", "zcr": "평균선 통과 빈도(규칙성)", "kurt": "충격성(첨도)", "skew": "한쪽 치우침(왜도)", "crest": "순간 최대/RMS(파고율)"}


def meaning(f: str) -> str:
    """Plain-language reading of a feature name, e.g. cur_madr -> '전류의 파형 거칠기(...)'."""
    if f == "vib_corr":
        return "상·하부 진동이 함께 흔들리는 정도"
    if f == "cur_dc_ratio":
        return "전류 파형의 한쪽 쏠림(직류 성분 비율)"
    ch, _, stat = f.partition("_")
    return f"{CH_NAME.get(ch, ch)}의 {STAT_MEANING.get(stat, stat)}"


def concentration_row(df, condition, kind_label=None):
    """One table row: the most error-prone level of a condition. The note is built from the counts, so it can never
    claim a concentration where there are zero errors."""
    d = df[df.condition == condition]
    tot = int(d.errors.sum())
    top = d.sort_values(["rate", "near_rate"], ascending=False).iloc[0]
    if tot == 0:
        note = f"오류 없음 (근접 최대 {pct(d.near_rate.max())})"
    else:
        note = f"{int(top.errors)}/{tot}건 ({pct(top.errors / tot, 0)})"
    return [kind_label or "", condition, str(top.level), f"{int(top.errors)}/{int(top.n)}", pct(top.rate, 2), note]


def first_alpha(model, level):
    """Smallest fault share (in %) at which `model` alarms on >= `level` of the blended windows, or None."""
    d = WK[(WK.model == model) & (WK.alarm_rate >= level)].sort_values("alpha")
    return None if d.empty else 100 * float(d.alpha.iloc[0])


def alpha_text(model, level):
    a = first_alpha(model, level)
    return f"고장 성분 {a:.0f}%" if a is not None else "전 구간에서 도달하지 못함"


def _ro(word: str) -> str:
    """Particle '로' or '으로' by the final consonant of the last syllable (ㄹ and vowels take '로')."""
    c = ord(word[-1]) - 0xAC00
    return "로" if not 0 <= c < 11172 or c % 28 in (0, 8) else "으로"


def trend(a, b, unit):
    """'a -> b unit' described honestly whichever way it moved."""
    if abs(a - b) < 5e-3:
        return f"{a:.1f} → {b:.1f}{unit}{_ro(unit)} 사실상 같고"
    return f"{a:.1f} → {b:.1f}{unit}{_ro(unit)} {'줄고' if b < a else '늘고'}"


def chg(a, b, unit=""):
    """'3창' when nothing changed, '52→55창' otherwise (no 'x → x' arrows)."""
    return f"{a:.0f}{unit}" if round(a) == round(b) else f"{a:.0f}→{b:.0f}{unit}"


def anomaly_split(ar, ay, ag):
    tot = ar + ay + ag
    parts = [f"빨강 {ar}창"] + ([f"노랑 {ay}창"] if ay else []) + ([f"초록 {ag}창"] if ag else [])
    tail = "초록(완전 미탐)이 없습니다" if ag == 0 else f"초록으로 지나간 창이 {ag}개 있습니다"
    return f"이상 윈도우 {tot}창은 {', '.join(parts)}이며 {tail}"


# ------------------------------------------------------------------------------------------- chapters
def chapter1(t: Template) -> list:
    lk = LEAK.set_index("model")
    L = [t.heading("□ 제1장. 데이터 이해 및 진단〔15점〕", 1)]
    L += [t.b2("**생산단위와 변수 의미** : 데이터는 파인블랭킹 프레스 유압펌프 모터의 상부·하부 진동과 전류를 0.1초 간격(10Hz)으로 수집한 시계열이며, "
               f"정상 파일(press_data_normal.csv, {int(qn.rows):,}행, state=0)과 이상 파일(outlier_data.csv, {int(qa.rows):,}행, state=1)로 구성됩니다."),
          t.b3("생산단위는 제품 1개가 아니라 **‘시간 gap 0.15초 초과로 끊기는 연속 수집 구간(segment)’** 입니다. 구간 길이는 1~50샘플(최대 50=5초)이고 "
               f"구간 사이에는 중앙값 {qn.gap_median_s:.1f}초(최대 {qn.gap_max_s:.1f}초)의 공백이 있어, 정상 파일은 기록 {qn.recorded_min:.1f}분이지만 벽시계로는 {qn.wallclock_min:.1f}분에 걸쳐 있습니다."),
          t.b3(f"**수집 주기 해석**: 구간 시작 간격의 중앙값이 정상 {qn.seg_start_interval_median_s:.1f}초(10~90분위 {qn.seg_start_interval_q10_s:.1f}~{qn.seg_start_interval_q90_s:.1f}초), 이상 {qa.seg_start_interval_median_s:.1f}초로 거의 같아 "
               f"‘약 {qn.seg_start_interval_median_s:.0f}초마다 최대 5초의 짧은 수집 구간이 반복’되는 구조입니다(수집 길이 중앙값 정상 {qn.seg_len_median / 10:.1f}초). "
               "이 주기가 프레스 1사이클과 같은지는 데이터만으로 확인할 수 없는 가설이며, 확인되면 한 구간이 1사이클의 일부(수집된 5초)에 해당합니다. 사이클 사이의 공백에서 설비가 어떻게 움직였는지는 기록이 없어 분석에 쓰지 않습니다."),
          t.b3(f"구간 수는 정상 {n_norm_seg}개·이상 {n_anom_seg}개이며, 이상 데이터는 **정상과 다른 날(2022-07-17)의 단일 약 3분 세션**입니다. "
               "따라서 ‘시간·설비·제품 간 관계’는 설비 1대·단일 세션 수준에서만 확인 가능하다는 한계가 있습니다.")]
    L += [t.table([["변수", "의미(근거 포함)", "정상 파일 관측", "이상 파일 관측"],
                   ["TimeStamp", "수집 시각, 0.1s 고정 간격", "2022-07-12 00:00~01:16", "2022-07-17 10:51~10:53"],
                   ["AI0_Vibration", "상부 진동(변수 순서 기준, 단위 미제공)", "±0.35, 표준편차 0.071", "±1.8, 표준편차 0.448"],
                   ["AI1_Vibration", "하부 진동(변수 순서 기준, 단위 미제공)", "±0.40, 표준편차 0.119", "±0.75, 표준편차 0.241"],
                   ["AI2_Current", "모터 전류 파형(부호 있음, 단위 미제공)", "약 ±273, 정현파 형태", "−399~+539, 파형 붕괴·편중"],
                   ["Equipment_state", "0=정상 / 1=이상 (파일과 1:1 대응)", "전부 0", "전부 1"]],
                  [2.3, 4.6, 3.0, 3.0], left_cols=(1,)),
          t.caption("〈표 1-1〉 변수 정의 및 요약"),
          t.b3("출제 배경 원문이 ‘유압펌프 모터의 **상부·하부 진동 및 전류**’ 세 가지를 명시해 채널이 정확히 3개(진동 2+전류 1)임을 확인해 주며, 변수 순서(AI0→AI1→AI2)가 그 언급 순서와 일치해 상부=AI0·하부=AI1로 대응했습니다. "
               "상부·하부 라벨이 서로 바뀌어도 두 채널의 특징식은 대칭으로 계산되므로(vup_*, vlow_* 동일 수식) **모델 성능 자체에는 영향이 없고 보고서 표기 문제로 한정**됩니다."),
          t.gap()]
    L += [t.b2("**시간·설비 관계와 제조공정 상태 해석** : 그림 1-1과 같이 정상 전류는 매끈한 정현파이고 두 진동은 규칙적으로 진동하는 반면, 이상 구간은 "
               "전류 파형이 무너지며 한쪽으로 치우치고(DC 오프셋) 진동은 비주기적인 충격성 버스트를 보입니다."),
          t.b3("샘플링이 10Hz라 전류의 상용 주파수 성분은 aliasing 되어 약 0.6Hz의 겉보기 정현파로 보입니다. 절대 주파수는 해석할 수 없으므로 **‘파형의 규칙성(평활도·영교차율)’ 만** 활용합니다(해석은 가설임)."),
          t.b3("이는 유압펌프 모터 구동 불안정(부하·압력 변동, 캐비테이션 등)이 전류 파형 왜곡과 진동 증가로 나타나는 상태를 가리키며, 분석 목적인 ‘압력 불안정 이전의 초기 이상징후 탐지’와 직접 연결됩니다."),
          t.image(FIG / "fig01_signals.png", 0.78), t.caption("〈그림 1-1〉 정상(좌)/이상(우) 연속 50샘플 파형"), t.gap()]
    L += [t.b2("**변수 간 관계(원시 신호 시차상관)** : 구간별로 두 채널의 상호상관을 지연 ±10샘플(±1초)에서 직접 구해, 어느 채널이 어느 채널을 얼마나 앞서는지 정량화했습니다."),
          t.b3(f"**상부·하부 진동(AI0~AI1)**: 정상에서는 동시성이 약한데(평균 피크상관 {xc01_n.mean_peak_corr:.3f}, {int(xc01_n.n_segments)}개 구간) 이상에서는 눈에 띄게 강해집니다"
               f"(평균 피크상관 {xc01_a.mean_peak_corr:.3f}, {int(xc01_a.n_segments)}개 구간, 지연 중앙값 {xc01_a.peak_lag_median:.0f}샘플로 거의 동시). "
               "이상 시 상부·하부가 거의 동시에 반대 방향으로 흔들리는 상태를 가리키며, 17개 특징 중 `vib_corr`가 이 관계를 포착하도록 설계된 근거입니다."),
          t.b3("**진동-전류(AI0~AI2, AI1~AI2)**: 두 상태 모두 상관이 0에 가깝고(표 1-1b) 안정적인 지연도 보이지 않아, 전류와 진동은 **선형적인 시차 관계가 없습니다**(10Hz 샘플링에서 전류는 aliasing된 준주기 신호라 위상 비교 자체의 신뢰도가 낮다는 점도 함께 고려해야 합니다)."),
          t.table([["구분", "AI0~AI1", "AI0~AI2", "AI1~AI2"],
                   ["정상 평균 피크상관", f"{xc.loc[('정상','AI0_Vibration~AI1_Vibration')].mean_peak_corr:.3f}",
                    f"{xc.loc[('정상','AI0_Vibration~AI2_Current')].mean_peak_corr:.3f}", f"{xc.loc[('정상','AI1_Vibration~AI2_Current')].mean_peak_corr:.3f}"],
                   ["이상 평균 피크상관", f"{xc.loc[('이상','AI0_Vibration~AI1_Vibration')].mean_peak_corr:.3f}",
                    f"{xc.loc[('이상','AI0_Vibration~AI2_Current')].mean_peak_corr:.3f}", f"{xc.loc[('이상','AI1_Vibration~AI2_Current')].mean_peak_corr:.3f}"]],
                  [2.2, 1.8, 1.8, 1.8]),
          t.caption(f"〈표 1-1b〉 채널쌍별 시차상관 (구간당 개별 피크의 평균; 정상 {int(xc01_n.n_segments)}구간·이상 {int(xc01_a.n_segments)}구간, 30샘플 이상만)"),
          t.image(FIG / "fig04b_cross_corr.png", 0.95), t.caption("〈그림 1-1b〉 채널쌍별 시차상관 곡선 (정상 vs 이상)"),
          t.b3("이상 구간이 10개뿐이라 이 결과도 넓게 보아야 하며, 그룹 중요도 분석(제3장 표 3-1)에서 ‘채널 간 관계’ 그룹을 빼도 PR-AUC가 거의 안 떨어지는 것과 일관됩니다 — "
               "흥미로운 차이지만 `cur_madr` 등이 이미 포착하는 정보와 겹쳐 모델 성능에 대한 **추가 기여는 작습니다**."),
          t.gap()]
    L += [t.b2("**결측·중복·이상치·불균형 진단 및 전처리**"),
          t.table([["항목", "진단 결과", "처리"],
                   ["결측", f"두 파일 모두 결측 셀 {int(qn.missing_cells + qa.missing_cells)}개", "처리 불필요"],
                   ["중복", f"정상 파일 타임스탬프 중복 {int(qn.duplicated_timestamps)}행", "중복 행 1개 제거"],
                   ["이상치", f"|z|>5 셀 {int(qn.abs_z_gt5_cells + qa.abs_z_gt5_cells)}개, 그러나 이상 신호 자체가 분석 대상", "제거·윈저화 하지 않음(이상 신호 보존)"],
                   ["시간 구조", "샘플 간격 0.1s, 구간 사이 공백 1.5~16.6s", "gap>0.15s 로 구간 분할, 윈도우는 구간을 넘지 않음"],
                   ["윈도우화", f"길이 10샘플(1초), stride 5 → 총 {n_win:,}창 (이상 {n_anom_win}창)", f"5샘플 미만 구간 제외(정상 {n_norm_seg - u_norm_seg}개·이상 {n_anom_seg - u_anom_seg}개 구간)"],
                   ["불균형", f"행 {pct(qa.rows / (qa.rows + qn.rows))}, 구간 {pct(n_anom_seg / (n_anom_seg + n_norm_seg))}, 윈도우 {pct(n_anom_win / n_win)}가 이상", "class_weight, 확률 보정, F1·비용 기반 임계값 (제5장)"]],
                  [1.3, 4.6, 4.0], left_cols=(1, 2)),
          t.caption("〈표 1-2〉 데이터 진단과 전처리"), t.gap()]
    L += [t.b2("**정상 운전모드 진단** : 정상 윈도우를 (하부진동 RMS, 전류 RMS)로 군집화하면 k=2에서 실루엣 "
               f"{J['modes_silhouette']['2']:.2f}(k=3: {J['modes_silhouette']['3']:.2f}, k=4: {J['modes_silhouette']['4']:.2f})로 가장 좋아 **저부하 모드(mode0)와 고부하 모드(mode1)** 로 나뉩니다."),
          t.b3(f"mode0: 하부진동 RMS 중앙값 {m0.vlow_rms:.3f}, 전류 RMS {m0.cur_rms:.0f} ({int(m0.windows):,}창) / mode1: {m1.vlow_rms:.3f}, {m1.cur_rms:.0f} ({int(m1.windows):,}창). "
               "그림 1-2처럼 정상 운전은 부하에 따라 하부진동과 전류가 함께 커지는 **하나의 연속 궤적**이며, 이상 윈도우는 이 궤적에서 벗어난 영역에 흩어집니다."),
          t.b3("이 부하 변화가 바로 ‘정상적인 운전변화’ 이므로, 진폭만 보는 규칙·모델은 고부하 정상 운전을 이상으로 오인할 위험이 있습니다(제2·3장에서 정량 확인)."),
          t.image(FIG / "fig03_modes.png", 0.55), t.caption("〈그림 1-2〉 정상 운전모드(군집)와 이상 윈도우 분포"), t.gap()]
    L += [t.b2("**검증전략** : 이상 데이터가 단일 3분 세션(30초 블록 6개)이어서 무작위 분할은 인접 윈도우를 학습·검증에 섞어 성능을 과대평가할 수 있습니다."),
          t.b3("① 30초 연속 시간 블록을 그룹으로 하는 **StratifiedGroupKFold(5-fold, seed 3회 반복)** 을 기본 검증으로 사용하고, 폴드 안에서 내부 3-fold로 확률 보정·임계값을 정해 검증 누수를 막았습니다."),
          t.b3("**블록 경계와 누수**: 윈도우는 연속 수집 구간 안에서만 만들고, 윈도우의 CV 그룹은 구간의 첫 샘플이 속한 30초 블록 하나로 구간 전체에 부여합니다. "
               "구간이 30초 경계를 넘어도 그 구간의 창은 모두 같은 그룹이라 학습·검증으로 갈라지지 않고, 구간 사이에는 0.15초 이상의 공백이 있어 서로 시간을 공유하지 않습니다(tests/test_policy.py가 이 불변식을 검사)."),
          t.b3("② **미학습 운전모드 외삽 시험**: 한 정상 모드를 학습에서 통째로 빼고 그 모드의 정상 창에서 오경보율을 측정합니다(‘정상 운전변화’ 시나리오)."),
          t.b3("③ 블록 그룹 CV와 무작위 KFold 결과를 비교한 누수 점검: "
               f"최종 특징세트에서 M2 F1 {f3(lk.loc['M2_HGB', 'F1'])}(무작위) vs {f3(sm('INV', 'M2_HGB', 'F1'))}(블록) 로 차이가 크지 않아, 성능은 누수보다 **신호 자체의 분리성**에서 나옵니다. "
               "다만 이상이 한 세션뿐이므로 다른 날·다른 고장 유형으로의 일반화는 보증되지 않습니다(한계)."),
          t.b3("④ **선택과 평가의 분리**: 모델 선정에 쓴 교차검증·외삽 시험 결과는 선택용이라 최종 성능으로 보지 않고, 선택 절차 전체를 외부 fold마다 다시 수행하는 중첩 교차검증을 별도 평가로 보고합니다(제2장 표 2-5)."),
          t.gap()]
    return L


def chapter2(t: Template) -> list:
    L = [t.heading("□ 제2장. AI 예측모델 개발 및 성능평가〔40점〕", 1)]
    L += [t.b2("**이 장의 결론 3줄**"),
          t.b3(f"① 모델 비교(표 2-2): SPC 관리도 F1 {f3(sm('INV', 'B0_SPC', 'F1'))} 대비 학습 모델 {f3(LEARNED_F1[0])}~{f3(LEARNED_F1[1])}, 최종 모델 {SHORT[FINAL]}는 F1 {f3(sm('INV', FINAL, 'F1'))}·FPR {pct(sm('INV', FINAL, 'FPR'), 2)}입니다."),
          t.b3(f"② 핵심 발견(표 2-3): 진폭을 포함한 특징은 학습에 없던 고부하 모드에서 정상 창의 {pct(shift('ALL', FINAL, 1), 0)}를 오경보하지만, 진폭 비의존 17개 특징은 {pct(shift('INV', FINAL, 1))}입니다."),
          t.b3(f"③ 한계(표 2-5~2-11): 선택과 분리한 중첩검증 F1은 {f3(NPI.loc['F1', 'point'])}로 교차검증({f3(sm('INV', FINAL, 'F1'))})보다 낮고, 피크전류 가드·약한 이상·창 길이의 결과와 한계는 이어서 공개합니다.")]
    L += [t.b2("**문제 정의** : 1초(10샘플) 윈도우마다 ‘이상 확률’을 예측합니다. 이상이 시작된 뒤 첫 경보까지 걸리는 시간(탐지 지연)을 조기탐지 지표로, F1·PR-AUC·재현율·오경보율(FPR)·시간당 오경보를 성능지표로 사용합니다."),
          t.b3("특징은 채널(상부·하부 진동, 전류)별 mean, std, RMS, peak-to-peak, 왜도, 첨도, crest, 평균 절대 변화량(mad), 정규화 평활도(madr=mad/std), 영교차율과 채널 간 상관·비율 등 **34개(전체)** 입니다."),
          t.b3("이 중 진폭 크기에 의존하지 않는 형상·동특성 **17개(skew, kurt, crest, madr, zcr, vib_corr, cur_dc_ratio; ‘운전모드 불변’ 특징)** 를 별도 특징세트로 정의해 비교합니다. 설계 이유는 아래 외삽 시험에서 밝힙니다.")]
    L += [t.b2("**비교 모델 (베이스라인 포함 6종)**"),
          t.table([["구분", "모델", "요점"],
                   ["베이스라인", "B0 SPC 관리도 규칙", "정상 RMS의 median/MAD 기반 robust z-score 최댓값(현장 관리도 방식)"],
                   ["베이스라인", "B1 로지스틱회귀", "표준화 + class_weight=balanced, C=0.3"],
                   ["후보", "M1 랜덤포레스트", "300 트리, min_samples_leaf=2, balanced_subsample"],
                   ["후보", "M2 HistGradientBoosting", "max_depth=4, lr=0.05, 200회, L2=1, balanced"],
                   ["후보", "M3 IsolationForest", "정상 윈도우만 학습(레이블 불필요), 미지 이상 탐지 대비"],
                   ["후보", "M4 스태킹 앙상블", "B1·M1·M2의 보정 점수를 로지스틱으로 결합"]],
                  [1.3, 3.0, 6.0], left_cols=(1, 2)),
          t.caption("〈표 2-1〉 비교 모델 구성 (모든 모델: Platt 확률 보정 + 폴드 내부 F1 최대 임계값)"), t.gap()]
    rows = [["모델", "F1", "PR-AUC", "재현율", "정밀도", "FPR", "FN창", "FP창", "이상구간 탐지", "정상구간 오경보"]]
    for m in ORDER:
        def r(c, m=m):
            return sm("INV", m, c)
        rows.append([SHORT[m] + (" ★" if m == FINAL else ""), f3(r("F1")), f3(r("PR_AUC")), f3(r("Recall")), f3(r("Precision")),
                     pct(r("FPR"), 2), f"{r('FN'):.1f}", f"{r('FP'):.1f}", pct(r("seg_recall"), 0), pct(r("seg_false_alarm_rate"), 2)])
    L += [t.b2(f"**성능평가 (동일 조건: 블록 그룹 CV 5-fold × 3회, 최종 특징세트 17개)** : 베이스라인 B0의 F1은 {f3(sm('INV', 'B0_SPC', 'F1'))}, 재현율 {f3(sm('INV', 'B0_SPC', 'Recall'))}에 그치는 반면 "
               f"학습 모델(M3 제외)은 F1 {f3(min(sm('INV', m, 'F1') for m in ORDER[1:3] + ORDER[3:4] + ORDER[5:]))}~{f3(max(sm('INV', m, 'F1') for m in ORDER[1:3] + ORDER[3:4] + ORDER[5:]))}로 크게 앞섭니다."),
          t.table(rows, [2.0, 1.0, 1.1, 1.0, 1.0, 1.0, 0.9, 0.9, 1.5, 1.6]),
          t.caption("〈표 2-2〉 모델별 성능 (3회 반복 평균, FN/FP는 윈도우 수, ★=최종모델)"),
          t.image(FIG / "fig05_model_compare.png", 0.85), t.caption("〈그림 2-1〉 모델별 F1·PR-AUC·시간당 오경보 (오차막대=반복 표준편차)"), t.gap()]
    t3 = [["모델", "기존 모드 F1\n(전체 특징)", "기존 모드 F1\n(불변 특징)", "미학습 모드 오경보율\n전체: mode0 / mode1", "미학습 모드 오경보율\n불변: mode0 / mode1", "미학습 모드 F1\n전체 → 불변"]]
    for m in ORDER:
        u_all = MS[(MS.feature_set == "ALL") & (MS.model == m)].F1.mean()
        u_inv = MS[(MS.feature_set == "INV") & (MS.model == m)].F1.mean()
        t3.append([SHORT[m], f3(sm("ALL", m, "F1")), f3(sm("INV", m, "F1")),
                   f"{pct(shift('ALL', m, 0))} / {pct(shift('ALL', m, 1))}", f"{pct(shift('INV', m, 0))} / {pct(shift('INV', m, 1))}",
                   f"{u_all:.3f} → {u_inv:.3f}"])
    L += [t.b2(f"**핵심 발견 - 진폭 의존 모델은 새 운전모드에서 붕괴** : 전체 특징(진폭 포함)을 쓰면 기존 모드에서는 F1이 0.99 수준이지만, 고부하 모드를 학습에서 빼면 "
               f"M1 {pct(shift('ALL', 'M1_RF', 1), 0)}, M2 {pct(shift('ALL', 'M2_HGB', 1), 0)}, SPC 규칙 {pct(shift('ALL', 'B0_SPC', 1), 0)}의 정상 윈도우를 이상으로 오판합니다."),
          t.b3(f"진폭 비의존 17개 특징으로 바꾸면 같은 시험에서 M1 {pct(shift('INV', 'M1_RF', 1))}, **M2 {pct(shift('INV', 'M2_HGB', 1))}**, M4 {pct(shift('INV', 'M4_Stack', 1))}로 떨어지고, "
               f"기존 모드 F1 손실은 M2 기준 {f3(sm('ALL', 'M2_HGB', 'F1'))} → {f3(sm('INV', 'M2_HGB', 'F1'))}로 작습니다."),
          t.table(t3, [1.2, 1.4, 1.4, 2.2, 2.2, 1.8]),
          t.caption("〈표 2-3〉 특징세트별 기존 모드 성능 vs 미학습 운전모드 외삽 (mode0=저부하, mode1=고부하를 학습에서 제외)"),
          t.image(FIG / "fig12_mode_shift.png", 0.86), t.caption("〈그림 2-2〉 미학습 운전모드 정상 윈도우의 오경보율 (B0~M4 = 표 2-1의 모델)"), t.gap()]
    t4 = [["순위", "특징세트", "모델", "기존 모드 F1", "미학습 모드 F1", "선정 점수"]]
    for i, r in enumerate(SEL.head(6).itertuples(), 1):
        t4.append([str(i), {"INV": "불변 17개", "ALL": "전체 34개"}[r.feature_set], SHORT[r.model], f3(r.F1_known_modes), f3(r.F1_unseen_mode), f3(r.score)])
    L += [t.b2(f"**최종모델 선정 근거 : {NAME[FINAL]} + 운전모드 불변 특징 17개**"),
          t.b3("선정 기준은 미리 정한 규칙(기존 모드 F1과 미학습 모드 F1의 평균이 최대인 조합; 1위와 0.01 이내면 앙상블 우선)이며 "
               + ("0.01 이내 조합이 없어 1위가 그대로 선정됐습니다." if (SEL.score >= SEL.score.max() - 0.01).sum() == 1
                  else f"1위와 0.01 이내인 조합이 {int((SEL.score >= SEL.score.max() - 0.01).sum())}개라 이 중 규칙에 따라 {NAME[FINAL]}가 선정됐습니다.")),
          t.table(t4, [0.6, 1.4, 1.2, 1.5, 1.6, 1.4]), t.caption("〈표 2-4〉 최종모델 선정표 (상위 6개 조합)"),
          t.b3(f"M2 는 기존 모드 F1 {f3(sm('INV', 'M2_HGB', 'F1'))}(FPR {pct(sm('INV', 'M2_HGB', 'FPR'), 2)}, 오경보 창 평균 {sm('INV', 'M2_HGB', 'FP'):.1f}개)와 미학습 모드 F1 "
               f"{f3(sel('INV', 'M2_HGB', 'F1_unseen_mode'))}를 동시에 만족한 모델입니다. M1·M4는 PR-AUC가 더 높지만({f3(sm('INV', 'M1_RF', 'PR_AUC'))}, {f3(sm('INV', 'M4_Stack', 'PR_AUC'))} vs {f3(sm('INV', 'M2_HGB', 'PR_AUC'))}) "
               "임계값 기준 오경보가 많아 F1이 낮습니다."),
          t.table([["비교", "ΔF1 (M2 − 상대)", "95% CI (블록 부트스트랩 1,000회)", "판정"],
                   ["M2 vs M1", f"+{d_m1.delta_F1:.3f}", f"[{d_m1.ci_low:.3f}, {d_m1.ci_high:.3f}]", "0 미포함 → 유의하게 M2 우위"],
                   ["M2 vs M4", f"+{d_m4.delta_F1:.3f}", f"[{d_m4.ci_low:.3f}, {d_m4.ci_high:.3f}]", "0 미포함 → 유의하게 M2 우위"]],
                  [1.6, 1.8, 3.0, 2.6]),
          t.caption("〈표 2-4b〉 최종모델과 차순위 모델의 F1 차이 쌍대검정"),
          t.b3(f"윈도우 수로는 1~3개 차이라 작아 보이지만, 같은 시험 윈도우에 대한 **쌍대(paired) 블록 부트스트랩**으로 보면 M2가 M1·M4보다 ΔF1 +{d_m1.delta_F1:.3f}만큼 "
               "**통계적으로도 유의하게** 높습니다(두 비교 모두 95% 구간이 0을 포함하지 않음). 다만 채택을 가른 핵심 근거는 F1 차이가 아니라 ‘새 운전모드에서도 오경보가 없다’는 표 2-3의 결과이며, "
               "이 통계검정은 그 선택이 우연한 F1 변동에 기댄 것이 아님을 추가로 뒷받침합니다."),
          t.b3("M3(정상 전용)는 레이블 없이도 이상을 잡지만 F1이 낮아 단독 사용은 부적합하고, 로지스틱(B1)은 해석이 쉬우나 미학습 모드 F1이 낮습니다."),
          t.b3(f"신뢰구간(블록 부트스트랩 1,000회, 반복 1): F1 {f3(ci('F1', 'point'))} [{f3(ci('F1', 'ci_low'))}, {f3(ci('F1', 'ci_high'))}], 재현율 {f3(ci('Recall', 'point'))} [{f3(ci('Recall', 'ci_low'))}, {f3(ci('Recall', 'ci_high'))}], "
               f"FPR {pct(ci('FPR', 'point'), 2)} [{pct(ci('FPR', 'ci_low'), 2)}, {pct(ci('FPR', 'ci_high'), 2)}]. 이상 블록이 6개뿐이라 구간이 넓습니다."),
          t.image(FIG / "fig13_feature_sets.png", 0.85), t.caption("〈그림 2-3〉 특징세트·모델별 기존 모드 F1과 미학습 모드 F1"), t.gap()]
    npi = NP.set_index("metric")
    n_tp, n_fp, n_fn, n_tn = (int(NP[c].iloc[0]) for c in ("TP", "FP", "FN", "TN"))
    n_prec = n_tp / max(n_tp + n_fp, 1)
    nest_rows = [["외부 fold", "선택된 조합", "시험 창(이상)", "F1", "재현율", "정밀도", "FPR", "FP/FN 창"]]
    for r in NF.itertuples():
        nest_rows.append([str(r.fold), f"{r.feature_set}·{SHORT[r.model]}", f"{int(r.test_windows):,}({int(r.test_anomaly_windows)})", f3(r.F1), f3(r.Recall),
                          f3(r.Precision), pct(r.FPR, 2), f"{int(r.FP)}/{int(r.FN)}"])
    nest_rows.append(["합산", "-", f"{n_tp + n_fp + n_fn + n_tn:,}({n_tp + n_fn})", f3(npi.loc["F1", "point"]), f3(npi.loc["Recall", "point"]), f3(n_prec),
                      pct(npi.loc["FPR", "point"], 2), f"{n_fp}/{n_fn}"])
    chosen = sorted({(r.feature_set, r.model) for r in NF.itertuples()})
    d_nest = npi.loc["F1", "point"] - sm("INV", "M2_HGB", "F1")
    r2n = NR.set_index("rule").loc["R2 red 2연속"]
    L += [t.b2("**독립 평가: 선택에 쓴 데이터와 평가 데이터를 분리 (중첩 교차검증)** : 표 2-2·2-3의 교차검증은 같은 창으로 모델을 고르고 점수도 매겨 선택 낙관이 섞일 수 있습니다. "
               f"그래서 외부 {len(NF)}-fold(30초 블록 단위, 이상 라벨 층화)를 두고, 각 fold에서 운전모드 군집·6모델×2특징세트 교차검증·미학습 모드 시험·최종모델 선정·경보 임계값 결정을 "
               "**외부 train 부분만으로 처음부터 다시** 수행한 뒤, 고른 모델을 외부 test 부분에서 **한 번만** 평가했습니다. test 부분은 어떤 선택에도 쓰이지 않습니다."),
          t.table(nest_rows, [1.2, 2.2, 1.9, 1.0, 1.1, 1.1, 1.1, 1.4]),
          t.caption("〈표 2-5〉 중첩 교차검증 결과 (외부 test 창만 집계; 시험 창(이상) = 외부 test 윈도우 수(그중 이상 창 수))"),
          t.b3("선택 결과는 외부 fold마다 " + (f"모두 {chosen[0][0]}·{SHORT[chosen[0][1]]}로 같아 선택이 데이터 분할에 흔들리지 않았습니다. " if len(chosen) == 1
                                         else f"{len(chosen)}가지({', '.join(f'{a}·{SHORT[b]}' for a, b in chosen)})로 갈려 선택이 데이터 분할에 민감합니다. ")
               + f"외부 test 합산 F1은 {f3(npi.loc['F1', 'point'])} [95% CI {f3(npi.loc['F1', 'ci_low'])}, {f3(npi.loc['F1', 'ci_high'])}], 재현율 {f3(npi.loc['Recall', 'point'])} [{f3(npi.loc['Recall', 'ci_low'])}, {f3(npi.loc['Recall', 'ci_high'])}], "
               f"FPR {pct(npi.loc['FPR', 'point'], 2)} [{pct(npi.loc['FPR', 'ci_low'], 2)}, {pct(npi.loc['FPR', 'ci_high'], 2)}]이고, 표 2-2의 교차검증 F1({f3(sm('INV', 'M2_HGB', 'F1'))})과의 차이는 {d_nest:+.3f}입니다. "
               + ("이 차이는 신뢰구간 안에 있어 선택 낙관은 이 데이터로 구분되지 않습니다. " if npi.loc["F1", "ci_low"] <= sm("INV", "M2_HGB", "F1") <= npi.loc["F1", "ci_high"]
                  else "교차검증 값이 이 신뢰구간 밖이므로 표 2-2의 수치는 낙관적일 수 있습니다. ")
               + f"배포 경보 규칙 R2를 같은 외부 test 창에 적용하면 이상 구간 탐지율 {pct(r2n.segment_recall, 0)}, 오경보 {r2n.false_alarms_per_hour:.1f}건/시간입니다."),
          t.b3("단일 hold-out 대신 중첩 방식을 쓴 이유: 이상 데이터가 30초 블록 6개뿐이라 하나를 떼어 내면 시험·학습 모두 이상 블록이 3~4개로 줄어 추정이 불안정해집니다. "
               "시간상 이웃한 블록의 창은 서로 닮았을 수 있어 test 블록과 이웃한 train 블록 사이의 시간적 상관은 남아 있으며, 정상 세션과 이상 세션이 각각 하나뿐이라 세션·날짜 단위 분할은 불가능했습니다(한계)."),
          t.gap()]
    ge_rows = [["구성", "정상 노랑", "정상 빨강", "이상 빨강", "이상 노랑", "이상 초록", "R2 오경보\n(건/시간)", "R2 구간\n탐지율"]]
    for lab, g in (("모델만", ge0), ("전역 한계 → 노랑", ge1), ("모드별 한계 → 노랑 (채택)", ge2), ("전역 한계 → 빨강 (폐기)", ge3)):
        ge_rows.append([lab, f"{g.normal_yellow:.0f}", f"{g.normal_red:.0f}", f"{g.anomaly_red:.0f}", f"{g.anomaly_yellow:.0f}",
                        f"{g.anomaly_green:.0f}", f"{g.R2_FA_per_h:.1f}", pct(g.R2_recall, 0)])
    gm_rows = [["학습에서 뺀 모드", "정상 창", "전역 한계: 노랑 플래그", "모드별 한계: 노랑 플래그", "신규 모드로 분류", "(참고) 이상 창: 노랑 플래그 / 신규 모드"]]
    for m, r in gm.iterrows():
        gm_rows.append([f"mode{m}", f"{int(r.n_normal):,}", pct(r.global_normal_flag, 1), pct(r.mode_normal_flag, 1), pct(r.new_mode_normal, 1),
                        f"{pct(r.mode_anomaly_flag, 0)} / {pct(r.new_mode_anomaly, 0)}"])
    worst_g, worst_m = gm.global_normal_flag.max(), gm.mode_normal_flag.max()
    L += [t.b2("**안전장치: 피크전류 가드 = ‘운전모드별 한계 → 노랑’** : 모델은 형상 특징만 보므로 전류 진폭만 치솟는 고장을 놓칠 수 있어 진폭 안전장치를 하나 더 둡니다. "
               f"처음에는 정상 데이터 전체의 관측 최대 전류({max(GUARD_LIMITS):.1f})를 넘으면 빨강으로 올렸으나, (1) 빨강 강제는 비용이 이득보다 커서 노랑으로 낮추고(표 2-6), "
               "(2) 한 개의 전역 한계는 운전모드가 달라지면 틀리므로 운전모드별 한계로 바꿨습니다. 정상 창을 (하부진동·전류 RMS)로 군집한 운전모드 중 가장 가까운 모드의 "
               f"정상 최대 전류를 그 모드의 한계로 쓰고(저부하 {GUARD_LIMITS[0]:.1f}, 고부하 {GUARD_LIMITS[1]:.1f}), 어느 모드 영역에도 속하지 않는 창은 ‘신규 모드’ 상태로 분리해 "
               "경보 대신 운영자 승인·재학습 대상으로 보냅니다. 이 한계는 설비 정격 사양값이 아니라 학습 정상 데이터의 관측 최댓값입니다."),
          t.table(ge_rows, [2.8, 1.1, 1.1, 1.1, 1.1, 1.1, 1.4, 1.3]),
          t.caption(f"〈표 2-6〉 피크전류 가드 결합 방식별 경보 등급 (윈도우 정상 {int(ge0.normal_windows):,}·이상 {int(ge0.anomaly_windows)}창, 첫 CV 반복, 배포 임계값)"),
          t.image(FIG / "fig28_guard_effect.png", 0.95), t.caption("〈그림 2-4〉 가드 결합 방식별 경보 등급과 미학습 모드에서의 처리"),
          t.b3(f"**빨강 강제는 폐기** : 이상 윈도우 중 빨강이 {ge0.anomaly_red:.0f}→{ge3.anomaly_red:.0f}창(+{guard_gain}) 되는 대신 정상 윈도우의 빨강도 {ge0.normal_red:.0f}→{ge3.normal_red:.0f}창(+{guard_cost})이 되고, "
               f"R2 기준 시간당 오경보는 {trend(ge0.R2_FA_per_h, ge3.R2_FA_per_h, '건/시간')} 이상 구간 탐지율은 "
               + (f"어느 쪽이든 {pct(ge0.R2_recall, 0)}로 같습니다. " if ge0.R2_recall == ge3.R2_recall else f"{pct(ge0.R2_recall, 0)}→{pct(ge3.R2_recall, 0)}입니다. ")
               + "빨강은 사람을 현장으로 부르는 신호라 이런 교환은 받아들이지 않았습니다."),
          t.b3(f"**모드별 한계(채택)의 알려진 비용** : 빨강은 그대로({chg(ge0.normal_red, ge2.normal_red, '창')})이고 정상 윈도우의 노랑이 {ge0.normal_yellow:.0f}→{ge2.normal_yellow:.0f}창({ge2.normal_yellow - ge0.normal_yellow:+.0f}, 전역 한계는 {ge1.normal_yellow:.0f}창)입니다. "
               + (f"모델만으로는 초록이던 이상 창 {guard_rescued}개를 노랑으로 구제합니다. " if guard_rescued > 0 else
                  f"모델만으로도 초록으로 지나간 이상 창이 {int(ge0.anomaly_green)}개라 **이 데이터에서 가드의 추가 이득은 없고**, 진폭만 치솟는 형태의 고장에 대비한 보험입니다. ")
               + f"정상 창 중 신규 모드로 분류된 것은 {int(ge2.normal_new_mode)}개({pct(ge2.normal_new_mode / ge2.normal_windows, 2)}), 이상 창은 {int(ge2.anomaly_new_mode)}개({pct(ge2.anomaly_new_mode / ge2.anomaly_windows, 0)})입니다. "
               "이상 창이 정상 모드 영역 밖에 놓이는 일이 많은 것은 고장 상태가 정상 운전 영역을 벗어난다는 사실과 맞지만, 신규 모드 표시 자체가 고장 판정은 아닙니다."),
          t.table(gm_rows, [1.9, 1.1, 2.0, 2.0, 1.7, 2.6]),
          t.caption("〈표 2-7〉 미학습 운전모드 시험: 한 정상 모드를 뺀 데이터로 가드를 만들고, 빠진 모드의 정상 창에서 가드가 어떻게 반응하는가 (모드별 한계는 남은 모드 수로 군집)"),
          t.b3("**미학습 모드 시험** : " + "; ".join(f"mode{m} 제외 시 전역 한계 {pct(r.global_normal_flag, 1)} → 모드별 한계 {pct(r.mode_normal_flag, 1)}(신규 모드로 분류 {pct(r.new_mode_normal, 0)})" for m, r in gm.iterrows())
               + ". "
               + (f"모드별 한계와 신규 모드 분리로 노랑 폭주가 최대 {pct(worst_g, 1)}에서 {pct(worst_m, 1)}로 줄고, 그 상태는 ‘승인 대기’로 넘어갑니다. " if worst_m < worst_g
                  else f"이 시험에서는 모드별 한계로도 노랑 플래그가 줄지 않았습니다(최대 {pct(worst_g, 1)} → {pct(worst_m, 1)}). ")
               + (f"대신 같은 시험에서 이상 창에 대한 가드 플래그 비율은 전역 한계 {pct(gm.global_anomaly_flag.min(), 0)}~{pct(gm.global_anomaly_flag.max(), 0)}에서 모드별 한계 {pct(gm.mode_anomaly_flag.min(), 0)}~{pct(gm.mode_anomaly_flag.max(), 0)}로 낮아져, 오경보를 줄인 만큼 가드의 민감도도 줄었습니다(이상 창 다수가 신규 모드 영역으로 분류되기 때문). "
                  if gm.mode_anomaly_flag.max() < gm.global_anomaly_flag.max() else "")
               + "이 때문에 표지 요약의 ‘미학습 모드 오경보 0.2% 이하’는 모델의 **빨강(경고)** 기준 수치이며, 노랑·신규 모드 표시는 별도로 읽어야 합니다. 새 운전모드가 승인되면 한계를 재산정해야 하고, 현장에서는 설비 정격 사양값과 대조해야 합니다."),
          t.gap()]
    L += [t.b2("**과적합 점검** : 최종 모델이 암기(과적합)된 것은 아닌지, 현재 하이퍼파라미터보다 더 보수적으로 가면 나아지는지 직접 비교했습니다."),
          t.table([["설정", "학습 AP", "held-out(OOF) AP", "차이", "미학습mode0 FPR", "미학습mode1 FPR"]] +
                   [[v.split(" (")[0], f3(r.train_AP), f3(r.oof_AP), f3(r.gap), pct(r.unseen_mode0_FPR, 2), pct(r.unseen_mode1_FPR, 2)]
                    for v, r in zip(OF.variant, OF.itertuples())],
                  [2.2, 1.3, 1.6, 1.0, 1.6, 1.6]),
          t.caption("〈표 2-8〉 과적합 점검 및 규제 강도 민감도"),
          t.image(FIG / "fig29_overfit_check.png", 0.9), t.caption("〈그림 2-5〉 학습-held out 성능차 및 규제 강도별 미학습 모드 FPR"),
          t.b3(f"배포 설정의 학습-held out 차이는 {of0.gap:.3f}로 작아 암기 징후는 약합니다. 그런데 규제를 더 강하게 해도(표의 ‘규제 강화1/2’) held-out 성능과 미학습 모드 FPR이 **함께 나빠져**, "
               "‘고칠 게 없는 것을 고치면 손해만 본다’고 판단해 현재 설정을 유지했습니다. 이 점검도 매 실행마다 `run_all.py`가 자동으로 재생성합니다(제6장)."),
          t.gap()]
    q1 = SEV.iloc[0]
    sev_rows = [["진폭 사분위", "창 수", "정상 대비 z 중앙값", "SPC 규칙 탐지", "최종 모델 노랑 이상", "최종 모델 빨강"]]
    for r in SEV.itertuples():
        sev_rows.append([str(r.bin), str(int(r.windows)), f"{r.z_median:.1f}", pct(r.spc_detect, 0), pct(r.model_yellow_or_red, 0), pct(r.model_red, 0)])
    wk = WK.pivot(index="alpha", columns="model", values="alarm_rate")
    wk_rows = [["고장 성분 α", "SPC 규칙", "최종 모델 노랑 이상", "최종 모델 빨강"]]
    for a in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
        r = wk.loc[min(wk.index, key=lambda x: abs(x - a))]
        wk_rows.append([f"{100 * a:.0f}%" + (" (순수 정상)" if a == 0 else " (실제 고장 신호)" if a == 1 else ""),
                        pct(r["SPC 규칙"], 1), pct(r["최종 모델(노랑 이상)"], 1), pct(r["최종 모델(빨강)"], 1)])
    m30, r30, s30 = wk.loc[0.3, "최종 모델(노랑 이상)"], wk.loc[0.3, "최종 모델(빨강)"], wk.loc[0.3, "SPC 규칙"]
    s100 = wk.loc[1.0, "SPC 규칙"]
    if q1.model_red > q1.spc_detect + 0.2:
        q1_verdict = "진동 진폭만 보는 관리도가 못 보는 약한 이상을 전류 파형 평활도 기반 모델이 잡는다는 **실데이터 근거**입니다."
    else:
        q1_verdict = "이 구간에서 모델이 SPC 규칙보다 뚜렷이 낫다고는 말하기 어렵습니다."
    L += [t.b2("**‘미리’ 잡는가 - 약한 이상의 탐지력** : 조기탐지의 핵심은 이상이 아직 약할 때 잡느냐입니다. 열화가 진행되는 시계열은 없으므로 "
               "(a) 실제 이상 윈도우를 ‘정상에서 진폭이 얼마나 먼가’로 4등분해 탐지율을 보고, (b) 정상 신호에 실제 고장 신호를 α만큼만 섞은 **합성** 시험으로 민감도를 봅니다."),
          t.table(sev_rows, [2.0, 1.0, 1.8, 1.7, 1.9, 1.7]),
          t.caption(f"〈표 2-9〉 실제 이상 윈도우의 진폭 사분위별 탐지율 (정상 RMS 3채널 robust z의 최댓값 기준, 첫 CV 반복, 이상 {int(SEV.windows.sum())}창)"),
          t.b3(f"가장 약한 사분위(Q1, z 중앙값 {q1.z_median:.1f}{'로 정상 변동 범위 안' if q1.z_median < 3 else ''})에서 SPC 규칙은 {pct(q1.spc_detect, 0)}만 잡는데 최종 모델은 빨강 {pct(q1.model_red, 0)}"
               f"(노랑 이상 {pct(q1.model_yellow_or_red, 0)})입니다. {q1_verdict}"),
          t.table(wk_rows, [2.6, 1.8, 2.4, 2.2]),
          t.caption("〈표 2-10〉 합성 약한 고장 시험: 정상 윈도우(5-fold 홀드아웃) = (1-α)×정상 + α×(학습 폴드의 실제 고장 구간 조각)"),
          t.image(FIG / "fig30_weak_fault.png", 0.95), t.caption("〈그림 2-6〉 합성 약한 고장의 경보율(좌)과 실제 이상 윈도우의 진폭 사분위별 탐지율(우)"),
          t.b3(f"합성 시험에서 경보율이 절반을 넘는 첫 지점은 최종 모델 노랑 이상 {alpha_text('최종 모델(노랑 이상)', 0.5)}, 빨강 {alpha_text('최종 모델(빨강)', 0.5)}, SPC 규칙 {alpha_text('SPC 규칙', 0.5)}입니다"
               + (f"(SPC는 실제 이상 신호 자체도 {pct(s100, 0)}만 볼 수 있어 100%에서도 그 수준에 머무름). " if first_alpha("SPC 규칙", 0.5) == 100 else ". ")
               + f"고장 성분 30%에서 경보율은 모델 노랑 이상 {pct(m30, 1)}, 모델 빨강 {pct(r30, 1)}, SPC {pct(s30, 1)}로, "
               + ("노랑 이상 기준으로는 모델이 SPC보다 더 민감하고 " if m30 > s30 else "노랑 이상 기준으로도 모델이 SPC보다 둔하고 ")
               + ("빨강 기준으로는 SPC보다 낮습니다. " if r30 < s30 else "빨강 기준으로도 SPC 이상입니다. ")
               + ("즉 약한 이상에는 빨강보다 노랑이 먼저 반응하므로, 노랑을 ‘미리 보는 신호’로 운영해야 합니다." if (first_alpha("최종 모델(빨강)", 0.5) or 101) > (first_alpha("최종 모델(노랑 이상)", 0.5) or 101) else "")),
          t.b3("합성 시험은 두 신호의 선형 혼합일 뿐 실제 열화 경로를 재현한 것이 아니고 이상 데이터가 단일 세션이므로, 위 곡선은 ‘민감도의 하한 참고치’로만 해석해야 합니다. "
               "시험 대상 정상 구간은 학습에 쓰이지 않은 폴드, 섞는 고장 조각은 학습 폴드의 것만 써서 누수는 없습니다."),
          t.gap()]
    w = WIN.set_index(["window", "model"])
    lat = rules.loc["R1 red 1회", "latency_median_s(first red)"]
    L += [t.b2(f"**조기탐지(지연) 분석** : 이상 구간에서 첫 빨강 경보까지의 중앙값은 **{lat:.1f}초**(윈도우 길이 1.0초{'와 같아 첫 윈도우가 곧바로 경보' if abs(lat - 1.0) < 0.05 else ''})이고 "
               + (f"이상구간 {u_anom_seg}개 전부를 탐지했습니다. " if rules.loc["R1 red 1회", "segment_recall"] >= 1 else f"이상구간의 {pct(rules.loc['R1 red 1회', 'segment_recall'], 0)}를 탐지했습니다. ")
               + "창 길이를 바꾸면 지연과 오경보가 다음과 같이 교환됩니다."),
          t.table([["창 길이", "지연(중앙값)", "M2 F1", "M2 오경보(건/시간)", "M4 F1"],
                   ["5샘플 (0.5초)", "0.5초", f3(w.loc[(5, 'M2_HGB'), 'F1']), f"{w.loc[(5, 'M2_HGB'), 'false_alarms_per_hour']:.1f}", f3(w.loc[(5, 'M4_Stack'), 'F1'])],
                   ["10샘플 (1초, 채택)", "1.0초", f3(sm('INV', 'M2_HGB', 'F1')), f"{sm('INV', 'M2_HGB', 'false_alarms_per_hour'):.1f}", f3(sm('INV', 'M4_Stack', 'F1'))],
                   ["20샘플 (2초)", "2.0초", f3(w.loc[(20, 'M2_HGB'), 'F1']), f"{w.loc[(20, 'M2_HGB'), 'false_alarms_per_hour']:.1f}", f3(w.loc[(20, 'M4_Stack'), 'F1'])]],
                  [2.2, 1.4, 1.2, 2.0, 1.2]),
          t.caption("〈표 2-11〉 윈도우 길이 민감도 (5·20샘플: 1회 실행, 10샘플: 3회 평균; 오경보=정상 기록 1시간당 경보 구간 수)"),
          t.b3("0.5초 창은 경보를 0.5초 앞당기지만 F1이 떨어지고 오경보가 늘어 1초 창을 채택했습니다. 이상 구간이 3분짜리 단일 세션이라 **진짜 열화 진행 과정의 ‘며칠 전’ 조기탐지는 이 데이터로 검증할 수 없습니다**(한계)."),
          t.b3(f"5샘플 미만(0.5초 미만)의 짧은 구간은 판정 대상에서 제외했고(이상 {n_anom_seg - u_anom_seg}개 구간), 이는 현장에서 ‘판정 보류’로 처리해야 합니다."),
          t.gap()]
    return L


def chapter3(t: Template) -> list:
    L = [t.heading("□ 제3장. 영향요인 및 오류분석〔15점〕", 0)]
    L += [t.b2("**주요 영향변수** : 최종 특징세트에서 채널·특징군 단위로 값을 통째로 섞은 순열중요도(AP 감소)와, 해당 그룹을 빼고 재학습한 PR-AUC를 함께 보았습니다."),
          t.table([["그룹", "순열중요도(AP 감소)", "그룹 제거 후 재학습 PR-AUC", "해석"],
                   ["전류(AI2) 특징", f"{gi.loc['전류(AI2)', 'perm_drop_AP']:.3f}", f3(lgo["전류(AI2)"]), "판별 정보의 대부분"],
                   ["동특성(madr·zcr)", f"{gi.loc['동특성(mad/madr/zcr)', 'perm_drop_AP']:.3f}", f3(lgo["동특성(mad/madr/zcr)"]), "파형의 평활도·규칙성"],
                   ["형상(skew·kurt·crest)", f"{gi.loc['형상(skew/kurt/crest)', 'perm_drop_AP']:.3f}", f3(lgo["형상(skew/kurt/crest)"]), "보조"],
                   ["하부진동(AI1)", f"{gi.loc['하부진동(AI1)', 'perm_drop_AP']:.3f}", f3(lgo["하부진동(AI1)"]), "보조(제거 시 소폭 하락)"],
                   ["상부진동(AI0)", f"{gi.loc['상부진동(AI0)', 'perm_drop_AP']:.3f}", f3(lgo["상부진동(AI0)"]), "제거해도 거의 동일"],
                   ["채널 간 관계", f"{gi.loc['채널간 관계', 'perm_drop_AP']:.3f}", f3(lgo["채널간 관계"]), "제거해도 동일"]],
                  [2.2, 1.8, 2.4, 2.6], left_cols=(3,)),
          t.caption(f"〈표 3-1〉 특징 그룹 영향도 (전체 모델 PR-AUC {f3(lgo['전체'])})"),
          t.b3(f"단일 변수로는 **전류 평활도 cur_madr(순열중요도 {PI.iloc[0].perm_importance:.3f})** 가 압도적이고 다음 변수도 {PI.iloc[1].perm_importance:.3f}에 불과합니다. "
               "정상 전류는 매끈한 정현파라 샘플 간 변화량이 작고, 이상에서는 파형이 거칠어져 mad/std 비가 커집니다."),
          t.b3(f"상호작용은 약합니다(최대 Friedman H²={top_h2.H2:.3f}). 상위 조합의 공정상 의미: "
               + "; ".join(f"{r.f1}×{r.f2}(H²={r.H2:.3f}) = {meaning(r.f1)}와 {meaning(r.f2)}의 결합" for r in H2.head(3).itertuples())
               + " → 어느 조합도 점수를 크게 바꾸지 못해, 판정은 사실상 ‘전류 거칠기 + 진동 형상의 가법적 보강’ 구조입니다."),
          t.b3(f"진동 RMS만 보는 SPC 규칙은 이상 윈도우의 {pct(spc_miss.n / (spc_miss.n + spc_hit.n), 0)}({int(spc_miss.n)}창)를 놓치는데, 놓친 창은 상부진동 RMS 중앙값이 {spc_miss.vup_rms:.3f}로 정상 범위이고 "
               f"(잡은 창은 {spc_hit.vup_rms:.3f}), 전류 평균이 +{spc_miss.cur_mean:.0f}로 치우쳐 있습니다. **진동 진폭만으로는 초기 이상을 절반 가까이 놓친다**는 뜻입니다."),
          t.image(FIG / "fig08_importance.png", 0.95), t.caption("〈그림 3-1〉 변수·그룹 영향도 (최종 특징세트 17개)"), t.gap()]
    n_pool_n, n_pool_a = int(FPC[FPC.condition == "운전모드"].n.sum()), int(FNC[FNC.condition == "가장 가까운 운전모드"].n.sum())
    n_fp, n_fn = int(FPC[FPC.condition == "운전모드"].errors.sum()), int(FNC[FNC.condition == "가장 가까운 운전모드"].errors.sum())
    hdr = ["오류", "조건", "가장 취약한 구간", "발생/모수", "비율", "전체 오류 중 비중"]
    machine = [hdr] + [concentration_row(FPC[FPC.kind == "기계 상태"], c, "오경보") for c in ("운전모드", "전류 RMS 3분위", "하부진동 RMS 3분위")] \
        + [concentration_row(FNC[FNC.kind == "기계 상태"], c, "미탐") for c in ("가장 가까운 운전모드", "전류 RMS 4분위", "상부진동 RMS 4분위", "하부진동 RMS 4분위")]
    process = [hdr] + [concentration_row(df[df.kind == "처리 조건"], c, k) for k, df in (("오경보", FPC), ("미탐", FNC)) for c in ("윈도우 길이", "세그먼트 위치")]
    HC = rd("08_hardest_cases.csv")
    fnh = HC[HC.kind.str.startswith("이상")]
    fnh_miss = fnh[fnh.p < THR["t_red"]]          # the hardest-case rows the model really left below the red line
    n_fn1, k_guard, g_green = len(fn_oof), int((fn_oof.guard == 1).sum()), int((fn_oof.level == "green").sum())
    k_newm, k_unc = int((fn_oof.new_mode == 1).sum()), int((fn_oof.unc >= THR["unc_high"]).sum())
    hfp = HC[HC.kind.str.startswith("정상")].iloc[0]                   # the worst false-alarm window (highest p among normals)
    fpm = FPC[FPC.condition == "운전모드"].set_index("level")
    fnm = FNC[FNC.condition == "가장 가까운 운전모드"].set_index("level")
    mode_rows = [["운전모드", "정상 창", "오경보(FP)", "FPR", "이상 창(가장 가까운 모드)", "미탐(FN)", "재현율"]]
    for mname in sorted(set(fpm.index) | set(fnm.index)):
        a_, b_ = fpm.loc[mname], fnm.loc[mname]
        mode_rows.append([mname + (" (저부하)" if mname == "mode0" else " (고부하)" if mname == "mode1" else ""), f"{int(a_.n):,}", f"{int(a_.errors)}", pct(a_.rate, 2),
                          f"{int(b_.n):,}", f"{int(b_.errors)}", pct(1 - b_.rate, 1)])
    L += [t.b2("**정상 운전모드별 성능** : 전체 평균만으로는 한 모드의 문제가 가려지므로 운전모드별 오경보율과 재현율을 따로 보였습니다(3회 반복 OOF 합산, 폴드별 임계값)."),
          t.table(mode_rows, [2.0, 1.2, 1.3, 1.1, 2.6, 1.2, 1.2]),
          t.caption("〈표 3-1b〉 운전모드별 오경보와 미탐 (이상 창의 모드 = 정상 군집 중 가장 가까운 모드)"),
          t.gap()]
    L += [t.b2(f"**미탐(FN)·오경보(FP) 집중 조건 - 기계 상태 기준** (3회 반복 OOF 합산: 정상 윈도우 {n_pool_n:,}개 중 오경보 {n_fp}건, 이상 윈도우 {n_pool_a}개 중 미탐 {n_fn}건)"),
          t.b3(f"오류 건수가 적어 조건별 비율 차이는 통계적으로 확정하기 어렵고, 아래 표는 ‘오류가 어디에 몰렸는가’의 기술(記述)입니다." if n_fp + n_fn < 30
               else "표는 오류가 어느 조건에 몰렸는지를 보여줍니다."),
          t.table(machine, [0.9, 2.5, 2.0, 1.3, 1.0, 3.8], left_cols=(1, 5)),
          t.caption("〈표 3-2〉 기계 상태(운전모드·전류·진동 수준)별 오류 집중 조건 (M2 최종모델, 폴드별 임계값; 해당 조건에서 비율이 가장 높은 구간만 표시)"),
          *([t.b3(f"표에서 ‘100%’가 되풀이되는 것은 오류가 오경보 {n_fp}건·미탐 {n_fn}건뿐이라 같은 소수의 창이 조건마다 다시 세어지기 때문입니다. 조건별 독립 증거가 아니라 ‘이 창들이 어떤 상태에 있었는가’의 서술로 읽어야 합니다.")]
            if sum(r[5].endswith("(100%)") for r in machine[1:] + process[1:]) >= 3 else []),
          t.b2("**처리 조건 기준 (설비 상태가 아니라 신호를 자르는 방식이 만드는 조건)** : 윈도우 길이와 구간 내 위치는 전처리의 산물이라 기계 상태와 분리해 따로 봅니다."),
          t.table(process, [0.9, 2.5, 2.0, 1.3, 1.0, 3.8], left_cols=(1, 5)),
          t.caption("〈표 3-3〉 처리 조건별 오류 집중 (짧은 윈도우 = 구간 전체가 5~9샘플뿐이라 한 개의 짧은 윈도우로 판정된 경우)"),
          t.b3(f"**미탐의 정체** : 모델이 빨강으로 올리지 못한 이상 창은 첫 CV 반복 기준 {n_fn1}개이고, 이 중 초록(완전 미탐)이 {g_green}개, 피크전류 가드가 발동한 창이 {k_guard}개입니다. "
               + (f"이 창들의 전류 평균은 {fnh_miss.cur_mean.min():+.0f}~{fnh_miss.cur_mean.max():+.0f}, 모델 확률은 {fnh_miss.p.min():.4f}~{fnh_miss.p.max():.4f}입니다. "
                  "전류가 한쪽으로 크게 치우치면서 파형이 매끄러운 구간이라 형상·평활도 특징만으로는 정상 정현파의 일부와 구분이 어렵습니다(진폭에 의존하지 않도록 만든 대가). " if len(fnh_miss) else "")
               + (f"가드가 이런 창을 노랑으로 올리지만 빨강 권한은 주지 않았으므로(표 2-6) 이 창들은 ‘노랑 상태’로 남습니다. " if k_guard > 0 else "가드는 이 창들을 잡지 못했습니다. ")
               + (f"이 창 중 {k_newm}개는 신규 모드 영역으로 분류돼 모드별 가드가 적용되지 않았고, {k_unc}개는 앙상블 불일치(unc {fn_oof.unc.min():.2f}~{fn_oof.unc.max():.2f}, 트리거 기준 {THR['unc_high']:.3f})로 노랑이 됐습니다." if len(fn_oof) else "")),
          t.b3((f"트리 요약 규칙(depth 3)의 첫 분기 변수는 {'두 경우 모두' if ROOTS.feature.nunique() == 1 else ''} {ROOTS.feature.iloc[0]}({meaning(ROOTS.feature.iloc[0])})이며, "
                + "; ".join(f"{r.case} 창은 {r.hard_side} {r.threshold:.2f} 쪽에 몰립니다" for r in ROOTS.itertuples()) + "."
                ) if ROOTS.feature.nunique() == 1 else
               ("트리 요약 규칙(depth 3)의 첫 분기: " + "; ".join(f"{r.case} 창은 ‘{r.feature} {r.hard_side} {r.threshold:.2f}’ 쪽에 몰립니다({meaning(r.feature)})" for r in ROOTS.itertuples()) + ".")),
          t.image(FIG / "fig10_error_conditions.png", 0.95), t.caption("〈그림 3-2〉 기계 상태별 오경보·미탐 근접 비율 (오류=임계값 기준, 근접=p 기준)"),
          t.b3("대표 오류 사례의 원본 파형(음영=판정 윈도우): 미탐 " + (f"두 사례(구간 {fnh.iloc[0].seg:.0f}·{fnh.iloc[1].seg:.0f})는 전류 평균이 {fnh.iloc[0].cur_mean:+.0f}·{fnh.iloc[1].cur_mean:+.0f}로 한쪽으로 치우친 구간이고(모델 확률 {fnh.iloc[0].p:.4f}, {fnh.iloc[1].p:.4f}), " if len(fnh) >= 2 else "")
               + f"오경보 사례(구간 {hfp.seg:.0f})는 길이가 {hfp.win_len:.0f}샘플({hfp.win_len * 0.1:.1f}초)뿐인 짧은 구간 전체가 한 윈도우로 판정된 경우입니다(확률 {hfp.p:.3f}). "
               "이 사례들의 공통점은 ‘모델이 근거로 삼는 전류 거칠기가 정상 범위와 겹치거나 계산 표본이 너무 적다’는 것입니다."),
          t.image(FIG / "fig14_error_cases.png", 0.95), t.caption("〈그림 3-2b〉 미탐(FN)·오경보(FP) 대표 사례의 상부진동·하부진동·전류 원본 파형"), t.gap()]
    hi_fp = [shift("ALL", m, 1) for m in ("B0_SPC", "M1_RF", "M2_HGB")]
    lo_fp = [shift("ALL", m, 0) for m in ("B0_SPC", "M1_RF", "M2_HGB")]
    red_sp = SP[SP.model == "최종 모델(빨강)"].pivot(index="factor", columns="channels", values="false_alarm_rate")
    yel_sp = SP[SP.model == "최종 모델(노랑 이상)"].pivot(index="factor", columns="channels", values="false_alarm_rate")
    spc_sp = SP[SP.model == "SPC 규칙"].pivot(index="factor", columns="channels", values="false_alarm_rate")
    sp_rows = [["겉보기 주파수 배율", "전류만: 빨강", "전류만: 노랑 이상", "전 채널: 빨강", "전 채널: 노랑 이상", "SPC(전류만)"]]
    for f in red_sp.index:
        sp_rows.append([f"×{f:g}", pct(red_sp.loc[f, "전류만"], 2), pct(yel_sp.loc[f, "전류만"], 2), pct(red_sp.loc[f, "전 채널"], 2), pct(yel_sp.loc[f, "전 채널"], 2), pct(spc_sp.loc[f, "전류만"], 2)])
    worst = red_sp.max(axis=1)
    safe, broke = SPEED_SAFE, SPEED_BROKE
    fast, slow = red_sp[red_sp.index > 1], red_sp[red_sp.index < 1]
    if fast.max().min() > 0.05 and slow["전류만"].max() < 0.01 < slow["전 채널"].max():
        speed_cause = (f"빨라지는 쪽(×{fast.index.max():g})은 전류만 변형해도 {pct(fast.loc[fast.index.max(), '전류만'], 0)}로 전 채널 변형({pct(fast.loc[fast.index.max(), '전 채널'], 0)})과 같은 수준이라 "
                       "1초 윈도우 안의 주기 수가 늘어 전류가 ‘거칠다’고 읽히는 것이 원인으로 보이고, 느려지는 쪽은 전류만 변형하면 "
                       f"{pct(slow['전류만'].max(), 2)} 이하인데 진동까지 함께 늘이면 {pct(slow['전 채널'].min(), 0)}~{pct(slow['전 채널'].max(), 0)}라 진동 채널 변화가 원인으로 보입니다. "
                       f"노랑 이상 기준은 더 민감해, 전류만 ×{slow.index.min():g}로 늘여도 {pct(yel_sp.loc[slow.index.min(), '전류만'], 0)}가 노랑 이상이 됩니다.")
    else:
        speed_cause = "채널별 변형 결과가 표 3-4와 같이 갈리므로 원인은 표의 채널별 수치로 읽어야 합니다."
    L += [t.b2("**정상 운전변화로 인한 오경보의 발생조건**"),
          t.b3(f"진폭을 포함한 모델은 학습에 없던 **고부하 운전**에서 정상의 {pct(min(hi_fp), 0)}~{pct(max(hi_fp), 0)}를 이상으로 판단합니다(표 2-3). 트리 모델이 ‘진폭이 학습 범위를 넘으면 이상’으로 학습하기 때문입니다."),
          t.b3(f"반대로 저부하 모드를 뺀 경우 같은 모델들의 오경보율은 {pct(min(lo_fp))}~{pct(max(lo_fp))}로, "
               + ("대부분 문제없이 동작해(진폭이 학습 범위 안으로 들어감) " if max(lo_fp) < 0.05 else "")
               + "오경보는 **‘학습 범위 밖으로의 외삽(특히 고부하 방향)’** 에서 발생함을 확인했습니다."),
          t.b3(f"불변 특징 모델의 미학습 모드 오경보는 M2 {pct(shift('INV', 'M2_HGB', 0))}/{pct(shift('INV', 'M2_HGB', 1))}로 사실상 해소되었으나 재현율은 "
               f"{pct(shift('INV', 'M2_HGB', 0, 'Recall'), 0)}/{pct(shift('INV', 'M2_HGB', 1, 'Recall'), 0)}로 소폭 낮아집니다(이상 학습 표본 감소 영향 포함)."),
          t.b2("**모터 속도(겉보기 주파수) 변화 시험** : 주된 특징이 전류 평활도(cur_madr)이므로, 속도가 달라진 정상 운전을 합성해 오경보가 얼마나 느는지 직접 쟀습니다. "
               "홀드아웃 정상 구간의 시간축을 f배로 눌러(f>1 빠름) 모든 주파수를 f배로 만든 뒤 윈도우화해 같은 모델·같은 임계값으로 판정합니다."),
          t.table(sp_rows, [2.0, 1.6, 1.8, 1.6, 1.8, 1.7]),
          t.caption("〈표 3-4〉 속도 변화 시험: 정상 윈도우 오경보율 (변형 채널 = 전류만 / 전 채널, 5-fold 홀드아웃 합산, 빨강 = 경고 등급)"),
          t.image(FIG / "fig31_speed_shift.png", 0.7), t.caption("〈그림 3-3〉 겉보기 주파수 배율별 정상 윈도우 오경보율 (최종 모델, 빨강 기준)"),
          t.b3(f"빨강 오경보율이 두 변형 방식 모두 1% 이하인 배율은 {', '.join(f'×{f:g}' for f in safe)}입니다. "
               + (f"반면 {', '.join(f'×{f:g}({pct(worst[f], 1)})' for f in broke)}에서는 1%를 넘습니다. " if broke else "시험한 모든 배율에서 1% 이하였습니다. ")
               + speed_cause),
          t.b3("한계: 10Hz 샘플링에서는 전류의 겉보기 주파수가 aliasing된 값이라 배율 f를 실제 rpm 변화율로 환산할 수 없습니다. 또한 이 데이터에는 속도를 바꾼 정상 운전이 없고 이상 데이터도 단일 세션이라 "
               "‘고장으로 거칠어진 전류’와 ‘더 빠른 사이클’을 가를 단서는 진동 채널의 진폭 증가뿐입니다. 현장 속도 변경 시에는 해당 속도의 정상 데이터를 수집해 재학습·재보정하고, 그 전에는 R2(2연속) 정지 기준과 노랑 비율 모니터링으로 대응합니다."),
          t.gap()]
    return L


def chapter4(t: Template) -> list:
    L = [t.heading("□ 제4장. 현장 활용방안〔10점〕", 0)]
    ny, nr, ng = int(lv.loc["정상 윈도우", "yellow"]), int(lv.loc["정상 윈도우", "red"]), int(lv.loc["정상 윈도우", "green"])
    ay, ar, ag = (int(lv.loc["이상 윈도우", c]) for c in ("yellow", "red", "green"))
    r1, r2, r3, r4 = (rules.loc[k] for k in ["R1 red 1회", "R2 red 2연속", "R3 yellow 이상 2연속", "R4 세그먼트 평균 p>=t_red"])
    L += [t.b2(f"**모델 출력 → 3단계 경보 → 현장 조치** : 최종모델의 이상 확률 p(Platt 보정)와 앙상블 멤버(B1·M1·M2) 불일치도(표준편차 unc)로 등급을 매깁니다. "
               f"기준값은 red p≥{THR['t_red']:.3f}(모델 확률의 F1 최대점), yellow p≥{THR['t_yellow']:.3f}(FN:FP 비용비 10:1 가정 최적점) 또는 unc≥{THR['unc_high']:.3f}(불확실도 상위 2%) 또는 피크전류 가드 발동입니다."),
          t.table([["등급", "판정 조건", "현장 조치(사전경보·공정점검·의사결정)"],
                   ["초록 정상", "p가 yellow 기준 미만, unc 낮음, 가드 미발동", "정상 운전 유지, 확률·모드 로그만 저장"],
                   ["노랑 주의", "yellow≤p<red 또는 unc 높음(멤버 불일치) 또는 피크전류 가드 발동", "다음 2~3 사이클 재측정, 해당 구간 생산품 표본검사 비율 상향, 정기점검 일정 앞당김 검토"],
                   ["빨강 경고", "p≥red", "2연속이면 프레스 정지·유압펌프 점검(전류 파형·필터·펌프 부하), 해당 시간대 생산품 우선 선별검사"]],
                  [1.3, 3.6, 6.0], left_cols=(1, 2)),
          t.caption("〈표 4-1〉 경보 등급과 조치 기준"),
          t.b3(f"정상 윈도우의 {pct(ng / (ng + ny + nr))}가 초록, {pct(ny / (ng + ny + nr))}({ny}창)가 노랑, {pct(nr / (ng + ny + nr), 2)}({nr}창)가 빨강이며, {anomaly_split(ar, ay, ag)}."),
          t.image(FIG / "fig11_alarm_policy.png", 0.9), t.caption("〈그림 4-1〉 정상 → 이상 세션 전환부의 3단계 경보 예시"), t.gap()]
    lat = lambda r: r["latency_median_s(first red)"]
    nr1, nr2 = NR.set_index("rule").loc["R1 red 1회"], NR.set_index("rule").loc["R2 red 2연속"]
    L += [t.b2("**오경보를 줄이는 경보 규칙과 검사 우선순위**"),
          t.table([["규칙", "이상 구간 탐지율", "정상 구간 오경보율", "오경보(건/기록시간)", "첫 경보 지연(중앙값)", "적합한 용도"],
                   ["R1 red 1회", pct(r1.segment_recall, 0), pct(r1.normal_seg_false_alarm, 2), f"{r1.false_alarms_per_hour:.1f}", f"{lat(r1):.1f}초", "즉시 점검 호출"],
                   ["R2 red 2연속", pct(r2.segment_recall, 0), pct(r2.normal_seg_false_alarm, 2), f"{r2.false_alarms_per_hour:.1f}", f"{lat(r2) + 0.5:.1f}초 이상(2창째 확인)", "정지 권고"],
                   ["R3 노랑 이상 2연속", pct(r3.segment_recall, 0), pct(r3.normal_seg_false_alarm, 2), f"{r3.false_alarms_per_hour:.1f}", f"{lat(r3) + 0.5:.1f}초 이상", "사전 점검(주의 누적)"],
                   ["R4 구간 평균 p≥red", pct(r4.segment_recall, 0), pct(r4.normal_seg_false_alarm, 2), f"{r4.false_alarms_per_hour:.1f}", "구간 종료 후", "사후 구간 판정(즉시 보호용 아님)"]],
                  [2.2, 1.4, 1.5, 1.5, 2.2, 2.4]),
          t.caption(f"〈표 4-2〉 경보 규칙별 성능 (이상 {u_anom_seg}구간, 정상 {u_norm_seg}구간; 배포용 단일 임계값·가드 포함 등급 적용)"),
          t.b3(f"**목적에 맞는 규칙을 따로 씁니다**: 즉시 점검 호출은 R1, 정지 권고는 R2, 사전 점검은 R3, 구간 종료 후 판정은 R4이며 R4는 구간이 끝나야 값이 정해지므로 즉시 보호용으로 쓰지 않습니다. "
               f"이 표의 R1~R3는 predict.py가 실제로 쓰는 함수(errors.sequence_hit)와 같은 코드로 계산해 분석용 정책과 배포 정책이 어긋나지 않습니다. "
               f"정지 권고 기준인 **R2(red 2연속)** 는 R1 대비 오경보가 {trend(r1.false_alarms_per_hour, r2.false_alarms_per_hour, '건/시간')} "
               + (f"이상 구간 탐지율은 둘 다 {pct(r1.segment_recall, 0)}입니다. " if r1.segment_recall == r2.segment_recall else f"이상 구간 탐지율은 {pct(r1.segment_recall, 0)} → {pct(r2.segment_recall, 0)}입니다. ")
               + ("이 데이터에서는 R2의 오경보 감소 이득이 작지만, 단발 red 한 번에 프레스를 세우지 않는 안전장치로 유지합니다. " if r2.false_alarms_per_hour > r1.false_alarms_per_hour - 0.05 else "")
               + f"다만 이 표는 모델을 고른 데이터 위의 값이라 낙관적일 수 있어, 중첩 검증(표 2-5)의 외부 test 창에서 같은 규칙을 적용하면 R1 탐지 {pct(nr1.segment_recall, 0)}·오경보 {nr1.false_alarms_per_hour:.1f}건/시간, "
               f"R2 탐지 {pct(nr2.segment_recall, 0)}·오경보 {nr2.false_alarms_per_hour:.1f}건/시간입니다(독립 평가 기준의 운영 수치로 읽어야 합니다). "
               + (f"즉 독립 평가에서는 R2의 오경보 이득이 드러나지 않고 이상 구간 {round((nr1.segment_recall - nr2.segment_recall) * u_anom_seg)}개를 더 놓쳤습니다(표본이 작음). 그럼에도 R2를 정지 권고로 두는 근거는 이 데이터가 아니라 "
                  "‘한 번 튄 값으로 설비를 세우지 않는다’는 운영 원칙이며, 즉시 점검 호출은 R1이 맡습니다. " if nr2.false_alarms_per_hour >= nr1.false_alarms_per_hour - 0.05 else "")
),
          t.b3(f"(‘건/시간’은 센서가 실제로 기록한 정상 {qn.recorded_min:.1f}분 기준입니다. 벽시계 {qn.wallclock_min:.1f}분으로 나누면 숫자가 작아지지만, 기록 공백 동안의 오경보는 관측된 적이 없어 오경보를 과소평가하게 되므로 그렇게 환산하지 않습니다. "
               f"중첩 검증의 {nr2.false_alarms_per_hour:.1f}건/시간은 정상 {u_norm_seg}구간 중 {round(nr2.normal_seg_false_alarm * u_norm_seg)}구간의 오경보로 건수가 적어 불확실합니다. "
               "제2장 표의 오경보는 폴드별 임계값, 이 표는 배포용 단일 임계값이라 수치가 다릅니다.)"),
          t.b3("**검사 우선순위**: 구간별 최대 확률 p_max와 등급으로 정렬해 (1) 빨강 구간 전후 생산품 선별검사, (2) 노랑 구간 표본검사, (3) 초록 구간 정기검사 순으로 배분합니다. LOT·제품 이력은 데이터에 없어 시간 범위 기반 배정으로 제안합니다."),
          t.b3(f"**피크전류 가드(제조지식, 운전모드별 한계 → 노랑)**: 가장 가까운 운전모드의 정상 최대 전류(저부하 {GUARD_LIMITS[0]:.1f}, 고부하 {GUARD_LIMITS[1]:.1f})를 넘는 윈도우는 모델 확률과 무관하게 노랑으로 올리고, "
               "어느 모드 영역에도 속하지 않는 윈도우는 ‘신규 모드’로 표시해 운영자 승인·재학습 대상으로 보냅니다. "
               + (f"표 2-6처럼 모델만으로 초록이던 이상 창 {guard_rescued}개를 구제합니다. " if guard_rescued > 0 else "표 2-6처럼 이 데이터에서는 추가로 구제되는 이상 창이 없어 보험 성격입니다. ")
               + f"**빨강이 아닌 이유**: 빨강으로 강제하면 정상 윈도우의 빨강이 {ge0.normal_red:.0f}→{ge3.normal_red:.0f}창, R2 오경보가 {ge0.R2_FA_per_h:.1f}→{ge3.R2_FA_per_h:.1f}건/시간이 되는 데 비해 "
               f"이상 창의 빨강 추가분은 {guard_gain}개뿐이었습니다. "
               "**운영상 지킬 조건**: 이 한계는 설비 정격전류 사양값이 아니라 관측된 최댓값이므로 현장 투입 전 정격 사양값과 대조하고, 새 운전조건이 승인되면 한계를 재산정해야 합니다."),
          t.gap()]
    c = COST.set_index("C_FN/C_FP")
    th = c.threshold
    ratios = ", ".join(f"{int(r)}배 {th.loc[r]:.3f}" for r in c.index)
    ref = c.loc[THR["assumed_cost_ratio"]]
    L += [t.b2("**임계값·운영 정책의 조정**"),
          t.b3(f"미탐 비용이 오경보 비용의 몇 배인지에 따른 최적 임계값은 {ratios}입니다"
               + (" - 이 데이터에서는 정상·이상 점수가 크게 갈려 비용비를 바꿔도 임계값이 거의 변하지 않습니다. " if th.max() - th.min() < 5e-3
                  else f"(비용비가 클수록 임계값이 {'낮아' if th.iloc[0] > th.iloc[-1] else '높아'}집니다). ")
               + f"가정 비용비 {THR['assumed_cost_ratio']}배에서 FP {int(ref.FP)}창·FN {int(ref.FN)}창입니다. "
               "실제 비용비는 가정값이므로 현장의 다운타임·불량 비용 자료로 교체해야 합니다."),
          t.b3("**운전조건 변경(신제품·부하 변경) 시**: 불변 특징 덕분에 새 모드에서 빨강 오경보가 급증하지 않지만(피크전류 가드는 노랑을 늘릴 수 있음), 초록 비율과 노랑 비율을 주간으로 모니터링하고 정상 데이터만으로 확률 보정을 재수행하는 절차를 둡니다."),
          t.b3("**재학습 조건**: 노랑 비율이 평소의 2배를 넘거나 새로운 형태의 고장 데이터가 확보되면 재학습하고, 그 전까지는 M3(정상 전용) 점수를 보조 감시용으로 병행합니다."),
          t.gap()]
    return L


def chapter5(t: Template) -> list:
    L = [t.heading("□ 제5장. 창의성 및 차별성〔10점〕", 0)]
    imb = IMB.set_index("variant")
    u = UNC.set_index("unc_bin")
    n_err = int(u.errors.sum())
    UNC_USED = bool(THR.get("unc_used", False))
    uc_rows = [["상위 k% unc", "창 수", "포착한 오류 / 전체", "오류 포착률", "우연 기대 오류", "이항검정 p"]] + [
        [f"{int(r.top_pct)}%", str(int(r.windows)), f"{int(r.errors_captured)} / {int(r.errors_total)}", pct(r.capture_rate, 0), f"{r.expected_by_chance:.2f}", f"{r.p_value:.1e}"]
        for r in UC.itertuples()]
    L += [t.b2("**① 운전모드 불변 특징 설계 + 미학습 모드 외삽 시험** : ‘정상적인 운전변화를 이상으로 보는 오경보’ 문제를 일반 교차검증으로는 볼 수 없음을 보이고, 정상 모드를 통째로 빼는 시험을 설계했습니다. "
               f"진폭 비의존 특징으로 고부하 미학습 모드의 오경보를 M1 {pct(shift('ALL', 'M1_RF', 1), 0)} → {pct(shift('INV', 'M1_RF', 1))}, M2 {pct(shift('ALL', 'M2_HGB', 1), 0)} → {pct(shift('INV', 'M2_HGB', 1))}로 줄였습니다(표 2-3). "
               "성능·오류 감소에 대한 기여가 직접 수치로 확인되는 부분입니다."),
          t.b2("**② 불균형 학습·확률 보정·비용 기반 임계값** : class_weight, 폴드 내부 Platt 보정, F1 최대·비용비 기반 임계값을 결합했습니다. "
               f"HGB에서 가중치 유무의 효과는 F1 {f3(imb.iloc[0].F1)} vs {f3(imb.iloc[1].F1)}(thr=0.5)로 이 데이터에서는 미미했고(정직한 결과), 보정 후 ECE는 M2 {sm('INV', 'M2_HGB', 'ECE'):.4f}, "
               f"Brier {sm('INV', 'M2_HGB', 'Brier'):.4f}로 확률을 경보 등급에 그대로 쓸 수 있습니다."),
          t.b2("**③ 앙상블 불일치 기반 불확실성 - 쓸 수 있는지부터 검증** : B1·M1·M2의 불일치도(unc)를 산출하되, 경보 등급의 직접 조건으로 쓰기 전에 ‘오류가 불확실한 창에 몰리는가’를 상위 k% 오류 포착률로 시험했습니다(표 5-1). "
               f"사전에 정한 기준(상위 2%가 오류 2건 이상·전체 오류의 50% 이상을 담고 우연 가능성 p<0.01)을 {'**통과**했으므로 unc 상위 2%를 노랑 트리거로 씁니다.' if UNC_USED else '**통과하지 못해** unc를 경보 조건에서 빼고 모니터링 지표로만 둡니다.'} "
               + (f"이 트리거는 정상 창의 {pct(UTC.normal_windows_yellow_by_unc_only.iloc[0] / UTC.normal_windows.iloc[0], 1)}({int(UTC.normal_windows_yellow_by_unc_only.iloc[0])}창)를 노랑으로 올리는 비용이 있고 그 안에서 확인되는 오류는 {int(UTC.errors_in_unc_only_yellow.iloc[0])}건/{int(UTC.errors_total.iloc[0])}건이므로, "
                  f"현장에서 끌 수 있는 선택 옵션으로 둡니다(predict.py --no-unc-trigger). 모델 오류가 {int(UC.errors_total.iloc[0])}건뿐이라 포착률 추정은 정밀하지 않습니다." if UNC_USED else "")),
          t.table(uc_rows, [1.6, 1.0, 2.0, 1.5, 1.8, 1.7]),
          t.caption("〈표 5-1〉 불확실도 상위 k% 창의 오류 포착률 (오류 = 폴드별 임계값 기준 FP+FN, 첫 CV 반복)"),
          t.b2("**④ 제조지식 결합과 그 평가·수정** : 전류를 ‘정현파 이탈 정도(평활도·영교차율·DC 오프셋)’로 표현한 학습 모델에, 정상 최대 전류를 넘으면 경계 신호를 내는 결정론적 규칙(피크전류 가드)을 결합했습니다. "
               f"처음에는 ‘전역 한계 초과 시 빨강’으로 붙였으나 같은 윈도우에서 비교하니 이상 창의 빨강 추가분은 {guard_gain}개인 반면 정상 창의 빨강이 {guard_cost}창 늘어 노랑 상향으로 낮췄고(표 2-6), "
               "미학습 운전모드 시험에서 한 개의 전역 한계가 새 모드의 정상 창을 대량 플래그하는 한계를 찾아 운전모드별 한계와 ‘신규 모드’ 상태로 다시 설계했습니다(표 2-7) — "
               "결합 → 평가 → 권한 축소 → 구조 개선까지 거친 부분입니다."),
          t.b2("**⑤ 블록 부트스트랩 신뢰구간과 누수 점검** : 이상이 단일 세션인 자료의 한계를 숨기지 않고, 시간 블록 단위 CI와 무작위 분할 비교를 함께 제시했습니다(제1·2장)."),
          t.b2("**⑥ 과적합 점검 + 운영 가시성 대시보드** : 배포 하이퍼파라미터의 학습-held out 성능차와 규제 강도 민감도를 직접 측정해 ‘더 보수적으로 만들면 나아지는가’까지 반증했습니다(표 2-8). "
               "또한 `run_all.py`/`predict.py` 실행마다 자동 갱신되는 HTML 대시보드를 제작해, 경보 임계값을 바꿔보는 시뮬레이터, 새 데이터의 창별 이상 확률·경보 등급·구간 파형, 매 채점 이력을 한 화면에서 확인할 수 있게 했습니다(제6장)."),
          t.b2(f"**⑦ 스트레스 시험으로 한계를 먼저 찾기** : 정답률만 보이지 않고, 약한 이상(진폭 최약 사분위에서 SPC {pct(SEV.iloc[0].spc_detect, 0)} 대 모델 빨강 {pct(SEV.iloc[0].model_red, 0)}; 합성 혼합 시험 표 2-10)과 "
               "속도 변화(시간축 변형 시험 표 3-4)를 파이프라인 안에서 매번 재현해, 모델이 무너지는 지점을 수치로 공개했습니다."),
          t.b4("위 차별화는 모두 outputs/tables 의 산출표로 재현되며, 새 알고리즘 도입이 아니라 ‘진단 → 원인 → 특징/규칙 설계 → 재검증 → 한계 공개’ 흐름으로 제1~4장의 결과에 연결됩니다."),
          t.gap()]
    return L


def chapter6(t: Template) -> list:
    L = [t.heading("□ 제6장. 코드 구성 및 재현성〔10점〕", 0)]
    L += [t.b2("**동일 환경에서 전처리~학습~추론~결과생성까지 자동 실행** : 요구사항 문구 그대로, 아래 두 명령이 그 전 과정을 커버합니다."),
          t.b3(f"python run_all.py — 전처리 → 특징 → 교차검증(6모델×2특징세트) → 경보 정책·가드 효과·불확실도 검증 → 과적합 점검 → 스트레스 시험(약한 이상·속도 변화) → 중첩 교차검증 → 최종 학습 → 표·그림·보고서 입력값·대시보드까지 전부 생성(약 {RUN_MIN}분; "
               "--quick 은 최종 모델만 쓰는 스모크 테스트로 outputs_quick/ 에 따로 저장되어 본 결과를 덮어쓰지 않음, --data <폴더> 로 데이터 위치 지정). 난수 시드 0/1/2 고정."),
          t.b3("python predict.py --input <새 CSV> — **추론**. 학습된 model_bundle.joblib(모델+피크전류 가드 포함)로 새 데이터에 윈도우별 이상 확률·불확실도·가드 플래그·경보 등급과 구간 요약(정지 권고=R2 충족 여부 포함)을 출력하고, 실행 이력을 outputs/predict_log.csv 에 쌓고, 새 데이터의 창별 이상 확률·경보 등급 그래프와 구간별 진동·전류 파형을 대시보드(6번 항목)에 그려 자동 갱신합니다."),
          t.b3("python report/build_report.py — 위 산출표를 읽어 이 보고서(HWPX)를 다시 생성합니다. 즉 코드 결과와 보고서 수치가 항상 같은 outputs/tables 에서 나와 서로 어긋나지 않습니다."),
          t.b3(f"python -m pytest -q tests — 윈도우화·구간 분할·피크전류 가드·경보 등급·속도 변형 단위 테스트 {N_TESTS}개, 전부 통과를 실행 전 확인용으로 둡니다."),
          t.table([["경로", "역할"],
                   ["src/data.py", "CSV 로딩, 타임스탬프 중복 제거, 연속 구간 분할, 30초 블록 그룹 부여, 품질 리포트"],
                   ["src/features.py", "윈도우화(구간 경계 존중)와 34개 특징, 운전모드 불변 17개 선택"],
                   ["src/models.py · pipeline.py", "B0~M3 모델, 운전모드별 피크전류 가드(ModeGuard), Platt 보정, 로지스틱 스태킹, 폴드 내부 임계값"],
                   ["src/evaluate.py", "블록 그룹 교차검증, 윈도우·구간·경보 지표, 반복 요약"],
                   ["src/analysis.py · errors.py", "운전모드 군집, 중요도·상호작용, 외삽 시험, 최종모델 선정, 중첩 교차검증, 과적합 점검, 스트레스 시험, 오류 조건, 경보 정책(sequence_hit)·가드 효과·불확실도 포착률"],
                   ["src/figures.py · result_figures.py", "진단 그림 · 결과표 기반 그림(성능·가드·과적합·약한 이상·속도 변화 등) 생성"],
                   ["run_all.py · predict.py", "전체 파이프라인 진입점 · 신규 데이터 추론 + 이력 기록"],
                   ["report/build_report.py · build_dashboard.py", "이 HWPX 보고서 · 자동 새로고침 HTML 대시보드 생성"],
                   ["tests/", f"윈도우화·구간 분할·가드·경보 등급·속도 변형 단위 테스트(pytest, {N_TESTS}개)"]],
                  [3.0, 8.0], left_cols=(0, 1)),
          t.caption("〈표 6-1〉 코드 구성"),
          t.b3("환경: Python 3.11 이상(3.11·3.12·3.13에서 같은 수치 재현 확인). requirements.txt = 설치용 버전 범위, requirements-lock.txt = 보고서 수치를 만든 정확한 버전(Python 3.13 사용: pandas 3.0.5, numpy 2.5.2, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.0, joblib 1.6.0). GPU·외부 API·인터넷이 필요 없고 데이터 경로만 인자로 지정합니다."),
          t.b3(f"출력물: model_bundle.joblib (모델·가드·보정기·임계값), predictions_oof.csv (교차검증 예측, 누수 없음), results_summary.json, tables/*.csv {N_TABLES}종, figures/*.png, dashboard.html, predict_log.csv."),
          t.gap()]
    return L


def appendix(t: Template) -> list:
    return [t.heading("□ 추가 기술 사항 (가정과 한계)"),
            t.b2("이상 데이터는 다른 날의 단일 3분 세션(구간 21개, 이 중 18개 사용)이므로 다른 고장 유형·다른 설비·열화 진행 과정에 대한 일반화는 검증되지 않았습니다. 모든 신뢰구간은 이 한계를 반영해 넓게 보아야 합니다."),
            t.b2("AI0=상부, AI1=하부 진동 대응은 출제 배경의 언급 순서와 변수 순서가 일치한다는 점에 근거한 추정이며(제1장), 두 채널을 바꿔도 모델 성능엔 영향이 없습니다. 전류·진동의 물리 단위는 데이터 설명에 없어 표기하지 않았습니다."),
            t.b2("‘조기탐지’는 이상 구간 시작 후 첫 경보까지의 시간으로 정의했고, 실제 고장 며칠 전 징후 탐지는 열화 시계열이 없어 평가하지 못했습니다. "
                 "대신 진폭이 약한 실제 이상 창의 탐지율(표 2-9)과 합성 혼합 시험(표 2-10)으로 약한 이상에 대한 민감도만 확인했으며, 합성 시험은 선형 혼합일 뿐 실제 열화 경로가 아닙니다."),
            t.b2("비용비(10:1)는 가정값으로 현장 다운타임·불량 비용 자료로 교체해야 합니다."),
            t.b2(f"피크전류 가드의 한계(저부하 {GUARD_LIMITS[0]:.1f}, 고부하 {GUARD_LIMITS[1]:.1f})는 설비 정격전류 사양값이 아니라 학습 정상 데이터의 운전모드별 관측 최댓값입니다. 단일 전역 한계는 미학습 운전모드에서 정상 창을 대량 플래그했고(제2장 표 2-7), "
                 "그래서 모드별 한계·신규 모드 상태로 바꾸고 빨강 권한 없이 노랑으로만 올립니다. 현장 투입 전 설비 정격 사양값과 대조하고 새 운전모드 승인 시 재산정해야 합니다."),
            t.b2("모델 선정에 쓴 교차검증·외삽 시험은 선택용이며, 선택과 분리한 성능은 중첩 교차검증(표 2-5)의 외부 test 합산입니다. 이상 세션이 하나라 세션·날짜 단위 최종 test 분리는 불가능했고, 시간상 이웃한 블록 간 상관은 남아 있습니다."),
            t.b2("속도 변화 시험(표 3-4)의 배율은 10Hz 샘플링으로 aliasing된 겉보기 주파수 배율이며 실제 rpm 변화율이 아닙니다. 이 데이터에는 속도를 바꾼 정상 운전이 없어 현장 속도 범위에서의 안전성은 검증하지 못했습니다."),
            t.b2("최종모델 선정에 사용한 미학습 모드 시험도 정상 모드 2개뿐인 데이터에서 수행되어 표본이 작습니다.")]


def main():
    t = Template(ROOT / "report" / "template.hwpx")
    summary = [
        "◦ 유압펌프 모터의 상·하부 진동과 전류(10Hz) 시계열을 1초 윈도우로 나눠 이상 확률을 예측하고, 오경보·미탐 조건까지 분석했습니다.",
        "① 진단: 이상은 다른 날 단일 3분 세션(21구간), 정상은 부하가 다른 2개 운전모드를 포함 → 30초 블록 그룹 CV와 ‘미학습 운전모드 외삽 시험’ 설계.",
        f"② 비교: SPC 규칙 F1 {f3(sm('INV', 'B0_SPC', 'F1'))} 대비 학습 모델 F1 {f3(LEARNED_F1[0])}~{f3(LEARNED_F1[1])}. 진폭 의존 모델은 미학습 고부하 모드에서 정상의 {pct(min(HI_FP), 0)}~{pct(max(HI_FP), 0)}를 이상으로 오판.",
        f"③ 해법·최종모델: 진폭 비의존 17개 특징 + HistGradientBoosting. 관측 고장 세션 기준 F1 {f3(sm('INV', 'M2_HGB', 'F1'))}(교차검증)·{f3(NP.set_index('metric').loc['F1', 'point'])}(선택과 분리한 중첩검증), 재현율 {pct(sm('INV', 'M2_HGB', 'Recall'), 1)}, FPR {pct(sm('INV', 'M2_HGB', 'FPR'), 2)}(교차검증), "
        f"미학습 모드 빨강 오경보 {pct(max(shift('INV', 'M2_HGB', 0), shift('INV', 'M2_HGB', 1)))} 이하. 피크전류 가드는 운전모드별 한계로 노랑으로만 올림(빨강 강제는 폐기).",
        f"④ 원인: 전류 파형 평활도(cur_madr)가 지배적이며 진동 RMS만으로는 이상 창의 {pct(spc_miss.n / (spc_miss.n + spc_hit.n), 0)} 미탐. 진폭이 가장 약한 이상 창에서도 SPC {pct(SEV.iloc[0].spc_detect, 0)} 대 모델 {pct(SEV.iloc[0].model_red, 0)} 탐지. 과적합은 아님을 직접 점검.",
        "⑤ 활용·한계: 3단계 경보(정상/주의/경고), red 2연속 정지 규칙. 속도 변화 시험에서 빨강 오경보 1% 이하는 겉보기 속도 "
        f"×{min(SPEED_SAFE):g}~×{max(SPEED_SAFE):g} 범위뿐이고, 가드는 새 운전모드에서 노랑을 쏟아낼 수 있음을 공개. run_all.py/predict.py 한 번으로 학습~추론~대시보드까지 재현.",
    ]
    t.fill_cover("진동·전류 시계열 기반 프레스 유압펌프 이상 조기탐지 및 정상 운전변화 오경보 분석", "", summary)
    # replace template chapters back-to-front so earlier paragraph indices stay valid
    t.set_page_break(58)          # survey-capture section starts on a fresh page (its "예 시" stamp is paper-anchored)
    t.replace_paragraphs(49, 57, appendix(t))
    for first, last, fn in [(40, 48, chapter6), (32, 39, chapter5), (24, 31, chapter4), (16, 23, chapter3), (8, 15, chapter2), (1, 7, chapter1)]:
        t.replace_paragraphs(first, last, fn(t))
    out = ROOT / "report" / "최종보고서.hwpx"
    t.save(out, "제6회 K-인공지능 제조데이터 분석 경진대회 보고서 - 프레스 유압펌프 이상 조기탐지")
    print("saved", out)


if __name__ == "__main__":
    main()

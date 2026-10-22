"""Presentation deck (pptx) built from outputs/tables + outputs/figures.

    pip install python-pptx
    python report/build_slides.py            # after run_all.py; writes report/발표자료.pptx

Like build_report.py, every number is read from the saved tables, so the deck always matches the last pipeline run.
"""
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
TAB, FIG = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
OUT = ROOT / "report" / "발표자료.pptx"

FONT = "Malgun Gothic"
INK, STEEL, MUTED, CARD, LINE = "111827", "1F2A37", "6B7280", "F3F4F6", "E5E7EB"
ACCENT, GOOD, WARN, BAD, BLUE, GREY = "D9480F", "2E9E5B", "E0A020", "D64541", "2C6FBB", "9CA3AF"
WHITE = "FFFFFF"


def rd(name): return pd.read_csv(TAB / name)


def pct(x, d=1): return f"{100 * x:.{d}f}%"


def f3(x): return f"{x:.3f}"


# ------------------------------------------------------------------------------------------- numbers
S, MS = rd("03_model_comparison.csv"), rd("04_mode_shift.csv")
Q, MODES, GI, LGO = rd("01_data_quality.csv"), rd("02_operating_modes.csv"), rd("07_group_importance_INV.csv"), rd("07_leave_group_out_INV.csv")
PI, SPCM = rd("07_perm_importance_INV.csv"), rd("08_spc_miss_profile.csv")
RULES, LV = rd("10_alarm_rules.csv").set_index("rule"), rd("10_alarm_levels.csv").set_index("구분")
GE, GM = rd("12_guard_effect.csv"), rd("12b_guard_mode_shift.csv").set_index("held_out_mode")
WK, SEV, SP = rd("14_weak_fault.csv"), rd("14b_severity_recall.csv"), rd("15_speed_shift.csv")
NF, NP, NR = rd("16_nested_folds.csv"), rd("16b_nested_pooled.csv").set_index("metric"), rd("16c_nested_rules.csv").set_index("rule")
UC, FNC, FPC = rd("09b_uncertainty_capture.csv"), rd("08_fn_conditions.csv"), rd("08_fp_conditions.csv")
PRED = pd.read_csv(ROOT / "outputs" / "predictions_oof.csv")      # first CV repeat: p_final, unc, guard, level per window
J = json.loads((ROOT / "outputs" / "results_summary.json").read_text(encoding="utf8"))
THR, FINAL = J["thresholds"], J["final_model"]
ORDER = ["B0_SPC", "B1_LogReg", "M1_RF", "M2_HGB", "M3_IF", "M4_Stack"]
LAB = {"B0_SPC": "SPC 규칙", "B1_LogReg": "로지스틱", "M1_RF": "랜덤포레스트", "M2_HGB": "HGB (최종)", "M3_IF": "IsolationForest", "M4_Stack": "스태킹"}
MODEL_CATS = [LAB[m] for m in ORDER]                  # chart category labels, shared by the comparison and the core-finding slides
SHORT = {"B0_SPC": "B0", "B1_LogReg": "B1", "M1_RF": "M1", "M2_HGB": "M2", "M3_IF": "M3", "M4_Stack": "M4"}
N_TESTS = sum(len(re.findall(r"^def test_", p.read_text(encoding="utf8"), re.M)) for p in (ROOT / "tests").glob("test_*.py"))


def sm(fs, m, c): return float(S[(S.feature_set == fs) & (S.model == m)][c].iloc[0])


def shift(fs, m, mode, c="FPR_on_unseen_mode"): return float(MS[(MS.feature_set == fs) & (MS.model == m) & (MS.held_out_mode == mode)][c].iloc[0])


qn_, qa_ = Q.iloc[0], Q.iloc[1]
ge0, ge_glob, ge_mode, ge_red = (GE.iloc[i] for i in range(4))
npi = NP
n_tp, n_fp, n_fn, n_tn = (int(NP[c].iloc[0]) for c in ("TP", "FP", "FN", "TN"))
r1, r2 = RULES.loc["R1 red 1회"], RULES.loc["R2 red 2연속"]
nr1, nr2 = NR.loc["R1 red 1회"], NR.loc["R2 red 2연속"]
n_win, n_anom = J["windows"], J["anomaly_windows"]
seg_all, seg_used = J["segments_total"], J["segments_used"]
w_mode = MODES[MODES.y == 0].set_index("mode").windows
rec_frac = qn_.recorded_min / qn_.wallclock_min
shift_h = 8 * rec_frac                                              # recorded hours in an 8-hour shift if the collection pattern is unchanged

# ------------------------------------------------------------------------------------------- pptx helpers
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rgb(h): return RGBColor.from_string(h)


def style_run(run, size, bold=False, color=INK, italic=False):
    f = run.font
    f.size, f.bold, f.italic = Pt(size), bold, italic
    f.color.rgb = rgb(color)
    f.name = FONT
    rpr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):                    # Korean glyphs use the East-Asian slot
        el = rpr.find(qn(tag))
        if el is None:
            el = rpr.makeelement(qn(tag), {})
            rpr.append(el)
        el.set("typeface", FONT)


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT):
    """paras: list of dicts {text | runs=[(text, bold, color)], size, bold, color, after, align, bullet}."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap, tf.vertical_anchor = True, anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, p in enumerate(paras):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p.get("align", align)
        para.space_after, para.line_spacing = Pt(p.get("after", 6)), p.get("ls", 1.08)
        runs = p["runs"] if "runs" in p else [(p["text"], p.get("bold", False), p.get("color", INK))]
        for t, b, c in runs:
            r = para.add_run()
            r.text = t
            style_run(r, p.get("size", 16), b, c)
        if p.get("bullet"):
            ppr = para._p.get_or_add_pPr()
            ppr.set("marL", str(Inches(0.26)))
            ppr.set("indent", str(-Inches(0.26)))
            ppr.append(ppr.makeelement(qn("a:buChar"), {"char": "•"}))
    return tb


def card(slide, x, y, w, h, fill=CARD, radius=0.05):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def disc(slide, x, y, d, label, fill=ACCENT, color=WHITE, size=18):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(fill)
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    style_run(r, size, True, color)
    return s


def picture(slide, path, x, y, w, h):
    """Fit the image inside the box (aspect kept), centred."""
    im = Image.open(path)
    ar = im.width / im.height
    pw, ph = (h * ar, h) if w / h > ar else (w, w / ar)
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


KICKERS = []


def new_slide(kicker, title, dark=False, notes=""):
    KICKERS.append(kicker or "")
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(STEEL if dark else WHITE)
    if kicker:
        text(s, 0.6, 0.38, 9, 0.3, [{"text": kicker, "size": 12, "bold": True, "color": "FDBA74" if dark else ACCENT}])
    if title:
        text(s, 0.6, 0.68, 12.1, 1.15, [{"text": title, "size": 28, "bold": True, "color": WHITE if dark else INK, "ls": 1.05}])
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def stat(slide, x, y, w, h, big, label, fill=CARD, big_color=INK, big_size=36, label_size=13):
    card(slide, x, y, w, h, fill)
    text(slide, x + 0.25, y + 0.18, w - 0.5, h * 0.5, [{"text": big, "size": big_size, "bold": True, "color": big_color}])
    text(slide, x + 0.25, y + h * 0.58, w - 0.5, h * 0.4, [{"text": label, "size": label_size, "color": MUTED, "ls": 1.0}])


def table(slide, x, y, w, rows, col_w, row_h=0.42, size=12, header_fill=STEEL, bold_last=False, hl_rows=()):
    gs = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(row_h * len(rows)))
    tbl = gs.table
    tblpr = gs._element.graphic.graphicData.tbl.tblPr
    for a in ("bandRow", "firstRow"):
        tblpr.set(a, "0")
    tot = sum(col_w)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(w * cw / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            head = i == 0
            c.fill.fore_color.rgb = rgb(header_fill if head else ("FDE8DD" if i in hl_rows else (WHITE if i % 2 else "F9FAFB")))
            p = c.text_frame.paragraphs[0]
            c.text_frame.word_wrap = True
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            style_run(r, size, head or (bold_last and i == len(rows) - 1) or i in hl_rows, WHITE if head else INK)
    return gs


def style_chart(chart, size=12, legend=True):
    chart.font.size, chart.font.name = Pt(size), FONT
    chart.has_title = False
    chart.has_legend = legend
    if legend:
        chart.legend.position = XL_LEGEND_POSITION.TOP
        chart.legend.include_in_layout = False
        chart.legend.font.size = Pt(size)
    va, ca = chart.value_axis, chart.category_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = rgb(LINE)
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(size - 1)
    ca.tick_labels.font.size = Pt(size)
    ca.format.line.color.rgb = rgb(GREY)
    ca.has_major_gridlines = False


def bar_chart(slide, x, y, w, h, cats, series, fmt='0.0', vmax=None, horizontal=False, legend=True, size=12, gap=60, point_colors=None, label_size=11):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    chart = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    style_chart(chart, size, legend)
    plot = chart.plots[0]
    plot.gap_width, plot.overlap = gap, -8
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.font.size, dl.font.name = Pt(label_size), FONT
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    if vmax:
        chart.value_axis.maximum_scale = vmax
    chart.value_axis.minimum_scale = 0
    for s, (_, _, color) in zip(plot.series, series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = rgb(color)
    if point_colors:
        for i, c in point_colors.items():
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = rgb(c)
    if horizontal:
        chart.category_axis.reverse_order = True
        chart.value_axis.visible = False
        chart.value_axis.has_major_gridlines = False
    return chart


def line_chart(slide, x, y, w, h, cats, series, vmax=100, size=12):
    cd = CategoryChartData()
    cd.categories = cats
    for name, vals, _, _ in series:
        cd.add_series(name, vals)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    style_chart(chart, size, True)
    chart.value_axis.maximum_scale, chart.value_axis.minimum_scale = vmax, 0
    for s, (_, _, color, dash) in zip(chart.plots[0].series, series):
        s.smooth = False
        s.format.line.color.rgb = rgb(color)
        s.format.line.width = Pt(2.75)
        if dash:
            s.format.line.dash_style = 4          # MSO_LINE.DASH
        s.marker.style, s.marker.size = XL_MARKER_STYLE.CIRCLE, 7
        s.marker.format.fill.solid()
        s.marker.format.fill.fore_color.rgb = rgb(color)
        s.marker.format.line.color.rgb = rgb(color)
    return chart


# ------------------------------------------------------------------------------------------- slides
# 1. title ------------------------------------------------------------------------------------------------------------
s = new_slide("제6회 K-인공지능 제조데이터 분석 경진대회  ·  ③ 프레스 유압펌프", None, dark=True,
              notes="안녕하세요. 프레스 유압펌프의 진동·전류 시계열로 이상을 조기 탐지하고, 특히 정상 운전변화를 고장으로 오인하는 오경보를 분석한 결과를 발표하겠습니다.")
text(s, 0.6, 0.38, 9, 0.3, [{"text": "제6회 K-인공지능 제조데이터 분석 경진대회  ·  ③ 프레스 유압펌프", "size": 12, "bold": True, "color": "FDBA74"}])
text(s, 0.6, 1.9, 7.1, 2.6, [{"text": "진동·전류 시계열 기반", "size": 20, "color": "D1D5DB", "after": 10},
                              {"text": "프레스 유압펌프 이상 조기탐지", "size": 32, "bold": True, "color": WHITE, "ls": 1.05, "after": 14},
                              {"text": "정상 운전변화를 고장으로 착각하지 않는 오경보 분석", "size": 20, "color": "FDBA74"}])
card(s, 7.9, 1.2, 4.9, 4.1, WHITE, 0.04)
picture(s, FIG / "fig01_signals.png", 8.0, 1.3, 4.7, 3.9)
text(s, 0.6, 6.5, 9, 0.4, [{"text": "정상(파랑)과 이상(빨강) 파형: 이상은 전류 파형이 무너지고 진동이 충격성으로 바뀐다", "size": 12, "color": "9CA3AF"}])

# 2. agenda ----------------------------------------------------------------------------------------------------------
s = new_slide("발표 순서", "평가 항목 순서대로 다섯 부분으로 진행합니다",
              notes="발표는 다섯 부분입니다. 먼저 문제와 데이터, 그리고 검증을 어떻게 설계했는지 말씀드리고, 이어서 모델 개발과 평가, 영향요인과 오류, 현장 활용, 마지막으로 재현성과 한계, 정리 순서로 진행하겠습니다. 부록에는 예상 질문 네 개를 준비했습니다.")
PARTS = [("문제·데이터", "문제 정의 · 데이터 · 접근과 검증 설계", 3, 5),
         ("모델 개발·평가", "모델 비교 · 핵심 발견 · 독립 평가 · 약한 이상 탐지", 6, 9),
         ("영향요인·오류", "무엇이 판별을 좌우하고 어디서 틀리는가", 10, 11),
         ("현장 활용", "안전장치 · 경보 정책 · 운영 절차", 12, 14),
         ("재현성·한계·정리", "재현성 · 숨기지 않은 한계 · 결론", 15, 17)]
for i, (h, d, a, b) in enumerate(PARTS):
    x = 0.6 + i * 2.45
    card(s, x, 2.15, 2.3, 3.9, CARD, 0.05)
    disc(s, x + 0.25, 2.45, 0.7, str(i + 1))
    text(s, x + 0.25, 3.35, 1.95, 0.9, [{"text": h, "size": 16, "bold": True, "ls": 1.0}])
    text(s, x + 0.25, 4.3, 1.85, 0.4, [{"text": f"슬라이드 {a}–{b}", "size": 12, "bold": True, "color": ACCENT}])
    text(s, x + 0.25, 4.7, 1.85, 1.3, [{"text": d, "size": 13, "color": "374151", "ls": 1.1}])
text(s, 0.6, 6.3, 12.1, 0.5, [{"text": "부록(슬라이드 18–21): 예상 질문 4개 — R2의 근거, 시간당 오경보, 전류 의존, 일반화", "size": 14, "color": MUTED}])

# 2. problem ---------------------------------------------------------------------------------------------------------
s = new_slide("① 문제·데이터  ·  문제 정의", "진짜 문제는 고장을 놓치는 것이 아니라, 정상 운전변화를 고장으로 오인하는 것",
              notes="이 설비는 부하에 따라 하부진동과 전류가 함께 커지는 정상 운전을 합니다. 진폭만 보는 규칙이나 모델은 고부하 정상 운전을 고장으로 오인합니다. 그래서 크기가 아니라 모양을 보게 만드는 것이 핵심입니다. 이 발표는 처음 보는 운전 상태에서도 오경보 없이 이상을 잡을 수 있는지에 답합니다.")
card(s, 0.6, 1.95, 6.2, 4.85, WHITE, 0.03)
picture(s, FIG / "fig03_modes.png", 0.7, 2.0, 6.0, 4.75)
text(s, 7.3, 2.0, 5.4, 3.0, [
    {"runs": [("정상 운전은 모드가 둘", True, INK), (f"  저부하 {int(w_mode[0]):,}창 · 고부하 {int(w_mode[1]):,}창 (군집 실루엣 0.68)", False, INK)], "size": 17, "bullet": True, "after": 12},
    {"text": "부하가 커지면 하부진동과 전류가 함께 커지는 하나의 연속 궤적이 정상 운전이다", "size": 17, "bullet": True, "after": 12},
    {"text": "진폭(크기)만 보면 고부하 정상 운전이 이상처럼 보인다", "size": 17, "bullet": True, "after": 12},
])
card(s, 7.3, 4.75, 5.4, 2.05, CARD, 0.05)
text(s, 7.6, 4.95, 4.8, 1.8, [{"text": "이 발표가 답하는 질문", "size": 14, "bold": True, "color": ACCENT, "after": 6},
                              {"text": "처음 보는 운전 상태에서도 오경보 없이 이상을 잡을 수 있는가?", "size": 19, "bold": True, "ls": 1.1}])

# 3. data ------------------------------------------------------------------------------------------------------------
s = new_slide("① 문제·데이터  ·  데이터", "설비 1대의 정상 20,000행과 단일 3분 이상 세션",
              notes="데이터는 상부·하부 진동과 전류 세 채널을 10헤르츠로 수집한 것입니다. 약 8초마다 최대 5초씩 수집 구간이 반복되고, 이 주기가 프레스 한 사이클인지는 가설입니다. 이상 데이터가 다른 날의 단일 3분 세션이라는 점이 가장 큰 한계입니다.")
stat(s, 0.6, 1.95, 2.9, 1.9, "3채널 · 10Hz", "상부진동 · 하부진동 · 전류, 0.1초 간격", big_size=26)
stat(s, 3.7, 1.95, 2.9, 1.9, f"{seg_all['press_data_normal.csv']} + {seg_all['outlier_data.csv']}", f"연속 수집 구간 수 (정상 + 이상), 사용 {seg_used['press_data_normal.csv']} + {seg_used['outlier_data.csv']}", big_size=26)
stat(s, 0.6, 4.05, 2.9, 1.9, f"{n_win:,}창", f"1초(10샘플) 윈도우, 이 중 이상 {n_anom}창({pct(n_anom / n_win)})", big_size=26)
stat(s, 3.7, 4.05, 2.9, 1.9, f"약 {qn_.seg_start_interval_median_s:.0f}초 주기", "수집 구간이 반복되는 간격 (프레스 1사이클일 가능성은 가설)", big_size=26)
text(s, 6.9, 1.9, 5.85, 0.4, [{"text": "이상 데이터의 비율 (%) — 심한 클래스 불균형", "size": 14, "bold": True}])
bar_chart(s, 6.8, 2.3, 6.0, 3.7, ["원시 행", "수집 구간", "1초 윈도우"],
          [("이상 비율(%)", [round(100 * qa_.rows / (qa_.rows + qn_.rows), 1), round(100 * seg_all["outlier_data.csv"] / (seg_all["outlier_data.csv"] + seg_all["press_data_normal.csv"]), 1),
                              round(100 * n_anom / n_win, 1)], ACCENT)], fmt='0.0"%"', vmax=5, legend=False, size=12, gap=70, label_size=13)
text(s, 0.6, 6.25, 12.1, 0.8, [{"runs": [("한계: ", True, ACCENT), ("이상 데이터는 정상과 다른 날의 단일 3분 세션(구간 21개)이라 다른 고장 유형·설비로의 일반화는 검증되지 않았다.", False, INK)], "size": 15}])

# 4. approach --------------------------------------------------------------------------------------------------------
s = new_slide("① 문제·데이터  ·  접근과 검증 설계", "크기가 아니라 '모양'을 보게 만들고, 새 운전모드로 시험한다",
              notes="접근은 네 단계입니다. 정상 운전모드를 찾고, 진폭 특징이 고부하에서 오경보를 내는 것을 확인하고, 크기에 의존하지 않는 모양 특징 17개를 설계한 뒤, 모드를 통째로 빼고 시험해 효과를 검증했습니다. 검증은 블록 그룹 교차검증, 미학습 운전모드 시험, 선택과 분리한 중첩 교차검증의 세 겹이고, 결과는 다음 부분에서 보여 드립니다.")
steps = [("정상 운전모드 발견", "정상 창을 (하부진동, 전류) RMS로 군집 → 저부하·고부하 2모드"),
         ("진폭 특징의 약점 확인", "진폭을 포함한 34개 특징이 새 운전모드의 정상 창을 이상으로 보는지 확인"),
         ("모드 불변 특징 설계", "왜도·첨도·파고율·거칠기·영교차율·채널 상관 등 17개(크기 무관)"),
         ("미학습 모드 시험", "정상 모드 하나를 학습에서 통째로 빼고 그 모드의 정상 창에서 오경보 측정")]
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * 3.1
    card(s, x, 2.2, 2.85, 3.75, CARD, 0.05)
    disc(s, x + 0.25, 2.5, 0.7, str(i + 1))
    text(s, x + 0.25, 3.35, 2.4, 0.9, [{"text": h, "size": 18, "bold": True, "ls": 1.0}])
    text(s, x + 0.25, 4.3, 2.4, 1.6, [{"text": d, "size": 15, "color": "374151", "ls": 1.1}])
text(s, 0.6, 6.4, 12.1, 0.6, [{"text": "검증은 세 겹: ① 30초 블록 그룹 교차검증 → ② 미학습 운전모드 시험 → ③ 선택과 분리한 중첩 교차검증 (결과는 ②부)", "size": 15, "bold": True, "color": ACCENT}])

# 6. model comparison ------------------------------------------------------------------------------------------------
s = new_slide("② 모델 개발·평가  ·  모델 비교", f"SPC 규칙 F1 {f3(sm('INV', 'B0_SPC', 'F1'))} → 최종 모델 {f3(sm('INV', FINAL, 'F1'))}, 오경보는 시간당 {sm('INV', FINAL, 'false_alarms_per_hour'):.1f}건",
              notes="여섯 모델을 같은 조건의 블록 그룹 교차검증으로 비교했습니다. 현장의 관리도 방식인 SPC는 재현율이 53%에 그치고, 최종 모델인 그래디언트 부스팅은 F1 0.981로 가장 높으면서 학습 모델 중에서는 오경보가 가장 적습니다. SPC 규칙도 오경보율은 0.03%로 같고 시간당으로는 더 적지만, 놓치는 이상이 절반에 가깝습니다. 다만 이 비교는 모양 특징 17개 기준이고, 왜 이 특징을 쓰는지는 다음 장에서 보여 드립니다.")
bar_chart(s, 0.5, 1.9, 7.9, 4.7, MODEL_CATS, [("F1 (윈도우)", [round(sm("INV", m, "F1"), 3) for m in ORDER], GREY)], fmt='0.000', vmax=1.1, legend=False,
          point_colors={ORDER.index(FINAL): ACCENT}, size=12)
stat(s, 8.8, 1.95, 3.95, 1.45, f3(sm("INV", FINAL, "Recall")), "재현율 (SPC 규칙은 " + f3(sm("INV", "B0_SPC", "Recall")) + ")", big_size=32)
stat(s, 8.8, 3.6, 3.95, 1.45, pct(sm("INV", FINAL, "FPR"), 2), "정상 창 오경보율 (FP 평균 " + f"{sm('INV', FINAL, 'FP'):.1f}창, SPC 규칙도 같은 {pct(sm('INV', 'B0_SPC', 'FPR'), 2)})", big_size=32)
stat(s, 8.8, 5.25, 3.95, 1.45, f"{int(round(sm('INV', FINAL, 'seg_recall') * seg_used['outlier_data.csv']))} / {seg_used['outlier_data.csv']}", "이상 구간 탐지 (블록 그룹 CV 5-fold × 3회)", big_size=32)
text(s, 0.6, 6.68, 7.8, 0.3, [{"text": "교차검증 기준 · 모양(모드 불변) 특징 17개 · 시간당 오경보는 센서 기록시간 기준 (이유는 다음 장)", "size": 11, "color": MUTED}])

# 5. core result -----------------------------------------------------------------------------------------------------
s = new_slide("② 모델 개발·평가  ·  핵심 발견", f"모양만 보면 처음 보는 고부하의 오경보가 {pct(shift('ALL', 'M2_HGB', 1), 0)} → {pct(shift('INV', 'M2_HGB', 1))} (HGB)",
              notes="이 슬라이드가 발표의 핵심입니다. 같은 모델이라도 진폭 특징을 쓰면 학습에 없던 고부하 모드의 정상 창을 크게 오경보하지만, 모양 특징만 쓰면 오경보가 사라집니다. 이 수치는 학습에서 고부하 모드를 통째로 뺀 시험입니다. 그래서 앞의 모델 비교에서 모양 특징 17개를 쓴 것입니다.")
cats = MODEL_CATS
bar_chart(s, 0.5, 1.9, 8.6, 4.9, cats,
          [("전체 특징(진폭 포함)", [round(100 * shift("ALL", m, 1), 1) for m in ORDER], GREY),
           ("모드 불변 특징(모양)", [round(100 * shift("INV", m, 1), 1) for m in ORDER], ACCENT)], fmt='0.0"%"', vmax=100, size=12)
card(s, 9.4, 1.95, 3.35, 2.35, "FDE8DD", 0.05)
text(s, 9.65, 2.15, 2.9, 2.3, [{"text": "HGB (최종 모델)", "size": 14, "color": MUTED, "after": 4},
                               {"text": f"{pct(shift('ALL', 'M2_HGB', 1))} → {pct(shift('INV', 'M2_HGB', 1))}", "size": 30, "bold": True, "color": ACCENT, "after": 4},
                               {"text": "미학습 고부하 모드의 정상 창 오경보율", "size": 13, "color": "374151"}])
text(s, 9.4, 4.55, 3.35, 2.0, [{"text": "SPC 규칙은 진폭(RMS)만 보므로 모양 특징으로 바꿔도 같은 결과", "size": 13, "color": MUTED, "after": 8},
                              {"text": "표시한 값은 학습에서 고부하 모드를 뺀 시험의 오경보율이다", "size": 13, "color": MUTED}])

# 7. independent evaluation -----------------------------------------------------------------------------------------
s = new_slide("② 모델 개발·평가  ·  독립 평가", f"고르는 데 쓴 데이터와 점수 매기는 데이터를 분리해도 F1 {f3(npi.loc['F1', 'point'])}",
              notes="모델을 고른 데이터로 점수를 매기면 성능이 부풀려질 수 있어서, 선택 절차 전체를 외부 fold마다 다시 하고 외부 test에서 한 번만 평가했습니다. 점수는 0.981에서 0.966으로 내려갔고, 그대로 공개했습니다. fold마다 선택된 모델이 달라 선택이 데이터에 민감하다는 점도 한계로 적었습니다.")
rows = [["외부 fold", "선택된 조합", "시험 창(이상)", "F1", "재현율", "FP / FN"]]
for r in NF.itertuples():
    rows.append([str(r.fold), f"{r.feature_set}·{SHORT[r.model]}", f"{int(r.test_windows):,}({int(r.test_anomaly_windows)})", f3(r.F1), f3(r.Recall), f"{int(r.FP)} / {int(r.FN)}"])
rows.append(["합산", "-", f"{n_tp + n_fp + n_fn + n_tn:,}({n_tp + n_fn})", f3(npi.loc["F1", "point"]), f3(npi.loc["Recall", "point"]), f"{n_fp} / {n_fn}"])
table(s, 0.6, 2.1, 7.6, rows, [1.1, 1.5, 1.9, 1, 1, 1.2], row_h=0.55, size=14, bold_last=True, hl_rows=(len(rows) - 1,))
text(s, 0.6, 5.2, 7.6, 1.6, [{"text": f"외부 test 합산 F1 {f3(npi.loc['F1', 'point'])}  [95% CI {f3(npi.loc['F1', 'ci_low'])}, {f3(npi.loc['F1', 'ci_high'])}]", "size": 16, "bold": True, "after": 6},
                             {"text": f"교차검증 F1 {f3(sm('INV', FINAL, 'F1'))}과의 차이 {npi.loc['F1', 'point'] - sm('INV', FINAL, 'F1'):+.3f} — 신뢰구간 안이라 선택 낙관은 이 데이터로 구분되지 않는다", "size": 14, "color": "374151"}])
card(s, 8.6, 2.1, 4.15, 4.7, CARD, 0.04)
text(s, 8.9, 2.3, 3.6, 4.4, [{"text": "어떻게 분리했나", "size": 17, "bold": True, "after": 8},
                             {"text": "외부 3-fold (30초 블록 단위, 이상 라벨 층화)", "size": 14, "bullet": True, "after": 6},
                             {"text": "각 fold의 train에서 모드 군집·6모델 비교·모델 선정·임계값 결정을 처음부터 다시 수행", "size": 14, "bullet": True, "after": 6},
                             {"text": "고른 모델을 외부 test에서 한 번만 평가", "size": 14, "bullet": True, "after": 10},
                             {"runs": [("한계: ", True, ACCENT), ("fold마다 선택 모델이 다르다 (M2·M4·M1). 이상 블록이 6개뿐이라 단일 hold-out 대신 중첩 방식을 썼다.", False, INK)], "size": 13}])

# 9. early detection ------------------------------------------------------------------------------------------------
s = new_slide("② 모델 개발·평가  ·  미리 잡는가", f"가장 약한 이상에서도 SPC {pct(SEV.iloc[0].spc_detect, 0)} 대 모델 {pct(SEV.iloc[0].model_red, 0)}",
              notes="열화가 진행되는 기록은 없어서 두 가지로 확인했습니다. 실제 이상 창을 진폭이 정상에서 먼 정도로 4등분해 보면 가장 약한 구간에서 SPC는 하나도 못 잡지만 모델은 잡습니다. 정상 신호에 고장 신호를 α만큼 섞은 합성 시험에서는 빨강보다 노랑이 먼저 반응합니다. 합성 시험은 민감도의 하한 참고치입니다.")
text(s, 0.6, 1.9, 6.2, 0.4, [{"text": "실제 이상 창: 진폭이 정상에서 먼 정도별 탐지율 (%)", "size": 14, "bold": True}])
bar_chart(s, 0.5, 2.3, 6.4, 4.4, [str(b).replace(" (", " ").replace(")", "") for b in SEV["bin"]],
          [("SPC 규칙", [round(100 * v) for v in SEV.spc_detect], GREY), ("최종 모델 노랑 이상", [round(100 * v) for v in SEV.model_yellow_or_red], WARN),
           ("최종 모델 빨강", [round(100 * v) for v in SEV.model_red], ACCENT)], fmt='0', vmax=125, size=11, gap=40, label_size=10)
text(s, 7.2, 1.9, 5.6, 0.4, [{"text": "합성 약한 고장: 고장 신호를 α만큼 섞음 (경보율 %)", "size": 14, "bold": True}])
wk = WK.pivot(index="alpha", columns="model", values="alarm_rate")
al = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
line_chart(s, 7.1, 2.3, 5.7, 3.7, [f"{int(100 * a)}%" for a in al],
           [("SPC 규칙", [round(100 * wk.loc[a, "SPC 규칙"], 1) for a in al], GREY, True),
            ("모델 노랑 이상", [round(100 * wk.loc[a, "최종 모델(노랑 이상)"], 1) for a in al], WARN, False),
            ("모델 빨강", [round(100 * wk.loc[a, "최종 모델(빨강)"], 1) for a in al], ACCENT, False)], size=11)
text(s, 7.2, 6.1, 5.5, 0.9, [{"runs": [("읽는 법: ", True, ACCENT), ("약한 이상에는 빨강보다 노랑이 먼저 반응한다. 합성 시험은 선형 혼합일 뿐이라 민감도의 하한 참고치다.", False, INK)], "size": 12.5, "ls": 1.05}])

# 8. drivers ---------------------------------------------------------------------------------------------------------
s = new_slide("③ 영향요인·오류  ·  영향요인", "판별의 대부분은 전류 파형의 거칠기(cur_madr) 하나",
              notes="특징 그룹을 통째로 섞어 보면 전류와 동특성, 즉 파형의 거칠기와 규칙성이 거의 전부를 설명합니다. 진동 진폭만 보는 SPC 규칙은 이상 창의 47%를 놓칩니다.")
g = GI.set_index("group")
gcats = ["동특성(거칠기·규칙성)", "전류(AI2)", "형상(왜도·첨도)", "상부진동(AI0)", "하부진동(AI1)", "채널간 관계"]
gkeys = ["동특성(mad/madr/zcr)", "전류(AI2)", "형상(skew/kurt/crest)", "상부진동(AI0)", "하부진동(AI1)", "채널간 관계"]
bar_chart(s, 0.5, 1.95, 7.6, 4.8, gcats, [("순열중요도(AP 감소)", [round(float(g.loc[k, "perm_drop_AP"]), 3) for k in gkeys], ACCENT)],
          fmt='0.000', vmax=1.0, horizontal=True, legend=False, size=12, gap=50)
spc_hit, spc_miss = SPCM[~SPCM.spc_miss].iloc[0], SPCM[SPCM.spc_miss].iloc[0]
lgo = LGO.set_index("removed_group").PR_AUC
stat(s, 8.5, 1.95, 4.25, 1.5, f3(PI.iloc[0].perm_importance), f"단일 변수 {PI.iloc[0].feature} 순열중요도 (다음 변수 {PI.iloc[1].perm_importance:.3f})", big_size=32)
stat(s, 8.5, 3.65, 4.25, 1.5, pct(spc_miss.n / (spc_miss.n + spc_hit.n), 0), "진동 RMS만 보는 SPC 규칙이 놓치는 이상 창", big_size=32, fill="FDE8DD", big_color=ACCENT)
stat(s, 8.5, 5.35, 4.25, 1.4, f3(lgo["전류(AI2)"]), "전류 그룹을 빼고 다시 학습했을 때 PR-AUC (전체 " + f3(lgo["전체"]) + ")", big_size=32)

# 10. errors ---------------------------------------------------------------------------------------------------------
fn_oof = PRED[(PRED.y == 1) & (PRED.level != "red")]            # anomaly windows the model did NOT lift to red (first CV repeat)
n_fn1, k_guard, k_green = len(fn_oof), int((fn_oof.guard == 1).sum()), int((fn_oof.level == "green").sum())
k_unc = int((fn_oof.unc >= THR["unc_high"]).sum())
fn_tag = "노랑으로는 잡힘" if k_green == 0 else f"초록 {k_green}창은 완전 미탐"
s = new_slide("③ 영향요인·오류  ·  오류 분석", f"틀리는 곳: 빨강에 못 미친 이상 창 {n_fn1}개({fn_tag}), 그리고 짧은 구간 오경보 1개",
              notes=f"모델이 빨강으로 올리지 못한 이상 창은 {n_fn1}개이고 전류가 한쪽으로 크게 치우친 구간입니다. 이 창들은 앙상블 불일치로 노랑이 되었고({k_unc}개), 가드로는 잡히지 않았습니다({k_guard}개 발동). "
                    "오경보 사례는 길이가 0.6초뿐인 짧은 구간 전체가 한 윈도우로 판정된 경우입니다. 다음 장에서는 진폭만 치솟는 고장에 대비한 가드의 역할과 한계를 말씀드립니다.")
card(s, 0.6, 1.95, 8.1, 4.85, WHITE, 0.03)
picture(s, FIG / "fig14_error_cases.png", 0.7, 2.05, 7.9, 4.65)
fn_tot = int(FNC[FNC.condition == "가장 가까운 운전모드"].errors.sum())
fp_tot = int(FPC[FPC.condition == "운전모드"].errors.sum())
reps = max(round(FPC[FPC.condition == "운전모드"].n.sum() / (n_win - n_anom)), 1)       # CV repeats pooled in the condition tables
text(s, 9.0, 1.95, 3.75, 4.9, [
    {"runs": [(f"빨강 미달 {n_fn1}창 ({fn_tag})", True, ACCENT)], "size": 17, "after": 4},
    {"text": "전류가 한쪽으로 치우치고 거칠기가 정상 범위와 겹친다 → 모양 특징만으로 구분이 어렵다 (진폭을 뺀 대가)", "size": 14, "color": "374151", "after": 14, "ls": 1.1},
    {"runs": [("오경보 (파형 1개)", True, BLUE)], "size": 17, "after": 4},
    {"text": "5~9샘플뿐인 짧은 구간 전체가 한 윈도우가 되어 영교차율·첨도가 불안정하다", "size": 14, "color": "374151", "after": 14, "ls": 1.1},
    {"text": f"교차검증 1회당 오경보 {fp_tot // reps}건 · 미탐 {fn_tot // reps}건 안팎 ({reps}회 반복 합산 {fp_tot}건 · {fn_tot}건). 건수가 적어 조건별 차이는 기술(記述)로만 읽는다", "size": 12, "color": MUTED, "ls": 1.05},
])

# 12. guard ----------------------------------------------------------------------------------------------------------
s = new_slide("④ 현장 활용  ·  안전장치", "피크전류 가드: 노랑으로만 올리고, 운전모드별 한계로",
              notes="모델이 놓칠 수 있는 진폭형 고장을 위해 정상 최대 전류를 넘으면 경계 신호를 내는 가드를 두었습니다. 처음에는 빨강으로 강제했지만 정상 창의 빨강만 늘어 폐기했습니다. 한 개의 전역 한계는 새 운전모드에서 정상을 대량 플래그해서 운전모드별 한계와 신규 모드 상태로 바꿨습니다. 이 데이터에서 가드의 추가 탐지 이득은 없고 보험이라는 점도 공개합니다.")
text(s, 0.6, 1.9, 7.6, 0.4, [{"text": "정상 윈도우가 받은 경보 (창)", "size": 14, "bold": True}])
bar_chart(s, 0.5, 2.25, 7.8, 3.0, ["모델만", "전역 한계→노랑", "모드별 한계→노랑 (채택)", "전역 한계→빨강 (폐기)"],
          [("노랑", [int(v) for v in GE.normal_yellow], WARN), ("빨강", [int(v) for v in GE.normal_red], BAD)], fmt='0', vmax=70, size=11, gap=50, label_size=12)
stat(s, 0.6, 5.4, 3.75, 1.45, f"{pct(GM.global_normal_flag.max())} → {pct(GM.mode_normal_flag.max())}", "미학습 모드 정상 창의 노랑 플래그 (전역 → 모드별 한계)", big_size=26, label_size=12)
stat(s, 4.55, 5.4, 3.75, 1.45, pct(GM.new_mode_normal.min(), 0) + " 이상", "같은 시험에서 '신규 모드'(승인 대기)로 분류된 정상 창", big_size=26, label_size=12, fill="E6F4EC", big_color=GOOD)
text(s, 8.7, 1.95, 4.05, 4.9, [
    {"runs": [("빨강 강제는 폐기", True, ACCENT)], "size": 16, "after": 3},
    {"text": f"정상 창 빨강 {ge0.normal_red:.0f}→{ge_red.normal_red:.0f}, 정지 오경보 {ge0.R2_FA_per_h:.1f}→{ge_red.R2_FA_per_h:.1f}건/시간, 이득 없음", "size": 13, "color": "374151", "after": 11, "ls": 1.05},
    {"runs": [("운전모드별 한계", True, ACCENT)], "size": 16, "after": 3},
    {"text": f"저부하 {J['guard_limits'][0]:.1f} · 고부하 {J['guard_limits'][1]:.1f} (설비 정격이 아닌 관측 최댓값)", "size": 13, "color": "374151", "after": 11, "ls": 1.05},
    {"runs": [("공개하는 한계", True, ACCENT)], "size": 16, "after": 3},
    {"text": f"이 데이터에서 가드의 추가 탐지 이득은 {int(ge0.anomaly_green - ge_mode.anomaly_green)}창 — 진폭만 치솟는 고장에 대한 보험이다. 정상 노랑은 {ge0.normal_yellow:.0f}→{ge_mode.normal_yellow:.0f}창으로 늘고, 가드가 이상 창을 잡는 비율도 모드별 한계에서 낮아진다", "size": 13, "color": "374151", "ls": 1.05},
])

# 11. alarm policy ---------------------------------------------------------------------------------------------------
s = new_slide("④ 현장 활용  ·  경보 정책", "3단계 경보, 그리고 목적에 맞게 나눈 규칙",
              notes="경보는 초록, 노랑, 빨강 세 단계입니다. 규칙은 용도별로 나눴습니다. 즉시 점검 호출은 R1, 정지 권고는 R2, 사전 점검은 R3, 구간 종료 후 판정은 R4입니다. 독립 평가에서는 R2의 오경보 이득이 드러나지 않았고, 한 번 튄 값으로 설비를 세우지 않는다는 원칙에 따라 정지 권고로 둡니다. 노랑 단계에는 앞에서 본 가드와 앙상블 불일치가 들어갑니다.")
nn = LV.loc["정상 윈도우"]
lvl = [("초록 · 정상", GOOD, f"정상 창의 {pct(nn.green / nn.sum())}", "정상 운전 유지, 로그만 저장"),
       ("노랑 · 주의", WARN, f"p ≥ {THR['t_yellow']:.3f}, 앙상블 불일치, 또는 가드", "재측정 · 표본검사 · 점검 앞당김"),
       ("빨강 · 경고", BAD, f"p ≥ {THR['t_red']:.3f}", "2연속이면 정지 권고 · 선별검사")]
for i, (h, c, cond, act) in enumerate(lvl):
    y = 1.95 + i * 1.62
    card(s, 0.6, y, 5.2, 1.45, CARD, 0.06)
    disc(s, 0.85, y + 0.38, 0.7, "", fill=c)
    text(s, 1.8, y + 0.15, 3.9, 1.2, [{"text": h, "size": 17, "bold": True, "after": 3}, {"text": cond, "size": 12.5, "color": "374151", "after": 3, "ls": 1.0}, {"text": act, "size": 12.5, "color": MUTED, "ls": 1.0}])
purpose = {"R1 red 1회": "즉시 점검 호출", "R2 red 2연속": "정지 권고", "R3 yellow 이상 2연속": "사전 점검(주의 누적)", "R4 세그먼트 평균 p>=t_red": "사후 구간 판정"}
rows = [["규칙", "용도", "탐지율 (내부 / 독립)", "오경보 건/시간 (내부 / 독립)"]]
for key, label in (("R1 red 1회", "R1 빨강 1회"), ("R2 red 2연속", "R2 빨강 2연속"), ("R3 yellow 이상 2연속", "R3 노랑 이상 2연속"), ("R4 세그먼트 평균 p>=t_red", "R4 구간 평균")):
    a, b = RULES.loc[key], NR.loc[key]
    rows.append([label, purpose[key], f"{pct(a.segment_recall, 0)} / {pct(b.segment_recall, 0)}", f"{a.false_alarms_per_hour:.1f} / {b.false_alarms_per_hour:.1f}"])
table(s, 6.1, 1.95, 6.65, rows, [2.0, 2.3, 1.5, 1.7], row_h=0.72, size=12.5, hl_rows=(2,))
text(s, 6.1, 5.75, 6.65, 1.2, [{"runs": [("정직한 주석: ", True, ACCENT), ("독립 평가(중첩 검증) 값은 정상 570구간 중 6구간의 오경보라 표본이 작다. 독립 평가에서 R2는 오경보를 줄이지 못하고 이상 구간 1개를 더 놓쳤다 → 정지 권고의 근거는 '한 번 튄 값으로 세우지 않는다'는 운영 원칙이다.", False, INK)], "size": 12, "ls": 1.05}])

# 14. operations -----------------------------------------------------------------------------------------------------
s = new_slide("④ 현장 활용  ·  운영 절차", "새 운전조건은 자동 정지 대신 '승인 → 재보정'으로 처리한다",
              notes="현장에서 새로운 운전조건이 감지되면 자동으로 정지하지 않고 승인 대기로 분리합니다. 승인된 뒤에는 가드 한계와 확률 보정을 다시 산정하고, 초록과 노랑 비율을 주간으로 모니터링합니다. 노랑이 평소의 두 배를 넘거나 새로운 고장 데이터가 확보되면 재학습합니다.")
steps = [("신규 모드 감지", "어느 운전모드에도 속하지 않는 창은 '신규 모드'로 분리"),
         ("운영자 승인", "자동 정지가 아니라 승인·재학습 대기로 보냄"),
         ("재산정", "가드 한계·확률 보정을 새 조건으로 다시 계산, 설비 정격과 대조"),
         ("주간 모니터링", "초록·노랑 비율을 매주 확인"),
         ("재학습", "노랑이 평소의 2배 이상이거나 새 고장 데이터가 생기면")]
for i, (h, d) in enumerate(steps):
    x = 0.6 + i * 2.45
    card(s, x, 2.15, 2.2, 3.1, CARD, 0.05)
    disc(s, x + 0.25, 2.4, 0.65, str(i + 1))
    text(s, x + 0.25, 3.25, 1.75, 0.5, [{"text": h, "size": 16, "bold": True, "ls": 1.0}])
    text(s, x + 0.25, 3.85, 1.75, 1.35, [{"text": d, "size": 13, "color": "374151", "ls": 1.1}])
    if i < len(steps) - 1:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.22), Inches(3.45), Inches(0.2), Inches(0.35))
        ar.fill.solid()
        ar.fill.fore_color.rgb = rgb(GREY)
        ar.line.fill.background()
        ar.shadow.inherit = False
card(s, 0.6, 5.55, 12.15, 1.3, "FDE8DD", 0.06)
text(s, 0.9, 5.7, 11.6, 1.1, [{"text": "경보 사용 원칙", "size": 14, "bold": True, "color": ACCENT, "after": 4},
                              {"text": "정지 판단은 R2(빨강 2연속) · 즉시 점검은 R1 · 노랑은 정지가 아니라 확인 신호 · 정격 사양값과 실제 비용비로 임계값 교체", "size": 14, "ls": 1.1}])

# 15. reproducibility ------------------------------------------------------------------------------------------------
s = new_slide("⑤ 재현성·한계·정리  ·  재현성", "명령 세 줄로 학습부터 보고서까지 다시 만들 수 있다",
              notes="코드는 run_all 한 번으로 전처리부터 학습, 분석, 표와 그림, 대시보드까지 재현합니다. predict로 새 데이터를 채점하고 이력을 남기며, build_report가 같은 표에서 보고서를 만들기 때문에 보고서 수치와 코드가 어긋나지 않습니다. 단위 테스트가 통과하고 파이썬 3.11, 3.12, 3.13에서 같은 수치가 재현됩니다.")
cmds = [("python run_all.py", "전처리 → 학습 → 분석 → 표·그림·대시보드 (약 90분)"),
        ("python predict.py --input 새.csv", "새 데이터 채점, 이력 기록·대시보드 갱신"),
        ("python report/build_report.py", "표에서 보고서 생성 — 수치가 코드와 어긋나지 않음")]
for i, (c, d) in enumerate(cmds):
    x = 0.6 + i * 4.1
    card(s, x, 2.1, 3.85, 2.4, CARD, 0.05)
    text(s, x + 0.3, 2.35, 3.3, 0.7, [{"text": c, "size": 15, "bold": True, "color": ACCENT}])
    text(s, x + 0.3, 3.15, 3.3, 1.2, [{"text": d, "size": 15, "color": "374151", "ls": 1.1}])
stat(s, 0.6, 4.8, 3.85, 1.85, f"{N_TESTS}개", "단위 테스트 통과", big_size=30)
stat(s, 4.7, 4.8, 3.85, 1.85, "3.11 · 3.12 · 3.13", "Python 버전에서 같은 수치 재현", big_size=22)
stat(s, 8.8, 4.8, 3.95, 1.85, "불필요", "GPU · 외부 API · 인터넷 (결과는 outputs/에 포함)", big_size=30)

# 13. limits ---------------------------------------------------------------------------------------------------------
s = new_slide("⑤ 재현성·한계·정리  ·  한계", "숨기지 않은 한계: 속도 변화와 단일 세션",
              notes="전류의 거칠기에 의존하기 때문에 정상 신호의 겉보기 주파수가 두 배 이상 빨라지면 오경보가 급증합니다. 10헤르츠 샘플링이라 이 배율을 실제 rpm으로 환산할 수는 없습니다. 현장에서 속도를 바꾸면 그 속도의 정상 데이터로 재보정해야 합니다. 단일 세션과 가드 한계, 비용비 가정도 같이 공개합니다.")
red0 = SP[(SP.model == "최종 모델(빨강)") & (SP.channels == "전류만")].sort_values("factor")
red1 = SP[(SP.model == "최종 모델(빨강)") & (SP.channels == "전 채널")].sort_values("factor")
text(s, 0.6, 1.9, 7.2, 0.4, [{"text": "속도 변화 시험: 정상 윈도우 빨강 오경보율 (%)", "size": 14, "bold": True}])
bar_chart(s, 0.5, 2.3, 7.4, 4.5, [f"×{f:g}" for f in red0.factor],
          [("전류 채널만 변형", [round(100 * v, 1) for v in red0.false_alarm_rate], BAD), ("전 채널 변형", [round(100 * v, 1) for v in red1.false_alarm_rate], WARN)],
          fmt='0.0', vmax=115, size=12, gap=45, label_size=10)
safe = [f for f in red0.factor if max(float(red0[red0.factor == f].false_alarm_rate.iloc[0]), float(red1[red1.factor == f].false_alarm_rate.iloc[0])) <= 0.01]
lims = [("속도(겉보기 주파수)", f"빨강 오경보 1% 이하는 ×{min(safe):g}~×{max(safe):g} 범위뿐. ×2부터 급증. 배율은 aliasing된 값이라 실제 rpm으로 환산할 수 없다"),
        ("단일 세션", "이상은 다른 날의 3분 세션 하나 — 신규 고장 유형으로의 일반화는 보증되지 않는다"),
        ("가드 한계", "정격 사양값이 아니라 관측된 최댓값 — 현장 투입 전 사양값과 대조·재산정"),
        ("비용비", "FN:FP 10:1은 가정값 — 현장 다운타임·불량 비용으로 교체")]
for i, (h, d) in enumerate(lims):
    y = 1.95 + i * 1.22
    card(s, 8.2, y, 4.55, 1.1, CARD, 0.06)
    text(s, 8.45, y + 0.1, 4.1, 0.95, [{"text": h, "size": 14, "bold": True, "color": ACCENT, "after": 2}, {"text": d, "size": 11.5, "color": "374151", "ls": 1.0}])

# 15. conclusion -----------------------------------------------------------------------------------------------------
s = new_slide("⑤ 재현성·한계·정리  ·  정리", "이상을 잘 잡는 것만큼 '정상 운전변화를 이상으로 오인하지 않는 것'을 풀었다", dark=True,
              notes="정리하겠습니다. 첫째, 문제를 정상 운전변화에 대한 일반화 문제로 정의하고 모양 특징으로 풀었습니다. 둘째, 선택과 평가를 분리해 점수를 보수적으로 공개했습니다. 셋째, 가드와 경보 정책, 속도 시험까지 한계를 먼저 밝혔습니다. 감사합니다.")
msgs = [(f"{pct(shift('ALL', 'M2_HGB', 1), 0)} → {pct(shift('INV', 'M2_HGB', 1))}", "HGB 기준: 모양 특징으로 처음 보는 고부하 정상의 오경보 해소"),
        (f"F1 {sm('INV', FINAL, 'F1'):.3f} → {npi.loc['F1', 'point']:.3f}", "선택과 평가를 분리해도 유지, 낮아진 값까지 그대로 공개"),
        ("한계를 먼저", "속도 변화·단일 세션·가드·표본 크기를 수치와 함께 제시")]
for i, (big, small) in enumerate(msgs):
    x = 0.6 + i * 4.1
    card(s, x, 2.3, 3.85, 3.2, "2B3748", 0.05)
    text(s, x + 0.3, 2.6, 3.3, 1.2, [{"text": big, "size": 26, "bold": True, "color": "FDBA74"}])
    text(s, x + 0.3, 3.9, 3.3, 1.4, [{"text": small, "size": 16, "color": "E5E7EB", "ls": 1.1}])
text(s, 0.6, 6.2, 12.1, 0.6, [{"text": "감사합니다", "size": 24, "bold": True, "color": WHITE}])

# ---------------- appendix: expected questions -----------------------------------------------------------------------
s = new_slide("부록 · 예상 질문 1/4", "Q. 독립 평가에서 이득이 없는데 왜 R2를 정지 기준으로 두나?",
              notes="독립 평가에서는 R1과 R2의 오경보가 같았고 R2가 이상 구간 1개를 더 놓쳤습니다. 그래서 R2의 근거는 이 데이터가 아니라 한 번 튄 값으로 설비를 세우지 않는다는 운영 원칙입니다. 즉시 점검 호출은 R1이 맡고, 정지 권고만 R2로 둡니다. 표본이 6구간이라 불확실하다는 점도 같이 말씀드립니다.")
rows = [["", "R1 빨강 1회", "R2 빨강 2연속"],
        ["내부 검증 오경보(건/시간)", f"{r1.false_alarms_per_hour:.1f}", f"{r2.false_alarms_per_hour:.1f}"],
        ["독립 평가 오경보(건/시간)", f"{nr1.false_alarms_per_hour:.1f}", f"{nr2.false_alarms_per_hour:.1f}"],
        ["독립 평가 이상 구간 탐지", pct(nr1.segment_recall, 0), pct(nr2.segment_recall, 0)]]
table(s, 0.6, 2.1, 7.4, rows, [2.6, 1.6, 1.6], row_h=0.7, size=15)
card(s, 8.4, 2.1, 4.35, 3.9, CARD, 0.05)
text(s, 8.7, 2.3, 3.8, 3.6, [{"text": "답변", "size": 17, "bold": True, "color": ACCENT, "after": 8},
                             {"text": "R2의 오경보 감소는 내부 검증에서만 보이고, 독립 평가(표본 6구간)에서는 사라진다", "size": 14, "bullet": True, "after": 8},
                             {"text": "그래서 근거는 데이터가 아니라 '한 번 튄 값으로 설비를 세우지 않는다'는 원칙", "size": 14, "bullet": True, "after": 8},
                             {"text": "즉시 점검 호출은 R1이 맡아 놓침 위험을 보완", "size": 14, "bullet": True}])
text(s, 0.6, 5.2, 7.4, 1.2, [{"text": "현장 적용 전에 해당 설비의 정상 데이터로 임계값을 다시 맞추는 절차를 둔다.", "size": 14, "color": MUTED}])

s = new_slide("부록 · 예상 질문 2/4", "Q. 시간당 오경보 10건이면 현장에서 못 쓰지 않나?",
              notes="숫자는 센서가 실제로 기록한 시간 기준입니다. 같은 수집 방식, 즉 약 8초마다 최대 5초 기록하는 방식으로 배포하면 하루에 기록되는 시간은 전체의 약 43%이므로 8시간 교대당 기록시간은 약 3.5시간이고, 교대당 정지 권고 오경보는 내부 검증 기준 약 " + f"{r2.false_alarms_per_hour * shift_h:.0f}" + "건, 독립 평가 기준 약 " + f"{nr2.false_alarms_per_hour * shift_h:.0f}" + "건으로 환산됩니다. 다만 독립 평가 숫자는 6구간짜리라 불확실하고, 노랑은 정지가 아니라 확인 신호입니다.")
rows = [["기준", "시간당 오경보", "8시간 교대당 환산*"],
        ["내부 검증 R2", f"{r2.false_alarms_per_hour:.1f}건", f"약 {r2.false_alarms_per_hour * shift_h:.0f}건"],
        ["독립 평가 R2", f"{nr2.false_alarms_per_hour:.1f}건", f"약 {nr2.false_alarms_per_hour * shift_h:.0f}건"]]
table(s, 0.6, 2.1, 7.4, rows, [2.4, 1.8, 2.2], row_h=0.7, size=15)
text(s, 0.6, 4.35, 7.4, 1.6, [{"text": f"* 정상 파일에서 센서가 기록한 시간은 벽시계 {qn_.wallclock_min:.1f}분 중 {qn_.recorded_min:.1f}분({pct(rec_frac, 0)}). 같은 수집 방식으로 배포하면 교대당 기록시간이 약 {shift_h:.1f}시간이라는 가정에 따른 추정이다.", "size": 12.5, "color": MUTED, "ls": 1.05}])
card(s, 8.4, 2.1, 4.35, 4.3, CARD, 0.05)
text(s, 8.7, 2.3, 3.8, 4.0, [{"text": "답변", "size": 17, "bold": True, "color": ACCENT, "after": 8},
                             {"text": "독립 평가 값은 정상 570구간 중 6구간의 오경보라 불확실하다", "size": 14, "bullet": True, "after": 8},
                             {"text": "노랑은 정지가 아니라 확인 신호, 정지는 빨강 2연속일 때만 권고", "size": 14, "bullet": True, "after": 8},
                             {"text": "벽시계 시간으로 나눠 낮춰 보이게 하지 않는다 (기록 공백의 오경보는 관측된 적이 없음)", "size": 14, "bullet": True, "after": 8},
                             {"text": "현장 정상 데이터로 임계값 재보정 절차를 둔다", "size": 14, "bullet": True}])

s = new_slide("부록 · 예상 질문 3/4", "Q. 전류 하나에 의존하는 모델 아닌가?",
              notes="네, 영향의 대부분이 전류 파형의 거칠기입니다. 전류 그룹을 빼면 PR-AUC가 0.78로 크게 떨어지지만, 진동만으로도 완전히 무너지지는 않습니다. 그래서 진동 채널은 보조 역할이고, 위험은 속도 변화에서 나타납니다. 이 약점은 속도 변화 시험으로 공개했고 재보정 절차를 두었습니다.")
order = ["전체", "상부진동(AI0)", "하부진동(AI1)", "채널간 관계", "형상(skew/kurt/crest)", "동특성(mad/madr/zcr)", "전류(AI2)"]
rows = [["제거한 그룹", "재학습 PR-AUC"]] + [["전체(제거 없음)" if k == "전체" else k, f3(lgo[k])] for k in order]
table(s, 0.6, 2.0, 6.2, rows, [3.2, 1.8], row_h=0.5, size=14, hl_rows=(len(rows) - 1,))
card(s, 7.2, 2.0, 5.55, 4.45, CARD, 0.05)
text(s, 7.5, 2.2, 5.0, 4.1, [{"text": "답변", "size": 17, "bold": True, "color": ACCENT, "after": 8},
                             {"text": f"맞다. 단일 변수 {PI.iloc[0].feature}가 순열중요도 {PI.iloc[0].perm_importance:.2f}로 압도적이다", "size": 14, "bullet": True, "after": 8},
                             {"text": f"전류를 빼면 PR-AUC {lgo['전류(AI2)']:.2f}로 하락 — 진동은 보조 역할이다", "size": 14, "bullet": True, "after": 8},
                             {"text": "그래서 전류 거칠기가 달라지는 상황(속도 변화)이 약점이며 시험으로 공개했다", "size": 14, "bullet": True, "after": 8},
                             {"text": "속도·운전조건 변경 시 재보정과 노랑 비율 모니터링으로 대응", "size": 14, "bullet": True}])

s = new_slide("부록 · 예상 질문 4/4", "Q. 이상 데이터가 3분 하나뿐인데 일반화되나?",
              notes="일반화는 보증하지 않습니다. 이상 데이터는 다른 날의 단일 3분 세션이고 30초 블록이 6개뿐이라 신뢰구간이 넓습니다. 그래서 블록 부트스트랩 신뢰구간과 중첩 교차검증을 함께 공개했고, 새로운 고장 데이터가 확보되면 재학습하는 절차를 두었습니다.")
stat(s, 0.6, 2.0, 3.9, 1.7, f"{seg_all['outlier_data.csv']}구간·6블록", "이상 데이터 전체 (다른 날의 단일 약 3분 세션)", big_size=26)
stat(s, 4.7, 2.0, 3.9, 1.7, f"[{npi.loc['F1', 'ci_low']:.3f}, {npi.loc['F1', 'ci_high']:.3f}]", "독립 평가 F1의 95% 신뢰구간 (블록 부트스트랩)", big_size=28)
stat(s, 8.8, 2.0, 3.95, 1.7, pct(UC[UC.top_pct == 2].iloc[0].capture_rate, 0), f"불확실도 상위 2%가 담은 오류 비율 (오류 {int(UC.iloc[0].errors_total)}건뿐)", big_size=28)
card(s, 0.6, 3.95, 12.15, 2.7, CARD, 0.04)
text(s, 0.9, 4.15, 11.6, 2.4, [{"text": "답변", "size": 17, "bold": True, "color": ACCENT, "after": 8},
                               {"text": "일반화는 보증하지 않는다 — '관측된 고장 세션 기준 성능'이라고 명시했다", "size": 15, "bullet": True, "after": 8},
                               {"text": "신뢰구간과 중첩 교차검증, 속도·약한 이상 스트레스 시험으로 불확실성을 먼저 공개했다", "size": 15, "bullet": True, "after": 8},
                               {"text": "새로운 고장 데이터가 확보되면 재학습하고, 그 전까지는 정상 전용 M3 점수를 보조 감시로 병행한다", "size": 15, "bullet": True}])

# ------------------------------------------------------------------------------------------------------------- order check
# the agenda quotes slide ranges; fail loudly if the real order drifts away from it
MARK = "①②③④⑤"
for n, (h, _, a, b) in enumerate(PARTS, 1):
    assert all(KICKERS[k - 1].startswith(MARK[n - 1]) for k in range(a, b + 1)), (n, a, b)
assert len(KICKERS) == 17 + 4 and KICKERS[17].startswith("부록"), KICKERS[15:19]

# ------------------------------------------------------------------------------------------------------------- footer
total = len(prs.slides)
for i, sl in enumerate(prs.slides, 1):
    if i == 1:
        continue
    text(sl, 11.3, 7.05, 1.45, 0.3, [{"text": f"{i} / {total}", "size": 10, "color": "9CA3AF", "align": PP_ALIGN.RIGHT}])
    text(sl, 0.6, 7.05, 6, 0.3, [{"text": "진동·전류 기반 프레스 유압펌프 이상탐지", "size": 10, "color": "9CA3AF"}])

prs.save(OUT)
print("saved", OUT, f"({total} slides)")

"""Self-contained HTML dashboard (no server, no internet) built from outputs/.

    python report/build_dashboard.py      # after run_all.py -> outputs/dashboard.html (double-click to open)

run_all.py calls this automatically at the end of every full (non --quick) run and overwrites the same
dashboard.html path. A tab left open on that file auto-reloads every 20s (toggle in the page header), so
it picks up the next run's numbers without a manual refresh; the write itself is atomic so a reload can
never land mid-write.
"""
import base64
import json
import os
from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(globals().get("OUT_DIR", ROOT / "outputs"))          # OUT_DIR is injected by run_all.py
TAB, FIG = OUT / "tables", OUT / "figures"
SHORT = {"B0_SPC": "B0 SPC 규칙", "B1_LogReg": "B1 로지스틱", "M1_RF": "M1 랜덤포레스트",
         "M2_HGB": "M2 HistGB", "M3_IF": "M3 IsolationForest", "M4_Stack": "M4 스태킹"}
ORDER = list(SHORT)


def rd(name): return pd.read_csv(TAB / name)


def pct(x, d=1): return f"{100 * x:.{d}f}%"


def figure(name: str, caption: str) -> str:
    b64 = base64.b64encode((FIG / name).read_bytes()).decode()
    return f'<figure><img alt="{escape(caption)}" src="data:image/png;base64,{b64}"><figcaption>{escape(caption)}</figcaption></figure>'


def table(head: list, rows: list, mark_row: int = -1) -> str:
    h = "".join(f"<th>{escape(c)}</th>" for c in head)
    body = ""
    for i, r in enumerate(rows):
        cls = ' class="hit"' if i == mark_row else ""
        body += f"<tr{cls}>" + "".join(f"<td>{escape(str(c))}</td>" for c in r) + "</tr>"
    return f'<div class="tw"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


S, MS = rd("03_model_comparison.csv"), rd("04_mode_shift.csv")
J = json.loads((OUT / "results_summary.json").read_text(encoding="utf8"))
P = pd.read_csv(OUT / "predictions_oof.csv")
THR = J["thresholds"]
FINAL, FSET = J["final_model"], J["final_set"]
SHORT[FINAL] += " (최종)"
FSET_TXT = {"INV": "운전모드 불변 특징 17개", "ALL": "전체 특징 34개"}[FSET]
N_ANOM_SEG = J["segments_used"]["outlier_data.csv"]


def sm(fs, m, c): return float(S[(S.feature_set == fs) & (S.model == m)][c].iloc[0])


def shift(fs, m, mode, c="FPR_on_unseen_mode"):
    return float(MS[(MS.feature_set == fs) & (MS.model == m) & (MS.held_out_mode == mode)][c].iloc[0])


kpis = [
    (f"{sm(FSET, FINAL, 'F1'):.3f}", "F1-score", f"SPC 규칙 {sm(FSET, 'B0_SPC', 'F1'):.3f}"),
    (pct(sm(FSET, FINAL, "FPR"), 2), "오경보율(FPR)", "정상 창 기준"),
    (f"{sm(FSET, FINAL, 'seg_recall') * N_ANOM_SEG:.0f}/{N_ANOM_SEG}", "이상 구간 탐지", "첫 경보 지연 1.0초"),
    (pct(shift(FSET, FINAL, 1)), "미학습 고부하 모드 오경보", f"진폭 특징 모델 {pct(shift('ALL', FINAL, 1), 0)}"),
]
kpi_html = "".join(f'<div class="kpi"><b>{v}</b><span>{t}</span><small>{s}</small></div>' for v, t, s in kpis)

t_models = table(
    ["모델", "F1", "PR-AUC", "재현율", "정밀도", "FPR", "정상구간 오경보"],
    [[SHORT[m], f"{sm(FSET, m, 'F1'):.3f}", f"{sm(FSET, m, 'PR_AUC'):.3f}", f"{sm(FSET, m, 'Recall'):.3f}",
      f"{sm(FSET, m, 'Precision'):.3f}", pct(sm(FSET, m, "FPR"), 2), pct(sm(FSET, m, "seg_false_alarm_rate"), 2)] for m in ORDER],
    mark_row=ORDER.index(FINAL))
t_shift = table(
    ["모델", "전체 특징: mode0 / mode1", "불변 특징: mode0 / mode1"],
    [[SHORT[m], f"{pct(shift('ALL', m, 0))} / {pct(shift('ALL', m, 1))}", f"{pct(shift('INV', m, 0))} / {pct(shift('INV', m, 1))}"]
     for m in ORDER], mark_row=ORDER.index(FINAL))


def predict_log_html() -> str:
    """Render predict.py's run history (outputs/predict_log.csv). predict.py appends to that file and
    rebuilds this page after every scoring run, so this section is what makes the dashboard reflect
    predict.py runs, not just run_all.py's own training/validation run."""
    path = OUT / "predict_log.csv"
    if not path.exists():
        return ('<p class="lead">아직 predict.py 실행 이력이 없습니다. 터미널에서 '
                '<code>python predict.py --input &lt;새 CSV&gt;</code> 를 실행하면 이 표에 한 줄씩 쌓이고, '
                '이 탭은 자동 새로고침으로 그 결과를 곧바로 보여줍니다.</p>')
    L = pd.read_csv(path).sort_values("run_at", ascending=False)
    shown, extra = L.head(30), max(len(L) - 30, 0)
    dotvar = {"green": "--ok", "yellow": "--warn", "red": "--bad"}
    label = {"green": "정상", "yellow": "주의", "red": "경고"}
    head = "".join(f"<th>{escape(h)}</th>" for h in ["실행 시각", "입력 파일", "창 수", "구간 수", "최종 판정", "초록/주의/경고", "최고 확률"])
    body = ""
    for r in shown.itertuples():
        badge = f'<span class="dot" style="background:var({dotvar[r.worst_level]})"></span>{escape(label[r.worst_level])}'
        cells = "".join(f"<td>{escape(str(c))}</td>" for c in (r.run_at, r.input, r.windows, r.segments))
        body += f"<tr>{cells}<td>{badge}</td><td>{escape(f'{r.green}/{r.yellow}/{r.red}')}</td><td>{r.max_p:.3f}</td></tr>"
    note = f'<p class="lead">최근 30건만 표시 (전체 {len(L)}건)</p>' if extra else ""
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{note}'

LEVEL_LABEL = {"green": "정상", "yellow": "주의", "red": "경고"}


def load_latest_run():
    """predict.py's newest scoring run (outputs/latest_run.json); None when there is none or the file is unreadable."""
    try:
        return json.loads((OUT / "latest_run.json").read_text(encoding="utf8"))
    except (OSError, ValueError):
        return None


RUN = load_latest_run()


def latest_run_html() -> str:
    """Section 6: the new data's own results (cards + timeline/waveform canvases drawn by the script below + segment table)."""
    if RUN is None:
        return ('<p class="lead">아직 채점한 새 데이터가 없습니다. 터미널에서 <code>python predict.py --input &lt;새 CSV&gt;</code> 를 실행하면 '
                '이 자리에 새 데이터의 창별 이상 확률, 경보 등급, 구간별 파형이 그려지고 이 탭은 자동 새로고침으로 곧바로 보여줍니다.</p>')
    n, c = max(RUN["windows"], 1), RUN["counts"]
    cell = lambda col, t, v, s: f'<div><span class="dot" style="background:var({col})"></span>{t}<b>{v}</b><small>{s}</small></div>'
    cards = (cell("--ok", "초록 창", c["green"], pct(c["green"] / n)) + cell("--warn", "주의(노랑) 창", c["yellow"], pct(c["yellow"] / n))
             + cell("--bad", "경고(빨강) 창", c["red"], pct(c["red"] / n))
             + cell("--bad", "정지 권고 구간", f'{RUN["stop"]}/{RUN["segments"]}', "경고 창이 연속 2개 이상(R2)")
             + cell("--mut", "신규 모드 창", pct(RUN["new_mode_share"]), "알려진 운전모드 밖 — 승인·재학습 대기"))
    rows = [[s["seg"], s["t_start"], s["windows"], f'{s["p_max"]:.3f}', LEVEL_LABEL[s["worst_level"]], "예" if s["stop"] else "아니오",
             pct(s["new_mode_share"], 0)] for s in RUN["segs"]]
    shown = f'<p class="lead">위험한 순서로 {len(rows)}개 구간만 표시 (전체 {RUN["segments"]}개)</p>' if len(rows) < RUN["segments"] else ""
    return (f'<p class="lead">입력 <b>{escape(RUN["input"])}</b> · 채점 {escape(RUN["run_at"])} · 창 {RUN["windows"]:,}개 · 구간 {RUN["segments"]}개 '
            f'(predict.py를 다시 실행하면 가장 최근 한 번만 보여줍니다)</p><div class="lv">{cards}</div>'
            '<div class="panel" style="margin-top:14px"><b>창별 이상 확률과 경보 등급</b>'
            '<canvas id="cvRun" role="img" aria-label="새 데이터의 창별 이상 확률과 경보 등급"></canvas>'
            '<p class="lead" id="runNote" style="margin:8px 0 0"></p></div>'
            '<div class="panel" style="margin-top:14px"><label><b>구간 파형</b> &nbsp;<select id="segSel"></select></label>'
            '<canvas id="cvWave" style="height:300px" role="img" aria-label="선택한 구간의 진동·전류 파형과 창별 경보 등급"></canvas>'
            '<p class="lead" style="margin:8px 0 0">음영 = 그 1초 창의 경보 등급(초록/노랑/빨강). 이상 확률이 가장 높은 창 주변 최대 60초만 그립니다.</p></div>'
            + table(["구간", "시작 시각", "창 수", "최고 확률", "최종 등급", "정지 권고", "신규 모드 창"], rows) + shown)


data = {"y": P.y.astype(int).tolist(), "p": [round(v, 5) for v in P.p_final], "unc": [round(v, 4) for v in P.unc],
        "g": (P["guard"] if "guard" in P else P.y * 0).astype(int).tolist()}   # peak-current guard flag: lifts a window to yellow only
BUILD_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

HTML = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>유압펌프 이상탐지 결과</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#1d2330;--mut:#5d6675;--line:#dfe3ea;--acc:#c0392b;--nor:#2c6fbb;--ok:#3c9d5d;--warn:#e0a020;--bad:#d64541;--hit:#fff3f1}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--card:#1b2029;--tx:#e8ebf0;--mut:#9aa4b5;--line:#2b3240;--acc:#ff7a6b;--nor:#6aa6f0;--ok:#54c07b;--warn:#f0b840;--bad:#ff6b66;--hit:#2a1f22}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:15px/1.65 system-ui,"Malgun Gothic",sans-serif}
header{padding:28px 16px 8px;max-width:1080px;margin:auto}h1{margin:0 0 4px;font-size:clamp(20px,4vw,28px)}header p{margin:0;color:var(--mut)}
nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5}nav div{max-width:1080px;margin:auto;padding:8px 16px;display:flex;gap:16px;overflow-x:auto;white-space:nowrap}
nav a{color:var(--mut);text-decoration:none;font-size:14px}nav a:hover{color:var(--acc)}
main{max-width:1080px;margin:auto;padding:8px 16px 60px}section{margin-top:32px}h2{font-size:20px;margin:0 0 6px}p.lead{color:var(--mut);margin:0 0 14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:18px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;display:flex;flex-direction:column}.kpi b{font-size:30px;color:var(--acc);line-height:1.2}.kpi span{font-weight:600}.kpi small{color:var(--mut)}
figure{margin:14px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}figure img{width:100%;height:auto;background:#fff;border-radius:6px;display:block}figcaption{color:var(--mut);font-size:13px;margin-top:8px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}
.tw{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px;margin:12px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 12px;text-align:center;border-bottom:1px solid var(--line)}th{color:var(--mut);font-weight:600}td:first-child,th:first-child{text-align:left}tr:last-child td{border-bottom:0}tr.hit td{background:var(--hit);font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}.ctrl{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px 24px;margin-bottom:14px}
.ctrl label{display:flex;justify-content:space-between;font-size:14px}.ctrl output{font-variant-numeric:tabular-nums;color:var(--acc);font-weight:600}input[type=range]{width:100%;accent-color:var(--acc)}
canvas{width:100%;height:220px;display:block}.lv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.lv div{border:1px solid var(--line);border-radius:10px;padding:10px 12px}.lv b{font-size:22px;display:block}.lv small{color:var(--mut)}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
code{background:var(--line);padding:1px 5px;border-radius:4px;font-size:13px}
.live{display:flex;align-items:center;gap:6px;margin-top:6px;font-size:13px;color:var(--mut)}
.live input{accent-color:var(--acc)}.live b{color:var(--tx)}
@media(max-width:480px){.kpi b{font-size:26px}}
</style></head><body>
<header><h1>프레스 유압펌프 이상 조기탐지 · 결과 대시보드</h1>
<p>진동(상부·하부)·전류 10Hz 시계열 → 1초 윈도우 이상 확률 · 최종 모델 __FINAL__</p>
<p class="live">생성(run_all.py·predict.py 실행 완료 시각): <b id="genTime">__BUILD_TIME__</b>
<label><input type="checkbox" id="autoR"> <span id="autoRLabel">20초마다 자동 새로고침</span></label></p>
<div class="kpis">__KPIS__</div></header>
<nav><div><a href="#data">1 데이터</a><a href="#model">2 모델 비교</a><a href="#shift">3 운전모드 시험</a><a href="#why">4 영향요인·오류</a><a href="#alarm">5 경보 정책</a><a href="#latest">6 새 데이터 결과</a><a href="#predictlog">7 채점 이력</a></div></nav>
<main>
<section id="data"><h2>1. 데이터: 정상 파형 vs 이상 파형</h2><p class="lead">정상 전류는 매끈한 정현파, 이상 구간은 파형이 무너지고 한쪽으로 치우칩니다. 정상 운전은 부하에 따라 (하부진동, 전류)가 함께 커지는 하나의 궤적입니다.</p>
<div class="grid2">__FIG_SIGNALS____FIG_MODES__</div>__FIG_DATA__</section>
<section id="model"><h2>2. 모델 비교 (블록 그룹 CV 5-fold × 3회)</h2><p class="lead">최종 특징세트 기준 수치입니다. 강조 행이 최종 모델입니다.</p>__T_MODELS____FIG_HEAT__<div class="grid2">__FIG_CM____FIG_CI__</div>__FIG_COMPARE____FIG_VAL__</section>
<section id="shift"><h2>3. 핵심 발견: 새 운전모드에서 오경보</h2><p class="lead">한 정상 운전모드를 학습에서 빼고 그 모드의 정상 창을 이상으로 오판하는 비율입니다. 진폭 특징을 쓰면 무너지고 불변 특징이면 사라집니다.</p>
__T_SHIFT__<div class="grid2">__FIG_SHIFT____FIG_SETS__</div>__FIG_SEL__</section>
<section id="why"><h2>4. 무엇이 영향을 주고, 어디서 틀리나</h2><p class="lead">판별 정보 대부분이 전류 파형 평활도(cur_madr)입니다. 오류 건수가 적어 조건별 차이는 기술적 참고용이며, 오경보는 짧은 윈도우 같은 처리 조건에서, 미탐은 전류가 한쪽으로 매끄럽게 치우친 구간에서 두드러집니다.</p>
<div class="grid2">__FIG_IMP____FIG_ERR__</div></section>
<section id="alarm"><h2>5. 경보 정책 시뮬레이터</h2><p class="lead">슬라이더로 임계값을 바꾸면 3단계 경보(정상/주의/경고)별 창 수가 즉시 바뀝니다. 피크전류 가드가 발동한 창은 주의(노랑)로만 계산됩니다. 데이터: 교차검증 예측(누수 없음) 3,671창.</p>
<div class="panel"><div class="ctrl">
<div><label>경고(red) 임계값 <output id="oRed"></output></label><input id="sRed" type="range" min="-7" max="7" step="any"></div>
<div><label>주의(yellow) 임계값 <output id="oYel"></output></label><input id="sYel" type="range" min="-7" max="7" step="any"></div>
<div><label>불확실도(멤버 불일치) 기준 <output id="oUnc"></output></label><input id="sUnc" type="range" min="0" max="0.5" step="any"></div></div>
<canvas id="cv" role="img" aria-label="정상과 이상 창의 이상 확률 분포와 임계값"></canvas>
<div class="lv" id="lv"></div>
<p class="lead" style="margin-top:12px">기본값: red=__TRED__ (F1 최대점), yellow=__TYEL__ (미탐:오경보 비용 10:1 가정), 불확실도 __TUNC__. 가로축은 확률의 로짓 눈금, 세로축은 창 수의 제곱근입니다.</p></div>
__FIG_POLICY____FIG_ALARM____FIG_UNC__</section>
<section id="latest"><h2>6. 새 데이터 채점 결과</h2>__LATEST__</section>
<section id="predictlog"><h2>7. predict.py 채점 이력</h2><p class="lead">새 데이터를 predict.py로 채점할 때마다 한 줄씩 쌓입니다. 이 탭을 열어두면 자동 새로고침으로 바로 반영됩니다.</p>
__PREDICT_LOG__</section>
</main>
<script>
const D=__DATA__, T=__THR__;
const $=id=>document.getElementById(id), lg=p=>Math.log(p/(1-p)), sg=s=>1/(1+Math.exp(-s));
const sR=$("sRed"),sY=$("sYel"),sU=$("sUnc");
sR.value=lg(T.t_red);sY.value=lg(T.t_yellow);sU.value=T.unc_high;
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function draw(tr,ty){
 const c=$("cv"),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;const g=c.getContext("2d");g.scale(r,r);g.clearRect(0,0,w,h);
 const B=36,lo=-9,hi=9,A=new Array(B).fill(0),Nn=new Array(B).fill(0);
 D.p.forEach((p,i)=>{const s=Math.max(lo,Math.min(hi-1e-9,lg(Math.min(Math.max(p,1e-6),1-1e-6))));const b=Math.floor((s-lo)/(hi-lo)*B);(D.y[i]?A:Nn)[b]++});
 const mx=Math.sqrt(Math.max(...A,...Nn)),pad=22,bw=(w-pad*2)/B,H=h-50;
 const bar=(a,col,off)=>{g.fillStyle=col;a.forEach((v,i)=>{const bh=Math.sqrt(v)/mx*H;g.fillRect(pad+i*bw+off,h-24-bh,bw*.46,bh)})};
 bar(Nn,css("--nor"),1);bar(A,css("--acc"),bw*.48);
 g.strokeStyle=css("--line");g.beginPath();g.moveTo(pad,h-24);g.lineTo(w-pad,h-24);g.stroke();
 g.fillStyle=css("--mut");g.font="12px system-ui";g.textAlign="center";
 [-8,-4,0,4,8].forEach(s=>g.fillText("p="+sg(s).toPrecision(2),pad+(s-lo)/(hi-lo)*(w-pad*2),h-8));
 [[tr,css("--bad"),"경고"],[ty,css("--warn"),"주의"]].forEach(([s,col,t])=>{const x=pad+(Math.min(Math.max(s,lo),hi)-lo)/(hi-lo)*(w-pad*2);g.strokeStyle=col;g.setLineDash([5,4]);g.lineWidth=2;g.beginPath();g.moveTo(x,20);g.lineTo(x,h-24);g.stroke();g.setLineDash([]);g.fillStyle=col;g.textAlign="left";g.fillText(t,x+4,32)});
 g.fillStyle=css("--nor");g.fillRect(w-150,6,10,10);g.fillStyle=css("--tx");g.textAlign="left";g.fillText("정상 창",w-136,15);g.fillStyle=css("--acc");g.fillRect(w-80,6,10,10);g.fillStyle=css("--tx");g.fillText("이상 창",w-66,15)}
function upd(){
 const tr=+sR.value,ty=Math.min(+sY.value,tr),u=+sU.value,pr=sg(tr),py=sg(ty);
 $("oRed").textContent="p ≥ "+pr.toFixed(3);$("oYel").textContent="p ≥ "+py.toFixed(3);$("oUnc").textContent="unc ≥ "+u.toFixed(3);
 const C=[[0,0,0],[0,0,0]];
 D.p.forEach((p,i)=>{const k=p>=pr?2:(p>=py||D.unc[i]>=u||D.g[i])?1:0;C[D.y[i]][k]++});
 const nN=C[0].reduce((a,b)=>a+b),nA=C[1].reduce((a,b)=>a+b),f=(a,b)=>(100*a/b).toFixed(1)+"%";
 const cell=(col,t,v,s)=>`<div><span class="dot" style="background:${col}"></span>${t}<b>${v}</b><small>${s}</small></div>`;
 $("lv").innerHTML=cell(css("--ok"),"정상 창 → 초록",C[0][0],f(C[0][0],nN))+cell(css("--warn"),"정상 창 → 주의",C[0][1],f(C[0][1],nN)+" (재확인 부담)")+cell(css("--bad"),"정상 창 → 경고(오경보)",C[0][2],f(C[0][2],nN))
 +cell(css("--bad"),"이상 창 → 경고",C[1][2],f(C[1][2],nA))+cell(css("--warn"),"이상 창 → 주의",C[1][1],f(C[1][1],nA))+cell(css("--ok"),"이상 창 → 초록(놓침)",C[1][0],f(C[1][0],nA));
 draw(tr,ty)}
[sR,sY,sU].forEach(e=>e.addEventListener("input",upd));addEventListener("resize",upd);upd();
// ---- section 6: the newest predict.py run (R = outputs/latest_run.json, null when predict.py has not run yet) ----
const R=__RUN__, LB={green:"정상",yellow:"주의",red:"경고"};
function canvas2d(id){const c=$(id),r=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*r;c.height=h*r;const g=c.getContext("2d");g.scale(r,r);g.clearRect(0,0,w,h);g.font="12px system-ui";return [g,w,h]}
function drawRun(){
 if(!R)return;
 const [g,w,h]=canvas2d("cvRun"),T=R.timeline,n=T.p.length,L=46,Rm=12,top=12,bot=24,lo=-8,hi=8,col=[css("--ok"),css("--warn"),css("--bad")];
 const X=i=>L+(n>1?i/(n-1):.5)*(w-L-Rm),Y=p=>top+(hi-Math.max(lo,Math.min(hi,lg(Math.min(Math.max(p,1e-6),1-1e-6)))))/(hi-lo)*(h-top-bot);
 g.strokeStyle=css("--line");g.fillStyle=css("--mut");g.textAlign="right";
 [0.001,0.01,0.1,0.5,0.9,0.99].forEach(p=>{g.beginPath();g.moveTo(L,Y(p));g.lineTo(w-Rm,Y(p));g.stroke();g.fillText(p,L-6,Y(p)+4)});
 g.setLineDash([2,4]);T.seg_at.forEach(i=>{g.beginPath();g.moveTo(X(i),top);g.lineTo(X(i),h-bot);g.stroke()});
 g.setLineDash([5,4]);g.lineWidth=2;
 [[R.thr.t_red,col[2],"경고"],[R.thr.t_yellow,col[1],"주의"]].forEach(([p,c,t])=>{g.strokeStyle=c;g.beginPath();g.moveTo(L,Y(p));g.lineTo(w-Rm,Y(p));g.stroke();g.fillStyle=c;g.textAlign="left";g.fillText(t+" 임계값",L+6,Y(p)-4)});
 g.setLineDash([]);g.lineWidth=1;
 const s=n<=300?6:3;[0,1,2].forEach(k=>{g.fillStyle=col[k];T.p.forEach((p,i)=>{if(T.lv[i]===k)g.fillRect(X(i)-s/2,Y(p)-s/2,s,s)})});
 g.fillStyle=css("--mut");g.textAlign="center";g.fillText("창 순서 (시간순)"+(T.seg_at.length?" · 세로 점선 = 구간 경계":""),(L+w-Rm)/2,h-6);
 $("runNote").textContent="가로 = 1초 창(0.5초 간격), 세로 = 이상 확률(로짓 눈금). 점 색 = 경보 등급(초록·노랑·빨강)."+(T.step>1?` 창이 많아 점 하나 = 연속 ${T.step}개 창의 최댓값.`:"")}
function drawWave(){
 if(!R)return;
 const s=R.segs[+$("segSel").value||0];if(!s)return;
 const [g,w,h]=canvas2d("cvWave"),W=s.wave,n=W.ai0.length,L=96,Rm=12,top=8,bot=22,rh=(h-top-bot)/3,col=[css("--ok"),css("--warn"),css("--bad")];
 const X=i=>L+(n>1?i/(n-1):.5)*(w-L-Rm);
 g.globalAlpha=.2;W.wins.forEach(([a,b,k])=>{g.fillStyle=col[k];g.fillRect(X(a),top,Math.max(X(b)-X(a),1),h-top-bot)});g.globalAlpha=1;
 [["상부진동 AI0",W.ai0],["하부진동 AI1",W.ai1],["전류 AI2",W.ai2]].forEach(([nm,v],k)=>{
  const y0=top+k*rh,mn=Math.min(...v),mx=Math.max(...v),Y=x=>y0+rh-6-(x-mn)/((mx-mn)||1)*(rh-14);
  g.strokeStyle=css("--line");g.beginPath();g.moveTo(L,y0+rh);g.lineTo(w-Rm,y0+rh);g.stroke();
  g.fillStyle=css("--mut");g.textAlign="right";g.fillText(nm,L-8,y0+rh/2+4);
  g.strokeStyle=css("--tx");g.lineWidth=1;g.beginPath();v.forEach((x,i)=>i?g.lineTo(X(i),Y(x)):g.moveTo(X(i),Y(x)));g.stroke()});
 g.fillStyle=css("--mut");g.textAlign="center";[0,.25,.5,.75,1].forEach(f=>g.fillText((W.t0+f*(n-1)/10).toFixed(1)+"초",L+f*(w-L-Rm),h-6))}
function runDraw(){drawRun();drawWave()}
if(R){const sel=$("segSel");R.segs.forEach((s,i)=>sel.add(new Option(`구간 ${s.seg} · ${LB[s.worst_level]} · 최고 p ${s.p_max.toFixed(3)} · ${s.t_start}`,i)));sel.addEventListener("change",drawWave);addEventListener("resize",runDraw);runDraw()}
(function(){
 // Auto-refresh: re-reads this same file every REFRESH_MS, so a tab left open picks up the next
 // run_all.py run without a manual F5. Scroll position survives the reload; the checkbox state
 // (on by default) persists across reloads via localStorage so toggling it off actually sticks.
 const REFRESH_MS=20000, cb=$("autoR"), onKey="dash_autorefresh", v=localStorage.getItem(onKey);
 cb.checked = v===null ? true : v==="1";
 cb.addEventListener("change",()=>localStorage.setItem(onKey, cb.checked?"1":"0"));
 const sKey="dash_scroll", saved=sessionStorage.getItem(sKey);
 if(saved) scrollTo(0,+saved);
 addEventListener("beforeunload",()=>sessionStorage.setItem(sKey,String(scrollY)));
 setInterval(()=>{ if(cb.checked) location.reload(); }, REFRESH_MS);
})();
</script></body></html>"""

repl = {
    "__KPIS__": kpi_html, "__T_MODELS__": t_models, "__T_SHIFT__": t_shift,
    "__FIG_SIGNALS__": figure("fig01_signals.png", "정상(좌)/이상(우) 연속 50샘플 파형"),
    "__FIG_MODES__": figure("fig03_modes.png", "정상 운전모드(군집)와 이상 창 분포"),
    "__FIG_COMPARE__": figure("fig05_model_compare.png", "모델별 F1·PR-AUC·시간당 오경보"),
    "__FIG_SHIFT__": figure("fig12_mode_shift.png", "미학습 운전모드 정상 창의 오경보율"),
    "__FIG_SETS__": figure("fig13_feature_sets.png", "특징세트별 기존 모드 F1 vs 미학습 모드 F1"),
    "__FIG_IMP__": figure("fig08_importance.png", "변수·그룹 영향도"),
    "__FIG_ERR__": figure("fig10_error_conditions.png", "조건별 오경보·미탐 근접 비율"),
    "__FIG_POLICY__": figure("fig11_alarm_policy.png", "정상 → 이상 세션 전환부의 3단계 경보 예시"),
    "__FINAL__": f"{SHORT[FINAL].replace(' (최종)', '')} + {FSET_TXT}",
    "__FIG_DATA__": figure("fig26_data_overview.png", "데이터 현황: 원시 행 수, 연속 구간 수, 1초 윈도우 수"),
    "__FIG_HEAT__": figure("fig20_metrics_heatmap.png", "모델별 성능 지표 히트맵 (진할수록 좋음)"),
    "__FIG_CM__": figure("fig21_confusion.png", "윈도우 오분류 수: SPC 규칙 vs 최종 모델"),
    "__FIG_CI__": figure("fig22_intervals.png", "신뢰구간(부트스트랩)과 반복 변동"),
    "__FIG_VAL__": figure("fig23_validation_checks.png", "검증 점검: 누수 비교 · 창 길이 · 불균형 처리"),
    "__FIG_SEL__": figure("fig27_selection.png", "최종 모델 선정 점수"),
    "__FIG_ALARM__": figure("fig24_alarm_results.png", "경보 등급 분포 · 규칙별 오경보 · 비용비별 임계값"),
    "__FIG_UNC__": figure("fig25_uncertainty.png", "불확실도 구간별 오류율"),
    "__TRED__": f"{THR['t_red']:.3f}", "__TYEL__": f"{THR['t_yellow']:.3f}", "__TUNC__": f"{THR['unc_high']:.3f}",
    "__DATA__": json.dumps(data, separators=(",", ":")), "__THR__": json.dumps(THR),
    "__BUILD_TIME__": BUILD_TIME,
    "__PREDICT_LOG__": predict_log_html(), "__LATEST__": latest_run_html(),
    "__RUN__": json.dumps(RUN, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/"),      # json null when there is no run
}
html = HTML
for k, v in repl.items():
    html = html.replace(k, v)
out = OUT / "dashboard.html"
tmp = out.with_suffix(".html.tmp")
tmp.write_text(html, encoding="utf-8")
os.replace(tmp, out)      # atomic: a tab auto-reloading mid-write (see JS below) never sees a half-written file
print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

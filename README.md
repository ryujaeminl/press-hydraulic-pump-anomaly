# 프레스 유압펌프 이상 조기탐지 및 오경보 분석

진동(상부·하부)·전류 10Hz 시계열로 1초 윈도우별 이상 확률을 예측하고, 미탐·오경보 발생 조건을 분석한다.
(제6회 K-인공지능 제조데이터 분석 경진대회 · 소성가공 예지보전 AI 데이터셋)

## 환경

- **Python 3.11 이상 권장.** 3.11(범위형 requirements), 3.12·3.13(lock)에서 설치, pytest 15개 통과, `--quick` 결과가 같은 수치(F1 0.976)로 재현되는 것을 확인했다.
  3.10에서도 설치·실행은 되지만 scikit-learn이 1.7로 내려가 수치가 달라진다(`--quick` F1 0.942 vs 0.976).
- `requirements.txt` = 버전 범위(어느 환경에서나 설치), `requirements-lock.txt` = 보고서 수치를 만든 정확한 버전(Python 3.12 이상).
- **결과만 보려면 다시 돌릴 필요가 없다**: `outputs/`에 표(`tables/`), 그림(`figures/`), `dashboard.html`, 학습된 모델이 이미 들어 있고
  `python report/build_report.py`가 그 표에서 보고서를 만든다. 전체 재실행은 약 90분이다.
- **데이터 위치**: `python run_all.py`는 `solution/data/`(zip에 데이터를 같이 넣은 경우)를 먼저, 없으면 상위 폴더에서 `press_data_normal.csv`, `outlier_data.csv`를 찾는다. 다른 곳이면 `--data <폴더>`.

## 실행

```bash
pip install -r requirements.txt   # 보고서 수치까지 똑같이 맞추려면: pip install -r requirements-lock.txt
python run_all.py                 # 전처리 → 학습/검증 → 분석 → 스트레스 시험 → 최종 모델 → 표·그림 (약 90분)
python run_all.py --quick         # 스모크 테스트 (최종 모델만, 약 3분) → outputs_quick/ 에 따로 저장, outputs/ 는 건드리지 않음
python run_all.py --data <폴더>   # press_data_normal.csv, outlier_data.csv 가 있는 폴더 (기본: 이 폴더의 상위)
python predict.py --input <새 CSV>   # 학습된 모델로 새 데이터 추론 (TimeStamp, AI0_Vibration, AI1_Vibration, AI2_Current)
                                  #   → outputs/predictions_new.csv(창별) · predictions_new_segments.csv(구간별) + outputs/dashboard.html 의 "6. 새 데이터 채점 결과"
                                  #     (창별 이상 확률·경보 등급 그래프, 구간별 진동·전류 파형, 위험 구간 표). 재학습은 하지 않는다.
                                  #   --no-unc-trigger: 앙상블 불일치로 노랑을 올리는 조건을 끈다(정상 창 노랑 약 1.3% 감소)
python report/build_report.py     # outputs/tables 를 읽어 report/최종보고서.hwpx 재생성
python report/build_slides.py     # 발표자료 report/발표자료.pptx 재생성 (python-pptx 필요: pip install python-pptx)
python report/build_dashboard.py  # outputs/dashboard.html 재생성 (run_all.py 가 끝나면 자동 생성·자동 열림; 끄려면 run_all.py --no-open)
python -m pytest -q tests        # 단위 테스트 15개
```

## 제출 zip 구성

| 요구 항목 | 위치 |
|---|---|
| 소스코드 + 패키지 목록 | `src/`, `run_all.py`, `predict.py`, `report/*.py`(대시보드·보고서·발표자료 생성기)와 양식 `report/template.hwpx`, `tests/`, `requirements.txt`(범위) · `requirements-lock.txt`(정확한 버전) |
| 학습용 데이터 | `data/press_data_normal.csv`, `data/outlier_data.csv` (`run_all.py`가 이 폴더를 먼저 찾는다) |
| README | 이 파일 |
| 예측 결과 파일 | `predictions/predictions_oof.csv` (`outputs/predictions_oof.csv`와 같은 파일) |

**예측 결과 파일에 대해**: 대회가 따로 준 시험(test) 데이터는 없다. 그래서 학습용 데이터를 시간 블록 단위로 나눈 교차검증(5-fold, 첫 번째 반복)에서
*그 창이 속한 블록을 학습에 쓰지 않은 모델*이 낸 예측만 모아 두었다(1초 창 3,671개, 한 줄이 창 하나). 학습에 쓴 데이터를 그대로 다시 채점한 값이 아니므로 시험 데이터 예측에 가장 가까운 값이다.

| 열 | 뜻 |
|---|---|
| `seg`, `source`, `t_start`, `t_end`, `pos_in_seg`, `win_len` | 연속 구간 번호, 원본 파일, 창의 시작·끝 시각, 구간 안 시작 위치(초), 창 샘플 수 |
| `mode` | 운전모드 군집 번호 |
| `y` | 정답(0 = 정상 파일, 1 = 이상 파일) |
| `p_B0_SPC` … `p_M4_Stack` | 모델별 이상 확률 |
| `p_final` | 최종 모델(HGB, 모양 특징 17개) 이상 확률 — 경보는 이 값으로 정한다 |
| `unc` | 앙상블 불일치(불확실도) |
| `guard`, `new_mode` | 모드별 피크전류 가드 발동 여부, 알려진 모드 밖(신규 모드) 여부 |
| `level`, `unc_flag` | 경보 등급 green / yellow / red, 불확실도 기준을 넘은 창 여부 |

새 데이터를 채점하려면 `python predict.py --input <새 CSV>`(입력 형식은 위 실행 절)를 쓴다.

## 구조

| 경로 | 내용 |
|---|---|
| `src/data.py` | 로딩, 중복 타임스탬프 제거, 연속 구간(gap>0.15s) 분할, 30초 블록 그룹 |
| `src/features.py` | 1초(10샘플) 윈도우, 특징 34개 / 운전모드 불변 17개 |
| `src/models.py`, `pipeline.py` | B0 SPC, B1 로지스틱, M1 RF, M2 HistGB, M3 IsolationForest, M4 스태킹 + Platt 보정 + 임계값 |
| `src/evaluate.py` | 블록 그룹 교차검증, 윈도우·구간·경보 지표 |
| `src/analysis.py`, `errors.py` | 운전모드 군집, 중요도, 미학습 모드 외삽 시험, 모델 선정, 약한 이상·속도 변화 스트레스 시험, 중첩 교차검증, 오류 조건(기계 상태/처리 조건), 경보 정책(`sequence_hit`: 리포트 표와 `predict.py`가 같은 함수 사용), 모드별 가드 효과, 불확실도 포착률 |
| `outputs/` | `tables/*.csv`, `figures/*.png`, `model_bundle.joblib`, `predictions_oof.csv`, `results_summary.json` |
| `report/` | `build_dashboard.py`(`predict.py`·`run_all.py`가 부르는 html 대시보드 생성기), 양식 기반 보고서 생성기(`build_report.py`, `hwpx_engine.py`, 양식 `template.hwpx`), 발표자료 생성기(`build_slides.py`). 생성 결과(`최종보고서.hwpx`·`발표자료.pptx`)는 따로 제출하므로 zip에는 넣지 않았다 |

## 핵심 결과 (블록 그룹 CV 5-fold × 3회, 모델 단독; 관측된 고장 세션 기준)

- 최종 모델: **HistGradientBoosting + 운전모드 불변 특징 17개** — F1 0.981(교차검증), 재현율 0.971, FPR 0.03%, 이상 구간 18/18 탐지
- **선택과 평가 분리(중첩 교차검증, 외부 3-fold)**: 외부 test 창 합산 F1 0.966 [0.911, 0.996], 재현율 0.962. 외부 fold마다 선택된 모델이 달라(M2/M4/M1) 선택이 데이터 분할에 민감하다. 같은 test 창에서 R2 규칙은 탐지 94%, 오경보 10.8건/시간으로 표 4-2의 내부 수치보다 나쁘다.
- SPC 규칙(베이스라인) F1 0.687. 진폭 특징 모델은 학습에 없던 고부하 모드에서 정상의 70~88%를 오경보 → 불변 특징으로 0.2% 이하(빨강 기준)
- **피크전류 가드**는 *운전모드별 한계*(저부하 174.6 / 고부하 273.2, 설비 정격이 아니라 관측 최댓값)로 경보를 **노랑으로만** 올린다. 어느 모드 영역에도 속하지 않는 창은 ‘신규 모드’(승인·재학습 대기)로 분리한다. 미학습 모드 시험에서 전역 한계의 노랑 폭주 90.6% → 5.6%로 줄지만 이상 창에 대한 가드 민감도도 함께 낮아진다. 이 데이터에서 가드의 추가 탐지 이득은 0이다. 빨강 강제는 오경보만 늘어 폐기했다.
- 불확실도: 오류 5건 중 4건이 상위 2%에 몰려(이항검정 p≈8e-7) 사전 기준을 통과해 노랑 트리거로 쓰지만, 오류 건수가 적고 unc만으로 노랑이 되는 정상 창이 46개라 비용이 크다.
- 약한 이상: 진폭이 가장 약한 실제 이상 창 사분위에서 SPC 0% vs 모델 100%. 합성 혼합 시험에서는 빨강보다 노랑이 먼저 반응한다.
- 속도 변화(겉보기 주파수 배율) 시험: 빨강 오경보 1% 이하는 ×1~×1.5 범위뿐. 이 데이터에는 속도를 바꾼 정상 운전이 없어 현장 속도 범위의 안전성은 검증하지 못했다.
- 이상 데이터가 다른 날의 단일 3분 세션이므로 신뢰구간이 넓고 신규 고장 유형으로의 일반화는 보증되지 않는다 (보고서 '추가 기술 사항' 참조)

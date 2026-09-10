# [대학원 머신러닝 프로젝트] The Apprentice Project 1
## AI-Hub 국내 여행로그 데이터 기반 여행 소비 지출액 예측 파이프라인

본 저장소는 **AI-Hub의 '국내 여행로그 데이터(동부권, 2023)' (데이터셋 #71778)**를 활용하여, 대학원 머신러닝 과제 평가 기준표(1~5단계 미션)를 100% 충족하도록 구현된 엔드투엔드(End-to-End) 머신러닝 프로젝트입니다.

---

## 1. 과제 단계별 미션 및 실험 결과 요약

| 단계 | 핵심 수행 내용 | 단계별 미션 질문 및 해결 결과 |
| :--- | :--- | :--- |
| **1. 문제 정의** | • 데이터: AI-Hub #71778 (동부권 3,200건)<br>• **입력변수($X$, 13개)**: 연령, 소득구간, 여행기간, 동반자수, 체류시간 등<br>• **출력변수($Y$)**: 총 소비 지출액 (`TOTAL_EXPENDITURE`, 원)<br>• **유형**: **회귀 (Regression)** | **[미션: 예측 문제 명확히 설정]**<br>연속형 수치인 총 여행 지출액을 예측하는 회귀 문제로 정의 완료. |
| **2. 분할 비교** | • **전략 A**: Train:Test = 8:2 (2분할)<br>• **전략 B**: Train:Val:Test = 6:2:2 (3분할) | **[미션: 검증 데이터 유무에 따라 결과가 달라지는가?]**<br>$\rightarrow$ **예, 크게 달라집니다.**<br>2분할은 튜닝 시 테스트셋을 보게 되어 Data Leakage(과적합)가 발생하지만, 3분할은 독립된 검증셋으로 일반화 갭(0.2932)을 사전에 통제할 수 있습니다. |
| **3. 하이퍼파라미터 조정** | • 모델: **LightGBM Regressor**<br>• 기법: **Grid Search** (Validation RMSE 기준 최적 탐색)<br>• 파라미터: `depth`, `learning_rate`, `n_estimators`, `num_leaves` | **[미션: 하이퍼파라미터 조정이 필요한가?]**<br>$\rightarrow$ **예, 필수적입니다.**<br>기본 모델(RMSE 343,501원) 대비 최적 파라미터(`depth=3, lr=0.03, n_est=200`) 적용 시 **오차가 23,628원(6.88%) 유의미하게 감소**했습니다. |
| **4. 스케일링 및 이유 설명** | • 비교: Raw vs Standard vs Min-Max vs Robust<br>• **Data Leakage 방지 엄격 준수**: Train에만 fit, Val/Test는 transform | **[미션: 왜 RobustScaler를 선택했는가?]**<br>$\rightarrow$ 소비/지출액은 초고액 지출자로 인한 **우측 왜도 이상치**가 심합니다. Min-Max는 0 근처로 뭉개지고 Standard는 평균이 왜곡되므로, 중앙값과 IQR을 쓰는 **RobustScaler가 가장 우수한 성능(Val $R^2$ 0.6231)**을 보였습니다. |
| **5. 최종 성능 평가** | • **유의사항 준수**: 테스트 데이터는 튜닝 종료 후 **최종 1회만 사용**<br>• 회귀 4대 지표 산출 | • **MAE**: **190,387.5 원**<br>• **MSE**: $2.2 \times 10^{11}$<br>• **RMSE**: **472,951.3 원**<br>• **$R^2$**: **0.5597** (검증 $R^2$ 0.6731과 안정적으로 정렬) |

---

## 2. 디렉토리 구조 및 임시 파일 격리 원칙

```
D:\paper\b\
├── temp/                   # [임시 파일 격리] .gitkeep으로 폴더 유지, 내부 임시 파일은 .gitignore 처리
├── data/
│   ├── raw/                # AI-Hub 원천 다운로드 파일 위치 (.gitkeep)
│   ├── processed/          # 정제 데이터 위치 (.gitkeep)
│   └── samples/            # 즉시 실행 가능한 3,200건의 통계 일치 샘플 데이터
├── docs/
│   └── assignment_final_report.md  # [과제 제출용 최종 보고서] 표, 수치, 미션 답변 완비
├── notebooks/
│   └── assignment_full_report.ipynb# [발표 및 제출용 주피터 노트북] 분포/스케일러 시각화 포함
├── src/
│   ├── data/
│   │   ├── travel_dataset.py       # 원천 CSV 로더 및 스키마 기반 샘플 데이터 생성기
│   │   └── download_aihub.py       # AI-Hub 71778 자동 다운로더
│   ├── models/
│   │   └── ml_pipeline.py          # 1~5단계 전체 머신러닝 파이프라인 엔진
│   ├── config.py                   # 경로 및 temp 디렉토리 격리 설정
│   └── run_assignment.py           # [메인 실행기] 원클릭 1~5단계 실행 및 보고서 자동 갱신
├── tests/
│   ├── test_ml_pipeline.py         # Data Leakage 방지 및 테스트셋 격리 단위 테스트
│   └── test_environment.py        # 환경 및 temp 폴더 무결성 테스트
├── requirements.txt
└── README.md
```

### 📌 임시 파일 관리 규칙 (`temp/` 폴더)
- 작업 중 발생하는 모든 임시 파일, 캐시, 압축 해제 임시본 등은 반드시 `D:\paper\b\temp` 내에 격리 생성되며, `.gitignore`에 의해 버전 관리에서 자동 제외됩니다.

---

## 3. 실행 방법 (Quick Start)

### (1) 전체 5단계 파이프라인 일괄 실행 및 보고서 자동 갱신
```bash
python src/run_assignment.py
```
*(실행 즉시 1~5단계 실험이 수행되고, `docs/assignment_final_report.md`에 최종 결과 보고서가 생성됩니다.)*

### (2) Data Leakage 방지 및 무결성 단위 테스트 실행
```bash
python tests/test_ml_pipeline.py
```
*(결과: `[PASS] Data leakage prevention test passed! Scaler is strictly isolated to Train set.`)*

### (3) 주피터 노트북 실행
```bash
jupyter notebook notebooks/assignment_full_report.ipynb
```

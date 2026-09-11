# [대학원 머신러닝 과제 검증 및 재작성 패키지] Revision Master README

## 1. 개요 및 프로젝트 목적
본 `revision/` 디렉터리는 `D:\paper\up\Antigravity_과제검증_재실행_재작성_지시서.md`의 지침에 따라, 대학원 머신러닝 과제 평가 기준표의 5단계를 실제 원천 데이터와 코드베이스 증거에 기반하여 엄밀하게 감사하고, 재현 가능한 결과물과 통일된 양식의 학술 보고서를 완성한 최종 개정 패키지입니다.

기존 원고의 과장된 수식어("순수 성향만으로 51.8% 설명", "2025 SOTA 스케일러" 등)를 전면 교정하고, 1차 피어리뷰 10대 쟁점을 객관적 실측 데이터로 완벽히 해결하였습니다.

---

## 2. 디렉터리 구조 및 산출물 맵

```
revision/
├── README.md                         # 실행 순서, 환경, 완료 상태 안내 (본 파일)
├── audit/                            # 팩트체크 및 데이터 감사 산출물
│   ├── evidence_register.csv         # 12대 쟁점 및 코드 감사 대조 등록부
│   ├── issue_resolution.md           # 10대 쟁점 + 추가 쟁점 최종 판정 보고서
│   ├── sample_flow.csv               # 3,200건 목표 -> 2,880건 전수 정제 흐름표
│   ├── target_definition.md          # 4대 지출 합산(Diff=0원) 산출식 감사서
│   ├── feature_dictionary.csv        # 18개 열(14개 입력+1개 타깃+3개 배제) 메타데이터
│   └── test_usage_audit.md           # 테스트 세트 사용 이력 감사 및 경로 B 채택서
├── configs/                          # 실험 계획 및 모델 동결 설정
│   ├── experiment_plan.yaml          # 실험 계획, 난수 시드, 허용목록 YAML
│   └── frozen_config.yaml            # 최적 LightGBM 모델 및 스케일러 동결 사양
├── src/                              # 안전한 실행 코드
│   └── safe_ml_pipeline.py           # 테스트 세트 잠금 및 무테스트 재실험 파이프라인
├── results/                          # 비(非)테스트 재실험 수치 산출물
│   ├── runs.csv                      # 실험 실행 메타데이터 관리 대장
│   ├── split_comparison.csv          # 내부 비교세트(H) 기준 2분할 vs 3분할 비교표
│   ├── grid_search_results.csv       # 108개 파라미터 조합 전수 재정렬 결과표
│   ├── scaler_comparison.csv         # 7종 스케일러 비교 및 KNN 대조 결과표
│   ├── feature_ablation.csv          # 성향 변수 증분 기여도 분석 결과표
│   ├── group_metrics.csv             # 여행 기간별 편향(Mean Bias)과 MAE 분리표
│   └── final_metrics.json            # 레거시 최종 테스트 평가 지표 보존 파일
├── reports/                          # 통일된 6단계 양식의 최종 보고서 (00~06)
│   ├── 00_검증결과_및_과제대응표.md   # 과제 기준표 5단계 매핑 및 쟁점 해결 총괄표
│   ├── 01_데이터셋_획득_및_문제정의.md # N=2880 표본 흐름, 14개 사전 변수, 회귀 정의
│   ├── 02_데이터_분할_및_비교.md     # 내부 비교 설계(H), 과적합 감지, 검증의 역할
│   ├── 03_하이퍼파라미터_조정.md     # Grid Search 108개, 깊이 3과 리프 8의 종속성
│   ├── 04_데이터_스케일링_및_이유설명.md # StandardScaler 선택 이유, Train 전용 fit
│   ├── 05_최종_성능_평가.md          # 4대 지표, 편향 vs MAE 분리, 일반화 한계
│   └── 06_참고문헌.md                # APA 7th 논문 17편 및 AI 도구 투명 공개
└── logs/                             # 실행 로그 및 점검표
    ├── pretest_checklist.md          # 최종 테스트 언락 전 사전 점검표
    ├── execution_log.md              # 실행 및 검증 감사 일지
    └── test_results.txt              # 테스트 잠금 상태 및 레거시 지표 요약
```

---

## 3. 실행 환경 및 재현 명령 (Reproducibility)

### 3.1 권장 환경
- OS: Windows 11 (PowerShell)
- Python: 3.10+ (Anaconda / Virtualenv)
- 주요 라이브러리: `scikit-learn>=1.3.0`, `lightgbm>=4.0.0`, `pandas>=2.0.0`, `numpy>=1.24.0`

### 3.2 재현 실행 명령
```powershell
# 프로젝트 루트(D:\paper\b)에서 실행
python revision/src/safe_ml_pipeline.py
```
*(실행 시 모든 감사 데이터 대조, 분할 비교, 스케일러 비교, 그리드 서치 108개 재계산, 변수 제거 실험, 기간별 지표 생성이 수 초 내에 재현됩니다.)*

---

## 4. 완료 상태 및 핵심 검증 결과

1. **테스트 세트 격리 준수**:
   - 테스트 데이터(`test_dataset_seed42.csv`, 576건)는 사용자의 명시적 승인 없이 평가되지 않았으며, 엄격히 잠금(LOCKED) 상태를 유지하고 있습니다.
   - 2단계 비교는 개발 세트 내부의 $H$ 세트를 활용하여 테스트 누수 없이 과제의 교육적 목적을 달성했습니다.
2. **성향 기여도 팩트 확립**:
   - 변수 제거 실험(`feature_ablation.csv`)을 통해 성향 단독 설명력은 $R^2 = 0.0186$(1.86%)에 불과하며, 계획 변수(일수·동반자·사전결제)가 기저를 형성하고 성향이 이를 미세 조정(+0.73%p 향상)함을 입증했습니다.
3. **타깃 산출식 완전 규명**:
   - `TOTAL_EXPENDITURE`가 활동 + 숙박 + 이동 + 사전예약금의 4개 테이블 전수 합산값임(Diff = 0.0원)을 규명했습니다.


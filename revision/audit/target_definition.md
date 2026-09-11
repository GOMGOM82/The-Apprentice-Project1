# [타깃 변수 정의 및 산출식 감사 보고서] TOTAL_EXPENDITURE

## 1. 개요 및 산출식 진실 규명

기존 원고 및 1차 요청서에서는 "활동 지출과 사전 예약금만 합산하고, 숙박과 이동비는 중복/누락 우려로 배제했다"고 서술되었으나, **실제 코드베이스(`src/data/travel_dataset.py`)와 원천 데이터 대조 결과 이는 사실과 다름이 확인**되었습니다.

### 실제 구현 산출식 (Ground Truth):
$$\text{TOTAL\_EXPENDITURE} = \text{ACT\_SUM} + \text{LOD\_SUM} + \text{MVM\_SUM} + \text{ADV\_SUM}$$

- **실제 소스코드 (`src/data/travel_dataset.py` lines 40-44)**:
  ```python
  total_exp = (
      act_sum.add(lod_sum, fill_value=0)
      .add(mvm_sum, fill_value=0)
      .add(adv_sum, fill_value=0)
  ).rename("TOTAL_EXPENDITURE")
  ```
- **실측 검증 결과**: 원천 4개 테이블의 여행별(`TRAVEL_ID`) 결제 금액 합산값과 `real_travel_dataset.csv`의 `TOTAL_EXPENDITURE` 간 편차(Diff)는 **전체 2,880건에서 정확히 0.0원**임이 수학적으로 증명되었습니다.

---

## 2. 4대 구성요소별 기술 통계량 ($N = 2,880$)

| 항목 | 영문 변수명 | 원천 테이블 | 평균값 (KRW) | 중앙값 (KRW) | 최댓값 (KRW) | 전체 대비 비중 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **활동 지출** | `ACTIVITY_CONSUME_KRW` | `tn_activity_consume_his.csv` | 177,123.5 원 | 127,155 원 | 5,297,600 원 | 51.52% |
| **숙박 지출** | `LODGE_CONSUME_KRW` | `tn_lodge_consume_his.csv` | 80,293.5 원 | 0 원 | 1,764,000 원 | 23.35% |
| **이동 지출** | `MVMN_CONSUME_KRW` | `tn_mvmn_consume_his.csv` | 74,901.0 원 | 56,950 원 | 1,101,600 원 | 21.79% |
| **사전 지출** | `ADV_CONSUME_KRW` | `tn_adv_consume_his.csv` | 11,491.1 원 | 0 원 | 880,000 원 | 3.34% |
| **총합계 (타깃)** | **`TOTAL_EXPENDITURE`** | **4개 테이블 통합 합산** | **343,809.0 원** | **248,198 원** | **5,340,850 원** | **100.00%** |

*(※ 당일 여행자가 포함되어 있어 숙박 지출의 중앙값은 0원이며, 사전 예약이 없는 여행자도 다수 존재함)*

---

## 3. 학술적 성격 및 한계 규정 (Naming & Limitations)

1. **명칭의 객관화**:
   - 본 변수를 '대한민국 전체 여행 소비의 완전한 총경비'라고 과장하지 않고, **"사전 예약 결제액과 현장 활동·숙박·이동 영수증 결제 내역을 체계적으로 집계한 관측 총소비액(Observed Total Expenditure)"**으로 명확히 규정합니다.
2. **사전 지출액(`ADV_CONSUME_KRW`)과 타깃 간의 관계**:
   - `ADV_CONSUME_KRW`는 타깃의 구성요소(약 3.34%)로 포함되어 있으면서 동시에 모델의 입력변수로도 사용됩니다.
   - 이는 여행 출발 전 확정된 예약금 정보를 활용하여 최종 총지출액을 예측하는 실무적 시나리오(예: 여행사 사전 예약 기반 총경비 추정)에서는 지극히 자연스러운 **사전 계획 특성 변수**입니다.
   - 보고서에서는 이를 숨기지 않고, "타깃의 일부 구성요소가 알려진 상태에서의 조건부 예측(Conditional Prediction with Known Component)"임을 솔직하게 기술합니다.
3. **영수증 미인증 소비의 한계**:
   - 영수증이 발행되지 않은 현금 거래나 설문 미기입 내역은 관측되지 않을 수 있음을 연구 한계로 명시합니다.


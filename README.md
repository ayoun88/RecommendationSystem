# 🛍️ Commerce Behavior Purchase Prediction
목적 : 온라인 쇼핑몰 유저의 행동 데이터(view·cart·purchase)를 분석하여, 다음 1주일(next 1 week) 안에 유저가 구매할 상품 Top-10을 예측하는 추천 시스템 파이프라인 구현

---

## 📂 ReadME Index
[🎯 Project Overview (프로젝트 개요 및 목표)](#project-overview)

[⏱️ Project Duration & 🔧 Tech Stack (기간 및 기술스택)](#projectduration-techstack)

[📊 Data Analysis & Hypothesis (데이터 분석 및 실험 방향성 설정)](#data-analysis)

[🚀 Experimental Progression (실험 과정 및 빌드업)](#experimental-progression)

[🧪 Final SOTA & Experiment Results (핵심 실험 결과 전체)](#final-sota)

[🛠️ Troubleshooting & Engineering (문제 해결 및 인프라 안정화)](#troubleshooting-engineering)

[📈 Retrospective & Future Work (회고 및 향후 계획)](#retrospective-futurework)

---

<a id="project-overview"></a>

## 🎯 Project Overview

### 프로젝트 배경
이커머스 환경에서 추천 시스템은 사용자의 쇼핑 패턴, 관심사, 과거 구매 이력을 분석해 맞춤형 상품을 추천함으로써 사용자 경험을 높이고 기업의 매출 향상에 기여합니다. 본 프로젝트는 638,257명 유저의 4개월치 행동 로그를 기반으로, **협업 필터링(Collaborative Filtering) 모델을 고도화해 다음 1주일간의 구매 상품을 예측하는 추천 파이프라인**을 구축했습니다.

### 핵심 과제
전체 이벤트의 99.8%가 단순 조회(view)이고 실제 구매(purchase)는 0.02%에 불과한 **극단적 행동 불균형**, 그리고 전체 유저의 99.7%가 구매 이력이 없는 **Cold-Start 환경**에서, ALS와 SASRec 두 협업 필터링 모델을 각각 고도화하고 앙상블하여 NDCG@10을 최적화하는 추천 파이프라인 설계

### 핵심 평가 지표

평가 지표는 **NDCG@10 (Binary Relevance 기반)** 입니다.

```python
def calc_ndcg10(gt_purchases, pred_top10):
    for user_id, items in pred_top10.items():
        relevance = [1 if item in gt_purchases[user_id] else 0 for item in items]
        ndcg = compute_ndcg(relevance, k=10)   # 실제 구매(1) / 미구매(0)
```

- **제출 조건** : 학습 데이터의 모든 유저(638,257명)에게 10개씩 중복 없이 추천 → 총 6,382,570행 제출 필요
- **동점 처리** : NDCG@10이 동일하면 제출 횟수가 적을수록 높은 순위 할당
- **구조적 난이도** : 실제 구매 유저가 1,681명(0.26%)뿐인 데이터이므로, 점수 자체가 구조적으로 낮게 형성되는 환경 — 모델 간 미세한 차이가 곧 순위 차이로 직결됨

---

<a id="projectduration-techstack"></a>

## ⏱️ Project Duration & 🔧 Tech Stack

### ⏱️ Project Duration
- 2026.05.07 ~ 2026.05.14 (8일)

### 🔧 Tech Stack
| Category | Tech Stack |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Collaborative Filtering** | `implicit` (AlternatingLeastSquares) |
| **Sequential Recommendation** | RecBole (`SASRec`) |
| **Data Processing** | Pandas, NumPy, SciPy (`csr_matrix`) |
| **Framework** | PyTorch (RecBole backend) |
| **Environment** | Ubuntu, GPU Server (CUDA) |
| **Collaboration** | Slack, Notion |

---

<a id="data-analysis"></a>

## 📊 Data Analysis & Hypothesis

직관이 아닌 **통계적 근거를 바탕**으로 실험 방향성을 수립하기 위해, 8,350,311건의 행동 로그를 EDA를 통해 먼저 분석했습니다.

### Insight 1. 이벤트의 극단적 불균형 → Event 가중치 설계의 출발점

- **분석** : view(99.8%) : cart(0.2%) : purchase(0.02%) = **4015:8:1**의 극단적 비율. 실제 구매까지 이어진 유저는 1,681명(0.26%)뿐이며, purchase 기준 matrix density는 거의 0%에 가까운 수준입니다.
- **실험 방향** : 단순 빈도 기반 신호로는 구매 의도를 포착하기 어렵다고 판단해, 이벤트 별 confidence를 강하게 차등 부여(view=1, cart=100, purchase=500)하는 전략을 채택하고, ALS의 `factor`·`alpha` 등 모델 표현력 확장 여지를 함께 탐색하기로 했습니다.

### Insight 2. 브랜드 익명화 → Feature Engineering 포기, 협업필터링 집중

- **분석** : 카테고리 코드가 전부 `apparel.*` 단일 도메인이며, 브랜드명은 삼성·샤오미 등 전자기기 브랜드로 익명화되어 있어 실제 브랜드-상품 간 연관성이 존재하지 않습니다. 외부 이커머스 데이터에서 apparel만 필터링한 뒤 브랜드명을 임의 치환한 것으로 추정됩니다.
- **실험 방향** : brand/category를 활용한 LightGBM 2단계 re-ranking, Wide & Deep, Factorization Machine 등 side information 기반 접근을 모두 배제하고, **순수 상호작용 패턴 기반의 협업필터링(ALS, SASRec) 고도화**에 집중하기로 결정했습니다.

### Insight 3. 구매 시점의 극단적 집중 → 시간 감쇠 도입

- **분석** : 월별 purchase는 11월 94건 → 12월 137건 → 1월 260건 → 2월 1,585건으로 급증했고, **Week 9(2월 말)에 전체 purchase의 약 70%**가 집중되어 있습니다. 4개월이라는 짧은 수집 기간이지만 명확한 트렌드 변화가 존재합니다.
- **실험 방향** : 오래된 행동일수록 confidence를 낮추는 시간 감쇠(`confidence × exp(-λ × days_ago)`)를 ALS 신뢰도 계산에 도입하여, 최근 구매 패턴에 더 민감하게 반응하도록 설계했습니다.

### Insight 4. 99.7% Cold-Start → Fallback 전략의 필요성과 한계

- **분석** : 전체 유저의 99.7%가 purchase 이력이 없습니다. cart 이력이 있는 유저도 2월 기준 4,443명뿐이지만, **cart → purchase 전환율은 12.69%**로 view(0.02%) 대비 월등히 높습니다.
- **실험 방향** : SASRec의 `inter_num_interval` 임계값을 낮춰 cold-start 비율을 축소하고(42%→27.5%), 남은 cold-start 유저에게는 cart 미구매 이력을 우선 활용하는 fallback을 설계해 단순 popular 추천 대비 개인화 여지를 확보하고자 했습니다.

---

<a id="experimental-progression"></a>

## 🚀 Experimental Progression

총 16회의 점진적 실험을 통해 NDCG@10 **0.0844 → 0.1256**으로 성능을 끌어올렸습니다.

### Phase 1. 베이스라인 분석 및 ALS 초기 실험 (05.07)

- **베이스라인 점검** : ALS와 SASRec의 public score는 큰 차이가 없으나 학습 시간이 크게 차이남을 확인하고, **ALS 위주로 빠르게 반복 실험**하기로 결정했습니다(NDCG 0.0844).
- **표현력 vs 가중치** : *"latent factor 차원을 최대한 크게 가져가는 것이 성능 향상에 가장 효과적일까?"* 라는 의문으로 factor를 32→512까지 확장한 결과(0.1028)가, 단순 event 가중치만 부여한 결과(1:5:20, 0.1004)보다 더 효과적임을 확인했습니다. 데이터 희소성 환경에서는 정교한 가중치 설계보다 **표현 공간 확보가 우선**임을 파악한 단계입니다.

### Phase 2. EDA 기반 ALS 고도화 — 가중치 + 시간 감쇠 (05.11)

- EDA에서 확인한 event 불균형(4015:8:1)을 근거로 event 가중치를 `view=1, cart=100, purchase=500`으로 강화했습니다.
- Week 9 구매 급등을 확인한 뒤 시간 감쇠 λ를 0.03 → 0.1 → 0.3 순으로 탐색해, **λ=0.1에서 NDCG 0.1208로 ALS 단독 최고점**을 달성했습니다.
- `filter_already_liked_items=True`를 시도했으나 view(99.8%)까지 전부 필터링되어 추천 후보가 고갈되는 문제를 확인하고 `False`로 유지하기로 결정했습니다.
- 희소 데이터에 강하다고 알려진 **EASE**를 시도했으나, 극단적 event 가중치가 행렬 역산 과정에서 수치 불안정을 유발해 NDCG가 0.0277까지 하락 — 도입을 포기했습니다.

### Phase 3. SASRec 고도화 — 구조 튜닝과 Cold-Start Fallback (05.13~14)

- ALS(전체 상호작용 패턴)와 SASRec(최근 시퀀스 패턴)의 상호보완을 기대하고, SASRec 단독 성능 개선에 착수했습니다.
- `n_heads`를 8→4로 조정해 head당 표현 차원을 32로 확보했고, 모델 크기 확장 중 GPU OOM이 발생하자 `batch_size`를 4096→2048로 낮췄다가, 이후 `n_layers`를 줄여 메모리 여유를 확보한 뒤 4096으로 복구했습니다.
- `inter_num_interval`을 5→3으로 낮춰 cold-start 비율을 42%→27.5%로 축소(0.0842→0.0930)했습니다.
- cart → purchase 전환율(12.69%)에 근거해 **3단계 fallback**(cart→서브카테고리 인기→전체 인기)을 시도했으나 오히려 노이즈로 작용해 성능이 하락(0.0898)했고, **2단계로 단순화**(cart→전체 인기)한 결과 SASRec 단독 최고점인 **0.0933**을 기록했습니다.

### Phase 4. ALS + SASRec 앙상블 (05.14, 최종 SOTA)

- ALS가 SASRec보다 단독 성능이 압도적으로 높았기 때문에, 단순 평균이 아닌 가중치를 명시적으로 조절할 수 있는 **Weighted Rank Score Fusion**을 채택했습니다.
- ALS 0.7 : SASRec 0.3 가중치 조합에서 **NDCG@10 0.1256**으로 최종 SOTA를 달성했습니다(0.6:0.4 → 0.1251, 0.8:0.2 → 0.1252로 모두 0.7:0.3보다 낮음을 확인).

---

<a id="final-sota"></a>

## 🧪 Final SOTA & Experiment Results

### 🏆 Final SOTA 아키텍처

```
[ALS 파이프라인]                              [SASRec 파이프라인]
train.parquet (8.35M rows)                    train.parquet
        ↓                                              ↓
event 가중치(1:100:500)                        recbole_dataset.py 변환
+ 시간 감쇠(λ=0.1)                                      ↓
        ↓                                     SASRec (hidden=128, n_heads=4)
implicit.ALS (factors=512)                              ↓
        ↓                                     상호작용 3회+ → 모델 추론
output_als.csv                                cold-start → 2단계 fallback
NDCG@10 = 0.1208                                        ↓
        │                                     output_sasrec.csv
        │                                     NDCG@10 = 0.0933
        └──────────────┬──────────────────────────────┘
                        ↓
          ensemble.py — Weighted Rank Score Fusion
          final_score = ALS×0.7 + SASRec×0.3
                        ↓
              output_ensemble.csv
              NDCG@10 = 0.1256  ⭐ 최종 SOTA
```

---

### 📊 전체 실험 결과 테이블

| 버전 | NDCG@10 | 핵심 변경 | 결과 | 인사이트 |
|------|---------|-----------|------|----------|
| ALS_baseline | 0.0844 | factor=32, label=1 (기본) | — | ALS·SASRec 학습 시간 차이를 확인하고 ALS 위주 실험으로 전환 |
| ALS_factor256 | 0.1000 | factor 256으로 증가 | ⬆️ | factor 확장이 즉각적인 성능 향상으로 이어짐 |
| v2_event_weighted | 0.1004 | view=1, cart=5, purchase=20 | ⬆️ | 단순 가중치 부여만으로는 factor 확장 대비 효과 미미 |
| v3_factor512 | 0.1028 | factor=512, label=1 | ⬆️ | 표현력 확보가 가중치 설계보다 선행 과제임을 확인 |
| v4_weighted_decay03 | 0.1156 | weight(1:100:500) + time_decay=0.03 | ⬆️ | event 가중치와 시간 감쇠 결합이 단일 요소보다 효과적 |
| **v5_decay01** | **0.1208** | **time_decay=0.1** | ⬆️ **ALS 단독 최고** | 30일 전 데이터를 현재의 약 5%로 감쇠하는 강도가 최적 |
| v6_decay03 | 0.1165 | time_decay=0.3 | ⬇️ | 감쇠가 과해지면 유효 학습 데이터가 줄어 오히려 하락 |
| v8_EASE | 0.0277 | EASE λ=500, decay=0.1 | ⬇️ | 극단적 가중치가 행렬 역산 시 수치 불안정 유발, 도입 포기 |
| v9_SASRec_baseline | 0.0842 | RecBole 기본 config, inter_num=5 | — | cold-start 42%가 동일한 popular 추천을 받는 구조 확인 |
| v9-1_SASRec_heads8 | 0.0825 | hidden=128, n_heads=8, batch=2048 | ⬇️ | head당 16차원으로 표현력 부족, OOM 회피용 batch 축소도 효과 없음 |
| v9-2_SASRec_heads4 | 0.0930 | n_heads=4, inter_num 5→3 | ⬆️ | head당 32차원 확보 + cold-start 비율 축소(42%→27.5%)가 동시에 기여 |
| v9-3_SASRec_3step | 0.0898 | cart→서브카테고리 인기→전체 인기 | ⬇️ | cold-start 유저는 cart·카테고리 이력 자체가 부족해 단계 추가가 노이즈로 작용 |
| **v9-4_SASRec_2step** | **0.0933** | **cart→전체 인기 (단순화)** | ⬆️ **SASRec 단독 최고** | fallback 단계를 줄이는 단순화가 오히려 성능을 개선 |
| **v10-1_Ensemble_7-3** | **0.1256** | **ALS×0.7 + SASRec×0.3** | ⬆️ **최종 SOTA** | 전체 패턴(ALS)과 최근 시퀀스(SASRec)의 상호보완 효과 입증 |
| v10-2_Ensemble_6-4 | 0.1251 | ALS×0.6 + SASRec×0.4 | ⬇️ | SASRec 비중을 높일수록 약한 단독 모델의 영향이 커져 하락 |
| v10-3_Ensemble_8-2 | 0.1252 | ALS×0.8 + SASRec×0.2 | ⬇️ | SASRec 비중을 과도하게 낮추면 상호보완 효과가 줄어듦 |

---

<a id="troubleshooting-engineering"></a>

## 🛠️ Troubleshooting & Engineering

### 1. EASE 모델의 수치 불안정으로 인한 성능 붕괴

#### 문제 정의
희소 데이터에 강하다고 알려진 EASE(closed-form 행렬 역산 모델)를 도입했으나, NDCG가 0.0277로 베이스라인(0.0844)보다 크게 하락했습니다.

#### 원인 분석
두 가지 요인이 동시에 작용했습니다. 첫째, view(99.8%) 이벤트를 필터링하는 과정에서 추천 후보 자체가 고갈되었습니다. 둘째, purchase=500이라는 극단적 가중치가 `X^TX` 행렬 역산 과정에서 수치 불안정을 유발했습니다.

#### 해결 방안
EASE 도입을 포기하고 ALS·SASRec 고도화에 집중했습니다. 파이프라인이 올바르게 구성되었는지 자체를 판단하기 어려운 수준의 불안정성이었기 때문에, 추가 디버깅보다 검증된 모델 계열에 시간을 투자하는 것이 효율적이라고 판단했습니다.

#### 인사이트
"희소 데이터에 강한 모델"이라는 일반적 특성이, 이 데이터처럼 **이벤트 가중치 자체가 극단적인 경우**에는 그대로 적용되지 않을 수 있습니다. 모델의 이론적 강점과 실제 데이터의 구조적 특성이 충돌할 수 있다는 점을 확인했습니다.

---

### 2. `filter_already_liked_items=True` 적용 시 제출 행 수 부족

#### 문제 정의
재구매율이 7%로 낮다는 EDA 결과를 바탕으로 이미 본 상품을 추천에서 제외(`filter_already_liked_items=True`)해보았으나, 일부 유저의 추천 슬롯이 10개 미만이 되어 총 제출 행 수(6,382,570)를 채우지 못하는 Assertion 오류가 발생했습니다.

#### 원인 분석
view 이벤트가 전체의 99.8%를 차지하는 데이터에서 "이미 본 상품"을 필터링하면, 대부분의 유저에게 추천할 후보 자체가 사라집니다. 이커머스에서는 view → 고민 → purchase로 이어지는 흐름이 자연스러우므로, 관심 있던 상품을 제거하는 것이 오히려 손해라는 점도 함께 확인했습니다.

#### 해결 방안
미달 슬롯을 인기 아이템으로 채우는 패딩 로직으로 임시 조치한 뒤, 이후 모든 실험에서는 `filter_already_liked_items=False`로 고정했습니다.

#### 인사이트
도메인 특성(재구매율)만으로 파라미터를 결정하면 데이터의 다른 구조적 특성(이벤트 비율)과 충돌할 수 있습니다. 파라미터 변경 전 **제출 포맷 제약(고정된 추천 개수)** 까지 함께 고려해야 한다는 것을 배웠습니다.

---

### 3. 3단계 Cold-Start Fallback의 역효과

#### 문제 정의
cart → purchase 전환율(12.69%)이 높다는 근거로 3단계 fallback(cart 미구매 → 서브카테고리 인기 → 전체 인기)을 설계했으나, 2단계 대비 성능이 0.0930 → 0.0898로 하락했습니다.

#### 원인 분석
cold-start 유저(상호작용 3회 미만)는 cart 이력뿐 아니라 서브카테고리 이력도 극도로 부족합니다. fallback 단계를 늘릴수록 신뢰할 수 없는 적은 이력에 기반한 추천이 추가될 뿐, 실제 개인화 신호로 이어지지 않았습니다.

#### 해결 방안
fallback을 cart 미구매 → 전체 인기의 2단계로 단순화해 NDCG 0.0933을 기록했습니다.

#### 인사이트
**더 많은 fallback 단계 = 더 정교한 개인화가 아닙니다.** 각 단계가 의지할 수 있는 데이터가 충분한지 먼저 확인해야 하며, 데이터가 부족한 환경에서는 단순한 구조가 오히려 노이즈를 줄여줍니다.

---

### 4. GPU OOM — SASRec 모델 확장 시 메모리 부족

#### 문제 정의
SASRec의 `hidden_size`, `n_layers`를 늘려 모델 표현력을 확장하는 과정에서 GPU Out-of-Memory가 발생했습니다.

#### 원인 분석
`hidden_size=128`에 `n_layers=3`을 함께 적용하면서 Transformer 레이어 수와 batch 크기(4096)가 결합되어 메모리 사용량이 한계를 초과했습니다.

#### 해결 방안
`train_batch_size`를 4096→2048로 낮춰 우선 OOM을 해결했고, 이후 `n_layers`를 2로 줄여 메모리 여유가 생기자 batch_size를 다시 4096으로 복구해 학습 효율을 유지했습니다.

#### 인사이트
모델 구조(레이어 수, hidden size)와 학습 설정(batch size)은 독립적인 파라미터가 아니라 **GPU 메모리라는 공유 자원 안에서 함께 조정해야 하는 트레이드오프** 관계임을 확인했습니다.

---

### 5. 기타 환경 설정 이슈

| 문제 | 원인 | 해결 |
|------|------|------|
| **제출 시 Network Error 다수 발생** | 제출 파일 크기가 약 500MB로 브라우저 업로드 타임아웃 발생 | 재시도로 해결 |

---

<a id="retrospective-futurework"></a>

## 📈 Retrospective & Future Work

### 📌 회고
ALS와 SASRec의 학습 시간 차이를 베이스라인 단계에서 빠르게 파악하고 실험 우선순위를 조정한 것이, 짧은 기간 안에 성능을 끌어올리는 데 핵심적인 역할을 했습니다. EDA에서 확인한 사실들(이벤트 불균형, 브랜드 익명화, 구매 시점 집중)이 이후 모든 모델링 의사결정의 직접적인 근거가 되었다는 점에서, 가설을 세우기 전에 데이터를 충분히 들여다보는 과정의 중요성을 체감했습니다.

### 📌 아쉬운 점
- cold-start 유저가 99.7%에 달하는 근본적인 데이터 한계로 인해, fallback 전략을 정교하게 설계해도 개선 폭이 제한적이었습니다. popular 추천을 강력한 baseline으로 먼저 받아들이고 그 위에 점진적으로만 개인화를 더하는 접근이 더 효율적이었을 것이라고 뒤늦게 체감했습니다.
- 브랜드·카테고리 등 side information이 사실상 사용 불가능한 데이터라는 점을 EDA 초반에 확인했지만, 이를 검증하기 전까지 feature 기반 모델(LightGBM 등) 검토에 일부 시간을 소요했습니다. 데이터의 신뢰 가능 여부를 가장 먼저 검증하는 순서가 더 효율적이었을 것입니다.
- 앙상블 가중치(ALS:SASRec) 탐색을 0.7/0.3 근방의 3개 값으로만 진행했는데, 더 넓은 grid 또는 cold-start 여부에 따른 동적 가중치를 시도해볼 여지가 있었습니다.

### 📗 향후 계획
- cold-start 유저 비율이 압도적인 데이터 특성을 고려해, 노출 빈도에 페널티를 주는 방식으로 popularity bias를 보정하는 re-ranking 기법을 적용해보고 싶습니다.
- ALS·SASRec의 출력을 입력으로 받아 cold-start 여부에 따라 가중치를 동적으로 학습하는 경량 Re-ranker(LightGBM 등) 구조를 실험해보고 싶습니다.
- 이번 대회에서 구축한 시간 감쇠 + 이벤트 가중치 ALS 파이프라인을 실제 커머스 도메인 추천 시스템 프로토타입에 적용해, A/B 테스트 기반 온라인 평가까지 확장해보고 싶습니다.
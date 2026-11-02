---
title: "Q-41. 모델 정확도를 평가할 때 가장 자주 사용되는 metric은 무엇인가요?"
date: 2026-11-02 12:41:00 +0900
category: study
topic: basic
series: knowledge
lecture: 41
card_title: "Q 41: Common Accuracy Metrics"
card_desc: "Accuracy · Precision · Recall · F1 · ROC-AUC · RMSE"
lang: ko
description: "분류는 Accuracy, Precision, Recall, F1, ROC-AUC, Log loss를, 회귀는 MAE, MSE, RMSE, R²를 주로 쓰며, 데이터의 불균형과 오류 비용에 따라 고른다."
---

## 면접 답변

> 분류에서는 전체 중 맞힌 비율인 Accuracy가 가장 기본이지만, 불균형 데이터에서는 Precision, Recall, F1, ROC-AUC를 함께 봅니다.
> Accuracy는 모든 오류를 같은 비용으로 취급하고 클래스 비율에 크게 좌우되기 때문입니다.
> 예를 들어 양성이 1%인 데이터에서는 전부 음성으로 찍어도 Accuracy가 99%이지만, 양성은 하나도 찾지 못합니다.
> 회귀에서는 '맞혔다'는 개념 대신 오차의 크기를 보므로 MAE, RMSE, 결정계수 같은 지표를 씁니다.
> 그래서 실무에서는 문제의 오류 비용에 맞는 지표를 먼저 정하고, 그 지표로 모델을 비교합니다.

## 자세한 설명

### 분류 지표

TP, FP, FN, TN은 [혼동 행렬](/blog/2026/11/02/knowledge-17-confusion-matrix/)의 네 칸이다.

| 지표 | 정의 | 무엇을 묻는가 | 언제 중요한가 |
|---|---|---|---|
| Accuracy | $$\dfrac{TP + TN}{TP + TN + FP + FN}$$ | 전체 중 맞힌 비율 | 클래스가 균형이고 오류 비용이 비슷할 때 |
| Precision | $$\dfrac{TP}{TP + FP}$$ | 양성 예측 중 진짜 양성 비율 | FP의 비용이 클 때 (스팸 차단) |
| Recall | $$\dfrac{TP}{TP + FN}$$ | 실제 양성 중 찾아낸 비율 | FN의 비용이 클 때 (암 진단) |
| F1 | $$\dfrac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$ | 둘의 조화 평균 | 둘의 균형이 필요할 때 |
| ROC-AUC | ROC 곡선 아래 면적 | 양성이 음성보다 높은 점수를 받을 확률 | 임계값과 무관하게 순위를 볼 때 |
| Log loss | $$-\dfrac{1}{n}\sum_i \left[y_i \log p_i + (1 - y_i)\log(1 - p_i)\right]$$ | 예측 확률의 품질 | 확률값 자체를 쓸 때 |

Log loss에서 $$y_i \in \{0, 1\}$$은 정답, $$p_i$$는 양성으로 예측한 확률이다.

**예제.** 샘플 1,000개에서 $$TP = 5$$, $$FN = 5$$, $$FP = 10$$, $$TN = 980$$이라고 하자.

- Accuracy = $$985/1000 = 0.985$$
- Precision = $$5/15 \approx 0.333$$
- Recall = $$5/10 = 0.5$$
- F1 = $$2 \times 0.333 \times 0.5 / 0.833 = 0.4$$

항상 음성으로 예측하는 모델의 Accuracy는 $$990/1000 = 0.99$$로 이 모델보다 높다. 하지만 Recall은 0이다. Accuracy만 보면 쓸모없는 모델을 고르게 된다.

**Log loss 예제.** 정답 $$(1, 1, 0)$$에 양성 확률 $$(0.9, 0.6, 0.2)$$를 예측하면 Log loss는 약 $$0.280$$이다. 임계값 0.5에서 Accuracy는 1이지만, Log loss는 0.6처럼 확신이 약한 예측에 벌점을 준다.

### 회귀 지표

$$y_i$$는 정답, $$\hat{y}_i$$는 예측, $$\bar{y}$$는 정답의 평균이다.

| 지표 | 정의 | 특징 |
|---|---|---|
| MAE | $$\frac{1}{n}\sum_i \lvert y_i - \hat{y}_i \rvert$$ | 타깃과 같은 단위. 이상치에 상대적으로 강함 |
| MSE | $$\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$$ | 큰 오차에 큰 벌점. 단위가 제곱 |
| RMSE | $$\sqrt{\text{MSE}}$$ | 타깃과 같은 단위. 큰 오차에 민감 |
| $$R^2$$ | $$1 - \dfrac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$ | 평균 예측 대비 개선 정도. 음수도 가능 |

**예제.** $$y = (3, 5, 2, 7)$$, $$\hat{y} = (2.5, 5, 4, 8)$$이면 오차는 $$(0.5, 0, -2, -1)$$이다.

- MAE = $$3.5/4 = 0.875$$
- MSE = $$5.25/4 = 1.3125$$
- RMSE $$\approx 1.146$$
- $$R^2 = 1 - 5.25/14.75 \approx 0.644$$

### 다중 클래스

Precision, Recall, F1은 클래스별로 계산한 뒤 평균한다. macro는 클래스를 동등하게, weighted는 표본 수에 비례해 평균한다. 샘플마다 레이블이 하나일 때 micro 평균 F1은 Accuracy와 같다.

## 관련 개념

- **Accuracy와 Performance**: Accuracy는 성능 척도의 하나일 뿐이다. [Q-31](/blog/2026/11/02/knowledge-31-accuracy-vs-performance/)에서 다룬다.
- **F1-score**: 조화 평균을 쓰는 이유는 [Q-27](/blog/2026/11/02/knowledge-27-f1-score/)에서 다룬다.
- **ROC curve**: 임계값 전체에 걸친 평가이다. [Q-38](/blog/2026/11/02/knowledge-38-roc-curve/)에서 정의한다.
- **유용한 metric 전반**: 지표 선택의 큰 그림은 [Q-22](/blog/2026/11/02/knowledge-22-useful-metrics/)에서 이어진다.

## 꼬리 질문

**Q.** 양성 비율이 1%인 사기 탐지에서는 어떤 지표를 보겠습니까?
Accuracy는 제외하고, Precision과 Recall을 PR 곡선 또는 PR-AUC로 봅니다. 운영 단계에서는 사기를 놓치는 비용과 정상 거래를 막는 비용으로 임계값을 정합니다.

**Q.** RMSE와 MAE 중 어느 것을 쓸지는 어떻게 정하나요?
큰 오차를 특히 피해야 하면 RMSE를, 이상치의 영향을 줄이고 평균적인 오차를 보려면 MAE를 씁니다. MSE를 최소화하는 상수 예측은 평균, MAE를 최소화하는 상수 예측은 중앙값입니다.

**Q.** $$R^2$$가 음수일 수 있나요?
네. 테스트 데이터에서 모델이 평균 예측보다 나쁘면 음수가 됩니다.

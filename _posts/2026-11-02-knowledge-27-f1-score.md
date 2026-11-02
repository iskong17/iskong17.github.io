---
title: "Q-27. F1-score를 정의해 주세요"
date: 2026-11-02 12:27:00 +0900
category: study
topic: basic
series: knowledge
lecture: 27
knowledge_group: machine-learning
card_title: "Q 27: F1-score"
card_desc: "Precision · Recall · Harmonic mean · F-beta · Macro and micro averaging"
lang: ko
description: "F1-score는 precision과 recall의 조화 평균으로, 둘 중 낮은 값에 크게 끌려가므로 양성 클래스가 드문 문제에서 균형을 보는 데 유용하다."
---

## 면접 답변

> F1-score는 precision과 recall의 조화 평균입니다. 조화 평균을 쓰는 이유는 둘 중 하나만 높은 모델이 좋은 점수를 받지 못하게 하기 위해서입니다. 예를 들어 precision이 0.9이고 recall이 0.1이면 산술 평균은 0.5이지만 F1은 0.18입니다. 그래서 양성 클래스가 드물어 accuracy가 큰 의미를 갖지 못할 때 자주 사용합니다. 다만 F1은 true negative를 전혀 반영하지 않고, precision과 recall을 동등하게 취급합니다. 따라서 어느 한쪽 오류의 비용이 더 크다면 F-beta score나 비용 기반 지표를 함께 검토해야 합니다.

## 자세한 설명

### 정의

confusion matrix의 TP(true positive), FP(false positive), FN(false negative)으로 **precision**(양성 예측 중 실제 양성 비율)과 **recall**(실제 양성 중 찾아낸 비율)을 정의한다.

$$
\text{Precision} = \frac{TP}{TP + FP}, \qquad \text{Recall} = \frac{TP}{TP + FN}
$$

F1-score는 이 둘의 조화 평균이다. 범위는 0에서 1이다.

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}
$$

오른쪽 형태를 보면 TN(true negative)이 식에 없다. 음성을 아무리 많이 맞혀도 F1은 변하지 않는다.

### 계산 예

$$TP = 80$$, $$FP = 20$$, $$FN = 40$$이라면 다음과 같다.

- Precision $$= 80/100 = 0.8$$
- Recall $$= 80/120 \approx 0.667$$
- $$F_1 = 160/(160 + 20 + 40) \approx 0.727$$

### F-beta

오류 비용이 다르면 $$\beta$$로 recall의 비중을 조절한다.

$$
F_\beta = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

$$\beta > 1$$이면 recall을, $$\beta < 1$$이면 precision을 더 중시한다. 위의 예에서 recall이 precision보다 낮으므로, recall을 중시하는 $$F_2 \approx 0.690$$은 $$F_1$$보다 낮고 precision을 중시하는 $$F_{0.5} \approx 0.769$$는 $$F_1$$보다 높다.

### 다중 클래스 평균

클래스 A, B로 이루어진 110개 표본에서 A 100개 중 90개, B 10개 중 5개를 맞혔다고 하자. 이때 클래스별 F1은 A가 $$180/195 \approx 0.923$$, B가 $$10/25 = 0.4$$이다.

| 방식 | 계산 | 값 | 의미 |
|---|---|---|---|
| Macro | 클래스별 F1의 단순 평균 | 0.662 | 소수 클래스도 동등하게 반영 |
| Micro | TP, FP, FN을 모두 합쳐 계산 | 0.864 | 많은 클래스의 영향이 큼 |
| Weighted | 클래스별 F1을 표본 수로 가중 평균 | 0.876 | 불균형을 표본 수로 반영 |

각 표본에 라벨이 하나인 다중 클래스 문제에서 micro F1은 accuracy와 같다. 이 예에서도 $$95/110 \approx 0.864$$로 일치한다.

### 한계

- 임계값에 의존한다. 분류기의 점수를 어디서 자르느냐에 따라 F1이 달라지므로, 검증 데이터로 임계값을 정하거나 PR 곡선을 함께 본다.
- 양성의 정의에 의존한다. 어느 클래스를 양성으로 두느냐에 따라 값이 바뀐다.
- 비용을 직접 반영하지 않는다. FP와 FN의 비용을 알고 있다면 기대 비용으로 평가하는 편이 낫다.

## 관련 개념

- **Confusion matrix**: TP, FP, FN, TN을 정리한 표로, F1의 재료이다. [Q-17](/blog/2026/11/02/knowledge-17-confusion-matrix/)에서 다룬다.
- **Recall과 TPR**: 둘은 같은 양이다. 관계는 [Q-36](/blog/2026/11/02/knowledge-36-tpr-recall/)에서 다룬다.
- **PR 곡선(precision-recall curve)**: 임계값을 바꾸며 precision과 recall의 변화를 그린 곡선으로, F1은 그중 한 점을 요약한 값이다.
- **ROC 곡선**: TPR과 FPR로 임계값 전체의 순서 성능을 본다. [Q-38](/blog/2026/11/02/knowledge-38-roc-curve/)을 참고한다.

## 꼬리 질문

**Q.** 왜 산술 평균이 아니라 조화 평균인가요?
조화 평균은 작은 값에 크게 끌려갑니다. 그래서 전부 양성으로 예측해 recall만 1에 가깝게 만든 모델이 precision이 낮으면 높은 점수를 받을 수 없습니다.

**Q.** F1이 높으면 항상 좋은 모델인가요?
아닙니다. TN을 보지 않고 FP와 FN의 비용도 같게 다룹니다. 또한 양성 비율이 바뀌면 precision이 변하므로 F1도 함께 달라집니다.

**Q.** Macro F1과 Micro F1 중 어느 것을 써야 하나요?
소수 클래스의 성능도 동등하게 중요하면 macro를, 전체 표본 단위의 성능이 중요하면 micro를 씁니다. 어느 쪽이든 클래스별 F1도 함께 보는 것이 좋습니다.

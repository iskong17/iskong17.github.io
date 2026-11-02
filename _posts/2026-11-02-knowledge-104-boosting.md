---
title: "Q-104. Boosting이란?"
date: 2026-11-02 15:00:00 +0900
category: study
topic: basic
series: knowledge
lecture: 104
knowledge_group: machine-learning
card_title: "Q 104: Boosting"
card_desc: "Weak learner · Sequential training · Additive model · Learning rate"
lang: ko
description: "부스팅은 약한 모델을 순차적으로 더해 이전 모델이 틀린 부분을 보완하는 앙상블로, 주로 편향을 줄여 강한 모델을 만든다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 부스팅(boosting)은 성능이 낮은 약한 모델을 순차적으로 학습해, 각 모델이 앞선 모델들이 틀린 부분을 보완하도록 만드는 앙상블입니다.
> 단순한 모델도 조금씩 오차를 고쳐 가며 더하면 복잡한 함수를 표현할 수 있기 때문에 주로 편향을 줄입니다.
> AdaBoost는 틀린 샘플의 가중치를 높이고, Gradient Boosting은 손실의 음의 기울기를 다음 모델의 목표로 삼습니다.
> 표 형태의 데이터에서 널리 쓰이는 XGBoost와 LightGBM이 대표적인 구현입니다.
> 다만 순차적으로 오차에 집중하므로 노이즈에 민감할 수 있어서, 학습률과 나무 깊이를 제한하고 조기 종료를 함께 사용합니다.

## 자세한 설명

### 가법 모델

부스팅의 최종 모델은 약한 모델 $$h_t$$의 가중합이다.

$$
F_T(\mathbf{x}) = \sum_{t=1}^{T}\alpha_t h_t(\mathbf{x})
$$

기존 모델을 고정하고 손실 $$L$$을 가장 줄이는 다음 항을 찾는 과정을 반복한다. $$\nu$$는 학습률이다.

$$
F_t(\mathbf{x}) = F_{t-1}(\mathbf{x}) + \nu\,\alpha_t h_t(\mathbf{x}), \qquad (\alpha_t, h_t) \approx \arg\min_{\alpha, h} \sum_{i=1}^{n} L\left(y_i, F_{t-1}(\mathbf{x}_i) + \alpha h(\mathbf{x}_i)\right)
$$

즉, 부스팅은 함수 공간에서 탐욕적으로 손실을 줄여 가는 최적화로 볼 수 있다. 이 관점은 Friedman, Hastie, Tibshirani(2000)와 Friedman(2001)이 정리했다.

### 약한 모델

약한 학습기(weak learner)는 무작위 추측보다 조금 나은 모델을 말한다. 약한 학습기를 강한 학습기로 바꿀 수 있는지는 이론적 질문이었고, Schapire(1990)가 가능함을 보였다. 실제로는 스텀프(깊이 1 나무)나 깊이가 얕은 나무를 많이 쓴다. 나무를 얕게 두면 각 단계의 변화가 작아 분산이 억제된다.

### 조절 항목

| 항목 | 값을 키우면 | 비고 |
|---|---|---|
| 모델 수 $$T$$ | 학습 오차 감소, 결국 과적합 가능 | 조기 종료로 정함 |
| 학습률 $$\nu$$ | 빠르게 학습, 과적합 쉬움 | 작게 잡으면 $$T$$가 더 필요 |
| 나무 깊이 | 표현력과 특징 상호작용 증가 | 보통 얕게 사용 |
| 서브샘플 비율 | 비율을 낮추면 분산 감소 효과 | stochastic gradient boosting |

배깅과 달리 모델 수를 계속 늘리면 검증 오차가 다시 나빠질 수 있으므로 $$T$$가 중요한 하이퍼파라미터이다.

### 장단점

- **장점**: 표 형태 데이터에서 성능이 높은 편이고, 나무 기반 구현은 전처리 부담이 작다.
- **단점**: 순차 학습이어서 트리 단위 병렬화가 어렵다. 레이블 노이즈나 이상치에 민감할 수 있으며, 특히 지수 손실을 쓰는 AdaBoost에서 두드러진다.

## 관련 개념

- **알고리즘별 차이**: AdaBoost, LogitBoost, Gradient Boosting은 [Q-106](/blog/2026/11/02/knowledge-106-adaboost-logitboost-gradientboost/)에서 다룬다.
- **배깅과의 비교**: 실험과 함께 [Q-105](/blog/2026/11/02/knowledge-105-bagging-vs-boosting/)에서 다룬다.
- **경사하강법**: Gradient Boosting은 파라미터 대신 함수에 경사하강을 적용한다. [Q-15](/blog/2026/11/02/knowledge-15-gradient-descent/)를 참고한다.
- **앙상블**: 부스팅이 속한 범주이다. [Q-100](/blog/2026/11/02/knowledge-100-ensemble/)을 참고한다.

## 꼬리 질문

**Q.** 부스팅은 왜 복잡한 나무 대신 얕은 나무를 쓰나요?
각 단계에서 조금씩 고쳐 나가는 방식이므로 기본 모델이 단순해도 전체 모델은 복잡해집니다. 깊은 나무는 몇 단계 만에 학습 데이터에 과적합될 수 있습니다.

**Q.** 학습률을 낮추면 어떤 효과가 있나요?
각 모델의 기여가 줄어드는 shrinkage 효과로 일반화가 좋아지는 경향이 있습니다. 대신 더 많은 모델이 필요해 학습과 추론 시간이 늘어납니다.

**Q.** 부스팅은 항상 과적합되나요?
항상은 아닙니다. AdaBoost에서 학습 오차가 0이 된 뒤에도 테스트 오차가 계속 줄어드는 현상이 관찰되어 마진 관점의 설명이 제안되었습니다. 그래도 노이즈가 많으면 과적합되므로 검증으로 $$T$$를 정해야 합니다.

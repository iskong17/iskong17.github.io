---
title: "Q-49. Conditional Distribution이란 무엇인가요?"
date: 2026-11-02 14:05:00 +0900
category: study
topic: basic
series: knowledge
lecture: 49
knowledge_group: statistics
card_title: "Q 49: Conditional Distribution"
card_desc: "Conditioning · Product rule · Independence · Gaussian conditional"
lang: ko
description: "조건부 분포는 한 변수의 값이 주어졌을 때 다른 변수의 분포로, 결합 분포에서 해당 조건에 맞는 부분만 남겨 합이 1이 되도록 정규화한 것이다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 조건부 분포는 어떤 변수의 값을 알게 되었을 때 다른 변수가 따르는 분포입니다.
> 결합 확률을 조건의 주변 확률로 나누어 정의하는데, 조건에 맞는 부분만 남긴 뒤 합이 1이 되도록 다시 맞추는 연산입니다.
> 예를 들어 평소 우산을 쓸 확률이 45%여도, 비가 온다는 것을 알면 80%로 달라질 수 있습니다.
> 지도 학습의 분류와 회귀는 대부분 입력 x가 주어졌을 때 y의 조건부 분포나 그 평균을 학습하는 문제라고 볼 수 있습니다.
> 또한 A가 주어졌을 때 B의 확률과 B가 주어졌을 때 A의 확률은 일반적으로 다르므로, 둘을 혼동하지 않아야 합니다.

## 자세한 설명

### 정의

$$P(Y = y) > 0$$일 때, $$Y = y$$가 주어졌을 때 $$X$$의 조건부 분포는 다음과 같다. 연속인 경우에는 밀도로 같은 식 $$p(x \mid y) = p(x, y) / p(y)$$를 쓴다.

$$
P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}
$$

식을 바꾸면 곱 규칙(product rule) $$P(x, y) = P(x \mid y) P(y)$$가 된다. $$x$$의 모든 값에 대해 $$P(x \mid y) = P(x)$$이면 $$X$$와 $$Y$$는 독립이다.

### 예시

[Q-48](/blog/2026/11/02/knowledge-48-marginal/)의 결합 분포를 그대로 쓰면, $$P(\text{비}, \text{씀}) = 0.40$$, $$P(\text{맑음}, \text{씀}) = 0.05$$, $$P(\text{비}) = P(\text{맑음}) = 0.50$$, $$P(\text{씀}) = 0.45$$이다.

| 조건부 확률 | 계산 | 값 |
|---|---|---|
| $$P(\text{씀} \mid \text{비})$$ | $$0.40 / 0.50$$ | 0.80 |
| $$P(\text{씀} \mid \text{맑음})$$ | $$0.05 / 0.50$$ | 0.10 |
| $$P(\text{비} \mid \text{씀})$$ | $$0.40 / 0.45$$ | 약 0.89 |

$$P(\text{씀} \mid \text{비}) = 0.80$$과 $$P(\text{비} \mid \text{씀}) \approx 0.89$$는 서로 다른 값이다. 둘을 잇는 것이 [베이즈 정리](/blog/2026/11/02/knowledge-34-bayes-theorem/)이다.

### 가우시안의 경우

$$(X, Y)$$가 이변량 정규 분포이고 상관계수가 $$\rho$$이면 조건부 분포도 정규 분포이다.

$$
Y \mid X = x \sim \mathcal{N}\left(\mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(x - \mu_X),\; \sigma_Y^2 (1 - \rho^2)\right)
$$

여기서 $$\mu$$와 $$\sigma$$는 각 변수의 평균과 표준편차이다. 조건부 평균은 $$x$$에 대해 선형이고, 분산은 $$x$$와 무관하게 $$(1 - \rho^2)$$배로 줄어든다. 표준화된 경우 $$\rho = 0.8$$, $$x = 1$$이면 $$\mathcal{N}(0.8, 0.36)$$이며, [Q-48의 그림](/blog/2026/11/02/knowledge-48-marginal/) 오른쪽 빨간 곡선이 이것이다. 조건을 알면 분포가 더 좁아진다.

### 머신러닝에서의 사용

- **제곱 오차 회귀**: 최적 예측은 조건부 평균 $$\mathbb{E}[Y \mid X = x]$$이다. 선형 회귀는 정규 노이즈 가정에서 조건부 분포를 모델링하는 것으로 해석할 수 있다.
- **분류**: 로지스틱 회귀와 소프트맥스 신경망은 $$P(y \mid x)$$를 직접 모델링한다. [판별 모델](/blog/2026/11/02/knowledge-85-discriminative-model/)과 [생성 모델](/blog/2026/11/02/knowledge-84-generative-model/)의 차이가 여기서 나온다.
- **조건부 독립**: 나이브 베이즈는 클래스가 주어지면 특징이 독립이라고 가정한다. 이는 그냥 독립과 다른 가정이다. [Q-37](/blog/2026/11/02/knowledge-37-naive-bayes/)에서 다룬다.

## 관련 개념

- **주변 분포**: 분모가 되는 분포이다. [Q-48](/blog/2026/11/02/knowledge-48-marginal/)에서 다룬다.
- **연쇄 법칙(chain rule)**: $$p(x_1, \dots, x_n) = \prod_i p(x_i \mid x_1, \dots, x_{i-1})$$로 곱 규칙을 반복한 것이다. 언어 모델 같은 자기회귀 모델의 기반이다.
- **사후 분포(posterior)**: 데이터가 주어졌을 때 파라미터의 조건부 분포이다. [Q-39](/blog/2026/11/02/knowledge-39-likelihood-prior-posterior/)에서 다룬다.

## 꼬리 질문

**Q.** 연속 변수에서는 $$P(Y = y) = 0$$인데 어떻게 조건을 거나요?
밀도의 비 $$p(x, y) / p(y)$$로 정의합니다. $$Y$$가 $$y$$ 근처의 좁은 구간에 있을 때의 조건부 확률에서 구간을 줄인 극한으로 이해할 수 있습니다.

**Q.** 상관이 0이면 조건부 분포가 주변 분포와 같은가요?
이변량 정규 분포에서는 그렇지만 일반적으로는 아닙니다. 상관은 선형 관계만 재기 때문에, $$X$$가 $$[-1, 1]$$에서 균등하고 $$Y = X^2$$이면 상관은 0이어도 $$X$$를 알면 $$Y$$가 정확히 정해집니다.

---
title: "Q-56. Unbiased Estimation은 무조건 좋은가요?"
date: 2026-11-02 14:12:00 +0900
category: study
topic: basic
series: knowledge
lecture: 56
knowledge_group: statistics
card_title: "Q 56: Is Unbiased Always Better?"
card_desc: "MSE · James-Stein · Shrinkage · Nonexistence"
lang: ko
description: "아니다. 불편성은 평균에 대한 성질일 뿐 한 번의 오차가 작다는 뜻이 아니며, 편향을 조금 허용해 MSE를 줄일 수 있고, 합리적인 불편 추정량이 없거나 아예 존재하지 않는 경우도 있다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 무조건 좋은 것은 아닙니다.
> 불편하다는 것은 평균적으로 맞는다는 뜻일 뿐이어서, 분산이 크면 실제 한 번의 추정은 크게 빗나갈 수 있습니다.
> 오차는 편향의 제곱과 분산의 합이므로, 약간의 편향을 받아들이고 분산을 크게 줄이는 편이 더 정확할 때가 많습니다.
> 대표적인 예가 정규화인데, 릿지 회귀는 편향이 있지만 분산이 줄어 불편한 최소제곱법보다 예측이 좋아질 수 있습니다.
> 또 어떤 대상은 불편 추정량이 아예 존재하지 않거나, 존재해도 참값이 될 수 없는 엉뚱한 값을 내놓기도 합니다.
> 다만 평균을 여러 번 내거나 인과 효과를 보고할 때처럼 편향이 쌓이면 곤란한 경우에는 불편성이 여전히 중요합니다.

## 자세한 설명

### 이유 1: 오차를 결정하는 것은 MSE이다

$$\text{MSE} = \text{Bias}^2 + \text{Var}$$이므로([Q-52](/blog/2026/11/02/knowledge-52-bias-variance-mse/)), 불편성은 첫 항만 0으로 만든다. 정규 분산 추정에서 불편한 $$N - 1$$은 MSE가 가장 나쁘다. $$N = 5$$이면 불편인 경우 0.5, MLE는 0.36이다([Q-55](/blog/2026/11/02/knowledge-55-mle-vs-sample-variance/)).

### 이유 2: 고차원에서는 축소가 항상 이긴다

$$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\theta}, I_p)$$에서 관측 하나로 $$\boldsymbol{\theta}$$를 추정한다고 하자. $$\hat{\boldsymbol{\theta}} = \mathbf{x}$$는 불편이고 MLE이다. 그러나 $$p \ge 3$$이면 제임스-스타인(James-Stein) 추정량이 모든 $$\boldsymbol{\theta}$$에서 총 제곱 오차가 더 작다.

$$
\hat{\boldsymbol{\theta}}_{\text{JS}} = \left(1 - \frac{p - 2}{\lVert \mathbf{x} \rVert^2}\right)\mathbf{x}
$$

$$p = 10$$에서 500,000회 시뮬레이션한 총 제곱 오차는 다음과 같다.

| 참값 $$\boldsymbol{\theta}$$ | $$\mathbf{x}$$ (불편) | James-Stein (편향) |
|---|---|---|
| 모두 0 | 10.00 | 2.01 |
| 모두 1 | 10.00 | 6.21 |
| 모두 3 | 10.00 | 9.33 |

이득은 축소 지점에 참값이 가까울수록 크고, 멀어지면 줄어들지만 평균적으로 $$\mathbf{x}$$보다 나빠지지 않는다. 불편 추정량 $$\mathbf{x}$$는 이 기준에서 허용되지 않는(inadmissible) 추정량이다.

[릿지 회귀](/blog/2026/11/02/knowledge-90-regularization/)도 같은 맥락이다. OLS보다 MSE가 작아지는 양의 $$\lambda$$가 항상 존재한다는 것이 알려져 있다(Hoerl & Kennard, 1970). 다만 그 $$\lambda$$의 값은 참 계수에 의존하므로 교차 검증 등으로 골라야 한다.

### 이유 3: 불편 추정량이 터무니없을 수 있다

$$X \sim \text{Poisson}(\lambda)$$ 하나를 관측해 $$e^{-2\lambda}$$를 추정한다고 하자. 포아송 분포의 $$\mathbb{E}[(-1)^X] = e^{-2\lambda}$$이므로 $$(-1)^X$$가 불편 추정량이고, 이 경우 유일한 불편 추정량이다. 그러나 이 값은 $$+1$$ 또는 $$-1$$인 반면 대상은 항상 $$(0, 1)$$ 안에 있다. $$\lambda = 1$$로 2,000,000회 시뮬레이션하면 평균은 0.134로 참값 0.135에 맞지만, 개별 추정값은 전부 참값과 멀다.

유사한 예로, 분산 성분을 불편 추정하면 음수 추정값이 나올 수 있다.

### 이유 4: 불편 추정량이 존재하지 않을 수 있다

$$X \sim \text{Binomial}(n, p)$$에서 $$1/p$$를 추정한다고 하자. 어떤 $$g$$에 대해서도 $$\mathbb{E}[g(X)]$$는 $$p$$의 $$n$$차 이하 다항식이다.

$$
\mathbb{E}[g(X)] = \sum_{k=0}^{n} g(k) \binom{n}{k} p^k (1 - p)^{n - k}
$$

$$1/p$$는 $$p \to 0$$에서 발산하므로 어떤 다항식과도 같을 수 없다. 즉 불편 추정량이 존재하지 않는다. 또한 불편성은 변환에 대해 유지되지 않으므로, $$\sigma^2$$에 불편이면 $$\sigma$$에는 불편이 아니다.

### 그럼 언제 불편성을 고집하는가

편향이 평균으로 사라지지 않는 상황, 또는 값 자체의 해석이 중요한 상황이다. 자세한 내용은 [Q-57](/blog/2026/11/02/knowledge-57-unbiased-advantages/)에서 다룬다.

## 관련 개념

- **편향-분산 트레이드오프**: 예측 모델에 대한 같은 논리이다. [Q-28](/blog/2026/11/02/knowledge-28-bias-variance/)에서 다룬다.
- **정규화**: 의도적으로 편향을 넣어 일반화를 개선한다. [Q-90](/blog/2026/11/02/knowledge-90-regularization/)에서 다룬다.
- **MAP 추정**: 사전 분포는 축소를 일으키는 또 다른 해석이다. [Machine Learning 11](/blog/2026/10/22/ml-11-model-selection/)에서 다룬다.

## 꼬리 질문

**Q.** 제임스-스타인은 서로 무관한 평균들을 함께 추정해도 효과가 있나요?
총 제곱 오차 기준으로는 그렇습니다. 다만 이득은 합에 대한 것이므로, 특정 좌표 하나의 오차는 오히려 커질 수 있습니다.

**Q.** 편향된 추정량에서는 무엇을 조심해야 하나요?
보통의 신뢰구간이 참값을 정해진 비율로 포함하지 못할 수 있습니다. 예를 들어 라쏘의 계수에 일반적인 표준오차와 구간을 그대로 적용하는 것은 적절하지 않습니다.

---
title: "Q-52. Bias, Variance, MSE란 무엇이고 그들의 관계는 무엇인가요?"
date: 2026-11-02 14:08:00 +0900
category: study
topic: basic
series: knowledge
lecture: 52
knowledge_group: statistics
card_title: "Q 52: Bias, Variance, MSE"
card_desc: "MSE decomposition · Estimator · Shrinkage · Tradeoff"
lang: ko
description: "추정량의 MSE는 참값과의 제곱 오차의 기댓값이며, 편향의 제곱과 분산의 합으로 정확히 분해된다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 추정량 기준으로 말하면, 편향은 추정량의 기댓값이 참값에서 벗어난 정도, 분산은 추정량이 자신의 기댓값 주변에서 흔들리는 정도, MSE는 추정량과 참값의 제곱 오차의 기댓값입니다.
> 세 값의 관계는 MSE가 편향의 제곱과 분산의 합이라는 식으로 정확히 정리됩니다.
> 따라서 평균적인 오차를 줄이는 방법은 불편하게 만드는 것만이 아니며, 편향을 약간 늘려도 분산이 더 줄면 MSE는 작아집니다.
> 예를 들어 표본 평균에 1보다 작은 값을 곱해 0 쪽으로 축소하면 편향은 생기지만 분산이 줄어, 참값이 0에 가까울 때 MSE가 더 작아질 수 있습니다.
> 예측 모델에서도 같은 분해를 쓸 수 있는데, 이때는 줄일 수 없는 노이즈 항이 하나 더 붙습니다.

## 자세한 설명

### 정의와 분해

$$\theta$$는 고정된 참값, $$\hat{\theta}$$는 그 추정량이다.

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2], \quad
\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta, \quad
\text{Var}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]
$$

$$\mathbb{E}[\hat{\theta}]$$를 빼고 더해 전개한다.

$$
\mathbb{E}\left[\left((\hat{\theta} - \mathbb{E}[\hat{\theta}]) + (\mathbb{E}[\hat{\theta}] - \theta)\right)^2\right]
= \text{Var}(\hat{\theta}) + 2\,\text{Bias}(\hat{\theta})\,\mathbb{E}[\hat{\theta} - \mathbb{E}[\hat{\theta}]] + \text{Bias}(\hat{\theta})^2
$$

$$\mathbb{E}[\hat{\theta}] - \theta$$는 상수이고 가운데 항의 기댓값은 0이므로 다음을 얻는다.

$$
\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})
$$

불편 추정량이면 MSE와 분산이 같다.

### 축소 추정 예시

평균이 $$\mu$$이고 $$\bar{X}$$의 분산이 $$\sigma^2/n$$이라 하자. 축소 추정량 $$\hat{\mu}_c = c\bar{X}$$ ($$0 \le c \le 1$$)의 MSE는 다음과 같다.

$$
\text{MSE}(\hat{\mu}_c) = (1 - c)^2 \mu^2 + c^2 \frac{\sigma^2}{n}
$$

첫 항이 편향의 제곱, 둘째 항이 분산이다. $$c$$로 미분하면 최적값 $$c^* = \mu^2 / (\mu^2 + \sigma^2/n)$$을 얻는다. $$\mu = 1$$, $$\sigma^2/n = 1$$이면 다음과 같다.

| $$c$$ | Bias² | Variance | MSE |
|---|---|---|---|
| 1 (표본 평균) | 0 | 1 | 1 |
| 0.5 (최적) | 0.25 | 0.25 | 0.5 |

주의할 점이 있다. $$c^*$$는 모르는 $$\mu$$에 의존하므로 이 값을 바로 쓸 수는 없고, 고정된 $$c$$가 모든 $$\mu$$에서 $$\bar{X}$$보다 나은 것도 아니다. $$\mu$$가 클수록 축소의 손해가 커진다. 그래도 정규화나 사전 분포가 왜 도움이 될 수 있는지를 잘 보여 준다.

### 추정량과 예측

| | 파라미터 추정 | 예측 모델(제곱 오차) |
|---|---|---|
| 대상 | 고정된 $$\theta$$ | 고정된 $$\mathbf{x}$$에서의 $$y$$ |
| 분해 | Bias² + Variance | Bias² + Variance + 노이즈 $$\sigma^2$$ |

예측에서는 [Q-28](/blog/2026/11/02/knowledge-28-bias-variance/)에서 다룬 것처럼 관측 $$y$$ 자체의 노이즈가 더해진다. 그리고 모델 평가 지표로서의 MSE $$\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$$는 이런 기댓값을 표본으로 추정한 값이다. 이름은 같지만 편향과 분산으로 바로 나눌 수 있는 값은 아니다.

## 관련 개념

- **불편 추정량**: 편향이 0이어서 MSE가 곧 분산이다. [Q-51](/blog/2026/11/02/knowledge-51-biased-unbiased/)에서 다룬다.
- **분산 추정의 비교**: 분모 $$N-1$$, $$N$$, $$N+1$$이 편향과 분산을 맞바꾸는 예이다. [Q-55](/blog/2026/11/02/knowledge-55-mle-vs-sample-variance/)에서 다룬다.
- **정규화(regularization)**: 파라미터를 축소해 편향을 조금 늘리고 분산을 줄인다. [Q-90](/blog/2026/11/02/knowledge-90-regularization/)에서 다룬다.

## 꼬리 질문

**Q.** 다차원 파라미터에서는 어떻게 되나요?
$$\mathbb{E}\lVert \hat{\boldsymbol{\theta}} - \boldsymbol{\theta} \rVert^2 = \lVert \text{Bias} \rVert^2 + \text{tr}(\text{Cov}(\hat{\boldsymbol{\theta}}))$$으로 같은 형태가 됩니다.

**Q.** 제곱 오차가 아닌 손실에서도 이렇게 분해되나요?
같은 형태로 깔끔하게 더해지지는 않습니다. 이 분해는 제곱 전개에서 교차항이 사라지기 때문에 성립합니다.

---
title: "Q-51. Biased/Unbiased estimation의 차이는 무엇인가요?"
date: 2026-11-02 14:07:00 +0900
category: study
topic: basic
series: knowledge
lecture: 51
knowledge_group: statistics
card_title: "Q 51: Biased vs. Unbiased Estimation"
card_desc: "Unbiasedness · Consistency · Sample variance · Jensen's inequality"
lang: ko
description: "불편 추정은 추정량의 기댓값이 참값과 같은 경우이고, 편향 추정은 그렇지 않은 경우이며, 이는 평균에 대한 성질이므로 표본이 커질 때의 수렴을 말하는 일치성과는 다른 개념이다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 불편 추정량은 기댓값이 참값과 정확히 같은 추정량이고, 편향 추정량은 그 기댓값이 참값과 다른 추정량입니다.
> 즉 표본을 무수히 다시 뽑아 평균을 내면 불편 추정량은 참값에 맞고, 편향 추정량은 한쪽으로 벗어납니다.
> 대표적인 예로 분산을 구할 때 N-1로 나누면 불편이고, N으로 나누면 참 분산을 약간 작게 추정하는 편향 추정량입니다.
> 다만 불편하다고 해서 한 번의 추정이 정확하다는 뜻은 아니고, 표본이 커질 때 참값으로 수렴한다는 일치성과도 다른 개념입니다.
> 실제로 N으로 나눈 분산은 편향이 있어도 일치 추정량이고, 표본의 첫 번째 값만 쓰는 평균 추정은 불편해도 일치 추정량이 아닙니다.

## 자세한 설명

### 정의

참값 $$\theta$$의 추정량 $$\hat{\theta}$$가 모든 $$\theta$$에 대해 $$\mathbb{E}[\hat{\theta}] = \theta$$이면 **불편(unbiased)**, 그렇지 않으면 **편향(biased)** 추정량이라 한다. [편향의 정의](/blog/2026/11/02/knowledge-50-bias/)에서 편향이 0인 경우가 불편이다. "모든 $$\theta$$에 대해"가 중요하다. 특정 참값에서만 맞으면 불편이 아니다. 예를 들어 데이터와 관계없이 항상 5라고 답하는 추정량은 $$\theta = 5$$에서만 맞을 뿐이다.

### 불편성과 일치성

일치성(consistency)은 $$n \to \infty$$일 때 $$\hat{\theta}_n$$이 $$\theta$$로 확률 수렴한다는 성질이다. 두 성질은 서로 독립적이다. 평균이 $$\mu$$, 분산이 $$\sigma^2$$인 i.i.d. 표본 $$X_1, \dots, X_n$$의 예이다.

| 추정량 | 대상 | 불편 | 일치 |
|---|---|---|---|
| $$\bar{X}$$ | $$\mu$$ | O | O |
| $$X_1$$ (첫 값만 사용) | $$\mu$$ | O | X |
| $$\frac{1}{n}\sum_i (X_i - \bar{X})^2$$ | $$\sigma^2$$ | X | O |
| $$\frac{1}{n-1}\sum_i (X_i - \bar{X})^2$$ | $$\sigma^2$$ | O | O |
| $$\bar{X} + 1$$ | $$\mu$$ | X | X |

세 번째 추정량의 편향은 $$n \to \infty$$에서 0으로 사라지므로 점근적 불편(asymptotically unbiased)이다.

### 변환하면 불편성이 깨진다

$$\hat{\theta}$$가 불편해도 비선형 함수 $$g$$에 대해 $$g(\hat{\theta})$$는 일반적으로 $$g(\theta)$$의 불편 추정량이 아니다. 젠센 부등식(Jensen's inequality)에 따라 볼록한 $$g$$이면 $$\mathbb{E}[g(\hat{\theta})] \ge g(\mathbb{E}[\hat{\theta}])$$이기 때문이다.

대표적인 예가 표본 표준편차 $$S = \sqrt{S^2}$$이다. $$S^2$$는 $$\sigma^2$$에 대해 불편이지만 $$S$$는 $$\sigma$$를 평균적으로 작게 추정한다. 정규 분포에서 $$\mathbb{E}[S] = c_4(N)\sigma$$이고 다음과 같다.

| $$N$$ | 2 | 5 | 10 | 30 |
|---|---|---|---|---|
| $$c_4(N)$$ | 0.7979 | 0.9400 | 0.9727 | 0.9914 |

반면 최대우도 추정량은 재매개변수화에 불변이어서, 변환 전후의 답이 서로 일관되지만 불편성은 보장하지 않는다.

## 관련 개념

- **MSE**: 불편인지뿐 아니라 분산까지 합쳐 추정량을 평가한다. [Q-52](/blog/2026/11/02/knowledge-52-bias-variance-mse/)에서 다룬다.
- **N-1로 나누는 이유**: 불편성을 맞추기 위한 보정이다. [Q-54](/blog/2026/11/02/knowledge-54-n-minus-one/)에서 다룬다.
- **불편 추정의 장단점**: [Q-56](/blog/2026/11/02/knowledge-56-unbiased-always-good/)과 [Q-57](/blog/2026/11/02/knowledge-57-unbiased-advantages/)에서 다룬다.

## 꼬리 질문

**Q.** 불편인데 일치하지 않으면 왜 나쁜가요?
데이터를 아무리 모아도 정확해지지 않기 때문입니다. $$X_1$$은 평균적으로 맞지만 분산이 $$\sigma^2$$로 고정되어, 표본 100만 개가 있어도 한 개를 쓸 때와 같습니다.

**Q.** 어떤 조건이면 일치성이 보장되나요?
편향과 분산이 모두 0으로 가면 MSE가 0으로 가므로 일치 추정량입니다. 이것은 충분 조건이며 필요 조건은 아닙니다.

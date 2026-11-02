---
title: "Q-61. Poisson Distribution은 어디에 쓰이나요?"
date: 2026-11-02 14:17:00 +0900
category: study
topic: basic
series: knowledge
lecture: 61
knowledge_group: statistics
card_title: "Q 61: Poisson Distribution"
card_desc: "Count data · Rare events · Poisson regression · Overdispersion · Poisson process"
lang: ko
description: "Poisson 분포는 일정한 시간이나 공간에서 독립적으로 일어나는 사건의 횟수를 모델링하며, 트래픽, 결함 수 같은 개수 데이터와 Poisson 회귀에 쓰인다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> Poisson 분포는 정해진 시간이나 공간에서 사건이 몇 번 일어나는지, 즉 개수 데이터를 모델링할 때 씁니다. 사건들이 서로 독립이고 평균 발생률이 일정하다는 가정에서 나오는 분포이기 때문입니다. 예를 들어 시간당 서버 요청 수, 제품당 결함 수, 하루 고객 문의 수가 대표적입니다. 파라미터는 평균 발생률 λ 하나이며, 평균과 분산이 모두 λ라는 특징이 있습니다. 머신러닝에서는 입력에 따라 λ를 예측하는 Poisson 회귀로 수요나 사건 수를 예측합니다. 다만 실제 데이터는 분산이 평균보다 큰 경우가 많아서, 그럴 때는 음이항 분포를 검토합니다.

## 자세한 설명

### 정의

단위 구간당 평균 발생 횟수가 $$\lambda > 0$$일 때 횟수 $$k = 0, 1, 2, \dots$$의 확률은 다음과 같다.

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad \mathbb{E}[X] = \text{Var}[X] = \lambda
$$

$$\lambda = 2$$이면 확률은 다음과 같다.

| $$k$$ | 0 | 1 | 2 | 3 | 4 이상 |
|---|---:|---:|---:|---:|---:|
| $$P(X = k)$$ | 0.135 | 0.271 | 0.271 | 0.180 | 0.143 |

### 왜 이 분포가 나오는가

Binomial에서 $$np = \lambda$$를 고정하고 $$n \to \infty$$로 보내면 Poisson이 된다. 즉 많은 기회 중 각각의 확률이 아주 작은 사건의 횟수이다. $$n = 1000$$, $$p = 0.002$$이면 사건이 0번일 확률은 Binomial로 0.13506, $$\lambda = 2$$인 Poisson으로 0.13534이다.

독립인 Poisson 변수의 합도 Poisson이며 비율은 더해진다. 그래서 하루 평균 2건이면 3일간의 횟수는 $$\lambda = 6$$인 Poisson이다.

### Poisson 회귀

횟수 $$y_i$$를 입력 $$\mathbf{x}_i$$로 예측하는 GLM이다. 로그 링크를 쓰므로 예측값이 항상 양수이다.

$$
y_i \sim \text{Poisson}(\lambda_i), \qquad \log \lambda_i = \mathbf{x}_i^\top\boldsymbol{\beta} + \log t_i
$$

$$t_i$$는 관측 기간이나 노출량이고, $$\log t_i$$를 **offset**이라 한다. 관측 기간이 사용자마다 다를 때 필요하다. 음의 로그 가능도는 상수를 빼면 $$\sum_i (\lambda_i - y_i \log\lambda_i)$$이며, 이것이 신경망에서 쓰는 Poisson loss이다.

### 가정이 깨질 때

| 현상 | 원인의 예 | 대안 |
|---|---|---|
| 과대산포(overdispersion): 분산 > 평균 | 관측 안 된 개인차, 사건의 군집 | 음이항 분포(negative binomial) |
| 0이 지나치게 많음 | 사건이 아예 일어날 수 없는 집단이 섞임 | zero-inflated 모델 |
| 시간에 따라 비율이 변함 | 요일, 시간대 효과 | 시간 공변량, 비동차 Poisson 과정 |
| 사건이 서로 영향을 줌 | 리트윗, 여진 | Hawkes 과정 등 자기 흥분 모델 |

표본의 분산과 평균을 비교하는 것이 가장 간단한 점검이다. 입력이 있는 모델이라면 입력별 평균이 다르므로, 원 데이터가 아니라 잔차를 기준으로 판단한다.

## 관련 개념

- **Binomial 분포**: Poisson은 $$n$$이 크고 $$p$$가 작은 Binomial의 극한이다. [Q-58](/blog/2026/11/02/knowledge-58-binomial-family/)을 참고한다.
- **Gamma와 지수 분포**: 사건 간격은 지수 분포, $$\alpha$$번째 사건까지의 시간은 Gamma이다. [Q-60](/blog/2026/11/02/knowledge-60-gamma/)을 참고한다.
- **켤레 사전 분포**: $$\lambda$$에 Gamma 사전 분포를 두면 사후 분포도 Gamma이다. [Q-62](/blog/2026/11/02/knowledge-62-conjugate-prior/)에서 계산한다.
- **GLM**: Poisson 회귀는 지수족과 로그 링크를 쓰는 GLM의 한 예이다. [Machine Learning 4강](/blog/2026/10/22/ml-04-glm/)에서 다룬다.

## 꼬리 질문

**Q.** 음이항 분포는 Poisson과 어떤 관계인가요?
Poisson의 λ가 Gamma 분포를 따른다고 두고 λ를 적분하면 음이항 분포가 됩니다. 개체마다 비율이 다른 상황을 반영하므로 분산이 평균보다 커집니다.

**Q.** 횟수 데이터에 일반 선형 회귀를 쓰면 무엇이 문제인가요?
예측값이 음수가 될 수 있고, 등분산 가정도 맞지 않습니다. Poisson에서는 평균이 클수록 분산도 커지기 때문입니다. 평균이 충분히 크면 근사가 쓸 만할 수는 있습니다.

**Q.** λ가 커지면 분포 모양은 어떻게 되나요?
정규 분포 $$N(\lambda, \lambda)$$에 가까워집니다. Poisson 변수는 비율 1인 독립 Poisson 변수의 합으로 볼 수 있어 중심극한정리가 적용되기 때문입니다.

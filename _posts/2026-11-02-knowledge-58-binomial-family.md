---
title: "Q-58. Binomial, Bernoulli, Multinomial, Multinoulli란 무엇인가요?"
date: 2026-11-02 14:14:00 +0900
category: study
topic: basic
series: knowledge
lecture: 58
knowledge_group: statistics
card_title: "Q 58: Binomial Family"
card_desc: "Bernoulli · Binomial · Categorical · Multinomial · One-hot"
lang: ko
description: "Bernoulli와 Multinoulli는 한 번의 시행에서 결과가 2개 또는 K개인 분포이고, Binomial과 Multinomial은 이를 n번 독립 반복했을 때 결과별 횟수의 분포이다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 네 분포는 결과의 가짓수와 시행 횟수라는 두 기준으로 정리할 수 있습니다. Bernoulli는 결과가 두 가지인 시행을 한 번 하는 분포이고, Binomial은 그 시행을 n번 독립으로 반복했을 때 성공 횟수의 분포입니다. Multinoulli는 결과가 K가지인 시행을 한 번 하는 분포로 Categorical 분포라고도 하고, Multinomial은 그 시행을 n번 반복했을 때 결과별 횟수의 분포입니다. 예를 들어 동전을 한 번 던지면 Bernoulli, 열 번 던져 앞면 수를 세면 Binomial, 주사위를 한 번 던지면 Multinoulli, 열 번 던져 눈별 횟수를 세면 Multinomial입니다. 머신러닝에서는 이진 분류의 출력을 Bernoulli로, 다중 클래스 분류의 출력을 Multinoulli로 모델링하며, 이 가정에서 음의 로그 가능도를 구하면 cross-entropy 손실이 나옵니다.

## 자세한 설명

### 네 분포의 관계

|  | 시행 1번 | 시행 $$n$$번 (횟수를 셈) |
|---|---|---|
| 결과 2가지 | Bernoulli | Binomial |
| 결과 $$K$$가지 | Multinoulli (Categorical) | Multinomial |

$$n = 1$$이면 Binomial은 Bernoulli가, Multinomial은 Multinoulli가 된다. $$K = 2$$이면 Multinoulli는 Bernoulli가, Multinomial은 Binomial이 된다.

### Bernoulli와 Binomial

성공 확률이 $$p$$일 때 **Bernoulli** 변수 $$x \in \{0, 1\}$$의 확률은 다음과 같다. 평균은 $$p$$, 분산은 $$p(1-p)$$이다.

$$
P(x) = p^x (1-p)^{1-x}
$$

독립인 Bernoulli 변수 $$n$$개의 합 $$X$$가 **Binomial**이다. 이항 계수 $$\binom{n}{k}$$는 $$k$$번의 성공이 놓일 자리를 고르는 경우의 수이다. 평균은 $$np$$, 분산은 $$np(1-p)$$이다.

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

$$n = 10$$, $$p = 0.3$$이면 $$P(X = 3) = 120 \times 0.3^3 \times 0.7^7 \approx 0.267$$이고, 평균은 3, 분산은 2.1이다.

### Multinoulli와 Multinomial

**Multinoulli**는 확률 벡터 $$\boldsymbol{\pi} = (\pi_1, \dots, \pi_K)$$, $$\sum_j \pi_j = 1$$로 정해진다. 결과를 one-hot 벡터 $$\mathbf{x}$$로 쓰면 다음과 같다.

$$
P(\mathbf{x}) = \prod_{j=1}^{K} \pi_j^{x_j}
$$

**Multinomial**은 $$n$$번 시행에서 결과 $$j$$가 나온 횟수 $$n_j$$의 결합 분포이다. $$\sum_j n_j = n$$이다.

$$
P(n_1, \dots, n_K) = \frac{n!}{n_1! \cdots n_K!} \prod_{j=1}^{K} \pi_j^{n_j}
$$

$$\boldsymbol{\pi} = (0.5, 0.3, 0.2)$$이고 $$n = 6$$일 때 횟수가 $$(3, 2, 1)$$일 확률은 $$60 \times 0.5^3 \times 0.3^2 \times 0.2 = 0.135$$이다.

각 $$n_j$$의 평균은 $$n\pi_j$$, 분산은 $$n\pi_j(1-\pi_j)$$이다. 합이 $$n$$으로 고정되어 있어 한 결과가 많아지면 다른 결과는 줄어드므로, 서로 다른 결과의 공분산은 $$-n\pi_i\pi_j$$로 음수이다.

### 분류 손실과의 연결

| 출력 가정 | 모델 출력 | 음의 로그 가능도 |
|---|---|---|
| Bernoulli | sigmoid | binary cross-entropy |
| Multinoulli | softmax | categorical cross-entropy |

정답 $$y$$가 Bernoulli를 따르고 모델이 $$\hat{p}$$를 낸다면 음의 로그 가능도는 $$-y\log\hat{p} - (1-y)\log(1-\hat{p})$$이다. 즉 cross-entropy 최소화는 이 분포 가정 아래의 최대 가능도 추정이다.

## 관련 개념

- **Sigmoid와 로지스틱 회귀**: Bernoulli의 $$p$$를 입력의 함수로 모델링한다. [Q-20](/blog/2026/11/02/knowledge-20-sigmoid/)과 [Machine Learning 3강](/blog/2026/10/22/ml-03-logistic-regression/)에서 다룬다.
- **Beta와 Dirichlet 분포**: 각각 Bernoulli의 $$p$$와 Multinoulli의 $$\boldsymbol{\pi}$$에 대한 사전 분포이다. [Q-59](/blog/2026/11/02/knowledge-59-beta-dirichlet/)를 참고한다.
- **Poisson 분포**: $$n$$이 크고 $$p$$가 작은 Binomial의 극한이다. [Q-61](/blog/2026/11/02/knowledge-61-poisson/)에서 다룬다.
- **Naive Bayes의 사건 모델**: 단어의 등장 여부를 보면 Bernoulli, 등장 횟수를 보면 Multinomial 모델이다. [Machine Learning 6강](/blog/2026/10/22/ml-06-naive-bayes/)을 참고한다.

## 꼬리 질문

**Q.** Multinoulli와 Multinomial이 자주 혼용되는 이유는 무엇인가요?
$$n = 1$$인 Multinomial이 Multinoulli이기 때문입니다. 그래서 일부 자료는 둘을 구분하지 않습니다. 라이브러리 이름도 제각각이므로 시행 횟수와 반환값이 횟수인지 범주인지 확인해야 합니다.

**Q.** 이진 분류에서 softmax를 쓸 수도 있나요?
가능합니다. 출력 두 개에 softmax를 적용하면 로짓 차이에 sigmoid를 적용한 것과 같아서 표현력은 같고, 파라미터만 중복됩니다.

**Q.** 다중 라벨 분류는 어떤 분포로 모델링하나요?
한 표본에 라벨이 여러 개 붙을 수 있으므로 라벨마다 독립인 Bernoulli로 두고 sigmoid와 binary cross-entropy를 씁니다. 합이 1이 되는 Multinoulli는 맞지 않습니다.

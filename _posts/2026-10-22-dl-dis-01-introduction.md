---
title: "Deep Learning Discussion 1: Introduction & Logistics"
date: 2026-10-22 12:01:00 +0900
category: study
topic: deep-learning
group: discussion
series: dl-dis
lecture: 1
card_title: "Discussion 1: Introduction & Logistics"
card_desc: "Train/validation/test splits · Bayes' theorem · Estimators · Sigmoid and softmax derivatives · Risk"
lang: ko
problems: "https://cs182sp21.github.io/static/discussions/dis1.pdf"
solutions: "https://cs182sp21.github.io/static/discussions/dis1-sol.pdf"
description: "데이터셋 분할 원칙, 베이즈 정리 계산, 추정량의 편향과 분산, 시그모이드와 소프트맥스의 미분, 위험과 경험적 위험을 정리한다."
---

<ul class="post-sources">
<li markdown="span">[UC Berkeley CS 182 (Spring 2021)](https://cs182sp21.github.io/) Discussion 1 정리</li>
<li markdown="span">관련 강의: [Deep Learning 1]({% post_url 2026-10-22-dl-01-introduction %})</li>
</ul>

첫 Discussion은 머신러닝의 기본 틀과 통계 복습을 다룬다. 문제 0은 강의 일정, 중간고사 날짜, slip day 등 수업 운영에 관한 질문이므로 생략한다.

강의에서 다루는 학습 문제는 세 종류다.

- **지도학습:** 데이터셋 $$\mathcal{D} = \{(x_1, y_1), \dots, (x_n, y_n)\}$$에서 $$f_\theta(x)$$가 $$y$$를 근사하도록 학습한다.
- **비지도학습:** 레이블 없는 $$\mathcal{D} = \{x_1, \dots, x_n\}$$에서 분포의 성질을 학습한다.
- **강화학습:** 고정된 데이터셋 없이 환경과 상호작용하며, 보상의 합 $$\sum_t r(s_t, a_t)$$를 최대화하는 정책 $$\pi(a \mid s)$$를 학습한다.

문제를 풀려면 모델 집합, 손실 함수, 최적화기를 정한 뒤 큰 CPU나 GPU에서 돌린다.

## 1. 검증 집합 이모저모

### 문제

1. 왜 테스트 집합으로 하이퍼파라미터를 튜닝하면 안 되는가?
2. 검증 집합은 무엇에 쓰는가?
3. 데이터셋을 다루는 일반적인 머신러닝 작업 흐름을 설명하라.

### 풀이

1. 테스트 집합으로 하이퍼파라미터를 고르면 사람이 손으로 모델을 테스트 집합에 **과적합**시키는 셈이다. 그러면 테스트 오차는 더 이상 실제 오차의 불편 추정치가 아니고, 낙관적으로 편향된다. 테스트 집합의 타당성을 지키려면 검증 집합 성능으로만 튜닝해야 한다.
2. 검증 집합은 하이퍼파라미터 튜닝(그리고 모델 선택)에 쓴다.
3. 작업 흐름은 다음과 같다.
   - 테스트 집합을 따로 떼어 두고, 나머지를 학습 집합과 검증 집합으로 나눈다.
   - 학습 집합으로만 모델을 학습한다.
   - 검증 집합 성능으로 하이퍼파라미터를 고르고 모델을 확정한다.
   - 모델을 확정한 뒤 테스트 집합에서 **한 번만** 평가한다. 이 단계에서는 더 이상 튜닝하지 않는다.

교차 검증을 쓰든 한 번의 학습/검증 분할을 쓰든 테스트 집합은 끝까지 보류해야 하며, 새 모델을 학습하면 새 테스트 집합을 쓰는 것이 원칙이다.

## 2. 독감에 걸렸을까?

### 문제

두통이 있을 확률을 $$P(H)$$, 독감에 걸렸을 확률을 $$P(F)$$라 하자. 다음 여섯 관측(두통, 독감)이 주어진다: (N, N), (Y, N), (N, N), (Y, Y), (Y, Y), (N, Y). $$P(F)$$, $$P(H)$$, $$P(H \mid F)$$를 구하고, 베이즈 정리로 $$P(F \mid H)$$를 구하라.

### 풀이

먼저 필요한 확률 개념을 정리한다. 두 사건의 결합확률은 $$P(A, B)$$이고, 독립이면 $$P(A, B) = P(A)P(B)$$, 즉 $$P(A) = P(A \mid B)$$다. 데이터가 i.i.d.(independent, identically distributed)라고 가정하면 $$P(x_1, \dots, x_n) = \prod_{i=1}^n P(x_i)$$다. 또 $$P(A, B) = P(A \mid B)P(B) = P(B \mid A)P(A)$$의 양변을 $$P(B)$$로 나누면 **베이즈 정리(Bayes' theorem)** 를 얻는다.

$$
P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}
$$

이제 표에서 센다.

- 독감은 6번 중 3번: $$P(F) = \frac{3}{6} = \frac{1}{2}$$.
- 두통도 6번 중 3번: $$P(H) = \frac{3}{6} = \frac{1}{2}$$.
- 독감인 3번 중 두통은 2번: $$P(H \mid F) = \frac{2}{3}$$.

베이즈 정리를 적용하면 다음과 같다.

$$
P(F \mid H) = \frac{P(H \mid F)P(F)}{P(H)} = \frac{\frac{2}{3} \cdot \frac{1}{2}}{\frac{1}{2}} = \frac{2}{3}
$$

표에서 직접 세어도 두통인 3번 중 독감이 2번이므로 결과가 일치한다.

### 참고: 추정량, 편향, 분산

데이터 $$X \sim P_\theta$$를 관측하고 미지의 $$\theta$$에 대한 어떤 양 $$g(\theta)$$를 추정한다고 하자. 관측 데이터의 함수 $$T(X)$$를 **통계량(statistic)** 이라 하고, 추정 대상을 맞히려는 통계량을 **추정량(estimator)** 이라 한다. 예를 들어 표본평균 $$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$는 모평균의 추정량이다.

- **편향(bias):** 추정량의 기댓값이 참값과 얼마나 다른가. 기댓값이 참값과 같으면 불편추정량(unbiased estimator)이다.
- **분산(variance):** 추정량이 자기 기댓값 주위로 얼마나 흩어지는가.

좋은 추정량은 편향과 분산이 모두 작다. 그런데 분산을 크게 줄일 수 있다면 약간의 편향을 감수하는 것이 나을 때가 있다. 이 트레이드오프는 [Discussion 2]({% post_url 2026-10-22-dl-dis-02-matrix-calculus %})에서 유도한다.

## 3. 시그모이드의 미분

### 문제

시그모이드 $$\sigma(x) = \frac{1}{1 + e^{-x}}$$의 미분을 $$\sigma(x)$$로 나타내라.

### 풀이

$$\sigma(x) = (1 + e^{-x})^{-1}$$로 쓰고 연쇄 법칙을 적용한다.

$$
\begin{aligned}
\frac{d}{dx}\sigma(x) &= -(1 + e^{-x})^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1 + e^{-x})^2} \\
&= \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
&= \frac{1}{1 + e^{-x}} \cdot \left( 1 - \frac{1}{1 + e^{-x}} \right) = \sigma(x)\,(1 - \sigma(x))
\end{aligned}
$$

두 번째 줄에서 $$\frac{e^{-x}}{1 + e^{-x}} = \frac{(1 + e^{-x}) - 1}{1 + e^{-x}}$$를 썼다. 미분값이 함수값만으로 표현되므로 역전파에서 순전파 결과를 재사용할 수 있다.

## 4. 소프트맥스의 미분 (도전 문제)

### 문제

소프트맥스 $$p_i = \frac{e^{f_i(x)}}{\sum_{j=1}^n e^{f_j(x)}}$$에 대해, 각 $$k$$마다 $$\frac{\partial p_i}{\partial f_k}$$를 구하라. 소프트맥스는 시그모이드를 다중 클래스로 확장한 것으로 볼 수 있다.

### 풀이

표기를 줄이려고 $$f_j = f_j(x)$$, $$S = \sum_{j=1}^n e^{f_j}$$라 하자. 몫의 미분법 $$\left(\frac{g}{h}\right)' = \frac{g'h - h'g}{h^2}$$을 $$g = e^{f_i}$$, $$h = S$$에 적용한다. $$\frac{\partial S}{\partial f_k} = e^{f_k}$$다.

**$$i = k$$인 경우.** $$\frac{\partial g}{\partial f_k} = e^{f_i}$$이므로 다음과 같다.

$$
\frac{\partial p_i}{\partial f_i} = \frac{e^{f_i} S - e^{f_i} e^{f_i}}{S^2} = \frac{e^{f_i}}{S} \cdot \frac{S - e^{f_i}}{S} = p_i (1 - p_i)
$$

**$$i \neq k$$인 경우.** $$g$$는 $$f_k$$에 의존하지 않으므로 다음과 같다.

$$
\frac{\partial p_i}{\partial f_k} = \frac{0 \cdot S - e^{f_k} e^{f_i}}{S^2} = -\frac{e^{f_i}}{S} \cdot \frac{e^{f_k}}{S} = -p_i p_k
$$

두 경우를 크로네커 델타 $$\delta_{ik}$$로 합치면 다음과 같다.

$$
\frac{\partial p_i}{\partial f_k} = p_i(\delta_{ik} - p_k)
$$

두 클래스일 때 $$i = k$$의 결과 $$p_i(1 - p_i)$$는 시그모이드 미분과 같은 꼴이다.

### 참고: 위험과 경험적 위험

**손실 함수** $$L(x, y, \theta)$$는 추정의 "나쁨"을 잰다. 예를 들어 0-1 손실 $$\sum_{i=1}^n \delta(f_\theta(x_i) \neq y_i)$$는 틀린 개수를 센다. **위험(risk)** 은 실제 분포에 대한 손실의 기댓값이다.

$$
R(\theta) = \mathbb{E}_{x \sim p(x),\, y \sim p(y \mid x)}[L(x, y, \theta)]
$$

제곱오차 $$L = (y - f_\theta(x))^2$$이면 위험은 평균제곱오차(MSE)다. 실제 분포에서 표본을 새로 뽑을 수 없으므로, 데이터셋으로 근사한 **경험적 위험** $$\frac{1}{n}\sum_{i=1}^n L(x_i, y_i, \theta)$$를 최소화한다. 이것이 실제 위험 최소화와 같은지는 편향-분산 분석으로 다음 Discussion에서 다룬다.

## 정리

- 학습 집합은 학습에, 검증 집합은 하이퍼파라미터 튜닝에, 테스트 집합은 모델 확정 후 한 번의 최종 평가에만 쓴다. 테스트 집합으로 튜닝하면 테스트 오차가 편향된다.
- 베이즈 정리 $$P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}$$로 조건을 뒤집을 수 있다. 독감 예제에서 $$P(F \mid H) = \frac{2}{3}$$다.
- 추정량은 편향과 분산으로 평가하며, 분산을 크게 줄인다면 약간의 편향은 감수할 만하다.
- $$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$이고, 소프트맥스는 $$\frac{\partial p_i}{\partial f_k} = p_i(\delta_{ik} - p_k)$$다.
- 위험은 손실의 기댓값이고, 지도학습은 보통 그 표본 근사인 경험적 위험을 최소화한다.

---
title: "Machine Learning 3: Weighted Least Squares, Logistic Regression, Newton's Method"
date: 2026-10-22 13:03:00 +0900
category: study
topic: machine-learning
card_title: "Lecture 3: Weighted Least Squares, Logistic Regression, Newton's Method"
card_desc: "Probabilistic interpretation of least squares · Locally weighted regression · Logistic regression · Perceptron · Newton's method"
lang: ko
notes: "https://cs229.stanford.edu/main_notes.pdf"
notes_label: "CS229 lecture notes (PDF)"
description: "가우시안 잡음 가정 아래 최소제곱이 최대우도 추정임을 보이고, 국소 가중 선형 회귀, 로지스틱 회귀와 그 경사상승 규칙, 퍼셉트론, 그리고 로그우도를 최대화하는 뉴턴법을 정리한다."
---

> [Stanford CS229 Machine Learning (Spring 2022)](https://cs229.stanford.edu/syllabus-spring2022.html) 3강 정리. 강의: Tengyu Ma, Christopher Ré.
> [KAIST DAVIAN Lab 기초 과목 스터디](https://davian.kaist.ac.kr/joinus)의 머신러닝 강의다.
> 이전: [Machine Learning 2]({% post_url 2026-10-22-ml-02-lms %}) · 다음: [Machine Learning 4]({% post_url 2026-10-22-ml-04-glm %})
>
> [▶ 강의 영상](https://www.youtube.com/watch?v=k_pDh_68K6c) · 참고: CS229 lecture notes Chapter 1 (1.3 Probabilistic interpretation, 1.4 Locally weighted linear regression), Chapter 2 (2.1 Logistic regression, 2.2 Perceptron, 2.4 Newton's method)

## 최소제곱의 확률적 해석

[2강]({% post_url 2026-10-22-ml-02-lms %})에서는 최소제곱 비용 $$J(\theta)$$를 그냥 정의했다. 왜 하필 제곱 오차가 합리적인 선택인가? 몇 가지 확률적 가정을 두면 최소제곱 회귀가 매우 자연스러운 알고리즘으로 유도된다.

### 가우시안 잡음 가정

타깃과 입력이 다음 관계로 이어져 있다고 가정한다.

$$
y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}
$$

$$\epsilon^{(i)}$$는 모델에 넣지 않은 효과(예: 집값에 중요한데 회귀에서 빠진 특성)나 무작위 잡음을 담는 오차항이다. 이 $$\epsilon^{(i)}$$들이 평균 0, 분산 $$\sigma^2$$인 가우시안(정규) 분포를 IID(independently and identically distributed)로 따른다고 가정한다. 즉 $$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$이고 밀도는 다음과 같다.

$$
p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(\epsilon^{(i)})^2}{2\sigma^2} \right)
$$

이로부터 $$y^{(i)}$$의 조건부 분포를 얻는다.

$$
p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)
$$

즉 $$y^{(i)} \mid x^{(i)}; \theta \sim \mathcal{N}(\theta^T x^{(i)}, \sigma^2)$$이다. 세미콜론 표기 $$p(y^{(i)} \mid x^{(i)}; \theta)$$는 $$\theta$$로 매개변수화된 분포라는 뜻이다. $$\theta$$는 확률 변수가 아니므로 $$\theta$$에 "조건부"라고 쓰지 않는다.

### 최대우도

데이터 전체의 확률 $$p(\vec{y} \mid X; \theta)$$를 $$\theta$$의 함수로 볼 때 이를 우도(likelihood) 함수라 한다. 오차가 독립이므로 우도는 곱으로 분해된다.

$$
L(\theta) = \prod_{i=1}^{n} p(y^{(i)} \mid x^{(i)}; \theta) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \right)
$$

**최대우도(maximum likelihood)** 원리는 데이터의 확률이 가장 높아지도록 $$\theta$$를 고르라는 것이다. 순증가 함수를 씌워도 최대점은 같으므로, 계산이 쉬운 로그우도 $$\ell(\theta) = \log L(\theta)$$를 최대화한다.

$$
\ell(\theta) = n \log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{\sigma^2} \cdot \frac{1}{2} \sum_{i=1}^{n} \left( y^{(i)} - \theta^T x^{(i)} \right)^2
$$

첫 항은 $$\theta$$와 무관하므로 $$\ell(\theta)$$를 최대화하는 것은 다음을 최소화하는 것과 같다.

$$
\frac{1}{2} \sum_{i=1}^{n} \left( y^{(i)} - \theta^T x^{(i)} \right)^2 = J(\theta)
$$

정리하면, 가우시안 잡음 가정 아래 **최소제곱 회귀는 $$\theta$$의 최대우도 추정과 같다**. 다만 이 가정이 최소제곱의 필요조건은 아니며, 최소제곱을 정당화하는 다른 자연스러운 가정들도 있다. 또 최종 $$\theta$$는 $$\sigma^2$$에 의존하지 않으므로, $$\sigma^2$$를 몰라도 같은 답을 얻는다. 이 사실은 [4강]({% post_url 2026-10-22-ml-04-glm %})의 지수족과 GLM에서 다시 쓰인다.

## 국소 가중 선형 회귀

### 과소적합과 과적합

$$x \in \mathbb{R}$$에서 $$y$$를 예측할 때 $$y = \theta_0 + \theta_1 x$$로 맞추면 데이터가 직선 위에 있지 않아 잘 맞지 않는다. 특성 $$x^2$$를 추가해 $$y = \theta_0 + \theta_1 x + \theta_2 x^2$$로 맞추면 조금 낫다. 그렇다고 특성을 계속 늘리는 것이 좋은 것은 아니다. 5차 다항식 $$y = \sum_{j=0}^{5} \theta_j x^j$$은 데이터를 완벽히 통과하지만 새 입력에 대해 좋은 예측기가 되리라 기대하기 어렵다. 앞의 경우처럼 데이터의 구조를 모델이 포착하지 못하는 것을 **과소적합(underfitting)**, 뒤의 경우를 **과적합(overfitting)** 이라 한다. 이 개념은 [10강]({% post_url 2026-10-22-ml-10-bias-variance %})에서 형식화한다.

이처럼 특성 선택은 성능에 중요하다. **국소 가중 선형 회귀(locally weighted linear regression, LWR)** 는 학습 데이터가 충분하다면 특성 선택을 덜 중요하게 만들어 준다.

### 알고리즘

일반 선형 회귀는 질의점 $$x$$에서 예측할 때 다음을 한다.

1. $$\sum_i (y^{(i)} - \theta^T x^{(i)})^2$$을 최소화하도록 $$\theta$$를 맞춘다.
2. $$\theta^T x$$를 출력한다.

반면 LWR은 다음을 한다.

1. $$\sum_i w^{(i)} (y^{(i)} - \theta^T x^{(i)})^2$$을 최소화하도록 $$\theta$$를 맞춘다.
2. $$\theta^T x$$를 출력한다.

$$w^{(i)}$$는 음이 아닌 가중치다. $$w^{(i)}$$가 크면 $$i$$번째 예제의 오차를 작게 만들려고 애쓰고, 작으면 그 오차는 거의 무시된다. 흔히 쓰는 가중치는 다음과 같다.

$$
w^{(i)} = \exp\left( -\frac{(x^{(i)} - x)^2}{2\tau^2} \right)
$$

가중치는 예측하려는 질의점 $$x$$에 의존한다. $$\lvert x^{(i)} - x \rvert$$가 작으면 $$w^{(i)} \approx 1$$, 크면 $$w^{(i)}$$가 작아진다. 따라서 질의점 근처의 예제에 훨씬 큰 가중치를 두고 $$\theta$$를 고른다. $$x$$가 벡터이면 $$w^{(i)} = \exp(-(x^{(i)} - x)^T (x^{(i)} - x) / (2\tau^2))$$ 또는 $$\exp(-(x^{(i)} - x)^T \Sigma^{-1} (x^{(i)} - x) / (2\tau^2))$$로 일반화한다.

가중치 식이 가우시안 밀도와 겉모습이 비슷하지만, $$w^{(i)}$$는 가우시안과 직접적인 관계가 없고 확률 변수도 아니다. $$\tau$$는 예제의 가중치가 거리에 따라 얼마나 빨리 줄어드는지를 정하며 **대역폭(bandwidth)** 매개변수라 한다.

### 비모수적 알고리즘

LWR은 처음 보는 **비모수적(non-parametric)** 알고리즘이다. 일반 선형 회귀는 고정된 유한 개의 매개변수 $$\theta$$를 데이터에 맞추는 **모수적(parametric)** 알고리즘이어서, 학습이 끝나면 학습 데이터를 버려도 예측할 수 있다. 반면 LWR은 질의점마다 새로 $$\theta$$를 맞추므로 학습 집합 전체를 들고 있어야 한다. "비모수적"이란 가설을 표현하는 데 필요한 양이 학습 집합 크기에 선형으로 늘어난다는 뜻이다.

## 로지스틱 회귀

### 분류 문제

이제 $$y$$가 소수의 이산값만 갖는 분류를 다룬다. 우선 $$y \in \{0, 1\}$$인 **이진 분류(binary classification)** 에 집중한다. 예를 들어 스팸 분류기에서 $$x^{(i)}$$는 이메일의 특성, $$y$$는 스팸이면 1, 아니면 0이다. 0을 음성 클래스(negative class), 1을 양성 클래스(positive class)라 하며, $$y^{(i)}$$를 레이블(label)이라 한다.

$$y$$가 이산값이라는 사실을 무시하고 선형 회귀를 쓸 수도 있지만, 매우 나쁘게 동작하는 예를 쉽게 만들 수 있다. 직관적으로도 $$y \in \{0, 1\}$$인데 $$h_\theta(x)$$가 1보다 크거나 0보다 작은 값을 내는 것은 이상하다.

### 시그모이드 함수

그래서 가설의 형태를 바꾼다.

$$
h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}, \qquad g(z) = \frac{1}{1 + e^{-z}}
$$

$$g$$를 **로지스틱 함수(logistic function)** 또는 **시그모이드 함수(sigmoid function)** 라 한다. $$z \to \infty$$이면 $$g(z) \to 1$$, $$z \to -\infty$$이면 $$g(z) \to 0$$이고, 항상 0과 1 사이에 있다. 0에서 1로 매끄럽게 증가하는 다른 함수도 쓸 수 있지만, GLM과 생성 학습 알고리즘을 보면 로지스틱 함수가 꽤 자연스러운 선택임을 알게 된다([4강]({% post_url 2026-10-22-ml-04-glm %}), [5강]({% post_url 2026-10-22-ml-05-gda %})).

시그모이드의 도함수는 유용한 성질을 가진다.

$$
g'(z) = \frac{1}{(1 + e^{-z})^2} e^{-z} = \frac{1}{1 + e^{-z}} \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) = g(z)(1 - g(z))
$$

### 우도와 로그우도

최소제곱을 최대우도로 유도했던 것처럼, 분류 모델에도 확률적 가정을 두고 최대우도로 $$\theta$$를 맞춘다.

$$
P(y = 1 \mid x; \theta) = h_\theta(x), \qquad P(y = 0 \mid x; \theta) = 1 - h_\theta(x)
$$

이를 한 식으로 쓰면 $$p(y \mid x; \theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y}$$이다. $$n$$개 예제가 독립으로 생성되었다고 가정하면 우도와 로그우도는 다음과 같다.

$$
L(\theta) = \prod_{i=1}^{n} \left( h_\theta(x^{(i)}) \right)^{y^{(i)}} \left( 1 - h_\theta(x^{(i)}) \right)^{1 - y^{(i)}}
$$

$$
\ell(\theta) = \sum_{i=1}^{n} y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log (1 - h(x^{(i)}))
$$

### 경사상승 규칙

이번에는 최대화이므로 **경사상승(gradient ascent)** $$\theta := \theta + \alpha \nabla_\theta \ell(\theta)$$를 쓴다. 부호가 음이 아니라 양이다. 예제 하나 $$(x, y)$$에 대해 $$g' = g(1-g)$$를 쓰면 다음과 같다.

$$
\begin{aligned}
\frac{\partial}{\partial \theta_j} \ell(\theta)
&= \left( y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \right) \frac{\partial}{\partial \theta_j} g(\theta^T x) \\
&= \left( y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \right) g(\theta^T x)(1 - g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x \\
&= \left( y (1 - g(\theta^T x)) - (1 - y) g(\theta^T x) \right) x_j \\
&= (y - h_\theta(x)) x_j
\end{aligned}
$$

따라서 확률적 경사상승 규칙은 다음과 같다.

$$
\theta_j := \theta_j + \alpha \left( y^{(i)} - h_\theta(x^{(i)}) \right) x_j^{(i)}
$$

LMS 갱신 규칙과 모양이 똑같다. 하지만 $$h_\theta(x^{(i)})$$가 이제 $$\theta^T x^{(i)}$$의 비선형 함수이므로 같은 알고리즘이 아니다. 전혀 다른 학습 문제에서 같은 갱신 규칙이 나오는 것은 우연이 아니며, 그 이유는 [4강]({% post_url 2026-10-22-ml-04-glm %})의 GLM에서 밝혀진다.

### 로지스틱 손실로 보기

같은 손실을 다른 관점으로 쓰면 비선형 모델로 확장할 때 편하다. **로지스틱 손실(logistic loss)** 을 다음과 같이 정의한다.

$$
\ell_{\text{logistic}}(t, y) = y \log(1 + \exp(-t)) + (1 - y) \log(1 + \exp(t))
$$

$$h_\theta(x) = 1/(1 + e^{-\theta^T x})$$를 대입하면 예제 하나의 음의 로그우도는 $$\ell_{\text{logistic}}(\theta^T x, y)$$와 같다. $$t = \theta^T x$$를 **로짓(logit)** 이라 부른다. 미분하면 다음을 얻는다.

$$
\frac{\partial \ell_{\text{logistic}}(t, y)}{\partial t} = y \frac{-\exp(-t)}{1 + \exp(-t)} + (1 - y) \frac{1}{1 + \exp(-t)} = \frac{1}{1 + \exp(-t)} - y
$$

연쇄 법칙으로 $$\partial \ell / \partial \theta_j = -(\partial \ell_{\text{logistic}} / \partial t) \cdot x_j = (y - h_\theta(x)) x_j$$가 되어 앞의 유도와 일치한다. 이 관점은 [8강]({% post_url 2026-10-22-ml-08-neural-networks-1 %})의 신경망에서 다시 쓰인다.

## 여담: 퍼셉트론

로지스틱 회귀가 0 또는 1만 정확히 출력하도록 "강제"하려면 $$g$$를 임계 함수(threshold function)로 바꾸면 된다.

$$
g(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}
$$

이 $$g$$로 $$h_\theta(x) = g(\theta^T x)$$를 정의하고 같은 갱신 규칙 $$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$를 쓰면 **퍼셉트론 학습 알고리즘(perceptron learning algorithm)** 이 된다.

1960년대에는 퍼셉트론이 뇌의 개별 뉴런이 작동하는 방식의 대략적인 모델로 여겨졌다. 겉모습은 앞의 알고리즘들과 비슷하지만 성격은 매우 다르다. 퍼셉트론의 예측에 의미 있는 확률적 해석을 부여하거나, 퍼셉트론을 최대우도 추정 알고리즘으로 유도하기는 어렵다.

## 뉴턴법

### 영점 찾기

로지스틱 회귀의 $$\ell(\theta)$$를 최대화하는 또 다른 알고리즘을 살펴본다. 먼저 함수 $$f : \mathbb{R} \to \mathbb{R}$$의 영점 $$f(\theta) = 0$$을 찾는 **뉴턴법(Newton's method)** 은 다음 갱신을 반복한다.

$$
\theta := \theta - \frac{f(\theta)}{f'(\theta)}
$$

현재 추정값 $$\theta$$에서 $$f$$에 접하는 직선으로 $$f$$를 근사하고, 그 직선이 0이 되는 점을 다음 추정값으로 삼는다는 해석이 있다. 노트의 예에서는 영점이 약 1.3인 함수에 대해 $$\theta = 4.5$$에서 시작하면, 한 번의 반복 후 약 2.8, 다음 반복 후 약 1.8이 되고, 몇 번 더 반복하면 빠르게 1.3에 다가간다.

### 최대화에 적용하기

$$\ell$$의 최대점은 1계 도함수 $$\ell'(\theta)$$가 0인 점이다. 따라서 $$f(\theta) = \ell'(\theta)$$로 두면 같은 알고리즘으로 $$\ell$$을 최대화할 수 있다.

$$
\theta := \theta - \frac{\ell'(\theta)}{\ell''(\theta)}
$$

이 갱신은 $$\ell'$$의 영점을 찾을 뿐이므로, 최소화에도 똑같은 식을 쓸 수 있다.

### 다차원: 뉴턴-랩슨

로지스틱 회귀의 $$\theta$$는 벡터이므로 다차원으로 일반화해야 한다. 이를 **뉴턴-랩슨(Newton-Raphson)** 법이라 한다.

$$
\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)
$$

$$H$$는 **헤시안(Hessian)** 이라 부르는 $$d \times d$$ 행렬(절편항을 포함하면 $$(d+1) \times (d+1)$$)이며, 성분은 $$H_{ij} = \partial^2 \ell(\theta) / \partial \theta_i \partial \theta_j$$다.

| | (배치) 경사하강법 | 뉴턴법 |
|---|---|---|
| 필요한 반복 횟수 | 많다 | 훨씬 적다, 최솟값에 매우 빨리 근접한다 |
| 반복 하나의 비용 | 싸다 | $$d \times d$$ 헤시안을 구하고 역행렬을 풀어야 해서 비싸다 |

$$d$$가 아주 크지 않다면 뉴턴법이 전체적으로 보통 훨씬 빠르다. 로지스틱 회귀의 로그우도 최대화에 뉴턴법을 적용한 것을 **피셔 스코어링(Fisher scoring)** 이라고도 한다.

## 정리

- 오차가 IID 가우시안 $$\mathcal{N}(0, \sigma^2)$$라고 가정하면 로그우도 최대화는 최소제곱 $$J(\theta)$$ 최소화와 같다. 결과는 $$\sigma^2$$에 의존하지 않는다.
- 국소 가중 선형 회귀는 질의점 근처의 예제에 가중치 $$w^{(i)} = \exp(-(x^{(i)} - x)^2 / (2\tau^2))$$를 두고 매번 $$\theta$$를 맞추는 비모수적 알고리즘이다.
- 로지스틱 회귀는 $$h_\theta(x) = 1/(1 + e^{-\theta^T x})$$로 $$P(y = 1 \mid x)$$를 모델링하고 베르누이 로그우도를 최대화한다. 시그모이드는 $$g' = g(1 - g)$$를 만족한다.
- 로지스틱 회귀의 경사상승 규칙 $$\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)}$$는 LMS와 모양이 같지만 $$h_\theta$$가 비선형이다.
- 퍼셉트론은 시그모이드를 임계 함수로 바꾼 것으로, 확률적 해석이 어렵다.
- 뉴턴법 $$\theta := \theta - H^{-1} \nabla_\theta \ell(\theta)$$는 반복 횟수가 훨씬 적지만 반복마다 헤시안의 역행렬이 필요하다.

---
title: "Deep Learning Discussion 4: CNN Architectures & Dropout"
date: 2026-10-22 12:04:00 +0900
category: study
topic: deep-learning
group: discussion
card_title: "Discussion 4: CNN Architectures & Dropout"
card_desc: "ResNet skip connections · Batch norm backward pass · Ensembles and dropout · Xavier initialization · ReLU"
lang: ko
problems: "https://cs182sp21.github.io/static/discussions/dis4.pdf"
solutions: "https://cs182sp21.github.io/static/discussions/dis4-sol.pdf"
description: "대표적인 CNN 구조와 ResNet의 기울기 흐름, 배치 정규화의 역전파, 앙상블과 드롭아웃, Xavier 초기화 유도, ReLU의 순전파·역전파와 그 한계를 문제로 정리한다."
---

> [UC Berkeley CS 182 (Spring 2021)](https://cs182sp21.github.io/) Discussion 4 정리. 관련 강의: [Deep Learning 6]({% post_url 2026-10-22-dl-06-cnn %}), [Deep Learning 7]({% post_url 2026-10-22-dl-07-init-batchnorm %})
> [KAIST DAVIAN Lab 기초 과목 스터디](https://davian.kaist.ac.kr/joinus)의 딥러닝 강의다.
> 이전: [Discussion 3]({% post_url 2026-10-22-dl-dis-03-backprop-cnn %}) · 다음: [Discussion 5]({% post_url 2026-10-22-dl-dis-05-vision %})

이 디스커션은 CNN 구조, 배치 정규화, 가중치 초기화, 앙상블과 드롭아웃을 다룬다. 먼저 유명한 CNN 구조를 간단히 훑는다.

- **LeNet**: 초기 CNN 중 가장 유명하며 MNIST 손글씨 숫자 인식에 쓰였다. 합성곱 층과 풀링 층을 반복한 뒤 여러 FC 층을 둔다.
- **AlexNet**: 2012년 ILSVRC에서 큰 차이로 우승하며 컴퓨터 비전에서 CNN을 대중화했다. LeNet과 비슷하지만 더 크고 깊으며, ReLU와 드롭아웃의 효과, GPU 학습의 이점을 보여 주었다.
- **VGGNet**: ILSVRC 2014 준우승(우승은 GoogLeNet). 층 수를 늘리고 합성곱만 쌓는 것의 이점을 보였다. 파라미터가 약 1억 3800만 개로 많다는 단점이 있다.
- **ResNet**: 건너뛰는 연결(skip connection)로 입력과 기울기가 앞뒤로 빠르게 전파되게 한다. 2016년 중반 이미지 인식의 최고 성능이었고 지금도 백본으로 널리 쓰인다. VGG보다 파라미터가 훨씬 적다(ResNet-152가 약 6000만 개).

## 1. ResNet의 기울기 소실

### 문제

더 나은 초기화와 배치 정규화 외에, ResNet의 어떤 특징이 기울기 소실 문제를 완화하는가?

### 풀이

건너뛰는 연결이다. 잔차 블록 $$y = x + F(x)$$의 야코비안은 다음과 같다.

$$
\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}
$$

따라서 역전파 시 상류 기울기 $$\frac{\partial \mathcal{L}}{\partial y}$$가 항등 경로를 통해 **그대로** $$x$$로 전달된다.

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} + \left(\frac{\partial F}{\partial x}\right)^T \frac{\partial \mathcal{L}}{\partial y}
$$

블록 내부 $$F$$의 기울기가 소실되더라도 첫째 항이 남아 기울기가 앞쪽 층까지 도달한다. 다만 원 논문의 저자들은 기울기 소실 문제 자체는 대부분 더 나은 초기화와 배치 정규화로 해결된다고 지적한다.

## 2. 배치 정규화 층 살펴보기

### 문제

배치 정규화는 크기 $$m$$인 미니배치 $$\mathcal{B}$$의 경험적 평균과 분산을 계산한다.

$$
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i, \qquad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2
$$

그리고 각 차원을 정규화한 뒤 학습 가능한 스케일 $$\gamma$$와 편향 $$\beta$$를 적용한다($$\epsilon$$은 수치 안정성을 위한 작은 상수).

$$
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta
$$

$$\gamma$$, $$\beta$$ 덕분에 신경망은 필요하면 원래 활성화를 복원할 수 있고, 표준 가우시안보다 나은 분포를 학습할 수도 있다. 배치 정규화가 왜 잘 작동하는지는 완전히 이해되지 않았다(공변량 이동 감소, 최적화 지형의 평활화 등 견해가 엇갈린다).

1. 배치 정규화 층의 계산 그래프를 그려라.
2. 출력에 대한 미분 $$\frac{\partial f}{\partial y_i}$$가 주어졌을 때 $$\gamma$$, $$\beta$$에 대한 미분을 구하라.

### 풀이

**1. 계산 그래프.** 입력 $$x_1, \dots, x_m$$에서 $$\mu$$를 계산하고, $$x_i$$와 $$\mu$$에서 $$\sigma^2$$를 계산한다. $$x_i$$, $$\mu$$, $$\sigma^2$$가 모여 $$\hat{x}_i$$가 되고, $$\hat{x}_i$$에 $$\gamma$$를 곱하고 $$\beta$$를 더해 $$y_i$$가 된다. 핵심은 $$x_i$$가 $$\hat{x}_i$$로 가는 직접 경로 외에 $$\mu$$와 $$\sigma^2$$를 통해 **모든** $$\hat{x}_j$$에 영향을 준다는 점이다.

**2. 파라미터에 대한 미분.** $$\gamma$$와 $$\beta$$는 배치의 모든 $$y_i$$에 쓰이므로 배치 전체에 대해 합한다. $$\partial y_i / \partial \gamma = \hat{x}_i$$, $$\partial y_i / \partial \beta = 1$$이므로 다음과 같다.

$$
\frac{\partial f}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial f}{\partial y_i} \hat{x}_i, \qquad \frac{\partial f}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial f}{\partial y_i}
$$

## 3. (도전) 배치 정규화의 입력에 대한 미분

### 문제

$$\frac{\partial f}{\partial y_i}$$가 주어졌을 때 입력 $$x_i$$에 대한 미분을 구하라.

### 풀이

$$x_i$$는 세 경로, 즉 $$\hat{x}_i$$, $$\mu$$, $$\sigma^2$$를 통해 $$f$$에 영향을 준다(아래에서 $$\mu = \mu_{\mathcal{B}}$$, $$\sigma^2 = \sigma_{\mathcal{B}}^2$$).

$$
\frac{\partial f}{\partial x_i} = \frac{\partial f}{\partial \hat{x}_i} \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial f}{\partial \mu} \frac{\partial \mu}{\partial x_i} + \frac{\partial f}{\partial \sigma^2} \frac{\partial \sigma^2}{\partial x_i}
$$

각 조각을 계산한다.

**$$\hat{x}$$에 대한 미분과 직접 경로.**

$$
\frac{\partial f}{\partial \hat{x}_i} = \frac{\partial f}{\partial y_i} \gamma, \qquad \frac{\partial \hat{x}_i}{\partial x_i} = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
$$

**분산 경로.** $$\sigma^2$$는 모든 $$\hat{x}_j$$에 들어가므로 $$j$$에 대해 합한다.

$$
\frac{\partial f}{\partial \sigma^2} = \sum_{j=1}^{m} \frac{\partial f}{\partial \hat{x}_j} (x_j - \mu) \left(-\tfrac{1}{2}\right) (\sigma^2 + \epsilon)^{-3/2}, \qquad \frac{\partial \sigma^2}{\partial x_i} = \frac{2(x_i - \mu)}{m}
$$

**평균 경로.** $$\mu$$는 $$\hat{x}_j$$에 직접 들어가고 $$\sigma^2$$를 통해서도 들어간다.

$$
\frac{\partial f}{\partial \mu} = \sum_{j=1}^{m} \frac{\partial f}{\partial \hat{x}_j} \cdot \frac{-1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial f}{\partial \sigma^2} \cdot \frac{1}{m} \sum_{j=1}^{m} -2(x_j - \mu)
$$

$$\sum_j (x_j - \mu) = 0$$이므로 둘째 항은 사라진다.

$$
\frac{\partial f}{\partial \mu} = -\frac{1}{\sqrt{\sigma^2 + \epsilon}} \sum_{j=1}^{m} \frac{\partial f}{\partial \hat{x}_j}, \qquad \frac{\partial \mu}{\partial x_i} = \frac{1}{m}
$$

**합치기.**

$$
\frac{\partial f}{\partial x_i} = \frac{\partial f}{\partial \hat{x}_i} \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{\partial f}{\partial \mu} \frac{1}{m} + \frac{\partial f}{\partial \sigma^2} \frac{2(x_i - \mu)}{m}
$$

$$(x_j - \mu)/\sqrt{\sigma^2 + \epsilon} = \hat{x}_j$$를 이용해 정리하면 구현에 자주 쓰이는 간결한 형태가 된다. $$g_i = \partial f / \partial y_i$$라 하면 다음과 같다.

$$
\frac{\partial f}{\partial x_i} = \frac{\gamma}{m \sqrt{\sigma^2 + \epsilon}} \left( m g_i - \sum_{j=1}^{m} g_j - \hat{x}_i \sum_{j=1}^{m} g_j \hat{x}_j \right)
$$

괄호 안의 둘째 항은 평균 경로에서, 셋째 항은 분산 경로에서 온다. 상류 기울기에서 평균 성분과 $$\hat{x}$$ 방향 성분을 빼는 형태다.

## 앙상블 (개념 정리)

이 절에는 문제가 없고 개념만 정리한다. 앙상블은 같은 과제로 학습한 여러 모델을 묶어 예측을 집계한다. 신경망은 파라미터가 많아 분산이 높으므로, 여러 학습기를 평균하면 분산을 줄일 수 있다.

- **예측 평균**: $$N$$개의 신경망을 독립적으로 학습하고 예측을 확률 평균이나 다수결로 합친다.
- **파라미터 평균**: 예측 평균과 같은 방식으로는 작동하지 않는다. 독립적인 학습 실행끼리가 아니라, 스냅샷 앙상블처럼 **하나의 학습 궤적** 위의 파라미터만 평균한다.

가중치 초기화, 미니배치 섞기, SGD에 이미 무작위성이 많으므로 실제로는 복원 추출로 데이터를 재표본할 필요가 없다. 앙상블은 느리다는 단점이 있어, 분류 층(FC 헤드)만 앙상블하거나 스냅샷 앙상블을 써서 속도를 높인다.

## 4. 드롭아웃 복습

### 문제

드롭아웃은 순전파에서 각 노드를 확률 $$1 - p_{\text{keep}}$$로 제거하고, 테스트 시에는 모델을 그대로 쓴다. 이를 위해 학습 중 남은 활성화를 $$p$$로 나눈다(역 드롭아웃). 테스트 때는 어떤 차원도 0이 되지 않으므로, 학습 때 기댓값을 맞춰 두는 것이다.

$$
\tilde{h} = \frac{m \odot h}{p}, \qquad m_j \sim \mathrm{Bernoulli}(p), \qquad E[\tilde{h}] = h
$$

드롭아웃이 왜 성능을 높일 수 있는지, 언제 써야 하는지 설명하라.

### 풀이

드롭아웃은 학습 중 무작위 활성화를 제거하므로, 모델이 특정 특징에 과적합하는 것을 막고 **중복된 표현**을 갖게 한다. 하나의 특징이 빠져도 다른 특징들로 같은 결론을 낼 수 있어야 하기 때문이다. 또 매 순전파가 사실상 서로 다른 신경망이므로 거대한 앙상블처럼 작동한다. 따라서 신경망을 더 강건하게 하고 분산을 낮추고 싶을 때, 즉 과적합이 문제일 때 쓴다.

## 5. Xavier 초기화 유도

### 문제

활성화 함수가 tanh라 하자. tanh는 작은 입력에서 거의 선형이므로 $$\mathrm{Var}(a) = \mathrm{Var}(z)$$로 본다. 가중치와 입력은 i.i.d.이고 평균 0이며, 편향은 0으로 초기화한다. 층마다 분산의 크기가 유지되도록 하는 초기화 $$W_{ij} \sim \mathcal{N}(0, 1/D_a)$$를 유도하라. 여기서 $$D_a$$는 $$a$$의 차원이다.

### 풀이

편향이 0이므로 $$z_i = \sum_{j=1}^{D_a} W_{ij} a_j$$다. tanh의 선형성 가정과 독립인 항들의 합의 분산 공식을 쓴다.

$$
\mathrm{Var}(a_i) = \mathrm{Var}(z_i) = \mathrm{Var}\!\left(\sum_{j=1}^{D_a} W_{ij} a_j\right) = \sum_{j=1}^{D_a} \mathrm{Var}(W_{ij} a_j)
$$

독립인 두 확률변수의 곱의 분산은 다음과 같다.

$$
\mathrm{Var}(W_{ij} a_j) = E[W_{ij}]^2 \mathrm{Var}(a_j) + E[a_j]^2 \mathrm{Var}(W_{ij}) + \mathrm{Var}(W_{ij}) \mathrm{Var}(a_j)
$$

평균이 0이므로 앞의 두 항은 사라진다.

$$
\mathrm{Var}(a_i) = D_a \, \mathrm{Var}(W_{ij}) \, \mathrm{Var}(a_j)
$$

층마다 분산이 유지되려면 $$\mathrm{Var}(a_i) = \mathrm{Var}(a_j)$$여야 하므로 다음을 얻는다.

$$
D_a \, \mathrm{Var}(W_{ij}) = 1 \iff \mathrm{Var}(W_{ij}) = \frac{1}{D_a}
$$

(ReLU라면 분산이 절반으로 줄어 $$\mathrm{Var}(W_{ij}) = 2/D_a$$가 된다. 이것이 He 초기화다.)

## 6. (복습) ReLU의 순전파와 역전파

### 문제

$$\mathrm{ReLU}(x) = \max(0, x)$$다. 다음 $$4 \times 4$$ 입력 $$x$$에 대해 $$y = \mathrm{ReLU}(x)$$를 구하고, 출력에 대한 기울기 $$\frac{dL}{dy}$$가 주어졌을 때 $$\frac{dL}{dx}$$를 구하라.

$$
x = \begin{bmatrix} 1.5 & 2.2 & 1.3 & 6.7 \\ 4.3 & -0.3 & -0.2 & 4.9 \\ -4.5 & 1.4 & 5.5 & 1.8 \\ 0.1 & -0.5 & -0.1 & 2.2 \end{bmatrix}, \qquad \frac{dL}{dy} = \begin{bmatrix} 4.5 & 1.2 & 2.3 & 1.3 \\ -1.3 & -6.3 & 4.1 & -2.9 \\ -0.5 & 1.2 & 3.5 & 1.2 \\ -6.1 & 0.5 & -4.1 & -3.2 \end{bmatrix}
$$

### 풀이

순전파는 각 성분을 독립적으로 처리해 음수를 0으로 만든다.

$$
y = \begin{bmatrix} 1.5 & 2.2 & 1.3 & 6.7 \\ 4.3 & 0 & 0 & 4.9 \\ 0 & 1.4 & 5.5 & 1.8 \\ 0.1 & 0 & 0 & 2.2 \end{bmatrix}
$$

ReLU의 국소 미분은 $$x > 0$$이면 1, 아니면 0이다. 따라서 역전파는 순전파에서 0이 된 바로 그 위치의 기울기를 0으로 만든다. 상류 기울기의 부호는 상관없다.

$$
\frac{dL}{dx} = \frac{dL}{dy} \odot \mathbf{1}[x > 0] = \begin{bmatrix} 4.5 & 1.2 & 2.3 & 1.3 \\ -1.3 & 0 & 0 & -2.9 \\ 0 & 1.2 & 3.5 & 1.2 \\ -6.1 & 0 & 0 & -3.2 \end{bmatrix}
$$

## 7. ReLU 이모저모

### 문제

1. ReLU 활성화가 시그모이드 활성화보다 나은 점은 무엇인가?
2. ReLU 층의 출력은 음이 아니다. 이로 인한 부정적인 결과는 무엇이며, 이를 해결하기 위해 어떤 층이 개발되었는가?

### 풀이

**1.** 두 가지 장점이 있다.

- ReLU와 그 기울기는 시그모이드보다 계산이 훨씬 싸다.
- 기울기 소실 가능성을 줄인다. 시그모이드의 미분은 $$\sigma'(x) = \sigma(x)(1 - \sigma(x)) \le 1/4$$로 항상 1보다 작아, 여러 층에 걸쳐 곱하면 기울기가 빠르게 0으로 간다. ReLU의 미분은 양수 영역에서 정확히 1이다. 다만 배치 정규화로 입력을 중심화할 수 있으므로 이 장점은 상대적으로 덜 중요하다.

**2.** **죽은 ReLU(dying ReLU)** 문제가 있다. 어떤 유닛이 모든 입력에 대해 0을 출력하는 상태가 되면, 기울기도 0이므로 경사 하강법이 그 가중치를 바꾸지 못해 회복하기 어렵다. 이를 해결하려고 음수 영역에서도 기울기가 있는 변형들이 개발되었다.

- **Leaky ReLU**: $$x < 0$$에서 0 대신 작은 선형 성분 $$\alpha x$$를 출력한다.
- **ELU**: $$x < 0$$에서 $$\alpha(e^x - 1)$$을 출력한다.

## 정리

- ResNet의 건너뛰는 연결은 야코비안에 항등행렬을 더해 기울기가 깊은 층을 그대로 통과하게 한다.
- 배치 정규화의 $$\gamma$$, $$\beta$$ 기울기는 배치 전체에 대한 합이고, 입력 기울기는 $$\hat{x}_i$$, $$\mu$$, $$\sigma^2$$ 세 경로를 모두 고려해야 하며, 정리하면 상류 기울기에서 평균과 $$\hat{x}$$ 방향 성분을 뺀 형태가 된다.
- 앙상블은 분산을 줄이며, 헤드만 앙상블하거나 스냅샷 앙상블로 비용을 줄인다. 드롭아웃은 중복 표현을 강제하는 암묵적 앙상블이고, 역 드롭아웃은 학습 때 $$1/p$$로 스케일해 테스트 시 모델을 바꾸지 않는다.
- 분산 보존 조건 $$D_a \mathrm{Var}(W) = 1$$에서 Xavier 초기화 $$\mathrm{Var}(W) = 1/D_a$$가 나온다.
- ReLU의 역전파는 순전파에서 0이 된 위치의 기울기를 막는다. 계산이 싸고 기울기 소실이 덜하지만, 죽은 ReLU 문제를 Leaky ReLU와 ELU가 완화한다.

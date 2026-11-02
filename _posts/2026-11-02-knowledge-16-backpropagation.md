---
title: "Q-16. Backpropagation(역전파)이란 무엇인가요?"
date: 2026-11-02 12:16:00 +0900
category: study
topic: basic
series: knowledge
lecture: 16
knowledge_group: machine-learning
card_title: "Q 16: Backpropagation"
card_desc: "Chain rule · Computational graph · Reverse-mode autodiff · Vanishing gradient"
lang: ko
description: "역전파는 연쇄 법칙을 출력에서 입력 방향으로 적용해 모든 파라미터에 대한 손실의 그래디언트를 효율적으로 계산하는 알고리즘이다."
---

## 면접 답변

> 역전파는 신경망의 모든 파라미터에 대한 손실의 그래디언트를 연쇄 법칙으로 효율적으로 계산하는 알고리즘입니다. 먼저 순전파로 예측과 손실을 구하면서 중간값을 저장하고, 그다음 출력 쪽에서 입력 쪽으로 편미분을 차례로 곱해 나갑니다. 파라미터마다 따로 미분하면 같은 계산이 반복되지만, 역전파는 공통되는 뒤쪽 계산을 재사용하기 때문에 파라미터가 수백만 개여도 전체 그래디언트를 순전파와 비슷한 규모의 비용으로 구할 수 있습니다. 예를 들어 $$\hat{y} = w_2 w_1 x$$에서 $$w_1$$의 그래디언트는 손실의 $$\hat{y}$$ 미분에 $$w_2$$와 $$x$$를 곱해 얻습니다. 중요한 구분은 역전파가 그래디언트를 계산할 뿐이고, 파라미터를 실제로 바꾸는 것은 SGD나 Adam 같은 옵티마이저라는 점입니다. 깊은 망에서는 곱셈이 누적되어 그래디언트 소실이나 폭발이 생길 수 있어서 ReLU, 잔차 연결, 정규화 같은 기법을 함께 씁니다.

## 자세한 설명

### 손으로 계산하는 작은 예

입력 $$x = 2$$, 정답 $$y = 1$$, 파라미터 $$w_1 = 0.5$$, $$w_2 = 1.5$$인 선형 네트워크를 생각한다.

$$
h = w_1 x, \qquad \hat{y} = w_2 h, \qquad L = \frac{1}{2} (\hat{y} - y)^2
$$

**순전파**는 입력에서 출력으로 값을 계산한다.

$$
h = 1.0, \qquad \hat{y} = 1.5, \qquad L = 0.125
$$

**역전파**는 출력에서 입력으로 그래디언트를 계산한다.

$$
\frac{\partial L}{\partial \hat{y}} = \hat{y} - y = 0.5
$$

$$
\frac{\partial L}{\partial w_2} = \frac{\partial L}{\partial \hat{y}} \cdot h = 0.5 \times 1.0 = 0.5
$$

$$
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial \hat{y}} \cdot w_2 = 0.5 \times 1.5 = 0.75
$$

$$
\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial h} \cdot x = 0.75 \times 2 = 1.5
$$

$$w_1$$을 계산할 때 $$\partial L / \partial h$$를 재사용한 것이 핵심이다. 앞쪽 층으로 갈수록 뒤쪽 층에서 이미 계산한 결과에 국소 미분만 하나씩 더 곱하면 된다.

### 일반적인 형태

$$\ell$$번째 층의 출력을 $$\mathbf{h}^{(\ell)} = f^{(\ell)}(\mathbf{h}^{(\ell - 1)}; \theta^{(\ell)})$$라 하자. $$\theta^{(\ell)}$$은 그 층의 파라미터이고, 층은 $$K$$개이며, $$\mathbf{h}^{(K)}$$에서 스칼라 손실 $$L$$을 계산한다. 손실의 층 출력에 대한 그래디언트를 $$\boldsymbol{\delta}^{(\ell)} = \partial L / \partial \mathbf{h}^{(\ell)}$$라 하면 다음 관계가 성립한다.

$$
\boldsymbol{\delta}^{(\ell - 1)} = \left( \frac{\partial \mathbf{h}^{(\ell)}}{\partial \mathbf{h}^{(\ell - 1)}} \right)^{\top} \boldsymbol{\delta}^{(\ell)}, \qquad
\frac{\partial L}{\partial \theta^{(\ell)}} = \left( \frac{\partial \mathbf{h}^{(\ell)}}{\partial \theta^{(\ell)}} \right)^{\top} \boldsymbol{\delta}^{(\ell)}
$$

각 층은 야코비안 전체를 만들 필요 없이 야코비안 전치와 벡터의 곱(vector-Jacobian product)만 계산하면 된다. 그래서 이 방식이 효율적이다.

### 왜 효율적인가

파라미터가 $$d$$개일 때 그래디언트를 얻는 방법을 비교하면 다음과 같다.

| 방법 | 전체 그래디언트의 비용 | 특징 |
|---|---|---|
| 수치 미분 | 순전파 약 $$d$$번 이상 | 구현이 쉽지만 느리고 근사이다. 검증용으로 쓴다 |
| 순방향 자동 미분 | 순전파 약 $$d$$번 규모 | 입력이 적고 출력이 많을 때 유리하다 |
| 역전파(역방향 자동 미분) | 순전파의 상수 배 | 입력이 많고 출력이 스칼라 하나일 때 유리하다 |

신경망 학습은 파라미터가 많고 출력은 손실 하나이므로 역방향이 알맞다. 대가는 메모리이다. 역방향 단계에서 쓰기 위해 순전파의 중간값을 저장해야 한다.

### 코드로 확인하기

은닉층이 하나인 $$\tanh$$ 네트워크의 그래디언트를 직접 구하고 유한 차분과 비교한다.

```python
import numpy as np

rng = np.random.default_rng(0)
X, y = rng.normal(size=(8, 3)), rng.normal(size=(8, 1))
W1, W2 = rng.normal(size=(3, 4)), rng.normal(size=(4, 1))

def loss(W1, W2):
    return 0.5 * np.mean((np.tanh(X @ W1) @ W2 - y) ** 2)

# 순전파: 중간값을 저장한다
H = np.tanh(X @ W1)
y_hat = H @ W2

# 역전파: 출력 쪽 오차 신호를 입력 쪽으로 전달한다
d_yhat = (y_hat - y) / len(X)          # dL/dy_hat
g2 = H.T @ d_yhat                      # dL/dW2
d_H = d_yhat @ W2.T                    # dL/dH
g1 = X.T @ (d_H * (1 - H ** 2))        # dL/dW1, tanh' = 1 - tanh^2

# 유한 차분으로 W1의 한 원소를 검증한다
eps = 1e-6
E = np.zeros_like(W1); E[0, 1] = eps
numeric = (loss(W1 + E, W2) - loss(W1 - E, W2)) / (2 * eps)
print(f"{g1[0, 1]:.8f} {numeric:.8f}")
# -0.25956966 -0.25956966
```

### 깊은 망의 문제

앞쪽 층의 그래디언트에는 뒤쪽 층들의 야코비안이 차례로 곱해진다. 이 곱의 크기가 층마다 1보다 작으면 **그래디언트 소실(vanishing gradient)**이, 1보다 크면 **그래디언트 폭발(exploding gradient)**이 생긴다. 역전파가 틀려서 생기는 문제가 아니라 깊은 합성 함수의 미분 자체의 성질이다. ReLU, 좋은 가중치 초기화, 정규화 층, 잔차 연결이 소실을 완화하고, 그래디언트 클리핑이 폭발을 막는 데 쓰인다.

## 관련 개념

- **연쇄 법칙(chain rule)**: 합성 함수의 미분을 각 단계의 미분의 곱으로 나타내는 규칙이다. 역전파는 이 곱을 출력 쪽부터 계산하는 순서를 정한 것이다.
- **계산 그래프(computational graph)**: 연산을 노드로 나타낸 그래프로, 분기가 있어도 역방향 계산을 기계적으로 적용할 수 있게 한다. [Deep Learning 5](/blog/2026/10/22/dl-05-backpropagation/)에 층별 공식과 함께 정리되어 있다.
- **자동 미분(automatic differentiation)**: PyTorch, JAX 등이 쓰는 미분 기법이다. 기호 미분이나 수치 미분이 아니라 연산별 미분 규칙을 합성한다. 역방향 모드를 신경망에 적용한 것이 역전파이다.
- **경사하강법(gradient descent)**: 역전파가 구한 그래디언트로 파라미터를 업데이트한다. [Q-15](/blog/2026/11/02/knowledge-15-gradient-descent/)에서 다룬다.
- **역방향 함수(backward function)**: 모듈마다 역방향 계산을 정의하는 관점이다. [Machine Learning 9](/blog/2026/10/22/ml-09-backprop/)에서 다룬다.

## 꼬리 질문

**Q.** 역전파와 경사하강법은 어떻게 다른가요?
역전파는 그래디언트를 구하는 계산 방법이고, 경사하강법은 그 그래디언트로 파라미터를 바꾸는 최적화 방법이다. 역전파로 구한 그래디언트는 Adam이나 다른 옵티마이저에도 그대로 쓸 수 있다.

**Q.** 활성화 함수가 미분 불가능한 점을 가지면 어떻게 하나요?
ReLU는 0에서 미분되지 않지만, 구현에서는 그 점의 미분을 0 같은 정해진 값(서브그래디언트의 하나)으로 정해 계산한다. 그 점에 정확히 걸리는 경우가 드물어 실무에서 문제 되는 일은 거의 없다.

**Q.** 모든 가중치를 0으로 초기화하면 역전파에 어떤 일이 생기나요?
$$\tanh$$ 같은 활성화를 쓰는 MLP에서는 같은 층의 뉴런이 같은 값을 출력하고 같은 그래디언트를 받아 대칭이 깨지지 않는다. 앞의 코드 구조라면 $$W_1$$과 $$W_2$$가 모두 0일 때 그래디언트도 0이 되어 학습이 시작되지 않으므로, 무작위 초기화가 필요하다.

**Q.** 구현한 역전파가 맞는지 어떻게 확인하나요?
위 코드처럼 일부 파라미터에 대해 유한 차분과 비교하는 그래디언트 검사(gradient checking)를 한다. 중심 차분을 쓰고, 값이 비슷한지는 상대 오차로 판단한다.

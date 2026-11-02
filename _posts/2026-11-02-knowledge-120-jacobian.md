---
title: "Q-120. Jacobian Matrix란 무엇인가?"
date: 2026-11-02 15:16:00 +0900
category: study
topic: basic
series: knowledge
lecture: 120
knowledge_group: linear-algebra
card_title: "Q 120: Jacobian Matrix"
card_desc: "Partial derivatives · Linear approximation · Chain rule · Change of variables"
lang: ko
description: "야코비 행렬은 벡터 함수의 모든 1차 편미분을 모은 행렬로, 함수의 국소적인 선형 근사이며 역전파와 변수 변환의 기반이 된다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 야코비 행렬(Jacobian matrix)은 벡터를 입력받아 벡터를 내는 함수의 모든 1차 편미분을 모은 행렬입니다. 입력이 n차원, 출력이 m차원이면 크기는 m × n이고, (i, j) 성분은 i번째 출력을 j번째 입력으로 미분한 값입니다.
> 의미상으로는 스칼라 함수의 미분을 일반화한 것으로, 어떤 점 근처에서 비선형 함수를 가장 잘 근사하는 선형 변환입니다.
> 합성 함수의 야코비 행렬은 각 함수의 야코비 행렬의 곱이므로, 역전파는 이 곱을 출력 쪽부터 효율적으로 계산하는 방법입니다.
> 예를 들어 극좌표 변환의 야코비 행렬식은 r이고, 이것이 넓이 요소가 r dr dθ가 되는 이유입니다.
> Normalizing flow에서도 야코비 행렬식의 로그로 변환에 따른 밀도 변화를 보정합니다.

## 자세한 설명

### 정의

$$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$의 야코비 행렬은 다음과 같다.

$$
J_{\mathbf{f}}(\mathbf{x}) = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

스칼라 함수($$m = 1$$)이면 한 행으로, 그래디언트의 전치와 같다. 그래디언트 함수의 야코비 행렬은 헤시안이다.

### 선형 근사

점 $$\mathbf{a}$$ 근처에서 $$\mathbf{f}$$는 다음과 같이 근사된다. $$J$$는 함수의 국소적인 동작을 나타내는 선형 변환이다.

$$
\mathbf{f}(\mathbf{a} + \boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{a}) + J_{\mathbf{f}}(\mathbf{a})\boldsymbol{\delta}
$$

**수치 예제.** $$\mathbf{f}(x, y) = (x^2 y, 5x + \sin y)$$라 하자.

$$
J = \begin{bmatrix} 2xy & x^2 \\ 5 & \cos y \end{bmatrix}, \qquad J(1, 0) = \begin{bmatrix} 0 & 1 \\ 5 & 1 \end{bmatrix}
$$

$$\mathbf{f}(1, 0) = (0, 5)$$이고 변화량을 $$\boldsymbol{\delta} = (0.01, 0.02)$$라 하면, 근사값은 $$(0, 5) + (0.02, 0.07) = (0.02, 5.07)$$이다. 실제 값은 $$(0.020402, 5.069999)$$로 매우 가깝다.

### 연쇄 법칙과 역전파

$$\mathbf{h} = \mathbf{g} \circ \mathbf{f}$$이면 야코비 행렬은 곱으로 표현된다.

$$
J_{\mathbf{h}}(\mathbf{x}) = J_{\mathbf{g}}(\mathbf{f}(\mathbf{x}))\,J_{\mathbf{f}}(\mathbf{x})
$$

깊은 신경망은 여러 층의 합성이므로 전체 야코비 행렬은 여러 행렬의 곱이다. 하지만 전체 행렬을 직접 만들 필요는 없다.

| 모드 | 계산 | 곱하는 순서 | 효율적인 경우 |
|---|---|---|---|
| 순방향(forward) | JVP, $$J\mathbf{v}$$ | 입력 쪽부터 | 입력이 출력보다 적음 |
| 역방향(reverse) | VJP, $$\mathbf{v}^\top J$$ | 출력 쪽부터 | 출력이 입력보다 적음 |

학습 손실은 스칼라 하나이고 파라미터는 매우 많으므로 역방향이 유리하며, 이것이 [역전파](/blog/2026/11/02/knowledge-16-backpropagation/)이다. RNN처럼 야코비 행렬을 여러 번 곱하면 그 크기가 커지거나 작아져 기울기 폭발과 소실이 생긴다.

### 야코비 행렬식과 변수 변환

입력과 출력의 차원이 같으면 행렬식 $$\det J$$는 그 점 근처의 국소 부피 배율이다. 선형 변환의 [행렬식](/blog/2026/11/02/knowledge-115-determinant/)과 같은 의미이다.

**극좌표 예제.** $$x = r\cos\theta$$, $$y = r\sin\theta$$이면 행렬식은 $$r$$이다. 그래서 $$dx\,dy = r\,dr\,d\theta$$이다.

$$
\det\begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & r\cos\theta \end{bmatrix} = r\cos^2\theta + r\sin^2\theta = r
$$

**확률 밀도 변환.** 가역 함수 $$\mathbf{z} = \mathbf{f}(\mathbf{x})$$에 대해 다음이 성립한다.

$$
\log p_X(\mathbf{x}) = \log p_Z(\mathbf{f}(\mathbf{x})) + \log\left\lvert\det J_{\mathbf{f}}(\mathbf{x})\right\rvert
$$

부피가 늘어나는 만큼 밀도가 낮아지므로 전체 확률이 1로 유지된다. 일반적인 $$\det J$$의 계산 비용은 $$O(n^3)$$이어서, normalizing flow는 삼각 야코비처럼 쉽게 계산되는 구조를 설계한다.

## 관련 개념

- **행렬 미분**: 야코비 행렬을 사용한 행렬 미분은 [Deep Learning Discussion 2](/blog/2026/10/22/dl-dis-02-matrix-calculus/)에서 다룬다.
- **역전파**: 야코비 행렬 곱의 효율적인 계산이다. [Deep Learning 5](/blog/2026/10/22/dl-05-backpropagation/)를 참고한다.
- **Normalizing Flow**: 야코비 행렬식을 직접 쓰는 생성 모델이다. [임커밋 강의 정리](/blog/2026/10/24/imc-08-flow/)를 참고한다.
- **RNN의 기울기**: 연속된 야코비 행렬의 곱이 문제가 된다. [Deep Learning 10](/blog/2026/10/22/dl-10-rnn/)을 참고한다.
- **뉴턴 방법**: 그래디언트의 야코비 행렬인 헤시안을 사용한다. [Q-94](/blog/2026/11/02/knowledge-94-newton-method/)를 참고한다.

## 꼬리 질문

**Q.** 야코비 행렬과 헤시안 행렬의 차이는 무엇인가요?
야코비 행렬은 벡터 함수의 1차 미분이고, 헤시안은 스칼라 함수의 2차 미분입니다. 헤시안은 그래디언트의 야코비 행렬로 볼 수 있습니다.

**Q.** 역방향 모드가 항상 더 좋은가요?
아닙니다. 출력이 입력보다 많으면 순방향이 유리하고, 역방향은 중간값을 저장해야 하므로 메모리도 더 필요합니다. 전체 야코비 행렬이 필요할 때는 출력 수만큼 VJP를 반복합니다.

**Q.** 야코비 행렬을 모델 분석에 어떻게 활용하나요?
입력에 대한 야코비 행렬의 크기는 입력 변화에 대한 민감도를 나타냅니다. 특이값이 크면 작은 입력 변화에도 출력이 크게 변하므로, 안정성 분석이나 야코비 정규화에 활용합니다.

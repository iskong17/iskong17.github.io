---
title: "SEE Matrix 3: SVD를 눈으로 보기"
date: 2026-10-23 10:03:00 +0900
category: study
topic: basic
series: see-matrix
lecture: 3
card_title: "Chapter 3: Singular Value Decomposition"
card_desc: "Rectangular matrix · AAᵀ, AᵀA · Singular vector · Singular value · A = UΣVᵀ"
lang: ko
description: "직사각 행렬의 시각화, AAᵀ와 AᵀA에서 나오는 특이벡터와 특이값, 그리고 임의의 행렬을 회전, 차원 제거와 늘이기, 회전으로 보는 SVD A = UΣVᵀ를 정리한다."
---

<ul class="post-sources">
<li markdown="span">[Visual Kernel](https://www.youtube.com/@visualkernel/videos)의 SEE Matrix 시리즈 정리</li>
<li markdown="span">[▶ Chapter 3](https://www.youtube.com/watch?v=vSczTbgc8Rc)</li>
</ul>

## 개요

**특이값 분해(singular value decomposition, SVD)**는 선형대수의 피날레라고 불린다. 선형대수의 중요한 개념을 정리 하나에 모으고, 데이터 과학과 머신러닝에서 매우 많이 쓰인다. 겉으로 보면 SVD는 **대칭성, 랭크, 모양에 관계없이 모든 행렬**을 세 개의 특별한 행렬로 분해할 수 있다는 정리이다. 그 밑에는 직관적인 시각적 해석이 있다.

[Chapter 2의 스펙트럼 분해](/blog/2026/10/23/see-matrix-02-spectral/)를 이해했다면 SVD의 80%는 이해한 것이다. 남은 것은 세 걸음이다.

1. 직사각 행렬을 시각화하는 법
2. 아무 행렬에서나 대칭 행렬을 만드는 법
3. 특이벡터와 특이값

## 1. 직사각 행렬: 차원 사이의 변환

### $$\mathbb{R}^2$$의 벡터와 $$\mathbb{R}^3$$의 벡터는 다르다

$$(1, 2)$$와 $$(1, 2, 0)$$은 둘 다 $$x$$가 1, $$y$$가 2이고 그림으로도 비슷해 보인다. 하지만 $$(1, 2, 0)$$에게 $$z$$값을 물으면 0이라고 답하고, $$(1, 2)$$에게 물으면 3차원이 무엇인지조차 모른다. $$\mathbb{R}^3$$의 벡터는 $$z$$가 0이 아니게 움직일 수 있지만, $$\mathbb{R}^2$$의 벡터는 어떻게 회전하고 늘여도 $$z$$를 가질 수 없다. 둘은 다른 차원에 사는 다른 종류의 벡터이다.

### 직사각 행렬이 차원을 바꾼다

이 둘을 오가게 하는 것이 **직사각 행렬(rectangular matrix)**이다. $$m \times n$$ 행렬은 $$n$$차원 벡터를 받아 $$m$$차원 벡터를 내놓는다. 그래서 행렬을 $$\mathbb{R}^n$$에서 $$\mathbb{R}^m$$으로 가는 선형 변환이라고 한다. $$2 \times 3$$ 행렬은 $$\mathbb{R}^3$$의 벡터를 $$\mathbb{R}^2$$로 보낸다.

가장 단순한 $$\mathbb{R}^3 \to \mathbb{R}^2$$ 변환은 항등 행렬의 직사각형 버전이다. 영상에서는 이것을 **dimension eraser**라고 부른다.

$$
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
\begin{bmatrix} x \\ y \\ z \end{bmatrix}
= \begin{bmatrix} x \\ y \end{bmatrix}
$$

$$x$$, $$y$$는 그대로 두고 $$z$$를 완전히 지운다. $$(1, 2, 1)$$은 $$(1, 2)$$로 가고, $$z$$가 무엇이든 $$(1, 2, z)$$ 꼴의 벡터는 모두 $$(1, 2)$$로 간다.

![All vectors (1, 2, z) in R^3 land on the single point (1, 2) after the dimension eraser](/images/study/basic/see-matrix-03-2.svg)
*$$\mathbb{R}^3$$에서 $$z$$만 다른 점들(왼쪽)이 dimension eraser를 거치면 $$\mathbb{R}^2$$의 한 점 $$(1, 2)$$로 모인다(오른쪽).*

반대로 $$3 \times 2$$ 행렬 $$\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$$은 **dimension adder**이다. $$\mathbb{R}^2$$의 벡터에 $$z = 0$$을 붙인다.

합성도 Chapter 1과 같다. dimension eraser 다음에 대각 행렬을 곱하면

$$
\begin{bmatrix} 5 & 0 \\ 0 & 3 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \end{bmatrix}
= \begin{bmatrix} 5 & 0 & 0 \\ 0 & 3 & 0 \end{bmatrix}
$$

이다. $$z$$를 지운 뒤 $$x$$축을 5배, $$y$$축을 3배 하는 변환을 한 번에 한다. 이 모양을 기억해 두자. SVD의 $$\Sigma$$가 바로 이런 모양이다.

## 2. 아무 행렬에서나 대칭 행렬 만들기

Chapter 2에서 본 대칭 행렬의 강한 성질은 **고유벡터들이 서로 수직**이라는 것이었다. 그래서 정규화된 고유벡터를 모은 행렬은 직교 행렬, 즉 회전이 된다. 그런데 자연에서 만나는 행렬은 대부분 대칭이 아니다.

다행히 대칭성은 인위적으로 만들 수 있다. 대칭이 아닌 $$2 \times 3$$ 행렬

$$
A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}
$$

에 대해

$$
AA^T = \begin{bmatrix} 17 & 8 \\ 8 & 17 \end{bmatrix}, \qquad
A^TA = \begin{bmatrix} 13 & 12 & 2 \\ 12 & 13 & -2 \\ 2 & -2 & 8 \end{bmatrix}
$$

는 둘 다 정사각이고 대칭이다. 일반적으로 모든 행렬 $$A$$에 대해 $$AA^T$$와 $$A^TA$$는 대칭 행렬이다. ($$(AA^T)^T = (A^T)^TA^T = AA^T$$이다.) 영상에서는 $$AA^T$$를 $$S_{\text{left}}$$, $$A^TA$$를 $$S_{\text{right}}$$라고 부른다.

## 3. 특이벡터와 특이값

$$S_{\text{left}}$$와 $$S_{\text{right}}$$는 대칭이므로 각각 서로 수직인 고유벡터를 갖는다. $$A$$가 $$2 \times 3$$이면 $$S_{\text{left}}$$는 $$\mathbb{R}^2$$에 고유벡터 2개, $$S_{\text{right}}$$는 $$\mathbb{R}^3$$에 고유벡터 3개를 갖는다.

- $$S_{\text{left}} = AA^T$$의 고유벡터를 $$A$$의 **왼쪽 특이벡터(left singular vector)**라 한다.
- $$S_{\text{right}} = A^TA$$의 고유벡터를 $$A$$의 **오른쪽 특이벡터(right singular vector)**라 한다.

여기에 두 가지 사실이 더 있다.

1. $$S_{\text{left}}$$와 $$S_{\text{right}}$$는 **양의 준정부호(positive semi-definite, PSD)** 행렬이다. 그래서 고유값이 모두 0 이상이다.
2. 두 행렬의 고유값을 내림차순으로 정렬하면 **겹치는 부분은 정확히 같다.** 가장 큰 고유값끼리 같고, 그다음끼리 같다. 남는 고유값은 0이다.

이 공통 고유값의 제곱근이 $$A$$의 **특이값(singular value)** $$\sigma_i$$이다.

예제에서 $$AA^T$$의 고유값은 25, 9이고, $$A^TA$$의 고유값은 25, 9, 0이다. 따라서 특이값은 $$\sigma_1 = 5$$, $$\sigma_2 = 3$$이다. 고유벡터는 다음과 같다.

$$
\mathbf{u}_1 = \tfrac{1}{\sqrt{2}}(1, 1), \quad \mathbf{u}_2 = \tfrac{1}{\sqrt{2}}(1, -1)
$$

$$
\mathbf{v}_1 = \tfrac{1}{\sqrt{2}}(1, 1, 0), \quad \mathbf{v}_2 = \tfrac{1}{3\sqrt{2}}(1, -1, 4), \quad \mathbf{v}_3 = \tfrac{1}{3}(2, -2, -1)
$$

## SVD

이제 정리를 볼 수 있다.

> 모든 행렬 $$A$$는 조건 없이 다음과 같이 분해된다.
>
> $$
> A = U \Sigma V^T
> $$

- $$\Sigma$$는 $$A$$와 같은 크기의 **직사각 대각 행렬(rectangular diagonal matrix)**이다. 대각선에 특이값이 내림차순으로 있고 나머지는 0이다.
- $$U$$는 직교 행렬이다. 열은 $$S_{\text{left}}$$의 정규화된 고유벡터, 즉 **왼쪽 특이벡터**이고 고유값의 내림차순으로 놓인다.
- $$V$$는 직교 행렬이다. 열은 정규화된 **오른쪽 특이벡터**이고 역시 내림차순이다. 이것을 전치해서 $$V^T$$를 쓴다.

예제에서는

$$
\begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}
= \underbrace{\tfrac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}}_{U}
\underbrace{\begin{bmatrix} 5 & 0 & 0 \\ 0 & 3 & 0 \end{bmatrix}}_{\Sigma}
\underbrace{\begin{bmatrix} \mathbf{v}_1^T \\ \mathbf{v}_2^T \\ \mathbf{v}_3^T \end{bmatrix}}_{V^T}
$$

이다. 확인해 보면 $$A\mathbf{v}_1 = \tfrac{1}{\sqrt{2}}(5, 5) = 5\mathbf{u}_1$$, $$A\mathbf{v}_2 = \tfrac{1}{3\sqrt{2}}(9, -9) = 3\mathbf{u}_2$$, $$A\mathbf{v}_3 = \mathbf{0}$$이다.

### 네 단계로 보는 SVD

$$A$$ 자체는 $$\mathbb{R}^3$$에서 $$\mathbb{R}^2$$로 가는 복잡한 변환이지만, SVD로 보면 단순한 변환의 연속이다. 오른쪽부터 차례로 적용한다.

1. **$$V^T$$ (회전, $$\mathbb{R}^3$$ 안에서)**: 오른쪽 특이벡터를 표준 기저로 돌린다. 특이값이 가장 큰 $$\mathbf{v}_1$$은 $$x$$축으로, 두 번째인 $$\mathbf{v}_2$$는 $$y$$축으로, $$\mathbf{v}_3$$는 $$z$$축으로 간다.
2. **$$\Sigma$$의 dimension eraser 부분**: $$z$$를 지운다. $$\Sigma$$는 정사각 대각 행렬과 dimension eraser의 합성이다.
3. **$$\Sigma$$의 대각 부분**: $$x$$축을 $$\sigma_1$$배, $$y$$축을 $$\sigma_2$$배 한다.
4. **$$U$$ (회전, $$\mathbb{R}^2$$ 안에서)**: 표준 기저를 왼쪽 특이벡터 $$\mathbf{u}_1$$, $$\mathbf{u}_2$$ 쪽으로 돌린다.

이 네 단계를 합성한 것이 정확히 $$A$$이다.

![The unit sphere of R^3 through V transpose, Sigma and U, ending as an ellipse in R^2](/images/study/basic/see-matrix-03-1.svg)
*$$A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}$$가 단위 구를 옮기는 과정. 초록 원은 $$\mathbf{v}_1$$, $$\mathbf{v}_2$$를 지나는 대원이다. $$V^T$$가 $$\mathbf{v}_1$$(파랑), $$\mathbf{v}_2$$(보라), $$\mathbf{v}_3$$(회색)를 좌표축으로 돌리고, $$\Sigma$$가 $$z$$를 지운 뒤 $$x$$를 5배, $$y$$를 3배 하고, $$U$$가 타원을 $$\mathbf{u}_1$$, $$\mathbf{u}_2$$ 방향으로 돌린다. 초록 대원은 타원의 경계가 된다.*

### 구는 타원체로 간다

위키백과에는 SVD를 "모든 행렬은 구를 타원체로 보낸다"라고 설명하는 대목이 있다. 네 단계로 보면 이유가 보인다.

- 구를 회전해도 구이다.
- $$\mathbb{R}^3$$의 구에서 세 번째 차원을 지우면 $$\mathbb{R}^2$$의 원판이 된다.
- 원을 $$x$$, $$y$$ 방향으로 늘이면 타원이 된다.
- 마지막 회전은 타원의 모양을 유지하고 위치만 바꾼다.

엄밀한 증명은 아니지만 직관은 준다. 타원의 반지름은 특이값이고, 축 방향은 왼쪽 특이벡터이다.

### 반대 방향: $$\mathbb{R}^2 \to \mathbb{R}^3$$

$$3 \times 2$$ 행렬에도 같은 정신이 적용된다. 먼저 $$\mathbb{R}^2$$에서 회전하고, $$3 \times 2$$인 $$\Sigma$$가 특이값만큼 늘인 뒤 새 차원을 붙이고(dimension adder), 마지막으로 $$\mathbb{R}^3$$에서 표준 기저 $$x$$, $$y$$, $$z$$를 왼쪽 특이벡터 쪽으로 회전시킨다. 예를 들어 $$A^T = V\Sigma^TU^T$$가 그런 변환이다.

## 다른 해석들

이 네 단계가 SVD의 유일한 해석은 아니다. 행렬 자체가 맥락에 따라 다르게 해석되듯 SVD도 그렇다.

- **랭크 1 행렬의 합.** $$A = \sigma_1\mathbf{u}_1\mathbf{v}_1^T + \sigma_2\mathbf{u}_2\mathbf{v}_2^T + \cdots$$로, 랭크가 높은 행렬을 랭크 1 행렬들의 합으로 본다. 이미지를 거대한 픽셀 행렬로 보고 랭크 1 행렬을 하나씩 더해 가며 근사하는 **저랭크 근사(low-rank approximation)**가 그 응용이다. ([Linear Algebra 9](/blog/2026/10/22/la-09-svd/)에 예시가 있다.)
- **주성분 분석.** 첫 단계, 즉 오른쪽 특이벡터를 표준 기저로 돌리는 회전은 **주성분 분석(principal component analysis, PCA)**의 핵심을 담고 있다.
- **고차원에서의 직관.** 3차원을 넘어 그림으로 볼 수 없는 경우에도, 회전은 아무것도 키우거나 줄이지 않는다는 것을 안다. 그래서 고차원 행렬도 "원점 둘레의 회전, 이후 마지막 몇 차원 제거, 축 방향 늘이기, 다시 회전"으로 말할 수 있다.

영상은 "계산의 목적은 숫자가 아니라 통찰"이라는 말을 인용한다. 특성 다항식과 행렬 곱셈 세 페이지로 SVD를 구하면 무엇을 했는지 모를 수 있지만, 네 단계의 단순한 변환으로 나누어 보면 각 단계가 무엇을 하는지 정확히 안다. 마지막으로 영상은 SVD를 푸리에 변환과 나란히 놓는다. 두 정리 모두 극단적인 일반화와 분해에 집착한다. 푸리에 변환은 모든 함수를 사인파의 합으로 분해한다.

## 정리

| | 스펙트럼 분해 | SVD |
|---|---|---|
| 대상 | 대칭 행렬 $$S$$ | 모든 행렬 $$A$$ ($$m \times n$$) |
| 형태 | $$S = Q\Lambda Q^T$$ | $$A = U\Sigma V^T$$ |
| 회전 | $$Q$$: $$S$$의 고유벡터 | $$V$$: $$A^TA$$의 고유벡터, $$U$$: $$AA^T$$의 고유벡터 |
| 늘이기 | $$\Lambda$$: 고유값 (음수 가능) | $$\Sigma$$: 특이값 $$\sigma_i = \sqrt{\lambda_i} \ge 0$$, 차원 제거·추가 포함 |
| 변환 | 회전, 늘이기, 역회전 | $$\mathbb{R}^n$$에서 회전, 차원 맞추기와 늘이기, $$\mathbb{R}^m$$에서 회전 |

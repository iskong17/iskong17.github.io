---
title: "Linear Algebra 5: Inner Products and Least Squares"
date: 2026-10-22 10:05:00 +0900
category: study
topic: linear-algebra
card_title: "Lecture 5: Least Squares"
card_desc: "Inner product · Norm · Orthogonality · Least squares · Normal equation"
lang: ko
description: 내적, 노름, 직교를 정의한 뒤, 과결정 연립방정식의 최소제곱 문제와 그 기하학적 의미(수선의 발), 그리고 기하와 미분 두 방식으로 정규방정식을 유도한다.
---

> [선형대수 강의](https://www.youtube.com/playlist?list=PLVuwEl3g6OcIF-EEcerd5m5-Vw5jDpKN1) 19–25강 정리.
> [KAIST DAVIAN Lab 기초 과목 스터디](https://davian.kaist.ac.kr/joinus)의 선형대수 강의다.
> 이전: [Linear Algebra 4]({% post_url 2026-10-22-la-04-linear-transformation %}) · 다음: [Linear Algebra 6]({% post_url 2026-10-22-la-06-gram-schmidt-qr %})

## 내적

[▶ 19강](https://www.youtube.com/watch?v=JODLlI-j0dM)

### 동기: 과결정 연립방정식

수명 예측 예제에서 사람 3명과 미지의 가중치 3개는 해가 정확히 하나인 정사각 연립방정식을 만들었다. 실제 데이터는 특성 수보다 샘플 수가 훨씬 많다. 사람 100명, 특성 3개라면 $$A$$는 $$100 \times 3$$이고 $$\mathbf{b} \in \mathbb{R}^{100}$$이다. 열 3개는 $$\mathbb{R}^{100}$$ 안에서 많아야 3차원 부분공간을 생성하는데, 이는 매우 작다. 따라서 $$\mathbf{b}$$는 거의 확실히 그 바깥에 있고 $$A\mathbf{x} = \mathbf{b}$$는 해가 없다. 이런 시스템을 과결정(over-determined) 시스템이라 한다.

해가 없다고 포기하는 대신, 모든 방정식을 **근사적으로**, 그것도 가장 좋은 의미로 만족하는 $$\mathbf{x}$$를 찾는 것이 목표가 된다. 이것이 최소제곱(least squares) 방법이다. 이를 위해 먼저 몇 가지 도구가 필요하며, 그중 핵심은 두 벡터가 수직이라는 개념이다. 수직은 내적으로 정의된다.

### 정의

$$\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$$의 내적(inner product, dot product)은

$$
\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n
$$

이다. 행렬 곱으로 보면 행벡터($$1 \times n$$)와 열벡터($$n \times 1$$)의 곱이므로 $$1 \times 1$$ 행렬, 즉 스칼라가 나온다. 예를 들어 $$\mathbf{u} = (1, 2)$$, $$\mathbf{v} = (3, 4)$$이면 $$\mathbf{u} \cdot \mathbf{v} = 3 + 8 = 11$$이다.

### 성질

벡터 $$\mathbf{u}, \mathbf{v}, \mathbf{w}$$와 스칼라 $$c$$에 대해

1. $$\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$$.
2. $$(\mathbf{u} + \mathbf{v}) \cdot \mathbf{w} = \mathbf{u} \cdot \mathbf{w} + \mathbf{v} \cdot \mathbf{w}$$.
3. $$(c\mathbf{u}) \cdot \mathbf{v} = c(\mathbf{u} \cdot \mathbf{v})$$. 위 예에서 $$(5\mathbf{u}) \cdot \mathbf{v} = (5, 10) \cdot (3, 4) = 55 = 5 \cdot 11$$이다.
4. $$\mathbf{u} \cdot \mathbf{u} \geq 0$$이고, $$\mathbf{u} \cdot \mathbf{u} = 0$$인 것은 $$\mathbf{u} = \mathbf{0}$$인 것과 동치다. $$\mathbf{u} \cdot \mathbf{u} = u_1^2 + \cdots + u_n^2$$은 제곱의 합이므로 모든 성분이 0일 때만 0이다.

2와 3을 합치면

$$
(c_1 \mathbf{u}_1 + \cdots + c_p \mathbf{u}_p) \cdot \mathbf{w} = c_1 (\mathbf{u}_1 \cdot \mathbf{w}) + \cdots + c_p (\mathbf{u}_p \cdot \mathbf{w})
$$

이다. 선형결합을 먼저 하고 $$\mathbf{w}$$와 내적한 결과가, 내적을 먼저 하고 나중에 결합한 결과와 같다. 이는 앞 포스트의 선형성 조건 그대로다. $$\mathbf{w}$$를 고정하면 $$\mathbf{u} \mapsto \mathbf{u} \cdot \mathbf{w}$$는 선형변환이다.

성질 4는 길이를 암시한다. $$\mathbf{u} = (3, 4)$$이면 $$\mathbf{u} \cdot \mathbf{u} = 25$$로, 길이 $$5$$의 제곱이다.

## 벡터의 노름과 직교 벡터

[▶ 20강](https://www.youtube.com/watch?v=5v7xhc43zFk)

### 노름과 단위벡터

$$\mathbf{v}$$의 노름(norm), 즉 길이는

$$
\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v} \cdot \mathbf{v}} = \sqrt{\mathbf{v}^T \mathbf{v}} = \sqrt{v_1^2 + \cdots + v_n^2}
$$

이다. 스칼라 $$c$$에 대해 $$\lVert c\mathbf{v} \rVert = \lvert c \rvert \lVert \mathbf{v} \rVert$$이다.

길이가 1인 벡터를 단위벡터(unit vector)라 한다. 영이 아닌 벡터를 자신의 노름으로 나누면 방향은 유지되고 길이는 1이 된다. 이를 정규화(normalizing)라 한다.

$$
\mathbf{u} = \frac{1}{\lVert \mathbf{v} \rVert} \mathbf{v}.
$$

$$\mathbf{v} = (2, 0, 4)$$이면 $$\lVert \mathbf{v} \rVert = \sqrt{4 + 0 + 16} = \sqrt{20} = 2\sqrt{5}$$이므로 $$\mathbf{u} = \frac{1}{2\sqrt{5}}(2, 0, 4) = \frac{1}{\sqrt{5}}(1, 0, 2)$$이고, 길이는 1이다.

### 거리

$$\mathbf{u}$$와 $$\mathbf{v}$$ 사이의 거리(distance)는 차의 길이다.

$$
\operatorname{dist}(\mathbf{u}, \mathbf{v}) = \lVert \mathbf{u} - \mathbf{v} \rVert.
$$

기하학적으로 $$\mathbf{u} - \mathbf{v} = \mathbf{u} + (-\mathbf{v})$$다. $$\mathbf{v}$$를 뒤집어 $$\mathbf{u}$$에 더하면 $$\mathbf{v}$$의 끝점에서 $$\mathbf{u}$$의 끝점으로 가는 화살표를 원점으로 옮긴 벡터가 된다. 그 길이가 두 점 사이의 거리다. $$\mathbf{u} = (7, 1)$$, $$\mathbf{v} = (3, 2)$$이면 $$\mathbf{u} - \mathbf{v} = (4, -1)$$이고 거리는 $$\sqrt{17}$$이다. 좌표로 쓴 피타고라스 정리다.

### 내적과 각도

$$\mathbb{R}^2$$와 $$\mathbb{R}^3$$에서는 코사인 제2법칙으로

$$
\mathbf{u} \cdot \mathbf{v} = \lVert \mathbf{u} \rVert \lVert \mathbf{v} \rVert \cos \theta
$$

를 얻는다. $$\theta$$는 두 벡터 사이의 각이다. 고차원에서는 이 식으로 각을 정의한다. $$\mathbf{u} = (2, 0)$$, $$\mathbf{v} = (1, 1)$$이면

$$
\cos \theta = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \lVert \mathbf{v} \rVert} = \frac{2}{2 \cdot \sqrt{2}} = \frac{1}{\sqrt{2}}, \qquad \theta = 45^\circ
$$

이다. 단위원에서 $$\cos \theta$$는 가로 좌표다. $$0^\circ$$(같은 방향)에서 $$1$$, $$90^\circ$$에서 $$0$$이다.

### 직교

$$\mathbf{u} \cdot \mathbf{v} = 0$$이면 두 벡터가 직교(orthogonal)한다고 한다. 영이 아닌 벡터라면 두 노름이 양수이므로, 내적이 0이면 $$\cos \theta = 0$$, 즉 $$\theta = 90^\circ$$다.

예: $$\mathbf{u} = (1, 2)$$, $$\mathbf{v} = (4, -2)$$이면 $$\mathbf{u} \cdot \mathbf{v} = 4 - 4 = 0$$이므로 수직이다. 그림을 그려 각을 잴 필요가 없다. 이 판정법이 과결정 시스템의 근사해를 구하는 핵심 도구다.

## 최소제곱 문제

[▶ 21강](https://www.youtube.com/watch?v=4Zwt_TzgqHc)

### 후보 해의 오차 측정

수명 모델로 돌아간다. 사람 3명은 정확한 해가 있는 $$3 \times 3$$ 시스템을 만들었다. 네 번째 사람을 추가하면 방정식 4개, 미지수 3개가 된다. 기존 해를 대입하면 처음 세 예측은 정확히 맞지만, 네 번째 사람은 모델이 60세로 예측하는데 실제 수명은 72세다. 그 사람의 오차는 12다.

모든 방정식의 오차를 한 벡터로 모은 것이 잔차(residual) 벡터, 곧 오차 벡터다.

$$
\mathbf{b} - A\mathbf{x}.
$$

$$A\mathbf{x}$$는 예측값 벡터, $$\mathbf{b}$$는 실제값 벡터다. 기존 해의 잔차는 $$(0, 0, 0, 12)$$다. 다른 후보 $$\mathbf{x}$$는 잔차가 $$(1, -2, 2, 3)$$일 수 있다. 모든 사람을 조금씩 틀린다.

어느 쪽이 나은가? 각 후보를 오차 제곱합(sum of squared errors)으로 평가한다.

$$
0^2 + 0^2 + 0^2 + 12^2 = 144, \qquad 1^2 + (-2)^2 + 2^2 + 3^2 = 18.
$$

제곱은 큰 오차를 크게 벌한다. 이 기준으로 두 번째 후보가 훨씬 낫다. 어느 한 사람도 정확히 맞히지 못하는데도 그렇다. 제곱합은 잔차의 노름의 제곱이므로, 점수는 $$\lVert \mathbf{b} - A\mathbf{x} \rVert^2$$이다.

### 문제

**최소제곱 문제.** $$A \in \mathbb{R}^{m \times n}$$ ($$m > n$$)과 $$\mathbf{b} \in \mathbb{R}^m$$이 주어질 때

$$
\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \lVert \mathbf{b} - A\mathbf{x} \rVert
$$

를 찾는다. $$\hat{\mathbf{x}}$$를 최소제곱해(least squares solution)라 한다. 오차 벡터를 가장 짧게 만드는 최적의 근사해다.

### $$A\mathbf{x}$$가 사는 곳

$$\mathbf{b}$$는 고정이다. $$\mathbf{x}$$가 변하면 $$A\mathbf{x} = x_1 \mathbf{a}_1 + \cdots + x_n \mathbf{a}_n$$은 $$A$$의 열들의 모든 선형결합, 곧 열공간(column space) $$\operatorname{Col} A$$ 위를 움직인다. 시스템에 정확한 해가 없는 이유는 바로 $$\mathbf{b}$$가 $$\operatorname{Col} A$$ 바깥에 있기 때문이다. 따라서 문제는 $$\operatorname{Col} A$$에서 $$\mathbf{b}$$에 가장 가까운 점 $$A\mathbf{x}$$를 고르는 것이다. $$\mathbf{b}$$에서 그 점까지의 거리가 잔차의 길이다.

## 최소제곱의 기하학적 해석

[▶ 22강](https://www.youtube.com/watch?v=aQuY0qZVqnY)

### 가장 가까운 점은 수선의 발이다

열이 2개, 행이 3개인 $$A$$, 즉 미지수 2개와 방정식 3개를 생각한다. $$\operatorname{Col} A$$는 $$\mathbb{R}^3$$에서 원점을 지나는 평면이다. 평면이 수평으로 놓이도록 그림을 돌리면 $$\mathbf{b}$$는 평면 밖으로 튀어나와 있다.

가장 가까운 점을 $$\hat{\mathbf{b}}$$라 하자. 이는 $$\mathbf{b}$$에서 평면에 내린 **수선의 발**이다. $$\hat{\mathbf{b}}$$는 $$\operatorname{Col} A$$에 있으므로, 아직 모르더라도 $$A\hat{\mathbf{x}} = \hat{\mathbf{b}}$$인 $$\hat{\mathbf{x}}$$가 존재한다.

왜 수선의 발이 가장 가까운가? 평면의 다른 점 $$\mathbf{p}$$를 하나 잡는다. $$\mathbf{b} - \hat{\mathbf{b}}$$는 $$\hat{\mathbf{b}}$$를 지나는 평면 위의 모든 직선과 수직이므로, $$\mathbf{b}$$, $$\hat{\mathbf{b}}$$, $$\mathbf{p}$$는 $$\hat{\mathbf{b}}$$에서 직각인 직각삼각형을 이룬다. 피타고라스 정리에 의해 $$\mathbf{p} \neq \hat{\mathbf{b}}$$이면

$$
\lVert \mathbf{b} - \mathbf{p} \rVert^2 = \lVert \mathbf{b} - \hat{\mathbf{b}} \rVert^2 + \lVert \hat{\mathbf{b}} - \mathbf{p} \rVert^2 > \lVert \mathbf{b} - \hat{\mathbf{b}} \rVert^2
$$

이다. 빗변은 항상 다른 변보다 길므로 $$\hat{\mathbf{b}}$$가 평면의 어떤 점보다도 가깝다.

### "평면에 수직"을 방정식으로

조건은 잔차 $$\mathbf{b} - A\hat{\mathbf{x}}$$가 $$\operatorname{Col} A$$의 모든 벡터와 직교하는 것이다. 그 벡터들은 모든 $$c_1, c_2$$에 대한 $$c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2$$이므로 조건이 무수히 많다.

이를 유한 개로 줄일 수 있다. 내적의 선형성에 의해

$$
(\mathbf{b} - A\hat{\mathbf{x}}) \cdot (c_1 \mathbf{a}_1 + c_2 \mathbf{a}_2) = c_1 \, (\mathbf{b} - A\hat{\mathbf{x}}) \cdot \mathbf{a}_1 + c_2 \, (\mathbf{b} - A\hat{\mathbf{x}}) \cdot \mathbf{a}_2
$$

이다. 잔차가 각 열 $$\mathbf{a}_1$$, $$\mathbf{a}_2$$와 직교하면 $$c_1, c_2$$가 무엇이든 우변은 0이다. 따라서 공간을 생성하는 "재료" 벡터 각각과 직교하기만 하면 충분하다.

$$
\mathbf{a}_1^T (\mathbf{b} - A\hat{\mathbf{x}}) = 0, \qquad \mathbf{a}_2^T (\mathbf{b} - A\hat{\mathbf{x}}) = 0.
$$

$$A$$의 열마다 방정식이 하나씩 생긴다. 행벡터 $$\mathbf{a}_1^T, \mathbf{a}_2^T$$를 쌓아 행렬을 만들면, $$A$$의 열이 행이 되므로 그 행렬은 정확히 $$A^T$$다.

$$
\begin{bmatrix} \mathbf{a}_1^T \\ \mathbf{a}_2^T \end{bmatrix} (\mathbf{b} - A\hat{\mathbf{x}}) = A^T (\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}.
$$

이 행렬 방정식 하나가 "오차 벡터는 $$A$$의 열공간과 수직이다"를 표현한다.

## 정규방정식

[▶ 23강](https://www.youtube.com/watch?v=4sRsYlhISPU)

$$A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$$을 전개하고 항을 옮기면 정규방정식(normal equation)을 얻는다.

$$
A^T A \hat{\mathbf{x}} = A^T \mathbf{b}.
$$

이름은 평면에 수직인 벡터인 법선벡터(normal vector)에서 왔다. 이 방정식은 잔차가 $$\operatorname{Col} A$$에 수직이라는 조건을 담고 있다.

### 크기

$$A$$가 $$m \times n$$이고 $$m > n$$이면

- $$A^T A$$는 $$n \times n$$ 정사각행렬이고, 크기는 미지수 개수와 같다.
- $$A^T \mathbf{b}$$는 $$\mathbb{R}^n$$의 벡터다.

즉 과결정 $$m \times n$$ 시스템이 새로운 정사각 시스템 $$C\mathbf{x} = \mathbf{d}$$ ($$C = A^T A$$, $$\mathbf{d} = A^T \mathbf{b}$$)로 바뀐다. 이 시스템의 해가 정확히 원래 문제의 최소제곱해다. 최소화 문제가 연립방정식을 푸는 문제로 바뀐 것이다. $$A^T A$$가 가역이면

$$
\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}
$$

이다.

### 계산 예

미지수 2개, 방정식 3개를 맞춘다.

$$
A = \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix}, \qquad \mathbf{b} = \begin{bmatrix} 6 \\ 0 \\ 0 \end{bmatrix}.
$$

정확한 해는 없다. 처음 두 방정식이 $$x_1 = 6$$, $$x_2 = -6$$을 강제하는데, 세 번째 방정식은 $$-6 \neq 0$$이 된다. 계산하면

$$
A^T A = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}, \qquad A^T \mathbf{b} = \begin{bmatrix} 6 \\ 0 \end{bmatrix}
$$

이다. $$3x_1 + 3x_2 = 6$$과 $$3x_1 + 5x_2 = 0$$을 풀면 $$\hat{\mathbf{x}} = (5, -3)$$이다. 그러면

$$
\hat{\mathbf{b}} = A\hat{\mathbf{x}} = \begin{bmatrix} 5 \\ 2 \\ -1 \end{bmatrix}, \qquad \mathbf{b} - A\hat{\mathbf{x}} = \begin{bmatrix} 1 \\ -2 \\ 1 \end{bmatrix}.
$$

기하를 확인한다. 잔차는 두 열과 모두 직교한다: $$1 - 2 + 1 = 0$$, $$0 - 2 + 2 = 0$$. 최소 오차 제곱합은 $$1 + 4 + 1 = 6$$이다.

## 정규방정식의 또 다른 유도

[▶ 24강](https://www.youtube.com/watch?v=VDgQ-f1ZLwY)

같은 방정식이 미분에서도 나온다. 오차의 도함수를 0으로 두면 된다.

### 일변수 준비 운동

$$f(x) = x^2 - 4x + 3$$을 최소화하려면 완전제곱식 $$f(x) = (x - 2)^2 - 1$$로 바꿔 $$x = 2$$에서 최소임을 알 수 있다. 또는 $$f'(x) = 2x - 4 = 0$$을 푼다. 최소화 문제가 방정식이 되고, 그 방정식은 도함수를 0으로 두어 얻는다. 최소제곱 문제도 같은 방식이다.

### 노름 대신 노름의 제곱

$$\lVert \mathbf{b} - A\mathbf{x} \rVert$$를 최소화하는 것과 $$\lVert \mathbf{b} - A\mathbf{x} \rVert^2$$를 최소화하는 것은 같은 $$\hat{\mathbf{x}}$$를 준다. 제곱근은 음이 아닌 수에서 증가함수이기 때문이다. (제곱근을 연쇄법칙으로 미분하면 안쪽 도함수를 양수로 나눈 꼴이 되므로, 0이 되는 점이 같다.) 제곱 버전은 제곱근이 없어 편하므로 다음을 쓴다.

$$
f(\mathbf{x}) = \lVert \mathbf{b} - A\mathbf{x} \rVert^2 = (\mathbf{b} - A\mathbf{x})^T (\mathbf{b} - A\mathbf{x}).
$$

$$(A\mathbf{x})^T = \mathbf{x}^T A^T$$와, 스칼라는 전치해도 같으므로 $$\mathbf{b}^T A \mathbf{x} = \mathbf{x}^T A^T \mathbf{b}$$임을 써서 전개하면

$$
f(\mathbf{x}) = \mathbf{b}^T \mathbf{b} - 2\mathbf{x}^T A^T \mathbf{b} + \mathbf{x}^T A^T A \mathbf{x}
$$

이다.

### 그래디언트

$$f$$는 벡터를 받아 스칼라를 내놓는다. 그 그래디언트(gradient)는 편미분을 모은 열벡터다.

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \partial f / \partial x_1 \\ \vdots \\ \partial f / \partial x_n \end{bmatrix}.
$$

편미분은 한 변수만 변수로 보고 나머지는 상수로 취급한다.

- $$f(x_1, x_2) = 3x_1 + 5x_2$$이면 $$\nabla f = (3, 5)$$로 상수 벡터다.
- $$f(x_1, x_2) = 3x_1^2 + x_1 x_2 + 5x_2^3$$이면

$$
\nabla f = \begin{bmatrix} 6x_1 + x_2 \\ x_1 + 15x_2^2 \end{bmatrix}
$$

이다. 점 $$(1, -1)$$에서 함숫값은 $$3 - 1 - 5 = -3$$이고 그래디언트는 $$(5, 16)$$이다. $$f$$의 그래프를 $$(x_1, x_2)$$ 평면 위의 곡면으로 그려 보자. 그 점을 지나며 $$x_1$$축에 평행한 평면으로 자르면, 단면 곡선의 그 점에서의 기울기가 $$5$$다. $$x_2$$축에 평행하게 자르면 기울기가 $$16$$이다. 최솟점에서는 두 기울기가 모두 0이어야 하므로 그래디언트가 영벡터여야 한다.

### 벡터 미분의 두 규칙

1. **일차항.** 상수 벡터 $$\mathbf{a}$$에 대해 $$\nabla(\mathbf{a}^T \mathbf{x}) = \nabla(\mathbf{x}^T \mathbf{a}) = \mathbf{a}$$. 예: $$3x_1 + 5x_2 = (3, 5)^T \mathbf{x}$$의 그래디언트는 $$(3, 5)$$다.
2. **이차형식.** 정사각행렬 $$M$$에 대해 $$\nabla(\mathbf{x}^T M \mathbf{x}) = (M + M^T)\mathbf{x}$$. 곱의 미분법에서 나온다. $$\mathbf{x}$$가 나타나는 곳 하나만 변수로 보고 나머지는 상수로 두어 미분한 뒤, 역할을 바꿔 더한다. 오른쪽 인수를 고정하면 $$\mathbf{x}^T (M\mathbf{x})$$는 $$\mathbf{x}$$에 대해 일차이고 그래디언트는 $$M\mathbf{x}$$다. 왼쪽 인수를 고정하면 $$(\mathbf{x}^T M)\mathbf{x} = (M^T\mathbf{x})^T \mathbf{x}$$의 그래디언트는 $$M^T \mathbf{x}$$다. $$M$$이 대칭이면 결과는 $$2M\mathbf{x}$$다.

### 그래디언트를 0으로

$$f$$의 세 항에 규칙을 적용한다. 상수 $$\mathbf{b}^T \mathbf{b}$$는 사라진다. $$-2\mathbf{x}^T A^T \mathbf{b}$$는 일차항이고 그래디언트는 $$-2A^T \mathbf{b}$$다. $$\mathbf{x}^T (A^T A) \mathbf{x}$$는 이차형식이고, $$(A^T A)^T = A^T A$$이므로 $$A^T A$$는 대칭이어서 그래디언트는 $$2A^T A \mathbf{x}$$다. 따라서

$$
\nabla f(\mathbf{x}) = -2A^T \mathbf{b} + 2A^T A \mathbf{x} = \mathbf{0}
\quad \Longleftrightarrow \quad
A^T A \mathbf{x} = A^T \mathbf{b}.
$$

다시 정규방정식이다. 기하학적 논증(잔차가 열공간에 수직)과 미분 논증(제곱오차의 그래디언트가 0)이 같은 방정식에 도달한다.

## $$A^T A$$가 가역이 아닐 때

[▶ 25강](https://www.youtube.com/watch?v=xtxo7wZyNYQ)

$$A^T A$$가 가역이면 정규방정식은 유일한 해 $$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$를 가진다. 가역이 아니면 어떻게 되는가?

### 정규방정식은 항상 해를 가진다

일반적인 정사각 시스템에서 행렬이 특이(singular)하면 해가 없거나(만나지 않는 평행선) 무수히 많다(같은 직선 두 번). 정규방정식에서는 "해가 없는" 경우가 생길 수 없다.

직관은 기하학적이다. 정규방정식의 해는 $$A\mathbf{x}$$가 $$\mathbf{b}$$에서 $$\operatorname{Col} A$$에 내린 수선의 발과 같아지는 $$\mathbf{x}$$다. 그런 수선의 발은 항상 존재한다.

- $$\operatorname{Col} A$$가 $$\mathbb{R}^3$$의 직선이나 평면이면 수선은 항상 내릴 수 있다.
- $$\operatorname{Col} A$$가 $$\mathbb{R}^3$$ 전체이면 $$\mathbf{b}$$는 이미 그 안에 있고, 가장 가까운 점은 $$\mathbf{b}$$ 자신이다.

임의의 차원에서도 마찬가지다. 또한 수선의 발 $$\hat{\mathbf{b}}$$는 항상 **유일**하다. 가장 가까운 점은 하나뿐이다.

### 해가 무수히 많다는 것은 열이 종속이라는 것

따라서 $$A^T A$$가 특이하면 정규방정식의 해는 무수히 많다. $$\hat{\mathbf{b}}$$는 유일하므로, 유일하지 않은 것은 $$\hat{\mathbf{b}}$$를 표현하는 방법이다. $$A\mathbf{x} = \hat{\mathbf{b}}$$인 가중치 벡터 $$\mathbf{x}$$가 여러 개다. 이는 $$A$$의 열들이 선형종속일 때, 즉 새로운 방향을 더하지 않는 중복된 열이 있을 때 정확히 일어난다.

**사실.** $$A^T A$$가 가역인 것은 $$A$$의 열들이 선형독립인 것과 동치다.

예: $$A$$의 열이 $$(1, 1, 1)$$과 $$(2, 2, 2)$$이고 $$\mathbf{b} = (1, 2, 3)$$이라 하자. $$\operatorname{Col} A$$는 $$(1, 1, 1)$$ 방향의 직선이고 수선의 발은 $$\hat{\mathbf{b}} = (2, 2, 2)$$다. 이때

$$
A^T A = \begin{bmatrix} 3 & 6 \\ 6 & 12 \end{bmatrix}, \qquad A^T \mathbf{b} = \begin{bmatrix} 6 \\ 12 \end{bmatrix}
$$

이고 $$A^T A$$는 특이하다. 두 방정식 모두 $$x_1 + 2x_2 = 2$$로 정리되므로 그 직선 위의 모든 $$(x_1, x_2)$$가 최소제곱해다. 모두 같은 $$A\mathbf{x} = (2, 2, 2)$$를 준다.

### 실제로는 $$A^T A$$가 거의 항상 가역이다

머신러닝에서 최소제곱 행렬 $$A$$는 샘플마다 행 하나, 특성마다 열 하나를 가지며 샘플이 특성보다 훨씬 많다. 두 특성 열이 선형종속이려면 똑같은 선형 관계가 **모든** 샘플에서 정확히 성립해야 한다.

네 사람의 몸무게가 $$(70, 60, 85, 62)$$라 하자. 키 열이 몸무게 열에 종속이려면 그 정확한 스칼라배, 예컨대 $$(140, 120, 170, 124)$$여야 한다. 모든 사람의 키가 정확히 몸무게의 두 배여야 한다는 뜻이다. 한 사람이라도 이 패턴을 벗어나면(120 대신 130) 두 열은 독립이 된다. 나이 같은 세 번째 특성이 몸무게와 키의 정확한 선형결합이 되는 경우도 마찬가지다. 샘플이 많으면 이런 정확한 우연은 사실상 일어나지 않으므로, 열은 독립이고 $$A^T A$$는 가역이며 $$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$를 쓸 수 있다.

반대 측면도 있다. 방정식이 많아질수록 열공간은 전체 공간에서 점점 더 작은 부분이 되므로, 남는 오차는 자연히 커진다. 미지수가 적은 모델은 데이터가 늘어도 모든 점을 맞출 수 없다.

## 정리

- 내적 $$\mathbf{u}^T \mathbf{v}$$는 대칭이고 각 인수에 대해 선형이다. 길이 $$\lVert \mathbf{v} \rVert = \sqrt{\mathbf{v}^T \mathbf{v}}$$, 거리 $$\lVert \mathbf{u} - \mathbf{v} \rVert$$, 그리고 $$\mathbf{u} \cdot \mathbf{v} = \lVert \mathbf{u} \rVert \lVert \mathbf{v} \rVert \cos \theta$$를 통한 각도를 준다.
- 두 벡터는 내적이 0일 때 정확히 직교한다.
- 과결정 시스템 $$A\mathbf{x} = \mathbf{b}$$는 $$\mathbf{b}$$가 $$\operatorname{Col} A$$ 바깥에 있어 보통 해가 없다. 최소제곱은 $$\lVert \mathbf{b} - A\mathbf{x} \rVert$$를 최소화하는 $$\hat{\mathbf{x}}$$를 고른다.
- 기하학적으로 $$A\hat{\mathbf{x}}$$는 $$\mathbf{b}$$에서 $$\operatorname{Col} A$$에 내린 수선의 발이다. 피타고라스 정리에 의해 열공간의 어떤 점보다도 가깝다.
- 잔차가 $$A$$의 각 열과 직교해야 한다는 조건이 정규방정식 $$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$를 준다.
- $$\lVert \mathbf{b} - A\mathbf{x} \rVert^2$$의 그래디언트를 0으로 두어도 같은 방정식이 나온다.
- 정규방정식은 항상 해를 가진다. 해가 유일한 것은 $$A$$의 열이 선형독립일 때이며, 그때 $$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$다. 샘플이 많으면 거의 항상 이 경우다.

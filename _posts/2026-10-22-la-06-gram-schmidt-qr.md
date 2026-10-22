---
title: "Linear Algebra 6: Orthogonal Projection, Gram-Schmidt, and QR"
date: 2026-10-22 10:06:00 +0900
category: study
topic: linear-algebra
lang: ko
description: 선형변환으로서의 정사영, 직교기저와 정규직교기저가 최소제곱을 단순하게 만드는 이유, 그람-슈미트 과정, 그리고 최소제곱을 더 빠르고 정확하게 푸는 QR 분해를 다룬다.
---

> [선형대수 강의](https://www.youtube.com/playlist?list=PLVuwEl3g6OcIF-EEcerd5m5-Vw5jDpKN1) 26강 정리.
> [KAIST DAVIAN Lab 기초 과목 스터디](https://davian.kaist.ac.kr/joinus)의 선형대수 강의다.
> 이전: [Linear Algebra 5]({% post_url 2026-10-22-la-05-least-squares %}) · 다음: [Linear Algebra 7]({% post_url 2026-10-22-la-07-eigendecomposition %})

이 섹션은 마지막의 질의응답을 포함해 약 1시간 30분 길이의 강의 하나를 다룬다. 여기서는 주제별로 나누어 정리한다.

## 직교기저와 정규직교기저

[▶ 26강](https://www.youtube.com/watch?v=bMAQ1dnM-_s)

### 복습: 정규방정식의 해가 여러 개일 때

최소제곱해는 정규방정식 $$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$를 만족한다. $$A^T A$$가 가역이면 해는 유일하다. 가역이 아니면 해는 무수히 많다(수선의 발은 항상 존재하므로 해가 없는 경우는 없다). 머신러닝에서는 L1, L2 정규화(regularization), 곧 라쏘(lasso)나 릿지(ridge) 회귀처럼 어떤 해를 선호할지 기준을 추가해 그중 최적의 해 하나를 고르는 방법이 있다. 다만 특성보다 샘플이 훨씬 많으면 $$A^T A$$는 거의 항상 가역이다.

### 선형변환으로서의 정사영

최소제곱해 $$\hat{\mathbf{x}}$$에 대해 점 $$A\hat{\mathbf{x}}$$는 $$\mathbf{b}$$에서 $$\operatorname{Col} A$$에 내린 수선의 발이다. 따라서 $$\mathbf{b} \mapsto A\hat{\mathbf{x}}$$는 임의의 벡터를 열공간에서 가장 가까운 점으로 보낸다. 그림으로 보면, 정오에 해가 바로 머리 위에 있을 때 하늘을 나는 새의 그림자가 땅에 생기는 위치다. 새를 지면에 수직으로 떨어뜨린 위치와 같다. 이를 $$\mathbf{b}$$의 $$\operatorname{Col} A$$ 위로의 정사영(orthogonal projection)이라 하고 $$\hat{\mathbf{b}}$$로 쓴다.

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$를 대입하면

$$
\hat{\mathbf{b}} = A(A^T A)^{-1} A^T \mathbf{b} = C\mathbf{b}, \qquad C = A(A^T A)^{-1} A^T
$$

이다. 즉 정사영 자체가 행렬 $$C$$로 주어지는 선형변환이다.

### 직교집합과 정규직교집합

$$\mathbb{R}^n$$의 벡터 집합 $$\{\mathbf{u}_1, \dots, \mathbf{u}_p\}$$가

- 서로 다른 모든 쌍이 직교하면, 즉 $$i \neq j$$일 때 $$\mathbf{u}_i \cdot \mathbf{u}_j = 0$$이면 직교집합(orthogonal set)이라 한다.
- 직교집합이면서 모든 벡터가 단위벡터, 즉 $$\lVert \mathbf{u}_i \rVert = 1$$이면 정규직교집합(orthonormal set)이라 한다.

부분공간의 기저이면서 직교(정규직교)집합인 것을 직교기저(orthogonal basis), 정규직교기저(orthonormal basis)라 한다.

**영이 아닌 벡터들의 직교집합은 선형독립이다.** 역은 성립하지 않는다. 독립인 벡터가 수직일 필요는 없다. $$\mathbb{R}^3$$에서 평행하지 않은 두 벡터 $$\mathbf{v}_1, \mathbf{v}_2$$는 평면을 생성한다. 세 번째 벡터 $$\mathbf{v}_3$$는 그 평면 밖으로 조금이라도 비스듬히 튀어나오면 독립이다. 반면 $$\mathbf{v}_3$$가 $$\mathbf{v}_1$$, $$\mathbf{v}_2$$ 모두에 수직이면 평면의 법선벡터이므로 반드시 평면 밖을 향하고, 독립성은 자동으로 성립한다. 대수적으로는 $$c_1 \mathbf{u}_1 + \cdots + c_p \mathbf{u}_p = \mathbf{0}$$의 양변을 $$\mathbf{u}_j$$와 내적하면 교차항이 모두 사라져 $$c_j \lVert \mathbf{u}_j \rVert^2 = 0$$, 곧 $$c_j = 0$$이다.

### 이번 강의의 목표

수직이 아닌 선형독립 벡터 $$\mathbf{v}_1, \dots, \mathbf{v}_p$$가 주어질 때, **같은 부분공간**을 생성하는 정규직교 벡터 $$\mathbf{u}_1, \dots, \mathbf{u}_p$$를 찾는다. 그 절차가 그람-슈미트 과정(Gram-Schmidt process)이고, 이를 행렬로 기록한 것이 QR 분해(QR factorization)다. 벡터를 수직으로 만드는 이유는 분명하다. 최소제곱을 더 빠르고 정확하게 풀 수 있다.

### 정규직교 열이 도움이 되는 이유

공식 $$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$는 깔끔하지만 역행렬이 비싼 부분이다.

- 역행렬 계산은 연산이 많다. $$n \times n$$ 행렬에 대해 대략 $$n^3$$ 규모다.
- 역행렬에는 나눗셈이 들어가고, 부동소수점 연산에서 정확도를 잃는 곳이 바로 나눗셈이다. $$1 \times 1$$ 행렬 $$[3]$$의 역은 $$1/3 = 0.333\ldots$$으로 무한소수다. 컴퓨터는 유한한 비트만 저장하므로 값은 어딘가에서 잘린다. 그러면 $$3 \times (1/3)$$은 정확히 $$1$$이 아니라 $$0.999\ldots9$$가 된다. 큰 계산에서는 이런 작은 오차가 해를 크게 왜곡할 수 있다.

이제 $$A$$를 같은 열공간을 생성하는 **정규직교 열**을 가진 행렬 $$U$$로 바꾼다고 하자. 같은 부분공간 위로 $$\mathbf{b}$$를 정사영하면 같은 점이 나와야 하므로 $$\hat{\mathbf{b}}$$는 변하지 않는다. $$U = [\mathbf{u}_1 \ \mathbf{u}_2 \ \mathbf{u}_3]$$에 대해 $$U^T U$$를 계산한다. $$(i, j)$$ 성분은 내적 $$\mathbf{u}_i^T \mathbf{u}_j$$($$U^T$$의 행과 $$U$$의 열의 곱)다.

$$
U^T U = \begin{bmatrix}
\mathbf{u}_1^T \mathbf{u}_1 & \mathbf{u}_1^T \mathbf{u}_2 & \mathbf{u}_1^T \mathbf{u}_3 \\
\mathbf{u}_2^T \mathbf{u}_1 & \mathbf{u}_2^T \mathbf{u}_2 & \mathbf{u}_2^T \mathbf{u}_3 \\
\mathbf{u}_3^T \mathbf{u}_1 & \mathbf{u}_3^T \mathbf{u}_2 & \mathbf{u}_3^T \mathbf{u}_3
\end{bmatrix}
= \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix} = I.
$$

대각 성분은 길이의 제곱이므로 1이고, 비대각 성분은 서로 다른 벡터의 내적이므로 0이다. 따라서 역행렬이 사라진다.

$$
\hat{\mathbf{x}} = (U^T U)^{-1} U^T \mathbf{b} = U^T \mathbf{b}, \qquad \hat{\mathbf{b}} = U U^T \mathbf{b}.
$$

역행렬도 나눗셈도 없다. 단 $$UU^T$$는 ($$U$$가 정사각이 아닌 한) 일반적으로 단위행렬이 **아니다**. 행렬 곱의 외적(outer product) 관점으로 쓰면

$$
U U^T = \mathbf{u}_1 \mathbf{u}_1^T + \mathbf{u}_2 \mathbf{u}_2^T + \mathbf{u}_3 \mathbf{u}_3^T
$$

이고, 이 행렬이 $$\operatorname{Col} U$$ 위로의 정사영이다.

## 정사영

### 직선 위로의 정사영

$$\mathbf{u}$$를 영이 아닌 벡터(길이가 1일 필요는 없다), $$\mathbf{y}$$를 임의의 벡터라 하자. $$\mathbf{u}$$가 생성하는 직선 위로의 $$\mathbf{y}$$의 정사영은

$$
\hat{\mathbf{y}} = \frac{\mathbf{y}^T \mathbf{u}}{\mathbf{u}^T \mathbf{u}} \, \mathbf{u}
$$

이다.

**일반 공식에서.** $$A = \mathbf{u}$$, 즉 열이 하나인 경우다. $$A^T A = \mathbf{u}^T \mathbf{u} = \lVert \mathbf{u} \rVert^2$$은 스칼라이고, $$\mathbf{u}(\mathbf{u}^T \mathbf{u})^{-1} \mathbf{u}^T \mathbf{y}$$에서 스칼라를 자유롭게 옮기면 위 공식이 된다. 결과는 $$\mathbf{u}$$의 스칼라배이며, 직선 위에 있어야 하므로 당연하다.

**직각삼각형에서.** $$\mathbf{y}$$, $$\hat{\mathbf{y}}$$, $$\mathbf{y} - \hat{\mathbf{y}}$$는 직각삼각형을 이룬다. $$\mathbf{y}$$와 $$\mathbf{u}$$ 사이의 각을 $$\theta$$라 하면 $$\hat{\mathbf{y}}$$의 길이는 $$\lVert \mathbf{y} \rVert \cos \theta$$이고, 방향은 단위벡터 $$\mathbf{u} / \lVert \mathbf{u} \rVert$$다. $$\cos \theta = \mathbf{y}^T \mathbf{u} / (\lVert \mathbf{y} \rVert \lVert \mathbf{u} \rVert)$$를 대입하면

$$
\hat{\mathbf{y}} = \lVert \mathbf{y} \rVert \cos \theta \, \frac{\mathbf{u}}{\lVert \mathbf{u} \rVert} = \frac{\mathbf{y}^T \mathbf{u}}{\lVert \mathbf{u} \rVert^2} \, \mathbf{u}
$$

이다. $$\lVert \mathbf{y} \rVert$$가 약분되어 두 유도가 일치한다.

**예.** $$\mathbf{y} = (7, 6)$$, $$\mathbf{u} = (4, 2)$$. $$\mathbf{y}^T \mathbf{u} = 40$$, $$\mathbf{u}^T \mathbf{u} = 20$$이므로 $$\hat{\mathbf{y}} = 2\mathbf{u} = (8, 4)$$다. 남은 부분 $$\mathbf{y} - \hat{\mathbf{y}} = (-1, 2)$$는 $$\mathbf{u}$$와 직교한다: $$-4 + 4 = 0$$.

$$\mathbf{u}$$가 단위벡터이면 분모가 1이 되어

$$
\hat{\mathbf{y}} = (\mathbf{y}^T \mathbf{u}) \, \mathbf{u} = \mathbf{u}\mathbf{u}^T \mathbf{y}
$$

이다. 이는 $$UU^T \mathbf{y}$$에서 열이 하나인 경우이며, 이 강의의 핵심 공식이다.

### 직교기저를 가진 부분공간 위로의 정사영

$$\{\mathbf{u}_1, \dots, \mathbf{u}_p\}$$가 부분공간 $$W$$의 **직교**기저이면, $$W$$ 위로의 정사영은 각 기저 직선 위로의 정사영의 합이다.

$$
\hat{\mathbf{y}} = \frac{\mathbf{y}^T \mathbf{u}_1}{\mathbf{u}_1^T \mathbf{u}_1} \mathbf{u}_1 + \cdots + \frac{\mathbf{y}^T \mathbf{u}_p}{\mathbf{u}_p^T \mathbf{u}_p} \mathbf{u}_p.
$$

기하학적으로, 수직인 $$\mathbf{u}_1, \mathbf{u}_2$$가 생성하는 평면에서 두 정사영은 직사각형의 두 변을 이루고, 그 직사각형의 꼭짓점이 $$\mathbf{y}$$에서 평면에 내린 수선의 발이다(학교 기하의 삼수선 정리). 이는 기저가 직교일 때만 성립한다. 직교하지 않는 기저에서 직선 정사영들을 더하면 엉뚱한 점이 나온다.

기저가 **정규직교**이면 분모가 모두 1이다.

$$
\hat{\mathbf{y}} = (\mathbf{y}^T \mathbf{u}_1)\mathbf{u}_1 + \cdots + (\mathbf{y}^T \mathbf{u}_p)\mathbf{u}_p = UU^T \mathbf{y}.
$$

**예.** $$\mathbf{u}_1 = (2, 5, -1)$$과 $$\mathbf{u}_2 = (-2, 1, 1)$$은 직교한다($$-4 + 5 - 1 = 0$$). $$\mathbf{y} = (1, 2, 3)$$이면

$$
\hat{\mathbf{y}} = \frac{9}{30} \begin{bmatrix} 2 \\ 5 \\ -1 \end{bmatrix} + \frac{3}{6} \begin{bmatrix} -2 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.4 \\ 2 \\ 0.2 \end{bmatrix}, \qquad
\mathbf{y} - \hat{\mathbf{y}} = \begin{bmatrix} 1.4 \\ 0 \\ 2.8 \end{bmatrix}.
$$

잔차가 두 기저 벡터와 모두 직교하므로($$2.8 - 2.8 = 0$$, $$-2.8 + 2.8 = 0$$) $$\hat{\mathbf{y}}$$는 실제로 수선의 발이다.

## 그람-슈미트 직교화

### 아이디어

$$\mathbb{R}^3$$의 독립 벡터 $$\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3$$에서 시작한다.

1. **첫 번째 벡터.** $$\mathbf{x}_1$$을 정규화한다. 예를 들어 $$(1, 2, 3)$$의 길이는 $$\sqrt{14}$$이므로 $$\mathbf{u}_1 = (1, 2, 3)/\sqrt{14}$$다. 생성하는 직선은 그대로다.
2. **두 번째 벡터.** 벡터를 직사각형의 두 변으로 쪼개듯, $$\mathbf{x}_2$$를 서로 수직인 두 조각으로 나눈다. $$\mathbf{u}_1$$의 직선 위로의 정사영과 나머지다. 나머지 $$\mathbf{v}_2 = \mathbf{x}_2 - (\text{정사영})$$은 $$\mathbf{u}_1$$에 수직이다. 이를 정규화해 $$\mathbf{u}_2$$를 얻는다. $$\mathbf{u}_1, \mathbf{u}_2$$가 생성하는 평면은 $$\mathbf{x}_1, \mathbf{x}_2$$가 생성하는 평면과 같다.
3. **세 번째 벡터.** $$\mathbf{x}_3$$를 $$\mathbf{u}_1, \mathbf{u}_2$$의 평면 위로 정사영해 빼고, 나머지를 정규화해 $$\mathbf{u}_3$$를 얻는다.

각 단계는 새 벡터에서 이미 만든 방향들의 성분을 모두 제거하고, 수직인 부분만 남겨 길이를 1로 맞춘다. 이를 직교화(orthogonalization), 정규화까지 하면 정규직교화(orthonormalization)라 한다.

### 알고리즘

부분공간 $$W$$의 기저 $$\mathbf{x}_1, \dots, \mathbf{x}_p$$에 대해

$$
\begin{aligned}
\mathbf{v}_1 &= \mathbf{x}_1, \\
\mathbf{v}_2 &= \mathbf{x}_2 - \frac{\mathbf{x}_2^T \mathbf{v}_1}{\mathbf{v}_1^T \mathbf{v}_1} \mathbf{v}_1, \\
\mathbf{v}_3 &= \mathbf{x}_3 - \frac{\mathbf{x}_3^T \mathbf{v}_1}{\mathbf{v}_1^T \mathbf{v}_1} \mathbf{v}_1 - \frac{\mathbf{x}_3^T \mathbf{v}_2}{\mathbf{v}_2^T \mathbf{v}_2} \mathbf{v}_2, \\
&\ \ \vdots
\end{aligned}
$$

로 두고, 마지막으로 $$\mathbf{u}_k = \mathbf{v}_k / \lVert \mathbf{v}_k \rVert$$로 정규화한다. 그러면 $$\{\mathbf{u}_1, \dots, \mathbf{u}_p\}$$는 $$W$$의 정규직교기저이고, 각 $$k$$에 대해 $$\operatorname{Span}\{\mathbf{u}_1, \dots, \mathbf{u}_k\} = \operatorname{Span}\{\mathbf{x}_1, \dots, \mathbf{x}_k\}$$다. 각 $$\mathbf{u}_k$$를 계산하자마자 정규화하면 정사영은 $$(\mathbf{x}_k^T \mathbf{u}_i)\mathbf{u}_i$$로 단순해진다.

### 계산 예

$$
\mathbf{x}_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \\ 1 \end{bmatrix}, \quad
\mathbf{x}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 1 \end{bmatrix}, \quad
\mathbf{x}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}.
$$

**1단계.** $$\lVert \mathbf{x}_1 \rVert = 2$$이므로 $$\mathbf{u}_1 = \frac{1}{2}(1, 1, 1, 1)$$.

**2단계.** $$\mathbf{x}_2^T \mathbf{u}_1 = \frac{3}{2}$$. 정사영을 뺀다.

$$
\mathbf{v}_2 = \mathbf{x}_2 - \tfrac{3}{2}\mathbf{u}_1 = \begin{bmatrix} 0 \\ 1 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} 3/4 \\ 3/4 \\ 3/4 \\ 3/4 \end{bmatrix} = \begin{bmatrix} -3/4 \\ 1/4 \\ 1/4 \\ 1/4 \end{bmatrix}.
$$

$$\lVert \mathbf{v}_2 \rVert = \frac{\sqrt{12}}{4} = \frac{\sqrt{3}}{2}$$이므로 $$\mathbf{u}_2 = \frac{1}{\sqrt{12}}(-3, 1, 1, 1)$$.

**3단계.** $$\mathbf{x}_3^T \mathbf{u}_1 = 1$$, $$\mathbf{x}_3^T \mathbf{u}_2 = \frac{2}{\sqrt{12}} = \frac{1}{\sqrt{3}}$$. 두 정사영을 모두 뺀다.

$$
\mathbf{v}_3 = \mathbf{x}_3 - 1 \cdot \mathbf{u}_1 - \tfrac{1}{\sqrt{3}} \mathbf{u}_2
= \begin{bmatrix} 0 \\ 0 \\ 1 \\ 1 \end{bmatrix} - \begin{bmatrix} 1/2 \\ 1/2 \\ 1/2 \\ 1/2 \end{bmatrix} - \begin{bmatrix} -1/2 \\ 1/6 \\ 1/6 \\ 1/6 \end{bmatrix}
= \begin{bmatrix} 0 \\ -2/3 \\ 1/3 \\ 1/3 \end{bmatrix}.
$$

$$\lVert \mathbf{v}_3 \rVert = \frac{\sqrt{6}}{3}$$이므로 $$\mathbf{u}_3 = \frac{1}{\sqrt{6}}(0, -2, 1, 1)$$.

확인: $$\mathbf{u}_1^T \mathbf{u}_2 \propto -3 + 1 + 1 + 1 = 0$$, $$\mathbf{u}_1^T \mathbf{u}_3 \propto 0 - 2 + 1 + 1 = 0$$, $$\mathbf{u}_2^T \mathbf{u}_3 \propto 0 - 2 + 1 + 1 = 0$$.

## QR 분해

### 그람-슈미트에서 $$A = QR$$로

$$A = [\mathbf{x}_1 \ \mathbf{x}_2 \ \mathbf{x}_3]$$의 열이 선형독립이고, $$Q = [\mathbf{u}_1 \ \mathbf{u}_2 \ \mathbf{u}_3]$$가 그람-슈미트로 얻은 정규직교 벡터를 담는다고 하자. 문제는 $$A = QR$$을 만족하는 행렬 $$R$$이 무엇인가다.

행렬 곱의 열 관점을 쓴다. $$QR$$의 $$j$$번째 열은 $$R$$의 $$j$$번째 열을 가중치로 한 $$Q$$의 열들의 선형결합이다. 따라서 각 $$\mathbf{x}_j$$를 $$\mathbf{u}$$들로 표현해야 한다. 그람-슈미트가 이미 그 표현을 준다.

- $$\mathbf{u}_1$$은 $$\mathbf{x}_1$$만으로 만들었다: $$\mathbf{x}_1 = \lVert \mathbf{x}_1 \rVert \, \mathbf{u}_1$$.
- $$\mathbf{x}_2$$는 $$\mathbf{u}_1$$ 위로의 정사영과 수직 성분의 합이다: $$\mathbf{x}_2 = (\mathbf{x}_2^T \mathbf{u}_1)\mathbf{u}_1 + \lVert \mathbf{v}_2 \rVert \, \mathbf{u}_2$$. $$\mathbf{u}_3$$는 필요 없으므로 그 가중치는 0이다.
- $$\mathbf{x}_3 = (\mathbf{x}_3^T \mathbf{u}_1)\mathbf{u}_1 + (\mathbf{x}_3^T \mathbf{u}_2)\mathbf{u}_2 + \lVert \mathbf{v}_3 \rVert \, \mathbf{u}_3$$.

가중치를 모으면

$$
A = QR, \qquad R = \begin{bmatrix}
\lVert \mathbf{v}_1 \rVert & \mathbf{x}_2^T \mathbf{u}_1 & \mathbf{x}_3^T \mathbf{u}_1 \\
0 & \lVert \mathbf{v}_2 \rVert & \mathbf{x}_3^T \mathbf{u}_2 \\
0 & 0 & \lVert \mathbf{v}_3 \rVert
\end{bmatrix}
$$

이다.

**정리 (QR 분해).** $$A$$가 선형독립인 열을 가진 $$m \times n$$ 행렬이면 $$A = QR$$로 쓸 수 있다. 여기서 $$Q$$는 $$\operatorname{Col} A$$를 생성하는 정규직교 열을 가진 $$m \times n$$ 행렬이고, $$R$$은 대각 성분이 양수인 $$n \times n$$ 가역 상삼각행렬(upper triangular matrix)이다.

$$R$$은 추가 비용이 없다. 성분이 모두 그람-슈미트가 어차피 계산하는 내적과 정규화 길이다. ($$Q^T Q = I$$이므로 $$R = Q^T A$$이기도 하다.)

위 계산 예에서는

$$
\begin{bmatrix} 1 & 0 & 0 \\ 1 & 1 & 0 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
= \begin{bmatrix}
1/2 & -3/\sqrt{12} & 0 \\
1/2 & 1/\sqrt{12} & -2/\sqrt{6} \\
1/2 & 1/\sqrt{12} & 1/\sqrt{6} \\
1/2 & 1/\sqrt{12} & 1/\sqrt{6}
\end{bmatrix}
\begin{bmatrix} 2 & 3/2 & 1 \\ 0 & \sqrt{3}/2 & 1/\sqrt{3} \\ 0 & 0 & \sqrt{6}/3 \end{bmatrix}
$$

이다. 예컨대 2열은 $$\frac{3}{2}\mathbf{u}_1 + \frac{\sqrt{3}}{2}\mathbf{u}_2 = \frac{3}{4}(1,1,1,1) + \frac{1}{4}(-3,1,1,1) = (0, 1, 1, 1) = \mathbf{x}_2$$이다.

### QR을 배우는 이유: 최소제곱 풀기

QR의 목적은 최소제곱의 연립방정식을 푸는 것이다. 머신러닝에서는 대개 특성보다 샘플이 훨씬 많으므로 최소제곱이 어디에나 등장한다. 정규방정식에 $$A = QR$$을 대입하면

$$
A^T A = R^T Q^T Q R = R^T R, \qquad A^T \mathbf{b} = R^T Q^T \mathbf{b}
$$

이므로 $$R^T R \hat{\mathbf{x}} = R^T Q^T \mathbf{b}$$다. $$R^T$$는 가역이므로

$$
R \hat{\mathbf{x}} = Q^T \mathbf{b}, \qquad \text{즉} \qquad \hat{\mathbf{x}} = R^{-1} Q^T \mathbf{b}
$$

로 줄어든다. 정사영은 역행렬 없이 $$\hat{\mathbf{b}} = QQ^T \mathbf{b}$$가 된다.

실제로는 $$R^{-1}$$을 만들지 않는다. $$R$$이 상삼각이므로 $$R\hat{\mathbf{x}} = Q^T \mathbf{b}$$는 후진대입(back substitution)으로 푼다. 마지막 방정식에는 미지수가 하나, 그 위 방정식에는 둘, 하는 식이다. 예를 들어

$$
\begin{bmatrix} 2 & 1 \\ 0 & 3 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} 4 \\ 6 \end{bmatrix}
\quad \Rightarrow \quad x_2 = 2, \quad x_1 = \frac{4 - 2}{2} = 1.
$$

또한 $$Q$$를 행렬 전체로 저장할 필요도 없다. 각 열 $$\mathbf{u}_k$$를 계산하는 즉시 내적 $$\mathbf{u}_k^T \mathbf{b}$$($$Q^T \mathbf{b}$$의 한 성분)와 $$R$$의 성분을 만드는 데 쓰고 버리면 된다. 신경망 추론에서 한 레이어의 중간 결과를 다음 레이어가 입력으로 쓴 뒤 메모리에서 지워도 되는 것과 비슷하다.

### 속도, 정확도, 다른 방법들

QR 계산 자체도 추가 작업이므로 그만한 가치가 있는지 물을 수 있다. 가치가 있다. $$A^T A$$를 만들고 역행렬을 구하는 비싸고 수치적으로 불안정한 과정을 피하고, 그 대신 정규직교 열과 삼각 시스템 풀이를 쓴다. 결과적으로 더 정확한 최소제곱해를 얻는다.

관련된 아이디어는 연구 논문에서 조건수(condition number)와 전처리(preconditioning)라는 이름으로 등장한다. 완전한 정규직교 $$Q$$를 구하지 않더라도, 같은 열공간을 유지하면서 $$A$$의 열들이 직교에 가깝도록 변환하면 최소제곱해가 더 믿을 만해진다.

그람-슈미트는 학부 수준에서 QR을 계산하는 방법이다. 유한 정밀도 연산에서 더 빠르거나 더 정확한 하우스홀더 반사(Householder reflection)나 기븐스 회전(Givens rotation) 같은 방법도 있다. 수치 라이브러리(예: `numpy.linalg.qr`)는 내부에서 이들 중 하나를 골라 쓰므로, 실제로는 세부 구현을 신경 쓰지 않고 활용해도 된다.

## 정리

- $$\operatorname{Col} A$$ 위로의 정사영은 선형변환 $$\hat{\mathbf{b}} = A(A^T A)^{-1} A^T \mathbf{b}$$다. 정오의 햇빛 아래 그림자처럼 $$\mathbf{b}$$를 수선의 발로 보낸다.
- 영이 아닌 벡터들의 직교집합은 선형독립이다. 정규직교집합은 길이가 모두 1인 직교집합이다.
- $$U$$의 열이 정규직교이면 $$U^T U = I$$이므로 최소제곱이 $$\hat{\mathbf{x}} = U^T \mathbf{b}$$, $$\hat{\mathbf{b}} = UU^T \mathbf{b}$$로 단순해진다. 역행렬도 나눗셈도 없다.
- 직선 위로의 정사영은 $$\hat{\mathbf{y}} = \frac{\mathbf{y}^T \mathbf{u}}{\mathbf{u}^T \mathbf{u}} \mathbf{u}$$다. 직교기저를 가진 부분공간 위로의 정사영은 각 기저 벡터 위로의 정사영을 더한 것이다.
- 그람-슈미트는 앞서 만든 방향들 위로의 정사영을 반복해서 빼고 정규화하여, 독립 벡터들을 같은 부분공간의 정규직교기저로 바꾼다.
- 그람-슈미트의 계수를 기록하면 정규직교 $$Q$$와 상삼각 $$R$$로 $$A = QR$$을 추가 비용 없이 얻는다.
- QR을 쓰면 최소제곱은 $$R\hat{\mathbf{x}} = Q^T \mathbf{b}$$가 되고 후진대입으로 풀린다. $$A^T A$$의 역행렬을 구하는 것보다 빠르고 정확하다. 라이브러리는 하우스홀더나 기븐스 방법으로도 QR을 계산한다.

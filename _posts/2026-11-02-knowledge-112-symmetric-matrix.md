---
title: "Q-112. Symmetric Matrix란?"
date: 2026-11-02 15:08:00 +0900
category: study
topic: basic
series: knowledge
lecture: 112
knowledge_group: linear-algebra
card_title: "Q 112: Symmetric Matrix"
card_desc: "Transpose · Spectral theorem · Orthogonal eigenvectors · Covariance matrix"
lang: ko
description: "대칭 행렬은 전치해도 같은 행렬로, 고유값이 모두 실수이고 고유벡터를 직교하게 고를 수 있어 직교 대각화가 가능하다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 대칭 행렬(symmetric matrix)은 전치해도 자기 자신과 같은, 즉 모든 i, j에서 (i, j) 성분과 (j, i) 성분이 같은 정방행렬입니다.
> 대칭 행렬이 중요한 이유는 스펙트럴 정리 때문으로, 고유값이 모두 실수이고 서로 직교하는 고유벡터의 기저를 항상 가집니다.
> 이 덕분에 행렬을 서로 수직인 축 방향의 단순한 늘림과 줄임으로 해석할 수 있습니다.
> 머신러닝에서는 공분산 행렬, 헤시안, 커널 행렬이 모두 대칭이어서 자주 등장합니다.
> 예를 들어 PCA는 대칭인 공분산 행렬을 직교 대각화해 주성분을 구합니다.

## 자세한 설명

### 정의

정방행렬 $$A$$가 $$A = A^\top$$, 즉 $$a_{ij} = a_{ji}$$이면 **대칭**이다. 임의의 행렬 $$X$$에 대해 $$X^\top X$$와 $$XX^\top$$은 항상 대칭이다.

$$
(X^\top X)^\top = X^\top (X^\top)^\top = X^\top X
$$

### 스펙트럴 정리

실수 대칭 행렬은 다음과 같이 분해된다. $$Q$$는 고유벡터를 열로 갖는 직교 행렬($$Q^\top Q = I$$), $$\Lambda$$는 고유값을 담은 실수 대각 행렬이다.

$$
A = Q\Lambda Q^\top = \sum_{i=1}^{n}\lambda_i\mathbf{q}_i\mathbf{q}_i^\top
$$

서로 다른 고유값의 고유벡터가 직교한다는 사실은 다음에서 바로 나온다.

$$
\lambda_1\mathbf{q}_1^\top\mathbf{q}_2 = (A\mathbf{q}_1)^\top\mathbf{q}_2 = \mathbf{q}_1^\top A\mathbf{q}_2 = \lambda_2\mathbf{q}_1^\top\mathbf{q}_2
$$

$$\lambda_1 \ne \lambda_2$$이면 $$\mathbf{q}_1^\top\mathbf{q}_2 = 0$$이어야 한다.

### 수치 예제

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$$의 고유값은 3과 1이고, 고유벡터는 $$(1, 1)/\sqrt{2}$$와 $$(1, -1)/\sqrt{2}$$로 서로 직교한다.

반면 대칭이 아닌 $$\begin{bmatrix} 1 & 1 \\ 0 & 2 \end{bmatrix}$$의 고유벡터는 $$(1, 0)$$과 $$(1, 1)$$로 직교하지 않는다.

### 2차 형식과 대칭 부분

임의의 정방행렬은 대칭 부분과 반대칭 부분으로 나눌 수 있다.

$$
A = \underbrace{\frac{A + A^\top}{2}}_{\text{symmetric}} + \underbrace{\frac{A - A^\top}{2}}_{\text{skew-symmetric}}
$$

반대칭 부분은 $$\mathbf{x}^\top(\cdot)\mathbf{x}$$에 기여하지 않는다. 따라서 2차 형식 $$\mathbf{x}^\top A\mathbf{x}$$를 다룰 때는 $$A$$를 대칭으로 가정해도 일반성을 잃지 않는다.

### 머신러닝에 등장하는 대칭 행렬

| 행렬 | 정의 | 대칭인 이유 | 쓰임 |
|---|---|---|---|
| 공분산 행렬 | $$\Sigma_{ij} = \text{Cov}(x_i, x_j)$$ | 공분산은 순서와 무관 | PCA, 정규 분포 |
| 헤시안 | $$H_{ij} = \partial^2 f / \partial x_i \partial x_j$$ | 2차 미분이 연속이면 순서와 무관 | 뉴턴 방법, 곡률 |
| 커널 행렬 | $$K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$$ | 커널이 대칭 | SVM, GP |
| 그래프 라플라시안 | $$L = D - W$$ | 무향 그래프의 $$W$$가 대칭 | 스펙트럴 클러스터링 |

## 관련 개념

- **스펙트럴 정리 증명**: [Linear Algebra 8](/blog/2026/10/22/la-08-symmetric-spectral/)에서 다룬다.
- **양의 정부호**: 고유값이 모두 양수인 대칭 행렬이다. [Q-113](/blog/2026/11/02/knowledge-113-positive-definite/)에서 다룬다.
- **고유값 분해**: 일반 행렬의 경우는 [Q-116](/blog/2026/11/02/knowledge-116-eigenvector/)과 [Linear Algebra 7](/blog/2026/10/22/la-07-eigendecomposition/)을 참고한다.
- **PCA**: 공분산 행렬의 직교 대각화를 활용한다. [Q-43](/blog/2026/11/02/knowledge-43-pca-rotation/)을 참고한다.

## 꼬리 질문

**Q.** 대칭 행렬은 항상 역행렬이 있나요?
아닙니다. 고유값에 0이 있으면 역행렬이 없습니다. 예를 들어 모든 성분이 1인 행렬은 대칭이지만 특이 행렬입니다.

**Q.** 두 대칭 행렬의 곱도 대칭인가요?
일반적으로 아닙니다. $$(AB)^\top = BA$$이므로 $$A$$와 $$B$$가 교환 가능할 때만 대칭입니다.

**Q.** 복소수 행렬에서는 어떻게 되나요?
켤레 전치에 대해 같은 $$A = A^{\mathsf{H}}$$인 에르미트 행렬이 대칭 행렬의 역할을 합니다. 고유값은 실수이고 고유벡터는 유니터리로 대각화됩니다.

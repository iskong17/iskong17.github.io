---
title: "Q-117. Eigen Vector는 왜 중요한가?"
date: 2026-11-02 15:13:00 +0900
category: study
topic: basic
series: knowledge
lecture: 117
knowledge_group: linear-algebra
card_title: "Q 117: Why Eigenvectors Matter"
card_desc: "Decoupling · Matrix powers · PCA · PageRank · Spectral clustering"
lang: ko
description: "고유벡터는 복잡한 행렬 연산을 축별 곱셈으로 분해해 주기 때문에 중요하며, PCA, PageRank, 스펙트럴 클러스터링, 시스템 안정성 분석의 기반이 된다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 고유벡터가 중요한 이유는 복잡한 행렬 변환을 서로 간섭하지 않는 축별 곱셈으로 분해해 주기 때문입니다.
> 고유벡터 방향에서는 행렬이 단순히 스칼라 곱처럼 작동하므로, 행렬의 거듭제곱이나 장기적인 동작을 분석하기 쉬워집니다.
> 예를 들어 마르코프 연쇄를 오랫동안 반복하면 고유값 1의 고유벡터인 정상 분포로 수렴합니다.
> 머신러닝에서는 PCA의 주성분이 공분산 행렬의 고유벡터이고, 스펙트럴 클러스터링은 그래프 라플라시안의 고유벡터로 군집 구조를 찾습니다.
> 최적화에서도 헤시안의 고유벡터가 곡률이 큰 방향과 작은 방향을 알려 주어 수렴 속도를 설명합니다.

## 자세한 설명

고유벡터의 정의는 [Q-116](/blog/2026/11/02/knowledge-116-eigenvector/)에서 다룬다.

### 핵심: 축별로 분리된다

대각화 가능한 $$A = V\Lambda V^{-1}$$에서 임의의 벡터를 고유벡터의 결합 $$\mathbf{x} = \sum_i c_i\mathbf{v}_i$$로 쓰면 $$k$$번 곱한 결과가 다음과 같이 간단해진다.

$$
A^k\mathbf{x} = \sum_{i} c_i\lambda_i^k\mathbf{v}_i
$$

행렬 곱은 복잡하지만, 고유벡터 좌표계에서는 각 좌표에 $$\lambda_i^k$$만 곱하면 된다. 큰 $$k$$에서는 $$\lvert\lambda_i\rvert$$가 가장 큰 항이 지배한다.

### 수치 예제: 마르코프 연쇄

두 상태의 전이 행렬이 다음과 같다고 하자. 열의 합이 1이다.

$$
P = \begin{bmatrix} 0.9 & 0.5 \\ 0.1 & 0.5 \end{bmatrix}
$$

고유값은 1과 0.4이고, 대응하는 고유벡터는 $$\boldsymbol{\pi} = (5/6, 1/6)$$과 $$(1, -1)$$이다. 시작 분포 $$(1, 0)$$은 다음과 같이 분해된다.

$$
\begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 5/6 \\ 1/6 \end{bmatrix} + \frac{1}{6}\begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

따라서 $$k$$단계 후의 분포는 $$\boldsymbol{\pi} + \tfrac{1}{6}(0.4)^k(1, -1)$$이다.

| $$k$$ | 3 | 10 | $$\infty$$ |
|---|---:|---:|---:|
| 첫 상태의 확률 | 0.8440 | 0.8334 | 0.8333 |

고유값 1의 벡터는 그대로 남고, 고유값 0.4의 성분은 빠르게 사라진다. 이처럼 고유벡터는 극한을, 두 번째 고유값은 수렴 속도를 알려 준다.

### 활용

| 분야 | 행렬 | 고유벡터의 의미 |
|---|---|---|
| PCA | 공분산 행렬 | 분산이 큰 순서의 직교 축 |
| PageRank | 링크 전이 행렬 | 가장 큰 고유값의 벡터가 페이지 중요도 |
| 스펙트럴 클러스터링 | 그래프 라플라시안 | 작은 고유값의 벡터가 군집 구조 |
| 최적화 | 헤시안 | 곡률의 주축 |
| RNN과 동역학 | 가중치 또는 상태 전이 행렬 | 증폭 또는 감쇠되는 방향 |

### 최적화 예시

2차 함수 $$f(\mathbf{x}) = \tfrac{1}{2}\mathbf{x}^\top H\mathbf{x}$$에 경사하강법을 쓰면 $$\mathbf{x}_{t+1} = (I - \eta H)\mathbf{x}_t$$이다. $$H$$의 고유벡터 좌표계에서는 각 좌표가 독립적으로 $$(1 - \eta\lambda_i)$$배가 된다. 곡률이 큰 축은 금세 수렴하거나 진동하고, 작은 축은 천천히 움직인다. 구체적인 값은 [Q-118](/blog/2026/11/02/knowledge-118-eigenvalue/)에서 계산한다.

### 한계

대각화할 수 없는 행렬에는 이 분해를 쓸 수 없고, 고유벡터가 직교하지 않으면 고유값의 절댓값이 모두 1보다 작아도 일시적으로 벡터가 커질 수 있다. 비정방 행렬이나 수치적으로 안정된 분해가 필요할 때는 [SVD](/blog/2026/11/02/knowledge-119-svd/)가 알맞다.

## 관련 개념

- **PCA**: 공분산 행렬의 고유벡터를 사용한다. [Q-25](/blog/2026/11/02/knowledge-25-dimensionality-reduction/)와 [Machine Learning 15](/blog/2026/10/22/ml-15-pca-ica/)를 참고한다.
- **대칭 행렬**: 고유벡터가 직교하므로 좌표 변환이 특히 깔끔하다. [Q-112](/blog/2026/11/02/knowledge-112-symmetric-matrix/)를 참고한다.
- **기저**: 고유벡터는 행렬이 대각이 되는 기저이다. [Q-110](/blog/2026/11/02/knowledge-110-basis-dimension/)을 참고한다.
- **RNN의 기울기**: 곱해지는 행렬이 커지거나 줄어드는 문제와 연결된다. [Q-93](/blog/2026/11/02/knowledge-93-rnn/)을 참고한다.

## 꼬리 질문

**Q.** 멱반복법은 왜 가장 큰 고유값의 벡터로 수렴하나요?
$$A^k\mathbf{x}$$에서 $$\lvert\lambda_1\rvert$$의 항이 나머지보다 빠르게 커지기 때문입니다. 시작 벡터에 그 성분이 있고 $$\lvert\lambda_1\rvert > \lvert\lambda_2\rvert$$이면 수렴하며, 속도는 $$\lvert\lambda_2/\lambda_1\rvert$$로 결정됩니다.

**Q.** 왜 PCA의 축이 고유벡터가 되나요?
단위 벡터 $$\mathbf{u}$$ 방향의 분산은 $$\mathbf{u}^\top\Sigma\mathbf{u}$$입니다. 이를 제약 조건 $$\lVert\mathbf{u}\rVert = 1$$로 최대화하면 라그랑주 조건이 $$\Sigma\mathbf{u} = \lambda\mathbf{u}$$가 됩니다.

**Q.** 스펙트럴 클러스터링에서는 왜 작은 고유값을 쓰나요?
라플라시안의 2차 형식 $$\mathbf{x}^\top L\mathbf{x}$$는 연결된 노드의 값 차이를 잽니다. 작은 고유값의 벡터는 연결된 노드들에서 값이 비슷해 군집 구조를 드러냅니다.

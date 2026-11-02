---
title: "Q-118. Eigen Value란?"
date: 2026-11-02 15:14:00 +0900
category: study
topic: basic
series: knowledge
lecture: 118
knowledge_group: linear-algebra
card_title: "Q 118: Eigenvalue"
card_desc: "Scaling factor · Trace and determinant · Explained variance · Condition number"
lang: ko
description: "고유값은 고유벡터 방향으로 행렬이 벡터를 몇 배 하는지를 나타내는 값으로, 특성 방정식의 근이며 곡률, 분산, 안정성을 해석하는 데 쓰인다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 고유값(eigenvalue)은 Av = λv에서 고유벡터 방향으로 행렬이 벡터를 몇 배 하는지를 나타내는 스칼라 λ입니다.
> 절댓값이 1보다 크면 그 방향으로 늘어나고, 1보다 작으면 줄어들며, 음수이면 뒤집히고, 0이면 그 방향의 정보가 사라집니다.
> 고유값의 합은 대각합, 곱은 행렬식과 같아서 행렬의 전체적인 성질을 요약합니다.
> 예를 들어 PCA에서 공분산 행렬의 고유값은 각 주성분의 분산이어서, 고유값이 4와 1이면 첫 주성분이 전체 분산의 80%를 설명합니다.
> 또 헤시안의 고유값은 곡률이어서 경사하강법의 학습률 상한과 수렴 속도를 결정합니다.

## 자세한 설명

### 구하기

고유값은 특성 방정식 $$\det(A - \lambda I) = 0$$의 근이다. $$n \times n$$ 행렬에서 이 식은 $$n$$차 다항식이고, 복소수까지 중복을 포함해 $$n$$개의 근을 가진다.

**수치 예제.** $$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$이면 $$\lambda^2 - 7\lambda + 10 = 0$$이므로 $$\lambda = 5, 2$$이다. 합 7은 대각합 $$4 + 3$$과 같고, 곱 10은 행렬식 $$4 \cdot 3 - 1 \cdot 2$$와 같다.

$$
\text{tr}(A) = \sum_{i}\lambda_i, \qquad \det(A) = \prod_{i}\lambda_i
$$

### 값으로 알 수 있는 것

| 고유값 | 의미 |
|---|---|
| $$\lvert\lambda\rvert > 1$$ | 반복 적용 시 그 방향이 커짐 |
| $$\lvert\lambda\rvert < 1$$ | 반복 적용 시 그 방향이 사라짐 |
| $$\lambda = 0$$ | 특이 행렬, 영공간이 존재 |
| 모두 양수(대칭 행렬) | [양의 정부호](/blog/2026/11/02/knowledge-113-positive-definite/) |
| 복소수 | 회전 성분이 포함됨 |

### 머신러닝에서의 해석

**PCA의 분산.** 공분산 행렬의 고유값 $$\lambda_i$$는 해당 주성분 방향의 분산이다. 설명된 분산 비율은 다음과 같다.

$$
\text{explained variance ratio}_i = \frac{\lambda_i}{\sum_{j}\lambda_j}
$$

**경사하강법의 안정성.** 2차 함수에서 각 고유 방향의 오차는 매 단계 $$(1 - \eta\lambda_i)$$배가 된다. $$\eta$$는 학습률, $$\lambda_i$$는 헤시안의 고유값이다. 따라서 수렴 조건은 다음과 같다.

$$
0 < \eta < \frac{2}{\lambda_{\max}}
$$

헤시안의 고유값이 100과 1이면 $$\eta < 0.02$$여야 한다. $$\eta = 0.01$$이면 큰 곡률 방향은 $$1 - 0.01 \cdot 100 = 0$$배로 한 번에 수렴하지만, 작은 곡률 방향은 매번 0.99배밖에 줄지 않는다. 비율 $$\kappa = \lambda_{\max}/\lambda_{\min}$$가 **조건수**이며, 이 값이 크면 수렴이 느리다.

**순환 구조의 안정성.** 선형 RNN에서 상태 전이 행렬의 고유값 절댓값이 1보다 크면 기울기가 폭발하고, 1보다 작으면 사라지는 경향이 있다.

### 고유값과 특이값

둘은 비슷해 보이지만 다르다. 특이값은 항상 0 이상의 실수이고 모든 크기의 행렬에 정의된다. 다음 전단 행렬이 차이를 보여 준다.

$$
\begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}: \quad \text{eigenvalues } 1, 1, \quad \text{singular values } 1.618, 0.618
$$

고유값만 보면 크기가 변하지 않을 것 같지만, 실제로는 어떤 방향의 벡터를 1.618배까지 늘린다. 대칭 행렬에서는 특이값이 고유값의 절댓값과 같다.

## 관련 개념

- **고유벡터**: 고유값에 대응하는 방향이다. [Q-116](/blog/2026/11/02/knowledge-116-eigenvector/)을 참고한다.
- **행렬식**: 고유값의 곱이다. [Q-115](/blog/2026/11/02/knowledge-115-determinant/)를 참고한다.
- **뉴턴 방법**: 헤시안의 역행렬로 곡률의 차이를 보정한다. [Q-94](/blog/2026/11/02/knowledge-94-newton-method/)를 참고한다.
- **SVD**: 특이값은 $$A^\top A$$의 고유값의 제곱근이다. [Q-119](/blog/2026/11/02/knowledge-119-svd/)를 참고한다.

## 꼬리 질문

**Q.** $$A^2$$와 $$A^{-1}$$의 고유값은 어떻게 되나요?
고유벡터는 그대로이고 고유값은 각각 $$\lambda^2$$, $$1/\lambda$$입니다. $$A\mathbf{v} = \lambda\mathbf{v}$$에 $$A$$나 $$A^{-1}$$을 한 번 더 적용하면 바로 나옵니다.

**Q.** 릿지 정규화는 고유값에 어떤 영향을 주나요?
$$X^\top X$$의 고유값 $$\lambda_i$$를 $$\lambda_i + \alpha$$로 옮깁니다. 작은 고유값이 커지므로 조건수가 줄어들어 역행렬 계산이 안정됩니다.

**Q.** 고유값을 직접 계산하기 어려울 때 범위를 알 수 있나요?
네. 게르슈고린 원판 정리는 각 고유값이 대각 성분을 중심으로 하고, 그 행의 나머지 성분 절댓값 합을 반지름으로 하는 원판 중 하나에 있음을 알려 줍니다.

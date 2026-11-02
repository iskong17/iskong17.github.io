---
title: "Q-119. SVD란 무엇인가? 중요한 이유는?"
date: 2026-11-02 15:15:00 +0900
category: study
topic: basic
series: knowledge
lecture: 119
knowledge_group: linear-algebra
card_title: "Q 119: SVD"
card_desc: "Singular values · Low-rank approximation · Pseudoinverse · PCA · Numerical stability"
lang: ko
description: "SVD는 모든 행렬을 회전, 축 방향 스케일링, 회전의 곱으로 분해하는 방법으로, 최적의 저랭크 근사와 안정적인 수치 계산을 제공하기 때문에 중요하다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> SVD(singular value decomposition, 특이값 분해)는 임의의 행렬을 A = UΣV 전치로 분해하는 방법입니다. U와 V는 직교 행렬이고, Σ는 0 이상의 특이값을 담은 대각 행렬입니다.
> 기하학적으로는 어떤 선형 변환이든 회전 또는 반사, 축 방향 스케일링, 회전 또는 반사의 순서로 나타낼 수 있다는 뜻입니다.
> 고유값 분해와 달리 정방행렬이 아니어도 항상 존재하고, 큰 특이값만 남기면 랭크가 낮은 근사 중 오차가 가장 작은 행렬이 되어 중요합니다.
> 또한 랭크와 조건수를 알려 주고 수치적으로 안정적이어서, 의사역행렬과 최소제곱 계산에 쓰입니다.
> 예를 들어 PCA는 보통 중심화한 데이터 행렬의 SVD로 계산하고, 추천 시스템과 잠재 의미 분석도 같은 원리의 저랭크 근사를 사용합니다.

## 자세한 설명

### 정의

$$m \times n$$ 행렬 $$A$$는 항상 다음과 같이 분해된다. $$U$$는 $$m \times m$$ 직교 행렬, $$V$$는 $$n \times n$$ 직교 행렬이고, $$\Sigma$$는 대각에 $$\sigma_1 \ge \sigma_2 \ge \dots \ge 0$$을 담은 $$m \times n$$ 행렬이다. $$r$$은 랭크이다.

$$
A = U\Sigma V^\top = \sum_{i=1}^{r}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top
$$

고유값 분해와는 다음과 같이 연결된다.

$$
A^\top A = V\Sigma^\top\Sigma V^\top, \qquad AA^\top = U\Sigma\Sigma^\top U^\top
$$

즉, $$\mathbf{v}_i$$와 $$\mathbf{u}_i$$는 각각 $$A^\top A$$와 $$AA^\top$$의 고유벡터이고, $$\sigma_i^2$$은 그 고유값이다.

### 수치 예제

$$
A = \begin{bmatrix} 3 & 0 \\ 4 & 5 \end{bmatrix}, \qquad A^\top A = \begin{bmatrix} 25 & 20 \\ 20 & 25 \end{bmatrix}
$$

$$A^\top A$$의 고유값은 45와 5이므로 $$\sigma_1 = \sqrt{45} \approx 6.708$$, $$\sigma_2 = \sqrt{5} \approx 2.236$$이다. 오른쪽 특이 벡터는 $$\mathbf{v}_1 = (1, 1)/\sqrt{2}$$, $$\mathbf{v}_2 = (1, -1)/\sqrt{2}$$이다. 왼쪽 특이 벡터는 $$\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$$로 구하면 $$\mathbf{u}_1 = (1, 3)/\sqrt{10}$$, $$\mathbf{u}_2 = (3, -1)/\sqrt{10}$$이다.

첫 항만 남기면 다음과 같다.

$$
A_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top = \begin{bmatrix} 1.5 & 1.5 \\ 4.5 & 4.5 \end{bmatrix}
$$

오차의 프로베니우스 노름은 $$\lVert A - A_1\rVert_F = \sigma_2 \approx 2.236$$이고, 첫 항은 전체 제곱합의 $$45/50 = 90\%$$를 설명한다.

### 중요한 이유

**1. 항상 존재한다.** 고유값 분해는 정방행렬이어야 하고 대각화 가능해야 하지만, SVD는 모든 실수 행렬에 대해 존재한다.

**2. 최적의 저랭크 근사를 준다.** 상위 $$k$$개 항으로 만든 $$A_k$$는 랭크 $$k$$ 이하의 행렬 중 오차가 가장 작다(Eckart-Young 정리). 프로베니우스 노름과 스펙트럴 노름 모두에서 성립한다.

$$
\min_{\text{rank}(B) \le k}\lVert A - B\rVert_F = \lVert A - A_k\rVert_F = \sqrt{\sum_{i > k}\sigma_i^2}
$$

**3. 행렬의 구조를 드러낸다.** 0이 아닌 특이값의 개수가 랭크이고, $$\sigma_1/\sigma_r$$이 조건수이다. 네 기본 부분 공간의 직교 기저도 $$U$$와 $$V$$의 열로 바로 얻는다.

**4. 수치적으로 안정적이다.** $$X^\top X$$를 만들면 조건수가 제곱되므로, 나쁜 조건의 행렬에서는 $$X$$에 직접 SVD를 쓰는 것이 더 정확하다.

### 머신러닝에서의 활용

| 활용 | 방법 |
|---|---|
| PCA | 중심화한 $$X$$의 $$\mathbf{v}_i$$가 주성분, $$\sigma_i^2/(n-1)$$이 분산 |
| 최소제곱과 의사역행렬 | $$A^{+} = V\Sigma^{+}U^\top$$, 0이 아닌 특이값만 역수 |
| 잠재 의미 분석 | 단어-문서 행렬의 저랭크 근사 |
| 추천 시스템 | 사용자-아이템 행렬의 저차원 요인 |
| 압축과 잡음 제거 | 작은 특이값 제거 |

참고로 평점 행렬처럼 값이 비어 있는 데이터에는 표준 SVD를 그대로 쓸 수 없어, 관측된 값에 대해 행렬 분해를 최적화하는 변형을 사용한다.

## 관련 개념

- **자세한 유도**: 기하학적 설명과 유도는 [Linear Algebra 9](/blog/2026/10/22/la-09-svd/)와 [SVD 심화](/blog/2026/10/23/see-matrix-03-svd/)에서 다룬다.
- **고유값**: 특이값과의 차이는 [Q-118](/blog/2026/11/02/knowledge-118-eigenvalue/)을 참고한다.
- **랭크**: 0이 아닌 특이값의 개수이다. [Q-114](/blog/2026/11/02/knowledge-114-rank/)를 참고한다.
- **차원 축소**: PCA와의 연결은 [Q-25](/blog/2026/11/02/knowledge-25-dimensionality-reduction/)를 참고한다.

## 꼬리 질문

**Q.** SVD로 PCA를 할 때의 장점은 무엇인가요?
공분산 행렬을 만들지 않아 조건수가 제곱되는 문제를 피합니다. 또한 특징 수가 많을 때 필요한 성분만 근사 계산할 수 있습니다.

**Q.** 대규모 행렬에서는 어떻게 계산하나요?
전체 SVD는 비용이 커서 상위 $$k$$개만 구하는 Lanczos 방법이나 무작위 SVD(randomized SVD)를 사용합니다.

**Q.** SVD는 유일한가요?
특이값은 유일합니다. 특이 벡터는 짝을 이룬 부호 변경이 가능하고, 특이값이 중복되면 해당 부분 공간 안에서 기저를 자유롭게 고를 수 있습니다.

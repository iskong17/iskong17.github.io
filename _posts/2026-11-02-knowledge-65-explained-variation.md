---
title: "Q-65. Explained variation이란 무엇인가요?"
date: 2026-11-02 14:21:00 +0900
category: study
topic: basic
series: knowledge
lecture: 65
knowledge_group: statistics
card_title: "Q 65: Explained Variation"
card_desc: "SSR · F-test · Regression · ANOVA · Explained variance"
lang: ko
description: "Explained variation은 모델의 예측값이 평균에서 벗어난 정도의 제곱합 SSR로, 총 변동 중 모델이 입력으로 설명한 부분이다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> Explained variation은 총 변동 중 모델이 입력 변수로 설명해 낸 부분으로, 예측값이 전체 평균에서 벗어난 정도를 제곱해 더한 SSR입니다. 평균만으로 예측하던 기준 모델과 비교했을 때 회귀 모델이 줄여 준 오차로 해석할 수 있습니다. 절편이 있는 OLS에서는 총 변동 SST에서 잔차 제곱합 SSE를 뺀 값과 같습니다. 예를 들어 SST가 6이고 SSE가 2.4이면 SSR은 3.6이므로, 변동의 60%를 설명한 것입니다. 다만 여기서 설명했다는 것은 통계적으로 함께 변한다는 뜻이지 인과 관계가 있다는 뜻은 아닙니다. 또 학습 데이터에서는 변수를 추가할수록 SSR이 줄지 않으므로, 값이 커졌다고 곧 더 좋은 모델은 아닙니다.

## 자세한 설명

### 정의

**설명된 변동(explained variation, regression sum of squares)**은 다음과 같다. $$\hat{y}_i$$는 모델의 예측값, $$\bar{y}$$는 관측값의 평균이다.

$$
\text{SSR} = \sum_{i=1}^{n} (\hat{y}_i - \bar{y})^2
$$

절편이 있는 OLS에서는 $$\text{SST} = \text{SSR} + \text{SSE}$$가 성립한다. 유도와 조건은 [Q-64](/blog/2026/11/02/knowledge-64-total-variation/)에서 다뤘다. 입력 변수가 $$p$$개이면 자유도는 $$p$$이다.

단순 선형 회귀에서는 $$\hat{y}_i - \bar{y} = \hat{\beta}_1 (x_i - \bar{x})$$이므로 다음과 같이 쓸 수 있다. 기울기가 크고 $$x$$가 넓게 퍼져 있을수록 SSR이 크다.

$$
\text{SSR} = \hat{\beta}_1^2 \sum_{i} (x_i - \bar{x})^2
$$

[Q-64](/blog/2026/11/02/knowledge-64-total-variation/)의 예제에서는 $$\hat{\beta}_1 = 0.6$$, $$\sum_i (x_i - \bar{x})^2 = 10$$이므로 $$\text{SSR} = 0.36 \times 10 = 3.6$$이다. 표에서 직접 더한 값과 일치한다.

### 설명된 변동이 우연보다 큰가: F 검정

입력이 $$y$$와 무관해도 SSR은 0보다 크게 나온다. 그래서 자유도당 설명된 변동을 자유도당 잔차 변동과 비교한다.

$$
F = \frac{\text{SSR}/p}{\text{SSE}/(n - p - 1)}
$$

오차가 독립이고 등분산인 정규 분포이면, 모든 기울기가 0이라는 귀무가설에서 이 값은 $$F_{p,\, n-p-1}$$ 분포를 따른다. 예제에서는 $$F = 3.6 / (2.4 / 3) = 4.5$$이고 p-value는 약 0.124이다. 변동의 60%를 설명했지만, 점이 5개뿐이라 유의수준 5%에서 우연을 배제하지 못한다. 설명 비율과 통계적 유의성은 다른 질문이다.

### 비슷한 이름의 개념

| 용어 | 정의 | 맥락 |
|---|---|---|
| SSR | $$\sum_i (\hat{y}_i - \bar{y})^2$$ | 회귀 |
| 집단 간 제곱합 | $$\sum_g n_g (\bar{y}_g - \bar{y})^2$$ | ANOVA, 예측값이 그룹 평균 $$\bar{y}_g$$인 SSR |
| Explained variance score | $$1 - \text{Var}(y - \hat{y}) / \text{Var}(y)$$ | 회귀 평가 지표. 잔차의 평균을 빼므로 체계적 편향을 벌점으로 보지 않음 |
| PCA의 explained variance | 주성분의 분산, 즉 공분산 행렬의 고유값 | 비지도 차원 축소. 예측 대상 $$y$$가 없음 |

### 해석 시 주의

- **인과 아님**: 교란 변수가 있으면 $$x$$가 원인이 아니어도 SSR이 커진다.
- **학습 데이터 기준**: 변수를 늘리면 SSR은 감소하지 않으므로 과적합과 구분되지 않는다. 검증 데이터나 조정 결정계수로 확인해야 한다.
- **기여도 분리의 어려움**: 입력들이 상관되어 있으면 SSR을 변수별로 나누는 값이 투입 순서에 따라 달라진다.

## 관련 개념

- **Total variation (SST)**: 설명의 대상이 되는 전체 변동이다. [Q-64](/blog/2026/11/02/knowledge-64-total-variation/)에서 다룬다.
- **Unexplained variation (SSE)**: 설명하지 못한 나머지이다. [Q-66](/blog/2026/11/02/knowledge-66-unexplained-variation/)에서 다룬다.
- **결정계수**: SSR을 SST로 나눈 비율이다. [Q-67](/blog/2026/11/02/knowledge-67-r-squared/)에서 다룬다.
- **에타 제곱**: ANOVA에서 집단 간 제곱합의 비율이다. [Q-40](/blog/2026/11/02/knowledge-40-continuous-categorical-correlation/)을 참고한다.
- **PCA의 설명 분산**: [Q-25](/blog/2026/11/02/knowledge-25-dimensionality-reduction/)와 [Machine Learning 15강](/blog/2026/10/22/ml-15-pca-ica/)에서 차원 축소의 맥락으로 다룬다.

## 꼬리 질문

**Q.** 변수를 추가했을 때 학습 SSR이 줄지 않는 이유는 무엇인가요?
새 변수의 계수를 0으로 두면 이전 모델과 같으므로, OLS는 최소한 이전만큼의 SSE를 달성할 수 있습니다. SST는 고정이므로 SSR은 줄어들 수 없습니다.

**Q.** SSR이 0이면 어떤 의미인가요?
모든 예측값이 평균과 같다는 뜻으로, 선형 모델이 평균 이상의 설명을 하지 못한 것입니다. 관계가 없다는 뜻은 아닙니다. $$y = x^2$$처럼 대칭인 비선형 관계에서도 직선의 SSR은 0일 수 있습니다.

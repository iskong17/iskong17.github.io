---
title: "Q-107. Support Vector Machine이란 무엇인가?"
date: 2026-11-02 15:03:00 +0900
category: study
topic: basic
series: knowledge
lecture: 107
knowledge_group: machine-learning
card_title: "Q 107: Support Vector Machine"
card_desc: "Maximum margin · Hinge loss · Dual problem · Kernel trick · Support vectors"
lang: ko
description: "SVM은 두 클래스 사이의 마진을 최대화하는 초평면을 찾는 모델로, 결정 경계는 서포트 벡터로만 정해지고 커널로 비선형 경계로 확장된다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> SVM(support vector machine)은 두 클래스를 나누는 초평면 중에서 가장 가까운 샘플까지의 거리, 즉 마진이 가장 큰 초평면을 찾는 분류 모델입니다.
> 경계와 가까운 샘플까지 여유를 크게 두면 새 데이터의 작은 변화에도 예측이 덜 바뀌기 때문입니다.
> 결정 경계는 경계에 가장 가까운 서포트 벡터만으로 정해지고, 나머지 샘플은 조금 움직여도 경계에 영향을 주지 않습니다.
> 데이터가 완벽히 나뉘지 않으면 슬랙 변수로 위반을 허용하고, 이때 C가 위반에 대한 벌점을 조절합니다.
> 비선형 문제는 RBF 같은 커널로 해결합니다. 예를 들어 원 안팎으로 나뉘는 데이터도 RBF 커널을 쓰면 입력 공간에서 곡선 경계를 만들 수 있습니다.

## 자세한 설명

### 하드 마진

레이블 $$y_i \in \{-1, +1\}$$, 초평면 $$\mathbf{w}^\top\mathbf{x} + b = 0$$에 대해, 샘플과 초평면의 거리는 $$\lvert \mathbf{w}^\top\mathbf{x} + b \rvert / \lVert\mathbf{w}\rVert$$이다. $$\mathbf{w}$$와 $$b$$의 크기를 같이 바꿔도 초평면이 그대로이므로, 가장 가까운 샘플에서 $$y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$$이 되도록 맞춘다. 그러면 마진은 $$1/\lVert\mathbf{w}\rVert$$이고, 최대화 문제는 다음 볼록 2차 계획 문제가 된다.

$$
\min_{\mathbf{w}, b} \frac{1}{2}\lVert\mathbf{w}\rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1, \quad i = 1, \dots, n
$$

**수치 예제.** $$(2, 2)$$는 양성, $$(0, 0)$$은 음성이라 하자. 해는 $$\mathbf{w} = (0.5, 0.5)$$, $$b = -1$$이다. 실제로 $$(2, 2)$$의 값은 $$+1$$, $$(0, 0)$$의 값은 $$-1$$이다. $$\lVert\mathbf{w}\rVert \approx 0.7071$$이므로 마진은 $$\sqrt{2} \approx 1.414$$로, 두 점 사이 거리 $$2\sqrt{2}$$의 절반이다.

### 소프트 마진과 힌지 손실

슬랙 변수 $$\xi_i \ge 0$$으로 위반을 허용한다.

$$
\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^{n}\xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \ge 1 - \xi_i
$$

$$\xi_i$$를 소거하면 정규화된 힌지 손실 최소화와 같다.

$$
\min_{\mathbf{w}, b} \frac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^{n}\max\left(0, 1 - y_i(\mathbf{w}^\top\mathbf{x}_i + b)\right)
$$

$$C$$가 크면 위반을 강하게 벌해 마진이 좁아지고, 작으면 위반을 더 허용해 마진이 넓어진다.

### 쌍대 문제와 커널

라그랑주 쌍대 문제는 다음과 같다. $$\alpha_i$$는 라그랑주 승수이다.

$$
\max_{\boldsymbol{\alpha}} \sum_{i}\alpha_i - \frac{1}{2}\sum_{i, j}\alpha_i\alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j) \quad \text{s.t.} \quad 0 \le \alpha_i \le C, \quad \sum_{i}\alpha_i y_i = 0
$$

선형 SVM은 $$K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top\mathbf{x}'$$이다. 입력이 내적으로만 나타나므로 다른 커널로 바꿔 비선형 경계를 얻을 수 있으며, 이를 **커널 트릭(kernel trick)**이라 한다. 예측은 다음과 같다.

$$
\hat{y}(\mathbf{x}) = \text{sign}\left(\sum_{i}\alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b\right)
$$

$$\alpha_i > 0$$인 샘플이 **서포트 벡터**이며, 경계 바깥에 충분히 떨어진 샘플은 $$\alpha_i = 0$$이어서 예측에 쓰이지 않는다.

### 알아 둘 점

- 입력 스케일에 민감하므로 표준화가 사실상 필수이다.
- 출력은 확률이 아니다. 확률이 필요하면 Platt scaling 같은 보정을 쓴다.
- 커널 SVM의 학습 비용은 샘플 수에 대해 빠르게 늘어나 대규모 데이터에서는 부담이 된다.

어떤 상황에서 쓰는지는 [Q-42](/blog/2026/11/02/knowledge-42-svm-when/)에서 다룬다.

## 관련 개념

- **마진**: 최대화의 장점과 한계는 [Q-108](/blog/2026/11/02/knowledge-108-margin/)에서 다룬다.
- **커널 방법**: 커널의 조건과 SVM으로의 적용은 [Machine Learning 7](/blog/2026/10/22/ml-07-kernels/)에서 이어진다.
- **로지스틱 회귀**: 같은 선형 경계를 로지스틱 손실로 학습한다. 힌지 손실과 달리 모든 샘플이 경계에 영향을 준다.
- **정규화**: $$\tfrac{1}{2}\lVert\mathbf{w}\rVert^2$$는 L2 정규화 항과 같은 꼴이다. [Q-90](/blog/2026/11/02/knowledge-90-regularization/)을 참고한다.

## 꼬리 질문

**Q.** 왜 쌍대 문제를 푸나요?
커널을 쓸 수 있고, 입력 차원보다 샘플 수가 적으면 변수가 더 적기 때문입니다. 샘플 수가 매우 많은 선형 SVM은 원 문제를 직접 푸는 편이 효율적입니다.

**Q.** RBF 커널의 $$\gamma$$는 어떤 역할을 하나요?
$$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma\lVert\mathbf{x} - \mathbf{x}'\rVert^2)$$에서 샘플 영향의 범위를 정합니다. $$\gamma$$가 크면 경계가 구불구불해져 과적합되기 쉽고, 작으면 경계가 단순해집니다.

**Q.** 다중 클래스 분류는 어떻게 하나요?
SVM은 기본적으로 이진 분류이므로 일대다(one-vs-rest) 또는 일대일(one-vs-one) 방식으로 여러 분류기를 결합합니다.

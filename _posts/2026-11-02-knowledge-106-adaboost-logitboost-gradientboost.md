---
title: "Q-106. AdaBoost / Logit Boost / Gradient Boost"
date: 2026-11-02 15:02:00 +0900
category: study
topic: basic
series: knowledge
lecture: 106
knowledge_group: machine-learning
card_title: "Q 106: AdaBoost, LogitBoost, Gradient Boost"
card_desc: "Exponential loss · Logistic loss · Newton step · Pseudo-residual"
lang: ko
description: "AdaBoost는 지수 손실, LogitBoost는 로지스틱 손실을 단계적으로 줄이는 방법이고, Gradient Boosting은 임의의 미분 가능한 손실로 이를 일반화한다."
---

<ul class="post-sources">
<li markdown="span">질문 출처: [Science Vibe, 머신러닝 면접 질문 모음](https://jrc-park.tistory.com/259)</li>
</ul>

## 면접 답변

> 세 방법은 모두 약한 모델을 순차적으로 더하는 부스팅이며, 어떤 손실을 어떻게 줄이는지가 다릅니다.
> AdaBoost는 틀린 샘플의 가중치를 높이는 방식으로, 나중에 지수 손실을 단계적으로 최소화하는 것과 같다는 사실이 밝혀졌습니다.
> LogitBoost는 지수 손실 대신 로지스틱 손실을 뉴턴 방식으로 최소화해서, 크게 틀린 샘플에 대한 벌점이 덜 급격하고 노이즈에 덜 민감합니다.
> Gradient Boosting은 손실의 음의 기울기, 즉 의사 잔차를 다음 모델이 근사하게 해서 미분 가능한 손실이면 무엇이든 쓸 수 있게 일반화한 방법입니다.
> 예를 들어 제곱 손실에서는 의사 잔차가 곧 실제 잔차이므로 잔차를 맞추는 나무를 계속 더하는 알고리즘이 되고, 이 유연성 덕분에 실무에서 가장 널리 쓰입니다.

## 자세한 설명

### AdaBoost

레이블이 $$y_i \in \{-1, +1\}$$이고 샘플 가중치 $$w_i$$의 합이 1일 때, 라운드 $$t$$마다 다음을 수행한다(Freund & Schapire, 1997).

1. 가중 오차 $$\varepsilon_t = \sum_i w_i\,\mathbf{1}[h_t(\mathbf{x}_i) \ne y_i]$$가 작은 약한 분류기 $$h_t$$를 학습한다.
2. 모델 가중치를 $$\alpha_t = \tfrac{1}{2}\ln\frac{1-\varepsilon_t}{\varepsilon_t}$$로 계산한다.
3. $$w_i \leftarrow w_i e^{-\alpha_t y_i h_t(\mathbf{x}_i)}$$로 갱신하고 합이 1이 되도록 정규화한다.

최종 분류기는 $$\text{sign}\left(\sum_t \alpha_t h_t(\mathbf{x})\right)$$이다.

**수치 예제.** 샘플 10개의 가중치가 모두 0.1이고 분류기가 3개를 틀렸다고 하자. $$\varepsilon = 0.3$$이므로 $$\alpha = \tfrac{1}{2}\ln(0.7/0.3) \approx 0.4236$$이다. 맞힌 샘플에는 $$e^{-\alpha} \approx 0.6547$$, 틀린 샘플에는 $$e^{\alpha} \approx 1.5275$$를 곱한다. 정규화 상수는 $$0.7e^{-\alpha} + 0.3e^{\alpha} \approx 0.9165$$이다. 정규화 후 틀린 샘플의 가중치는 약 0.1667, 맞힌 샘플의 가중치는 약 0.0714이다. 틀린 샘플 3개의 가중치 합은 정확히 0.5가 되어, 방금 쓴 분류기는 새 가중치에서 무작위 추측과 같아진다. 그래서 다음 분류기는 다른 패턴을 찾게 된다.

정규화 상수 $$2\sqrt{\varepsilon_t(1-\varepsilon_t)}$$는 1보다 작고, 학습 오차는 이 값들의 곱 이하이다. 즉, 약한 분류기가 무작위보다 일정하게 낫다면 학습 오차는 지수적으로 줄어든다.

### LogitBoost

Friedman, Hastie, Tibshirani(2000)는 AdaBoost가 지수 손실에 대한 전진 단계적 가산 모델임을 보이고, 로지스틱 손실로 바꾼 LogitBoost를 제안했다. 정답을 $$y_i^{\ast} \in \{0, 1\}$$, 모델을 $$p(\mathbf{x}) = e^{F(\mathbf{x})}/(e^{F(\mathbf{x})} + e^{-F(\mathbf{x})})$$라 할 때 각 라운드는 다음과 같다.

$$
z_i = \frac{y_i^{\ast} - p(\mathbf{x}_i)}{p(\mathbf{x}_i)(1 - p(\mathbf{x}_i))}, \qquad w_i = p(\mathbf{x}_i)(1 - p(\mathbf{x}_i))
$$

$$w_i$$를 가중치로 $$z_i$$에 대해 회귀 모델 $$f_t$$를 가중 최소제곱 학습한 뒤, $$F \leftarrow F + \tfrac{1}{2}f_t$$로 갱신한다. 이는 로그 우도에 대한 뉴턴 단계이다. 이미 확신을 가지고 맞힌 샘플은 $$w_i$$가 작아 영향이 줄어든다. 예를 들어 $$p = 0.5$$이면 $$w = 0.25$$, $$p = 0.9$$이면 $$w = 0.09$$이다.

### Gradient Boosting

Friedman(2001)은 임의의 미분 가능한 손실 $$L$$에 대해 다음 절차를 제시했다.

1. 상수 모델 $$F_0$$로 시작한다.
2. 샘플마다 의사 잔차(pseudo-residual) $$r_{i} = -\partial L(y_i, F(\mathbf{x}_i))/\partial F(\mathbf{x}_i)$$를 계산한다.
3. 회귀 나무 $$h_t$$를 $$r_i$$에 맞춘다.
4. $$F_t = F_{t-1} + \nu h_t$$로 갱신한다. 나무에서는 잎마다 손실을 최소화하도록 값을 다시 정한다.

**수치 예제.** 손실이 $$L = \tfrac{1}{2}(y - F)^2$$이고 $$y = (1, 2, 6, 7)$$이라 하자. 이때 의사 잔차는 $$y - F$$이다. $$F_0 = 4$$이므로 잔차는 $$(-3, -2, 2, 3)$$이고 MSE는 6.5이다. 처음 두 샘플과 나중 두 샘플로 나누는 스텀프의 출력은 $$(-2.5, -2.5, 2.5, 2.5)$$이다. $$\nu = 0.5$$로 더하면 $$F_1 = (2.75, 2.75, 5.25, 5.25)$$이고 MSE는 1.8125로 줄어든다.

XGBoost는 2차 미분까지 사용하고 나무 복잡도 정규화를 더한 구현이다. 2차 정보를 쓴다는 점에서 LogitBoost의 뉴턴 단계와 닮았다.

### 비교

| 항목 | AdaBoost | LogitBoost | Gradient Boosting |
|---|---|---|---|
| 손실 | 지수 $$e^{-yF}$$ | 로지스틱 $$\log(1 + e^{-2yF})$$ | 미분 가능한 임의의 손실 |
| 다음 모델의 목표 | 재가중된 레이블 | 뉴턴 작업 응답 $$z_i$$ | 의사 잔차 $$r_i$$ |
| 기본 모델 | 분류기 | 회귀 모델 | 회귀 나무 |
| 주된 용도 | 이진 분류 | 이진/다중 분류 | 분류, 회귀, 순위 |

손실의 차이는 $$m = yF$$(마진) 값으로 보면 분명하다.

| $$m$$ | 지수 손실 | 로지스틱 손실 |
|---:|---:|---:|
| $$-2$$ | 7.389 | 4.018 |
| 0 | 1.000 | 0.693 |
| 2 | 0.135 | 0.018 |

크게 틀린 샘플에서 지수 손실은 $$m$$에 지수적으로, 로지스틱 손실은 거의 선형으로 커진다. 레이블 노이즈가 있을 때 AdaBoost가 더 민감한 이유이다.

## 관련 개념

- **부스팅의 개념**: 약한 모델과 가산 모델은 [Q-104](/blog/2026/11/02/knowledge-104-boosting/)에서 다룬다.
- **교차 엔트로피 손실**: 로지스틱 손실은 이진 교차 엔트로피와 같은 형태이다. [Q-83](/blog/2026/11/02/knowledge-83-cross-entropy-loss/)을 참고한다.
- **뉴턴 방법**: LogitBoost의 단계는 2차 근사에서 나온다. [Q-94](/blog/2026/11/02/knowledge-94-newton-method/)를 참고한다.
- **경사하강법**: Gradient Boosting은 이를 함수 공간에서 수행한다. [Q-15](/blog/2026/11/02/knowledge-15-gradient-descent/)를 참고한다.

## 꼬리 질문

**Q.** AdaBoost에서 $$\varepsilon_t$$가 0.5보다 크면 어떻게 되나요?
$$\alpha_t$$가 음수가 되어 해당 분류기의 예측을 뒤집어 사용하는 셈이 됩니다. 일반적인 구현은 무작위보다 나은 약한 분류기를 요구하며, 이 조건이 깨지면 학습을 멈춥니다.

**Q.** 왜 "기울기"를 맞추는 것이 손실을 줄이나요?
학습 샘플에서 $$F$$의 값을 음의 기울기 방향으로 조금 옮기면 손실이 1차 근사로 줄어듭니다. 나무는 이 방향을 새 입력에도 일반화할 수 있게 근사합니다.

**Q.** 제곱 손실이 이상치에 민감하면 어떻게 하나요?
의사 잔차의 크기가 제한되는 절대 손실이나 Huber 손실을 쓸 수 있습니다. Gradient Boosting에서는 손실만 바꾸면 되는 것이 장점입니다.

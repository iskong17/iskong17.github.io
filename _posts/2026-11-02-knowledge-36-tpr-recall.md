---
title: "Q-36. True Positive Rate와 Recall은 어떤 관계인가요?"
date: 2026-11-02 12:36:00 +0900
category: study
topic: basic
series: knowledge
lecture: 36
card_title: "Q 36: TPR and Recall"
card_desc: "True positive rate · Recall · Sensitivity · False negative"
lang: ko
description: "True Positive Rate와 Recall은 같은 지표로, 실제 양성 중 모델이 양성으로 맞힌 비율인 TP/(TP+FN)을 가리키며 민감도라고도 한다."
---

## 면접 답변

> 둘은 이름만 다른 같은 지표입니다.
> 둘 다 실제 양성 중에서 모델이 양성으로 맞힌 비율, 즉 TP를 TP와 FN의 합으로 나눈 값입니다.
> 의학에서는 민감도(sensitivity)라고도 부르며, ROC 곡선을 말할 때는 TPR, 정밀도와 함께 말할 때는 Recall이라고 부르는 것이 관례입니다.
> 예를 들어 실제 환자 100명 중 80명을 찾아냈다면 TPR과 Recall은 모두 0.8입니다.
> FN의 비용이 큰 암 진단이나 사기 탐지에서 특히 중요하게 봅니다.

## 자세한 설명

[혼동 행렬(confusion matrix)](/blog/2026/11/02/knowledge-17-confusion-matrix/)의 항으로 쓰면 다음과 같다. TP는 양성을 양성으로 맞힌 수, FN은 양성을 음성으로 놓친 수이다.

$$
\text{TPR} = \text{Recall} = \text{Sensitivity} = \frac{TP}{TP + FN}
$$

확률로 쓰면 $$P(\hat{y} = 1 \mid y = 1)$$이다. 분모가 **실제 양성** 전체이므로 정밀도(precision)나 위양성률(false positive rate, FPR)과는 기준 집합이 다르다.

| 지표 | 식 | 분모의 의미 | 다른 이름 |
|---|---|---|---|
| TPR | $$TP / (TP + FN)$$ | 실제 양성 | Recall, Sensitivity |
| FNR | $$FN / (TP + FN)$$ | 실제 양성 | $$1 - \text{TPR}$$ |
| FPR | $$FP / (FP + TN)$$ | 실제 음성 | $$1 - \text{Specificity}$$ |
| Precision | $$TP / (TP + FP)$$ | 양성 예측 | PPV |

### 예제

실제 양성 100개, 실제 음성 100개에서 $$TP = 80$$, $$FN = 20$$, $$FP = 30$$, $$TN = 70$$이라고 하자.

- TPR = Recall = $$80/100 = 0.8$$
- FPR = $$30/100 = 0.3$$
- Precision = $$80/110 \approx 0.727$$

### 클래스 비율의 영향

TPR은 실제 양성 안에서만 계산하므로, 클래스별 점수 분포가 그대로라면 양성 비율이 변해도 값이 변하지 않는다. 정밀도는 다르다. 위 모델에서 TPR 0.8과 FPR 0.3을 유지한 채 음성만 900개로 늘리면 $$FP = 270$$이 되어, Recall은 0.8 그대로이지만 Precision은 $$80/350 \approx 0.229$$로 떨어진다.

## 관련 개념

- **ROC 곡선**: 가로축 FPR, 세로축 TPR로 그린 곡선이다. [Q-38](/blog/2026/11/02/knowledge-38-roc-curve/)에서 다룬다.
- **F1-score**: Precision과 Recall의 조화 평균이다. [Q-27](/blog/2026/11/02/knowledge-27-f1-score/)에서 다룬다.
- **특이도(specificity)**: 실제 음성 중 음성으로 맞힌 비율로, TPR의 음성 쪽 대응 지표이다.
- **임계값(threshold)**: 낮추면 양성 예측이 늘어 TPR은 올라가거나 그대로이지만, 보통 FPR도 함께 올라간다.

## 꼬리 질문

**Q.** Recall만 높이는 가장 쉬운 방법은 무엇이고, 왜 문제인가요?
모두 양성으로 예측하면 Recall은 1이 됩니다. 하지만 FP가 최대가 되어 Precision과 FPR이 나빠지므로 Recall은 다른 지표와 함께 봐야 합니다.

**Q.** 다중 클래스에서는 Recall을 어떻게 계산하나요?
클래스별로 일대다(one-vs-rest) Recall을 구한 뒤 평균합니다. 단순 평균이 macro, 표본 수 가중 평균이 weighted이고, 샘플마다 레이블이 하나인 문제에서 micro Recall은 accuracy와 같습니다.

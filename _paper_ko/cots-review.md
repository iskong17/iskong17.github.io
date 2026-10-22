---
title: "CoTS: TPT가 zero-shot CLIP만큼만 확신하도록 온도를 맞추기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
description: CoTS는 테스트 이미지마다 적응 후 확신도를 zero-shot CLIP의 확신도에 맞추는 온도 하나를 학습해 테스트 시간 프롬프트 튜닝을 사후 보정하므로 정확도는 변하지 않으며, E-CoTS는 약한-강한 뷰 앙상블을 더해 정확도를 높인다.
conference: arXiv
year: 2026
paper:
  title: "Bridging the Confidence Gap: Temperature Scaling for Calibrating Test-Time Prompt Tuning"
  authors: "Yuwei Liang, Jian Liang, Dapeng Hu, Yinuo Xu, Ran He"
  venue: "arXiv 2026"
  url: "https://arxiv.org/abs/2609.17386"
en: cots-review
date: 2026-10-22 00:00:00 +0900
---

## 초록 (Abstract)

테스트 시간 프롬프트 튜닝(test-time prompt tuning, TPT)은 단일 테스트 이미지에 대한 CLIP의 정확도를 높이지만 예측을 과신하게 만든다. 대부분의 해결책(C-TPT, O-TPT, A-TPT, SoC)은 프롬프트 튜닝 중 클래스 텍스트 임베딩을 서로 멀어지게 하는 정규화 항을 추가한다. 이들은 ECE를 낮추지만 TPT의 정확도 향상 일부를, 때로는 거의 전부를 잃는다. 유일한 사후(post-hoc) 방법인 SaLS는 정확도를 유지하지만 보정 오차는 조금밖에 줄이지 못한다.

논문은 두 가지 사실에서 출발한다. zero-shot CLIP은 이미 보정이 잘 되어 있고, TPT는 정확도를 조금만 바꾸므로, 늘어난 오차의 대부분은 부풀려진 확신도에서 나온다. 각 TPT 예측에 단순히 zero-shot 확신도를 부여하는 예비 실험만으로도 ECE가 크게 낮아진다. CoTS(Confidence-based Temperature Scaling)는 이를 일관된 방식으로 만든다. TPT 이후, 테스트 이미지 자신의 증강 뷰에서 적응 후 확신도가 zero-shot 확신도와 일치하도록 온도를 학습한다. E-CoTS는 여기에 뷰에 대한 약한-강한(weak-strong) 앙상블을 더해 보정을 유지하면서 정확도를 높인다.

## 방법 (Method)

![CoTS overview](/images/blog/papers/cots-main.png)
*ViT-B/16으로 측정한 11개 세분화 데이터셋(왼쪽)과 4개 ImageNet 변형(오른쪽)에서의 평균 정확도 대 ECE. 정규화 방법들은 왼쪽(낮은 ECE)으로 이동하지만 아래쪽(낮은 정확도)으로도 이동한다. CoTS는 TPT의 정확도를 유지한 채 왼쪽으로 이동하고, E-CoTS는 위쪽으로도 이동한다. 논문에는 별도의 파이프라인 그림이 없으며, 이것은 동기를 설명하는 Figure 1이다. 그림은 논문에서 가져왔다.*

### 설정

CLIP은 이미지 임베딩 $$v$$와 클래스 텍스트 임베딩 $$t_k$$ 사이의 로짓 $$\textbf{z}_{k}=(v^{T} t_{k})/\tau_{\text{clip}}$$을 계산한다. TPT는 테스트 이미지의 64개 뷰를 만든다. 약한 뷰 $$\mathcal{A}_1(x)$$(원본 이미지)와 63개의 AugMix 뷰이다. 확신도가 가장 높은 10%의 뷰를 남기고, 그 평균 예측의 엔트로피를 프롬프트에 대한 한 번의 그래디언트 스텝으로 최소화한다. 튜닝 후와 튜닝 전의 로짓을 각각 $$\textbf{z}^{tpt}$$와 $$\textbf{z}^{zs}$$라 하자.

### 예비 연구: 확신도 교체

각 샘플에 대해 TPT의 예측 클래스는 유지하되, 확신도로는 zero-shot 최대 확률을 보고한다. 이 "CoR"은 ECE를 SaLS와 C-TPT보다 낮추는데, 이는 TPT의 보정 오류가 주로 확신도 부풀림임을 시사한다. 하지만 이 경우 클래스와 확신도가 서로 다른 두 분포에서 나온다. CoTS는 온도만 바꾸는 방식으로 이를 해결한다.

### CoTS: 확신도 차이로부터 구하는 온도

표준 온도 스케일링은 레이블이 있는 검증 세트에서 NLL을 최소화해 $$\tau$$를 맞추는데, 테스트 시점에는 그런 세트가 없다. CoTS는 대신 zero-shot 확신도를 목표로 삼는다. 적응 후 엔트로피가 낮은 강한 뷰 $$\Omega$$(약한 뷰를 제외한 상위 10%)를 고르고 다음을 맞춘다.

$$
\tau^{*}=\arg\min_{\tau}\frac{1}{|\Omega|}\sum_{i\in\Omega}\left(\max_{k}\hat{\textbf{p}}^{tpt}_{k}(\mathcal{A}_{i}(x);\tau)-\max_{k}\textbf{p}_{k}^{zs}(\mathcal{A}_{i}(x))\right)^{2},
$$

여기서 $$\hat{\textbf{p}}^{tpt}(\cdot;\tau)$$는 $$\textbf{z}^{tpt}/\tau$$의 softmax이다. 온도는 50스텝 동안 학습된다. 출력은 맞춘 온도에서의 약한 뷰 예측 $$\hat{\textbf{p}}^{tpt}(\mathcal{A}_1(x);\tau^{*})$$이다. 양의 온도 하나는 argmax를 바꾸지 않으므로 정확도는 TPT와 정확히 같다. 약한 뷰 하나 대신 여러 강한 뷰로 맞추면 확신도에 더 많은 변화가 생긴다. ablation은 약한 뷰만으로 맞추면 CoR보다 나쁘다는 것을 보여 준다.

### E-CoTS: 약한-강한 앙상블

정확도도 얻기 위해, E-CoTS는 약한 뷰와 선택된 강한 뷰들을 각각 맞춘 온도에서 평균한다.

$$
\hat{\textbf{p}}_{ens}=\alpha\cdot\hat{\textbf{p}}^{tpt}(\mathcal{A}_{1}(x);\tau^{*})+(1-\alpha)\cdot\frac{1}{|\Omega|}\sum_{i\in\Omega}\hat{\textbf{p}}^{tpt}(\mathcal{A}_{i}(x);\tau^{*}).
$$

가중치는 $$K$$개 텍스트 임베딩의 평균 쌍별 코사인 유사도 $$\alpha=\frac{1}{K(K-1)}\sum_{i\neq j}t_{i}^{T}t_{j}$$이다. 클래스 텍스트들이 서로 비슷하면 엔트로피 기반 뷰 선택의 신뢰도가 떨어지므로, 원본 이미지에 더 큰 가중치를 준다. 이 $$\alpha$$는 텍스트 쪽에만 의존하므로 튜닝 없이 과제별로 정해진다. 논문은 앙상블만으로는 정확도가 오르지만 보정이 나빠진다고 보고하며, 그래서 앙상블 안에서 온도를 적용한다.

CoTS와 E-CoTS는 모두 적응 이후에 실행되므로, 어떤 정규화 방법 위에도 쌓을 수 있다.

## 주요 실험 (Main Experiments)

백본은 ViT-B/16과 ResNet-50이며, 단일 샘플 에피소드 방식의 TPT로 3개 시드의 평균을 냈다. 평가 지표는 11개 세분화 데이터셋과 ImageNet-A/V2/R/Sketch에 대한 평균 top-1 정확도(%)와 ECE(%)이다. 표는 논문의 Table 6(Table 1, 2와 ResNet-50 부록 표들의 평균)이다. Penalty와 SaLS는 Murugesan et al.(ECCV 2024)의 방법이다.

| 방법 | ViT FG Acc | ViT FG ECE | ViT IN-var Acc | ViT IN-var ECE | RN50 FG Acc | RN50 FG ECE | RN50 IN-var Acc | RN50 IN-var ECE |
|---|---|---|---|---|---|---|---|---|
| Zero-shot | 63.95 | 4.33 | 57.22 | 4.98 | 56.04 | 5.38 | 40.70 | 7.09 |
| TPT | 65.19 | 11.30 | 60.74 | 11.90 | **58.06** | 11.31 | 43.83 | 17.55 |
| C-TPT | 64.57 | 5.23 | 59.25 | 6.67 | 57.69 | 6.44 | 42.75 | 13.14 |
| Penalty | 64.21 | 8.28 | 60.24 | 10.99 | 57.52 | 8.41 | 43.66 | 17.02 |
| SaLS | 65.20 | 9.94 | 60.74 | 10.98 | **58.06** | 9.31 | 43.83 | 15.37 |
| O-TPT | 64.05 | 4.77 | 57.51 | 4.97 | 57.36 | 5.67 | 40.82 | **6.46** |
| A-TPT | 64.12 | 4.33 | 58.20 | **4.34** | 57.30 | **4.55** | 41.94 | 7.76 |
| SoC | 63.95 | 4.31 | 58.12 | 4.87 | 56.19 | 5.50 | 41.61 | 7.52 |
| **CoTS** | 65.19 | **4.28** | 60.74 | 5.30 | **58.06** | 5.04 | 43.83 | 6.87 |
| **E-CoTS** | **65.22** | 4.31 | **62.95** | 5.38 | 57.72 | 5.04 | **45.39** | 7.24 |

**다른 방법에 플러그인 (ViT-B/16, Table 3에서 발췌).** 각 기반 방법과 E-CoTS를 추가한 후의 정확도 / ECE.

| 기반 방법 | FG 기반 | FG + E-CoTS | IN-var 기반 | IN-var + E-CoTS |
|---|---|---|---|---|
| C-TPT | 64.57 / 5.23 | 64.64 / 4.16 | 59.25 / 6.67 | 62.74 / 5.14 |
| Penalty | 64.21 / 8.28 | 65.25 / 3.86 | 60.24 / 10.99 | 62.65 / 5.39 |
| O-TPT | 64.05 / 4.77 | 64.93 / 4.30 | 57.51 / 4.97 | 62.01 / 4.55 |
| A-TPT | 64.12 / 4.33 | 65.07 / 3.85 | 58.20 / 4.34 | 62.32 / 4.87 |
| SoC | 63.95 / 4.31 | 65.01 / 4.20 | 58.12 / 4.87 | 61.84 / 4.58 |

CoTS는 구조상 TPT의 정확도를 유지하며 ECE를 zero-shot 수준 가까이 낮춘다. ViT-B/16 세분화 설정에서는 표에서 가장 낮은 ECE를 기록한다. 하지만 모든 곳에서 ECE가 가장 좋은 것은 아니다. A-TPT는 ImageNet 변형(4.34 vs. 5.30)과 RN50 세분화 설정에서 더 낮고, O-TPT는 RN50 ImageNet 변형에서 더 낮다. E-CoTS는 ImageNet 변형에서 가장 큰 정확도 향상을 보이며(ViT-B/16에서 TPT의 60.74 대비 62.95), 추가되는 모든 정규화 방법의 정확도를 끌어올린다. 다만 ImageNet 변형에서 A-TPT의 ECE는 높인다(4.34에서 4.87).

## 한계 (Limitations)

- **범위 (저자들이 직접 언급).** 분류만 평가했다. 분할(segmentation), 검출(detection), 적대적 강건성(adversarial robustness)은 열린 문제로 남아 있다.
- **추가 연산 (저자들이 직접 언급).** 온도 맞춤은 TPT의 0.19~0.22 s에 더해 샘플당 약 0.053~0.054 s를 추가하며(DTD와 ImageNet-A, ViT-B/16), 모든 뷰에 대한 zero-shot 로짓이 필요하다.
- **보정이 zero-shot에 의해 제한된다.** 목표가 zero-shot 확신도이므로, CoTS는 기껏해야 zero-shot CLIP의 보정을 물려받을 수 있다. 표에서 CoTS의 ECE는 zero-shot ECE 근처에 머물며, ViT-B/16의 ImageNet 변형에서는 그보다 높다(5.30 vs. 4.98). 어떤 도메인에서 zero-shot CLIP의 보정이 나쁘다면, 이 방법에는 그것을 바로잡을 수단이 없다.
- **CoTS만으로는 순위가 개선되지 않는다.** 샘플당 온도 하나는 어떤 클래스를 예측하는지 바꾸지 않는다. 정확도 향상은 모두 E-CoTS의 앙상블에서 나온다.
- **E-CoTS는 보정을 일부 되돌려 줄 수 있다.** ViT-B/16의 ImageNet 변형에서 E-CoTS는 CoTS보다 ECE가 약간 높고(5.38 vs. 5.30), RN50 세분화 설정에서는 정확도가 TPT보다 낮다(57.72 vs. 58.06).
- **프리프린트.** 이 논문은 arXiv 프리프린트이며 동료 심사를 거치지 않았다.

## 내 생각 (My Thoughts)

<!-- TODO -->

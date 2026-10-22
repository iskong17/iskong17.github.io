---
title: "SaLS: 적응된 로짓 벡터를 샘플마다 zero-shot 범위로 다시 압축하기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
description: 적응된 CLIP 모델(어댑터, 프롬프트 학습, TPT)은 분포 이동 하에서 로짓 범위가 커지기 때문에 보정이 나빠진다. 각 샘플의 로짓을 그 샘플의 zero-shot 로짓의 min-max 범위로 재조정하면 예측을 바꾸지 않고도 이 문제의 상당 부분을 해결할 수 있다.
conference: ECCV
year: 2024
en: sals-review
date: 2026-10-22 00:00:00 +0900
paper:
  title: "Robust Calibration of Large Vision-Language Adapters"
  authors: "Balamurali Murugesan, Julio Silva-Rodriguez, Ismail Ben Ayed, Jose Dolz"
  venue: "ECCV 2024"
  url: "https://arxiv.org/abs/2407.13588"
---

## 초록 (Abstract)

Zero-shot CLIP은 비교적 보정(calibration)이 잘 되어 있다. 이를 적응시키는 일반적인 방법들, 즉 few-shot 어댑터(CLIP-Adapter, TIP-Adapter, TaskRes), 프롬프트 학습(CoOp, CoCoOp, MaPLe), 테스트 시간 프롬프트 튜닝(TPT)은 정확도를 높이지만 확신도를 더 나쁘게 만들며, 그 격차는 ImageNet 분포 이동 같은 분포 외(out-of-distribution) 데이터에서 가장 크다. 표준적인 해결책은 이 설정에 맞지 않는다. 온도 스케일링(temperature scaling)은 테스트 분포에서 온 레이블된 검증 세트가 필요하고, 지도 학습에서의 보정 불량에 대한 최근 설명(로짓 노름 증가, LogitNorm이 다루는 문제)은 여기서 일어나는 현상과 맞지 않는다.

논문은 원인이 로짓 노름이 아니라 로짓 범위(최대 로짓 빼기 최소 로짓)의 증가라고 주장한다. 그런 다음 각 샘플의 로짓이 그 샘플 자신의 zero-shot 로짓 범위 안에 머물도록 제약한다. 세 가지 버전이 제시된다. ZS-Norm과 Penalty는 적응 과정에서 작용하고, SaLS(Sample-adaptive Logit Scaling)는 추론 시점의 사후(post-hoc) 재조정으로서 TPT를 포함한 어떤 적응 모델에도 적용된다.

## 방법 (Method)

![SaLS overview](/images/blog/papers/sals-main.png)
*ImageNet-V2/Sketch/A/R에서 zero-shot CLIP(사각형)과 적응된 모델(별: CLIP-Adapter, CoOp, TPT)의 로짓 노름(위)과 로짓 범위(아래)에 따른 ECE. 적응 후 노름은 자주 줄어드는 반면 ECE는 오르지만, 범위는 ECE와 함께 오른다. 논문에는 파이프라인 그림이 없으며, 이 동기 부여 그림이 핵심 논지이다. 그림은 논문에서 가져왔다.*

### 노름인가 범위인가?

로짓 벡터를 $$\boldsymbol{l} = \lVert\boldsymbol{l}\rVert\,\hat{\boldsymbol{l}}$$로 쓴다. 지도 학습 네트워크에 대한 선행 연구는 과신의 원인을 $$\lVert\boldsymbol{l}\rVert$$의 증가로 보았다. 두 가지 단순한 사실이 노름과 범위를 구분한다.

- 음이 아닌 모든 로짓에 상수 $$a>0$$를 더하면 노름은 커지지만 softmax는 변하지 않는다(Proposition 1). 따라서 노름이 크다는 것만으로 확신도가 높다는 뜻은 아니다.
- 범위 $$R(\boldsymbol{l})=\max(\boldsymbol{l})-\min(\boldsymbol{l})$$에 대해, $$a>1$$로 스케일링하면 범위와 최상위 클래스 확률이 모두 커진다(Proposition 2).

위 그림은 이를 실험적으로 확인한다. 세 계열 모두에서 zero-shot(사각형)에서 적응 모델(별)로 갈 때 ECE와 로짓 범위가 함께 증가하는 반면, 로짓 노름은 자주 감소한다. 엔트로피 최소화는 이긴 로짓을 다른 로짓들로부터 멀어지게 하므로, 같은 논리가 TPT에도 적용된다.

### 제약

목표는 통상적인 목적함수 $$\mathcal{H}$$(few-shot 방법에서는 교차 엔트로피, TPT에서는 엔트로피)로 학습하면서, 모든 로짓을 같은 샘플의 zero-shot 최솟값과 최댓값 안에 두는 것이다.

$$
\min\ \mathcal{H}(\boldsymbol{Y},\boldsymbol{P}) \quad \text{s.t.} \quad l^{\text{ZS-min}}_{i}\mathbf{1}\leq\boldsymbol{l}_{i}\leq l^{\text{ZS-max}}_{i}\mathbf{1}\quad\forall i.
$$

라그랑주 방법은 딥러닝 모델에 쓰기 어려우므로, 논문은 세 가지 근사를 제시한다.

### ZS-Norm과 Penalty (적응 중)

**ZS-Norm**은 손실을 계산하기 전에 로짓을 zero-shot 범위로의 min-max 사상으로 재조정한다.

$$
\boldsymbol{l}^{\prime}_{i}=\frac{l^{\text{ZS-max}}_{i}-l^{\text{ZS-min}}_{i}}{l^{\text{max}}_{i}-l^{\text{min}}_{i}}\left(\boldsymbol{l}_{i}-l^{\text{min}}_{i}\mathbf{1}\right)+l^{\text{ZS-min}}_{i}\mathbf{1}.
$$

**Penalty**는 원래 로짓을 유지하고, zero-shot 범위를 벗어나는 모든 로짓에 대해 ReLU 페널티를 더한다($$\lambda=10$$).

$$
\mathcal{H}(\boldsymbol{Y},\boldsymbol{P})+\lambda\sum_{i}\sum_{k=1}^{K}\left(\text{ReLU}(l_{ik}-l^{\text{ZS-max}}_{i})+\text{ReLU}(l^{\text{ZS-min}}_{i}-l_{ik})\right).
$$

### SaLS (추론 시)

SaLS는 같은 min-max 사상 $$\boldsymbol{l}^{\prime}_{i}$$를 테스트 시점에만, 이미 적응된 모델의 로짓에 적용한다. 각 테스트 샘플에 대해 zero-shot 모델을 한 번 실행하여 $$l^{\text{ZS-min}}_{i}$$와 $$l^{\text{ZS-max}}_{i}$$를 얻고, 적응된 로짓을 그 구간으로 재조정한 뒤 softmax를 취한다. 이 사상은 샘플별 양의 아핀 변환이므로 예측 클래스는 바뀌지 않으며, 정확도는 기본 방법과 정확히 같다. 저자들은 이를 비지도 방식의 샘플별 온도 스케일링으로 설명한다. 온도는 zero-shot 범위와 적응된 범위의 비율로 정해지며, 검증 데이터가 필요 없다.

범위를 더 줄인다고 더 좋아지지는 않는다. SaLS에서 zero-shot 범위를 1/2이나 1/4로 줄이면 예측이 균등 분포에 가까워져 ECE가 다시 증가한다.

## 주요 실험 (Main Experiments)

**테스트 시간 프롬프트 튜닝, 11개 fine-grained 데이터셋(평균).** 정확도와 ECE는 % 단위이다. RN50 수치는 본문(Table 3), ViT-B/16 수치는 부록(Table 9)에서 가져왔다. SaLS는 구성상 정확도를 바꾸지 않는다.

| 방법 | RN50 Acc | RN50 ECE | ViT-B/16 Acc | ViT-B/16 ECE |
|---|---|---|---|---|
| Zero-shot | 56.03 | 5.04 | 63.92 | 3.91 |
| TPT | 58.03 | 11.27 | 65.18 | 11.36 |
| TPT w/ ZS-Norm | 57.94 | 10.57 | 65.13 | 10.82 |
| TPT w/ Penalty | 57.69 | 9.58 | 65.22 | 9.27 |
| TPT w/ SaLS | 58.03 | 9.26 | 65.18 | 9.88 |
| C-TPT | 57.54 | 6.33 | 64.59 | 4.81 |
| C-TPT w/ ZS-Norm | 57.63 | 5.74 | 64.41 | 5.03 |
| C-TPT w/ Penalty | 57.52 | 3.14 | 64.68 | 4.61 |
| C-TPT w/ SaLS | 57.54 | 5.22 | 64.59 | 4.48 |

참고: 논문의 RN50 표에서 C-TPT 평균 ECE 값은 나열된 데이터셋별 값의 평균과 일치하지 않는다(C-TPT w/ Penalty의 경우 데이터셋별 값의 평균은 5.93이며, 이는 ImageNet 열에 인쇄된 숫자이다. TPT 행들은 일치한다). 위 수치는 인쇄된 그대로 옮겼다.

**Few-shot 적응(16-shot ImageNet), ViT-B/16, ImageNet-V2/Sketch/A/R 평균.** Table 1(b)와 2(d)에서 가져왔다. Zero-shot은 57.15 / 4.78이다. 각 칸은 Acc / ECE이다.

| 방법 | 기본 | w/ Penalty | w/ SaLS |
|---|---|---|---|
| CLIP-Adapter | 50.61 / 7.82 | 51.59 / 6.38 | 50.61 / 4.38 |
| TIP-Adapter(f) | 25.86 / 63.63 | 49.23 / 40.98 | 25.86 / 44.37 |
| TaskRes | 58.01 / 7.52 | 58.31 / 6.65 | 58.01 / 6.21 |
| CoOp | 58.41 / 6.61 | 59.18 / 4.91 | 58.41 / 4.90 |
| CoCoOp | 59.74 / 4.83 | 60.20 / 3.89 | 59.74 / 4.81 |
| MaPLe | 60.07 / 4.13 | 60.62 / 3.78 | 60.07 / 4.38 |

TPT에서 SaLS는 fine-grained 데이터에서 정확도 변화 없이 ECE를 약 1.5~2포인트 낮추지만, 그 결과(9.26과 9.88)는 여전히 zero-shot보다 훨씬 높고 C-TPT보다도 높다. C-TPT 위에 적용하면 표에서 가장 낮은 ViT-B/16 ECE(4.48)를 얻는다. TPT를 사용한 ImageNet 분포 이동(부록 Table 8)에서는 효과가 더 작다. RN50에서 ECE가 17.58에서 16.74로, ViT-B/16에서 11.81에서 10.86으로 줄어든다. few-shot 적응에서는 어댑터에서 향상이 더 크지만(예: CLIP-Adapter 7.82에서 4.38), CoCoOp이나 MaPLe 같은 강한 프롬프트 학습 방법에서는 변화가 거의 없거나 ECE가 약간 나빠지며, Penalty가 같거나 더 나은 경우가 많다.

## 한계 (Limitations)

- **zero-shot이 좋은 기준이라고 가정.** 목표 범위는 zero-shot CLIP에서 온다. zero-shot 자체의 보정이 나쁜 경우(ImageNet-A에서 RN50 zero-shot의 ECE는 21.32), 그 범위에 맞추는 것으로는 크게 개선할 수 없다.
- **TPT 단독에는 효과가 제한적.** TPT w/ SaLS는 fine-grained 데이터셋에서 여전히 zero-shot의 약 두 배 ECE를 가지며, ImageNet 분포 이동에서는 ECE가 훨씬 높다. 후속 연구(CoTS)는 비교한 TPT 보정 기준선들 중 SaLS를 가장 약한 것으로 보고한다.
- **항상 도움이 되지는 않음.** SaLS는 MaPLe의 ECE를 높이고(4.13에서 4.38) CoCoOp은 거의 그대로 둔다. ZS-Norm은 CLIP-Adapter의 정확도와 ECE를 해친다. 가장 좋은 변형은 방법에 따라 다르다.
- **범위만 통제됨.** 재조정은 최대 로짓과 최소 로짓 사이의 간격을 정하지만, 나머지 로짓들이 그 구간 안에서 어떻게 퍼져 있는지는 정하지 않는다.
- **추가 zero-shot 순전파.** 각 테스트 샘플마다 적응된 로짓 외에 zero-shot 로짓이 필요하다. TPT에서는 zero-shot 로짓이 이미 계산되어 있고 CoTS는 SaLS로 인한 추가 시간을 측정하지 못했지만, 다른 적응 모델에서는 두 번째 순전파가 된다.
- **보고 불일치.** Table 3의 일부 RN50 C-TPT 평균이 데이터셋별 값과 맞지 않아, RN50에서의 C-TPT 비교를 해석하기 어렵다.

## 내 생각 (My Thoughts)

<!-- TODO -->

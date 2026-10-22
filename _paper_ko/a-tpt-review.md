---
title: "A-TPT: 클래스 간 최소 각도를 최대화해 테스트 시간 프롬프트를 보정하기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
description: A-TPT는 각 클래스가 가장 가까운 이웃 클래스 텍스트 특징과 이루는 각도를 최대화하는 방식으로 테스트 시간 프롬프트 튜닝을 정규화한다. 이를 통해 특징이 초구 위에 고르게 퍼지고, C-TPT와 O-TPT보다 보정 오차가 더 낮아진다.
conference: ICLR
year: 2026
paper:
  title: "A-TPT: Angular Diversity Calibration Properties for Test-Time Prompt Tuning of Vision-Language Models"
  authors: "Shihab Aaqil Ahamed, Udaya S. K. P. Miriya Thanthrige, Ranga Rodrigo, Muhammad Haris Khan"
  venue: "ICLR 2026"
  url: "https://arxiv.org/abs/2510.26441"
en: a-tpt-review
date: 2026-10-22 00:00:00 +0900
---

## 초록 (Abstract)

테스트 시간 프롬프트 튜닝(test-time prompt tuning, TPT)은 레이블이 없는 각 테스트 이미지에 대해 엔트로피 최소화로 CLIP의 프롬프트를 적응시킨다. 정확도는 오르지만 예측이 과신(overconfident)하게 된다. 기존의 두 해결책은 클래스 텍스트 특징에 정규화 항을 추가한다. C-TPT는 특징들을 중심점(centroid)으로부터 L2 거리 기준으로 멀어지게 한다(ATFD). O-TPT는 쌍별 코사인 유사도 행렬을 단위 행렬 쪽으로 밀어 특징들이 직교하도록 만든다. 저자들은 둘 중 어느 것도 초구(hypersphere) 위에서 특징이 고르게 퍼지는 것을 보장하지 않는다고 주장한다. ATFD는 일부 특징이 뭉쳐 있는 채로 중심점만 옮길 수 있다. 직교성은 클래스 수 $$N$$이 임베딩 차원 $$\lvert D \rvert$$보다 크면 달성할 수 없고(예: CLIP ViT-B/16의 512차원 공간에 ImageNet 1000개 클래스), $$N < \lvert D \rvert$$인 경우에도 각도를 균일하게 만들지는 않는다.

A-TPT는 Tammes 문제(구 위에 점들을 배치해 임의의 두 점 사이의 최소 거리를 최대한 크게 만드는 문제)의 아이디어를 빌려 온다. 각 클래스에 대해 가장 가까운 다른 클래스 특징과의 각도를 최대화하는 손실을 추가한다. 이 손실은 $$N > \lvert D \rvert$$와 $$N < \lvert D \rvert$$ 두 경우 모두에서 특징을 균일하게 퍼뜨리며, 비슷한 비용으로 C-TPT와 O-TPT보다 ECE를 더 낮춘다.

## 방법 (Method)

![A-TPT overview](/images/blog/papers/a-tpt-main.png)
*단위 초구 위에서 ATFD(C-TPT, 위), 직교성(O-TPT, 가운데), 각도 다양성(A-TPT, 아래)이 클래스 텍스트 특징을 움직이는 방식. O-TPT는 $$N > \lvert D \rvert$$와 $$N < \lvert D \rvert$$에서 서로 다른 방식으로 실패한다. A-TPT는 두 경우 모두 특징을 고르게 퍼뜨린다. 그림은 논문에서 가져왔다.*

### 설정

CLIP은 각 클래스 프롬프트 $$\mathtt{p}_k$$를 텍스트 특징 $$t_k = f_t(\mathtt{p}_k)$$로, 테스트 이미지를 $$v = f_i(x)$$로 변환한다. 클래스 확률은 온도 0.01에서 코사인 유사도 $$\text{sim}(v, t_k)$$에 대한 softmax이다. TPT에서는 프롬프트 토큰 $$\mathtt{p}$$가 학습 가능하며, 테스트 이미지마다 엔트로피 손실 $$\mathcal{L}_{\text{TPT}}$$에 대해 한 번의 그래디언트 스텝을 받는다. 문제는 텍스트 특징에 어떤 추가 항을 걸어야 보정(calibration)이 좋아지는가이다.

### 분산과 직교성의 문제점

그림의 윗줄은 C-TPT이다. ATFD는 특징들이 중심점에서 떨어진 L2 거리를 최대화한다. 중심점은 원점 쪽으로 이동하지만, 실패 사례에서는 여전히 두 특징이 서로 붙어 있다.

가운데 줄은 O-TPT이다. 이 손실은 모든 쌍별 코사인을 0으로 민다. $$N > \lvert D \rvert$$이면 서로 직교하는 벡터 $$N$$개가 존재하지 않으며, 저자들은 이때 특징들이 오히려 더 가깝게 모이는 것을 관찰한다. $$N < \lvert D \rvert$$이면 직교성은 달성 가능하지만, 직교하는 특징들이 구를 가능한 한 고르게 쓰지는 않는다(세 클래스가 서로 최대한 멀리 퍼지는 대신 서로 90도를 이룬다).

저자들은 그래디언트 관점에서도 논증한다. O-TPT에서 쌍별 코사인의 특징에 대한 그래디언트 노름은

$$
\left\lVert \frac{\partial \, Cos_{ij}}{\partial \mathbf{e}_i} \right\rVert = \frac{\lvert \sin\theta_{ij} \rvert}{\lVert \mathbf{e}_i \rVert},
$$

이며, 이는 $$\theta_{ij} \to 0$$일 때 0으로 간다. 즉 보정에 가장 해로운, 거의 같은 방향을 향하는 쌍이 정작 거의 밀려나지 않는다. 각도 자체를 최적화하면

$$
\left\lVert \frac{\partial \theta_{ij}}{\partial \mathbf{e}_i} \right\rVert = \frac{1}{\sin\theta_{ij}} \cdot \frac{\sin\theta_{ij}}{\lVert \mathbf{e}_i \rVert} = \frac{1}{\lVert \mathbf{e}_i \rVert},
$$

가 되어 각도에 의존하지 않는다.

### 각도 다양성 손실

클래스 텍스트 특징을 $$\mathbf{E} \in \mathbb{R}^{N\times \lvert D \rvert}$$로 쌓고 각 행을 L2 정규화해 $$\hat{\mathbf{E}}$$를 얻는다. 쌍별 코사인과 각도는 다음과 같다.

$$
Cos = \hat{\mathbf{E}}\hat{\mathbf{E}}^T, \qquad \theta = \arccos(Cos).
$$

각도 다양성(angular diversity, AD)은 각 클래스에서 가장 가까운 다른 클래스까지의 각도를 클래스에 대해 평균한 값이다.

$$
\text{AD} = \frac{1}{N} \sum_{i=1}^{N} \min_{j \neq i} \theta_{ij}.
$$

프롬프트는 다음과 같이 튜닝된다.

$$
\mathtt{p}^{*} = \arg\min_{\mathtt{p}} \left( \mathcal{L}_{\text{TPT}} + \lambda \cdot \mathcal{L}_{\text{A-TPT}} \right), \qquad \mathcal{L}_{\text{A-TPT}} = -\text{AD}.
$$

각 클래스의 최근접 이웃만 고려하므로, 이 손실은 가장 덜 분리된 쌍을 겨냥한다. 이것이 Tammes 문제의 max-min 아이디어이다. 그림의 아랫줄이 이를 보여 준다. $$N > \lvert D \rvert$$에서는 특징들이 같은 각도로 구 둘레에 퍼지고, $$N < \lvert D \rvert$$에서는 세 특징이 90도가 아니라 서로 같은, 최대의 각도를 이룬다.

### 근거와 세부 사항

RN50을 사용한 Caltech101에서, 정확도가 비슷한 하드 프롬프트(hard prompt)들도 ECE가 다르며, 보정이 더 잘 된 프롬프트일수록 AD가 크다(예: "this is a photo of [class]": ECE 6.15, AD 0.622; "the nearest shape in this image is [class]": ECE 10.84, AD 0.461). TPT에 A-TPT를 더하면, 보고된 세 프롬프트의 AD는 0.633~0.659, ECE는 2.12~3.74가 된다. 일반 TPT는 AD 0.622, ECE 6.18이다. 구현은 C-TPT와 TPT를 따른다. 하드 프롬프트로 초기화하고, 학습률 5e-3의 AdamW 한 스텝, 배치 크기 64를 사용한다. 가중치는 세분화(fine-grained) 데이터셋에서 $$\lambda = 80$$, ImageNet 변형에서 $$\lambda = 10$$이다. 점근적 비용은 O-TPT와 같은 $$O(N^2 \lvert D \rvert)$$이다.

## 주요 실험 (Main Experiments)

백본은 CLIP ViT-B/16(512차원)과 CLIP RN50(1024차원)이다. 평가 지표는 top-1 정확도(%)와 ECE(%)이다. 표에는 Average 열만 실었다. 세분화 평균은 11개 데이터셋(ImageNet, DTD, Flowers102, Food101, SUN397, Aircraft, OxfordPets, Caltech101, UCF101, EuroSAT, StanfordCars)을 포함한다. "Zero-shot"은 논문의 "Baseline" 행이다. C-TPT와 O-TPT 수치는 O-TPT 논문에 보고된 것과 같다.

**세분화 분류, 11개 데이터셋 평균** (Table 2)

| 방법 | ViT-B/16 Acc. | ViT-B/16 ECE | RN50 Acc. | RN50 ECE |
|---|---|---|---|---|
| Zero-shot CLIP | 63.70 | 4.43 | 55.90 | 5.61 |
| TPT | 65.00 | 11.60 | 57.70 | 11.70 |
| C-TPT | 64.57 | 5.13 | 57.75 | 6.19 |
| O-TPT | 64.41 | 4.23 | 57.47 | 5.45 |
| **A-TPT** | 64.92 | **2.61** | 57.53 | **2.92** |

**자연 분포 변화, ImageNet-A, -V2, -R, -Sketch 평균** (Table 3)

| 방법 | ViT-B/16 Acc. | ViT-B/16 ECE | RN50 Acc. | RN50 ECE |
|---|---|---|---|---|
| Zero-shot CLIP | 57.20 | 5.04 | 40.60 | 7.46 |
| TPT | 59.90 | 12.00 | 43.50 | 16.70 |
| C-TPT | 59.60 | 5.82 | 42.80 | 12.10 |
| O-TPT | 57.80 | 4.88 | 41.16 | 9.69 |
| **A-TPT** | 58.06 | **3.92** | 40.57 | **7.82** |

A-TPT는 네 가지 설정 모두에서 테스트 시간 방법 중 평균 ECE가 가장 낮으며, ViT-B/16의 세분화 데이터에서는 TPT의 정확도를 거의 그대로 유지한다(64.92 vs. 65.00). 두 백본 모두에서 11개 세분화 데이터셋 각각의 ECE가 O-TPT보다 낮지만, 평균 향상의 상당 부분은 두 데이터셋에서 나온다. EuroSAT에서는 ECE가 ViT-B/16 기준 12.98(O-TPT)에서 3.92로, RN50 기준 14.71에서 2.51로 떨어지고, DTD에서는 7.88에서 4.76으로, 16.53에서 6.41로 떨어진다. 논문 본문은 세분화 평균 ECE를 2.92(ViT-B/16)와 2.79(RN50)로 인용하지만 Table 2는 2.61과 2.92를 제시하며, 여기서는 표의 값을 사용했다. ImageNet-V2에서 C-TPT 대비 추가 비용은 작다. 배치당 1.058 s, 21840 MiB로, C-TPT의 1.055 s, 21838 MiB와 O-TPT의 1.064 s, 23740 MiB와 비교된다.

## 한계 (Limitations)

- **소폭의 정확도 하락 (저자들이 직접 언급).** 일부 데이터셋에서는 정확도가 떨어지며, 저자들은 그 폭을 1~2% 미만으로 본다. RN50의 자연 분포 변화에서는 A-TPT의 평균 정확도(40.57)가 zero-shot 수준(40.60)까지 내려가므로, 그곳에서는 TPT의 정확도 향상이 사라진다.
- **레이블 없이 고정한 $$\lambda$$ (저자들이 직접 언급).** 트레이드오프는 하나의 가중치로 정해지며, 저자들은 이를 ablation으로 고정했고 샘플별로 최적이 아닐 수 있다고 말한다. 실제로는 벤치마크마다 값이 바뀌므로(세분화 데이터 80, ImageNet 변형 10), 대상 설정에 대한 어느 정도의 사전 지식이 필요하다.
- **항상 zero-shot보다 나은 것은 아니다.** 논문은 A-TPT가 두 백본 모두에서 자연 분포 변화에 대해 zero-shot보다 보정이 낫다고 말하지만, RN50에서 평균 ECE(7.82)는 zero-shot CLIP(7.46)보다 높다.
- **평균 향상이 일부에 집중되어 있다.** O-TPT와의 평균 ECE 차이 중 상당 부분은 EuroSAT과 DTD에서 나온다. 나머지 9개 데이터셋에서의 향상은 일관되지만 더 작아서, 약 0.1~2 ECE 포인트이다.
- **손실 안의 하드 최솟값.** 각 클래스는 단 하나의 최근접 이웃을 통해서만 그래디언트를 받으며, $$\arccos$$는 코사인 값이 $$\pm 1$$ 근처일 때 기울기가 한없이 커진다. 논문은 이 둘로 인한 안정성 문제를 논의하지 않는다. 다만 부록에 여러 시드에 걸친 결과가 보고되어 있다.
- **본문 수치의 불일치.** 결과 문단에 인용된 평균값이 Table 2와 맞지 않으므로, 독자는 표를 확인해야 한다.

## 내 생각 (My Thoughts)

<!-- TODO -->

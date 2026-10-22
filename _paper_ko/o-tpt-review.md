---
title: "O-TPT: 클래스 텍스트 특징을 직교하게 만들어 테스트 시간 프롬프트를 보정하기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
description: O-TPT는 테스트 시간 프롬프트 튜닝에 CLIP 클래스 텍스트 특징의 Gram 행렬을 단위 행렬 쪽으로 미는 정규화 항 하나를 더한다. 이를 통해 클래스들이 각도상으로 멀어지고 보정 오차가 줄어든다.
conference: CVPR
year: 2025
en: o-tpt-review
date: 2026-10-22 00:00:00 +0900
paper:
  title: "O-TPT: Orthogonality Constraints for Calibrating Test-time Prompt Tuning in Vision-Language Models"
  authors: "Ashshak Sharifdeen, Muhammad Akhtar Munir, Sanoojan Baliah, Salman Khan, Muhammad Haris Khan"
  venue: "CVPR 2025"
  url: "https://arxiv.org/abs/2503.12096"
---

## 초록 (Abstract)

테스트 시간 프롬프트 튜닝(test-time prompt tuning, TPT)은 증강된 뷰들에 대한 예측 엔트로피를 최소화하여 CLIP의 텍스트 프롬프트를 각 테스트 이미지에 맞게 조정한다. 이는 zero-shot 정확도를 높이지만, 동시에 모델을 과신(overconfident)하게 만들어 기대 보정 오차(expected calibration error, ECE)를 크게 증가시킨다. C-TPT는 클래스 텍스트 특징을 L2 거리 기준으로 중심(centroid)에서 멀어지게 하는 ATFD(Average Text Feature Dispersion) 손실로 이 문제를 다루었다. 저자들은 이것만으로는 충분하지 않다고 주장한다. 단위 초구(hypersphere) 위에서 중심을 원점 쪽으로 옮긴다고 해서 개별 클래스 특징들이 서로 멀리 떨어진다는 보장은 없다. 어려운 경우에는 일부 특징이 여전히 가깝게 모여 있으며, 이러한 경우에 ECE가 높다.

O-TPT는 분산(dispersion)을 각도 기반 목적함수로 대체한다. 클래스 텍스트 특징 간 쌍별 코사인 유사도 행렬을 단위 행렬 쪽으로 미는 페널티를 추가하여, 특징들이 직교에 가까워지도록 한다. 저자들은 텍스트 특징 간 코사인 유사도가 낮을수록 ECE도 낮다는 것을 보이고, 이 단순한 제약이 C-TPT와 Robust-Adapt 변형들보다 ECE를 더 많이 낮춘다는 것을 보인다.

## 방법 (Method)

![O-TPT overview](/images/blog/papers/o-tpt-main.png)
*텍스트 특징 초구 위에서의 각도 최적화(O-TPT, 위)와 ATFD 최적화(C-TPT, 아래) 비교. ATFD는 중심을 원점 쪽으로 옮기는데, 이는 성공할 수도 있지만 특징들이 여전히 뭉쳐 있는 채로 남을 수도 있다. O-TPT는 특징들을 각도상으로 퍼뜨린다. 그림은 논문에서 가져왔다.*

### 설정

CLIP은 텍스트 인코더 $$\mathbf{f}_T$$와 이미지 인코더 $$\mathbf{f}_I$$를 가진다. 각 클래스 $$c_i$$에 대해 프롬프트 $$\mathbf{t}_{c_i}$$(예: "a photo of a {class}")는 텍스트 특징 $$\mathbf{e}_{c_i} = \mathbf{f}_T(\mathbf{t}_{c_i})$$를 만든다. 테스트 이미지는 $$\mathbf{v} = \mathbf{f}_I(I)$$를 만들고, 클래스 확률은 온도 $$\tau$$를 사용한 코사인 유사도 $$s_i = \cos(\mathbf{v}, \mathbf{e}_{c_i})$$에 대한 softmax이다. TPT에서는 프롬프트 토큰이 학습 가능하며, 확신도가 높은 증강 뷰들에 대한 엔트로피 손실 $$L_{TPT}$$를 사용해 테스트 이미지마다 한 번의 경사 단계로 갱신된다. 보정은 ECE로 측정한다.

$$
\text{ECE} = \sum_{m=1}^{M} \frac{\lvert A_m \rvert}{N} \left\lvert \text{acc}(A_m) - \text{conf}(A_m) \right\rvert.
$$

### 왜 L2 분산이 아니라 각도인가

그림이 이 논지를 보여준다. 정규화된 CLIP 텍스트 특징은 단위 초구 위에 놓인다. ATFD(아래 행)는 각 특징과 중심 사이의 거리를 최대화하며, 실제로는 중심을 원점 쪽으로 옮기는 효과를 낸다. 성공 사례에서는 이것이 특징들을 퍼뜨린다. 실패 사례에서는 중심이 움직이지만 일부 특징 쌍은 여전히 가깝게 남아 있어 공간이 충분히 활용되지 않는다.

저자들은 두 가지 분석으로 이를 뒷받침한다. 첫째, 여러 수작업 프롬프트 스타일에 걸쳐, ECE가 낮은 프롬프트일수록 코사인 유사도 분포가 더 낮은 값 쪽으로 이동해 있다. 둘째, TPT 텍스트 특징의 평균 코사인 유사도가 중앙값보다 높은지(Group 1) 낮은지(Group 2)에 따라 테스트 샘플을 나눈다. 더 어려운 Group 1에서 ECE는 TPT가 15.97, C-TPT가 6.04, O-TPT가 4.75이다. 즉 텍스트 특징이 붕괴하는 샘플에서 보정이 실패하며, C-TPT는 이를 완전히 해결하지 못한다.

### 직교성 제약

모든 $$C$$개 클래스의 텍스트 특징을 쌓아 $$\mathbf{E} \in \mathbb{R}^{C\times D}$$를 만든다. 그러면 $$\mathbf{E}\mathbf{E}^T$$는 쌍별 코사인 유사도를 담는다. 이상적인 경우는 모든 클래스 특징이 직교하는 $$\mathbf{E}\mathbf{E}^T = I_C$$이다. O-TPT는 이를 TPT 손실에 페널티로 더한다.

$$
\mathbf{t}^{*} = \arg\min_{\mathbf{t}} \left( L_{TPT} + \lambda \left\lVert \mathbf{E}\mathbf{E}^{T} - I_C \right\rVert_2^2 \right).
$$

최적화되는 것은 프롬프트 $$\mathbf{t}$$뿐이다. 정규화된 특징에서는 $$\mathbf{E}\mathbf{E}^T$$의 대각 성분이 이미 1이므로, 사실상 이 항은 모든 비대각 코사인 유사도를 0 쪽으로 민다. 그림의 위쪽 행이 보여주는 것이 바로 이것으로, 중심만 옮기는 대신 특징들이 각도상으로 퍼진다.

### 세부 사항

프롬프트는 C-TPT와 마찬가지로 하드 프롬프트(hard prompt)로 초기화되며, 단 한 번의 AdamW 단계(학습률 0.005, 증강 뷰 64개의 배치)로 갱신된다. 한 단계 최적화를 돕기 위해 $$\mathbf{E}\mathbf{E}^T$$ 위에 Householder 변환이 적용된다. 논문은 이 단계를 자세히 설명하지 않지만, 이를 제거하면 fine-grained 벤치마크의 평균 ECE가 4.23에서 4.45로 오른다고 보고한다(ViT-B/16). 가중치는 fine-grained 데이터셋에 $$\lambda = 18$$, ImageNet 변형들에 $$\lambda = 2$$이다.

## 주요 실험 (Main Experiments)

백본은 CLIP ViT-B/16과 CLIP RN50이다. 지표는 top-1 정확도(%)와 ECE(%)이다. fine-grained 표는 논문 Table 4와 5의 Average 열만 보여주며, 이는 11개 데이터셋(ImageNet, DTD, Flowers, Food101, SUN397, Aircraft, Pets, Caltech101, UCF101, EuroSAT, Cars)의 평균이다. Robust-Adapt 행은 Murugesan 등의 로짓 정규화(logit normalization) 방법을 C-TPT와 함께 적용한 것이다.

**Fine-grained 분류, 11개 데이터셋 평균**

| 방법 | ViT-B/16 Acc. | ViT-B/16 ECE | RN50 Acc. | RN50 ECE |
|---|---|---|---|---|
| Zero-shot CLIP | 63.7 | 4.43 | 55.9 | 5.61 |
| TPT | 65.0 | 11.6 | 57.7 | 11.7 |
| C-TPT | 64.57 | 5.13 | 57.75 | 6.19 |
| Robust-adapt-SaLs-CTPT | 64.55 | 5.69 | 57.20 | 6.82 |
| Robust-adapt-Penalty-CTPT | 64.65 | 4.66 | 57.40 | 6.41 |
| Robust-adapt-ZS-CTPT | 64.51 | 5.09 | 57.41 | 6.34 |
| **O-TPT** | 64.41 | **4.23** | 57.47 | **5.45** |

**자연 분포 이동, ImageNet-A, -V2, -R, -Sketch 평균** (Table 6의 Average 열)

| 방법 | ViT-B/16 Acc. | ViT-B/16 ECE | RN50 Acc. | RN50 ECE |
|---|---|---|---|---|
| Zero-shot CLIP | 57.2 | 5.04 | 40.6 | 7.46 |
| TPT | 59.9 | 12.0 | 43.5 | 16.7 |
| C-TPT | 59.6 | 5.82 | 42.8 | 12.1 |
| **O-TPT** | 57.80 | **4.88** | 41.16 | **9.69** |

O-TPT는 두 벤치마크와 두 백본 모두에서 테스트 시간 방법들 중 가장 낮은 평균 ECE를 기록한다. fine-grained 벤치마크에서는 ECE가 zero-shot CLIP보다도 낮은데, 표의 다른 어떤 테스트 시간 방법도 이를 달성하지 못한다. 논문 본문은 ViT-B/16 평균 ECE를 4.21로 인용하지만 표에는 4.23으로 되어 있으며, 여기서는 표의 값을 사용했다. 보정 향상에는 정확도 손실이 따른다. ViT-B/16 기준 fine-grained 데이터셋에서 TPT 대비 약 0.6포인트, ImageNet 변형에서 2.1포인트이다. O-TPT는 CoOp 프롬프트 위에 더했을 때(평균 ECE 7.91, C-TPT는 10.04, TPT는 18.36)와 MaPLe 위에 더했을 때에도 ECE를 낮춘다.

## 한계 (Limitations)

- **명시된 한계 없음.** 논문에는 한계 섹션이 없다.
- **분포 이동에서의 정확도 하락.** ViT-B/16을 사용한 ImageNet 변형에서 정확도는 59.9(TPT)에서 57.80으로 떨어지며, ImageNet-R은 72.55로 zero-shot CLIP(74.0)보다 낮아진다. 여기서는 TPT의 정확도 향상 대부분이 사라진다.
- **항상 zero-shot보다 나은 것은 아님.** 자연 분포 이동에서 RN50을 사용할 때 O-TPT의 평균 ECE(9.69)는 여전히 zero-shot CLIP(7.46)보다 높다.
- **클래스 수가 차원 수보다 많으면 정확한 직교는 불가능.** $$\mathbf{E}\mathbf{E}^T$$의 랭크는 최대 $$D$$이므로, ImageNet의 1000개 클래스와 512차원 ViT-B/16 임베딩에서는 목표 $$I_C$$에 도달할 수 없다. 페널티는 여전히 평균 유사도를 낮추지만, 그 최적점이 무엇인지는 더 이상 명확하지 않다. 후속 연구인 A-TPT가 이 지점에서 출발한다.
- **하이퍼파라미터 선택.** $$\lambda$$는 fine-grained 데이터에서 18, ImageNet 변형에서 2이다. 레이블 없이 새로운 대상 도메인에서 이를 어떻게 정할지는 논의되지 않는다. $$\lambda$$에 대한 파레토 플롯은 Food101과 Flowers에 대해서만 제시된다.
- **비용.** 페널티는 테스트 이미지마다 전체 $$C\times C$$ 유사도 행렬을 필요로 하므로, TPT에 더해 $$O(C^2 D)$$의 비용이 든다. 본문은 실행 시간이나 메모리를 보고하지 않는다. Householder 단계는 짧게만 설명된다.
- **일부 데이터셋은 여전히 보정이 좋지 않음.** EuroSAT의 ECE는 ViT-B/16에서 12.98, RN50에서 14.71로 C-TPT와 거의 차이가 없다.

## 내 생각 (My Thoughts)

<!-- TODO -->

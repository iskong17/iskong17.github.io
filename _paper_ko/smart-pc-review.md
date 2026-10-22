---
title: "SMART-PC: 스켈레톤을 예측하고, BatchNorm 통계만으로 적응하기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
description: SMART-PC는 MATE의 점 재구성을 스켈레톤 예측으로 대체한다. 이렇게 얻은 특징은 충분히 강건하여, 온라인 적응에서 역전파 없이 BatchNorm 통계 갱신만으로 충분하다.
conference: ICML
year: 2025
en: smart-pc-review
date: 2026-10-22 00:00:00 +0900
paper:
  title: "SMART-PC: Skeletal Model Adaptation for Robust Test-Time Training in Point Clouds"
  authors: "Ali Bahri, Moslem Yazdanpanah, Sahar Dastani, Mehrdad Noori, Gustavo Adolfo Vargas Hakim, David Osowiechi, Farzad Beizaee, Ismail Ben Ayed, Christian Desrosiers"
  venue: "ICML 2025"
  url: "https://arxiv.org/abs/2505.19546"
---

## 초록 (Abstract)

MATE는 네트워크가 마스킹된 각 테스트 샘플을 분류하기 전에 재구성하도록 하면 3D 포인트 클라우드에 대한 테스트 시간 학습(test-time training, TTT)이 효과가 있음을 보였다. 저자들은 두 가지 문제를 지적한다. 첫째, MATE는 표면 점을 재구성하므로 그 특징이 노이즈 같은 표면 수준 손상에 여전히 민감하다. 둘째, 테스트 시점에 역전파로 네트워크의 모든 파라미터를 갱신하므로 프레임 속도가 낮다.

SMART-PC는 재구성 과제를 스켈레톤 예측으로 바꾼다. 각 패치는 지역 형태를 요약하는 구(스켈레톤 점과 반지름)를 예측한다. 이러한 추상화는 고주파 노이즈를 평균화해 없애므로, 인코더는 손상 하에서 더 잘 버티는 특징을 학습한다. 이런 특징이 있으면 온라인 적응에서 역전파를 생략하고 BatchNorm의 이동 평균(running mean)과 분산만 갱신할 수 있다.

## 방법 (Method)

![SMART-PC overview](/images/blog/papers/smart-pc-main.png)
*SMART-PC: 공유된 Point-MAE 인코더-디코더가 스켈레톤 구의 중심과 반지름을 예측하는 두 MLP, 그리고 인코더와 디코더 특징의 합을 입력으로 받는 분류기에 특징을 공급한다. 테스트 시점에 모델은 스켈레톤 손실과 역전파로 적응하거나, BatchNorm 통계만 갱신하여 적응한다. 그림은 논문에서 가져왔다.*

### 패치화

포인트 클라우드 $$P\in\mathbb{R}^{N\times 3}$$를 패치로 나눈다. 최원점 샘플링(farthest point sampling)으로 $$M$$개의 중심을 고르고, 각 중심은 $$K$$개의 최근접 이웃을 취한다.

$$
C = \mathit{FPS}(P)\in\mathbb{R}^{M\times 3}, \qquad P_{local} = \mathit{kNN}(C,P)\in\mathbb{R}^{M\times K\times 3}
$$

### 스켈레톤 분기

스켈레톤 구는 $$s=\big(c_s, r(c_s)\big)\in\mathbb{R}^4$$로, 중심 $$c_s$$와 형태 표면까지 닿는 반지름 $$r(c_s)$$로 이루어진다. 구 위의 임의의 점은 단위 벡터 $$v$$에 대해 $$p = c_s + r(c_s)\cdot v$$이므로, 구들의 집합으로 전체 형태를 기술할 수 있다.

인코더와 디코더가 패치를 처리하고, 두 MLP가 패치마다 구 하나를 예측한다.

$$
F_{enc}=E(P_{local}), \quad F_{dec}=D(F_{enc}), \quad c_s=\mathit{MLP}_s(F_{dec})\in\mathbb{R}^{M\times 3}, \quad r=\mathit{MLP}_r(F_{dec})\in\mathbb{R}^{M\times 1}
$$

Point2Skeleton 같은 이전의 스켈레톤 방법들은 스켈레톤 점을 입력 점들의 볼록 결합(convex combination)으로 만든다. SMART-PC는 특징으로부터 이를 직접 예측한다. 저자들은 원시 점을 예측에 입력하면 과제가 너무 쉬워져 인코더가 강건한 특징을 학습하지 못한다고 주장한다.

### 분류 분기

분류기는 인코더와 디코더 특징의 합을 사용하므로, 스켈레톤과 관련된 디코더 특징도 함께 본다.

$$
F_{combined} = F_{enc} + F_{dec}, \qquad p = \mathit{Softmax}\big(\mathit{MLP}_{cls}(F_{combined})\big)
$$

### 학습 손실

스켈레톤 분기는 세 가지 손실로 자기 지도(self-supervised) 학습된다. 점-구(point-to-sphere) 손실은 입력 점들을 구 표면으로, 구들을 입력 점 쪽으로 끌어당긴다.

$$
\mathcal{L}_{p2s} = \sum_{p\in P}\Big(\min_{s\in S}\|p-c_s\|_2 - r(c_s)\Big) + \sum_{s\in S}\Big(\min_{p\in P}\|c_s-p\|_2 - r(c_s)\Big)
$$

샘플링 손실은 입력 $$P$$와 구 표면에서 샘플링한 점 $$T$$ 사이의 Chamfer 거리이다. 반지름 손실 $$\mathcal{L}_{radius} = -\sum_{s\in S} r(c_s)$$는 반지름을 키우도록 밀어서, 각 구가 더 넓은 영역을 덮고 개별 노이즈 점의 영향이 줄어들게 한다. 전체 손실은 다음과 같다.

$$
\mathcal{L}_{total} = \underbrace{\mathcal{L}_{p2s} + \lambda_1\mathcal{L}_{sampling} + \lambda_2\mathcal{L}_{radius}}_{\mathcal{L}_{skel}} + \mathcal{L}_{cls}
$$

여기서 $$\mathcal{L}_{cls}$$는 레이블된 소스 데이터에 대한 교차 엔트로피이다. ablation에서 가장 좋은 계수는 Point2Skeleton을 따른다.

### 테스트 시간 적응

그림의 위쪽 행은 두 가지 적응 방식을 보여준다. 둘 다 MATE의 프로토콜을 따른다(배치 크기 1로, 샘플을 증강 뷰들의 배치로 복사하며, 여기서는 무작위 마스크 대신 무작위 회전을 사용한다).

- **온라인, BatchNorm만 사용(† 표시)**: 샘플 사이에는 모델을 초기화하지 않고, 손상 종류가 바뀔 때만 초기화한다. 기울기는 계산하지 않는다. $$y_i = \gamma \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta$$에서 BatchNorm의 이동 평균 $$\mu$$와 분산 $$\sigma^2$$만 대상 데이터로 갱신된다.
- **$$\mathcal{L}_{skel}$$를 사용하는 Online 또는 Standard**: 분류기를 고정한 채, 스켈레톤 손실을 역전파하여 인코더, 디코더, 스켈레톤 MLP를 갱신한다. Standard 모드는 배치마다 모델을 초기화하고 20회 반복을 실행하므로, 여기서는 BatchNorm 통계만으로는 도움이 되지 않는다.

## 주요 실험 (Main Experiments)

백본: Point-MAE. 지표: 15개 손상에 대한 평균 top-1 정확도(%). 배치 크기 1, 1회 반복(online) 또는 20회 반복(standard). Table 1에서 옮겼으며, 논문의 MATE 대비 차이는 생략했다. *는 저자들이 재현한 결과, †는 역전파 없는 적응, ‡는 확산(diffusion) 기반 방법을 나타낸다.

| 방법 | ModelNet-40C | ScanObjectNN-C | ShapeNet-C |
|---|---|---|---|
| Org-SO* | 54.0 | 37.0 | 61.3 |
| MATE-SO* | 54.4 | 34.5 | 56.5 |
| SMART-PC-SO | 61.7 | 38.7 | 64.5 |
| DUA | 54.7 | 38.7 | 60.8 |
| TTT-Rot | 53.0 | – | 60.9 |
| SHOT | 26.6 | 29.6 | 36.2 |
| T3A | 55.7 | – | 54.4 |
| TENT | 26.5 | 27.7 | 37.4 |
| CloudFixer-Standard‡ | 68.0 | - | - |
| CloudFixer-Online‡ | 77.2 | - | - |
| DDA-Standard‡ | 68.1 | - | - |
| SVWA-Standard* | 57.1 | 37.4 | 50.5 |
| MATE-Standard* | 63.0 | 36.9 | 63.1 |
| SMART-PC-Standard | 63.1 | 39.6 | 64.4 |
| BFTT3D* | 57.2 | 33.0 | 60.7 |
| MATE-Online* | 70.6 | 36.9 | **69.1** |
| SMART-PC-Online† | 70.8 | 46.7 | 65.9 |
| SMART-PC-Online | 72.9 | **47.4** | 67.1 |

ModelNet40-C에서의 적응 속도(부록 Table 4):

| 방법 | DDA | CloudFixer | BFTT3D | SVWA (Nv=2) | MATE | SMART-PC |
|---|---|---|---|---|---|---|
| FPS | 0.04 | 1.07 | 6.83 | 10.86 | 10.79 | 59.52 |

가장 뚜렷한 향상은 적응 이전에 나타난다. 스켈레톤 사전 학습만으로도 MATE-SO 대비 source-only 정확도가 ModelNet-40C에서 7.3포인트, ShapeNet-C에서 8.0포인트 오른다. Standard 모드에서 SMART-PC와 MATE는 비슷하다(ModelNet-40C에서 63.1 vs 63.0). BatchNorm만 사용하는 변형은 ModelNet-40C에서 MATE-Online과 비슷하고, ScanObjectNN-C에서는 약 5.5배의 프레임 속도로 약 10포인트 앞서지만, ShapeNet-C에서는 MATE-Online보다 뒤처진다(65.9 vs 69.1). CloudFixer-Online은 ModelNet-40C에서 여전히 더 높지만(77.2), 약 1 FPS로 동작한다.

## 한계 (Limitations)

- 논문에는 한계 섹션이 없다. 결론에서 다른 3D 과제(분할, 검출)를 향후 연구로 언급하므로, 분류만 시험되었다.
- 스켈레톤 분기를 포함한 특별한 사전 학습 단계가 필요하다. BatchNorm만 사용하는 모드는 이러한 특징 덕분에 동작하므로, 임의의 사전 학습 모델에는 적용되지 않는다.
- ShapeNet-C에서 두 온라인 변형 모두 MATE-Online보다 낮다(67.1과 65.9 vs 69.1).
- MATE 기준선은 재현된 것이며 MATE 논문보다 약간 낮게 나온다(ModelNet-40C에서 MATE-Online 70.6 vs 71.3, MATE-Standard 63.0 vs 64.3). 원래 보고된 71.3과 비교하면 SMART-PC-Online†(70.8)는 더 이상 앞서지 않는다. 본문도 곳곳에서 표와 다른 MATE 수치(53.7과 69.6)를 인용한다.
- 배치 크기 1이라도 각 단계마다 샘플당 48개 또는 24개의 회전된 사본을 사용하며, BatchNorm만 사용하는 모드는 스트림에 걸쳐 이동 통계를 갱신한다. 유용한 통계를 만들려면 샘플 스트림이 필요하다. 프로토콜은 또한 새로운 손상 종류마다 모델을 초기화하는데, 이는 분포 이동이 바뀌는 시점을 안다고 가정하는 것이다. 혼합되거나 계속 변하는 분포 이동은 시험되지 않았다.
- 속도에 대한 주장은 주로 BatchNorm만 사용하는 온라인 모드에 해당한다. 전체 모델을 통해 $$\mathcal{L}_{skel}$$를 역전파하는 모드들은, 20단계 standard 모드를 포함하여, MATE와 비슷한 비용이 든다.

## 내 생각 (My Thoughts)

<!-- TODO -->

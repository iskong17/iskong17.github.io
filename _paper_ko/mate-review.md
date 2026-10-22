---
title: "MATE: 테스트 포인트 클라우드를 하나씩 재구성하며 3D 분류기를 적응시키기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
en: mate-review
date: 2026-10-22 00:00:00 +0900
description: MATE는 Point-MAE 분류기를 마스크 재구성과 함께 공동 학습한 뒤, 손상된 각 테스트 포인트 클라우드를 분류하기 전에 이를 재구성하며 인코더를 갱신한다.
conference: ICCV
year: 2023
paper:
  title: "MATE: Masked Autoencoders are Online 3D Test-Time Learners"
  authors: "M. Jehanzeb Mirza, Inkyu Shin, Wei Lin, Andreas Schriebl, Kunyang Sun, Jaesung Choe, Mateusz Kozinski, Horst Possegger, In So Kweon, Kuk-Jin Yoon, Horst Bischof"
  venue: "ICCV 2023"
  url: "https://arxiv.org/abs/2211.11432"
---

## 초록 (Abstract)

포인트 클라우드 분류기는 센서 잡음, LiDAR 아티팩트, 영역 누락 같은 손상 앞에서 무너진다. 가능한 모든 분포 변화에 대비해 학습하는 것은 불가능하므로, 모델은 테스트 시간에 레이블 없이 온라인으로 적응해야 한다. 2D 이미지를 위한 테스트 시간 학습(test-time training, TTT) 방법은 존재하지만, 논문은 이를 포인트 클라우드로 옮기면 잘 작동하지 않음을 보인다. TENT와 SHOT 같은 엔트로피 방법은 한 번에 샘플 하나만 볼 때 붕괴하며, 회전 예측(TTT-Rot)은 평균적으로 적응하지 않은 모델보다도 낮다.

MATE는 3D 포인트 클라우드를 위해 만들어진 최초의 TTT 방법이다. 자기 지도(self-supervised) 과제로 마스크 오토인코딩(masked autoencoding)을 사용한다. 각 테스트 포인트 클라우드에 대해 대부분의 토큰을 마스킹하고, 네트워크에게 전체 형상을 재구성하게 한 뒤, 그 재구성 손실로 그래디언트 스텝을 밟고, 갱신된 인코더로 샘플을 분류한다.

## 방법 (Method)

![MATE overview](/images/blog/papers/mate-main.png)
*MATE의 네 단계: 토큰화와 마스킹, 분류와 재구성의 공동 학습, 재구성 손실만을 사용한 테스트 시간 학습, 갱신된 인코더를 사용한 온라인 평가. 그림은 논문에서 가져왔다.*

### 토큰화와 마스킹

MATE는 Point-MAE를 기반으로 한다. 포인트 클라우드 $$\mathcal{X}=\{\mathbf{p}_i\}_{i=1}^{N}$$는 토큰(최원점 샘플링(farthest point sampling)으로 뽑은 중심 주변의 지역 패치)으로 묶인다. 토큰의 비율 $$m$$이 무작위로 마스킹된다. 마스킹된 토큰은 $$\mathcal{X}^{m}$$, 보이는 토큰은 $$\mathcal{X}^{v}$$이다. MATE는 토큰의 90%를 마스킹하므로, 인코더는 10%만 본다.

네트워크는 네 부분으로 이루어진다: 보이는 토큰만 보는 인코더 $$E$$(Transformer 블록 12개), 가벼운 디코더 $$D$$(블록 4개), 점 좌표를 출력하는 선형 예측 헤드 $$P$$, 그리고 분류기 헤드 $$C$$.

### 공동 학습

Point-MAE는 보통 재구성으로 인코더를 사전학습한 다음 분류기를 미세조정한다. MATE는 대신 레이블이 있는 소스 데이터 $$\mathcal{S}$$로 두 헤드를 함께 학습하여, 인코더가 두 과제 모두에 쓸모 있는 특징을 배우게 한다.

$$
\min_{\theta_E,\theta_D,\theta_P,\theta_C} \mathbb{E}_{(\mathcal{X},\mathcal{Y})\in\mathcal{S}} \big[ L_c(\mathcal{X},\mathcal{Y};\theta_E,\theta_C) + \lambda \cdot L_s(\mathcal{X};\theta_E,\theta_D,\theta_P) \big]
$$

여기서 $$\lambda=1$$이다. 분류 손실은 보이는 토큰에 대한 교차 엔트로피이고, 자기 지도 손실은 재구성 결과와 입력 사이의 Chamfer 거리이다.

$$
L_c = CE\big(C\circ E(\mathcal{X}^{v}), \mathcal{Y}\big), \qquad
L_s = CD\big(P\circ D\circ E(\mathcal{X}^{v}), \mathcal{X}\big)
$$

### 테스트 시간 학습

테스트 시간에 MATE는 레이블 없는 손상된 포인트 클라우드 $$\tilde{\mathcal{X}}$$ 하나를 받는다. 샘플을 마스킹하고 재구성한 뒤, 재구성 손실만 최소화한다.

$$
L_{TTT} = \min_{\theta_E,\theta_D,\theta_P} L_s(\tilde{\mathcal{X}};\theta_E,\theta_D,\theta_P)
$$

분류기 헤드 $$C$$는 고정된다. 샘플 하나로 쓸 만한 손실을 얻기 위해, MATE는 샘플을 48개로 복사해 배치를 만들고 각 사본을 서로 다르게 마스킹한다. 무작위 마스킹이 증강 역할을 하므로 다른 테스트 시간 증강은 쓰지 않는다.

### 온라인 평가

갱신 후에는 전체 포인트 클라우드(모든 토큰, 마스킹 없음)를 갱신된 인코더와 고정된 분류기에 통과시킨다. 두 가지 변형이 있다.

- **MATE-Standard**: 샘플마다 그래디언트 스텝 20번을 밟고, 다음 샘플 전에 모든 가중치를 초기화한다.
- **MATE-Online**: 샘플마다 그래디언트 스텝 1번을 밟고, 갱신이 테스트 스트림에 걸쳐 누적된다.

인코더는 보이는 토큰만 처리하므로, 높은 마스킹 비율이 테스트 시간 비용을 낮게 유지시킨다. 논문은 토큰의 95%를 마스킹해도 90%와 거의 비슷하게 작동한다고 보고한다.

## 주요 실험 (Main Experiments)

백본: Point-MAE. 지표: 각 벤치마크의 15개 손상에 대해 평균한 top-1 정확도(%). 모든 방법은 배치 크기 1(한 번에 테스트 샘플 하나)로 적응한다. 논문은 손상별 수치를 보고하지만, 이 표는 Table 1, 3, 4의 평균 열만 보여준다.

| 방법 | ModelNet-40C | ShapeNet-C | ScanObjectNN-C |
|---|---|---|---|
| Source-Only | 53.9 | 57.6 | 45.7 |
| Joint-Training | 57.6 | 59.3 | 45.6 |
| DUA | 54.7 | 60.8 | 46.0 |
| TTT-Rot | 53.0 | 60.9 | 46.1 |
| SHOT | 26.6 | 36.2 | 38.3 |
| T3A | 55.7 | 54.4 | 40.3 |
| TENT | 26.5 | 37.4 | – |
| MATE-Standard | 64.3 | 63.1 | 47.0 |
| MATE-Online | **71.3** | **69.1** | **48.5** |

2D 베이스라인에 더 공정한 기회를 주기 위해 ModelNet-40C에서 배치 크기 128을 사용한 경우(Table 2, 평균 정확도 %):

| Source | TENT | SHOT | T3A | MATE-Online |
|---|---|---|---|---|
| 53.9 | 65.6 | 63.8 | 55.9 | **74.5** |

두 MATE 변형 모두 세 벤치마크에서 모든 베이스라인을 앞선다. 엔트로피 기반 방법(TENT, SHOT)은 단일 샘플에서 크게 실패하며, 큰 배치에서도 일부만 회복한다. 실제 데이터인 ScanObjectNN-C에서의 향상(MATE-Online이 Source-Only 대비 +2.8점)은 합성 벤치마크보다 훨씬 작다. 5번째 샘플마다 한 번만 적응해도 정확도는 매 샘플 적응 대비 약 3점 이내로 유지되며, ShapeNet-C에서 20 FPS로 동작한다(RTX 3090).

## 한계 (Limitations)

- 논문이 직접 밝힌 한계: 포인트 클라우드 분류만 다룬다. 분할(segmentation)이나 검출(detection) 같은 다른 과제는 향후 과제로 남겨둔다.
- 특수한 학습 단계가 필요하다. 소스 모델은 재구성 분기와 함께 공동 학습되어야 하므로, MATE는 임의의 기성(off-the-shelf) 분류기를 적응시킬 수 없다.
- 각 적응 스텝은 샘플의 마스킹된 사본 48개로 이루어진 배치에 대해 순전파와 역전파를 필요로 한다. MATE-Standard는 샘플마다 이런 스텝을 20번 밟으므로 실시간과는 거리가 멀다. 속도 향상은 샘플을 건너뛰는 것(stride)에서 오며, 정확도를 어느 정도 희생한다.
- 사본 배치가 중요하다. 배치 크기 1에서는 MATE-Online이 ModelNet-40C에서 43.1%로 떨어져, 적응하지 않은 모델보다도 낮다(Table 6).
- 마스크 오토인코더 Transformer(Point-MAE)에 묶여 있다. 논문은 다른 백본을 검증하지 않는다.
- 재구성하기 어려운 손상에는 재구성이 도움이 되지 않는다. background 잡음에서 MATE-Standard는 ModelNet-40C에서 27.5%(TTT-Rot는 34.5%), ShapeNet-C에서 4.3%로 Source-Only(10.3%)보다 낮다.
- MATE-Online의 가장 큰 향상은 하나의 손상 유형만 담긴 스트림에서 갱신을 누적하는 데서 나온다. 논문은 분포 변화가 계속 바뀌는 스트림을 다루지 않지만, 샘플마다 두 손상을 결합한 테스트 세트는 검증한다(Table 7).

## 내 생각 (My Thoughts)

<!-- TODO -->

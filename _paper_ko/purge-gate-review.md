---
title: "Purge-Gate: 가장 많이 이동한 토큰을 어텐션 전에 제거하기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
description: 포인트 클라우드 트랜스포머를 위한 역전파 없는 TTA 방법으로, 각 토큰이 소스 프로토타입에서 얼마나 떨어져 있는지 점수를 매기고 가장 크게 벗어난 토큰들을 어텐션에 들어가기 전에 제거한다.
conference: ICCV
year: 2025
en: purge-gate-review
date: 2026-10-22 00:00:00 +0900
paper:
  title: "Purge-Gate: Backpropagation-Free Test-Time Adaptation for Point Clouds Classification via Token Purging"
  authors: "Moslem Yazdanpanah, Ali Bahri, Mehrdad Noori, Sahar Dastani, Gustavo Adolfo Vargas Hakim, David Osowiechi, Ismail Ben Ayed, Christian Desrosiers"
  venue: "ICCV 2025"
  url: "https://arxiv.org/abs/2509.09785"
---

## 초록 (Abstract)

포인트 클라우드 테스트 시간 적응(test-time adaptation, TTA) 방법들은 대부분 여러 번의 역전파 단계로 모델을 적응시킨다. 이 때문에 느리고, 갱신 단계 수에 민감하다. MATE처럼 학습 시점에 보조 과제(auxiliary task)가 필요한 방법도 있어, 임의의 사전 학습 모델에는 적용할 수 없다. BFTT3D는 역전파를 피하지만, 테스트 시점에 클래스 수준의 소스 프로토타입이 필요하고 시간과 메모리 부담이 크다.

Purge-Gate(PG)는 토큰 수준에서 동작한다. 배경 점이나 임펄스 노이즈 같은 손상(corruption)은 일부 지역 패치에 다른 패치보다 훨씬 크게 영향을 준다. PG는 각 토큰이 소스 프로토타입에서 얼마나 떨어져 있는지 측정하고, 가장 크게 벗어난 토큰들을 어텐션 층 이전에 제거한다. 아무것도 갱신하지 않는다. 두 가지 변형이 있다. PG-SP는 저장된 소스 통계를 사용하고, PG-SF는 완전히 소스 프리(source-free)하며 모델 자신의 [cls] 토큰을 프로토타입으로 사용한다.

## 방법 (Method)

![Purge-Gate overview](/images/blog/papers/purge-gate-main.png)
*입력 토큰 $$X_j^T$$는 소스 프로토타입 $$G$$까지의 거리 $$\Delta$$로 점수가 매겨진다. 가장 크게 벗어난 $$L_{\text{pg}}$$개의 토큰이 제거되어 정제된 입력 $$\hat{X}_j^T$$가 된다. 그림은 논문에서 가져왔다.*

### 토큰

백본은 Point-MAE 방식의 트랜스포머이다. 포인트 클라우드는 $$L_t$$개의 토큰으로 나뉜다. 각 토큰 $$x_{i,t} = (c_t, \mathcal{N}_t)$$는 최원점 샘플링(farthest point sampling)으로 고른 중심 $$c_t$$와 그 $$k$$개의 최근접 이웃 $$\mathcal{N}_t$$로 이루어진다. 그림 왼쪽의 색칠된 패치들이 이것이다.

### 노이즈 토큰이 어텐션을 해치는 이유

논문은 분포 이동을 임베딩에 더해지는 노이즈 $$x_{j,t}^T = x_{j,t}^S + \epsilon_{j,t}$$로 모델링한다. LayerNorm은 국소적으로 립시츠(Lipschitz) 연속이므로, 작은 노이즈는 토큰을 깨끗한 버전 근처에 머물게 한다. 노이즈가 지배적이면 $$\mathit{LN}(x_{j,t}^T) \approx \mathit{LN}(\epsilon_{j,t})$$가 되며, 이는 고차원에서 무작위 방향에 가깝다. 무작위 방향들은 거의 직교하므로 이러한 토큰에 대한 어텐션 점수는 거의 0이 되고, softmax는 균등 가중치 쪽으로 흘러간다.

$$
A_{ij} = \frac{\exp(S_{ij}/\sqrt{d})}{\sum_k \exp(S_{ik}/\sqrt{d})} \approx \frac{1}{L_t}
$$

균등한 어텐션은 층이 유용한 정보를 더 이상 모으지 못한다는 뜻이다. 따라서 노이즈가 지배하는 토큰을 어텐션에 들어가기 전에 제거하자는 것이 핵심 아이디어이다.

### 게이트

그림의 가운데 부분이다. PG는 모든 대상 토큰과 소스 프로토타입 $$G$$ 사이의 거리를 계산한다.

$$
\Delta(X_j^T, G) = (\delta_{j,1}^T, \delta_{j,2}^T, \dots, \delta_{j,L_t}^T)
$$

그런 다음 거리가 가장 큰 $$L_{\text{pg}}$$개의 토큰을 버린다.

$$
\hat{X}_j^T = X_j^T \setminus \big\{ x_{j,t}^T \mid t \in \arg\max_{S \subseteq [L_t],\, |S| = L_{\text{pg}}} \sum_{t \in S} \delta_{j,t}^T \big\}
$$

제거 크기(purge size)는 배치를 병렬로 처리할 수 있도록 순전파마다 고정된다. 남은 문제는 $$G$$와 $$\Delta$$를 어떻게 정하느냐뿐이다.

### PG-SP: 소스 프로토타입

$$G = (\mu_S, \sigma_S)$$는 소스 토큰 임베딩의 평균과 표준편차이다. 이는 (a) Welford 온라인 알고리즘으로 소스 데이터를 한 번 훑어서 수집하거나, (b) 학습 중에 이 통계를 누적하는 수정된 첫 번째 LayerNorm으로 수집한다. 거리는 대각 마할라노비스(Mahalanobis) 거리이다.

$$
\Delta(X_j^T, G) = \sqrt{(X_j^T - \mu_S)^T \Sigma_S^{-1} (X_j^T - \mu_S)}, \qquad \Sigma_S = \text{diag}(\sigma_S^2)
$$

### PG-SF: 소스 프리 프로토타입

학습 가능한 [cls] 토큰은 소스 토큰에 어텐션을 주고 이를 요약하도록 학습된다. 그래서 PG-SF는 이를 쿼리 공간으로 사영한 $$G = \mathit{LN}([\texttt{cls}])W_Q$$를 하나의 벡터로 된 프로토타입으로 사용한다. 각 토큰의 키는 음의 코사인 유사도로 이와 비교된다.

$$
\Delta(X_j^T, G) = -\frac{(\mathit{LN}(X_j^T)W_K)\, G}{\|\mathit{LN}(X_j^T)W_K\| \cdot \|G\|}
$$

[cls] 쿼리가 거의 어텐션을 주지 않을 토큰들이 제거된다.

### 제거 크기 선택과 BatchNorm

최적의 제거 크기는 손상 종류에 따라 다르며, 기울기 없이는 학습할 수 없다. PG는 작은 집합 $$\mathcal{L}_{\text{pg}} = \{0, 2, 4, 8, 16, 32\}$$에 대해 모델을 실행하고 엔트로피가 가장 낮은 출력을 택한다.

$$
L_{\text{pg}}^* = \arg\min_{L \in \mathcal{L}_{\text{pg}}} H(f(x; L))
$$

마지막으로, 적응 단계가 없으므로 BatchNorm 층은 순전파마다 현재 배치 통계를 사용하도록 재설정된다.

## 주요 실험 (Main Experiments)

15개 손상에 대한 평균 top-1 정확도(%, 높을수록 좋음)이다. 모든 방법은 MATE가 공개한 Point-MAE 백본을 사용한다. 배치 크기 32, 1회 반복, 배치마다 모델을 초기화한다. 이는 Table 1~3의 "Mean" 열이며, 논문은 손상별 정확도도 제시한다. *로 표시된 행은 저자들이 재현한 결과이다. MATE와 SMART-PC는 학습 시점 보조 과제가 필요하며, 그 수치는 SMART-PC 논문에서 가져왔다.

| 방법 | 유형 | ScanObjectNN-C | ModelNet40-C | ShapeNet-C |
|---|---|---|---|---|
| Source Only | – | 36.96 | 54.16 | 61.35 |
| MATE | TTT | 36.90 | 58.90 | 63.10 |
| SMART-PC | TTT | 39.60 | 63.10 | **64.40** |
| TTT-Rot | TTA, 역전파 | 29.58 | 53.05 | 60.93 |
| SHOT | TTA, 역전파 | 27.69 | 26.60 | 36.19 |
| T3A* | TTA, 역전파 | 42.46 | 57.22 | 54.18 |
| TENT* | TTA, 역전파 | 43.45 | 63.53 | 54.98 |
| DUA | TTA, 역전파 없음 | 38.66 | 54.73 | 60.77 |
| BFTT3D* | TTA, 역전파 없음 | 33.00 | 57.16 | 60.73 |
| **PG-SP** | TTA, 역전파 없음 | **49.13** | **68.93** | 64.17 |
| **PG-SF** | TTA, 역전파 없음, 소스 프리 | 47.54 | 68.68 | **64.40** |

ShapeNet-C에서 MATE와 SMART-PC 행은 "Standard" 변형이다.

BFTT3D 대비 비용(배치 크기 32, NVIDIA A6000 한 장, 여섯 가지 제거 크기는 병렬이 아니라 순차적으로 실행):

| 방법 | 배치당 시간 | GPU 메모리 |
|---|---|---|
| BFTT3D | 173.94 ms | 7400 MiB |
| PG-SP | 14.07 ms | 1300 MiB |

PG-SP와 PG-SF는 ScanObjectNN-C와 ModelNet40-C에서 가장 높은 평균 정확도를 기록하며, 경사 단계 없이도 TENT와 TTT 방법들을 앞선다. ShapeNet-C는 더 어렵다. 대부분의 TTA 방법이 Source Only보다 낮아지는 반면, PG는 약 3포인트를 얻어 SMART-PC와 동률이다. PG-SP는 BFTT3D보다 12.4배 빠르고 메모리를 5.5배 적게 쓴다. Point-Mamba 백본(Table 4)에서 PG-SP는 ScanObjectNN-C를 29.11에서 41.42로, ModelNet-C를 58.32에서 68.61로 향상시키지만, ShapeNet-C에서는 Source Only보다 약간 낮다(51.66 vs. 51.92).

## 한계 (Limitations)

논문에는 한계 섹션이 없다. 근거가 있는 관찰은 다음과 같다.

- **배치마다 여러 번의 순전파.** 제거 크기 탐색은 $$\mathcal{L}_{\text{pg}}$$의 값마다(여섯 개) 모델을 한 번씩 실행한다. 이는 병렬 또는 순차로 실행되므로, 단일 순전파보다 메모리나 시간이 더 든다.
- **탐색 범위가 너무 좁을 수 있음.** 보충 자료는 심한 손상에서는 훨씬 더 많은 제거가 필요함을 보여준다. 심각도 7의 배경 손상에서 정확도는 $$L_{\text{pg}} \approx 112$$ 근처에서 최고(33%)에 달하지만, $$[0, 32]$$ 범위에서 가장 좋은 값은 16%에 그친다. 고정된 집합 $$\{0, 2, 4, 8, 16, 32\}$$로는 여기에 도달할 수 없다.
- **BatchNorm 재설정의 영향이 큼.** 재설정이 없으면 PG-SP는 ScanObjectNN-C에서 49.13에서 39.86으로 떨어진다. 같은 재설정은 BFTT3D를 33.00에서 45.76으로 끌어올리지만, Table 1에는 BFTT3D가 33.00으로 기재되어 있다. BFTT3D 대비 주요 격차의 일부는 제거가 아니라 재설정에서 온다. 재설정은 배치 통계에도 의존한다. 배치 크기 연구(ShapeNet-C)는 배치 크기가 커질수록 정확도가 여전히 오른다는 것을 보여주지만, PG-SP는 모든 크기에서 TENT보다 높다.
- **PG-SP는 소스 접근 또는 재학습이 필요.** PG-SP는 통계를 위한 소스 데이터나, 수정된 LayerNorm으로 학습된 모델이 필요하다. PG-SF는 이를 피하지만 [cls] 토큰과 쿼리/키 사영이 필요하므로 트랜스포머 분류기에 묶여 있다. Mamba 실험은 PG-SP만 사용한다.
- **전역 토큰 거리만 사용.** 소스 평균에서 멀리 떨어진 토큰이 항상 손상된 것은 아니다. 너무 많이 제거하면 정보가 담긴 토큰까지 사라지며, ablation에서 제거 토큰이 약 80~100개를 넘으면 정확도가 급격히 떨어진다.
- **합성 분포 이동 위주.** 실제 분포 이동은 두 가지 ScanObjectNN 변형에서만 시험되며, 향상 폭도 더 작다(OBJ-BG에서 +1.38, PB-T50-RS에서 +4.30).

## 내 생각 (My Thoughts)

<!-- TODO -->

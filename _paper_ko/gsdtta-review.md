---
title: "GSDTTA: 가장 낮은 그래프 주파수만 편집해 포인트 클라우드를 적응시키기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
en: gsdtta-review
date: 2026-10-22 00:00:00 +0900
description: GSDTTA는 모든 점을 움직이는 대신 가장 낮은 그래프 푸리에 계수 100개에 대한 오프셋만 학습해 테스트 포인트 클라우드를 이동시키고, 라플라시안 고유맵의 도움으로 레이블을 붙인다.
conference: ICCV
year: 2025
paper:
  title: "3D Test-time Adaptation via Graph Spectral Driven Point Shift"
  authors: "Xin Wei, Qin Yang, Yijie Fang, Mingrui Zhu, Nannan Wang"
  venue: "ICCV 2025"
  url: "https://arxiv.org/abs/2507.18225"
---

## 초록 (Abstract)

깨끗한 데이터로 학습된 포인트 클라우드 분류기는 배경 잡동사니(background clutter), 가림(occlusion), LiDAR 방식 샘플링 같은 손상이 있는 테스트 데이터에서 정확도를 크게 잃는다. 테스트 시간 적응(test-time adaptation, TTA)은 레이블 없는 테스트 배치만을 사용해 이를 온라인으로 바로잡으려 한다. 기존 3D TTA 방법은 대부분 공간 영역(spatial domain)에서 작동한다. MATE는 소스 학습 단계에서 추가 재구성 과제가 필요하다. BFTT3D는 소스 데이터의 특징 프로토타입이 필요하다. CloudFixer와 3DD-TTA는 소스 데이터로 학습된 확산 모델(diffusion model)이 필요하다. 그리고 점별 오프셋 $$\Delta P \in R^{N\times 3}$$을 학습하는 방법들은 소수의 스트리밍 테스트 샘플로 풀기 어려운 고차원 문제에 직면한다.

GSDTTA는 적응을 그래프 스펙트럼 영역(graph spectral domain)으로 옮긴다. 각 테스트 포인트 클라우드는 kNN 그래프가 되어 그래프 푸리에 변환(Graph Fourier Transform, GFT)으로 변환되고, 가장 낮은 $$M=100$$개의 주파수 계수(약 10%)에만 학습 가능한 오프셋이 주어진다. 역 GFT는 이 오프셋을 매끄럽고 전역적인 점 이동으로 바꾼다. 최적화를 위한 의사 레이블(pseudo-label)은 모델의 깊은 특징과 라플라시안 고유맵(Laplacian eigenmap) 특징을 혼합하며, 저자들은 후자가 소스 도메인에 덜 편향되어 있다고 주장한다.

## 방법 (Method)

![GSDTTA overview](/images/blog/papers/gsdtta-main.png)
*GSDTTA 파이프라인. 위쪽 절반(GSDPS)은 스펙트럼 영역을 통해 입력 포인트 클라우드를 이동시킨다. 아래쪽 절반(GSGMA)은 모델을 갱신한다. 둘 다 고유맵 기반 의사 레이블로 학습된다. 그림은 논문에서 가져왔다.*

이 방법은 각 테스트 배치 $$\{X_i\}_{i=1}^{B}$$에 대해 번갈아 작동하는 두 모듈로 구성된다. 그래프 스펙트럼 기반 점 이동(Graph Spectral Driven Point Shift, GSDPS)은 입력을 적응시키고, 그래프 스펙트럼 유도 모델 적응(Graph Spectral Guided Model Adaptation, GSGMA)은 모델 $$f_\theta$$를 적응시킨다.

### 이상치를 고려한 그래프

포인트 클라우드 $$X \in \mathbb{R}^{N\times 3}$$에 대해, 간선은 각 점을 $$k$$개의 최근접 이웃과 연결하며 RBF 커널로 가중치가 매겨진다.

$$
w_{ij} = \exp\left(-\frac{d^2(x_i, x_j)}{2\delta^2}\right), \qquad A_{ij} = w_{ij}\cdot \mathbb{I}(x_j \in \mathcal{N}(x_i)).
$$

스펙트럼 분석은 이상치(outlier)에 민감하다. 이상치는 다른 점들과 멀리 떨어져 있으므로 차수(간선 가중치의 합)가 낮다. GSDTTA는 차수가 임계값 $$\tau$$보다 낮은 정점을 제거한다.

$$
A^o_{ij} = w_{ij}\cdot \mathbb{I}(x_j \in \mathcal{N}(x_i)) \cdot \mathbb{I}\Big(\sum_{j=1}^{N} A_{ij} > \tau\Big), \qquad \tau = \frac{\gamma}{Nk}\sum_{i=1}^{N}\sum_{j=1}^{N} A_{ij}.
$$

### 스펙트럼 조정과 점 이동

라플라시안 $$L_o = D_o - A_o$$는 $$L_o = U_o \Lambda_o U_o^T$$로 고유분해된다. 좌표의 GFT는 다음과 같다.

$$
\hat{X} = U_o^T X, \qquad \hat{X} \in \mathbb{R}^{N\times 3}.
$$

$$M \ll N$$인 학습 가능한 오프셋 $$\Delta\hat{X} \in R^{M\times 3}$$을 저주파 행에만 더하고, 그 결과를 역 GFT로 되돌린다.

$$
\hat{X}_a = \hat{X} + [\Delta\hat{X}, O], \qquad X_s = U_o \hat{X}_a.
$$

여기서 $$O$$는 $$(N-M)\times 3$$ 영행렬이다. 동기는 에너지 집중(energy compaction)이다. 의자 형상의 경우 스펙트럼 에너지의 약 95%가 저주파에 있으며, 가장 낮은 10%의 계수만으로 전역 형상을 재구성할 수 있다. 따라서 최적화는 포인트 클라우드당 $$N\times 3$$개 대신 $$M\times 3 = 300$$개의 파라미터만 가진다.

### 고유맵 기반 자기 학습

고유벡터는 형상 기술자(shape descriptor)도 제공한다. $$\lambda_1 = 0$$에 대응하는 고유벡터를 버리고 나머지 $$m$$개를 남기면 각 노드를 $$E(v_i) = [u_1(v_i), \dots, u_m(v_i)]$$로 임베딩할 수 있다. 노드들에 대해 최대 풀링(max-pooling)을 하면 전역 스펙트럼 기술자 $$f_s$$를 얻으며, 이는 Global Point Signature를 단순화한 것이다. 모델 자체의 전역 특징은 깊은 기술자 $$f_d$$이다.

두 공간 모두에서 각 샘플을 예측 확률로 가중하여 클래스 중심(centroid)을 만든다.

$$
q_d^c = \frac{\sum_{i=1}^{B} (f_\theta(X_i))_c\, f_d^i}{\sum_{i=1}^{B} (f_\theta(X_i))_c}, \qquad
q_s^c = \frac{\sum_{i=1}^{B} (f_\theta(X_i))_c\, f_s^i}{\sum_{i=1}^{B} (f_\theta(X_i))_c}.
$$

의사 레이블은 두 중심 집합에 대한 코사인 유사도를 가중치 $$\alpha$$로 혼합한다.

$$
\hat{y}_i = \arg\max_c \left( \alpha \frac{(f_d^i)^T q_d^c}{\|f_d^i\|\|q_d^c\|} + (1-\alpha)\frac{(f_s^i)^T q_s^c}{\|f_s^i\|\|q_s^c\|} \right).
$$

(논문의 Eq. 16은 $$\arg\min$$으로 적혀 있다. 각 항이 유사도이므로, 여기서는 가장 유사한 중심을 고르는 형태로 적었다.)

### 무엇을 최적화하는가

입력 단계는 스펙트럼 오프셋만 최적화한다.

$$
\arg\min_{\Delta\hat{X}} \mathcal{L}_{IA} = \mathcal{L}_{pl} + \beta_1(\mathcal{L}_{ent} + \mathcal{L}_{div}) + \beta_2 \mathcal{L}_{cd},
$$

여기서 $$\mathcal{L}_{pl}$$은 $$f_\theta(X_s)$$와 $$\hat{y}$$ 사이의 교차 엔트로피, $$\mathcal{L}_{ent} + \mathcal{L}_{div}$$는 일반적인 정보 최대화(information-maximisation) 손실, $$\mathcal{L}_{cd}$$는 $$X$$에서 $$X_s$$로의 단방향 Chamfer 거리로 이동된 클라우드를 입력 가까이에 유지시킨다. 모델 단계는 Chamfer 항을 뺀 같은 손실로 $$\theta$$를 갱신한다.

$$
\arg\min_{\theta} \mathcal{L}_{MA} = \mathcal{L}_{pl} + \beta_3(\mathcal{L}_{ent} + \mathcal{L}_{div}).
$$

배치마다 GSDTTA는 입력 단계 4번 후 모델 단계 1번을 실행하며, 이를 반복해 총 10스텝을 진행한다(AdamW, lr 0.0001, 배치 크기 32, $$k=10$$, $$M=100$$).

## 주요 실험 (Main Experiments)

백본은 DGCNN, CurveNet, PointNeXt이며, 깨끗한 데이터셋으로 학습한 뒤 배치 크기 32로 적응시킨다. 지표는 15개 손상에 대해 평균한 분류 정확도(%)이다. 아래 표는 논문 Table 1과 2의 평균 열만 보여준다. 3DD-TTA는 DGCNN에 대해서만 보고된다. 베이스라인 수치는 저자들이 각 방법의 공개 코드를 같은 사전학습 가중치로 실행해 얻은 것이다.

**ModelNet40-C, 평균 정확도 (%)**

| 방법 | DGCNN | CurveNet | PointNeXt |
|---|---|---|---|
| Source-only | 66.51 | 71.38 | 66.99 |
| BN | 73.34 | 77.45 | 80.02 |
| PL | 75.02 | 78.56 | 80.81 |
| DUA | 73.81 | 77.41 | 80.57 |
| TENT | 75.91 | 77.88 | 81.08 |
| SHOT | 77.36 | 81.24 | 80.46 |
| BFTT3D | 65.86 | 70.95 | 68.39 |
| CloudFixer | 76.54 | 77.91 | 76.04 |
| 3DD-TTA | 71.69 | – | – |
| **GSDTTA** | **79.07** | **82.63** | **82.51** |

**ScanObjectNN-C, 평균 정확도 (%)**

| 방법 | DGCNN | CurveNet | PointNeXt |
|---|---|---|---|
| Source-only | 55.72 | 53.06 | 50.66 |
| BN | 57.15 | 57.33 | 56.68 |
| PL | 57.68 | 57.90 | 56.41 |
| DUA | 57.47 | 58.35 | 58.13 |
| TENT | 57.31 | 58.71 | 58.05 |
| SHOT | 56.32 | 58.17 | 57.06 |
| BFTT3D | 56.47 | 52.57 | 51.10 |
| CloudFixer | 60.73 | 58.96 | 59.01 |
| 3DD-TTA | 57.08 | – | – |
| **GSDTTA** | **61.83** | **62.59** | **60.84** |

GSDTTA는 두 벤치마크와 세 백본 모두에서 가장 좋은 평균을 기록한다. 단일 항목으로 가장 큰 향상은 background 손상에서 나타나며(예: ModelNet40-C의 DGCNN에서 source-only 49.71 대비 88.57), 절제 실험은 이를 이상치를 고려한 그래프와 연결한다. $$\gamma=0$$이면 ScanObjectNN-C의 background 정확도가 69.54%에서 18.24%로 떨어진다. ScanObjectNN-C의 단순 잡음 손상(uniform, gaussian, impulse, upsampling)에서는 확산 기반 잡음 제거기를 사용하는 CloudFixer가 여전히 평균 8.35% 더 좋다. ScanObjectNN-C에 대한 DGCNN 절제 실험에서 입력 이동을 제거하면 4.55점, 모델 적응을 제거하면 3.48점, 고유맵 기반 의사 레이블을 깊은 특징만 쓰는 의사 레이블로 바꾸면 0.63점이 떨어진다.

## 한계 (Limitations)

- **확장성 (저자들이 명시).** 전역 스펙트럼 연산은 비용이 커서 대규모 포인트 클라우드에 적용하기 어렵다. 저자들은 분할(segmentation)과 다중 스케일 지역 스펙트럼 분석을 향후 과제로 제시한다.
- **샘플마다 고유분해.** 모든 테스트 포인트 클라우드마다 라플라시안 고유분해가 필요하다. $$N\times N$$ 행렬의 완전한 고유분해는 $$O(N^3)$$으로 늘어나며, 고유벡터 일부만 계산하더라도 $$N$$에 따라 증가하는 비용이 추가된다.
- **배치당 많은 최적화 스텝.** 각 배치는 역전파 스텝을 10번 거친다(입력 8번, 모델 2번). 본문에는 실행 시간이나 메모리가 보고되지 않아, TENT나 BN 대비 지연 비용이 분명하지 않다.
- **배치 가정.** 클래스 중심과 다양성 손실은 배치 단위로 계산된다(모든 실험에서 크기 32). 작은 배치나 단일 샘플 스트림에서의 동작은 본문에 나와 있지 않다.
- **어려운 손상은 여전히 어렵다.** Occlusion과 LiDAR는 모든 방법에서 낮게 유지된다. 예를 들어 ModelNet40-C의 DGCNN에서 GSDTTA는 45.38과 31.52이고, ScanObjectNN-C에서는 10% 안팎이다.
- **많은 하이퍼파라미터.** $$k$$, $$\delta$$, $$\gamma$$, $$M$$, $$\alpha$$, $$\beta_1$$, $$\beta_2$$, $$\beta_3$$는 수동으로 고정된다. 민감도는 세 $$\beta$$ 가중치에 대해서만, 하나의 손상(shear)과 하나의 백본에서만 제시된다.

## 내 생각 (My Thoughts)

<!-- TODO -->

---
title: "SVWA: 여러 샘플링에서 적응한 뒤 가중치를 평균하기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
description: 같은 테스트 배치를 FPS/KNN으로 여러 번 다시 샘플링하여 각각 TENT를 실행하고, 적응된 정규화 파라미터를 평균하여 더 강건한 포인트 클라우드 분류기를 얻는다.
conference: WACV
year: 2025
en: sampling-variation-wa-review
date: 2026-10-22 00:00:00 +0900
paper:
  title: "Test-Time Adaptation in Point Clouds: Leveraging Sampling Variation with Weight Averaging"
  authors: "Ali Bahri, Moslem Yazdanpanah, Mehrdad Noori, Sahar Dastani Oghani, Milad Cheraghalikhani, David Osowiechi, Farzad Beizaee, Gustavo A. Vargas Hakim, Ismail Ben Ayed, Christian Desrosiers"
  venue: "WACV 2025"
  url: "https://arxiv.org/abs/2411.01116"
---

## 초록 (Abstract)

포인트 클라우드 분류기는 노이즈, 밀도 변화, 가림(occlusion), LiDAR 방식의 희소성 같은 손상 하에서 정확도를 잃는다. 이 논문이 나올 당시에는 3D 분류를 위한 테스트 시간 적응(test-time adaptation, TTA) 방법이 거의 없었다. MATE는 사전 학습 중 마스크드 오토인코더 목적함수가 필요하므로 실제로는 테스트 시간 학습(test-time training)이다. BFTT3D는 소스 프로토타입 메모리를 유지하므로 완전히 소스 프리하지 않다. TENT는 어떤 사전 학습 모델에도 적용되지만 각 배치의 단일 뷰에서만 적응하므로, 모델을 날카롭고 불안정한 해로 몰아갈 수 있다.

논문은 SWA와 SWAD의 가중치 평균(weight averaging) 아이디어를 빌려온다. 이들에서는 학습 과정을 따라 가중치를 평균하면 더 평탄한 최솟값(flat minima)에 도달하는 경향이 있다. 여기서는 다양성이 학습 단계가 아니라 입력에서 나온다. 같은 포인트 클라우드를 서로 다른 FPS/KNN 샘플링으로 여러 번 토큰화한다. 각 샘플링에 TENT를 실행하고, 그 결과 얻은 정규화 파라미터를 평균한다. 논문은 방법에 이름을 붙이지 않았으며, 이 글에서는 저자들의 코드 저장소 이름을 따라 SVWA라고 부른다.

## 방법 (Method)

![SVWA overview](/images/blog/papers/sampling-variation-wa-main.png)
*입력 포인트 클라우드를 FPS로 여러 번 샘플링하고 KNN으로 패치화한다(패치 a와 b는 샘플링마다 다르다). 정규화 층(NL)만 학습 가능하다. 각 샘플링마다 TENT로 따로 적응시킨 뒤, 적응된 NL 가중치를 평균한다(WA). 논문의 Figure 1.*

### 샘플링 변이

포인트 클라우드는 $$P \in \mathbb{R}^{N^p \times 3}$$이다. 패치 중심은 최원점 샘플링(Farthest Point Sampling, FPS)으로 고른다.

$$
c_i = \arg\max_{p \in P} \min_{c_j \in \{c_1, \dots, c_{i-1}\}} \|p - c_j\|_2 .
$$

각 중심은 $$K$$개의 최근접 이웃으로 이루어진 패치를 가진다.

$$
P_i = \Bigl\{ p \in P \mid p \in \mathrm{KNN}(c_i) \Bigr\},
$$

이로써 $$\mathcal{P} \in \mathbb{R}^{N \times K \times 3}$$를 얻는다. FPS는 시작점에 의존하므로, 다른 중심과 이웃으로 다시 실행하면 같은 물체의 $$N^V$$개 버전 $$\{\mathcal{P}_1, \dots, \mathcal{P}_V\}$$를 얻는다. 각 버전은 지역 패치 기하를 조금씩 바꾸지만 전역 형태는 유지한다. 트랜스포머가 아닌 백본(PointNet, DGCNN)에서는 $$K = 1$$로 두므로, 각 "패치"는 FPS로 선택된 점 하나일 뿐이다.

### 각 변이에 대한 TENT

각 $$\mathcal{P}_v$$에 대해 TENT와 같이 예측 엔트로피 $$\mathcal{H}$$를 최소화하여 정규화 파라미터 $$\gamma, \beta$$만 갱신한다.

$$
\gamma^*_i, \beta^*_i = \arg\min_{\gamma, \beta} \mathbb{E}\left[ \mathcal{H}\left( f_\theta(\mathcal{P}_v) \right) \right].
$$

다른 모든 가중치는 고정된다. Point-MAE에서는 BN과 LN 층을 모두 적응시키며, PointNet과 DGCNN에는 BN만 있다.

### 가중치 평균

각 변이에서 적응한 후, 가중치 $$\theta_v$$를 저장하고 평균한다.

$$
\theta_{\text{avg}} = \frac{1}{N^V} \sum_{v=1}^{V} \theta_v .
$$

동기는 SWAD의 강건 위험 최소화(robust risk minimization) $$\hat{\mathcal{E}}^\gamma_{\mathcal{D}}(\theta) = \max_{\|\Delta\| \le \gamma} \hat{\mathcal{E}}_{\mathcal{D}}(\theta + \Delta)$$에서 나온 평탄한 최솟값 논거이다. 논문은 이 목적함수를 최적화하지는 않는다. 서로 다른 샘플링에서 적응한 모델들을 평균하면 더 평탄한 영역에 도달하고, 개별 적응의 오차가 완화된다고 주장한다.

### 병렬 구현

기본 "병렬(parallel)" 모드에서는 각 변이 전에 모델을 초기화하므로, $$N^V$$개의 적응은 서로 독립적이다. 정규화 파라미터만 바뀌므로 백본은 공유되고, $$(\gamma, \beta)$$의 $$N^V$$개 사본은 하나의 "가중치 평균 정규화 층(weight average normalization layer)" 안에 존재한다. 기본값은 $$N^V = 6$$, TENT 1회 반복, 학습률 $$10^{-3}$$의 AdamW, Point-MAE와 PointNet에서 배치 크기 128이다.

## 주요 실험 (Main Experiments)

Top-1 정확도(%)로, Table 1-3에서 가져온 15개 손상의 평균이다(논문은 모든 손상별 결과도 제시한다). 백본은 Point-MAE, PointNet, DGCNN 세 가지이다. 기준선의 일부만 실었으며, 가장 강한 소스 프리 방법들과 BFTT3D이다. *로 표시된 결과는 저자들이 재현하지 않은 것이고, 나머지는 모두 재현되었다. †는 적응 중 소스 데이터를 사용한다. "-"는 논문에 해당 항목이 없음을 뜻한다.

| 백본 | 방법 | ModelNet40-C | ShapeNet-C | ScanObjectNN-C |
|---|---|---|---|---|
| Point-MAE | Source-Only | 53.9 | 61.3 | 33.1 |
| | MATE* | 64.3 | 63.1 | **47.0** |
| | PL | 71.0 | 66.2 | 45.8 |
| | MEMO | 69.4 | 42.3 | 46.1 |
| | TENT | 72.1 | 66.1 | 45.9 |
| | **Ours** | **75.0** | **67.0** | 46.1 |
| PointNet | Source-Only | 46.5 | 49.8 | 29.7 |
| | PL | 59.6 | 54.3 | 39.5 |
| | BFTT3D† | **66.2** | - | **45.5** |
| | MEMO | 58.5 | 28.0 | 40.2 |
| | TENT | 60.4 | 55.5 | 39.7 |
| | **Ours** | 61.3 | **56.9** | 40.6 |
| DGCNN | Source-Only | 58.8 | 63.6 | 37.9 |
| | PL | 70.7 | **67.0** | 48.1 |
| | BFTT3D† | 70.7 | - | **56.1** |
| | MEMO | 63.5 | 28.7 | 45.0 |
| | TENT | 71.6 | **67.0** | 48.3 |
| | **Ours** | **74.2** | 66.0 | 49.9 |

이 방법은 9개의 백본-데이터셋 조합 중 8개에서 TENT를 앞선다. 가장 뚜렷한 향상은 ModelNet40-C에서 나타난다(Point-MAE에서 +2.9, DGCNN에서 +2.6). 그 외에는 차이가 1포인트 미만인 경우가 많고, DGCNN의 ShapeNet-C에서는 TENT보다 1포인트 낮다. BFTT3D는 PointNet과 DGCNN/ScanObjectNN-C에서 더 높지만, 소스 프로토타입을 사용한다. ablation에서 Point-MAE의 ModelNet40-C 정확도는 $$N^V$$가 2에서 12로 늘어남에 따라 약 74.0%에서 76.0%로 오르며, 샘플링 변이를 뒤집기(flip)나 회전으로 대체하면 성능이 나빠지고 때로는 Source-Only보다 낮아진다.

## 한계 (Limitations)

- 논문에는 한계 섹션이 없다.
- 여전히 역전파가 필요하며, 배치마다 $$N^V$$번의 순전파와 역전파를 실행한다. PointNet에서 $$N^V = 6$$일 때 보고된 적응 시간은 TENT의 21 ms에 비해 약 26 ms이고, 메모리 오버헤드는 약 1.8%이다. 이 수치는 병렬 모드의 PointNet에 대해서만 제시된다.
- 기본 배치 크기가 크다(Point-MAE와 PointNet에서 128). 배치 크기 ablation은 배치 크기 8까지 TENT 대비 향상이 유지됨을 보여주지만, Point-MAE의 ModelNet40-C에서만 확인되었다.
- 많은 설정에서 TENT 대비 향상이 작고 DGCNN의 ShapeNet-C에서는 오히려 음수이므로, 이점이 백본과 데이터셋에 걸쳐 균일하지 않다.
- 평탄한 최솟값에 의한 설명은 유추에 기댄 것이다. 논문은 평탄도를 측정하거나 강건 위험을 최적화하지 않는다.
- 샘플링 변이는 FPS/KNN 토큰화에 의존한다. PointNet과 DGCNN에서는 서로 다른 FPS 부분집합을 고르는 것으로 축소된다.

## 내 생각 (My Thoughts)

<!-- TODO -->

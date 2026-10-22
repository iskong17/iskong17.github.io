---
title: "3DD-TTA: 테스트 포인트 클라우드를 잠재 공간에서 디노이징하기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
description: 3DD-TTA는 손상된 포인트 클라우드를 LION의 VAE로 인코딩하고, 잠재 점(latent point)에 약간의 노이즈를 더한 뒤 Chamfer 가이던스 아래 소스 확산 모델로 디노이징한다. 분류기는 전혀 건드리지 않는다.
conference: WACV
year: 2025
paper:
  title: "Test-Time Adaptation of 3D Point Clouds via Denoising Diffusion Models"
  authors: "Hamidreza Dastmalchi, Aijun An, Ali Cheraghian, Shafin Rahman, Sameera Ramasinghe"
  venue: "WACV 2025"
  url: "https://arxiv.org/abs/2411.14495"
en: 3dd-tta-review
date: 2026-10-22 00:00:00 +0900
---

## 초록 (Abstract)

테스트 시점의 포인트 클라우드는 LiDAR 센서 고장이나 환경 요인 등으로 손상되어 있는 경우가 많다. 대부분의 테스트 시간 적응(test-time adaptation, TTA) 방법은 의사 레이블(pseudo-label)이나 자기 지도(self-supervised) 손실로 소스 모델을 미세 조정한다. 테스트 스트림이 길어지면 이러한 업데이트가 오류를 누적시키고, 모델이 유용한 소스 지식을 잊게 만들 수 있다. 2D에서는 모델을 그대로 두고 확산 모델(diffusion model)로 테스트 이미지를 소스 도메인으로 되돌리는 또 다른 연구 흐름이 있다. 저자들은 3D 포인트 클라우드에서는 이것이 시도된 적이 없다고 말한다. 포인트 클라우드는 비정형 데이터이고, 디노이징 과정에서 원본에 대한 충실도를 유지하기 어렵기 때문이다.

3DD-TTA는 이러한 입력 적응(input adaptation) 아이디어를 3D로 가져온다. 포인트 클라우드용 잠재 확산 모델인 LION을 사용한다. 테스트 포인트 클라우드는 형상 잠재 변수(shape latent)와 잠재 점(latent point) 집합으로 인코딩된다. 잠재 점에 소량의 노이즈를 더한 뒤 몇 단계의 DDIM으로 디노이징한다. 디노이징 중에는 Selective Chamfer Distance가 잠재 점이 원본에서 멀어지지 않도록 붙잡아 둔다. 정제된 잠재 변수는 디코딩되어 변경되지 않은 분류기에 입력된다. 어떤 네트워크 가중치도 업데이트되지 않는다.

## 방법 (Method)

![3DD-TTA overview](/images/blog/papers/3dd-tta-main.png)
*손상된 입력 $$\tilde{\mathbf{x}}$$는 형상 잠재 변수 $$\mathbf{z}_0$$와 잠재 점 $$\mathbf{h}_0$$로 인코딩된다. 잠재 점은 단계 $$t_w$$까지 노이즈가 더해진 뒤, $$\mathbf{h}_t$$와 $$\mathbf{z}_0$$ 모두에 SCD 가이던스를 걸어 $$t_w$$ 단계 동안 디노이징된다. 결과는 디코딩되어 $$p_c$$로 분류된다. 그림은 논문에서 가져왔다.*

### 백본: LION

LION은 확산 사전 분포(diffusion prior)를 가진 계층적 VAE이다. 인코더는 두 개다.

- $$q_z(\mathbf{z}_0 \mid \mathbf{x})$$는 전역 **형상 잠재 변수** $$\mathbf{z}_0 \in \mathbb{R}^{D_z}$$를 만든다.
- $$q_h(\mathbf{h}_0 \mid \mathbf{z}_0, \mathbf{x})$$는 **잠재 점** $$\mathbf{h}_0 \in \mathbb{R}^{4 \times n}$$를 만든다. 이는 점마다 $$xyz$$ 좌표와 추가 특징 하나로 이루어진다.

디노이저 $$\boldsymbol{\epsilon}_h(\mathbf{h}_t, \mathbf{z}_0, t)$$는 형상 잠재 변수를 조건으로 잠재 점을 모델링한다. 디코더 $$p_d(\mathbf{x} \mid \mathbf{z}_0, \mathbf{h}_0)$$는 잠재 변수를 다시 포인트 클라우드로 변환한다. 전체 모델은 소스 도메인인 ShapeNet에서 사전 학습된다. 분류기 $$p_c$$(Point-MAE)와 확산 모델은 모두 고정된다.

### 1단계: 인코딩과 섭동 (그림의 왼쪽 절반)

손상된 입력을 인코딩한다.

$$
\mathbf{z}_0 \sim q_z(\mathbf{z}_0 \mid \tilde{\mathbf{x}}), \qquad \mathbf{h}_0 \sim q_h(\mathbf{h}_0 \mid \mathbf{z}_0, \tilde{\mathbf{x}})
$$

그다음 잠재 점에 순방향으로 단계 $$t_w$$까지 노이즈를 더한다.

$$
\mathbf{h}_{t_w} = \sqrt{\bar\alpha_{t_w}}\, \mathbf{h}_0 + \sqrt{1 - \bar\alpha_{t_w}}\, \boldsymbol{\epsilon}
$$

저자들은 이 노이즈가 입력의 특정 손상 유형에 대한 의존성을 줄인다고 주장한다. 논문의 Figure 2는 섭동 이후 깨끗한 형상까지의 Chamfer 거리가 균일 노이즈, 가우시안 노이즈, RBF 손상에서 거의 같은 분포를 보임을 보여 준다. 따라서 모델은 자신이 어떤 손상을 마주했는지 알 필요가 없다.

### 2단계: 가이던스가 있는 DDIM 디노이징 (가운데 루프)

$$t = t_w, \dots, 1$$에 대해 디노이저는 노이즈 $$\boldsymbol{\epsilon}_t = \boldsymbol{\epsilon}_h(\mathbf{h}_t, \mathbf{z}_0, t)$$를 예측한다. 이로부터 깨끗한 잠재 점의 추정치를 얻는다.

$$
\bar{\mathbf{h}}_0 = \frac{\mathbf{h}_t - \sqrt{1 - \bar\alpha_t}\, \boldsymbol{\epsilon}_t}{\sqrt{\bar\alpha_t}}
$$

결정론적 DDIM 단계에 가이던스 항을 더해 다음 잠재 변수를 얻는다.

$$
\mathbf{h}_{t-1} = \sqrt{\bar\alpha_{t-1}}\, \bar{\mathbf{h}}_0 + \sqrt{1 - \bar\alpha_{t-1}}\, \boldsymbol{\epsilon}_t - \eta \nabla_{\mathbf{h}_t} l^{\lambda}_{cd}
$$

가이던스가 없으면 디노이저는 그럴듯하지만 다른 형상으로 흘러갈 수 있다. 가이던스는 $$\bar{\mathbf{h}}_0$$를 원래 잠재 점 $$\mathbf{h}_0$$에 가깝게 유지한다.

**Selective Chamfer Distance (SCD).** 원래의 $$\mathbf{h}_0$$ 자체가 손상되어 있으므로, 이를 정확히 맞추면 손상이 다시 들어온다. SCD는 각 방향에서 최근접 이웃 거리 중 가장 가까운 $$\lambda$$ 비율만 남기고, 나머지는 이상치(outlier)일 가능성이 높다고 보고 버린다.

$$
l^{\lambda}_{cd}(\mathbf{h}_0, \bar{\mathbf{h}}_0) = \frac{1}{|\mathbf{h}_0|} \sum \mathrm{lower}_\lambda\big(\mathrm{sort}(D_{\mathbf{h}_0})\big) + \frac{1}{|\bar{\mathbf{h}}_0|} \sum \mathrm{lower}_\lambda\big(\mathrm{sort}(D_{\bar{\mathbf{h}}_0})\big)
$$

$$D_{\mathbf{h}_0}$$는 $$\mathbf{h}_0$$의 각 점에서 $$\bar{\mathbf{h}}_0$$의 가장 가까운 점까지의 제곱 거리 집합이다. $$D_{\bar{\mathbf{h}}_0}$$는 반대 방향으로 같은 것을 구한 것이다. 논문은 $$\lambda = 0.96$$을 사용한다.

**형상 잠재 변수 업데이트.** 손상된 입력에서 얻은 형상 잠재 변수는 디노이저에 부적절한 조건을 줄 수 있다. 그래서 형상 잠재 변수도 매 단계 SCD 그래디언트 방향으로 이동시킨다(위쪽 점선 화살표).

$$
\mathbf{z}_0 \leftarrow \mathbf{z}_0 - \gamma \nabla_{\mathbf{z}_0} l^{\lambda}_{cd}
$$

### 3단계: 디코딩과 분류 (오른쪽)

$$t_w$$ 단계가 끝나면 교정된 잠재 점 $$\mathbf{h}_0^r$$와 업데이트된 $$\mathbf{z}_0$$를 $$\mathbf{x}$$로 디코딩한다. 그다음 고정된 분류기가 $$c \sim p_c(c \mid \mathbf{x})$$를 예측한다.

설정: 전체 100단계의 DDIM을 사용하므로 각 단계는 LION의 1000개 학습 단계 중 10개를 건너뛴다. 입력에는 단계 $$t_w = 5$$까지 노이즈를 더하고 5단계 동안 디노이징하며, $$\gamma = \eta = 0.01$$이다.

## 주요 실험 (Main Experiments)

소스 분류기는 MATE에서 학습한 Point-MAE이다. 평가 지표는 15가지 손상에 대한 평균 정확도(%)이다. LION은 ShapeNet에서만 사전 학습되었으므로, ModelNet40-c와 ScanObjectNN-c는 데이터셋 간 일반화(cross-dataset generalization)도 함께 시험한다. ScanObjectNN-c에서는 소스 모델과만 비교했다.

| 방법 | ShapeNet-c | ModelNet40-c | ScanObjectNN-c |
|---|---|---|---|
| Point-MAE (src) | 59.3 | 57.6 | 61.5 |
| DUA | 60.8 | 54.7 | - |
| TTT-Rot | 60.9 | 53.0 | - |
| SHOT | 36.2 | 26.6 | - |
| T3A | 54.4 | 55.7 | - |
| TENT | 37.4 | 26.5 | - |
| MATE-S | 63.1 | 64.3 | - |
| **3DD-TTA** | **69.8** | **65.7** | **64.3** |

ShapeNet-c의 일부 손상 유형 (Table 1의 일부, 정확도 %):

| 방법 | gauss | back | impu | shear | rot | occ | 평균 |
|---|---|---|---|---|---|---|---|
| Point-MAE (src) | 66.4 | 15.0 | 60.6 | 74.1 | 42.8 | 8.4 | 59.3 |
| DUA | 70.1 | 14.3 | 60.9 | 77.1 | 57.5 | 11.9 | 60.8 |
| MATE-S | 74.7 | 4.3 | 66.2 | 79.2 | 56.1 | 12.3 | 63.1 |
| 3DD-TTA | 80.7 | 77.6 | 77.2 | 77.2 | 50.4 | 14.9 | 69.8 |

정리:
- 성능 향상은 주로 노이즈 계열 손상에서 나온다. ShapeNet-c의 배경 노이즈(background noise)는 15.0에서 77.6으로 오른다. 이 항목에서 가장 좋은 베이스라인인 TTT-Rot도 23.1에 그친다.
- 변환 계열 손상이 약점이다. ShapeNet-c에서 회전(50.4)은 DUA와 MATE-S에 뒤진다. ModelNet40-c에서 전단(shear, 63.7)과 회전(33.4)은 소스 모델과 비슷하거나 그보다 낮다.
- 적응이 빠르다. 5단계 디노이징은 A6000에서 샘플당 40 ms가 걸리고(Table 4), 시간은 단계 수에 비례해 40단계에서 290 ms까지 늘어난다.

## 한계 (Limitations)

- **회전과 전단** (저자들이 직접 언급). 학습이 필요 없는 디노이저는 전역 기하 변환을 잘 되돌리지 못한다. ModelNet40-c에서 전단은 소스 모델보다 낮아진다(63.7 vs 67.6).
- **고정된 단계 수** (저자들이 직접 언급). 대부분의 손상에는 5단계면 충분하지만, 배경 노이즈는 최대 35단계가 필요해 시간이 더 걸린다. 입력마다 단계 수를 정하는 문제는 향후 과제로 남겨 두었다.
- **대규모 사전 학습 생성 모델이 필요하다.** LION(VAE와 잠재 확산 사전 분포)을 소스 도메인에서 학습해야 한다. 여기서는 ShapeNet이며, 같은 모델을 다른 데이터셋에도 재사용한다.
- **디노이저를 통한 그래디언트.** $$\bar{\mathbf{h}}_0$$는 $$\boldsymbol{\epsilon}_h$$를 통해 $$\mathbf{h}_t$$와 $$\mathbf{z}_0$$에 의존한다. 따라서 두 SCD 업데이트 모두 매 단계 순전파뿐 아니라 디노이징 네트워크를 통한 역전파가 필요하다.
- **제한적인 베이스라인.** 베이스라인은 대부분 2D TTA 방법과 MATE이다. DDA를 3D로 옮긴 것과 같은 다른 확산 기반 입력 적응 방법과의 비교는 없다. ScanObjectNN-c에서는 소스 모델과의 비교만 있다.
- **가림(occlusion)과 LiDAR는 여전히 어렵다.** ShapeNet-c에서 가림과 LiDAR는 각각 14.9와 14.2에 머문다.

## 내 생각 (My Thoughts)

<!-- TODO -->

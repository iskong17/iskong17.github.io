---
title: "LN-TTA: LayerNorm은 그대로 두고 아핀만 살짝 조정하기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
en: layernorm-pc-tta-review
date: 2026-10-22 00:00:00 +0900
description: 포인트 클라우드 Transformer를 위한 순전파 전용, 소스 데이터 없는 테스트 시간 적응으로, LayerNorm의 정규화는 그대로 유지하고 [cls] 토큰에서 얻은 프로토타입을 이용해 아핀 이후 출력의 스케일과 이동만 조정한다.
conference: WACV
year: 2026
paper:
  title: "Revisiting Layer Normalization for Point Cloud Test Time Adaptation"
  authors: "Moslem Yazdanpanah, Ali Bahri, Mehrdad Noori, Sahar Dastani, Samuel Barbeau, David Osowiechi, Gustavo Adolfo Vargas Hakim, Ismail Ben Ayed, Christian Desrosiers"
  venue: "WACV 2026"
  url: "https://openaccess.thecvf.com/content/WACV2026/html/Yazdanpanah_Revisiting_Layer_Normalization_for_Point_Cloud_Test_Time_Adaptation_WACV_2026_paper.html"
---

## 초록 (Abstract)

대부분의 테스트 시간 적응(test-time adaptation, TTA) 기법은 Batch Normalization(BN)을 사용하는 CNN에서 비롯되었다. 이들은 배치 통계를 다시 추정하거나 엔트로피 최소화로 BN의 아핀 파라미터를 갱신한다. Point-MAE 같은 포인트 클라우드 분류기는 Transformer이고, Transformer는 Layer Normalization(LN)을 사용한다. LN은 각 토큰을 특징 축 방향으로 정규화하고 배치 축은 전혀 보지 않으므로, BN 방식의 해결책이 깔끔하게 이전되지 않는다. 기존 포인트 클라우드 TTA 방법들에는 실용적인 비용도 있다. TENT는 역전파가 필요하고, MATE는 특수한 사전학습 목적 함수가 필요하며, BFTT3D는 소스 프로토타입이 필요하고, 다중 샘플링 방법은 많은 추가 패스가 필요하다.

이 논문은 배치(도메인) 관점에서 LN을 분석한다. 특징 차원과 배치 크기가 클 때 LN의 아핀 이전 출력이 이미 배치에 걸쳐 특징별로 평균 0, 분산 1에 가깝다는 것을 보인다. 따라서 다시 표준화할 필요가 없다. 조정할 가치가 있는 것은 아핀 이후의 평균과 이득(gain)뿐이다. LN-TTA는 사전학습된 [cls] 토큰에서 읽은 프로토타입을 사용해, 한 번의 순전파로 그래디언트도 소스 데이터도 없이 이를 수행한다.

## 방법 (Method)

![LN-TTA overview](/images/blog/papers/layernorm-pc-tta-main.png)
*(a) 사전학습 동안 [cls] 토큰은 데이터 토큰들의 정보를 모은다. (b) 오프라인에서 [cls] 토큰만으로 순전파를 한 번 하여 모든 LN 층의 프로토타입 $$y_{\text{proto}}$$를 기록한다. (c) 테스트 시간에는 저장된 프로토타입을 사용해 각 LN 출력을 Eq. 16으로 다시 채색(recolour)한다. 논문의 Figure 2.*

### 배경: 투영과 재스케일링으로서의 LN

토큰 $$x_t \in \mathbb{R}^d$$에 대해 LN은 다음과 같다.

$$
\mathrm{LN}(x_t) = \gamma \odot \frac{x_t - \mu_d(x_t)\mathbf{1}}{\sqrt{\sigma_d^2(x_t) + \varepsilon}} + \beta .
$$

논문은 이를 아핀 이전 사상과 특징별 아핀으로 나눈다. $$u = \mathbf{1}/\sqrt{d}$$와 (토큰 평균을 제거하는) $$C = I - uu^\top$$를 두면,

$$
\widehat{\mathrm{LN}}(x_t) = \sqrt{d}\,\frac{C x_t}{\sqrt{\|C x_t\|^2 + d\varepsilon}}, \qquad
\mathrm{LN}(x_t) = \Gamma\,\widehat{\mathrm{LN}}(x_t) + \beta, \quad \Gamma = \mathrm{diag}(\gamma).
$$

즉 LN은 먼저 평균이 0인 초평면 $$H$$로 투영하고, 반지름을 정규화한 다음, $$(\gamma, \beta)$$를 적용한다. 아핀 이전 부분은 모든 특징에 상수를 더하는 것에 불변이며, ($$\varepsilon = 0$$일 때) 전역 재스케일링에도 불변이다.

### 이론: LN은 이미 배치에 걸쳐 표준화한다

토큰별로 $$\widehat{\mathrm{LN}}$$은 특징에 걸쳐 정확히 평균 0, 2차 모멘트 1을 준다(Proposition 1, $$\varepsilon = 0$$). 핵심 결과는 배치 축에 관한 것이다. 토큰 $$t$$와 특징 $$j$$를 고정하고 $$H_{b,t} = \widehat{\mathrm{LN}}(x_{b,t})$$라 하자. Theorem 1은 $$d \to \infty$$이고 $$B = B(d) \to \infty$$일 때 다음이 성립한다고 말한다.

$$
\bar H_{j,B} \xrightarrow{P} 0, \qquad S^2_{j,B} \xrightarrow{P} 1,
$$

여기서 $$\bar H_{j,B}$$와 $$S^2_{j,B}$$는 특징 $$j$$의 배치 평균과 분산이다. 아핀 이후에는 특징별 배치 극한이 $$(\beta_j, \gamma_j^2)$$가 된다. 말하자면 LN은 BN의 일을 암묵적으로 수행하며, 아핀 이후의 평균과 스케일은 오직 $$(\gamma, \beta)$$로만 제어된다. 설계 원칙은 아핀 이전 단계는 유지하고 아핀만 조정하는 것이다.

### (a)-(b) [cls] 토큰에서 얻는 프로토타입

소스의 아핀 이후 모멘트를 알 수 있다면 특징별로 이를 맞출 수 있을 것이다(논문의 Eq. 13-14). 소스 데이터 없는(source-free) 설정에서는 이를 알 수 없다. 저자들은 [cls] 토큰이 사전학습 동안 도메인 정보를 모은다는 점에 주목한다. 그래서 모델을 고정한 채 [cls] 토큰만으로 순전파를 한 번 하고(길이 1인 시퀀스에 대한 어텐션은 $$V$$에 대한 항등 사상이다), 각 LN 층에서 다음을 기록한다.

$$
y_{\text{proto}} = \mathrm{LN}([\text{cls}]_{\text{emb}}), \quad
m^{\text{proto}}_\star = \bar m(y_{\text{proto}}), \quad
s^{\text{proto}}_\star = \frac{\|y_{\text{proto}} - m^{\text{proto}}_\star \mathbf{1}\|}{\sqrt{d}} .
$$

층마다 이 두 스칼라가 소스의 평균과 스케일을 대신한다. 이 작업은 한 번만 수행되며 소스나 타깃 샘플이 필요 없다.

### (c) 테스트 시간 재채색

테스트 시간에는 타깃 토큰의 모든 LN 출력을, 공통 모드(common mode)와 전역 이득만 바꾸는 고정된 아핀으로 사상한다.

$$
y^{(T)}_{\text{adapt},b,t} = \frac{s^{\text{proto}}_\star}{\bar s(\gamma^S_0)}\, y^{(T)}_{b,t} + \left( m^{\text{proto}}_\star - \frac{s^{\text{proto}}_\star}{\bar s(\gamma^S_0)}\, \bar m(\beta^S_0) \right)\mathbf{1},
$$

여기서 $$\bar m(\beta^S_0) = d^{-1}\mathbf{1}^\top \beta^S_0$$, $$\bar s(\gamma^S_0) = (d^{-1}\|\gamma^S_0\|_2^2)^{1/2}$$이다. 근거는 타깃 LN 출력의 특징 방향 평균과 스케일이 $$\bar m(\beta^S_0)$$와 $$\bar s(\gamma^S_0)$$ 주변에 집중된다는 것이며, 따라서 이는 "표준화한 뒤 프로토타입으로 다시 채색하는" 단계이다. 동등하게, 각 LN 층의 파라미터를 다음으로 교체하는 것이다.

$$
\gamma'^T = \frac{s^{\text{proto}}_\star}{\bar s(\gamma^S_0)}\,\gamma^S_0, \qquad
\beta'^T = \frac{s^{\text{proto}}_\star}{\bar s(\gamma^S_0)}\,\beta^S_0 + \left( m^{\text{proto}}_\star - \frac{s^{\text{proto}}_\star}{\bar s(\gamma^S_0)}\,\bar m(\beta^S_0) \right).
$$

특징 간 상대적 이득은 변하지 않으므로 $$H$$ 내부의 기하 구조는 보존된다. 이 절차는 상태가 없으며(stateless) 토큰당 $$O(d)$$의 연산만 추가한다. 아무것도 갱신되지 않으므로 "standard"(에피소드 방식) 예측과 "online" 예측은 동일하다.

## 주요 실험 (Main Experiments)

백본: Point-MAE(MATE 공개 버전). 지표: 15개 손상에 대한 평균 top-1 정확도(%). 배치 크기 32. 논문 표는 모든 손상을 나열하지만, 여기서는 "Mean" 열만 보여준다. *로 표시된 결과는 저자들이 재현한 것이고, 나머지는 MATE에서 가져온 것이다.

| 방법 | 유형 | ScanObjectNN-C | ModelNet40-C | ShapeNet-C |
|---|---|---|---|---|
| Source Only | - | 36.96 | 54.16 | 61.35 |
| MATE | TTT, 역전파 | 36.90 | 58.90 | 63.10 |
| SMART-PC | TTT, 역전파 | 39.60 | 63.10 | **64.40** |
| T3A | TTA | 42.46 | 57.22 | 54.18 |
| TENT* | TTA, 역전파 | 43.45 | 63.53 | 54.98 |
| AdaBN | 역전파 없음 | 42.69 | 67.91 | 60.44 |
| DUA | 역전파 없음 | 38.66 | 54.73 | 60.77 |
| BFTT3D* | 역전파 없음, 소스 프로토타입 | 33.00 | 57.16 | 60.73 |
| **LN-TTA** | 역전파 없음, 소스 데이터 없음 | **49.31** | **69.74** | 64.38 |
| TENT* (online) | TTA, 역전파 | 45.70 | 64.22 | 63.00 |

초당 처리 포인트 클라우드 수(PPS)로 측정한 처리량(같은 표):

| 방법 | ScanObjectNN-C | ModelNet40-C | ShapeNet-C |
|---|---|---|---|
| Source Only | 88.78 | 112.41 | 93.89 |
| TENT* | 40.72 | 69.17 | 56.90 |
| BFTT3D* | 1.67 | 2.20 | 1.90 |
| LN-TTA | 64.24 | 93.03 | 73.12 |

LN-TTA는 ScanObjectNN-C와 ModelNet40-C에서 가장 좋은 평균을 기록하며, Source Only 대비 +12.35점과 +15.58점 향상되고, standard와 online 두 모드 모두에서 역전파 기반 TENT를 앞선다. ShapeNet-C에서는 향상이 +3.03에 그치며, 사전학습을 바꾸는 TTT 방법인 SMART-PC가 0.02점 더 높다(64.40 대 64.38). 그곳에서 LN-TTA는 역전파 없는 방법 중에서 가장 좋다. 속도는 Source Only의 상당 부분 수준이며, 배치 크기 32에서 최대 GPU 메모리는 1.468 GB로 TENT의 2.890 GB, BFTT3D의 7.443 GB보다 작다.

## 한계 (Limitations)

- 저자들은 성능이 여전히 배치 크기에 의존하며, 다른 모달리티와 백본으로의 확장은 향후 과제라고 밝힌다. 보충 자료는 ShapeNet-C에서 배치 크기가 커질수록 정확도가 오르는 것을 보여준다.
- 모든 실험은 단일 백본(Point-MAE)을 사용한다. 이 방법은 LN 층과 [cls] 토큰이 필요하므로, PointNet이나 DGCNN 같은 BN 전용 포인트 네트워크에는 적용되지 않는다.
- Eq. 16의 재채색은 프로토타입과 소스 LN 파라미터만 사용한다. 모든 타깃 도메인에 대해 같은 고정된 재파라미터화이며, 테스트 배치의 통계는 전혀 읽지 않는다.
- 보충 자료에 따르면 설정은 TENT를 따라 BN 통계를 초기화한다. AdaBN만으로도 ModelNet40-C에서 이미 67.91에 도달한다. 논문에는 BN 초기화와 LN 편집을 분리하는 절제 실험이 없어, 향상의 얼마가 각 부분에서 오는지 불분명하다.
- 이론은 점근적이다($$d, B \to \infty$$, $$\varepsilon = 0$$, i.i.d. 샘플). [cls] 전용 순전파를 소스 모멘트의 대리값으로 쓰는 것은 동기가 제시되었을 뿐 증명되지 않았다.
- Occlusion과 LiDAR는 매우 낮게 유지되며(ScanObjectNN-C에서 7.40과 9.81), ShapeNet-C에서는 이 두 손상에서 DUA가 더 좋다.

## 내 생각 (My Thoughts)

<!-- TODO -->

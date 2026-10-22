---
title: "FPP: 평탄한 지점에서 테스트 시간 프롬프트 튜닝을 시작하기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
en: fpp-review
date: 2026-10-22 00:00:00 +0900
description: FPP는 C-TPT와 O-TPT가 선명도 인식 최소화(SAM)처럼 작동하기 때문에 TPT를 보정한다고 주장하고, 그 정규화 항을 초기 프롬프트를 손실 지형의 평탄한 영역에 두는 데이터 없는 사전학습 단계로 대체한다.
conference: CVPR
year: 2026
paper:
  title: "Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining"
  authors: "Hyeonseo Jang, Jaebyeong Jeon, Joong-Won Hwang, Kibok Lee"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2604.27715"
---

## 초록 (Abstract)

테스트 시간 프롬프트 튜닝(test-time prompt tuning, TPT)은 엔트로피 최소화(entropy minimisation, EM) 손실로 CLIP의 텍스트 프롬프트를 각 테스트 이미지에 맞게 조정한다. 정확도는 높아지지만 예측이 과신하게 된다. C-TPT와 O-TPT는 클래스 텍스트 특징을 서로 퍼뜨리는 정규화 항을 더한다. 이는 보정 오차를 낮추지만 정확도를 희생하는 경우가 많고, 특징을 퍼뜨리는 것이 왜 보정에 도움이 되는지 설명하지 못한다.

이 논문은 하나의 설명을 제시한다. TPT는 테스트 샘플마다 프롬프트를 초기화하고 한 번의 스텝만 밟기 때문에, 정규화 항의 그래디언트는 고정된 오프셋이 된다. 그러면 정규화된 TPT는 선명도 인식 최소화(sharpness-aware minimisation, SAM)처럼 보인다. 즉 EM 그래디언트를 섭동된 한 지점에서 계산하고 다른 지점에 적용하는 것이다. 저자들은 더 평탄한 최솟값이 더 낮은 보정 오차와 함께 나타남을 보인다. 이들의 방법인 평탄성 인식 프롬프트 사전학습(Flatness-aware Prompt Pretraining, FPP)은 정규화 항을 버리고, 대신 이미지 없이 초기 프롬프트를 사전학습하여 처음부터 평탄한 영역에 놓이게 한다. 기존 TPT 파이프라인은 초기 프롬프트만 교체하면 된다.

## 방법 (Method)

![FPP overview](/images/blog/papers/fpp-main.png)
*위: 일반 TPT는 zero-shot 프롬프트에서 EM 스텝을 한 번 밟아 날카로운 최솟값에 도달한다. 가운데: 정규화된 TPT는 이동된 프롬프트에서 시작해 섭동된 지점에서 EM 그래디언트를 계산하는 것과 동등하며, 더 평탄한 최솟값으로 이어진다. 아래: FPP는 초기 프롬프트를 더 평탄한 영역 안에서 사전학습하므로, 일반 EM 스텝만으로도 훨씬 더 평탄한 최솟값에 도달한다. 그림은 논문에서 가져왔다.*

### SAM으로서의 정규화된 TPT (위와 가운데 행)

TPT는 확신도가 높은 증강 뷰 $$V$$에 대해 평균한 예측의 엔트로피 $$L_{\text{ent}}(\theta, V)$$로 프롬프트 $$\theta$$를 최적화한다. C-TPT와 O-TPT는 $$\lambda L_{\text{reg}}(\theta)$$를 더하는데, 여기서 $$L_{\text{reg}}$$는 고정된 클래스 이름 $$C$$의 텍스트 특징 $$f_T(C;\theta)$$에만 의존한다.

$$
L_{\text{reg}}^{\text{C-TPT}} = -\frac{1}{K}\sum_{k=1}^{K}\|t_k - \mu\|_2, \qquad
L_{\text{reg}}^{\text{O-TPT}} = \|TT^\top - I_K\|_2^2 .
$$

모든 테스트 샘플은 같은 프롬프트 $$\theta_0^{\text{zs}}$$에서 시작하므로, $$\varepsilon_{\text{reg}} = \lambda \nabla_\theta L_{\text{reg}}\big|_{\theta_0^{\text{zs}}}$$는 모든 샘플에 대해 같은 상수이다. 학습률이 1일 때 한 스텝은 다음과 같다.

$$
\theta_1^{\text{reg}} = \theta_0^{\text{zs}} - \nabla_\theta L_{\text{ent}}(\theta, V)\big|_{\theta_0^{\text{zs}}} - \varepsilon_{\text{reg}}
= \theta_0^{\text{reg}} - \nabla_\theta L_{\text{ent}}(\theta, V)\big|_{\theta_0^{\text{reg}} + \varepsilon_{\text{reg}}},
$$

여기서 $$\theta_0^{\text{reg}} := \theta_0^{\text{zs}} - \varepsilon_{\text{reg}}$$이다. 따라서 정규화된 TPT는 이동된 시작점에서 출발하고 섭동된 지점에서 그래디언트를 취하는 일반 TPT이다. Theorem 1은 두 손실을 연결한다. 이미지 특징이 단위 구 위에서 균등하게 뽑힐 때, EM 손실의 기댓값은 $$O(D^{-3/2})$$ 오차 내에서 $$L_{\text{reg}}$$의 아핀 함수이다. 따라서 섭동된 지점은 EM 손실이 더 높은 경향이 있으며, 이는 SAM의 갱신 $$w_{t+1} = w_t - \eta \nabla_w L(w, v)\big|_{w_t + \hat{\varepsilon}_t}$$와 같은 패턴이다. 실험적으로 C-TPT와 O-TPT는 TPT보다 낮은 선명도(sharpness)에 도달하며, 프롬프트를 선명도에 따라 그룹으로 나누면 더 평탄한 그룹일수록 ECE가 낮다.

### 평탄성 인식 프롬프트 사전학습 (아래 행)

FPP는 평탄성 탐색을 테스트 시간 이전으로 옮긴다. 두 손실로 새로운 초기 프롬프트 $$\theta$$를 학습한다. 평탄성 손실은 클래스 이름 임베딩과 프롬프트를 작은 가우시안 잡음 $$\varepsilon_1, \varepsilon_2$$로 섭동해도 텍스트 특징이 그대로 유지되도록 요구한다.

$$
\mathcal{L}_{\text{flat}} = \operatorname{dist}_{\cos}\big(f_T(C + \varepsilon_1;\, \theta + \varepsilon_2),\; f_T(C;\, \theta)\big).
$$

출력 민감도가 낮으면 텍스트 특징 위에 놓인 임의의 미분 가능한 손실에 대해 선명도가 낮아진다(부록에서 증명). 이 손실만 쓰면 zero-shot 특징이 망가지므로, 정렬 손실이 새 텍스트 특징을 원래 프롬프트의 특징 가까이에 유지시킨다.

$$
\mathcal{L}_{\text{align}} = \operatorname{dist}_{L2}\big(f_T(C;\, \theta),\; f_T(C;\, \theta_0^{\text{zs}})\big), \qquad
\mathcal{L}_{\text{FPP}} = \mathcal{L}_{\text{align}} + \lambda\, \mathcal{L}_{\text{flat}} .
$$

가중치는 $$\lambda = \gamma_1 + \gamma_2 / K$$이며 $$\gamma_1 = 1.0$$, $$\gamma_2 = 0.15$$이다. 이는 클래스 수가 많을 때 평탄성 가중치를 낮춘다. 사전학습은 lr 0.01의 AdamW로 1K 반복 실행하며, 잡음 분산은 $$\varepsilon_1$$에 0.02, $$\varepsilon_2$$에 0.005이다. 두 손실 모두 프롬프트와 클래스 이름만 사용하므로 이미지가 필요 없다. 테스트 시간의 TPT 절차는 바뀌지 않는다. $$\theta_0^{\text{zs}}$$만 사전학습된 프롬프트로 교체되며, 정규화 항은 더해지지 않는다.

## 주요 실험 (Main Experiments)

설정: CLIP ViT-B/16, 하드 프롬프트 "a photo of a", C-TPT 및 O-TPT와 동일한 표준 TPT 테스트 시간 설정. 지표는 top-1 정확도(%), ECE(%), SCE(static calibration error)이다. 여기서 FPP는 FPP 프롬프트에서 시작한 TPT를 뜻한다. 표는 논문 Table 1(10개 세분화 데이터셋: Aircraft, Caltech, Cars, DTD, EuroSAT, Flowers, Food, Pets, SUN, UCF)과 Table 4(ImageNet-A, -V2, -R, -Sketch에 대한 OOD 평균)의 평균 열, 그리고 Table 4의 ImageNet 열만 보여준다.

| 방법 | 세분화 Acc | ECE | SCE | ImageNet Acc | ECE | OOD Acc | ECE | SCE |
|---|---|---|---|---|---|---|---|---|
| CLIP | 63.41 | 4.67 | 1.06 | 66.7 | 2.12 | 57.18 | 5.04 | 0.15 |
| TPT | 64.62 | 11.67 | 1.15 | 69.0 | 10.6 | 59.95 | 11.99 | 0.14 |
| C-TPT | 64.48 | 5.32 | 1.11 | 68.5 | 3.15 | 59.55 | 5.82 | 0.14 |
| O-TPT | 64.12 | 4.46 | 1.15 | 67.3 | 1.97 | 57.82 | 4.88 | 0.15 |
| FPP | 65.37 | 4.13 | 0.96 | 67.8 | 2.97 | 59.53 | 4.63 | 0.12 |

공개된 CoOp 및 MaPLe 체크포인트로 프롬프트를 초기화한 경우(논문 Table 2, 세분화 평균):

| 방법 | Acc | ECE |
|---|---|---|
| CoOp+TPT | 63.91 | 18.36 |
| CoOp+C-TPT | 63.56 | 10.04 |
| CoOp+O-TPT | 63.29 | 7.91 |
| CoOp+FPP | 64.78 | 5.67 |
| MaPLe+TPT | 65.75 | 6.94 |
| MaPLe+O-TPT | 65.05 | 4.11 |
| MaPLe+FPP | 67.01 | 4.07 |

세분화 평균에서 FPP는 정확도로 TPT를 앞서면서 동시에 가장 낮은 ECE와 SCE를 갖는 유일한 방법이다. OOD 데이터셋에서는 TPT 정확도의 대부분을 유지하면서(59.53 대 59.95) O-TPT보다 낮은 ECE를 보이지만, 분포 변화 실험에서는 고정된 $$\lambda = 1.25$$를 사용한다. 절제 실험(ablation)은 두 부분이 모두 필요함을 보여준다. 정렬 손실만 쓰면 ECE가 7.05이고, 평탄성 손실만 쓰면 zero-shot 정확도가 4.65%로 무너지며, 전체 손실은 4.13을 준다. TPT 안에서 SAM이나 ASAM을 직접 적용해도 ECE가 낮아지지만, SAM은 정확도를 61.59로 떨어뜨린다.

## 한계 (Limitations)

- **EuroSAT이 세분화 평균을 좌우한다.** FPP는 EuroSAT에서 50.66%를 기록하는 반면 TPT는 42.4%이다. 논문 Table 1에서 나머지 9개 데이터셋만 평균하면 FPP는 약 67.0%, TPT는 약 67.1%이고, ECE는 FPP가 약 4.0, O-TPT가 3.5이다. 전체적인 우위는 이 한 데이터셋에 달려 있다.
- **데이터셋별 보정 결과가 엇갈린다.** FPP의 ECE는 Aircraft(7.26 대 3.68), Caltech(5.85 대 3.80), ImageNet(2.97 대 1.97), ImageNet-R(7.07 대 1.46)에서 O-TPT보다 확연히 나쁘다.
- **분포 변화에서의 정확도.** OOD 평균에서 FPP는 정확도가 TPT와 C-TPT보다 약간 낮고, ImageNet에서는 둘 다보다 낮다.
- **강한 가정 아래의 이론.** SAM 관점은 학습률 1, 모멘텀 없는 단일 스텝을 가정한다. Theorem 1은 이미지 특징이 구 위에서 균등하다고 가정하며 $$O(D^{-3/2})$$ 항까지만 성립한다. 평탄성과 보정 사이의 연결은 선명도 그룹 간 상관관계로 제시될 뿐 증명되지 않았다.
- **작업별 사전학습.** 클래스 집합마다 새 프롬프트를 사전학습해야 하며, SUN397의 경우 약 10분이 걸린다. Table 6은 ImageNet 클래스 이름이나 가우시안 잡음을 대용으로 써도 거의 비슷하게 작동함을 보여주어, 이 부담을 다소 덜어준다.
- **단일 백본.** 모든 주요 결과는 CLIP ViT-B/16을 사용한다. DynaPrompt 베이스라인은 재현 결과로, 원 논문의 정확도보다 낮다(64.22 대 65.43).

## 내 생각 (My Thoughts)

<!-- TODO -->

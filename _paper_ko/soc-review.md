---
title: "SoC: 클래스 프롬프트를 떨어뜨리되, 너무 세게 밀지는 않기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
description: SoC는 CLIP 텍스트 프로토타입에 대한 O-TPT의 완전 직교 페널티를, 의미적으로 가까운 클래스 사이의 반발력에 상한을 두는 Huber 페널티로 대체하여 테스트 시간 프롬프트 튜닝의 보정 오차를 낮춘다.
conference: CVPR
year: 2026
en: soc-review
date: 2026-10-22 00:00:00 +0900
paper:
  title: "SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning"
  authors: "Leo Fillioux, Omprakash Chakraborty, Ismail Ben Ayed, Paul-Henry Cournède, Stergios Christodoulidis, Maria Vakalopoulou, Jose Dolz"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2601.08617"
---

## 초록 (Abstract)

테스트 시간 프롬프트 튜닝(test-time prompt tuning, TPT)은 증강 뷰들에 대한 평균 예측의 엔트로피를 최소화하여 CLIP의 텍스트 프롬프트를 각 테스트 이미지에 맞게 조정한다. 엔트로피 최소화는 모델을 과신하게 만들어 보정(calibration)을 악화시킨다. C-TPT와 O-TPT는 클래스 텍스트 임베딩(프로토타입)에 정규화 항을 두어 이를 해결한다. C-TPT는 프로토타입을 중심(centroid)에서 멀어지게 퍼뜨린다. O-TPT는 모든 프로토타입 쌍을 직교 쪽으로 민다. 저자들은 EuroSAT의 "annual crop land"와 "permanent crop land"처럼 실제로 비슷한 클래스에 대해서는 완전한 직교가 너무 공격적이라고 주장한다. 이런 쌍을 억지로 떼어 놓으면 모델이 틀렸을 때조차 확신하게 된다.

SoC(Semantic Orthogonal Calibration)는 프로토타입을 분리한다는 아이디어는 유지하되, 쌍별 코사인 유사도에 Huber 페널티를 사용한다. 이 페널티는 작은 유사도에서는 이차, 큰 유사도에서는 선형이므로, 매우 비슷한 클래스 사이의 반발력에 상한이 생긴다. 논문은 이를 최대 쌍별 유사도로 표현한 확신도의 하한과, O-TPT가 SoC보다 그 유사도를 더 빠르게 줄인다는 한 단계 분석으로 뒷받침한다.

## 방법 (Method)

![SoC overview](/images/blog/papers/soc-main.png)
*EuroSAT에서의 동기 부여 예시. Zero-shot CLIP은 틀렸지만 가까운 두 클래스 사이에서 불확실해한다. O-TPT의 직교 제약은 모델을 틀리면서도 확신하게 만든다. SoC의 의미 인식(semantic-aware) Huber 제약은 두 클래스의 확신도를 가깝게 유지한다. 그림은 논문에서 가져왔다.*

논문에는 별도의 구조도가 없다. SoC는 TPT 손실에 항 하나를 바꾼 것이며, 위 그림은 그것이 겨냥하는 효과를 보여준다.

### 설정

CLIP은 이미지 임베딩 $$\mathbf{v}$$와 클래스 프로토타입 $$\mathbf{t}_k = f_\phi(\text{"a photo of a [CLASS]"})$$를 주며, 모두 단위 노름이다. 로짓은 CLIP이 학습한 스케일 $$\alpha$$를 사용해 $$z_k = \alpha\,\mathbf{v}^\top \mathbf{t}_k$$이다. 프로토타입을 $$\mathbf{E} \in \mathbb{R}^{K\times d}$$로 쌓으면 원소가 $$s_{ij} = \mathbf{t}_i^\top \mathbf{t}_j$$인 유사도 행렬 $$\mathbf{S} = \mathbf{E}\mathbf{E}^\top$$를 얻는다. 두 기준선은 TPT 엔트로피 손실에 정규화 항을 더한다.

$$
\mathcal{L}_{\text{C-TPT}} = \mathcal{L}_{\text{TPT}} - \lambda \frac{1}{K}\sum_{k=1}^{K} \|\bar{\mathbf{t}} - \mathbf{t}_k\|_2, \qquad
\mathcal{L}_{\text{O-TPT}} = \mathcal{L}_{\text{TPT}} + \lambda \|\mathbf{S} - \mathbf{I}_K\|_2^2 .
$$

### SoC 정규화 항

O-TPT의 페널티는 $$s_{ij}$$에 대해 이차이므로, 기울기가 유사도에 따라 커진다. 가장 비슷한 쌍이 가장 세게 밀린다. SoC는 마진 $$\delta \in [0,1]$$을 가진 Huber 함수를 사용한다.

$$
\mathcal{L}_{\text{Huber}}(s,\delta) =
\begin{cases}
\frac{s^2}{2} & \text{if } s \le \delta \\
\delta\left(s - \frac{\delta}{2}\right) & \text{otherwise}
\end{cases}
$$

$$
\mathcal{L}_{\text{SoC}} = \mathcal{L}_{\text{TPT}} + \lambda \cdot \frac{2}{K(K-1)} \sum_{i<j} \mathcal{L}_{\text{Huber}}(s_{ij}, \delta).
$$

$$\mathbf{t}_i$$에 대한 기울기는 $$g_\delta(s) = \min(s, \delta)$$일 때 $$\sum_{k\neq i} g_\delta(s_{ik})\,\mathbf{t}_k$$이다. 따라서 마진을 넘는 쌍들은 모두 동일하고 유계인 힘을 받는다.

실제로는 유사도를 min-max 정규화하여 $$\tilde{\mathbf{S}} = (\mathbf{S} - \mathbf{S}_{\min}) / (\mathbf{S}_{\max} - \mathbf{S}_{\min})$$로 두고, $$\delta$$는 데이터셋마다 $$\tilde{\mathbf{S}}$$의 20번째 백분위수로 정한다. 가중치는 $$\lambda = 30$$(ImageNet 분포 이동 실험에서는 14)이며, 두 손실 크기의 비율로 재조정된다.

### 이것이 보정에 도움이 되는 이유

$$\mu = \max_{i\neq j} \mathbf{t}_i^\top \mathbf{t}_j$$(최악의 경우 유사도)라 하면, Proposition 1은 임의의 단위 벡터 $$\mathbf{v}$$에 대해 최상위 softmax 확률의 하한을 준다.

$$
p_{\max}(\mathbf{v}) \ge \frac{1}{1 + (K-1)\exp(-\alpha(1-\mu))}.
$$

$$\mu$$가 낮을수록 확신도의 하한이 높아진다. TPT 방법들은 한 번의 경사 단계를 취한다. 가장 비슷한 쌍만 남기고 1차 전개를 사용하면, 크기 $$\eta$$의 한 단계는 다음을 준다.

$$
\mu'_{\text{O-TPT}} \approx (1-4\eta)\mu, \qquad
\mu'_{\text{Huber}} \approx
\begin{cases}
(1-2\eta)\mu & \mu \le \delta \\
\mu - 2\eta\delta & \mu > \delta .
\end{cases}
$$

O-TPT는 두 영역 모두에서 $$\mu$$를 더 많이 줄이므로 확신도의 하한을 더 많이 높인다(Corollary 1). 저자들은 이를 의미적으로 가까운 클래스 쌍에서 O-TPT가 보이는 과신과 연결하며, EuroSAT에서 zero-shot 유사도에 따른 쌍별 ECE로 이를 보여준다.

## 주요 실험 (Main Experiments)

설정: 별도 언급이 없으면 CLIP ViT-L/14, 프롬프트는 "a photo of a"로 초기화, 학습률 0.005의 AdamW 한 단계, 이미지당 증강 뷰 64개. 지표는 top-1 정확도(%)와 ECE(%)이다. "11개 데이터셋" 평균은 ImageNet과 10개의 fine-grained 및 특수 데이터셋(DTD, Flowers, Food101, SUN397, Aircraft, Pets, Caltech, UCF101, EuroSAT, Cars)을 포함한다. 아래 표는 논문 Table 1, 2, 3의 평균 열만 모은 것이다.

| 방법 | 11개 데이터셋, ViT-L/14 Acc | ECE | 11개 데이터셋, ViT-B/16 Acc | ECE | ImageNet-A/v2/R/Sketch, ViT-L/14 Acc | ECE |
|---|---|---|---|---|---|---|
| Zero-Shot | 71.1 | 5.1 | 63.9 | 4.2 | 70.0 | 4.4 |
| TPT | 72.0 | 14.9 | 65.2 | 11.2 | 72.9 | 14.2 |
| C-TPT | 72.1 | 10.0 | 64.6 | 5.0 | 72.0 | 12.6 |
| O-TPT | 71.4 | 7.7 | 64.0 | 4.8 | 71.3 | 9.9 |
| SoC | 72.3 | 5.4 | 64.6 | 4.3 | 71.3 | 8.4 |

ViT-L/14에서의 데이터셋별 ECE(논문 Table 1), SoC vs. O-TPT:

| 방법 | ImgNet | DTD | Flowers | Food101 | SUN397 | Aircraft | Pets | Caltech | UCF101 | EuroSAT | Cars |
|---|---|---|---|---|---|---|---|---|---|---|---|
| O-TPT | 5.5 | 13.8 | 7.0 | 2.8 | 7.6 | 16.8 | 1.4 | 2.0 | 8.5 | 17.7 | 2.2 |
| SoC | 7.2 | 10.9 | 5.3 | 2.1 | 7.2 | 12.7 | 0.5 | 1.3 | 6.5 | 3.2 | 2.2 |

SoC는 모든 설정에서 적응 방법들 중 가장 낮은 평균 ECE를 기록하며, ViT-L/14에서는 ECE(5.4)가 zero-shot CLIP(5.1)에 가까우면서 정확도는 약간 더 높다. 가장 큰 단일 변화는 EuroSAT로, ECE가 17.7에서 3.2로 떨어지고 정확도는 53.6에서 58.3으로 오른다. ImageNet 분포 이동에서는 정확도가 O-TPT와 같고 TPT와 C-TPT보다 낮다. 경사 단계를 한 번이 아니라 두 번 취하면 SoC의 ECE는 약 23% 나빠지는 반면 O-TPT는 39% 나빠지며, 이는 한 단계 분석과 일치한다.

## 한계 (Limitations)

- **향상은 정확도가 아님.** ImageNet 변형에서 SoC는 O-TPT와 같고(71.3) TPT(72.9)와 C-TPT(72.0)보다 낮다. ViT-B/16에서는 TPT보다 낮다(64.6 vs. 65.2).
- **모든 곳에서 더 낫지는 않음.** ImageNet에서 SoC의 ECE는 O-TPT보다 나쁘며(7.2 vs. 5.5), Cars에서는 같다.
- **한 데이터셋이 평균의 상당 부분을 차지.** 11개 데이터셋 중 EuroSAT 하나가 14.5포인트의 ECE 감소를 차지한다. 나머지 10개 데이터셋에서 O-TPT와의 ECE 차이는 최대 4.1포인트이다.
- **단순화된 이론.** 분석은 가장 비슷한 쌍만 남기고, 한 경사 단계의 1차 전개를 사용하며, 확신도의 하한을 준다. 변화의 방향을 설명할 뿐, 보정 오차의 크기를 설명하지는 않는다.
- **추가적인 선택 사항.** 마진은 min-max 정규화된 유사도의 데이터셋별 백분위수이고, $$\lambda$$는 벤치마크마다 다르며(30 vs. 14), 손실은 손실 크기의 비율로 재조정된다. 부록은 네 개 데이터셋에서 $$\lambda$$에 대한 ablation을 제시하지만, $$\delta$$에 쓰인 백분위수에 대한 ablation은 보고되지 않는다.
- **여전히 TPT의 비용에 묶임.** SoC는 64개의 증강 뷰를 사용하는 TPT의 이미지별 최적화를 그대로 유지한다. 정규화 항의 비용은 클래스 수에 대해 $$O(n^2)$$로 O-TPT의 $$O(n^3)$$보다 작지만, 주된 추론 비용은 변하지 않는다.

## 내 생각 (My Thoughts)

<!-- TODO -->

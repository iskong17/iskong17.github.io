---
title: "C-TPT: 텍스트 특징을 퍼뜨려 TPT의 과신을 바로잡기"
category: paper-review
subfields: [calibration, tta, vlm]
lang: ko
description: C-TPT는 테스트 시간 프롬프트 튜닝에 CLIP의 클래스 텍스트 특징을 서로 멀어지게 하는 레이블 불필요 항을 추가해, TPT의 정확도 향상은 유지하면서 보정 오차를 대략 절반으로 줄인다.
conference: ICLR
year: 2024
paper:
  title: "C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models via Text Feature Dispersion"
  authors: "Hee Suk Yoon, Eunseop Yoon, Joshua Tian Jin Tee, Mark Hasegawa-Johnson, Yingzhen Li, Chang D. Yoo"
  venue: "ICLR 2024"
  url: "https://arxiv.org/abs/2403.14119"
en: c-tpt-review
date: 2026-10-22 00:00:00 +0900
---

## 초록 (Abstract)

테스트 시간 프롬프트 튜닝(test-time prompt tuning, TPT)은 레이블이 없는 각 테스트 이미지에서 엔트로피 최소화 목적 함수로 프롬프트를 튜닝해 CLIP의 zero-shot 정확도를 높인다. 엔트로피 최소화는 예측을 더 확신하게 만들며, 논문은 이 때문에 TPT의 보정(calibration)이 나빠진다는 것을 보인다. 즉 확신도가 실제로 맞히는 빈도와 더 이상 일치하지 않는다. 온도 스케일링(temperature scaling)이나 학습 가능한 보정 손실 같은 표준적인 해결책은 레이블이 있는 데이터가 필요한데, 테스트 시점에는 그런 데이터가 없다.

논문은 CLIP 내부에서 레이블 없이 쓸 수 있는 신호를 찾는다. 정확도가 비슷한 여러 수작업 프롬프트를 비교해 보면, 클래스 텍스트 특징이 더 넓게 퍼진 프롬프트일수록 보정 오차가 낮다. 저자들은 이 퍼짐을 평균 텍스트 특징 분산(Average Text Feature Dispersion, ATFD)으로 측정하고, $$-\text{ATFD}$$를 TPT의 정규화 항으로 추가한다. 그 결과인 C-TPT는 TPT의 정확도 향상 대부분을 유지하면서, 세분화(fine-grained) 벤치마크에서 기대 보정 오차(expected calibration error, ECE)를 대략 절반으로 줄인다.

## 방법 (Method)

![C-TPT overview](/images/blog/papers/c-tpt-main.png)
*C-TPT. 튜닝 가능한 프롬프트를 모든 클래스 이름과 결합해 텍스트 인코더에 통과시킨다. 테스트 시간 튜닝 동안, 그 결과로 얻은 클래스 텍스트 특징의 분산(ATFD)을 TPT 손실과 함께 최대화한다. 튜닝된 텍스트 특징은 이후 이미지 특징과 함께 추론에 쓰인다. 그림은 논문에서 가져왔다.*

### 세 가지 관찰

이 방법은 주로 CLIP-ViT-B/16으로 보인 세 가지 관찰에 기반한다.

1. **TPT는 보정 오차를 키운다.** StanfordCars의 신뢰도 다이어그램(reliability diagram)은 TPT가 하드 프롬프트(hard prompt)보다 정확도를 높이지만 과신하게 됨을 보여 준다.
2. **정확도가 비슷한 프롬프트도 ECE는 크게 다를 수 있다.** 80개의 수작업 CLIP 템플릿에 걸쳐, 같은 정확도 수준에서도 ECE가 넓게 분포한다.
3. **보정이 잘 된 프롬프트는 텍스트 특징이 퍼져 있다.** t-SNE 그림에서 보정이 나쁜 프롬프트의 클래스 텍스트 특징은 촘촘한 군집으로 뭉친다. 보정이 잘 된 프롬프트의 특징은 퍼져 있고 각자의 클래스 근처에 놓인다.

이미지 특징은 프롬프트에 의존하지 않으므로, 프롬프트 사이의 보정 차이는 반드시 텍스트 쪽에서 비롯된다. 그래서 논문은 텍스트 특징의 기하 구조에 집중한다.

### 평균 텍스트 특징 분산

프롬프트 $$\mathbf{p}$$와 $$N$$개 클래스에 대해, $$\mathbf{t}_{[\mathbf{p};y_i]}$$를 "$$\mathbf{p}$$ + 클래스 $$y_i$$"라는 설명의 텍스트 특징이라 하자. 클래스 특징들의 중심점(centroid)은

$$
\mathbf{t}_{\text{centroid}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{t}_{[\mathbf{p};y_i]},
$$

이고, ATFD는 그 중심점으로부터의 평균 L2 거리이다.

$$
\text{ATFD} = \frac{1}{N}\sum_{i=1}^{N} \left\| \mathbf{t}_{\text{centroid}} - \mathbf{t}_{[\mathbf{p};y_i]} \right\|_2 .
$$

저자들은 각 데이터셋에서 최고 프롬프트와 정확도 차이가 3% 이내인 하드 프롬프트들을 골라 ATFD와 ECE를 비교한다. ATFD와 ECE 사이의 피어슨 상관계수는 데이터셋 평균으로 CLIP-RN50에서 $$-0.70$$, CLIP-ViT-B/16에서 $$-0.76$$이다.

### 보정된 테스트 시간 프롬프트 튜닝

C-TPT는 TPT 목적 함수에 음의 ATFD를 더한다.

$$
\mathbf{p}^{*} = \arg\min_{\mathbf{p}} \left[ \mathcal{L}_{\text{TPT}} + \lambda \cdot \mathcal{L}_{\text{C-TPT}} \right], \qquad \mathcal{L}_{\text{C-TPT}} = -\text{ATFD}\left(\mathbf{t}_{[\mathbf{p};y_1]}, \dots, \mathbf{t}_{[\mathbf{p};y_N]}\right).
$$

$$\mathcal{L}_{\text{TPT}}$$는 통상적인 TPT 손실로, 테스트 이미지의 확신도 높은 증강 뷰들에 대한 평균 예측의 엔트로피이다. 새 항은 클래스 설명의 텍스트 특징만 필요로 하므로, 레이블도 이미지 통계도 쓰지 않는다. 두 항 모두 프롬프트 임베딩만 업데이트하며, CLIP의 두 인코더는 고정된다.

설정은 TPT를 따른다. 프롬프트는 "a photo of a"에서 시작해 학습률 0.005의 AdamW 한 스텝을 받으며, 나머지 TPT 설정은 모두 유지한다. $$\lambda$$는 세분화 데이터셋에서 50.0, ImageNet 변형에서 20.0이다. 논문은 앙상블 설정도 보고하는데, 네 가지 하드 프롬프트 초기화("a photo of a", "a photo of the", "a picture of a", "a picture of the")를 각각 튜닝하고 그 로짓을 평균한다.

Caltech101에 대한 t-SNE 분석에서, 일반 TPT로 튜닝한 프롬프트는 보정이 나쁜 하드 프롬프트처럼 텍스트 특징이 뭉친다. $$\lambda$$를 키우면 특징이 퍼지고 클래스별로 모인다.

## 주요 실험 (Main Experiments)

백본은 CLIP-RN50과 CLIP-ViT-B/16이다. 평가 지표는 top-1 정확도(%, 높을수록 좋음)와 ECE(%, 낮을수록 좋음)이다. 표에는 하드 프롬프트 및 앙상블 초기화에 대해 논문 Table 1과 2의 Average 열만 실었다.

**세분화 분류, 11개 데이터셋 평균 (ImageNet, Caltech, Pets, Cars, Flower, Food101, Aircraft, SUN397, DTD, EuroSAT, UCF101)**

| 설정 | RN50 Acc. | RN50 ECE | ViT-B/16 Acc. | ViT-B/16 ECE |
|---|---|---|---|---|
| CLIP (하드 프롬프트) | 55.9 | 5.61 | 63.7 | 4.43 |
| + TPT | 57.7 | 11.7 | 65.0 | 11.6 |
| + TPT + C-TPT | 57.8 | **6.20** | 64.8 | **5.13** |
| CLIP (앙상블) | 57.1 | 5.58 | 64.2 | 5.82 |
| + TPT | 58.2 | 10.7 | 65.5 | 9.91 |
| + TPT + C-TPT | 58.5 | **5.20** | 65.8 | **4.62** |

**자연 분포 변화, ImageNet-A, -V2, -R, -Sketch 평균**

| 설정 | RN50 Acc. | RN50 ECE | ViT-B/16 Acc. | ViT-B/16 ECE |
|---|---|---|---|---|
| CLIP (하드 프롬프트) | 40.6 | 7.46 | 57.2 | 5.04 |
| + TPT | 43.5 | 16.7 | 59.9 | 12.0 |
| + TPT + C-TPT | 42.8 | **12.1** | 59.6 | **5.82** |
| CLIP (앙상블) | 42.0 | 9.06 | 58.4 | 6.10 |
| + TPT | 44.5 | 15.8 | 61.2 | 10.9 |
| + TPT + C-TPT | 44.0 | **13.6** | 60.7 | **8.32** |

TPT는 모든 설정에서 ECE를 1.7배에서 2.6배까지 키운다. C-TPT를 더하면 세분화 데이터셋에서 TPT의 ECE가 약 절반으로 줄어들고(47%~56% 감소), 평균 정확도는 거의 변하지 않는다. 자연 분포 변화에서는 향상이 더 작고(ECE 14%~52% 감소), C-TPT가 평균 정확도를 최대 0.7포인트 깎는다. 대부분의 설정에서 C-TPT의 ECE는 여전히 튜닝하지 않은 CLIP 프롬프트보다 높다. ViT-B/16에서의 별도 비교에서, C-TPT는 레이블이 있는 ImageNet으로 온도를 학습한 TPT보다 평균 ECE가 낮다.

## 한계 (Limitations)

- **소폭의 정확도 하락 (저자들이 직접 언급).** 다른 정규화 기반 보정 방법처럼 C-TPT도 때때로 정확도를 떨어뜨린다. 예로 ImageNet-A의 RN50(하드 프롬프트 기준 25.2에서 23.4)과 UCF101의 ViT-B/16(67.3에서 65.7)이 있다.
- **$$\lambda$$를 테스트 시점에 튜닝할 수 없다 (저자들이 직접 언급).** 저자들은 ECE와 ATFD가 포화되는 검증 세트 추세를 보고 값을 고르며, 샘플별로 최적이 아닐 수 있다고 언급한다. 부록은 50.0이라는 단일 값을 설명하지만 자연 분포 변화 실험에서는 20.0을 사용하므로, 실제로는 벤치마크마다 값이 다르다.
- **보정은 개선되었을 뿐 해결되지 않았다.** RN50의 ImageNet-A에서 ECE는 25.4(하드 프롬프트)와 27.0(앙상블)에 머문다. DTD에서는 네 가지 설정에 걸쳐 11.9~19.8에 머문다.
- **상관관계에 기반한 동기.** ATFD/ECE 관계는 정확도가 비슷한 하드 프롬프트들에 대한 경험적 상관관계이며, 7.2절의 설명은 t-SNE 그림에 기반한 정성적 설명이다.
- **TPT의 비용을 그대로 물려받는다.** 각 테스트 이미지에는 여전히 64개의 증강 뷰와 텍스트 인코더를 통한 역전파가 필요하다. ATFD 항은 모든 클래스의 텍스트 특징을 필요로 하지만, 이는 TPT 순전파에서 이미 계산된다.

## 내 생각 (My Thoughts)

<!-- TODO -->

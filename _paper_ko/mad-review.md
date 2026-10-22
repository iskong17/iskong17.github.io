---
title: "MAD: 질문에 필요한 모달리티를 모델에게 묻기"
category: paper-review
subfields: [hallucination, avllm]
lang: ko
en: mad-review
date: 2026-10-22 00:00:00 +0900
description: 질문에 오디오, 비디오, 혹은 둘 다가 필요한지를 모델에게 묻고, 그 답으로 대조 디코딩의 각 분기에 가중치를 준다.
conference: CVPR
year: 2026
paper:
  title: "MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models"
  authors: "Sangyun Chung, Se Yeon Kim, Youngchae Chee, Yong Man Ro"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2601.21181"
---

## 초록 (Abstract)

멀티모달 LLM은 **교차 모달 환각**(cross-modal hallucination)을 겪는다. 한 모달리티가 다른 모달리티에 대한 생성에 간섭하는 현상으로, 예를 들어 비디오 때문에 모델이 오디오에 없는 소리를 "듣는" 경우이다. 저자들은 이것이 각 모달리티 내부의 표현 문제가 아니라고 주장한다. 모델이 모달리티들이 서로 미치는 영향을 제어하지 못하는 것이다. 답해야 할 질문은 다음과 같다: *이 질문에 대해 모델은 오디오를 얼마나, 비디오를 얼마나 신뢰해야 하는가?*

기존 대조 디코딩(contrastive decoding, CD) 방법은 모든 질문에 고정된 대조 강도를 사용한다. MAD는 모델에게 추가로 한 가지 질문을 한다: 이 질문에는 오디오, 비디오, 혹은 둘 다가 필요한가? 그 답을 4분기 CD 수식의 분기별 가중치로 바꾼다. 학습은 필요 없다.

## 방법 (Method)

![MAD overview](/images/blog/papers/mad-main.png)
*두 단계로 이루어진 MAD. 1단계는 모달리티 질의 프롬프트로부터 모달리티 가중치를 추출한다. 2단계는 그 가중치로 네 분기의 로짓을 결합한다. 그림은 논문에서 가져왔다.*

### 가중 대조 디코딩

모달리티 $$m$$에 대해 대조 신호는 $$X_m$$이 있을 때의 로짓에서 열화된 $$\tilde X_m$$이 있을 때의 로짓을 뺀 것이다. 예를 들어 $$\Delta_a = z(A) - z(\tilde A)$$이다. MAD는 대조 강도를 두 인자로 나눈다.

$$
\mathrm{logit}^{(m)}(y_t) = \mathrm{logit}(y_t \mid X_m, X_q, y_{<t}) + \gamma\, w_m \Delta_m
$$

여기서 $$\gamma$$는 전역 강도이고, $$w_m$$은 **이 질문에 대해** 모달리티 $$m$$이 얼마나 중요한지를 나타낸다.

### 1단계: 모달리티 적응 가중치 추출 (위)

모델은 입력과 함께, 질문에 어떤 모달리티가 필요한지 묻는 모달리티 질의 프롬프트 $$X_m$$을 받는다. 모델은 답을 생성하지 않는다. MAD는 세 후보 단어의 다음 토큰 로짓을 읽고 softmax를 적용한다.

$$
[w_{av}, w_v, w_a] = \mathrm{softmax}\big([z_{\text{both}}, z_{\text{video}}, z_{\text{audio}}]\big)
$$

가중치는 질문 유형을 따라간다. 벤치마크 전체 평균으로 시각 질문은 $$w_v = 0.565$$, 오디오 질문은 $$w_a = 0.482$$, 시청각 질문은 $$w_{av} = 0.464$$를 받는다.

### 2단계: 모달리티 적응 생성 (아래)

AVCD와 마찬가지로 MAD는 네 분기를 사용한다: 시청각 $$z_{VA}$$, 비디오 전용 $$z_{V\tilde A}$$, 오디오 전용 $$z_{\tilde V A}$$, 질문 전용 $$z_{\tilde V \tilde A}$$. 이들은 정사각형을 이루며, 각 변이 하나의 CD 항이다. 오디오가 있을 때와 없을 때의 시각 CD, 비디오가 있을 때와 없을 때의 오디오 CD이다. 다른 모달리티가 있는 상태에서 취한 항은 $$w_{av}$$를, 없는 상태에서 취한 항은 $$w_v$$ 또는 $$w_a$$를 사용한다. 네 항을 더하고 계수를 정리하면 그림의 Eq. (9)가 된다.

$$
z_{\text{MAD}} = (2 + 2\gamma w_{av})\, z_{VA} + (1 - \gamma w_{av} + \gamma w_a)\, z_{\tilde V A} + (1 - \gamma w_{av} + \gamma w_v)\, z_{V \tilde A} - \gamma (w_v + w_a)\, z_{\tilde V \tilde A}
$$

그림의 예시에서 질문은 망치 소리에 대해 묻는다. 가중치는 오디오를 선호하고, 오디오 전용 분기는 "No"라고 답하며, MAD는 원래의 "Yes"를 올바른 "No"로 뒤집는다.

## 주요 실험 (Main Experiments)

CMM은 시각 우세, 오디오 우세, 언어 우세 하위 집합에 대한 정확도를 보고한다. AVHBench는 비디오로 인한 오디오 환각과 오디오로 인한 비디오 환각에 대한 정확도를 보고한다.

| 모델 | Visual Dom. | Audio Dom. | Lang. Dom. | CMM Overall | V→A Hall. | A→V Hall. | AVHBench Overall |
|---|---|---|---|---|---|---|---|
| VideoLLaMA2-AV | 71.8 | 80.0 | 68.8 | 73.5 | 75.7 | 79.0 | 77.4 |
| + VCD (extended) | 71.3 | 83.3 | 74.8 | 76.4 | 66.0 | 74.8 | 70.4 |
| + AVCD | 71.8 | 84.0 | 71.5 | 75.8 | 78.3 | **80.3** | 79.3 |
| **+ MAD** | **82.3** | **84.3** | **77.5** | **81.3** | **79.7** | 79.1 | **79.4** |
| Qwen2.5-Omni-7B | 64.5 | 72.3 | 81.3 | 72.7 | 73.0 | 80.7 | 76.9 |
| + VCD (extended) | 62.5 | 71.3 | **84.5** | 72.8 | 70.3 | 77.1 | 73.7 |
| + AVCD | 66.3 | 72.8 | 81.0 | 73.3 | 75.8 | 79.7 | 77.8 |
| **+ MAD** | **76.8** | **84.3** | 83.3 | **81.4** | **78.7** | **84.4** | **81.6** |

- 가장 큰 도약은 CMM 시각 우세 질문에서 나타난다: VideoLLaMA2-AV는 +10.5, Qwen2.5-Omni는 +12.3. AVCD는 이 열을 거의 움직이지 못한다.
- 전체 softmax로 가중하는 방식(81.3)이 균등 가중치(79.4)와 argmax에 모든 가중치를 두는 방식(78.7)보다 낫다.
- MAD는 VideoLLaMA2-AV의 일반적인 시청각 이해를 해치지 않는다: OmniBench 36.3 → 36.8, WorldSense 23.3 → 25.6, MUSIC-AVQA 78.1 → 79.1.

## 한계 (Limitations)

- **비용.** 토큰마다 네 분기가 필요하고, 모달리티 질의 프롬프트를 위한 추가 패스가 한 번 더 필요하다.
- **관련성은 일치가 아니다.** 가중치는 어떤 모달리티가 *관련 있는지*를 말해줄 뿐, 단일 모달 분기들이 공동 예측과 일치하는지는 말해주지 않는다. 관련 있는 두 모달리티가 여전히 충돌할 수 있으며, MAD는 이를 감지하지 못한다.
- **자기 평가가 정확해야 한다.** 모델이 질문에 필요한 모달리티를 잘못 판단하면, 가중치가 디코딩을 잘못된 방향으로 민다.

## 내 생각 (My Thoughts)

<!-- TODO -->

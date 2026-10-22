---
title: "AVCD: 세 가지 모달리티를 위한 대조 디코딩"
category: paper-review
subfields: [hallucination, avllm]
lang: ko
description: 어텐션으로 지배적인 모달리티를 찾고 나머지를 마스킹한 뒤, 대조 디코딩을 오디오, 비디오, 텍스트로 확장한다.
conference: NeurIPS
year: 2025
paper:
  title: "AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding"
  authors: "Chaeyoung Jung, Youngjoon Jang, Joon Son Chung"
  venue: "NeurIPS 2025"
  url: "https://arxiv.org/abs/2505.20862"
en: avcd-review
date: 2026-10-22 00:00:00 +0900
---

## 초록 (Abstract)

환각(hallucination)은 멀티모달 LLM에서 여전히 큰 문제이다. VCD와 같은 대조 디코딩(contrastive decoding, CD) 방법은 비전-언어 모델(VLM)에서 잘 작동한다. 이미지를 손상시켜 시각적 근거 없이 모델이 무엇을 예측하는지 본 뒤, 그것을 빼는 방식이다. 오디오-비주얼 LLM(AV-LLM)은 비디오, 오디오, 언어의 세 입력을 가지므로 어느 것을 섭동해야 할지 분명하지 않다. 또한 환각은 이들 중 어느 하나에서, 혹은 이들 사이의 상호작용에서 생길 수 있다.

AVCD는 이러한 세 모달리티 설정을 위한 학습 불필요(training-free) 디코딩 방법이다. 현재 예측을 어떤 모달리티가 지배하는지 측정하고, 덜 지배적인 모달리티들을 마스킹한 뒤, 세 모달리티에 대해 유도한 CD 식으로 그 결과 브랜치들을 결합한다. 모델이 이미 확신하고 있을 때는 엔트로피 게이트가 추가 연산을 건너뛴다.

## 방법 (Method)

![AVCD overview](/images/blog/papers/avcd-main.png)
*AVCD 파이프라인. 어텐션으로 모달리티 지배 순서를 구하고, 덜 지배적인 모달리티를 마스킹한 뒤, 마스킹된 브랜치들을 삼중 모달 CD로 결합한다. 그림은 논문에서 가져왔다.*

### 대조 디코딩 복습

VLM에서 CD는 원래 브랜치와 시각 입력이 섭동된 브랜치를 대조한다.

$$
\mathrm{logit} = (1+\alpha)\,\mathrm{logit}(y \mid x^v, x^l) - \alpha\,\mathrm{logit}(y \mid x^{\neg v}, x^l)
$$

섭동된 브랜치는 모델이 언어 편향만으로 무엇을 예측하는지 보여 준다. 이를 빼면 예측 중 실제 시각적 근거에서 나온 부분이 증폭된다.

### 1. 모달리티 지배도 (그림 왼쪽)

$$A^j_{Q_K, i}$$를 층 $$j$$에서 마지막 쿼리 토큰이 키 토큰 $$i$$에 주는 어텐션이라 하자. 모달리티 $$M \in \{V, A, L\}$$의 지배도(dominance)는 그 모달리티가 받는 전체 어텐션을 $$J$$개 층 전체에 대해 평균한 값이다.

$$
D_M = \frac{1}{J} \sum_{j=1}^{J} \sum_{i \in M} A^j_{Q_K, i}
$$

그림에서는 $$D_L > D_V > D_A$$이므로 언어가 지배적이다.

### 2. 어텐션 기반 마스킹 (가운데)

AVCD는 **지배적이지 않은** 모달리티를 섭동한다. 토큰을 삭제하지는 않는다. 해당 모달리티에서 어텐션을 가장 많이 받은 상위 $$P\%$$ 토큰의 표현을 0으로 만든다. 언어가 지배적이면 비디오 마스킹, 오디오 마스킹, 둘 다 마스킹의 세 가지 섭동 브랜치가 생긴다.

### 3. 삼중 모달 대조 디코딩 (오른쪽)

비디오 CD는 오디오가 우연히 존재하는지 여부에 좌우되어서는 안 되므로, AVCD는 오디오가 온전한 경우와 마스킹된 경우를 평균한다. 이 평균을 로그 공간에서 근사하는데, 두 확률이 가까우면 이 근사는 정확하다. 네 브랜치

$$
L_{VA},\quad L_{\neg V A},\quad L_{V \neg A},\quad L_{\neg V \neg A} \qquad (L_{\cdot} = \log p(y_t \mid \cdot, L))
$$

를 사용하면 비디오 CD와 오디오 CD는 다음과 같다.

$$
\mathrm{logit}_v \propto (1+\alpha_v)(L_{VA} + L_{V\neg A}) - \alpha_v (L_{\neg V A} + L_{\neg V \neg A})
$$

$$
\mathrm{logit}_a \propto (1+\alpha_a)(L_{VA} + L_{\neg V A}) - \alpha_a (L_{V \neg A} + L_{\neg V \neg A})
$$

그리고 둘의 합이 최종 로짓이다.

$$
\mathrm{logit}_{\text{AVCD}} = (2 + \alpha_v + \alpha_a) L_{VA} + (1 - \alpha_v + \alpha_a) L_{\neg V A} + (1 + \alpha_v - \alpha_a) L_{V \neg A} - (\alpha_v + \alpha_a) L_{\neg V \neg A}
$$

### 4. 엔트로피 기반 적응형 디코딩 (위)

원래의 순전파가 먼저 실행된다. 그 엔트로피 $$H_t = -\sum_v p_t(v) \log p_t(v)$$가 임계값 $$\tau$$보다 낮으면 모델이 확신하고 있는 것이므로, AVCD는 해당 토큰에 대해 원래 예측을 유지한다.

## 주요 실험 (Main Experiments)

MUSIC-AVQA와 AVHBench의 정확도(%), 그리고 AVHBench 캡셔닝 점수이다. VCD\* 변형은 논문을 참고하라.

| 모델 | 디코딩 | MUSIC-AVQA | AVHBench | AVHBench-Cap |
|---|---|---|---|---|
| VideoLLaMA2 | Base | 81.30 | 70.52 | 2.84 |
| | VCD | 77.66 | 65.18 | 2.86 |
| | VCD\* | 81.49 | 69.18 | 3.00 |
| | **AVCD** | **81.58** | **72.15** | **3.03** |
| video-SALMONN | Base | 48.50 | 58.19 | 1.83 |
| | VCD | 41.57 | 60.61 | 2.41 |
| | VCD\* | 49.00 | 60.66 | 2.44 |
| | **AVCD** | **49.73** | **62.18** | **2.47** |

AVCD는 비디오 전용 LLM에도 도움이 된다. MSVD-QA에서 VideoLLaMA2는 74.43에서 75.20으로, Video-LLaVA는 70.20에서 72.16으로 오른다.

- 일반 VCD는 AV-LLM의 성능을 오히려 *떨어뜨리는* 경우가 많다(VideoLLaMA2의 AVHBench에서 70.52 → 65.18). VLM 방식으로 단일 모달리티만 섭동하는 것은 그대로 옮겨지지 않는다.
- ablation에서 마스킹된 단일 모달리티에 대한 두 브랜치 CD는 조금만 도움이 되고, 마스킹된 세 브랜치를 모두 단순히 빼면 70.94가 나온다. 유도된 삼중 모달 식은 72.15에 도달한다.

## 한계 (Limitations)

- **비용.** 엔트로피 게이트가 없으면 토큰마다 네 번의 순전파가 필요하다. 게이트는 정확도와 속도를 맞바꾼다. 논문의 그래프에서 $$\tau$$를 낮추면 정확도는 오르지만, 지연 시간은 토큰당 약 1.3 s에서 약 3.75 s로 늘어난다.
- **어텐션은 필요와 같지 않다.** 지배도는 모델이 *실제로* 무엇에 주목하는지를 말해 줄 뿐, 질문이 무엇을 요구하는지는 말해 주지 않는다. 소리에 관한 질문이어도 어텐션의 대부분이 비디오 토큰에 갈 수 있다. [MAD](/blog/paper-review/ko/mad-review/)는 질문에 어떤 모달리티가 필요한지 모델에게 직접 물어 이 문제를 다룬다.
- **일부 설정에서는 향상이 작다.** VideoLLaMA2의 MUSIC-AVQA에서 기본 모델 대비 향상은 0.3포인트 미만이다.

## 내 생각 (My Thoughts)

<!-- TODO -->

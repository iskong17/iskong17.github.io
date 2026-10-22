---
title: "Fork-Merge Decoding: 오디오와 비디오가 먼저 따로 생각하게 하기"
category: paper-review
subfields: [avllm]
lang: ko
en: fmd-review
date: 2026-10-22 00:00:00 +0900
description: 초기 디코더 층에서는 오디오 전용 분기와 비디오 전용 분기를 따로 실행하고, 두 은닉 상태를 병합한 뒤 그 이후부터 함께 디코딩한다.
conference: EMNLP
year: 2026
paper:
  title: "Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visual Large Language Models"
  authors: "Chaeyoung Jung, Youngjoon Jang, Jongmin Choi, Joon Son Chung"
  venue: "EMNLP 2026 Findings"
  url: "https://arxiv.org/abs/2505.20873"
---

## 초록 (Abstract)

AV-LLM은 오디오와 비디오를 함께 디코더에 입력한다. 공동 처리 덕분에 모델은 두 모달리티를 함께 추론할 수 있지만, 두 모달리티를 균형 있게 사용한다는 보장은 없다. 모델은 한 모달리티에 과도하게 의존할 수 있다(**모달리티 편향(modality bias)**). 일반 디코딩(vanilla decoding)에서는 비디오에 대한 어텐션이 오디오에 대한 어텐션보다 훨씬 강하며, 이 불균형이 환각(hallucination)과 잘못된 추론으로 이어진다.

Fork-Merge Decoding(FMD)은 학습이 필요 없는 해결책이다. 입력을 비디오가 마스킹된 사본과 오디오가 마스킹된 사본으로 **분기**(fork)하고, 각각을 처음 $$L_{\text{fork}}$$개 디코더 층에 독립적으로 통과시킨 뒤, 은닉 상태를 **병합**(merge)하고 나머지 층은 한 번만 실행한다. 초기 층만 두 번 실행되므로 대조 디코딩(contrastive decoding)보다 저렴하다.

## 방법 (Method)

![FMD overview](/images/blog/papers/fmd-main.png)
*Fork: 마스킹된 두 입력이 초기 디코더 층을 각각 따로 통과한다. Merge: 두 은닉 상태를 모달리티 영역별로 어텐션 기반 가중치로 융합하고, 나머지 디코더는 한 번만 실행한다. 그림은 논문에서 가져왔다.*

### Fork

입력의 각 사본은 한 모달리티가 마스킹되어 있고, 질문은 두 사본 모두에 그대로 남는다.

$$
x_{v[\text{masked}]} = x^{\neg v} \oplus x^a \oplus x^l, \qquad
x_{a[\text{masked}]} = x^v \oplus x^{\neg a} \oplus x^l
$$

마스킹은 인코더에 들어가기 전에 원본 비디오 프레임이나 오디오 파형을 0으로 만드는 방식이므로, 시퀀스 형태와 위치는 그대로 유지된다. 디코더 $$\phi$$에 대해 각 사본은 처음 $$L_{\text{fork}}$$개 층만 통과한다.

$$
h^{\neg v}_{\text{fork}} = \phi_{\le L_{\text{fork}}}\big(x_{v[\text{masked}]}\big), \qquad
h^{\neg a}_{\text{fork}} = \phi_{\le L_{\text{fork}}}\big(x_{a[\text{masked}]}\big)
$$

$$h^{\neg v}$$는 오디오와 텍스트를, $$h^{\neg a}$$는 비디오와 텍스트를 담는다. 초기 층에서 모델은 오디오와 비디오를 동시에 보지 않으므로, 어느 쪽도 다른 쪽을 압도할 수 없다.

### Merge

은닉 상태는 비디오($$\mathcal{V}$$), 오디오($$\mathcal{A}$$), 언어($$\mathcal{L}$$) 토큰 위치별로 따로 병합된다.

$$
\begin{aligned}
h_{\text{merge}}[\mathcal{V}] &= (1-\alpha)\, h^{\neg v}_{\text{fork}}[\mathcal{V}] + \alpha\, h^{\neg a}_{\text{fork}}[\mathcal{V}] \\
h_{\text{merge}}[\mathcal{A}] &= \alpha\, h^{\neg v}_{\text{fork}}[\mathcal{A}] + (1-\alpha)\, h^{\neg a}_{\text{fork}}[\mathcal{A}] \\
h_{\text{merge}}[\mathcal{L}] &= \tfrac{1}{2}\big(h^{\neg v}_{\text{fork}}[\mathcal{L}] + h^{\neg a}_{\text{fork}}[\mathcal{L}]\big)
\end{aligned}
$$

$$\alpha$$는 해당 위치의 모달리티를 실제로 본 분기의 가중치이다. 언어 위치는 두 분기 모두 텍스트를 유지했으므로 평균을 취한다.

**어텐션 기반 $$\alpha$$.** $$\alpha$$는 각 분기의 마지막 토큰 어텐션 중 그 분기가 여전히 볼 수 있는 모달리티에 할당된 비율이다.

$$
\alpha = \frac{\sum_{i \in \mathcal{A}} a^{v[\text{masked}]}_{\text{last}}[i] + \sum_{i \in \mathcal{V}} a^{a[\text{masked}]}_{\text{last}}[i]}
{\sum_{i \in \mathcal{V} \cup \mathcal{A}} \big(a^{v[\text{masked}]}_{\text{last}}[i] + a^{a[\text{masked}]}_{\text{last}}[i]\big)}
$$

이후 나머지 층은 병합된 상태에 대해 한 번 실행된다: $$h_{\text{final}} = \phi_{> L_{\text{fork}}}(h_{\text{merge}})$$.

**채널 방향(channel-wise) 모델.** 일부 AV-LLM은 비디오와 오디오 특징을 같은 토큰 $$[V;A]$$에 넣기 때문에, 가중치를 줄 별도의 영역이 없다. 이런 모델에 대해 FMD는 AV 영역에서는 두 분기를 더하고 텍스트 영역에서는 평균을 취한다.

## 주요 실험 (Main Experiments)

FMD는 저자들이 실행할 수 있었던 모든 벤치마크에서 세 백본 모두를 향상시킨다. 정확도는 %이며, AV Cap.은 점수이다. Qwen2.5-Omni는 마지막 세 벤치마크에서 메모리 부족이 발생했다.

| 모델 | 디코딩 | A→V | V→A | AV Mat. | AV Cap. | AVQA | MUSIC-AVQA | AVUT | WorldSense |
|---|---|---|---|---|---|---|---|---|---|
| VideoLLaMA2 | Vanilla | 80.02 | 77.03 | 57.75 | 2.84 | 60.23 | 81.30 | 44.44 | 29.41 |
| | **FMD** | **80.45** | **77.52** | **59.01** | **2.95** | **61.46** | **81.50** | **45.13** | **29.91** |
| video-SALMONN | Vanilla | 68.69 | 62.39 | 49.46 | 1.83 | 28.20 | 44.48 | 32.36 | 56.20 |
| | **FMD** | **70.51** | **65.41** | **54.77** | **2.01** | **40.46** | **50.60** | **32.85** | **57.83** |
| Qwen2.5-Omni | Vanilla | 80.77 | 71.20 | 77.45 | 3.25 | 86.49 | N/A | N/A | N/A |
| | **FMD** | **81.30** | **71.77** | **78.22** | **3.36** | **86.61** | N/A | N/A | N/A |

VideoLLaMA2에서 학습이 필요 없는 다른 디코딩 방법들과 비교하면(AVHBench A→V / V→A / AV Mat.), LLM이나 VLM용으로 설계된 방법들은 모두 vanilla보다 낮다: DoLa 69.34 / 63.44 / 48.33, VCD 75.96 / 69.67 / 52.52, SID 78.53 / 72.82 / 53.52. FMD는 80.45 / 77.52 / 59.01을 기록한다.

| 디코딩 | 지연 시간 (s/token) | 상대값 |
|---|---|---|
| Vanilla | 0.34 | 1.00× |
| SID | 0.71 | 2.09× |
| VCD | 0.69 | 2.03× |
| DoLa | 0.54 | 1.59× |
| **FMD** | **0.43** | **1.26×** |

- video-SALMONN에서는 향상 폭이 크지만(AVQA +12.3, MUSIC-AVQA +6.1), VideoLLaMA2와 Qwen2.5-Omni에서는 대부분 1.5점 미만이다.
- 병합 규칙이 중요하다. 단순 평균을 쓰면 V→A 정확도가 55.31로 무너진다.

## 한계 (Limitations)

- **분기 깊이는 타협의 산물이다.** $$L_{\text{fork}}$$ 탐색에서 A→V 질문은 일찍 병합할 때 가장 좋고, V→A 질문은 더 깊이 분기할수록 좋아지며, AV 매칭 정확도는 분기가 깊어질수록 떨어진다. 하나의 고정값이 모든 질문 유형에 최적일 수는 없다.
- **강한 백본에서는 향상이 작다.** VideoLLaMA2와 Qwen2.5-Omni에서는 대부분의 향상이 1.5점 미만이다.
- **환각이 아니라 불균형을 다룬다.** FMD는 모달리티 사용의 균형을 다시 맞춘다. 대조 방법인 [AVCD](/blog/paper-review/ko/avcd-review/)나 [MAD](/blog/paper-review/ko/mad-review/)와 달리, 환각된 답을 억제하는 명시적 메커니즘은 없다.

## 내 생각 (My Thoughts)

<!-- TODO -->

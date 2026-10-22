---
title: "Fork-Merge Decoding: Let Audio and Video Think Separately First"
category: paper-review
topic: mllm
lang: en
description: Run audio-only and video-only branches through the early decoder layers, merge their hidden states, and decode jointly from there.
paper:
  title: "Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visual Large Language Models"
  authors: "Chaeyoung Jung, Youngjoon Jang, Jongmin Choi, Joon Son Chung"
  venue: "EMNLP 2026 Findings"
  url: "https://arxiv.org/abs/2505.20873"
---

## TL;DR

When an AV-LLM sees audio and video together from the first layer, one modality, usually video, takes over. Fork-Merge Decoding (FMD) **forks** the input into a video-masked and an audio-masked copy. It runs each copy through the first $$L_{\text{fork}}$$ decoder layers on its own, **merges** the two sets of hidden states, and runs the remaining layers once on the merged state. It needs no training and is cheaper than contrastive decoding, because only the early layers run twice.

## Problem

AV-LLMs feed audio and video into the decoder together. Joint processing lets the model reason over both, but it does not guarantee that the model *uses* both in a balanced way.

- The model can over-rely on one modality (**modality bias**).
- In vanilla decoding, attention to video is much stronger than attention to audio. With FMD, average video attention falls from about 0.228 to 0.211 and audio attention rises from about 0.041 to 0.046.
- That imbalance leads to hallucination and false reasoning.

## Method

### Preliminaries: two ways to feed audio and video

**Token-wise fusion.** Video, audio, and text tokens are concatenated along the sequence:

$$
x = x^v \oplus x^a \oplus x^l, \qquad
x^v = \{x_1, \dots, x_M\},\;
x^a = \{x_{M+1}, \dots, x_{M+N}\},\;
x^l = \{x_{M+N+1}, \dots, x_{M+N+L}\}
$$

$$
[V_1, \dots, V_M \,;\, A_1, \dots, A_N \,;\, Q_1, \dots, Q_L]
$$

**Channel-wise fusion.** Video and audio are projected to the same length $$U$$, and the features at each position $$i$$ are concatenated along the channel dimension:

$$
x^{av}_i = [x^v_i ; x^a_i], \qquad
x = \{x^{av}_1, \dots, x^{av}_U, x^l_{U+1}, \dots, x^l_{U+L}\}
$$

So the token-wise sequence looks like $$[V][V][V][A][A][A][Q][Q]$$, and the channel-wise one looks like $$[V;A][V;A][V;A][Q][Q]$$.

In both cases vanilla decoding processes audio and video jointly from the very first layer, $$A + V \to \phi_1 \to \phi_2 \to \dots \to \phi_L$$. FMD changes that to "separate first, joint later":

$$
\underbrace{A \parallel V}_{\text{early: independent}} \;\to\; \underbrace{A + V}_{\text{later: joint}} \;\to\; \text{output}
$$

### Fork (token-wise)

Make two inputs, each with one modality masked. The question stays intact in both:

$$
x_{v[\text{masked}]} = x^{\neg v} \oplus x^a \oplus x^l \approx (\cancel{V}, A, Q), \qquad
x_{a[\text{masked}]} = x^v \oplus x^{\neg a} \oplus x^l \approx (V, \cancel{A}, Q)
$$

Masking does **not** delete tokens from the sequence. It zeroes out the raw video frames or audio waveform before the encoder. The shape and positions stay the same, and only that modality's content is removed.

Write the decoder as $$\phi$$. Each masked input goes through only the first $$L_{\text{fork}}$$ layers:

$$
h^{\neg v}_{\text{fork}} = \phi_{\le L_{\text{fork}}}\big(x_{v[\text{masked}]}\big), \qquad
h^{\neg a}_{\text{fork}} = \phi_{\le L_{\text{fork}}}\big(x_{a[\text{masked}]}\big)
$$

- $$h^{\neg v}_{\text{fork}}$$ is the video-masked branch, so it carries audio and text information.
- $$h^{\neg a}_{\text{fork}}$$ is the audio-masked branch, so it carries video and text information.

The point of the fork is that **in the early layers the model never sees A and V at the same time**, so neither can drown out the other.

### Merge (token-wise)

Let $$\mathcal{V} = [1:M]$$, $$\mathcal{A} = [M+1 : M+N]$$, and $$\mathcal{L} = [M+N+1 : M+N+L]$$ be the positions of the video, audio, and language tokens. Each region is merged separately:

$$
\begin{aligned}
h_{\text{merge}}[\mathcal{V}] &= (1-\alpha)\, h^{\neg v}_{\text{fork}}[\mathcal{V}] + \alpha\, h^{\neg a}_{\text{fork}}[\mathcal{V}] \\
h_{\text{merge}}[\mathcal{A}] &= \alpha\, h^{\neg v}_{\text{fork}}[\mathcal{A}] + (1-\alpha)\, h^{\neg a}_{\text{fork}}[\mathcal{A}] \\
h_{\text{merge}}[\mathcal{L}] &= \tfrac{1}{2}\big(h^{\neg v}_{\text{fork}}[\mathcal{L}] + h^{\neg a}_{\text{fork}}[\mathcal{L}]\big)
\end{aligned}
$$

$$\alpha$$ is the weight of the branch in which that position's modality was **not** masked. At video positions, a larger $$\alpha$$ puts more weight on the branch that actually saw the video ($$h^{\neg a}$$). At audio positions, a larger $$\alpha$$ puts more weight on the branch that actually heard the audio ($$h^{\neg v}$$). Both branches kept the text, so language positions are averaged equally.

### Attention-guided fusion: choosing α

$$\alpha$$ comes from the attention of the last token. Let $$A_{\text{final}} \in \mathbb{R}^{T \times T}$$ be the final attention map and $$a_{\text{last}} = A_{\text{final}}[T, :]$$ its last row. Each branch has its own: $$a^{v[\text{masked}]}_{\text{last}}$$ and $$a^{a[\text{masked}]}_{\text{last}}$$.

$$\alpha$$ measures how much attention each branch gives to the modality it can still see, relative to its total attention on audio and video:

$$
\alpha = \frac{\sum_{i \in \mathcal{A}} a^{v[\text{masked}]}_{\text{last}}[i] + \sum_{i \in \mathcal{V}} a^{a[\text{masked}]}_{\text{last}}[i]}
{\sum_{i \in \mathcal{V} \cup \mathcal{A}} \big(a^{v[\text{masked}]}_{\text{last}}[i] + a^{a[\text{masked}]}_{\text{last}}[i]\big)}
$$

### Continue decoding

Layers $$1, \dots, L_{\text{fork}}$$ run as two forks. Layers $$L_{\text{fork}}+1, \dots, L$$ process the single merged representation:

$$
h_{\text{final}} = \phi_{> L_{\text{fork}}}\big(h_{\text{merge}}\big)
$$

### Channel-wise variant

With channel-wise fusion, each AV token already contains $$[V;A]$$, so there is no "video region" or "audio region" to weight separately. The fork is the same: mask the video half or the audio half of each AV token, giving $$x_{v[\text{masked}]} = x^{a\neg v} \oplus x^l$$ and $$x_{a[\text{masked}]} = x^{\neg a v} \oplus x^l$$. The merge adds the AV region and averages the text region:

$$
h_{\text{merge}}[i] =
\begin{cases}
h^{\neg v}_{\text{fork}}[i] + h^{\neg a}_{\text{fork}}[i], & i \le U \\[4pt]
\tfrac{1}{2}\big(h^{\neg v}_{\text{fork}}[i] + h^{\neg a}_{\text{fork}}[i]\big), & i > U
\end{cases}
$$

The audio-only and video-only information is added back together in the AV region. The duplicated text is averaged.

**In short:** token-wise models use an attention-guided weighted merge. Channel-wise models use addition in the AV region and averaging in the text region.

## Experiments

**Main results.** FMD improves all three backbones on every benchmark they could run. Accuracy is in %, and AV Cap. is a score. Qwen2.5-Omni ran out of memory on the last three benchmarks.

| Model | Decoding | A→V | V→A | AV Mat. | AV Cap. | AVQA | MUSIC-AVQA | AVUT | WorldSense |
|---|---|---|---|---|---|---|---|---|---|
| VideoLLaMA2 | Vanilla | 80.02 | 77.03 | 57.75 | 2.84 | 60.23 | 81.30 | 44.44 | 29.41 |
| | **FMD** | **80.45** | **77.52** | **59.01** | **2.95** | **61.46** | **81.50** | **45.13** | **29.91** |
| video-SALMONN | Vanilla | 68.69 | 62.39 | 49.46 | 1.83 | 28.20 | 44.48 | 32.36 | 56.20 |
| | **FMD** | **70.51** | **65.41** | **54.77** | **2.01** | **40.46** | **50.60** | **32.85** | **57.83** |
| Qwen2.5-Omni | Vanilla | 80.77 | 71.20 | 77.45 | 3.25 | 86.49 | N/A | N/A | N/A |
| | **FMD** | **81.30** | **71.77** | **78.22** | **3.36** | **86.61** | N/A | N/A | N/A |

The gains on video-SALMONN are much larger (AVQA +12.3, MUSIC-AVQA +6.1) than on VideoLLaMA2 and Qwen2.5-Omni (mostly under 1.5 points).

**Against other training-free decoding methods** (VideoLLaMA2, AVHBench). Methods designed for LLMs or VLMs all fall *below* vanilla on AV-LLMs.

| Decoding | Designed for | A→V | V→A | AV Mat. |
|---|---|---|---|---|
| Vanilla | – | 80.02 | 77.03 | 57.75 |
| DoLa | LLM | 69.34 | 63.44 | 48.33 |
| VCD | VLM | 75.96 | 69.67 | 52.52 |
| SID | VLM | 78.53 | 72.82 | 53.52 |
| FMD w/ noise masking | AV-LLM | 79.17 | **78.03** | 57.76 |
| **FMD w/ zero-out** | AV-LLM | **80.45** | 77.52 | **59.01** |

**Merge strategy** (VideoLLaMA2, AVHBench). Plain averaging collapses on V→A (55.31). The attention-guided merge is what makes FMD work. The paper reports two variants of it, adaptive and fixed.

| Merge | A→V | V→A | AV Mat. |
|---|---|---|---|
| Vanilla | 80.02 | 77.03 | 57.75 |
| Exclusion | 79.38 | 76.79 | 54.83 |
| Average | 75.11 | 55.31 | 57.16 |
| Attention (adaptive) | 79.49 | 72.78 | **59.19** |
| **Attention (fixed)** | **80.45** | **77.52** | 59.01 |

**Where to fork.** The best $$L_{\text{fork}}$$ depends on the question type. A→V questions do best with an early merge, V→A questions improve as the fork gets deeper, and AV-matching accuracy falls steadily as $$L_{\text{fork}}$$ grows. No single depth is best for everything.

**Speed.** Only the first $$L_{\text{fork}}$$ layers run twice, so FMD is the cheapest of the training-free methods.

| Decoding | Latency (s/token) | Relative |
|---|---|---|
| Vanilla | 0.34 | 1.00× |
| SID | 0.71 | 2.09× |
| VCD | 0.69 | 2.03× |
| DoLa | 0.54 | 1.59× |
| **FMD** | **0.43** | **1.26×** |

## Thoughts

- FMD comes from the same group as [AVCD]({% post_url 2026-10-22-avcd-review %}) and takes the opposite route. AVCD works on **output logits** and needs four full forward passes per token. FMD works on **hidden states**, so the extra cost is only in the early layers.
- AVCD and [MAD]({% post_url 2026-10-22-mad-review %}) both subtract a masked branch. FMD never subtracts anything. It keeps both unimodal views and combines them, so it targets modality *imbalance* more than hallucination in the narrow sense.
- The $$L_{\text{fork}}$$ sweep is the weak spot: the best depth for A→V hurts V→A, and the reverse. The single fixed value is a compromise across question types.

<!-- TODO: add your own thoughts -->

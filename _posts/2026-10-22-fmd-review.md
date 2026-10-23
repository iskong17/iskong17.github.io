---
title: "Fork-Merge Decoding: Let Audio and Video Think Separately First"
category: paper-review
subfields: [avllm]
lang: en
description: Run audio-only and video-only branches through the early decoder layers, merge their hidden states, and decode jointly from there.
conference: EMNLP
year: 2026
paper:
  title: "Fork-Merge Decoding: Enhancing Multimodal Understanding in Audio-Visual Large Language Models"
  authors: "Chaeyoung Jung, Youngjoon Jang, Jongmin Choi, Joon Son Chung"
  venue: "EMNLP 2026 Findings"
  url: "https://arxiv.org/abs/2505.20873"
---

## Abstract

AV-LLMs feed audio and video into the decoder together. Joint processing lets the model reason over both, but it does not guarantee balanced use of both. The model can over-rely on one modality (**modality bias**). In vanilla decoding, attention to video is much stronger than attention to audio, and that imbalance leads to hallucination and false reasoning.

Fork-Merge Decoding (FMD) is a training-free fix. It **forks** the input into a video-masked and an audio-masked copy, runs each through the first $$L_{\text{fork}}$$ decoder layers on its own, **merges** the hidden states, and runs the remaining layers once. Only the early layers run twice, so it is cheaper than contrastive decoding.

## Motivation

The paper's evidence for modality bias is an attention probe. On 100 AVHBench samples, the authors read the last decoder layer of VideoLLaMA2 and look at how much attention the final question token puts on video tokens versus audio tokens. With vanilla decoding, video gets about 0.23 of the attention and audio about 0.04, more than five times less. Earlier work on MLLMs links this kind of imbalance to hallucination and flawed reasoning, so the question is how to give the weaker modality a fair hearing without retraining the model.

![Attention to video and audio](/images/blog/papers/fmd-motivation-attention.webp)
*Last-layer attention of the final question token to video (left) and audio (right) tokens in VideoLLaMA2 on 100 AVHBench samples. Vanilla decoding favors video by a wide margin; FMD lowers video attention and raises audio attention. Note the different y-axis ranges. Figure from the paper.*

The second observation decides where to split. Following prior work on LLM layer structure, the authors compute the cosine similarity between hidden states of all layer pairs in VideoLLaMA2 and find the same three blocks: a small early block (layers 1 to 5), a large middle block, and a late block. Early layers are where each modality's features are aligned and encoded on their own, and cross-modal fusion mostly happens in the middle. So letting audio and video pass the early block separately, then fusing before the middle block, should strengthen each modality's own representation without changing how the model fuses them.

![Layer-wise hidden state similarity](/images/blog/papers/fmd-layer-similarity.webp)
*Cosine similarity between hidden states of every pair of decoder layers in VideoLLaMA2. The layers group into early, middle and late blocks; FMD picks the fork depth inside the early block. Figure from the paper.*

## Method

![FMD overview](/images/blog/papers/fmd-main.png)
*Fork: two masked inputs pass through the early decoder layers separately. Merge: their hidden states are fused per modality region with attention-guided weights, and the rest of the decoder runs once. Figure from the paper.*

### Fork

Each copy of the input has one modality masked, and the question stays intact in both:

$$
x_{v[\text{masked}]} = x^{\neg v} \oplus x^a \oplus x^l, \qquad
x_{a[\text{masked}]} = x^v \oplus x^{\neg a} \oplus x^l
$$

Masking zeroes the raw video frames or audio waveform before the encoder, so the sequence shape and positions stay the same. With decoder $$\phi$$, each copy goes through only the first $$L_{\text{fork}}$$ layers:

$$
h^{\neg v}_{\text{fork}} = \phi_{\le L_{\text{fork}}}\big(x_{v[\text{masked}]}\big), \qquad
h^{\neg a}_{\text{fork}} = \phi_{\le L_{\text{fork}}}\big(x_{a[\text{masked}]}\big)
$$

$$h^{\neg v}$$ carries audio and text, and $$h^{\neg a}$$ carries video and text. In the early layers the model never sees audio and video at the same time, so neither can drown out the other.

### Merge

The hidden states are merged separately over the video ($$\mathcal{V}$$), audio ($$\mathcal{A}$$), and language ($$\mathcal{L}$$) token positions:

$$
\begin{aligned}
h_{\text{merge}}[\mathcal{V}] &= (1-\alpha)\, h^{\neg v}_{\text{fork}}[\mathcal{V}] + \alpha\, h^{\neg a}_{\text{fork}}[\mathcal{V}] \\
h_{\text{merge}}[\mathcal{A}] &= \alpha\, h^{\neg v}_{\text{fork}}[\mathcal{A}] + (1-\alpha)\, h^{\neg a}_{\text{fork}}[\mathcal{A}] \\
h_{\text{merge}}[\mathcal{L}] &= \tfrac{1}{2}\big(h^{\neg v}_{\text{fork}}[\mathcal{L}] + h^{\neg a}_{\text{fork}}[\mathcal{L}]\big)
\end{aligned}
$$

$$\alpha$$ is the weight of the branch that actually saw the modality at that position. Language positions are averaged, because both branches kept the text.

**Attention-guided $$\alpha$$.** $$\alpha$$ is the share of last-token attention that each branch gives to the modality it can still see:

$$
\alpha = \frac{\sum_{i \in \mathcal{A}} a^{v[\text{masked}]}_{\text{last}}[i] + \sum_{i \in \mathcal{V}} a^{a[\text{masked}]}_{\text{last}}[i]}
{\sum_{i \in \mathcal{V} \cup \mathcal{A}} \big(a^{v[\text{masked}]}_{\text{last}}[i] + a^{a[\text{masked}]}_{\text{last}}[i]\big)}
$$

The remaining layers then run once on the merged state: $$h_{\text{final}} = \phi_{> L_{\text{fork}}}(h_{\text{merge}})$$.

**Channel-wise models.** Some AV-LLMs put video and audio features in the same token, $$[V;A]$$, so there are no separate regions to weight. For these models FMD adds the two branches in the AV region and averages them in the text region.

## Main Experiments

FMD improves all three backbones on every benchmark the authors could run. Accuracy is in %, and AV Cap. is a score. Qwen2.5-Omni ran out of memory on the last three benchmarks.

| Model | Decoding | A→V | V→A | AV Mat. | AV Cap. | AVQA | MUSIC-AVQA | AVUT | WorldSense |
|---|---|---|---|---|---|---|---|---|---|
| VideoLLaMA2 | Vanilla | 80.02 | 77.03 | 57.75 | 2.84 | 60.23 | 81.30 | 44.44 | 29.41 |
| | **FMD** | **80.45** | **77.52** | **59.01** | **2.95** | **61.46** | **81.50** | **45.13** | **29.91** |
| video-SALMONN | Vanilla | 68.69 | 62.39 | 49.46 | 1.83 | 28.20 | 44.48 | 32.36 | 56.20 |
| | **FMD** | **70.51** | **65.41** | **54.77** | **2.01** | **40.46** | **50.60** | **32.85** | **57.83** |
| Qwen2.5-Omni | Vanilla | 80.77 | 71.20 | 77.45 | 3.25 | 86.49 | N/A | N/A | N/A |
| | **FMD** | **81.30** | **71.77** | **78.22** | **3.36** | **86.61** | N/A | N/A | N/A |

Compared with other training-free decoding methods on VideoLLaMA2 (AVHBench A→V / V→A / AV Mat.), methods designed for LLMs or VLMs all fall below vanilla: DoLa 69.34 / 63.44 / 48.33, VCD 75.96 / 69.67 / 52.52, SID 78.53 / 72.82 / 53.52. FMD gets 80.45 / 77.52 / 59.01.

| Decoding | Latency (s/token) | Relative |
|---|---|---|
| Vanilla | 0.34 | 1.00× |
| SID | 0.71 | 2.09× |
| VCD | 0.69 | 2.03× |
| DoLa | 0.54 | 1.59× |
| **FMD** | **0.43** | **1.26×** |

- The gains are large on video-SALMONN (AVQA +12.3, MUSIC-AVQA +6.1) but mostly under 1.5 points on VideoLLaMA2 and Qwen2.5-Omni.
- The merge rule matters. Plain averaging collapses V→A accuracy to 55.31.

In the qualitative example below, vanilla decoding describes only what it sees, while FMD also reports the sound of the stream and the splashes, which is the kind of audio detail the attention probe suggests is being ignored.

![FMD qualitative example](/images/blog/papers/fmd-qualitative.webp)
*An audio-visual description example with VideoLLaMA2. Vanilla decoding gives a video-only caption; FMD adds the audio content (the stream and the occasional splash). Figure from the paper.*

The fork-depth sweep on 200 AVHBench samples per task shows the trade-off behind the chosen $$L_{\text{fork}}$$. Merging early keeps A→V and AV-matching accuracy at or above vanilla, while forking deeper helps V→A (audio-targeted) questions but hurts the other two sharply after about layer 15.

![Fork layer ablation](/images/blog/papers/fmd-fork-layer-ablation.webp)
*Accuracy of VideoLLaMA2 with FMD as a function of $$L_{\text{fork}}$$ for A→V, V→A and AV-matching questions (200 AVHBench samples each; dashed line = vanilla). No single depth is best for all three, and the paper picks an early one. Figure from the paper.*

## Limitations

- **The fork depth is a compromise.** In the $$L_{\text{fork}}$$ sweep, A→V questions do best with an early merge, V→A questions improve with a deeper fork, and AV-matching accuracy falls as the fork gets deeper. One fixed value cannot be best for all question types.
- **Small gains on strong backbones.** On VideoLLaMA2 and Qwen2.5-Omni most improvements are under 1.5 points.
- **Imbalance, not hallucination.** FMD rebalances how the modalities are used. It has no explicit mechanism to suppress a hallucinated answer, unlike the contrastive methods [AVCD]({% post_url 2026-10-22-avcd-review %}) and [MAD]({% post_url 2026-10-22-mad-review %}).

## My Thoughts

<!-- TODO -->

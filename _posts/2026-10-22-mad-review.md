---
title: "MAD: Ask the Model Which Modality the Question Needs"
category: paper-review
subfields: [hallucination, avllm]
lang: en
description: Ask the model whether a question needs audio, video, or both, and use the answer to weight each contrastive decoding branch.
conference: CVPR
year: 2026
paper:
  title: "MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models"
  authors: "Sangyun Chung, Se Yeon Kim, Youngchae Chee, Yong Man Ro"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2601.21181"
---

## Abstract

Multimodal LLMs suffer from **cross-modal hallucination**: one modality interferes with generation about another, such as a video making the model "hear" a sound that is not in the audio. The authors argue this is not a representation problem inside each modality. The model fails to control how the modalities influence each other. The question to answer is: *for this question, how much should the model trust audio, and how much video?*

Existing contrastive decoding (CD) methods use a fixed contrastive strength for every question. MAD asks the model one extra question: does this question need audio, video, or both? It turns the answer into per-branch weights in a four-branch CD formula. It needs no training.

## Method

![MAD overview](/images/blog/papers/mad-main.png)
*MAD in two steps. Step 1 extracts modality weights from a modality query prompt. Step 2 combines four branch logits with those weights. Figure from the paper.*

### Weighted contrastive decoding

For modality $$m$$, the contrastive signal is the logit with $$X_m$$ minus the logit with a degraded $$\tilde X_m$$, for example $$\Delta_a = z(A) - z(\tilde A)$$. MAD splits the contrastive strength into two factors:

$$
\mathrm{logit}^{(m)}(y_t) = \mathrm{logit}(y_t \mid X_m, X_q, y_{<t}) + \gamma\, w_m \Delta_m
$$

Here $$\gamma$$ is the global strength and $$w_m$$ is how important modality $$m$$ is **for this question**.

### Step 1: Modality-adaptive weight extraction (top)

The model gets the input plus a modality query prompt $$X_m$$ asking which modality the question needs. It does not generate an answer. MAD reads the next-token logits of three candidate words and applies a softmax:

$$
[w_{av}, w_v, w_a] = \mathrm{softmax}\big([z_{\text{both}}, z_{\text{video}}, z_{\text{audio}}]\big)
$$

The weights follow the question type. Averaged over the benchmark, visual questions get $$w_v = 0.565$$, audio questions get $$w_a = 0.482$$, and audio-visual questions get $$w_{av} = 0.464$$.

### Step 2: Modality-adaptive generation (bottom)

Like AVCD, MAD uses four branches: audio-visual $$z_{VA}$$, video-only $$z_{V\tilde A}$$, audio-only $$z_{\tilde V A}$$, and question-only $$z_{\tilde V \tilde A}$$. They form a square, and each edge is one CD term: visual CD with audio present or absent, and audio CD with video present or absent. The terms taken with the other modality present use $$w_{av}$$, and the terms taken with it absent use $$w_v$$ or $$w_a$$. Summing the four terms and collecting coefficients gives Eq. (9) in the figure:

$$
z_{\text{MAD}} = (2 + 2\gamma w_{av})\, z_{VA} + (1 - \gamma w_{av} + \gamma w_a)\, z_{\tilde V A} + (1 - \gamma w_{av} + \gamma w_v)\, z_{V \tilde A} - \gamma (w_v + w_a)\, z_{\tilde V \tilde A}
$$

In the figure's example, the question asks about a hammer sound. The weights favour audio, the audio-only branch says "No", and MAD flips the original "Yes" to the correct "No".

## Main Experiments

CMM reports accuracy on visual-dominant, audio-dominant, and language-dominant subsets. AVHBench reports accuracy on video-driven audio hallucination and audio-driven video hallucination.

| Model | Visual Dom. | Audio Dom. | Lang. Dom. | CMM Overall | V→A Hall. | A→V Hall. | AVHBench Overall |
|---|---|---|---|---|---|---|---|
| VideoLLaMA2-AV | 71.8 | 80.0 | 68.8 | 73.5 | 75.7 | 79.0 | 77.4 |
| + VCD (extended) | 71.3 | 83.3 | 74.8 | 76.4 | 66.0 | 74.8 | 70.4 |
| + AVCD | 71.8 | 84.0 | 71.5 | 75.8 | 78.3 | **80.3** | 79.3 |
| **+ MAD** | **82.3** | **84.3** | **77.5** | **81.3** | **79.7** | 79.1 | **79.4** |
| Qwen2.5-Omni-7B | 64.5 | 72.3 | 81.3 | 72.7 | 73.0 | 80.7 | 76.9 |
| + VCD (extended) | 62.5 | 71.3 | **84.5** | 72.8 | 70.3 | 77.1 | 73.7 |
| + AVCD | 66.3 | 72.8 | 81.0 | 73.3 | 75.8 | 79.7 | 77.8 |
| **+ MAD** | **76.8** | **84.3** | 83.3 | **81.4** | **78.7** | **84.4** | **81.6** |

- The biggest jump is on CMM visual-dominant questions: +10.5 for VideoLLaMA2-AV and +12.3 for Qwen2.5-Omni. AVCD barely moves that column.
- Weighting by the full softmax (81.3) beats equal weights (79.4) and putting all weight on the argmax (78.7).
- MAD does not hurt general audio-visual understanding on VideoLLaMA2-AV: OmniBench 36.3 → 36.8, WorldSense 23.3 → 25.6, MUSIC-AVQA 78.1 → 79.1.

## Limitations

- **Cost.** Each token needs the four branches, plus one extra pass with the modality query prompt.
- **Relevance is not agreement.** The weights say which modality is *relevant*, not whether the unimodal branches agree with the joint prediction. Two relevant modalities can still conflict, and MAD cannot detect that.
- **The self-assessment must be right.** If the model misjudges which modality a question needs, the weights push the decoding the wrong way.

## My Thoughts

<!-- TODO -->

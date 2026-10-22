---
title: "MAD: Ask the Model Which Modality the Question Needs"
category: paper-review
topic: mllm
lang: en
description: Ask the model whether a question needs audio, video, or both, and use the answer to weight each contrastive decoding branch.
paper:
  title: "MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models"
  authors: "Sangyun Chung, Se Yeon Kim, Youngchae Chee, Yong Man Ro"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2601.21181"
---

## TL;DR

Cross-modal hallucination happens when one modality interferes with generation about another, for example when a video of a fishing boat makes the model "hear" a fish splashing. MAD asks the model one extra question: does this question need audio, video, or both? The softmax over those three answers becomes per-branch weights in a four-branch contrastive decoding formula. Contrastive decoding (CD) strength then adapts to each question instead of staying fixed. It needs no training.

## Problem

- Multimodal LLMs suffer from **cross-modal hallucination**: one modality disturbs generation about another.
- The authors argue this is not a representation problem inside each modality. The model fails to **control how the modalities influence each other**.
- So the real question is: *to answer this question, how much should the model trust audio, and how much video?*

Existing CD methods use a fixed contrastive strength $$\alpha_m$$ for every task, whether the question is about sound or about what is on screen.

## Method

### Preliminaries

The inputs are video, audio, and question tokens:

$$
X_v = \{x^v_1, \dots, x^v_{n_v}\}, \quad
X_a = \{x^a_1, \dots, x^a_{n_a}\}, \quad
X_q = \{x^q_1, \dots, x^q_{n_q}\}
$$

$$
y_t \sim p_\theta(y_t \mid X_v, X_a, X_q, y_{<t}) \propto \mathrm{logit}_\theta(y_t \mid X_v, X_a, X_q, y_{<t})
$$

Contrastive decoding for VLMs, written in terms of a clean and a degraded logit:

$$
z_{\text{CD}} = (1+\alpha)\, z_{\text{clean}} - \alpha\, z_{\text{degraded}}
= z_{\text{clean}} + \alpha\,\underbrace{(z_{\text{clean}} - z_{\text{degraded}})}_{\text{contrastive signal}}
$$

In words, take the original logit and add $$\alpha$$ times the difference that the visual input makes. With $$\alpha = 0$$ it is plain decoding.

### Weighted contrastive decoding

For a modality $$m$$, define the contrastive signal $$\Delta_m$$ as the logit with $$X_m$$ minus the logit with a degraded $$\tilde X_m$$, so that $$\Delta_a = z(A) - z(\tilde A)$$ and $$\Delta_v = z(V) - z(\tilde V)$$:

$$
\mathrm{logit}^{(m)}_{\text{CD}}(y_t) = \mathrm{logit}(y_t \mid X_m, X_q, y_{<t}) + \alpha_m \Delta_m
$$

MAD splits the strength into two factors:

$$
\alpha_m = \gamma \cdot w_m
$$

- $$\gamma$$ is the global contrastive strength.
- $$w_m$$ is how important modality $$m$$ is **for this question**.

If $$w_a$$ is large, the term $$\gamma w_a \Delta_a$$ is large, so the audio contrast counts more.

### Step 1: Extract modality-adaptive weights

The model gets the input plus a **modality query prompt** $$X_m$$ asking which modality the question needs. It does not generate a full answer. MAD reads the next-token logits for three candidate words:

$$
z_{av} = \mathrm{logit}(\text{both} \mid X_v, X_a, X_q, X_m), \quad
z_v = \mathrm{logit}(\text{video} \mid \cdot), \quad
z_a = \mathrm{logit}(\text{audio} \mid \cdot)
$$

$$
[w_{av}, w_v, w_a] = \mathrm{softmax}([z_{av}, z_v, z_a]), \qquad w_{av} + w_v + w_a = 1
$$

For an audio question this might give $$[w_{av}, w_v, w_a] = [0.10, 0.05, 0.85]$$. Averaged over the benchmark, the weights do follow the question type:

| Question type | $$w_v$$ | $$w_a$$ | $$w_{av}$$ |
|---|---|---|---|
| Visual-related | **0.565** | 0.106 | 0.328 |
| Audio-related | 0.237 | **0.482** | 0.279 |
| Audio-visual-related | 0.297 | 0.237 | **0.464** |

### Step 2: Modality-adaptive generation

Write $$z_{m_1 m_2 \dots}$$ for the logit given those modalities, with a tilde for a degraded one. As in AVCD, there are four branches: $$VA,\ \tilde V A,\ V\tilde A,\ \tilde V \tilde A$$. They form a square: removing video moves horizontally, removing audio moves vertically.

$$
\begin{array}{ccc}
(V, A) & \xrightarrow{\text{remove video}} & (\tilde V, A) \\
\downarrow \text{remove audio} & & \downarrow \text{remove audio} \\
(V, \tilde A) & \xrightarrow{\text{remove video}} & (\tilde V, \tilde A)
\end{array}
$$

Each edge of the square is one CD term. MAD adds all four:

$$
\begin{aligned}
\mathrm{logit}(y_t) =\;& \underbrace{(1+\alpha_{av}) z_{VA} - \alpha_{av} z_{\tilde V A}}_{\text{visual CD, audio present}}
+ \underbrace{(1+\alpha_{av}) z_{VA} - \alpha_{av} z_{V \tilde A}}_{\text{audio CD, video present}} \\
&+ \underbrace{(1+\alpha_{v}) z_{V\tilde A} - \alpha_{v} z_{\tilde V \tilde A}}_{\text{visual CD, audio absent}}
+ \underbrace{(1+\alpha_{a}) z_{\tilde V A} - \alpha_{a} z_{\tilde V \tilde A}}_{\text{audio CD, video absent}}
\end{aligned}
$$

The two CD terms taken with the other modality present use $$\alpha_{av}$$. The two taken with the other modality absent use the unimodal weights. Substituting $$\alpha_{av} = \gamma w_{av}$$, $$\alpha_v = \gamma w_v$$, $$\alpha_a = \gamma w_a$$ and collecting coefficients:

$$
\begin{aligned}
z_{\text{MAD}} =\;& (2 + 2\gamma w_{av})\, z_{VA}
+ (1 - \gamma w_{av} + \gamma w_a)\, z_{\tilde V A} \\
&+ (1 - \gamma w_{av} + \gamma w_v)\, z_{V \tilde A}
- \gamma (w_v + w_a)\, z_{\tilde V \tilde A}
\end{aligned}
$$

### Pipeline

**Question → self-assessment → $$(w_{av}, w_v, w_a)$$ → adaptive CD coefficients → $$z_{\text{MAD}}$$**

## Experiments

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

The biggest jump is on CMM visual-dominant questions: +10.5 for VideoLLaMA2-AV and +12.3 for Qwen2.5-Omni. AVCD barely moves that column.

**How to fuse the weights** (VideoLLaMA2-AV, CMM). Weighting by the full softmax beats using equal weights or putting all weight on the argmax.

| Fusion | Visual Dom. | Audio Dom. | Lang. Dom. | Overall |
|---|---|---|---|---|
| Baseline | 71.8 | 80.0 | 68.8 | 73.5 |
| Uniform | 77.5 | 83.3 | **77.5** | 79.4 |
| Argmax | 78.5 | 80.5 | 77.0 | 78.7 |
| **Weighted** | **82.3** | **84.3** | **77.5** | **81.3** |

**Which weights matter.** Dropping any one of $$w_a, w_v, w_{av}$$ costs 2.4–3.3 points overall (78.0, 78.3, and 78.9 vs. 81.3).

**General benchmarks.** On VideoLLaMA2-AV, MAD does not hurt general audio-visual understanding: OmniBench 36.3 → 36.8, WorldSense 23.3 → 25.6, MUSIC-AVQA 78.1 → 79.1.

## Thoughts

- Compared with [AVCD]({% post_url 2026-10-22-avcd-review %}), the branch structure is the same four logits. What changes is **where the weights come from**. AVCD uses attention dominance, which says what the model *is* looking at. MAD uses a self-assessment prompt, which says what the question *needs*. The visual-dominant column suggests that this difference matters a lot.
- The self-assessment costs one extra forward pass with the modality query prompt, on top of the four branches.
- The weights only say which modality is *relevant*. They do not say whether the audio-only and video-only branches agree with the joint prediction. Two relevant modalities can still conflict, and MAD has no way to detect that.

<!-- TODO: add your own thoughts -->

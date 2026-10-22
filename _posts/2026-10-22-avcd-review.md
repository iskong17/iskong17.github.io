---
title: "AVCD: Contrastive Decoding for Three Modalities"
category: paper-review
topic: mllm
lang: en
description: Find the dominant modality from attention, mask the others, and extend contrastive decoding to audio, video, and text.
paper:
  title: "AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding"
  authors: "Chaeyoung Jung, Youngjoon Jang, Joon Son Chung"
  venue: "NeurIPS 2025"
  url: "https://arxiv.org/abs/2505.20862"
---

## TL;DR

Contrastive decoding (CD) reduces hallucination in vision-language models, but an audio-visual LLM (AV-LLM) has three modalities, not two. AVCD measures which modality dominates the current prediction, masks the high-attention tokens of the less dominant ones, and combines four branches into a single trimodal CD formula. An entropy gate skips the extra work when the model is already confident. It needs no training.

## Problem

Hallucination is still a major issue in multimodal LLMs. CD methods such as VCD work well for VLMs: perturb the image, see what the model predicts without real visual evidence, and subtract that. An AV-LLM is harder for two reasons:

- There are three inputs (video, audio, language), so it is unclear which one to perturb.
- A hallucination can come from any of them, or from how they interact.

## Method

### Preliminaries

The video encoder, audio encoder, and tokenizer produce three token sequences:

$$
x^v = \{x_1, \dots, x_M\}, \quad
x^a = \{x_{M+1}, \dots, x_{M+N}\}, \quad
x^l = \{x_{M+N+1}, \dots, x_{M+N+L}\}
$$

They are concatenated into $$x = [x^v; x^a; x^l]$$ of length $$K = M + N + L$$, and the LLM decodes autoregressively:

$$
y_t \sim p(y_t \mid x^v, x^a, x^l, y_{<t}) \propto \exp\big(\mathrm{logit}(y_t \mid x^v, x^a, x^l, y_{<t})\big)
$$

Standard CD for a VLM uses an original branch and a perturbed branch:

$$
\mathrm{logit} = (1+\alpha)\,\mathrm{logit}(y \mid x^v, x^l) - \alpha\,\mathrm{logit}(y \mid x^{\neg v}, x^l)
= L_{\text{orig}} + \alpha\,(L_{\text{orig}} - L_{\text{corr}})
$$

**Why subtract the perturbed branch?** With the visual input corrupted, the model cannot use visual evidence, so the perturbed branch shows what the model predicts from language bias alone. Subtracting it removes that bias and amplifies the part of the prediction that actually comes from the visual evidence.

### 1. Find the dominant modality from attention

The main question is which of audio, video, and language dominates the current prediction. Let $$A^j_{Q_K, i}$$ be the attention weight that the last query token gives to key token $$i$$ in layer $$j$$. The attention a modality $$M \in \{V, A, L\}$$ receives in that layer is

$$
D^j_M = \sum_{i \in M} A^j_{Q_K, i}
$$

and averaging over all $$J$$ layers gives the dominance score

$$
D_M = \frac{1}{J} \sum_{j=1}^{J} D^j_M .
$$

For example, $$D_L > D_V > D_A$$ means language currently dominates the prediction.

### 2. Attentive masking of the less dominant modalities

AVCD perturbs the **two non-dominant modalities**. If language dominates, it masks video and audio. The method does not delete tokens. It takes the tokens in the top $$P\%$$ of attention within a modality and sets their representations to zero. For example, $$x^{\neg v}$$ is the video input with its most-attended tokens zeroed, which gives branches like $$\mathrm{logit}(x_t \mid x^v, x^{\neg a}, x^l)$$.

The reason for masking the *less* dominant modalities is the same as in VCD. The perturbed branch should expose bias, so the weak modalities are the ones whose evidence gets removed and contrasted.

### 3. Trimodal contrastive decoding

Two-modality CD has one question: what changes when vision is removed? With three modalities, video CD must decide what to do with audio.

AVCD marginalizes over audio, averaging the prediction with audio intact and with audio masked:

$$
p(y_t \mid x^v, x^l) = \tfrac{1}{2}\, p(y_t \mid x^v, x^a, x^l) + \tfrac{1}{2}\, p(y_t \mid x^v, x^{\neg a}, x^l)
$$

CD works on logits, though, and the log of an average is not the average of the logs:

$$
\log \frac{A + B}{2} \neq \frac{\log A + \log B}{2}
$$

A first-order Taylor expansion shows that the two are close when $$A \approx B$$, so AVCD approximates the marginal by averaging the two log-probabilities:

$$
\mathrm{logit}(y_t \mid x^v, x^l) \simeq \tfrac{1}{2}\big[\log p(y_t \mid x^v, x^a, x^l) + \log p(y_t \mid x^v, x^{\neg a}, x^l)\big]
$$

This needs four branches:

$$
L_{VA} = \log p(y_t \mid V, A, L), \quad
L_{\neg V A} = \log p(y_t \mid \neg V, A, L), \quad
L_{V \neg A} = \log p(y_t \mid V, \neg A, L), \quad
L_{\neg V \neg A} = \log p(y_t \mid \neg V, \neg A, L)
$$

Video CD compares "video intact" with "video masked", averaging over audio. Audio CD does the same with the roles swapped:

$$
\mathrm{logit}_v \propto (1+\alpha_v)(L_{VA} + L_{V\neg A}) - \alpha_v (L_{\neg V A} + L_{\neg V \neg A})
$$

$$
\mathrm{logit}_a \propto (1+\alpha_a)(L_{VA} + L_{\neg V A}) - \alpha_a (L_{V \neg A} + L_{\neg V \neg A})
$$

Adding them and collecting coefficients gives the final AVCD logit:

$$
\begin{aligned}
\mathrm{logit}_{\text{AVCD}} = \;& (2 + \alpha_v + \alpha_a)\, L_{VA}
+ (1 - \alpha_v + \alpha_a)\, L_{\neg V A} \\
&+ (1 + \alpha_v - \alpha_a)\, L_{V \neg A}
- (\alpha_v + \alpha_a)\, L_{\neg V \neg A}
\end{aligned}
$$

### 4. Entropy-guided adaptive decoding

Four branches per token are expensive. AVCD first runs the original forward pass and checks how confident it is:

$$
p_t = \mathrm{softmax}(\mathrm{logit}_t), \qquad
H_t = -\sum_{v \in \mathcal{V}} p_t(v) \log p_t(v)
$$

Low entropy means a confident model, high entropy means an uncertain one. If $$H_t < \tau$$, AVCD skips CD for that token and keeps the original prediction.

### Pipeline

**Dominance score → entropy gate → attentive masking of the less dominant modalities → trimodal CD**

## Experiments

**Audio-visual LLMs.** Accuracy (%) on MUSIC-AVQA and AVHBench, and a score for AVHBench captioning (see the paper for the VCD\* variant).

| Model | Decoding | MUSIC-AVQA | AVHBench | AVHBench-Cap |
|---|---|---|---|---|
| VideoLLaMA2 | Base | 81.30 | 70.52 | 2.84 |
| | VCD | 77.66 | 65.18 | 2.86 |
| | VCD\* | 81.49 | 69.18 | 3.00 |
| | **AVCD** | **81.58** | **72.15** | **3.03** |
| video-SALMONN | Base | 48.50 | 58.19 | 1.83 |
| | VCD | 41.57 | 60.61 | 2.41 |
| | VCD\* | 49.00 | 60.66 | 2.44 |
| | **AVCD** | **49.73** | **62.18** | **2.47** |

Plain VCD often *hurts* AV-LLMs (for example, 70.52 → 65.18 on AVHBench). Perturbing a single modality the VLM way does not transfer to three modalities.

**Video-only LLMs.** AVCD also works with two modalities.

| Model | Decoding | MSVD-QA | ActivityNet-QA |
|---|---|---|---|
| VideoLLaMA2 | Base | 74.43 | 47.19 |
| | VCD | 71.30 | 45.65 |
| | **AVCD** | **75.20** | **48.22** |
| Video-LLaVA | Base | 70.20 | 47.48 |
| | VCD | 71.80 | 47.25 |
| | **AVCD** | **72.16** | **48.03** |

**Which branches matter** (VideoLLaMA2, AVHBench). Two-branch CD on a single masked modality helps only a little. Naively subtracting all three masked branches, $$(1+3\alpha)L_{VA} - \alpha(L_{\neg VA} + L_{V\neg A} + L_{\neg V\neg A})$$, gains only 0.4 points. The derived trimodal formula is what makes the difference.

| Decoding | Masked | Acc |
|---|---|---|
| Base | – | 70.52 |
| 2-branch CD | A | 71.88 |
| 2-branch CD | V | 70.16 |
| 2-branch CD | AV | 72.07 |
| Naive 4-branch, $$(1+3\alpha)$$ form | A, V, AV | 70.94 |
| **AVCD** | A, V, AV | **72.15** |

**Dominant-modality choice.** Always treating language as dominant works as well as the adaptive choice on AVHBench, but the adaptive choice is best on MSVD-QA.

| Dominant modality | AVHBench | MSVD-QA |
|---|---|---|
| Audio | 68.67 | – |
| Vision | 67.79 | 70.70 |
| Language | 72.15 | 74.20 |
| **Adaptive** | **72.15** | **75.20** |

**Entropy threshold.** $$\tau$$ trades accuracy for speed. At $$\tau = 1.0$$, AVCD runs at about the base model's speed (≈1.3 s/token) with similar accuracy. At $$\tau = 0.6$$, accuracy rises to about 82% at ≈2.4 s/token. At $$\tau = 0.4$$, accuracy is about the same but the cost is ≈3.75 s/token.

## Thoughts

- AVCD is the baseline that later AV-LLM decoding work compares against. The four-branch structure $$(V,A), (\neg V,A), (V,\neg A), (\neg V,\neg A)$$ shows up again almost unchanged in [MAD]({% post_url 2026-10-22-mad-review %}). The same authors' [Fork-Merge Decoding]({% post_url 2026-10-22-fmd-review %}) takes a different route and works on hidden states instead of logits.
- The dominance score is about **attention**, not about what the question needs. A question about sound can still put most of its attention on video tokens. MAD addresses exactly this by asking the model directly which modality the question needs.
- The cost is real. Without the entropy gate, every token needs four forward passes.

<!-- TODO: add your own thoughts -->

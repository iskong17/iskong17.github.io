---
title: "AVCD: Contrastive Decoding for Three Modalities"
category: paper-review
subfields: [hallucination, avllm]
lang: en
description: Find the dominant modality from attention, mask the others, and extend contrastive decoding to audio, video, and text.
conference: NeurIPS
year: 2025
paper:
  title: "AVCD: Mitigating Hallucinations in Audio-Visual Large Language Models through Contrastive Decoding"
  authors: "Chaeyoung Jung, Youngjoon Jang, Joon Son Chung"
  venue: "NeurIPS 2025"
  url: "https://arxiv.org/abs/2505.20862"
---

## Abstract

Hallucination is still a major issue in multimodal LLMs. Contrastive decoding (CD) methods such as VCD work well for vision-language models: corrupt the image, see what the model predicts without visual evidence, and subtract it. An audio-visual LLM (AV-LLM) has three inputs (video, audio, language), so it is unclear which one to perturb. A hallucination can also come from any of them or from how they interact.

AVCD is a training-free decoding method for this three-modality setting. It measures which modality dominates the current prediction, masks the less dominant ones, and combines the resulting branches in a CD formula derived for three modalities. An entropy gate skips the extra work when the model is already confident.

## Motivation

The paper opens with an example of how AV-LLM hallucinations differ from VLM ones. Asked to describe a clip of a man playing cello and drums, VideoLLaMA2 invents a piano in the frame (a visual error) and then claims that "the cello and piano are both sounding" (an error that only exists in the combination of audio and video). VLM-style CD perturbs one fixed modality, the image, and would not target the second kind of mistake.

![Base decoding versus AVCD on an audio-visual captioning example](/images/blog/papers/avcd-review-motivation.webp)
*A captioning example with VideoLLaMA2. Base decoding (right) produces a visual mismatch (red: a piano, a black shirt) and an audio-visual mismatch (blue: cello and piano sounding together). AVCD (left) describes the cello and drum sounds and the shirt with a bird on it. Figure from the paper.*

The second piece of evidence is where the model's attention goes. Over 200 AVHBench samples, the last query token of audio-visual VideoLLaMA2 puts 70% of its attention on language tokens, 17% on video and only 13% on audio. Language is the dominant modality in every one of those samples. In the video-only variant on MSVD-QA the split is 56% language versus 44% video and the dominant modality changes from sample to sample. This is the basis for the method: find the dominant modality per sample from attention, then perturb the others, rather than always perturbing the image as VCD does.

![Attention share of language, video and audio tokens](/images/blog/papers/avcd-review-modality-dominance.webp)
*Average modality dominance in VideoLLaMA2. (a) Audio-visual model on AVHBench: language takes 70% of the final token's attention, audio only 13%. (b) Video-only model on MSVD-QA: a more balanced 56% vs. 44%. Figure from the paper (appendix).*

## Method

![AVCD overview](/images/blog/papers/avcd-main.png)
*AVCD pipeline. Attention gives a modality dominance order, the less dominant modalities are masked, and the masked branches are combined by trimodal CD. Figure from the paper.*

### Contrastive decoding recap

For a VLM, CD contrasts an original branch with a branch whose visual input is perturbed:

$$
\mathrm{logit} = (1+\alpha)\,\mathrm{logit}(y \mid x^v, x^l) - \alpha\,\mathrm{logit}(y \mid x^{\neg v}, x^l)
$$

The perturbed branch shows what the model predicts from language bias alone. Subtracting it amplifies the part of the prediction that comes from real visual evidence.

### 1. Modality dominance (left of the figure)

Let $$A^j_{Q_K, i}$$ be the attention that the last query token gives to key token $$i$$ in layer $$j$$. The dominance of modality $$M \in \{V, A, L\}$$ is its total attention, averaged over all $$J$$ layers:

$$
D_M = \frac{1}{J} \sum_{j=1}^{J} \sum_{i \in M} A^j_{Q_K, i}
$$

In the figure, $$D_L > D_V > D_A$$, so language dominates.

### 2. Attentive masking (middle)

AVCD perturbs the **non-dominant** modalities. It does not delete tokens. It zeroes the representations of the top-$$P\%$$ most attended tokens in that modality. With language dominant, this gives three perturbed branches: video masked, audio masked, and both masked.

### 3. Trimodal contrastive decoding (right)

Video CD should not depend on whether audio happens to be present, so AVCD averages over audio intact and audio masked. It approximates that average in log space, which is accurate when the two probabilities are close. Using the four branches

$$
L_{VA},\quad L_{\neg V A},\quad L_{V \neg A},\quad L_{\neg V \neg A} \qquad (L_{\cdot} = \log p(y_t \mid \cdot, L))
$$

video CD and audio CD are

$$
\mathrm{logit}_v \propto (1+\alpha_v)(L_{VA} + L_{V\neg A}) - \alpha_v (L_{\neg V A} + L_{\neg V \neg A})
$$

$$
\mathrm{logit}_a \propto (1+\alpha_a)(L_{VA} + L_{\neg V A}) - \alpha_a (L_{V \neg A} + L_{\neg V \neg A})
$$

and their sum is the final logit:

$$
\mathrm{logit}_{\text{AVCD}} = (2 + \alpha_v + \alpha_a) L_{VA} + (1 - \alpha_v + \alpha_a) L_{\neg V A} + (1 + \alpha_v - \alpha_a) L_{V \neg A} - (\alpha_v + \alpha_a) L_{\neg V \neg A}
$$

### 4. Entropy-guided adaptive decoding (top)

The original forward pass runs first. If its entropy $$H_t = -\sum_v p_t(v) \log p_t(v)$$ is below a threshold $$\tau$$, the model is confident, and AVCD keeps the original prediction for that token.

## Main Experiments

Accuracy (%) on MUSIC-AVQA and AVHBench, and a score for AVHBench captioning. See the paper for the VCD\* variant.

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

AVCD also helps video-only LLMs: VideoLLaMA2 on MSVD-QA goes from 74.43 to 75.20, and Video-LLaVA from 70.20 to 72.16.

- Plain VCD often *hurts* AV-LLMs (70.52 → 65.18 on AVHBench for VideoLLaMA2). Perturbing a single modality the VLM way does not transfer.
- In the ablation, two-branch CD on a single masked modality helps only a little, and naively subtracting all three masked branches gives 70.94. The derived trimodal formula reaches 72.15.

The qualitative examples show what "ignored modality" errors look like. In the first, the video shows a man with an acoustic guitar but the audio is a violin; Base and VCD answer that one acoustic guitar is sounding, while AVCD answers zero. In the second (a video-only model), the question contains a misleading statement, and only AVCD answers from the snowy frames.

![Qualitative comparison of Base, VCD and AVCD](/images/blog/papers/avcd-review-qualitative.webp)
*Top: AV-LLM example where the audio is a violin, so no acoustic guitar is sounding; only AVCD says "Zero". Bottom: video-LLM example with a misleading prompt; Base and VCD follow the text, AVCD uses the snow in the video. Figure from the paper.*

The entropy gate is evaluated on the AVHBench validation split. Lower $$\tau$$ means more tokens go through all branches, which raises accuracy but costs time. At $$\tau = 0.8$$ AVCD is faster than VCD and still improves on Base (78.05% to 80.98%), while VCD itself falls far below Base on this set.

![Accuracy versus inference time for different entropy thresholds](/images/blog/papers/avcd-review-entropy-tradeoff.webp)
*Accuracy vs. seconds per token for VideoLLaMA2 on the AVHBench validation set. Blue circles are AVCD with different thresholds $$\tau$$; $$\tau = 1.0$$ falls back to roughly Base speed and accuracy. VCD (green) is both slower and much less accurate. Figure from the paper.*

## Limitations

- **Cost.** Without the entropy gate, each token needs four forward passes. The gate trades accuracy for speed: in the paper's plot, accuracy rises as $$\tau$$ drops, but latency grows from about 1.3 to about 3.75 s/token.
- **Attention is not need.** Dominance says what the model *is* attending to, not what the question requires. A question about sound can still put most of its attention on video tokens. [MAD]({% post_url 2026-10-22-mad-review %}) addresses this by asking the model which modality the question needs.
- **Small gains on some settings.** On MUSIC-AVQA with VideoLLaMA2 the gain over the base model is under 0.3 points.

## My Thoughts

<!-- TODO -->

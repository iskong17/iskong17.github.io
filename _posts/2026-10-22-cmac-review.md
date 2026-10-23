---
title: "CMAC: Contrast Against the Strong Image-Text Links, Not a Blurred Image"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: Build the contrastive branch by masking the value vectors behind the strongest text-to-image attention weights, and compress image-token positions so RoPE stops favouring the end of the image.
conference: CVPR
year: 2026
paper:
  title: "Cross-Modal Attention Calibration for LVLM Hallucination Mitigation"
  authors: "Jiaming Li, Jiacheng Zhang, Zequn Jie, Lin Ma, Ming Li, Xiaonan Luo, Guanbin Li"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2501.01926"
---

## Abstract

Large vision-language models (LVLMs) still **hallucinate**: they describe objects or attributes that are not in the image. Training-free contrastive decoding (CD) methods such as VCD and ICD reduce this by contrasting the normal output with the output on a distorted input (a noised image or a perturbed instruction). That "hallucinatory" branch captures over-reliance on the language prior, but the authors argue it misses two other sources of error: **spurious inter-modality correlations** (food in the image makes the model say "dining table") and a **position bias** from RoPE that makes text tokens attend mostly to the last image tokens.

CMAC (Cross-Modal Attention Calibration) changes where the distortion happens. Its Inter-Modality Decoding (IMD) module builds the contrastive branch inside the decoder: it finds the strongest text-to-image attention weights and replaces the value vectors behind them with a mean vector, so the branch keeps only weak, possibly spurious links and the language prior. Its Cross-Modal Position Calibration (CMPC) module shrinks the RoPE position gap between image tokens when computing text-to-image attention. Both are training-free.

## Motivation

The paper starts from the gap in existing CD methods. Distorting the whole image yields a branch that "cannot see the clear image" and answers from the language prior. That is one failure mode. A second one is that the model sees *the wrong part* of the image and answers from a superficial text-image association learned from web data. A globally distorted image cannot isolate that second mode, so the contrast is too coarse to remove it.

![Existing contrastive decoding vs. CMAC's inter-modality decoding](/images/blog/papers/cmac-review-motivation.webp)
*Top: existing CD distorts the whole image, so the hallucinatory branch only reflects the unimodal (language) prior. Bottom: CMAC masks selected entries of the cross-modal attention map, so the branch misses the important areas and keeps attending to unimportant ones, reflecting both the language prior and spurious text-image correlations. Figure from the paper.*

The second observation is a positional one. Plotting the attention weight that text tokens give to each image token shows a sharp rise toward the end of the image sequence for both Qwen-VL and LLaVA-1.5. The authors trace this to RoPE: image tokens are placed before the query, so the last image tokens are closest to the text and get more attention. The hallucination statistics follow the same pattern. Objects in the head part of the image token sequence are hallucinated at a rate of 0.28, the middle part 0.21, and the tail part 0.15.

![Cross-modal attention over image token index and hallucination rate by position](/images/blog/papers/cmac-review-position-bias.webp)
*Left and middle: cross-modal attention weight over the image token index for Qwen-VL and LLaVA-1.5, rising sharply at the tail. Right: hallucination frequency by position of the object's tokens in the image sequence (head 0.28, middle 0.21, tail 0.15). Figure from the paper.*

The paper also checks the spurious-correlation claim on POPE with LLaVA-1.5. For objects whose top co-occurring object is present in the image, both the true-negative and false-negative error rates are clearly higher than when that co-occurring object is absent (see Figure 5 below).

## Method

![CMAC overview](/images/blog/papers/cmac-review-main.webp)
*CMAC overview. CMPC recomputes the text-to-image attention logits with compressed image positions. IMD builds a global mask over the strongest cross-modal attention logits and replaces the corresponding value vectors with their mean to induce a hallucinated distribution, which is then contrasted with the original one. Figure from the paper.*

The input sequence is $$[T_{0:m_b}, X, T_{m_b+1:m}]$$: a system prompt of $$m_b$$ tokens, $$n$$ image tokens, then the query prompt. In each attention head, $$A^l = Q_p K_p^\top / \sqrt{d}$$ are the attention logits, $$A = \mathrm{softmax}(A^l)$$, and the output is $$O = AV$$.

### Inter-Modality Decoding (IMD)

IMD takes the cross-modal block of the logits, the rows of the query-prompt (and generated) tokens and the columns of the image tokens, $$A^l_{\text{cross}} = A^l[m_b{+}n : m{+}n,\ m_b : m_b{+}n]$$, and marks every entry above the block mean:

$$
M_{\text{cross}} = \mathbb{I}\big(A^l_{\text{cross}} > \mu(A^l_{\text{cross}})\big)
$$

$$M_{\text{cross}}$$ is zero-padded to a full-size mask $$M$$. The attention weights stay unchanged. Only the value vectors behind the selected entries are swapped for the mean value vector $$\mu(V)$$:

$$
\tilde O = (M \odot A)\, \mu(V) + ((1 - M) \odot A)\, V
$$

So the distorted branch loses exactly the strong, legitimate text-image links and keeps the weak links, the intra-modal attention, and the language prior. Running the decoder with $$\tilde O$$ gives logits $$\tilde l_t$$, and decoding uses the standard CD rule:

$$
p_\theta(Y_t \mid T, X) = \mathrm{softmax}\big((1 + \alpha)\, l_t - \alpha\, \tilde l_t\big)
$$

The authors list two side benefits. Because attention weights are not altered, the hallucinated branch is not over- or under-weighted toward some image regions, as can happen with attention masking or token pruning. And because the visual part of attention is unchanged, keys, values, and attention weights can be reused from the original forward pass, which makes IMD somewhat faster than VCD and ICD, which run a second full forward pass on a distorted input.

### Cross-Modal Position Calibration (CMPC)

CMPC builds a second set of position indices in which image tokens are compressed by a factor $$\gamma$$:

$$
P^c = \big[\{i\}_{i=1}^{m_b},\ \{m_b + i/\gamma\}_{i=1}^{n},\ \{i + n/\gamma\}_{i=m_b+1}^{m}\big]
$$

RoPE is applied with $$P^c$$ to get logits $$A^c$$, and only the text-to-image block of $$A^l$$ is replaced by the corresponding entries of $$A^c$$. Everything else keeps the original positions. Image tokens stay in the same place relative to the text, but the distance between the first and last image token shrinks, so the head of the image is not pushed far away from the query. The refined logits are used in both the original and the distorted branch, and also for computing the IMD mask.

The defaults are $$\gamma = 2$$ and $$\alpha = 3$$, with top-p set to 1.

## Main Experiments

Backbones are LLaVA-1.5, InstructBLIP, and Qwen-VL, all with a 7B language model. POPE reports accuracy and F1, averaged over the COCO, A-OKVQA, and GQA subsets. The table below is the nucleus-sampling setting.

| Model | Method | Random Acc | Random F1 | Popular Acc | Popular F1 | Adversarial Acc | Adversarial F1 |
|---|---|---|---|---|---|---|---|
| LLaVA-1.5 | Baseline | 83.49 | 82.28 | 79.98 | 79.34 | 76.03 | 76.26 |
| | ICD | 85.70 | 84.43 | 81.61 | 81.46 | 78.66 | 78.57 |
| | VCD | 86.84 | 86.83 | 82.65 | 83.37 | 77.31 | 79.28 |
| | PAI | 87.73 | 87.65 | 83.45 | 84.21 | 78.36 | 78.53 |
| | **CMAC** | **89.10** | **88.60** | **86.00** | **85.70** | **81.41** | **81.84** |
| InstructBLIP | Baseline | 80.41 | 80.94 | 76.09 | 77.65 | 72.37 | 75.42 |
| | VCD | 84.11 | 84.13 | 79.96 | 80.80 | 76.32 | 78.08 |
| | PAI | 84.60 | 84.54 | 80.13 | 80.94 | 77.00 | 78.33 |
| | **CMAC** | **86.67** | **86.10** | **82.20** | **82.37** | **78.37** | **79.25** |
| Qwen-VL | Baseline | 84.12 | 82.42 | 81.89 | 80.51 | 79.10 | 78.07 |
| | VCD | 87.82 | 87.68 | 85.60 | 85.48 | 81.85 | 82.34 |
| | PAI | 88.15 | 87.94 | 85.73 | 85.51 | 81.98 | 82.23 |
| | **CMAC** | **88.61** | **88.26** | **86.67** | **86.34** | **82.68** | **83.05** |

CHAIR on 500 MS-COCO validation images with the prompt "Please describe this image in detail" (nucleus sampling; lower $$C_i$$ and $$C_s$$ is better):

| Method | LLaVA-1.5 Ci ↓ | LLaVA-1.5 Cs ↓ | LLaVA-1.5 F1 ↑ | InstructBLIP Ci ↓ | InstructBLIP Cs ↓ | InstructBLIP F1 ↑ | Qwen-VL Ci ↓ | Qwen-VL Cs ↓ | Qwen-VL F1 ↑ |
|---|---|---|---|---|---|---|---|---|---|
| Sampling | 55.6 | 17.8 | 77.0 | 57.0 | 16.7 | 76.3 | 44.8 | 11.3 | 81.1 |
| VCD | 54.2 | 16.4 | 80.0 | 53.6 | 15.6 | 79.2 | 47.2 | 12.1 | 81.6 |
| ICD | 54.0 | 15.4 | 80.7 | **49.6** | 14.5 | 78.5 | 43.2 | 11.2 | 81.5 |
| **CMAC** | **47.0** | **12.7** | **81.0** | 50.6 | **13.2** | **79.7** | **41.2** | **10.6** | **81.8** |

- The POPE gains are largest on the harder splits: on LLaVA-1.5, Popular accuracy goes from 79.98 to 86.00 and Adversarial from 76.03 to 81.41, 2.5 to 3 points above the next best method in both.
- On the MME hallucination subset, LLaVA-1.5's total rises from 566.67 to 612.49 (VCD 581.25, ICD 582.50), with the largest gain on Count (110.00 → 135.41).
- In the ablation on LLaVA-1.5 (POPE Popular), IMD alone gives 85.10 accuracy and CHAIRs 13.6, CMPC alone 82.31 and 16.2, and both together 86.04 and 12.7. Masking values to the mean (86.04) beats masking attention weights (84.92) and adding noise to values (85.72).

The co-occurrence analysis supports the IMD argument directly. On POPE with LLaVA-1.5, for objects whose top co-occurring object is in the image, CMAC brings the true-negative error rate down to about the level of ICD and far below the baseline and VCD, while keeping the false-negative rate low.

![Hallucination rate with and without the top co-occurring object](/images/blog/papers/cmac-review-cooccurrence.webp)
*(a) LLaVA-1.5 on POPE: TNR and FNR errors are higher when the queried object's top co-occurring object is in the image. (b) The same errors in the co-occurring case for Baseline, VCD, ICD, and CMAC, averaged over 5 high-frequency object pairs. Figure from the paper.*

CMPC also flattens the positional pattern. With CMAC, the hallucination frequency on the head, middle, and tail parts of the image tokens drops from 28% / 21% / 15% to 11% / 13% / 10%.

On the full MME benchmark, CMAC scores best among the compared methods on 10 of 14 subtasks and clearly better on perception subtasks such as Posters, Celebrity, Scene, and Landmark. It is weaker on Numerical calculation and Text translation, which depend on language reasoning rather than on the image.

![MME full-set scores for LLaVA-1.5](/images/blog/papers/cmac-review-mme.webp)
*MME scores per subtask for LLaVA-1.5 with the baseline, VCD, ICD, and CMAC. Figure from the paper.*

## Limitations

- **Attention magnitude is a proxy.** The paper notes that a large attention weight does not necessarily mean a causally relevant text-image link, so the mask can pick the wrong entries.
- **Bounded by the vision encoder.** CMAC does not improve what the visual encoder extracts. If an object is not represented in the image tokens, no amount of re-weighting will ground it (also noted by the authors).
- **Still a two-branch decoder.** Even with reused keys and values, throughput on POPE drops from 12.93 to 7.75 tokens/s on LLaVA-1.5 (VCD 7.04, ICD 6.90).
- **White-box only.** Both modules modify attention internals and RoPE indices, so they need access to the decoder and a known layout of system prompt, image, and query tokens.
- **Language-heavy tasks can regress.** On MME, Text translation and Numerical calculation drop slightly relative to the baseline.

## My Thoughts

<!-- TODO -->

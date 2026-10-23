---
title: "Nullu: Project the Hallucination Subspace Out of the Weights"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: "Find the low-rank directions that separate hallucinated from truthful features, then edit the MLP weights so every input is projected away from them, at zero inference cost."
conference: CVPR
year: 2025
paper:
  title: "Nullu: Mitigating Object Hallucinations in Large Vision-Language Models via HalluSpace Projection"
  authors: "Le Yang, Ziwei Zheng, Boxu Chen, Zhengyu Zhao, Chenhao Lin, Chao Shen"
  venue: "CVPR 2025"
  url: "https://arxiv.org/abs/2412.13817"
---

## Abstract

Large vision-language models (LVLMs) suffer from **object hallucination (OH)**: they describe objects that are not in the image. Existing fixes come in two kinds. Fine-tuning methods (robust instruction tuning, preference alignment) need curated data and a lot of compute. Post-hoc decoding methods such as VCD, OPERA and HALC avoid training but change the inference procedure, with extra forward passes, beam search or several adaptive branches, so every generated token gets slower.

Nullu takes a third route: **weight editing**. It feeds the model pairs of inputs with the same image but a truthful or a hallucinated caption, and collects the hidden features of both. It then takes the top singular vectors of their difference as a low-rank **HalluSpace**, and projects the MLP weights of selected layers onto the null space of that subspace. The edited weights replace the original ones, so the model runs exactly as before, with no extra cost at inference. The authors also show that the HalluSpace overlaps with the language prior of the underlying LLM, which links Nullu to contrastive decoding methods like VCD that try to remove the same prior at decoding time.

## Motivation

The paper starts from a gap: many methods mitigate OH, but few look at where hallucination lives *inside* the model's features. If truthful and hallucinated responses differ along a few consistent directions in the feature space, those directions can simply be removed once, offline, instead of being corrected token by token at inference.

![Nullu editing and inference phases](/images/blog/papers/nullu-review-motivation.webp)
*(a) Editing phase: hidden features of truthful (T.) and hallucinated (H.) inputs are extracted, a low-rank HalluSpace $$v$$ is found from their difference, and the weights are projected onto its null space. (b) Inference phase: using the edited weights is the same as projecting every input feature into the safe subspace, away from the hallucinated area. Figure from the paper.*

Two pieces of evidence support the premise. First, a low rank is enough: on LLaVA-1.5, $$k = 4$$ directions per layer give the best CHAIR scores (Table 3 below), so the truthful/hallucinated difference is concentrated in a small subspace. Second, the HalluSpace generalizes to unseen test samples. For 100 CHAIR test samples where editing removed the hallucination, the change in features between the raw and edited model has a projection onto the rank-4 HalluSpace about 10 times larger than that of 100 random vectors, at every edited layer. The authors read this as the HalluSpace capturing directions actually tied to hallucination, not noise from the paired training data.

The cost argument is the practical motivation. VCD needs two forward passes per token and HALC runs several adaptive branches. A weight edit has none of these costs.

## Method

![Nullu overview](/images/blog/papers/nullu-review-main.webp)
*(a) A paired sample: the same image with a ground-truth description and with a hallucinated one ("a desk lamp ... illuminating the shelves"). (b) For each target layer, Nullu stacks hidden features of the two inputs, takes the SVD of their difference, keeps the top-k right singular vectors as the HalluSpace, and edits the MLP weights. Figure from the paper.*

### Paired data

Each pair shares an image. One text is the ground-truth description $$x^-_i$$ and the other is a hallucinated version $$x^+_i$$. The pairs come from LURE: GPT-3.5 rewrites an accurate description by swapping in objects likely to be hallucinated, guided by co-occurrence, object uncertainty and object position.

### Finding the HalluSpace

At layer $$\ell$$ of the LLM, the features of each sample are averaged over tokens and stacked into $$X^+_\ell, X^-_\ell \in \mathbb{R}^{N \times D}$$. Their difference is decomposed with SVD:

$$
E_\ell = X^+_\ell - X^-_\ell = U_\ell \Sigma_\ell V_\ell^\top
$$

The first $$k$$ right singular vectors $$V_{\ell,k} = [v_1, \dots, v_k] \in \mathbb{R}^{D \times k}$$ are the main directions along which hallucinated features differ from truthful ones. This is the HalluSpace.

### Editing the weights

The MLP weight of layer $$\ell$$ is projected onto the null space of the HalluSpace:

$$
W^{\text{ed}}_\ell = (I - V_{\ell,k} V_{\ell,k}^\top)\, W^{\text{org}}_\ell
$$

For an input feature $$x_i$$, the edited output is $$x_i (I - V_k V_k^\top) W^{\text{org}}$$: the component of $$x_i$$ that lies in the HalluSpace is removed before the MLP sees it. If the output is modeled as truthful content plus hallucinated bias plus noise, the projection subtracts the part explained by the top-k difference directions and keeps the rest. The edit is applied to layers $$\ell \in \{16, \dots, 32\}$$ of the LLM with $$k = 4$$, and the edited weights are saved back into the model.

### What the HalluSpace contains

![HalluSpace decoding and its relation to VCD](/images/blog/papers/nullu-review-llm-prior.webp)
*(a) HalluSpace vectors are decoded through the output embedding to find their closest words. (b) VCD contrasts logits from a clean and a noised image. (c) Frequency of the decoded words in outputs of Nullu (1), the model with a distorted image, i.e. the LLM prior (2), plain LLaVA (3), and VCD (4). Figure from the paper.*

Decoding the HalluSpace vectors into vocabulary gives frequent, generic words such as "people", "background", "surrounded", "located" and "middle". These are the same words that LLaVA-1.5 over-produces when the image is distorted, which is how VCD estimates the LLM prior. Both VCD and Nullu cut how often these words appear (for "background": 70 in plain LLaVA, 66 with VCD, 56 with Nullu). The authors conclude that Nullu removes the LLM prior from the weights, while VCD removes it at decoding time.

The paper also links Nullu to DPO. For a simple log-linear model, the gradient of the DPO loss contains a term built from the feature difference $$x^+_i - x^-_i$$. DPO lowers this difference by gradient steps, and Nullu approximates it with SVD and removes it in closed form.

## Main Experiments

CHAIR on 500 MSCOCO val images, prompt "Please describe this image in detail.", max 64 new tokens, mean of 3 seeds. CS is the share of captions with a hallucinated object and CI the share of hallucinated objects (lower is better). All three models are 7B-class.

| Method | LLaVA-1.5 CS | LLaVA-1.5 CI | MiniGPT-4 CS | MiniGPT-4 CI | mPLUG-Owl2 CS | mPLUG-Owl2 CI |
|---|---|---|---|---|---|---|
| Greedy | 20.40 | 7.08 | 32.40 | 12.20 | 22.90 | 8.62 |
| Beam Search | 19.50 | 6.84 | 30.10 | 11.87 | 20.30 | 7.62 |
| DoLa | 20.20 | 6.75 | 31.90 | 12.15 | 22.40 | 8.36 |
| OPERA | 17.50 | 6.07 | 29.70 | 11.96 | 20.07 | 7.18 |
| VCD | 20.30 | 7.28 | 29.00 | 12.64 | 22.80 | 8.68 |
| Woodpecker | 23.85 | 7.50 | 28.87 | 10.20 | 26.33 | 8.43 |
| LURE | 19.48 | 6.50 | 27.88 | 10.20 | 21.27 | 7.67 |
| HALC | 16.90 | 5.72 | 25.20 | 9.42 | 18.80 | 7.00 |
| **Nullu** | **15.20** | **5.30** | **21.40** | **8.99** | **15.60** | **5.77** |

OPOPE asks the model for a detailed description and then checks offline whether the sampled positive and negative POPE objects appear in it. Numbers are averaged over the random, popular and adversarial splits.

| Method | LLaVA-1.5 Acc | LLaVA-1.5 F | MiniGPT-4 Acc | MiniGPT-4 F | mPLUG-Owl2 Acc | mPLUG-Owl2 F |
|---|---|---|---|---|---|---|
| Greedy | 79.14 | 90.45 | 71.22 | 90.04 | 76.46 | 87.29 |
| OPERA | 79.29 | 90.71 | 70.48 | 90.66 | 75.49 | 89.11 |
| VCD | 78.01 | 89.69 | 70.83 | 88.76 | 75.49 | 87.02 |
| HALC | 77.87 | 91.25 | 71.17 | 90.95 | 74.93 | 88.12 |
| **Nullu** | **79.52** | **91.79** | **71.92** | **92.07** | **77.09** | **90.80** |

![Throughput vs. CHAIR-S and effect of max tokens](/images/blog/papers/nullu-review-throughput.webp)
*(a) CHAIR-S against throughput on an RTX 4090 for LLaVA-1.5. Nullu runs at the same speed as greedy decoding with about 5 points lower CHAIR-S, and is about 7 times faster than OPERA. (b) CHAIR-I and CHAIR-S for greedy and Nullu at 64, 128 and 256 max tokens. Figure from the paper.*

- Nullu has the lowest CHAIR on all three models, and the gap is largest where the base model hallucinates most: MiniGPT-4 CS goes from 32.40 to 21.40, and mPLUG-Owl2 from 22.90 to 15.60. BLEU stays at the greedy level (15.69 vs. 15.72 on LLaVA-1.5).
- The gain holds for longer captions. At 256 tokens, CHAIR-S drops from 44.6 to 30.2 and CHAIR-I from 13.6 to 9.5.
- General ability does not suffer. On MME, LLaVA-1.5's recognition total rises from 321.19 to 351.31, mostly from code reasoning (71.67 to 92.50). GPT-4V-judged accuracy on LLaVA-Bench rises for all three models (LLaVA-1.5 5.89 to 6.53, MiniGPT-4 4.11 to 5.63).
- Ablation on LLaVA-1.5: editing layers 16-32 is best (CS 15.2), compared with 24-32 (16.2) and 28-32 (18.2). For the rank, $$k = 4$$ is best (15.2). Both $$k = 2$$ (20.2) and $$k = 16$$ (17.2) are worse.

![Qualitative examples on LLaVA-Bench](/images/blog/papers/nullu-review-qualitative.webp)
*LLaVA-1.5-7B before and after editing on two LLaVA-Bench images. The original model adds co-occurring objects (chairs, dining table, bowl, clock, pedestrians); the edited model does not. The inset shows the HalluSpace applied online at the hallucinated token: the logit of "person" drops and the model writes "mountainous" instead. Figure from the paper.*

## Limitations

- **Paired data is needed.** The HalluSpace comes from LURE pairs whose hallucinated captions were written by GPT-3.5. The method is training-free, but not data-free, and the HalluSpace can only reflect the kinds of hallucination that the rewriting rules produce.
- **The hyperparameters are found empirically.** The paper says there is no theoretical argument for which layers to edit or for $$k$$. Both were picked by a sweep on CHAIR with LLaVA-1.5-7B and then fixed, so they were tuned on the same benchmark used for the headline results.
- **The edit is static.** The same projection is applied to every input and every task. Directions that matter for some legitimate outputs could be removed too. MME and LLaVA-Bench do not show this, but the evaluation only covers older 7B models.
- **Part of the POPE gain comes from saying "no" more often.** In the supplementary POPE results, recall falls with Nullu: LLaVA-1.5 random from 89.43 to 87.10, and MiniGPT-4 from 97.13 to 78.53, while precision rises. For MiniGPT-4, which almost always answers "yes", this is a real fix of the bias. Still, the F1 gain is partly a shift in the answer distribution, not only better grounding. The [CD Mirage review](/blog/2026/10/22/cd-mirage-review/) shows why this matters when reading POPE results.
- **The link to the LLM prior is correlational.** It rests on word-frequency overlaps for a few hand-picked decoded words, which the authors themselves call "the most representative results".

## My Thoughts

<!-- TODO -->

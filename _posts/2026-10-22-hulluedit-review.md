---
title: "HulluEdit: Split the Hidden State, Shrink Only the Prior"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: "Split each hidden state into a visual-evidence part, a language-prior part and a residual, then shrink the last two in closed form while leaving the visual part untouched."
conference: CVPR
year: 2026
paper:
  title: "HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models"
  authors: "Yangguang Lin, Quan Fang, Yufei Li, Jiachen Sun, Junyu Gao, Jitao Sang"
  venue: "CVPR 2026"
  url: "https://arxiv.org/abs/2602.22727"
---

## Abstract

LVLMs hallucinate objects when strong language priors override weak visual evidence. The training-free fixes each pay a price. Contrastive decoding (VCD, ICD, DoLa) needs a second forward pass or a reference distribution, which costs latency. Static subspace editing (Nullu) builds one "hallucination subspace" offline for the whole dataset and projects it out at every token, so it cannot adapt to the current sample and can remove real visual signal along with the prior.

HulluEdit edits the hidden state of the last layer in a single pass. For every token it estimates, online, a **visual evidence subspace** $$U$$ from the image tokens and an **anti-prior subspace** $$P$$ from recent text tokens, built inside the orthogonal complement of $$U$$. The hidden state then splits into three orthogonal parts: visual, prior, and residual. HulluEdit shrinks the prior and residual parts with a closed-form update and leaves the visual part exactly as it was. No training, no reference model, no extra forward pass.

## Motivation

The paper frames the problem with a single picture. In a normal LVLM the hidden state $$h$$ mixes a visual-evidence direction with a language-prior direction. When the prior is strong, decoding follows it and produces objects that are not in the image ("a dog", "a cup of coffee" next to a laptop). If the two directions could be separated, the prior could be weakened without touching what the image actually supports.

![Entangled visual evidence and language prior vs. HulluEdit's orthogonal decomposition](/images/blog/papers/hulluedit-review-motivation.webp)
*(a) In a standard LVLM, visual evidence and language prior are entangled in the hidden state, and the prior produces hallucinated tokens. (b) HulluEdit splits $$h$$ into $$h_U$$ (visual evidence), $$h_P$$ (anti-prior) and $$h_R$$ (residual), keeps $$h_U$$, and shrinks $$h_P$$. Figure from the paper.*

Two observations shape the design. First, existing methods lack this separation: contrastive decoding works on whole output distributions and can weaken real visual cues along with the prior, and Nullu's subspace is fixed per dataset, not per sample or token. Second, following [DeCo](/blog/2026/10/22/deco-review/), the authors take intermediate layers as a more reliable source of visual evidence than the last layer. DeCo uses a mid layer to correct the output logits. HulluEdit instead uses a mid layer (the *anchor layer*, layer 26 in LLaVA) to build the visual subspace, and edits the last layer's hidden state.

## Method

![HulluEdit overview](/images/blog/papers/hulluedit-review-main.webp)
*Visual tokens $$V$$ and a text cache $$T$$ come from the anchor layer. A weighted SVD of $$V$$ gives $$U$$, an SVD of $$T$$ projected off $$U$$ gives $$P$$, and the rest is $$R$$. The last-layer hidden state $$h$$ is split into $$h_U, h_P, h_R$$ and recombined with strengths set by VCR and PCR. Figure from the paper.*

### Subspace construction

The visual features $$V \in \mathbb{R}^{n_v \times d}$$ are taken once per image from the anchor layer. A text cache $$T \in \mathbb{R}^{n_t \times d}$$ keeps the non-visual hidden states of recent decoding steps in a sliding window.

**Visual evidence subspace.** At each step, the visual tokens are weighted by their cosine similarity to the current hidden state $$h$$, $$w_i = \mathrm{softmax}_i\big(v_i^\top h / (\lVert v_i \rVert \lVert h \rVert + \epsilon)\big)$$. A truncated SVD of the weighted matrix gives the top-$$r$$ left singular vectors:

$$
U, \Sigma, V^\top = \mathrm{SVD}(W^{1/2} V), \qquad U = U_{[:,1:r]}
$$

So $$U$$ is sample-adaptive and also token-adaptive: it emphasises the image tokens most related to what is being generated now.

**Anti-prior subspace.** The text cache is first projected off $$U$$, and its top-$$q$$ singular vectors form $$P$$:

$$
\tilde T = T (I_d - U U^\top), \qquad P = \mathrm{SVD}_q(\tilde T)
$$

By construction $$U^\top P = 0$$. Everything orthogonal to both is the residual, $$\Pi_R = I_d - UU^\top - PP^\top$$. The hidden state splits as $$h = h_U + h_P + h_R$$ with $$\lVert h \rVert^2 = \lVert h_U \rVert^2 + \lVert h_P \rVert^2 + \lVert h_R \rVert^2$$.

### Certificate-aware editing

Two ratios measure how visual and how prior-driven the current state is:

$$
\mathrm{VCR}(h) = \frac{\lVert h_U \rVert^2}{\lVert h \rVert^2 + \epsilon}, \qquad \mathrm{PCR}(h) = \frac{\lVert h_P \rVert^2}{\lVert h \rVert^2 + \epsilon}
$$

They set two strengths: $$\lambda_n = \min\big(\kappa (1 - \mathrm{VCR}) / \mathrm{VCR}, \lambda_{\max}\big)$$ grows when visual evidence is weak, and $$\lambda_p = \min\big(\lambda_0\, \mathrm{PCR} / (1 - \mathrm{PCR}), \lambda_{\max}\big)$$ grows when the prior is strong. The edit is the minimum-norm perturbation $$\delta$$ that shrinks the non-visual part and the prior part:

$$
\min_{\delta}\; \tfrac{1}{2}\lVert \delta \rVert^2 + \tfrac{\lambda_n}{2}\lVert \Pi_\perp (h+\delta) \rVert^2 + \tfrac{\lambda_p}{2}\lVert \Pi_P (h+\delta) \rVert^2
$$

where $$\Pi_\perp = \Pi_P + \Pi_R$$. Because the projectors are orthogonal, the solution is closed-form:

$$
h' = h_U + \frac{1}{1 + \lambda_n + \lambda_p}\, h_P + \frac{1}{1 + \lambda_n}\, h_R
$$

The visual component is kept exactly, the prior component is shrunk the most, and the residual is shrunk more gently. A gate applies the edit only when risk is high: $$h'$$ is used if $$\mathrm{VCR} < \gamma_v$$ or $$\mathrm{PCR} > \gamma_p$$, and $$h$$ is left alone otherwise. From the closed form, the paper proves that VCR can only go up and PCR can only go down after the edit.

The per-token overhead is two small SVDs and a few projections. With $$r = 8$$, $$q = 5$$, $$d = 4096$$, $$n_v \le 576$$ and $$n_t \le 512$$, the authors put it below 2% of one transformer layer. Other settings for LLaVA-1.5-7B: anchor layer 26, edit at the last layer, $$\kappa = 0.60$$, $$\lambda_0 = 0.26$$, $$\lambda_{\max} = 3.6$$.

## Main Experiments

POPE accuracy and F1 (greedy decoding) on the three splits. HulluEdit is best in all 18 settings.

| Split | Method | LLaVA-1.5-7B Acc | F1 | LLaVA-1.5-13B Acc | F1 | Qwen-VL-Chat-7B Acc | F1 |
|---|---|---|---|---|---|---|---|
| Random | Greedy | 87.8 | 87.5 | 87.6 | 87.4 | 88.2 | 87.9 |
| | VCD | 88.4 | 87.7 | 88.9 | 87.8 | 89.1 | 88.4 |
| | VAF | 89.6 | 89.3 | 90.1 | 89.9 | 90.0 | 89.7 |
| | DeCo | 89.1 | 89.8 | 82.2 | 84.9 | 87.8 | 87.2 |
| | **HulluEdit** | **90.4** | **90.5** | **90.6** | **90.8** | **90.2** | **89.9** |
| Popular | Greedy | 82.5 | 83.2 | 82.7 | 84.1 | 82.4 | 83.1 |
| | VCD | 83.1 | 84.1 | 83.7 | 85.1 | 83.0 | 84.1 |
| | VAF | 84.5 | 84.9 | 85.2 | 86.4 | 84.9 | 85.1 |
| | DeCo | 84.6 | 85.8 | 78.4 | 81.7 | 85.5 | 84.9 |
| | **HulluEdit** | **87.5** | **87.6** | **88.0** | **88.3** | **88.2** | **87.7** |
| Adversarial | Greedy | 77.6 | 79.4 | 77.8 | 79.5 | 77.2 | 78.9 |
| | VCD | 78.1 | 79.6 | 78.2 | 79.7 | 78.8 | 80.1 |
| | VAF | 80.1 | 81.0 | 80.7 | 81.7 | 80.4 | 81.2 |
| | DeCo | 78.3 | 81.1 | 72.6 | 77.9 | 81.5 | 81.5 |
| | **HulluEdit** | **82.5** | **83.4** | **82.7** | **84.0** | **84.3** | **84.2** |

CHAIR on 500 MSCOCO images (max 64 tokens), lower is better. BLEU measures caption quality.

| Method | LLaVA-1.5 CHAIRs | CHAIRi | BLEU | MiniGPT-4 CHAIRs | CHAIRi | BLEU | mPLUG-Owl2 CHAIRs | CHAIRi | BLEU |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | 20.40 | 7.08 | 15.72 | 32.40 | 12.20 | 14.57 | 22.90 | 8.62 | 15.01 |
| OPERA | 17.50 | 6.07 | 16.02 | 29.70 | 11.96 | 14.82 | 20.07 | 7.18 | 15.41 |
| VCD | 20.30 | 7.28 | 14.53 | 29.00 | 12.64 | 14.42 | 22.80 | 8.68 | 15.14 |
| HALC | 16.90 | 5.72 | 16.02 | 25.20 | 9.42 | 14.91 | 18.80 | 7.00 | 15.33 |
| Nullu | 15.20 | 5.30 | 15.69 | **21.40** | 8.99 | 14.81 | 15.60 | 5.77 | 15.45 |
| **HulluEdit** | **13.00** | **4.18** | 15.49 | 23.60 | **8.28** | 14.76 | **13.60** | **3.35** | 15.34 |

- The gains are largest where priors matter most: on POPE Adversarial, +4.9 accuracy for LLaVA-1.5-7B and +7.1 for Qwen-VL-Chat over greedy.
- On CHAIR, HulluEdit roughly cuts LLaVA-1.5's CHAIRi from 7.08 to 4.18 and mPLUG-Owl2's from 8.62 to 3.35. On MiniGPT-4 its CHAIRs (23.60) is worse than Nullu's (21.40). BLEU stays close to greedy.
- The ablation on LLaVA-1.5-7B shows the gate matters most: without it, CHAIRs rises to 22.90, worse than plain greedy. Dropping the orthogonal complement (15.90), using a single layer for both estimate and edit (18.20), or anchor layer 20 (19.72) also hurt.
- On MME, Existence (+13.33), Position (+22.23) and Color (+7.22) improve, but Count drops by 13.33. The authors suggest count information may live in the residual subspace that gets shrunk.

Speed is the other selling point. Without a second forward pass, HulluEdit runs at 22.31 tokens/s, close to DoLa and faster than VCD, OPERA and HALC.

![Decoding throughput of HulluEdit and baselines](/images/blog/papers/hulluedit-review-throughput.webp)
*Throughput on 500 MSCOCO images. Greedy 31.25, DoLa 23.61, HulluEdit 22.31, VCD 17.58, OPERA 4.36, HALC 2.15 tokens/s. Figure from the paper.*

The appendix checks the non-interference claim directly by measuring how much each edit moves the hidden state inside $$U$$ and inside $$P$$.

![Distribution of edit magnitudes in the visual and anti-prior subspaces](/images/blog/papers/hulluedit-review-subspace-change.webp)
*Norm of the per-token change $$\Delta h$$ projected onto $$U$$ (blue) and $$P$$ (orange). Changes in the visual subspace sit around $$10^{-3}$$, changes in the anti-prior subspace around $$10^{1}$$. Figure from the paper (appendix).*

![Qualitative example: LLaVA vs. HulluEdit](/images/blog/papers/hulluedit-review-qualitative.webp)
*LLaVA-1.5 invents a rider and a backpack (red). HulluEdit drops both and mentions the White Mountain National Forest sign, but its caption repeats "sign" several times. Figure from the paper.*

## Limitations

- **Count gets worse.** On MME, Count drops from 118.33 to 105.00. Shrinking the residual subspace removes information that is not in $$U$$ but is still needed.
- **The "visual subspace" is an assumption.** $$U$$ comes from anchor-layer image tokens, but it is used to decompose a last-layer hidden state. Nothing ensures that these directions mean the same thing at both layers. The guarantee "edits never touch visual evidence" holds for $$U$$, not for visual evidence as such.
- **Not exactly zero.** The appendix shows changes inside $$U$$ of order $$10^{-3}$$, while the closed form implies they should be exactly zero. The paper does not explain the gap.
- **Fluency.** The paper's own qualitative example repeats "sign" four times in two sentences. BLEU is reported but no fluency or repetition metric is.
- **Many knobs.** Anchor layer, $$r$$, $$q$$, $$\kappa$$, $$\lambda_0$$, $$\lambda_{\max}$$, $$\gamma_v$$, $$\gamma_p$$ and the text-cache window. The gate thresholds matter a lot (without the gate CHAIRs is 22.90), and the anchor layer has to be picked per architecture. The sensitivity study covers only $$r$$, $$q$$ and $$\kappa$$.
- **Short outputs.** All captioning runs cap generation at 64 tokens, which favours CHAIR and leaves long-form behaviour untested.

## My Thoughts

<!-- TODO -->

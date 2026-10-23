---
title: "DeCo: The Middle Layers Saw the Right Object"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: "MLLMs often rank the true object highest in a middle layer and lose it in the last few layers, so DeCo adds a dynamically chosen middle layer's logits back into the final logits."
conference: ICLR
year: 2025
paper:
  title: "MLLM can see? Dynamic Correction Decoding for Hallucination Mitigation"
  authors: "Chenxi Wang, Xiang Chen, Ningyu Zhang, Bozhong Tian, Haoming Xu, Shumin Deng, Huajun Chen"
  venue: "ICLR 2025"
  url: "https://arxiv.org/abs/2410.11779"
---

## Abstract

When a multimodal LLM describes an object that is not in the image, there are two possible explanations: the model never recognized what was in the image, or it recognized it and the information was overwritten later. Earlier training-free fixes (VCD, OPERA) assume language priors are to blame, but they do not check where in the network the visual fact is lost. They also cost a second forward pass or an expensive beam-level penalty.

The paper first shows that the second explanation is common. Probes on intermediate hidden states detect object presence well, and in an early-exit view the correct token often has high probability in layers around 20 to 28 before it drops in the last layers. **DeCo** (Dynamic Correction Decoding with preCeding-Layer Knowledge) uses this. At each step it picks the intermediate layer where a candidate token is most confident, and adds that layer's logits, scaled by its confidence, to the final logits. It needs no training, works with greedy, beam search or nucleus sampling, and adds about 1.2x latency.

## Motivation

**Finding 1: the model knows, to some extent, whether an object exists.** Using the prompt "Describe the image. ASSISTANT: The image contains obj.", the authors train a probe classifier on the last-position hidden state of each of the 32 layers of LLaVA-1.5 (7B), using 1,200 balanced object/label pairs built from MSCOCO. The best probe detects object existence well on AMBER captions and on the three POPE splits. Although the model mentions many non-existent objects in its captions, the probe stays at around 80% accuracy or higher. Probe accuracy peaks in the middle layers and declines toward the output.

![Object probing results and probe accuracy by layer](/images/blog/papers/deco-review-probing.webp)
*(a) Best-layer probe accuracy for object existence on AMBER (in-distribution and OOD) and POPE. (b) Probe accuracy by layer for 224 px and 336 px vision encoders. Accuracy peaks in the middle layers and falls off toward the last layer. Figure from the paper.*

**Finding 2: language priors suppress what the model already sees.** Following each token through the layers with early exit shows the pattern directly. In the example below, correct tokens ("people", "left") reach high probability around layer 18 and keep it. For the hallucinated "green" and "bird", the ground-truth alternatives ("blue", "umbrella") reach high probability in the intermediate layers (around layers 20 to 30), then drop sharply from layer 30 on, and the hallucinated token wins in the last layer.

![Token probabilities across layers for hallucinated and non-hallucinated tokens](/images/blog/papers/deco-review-motivation.webp)
*Next-token probabilities by layer for four positions of a LLaVA caption. For correct tokens (green, left) the probability rises in the middle layers and stays. For hallucinated tokens (orange, right) the ground-truth token ("blue", "umbrella") is high in the intermediate layers and falls off at the end, where "green" and "bird" take over. Figure from the paper.*

The authors quantify this on 500 MSCOCO captions. For each hallucinated object they check whether a ground-truth token in the candidate set beats the hallucinated token by a margin in some earlier layer. Such "activated" ground-truth tokens appear mostly between layers 20 and 28. To ask what pushes them down, they re-run the prefix without the image: 91.05% of the hallucinated tokens are also in the candidate set generated with no image at all. So the hallucinated token is largely what the language model would say anyway.

## Method

![DeCo framework](/images/blog/papers/deco-review-main.webp)
*Top: candidate tokens come from top-p truncation of the final layer; the anchor layer is the preceding layer where a candidate reaches the highest probability (layer 24, "tree", 0.92). Bottom: the anchor layer's logits, scaled by $$\alpha$$ and max_prob, are added to the final logits. Plain greedy says "car" and goes on to a backpack; DeCo says "tree". Without max_prob the output drifts into more hallucination. Figure from the paper.*

### Dynamic preceding-layer selection

The search is restricted to a few high-quality tokens. The candidate set $$\mathcal{V}_{\text{candidate}}$$ is the top-p truncation of the final layer's distribution, so tokens that only look good in early layers but are nonsense at the output are excluded. The **anchor layer** is the layer in a fixed range $$[a, b]$$ where some candidate token has the highest probability:

$$
A = \arg\max_{i \in [a, b]} \; \max_{x \in \mathcal{V}_{\text{candidate}}} \mathrm{softmax}\big(\phi(h^i_{T-1})\big)_x
$$

Here $$\phi$$ is the LM head applied to an intermediate hidden state, as in early exit. With $$[a, b] = [20, 28]$$ for 32-layer 7B models, the token picked this way is the ground-truth token 61.69% of the time (71.14% with $$[15, 28]$$).

### Preceding-layer knowledge guided decoding

The anchor layer's logits are added to the final logits:

$$
\mathrm{logits} = \phi(h^N_{T-1}) + \alpha \cdot \text{max\_prob} \cdot \phi(h^A_{T-1}), \qquad \text{max\_prob} = \max \mathrm{softmax}\big(\phi(h^A_{T-1})\big)
$$

$$\alpha$$ (0.1 to 0.6) sets the overall strength. The **dynamic soft modulation** term max_prob makes the correction weak when the anchor layer is itself unsure, so close calls between candidates do not cause abrupt jumps in the logits. The corrected logits then go to any decoding strategy. Unlike DoLa, which subtracts an early layer to remove shallow semantics, DeCo adds an intermediate layer to bring back visual knowledge.

## Main Experiments

CHAIR on 500 MSCOCO val2014 images with the prompt "Please help me describe the image in detail." (lower is better). All models are 7B. Each baseline is compared under its own decoding strategy.

| Decoding | Method | InstructBLIP CHAIRs | CHAIRi | MiniGPT-4 CHAIRs | CHAIRi | LLaVA-1.5 CHAIRs | CHAIRi | Qwen-VL CHAIRs | CHAIRi |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | Vanilla | 58.8 | 23.7 | 31.8 | 9.9 | 45.0 | 14.7 | 46.0 | 12.5 |
| | DoLa | 48.4 | 15.9 | 32.2 | 10.0 | 47.8 | 13.8 | 46.8 | 12.9 |
| | **DeCo** | **41.2** | **14.4** | **27.0** | **8.8** | **37.8** | **11.1** | **42.2** | **10.7** |
| Beam Search | Vanilla | 55.6 | 15.8 | 30.6 | 9.5 | 48.8 | 13.9 | 41.8 | 10.8 |
| | OPERA | 46.4 | 14.2 | 26.2 | 9.5 | 44.6 | 12.8 | 34.6 | 9.5 |
| | **DeCo** | **43.8** | **12.7** | **24.8** | **7.5** | **33.0** | **9.7** | **32.0** | **8.7** |
| Nucleus | Vanilla | 54.6 | 24.8 | 32.6 | 10.7 | 48.8 | 14.2 | 49.2 | 13.1 |
| | VCD | 58.0 | 17.0 | 33.8 | 11.1 | 54.0 | 16.0 | 46.4 | 11.9 |
| | **DeCo** | **43.6** | **12.9** | **30.8** | **9.5** | **42.8** | **13.2** | **43.8** | **11.8** |

POPE F1 (higher is better):

| Decoding | Method | InstructBLIP | MiniGPT-4 | LLaVA-1.5 | Qwen-VL |
|---|---|---|---|---|---|
| Greedy | Vanilla | 80.0 | 58.5 | 82.2 | 85.2 |
| | DoLa | 83.4 | 72.8 | 83.2 | 85.8 |
| | **DeCo** | **84.9** | **77.4** | **86.7** | **86.3** |
| Beam Search | Vanilla | 84.4 | 70.3 | 84.9 | 85.3 |
| | OPERA | 84.8 | 73.3 | 85.4 | 86.1 |
| | **DeCo** | **84.9** | **77.9** | **86.7** | **86.4** |
| Nucleus | Vanilla | 79.8 | 52.8 | 83.1 | 84.5 |
| | VCD | 79.9 | 56.0 | 83.1 | 84.7 |
| | **DeCo** | **81.8** | **63.8** | **85.4** | **85.2** |

- DeCo lowers CHAIR in every model and decoding setting, with an average suppression of about 10.8%. With beam search on LLaVA-1.5, CHAIRs drops from 48.8 to 33.0, against 44.6 for OPERA.
- In the paper's runs VCD often increases CHAIRs under nucleus sampling (e.g. 48.8 → 54.0 on LLaVA-1.5), while DeCo decreases it.
- GPT-4o judging on 100 COCO images gives higher accuracy for DeCo on all four models (LLaVA-1.5: 5.21 → 7.42), with detailedness and coherence roughly unchanged and sometimes slightly lower. On AMBER (LLaVA-1.5, greedy), CHAIR goes 8.2 → 6.6 and Hal 34.3 → 28.1, with Cover 48.9 → 47.5.
- Removing max_prob raises CHAIRs from 37.8 to 41.2 on LLaVA-1.5 and from 42.2 to 45.8 on Qwen-VL. Randomly shifting the selected layer by up to ±5 also hurts on all four models.

Cost is low because DeCo only applies the LM head to the hidden states of the nine layers in the interval, which the forward pass already computes.

![Latency and throughput of DeCo vs. baselines](/images/blog/papers/deco-review-latency.webp)
*Latency (ms/token) and throughput (tokens/s) for vanilla decoding, DeCo, and the matching baseline (DoLa for greedy, OPERA for beam search, VCD for nucleus). DeCo adds about 1.2x latency, against 1.8x for VCD and 5.1x for OPERA. Figure from the paper.*

![Ablation of alpha and of the layer interval](/images/blog/papers/deco-review-ablation.webp)
*(a) CHAIRs vs. $$\alpha$$ for four MLLMs: hallucination generally falls as $$\alpha$$ grows, with the paper reporting the strongest suppression around 0.6. (b) CHAIRs when the anchor layer is chosen from different 4-layer intervals: layers 1-16 do not help, 17-28 help most, and 29-32 make things worse than greedy for MiniGPT-4. Figure from the paper.*

## Limitations

- **Fixed layer interval.** $$[20, 28]$$ is chosen for 7B, 32-layer models. The authors state that other families and larger models need their own interval, found empirically, and only 7B models are tested.
- **Hit rate is modest.** The token picked from the anchor layer is the ground-truth one only 61.69% of the time with layers 20-28. When it is wrong, DeCo pushes the logits toward the wrong token, and the max_prob scaling softens this only when the anchor layer is unsure.
- **Repetition and detail.** The authors note that large $$\alpha$$ produces atypical descriptions with repeated words, and that DeCo slightly lowers detailedness in GPT-4o judging and coverage on AMBER.
- **Additive, not contrastive.** Adding intermediate logits also adds whatever priors already exist in those layers. The method relies on the empirical finding that those layers are more visual, not on removing the prior directly.
- **Evaluation scope.** Object hallucination in captions and yes/no POPE are the main tests. The 91.05% overlap with no-image candidates supports the "language prior" explanation but is correlational.

## My Thoughts

<!-- TODO -->

---
title: "OPERA: Penalize Attention Columns, Roll Back When They Persist"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: "Hallucinations tend to follow tokens that soak up all the attention, so beam search penalizes candidates that over-trust such a token and rolls back to it when the pattern does not go away."
conference: CVPR
year: 2024
paper:
  title: "OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation"
  authors: "Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, Nenghai Yu"
  venue: "CVPR 2024"
  url: "https://arxiv.org/abs/2311.17911"
---

## Abstract

Multimodal LLMs hallucinate objects, attributes and relations that are not in the image, especially in long descriptions. Existing fixes either train on specially designed data or bring in external knowledge or models, and both cost extra. OPERA works purely at decoding time, without extra data, knowledge or training.

It starts from an observation in the self-attention maps. Many hallucinations begin shortly after a **knowledge aggregation pattern**: a column in the attention map where later tokens all attend strongly to one earlier token, usually an uninformative one such as a period. The model "over-trusts" these summary tokens and neglects the image tokens. OPERA adds a penalty for this pattern to the beam search score, and when the pattern persists, it rolls the decoding back to the summary token and picks a different next token.

## Motivation

The authors visualize the self-attention of generated tokens and see a recurring picture: a column-shaped attention pattern on a token with little content, followed by hallucinated content.

![InstructBLIP attention map with a columnar pattern before a hallucination](/images/blog/papers/opera-review-motivation.webp)
*InstructBLIP describes a skateboarding image. After the period following "him", the following tokens all attend strongly to that period (red box), and the next sentence hallucinates "two cars parked in the area". Figure from the paper.*

This matches the "anchor token" finding in NLP: LLMs aggregate earlier information onto a few anchor tokens at shallow layers and predict from them at deep layers. In an MLLM, the image tokens sit at the start of the sequence. As the text grows, visual information must pass through a chain of summary tokens, and a single summary token cannot hold everything. Later tokens then over-trust the nearby summary token instead of the image and fall back on learned bias, for example "cars" because "road" was mentioned earlier.

Two measurements support this. Across 100 images per model, most hallucinations start within 10 tokens after such a pattern, on all four models. And when long responses are split at the summary tokens, CHAIR rises with the split number, with only small dips along the way.

![Anchor patterns and CHAIR by split of generated text](/images/blog/papers/opera-review-chair-split.webp)
*(a) The aggregation pattern matches the "anchor token" picture: information flows through a few columns. (b), (c) CHAIR-S and CHAIR-I on 5,000 random MSCOCO images grow as more summary tokens appear in the context (split 0 to 6). Figure from the paper.*

## Method

OPERA is built on beam search with $$N_{\text{beam}}$$ beams. It changes the score of each candidate and adds a rollback rule.

### Over-trust logit penalty

![Over-trust penalty computation](/images/blog/papers/opera-review-main.webp)
*From the last-layer attention (max over heads), OPERA cuts a local window over the generated tokens, scales it by $$\sigma$$, multiplies down each column, and takes the largest column score as the penalty. Figure from the paper.*

For the current sequence, take the last-layer attention weights among the last $$k$$ generated tokens (image and prompt tokens are excluded), using the maximum over heads. Zero the upper triangle and scale the values by $$\sigma$$, since raw attention values are small. A strong column means many later tokens attend to the same token, so the column-wise product measures it:

$$
\phi(\omega_{<t}) = \prod_{i=c}^{t-1} \sigma \omega_{i,c}, \qquad c = \arg\max_{t-k \le j \le t-1} \prod_{i=j}^{t-1} \sigma \omega_{i,j}
$$

With $$\sigma = 50$$, attention values in a strong column end up above 1 and weak ones below 1, so the product grows only for real columns. For efficiency, and to avoid strange outputs, each beam only considers its top-$$N_{\text{can}}$$ tokens, which gives a candidate set $$\mathcal{Y}$$. The penalty enters the logits of each candidate:

$$
p(x_t \mid x_{<t}) = \mathrm{Softmax}\big[\mathcal{H}(h_t) - \alpha\, \phi(w_{\le t})\big]_{x_t}, \qquad x_t \in \mathcal{Y}
$$

where $$w_{\le t}$$ is the attention after appending candidate $$x_t$$. The penalty accumulates in the beam score, so a beam that keeps over-trusting a summary token falls behind.

### Retrospection-allocation

![Retrospection strategy](/images/blog/papers/opera-review-retrospection.webp)
*For each of the recent tokens, OPERA records the location of the maximum column score. If the same location $$s$$ appears at least $$r$$ times, the decoding rolls back to $$s$$ and picks a different next token. Figure from the paper.*

The pattern shows up with a delay: by the time it is visible, every beam may already contain the hallucination, and the penalty cannot help. OPERA then goes back. It collects the column locations $$c$$ for the last $$l$$ tokens into a set $$\mathcal{C}$$ and counts how often the most common one appears:

$$
N_{\text{overlap}} = \sum_{c \in \mathcal{C}} \mathbb{1}_{c = s}, \qquad s = \mathrm{Mode}(\mathcal{C})
$$

If $$N_{\text{overlap}} \ge r$$, the sequence is cut back to $$x_0, \dots, x_s$$, and the next token is chosen from $$\mathcal{Y} \setminus \{x_{s+1}\}$$, excluding the token picked before. Rollback positions may not move backward, and after $$\beta$$ rollbacks at the same position, the rollback goes one token further back. Counting locations is used instead of a threshold on the score, because the score scale varies across models.

The defaults are $$N_{\text{beam}} = 5$$, $$N_{\text{can}} = 5$$, $$\sigma = 50$$, $$\alpha = 1$$, $$\beta = 5$$, $$r = 15$$, and $$l = k$$, for all models.

## Main Experiments

Four 7B MLLMs are tested: InstructBLIP and MiniGPT-4 (Q-Former, 32 image tokens), and LLaVA-1.5 and Shikra (linear projection, 256 or 576 image tokens). CHAIR is measured on 500 random COCO val2014 images with the prompt "Please describe this image in detail." and up to 512 new tokens. CS is CHAIR-S (sentence level: share of captions with a hallucinated object) and CI is CHAIR-I (instance level: share of mentioned objects that are hallucinated); lower is better:

| Method | InstructBLIP CS | InstructBLIP CI | MiniGPT-4 CS | MiniGPT-4 CI | LLaVA-1.5 CS | LLaVA-1.5 CI | Shikra CS | Shikra CI |
|---|---|---|---|---|---|---|---|---|
| Greedy | 58.8 | 23.7 | 31.8 | 9.9 | 45.0 | 14.7 | 55.8 | 15.4 |
| Nucleus | 54.6 | 24.8 | 32.6 | 10.7 | 48.8 | 14.2 | 55.6 | 15.4 |
| Beam Search | 55.6 | 15.8 | 30.6 | **9.5** | 48.8 | 13.9 | 50.4 | 13.3 |
| DoLa | 48.4 | 15.9 | 32.2 | 10.0 | 47.8 | 13.8 | 55.8 | 15.1 |
| **OPERA** | **46.4** | **14.2** | **26.2** | **9.5** | **44.6** | **12.8** | **36.2** | **12.1** |

POPE (average F1 over random, popular and adversarial splits):

| Method | InstructBLIP | MiniGPT-4 | LLaVA-1.5 | Shikra |
|---|---|---|---|---|
| Greedy | 80.0 | 58.5 | 82.2 | 81.1 |
| Nucleus | 80.1 | 57.8 | 82.5 | 81.2 |
| Beam Search | 84.4 | 70.3 | 84.9 | 82.5 |
| DoLa | 83.4 | 72.8 | 83.2 | 82.1 |
| **OPERA** | **84.8** | **73.3** | **85.4** | **82.7** |

- On long captions, OPERA has the lowest CHAIR on every model. The largest gain is on Shikra, with CHAIR-S 50.4 → 36.2 over beam search. With 64 new tokens it still wins on most entries, for example LLaVA-1.5 CHAIR-S 18.8 → 14.2.
- On POPE the gain over beam search is small (0.2 to 3.0 F1). The authors say this is expected: one-word answers end before an aggregation pattern can form.
- Both parts help. Over beam search on Shikra, the penalty alone brings CHAIR-S to 41.8, the rollback alone to 46.8, and both to 36.2. The penalty contributes more.
- Text quality holds up on LLaVA-1.5: perplexity is close to beam search (PPL1 11.67 vs. 11.11), and MME rises slightly (1504.3 → 1515.4).

The authors also use GPT-4 to judge descriptions sentence by sentence on VG-100K (HalluBench), which also covers attribute and relation errors.

![GPT-4 assisted hallucination evaluation on VG-100K](/images/blog/papers/opera-review-gpt4-eval.webp)
*Six measures on four models: sentences and words per image (SPI, WPI; larger is better) and hallucinated sentences and words per image and their ratios (HSPI, HWPI, HSR, HWR; smaller is better, and the axes are flipped so a larger radar is better). OPERA (red) has the largest area on the hallucination axes, with somewhat shorter outputs on some models. Figure from the paper.*

With GPT-4V comparing beam search and OPERA side by side on 500 COCO images, correctness goes from about 5.3-5.5 to 6.3-6.9 out of 10 on all four models, while detailedness stays about the same.

## Limitations

- **Short answers.** The authors state that gains are marginal for answers under about 10 tokens, because the pattern appears with a delay. This shows in the POPE table.
- **Not every hallucination.** If the model's bias is too strong (for example "road" → "cars"), the rollback may find no good alternative. Errors from weak perception, such as similar shapes or low resolution, are out of reach.
- **Cost.** OPERA runs beam search with 5 beams and 5 candidates per beam, needs the attention of each candidate continuation, and can roll back and regenerate. That is much slower than greedy decoding, and the paper reports no runtime.
- **Tied to beam search and attention access.** It cannot be dropped into sampling-based decoding, and it needs explicit attention weights, which fused attention kernels do not return.
- **Shorter outputs.** The GPT-4 evaluation shows somewhat fewer sentences and words per image. The authors attribute this to removing hallucinated content, but part of the lower hallucination count may simply come from saying less.

## My Thoughts

<!-- TODO -->

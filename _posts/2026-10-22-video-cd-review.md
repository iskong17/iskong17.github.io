---
title: "VTD: Contrast Against a Video With Its Motion Wiped Out"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: "Build a copy of the video with the temporal cues removed but the misleading static content kept, and decode away from what the model says about it."
conference: NeurIPS
year: 2025
paper:
  title: "Improve Temporal Reasoning in Multimodal Large Language Models via Video Contrastive Decoding"
  authors: "Daiqing Qi, Dongliang Guo, Hanzhang Yuan, Handong Zhao, Mengxuan Hu, Lehan Yang, Sheng Li"
  venue: "NeurIPS 2025"
  url: "https://openreview.net/forum?id=2nIAtsUC27"
---

## Abstract

Video LLMs handle captioning and recognition within single frames well, but often fail at **temporal reasoning**: continuous actions, direction, speed, order of events, or how an object changes over time. Prior fixes improve the video encoder or the LLM's sensitivity to order, and all of them need training data and compute. Findings from one architecture also do not always carry over to another.

The paper looks at model behaviour instead of model internals. When a video LLM gets temporal questions wrong, it is often following a **language prior** (what the text alone suggests) or an **"image" prior** (what a single static frame suggests). The method turns this around: it builds a temporally distorted copy of the video that removes temporal cues but keeps the misleading static content, and uses the model's answer on that copy as the negative in contrastive decoding. The paper does not give the method a name. Its distribution is written $$p_{\text{vtd}}$$ and its distortion step is called Video Temporal Distortion, so this review calls it **VTD**. It needs no training.

## Motivation

The authors first check whether language priors matter for video. They take 200 TempCompass questions that LLaVA-Video-7B-Qwen2 and Video-LLaVA got wrong and ask the LLM backbone the same question without the video. Among the wrong predictions, **46.7%** (LLaVA-Video) and **38.9%** (Video-LLaVA) match the answer of the blind LLM, far above chance.

![Language and image priors in image and video understanding](/images/blog/papers/video-cd-review-motivation.webp)
*(1) LLaVA-1.5 calls a cube-shaped watermelon an ellipsoid, the same answer its LLM gives without the image. (3) LLaVA-Video says the toast is being eaten, matching the blind LLM, while the video shows the reverse. (4) For a rising sun, the blind LLM answers correctly ("Rising"), yet LLaVA-Video answers "Falling". (2) A single middle frame fed to Qwen2-VL looks like a sunset and also gets "Falling", which exposes the "image" prior. VTD fixes both (3) and (4). Figure from the paper.*

Case (4) is the new part. The language prior was right, and the model was still wrong because a static frame looks like a sunset. The authors call this the **"image" prior**: static visual features that carry no temporal information but mislead the model about temporal dynamics. A good negative for contrastive decoding should therefore keep these misleading static cues and remove only the temporal ones. Masking the entire video is not enough: in case (4), the video-free model gives the correct answer, and using a correct answer as the negative would push the output away from it.

Simple distortions (feature noise, frame shuffling, frame dropping) do help on EventHallusion with LLaVA-Video-7B, but the gains depend on whether the distortion hits the first or the last half of the frames. That instability motivates an adaptive distortion.

## Method

![VTD overview](/images/blog/papers/video-cd-review-main.webp)
*Left: the original video embeddings $$V$$ go through the LLM to get $$\mathrm{logit}(y \mid V, x)$$. The attention maps of that pass guide the Temporal Distortion Unit, which builds $$V'$$. The wrong option "B" (climbing up) gets a high score under $$V'$$, so the contrast leaves the ground truth "A" (falling down). Right: the Temporal Distortion Unit fuses key frames, distorts key tokens in the remaining frames, and disrupts moving content. Figure from the paper.*

### Contrastive decoding

The decoding rule is the [VCD](/blog/2026/10/22/vcd-review/) form with a video:

$$
p_{\text{vtd}}(y \mid V, V', x) = \mathrm{softmax}\big[(1+\alpha)\,\mathrm{logit}_\theta(y \mid V, x) - \alpha\,\mathrm{logit}_\theta(y \mid V', x)\big]
$$

with the adaptive plausibility constraint of Li et al.: only tokens whose probability is at least $$\beta$$ times the top probability are kept. The defaults are $$\alpha = 1$$ and $$\beta = 0.2$$. All the work is in how $$V'$$ is built. The authors list three criteria: remove temporal information, keep priors that are likely to mislead (salient static features), and adapt to each video instead of distorting at random.

### Token and frame importance

From the first forward pass, the attention of the last query position over each visual token is averaged over heads at each layer, $$S_l$$. To avoid relying on the last layer only, the scores are accumulated across layers with momentum:

$$
\tilde S_l = w_m \tilde S_{l-1} + (1 - w_m) S_l
$$

Frame importance is the sum of token importances in that frame.

### Three distortions

1. **Key frame fusion.** The top-$$w_{fdr}$$ most important frames are each replaced by the mean of those frames, with a little Gaussian noise (weight $$w_{fpr}$$). Temporal order among them disappears, but the coarse scene stays. Mean pooling keeps more confounding context than dropping and is more stable than shuffling.
2. **Key token distortion.** In every remaining frame, the top-$$w_{tdr}$$ most important tokens are masked. Fusing the remaining frames as well would remove too much and destroy the misleading priors, so the distortion is token-level here.
3. **Moving content disruption.** Attention is not always reliable, so this step uses visual similarity. Frames are grouped into non-overlapping windows of size $$w_{ws}$$, and tokens are averaged within $$w_{bs} \times w_{bs}$$ blocks. A block's similarity is its average cosine similarity with the same block in the other frames of the window. The $$w_{cfr}$$ fraction of least similar blocks are marked dynamic, and dynamic blocks at the same position are replaced by their mean across frames.

![Disrupting moving content within a sliding window](/images/blog/papers/video-cd-review-moving-content.webp)
*Moving content disruption with window size 3. Left: blocks marked dynamic (orange) in each frame. Right: dynamic blocks at the same position are averaged across frames, which smooths out the motion and keeps the static context. Figure from the paper.*

The distorted key frames and the distorted remaining frames together form $$V'$$.

## Main Experiments

Backbones are Video-LLaVA (8 frames) and LLaVA-Video-7B-Qwen2 (32 frames). The baselines are image-level contrastive decoding methods applied to video (VCD, SID) and TCD, which drops frames at random. TempCompass reports accuracy for four answer formats:

| Model | Multi-Choice | Yes/No | Matching | Generation | Overall |
|---|---|---|---|---|---|
| Video-LLaVA | 45.5 | 56.3 | 63.3 | 34.8 | 49.8 |
| + VCD | 46.5 | 57.1 | 64.1 | 35.3 | 50.4 |
| + SID | 46.6 | 57.1 | 64.3 | 35.4 | 50.5 |
| + TCD | 46.8 | 57.2 | 64.3 | 35.4 | 50.6 |
| **+ VTD** | **48.1** | **58.3** | **66.4** | **36.9** | **52.1** |
| LLaVA-Video-7B | 65.8 | 66.8 | 71.8 | 55.8 | 65.0 |
| + VCD | 66.1 | 67.4 | 73.1 | 56.5 | 65.7 |
| + SID | 66.2 | 67.5 | 73.2 | 56.4 | 65.8 |
| + TCD | 66.1 | 67.4 | 73.2 | 56.5 | 65.7 |
| **+ VTD** | **68.0** | **69.4** | **75.2** | **57.8** | **67.5** |

EventHallusion (continuous events; accuracy) and Video-MME (accuracy), both with LLaVA-Video-7B-Qwen2:

| Model | EH Entire | EH Interleave | EH Misleading | EH Overall | MME Temp. Perception | MME Temp. Reasoning | MME Overall |
|---|---|---|---|---|---|---|---|
| LLaVA-Video-7B | 52.1 | 60.3 | 82.5 | 63.5 | 61.1 | 54.3 | 59.8 |
| + VCD | 53.4 | 65.7 | 82.8 | 66.5 | 75.4 | 55.3 | 60.0 |
| + SID | 53.3 | 65.4 | 82.9 | 66.4 | 75.3 | 55.4 | 59.9 |
| + TCD | 53.6 | 66.8 | 83.3 | 67.2 | 75.9 | 55.8 | 59.8 |
| **+ VTD** | **55.3** | **75.6** | **84.2** | **72.1** | **84.1** | **57.8** | **61.6** |

- VTD beats every contrastive decoding baseline on every TempCompass format for both backbones, by +2.0 to +3.4 points over the base model. The largest per-category jump is event order in matching, +9.0 on LLaVA-Video.
- On EventHallusion the gain is concentrated on interleaved events (60.3 → 75.6), and the overall gain (+8.6) is much larger than VCD's (+3.0).
- On MLVU the overall score goes 67.4 → 68.5, driven by action count (41.8 → 44.1) and action order (57.9 → 62.6), but anomaly recognition (67.8 → 65.7) and topic reasoning (84.5 → 83.8) drop. The authors read this as a trade-off between temporal and static understanding.

The ablation on LLaVA-Video shows every part matters. The full method gets 67.5 on TempCompass and 72.1 on EventHallusion. Removing attention guidance gives 65.7 / 67.1, removing key frame fusion 66.1 / 66.3, removing key token distortion 66.5 / 68.3, and removing moving content disruption 66.9 / 69.7.

![Hyperparameter sensitivity on TempCompass](/images/blog/papers/video-cd-review-hyperparameters.webp)
*TempCompass accuracy of LLaVA-Video-7B-Qwen2 while each hyperparameter is varied. The distortion-strength knobs (frame distortion ratio, token distortion ratio, content fusion ratio, block size) peak at moderate values and fall when the distortion is too strong. Figure from the paper.*

The sensitivity plots support the central argument: too little distortion leaves the temporal cues, too much distortion makes the negative branch random, and only a moderate distortion produces the "most probable mistake" that is useful to subtract.

## Limitations

- **Attention as a guide.** The authors note that attention maps do not always reflect real token importance and give only coarse estimates.
- **Visual only.** Subtitles are not distorted, though many videos come with them.
- **Cost.** Each token needs two forward passes, and the first one must return attention maps from intermediate layers. The appendix admits CD-based methods are slower and proposes running the two passes on parallel CUDA streams, which it says matches the vanilla speed at a higher peak memory. No timing table is given, and $$V'$$ depends on the attention of the first pass, so the two passes cannot be fully independent.
- **Many hyperparameters, tuned per benchmark.** There are eight knobs plus the momentum ($$\alpha$$, $$\beta$$, $$w_{fdr}$$, $$w_{tdr}$$, $$w_{ws}$$, $$w_{cfr}$$, $$w_{bs}$$, $$w_{fpr}$$), and the appendix uses different values for TempCompass, EventHallusion, and Video-MME/MLVU.
- **Static understanding can suffer.** MLVU anomaly recognition and topic reasoning drop, so the method is best used when the question is known to be temporal.
- **An odd baseline jump.** On Video-MME temporal perception, plain VCD alone already moves LLaVA-Video from 61.1 to 75.4, so much of the gain on that subset comes from contrastive decoding in general, not from the temporal distortion.

## My Thoughts

<!-- TODO -->

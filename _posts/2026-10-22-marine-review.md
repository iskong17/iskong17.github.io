---
title: "MARINE: Let an Object Detector Guide the LVLM's Logits"
category: paper-review
subfields: [hallucination, vlm]
lang: en
description: "Turn the objects found by DETR and RAM++ into a text prompt, run the LVLM with and without it, and mix the two logit vectors in a classifier-free-guidance style to suppress hallucinated objects."
conference: ICML
year: 2025
paper:
  title: "Mitigating Object Hallucination in Large Vision-Language Models via Image-Grounded Guidance"
  authors: "Linxi Zhao, Yihe Deng, Weitong Zhang, Quanquan Gu"
  venue: "ICML 2025"
  url: "https://arxiv.org/abs/2402.08680"
---

## Abstract

LVLMs often mention objects that are not in the image. Early fixes either fine-tune on curated data (LURE and others), which is costly, or send the output to a proprietary LLM for correction (Woodpecker uses GPT-3.5), which costs API calls and raises privacy concerns. Neither addresses the visual side of the problem.

MARINE (Mitigating hallucinAtion via image-gRounded guIdaNcE) is **training-free and API-free**. It runs open-source vision models (the DETR object detector and the RAM++ tagger) on the image, keeps the objects they agree on, and writes them into a short text prompt. The LVLM then decodes twice per token, with and without this guidance prompt, and the two log-probability vectors are mixed with a guidance strength $$\gamma$$, similar to classifier-free guidance. Across five LVLMs, MARINE lowers CHAIR and improves POPE while keeping the extra decoding cost at 1.98 times greedy.

## Motivation

The authors trace object hallucination in an LVLM to three sources: (1) the vision encoder gives too little visual detail, (2) information is distorted or lost when visual features are projected into the text space, and (3) the language model has its own hallucination tendencies. The first two are specific to LVLMs. The vision encoders of most LVLMs come from CLIP, which is trained to match an image with a caption as a whole, not to list every object. Dedicated object detection models are trained for exactly that. So the question is how to pass a detector's object-level information to the LVLM without training a new alignment layer.

![Hallucination examples fixed by MARINE](/images/blog/papers/marine-review-motivation.webp)
*Top: asked "Is there a chair in the image?" about a cockatoo, LLaVA answers "Yes, there is a white chair". With MARINE it answers "No" and names the white bird. Bottom: LLaVA-v1.5 adds a toaster to a kitchen caption, and with MARINE it names the laptop that is actually there. Figure from the paper.*

The appendix shows that simply prompting the model to be careful does not solve this. A hand-written prompt ("Describe the visible contents of this image ... without adding any information not clearly visible") makes CHAIR worse when it replaces the query, for example on LLaVA-v1.5 CHAIR-S goes from 8.8 to 19.6. It also makes CHAIR worse when added as extra context (8.8 to 12.6). MARINE gives 6.2. The model needs to be told *which* objects are there, and the guidance has to act on the logits, not only appear in the prompt.

## Method

![MARINE overview](/images/blog/papers/marine-review-main.webp)
*The image goes to the LVLM's own vision encoder and also to a "vision toolbox" of guidance models. Their detections become the guidance text ("focusing on the visible objects in this image: ..."). The LLM decoder runs with and without that text, and the two output distributions are mixed in log space with weight $$\gamma$$. In the example, the unguided branch prefers the hallucinated "mouse" (0.5), while the mixed distribution picks "magazine". Figure from the paper.*

### Image-grounded guidance prompt

DETR (score threshold 0.95) and RAM++ (threshold 0.68) are run on the image. MARINE keeps the **intersection** of the objects they detect and turns it into a text prompt $$c$$: "focusing on the visible objects in this image:" followed by the object list. Because the guidance is plain text, the LVLM can read it without any new alignment training.

### Guided decoding

At every step, the LVLM is run twice: once on the normal input $$[v, x, y_{<t}]$$ and once with the guidance added, $$[v, c, x, y_{<t}]$$. The two are combined as

$$
\hat p_\theta(y_t \mid v, c, x, y_{<t}) \propto \frac{p_\theta(y_t \mid v, c, x, y_{<t})^{\gamma}}{p_\theta(y_t \mid v, x, y_{<t})^{\gamma - 1}}
$$

or, in log space,

$$
\log \hat p_\theta(y_t) = \gamma \log p_\theta(y_t \mid v, c, x, y_{<t}) + (1 - \gamma) \log p_\theta(y_t \mid v, x, y_{<t})
$$

The formula has the shape of classifier-free guidance, but with $$\gamma \in (0, 1)$$ it is an interpolation, not an extrapolation: $$\gamma = 0$$ is the original model and $$\gamma = 1$$ uses only the guided branch. Because the log-probabilities are added, a token stays near the top only if both branches give it a reasonably high probability. An object that the unguided model likes but the detector-informed branch does not, such as "mouse" in the figure, drops. The default is $$\gamma = 0.7$$ with greedy decoding.

![Token probabilities with and without MARINE](/images/blog/papers/marine-review-logits.webp)
*(a) LLaVA hallucinates a "fork"; with MARINE it does not. (b) At that token, the original probability of "fork" is the highest, the guided ("control") branch gives it little weight, and the mixed output lowers it. (c) Function words like "with" and "various" keep similar probabilities in all three distributions. Figure from the paper.*

## Main Experiments

CHAIR on 500 MSCOCO images, with CS (sentence-level), CI (instance-level), both lower is better, and recall R (higher is better). LLaVA uses LLaMA-2-7B-Chat and LLaVA-v1.5 uses Vicuna-7B. "Avg" is over all five LVLMs (LLaVA, LLaVA-v1.5, MiniGPT-v2, mPLUG-Owl2, InstructBLIP). MARINE-Truth uses ground-truth objects as guidance, as a reference.

| Method | LLaVA CS | LLaVA CI | LLaVA R | LLaVA-v1.5 CS | LLaVA-v1.5 CI | LLaVA-v1.5 R | Avg CS | Avg CI | Avg R |
|---|---|---|---|---|---|---|---|---|---|
| Greedy | 26.6 | 10.5 | 47.4 | 8.8 | 4.6 | 41.1 | 11.0 | 5.2 | 40.3 |
| LURE | 33.8 | 11.6 | 54.8 | 38.9 | 11.2 | 56.3 | 36.2 | 11.4 | 55.2 |
| Woodpecker | 19.5 | 8.9 | 44.3 | 8.5 | 4.5 | 38.4 | 10.3 | 5.7 | 38.0 |
| VCD | 28.1 | 11.0 | 46.6 | 7.3 | 4.1 | 40.8 | 10.1 | 4.8 | 39.4 |
| OPERA | 22.4 | 9.9 | 43.6 | 11.0 | 6.7 | 40.2 | 10.6 | 5.5 | 40.3 |
| **MARINE** | **17.8** | **7.2** | 50.8 | **6.2** | **3.0** | 44.3 | **8.4** | **3.7** | 44.5 |
| MARINE-Truth | 19.6 | 5.1 | 79.0 | 6.0 | 2.5 | 55.3 | 8.9 | 2.9 | 57.5 |

POPE, adversarial setting, averaged over the five LVLMs. The ideal Yes rate is 50%. LURE is left out because it does not produce yes/no answers.

| Method | Accuracy | F1 | Yes (%) |
|---|---|---|---|
| Greedy | 73.2 | 77.9 | 67.0 |
| Woodpecker | 78.8 | 78.3 | 47.8 |
| VCD | 73.2 | 77.5 | 65.6 |
| OPERA | 72.4 | 78.0 | 69.7 |
| **MARINE** | **79.9** | **80.4** | 51.1 |
| MARINE-Truth | 88.3 | 89.7 | 59.8 |

- MARINE has the lowest average CHAIR-S and CHAIR-I and raises recall at the same time (40.3 to 44.5), so it does not reduce hallucination by mentioning fewer objects. LURE has higher recall but hallucinates far more (average CS 36.2).
- On POPE, the main effect is to correct the strong "Yes" bias of the base models. The average Yes rate goes from 67.0% to 51.1%, and on the original LLaVA from 97.7% to 72.3% (accuracy 51.8 to 66.9).
- It is cheap: 52.2 ms per token on LLaVA-7B, 1.98 times greedy, compared with 2.03 times for VCD, 3.59 times for Woodpecker and 7.0 times for OPERA.
- Both detectors are needed, and they must agree. On LLaVA, DETR alone gives CS 27.6 and RAM++ alone 29.0, both worse than greedy (26.6), while their intersection gives 17.8. Taking the union of their objects gives 30.4.
- GPT-4V-judged accuracy goes up with the same detail level. For example, on image captioning, LLaVA goes from 5.27 to 6.11 and mPLUG-Owl2 from 7.97 to 8.63.

![Effect of guidance strength](/images/blog/papers/marine-review-guidance-strength.webp)
*CHAIR-S, CHAIR-I and recall as $$\gamma$$ goes from 0 to 1 for LLaVA, LLaVA-v1.5 and mPLUG-Owl2. Hallucination falls and recall rises as $$\gamma$$ grows. LLaVA-v1.5 and mPLUG-Owl2 reach their lowest CHAIR around 0.3-0.8. Figure from the paper.*

The authors recommend $$\gamma$$ between 0.3 and 0.7. Some models score best on CHAIR at $$\gamma = 1$$, but the appendix shows that too much guidance makes the model list objects from the guidance even when the question is about something else.

## Limitations

- **Every gain depends on the detectors.** MARINE can only remove objects that DETR and RAM++ agree are absent. Their errors pass straight into the guidance. The large gap to MARINE-Truth on POPE (79.9 vs. 88.3 accuracy) shows how much is left on the table.
- **The detector's vocabulary matches the benchmark.** DETR is trained on COCO's object classes, the same vocabulary that CHAIR and POPE-COCO score against. How much of the gain carries over to objects outside that label space, or to attribute and relation hallucinations, is not tested. The authors themselves list other hallucination types as future work.
- **Results vary across models.** On MiniGPT-v2, MARINE makes CHAIR-S worse (8.2 to 11.8). On InstructBLIP POPE, it pushes the Yes rate below balance to 38.8%, an over-correction toward "No". As the [CD Mirage review](/blog/2026/10/22/cd-mirage-review/) argues, POPE gains that come with a large Yes-rate shift should be read with care. Here the starting point is a strong "Yes" bias, so moving toward 50% is the right direction.
- **It needs two decoder passes and extra models.** Decoding takes two LVLM passes per token, plus DETR and RAM++ once per image. This is cheaper than OPERA or Woodpecker but not free.
- **The guidance strength trades accuracy against instruction following.** It has to be tuned, and the best value differs by model.

## My Thoughts

<!-- TODO -->

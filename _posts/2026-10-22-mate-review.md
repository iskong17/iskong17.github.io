---
title: "MATE: Adapt a 3D Classifier by Reconstructing Each Test Point Cloud"
category: paper-review
subfields: [tta, 3dpc]
lang: en
description: MATE trains a Point-MAE classifier jointly with masked reconstruction, then updates the encoder on each corrupted test point cloud by reconstructing it before classifying.
conference: ICCV
year: 2023
paper:
  title: "MATE: Masked Autoencoders are Online 3D Test-Time Learners"
  authors: "M. Jehanzeb Mirza, Inkyu Shin, Wei Lin, Andreas Schriebl, Kunyang Sun, Jaesung Choe, Mateusz Kozinski, Horst Possegger, In So Kweon, Kuk-Jin Yoon, Horst Bischof"
  venue: "ICCV 2023"
  url: "https://arxiv.org/abs/2211.11432"
---

## Abstract

Point cloud classifiers break under corruptions such as sensor noise, LiDAR artifacts, or missing regions. Training for every possible shift is not feasible, so the model should adapt online at test time, without labels. Test-time training (TTT) methods exist for 2D images, but the paper shows that porting them to point clouds works poorly. Entropy methods such as TENT and SHOT collapse when they see one sample at a time, and rotation prediction (TTT-Rot) falls below the unadapted model on average.

MATE is the first TTT method built for 3D point clouds. It uses masked autoencoding as the self-supervised task. For each test point cloud, it masks most of the tokens, asks the network to reconstruct the full shape, takes a gradient step on that reconstruction loss, and then classifies the sample with the updated encoder.

## Motivation

The paper starts from two facts. Small perturbations such as jitter or a little noise already hurt state-of-the-art 3D recognition models, and the shifts real sensors produce are too varied to cover in training. TTT is the natural answer, but at the time every TTT method came from 2D images: entropy minimization (TENT, SHOT), BN statistic updates (DUA), and self-supervised auxiliary tasks such as rotation prediction (TTT-Rot). The authors show these fail when applied naively to point clouds, and in the realistic online case of one sample at a time the entropy methods collapse to about half of Source-Only accuracy.

![MATE teaser: test-time training by masked reconstruction, then online evaluation](/images/blog/papers/mate-teaser.webp)
*Top: a single OOD test sample is masked, reconstructed, and the reconstruction error (drawn as $$l_2$$) updates the encoder. Bottom: the same sample is then classified with the updated encoder. Figure from the paper.*

The key insight is that a self-supervised task that fits 3D data already exists. Masked autoencoding works well for pretraining 3D recognition networks (Point-MAE) and for adapting 2D networks to corruptions, and it needs no labels: remove most of the point cloud and ask the network to rebuild it. If the classifier and the reconstruction head share an encoder, then improving reconstruction on a corrupted test sample should also move the encoder toward features that classify that sample better. A side benefit is cost. The encoder only sees the visible tokens, so with 90-95% masking each update touches a small fraction of the input.

## Method

![MATE overview](/images/blog/papers/mate-main.png)
*The four stages of MATE: tokenization and masking, joint training of classification and reconstruction, test-time training with the reconstruction loss only, and online evaluation with the updated encoder. Figure from the paper.*

### Tokenization and masking

MATE builds on Point-MAE. A point cloud $$\mathcal{X}=\{\mathbf{p}_i\}_{i=1}^{N}$$ is grouped into tokens (local patches around farthest-point-sampled centers). A fraction $$m$$ of the tokens is masked at random. The masked tokens are $$\mathcal{X}^{m}$$ and the visible ones are $$\mathcal{X}^{v}$$. MATE masks 90% of the tokens, so the encoder sees only 10%.

The network has four parts: an encoder $$E$$ (12 transformer blocks) that sees only visible tokens, a light decoder $$D$$ (4 blocks), a linear prediction head $$P$$ that outputs point coordinates, and a classifier head $$C$$.

### Joint training

Point-MAE normally pre-trains the encoder with reconstruction and then fine-tunes a classifier. MATE instead trains both heads together on the labeled source data $$\mathcal{S}$$, so the encoder learns features that serve both tasks:

$$
\min_{\theta_E,\theta_D,\theta_P,\theta_C} \mathbb{E}_{(\mathcal{X},\mathcal{Y})\in\mathcal{S}} \big[ L_c(\mathcal{X},\mathcal{Y};\theta_E,\theta_C) + \lambda \cdot L_s(\mathcal{X};\theta_E,\theta_D,\theta_P) \big]
$$

with $$\lambda=1$$. The classification loss is cross entropy on the visible tokens, and the self-supervised loss is the Chamfer distance between the reconstruction and the input:

$$
L_c = CE\big(C\circ E(\mathcal{X}^{v}), \mathcal{Y}\big), \qquad
L_s = CD\big(P\circ D\circ E(\mathcal{X}^{v}), \mathcal{X}\big)
$$

### Test-time training

At test time MATE receives one corrupted point cloud $$\tilde{\mathcal{X}}$$ with no label. It masks the sample, reconstructs it, and minimizes only the reconstruction loss:

$$
L_{TTT} = \min_{\theta_E,\theta_D,\theta_P} L_s(\tilde{\mathcal{X}};\theta_E,\theta_D,\theta_P)
$$

The classifier head $$C$$ stays frozen. To get a usable loss from one sample, MATE copies it into a batch of 48 and masks each copy differently. The random masking acts as the augmentation, so no other test-time augmentation is used.

### Online evaluation

After the update, the full point cloud (all tokens, no masking) goes through the updated encoder and the frozen classifier. There are two variants:

- **MATE-Standard**: 20 gradient steps per sample, then reset all weights before the next sample.
- **MATE-Online**: one gradient step per sample, and the updates accumulate over the test stream.

Because the encoder only processes the visible tokens, a high mask ratio keeps the test-time cost low. The paper reports that masking 95% of tokens still works about as well as 90%.

## Main Experiments

Backbone: Point-MAE. Metric: top-1 accuracy (%), averaged over the 15 corruptions of each benchmark. All methods adapt with a batch size of 1 (one test sample at a time). The paper reports per-corruption numbers; this table only shows the mean column of Tables 1, 3 and 4.

| Method | ModelNet-40C | ShapeNet-C | ScanObjectNN-C |
|---|---|---|---|
| Source-Only | 53.9 | 57.6 | 45.7 |
| Joint-Training | 57.6 | 59.3 | 45.6 |
| DUA | 54.7 | 60.8 | 46.0 |
| TTT-Rot | 53.0 | 60.9 | 46.1 |
| SHOT | 26.6 | 36.2 | 38.3 |
| T3A | 55.7 | 54.4 | 40.3 |
| TENT | 26.5 | 37.4 | – |
| MATE-Standard | 64.3 | 63.1 | 47.0 |
| MATE-Online | **71.3** | **69.1** | **48.5** |

The 2D baselines get a fairer chance with a batch size of 128 on ModelNet-40C (Table 2, mean accuracy %):

| Source | TENT | SHOT | T3A | MATE-Online |
|---|---|---|---|---|
| 53.9 | 65.6 | 63.8 | 55.9 | **74.5** |

The per-corruption ModelNet-40C table shows that MATE-Online wins almost every column, with the exception of background, where TTT-Rot (34.5) and T3A (33.4) are ahead.

![ModelNet-40C per-corruption accuracy table](/images/blog/papers/mate-results-table.webp)
*Table 1 of the paper: top-1 accuracy (%) for every ModelNet-40C corruption with batch size 1. SHOT and TENT fall to about 26% on average, while MATE-Online reaches 71.3%. The background column (7.2 for Source-Only) is the one MATE does not fix. Figure from the paper.*

Both MATE variants beat every baseline on all three benchmarks. Entropy-based methods (TENT, SHOT) fail badly with single samples and recover only partly with large batches. The gain on the real-world ScanObjectNN-C is much smaller (+2.8 points for MATE-Online over Source-Only) than on the synthetic benchmarks. Adapting only on every 5th sample keeps accuracy within about 3 points of adapting on every sample while running at 20 FPS on ShapeNet-C (RTX 3090).

![Accuracy vs. frames per second for MATE-Online with different adaptation strides](/images/blog/papers/mate-fps.webp)
*Mean ShapeNet-C accuracy against FPS when MATE-Online adapts only every $$k$$-th sample. Stride 1 gives about 69% at about 5 FPS, stride 5 about 65.6% at 20 FPS, and even stride 300 (about 61 FPS) stays above Source-Only (57.6%). Figure from the paper.*

The supplement looks at why background is the weak spot by showing what the network reconstructs after 20 MATE-Standard steps.

![Reconstructions for distortion, background, cutout, and density corruptions](/images/blog/papers/mate-reconstruction.webp)
*Corrupted input (left), the 10% visible tokens (center), and the reconstruction (right) on ModelNet-40C. Distortion, cutout, and density come back as clean airplanes, but with background noise the reconstruction is a diffuse cloud. Figure from the paper (supplement).*

## Limitations

- The paper's own stated limitation: it only studies point cloud classification. Other tasks such as segmentation or detection are left for future work.
- It needs a special training stage. The source model must be trained jointly with the reconstruction branch, so MATE cannot adapt an arbitrary off-the-shelf classifier.
- Each adaptation step needs a forward and backward pass over a batch of 48 masked copies of the sample. MATE-Standard takes 20 such steps per sample, which is far from real time. Speed-ups come from skipping samples (strides), at some cost in accuracy.
- The batch of copies matters. With a batch of 1, MATE-Online drops to 43.1% on ModelNet-40C, below the unadapted model (Table 6).
- It is tied to a masked-autoencoder transformer (Point-MAE). The paper does not test other backbones.
- Reconstruction does not help when the corruption is hard to reconstruct. On background noise, MATE-Standard gets 27.5% on ModelNet-40C (TTT-Rot gets 34.5%) and 4.3% on ShapeNet-C, below Source-Only (10.3%).
- MATE-Online's largest gains come from accumulating updates over a stream that contains one corruption type. The paper does not study streams where the shift keeps changing, although it does test test sets that combine two corruptions per sample (Table 7).

## My Thoughts

<!-- TODO -->

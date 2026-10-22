---
# 논문 리뷰 템플릿
# 사용법: 이 파일을 _posts/YYYY-MM-DD-<영문-슬러그>.md 로 복사한 뒤 내용을 채우세요.
#   예) _posts/2026-11-05-tent-review.md
# 수식: 인라인은 $$ ... $$ (문장 안), 블록은 빈 줄로 감싼 $$ ... $$ 를 쓰면 됩니다.
# subfields: _data/categories.yml 의 paper-review subfields 키 (tta, calibration, 3dpc, hallucination, vlm, avllm)
#   글 상단과 Paper Review 목록의 바에 태그로 표시되고, 목록에서 필터로 쓰입니다.
# paper.url: arXiv 주소. 목록에서 논문 제목을 누르면 이 주소로 이동합니다.
# Main figure 는 images/blog/papers/<슬러그>-main.png, 추가 그림은 <슬러그>-<이름>.webp 에 넣으세요.
# 한국어판은 _paper_ko/<슬러그>.md (en: <슬러그>), 문장은 한국어·용어는 영어, 섹션 제목은 영어 그대로.
title: "Tent: Test-Time Adaptation by Entropy Minimization"
category: paper-review
subfields: [tta]
lang: en
description: 한 줄 요약
paper:
  title: "Tent: Fully Test-Time Adaptation by Entropy Minimization"
  authors: "Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, Trevor Darrell"
  venue: "ICLR 2021"
  url: "https://arxiv.org/abs/2006.10726"
---

## Abstract

## Motivation

![Motivation figure](/images/blog/papers/tent-motivation.webp)
*What the motivating observation shows. Figure from the paper.*

## Method

![Main figure](/images/blog/papers/tent-main.png)
*Overview of Tent. Figure from the paper.*

## Main Experiments

## Limitations

## My Thoughts

<!-- TODO -->

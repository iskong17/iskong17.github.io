---
# 공부 정리 템플릿
# 사용법: 이 파일을 _posts/YYYY-MM-DD-<영문-슬러그>.md 로 복사한 뒤 내용을 채우세요.
#   예) _posts/2026-11-05-entropy-minimization.md
# 수식: 인라인은 $$ ... $$ (문장 안), 블록은 빈 줄로 감싼 $$ ... $$ 를 쓰면 됩니다.
title: "공부 주제"
category: study
description: 한 줄 요약
---

## 배경

## 핵심 아이디어

$$
\mathcal{L}_{\text{ent}} = -\sum_{c} p_c \log p_c
$$

## 코드

```python
import torch

def entropy(logits):
    p = logits.softmax(dim=-1)
    return -(p * p.log()).sum(dim=-1).mean()
```

## 정리

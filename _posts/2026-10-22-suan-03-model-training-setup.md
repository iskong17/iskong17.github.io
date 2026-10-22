---
title: "PyTorch 한번에 끝내기 3: Models, Loss, Optimizers and Schedulers"
date: 2026-10-22 16:03:00 +0900
category: study
topic: pytorch
series: suan
lecture: 3
card_title: "Lecture 3: Models, Loss, Optimizers and Schedulers"
card_desc: "nn.Module · nn.Sequential · Pretrained models · Loss functions · Optimizers · LR schedulers · torchmetrics"
lang: ko
description: "nn.Module 상속과 nn.Sequential로 모델을 정의하고, torchvision의 사전학습 모델을 불러온 뒤, 손실 함수, 옵티마이저, 학습률 스케줄러, torchmetrics 지표를 정리한다."
---

<ul class="post-sources">
<li markdown="span">[파이토치 한번에 끝내기 (이수안컴퓨터연구소)](https://www.youtube.com/watch?v=k60oT_8lyFw) 3편 정리</li>
<li markdown="span">[Colab 노트북](https://colab.research.google.com/drive/1dTWN0jiVIWcPUjfS7J9IT-LE4ACSB2PE)</li>
<li markdown="span">코드: 노트북의 각 실습 설명에 맞춰 작성</li>
</ul>

이 글의 코드는 노트북의 빈 실습 셀 앞에 적힌 설명에 맞춰 작성하고 PyTorch 2.x(CPU)에서 실행한 것이다. 모든 코드는 아래 import 이후에 실행한다고 가정한다.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## 모델 정의

### `nn.Module` 상속 클래스 정의

PyTorch에서 모델은 `nn.Module`을 상속받는 클래스로 정의한다. 두 메서드를 구현한다.

- `__init__()`: 모델에서 사용할 모듈(층)과 활성화 함수 등을 정의한다.
- `forward()`: 입력이 들어왔을 때 모델이 실행할 연산을 정의한다.

```python
class Model(nn.Module):
    def __init__(self, inputs):
        super(Model, self).__init__()
        self.layer = nn.Linear(inputs, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x
```

입력 특성 `inputs`개를 받아 선형 층으로 값 하나를 만들고, 시그모이드로 0과 1 사이의 값으로 바꾸는 모델이다. 로지스틱 회귀와 같은 구조다.

```python
torch.manual_seed(42)
model = Model(1)
print(list(model.children()))
print(list(model.modules()))
print(model(torch.tensor([[0.0], [1.0]])))
```

```text
[Linear(in_features=1, out_features=1, bias=True), Sigmoid()]
[Model(
  (layer): Linear(in_features=1, out_features=1, bias=True)
  (activation): Sigmoid()
), Linear(in_features=1, out_features=1, bias=True), Sigmoid()]
tensor([[0.6964],
        [0.8313]], grad_fn=<SigmoidBackward0>)
```

`children()`은 바로 아래의 하위 모듈만, `modules()`는 모델 자신을 포함한 모든 모듈을 재귀적으로 돌려준다. 모델 객체를 함수처럼 호출하면 내부에서 `forward()`가 실행된다.

### `nn.Sequential`을 이용한 신경망 정의

`nn.Sequential` 객체는 그 안에 넣은 모듈을 순서대로 실행한다. `__init__()`에서 여러 층을 `nn.Sequential`로 묶어 두면, `forward()`에서 실행할 계산을 블록 단위로 가독성 높게 작성할 수 있다.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30 * 5 * 5, out_features=10, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)
        x = self.layer3(x)
        return x
```

32×32 컬러 이미지를 가정한 구조다. 크기는 합성곱(5×5)에서 28, 풀링에서 14, 다시 합성곱에서 10, 풀링에서 5로 줄어들므로, 펼친 뒤의 특성 수는 $$30 \times 5 \times 5 = 750$$이다.

```python
model = Model()
print(list(model.children()))
print(model(torch.randn(2, 3, 32, 32)).shape)
```

```text
[Sequential(
  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Sequential(
  (0): Conv2d(64, 30, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Sequential(
  (0): Linear(in_features=750, out_features=10, bias=True)
  (1): ReLU(inplace=True)
)]
torch.Size([2, 10])
```

`children()`이 세 개의 `Sequential` 블록을 돌려주며, 각 블록 안의 층에는 0부터 번호가 붙는다. 이미지 2장을 넣으면 10개 클래스 점수가 담긴 `(2, 10)`이 나온다.

### 파이토치 사전학습 모델

`torchvision.models`는 ImageNet 등으로 미리 학습된 모델을 제공한다([목록](https://pytorch.org/vision/stable/models.html)). 현재 버전에서는 `weights` 인자에 가중치 열거형을 넘겨 사전학습 가중치를 받는다(예전의 `pretrained=True`는 더 이상 쓰지 않는다).

```python
from torchvision import models
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet.eval()
print(sum(p.numel() for p in resnet.parameters()))
print(resnet.fc)
```

```text
11689512
Linear(in_features=512, out_features=1000, bias=True)
```

처음 실행하면 가중치 파일(약 45MB)을 내려받는다. ResNet-18의 파라미터는 약 1,170만 개이고, 마지막 층 `fc`가 ImageNet의 1,000개 클래스 점수를 낸다. 다른 문제에 쓰려면 이 층을 `nn.Linear(512, 클래스 수)`로 바꿔 미세 조정(fine-tuning)한다.

## 모델 파라미터

### 손실 함수(Loss function)

손실 함수는 예측 값과 실제 값 사이의 오차를 측정한다.

- 학습이 얼마나 잘 진행되고 있는지 나타내는 지표다.
- 모델이 훈련되는 동안 최소화될 값으로, 주어진 문제에 대한 성공 지표 역할을 한다.
- 손실 함수의 결과를 통해 학습 파라미터를 조정한다.
- 최적화 이론에서 최소화하고자 하는 함수이며, 기울기를 구해야 하므로 미분 가능한 함수를 사용한다.

PyTorch의 주요 손실 함수는 다음과 같다.

| 손실 함수 | 용도 |
| --- | --- |
| `torch.nn.BCELoss` | 이진 분류 |
| `torch.nn.CrossEntropyLoss` | 다중 클래스 분류 |
| `torch.nn.MSELoss` | 회귀 |

```python
pred = torch.tensor([2.5, 0.0, 2.1])
target = torch.tensor([3.0, -0.5, 2.0])
print(nn.MSELoss()(pred, target))

prob = torch.tensor([0.9, 0.2, 0.7])
label = torch.tensor([1.0, 0.0, 1.0])
print(nn.BCELoss()(prob, label))

logits = torch.tensor([[2.0, 0.5, -1.0],
                       [0.1, 1.5, 0.3]])
classes = torch.tensor([0, 2])
print(nn.CrossEntropyLoss()(logits, classes))
```

```text
tensor(0.1700)
tensor(0.2284)
tensor(0.9391)
```

`MSELoss`는 오차 제곱의 평균으로, $$(0.25 + 0.25 + 0.01)/3 = 0.17$$이다. `BCELoss`는 시그모이드를 거친 0~1 사이의 확률을 입력으로 받는다. `CrossEntropyLoss`는 소프트맥스를 거치지 않은 점수(logit)와 정수 클래스 번호를 받으며, 내부에서 log-softmax와 음의 로그 가능도(NLL)를 함께 계산한다.

### 옵티마이저(Optimizer)

옵티마이저는 손실 함수를 바탕으로 모델을 어떻게 갱신할지 결정한다. 각 옵티마이저는 특정한 종류의 확률적 경사 하강법을 구현한다.

- 옵티마이저는 `step()`으로 전달받은 파라미터를 갱신한다.
- 모든 옵티마이저의 기반 클래스는 `torch.optim.Optimizer(params, defaults)`다.
- `zero_grad()`로 옵티마이저가 다루는 파라미터들의 기울기를 0으로 초기화한다. `.grad`는 누적되므로 매 반복마다 필요하다.
- `torch.optim.lr_scheduler`로 에포크에 따라 학습률(learning rate)을 조절한다.
- 주요 옵티마이저: `optim.Adadelta`, `optim.Adagrad`, `optim.Adam`, `optim.RMSprop`, `optim.SGD`

```python
import torch.optim as optim
from torch.optim import lr_scheduler

model = Model()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.5)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)
```

옵티마이저에는 갱신할 파라미터(`model.parameters()`)와 학습률 등의 설정을 넘기고, 스케줄러에는 옵티마이저를 넘긴다. 위 코드는 생성 방법을 나열한 것이며, 실제로는 이 중 하나를 골라 쓴다.

노트북에는 이 부분에 여러 옵티마이저의 발전 계보를 정리한 외부 그림이 링크되어 있다. SGD에서 출발해 관성을 더한 Momentum, NAG 계열과, 파라미터마다 학습률을 조정하는 Adagrad, RMSProp, AdaDelta 계열이 Adam으로 합쳐지는 흐름이다.

### 학습률 스케줄러(Learning rate scheduler)

학습률 스케줄러는 학습 중 특정 조건에 따라 학습률을 조정하며 최적화를 진행한다. 예를 들어 일정 횟수 이상 학습하면 학습률을 감소(decay)시키거나, 전역 최소점(global minimum) 근처에 가면 학습률을 줄인다.

| 스케줄러 | 동작 |
| --- | --- |
| `optim.lr_scheduler.LambdaLR` | 람다(lambda) 함수의 결과를 초기 학습률에 곱해 학습률로 설정한다. |
| `optim.lr_scheduler.StepLR` | `step_size` 에포크마다 학습률을 감마(gamma) 비율만큼 감소시킨다. |
| `optim.lr_scheduler.MultiStepLR` | `StepLR`과 비슷하지만 일정 간격이 아니라 지정한 에포크(`milestones`)에서만 감마 비율로 감소시킨다. |
| `optim.lr_scheduler.ExponentialLR` | 에포크마다 이전 학습률에 감마를 곱한다. |
| `optim.lr_scheduler.CosineAnnealingLR` | 학습률을 코사인 함수 형태로 변화시켜, 학습률이 작아졌다가 다시 커지기도 한다. |
| `optim.lr_scheduler.ReduceLROnPlateau` | 검증 손실 등 지표가 더 이상 좋아지지 않으면 학습률을 줄여, 학습 상황에 따라 동적으로 바꾼다. |

`StepLR`이 실제로 학습률을 어떻게 바꾸는지 확인해 본다. 스케줄러의 `step()`은 옵티마이저의 `step()` 뒤에, 보통 에포크마다 한 번 호출한다.

```python
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
for epoch in range(6):
    # 실제 학습에서는 여기서 forward, loss.backward(), optimizer.step()을 수행
    optimizer.step()
    print(epoch, scheduler.get_last_lr())
    scheduler.step()
```

```text
0 [0.001]
1 [0.001]
2 [0.0005]
3 [0.0005]
4 [0.00025]
5 [0.00025]
```

2 에포크마다 학습률이 절반으로 줄었다. `ReduceLROnPlateau`만은 `scheduler.step(val_loss)`처럼 감시할 지표 값을 넘겨 호출한다.

### 지표(Metrics)

지표는 모델의 학습과 테스트 단계를 모니터링하는 데 쓴다. 손실과 달리 미분 가능할 필요가 없으며, 정확도처럼 사람이 해석하기 쉬운 값을 주로 쓴다. 여기서는 [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/) 라이브러리를 사용한다.

```python
# !pip install torchmetrics
import torchmetrics
print(torchmetrics.__version__)
```

```text
1.9.0
```

함수형 API는 예측과 정답을 받아 바로 지표를 계산한다.

```python
torch.manual_seed(42)
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))
print(preds.argmax(dim=-1))
print(target)

acc = torchmetrics.functional.accuracy(preds, target, task="multiclass", num_classes=5)
print(acc)
```

```text
tensor([0, 4, 0, 1, 3, 2, 1, 2, 3, 0])
tensor([3, 0, 1, 1, 2, 4, 4, 3, 3, 4])
tensor(0.2000)
```

확률이 가장 높은 클래스가 정답과 일치하는 것은 4번째(1)와 9번째(3), 2개이므로 정확도는 0.2다. TorchMetrics 1.x에서는 `task`(`"binary"`, `"multiclass"`, `"multilabel"`)와 클래스 수를 지정해야 한다.

모듈형 API는 배치마다 값을 누적해 두었다가 `compute()`로 전체 결과를 계산한다.

```python
metric = torchmetrics.Accuracy(task="multiclass", num_classes=5)

n_batches = 10
for i in range(n_batches):
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))

    acc = metric(preds, target)
    print(acc)

acc = metric.compute()
print(acc)
```

```text
tensor(0.2000)
tensor(0.4000)
tensor(0.3000)
tensor(0.1000)
tensor(0.1000)
tensor(0.1000)
tensor(0.2000)
tensor(0.2000)
tensor(0.1000)
tensor(0.1000)
tensor(0.1800)
```

`metric(preds, target)`은 그 배치의 정확도를 돌려주면서 동시에 내부 상태에 누적한다. 마지막 `compute()`는 100개 샘플 전체에 대한 정확도(18/100)다. 무작위 예측이므로 5개 클래스에서 기대되는 0.2 근처가 나온다. 다음 에포크를 시작하기 전에는 `metric.reset()`으로 상태를 비운다.

## 정리

- 모델은 `nn.Module`을 상속해 `__init__()`에 층을, `forward()`에 연산을 정의한다. `children()`과 `modules()`로 하위 모듈을 확인한다.
- `nn.Sequential`로 여러 층을 블록으로 묶으면 `forward()`가 간결해진다. 합성곱과 풀링 뒤 선형 층의 입력 크기는 직접 계산해 맞춘다.
- `torchvision.models`의 사전학습 모델은 `weights=...ResNet18_Weights.DEFAULT`처럼 불러오고, 마지막 층을 바꿔 다른 문제에 쓴다.
- 손실 함수는 회귀에 `MSELoss`, 이진 분류에 `BCELoss`, 다중 분류에 `CrossEntropyLoss`(logit 입력)를 쓴다.
- 옵티마이저는 `zero_grad()`, `backward()`, `step()` 순서로 쓰고, 스케줄러는 에포크마다 `step()`을 호출해 학습률을 조절한다.
- TorchMetrics는 함수형(`functional.accuracy`)과 누적형(`Accuracy` 모듈, `compute()`, `reset()`) 두 방식으로 지표를 계산한다.

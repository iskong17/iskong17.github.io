---
title: "Deep Learning & PyTorch 2: PyTorch Basics"
date: 2026-10-22 17:02:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 2
card_title: "Chapter 2: PyTorch Basics"
card_desc: "Tensors and storage · Dataset and DataLoader · nn.Module and nn.Sequential · Loss, optimizer, LR scheduler · Embedding-based classifier · Precision, recall, F1"
lang: ko
notes: "https://thebook.io/080289/0022/"
notes_label: "thebook.io chapter 2"
description: "파이토치의 특징과 계층 구조, 텐서 조작부터 데이터 준비·모델 정의·훈련·평가·모니터링까지의 기초 문법을 정리하고, 자동차 평가 데이터로 임베딩 기반 분류 모델을 처음부터 끝까지 만들어 본다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 2장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap02)</li>
</ul>

2장은 파이토치가 어떤 도구인지 살펴본 뒤, 텐서 다루기부터 모델 훈련과 평가까지 이후 모든 장에서 반복될 기본 문법을 훑는다. 마지막 절에서는 범주형 데이터로 작은 분류 신경망을 끝까지 만들어 본다. 책의 실습 환경은 Python 3.9, PyTorch 1.9이며, 최신 버전과 달라진 부분은 따로 표시했다.

## 2.1 파이토치 개요

[📖 thebook.io](https://thebook.io/080289/0023/)

**파이토치(PyTorch)** 는 루아(Lua) 언어로 만들어졌던 과학 계산 라이브러리 토치(Torch)를 페이스북(현 Meta)이 파이썬용으로 다시 만들어 2017년 초에 공개한 딥러닝 프레임워크다. 처음에는 넘파이(NumPy)처럼 수치 계산용 라이브러리였지만 딥러닝 기능이 붙으면서 지금의 모습이 되었다. 공식 소개를 요약하면 파이토치는 두 부류의 사용자를 겨냥한다.

- 넘파이와 비슷하게 쓰되 GPU의 계산 능력이 필요한 사람
- 유연하고 빠르게 실험할 수 있는 딥러닝 연구 도구가 필요한 사람

무엇보다 코드가 간결하고 구현이 빠르다는 점이 인기의 이유다.

### 2.1.1 파이토치 특징 및 장점

파이토치를 한 문장으로 요약하면 "GPU 위에서 텐서를 다루고 신경망을 동적으로 구성하는 프레임워크"다. 세 단어를 풀어 보면 다음과 같다.

- **GPU(Graphics Processing Unit)**: 학습 중 기울기를 구하려면 미분 계산을 엄청나게 많이 해야 하는데, 이런 대규모 병렬 계산은 CPU보다 GPU가 훨씬 빠르다. 파이토치는 내부적으로 CUDA와 cuDNN을 통해 GPU를 쓴다. 현실적으로 딥러닝 학습에 GPU는 거의 필수다.
- **텐서(tensor)**: 파이토치가 데이터를 담는 기본 자료 구조로, 같은 자료형의 값들이 모인 다차원 배열이다. `.cuda()`나 `.to("cuda")` 한 줄로 GPU 연산 대상으로 옮길 수 있다.
- **동적 신경망**: 반복마다 네트워크 구조를 바꿀 수 있는 신경망이다. 파이토치는 코드를 실행하는 순간 연산 그래프가 만들어지는 **Define by Run** 방식을 쓰기 때문에, 그래프를 먼저 정의하고 나중에 값을 흘려보내는 방식보다 코드를 읽고 디버깅하기 쉽다.

#### 벡터, 행렬, 텐서

머신 러닝에서 데이터는 숫자 배열로 표현된다. 차원 수에 따라 부르는 이름이 다르다.

| 이름 | 차원 | 예 | 쓰임 |
| --- | --- | --- | --- |
| 벡터(vector) | 1차원 | `[1.0, 1.1, 1.2]` | 데이터 한 건(레코드)의 특징들 |
| 행렬(matrix) | 2차원 (행, 열) | 3×2 표 | 레코드 여러 건. 보통 한 행이 한 레코드 |
| 텐서(tensor) | 3차원 이상 | 같은 크기 행렬 여러 장 | 이미지(채널, 높이, 너비), 배치 등 |

레코드 하나를 벡터로 쓸 때는 보통 열벡터로 적지만, 여러 레코드를 모아 행렬로 만들 때는 레코드 하나가 한 행이 된다. $$m$$개의 레코드가 각각 $$n$$개의 특징을 가지면 $$m \times n$$ 행렬이다.

$$
X = \begin{bmatrix} x_{11} & x_{12} & \cdots & x_{1n} \\ x_{21} & x_{22} & \cdots & x_{2n} \\ \vdots & & \ddots & \vdots \\ x_{m1} & x_{m2} & \cdots & x_{mn} \end{bmatrix}
$$

텐서는 이런 행렬을 여러 장 겹친 것으로 생각하면 된다. 파이토치에서는 `torch.tensor()`로 만든다.

```python
import torch
torch.tensor([[1., -1.], [1., -1.]])
```

```text
tensor([[ 1., -1.],
        [ 1., -1.]])
```

#### 연산 그래프

**연산 그래프(computational graph)** 는 변수(텐서)와 연산(덧셈, 곱셈 등)을 방향이 있는 그래프로 표현한 것이다. 예를 들어 입력 `a`, `b`에서 `c = a * b`, `d = a + b`, `e = c + d`를 차례로 계산했다면, `a`와 `b`에서 시작해 `e`로 모이는 그래프가 생긴다. 신경망 학습에서 손실의 기울기를 가중치와 바이어스에 대해 구할 때 이 그래프를 거꾸로 따라가며 미분을 계산하고, 그 기울기로 경사 하강법을 수행한다.

```python
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b        # 곱셈 노드
d = a + b        # 덧셈 노드
e = c + d        # 최종 출력
e.backward()     # 그래프를 거꾸로 따라 미분
print(a.grad, b.grad)   # de/da = b + 1 = 4, de/db = a + 1 = 3
```

```text
tensor(4.) tensor(3.)
```

#### 파이토치의 장점

- **단순함**: 파이썬 코드와 자연스럽게 섞이고, 일반 파이썬 디버거로 중간값을 바로 확인할 수 있다.
- **성능**: 학습 시 CPU 사용률이 비교적 낮고 학습과 추론 속도가 빠르다.
- **직관적인 인터페이스**: 텐서플로가 고수준 API를 여러 번 갈아엎은 것(layers → slim → estimators → tf.keras)과 달리 API가 안정적이라 배우기 쉽다.

### 2.1.2 파이토치의 아키텍처

[📖 thebook.io](https://thebook.io/080289/0027/)

파이토치는 세 층으로 볼 수 있다.

```text
┌────────────────────────────────────────────┐
│ 파이토치 API (Python)                       │  torch, torch.autograd, torch.nn,
│  사용자가 호출하는 인터페이스, 계산은 안 함   │  torch.multiprocessing, torch.utils
├────────────────────────────────────────────┤
│ 파이토치 엔진 (C++)                         │  Autograd C++, ATen C++, JIT C++
│  다차원 텐서와 자동 미분 처리                │  → Python API로 래핑
├────────────────────────────────────────────┤
│ 연산 처리 (C / CUDA)                        │  TH(CPU), THC(GPU)
│  실제 텐서 계산 수행                         │
└────────────────────────────────────────────┘
```

**파이토치 API 계층**은 사용자가 텐서를 다루고 신경망을 만들고 훈련하기 위한 함수와 클래스를 제공한다. 이 계층은 직접 계산하지 않고 C++ 엔진에 일을 넘긴다. 주요 패키지는 다음과 같다.

| 패키지 | 역할 |
| --- | --- |
| `torch` | GPU를 지원하는 다차원 텐서와 수학 연산 |
| `torch.autograd` | 자동 미분. 연산 그래프가 실행과 동시에 만들어지므로 은닉층 크기 같은 구조를 바꿔도 처음부터 다시 만들 필요가 없다. 텐서플로(초기 버전), 카페(Caffe), CNTK와 가장 차별되는 부분이다. |
| `torch.nn` | 합성곱층, 순환층, 정규화 등 신경망 구성 요소 |
| `torch.multiprocessing` | 여러 프로세스가 같은 텐서 메모리를 공유하도록 지원 |
| `torch.utils` | `DataLoader`(데이터 공급), `bottleneck`(병목 분석), `checkpoint` 등 유틸리티 |

**파이토치 엔진**은 C++로 작성되어 있다. Autograd C++는 가중치와 바이어스 갱신에 필요한 미분을 자동으로 계산하고, ATen C++는 텐서 라이브러리를, JIT C++는 계산을 최적화하는 **JIT(Just-In-Time) 컴파일러**를 제공한다. 이들을 감싼 Python API 덕분에 사용자는 파이썬만으로 모델을 만들 수 있다.

**연산 처리 계층**의 C/CUDA 패키지가 위에서 내려온 거의 모든 계산을 실제로 수행한다. CPU용(TH)과 GPU용(THC) 구현이 있다.

#### 텐서는 메모리에 어떻게 저장되는가

텐서는 몇 차원이든 메모리에는 1차원으로 늘어놓아 저장한다. 이 1차원 배열을 **스토리지(storage)** 라 하고, 텐서는 스토리지 위에 "어떻게 읽을지"에 대한 정보만 얹은 뷰(view)다. 그 정보가 두 가지다.

- **오프셋(offset)**: 텐서의 첫 원소가 스토리지의 몇 번째에 있는지.
- **스트라이드(stride)**: 각 차원에서 인덱스를 1 늘릴 때 스토리지에서 몇 칸을 건너뛰어야 하는지.

예를 들어 2×3 행렬 $$A$$와 그 전치 $$A^T$$(3×2)를 보자.

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \qquad A^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

$$A$$는 스토리지 `[1, 2, 3, 4, 5, 6]`에 저장되고, 다음 행으로 가려면 3칸, 다음 열로 가려면 1칸 이동하므로 스트라이드는 `(3, 1)`이다. 파이토치에서 `A.t()`는 데이터를 복사하지 않고 **같은 스토리지**를 공유하면서 스트라이드만 `(1, 3)`으로 바꾼다. 반대로 전치 결과를 새로 연속 저장(`contiguous()`)하면 스토리지가 `[1, 4, 2, 5, 3, 6]`이 되어 스트라이드가 `(2, 1)`이 된다. 즉 모양(shape)이 다른 두 텐서가 같은 데이터를 가리킬 수 있고, 이를 구분하는 것이 오프셋과 스트라이드다.

```python
A = torch.tensor([[1, 2, 3], [4, 5, 6]])
At = A.t()
print(A.stride(), At.stride())                    # 같은 스토리지, 읽는 방법만 다름
print(A.data_ptr() == At.data_ptr())              # 메모리 공유 여부
print(At.is_contiguous(), At.contiguous().stride())
```

```text
(3, 1) (1, 3)
True
False (2, 1)
```

이 개념은 뒤에서 `view()`가 비연속 텐서에서 실패하는 이유(그럴 때는 `contiguous()`나 `reshape()`를 쓴다)를 이해하는 데 쓸모가 있다.

## 2.2 파이토치 기초 문법

[📖 thebook.io](https://thebook.io/080289/0032/)

파이토치 코드는 텐서로 시작해 텐서로 끝난다. 입력과 출력, 가중치가 모두 텐서이므로 텐서 조작에 익숙해야 신경망의 데이터 흐름을 제어할 수 있다. 이 절의 코드는 대부분 문법을 보여 주기 위한 예시다.

### 2.2.1 텐서 다루기

#### 텐서 생성 및 변환

텐서는 넘파이의 `ndarray`와 비슷하지만 GPU에서도 계산할 수 있다. `device`로 위치를, `dtype`으로 자료형을 정한다.

```python
import torch
print(torch.tensor([[1, 2], [3, 4]]))                         # 2차원 정수 텐서
# print(torch.tensor([[1, 2], [3, 4]], device="cuda:0"))      # GPU에 생성 (GPU가 없으면 오류)
print(torch.tensor([[1, 2], [3, 4]], dtype=torch.float64))    # 64비트 실수로 생성
```

```text
tensor([[1, 2],
        [3, 4]])
tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64)
```

넘파이 배열로 바꿀 때는 `.numpy()`를 쓴다. 단, GPU에 있는 텐서는 먼저 CPU로 옮겨야 한다.

```python
temp = torch.tensor([[1, 2], [3, 4]])
print(temp.numpy())

temp = torch.tensor([[1, 2], [3, 4]], device="cuda:0")
print(temp.to("cpu").numpy())   # GPU → CPU → ndarray
```

```text
[[1 2]
 [3 4]]
[[1 2]
 [3 4]]
```

#### 인덱스 조작

텐서는 파이썬 리스트나 넘파이처럼 인덱스와 슬라이스로 접근한다. 자주 보는 자료형은 다음과 같다.

| 텐서 타입 | 자료형 |
| --- | --- |
| `torch.FloatTensor` | 32비트 부동 소수점 (`torch.float32`) |
| `torch.DoubleTensor` | 64비트 부동 소수점 (`torch.float64`) |
| `torch.LongTensor` | 64비트 부호 있는 정수 (`torch.int64`) |

```python
temp = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
print(temp[0], temp[1], temp[-1])     # -1은 맨 뒤
print('------------------------')
print(temp[2:5], temp[4:-1])          # 끝 인덱스는 포함하지 않음
```

```text
tensor(1.) tensor(2.) tensor(7.)
------------------------
tensor([3., 4., 5.]) tensor([5., 6.])
```

#### 연산과 차원 조작

텐서끼리 사칙 연산이 가능하다. 다만 자료형이 다르면(예: `FloatTensor`와 `DoubleTensor`) 연산이 실패할 수 있으므로 맞춰 주는 습관이 필요하다.

```python
v = torch.tensor([1, 2, 3])
w = torch.tensor([3, 4, 6])
print(w - v)      # 원소별 뺄셈
```

```text
tensor([2, 2, 3])
```

신경망에서는 층 사이에서 모양을 맞추는 일이 잦기 때문에 차원 조작이 특히 중요하다. 주요 함수는 다음과 같다.

| 함수 | 역할 |
| --- | --- |
| `view` | 원소 수를 유지한 채 모양 변경 (넘파이 `reshape`와 비슷) |
| `stack` | 새 축을 만들며 텐서들을 쌓음 |
| `cat` | 기존 축을 따라 이어 붙임 |
| `t`, `transpose` | 전치, 또는 두 차원의 순서를 맞바꿈 |

```python
temp = torch.tensor([[1, 2], [3, 4]])   # 2×2
print(temp.shape)
print(temp.view(4, 1))     # 4×1
print(temp.view(-1))       # 1차원으로 펼침
print(temp.view(1, -1))    # -1은 "나머지에서 계산": 원소 4개 → (1, 4)
print(temp.view(-1, 1))    # (?, 1) → (4, 1)
```

```text
torch.Size([2, 2])
tensor([[1],
        [2],
        [3],
        [4]])
tensor([1, 2, 3, 4])
tensor([[1, 2, 3, 4]])
tensor([[1],
        [2],
        [3],
        [4]])
```

`-1`은 한 번만 쓸 수 있고, 나머지 차원과 전체 원소 수로부터 자동으로 계산된다.

### 2.2.2 데이터 준비

[📖 thebook.io](https://thebook.io/080289/0037/)

데이터를 불러오는 방법은 크게 판다스(pandas) 같은 파이썬 라이브러리로 읽는 방법과 파이토치의 데이터 도구를 쓰는 방법이 있다. 이미지라면 흩어진 파일을 읽어 전처리한 뒤 배치로 나누고, 텍스트라면 임베딩을 거쳐 길이가 다른 시퀀스를 배치로 묶는다.

#### 파일을 그대로 불러오기

데이터가 단순한 표 형태라면 판다스로 CSV를 읽은 뒤 텐서로 바꾸면 충분하다.

```python
import pandas as pd
import torch

data = pd.read_csv('../class2.csv')
# 칼럼 → 넘파이 배열 → 텐서, unsqueeze로 (N,) 를 (N, 1)로, float로 실수형 변환
x = torch.from_numpy(data['x'].values).unsqueeze(dim=1).float()
y = torch.from_numpy(data['y'].values).unsqueeze(dim=1).float()
```

#### 커스텀 데이터셋

데이터가 크면 전부 메모리에 올려 두는 것이 비효율적이다. 필요한 만큼만 꺼내 쓰도록 `torch.utils.data.Dataset`을 상속한 **커스텀 데이터셋(custom dataset)** 을 만든다. 세 메서드를 구현하면 된다.

| 메서드 | 역할 |
| --- | --- |
| `__init__(self, ...)` | 파일 경로 등 필요한 변수를 받고 전처리를 준비 |
| `__len__(self)` | 전체 샘플 수 반환 |
| `__getitem__(self, idx)` | `idx`번째 샘플을 텐서로 반환 |

```python
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.label = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = torch.tensor(self.label.iloc[idx, 0:3]).int()   # 앞의 3개 칼럼이 입력
        label = torch.tensor(self.label.iloc[idx, 3]).int()      # 4번째 칼럼이 정답
        return sample, label

tensor_dataset = CustomDataset('../covtype.csv')
dataset = DataLoader(tensor_dataset, batch_size=4, shuffle=True)
```

#### DataLoader

`DataLoader`는 데이터셋 전체를 들고 있다가 학습할 때 배치 크기만큼씩 꺼내 준다. 미리 잘라 두는 것이 아니라, 내부 반복자(iterator)가 인덱스를 골라 그때그때 `__getitem__`을 호출해 묶는다. 따라서 `for` 문으로 돌면 배치가 하나씩 나온다.

```python
for i, data in enumerate(dataset, 0):
    print(i, end='')
    batch = data[0]          # 입력 부분
    print(batch.size())
```

```text
0torch.Size([4, 3])
1torch.Size([4, 3])
2torch.Size([4, 3])
3torch.Size([4, 3])
4torch.Size([3, 3])
```

샘플이 19개라 마지막 배치에는 3개만 남았다. 이렇게 남는 배치를 버리려면 `drop_last=True`를 준다.

#### 파이토치 제공 데이터셋

**토치비전(torchvision)** 은 MNIST, ImageNet 등 유명한 이미지 데이터셋과 변환 도구를 제공한다. 다운로드에 `requests` 라이브러리가 필요할 수 있다.

```python
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

mnist_transform = transforms.Compose([
    transforms.ToTensor(),                  # PIL 이미지 → [0, 1] 범위 텐서
    transforms.Normalize((0.5,), (1.0,))    # (x - 0.5) / 1.0
])

download_root = '../chap02/data/MNIST_DATASET'
train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
valid_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
```

`train=True`이면 훈련용 6만 장, `False`이면 테스트용 1만 장을 받는다. 위 코드는 검증과 테스트에 같은 데이터를 쓰고 있으므로, 실제로는 훈련 데이터에서 검증용을 따로 떼어 내는 편이 맞다.

### 2.2.3 모델 정의

[📖 thebook.io](https://thebook.io/080289/0042/)

먼저 용어를 정리한다.

- **계층(layer)**: 합성곱층, 선형 계층처럼 하나의 연산 단위.
- **모듈(module)**: 계층이 하나 이상 모인 것. 모듈이 모여 더 큰 모듈이 될 수도 있다.
- **모델(model)**: 최종적으로 원하는 네트워크. 모듈 하나가 곧 모델일 수도 있다.

파이토치에서 모델을 정의하는 방법은 네 가지다.

**① 단순 신경망.** 계층 하나로 충분하면 클래스를 만들 필요가 없다.

```python
model = nn.Linear(in_features=1, out_features=1, bias=True)
```

**② `nn.Module` 상속.** 가장 일반적인 방법이다. `__init__()`에서 사용할 계층과 활성화 함수를 만들고, `forward()`에서 입력이 어떤 순서로 계산되는지 적는다.

```python
from torch.nn import Module, Linear, Sigmoid

class MLP(Module):
    def __init__(self, inputs):
        super(MLP, self).__init__()
        self.layer = Linear(inputs, 1)    # 계층
        self.activation = Sigmoid()       # 활성화 함수

    def forward(self, X):
        X = self.layer(X)
        X = self.activation(X)
        return X
```

**③ `nn.Sequential` 사용.** 순서대로 실행할 계층들을 하나로 묶는다. 묶음 단위로 `forward()`를 짧게 쓸 수 있어 계층이 많을수록 읽기 쉬워진다.

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=30, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=30*5*5, out_features=10, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.shape[0], -1)    # (배치, 30, 5, 5) → (배치, 750)
        x = self.layer3(x)
        return x

model = MLP()
print(list(model.children()))
print(list(model.modules()))
```

`children()`과 `modules()`의 차이는 탐색 깊이다.

- `model.children()`: 바로 아래 한 단계의 자식만 돌려준다. 여기서는 `Sequential` 세 개.
- `model.modules()`: 모델 자신을 포함해 트리 전체를 깊이 우선으로 모두 돌려준다. `MLP` → `Sequential` → 그 안의 `Conv2d`, `ReLU`, `MaxPool2d` → 다음 `Sequential` … 순서다.

```text
[Sequential(
  (0): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
), Sequential(
  (0): Conv2d(64, 30, kernel_size=(5, 5), stride=(1, 1))
  ...
), Sequential(
  (0): Linear(in_features=750, out_features=10, bias=True)
  (1): ReLU(inplace=True)
)]
```

**④ 함수로 정의.** `Sequential`을 돌려주는 함수를 만든다. 변수에 담아 둔 계층을 재사용하기 편하지만, 모델이 복잡해지면 오히려 관리가 어려워지므로 그때는 `nn.Module` 상속이 낫다.

```python
def MLP(in_features=1, hidden_features=20, out_features=1):
    hidden = nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
    activation = nn.ReLU()
    output = nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
    net = nn.Sequential(hidden, activation, output)
    return net
```

ReLU, Softmax, Sigmoid 같은 활성화 함수는 이처럼 모델을 정의할 때 함께 넣는다.

### 2.2.4 모델의 파라미터 정의

[📖 thebook.io](https://thebook.io/080289/0046/)

학습을 시작하기 전에 정해 둘 것들이 있다.

**손실 함수(loss function).** 모델 출력($$wx + b$$)과 정답 $$y$$의 차이를 재어 모델이 얼마나 틀렸는지 수치로 나타낸다.

| 손실 함수 | 용도 |
| --- | --- |
| `nn.BCELoss` | 이진 분류 |
| `nn.CrossEntropyLoss` | 다중 클래스 분류 |
| `nn.MSELoss` | 회귀 |

**옵티마이저(optimizer).** 손실을 바탕으로 파라미터를 어떻게 고칠지 정한다.

- `step()`이 전달받은 파라미터를 실제로 갱신한다.
- `zero_grad()`는 파라미터들의 기울기(gradient)를 0으로 만든다.
- 파라미터 그룹별로 학습률 등을 다르게 줄 수 있다.
- 모든 옵티마이저는 `torch.optim.Optimizer(params, defaults)`를 상속한다.
- 종류: `Adadelta`, `Adagrad`, `Adam`, `SparseAdam`, `Adamax`, `ASGD`, `LBFGS`, `RMSprop`, `Rprop`, `SGD` (4장에서 다시 다룬다).

**학습률 스케줄러(learning rate scheduler).** 에포크가 지남에 따라 학습률을 조절한다. 초반에는 크게 움직여 빨리 내려가고, 최저점 근처에서는 작게 움직여 정밀하게 다가가기 위해서다.

| 스케줄러 | 동작 |
| --- | --- |
| `LambdaLR` | 사용자가 준 함수(람다)의 값을 학습률 배수로 사용 |
| `StepLR` | 정해진 스텝 간격마다 학습률에 `gamma`를 곱함 |
| `MultiStepLR` | 지정한 에포크들에서만 `gamma`를 곱함 |
| `ExponentialLR` | 매 에포크 `gamma`를 곱함 |
| `CosineAnnealingLR` | 코사인 곡선을 따라 학습률을 내렸다 올림 |
| `ReduceLROnPlateau` | 검증 성능이 나아지지 않으면 학습률을 낮춤 |

**지표(metrics).** 훈련과 테스트 과정을 모니터링할 기준(정확도 등)이다.

#### 전역 최소점과 지역 최소점

손실 함수 값은 오차가 클수록 크다. 학습의 목표는 손실을 가장 작게 만드는 가중치와 바이어스를 찾는 것이다. 손실이 가장 낮은 지점이 **전역 최소점(global minimum)** 이며 최종적으로 찾고 싶은 최적점이다. 그런데 손실 곡면에는 주변보다만 낮은 움푹한 곳, **지역 최소점(local minimum)** 도 있다. 옵티마이저가 여기에 빠져 멈추면 더 좋은 해를 놓친다. 학습률 스케줄링이나 모멘텀 같은 기법은 이런 문제를 줄이려는 장치이기도 하다.

다음은 손실 함수, 옵티마이저, 스케줄러를 함께 쓰는 예시다.

```python
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# 에포크마다 학습률을 0.95배씩 (lr = 0.01 * 0.95**epoch)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                              lr_lambda=lambda epoch: 0.95 ** epoch)

for epoch in range(1, 100 + 1):        # 에포크 반복
    for x, y in dataloader:            # 배치 반복
        optimizer.zero_grad()
        criterion(model(x), y).backward()
        optimizer.step()
    scheduler.step()                   # 에포크가 끝날 때마다 학습률 갱신
```

### 2.2.5 모델 훈련

[📖 thebook.io](https://thebook.io/080289/0050/)

학습이란 $$y = wx + b$$에서 적절한 $$w$$와 $$b$$를 찾는 일이다. 임의의 값에서 출발해 오차가 줄어드는 방향으로 파라미터를 계속 고친다.

여기서 꼭 기억할 점은 **기울기가 누적된다**는 것이다. `loss.backward()`는 새로 계산한 기울기를 기존 값에 더한다. 여러 단계에 걸쳐 기울기를 모아야 하는 경우(예: RNN의 일부 구현, 큰 배치를 나눠 계산하는 경우)에는 유용하지만, 보통은 배치마다 새로 계산해야 하므로 역전파 전에 `optimizer.zero_grad()`로 기울기를 0으로 지운다.

한 번의 학습 단계는 다음 순서다.

1. 순전파: `model(x)`로 예측을 만든다.
2. 손실 계산: `criterion(yhat, y)`.
3. 기울기 초기화: `optimizer.zero_grad()`.
4. 역전파: `loss.backward()`로 기울기를 계산한다.
5. 갱신: `optimizer.step()`으로 파라미터를 고친다.

```python
for epoch in range(100):
    yhat = model(x_train)
    loss = criterion(yhat, y_train)
    optimizer.zero_grad()    # 이전 배치의 기울기 제거
    loss.backward()
    optimizer.step()
```

### 2.2.6 모델 평가

[📖 thebook.io](https://thebook.io/080289/0052/)

테스트 데이터로 모델을 평가할 때는 `torchmetrics` 패키지(`pip install torchmetrics`)를 쓸 수 있다. 함수형과 모듈형 두 방식이 있다.

```python
import torch
import torchmetrics

preds = torch.randn(10, 5).softmax(dim=-1)     # 샘플 10개, 클래스 5개의 확률
target = torch.randint(5, (10,))               # 정답 클래스
acc = torchmetrics.functional.accuracy(preds, target)
```

모듈형은 배치마다 값을 누적해 두었다가 마지막에 전체 결과를 계산할 수 있다.

```python
metric = torchmetrics.Accuracy()
n_batches = 10
for i in range(n_batches):
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    acc = metric(preds, target)                  # 이번 배치의 정확도
    print(f"Accuracy on batch {i}: {acc}")

acc = metric.compute()                           # 지금까지 모든 배치의 정확도
print(f"Accuracy on all data: {acc}")
```

최신 `torchmetrics`(0.11 이후)에서는 `torchmetrics.Accuracy(task="multiclass", num_classes=5)`처럼 과제 종류를 명시해야 한다. 사이킷런의 `confusion_matrix`, `accuracy_score`, `classification_report`도 많이 쓰인다(2.4절에서 사용).

### 2.2.7 훈련 과정 모니터링

[📖 thebook.io](https://thebook.io/080289/0053/)

학습 중 파라미터와 손실이 어떻게 변하는지 숫자만으로는 파악하기 어렵다. **텐서보드(TensorBoard)** 를 쓰면 이를 그래프로 볼 수 있다. 절차는 설정 → 기록 → 확인의 세 단계다(`pip install tensorboard`).

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../chap02/tensorboard")    # 로그 저장 위치

for epoch in range(num_epochs):
    model.train()                                  # 학습 모드 (드롭아웃 켜짐)
    for i, (x, y) in enumerate(dataloader):
        x, y = x.to(device).float(), y.to(device).float()
        outputs = model(x)
        loss = criterion(outputs, y)
        writer.add_scalar("Loss", loss, epoch)     # 스칼라 값 기록
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

writer.close()                                     # 기록이 끝나면 닫기
```

터미널에서 `tensorboard --logdir=../chap02/tensorboard --port=6006`을 실행하고 브라우저로 `http://localhost:6006`에 접속하면 손실 곡선을 볼 수 있다.

#### model.train()과 model.eval()

- `model.train()`: 학습 모드. 드롭아웃이 켜지고 배치 정규화는 현재 배치의 통계를 쓴다.
- `model.eval()`: 평가 모드. 드롭아웃을 끄고 모든 노드를 사용하며, 배치 정규화는 학습 중 모아 둔 통계를 쓴다. 검증과 테스트에서 쓴다.

두 모드를 제대로 바꿔 주지 않으면 평가 결과가 들쭉날쭉해진다. 평가에서는 보통 `torch.no_grad()`와 함께 쓴다.

```python
model.eval()                    # 평가 모드 (드롭아웃 끔)
with torch.no_grad():           # 기울기 기록 끔
    valid_loss = 0
    for x, y in valid_dataloader:
        outputs = model(x)
        loss = F.cross_entropy(outputs, y.long().squeeze())
        valid_loss += float(loss)
        y_hat += [outputs]
valid_loss = valid_loss / len(valid_loader)
```

`torch.no_grad()`를 쓰는 이유는 메모리와 시간이다. 파이토치는 기본적으로 역전파를 위해 연산 기록과 중간값을 저장하는데, 평가에서는 역전파를 하지 않으니 이를 꺼서 자원을 아낀다.

## 2.3 실습 환경 설정

[📖 thebook.io](https://thebook.io/080289/0056/)

책은 아나콘다(Anaconda)를 설치하고, 가상 환경을 만들어 그 안에 파이토치를 설치하는 방식을 쓴다.

### 2.3.1 아나콘다 설치

아나콘다 웹 사이트에서 운영체제에 맞는 설치 파일을 받아 기본 옵션으로 설치한다. 설치 옵션 화면에서 **환경 변수(PATH)에 아나콘다를 등록하는 옵션**을 함께 체크하는 것이 요점이다. 등록하지 않으면 명령 프롬프트에서 `python`을 실행할 때 "'python' is not recognized ..." 같은 오류가 날 수 있다. 설치 후 시스템 속성의 환경 변수 목록에서 아나콘다 경로가 추가되었는지 확인할 수 있다.

### 2.3.2 가상 환경 생성 및 파이토치 설치

[📖 thebook.io](https://thebook.io/080289/0063/)

파이썬과 라이브러리 버전을 프로젝트별로 분리하기 위해 가상 환경을 쓴다. Anaconda Prompt에서 다음을 실행한다.

```bash
conda create -n torch_book python=3.9.0    # 가상 환경 생성 (책 시점에 3.10은 호환 문제)
conda env list                             # 환경 목록 확인
activate torch_book                        # 활성화 (최신 conda는 conda activate)
# conda env remove -n torch_book           # 잘못 만들었으면 삭제

pip install ipykernel                      # 주피터에서 이 환경을 커널로 쓰기 위해
python -m ipykernel install --user --name torch_book --display-name "torch_book"
```

파이토치는 책 기준 1.9.0을 설치한다(CPU 환경 기준).

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 -c pytorch
# 또는
pip install torch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0

pip install jupyter notebook
jupyter notebook
```

GPU용이나 최신 버전 설치 명령은 pytorch.org의 설치 안내에서 환경(OS, 패키지 관리자, CUDA 버전)을 고르면 생성된다.

개인 PC 사양이 부족하면 **구글 코랩(Colab)** 을 쓴다. 클라우드에서 주피터 노트북을 무료로 실행할 수 있고 GPU도 제공한다(사용법은 책 부록).

## 2.4 파이토치 코드 맛보기

[📖 thebook.io](https://thebook.io/080289/0066/)

자동차 평가 데이터(`car_evaluation.csv`, 1,728건)로 간단한 분류 모델을 만든다. 입력 특성 6개로 차 상태를 예측한다.

| 칼럼 | 의미 |
| --- | --- |
| `price` | 가격 |
| `maint` | 유지 비용 |
| `doors` | 문 개수 |
| `persons` | 수용 인원 |
| `lug_capacity` | 수하물 용량 |
| `safety` | 안전성 |
| `output` (정답) | 차 상태: `unacc`(허용 불가), `acc`(허용 가능), `good`(양호), `vgood`(매우 좋음) |

사용하는 보조 라이브러리는 matplotlib(그래프), seaborn(통계 시각화), scikit-learn(머신 러닝 알고리즘과 평가 도구)이다.

### 데이터 불러오기와 분포 확인

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('../chap02/data/car_evaluation.csv')
dataset.head()       # 앞 5행 확인 (tail()은 뒤 5행)
```

모든 칼럼이 `vhigh`, `small`, `low` 같은 **문자열**이다. 신경망은 숫자만 계산할 수 있으므로 이를 숫자로 바꾸는 과정이 필요하다.

정답 칼럼의 분포를 파이 차트로 그려 보면 클래스 비율이 매우 치우쳐 있다.

```python
plt.rcParams["figure.figsize"] = [8, 6]
dataset.output.value_counts().plot(kind='pie', autopct='%0.05f%%',
                                   colors=['lightblue', 'lightgreen', 'orange', 'pink'],
                                   explode=(0.05, 0.05, 0.05, 0.05))
```

`unacc`가 약 70%, `acc`가 약 22%이고 `good`과 `vgood`은 각각 4% 안팎에 불과하다. 이런 **클래스 불균형**은 평가 결과를 해석할 때 중요한 단서가 된다.

### 범주형 데이터를 숫자로

모든 입력 칼럼은 **범주형(categorical)** 데이터다(성별처럼 정해진 값 중 하나를 갖는 데이터). 변환 경로는 다음과 같다.

```text
문자열 칼럼 → category 타입 → 정수 코드(넘파이 배열) → 텐서
```

```python
categorical_columns = ['price', 'maint', 'doors', 'persons', 'lug_capacity', 'safety']
for category in categorical_columns:
    dataset[category] = dataset[category].astype('category')   # 범주형 타입으로

# cat.codes: 각 범주를 0, 1, 2 ... 정수로 매핑
price = dataset['price'].cat.codes.values
maint = dataset['maint'].cat.codes.values
doors = dataset['doors'].cat.codes.values
persons = dataset['persons'].cat.codes.values
lug_capacity = dataset['lug_capacity'].cat.codes.values
safety = dataset['safety'].cat.codes.values

# 1차원 배열 6개를 열 방향(새 축 1)으로 쌓아 (1728, 6) 행렬로
categorical_data = np.stack([price, maint, doors, persons, lug_capacity, safety], 1)
categorical_data[:10]
```

```text
array([[3, 3, 0, 0, 2, 1],
       [3, 3, 0, 0, 2, 2],
       [3, 3, 0, 0, 2, 0],
       ...
       [3, 3, 0, 1, 2, 1]], dtype=int8)
```

`cat.codes`는 범주를 알파벳 순서로 번호를 매기므로, 어떤 문자열이 어떤 숫자가 되었는지는 `dataset[col].cat.categories`로 따로 확인해야 한다.

#### np.stack과 np.concatenate

둘 다 배열을 합치지만 차원을 다루는 방식이 다르다.

- `np.concatenate`: **기존 축**을 따라 이어 붙인다. 차원 수는 그대로이고, 붙이는 축 이외의 크기만 같으면 된다.
- `np.stack`: **새 축**을 만들어 쌓는다. 차원이 하나 늘어나며, 모든 배열의 모양이 완전히 같아야 한다.

```python
a = np.array([[1, 2], [3, 4]])            # (2, 2)
b = np.array([[5, 6], [7, 8]])            # (2, 2)
c = np.array([[5, 6], [7, 8], [9, 10]])   # (3, 2)

print(np.concatenate((a, b), axis=0).shape)   # (4, 2)
print(np.stack((a, b), axis=0).shape)         # (2, 2, 2)
print(np.concatenate((a, c), axis=0).shape)   # (5, 2): 열 수만 같으면 됨
np.stack((a, c), axis=0)                      # 오류
```

```text
(4, 2)
(2, 2, 2)
(5, 2)
ValueError: all input arrays must have the same shape
```

이어서 넘파이 배열을 정수 텐서로 바꾼다. 임베딩 층의 입력은 `int64`(Long)여야 한다.

```python
categorical_data = torch.tensor(categorical_data, dtype=torch.int64)
```

### 레이블 변환

```python
outputs = pd.get_dummies(dataset.output)    # 문자열 → 0/1 가변수(원-핫)
outputs = outputs.values
outputs = torch.tensor(outputs).flatten()   # 1차원으로 펼침
print(categorical_data.shape)
print(outputs.shape)
```

```text
torch.Size([1728, 6])
torch.Size([6912])
```

`get_dummies()`는 범주형 칼럼을 범주마다 하나의 0/1 칼럼으로 펼친다(**가변수(dummy variable)**, 원-핫 인코딩). 예를 들어 `gender`가 `male`/`female`인 표에 적용하면 `gender_female`, `gender_male` 두 칼럼이 생기고, 숫자 칼럼은 그대로 남는다.

`flatten()`, `ravel()`, `reshape(-1)`은 모두 다차원 배열을 1차원으로 펼친다.

```python
a = np.array([[1, 2], [3, 4]])
print(a.ravel(), a.reshape(-1), a.flatten())   # 모두 [1 2 3 4]
```

**주의할 점이 있다.** 출력 모양이 `[6912]`, 즉 1728 × 4다. 원-핫 행렬(1728, 4)을 그대로 펼쳤기 때문에 이 텐서는 "샘플별 클래스 번호"가 아니라 0과 1이 길게 이어진 값이다. 아래에서 앞 1,383개를 훈련 레이블로 자르면, 실제로는 앞쪽 약 346개 샘플의 원-핫 비트가 1,383개 입력 샘플의 레이블로 짝지어진다. 그래서 뒤의 평가 결과에 클래스가 0과 1 두 개만 나타난다. 샘플마다 클래스 번호(0~3) 하나를 레이블로 쓰려면 다음처럼 바꾸는 것이 맞다.

```python
outputs = torch.tensor(pd.get_dummies(dataset.output).values).argmax(dim=1)  # (1728,)
# 또는 outputs = torch.tensor(dataset.output.astype('category').cat.codes.values, dtype=torch.int64)
```

아래 결과는 책(공식 예제)의 코드를 그대로 실행한 것이므로, 흐름을 익히는 용도로 보면 된다.

### 임베딩 크기 정하기

정수 코드 하나로 범주를 나타내면 "3이 1보다 크다" 같은 의미 없는 순서가 생긴다. 대신 범주마다 학습 가능한 작은 벡터를 할당하는 **임베딩(embedding)** 을 쓰면, 비슷한 범주는 비슷한 벡터를 갖도록 학습된다. 차원이 클수록 세밀한 관계를 담을 수 있다.

임베딩 차원에 정해진 규칙은 없지만, 흔히 고유값 개수의 절반 정도(최대 50)로 정한다.

```python
categorical_column_sizes = [len(dataset[column].cat.categories) for column in categorical_columns]
categorical_embedding_sizes = [(col_size, min(50, (col_size + 1) // 2))
                               for col_size in categorical_column_sizes]
print(categorical_embedding_sizes)
```

```text
[(4, 2), (4, 2), (4, 2), (3, 2), (3, 2), (3, 2)]
```

각 튜플은 (해당 칼럼의 범주 수, 임베딩 차원)이다. 예를 들어 `price`는 범주가 4개라 2차원 벡터로 표현한다.

### 훈련/테스트 분리

앞 80%를 훈련, 뒤 20%를 테스트로 쓴다.

```python
total_records = 1728
test_records = int(total_records * .2)     # 345

categorical_train_data = categorical_data[:total_records - test_records]
categorical_test_data = categorical_data[total_records - test_records:total_records]
train_outputs = outputs[:total_records - test_records]
test_outputs = outputs[total_records - test_records:total_records]
```

```text
1383
1383
345
345
```

이 데이터는 파일에 범주 조합 순서대로 정렬되어 있으므로, 실전에서는 섞은 뒤 나누는 것이 안전하다.

### 모델 정의

```python
class Model(nn.Module):
    def __init__(self, embedding_size, output_size, layers, p=0.4):
        super().__init__()
        # 칼럼마다 임베딩 층 하나씩: Embedding(범주 수, 임베딩 차원)
        self.all_embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_size])
        self.embedding_dropout = nn.Dropout(p)

        all_layers = []
        input_size = sum(nf for ni, nf in embedding_size)   # 임베딩을 이어 붙인 길이 = 12

        for i in layers:                                    # 은닉층마다 4개 블록
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))  # 출력층
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical):
        # i번째 칼럼의 정수 코드를 i번째 임베딩에 통과
        embeddings = [e(x_categorical[:, i]) for i, e in enumerate(self.all_embeddings)]
        x = torch.cat(embeddings, 1)      # (배치, 12)
        x = self.embedding_dropout(x)
        x = self.layers(x)
        return x
```

주요 부분은 다음과 같다.

- `nn.Module`을 상속하고, `__init__()`에서 `super().__init__()`으로 부모 클래스를 먼저 초기화한다. `self`는 생성된 객체 자신을 가리킨다.
- 생성자 인자: `embedding_size`(칼럼별 임베딩 크기 목록), `output_size`(출력 뉴런 수), `layers`(은닉층 뉴런 수 목록), `p`(드롭아웃 비율).
- `nn.ModuleList`는 여러 계층을 리스트로 보관하면서 파라미터가 모델에 정상 등록되게 한다(일반 리스트에 넣으면 등록되지 않는다).
- 은닉층 하나는 네 부품으로 구성된다.
  - `nn.Linear`: 선형 변환 $$y = Wx + b$$ ($$W$$ 가중치, $$b$$ 바이어스).
  - `nn.ReLU`: 활성화 함수.
  - `nn.BatchNorm1d`: 배치 정규화(batch normalization). 층의 출력을 배치 단위로 정규화해 학습을 안정시킨다.
  - `nn.Dropout`: 학습 중 뉴런 일부를 무작위로 꺼 과적합을 줄인다.
- `forward()`는 `model(x)`처럼 모델을 호출하면 자동으로 실행된다. `torch.cat`은 넘파이 `concatenate`의 텐서 버전이다.

은닉층을 200, 100, 50으로 두고 출력 4개(클래스 수)로 모델을 만든다.

```python
model = Model(categorical_embedding_sizes, 4, [200, 100, 50], p=0.4)
print(model)
```

```text
Model(
  (all_embeddings): ModuleList(
    (0): Embedding(4, 2)
    (1): Embedding(4, 2)
    (2): Embedding(4, 2)
    (3): Embedding(3, 2)
    (4): Embedding(3, 2)
    (5): Embedding(3, 2)
  )
  (embedding_dropout): Dropout(p=0.4, inplace=False)
  (layers): Sequential(
    (0): Linear(in_features=12, out_features=200, bias=True)
    (1): ReLU(inplace=True)
    (2): BatchNorm1d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.4, inplace=False)
    (4): Linear(in_features=200, out_features=100, bias=True)
    ...
    (12): Linear(in_features=50, out_features=4, bias=True)
  )
)
```

첫 선형층의 입력이 12인 것은 임베딩 6개(각 2차원)를 이어 붙였기 때문이다.

### 손실 함수, 옵티마이저, 장치

분류 문제이므로 크로스 엔트로피 손실과 Adam 옵티마이저를 쓴다. `nn.CrossEntropyLoss`는 내부에서 소프트맥스를 적용하므로 모델 마지막에 소프트맥스를 따로 넣지 않는다.

```python
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# GPU가 있으면 GPU, 없으면 CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
```

### 학습

```python
epochs = 500
aggregated_losses = []
train_outputs = train_outputs.to(device=device, dtype=torch.int64)

for i in range(epochs):
    i += 1
    y_pred = model(categorical_train_data)
    single_loss = loss_function(y_pred, train_outputs)
    aggregated_losses.append(single_loss)

    if i % 25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    optimizer.zero_grad()
    single_loss.backward()
    optimizer.step()

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
```

```text
epoch:   1 loss: 1.63872778
epoch:  26 loss: 1.46383297
epoch:  51 loss: 1.36062038
...
epoch: 251 loss: 0.63274646
...
epoch: 476 loss: 0.57470286
epoch: 500 loss: 0.5725595951
```

훈련 데이터 전체를 한 배치로 넣어 500번 반복하며, 손실이 1.64에서 0.57로 줄었다. 위 코드는 레이블만 `device`로 옮기고 있으므로, GPU를 실제로 쓰려면 모델과 입력도 `.to(device)`로 옮겨야 한다. 또 `aggregated_losses`에 손실 텐서를 그대로 쌓으면 연산 그래프까지 붙들고 있게 되므로 `single_loss.item()`을 저장하는 편이 낫다.

### 예측과 평가

```python
test_outputs = test_outputs.to(device=device, dtype=torch.int64)
with torch.no_grad():
    y_val = model(categorical_test_data)
    loss = loss_function(y_val, test_outputs)
print(f'Loss: {loss:.8f}')
```

```text
Loss: 0.55525565
```

테스트 손실이 훈련 손실과 비슷하므로 과적합은 크지 않다고 볼 수 있다. 엄밀히는 이 코드에서 `model.eval()`을 호출하지 않아 드롭아웃과 배치 정규화가 학습 모드로 동작한 채 평가되었다는 점도 짚어 둔다.

출력층이 4개이므로 각 샘플에 대해 점수(로짓) 4개가 나온다. 가장 큰 점수의 인덱스가 예측 클래스다.

```python
print(y_val[:5])
y_val = np.argmax(y_val, axis=1)     # 행마다 최댓값의 위치
print(y_val[:5])
```

```text
tensor([[ 2.7215,  1.6601, -2.2784, -2.1693],
        [ 2.8467,  1.7512, -2.3375, -2.2104],
        [ 1.5050,  1.0164, -2.3425, -2.1092],
        [ 2.9343,  1.4001, -5.3671, -5.7565],
        [ 3.7045,  2.1831, -7.9224, -7.9769]])
tensor([0, 0, 0, 0, 0])
```

사이킷런으로 혼동 행렬과 지표를 계산한다.

```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(test_outputs, y_val))
print(classification_report(test_outputs, y_val))
print(accuracy_score(test_outputs, y_val))
```

```text
[[257   2]
 [ 84   2]]
              precision    recall  f1-score   support

           0       0.75      0.99      0.86       259
           1       0.50      0.02      0.04        86

    accuracy                           0.75       345
   macro avg       0.63      0.51      0.45       345
weighted avg       0.69      0.75      0.65       345

0.7507246376811594
```

정확도 75%가 나왔지만, 혼동 행렬을 보면 모델이 거의 모든 샘플을 0으로 예측했다. 0이 전체의 75%(259/345)를 차지하므로 "모두 0"이라고만 답해도 정확도가 75%에 가깝다. 클래스 1의 재현율이 0.02라는 점이 이를 보여 준다(앞에서 짚은 레이블 구성 문제의 영향도 크다). 정확도 하나만 보고 판단하면 안 되는 이유이며, 아래 지표들이 필요한 이유다.

### 분류 성능 지표

이진 분류에서 예측과 실제의 조합은 네 가지다.

| | 실제 1 (Positive) | 실제 0 (Negative) |
| --- | --- | --- |
| **예측 1** | TP (True Positive) | FP (False Positive, 1종 오류) |
| **예측 0** | FN (False Negative, 2종 오류) | TN (True Negative) |

이를 바탕으로 네 가지 지표를 정의한다.

**정확도(accuracy)**: 전체 중 맞힌 비율. 긍정이든 부정이든 맞힌 것을 모두 센다.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

**재현율(recall)**: 실제 1인 것 중 모델이 1로 찾아낸 비율. 1이 드문 데이터(질병, 사기 탐지 등)에서 놓치지 않는 것이 중요할 때 본다.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

**정밀도(precision)**: 모델이 1이라고 한 것 중 실제 1인 비율. 잘못된 경보를 줄이는 것이 중요할 때 본다.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

**F1-스코어(F1-score)**: 정밀도와 재현율은 한쪽을 올리면 다른 쪽이 내려가는 **트레이드오프(trade-off)** 관계인 경우가 많다. 둘을 하나로 요약하기 위해 **조화 평균(harmonic mean)** 을 쓴다. 조화 평균은 두 값 중 작은 쪽에 끌려가므로 어느 한쪽이 낮으면 점수가 크게 떨어진다.

$$
F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

위 결과의 클래스 1은 정밀도 0.50, 재현율 0.02라서 F1이 0.04에 그친다.

## 정리

- 파이토치는 GPU 텐서 연산, 자동 미분(`torch.autograd`), 실행과 동시에 그래프를 만드는 Define by Run 방식이 특징이며, Python API → C++ 엔진 → C/CUDA 연산의 3계층 구조다.
- 텐서는 메모리에 1차원 스토리지로 저장되고, 오프셋과 스트라이드로 모양을 해석한다. `t()`는 복사 없이 스트라이드만 바꾼다.
- 텐서는 `torch.tensor(..., dtype, device)`로 만들고, `view(-1, n)`으로 모양을 바꾸며, GPU 텐서는 `.to("cpu")` 후 `.numpy()`로 변환한다.
- 데이터는 `Dataset`(`__init__`, `__len__`, `__getitem__`)과 `DataLoader`(배치, 셔플)로 공급하고, 토치비전은 MNIST 같은 표준 데이터셋을 제공한다.
- 모델은 `nn.Module` 상속(`__init__` + `forward`)이 기본이며, `nn.Sequential`로 계층을 묶고, `children()`은 직속 자식만, `modules()`는 전체 트리를 돌려준다.
- 학습 단계는 순전파 → 손실 → `zero_grad()` → `backward()` → `step()`이며, 기울기가 누적되므로 매번 초기화한다. 학습률 스케줄러는 에포크에 따라 학습률을 조절한다.
- 평가는 `model.eval()`과 `torch.no_grad()` 아래에서 하고, 텐서보드의 `SummaryWriter`로 학습 과정을 시각화한다.
- 범주형 데이터는 `cat.codes`로 정수화한 뒤 `nn.Embedding`으로 벡터화할 수 있다. 불균형 데이터에서는 정확도보다 혼동 행렬, 정밀도, 재현율, F1-스코어를 함께 봐야 한다.

---
title: "PyTorch 9: Custom Datasets"
date: 2026-10-22 15:09:00 +0900
category: study
topic: python
series: pytorch
lecture: 9
card_title: "Lecture 9: Custom Datasets"
card_desc: "Pizza, steak, sushi · Image classification folder format · torchvision.transforms · ImageFolder · Subclassing torch.utils.data.Dataset · DataLoader"
lang: ko
notes: "https://www.learnpytorch.io/04_pytorch_custom_datasets/"
notes_label: "learnpytorch.io notebook"
description: "Food101에서 뽑은 피자, 스테이크, 초밥 이미지를 내려받아 폴더 구조를 살피고, transforms로 텐서로 바꾼 뒤, ImageFolder와 직접 만든 Dataset 서브클래스 두 가지 방법으로 불러와 DataLoader로 만든다."
---

> [PyTorch for Deep Learning (Daniel Bourke, freeCodeCamp)](https://www.youtube.com/watch?v=V_xro1bcAuA) Chapter 4 정리. 교재: [learnpytorch.io](https://www.learnpytorch.io/), 코드: MIT License.
> 이전: [PyTorch 8]({% post_url 2026-10-22-py-08-cnn %}) · 다음: [PyTorch 10]({% post_url 2026-10-22-py-10-augmentation-experiments %})

## 사용자 데이터셋이란

[▶ 영상 19:44:05](https://www.youtube.com/watch?v=V_xro1bcAuA&t=71045s)

[지난 글]({% post_url 2026-10-22-py-08-cnn %})까지는 PyTorch에 내장된 FashionMNIST를 썼다. 실제로는 자기 문제에 맞는 **사용자 데이터셋(custom dataset)** 을 다루는 경우가 훨씬 많다. 사용자 데이터셋은 거의 무엇이든 될 수 있다. 음식 사진 분류 앱이라면 음식 이미지, 리뷰의 긍정/부정 분류라면 리뷰 텍스트와 평점, 소리 분류라면 오디오 샘플과 레이블, 추천 시스템이라면 다른 사람들의 구매 기록이다.

![PyTorch 도메인 라이브러리](/images/study/py/py-09-custom-datasets-1.webp)
*PyTorch 도메인 라이브러리. 피자/스테이크/초밥 사진 분류는 TorchVision, 리뷰의 긍정/부정 판별은 TorchText, 지금 나오는 노래 찾기는 TorchAudio, 비슷한 상품 추천은 TorchRec이 담당한다. 각 라이브러리는 해당 데이터 소스를 불러오는 함수를 갖고 있다. 출처: learnpytorch.io (Daniel Bourke, MIT License).*

도메인 라이브러리의 함수로 부족하면 `torch.utils.data.Dataset`을 상속해 원하는 대로 만들 수 있다. 이번 장의 목표는 피자, 스테이크, 초밥 이미지를 불러오고 이를 분류하는 모델을 만드는 것이다.

![이번 장에서 만들 것](/images/study/py/py-09-custom-datasets-2.webp)
*데이터 불러오기(스테이크, 초밥, 피자 사진) → 모델 만들기 → 모델로 예측하기(음식 앱이 각 음식을 맞게 분류). 출처: learnpytorch.io (Daniel Bourke, MIT License).*

이 글은 앞부분, 즉 이미지 폴더를 PyTorch `Dataset`과 `DataLoader`로 만드는 데까지를 다루고, 모델 학습은 [다음 글]({% post_url 2026-10-22-py-10-augmentation-experiments %})에서 한다.

### 라이브러리와 장치 설정

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
device
```

```text
'cuda'
```

## 데이터 내려받기

[▶ 영상 19:59:54](https://www.youtube.com/watch?v=V_xro1bcAuA&t=71994s)

데이터는 [Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)의 일부다. Food101은 101가지 음식을 1000장씩, 총 101,000장(학습 75,750장, 테스트 25,250장) 담은 유명한 컴퓨터 비전 벤치마크다. 처음부터 크게 하지 않고 **작게 시작한다.** 클래스는 피자, 스테이크, 초밥 3개만, 이미지는 클래스마다 무작위 10%만 쓴다. 머신러닝은 반복적인 과정이므로, 일단 작게 동작시킨 뒤 필요할 때 키운다.

```python
import requests
import zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# 폴더가 없을 때만 내려받고 압축을 푼다
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak, sushi data...")
        f.write(request.content)

    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)
```

이 데이터는 미리 형식을 맞춰 둔 것이다. 실제 프로젝트에서는 자기 데이터의 형식을 직접 맞추는 일이 흔하다.

## 데이터와 하나 되기

[▶ 영상 20:13:59](https://www.youtube.com/watch?v=V_xro1bcAuA&t=72839s)

모델을 만들기 전에 데이터를 이해하는 것이 무엇보다 중요하다. "머신러닝 모델을 만드는 데 8시간이 있다면 처음 6시간은 데이터셋 준비에 쓰겠다"는 말이 있을 정도다.

이 데이터는 **표준 이미지 분류 형식**이다. 클래스마다 별도 디렉터리가 있고, 디렉터리 이름이 곧 클래스 이름이다. ImageNet 같은 많은 벤치마크가 이 형식을 쓴다.

```text
pizza_steak_sushi/      <- 전체 데이터셋 폴더
    train/              <- 학습 이미지
        pizza/          <- 클래스 이름 = 폴더 이름
            image01.jpeg
            ...
        steak/
        sushi/
    test/               <- 테스트 이미지
        pizza/
        steak/
        sushi/
```

목표는 **이 저장 구조를 PyTorch에서 쓸 수 있는 데이터셋으로 바꾸는 것**이다. `os.walk()`로 하위 디렉터리를 돌며 파일 수를 센다.

```python
import os
def walk_through_dir(dir_path):
    """dir_path의 하위 디렉터리 수와 각 디렉터리의 이미지(파일) 수를 출력한다."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(image_path)
```

```text
There are 2 directories and 1 images in 'data/pizza_steak_sushi'.
There are 3 directories and 0 images in 'data/pizza_steak_sushi/test'.
There are 0 directories and 19 images in 'data/pizza_steak_sushi/test/steak'.
There are 0 directories and 31 images in 'data/pizza_steak_sushi/test/sushi'.
There are 0 directories and 25 images in 'data/pizza_steak_sushi/test/pizza'.
There are 3 directories and 0 images in 'data/pizza_steak_sushi/train'.
There are 0 directories and 75 images in 'data/pizza_steak_sushi/train/steak'.
There are 0 directories and 72 images in 'data/pizza_steak_sushi/train/sushi'.
There are 0 directories and 78 images in 'data/pizza_steak_sushi/train/pizza'.
```

학습 이미지는 클래스당 약 75장, 테스트 이미지는 약 25장이다. 학습/테스트 경로를 변수로 둔다.

```python
train_dir = image_path / "train"
test_dir = image_path / "test"
```

### 이미지 하나 시각화하기

1. `pathlib.Path.glob()`으로 `.jpg`로 끝나는 모든 이미지 경로를 얻는다.
2. `random.choice()`로 하나를 고른다.
3. `pathlib.Path.parent.stem`으로 클래스 이름(상위 디렉터리 이름)을 얻는다.
4. `PIL.Image.open()`으로 이미지를 연다.
5. 이미지와 메타데이터를 출력한다.

```python
import random
from PIL import Image

random.seed(42)

image_path_list = list(image_path.glob("*/*/*.jpg"))  # train|test / 클래스 / 파일
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem           # 예: "pizza"
img = Image.open(random_image_path)

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")
img
```

```text
Random image path: data/pizza_steak_sushi/test/pizza/2124579.jpg
Image class: pizza
Image height: 384
Image width: 512
```

`matplotlib`으로 그리려면 NumPy 배열로 바꾼다. 이때 모양은 `[height, width, color_channels]`, 즉 HWC다.

```python
import numpy as np
import matplotlib.pyplot as plt

img_as_array = np.asarray(img)

plt.figure(figsize=(10, 7))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False)
```

배열 모양은 `(384, 512, 3)`이다.

## 이미지를 텐서로 바꾸기

[▶ 영상 20:39:11](https://www.youtube.com/watch?v=V_xro1bcAuA&t=74351s)

이미지 데이터를 PyTorch에서 쓰려면 두 가지가 필요하다.

1. 텐서(이미지의 수치 표현)로 바꾼다.
2. `torch.utils.data.Dataset`으로, 이어서 `torch.utils.data.DataLoader`로 만든다.

문제 영역마다 미리 만들어진 데이터셋 함수가 있다.

| 문제 영역 | 미리 만들어진 데이터셋과 함수 |
|---|---|
| 비전 | `torchvision.datasets` |
| 오디오 | `torchaudio.datasets` |
| 텍스트 | `torchtext.datasets` |
| 추천 시스템 | `torchrec.datasets` |

비전 문제이므로 `torchvision.datasets`와 `torchvision.transforms`를 쓴다.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

### torchvision.transforms로 변환하기

`torchvision.transforms`에는 이미지를 정리하고, 텐서로 바꾸고, **데이터 증강(data augmentation)** 용으로 조작하는 함수들이 있다. 다음 세 단계를 `transforms.Compose()`로 묶는다.

1. `transforms.Resize()`: 약 512×512 이미지를 CNN Explainer와 같은 64×64로 줄인다.
2. `transforms.RandomHorizontalFlip()`: 무작위로 좌우 반전한다(인위적으로 데이터를 바꾸므로 일종의 데이터 증강이다).
3. `transforms.ToTensor()`: PIL 이미지를 PyTorch 텐서로 바꾼다.

```python
data_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% 확률로 반전
    transforms.ToTensor()  # 0~255 픽셀을 0.0~1.0으로, HWC를 CHW로 바꾼다
])
```

원본과 변환된 이미지를 나란히 그리는 함수를 만든다.

```python
def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """image_paths에서 n개를 무작위로 골라 원본과 변환 결과를 나란히 그린다."""
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # PyTorch는 [C, H, W], matplotlib은 [H, W, C]이므로 permute로 축 순서를 바꾼다
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)

plot_transformed_images(image_path_list, transform=data_transform, n=3)
```

변환된 이미지는 64×64로 작아지고 때로 좌우가 뒤집힌다. 이미지가 클수록 모델이 얻을 수 있는 정보는 많지만 계산량도 늘어난다. 예를 들어 `[256, 256, 3]`은 `[64, 64, 3]`보다 픽셀이 16배 많다($$(256 \cdot 256 \cdot 3)/(64 \cdot 64 \cdot 3) = 16$$).

## 방법 1: ImageFolder로 불러오기

데이터가 표준 이미지 분류 형식이므로 `torchvision.datasets.ImageFolder`를 바로 쓸 수 있다. 디렉터리 경로와 적용할 변환만 넘기면 된다.

```python
train_data = datasets.ImageFolder(root=train_dir,            # 이미지 폴더
                                  transform=data_transform,  # 이미지 변환
                                  target_transform=None)     # 레이블 변환(필요하면)

test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform)

print(f"Train data:\n{train_data}\nTest data:\n{test_data}")
```

```text
Train data:
Dataset ImageFolder
    Number of datapoints: 225
    Root location: data/pizza_steak_sushi/train
    StandardTransform
Transform: Compose(
               Resize(size=(64, 64), interpolation=bilinear, max_size=None, antialias=None)
               RandomHorizontalFlip(p=0.5)
               ToTensor()
           )
Test data:
Dataset ImageFolder
    Number of datapoints: 75
    ...
```

`classes`와 `class_to_idx` 속성으로 클래스 정보를 얻는다. 폴더 이름을 알파벳순으로 정렬해 0부터 번호를 붙인 것이다.

```python
class_names = train_data.classes
class_dict = train_data.class_to_idx
class_names, class_dict, len(train_data), len(test_data)
```

```text
(['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2}, 225, 75)
```

인덱싱하면 `(이미지 텐서, 정수 레이블)` 튜플이 나온다.

```python
img, label = train_data[0][0], train_data[0][1]
print(f"Image shape: {img.shape}")
print(f"Image datatype: {img.dtype}")
print(f"Image label: {label}")
print(f"Label datatype: {type(label)}")
```

```text
Image shape: torch.Size([3, 64, 64])
Image datatype: torch.float32
Image label: 0
Label datatype: <class 'int'>
```

`matplotlib`으로 그리려면 CHW를 HWC로 바꿔야 한다. `permute(1, 2, 0)`은 원래 1번 축(H)을 0번으로, 2번 축(W)을 1번으로, 0번 축(C)을 2번으로 옮긴다.

```python
img_permute = img.permute(1, 2, 0)
print(f"Original shape: {img.shape} -> [color_channels, height, width]")
print(f"Image permute shape: {img_permute.shape} -> [height, width, color_channels]")

plt.figure(figsize=(10, 7))
plt.imshow(img_permute)
plt.axis("off")
plt.title(class_names[label], fontsize=14)
```

```text
Original shape: torch.Size([3, 64, 64]) -> [color_channels, height, width]
Image permute shape: torch.Size([64, 64, 3]) -> [height, width, color_channels]
```

512×512에서 64×64로 줄였기 때문에 이미지가 눈에 띄게 뭉개진다. 사람이 보기에 알아보기 어려워진 이미지는 모델에게도 어렵다고 생각하면 된다.

### DataLoader로 만들기

[▶ 영상 21:16:16](https://www.youtube.com/watch?v=V_xro1bcAuA&t=76576s)

`DataLoader`로 감싸면 모델이 샘플과 레이블을 순회하며 학습할 수 있다. 여기서는 단순하게 `batch_size=1`, `num_workers=1`로 둔다. `num_workers`는 데이터를 불러올 하위 프로세스 수다. 클수록 데이터 로딩에 더 많은 연산 자원을 쓰며, 흔히 `os.cpu_count()`로 CPU 코어 수만큼 준다.

```python
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,     # 배치당 샘플 수
                              num_workers=1,    # 데이터 로딩 하위 프로세스 수
                              shuffle=True)     # 학습 데이터는 섞는다

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)     # 테스트 데이터는 보통 섞지 않는다

img, label = next(iter(train_dataloader))
print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label.shape}")
```

```text
Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])
```

## 방법 2: Dataset을 직접 만들기

[▶ 영상 21:25:20](https://www.youtube.com/watch?v=V_xro1bcAuA&t=77120s)

`ImageFolder` 같은 함수가 없거나, 내 문제에 맞는 것이 없다면 직접 만들면 된다. 장단점은 다음과 같다.

| 직접 만드는 Dataset의 장점 | 직접 만드는 Dataset의 단점 |
|---|---|
| 거의 무엇이든 `Dataset`으로 만들 수 있다 | 무엇이든 만들 수 있다고 해서 잘 작동한다는 보장은 없다 |
| PyTorch가 미리 만든 함수에 묶이지 않는다 | 코드가 길어지고, 그만큼 오류나 성능 문제가 생기기 쉽다 |

연습 삼아 `torch.utils.data.Dataset`(모든 PyTorch `Dataset`의 기반 클래스)을 상속해 `ImageFolder`를 재현한다. 필요한 모듈을 가져온다.

```python
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List
```

`os`는 디렉터리를, `pathlib`은 파일 경로를, `PIL.Image`는 이미지 로딩을, `typing`은 타입 힌트를 위해 쓴다.

### 클래스 이름을 찾는 도우미 함수

`ImageFolder`의 `classes`와 `class_to_idx` 속성을 재현하는 함수를 만든다.

1. `os.scandir()`로 대상 디렉터리를 훑어 클래스 이름을 얻는다(표준 이미지 분류 형식을 가정).
2. 클래스 이름을 못 찾으면 오류를 낸다(디렉터리 구조에 문제가 있다는 뜻이다).
3. 클래스 이름을 숫자 레이블로 대응시키는 사전을 만든다.

```python
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """대상 디렉터리에서 클래스 폴더 이름을 찾는다.

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. 디렉터리만 골라 이름을 정렬
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. 못 찾으면 오류
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. 이름 -> 인덱스 사전 (컴퓨터는 문자열보다 숫자 레이블을 선호한다)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

find_classes(train_dir)
```

```text
(['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})
```

### ImageFolder를 재현하는 Dataset 클래스

[▶ 영상 21:42:29](https://www.youtube.com/watch?v=V_xro1bcAuA&t=78149s)

순서는 다음과 같다.

1. `torch.utils.data.Dataset`을 상속한다.
2. 대상 디렉터리 `targ_dir`과 선택적 `transform`을 받아 초기화한다.
3. 속성을 만든다. `paths`(이미지 경로들), `transform`(없을 수도 있음), `classes`와 `class_to_idx`(`find_classes()`로).
4. 파일에서 이미지를 불러오는 함수를 만든다(PIL 또는 `torchvision.io`).
5. `__len__`을 덮어써서 샘플 수를 반환한다. 필수는 아니지만 권장된다. `len(dataset)`을 쓸 수 있게 해 준다.
6. `__getitem__`을 덮어써서 샘플 하나를 반환한다. **필수**다.

```python
from torch.utils.data import Dataset

# 1. Dataset 상속
class ImageFolderCustom(Dataset):

    # 2. 디렉터리와 (선택) transform으로 초기화
    def __init__(self, targ_dir: str, transform=None) -> None:
        # 3. 속성 만들기
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))  # .png 등이 있으면 수정 필요
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. 이미지 불러오기
    def load_image(self, index: int) -> Image.Image:
        "경로로 이미지를 열어 반환한다."
        image_path = self.paths[index]
        return Image.open(image_path)

    # 5. __len__ (선택이지만 권장)
    def __len__(self) -> int:
        "전체 샘플 수를 반환한다."
        return len(self.paths)

    # 6. __getitem__ (필수)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "샘플 하나, 즉 (X, y)를 반환한다."
        img = self.load_image(index)
        class_name = self.paths[index].parent.name  # data_folder/class_name/image.jpg 를 가정
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
```

이미지는 `__init__`에서 한꺼번에 읽지 않고 **경로만** 저장해 두었다가, `__getitem__`이 호출될 때 한 장씩 연다. 그래서 메모리에 다 들어가지 않는 큰 데이터셋도 다룰 수 있다. 코드는 길지만 한 번 써 두면 `data_loader.py` 같은 파일로 옮겨 재사용할 수 있다.

학습용과 테스트용 변환을 따로 만든다. 데이터 증강(좌우 반전)은 학습 데이터에만 한다.

```python
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([  # 테스트 데이터는 크기만 맞춘다
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)

len(train_data_custom), len(test_data_custom), train_data_custom.classes, train_data_custom.class_to_idx
```

```text
(225, 75, ['pizza', 'steak', 'sushi'], {'pizza': 0, 'steak': 1, 'sushi': 2})
```

`ImageFolder`로 만든 데이터셋과 같은지 확인한다.

```python
print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)
```

```text
True
True
True
```

### 무작위 이미지를 보여 주는 함수

`__getitem__`이 제대로 동작하는지 그림으로 확인한다. 함수는 다음을 한다.

1. `Dataset`과 클래스 이름 목록, 표시할 개수 `n`, 난수 시드를 받는다.
2. 화면이 넘치지 않게 `n`을 최대 10으로 제한한다.
3. 시드가 있으면 설정한다.
4. `random.sample()`로 무작위 인덱스를 고른다.
5. `matplotlib` 그림을 준비한다.
6. 인덱스를 돌며 샘플을 그린다.
7. 그리기 전에 이미지를 HWC로 바꾼다.

```python
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    # 2. n이 너무 크면 조정
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    # 3. 시드
    if seed:
        random.seed(seed)

    # 4. 무작위 인덱스
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. 그림 준비
    plt.figure(figsize=(16, 8))

    # 6. 샘플 그리기
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. [C, H, W] -> [H, W, C]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        plt.subplot(1, n, i + 1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
            plt.title(title)

display_random_images(train_data, n=5, classes=class_names, seed=None)
display_random_images(train_data_custom, n=12, classes=class_names, seed=None)
```

```text
For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.
```

두 데이터셋 모두 64×64 피자, 스테이크, 초밥 이미지를 클래스 이름과 함께 보여 준다. `n=12`는 10으로 줄어든다.

## 직접 만든 Dataset을 DataLoader로

[▶ 영상 22:21:50](https://www.youtube.com/watch?v=V_xro1bcAuA&t=80510s)

`ImageFolderCustom`은 `torch.utils.data.Dataset`을 상속했으므로 `DataLoader`에 그대로 넣을 수 있다.

```python
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=1,
                                     num_workers=0,  # 0이면 메인 프로세스에서 불러온다
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False)

img_custom, label_custom = next(iter(train_dataloader_custom))
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")
```

```text
Image shape: torch.Size([1, 3, 64, 64]) -> [batch_size, color_channels, height, width]
Label shape: torch.Size([1])
```

`ImageFolder`로 만든 `DataLoader`와 모양이 같다. 이로써 이미지 폴더를 텐서 배치로 바꾸는 두 가지 경로가 모두 준비되었다.

## 정리

- 사용자 데이터셋은 자기 문제에 맞는 데이터 모음이다. TorchVision, TorchText, TorchAudio, TorchRec 같은 도메인 라이브러리에 불러오기 함수가 있고, 부족하면 `Dataset`을 상속해 직접 만든다.
- Food101에서 피자, 스테이크, 초밥 3개 클래스의 10%(학습 225장, 테스트 75장)만 가져와 작게 시작했다.
- 표준 이미지 분류 형식은 `train/클래스명/이미지` 구조이며, 폴더 이름이 곧 레이블이다. `os.walk()`, `Path.glob()`, `Path.parent.stem`으로 데이터를 살핀다.
- `transforms.Compose([Resize, RandomHorizontalFlip, ToTensor])`로 PIL 이미지를 0~1 범위 `[3, 64, 64]` float32 텐서로 바꾼다. 그릴 때는 `permute(1, 2, 0)`으로 CHW를 HWC로 바꾼다.
- `datasets.ImageFolder(root, transform)`는 폴더 구조에서 `classes`, `class_to_idx`, `(이미지, 레이블)` 샘플을 자동으로 만든다.
- 직접 만드는 `Dataset`은 `__init__`에서 경로 목록과 클래스 사전을 준비하고, `__len__`(권장)과 `__getitem__`(필수)을 구현한다. 이미지는 요청될 때 한 장씩 연다.
- 어느 방법으로 만든 `Dataset`이든 `DataLoader(batch_size, shuffle, num_workers)`로 감싸면 `[batch_size, 3, 64, 64]` 배치를 순회할 수 있다.

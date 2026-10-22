---
title: "Deep Learning & PyTorch 11: Clustering"
date: 2026-10-22 17:11:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 11
card_title: "Chapter 11: Clustering"
card_desc: "K-means and the elbow method · kmeans_pytorch on Iris · Feature scaling · Gaussian mixture models · Self-organizing maps"
lang: ko
notes: "https://thebook.io/080289/0795/"
notes_label: "thebook.io chapter 11"
description: "K-평균 군집화의 원리와 엘보 방법을 복습하고 kmeans_pytorch로 아이리스 데이터를 군집화한 뒤, 가우시안 혼합 모델과 자기 조직화 지도를 사이킷런과 MiniSom으로 실습한다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 11장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap11)</li>
</ul>

11장은 정답 없이 데이터를 비슷한 것끼리 묶는 클러스터링을 다룬다. 3장에서 머신러닝 알고리즘으로 소개한 K-평균을 PyTorch 위에서 돌려 보고, 3장에서 다루지 않은 가우시안 혼합 모델과 자기 조직화 지도를 추가로 살펴본다. 분량이 짧은 장이다.

## 11.1 클러스터링이란

[📖 thebook.io](https://thebook.io/080289/0796/)

**클러스터링(clustering)** 은 레이블 없이 주어진 데이터를 특성이 비슷한 것끼리 몇 개의 그룹(클러스터)으로 묶는 비지도 학습(unsupervised learning) 기법이다. 고객 세분화, 이미지 분할, 이상 탐지의 전처리 등에 쓰인다. 전통적으로는 머신러닝 알고리즘이지만, 딥러닝으로 뽑은 특성 위에서 클러스터링하거나 GPU 텐서 연산으로 구현하면 성능과 속도를 끌어올릴 수 있다.

## 11.2 클러스터링 알고리즘 유형

[📖 thebook.io](https://thebook.io/080289/0797/)

이 절에서는 세 가지 알고리즘을 다룬다.

| 알고리즘 | 클러스터를 보는 방식 | 실습 도구 |
| --- | --- | --- |
| K-평균 | 중심점까지의 거리가 가장 가까운 쪽에 하나씩 배정(하드 할당) | `kmeans_pytorch` |
| 가우시안 혼합 모델 | 여러 정규분포가 섞인 확률 모델, 각 분포에 속할 확률로 배정(소프트 할당) | 사이킷런 `GaussianMixture` |
| 자기 조직화 지도 | 2차원 격자의 뉴런들이 경쟁하며 입력 공간을 지도처럼 펼침 | `MiniSom` |

### 11.2.1 K-평균 군집화

**K-평균 군집화(K-means clustering)** 는 데이터를 K개의 클러스터로 나누되, 각 데이터가 자기 클러스터 중심(centroid)에 최대한 가깝도록 만드는 알고리즘이다. 학습은 다음 반복으로 이루어진다.

1. 클러스터 중심이 될 점 K개를 임의로 고른다.
2. 각 데이터를 가장 가까운 중심의 클러스터에 배정한다.
3. 클러스터마다 배정된 데이터의 평균을 구해 새 중심으로 삼는다.
4. 배정이 더 이상 바뀌지 않을 때까지 2와 3을 반복한다.

이 반복은 다음의 **WCSS(Within-Cluster Sum of Squares)**, 즉 각 데이터와 소속 클러스터 중심 사이 거리 제곱의 합을 줄여 나가는 과정이다. $$C_k$$는 $$k$$번째 클러스터, $$\mu_k$$는 그 중심이다.

$$
\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} \lVert x - \mu_k \rVert^2
$$

**K를 정하는 법: 엘보 방법**

K를 크게 할수록 WCSS는 항상 줄어든다. 극단적으로 데이터 개수만큼 클러스터를 만들면 모든 점이 자기 자신이 중심이라 WCSS가 0이지만, 아무 의미 없는 군집이다. 그래서 K를 1부터 늘려 가며 WCSS를 그려 보고, 감소 폭이 급격히 줄어들어 곡선이 팔꿈치처럼 꺾이는 지점을 고른다. 이것이 **엘보 방법(elbow method)** 이다. 꺾이는 지점을 기계적으로 찾으려면 다음과 같이 한다.

1. 곡선의 첫 점과 마지막 점을 직선으로 잇는다.
2. 곡선 위 각 점에서 그 직선까지의 수직 거리를 잰다.
3. 거리가 가장 먼 점을 엘보로 고른다.

**PyTorch로 K-평균 구현하기**

PyTorch는 딥러닝 프레임워크라서 사이킷런처럼 머신러닝 알고리즘을 바로 쓸 수 있는 함수를 제공하지 않는다. 실무에서는 사이킷런이 편하지만, 논문이나 오픈소스에서 PyTorch로 구현된 머신러닝 코드를 만났을 때 읽을 수 있도록 한 번 써 본다. 여기서는 텐서 연산으로 K-평균을 구현해 둔 `kmeans_pytorch` 패키지를 쓴다(`pip install kmeans_pytorch`).

데이터는 붓꽃(Iris) 150개다. 꽃받침(sepal)과 꽃잎(petal)의 길이, 너비 네 가지 측정값과 품종(`Species`, 세 종류) 열이 있다.

```python
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from kmeans_pytorch import kmeans, kmeans_predict

df = pd.read_csv('../chap11/data/iris.csv')
df.info()   # 열별 자료형 확인: 숫자가 아닌 열을 찾는다
```

```text
 #   Column         Non-Null Count  Dtype
 0   Id             150 non-null    int64
 1   SepalLengthCm  150 non-null    float64
 2   SepalWidthCm   150 non-null    float64
 3   PetalLengthCm  150 non-null    float64
 4   PetalWidthCm   150 non-null    float64
 5   Species        150 non-null    object
```

`Species`가 문자열(`object`)이므로 숫자로 바꿔야 한다. `pd.get_dummies`로 품종마다 0/1 열을 만드는 원-핫 인코딩을 하면 `Species_Iris-setosa`, `Species_Iris-versicolor`, `Species_Iris-virginica` 세 열이 생겨 전체가 8열이 된다. 그다음 80:20으로 나눈다. PyTorch의 `random_split`은 원본을 가리키는 인덱스 묶음(`Subset`)을 돌려주므로, 데이터프레임 자체가 필요한 이 예제에서는 사이킷런의 `train_test_split`을 쓴다.

```python
data = pd.get_dummies(df, columns=['Species'])          # 품종 -> 원-핫 3열
x, y = train_test_split(data, test_size=0.2, random_state=123)  # x: 학습 120행, y: 테스트 30행

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

열마다 값의 범위가 다르면 거리 계산에서 범위가 큰 열이 결과를 좌우한다. 그래서 **특성 스케일링(feature scaling)** 으로 범위를 맞춘다. 대표적인 방법은 평균 0, 분산 1로 맞추는 **표준화(standardization)** 와 일정 구간(예: 0~1)으로 압축하는 **정규화(normalization)** 이며, 여기서는 표준화를 쓴다.

$$
z = \frac{x - \mu}{\sigma}
$$

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit(data).transform(x)   # fit: 평균, 표준편차 계산 / transform: 적용
y_scaled = scaler.fit(y).transform(y)

x = torch.from_numpy(X_scaled)             # 넘파이 배열 -> 텐서 (메모리 공유)
y = torch.from_numpy(y_scaled)
print(x.size(), y.size())
```

```text
torch.Size([120, 8]) torch.Size([30, 8])
```

`torch.from_numpy()`는 데이터를 복사하지 않고 넘파이 배열과 메모리를 공유하는 텐서를 만든다. 그래서 텐서 쪽 값을 바꾸면 원래 배열도 함께 바뀐다.

```python
import numpy as np
a = np.array([1, 2, 3])
t = torch.from_numpy(a)
t[0] = -1
print(a)   # [-1  2  3]
```

전처리 코드에서 눈여겨볼 점이 몇 가지 있다.

- 8개 열에 `Id`와 원-핫 품종 세 열이 그대로 들어 있다. `Id`는 순번일 뿐인데, 이 데이터는 품종 순서로 정렬되어 있어 `Id`가 사실상 품종 정보를 담고 있고, 원-핫 열은 정답 그 자체다. 즉 군집화에 정답이 섞여 들어가므로 결과가 품종과 잘 맞는 것은 당연하다. 순수한 비지도 군집화를 하려면 측정값 네 열만 써야 한다.
- 학습 데이터 변환에 전체 데이터(`data`)로 맞춘 스케일러를, 테스트 데이터에는 테스트 데이터 자신으로 다시 맞춘 스케일러를 쓴다. 일반적으로는 학습 데이터로만 `fit`하고 그 스케일러로 학습, 테스트를 모두 `transform`하는 것이 원칙이다.

이제 K=3(붓꽃 품종 수)으로 군집화한다.

```python
num_clusters = 3
cluster_ids_x, cluster_centers = kmeans(
    X=x,                          # 학습 데이터
    num_clusters=num_clusters,    # K
    distance='euclidean',         # 거리 척도 ('cosine'도 가능)
    device=device
)
```

```text
running k-means on cpu..
[running kmeans]: 3it [00:00, 176.95it/s, center_shift=0.000000, iteration=4, tol=0.000100]
```

거리 척도로는 유클리드 거리(Euclidean distance)를 썼다. 두 점 $$(x_1, y_1)$$, $$(x_2, y_2)$$ 사이의 직선 거리이고, n차원이면 좌표 차이의 제곱합에 제곱근을 씌운다. 그 밖에 맨해튼 거리, 민코프스키 거리, 코사인 유사도 등이 있다.

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

진행 표시줄의 `center_shift`는 이번 반복에서 중심이 움직인 정도, `tol`은 이 값이 그보다 작아지면 멈추는 허용 오차(tolerance), `iteration`은 멈출 때까지의 반복 횟수다. 4번 만에 중심 이동이 0이 되어 수렴했다.

```python
print(cluster_ids_x)     # 학습 데이터 120개 각각의 클러스터 번호 (0, 1, 2)
print(cluster_centers)   # (3, 8): 각 클러스터 중심의 좌표
```

```text
tensor([2, 2, 1, 1, 0, 0, 2, 1, 1, 0, 0, 1, 2, 2, 2, 2, 2, 0, 1, 1, ...])
tensor([[-0.0063,  0.1292, -0.6246,  0.2742,  0.1631, -0.7071,  1.4142, -0.7071],
        [-1.1850, -1.0022,  0.8757, -1.3028, -1.2384,  1.4142, -0.7071, -0.7071],
        [ 1.1509,  0.9013, -0.2199,  1.0354,  1.0906, -0.7071, -0.7071,  1.4142]])
```

중심의 마지막 세 열을 보면 각 클러스터가 원-핫 품종 열 하나에만 1.4142(표준화된 1)를 갖는다. 앞서 말한 대로 정답 열이 군집을 결정했다는 증거다. 학습한 중심으로 테스트 데이터의 클러스터를 예측하고, 처음 두 특성으로 산점도를 그린다.

```python
cluster_ids_y = kmeans_predict(y, cluster_centers, 'euclidean', device=device)
print(cluster_ids_y)

import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3), dpi=160)
plt.scatter(y[:, 0], y[:, 1], c=cluster_ids_y, cmap='viridis', marker='x')  # 테스트 데이터
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],                    # 클러스터 중심
            c='white', alpha=0.6, edgecolors='black', linewidths=2)
plt.tight_layout()   # 축 레이블이 잘리지 않게 여백 자동 조정
plt.show()
```

```text
tensor([0, 2, 2, 0, 1, 2, 0, 1, 1, 0, 2, 1, 0, 2, 2, 2, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 2, 2, 1])
```

`kmeans_pytorch`는 편하지만 조절할 수 있는 하이퍼파라미터가 거의 없다. 초기화 방식이나 반복 조건을 바꾸고 싶다면 거리 계산과 중심 갱신을 직접 구현해야 한다. 결국 전통적인 머신러닝 알고리즘은 사이킷런 같은 라이브러리로 쓰는 편이 낫고, 이후 예제는 PyTorch 없이 진행한다.

### 11.2.2 가우시안 혼합 모델

**가우시안 혼합 모델(Gaussian Mixture Model, GMM)** 은 데이터가 K개의 가우시안(정규) 분포가 섞인 분포에서 나왔다고 가정한다. 하나의 정규분포로는 표현할 수 없는 봉우리가 여러 개인 복잡한 분포도, 정규분포 여러 개를 가중치를 주어 더하면 근사할 수 있다는 아이디어다. K는 사람이 정하는 하이퍼파라미터다.

$$
p(x) = \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k), \qquad \sum_{k=1}^{K} \pi_k = 1
$$

- $$\pi_k$$: $$k$$번째 분포가 선택될 확률(혼합 계수).
- $$\mu_k$$, $$\Sigma_k$$: $$k$$번째 분포의 평균 벡터와 공분산 행렬. 공분산 덕분에 각 클러스터가 원이 아니라 기울어진 타원 모양일 수 있다.

GMM으로 군집화한다는 것은 각 데이터 $$x_n$$이 어느 분포에서 나왔을지 맞히는 것이다. $$z_{nk}$$를 "$$x_n$$이 $$k$$번째 분포에서 나왔으면 1, 아니면 0"인 숨은 변수라 하면, $$x_n$$을 관측했을 때 $$z_{nk} = 1$$일 사후 확률을 베이즈 정리로 계산할 수 있다.

$$
\gamma(z_{nk}) = p(z_{nk} = 1 \mid x_n) = \frac{\pi_k \, \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \, \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}
$$

$$\pi_k$$는 데이터를 보기 전의 사전 확률이고, $$\gamma(z_{nk})$$는 데이터를 본 뒤의 사후 확률로, $$k$$번째 성분의 **책임값(responsibility)** 이라 부른다. 데이터마다 K개의 책임값을 계산해 가장 큰 쪽에 배정하면 하드 군집이 되고, 확률 그대로 쓰면 "이 점은 클러스터 A일 확률 70%" 같은 소프트 군집이 된다. 파라미터 $$\pi$$, $$\mu$$, $$\Sigma$$는 보통 EM 알고리즘으로 추정한다. 책임값을 계산하는 단계(E)와 책임값으로 가중 평균을 내어 파라미터를 다시 구하는 단계(M)를 번갈아 반복하는데, K-평균은 이 과정에서 공분산을 고정하고 하드 할당만 쓰는 특수한 경우로 볼 수 있다.

사이킷런의 `GaussianMixture`로 2차원 데이터(`data.npy`, `np.save`로 저장한 넘파이 배열 하나)를 두 개의 가우시안으로 맞춘다.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

X_train = np.load('../chap11/data/data.npy')   # np.save()로 저장한 배열 불러오기

gmm = GaussianMixture(n_components=2)          # 가우시안 2개
gmm.fit(X_train)
print(gmm.means_)          # 각 성분의 평균
print(gmm.covariances_)    # 각 성분의 공분산 행렬

# -1~6 범위의 50x50 격자에서 로그 밀도를 계산해 등고선으로 그린다
X, Y = np.meshgrid(np.linspace(-1, 6), np.linspace(-1, 6))
XX = np.array([X.ravel(), Y.ravel()]).T        # (2500, 2) 격자 좌표
Z = gmm.score_samples(XX).reshape((50, 50))    # 각 좌표의 log p(x)
plt.contour(X, Y, Z)
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.show()
```

```text
[[3.04641134 3.10654272]
 [1.60718016 1.35251723]]

[[[ 0.83656079  0.37865596]
  [ 0.37865596  0.72727426]]

 [[ 0.74995307 -0.5010097 ]
  [-0.5010097   0.74377694]]]
```

두 성분의 중심은 (3.05, 3.11)과 (1.61, 1.35)다. 공분산의 비대각 원소를 보면 첫 번째 성분은 양수(0.38)라 오른쪽 위로 기운 타원, 두 번째 성분은 음수(-0.50)라 오른쪽 아래로 기운 타원이다. 그림에서는 두 타원 등고선이 겹쳐 있고, 데이터 대부분이 각 타원의 가장 안쪽, 즉 평균 근처에 모여 있다. 이처럼 GMM은 데이터가 둘 이상의 정규분포에서 나왔다고 볼 수 있을 때, 모양과 방향이 다른 클러스터까지 잡아낼 수 있다.

### 11.2.3 자기 조직화 지도

**자기 조직화 지도(Self-Organizing Map, SOM)** 는 핀란드의 테우보 코호넌(Teuvo Kohonen)이 제안한 신경망으로, 대뇌 피질에서 비슷한 자극에 반응하는 뉴런들이 가까이 모여 있는 현상을 모델링했다. 정답 없이 스스로 학습해 고차원 데이터를 2차원 격자 위에 펼쳐 놓으며, 비슷한 입력은 격자에서 가까운 위치에 놓인다. 그래서 군집화와 시각화를 동시에 할 수 있다.

**구조**

층은 두 개다.

- **입력층(input layer)**: 입력 벡터의 각 성분.
- **경쟁층(competitive layer)**: 2차원 격자로 배열된 뉴런들. 모든 입력 노드가 모든 경쟁층 뉴런과 연결되어 있고, 뉴런 $$j$$에 연결된 가중치 벡터 $$W_j$$(연결 강도, 0~1로 정규화)는 입력 공간에서 그 뉴런이 대표하는 위치다.

**학습 과정**

1. **초기화(initialization)**: 모든 연결 가중치를 작은 임의의 값으로 둔다.
2. **경쟁(competition)**: 입력 $$X$$가 들어오면 각 뉴런의 가중치 벡터와의 거리를 계산하고, 가장 가까운 뉴런 하나가 승리한다(승자 독식, winner-take-all). 이 뉴런을 **BMU(Best Matching Unit)** 라 한다. $$n$$은 입력 벡터 크기다.

$$
D_j = \sum_{i=1}^{n} (X_i - W_{ij})^2, \qquad j^\ast = \arg\min_j D_j
$$

3. **협력(cooperation)**: 승자 뉴런을 중심으로 격자 위의 일정 반경 안 이웃 뉴런을 정한다. 모든 뉴런이 아니라 이 제한된 이웃만 함께 학습한다.
4. **적응(adaptation)**: 승자와 이웃의 가중치를 입력 쪽으로 끌어당긴다. $$\alpha$$는 학습률이다. 실제 구현에서는 승자에서 멀수록 작게 움직이도록 이웃 함수 $$h$$(보통 가우시안)를 곱한다.

$$
W_{ij}^{\text{new}} = W_{ij}^{\text{old}} + \alpha \, h_{j^\ast j} \, (X_i - W_{ij}^{\text{old}})
$$

2~4를 원하는 횟수만큼 반복하며, 학습이 진행될수록 학습률과 이웃 반경을 줄여 지도를 점차 고정한다.

**MiniSom으로 손글씨 숫자 펼치기**

사이킷런의 손글씨 숫자 데이터(`load_digits`, 8x8 픽셀 = 64차원, 숫자 0~9)를 `MiniSom`(`pip install MiniSom`)으로 16x16 격자에 펼친다.

```python
import numpy as np
from sklearn.datasets import load_digits
from minisom import MiniSom
from pylab import plot, axis, show, pcolor, colorbar, bone

digits = load_digits()
data = digits.data       # (1797, 64)
labels = digits.target   # 0~9

som = MiniSom(16, 16, 64,          # 격자 가로, 세로, 입력 벡터 차원
              sigma=1.0,           # 이웃 반경 초깃값
              learning_rate=0.5)   # 학습률 초깃값
som.random_weights_init(data)      # 가중치를 데이터 샘플로 초기화
som.train_random(data, 10000)      # 무작위 샘플로 10000번 갱신

bone()
pcolor(som.distance_map().T)       # U-matrix: 이웃 뉴런 가중치 사이 평균 거리
colorbar()
```

MiniSom의 기본 설정에서 이웃 반경과 학습률은 반복 횟수 $$t$$에 따라 $$\sigma(t) = \sigma / (1 + t/T)$$ 꼴로 줄어든다($$T$$는 총 반복 횟수의 절반). `distance_map()`이 그리는 그림은 **U-행렬(U-matrix)** 로, 각 뉴런과 이웃 뉴런의 가중치 거리를 색으로 나타낸다. 밝은 칸은 이웃과 멀리 떨어진 곳, 즉 클러스터 사이의 경계이고, 어두운 칸이 모인 영역이 하나의 클러스터다.

이제 각 데이터의 BMU 위치에 숫자별로 다른 모양과 색의 마커를 찍어 본다.

```python
markers = ['o', 'v', '1', '3', '8', 's', 'p', 'x', 'D', '*']
colors = ["r", "g", "b", "y", "c", (0, 0.1, 0.8), (1, 0.5, 0), (1, 1, 0.3), "m", (0.4, 0.6, 0)]

for cnt, xx in enumerate(data):
    w = som.winner(xx)                        # 이 데이터의 BMU 격자 좌표
    plot(w[0] + .5, w[1] + .5,                # 칸 가운데에 표시
         markers[labels[cnt]], markerfacecolor='None',
         markeredgecolor=colors[labels[cnt]], markersize=12, markeredgewidth=2)
show()
```

책의 코드는 이 앞에 `labels[labels=='0'] = 0`처럼 문자열 레이블을 정수로 바꾸는 줄을 열 번 두지만, `load_digits`의 레이블은 처음부터 정수라 실제로는 아무 일도 하지 않는다.

결과 지도는 16x16 칸 위에 숫자별 마커가 흩어진 모습이다. 일부 겹치는 곳이 있지만 대부분의 숫자가 각자 별도의 영역을 차지하고, U-행렬의 밝은 경계선이 그 영역들을 나눈다. 레이블을 한 번도 보여 주지 않았는데도 64차원 픽셀 공간에서 비슷한 이미지끼리 지도 위에 모인 것이다.

## 정리

- 클러스터링은 레이블 없이 비슷한 데이터끼리 묶는 비지도 학습이며, 이 장은 K-평균, 가우시안 혼합 모델, 자기 조직화 지도를 다룬다.
- K-평균은 중심 배정과 중심 재계산을 반복해 WCSS(클러스터 내 거리 제곱합)를 줄이고, K는 WCSS 곡선이 꺾이는 엘보 지점으로 고른다.
- `kmeans_pytorch`로 붓꽃 데이터를 3개로 군집화하면 4번 만에 수렴하지만, 예제는 `Id`와 원-핫 품종 열까지 특성에 넣어 정답이 군집에 섞인다. 거리 기반 알고리즘에는 표준화 같은 특성 스케일링이 필수다.
- `torch.from_numpy()`는 넘파이 배열과 메모리를 공유하는 텐서를 만든다.
- 가우시안 혼합 모델은 데이터를 K개 정규분포의 가중합으로 보고, 베이즈 정리로 구한 책임값 $$\gamma(z_{nk})$$로 각 점을 확률적으로 배정한다. 공분산 덕분에 기울어진 타원 모양 클러스터도 표현한다.
- 자기 조직화 지도는 2차원 격자 뉴런이 경쟁(BMU 선택), 협력(이웃 결정), 적응(가중치를 입력 쪽으로 이동)을 반복하며 고차원 데이터를 지도처럼 펼친다. 손글씨 숫자를 16x16 격자에 펼치면 숫자별로 영역이 나뉜다.
- 전통적인 머신러닝 알고리즘은 PyTorch로 직접 구현하기 번거로우므로, 실무에서는 사이킷런 같은 전용 라이브러리를 쓰는 편이 낫다.

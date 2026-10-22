---
title: "Deep Learning & PyTorch 3: Core Machine Learning Algorithms"
date: 2026-10-22 17:03:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 3
card_title: "Chapter 3: Core Machine Learning Algorithms"
card_desc: "KNN · SVM and kernels · Decision tree, entropy, Gini · Logistic and linear regression · K-means and elbow · DBSCAN · PCA"
lang: ko
notes: "https://thebook.io/080289/0091/"
notes_label: "thebook.io chapter 3"
description: "지도 학습(K-최근접 이웃, 서포트 벡터 머신, 결정 트리, 로지스틱·선형 회귀)과 비지도 학습(K-평균 군집화, DBSCAN, PCA)의 핵심 원리를 정리하고 사이킷런 예제로 각 알고리즘을 실행해 본다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 3장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap03)</li>
</ul>

1장의 표에 있던 머신 러닝 알고리즘들을 하나씩 실행해 본다. 깊은 신경망이 필요 없는 알고리즘이므로 파이토치 대신 **사이킷런(scikit-learn)** 을 쓴다. 이 장의 목표는 각 알고리즘이 "무엇을 기준으로 판단하는지"를 잡는 것이다. 알고리즘마다 언제 쓰면 좋은지를 먼저 요약하고, 원리와 코드를 본다.

## 3.1 지도 학습

[📖 thebook.io](https://thebook.io/080289/0092/)

**지도 학습(supervised learning)** 은 입력과 정답(레이블)을 함께 주고 둘의 관계를 학습시키는 방법이다. 예측하려는 값의 종류에 따라 두 갈래로 나뉜다.

| 구분 | 분류(classification) | 회귀(regression) |
| --- | --- | --- |
| 예측 대상 | 이산형 값 (정해진 범주) | 연속형 값 (실수) |
| 출력 | 훈련 데이터에 있던 레이블 중 하나 | 어떤 값이든 나올 수 있음 |
| 예 | 메일이 스팸인지 아닌지 | 내일 주가, 최고 기온 |

평면에 그려 보면 분류는 점들을 나누는 **경계선**을 찾는 문제이고, 회귀는 점들의 추세를 따라가는 **선(또는 곡선)** 을 찾는 문제다.

### 3.1.1 K-최근접 이웃

[📖 thebook.io](https://thebook.io/080289/0093/)

> **언제 쓰나**: 분류. 원리가 직관적이고 구현이 쉬워 입문용으로 좋으며, 훈련 데이터를 충분히 모을 수 있을 때 잘 동작한다.

**K-최근접 이웃(K-Nearest Neighbor, KNN)** 은 새로운 데이터가 들어오면 기존 데이터 전부와의 거리를 재고, 가장 가까운 $$K$$개의 이웃 중 다수가 속한 클래스로 분류한다. 미리 모델을 만들어 두는 대신 훈련 데이터를 그대로 저장해 두었다가 예측할 때마다 비교하는 방식이라, "학습"이 사실상 데이터 저장에 가깝다(이런 방식을 게으른 학습(lazy learning)이라 부른다).

$$K=3$$일 때의 예를 들면 다음과 같다.

- 가장 가까운 세 점이 모두 주황이면 → 주황
- 주황 둘, 초록 하나면 → 주황 (다수결)
- 초록 둘, 주황 하나면 → 초록

같은 점이라도 $$K$$를 몇으로 두느냐에 따라 결과가 바뀔 수 있으므로 $$K$$는 중요한 하이퍼파라미터다. 너무 작으면 잡음에 민감하고, 너무 크면 멀리 있는 다른 클래스 점까지 끌어들여 경계가 뭉개진다.

#### 붓꽃 분류 예제

붓꽃(iris) 데이터는 꽃받침(sepal)과 꽃잎(petal)의 길이·너비 4개 특성으로 품종 3가지를 구분하는 150개 샘플이다.

```python
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv('../chap03/data/iris.data', names=names)   # 헤더가 없는 파일이라 이름 지정

X = dataset.iloc[:, :-1].values     # 마지막 열을 뺀 4개 특성
y = dataset.iloc[:, 4].values       # 5번째 열: 품종
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

s = StandardScaler()                # 평균 0, 표준편차 1로 맞춤
X_train = s.fit_transform(X_train)  # 훈련 데이터로 평균·표준편차를 구하고 변환
X_test = s.transform(X_test)        # 같은 기준으로 테스트 데이터 변환

knn = KNeighborsClassifier(n_neighbors=50)   # K = 50
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("정확도: {}".format(metrics.accuracy_score(y_test, y_pred)))
```

```text
정확도: 0.9333333333333333
```

KNN은 거리로 판단하므로 특성 간 단위(범위)가 다르면 범위가 큰 특성이 거리를 좌우한다. 그래서 거리 기반 알고리즘에서는 **스케일링**이 사실상 필수다. 스케일러는 훈련 데이터로만 `fit`하고 테스트 데이터에는 `transform`만 적용해야 테스트 정보가 학습에 새어 들어가지 않는다(공식 노트북은 테스트 데이터에도 `fit_transform`을 쓰고 있다).

`train_test_split`이 매번 무작위로 나누므로 실행할 때마다 정확도가 조금씩 다르다. 여러 번 돌려 평균을 보는 것이 좋다.

#### 최적의 K 찾기

$$K$$를 1부터 10까지 바꿔 가며 정확도를 비교한다.

```python
k = 10
acc_array = np.zeros(k)
for k in np.arange(1, k + 1, 1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc_array[k - 1] = metrics.accuracy_score(y_test, y_pred)

max_acc = np.amax(acc_array)
k = list(acc_array).index(max_acc)
print("정확도", max_acc, "으로 최적의 k는", k + 1, "입니다.")
```

```text
정확도 1.0 으로 최적의 k는 1 입니다.
```

$$K=50$$에서 93%였던 정확도가 $$K=1$$에서 100%가 되었다. 다만 테스트 샘플이 30개뿐이고 분할도 한 번뿐이라 우연의 영향이 크다. 실제로 $$K$$를 고를 때는 테스트 데이터가 아니라 교차 검증으로 정하는 것이 바른 절차다.

### 3.1.2 서포트 벡터 머신

[📖 thebook.io](https://thebook.io/080289/0099/)

> **언제 쓰나**: 분류. 커널을 잘 고르면 정확도가 높아서 정확도가 중요한 분류 문제, 특히 텍스트 분류에 많이 쓰인다.

**서포트 벡터 머신(Support Vector Machine, SVM)** 은 클래스를 나누는 **결정 경계(decision boundary)** 를 찾는 모델이다. 새 데이터는 경계의 어느 쪽에 있느냐로 분류된다.

두 클래스를 나누는 직선은 무수히 많다. 한쪽 클래스에 바짝 붙은 선은 새 데이터가 조금만 흔들려도 틀리기 쉽다. 그래서 SVM은 **양쪽 클래스에서 최대한 멀리 떨어진 경계**를 고른다. 이를 위해 두 가지 개념을 쓴다.

- **서포트 벡터(support vector)**: 경계에 가장 가까이 있는 데이터 점들. 경계의 위치는 사실상 이 점들만으로 결정된다.
- **마진(margin)**: 결정 경계와 서포트 벡터 사이의 거리. SVM은 이 마진을 최대화한다.

선형 SVM에서 경계가 $$w^\top x + b = 0$$이면, 양쪽 서포트 벡터를 지나는 평행선은 $$w^\top x + b = \pm 1$$이고 그 사이 폭은 $$\frac{2}{\lVert w \rVert}$$다. 따라서 마진 최대화는 모든 점을 올바르게 분류한다는 조건 아래 $$\lVert w \rVert$$를 최소화하는 문제가 된다.

$$
\min_{w,\, b} \ \frac{1}{2} \lVert w \rVert^2 \quad \text{subject to} \quad y_i (w^\top x_i + b) \ge 1
$$

현실 데이터에는 반대편 영역에 섞여 있는 **이상치(outlier)** 가 있다. 이를 어떻게 다루느냐에 따라 두 방식으로 나뉜다.

- **하드 마진(hard margin)**: 마진 안이나 반대편에 점이 하나도 없도록 강제한다. 이상치 하나에 경계가 크게 흔들리거나 해가 없을 수 있다.
- **소프트 마진(soft margin)**: 일부 점이 마진 안으로 들어오거나 잘못 분류되는 것을 벌점을 주고 허용한다.

#### 붓꽃 분류 예제

```python
from sklearn import svm, metrics, datasets, model_selection

iris = datasets.load_iris()                     # 사이킷런 내장 붓꽃 데이터
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.6, random_state=42)   # 60%를 테스트로

svm = svm.SVC(kernel='linear', C=1.0, gamma=0.5)
svm.fit(x_train, y_train)
predictions = svm.predict(x_test)
score = metrics.accuracy_score(y_test, predictions)
print('정확도: {0:f}'.format(score))
```

```text
정확도: 0.988889
```

책 코드는 텐서플로를 불러와 `TF_CPP_MIN_LOG_LEVEL`로 로그 수준을 조정하는 줄을 포함하지만(0: 모든 로그, 1: INFO 숨김, 2: WARNING까지 숨김, 3: ERROR까지 숨김), SVM 실행 자체에는 필요 없다.

#### 커널 트릭

직선 하나로 나눌 수 없는 데이터(예: 안쪽 원과 바깥 고리)도 있다. 이럴 때 데이터를 더 높은 차원으로 옮기면 평면 하나로 나눌 수 있게 되는 경우가 많다. 예를 들어 1차원에서 $$x$$를 $$(x, x^2)$$로 보내면 가운데 몰린 점과 양 끝 점을 직선으로 가를 수 있다. 하지만 실제로 고차원 좌표를 계산하면 비용이 크다.

**커널 트릭(kernel trick)** 은 고차원 좌표를 직접 만들지 않고, 고차원에서의 내적 값만 원래 공간에서 바로 계산하는 방법이다. SVM의 계산은 데이터 간 내적만 필요하므로 이것으로 충분하다. 사이킷런 `SVC`의 `kernel` 인자로 고른다.

| 커널 | 식 | 특징 |
| --- | --- | --- |
| 선형(`linear`) | $$K(a, b) = a^\top b$$ | 선형 분리 가능한 데이터용. 사실상 커널 트릭을 안 쓰는 것과 같다. |
| 다항식(`poly`) | $$K(a, b) = (\gamma\, a^\top b + r)^d$$ | 특성을 실제로 늘리지 않고도 $$d$$차 다항 특성 조합을 모두 추가한 효과 |
| 가우시안 RBF(`rbf`) | $$K(a, b) = \exp(-\gamma \lVert a - b \rVert^2)$$ | 무한 차원으로 매핑한 것과 같은 효과. 모든 차수의 다항식을 고려하는 셈 |

하이퍼파라미터 두 개의 의미는 다음과 같다.

- **`C`**: 오류를 얼마나 용납하지 않을지. 크면 오류에 큰 벌점을 주어 하드 마진에 가까워지고, 작으면 오류를 너그럽게 허용해 소프트 마진이 된다.
- **`gamma`** ($$\gamma$$): 데이터 한 점의 영향이 미치는 범위. 크면 각 점 주변만 보게 되어 경계가 구불구불해지고 과적합 위험이 커진다. 작으면 경계가 완만해진다. (선형 커널에서는 쓰이지 않는다.)

### 3.1.3 결정 트리

[📖 thebook.io](https://thebook.io/080289/0107/)

> **언제 쓰나**: 분류(회귀도 가능). 이상치가 많은 데이터에도 비교적 강하고, 판단 과정이 나무 그림으로 드러나므로 모델이 왜 그렇게 결정했는지 보여 주고 싶을 때 좋다.

**결정 트리(decision tree)** 는 "나이가 10세 이하인가?", "성별이 여성인가?"처럼 예/아니요 질문을 차례로 던져 데이터를 나눠 가는 모델이다. 결과가 뿌리에서 잎으로 뻗는 나무 모양이라 이런 이름이 붙었다.

나누는 기준은 **나눈 뒤 각 영역이 얼마나 한 종류로 깔끔하게 모이는가**다. 한 영역에 같은 클래스만 있으면 **순도(homogeneity)** 가 높고, 여러 클래스가 섞여 있으면 **불순도(impurity)** 가 높다. 트리는 매 분기에서 불순도(불확실성)가 가장 크게 줄어드는 질문을 고르며, 이 감소량을 **정보 획득(information gain)** 이라 한다. 불순도를 재는 지표는 두 가지가 많이 쓰인다.

#### 엔트로피

**엔트로피(entropy)** 는 확률 변수의 불확실성을 수치화한 것이다. 영역 $$A$$에 클래스가 $$m$$개 있고 클래스 $$k$$의 비율이 $$p_k$$라면

$$
\text{Entropy}(A) = -\sum_{k=1}^{m} p_k \log_2 p_k
$$

한 클래스만 있으면 0(완전히 순수), 이진 분류에서 반반 섞이면 1로 최대다. 예를 들어 어떤 결과가 확률 $$\frac{1}{4}$$, $$\frac{3}{4}$$로 나온다면

$$
-\frac{1}{4}\log_2\frac{1}{4} - \frac{3}{4}\log_2\frac{3}{4} = 0.5 + 0.311 \approx 0.81
$$

으로, 반반(1.0)보다는 덜 불확실하다.

#### 지니 계수

**지니 계수(Gini index)** 는 영역에서 무작위로 두 개를 뽑았을 때 서로 다른 클래스일 확률이다.

$$
G(A) = 1 - \sum_{k=1}^{m} p_k^2
$$

역시 순수하면 0이고, 섞일수록 커진다(이진 분류 최댓값 0.5). 위와 같은 $$\frac{1}{4}$$, $$\frac{3}{4}$$의 경우 $$1 - (\frac{1}{16} + \frac{9}{16}) = 0.375$$다. 로그 계산이 없어 엔트로피보다 빠르기 때문에 많은 구현(사이킷런 기본값 포함)이 지니 계수를 쓴다.

#### 타이타닉 생존 예측 예제

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('../chap03/data/titanic/train.csv', index_col='PassengerId')

# 생존 예측에 쓸 칼럼만 남김: 객실 등급, 성별, 나이, 동승 형제·배우자 수, 동승 부모·자녀 수, 요금
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})   # 문자열 → 정수
df = df.dropna()                                     # 결측치(주로 Age)가 있는 행 제거
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)  # 기본 25% 테스트

model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
accuracy_score(y_test, y_predict)
```

```text
0.8379888268156425
```

혼동 행렬로 결과를 나눠 본다.

```python
pd.DataFrame(confusion_matrix(y_test, y_predict),
             columns=['Predicted Not Survival', 'Predicted Survival'],
             index=['True Not Survival', 'True Survival'])
```

```text
                   Predicted Not Survival  Predicted Survival
True Not Survival                      99                  13
True Survival                          16                  51
```

혼동 행렬(confusion matrix)은 실제 값(행)과 예측 값(열)의 조합별 개수를 센 표다.

| | 예측 Positive | 예측 Negative |
| --- | --- | --- |
| **실제 Positive** | TP | FN |
| **실제 Negative** | FP | TN |

대각선(99, 51)이 맞힌 경우이고, 나머지(13, 16)가 틀린 경우다. 맞힌 경우가 훨씬 많아 $$\frac{99 + 51}{179} \approx 0.838$$로 위 정확도와 일치한다. 2장에서 본 정밀도와 재현율도 이 표에서 바로 계산할 수 있다(생존을 Positive로 보면 정밀도 $$51/64 \approx 0.80$$, 재현율 $$51/67 \approx 0.76$$).

`max_depth` 같은 제한 없이 학습한 결정 트리는 훈련 데이터에 과적합하기 쉽다. 결정 트리 여러 개를 서로 다른 데이터·특성 부분집합으로 학습시켜 투표하게 만든 것이 **랜덤 포레스트(random forest)** 로, 단일 트리의 불안정함을 크게 줄인다.

### 3.1.4 로지스틱 회귀와 선형 회귀

[📖 thebook.io](https://thebook.io/080289/0116/)

**회귀(regression)** 는 변수 사이의 관계를 찾아 한 변수로 다른 변수를 예측하는 방법이다.

- **독립 변수(independent variable, 예측 변수)**: 영향을 주는 쪽. 예: 키
- **종속 변수(dependent variable, 기준 변수)**: 영향을 받는 쪽. 예: 몸무게

어느 쪽이 원인이고 결과인지는 논리적으로 타당하게 설정해야 한다.

#### 로지스틱 회귀

> **언제 쓰나**: 분류. 결과를 확률로 내주므로 분류 결과에 대한 확신 정도가 필요할 때, 그리고 데이터를 계속 추가하며 다시 학습시킬 수 있는 환경에서 유용하다.

**로지스틱 회귀(logistic regression)** 는 이름에 "회귀"가 들어가지만 **분류** 기법이다. 각 관측치가 어느 집단에 속할지를 확률로 예측한다. 선형식 $$z = w^\top x + b$$를 **시그모이드(sigmoid)** 함수에 넣어 0과 1 사이 확률로 바꾼다.

$$
P(Y=1 \mid x) = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

일반 회귀와 비교하면 다음과 같다.

| 구분 | 일반(선형) 회귀 | 로지스틱 회귀 |
| --- | --- | --- |
| 종속 변수 | 연속형 | 이산형(범주) |
| 파라미터 추정 | 최소제곱법 | 최대우도법 |
| 모형 검정 | F-검정, t-검정 | 카이제곱($$\chi^2$$) 검정 |

**최소제곱법과 최대우도법.** 둘 다 표본으로부터 모델 파라미터를 추정하는 방법이다.

- **최소제곱법(least squares)**: 실제 값과 예측 값의 차이를 제곱해 더한 값을 가장 작게 만드는 파라미터를 찾는다. 1장의 평균 제곱 오차를 최소화하는 것과 같다.

$$
\hat{\theta} = \arg\min_{\theta} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

- **최대우도법(maximum likelihood estimation)**: **우도(likelihood, 가능도)** 는 관측된 결과가 주어졌을 때 각 가설(파라미터 값)이 그 결과를 얼마나 그럴듯하게 설명하는지를 나타내는 값이다. 최대우도법은 관측된 데이터가 나올 확률을 가장 크게 만드는 파라미터를 고른다.

$$
\hat{\theta} = \arg\max_{\theta} P(Y \mid X; \theta) = \arg\max_{\theta} \prod_{i=1}^{m} P(y_i \mid x_i; \theta)
$$

관측치들이 서로 독립이라고 가정하면 우도는 확률의 곱이다. 작은 확률을 수없이 곱하면 컴퓨터가 표현할 수 있는 범위보다 작아지는 **언더플로(underflow)** 가 생기므로, 로그를 취해 곱을 합으로 바꾼다.

$$
\hat{\theta} = \arg\max_{\theta} \sum_{i=1}^{m} \log P(y_i \mid x_i; \theta)
$$

이 로그 우도에 음수를 붙인 것이 분류에서 쓰는 크로스 엔트로피 손실이다.

로지스틱 회귀의 분류 절차는 두 단계다.

1. 집단 1에 속할 확률 $$P(Y=1)$$을 추정한다.
2. 기준값(cut-off)으로 나눈다. 예: $$P(Y=1) \ge 0.5$$이면 집단 1, 아니면 집단 0.

#### 손글씨 숫자 분류 예제

사이킷런의 `digits` 데이터는 8×8 흑백 손글씨 숫자 이미지 1,797장이다. 각 이미지를 64개 픽셀 값으로 펼친 벡터가 입력이다.

```python
from sklearn.datasets import load_digits
digits = load_digits()
print("Image Data Shape", digits.data.shape)     # (1797, 64)
print("Label Data Shape", digits.target.shape)   # (1797,)
```

```text
Image Data Shape (1797, 64)
Label Data Shape (1797,)
```

```python
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (8, 8)), cmap=plt.cm.gray)   # 64 → 8×8로 되돌려 표시
    plt.title('Training: %i\n' % label, fontsize=20)
```

앞의 다섯 장은 흐릿한 0, 1, 2, 3, 4다. 학습과 예측은 다음과 같다.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

logisticRegr.predict(x_test[0].reshape(1, -1))   # 샘플 하나: (64,) → (1, 64)로 모양을 맞춤
logisticRegr.predict(x_test[0:10])               # 열 개를 한 번에
```

```text
array([2, 8, 2, 6, 6, 7, 1, 9, 8, 5])
```

사이킷런 모델은 항상 2차원 입력(샘플 수, 특성 수)을 기대하므로 샘플 하나를 넣을 때도 `reshape(1, -1)`로 차원을 맞춘다. 학습 시 반복 횟수 한도에 걸렸다는 `ConvergenceWarning`이 뜰 수 있는데, `max_iter`를 늘리거나 입력을 스케일링하면 사라진다. 클래스가 10개이므로 사이킷런은 소프트맥스(다항) 로지스틱 회귀로 자동 확장한다.

```python
predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)    # 분류 모델의 score는 정확도
print(score)
```

```text
0.9511111111111111
```

혼동 행렬을 seaborn 히트맵으로 그리면 클래스별 오류를 볼 수 있다.

```python
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Accuracy Score: {0}'.format(score), size=15)
plt.show()
```

10×10 행렬의 대각선에 대부분의 값이 몰려 있고, 비대각선 칸에 드문드문 있는 숫자가 혼동한 경우(예: 실제 8을 1로 예측)다. 클래스가 많아지면 표가 커져 한눈에 읽기 어려우므로, 혼동 행렬은 어떤 클래스끼리 헷갈리는지 확인하는 보조 도구로 쓰는 편이 좋다.

#### 선형 회귀

> **언제 쓰나**: 회귀. 독립 변수와 종속 변수가 선형 관계일 때 유용하고, 계산이 가벼워 컴퓨팅 자원이 부족한 환경에서도 쓸 수 있다.

**선형 회귀(linear regression)** 는 독립 변수 $$x$$로 종속 변수 $$y$$를 직선(또는 평면)으로 설명한다. 독립 변수가 하나면 **단순 선형 회귀(simple linear regression)**, 여러 개면 **다중 선형 회귀(multiple linear regression)** 다.

$$
\hat{y} = w x + b \qquad \text{또는} \qquad \hat{y} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b
$$

예를 들어 여름에 아이스크림이 시간당 100개 팔린다면 판매량을 $$y = 100x$$로 모델링하고, 이를 이용해 몇 시간 뒤의 판매량이나 매출을 추정할 수 있다.

로지스틱 회귀와의 차이를 정리하면, 선형 회귀는 값 자체(판매량, 기온)를 예측하고 출력이 **직선**이다. 로지스틱 회귀는 사건이 일어날 확률(구매할지 말지)을 예측하고 출력이 0과 1 사이를 잇는 **S자 곡선**이다.

#### 날씨 데이터 예제

최저 기온(`MinTemp`)으로 최고 기온(`MaxTemp`)을 예측한다.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('../chap03/data/weather.csv')

dataset.plot(x='MinTemp', y='MaxTemp', style='o')   # 두 변수의 산점도
plt.title('MinTemp vs MaxTemp')
plt.show()
```

산점도는 오른쪽 위로 올라가는 띠 모양이라 선형 관계가 있어 보인다.

```python
X = dataset['MinTemp'].values.reshape(-1, 1)   # 독립 변수: (N, 1) 2차원으로
y = dataset['MaxTemp'].values.reshape(-1, 1)   # 종속 변수
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression()
regressor.fit(X_train, y_train)                # 최소제곱법으로 w, b 추정

y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
```

```text
    Actual  Predicted
0     25.2  23.413030
1     11.5  13.086857
2     21.1  27.264856
3     22.2  25.461874
...
```

테스트 데이터 위에 회귀선을 겹쳐 그린다.

```python
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
```

빨간 직선이 회색 점들의 중심을 따라 지나간다. 회귀 모델은 **평균 제곱 오차(MSE)** 와 그 제곱근인 **루트 평균 제곱 오차(Root Mean Squared Error, RMSE)** 로 평가한다.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \qquad \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

```python
print('평균제곱법:', metrics.mean_squared_error(y_test, y_pred))
print('루트 평균제곱법:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
```

```text
평균제곱법: 17.011877668640622
루트 평균제곱법: 4.124545753006096
```

RMSE는 원래 데이터와 같은 단위(°C)라 해석이 쉽다. 예측이 평균적으로 4.1°C 정도 빗나간다는 뜻이고, 최고 기온 평균(약 22.4°C)의 18% 정도다. 최저 기온 하나만으로 예측한 것치고는 합리적인 수준이다.

## 3.2 비지도 학습

[📖 thebook.io](https://thebook.io/080289/0132/)

**비지도 학습(unsupervised learning)** 은 레이블 없이 데이터 자체의 구조를 찾는다. 대표적인 두 과제는 다음과 같다.

| 구분 | 군집(clustering) | 차원 축소(dimensionality reduction) |
| --- | --- | --- |
| 목표 | 비슷한 데이터끼리 그룹으로 묶기 | 특성 수를 줄여 데이터를 간소화 |
| 주요 알고리즘 | K-평균 군집화 | 주성분 분석(PCA) |
| 예 | 관심사가 비슷한 고객끼리 묶어 마케팅 | 데이터 압축, 중요한 속성 도출, 시각화 |

군집은 데이터 간 유사도(거리)를 재어 가까운 것끼리 묶는다. 통계학의 "군집", 머신 러닝의 "클러스터", "군집화"는 모두 같은 뜻으로 쓰인다. 거리 척도로는 **유클리드 거리**($$\sqrt{\sum_j (a_j - b_j)^2}$$), **맨해튼 거리**($$\sum_j \lvert a_j - b_j \rvert$$), 이 둘을 일반화한 **민코프스키 거리**, 벡터 방향의 유사성을 보는 **코사인 유사도** 등이 있다.

### 3.2.1 K-평균 군집화

[📖 thebook.io](https://thebook.io/080289/0133/)

> **언제 쓰나**: 군집. 데이터를 몇 개의 그룹으로 나눌지 미리 알고 있을 때 유용하다.

**K-평균 군집화(K-means clustering)** 는 레이블 없는 데이터를 $$K$$개의 그룹으로 나누고 각 데이터에 그룹 번호를 붙인다. 절차는 다음과 같다.

1. **중심점 선택**: $$K$$개의 초기 **중심점(centroid)** 을 무작위로 고른다.
2. **클러스터 할당**: 각 데이터를 가장 가까운 중심점의 클러스터에 배정한다.
3. **중심점 갱신**: 클러스터마다 속한 데이터의 평균 위치로 중심점을 옮긴다.
4. **수렴 확인(convergence)**: 중심점이 더 이상 움직이지 않으면 멈추고, 아니면 2~3을 반복한다.

반복할수록 중심점이 데이터 덩어리의 가운데로 이동하며 경계가 안정된다. 이 과정은 각 점에서 소속 중심점까지 거리 제곱의 합을 줄여 나가는 것과 같다.

K-평균은 클러스터를 "중심점 주변의 둥근 덩어리"로 가정하기 때문에 다음 경우에는 원하는 결과가 나오지 않는다.

- 데이터가 **비선형** 구조일 때(예: 초승달 두 개가 맞물린 모양, 동심원)
- 군집마다 **크기**(점 개수, 퍼진 정도)가 크게 다를 때
- 군집마다 **밀집도**와 서로 간 **거리**가 다를 때

#### 도매 고객 데이터 예제

도매 유통업체 고객 440명의 품목별 연간 지출 데이터로 적절한 $$K$$를 찾는다.

| 칼럼 | 의미 | 유형 |
| --- | --- | --- |
| `Channel` | 고객 채널(호텔·레스토랑·카페 / 소매) | 명목형 |
| `Region` | 고객 지역 | 명목형 |
| `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen` | 신선 식품, 유제품, 식료품, 냉동식품, 세제·종이, 조제 식품 연간 지출 | 연속형 |

자료 유형을 정리하면 다음과 같다.

| 유형 | 설명 | 예 |
| --- | --- | --- |
| 수치형 – 연속형 | 값이 연속적 | 키, 몸무게 |
| 수치형 – 이산형 | 셀 수 있는 값 | 자동차 사고 건수 |
| 범주형 – 순위형 | 범주 사이에 순서가 있음 | 매우 좋다 / 좋다 / 보통 / 싫다 |
| 범주형 – 명목형 | 범주 사이에 순서가 없음 | 혈액형 |

명목형 칼럼은 원-핫 인코딩으로 바꾼다.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('../chap03/data/sales data.csv')

categorical_features = ['Channel', 'Region']
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)   # Channel_1, Channel_2, Region_1 ...
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)              # 원래 칼럼은 제거
```

지출 금액은 칼럼마다 범위가 크게 다르다. 그대로 두면 큰 값을 갖는 칼럼이 거리 계산을 지배하므로(1천 원 차이는 1억 원 차이에 묻힌다) 모든 칼럼을 0~1 범위로 맞춘다.

```python
mms = MinMaxScaler()                 # (x - min) / (max - min)
mms.fit(data)
data_transformed = mms.transform(data)
```

#### 엘보 방법으로 K 고르기

$$K$$를 1부터 14까지 바꾸며 학습하고, 각 점에서 소속 중심점까지 거리 제곱의 합(**SSD**, Sum of Squared Distances)을 기록한다. 사이킷런에서는 `inertia_` 속성이다.

$$
\text{SSD} = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2
$$

여기서 $$\mu_{c(i)}$$는 점 $$x_i$$가 속한 클러스터의 중심점이다.

```python
Sum_of_squared_distances = []
K = range(1, 15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal k')
plt.show()
```

$$K$$가 커질수록 SSD는 계속 줄어든다. 극단적으로 $$K$$를 샘플 수와 같게 하면 모든 점이 자기 자신이 중심점이 되어 SSD가 0이 된다. 그래서 SSD가 가장 작은 $$K$$가 아니라, **감소폭이 급격히 줄어드는 꺾이는 지점(팔꿈치, elbow)** 을 고른다. 이 예제의 그래프에서는 $$K=5$$ 이후로 곡선이 거의 평평해져, 책은 $$K=5$$를 적정값으로 본다.

### 3.2.2 밀도 기반 군집 분석

[📖 thebook.io](https://thebook.io/080289/0142/)

> **언제 쓰나**: 군집. 클러스터 수를 미리 모를 때, 그리고 데이터에 이상치가 많을 때 유용하다.

**밀도 기반 군집 분석(Density-Based Spatial Clustering of Applications with Noise, DBSCAN)** 은 점들이 빽빽하게 모인 영역을 하나의 군집으로 본다. 밀도가 낮은 곳의 점은 어느 군집에도 넣지 않고 **노이즈**로 남긴다. K-평균보다 계산량은 많지만, 군집 수를 지정할 필요가 없고 초승달이나 고리처럼 오목·볼록한 모양의 군집도 찾아낼 수 있다.

노이즈와 이상치는 다른 개념이다. **노이즈(noise)** 는 데이터셋과 무관하거나 무작위로 섞인 값으로 전처리에서 걸러 낼 대상이고, **이상치(outlier)** 는 관측 범위에서 크게 벗어난 아주 작거나 큰 값이다.

DBSCAN은 두 하이퍼파라미터를 쓴다.

- **엡실론(epsilon, `eps`)**: 이웃으로 인정할 반경.
- **minPts(`min_samples`)**: 중심점이 되기 위해 반경 안에 있어야 할 최소 점 개수.

절차는 다음과 같다.

1. **중심점 결정**: 어떤 점 $$P_1$$의 반경 $$\epsilon$$ 안에 점이 minPts개 이상 있으면 $$P_1$$을 **중심점(core point)** 으로 삼고 군집을 만든다. 예를 들어 minPts가 3이고 반경 안에 점이 4개 있으면 군집이 된다.
2. **군집 확장**: 그 군집 안의 다른 점 $$P_2$$도 같은 조건을 만족하면 $$P_2$$의 이웃까지 같은 군집으로 합친다. 이렇게 밀도가 이어지는 동안 군집이 퍼져 나간다.
3. **반복**: 더 이상 새 중심점을 찾을 수 없을 때까지 1~2를 반복한다.
4. **노이즈 정의**: 끝까지 어느 군집에도 들지 못한 점은 노이즈로 표시한다(사이킷런에서는 레이블 `-1`).

### 3.2.3 주성분 분석(PCA)

[📖 thebook.io](https://thebook.io/080289/0147/)

> **언제 쓰나**: 데이터 간소화. 특성이 너무 많아 그래프 하나에 그릴 수 없을 때, 특성 $$p$$개를 2~3개로 압축해 시각화하고 싶을 때 유용하다.

특성이 많은 고차원 데이터는 중요하지 않은 특성까지 처리하느라 계산이 무겁고 성능도 떨어지기 쉽다. 데이터를 대표하는 방향 몇 개만 남겨 저차원으로 줄이면 계산이 가벼워지고 다루기도 편해진다. 이를 위한 대표 알고리즘이 **주성분 분석(Principal Component Analysis, PCA)** 이다.

PCA는 데이터 한 점 한 점이 아니라 **데이터 전체가 이루는 분포**를 본다. 2차원에 비스듬한 타원 모양으로 퍼진 데이터라면, 타원의 긴 축 방향 $$e_1$$과 그에 수직인 짧은 축 방향 $$e_2$$ 두 벡터만 알면 분포의 모양을 설명할 수 있다. 이 방향들이 **주성분(principal component)** 이다.

- 첫 번째 주성분은 데이터를 그 방향으로 투영했을 때 **분산이 가장 큰** 방향이다.
- 두 번째 주성분은 첫 번째와 직교하면서 남은 분산이 가장 큰 방향이다.

수학적으로는 데이터 공분산 행렬의 고유벡터가 주성분이고, 고윳값이 그 방향의 분산이다. 분산이 작은 방향을 버리고 큰 방향 몇 개로만 투영하면 정보 손실을 최소화하면서 차원을 줄일 수 있다.

```python
from sklearn import decomposition
pca = decomposition.PCA(n_components=1)   # 1차원으로 축소
pca_x = pca.fit_transform(x_std)          # 표준화된 데이터에 적용
```

#### 신용카드 고객 데이터: DBSCAN + PCA

신용카드 고객 약 9천 명의 사용 패턴(잔액, 구매 금액, 현금 서비스, 한도, 결제액 등 17개 특성)을 DBSCAN으로 군집화하고, 결과를 눈으로 보기 위해 PCA로 2차원으로 줄인다.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

X = pd.read_csv('../chap03/data/credit card.csv')
X = X.drop('CUST_ID', axis=1)             # 고객 ID는 특성이 아니므로 제거
X.fillna(method='ffill', inplace=True)    # 결측치를 바로 위 행의 값으로 채움
```

`fillna(method='ffill')`은 결측값을 직전 값으로 채운다(forward fill). 최신 판다스에서는 `X.ffill(inplace=True)`로 쓴다.

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # 특성별 평균 0, 표준편차 1
X_normalized = normalize(X_scaled)        # 각 행(샘플) 벡터의 길이를 1로
X_normalized = pd.DataFrame(X_normalized)

pca = PCA(n_components=2)                 # 17차원 → 2차원
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']
print(X_principal.head())
```

```text
         P1        P2
0 -0.489949 -0.679976
1 -0.519099  0.544827
2  0.330633  0.268880
3 -0.481656 -0.097611
4 -0.563512 -0.482506
```

`StandardScaler`는 **칼럼(특성)** 단위로 표준화하고, `normalize`는 **행(샘플)** 단위로 벡터 길이를 1로 맞춘다. 행 정규화 후에는 모든 점이 단위 구 위에 놓이므로 거리 비교가 방향 위주로 바뀐다.

이제 2차원 데이터에 DBSCAN을 적용한다.

```python
db_default = DBSCAN(eps=0.0375, min_samples=3).fit(X_principal)
labels = db_default.labels_                # 각 점의 군집 번호, 노이즈는 -1

colours = {0: 'y', 1: 'g', 2: 'b', -1: 'k'}
cvec = [colours[label] for label in labels]    # 점마다 색 지정

plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.show()
```

책 코드는 범례를 만들기 위해 같은 데이터를 색만 바꿔 여러 번 `scatter`한 뒤 그 핸들을 `plt.legend()`에 넘기는데, 핵심은 `labels_`로 색을 정하는 위 부분이다. `min_samples=3`에서는 군집이 제대로 나뉘지 않아 대부분이 한 색으로 뭉쳐 보인다. 기준이 너무 느슨하면 밀도가 이어진 거의 모든 점이 한 군집으로 연결되기 때문이다.

`min_samples`를 50으로 올린다.

```python
db = DBSCAN(eps=0.0375, min_samples=50).fit(X_principal)
labels1 = db.labels_

colours1 = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'y', 5: 'm', -1: 'k'}
cvec = [colours1[label] for label in labels1]

plt.figure(figsize=(9, 9))
plt.scatter(X_principal['P1'], X_principal['P2'], c=cvec)
plt.show()
```

이번에는 여러 개(레이블 0~5)의 군집이 색으로 구분되고, 밀도가 낮은 가장자리 점들은 노이즈(검정)로 빠진다. 앞의 결과보다 군집 구조가 훨씬 잘 드러난다.

반대로 `min_samples=100`까지 올리면 기준이 너무 엄격해져 상당수의 작은 군집이 군집으로 인정받지 못하고 노이즈로 처리된다. 즉 같은 알고리즘이라도 하이퍼파라미터에 따라 결과가 크게 달라지므로 튜닝이 중요하다(성능 최적화는 8장에서 다룬다).

matplotlib 그림의 구성 요소 이름도 알아 두면 코드를 읽기 쉽다. 전체 캔버스가 Figure, 그 안의 그래프 영역이 Axes이고, Axes는 x축·y축(Axis), 눈금(tick), 눈금 라벨, 축 제목(label), 제목(title), 범례(legend), 격자(grid) 등으로 이루어진다.

## 정리

- 지도 학습은 이산값을 예측하는 분류와 연속값을 예측하는 회귀로 나뉘고, 비지도 학습은 그룹을 찾는 군집과 특성을 줄이는 차원 축소로 나뉜다.
- KNN은 가장 가까운 $$K$$개 이웃의 다수결로 분류하며, 거리 기반이므로 스케일링이 필요하고 $$K$$ 선택이 성능을 좌우한다.
- SVM은 서포트 벡터와의 마진을 최대화하는 결정 경계를 찾는다. `C`는 오류 허용 정도(하드/소프트 마진), `gamma`는 경계의 유연성을 정하고, 커널 트릭(linear, poly, rbf)으로 비선형 문제를 푼다.
- 결정 트리는 엔트로피 $$-\sum p_k \log_2 p_k$$나 지니 계수 $$1 - \sum p_k^2$$가 가장 크게 줄어드는 질문으로 데이터를 나누며, 여러 트리를 묶으면 랜덤 포레스트가 된다.
- 로지스틱 회귀는 시그모이드로 확률을 내는 분류 모델이고 최대우도법으로 학습한다. 선형 회귀는 최소제곱법으로 직선을 맞추고 MSE, RMSE로 평가한다.
- K-평균은 중심점 할당과 갱신을 반복하며, SSD(`inertia_`) 곡선의 꺾이는 지점(엘보)으로 $$K$$를 고른다. 둥글지 않거나 크기·밀도가 다른 군집에는 약하다.
- DBSCAN은 `eps`와 `min_samples`로 정의한 밀도로 군집을 확장하고 나머지를 노이즈(-1)로 둔다. 군집 수를 몰라도 되고 모양이 복잡한 군집도 찾는다.
- PCA는 분산이 가장 큰 직교 방향(주성분)으로 데이터를 투영해 차원을 줄이며, 고차원 군집 결과를 2차원으로 시각화할 때 유용하다.

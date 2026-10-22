---
title: "Deep Learning & PyTorch 10: Embeddings for NLP"
date: 2026-10-22 17:10:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 10
card_title: "Chapter 10: Embeddings for NLP"
card_desc: "One-hot, count vectors and TF-IDF · Word2Vec (CBOW, skip-gram) · FastText and GloVe · Attention and seq2seq · BERT fine-tuning · Korean BERT embeddings"
lang: ko
notes: "https://thebook.io/080289/0703/"
notes_label: "thebook.io chapter 10"
description: "단어를 벡터로 바꾸는 희소 표현, 횟수 기반, 예측 기반, 횟수/예측 결합 임베딩을 차례로 살펴보고, 어텐션과 seq2seq 번역 모델, 버트 미세 조정, 다국어 버트로 만든 한국어 문맥 임베딩까지 정리한다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 10장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap10)</li>
</ul>

9장에서 텍스트를 토큰으로 나누고 정제하는 전처리를 다뤘다면, 10장은 그 토큰을 신경망이 계산할 수 있는 숫자 벡터로 바꾸는 방법을 다룬다. 단순한 원-핫 벡터에서 출발해 빈도 통계, 예측 신경망, 그리고 문맥에 따라 벡터가 달라지는 트랜스포머 계열 모델까지 순서대로 올라간다.

## 10.1 임베딩

[📖 thebook.io](https://thebook.io/080289/0704/)

**임베딩(embedding)** 은 자연어(단어, 문장)를 컴퓨터가 계산할 수 있는 실수 벡터로 옮기는 일, 또는 그렇게 얻은 벡터 자체를 말한다. 좋은 임베딩은 두 가지를 해낸다.

- 벡터끼리의 거리나 각도로 단어, 문장 사이의 관련성을 계산할 수 있다.
- 의미나 문법 관계가 벡터 공간의 방향으로 담긴다. 예를 들어 "왕 - 남자 + 여자 ≈ 여왕" 같은 관계가 산술로 드러난다.

책은 임베딩을 만드는 방식에 따라 네 갈래로 나눈다.

| 분류 | 아이디어 | 대표 기법 |
| --- | --- | --- |
| 희소 표현 기반 | 단어마다 자리 하나를 배정하고 그 자리만 1 | 원-핫 인코딩 |
| 횟수 기반 | 단어가 문서에 몇 번 나오는지 센다 | 카운터 벡터, TF-IDF |
| 예측 기반 | 주변 단어를 예측하는 신경망을 학습하고 그 가중치를 벡터로 쓴다 | Word2Vec, FastText |
| 횟수/예측 결합 | 전역 동시 출현 통계를 예측 방식의 학습 목표로 쓴다 | GloVe |

### 10.1.1 희소 표현 기반 임베딩

**희소 표현(sparse representation)** 은 벡터의 원소 대부분이 0인 표현이다. 대표가 **원-핫 인코딩(one-hot encoding)** 으로, 어휘가 N개면 길이 N인 벡터에서 해당 단어의 위치만 1로 둔다. 어휘가 `[calm, fast, cat]`이면 `fast`는 `[0, 1, 0]`이다.

예제는 9장의 `class2.csv`(학생 등급 같은 범주 열)에 사이킷런의 `LabelEncoder`를 적용한다. `LabelEncoder`는 범주 문자열을 정수 번호로 바꾸고, 이 번호를 `OneHotEncoder`에 넘기면 원-핫 행렬이 된다.

```python
import pandas as pd
from sklearn import preprocessing

class2 = pd.read_csv("../chap10/data/class2.csv")

label_encoder = preprocessing.LabelEncoder()    # 범주 -> 정수 번호
onehot_encoder = preprocessing.OneHotEncoder()  # 정수 번호 -> 원-핫 벡터

train_x = label_encoder.fit_transform(class2['class2'])
train_x
```

```text
array([2, 2, 1, 0, 1, 0])
```

여섯 개 행의 범주가 0, 1, 2 세 개의 번호로 바뀌었다. 원-핫 표현에는 큰 약점이 두 가지 있다.

1. **관계를 담지 못한다.** 서로 다른 두 원-핫 벡터의 내적은 항상 0이다. 즉 모든 단어 쌍이 직교(orthogonal)하므로, "고양이"와 "강아지"가 "고양이"와 "자동차"보다 가깝다는 정보가 전혀 없다.
2. **차원이 어휘 크기만큼 커진다.** 어휘가 10만 개면 단어 하나가 10만 차원 벡터가 되고, 그중 하나만 1이다. 이른바 **차원의 저주(curse of dimensionality)** 다.

이 한계 때문에 신경망으로 짧고 조밀한(dense) 벡터를 학습하는 Word2Vec, GloVe, FastText 같은 방법이 쓰인다.

### 10.1.2 횟수 기반 임베딩

단어가 등장한 횟수를 그대로, 혹은 가중치를 붙여 벡터 값으로 쓰는 방식이다.

**카운터 벡터**

**카운터 벡터(counter vector)** 는 말뭉치의 모든 단어로 어휘 사전을 만든 뒤, 각 문서를 "사전의 각 단어가 이 문서에 몇 번 나왔는가"로 표현한다. 사이킷런의 `CountVectorizer`가 토큰화, 빈도 세기, 벡터화를 한 번에 해 준다.

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is last chance.',
    'and if you do not have this chance.',
    'you will never get any chance.',
    'will you do get this one?',
    'please, get this chance',
]
vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_   # 단어 -> 열 번호 (알파벳 순)
```

```text
{'this': 13, 'is': 7, 'last': 8, 'chance': 2, 'and': 0, 'if': 6, 'you': 15, 'do': 3,
 'not': 10, 'have': 5, 'will': 14, 'never': 9, 'get': 4, 'any': 1, 'one': 11, 'please': 12}
```

사전이 만들어지면 새 문장을 길이 16의 빈도 벡터로 바꿀 수 있다.

```python
vect.transform(['you will never get any chance.']).toarray()
```

```text
array([[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]], dtype=int64)
```

1인 위치는 1(`any`), 2(`chance`), 4(`get`), 9(`never`), 14(`will`), 15(`you`)다. `stop_words` 인자로 분석에 쓸모없는 단어를 빼면 사전이 작아진다.

```python
vect = CountVectorizer(stop_words=["and", "is", "please", "this"]).fit(corpus)
vect.vocabulary_
```

```text
{'last': 6, 'chance': 1, 'if': 5, 'you': 11, 'do': 2, 'not': 8, 'have': 4,
 'will': 10, 'never': 7, 'get': 3, 'any': 0, 'one': 9}
```

**TF-IDF**

빈도만 세면 "the", "a"처럼 어디에나 나오는 단어가 큰 값을 갖는다. **TF-IDF(Term Frequency-Inverse Document Frequency)** 는 정보 검색(information retrieval) 분야에서 쓰는 가중치로, "이 문서에는 자주 나오지만 다른 문서에는 드문 단어"에 높은 점수를 준다.

- **TF(단어 빈도)**: 문서 $$d$$ 안에서 단어 $$t$$가 나온 횟수 $$tf_{t,d}$$. 그대로 쓰거나 $$\log(1 + tf_{t,d})$$처럼 완만하게 만들어 쓴다.
- **DF(문서 빈도)**: 단어 $$t$$가 들어 있는 문서의 개수 $$df_t$$. 흔한 단어일수록 크다.
- **IDF(역문서 빈도)**: DF가 클수록 작아지도록 역수를 취하고, 전체 문서 수 $$N$$이 커질 때 값이 폭주하지 않도록 로그를 씌운다. 분모가 0이 되는 것을 막으려고 1을 더하는 것을 **스무딩(smoothing)** 이라 한다.

$$
idf_t = \log\frac{N}{1 + df_t}, \qquad \text{tf-idf}(t, d) = tf_{t,d} \times idf_t
$$

TF-IDF는 키워드 기반 검색 엔진, 문서의 핵심 키워드 추출, 검색 결과 순위 결정 등에 쓰인다. 예제는 세 문장을 TF-IDF 행렬로 바꾼 뒤, 행렬을 자기 전치와 곱해 문서 간 유사도 행렬을 만든다(`TfidfVectorizer`는 각 행을 L2 정규화하므로 이 곱이 곧 코사인 유사도다).

```python
from sklearn.feature_extraction.text import TfidfVectorizer

doc = ['I like machine learning', 'I love deep learning', 'I run everyday']
tfidf_vectorizer = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorizer.fit_transform(doc)    # (3, 어휘 크기) 희소 행렬

doc_distance = (tfidf_matrix * tfidf_matrix.T)        # 문서 x 문서 유사도
print(doc_distance.toarray())
```

```text
[[1.       0.224325 0.      ]
 [0.224325 1.       0.      ]
 [0.       0.       1.      ]]
```

1번과 2번 문서는 `learning`을 공유해 0.22의 유사도를 갖고, 3번 문서는 공유 단어가 없어 0이다(`I` 같은 한 글자 토큰은 기본 토크나이저가 무시한다). 대각선은 자기 자신과의 유사도이므로 1이다.

### 10.1.3 예측 기반 임베딩

예측 기반 임베딩은 "어떤 문맥에서 어떤 단어가 나올지"를 맞히는 신경망을 학습시키고, 그 과정에서 얻은 가중치를 단어 벡터로 쓴다. 비슷한 문맥에 나타나는 단어는 비슷한 벡터를 갖게 된다는 분포 가설이 바탕이다.

**Word2Vec**

**Word2Vec(워드투벡터)** 은 텍스트의 각 단어에 대해 고정 길이 벡터를 하나씩 학습하는 신경망 모델이다. 학습된 벡터 공간에서 의미가 비슷한 단어는 **코사인 유사도(cosine similarity)** 가 높게, 즉 서로 가까이 놓인다. 그래서 어떤 단어의 동의어나 관련어를 찾는 데 쓸 수 있다.

학습 데이터는 문장 위로 일정 크기의 **윈도우(window)** 를 밀면서 만든다. 윈도우 가운데 단어를 대상 단어(target), 양옆 단어들을 문맥(context)으로 묶어 한 쌍씩 신경망에 넣는다. 예를 들어 "and I love this one ..."에서 윈도우를 `one`에 맞추면 `and`, `I`, `love`, `this`가 문맥이 된다. 은닉층의 가중치가 바로 단어별 임베딩이다.

예제는 『피터팬』 텍스트(`peter.txt`)를 문장 단위, 단어 단위로 토큰화하고 소문자로 바꿔 2차원 리스트를 만든다.

```python
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
from gensim.models import Word2Vec

sample = open("../chap10/data/peter.txt", "r", encoding='UTF8')
f = sample.read().replace("\n", " ")   # 줄바꿈을 공백으로

data = []
for sent in sent_tokenize(f):          # 문장 단위로 나눈 뒤
    data.append([w.lower() for w in word_tokenize(sent)])  # 단어 단위 토큰, 소문자화
```

```text
[['once', 'upon', 'a', 'time', 'in', 'london', ',', 'the', 'darlings', 'went', 'out', ...], ...]
```

Word2Vec에는 학습 목표에 따라 두 가지 구조가 있다.

**CBOW**

**CBOW(Continuous Bag Of Words)** 는 주변 단어들로 가운데 단어를 맞힌다. "calm cat ___ on the sofa"가 주어지면 빈칸의 `slept`를 예측하는 식이다.

구조는 입력층, 은닉층, 출력층의 얕은 신경망이다. 어휘 크기를 $$V$$, 원하는 임베딩 차원을 $$N$$이라 하면 다음과 같다.

- 입력: 문맥 단어 각각의 원-핫 벡터(길이 $$V$$).
- 입력층 → 은닉층 가중치 $$W$$: $$V \times N$$ 행렬. 원-핫 벡터에 곱하면 $$W$$의 한 행이 뽑히므로, $$W$$의 각 행이 그 단어의 임베딩이다. 모든 문맥 단어가 같은 $$W$$를 공유한다.
- 은닉층: 문맥 단어 벡터들의 평균(길이 $$N$$).
- 은닉층 → 출력층 가중치 $$W'$$: $$N \times V$$ 행렬. 결과에 소프트맥스를 씌워 어휘 전체에 대한 확률을 낸다.

예를 들어 어휘가 7개이고 $$N = 5$$이면 $$W$$는 $$7 \times 5$$, $$W'$$는 $$5 \times 7$$이며 학습이 끝나면 각 단어는 길이 5의 벡터가 된다.

Gensim으로 CBOW를 학습하고 단어 쌍의 유사도를 본다.

```python
model1 = gensim.models.Word2Vec(data,            # 토큰화된 문장 리스트
                                min_count=1,     # 이 빈도 미만 단어는 무시
                                vector_size=100, # 임베딩 차원 N
                                window=5,        # 문맥 윈도우 크기
                                sg=0)            # 0: CBOW, 1: skip-gram (기본 0)

print("Cosine similarity between 'peter' 'wendy' - CBOW : ",
      model1.wv.similarity('peter', 'wendy'))
print("Cosine similarity between 'peter' 'hook' - CBOW : ",
      model1.wv.similarity('peter', 'hook'))
```

```text
Cosine similarity between 'peter' 'wendy' - CBOW :  1.0
Cosine similarity between 'peter' 'hook' - CBOW :  0.027709894
```

`peter`와 `wendy`는 거의 같은 방향, `peter`와 `hook`은 거의 무관하게 나왔다. Word2Vec은 초기화와 학습 순서에 무작위성이 있어 실행할 때마다 값이 달라지며, 작은 말뭉치에서는 특히 그렇다.

**skip-gram**

**skip-gram** 은 CBOW를 뒤집어, 가운데 단어 하나로 주변 단어들을 예측한다. `slept` 하나로 앞뒤의 `calm`, `cat`, `on`, `the`를 맞히는 식이다. 가운데 단어 기준으로 앞뒤 몇 개 단어를 문맥으로 볼지가 윈도우 크기다. 하나의 입력에서 여러 개의 예측 문제가 생기므로, 드문 단어도 더 많이 학습되는 경향이 있다.

```python
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)  # skip-gram

print(model2.wv.similarity('peter', 'wendy'))
print(model2.wv.similarity('peter', 'hook'))
```

```text
0.40088683
0.52016735
```

이번에는 `peter`와 `wendy`가 0.40, `peter`와 `hook`이 0.52로 CBOW와 정반대 경향이 나왔다. 어느 쪽이 우월하다기보다, 데이터의 성격과 분석 목적에 따라 골라 쓰고 결과를 비교해 보아야 한다는 것이 요점이다.

**FastText**

**FastText(패스트텍스트)** 는 페이스북이 Word2Vec의 약점을 보완하려고 만든 임베딩이다. Word2Vec은 단어 하나를 더 쪼갤 수 없는 단위로 보고 벡터를 배정하므로 두 가지 문제가 있다.

- **사전에 없는 단어(OOV, out-of-vocabulary)** 에는 벡터가 없다.
- **드물게 나온 단어** 는 학습 기회가 적어 벡터가 불안정하다.

FastText는 단어를 더 작은 조각인 **n-그램(n-gram)** 부분 단어(subword)의 묶음으로 보고, 각 조각마다 벡터를 학습한 뒤 그 합으로 단어 벡터를 만든다. 그래서 처음 보는 단어도 조각들이 겹치는 다른 단어의 정보를 빌려 벡터를 얻을 수 있고, 드문 단어도 흔한 조각을 공유하므로 더 안정적으로 학습된다. 오타나 활용형 같은 잡음에도 강하다.

n-그램은 연속된 n개의 단위를 묶은 것이다. n이 1이면 유니그램(unigram), 2면 바이그램(bigram), 3이면 트라이그램(trigram)이다. 책은 단어를 단위로 예를 든다.

| n | "This is Deep Learning Book"의 분리 |
| --- | --- |
| 1 | This / is / Deep / Learning / Book |
| 2 | This is / is Deep / Deep Learning / Learning Book |
| 3 | This is Deep / is Deep Learning / Deep Learning Book |

실제 FastText는 단어 안의 문자 n-그램을 쓴다. 예를 들어 n=3이면 `<where>`는 `<wh`, `whe`, `her`, `ere`, `re>`로 나뉜다(`<`, `>`는 단어 경계 표시).

Gensim의 `FastText`는 Word2Vec과 같은 인자를 받는다.

```python
from gensim.models import FastText

model = FastText('../chap10/data/peter.txt',  # 학습할 텍스트 파일
                 vector_size=4,  # 임베딩 차원
                 window=3,       # 앞뒤로 볼 단어 수
                 min_count=1,    # 최소 빈도
                 epochs=10)      # 반복 횟수 (예전 이름은 iter)

print(model.wv.similarity('peter', 'wendy'))
print(model.wv.similarity('peter', 'hook'))
```

```text
0.45924556
0.043825187
```

직접 학습하는 대신 미리 학습된 벡터를 가져다 쓸 수도 있다. 페이스북은 위키백과로 학습한 294개 언어의 FastText 벡터를 공개했고([fasttext.cc](https://fasttext.cc/docs/en/pretrained-vectors.html)), 한국어 파일 `wiki.ko.vec`은 Gensim의 `KeyedVectors`로 불러온다.

```python
from gensim.models import KeyedVectors

model_kr = KeyedVectors.load_word2vec_format('../chap10/data/wiki.ko.vec')

for word, score in model_kr.similar_by_word('노력'):
    print("Word: {0}, Similarity: {1:.2f}".format(word, score))
```

```text
Word: 노력함, Similarity: 0.80
Word: 노력중, Similarity: 0.75
Word: 노력만, Similarity: 0.72
Word: 노력과, Similarity: 0.71
...
```

비슷한 단어 대부분이 "노력"에 조사나 어미가 붙은 형태다. 형태소 분석 없이 어절 단위로 학습한 한국어 벡터의 특징이자, 부분 단어를 공유하는 FastText의 특징이 함께 드러난다. 벡터 산술도 된다. `positive`의 벡터는 더하고 `negative`의 벡터는 빼서 가장 가까운 단어를 찾는다.

```python
model_kr.most_similar(positive=['동물', '육식동물'], negative=['사람'])
```

```text
[('초식동물', 0.780...), ('거대동물', 0.754...), ('육식동물의', 0.754...), ('유두동물', 0.753...), ...]
```

### 10.1.4 횟수/예측 기반 임베딩

횟수 기반 방법(예: 단어-문서 행렬을 특잇값 분해하는 **LSA(Latent Semantic Analysis, 잠재 의미 분석)**)은 말뭉치 전체의 통계를 쓰지만 단어 유추 같은 작업에 약하고, Word2Vec 같은 예측 기반 방법은 유추에 강하지만 작은 윈도우 안만 보느라 전역 통계를 활용하지 못한다.

**GloVe(Global Vectors for Word Representation, 글로브)** 는 둘을 합친다. 먼저 말뭉치 전체에서 단어 $$i$$와 $$j$$가 같은 윈도우에 함께 나온 횟수 $$X_{ij}$$(전역 동시 출현 행렬)를 센 다음, 두 단어 벡터의 내적이 그 동시 출현 횟수의 로그를 맞히도록 학습한다. 즉 skip-gram식 학습에 통계 정보를 학습 목표로 넣은 형태다.

$$
w_i^\top \tilde{w}_j + b_i + \tilde{b}_j \approx \log X_{ij}
$$

이렇게 학습하면 "man → woman"과 "king → queen"처럼 같은 관계의 단어 쌍이 벡터 공간에서 평행한 방향으로 놓인다.

예제는 스탠퍼드가 공개한 `glove.6B.100d.txt`(40만 단어, 100차원)를 Gensim에서 읽을 수 있는 Word2Vec 형식으로 변환해 쓴다.

```python
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

glove_file = datapath('../chap10/data/glove.6B.100d.txt')     # 경로 오류 시 절대 경로로
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt")
glove2word2vec(glove_file, word2vec_glove_file)  # GloVe -> word2vec 형식 (헤더 한 줄 추가)
```

```text
(400000, 100)
```

변환된 파일을 `KeyedVectors`로 읽으면 Word2Vec과 같은 함수를 쓸 수 있다.

```python
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.most_similar('bill')
```

```text
[('legislation', 0.807...), ('proposal', 0.730...), ('senate', 0.714...), ('bills', 0.704...),
 ('measure', 0.695...), ('passed', 0.690...), ('amendment', 0.684...), ... ('clinton', 0.666...)]
```

뉴스 말뭉치로 학습한 벡터라 `bill`이 "법안"의 의미로 묶이고, 빌 클린턴의 `clinton`까지 등장한다. `cherry`로 검색하면 `peach`, `mango`, `plum`, `berry`, `blossom` 같은 과일과 꽃 관련 단어가 나오고, `negative=['cherry']`로 반대 방향을 찾으면 `str94`, URL 같은 의미 없는 토큰이 나온다. 반대 방향은 "반의어"가 아니라 그냥 무관한 영역이라는 점을 보여 준다.

유명한 단어 유추도 확인한다. "king - man + woman"에 가장 가까운 단어를 찾는다.

```python
result = model.most_similar(positive=['woman', 'king'], negative=['man'])
print("{}: {:.4f}".format(*result[0]))

def analogy(x1, x2, y1):
    # x1 : x2 = y1 : ?  ->  y1 + x2 - x1 에 가장 가까운 단어
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]

print(analogy('australia', 'beer', 'france'))
print(analogy('tall', 'tallest', 'long'))
print(model.doesnt_match("breakfast cereal dinner lunch".split()))  # 나머지와 가장 동떨어진 단어
```

```text
queen: 0.7699
champagne
longest
cereal
```

"호주에 맥주가 있다면 프랑스에는 샴페인", "tall의 최상급이 tallest라면 long은 longest"처럼 의미 관계와 문법 관계가 모두 벡터 산술로 풀린다. `doesnt_match`는 세 끼 식사와 다른 하나인 `cereal`을 골랐다.

## 10.2 트랜스포머 어텐션

[📖 thebook.io](https://thebook.io/080289/0736/)

번역 같은 시퀀스 변환 문제는 보통 **인코더(encoder)** 와 **디코더(decoder)** 로 푼다. 인코더가 입력 문장을 벡터로 압축하고, 디코더가 그 벡터를 받아 출력 문장을 한 단어씩 만든다. RNN 기반 인코더의 마지막 은닉 상태 하나에 문장 전체를 담으면 문장이 길수록 앞부분 정보가 흐려진다. 그렇다고 인코더의 모든 시점 은닉 상태를 그대로 넘기면 디코더가 다룰 양이 너무 많다.

**어텐션(attention)** 은 이 사이의 절충이다. 디코더는 출력 단어를 하나 만들 때마다 인코더의 모든 은닉 상태에 대해 "지금 얼마나 중요한가"를 점수로 매기고, 소프트맥스로 정규화한 가중치로 은닉 상태들을 가중합해 하나의 벡터로 만든다. 즉 매 시점마다 꼭 봐야 할 입력 위치에 집중한다.

**트랜스포머(transformer)** 는 2017년 논문 "Attention Is All You Need"(Vaswani 외)에서 제안된 구조로, RNN 없이 어텐션만으로 인코더와 디코더를 만든다. 인코더 블록과 디코더 블록을 여러 층 쌓는 것이 특징이다(논문은 각각 6개).

- **인코더 블록**: 셀프 어텐션(self-attention) 층과 전방향 신경망(feed-forward network)으로 이루어진다. 셀프 어텐션은 같은 문장 안의 단어들이 서로 얼마나 관련 있는지 계산해 각 단어 표현에 반영하고, 그 결과를 전방향 신경망이 위치별로 변환한다. 모든 인코더 블록은 구조가 같다.
- **디코더 블록**: 세 층으로 이루어진다. 먼저 지금까지 생성한 출력끼리의 셀프 어텐션, 그다음 인코더 출력을 참조하는 **인코더-디코더 어텐션(encoder-decoder attention)**, 마지막으로 전방향 신경망이다.

어텐션 계산을 RNN 인코더-디코더(바다나우 어텐션) 기준으로 식으로 정리하면 다음과 같다. 인코더의 $$j$$번째 은닉 상태를 $$h_j$$, 디코더의 $$i$$번째 은닉 상태를 $$s_i$$라 한다.

1. **어텐션 스코어**: 디코더가 $$i$$번째 단어를 만들 때, 직전 디코더 상태 $$s_{i-1}$$과 각 인코더 상태 $$h_j$$가 얼마나 잘 맞는지 점수를 낸다. $$a$$는 작은 신경망(정렬 모델)이다.

$$
e_{ij} = a(s_{i-1}, h_j)
$$

2. **어텐션 가중치**: 점수를 소프트맥스로 0과 1 사이 확률로 바꾼다. 이것이 시점 $$j$$에 주는 가중치다.

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

3. **컨텍스트 벡터(context vector)**: 가중치로 인코더 은닉 상태를 가중합한다.

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

4. **디코더 상태 갱신**: 직전 상태, 직전 출력, 그리고 이번 컨텍스트 벡터로 새 상태를 만든다.

$$
s_i = f(s_{i-1}, y_{i-1}, c_i)
$$

어텐션이 없는 인코더-디코더는 마지막 식이 $$s_i = f(s_{i-1}, y_{i-1}, c)$$로, 모든 시점이 같은 고정 컨텍스트 $$c$$를 쓴다. 어텐션은 시점마다 $$c_i$$가 달라진다는 점이 핵심이다.

예를 들어 "They are cats"를 번역하면서 두 번째 출력 $$y_2$$를 만들 때는, $$s_1$$과 세 인코더 상태로 $$e_{21}, e_{22}, e_{23}$$을 계산하고, 소프트맥스로 $$\alpha_{21}, \alpha_{22}, \alpha_{23}$$을 얻고, 가중합으로 $$c_2$$를 만든 뒤 $$s_2$$와 $$y_2$$를 계산한다.

### 10.2.1 seq2seq

**seq2seq(sequence to sequence)** 는 입력 시퀀스를 받아 다른 출력 시퀀스를 만드는 모델이다. 품사 태깅 같은 **시퀀스 레이블링(sequence labeling)** 은 입력 $$x_1, \dots, x_n$$마다 출력 $$y_1, \dots, y_n$$이 하나씩 대응하지만, 번역은 입력 $$x_{1:n}$$과 뜻이 같은 출력 $$y_{1:m}$$을 만드는 문제라 길이가 달라도 되고 단어끼리 일대일로 대응하지도 않는다.

예제는 PyTorch 튜토리얼을 바탕으로 영어를 프랑스어로 번역하는 seq2seq를 구현한다. 데이터는 Tatoeba 프로젝트의 문장 쌍을 정리한 [manythings.org/anki](http://www.manythings.org/anki/)의 `eng-fra.txt`로, 한 줄에 영어 문장과 프랑스어 문장이 탭으로 구분되어 있다.

```text
You are a good student.   Tu es un bon étudiant.
```

준비 과정에서 정규표현식 모듈 `re`를 쓴다. 대괄호 `[...]`는 "괄호 안 문자 중 하나"를 뜻하므로 `re.compile('[cats]').findall('I love cats.')`는 단어 `cats`가 아니라 `['c', 'a', 't', 's']`처럼 문자 하나하나를 찾는다.

**데이터 준비**

먼저 단어와 인덱스를 오가는 사전 클래스를 만든다. 인덱스 0과 1은 문장 시작(SOS)과 끝(EOS) 토큰에 예약한다.

```python
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20   # 이보다 긴 문장은 버린다

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2              # SOS, EOS 포함 단어 수

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:   # 새 단어면 번호 부여
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:                             # 이미 있으면 빈도만 증가
            self.word2count[word] += 1
```

문장은 소문자로 바꾸고, 알파벳과 공백 외의 문자를 공백으로 치환하고, 악센트 같은 유니코드 문자를 ASCII로 바꾸는 정규화를 거친다. 그 뒤 두 언어 모두 `MAX_LENGTH`보다 짧은 쌍만 남기며 사전을 채운다.

```python
def normalizeString(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace('[^A-Za-z\s]+', ' ')   # 알파벳, 공백 외 제거
    sentence = sentence.str.normalize('NFD')               # 유니코드 분해
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence

def process_data(lang1, lang2):
    df = pd.read_csv('../chap10/data/%s-%s.txt' % (lang1, lang2),
                     delimiter='\t', header=None, names=[lang1, lang2])  # 탭 구분, 헤더 없음
    sentence1, sentence2 = normalizeString(df, lang1), normalizeString(df, lang2)
    input_lang, output_lang, pairs = Lang(), Lang(), []
    for i in range(len(df)):
        if len(sentence1[i].split(' ')) < MAX_LENGTH and len(sentence2[i].split(' ')) < MAX_LENGTH:
            input_lang.addSentence(sentence1[i])    # 입력: 영어
            output_lang.addSentence(sentence2[i])   # 출력: 프랑스어
            pairs.append([sentence1[i], sentence2[i]])
    return input_lang, output_lang, pairs
```

신경망은 텐서만 받으므로 문장을 인덱스 텐서로 바꾸고, 끝에 EOS를 붙여 문장이 끝났음을 알린다.

```python
def tensorFromSentence(lang, sentence):
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)  # (단어 수, 1)

def tensorsFromPair(input_lang, output_lang, pair):
    return tensorFromSentence(input_lang, pair[0]), tensorFromSentence(output_lang, pair[1])
```

**인코더와 디코더**

인코더는 임베딩 층과 GRU 층으로 이루어진다. 임베딩 층은 단어 인덱스로 임베딩 벡터를 조회하는 테이블이고, GRU는 단어를 차례로 받아 은닉 상태를 갱신한다.

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embbed_dim)       # 어휘 크기 x 임베딩 차원
        self.gru = nn.GRU(embbed_dim, hidden_dim, num_layers=num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)  # (시퀀스 1, 배치 1, 임베딩)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden
```

디코더는 임베딩, GRU, 선형 층으로 이루어지고, 선형 층 출력에 로그 소프트맥스를 씌워 다음 단어의 로그 확률을 낸다. **로그 소프트맥스(LogSoftmax)** 는 소프트맥스에 로그를 취한 것으로, 아주 작은 확률을 다룰 때 수치적으로 안정적이다.

$$
\text{softmax}(z)_k = \frac{\exp(z_k)}{\sum_{l} \exp(z_l)}, \qquad \text{LogSoftmax}(z)_k = z_k - \log \sum_{l} \exp(z_l)
$$

```python
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embbed_dim)
        self.gru = nn.GRU(embbed_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, output_dim)   # 은닉 상태 -> 어휘 점수
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)                       # (1, 배치)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))  # 다음 단어의 로그 확률
        return prediction, hidden
```

**seq2seq 모델과 티처 포싱**

`Seq2Seq`는 입력 단어를 모두 인코더에 통과시킨 뒤, 인코더의 마지막 은닉 상태를 디코더의 첫 은닉 상태로 넘긴다. 디코더의 첫 입력은 SOS 토큰이고, 이후에는 한 단어씩 예측하며 다음 입력을 정한다.

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        input_length = input_lang.size(0)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):                     # 입력 문장 인코딩
            encoder_output, encoder_hidden = self.encoder(input_lang[i])

        decoder_hidden = encoder_hidden.to(device)        # 인코더 마지막 상태 -> 디코더 초기 상태
        decoder_input = torch.tensor([SOS_token], device=device)

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)           # 가장 확률 높은 단어
            input = (output_lang[t] if teacher_force else topi)
            if (teacher_force == False and input.item() == EOS_token):
                break
        return outputs
```

**티처 포싱(teacher forcing)** 은 디코더의 다음 입력으로 모델 자신의 예측 대신 정답 단어를 넣어 주는 기법이다. 학습 초기에 모델이 엉뚱한 단어를 예측하면 그 오류가 다음 입력으로 이어져 학습이 흔들리는데, 정답을 넣어 주면 안정적이고 빠르게 수렴한다. 대신 실제 추론에서는 정답이 없으므로, 정답에 익숙해진 모델이 자기 예측에 의존할 때 불안정해질 수 있다. 그래서 `teacher_forcing_ratio=0.5`처럼 확률적으로 섞는다.

이 코드에는 짚어 둘 점이 두 가지 있다. 첫째, 다음 입력을 담는 변수 이름이 `decoder_input`이 아니라 `input`이라, 루프 안에서 디코더에 들어가는 입력은 계속 SOS로 남는다(튜토리얼 원본은 `decoder_input = ...`이다). 둘째, `Encoder.forward`가 이전 은닉 상태를 받지 않으므로 단어마다 은닉 상태가 0에서 새로 시작하고, 디코더로 넘어가는 `encoder_hidden`은 사실상 마지막 토큰(EOS) 하나만 인코딩한 값이다. 아래 번역 결과가 입력과 무관하게 거의 같은 문장을 내놓는 주된 이유로 보인다.

**학습과 평가**

손실은 출력 시점마다 계산해 더한다. 디코더가 이미 로그 소프트맥스를 출력하므로 손실 함수는 `nn.NLLLoss`를 쓴다. `nn.CrossEntropyLoss`는 내부에 LogSoftmax와 NLLLoss를 모두 포함하므로, 모델 출력이 원시 점수(logit)일 때 쓰고, 출력에 이미 로그 소프트맥스를 씌웠다면 `NLLLoss`를 쓴다.

```python
def Model(model, input_tensor, target_tensor, model_optimizer, criterion):
    model_optimizer.zero_grad()
    loss = 0
    output = model(input_tensor, target_tensor)
    num_iter = output.size(0)
    for ot in range(num_iter):                          # 시점별 손실 합산
        loss += criterion(output[ot], target_tensor[ot])
    loss.backward()
    model_optimizer.step()
    return loss.item() / num_iter

def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0
    training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(num_iteration)]     # 무작위 문장 쌍을 미리 뽑는다
    for iter in range(1, num_iteration + 1):
        input_tensor, target_tensor = training_pairs[iter - 1]
        total_loss_iterations += Model(model, input_tensor, target_tensor, optimizer, criterion)
        if iter % 5000 == 0:
            print('%d %.4f' % (iter, total_loss_iterations / 5000))
            total_loss_iterations = 0
    torch.save(model.state_dict(), '../chap10/data/mytraining.pt')
    return model
```

평가 함수는 `torch.no_grad()` 아래에서 모델 출력의 각 시점마다 `topk(1)`로 가장 높은 단어를 고르고, EOS를 만나면 멈춘다. `evaluateRandomly`는 학습 쌍 중 임의로 몇 개를 골라 입력, 정답, 예측을 나란히 출력한다.

```python
lang1, lang2 = 'eng', 'fra'
input_lang, output_lang, pairs = process_data(lang1, lang2)

embed_size, hidden_size, num_layers = 256, 512, 1
num_iteration = 75000

encoder = Encoder(input_lang.n_words, hidden_size, embed_size, num_layers)
decoder = Decoder(output_lang.n_words, hidden_size, embed_size, num_layers)
model = Seq2Seq(encoder, decoder, device).to(device)
model = trainModel(model, input_lang, output_lang, pairs, num_iteration)
```

```text
Input : 13366 Output : 25937
5000 4.8727
10000 4.6251
...
70000 4.4384
75000 4.4197
```

영어 어휘 13,366개, 프랑스어 어휘 25,937개이고, 손실은 4.87에서 4.42로 조금씩 줄어든다. 그러나 무작위 문장의 번역 결과를 보면 입력과 상관없이 거의 모두 `je ne pas <EOS>`를 출력한다.

```text
input he is eight
output il a huit ans
predicted je ne pas <EOS>
input you have many books
output tu as de nombreux livres
predicted je ne pas <EOS>
```

가장 흔한 단어 조합만 내놓는 단계에 머문 것이다. 단어 하나씩, 배치 크기 1로 학습했고 반복 횟수도 충분하지 않은 데다, 위에서 짚은 디코더 입력 문제까지 겹친 결과로 보인다.

**어텐션 디코더**

기본 seq2seq는 인코더 RNN의 마지막 은닉 상태 하나만 디코더로 넘긴다. 그래서 두 가지 문제가 생긴다.

1. 고정된 크기의 벡터 하나에 문장 전체를 담느라 정보가 손실된다.
2. 긴 시퀀스에서 RNN의 기울기 소멸(vanishing gradient)로 앞부분 정보가 약해진다.

**어텐션 메커니즘(attention mechanism)** 은 디코딩하는 동안 인코더의 모든 은닉 상태를 보관하고, 시점마다 다른 컨텍스트 벡터를 만들어 쓴다. 컨텍스트 벡터는 인코더 전체 은닉 상태와 디코더의 이전 은닉 상태로부터 계산된다.

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size * 2, max_length)          # [임베딩; 은닉] -> 위치별 점수
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input).view(1, 1, -1))
        # 현재 입력 임베딩과 이전 은닉 상태로 입력 위치별 어텐션 가중치 계산
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # 가중치 (1, 1, max_length) x 인코더 출력 (1, max_length, hidden) -> (1, 1, hidden)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)       # 입력 + 컨텍스트
        output = F.relu(self.attn_combine(output).unsqueeze(0))
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
```

- `self.attn`은 입력 시퀀스의 각 위치에 점수를 하나씩 내므로 출력 크기가 `max_length`다. 따라서 `max_length`는 모든 입력 문장의 최대 길이 이상이어야 한다.
- `torch.bmm`은 **배치 행렬 곱(batch matrix multiplication)** 이다. `(B, A, B')`와 `(B, B', C)` 모양의 텐서를 받아 배치마다 뒤 두 차원끼리 행렬 곱을 해서 `(B, A, C)`를 만든다. 여기서는 어텐션 가중치와 인코더 출력을 곱해 컨텍스트 벡터를 얻는다.

책의 학습 코드 `trainIters`는 인코더와 어텐션 디코더를 받아 각각 SGD 옵티마이저를 만들지만, 실제 손실 계산에는 앞에서 만든 `model`(어텐션 없는 `Seq2Seq`)과 `decoder_optimizer`를 넘긴다. 그래서 출력되는 손실은 어텐션 디코더의 학습 결과라고 보기 어렵다.

```text
AttnDecoderRNN(
  (embedding): Embedding(25937, 512)
  (attn): Linear(in_features=1024, out_features=20, bias=True)
  (attn_combine): Linear(in_features=1024, out_features=512, bias=True)
  (dropout): Dropout(p=0.1, inplace=False)
  (gru): GRU(512, 512)
  (out): Linear(in_features=512, out_features=25937, bias=True)
)
5000,   4.6978
10000,   4.7173
...
75000,   4.7222
```

손실이 4.7 근처에서 오르내릴 뿐 줄지 않는다. 책은 어텐션의 효과는 긴 시퀀스에서 잘 드러난다고 덧붙인다. 이 데이터의 문장은 대부분 짧다는 점도 함께 기억해 둔다.

### 10.2.2 버트(BERT)

**버트(BERT, Bidirectional Encoder Representations from Transformers)** 는 2018년 구글이 공개한 언어 모델이다. 트랜스포머의 인코더 블록만 쌓은 구조로, 문장을 왼쪽에서 오른쪽으로 한 방향으로만 읽는 기존 언어 모델과 달리 양방향 문맥을 한꺼번에 본다. 방대한 텍스트로 미리 학습(사전 학습, pre-training)해 두고, 분류 같은 실제 과제에서는 그 위에 출력층 하나를 얹어 과제 데이터로 조금 더 학습한다(미세 조정, fine-tuning). 사전 학습 과제로는 가려진 단어 맞히기와 두 문장이 이어지는지 맞히는 다음 문장 예측(NSP, Next Sentence Prediction)을 쓴다.

입력은 다음 순서로 준비한다.

1. 문장 앞에 `[CLS]`, 문장 끝에 `[SEP]` 토큰을 붙인다. 분류 과제에서는 `[CLS]` 위치의 출력이 문장 전체를 대표한다.
2. **워드피스(WordPiece)** 토크나이저로 단어를 부분 단어로 나눈다. 단어 중간에 이어지는 조각은 `##`로 시작한다(예: `딥`, `##러`, `##닝`).
3. 각 토큰을 사전의 고유 번호로 바꾸고, 길이를 맞추기 위해 빈 자리는 0(패딩)으로 채운다.

버트는 크기에 따라 두 가지가 있다. $$L$$은 트랜스포머 블록 수, $$H$$는 은닉 크기, $$A$$는 셀프 어텐션 헤드 수다.

| 모델 | L | H | A | 파라미터 |
| --- | --- | --- | --- | --- |
| BERT-base | 12 | 768 | 12 | 약 1.1억 |
| BERT-large | 24 | 1024 | 16 | 약 3.4억 |

예제는 네이버 영화 리뷰 감성 분류(NSMC)를 버트로 미세 조정한다. 라이브러리는 `pytorch-transformers`(현재의 Hugging Face `transformers`)를 쓴다.

```python
from torch.utils.data import Dataset, DataLoader
from pytorch_transformers import BertTokenizer, BertForSequenceClassification

train_df = pd.read_csv('../chap10/data/training.txt', sep='\t')
valid_df = pd.read_csv('../chap10/data/validing.txt', sep='\t')   # 훈련 데이터에서 떼어 낸 검증용
test_df = pd.read_csv('../chap10/data/testing.txt', sep='\t')

# 빠른 실습을 위해 10%만 사용
train_df = train_df.sample(frac=0.1, random_state=500)
valid_df = valid_df.sample(frac=0.1, random_state=500)
test_df = test_df.sample(frac=0.1, random_state=500)
```

데이터는 `id`, `document`(리뷰 문장), `label`(1 긍정, 0 부정)의 세 열이다. `Dataset`은 `id`를 건너뛰고 문장과 레이블만 돌려준다.

```python
class Datasets(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return self.df.iloc[idx, 1], self.df.iloc[idx, 2]   # (document, label)

train_loader = DataLoader(Datasets(train_df), batch_size=2, shuffle=True, num_workers=0)
valid_loader = DataLoader(Datasets(valid_df), batch_size=2, shuffle=True, num_workers=0)
test_loader = DataLoader(Datasets(test_df), batch_size=2, shuffle=True, num_workers=0)
```

사전 학습된 토크나이저와 분류용 버트 모델을 내려받는다.

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')  # 버트 + 분류층
model.to(device)
```

`bert-base-uncased`는 영어 소문자 텍스트로 학습한 기본 모델이다. 한국어를 다루려면 100개 이상 언어로 학습한 `bert-base-multilingual-cased`가 맞지만, 예제는 일부러 기본 모델을 쓴다. 출력된 구조를 요약하면 다음과 같다.

```text
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768) ...)
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(768, 768) (key): Linear(768, 768) (value): Linear(768, 768) ...)
          (intermediate): ... Linear(768, 3072)
          (output): ... Linear(3072, 768) ...)
        ... (1) ~ (11) 동일 ...
    (pooler): BertPooler( ... Linear(768, 768), Tanh() ))
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
```

임베딩은 단어, 위치(최대 512), 문장 구분(2종) 세 가지를 더해 만들고, 그 위에 셀프 어텐션(query, key, value)과 전방향 신경망(768 → 3072 → 768)으로 된 블록이 12번 반복된다. 맨 위의 `classifier`가 미세 조정 때 새로 붙는 2-클래스 분류층이다.

학습 함수는 검증 손실이 가장 낮을 때마다 체크포인트를 저장하는 구조다. 저장, 불러오기 함수는 `state_dict`와 검증 손실, 손실 기록을 딕셔너리로 묶어 `torch.save`/`torch.load`하는 간단한 래퍼다. 학습 루프의 핵심은 다음과 같다.

```python
def train(model, optimizer, num_epochs=5, eval_every=len(train_loader) // 2,
          best_valid_loss=float("Inf")):
    ...
    model.train()
    for epoch in range(num_epochs):
        for text, label in train_loader:
            optimizer.zero_grad()
            # 문장 -> 토큰 번호 ([CLS], [SEP] 자동 추가) -> 길이 512로 0 패딩
            encoded_list = [tokenizer.encode(t, add_special_tokens=True) for t in text]
            padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]
            sample = torch.tensor(padded_list).to(device)
            labels = label.to(device)

            loss, logits = model(sample, labels=labels)   # labels를 주면 손실도 함께 반환
            pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
            ...
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_every == 0:            # 에폭 중간마다 검증
                model.eval()
                with torch.no_grad():
                    ...                                   # 검증 손실 평균 계산
                if best_valid_loss > average_valid_loss:   # 더 좋아지면 저장
                    best_valid_loss = average_valid_loss
                    save_checkpoint('../chap10/data/model.pt', model, best_valid_loss)
                model.train()

optimizer = optim.Adam(model.parameters(), lr=2e-5)   # 미세 조정용 작은 학습률
train(model=model, optimizer=optimizer)
```

- `BertForSequenceClassification`에 `labels`를 넘기면 내부에서 교차 엔트로피 손실을 계산해 `(loss, logits)`를 돌려준다. 책의 `train` 함수는 `criterion=nn.BCELoss()`를 인자로 받지만 실제로는 쓰지 않는다.
- `torch.argmax`는 가장 큰 값의 인덱스를 돌려준다. 차원을 지정하지 않으면 텐서를 1차원으로 펼친 뒤의 인덱스이고, `dim=1`이면 행마다 가장 큰 열의 인덱스다. 소프트맥스는 순서를 바꾸지 않으므로 logits에 바로 `argmax`를 해도 결과는 같다.
- 사전 학습된 큰 모델을 미세 조정할 때는 2e-5처럼 매우 작은 학습률을 쓴다. 학습률이 크면 사전 학습으로 얻은 표현이 망가진다.

```text
Epoch [1/5], Step [510/5100], Train Loss: 0.7124, Valid Loss: 0.6968
Epoch [2/5], Step [1530/5100], Train Loss: 0.7067, Valid Loss: 0.6943
Epoch [3/5], Step [3060/5100], Train Loss: 0.6999, Valid Loss: 0.6927
Epoch [5/5], Step [5100/5100], Train Loss: 0.6984, Valid Loss: 0.6968
훈련 종료!
```

두 클래스를 무작위로 맞힐 때의 교차 엔트로피가 $$\ln 2 \approx 0.693$$이므로, 손실이 거의 그 값에 머문다는 것은 사실상 아무것도 배우지 못했다는 뜻이다. 손실 곡선을 그려 보면 훈련 손실은 아주 조금 내려가지만 검증 손실은 2000 스텝 부근에서 튄 뒤 오르내린다.

테스트 데이터로 평가할 때는 사이킷런의 `classification_report`(클래스별 정밀도, 재현율, F1)와 `confusion_matrix`를 쓰고, 혼동 행렬은 seaborn의 `heatmap`으로 그린다. `labels=[1, 0]`은 행과 열의 클래스 순서를 1, 0으로 바꾸는 옵션이고, `heatmap`의 `annot=True`, `fmt="d"`는 칸마다 정수 개수를 적는다.

```python
def evaluate(model, test_loader):
    y_pred, y_true = [], []
    model.eval()
    with torch.no_grad():
        for text, label in test_loader:
            ...                                            # 학습 때와 같은 토큰화, 패딩
            _, output = model(sample, labels=labels)
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(labels.tolist())
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    sns.heatmap(cm, annot=True, cmap='Blues', fmt="d")
```

```text
              precision    recall  f1-score   support
           1     0.5091    1.0000    0.6747       558
           0     0.0000    0.0000    0.0000       538
    accuracy                         0.5091      1096
```

모델이 모든 리뷰를 긍정(1)으로 예측해서, 정확도가 긍정 비율과 같은 50.9%다. 영어 전용 모델에 한국어를 넣었으니 대부분의 한국어 글자가 미등록 토큰이 되었고, 학습 데이터도 10%만 썼기 때문이다. 다국어 모델은 다음 절에서 쓴다.

## 10.3 한국어 임베딩

[📖 thebook.io](https://thebook.io/080289/0781/)

한국어 임베딩도 절차는 영어와 같다. 이 절은 다국어 버트 `bert-base-multilingual-cased`로 한국어 문장의 **문맥 임베딩(contextual embedding)** 을 뽑아 본다. Word2Vec 같은 정적 임베딩은 단어마다 벡터가 하나뿐이지만, 버트는 같은 단어라도 주변 문맥에 따라 다른 벡터를 낸다. "사과"(과일)와 "사과"(용서를 구함)를 구분할 수 있는지 확인하는 것이 목표다.

```python
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

text = "나는 파이토치를 이용한 딥러닝을 학습중이다."
marked_text = "[CLS] " + text + " [SEP]"
print(tokenizer.tokenize(marked_text))
```

```text
['[CLS]', '나는', '파', '##이', '##토', '##치를', '이', '##용한', '딥', '##러', '##닝', '##을', '학', '##습', '##중', '##이다', '.', '[SEP]']
```

워드피스는 사전에 있는 가장 긴 조각부터 맞춰 쪼개는데, 다국어 사전에는 한국어 조각이 많지 않아 "파이토치"가 네 조각으로 나뉘는 등 지나치게 잘게 쪼개진다. 한국어를 진지하게 다루려면 KoBERT처럼 한국어 말뭉치로 학습한 모델이 낫다.

"사과"가 서로 다른 뜻으로 쓰인 세 문장을 하나로 이어 토큰과 번호를 확인한다.

```python
text = "과수원에 사과가 많았다." \
       "친구가 나에게 사과했다." \
       "백설공주는 독이 든 사과를 먹었다."
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)   # 토큰 -> 사전 번호

for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))
```

```text
[CLS]           101
과             8,898
##수          15,891
##원에        108,280
사             9,405
##과          11,882
##가          11,287
...
[SEP]           102
```

"사과"는 매번 `사` + `##과` 두 토큰으로 나뉘고 같은 번호(9,405, 11,882)를 받는다. 번호는 같지만 버트를 통과한 뒤의 벡터는 문맥에 따라 달라질 것이다.

버트는 두 문장을 한 입력에 넣을 때 토큰마다 어느 문장에 속하는지 0 또는 1로 표시한다(segment id). 여기서는 33개 토큰 전체를 한 문장으로 취급해 모두 1로 둔다.

```python
segments_ids = [1] * len(tokenized_text)     # 33개 모두 같은 문장

tokens_tensor = torch.tensor([indexed_tokens])     # (1, 33)
segments_tensors = torch.tensor([segments_ids])    # (1, 33)

model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                  output_hidden_states=True)  # 모든 층의 은닉 상태 반환
model.eval()

with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
```

이 모델의 단어 임베딩 사전은 119,547개다. `output_hidden_states=True`로 받은 `hidden_states`는 임베딩 층 출력과 12개 인코더 층 출력, 모두 13개 텐서의 튜플이고, 각 텐서는 `(배치 1, 토큰 33, 은닉 768)` 모양이다.

```text
계층 개수: 13   (initial embeddings + 12 BERT layers)
배치 개수: 1
토큰 개수: 33
은닉층의 유닛 개수: 768
```

토큰별로 다루기 쉽게 `(층, 배치, 토큰, 은닉)` 4차원을 `(토큰, 층, 은닉)` 3차원으로 정리한다.

```python
token_embeddings = torch.stack(hidden_states, dim=0)        # (13, 1, 33, 768)
token_embeddings = torch.squeeze(token_embeddings, dim=1)   # 배치 차원 제거 -> (13, 33, 768)
token_embeddings = token_embeddings.permute(1, 0, 2)        # 차원 순서 교환 -> (33, 13, 768)
```

`permute`는 `transpose`의 일반화로, 원하는 차원 순서를 모두 적어 한 번에 바꾼다. 예를 들어 `(2, 3, 3)` 텐서에 `permute(2, 1, 0)`을 하면 `(3, 3, 2)`가 되고, 원래 `x[i, j, k]`가 `y[k, j, i]`로 옮겨 간다.

어느 층의 출력을 단어 벡터로 쓸지는 정해진 답이 없다. 흔한 방법 두 가지를 해 본다.

```python
# 방법 1: 마지막 네 층을 이어 붙이기 -> 토큰당 4 x 768 = 3072차원
token_vecs_cat = []
for token in token_embeddings:                   # token: (13, 768)
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    token_vecs_cat.append(cat_vec)
print('형태는: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))

# 방법 2: 마지막 네 층을 더하기 -> 토큰당 768차원
token_vecs_sum = []
for token in token_embeddings:
    token_vecs_sum.append(torch.sum(token[-4:], dim=0))
print('형태는: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
```

```text
형태는: 33 x 3072
형태는: 33 x 768
```

문장 전체를 벡터 하나로 나타내려면, 끝에서 두 번째 층의 토큰 벡터 33개를 평균한다.

```python
token_vecs = hidden_states[-2][0]                    # (33, 768)
sentence_embedding = torch.mean(token_vecs, dim=0)   # (768,)
```

이제 세 "사과"의 벡터를 비교한다. 토큰 번호를 출력해 보면 "사과"의 `##과` 조각은 5번(과수원의 사과), 16번(친구의 사과), 27번(백설공주의 사과)에 있다.

```python
from scipy.spatial.distance import cosine

diff_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[27])
same_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[16])
print('*유사한* 의미에 대한 벡터 유사성: %.2f' % same_apple)
print('*다른* 의미에 대한 벡터 유사성: %.2f' % diff_apple)
```

```text
*유사한* 의미에 대한 벡터 유사성: 0.86
*다른* 의미에 대한 벡터 유사성: 0.91
```

`scipy`의 `cosine`은 코사인 **거리**이므로 1에서 빼서 유사도로 바꾼다. 책은 이 결과를 보고 다국어 버트가 한국어 의미를 잘 구분하지 못한다고 해석하는데, 변수 이름과 비교 대상이 서로 뒤바뀌어 있다는 점에 주의한다. 5번과 27번은 둘 다 **과일** 사과이고(`diff_apple`이라는 이름과 달리 같은 의미), 5번과 16번이 과일과 **사과(謝過)** 의 비교다. 즉 실제로는 같은 의미끼리 0.91, 다른 의미끼리 0.86으로, 차이는 작지만 방향은 맞게 나왔다. 그래도 차이가 작고 단어가 `사`, `##과`로 쪼개져 있어서, 한국어에는 한국어 전용 모델이 더 정확하다는 결론은 유효하다. (책의 코드 10-68이 출력하는 `token_vecs_sum[6]`, `[10]`, `[19]`도 "사과"가 아니라 `##가`, `친`, `백` 토큰이다.)

정리하면 언어가 달라도 임베딩 절차는 같다. 토큰화하고, 번호로 바꾸고, 사전 학습된 모델에 넣어 벡터를 얻은 뒤, 그 벡터로 유사도 계산, 분류, 예측을 하면 된다.

## 정리

- 원-핫 인코딩은 단어 사이 관계를 담지 못하고(내적이 항상 0) 차원이 어휘 크기만큼 커지는 희소 표현이다.
- 횟수 기반 임베딩은 단어 빈도를 쓴다. 카운터 벡터는 빈도 그대로, TF-IDF는 문서 안 빈도(TF)에 흔한 단어를 깎는 역문서 빈도(IDF)를 곱한다.
- Word2Vec은 주변 단어로 가운데 단어를 맞히는 CBOW와 그 반대인 skip-gram으로 조밀한 단어 벡터를 학습한다. FastText는 문자 n-그램 조각을 학습해 미등록어와 드문 단어에 강하다.
- GloVe는 전역 동시 출현 통계의 로그를 단어 벡터 내적으로 맞히도록 학습하며, "king - man + woman ≈ queen" 같은 유추가 벡터 산술로 풀린다.
- 어텐션은 디코더가 시점마다 인코더 은닉 상태 전체에 소프트맥스 가중치를 주어 가중합한 컨텍스트 벡터 $$c_i$$를 쓰는 방식이며, 트랜스포머는 RNN 없이 셀프 어텐션과 전방향 신경망 블록을 쌓는다.
- seq2seq는 인코더(임베딩 + GRU)와 디코더(임베딩 + GRU + 선형 + LogSoftmax, 손실은 NLLLoss)로 번역을 학습하며, 티처 포싱으로 학습을 안정시킨다. 책의 예제는 거의 같은 문장만 출력하는 수준에 머문다.
- 버트는 트랜스포머 인코더를 쌓아 양방향 문맥으로 사전 학습한 모델로, `[CLS]`/`[SEP]`와 워드피스 토큰을 입력으로 받고 작은 학습률로 미세 조정한다. 영어 전용 모델로 한국어 리뷰를 분류하면 한 클래스만 예측하게 된다.
- 다국어 버트의 은닉 상태(13층 x 토큰 x 768)에서 마지막 네 층을 더하거나 이어 붙여 문맥 단어 벡터를, 끝에서 두 번째 층 평균으로 문장 벡터를 만들 수 있다.

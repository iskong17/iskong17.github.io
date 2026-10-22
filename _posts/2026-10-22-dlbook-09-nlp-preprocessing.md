---
title: "Deep Learning & PyTorch 9: NLP Preprocessing"
date: 2026-10-22 17:09:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 9
card_title: "Chapter 9: NLP Preprocessing"
card_desc: "NLP terms and pipeline · NLTK · KoNLPy · Missing values · Tokenization · Stop words · Stemming and lemmatization · Feature scaling"
lang: ko
notes: "https://thebook.io/080289/0653/"
notes_label: "thebook.io chapter 9"
description: "자연어 처리의 기본 용어와 처리 과정, NLTK·KoNLPy·Gensim·사이킷런의 역할을 정리하고, 결측치 처리, 문장·단어 토큰화와 한국어 형태소 분석, 불용어 제거, 어간·표제어 추출, 그리고 스케일러를 이용한 정규화까지 텍스트 전처리 단계를 코드로 살펴본다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 9장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap09)</li>
</ul>

9장은 텍스트를 모델에 넣기 전에 거치는 전처리를 다룬다. 앞부분은 자연어 처리의 용어와 라이브러리를, 뒷부분은 결측치 처리 → 토큰화 → 불용어 제거 → 어간 추출 → 정규화 순서로 실제 코드를 본다. 전처리된 단어를 벡터로 바꾸는 임베딩은 10장에서 이어진다.

## 9.1 자연어 처리란

[📖 thebook.io](https://thebook.io/080289/0654/)

**자연어 처리(Natural Language Processing, NLP)** 는 사람이 쓰는 언어의 의미를 분석해 컴퓨터가 다룰 수 있게 만드는 분야다. 딥러닝뿐 아니라 언어 자체에 대한 이해도 필요해서 진입 장벽이 높다. 언어마다 구조가 다르다는 점도 어렵다. 영어는 띄어쓰기로 단어가 잘 나뉘지만, 중국어는 띄어쓰기가 없어 단어 단위로 자르는 것부터 문제가 되고, 한국어는 띄어쓰기 단위(어절) 안에 조사와 어미가 붙어 있다.

과제별 완성도도 다르다. 스팸 분류나 맞춤법 검사는 이미 실용 수준이지만, 질의응답이나 자유로운 대화는 여전히 발전 중인 영역으로 꼽힌다(책 출간 시점 기준).

### 9.1.1 자연어 처리 용어 및 과정

#### 자주 쓰는 용어

| 용어 | 의미 |
| --- | --- |
| 말뭉치(corpus) | 모델 학습에 쓰려고 특정 목적에 맞게 모은 텍스트 집합 |
| 토큰(token) | 문서를 나눈 최소 처리 단위(단어, 형태소, 문자 등). 텍스트를 토큰으로 나누는 함수를 토큰 생성 함수라고 한다. |
| 토큰화(tokenization) | 텍스트를 문장이나 단어 같은 토큰으로 나누는 작업 |
| 불용어(stop words) | "a", "the", "he"처럼 자주 나오지만 분석에 별 의미가 없는 단어. 빈도가 높아 오히려 분석을 방해하므로 미리 제거한다. |
| 어간 추출(stemming) | 단어의 변형된 형태를 공통된 기본 형태로 줄이는 작업. 예: consigned, consigning, consignment → consign |
| 품사 태깅(part-of-speech tagging) | 문장의 각 토큰에 명사, 동사 같은 품사 표시를 붙이는 작업 |

품사 태깅을 NLTK로 해 본다. 먼저 단어 단위로 토큰화하고, 태거 자원을 내려받아 `pos_tag`를 적용한다.

```python
import nltk
nltk.download('punkt')                        # 토큰화 자원
nltk.download('averaged_perceptron_tagger')   # 품사 태거 자원

text = nltk.word_tokenize("Is it possible distinguishing cats and dogs")
print(text)
print(nltk.pos_tag(text))
```

```text
['Is', 'it', 'possible', 'distinguishing', 'cats', 'and', 'dogs']
[('Is', 'VBZ'), ('it', 'PRP'), ('possible', 'JJ'), ('distinguishing', 'VBG'),
 ('cats', 'NNS'), ('and', 'CC'), ('dogs', 'NNS')]
```

태그는 펜 트리뱅크(Penn Treebank) 표기를 따른다.

| 태그 | 의미 |
| --- | --- |
| VBZ | 동사, 3인칭 단수 현재형 |
| PRP | 인칭 대명사 |
| JJ | 형용사 |
| VBG | 동사, 동명사 또는 현재 분사 |
| NNS | 명사, 복수형 |
| CC | 등위 접속사 |

책은 `nltk.download()`를 인자 없이 호출해 다운로더 창에서 전체 자원을 받는다. 위처럼 필요한 자원 이름만 지정하면 창 없이 받을 수 있다.

#### 처리 과정

자연어 처리는 대체로 네 단계로 진행된다.

1. **입력**: 사람의 언어로 된 텍스트가 들어온다. 언어에 따라 처리 방법이 조금씩 다르며, 영어 처리 기법이 가장 잘 정리되어 있다.
2. **전처리**: 토큰화, 불용어 제거, 어간 추출 등으로 텍스트를 정리한다. 이 장의 주제다.
3. **임베딩**: 정리된 토큰을 숫자 벡터로 바꾼다(10장).
4. **모델링**: 벡터를 모델(결정 트리, 신경망 등)에 넣어 분류나 예측을 한다.

### 9.1.2 자연어 처리를 위한 라이브러리

#### NLTK

**NLTK(Natural Language ToolKit)** 는 교육용으로 시작했지만 연구와 실무에서도 널리 쓰이는 파이썬 자연어 처리 라이브러리다. 말뭉치, 토큰화, 형태소 분석, 품사 태깅 등을 제공한다.

```python
import nltk
nltk.download('punkt')
string1 = "my favorite subject is math"
string2 = "my favorite subject is math, english, economic and computer science"
print(nltk.word_tokenize(string1))
print(nltk.word_tokenize(string2))
```

```text
['my', 'favorite', 'subject', 'is', 'math']
['my', 'favorite', 'subject', 'is', 'math', ',', 'english', ',', 'economic', 'and', 'computer', 'science']
```

단순히 공백으로 자르는 것과 달리 쉼표 같은 문장 부호를 별도 토큰으로 떼어 낸다.

#### KoNLPy

**KoNLPy** 는 한국어 처리용 파이썬 라이브러리다. 기존 형태소 분석기인 꼬꼬마(Kkma), 코모란(Komoran), 한나눔(Hannanum), 트위터(Twitter, 현재 이름은 Okt), 메카브(Mecab)를 같은 인터페이스로 쓸 수 있게 묶었다. 여러 분석기가 자바로 작성되어 있어서 설치 순서가 있다.

- **윈도**: (1) Oracle JDK를 설치한다. (2) 파이썬 버전과 운영체제 비트 수에 맞는 JPype1(파이썬에서 자바를 호출하는 다리 역할)을 설치한다. 설치가 실패하면 `pip install --upgrade pip`로 pip을 먼저 올리고, 권한 오류가 나면 관리자 권한 프롬프트에서 다시 시도한다. (3) `pip install konlpy`.
- **우분투**: `sudo apt-get install g++ openjdk-11-jdk python3-dev` 후 `pip3 install konlpy`.

**형태소(morpheme)** 는 뜻을 가진 가장 작은 언어 단위다. "고양이인가요"는 '고양이' + '이(서술격 조사)' + 'ㄴ가요(어미)'로 나뉜다. 한국어는 어절 하나에 여러 형태소가 붙어 있으므로 형태소 분석이 토큰화의 핵심이다.

```python
from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs('딥러닝이 쉽나요? 어렵나요?'))          # 형태소 목록
print(komoran.pos('소파 위에 있는 것이 고양이인가요? 강아지인가요?'))   # (형태소, 품사)
```

```text
['딥러닝이', '쉽', '나요', '?', '어렵', '나요', '?']
[('소파', 'NNP'), ('위', 'NNG'), ('에', 'JKB'), ('있', 'VV'), ('는', 'ETM'), ('것', 'NNB'),
 ('이', 'JKS'), ('고양이', 'NNG'), ('이', 'VCP'), ('ㄴ가요', 'EF'), ('?', 'SF'), ...]
```

`morphs`는 형태소만, `pos`는 품사 태그까지 돌려준다. 사전에 없는 '딥러닝이'는 조사가 분리되지 않은 채 남았다. 분석기는 사전에 의존하므로 신조어나 전문 용어는 사용자 사전을 추가해야 잘 처리된다. 품사 태그는 세종 품사 체계를 따른다(NNG 일반 명사, NNP 고유 명사, JKB 부사격 조사, VV 동사, EF 종결 어미 등).

#### Gensim

**Gensim** 은 Word2Vec 임베딩, 토픽 모델링, LDA(Latent Dirichlet Allocation) 등을 제공하는 라이브러리다. 딥러닝 프레임워크는 아니지만 효율적이고 큰 말뭉치에도 확장되므로 널리 쓰인다. `pip install -U gensim`으로 설치한다.

#### 사이킷런

**사이킷런(scikit-learn)** 은 텍스트에서 특성을 뽑는 도구를 제공한다.

| 클래스 | 하는 일 |
| --- | --- |
| `CountVectorizer` | 문서마다 각 단어의 등장 횟수를 세어 특성으로 쓴다. |
| `TfidfVectorizer` | 등장 횟수에 "여러 문서에 흔한 단어일수록 가중치를 낮추는" TF-IDF 가중을 적용한다. |
| `HashingVectorizer` | `CountVectorizer`와 같은 일을 하되 단어를 해시 함수로 열 번호에 대응시켜, 단어 사전을 저장하지 않으므로 빠르고 메모리를 적게 쓴다. |

## 9.2 전처리

[📖 thebook.io](https://thebook.io/080289/0672/)

모델은 텍스트를 그대로 입력으로 받을 수 없다. 먼저 정제와 토큰화를 거쳐 다룰 수 있는 단위로 만들어야 한다. 이 절의 흐름은 결측치 확인 → 토큰화 → 불용어 제거 → 어간 추출 → 정규화다.

### 9.2.1 결측치 확인

**결측치(missing value)** 는 값이 비어 있는(`NaN`) 칸이다. 예를 들어 학생 명단에서 한 학생의 몸무게만 비어 있는 경우다. 결측치가 있으면 계산이 실패하거나 결과가 왜곡되므로 먼저 찾아서 처리한다.

예제 데이터 `class2.csv`는 6행짜리 표로, 좌표를 나타내는 `x`, `y`, `r` 열 일부가 비어 있다.

```python
import pandas as pd
df = pd.read_csv('../chap09/data/class2.csv')

print(df.isnull().sum())             # 열별 결측치 개수
print(df.isnull().sum() / len(df))   # 열별 결측치 비율
```

```text
x    2          x    0.333333
y    2          y    0.333333
r    2          r    0.333333
(나머지 열은 0)
```

`isnull()`은 칸마다 결측 여부를 `True/False`로 표시하고, `sum()`이 열별로 `True`의 개수를 센다. 개수를 행 수로 나누면 비율이 된다. `x`, `y`, `r`이 각각 2개(33%)씩 비어 있다.

처리 방법은 크게 **삭제** 와 **대체** 다.

```python
df = df.dropna(how='all')   # 모든 값이 NaN인 행만 삭제 -> 여기서는 삭제되는 행 없음
df1 = df.dropna()           # NaN이 하나라도 있는 행 삭제 -> 2번, 3번 행이 사라짐
df2 = df.fillna(0)          # NaN을 0으로 대체
df['x'] = df['x'].fillna(df['x'].mean())   # x 열의 NaN을 x 열 평균(500.0)으로 대체
```

```text
   id      tissue class class2      x      y      r
0  mdb000  C      CIRC  N       535.0  475.0  192.0
1  mdb001  A      CIRA  N       433.0  268.0   58.0
2  mdb002  A      CIRA  I       500.0    NaN    NaN
3  mdb003  C      CIRC  B       500.0    NaN    NaN
...
```

어떤 방법을 쓸지는 데이터에 따라 정한다.

- 행 하나라도 비면 그 행을 지운다. 데이터가 충분하고 결측이 드물 때 적합하다.
- 값이 거의 없는 열은 열 자체를 지운다.
- 평균값이나 최빈값으로 채운다. 데이터를 잃지 않지만 분포를 왜곡할 수 있다.

책은 `df['x'].fillna(..., inplace=True)`를 쓰는데, 최신 pandas에서는 열을 꺼낸 뒤 제자리 수정하는 방식이 경고를 내고 향후 동작하지 않을 수 있으므로 위처럼 결과를 다시 대입하는 편이 안전하다.

### 9.2.2 토큰화

**토큰화** 는 텍스트를 문장 단위로 나누는 **문장 토큰화** 와 단어 단위로 나누는 **단어 토큰화** 로 구분된다.

#### 문장 토큰화

마침표, 느낌표, 물음표처럼 문장 끝을 나타내는 기호를 기준으로 나눈다. 단순히 마침표마다 자르면 "Dr."나 "3.14" 같은 경우에 틀리므로, NLTK의 `sent_tokenize`는 학습된 규칙을 사용한다.

```python
from nltk import sent_tokenize
text_sample = ('Natural Language Processing, or NLP, is the process of extracting the meaning, '
               'or intent, behind human language. In the field of Conversational artificial '
               'intelligence (AI), NLP allows machines and applications to understand the intent '
               'of human language inputs, and then generate appropriate responses, resulting in '
               'a natural conversation flow.')
print(sent_tokenize(text_sample))
```

```text
['Natural Language Processing, or NLP, is the process of extracting the meaning, or intent, behind human language.',
 'In the field of Conversational artificial intelligence (AI), NLP allows machines and applications to ... flow.']
```

문장 안의 쉼표나 괄호에 흔들리지 않고 정확히 두 문장으로 나뉜다.

#### 단어 토큰화

영어는 기본적으로 띄어쓰기를 기준으로 단어를 나눈다.

```python
from nltk import word_tokenize
print(word_tokenize("This book is for deep learning learners"))
```

```text
['This', 'book', 'is', 'for', 'deep', 'learning', 'learners']
```

아포스트로피(')가 들어간 축약형은 토크나이저마다 처리가 다르다. `WordPunctTokenizer`는 알파벳·숫자 묶음과 문장 부호 묶음을 경계로 자르므로, "it's"는 `it`, `'`, `s`로, "don't"는 `don`, `'`, `t`로 나뉜다.

```python
from nltk.tokenize import WordPunctTokenizer
sentence = "it's nothing that you don't already know except most people aren't aware of how their inner world works."
print(WordPunctTokenizer().tokenize(sentence))
```

```text
['it', "'", 's', 'nothing', 'that', 'you', 'don', "'", 't', 'already', 'know', 'except',
 'most', 'people', 'aren', "'", 't', 'aware', 'of', 'how', 'their', 'inner', 'world', 'works', '.']
```

참고로 `word_tokenize`는 축약형을 `it`, `'s`와 `do`, `n't`처럼 의미 단위에 가깝게 나눈다. 어떤 토크나이저를 쓸지는 이후 단계에서 무엇이 필요한지에 따라 고른다.

#### 한국어 토큰화

한국어는 띄어쓰기만으로는 토큰을 제대로 나눌 수 없으므로 형태소 분석기를 쓴다. 네이버 영화 리뷰 데이터(`ratings_train.txt`, 탭으로 구분된 `id`, `document`, `label`)를 Okt로 분석하고, 결과로 Word2Vec 모델까지 만든다.

```python
import csv
from konlpy.tag import Okt
from gensim.models import word2vec

with open('../chap09/data/ratings_train.txt', 'r', encoding='utf-8') as f:
    rdw = list(csv.reader(f, delimiter='\t'))

twitter = Okt()
result = []
for line in rdw:
    malist = twitter.pos(line[1], norm=True, stem=True)   # 리뷰 본문 형태소 분석
    r = [word for word, tag in malist
         if tag not in ["Josa", "Eomi", "Punctuation"]]    # 조사, 어미, 문장 부호 제외
    result.append(" ".join(r).strip())
```

- `norm=True`: 정규화. "ㅋㅋㅋㅋ"나 "그랰ㅋ" 같은 비표준 표현을 다듬는다.
- `stem=True`: 어간 추출. "짜증나네요"를 "짜증나다"처럼 기본형으로 바꾼다.
- 조사(`Josa`), 어미(`Eomi`), 문장 부호(`Punctuation`)는 의미 분석에 덜 중요하므로 뺀다. 한국어에서 일종의 불용어 처리다.

```text
document
아 더빙 진짜 짜증나다 목소리
흠 포스터 보고 초딩 영화 줄 오버 연기 가볍다 않다
너 무재 밓었 다그 래서 보다 추천 다
교도소 이야기 구먼 솔직하다 재미 없다 평점 조정
...
```

첫 줄의 `document`는 파일의 헤더 행까지 처리했기 때문이다(실제로는 건너뛰어야 한다). 세 번째 줄처럼 띄어쓰기와 맞춤법이 엉망인 문장은 분석기도 제대로 나누지 못한다. 실제 데이터의 전처리가 어려운 이유다.

결과를 파일로 저장하고 Word2Vec 모델을 학습해 저장한다. 임베딩 자체는 10장에서 자세히 다룬다.

```python
with open("NaverMovie.nlp", 'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))

mData = word2vec.LineSentence("NaverMovie.nlp")   # 한 줄 = 공백으로 토큰이 구분된 한 문장
mModel = word2vec.Word2Vec(mData, vector_size=200, window=10, hs=1, min_count=2, sg=1)
mModel.save("NaverMovie.model")
```

`vector_size`는 임베딩 차원, `window`는 앞뒤로 몇 단어까지 문맥으로 볼지, `min_count`는 이보다 적게 나온 단어를 무시하는 기준, `sg=1`은 Skip-gram 방식, `hs=1`은 계층적 소프트맥스를 쓴다는 뜻이다. 언어가 달라도 "왜 토큰으로 나누는가, 어떤 단위로 나눌 것인가"만 이해하면 토큰화의 흐름은 같다.

### 9.2.3 불용어 제거

**불용어** 는 거의 모든 문장에 등장해 정보량이 적은 단어다. 남겨 두면 처리할 데이터만 늘고 분석의 초점을 흐리므로 제거한다. NLTK에는 언어별 불용어 목록이 들어 있다.

```python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

sample_text = "One of the first things that we ask ourselves is what are the pros and cons of any task we perform."
text_tokens = word_tokenize(sample_text)
stop_set = set(stopwords.words('english'))   # 조회가 빠르도록 집합으로
tokens_without_sw = [word for word in text_tokens if word not in stop_set]
print("불용어 제거 미적용:", text_tokens)
print("불용어 제거 적용:", tokens_without_sw)
```

```text
불용어 제거 미적용: ['One', 'of', 'the', 'first', 'things', 'that', 'we', 'ask', 'ourselves', 'is', 'what', 'are', 'the', 'pros', 'and', 'cons', 'of', 'any', 'task', 'we', 'perform', '.']
불용어 제거 적용: ['One', 'first', 'things', 'ask', 'pros', 'cons', 'task', 'perform', '.']
```

'of', 'the', 'we', 'is' 등이 빠지고 내용어만 남았다. 'One'이 남은 것은 NLTK 불용어 목록이 소문자로만 되어 있기 때문이다. 보통은 비교 전에 소문자로 바꾼다. 책은 반복문 안에서 매번 `stopwords.words('english')` 리스트를 새로 불러 검색하는데, 위처럼 집합으로 한 번 만들어 두면 훨씬 빠르다.

불용어 목록은 과제에 맞게 조정해야 한다. 감성 분석에서 "not"을 지우면 "not good"이 "good"이 되어 의미가 뒤집힌다.

### 9.2.4 어간 추출

**어간 추출(stemming)** 과 **표제어 추출(lemmatization)** 은 모두 단어의 여러 변형을 하나의 원형으로 모으는 작업이다. writing, writes, wrote를 write로 모으면 같은 의미의 단어가 하나의 특성으로 합쳐진다. 둘의 차이는 다음과 같다.

| | 어간 추출 | 표제어 추출 |
| --- | --- | --- |
| 방식 | 정해진 규칙으로 접미사를 잘라 냄 | 사전(WordNet)과 품사 정보로 사전상의 기본형을 찾음 |
| 품사 고려 | 하지 않음. 품사가 달라도 같은 줄기로 자름 | 함. 같은 단어도 품사에 따라 결과가 다름 |
| 결과 | 사전에 없는 형태가 나올 수 있음 (automates, automatic, automation → automat) | 항상 사전에 있는 단어 (am, are, is → be; cars, car's → car) |
| 속도와 품질 | 빠르지만 거칠다 | 느리지만 대체로 더 정확하다 |

NLTK의 대표적인 어간 추출 알고리즘은 **포터(Porter)** 와 **랭커스터(Lancaster)** 다.

```python
from nltk.stem import PorterStemmer, LancasterStemmer

words = [('obsesses', 'obsessed'), ('standardizes', 'standardization'),
         ('national', 'nation'), ('absentness', 'absently'),
         ('tribalical', 'tribalicalized')]           # 마지막 쌍은 사전에 없는 단어

for stemmer in [PorterStemmer(), LancasterStemmer()]:
    print(type(stemmer).__name__)
    for a, b in words:
        print(stemmer.stem(a), stemmer.stem(b))
```

```text
PorterStemmer
obsess obsess
standard standard
nation nation
absent absent
tribal tribalic
LancasterStemmer
obsess obsess
standard standard
nat nat
abs abs
trib trib
```

포터는 원래 단어를 알아볼 수 있는 정도로만 자른다. 랭커스터는 훨씬 공격적으로 잘라 national이 nat, absentness가 abs가 되어 의미를 알아보기 어렵다. 그래서 랭커스터는 어휘 수를 크게 줄여야 하는 특수한 상황에서나 쓰고, 일반적으로는 포터가 무난하다. 두 방법 모두 사전에 없는 단어(tribalical)에도 규칙을 적용해 결과를 낸다.

#### 표제어 추출

표제어 추출은 문법(품사)과 사전을 함께 보므로 일반적으로 어간 추출보다 결과가 좋지만 시간이 더 걸린다. NLTK에서는 `WordNetLemmatizer`를 주로 쓴다.

```python
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemma = WordNetLemmatizer()
for a, b in words:
    print(lemma.lemmatize(a), lemma.lemmatize(b))      # 품사를 주지 않으면 명사로 가정
```

```text
obsesses obsessed
standardizes standardization
national nation
absentness absently
tribalical tribalicalized
```

품사를 주지 않으면 모든 단어를 명사로 보고 처리하므로, 동사 변형인 obsesses, standardizes가 그대로 남는다. 두 번째 인자로 품사(`'n'` 명사, `'v'` 동사, `'a'` 형용사, `'r'` 부사)를 알려 주면 정확해진다.

```python
print(lemma.lemmatize('obsesses', 'v'), lemma.lemmatize('obsessed', 'a'))
print(lemma.lemmatize('standardizes', 'v'), lemma.lemmatize('standardization', 'n'))
print(lemma.lemmatize('national', 'a'), lemma.lemmatize('nation', 'n'))
```

```text
obsess obsessed
standardize standardization
national nation
```

obsesses와 standardizes가 동사 원형으로 바뀌었다. 사전에 없는 tribalical 같은 단어는 표제어 추출로는 바뀌지 않는다. 실제로는 앞의 `pos_tag` 결과를 WordNet 품사로 변환해 함께 넘기는 방식을 많이 쓴다. 단어 몇 개로는 차이가 작아 보이지만, 수천 개 단어를 처리하면 품사 정보 유무에 따른 차이가 크게 벌어진다.

(책의 코드 9-23과 9-25에는 단어 철자 오류와 `lemma` 대신 `stemmer`를 부른 줄이 있어 위에서는 철자를 바로잡고 같은 단어 목록으로 통일했다.)

### 9.2.5 정규화

여기서 말하는 **정규화(normalization)** 는 텍스트가 아니라 수치 특성의 스케일을 맞추는 작업이다. 텍스트를 벡터로 바꾼 뒤나 텍스트 외의 수치 특성을 함께 쓸 때 필요하다.

예를 들어 월 소득(MonthlyIncome)은 수천 단위, 관계 만족도(RelationshipSatisfaction)는 1~4 범위라면, 그대로 학습할 때 값이 큰 소득이 거리나 기울기 계산을 지배한다. 값이 크다고 더 중요한 특성은 아니므로, 모든 특성이 비슷한 범위를 갖도록 맞춘다.

당뇨병 데이터(`diabetes.csv`, 768행)로 확인한다. 임신 횟수, 혈당, 혈압, BMI 등 8개 독립 변수로 `Outcome`(당뇨병 여부 0/1)을 예측하는 이진 분류다.

```python
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('../chap09/data/diabetes.csv')
X = df[df.columns[:-1]].values          # 8개 특성
y = torch.tensor(df['Outcome'].values)  # 레이블
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)   # 67:33 분할
```

사이킷런의 스케일러 네 가지를 정리하면 다음과 같다.

**`MinMaxScaler`**: 각 열을 0~1로 옮긴다.

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

최솟값과 최댓값만 보므로 이상치가 하나 있으면 나머지 값이 좁은 구간에 몰린다. 이상치에 민감하다.

**`StandardScaler`**: 각 열을 평균 0, 표준편차 1로 바꾼다.

$$
x' = \frac{x - \mu}{\sigma}
$$

**`RobustScaler`**: 평균과 표준편차 대신 **중앙값(median)** 과 **사분위수 범위(IQR)** 를 쓴다. 사분위수는 데이터를 크기 순으로 정렬해 네 등분하는 값으로, $$Q_1$$은 25번째, $$Q_2$$(중앙값)는 50번째, $$Q_3$$은 75번째 백분위수다.

$$
\mathrm{IQR} = Q_3 - Q_1, \qquad x' = \frac{x - Q_2}{\mathrm{IQR}}
$$

가운데 50% 데이터의 퍼짐만 기준으로 삼으므로 이상치의 영향을 덜 받는다. `StandardScaler`와 비교하면 변환 후 대부분의 값이 더 넓게 퍼져 있다.

**`MaxAbsScaler`**: 각 열을 절댓값의 최댓값으로 나눠 -1~1 범위로 만든다. 양수만 있는 데이터라면 `MinMaxScaler`와 비슷하게 동작하고, 역시 큰 이상치에 민감하다.

사용법은 모두 같다. `fit`으로 통계(최솟값, 평균 등)를 구하고 `transform`으로 변환한다. 중요한 원칙은 **통계는 훈련 데이터에서만 구하고, 같은 변환을 테스트 데이터에 적용한다** 는 것이다.

```python
ss = StandardScaler()
X_train = ss.fit_transform(X_train)   # 훈련 데이터로 평균/표준편차 계산 + 변환
X_test = ss.transform(X_test)         # 훈련 데이터의 통계로 변환만

y_train = y_train.reshape(-1, 1)      # (n,) -> (n, 1)
y_test = y_test.reshape(-1, 1)
```

책은 테스트 데이터에도 `fit_transform`을 다시 호출하는데, 이렇게 하면 테스트 데이터의 통계가 변환에 쓰여 훈련 때와 다른 기준으로 스케일이 맞춰진다. 실제 서비스에서는 새 데이터의 통계를 미리 알 수 없으므로 위처럼 `transform`만 쓰는 것이 맞다. 또 책은 레이블에도 `MinMaxScaler`를 적용하지만, 레이블이 이미 0과 1이라 변환 결과가 그대로이므로 생략해도 된다.

PyTorch의 `Dataset`과 `DataLoader`로 데이터를 감싼다. 이번 데이터는 작아서 꼭 필요하지는 않지만, 배치 분할과 셔플을 자동으로 해 주고 데이터가 커져도 같은 코드를 쓸 수 있다는 장점이 있다.

```python
class customdataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
        self.len = len(self.X)
    def __getitem__(self, index):      # index번째 샘플 하나
        return self.X[index], self.y[index]
    def __len__(self):                 # 전체 샘플 수
        return self.len

train_data = customdataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_data = customdataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
```

모델은 은닉층 두 개에 배치 정규화와 드롭아웃을 넣은 이진 분류기다.

```python
class binaryClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(8, 64)      # 입력 특성 8개
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1)    # 출력 1개(logit)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, inputs):
        x = self.batchnorm1(self.relu(self.layer_1(inputs)))
        x = self.batchnorm2(self.relu(self.layer_2(x)))
        x = self.dropout(x)
        return self.layer_out(x)

model = binaryClassification().to(device)
BCE = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
```

- 이진 분류의 손실 함수로는 `nn.BCELoss`(이진 교차 엔트로피)와 `nn.BCEWithLogitsLoss`가 있다. 후자는 시그모이드와 BCE를 한 번에 계산하므로 모델 출력에 시그모이드를 넣지 않고 logit을 그대로 넘긴다. 두 단계를 따로 하는 것보다 수치적으로도 안정적이다.
- 그래서 정확도를 잴 때는 직접 시그모이드를 씌우고 반올림해 0/1 예측을 만든다.

```python
def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))   # 0.5 기준으로 0 또는 1
    correct = (y_pred_tag == y_test).sum().float()
    return torch.round(correct / y_test.shape[0] * 100)   # 백분율로 반올림
```

학습 루프는 에폭마다 훈련 데이터로 가중치를 갱신하고, 100 에폭마다 훈련·테스트 손실과 정확도를 출력한다.

```python
for epoch in range(1001):
    iteration_loss, iteration_accuracy = 0., 0.
    model.train()
    for i, (X, y) in enumerate(train_loader):
        y_pred = model(X.float().to(device))
        y = y.to(device)
        loss = BCE(y_pred, y.reshape(-1, 1).float())
        iteration_loss += loss.item()
        iteration_accuracy += accuracy(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 100 == 0:
        print('Train: epoch: {0} - loss: {1:.5f}; acc: {2:.3f}'.format(
            epoch, iteration_loss / (i + 1), iteration_accuracy / (i + 1)))

    model.eval()
    # with torch.no_grad(): test_loader에 대해 같은 방식으로 손실과 정확도 계산
```

책의 코드는 손실 텐서 자체(`loss`)를 누적해 계산 그래프가 계속 붙어 다니고, 평가 때 `torch.no_grad()`를 쓰지 않는다. 위처럼 `loss.item()`으로 숫자만 누적하고, 평가는 `torch.no_grad()` 안에서 하는 것이 메모리와 속도 면에서 낫다.

```text
Train: epoch: 0 - loss: 0.68347; acc: 61.889
Test: epoch: 0 - loss: 0.66168; acc: 68.750
Train: epoch: 100 - loss: 0.44345; acc: 77.667
Test: epoch: 100 - loss: 0.50527; acc: 74.500
...
Train: epoch: 1000 - loss: 0.30108; acc: 87.444
Test: epoch: 1000 - loss: 0.55002; acc: 74.500
```

훈련 정확도는 62%에서 87%까지 오르지만 테스트 정확도는 100 에폭 이후 73~76%에서 멈추고, 테스트 손실은 오히려 조금씩 늘어난다. 8장에서 본 과적합의 전형적인 모습이다. 데이터가 768행으로 적은 것이 주된 이유이며, 데이터가 더 많다면 정규화와 규제의 효과가 더 분명하게 드러난다.

## 정리

- 자연어 처리는 입력 → 전처리 → 임베딩 → 모델링 순서로 진행되며, 이 장은 전처리를 다룬다. 핵심 용어는 말뭉치, 토큰, 토큰화, 불용어, 어간 추출, 품사 태깅이다.
- NLTK는 영어 토큰화·품사 태깅·불용어·어간 추출을, KoNLPy는 한국어 형태소 분석(Komoran, Okt 등)을, Gensim은 Word2Vec 등 임베딩을, 사이킷런은 `CountVectorizer`·`TfidfVectorizer` 같은 특성 추출을 맡는다.
- 결측치는 `isnull().sum()`으로 찾고 `dropna()`(삭제)나 `fillna()`(0, 평균, 최빈값으로 대체)로 처리한다.
- 토큰화는 문장 토큰화(`sent_tokenize`)와 단어 토큰화(`word_tokenize`, `WordPunctTokenizer`)로 나뉜다. 한국어는 어절에 조사와 어미가 붙으므로 형태소 분석기로 나누고, 조사·어미·문장 부호를 걸러 낸다.
- 불용어는 NLTK의 언어별 목록으로 제거하되, 과제에 따라 "not"처럼 남겨야 할 단어가 있는지 확인한다.
- 어간 추출(포터, 랭커스터)은 규칙으로 접미사를 잘라 빠르지만 사전에 없는 형태가 나오고, 표제어 추출(`WordNetLemmatizer`)은 품사와 사전을 써서 정확하지만 느리다. 표제어 추출에는 품사 정보를 함께 넘겨야 제대로 동작한다.
- 수치 특성은 `MinMaxScaler`, `StandardScaler`, `RobustScaler`(이상치에 강함), `MaxAbsScaler`로 스케일을 맞춘다. 통계는 훈련 데이터에서만 `fit`하고 테스트 데이터에는 `transform`만 적용한다.
- 이진 분류에서는 `nn.BCEWithLogitsLoss`에 logit을 그대로 넘기고, 예측 레이블은 시그모이드 후 0.5 기준으로 만든다.

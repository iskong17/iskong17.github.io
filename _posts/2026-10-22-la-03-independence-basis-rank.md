---
title: "Linear Algebra 3: Linear Independence, Basis, and Rank"
date: 2026-10-22 10:03:00 +0900
category: study
topic: linear-algebra
lang: ko
description: 선형 독립과 그것이 해의 유일성을 결정하는 이유, 부분 공간과 기저, 그리고 부분 공간의 차원, 열 공간, 행렬의 랭크를 정리한다.
---

> [선형대수 강의](https://www.youtube.com/playlist?list=PLVuwEl3g6OcIF-EEcerd5m5-Vw5jDpKN1) 11–14강 정리.
> [KAIST DAVIAN Lab 기초 과목 스터디](https://davian.kaist.ac.kr/joinus)의 선형대수 강의다.
> 이전: [Linear Algebra 2]({% post_url 2026-10-22-la-02-span %}) · 다음: [Linear Algebra 4]({% post_url 2026-10-22-la-04-linear-transformation %})

## 선형 독립

[▶ 11강](https://www.youtube.com/watch?v=C2hArJwiQ7g)

### 동기: 해가 하나인가, 무수히 많은가

[Linear Algebra 2]({% post_url 2026-10-22-la-02-span %})에서 $$A\mathbf{x} = \mathbf{b}$$는 $$\mathbf{b}$$가 $$A$$의 열들의 span 안에 있을 때에만 해를 갖는다는 것을 보았다. 이제 $$\mathbf{b}$$가 span 안에 있어서 해가 존재한다고 가정한다. 다음 질문은 그 해가 **유일한지**, 아니면 **무수히 많은지**이다. 답은 열들이 **선형 독립(linearly independent)**인지에 달려 있다.

- 열들이 선형 독립이면 해는 유일하다.
- 열들이 **선형 종속(linearly dependent)**이면 해가 무수히 많다.

### 직관적인 정의: 벡터를 하나씩 추가하기

같은 공간 $$\mathbb{R}^n$$에 있는 벡터 $$\mathbf{v}_1, \dots, \mathbf{v}_p$$를 생각한다. 이 벡터들을 span의 재료로 하나씩 추가하면서, 지금까지 만들어진 span을 따라간다. 매 단계에서 새 벡터가 이전 벡터들의 span 안에 이미 들어 있는지 확인한다.

- 끝까지 어떤 새 벡터도 이전 벡터들의 span 안에 들어 있지 않으면 **선형 독립**이다.
- 어느 시점에 어떤 벡터가 이전 벡터들의 span 안에 들어 있으면 **선형 종속**이다.

**예시(독립).** $$\mathbf{v}_1 = (1, 1, 0)$$, $$\mathbf{v}_2 = (1, 0, 1)$$, $$\mathbf{v}_3 = (0, 1, 1)$$.

- $$\operatorname{Span}\{\mathbf{v}_1\}$$은 직선이다. $$\mathbf{v}_2$$는 $$\mathbf{v}_1$$의 스칼라배가 아니므로 직선 밖으로 벗어난다.
- $$\operatorname{Span}\{\mathbf{v}_1, \mathbf{v}_2\}$$는 평면이다. $$\mathbf{v}_3 = a\mathbf{v}_1 + b\mathbf{v}_2$$라고 쓰려면 $$a + b = 0$$, $$a = 1$$, $$b = 1$$이어야 하는데 이는 불가능하다. 따라서 $$\mathbf{v}_3$$는 평면 밖에 있다.

어떤 벡터도 이전 span에 들어가지 않았으므로 이 집합은 선형 독립이다.

**예시(종속).** $$\mathbf{v}_3$$ 대신 $$\mathbf{w} = (2, 1, 1) = \mathbf{v}_1 + \mathbf{v}_2$$를 쓰면, $$\mathbf{w}$$는 $$\mathbf{v}_1, \mathbf{v}_2$$가 span하는 평면 안에 있다. 따라서 $$\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{w}\}$$는 선형 종속이다.

종속은 더 일찍 나타날 수도 있다. $$\mathbf{v}_2$$가 이미 $$\mathbf{v}_1$$의 스칼라배이면(예: $$\mathbf{v}_2 = 2\mathbf{v}_1$$), 세 번째 벡터를 볼 필요도 없이 집합은 종속이다.

4차원이나 100차원에서도 같은 절차가 적용된다. 그림으로 그릴 수는 없지만, 어떤 벡터가 span 안에 있는지는 **가우스 소거법(Gaussian elimination)**으로 기계적으로 확인할 수 있다.

### 형식적인 정의

벡터 방정식

$$
x_1 \mathbf{v}_1 + x_2 \mathbf{v}_2 + \cdots + x_p \mathbf{v}_p = \mathbf{0}
$$

이 **자명해(trivial solution)** $$x_1 = x_2 = \cdots = x_p = 0$$ 하나만 가질 때 $$\mathbf{v}_1, \dots, \mathbf{v}_p$$는 **선형 독립**이다. 적어도 하나의 $$x_j$$가 0이 아닌 해, 즉 **비자명해(nontrivial solution)**가 있으면 **선형 종속**이다.

이것은 우변이 $$\mathbf{b} = \mathbf{0}$$인 벡터 방정식이다. 가중치를 모두 0으로 둔 자명해는 항상 성립하므로 해는 언제나 존재한다. 문제는 그 *외의* 해가 있느냐이다.

앞의 종속 예시에서는 $$1 \cdot \mathbf{v}_1 + 1 \cdot \mathbf{v}_2 - 1 \cdot \mathbf{w} = \mathbf{0}$$이 비자명해다.

### 두 정의가 같은 이유

비자명해가 있다고 하자. 0이 아닌 가중치 중 **마지막** 것을 $$x_j$$라 하면 $$x_{j+1} = \cdots = x_p = 0$$이다. 방정식은

$$
x_1 \mathbf{v}_1 + \cdots + x_{j-1}\mathbf{v}_{j-1} + x_j \mathbf{v}_j = \mathbf{0}
$$

이 된다. 나머지 항을 우변으로 옮기고 $$x_j \neq 0$$으로 나누면

$$
\mathbf{v}_j = -\frac{x_1}{x_j}\mathbf{v}_1 - \cdots - \frac{x_{j-1}}{x_j}\mathbf{v}_{j-1}
$$

이다. 즉 $$\mathbf{v}_j$$는 앞선 벡터들의 선형 결합이고, 그들의 span 안에 있다. 예를 들어 마지막으로 0이 아닌 가중치가 $$x_{37} = 3$$이면, $$\mathbf{v}_{37}$$은 $$\mathbf{v}_1, \dots, \mathbf{v}_{36}$$을 가중치 $$-x_i / 3$$로 결합한 것이다. 반대로 어떤 $$\mathbf{v}_j = c_1\mathbf{v}_1 + \cdots + c_{j-1}\mathbf{v}_{j-1}$$이면, $$\mathbf{v}_j$$를 좌변으로 옮겨 $$\mathbf{v}_j$$의 가중치가 $$-1$$인 비자명해를 얻는다. 따라서 두 정의는 같다.

## 선형 종속의 기하학적 이해

[▶ 12강](https://www.youtube.com/watch?v=Wp6AcmIZLhc)

### 독립인 벡터: 평행사변형은 하나

선형 독립인 두 벡터 $$\mathbf{v}_1, \mathbf{v}_2$$와 그 span(평면) 안의 점 $$\mathbf{b}$$를 생각한다. $$x_1\mathbf{v}_1 + x_2\mathbf{v}_2 = \mathbf{b}$$를 푸는 것은 평행사변형의 맞은편 꼭짓점이 $$\mathbf{b}$$에 오도록 변 $$\mathbf{v}_1$$, $$\mathbf{v}_2$$를 늘이는 것이다. 그런 평행사변형은 **단 하나**뿐이므로 해는 유일하다.

### 중복된 벡터: 평행사변형이 여러 개

이제 같은 평면 안에 있는 세 번째 벡터 $$\mathbf{v}_3$$를 추가한다. $$\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$$는 종속이다. 평면을 span하는 데 벡터는 두 개면 충분하므로 $$\mathbf{v}_3$$는 중복이다. span은 여전히 같은 평면이다.

이제 점 $$\mathbf{b}$$에 도달하는 방법이 여러 가지가 된다. $$\mathbf{v}_3$$의 가중치를 0으로 두고 $$\mathbf{v}_1, \mathbf{v}_2$$만 써도 되고, $$\mathbf{v}_1$$과 $$\mathbf{v}_3$$로 평행사변형을 만들어도 되고, 셋을 섞어도 된다. 선택마다 가중치가 달라진다.

**예시.** $$\mathbb{R}^2$$에서 $$\mathbf{v}_1 = (1, 0)$$, $$\mathbf{v}_2 = (0, 1)$$, $$\mathbf{v}_3 = (1, 1) = \mathbf{v}_1 + \mathbf{v}_2$$, $$\mathbf{b} = (3, 2)$$라 하자.

- $$3\mathbf{v}_1 + 2\mathbf{v}_2 + 0\mathbf{v}_3 = (3, 2)$$
- $$1\mathbf{v}_1 + 0\mathbf{v}_2 + 2\mathbf{v}_3 = (3, 2)$$
- 일반적으로 모든 $$t \in \mathbb{R}$$에 대해 $$(3 - t)\mathbf{v}_1 + (2 - t)\mathbf{v}_2 + t\mathbf{v}_3 = (3, 2)$$

해 $$\mathbf{x} = (3 - t, 2 - t, t)$$가 무수히 많다. $$\mathbf{x} = (3, 2, 0) + t(-1, -1, 1)$$이고, $$(-1, -1, 1)$$은 바로 $$x_1\mathbf{v}_1 + x_2\mathbf{v}_2 + x_3\mathbf{v}_3 = \mathbf{0}$$의 비자명해다. 해 하나에 이것의 스칼라배를 더하면 또 다른 해가 된다.

### 결론

$$\mathbf{b}$$가 $$A$$의 열들의 span 안에 있어서 $$A\mathbf{x} = \mathbf{b}$$의 해가 존재한다고 하자.

- 열들이 **선형 독립**이면 span이 중복 없이 최소 개수의 벡터로 만들어진다. $$\mathbf{b}$$에 도달하는 평행사변형(평행육면체)은 하나뿐이고, 해도 유일하다.
- 열들이 **선형 종속**이면 어떤 열은 중복이고, 같은 점에 여러 방법으로 도달할 수 있어 해가 **무수히 많다**.

[Linear Algebra 1]({% post_url 2026-10-22-la-01-linear-systems %})의 과소결정 수명 예측 예시가 그런 경우다. 흡연 열과 미세먼지 열이 같아서 열들이 종속이었고, 해가 직선 하나를 이루었다.

## 부분 공간과 기저

[▶ 13강](https://www.youtube.com/watch?v=GTq8oRe675w)

### 닫혀 있음

어떤 집합의 원소들에 연산을 적용한 결과가 항상 다시 그 집합의 원소이면, 그 집합은 그 연산에 대해 **닫혀 있다(closed)**고 한다. 자연수는 덧셈에 닫혀 있지만($$2 + 3 = 5$$) 뺄셈에는 닫혀 있지 않다($$2 - 3 = -1$$은 자연수가 아니다).

### 부분 공간

$$\mathbb{R}^n$$의 **부분 공간(subspace)** $$H$$는 **선형 결합에 닫혀 있는** 부분집합이다. 즉 임의의 $$\mathbf{u}, \mathbf{v} \in H$$와 임의의 스칼라 $$c, d$$에 대해 $$c\mathbf{u} + d\mathbf{v}$$도 $$H$$에 속한다. $$c = d = 0$$으로 두면 부분 공간은 항상 영벡터를 포함한다는 것을 알 수 있다.

**반례.** 벡터 두 개로 된 집합 $$\{(1, 2, 3), (4, 5, 6)\}$$은 부분 공간이 아니다. $$(1, 2, 3) + (4, 5, 6) = (5, 7, 9)$$가 이 집합에 없기 때문이다.

**예시.** 모든 span은 부분 공간이다. $$\mathbf{u} = a_1\mathbf{v}_1 + a_2\mathbf{v}_2$$, $$\mathbf{w} = b_1\mathbf{v}_1 + b_2\mathbf{v}_2$$이면

$$
c\mathbf{u} + d\mathbf{w} = (ca_1 + db_1)\mathbf{v}_1 + (ca_2 + db_2)\mathbf{v}_2
$$

로 다시 $$\mathbf{v}_1, \mathbf{v}_2$$의 선형 결합이다. 따라서 $$\operatorname{Span}\{\mathbf{v}_1, \mathbf{v}_2\}$$는 닫혀 있다. 거꾸로 $$\mathbb{R}^n$$의 모든 부분 공간은 유한 개 벡터의 span이다. 실용적으로는 "부분 공간"과 "어떤 벡터들의 span"을 같은 것으로 생각해도 된다. 기하학적으로 $$\mathbb{R}^3$$의 부분 공간은 $$\{\mathbf{0}\}$$, 원점을 지나는 직선, 원점을 지나는 평면, 그리고 $$\mathbb{R}^3$$ 자신이다.

### 기저

이번에는 반대 방향이다. 원점을 지나는 $$\mathbb{R}^3$$의 평면 같은 부분 공간 $$H$$가 주어질 때, 그것을 span하는 적은 수의 벡터를 찾는다. $$H$$의 **기저(basis)**는 다음 두 조건을 만족하는 벡터들의 집합이다.

1. $$H$$를 정확히 **span**한다.
2. **선형 독립**이다(중복이 없다).

**예시.** $$H = \operatorname{Span}\{(1, 2, 3), (4, 5, 6), (5, 7, 9)\}$$라 하자. 벡터를 하나씩 추가해 본다. $$(1, 2, 3)$$과 $$(4, 5, 6)$$은 서로의 스칼라배가 아니므로 둘 다 필요하다. 그런데 $$(5, 7, 9) = (1, 2, 3) + (4, 5, 6)$$은 둘의 span 안에 있어서, 추가해도 span이 전혀 넓어지지 않는다. 따라서 $$H$$는 평면이고, $$\{(1, 2, 3), (4, 5, 6)\}$$이 $$H$$의 기저다.

### 기저는 유일하지 않지만 개수는 같다

원점을 지나는 평면에서 영이 아닌 벡터 하나를 아무렇게나 고르고, 그와 같은 직선 위에 있지 않은 평면 안의 벡터를 하나 더 고르면 두 벡터가 기저가 된다. 다른 방향을 가리키는 또 다른 한 쌍도 기저가 된다. 따라서 기저는 **유일하지 않다**.

하지만 이 평면의 기저는 어느 것이나 정확히 **두 개**의 벡터로 이루어진다. 일반적으로 부분 공간 $$H$$의 모든 기저는 같은 개수의 벡터를 갖는다.

## 차원과 랭크

[▶ 14강](https://www.youtube.com/watch?v=dbtBvhZJdD0)

### 차원

부분 공간 $$H$$의 모든 기저가 같은 개수의 벡터를 가지므로, 그 개수는 $$H$$ 자체의 성질이다. $$H$$의 기저에 들어 있는 벡터의 개수를 $$H$$의 **차원(dimension)** $$\dim H$$라 한다. (영 부분 공간 $$\{\mathbf{0}\}$$의 차원은 0이다.) 원점을 지나는 직선은 1차원, 원점을 지나는 평면은 2차원이다.

이는 일상적으로 말하는 "3차원"과 일치한다. $$\mathbb{R}^3$$의 가장 큰 부분 공간은 $$\mathbb{R}^3$$ 자신이다. 3차원 벡터를 아무렇게나 선형 결합해도 3차원 벡터가 나오기 때문이다. 기저의 하나는 **표준 기저(standard basis)**다.

$$
\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

$$\mathbb{R}^3$$의 모든 벡터는 이들의 선형 결합으로 유일하게 표현된다. 예를 들어 $$(2, -1, 5) = 2\mathbf{e}_1 - 1\mathbf{e}_2 + 5\mathbf{e}_3$$이다. 다른 기저도 많지만 모두 벡터가 3개이므로 $$\dim \mathbb{R}^3 = 3$$이다. 같은 방식으로 표준 기저의 벡터가 $$n$$개이므로 $$\dim \mathbb{R}^n = n$$이다.

### 열 공간

열이 $$\mathbf{a}_1, \dots, \mathbf{a}_n$$인 행렬 $$A$$의 **열 공간(column space)**은 열들의 span이다.

$$
\operatorname{Col} A = \operatorname{Span}\{\mathbf{a}_1, \dots, \mathbf{a}_n\}
$$

span이므로 부분 공간이다. 열 공간은 가능한 모든 $$A\mathbf{x}$$의 집합과 같으므로, $$A\mathbf{x} = \mathbf{b}$$는 $$\mathbf{b} \in \operatorname{Col} A$$일 때에만 해를 갖는다.

### 랭크

$$A$$의 **랭크(rank)**는 열 공간의 차원이다.

$$
\operatorname{rank} A = \dim \operatorname{Col} A
$$

랭크를 구하려면 다른 열들의 선형 결합인 열을 중복이 없어질 때까지 제거한다. 그래도 span은 변하지 않고, 남은 열의 개수가 랭크다.

**예시.**

$$
A = \begin{bmatrix} 1 & 4 & 5 \\ 2 & 5 & 7 \\ 3 & 6 & 9 \end{bmatrix}
$$

처음 두 열은 독립이고, 세 번째 열은 두 열의 합이다. 따라서 $$\operatorname{Col} A = \operatorname{Span}\{\mathbf{a}_1, \mathbf{a}_2\}$$는 평면이고, 열이 세 개인데도 $$\operatorname{rank} A = 2$$이다.

일반적으로 열이 10개인 행렬에서 4개의 열이 나머지 (서로 독립인) 6개 열의 선형 결합이면 랭크는 6이다.

### 랭크와 행렬 분해

열은 4개이지만 그중 독립인 열이 2개뿐인 행렬을 보자.

$$
A = \begin{bmatrix} 1 & 0 & 1 & 2 \\ 0 & 1 & 1 & 3 \\ 1 & 1 & 2 & 5 \end{bmatrix}
$$

$$\mathbf{a}_3 = \mathbf{a}_1 + \mathbf{a}_2$$, $$\mathbf{a}_4 = 2\mathbf{a}_1 + 3\mathbf{a}_2$$이므로 $$\operatorname{rank} A = 2$$이다. [Linear Algebra 2]({% post_url 2026-10-22-la-02-span %})의 열 관점 행렬 곱셈에 따르면, $$A$$의 모든 열이 $$\mathbf{a}_1, \mathbf{a}_2$$의 결합이므로

$$
A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
\begin{bmatrix} 1 & 0 & 1 & 2 \\ 0 & 1 & 1 & 3 \end{bmatrix}
$$

이다. 왼쪽 인수는 $$\operatorname{Col} A$$의 기저를, 오른쪽 인수는 가중치를 담는다. 랭크는 왼쪽 인수에 필요한 열의 개수다.

Linear Algebra 2의 외적 $$\mathbf{u}\mathbf{v}^T$$는 극단적인 경우다. 열이 몇 개든 모두 $$\mathbf{u}$$의 스칼라배이므로 기저 벡터 하나로 충분하고, 랭크는 1이다. 외적 하나로 만든 행렬은 아무리 커도 랭크가 1이다.

## 정리

- 벡터들은 어느 벡터도 앞선 벡터들의 span 안에 있지 않을 때 선형 독립이다. 동치로, $$x_1\mathbf{v}_1 + \cdots + x_p\mathbf{v}_p = \mathbf{0}$$이 자명해만 가질 때 선형 독립이다.
- $$\mathbf{b}$$가 열들의 span 안에 있을 때, 열들이 독립이면 $$A\mathbf{x} = \mathbf{b}$$의 해는 유일하고, 종속이면 무수히 많다.
- 기하학적으로 중복된 벡터가 있으면 같은 점을 서로 다른 여러 평행사변형으로 만들 수 있다. 그래서 종속이면 해가 무수히 많다.
- 부분 공간은 선형 결합에 닫혀 있는 집합이다. 모든 span은 부분 공간이고, 모든 부분 공간은 어떤 벡터들의 span이다.
- 기저는 부분 공간을 중복 없이 span하는 벡터들이다. 기저는 유일하지 않지만 벡터의 개수는 항상 같고, 그 개수가 차원이다.
- 열 공간 $$\operatorname{Col} A$$는 열들의 span이고, $$\operatorname{rank} A = \dim \operatorname{Col} A$$는 독립인 열의 개수다. 랭크가 $$r$$인 행렬은 $$m \times r$$ 기저 행렬과 $$r \times n$$ 가중치 행렬의 곱으로 분해된다.

---
title: "Algorithms 2: Data Structures and Dynamic Arrays"
date: 2026-10-22 14:02:00 +0900
category: study
topic: algorithm
series: algo
lecture: 2
card_title: "Lecture 2: Data Structures and Dynamic Arrays"
card_desc: "Sequence and Set interfaces · Array · Linked list · Dynamic array · Amortized analysis"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/79a07dc1cb47d76dae2ffedc701e3d2b_MIT6_006S20_lec2.pdf"
description: "시퀀스와 집합 인터페이스를 정의하고, 배열, 연결 리스트, 동적 배열로 시퀀스를 구현하며, 동적 배열의 끝 삽입이 분할상환 O(1)임을 보인다."
---

<ul class="post-sources">
<li markdown="span">[MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 2강 정리</li>
<li markdown="span">강의: Erik Demaine, Jason Ku, Justin Solomon</li>
<li markdown="span">MIT OpenCourseWare, CC BY-NC-SA</li>
<li markdown="span">[▶ 강의 영상](https://www.youtube.com/watch?v=CHhwJjR0mZA)</li>
<li markdown="span">[Recitation 2 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/c08a3b63dfe5f6f6b32257d35f86ae63_MIT6_006S20_r02.pdf)</li>
</ul>

## 자료구조 인터페이스

자료구조는 데이터를 저장하는 방법이며, 그 데이터에 대한 연산을 지원하는 알고리즘을 함께 가진다. 지원하는 연산의 모음을 **인터페이스(interface)** 라 하고, API나 ADT(abstract data type)라고도 부른다.

- **인터페이스는 명세(specification)** 다. 어떤 연산을 지원하는지 정한다. 즉 **문제**다.
- **자료구조는 표현(representation)** 이다. 그 연산을 어떻게 지원하는지 정한다. 즉 **해법**이다.

이 과목의 두 핵심 인터페이스는 시퀀스(Sequence)와 집합(Set)이다.

### 시퀀스 인터페이스

**시퀀스(sequence)** 는 항목의 순서를 유지한다. 이 순서는 **외재적(extrinsic)** 이다. 첫 항목이 첫째인 이유는 항목 자체가 무엇인지가 아니라 외부에서 그 자리에 두었기 때문이다. 항목은 $$(x_0, x_1, x_2, \ldots, x_{n-1})$$처럼 0부터 번호를 매기며, $$n$$은 저장된 항목 수다.

| 분류 | 연산 | 설명 |
|---|---|---|
| 컨테이너 | `build(X)` | 반복 가능한 $$X$$의 항목으로 시퀀스를 만든다 |
| | `len()` | 저장된 항목 수를 반환한다 |
| 정적 | `iter_seq()` | 저장된 항목을 시퀀스 순서대로 하나씩 반환한다 |
| | `get_at(i)` | $$i$$번째 항목을 반환한다 |
| | `set_at(i, x)` | $$i$$번째 항목을 $$x$$로 바꾼다 |
| 동적 | `insert_at(i, x)` | $$x$$를 $$i$$번째 항목으로 추가한다 |
| | `delete_at(i)` | $$i$$번째 항목을 제거하고 반환한다 |
| | `insert_first(x)` | $$x$$를 첫 항목으로 추가한다 |
| | `delete_first()` | 첫 항목을 제거하고 반환한다 |
| | `insert_last(x)` | $$x$$를 마지막 항목으로 추가한다 |
| | `delete_last()` | 마지막 항목을 제거하고 반환한다 |

![Sequence interface operations grouped as container, static and dynamic](/images/study/algo/algo-02-data-structures-1.webp)
*강의 노트의 시퀀스 인터페이스 표. 연산이 컨테이너, 정적, 동적의 세 묶음으로 나뉘어 있다. 출처: MIT 6.006 Lecture 2 notes (CC BY-NC-SA 4.0).*

표의 세 묶음은 연산이 자료구조에 무엇을 요구하는지에 따라 나뉜다. **컨테이너** 연산 `build`와 `len`은 자료구조를 만들고 크기를 묻는다. **정적** 연산은 항목의 개수와 위치를 바꾸지 않고 읽거나 덮어쓰기만 한다. **동적** 연산은 항목을 넣거나 빼서 시퀀스의 길이를 바꾼다. 동적 연산을 하면 바뀐 위치 뒤의 모든 항목의 순위(index)가 하나씩 밀리거나 당겨진다는 점이 중요하다. 배열에서 이것이 곧 항목 이동 비용으로 이어진다.

동적 연산은 다시 위치에 따라 `_at(i)`(임의 위치), `_first`(맨 앞), `_last`(맨 뒤)로 나뉜다. 이 구분은 이후 비교표의 열과 그대로 대응한다. 어떤 자료구조는 맨 앞에서만, 어떤 자료구조는 맨 뒤에서만 빠르기 때문이다.

특수한 경우로, **스택(stack)** 은 `insert_last(x)`와 `delete_last()`만, **큐(queue)** 는 `insert_last(x)`와 `delete_first()`만 지원한다.

### 집합 인터페이스

시퀀스가 외재적 순서에 관한 것이라면, **집합(set)** 은 **내재적(intrinsic)** 순서에 관한 것이다. 집합은 고유한 키를 가진 항목들을 유지한다(항목 $$x$$의 키는 `x.key`). 지금은 키가 중복되지 않는 경우(멀티셋이 아닌 경우)만 다룬다. 항목 자체를 키로 쓰는 경우가 많지만, 키 외에 더 많은 정보를 함께 저장할 수도 있다.

| 분류 | 연산 | 설명 |
|---|---|---|
| 컨테이너 | `build(X)` | 반복 가능한 $$X$$의 항목으로 집합을 만든다 |
| | `len()` | 저장된 항목 수를 반환한다 |
| 정적 | `find(k)` | 키가 $$k$$인 항목을 반환한다 |
| 동적 | `insert(x)` | $$x$$를 추가한다(같은 키의 항목이 있으면 교체) |
| | `delete(k)` | 키가 $$k$$인 항목을 제거하고 반환한다 |
| 순서 | `iter_ord()` | 저장된 항목을 키 순서대로 하나씩 반환한다 |
| | `find_min()` | 키가 가장 작은 항목을 반환한다 |
| | `find_max()` | 키가 가장 큰 항목을 반환한다 |
| | `find_next(k)` | 키가 $$k$$보다 큰 것 중 가장 작은 항목을 반환한다 |
| | `find_prev(k)` | 키가 $$k$$보다 작은 것 중 가장 큰 항목을 반환한다 |

![Set interface operations grouped as container, static, dynamic and order](/images/study/algo/algo-02-data-structures-2.webp)
*강의 노트의 집합 인터페이스 표. 시퀀스 표의 컨테이너, 정적, 동적 묶음에 키 순서를 다루는 순서(Order) 묶음이 더해져 있다. 출처: MIT 6.006 Lecture 2 notes (CC BY-NC-SA 4.0).*

시퀀스 표와 나란히 놓고 보면 차이가 분명하다. 시퀀스의 연산은 위치 $$i$$를 인자로 받지만, 집합의 연산은 키 $$k$$를 받는다. 집합의 `insert(x)`에는 위치 인자가 없다. 항목이 어디에 놓일지는 키가 정하기 때문이다. 정적 묶음에는 `find(k)` 하나만 있고, 동적 묶음은 `insert`와 `delete` 두 개뿐이다.

마지막 **순서** 묶음은 키들 사이의 대소 관계를 묻는다. `find_next(k)`와 `find_prev(k)`는 $$k$$ 자체가 집합에 없어도 동작한다는 점에 주의한다. 예를 들어 키가 $$\{3, 7, 10\}$$일 때 `find_next(5)`는 7을 반환한다. 해시 테이블(4강)처럼 순서 묶음을 효율적으로 지원하지 못하는 자료구조가 있기 때문에, 이 묶음을 뺀 인터페이스를 따로 사전이라 부른다.

순서 연산을 뺀 집합을 **사전(dictionary)** 이라 한다. 해당하는 항목이 없으면 find 계열 연산은 None을 반환한다.

Recitation에서는 시퀀스 자료구조만으로 집합을 구현하는 연습을 한다. 예를 들어 `find(k)`는 모든 항목을 훑으면 되고, `insert(x)`는 같은 키를 찾아 `set_at`으로 바꾸거나 없으면 `insert_last`로 붙이면 된다. 효율적이지는 않지만 인터페이스와 자료구조의 분리를 잘 보여 준다.

## 배열 시퀀스

배열은 정적 연산에 아주 좋다. `get_at(i)`와 `set_at(i, x)`가 $$\Theta(1)$$이다. 하지만 동적 연산에는 좋지 않다. 일관성을 위해 배열은 항상 꽉 차 있다는 불변식(invariant)을 유지하면, 항목을 삽입하거나 제거할 때마다 다음이 필요하다.

- 배열을 새로 할당(reallocate)하고,
- 바뀐 위치 뒤의 모든 항목을 옮긴다.

Recitation의 구현은 새 배열을 만들고 앞뒤 부분을 복사한다.

```python
def insert_at(self, i, x):                   # O(n)
    n = len(self)
    A = [None] * (n + 1)
    self._copy_forward(0, i, A, 0)
    A[i] = x
    self._copy_forward(i, n - i, A, i + 1)
    self.build(A)
```

따라서 모든 동적 연산이 $$\Theta(n)$$이다.

## 연결 리스트 시퀀스

**연결 리스트(linked list)** 는 포인터 기반 자료구조다(Python의 "list"와는 무관하다). 각 항목은 **노드(node)** 에 저장되며, 노드는 두 필드 `node.item`과 `node.next`를 가진다. `next`는 시퀀스에서 다음 노드를 가리키는 포인터다. 리스트는 첫 노드인 **헤드(head)** 를 가리키는 포인터(와 크기)를 유지한다.

노드는 포인터만 다시 이으면(relink) 조작할 수 있으므로, 앞에서 삽입하고 삭제하는 연산이 $$\Theta(1)$$이 된다.

```python
def insert_first(self, x):                   # O(1)
    new_node = Linked_List_Node(x)
    new_node.next = self.head
    self.head = new_node
    self.size += 1

def delete_first(self):                      # O(1)
    x = self.head.item
    self.head = self.head.next
    self.size -= 1
    return x
```

대신 $$i$$번째 항목을 찾으려면 헤드부터 하나씩 따라가야 하므로 `get_at(i)`와 `set_at(i, x)`는 $$O(n)$$이다. 뒤쪽 삽입/삭제를 효율적으로 하는 것도 가능하다(꼬리 포인터 등, PS1에서 다룬다).

두 세계의 장점을 모두 얻을 수 있는가? (어느 정도는) 그렇다.

## 동적 배열 시퀀스

**동적 배열(dynamic array)** 은 배열을 마지막 위치의 동적 연산에 효율적으로 만든다. Python의 "list"가 동적 배열이다.

**아이디어:** 여분의 공간을 할당해 두어, 동적 연산마다 재할당이 일어나지 않게 한다.

- **채움 비율(fill ratio)** $$0 \le r \le 1$$은 항목 수와 공간의 비율이다.
- 배열이 꽉 차면($$r = 1$$) 끝에 $$\Theta(n)$$의 여분 공간을 할당해 채움 비율을 $$r_i$$(예: $$1/2$$)로 만든다. 즉 크기를 두 배로 늘리는 **테이블 두 배 늘리기(table doubling)** 다.
- 그러면 다음 재할당 전까지 $$\Theta(n)$$개의 항목을 삽입해야 한다.
- 한 번의 연산은 재할당 때문에 $$\Theta(n)$$이 걸릴 수 있다.
- 하지만 $$\Theta(n)$$개 연산의 임의의 열은 총 $$\Theta(n)$$ 시간이 걸리므로, 각 연산은 "평균적으로" $$\Theta(1)$$이다.

두 배 늘리기의 비용을 직접 세어 보면, $$n$$번 삽입하는 동안 크기 $$1, 2, 4, \ldots$$에서 재할당이 일어나고 복사 비용의 합은 다음과 같다.

$$
\sum_{i=0}^{\lfloor \log_2 n \rfloor} 2^i = O(n)
$$

Python list는 실제로 약 $$1/8$$ 비율의 여분 공간을 할당한다(`new_allocated = (newsize >> 3) + ...`). 비율이 작아도 여분이 크기에 선형이므로 같은 결론이 나온다.

## 분할상환 분석

**분할상환 분석(amortized analysis)** 은 비용을 여러 연산에 나누어 분배하는 자료구조 분석 기법이다.

- $$k$$번의 연산이 총 $$k \, T(n)$$ 이하의 비용이 들면, 그 연산의 **분할상환 비용(amortized cost)** 은 $$T(n)$$이다.
- "$$T(n)$$ 분할상환"은 대략 많은 연산에 걸친 "평균" $$T(n)$$을 뜻한다.
- 동적 배열에 삽입하는 연산은 분할상환 $$\Theta(1)$$이다.

더 많은 분할상환 기법은 6.046에서 다룬다.

### 동적 배열의 삭제

뒤에서 삭제하는 것은 길이만 줄이면 되므로 그대로 $$\Theta(1)$$이다. 하지만 공간을 많이 낭비할 수 있다. 자료구조의 크기는 $$\Theta(n)$$으로 유지하고 싶다.

- **시도:** 많이 비면 $$r = 1$$이 되도록 크기를 줄인다. 그러면 꽉 찬 상태에서 삽입과 삭제를 번갈아 하면 매번 재할당이 일어나 나빠질 수 있다.
- **아이디어:** $$r < r_d$$가 되면 비율이 $$r_i$$가 되도록 크기를 바꾼다. 단 $$r_d < r_i$$로 둔다(예: $$r_d = 1/4$$, $$r_i = 1/2$$).
- 그러면 다음 비싼 재할당 전까지 $$\Theta(n)$$번의 싼 연산이 일어나야 한다.

임의의 $$\varepsilon > 0$$에 대해 $$r_d = \frac{1}{1+\varepsilon}$$, $$r_i = \frac{r_d + 1}{2}$$로 두면 공간 사용을 $$(1+\varepsilon)n$$으로 제한할 수 있다.

Recitation의 구현은 꽉 차면 두 배로, $$1/4$$까지 줄면 절반으로 재할당한다.

```python
def _resize(self, n):                        # O(1) or O(n)
    if (self.lower < n < self.upper): return
    m = max(n, 1) * self.r
    A = [None] * m
    self._copy_forward(0, self.size, A, 0)
    self.A = A
    self._compute_bounds()                   # upper = len(A), lower = len(A) // r^2

def insert_last(self, x):                    # O(1) amortized
    self._resize(self.size + 1)
    self.A[self.size] = x
    self.size += 1

def delete_last(self):                       # O(1) amortized
    self.A[self.size - 1] = None
    self.size -= 1
    self._resize(self.size)
```

동적 배열은 **마지막 위치의** 동적 연산만 $$\Theta(1)$$에 지원한다. Python list의 `append`와 `pop`은 분할상환 $$O(1)$$이지만, 다른 연산은 $$O(n)$$일 수 있다. 앞쪽 삽입/삭제를 효율적으로 하는 것도 가능하다(PS1).

## 비교

최악 경우 $$O(\cdot)$$ 실행 시간이다. (a)는 분할상환(amortized)을 뜻한다.

| 자료구조 | `build(X)` | `get_at(i)`, `set_at(i, x)` | `insert_first(x)`, `delete_first()` | `insert_last(x)`, `delete_last()` | `insert_at(i, x)`, `delete_at(i)` |
|---|---|---|---|---|---|
| 배열 | $$n$$ | $$1$$ | $$n$$ | $$n$$ | $$n$$ |
| 연결 리스트 | $$n$$ | $$n$$ | $$1$$ | $$n$$ | $$n$$ |
| 동적 배열 | $$n$$ | $$1$$ | $$n$$ | $$1$$ (a) | $$n$$ |

![Worst-case cost table for array, linked list and dynamic array](/images/study/algo/algo-02-data-structures-3.webp)
*강의 노트의 시퀀스 자료구조 비교표. 파란 칸은 상수 시간, 붉은 칸은 선형 시간이며 $$1_{(a)}$$는 분할상환 상수 시간이다. 출처: MIT 6.006 Lecture 2 notes (CC BY-NC-SA 4.0).*

색으로 읽으면 각 자료구조의 성격이 한눈에 보인다. 배열은 정적 열(`get_at`/`set_at`)만 파랗다. 메모리에서 연속된 위치라 주소 계산으로 바로 접근하지만, 길이를 바꾸려면 통째로 다시 만들어야 하기 때문이다. 연결 리스트는 반대로 `insert_first`/`delete_first` 열만 파랗다. 헤드 포인터만 다시 이으면 되지만, $$i$$번째 노드에 가려면 포인터를 $$i$$번 따라가야 한다.

동적 배열은 배열의 파란 칸을 그대로 유지하면서 `insert_last`/`delete_last` 열을 분할상환 상수로 바꾼다. 아래첨자 (a)가 붙은 이유는, 대부분의 호출은 $$O(1)$$이지만 가끔 재할당이 일어나는 호출은 $$\Theta(n)$$이 걸리기 때문이다. 맨 오른쪽 `insert_at`/`delete_at` 열은 세 자료구조 모두 붉다.

어느 자료구조도 임의 위치의 동적 연산을 선형보다 빠르게 지원하지 못한다. 이는 7강(Sequence AVL)에서 개선한다.

## 정리

- 인터페이스는 무엇을 지원하는지(문제), 자료구조는 어떻게 지원하는지(해법)다.
- 시퀀스는 외재적 순서를, 집합은 고유 키에 따른 내재적 순서를 유지한다. 스택, 큐, 사전은 이들의 특수한 경우다.
- 배열은 `get_at`/`set_at`이 $$\Theta(1)$$이지만 동적 연산이 $$\Theta(n)$$이다.
- 연결 리스트는 앞쪽 삽입/삭제가 $$\Theta(1)$$이지만 인덱스 접근이 $$O(n)$$이다.
- 동적 배열은 크기에 비례하는 여분 공간을 두어, 끝에서의 삽입/삭제를 분할상환 $$\Theta(1)$$에 하면서 인덱스 접근도 $$\Theta(1)$$로 유지한다.
- 연산의 분할상환 비용이 $$T(n)$$이라는 것은 $$k$$번의 연산이 총 $$k \, T(n)$$ 이하라는 뜻이다. 축소 기준 $$r_d$$를 확장 후 비율 $$r_i$$보다 작게 두어야 삽입과 삭제가 번갈아 와도 분할상환 상수 시간이 보장된다.

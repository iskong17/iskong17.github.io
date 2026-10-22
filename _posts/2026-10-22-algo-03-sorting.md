---
title: "Algorithms 3: Sets and Sorting"
date: 2026-10-22 14:03:00 +0900
category: study
topic: algorithm
series: algo
lecture: 3
card_title: "Lecture 3: Sets and Sorting"
card_desc: "Sorted array and binary search · Selection sort · Insertion sort · Merge sort · Solving recurrences"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/6d1ae5278d02bbecb5c4428928b24194_MIT6_006S20_lec3.pdf"
description: "정렬된 배열로 집합 인터페이스를 구현하기 위해 정렬 문제를 정의하고, 순열 정렬, 선택 정렬, 삽입 정렬, 병합 정렬의 정확성과 실행 시간을 귀납법과 점화식으로 분석한다."
---

<ul class="post-sources">
<li markdown="span">[MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 3강 정리</li>
<li markdown="span">강의: Erik Demaine, Jason Ku, Justin Solomon</li>
<li markdown="span">MIT OpenCourseWare, CC BY-NC-SA</li>
<li markdown="span">[▶ 강의 영상](https://www.youtube.com/watch?v=oS9aPzUNG-s)</li>
<li markdown="span">[Recitation 3 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/1869dbf640ded6b31f1bd369d2001ef5_MIT6_006S20_r03.pdf)</li>
</ul>

## 집합 인터페이스와 정렬된 배열

집합 인터페이스(3강~8강에서 다룬다)는 2강에서 정의했다. 컨테이너 연산 `build(X)`, `len()`, 정적 연산 `find(k)`, 동적 연산 `insert(x)`, `delete(k)`, 순서 연산 `iter_ord()`, `find_min()`, `find_max()`, `find_next(k)`, `find_prev(k)`로 이루어진다.

항목을 배열에 임의 순서로 저장해도 집합을 구현할 수 있지만 효율적이지 않다. 모든 연산이 배열 전체를 훑어야 한다. 항목을 **키의 오름차순으로 정렬해** 저장하면 다음이 가능하다.

- `find_min()`/`find_max()`가 빨라진다. 각각 배열의 첫 인덱스와 마지막 인덱스에 있다.
- **이진 탐색(binary search)** 으로 `find`가 $$O(\log n)$$이 된다. `find_next`, `find_prev`도 마찬가지다.

Recitation의 이진 탐색은 다음과 같다. 점화식은 $$T(n) = T(n/2) + O(1)$$이므로 $$T(n) = O(\log n)$$이다.

```python
def _binary_search(self, k, i, j):           # O(log n)
    if i >= j:
        return i
    m = (i + j) // 2
    x = self.A.get_at(m)
    if x.key > k:
        return self._binary_search(k, i, m - 1)
    if x.key < k:
        return self._binary_search(k, m + 1, j)
    return m
```

$$O(\cdot)$$ 실행 시간은 다음과 같다.

| 자료구조 | `build(X)` | `find(k)` | `insert(x)`, `delete(k)` | `find_min()`, `find_max()` | `find_prev(k)`, `find_next(k)` |
|---|---|---|---|---|---|
| 배열 | $$n$$ | $$n$$ | $$n$$ | $$n$$ | $$n$$ |
| 정렬된 배열 | $$n \log n$$ | $$\log n$$ | $$n$$ | $$1$$ | $$\log n$$ |

![Set operation costs for an unsorted array and a sorted array](/images/study/algo/algo-03-sorting-1.webp)
*강의 노트의 집합 비용표. 정렬되지 않은 배열은 모든 칸이 $$n$$이지만, 정렬된 배열은 `find`와 순서 연산이 $$\log n$$ 또는 $$1$$로 내려가고 대신 `build`가 $$n \log n$$이 된다. 출처: MIT 6.006 Lecture 3 notes (CC BY-NC-SA 4.0).*

두 행을 칸별로 비교하면 정렬이 무엇을 사고 무엇을 지불하는지 드러난다. `find(k)`는 이진 탐색 덕분에 $$n$$에서 $$\log n$$으로, `find_min`/`find_max`는 양 끝 인덱스를 읽기만 하면 되므로 $$1$$로 내려간다. `find_prev`/`find_next`도 이진 탐색으로 $$k$$가 들어갈 자리를 찾은 뒤 바로 옆 칸을 보면 되므로 $$\log n$$이다.

그 대가는 `build` 칸이다. 정렬되지 않은 배열은 항목을 그대로 복사하면 되므로 $$n$$이지만, 정렬된 배열은 정렬을 해야 하므로 $$n \log n$$이 든다(노트에서 이 칸이 가장 진한 붉은색인 이유다). `insert`/`delete`는 두 경우 모두 $$n$$이다. 정렬된 배열에서는 삽입 위치는 $$\log n$$에 찾지만, 그 뒤의 항목을 한 칸씩 밀어야 하기 때문이다.

삽입과 삭제는 여전히 항목을 밀어야 하므로 $$n$$이다. 남은 질문은 정렬된 배열을 어떻게 효율적으로 만드는가, 즉 `build`를 어떻게 $$n \log n$$에 하는가다. 이렇게 질의를 빠르게 하려고 미리 시간을 들이는 것을 **전처리(preprocessing)** 라 한다.

## 정렬 문제

- **입력:** $$n$$개의 수로 이루어진 (정적) 배열 $$A$$
- **출력:** $$A$$의 정렬된 순열인 (정적) 배열 $$B$$
  - **순열(permutation):** 같은 원소를 다른 순서로 담은 배열
  - **정렬됨(sorted):** 모든 $$i \in \{1, \ldots, n-1\}$$에 대해 $$B[i-1] \le B[i]$$
- 예: $$[8, 2, 4, 9, 3] \to [2, 3, 4, 8, 9]$$

새 배열 $$B$$를 만들지 않고 $$A$$를 덮어쓰면 **파괴적(destructive)** 정렬이다. 추가 공간을 $$O(1)$$만 쓰면 **제자리(in place)** 정렬이다. 제자리 정렬은 파괴적이다(in place $$\subseteq$$ destructive).

## 순열 정렬

$$A$$의 순열은 $$n!$$개이고, 그중 적어도 하나는 정렬되어 있다. 각 순열이 정렬되어 있는지 $$\Theta(n)$$에 확인한다.

```python
def permutation_sort(A):
    '''Sort A'''
    for B in permutations(A):               # O(n!)
        if is_sorted(B):                    # O(n)
            return B                        # O(1)
```

예: $$[2, 3, 1]$$이면 $$\{[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]\}$$을 확인한다.

- **정확성:** 모든 가능성을 시도하므로 경우 분석으로 맞다(완전 탐색).
- **실행 시간:** $$\Omega(n! \cdot n)$$으로 지수 시간이다.

## 점화식 풀기

재귀 알고리즘의 실행 시간은 점화식(recurrence)으로 표현된다. 푸는 방법은 세 가지다.

- **치환법(substitution):** 해를 추측하고, 대표 함수로 바꿔 넣어 점화식이 성립함을 보인다.
- **재귀 트리(recurrence tree):** 재귀 호출을 트리로 그리고 각 노드의 계산량을 합한다.
- **마스터 정리(Master Theorem):** 많은 점화식을 푸는 공식이다(recitation 3).

마스터 정리는 $$T(n) = a T(n/b) + f(n)$$ 꼴($$a \ge 1$$, $$b > 1$$)에서 $$f(n)$$을 재귀 트리의 잎 수 $$n^{\log_b a}$$와 비교한다.

![Recursion tree for T(n) = aT(n/b) + f(n) and the three cases of the Master Theorem](/images/study/algo/algo-03-sorting-4.webp)
*Recitation의 재귀 트리와 마스터 정리 표. 오른쪽 열은 각 레벨의 노드 수와 노드당 일을 곱한 레벨별 총 일이고, 아래 표는 $$f(n)$$이 잎 수 $$n^{\log_b a}$$보다 작은지, 같은지, 큰지에 따른 세 경우다. 출처: MIT 6.006 Recitation 3 notes (CC BY-NC-SA 4.0).*

트리는 위에서 아래로 읽는다. 루트는 크기 $$n$$인 문제 하나로 $$f(n)$$의 일을 한다. 루트는 $$a$$개의 자식을 가지고 각 자식은 크기 $$n/b$$인 문제이므로, 둘째 레벨의 총 일은 $$a \cdot f(n/b)$$다. 일반적으로 깊이 $$i$$에는 $$a^i$$개의 노드가 있고 각각 크기 $$n/b^i$$이므로 레벨 총 일은 $$a^i f(n/b^i)$$다. 크기가 1이 되는 깊이는 $$\log_b n$$이고, 그때 잎의 수는 $$a^{\log_b n} = n^{\log_b a}$$다. 전체 실행 시간은 오른쪽 열을 모두 더한 값이다.

세 경우는 이 합에서 어느 쪽이 지배하는지로 나뉜다. $$f(n)$$이 $$n^{\log_b a}$$보다 다항식만큼 느리게 자라면(경우 1) 레벨이 내려갈수록 일이 기하급수로 늘어 잎 레벨이 지배한다. 둘이 같은 차수면(경우 2) 각 레벨의 일이 비슷해서 레벨 수 $$\log n$$만큼 곱해진다. $$f(n)$$이 더 빠르게 자라면(경우 3) 일이 기하급수로 줄어 루트가 지배한다. 경우 3의 추가 조건 $$a f(n/b) < c f(n)$$은 둘째 레벨의 일이 루트보다 확실히 작다는 정규성 조건이다.

$$f(n) = \Theta(n^c)$$인 단순한 경우는 다음과 같다.

| 조건 | 해 | 직관 |
|---|---|---|
| $$c < \log_b a$$ | $$T(n) = \Theta(n^{\log_b a})$$ | 잎의 일이 지배한다 |
| $$c = \log_b a$$ | $$T(n) = \Theta(n^c \log n)$$ | 일이 트리 전체에 고르게 퍼진다 |
| $$c > \log_b a$$ | $$T(n) = \Theta(n^c)$$ | 루트의 일이 지배한다 |

## 선택 정렬

**선택 정렬(selection sort)** 은 다음과 같이 동작한다.

- 접두사 `A[:i + 1]`에서 가장 큰 수를 찾아 `A[i]`와 맞바꾼다.
- 접두사 `A[:i]`를 재귀적으로 정렬한다.

예: $$[8, 2, 4, 9, 3] \to [8, 2, 4, 3, 9] \to [3, 2, 4, 8, 9] \to [3, 2, 4, 8, 9] \to [2, 3, 4, 8, 9]$$

```python
def selection_sort(A, i = None):            # T(i)
    '''Sort A[:i + 1]'''
    if i is None: i = len(A) - 1            # O(1)
    if i > 0:                               # O(1)
        j = prefix_max(A, i)                # S(i)
        A[i], A[j] = A[j], A[i]             # O(1)
        selection_sort(A, i - 1)            # T(i - 1)

def prefix_max(A, i):                       # S(i)
    '''Return index of maximum in A[:i + 1]'''
    if i > 0:                               # O(1)
        j = prefix_max(A, i - 1)            # S(i - 1)
        if A[i] < A[j]:                     # O(1)
            return j                        # O(1)
    return i                                # O(1)
```

![selection_sort and prefix_max code with the example trace and cost comments](/images/study/algo/algo-03-sorting-2.webp)
*강의 노트의 선택 정렬 코드. 맨 위 줄은 $$[8, 2, 4, 9, 3]$$을 정렬하는 과정의 각 단계이고, 오른쪽 주석은 각 줄의 비용을 $$T$$와 $$S$$로 표시한다. 출처: MIT 6.006 Lecture 3 notes (CC BY-NC-SA 4.0).*

예제 줄을 코드와 맞춰 읽으면 다음과 같다. 처음에 $$i = 4$$이고 `prefix_max(A, 4)`는 최댓값 9의 인덱스 3을 돌려준다. `A[4]`와 `A[3]`을 바꾸면 $$[8, 2, 4, 3, 9]$$가 된다. 다음으로 $$i = 3$$에서 접두사 $$[8, 2, 4, 3]$$의 최댓값 8을 `A[3]`과 바꿔 $$[3, 2, 4, 8, 9]$$가 된다. $$i = 2$$에서는 최댓값 4가 이미 `A[2]`에 있으므로 자기 자신과 바꾸어 배열이 그대로다. 마지막으로 $$i = 1$$에서 3과 2를 바꿔 $$[2, 3, 4, 8, 9]$$를 얻는다.

주석에서 `T(i)`는 `selection_sort`, `S(i)`는 `prefix_max`의 실행 시간이다. `selection_sort`의 본문은 `S(i)`를 한 번, `T(i - 1)`을 한 번 부르고 나머지는 $$O(1)$$이므로 $$T(i) = T(i-1) + S(i) + O(1)$$이 바로 읽힌다. 마찬가지로 `prefix_max`의 본문에서 $$S(i) = S(i-1) + O(1)$$이 나온다. 아래 분석은 이 두 점화식을 푼 것이다.

**`prefix_max` 분석**

- 기저: $$i = 0$$이면 원소가 하나이므로 최댓값의 인덱스는 $$i$$다.
- 귀납: $$i$$에 대해 옳다고 하자. 최댓값은 `A[:i]`의 최댓값이거나 `A[i]`이며, 어느 경우든 올바른 인덱스를 반환한다.
- $$S(1) = \Theta(1)$$, $$S(n) = S(n-1) + \Theta(1)$$
  - 치환법: $$S(n) = \Theta(n)$$으로 추측한다. $$cn = \Theta(1) + c(n-1) \Rightarrow 1 = \Theta(1)$$
  - 재귀 트리: 노드마다 $$\Theta(1)$$ 일을 하는 길이 $$n$$의 사슬이므로 $$\sum_{i=0}^{n-1} 1 = \Theta(n)$$

**`selection_sort` 분석**

- 기저: $$i = 0$$이면 원소가 하나이므로 정렬되어 있다.
- 귀납: $$i$$에 대해 옳다고 하자. 정렬된 출력의 마지막 수는 배열의 최댓값 중 하나이며 알고리즘이 그것을 그 자리에 둔다. 이후 `A[:i]`는 귀납 가설에 의해 정렬된다.
- $$T(1) = \Theta(1)$$, $$T(n) = T(n-1) + \Theta(n)$$
  - 치환법: $$T(n) = \Theta(n^2)$$으로 추측한다. $$cn^2 = \Theta(n) + c(n-1)^2 \Rightarrow c(2n-1) = \Theta(n)$$
  - 재귀 트리: 노드마다 $$\Theta(i)$$ 일을 하는 길이 $$n$$의 사슬이므로 $$\sum_{i=0}^{n-1} i = \Theta(n^2)$$

선택 정렬은 비교를 $$\Omega(n^2)$$번 하지만 교환은 최악에도 $$O(n)$$번만 한다.

## 삽입 정렬

**삽입 정렬(insertion sort)** 은 다음과 같이 동작한다.

- 접두사 `A[:i]`를 재귀적으로 정렬한다.
- `A[:i]`가 정렬되어 있다고 가정하고, 반복적인 교환으로 접두사 `A[:i + 1]`을 정렬한다.

예: $$[8, 2, 4, 9, 3] \to [2, 8, 4, 9, 3] \to [2, 4, 8, 9, 3] \to [2, 4, 8, 9, 3] \to [2, 3, 4, 8, 9]$$

```python
def insertion_sort(A, i = None):            # T(i)
    '''Sort A[:i + 1]'''
    if i is None: i = len(A) - 1            # O(1)
    if i > 0:                               # O(1)
        insertion_sort(A, i - 1)            # T(i - 1)
        insert_last(A, i)                   # S(i)

def insert_last(A, i):                      # S(i)
    '''Sort A[:i + 1] assuming sorted A[:i]'''
    if i > 0 and A[i] < A[i - 1]:           # O(1)
        A[i], A[i - 1] = A[i - 1], A[i]     # O(1)
        insert_last(A, i - 1)               # S(i - 1)
```

**`insert_last` 분석**

- 기저: $$i = 0$$이면 원소가 하나이므로 정렬되어 있다.
- 귀납: $$i$$에 대해 옳다고 하자. `A[i] >= A[i - 1]`이면 배열은 이미 정렬되어 있다. 그렇지 않으면 마지막 두 원소를 바꾼 뒤, 귀납 가설에 의해 `A[:i]`를 정렬할 수 있다.
- $$S(1) = \Theta(1)$$, $$S(n) = S(n-1) + \Theta(1) \Rightarrow S(n) = \Theta(n)$$

**`insertion_sort` 분석**

- 기저: $$i = 0$$이면 원소가 하나이므로 정렬되어 있다.
- 귀납: $$i$$에 대해 옳다고 하자. 알고리즘은 귀납 가설에 의해 `A[:i]`를 정렬하고, 위에서 보인 대로 `insert_last`가 나머지를 올바르게 정렬한다.
- $$T(1) = \Theta(1)$$, $$T(n) = T(n-1) + \Theta(n) \Rightarrow T(n) = \Theta(n^2)$$

삽입 정렬과 선택 정렬은 모두 제자리 정렬이다. 삽입 정렬은 **안정적(stable)** 이다. 즉 값이 같은 항목들이 입력과 같은 순서로 출력된다. 위의 선택 정렬은 안정적이지 않다. 예를 들어 $$(2, 1, 1')$$를 정렬하면 $$(1', 1, 2)$$가 된다.

## 병합 정렬

**병합 정렬(merge sort)** 은 다음과 같이 동작한다.

- 앞쪽 절반과 뒤쪽 절반을 재귀적으로 정렬한다($$n$$은 2의 거듭제곱이라고 가정해도 된다).
- 정렬된 두 절반을 하나의 정렬된 리스트로 병합한다(**두 손가락 알고리즘**, two finger algorithm).

예: $$[7, 1, 5, 6, 2, 4, 9, 3] \to [1, 7, 5, 6, 2, 4, 3, 9] \to [1, 5, 6, 7, 2, 3, 4, 9] \to [1, 2, 3, 4, 5, 6, 7, 9]$$

```python
def merge_sort(A, a = 0, b = None):         # T(b - a = n)
    '''Sort A[a:b]'''
    if b is None: b = len(A)                # O(1)
    if 1 < b - a:                           # O(1)
        c = (a + b + 1) // 2                # O(1)
        merge_sort(A, a, c)                 # T(n / 2)
        merge_sort(A, c, b)                 # T(n / 2)
        L, R = A[a:c], A[c:b]               # O(n)
        merge(L, R, A, len(L), len(R), a, b)  # S(n)

def merge(L, R, A, i, j, a, b):             # S(b - a = n)
    '''Merge sorted L[:i] and R[:j] into A[a:b]'''
    if a < b:                               # O(1)
        if (j <= 0) or (i > 0 and L[i - 1] > R[j - 1]):  # O(1)
            A[b - 1] = L[i - 1]             # O(1)
            i = i - 1                       # O(1)
        else:                               # O(1)
            A[b - 1] = R[j - 1]             # O(1)
            j = j - 1                       # O(1)
        merge(L, R, A, i, j, a, b - 1)      # S(n - 1)
```

![merge_sort and merge code with the example trace and cost comments](/images/study/algo/algo-03-sorting-3.webp)
*강의 노트의 병합 정렬 코드. 맨 위 줄은 $$[7, 1, 5, 6, 2, 4, 9, 3]$$이 크기 2, 4, 8 단위로 병합되어 가는 과정이다. 출처: MIT 6.006 Lecture 3 notes (CC BY-NC-SA 4.0).*

예제는 재귀가 풀려 나오는 순서를 레벨별로 보여 준다. 첫 단계에서는 이웃한 두 원소끼리 병합되어 $$[1, 7], [5, 6], [2, 4], [3, 9]$$가 된다. 다음 단계에서는 크기 2짜리 두 개가 합쳐져 $$[1, 5, 6, 7]$$과 $$[2, 3, 4, 9]$$가 되고, 마지막에 이 둘이 합쳐져 전체가 정렬된다. 각 단계에서 모든 원소가 정확히 한 번씩 병합에 참여하므로 단계당 $$\Theta(n)$$이고, 단계 수가 $$\log_2 8 = 3$$이므로 총 $$\Theta(n \log n)$$이다.

코드의 주석도 같은 구조를 보여 준다. `merge_sort`는 `T(n / 2)` 호출 두 번, 슬라이스 복사 `O(n)`, `merge` 한 번 `S(n)`으로 이루어진다. `merge`는 한 번 호출될 때 원소 하나를 `A[b - 1]`에 놓고 `S(n - 1)`로 재귀하므로 $$S(n) = \Theta(n)$$이다. 노트의 `merge`가 반복문이 아닌 재귀로 쓰인 것은 귀납 증명과 점화식을 코드 구조에 그대로 맞추기 위해서다.

`merge`는 두 리스트의 끝에서부터 더 큰 원소를 골라 `A`의 뒤쪽부터 채운다.

**`merge` 분석**

- 기저: $$n = 0$$이면 배열이 비어 있으므로 자명하게 옳다.
- 귀납: $$n$$에 대해 옳다고 하자. `A[b - 1]`에 들어갈 항목은 `L`과 `R`의 남은 접두사 중 가장 큰 수여야 한다. 두 리스트는 정렬되어 있으므로 각 마지막 항목 중 큰 것을 고르면 충분하다. 나머지는 귀납 가설에 의해 병합된다.
- $$S(0) = \Theta(1)$$, $$S(n) = S(n-1) + \Theta(1) \Rightarrow S(n) = \Theta(n)$$

**`merge_sort` 분석**

- 기저: $$n = 1$$이면 원소가 하나이므로 정렬되어 있다.
- 귀납: $$k < n$$에 대해 옳다고 하자. 알고리즘은 더 작은 두 절반을 귀납 가설에 의해 정렬하고, 위에서 보인 대로 `merge`가 이를 정렬된 배열로 합친다.
- $$T(1) = \Theta(1)$$, $$T(n) = 2T(n/2) + \Theta(n)$$
  - 치환법: $$T(n) = \Theta(n \log n)$$으로 추측한다. $$cn \log n = \Theta(n) + 2c(n/2)\log(n/2) \Rightarrow cn \log 2 = \Theta(n)$$
  - 재귀 트리: 깊이 $$\log_2 n$$, 잎 $$n$$개인 완전 이진 트리다. 레벨 $$i$$에는 노드가 $$2^i$$개 있고 각각 $$O(n/2^i)$$ 일을 한다.

$$
\sum_{i=0}^{\log_2 n} (2^i)(n / 2^i) = \sum_{i=0}^{\log_2 n} n = \Theta(n \log n)
$$

마스터 정리로 보면 $$a = 2$$, $$b = 2$$, $$c = 1 = \log_2 2$$이므로 가운데 경우에 해당한다.

병합 정렬은 병합할 때 선형 크기의 임시 공간을 쓰므로 제자리 정렬이 아니다. 동률을 어느 쪽에서 먼저 가져가는지에 따라 안정적으로 만들 수 있다. 위 구현은 동률일 때 `R`의 원소를 먼저 뒤쪽에 놓으므로 안정적이다. $$\Theta(n \log n)$$은 이차보다 선형에 훨씬 가깝다. $$\log n$$은 어떤 $$\varepsilon > 0$$에 대해서도 $$n^\varepsilon$$보다 느리게 자라기 때문이다.

이제 정렬된 배열의 `build`를 $$\Theta(n \log n)$$에 할 수 있다.

## 정리

- 키 순서로 정렬된 배열은 이진 탐색으로 `find`, `find_next`, `find_prev`를 $$O(\log n)$$에, `find_min`/`find_max`를 $$O(1)$$에 지원한다. 삽입과 삭제는 $$O(n)$$이다.
- 정렬은 입력의 정렬된 순열을 찾는 문제다. 파괴적 정렬은 입력을 덮어쓰고, 제자리 정렬은 $$O(1)$$ 추가 공간만 쓴다.
- 순열 정렬은 완전 탐색으로 $$\Omega(n! \cdot n)$$이다.
- 선택 정렬(최댓값을 맨 뒤로)과 삽입 정렬(마지막 원소를 제자리로)은 $$T(n) = T(n-1) + \Theta(n)$$이므로 $$\Theta(n^2)$$이고 제자리 정렬이다.
- 병합 정렬은 $$T(n) = 2T(n/2) + \Theta(n)$$이므로 $$\Theta(n \log n)$$이다.
- 재귀 알고리즘의 정확성은 귀납법으로, 실행 시간은 치환법, 재귀 트리, 마스터 정리로 점화식을 풀어 보인다.

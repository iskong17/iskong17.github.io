---
title: "Algorithms 6: Binary Trees, Part 1"
date: 2026-10-22 14:06:00 +0900
category: study
topic: algorithm
series: algo
lecture: 6
card_title: "Lecture 6: Binary Trees, Part 1"
card_desc: "Binary tree and traversal order · Successor and predecessor · Insert and delete in O(h) · Binary search tree · Subtree size augmentation"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/376714cc85c6c784d90eec9c575ec027_MIT6_006S20_lec6.pdf"
description: "이진 트리의 순회 순서를 정의하고, 탐색, 삽입, 삭제를 트리 높이 h에 대해 O(h)에 수행하는 법을 익힌 뒤, 순회 순서에 의미를 부여해 집합(이진 탐색 트리)과 시퀀스를 구현한다."
---

> [MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 6강 정리. 강의: Erik Demaine, Jason Ku, Justin Solomon. MIT OpenCourseWare, CC BY-NC-SA.
> 이전: [Algorithms 5]({% post_url 2026-10-22-algo-05-linear-sorting %}) · 다음: [Algorithms 7]({% post_url 2026-10-22-algo-07-avl %})
>
> [▶ 강의 영상](https://www.youtube.com/watch?v=76dhtgZt38A) · [Recitation 6 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/e215840fd7b0e8cb07916a65c17252b2_MIT6_006S20_r06.pdf)

## 지금까지와 새 목표

지금까지 본 시퀀스 자료구조와 목표는 다음과 같다($$O(\cdot)$$, (a)는 분할상환).

| 자료구조 | `build(X)` | `get_at(i)`, `set_at(i, x)` | `insert_first(x)`, `delete_first()` | `insert_last(x)`, `delete_last()` | `insert_at(i, x)`, `delete_at(i)` |
|---|---|---|---|---|---|
| 배열 | $$n$$ | $$1$$ | $$n$$ | $$n$$ | $$n$$ |
| 연결 리스트 | $$n$$ | $$n$$ | $$1$$ | $$n$$ | $$n$$ |
| 동적 배열 | $$n$$ | $$1$$ | $$n$$ | $$1$$ (a) | $$n$$ |
| **목표** | $$n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ |

집합 자료구조와 목표는 다음과 같다((e)는 기대값).

| 자료구조 | `build(X)` | `find(k)` | `insert(x)`, `delete(k)` | `find_min()`, `find_max()` | `find_prev(k)`, `find_next(k)` |
|---|---|---|---|---|---|
| 배열 | $$n$$ | $$n$$ | $$n$$ | $$n$$ | $$n$$ |
| 정렬된 배열 | $$n \log n$$ | $$\log n$$ | $$n$$ | $$1$$ | $$\log n$$ |
| 직접 접근 배열 | $$u$$ | $$1$$ | $$1$$ | $$u$$ | $$u$$ |
| 해시 테이블 | $$n$$ (e) | $$1$$ (e) | $$1$$ (a)(e) | $$n$$ | $$n$$ |
| **목표** | $$n \log n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ |

어떻게? **이진 트리(binary tree)** 로 한다.

## 이진 트리

연결 리스트 같은 포인터 기반 자료구조는 (분할상환이나 기대값이 아닌) 최악 경우 성능을 낼 수 있다. 이진 트리는 노드마다 포인터가 세 개인 포인터 기반 자료구조다. 노드는 `node.{item, parent, left, right}`로 표현한다.

```python
class Binary_Node:
    def __init__(A, x):                     # O(1)
        A.item   = x
        A.left   = None
        A.right  = None
        A.parent = None
```

강의 노트의 예시 트리는 다음과 같다.

```text
          ________<A>_____
     __<B>_____          <C>
  __<D>      <E>
<F>
```

| 노드 | item | parent | left | right |
|---|---|---|---|---|
| `<A>` | A | - | `<B>` | `<C>` |
| `<B>` | B | `<A>` | `<D>` | `<E>` |
| `<C>` | C | `<A>` | - | - |
| `<D>` | D | `<B>` | `<F>` | - |
| `<E>` | E | `<B>` | - | - |
| `<F>` | F | `<D>` | - | - |

연결 리스트에서는 많은 노드가 헤드에서 $$O(n)$$ 포인터만큼 떨어져 있지만, 이진 트리는 어떤 노드도 루트에서 $$O(\log n)$$ 포인터 이상 떨어지지 않게 만들 수 있다.

### 용어

- **루트(root):** 부모가 없는 노드다(예: `<A>`).
- **잎(leaf):** 자식이 없는 노드다(예: `<C>`, `<E>`, `<F>`).
- 루트에서 부모 포인터를 따라 올라가며 지나는 노드들을 **조상(ancestor)** 이라 한다.
- `<R>`을 루트로 하는 트리에서 노드 `<X>`의 **깊이(depth)** 는 `<X>`에서 `<R>`까지 경로의 길이다.
- 노드 `<X>`의 **높이(height)** 는 `<X>`를 루트로 하는 서브트리에 있는 노드의 깊이 중 최댓값이다.

**아이디어:** 연산을 루트의 높이 $$h$$에 대해 $$O(h)$$ 시간에 동작하도록 설계하고, $$h = O(\log n)$$을 유지한다(다음 강의).

## 순회 순서

이진 트리에는 고유한 순서가 있다. **순회 순서(traversal order)** 다.

- 노드 `<X>`의 왼쪽 서브트리의 모든 노드는 `<X>`보다 앞에 온다.
- 노드 `<X>`의 오른쪽 서브트리의 모든 노드는 `<X>`보다 뒤에 온다.

루트에서 시작하는 재귀 알고리즘으로 노드를 순회 순서대로 나열할 수 있다. 왼쪽 서브트리를 재귀적으로 나열하고, 자신을 나열한 뒤, 오른쪽 서브트리를 재귀적으로 나열한다. 노드마다 $$O(1)$$ 일을 하므로 $$O(n)$$ 시간이다.

```python
def subtree_iter(A):                        # O(n)
    if A.left:  yield from A.left.subtree_iter()
    yield A
    if A.right: yield from A.right.subtree_iter()
```

예시 트리의 순회 순서는 (`<F>`, `<D>`, `<B>`, `<E>`, `<A>`, `<C>`)이다.

지금은 순회 순서가 저장된 항목과 아무 관계가 없다. 나중에 순회 순서에 의미를 부여해 시퀀스와 집합 인터페이스를 구현한다. 그때까지는 트리를 조작해도 순회 순서를 보존하는 것만 신경 쓴다.

## 트리 탐색

### 서브트리의 첫 노드

노드 `<X>`의 서브트리에서 순회 순서상 첫 노드를 찾는다(마지막은 대칭이다).

- `<X>`에 왼쪽 자식이 있으면, 왼쪽 서브트리의 첫 노드를 재귀적으로 반환한다.
- 없으면 `<X>`가 첫 노드이므로 그것을 반환한다.
- 아래로만 내려가므로 실행 시간은 $$O(h)$$다.
- 예: `<A>`의 서브트리의 첫 노드는 `<F>`다.

### 다음 노드

노드 `<X>`의 순회 순서상 **다음 노드(successor)** 를 찾는다(**이전 노드(predecessor)** 는 대칭이다).

- `<X>`에 오른쪽 자식이 있으면, 오른쪽 서브트리의 첫 노드를 반환한다.
- 없으면, `<X>`가 왼쪽 서브트리에 속하는 가장 낮은 조상을 반환한다.
- 첫째 경우는 아래로만, 둘째 경우는 위로만 움직이므로 실행 시간은 $$O(h)$$다.
- 예: `<B>`의 다음 노드는 `<E>`, `<E>`의 다음 노드는 `<A>`, `<C>`의 다음 노드는 None이다.

```python
def subtree_first(A):                       # O(h)
    if A.left: return A.left.subtree_first()
    else:      return A

def successor(A):                           # O(h)
    if A.right: return A.right.subtree_first()
    while A.parent and (A is A.parent.right):
        A = A.parent
    return A.parent
```

`subtree_last`와 `predecessor`는 left와 right를 바꾸면 된다.

## 동적 연산

트리를 항목 하나만큼 바꾼다. 잎만 추가하거나 제거한다.

- 순회 순서에서 어떤 노드 뒤에(앞은 대칭) 노드를 추가한다.
- 트리에서 항목을 제거한다.

### 삽입

순회 순서에서 노드 `<X>` 뒤에 노드 `<Y>`를 삽입한다.

- `<X>`에 오른쪽 자식이 없으면, `<Y>`를 `<X>`의 오른쪽 자식으로 만든다.
- 그렇지 않으면, `<Y>`를 `<X>`의 다음 노드의 왼쪽 자식으로 만든다. 그 다음 노드는 오른쪽 서브트리의 첫 노드이므로 왼쪽 자식이 없다.
- 실행 시간은 $$O(h)$$다.

```python
def subtree_insert_after(A, B):             # O(h)
    if A.right:
        A = A.right.subtree_first()
        A.left, B.parent = B, A
    else:
        A.right, B.parent = B, A
```

**예:** `<E>` 앞에 `<G>`를 삽입한다. `<E>`에 왼쪽 자식이 없으므로 `<G>`는 `<E>`의 왼쪽 자식이 된다.

```text
          ________<A>_____                       ________<A>_____
     __<B>_____          <C>      =>        __<B>_____          <C>
  __<D>      <E>                         __<D>      __<E>
<F>                                    <F>        <G>
```

**예:** `<A>` 뒤에 `<H>`를 삽입한다. `<A>`에 오른쪽 자식 `<C>`가 있으므로, `<A>`의 다음 노드인 `<C>`의 왼쪽 자식이 된다.

```text
          ________<A>_____                       ________<A>_____
     __<B>_____          <C>      =>        __<B>_____        __<C>
  __<D>      __<E>                       __<D>      __<E>    <H>
<F>        <G>                         <F>        <G>
```

### 삭제

노드 `<X>`의 항목을 `<X>`의 서브트리에서 삭제한다.

- `<X>`가 잎이면, 부모에서 떼어 내고 반환한다.
- 그렇지 않으면 `<X>`에 자식이 있다.
  - 왼쪽 자식이 있으면, `<X>`의 이전 노드와 항목을 맞바꾸고 재귀한다.
  - 그렇지 않으면 오른쪽 자식이 있으므로, `<X>`의 다음 노드와 항목을 맞바꾸고 재귀한다.
- 이전/다음 노드는 왼쪽/오른쪽 서브트리 안에 있으므로 항목은 아래로만 내려간다. 실행 시간은 $$O(h)$$다.

`<X>`에서 이전 노드와 항목을 맞바꾸면 두 항목의 순회 순서가 바뀌지만, `<X>`의 항목은 결국 삭제되므로 나머지 항목의 순서는 그대로다.

```python
def subtree_delete(A):                      # O(h)
    if A.left or A.right:                   # A is not a leaf
        if A.left: B = A.predecessor()
        else:      B = A.successor()
        A.item, B.item = B.item, A.item
        return B.subtree_delete()
    if A.parent:                            # A is a leaf
        if A.parent.left is A: A.parent.left  = None
        else:                  A.parent.right = None
    return A
```

**예:** 잎인 `<F>`를 제거한다.

```text
          ________<A>_____                     ________<A>_____
     __<B>_____        __<C>      =>      __<B>_____        __<C>
  __<D>      __<E>    <H>                <D>      __<E>    <H>
<F>        <G>                                  <G>
```

**예:** `<A>`를 제거한다. 잎이 아니므로 먼저 잎까지 맞바꾸며 내려간다. `<A>`의 이전 노드는 `<E>`이므로 항목을 맞바꾸고, 다시 그 위치의 이전 노드 `<G>`와 맞바꾼 뒤, 잎이 된 위치를 떼어 낸다.

```text
       ________<A>_____             ________<E>_____             _____<E>_____
  __<B>_____        __<C>   =>  __<B>_____        __<C>   =>  __<B>__      __<C>
<D>      __<E>    <H>         <D>      __<G>    <H>         <D>     <G>   <H>
       <G>                           <A>
```

## 응용: 집합

**아이디어:** **집합 이진 트리(Set Binary Tree)**, 즉 **이진 탐색 트리(Binary Search Tree, BST)** 는 순회 순서가 키의 오름차순인 이진 트리다.

이는 **BST 성질**과 동치다. 모든 노드에 대해, 왼쪽 서브트리의 모든 키 $$\le$$ 노드의 키 $$\le$$ 오른쪽 서브트리의 모든 키다.

그러면 노드 `<X>`의 서브트리에서 키가 $$k$$인 노드를 이진 탐색처럼 $$O(h)$$에 찾을 수 있다.

- $$k$$가 `<X>`의 키보다 작으면, 왼쪽 서브트리에서 재귀한다(없으면 None을 반환한다).
- $$k$$가 `<X>`의 키보다 크면, 오른쪽 서브트리에서 재귀한다(없으면 None을 반환한다).
- 그렇지 않으면 `<X>`에 저장된 항목을 반환한다.

```python
def subtree_find(A, k):                     # O(h)
    if k < A.item.key:
        if A.left:  return A.left.subtree_find(k)
    elif k > A.item.key:
        if A.right: return A.right.subtree_find(k)
    else:           return A
    return None
```

다른 집합 연산도 비슷한 패턴을 따른다. 예를 들어 `insert`는 같은 방식으로 내려가다가 빈 자리에서 `subtree_insert_before`/`subtree_insert_after`를 호출하고, `find_min`/`find_max`는 루트의 `subtree_first`/`subtree_last`다. `find_next(k)`는 다음과 같다.

```python
def subtree_find_next(A, k):                # O(h)
    if A.item.key <= k:
        if A.right: return A.right.subtree_find_next(k)
        else:       return None
    elif A.left:
        B = A.left.subtree_find_next(k)
        if B: return B
    return A
```

## 응용: 시퀀스

**아이디어:** **시퀀스 이진 트리(Sequence Binary Tree)** 는 순회 순서가 시퀀스 순서인 이진 트리다.

서브트리의 순회 순서에서 $$i$$번째 노드를 어떻게 찾을까? 이 연산을 `subtree_at(i)`라 하자. 순회 순서 전체를 훑을 수도 있지만 $$O(n)$$으로 나쁘다. 그런데 서브트리의 크기를 $$O(1)$$에 알 수 있다면 $$O(h)$$에 풀 수 있다.

- 왼쪽 서브트리의 크기 $$n_L$$을 $$i$$와 비교한다.
- $$i < n_L$$이면, 왼쪽 서브트리에서 재귀한다.
- $$i > n_L$$이면, 오른쪽 서브트리에서 $$i' = i - n_L - 1$$로 재귀한다.
- 그렇지 않으면 $$i = n_L$$이고, 찾는 노드에 도달했다.

```python
def subtree_at(A, i):                       # O(h)
    assert 0 <= i
    if A.left: L_size = A.left.size
    else:      L_size = 0
    if i < L_size:   return A.left.subtree_at(i)
    elif i > L_size: return A.right.subtree_at(i - L_size - 1)
    else:            return A
```

각 노드에 서브트리 크기를 **보강(augmentation)** 으로 저장한다.

- 각 노드에 `node.size` 필드를 추가한다.
- 새 잎을 추가하면, 모든 조상 $$a$$의 `a.size`에 $$+1$$을 $$O(h)$$ 시간에 더한다.
- 잎을 삭제하면, 모든 조상 $$a$$의 `a.size`에 $$-1$$을 $$O(h)$$ 시간에 더한다.

시퀀스 연산은 빠른 `subtree_at(i)`에서 바로 나온다. `get_at(i)`/`set_at(i, x)`는 $$i$$번째 노드의 항목을 읽거나 쓰고, `insert_at(i, x)`는 $$(i-1)$$번째 노드 뒤에 삽입하며, `delete_at(i)`는 $$i$$번째 노드를 삭제한다.

`build(X)`를 단순하게 하면 $$O(nh)$$이지만, $$O(n)$$에 할 수 있다(recitation). 가운데 항목을 루트에 두고 왼쪽 절반과 오른쪽 절반을 재귀적으로 각 서브트리로 만들면 된다. 노드마다 상수 일을 하므로 $$O(n)$$이고, 높이는 $$H(n) = 1 + H(n/2)$$를 따르므로 $$O(\log n)$$이다.

```python
def build_subtree(A, i, j):                 # build A[i..j], O(j - i + 1)
    c = (i + j) // 2
    root = Node_Type(A[c])
    if i < c:
        root.left = build_subtree(A, i, c - 1)
        root.left.parent = root
    if c < j:
        root.right = build_subtree(A, c + 1, j)
        root.right.parent = root
    return root
```

## 지금까지

| 집합 | `build(X)` | `find(k)` | `insert(x)`, `delete(k)` | `find_min()`, `find_max()` | `find_prev(k)`, `find_next(k)` |
|---|---|---|---|---|---|
| 이진 트리 | $$n \log n$$ | $$h$$ | $$h$$ | $$h$$ | $$h$$ |
| 목표 | $$n \log n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ |

| 시퀀스 | `build(X)` | `get_at(i)`, `set_at(i, x)` | `insert_first(x)`, `delete_first()` | `insert_last(x)`, `delete_last()` | `insert_at(i, x)`, `delete_at(i)` |
|---|---|---|---|---|---|
| 이진 트리 | $$n$$ | $$h$$ | $$h$$ | $$h$$ | $$h$$ |
| 목표 | $$n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ | $$\log n$$ |

다음 강의에서는 삽입이나 삭제 후에도 이진 트리의 균형을 유지해 $$h = O(\log n)$$을 지킴으로써, $$O(h)$$ 실행 시간을 $$O(\log n)$$으로 줄인다.

## 정리

- 이진 트리는 노드마다 item, parent, left, right를 가진 포인터 기반 자료구조이며, 연산을 높이 $$h$$에 대해 $$O(h)$$로 설계한다.
- 순회 순서는 "왼쪽 서브트리, 자신, 오른쪽 서브트리"이며 $$O(n)$$에 나열할 수 있다.
- 서브트리의 첫/마지막 노드와 다음/이전 노드는 아래로만 또는 위로만 움직여 $$O(h)$$에 찾는다.
- 삽입은 잎을 붙이는 것으로, 삭제는 이전/다음 노드와 항목을 바꿔 잎까지 내린 뒤 떼어 내는 것으로 $$O(h)$$에 한다. 둘 다 나머지 항목의 순회 순서를 보존한다.
- 순회 순서를 키 순서로 두면 이진 탐색 트리(집합)가, 시퀀스 순서로 두면 시퀀스 이진 트리가 된다.
- 시퀀스 트리는 서브트리 크기를 보강해 `subtree_at(i)`를 $$O(h)$$에 하며, `build`는 가운데를 루트로 하는 재귀로 $$O(n)$$이다.

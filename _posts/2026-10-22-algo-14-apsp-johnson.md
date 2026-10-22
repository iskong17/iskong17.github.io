---
title: "Algorithms 14: APSP and Johnson"
date: 2026-10-22 14:14:00 +0900
category: study
topic: algorithm
series: algo
lecture: 14
card_title: "Lecture 14: APSP and Johnson"
card_desc: "All-pairs shortest paths · Reweighting · Potential function · Johnson's algorithm · Reduction"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/7d7d5c35490f41b7b037cafbda7019ad_MIT6_006S20_lec14.pdf"
description: "모든 쌍 최단 경로 문제를 정의하고, 포텐셜 함수로 간선 가중치를 다시 매겨 최단 경로를 보존하면서 음수 가중치를 없앤 뒤 Bellman-Ford 한 번과 Dijkstra |V|번으로 푸는 Johnson 알고리즘을 정리한다."
---

> [MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 14강 정리. 강의: Erik Demaine, Jason Ku, Justin Solomon. MIT OpenCourseWare, CC BY-NC-SA.
> 이전: [Algorithms 13]({% post_url 2026-10-22-algo-13-dijkstra %}) · 다음: [Algorithms 15]({% post_url 2026-10-22-algo-15-dp-1 %})
>
> [▶ 강의 영상](https://www.youtube.com/watch?v=EmSmaW-ud6A) · [Recitation 14 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/6027e2a155c948937fdfb93981f7070c_MIT6_006S20_r14.pdf)

## 복습: SSSP 알고리즘

| 그래프 | 가중치 | 알고리즘 | 실행 시간 $$O(\cdot)$$ |
|---|---|---|---|
| 일반 | 가중치 없음 | BFS | $$\lvert V \rvert + \lvert E \rvert$$ |
| DAG | 임의 | DAG Relaxation | $$\lvert V \rvert + \lvert E \rvert$$ |
| 일반 | 음이 아님 | Dijkstra | $$\lvert V \rvert \log \lvert V \rvert + \lvert E \rvert$$ |
| 일반 | 임의 | Bellman-Ford | $$\lvert V \rvert \cdot \lvert E \rvert$$ |

최단 경로 문제를 풀려면 먼저 문제에 맞는 그래프를 정의하거나 만들고, 그 그래프에서 SSSP 알고리즘을 실행한다. 일반적으로 적용 가능한 것 중 가장 빠른 알고리즘을 쓴다. Bellman-Ford는 모든 가중 그래프에 쓸 수 있지만 가장 느리므로, 다른 알고리즘을 쓸 수 있으면 그쪽을 택한다. 이 알고리즘들은 SSSP 외에도 Full-DFS/Full-BFS로 연결 요소 세기, DFS로 DAG 위상 정렬, Bellman-Ford로 음수 가중치 사이클 탐지 같은 문제를 푼다.

## 모든 쌍 최단 경로 (APSP)

- **입력:** 가중치 $$w : E \to \mathbb{Z}$$를 가진 방향 그래프 $$G = (V, E)$$
- **출력:** 모든 $$u, v \in V$$에 대한 $$\delta(u, v)$$. $$G$$에 음수 가중치 사이클이 있으면 중단한다.

교통, 회로 배치, 공급망처럼 네트워크 전체를 이해해야 할 때 유용하다.

출력 크기가 $$O(\lvert V \rvert^2)$$이므로 SSSP 알고리즘을 $$\lvert V \rvert$$번 실행하는 것도 사실 꽤 좋다.

| 조건 | 알고리즘 | APSP 시간 |
|---|---|---|
| 가중치가 양수이고 $$O(\lvert V \rvert + \lvert E \rvert)$$로 유계 | BFS | $$\lvert V \rvert \cdot O(\lvert V \rvert + \lvert E \rvert)$$ |
| 비순환 | DAG Relaxation | $$\lvert V \rvert \cdot O(\lvert V \rvert + \lvert E \rvert)$$ |
| 가중치가 음이 아님 (또는 무방향) | Dijkstra | $$\lvert V \rvert \cdot O(\lvert V \rvert \log \lvert V \rvert + \lvert E \rvert)$$ |
| 일반 | Bellman-Ford | $$\lvert V \rvert \cdot O(\lvert V \rvert \cdot \lvert E \rvert)$$ |

무방향 그래프에 음수 간선이 있으면 그 간선을 왕복하는 음수 가중치 사이클이 생기므로, 음수 사이클이 없는 무방향 그래프는 가중치가 음이 아니다. 그래서 Dijkstra를 쓸 수 있다.

예를 들어 가중치 없는 희소 그래프($$\lvert E \rvert = O(\lvert V \rvert)$$)에서는 각 정점에서 BFS를 실행하면 $$O(\lvert V \rvert^2)$$다. 출력 자체가 $$\Omega(\lvert V \rvert^2)$$이므로 이것은 최적이다. 음이 아닌 가중치에서 Dijkstra를 $$\lvert V \rvert$$번 실행하면 $$O(\lvert V \rvert^2 \log \lvert V \rvert + \lvert V \rvert\lvert E \rvert)$$로, 희소 그래프에서는 출력보다 $$\log \lvert V \rvert$$배 클 뿐이다. 반면 일반 그래프에서 Bellman-Ford를 $$\lvert V \rvert$$번 실행하면 $$O(\lvert V \rvert^2 \lvert E \rvert)$$로, 출력보다 $$\lvert E \rvert$$배나 크다.

**목표.** 임의의 가중 그래프에서 APSP를 $$\lvert V \rvert \cdot O(\lvert V \rvert \log \lvert V \rvert + \lvert E \rvert)$$에 푼다.

## 접근: 가중치 다시 매기기

**아이디어.** 최단 경로를 보존하면서 모든 간선 가중치를 음이 아니게 만든다. 즉 $$G$$를 음수 가중치가 없는 $$G'$$로 다시 가중치를 매기되(reweight), $$G$$의 최단 경로가 $$G'$$에서도 최단 경로이게 한다. 가중치가 음이 아니면 Dijkstra를 $$\lvert V \rvert$$번 실행해 APSP를 푼다.

**주장.** $$G'$$의 거리로부터 $$G$$의 거리를 $$O(\lvert V \rvert(\lvert V \rvert + \lvert E \rvert))$$에 계산할 수 있다.

- 각 $$s \in V$$에 대해 거리로부터 최단 경로 트리를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 계산한다(11강).
- 이 트리는 $$G$$에서도 최단 경로 트리이므로, DFS로 트리를 순회하며 $$G$$의 가중치로 거리를 계산한다.
- 총 $$O(\lvert V \rvert \cdot (\lvert V \rvert + \lvert E \rvert))$$이며, Dijkstra를 $$\lvert V \rvert$$번 실행하는 시간보다 작다.

(아래에서 보듯 Johnson의 재가중에서는 거리 차이가 간단한 식으로 주어지므로 $$O(\lvert V \rvert^2)$$에 바로 계산된다.)

그런데 음이 아닌 가중치를 가진 $$G'$$를 어떻게 만들까? 그런 것이 가능하기는 할까?

**주장.** $$G$$에 음수 가중치 사이클이 있으면 불가능하다.

**증명.** 음수 가중치가 없으면 최단 경로가 단순 경로이지만, 음수 가중치 사이클이 있으면 그렇지 않다. 따라서 최단 경로를 보존할 수 없다.

그러면 음수 가중치는 있지만 음수 가중치 사이클은 없는 그래프에서, 최단 경로를 보존하며 가중치를 음이 아니게 만들 수 있을까?

### 실패한 아이디어

**아이디어.** $$G$$에서 가장 작은 가중치의 음수를 모든 간선에 더한다. 그러면 모든 가중치가 음이 아니다.

**실패.** 최단 경로가 보존되지 않는다. 더한 상수를 $$c$$라 하면 간선이 $$k$$개인 경로는 가중치가 $$kc$$만큼 늘어나므로, 간선이 적은 경로가 유리해진다.

### 한 정점에서의 재가중

**아이디어.** 정점 $$v$$를 골라 $$v$$에서 나가는 모든 간선에 $$h$$를 더하고, $$v$$로 들어오는 모든 간선에서 $$h$$를 뺀다.

**주장.** 이렇게 다시 매겨도 최단 경로가 보존된다.

**증명.**

- $$v$$에서 시작하는 모든 경로의 가중치는 $$h$$만큼 바뀐다.
- $$v$$에서 끝나는 모든 경로의 가중치는 $$-h$$만큼 바뀐다.
- $$v$$를 지나가는 경로는 들어올 때 $$-h$$, 나갈 때 $$+h$$이므로 (그 부분에서) 가중치가 바뀌지 않는다.

같은 두 끝점을 가진 경로는 모두 같은 양만큼 바뀌므로 최단 경로가 보존된다. 이는 최단 경로를 보존하며 그래프를 변환하는 매우 일반적이고 유용한 기법이다.

## 포텐셜 함수

이 기법은 여러 정점에 동시에 적용해도 된다.

- 각 정점 $$v \in V$$를 **포텐셜(potential)** $$h(v)$$로 대응시키는 **포텐셜 함수** $$h : V \to \mathbb{Z}$$를 정의한다.
- $$G'$$는 $$G$$와 같지만 간선 $$(u, v) \in E$$의 가중치를 $$w'(u, v) = w(u, v) + h(u) - h(v)$$로 둔다.

**주장.** $$G$$의 최단 경로는 $$G'$$에서도 최단 경로다.

**증명.** $$G$$에서 경로 $$\pi = (v_0, \dots, v_k)$$의 가중치는 $$w(\pi) = \sum_{i=1}^{k} w(v_{i-1}, v_i)$$다. $$G'$$에서의 가중치는 다음과 같다.

$$
\sum_{i=1}^{k} \big( w(v_{i-1}, v_i) + h(v_{i-1}) - h(v_i) \big) = w(\pi) + h(v_0) - h(v_k)
$$

내부 정점 $$v_i$$마다 $$+h(v_i)$$와 $$-h(v_i)$$가 하나씩 있어 $$h$$의 합이 망원급수(telescoping)처럼 소거된다. 따라서 $$v_0$$에서 $$v_k$$로 가는 모든 경로가 같은 양 $$h(v_0) - h(v_k)$$만큼 바뀌고, 최단 경로는 여전히 최단 경로다.

## 가중치를 음이 아니게 만들기

$$G'$$에 음수 가중치가 없도록 하는 포텐셜 함수를 찾을 수 있을까? 즉 모든 $$(u, v) \in E$$에 대해 다음을 만족하는 $$h$$가 있을까?

$$
w(u, v) + h(u) - h(v) \ge 0
$$

이 조건을 정리하면 $$h(v) \le h(u) + w(u, v)$$로, 삼각 부등식과 같은 모양이다.

**아이디어.** 어떤 $$s$$에 대해 $$h(v) = \delta(s, v)$$이고 $$\delta(s, v)$$가 유한하면 조건이 만족된다. 그러나 그래프가 연결되어 있지 않을 수 있어, 모든 정점에 도달하는 그런 $$s$$가 없을 수 있다.

**아이디어.** 새 정점 $$s$$를 추가하고, $$s$$에서 모든 $$v \in V$$로 가중치 0인 방향 간선을 둔다. 가중치 0인 경로가 있으므로 모든 $$v \in V$$에 대해 $$\delta(s, v) \le 0$$이다.

**주장.** 어떤 $$v \in V$$에 대해 $$\delta(s, v) = -\infty$$이면 원래 그래프에 음수 가중치 사이클이 있다.

**증명.** $$s$$로 들어오는 간선이 없으므로 $$s$$를 추가해도 새 사이클은 생기지 않는다. 따라서 새 그래프에 음수 가중치 사이클이 있으면 원래 그래프에도 있다.

반대로 모든 $$v \in V$$에 대해 $$\delta(s, v)$$가 유한하면, 삼각 부등식에 의해 모든 $$(u, v) \in E$$에 대해 $$w'(u, v) = w(u, v) + h(u) - h(v) \ge 0$$이다. 새 가중치는 음이 아니면서 최단 경로를 보존한다.

## Johnson 알고리즘

1. $$G$$에 정점 $$x$$를 추가하고 각 정점 $$v \in V$$로 가중치 0인 간선을 이어 $$G_x$$를 만든다.
2. (Bellman-Ford로) 모든 $$v \in V$$에 대해 $$\delta_x(x, v)$$를 계산한다.
3. 어떤 $$v \in V$$에 대해 $$\delta_x(x, v) = -\infty$$이면 $$G$$에 음수 가중치 사이클이 있으므로 중단한다.
4. 그렇지 않으면 다음을 한다.
   - 각 간선을 $$w'(u, v) = w(u, v) + \delta_x(x, u) - \delta_x(x, v)$$로 다시 매겨 $$G'$$를 만든다.
   - 각 $$u \in V$$에 대해 (Dijkstra로) $$G'$$에서 모든 $$v$$까지의 최단 경로 거리 $$\delta'(u, v)$$를 계산한다.
   - 모든 $$v \in V$$에 대해 $$\delta(u, v) = \delta'(u, v) - \delta_x(x, u) + \delta_x(x, v)$$를 계산한다.

마지막 식은 포텐셜 함수의 주장에서 $$\delta'(u, v) = \delta(u, v) + h(u) - h(v)$$이고 $$h = \delta_x(x, \cdot)$$이기 때문이다.

12강의 `bellman_ford`와 13강의 `dijkstra`를 쓰면 다음처럼 쓸 수 있다. 정점은 $$0, \dots, \lvert V \rvert - 1$$이고 새 정점 $$x$$는 $$\lvert V \rvert$$로 둔다.

```python
def johnson(Adj, w):
    V = len(Adj)
    x = V                                       # 새 정점 x
    Adj_x = [list(nbrs) for nbrs in Adj] + [list(range(V))]
    def w_x(u, v):                              # x에서 나가는 간선은 가중치 0
        return 0 if u == x else w(u, v)
    h, _ = bellman_ford(Adj_x, w_x, x)          # 음수 사이클이면 예외 발생 (중단)
    def w2(u, v):                               # 음이 아닌 새 가중치
        return w(u, v) + h[u] - h[v]
    D = []
    for u in range(V):                          # Dijkstra를 |V|번
        d2, _ = dijkstra(Adj, w2, u)
        D.append([d2[v] - h[u] + h[v] for v in range(V)])
    return D                                    # D[u][v] = delta(u, v)
```

### 정확성

- $$G$$에서 $$G'$$로의 변환이 최단 경로를 보존함은 이미 증명했다.
- 나머지는 Bellman-Ford와 Dijkstra의 정확성으로 귀결된다.

부호가 있는 APSP를 음이 아닌 APSP로 **환원(reduction)** 한 것이다. 환원은 시간을 아껴 준다. 이번에는 귀납법이 필요 없다.

### 실행 시간

- $$G_x$$ 구성: $$O(\lvert V \rvert + \lvert E \rvert)$$
- Bellman-Ford: $$O(\lvert V \rvert\lvert E \rvert)$$
- $$G'$$ 구성: $$O(\lvert V \rvert + \lvert E \rvert)$$
- Dijkstra $$\lvert V \rvert$$번: $$O(\lvert V \rvert \cdot (\lvert V \rvert \log \lvert V \rvert + \lvert E \rvert))$$
- $$G'$$의 거리로부터 $$G$$의 거리 계산: $$O(\lvert V \rvert^2)$$

전체는 다음과 같다.

$$
O(\lvert V \rvert^2 \log \lvert V \rvert + \lvert V \rvert\lvert E \rvert)
$$

이는 Bellman-Ford를 $$\lvert V \rvert$$번 실행하는 $$O(\lvert V \rvert^2 \lvert E \rvert)$$보다 점근적으로 빠르고, 음이 아닌 가중치에서 Dijkstra를 $$\lvert V \rvert$$번 실행하는 것과 같은 시간이다. Bellman-Ford 한 번의 $$O(\lvert V \rvert\lvert E \rvert)$$는 Dijkstra $$\lvert V \rvert$$번의 비용에 흡수된다.

## 정리

- APSP는 모든 정점 쌍의 $$\delta(u, v)$$를 구하는 문제로, 출력 크기가 $$\Theta(\lvert V \rvert^2)$$이므로 SSSP를 $$\lvert V \rvert$$번 실행하는 것이 기본 전략이다. 다만 Bellman-Ford를 $$\lvert V \rvert$$번 쓰면 $$O(\lvert V \rvert^2\lvert E \rvert)$$로 느리다.
- 모든 간선에 같은 상수를 더하면 간선이 적은 경로가 유리해져 최단 경로가 보존되지 않는다.
- 포텐셜 함수 $$h$$로 $$w'(u, v) = w(u, v) + h(u) - h(v)$$라 두면 모든 경로 가중치가 $$h(v_0) - h(v_k)$$만큼 바뀌므로 최단 경로가 보존된다.
- 모든 정점으로 가중치 0인 간선을 가진 새 정점 $$x$$를 두고 $$h(v) = \delta(x, v)$$로 잡으면, 삼각 부등식에 의해 새 가중치가 음이 아니다. $$\delta(x, v) = -\infty$$인 정점이 있으면 음수 가중치 사이클이 있다.
- Johnson 알고리즘은 Bellman-Ford 한 번으로 $$h$$를 구하고, 재가중한 그래프에서 Dijkstra를 $$\lvert V \rvert$$번 실행한 뒤 거리를 되돌려, 임의의 가중 그래프의 APSP를 $$O(\lvert V \rvert^2 \log \lvert V \rvert + \lvert V \rvert\lvert E \rvert)$$에 푼다.

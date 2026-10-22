---
title: "Algorithms 11: Weighted Shortest Paths"
date: 2026-10-22 14:11:00 +0900
category: study
topic: algorithm
series: algo
lecture: 11
card_title: "Lecture 11: Weighted Shortest Paths"
card_desc: "Weighted graphs · Negative-weight cycles · Shortest-paths tree · Relaxation · DAG Relaxation"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/aa57a9785adf925bc85c1920f53755a0_MIT6_006S20_lec11.pdf"
description: "가중 그래프에서 최단 경로 가중치와 음수 가중치 사이클을 정의하고, 거리로부터 최단 경로 트리를 복원하는 방법과 간선 완화의 안전성을 정리한 뒤, 위상 순서로 간선을 완화하는 DAG Relaxation으로 DAG의 SSSP를 선형 시간에 푼다."
---

> [MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 11강 정리. 강의: Erik Demaine, Jason Ku, Justin Solomon. MIT OpenCourseWare, CC BY-NC-SA.
> 이전: [Algorithms 10]({% post_url 2026-10-22-algo-10-dfs %}) · 다음: [Algorithms 12]({% post_url 2026-10-22-algo-12-bellman-ford %})
>
> [▶ 강의 영상](https://www.youtube.com/watch?v=5cF5Bgv59Sc) · [Recitation 11 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/77ff36ad54779575d9ba8d8687f91b5b_MIT6_006S20_r11.pdf)

## 복습

- 단일 출발점 최단 경로(SSSP): BFS로 $$O(\lvert V \rvert + \lvert E \rvert)$$ (정점마다 거리를 반환)
- 단일 출발점 도달 가능성: BFS나 DFS로 $$O(\lvert E \rvert)$$ (도달 가능한 정점만 반환)
- 연결 요소: Full-BFS나 Full-DFS로 $$O(\lvert V \rvert + \lvert E \rvert)$$
- DAG의 위상 정렬: Full-DFS로 $$O(\lvert V \rvert + \lvert E \rvert)$$

지금까지 거리는 경로의 간선 수였다. 이번 강의에서는 거리의 의미를 일반화한다.

## 가중 그래프

**가중 그래프(weighted graph)** 는 그래프 $$G = (V, E)$$와 가중치 함수 $$w : E \to \mathbb{Z}$$의 쌍이다. 즉 각 간선 $$e = (u, v) \in E$$에 정수 가중치 $$w(e) = w(u, v)$$를 부여한다. 간선 가중치는 도로망의 거리, 네트워크 연결의 지연 시간, 소셜 네트워크에서 관계의 강도 등 다양한 곳에 쓰인다.

가중치를 표현하는 흔한 방법은 두 가지다.

- 그래프 표현 안에 저장: 인접 리스트의 각 정점과 함께 간선 가중치를 저장한다.
- 각 간선을 가중치로 대응시키는 별도의 집합 자료구조를 둔다.

이 수업에서는 간선의 가중치를 $$O(1)$$에 조회할 수 있는 표현을 가정한다. 예를 들어 Recitation처럼 인접 집합을 사전의 사전으로 두면 `def w(u, v): return W[u][v]`로 조회한다.

이 강의의 예제 그래프 $$G_1$$, $$G_2$$는 정점 $$a, \dots, h$$와 다음 간선을 가진다.

| 간선 | $$(a,b)$$ | $$(a,e)$$ | $$(b,e)$$ | $$(b,f)$$ | $$(e,f)$$ | $$(f,c)$$ | $$(f,g)$$ | $$(g,c)$$ | $$(g,h)$$ | $$(h,c)$$ | $$(h,d)$$ | $$(d,c)$$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 가중치 | $$-5$$ | $$7$$ | $$6$$ | $$-4$$ | $$3$$ | $$8$$ | $$2$$ | $$1$$ | $$-2$$ | $$9$$ | $$4$$ | $$5$$ |

여기에 $$G_1$$은 간선 $$(c, b)$$를, $$G_2$$는 간선 $$(b, c)$$를 가중치 $$-1$$로 더 가진다. 이 한 간선의 방향 차이로 $$G_1$$에는 사이클이 생기고 $$G_2$$는 DAG가 된다.

![Example weighted graphs G1 and G2](/images/study/algo/algo-11-weighted-shortest-paths-1.webp)
*강의의 예제 가중 그래프 $$G_1$$(왼쪽)과 $$G_2$$(오른쪽). 두 그래프는 $$b$$와 $$c$$ 사이 가중치 $$-1$$ 간선의 방향만 다르다. 출처: MIT 6.006 Lecture 11 notes (CC BY-NC-SA 4.0).*

그림을 읽을 때는 화살표 방향과 간선 옆의 숫자를 함께 본다. 위쪽 줄은 $$a, b, c, d$$, 아래쪽 줄은 $$e, f, g, h$$이고, 숫자가 그 간선의 가중치다. 음수 가중치 간선은 $$(a,b)$$의 $$-5$$, $$(b,f)$$의 $$-4$$, $$(g,h)$$의 $$-2$$, 그리고 $$b$$와 $$c$$ 사이의 $$-1$$ 네 개다.

$$G_1$$에서는 $$c \to b$$ 간선 때문에 $$b \to f \to g \to c \to b$$로 되돌아오는 사이클이 생긴다. $$G_2$$에서는 같은 간선이 $$b \to c$$로 뒤집혀 있어 어떤 정점에서 출발해도 자기 자신으로 돌아올 수 없고, 모든 경로가 결국 싱크인 $$c$$로 흘러든다. 같은 정점과 거의 같은 간선으로 "사이클이 있는 경우"와 "DAG인 경우"를 나란히 비교하려는 구성이다.

## 가중 경로와 최단 경로

- 가중 그래프에서 경로 $$\pi$$의 **가중치** $$w(\pi)$$는 경로에 있는 간선 가중치의 합이다.
- $$s \in V$$에서 $$t \in V$$로 가는 (가중) **최단 경로**는 $$s$$에서 $$t$$로 가는 최소 가중치 경로다.
- $$\delta(s, t) = \inf\{w(\pi) \mid \pi \text{는 } s \text{에서 } t \text{로 가는 경로}\}$$를 $$s$$에서 $$t$$로의 **최단 경로 가중치(shortest-path weight)** 라 한다.
- 가중 그래프에서는 간선 수가 아니라 최단 경로 가중치를 흔히 "거리"라 부른다.

가중치 없는 그래프와 마찬가지로 다음이 성립한다.

- $$s$$에서 $$t$$로 가는 경로가 없으면 $$\delta(s, t) = \infty$$다.
- 최단 경로의 부분 경로는 최단 경로다. 그렇지 않다면 더 짧은 경로를 끼워 넣을 수 있기 때문이다.

### 음수 가중치 사이클

왜 최솟값이 아니라 하한(infimum)일까? 가중치가 최소인 유한 길이 경로가 존재하지 않을 수 있기 때문이다. 그래프에 **음수 가중치 사이클(negative-weight cycle)**, 곧 같은 정점에서 시작하고 끝나며 $$w(\pi) < 0$$인 경로가 있으면 그렇다. 예를 들어 $$G_1$$의 $$(b, f, g, c, b)$$는 가중치가 $$-4 + 2 + 1 - 1 = -2$$인 음수 가중치 사이클이다.

$$s$$에서 $$t$$로 가는 경로가 음수 가중치 사이클 위의 정점을 지날 수 있으면, 사이클을 한 바퀴 더 돌 때마다 가중치가 줄어드므로 $$\delta(s, t) = -\infty$$다. 이때는 최단 경로가 아니라 음수 가중치 사이클 자체를 원할 수 있다.

## 가중 최단 경로 알고리즘

다음 네 강의에서는 가중 그래프에서 최단 경로 가중치를 구하는 알고리즘을 다룬다. 부모 포인터는 다루지 않는데, 거리를 알면 최단 경로 트리를 선형 시간에 나중에 복원할 수 있기 때문이다(다음 절).

| 그래프 | 가중치 | 알고리즘 | 실행 시간 $$O(\cdot)$$ | 강의 |
|---|---|---|---|---|
| 일반 | 가중치 없음 | BFS | $$\lvert V \rvert + \lvert E \rvert$$ | 9강 |
| DAG | 임의 | DAG Relaxation | $$\lvert V \rvert + \lvert E \rvert$$ | 11강 |
| 일반 | 임의 | Bellman-Ford | $$\lvert V \rvert \cdot \lvert E \rvert$$ | 12강 |
| 일반 | 음이 아님 | Dijkstra | $$\lvert V \rvert \log \lvert V \rvert + \lvert E \rvert$$ | 13강 |

![SSSP algorithms by graph restriction and running time](/images/study/algo/algo-11-weighted-shortest-paths-2.webp)
*강의 노트의 SSSP 알고리즘 표. 왼쪽 두 열은 그래프와 가중치에 대한 제한, 오른쪽은 알고리즘과 실행 시간이다. 출처: MIT 6.006 Lecture 11 notes (CC BY-NC-SA 4.0).*

표는 "입력에 대한 가정이 강할수록 더 빠른 알고리즘을 쓸 수 있다"는 교환 관계를 색으로 보여 준다. 왼쪽 열에서 파란 칸(General, Any)은 제한이 없다는 뜻이고, 붉은 칸(Unweighted, DAG)은 강한 제한이다. 오른쪽 실행 시간 열에서는 진한 파랑인 $$\lvert V \rvert + \lvert E \rvert$$이 가장 빠르고, 분홍색인 $$\lvert V \rvert \cdot \lvert E \rvert$$이 가장 느리다.

BFS는 가중치가 없다는 가장 강한 제한을 대가로 선형 시간을 얻는다. DAG Relaxation은 가중치 제한은 없지만 그래프가 비순환이어야 하고, 역시 선형 시간이다. Bellman-Ford는 아무 제한이 없는 대신 가장 느리며, Dijkstra는 가중치가 음수가 아니라는 중간 정도의 제한으로 선형에 가까운 시간을 얻는다. 문제를 풀 때는 입력이 만족하는 제한을 확인하고, 그 제한을 허용하는 행 가운데 가장 빠른 알고리즘을 고르면 된다.

이미 아는 알고리즘인 BFS도 다음과 같은 경우에는 $$O(\lvert V \rvert + \lvert E \rvert)$$에 가중 최단 경로를 푼다.

- 가중치가 모두 양수이고 서로 같을 때
- 가중치가 모두 양수이고 가중치의 총합이 $$O(\lvert V \rvert + \lvert E \rvert)$$ 이하일 때 (가중치 $$w$$인 간선을 길이 $$w$$인 경로로 쪼개면 된다)

일반 가중 그래프에서 SSSP를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 푸는 방법은 알려져 있지 않다. 그러나 그래프가 DAG이면 가능하다.

## 최단 경로 트리 복원

BFS에서는 탐색하면서 부모 포인터를 기록했다. 대신 나중에 계산할 수도 있다. 모든 정점 $$v$$에 대해 $$\delta(s, v)$$를 알면 최단 경로 트리를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 만들 수 있다. 가중 최단 경로에서는 $$\delta(s, v)$$가 유한한 정점만 부모 포인터가 필요하다.

1. 빈 $$P$$를 만들고 $$P(s) = \text{None}$$으로 둔다.
2. $$\delta(s, u)$$가 유한한 각 정점 $$u \in V$$에 대해, 나가는 이웃 $$v \in \text{Adj}^+(u)$$마다 다음을 한다.
   - $$P(v)$$가 아직 없고 $$\delta(s, v) = \delta(s, u) + w(u, v)$$이면, 간선 $$(u, v)$$를 지나는 최단 경로가 있으므로 $$P(v) = u$$로 둔다.

가중치 0인 사이클이 있으면 부모 포인터가 그 사이클을 따라 돌 수 있다. 이런 사이클 위의 정점에 모두 표시를 한다. 그리고 표시되지 않은 각 정점 $$u$$(나중에 표시가 풀린 정점도 포함)에 대해, 표시된 $$v \in \text{Adj}^+(u)$$가 $$\delta(s, v) = \delta(s, u) + w(u, v)$$를 만족하면 $$v$$에서 부모 포인터를 따라가며 사이클의 표시를 풀고 $$P(v) = u$$로 두어 사이클을 끊는다. (이 알고리즘이 부모 포인터를 선형 시간에 올바르게 계산함을 보이는 것은 연습 문제다.)

부모 포인터를 나중에 계산할 수 있으므로, 앞으로는 거리 계산에 집중한다.

## 완화

**아이디어.** 각 정점 $$v \in V$$에 대해 거리 추정값 $$d(s, v)$$(처음에는 $$\infty$$)를 유지한다. 이 값은 항상 실제 거리 $$\delta(s, v)$$의 상한이며, $$d(s, v) = \delta(s, v)$$가 될 때까지 점차 낮춘다.

언제 낮출까? 간선이 **삼각 부등식(triangle inequality)** 을 위반할 때다. 삼각 부등식은 $$u$$에서 $$v$$로 가는 최단 경로 가중치가 다른 정점 $$x$$를 거쳐 가는 최단 경로보다 클 수 없다는 것이다.

$$
\delta(u, v) \le \delta(u, x) + \delta(x, v) \quad \text{for all } u, v, x \in V
$$

어떤 간선 $$(u, v)$$에 대해 $$d(s, v) > d(s, u) + w(u, v)$$이면 삼각 부등식이 깨진 것이다. 이때 $$d(s, v)$$를 $$d(s, u) + w(u, v)$$로 낮춰 위반된 제약을 만족시킨다. 이를 간선 $$(u, v)$$를 **완화(relax)** 한다고 한다.

```python
def try_to_relax(Adj, w, d, parent, u, v):
    if d[v] > d[u] + w(u, v):       # u를 거치는 더 좋은 경로
        d[v] = d[u] + w(u, v)       # 더 짧은 경로로 간선 완화
        parent[v] = u
```

**주장(안전성).** 완화는 안전하다. 즉 모든 $$v \in V$$에 대해 $$d(s, v)$$는 항상 $$v$$로 가는 어떤 경로의 가중치(또는 $$\infty$$)다. 따라서 $$d(s, v) \ge \delta(s, v)$$가 유지된다.

**증명.** 모든 $$v' \in V$$에 대해 $$d(s, v')$$가 어떤 경로의 가중치(또는 $$\infty$$)라고 하자. 초기에는 $$d(s, s) = 0$$이 길이 0인 경로에 해당하고 나머지는 $$\infty$$이므로 성립한다. 간선 $$(u, v)$$를 완화하면 $$d(s, v)$$가 $$d(s, u) + w(u, v)$$가 되는데, 이것은 $$u$$를 거쳐 $$v$$로 가는 경로의 가중치다.

Recitation은 반대 방향의 보조정리도 제시한다. **더 이상 완화할 간선이 없으면** 모든 $$v$$에 대해 $$d(s, v) \le \delta(s, v)$$다. $$\delta(s, v) < d(s, v)$$라면 더 짧은 경로 $$\pi$$가 있고, $$\pi$$에서 $$d(s, b) > \delta(s, b)$$인 첫 간선 $$(a, b)$$는 완화 가능하므로 모순이다. 따라서 "완화 가능한 간선이 없을 때까지 아무 간선이나 완화"하는 일반적인 틀은 끝나기만 하면 올바르다.

문제는 완화 순서다. 도달 가능한 음수 가중치 사이클이 있으면 영원히 끝나지 않는다. 비순환 그래프에서도 나쁜 순서로 완화하면 지수 번의 갱신이 일어날 수 있다. Recitation의 예는 $$2n + 1$$개 정점과 $$3n$$개 간선으로 된 구간 $$n$$개짜리 그래프로, 각 구간 $$i$$의 세 간선 가중치가 $$2^{n-i}$$다. 나쁜 순서에서는 갱신 횟수가 $$T(n) = 3 + 2T(n - 2)$$, 곧 $$T(n) = \Theta(2^{n/2})$$이 된다.

![Graph on which a bad relaxation order performs exponentially many updates](/images/study/algo/algo-11-weighted-shortest-paths-3.webp)
*Recitation의 지수 완화 예. 구간 $$i$$는 $$v_{2i}$$에서 $$v_{2i+2}$$로 바로 가는 top 간선과, $$v_{2i+1}$$을 거쳐 가는 left, right 간선으로 이루어진다. 출처: MIT 6.006 Recitation 11 notes (CC BY-NC-SA 4.0).*

그림에서 각 구간은 두 갈래 길이다. $$v_{2i}$$에서 $$v_{2i+2}$$로 가는 방법은 top 간선 하나를 타는 것과 left, right 두 간선을 차례로 타는 것뿐이다. 세 간선의 가중치가 모두 $$2^{n-i}$$이므로 top 경로의 가중치는 $$2^{n-i}$$, 아래 경로는 $$2 \cdot 2^{n-i}$$다. 따라서 최단 경로는 언제나 top 간선만 이어 타는 경로이고, 위상 순서로 완화하면 각 간선을 한 번씩만 보면 된다.

나쁜 순서는 각 구간에서 left, right 간선을 먼저 완화해 $$v_{2i+2}$$에 일부러 더 큰 값을 넣은 뒤, 그 틀린 값으로 오른쪽 구간 전체를 끝까지 완화한다. 그다음에야 top 간선을 완화해 $$v_{2i+2}$$의 값을 고치고, 오른쪽 구간 전체를 처음부터 다시 완화한다. 오른쪽 부분을 두 번씩 다시 푸는 일이 구간마다 반복되므로 점화식 $$T(n) = 3 + 2T(n-2)$$가 나오고, 갱신 횟수가 그래프 크기에 대해 지수적으로 늘어난다.

## DAG Relaxation

DAG에는 음수 가중치 사이클이 있을 수 없다. 위상 순서대로 각 정점에서 나가는 간선을 정확히 한 번씩 완화하면 최단 경로가 계산된다.

1. 모든 $$v \in V$$에 대해 $$d(s, v) = \infty$$로 두고, $$d(s, s) = 0$$으로 둔다.
2. $$G$$의 위상 정렬 순서로 각 정점 $$u$$를 처리한다. 나가는 이웃 $$v \in \text{Adj}^+(u)$$마다 $$d(s, v) > d(s, u) + w(u, v)$$이면 간선 $$(u, v)$$를 완화해 $$d(s, v) = d(s, u) + w(u, v)$$로 둔다.

```python
def DAG_Relaxation(Adj, w, s):      # Adj: 인접 리스트, w: 가중치, s: 출발점
    _, order = dfs(Adj, s)          # DFS 실행
    order.reverse()                 # 종료 순서를 뒤집어 위상 순서로
    d = [float('inf') for _ in Adj] # 최단 경로 추정값 d(s, v)
    parent = [None for _ in Adj]    # 부모 포인터 초기화
    d[s], parent[s] = 0, s          # 출발점 초기화
    for u in order:                 # 위상 순서로 정점 순회
        for v in Adj[u]:            # u에서 나가는 간선 순회
            try_to_relax(Adj, w, d, parent, u, v)
    return d, parent
```

$$s$$에서의 DFS 종료 순서만 쓰므로 $$s$$에서 도달 가능한 정점만 처리한다. 도달할 수 없는 정점은 $$\infty$$로 남는데, 이는 올바른 값이다.

### 예제

$$G_2$$에서 $$a$$부터 DAG Relaxation을 실행하자. 위상 순서 $$a, b, e, f, g, h, d, c$$로 처리하면 다음 거리를 얻는다.

| $$v$$ | $$a$$ | $$b$$ | $$e$$ | $$f$$ | $$g$$ | $$h$$ | $$d$$ | $$c$$ |
|---|---|---|---|---|---|---|---|---|
| $$\delta(a, v)$$ | $$0$$ | $$-5$$ | $$1$$ | $$-9$$ | $$-7$$ | $$-9$$ | $$-5$$ | $$-6$$ |

예를 들어 $$e$$는 $$a$$를 처리할 때 7이 되었다가 $$b$$를 처리할 때 $$-5 + 6 = 1$$로 낮아진다. $$c$$로 들어오는 간선은 다섯 개인데, $$b$$에서 오는 $$-5 - 1 = -6$$과 $$g$$에서 오는 $$-7 + 1 = -6$$이 가장 작다.

### 정확성

**주장.** DAG Relaxation이 끝나면 모든 $$v \in V$$에 대해 $$d(s, v) = \delta(s, v)$$다.

**증명.** 위상 순서에서 처음 $$k$$개 정점의 모든 $$v$$에 대해 $$d(s, v) = \delta(s, v)$$임을 $$k$$에 대한 귀납법으로 보인다.

- 기저: 정점 $$s$$와 위상 순서에서 $$s$$보다 앞선 모든 정점은 처음부터 주장을 만족한다($$s$$보다 앞선 정점은 $$s$$에서 도달할 수 없으므로 $$\infty$$가 맞다).
- 귀납: 처음 $$k'$$개 정점에 대해 성립한다고 하고, $$v$$를 $$(k' + 1)$$번째 정점이라 하자. $$s$$에서 $$v$$로 가는 최단 경로를 생각하고, 경로에서 $$v$$ 바로 앞의 정점을 $$u$$라 하자. $$u$$는 위상 순서에서 $$v$$보다 앞에 있으므로 귀납 가정에 의해 $$d(s, u) = \delta(s, u)$$다. $$u$$를 처리할 때 $$d(s, v)$$는 $$\delta(s, u) + w(u, v) = \delta(s, v)$$ 이하가 된다. 한편 완화는 안전하므로 $$d(s, v) \ge \delta(s, v)$$다. 따라서 $$d(s, v) = \delta(s, v)$$다.

다른 관점도 있다. DAG Relaxation은 모든 정점 $$v$$에 대해 다음을 계산한다.

$$
d(s, v) = \min\{d(s, u) + w(u, v) \mid u \in \text{Adj}^-(v)\}
$$

$$v$$로 가는 최단 경로는 $$v$$의 들어오는 이웃 $$u$$ 중 하나를 지나야 한다. 귀납 가정에 의해 모든 $$u \in \text{Adj}^-(v)$$에 대해 $$d(s, u) = \delta(s, u)$$이면 $$d(s, v) = \delta(s, v)$$다.

### 실행 시간

- 초기화는 $$O(\lvert V \rvert)$$, 위상 정렬은 $$O(\lvert V \rvert + \lvert E \rvert)$$다.
- 추가 작업은 $$O(1) \times \sum_{u \in V} \deg^+(u) = O(\lvert E \rvert)$$다.

전체 실행 시간은 선형 $$O(\lvert V \rvert + \lvert E \rvert)$$다.

## Recitation 연습: 선수 과목 스트레스

**문제.** 한 학기에 한 과목만 듣는 학생이 19.854를 듣고 싶다. 각 과목은 선수 과목 중 적어도 하나를 먼저 들어야 들을 수 있고, (과목, 선수 과목) 쌍마다 스트레스 시간이 조사되어 있다. 스트레스 합이 최소인 수강 순서를 선형 시간에 구한다.

**풀이.** 과목마다 정점을 두고, $$b$$가 $$a$$의 선수 과목이면 $$a$$에서 $$b$$로 가는 간선을 둔다. 가중치는 $$b$$를 먼저 듣고 $$a$$를 들을 때의 스트레스다. 이 그래프는 DAG이므로 19.854에서 출발하는 DAG Relaxation으로 모든 과목까지의 최단 경로를 구한다. 선수 과목이 없는 과목(DAG의 싱크) 중 19.854까지의 총 스트레스가 최소인 것을 골라 그 최단 경로를 뒤집어 반환한다.

## 정리

- 가중 그래프에서 경로 가중치는 간선 가중치의 합이고, $$\delta(s, t)$$는 모든 경로 가중치의 하한이다. 경로가 없으면 $$\infty$$, 음수 가중치 사이클을 지날 수 있으면 $$-\infty$$다.
- 거리 $$\delta(s, v)$$를 알면 $$\delta(s, v) = \delta(s, u) + w(u, v)$$인 간선을 골라 최단 경로 트리를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 복원할 수 있으므로 거리 계산에 집중한다.
- 완화는 삼각 부등식을 위반한 간선 $$(u, v)$$에 대해 $$d(s, v)$$를 $$d(s, u) + w(u, v)$$로 낮추는 연산이며, 추정값이 항상 어떤 경로의 가중치이므로 안전하다.
- 완화 순서가 나쁘면 비순환 그래프에서도 지수 번의 갱신이 필요할 수 있다.
- DAG Relaxation은 위상 순서대로 각 정점의 나가는 간선을 한 번씩 완화해 DAG의 SSSP를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 푼다.

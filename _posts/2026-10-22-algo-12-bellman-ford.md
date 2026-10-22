---
title: "Algorithms 12: Bellman-Ford"
date: 2026-10-22 14:12:00 +0900
category: study
topic: algorithm
series: algo
lecture: 12
card_title: "Lecture 12: Bellman-Ford"
card_desc: "Simple shortest paths · k-edge distance · Negative cycle witness · Graph duplication · Bellman-Ford"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/2430d7903a5529451d80c17f89a41fe8_MIT6_006S20_lec12.pdf"
description: "음수 가중치가 있는 일반 그래프의 SSSP를 위해 k-간선 거리와 음수 사이클 증인을 정의하고, 그래프를 |V|+1개 층으로 복제한 DAG에서 DAG Relaxation을 실행하는 Bellman-Ford로 O(|V||E|)에 모든 거리와 음수 사이클을 구하는 방법을 정리한다."
---

> [MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 12강 정리. 강의: Erik Demaine, Jason Ku, Justin Solomon. MIT OpenCourseWare, CC BY-NC-SA.
> 이전: [Algorithms 11]({% post_url 2026-10-22-algo-11-weighted-shortest-paths %}) · 다음: [Algorithms 13]({% post_url 2026-10-22-algo-13-dijkstra %})
>
> [▶ 강의 영상](https://www.youtube.com/watch?v=f9cVS_URPc0) · [Recitation 12 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/1a34924a2ee4e08c580c2268f8e854ee_MIT6_006S20_r12.pdf)

## 복습과 목표

| 그래프 | 가중치 | 알고리즘 | 실행 시간 $$O(\cdot)$$ | 강의 |
|---|---|---|---|---|
| 일반 | 가중치 없음 | BFS | $$\lvert V \rvert + \lvert E \rvert$$ | 9강 |
| DAG | 임의 | DAG Relaxation | $$\lvert V \rvert + \lvert E \rvert$$ | 11강 |
| 일반 | 임의 | Bellman-Ford | $$\lvert V \rvert \cdot \lvert E \rvert$$ | 12강 (이번 강의) |
| 일반 | 음이 아님 | Dijkstra | $$\lvert V \rvert \log \lvert V \rvert + \lvert E \rvert$$ | 13강 |

지난 강의에서 가중 그래프, 최단 경로 가중치, 음수 가중치 사이클을 정의했고, 거리로부터 최단 경로 트리를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 구하는 방법과 가중 DAG의 SSSP를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 푸는 DAG Relaxation을 보았다. 이번 강의의 목표는 음수 가중치가 있는 일반 그래프의 SSSP다.

- 모든 $$v \in V$$에 대해 $$\delta(s, v)$$를 계산한다(음수 가중치 사이클을 거쳐 도달 가능하면 $$-\infty$$).
- $$s$$에서 도달 가능한 음수 가중치 사이클이 있으면 하나를 반환한다.

## 워밍업

**연습 1.** 무방향 그래프 $$G$$가 음수 가중치 사이클을 가지는지 판별하라.

**풀이.** 음수 가중치 간선이 하나라도 있으면 "예"다. 무방향 간선 $$\{u, v\}$$의 가중치가 음수면 $$u \to v \to u$$가 곧 음수 가중치 사이클이기 때문이다. $$O(\lvert E \rvert)$$ 시간이 든다. 그래서 이번 강의에서는 방향 그래프만 다룬다.

**연습 2.** $$O(\lvert V \rvert (\lvert V \rvert + \lvert E \rvert))$$에 동작하는 SSSP 알고리즘 $$A$$가 주어졌을 때, 이것으로 SSSP를 $$O(\lvert V \rvert \lvert E \rvert)$$에 풀어라.

**풀이.**

- BFS나 DFS로 $$s$$에서 도달 가능한 정점을 $$O(\lvert E \rvert)$$에 찾는다.
- 도달할 수 없는 정점 $$v$$에 $$\delta(s, v) = \infty$$를 $$O(\lvert V \rvert)$$에 표시한다.
- 도달 가능한 정점만으로 된 그래프 $$G' = (V', E')$$를 $$O(\lvert V \rvert + \lvert E \rvert)$$에 만든다.
- $$G'$$에서 $$s$$부터 $$A$$를 실행한다. $$G'$$는 연결되어 있으므로 $$\lvert V' \rvert = O(\lvert E' \rvert) = O(\lvert E \rvert)$$이고, $$A$$는 $$O(\lvert V \rvert \lvert E \rvert)$$에 동작한다.

이번 강의에서는 일반 그래프에서 이 실행 시간을 갖는 SSSP 알고리즘을 만든다.

## 단순 최단 경로

그래프에 사이클과 음수 가중치가 함께 있으면 음수 가중치 사이클이 있을 수 있다. 하지만 음수 가중치 사이클이 없으면 최단 경로는 단순 경로다.

**주장 1.** $$\delta(s, v)$$가 유한하면 $$v$$로 가는 단순 최단 경로가 존재한다.

**증명.** 귀류법을 쓴다. 단순 최단 경로가 없다고 하고, 최단 경로 중 정점 수가 가장 적은 것을 $$\pi$$라 하자. $$\pi$$는 단순하지 않으므로 사이클 $$C$$를 포함한다. $$C$$의 가중치는 음이 아니다(음수라면 $$\delta(s, v) = -\infty$$). $$\pi$$에서 $$C$$를 빼면 정점이 더 적고 $$w(\pi') \le w(\pi)$$인 경로 $$\pi'$$가 생기므로 모순이다.

단순 경로는 정점을 반복하지 않으므로, 유한한 최단 경로는 최대 $$\lvert V \rvert - 1$$개의 간선을 가진다.

## 음수 사이클 증인

**$$k$$-간선 거리($$k$$-edge distance)** $$\delta_k(s, v)$$는 간선을 $$k$$개 이하로 쓰는 $$s$$에서 $$v$$로의 경로 중 최소 가중치다.

**아이디어.** 모든 $$v \in V$$에 대해 $$\delta_{\lvert V \rvert-1}(s, v)$$와 $$\delta_{\lvert V \rvert}(s, v)$$를 계산한다.

- $$\delta(s, v) \ne -\infty$$이면 최단 경로가 단순하거나 존재하지 않으므로 $$\delta(s, v) = \delta_{\lvert V \rvert-1}(s, v)$$다.
- $$\delta_{\lvert V \rvert}(s, v) < \delta_{\lvert V \rvert-1}(s, v)$$이면 $$v$$로 가는 더 짧은 비단순 경로가 있으므로 $$\delta(s, v) = -\infty$$다. 이런 $$v$$를 (음수 사이클) **증인(witness)** 이라 한다.
- 그러나 최단 경로 가중치가 $$-\infty$$이면서 증인이 아닌 정점도 있을 수 있다.

**주장 2.** $$\delta(s, v) = -\infty$$이면 $$v$$는 어떤 증인에서 도달 가능하다.

**증명.** $$s$$에서 도달 가능한 모든 음수 가중치 사이클이 증인을 포함함을 보이면 충분하다. $$s$$에서 도달 가능한 음수 가중치 사이클 $$C$$를 생각한다. $$v \in C$$에 대해 $$C$$에서 $$v$$의 바로 앞 정점을 $$v'$$라 하면 $$\sum_{v \in C} w(v', v) < 0$$이다. 우변이 $$\lvert V \rvert$$개 이하의 간선을 쓰는 어떤 경로의 가중치이므로 다음이 성립한다.

$$
\delta_{\lvert V \rvert}(s, v) \le \delta_{\lvert V \rvert-1}(s, v') + w(v', v)
$$

이를 $$C$$ 전체에 대해 더하면, $$v'$$도 $$C$$ 전체를 한 번씩 돌므로 다음을 얻는다.

$$
\sum_{v \in C} \delta_{\lvert V \rvert}(s, v) \le \sum_{v \in C} \delta_{\lvert V \rvert-1}(s, v') + \sum_{v \in C} w(v', v) < \sum_{v \in C} \delta_{\lvert V \rvert-1}(s, v)
$$

$$C$$에 증인이 없다면 모든 $$v \in C$$에 대해 $$\delta_{\lvert V \rvert}(s, v) \ge \delta_{\lvert V \rvert-1}(s, v)$$이므로 모순이다.

## Bellman-Ford

**아이디어.** **그래프 복제(graph duplication)** 를 쓴다. 그래프의 사본(층, level)을 여러 개 만든다.

- 층을 $$\lvert V \rvert + 1$$개 둔다. $$k$$층의 정점 $$v_k$$는 $$s$$에서 간선 $$k$$개 이하로 $$v$$에 도달한 상태를 나타낸다.
- 간선이 층을 올라가는 방향으로만 있으면 결과 그래프는 DAG다.

$$G = (V, E)$$로부터 새 DAG $$G' = (V', E')$$를 만든다.

- $$G'$$의 정점은 모든 $$v \in V$$와 $$k \in \{0, \dots, \lvert V \rvert\}$$에 대한 $$v_k$$로, $$\lvert V \rvert(\lvert V \rvert + 1)$$개다.
- $$G'$$의 간선은 $$\lvert V \rvert(\lvert V \rvert + \lvert E \rvert)$$개다.
  - 각 $$v \in V$$마다 $$k \in \{1, \dots, \lvert V \rvert\}$$에 대해 가중치 0인 간선 $$(v_{k-1}, v_k)$$ $$\lvert V \rvert$$개
  - 각 $$(u, v) \in E$$마다 $$k \in \{1, \dots, \lvert V \rvert\}$$에 대해 가중치 $$w(u, v)$$인 간선 $$(u_{k-1}, v_k)$$ $$\lvert V \rvert$$개

![Layered DAG G' built from G by graph duplication](/images/study/algo/algo-12-bellman-ford-1.webp)
*아래 예제 그래프 $$G$$(정점 4개)를 $$\lvert V \rvert + 1 = 5$$개 층으로 복제한 $$G'$$. 회색 세로 간선은 가중치 0인 $$(v_{k-1}, v_k)$$이고, 검은 대각선 간선은 원래 간선 $$(u, v)$$를 한 층 아래로 옮긴 $$(u_{k-1}, v_k)$$다. 출처: MIT 6.006 Lecture 12 notes (CC BY-NC-SA 4.0).*

그림에서 행이 층 $$k$$, 열이 원래 정점이다. 맨 위 행 $$a_0, b_0, c_0, d_0$$는 "간선을 0개 쓴 상태", 맨 아래 행은 "간선을 4개까지 쓴 상태"에 해당한다. 가중치는 첫 층 사이에만 적혀 있지만 아래 층 사이의 같은 모양 간선도 가중치가 같다.

핵심은 모든 간선이 위에서 아래로만 향한다는 점이다. 원래 그래프 $$G$$에는 $$b \to c \to d \to b$$ 사이클이 있지만, $$G'$$에서는 $$b_0 \to c_1 \to d_2 \to b_3$$처럼 사이클을 돌 때마다 한 층씩 내려가므로 같은 정점으로 돌아올 수 없다. 그래서 $$G'$$는 DAG이고 DAG Relaxation을 그대로 쓸 수 있다. 회색 간선은 "이번 층에서는 간선을 쓰지 않고 머문다"는 선택을 나타내므로, $$v_k$$까지의 거리가 정확히 $$k$$개가 아니라 $$k$$개 **이하**의 간선을 쓰는 거리가 된다.

알고리즘은 다음과 같다.

1. $$G'$$에서 $$s_0$$부터 DAG Relaxation을 실행해 모든 $$v_k \in V'$$에 대해 $$\delta(s_0, v_k)$$를 계산한다.
2. 각 정점에 대해 $$d(s, v) = \delta(s_0, v_{\lvert V \rvert-1})$$로 둔다.
3. $$\delta(s_0, u_{\lvert V \rvert}) < \delta(s_0, u_{\lvert V \rvert-1})$$인 각 증인 $$u \in V$$에 대해, $$G$$에서 $$u$$로부터 도달 가능한 모든 정점 $$v$$에 $$d(s, v) = -\infty$$를 둔다.

### 예제

$$G$$의 정점은 $$a, b, c, d$$이고 간선은 $$(a, b)$$ $$-5$$, $$(a, c)$$ $$6$$, $$(b, c)$$ $$-4$$, $$(c, d)$$ $$3$$, $$(d, b)$$ $$-1$$이다. 사이클 $$(b, c, d, b)$$의 가중치는 $$-4 + 3 - 1 = -2$$다.

![Example graph G with a negative-weight cycle b, c, d](/images/study/algo/algo-12-bellman-ford-2.webp)
*예제 그래프 $$G$$. $$b \to c \to d \to b$$가 가중치 $$-2$$인 음수 가중치 사이클이다. 출처: MIT 6.006 Lecture 12 notes (CC BY-NC-SA 4.0).*

$$a$$는 사이클 밖에 있고 사이클로 들어가는 간선 $$a \to b$$와 $$a \to c$$만 가진다. 따라서 $$a$$ 자신의 거리는 0으로 유한하지만, $$a$$에서 사이클에 들어간 뒤에는 원하는 만큼 사이클을 돌아 가중치를 줄일 수 있으므로 $$b, c, d$$의 거리는 모두 $$-\infty$$가 되어야 한다. 앞 절의 $$G'$$가 바로 이 $$G$$를 복제한 것이다. $$a$$에서 출발하면 $$\delta(a_0, v_k)$$는 다음과 같다.

| $$k$$ | $$a$$ | $$b$$ | $$c$$ | $$d$$ |
|---|---|---|---|---|
| 0 | $$0$$ | $$\infty$$ | $$\infty$$ | $$\infty$$ |
| 1 | $$0$$ | $$-5$$ | $$6$$ | $$\infty$$ |
| 2 | $$0$$ | $$-5$$ | $$-9$$ | $$9$$ |
| 3 | $$0$$ | $$-5$$ | $$-9$$ | $$-6$$ |
| 4 | $$0$$ | $$-7$$ | $$-9$$ | $$-6$$ |
| $$\delta(a, v)$$ | $$0$$ | $$-\infty$$ | $$-\infty$$ | $$-\infty$$ |

$$\lvert V \rvert = 4$$이므로 3행과 4행을 비교한다. $$b$$에서 $$\delta_4(a, b) = -7 < -5 = \delta_3(a, b)$$이므로 $$b$$가 증인이다. $$b$$에서 도달 가능한 $$b, c, d$$는 모두 $$-\infty$$가 된다. $$c$$와 $$d$$는 증인이 아니지만 $$-\infty$$라는 점에서 주장 2가 필요한 이유를 볼 수 있다.

![Table of k-edge distances computed on G'](/images/study/algo/algo-12-bellman-ford-3.webp)
*$$G'$$에서 DAG Relaxation으로 구한 $$\delta(a_0, v_k)$$ 표. 굵은 숫자는 바로 위 행보다 값이 줄어든 칸이다. 출처: MIT 6.006 Lecture 12 notes (CC BY-NC-SA 4.0).*

표는 위에서 아래로 한 행씩 읽는다. $$k$$행은 $$G'$$의 $$k$$층 정점까지의 거리, 곧 간선 $$k$$개 이하를 쓰는 거리다. 각 칸은 바로 위 행의 같은 열(회색 간선으로 머무르기)과, 위 행의 들어오는 이웃 값에 간선 가중치를 더한 값 중 최솟값이다. 예를 들어 $$k = 2$$의 $$c$$는 $$\min(6, -5 + (-4)) = -9$$이고, $$k = 3$$의 $$d$$는 $$\min(9, -9 + 3) = -6$$이다.

굵은 숫자의 위치를 따라가면 음수 사이클이 드러난다. 1행에서 $$b$$와 $$c$$, 2행에서 $$c$$와 $$d$$, 3행에서 $$d$$, 4행에서 다시 $$b$$가 줄어든다. 간선 3개면 충분했던 $$b$$의 값이 간선 4개에서 $$-5$$에서 $$-7$$로 더 줄었다는 것은, 4번째 간선이 사이클을 한 바퀴 돌았다는 뜻이다. $$\lvert V \rvert - 1 = 3$$행과 $$\lvert V \rvert = 4$$행을 비교하는 것이 바로 이 현상을 잡아내는 검사다.

### 정확성

**주장 3.** 모든 $$v \in V$$와 $$k \in \{0, \dots, \lvert V \rvert\}$$에 대해 $$\delta(s_0, v_k) = \delta_k(s, v)$$다.

**증명.** $$k$$에 대한 귀납법이다.

- 기저: $$k = 0$$일 때 $$s_0$$에서 도달 가능한 $$v_0$$는 $$v = s$$뿐이므로 모든 $$v$$에 대해 성립한다.
- 귀납: $$k < k'$$에서 성립한다고 하고 $$k = k'$$를 보인다. $$v_{k'}$$로 들어오는 간선은 $$u \in \text{Adj}^-(v)$$에 대한 $$(u_{k'-1}, v_{k'})$$와 $$(v_{k'-1}, v_{k'})$$뿐이다.

$$
\begin{aligned}
\delta(s_0, v_{k'}) &= \min\{\delta(s_0, u_{k'-1}) + w(u_{k'-1}, v_{k'}) \mid u_{k'-1} \in \text{Adj}^-(v_{k'})\} \\
&= \min\left( \{\delta(s_0, u_{k'-1}) + w(u, v) \mid u \in \text{Adj}^-(v)\} \cup \{\delta(s_0, v_{k'-1})\} \right) \\
&= \min\left( \{\delta_{k'-1}(s, u) + w(u, v) \mid u \in \text{Adj}^-(v)\} \cup \{\delta_{k'-1}(s, v)\} \right) \\
&= \delta_{k'}(s, v)
\end{aligned}
$$

셋째 줄은 귀납 가정이다.

**주장 4.** Bellman-Ford가 끝나면 $$d(s, v) = \delta(s, v)$$다.

**증명.** 주장 3에 의해 모든 $$v \in V$$에 대해 $$\delta_{\lvert V \rvert-1}(s, v)$$와 $$\delta_{\lvert V \rvert}(s, v)$$가 올바르게 계산된다. $$\delta(s, v) \ne -\infty$$이면 $$d(s, v) = \delta_{\lvert V \rvert-1}(s, v) = \delta(s, v)$$로 올바르게 설정된다. 그다음 증인에서 도달 가능한 모든 $$v$$에 $$d(s, v) = -\infty$$를 두는데, 주장 2에 의해 이것이 정확히 $$-\infty$$인 정점들이다.

### 실행 시간

- $$G'$$의 크기는 $$O(\lvert V \rvert(\lvert V \rvert + \lvert E \rvert))$$이고 같은 시간에 만들 수 있다.
- $$G'$$에서 DAG Relaxation은 $$G'$$ 크기에 선형인 시간이 든다.
- 증인에서 도달 가능한 각 정점마다 $$O(1)$$ 작업을 한다. 증인 하나에서 도달 가능성을 찾는 데 $$O(\lvert E \rvert)$$, 증인은 최대 $$O(\lvert V \rvert)$$개이므로 $$O(\lvert V \rvert\lvert E \rvert)$$다. (또는 가중치 0인 간선으로 모든 증인에 연결된 슈퍼 노드 $$x$$를 두고 $$x$$에서 한 번만 선형 탐색해도 된다.)
- 처음에 $$G$$를 $$s$$에서 도달 가능한 부분 그래프로 줄이면(워밍업 2) 전체가 $$O(\lvert V \rvert\lvert E \rvert)$$ 알고리즘이 된다.

## 부가: 음수 사이클 반환과 공간 최적화

**주장 5.** 임의의 증인 $$v$$에 대해 $$s_0$$에서 $$v_{\lvert V \rvert}$$로 가는 최단 경로 $$\pi$$는 $$G$$의 음수 가중치 사이클을 포함한다.

**증명.** $$\pi$$는 $$\lvert V \rvert + 1$$개의 정점을 지나므로 $$G$$에서 적어도 하나의 사이클 $$C$$를 포함한다. $$C$$의 가중치가 음이 아니라면, $$C$$를 빼서 정점이 더 적고 $$w(\pi') \le w(\pi)$$인 경로 $$\pi'$$를 만들 수 있어 $$v$$가 증인이라는 사실과 모순이다. 따라서 $$C$$는 음수 가중치 사이클이다.

공간은 다음처럼 줄일 수 있다.

- 각 $$k = 1, \dots, \lvert V \rvert$$에 대해 $$\delta(s_0, v_{k-1})$$과 $$\delta(s_0, v_k)$$만 저장하면 $$O(\lvert V \rvert)$$ 공간이면 된다.
- 전통적인 Bellman-Ford는 정점마다 값 하나만 저장하고, 모든 간선을 $$\lvert V \rvert$$ 라운드 동안 완화한다. 다만 이때의 추정값은 $$k$$-간선 거리와 일치하지 않아 분석이 더 까다롭다.
- 이런 공간 최적화 버전은 음수 가중치 사이클을 반환하지 못한다.

## 전통적인 Bellman-Ford (Recitation)

원래의 Bellman-Ford는 더 간단하게 서술되지만 조금 덜 강력하다. 같은 시간에 $$O(\lvert V \rvert)$$ 공간만 쓰지만, 음수 가중치 사이클이 있는지만 탐지하고 사이클을 반환하지는 않는다. 거리 추정값을 초기화한 뒤 그래프의 모든 간선을 $$\lvert V \rvert - 1$$ 라운드 동안 완화한다. 그 후에도 삼각 부등식을 위반하는 간선이 있으면 음수 가중치 사이클이 있다.

```python
def bellman_ford(Adj, w, s):         # Adj: 인접 리스트, w: 가중치, s: 출발점
    infinity = float('inf')          # 양수 가중치 합보다 큰 수
    d = [infinity for _ in Adj]      # 최단 경로 추정값 d(s, v)
    parent = [None for _ in Adj]     # 부모 포인터 초기화
    d[s], parent[s] = 0, s           # 출발점 초기화
    V = len(Adj)                     # 정점 수
    for k in range(V - 1):           # V - 1 라운드 동안
        for u in range(V):           # 모든 간선 (u, v)를
            for v in Adj[u]:
                try_to_relax(Adj, w, d, parent, u, v)   # 완화 시도
    for u in range(V):               # s에서 도달 가능한 음수 사이클 확인
        for v in Adj[u]:
            if d[v] > d[u] + w(u, v):    # 아직 완화 가능하면 사이클 보고
                raise Exception('Ack! There is a negative weight cycle!')
    return d, parent
```

**보조정리 1.** Bellman-Ford의 $$i$$번째 완화 라운드가 끝나면, 간선 $$i$$개 이하로 된 최단 경로를 가지는 모든 정점 $$v$$에 대해 $$d(s, v) = \delta(s, v)$$다.

**증명.** 라운드 $$i$$에 대한 귀납법이다. 0라운드 끝에는 $$s$$만 해당하며 $$d(s, s) = 0 = \delta(s, s)$$다. $$i - 1$$에서 성립한다고 하자. $$v$$가 간선 $$i - 1$$개 이하인 최단 경로를 가지면 이미 $$d(s, v) = \delta(s, v)$$이고, 완화는 안전하므로 그대로 유지된다. 아니라면 간선이 정확히 $$i$$개인 최단 경로에서 끝에서 두 번째 정점을 $$u$$라 하자. $$u$$까지는 간선 $$i - 1$$개 이하인 최단 경로가 있으므로 라운드 $$i$$ 전에 $$d(s, u) = \delta(s, u)$$다. 라운드 $$i$$에서 $$(u, v)$$를 완화하면 $$d(s, v) = \delta(s, v)$$가 된다.

음수 가중치 사이클이 없으면 최단 경로는 단순하고 간선이 $$\lvert V \rvert - 1$$개 이하이므로, $$\lvert V \rvert - 1$$ 라운드 후 모든 거리가 정확하다. 그런데도 어떤 간선 $$(u, v)$$가 여전히 삼각 부등식을 위반하면, $$v$$로 가는 간선 $$\lvert V \rvert$$개짜리 경로가 더 적은 간선의 모든 경로보다 가벼운 것이다. 이런 경로는 단순할 수 없으므로 음수 가중치 사이클을 포함한다. 각 라운드는 간선마다 상수 작업을 하므로 실행 시간은 $$O(\lvert V \rvert\lvert E \rvert)$$다.

강의 버전과의 차이는 두 가지다.

- 전통적인 Bellman-Ford는 라운드마다 추정값 한 층만 유지하고, 강의 버전은 $$k \in \{0, \dots, \lvert V \rvert\}$$의 모든 층을 유지하므로 음수 가중치 사이클을 구성할 수 있다.
- 전통적인 버전의 라운드 $$k$$에서의 추정값은 $$k$$-간선 거리와 같지 않을 수 있다. 한 라운드에 최단 경로의 여러 간선을 완화할 수 있기 때문이다. 추정값은 $$k$$-간선 거리보다 크지 않고 훨씬 작을 수도 있어, 실제로는 더 빨리 수렴할 수 있다. 어떤 라운드에서 갱신이 하나도 없으면 일찍 멈춰도 된다.

## Recitation 연습: 티볼 모임

**문제.** 세 친구가 각자 집 $$a$$, $$b$$, $$c$$에서 자전거로 한 교차로에 모여 티볼을 하려 한다. 도로마다 방향별로 "재미" 값이 있고 음수일 수도 있다. 세 사람의 총 재미가 최대인 교차로를 찾거나, 재미가 끝없이 커지는 순환 경로가 있음을 판별한다.

**풀이.** 교차로와 세 집을 정점으로, 자전거로 갈 수 있는 방향의 도로를 간선으로 두고 가중치를 재미의 음수로 한다. $$a$$에서 Bellman-Ford를 실행해 음수 가중치 사이클이 탐지되면 끝없이 재미있는 순환 경로가 있다. 없으면 $$a$$, $$b$$, $$c$$에서 각각 Bellman-Ford를 실행해 $$d(a, v) + d(b, v) + d(c, v)$$가 최소인 $$v$$를 고른다. 전체 $$O(\lvert V \rvert\lvert E \rvert)$$다.

## 정리

- 음수 가중치 사이클이 없으면 유한한 최단 경로 중 단순한 것이 있으므로, 최단 경로는 간선이 $$\lvert V \rvert - 1$$개 이하다.
- $$\delta_{\lvert V \rvert}(s, v) < \delta_{\lvert V \rvert-1}(s, v)$$인 정점 $$v$$는 음수 사이클 증인이며, 거리가 $$-\infty$$인 정점은 모두 어떤 증인에서 도달 가능하다.
- Bellman-Ford는 그래프를 $$\lvert V \rvert + 1$$개 층으로 복제한 DAG에서 DAG Relaxation을 실행해 모든 $$k$$-간선 거리 $$\delta_k(s, v)$$를 구한다.
- $$d(s, v) = \delta_{\lvert V \rvert-1}(s, v)$$로 두고, 증인에서 도달 가능한 정점은 $$-\infty$$로 바꾸면 모든 거리가 정확하며, 실행 시간은 $$O(\lvert V \rvert\lvert E \rvert)$$다.
- 증인까지의 최단 경로에서 음수 가중치 사이클을 꺼낼 수 있다. 전통적인 Bellman-Ford는 $$O(\lvert V \rvert)$$ 공간으로 $$\lvert V \rvert - 1$$ 라운드 동안 모든 간선을 완화하지만, 음수 사이클의 존재만 탐지한다.

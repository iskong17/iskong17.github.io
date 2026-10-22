---
title: "Algorithms 19: Complexity"
date: 2026-10-22 14:19:00 +0900
category: study
topic: algorithm
series: algo
lecture: 19
card_title: "Lecture 19: Complexity"
card_desc: "Decision problems · Decidability · P, NP, EXP, R · Reductions · NP-hardness and NP-completeness"
lang: ko
notes: "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/fda666a4db1dc65b3d71be08115502bd_MIT6_006S20_lec19.pdf"
description: "결정 문제와 결정 가능성, 복잡도 클래스 P, NP, EXP, R을 정의하고, 환원으로 문제의 난이도를 비교해 NP-hard와 NP-complete를 설명한 뒤 대표적인 NP-complete 문제들을 살펴본다."
---

<ul class="post-sources">
<li markdown="span">[MIT 6.006 Introduction to Algorithms (Spring 2020)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/) 19강 정리</li>
<li markdown="span">강의: Erik Demaine, Jason Ku, Justin Solomon</li>
<li markdown="span">MIT OpenCourseWare, CC BY-NC-SA</li>
<li markdown="span">[▶ 강의 영상](https://www.youtube.com/watch?v=JbafQJx1CIA)</li>
<li markdown="span">[Recitation 19 notes (PDF)](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/7232b0ff982f501556118a8552654762_MIT6_006S20_r19.pdf)</li>
</ul>

## 결정 문제

**결정 문제(decision problem)** 는 각 입력에 YES(1) 또는 NO(0)를 대응시키는 문제다. 입력은 NO 입력이거나 YES 입력이다.

| 문제 | 결정 질문 |
|---|---|
| $$s$$-$$t$$ 최단 경로 | 주어진 $$G$$에 $$s$$에서 $$t$$까지 가중치가 $$d$$ 이하인 경로가 있는가? |
| 음의 사이클 | 주어진 $$G$$에 음의 가중치 사이클이 있는가? |
| 최장 단순 경로 | 주어진 $$G$$에 가중치가 $$d$$ 이상인 단순 경로가 있는가? |
| 부분집합 합 | 주어진 정수 집합 $$A$$에 합이 $$S$$인 부분집합이 있는가? |
| 테트리스 | 주어진 보드에서 주어진 조각 수열을 버텨낼 수 있는가? |
| 체스 | 주어진 판에서 한 플레이어가 승리를 강제할 수 있는가? |
| 정지 문제 | 주어진 프로그램이 주어진 입력에서 종료하는가? |

**알고리즘(프로그램)** 은 문제를 푸는 상수 길이의 코드다($$\Omega(\log n)$$비트 워드를 가진 word-RAM에서 동작). 모든 입력에 대해 올바른 출력을 내야 하고, 코드 길이는 입력 크기와 무관하다. 문제를 유한 시간에 푸는 프로그램이 존재하면 그 문제는 **결정 가능(decidable)** 하다.

## 결정 가능성

- 프로그램은 유한한(상수 길이의) 비트 문자열, 즉 음이 아닌 정수 하나($$\in \mathbb{N}$$)다.
- 문제는 함수 $$p : \mathbb{N} \to \{0, 1\}$$, 즉 무한한 비트 문자열이다.
- 프로그램의 수는 $$\lvert \mathbb{N} \rvert$$로 가산 무한이고, 문제의 수는 $$\lvert \mathbb{R} \rvert$$로 비가산 무한이다. 프로그램 수가 문제 수보다 훨씬 적다(칸토어의 대각선 논법(Cantor's diagonalization)으로 증명).
- 따라서 **대부분의 결정 문제는 어떤 프로그램으로도 풀 수 없다(결정 불가능, undecidable).** 예컨대 정지 문제는 결정 불가능하다.
- 다행히 우리가 생각하는 대부분의 문제는 알고리즘적 구조를 가지며 결정 가능하다.

## 결정 가능한 문제의 클래스

| 클래스 | 정의 | 비고 |
|---|---|---|
| R | 유한 시간에 결정 가능한 문제 | R은 재귀 언어(recursive languages)에서 유래 |
| EXP | 지수 시간 $$2^{n^{O(1)}}$$에 결정 가능한 문제 | 우리가 생각하는 대부분의 문제 |
| P | 다항 시간 $$n^{O(1)}$$에 결정 가능한 문제 | 효율적 알고리즘, 이 과목의 초점 |

시간 계층 정리(time hierarchy theorem)에 의해 이 집합들은 서로 다르다.

$$
\text{P} \subsetneq \text{EXP} \subsetneq \text{R}
$$

예를 들어 체스는 $$\text{EXP} \setminus \text{P}$$에 속한다.

## 비결정적 다항 시간 (NP)

**P**는 어떤 알고리즘 $$A$$가 존재해, 크기 $$n$$인 모든 입력 $$I$$에 대해 $$A$$가 $$\text{poly}(n)$$ 시간에 $$I$$를 올바르게 푸는 결정 문제의 집합이다.

**NP**는 다음을 만족하는 검증 알고리즘(verification algorithm) $$V$$가 존재하는 결정 문제의 집합이다. $$V$$는 문제의 입력 $$I$$와, 길이가 $$I$$의 크기에 대해 다항인 **증명서(certificate)** 비트 문자열 $$c$$를 입력으로 받는다.

- $$V$$는 항상 $$I$$의 크기에 대해 다항 시간에 동작한다.
- $$I$$가 YES 입력이면 $$V$$가 $$(I, c)$$에 YES를 출력하는 어떤 증명서 $$c$$가 존재한다.
- $$I$$가 NO 입력이면 어떤 증명서 $$c$$를 고르든 $$V$$는 $$(I, c)$$에 항상 NO를 출력한다.

증명서는 $$I$$가 YES 입력이라는 증명으로 생각할 수 있다. $$I$$가 실제로 NO 입력이라면 어떤 증명도 통하지 않아야 한다.

| 문제 | 증명서 | 검증기 |
|---|---|---|
| $$s$$-$$t$$ 최단 경로 | $$s$$에서 $$t$$까지의 경로 $$P$$ | $$P$$의 가중치를 더해 $$\le d$$인지 확인 |
| 음의 사이클 | 사이클 $$C$$ | $$C$$의 가중치를 더해 $$< 0$$인지 확인 |
| 최장 단순 경로 | 경로 $$P$$ | $$P$$가 가중치 $$\ge d$$인 단순 경로인지 확인 |
| 부분집합 합 | 원소 집합 $$A'$$ | $$A' \subseteq A$$이고 합이 $$S$$인지 확인 |
| 테트리스 | 수(move)의 수열 | 그 수들로 살아남을 수 있는지 확인 |

![Certificates and verifiers for example NP problems](/images/study/algo/algo-19-complexity-1.webp)
*강의 노트의 증명서와 검증기 표. 각 결정 문제에 대해 YES임을 보여 주는 증명서와, 그것을 다항 시간에 확인하는 검증기를 짝지었다. 출처: MIT 6.006 Lecture 19 notes (CC BY-NC-SA 4.0).*

표의 모든 행은 같은 구조를 가진다. 가운데 열의 증명서는 "답이 YES라면 그 근거가 되는 물건"이고, 오른쪽 열의 검증기는 그 물건이 정말 근거가 되는지를 확인하는 간단한 절차다. 확인은 모두 쉽다. 경로나 사이클의 가중치를 더하는 일은 선형 시간이고, 경로가 단순한지(정점이 반복되지 않는지)는 방문 표시로 선형 시간에 검사한다. 부분집합의 합을 구하는 것도 선형 시간이다.

어려운 것은 증명서를 **찾는** 일이다. 최장 단순 경로를 예로 들면, 검증기는 주어진 경로 하나를 확인하면 되지만 그런 경로를 직접 찾으려면 지수 개의 후보를 살펴봐야 할 수도 있다. NP라는 클래스는 바로 이 "찾기는 어려울 수 있지만 확인은 쉬운" 성질로 정의된다. 표의 처음 두 행(최단 경로, 음의 사이클)은 앞의 강의들에서 본 것처럼 찾기도 쉬워 P에 속하고, 나머지 세 행은 NP-complete로 알려져 있다.

### 클래스 사이의 관계

- $$\text{P} \subseteq \text{NP}$$: 검증기 $$V$$는 증명서를 무시하고 문제를 직접 풀면 된다.
- $$\text{NP} \subseteq \text{EXP}$$: 가능한 모든 증명서를 시도한다. 증명서는 최대 $$2^{n^{O(1)}}$$개이고 각각에 $$V$$를 돌린다.
- 미해결 문제: $$\text{P} = \text{NP}$$인가? $$\text{NP} = \text{EXP}$$인가?
- 대부분은 $$\text{P} \subsetneq \text{NP}$$ (그리고 $$\subsetneq \text{EXP}$$)라고 믿는다. 즉 해를 만드는 것이 해를 검사하는 것보다 어렵다는 것이다.
- 어느 쪽이든 증명하면 큰돈을 받는다(밀레니엄 상금 100만 달러).

왜 중요한가? 어떤 문제가 NP에서 가장 어려운 문제임을 보이면, $$\text{P} \ne \text{NP}$$인 한 그 문제는 다항 시간에 풀 수 없다. 문제의 난이도를 비교하는 도구가 환원이다.

## 환원

문제 A를 풀고 싶다고 하자. 한 가지 방법은 A를 풀 줄 아는 문제 B로 변환하는 것이다. B의 알고리즘으로 풀고 그 결과로 A의 해를 계산한다. 이를 A에서 B로의 **환원(reduction)** 이라 하고 $$A \to B$$로 쓴다. B를 써서 A를 풀 수 있으므로 B는 적어도 A만큼 어렵다($$A \le B$$).

일반적인 알고리즘 전략도 결국 풀 줄 아는 문제로 환원하는 것이다.

| A | 변환 | B |
|---|---|---|
| 가중치 없는 최단 경로 | 모든 간선에 같은 가중치 부여 | 가중치 있는 최단 경로 |
| 정수 가중치 최단 경로 | 간선을 잘게 나눔(subdivide) | 가중치 없는 최단 경로 |
| 최장 경로 | 가중치의 부호를 뒤집음 | 최단 경로 |

![Examples of reductions between shortest-path problems](/images/study/algo/algo-19-complexity-2.webp)
*강의 노트의 환원 예. 왼쪽 문제 A를 가운데 열의 변환으로 오른쪽 문제 B의 입력으로 바꾼다. 출처: MIT 6.006 Lecture 19 notes (CC BY-NC-SA 4.0).*

세 행 모두 앞의 강의에서 이미 써 본 기법이다. 첫 행은 가중치 없는 그래프의 모든 간선에 가중치 1을 주면 가중 최단 경로 알고리즘(예: Dijkstra)으로 풀린다는 뜻이다. 둘째 행은 11강에서 본 것처럼 가중치 $$w$$인 간선을 길이 $$w$$인 경로로 쪼개면 BFS로 풀린다는 뜻이다. 셋째 행은 가중치의 부호를 바꾸면 최장 경로가 최단 경로가 된다는 뜻이다.

각 환원에는 조건이 따른다는 점도 표에서 읽어야 한다. 둘째 행의 변환은 가중치가 양의 정수이고 그 합이 작을 때만 효율적이며, 가중치가 크면 정점 수가 폭발한다. 셋째 행은 부호를 바꾼 그래프에 음수 가중치 사이클이 생길 수 있으므로, 원래 그래프가 DAG일 때만 의미가 있다. 일반 그래프의 최장 **단순** 경로는 NP-complete이므로 이런 간단한 변환으로 풀 수 없다. 환원은 "B를 풀 수 있으면 A도 풀 수 있다"는 방향만 보장하고, 그 비용은 변환에 걸리는 시간과 B의 입력 크기에 달려 있다.

Recitation 19에는 곱 가중치 최단 경로(product weighted shortest path)를 로그를 취해 합 가중치 최단 경로로 환원하는 예도 있다.

### NP-hard와 NP-complete

- 문제 A가 **NP-hard**라는 것은 NP의 모든 문제가 A로 다항 시간 환원된다는 뜻이다. 즉 A는 NP의 모든 문제만큼 어렵다(모든 $$X \in \text{NP}$$에 대해 $$X \le A$$).
- **NP-complete** = NP $$\cap$$ NP-hard.
- 모든 NP-complete 문제는 서로 동치다. 즉 서로에게 환원된다.
- 첫 NP-complete 문제는? 모든 결정 문제는 논리 회로를 만족시키는 문제로 환원되며, 이 문제를 회로 충족 가능성(Circuit SAT)이라 한다.
- 최장 단순 경로와 테트리스는 NP-complete다. 따라서 $$\text{NP} \setminus \text{P}$$에 속하는 문제가 하나라도 있다면 이 문제들도 그렇다.
- 체스는 **EXP-complete**다. EXP에 속하고 EXP의 모든 문제가 체스로 환원된다. 따라서 체스는 P에 속하지 않는다.

![Informal problem difficulty line with P, NP, EXP, R and hardness classes](/images/study/algo/algo-19-complexity-3.webp)
*문제 난이도를 한 줄로 늘어놓은 비공식적인 그림. 초록 화살표는 클래스(그 점보다 쉬운 쪽), 빨간 화살표는 hard(그 점보다 어려운 쪽)를 나타내고, 파란 점이 complete 문제의 위치다. 출처: MIT 6.006 Lecture 19 notes (CC BY-NC-SA 4.0).*

그림은 왼쪽으로 갈수록 쉽고 오른쪽으로 갈수록 어려운 문제가 놓인 수직선이다. 초록 화살표는 모두 왼쪽을 가리키는데, 예를 들어 NP는 "NP 점과 그보다 왼쪽에 있는 모든 문제"이므로 P를 포함한다. 같은 방식으로 $$\text{P} \subseteq \text{NP} \subseteq \text{EXP} \subseteq \text{R}$$이 화살표의 포함 관계로 보인다. 반대로 빨간 화살표는 오른쪽을 가리킨다. NP-hard는 "NP의 어떤 문제보다도 어렵거나 같은" 문제들이므로 NP 점의 오른쪽 전체다.

파란 점은 초록과 빨간 화살표가 만나는 곳, 곧 클래스에 속하면서 동시에 hard인 문제다. NP-complete는 NP의 가장 오른쪽 끝, EXP-complete는 EXP의 가장 오른쪽 끝에 있다. 이 그림에서 "NP-complete 문제 하나라도 P에 있으면 P = NP"라는 사실이 직관적으로 보인다. NP의 가장 어려운 문제가 P의 점 안으로 들어오면, 그보다 쉬운 NP 전체가 함께 끌려 들어오기 때문이다. 다만 점들 사이의 간격은 실제로 다른지 아직 모르는 경우(P와 NP, NP와 EXP)를 포함하므로 "비공식적"이라는 단서가 붙는다.

Recitation 19는 0-1 배낭이 NP-hard임을 분할(partition) 문제로부터의 환원으로 보인다. 분할 입력 $$a_i$$에 대해 $$s_i = v_i = a_i$$, $$S = \frac{1}{2}\sum a_i$$로 둔 배낭 문제를 풀면 분할 문제의 답을 얻는다. 0-1 배낭(의 결정 버전)은 NP에도 속하므로 NP-complete다.

## NP-complete 문제의 예

- **부분집합 합**(18강): **약 NP-complete(weakly NP-complete)** 다. 그래서 의사 다항 시간 알고리즘은 있지만, $$\text{P} = \text{NP}$$가 아니면 다항 시간 알고리즘은 없다.
- **3-분할(3-Partition)**: $$n$$개의 정수를 합이 같은 세 개짜리 묶음들로 나눌 수 있는가? **강 NP-complete(strongly NP-complete)** 다. $$\text{P} = \text{NP}$$가 아니면 의사 다항 시간 알고리즘도 없다.
- **직사각형 채우기(Rectangle Packing)**: 직사각형 $$n$$개와, 넓이가 그 합과 같은 목표 직사각형이 주어질 때 겹치지 않게 채울 수 있는가?
  - 3-분할에서의 환원: 정수 $$a_i$$를 $$1 \times a_i$$ 직사각형으로 바꾸고, 목표 직사각형을 $$n/3 \times \left(\sum_i a_i\right)/3$$으로 둔다.
- **직소 퍼즐(jigsaw puzzle)**: 돌기/홈이 모호할 수 있는 조각 $$n$$개를 맞출 수 있는가?
  - 직사각형 채우기에서의 환원: 유일하게 맞는 돌기/홈으로 직사각형과 직사각형 경계를 강제로 만들고, 나머지 모든 경계에는 모호한 돌기/홈 하나를 쓴다.
- $$n$$개 문자열의 최장 공통 부분 수열 (두 문자열이면 16강의 $$O(\lvert A \rvert \lvert B \rvert)$$ 동적 계획법).
- 그래프의 최장 단순 경로.
- 외판원 문제(Traveling Salesman Problem): 주어진 그래프의 모든 정점을 방문하는 최단 경로(결정 버전: 최소 가중치가 $$d$$ 이하인가?).
- 3차원 장애물 사이의 최단 경로.
- 주어진 그래프의 3-색칠(3-coloring). 2-색칠은 P에 속한다.
- 주어진 그래프의 최대 클릭(largest clique).
- SAT: AND, OR, NOT으로 만든 불리언 식이 참이 될 수 있는가? 예: $$x \text{ AND NOT } x$$는 NO 입력이다.
- 지뢰찾기, 스도쿠 등 대부분의 퍼즐.
- 슈퍼 마리오, 젤다의 전설, 포켓몬 등 대부분의 비디오 게임은 NP-hard다(더 어려운 것도 많다).

## 정리

- 결정 문제는 입력에 YES/NO를 대응시키는 문제이며, 프로그램은 상수 길이 코드다.
- 프로그램은 가산 개, 문제는 비가산 개이므로 대부분의 결정 문제는 결정 불가능하다. 정지 문제가 대표적이다.
- $$\text{P} \subsetneq \text{EXP} \subsetneq \text{R}$$이며, 체스는 EXP에 속하지만 P에는 속하지 않는다.
- NP는 다항 길이 증명서를 다항 시간에 검증할 수 있는 문제의 클래스이고, $$\text{P} \subseteq \text{NP} \subseteq \text{EXP}$$다. $$\text{P} = \text{NP}$$ 여부는 미해결이다.
- $$A \to B$$ 환원은 B로 A를 푸는 것이며 $$A \le B$$를 뜻한다. NP-hard는 NP의 모든 문제가 환원되는 문제, NP-complete는 NP $$\cap$$ NP-hard다.
- 부분집합 합은 약 NP-complete라서 의사 다항 알고리즘이 있고, 3-분할은 강 NP-complete라서 그마저 없다($$\text{P} \ne \text{NP}$$ 가정).
- 최장 단순 경로, 외판원 문제, 3-색칠, SAT, 대부분의 퍼즐과 게임이 NP-complete 또는 NP-hard다.

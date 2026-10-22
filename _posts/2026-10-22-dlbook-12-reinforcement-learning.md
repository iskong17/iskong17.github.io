---
title: "Deep Learning & PyTorch 12: Reinforcement Learning"
date: 2026-10-22 17:12:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 12
card_title: "Chapter 12: Reinforcement Learning"
card_desc: "Markov decision processes · Value functions and the Bellman equations · Dynamic programming · Q-learning · Deep Q-network on CartPole · Monte Carlo tree search"
lang: ko
notes: "https://thebook.io/080289/0828/"
notes_label: "thebook.io chapter 12"
description: "마르코프 프로세스에서 마르코프 결정 과정까지 강화 학습의 수학적 틀을 쌓고, 벨만 방정식과 다이나믹 프로그래밍, 큐-러닝을 거쳐 PyTorch로 CartPole을 푸는 DQN을 구현한 뒤, 몬테카를로 트리 탐색의 원리를 정리한다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 12장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap12)</li>
</ul>

12장은 강화 학습의 기초를 다룬다. 앞 절반은 마르코프 결정 과정과 벨만 방정식이라는 이론의 뼈대이고, 뒤 절반은 큐-러닝에서 출발해 PyTorch로 딥 큐-네트워크(DQN)를 구현하는 실습과 몬테카를로 트리 탐색 소개다.

## 12.1 강화 학습이란

[📖 thebook.io](https://thebook.io/080289/0829/)

**강화 학습(reinforcement learning)** 은 정답 레이블 대신 **보상(reward)** 이라는 신호만으로 학습하는 분야다. 학습 주체가 어떤 상황에서 어떤 행동을 해 보고, 그 결과가 좋았는지 나빴는지를 보상(또는 벌칙)으로 돌려받으며, 이 시행착오를 반복해 보상을 많이 받는 행동 방식을 스스로 익힌다.

구성 요소는 두 가지다.

- **에이전트(agent)**: 행동을 결정하는 주체. 게임으로 치면 플레이어다.
- **환경(environment)**: 에이전트가 행동을 하면 그 결과(다음 상황과 보상)를 돌려주는 세계. 게임으로 치면 게임기다.

상호작용은 매 시점 $$t$$마다 다음처럼 돈다. 에이전트가 환경의 **상태(state)** $$S_t$$를 관찰하고, **행동(action)** $$A_t$$를 고르면, 환경은 보상 $$R_{t+1}$$(양수, 음수, 0)과 다음 상태 $$S_{t+1}$$을 돌려준다. 가능한 모든 상태의 집합을 $$\mathcal{S}$$, 모든 행동의 집합을 $$\mathcal{A}$$라 하면 $$S_t = s \in \mathcal{S}$$, $$A_t = a \in \mathcal{A}$$로 쓴다. 강화 학습의 목표는 이 상호작용 속에서 받는 보상의 총합을 최대화하도록 에이전트를 학습시키는 것이다.

## 12.2 마르코프 결정 과정

[📖 thebook.io](https://thebook.io/080289/0830/)

강화 학습 문제는 **마르코프 결정 과정(MDP)** 으로 표현한다. MDP는 마르코프 프로세스에 보상을 더하고(MRP), 거기에 행동을 더한 것이다. 이 순서대로 쌓아 올린다.

| 모델 | 구성 요소 |
| --- | --- |
| 마르코프 프로세스(MP) | 상태, 상태 전이 확률 |
| 마르코프 보상 프로세스(MRP) | MP + 보상, 할인율 |
| 마르코프 결정 과정(MDP) | MRP + 행동, 정책 |

### 12.2.1 마르코프 프로세스

**마르코프 프로세스(Markov Process, MP)** 는 상태가 이산적인 시간 간격으로 바뀌는 확률 과정(stochastic process)으로, 다음 상태가 오직 현재 상태에만 의존한다. 상태들이 사슬처럼 이어진다고 해서 **마르코프 체인(Markov chain)** 이라고도 한다.

다음 상태가 과거 이력과 무관하게 현재 상태만으로 정해지는 성질을 **마르코프 특성(Markov property)** 이라 한다. 과거 전체를 알 때와 현재만 알 때 다음 상태의 확률이 같다는 뜻이다.

$$
P(S_{t+1} \mid S_t) = P(S_{t+1} \mid S_1, S_2, \dots, S_t)
$$

상태가 바뀌는 것을 **전이(transition)** 라 하고, 그 확률을 **상태 전이 확률(state transition probability)** 이라 한다.

$$
P_{ss'} = P(S_{t+1} = s' \mid S_t = s)
$$

책의 예는 병원에 간 하루를 대기, 진찰, 독서, 웹 서핑, 취침의 다섯 상태로 나눈 마르코프 체인이다. 예를 들어 대기 상태에서는 0.8의 확률로 진찰, 0.1로 독서, 0.1로 웹 서핑으로 넘어가고, 취침은 하루가 끝나는 종료 상태다. 한 상태에서 나가는 전이 확률의 합은 항상 1이다. 이 확률들을 "행 = 현재 상태, 열 = 다음 상태"로 정리한 것이 **상태 전이 확률 행렬(state transition probability matrix)** 이며, 각 행의 합이 1이다.

마르코프 특성을 "현재 상태가 다음 상태를 하나로 확정한다"로 오해하면 안 된다. 다음 상태는 여전히 확률적으로 정해진다. 요점은 그 확률을 계산하는 데 현재 상태 하나면 충분하고, 그 전에 어떤 경로를 거쳐 왔는지는 필요 없다는 것이다.

### 12.2.2 마르코프 보상 프로세스

**마르코프 보상 프로세스(Markov Reward Process, MRP)** 는 마르코프 프로세스의 각 상태(또는 전이)에 보상을 붙인 모델이다. 한 시점 이후로 받는 보상의 합을 **리턴(return)** 이라 한다.

그런데 같은 보상이라도 언제 받느냐에 따라 가치가 다르다. 돈으로 치면 지금의 100만 원은 이자가 붙어 1년 뒤에 더 큰 돈이 되므로, 거꾸로 1년 뒤의 100만 원을 지금 기준으로 환산하면 100만 원보다 작다. 이 차이를 반영하는 것이 **할인율(discount factor)** $$\gamma$$(0 이상 1 이하)이며, 한 단계 미래의 보상마다 $$\gamma$$를 한 번씩 곱한다.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

- $$\gamma = 0$$이면 바로 다음 보상만 보는 근시안적인 에이전트가 된다.
- $$\gamma$$가 1에 가까울수록 먼 미래의 보상까지 고려하는 원시안적인 에이전트가 된다.

상태 $$s$$에서 출발했을 때 기대되는 리턴을 그 상태의 **가치(value)** 라 하고, 상태를 가치로 보내는 함수를 **가치 함수(value function)** 라 한다.

$$
v(s) = \mathbb{E}[G_t \mid S_t = s]
$$

미래에 얼마나 좋은 일이 기대되는지를 나타내므로, 가치 함수를 정확히 알면 가치가 큰 쪽으로 움직이기만 하면 된다. 그래서 가치 함수를 잘 추정하는 것이 강화 학습의 핵심이다.

가치는 "지금 받는 보상 + 할인된 다음 상태 가치의 기댓값"으로 계산할 수 있다(이 관계는 12.3절의 벨만 방정식이다). 병원 예에서 독서의 보상은 +10, 대기는 -2이고, 독서 다음에는 웹 서핑(0.3)이나 취침(0.7, 가치 0)으로, 대기 다음에는 웹 서핑(0.1), 독서(0.1), 진찰(0.8)로 간다고 하자. 웹 서핑의 가치를 -2.4, 진찰의 가치를 1.2로 두고 할인율을 바꿔 가며 계산하면 다음과 같다.

- $$\gamma = 0$$: 미래를 전혀 보지 않으므로 가치가 곧 즉시 보상이다. 독서 $$= 10 + 0 = 10$$, 대기 $$= -2 + 0 = -2$$.
- $$\gamma = 0.9$$: 독서 $$= 10 + 0.9 \times (-2.4 \times 0.3 + 0 \times 0.7) \approx 9.4$$, 대기 $$= -2 + 0.9 \times (-2.4 \times 0.1 + 9.4 \times 0.1 + 1.2 \times 0.8) \approx -0.5$$.
- $$\gamma = 1$$(효과를 보려고 웹 서핑 -24, 진찰 0.3으로 가정): 독서 $$= 10 + (-24 \times 0.3) = 2.8$$, 대기 $$= -2 + (-24 \times 0.1 + 2.8 \times 0.1 + 0.3 \times 0.8) \approx -3.9$$.

할인율이 클수록 다음 상태의 좋고 나쁨이 현재 가치에 크게 반영된다. 독서 자체는 보상이 크지만 그 뒤에 웹 서핑으로 빠질 위험이 있어서, 미래를 많이 볼수록 독서의 가치가 깎인다.

### 12.2.3 마르코프 결정 과정

**마르코프 결정 과정(Markov Decision Process, MDP)** 은 MRP에 에이전트의 행동을 더한 모델이다. 이제 다음 상태와 보상이 현재 상태뿐 아니라 에이전트가 고른 행동에도 달려 있다. MDP의 목표는 각 상태에서 장기적인 보상을 최대화하는 행동을 정하는 것이다.

상태마다 각 행동을 고를 확률을 정해 둔 것을 **정책(policy)** $$\pi$$라 한다.

$$
\pi(a \mid s) = P(A_t = a \mid S_t = s)
$$

정책 $$\pi$$를 고정하면 MDP는 다시 MRP가 된다. 상태 $$s$$에서 $$s'$$으로 갈 확률과 상태 $$s$$에서 기대되는 보상은 행동별 값을 정책 확률로 평균 낸 것이다($$P_{ss'}^a$$는 행동 $$a$$를 했을 때의 전이 확률, $$R_s^a$$는 그때의 기대 보상).

$$
P_{ss'}^{\pi} = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, P_{ss'}^{a}, \qquad R_s^{\pi} = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, R_s^{a}
$$

MDP에는 가치 함수가 두 가지다.

**상태-가치 함수**

**상태-가치 함수(state-value function)** $$v_\pi(s)$$는 상태 $$s$$에서 시작해 정책 $$\pi$$를 따를 때 기대되는 리턴이다. 리턴을 즉시 보상과 나머지로 쪼개면, 현재 상태의 가치가 다음 상태의 가치로 표현된다.

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi[R_{t+1} + \gamma \, v_\pi(S_{t+1}) \mid S_t = s]
$$

첫 항은 다음 시점에 바로 받는 보상, 둘째 항은 할인된 미래 가치다.

**행동-가치 함수**

**행동-가치 함수(action-value function)** $$q_\pi(s, a)$$는 상태 $$s$$에서 일단 행동 $$a$$를 하고 그 뒤로는 정책 $$\pi$$를 따를 때 기대되는 리턴이다. 큐-함수(Q-function)라고도 부른다.

$$
q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi[R_{t+1} + \gamma \, q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]
$$

가치 함수를 연립 방정식으로 정확히 풀면 상태 수 $$n$$에 대해 $$O(n^3)$$의 계산이 필요하다. 상태가 많으면 불가능하므로 다음과 같은 방법들이 발전해 왔다. 뒤의 방법일수록 앞 방법의 한계를 보완한다.

1. **다이나믹 프로그래밍(dynamic programming)**: 모든 상태와 전이 확률을 알고 상태, 행동 수가 적을 때 가치와 최적 행동을 정확히 계산한다.
2. **몬테카를로 방법(Monte Carlo method)**: 환경의 모델을 몰라도, 에피소드를 끝(종료 상태)까지 실제로 진행해 본 뒤 얻은 리턴으로 방문한 상태들의 가치를 추정한다.
3. **시간 차 학습(temporal difference learning)**: 에피소드가 끝나기를 기다리지 않고, 한 걸음 진행할 때마다 "보상 + 다음 상태의 추정 가치"로 현재 상태의 가치를 바로 갱신한다. 다이나믹 프로그래밍과 몬테카를로의 중간이며, 큐-러닝이 여기에 속한다.
4. **함수 근사 학습(function approximation)**: 상태가 너무 많거나 연속적이면 표로 저장할 수 없으므로, 상태의 특성 벡터를 입력으로 받는 함수(예: 신경망)로 가치를 근사하고 그 가중치를 학습한다. DQN이 여기에 속한다.

## 12.3 MDP를 위한 벨만 방정식

[📖 thebook.io](https://thebook.io/080289/0843/)

**벨만 방정식(Bellman equation)** 은 현재 상태(또는 상태-행동 쌍)의 가치를 다음 상태의 가치로 나타내는 재귀 관계식이다. 정책을 따를 때의 **벨만 기대 방정식** 과 최적 정책을 따를 때의 **벨만 최적 방정식** 이 있다.

### 12.3.1 벨만 기대 방정식

먼저 리턴의 재귀 구조를 보자. 할인율로 묶으면 $$t+2$$ 이후의 보상은 다음 시점의 리턴이 된다.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + \cdots) = R_{t+1} + \gamma G_{t+1}
$$

양변에 기댓값을 취하면 $$v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s]$$이다. 즉 현재 가치는 즉시 보상과 할인된 다음 가치의 합이며, 미래의 가치가 현재 가치에 재귀적으로 들어온다. 이것이 **벨만 기대 방정식(Bellman expectation equation)** 이다.

기댓값을 풀어 쓰면 두 단계로 나뉜다.

**상태-가치 함수**: 상태 $$s$$의 가치는 정책이 각 행동을 고를 확률로 행동-가치를 가중 평균한 것이다.

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, q_\pi(s, a)
$$

**행동-가치 함수**: 상태 $$s$$에서 행동 $$a$$를 했을 때의 가치는 즉시 보상에, 그 행동으로 갈 수 있는 다음 상태들의 가치를 전이 확률로 평균 내고 할인한 것을 더한 것이다.

$$
q_\pi(s, a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \, v_\pi(s')
$$

두 식을 서로 대입하면 한 종류의 가치 함수만으로 된 식을 얻는다.

$$
v_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \, v_\pi(s') \right)
$$

$$
q_\pi(s, a) = R_s^a + \gamma \sum_{s' \in \mathcal{S}} P_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' \mid s') \, q_\pi(s', a')
$$

이 관계를 그림으로 나타낸 것이 **백업 다이어그램(backup diagram)** 이다. 맨 위 상태 노드 $$s$$에서 정책에 따라 행동 노드 $$a$$들로 갈라지고, 각 행동 노드에서 다시 전이 확률에 따라 다음 상태 노드 $$s'$$들로 갈라지는 2단 트리를 그리고, 아래쪽 값들을 확률로 평균 내어 위로 올려 보낸다(back up). 상태 노드의 값은 아래 행동 노드 값의 정책 가중 평균이고, 행동 노드의 값은 보상 더하기 아래 상태 노드 값의 전이 확률 가중 평균이다.

학습 관점에서 정리하면 다음 순서다.

1. 가치 함수를 임의의 값(0이나 난수)으로 초기화한다.
2. 환경과 상호작용하며 보상과 상태 정보를 모은다.
3. 벨만 기대 방정식으로 가치 함수를 갱신해, 어떤 상태와 행동이 더 큰 보상으로 이어지는지 점점 정확히 안다.
4. 2와 3을 반복하며 가치가 큰 행동을 고르도록 정책을 개선한다.

### 12.3.2 벨만 최적 방정식

강화 학습이 최종적으로 원하는 것은 가치 함수 자체가 아니라 보상을 최대로 하는 정책, 즉 **최적 정책(optimal policy)** 이다. 최적 정책을 따를 때의 가치 함수가 **최적 가치 함수(optimal value function)** 다.

$$
v_\ast(s) = \max_\pi v_\pi(s), \qquad q_\ast(s, a) = \max_\pi q_\pi(s, a)
$$

$$v_\ast$$는 어떤 상태가 최대 얼마의 보상으로 이어질 수 있는지, $$q_\ast$$는 어떤 상태에서 어떤 행동을 해야 그 최대 보상을 얻는지 알려 준다. $$q_\ast$$를 알면 최적 정책은 그냥 각 상태에서 $$q_\ast$$가 가장 큰 행동을 고르는 것이다.

$$
\pi_\ast(a \mid s) =
\begin{cases}
1 & \text{if } a = \arg\max_{a \in \mathcal{A}} q_\ast(s, a) \\
0 & \text{otherwise}
\end{cases}
$$

최적 가치 함수가 만족하는 재귀식이 **벨만 최적 방정식(Bellman optimality equation)** 이다. 기대 방정식의 정책 평균이 최댓값으로 바뀐다.

$$
v_\ast(s) = \max_{a} \left( R_s^a + \gamma \sum_{s'} P_{ss'}^a \, v_\ast(s') \right), \qquad q_\ast(s, a) = R_s^a + \gamma \sum_{s'} P_{ss'}^a \max_{a'} q_\ast(s', a')
$$

### 12.3.3 다이나믹 프로그래밍

**다이나믹 프로그래밍(dynamic programming)** 은 큰 문제를 작은 부분 문제로 나눠 그 답을 재사용하며 푸는 최적화 기법이다(여기서 programming은 코딩이 아니라 수학적 계획을 뜻한다). 강화 학습에서는 MDP의 전이 확률과 보상을 모두 안다고 가정한다. 행동의 결과를 미리 알기 때문에 실제로 움직여 보지 않고 계산만으로 계획(planning)할 수 있다.

- **예측(prediction)**: MDP와 정책이 주어졌을 때 그 정책의 가치 함수를 구한다.
- **제어(control)**: MDP가 주어졌을 때 최적 가치 함수와 최적 정책을 구한다.

**정책 이터레이션**

**정책 이터레이션(policy iteration)** 은 평가와 개선을 번갈아 반복한다.

- **정책 평가(policy evaluation)**: 현재 정책의 가치 함수를 구한다. 모든 상태에 대해 벨만 기대 방정식의 우변(다음 상태 가치와 즉시 보상)으로 가치를 다시 계산하는 것을 반복하면, 처음 값이 틀렸더라도 실제 $$v_\pi$$로 수렴한다.

$$
v_{k+1}(s) = \sum_{a} \pi(a \mid s) \left( R_s^a + \gamma \sum_{s'} P_{ss'}^a \, v_k(s') \right)
$$

- **정책 개선(policy improvement)**: 평가된 가치로 정책을 더 좋게 바꾼다. 가장 흔한 방법은 **탐욕적 정책 개선(greedy policy improvement)** 으로, 각 상태에서 행동-가치가 가장 큰 행동만 고르도록 정책을 바꾼다. 이렇게 바꾼 정책의 가치는 이전 정책보다 항상 크거나 같으므로, 반복하면 최적 정책에 수렴한다.

$$
\pi'(s) = \arg\max_{a \in \mathcal{A}} q_\pi(s, a)
$$

**가치 이터레이션**

**가치 이터레이션(value iteration)** 은 정책을 따로 두지 않고, 처음부터 최적 정책을 따른다고 보고 벨만 최적 방정식으로 가치를 반복 갱신한다. 가치가 수렴하면 그 가치에 대해 탐욕적으로 행동하는 것이 곧 최적 정책이므로, 별도의 정책 개선 단계가 필요 없다.

$$
v_{k+1}(s) = \max_{a \in \mathcal{A}} \left( R_s^a + \gamma \sum_{s'} P_{ss'}^a \, v_k(s') \right)
$$

## 12.4 큐-러닝

[📖 thebook.io](https://thebook.io/080289/0856/)

다이나믹 프로그래밍은 환경의 전이 확률과 보상을 모두 알아야 쓸 수 있다. **큐-러닝(Q-learning)** 은 그런 환경 모델 없이(model-free) 실제 경험만으로 MDP의 최적 정책을 찾는 강화 학습 기법이다.

### 12.4.1 큐-러닝

큐-러닝은 상태-행동 쌍마다 기대 리턴의 추정치 $$Q(s, a)$$를 표(**큐-테이블, Q-table**)로 저장하고, 여러 번의 에피소드(episode)를 반복하며 이 표를 최적 행동-가치 $$q_\ast$$에 가깝게 갱신한다.

**활용과 탐험**

학습 중에는 두 가지 사이에서 균형을 잡아야 한다.

- **활용(exploitation)**: 지금까지의 경험상 가장 큰 보상을 줄 것으로 보이는 행동을 한다.
- **탐험(exploration)**: 더 좋은 길이 있을지 모르니 아직 해 보지 않은 행동을 시도한다. 경험을 넓히지만 당장은 시간과 보상을 손해 볼 수 있다.

흔한 방법이 **엡실론 탐욕(epsilon-greedy)** 정책이다. 0~1 사이 난수가 임계값 $$\epsilon$$(예: 0.1)보다 작으면 무작위 행동(탐험)을, 아니면 Q 값이 가장 큰 행동(활용)을 한다. 학습이 진행될수록 $$\epsilon$$을 줄여 결국 거의 활용만 하게 만든다. 책이 보여 주는 코드는 Q 값에 점점 작아지는 무작위 잡음을 더한 뒤 최댓값을 고르는 방식으로 같은 효과를 낸다.

```python
while not done:
    # Q 값 + (에피소드가 지날수록 작아지는) 잡음 -> 초반엔 탐험, 후반엔 활용
    action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))
    new_state, reward, done, _ = env.step(action)   # 행동 실행 -> 다음 상태, 보상, 종료 여부
    Q[state, action] = reward + dis * np.max(Q[new_state, :])   # 보상 + 할인된 다음 상태 최대 Q
    rAll += reward
    state = new_state
```

**학습 절차**

1. 큐-테이블의 모든 값을 0으로 초기화한다. 행은 상태, 열은 행동이다.

```python
Q = np.zeros([env.observation_space.n, env.action_space.n])
```

2. 현재 상태 $$s$$에서 행동 $$a$$를 골라 실행한다.
3. 보상 $$r$$과 다음 상태 $$s'$$을 관찰한다.
4. $$s'$$에서 가능한 행동 중 Q 값이 가장 큰 것을 찾는다.
5. 다음 식으로 $$Q(s, a)$$를 갱신한다.
6. 종료 상태에 이를 때까지 2~5를 반복한다.

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left( R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right)
$$

괄호 안의 $$R_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a')$$가 **목표값(target)** 이다. 다음 상태에서는 최선의 행동을 한다고 가정했을 때의 가치이며, 이것과 현재 추정치의 차이(시간 차 오차)에 학습률 $$\alpha$$를 곱한 만큼 움직인다. 위 코드는 $$\alpha = 1$$인 경우, 즉 목표값으로 그냥 덮어쓰는 단순한 형태다. 실제로 하는 행동(탐험이 섞인 행동)과 갱신에 쓰는 행동(최대 Q 행동)이 다르기 때문에 큐-러닝을 오프-폴리시(off-policy) 방법이라 한다.

표 기반 큐-러닝은 두 가지 이유로 현실 문제에서 잘 작동하지 않는다.

- 상태가 많거나 연속적이면(예: 화면 픽셀) 큐-테이블을 만들 수 없다.
- 연속된 경험은 서로 강하게 상관되어 있어, 들어오는 순서대로 학습하면 불안정하다.

이 한계를 신경망으로 보완한 것이 DQN이다.

### 12.4.2 딥 큐-러닝

**딥 큐-러닝(Deep Q-Learning)**, 또는 **DQN(Deep Q-Network)** 은 큐-테이블 대신 신경망 $$Q(s, a; \theta)$$로 큐-함수를 근사한다. 상태(예: 게임 화면)를 입력받아 행동마다 Q 값을 출력하는 합성곱 신경망을 쓰면, 표로는 다룰 수 없는 고차원 상태에서도 학습할 수 있다. 안정적인 학습을 위해 두 가지 장치를 더한다.

**타깃 네트워크**

큐-러닝의 목표값에도 Q가 들어 있어서, 신경망을 갱신하면 목표값도 함께 움직인다. 쫓아가는 대상이 계속 도망가는 셈이라 학습이 흔들린다. DQN은 구조는 같고 가중치만 다른 **타깃 네트워크(target network)** $$\theta^-$$를 따로 두고 목표값 계산에만 쓴다. 타깃 네트워크는 매 스텝이 아니라 일정 주기마다 학습 중인 네트워크의 가중치를 복사해 와서, 그 사이에는 목표값이 고정된다. 손실은 목표값과 예측의 차이로 정의한다(원 논문은 평균 제곱 오차, 아래 예제는 smooth L1).

$$
L(\theta) = \mathbb{E}\left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

**리플레이 메모리**

**리플레이 메모리(replay memory)** 는 에이전트가 겪은 경험 (상태, 행동, 다음 상태, 보상)을 쌓아 두는 버퍼다. 경험이 들어오는 즉시 학습하지 않고 버퍼에 모아 두었다가, 학습할 때마다 버퍼에서 무작위로 미니배치를 뽑는다. 연속된 경험 사이의 상관관계가 깨지고, 한 번 겪은 경험을 여러 번 재사용할 수 있다.

**시뮬레이션 환경**

강화 학습 실습은 주로 게임 같은 시뮬레이션 환경에서 한다. OpenAI Gym(`pip install gym`)의 환경은 세 가지 핵심 함수를 제공한다.

- `reset()`: 환경을 초기 상태로 되돌리고 첫 관찰값을 반환한다.
- `step(action)`: 행동을 실행하고 (관찰값, 보상, 종료 여부, 추가 정보)를 반환한다. 가장 자주 호출된다.
- `render()`: 현재 상태를 화면에 그린다.

예제는 **CartPole-v1** 이다. 수레(cart) 위에 막대(pole)가 경첩으로 세워져 있고, 에이전트는 매 시점 수레를 왼쪽 또는 오른쪽으로 밀어 막대가 쓰러지지 않게 한다. 버틸 때마다 +1의 보상을 받고, 막대가 너무 기울거나 수레가 화면 중앙에서 너무 멀어지면 에피소드가 끝난다. 즉 오래 버틸수록 리턴이 크다. 환경 내부 상태는 수레 위치, 수레 속도, 막대 각도, 막대 각속도의 네 값이지만, 이 예제는 그 값을 쓰지 않고 **화면 이미지** 를 입력으로 쓴다. 한 장의 그림으로는 속도를 알 수 없으므로, 현재 화면과 직전 화면의 차이를 상태로 삼는다. 행동은 왼쪽, 오른쪽 두 가지다.

이 예제는 PyTorch 공식 튜토리얼을 조금 고친 것이다. 이후 gym이 gymnasium으로 바뀌면서 `step()`의 반환값과 `render()` 사용법이 달라졌으므로, 최신 버전에서는 API를 맞춰야 한다.

**리플레이 메모리 구현**

```python
from collections import namedtuple
import random

# 경험 하나: (상태, 행동, 다음 상태, 보상). 이름과 인덱스로 모두 접근 가능
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity   # 가득 차면 가장 오래된 것부터 덮어씀

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)          # 무작위 미니배치

    def __len__(self):
        return len(self.memory)
```

**DQN 네트워크**

입력은 화면 차이 이미지(3채널), 출력은 두 행동의 Q 값이다. 합성곱 층 세 개(각각 배치 정규화와 ReLU) 뒤에 선형 층 하나를 둔다.

```python
class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # 합성곱 한 번을 거친 뒤의 크기: (size - kernel) // stride + 1
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        self.head = nn.Linear(convw * convh * 32, outputs)   # 입력 크기는 화면 크기에 따라 계산

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))   # (배치, 행동 수): 행동별 Q 값
```

선형 층의 입력 크기는 입력 이미지 크기와 합성곱 설정에 따라 달라지므로, 합성곱 출력 크기 공식을 세 번 적용해 계산한다.

**화면 가져오기**

`env.render(mode='rgb_array')`로 받은 화면(높이 x 너비 x 채널)을 `(채널, 높이, 너비)`로 바꾸고, 막대가 있는 세로 40~80% 구간만 남긴 뒤, 수레가 가운데 오도록 가로 60% 폭을 잘라 낸다. 그다음 0~1 실수로 바꾸고 높이 40픽셀로 줄여 `(1, 3, 40, 약 90)` 텐서를 만든다.

```python
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])

def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)   # 수레 중심의 픽셀 위치

def get_screen():
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))   # (C, H, W)
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height * 0.4):int(screen_height * 0.8)]   # 위아래 여백 제거
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:                      # 왼쪽 끝 근처
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):  # 오른쪽 끝 근처
        slice_range = slice(-view_width, None)
    else:                                                    # 수레를 가운데로
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)
    screen = screen[:, :, slice_range]
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255   # 연속 메모리 + 0~1 실수
    screen = torch.from_numpy(screen)
    return resize(screen).unsqueeze(0).to(device)            # 배치 차원 추가 -> (1, C, H, W)
```

- `slice(6)`은 `[:6]`과 같은 슬라이스 객체다. 범위를 변수에 담아 두었다가 인덱싱에 쓸 수 있다.
- 슬라이싱과 `transpose`를 거친 넘파이 배열은 메모리에 띄엄띄엄 놓인 뷰일 수 있다. `np.ascontiguousarray`는 값은 그대로 두고 메모리에 연속으로 배치한 배열을 만들어, 이후 연산과 텐서 변환이 빠르고 안전하게 한다.

**하이퍼파라미터와 행동 선택**

```python
BATCH_SIZE = 128
GAMMA = 0.999        # 할인율
EPS_START = 0.9      # 탐험 확률 시작값
EPS_END = 0.05       # 탐험 확률 최솟값
EPS_DECAY = 200      # 탐험 확률 감소 속도
TARGET_UPDATE = 10   # 타깃 네트워크 갱신 주기(에피소드)

init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape
n_actions = env.action_space.n   # 2

policy_net = DQN(screen_height, screen_width, n_actions).to(device)   # 학습하는 네트워크
target_net = DQN(screen_height, screen_width, n_actions).to(device)   # 목표값 계산용
target_net.load_state_dict(policy_net.state_dict())   # 같은 가중치로 시작
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    # 탐험 확률이 EPS_START에서 EPS_END로 지수적으로 감소
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:                       # 활용: Q 값이 가장 큰 행동
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:                                            # 탐험: 무작위 행동
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
```

탐험 확률은 다음과 같이 스텝 수 $$n$$에 따라 줄어든다($$\tau$$가 `EPS_DECAY`).

$$
\epsilon(n) = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \, e^{-n / \tau}
$$

**한 번의 최적화 스텝**

```python
def optimize_model():
    if len(memory) < BATCH_SIZE:          # 경험이 충분히 쌓일 때까지 대기
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))   # 경험 리스트 -> 필드별 묶음 (전치)

    # 다음 상태가 있는(종료되지 않은) 경험 표시
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a_t): 네트워크 출력 중 실제로 한 행동의 Q 값만 골라낸다
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # max_a Q(s_{t+1}, a): 타깃 네트워크로 계산, 종료 상태는 0
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch   # 목표값

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)      # 기울기 클리핑
    optimizer.step()
```

- `zip(*transitions)`: `*`는 리스트를 풀어서 각 원소를 별도 인자로 넘긴다. 그래서 `zip(*[[1, 2, 3], [4, 5, 6]])`은 `[(1, 4), (2, 5), (3, 6)]`, 즉 전치가 된다. 여기서는 "경험 128개 각각의 (상태, 행동, 다음 상태, 보상)"을 "상태 128개, 행동 128개, ..."로 바꾼다. 함수 정의의 `*args`는 반대로 여러 인자를 하나의 튜플로 모은다.
- `map(함수, 시퀀스)`는 각 원소에 함수를 적용한다. 다음 상태가 `None`(종료)인지 아닌지를 불리언 마스크로 만든다.
- `gather(1, action_batch)`는 행마다 `action_batch`가 가리키는 열의 값을 뽑는다. 신경망은 모든 행동의 Q 값을 내지만, 손실은 실제로 한 행동에 대해서만 계산한다.
- `.detach()`로 목표값 쪽으로는 기울기가 흐르지 않게 한다.

**smooth L1 손실** 은 오차가 작을 때는 제곱(L2), 클 때는 절댓값(L1)처럼 동작하는 손실이다. L1은 이상치에 덜 민감하지만 0에서 미분이 불연속이고, L2는 매끄럽지만 큰 오차에 과하게 반응한다. smooth L1은 둘의 장점을 합쳐, 큰 오차에서 기울기가 폭발하는 것을 막는다(Fast R-CNN에서 쓰여 유명해졌다). 오차를 $$z_i$$, 경계값을 $$\beta$$(기본 1.0)라 하면 다음과 같다.

$$
\text{loss}(z_i) =
\begin{cases}
0.5 \, z_i^2 / \beta & \text{if } \lvert z_i \rvert < \beta \\
\lvert z_i \rvert - 0.5 \, \beta & \text{otherwise}
\end{cases}
$$

**학습 루프**

에피소드마다 환경을 리셋하고, 매 스텝 행동을 골라 실행하고, 경험을 메모리에 넣고, `optimize_model()`로 정책 네트워크를 한 번 갱신한다. `TARGET_UPDATE` 에피소드마다 타깃 네트워크에 가중치를 복사한다.

```python
num_episodes = 50   # 제대로 학습하려면 300 이상 권장
for i_episode in range(num_episodes):
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen          # 화면 차이 = 상태 (움직임 정보 포함)
    for t in count():                             # 에피소드가 끝날 때까지 무한 반복
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = get_screen()
        next_state = current_screen - last_screen if not done else None

        memory.push(state, action, next_state, reward)   # 경험 저장
        state = next_state
        optimize_model()                                  # 미니배치 학습 한 번
        if done:
            episode_durations.append(t + 1)               # 버틴 스텝 수 기록
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())   # 타깃 네트워크 동기화
env.close()
```

실행하면 별도 창에서 수레가 좌우로 움직이며 학습이 진행된다. 50 에피소드로는 막대를 오래 세우지 못하며, 버틴 시간(`episode_durations`)이 늘어나는 것을 보려면 수백 에피소드가 필요하다.

## 12.5 몬테카를로 트리 탐색

[📖 thebook.io](https://thebook.io/080289/0880/)

**몬테카를로 트리 탐색(Monte Carlo Tree Search, MCTS)** 은 알파고에 쓰여 유명해진 탐색 알고리즘이다. 바둑처럼 둘 수 있는 수의 경우가 너무 많아 게임 트리 전체를 탐색할 수 없을 때 쓴다.

### 12.5.1 몬테카를로 트리 탐색 원리

MCTS는 모든 가지를 다 펼치는 대신, 무작위 시뮬레이션(끝까지 아무렇게나 두어 보기)을 많이 해서 승률이 높아 보이는 가지를 집중적으로 펼친다. 한 번의 반복은 네 단계다.

1. **선택(selection)**: 루트 R에서 시작해, 이미 펼쳐진 트리 안에서 가장 유망한 자식을 따라 내려가 끝 노드 L에 이른다. 유망함은 승률과 덜 시도된 정도를 함께 고려하는 UCT(Upper Confidence bounds applied to Trees) 점수로 잰다.
2. **확장(expansion)**: L에서 게임이 끝나지 않았다면 자식 노드를 하나 이상 만들고 그중 하나 C를 고른다.
3. **시뮬레이션(simulation)**: C에서 게임이 끝날 때까지 무작위로 수를 둔다(롤아웃, rollout).
4. **역전파(backpropagation)**: 시뮬레이션 결과(승패)를 C에서 L을 거쳐 R까지 경로 위의 모든 노드에 반영해 승리 횟수와 방문 횟수를 갱신한다.

선택 단계의 UCT 점수는 다음과 같다.

$$
\frac{w_i}{n_i} + C \sqrt{\frac{\ln t}{n_i}}
$$

- $$w_i$$: 노드 $$i$$를 거친 시뮬레이션 중 이긴 횟수.
- $$n_i$$: 노드 $$i$$를 거친 시뮬레이션 횟수.
- $$t$$: 부모 노드의 시뮬레이션 횟수(형제 노드 $$n_i$$의 합).
- $$C$$: 탐험 정도를 조절하는 상수. 이론적으로 $$\sqrt{2}$$를 많이 쓰지만 실험으로 조정한다.

첫 항은 지금까지의 승률(활용), 둘째 항은 적게 시도한 노드일수록 커지는 보너스(탐험)다. 강화 학습의 활용-탐험 균형이 트리 탐색에서도 그대로 나타난다. 이렇게 수많은 시뮬레이션으로 각 수의 가치를 통계적으로 추정해 최선의 수를 고르며, 알파고는 여기에 신경망으로 수의 후보와 판세를 평가하는 방식을 결합했다. 바둑 같은 보드 게임뿐 아니라 실시간 비디오 게임, 포커 같은 불완전 정보 게임에도 쓰인다.

### 12.5.2 몬테카를로 트리 탐색을 적용한 틱택토 게임 구현

책은 MCTS 예제로 흔히 쓰이는 **틱택토(tic-tac-toe)** 를 다룬다. 다만 책의 코드는 MCTS 에이전트가 아니라, 두 사람이 명령줄에서 번갈아 칸 번호를 입력해 두는 2인용 틱택토 게임 자체를 구현한 것이다. MCTS를 붙이려면 이 게임 규칙 위에 앞 절의 선택, 확장, 시뮬레이션, 역전파를 구현해야 한다.

보드는 키 `'1'`~`'9'`(키패드 배치), 값은 `' '`, `'X'`, `'Y'`인 딕셔너리다.

```python
boarder = {'1': ' ', '2': ' ', '3': ' ',
           '4': ' ', '5': ' ', '6': ' ',
           '7': ' ', '8': ' ', '9': ' '}

def visual_Board(board_num):     # 매 수마다 보드 출력
    print(board_num['1'] + '|' + board_num['2'] + '|' + board_num['3'])
    print('-+-+-')
    print(board_num['4'] + '|' + board_num['5'] + '|' + board_num['6'])
    print('-+-+-')
    print(board_num['7'] + '|' + board_num['8'] + '|' + board_num['9'])
```

게임 함수는 입력받은 칸이 비어 있으면 현재 플레이어의 표시를 놓고, 다섯 수 이상 두었으면 가로 세 줄, 세로 세 줄, 대각선 두 줄의 승리 조건을 검사한다. 책은 여덟 개의 `elif`를 나열하지만, 승리 줄을 리스트로 두면 짧게 쓸 수 있다.

```python
WIN_LINES = [('1','2','3'), ('4','5','6'), ('7','8','9'),   # 가로
             ('1','4','7'), ('2','5','8'), ('3','6','9'),   # 세로
             ('1','5','9'), ('3','5','7')]                  # 대각선

def game():
    turn, count = 'X', 0
    while count < 9:
        visual_Board(boarder)
        move = input("당신 차례입니다," + turn + ". 어디로 이동할까요?\n")
        if boarder[move] != ' ':
            print("이미 채워져 있습니다.")
            continue
        boarder[move] = turn
        count += 1
        if count >= 5 and any(boarder[a] == boarder[b] == boarder[c] != ' ' for a, b, c in WIN_LINES):
            visual_Board(boarder)
            print("게임 종료. " + turn + "가 승리했습니다.")
            return
        turn = 'Y' if turn == 'X' else 'X'   # 차례 교대
    print("게임 종료. 동점입니다")
```

(책의 원래 코드는 `for i in range(8)`로 최대 여덟 번만 입력을 받기 때문에, 이미 찬 칸을 입력하거나 아홉 수까지 가면 동점 판정에 이르지 못하고 루프가 끝난다. 위처럼 `while`로 바꾸면 이 문제가 없다.)

X가 3, 5, 7을 차지해 대각선으로 이기는 진행은 다음과 같다.

```text
Y|Y|X
-+-+-
 |X|
-+-+-
X| |
게임 종료. X가 승리했습니다.
```

강화 학습은 딥러닝에서도 가장 어렵고 넓은 분야 중 하나다. 이 장은 기초 개념과 가장 기본적인 예제만 다루었으므로, 더 깊이 공부하려면 강화 학습 전문 교재를 따로 보는 것이 좋다.

## 정리

- 강화 학습은 에이전트가 환경과 상호작용하며 상태를 보고 행동하고 보상을 받는 과정을 반복해, 할인된 보상의 합(리턴)을 최대화하는 정책을 스스로 배우는 분야다.
- 마르코프 프로세스(상태 + 전이 확률)에 보상과 할인율을 더하면 MRP, 행동과 정책을 더하면 MDP다. 다음 상태가 현재 상태에만 의존한다는 마르코프 특성이 바탕이다.
- 상태-가치 함수 $$v_\pi(s)$$와 행동-가치 함수 $$q_\pi(s, a)$$는 벨만 기대 방정식으로 "즉시 보상 + 할인된 다음 가치"라는 재귀 관계를 이루고, 최적 가치 함수는 평균 대신 최댓값을 쓰는 벨만 최적 방정식을 만족한다.
- 환경 모델을 알면 다이나믹 프로그래밍으로 풀 수 있다. 정책 이터레이션은 정책 평가와 탐욕적 정책 개선을 반복하고, 가치 이터레이션은 벨만 최적 방정식으로 가치만 반복 갱신한다.
- 큐-러닝은 모델 없이 큐-테이블을 $$Q \leftarrow Q + \alpha(r + \gamma \max Q' - Q)$$로 갱신하며, 엡실론 탐욕으로 활용과 탐험의 균형을 잡는다.
- DQN은 합성곱 신경망으로 큐-함수를 근사하고, 주기적으로만 갱신하는 타깃 네트워크와 무작위 미니배치를 뽑는 리플레이 메모리로 학습을 안정시킨다. CartPole 예제는 화면 차이를 상태로, smooth L1 손실과 기울기 클리핑을 쓴다.
- 몬테카를로 트리 탐색은 선택(UCT 점수), 확장, 무작위 시뮬레이션, 역전파를 반복해 유망한 수를 통계적으로 고르는 탐색이며, 알파고의 핵심 구성 요소다.

---
title: "Deep Learning Discussion 9: Policy Gradients & Q-Learning"
date: 2026-10-22 12:09:00 +0900
category: study
topic: deep-learning
group: discussion
card_title: "Discussion 9: Policy Gradients & Q-Learning"
card_desc: "Score function estimator · Reward-to-go and baselines · Bellman equations · Actor-critic · Q-learning and exploration"
lang: ko
problems: "https://cs182sp21.github.io/static/discussions/dis9.pdf"
solutions: "https://cs182sp21.github.io/static/discussions/dis9-sol.pdf"
description: "정책 경사를 일반적인 점수 함수 추정량으로 유도하고 보상-투-고와 기준선으로 분산을 줄이며, 상태 의존 기준선이 불편임을 보인 뒤, 벨만 방정식에 기반한 가치 함수 학습, 액터-크리틱, Q-learning과 탐험을 정리한다."
---

> [UC Berkeley CS 182 (Spring 2021)](https://cs182sp21.github.io/) Discussion 9 정리. 관련 강의: [Deep Learning 15]({% post_url 2026-10-22-dl-15-policy-gradients %}), [Deep Learning 16]({% post_url 2026-10-22-dl-16-actor-critic-q-learning %})
> 이전: [Discussion 8]({% post_url 2026-10-22-dl-dis-08-pretraining-imitation %}) · 다음: [Discussion 10]({% post_url 2026-10-22-dl-dis-10-generative %})

이번 Discussion은 강화학습의 세 축인 정책 경사, 액터-크리틱, Q-learning을 다룬다. 워크시트의 문제는 상태 의존 기준선에 관한 하나이고, 나머지는 개념과 유도를 정리한 내용이다.

## 1. 정책 경사

정책 $$\pi_\theta(a \mid s)$$는 궤적 $$\tau = ((s_1, a_1), \dots, (s_T, a_T))$$에 대한 분포를 만든다. 목표는 기대 총보상을 최대화하는 것이다.

$$
\max_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[R(\tau)] = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=1}^{T} r(s_t, a_t) \right]
$$

최적화 문제가 생겼으니 자연스럽게 확률적 경사 상승으로 푼다. 이것이 정책 경사(policy gradient) 알고리즘이다.

### 1.1 정책 경사의 유도

더 일반적인 문제부터 보자. 파라미터 $$\theta$$가 확률변수 $$X$$의 분포 $$p_\theta(X)$$를 정할 때 $$\nabla_\theta \mathbb{E}_{X \sim p_\theta}[f(X)]$$를 구하고 싶다. 먼저 다음 항등식을 확인한다.

$$
\nabla_\theta \log g(\theta) = \frac{\nabla_\theta g(\theta)}{g(\theta)} \quad\Longrightarrow\quad \nabla_\theta g(\theta) = g(\theta)\, \nabla_\theta \log g(\theta)
$$

기댓값을 적분으로 쓰고, (적절한 정칙 조건 아래) 미분과 적분의 순서를 바꾼다.

$$
\begin{aligned}
\nabla_\theta \mathbb{E}_{X \sim p_\theta}[f(X)] &= \nabla_\theta \int p_\theta(x) f(x)\, dx = \int \nabla_\theta p_\theta(x)\, f(x)\, dx \\
&= \int p_\theta(x)\, \nabla_\theta \log p_\theta(x)\, f(x)\, dx = \mathbb{E}_{X \sim p_\theta}\left[ \nabla_\theta \log p_\theta(X)\, f(X) \right]
\end{aligned}
$$

기울기가 기댓값으로 쓰였으므로 $$p_\theta$$의 샘플만으로 불편 추정량을 얻을 수 있다. $$X$$를 궤적 $$\tau$$, $$f$$를 총보상 $$R(\tau)$$로 두면 기본 정책 경사가 나온다.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(\tau)\, R(\tau) \right]
$$

궤적의 로그 확률을 펼치고(Discussion 8의 계산) $$\theta$$와 무관한 초기 상태·전이 항을 버리면 다음과 같다.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \left( \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right) R(\tau) \right]
$$

알고리즘은 현재 정책으로 궤적을 샘플링하고, 확률적 기울기를 계산하고, 정책을 갱신하는 과정을 반복한다.

### 1.2 분산 줄이기

이 알고리즘은 단순하지만 갱신마다 새 궤적을 샘플링해야 하고, 궤적 수가 적으면 기울기 추정에 잡음이 많다.

**보상-투-고(reward-to-go).** 나중의 행동은 이전 보상에 영향을 줄 수 없다. 그래서 전체 보상 $$R(\tau)$$ 대신 시각 $$t$$부터의 보상 $$R_t(\tau) = \sum_{t'=t}^{T} r(s_{t'}, a_{t'})$$를 쓴다. 이를 보이려면 $$J(\theta)$$를 시간 단계별 기대 보상의 합으로 쓴다. 시각 $$t'$$의 보상 $$r_{t'}$$의 분포는 $$t'$$까지의 행동에만 의존하므로, 각 항에 점수 함수 추정량을 적용하면 $$t'$$까지의 로그 확률만 남는다.

$$
\nabla_\theta J(\theta) = \sum_{t'=1}^{T} \nabla_\theta \mathbb{E}_{\pi_\theta}[r_{t'}] = \sum_{t'=1}^{T} \mathbb{E}_{\pi_\theta}\left[ r_{t'} \sum_{t=1}^{t'} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right]
$$

두 합의 순서를 바꾸면($$t \le t'$$인 쌍을 $$t$$ 기준으로 묶으면) 다음을 얻는다.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=t}^{T} r(s_{t'}, a_{t'}) \right]
$$

**기준선(baseline).** 수익 추정에서 상수 $$b$$를 빼도($$R(\tau) \to R(\tau) - b$$) 기울기는 편향되지 않는다.

$$
\begin{aligned}
\mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(\tau)\, b \right] &= b \int \pi_\theta(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\, d\tau = b \int \pi_\theta(\tau) \frac{\nabla_\theta \pi_\theta(\tau)}{\pi_\theta(\tau)}\, d\tau \\
&= b \int \nabla_\theta \pi_\theta(\tau)\, d\tau = b\, \nabla_\theta \int \pi_\theta(\tau)\, d\tau = b\, \nabla_\theta 1 = 0
\end{aligned}
$$

$$\pi_\theta(\tau)$$는 확률분포이므로 적분이 1이고, 상수의 미분은 0이다.

**평균 수익이 좋은 기준선인 이유.** $$g = \nabla_\theta \log \pi_\theta(\tau)$$로 쓰자. 추정량의 분산은 다음과 같다(벡터의 경우 성분별로 생각한다).

$$
\operatorname{Var}\left( g\,(R(\tau) - b) \right) = \mathbb{E}\left[ g^2 (R(\tau) - b)^2 \right] - \left( \mathbb{E}\left[ g\,(R(\tau) - b) \right] \right)^2
$$

두 번째 항은 기댓값 기울기의 제곱이며, 기준선은 불편이므로 이 항을 바꾸지 못한다. 그래서 첫 번째 항을 최소화하는 $$b$$를 찾는다. 사실이 아니지만 단순화를 위해 $$g$$와 $$R(\tau)$$가 독립이라고 가정하면 다음과 같다.

$$
\arg\min_b \mathbb{E}\left[ g^2 \right] \mathbb{E}\left[ (R(\tau) - b)^2 \right] = \arg\min_b \mathbb{E}\left[ (R(\tau) - b)^2 \right] = \mathbb{E}[R(\tau)]
$$

제곱 오차를 최소화하는 상수는 평균이다. 이것이 평균 수익을 기준선으로 쓰는 동기다. 독립 가정이 틀렸으므로 이 기준선이 분산을 최소화한다는(또는 줄인다는) 보장은 없지만, 실전에서는 꽤 잘 작동한다.

## 문제 1. 상태 의존 기준선

### 문제

상태에 의존하는 기준선 $$b(s_t)$$를 빼도 정책 경사 추정량이 편향되지 않음을 보여라. 힌트: 시간 단계 $$t$$에 대한 합을 나누어 각 시간 단계를 따로 생각하라.

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( \sum_{t'=t}^{T} r(s_{t'}, a_{t'}) - b(s_t) \right) \right]
$$

### 풀이

기댓값의 선형성에 의해, 각 $$t$$에 대해 기준선 항의 기댓값이 0임을 보이면 충분하다.

$$
\mathbb{E}_{s_t, a_t \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, b(s_t) \right]
$$

**1단계: 반복 기댓값.** $$s_t$$에 대한 바깥 기댓값과, $$s_t$$가 주어졌을 때 $$a_t \sim \pi_\theta(\cdot \mid s_t)$$에 대한 안쪽 기댓값으로 나눈다.

$$
= \mathbb{E}_{s_t}\left[ \mathbb{E}_{a_t \sim \pi_\theta(a_t \mid s_t)}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, b(s_t) \mid s_t \right] \right]
$$

**2단계: $$b(s_t)$$를 꺼낸다.** 안쪽 기댓값에서 $$s_t$$는 고정되어 있으므로 $$b(s_t)$$는 상수다. 기준선이 행동에 의존하지 않는다는 점이 핵심이다.

$$
= \mathbb{E}_{s_t}\left[ b(s_t)\, \mathbb{E}_{a_t \sim \pi_\theta(a_t \mid s_t)}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] \right]
$$

**3단계: 점수 함수의 기댓값은 0이다.** 상수 기준선 증명과 같은 논리를 행동 분포에 적용한다.

$$
\mathbb{E}_{a_t \sim \pi_\theta(a_t \mid s_t)}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \int \nabla_\theta \pi_\theta(a_t \mid s_t)\, da_t = \nabla_\theta \int \pi_\theta(a_t \mid s_t)\, da_t = \nabla_\theta 1 = 0
$$

**4단계: 결론.**

$$
\mathbb{E}_{s_t}\left[ b(s_t) \cdot 0 \right] = 0
$$

모든 $$t$$에서 기준선 항의 기댓값이 0이므로, 상태 의존 기준선을 빼도 추정량은 불편이다. 평균 수익을 기준선으로 쓰는 것과 같은 정신으로, 상태 $$s_t$$에서의 기대 미래 수익(다음 절의 $$V(s_t)$$)을 기준선으로 쓸 수 있다.

## 2. 가치 함수

정책 $$\pi$$에 대해 시간 의존 상태 가치 함수와 상태-행동 가치 함수를 다음과 같이 정의한다.

$$
V^\pi_t(s) = \mathbb{E}_\pi\left[ \sum_{t'=t}^{T} r_{t'} \;\middle|\; s_t = s \right], \qquad
Q^\pi_t(s, a) = \mathbb{E}_\pi\left[ \sum_{t'=t}^{T} r_{t'} \;\middle|\; s_t = s, a_t = a \right]
$$

### 2.1 할인

무한 시평($$T = \infty$$)에서는 미래 보상이 절대 시각 $$t$$에 의존하지 않으므로 시간 무관 함수 $$V^\pi$$, $$Q^\pi$$를 쓴다. 다만 수익이 무한히 커질 수 있으므로 할인율 $$\gamma \in (0, 1)$$로 가까운 보상을 우선한다.

$$
V^\pi(s) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s \right], \qquad
Q^\pi(s, a) = \mathbb{E}_\pi\left[ \sum_{t=0}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s, a_0 = a \right]
$$

조건부 기댓값의 반복 법칙(tower property)에 의해 $$V^\pi(s) = \mathbb{E}_{a \sim \pi(a \mid s)}[Q^\pi(s, a)]$$이다. 실전에서는 유한 시평이어도 분산을 줄이고 추정을 쉽게 하려고 할인을 자주 쓴다.

### 2.2 벨만 방정식으로 가치 함수 학습하기

단순한 방법은 샘플 궤적의 수익에 몬테카를로 회귀를 하는 것이지만, 정책 경사처럼 수익의 분산이 크다. 그래서 가치 함수가 만족하는 일관성 조건, 즉 **벨만 방정식(Bellman equation)** 에 기반한 동적 계획법을 자주 쓴다. 첫 보상을 분리하면 다음과 같다.

$$
\begin{aligned}
V^\pi(s) &= \mathbb{E}_\pi[r_0] + \mathbb{E}_\pi\left[ \sum_{t=1}^{\infty} \gamma^t r_t \;\middle|\; s_0 = s \right] \\
&= \mathbb{E}_\pi[r_0] + \gamma\, \mathbb{E}_{s_1}\left[ V^\pi(s_1) \mid s_0 = s \right]
\end{aligned}
$$

현재 상태의 가치는 기대 즉시 보상에 다음 상태의 기대 가치를 할인해 더한 것이다. Q-함수도 같은 방식으로 얻는다.

$$
Q^\pi(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s_1}\left[ \mathbb{E}_{a_1 \sim \pi(\cdot \mid s_1)}[Q^\pi(s_1, a_1)] \right]
$$

**표 형식 갱신.** 모든 상태의 값을 저장할 수 있다면, 정책의 전이 $$(s, a, r, s')$$를 받아 $$V(s)$$를 목표값 $$r + \gamma V(s')$$ 쪽으로 스텝 크기 $$\alpha$$만큼 옮긴다.

$$
V(s) \leftarrow V(s) + \alpha \left( r(s, a) + \gamma V(s') - V(s) \right)
$$

(워크시트 해설에는 부호가 $$-\alpha$$로 적혀 있지만, 목표값 쪽으로 옮기려면 $$+\alpha$$여야 한다.) $$V$$를 학습하려면 평가 대상 정책의 온폴리시 샘플이 필요하다. 즉시 보상을 낳는 첫 행동에 대한 기댓값이 그 정책을 따라야 하기 때문이다. Q-값은 다음과 같이 학습한다.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r(s, a) + \gamma\, \mathbb{E}_{a' \sim \pi(a' \mid s')} Q(s', a') - Q(s, a) \right)
$$

Q는 미래 가치 $$Q(s', a')$$만 평가 대상 정책에 의존하므로 **오프폴리시 데이터**로 학습할 수 있다.

### 2.2.1 함수 근사로 학습하기

표 형식 갱신은 현재 값과 회귀 목표값 사이의 제곱 오차에 대한 경사 단계와 닮았다. 그래서 파라미터 $$\phi$$를 갖는 가치 함수(예: 심층 신경망)로 확장하면, 손실 $$L$$(예: 제곱 오차)에 대해 다음 목적의 경사 단계를 밟는다.

$$
\mathbb{E}_{s, a, s' \sim \pi}\left[ L\left( V^\pi_\phi(s),\ r(s, a) + \gamma V_{\bar\phi}(s') \right) \right]
$$

$$\bar\phi$$는 현재 $$\phi$$의 복사본으로, 목표값 $$V_{\bar\phi}(s')$$를 통해 기울기를 흘리지 않는다는 뜻이다. 목표값이 최적화 중에 계속 바뀌므로 이 갱신은 어떤 고정된 목적 함수에 대한 경사 하강도 아니다.

움직이는 목표값 때문에 갱신이 불안정할 수 있으므로 $$\bar\phi$$의 변화를 늦춘다. 방법은 두 가지다. 여러 번의 갱신 동안 $$\bar\phi$$를 고정했다가 현재 파라미터로 복사하거나, $$\bar\phi$$를 $$\phi$$의 지수 이동 평균으로 갱신한다.

Q-값은 오프폴리시 데이터로 직접 학습할 수 있으므로 다음 방향으로 갱신한다.

$$
\mathbb{E}_{s, a, s' \sim \mathcal{D}}\left[ \nabla_\phi L\left( Q^\pi_\phi(s, a),\ r(s, a) + \gamma\, \mathbb{E}_{a' \sim \pi(a' \mid s')}[Q_{\bar\phi}(s', a')] \right) \right]
$$

데이터셋 $$\mathcal{D}$$에는 이전 정책들의 궤적처럼 다른 정책이 만든 전이도 들어갈 수 있다. 이를 **경험 재생(experience replay)** 이라 한다. 과거 정책의 경험을 재사용해 현재 정책의 Q-함수를 학습하므로 샘플 효율이 크게 좋아질 수 있다.

### 2.3 액터-크리틱에서 가치 함수 사용하기

정확한 $$V$$나 $$Q$$를 안다면 다음 정책 경사 추정량들은 모두 불편이다.

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \mathbb{E}_{\pi_\theta}\left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)\, Q^\pi_t(s_t, a_t) \right] \\
&= \mathbb{E}_{\pi_\theta}\left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( r(s_t, a_t) + \gamma V^\pi_{t+1}(s_{t+1}) \right) \right] \\
&= \mathbb{E}_{\pi_\theta}\left[ \sum_{t=1}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( r(s_t, a_t) + \gamma V^\pi_{t+1}(s_{t+1}) - V^\pi_t(s_t) \right) \right]
\end{aligned}
$$

첫 줄은 보상-투-고를 그 조건부 기댓값 $$Q^\pi$$로 바꾼 것이고, 둘째 줄은 $$Q^\pi$$를 벨만 방정식으로 쓴 것이며, 셋째 줄은 문제 1의 상태 의존 기준선 $$V^\pi_t(s_t)$$를 뺀 것이다. 정확한 가치 함수라면 잡음 많은 수익 추정을 기댓값으로 바꾸었으므로 분산이 더 낮다.

학습된 $$V^\pi_\phi$$나 $$Q^\pi_\phi$$를 쓰면 **액터-크리틱(actor-critic)** 알고리즘이 된다. 정책(액터)을 학습된 가치 함수(크리틱)로 갱신한다. 진짜 조건부 기댓값 대신 학습된 값을 쓰므로 보통 기울기에 편향이 생기지만, 분산이 크게 줄어 정책 학습이 좋아지는 경우가 많다.

## 3. Q-Learning

Q-learning은 정책을 명시적으로 유지하지 않고 가치 함수만으로 강화학습을 푼다. 현재 정책의 가치 대신 최적 정책 $$\pi^\star$$의 가치 $$Q^\star$$를 직접 학습한다. $$Q^\star$$도 벨만 방정식을 만족한다.

$$
Q^\star(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s'}\left[ \mathbb{E}_{a' \sim \pi^\star(\cdot \mid s')}[Q^\star(s', a')] \right]
$$

최적 정책은 미래 가치를 최대화하는 행동을 해야 한다. 단순화를 위해 $$\pi^\star$$를 결정적이라 하면 $$\pi^\star(s) = \arg\max_a Q^\star(s, a)$$이다. 이를 대입하면 Q-learning의 기초인 **벨만 최적 방정식(Bellman optimality equation)** 을 얻는다.

$$
Q^\star(s, a) = r(s, a) + \gamma\, \mathbb{E}_{s'}\left[ \max_{a'} Q^\star(s', a') \right]
$$

Q-learning은 현재 Q-값을 최대화하는 행동을 항상 고르는 결정적 정책의 가치를 학습하는 것으로 볼 수 있다. Q-값에 대한 최적 행동을 계산해야 하므로 보통 이산 행동 공간에 한정된다. 연속 행동에서는 $$\max$$를 근사하거나 Q-함수에 추가 구조를 가정해 우회한다.

고전 Q-learning은 Q-함수에서 유도한 정책으로 새 전이를 샘플링하고, 그 전이로 Q-함수를 갱신하는 과정을 번갈아 한다. 심층 Q-learning에서는 경험 재생을 써서 과거 경험의 배치로 다음 방향을 따라 갱신한다.

$$
\mathbb{E}_{s, a, s' \sim \mathcal{D}}\left[ \nabla_\phi L\left( Q^\star_\phi(s, a),\ r(s, a) + \gamma \max_{a'} Q^\star_{\bar\phi}(s', a') \right) \right]
$$

경험 재생은 샘플 효율을 높이고 갱신을 안정시킨다.

### 3.1 Q-learning의 탐험

Q-learning은 암묵적으로 결정적 정책을 주므로, 그 정책을 그대로 탐험에 쓰면 탐험이 매우 제한된다. 최적 행동을 안다고 믿고 그 믿음대로만 행동하면 더 나은 수익으로 이어질 새로운 경험을 결코 하지 못한다.

**$$\epsilon$$-탐욕($$\epsilon$$-greedy)** 은 확률 $$\epsilon$$으로 완전히 무작위 행동을 하고, 나머지 경우에는 최적 행동을 한다. 무작위 행동이 결국 더 나은 행동을 발견하게 해 준다.

**볼츠만 탐험(Boltzmann exploration)** 은 다음 분포로 탐험한다. 각 행동이 얼마나 좋을지에 대한 현재 믿음에 따라 가중된 확률적 정책이다.

$$
\pi(a \mid s) \propto \exp\left( Q^\star_\phi(s, a) \right)
$$

정책 경사와 달리 Q-learning은 Q-값을 직접 학습하므로 오프폴리시 데이터를 쓸 수 있고, 따라서 다른 정책으로 탐험해도 아무 문제가 없다.

## 정리

- 점수 함수 추정량 $$\nabla_\theta \mathbb{E}_{p_\theta}[f] = \mathbb{E}_{p_\theta}[\nabla_\theta \log p_\theta\, f]$$를 궤적에 적용하면 정책 경사가 나온다. 동역학 항은 사라진다.
- 인과성을 이용한 보상-투-고와 기준선이 분산을 줄인다. 점수 함수의 기댓값이 0이므로 상수 기준선은 물론 행동에 의존하지 않는 상태 의존 기준선 $$b(s_t)$$도 편향을 만들지 않는다.
- 가치 함수는 벨만 방정식을 만족하며, 부트스트랩 목표값 $$r + \gamma V(s')$$로 학습한다. 목표값으로는 기울기를 흘리지 않으며, 움직이는 목표를 안정시키려 타깃 파라미터를 천천히 갱신한다.
- $$V$$ 학습에는 온폴리시 데이터가 필요하지만 $$Q$$는 오프폴리시 데이터(경험 재생)로 학습할 수 있다.
- 액터-크리틱은 학습된 가치 함수로 어드밴티지를 추정해 정책 경사의 분산을 줄이는 대신 약간의 편향을 받아들인다.
- Q-learning은 벨만 최적 방정식 $$Q^\star(s, a) = r + \gamma \mathbb{E}[\max_{a'} Q^\star(s', a')]$$를 학습하며, 탐험에는 $$\epsilon$$-탐욕이나 볼츠만 탐험을 쓴다.

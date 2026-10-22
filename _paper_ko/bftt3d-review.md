---
title: "BFTT3D: 학습이 필요 없는 두 번째 브랜치로 3D 분류기를 적응시키기"
category: paper-review
subfields: [tta, 3dpc]
lang: ko
description: 소스 모델은 고정한 채, 공유 부분 공간에서 테스트 특징을 소스 프로토타입과 매칭하는 비모수 브랜치를 추가하고, 두 로짓을 엔트로피로 섞는다.
conference: CVPR
year: 2024
paper:
  title: "Backpropagation-free Network for 3D Test-time Adaptation"
  authors: "Yanshuo Wang, Ali Cheraghian, Zeeshan Hayder, Jie Hong, Sameera Ramasinghe, Shafin Rahman, David Ahmedt-Aristizabal, Xuesong Li, Lars Petersson, Mehrtash Harandi"
  venue: "CVPR 2024"
  url: "https://arxiv.org/abs/2403.18442"
en: bftt3d-review
date: 2026-10-22 00:00:00 +0900
---

## 초록 (Abstract)

3D 포인트 클라우드의 테스트 시간 적응(test-time adaptation, TTA)은 보통 역전파로 모델을 업데이트한다. 의사 레이블(pseudo-label) 방법은 잡음 섞인 레이블을 모델에 다시 넣을 수 있어, 테스트 스트림을 거치며 오류가 쌓이고 소스 지식이 잊힌다. MATE 같은 자기 지도(self-supervised) 방법은 더 잘 작동하지만 느리며, 이는 적응이 실시간으로 이루어져야 할 때 문제가 된다.

BFTT3D는 소스 모델을 전혀 업데이트하지 않는다. 대신 파라미터가 없는 구성 요소들로 만든 두 번째 브랜치를 추가한다. 비모수(non-parametric) 포인트 클라우드 인코더, 소스 프로토타입을 담은 작은 메모리, 그리고 닫힌 형태(closed-form)의 부분 공간 정렬이 그것이다. 이 브랜치는 타깃에 특화된 로짓을 만든다. 이 로짓은 두 예측 엔트로피로부터 계산한 가중치를 사용해 고정된 소스 모델의 로짓과 섞인다.

## 방법 (Method)

![BFTT3D overview](/images/blog/papers/bftt3d-main.png)
*두 개의 스트림: 고정된 소스 모델이 $$\mathbf{l}_{s,t}$$를 주고, 비모수 네트워크와 정적 프로토타입 메모리에 대한 부분 공간 매칭이 $$\mathbf{l}_{bf,t}$$를 준다. 적응형 융합(adaptive fusion)이 이 둘을 $$\mathbf{l}_t$$로 결합한다. 그림은 논문에서 가져왔다.*

소스 모델 $$h_{\theta_s}$$는 일반적인 사전 학습 분류기(PointNet, DGCNN 또는 CurveNet)이다. 이 모델은 고정되어 소스 특화 로짓 $$\mathbf{l}_{s,t}$$를 준다. 그림의 나머지 모든 부분은 학습 가능한 가중치 없이 두 번째 로짓 $$\mathbf{l}_{bf,t}$$를 만들기 위해 존재한다.

### 비모수 네트워크

아래 브랜치는 Point-NN의 비모수 인코더를 사용한다. 학습된 가중치가 없다. 최원점 샘플링(farthest point sampling), $$k$$-NN 그룹화, 풀링, 삼각함수 위치 임베딩만으로 구성된다. 각 좌표는 사인과 코사인으로 인코딩되고, 이웃 특징은 각 중심점에 이어 붙인 뒤 상대 위치 인코딩으로 재가중되며, 최대 풀링과 평균 풀링이 각 그룹을 압축한다. 이를 네 번 반복한 뒤 전역 풀링으로 테스트 샘플마다 특징 $$\mathbf{f}_{t,i} \in \mathbb{R}^d$$를 얻는다. 아무것도 학습하지 않으므로, 저자들은 이 인코더가 소스 도메인에 덜 편향되어 있다고 주장한다.

### 정적 프로토타입 메모리

테스트 전에 같은 인코더를 소스 데이터에 실행한다. 각 클래스 $$c$$에 대해 herding 알고리즘이 클래스 평균에 가장 가까운 특징들을 남겨 고정된 메모리 $$\mathcal{M}_c$$를 만든다. 논문은 소스 특징의 25%를 유지한다. 이 메모리는 한 번만 만들어지고 이후 바뀌지 않는다.

### 부분 공간 학습

타깃 특징을 소스 프로토타입과 직접 매칭하면 도메인 차이의 영향을 받는다. 그래서 둘 다 먼저 공유 부분 공간 $$\mathbf{f}^* = \psi(\mathbf{f})$$로 사영하며, 이 사영은 프로토타입과 테스트 특징 사이의 Maximum Mean Discrepancy(MMD)를 최소화하도록 정한다.

$$
\operatorname{Dist}(\mathcal{Q}_s, \mathcal{Q}_t) = \left\| \frac{1}{n_s}\sum_{i=1}^{n_s} \mathbf{f}_i^* - \frac{1}{n_t}\sum_{i=1}^{n_t} \mathbf{f}_{t,i}^* \right\|_{\mathcal{H}}^2
$$

사영은 Transfer Component Analysis(TCA)로 구한다. 전체 $$n_s + n_t$$개의 프로토타입 및 테스트 특징에 대한 커널 행렬 $$\mathbf{K}$$, 스케일링 행렬 $$\mathbf{L}$$, 중심화 행렬 $$\mathbf{H}$$를 두고 TCA는 다음을 푼다.

$$
\min_{\mathbf{W}} \operatorname{tr}(\mathbf{W}^T \mathbf{K}\mathbf{L}\mathbf{K}\mathbf{W}) + \mu \operatorname{tr}(\mathbf{W}^T\mathbf{W}) \quad \text{s.t. } \mathbf{W}^T\mathbf{K}\mathbf{H}\mathbf{K}\mathbf{W} = \mathbf{I}
$$

해는 닫힌 형태로, $$(\mathbf{KLK} + \mu\mathbf{I})^{-1}\mathbf{KHK}$$의 상위 $$m$$개 고유벡터이다. 사영된 특징은 $$\mathbf{W}^T\mathbf{K}$$의 열들이며, 논문에서는 $$m = 150$$이다. 그래디언트 스텝은 전혀 쓰이지 않는다.

부분 공간에서 테스트 특징은 코사인 유사도 $$\mathcal{J} = \tilde{\mathbf{F}}_t \cdot \tilde{\mathcal{M}}^T$$로 프로토타입과 비교되고, 프로토타입 레이블 $$\mathbf{L}_m$$을 이용해 클래스 점수로 바뀐다.

$$
\mathbf{l}_{bf,t} = \varphi(\mathcal{J}\mathbf{L}_m), \qquad \varphi(x) = \exp(-\gamma(1 - x))
$$

### 적응형 융합

두 로짓은 가중치 $$p$$로 섞인다.

$$
\mathbf{l}_t = p \cdot \mathbf{l}_{bf,t} + (1-p) \cdot \mathbf{l}_{s,t}, \qquad
p = 1 - \frac{\text{E}(\mathbf{l}_{bf,t})}{\text{E}(\mathbf{l}_{s,t}) + \text{E}(\mathbf{l}_{bf,t})}
$$

여기서 $$\text{E}(\cdot)$$는 softmax 엔트로피이다. 소스 모델이 불확실하면(엔트로피가 높으면) $$p$$가 커지고 프로토타입 브랜치가 더 큰 가중치를 받는다. 타깃이 소스에 가까우면 소스 모델이 주도한다. ablation에서 이 적응형 $$p$$는 검증 데이터 없이도 전수 탐색으로 찾은 최적의 고정 $$p$$에 가깝게 유지된다.

## 주요 실험 (Main Experiments)

심각도 5의 15가지 손상에 대한 평균 분류 오류율(%, 낮을수록 좋음)이다. 각 백본은 깨끗한 데이터셋에서 학습되어 소스 모델로 쓰인다. LAME과 DSS는 PointNet에서만 보고되었다. Table 1과 2의 일부이며, 논문에는 손상 유형별 오류율도 실려 있다.

**ModelNet-40C** (Table 1)

| 방법 | PointNet | DGCNN | CurveNet |
|---|---|---|---|
| Source-only | 35.88 | 34.65 | 28.47 |
| TENT | 34.62 | 30.99 | 22.60 |
| BN | 34.89 | 31.36 | 24.03 |
| SHOT | 34.25 | 30.17 | 19.79 |
| LAME | 83.16 | – | – |
| DSS | 35.23 | – | – |
| **BFTT3D** | **33.78** | **29.33** | **19.23** |

**ScanObjectNN-C** (Table 2)

| 방법 | PointNet | DGCNN | CurveNet |
|---|---|---|---|
| Source-only | 59.35 | 46.09 | 64.77 |
| TENT | 58.18 | 45.17 | 65.77 |
| BN | 57.87 | 45.11 | 64.26 |
| SHOT | 65.63 | 57.53 | 85.35 |
| LAME | 55.67 | – | – |
| DSS | 56.53 | – | – |
| **BFTT3D** | **54.46** | **43.91** | **57.80** |

BFTT3D는 여섯 가지 설정 모두에서 평균 오류율이 가장 낮다. ModelNet-40C에서는 최고 베이스라인과의 차이가 작다(예: CurveNet에서 SHOT의 19.79 대 19.23). 더 작고 실제 스캔으로 만든 ScanObjectNN-C에서는 차이가 더 큰데, 역전파 기반 베이스라인들이 그곳에서 개선에 실패하거나 오히려 나빠지는 경우가 많기 때문이다(CurveNet에서 SHOT은 85.35까지 오른다). 향상은 배경(background), 가림(occlusion), LiDAR 같은 큰 분포 변화에 집중된다. 논문은 실행 시간이나 메모리 수치를 보고하지 않는다.

## 한계 (Limitations)

- **테스트 시점의 소스 데이터.** 프로토타입 메모리는 소스 특징(기본적으로 그중 25%)으로 만든다. 이는 엄격한 소스 프리(source-free) 설정에 맞지 않는다.
- **배치마다 TCA.** 부분 공간은 모든 프로토타입과 현재 테스트 특징에 대한 커널 행렬로부터 풀며, 역행렬과 고유값 분해가 뒤따른다. 비용은 메모리 크기에 따라 늘어난다. 논문은 시간을 제시하지 않는다. 이후의 논문(Purge-Gate, ICCV 2025)은 Point-MAE 백본에서 BFTT3D가 배치당 173.94 ms와 7400 MiB의 GPU 메모리를 쓴다고 측정했다.
- **테스트 샘플 배치가 필요하다.** MMD/TCA 단계는 $$n_t$$개 테스트 특징의 집합을 프로토타입과 정렬한다. 사용한 테스트 배치 크기는 명시되지 않았고, 단일 샘플에서의 동작은 연구되지 않았다.
- **ModelNet-40C에서의 작은 향상.** PointNet에서 평균 오류율은 35.88에서 33.78로 내려간다. 개별 손상에서는 소스 모델보다 나빠질 수도 있다(DGCNN의 ModelNet-40C shear: 15.96에서 18.80).
- **추론 시 두 개의 인코더.** 모든 테스트 샘플이 소스 모델과 비모수 네트워크를 모두 거친다.

## 내 생각 (My Thoughts)

<!-- TODO -->

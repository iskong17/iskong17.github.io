---
title: "Deep Learning & PyTorch 13: Generative Models"
date: 2026-10-22 17:13:00 +0900
category: study
topic: pytorch
series: dlbook
lecture: 13
card_title: "Chapter 13: Generative Models"
card_desc: "Discriminative vs generative models · Denoising autoencoder · Variational autoencoder and the reparameterization trick · GAN on MNIST · DCGAN, cGAN, pix2pix and CycleGAN"
lang: ko
notes: "https://thebook.io/080289/0888/"
notes_label: "thebook.io chapter 13"
description: "데이터 분포를 배워 새 데이터를 만드는 생성 모델을 다룬다. 합성곱 오토인코더와 변형 오토인코더를 MNIST로 구현하고, 생성자와 판별자가 경쟁하는 GAN을 학습시킨 뒤, DCGAN, cGAN, PIX2PIX, CycleGAN의 아이디어를 정리한다."
---

<ul class="post-sources">
<li markdown="span">[딥러닝 파이토치 교과서 (서지영, 길벗, 2022)](https://thebook.io/080289/) 13장 정리</li>
<li markdown="span">[예제 코드 (gilbutITbook/080289)](https://github.com/gilbutITbook/080289/tree/main/chap13)</li>
</ul>

책의 마지막 장인 13장은 생성 모델을 다룬다. 확률 분포를 명시적으로 다루는 오토인코더 계열(변형 오토인코더)과, 분포를 직접 정의하지 않고 경쟁을 통해 샘플을 만드는 GAN 계열을 차례로 구현하고, GAN에서 파생된 대표 모델들의 구조를 살펴본다.

## 13.1 생성 모델이란

[📖 thebook.io](https://thebook.io/080289/0889/)

**생성 모델(generative model)** 은 주어진 데이터로부터 데이터의 분포를 배우고, 그 분포를 따르는 새로운 데이터를 만들어 내는 모델이다.

### 13.1.1 생성 모델 개념

5~6장의 이미지 분류는 입력 $$x$$가 주어졌을 때 레이블 $$y$$를 맞히는 문제였다. 개와 고양이 사진을 구별하려면 귀 모양, 꼬리처럼 두 클래스를 가르는 특성을 잘 찾아야 한다. 이런 모델을 **판별 모델(discriminative model)** 이라 하며, 확률로 쓰면 조건부 확률 $$p(y \mid x)$$를 배운다.

생성 모델은 한 걸음 더 나아가 데이터 자체의 분포 $$p(x)$$를 배운다. 개 사진들이 어떤 분포에서 나오는지 알면, 그 분포에서 새 점을 뽑아 학습 데이터에 없던, 그러나 개처럼 보이는 새 이미지를 만들 수 있다.

### 13.1.2 생성 모델의 유형

생성 모델은 $$p(x)$$를 다루는 방식에 따라 크게 둘로 나뉜다.

| 유형 | 방식 | 대표 모델 |
| --- | --- | --- |
| 명시적 밀도(explicit density) | 확률 분포 $$p(x)$$를 수식으로 정의하고(또는 근사하고) 그 가능도를 최대화한다 | 변형 오토인코더(VAE) |
| 암시적 밀도(implicit density) | $$p(x)$$를 정의하지 않고, 그 분포에서 나온 것 같은 샘플을 만드는 법만 배운다 | GAN |

변형 오토인코더는 이미지를 저차원의 **잠재 공간(latent space)** 으로 압축하고, 그 공간에서 점을 뽑아 디코더로 이미지를 만든다. 잠재 공간은 예를 들어 손글씨 숫자 이미지 하나를 2차원 좌표 하나로 나타내는 공간이고, 그 안의 한 점을 **잠재 벡터(latent vector)** 라 한다. GAN은 생성자와 판별자 두 신경망이 서로 경쟁하며, 생성자가 진짜와 구별되지 않는 가짜를 만들도록 학습한다.

## 13.2 변형 오토인코더

[📖 thebook.io](https://thebook.io/080289/0892/)

변형 오토인코더는 오토인코더를 확장한 모델이므로, 오토인코더부터 본다.

### 13.2.1 오토인코더란

**오토인코더(autoencoder)** 는 입력을 그대로 출력으로 복원하도록 학습하는 신경망이다. 입력을 그대로 베끼는 것이 무슨 의미가 있나 싶지만, 핵심은 가운데에 뉴런 수가 입력보다 훨씬 적은 **병목층(bottleneck)** 을 둔다는 데 있다. 적은 수의 값으로 입력을 다시 만들어 내야 하므로, 신경망은 데이터의 가장 중요한 특성만 골라 압축하는 법을 배운다.

구성은 다음 네 부분이다.

1. **인코더(encoder)**: 입력을 압축하며 특성을 학습한다. 인지 네트워크(recognition network)라고도 한다.
2. **병목층(bottleneck)**: 뉴런 수가 가장 적은 층. 입력의 압축된 표현(잠재 벡터)이 여기 담긴다.
3. **디코더(decoder)**: 압축된 표현에서 원래 입력을 재구성(reconstruction)한다. 생성 네트워크(generative network)라고도 한다.
4. **재구성 손실(reconstruction loss)**: 입력과 출력의 차이. 입력층과 출력층의 크기가 같다는 점만 빼면 일반적인 다층 퍼셉트론(MLP)과 같은 구조이며, 정답 레이블 대신 입력 자신을 정답으로 쓴다.

수식으로 쓰면, 입력 $$x \in \mathbb{R}^d$$를 인코더 $$h$$로 압축해 $$z$$를 얻고, 디코더 $$g$$로 같은 크기의 출력 $$y$$를 만든다. 손실은 입력과 출력의 차이다.

$$
z = h(x), \qquad y = g(z) = g(h(x)), \qquad L(x, y) = \lVert x - g(h(x)) \rVert^2
$$

오토인코더가 쓸모 있는 이유는 세 가지다.

- **데이터 압축**: 중요한 특성만 남겨 이미지나 음성을 작게 표현할 수 있다.
- **차원의 저주 완화**: 특성 수를 줄여 이후 모델이 다룰 차원을 낮춘다.
- **특성 추출**: 레이블 없이(비지도) 데이터를 잘 설명하는 특성을 자동으로 찾는다.

**합성곱 오토인코더 구현**

MNIST 손글씨 이미지로 합성곱 기반 오토인코더를 만든다. 이 예제는 입력에 잡음을 섞어 넣는 **잡음 제거 오토인코더(denoising autoencoder)** 형태로 구성된다.

```python
transform = transforms.Compose([transforms.ToTensor()])   # 0~1 범위 텐서로
train_dataset = datasets.MNIST(root="../chap13/data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="../chap13/data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
```

`DataLoader`의 `num_workers`는 데이터를 병렬로 불러올 프로세스 수다. 너무 크면 오버헤드가, 너무 작으면 GPU가 데이터를 기다리는 병목이 생긴다. `pin_memory=True`는 데이터를 페이지 고정(pinned) 메모리에 올려 CPU에서 GPU로의 복사를 빠르게 하는 옵션으로, GPU로 학습할 때 의미가 있다.

인코더는 합성곱 세 번으로 28x28 이미지를 32x3x3 특성 맵까지 줄이고, 선형 층 두 개로 길이 4의 잠재 벡터를 만든다. 디코더는 이를 정확히 거꾸로 거슬러 올라간다. 합성곱으로 줄인 크기를 되돌리는 데는 **전치 합성곱(transposed convolution, `ConvTranspose2d`)** 을 쓴다.

```python
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),    # 28 -> 14
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),   # 14 -> 7
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),  # 7 -> 3
            nn.ReLU(True)
        )
        self.flatten = nn.Flatten(start_dim=1)          # (N, 32, 3, 3) -> (N, 288)
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)           # 잠재 벡터
        )

    def forward(self, x):
        return self.encoder_lin(self.flatten(self.encoder_cnn(x)))

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))   # (N, 288) -> (N, 32, 3, 3)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),             # 3 -> 7
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),   # 7 -> 14
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)     # 14 -> 28
        )

    def forward(self, x):
        x = self.decoder_conv(self.unflatten(self.decoder_lin(x)))
        return torch.sigmoid(x)   # 픽셀 값을 0~1로
```

인코더와 디코더의 파라미터를 함께 최적화하도록 옵티마이저에 둘 다 넘긴다. 손실은 평균 제곱 오차를 쓴다. 오토인코더에서는 보통 평균 제곱 오차나 이진 교차 엔트로피를 쓰며, 픽셀 값이 0~1 범위라면 이진 교차 엔트로피도 잘 맞는다.

```python
encoder = Encoder(encoded_space_dim=4, fc2_input_dim=128).to(device)
decoder = Decoder(encoded_space_dim=4, fc2_input_dim=128).to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]   # 파라미터 그룹: 그룹별로 학습률 등을 다르게 줄 수도 있다
optim = torch.optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)
loss_fn = torch.nn.MSELoss()
```

잡음은 입력과 같은 모양의 표준정규 난수를 곱해 더한 뒤, 픽셀 범위를 벗어나지 않게 0~1로 자른다.

```python
def add_noise(inputs, noise_factor=0.3):
    noisy = inputs + torch.randn_like(inputs) * noise_factor   # 같은 모양의 가우시안 잡음
    return torch.clip(noisy, 0., 1.)                            # 값 범위를 [0, 1]로 제한
```

학습 함수는 잡음 섞인 이미지를 인코더와 디코더에 통과시키고 손실을 계산한다. 레이블은 쓰지 않는다.

```python
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer, noise_factor=0.3):
    encoder.train()
    decoder.train()
    train_loss = []
    for image_batch, _ in dataloader:              # 레이블은 버린다 (비지도 학습)
        image_noisy = add_noise(image_batch, noise_factor).to(device)
        decoded_data = decoder(encoder(image_noisy))
        loss = loss_fn(decoded_data, image_noisy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)
```

여기서 손실을 잡음 섞인 입력(`image_noisy`)과 비교한다는 점에 주의한다. 이렇게 하면 모델은 "잡음까지 포함해 그대로 복원"하도록 학습되므로 엄밀한 의미의 잡음 제거 오토인코더가 아니다. 잡음을 지우는 법을 배우게 하려면 `loss_fn(decoded_data, image_batch.to(device))`처럼 깨끗한 원본과 비교해야 한다. 그래도 병목이 길이 4로 매우 좁아서 잡음 같은 세부는 담지 못하므로, 결과적으로 출력은 상당히 깨끗해진다.

검증 함수는 깨끗한 테스트 이미지로 재구성 손실을 계산한다. 시각화 함수는 원본, 잡음을 넣은 이미지, 재구성 이미지를 세 줄로 그린다. 30 에폭 동안 학습한 결과는 다음과 같다.

```python
num_epochs = 30
for epoch in range(num_epochs):
    train_loss = train_epoch(encoder, decoder, device, train_loader, loss_fn, optim, noise_factor=0.3)
    val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn, noise_factor=0.3)
    print('EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(
        epoch + 1, num_epochs, train_loss, val_loss))
    plot_ae_outputs(encoder, decoder, noise_factor=0.3)
```

```text
EPOCH 1/30   train loss 0.089   val loss 0.050
...
EPOCH 12/30  train loss 0.050   val loss 0.039
...
EPOCH 30/30  train loss 0.049   val loss 0.037
```

에폭이 진행될수록 손실이 줄고, 잡음 낀 입력에서 재구성한 숫자가 점점 선명해진다. 검증 손실이 학습 손실보다 낮은 것은 검증 때는 잡음 없는 깨끗한 이미지를 쓰기 때문이다.

### 13.2.2 변형 오토인코더

일반 오토인코더는 입력을 잘 압축했다가 복원하는 것이 목적이라, 잠재 공간이 어떤 모양인지에는 관심이 없다. 그래서 학습 데이터가 놓이지 않은 잠재 공간의 빈 곳에서 점을 뽑아 디코더에 넣으면 엉뚱한 이미지가 나온다. 생성에 쓰기 어렵다.

**변형 오토인코더(variational autoencoder, VAE)** 는 인코더가 잠재 벡터 하나가 아니라 **분포** 를, 구체적으로 가우시안 분포의 평균과 분산을 출력하게 한다. 그 분포에서 잠재 벡터 $$z$$를 뽑아 디코더로 이미지를 만들고, 동시에 모든 입력의 잠재 분포가 표준정규분포 근처에 모이도록 제약을 건다. 그러면 잠재 공간이 빈틈없이 매끄럽게 채워져, 표준정규분포에서 아무 점이나 뽑아도 그럴듯한 이미지가 나온다. 즉 오토인코더가 차원 축소 도구라면, VAE는 데이터가 생성되는 확률 분포를 배우는 생성 모델이다.

VAE의 두 네트워크는 다음과 같다.

- **인코더** $$q_\phi(z \mid x)$$: 입력 $$x$$를 받아 잠재 변수 $$z$$의 분포(평균 $$\mu_{z \mid x}$$, 분산 $$\Sigma_{z \mid x}$$)를 출력한다. 진짜 사후 분포 $$p(z \mid x)$$는 계산할 수 없으므로, 다루기 쉬운 가우시안으로 근사한다. 이것이 **변분 추론(variational inference)** 이다.
- **디코더** $$p_\theta(x \mid z)$$: $$z$$를 받아 $$x$$의 분포를 출력한다. 실제 구현에서는 픽셀별 베르누이 확률(0~1)을 출력한다.

학습 목표는 데이터의 로그 가능도의 하한인 **ELBO(Evidence Lower BOund)** $$\mathcal{L}$$을 최대화하는 것이다.

$$
\mathcal{L}(x^{(i)}, \theta, \phi) = \underbrace{\mathbb{E}_{z \sim q_\phi(z \mid x^{(i)})}\left[\log p_\theta(x^{(i)} \mid z)\right]}_{\text{(1) reconstruction}} - \underbrace{D_{KL}\left(q_\phi(z \mid x^{(i)}) \,\Vert\, p(z)\right)}_{\text{(2) regularization}}
$$

- (1)항은 인코더가 뽑은 $$z$$로 디코더가 원래 $$x$$를 얼마나 잘 복원하는지, 즉 디코더의 가능도다. 클수록 좋다.
- (2)항은 인코더가 만든 분포가 사전 분포 $$p(z) = \mathcal{N}(0, I)$$와 얼마나 다른지를 재는 **쿨백-라이블러 발산(Kullback-Leibler divergence, KLD)** 이다. 두 확률 분포의 차이를 재는 척도로, 작을수록 좋다.

손실 함수로는 $$\mathcal{L}$$에 마이너스를 붙여 최소화한다. 즉 "재구성 오차 + KLD"를 줄인다. 두 항은 서로 당긴다. 재구성만 신경 쓰면 입력마다 동떨어진 위치에 뾰족한 분포를 만들고(KLD 증가), KLD만 신경 쓰면 모든 입력이 같은 분포가 되어 복원이 안 된다. 그 균형점에서 잠재 공간이 정돈된다.

**재매개변수화 기법**

$$z$$를 분포에서 무작위로 뽑는 연산은 미분할 수 없어서, 그대로는 인코더까지 기울기가 전달되지 않는다. **재매개변수화 기법(reparameterization trick)** 은 무작위성을 표준정규 잡음 $$\epsilon$$으로 따로 떼어 내, $$z$$를 평균과 표준편차의 결정적인 함수로 만든다.

$$
z = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)
$$

이제 $$z$$는 $$\mu$$, $$\sigma$$에 대해 미분 가능하고, 무작위성은 학습할 필요가 없는 $$\epsilon$$에만 있다.

**구현**

학습 과정을 TensorBoard로 보기 위해 `tensorboardX`를 쓴다(`pip install tensorboardX`). 이번에는 합성곱 없이 이미지를 784차원 벡터로 펴서 선형 층만으로 만든다.

```python
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.input1 = nn.Linear(input_dim, hidden_dim)
        self.input2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, latent_dim)   # 평균 출력
        self.var = nn.Linear(hidden_dim, latent_dim)    # 로그 분산 출력
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.LeakyReLU(self.input1(x))
        h_ = self.LeakyReLU(self.input2(h_))
        return self.mean(h_), self.var(h_)   # (mean, log_var)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden1 = nn.Linear(latent_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.hidden1(x))
        h = self.LeakyReLU(self.hidden2(h))
        return torch.sigmoid(self.output(h))   # 픽셀별 확률 (0~1)
```

인코더의 두 번째 출력은 분산이 아니라 **로그 분산** $$\log \sigma^2$$이다. 분산은 항상 양수여야 하는데, 신경망 출력은 음수가 될 수 있다. 로그 분산을 출력하게 하고 필요할 때 지수를 취하면 이 제약을 자연스럽게 만족한다. 표준편차는 $$\sigma = \exp(0.5 \log \sigma^2)$$이다.

```python
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super().__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):         # var 자리에 표준편차가 들어온다
        epsilon = torch.randn_like(var).to(device)   # ε ~ N(0, I)
        return mean + var * epsilon                  # z = μ + σ ε

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))   # σ = exp(0.5 log σ²)
        x_hat = self.Decoder(z)
        return x_hat, mean, log_var

x_dim, hidden_dim, latent_dim = 784, 400, 200
epochs, batch_size = 30, 100
encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = Model(Encoder=encoder, Decoder=decoder).to(device)
```

(`reparameterization`의 인자 이름이 `var`이지만 실제로 넘어오는 값은 표준편차다.)

손실 함수는 재구성 오차(이진 교차 엔트로피, 배치 합계)와 KLD 두 항을 돌려준다. 인코더 분포가 대각 공분산 가우시안이고 사전 분포가 표준정규분포이면 KLD는 닫힌 식으로 계산된다.

$$
D_{KL} = -\frac{1}{2} \sum_{j=1}^{J} \left( 1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
$$

```python
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')   # 재구성
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())                   # 정규화
    return reproduction_loss, KLD

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

학습 루프는 이미지를 `(100, 784)`로 펴서 넣고, 두 손실의 합으로 역전파한다. 각 손실을 `SummaryWriter.add_scalar(태그, 값, x축 스텝)`으로 기록해 두면 TensorBoard에서 곡선으로 볼 수 있다. `SummaryWriter`의 인자는 로그를 저장할 폴더(기본값 `./runs`)다.

```python
writer = SummaryWriter('scalar/')

def train(epoch, model, train_loader, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(batch_size, x_dim).to(device)
        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        BCE, KLD = loss_function(x, x_hat, mean, log_var)
        loss = BCE + KLD
        step = batch_idx + epoch * (len(train_loader.dataset) / batch_size)
        writer.add_scalar("Train/Reconstruction Error", BCE.item(), step)
        writer.add_scalar("Train/KL-Divergence", KLD.item(), step)
        writer.add_scalar("Train/Total Loss", loss.item(), step)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("======> Epoch: {} Average loss: {:.4f}".format(epoch, train_loss / len(train_loader.dataset)))
```

테스트 함수는 같은 방식으로 손실을 기록하고, 첫 배치에서 원본 8장과 재구성 8장을 `make_grid`로 붙여 `add_image`로 기록한다. 30 에폭 학습 후 학습을 마칠 때는 `writer.close()`로 로그를 디스크에 확실히 기록한다.

```text
Train Epoch: 0 [0/60000 (0%)]        Loss: 544.363125
Train Epoch: 0 [10000/60000 (17%)]   Loss: 191.183652
======> Epoch: 0 Average loss: 173.2792
======> Epoch: 1 Average loss: 128.7373
...
======> Epoch: 28 Average loss: 100.1766
======> Epoch: 29 Average loss: 100.0594
```

이미지 한 장당 손실이 545에서 100 정도로 줄었다. 노트북에서는 `%load_ext tensorboard`, `%tensorboard --logdir scalar --port=6013` 매직 커맨드로 TensorBoard를 띄운다(`%`로 시작하는 명령은 주피터 전용이며, 기본 포트는 6006이다).

TensorBoard 그래프에서는 재구성 오차가 꾸준히 내려가는 반면 KLD는 학습 초반에 올라갔다가 일정한 범위에서 안정된다. 처음에는 모든 분포가 표준정규분포 근처라 KLD가 작지만 복원을 못 하고, 학습이 진행되면 복원을 위해 분포를 조금씩 벌리면서 KLD가 늘어난다. 두 항이 서로 맞서는 가운데 전체 손실은 줄어든다. 테스트 데이터에서도 같은 경향이 나타나, 새 이미지를 잘 재구성하고 있음을 알 수 있다.

## 13.3 적대적 생성 신경망(GAN)이란

[📖 thebook.io](https://thebook.io/080289/0925/)

**적대적 생성 신경망(Generative Adversarial Network, GAN)** 은 2014년 이안 굿펠로(Ian Goodfellow)가 제안했다. 그는 GAN을 위조지폐범과 경찰의 게임에 비유했다. 위조지폐범은 경찰을 속일 만큼 진짜 같은 위조지폐를 만들려 하고, 경찰은 진짜와 가짜를 가려내려 한다. 이 경쟁이 계속되면 위조 기술이 계속 좋아져서, 결국 경찰이 진짜와 가짜를 구별하지 못하는 수준에 이른다.

딥러닝으로 옮기면 위조지폐범이 **생성자(generator)**, 경찰이 **판별자(discriminator)** 다. 학습은 두 단계를 번갈아 반복한다.

1. **판별자 학습**: 진짜 이미지를 넣었을 때는 "진짜"로, 생성자가 만든 가짜 이미지를 넣었을 때는 "가짜"로 분류하도록 판별자를 학습한다.
2. **생성자 학습**: 생성자가 만든 이미지를 판별자가 "진짜"로 분류하도록 생성자를 학습한다.

판별자는 맞히는 확률을 높이려 하고 생성자는 그 확률을 낮추려 하면서, 둘 다 점점 강해진다.

### 13.3.1 GAN 동작 원리

판별자 $$D$$는 이미지 $$x$$를 받아 그것이 진짜일 확률 $$D(x)$$(0~1)를 출력한다. 생성자 $$G$$는 무작위 잡음 벡터 $$z$$를 받아 가짜 이미지 $$G(z)$$를 만든다. 판별자를 학습할 때는 생성자를 고정한 채 진짜 이미지의 확률은 높이고 가짜 이미지의 확률은 낮추는 방향으로, 생성자를 학습할 때는 판별자를 고정한 채 가짜 이미지의 확률을 높이는 방향으로 가중치를 갱신한다.

이 경쟁을 하나의 식으로 쓴 것이 GAN의 목적 함수다. 판별자는 이 값을 최대화하고, 생성자는 최소화한다.

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}(x)}\left[\log D(x)\right] + \mathbb{E}_{z \sim p_z(z)}\left[\log\left(1 - D(G(z))\right)\right]
$$

- $$x \sim p_{data}(x)$$: 실제 데이터 분포에서 뽑은 진짜 샘플.
- $$z \sim p_z(z)$$: 잡음 분포(보통 표준정규분포)에서 뽑은 잠재 벡터.

**판별자 입장**: 두 항을 모두 키우고 싶다. 진짜에 대해 $$D(x) = 1$$이면 $$\log D(x) = 0$$으로 첫 항이 최대가 되고, 가짜에 대해 $$D(G(z)) = 0$$이면 $$\log(1 - D(G(z))) = 0$$으로 둘째 항이 최대가 된다.

$$
\max_D \; \mathbb{E}_{x}\left[\log D(x)\right] + \mathbb{E}_{z}\left[\log\left(1 - D(G(z))\right)\right]
$$

**생성자 입장**: 첫 항은 생성자와 무관하므로 둘째 항만 줄이면 된다. $$D(G(z)) = 1$$, 즉 판별자가 가짜를 진짜로 믿을 때 $$\log(1 - D(G(z)))$$가 가장 작아진다.

$$
\min_G \; \mathbb{E}_{z}\left[\log\left(1 - D(G(z))\right)\right]
$$

실제 구현에서는 학습 초반에 이 식의 기울기가 너무 작아지는 문제 때문에, 생성자 손실을 "가짜 이미지에 진짜 레이블(1)을 붙인 이진 교차 엔트로피", 즉 $$-\log D(G(z))$$를 최소화하는 형태로 바꿔 쓴다. 목표하는 방향은 같다.

### 13.3.2 GAN 구현

MNIST로 선형 층만 쓴 간단한 GAN을 만든다.

```python
batch_size = 512
epochs = 200
sample_size = 64   # 고정 잡음 벡터 개수 (시각화용)
nz = 128           # 잠재 벡터 크기 = 생성자 입력 크기
k = 1              # 생성자 1번 학습당 판별자 학습 횟수

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),   # (x - 0.5) / 0.5 -> 픽셀 범위 [-1, 1]
])
train_dataset = datasets.MNIST(root="../chap13/data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
```

`Normalize((0.5,), (0.5,))`는 평균 0.5를 빼고 표준편차 0.5로 나누는 변환이므로, 0~1 픽셀이 -1~1 범위로 옮겨진다. 생성자 출력에 `tanh`(-1~1)를 쓰기 때문에 진짜 이미지도 같은 범위로 맞춘 것이다.

생성자는 길이 128의 잡음을 선형 층 네 개로 키워 784(=28x28) 픽셀을 만든다. 판별자는 반대로 784 픽셀을 받아 진짜일 확률 하나를 낸다.

```python
class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 784), nn.Tanh(),     # -1~1 픽셀
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)  # (배치, 1, 28, 28) 이미지 모양으로

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid(),     # 진짜일 확률
        )
    def forward(self, x):
        return self.main(x.view(-1, 784))

generator = Generator(nz).to(device)
discriminator = Discriminator().to(device)
```

판별자는 이진 분류기이므로 마지막에 시그모이드를 두고 손실로 `nn.BCELoss`를 쓴다. 생성자와 판별자는 서로 다른 목표로 번갈아 학습하므로 **옵티마이저를 따로** 만든다.

```python
optim_g = optim.Adam(generator.parameters(), lr=0.0002)
optim_d = optim.Adam(discriminator.parameters(), lr=0.0002)
criterion = nn.BCELoss()
```

판별자 학습 함수는 진짜 이미지에는 레이블 1, 가짜 이미지에는 레이블 0을 붙여 두 손실을 더한다.

```python
def train_discriminator(optimizer, data_real, data_fake):
    b_size = data_real.size(0)
    real_label = torch.ones(b_size, 1).to(device)    # 진짜 = 1
    fake_label = torch.zeros(b_size, 1).to(device)   # 가짜 = 0
    optimizer.zero_grad()
    loss_real = criterion(discriminator(data_real), real_label)
    loss_fake = criterion(discriminator(data_fake), fake_label)
    loss_real.backward()
    loss_fake.backward()
    optimizer.step()
    return loss_real + loss_fake
```

생성자 학습 함수는 가짜 이미지에 **진짜 레이블(1)** 을 붙여 손실을 계산한다. 생성자의 목표가 판별자로 하여금 가짜를 진짜로 믿게 하는 것이기 때문이다. 이것이 앞에서 말한 $$-\log D(G(z))$$ 형태의 손실이다.

```python
def train_generator(optimizer, data_fake):
    b_size = data_fake.size(0)
    real_label = torch.ones(b_size, 1).to(device)   # 생성자 입장에서 정답은 "진짜"
    optimizer.zero_grad()
    loss = criterion(discriminator(data_fake), real_label)
    loss.backward()
    optimizer.step()
    return loss
```

학습 루프에서는 배치마다 판별자를 `k`번, 생성자를 한 번 학습한다.

```python
for epoch in range(epochs):
    loss_g, loss_d = 0.0, 0.0
    for idx, (image, _) in enumerate(train_loader):
        image = image.to(device)
        b_size = len(image)
        for step in range(k):
            # 판별자 학습용 가짜: detach()로 생성자 쪽으로 기울기가 흐르지 않게 한다
            data_fake = generator(torch.randn(b_size, nz).to(device)).detach()
            loss_d += train_discriminator(optim_d, image, data_fake)
        # 생성자 학습용 가짜: 기울기가 생성자까지 흘러야 하므로 detach하지 않는다
        data_fake = generator(torch.randn(b_size, nz).to(device))
        loss_g += train_generator(optim_g, data_fake)

    generated_img = make_grid(generator(torch.randn(b_size, nz).to(device)).cpu().detach())
    save_image(generated_img, f"../chap13/img/gen_img{epoch}.png")   # 에폭별 결과 저장
    losses_g.append(loss_g / idx)
    losses_d.append(loss_d / idx)
```

`torch.randn(b_size, nz)`로 평균 0, 표준편차 1인 잡음을 배치 크기만큼 만든다. 판별자를 학습할 때 가짜 이미지에 `.detach()`를 붙이는 이유는, 판별자 손실의 역전파가 생성자의 계산 그래프까지 내려가지 않게 하기 위해서다. 판별자를 학습하는 동안 생성자는 고정되어야 한다. (책의 코드에는 `loss_d / idxH`라는 오타가 있어 그대로 실행하면 오류가 난다. `idx`로 고쳐야 한다.)

```text
Epoch 0 of 200
Generator loss: 1.28965569, Discriminator loss: 0.94043517
Epoch 2 of 200
Generator loss: 6.74595404, Discriminator loss: 0.20054729
...
Epoch 199 of 200
Generator loss: 1.31970513, Discriminator loss: 1.07432187
```

에폭 0~1의 이미지는 잡음에 불과하고, 100 에폭 무렵에는 숫자 모양이 보이지만 주변에 잡음이 많으며, 199 에폭에서는 잡음이 많이 사라져 더 선명한 숫자가 나온다.

손실 곡선을 그려 보면 초반 몇 에폭 동안 생성자 손실은 치솟고 판별자 손실은 떨어진다. 생성자가 아직 어설퍼서 판별자가 쉽게 구별하기 때문이다. 학습이 진행되면 생성자가 그럴듯한 이미지를 만들면서 판별자가 일부 가짜를 진짜로 착각하게 되고, 생성자 손실은 내려가고 판별자 손실은 올라가 두 값이 일정 수준에서 맞선다. GAN에서는 손실이 0으로 수렴하는 것이 아니라 두 네트워크가 균형을 이루는 것이 정상이다. 이론적인 균형점에서는 판별자가 모든 입력에 0.5를 출력한다.

마지막으로 학습된 생성자에 새 잡음을 넣어 숫자 이미지를 한 장씩 그려 볼 수 있다.

```python
fake_images = generator(torch.randn(b_size, nz).to(device))
for i in range(10):
    fake_images_img = np.reshape(fake_images.data.cpu().numpy()[i], (28, 28))
    plt.imshow(fake_images_img, cmap='gray')
    plt.show()
```

## 13.4 GAN 파생 기술

[📖 thebook.io](https://thebook.io/080289/0946/)

GAN은 두 네트워크가 맞서며 학습하므로 매우 불안정하다. 한쪽이 너무 강해지면 학습이 무너지거나, 생성자가 몇 가지 비슷한 이미지만 반복해서 만드는 **모드 붕괴(mode collapse)** 가 일어난다. 이 장은 GAN을 발전시킨 대표 모델 넷을 소개한다.

| 모델 | 핵심 아이디어 |
| --- | --- |
| DCGAN | 생성자와 판별자를 합성곱 신경망으로 만들고, 안정적으로 학습되는 구조 지침을 제시 |
| cGAN | 생성자와 판별자에 조건(예: 클래스 레이블)을 함께 넣어 생성 결과를 통제 |
| PIX2PIX | 이미지를 입력받아 다른 스타일의 이미지로 바꾸는 조건부 GAN (쌍 데이터 필요) |
| CycleGAN | 두 방향 생성자와 순환 일관성 손실로 쌍 없는 데이터에서 이미지 변환 학습 |

### 13.4.1 DCGAN

**DCGAN(Deep Convolutional GAN)** 은 GAN의 생성자와 판별자를 합성곱 신경망으로 바꾸고, 어떻게 구성해야 안정적으로 학습되는지 구체적인 지침을 정리했다. 구조는 GAN과 같이 생성자와 판별자가 적대적으로 학습한다.

**생성자 네트워크**

생성자는 잡음 벡터를 받아 64x64 이미지를 출력한다. 잡음 벡터는 이미지 모양이 아니므로 먼저 작은 공간 크기의 특성 맵으로 형태를 바꾸고(reshape), **분수 스트라이드 합성곱(fractionally-strided convolution, 전치 합성곱)** 을 반복해 크기를 키운다.

- 풀링층을 모두 없애고, 크기를 키우는 데는 분수 스트라이드 합성곱을 쓴다.
- 배치 정규화로 층이 깊어도 기울기가 안정되게 한다. 단 출력층에는 쓰지 않는다.
- 활성화 함수는 ReLU, 출력층만 tanh를 쓴다.

**판별자 네트워크**

판별자는 64x64 이미지를 받아 진짜/가짜 확률 하나를 출력한다.

- 풀링층 대신 **스트라이드 합성곱(strided convolution)** 으로 크기를 줄인다.
- 배치 정규화를 쓰되 입력층에는 쓰지 않는다.
- 활성화 함수는 LeakyReLU, 출력층만 시그모이드를 쓴다.

**스트라이드 합성곱과 분수 스트라이드 합성곱**

- 스트라이드 합성곱은 필터를 2칸 이상씩 건너뛰며 적용하는 보통의 합성곱이다. 풀링처럼 출력 크기가 줄어들되, 줄이는 방식까지 학습된다. 예를 들어 4x4 입력에 3x3 필터를 스트라이드 2로 적용하면(패딩 1) 2x2 출력이 된다.
- 분수 스트라이드 합성곱은 스트라이드가 1보다 작은 것처럼 동작해 크기를 키운다. 2x2 입력의 원소 사이사이와 바깥에 0을 채워 넣어 5x5 정도로 벌려 놓은 뒤 3x3 필터로 보통 합성곱을 하면, 입력보다 큰 출력을 얻는다.

생성자는 작은 잡음에서 학습 이미지와 같은 해상도의 이미지를 만들어야 하므로 공간을 넓히는 분수 스트라이드 합성곱이 필요하고, 판별자는 이미지에서 진위를 가를 특성을 뽑아내야 하므로 공간을 줄여 가는 스트라이드 합성곱을 쓴다.

### 13.4.2 cGAN

**cGAN(conditional GAN, 조건부 GAN)** 은 GAN의 출력을 사람이 통제하려는 시도에서 나왔다. (책은 cGAN을 "convolutional GAN"으로 풀어 쓰지만, c는 조건(conditional)을 뜻한다.) 일반 GAN은 잡음 벡터 하나로 출력을 만드므로, MNIST로 학습해도 어떤 숫자가 나올지 정할 수 없다.

cGAN은 생성자와 판별자 **모두에** 조건 벡터 $$c$$를 추가 입력으로 넣는다. 예를 들어 숫자 1을 만들고 싶다면 생성자에 잡음과 함께 1을 뜻하는 원-핫 벡터를 넣고, 판별자에도 같은 조건을 넣어 "이 이미지가 진짜이면서 조건 1에 맞는가"를 판정하게 한다. 목적 함수는 GAN의 식에서 $$D(x)$$, $$G(z)$$를 $$D(x \mid c)$$, $$G(z \mid c)$$로 바꾼 것이다.

$$
\min_G \max_D V(D, G) = \mathbb{E}_{x}\left[\log D(x \mid c)\right] + \mathbb{E}_{z}\left[\log\left(1 - D(G(z \mid c) \mid c)\right)\right]
$$

조건으로는 클래스 레이블뿐 아니라 텍스트 태그나 다른 이미지도 쓸 수 있다. 덕분에 원하는 종류의 이미지를 만들거나, 기존 이미지를 원하는 방향으로 변형하는 것이 가능해진다.

### 13.4.3 CycleGAN

GAN과 DCGAN은 무작위 잡음을 입력으로 받으므로 원하는 결과를 정확히 얻기 어렵다. 이미지를 입력으로 받아 다른 이미지로 바꾸는 모델로 PIX2PIX와 CycleGAN이 있다.

**PIX2PIX**

**PIX2PIX** 는 잡음이 아니라 이미지를 입력받아 다른 스타일의 이미지를 출력하는 조건부 GAN이다. 입력 이미지 자체가 조건 역할을 한다. 스케치를 사진으로, 낮 사진을 밤 사진으로, 흑백을 컬러로 바꾸는 식이다. 학습에는 (입력 이미지, 정답 이미지) 쌍이 필요하므로 지도 학습이다.

- **생성자**: 입력과 출력이 모두 이미지이므로, 크기를 줄이며 특징을 뽑는 인코더와 다시 키우며 이미지를 만드는 디코더로 된 인코더-디코더 구조다. 원 논문은 인코더 층의 출력을 대칭되는 디코더 층에 바로 이어 주는 스킵 연결을 둔 U-Net을 쓴다. 출력층은 tanh(-1~1)이므로 입력도 -1~1로 정규화한다.
- **판별자**: DCGAN처럼 스트라이드 2 합성곱을 쌓되, 마지막 두 층은 스트라이드 1에 패딩 없는(valid) 합성곱을 써서 30x30 크기의 출력을 낸다. 이미지 전체에 대해 확률 하나를 내는 대신 이미지의 각 부분(패치)이 진짜인지를 따로 판정하는 것이다(**PatchGAN**). 그래서 질감 같은 세부에 더 집중할 수 있다.

손실은 조건부 GAN 손실에, 생성 이미지가 정답 이미지와 픽셀 단위로 가까워지게 하는 **L1 손실** 을 더한다. 판별자를 속이는 것만으로는 입력과 상관없는 그럴듯한 이미지를 만들 수 있기 때문에, 정답과의 거리를 직접 줄이는 항이 필요하다.

$$
\mathcal{L}_{cGAN}(G, D) = \mathbb{E}_{x, y}\left[\log D(x, y)\right] + \mathbb{E}_{x, z}\left[\log\left(1 - D(x, G(x, z))\right)\right]
$$

$$
\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}\left[\lVert y - G(x, z) \rVert_1\right], \qquad G^\ast = \arg\min_G \max_D \; \mathcal{L}_{cGAN}(G, D) + \lambda \, \mathcal{L}_{L1}(G)
$$

여기서 $$x$$는 입력 이미지, $$y$$는 정답 이미지, $$\lambda$$는 L1 항의 가중치다.

**CycleGAN**

PIX2PIX는 강력하지만 쌍 데이터가 반드시 필요하다. 신발 윤곽선에서 신발 사진을 만들려면 같은 신발의 윤곽선 이미지와 사진이 짝지어 있어야 하는데, 흑백-컬러처럼 같은 장면의 두 버전을 대량으로 구하기는 어렵다. **CycleGAN** 은 짝이 없는(unpaired) 두 이미지 집합, 예를 들어 말 사진 묶음과 얼룩말 사진 묶음만으로 둘 사이의 변환을 배운다.

CycleGAN에는 네트워크가 넷 있다.

- 생성자 $$G$$: 도메인 X(말)의 이미지를 도메인 Y(얼룩말)로 바꾼다.
- 생성자 $$F$$: 도메인 Y의 이미지를 도메인 X로 바꾼다.
- 판별자 $$D_Y$$: Y 도메인의 진짜와 $$G$$가 만든 가짜를 구별한다. $$G$$는 $$D_Y$$와 적대적으로 학습한다.
- 판별자 $$D_X$$: X 도메인의 진짜와 $$F$$가 만든 가짜를 구별한다. $$F$$는 $$D_X$$와 적대적으로 학습한다.

적대적 손실만 있으면 $$G$$는 입력 말과 상관없이 아무 얼룩말 사진이나 만들어도 $$D_Y$$를 속일 수 있다. 그래서 **순환 일관성(cycle consistency)** 을 요구한다. 말 사진 $$x$$를 얼룩말로 바꿨다가($$G(x)$$) 다시 말로 되돌리면($$F(G(x))$$) 원래 $$x$$와 같아야 하고(정방향 일관성), 반대로 $$G(F(y)) \approx y$$여야 한다(역방향 일관성). 이 차이를 L1 거리로 잰 것이 순환 일관성 손실이다.

$$
\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x}\left[\lVert F(G(x)) - x \rVert_1\right] + \mathbb{E}_{y}\left[\lVert G(F(y)) - y \rVert_1\right]
$$

전체 손실은 두 방향의 적대적 손실과 순환 일관성 손실의 합이며, 생성자는 이를 최소화하고 판별자는 최대화한다.

$$
\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda \, \mathcal{L}_{cyc}(G, F)
$$

$$
G^\ast, F^\ast = \arg\min_{G, F} \max_{D_X, D_Y} \mathcal{L}(G, F, D_X, D_Y)
$$

정리하면 PIX2PIX가 생성자 하나와 판별자 하나로 한 방향 변환을 배우고 학습이 L1 손실에 크게 기대는 반면, CycleGAN은 생성자 둘과 판별자 둘로 양방향 변환을 배우고 적대적 손실이 핵심 역할을 하므로 쌍 데이터 없이도 더 유연하게 이미지를 변환한다.

## 정리

- 판별 모델은 $$p(y \mid x)$$를, 생성 모델은 데이터 분포 $$p(x)$$를 배워 새 데이터를 만든다. 분포를 명시적으로 다루는 대표가 VAE, 암시적으로 샘플만 만드는 대표가 GAN이다.
- 오토인코더는 좁은 병목층을 거쳐 입력을 복원하도록 학습하며, 압축, 차원 축소, 비지도 특성 추출에 쓰인다. 책의 잡음 제거 예제는 손실을 잡음 섞인 입력과 비교하므로, 엄밀히 하려면 깨끗한 원본과 비교해야 한다.
- VAE의 인코더는 잠재 변수의 평균과 로그 분산을 출력하고, 재매개변수화 기법 $$z = \mu + \sigma \epsilon$$으로 미분 가능하게 샘플링한다. 손실은 재구성 오차(BCE)와 표준정규분포에 대한 KLD의 합이다.
- GAN은 판별자가 $$\log D(x) + \log(1 - D(G(z)))$$를 최대화하고 생성자가 이를 최소화하는 미니맥스 게임이다. 구현에서는 옵티마이저를 따로 두고, 판별자 학습 때 가짜 이미지를 `detach()`하며, 생성자 손실은 가짜에 진짜 레이블을 붙인 BCE로 계산한다.
- MNIST GAN은 200 에폭 동안 잡음에서 숫자 모양으로 발전하며, 손실은 0으로 가는 것이 아니라 생성자와 판별자가 균형을 이루는 수준에서 맞선다.
- DCGAN은 풀링 없이 스트라이드 합성곱(판별자)과 분수 스트라이드 합성곱(생성자), 배치 정규화, ReLU/LeakyReLU와 tanh/시그모이드 출력으로 GAN을 안정화했다.
- cGAN은 생성자와 판별자에 조건을 함께 넣어 출력을 통제하고, PIX2PIX는 이미지를 조건으로 받아 U-Net 생성자, PatchGAN 판별자, L1 손실로 쌍 데이터 기반 이미지 변환을 학습한다.
- CycleGAN은 두 생성자와 두 판별자, 그리고 $$F(G(x)) \approx x$$를 요구하는 순환 일관성 손실로 짝 없는 두 이미지 집합 사이의 변환을 배운다.

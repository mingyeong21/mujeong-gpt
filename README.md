# 무정GPT: 『무정』 데이터셋을 활용한 Character-Level GPT 구현

## 1. 프로젝트 개요

이 프로젝트는 Karpathy의 GPT 제작 강의와 수업에서 제공된 notebook을 바탕으로, PyTorch를 사용해 작은 GPT-style 언어 모델을 직접 구현한 프로젝트입니다.

본 프로젝트의 목표는 실제 GPT-2와 같은 대규모 언어 모델을 그대로 재현하는 것이 아니라, GPT 계열 모델의 핵심 구조인 **decoder-only Transformer**를 작은 규모에서 구현하고 이해하는 것입니다.

모델은 이광수의 소설 **『무정』** 텍스트를 학습 데이터로 사용하며, 단어 단위가 아니라 **문자 단위(character-level)** 로 텍스트를 처리합니다. 따라서 모델의 학습 목표는 주어진 문자 sequence를 바탕으로 **다음에 올 문자**를 예측하는 것입니다.

예를 들어 다음과 같은 텍스트가 있을 때,

```text
무정GPT
```

입력과 정답은 다음과 같이 구성됩니다.

```text
x: 무 정 G P
y: 정 G P T
```

즉, 모델은 각 위치에서 다음 문자가 무엇인지 예측하도록 학습됩니다.

이 프로젝트를 통해 다음 개념들을 직접 구현하고 이해하는 것을 목표로 했습니다.

* Character-level tokenization
* Token embedding
* Positional embedding
* Masked self-attention
* Multi-head attention
* Transformer block
* Residual connection
* Layer normalization
* Cross entropy loss
* Autoregressive text generation

---

## 2. 데이터셋

본 프로젝트에서는 이광수의 소설 **『무정』** 텍스트를 데이터셋으로 사용했습니다.

Karpathy의 GPT 강의에서는 주로 Shakespeare 텍스트를 사용하지만, 이 프로젝트에서는 수업 notebook의 구조를 바탕으로 데이터셋을 한국어 소설인 『무정』으로 변경했습니다.

### 『무정』을 선택한 이유

『무정』을 선택한 이유는 다음과 같습니다.

첫째, 『무정』은 한국 근대문학을 대표하는 작품 중 하나이기 때문에 한국어 문장 구조와 문체를 학습 데이터로 사용하기에 적합하다고 판단했습니다.

둘째, Shakespeare 데이터셋은 영어 기반 텍스트이므로 알파벳, 공백, 구두점 중심의 문자 패턴을 학습합니다. 반면 『무정』은 한글, 한자, 한국어 문장 부호 등이 포함되어 있어 영어 텍스트와는 다른 character-level language modeling을 실험할 수 있습니다.

셋째, 동일한 GPT-style 모델 구조를 사용하더라도 데이터셋이 달라지면 모델이 학습하는 문자 패턴과 생성 결과가 달라진다는 점을 확인할 수 있습니다.

### Shakespeare 데이터셋과 다른 점

Shakespeare 데이터셋과 『무정』 데이터셋의 차이는 다음과 같습니다.

| 구분    | Shakespeare 데이터셋 | 『무정』 데이터셋              |
| ----- | ---------------- | ---------------------- |
| 언어    | 영어               | 한국어                    |
| 기본 문자 | 알파벳 중심           | 한글 중심                  |
| 문체    | 희곡체, 대사 중심       | 소설체, 서술문 중심            |
| 문자 종류 | 상대적으로 적음         | 한글, 공백, 문장부호 등으로 더 다양함 |
| 학습 특징 | 영어 단어와 대사 패턴 학습  | 한국어 문장 흐름과 문자 패턴 학습    |

특히 character-level 모델에서는 vocabulary가 문자 단위로 구성됩니다. 영어 데이터셋은 알파벳 중심이므로 문자 종류가 비교적 적지만, 한국어 데이터셋은 한글 음절 단위 문자가 많아 vocabulary size가 더 커질 수 있습니다. 이 때문에 동일한 모델 구조를 사용하더라도 한국어 데이터셋에서는 학습 난이도가 더 높아질 수 있습니다.

본 프로젝트에서는 이러한 차이를 고려하여 『무정』 텍스트를 문자 단위로 변환한 뒤, 각 문자를 정수 index로 mapping하여 모델에 입력했습니다.

---

## 3. 모델 구조 및 학습 방식

본 프로젝트의 모델은 작은 GPT-style decoder-only Transformer 구조를 따릅니다.

전체적인 흐름은 다음과 같습니다.

```text
Input text
→ Character tokenization
→ Token embedding
→ Positional embedding
→ Transformer blocks
→ Final layer normalization
→ Linear language modeling head
→ Logits
→ Next-character prediction
```

### 3.1 Character Tokenization

모델은 텍스트를 문자 단위로 나누어 처리합니다.

예를 들어 다음 문장이 있다면,

```text
그는 말했다
```

이를 다음과 같이 문자 단위로 나눕니다.

```text
그 / 는 /   / 말 / 했 / 다
```

각 문자는 정수 index로 변환됩니다.

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
```

여기서 `stoi`는 문자를 정수로 바꾸는 dictionary이고, `itos`는 정수를 다시 문자로 바꾸는 dictionary입니다.

---

### 3.2 Token Embedding

문자 index는 그대로 모델이 이해할 수 없기 때문에 embedding layer를 통해 vector로 변환합니다.

입력 shape은 다음과 같습니다.

```text
[B, T]
```

여기서 `B`는 batch size, `T`는 sequence length입니다.

Token embedding을 거치면 shape은 다음과 같이 바뀝니다.

```text
[B, T] → [B, T, C]
```

여기서 `C`는 embedding dimension입니다.

---

### 3.3 Positional Embedding

Transformer는 RNN과 달리 입력의 순서를 자연스럽게 기억하지 않습니다. 따라서 각 문자가 sequence 안에서 몇 번째 위치에 있는지 알려주는 positional embedding이 필요합니다.

본 프로젝트에서는 token embedding과 positional embedding을 더해 모델의 입력으로 사용했습니다.

```python
x = token_embedding + positional_embedding
```

이를 통해 모델은 각 문자의 종류뿐만 아니라 위치 정보도 함께 사용할 수 있습니다.

---

### 3.4 Masked Multi-Head Self-Attention

GPT는 현재까지의 문자만 보고 다음 문자를 예측해야 하는 autoregressive model입니다. 따라서 현재 위치에서 미래 문자를 보면 안 됩니다.

이를 막기 위해 causal mask를 사용합니다.

```python
wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
```

이 코드는 현재 위치보다 뒤에 있는 미래 token의 attention score를 `-inf`로 바꿉니다. 이후 softmax를 적용하면 해당 위치의 확률은 0이 됩니다.

Self-attention에서는 Query, Key, Value를 사용합니다.

* Query: 현재 token이 찾고자 하는 정보
* Key: 각 token이 가지고 있는 정보의 특징
* Value: 실제로 가져올 정보

Attention score는 Query와 Key의 내적을 통해 계산됩니다.

```python
wei = q @ k.transpose(-2, -1)
```

Multi-head attention은 이러한 attention head를 여러 개 병렬로 사용합니다. 하나의 head가 한 가지 관점에서 문자 간 관계를 학습한다면, 여러 head는 서로 다른 관점에서 문맥 관계를 학습할 수 있습니다.

---

### 3.5 Feed-Forward Network

Attention을 통과한 각 token representation은 feed-forward network를 거칩니다.

Feed-forward network는 각 위치의 representation을 독립적으로 변환하는 MLP입니다.

```python
nn.Linear(emb_dim, 4 * emb_dim)
nn.ReLU()
nn.Linear(4 * emb_dim, emb_dim)
```

Attention이 token 간 정보를 섞는 역할을 한다면, feed-forward network는 각 token의 representation을 더 복잡하게 변환하는 역할을 합니다.

---

### 3.6 Residual Connection과 Layer Normalization

Transformer block 안에서는 residual connection과 layer normalization을 사용했습니다.

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

Residual connection은 입력을 그대로 더해주는 구조입니다. 이를 통해 기존 정보가 유지되고, gradient가 더 잘 흐르게 되어 학습이 안정됩니다.

Layer normalization은 각 layer의 입력 분포를 안정화하여 학습을 더 잘 되게 하는 역할을 합니다.

---

### 3.7 학습 방식

모델은 next-character prediction 방식으로 학습됩니다.

입력 sequence `x`와 target sequence `y`는 한 칸 차이가 납니다.

예를 들어 원본 텍스트가 다음과 같다면,

```text
무정GPT
```

입력과 target은 다음과 같이 구성됩니다.

```text
x: 무 정 G P
y: 정 G P T
```

모델은 각 위치에서 다음 문자를 예측하고, 예측 결과와 실제 target을 비교하여 loss를 계산합니다.

Loss function으로는 Cross Entropy Loss를 사용했습니다.

모델의 최종 출력은 확률이 아니라 logits입니다. 학습 시에는 logits와 target을 이용해 cross entropy loss를 계산하고, 생성 시에는 logits에 softmax를 적용하여 다음 문자에 대한 확률분포를 만든 뒤 sampling합니다.

---

## 4. 파일 구조

본 repository의 파일 구조는 다음과 같습니다.

```text
mujeong-gpt/
├── README.md
├── requirements.txt
├── model.py
├── train.py
├── generate.py
├── .gitignore
└── data/
    └── input.txt
```

각 파일의 역할은 다음과 같습니다.

| 파일                 | 설명                                                                                                        |
| ------------------ | --------------------------------------------------------------------------------------------------------- |
| `README.md`        | 프로젝트의 목적, 데이터셋, 모델 구조, 실행 방법, 결과 및 보완점을 설명하는 문서입니다.                                                       |
| `requirements.txt` | 프로젝트 실행에 필요한 Python package 목록을 정리한 파일입니다. 예를 들어 `torch`가 포함됩니다.                                          |
| `model.py`         | GPT-style 모델 구조를 정의한 파일입니다. `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, `TinyGPT` class가 포함됩니다. |
| `train.py`         | 『무정』 텍스트 데이터를 불러오고, character-level tokenization을 수행한 뒤 모델을 학습시키는 파일입니다. 학습이 끝나면 checkpoint를 저장합니다.       |
| `generate.py`      | 학습된 checkpoint를 불러와 새로운 텍스트를 생성하는 파일입니다. 시작 문자를 입력으로 받아 autoregressive하게 다음 문자를 생성합니다.                    |
| `.gitignore`       | `.venv`, `__pycache__`, checkpoint 파일 등 GitHub에 올릴 필요가 없는 파일을 제외하기 위한 설정 파일입니다.                           |
| `data/input.txt`   | 모델 학습에 사용되는 『무정』 텍스트 데이터 파일입니다.                                                                           |

### 주요 파일 설명

#### `model.py`

`model.py`는 모델의 구조를 담당합니다.

주요 class는 다음과 같습니다.

* `Head`: 하나의 masked self-attention head
* `MultiHeadAttention`: 여러 attention head를 병렬로 사용하는 모듈
* `FeedForward`: attention 이후 각 token representation을 변환하는 MLP
* `Block`: self-attention, feed-forward, residual connection, layer normalization을 포함한 Transformer block
* `TinyGPT`: 전체 GPT-style language model

#### `train.py`

`train.py`는 모델 학습을 담당합니다.

주요 과정은 다음과 같습니다.

1. `data/input.txt`에서 텍스트를 불러옵니다.
2. 전체 문자 집합을 만들고, 각 문자를 정수 index로 변환합니다.
3. 입력 sequence `x`와 target sequence `y`를 만듭니다.
4. `TinyGPT` 모델을 생성합니다.
5. Cross entropy loss를 사용해 모델을 학습합니다.
6. 학습된 모델과 vocabulary 정보를 checkpoint로 저장합니다.

#### `generate.py`

`generate.py`는 학습된 모델을 사용해 새로운 텍스트를 생성합니다.

주요 과정은 다음과 같습니다.

1. 저장된 checkpoint를 불러옵니다.
2. 모델 구조와 vocabulary를 복원합니다.
3. 시작 문자를 입력으로 넣습니다.
4. 모델이 다음 문자를 예측합니다.
5. 예측된 문자를 기존 입력 뒤에 붙입니다.
6. 이 과정을 반복하여 새로운 텍스트를 생성합니다.

#### `requirements.txt`

프로젝트 실행에 필요한 package를 정리합니다.

예시:

```text
torch
```

---

## 5. 결과

본 프로젝트에서는 『무정』 텍스트를 기반으로 character-level GPT 모델을 학습시켰습니다.

학습이 진행되면서 모델은 단순히 임의의 문자를 출력하는 단계에서 벗어나, 점차 데이터셋에 등장하는 문자 배열과 문장 패턴을 일부 학습하게 됩니다. 특히 반복적으로 등장하는 조사, 어미, 문장 부호, 공백 패턴 등을 학습할 수 있습니다.

생성 결과는 학습량, 데이터 크기, embedding dimension, layer 수, block size 등에 따라 달라집니다. 작은 모델과 짧은 학습 시간에서는 문법적으로 완전한 문장이 생성되기보다는, 『무정』 데이터셋에 등장하는 문자적 특징과 문장 흐름을 부분적으로 모방하는 결과가 나타납니다.

### 최종 결과 예시

아래는 학습된 모델을 이용해 생성한 텍스트 예시입니다.

```text
[여기에 generate.py 실행 결과를 붙여넣기]
```

예를 들어 generate.py를 실행한 뒤 출력된 문장을 위 코드 블록 안에 넣으면 됩니다.

### 결과 해석

생성 결과를 통해 character-level GPT가 다음과 같은 특징을 보인다는 점을 확인할 수 있었습니다.

1. 문자 단위의 반복 패턴을 학습한다.
2. 공백과 문장부호의 사용 패턴을 일부 학습한다.
3. 자주 등장하는 조사나 어미 형태를 따라 하려는 경향이 있다.
4. 하지만 작은 모델에서는 긴 문맥을 자연스럽게 유지하기 어렵다.
5. 단어 단위 의미를 정확히 이해하기보다는 문자 배열의 통계적 패턴을 학습한다.

---

## 실행 방법

필요한 package를 설치합니다.

```bash
pip install -r requirements.txt
```

모델을 학습합니다.

```bash
python train.py
```

학습된 모델로 텍스트를 생성합니다.

```bash
python generate.py
```

---

## 참고

* Karpathy GPT from Scratch Lecture
* 수업 Notebook
* PyTorch
* 이광수, 『무정』

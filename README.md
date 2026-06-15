# 무정GPT: 『무정』 데이터셋을 활용한 Character-Level GPT 구현

This repository implements a small GPT-style language model in PyTorch, based on Karpathy's GPT-from-scratch lecture and Notebook 6.

The goal of this project is not to reproduce the full-scale GPT-2 model, but to understand the core structure of a decoder-only Transformer.

## 1. 프로젝트 개요

이 프로젝트는 Andrej Karpathy의 GPT 구현 강의를 바탕으로, 작은 규모의 character-level GPT 언어모델을 직접 구현한 것이다.

기존 강의에서는 Tiny Shakespeare 데이터셋을 사용했지만, 본 프로젝트에서는 이광수의 장편소설 『무정』 텍스트를 사용하였다.

이 프로젝트의 목표는 대규모 GPT-2를 완전히 재현하는 것이 아니라, GPT의 핵심 원리인 다음 글자 예측, token embedding, positional embedding, masked self-attention, autoregressive generation을 이해하고 구현하는 것이다.

## 2. 데이터셋

본 프로젝트에서는 이광수의 장편소설 『무정』 전문 텍스트를 학습 데이터로 사용하였다.

『무정』은 한국 근대 장편소설로, 기존 Karpathy 강의의 영어 희곡 데이터셋인 Tiny Shakespeare와 다른 언어적·문체적 특징을 가진다.

### 데이터셋 선택 이유

1. Tiny Shakespeare와 다른 한국어 문학 데이터셋을 사용하기 위해
2. 한국어 장편소설의 문장 구조와 문체를 character-level GPT가 학습할 수 있는지 확인하기 위해
3. 영어 텍스트와 달리 한국어의 조사, 어미, 띄어쓰기, 문장부호 사용 패턴을 관찰하기 위해

### 전처리

원본 텍스트는 `input.txt`로 저장하였다.  
한국어 텍스트가 깨지지 않도록 UTF-8 인코딩으로 저장하고 읽도록 설정하였다.

전처리 과정에서는 다음을 수행하였다.

- 줄바꿈 정리
- 과도한 공백 제거
- 깨진 문자 제거
- 학습에 불필요한 메타데이터 제거

원문에 포함된 한자는 대부분 인명 표기나 원문 형식의 일부로 판단하여 필요한 경우만 정리하였다.

## File Structure

```text
mujeong-gpt/
├── README.md
├── requirements.txt
├── model.py
├── train.py
├── generate.py
└── data/
    └── input.txt

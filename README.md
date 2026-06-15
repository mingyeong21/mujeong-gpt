# 무정GPT: 『무정』 데이터셋을 활용한 Character-Level GPT 구현

This repository implements a small GPT-style language model in PyTorch, based on Karpathy's GPT-from-scratch lecture and Notebook 6.

The goal of this project is not to reproduce the full-scale GPT-2 model, but to understand the core structure of a decoder-only Transformer.

## Project Goal

This model learns to predict the next character from a given text sequence.

The overall process is:

1. Load text data
2. Convert characters into integer tokens
3. Create input-target pairs for next-token prediction
4. Train a small GPT-style Transformer model
5. Generate new text from the trained model

## Model Architecture

The model consists of:

- Token Embedding
- Positional Embedding
- Masked Multi-Head Self-Attention
- Feed-Forward Network
- Residual Connections
- Layer Normalization
- Linear Language Modeling Head

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

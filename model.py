import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """하나의 masked self-attention head"""

    def __init__(self, emb_dim, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )
        
    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """여러 개의 attention head를 병렬로 사용하는 모듈"""

    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList([
            Head(emb_dim, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    ### 32차원로 나눠진 것을 128차원으로 다시 합침
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """각 token representation에 적용되는 MLP"""

    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block 하나"""

    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ffwd = FeedForward(emb_dim, dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """작은 GPT-style language model"""

    def __init__(
        self,
        vocab_size,
        block_size,
        emb_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding_table = nn.Embedding(block_size, emb_dim)

        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)    # (B, T, C) = (64, 64, 128) 글자 정보
        pos = torch.arange(T, device=idx.device)    
        pos_emb = self.position_embedding_table(pos)    # (T, C) = (64, 128) 위치 정보
        
        ### broadcasting rule 적용
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)    # (64, 64, 1676) 각 위치마다 1676개 글자 후보에 대한 점수 출력

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

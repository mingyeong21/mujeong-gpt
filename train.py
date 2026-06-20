import torch
from torch.utils.data import Dataset, DataLoader
from model import TinyGPT


# -----------------------------
# Hyperparameters
# -----------------------------

batch_size = 64
block_size = 64
max_iters = 1000
learning_rate = 3e-4
eval_interval = 100

emb_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Dataset
# -----------------------------

class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


# -----------------------------
# Load text
# -----------------------------

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("dataset length:", len(text))
print("vocab size:", len(set(text)))

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return "".join([itos[i] for i in indices])


data = torch.tensor(encode(text), dtype=torch.long)

### train_data = 모델이 공부하는 데이터, val_data = 모델이 처음 보는 데이터로 실력 확인
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

train_dataset = NextTokenDataset(train_data, block_size)
val_dataset = NextTokenDataset(val_data, block_size)

### DataLoader = 데이터셋에서 batch 자동으로 뽑아옴.
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)


# -----------------------------
# Model
# -----------------------------

model = TinyGPT(
    vocab_size=vocab_size,
    block_size=block_size,
    emb_dim=emb_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# -----------------------------
# Training loop
# -----------------------------

for step, (xb, yb) in enumerate(train_loader):
    if step >= max_iters:
        break

    xb = xb.to(device)
    yb = yb.to(device)

    ### logits = 다음 글자 후보들의 점수, loss = 예측이 얼마나 틀렸는지
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}, loss {loss.item():.4f}")


# -----------------------------
# Save checkpoint
# -----------------------------

checkpoint = {
    "model_state_dict": model.state_dict(),
    "stoi": stoi,
    "itos": itos,
    "vocab_size": vocab_size,
    "block_size": block_size,
    "emb_dim": emb_dim,
    "num_heads": num_heads,
    "num_layers": num_layers,
    "dropout": dropout,
}

torch.save(checkpoint, "tiny_gpt_checkpoint.pt")

print("Training finished. Model saved as tiny_gpt_checkpoint.pt")
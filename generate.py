import torch
from model import TinyGPT


device = "cuda" if torch.cuda.is_available() else "cpu"


checkpoint = torch.load(
    "tiny_gpt_checkpoint.pt",
    map_location=device
)

stoi = checkpoint["stoi"]
itos = checkpoint["itos"]

def encode(s):
    return [stoi[c] for c in s]

def decode(indices):
    return "".join([itos[i] for i in indices])


model = TinyGPT(
    vocab_size=checkpoint["vocab_size"],
    block_size=checkpoint["block_size"],
    emb_dim=checkpoint["emb_dim"],
    num_heads=checkpoint["num_heads"],
    num_layers=checkpoint["num_layers"],
    dropout=checkpoint["dropout"],
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


start_text = " "
context = torch.tensor(
    [encode(start_text)],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=500,
    temperature=0.8
)

print(decode(generated[0].tolist()))
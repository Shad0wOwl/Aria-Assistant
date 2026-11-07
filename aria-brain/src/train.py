
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 16
EMBEDDING_SIZE = 32
BATCH_SIZE = 32
STEPS = 1000
LEARNING_RATE = 0.003

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training.txt"
MODEL_PATH = ROOT / "models" / "aria_v0.1.pt"

text = DATA_PATH.read_text(encoding="utf-8")

chars = sorted(set(text))
vocab_size = len(chars)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for char, i in stoi.items()}

data = torch.tensor(
    [stoi[char] for char in text],
    dtype=torch.long
)

if len(data) <= BLOCK_SIZE:
    raise ValueError("Training text is too short.")

print(f"Device: {DEVICE}")
print(f"Characters: {len(data)}")
print(f"Vocabulary size: {vocab_size}")

def get_batch():
    positions = torch.randint(
        len(data) - BLOCK_SIZE,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[i:i + BLOCK_SIZE] for i in positions
    ])

    y = torch.stack([
        data[i + 1:i + BLOCK_SIZE + 1]
        for i in positions
    ])

    return x.to(DEVICE), y.to(DEVICE)


class AriaBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_SIZE)
        self.network = nn.Sequential(
            nn.Linear(BLOCK_SIZE * EMBEDDING_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x, targets=None):
        embeddings = self.embedding(x)
        flattened = embeddings.reshape(x.shape[0], -1)

        logits = self.network(flattened)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets[:, -1])

        return logits, loss


model = AriaBrain().to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

for step in range(STEPS):
    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    "model_state": model.state_dict(),
    "vocabulary": chars,
    "block_size": BLOCK_SIZE,
    "embedding_size": EMBEDDING_SIZE
}, MODEL_PATH)

print(f"Model saved: {MODEL_PATH}")

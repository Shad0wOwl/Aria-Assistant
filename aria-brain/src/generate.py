
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "aria_v0.1.pt"

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

chars = checkpoint["vocabulary"]
block_size = checkpoint["block_size"]
embedding_size = checkpoint["embedding_size"]
vocab_size = len(chars)

stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}


class AriaBrain(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.network = nn.Sequential(
            nn.Linear(block_size * embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        embeddings = self.embedding(x)
        flattened = embeddings.reshape(x.shape[0], -1)
        return self.network(flattened)


model = AriaBrain().to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()


def generate(prompt, max_new_chars=150):
    unknown = set(prompt) - set(chars)

    if unknown:
        raise ValueError(
            f"Nieznane znaki: {unknown}. "
            "Użyj znaków występujących w danych treningowych."
        )

    # Uzupełniamy kontekst do 16 znaków.
    context = [stoi["\n"]] * block_size
    context += [stoi[c] for c in prompt]

    generated = prompt

    with torch.no_grad():
        for _ in range(max_new_chars):
            x = torch.tensor(
                [context[-block_size:]],
                dtype=torch.long,
                device=device
            )

            logits = model(x)
            next_id = torch.argmax(logits, dim=-1).item()

            next_char = itos[next_id]
            generated += next_char
            context.append(next_id)

    return generated


if __name__ == "__main__":
    prompt = input("Ty: ")
    print("\nAria:")
    print(generate(prompt))

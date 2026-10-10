import torch
from torchvision.datasets import OxfordIIITPet

classes = OxfordIIITPet(root="data", split="test", download=False).classes

def per_query_precision(emb, labels, k=5):
    out = []
    for q in range(len(emb)):
        scores = emb @ emb[q]
        top = torch.topk(scores, k=k + 1).indices[1:]  # [1:] skips the query itself (safe, see 09)
        out.append((labels[top] == labels[q]).sum().item() / k)
    return torch.tensor(out)

def per_breed(path):
    data = torch.load(path)
    labels = data["labels"]
    p = per_query_precision(data["emb"], labels)
    print(path, "mean Precision@5:", round(p.mean().item(), 4))

    # Average the per-image scores within each breed
    scores = {}
    for b in range(37):
        scores[classes[b]] = p[labels == b].mean().item()
    ranked = sorted(scores.items(), key=lambda x: x[1])
    print("  Hardest 5:", ranked[:5])
    print("  Easiest 5:", ranked[-5:])

per_breed("clip_pets.pt")
per_breed("dino_pets.pt")
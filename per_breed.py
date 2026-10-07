import torch
from torchvision.datasets import OxfordIIITPet

classes = OxfordIIITPet(root="data", split="test", download=False).classes

def per_query_precision(emb, labels, k=5):
    out = []
    for q in range(len(emb)):
        scores = emb @ emb[q]
        top = torch.topk(scores, k=k + 1).indices[1:]
        out.append((labels[top] == labels[q]).sum().item() / k)
    return torch.tensor(out)

clip = torch.load("clip_pets.pt")
p = per_query_precision(clip["emb"], clip["labels"])
print(p.shape)
print(p.mean())
import torch

data = torch.load("clip_pets.pt")

emb = data["emb"]
labels = data["labels"]
ids = data["ids"]

print("emb:", emb.shape)
print("labels:", labels.shape)
print("ids:", len(ids))
print("first 5 labels:", labels[:5])

from torchvision.datasets import OxfordIIITPet

ds = OxfordIIITPet(root="data", split="test", download=False)
classes = ds.classes

query = 0
scores = emb @ emb[query]
top = torch.topk(scores, k=6)

for score, idx in zip(top.values, top.indices):
    print(f"image {idx.item():3d}  score {score.item():.3f}  breed: {classes[labels[idx].item()]}")

query = 3
scores = emb @ emb[query]
top = torch.topk(scores, k=6)

for score, idx in zip(top.values, top.indices):
    print(f"image {idx.item():3d}  score {score.item():.3f}  breed: {classes[labels[idx].item()]}")

k = 5
precisions = []
for q in range(len(emb)):
    scores = emb @ emb[q]
    top = torch.topk(scores, k=k + 1).indices[1:]
    hits = (labels[top] == labels[q]).sum().item()
    precisions.append(hits / k)

print("Mean Precision@5:", sum(precisions) / len(precisions))
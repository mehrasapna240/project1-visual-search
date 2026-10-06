import torch

def precision_at_k(emb, labels, k=5):
    precisions = []
    for q in range(len(emb)):
        scores = emb @ emb[q]
        top = torch.topk(scores, k=k + 1).indices[1:]
        hits = (labels[top] == labels[q]).sum().item()
        precisions.append(hits / k)
    return sum(precisions) / len(precisions)

clip = torch.load("clip_pets.pt")
dino = torch.load("dino_pets.pt")

print("CLIP Precision@5:", round(precision_at_k(clip["emb"], clip["labels"]), 4))
print("DINO Precision@5:", round(precision_at_k(dino["emb"], dino["labels"]), 4))
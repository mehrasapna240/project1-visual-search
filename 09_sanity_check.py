import torch

def sanity_check(path):
    data = torch.load(path)
    emb = data["emb"]
    labels = data["labels"]

    # Q1: does an image score 1.0 against itself?
    print(path, "self-score:", f"{(emb @ emb[0])[0].item():.4f}")

    # Q2: is every image's best match itself?
    sims = emb @ emb.T
    print("  best match is itself for all:", (sims.argmax(dim=1) == torch.arange(len(emb))).all().item())

    # Q3: Precision@1, is the nearest OTHER image the same breed?
    sims.fill_diagonal_(-1)  # so an image can't pick itself
    nearest = sims.argmax(dim=1)
    print("  Precision@1:", f"{(labels[nearest] == labels).float().mean().item():.4f}")

sanity_check("clip_pets.pt")
sanity_check("dino_pets.pt")
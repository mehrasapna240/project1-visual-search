import torch

# for CLIP
clip  = torch.load('clip_pets.pt')
emb = clip['emb']

# Q1: does an image score 1.0 against itself? (checks the vectors are normalized)
scores = emb @ emb[0]
print(scores[0])

# Q2: is every image's best match itself? (if True, skipping the first result is safe)
sims = emb @ emb.T
best = sims.argmax(dim=1)
print((best == torch.arange(500)).all())

# Q3 (Precision@1): is the nearest OTHER image the same breed?
labels = clip["labels"]
sims.fill_diagonal_(-1)  # so an image can't pick itself
nearest = sims.argmax(dim=1)
print((labels[nearest] == labels).float().mean())


# for DINO

dino  = torch.load('dino_pets.pt')
emb = dino['emb']

# Q1: does an image score 1.0 against itself? (checks the vectors are normalized)
scores = emb @ emb[0]
print(scores[0])

# Q2: is every image's best match itself? (if True, skipping the first result is safe)
sims = emb @ emb.T
best = sims.argmax(dim=1)
print((best == torch.arange(500)).all())

# Q3 (Precision@1): is the nearest OTHER image the same breed?
labels = dino["labels"]
sims.fill_diagonal_(-1)  # so an image can't pick itself
nearest = sims.argmax(dim=1)
print((labels[nearest] == labels).float().mean())

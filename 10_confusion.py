import torch
from collections import Counter
from torchvision.datasets import OxfordIIITPet

classes = OxfordIIITPet(root="data", split="test", download=False).classes

def confusion_pairs(path):
    data = torch.load(path)
    emb = data["emb"]
    labels = data["labels"]

    sims = emb @ emb.T
    sims.fill_diagonal_(-1)  # so an image can't pick itself
    nearest = sims.argmax(dim=1)

    # Q: for which images is the nearest neighbour a different breed?
    wrong = labels[nearest] != labels
    print(path, "wrong:", wrong.sum().item())

    # Q: when it's wrong, which breed was found instead?
    pairs = Counter()
    for true, found in zip(labels[wrong], labels[nearest][wrong]):
        pairs[(classes[true.item()], classes[found.item()])] += 1

    for (true, found), n in pairs.most_common(10):
        print(f"  {true} -> {found}: {n}")

confusion_pairs("clip_pets.pt")
confusion_pairs("dino_pets.pt")
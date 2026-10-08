import random
import torch
from torchvision.datasets import OxfordIIITPet

ds = OxfordIIITPet(root='data', split='test', download=False)

random.seed(0)
picked = random.sample(range(len(ds)), 500)

clip_data = torch.load('clip_pets.pt')
print('Same images as CLIP:', picked == clip_data['ids'])

from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
model = AutoModel.from_pretrained("facebook/dino-vitb16")

all_emb, all_labels = [], []
for start in range(0, len(picked), 32):
    batch_ids = picked[start:start + 32]
    images, labels = [], []
    for i in batch_ids:
        img, label = ds[i]
        images.append(img.convert("RGB"))
        labels.append(label)

    inputs = processor(images=images, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    emb = out.last_hidden_state[:, 0]
    emb = emb / emb.norm(dim=-1, keepdim=True)

    all_emb.append(emb)
    all_labels.extend(labels)
    print(f"embedded {start + len(batch_ids)} / {len(picked)}")

all_emb = torch.cat(all_emb)
labels = torch.tensor(all_labels)
print("Embeddings:", all_emb.shape, "Labels:", labels.shape)
print("Same labels as CLIP:", torch.equal(labels, clip_data["labels"]))

torch.save({"emb": all_emb, "labels": labels, "ids": picked}, "dino_pets.pt")
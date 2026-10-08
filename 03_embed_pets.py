import random
import torch
from torchvision.datasets import OxfordIIITPet
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. Download the dataset (about 800 MB, only the first time)
ds = OxfordIIITPet(root="data", split="test", download=True)
print("Total images:", len(ds))
print("First breeds:", ds.classes[:5])

# 2. Pick 500 random images
random.seed(0)
picked = random.sample(range(len(ds)), 500)

# 3. Embed them in batches of 32
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
        pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
        emb = model.visual_projection(pooled)
    emb = emb / emb.norm(dim=-1, keepdim=True)

    all_emb.append(emb)
    all_labels.extend(labels)
    print(f"embedded {start + len(batch_ids)} / {len(picked)}")

all_emb = torch.cat(all_emb)
labels = torch.tensor(all_labels)
print("Embeddings:", all_emb.shape, "Labels:", labels.shape)

# 4. Save so we never have to recompute
torch.save({"emb": all_emb, "labels": labels, "ids": picked}, "clip_pets.pt")
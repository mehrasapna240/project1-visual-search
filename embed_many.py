import torch
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 1. Download 6 photos
ids = ["000000039769", "000000000139", "000000000285",
       "000000000632", "000000000724", "000000000776"]
images = []
for i in ids:
    url = f"http://images.cocodataset.org/val2017/{i}.jpg"
    images.append(Image.open(requests.get(url, stream=True).raw).convert("RGB"))

# 2. Embed all of them in one go
inputs = processor(images=images, return_tensors="pt")
with torch.no_grad():
    pooled = model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    emb = model.visual_projection(pooled)
print("Embeddings shape:", emb.shape)

# 3. Normalize each vector to length 1
emb = emb / emb.norm(dim=-1, keepdim=True)
print("Length of first vector:", emb[0].norm())

# 4. Compare every image with every other image
sims = emb @ emb.T
print(sims)

# 5. Search: pick a query image, find its closest matches
query = 2                      # image 2 = the bear
scores = sims[query]           # row 2 of the table: similarity to every image
top = torch.topk(scores, k=4)  # the 4 highest scores

for rank, (score, idx) in enumerate(zip(top.values, top.indices)):
    print(f"rank {rank}: image {idx.item()}  score {score.item():.4f}")
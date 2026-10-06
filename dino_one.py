import torch
import requests
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

processor = AutoImageProcessor.from_pretrained("facebook/dino-vitb16")
model = AutoModel.from_pretrained("facebook/dino-vitb16")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

inputs = processor(images=image, return_tensors='pt')
with torch.no_grad():
    out = model(**inputs)

print("Prepared shape:", inputs['pixel_values'].shape)
print('output shape:', out.last_hidden_state.shape)

emb = out.last_hidden_state[:, 0]
emb = emb / emb.norm(dim=-1, keepdim=True)

print("DINO embedding shape:", emb.shape)
print("Length:", emb.norm(dim=-1))
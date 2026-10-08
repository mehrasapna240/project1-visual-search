import torch
import requests
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. Load the pre-trained CLIP model and its image preparer
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 2. Download one photo
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 3. Prepare the image (resize + normalize) so CLIP can read it
inputs = processor(images=image, return_tensors="pt")
print("Prepared image shape:", inputs["pixel_values"].shape)

# 4. Turn the image into an embedding
with torch.no_grad():
    vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
    pooled = vision_out.pooler_output          # raw image features from the ViT
    emb = model.visual_projection(pooled)      # projected into the shared image-text space

print("Pooled shape:", pooled.shape)
print("Embedding shape:", emb.shape)
print("First 5 numbers:", emb[0][:5])

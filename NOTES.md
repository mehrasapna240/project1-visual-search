# Project 1 notes: what each file does and what we checked

Project: CLIP vs DINO image retrieval on the Oxford-IIIT Pet test split (500 images, seed 0).
Files are numbered in the order they were written.

## 01_embed_one.py
- Downloads one sample photo by URL, prepares it (resize and normalize), and turns it into one
  CLIP embedding with CLIP ViT-B/32 (vision model, then projection).
- Checked: it prints three shapes: the prepared image (1 x 3 x 224 x 224), the pooled features
  (1 x 768), and the final embedding (1 x 512), plus the first 5 numbers.
- Not normalized here. The length-1 step was added next, in 02_embed_many.py.

## 02_embed_many.py
- Downloads 6 sample photos by ID, embeds them all in one batch with CLIP ViT-B/32, and
  normalizes each embedding to length 1 so a dot product equals cosine similarity.
- Builds a 6 x 6 table of similarities (every image against every other image), then does a
  first search: pick image 2 (the bear) as the query and list its top 4 matches.
- Checked: embeddings shape is 6 x 512, the first vector's length prints as 1.0 after
  normalizing, and rank 0 of the search is the query image itself with a score of about 1.0.
- First time we saw that an image always matches itself best, which is why later scripts
  skip the first result (and why 09_sanity_check.py tests it).

## 03_embed_pets.py
- Picks 500 test images with `random.seed(0)`, embeds them with CLIP ViT-B/32 in batches of 32,
  normalizes each embedding to length 1, and saves them with labels and image ids to `clip_pets.pt`.
- Checked: shapes printed as 500 x 512 embeddings and 500 labels.
- Why normalize: so a dot product equals cosine similarity.

## 04_search_pets.py
- Loads `clip_pets.pt`, searches with one query image (top 6, with scores and breeds), then uses
  every image as a query and averages Precision@5.
- Checked: CLIP Precision@5 = 0.4708.

## 05_dino_one.py
- Loads DINO ViT-B/16, embeds the same sample photo as 01_embed_one.py, takes the CLS token
  (the first token, which summarizes the whole image) as the embedding, and normalizes it
  to length 1.
- Checked: prints the prepared image shape (1 x 3 x 224 x 224), the model output shape
  (1 x 197 x 768: 196 image patches plus the CLS token, 768 numbers each), the embedding shape
  (1 x 768), and the length (1.0).
- Difference from CLIP: CLIP uses a projection layer on top of its features, so it gives 512
  numbers. Here there is no projection, so the CLS token itself is the embedding, with 768 numbers.

## 06_embed_pets_dino.py
- Same 500 images, embedded with DINO ViT-B/16 (CLS token, 768 numbers), saved to `dino_pets.pt`.
- Checked in code: same 500 images in the same order as CLIP, and identical labels.

## 07_compare.py
- A `precision_at_k` function that scores both models the same way.
- Checked: CLIP Precision@5 = 0.4708, DINO Precision@5 = 0.7396.
- Prediction made before running DINO: DINO wins. Correct.

## 08_per_breed.py
- Precision@5 for every query image, then averaged per breed.
- Checked: 500 scores, mean matches 0.4708 (CLIP) and 0.7396 (DINO).
- Finding: cats are the hard cases for both models. Maine Coon is the worst breed for both; Siamese
  is much worse for CLIP (0.11) than for DINO. All five of DINO's easiest breeds are dogs.
- Caution: each breed has only about 11 to 14 images, so only big gaps mean something.
- Output saved in `results_per_breed.txt`.

## 09_sanity_check.py
- Q1: does an image score 1.0 against itself? Yes (1.0000), so vectors are normalized.
- Q2: is every image's best match itself? True for all 500, for both models, so skipping the
  first search result is safe.
- Q3: Precision@1, the nearest OTHER image has the same breed (self masked with
  `fill_diagonal_(-1)`). CLIP = 0.632, DINO = 0.830.
- Meaning: the CLIP embeddings are sensible (random guessing would be about 0.03), so the 0.47 is
  not a bug. DINO still leads at the first result.

## 10_confusion.py (in progress)
- Step 1: count how often CLIP's nearest neighbour is the wrong breed.
- Expected: 184 of 500 (500 minus 316 right at Precision@1 of 0.632).
- Next: which breed pairs get confused, and whether CLIP and DINO make the same mistakes.

## Results so far

| Model | Precision@1 | Precision@5 |
|---|---|---|
| CLIP ViT-B/32 | 0.6320 | 0.4708 |
| DINO ViT-B/16 | 0.8300 | 0.7396 |

## Still open
- Fairness check: CLIP ViT-B/16 (patch size may explain part of the gap).
- More seeds or all 3,669 images, report the average and spread.
- CLIP zero-shot, more metrics, supervised baseline, README, demo.

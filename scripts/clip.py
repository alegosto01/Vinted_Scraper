from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import cv2
import numpy as np
from scipy.stats import skew, kurtosis
from pathlib import Path



clip_model_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
clip_model = CLIPModel.from_pretrained(clip_model_id)
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)


def check_item(option1, option2, image):

    # Load and preprocess image
    image = Image.open(image)
    inputs = clip_processor(text=[option1, option2], images=image, return_tensors="pt", padding=True)

    # Perform image recognition
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # print(f"Probabilities: {probs[0]}")    
    return probs[0].detach().numpy()




from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

search = "jbl charge 5"
item = "jbl flip 5"

# embeddings
emb = model.encode([search, item], convert_to_tensor=True, normalize_embeddings=True)

# cosine similarity
score = util.cos_sim(emb[0], emb[1]).item()
print("similarity:", score)

if score > 0.75:   # tune threshold
    print("✅ adequate (same product)")
else:
    print("❌ not adequate")



# # pip install transformers torch
# from transformers import pipeline

# # Choose a local NLI model (multilingual):
# # "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"  (multilingual)
# # or "MoritzLaurer/deberta-v3-base-mnli"     (English-focused)
# clf = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

# labels = ["keep", "discard"]
# hypothesis_template = "This item should be {}."

# texts = [
#     "Bottiglia vuota senza profumo.",
#     "Profumo nuovo, imballo originale, scontrino."
# ]

# for t in texts:
#     out = clf(t, labels, hypothesis_template=hypothesis_template, multi_label=False)
#     print(t, "->", out["labels"][0], out["scores"][0])
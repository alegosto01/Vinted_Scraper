from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image


CLIP_MODEL_ID = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
TEXT_MODEL_ID = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


@lru_cache(maxsize=1)
def _get_clip_components():
    from transformers import CLIPImageProcessor, CLIPModel, CLIPProcessor, CLIPTokenizerFast

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    try:
        clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
    except OSError:
        # Older/local caches may have the tokenizer and image preprocessor but not a
        # processor_config.json. Build the processor from the cached components.
        tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
        image_processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID, local_files_only=True)
        clip_processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
    clip_model.eval()
    return clip_model, clip_processor


@lru_cache(maxsize=1)
def _get_text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TEXT_MODEL_ID)


def check_item(option1: str, option2: str, image: str | list[str]) -> np.ndarray:
    import torch

    clip_model, clip_processor = _get_clip_components()
    image_paths = image if isinstance(image, list) else [image]
    loaded_images = [Image.open(path).convert("RGB") for path in image_paths]
    inputs = clip_processor(text=[option1, option2], images=loaded_images, return_tensors='pt', padding=True)
    with torch.inference_mode():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()
    return probs if isinstance(image, list) else probs[0]


def text_similarity(search: str, item: str) -> float:
    from sentence_transformers import util

    model = _get_text_model()
    emb = model.encode([search, item], convert_to_tensor=True, normalize_embeddings=True)
    return float(util.cos_sim(emb[0], emb[1]).item())


def is_text_match(search: str, item: str, threshold: float = 0.75) -> bool:
    return text_similarity(search, item) > threshold


if __name__ == '__main__':
    score = text_similarity('jbl charge 5', 'jbl flip 5')
    print('similarity:', score)
    if score > 0.75:
        print('MATCH')
    else:
        print('NO_MATCH')

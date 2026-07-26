"""Generic CLIP defect prompts for the bad-photo detector (Phase 3).

Each defect has matched `good` and `bad` prompt lists. Prompts are deliberately
GENERIC: they never mention clothing, shoes, perfume, phones or any product type,
so the same prompts score every listing regardless of what it contains.

Scoring (see score_bad_photos.py) is:
    bad_similarity  = mean(image_embedding @ bad_prompt_embeddings.T)
    good_similarity = mean(image_embedding @ good_prompt_embeddings.T)
    defect_margin   = bad_similarity - good_similarity
No softmax, no sigmoid, no calibration, no type-specific prompt selection.
"""
from __future__ import annotations

DEFECT_PROMPTS: dict[str, dict[str, list[str]]] = {
    "blur": {
        "good": [
            "a sharply focused photograph",
            "a crisp photograph with clearly visible fine detail",
            "a photograph without motion blur",
        ],
        "bad": [
            "a visibly out of focus photograph",
            "a photograph with strong motion blur",
            "a blurry photograph with lost fine detail",
        ],
    },
    "exposure": {
        "good": [
            "a well exposed photograph with balanced brightness",
            "an evenly lit photograph with visible detail in shadows and highlights",
            "a photograph that is neither too dark nor too bright",
        ],
        "bad": [
            "a very dark underexposed photograph",
            "a very bright overexposed photograph with blown out highlights",
            "a photograph where detail is lost in deep shadow or harsh light",
        ],
    },
    "glare": {
        "good": [
            "a photograph with soft even lighting and no reflections",
            "a photograph free of harsh glare or hotspots",
            "a matte photograph without bright reflective spots",
        ],
        "bad": [
            "a photograph with strong glare and bright reflections",
            "a photograph with harsh specular hotspots from a flash",
            "a photograph washed out by a bright reflective spot",
        ],
    },
    "crop": {
        "good": [
            "a well framed photograph with the subject fully visible",
            "a photograph where the main subject is centered and complete",
            "a photograph with the subject entirely inside the frame",
        ],
        "bad": [
            "a badly cropped photograph with the subject cut off at the edge",
            "a photograph where the main subject is partly out of frame",
            "a poorly framed photograph missing part of the subject",
        ],
    },
    "resolution": {
        "good": [
            "a high resolution sharp photograph",
            "a clean photograph with no visible noise or pixelation",
            "a detailed photograph with smooth clear textures",
        ],
        "bad": [
            "a low resolution pixelated photograph",
            "a grainy noisy photograph",
            "a heavily compressed photograph with visible artifacts",
        ],
    },
    "clutter": {
        "good": [
            "a photograph with the item clearly visible on a clean background",
            "an uncluttered photograph focused on a single item",
            "a photograph where the subject stands out clearly",
        ],
        "bad": [
            "a cluttered photograph with many distracting objects",
            "a confusing photograph where the item is hard to see",
            "a busy photograph with a messy distracting background",
        ],
    },
    "tilt": {
        "good": [
            "a straight upright photograph with level horizon",
            "a photograph taken at a level straight angle",
            "a well aligned photograph that is not tilted",
        ],
        "bad": [
            "a strongly tilted crooked photograph",
            "a photograph rotated at an extreme slanted angle",
            "a skewed photograph with a heavily tilted horizon",
        ],
    },
}

# Defect key -> the blind-label defect tag it most corresponds to.
DEFECT_TO_TAG = {
    "blur": "blur",
    "exposure": "dark",          # exposure covers dark + overexposed
    "glare": "glare",
    "crop": "bad_crop",
    "resolution": "low_resolution",
    "clutter": "clutter",
    "tilt": "extreme_tilt",
}

DEFECTS = list(DEFECT_PROMPTS.keys())

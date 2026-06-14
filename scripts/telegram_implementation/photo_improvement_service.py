from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import time

import requests

PHOTO_EDIT_MODEL = "gpt-image-2"

# OpenAI org-level limit for gpt-image-2 input images is 5/min. Throttle our
# own calls to that rate so concurrent photo-improvement requests queue up
# instead of all hitting the API at once and getting 429s.
_RATE_LIMIT_MAX_CALLS = 5
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.IGNORECASE)

_rate_limit_lock = asyncio.Lock()
_call_timestamps: list[float] = []


async def _acquire_rate_limit_slot() -> None:
    """Block until issuing another gpt-image-2 call stays within the 5/min limit."""
    while True:
        async with _rate_limit_lock:
            now = time.monotonic()
            _call_timestamps[:] = [t for t in _call_timestamps if now - t < _RATE_LIMIT_WINDOW_SECONDS]
            if len(_call_timestamps) < _RATE_LIMIT_MAX_CALLS:
                _call_timestamps.append(now)
                return
            wait_time = _RATE_LIMIT_WINDOW_SECONDS - (now - _call_timestamps[0]) + 0.5
        await asyncio.sleep(wait_time)


_GEMINI_VISION_MODEL = "gemini-2.0-flash"
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

_IMPROVEMENT_PROMPT = """\
You are an expert e-commerce photo enhancement agent specialized in improving \
second-hand marketplace images for platforms like Vinted.

Your task is to automatically improve the uploaded item photo while preserving \
the real appearance of the product. The final image must look natural, realistic, \
trustworthy, and optimized for increasing click-through rate and sales conversion.

Improve the image using the following guidelines:
* Increase overall image quality and sharpness
* Correct blur and improve focus where possible
* Improve lighting and exposure
* Balance highlights and shadows
* Correct white balance and color accuracy
* Enhance contrast slightly for a cleaner professional look
* Remove digital noise and compression artifacts
* Improve texture visibility of fabrics/materials
* Make the item stand out clearly from the background
* Clean and simplify the background while keeping it realistic
* Remove distracting objects if possible
* Center and align the item properly
* Improve framing and crop intelligently for marketplace presentation
* Straighten the image if tilted
* Improve resolution/upscale naturally without creating fake details
* Preserve original item shape, proportions, logos, patterns, and colors
* Do NOT modify the actual product design or create misleading changes
* Keep the result realistic and compliant with marketplace authenticity expectations

Additional optimization goals:
* Make the image look similar to high-performing Vinted listings
* Prioritize clarity of the item over artistic effects
* Ensure the item is immediately recognizable in thumbnail view
* Optimize for mobile viewing
* Create a clean, minimal, bright, professional marketplace aesthetic

If the background is poor:
* Replace it with a clean neutral background (light gray, white, beige, or soft natural tones)
* Keep realistic shadows beneath the item
* Avoid artificial studio overprocessing

If the item is clothing:
* Preserve fabric texture and stitching details
* Improve wrinkle appearance naturally without making it fake
* Ensure colors remain faithful to the real product

If the item is shoes/accessories:
* Enhance material details and edges
* Improve separation from the background
* Highlight important product features naturally

The final output should look like a professional marketplace photo taken with \
a high-quality smartphone camera under ideal natural lighting.\
"""


def build_improvement_prompt(item_context: str | None = None) -> str:
    if item_context:
        return f"{_IMPROVEMENT_PROMPT}\n\nItem context: {item_context}"
    return _IMPROVEMENT_PROMPT


async def improve_photo(image_bytes: bytes, item_context: str | None = None, *, max_retries: int = 3) -> bytes:
    """
    Send image_bytes to gpt-image-2 for marketplace photo improvement.
    Returns improved image bytes. Raises on any API error.
    """
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("openai package is not installed. Run: pip install openai") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    client = openai.OpenAI(api_key=api_key)
    prompt = build_improvement_prompt(item_context)

    def _call_api():
        img_file = io.BytesIO(image_bytes)
        img_file.name = "photo.jpg"
        result = client.images.edit(
            model=PHOTO_EDIT_MODEL,
            image=img_file,
            prompt=prompt,
            n=1,
            size="1024x1024",
            quality="medium",
        )
        return base64.b64decode(result.data[0].b64_json)

    for attempt in range(max_retries + 1):
        await _acquire_rate_limit_slot()
        try:
            return await asyncio.to_thread(_call_api)
        except openai.RateLimitError as exc:
            if attempt >= max_retries:
                raise
            match = _RATE_LIMIT_RETRY_AFTER_RE.search(str(exc))
            wait_time = float(match.group(1)) + 0.5 if match else _RATE_LIMIT_WINDOW_SECONDS
            await asyncio.sleep(wait_time)

    raise RuntimeError("unreachable")  # pragma: no cover


async def extract_item_context_from_screenshot(image_bytes: bytes) -> str:
    """
    Use Gemini Vision to extract item title/info from a Vinted chat screenshot.
    Returns a short item description string, or empty string on failure.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ""

    b64 = base64.b64encode(image_bytes).decode()
    prompt = (
        "This is a screenshot from a Vinted marketplace conversation or listing page.\n"
        "Extract the item information visible in the screenshot.\n"
        'Return ONLY a short description in this format: "[Brand] [Item name], [Size if visible], [Condition if visible]"\n'
        'If you cannot identify the item, return "Unknown item".'
    )

    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 64},
    }

    url = f"{_GEMINI_API_BASE}/{_GEMINI_VISION_MODEL}:generateContent?key={api_key}"

    def _call_api():
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = parts[0].get("text", "").strip() if parts else ""
        return "" if text.lower() in ("", "unknown item") else text

    try:
        return await asyncio.to_thread(_call_api)
    except Exception:  # noqa: BLE001
        return ""

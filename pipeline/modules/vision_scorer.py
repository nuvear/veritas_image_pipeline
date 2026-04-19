"""
vision_scorer.py — Stage 4: Vision scoring of candidate images using Google Gemini 2.5 Flash.

Each candidate is scored against a slot description.
Returns a score dict with pass/fail and numeric scores.
"""

import os
import base64
import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

_client = None

def _get_client():
    global _client
    if _client is None:
        try:
            from google import genai
            _client = genai.Client(api_key=GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", ""))
        except Exception as e:
            logger.error(f"[VISION] Gemini client init failed: {e}")
    return _client


SCORE_PROMPT = """You are a food image quality assessor. Evaluate this image for use in a food database.

Food item: {food_name}
Slot type: {slot_type}
Slot description: {slot_description}

Respond with EXACTLY this format (no extra text):
REAL_PHOTO: YES or NO
FOOD_MATCH: YES or NO
CONFIDENCE: HIGH or MEDIUM or LOW
WATERMARK: YES or NO
FM_SCORE: integer 1-10
SF_SCORE: integer 1-10

Definitions:
- REAL_PHOTO: Is this a real photograph (not illustration, drawing, cartoon, or graphic)?
- FOOD_MATCH: Does this image show the correct food item ({food_name})?
- CONFIDENCE: How confident are you in FOOD_MATCH?
- WATERMARK: Does the image have a visible watermark, logo overlay, or text overlay?
- FM_SCORE: Food match quality score (1=wrong food, 10=perfect match)
- SF_SCORE: Slot fit score — how well does this image fit the slot description (1=poor fit, 10=perfect fit)
"""


def _encode_image(path: str) -> tuple:
    """Returns (base64_data, mime_type)."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    # Detect mime type from header bytes
    if data[:3] == b'\xff\xd8\xff':
        mime = "image/jpeg"
    elif data[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif data[:4] == b'RIFF' and data[8:12] == b'WEBP':
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    return b64, mime


def score_image(image_path: str, food_name: str, slot_type: str, slot_description: str) -> dict:
    """
    Score a single candidate image for a given slot using Gemini Vision.
    Returns dict with keys: real_photo, food_match, confidence, watermark, fm_score, sf_score, passes, raw
    """
    default_fail = {
        "real_photo": False, "food_match": False, "confidence": "LOW",
        "watermark": True, "fm_score": 0, "sf_score": 0, "passes": False, "raw": ""
    }

    client = _get_client()
    if client is None:
        return default_fail

    try:
        from google.genai import types as genai_types

        b64, mime = _encode_image(image_path)
        prompt = SCORE_PROMPT.format(
            food_name=food_name,
            slot_type=slot_type,
            slot_description=slot_description,
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(
                    data=base64.b64decode(b64),
                    mime_type=mime,
                ),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0,
                # No max_output_tokens — gemini-2.5-flash is a thinking model
                # and needs unrestricted output tokens to complete its response
            ),
        )

        raw = response.text.strip() if response.text else ""
        return _parse_score(raw, default_fail)

    except Exception as e:
        logger.warning(f"[VISION] Scoring failed for {image_path}: {e}")
        return default_fail


def _parse_score(raw: str, default: dict) -> dict:
    """Parse the structured response from the vision model."""
    result = dict(default)
    result["raw"] = raw

    def extract(pattern, text, default_val):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default_val

    real_photo_str = extract(r"REAL_PHOTO:\s*(YES|NO)", raw, "NO")
    food_match_str = extract(r"FOOD_MATCH:\s*(YES|NO)", raw, "NO")
    confidence_str = extract(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", raw, "LOW")
    watermark_str  = extract(r"WATERMARK:\s*(YES|NO)", raw, "YES")
    fm_str         = extract(r"FM_SCORE:\s*(\d+)", raw, "0")
    sf_str         = extract(r"SF_SCORE:\s*(\d+)", raw, "0")

    result["real_photo"]  = real_photo_str.upper() == "YES"
    result["food_match"]  = food_match_str.upper() == "YES"
    result["confidence"]  = confidence_str.upper()
    result["watermark"]   = watermark_str.upper() == "YES"
    result["fm_score"]    = min(10, max(0, int(fm_str)))
    result["sf_score"]    = min(10, max(0, int(sf_str)))

    # Pass criteria
    result["passes"] = (
        result["real_photo"]
        and result["food_match"]
        and result["confidence"] in ("HIGH", "MEDIUM")
        and not result["watermark"]
        and result["fm_score"] >= 6
        and result["sf_score"] >= 5
    )

    return result


def score_candidates(candidates: list, food_name: str, slot_type: str, slot_description: str,
                     max_to_score: int = 15) -> list:
    """
    Score a list of candidate image paths for a given slot.
    Stops early once 1 passing image is found (for efficiency).
    Returns list of (path, score_dict) sorted by fm_score desc.
    """
    results = []
    passing_found = 0

    for path in candidates[:max_to_score]:
        score = score_image(path, food_name, slot_type, slot_description)
        results.append((path, score))
        logger.debug(
            f"[VISION] [{slot_type}] fm={score['fm_score']} sf={score['sf_score']} "
            f"passes={score['passes']} | {score['raw'][:80]}"
        )
        if score["passes"]:
            passing_found += 1
            if passing_found >= 1:
                break  # Early exit once we have one passer

    # Sort by fm_score descending, then sf_score
    results.sort(key=lambda x: (x[1]["fm_score"], x[1]["sf_score"]), reverse=True)
    return results

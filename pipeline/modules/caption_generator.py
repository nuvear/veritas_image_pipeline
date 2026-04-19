"""
caption_generator.py — Generates multilingual captions for food images.

Per PROJECT_SPEC.md Section 5:
  - Model: gemini-2.5-flash (thinking model — no max_output_tokens)
  - Languages: English (en), Simplified Chinese (zh), Malay (ms), Tamil (ta)
  - Style: Detailed descriptive, 2-3 sentences
  - Content: food name, key ingredients, cuisine type, nutritional context
"""

import logging
import os
import json
import re

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.5-flash"

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY", "")
        _client = genai.Client(api_key=api_key)
    return _client


CAPTION_PROMPT = """You are a multilingual food description writer for a Southeast Asian nutrition app.

Generate detailed, descriptive captions for the following food item. Each caption should be 2-3 sentences covering:
1. What the food is and how it looks in the image
2. Key ingredients or components
3. Cuisine type, cultural context, or nutritional relevance

Food details:
- Name: {food_name}
- Aliases: {aliases}
- Food type: {food_type}
- Slot: {slot} ({slot_desc})
- Cuisine: Southeast Asian (primarily Singaporean/Malaysian)

Respond with EXACTLY this JSON format (no extra text, no markdown):
{{
  "en": "English caption here (2-3 sentences)",
  "zh": "Chinese caption here (2-3 sentences, Simplified Chinese)",
  "ms": "Malay caption here (2-3 sentences)",
  "ta": "Tamil caption here (2-3 sentences)"
}}"""


def generate_captions(record: dict, food_type: str, slot: str,
                      slot_desc: str) -> dict:
    """
    Generate multilingual captions for a food item.

    Returns dict with keys: en, zh, ms, ta
    Falls back to English-only if Gemini call fails.
    """
    food_name = record.get("food_name", "Unknown Food")
    aliases = record.get("aliases") or []
    if isinstance(aliases, dict):
        aliases = list(aliases.values())
    aliases_str = ", ".join(str(a) for a in aliases if a) or "none"

    default = {
        "en": f"A {slot} photo of {food_name}, a Southeast Asian food item.",
        "zh": f"{food_name}的{slot}照片，一种东南亚食品。",
        "ms": f"Foto {slot} {food_name}, makanan Asia Tenggara.",
        "ta": f"{food_name} இன் {slot} புகைப்படம், ஒரு தென்கிழக்கு ஆசிய உணவு.",
    }

    try:
        client = _get_client()
        from google.genai import types as genai_types

        prompt = CAPTION_PROMPT.format(
            food_name=food_name,
            aliases=aliases_str,
            food_type=food_type,
            slot=slot,
            slot_desc=slot_desc,
        )

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=genai_types.GenerateContentConfig(temperature=0.3),
        )

        raw = response.text.strip() if response.text else ""

        # Parse JSON response
        # Strip markdown code blocks if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        captions = json.loads(raw)

        # Validate all 4 languages present
        for lang in ["en", "zh", "ms", "ta"]:
            if lang not in captions or not captions[lang]:
                captions[lang] = default[lang]

        logger.debug(f"[CAPTION] Generated captions for {food_name} [{slot}]")
        return captions

    except Exception as e:
        logger.warning(f"[CAPTION] Failed for {food_name} [{slot}]: {e}")
        return default

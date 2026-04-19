"""
clip_scorer.py — CLIP-based image-text similarity scoring using HuggingFace.

Uses openai/clip-vit-base-patch32 to compute cosine similarity between
an image and a text description. This provides a fast, deterministic,
model-free (no API call) pre-filter before the more expensive Gemini scoring.

CLIP score interpretation:
  >= 0.28  → Strong match (pass pre-filter)
  0.22–0.27 → Moderate match (borderline)
  < 0.22   → Weak match (likely wrong food or irrelevant image)

The model is loaded once and cached for the lifetime of the process.
"""

import logging
import os
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (singleton, loaded once per process)
# ---------------------------------------------------------------------------

_model = None
_processor = None
_device = None

CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# Thresholds
CLIP_PASS_THRESHOLD = 0.30    # Minimum CLIP score to pass pre-filter (per PROJECT_SPEC.md)
CLIP_STRONG_THRESHOLD = 0.34  # Strong match threshold


def _load_model():
    global _model, _processor, _device
    if _model is not None:
        return _model, _processor, _device

    logger.info("[CLIP] Loading CLIP model (first call, may take ~30s)...")
    try:
        from transformers import CLIPProcessor, CLIPModel

        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[CLIP] Using device: {_device}")

        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(_device)
        _model.eval()

        logger.info("[CLIP] Model loaded successfully")
    except Exception as e:
        logger.error(f"[CLIP] Failed to load model: {e}")
        _model = None
        _processor = None
        _device = "cpu"

    return _model, _processor, _device


# ---------------------------------------------------------------------------
# CLIP scoring
# ---------------------------------------------------------------------------

def score_clip(image_path: str, text_queries: list) -> dict:
    """
    Compute CLIP similarity between an image and a list of text queries.

    Args:
        image_path: Path to the image file
        text_queries: List of text strings to compare against the image

    Returns:
        dict with keys:
          - scores: list of float similarity scores (0-1) per query
          - max_score: float, highest similarity score
          - mean_score: float, mean similarity score
          - clip_pass: bool, whether max_score >= CLIP_PASS_THRESHOLD
          - clip_strong: bool, whether max_score >= CLIP_STRONG_THRESHOLD
          - best_query: str, the query with highest similarity
    """
    default = {
        "scores": [],
        "max_score": 0.0,
        "mean_score": 0.0,
        "clip_pass": False,
        "clip_strong": False,
        "best_query": "",
    }

    model, processor, device = _load_model()
    if model is None:
        return default

    try:
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Truncate text queries to CLIP's 77-token limit
        truncated = [q[:200] for q in text_queries]

        # Process inputs
        inputs = processor(
            text=truncated,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits_per_image shape: [1, num_texts]
            logits = outputs.logits_per_image  # raw logits
            probs = logits.softmax(dim=-1).squeeze().cpu().tolist()

            # Also get raw cosine similarity (more interpretable)
            image_embeds = outputs.image_embeds  # [1, 512]
            text_embeds = outputs.text_embeds    # [N, 512]

            # Normalize
            image_norm = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_norm = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

            # Cosine similarity
            cosine_sims = (image_norm @ text_norm.T).squeeze().cpu().tolist()

        # Handle single query case
        if isinstance(cosine_sims, float):
            cosine_sims = [cosine_sims]

        max_score = max(cosine_sims)
        mean_score = sum(cosine_sims) / len(cosine_sims)
        best_idx = cosine_sims.index(max_score)
        best_query = text_queries[best_idx] if text_queries else ""

        return {
            "scores": [round(s, 4) for s in cosine_sims],
            "max_score": round(max_score, 4),
            "mean_score": round(mean_score, 4),
            "clip_pass": max_score >= CLIP_PASS_THRESHOLD,
            "clip_strong": max_score >= CLIP_STRONG_THRESHOLD,
            "best_query": best_query,
        }

    except Exception as e:
        logger.warning(f"[CLIP] Scoring failed for {image_path}: {e}")
        return default


def build_clip_queries(food_name: str, food_type: str, slot_type: str,
                       aliases: list = None) -> list:
    """
    Build CLIP text queries for a given food and slot.
    Returns a list of 3-5 text strings for similarity comparison.
    """
    alt = aliases[0] if aliases else ""

    base_queries = [f"a photo of {food_name}"]

    if food_type == "beverage":
        slot_queries = {
            "hero": [
                f"a photo of {food_name} drink in a glass",
                f"{food_name} beverage served",
            ],
            "macro": [
                f"close up of {food_name} drink texture",
                f"macro photo of {food_name}",
            ],
            "in_the_wild": [
                f"{food_name} served at a cafe or hawker stall",
                f"person drinking {food_name}",
            ],
        }
    elif food_type == "raw_ingredient":
        slot_queries = {
            "hero": [
                f"fresh {food_name} ingredient on white background",
                f"raw {food_name} food photography",
            ],
            "macro": [
                f"close up texture of {food_name}",
                f"macro photo of fresh {food_name}",
            ],
            "in_the_wild": [
                f"{food_name} at a market or grocery store",
                f"{food_name} being prepared in a kitchen",
            ],
        }
    elif food_type == "packaged_product":
        slot_queries = {
            "hero": [
                f"{food_name} product packaging",
                f"{food_name} food package on white background",
            ],
            "macro": [
                f"close up of {food_name} food texture",
                f"detail shot of {food_name}",
            ],
            "in_the_wild": [
                f"{food_name} on supermarket shelf",
                f"{food_name} in kitchen pantry",
            ],
        }
    else:  # cooked_dish
        slot_queries = {
            "hero": [
                f"a photo of {food_name} dish plated",
                f"{food_name} food photography overhead",
            ],
            "macro": [
                f"close up of {food_name} food texture",
                f"macro photo of {food_name} dish",
            ],
            "in_the_wild": [
                f"{food_name} served at a hawker stall or restaurant",
                f"{food_name} street food",
            ],
        }

    queries = base_queries + slot_queries.get(slot_type, [])
    if alt:
        queries.append(f"a photo of {alt}")

    return queries[:5]  # CLIP handles up to 5 efficiently


def score_candidates_clip(candidates: list, food_name: str, food_type: str,
                           slot_type: str, aliases: list = None,
                           min_clip_score: float = CLIP_PASS_THRESHOLD) -> list:
    """
    Pre-filter and rank candidates by CLIP score.

    Args:
        candidates: list of image file paths
        food_name: food item name
        food_type: classified food type
        slot_type: hero / macro / in_the_wild
        aliases: optional list of alternative names
        min_clip_score: minimum CLIP score to keep (pre-filter)

    Returns:
        list of (path, clip_result) sorted by max_score descending,
        filtered to only include images that pass the CLIP threshold.
        If no images pass, returns all sorted by score (no filtering).
    """
    queries = build_clip_queries(food_name, food_type, slot_type, aliases)
    results = []

    for path in candidates:
        clip_result = score_clip(path, queries)
        results.append((path, clip_result))
        logger.debug(
            f"[CLIP] [{slot_type}] {Path(path).name} "
            f"max={clip_result['max_score']:.3f} "
            f"pass={clip_result['clip_pass']} "
            f"strong={clip_result['clip_strong']}"
        )

    # Sort by max_score descending
    results.sort(key=lambda x: x[1]["max_score"], reverse=True)

    # Filter to passing candidates
    passing = [(p, r) for p, r in results if r["clip_pass"]]

    if passing:
        logger.info(
            f"[CLIP] [{slot_type}] {len(passing)}/{len(results)} passed "
            f"(threshold={min_clip_score:.2f})"
        )
        return passing
    else:
        # No candidates passed — return all sorted, let Gemini decide
        logger.warning(
            f"[CLIP] [{slot_type}] No candidates passed threshold "
            f"(best={results[0][1]['max_score']:.3f} if results else 0). "
            f"Passing all to Gemini."
        )
        return results

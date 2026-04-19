"""
slot_assigner.py — Stage 5: Slot assignment, pad-to-square, resize to 896×896 JPEG Q95.

LLM Training Dataset specs (Gemma 4 E4B / PROJECT_SPEC.md):
  - Output: 896×896 JPEG quality 95
  - Aspect ratio: preserve original + pad to square (white background)
  - Layout: FLAT — FD000031_hero.jpg, FD000031_macro.jpg, FD000031_in_the_wild.jpg
  - Quality gate: FM >= 7 AND SF >= 7 AND passes=True
  - No fallback to low-quality images — better empty slot than bad image
"""

import logging
from pathlib import Path
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# Output specs (per PROJECT_SPEC.md)
OUTPUT_SIZE = (896, 896)
JPEG_QUALITY = 95
PAD_COLOR = (255, 255, 255)  # White padding

# Quality gate (per PROJECT_SPEC.md Section 4.2)
MIN_FM_SCORE = 7
MIN_SF_SCORE = 7

# Flat filename convention
SLOT_SUFFIX = {
    "hero": "hero",
    "macro": "macro",
    "in_the_wild": "in_the_wild",
}


def pad_to_square(image: Image.Image) -> Image.Image:
    """
    Resize image to fit within OUTPUT_SIZE while preserving aspect ratio,
    then pad to exact square with white background.
    """
    img = image.copy()
    img.thumbnail(OUTPUT_SIZE, Image.LANCZOS)

    canvas = Image.new("RGB", OUTPUT_SIZE, PAD_COLOR)
    offset_x = (OUTPUT_SIZE[0] - img.width) // 2
    offset_y = (OUTPUT_SIZE[1] - img.height) // 2
    canvas.paste(img, (offset_x, offset_y))
    return canvas


def process_image(src_path: str, dst_path: str) -> bool:
    """
    Load, apply EXIF orientation, pad-to-square, resize, save JPEG Q95.
    Returns True on success.
    """
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = ImageOps.exif_transpose(img)
            processed = pad_to_square(img)
            processed.save(dst_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception as e:
        logger.warning(f"[SLOT] Failed to process {src_path}: {e}")
        return False


def assign_slots(scored_results: dict, output_dir: Path, fd_id: str) -> dict:
    """
    Pick the best qualifying candidate for each slot and save to flat output dir.

    Quality gate: passes=True AND fm_score >= MIN_FM_SCORE AND sf_score >= MIN_SF_SCORE
    If no candidate passes, slot is left empty (None) — no fallback to bad images.

    Args:
        scored_results: dict of slot → list of (path, score_dict), sorted best-first
        output_dir: flat output directory
        fd_id: food ID string (e.g. "FD000031")

    Returns:
        dict of slot → assignment info dict, or None if no qualifying candidate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment = {}

    for slot, candidates in scored_results.items():
        suffix = SLOT_SUFFIX.get(slot, slot)
        filename = f"{fd_id}_{suffix}.jpg"
        dst_path = output_dir / filename
        assigned = None

        for path, score in candidates:
            qualifies = (
                score.get("passes", False)
                and score.get("fm_score", 0) >= MIN_FM_SCORE
                and score.get("sf_score", 0) >= MIN_SF_SCORE
            )
            if qualifies:
                if process_image(path, str(dst_path)):
                    assigned = {
                        "file": str(dst_path),
                        "filename": filename,
                        "fm_score": score.get("fm_score", 0),
                        "sf_score": score.get("sf_score", 0),
                        "clip_max": score.get("clip_max", 0.0),
                        "passes": True,
                    }
                    logger.info(
                        f"[SLOT] ✓ {slot} → {filename} "
                        f"(fm={assigned['fm_score']} sf={assigned['sf_score']} "
                        f"clip={assigned['clip_max']:.3f})"
                    )
                    break

        if assigned is None:
            logger.warning(f"[SLOT] ✗ {slot} → no qualifying candidate for {fd_id} "
                           f"(gate: FM≥{MIN_FM_SCORE}, SF≥{MIN_SF_SCORE})")

        assignment[slot] = assigned

    return assignment


def count_published(assignment: dict) -> int:
    """Count how many slots have a successfully assigned image."""
    return sum(1 for v in assignment.values() if v is not None)

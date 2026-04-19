"""
worker.py — Processes a range of FD records from the input JSON file.

Read PROJECT_SPEC.md before modifying this file.

Dual-scoring pipeline (per PROJECT_SPEC.md Section 7):
  Stage 1: Classify food type
  Stage 2: Build queries + slot descriptions
  Stage 3: SerpAPI image search + download
  Stage 4a: CLIP pre-filter (threshold 0.30)
  Stage 4b: Gemini FM/SF scoring (FM>=7, SF>=7 gate)
  Stage 5: Slot assignment + pad-to-square + resize 896x896 JPEG Q95 (flat layout)
  Stage 6: Multilingual caption generation (EN/ZH/MS/TA)

Output per food item (flat directory):
  {fd_id}_hero.jpg
  {fd_id}_macro.jpg
  {fd_id}_in_the_wild.jpg
  + one line in annotations.jsonl
  + one line in manifest.jsonl
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.classifier import classify_food_type, build_queries, build_slot_descriptions
from modules.image_search import search_images, download_images
from modules.clip_scorer import score_candidates_clip
from modules.vision_scorer import score_candidates
from modules.slot_assigner import assign_slots, count_published
from modules.caption_generator import generate_captions


def setup_logging(worker_id: str, log_file: str, level: str = "INFO"):
    fmt = f"%(asctime)s [{worker_id}] %(message)s"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_checkpoint(checkpoint_path: str) -> set:
    done = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("status") in ("complete", "partial", "no_images"):
                            done.add(rec["fd_id"])
                    except Exception:
                        pass
    return done


def append_checkpoint(checkpoint_path: str, record: dict):
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def append_annotation(annotations_path: str, record: dict):
    Path(annotations_path).parent.mkdir(parents=True, exist_ok=True)
    with open(annotations_path, "a") as f:
        f.write(json.dumps(record) + "\n")


MAX_ATTEMPTS = 6

RETRY_QUERY_SUFFIXES = [
    "food photography",
    "homemade recipe",
    "traditional dish",
    "restaurant food",
    "cooked meal",
]


def process_food(record: dict, output_dir: Path, temp_dir: Path,
                 worker_id: str, annotations_path: str) -> dict:
    fd_id = record.get("fd_id") or record.get("food_id", "UNKNOWN")
    food_name = record.get("food_name", "Unknown Food")
    aliases = record.get("aliases") or []
    if isinstance(aliases, dict):
        aliases = list(aliases.values())
    aliases = [str(a) for a in aliases if a][:4]

    logger = logging.getLogger()
    logger.info(f"[{fd_id}] Processing: {food_name}")
    t0 = time.time()

    food_type = classify_food_type(record)
    logger.info(f"[{fd_id}] food_type={food_type}")

    queries = build_queries(record, food_type)
    slot_descs = build_slot_descriptions(record, food_type)

    food_temp_dir = temp_dir / fd_id
    food_temp_dir.mkdir(parents=True, exist_ok=True)

    slots = ["hero", "macro", "in_the_wild"]
    scored_results = {slot: [] for slot in slots}

    for slot in slots:
        slot_queries = queries.get(slot, [])
        slot_desc = slot_descs.get(slot, f"A photo of {food_name}")
        slot_temp = food_temp_dir / slot
        slot_temp.mkdir(parents=True, exist_ok=True)

        found_passing = False
        attempt = 0

        while not found_passing and attempt < MAX_ATTEMPTS:
            if attempt < len(slot_queries):
                query = slot_queries[attempt]
            else:
                retry_idx = attempt - len(slot_queries)
                if retry_idx < len(RETRY_QUERY_SUFFIXES):
                    query = f"{food_name} {RETRY_QUERY_SUFFIXES[retry_idx]}"
                else:
                    break

            attempt += 1
            logger.info(f"[{fd_id}] [{slot}] Attempt {attempt}/{MAX_ATTEMPTS}: {query}")

            urls = search_images(query, num=15)
            if not urls:
                continue

            attempt_temp = slot_temp / f"attempt_{attempt}"
            candidates = download_images(urls, attempt_temp, max_count=10)
            if not candidates:
                continue

            # Stage 4a: CLIP pre-filter
            clip_ranked = score_candidates_clip(
                candidates, food_name=food_name, food_type=food_type,
                slot_type=slot, aliases=aliases,
            )

            top_candidates = [path for path, _ in clip_ranked[:5]]
            clip_scores_map = {path: clip_r for path, clip_r in clip_ranked}

            # Stage 4b: Gemini FM/SF scoring
            gemini_results = score_candidates(
                top_candidates, food_name, slot, slot_desc, max_to_score=5
            )

            merged = []
            for path, gemini_score in gemini_results:
                clip_r = clip_scores_map.get(path, {})
                combined = dict(gemini_score)
                combined["clip_max"] = clip_r.get("max_score", 0.0)
                combined["clip_pass"] = clip_r.get("clip_pass", False)
                combined["clip_strong"] = clip_r.get("clip_strong", False)
                merged.append((path, combined))

            for path, sc in merged:
                logger.info(
                    f"[{fd_id}] [{slot}] clip={sc['clip_max']:.3f} "
                    f"fm={sc['fm_score']} sf={sc['sf_score']} passes={sc['passes']}"
                )

            scored_results[slot].extend(merged)

            for _, sc in merged:
                if (sc["passes"] and sc.get("fm_score", 0) >= 7
                        and sc.get("sf_score", 0) >= 7):
                    found_passing = True
                    logger.info(f"[{fd_id}] [{slot}] ✓ Found qualifying candidate (attempt {attempt})")
                    break

        if not found_passing:
            logger.warning(f"[{fd_id}] [{slot}] No qualifying candidate after {attempt} attempts")

    # Sort by: qualifies first, then fm_score, then clip_max
    for slot in slots:
        scored_results[slot].sort(
            key=lambda x: (
                x[1].get("passes", False) and x[1].get("fm_score", 0) >= 7 and x[1].get("sf_score", 0) >= 7,
                x[1].get("fm_score", 0),
                x[1].get("clip_max", 0),
            ),
            reverse=True,
        )

    # Stage 5: Assign slots (flat layout, pad-to-square, JPEG Q95)
    assignment = assign_slots(scored_results, output_dir, fd_id)
    published = count_published(assignment)

    # Stage 6: Generate multilingual captions for each assigned slot
    for slot, info in assignment.items():
        if info is not None:
            captions = generate_captions(record, food_type, slot, slot_descs.get(slot, ""))
            info["captions"] = captions

            # Write to annotations.jsonl
            annotation = {
                "image": info["filename"],
                "food_id": fd_id,
                "food_name": food_name,
                "food_type": food_type,
                "slot": slot,
                "captions": captions,
                "fm_score": info["fm_score"],
                "sf_score": info["sf_score"],
                "clip_score": round(info["clip_max"], 4),
                "publish": True,
            }
            append_annotation(annotations_path, annotation)

    # Cleanup temp
    try:
        shutil.rmtree(str(food_temp_dir), ignore_errors=True)
    except Exception:
        pass

    elapsed = time.time() - t0
    status = "complete" if published == 3 else ("partial" if published > 0 else "no_images")

    fm_scores = [v["fm_score"] for v in assignment.values() if v]
    sf_scores = [v["sf_score"] for v in assignment.values() if v]
    clip_scores = [v["clip_max"] for v in assignment.values() if v]

    avg_fm = round(sum(fm_scores) / len(fm_scores), 2) if fm_scores else 0
    avg_sf = round(sum(sf_scores) / len(sf_scores), 2) if sf_scores else 0
    avg_clip = round(sum(clip_scores) / len(clip_scores), 3) if clip_scores else 0

    result = {
        "fd_id": fd_id,
        "food_name": food_name,
        "food_type": food_type,
        "status": status,
        "publish": published,
        "avg_fm": avg_fm,
        "avg_sf": avg_sf,
        "avg_clip": avg_clip,
        "elapsed_s": round(elapsed, 1),
        "slots_complete": [s for s, v in assignment.items() if v],
        "slots_missing": [s for s, v in assignment.items() if not v],
    }

    logger.info(
        f"✓ {fd_id} done — publish={published}/3 "
        f"avg_fm={avg_fm} avg_sf={avg_sf} avg_clip={avg_clip} [{elapsed:.0f}s]"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Food Image Pipeline Worker")
    parser.add_argument("--input", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--annotations", required=True, help="Path to annotations.jsonl")
    parser.add_argument("--temp", default="/tmp/pipeline_temp")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--worker-id", default="W1")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.worker_id, args.log_file, args.log_level)
    logger = logging.getLogger()
    logger.info(f"[{args.worker_id}] Starting — range {args.start} to {args.end}")

    with open(args.input) as f:
        all_records = json.load(f)

    records = [
        r for r in all_records
        if args.start <= (r.get("fd_id") or "") <= args.end
    ]
    logger.info(f"[{args.worker_id}] {len(records)} records in range {args.start}–{args.end}")

    done_ids = load_checkpoint(args.checkpoint)
    logger.info(f"[{args.worker_id}] {len(done_ids)} already completed (skipping)")

    output_dir = Path(args.output)
    temp_dir = Path(args.temp) / args.worker_id
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for record in records:
        fd_id = record.get("fd_id") or record.get("food_id", "")
        if fd_id in done_ids:
            skipped += 1
            continue

        try:
            result = process_food(record, output_dir, temp_dir,
                                  args.worker_id, args.annotations)
            append_checkpoint(args.checkpoint, result)
            processed += 1
        except Exception as e:
            logger.error(f"[{args.worker_id}] FAILED {fd_id}: {e}", exc_info=True)
            append_checkpoint(args.checkpoint, {
                "fd_id": fd_id,
                "food_name": record.get("food_name", ""),
                "status": "error",
                "publish": 0,
                "error": str(e),
            })

    logger.info(f"[{args.worker_id}] DONE — processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()

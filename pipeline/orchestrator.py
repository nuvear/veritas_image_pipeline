"""
orchestrator.py — Manages batch-of-50 pipeline execution.

Per PROJECT_SPEC.md Operating Principles:
  1. Quality first (FM/SF >= 7, CLIP >= 0.30)
  2. Speed (5 parallel workers per batch)
  3. Save before proceeding:
     - Save images to Mac after each batch
     - Update manifest.jsonl and annotations.jsonl
     - Commit and push to GitHub
     - Update PROJECT_SPEC.md progress tracker
     - Then start next batch

Batch size: 50 records
Workers: 5 parallel (10 records each)
"""

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

REPO_DIR = Path(__file__).parent.parent
OUTPUT_DIR = REPO_DIR / "output" / "images"
CHECKPOINT_DIR = REPO_DIR / "output" / "checkpoints"
ANNOTATIONS_FILE = REPO_DIR / "output" / "annotations.jsonl"
MANIFEST_FILE = REPO_DIR / "output" / "manifest.jsonl"
SPEC_FILE = REPO_DIR / "PROJECT_SPEC.md"
MAC_OUTPUT_DIR = "/mnt/desktop/db_veritas1/food_images_gemma4_training"

BATCH_SIZE = 50
NUM_WORKERS = 5
WORKER_SCRIPT = Path(__file__).parent / "worker.py"


def setup_logging(log_file: str):
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [ORCH] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_completed_ids() -> set:
    """Load FD IDs already completed from manifest.jsonl."""
    done = set()
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        if rec.get("status") in ("complete", "partial"):
                            done.add(rec["fd_id"])
                    except Exception:
                        pass
    return done


def collect_batch_results(batch_dir: Path) -> list:
    """Read all worker checkpoint files for a batch."""
    records = []
    seen = set()
    for cp_file in sorted(batch_dir.glob("W*.jsonl")):
        with open(cp_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        fd_id = rec.get("fd_id")
                        if fd_id and fd_id not in seen:
                            records.append(rec)
                            seen.add(fd_id)
                    except Exception:
                        pass
    return records


def merge_annotations(batch_label: str):
    """Merge batch-specific annotations into main annotations.jsonl."""
    batch_ann = REPO_DIR / "output" / f"annotations_{batch_label}.jsonl"
    if batch_ann.exists():
        ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(batch_ann) as src, open(ANNOTATIONS_FILE, "a") as dst:
            for line in src:
                dst.write(line)
        logging.getLogger().info(f"Merged annotations for {batch_label}")


def update_manifest(batch_records: list):
    """Append batch records to manifest.jsonl."""
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "a") as f:
        for rec in batch_records:
            f.write(json.dumps(rec) + "\n")
    logging.getLogger().info(f"Updated manifest with {len(batch_records)} records")


def save_to_mac():
    """Rsync output images to Mac."""
    try:
        os.makedirs(MAC_OUTPUT_DIR, exist_ok=True)
        result = subprocess.run(
            ["rsync", "-av", "--ignore-existing",
             str(OUTPUT_DIR) + "/",
             MAC_OUTPUT_DIR + "/"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            logging.getLogger().info(f"Images synced to Mac: {MAC_OUTPUT_DIR}")
        else:
            logging.getLogger().warning(f"rsync warning: {result.stderr[:300]}")
    except Exception as e:
        logging.getLogger().warning(f"Mac sync failed: {e}")


def push_to_github(batch_label: str, batch_records: list):
    """Commit and push progress to GitHub."""
    logger = logging.getLogger()
    try:
        pub3 = sum(1 for r in batch_records if r.get("publish") == 3)
        total = len(batch_records)
        images = sum(r.get("publish", 0) for r in batch_records)
        msg = f"Batch {batch_label}: {total} items, {pub3}/{total} pub=3, {images} images"

        subprocess.run(
            ["git", "-C", str(REPO_DIR), "add",
             "output/manifest.jsonl", "output/annotations.jsonl", "PROJECT_SPEC.md"],
            check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "commit", "-m", msg],
            check=True, capture_output=True
        )
        subprocess.run(
            ["git", "-C", str(REPO_DIR), "push"],
            check=True, capture_output=True
        )
        logger.info(f"Pushed to GitHub: {msg}")
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode()[:300] if e.stderr else str(e)
        logger.warning(f"GitHub push failed: {stderr}")


def update_spec_progress(batch_label: str, fd_start: str, fd_end: str,
                         batch_records: list):
    """Update the progress tracker table in PROJECT_SPEC.md."""
    total = len(batch_records)
    pub3 = sum(1 for r in batch_records if r.get("publish") == 3)
    images = sum(r.get("publish", 0) for r in batch_records)
    today = datetime.now().strftime("%Y-%m-%d")

    old_row = f"| {batch_label} | {fd_start}–{fd_end} | 50 | PENDING | — | — | — |"
    new_row = (f"| {batch_label} | {fd_start}–{fd_end} | {total} | "
               f"DONE ({pub3}/{total} pub=3) | {images} | ✓ {today} | ✓ {today} |")

    try:
        spec_text = SPEC_FILE.read_text()
        if old_row in spec_text:
            SPEC_FILE.write_text(spec_text.replace(old_row, new_row))
            logging.getLogger().info(f"Updated PROJECT_SPEC.md for {batch_label}")
    except Exception as e:
        logging.getLogger().warning(f"Failed to update spec: {e}")


def run_batch(batch_num: int, batch_records_input: list, input_file: str) -> list:
    """Run one batch of 50 records with 5 parallel workers."""
    logger = logging.getLogger()
    batch_label = f"B{batch_num:02d}"
    fd_ids = [r.get("fd_id") or r.get("food_id", "") for r in batch_records_input]
    fd_start = fd_ids[0]
    fd_end = fd_ids[-1]

    logger.info("\n" + "=" * 60)
    logger.info(f"Starting {batch_label}: {fd_start}–{fd_end} ({len(fd_ids)} records)")
    logger.info("=" * 60)

    batch_dir = CHECKPOINT_DIR / batch_label
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Write batch input file
    batch_input = batch_dir / "input.json"
    with open(batch_input, "w") as f:
        json.dump(batch_records_input, f)

    # Split into 5 worker chunks
    chunk_size = math.ceil(len(fd_ids) / NUM_WORKERS)
    chunks = []
    for i in range(0, len(fd_ids), chunk_size):
        chunk_ids = fd_ids[i:i + chunk_size]
        if chunk_ids:
            chunks.append((chunk_ids[0], chunk_ids[-1]))

    # Launch workers
    procs = []
    for i, (w_start, w_end) in enumerate(chunks):
        wid = f"W{i+1}"
        checkpoint_file = batch_dir / f"{wid}.jsonl"
        log_file = batch_dir / f"{wid}.log"
        annotations_file = REPO_DIR / "output" / f"annotations_{batch_label}.jsonl"

        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--input", str(batch_input),
            "--start", w_start,
            "--end", w_end,
            "--output", str(OUTPUT_DIR),
            "--annotations", str(annotations_file),
            "--temp", f"/tmp/pipeline_{batch_label}_{wid}",
            "--checkpoint", str(checkpoint_file),
            "--worker-id", f"{batch_label}-{wid}",
            "--log-file", str(log_file),
        ]
        env = os.environ.copy()
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)
        logger.info(f"  Launched {batch_label}-{wid}: {w_start}–{w_end} (PID {proc.pid})")
        time.sleep(2)

    # Wait for all workers
    for proc in procs:
        proc.wait()

    logger.info(f"{batch_label}: All workers complete")

    # Collect results
    batch_results = collect_batch_results(batch_dir)
    pub3 = sum(1 for r in batch_results if r.get("publish") == 3)
    images = sum(r.get("publish", 0) for r in batch_results)
    logger.info(f"{batch_label}: {len(batch_results)} items | pub=3: {pub3} | images: {images}")

    # Post-batch: merge → manifest → Mac → GitHub → spec
    merge_annotations(batch_label)
    update_manifest(batch_results)
    save_to_mac()
    push_to_github(batch_label, batch_results)
    update_spec_progress(batch_label, fd_start, fd_end, batch_results)

    logger.info(f"{batch_label}: COMPLETE ✓\n")
    return batch_results


def main():
    parser = argparse.ArgumentParser(description="Food Image Pipeline Orchestrator")
    parser.add_argument("--input", required=True, help="Input JSON file with FD records")
    parser.add_argument("--start-batch", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=9999)
    args = parser.parse_args()

    log_file = str(REPO_DIR / "output" / "orchestrator.log")
    setup_logging(log_file)
    logger = logging.getLogger()

    logger.info("=" * 60)
    logger.info("Veritas Image Pipeline — Orchestrator v2")
    logger.info("Principles: Quality > Speed > Save-first")
    logger.info(f"Batch: {BATCH_SIZE} records | Workers: {NUM_WORKERS}")
    logger.info("=" * 60)

    with open(args.input) as f:
        all_records = json.load(f)

    all_records_sorted = sorted(
        all_records,
        key=lambda r: r.get("fd_id") or r.get("food_id", "")
    )

    completed_ids = load_completed_ids()
    remaining = [r for r in all_records_sorted
                 if (r.get("fd_id") or r.get("food_id", "")) not in completed_ids]

    logger.info(f"Total: {len(all_records_sorted)} | Completed: {len(completed_ids)} | Remaining: {len(remaining)}")

    if not remaining:
        logger.info("All records already processed.")
        return

    batches = [remaining[i:i + BATCH_SIZE] for i in range(0, len(remaining), BATCH_SIZE)]
    logger.info(f"Batches to run: {len(batches)}")

    total_processed = 0
    for idx, batch in enumerate(batches):
        batch_num = args.start_batch + idx
        if idx >= args.max_batches:
            logger.info(f"Reached max_batches={args.max_batches}. Stopping.")
            break
        results = run_batch(batch_num, batch, args.input)
        total_processed += len(results)
        logger.info(f"Cumulative: {total_processed}/{len(remaining)} processed")

    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — {total_processed} records processed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

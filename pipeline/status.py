"""
status.py — Quick pipeline status report.

Usage:
    python3 status.py --checkpoint-dir OUTPUT/checkpoints
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pipeline Status Report")
    parser.add_argument("--checkpoint-dir", required=True)
    args = parser.parse_args()

    cp_dir = Path(args.checkpoint_dir)
    if not cp_dir.exists():
        print(f"Checkpoint dir not found: {cp_dir}")
        return

    total_items = 0
    total_pub3 = 0
    total_pub0 = 0
    total_images = 0
    total_errors = 0
    needs_manual = []

    print(f"\n{'Batch':<35} {'Items':>6} {'pub=3':>6} {'pub=0':>6} {'Images':>7} {'avg_fm':>7} {'avg_sf':>7}")
    print("-" * 80)

    for batch_dir in sorted(cp_dir.iterdir()):
        if not batch_dir.is_dir():
            continue
        cp = batch_dir / "checkpoint.jsonl"
        if not cp.exists():
            continue

        items = pub3 = pub0 = images = errors = 0
        fm_scores = []
        sf_scores = []

        with open(cp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    items += 1
                    p = r.get("publish", 0)
                    images += p
                    if p == 3:
                        pub3 += 1
                    if p == 0:
                        pub0 += 1
                        needs_manual.append({
                            "fd_id": r.get("fd_id"),
                            "food_name": r.get("food_name"),
                            "status": r.get("status"),
                        })
                    if r.get("status") == "error":
                        errors += 1
                    if r.get("avg_fm"):
                        fm_scores.append(r["avg_fm"])
                    if r.get("avg_sf"):
                        sf_scores.append(r["avg_sf"])
                except Exception:
                    pass

        avg_fm = round(sum(fm_scores) / len(fm_scores), 2) if fm_scores else 0
        avg_sf = round(sum(sf_scores) / len(sf_scores), 2) if sf_scores else 0

        print(f"{batch_dir.name:<35} {items:>6} {pub3:>6} {pub0:>6} {images:>7} {avg_fm:>7.2f} {avg_sf:>7.2f}")

        total_items += items
        total_pub3 += pub3
        total_pub0 += pub0
        total_images += images
        total_errors += errors

    print("-" * 80)
    pub3_rate = total_pub3 / total_items * 100 if total_items else 0
    print(f"{'TOTAL':<35} {total_items:>6} {total_pub3:>6} {total_pub0:>6} {total_images:>7}")
    print(f"\npub=3 rate: {pub3_rate:.1f}%  |  errors: {total_errors}  |  NEEDS_MANUAL: {len(needs_manual)}")

    if needs_manual:
        print(f"\nNEEDS_MANUAL items ({len(needs_manual)}):")
        for item in needs_manual[:20]:
            print(f"  {item['fd_id']} — {item['food_name']} [{item['status']}]")
        if len(needs_manual) > 20:
            print(f"  ... and {len(needs_manual) - 20} more")


if __name__ == "__main__":
    main()

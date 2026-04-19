"""Show test results from checkpoint JSONL."""
import json
import sys
from pathlib import Path

cp_file = "/home/ubuntu/veritas_image_pipeline/output/checkpoints/test/checkpoint.jsonl"
image_dir = Path("/home/ubuntu/veritas_image_pipeline/output/images")

records = []
with open(cp_file) as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"\n{'='*70}")
print(f"  DUAL-SCORING PIPELINE TEST RESULTS  ({len(records)} items)")
print(f"{'='*70}")
print(f"{'FD ID':<12} {'Food Name':<30} {'Type':<16} {'Pub':>3} {'FM':>5} {'SF':>5} {'CLIP':>6} {'Time':>6}")
print(f"{'-'*70}")

total_pub = 0
total_fm = 0
total_sf = 0
total_clip = 0

for r in records:
    name = r.get("food_name", "")[:28]
    ftype = r.get("food_type", "")[:14]
    pub = r.get("publish", 0)
    fm = r.get("avg_fm", 0)
    sf = r.get("avg_sf", 0)
    clip = r.get("avg_clip", 0)
    elapsed = r.get("elapsed_s", 0)
    total_pub += pub
    total_fm += fm
    total_sf += sf
    total_clip += clip
    pub_str = f"{pub}/3"
    print(f"{r['fd_id']:<12} {name:<30} {ftype:<16} {pub_str:>3} {fm:>5.1f} {sf:>5.1f} {clip:>6.3f} {elapsed:>5.0f}s")

n = len(records)
print(f"{'-'*70}")
print(f"{'AVERAGE':<12} {'':<30} {'':<16} {'':>3} {total_fm/n:>5.1f} {total_sf/n:>5.1f} {total_clip/n:>6.3f}")
print(f"\n  pub=3 rate: {sum(1 for r in records if r.get('publish')==3)}/{n} ({100*sum(1 for r in records if r.get('publish')==3)/n:.0f}%)")
print(f"  Total images: {sum(r.get('publish',0) for r in records)}")

print(f"\n{'='*70}")
print("  IMAGES GENERATED")
print(f"{'='*70}")
for fd_dir in sorted(image_dir.iterdir()):
    if fd_dir.is_dir():
        imgs = sorted(fd_dir.glob("*.jpg"))
        print(f"\n  {fd_dir.name}/")
        for img in imgs:
            size_kb = img.stat().st_size // 1024
            print(f"    {img.name:<35} {size_kb:>5} KB")
print()

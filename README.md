# Veritas Image Pipeline

Automated food image collection pipeline for the Veritas food database.
Searches Google Images (via SerpAPI), scores candidates with OpenAI Vision (GPT-4.1-mini),
and outputs **3 curated 896×896px JPEGs per food item**:

| Slot | File | Description |
|------|------|-------------|
| Hero | `img_01_hero.jpg` | Single serving, clean background, overhead or 45° |
| Macro | `img_02_macro.jpg` | Close-up showing texture and colour detail |
| In-the-wild | `img_03_in_the_wild.jpg` | Food in hawker stall / restaurant / home context |

---

## Architecture

```
Input JSON
  → [Stage 1] Classify (cooked_dish / raw_ingredient / packaged_product / beverage)
  → [Stage 2] Build queries + slot descriptions
  → [Stage 3] SerpAPI Google Images search + download
  → [Stage 4] Vision scoring (GPT-4.1-mini, detail:low)
  → [Stage 5] Slot assignment + center-crop + resize to 896×896
  → output/images/{fd_id}/img_01_hero.jpg
                          img_02_macro.jpg
                          img_03_in_the_wild.jpg
```

**Orchestrator** splits the full record list into batches of 250, then launches
8 parallel workers per batch. Workers use a shared checkpoint JSONL for resume.

---

## Setup

### 1. Install dependencies

```bash
sudo pip3 install ddgs pillow openai requests
```

### 2. Set environment variables

```bash
export OPENAI_API_KEY="sk-..."
export SERP_API_KEY="your-serpapi-key"
export INPUT_JSON="/path/to/records.json"
export OUTPUT_DIR="/path/to/output/images"
export CHECKPOINT_DIR="/path/to/output/checkpoints"
```

### 3. Prepare input JSON

The input file must be a JSON array of food records. Minimum required fields:

```json
[
  {
    "fd_id": "FD000031",
    "food_name": "Soy Milk with Grass Jelly",
    "food_group": "Beverages",
    "cuisine": "Singaporean",
    "aliases": ["豆浆仙草冻", "tahu susu cincau"]
  }
]
```

---

## Running

### Full run (background)

```bash
cd pipeline/
bash run.sh FD000031 FD004315 8 250
```

### Custom range

```bash
python3 orchestrator.py \
  --input /path/to/records.json \
  --output /path/to/output/images \
  --start-fd FD000031 \
  --end-fd FD000100 \
  --workers 8 \
  --batch-size 250 \
  --checkpoint-dir /path/to/checkpoints
```

### Monitor progress

```bash
# Live orchestrator log
tail -f output/checkpoints/orchestrator.log

# Summary report
python3 status.py --checkpoint-dir output/checkpoints
```

---

## Directory Structure

```
veritas_image_pipeline/
├── README.md
├── pipeline/
│   ├── orchestrator.py      # Batch orchestrator
│   ├── worker.py            # Per-food pipeline worker
│   ├── run.sh               # Launch script
│   ├── status.py            # Progress report
│   └── modules/
│       ├── classifier.py    # Stage 1: food type + query generation
│       ├── image_search.py  # Stage 3: SerpAPI search + download
│       ├── vision_scorer.py # Stage 4: GPT-4.1-mini vision scoring
│       └── slot_assigner.py # Stage 5: crop + resize + save
├── input/
│   └── records.json         # (gitignored) Input food records
└── output/
    ├── images/              # (gitignored) Final output images
    └── checkpoints/         # Logs and checkpoint JSONL files
```

---

## Vision Scoring Pass Criteria

A candidate image passes if ALL of the following are true:

| Criterion | Requirement |
|-----------|-------------|
| REAL_PHOTO | YES (not illustration/graphic) |
| FOOD_MATCH | YES (correct food item) |
| CONFIDENCE | HIGH or MEDIUM |
| WATERMARK | NO |
| FM_SCORE | ≥ 6 / 10 |
| SF_SCORE | ≥ 5 / 10 |

---

## Retry Logic

Each slot gets up to **6 attempts** with progressively broader queries:
1. Primary slot query (e.g. "Char Kway Teow food photography")
2. Secondary slot query
3. Tertiary slot query
4. Retry with "food photography" suffix
5. Retry with "homemade recipe" suffix
6. Retry with "traditional dish" suffix

If no passing candidate is found after 6 attempts, the best-scoring candidate
(by FM_SCORE) is used as a fallback. Items with 0 images are flagged NEEDS_MANUAL.

---

## Cross-Reference

| Component | File | Purpose |
|-----------|------|---------|
| Food classification | `modules/classifier.py` | Determines food_type and generates slot-specific queries |
| Image search | `modules/image_search.py` | SerpAPI Google Images search + download with domain blocking |
| Vision scoring | `modules/vision_scorer.py` | GPT-4.1-mini pass/fail scoring per candidate |
| Slot assignment | `modules/slot_assigner.py` | Selects best candidate per slot, crops and resizes |
| Worker | `worker.py` | Runs the 5-stage pipeline for a range of FD records |
| Orchestrator | `orchestrator.py` | Manages batches and parallel workers |
| Status | `status.py` | Progress report across all batches |
